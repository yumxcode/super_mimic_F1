"""Phase-2 arm-retargeting study: why the X1 elbows saturate and which fix works.

Measured facts this study builds on (see docs/X1_PHASE2_REPORT.md):
  * the X1's elbow joint sits essentially AT the shoulder (|shoulder_yaw_link ->
    elbow_pitch_link| = 0.048 m) and the forearm is 0.260 m, so the wrist distance
    from the shoulder is confined to [0.244, 0.308] m (elbow 2.0 -> 0 rad);
  * GMR's wrist targets (pelvis-relative scale x orientation-driven offset) land at
    0.20-0.38 m for walking, i.e. 53% of frames beyond reach (left, 07_01) and the
    solution ignores them (wrist FK-vs-target error p50 0.29-0.44 m), riding the
    elbow at +2.0 rad instead.

Variants (all rebuild motions/x1_ik_config.json weights in-memory):
  stock          phase-1 config, unchanged
  soft           wrist pos 10->1 / wrist rot ->0, elbow rot 10->3 / 5->1  (stop fighting)
  margin         stock weights + ConfigurationLimit(min_distance_from_limits=0.1)
  shrel          soft + wrist target rebuilt shoulder-relative: sh_t + clip(ratio*(wr-sh))
  shrel_margin   shrel + limit margin 0.1
  split          soft solve, then arm-only refine pass (frozen base/legs/torso,
                 posture prior, margin, shoulder-relative wrist target)
  iter30         stock + max_iter 30 + damping 0.1 (checks under-convergence)

Metrics per clip/variant: per-arm-joint at-limit fraction (0.02 rad), elbow p50,
wrist-direction error vs the human arm vector, wrist FK error vs the
shoulder-relative reference target, arm jerk, max arm |qdot|; plus whole-body
at-limit fraction (the phase-1 gate metric).

Runs on the GMR venv python:
  ~/code/X1_29_AMP/_gmr_local/venv/bin/python tools/x1_arm_study.py --out data/arm_study
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GMR_LOCAL = Path.home() / 'code' / 'X1_29_AMP' / '_gmr_local'
GMR_DIR = GMR_LOCAL / 'GMR'
sys.path.insert(0, str(GMR_DIR))

import mink  # noqa: E402
import mujoco  # noqa: E402
from general_motion_retargeting import GeneralMotionRetargeting as GMR  # noqa: E402
from general_motion_retargeting.params import IK_CONFIG_DICT  # noqa: E402
from general_motion_retargeting.utils.smpl import (  # noqa: E402
    get_smplx_data_offline_fast,
    load_smplx_file,
)
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

SMPLX_DIR = GMR_DIR / 'assets' / 'body_models'
BASE_IK = ROOT / 'motions' / 'x1_ik_config.json'

ARM_JOINTS = (
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_pitch_joint', 'left_elbow_yaw_joint', 'left_wrist_pitch_joint', 'left_wrist_roll_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_pitch_joint', 'right_elbow_yaw_joint', 'right_wrist_pitch_joint', 'right_wrist_roll_joint',
)
# X1 arm geometry (FK-measured): elbow sits at the shoulder, forearm 0.260 m,
# wrist distance from shoulder_yaw_link confined to [0.244 (elbow 2.0), 0.308 (elbow 0)]
D_MIN, D_MAX = 0.268, 0.300
RATIO = 0.62
AF = 0.95  # forearm length ratio robot|elbow->wrist 0.293 / human ~0.30
SAT_MARGIN = 0.02  # rad, same threshold as the phase-1 audit


def edit_ik(base: dict, table: str, body: str, pos: float | None, rot: float | None) -> dict:
    entry = base[table][body]
    if pos is not None:
        entry[1] = pos
    if rot is not None:
        entry[2] = rot
    return base


def config_for(variant: str) -> dict:
    cfg = json.loads(BASE_IK.read_text())
    if variant in ('soft', 'shrel', 'shrel_margin', 'shrelf', 'shrelf_margin', 'split'):
        for side in ('left', 'right'):
            # NB: both weights at literal 0 would unregister the body from GMR's
            # offset tables (KeyError in offset_human_data), so keep rot at 1e-9.
            edit_ik(cfg, 'ik_match_table1', f'{side}_wrist_roll_link', 0, 1e-9)
            pos2 = 5 if variant != 'soft' else 1
            edit_ik(cfg, 'ik_match_table2', f'{side}_wrist_roll_link', pos2, 0)
            edit_ik(cfg, 'ik_match_table1', f'{side}_elbow_pitch_link', 0, 3)
            edit_ik(cfg, 'ik_match_table2', f'{side}_elbow_pitch_link', 10, 1)
    return cfg


class StudyRetarget(GMR):
    """GMR with: in-memory ik config, optional limit margin, shoulder-relative wrist targets."""

    def __init__(self, cfg: dict, human_height: float, variant: str, margin: float = 0.0,
                 max_iter: int | None = None, damping: float | None = None):
        import tempfile

        self._cfg_tmp = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
        json.dump(cfg, self._cfg_tmp)
        self._cfg_tmp.close()
        IK_CONFIG_DICT['smplx']['x1'] = self._cfg_tmp.name
        super().__init__(src_human='smplx', tgt_robot='x1', actual_human_height=human_height, verbose=False)
        if margin > 0:
            self.ik_limits = [mink.ConfigurationLimit(self.model, min_distance_from_limits=margin)]
        if max_iter is not None:
            self.max_iter = max_iter
        if damping is not None:
            self.damping = damping
        self.variant = variant
        self.use_shrel = variant in ('shrel', 'shrel_margin', 'shrelf', 'shrelf_margin', 'split')
        self.shoulder_bid = {s: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                                  f'{s}_shoulder_yaw_link') for s in ('left', 'right')}

    def update_targets(self, human_data, offset_to_ground=False):
        super().update_targets(human_data, offset_to_ground)
        if not self.use_shrel:
            return
        for side in ('left', 'right'):
            wr_h = np.asarray(human_data[f'{side}_wrist'][0], dtype=float)
            sh_h = np.asarray(human_data[f'{side}_shoulder'][0], dtype=float)
            el_h = np.asarray(human_data[f'{side}_elbow'][0], dtype=float)
            # anchor on the ROBOT's live shoulder FK, not the scaled human shoulder target:
            # the pelvis-relative shoulder target floats 0.25-0.45 m above the X1's short torso
            # (measured: target z 1.24 vs FK 0.77 on 09_01), which is what pushed every arm
            # target out of the reachable cone in the first place.
            sh_anchor = None
            if getattr(self, 'configuration', None) is not None:
                mujoco.mj_kinematics(self.model, self.configuration.data)
                sh_anchor = self.configuration.data.xpos[self.shoulder_bid[side]].copy()
            if sh_anchor is None:
                sh_anchor = np.asarray(self.scaled_human_data[f'{side}_shoulder'][0], dtype=float)
            if self.variant.startswith('shrelf'):  # forearm direction target
                direction = wr_h - el_h
                d = np.clip(AF * np.linalg.norm(direction), D_MIN, D_MAX)
            else:  # shrel*: shoulder->wrist direction
                direction = wr_h - sh_h
                d = np.clip(RATIO * np.linalg.norm(direction), D_MIN, D_MAX)
            n = np.linalg.norm(direction)
            if n < 1e-6:
                return
            pos = sh_anchor + direction * (d / n)
            quat = self.scaled_human_data[f'{side}_wrist'][1]  # wxyz, same as GMR passes to mink
            target = mink.SE3.from_rotation_and_translation(mink.SO3(quat), pos)
            for table in (self.human_body_to_task1, self.human_body_to_task2):
                task = table.get(f'{side}_wrist')
                if task is not None:
                    task.set_target(target)


def arm_refine(r: StudyRetarget, qpos_seq: np.ndarray, frames: list, margin: float = 0.15,
               posture_cost_arm: float = 0.30, freeze_cost: float = 20.0,
               wrist_pos_cost: float = 8.0, vel_limit: float = 30.0,
               warmup_iter: int = 300, max_iter: int = 60, dt: float = 0.01,
               tol: float = 1e-4, smooth_taps: int = 5) -> np.ndarray:
    """Split pass: freeze base/legs/torso at GMR's solution, re-solve the arms only.

    Frame k starts from GMR's whole-body qpos with the ARM joints carried over from
    the refined frame k-1 (temporal warm start), so the refined arm track is smooth.
    """
    m = r.model
    dof_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.dof_jntid[i]) for i in range(m.nv)]
    arm_dofs = np.array([i for i, n in enumerate(dof_names) if n in ARM_JOINTS])
    arm_qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINTS])
    natural = np.zeros(m.nq)
    for adr, nm in zip(arm_qadr, ARM_JOINTS):
        if 'elbow_pitch' in nm:
            natural[adr] = 0.5
        elif 'shoulder_roll' in nm:
            natural[adr] = -0.2
        elif 'wrist_roll' in nm:
            natural[adr] = 0.0
    cost = np.full(m.nv, freeze_cost)
    cost[arm_dofs] = posture_cost_arm
    cost[:6] = freeze_cost  # free joint

    # the ONLY arm task is the robot-anchored wrist position target (set per frame by
    # r.update_targets); the shoulder/elbow pos targets from GMR's scaled human data
    # float ~0.3-0.45 m above the X1's short torso and are pure noise here.
    tasks_by = {}
    for side in ('left', 'right'):
        tasks_by[(side, 'wr')] = mink.FrameTask(f'{side}_wrist_roll_link', 'body', wrist_pos_cost, 0)
    posture = mink.PostureTask(m, cost)
    vel_limits = {n: vel_limit for n in ARM_JOINTS}
    limits = [mink.ConfigurationLimit(m, min_distance_from_limits=margin),
              mink.VelocityLimit(m, vel_limits)]
    all_tasks = list(tasks_by.values()) + [posture]

    def set_targets(frame):
        r.update_targets(frame)
        sd = r.scaled_human_data
        for side in ('left', 'right'):
            wr = sd[f'{side}_wrist'][0]
            q_wr = sd[f'{side}_wrist'][1]  # wxyz
            tasks_by[(side, 'wr')].set_target(mink.SE3.from_rotation_and_translation(mink.SO3(q_wr), wr))

    cfg = mink.Configuration(m)
    out = qpos_seq.copy()
    prev_arm = None
    # mink's ConfigurationLimit only prevents ENTERING the margin band; a warm start
    # already parked ON the stop (GMR's stock arms) is never extracted. Clamp the arm
    # warm start into the band so the limit can hold it there.
    band_lo = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)][0] + margin for n in ARM_JOINTS])
    band_hi = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)][1] - margin for n in ARM_JOINTS])
    for k, frame in enumerate(frames[: len(qpos_seq)]):
        q = out[k].copy()
        if prev_arm is not None:
            q[arm_qadr] = prev_arm
        q[arm_qadr] = np.clip(q[arm_qadr], band_lo, band_hi)
        cfg.update(q)
        posture_target = q.copy()
        posture_target[arm_qadr] = natural[arm_qadr]
        posture.set_target(posture_target)
        set_targets(frame)
        n_iter = warmup_iter if k == 0 else max_iter
        prev_err = np.inf
        for _ in range(n_iter):
            v = mink.solve_ik(cfg, all_tasks, dt, r.solver, r.damping, limits=limits)
            cfg.integrate_inplace(v, dt)
            err = float(np.linalg.norm(np.concatenate(
                [t.compute_error(cfg) for t in tasks_by.values()])))
            if k > 0 and prev_err - err < tol:
                break
            prev_err = err
        out[k] = cfg.data.qpos.copy()
        prev_arm = out[k][arm_qadr].copy()
    if smooth_taps > 1:
        # binomial smoothing of the re-solved arm track only (base/legs untouched);
        # the per-frame QP adds mid-frequency chatter vs GMR's incremental solve.
        k = np.array([1, 4, 6, 4, 1], dtype=float)
        k = k[: smooth_taps + 2 if smooth_taps < 5 else 5]
        k /= k.sum()
        pad = len(k) // 2
        arm_cols = np.zeros(out.shape[1], dtype=bool)
        arm_cols[arm_qadr] = True
        for c in np.flatnonzero(arm_cols):
            padded = np.pad(out[:, c], pad, mode='edge')
            out[:, c] = np.convolve(padded, k, mode='valid')
    return out


def load_clip(path: Path, n_max: int = 0, stride: int = 1):
    smplx_data, body_model, smplx_output, height = load_smplx_file(str(path), str(SMPLX_DIR))
    src_fps = float(smplx_data['mocap_frame_rate'].item())
    frames, fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=src_fps)
    frames = frames[::stride]
    if n_max:
        frames = frames[:n_max]
    return frames, src_fps / stride, height


class Metrics:
    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.jn = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)]
        self.lo = np.array([model.jnt_range[j, 0] for j in range(1, model.njnt)])
        self.hi = np.array([model.jnt_range[j, 1] for j in range(1, model.njnt)])
        self.arm_idx = np.array([self.jn.index(n) for n in ARM_JOINTS])
        self.elbow_idx = {s: self.jn.index(f'{s}_elbow_pitch_joint') for s in ('left', 'right')}
        self.bid = {f'{s}_{b}': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'{s}_{b}')
                    for s in ('left', 'right') for b in ('shoulder_yaw_link', 'elbow_pitch_link', 'wrist_roll_link')}

    def fk(self, qpos):
        self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)
        return {k: self.d.xpos[i].copy() for k, i in self.bid.items()}

    def __call__(self, qpos: np.ndarray, frames: list, fps: float) -> dict:
        q = qpos[:, 7:]
        arm = q[:, self.arm_idx]
        arm_lo, arm_hi = self.lo[self.arm_idx], self.hi[self.arm_idx]
        sat_any = ((arm >= arm_hi - SAT_MARGIN) | (arm <= arm_lo + SAT_MARGIN))
        all_sat = ((q >= self.hi - SAT_MARGIN) | (q <= self.lo + SAT_MARGIN)).any(axis=1)
        # direction error vs the human arm vector, per side
        dir_err = {s: [] for s in ('left', 'right')}
        for k in range(len(qpos)):
            p = self.fk(qpos[k])
            fr = frames[k]
            for s in ('left', 'right'):
                v_r = p[f'{s}_wrist_roll_link'] - p[f'{s}_shoulder_yaw_link']
                v_h = (np.asarray(fr[f'{s}_wrist'][0]) - np.asarray(fr[f'{s}_shoulder'][0]))
                c = np.dot(v_r, v_h) / (np.linalg.norm(v_r) * np.linalg.norm(v_h) + 1e-9)
                dir_err[s].append(np.degrees(np.arccos(np.clip(c, -1, 1))))
        d1 = np.gradient(arm, axis=0) * fps
        d3 = np.gradient(np.gradient(d1, axis=0), axis=0) * fps ** 2
        return {
            'frames': len(qpos),
            'sat_any_arm': round(float(sat_any.any(axis=1).mean()), 3),
            'sat_all_body': round(float(all_sat.mean()), 3),
            'sat_per_joint': {ARM_JOINTS[i]: round(float(sat_any[:, i].mean()), 3)
                              for i in range(len(ARM_JOINTS)) if sat_any[:, i].mean() > 0.02},
            'elbow_p50': {s: round(float(np.percentile(q[:, self.elbow_idx[s]], 50)), 3) for s in ('left', 'right')},
            'sat_elbow': {s: round(float(((q[:, self.elbow_idx[s]] >= self.hi[self.elbow_idx[s]] - SAT_MARGIN) |
                                          (q[:, self.elbow_idx[s]] <= self.lo[self.elbow_idx[s]] + SAT_MARGIN)).mean()), 3)
                          for s in ('left', 'right')},
            'dir_err_deg': {s: round(float(np.median(dir_err[s])), 1) for s in ('left', 'right')},
            'jerk_arm': round(float(np.abs(d3).mean()), 1),
            'qd_max_arm': round(float(np.abs(d1).max()), 2),
        }


def run(out_dir: Path, clips: list[tuple[str, Path, int]], variants: list[str], render_clip: str | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    # one retargeter per variant, reused across clips of the same height (GMR is stateful)
    for variant in variants:
        cfg = config_for(variant)
        margin = {'margin': 0.10, 'shrel_margin': 0.10, 'shrelf_margin': 0.10, 'split': 0.0}.get(variant, 0.0)
        kw = {}
        if variant == 'iter30':
            kw = {'max_iter': 30, 'damping': 0.1}
        elif variant == 'reg':
            kw = {'max_iter': 10}
        per_clip = {}
        state = {'r': None, 'h': None}
        for name, path, n_max in clips:
            frames, fps, height = load_clip(path, n_max)
            if state['r'] is None or abs(state['h'] - height) > 0.01:
                state['r'] = StudyRetarget(cfg, height, variant, margin=margin, **kw)
                state['h'] = height
            r = state['r']
            r.reset_regularity()  # don't carry acc/jerk history across clips
            if variant == 'reg':
                r.set_regularity(enabled=True, acc_weight=0.05, jerk_weight=0.01)
                r.reset_regularity()
            t0 = time.time()
            qpos = None
            for _ in range(2):  # phase-1 protocol: two passes, second warm-started
                qpos = np.array([r.retarget(f).copy() for f in frames], dtype=np.float64)
            if variant == 'split':
                qpos = arm_refine(r, qpos, frames)
            met = Metrics(r.model)(qpos, frames, fps)
            met['retarget_s'] = round(time.time() - t0, 1)
            per_clip[name] = met
            np.save(out_dir / f'{variant}__{name}.npy', qpos)
            print(f'[{variant}] {name}: sat_any_arm {met["sat_any_arm"]} sat_elbow {met["sat_elbow"]} '
                  f'elbow_p50 {met["elbow_p50"]} dir_err {met["dir_err_deg"]} jerk {met["jerk_arm"]} '
                  f'({met["retarget_s"]}s)', flush=True)
        results[variant] = per_clip
    (out_dir / 'arm_study.json').write_text(json.dumps(results, indent=1))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', type=Path, default=ROOT / 'data' / 'arm_study')
    ap.add_argument('--clips', type=str, default='07_01,82_13,09_01,09_02,137_01')
    ap.add_argument('--variants', type=str,
                    default='stock,soft,margin,shrel,shrel_margin,split,iter30,reg')
    ap.add_argument('--render-clip', type=str, default=None)
    args = ap.parse_args()
    amass = Path.home() / 'code' / 'X1_29_AMP' / 'data' / 'amass' / 'CMU'
    stage = ROOT / 'data' / 'amass' / 'stageii'
    clips = []
    for name in args.clips.split(','):
        p = stage / f'{name}_full_stageii.npz'
        if not p.exists():
            p = amass / f'{name}_stageii.npz'
        n = 300 if name.startswith('137') else 0  # dance clip is long; study on a window
        clips.append((name, p, n))
    run(args.out, clips, [v.strip() for v in args.variants.split(',')], args.render_clip)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
