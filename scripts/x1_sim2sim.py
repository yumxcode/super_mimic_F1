"""X1 sim2sim: replay a trained policy in CPU MuJoCo against motion-library clips.

Replicates the training env's control loop exactly (x1_smoke.py's model build = the
asset the env compiles): 50 Hz policy, decimation 4 x 5 ms, position servos via
act_to_joint, obs history buffer, task_cfg(ckpt['cfg']) so the action encoding
(abs/res, goal frame, future offsets) matches the run.

Per clip: reset from frame 0 (as the benchmark protocol), roll to clip end or
termination; record root position / heading error and key-link position errors
(root, torso, wrists, ankles, knees) vs the reference FK; a clip FAILS if the
training termination triggers or the state goes non-finite.

  PYTHONPATH=. MJLAB_SRC=_scratch/mjlab_src python scripts/x1_sim2sim.py \
      --ckpt runs/x1_phase2/output/last.pt --motions motions/x1_phase2_train.npz \
      --clips 0-9 --render 1 --report runs/sim2sim.json

Optional --push-probe: while tracking, kick the root with a horizontal velocity
impulse (push_test.py's probe, CPU form) and report settle time / falls.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _cpu_mjlab_stub():
    """sim_env imports the full mjlab (mujoco_warp/CUDA); on CPU stub the modules it
    names at import time, the way scripts/x1_smoke.py's mjlab_light() does."""
    import types

    class _RecomputeLevel:
        NEVER = 'never'
        ON_RESET = 'on_reset'

    for name in ('mjlab', 'mjlab.managers', 'mjlab.managers.event_manager', 'mjlab.scene',
                 'mjlab.sim', 'mjlab.sim.sim', 'mjlab.terrains'):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod
    sys.modules['mjlab.managers.event_manager'].RecomputeLevel = _RecomputeLevel
    for attr in ('Scene', 'SceneCfg'):
        setattr(sys.modules['mjlab.scene'], attr, object)
    for attr in ('MujocoCfg', 'SimulationCfg'):
        setattr(sys.modules['mjlab.sim'], attr, object)
    sys.modules['mjlab.sim.sim'].Simulation = object
    sys.modules['mjlab.terrains'].TerrainEntityCfg = object


_cpu_mjlab_stub()

from allmimic.motion_lib import MotionLibrary  # noqa: E402
from allmimic.ppo import load_ppo_policy  # noqa: E402
from allmimic.rotations import quat_to_mat  # noqa: E402
from allmimic.robots import get_robot  # noqa: E402
from allmimic.robots import x1_29dof as x1  # noqa: E402
from allmimic.task import OBS_DIM, TaskCfg, task_cfg  # noqa: E402
from scripts.x1_smoke import DECIMATION, DT_SIM, build, make_task  # noqa: E402

LINKS = ('base_link', 'lumbar_pitch_link', 'left_wrist_roll_link', 'right_wrist_roll_link',
         'left_ankle_roll_link', 'right_ankle_roll_link', 'left_knee_pitch_link', 'right_knee_pitch_link')


def link_ids(m):
    return {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in LINKS}


def heading_yaw(rot: torch.Tensor) -> float:
    """yaw (rad) of a (3,3) rotation"""
    return float(torch.atan2(rot[1, 0], rot[0, 0]))


@torch.no_grad()
def rollout_clip(m, d, kin, task, policy, robot, act_to_joint, lib, clip: int, device,
                 max_steps: int = 0, push_at: int = -1, push_vec=None):
    """One clip from its first frame. Returns dict with per-step errors and failure flag."""
    names = [nm.split('/')[-1] for nm in kin.joint_names]
    lids = link_ids(m)
    start, length = int(lib.starts[clip]), int(lib.lengths[clip])
    n = max_steps or (length - 1)
    idx = torch.arange(start, start + n + 1, device=device)

    def state_of(d):
        qpos, qvel = d.qpos.copy(), d.qvel.copy()
        rot = quat_to_mat(torch.as_tensor(qpos[3:7], device=device, dtype=torch.float32)[None])
        return {
            'pos': torch.as_tensor(qpos[:3], device=device, dtype=torch.float32)[None],
            'rot': rot,
            'q': torch.as_tensor(qpos[7:], device=device, dtype=torch.float32)[None],
            'v': torch.as_tensor(qvel[:3], device=device, dtype=torch.float32)[None],
            'w': (rot[0] @ torch.as_tensor(qvel[3:6], device=device, dtype=torch.float32))[None],
            'qd': torch.as_tensor(qvel[6:], device=device, dtype=torch.float32)[None],
        }

    fr0 = lib.frame(idx[:1])
    d.qpos[:] = 0
    d.qpos[:3] = fr0['pos'][0].numpy()
    d.qpos[3:7] = fr0['quat'][0].numpy()
    d.qpos[7:] = fr0['q'][0].numpy()
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)

    hist = task.cfg.history
    future_off = torch.tensor([float(o) for o in (task.cfg.future_offsets or tuple(range(1, task.cfg.future + 1)))],
                              device=device, dtype=torch.long)

    def ref_future(fr_idx):
        idx = fr_idx[:, None] + future_off[None]
        idx = lib.clamp_to_clip(idx, fr_idx)
        return lib.frame(idx)

    obs_hist = torch.zeros(1, hist, OBS_DIM, device=device)
    enc_bias = torch.zeros(1, len(names), device=device)
    prev_action = torch.zeros(1, len(names), device=device)
    s = state_of(d)
    ref = {k: v for k, v in fr0.items()}
    kin_ref = kin.forward(ref['pos'], ref['rot'], ref['q'], ref['v'], ref['w'], ref['qd'])
    out = task.analytics(s, ref, kin_ref, ref_future(idx[:1]), prev_action, prev_action, enc_bias)
    obs_hist[:] = out['obs']
    qpos_track = [d.qpos.copy()]
    errs = []
    failed, term_at = False, -1
    push_speeds, settled_t = [], None
    for k in range(n):
        a = policy(obs_hist.flatten(1), out['goal'])
        a_sp = task.setpoint_action(a, lib.frame(idx[k + 1:k + 2])['q'])
        q_sp = task.action_scale * a_sp + task.q0
        d.ctrl[:] = (q_sp - enc_bias).reshape(-1).cpu().numpy()[act_to_joint]
        if k == push_at and push_vec is not None:
            d.qvel[:3] += push_vec
        for _ in range(DECIMATION):
            mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            failed, term_at = True, k
            break
        qpos_track.append(d.qpos.copy())
        s = state_of(d)
        ref = lib.frame(idx[k + 1:k + 2])
        kin_ref = kin.forward(ref['pos'], ref['rot'], ref['q'], ref['v'], ref['w'], ref['qd'])
        out = task.analytics(s, ref, kin_ref, ref_future(idx[k + 1:k + 2]), a, prev_action, enc_bias)
        obs_hist = torch.cat([obs_hist[:, 1:], out['obs'].unsqueeze(1)], dim=1)
        prev_action = a
        # errors: state FK vs reference FK
        mujoco.mj_kinematics(m, d)
        P_ref = kin_ref[0][0].cpu().numpy()  # (nbody-1, 3): kin drops the world body
        row = {'k': k}
        for nm, bid in lids.items():
            row[nm] = float(np.linalg.norm(d.xpos[bid] - P_ref[bid - 1]))
        yaw_r = heading_yaw(s['rot'][0])
        yaw_ref = heading_yaw(ref['rot'][0])
        row['yaw_err_deg'] = float(np.degrees(np.abs(np.angle(np.exp(1j * (yaw_r - yaw_ref))))))
        row['pos_err'] = row.pop('base_link')
        row['reward'] = float(out['r'])
        errs.append(row)
        if bool(out['hard']):
            failed, term_at = True, k
            break
        if push_at >= 0 and k >= push_at:
            push_speeds.append(float(np.linalg.norm(d.qvel[:2])))
    res = {'steps': len(errs), 'failed': failed, 'term_at': term_at, 'rows': errs,
           'qpos': np.array(qpos_track)}
    if push_at >= 0:
        # settle: 0.5 s (25 steps) continuously below 0.5 m/s after the push, or fall (root < 0.3 m)
        below = [sp < 0.5 for sp in push_speeds]
        settle = next((i + 25 for i in range(len(below) - 25) if all(below[i:i + 25])), None)
        res['push'] = {'settle_steps_after_push': settle,
                       'fell': bool(min(q[2] for q in res['qpos'][push_at:]) < 0.30),
                       'speed_peak': round(max(push_speeds), 2) if push_speeds else None}
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--motions', type=Path, required=True)
    ap.add_argument('--clips', default='all', help="'all', '0-9', '0,3,7'")
    ap.add_argument('--max-steps', type=int, default=0)
    ap.add_argument('--render', type=int, default=1, help='render the first N tracked clips')
    ap.add_argument('--render-dir', type=Path, default=ROOT / 'runs' / 'sim2sim_renders')
    ap.add_argument('--report', type=Path, default=ROOT / 'runs' / 'sim2sim.json')
    ap.add_argument('--push-probe', action='store_true', help='add a 1.5 m/s root kick mid-clip on clip 0')
    args = ap.parse_args()
    device = 'cpu'

    from scripts.x1_smoke import mjlab_light

    mjlab_light()  # build() needs mjlab.utils.spec (the light stub), ours above covers sim_env's
    m, d, kin, _, names, robot, act_to_joint = build(x1.X1_XML, device)
    # match the training env exactly: mjlab SimulationCfg defaults beyond MujocoCfg's
    # timestep/iterations -- implicitfast integrator (MuJoCo default is Euler), newton/pyramidal/1.0
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    m.vis.global_.offwidth, m.vis.global_.offheight = 960, 640  # offscreen render buffer
    ck = torch.load(args.ckpt, map_location=device)
    if 'actor' in ck:
        policy, meta = load_ppo_policy(args.ckpt, device)
    elif 'policy' in ck:  # SuperMimic format (wm + policy, e.g. every phase-1/2 run)
        from allmimic.evaluate import load_supermimic_policy

        pol, _ = load_supermimic_policy(args.ckpt, device)
        policy = lambda o, g: pol(o, g)
        meta = {'it': ck['it'], 'hours': ck['train_time'] / 3600, 'env_steps': ck['env_steps']}
    else:
        raise SystemExit(f'checkpoint keys {list(ck)[:8]} — neither actor nor policy')
    cfg = task_cfg(ck.get('cfg', {}) or {}, train=False)
    task = make_task(m, kin, robot, device, cfg)
    print(f'ckpt: it={meta["it"]} hours={meta["hours"]:.2f} action={cfg.action} '
          f'res_legs={cfg.res_legs} future={cfg.future}')

    lib = MotionLibrary(str(args.motions), device)
    if args.clips == 'all':
        clip_ids = list(range(lib.num_clips))
    elif '-' in args.clips:
        a, b = args.clips.split('-')
        clip_ids = list(range(int(a), int(b) + 1))
    else:
        clip_ids = [int(v) for v in args.clips.split(',')]
    print(f'{args.motions}: {lib.num_clips} clips, evaluating {len(clip_ids)}')

    results = {}
    t0 = time.time()
    for i, c in enumerate(clip_ids):
        push = None
        if args.push_probe and i == 0:
            push_at = min(100, int(lib.lengths[c]) // 2)
            push = (push_at, np.array([1.5, 0.0, 0.0]))
        r = rollout_clip(m, d, kin, task, policy, robot, act_to_joint, lib, c, device,
                         args.max_steps, *push) if push else rollout_clip(
            m, d, kin, task, policy, robot, act_to_joint, lib, c, device, args.max_steps)
        name = str(lib.names[c])
        agg = {
            'clip': name, 'steps': r['steps'], 'failed': r['failed'], 'term_at': r['term_at'],
            'pos_err_median': round(float(np.median([e['pos_err'] for e in r['rows']])), 4),
            'pos_err_p90': round(float(np.percentile([e['pos_err'] for e in r['rows']], 90)), 4),
            'yaw_err_p90_deg': round(float(np.percentile([e['yaw_err_deg'] for e in r['rows']], 90)), 1),
            'torso_err_median': round(float(np.median([e['lumbar_pitch_link'] for e in r['rows']])), 4),
            'wrist_err_median': round(float(np.median([max(e['left_wrist_roll_link'], e['right_wrist_roll_link'])
                                                       for e in r['rows']])), 4),
            'ankle_err_median': round(float(np.median([max(e['left_ankle_roll_link'], e['right_ankle_roll_link'])
                                                       for e in r['rows']])), 4),
            'mean_reward': round(float(np.mean([e['reward'] for e in r['rows']])), 4),
        }
        if 'push' in r:
            agg['push'] = r['push']
        results[name] = agg
        print(f'[{i + 1}/{len(clip_ids)}] {name:24s} {"FAIL" if r["failed"] else "ok":4s} '
              f'{r["steps"]:4d}st pos50 {agg["pos_err_median"]:.3f} pos90 {agg["pos_err_p90"]:.3f} '
              f'yaw90 {agg["yaw_err_p90_deg"]:5.1f} wrist50 {agg["wrist_err_median"]:.3f} '
              f'{"push:" + str(r["push"]) if "push" in r else ""}', flush=True)
        if i < args.render:
            from tools.x1_render_arms import render_variant

            args.render_dir.mkdir(parents=True, exist_ok=True)
            p, nframes = render_variant(m, r['qpos'], args.render_dir, f'sim2sim_{name}', 50.0)
            print(f'    rendered {nframes} frames -> {p}')

    failed = sum(1 for v in results.values() if v['failed'])
    summary = {
        'ckpt': str(args.ckpt), 'motions': str(args.motions), 'meta': meta,
        'num_clips': len(results), 'failed': failed,
        'pos_err_median_of_medians': round(float(np.median([v['pos_err_median'] for v in results.values()])), 4),
        'yaw_err_p90_mean': round(float(np.mean([v['yaw_err_p90_deg'] for v in results.values()])), 1),
        'clips': results, 'elapsed_s': round(time.time() - t0, 1),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=1, default=float))
    print(f'\nsim2sim: {failed}/{len(results)} failed; report {args.report}; {summary["elapsed_s"]}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
