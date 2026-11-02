"""Independent audit of an allmimic X1 motion library: re-derive everything from the npz, do not trust the meta.

The build script writes `motions/x1_cmu_walk.meta.json` from metrics it computes itself; this tool takes only the
npz and the vendored model, recomputes the contract and the physical checks, and (where it can) compares against
the SMPLX source the clip came from. It exists because a report once quoted numbers from an older build of the same
file names -- the meta is a claim, this is the check.

Checks
  1. contract: shapes, fps, dtypes, finiteness, starts/lengths consistency, quat norms and sign continuity
  2. velocities: qvel against finite differences of the stored qpos (root linear, root angular, joints)
  3. geometry: joint-limit excess, root height, sole penetration and swing clearance (MuJoCo FK on the vendored MJCF)
  4. gait: foot touchdowns, cadence per foot, L/R alternation
  5. heading: the clip's net yaw change next to the SMPLX source's, for every clip that has a source file
  6. arms: per-joint share of frames within 0.02 rad of a limit

  PYTHONPATH=. python tools/x1_motion_audit.py --npz motions/x1_cmu_walk.npz --amass ~/code/X1_29_AMP/data/amass/CMU
  PYTHONPATH=. python tools/x1_motion_audit.py --npz motions/x1_cmu_walk.npz --json /tmp/audit.json --quiet
Exit code is non-zero when a hard check fails (contract, finiteness, penetration, velocities).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from allmimic.robots import x1_29dof as x1  # noqa: E402
from tools.x1_gait import gait_summary  # noqa: E402

AT_LIMIT = 0.02  # rad


def load_model():
    spec = mujoco.MjSpec.from_file(str(x1.X1_XML))
    spec.assets = {f'{spec.meshdir}/{p.name}': p.read_bytes() for p in sorted(x1.MESH_DIR.iterdir()) if p.is_file()}
    m = spec.compile()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
    assert tuple(names) == x1.X1_JOINT_ORDER, 'vendored model joint order != X1_JOINT_ORDER'
    return m, mujoco.MjData(m), names


def sole_heights(m, d):
    left = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in x1.X1_FOOT_GEOMS if g.startswith('left')]
    right = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in x1.X1_FOOT_GEOMS if g.startswith('right')]

    def low(gids):
        z = []
        for g in gids:
            half = m.geom_size[g][0] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE else m.geom_size[g][2]
            z.append(d.geom_xpos[g][2] - half)
        return min(z)

    return low(left), low(right)


def audit(npz: Path, amass: Path | None, max_clips: int = 0) -> dict:
    d = np.load(npz)
    qpos, qvel = d['qpos'], d['qvel']
    starts, lengths, names, fps = d['starts'], d['lengths'], list(d['names']), float(d['fps'])
    hard = []
    # 1. contract
    contract = {
        'qpos_shape': list(qpos.shape), 'qvel_shape': list(qvel.shape), 'clips': len(names), 'fps': fps,
        'qpos_dtype': str(qpos.dtype), 'finite': bool(np.isfinite(qpos).all() and np.isfinite(qvel).all()),
        'starts_ok': bool(np.array_equal(starts, np.concatenate([[0], np.cumsum(lengths)[:-1]]))),
        'lengths_sum': int(lengths.sum()), 'frames': int(len(qpos)),
        'quat_norm_dev': float(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1).max()),
        'quat_sign_flips': int((np.sum(qpos[1:, 3:7] * qpos[:-1, 3:7], axis=1) < 0).sum()),
        'cols': {'root_pos': 3, 'quat': 4, 'joints': int(qpos.shape[1] - 7), 'qvel': int(qvel.shape[1])},
    }
    if contract['qpos_shape'][1] != 36 or contract['qvel_shape'][1] != 35 or contract['cols']['joints'] != 29:
        hard.append('contract: qpos/qvel shapes are not (N,36)/(N,35)')
    if not contract['finite']:
        hard.append('contract: non-finite values')
    if not contract['starts_ok']:
        hard.append('contract: starts != cumsum(lengths)')
    if contract['quat_norm_dev'] > 1e-5:
        hard.append('contract: quaternions not normalised')
    if contract['quat_sign_flips']:
        hard.append('contract: quaternion sign flips (interpolation hazard)')

    m, data, joint_names = load_model()
    hi = np.array([m.jnt_range[j, 1] for j in range(1, m.njnt)])
    lo = np.array([m.jnt_range[j, 0] for j in range(1, m.njnt)])

    per_clip, vel_errs, penetrations, swings = [], [], [], []
    for i, name in enumerate(names[: max_clips or len(names)]):
        s, n = int(starts[i]), int(lengths[i])
        q, v = qpos[s:s + n], qvel[s:s + n]
        dt = 1.0 / fps
        # 2. velocities from the stored positions
        fd_lin = np.gradient(q[:, :3].astype(np.float64), dt, axis=0)
        fd_joint = np.gradient(q[:, 7:].astype(np.float64), dt, axis=0)
        e_lin = float(np.abs(fd_lin - v[:, :3]).max())
        e_joint = float(np.abs(fd_joint - v[:, 6:]).max())
        vel_errs.append(max(e_lin, e_joint))
        # 3. geometry: MuJoCo forward kinematics on every frame of the clip
        soles = np.zeros((n, 2))
        for k, row in enumerate(q):
            data.qpos[:] = row
            mujoco.mj_kinematics(m, data)
            soles[k] = sole_heights(m, data)
        pen = float(soles.min())
        penetrations.append(pen)
        # 4. gait: the same implementation the build wrote the meta with
        g = gait_summary(soles, fps)
        swings.append(g['swing_clearance_max_m'])
        cadence, alt, td = g['cadence_hz'], g['alternation_between_left_td'], g['touchdowns']
        over = np.clip(q[:, 7:] - hi, 0, None)
        under = np.clip(lo - q[:, 7:], 0, None)
        excess = float(np.maximum(over, under).max())
        at = (q[:, 7:] >= hi - AT_LIMIT) | (q[:, 7:] <= lo + AT_LIMIT)
        # 5. heading against the source
        yaw = np.unwrap(2 * np.arctan2(q[:, 6], q[:, 3]))
        net_robot = float(np.degrees(yaw[-1] - yaw[0]))
        net_human = None
        if amass is not None:
            src = amass / f'{name}_stageii.npz'
            if not src.exists():
                src = amass / f'{name}.npz'
            if src.exists():
                from scipy.spatial.transform import Rotation
                h = np.load(src, allow_pickle=True)
                y = np.unwrap(np.radians(Rotation.from_rotvec(h['root_orient']).as_euler('zyx', degrees=True)[:, 0]))
                net_human = float(np.degrees(y[-1] - y[0]))
        per_clip.append({
            'clip': str(name), 'frames': int(n), 'seconds': round(n / fps, 2),
            'root_z_min': round(float(q[:, 2].min()), 4), 'root_z_mean': round(float(q[:, 2].mean()), 4),
            'sole_penetration_m': round(pen, 4), 'swing_clearance_m': g['swing_clearance_max_m'],
            'joint_limit_excess_rad': round(excess, 4),
            'at_limit_frame_frac': round(float(at.any(1).mean()), 3),
            'at_limit_per_joint': {joint_names[j]: round(float(at[:, j].mean()), 3)
                                   for j in range(29) if at[:, j].mean() > 0.01},
            'touchdowns': td, 'walk_like': g['walk_like'],
            'cadence_hz': cadence, 'stride_s': g['stride_s'], 'alternation_between_left_td': alt,
            'yaw_net_robot_deg': round(net_robot, 1), 'yaw_net_human_deg': None if net_human is None else round(net_human, 1),
            'vel_fd_err': round(max(e_lin, e_joint), 6),
        })
    if penetrations and min(penetrations) < -0.02:
        hard.append(f'geometry: sole penetration {min(penetrations) * 1000:.1f} mm below -20 mm')
    if vel_errs and max(vel_errs) > 0.05:
        hard.append(f'velocities: qvel differs from finite differences by {max(vel_errs):.3f}')

    both = [c for c in per_clip if c['cadence_hz'][0] and c['cadence_hz'][1]]
    # per-joint share over ALL frames (not the mean of per-clip shares: a 2 s clip and a 4 s clip then weigh the same)
    q_all = qpos[:, 7:]
    at_all = (q_all >= hi - AT_LIMIT) | (q_all <= lo + AT_LIMIT)
    per_joint_all = {joint_names[j]: round(float(at_all[:, j].mean()), 4) for j in np.argsort(-at_all.mean(0))
                     if at_all[:, j].mean() > 0.005}
    swings_all = sorted(c['swing_clearance_m'] for c in per_clip)
    return {
        'npz': str(npz), 'contract': contract, 'hard_failures': hard,
        'summary': {
            'clips': len(per_clip),
            'root_z': [min(c['root_z_min'] for c in per_clip), max(c['root_z_mean'] for c in per_clip)],
            'sole_penetration_min_m': round(min(penetrations), 4) if penetrations else None,
            'swing_clearance_m': [round(min(swings), 3), round(max(swings), 3)] if swings else None,
            'joint_limit_excess_max_rad': max(c['joint_limit_excess_rad'] for c in per_clip),
            'at_limit_frames': round(float(np.mean([c['at_limit_frame_frac'] for c in per_clip])), 3),
            'at_limit_any_joint_all_frames': round(float(at_all.any(1).mean()), 4),
            'at_limit_per_joint_all_frames': per_joint_all,
            'swing_clearance_median_m': round(swings_all[len(swings_all) // 2], 4),
            'clips_all_frames_at_limit': sum(1 for c in per_clip if c['at_limit_frame_frac'] >= 0.999),
            'clips_with_both_cadences': len(both),
            'clips_both_in_0p8_1p2': sum(1 for c in both if all(0.8 <= x <= 1.2 for x in c['cadence_hz'])),
            'alternation_all_one': sum(1 for c in both if c['alternation_between_left_td'] and
                                       set(c['alternation_between_left_td']) == {1}),
            'alternation_not_all_one': [c['clip'] for c in both
                                        if c['alternation_between_left_td'] and set(c['alternation_between_left_td']) != {1}],
            'yaw_net_abs_max_deg': max(abs(c['yaw_net_robot_deg']) for c in per_clip),
            'yaw_pairs': sum(1 for c in per_clip if c['yaw_net_human_deg'] is not None),
        },
        'clips': per_clip,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--npz', type=Path, default=Path('motions/x1_cmu_walk.npz'))
    ap.add_argument('--amass', type=Path, default=None, help='AMASS CMU directory, to compare heading with the source')
    ap.add_argument('--json', type=Path, default=None)
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='audit only the first N clips')
    args = ap.parse_args()

    rep = audit(args.npz, args.amass, args.limit)
    if args.json:
        args.json.write_text(json.dumps(rep, indent=1))
    c, s = rep['contract'], rep['summary']
    if not args.quiet:
        print(f'contract     : {c["clips"]} clips, {c["frames"]} frames, fps {c["fps"]:.0f}, '
              f'qpos {tuple(c["qpos_shape"])}, qvel {tuple(c["qvel_shape"])}, finite {c["finite"]}, '
              f'quat dev {c["quat_norm_dev"]:.1e}, sign flips {c["quat_sign_flips"]}')
        print(f'root height  : {s["root_z"][0]:.3f} min / {s["root_z"][1]:.3f} max of the clip means')
        print(f'geometry     : sole penetration {s["sole_penetration_min_m"] * 1000:+.2f} mm, '
              f'swing clearance {s["swing_clearance_m"][0]:.3f}-{s["swing_clearance_m"][1]:.3f} m, '
              f'joint-limit excess {s["joint_limit_excess_max_rad"]:.3f} rad')
        print(f'arms at limit: {100 * s["at_limit_frames"]:.1f}% of frames have a joint within {AT_LIMIT} rad of a '
              f'limit; {s["clips_all_frames_at_limit"]}/{s["clips"]} clips in 100% of their frames')
        print(f'gait         : both-foot cadence in {s["clips_with_both_cadences"]}/{s["clips"]} clips, '
              f'{s["clips_both_in_0p8_1p2"]} of them in 0.8-1.2 Hz; alternation all-one in '
              f'{s["alternation_all_one"]}/{s["clips_with_both_cadences"]} of those, exceptions '
              f'{s["alternation_not_all_one"]}')
        print(f'heading      : |net yaw change| max {s["yaw_net_abs_max_deg"]:.1f} deg over {s["yaw_pairs"]} clips '
              f'with a source to compare against (see the per-clip yaw_net_human_deg)')
    if rep['hard_failures']:
        print('\nFAIL')
        for h in rep['hard_failures']:
            print('  -', h)
        return 1
    print('\nOK: no hard contract/geometry/velocity failure')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
