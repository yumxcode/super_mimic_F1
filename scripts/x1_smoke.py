"""CPU smoke test of the X1 29-DoF path: load the model, check the kinematics, roll out zero actions.

This is the local, GPU-free half of the X1 validation (the GPU half runs on Gradmotion: mjlab / MuJoCo Warp needs
CUDA). It exercises the same code the training env uses -- `FastTreeKinematics`, `Task`, `MotionLibrary` -- against
a plain MuJoCo CPU rollout of the vendored MJCF with the X1 position servos, and asserts there is no NaN.

Checks
  1. model: nq/nv/nu, joint order == allmimic.robots.x1_29dof.X1_JOINT_ORDER, actuators <-> joints 1:1
  2. forward kinematics: FastTreeKinematics == TreeKinematics == mujoco.mj_kinematics (bodies + velocities),
     for random q, including the joints whose hinge axis is -z (the X1 has both signs; the G1 does not)
  3. zero-action rollout: q_sp = q0 (policy action 0) for N control steps, obs / goal / reward / termination
     from `Task`, all finite, robot still standing (root height, foot heights, tilt)
  4. with --motions: the same, but reset from the first frame of a motion library clip and tracking its reference

  PYTHONPATH=. python scripts/x1_smoke.py --steps 200
  PYTHONPATH=. python scripts/x1_smoke.py --motions data/x1/cmw_walk.npz --clip 0
Needs mujoco + torch (no GPU, no mjlab).

The rollout reproduces what mjlab builds, and three details of that are easy to get wrong -- all three were
wrong in the first version of this script and produced "Nan, Inf or huge value in QACC" at t~0.7 s:
  * the asset's actuator order is NOT the joint order (ctrl[0] is left_hip_pitch, dof 17), so a torque written
    into qfrc_applied[6:] in actuator order lands on the wrong joint -- measured: the wrist servos received the
    leg torques and the model exploded within 50 ms. The torque is therefore scattered by jnt_dofadr.
  * the MJCF carries no armature; mjlab adds X1_JOINT_PARAMS[*]['armature'] per joint. Without it the wrist
    links (9 g, 2.4e-7 kg m^2) are driven unstable by their own servo.
  * the robot must start with its soles on the floor: X1_INIT_HEIGHT is derived from the model here.

Zero action means "hold the rest pose with the X1's own PD gains", and that pose is an inverted pendulum on the
ankles: m*g*h = 35 * 9.81 * 0.686 = 233 Nm/rad while the ankle servo stiffness is kp = 50 Nm/rad, so the
straight-leg stance is open loop unstable (measured time constant ~0.3 s, it topples after ~2 s). --ankle-gain
scales only the two ankle kp values, which is how the asset's numerical soundness is tested separately from the
controller: with --ankle-gain 8 (kp 400 > m*g*h) the same model stands for the whole run at zero action.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch

from allmimic.kinematics import FastTreeKinematics, TreeKinematics
from allmimic.robots import get_robot
from allmimic.robots import x1_29dof as x1
from allmimic.rotations import axis_angle_to_mat, mat_to_axis_angle, quat_to_mat
from allmimic.task import GOAL_FRAME_DIM, OBS_DIM, Task, TaskCfg

DT_SIM = 0.005  # mjlab SimulationCfg timestep
DECIMATION = 4  # TrackingSimEnv.decimation -> 50 Hz control
SOLVER_ITERS = (10, 20)  # mjlab MujocoCfg(iterations=10, ls_iterations=20)


def build(xml: Path, device: str, iters=SOLVER_ITERS):
    """robot + the floor mjlab's Scene adds (the vendored asset is robot-only), at mjlab's simulation timestep"""
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.assets = {f'{spec.meshdir}/{p.name}': p.read_bytes() for p in sorted(x1.MESH_DIR.iterdir()) if p.is_file()}
    floor = spec.worldbody.add_geom()
    floor.name = 'floor'
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    floor.condim = 3
    floor.friction = [1.0, 0.005, 0.0001]
    # the asset's top-level <default> disables collision on every geom in its own spec, so the floor the scene adds
    # has to set these explicitly (mjlab's Scene attaches the robot into the world spec and is not affected)
    floor.contype = 1
    floor.conaffinity = 1
    m = spec.compile()
    m.opt.timestep = DT_SIM
    m.opt.iterations, m.opt.ls_iterations = iters
    # the MJCF ships no armature; mjlab's BuiltinPositionActuatorCfg sets it per joint (x1_actuator_cfgs)
    for j in range(1, m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        m.dof_armature[int(m.jnt_dofadr[j])] = x1.X1_JOINT_PARAMS[nm]['armature']
    # the env's CollisionCfg sets this; the local rollout has to match or it measures a different contact model
    for g, name in enumerate(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)):
        if name in x1.X1_FOOT_GEOMS:
            m.geom_solref[g] = x1.X1_FOOT_SOLREF
    d = mujoco.MjData(m)
    robot = get_robot('x1_29dof')
    kin = FastTreeKinematics(m, device, root_body=1)
    kin_slow = TreeKinematics(m, device, root_body=1)
    names = [nm.split('/')[-1] for nm in kin.joint_names]
    x1.joint_names_match(names)
    return m, d, kin, kin_slow, names, robot


def check_model(m, names, robot) -> None:
    print(f'model          : nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} mass={m.body_mass.sum():.3f} kg')
    print(f'joint order    : {len(names)} hinges, matches allmimic.robots.x1_29dof.X1_JOINT_ORDER')
    for k, n in enumerate(names):
        print(f'  [{k:2d}] {n}')
    jid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(1, m.njnt)}
    act_joint = [m.actuator_trnid[a, 0] for a in range(m.nu)]
    assert sorted(act_joint) == sorted(jid.values()), 'actuators are not 1:1 with the 29 hinges'
    assert len(set(act_joint)) == 29
    order = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in act_joint]
    same = order == list(names)
    print(f'actuators      : {m.nu}, 1:1 with the hinges; ctrl order {"==" if same else "!="} joint order '
          f'(ctrl[0] -> {order[0]}, dof {int(m.jnt_dofadr[act_joint[0]]) - 6})')
    for g in robot.foot_geoms:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        assert gid >= 0, f'{g} missing from the model'
    print(f'foot geoms     : {len(robot.foot_geoms)} present ({", ".join(robot.foot_geoms)})')
    bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b).split('/')[-1] for b in range(m.nbody)]
    for b in robot.tracked_bodies + (robot.root_body, robot.torso_body):
        assert b in bodies, f'tracked/root/torso body {b} missing'
    print(f'tracked bodies : {len(robot.tracked_bodies)} present, root={robot.root_body}, torso={robot.torso_body}')


def check_kinematics(m, d, kin, kin_slow, device, n=32) -> None:
    torch.manual_seed(0)
    q = (torch.rand(n, 29) - 0.5) * 1.2
    q[:, 20] = q[:, 20].abs()  # knees in [0, 2.7]
    q[:, 26] = q[:, 26].abs()
    pos = torch.zeros(n, 3)
    pos[:, 2] = 0.7
    rot = torch.eye(3).repeat(n, 1, 1)
    v = torch.zeros(n, 3)
    w = torch.zeros(n, 3)
    P, R, V, W = kin.forward(pos, rot, q, v, w, torch.zeros(n, 29))
    P2, R2, V2, W2 = kin_slow.forward(pos, rot, q, v, w, torch.zeros(n, 29))
    # reference: mujoco FK
    mp, mr = [], []
    for i in range(n):
        d.qpos[:] = 0
        d.qpos[0:3] = pos[i].numpy()
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[7:] = q[i].numpy()
        mujoco.mj_kinematics(m, d)
        mp.append(d.xpos.copy())
        mr.append(d.xmat.reshape(-1, 3, 3).copy())
    mp, mr = torch.tensor(np.stack(mp))[:, 1:], torch.tensor(np.stack(mr))[:, 1:]  # drop the world body
    err_fast = (P - mp).abs().max().item()
    err_slow = (P2 - mp).abs().max().item()
    err_rot = (R - mr).abs().max().item()
    err_rot_slow = (R2 - mr).abs().max().item()
    err_fs = (P - P2).abs().max().item()
    print(
        f'kinematics     : max |FK - mujoco| pos fast={err_fast:.2e} slow={err_slow:.2e} rot={err_rot:.2e}/'
        f'{err_rot_slow:.2e}; fast vs slow={err_fs:.2e}'
    )
    assert err_fast < 1e-5 and err_slow < 1e-5 and err_rot < 1e-5 and err_rot_slow < 1e-5, 'FK mismatch'
    assert err_fs < 1e-5, 'FastTreeKinematics != TreeKinematics'
    # velocities: finite-difference the fast kinematics over one small step of (v, w)
    vv = torch.tensor([[0.3, -0.1, 0.2]])
    ww = torch.tensor([[0.1, 0.2, -0.3]])
    P0, R0, V0, W0 = kin.forward(pos[:1], rot[:1], q[:1], vv, ww, torch.zeros(1, 29))
    dt = 1e-3
    R1_in = rot[:1] @ axis_angle_to_mat(ww * dt)
    P1, R1, _, _ = kin.forward(pos[:1] + vv * dt, R1_in, q[:1], vv, ww, torch.zeros(1, 29))
    V_fd = (P1 - P0) / dt
    W_fd = mat_to_axis_angle(R1 @ R0.transpose(-1, -2)) / dt
    err_v = float((V_fd - V0).abs().max())
    err_w = float((W_fd - W0).abs().max())
    print(f'kinematics     : body velocities vs finite difference: v {err_v:.2e}, w {err_w:.2e}')
    assert err_v < 1e-3 and err_w < 1e-3, 'velocity mismatch'


def make_task(m, kin, robot, device, cfg: TaskCfg, soft_factor=0.9):
    """reproduce what TrackingSimEnv builds: q0 from the robot's home pose, mjlab's soft limits, action scale"""
    names = [nm.split('/')[-1] for nm in kin.joint_names]
    q0 = torch.tensor([x1.X1_HOME_JOINT_POS[n] for n in names])
    scale = torch.tensor([robot.action_scale[n] for n in names])
    lo = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
    mid, half = lo.mean(1), (lo[:, 1] - lo[:, 0]) / 2
    lim = torch.tensor(np.stack([mid - soft_factor * half, mid + soft_factor * half], 1), dtype=torch.float32)
    task = Task(kin, q0, lim, scale, cfg, device, tracked_bodies=robot.tracked_bodies)
    task.set_joint_names(names)
    return task


def servo_torque(m, q, qd, q_sp, qd_des, ankle_gain=1.0, dof_names=None) -> np.ndarray:
    """mjlab's <position> actuator: tau = kp (q_sp - q) - kd qd, clamped to +/- effort_limit.

    Returns the torque indexed by joint (dof - 6), ready for qfrc_applied[6:]: the asset's actuator order is
    not the joint order, so an actuator-order array written there drives the wrong joints.
    """
    tau = np.zeros(m.nv - 6)
    for a in range(m.nu):
        j = int(m.actuator_trnid[a, 0])
        nm = dof_names[j - 1] if dof_names is not None else mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        p = x1.X1_JOINT_PARAMS[nm]
        gain = ankle_gain if 'ankle' in nm else 1.0
        adr, dof = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        t = gain * p['kp'] * (q_sp[dof - 6] - q[adr]) - p['kd'] * (qd[dof] - qd_des[dof - 6])
        tau[dof - 6] = np.clip(t, -p['effort'], p['effort'])
    return tau


def rollout(m, d, kin, task, robot, device, steps, ref=None, label='zero action', ankle_gain=1.0):
    """N control steps at 50 Hz (decimation 4); returns the per-step diagnostics"""
    import mujoco as mj

    names = [nm.split('/')[-1] for nm in kin.joint_names]
    q_sp = task.q0.clone()
    prev_action = torch.zeros(1, 29, device=device)
    enc_bias = torch.zeros(1, 29, device=device)
    rows = []
    for k in range(steps):
        d.qfrc_applied[:] = 0
        for _ in range(DECIMATION):
            tau = servo_torque(m, d.qpos, d.qvel, q_sp.numpy(), np.zeros(29), ankle_gain, names)
            d.ctrl[:] = 0
            d.qfrc_applied[6:] = tau
            mj.mj_step(m, d)
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                raise SystemExit(f'non-finite state at control step {k}')
        qpos, qvel = d.qpos.copy(), d.qvel.copy()
        rot = quat_to_mat(torch.as_tensor(qpos[3:7], device=device, dtype=torch.float32)[None])
        s = {
            'pos': torch.as_tensor(qpos[:3], device=device, dtype=torch.float32)[None],
            'rot': rot,
            'q': torch.as_tensor(qpos[7:], device=device, dtype=torch.float32)[None],
            'v': torch.as_tensor(qvel[:3], device=device, dtype=torch.float32)[None],
            'w': (rot[0] @ torch.as_tensor(qvel[3:6], device=device, dtype=torch.float32))[None],
            'qd': torch.as_tensor(qvel[6:], device=device, dtype=torch.float32)[None],
        }
        rf = ref if ref is not None else s
        kin_ref = kin.forward(rf['pos'], rf['rot'], rf['q'], rf['v'], rf['w'], rf['qd'])
        rf_future = {k: v.unsqueeze(1).expand(-1, task.cfg.future, *v.shape[1:]) for k, v in rf.items()}
        out = task.analytics(s, rf, kin_ref, rf_future, prev_action, prev_action, enc_bias)
        for key in ('obs', 'feats', 'goal', 'r'):
            t = out[key]
            if not torch.isfinite(t).all():
                raise SystemExit(f'non-finite {key} at control step {k}')
        foot_z = [geom_lowest_z(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)) for g in robot.foot_geoms]
        rows.append(
            {
                'k': k,
                'base_z': float(qpos[2]),
                'tilt': float(1.0 - out['kin'][1][0, 0, 2, 2].item()),
                'foot_min': min(foot_z),
                'foot_max': max(foot_z),
                'reward': float(out['r']),
                'term': bool(out['hard']),
                'obs_max': float(out['obs'].abs().max()),
                'goal_max': float(out['goal'].abs().max()),
                'qd_max': float(np.abs(qvel[6:]).max()),
            }
        )
        if (k + 1) % max(1, steps // 8) == 0 or k == 0:
            print(f'    t={k * DECIMATION * DT_SIM:5.2f}s  base_z={qpos[2]:.4f}  tilt={rows[-1]["tilt"]:.3f}  '
                  f'sole={min(foot_z):+.4f}  |qd|={rows[-1]["qd_max"]:6.2f}  ncon={d.ncon}  r={rows[-1]["reward"]:+.3f}')
        q_sp = task.q0.clone()  # zero policy action -> q_sp = q0 every step
        if ref is not None and k + 1 < ref['pos'].shape[0] - 1:
            nxt = {kk: vv[1:] for kk, vv in ref.items()}
            ref = nxt
    r = rows
    print(f'\n{label}: {steps} control steps ({steps * DECIMATION * DT_SIM:.2f} s of sim), '
          f'solver iters {m.opt.iterations}/{m.opt.ls_iterations}, ankle kp x{ankle_gain}')
    print(f'  obs          : shape {tuple(out["obs"].shape)} (OBS_DIM={OBS_DIM}), max |obs| {max(x["obs_max"] for x in r):.3f}')
    print(f'  goal         : {task.cfg.future} future frames x {GOAL_FRAME_DIM} = {out["goal"].shape[-1]}, '
          f'max |goal| {max(x["goal_max"] for x in r):.3f}')
    print(f'  base height  : start {r[0]["base_z"]:.4f} m -> min {min(x["base_z"] for x in r):.4f} / '
          f'end {r[-1]["base_z"]:.4f} m')
    print(f'  foot height  : min {min(x["foot_min"] for x in r):+.4f} m, max {max(x["foot_max"] for x in r):+.4f} m')
    print(f'  tilt (1-R_zz): max {max(x["tilt"] for x in r):.4f}   terminated: {any(x["term"] for x in r)}')
    print(f'  |qd| max     : {max(x["qd_max"] for x in r):.2f} rad/s   reward per step: mean {np.mean([x["reward"] for x in r]):+.4f}')
    return r


def geom_lowest_z(m, d, gid: int) -> float:
    """world z of the lowest point of a geom (box: its 8 corners; sphere/capsule: centre - radius)"""
    if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
        rot = d.geom_xmat[gid].reshape(3, 3)
        s = m.geom_size[gid]
        return float(min((d.geom_xpos[gid] + rot @ (s * np.array([sx, sy, sz])))[2]
                         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)))
    return float(d.geom_xpos[gid][2] - m.geom_size[gid][0])


def stand_height(m, q) -> float:
    """root height at which the lowest foot sole point touches z=0 for joint pose q (measured on the model)"""
    d = mujoco.MjData(m)
    d.qpos[:] = 0
    d.qpos[2] = 0.7
    d.qpos[7:] = q
    mujoco.mj_forward(m, d)
    low = 1e9
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        if 'marker' in nm or 'visual' in nm or nm == 'floor' or not nm.startswith(('left_foot', 'right_foot')):
            continue
        low = min(low, geom_lowest_z(m, d, g))
    return 0.7 - low


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--xml', type=Path, default=x1.X1_XML)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--motions', default='', help='optional allmimic motion library to track (first frame = reset)')
    ap.add_argument('--clip', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--ankle-gain', type=float, default=1.0,
                    help='scale the two ankle kp values (1 = the X1 PD gains; >~5 makes the stance stable)')
    ap.add_argument('--iters', default=f'{SOLVER_ITERS[0]},{SOLVER_ITERS[1]}', help='solver iterations,ls_iterations')
    args = ap.parse_args()
    device = args.device
    iters = tuple(int(v) for v in args.iters.split(','))

    m, d, kin, kin_slow, names, robot = build(args.xml, device, iters)
    print('=' * 100)
    check_model(m, names, robot)
    print('=' * 100)
    check_kinematics(m, d, kin, kin_slow, device)
    print('=' * 100)

    cfg = TaskCfg()
    task = make_task(m, kin, robot, device, cfg)
    print(f'task           : q0 = {x1.X1_HOME_JOINT_POS["left_knee_pitch_joint"]:.2f} knee, action scale '
          f'{float(task.action_scale.min()):.3f}..{float(task.action_scale.max()):.3f}, '
          f'history {cfg.history}, future {cfg.future}, dt {cfg.dt}')

    # start from a standing pose (q0) with the soles on the floor and let the servo hold it
    z0 = stand_height(m, task.q0.numpy())
    print(f'init height    : {z0:.4f} m (measured: sole on z=0 at q0; X1_INIT_HEIGHT={x1.X1_INIT_HEIGHT})')
    assert abs(z0 - x1.X1_INIT_HEIGHT) < 2e-3, 'X1_INIT_HEIGHT does not match the model'
    d.qpos[:] = 0
    d.qpos[2] = z0
    d.qpos[7:] = task.q0.numpy()
    mujoco.mj_forward(m, d)
    rollout(m, d, kin, task, robot, device, args.steps, label='zero action (policy output 0, 50 Hz)',
            ankle_gain=args.ankle_gain)

    if args.motions:
        from allmimic.motion_lib import MotionLibrary

        lib = MotionLibrary(args.motions, device)
        start = int(lib.starts[args.clip])
        print(f'\nreference      : {args.motions} clip {args.clip} ({lib.names[args.clip]}), '
              f'{lib.lengths[args.clip]} frames, fps {lib.fps:.0f}')
        fr = lib.frame(torch.tensor([start], device=device))
        d.qpos[:] = 0
        d.qpos[:3] = fr['pos'][0].numpy()
        d.qpos[3:7] = fr['quat'][0].numpy()
        d.qpos[7:] = fr['q'][0].numpy()
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        idx = torch.arange(start, min(start + args.steps + 1, int(lib.starts[args.clip] + lib.lengths[args.clip])),
                           device=device)
        ref = lib.frame(idx)
        rollout(m, d, kin, task, robot, device, min(args.steps, len(idx) - 1), ref=ref, label='tracking the clip',
                ankle_gain=args.ankle_gain)
    print('\nno NaN, no compile error: OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
