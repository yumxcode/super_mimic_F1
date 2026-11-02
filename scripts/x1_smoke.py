"""CPU smoke test of the X1 29-DoF path: load the model, check the kinematics, roll out zero actions.

This is the local, GPU-free half of the X1 validation (the GPU half runs on Gradmotion: mjlab / MuJoCo Warp needs
CUDA). It exercises the same code the training env uses -- `FastTreeKinematics`, `Task`, `MotionLibrary` -- against
a plain MuJoCo CPU rollout of the vendored MJCF with the X1 position servos, and asserts there is no NaN.

The model under test is built the way TrackingSimEnv builds it: the asset with its 29 <motor>s deleted, then
mjlab's own `CollisionCfg.edit_spec` and `create_position_actuator` applied to it (the same objects
`x1_collision_cfgs()` / `x1_actuator_groups()` hand to the entity), so what is asserted here is what the simulator
gets -- not a copy of it. Only mjlab's *own* dependency chain (mjlab/__init__ needs tyro, mjlab.actuator needs
mujoco_warp, neither is needed on CPU) is stubbed; see mjlab_light().

Checks
  1. model: nq/nv/nu, joint order == allmimic.robots.x1_29dof.X1_JOINT_ORDER, actuators <-> joints 1:1, and the
     compiled contact/actuator parameters: both sole geoms condim 3 + priority 1 + solref 0.02 + friction 0.6,
     every other collision geom condim 1, and each actuator's gainprm/biasprm/forcerange = the X1 table
  2. forward kinematics: FastTreeKinematics == TreeKinematics == mujoco.mj_kinematics (bodies + velocities),
     for random q, including the joints whose hinge axis is -z (the X1 has both signs; the G1 does not)
  3. zero-action rollout: ctrl = q_sp = q0 (policy action 0) for N control steps at 50 Hz, obs / goal / reward /
     termination from `Task`; then the numeric verdict (see --expect)
  4. with --motions: the same, but reset from the first frame of a motion library clip and tracking its reference

  PYTHONPATH=. python scripts/x1_smoke.py --steps 200                  # the X1's own PD gains
  PYTHONPATH=. python scripts/x1_smoke.py --steps 200 --ankle-gain 8    # 8x ankle stiffness, robustness probe
  PYTHONPATH=. python scripts/x1_smoke.py --motions motions/x1_cmu_walk.npz --clip 0
Needs mujoco + torch (no GPU). Exits non-zero when a check fails.

Three details produced "Nan, Inf or huge value in QACC at t~0.7 s" in the first version of this script. None of
them was a defect of the asset, and each is checked here:
  * the asset's actuator order is NOT the joint order (the asset's first <motor> is left_hip_pitch, dof 17), and a
    torque written into qfrc_applied[6:] in actuator order lands on the wrong joint (the wrist servos received the
    leg torques). All torque now goes through `d.ctrl[act_to_joint]`, the way sim_env does it.
  * the MJCF carries no armature; mjlab's create_position_actuator writes X1_JOINT_PARAMS[*]['armature'] onto each
    joint, i.e. at spec level, before compile. Applying the armature to the compiled model instead (`m.dof_armature`)
    leaves `dof_M0`, the inertia the constraint solver regularizes with, stale: measured on this robot, the knee's
    M0 entry stays at 0.106 instead of 0.379 and the same model that stands for 4 s here topples at ~2 s.
  * the robot must start with its soles on the floor: the standing height is measured on the model here
    (0.6275 m; a flat 0.66 leaves the soles 3.2 cm in the air and turns the start into an impact).

Zero action means "hold the rest pose with the X1's own PD gains", and at those gains the stance is stable: the
four sole spheres per foot span 0.14 x 0.06 m, wide enough that the centre of pressure can move under the centre of
mass, so the ankle servo (kp 50 Nm/rad) does not have to hold m*g*h = 233 Nm/rad on its own. Measured here: 4 s,
root 0.6275 -> 0.6256 m, tilt 0.0036, |qd| 0.16 rad/s, sole penetration -0.4 mm, no QACC/NaN. (An earlier version of
this docstring argued the opposite -- m*g*h > kp, so the stance must be open loop unstable and only --ankle-gain 8
could stand. That was measured on a model built the wrong way: armature applied after compile, see above. With the
build fixed the argument does not hold, and the real gains are what --expect stand grades.)

--ankle-gain scales only the two ankle kp values. It is kept as a robustness probe: 8x also survives the broken
build, which is how the inertia effect was first noticed.
"""

from __future__ import annotations

import argparse
import enum
import importlib
import os
import sys
import types
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


class _TransmissionType(enum.Enum):
    """stand-in for mjlab.actuator.actuator's TransmissionType (that module imports mujoco_warp)"""

    JOINT = 'joint'
    TENDON = 'tendon'
    SITE = 'site'


def mjlab_light():
    """mjlab's CollisionCfg and create_position_actuator, importable without mjlab/__init__ (tyro) or mujoco_warp.

    Only the dependency chain is stubbed: the two functions this script calls are mjlab's own code, so the contact
    parameters and servos that get asserted are the ones the training env would build.
    """
    root = Path(os.environ.get('MJLAB_SRC', '')) if os.environ.get('MJLAB_SRC') else None
    if root is None or not (root / 'mjlab').is_dir():
        raise SystemExit('set MJLAB_SRC to an unpacked mjlab install (the wheel in _scratch works: '
                         'MJLAB_SRC=_scratch/mjlab_src)')
    for name, rel in (('mjlab', ''), ('mjlab.utils', 'utils'), ('mjlab.actuator', 'actuator')):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(root / 'mjlab' / rel)]
            sys.modules[name] = mod
    act = types.ModuleType('mjlab.actuator.actuator')
    act.TransmissionType = _TransmissionType
    sys.modules['mjlab.actuator.actuator'] = act
    spec_mod = importlib.import_module('mjlab.utils.spec')
    cfg_mod = importlib.import_module('mjlab.utils.spec_config')
    return cfg_mod.CollisionCfg, spec_mod.create_position_actuator


def build(xml: Path, device: str, iters=SOLVER_ITERS, ankle_gain: float = 1.0, entity_cfg: bool = True):
    """the X1 exactly as TrackingSimEnv gets it: asset minus its motors, + mjlab's collision cfg + position servos"""
    _, create_position_actuator = mjlab_light()
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.assets = {f'{spec.meshdir}/{p.name}': p.read_bytes() for p in sorted(x1.MESH_DIR.iterdir()) if p.is_file()}
    for actuator in list(spec.actuators):  # the asset ships 29 <motor>s; mjlab replaces them with position servos
        spec.delete(actuator)
    if entity_cfg:  # mjlab EntityCfg.collisions: condim / priority / friction / solref on the sole geoms
        for cfg in x1.x1_collision_cfgs():
            cfg.edit_spec(spec)
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
    for kp, kd, effort, armature, frictionloss, names in x1.x1_actuator_groups():
        for nm in names:
            gain = ankle_gain if 'ankle' in nm else 1.0
            create_position_actuator(
                spec, nm, stiffness=kp * gain, damping=kd, effort_limit=effort, armature=armature,
                frictionloss=frictionloss, transmission_type=_TransmissionType.JOINT,
            )
    m = spec.compile()
    m.opt.timestep = DT_SIM
    m.opt.iterations, m.opt.ls_iterations = iters
    d = mujoco.MjData(m)
    robot = get_robot('x1_29dof')
    kin = FastTreeKinematics(m, device, root_body=1)
    kin_slow = TreeKinematics(m, device, root_body=1)
    names = [nm.split('/')[-1] for nm in kin.joint_names]
    x1.joint_names_match(names)
    # actuator -> joint index (the env's act_to_joint)
    jid_to_k = {j: k for k, j in enumerate(range(1, 1 + len(names)))}
    act_to_joint = np.array([jid_to_k[int(m.actuator_trnid[a, 0])] for a in range(m.nu)])
    return m, d, kin, kin_slow, names, robot, act_to_joint


def check_model(m, names, robot, ankle_gain: float = 1.0) -> None:
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
    # Contact parameters, read off the compiled model: this is the layer that decides what the simulator uses, and a
    # CollisionCfg regex that matches nothing fails silently (condim 1 = no friction, the asset's 5 ms solref = a
    # contact that is unstable at 200 Hz), so it is asserted, not assumed.
    for g in robot.foot_geoms:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        assert gid >= 0, f'{g} missing from the model'
        got = dict(condim=int(m.geom_condim[gid]), priority=int(m.geom_priority[gid]),
                   solref=tuple(np.round(m.geom_solref[gid], 6)), friction=float(m.geom_friction[gid][0]),
                   contype=int(m.geom_contype[gid]), conaffinity=int(m.geom_conaffinity[gid]))
        want = dict(condim=x1.X1_FOOT_CONDIM, priority=x1.X1_FOOT_PRIORITY,
                    solref=tuple(np.round(x1.X1_FOOT_SOLREF, 6)), friction=x1.X1_FOOT_FRICTION[0],
                    contype=1, conaffinity=1)
        assert got == want, f'{g}: compiled contact {got} != CollisionCfg {want}'
    print(f'foot geoms     : {len(robot.foot_geoms)} present ({", ".join(robot.foot_geoms)}), '
          f'condim {x1.X1_FOOT_CONDIM} priority {x1.X1_FOOT_PRIORITY} solref {x1.X1_FOOT_SOLREF} '
          f'friction {x1.X1_FOOT_FRICTION[0]} (asserted on the compiled model)')
    other = [n for n in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom))
             if n and n.endswith('_collision') and n not in robot.foot_geoms]
    for n in other:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        assert m.geom_condim[gid] == 1, f'{n}: non-foot collision geom should be condim 1, got {m.geom_condim[gid]}'
    print(f'other contacts : {other or "none"} at condim 1 (as CollisionCfg asks)')

    # Servos, read off the compiled model: gain = kp, bias = (-kp, -kd) per joint, force +- effort.
    for a in range(m.nu):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a, 0]))
        p = x1.X1_JOINT_PARAMS[nm]
        gain = ankle_gain if 'ankle' in nm else 1.0
        want_gain, want_bias = p['kp'] * gain, (-p['kp'] * gain, -p['kd'])
        assert abs(m.actuator_gainprm[a, 0] - want_gain) < 1e-6, f'{nm}: gain {m.actuator_gainprm[a, 0]} != {want_gain}'
        assert np.allclose([m.actuator_biasprm[a, 1], m.actuator_biasprm[a, 2]], want_bias), f'{nm}: bias mismatch'
        assert m.actuator_forcelimited[a] and np.allclose(m.actuator_forcerange[a], [-p['effort'], p['effort']]), \
            f'{nm}: force range != +-{p["effort"]}'
        jid = int(m.actuator_trnid[a, 0])
        assert abs(m.dof_armature[int(m.jnt_dofadr[jid])] - p['armature']) < 1e-9, f'{nm}: armature mismatch'
        assert abs(m.dof_frictionloss[int(m.jnt_dofadr[jid])] - p['frictionloss']) < 1e-9, f'{nm}: frictionloss mismatch'
    print(f'actuators      : {m.nu} position servos, gain/bias/forcerange/armature/frictionloss == '
          f'X1_JOINT_PARAMS (asserted on the compiled model)')
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


def rollout(m, d, kin, task, robot, device, steps, act_to_joint, ref=None, label='zero action'):
    """N control steps at 50 Hz (decimation 4); returns the per-step diagnostics"""
    names = [nm.split('/')[-1] for nm in kin.joint_names]
    q_sp = task.q0.clone()
    prev_action = torch.zeros(1, 29, device=device)
    enc_bias = torch.zeros(1, 29, device=device)
    rows = []
    for k in range(steps):
        # the env writes the position setpoint into ctrl through act_to_joint and lets MuJoCo's <position> servo
        # produce the torque, then steps `decimation` times with ctrl held
        d.ctrl[:] = (q_sp - enc_bias[0]).cpu().numpy()[act_to_joint]
        for _ in range(DECIMATION):
            mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                raise SystemExit(f'non-finite state at control step {k}')
        qpos, qvel = d.qpos.copy(), d.qvel.copy()
        rot = quat_to_mat(torch.as_tensor(qpos[3:7], device=device, dtype=torch.float32)[None])
        s_ = {
            'pos': torch.as_tensor(qpos[:3], device=device, dtype=torch.float32)[None],
            'rot': rot,
            'q': torch.as_tensor(qpos[7:], device=device, dtype=torch.float32)[None],
            'v': torch.as_tensor(qvel[:3], device=device, dtype=torch.float32)[None],
            'w': (rot[0] @ torch.as_tensor(qvel[3:6], device=device, dtype=torch.float32))[None],
            'qd': torch.as_tensor(qvel[6:], device=device, dtype=torch.float32)[None],
        }
        rf = ref if ref is not None else s_
        kin_ref = kin.forward(rf['pos'], rf['rot'], rf['q'], rf['v'], rf['w'], rf['qd'])
        rf_future = {k: v.unsqueeze(1).expand(-1, task.cfg.future, *v.shape[1:]) for k, v in rf.items()}
        out = task.analytics(s_, rf, kin_ref, rf_future, prev_action, prev_action, enc_bias)
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
            ref = {kk: vv[1:] for kk, vv in ref.items()}
    r = rows
    print(f'\n{label}: {steps} control steps ({steps * DECIMATION * DT_SIM:.2f} s of sim), '
          f'solver iters {m.opt.iterations}/{m.opt.ls_iterations}')
    print(f'  obs          : shape {tuple(out["obs"].shape)} (OBS_DIM={OBS_DIM}), max |obs| {max(x["obs_max"] for x in r):.3f}')
    print(f'  goal         : {task.cfg.future} future frames x {GOAL_FRAME_DIM} = {out["goal"].shape[-1]}, '
          f'max |goal| {max(x["goal_max"] for x in r):.3f}')
    print(f'  base height  : start {r[0]["base_z"]:.4f} m -> min {min(x["base_z"] for x in r):.4f} / '
          f'end {r[-1]["base_z"]:.4f} m')
    print(f'  foot height  : min {min(x["foot_min"] for x in r):+.4f} m, max {max(x["foot_max"] for x in r):+.4f} m')
    print(f'  tilt (1-R_zz): max {max(x["tilt"] for x in r):.4f}   terminated: {any(x["term"] for x in r)}')
    print(f'  |qd| max     : {max(x["qd_max"] for x in r):.2f} rad/s   reward per step: mean {np.mean([x["reward"] for x in r]):+.4f}')
    return r


def grade(r, expect: str, ankle_gain: float, init_z: float) -> bool:
    """turn the rollout numbers into a verdict (the script used to end with "OK" no matter what)"""
    qd = max(x['qd_max'] for x in r)
    pen = min(x['foot_min'] for x in r)
    tile = max(x['tilt'] for x in r)
    end_z = r[-1]['base_z']
    dz = abs(end_z - init_z)
    checks = [
        (f'|qd| max {qd:.2f} <= 15 rad/s (no blow-up)', qd <= 15.0),
        (f'sole penetration {pen * 1000:+.2f} mm within -2 mm', pen >= -0.002),
    ]
    if expect == 'stand':
        checks += [
            (f'end root height {end_z:.4f} m within 3 cm of the {init_z:.4f} m standing height', dz <= 0.03),
            (f'tilt max {tile:.4f} <= 0.05', tile <= 0.05),
            (f'|qd| max {qd:.2f} <= 1 rad/s at the end', max(x['qd_max'] for x in r[-10:]) <= 1.0),
        ]
    print(f'\nverdict ({expect}, ankle kp x{ankle_gain}):')
    for name, ok in checks:
        print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    return all(ok for _, ok in checks)


def geom_lowest_z(m, d, gid: int) -> float:
    """world z of the lowest point of a geom (box: its 8 corners; sphere/capsule: centre - radius)"""
    if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
        rot = d.geom_xmat[gid].reshape(3, 3)
        half = m.geom_size[gid]
        corners = [d.geom_xpos[gid] + rot @ (half * np.array([sx, sy, sz]))
                   for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        return float(min(c[2] for c in corners))
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
                    help='scale the two ankle kp values; 1 = the X1 PD gains (a robustness probe, not a fix)')
    ap.add_argument('--iters', default=f'{SOLVER_ITERS[0]},{SOLVER_ITERS[1]}', help='solver iterations,ls_iterations')
    ap.add_argument('--expect', default='auto', choices=('auto', 'stability', 'stand'),
                    help='stability: no blow-up, no penetration; stand: also holds the pose for the whole run; '
                         'auto: stand when --ankle-gain >= 4, else stability')
    args = ap.parse_args()
    device = args.device
    iters = tuple(int(v) for v in args.iters.split(','))
    expect = args.expect
    if expect == 'auto':
        expect = 'stand'

    m, d, kin, kin_slow, names, robot, act_to_joint = build(args.xml, device, iters, args.ankle_gain)
    print('=' * 100)
    check_model(m, names, robot, args.ankle_gain)
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
    rows = rollout(m, d, kin, task, robot, device, args.steps, act_to_joint,
                   label='zero action (policy output 0, 50 Hz)')
    ok = grade(rows, expect, args.ankle_gain, z0)

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
        track = rollout(m, d, kin, task, robot, device, min(args.steps, len(idx) - 1), act_to_joint, ref=ref,
                        label='tracking the clip (the servo holds q0, so this only checks finiteness + contacts)')
        ok = ok and min(x['foot_min'] for x in track) >= -0.002
    print(f'\n{"OK" if ok else "FAIL"} (expect={expect}, ankle kp x{args.ankle_gain}, {args.steps} steps)')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
