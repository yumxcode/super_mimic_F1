"""X1 29-DoF humanoid (Zhiyuan / roboparty X1) constants for allmimic.

The G1 is the upstream robot and stays the default; this module adds the X1 as a second, switchable robot
(``TrackingSimEnv(..., robot='x1_29dof')``). Joint count matches the G1 (29), so ``task.NJ``, ``OBS_DIM``,
``GOAL_FRAME_DIM`` and every network shape are unchanged -- what changes is the kinematic tree, the joint
order, the joint limits, the actuators and the tracked reward bodies.

Sources
  kinematics / inertias / joint limits : ``assets/x1_29dof/x1_29dof.xml`` (vendored by tools/vendor_x1_assets.py)
  kp / kd / armature / effort limits   : ``~/code/X1_29_AMP/x1_armature_pd.md`` ("方案 B", 2026-10-14: lab torque
                                         identification + on-robot walk-log fits, kd capped at 8 by the MIT firmware)
  tracked bodies                       : the X1 bodies that play the role of the G1's 14 reward bodies
                                         (pelvis->base_link, knee_link->knee_pitch_link, torso->lumbar_pitch_link,
                                         wrist_yaw->wrist_roll_link, ...)

Joint order is the MuJoCo model order (depth-first: waist, left arm, right arm, left leg, right leg), because
that is the order ``TrackingSimEnv`` reads from the compiled model and therefore the order of the motion
library's ``q`` / ``qd`` columns. ``X1_JOINT_ORDER`` records it so a mismatch fails loudly instead of silently
scrambling the reference.
"""

from __future__ import annotations

import re
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent.parent.parent / 'assets' / 'x1_29dof'
X1_XML = ASSET_DIR / 'x1_29dof.xml'
MESH_DIR = ASSET_DIR / 'meshes'

# MuJoCo model order (see the module docstring).
X1_JOINT_ORDER = (
    'lumbar_yaw_joint',
    'lumbar_roll_joint',
    'lumbar_pitch_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_pitch_joint',
    'left_elbow_yaw_joint',
    'left_wrist_pitch_joint',
    'left_wrist_roll_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_pitch_joint',
    'right_elbow_yaw_joint',
    'right_wrist_pitch_joint',
    'right_wrist_roll_joint',
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_pitch_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_pitch_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
)
assert len(X1_JOINT_ORDER) == 29 and len(set(X1_JOINT_ORDER)) == 29

# Per-joint actuator parameters. kp [Nm/rad], kd [Nms/rad], armature [kg m^2], effort_limit [Nm].
# frictionloss is the value in the vendored MJCF (kept, so the simulated joint keeps the asset's dry friction).
X1_JOINT_PARAMS = {
    'lumbar_yaw_joint': dict(kp=90.0, kd=8.0, armature=0.022, effort=180.0, frictionloss=3.0),
    'lumbar_roll_joint': dict(kp=70.0, kd=8.0, armature=0.021, effort=180.0, frictionloss=3.0),
    'lumbar_pitch_joint': dict(kp=100.0, kd=8.0, armature=0.012, effort=180.0, frictionloss=4.0),
    'left_hip_pitch_joint': dict(kp=65.0, kd=8.0, armature=0.208, effort=180.0, frictionloss=4.0),
    'right_hip_pitch_joint': dict(kp=65.0, kd=8.0, armature=0.208, effort=180.0, frictionloss=4.0),
    'left_hip_roll_joint': dict(kp=60.0, kd=8.0, armature=0.25, effort=180.0, frictionloss=3.0),
    'right_hip_roll_joint': dict(kp=60.0, kd=8.0, armature=0.25, effort=180.0, frictionloss=3.0),
    'left_hip_yaw_joint': dict(kp=100.0, kd=3.3, armature=0.0148, effort=180.0, frictionloss=3.0),
    'right_hip_yaw_joint': dict(kp=100.0, kd=3.3, armature=0.0148, effort=180.0, frictionloss=3.0),
    'left_knee_pitch_joint': dict(kp=85.0, kd=8.0, armature=0.273, effort=180.0, frictionloss=4.0),
    'right_knee_pitch_joint': dict(kp=85.0, kd=8.0, armature=0.273, effort=180.0, frictionloss=4.0),
    'left_ankle_pitch_joint': dict(kp=50.0, kd=4.0, armature=0.15, effort=80.0, frictionloss=1.5),
    'right_ankle_pitch_joint': dict(kp=50.0, kd=4.0, armature=0.15, effort=80.0, frictionloss=1.5),
    'left_ankle_roll_joint': dict(kp=50.0, kd=2.5, armature=0.035, effort=80.0, frictionloss=1.5),
    'right_ankle_roll_joint': dict(kp=50.0, kd=2.5, armature=0.035, effort=80.0, frictionloss=1.5),
    'left_shoulder_pitch_joint': dict(kp=40.0, kd=4.0, armature=0.02, effort=27.0, frictionloss=0.5),
    'right_shoulder_pitch_joint': dict(kp=40.0, kd=4.0, armature=0.02, effort=27.0, frictionloss=0.5),
    'left_shoulder_roll_joint': dict(kp=40.0, kd=4.0, armature=0.013, effort=27.0, frictionloss=0.5),
    'right_shoulder_roll_joint': dict(kp=40.0, kd=4.0, armature=0.013, effort=27.0, frictionloss=0.5),
    'left_shoulder_yaw_joint': dict(kp=40.0, kd=2.0, armature=0.01, effort=27.0, frictionloss=0.5),
    'right_shoulder_yaw_joint': dict(kp=40.0, kd=2.0, armature=0.01, effort=27.0, frictionloss=0.5),
    'left_elbow_pitch_joint': dict(kp=30.0, kd=1.9, armature=0.01, effort=27.0, frictionloss=0.5),
    'right_elbow_pitch_joint': dict(kp=30.0, kd=1.9, armature=0.01, effort=27.0, frictionloss=0.5),
    'left_elbow_yaw_joint': dict(kp=20.0, kd=1.0, armature=0.01, effort=27.0, frictionloss=0.5),
    'right_elbow_yaw_joint': dict(kp=20.0, kd=1.0, armature=0.01, effort=27.0, frictionloss=0.5),
    'left_wrist_pitch_joint': dict(kp=15.0, kd=1.0, armature=0.01, effort=10.0, frictionloss=0.2),
    'right_wrist_pitch_joint': dict(kp=15.0, kd=1.0, armature=0.01, effort=10.0, frictionloss=0.2),
    'left_wrist_roll_joint': dict(kp=15.0, kd=1.0, armature=0.01, effort=10.0, frictionloss=0.2),
    'right_wrist_roll_joint': dict(kp=15.0, kd=1.0, armature=0.01, effort=10.0, frictionloss=0.2),
}
assert set(X1_JOINT_PARAMS) == set(X1_JOINT_ORDER)

# Action scale, same definition as the G1 (mjlab: 0.25 * effort_limit / stiffness): the joint displacement at
# which the position actuator saturates its torque.
X1_ACTION_SCALE = {n: 0.25 * p['effort'] / p['kp'] for n, p in X1_JOINT_PARAMS.items()}

# Reward bodies: the X1 analogues of the G1's TRACKED_BODIES (allmimic.task.TRACKED_BODIES).
X1_TRACKED_BODIES = (
    'base_link',  # pelvis
    'left_hip_roll_link',
    'left_knee_pitch_link',
    'left_ankle_roll_link',
    'right_hip_roll_link',
    'right_knee_pitch_link',
    'right_ankle_roll_link',
    'lumbar_pitch_link',  # torso
    'left_shoulder_roll_link',
    'left_elbow_pitch_link',
    'left_wrist_roll_link',
    'right_shoulder_roll_link',
    'right_elbow_pitch_link',
    'right_wrist_roll_link',
)
X1_ROOT_BODY = 'base_link'
X1_TORSO_BODY = 'lumbar_pitch_link'
# Foot contact geoms: four 2 mm corner spheres per foot, the only collision geoms below the ankle in the asset (the
# vendored MJCF is byte-identical to the source; see assets/x1_29dof/PROVENANCE.md).
X1_FOOT_GEOMS = tuple(f'{s}_foot_{k}_collision' for s in ('left', 'right') for k in (1, 2, 3, 4))
# Regex the entity's CollisionCfg keys on, built from the tuple above. It has to be derived: when the asset's foot
# geoms were renamed once, a hand-written regex silently matched nothing, the soles kept the asset's contact defaults
# (condim 1 from the CollisionCfg fallback branch, the 5 ms solref) and nothing failed until the robot was simulated.
X1_FOOT_GEOM_RE = '^(' + '|'.join(X1_FOOT_GEOMS) + ')$'
# The asset's contact default is solref="0.005 1", tuned for its own 1 kHz scene. allmimic steps physics at 200 Hz
# (mjlab SimulationCfg timestep 0.005), where a 5 ms contact time constant blows up (measured: |qd| -> 1e6 rad/s
# within a second). MuJoCo's own default 20 ms is stable at 200 Hz, so the foot contacts use that instead. Nothing
# else about the contact model changes.
X1_FOOT_SOLREF = (0.02, 1.0)
X1_FOOT_CONDIM = 3  # normal + 2 friction directions; the soles carry the whole weight, so they need friction
X1_FOOT_PRIORITY = 1  # the sole contact wins over any other contact on the same pair
X1_FOOT_FRICTION = (0.6,)  # sliding friction; the domain randomization overwrites [0] per episode

# Initial root height used by EntityCfg (the reset pose comes from the motion reference, so this only sets the
# state the simulator starts in). Straight-leg stance, measured on the compiled model at q=0: the sole spheres'
# lowest point is 0.62750 m below base_link, so the soles touch z=0 there. Starting higher free-falls the robot onto
# its feet and injects a contact impulse at t=0 (a flat 0.66 would be 3.25 cm too high).
X1_INIT_HEIGHT = 0.6275

# Joint pose the observation / absolute actions are centred on (Task.q0). Zeros = the MJCF rest pose, which is
# a flat-footed straight-leg stance (all four foot spheres at the same height).
X1_HOME_JOINT_POS = {n: 0.0 for n in X1_JOINT_ORDER}


def x1_spec():
    """MjSpec of the X1 robot (no floor, no actuator -- mjlab adds position actuators from X1_ARTICULATION)."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(X1_XML))
    spec.assets = {f'{spec.meshdir}/{p.name}': p.read_bytes() for p in sorted(MESH_DIR.iterdir()) if p.is_file()}
    for actuator in list(spec.actuators):  # the asset ships 29 <motor>s; mjlab replaces them with position servos
        spec.delete(actuator)
    return spec


def x1_actuator_groups():
    """[(kp, kd, effort, armature, frictionloss, (joint, ...)), ...] -- one group per distinct parameter tuple.

    This is the single table both mjlab's actuator configs and scripts/x1_smoke.py iterate, so a joint cannot end up
    with different gains in the two paths.
    """
    groups: dict[tuple, list[str]] = {}
    for name in X1_JOINT_ORDER:
        p = X1_JOINT_PARAMS[name]
        key = (p['kp'], p['kd'], p['effort'], p['armature'], p['frictionloss'])
        groups.setdefault(key, []).append(name)
    return [(kp, kd, effort, armature, frictionloss, tuple(names))
            for (kp, kd, effort, armature, frictionloss), names in groups.items()]


def x1_actuator_cfgs():
    """(BuiltinPositionActuatorCfg, ...) -- one group per distinct (kp, kd, effort, armature, frictionloss)."""
    from mjlab.actuator import BuiltinPositionActuatorCfg

    return tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=names,
            stiffness=kp,
            damping=kd,
            effort_limit=effort,
            armature=armature,
            frictionloss=frictionloss,
        )
        for kp, kd, effort, armature, frictionloss, names in x1_actuator_groups()
    )


def x1_collision_cfgs():
    """(CollisionCfg,) for the entity -- the sole boxes get condim 3 + friction + priority + the 200 Hz solref.

    Kept separate from get_x1_robot_cfg() so scripts/x1_smoke.py can apply the *same* object to a real MjSpec and
    assert the compiled geoms carry these values: this is the layer that decides what the simulator actually uses,
    and a regex that matches nothing fails silently.
    """
    from mjlab.utils.spec_config import CollisionCfg

    foot = X1_FOOT_GEOM_RE
    # Everything that is not a foot keeps condim 1 (the asset only gives the base and the torso a collision mesh).
    return (
        CollisionCfg(
            geom_names_expr=('.*_collision',),
            condim={foot: X1_FOOT_CONDIM, '.*_collision': 1},
            priority={foot: X1_FOOT_PRIORITY},
            friction={foot: X1_FOOT_FRICTION},
            solref={foot: X1_FOOT_SOLREF},
        ),
    )


def get_x1_robot_cfg():
    """EntityCfg of the X1 29-DoF robot for mjlab (the G1 stays available as get_g1_robot_cfg())."""
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    articulation = EntityArticulationInfoCfg(actuators=x1_actuator_cfgs(), soft_joint_pos_limit_factor=0.9)
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, X1_INIT_HEIGHT),
            joint_pos={n: X1_HOME_JOINT_POS[n] for n in X1_JOINT_ORDER},
            joint_vel={'.*': 0.0},
        ),
        collisions=x1_collision_cfgs(),
        spec_fn=x1_spec,
        articulation=articulation,
    )


def joint_names_match(names) -> None:
    """Assert the compiled model's hinge order equals X1_JOINT_ORDER (cheap guard against a silent mismatch)."""
    got = [re.sub(r'^.*/', '', n) for n in names]
    if got != list(X1_JOINT_ORDER):
        raise RuntimeError(f'X1 joint order mismatch:\n  model: {got}\n  config: {list(X1_JOINT_ORDER)}')
