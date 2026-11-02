"""Robot registry: one place that says what allmimic needs to know about a humanoid.

The G1 is the upstream robot and every default points at it, so the existing commands are unchanged. A robot
entry names the mjlab entity factory, the actuator action scale, the 14 reward bodies, the root / torso bodies
used by the domain randomization, the foot contact geoms, and the per-joint world-model error weights
(allmimic.joint_weights). `TrackingSimEnv` plus `Task` are the only consumers.

  from allmimic.robots import get_robot
  robot = get_robot('x1_29dof')
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Robot:
    name: str
    entity_cfg: Callable[[], object]  # () -> mjlab EntityCfg
    action_scale: dict[str, float]  # joint name or regex -> rad per unit action
    tracked_bodies: tuple[str, ...]  # 14 reward bodies, in task.TRACKED_BODIES order
    root_body: str
    torso_body: str
    foot_geoms: tuple[str, ...]  # names of the contact geoms the friction randomization touches
    joint_order: tuple[str, ...] | None  # expected model order (None = no check)
    wm_joint_weights: str = 'tracked'  # table name in allmimic.joint_weights.JOINT_WEIGHTS

    def spec(self) -> str:
        return (
            f'robot={self.name} bodies={len(self.tracked_bodies)} foot_geoms={len(self.foot_geoms)} '
            f'root={self.root_body} torso={self.torso_body} wm_weights={self.wm_joint_weights}'
        )


def _g1() -> Robot:
    from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg

    from ..task import TRACKED_BODIES

    return Robot(
        name='g1',
        entity_cfg=get_g1_robot_cfg,
        action_scale=G1_ACTION_SCALE,
        tracked_bodies=TRACKED_BODIES,
        root_body='pelvis',
        torso_body='torso_link',
        # the G1's foot collision geoms: left_foot1_collision .. left_foot7_collision, same on the right
        foot_geoms=tuple(f'{s}_foot{k}_collision' for s in ('left', 'right') for k in range(1, 8)),
        joint_order=None,
        wm_joint_weights='tracked',
    )


def _x1() -> Robot:
    from . import x1_29dof as x

    return Robot(
        name='x1_29dof',
        entity_cfg=x.get_x1_robot_cfg,
        action_scale=x.X1_ACTION_SCALE,
        tracked_bodies=x.X1_TRACKED_BODIES,
        root_body=x.X1_ROOT_BODY,
        torso_body=x.X1_TORSO_BODY,
        foot_geoms=x.X1_FOOT_GEOMS,
        joint_order=x.X1_JOINT_ORDER,
        wm_joint_weights='tracked_x1',
    )


ROBOTS = {'g1': _g1, 'x1_29dof': _x1}
DEFAULT_ROBOT = 'g1'


def get_robot(name: str | None = None) -> Robot:
    key = (name or DEFAULT_ROBOT).lower()
    if key not in ROBOTS:
        raise KeyError(f'unknown robot {name!r}; available: {sorted(ROBOTS)}')
    return ROBOTS[key]()
