"""X1 29-DoF joint inventory: names, model order, limits, rest pose, inertias, foot links, actuator parameters.

The table this prints is the contract the motion library has to satisfy: the npz `q` columns are in the X1
MJCF model order (== the order TrackingSimEnv reads out of the compiled model), so the retargeted data must be
written in this order.

  PYTHONPATH=. python tools/x1_joint_table.py                     # print + write docs/x1_29dof_joints.md
  PYTHONPATH=. python tools/x1_joint_table.py --json out.json
Needs mujoco only (no GPU, no mjlab).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from allmimic.robots import x1_29dof as x1

ROOT = Path(__file__).resolve().parent.parent


def inventory(xml: Path = x1.X1_XML) -> dict:
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)  # qpos0: rest pose
    jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
    gn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)]
    hinges = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    assert len(hinges) == 29, len(hinges)
    joints = []
    for k, j in enumerate(hinges):
        b = int(m.jnt_bodyid[j])
        p = x1.X1_JOINT_PARAMS[jn[j]]
        joints.append(
            {
                'index': k,
                'name': jn[j],
                'body': bn[b],
                'parent_body': bn[int(m.body_parentid[b])],
                'axis': [float(a) for a in m.jnt_axis[j]],
                'range': [float(a) for a in m.jnt_range[j]],
                'rest_world_z': float(d.xpos[b][2]),
                'body_mass': float(m.body_mass[b]),
                'body_diaginertia': [float(a) for a in m.body_inertia[b]],
                'qposadr': int(m.jnt_qposadr[j]),
                'dofadr': int(m.jnt_dofadr[j]),
                'damping': float(m.dof_damping[m.jnt_dofadr[j]]),
                'frictionloss': float(m.dof_frictionloss[m.jnt_dofadr[j]]),
                'kp': p['kp'],
                'kd': p['kd'],
                'armature': p['armature'],
                'effort_limit': p['effort'],
                'action_scale': x1.X1_ACTION_SCALE[jn[j]],
            }
        )
    foot = {}
    for g in range(m.ngeom):
        if gn[g] in x1.X1_FOOT_GEOMS:
            foot[gn[g]] = {
                'body': bn[int(m.geom_bodyid[g])],
                'radius': float(m.geom_size[g][0]),
                'rest_world_pos': [float(v) for v in d.geom_xpos[g]],
            }
    return {
        'xml': str(xml),
        'nq': int(m.nq),
        'nv': int(m.nv),
        'nu': int(m.nu),
        'nbody': int(m.nbody),
        'total_mass': float(m.body_mass.sum()),
        'free_joint': next(jn[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE),
        'root_body': bn[int(m.jnt_bodyid[0])] if m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE else bn[1],
        'joints': joints,
        'foot_geoms': foot,
        'foot_links': sorted({v['body'] for v in foot.values()}),
        'sole_below_root': float(d.qpos if False else (d.xpos[1][2] - min(v['rest_world_pos'][2] for v in foot.values()))),
    }


def to_markdown(inv: dict) -> str:
    L = [
        '# X1 29-DoF joint inventory',
        '',
        f'Source: `{inv["xml"]}` (vendored; see `assets/x1_29dof/PROVENANCE.md` for sha256 and the changes).',
        '',
        f'- model sizes: `nq={inv["nq"]}` (7 free + 29 hinge), `nv={inv["nv"]}`, `nu={inv["nu"]}`, `nbody={inv["nbody"]}`',
        f'- floating base: free joint `{inv["free_joint"]}` on body `{inv["root_body"]}`',
        f'- total mass: **{inv["total_mass"]:.3f} kg**',
        f'- foot links: {", ".join(f"`{b}`" for b in inv["foot_links"])}',
        f'- sole sits **{inv["sole_below_root"]:.4f} m** below the root body in the rest pose',
        '',
        '## Joints (this order == the motion npz `q` column order == the policy action order)',
        '',
        '| # | joint | body | parent | axis | range [rad] | damping | frictionloss | kp | kd | armature | effort [Nm] | action scale |',
        '|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for j in inv['joints']:
        ax = ' '.join(f'{a:+.0f}' for a in j['axis'])
        L.append(
            f'| {j["index"]} | `{j["name"]}` | `{j["body"]}` | `{j["parent_body"]}` | {ax} | '
            f'[{j["range"][0]:+.2f}, {j["range"][1]:+.2f}] | {j["damping"]:.2f} | {j["frictionloss"]:.2f} | '
            f'{j["kp"]:.0f} | {j["kd"]:.1f} | {j["armature"]:.3f} | {j["effort_limit"]:.0f} | {j["action_scale"]:.4f} |'
        )
    L += [
        '',
        '`action scale = 0.25 * effort_limit / kp` (mjlab\'s definition, same as the G1 path): the joint',
        'displacement at which the position servo saturates its torque.',
        '',
        'Inertias are per link in the MJCF (`body_diaginertia`, see the JSON output); the joint table above',
        'reports the mass of the body each joint drives, which is what dominates the PD stability margin.',
        '',
        '## Foot contact geoms',
        '',
        '| geom | body | radius [m] | rest position (world, root at z=0.8) |',
        '|---|---|---:|---|',
    ]
    for name, f in inv['foot_geoms'].items():
        pos = ', '.join(f'{v:+.4f}' for v in f['rest_world_pos'])
        L.append(f'| `{name}` | `{f["body"]}` | {f["radius"]:.4f} | ({pos}) |')
    L += [
        '',
        'The asset models each sole with four 2 mm collision spheres and nothing else below the ankle: there is',
        'no sole box, so foot contact is four point contacts per foot and the shank / ankle has no collision',
        'geometry. The vendored MJCF keeps this as the source has it; allmimic.robots.x1_29dof gives these eight',
        'geoms condim 3, priority 1, friction 0.6 and solref 20 ms through the entity CollisionCfg (the asset',
        'default is 5 ms, unstable at the 200 Hz timestep), which the CPU smoke asserts on the compiled model.',
        'With that applied the four-sphere stance is stable at the real PD gains (4 s: 0.6275 -> 0.6256 m,',
        'tilt 0.0036, |qd| 0.16 rad/s); a flat sole box is a modelling option, not a fix for a measured defect.',
        '',
    ]
    return '\n'.join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--xml', type=Path, default=x1.X1_XML)
    ap.add_argument('--out', type=Path, default=ROOT / 'docs' / 'x1_29dof_joints.md')
    ap.add_argument('--json', type=Path, default=None, help='also write the raw inventory as JSON')
    args = ap.parse_args()
    inv = inventory(args.xml)
    table = to_markdown(inv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table)
    print(table)
    print(f'\nwrote {args.out}')
    if args.json:
        args.json.write_text(json.dumps(inv, indent=1))
        print(f'wrote {args.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
