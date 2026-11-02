# X1 29-DoF joint inventory

Source: `/Users/yumx/code/super_mimic_F1/assets/x1_29dof/x1_29dof.xml` (vendored; see `assets/x1_29dof/PROVENANCE.md` for sha256 and the changes).

- model sizes: `nq=36` (7 free + 29 hinge), `nv=35`, `nu=29`, `nbody=31`
- floating base: free joint `floating_base` on body `base_link`
- total mass: **35.031 kg**
- foot links: `left_ankle_roll_link`, `right_ankle_roll_link`
- sole sits **0.6255 m** below the root body in the rest pose

## Joints (this order == the motion npz `q` column order == the policy action order)

| # | joint | body | parent | axis | range [rad] | damping | frictionloss | kp | kd | armature | effort [Nm] | action scale |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `lumbar_yaw_joint` | `lumbar_yaw_link` | `base_link` | +0 +0 +1 | [-1.00, +1.00] | 1.50 | 3.00 | 90 | 8.0 | 0.022 | 180 | 0.5000 |
| 1 | `lumbar_roll_joint` | `lumbar_roll_link` | `lumbar_yaw_link` | +0 +0 +1 | [-0.50, +0.50] | 1.50 | 3.00 | 70 | 8.0 | 0.021 | 180 | 0.6429 |
| 2 | `lumbar_pitch_joint` | `lumbar_pitch_link` | `lumbar_roll_link` | +0 +0 -1 | [-0.50, +1.60] | 2.00 | 4.00 | 100 | 8.0 | 0.012 | 180 | 0.4500 |
| 3 | `left_shoulder_pitch_joint` | `left_shoulder_pitch_link` | `lumbar_pitch_link` | +0 +0 -1 | [-3.00, +1.50] | 0.20 | 0.50 | 40 | 4.0 | 0.020 | 27 | 0.1688 |
| 4 | `left_shoulder_roll_joint` | `left_shoulder_roll_link` | `left_shoulder_pitch_link` | +0 +0 -1 | [-3.00, +0.00] | 0.20 | 0.50 | 40 | 4.0 | 0.013 | 27 | 0.1688 |
| 5 | `left_shoulder_yaw_joint` | `left_shoulder_yaw_link` | `left_shoulder_roll_link` | +0 +0 +1 | [-1.80, +1.80] | 0.20 | 0.50 | 40 | 2.0 | 0.010 | 27 | 0.1688 |
| 6 | `left_elbow_pitch_joint` | `left_elbow_pitch_link` | `left_shoulder_yaw_link` | +0 +0 +1 | [+0.00, +2.00] | 0.20 | 0.50 | 30 | 1.9 | 0.010 | 27 | 0.2250 |
| 7 | `left_elbow_yaw_joint` | `left_elbow_yaw_link` | `left_elbow_pitch_link` | +0 +0 +1 | [-1.80, +1.80] | 0.20 | 0.50 | 20 | 1.0 | 0.010 | 27 | 0.3375 |
| 8 | `left_wrist_pitch_joint` | `left_wrist_pitch_link` | `left_elbow_yaw_link` | +1 +0 +0 | [-1.00, +1.00] | 0.10 | 0.20 | 15 | 1.0 | 0.010 | 10 | 0.1667 |
| 9 | `left_wrist_roll_joint` | `left_wrist_roll_link` | `left_wrist_pitch_link` | +0 +1 +0 | [-1.00, +1.00] | 0.10 | 0.20 | 15 | 1.0 | 0.010 | 10 | 0.1667 |
| 10 | `right_shoulder_pitch_joint` | `right_shoulder_pitch_link` | `lumbar_pitch_link` | +0 +0 +1 | [-3.00, +1.50] | 0.20 | 0.50 | 40 | 4.0 | 0.020 | 27 | 0.1688 |
| 11 | `right_shoulder_roll_joint` | `right_shoulder_roll_link` | `right_shoulder_pitch_link` | +0 +0 +1 | [-3.00, +0.00] | 0.20 | 0.50 | 40 | 4.0 | 0.013 | 27 | 0.1688 |
| 12 | `right_shoulder_yaw_joint` | `right_shoulder_yaw_link` | `right_shoulder_roll_link` | +0 +0 +1 | [-1.80, +1.80] | 0.20 | 0.50 | 40 | 2.0 | 0.010 | 27 | 0.1688 |
| 13 | `right_elbow_pitch_joint` | `right_elbow_pitch_link` | `right_shoulder_yaw_link` | +0 +0 -1 | [+0.00, +2.00] | 0.20 | 0.50 | 30 | 1.9 | 0.010 | 27 | 0.2250 |
| 14 | `right_elbow_yaw_joint` | `right_elbow_yaw_link` | `right_elbow_pitch_link` | +0 +0 -1 | [-1.80, +1.80] | 0.20 | 0.50 | 20 | 1.0 | 0.010 | 27 | 0.3375 |
| 15 | `right_wrist_pitch_joint` | `right_wrist_pitch_link` | `right_elbow_yaw_link` | +1 +0 +0 | [-1.00, +1.00] | 0.10 | 0.20 | 15 | 1.0 | 0.010 | 10 | 0.1667 |
| 16 | `right_wrist_roll_joint` | `right_wrist_roll_link` | `right_wrist_pitch_link` | +0 +1 +0 | [-1.00, +1.00] | 0.10 | 0.20 | 15 | 1.0 | 0.010 | 10 | 0.1667 |
| 17 | `left_hip_pitch_joint` | `left_hip_pitch_link` | `base_link` | +0 +0 +1 | [-2.00, +3.14] | 2.00 | 4.00 | 65 | 8.0 | 0.208 | 180 | 0.6923 |
| 18 | `left_hip_roll_joint` | `left_hip_roll_link` | `left_hip_pitch_link` | +0 +0 -1 | [-2.70, +0.30] | 1.50 | 3.00 | 60 | 8.0 | 0.250 | 180 | 0.7500 |
| 19 | `left_hip_yaw_joint` | `left_hip_yaw_link` | `left_hip_roll_link` | +0 +0 -1 | [-1.50, +1.50] | 1.50 | 3.00 | 100 | 3.3 | 0.015 | 180 | 0.4500 |
| 20 | `left_knee_pitch_joint` | `left_knee_pitch_link` | `left_hip_yaw_link` | +0 +0 +1 | [+0.00, +2.70] | 2.00 | 4.00 | 85 | 8.0 | 0.273 | 180 | 0.5294 |
| 21 | `left_ankle_pitch_joint` | `left_ankle_pitch_link` | `left_knee_pitch_link` | +0 +0 -1 | [-0.50, +0.80] | 0.80 | 1.50 | 50 | 4.0 | 0.150 | 80 | 0.4000 |
| 22 | `left_ankle_roll_joint` | `left_ankle_roll_link` | `left_ankle_pitch_link` | +0 +0 +1 | [-0.50, +0.50] | 0.80 | 1.50 | 50 | 2.5 | 0.035 | 80 | 0.4000 |
| 23 | `right_hip_pitch_joint` | `right_hip_pitch_link` | `base_link` | +0 +0 +1 | [-3.14, +2.00] | 2.00 | 4.00 | 65 | 8.0 | 0.208 | 180 | 0.6923 |
| 24 | `right_hip_roll_joint` | `right_hip_roll_link` | `right_hip_pitch_link` | +0 +0 +1 | [-0.30, +2.70] | 1.50 | 3.00 | 60 | 8.0 | 0.250 | 180 | 0.7500 |
| 25 | `right_hip_yaw_joint` | `right_hip_yaw_link` | `right_hip_roll_link` | +0 +0 +1 | [-1.50, +1.50] | 1.50 | 3.00 | 100 | 3.3 | 0.015 | 180 | 0.4500 |
| 26 | `right_knee_pitch_joint` | `right_knee_pitch_link` | `right_hip_yaw_link` | +0 +0 -1 | [-0.00, +2.70] | 2.00 | 4.00 | 85 | 8.0 | 0.273 | 180 | 0.5294 |
| 27 | `right_ankle_pitch_joint` | `right_ankle_pitch_link` | `right_knee_pitch_link` | +0 +0 +1 | [-0.50, +0.80] | 0.80 | 1.50 | 50 | 4.0 | 0.150 | 80 | 0.4000 |
| 28 | `right_ankle_roll_joint` | `right_ankle_roll_link` | `right_ankle_pitch_link` | +0 +0 +1 | [-0.50, +0.50] | 0.80 | 1.50 | 50 | 2.5 | 0.035 | 80 | 0.4000 |

`action scale = 0.25 * effort_limit / kp` (mjlab's definition, same as the G1 path): the joint
displacement at which the position servo saturates its torque.

Inertias are per link in the MJCF (`body_diaginertia`, see the JSON output); the joint table above
reports the mass of the body each joint drives, which is what dominates the PD stability margin.

## Foot contact geoms

| geom | body | radius [m] | rest position (world, root at z=0.8) |
|---|---|---:|---|
| `left_foot_1_collision` | `left_ankle_roll_link` | 0.0020 | (+0.0723, +0.1638, +0.1745) |
| `left_foot_2_collision` | `left_ankle_roll_link` | 0.0020 | (+0.0723, +0.1038, +0.1745) |
| `left_foot_3_collision` | `left_ankle_roll_link` | 0.0020 | (-0.0677, +0.1638, +0.1745) |
| `left_foot_4_collision` | `left_ankle_roll_link` | 0.0020 | (-0.0677, +0.1038, +0.1745) |
| `right_foot_1_collision` | `right_ankle_roll_link` | 0.0020 | (+0.0723, -0.1640, +0.1745) |
| `right_foot_2_collision` | `right_ankle_roll_link` | 0.0020 | (+0.0723, -0.1040, +0.1745) |
| `right_foot_3_collision` | `right_ankle_roll_link` | 0.0020 | (-0.0677, -0.1640, +0.1745) |
| `right_foot_4_collision` | `right_ankle_roll_link` | 0.0020 | (-0.0677, -0.1040, +0.1745) |

The asset models each sole with four 2 mm collision spheres and nothing else below the ankle: there is
no sole box, so foot contact is four point contacts per foot and the shank / ankle has no collision
geometry. Recorded as a known limitation (see the delivery report).
