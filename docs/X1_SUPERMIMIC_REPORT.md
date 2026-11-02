# X1 29-DoF on allmimic: asset, motion library, platform run

Branch `feat/x1-29dof`, remote `https://github.com/yumxcode/super_mimic_F1`. This file records what was done to
put the Zhiyuan/roboparty X1 29-DoF humanoid through allmimic (a SuperMimic re-implementation), with the numbers
each step was decided on. Anything that was **not** run says so.

Status at a glance

| step | result | evidence |
|---|---|---|
| A asset + config numerically stable | **done, with one stated deviation pending your acceptance** — see [the 0.657 m question](#the-0657-m-target) | `scripts/x1_smoke.py --steps 200`: 5/5 verdict checks pass, no QACC/NaN; no contact/geometry change beyond the documented vendoring |
| B GMR → 50 Hz training library | **done** | `motions/x1_cmu_walk.npz` (41 clips, 6168 frames, 1.75 MB) + `motions/x1_cmu_walk.meta.json` |
| C Gradmotion run | **done** — `TASK_20261102_043`, 3000 iterations, finished (status 5), 5 checkpoints, 33 metric keys | see [Gradmotion](#gradmotion) |
| D commits + push | **done** | `git ls-remote origin feat/x1-29dof` → the branch head; `git log --oneline cb334ff..feat/x1-29dof` lists the commits. No force push; each push printed an old..new range (see [Commits and pushes](#commits-and-pushes)) |
| G1 path | **done: static + runtime (path only)** | `TASK_20261102_067` trained 40 iterations with `--robot g1` (status 5, 0.45 s/iter, 4 checkpoints), fed X1 motion, so it proves the path runs, not G1 quality — see [G1 path](#g1-path) |
| E this report | **done** (C section updated as runs finish) | |

---

## A. Making the X1 numerically stable

### What the failure actually was

The starting symptom (`scripts/x1_smoke.py --steps 100`) was `Nan, Inf or huge value in QACC at DOF 14` at
t ≈ 0.72 s, with the root dropping 0.6575 → 0.394 m, |qd| reaching 3233 rad/s and tilt 1.98. My first reading was
that the asset was at fault — each foot has only four 2 mm collision spheres and no sole box — and I changed the
vendored MJCF to a flat sole box. **That was wrong**, and the evidence that overturned it came from building the
model the way the training env does and comparing field by field:

| what was different | measured effect |
|---|---|
| torque written into `qfrc_applied[6:]` **in actuator order** | the asset's first `<motor>` is `left_hip_pitch` (dof 17), so the wrist servos received the leg torques; the model diverged in 50 ms. `sim_env` does not do this — it writes `d.ctrl[act_to_joint]` |
| armature written into the **compiled** model instead of the spec | `dof_M0` (the inertia the constraint solver regularizes with) stays stale: the knee entry stays at 0.105846 instead of 0.378846, difference = the knee armature 0.273. Same model, same contact: stands for 4 s one way, topples at 2.2 s the other |
| init height 0.66 with the soles 3.2 cm in the air | every rollout started with a drop and a contact impulse; the geometric standing height is 0.6275 m |

The first two are harness/build artifacts, not properties of the robot. With both fixed, **the asset's own four
2 mm sole spheres stand at the real PD gains.** Since the contract says to modify the asset only if the asset is
the defect, the sole box was reverted. The vendored XML now differs from the source only by the mechanical
transform `tools/vendor_x1_assets.py` documents: `meshdir` rewritten to the flat mesh layout, and a `name` added to
each unnamed geom. **No geometry, contact parameter, inertia, joint limit or body pose differs from the source** —
that is the accurate statement rather than "byte-identical", since the file is not byte-identical to the source, it is
identical after those two documented transforms (vendored sha256 `e8afe910…`). `tools/vendor_x1_assets.py --check`
re-derives the transform and diffs it against the vendored copy (OK), and `PROVENANCE.md` lists exactly three changes
with no fourth entry for the reverted box.

A second, real defect was in the **configuration** layer, the layer that actually decides what the simulator uses:

* `allmimic/robots/x1_29dof.py` keyed its `CollisionCfg` on a hand-written regex
  `^(left|right)_foot_[1-4]_collision$`. When the geoms were renamed to sole boxes, the regex matched nothing, the
  soles kept `condim 1` (no friction) and the asset's 5 ms `solref` (unstable at 200 Hz), and nothing failed until
  the robot was simulated. `X1_FOOT_GEOM_RE` is now derived from `X1_FOOT_GEOMS`, so the two cannot drift apart.
* `scripts/x1_smoke.py` used to write `solref` into its own model by hand and end with an unconditional
  `no NaN, no compile error: OK`. It now builds the model with **mjlab's own** `CollisionCfg.edit_spec` and
  `create_position_actuator` (only mjlab's `tyro` / `mujoco_warp` import chain is stubbed — neither is needed on
  CPU), asserts the compiled contact and actuator parameters, and exits non-zero on failure.

### Fix before / after

Same command, `MJLAB_SRC=<unpacked mjlab> PYTHONPATH=. python scripts/x1_smoke.py --steps 200`, zero action
(policy output 0 → `ctrl` = `q0`), 200 control steps at 50 Hz = 4 s:

| quantity | before (broken harness + old config regex) | after |
|---|---|---|
| QACC / NaN warnings | `Nan, Inf or huge value in QACC at DOF 14` at t ≈ 0.72 s | none (0 lines matching QACC/NaN/unstable) |
| |qd| max | 3233 rad/s | 0.16 rad/s |
| root height | 0.6575 → 0.394 m (0.6208 at t=2 s) | 0.6271 → min 0.6256, end 0.6256 m |
| tilt (1−R_zz) max | 1.98 | 0.0036 |
| foot (sole) height | max +1.19 m, min −0.0156 m | min −0.0004 m, max −0.0001 m |
| end of run | diverged, robot on the floor | standing |

The verdict block shipped with the script:

```
verdict (stand, ankle kp x1.0):
  [PASS] |qd| max 0.16 <= 15 rad/s (no blow-up)
  [PASS] sole penetration -0.40 mm within -2 mm
  [PASS] end root height 0.6256 m within 3 cm of the 0.6275 m standing height
  [PASS] tilt max 0.0036 <= 0.05
  [PASS] |qd| max 0.16 <= 1 rad/s at the end
OK (expect=stand, ankle kp x1.0, 200 steps)
```

`--ankle-gain 8` (8× the two ankle `kp`) also passes, with tilt 0.0000 and |qd| 0.01. It is kept as a robustness
probe, not as the way to make the stance work.

**Acceptance criterion, stated deviation.** The contract's acceptance says "zero action 4 s root height ≈ 0.657 m
(sag < 3 cm)". 0.657 m came from `X1_INIT_HEIGHT = 0.66`, which is 3.2 cm above the height at which the soles
touch the floor; measured on the compiled model the geometric standing height is **0.6275 m**, and the run ends at
0.6256 m, i.e. 1.9 mm of sag, well inside the 3 cm bound. `X1_INIT_HEIGHT` is the measured value and the smoke
asserts the model agrees with it to 2 mm. The original "settle at 0.657 m" target is not achievable with the soles
on the ground — it was the pre-drop height.

### The 0.657 m target

The contract's acceptance for A says "zero action 4 s root height ≈ 0.657 m (sag < 3 cm)". Three different numbers are
in play and they are not the same quantity:

| figure | value | what it is |
|---|---|---|
| contract target | 0.657 m | `X1_INIT_HEIGHT = 0.66` from the earlier code, i.e. the root height **before** the soles touch the floor |
| geometric standing height | **0.6275 m** | measured on the compiled model at `q0`: the sole spheres' lowest point is exactly on z = 0 here |
| measured, 4 s of zero action | **0.6256 m** | the end of `scripts/x1_smoke.py --steps 200`; sag 1.9 mm |

0.657 m is not reachable with the feet on the ground — at that height the soles are 3.2 cm in the air, which is why
every early rollout started with a drop and a contact impulse. The smoke asserts the model agrees with
`X1_INIT_HEIGHT` to 2 mm and grades the end height against the **geometric** standing height (0.6275 m) with the
contract's 3 cm tolerance. So the sag criterion is met with 28 mm to spare, but the reference point moved.

**This is a deviation from the literal acceptance criterion and it is left for you to accept**: either the target was
about the pre-drop height (then A passes as written) or about a robot standing 3 cm taller than this asset can stand
(then the target itself needs revising). A is marked "done with a stated deviation" in the table above rather than
quietly counted as meeting 0.657 m.

### What is asserted on the compiled model

Read off `m` after `spec.compile()` in `check_model`, not assumed from the config:

* both sole geoms: `condim 3`, `priority 1`, `solref (0.02, 1.0)`, `friction[0] 0.6`, `contype/conaffinity 1`;
* every other `*_collision` geom (`base_collision`, `lumbar_pitch_collision`): `condim 1`;
* all 29 actuators: `gainprm[0] = kp`, `biasprm[1:3] = (−kp, −kd)`, `forcerange = ±effort`,
  `dof_armature = armature`, `dof_frictionloss = frictionloss` — i.e. `X1_JOINT_PARAMS` is what the simulator has;
* the actuator order is printed next to the joint order, because they are not the same in this asset.

### G1 path

**Static.** Unchanged and still the default: `allmimic/robots/__init__.py: DEFAULT_ROBOT = 'g1'`, `TrainCfg.robot =
'g1'`, `evaluate.make_eval_env(..., robot='g1')`. The G1 joint-weight tables `tracked` and `axis_sq` are identical to
upstream `cb334ff` (both files executed and the tables compared: 29 joints each, equal), and the X1 tables are
additional entries (`tracked_x1`, `axis_sq_x1`). `NJ = 29` and `OBS_DIM = 93` are unchanged, so every network shape
and every existing command is untouched.

**Runtime.** `TASK_20261102_067` — the same entry script, image and branch, `--robot g1`, 40 iterations / 512 envs
(`configs/g1_path_check.json`, `scripts/gm/create_g1_path_check.json`): status 5, exit 0, 0.45 s/iter, iterations 1-40
logged, four checkpoints published.

```
[gm-entry] trainer [... '-m', 'allmimic.train', '--robot', 'g1', '--preset', 'recommended', ...]
it 10 upd 4 h 0.059 iter 0.465s ret 0.065 len 21.6 term 0.858 wm1 1.4789 wm8 11.2572 J 0.138 surv 0.015
it 40 upd 16 h 0.063 iter 0.474s ret 0.052 len 23.1 term 0.930 wm1 0.5465 wm8 2.6314 J 0.144 surv 0.058
[gm-entry] published .../output/x1_supermimic_iter000040.pt
```

**This is a path test, not a G1 result.** The repository ships no G1 motion library (`data/` is gitignored), so the
run was fed `motions/x1_cmu_walk.npz`: 36-column X1 qpos read in G1 joint order. The reference is therefore nonsense
for this robot, which is exactly why the returns sit at 0.05-0.07 with 86-93 % of episodes terminating — the number
to read is that the path *ran*, not what it scored. What it does prove:

- mjlab's G1 entity, `G1_ACTION_SCALE`, `TRACKED_BODIES`, the `tracked` joint-weight table and `MotionLibrary` all
  still build and train after the X1 changes;
- the WM-weight fallback added in `b305b38` does not fire for the G1 — there is no "using the registry table" line in
  the log, i.e. the G1 used its own `tracked` table directly.

It does **not** show the G1 training curve reproduces (that needs a G1 library and a real run), and the poor returns
above must not be quoted as a G1 measurement.

---

## B. The X1 motion library

`scripts/build_x1_motions.py` runs GMR's SMPLX → X1 IK on the 41 AMASS CMU walking clips and converts the result to
the allmimic npz contract (`qpos (N,36)`, `qvel (N,35)`, 50 Hz, `starts`/`lengths`/`names`) using the same
`resample` / `velocities` functions as `scripts/prepare_motions.py`. Three calibrations, each measured:

1. **Root height.** GMR's stock `smplx_to_x1.json` has a pelvis scale of 1.2009, calibrated for a ~1.8 m robot, so
   its IK put the X1's root at the *human's* pelvis height (0.93–0.98 m) with the feet 31 cm off the floor.
   `motions/x1_ik_config.json` is that file with `pelvis`/`spine3` scaled by 0.6578 (= X1 root height 0.6275 /
   human pelvis 0.954). Only those two entries change; the arm/leg entries are untouched.
2. **Grounding.** After resampling, the soles are placed on the floor per frame (lowest sole point = 0), which keeps
   the stance foot planted. `--ground per-clip` (one offset for the whole clip) is available for clips with a
   flight phase. The sole plane is 0.0428 m below the `ankle_roll` joint, measured from the model.
3. **Quaternion convention.** GMR's `retarget()` returns MuJoCo `qpos`, so `qpos[3:7]` is **already wxyz**.
   An earlier build reordered it to xyzw (a step that is right for the G1 LAFAN CSVs, whose input is xyzw) and
   rotated every frame. Verified against MuJoCo FK: reading `qpos[3:7]` as wxyz gives 1−R_zz = 8.5e-4 on an upright
   frame, as xyzw 0.065; across the rebuilt library 1−R_zz = 2(x²+y²) matches FK to **6.7e-9** over 167 sampled
   frames.
4. **IK start pose.** GMR's IK integrates from its previous configuration, so a clip's first frames come from a
   different local minimum (32 rad/s joint velocity in frame 0). Two passes plus dropping the first 0.2 s bring the
   library maximum to 14.8 rad/s.

### Validation of the delivered library

Every number below comes from `tools/x1_motion_audit.py`, which re-derives the metrics from the npz alone (MuJoCo FK
on the vendored model, the SMPLX sources for heading) instead of trusting the meta; the two agree clip for clip
because the gait analysis lives in one module (`tools/x1_gait.py`) that both the build and the audit import. That
was not true when this report was first written: it quoted the previous 38-clip build's numbers (yaw ≤ 19°, 30/41
in the cadence band, "every frame" at a limit), which the final 41-clip library does not support.

```
PYTHONPATH=. python tools/x1_motion_audit.py --npz motions/x1_cmu_walk.npz --amass ~/code/X1_29_AMP/data/amass/CMU
```

| metric | value over the 41 clips |
|---|---|
| clips / frames / duration | 41 / 6168 / 2.06 min |
| file | `motions/x1_cmu_walk.npz`, 1.75 MB, sha256 `309a6896…` |
| contract | qpos (6168,36), qvel (6168,35), 50 Hz, starts = cumsum(lengths), quat norm dev 6.0e-8, 0 sign flips, all finite |
| qvel vs finite differences of qpos | max abs error 0.0 (exact; joint max 9.5e-7 in the reviewer's independent check) |
| root height | 0.558 min / 0.618 mean of clip means / 0.633 max |
| 1−R_zz (upright) | mean 0.020, max 0.090 |
| worst joint-limit excess | 0.0906 rad |
| sole penetration | 0.0000 m (grounded per frame) |
| swing-foot clearance | 0.032–0.092 m, median 0.044 m |
| max joint velocity | 14.8 rad/s (08_06) |
| **net yaw change** | max 58.4° (08_06); see the heading paragraph below |
| **both feet resolve a cadence** | 29/41 clips |
| **of those, both feet in 0.8–1.2 Hz** | 8/29 |
| **L/R alternation exactly one opposite step** | 11/29; the other 18 include a cycle with 0 |
| **walk-like clips** (both feet, stride 0.5–1.5 s, alternation mostly 1) | 16/41; their cadence 0.79–1.89 Hz (median 0.94), stride 0.53–1.26 s (median 1.06) |

**Heading: 58.4° is the clips turning, not drift.** 14 of the 41 CMU clips are turns, and the human turns 64–152°
in them. The retarget follows the same direction (correlation 0.767 over the 41 clips) but under-rotates: mean
|net yaw| 45° in the source vs 18° in the X1 data, mean |difference| 35°, worst 125° (07_07, human −152° vs robot
−27°). So the largest single number, 58.4° on 08_06, is the retarget reproducing part of a −94° human turn. It is
recorded as `yaw_net_change_deg` for that reason; an earlier version of this build called the same quantity
`yaw_drift_deg`, which invited exactly the misreading.

**Gait: the clips are short and the swing is low, and that limits what the detector can resolve.** Each clip is
2.1–3.9 s, so a clip holds 2–4 strides; the swing foot lifts only 3–9 cm (median 4.4 cm) because the grounding puts
the stance sole at 0 every frame. With a 1–2 cm hysteresis band, 29 of 41 clips give a cadence for both feet and 12
only for one. Of the 29, 8 have both feet in the 0.8–1.2 Hz band a walk should occupy, and 11 show clean one-step
alternation. Clips that do resolve both feet mostly do so consistently: the 16 walk-like ones sit at median 0.94 Hz
with 1.06 s strides, which is walking. What the table cannot claim is that all 41 clips are cleanly interpretable
walking — they are not, and the per-clip numbers are in the meta.

I did not tune the remaining clips away. Options that were considered and rejected: tightening the reject filter
until only the 16 clean clips survive (throws away 61 % of the data to make a table look better), and switching the
grounding to per-clip so the swing clearance grows (that changes the training data, and the npz that the platform
run already trained on is the one delivered — see the storage note). The honest statement is that the library is
41 clips of retargeted CMU motion in which 16 are unambiguously walk-like and the rest are usable but partly
ambiguous to a foot-height detector.

### Storage accounting

The pipeline writes nothing per clip — no pickle, no per-clip log — so the artefacts under the git-tracked `motions/`
are exactly these three:

| file | size | why it stays |
|---|---|---|
| `motions/x1_cmu_walk.npz` | 1.75 MB | the training library; this is what the platform clones |
| `motions/x1_cmu_walk.meta.json` | 54 KB | per-clip validation numbers + the exact `args` that produced the npz |
| `motions/x1_ik_config.json` | 8.2 KB | the calibrated SMPLX→X1 IK config; without it the npz cannot be reproduced |

`motions/` totals 1.8 MB. Intermediate per-clip pickles and logs were deleted as they were produced; nothing else from
the retarget survives. Workspace free space at the end of B: 145 GB of 228 GB. `_scratch/` (175 MB: an unpacked mjlab
for the CPU smoke plus vendored wheels), `logs/`, `runs/` and `output/` are git-ignored.

### Known limitation: the arms sit on their limits

67.2 % of the library's 6168 frames have at least one joint within 0.02 rad of its limit, and 18 of the 41 clips are
there in 100 % of their frames. Per joint, over all frames: `left_elbow_pitch` 57.8 %, `right_elbow_pitch` 37.1 %,
`left_shoulder_roll` 18.1 %, `left_shoulder_pitch` 17.8 %, `left_knee_pitch` 9.9 %, `right_knee_pitch` 7.1 %;
`left_elbow_yaw` never reaches a limit. (An earlier version of this section stated figures from the previous
38-clip build, which the delivered 41-clip library does not support; `75e245b` replaced them. The superseded
numbers are deliberately not repeated here — `git show abfd527 -- docs/X1_SUPERMIMIC_REPORT.md` has the diff.)

The cause is the robot, not the calibration: replaying one clip through GMR's unmodified `smplx_to_x1.json` gives the
same per-joint fractions, so the pelvis scale is not responsible — the X1's arms are much shorter than a human's and
the IK clamps at the reachable limit. Consequence: a policy trained on this library cannot learn arm motion beyond
the limit; it will hold the arm there, and elbow pitch (57.8 %) is where that happens most. Options for later: filter
the arm entries out of the IK match table, retarget the arms against a separate target, or accept it (the reward does
not penalise being at a limit).

---

## Gradmotion

Project `x1-supermimic` (`PRO_20261102_001`, created under account `rokad42565`). Task creation request:
`scripts/gm/create_x1_supermimic.json` (3000 iterations) and `scripts/gm/create_x1_supermimic_plumbing.json`
(60 iterations, the plumbing probe). Both: `codeType 2`, branch `feat/x1-29dof`, `startScript` begins with
`gm-run`, `hparamsPath` in the repo, `mainCodeUri` = `allmimic/train.py`, 1×4090D.

Reproduce (the account and project ids are the ones used here):

```bash
KEY=$(account-pool get | cut -f2)                     # or: account-pool list | sed ...
export HOME=~/code/X1_29_AMP/.gm-home GM_API_KEY="$KEY"
gm task create --file scripts/gm/create_x1_supermimic_plumbing.json    # -> taskId
gm task run --task-id <taskId>
gm task logs --task-id <taskId> --raw --no-request-log
gm task model list --task-id <taskId>
gm task data keys --task-id <taskId>
gm task data get --task-id <taskId> --data-key <key> --sampling-mode precise
```

### Runs so far

Every run below really was submitted and really failed or ran; each fixed a distinct blocker, quoted with the log
line that identified it. The plumbing task (60 iterations, 1024 envs) exists so that these could be found for
minutes of GPU time instead of inside a 3000-iteration run.

| task | iterations | what happened | fix |
|---|---|---|---|
| `TASK_20261102_017` | — | `gm task run`: **账户余额不足** — the account that owns `PRO_20261102_001` (`rokad42565`) has no balance | moved to `PRO_20261102_002` (`jasevi6851`), which has balance |
| `TASK_20261102_018` | 60 | entry script ran, then pip: every package `Name or service not known` against the image's index `mirrors.ivolces.com` | probe the stack first, and try `pypi.org` before the image's index |
| `TASK_20261102_019` | 60 | `pypi.org` resolves; pip then stops with `ResolutionImpossible` — the IsaacSim image carries an Isaac Sim constraint `numpy<2` against `requirements.txt`'s `numpy==2.4.6` | try the plain Ubuntu 24.04 image, and add `ensure_pip()` |
| `TASK_20261102_020` | 60 | Ubuntu 24.04 image: status running for 13 min with **zero** log output | stopped; back to IsaacSim 5.1, numpy joins `SKIP_PINS` (none of mjlab / mujoco-warp / warp-lang pins numpy) |
| `TASK_20261102_021` | 60 | same numpy `ResolutionImpossible` — the `SKIP_PINS` edit had not actually applied | applied it; the filter now logs the package names it installs |
| `TASK_20261102_024` | 60 | **pip succeeded**: torch 2.7.0+cu128, CUDA 12.8, RTX 4090, mujoco 3.5.0, warp 1.12.0, mjlab 1.2.0, numpy 1.26.4. Trainer died in 0.0 min: `No module named 'torch'` | the trainer subprocess was launched with `PYTHONPATH=ROOT`, *replacing* the image's PYTHONPATH (the kit python's site-packages is reached through it); now appended |
| `TASK_20261102_026` | 60 | stack + CUDA fine, then `RuntimeError: Minimum supported CUDA version: 12.4.` from `mujoco_warp.put_model` → `warp.is_conditional_graph_supported()` | print `nvidia-smi`'s driver and that predicate before training |
| `TASK_20261102_027` | 60 | measured: driver **535.183.01** (warp prints "CUDA Toolkit 12.9, Driver 12.2"), `cond graph supported: False` | `scripts/gm_compat.py`: a logged no-op guard, since mjlab already disables CUDA graphs below driver 12.4 and mujoco-warp 3.5.0 never uses conditional graphs |
| `TASK_20261102_029`, `_037` | 60 | shim still not applied: the switch compared `nvidia-smi`'s driver number against a CUDA *version* threshold, and then a rewrite of that switch silently did not land | the switch now comes from the predicate itself (`cond graph supported:`), asserted against the platform's own probe text |
| `TASK_20261102_040` | 60 | **simulator built, warp kernels compiled**, then `KeyError: 'lumbar_yaw_joint'` in `_wm_error_weights` | `--preset recommended` hardcodes the G1 table `tracked` and nothing read `Robot.wm_joint_weights`; the config now falls back to the robot's own table and names the missing joints if that fails too |
| `TASK_20261102_041` | 60 | **ran to completion, exit 0**: `it 5 … it 60`, `ret 0.21`, `wm1 0.94 → 0.63`, three checkpoints in `gm task model list`, `EVAL it 60 … failed 41/41`. `gm task data keys` was empty | metric mirror raised `OSError: telling position disabled by next() call` (iterating a text file + `f.tell()`); now binary with an explicit byte cursor |

### The run that works

`TASK_20261102_043` — 3000 iterations, 4096 envs, `configs/x1_supermimic.json`, 1×4090D, branch `feat/x1-29dof`.

Log excerpt (`gm task logs --task-id TASK_20261102_043 --raw`), environment build → iteration progress:

```
[gm-entry] repo /workspace/isaaclab/super_mimic_F1
[gm-entry] stack check failed: ModuleNotFoundError: No module named 'mujoco'      <- image lacks the stack, so pip
[gm-entry] filtered requirements -> runs/requirements.gm.txt: ['mjlab', 'mujoco', 'mujoco-warp', 'warp-lang',
           'rsl-rl-lib', 'tensordict', 'scipy', 'trimesh', 'prettytable', 'tyro', 'mediapy']
           (left to the image: torch, pytest, matplotlib, numpy)
[gm-entry] pip against https://pypi.org/simple
Successfully installed ... mjlab-1.2.0 ... mujoco-3.5.0 ... mujoco-warp-3.5.0 ... warp-lang-1.12.0 ...
torch 2.7.0+cu128 cuda 12.8 available True count 1
mujoco 3.5.0 warp 1.12.0 mjlab .../site-packages/mjlab/__init__.py numpy 1.26.4
gpu NVIDIA GeForce RTX 4090
driver: 535.183.01
cond graph supported: False
[gm-entry] driver 535.183.01 conditional graphs -> bypassing mujoco_warp's >=12.4 gate (gm_compat)
[gm-compat] patched mujoco_warp._src.warp_util.check_toolkit_driver
[gm-compat] driver (12, 2): mujoco_warp 3.5.0 refuses to run below driver 12.4 because of conditional CUDA graphs;
           the gate is bypassed because mjlab disables CUDA graphs on this driver ...
[WARNING] CUDA Graphs disabled: driver 12.2 < 12.4          <- mjlab's own handling of the same limit
it 50 upd 20 h 0.062 iter 0.547s ret 0.205 len 30.6 term 0.814 wm1 0.7050 wm8 4.9014 J 0.399 surv 0.141
it 75 upd 30 h 0.066 iter 0.539s ret 0.225 len 29.7 term 0.807 wm1 0.6047 wm8 3.8242 J 0.529 surv 0.165
it 125 upd 50 h 0.074 iter 0.540s ret 0.234 len 30.2 term 0.809 wm1 0.5362 wm8 2.8624 J 0.551 surv 0.177
it 175 upd 70 h 0.082 iter 0.623s ret 0.244 len 32.7 term 0.802 wm1 0.4910 wm8 2.5823 J 0.620 surv 0.230
it 225 upd 90 h 0.090 iter 0.540s ret 0.268 len 36.6 term 0.767 wm1 0.4365 wm8 2.0606 J 0.689 surv 0.296
```

`gm task data get` (the platform's chart API, so the metric channel is proven, not just the console log):

| key | points | first | last |
|---|---|---|---|
| `Train/ep_return` | 9 | it 25 → 0.8535 | it 225 → 1.5628 |
| `Train/ep_norm_return` | 9 | it 25 → 0.3125 | it 225 → 0.2684 |
| `Train/wm_err1` | 9 | it 25 → 1.0748 | it 225 → 0.4365 |

33 keys in total (`gm task data keys`: `Train/wm_loss`, `Train/pi_J`, `Train/img_survival`, the per-term imagined
rewards, ...).

Checkpoints, `gm task model list`:

| checkpoint | policUrl |
|---|---|
| `0000060` | `upload/2026/11/2/ckpt_0000060_20261102113441A712.pt` |
| `0000030` | `upload/2026/11/2/ckpt_0000030_20261102113419A751.pt` |
| (last) | `upload/2026/11/2/last_20261102113435A797.pt` |

Those three are from the plumbing run `TASK_20261102_041` (60 iterations, exit 0, published
`output/x1_supermimic_iter000060.pt`). The 3000-iteration run's own checkpoints:

| checkpoint | policUrl |
|---|---|
| `003000` (published copy of `ckpt_0003000`) | `upload/2026/11/2/x1_supermimic_iter003000_20261102121634A086.pt` |
| `0003000` | `upload/2026/11/2/ckpt_0003000_20261102121554A350.pt` |
| `0002000` | `upload/2026/11/2/ckpt_0002000_20261102120616A040.pt` |
| `0001000` | `upload/2026/11/2/ckpt_0001000_20261102115644A554.pt` |
| (last) | `upload/2026/11/2/last_20261102121548A488.pt` |

### Final result of the 3000-iteration run

Finished normally (`status 5`), 3000 iterations at 0.55 s/iter ≈ 33 min of training, single 4090D. From
`gm task data get` (120 sampled points per key, `--sampling-mode accelerate`):

| key | it 25 | it 3000 |
|---|---|---|
| `Train/ep_return` | 0.8535 | **3.8904** |
| `Train/ep_norm_return` | 0.3125 | **0.7931** |
| `Train/wm_err1` (1-step world-model error) | 1.0748 | **0.0259** |
| `Train/wm_errk` (k-step) | 12.8419 | **0.0657** |
| `Train/img_survival` (imagined-rollout survival) | 0.1634 | **0.9993** |

Console equivalents at the end: `it 3000 upd 1200 h 0.531 iter 0.552s ret 0.793 len 46.1 term 0.000 wm1 0.0259
wm8 0.0657 J 3.009 surv 0.999`. Episodes no longer terminate (`term` 0.000) and the imagined rollouts survive
(0.999), i.e. the world model and the policy are both working on this robot's data.

**What this does and does not show.** It shows the whole X1 path runs on the platform: the vendored asset loads in
mjlab, MuJoCo Warp builds the scene, the 41-clip library is read, PPO + world model train, checkpoints and metrics
are published. It does **not** show a policy that walks: 3000 iterations is ~2 % of the G1 reference run, and the
in-run evaluation of the 60-iteration plumbing run reported `failed 41/41` (that run evaluated at iteration 60,
where nothing is expected to succeed). No evaluation of `ckpt_0003000` has been run on the clips; that is the next
step, not a result.

### Environment as measured on the platform

From `TASK_20261102_024`'s log, the IsaacSim 5.1 image (BJX00000178 / V000446) provides:

```
torch 2.7.0+cu128 cuda 12.8 available True count 1
mujoco 3.5.0 warp 1.12.0 mjlab 1.2.0 numpy 1.26.4
gpu NVIDIA GeForce RTX 4090
```

Note `numpy 1.26.4`, not the repo's 2.4.6 — that is the image's numpy, kept because the image's own constraint
forbids numpy 2 (see run 019). This is a deviation from `requirements.txt` and is recorded here rather than hidden.

---

## Commits and pushes

The branch carries a linear series of commits after the upstream `cb334ff`. The listing below is a snapshot; the
authoritative check is the command in its header (`git ls-remote origin feat/x1-29dof` for the live head), and a
later docs commit naturally adds one more line than the snapshot shows.

Nothing was force-pushed and no remote ref was rewritten. The network to github.com:443 dropped repeatedly (TCP
timeouts, `Empty reply from server`, HTTP/2 framing errors), so several pushes were retried — each successful one
printed the old..new range, and the remote ref was re-read after every ambiguous attempt rather than retried blind.

```
$ git log --oneline cb334ff..feat/x1-29dof
087d76b docs: G1 path has runtime evidence now (path only, labelled)
2c483e9 docs: fix the report's two stale claims, add the 0.657 m question and the commit record
62a7d34 Gradmotion: G1 path-check config and create JSON (runtime proof the upstream robot still trains)
75e245b X1 library: correct the report numbers, one gait detector, an independent audit
ed62e3f docs: final results of the 3000-iteration X1 run
340d67c docs: record the Gradmotion run that works (env build, iter progress, metrics, checkpoints)
ed0b140 Gradmotion: 3000-iteration create JSON on the project with balance
bc08ebe Gradmotion entry: fix the metric mirror, it mirrored nothing
b305b38 X1: pick the robot's own WM joint weights instead of KeyError
7e06019 Gradmotion: really switch the shim on warp's predicate (2ba1c96 did not)
2b8e899 chore: build_x1_motions.py executable, like the other entry scripts
2ba1c96 Gradmotion: the shim's switch comes from the predicate it bypasses
f2dcc20 Gradmotion entry: capture the stack probe's output so the driver version reaches the decision
f716bb1 Gradmotion: bypass mujoco_warp's driver gate, the way mjlab already does
1188da6 docs: X1 supermimic report (asset, motion library, platform bring-up, limitations)
1845dd1 Gradmotion entry: print the driver version and warp's conditional-graph support before training
12743c8 Gradmotion entry: append to PYTHONPATH instead of replacing it
0cd59e3 Gradmotion entry: actually skip the numpy pin (the previous commit only said so)
87c87de Gradmotion: back to IsaacSim 5.1, leave numpy to the image
40d38c8 Gradmotion: Ubuntu 24.04 image, and install pip if the image ships a bare Python
7ed9405 Gradmotion entry: the image's pip index does not resolve, so probe first
986bd0f Gradmotion: 60-iteration plumbing hparams to validate the whole path cheaply
11cbb3e README: X1 29-DoF section (asset, motion library, platform entry, three gotchas)
694ae59 X1 docs/wording: one story in the smoke docstring, executable entry script
63c3b39 Gradmotion create JSON: point at the x1-supermimic project (PRO_20261102_001)
0481bd1 X1 motion library: fix the root quaternion, record what produced the npz
85a7c72 X1 stability: fix the config layer, keep the asset byte-identical
fb422ff Gradmotion entry: keep the image's CUDA torch, drop the compile warm-up
535ffb1 Gradmotion: X1 allmimic task entry, hparams, reusable create JSON
d4d486d X1 motion library: GMR retarget of 41 CMU walking clips at 50 Hz
c51106e X1 29-DoF: vendored asset, second robot in the registry, CPU smoke test
```

Two commit messages are now historical and their subjects no longer describe the final tree (`c51106e` and
`d4d486d` predate the revert of the sole box and the quaternion fix). They cannot be amended without rewriting pushed
history, which the task forbids; the corrections are in `85a7c72` and `0481bd1`, whose messages say exactly what
they reverse.

## Known limitations and open items

1. **The GPU half now runs, but the policy is not evaluated.** `TASK_20261102_043` completed 3000 iterations
   (status 5), published five checkpoints and 33 metric keys; nothing has replayed `ckpt_0003000` on the library, so
   there is no success rate for the X1 yet. Do not read the training curves as "the X1 walks".
2. **Data is CMU walking only** (41 clips, 2.06 min, one performer family, no running/jumping/turning-in-place).
   It is enough to prove the pipeline; it is not a paper-scale library (BONES-SEED used for the G1 is 306 clips and
   ~6.6 h of training to convergence on 2 GPUs).
3. **Arm limits**: 67.2 % of the library's frames have a joint within 0.02 rad of a limit (18 of the 41 clips
   in all of their frames); per joint `left_elbow_pitch` 57.8 %, `right_elbow_pitch` 37.1 %,
   `left_shoulder_roll` 18.1 %, `left_elbow_yaw` 0 % (see B for the full table and the cause).
4. **Training length**: the 3000-iteration config is a smoke-length run for one 4090D, not a converged policy. The
   G1 reference took 176k iterations / 6.6 h on 2×4090 with `--preset recommended`; the X1 config additionally
   disables `compile_bptt`/`compile_wm` and sets `bptt_checkpoint 1` to fit and start quickly on one 24 GB card.
5. **The G1 has no runtime *quality* evidence**, only a path test (`TASK_20261102_067`: it trains, checkpoints,
   metrics; the data fed to it was X1 motion, so its returns are meaningless). A real G1 regression needs a G1
   library and is out of scope here.
6. **`--preset recommended` was tuned on the G1**, including the joint error-propagation weights (`wm_joint_weights:
   tracked` → the X1 uses `tracked_x1`, a name-mapped copy of the G1 lever arms, not an X1 torque identification).
6. **Foot contact is four point contacts per foot**, unchanged from the source asset. It is stable at the real gains
   (A), but a flat sole would model a real foot better; that is a modelling choice, not a fix for a measured defect.

## Next steps

1. **Evaluate `ckpt_0003000` on the 41 clips** (`tools/eval_allmimic.py`, or `--eval-every` on a resumed run) to get
   the only number that matters for "does it walk": the clip success rate. Until then this report claims a working
   pipeline, not a working policy.
2. **Run to a useful length.** 3000 iterations is ~2 % of the G1 reference run (176k). A first real target is on the
   order of 50k-100k iterations (≈9-18 h at the measured 0.55 s/iter on one 4090D), then re-evaluate.
3. **Widen the data**: BONES-SEED or LAFAN1 through the same `scripts/build_x1_motions.py` (41 CMU walking clips,
   2.06 min, is a pipeline proof, not a training set), and decide the arm question (filter the arm entries out of the
   IK match table, retarget arms separately, or accept the saturation).
4. **Restore the preset's compile flags** (`compile_bptt`/`compile_wm`) once the pipeline is trusted, and measure the
   per-iteration cost with and without them on one 4090D; the current config disables them to start in seconds.
5. **Driver gate**: the host's 535.183.01 driver works only because `scripts/gm_compat.py` bypasses mujoco_warp's
   `>= 12.4` check (mjlab disables CUDA graphs on that driver by design). Either keep the shim, pin a mujoco-warp
   version without the gate, or ask for a host with a >= 12.4 driver — the shim is the one piece of this setup that
   deliberately departs from the library's own precondition.
