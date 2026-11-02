# allmimic: a SuperMimic re-implementation

<p align="center"><img src="assets/challenge_cases.gif" width="560" alt="flips, side rolls, broad jumps and other hard BONES-SEED motions"></p>

An unofficial re-implementation of **SuperMimic** (Liao et al., *SuperMimic: Scaling Humanoid Whole-Body Control
with One Differentiable World Model*, arXiv 2610.28480) for the Unitree G1 in
[mjlab](https://github.com/mujocolab/mjlab) / MuJoCo Warp.

SuperMimic fits one MLP world model to replayed simulator transitions and trains a motion-tracking policy by
backpropagating through 32 imagined steps of that frozen model, with a small critic bootstrapping the tail of each
imagined rollout. The authors did not release code. This code started from
[dhmbb2/supermimic_vibe](https://github.com/dhmbb2/supermimic_vibe), a re-implementation of the paper.

**Status: fully reproduced.** With the final defaults, all 306 hard BONES-SEED motions of the paper's failure table (flips, side rolls, broad jumps, crossed legs, lying down and standing up) play without a fall after 6.6 h of training on 2 RTX 4090 GPUs; the paper reports the same after 9.8 h on 2 H100. Details in [Results](#results).

## What is different

Compared with the paper and the original implementation:

- **Residual actions for the upper body, absolute actions for the legs.** The arms and torso track better as offsets
  from the reference, while leg targets tied to the reference fight balance.
- **A wider policy latent** (128 instead of 32), so the bottleneck between motion encoder and decoder does not limit
  tracking.
- **A share of the data is collected without exploration noise**, so the world model also sees the states a
  deterministic, deployed policy reaches. Otherwise the policy can exploit model errors there.
- **Per-joint world-model loss weights**, following D. Holden's
  ["Joint Error Propagation"](https://theorangeduck.com/page/joint-error-propagation): joints are weighted by how far
  their errors propagate to the tracked bodies.
- **A capped reset sampler and a policy gradient clip that takes effect.** Hard-example mining cannot pile onto a few
  unlearnable clips. The original clip never acted because the loss is divided by the horizon.
- **Balanced reset sampling.** The 1 s reset bins of the training data are clustered by reference kinematics and
  rare clusters start episodes more often: each cluster's share grows with size^0.3 instead of its size. Flips are
  0.04% of the BONES-SEED bins; without this, no flip is learned from the full training split.
- **A value clip.** Critic values used as targets and as the policy's bootstrap are clamped to the reachable range,
  ±(maximum step reward) / (1 − γ). Without it the critic eventually overestimates a few states, the policy follows
  the bootstrap there, and long runs diverge after 5 to 7 h. With it the critic loss stays flat.
- **Infeasible-clip filtering for BONES-SEED.** References a G1 cannot perform on a flat, empty floor, such as
  sitting on an invisible chair or leaning on a wall, are left out of training.
- **Compiled training steps with CUDA graphs** for the imagined rollout and the world-model loss, for faster
  training with unchanged gradients.

`--preset recommended` turns all of this on, except two data files: the infeasible-clip list and the bin clusters
for balanced sampling. Without the preset, every default is the original behavior. [CHANGES.md](CHANGES.md) lists every change, with its tests.

## Install

You need Linux, Python 3.11 and a CUDA GPU. We used RTX 4090 cards with CUDA 12.6.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt   # mjlab 1.2.0, mujoco 3.5.0, mujoco-warp 3.5.0, warp-lang 1.12.0, ...
export PYTHONPATH=$PWD            # all commands run from the repository root
pytest tests                      # CPU tests (no GPU needed); GPU checks are in tests/gpu/
```

## Data

No motion data is included. Check each dataset's license.

**LAFAN1.** Get the 40 G1 clips (`g1/*.csv`) of
[`lvhaidong/LAFAN1_Retargeting_Dataset`](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset):

```bash
python scripts/prepare_motions.py --format lafan --fps-in 30 \
    --input data/raw/lafan1_g1/g1 --output data/processed/lafan1_50hz.npz
mkdir -p data/library_eval && cp data/processed/lafan1_50hz.npz data/library_eval/lafan1.npz
python tools/make_segments.py --lib data/library_eval/lafan1.npz --seconds 10   # 10 s segments for evaluation
```

**BONES-SEED.** Get `g1.tar.gz` and `seed_metadata_v004.csv` from the gated dataset
[`bones-studio/seed`](https://huggingface.co/datasets/bones-studio/seed):

```bash
mkdir -p data/raw/bones_seed && tar -xzf g1.tar.gz -C data/raw/bones_seed
CSV=data/raw/bones_seed/g1/csv
# the paper's Appendix-E filter (GPU; the full library needs ~40 GB of RAM)
python scripts/prepare_motions.py --format bones --fps-in 120 --workers 32 --input $CSV --output data/processed/bones_all.npz
python scripts/filter_bones.py --input data/processed/bones_all.npz --output data/processed/bones_filtered.npz \
    --report data/processed/bones_filter.json
# held-out splits: train, test_content (unseen motions), test_actor (unseen performers), test_random
python tools/make_bones_splits.py --metadata data/raw/bones_seed/seed_metadata_v004.csv \
    --kept data/processed/bones_filtered.npz --out-dir data/splits/bones_seed
# training library and infeasible-clip filter -> data/infeasible/bones_train_feasible.txt
python tools/build_allmimic_library.py --csv-dir $CSV --format bones --fps-in 120 \
    --clip-list data/splits/bones_seed/train.txt --out data/library/bones_train.npz
python tools/filter_infeasible.py features --lib data/library/bones_train.npz --out data/feats/bones_train.npz
python tools/filter_infeasible.py flag --feats data/feats/bones_train.npz --dataset bones_train
# bin clusters for balanced reset sampling (CPU) -> data/bin_clusters/bones_train.npz
python tools/bin_clusters.py --lib data/library/bones_train.npz --workers 16 --out data/bin_clusters/bones_train
# test libraries (full clips and 10 s segments), e.g. for test_content
python tools/build_allmimic_library.py --csv-dir $CSV --format bones --fps-in 120 \
    --clip-list data/splits/bones_seed/test_content.txt --out data/library_eval/bones_test_content.npz
python tools/make_segments.py --lib data/library_eval/bones_test_content.npz --seconds 10
```

**Hard motions.** [`lists/bones_hard.txt`](lists/bones_hard.txt) holds 306 BONES-SEED training clips from the hard
motion groups of the paper's failure table: flips (54), side rolls (22), broad jumps (24), crossed legs (10) and
lying / resting / standing up (196). [`lists/bones_hard.csv`](lists/bones_hard.csv) gives each clip's group.

```bash
python tools/build_allmimic_library.py --csv-dir $CSV --format bones --fps-in 120 \
    --clip-list lists/bones_hard.txt --out data/library_eval/bones_hard.npz
```

Other G1 retargets in the LAFAN1 CSV layout can be built as extra test sets with
`tools/build_allmimic_library.py --format lafan`.

## Train

```bash
# LAFAN1, one GPU
python -m allmimic.train --preset recommended --motions data/processed/lafan1_50hz.npz --out runs/lafan1

# BONES-SEED, 2 GPUs on one node: the reported run (batch sizes are per GPU; the learning rate scales with sqrt(GPUs))
torchrun --standalone --nproc_per_node=2 -m allmimic.train --preset recommended --motions data/library/bones_train.npz \
    --clip-list data/infeasible/bones_train_feasible.txt --bin-prior data/bin_clusters/bones_train.npz \
    --pi-batch 2048 --eval-every 0 --out runs/bones

# hard motions only, one GPU. In-run evaluation on all 306 clips every 4k iterations -> eval.jsonl
python -m allmimic.train --preset recommended --motions data/library_eval/bones_hard.npz --eval-every 4000 \
    --eval-clips 0 --out runs/bones_hard
```

A run writes `log.jsonl`, checkpoints (`ckpt_*.pt`, `last.pt`; resume with `--resume`) and, on LAFAN1, `eval.jsonl`
from the in-run evaluation. The CUDA-graph compilation was validated on RTX 4090. On other GPUs, run
`python tests/gpu/test_bptt_compile.py` first (its docstring has the command).

## Evaluate

Evaluation follows the paper's protocol: every clip plays once from its first frame, without randomization or
noise, with a deterministic policy. A clip fails on termination. The return is normalized by its maximum.

```bash
python -m allmimic.evaluate --run runs/lafan1                       # every checkpoint of a run, on LAFAN1
python tools/eval_allmimic.py --ckpt runs/bones/last.pt --out-dir runs/bones/eval \
    --testsets lafan1 lafan1_seg10 bones_test_content bones_test_content_seg10   # libraries in data/library_eval/
python tools/failure_report.py runs/bones/eval --testsets lafan1_seg10 bones_test_content_seg10
```

## X1 29-DoF (second robot)

The upstream robot is the Unitree G1 and stays the default. The Zhiyuan/roboparty X1 29-DoF humanoid is a
second, switchable entry in `allmimic/robots` (`--robot x1_29dof`); it has the same 29 joints, so every
network shape, `OBS_DIM` and the task/reward code are shared, and what differs (kinematic tree, joint order,
limits, actuators, the 14 reward bodies, the joint-weight table) comes from `allmimic/robots/x1_29dof.py`.
`--robot g1` and every command above are unchanged.

```bash
# model + contact + servo checks and a 4 s zero-action stance, on CPU (no GPU, no mjlab install needed for mjlab
# itself: only its CollisionCfg / create_position_actuator are imported, from a checkout or the wheel on PYTHONPATH)
MJLAB_SRC=/path/to/unpacked/mjlab PYTHONPATH=. python scripts/x1_smoke.py --steps 200

# the CMU walking library that ships with this branch (41 clips, 50 Hz, 1.75 MB)
python -m allmimic.train --preset recommended --robot x1_29dof \
    --motions motions/x1_cmu_walk.npz --out runs/x1_cmu_walk
```

Measured on the platform (1x4090D, 3000 iterations, 0.55 s/iter): the run completes, publishes checkpoints and
metrics, and evaluates. An earlier report claimed "0 of 41 clips" for the X1 from that run — that number was produced
with the **G1** eval env (`make_eval_env` defaulted to it; fixed in `af76c5c`, with
`tests/test_eval_robot.py` guarding the call sites), so it is not an X1 result. The X1-env measurement is in
`docs/X1_SUPERMIMIC_REPORT.md`; 3000 iterations is ~2 % of the G1 reference run either way.

| item | where | notes |
|---|---|---|
| asset | `assets/x1_29dof/` | vendored MJCF + the 30 STLs it references; `tools/vendor_x1_assets.py` regenerates it and `--check` verifies it. `PROVENANCE.md` has the sha256s |
| joint table | `docs/x1_29dof_joints.md` | per-joint limits, kp/kd/effort/armature, masses, foot contact geoms; `tools/x1_joint_table.py` regenerates it |
| motion library | `motions/x1_cmu_walk.npz` | GMR SMPLX→X1 retarget of the 41 CMU walking clips; `scripts/build_x1_motions.py` rebuilds it, `motions/x1_cmu_walk.meta.json` has the per-clip validation numbers |
| library check | `tools/x1_motion_audit.py`, `tools/x1_gait.py` | re-derives every number from the npz (contract, velocities, FK geometry, gait, heading vs the SMPLX source) instead of trusting the meta; exits non-zero on a hard failure |
| platform run | `scripts/gm_x1_train.py`, `configs/x1_supermimic.json` | Gradmotion entry: pip-installs `requirements.txt`, verifies CUDA/mjlab, trains, mirrors `log.jsonl` to TensorBoard and publishes the checkpoint |

Three things about this robot are worth knowing before reading the numbers:

* **The asset's `<option>` block and contact defaults target its own 1 kHz scene.** allmimic steps physics at
  200 Hz, where the asset's 5 ms `solref` is unstable, so the entity's `CollisionCfg` gives the sole geoms
  `solref 0.02 1`, `condim 3`, `priority 1` and friction 0.6. `scripts/x1_smoke.py` asserts those on the
  *compiled* model rather than assuming the config landed.
* **Armature has to be written at spec level, before compile** (mjlab's `create_position_actuator` does).
  Setting it on a compiled model leaves `dof_M0` stale and the robot topples in ~2 s with no NaN to warn you.
* **The X1's arms are much shorter than a human's**, so GMR's IK saturates the arm limits: 67.2% of the library's
  frames have a joint within 0.02 rad of a limit (18 of the 41 clips are there in 100% of their frames), and over
  all frames `left_elbow_pitch` is at a limit 57.8% of the time, `right_elbow_pitch` 37.1%, `left_shoulder_roll`
  18.1%; `left_elbow_yaw` never is. That is GMR's stock `smplx_to_x1.json` too, not a side effect of the local
  calibration. The reward does not punish a joint at its limit, but a policy trained on this library cannot learn
  an arm trajectory beyond it.

What the shipped library is, stated plainly, because the numbers are easy to over-read:

* **41 clips, 6168 frames, 2.06 min, 50 Hz.** Root height 0.558-0.633 m, sole penetration 0, joint-limit excess
  at most 0.091 rad, max joint velocity 14.8 rad/s, all values finite, `qvel` exactly the finite difference of
  `qpos`. 29 of 41 clips give a cadence for both feet; of those 8 have both feet in the 0.8-1.2 Hz band a walk
  should occupy, and 16 clips are unambiguously walk-like (median 0.94 Hz, 1.06 s strides).
* **The clips are short and the swing foot lifts only 3-9 cm** (because the grounding puts the stance sole at 0
  every frame), so a foot-height detector cannot resolve a cadence for every clip: 12 of the 41 give one foot only,
  and 18 of the 29 two-foot clips include a cycle where the feet do not alternate cleanly. That is a property of
  2-4 s retargeted clips, not a tuning knob; the per-clip numbers are in `motions/x1_cmu_walk.meta.json` and can be
  re-derived with `tools/x1_motion_audit.py`.
* **The largest single heading number, 58.4 deg on 08_06, is a turn, not drift.** 14 of the 41 CMU clips are turns
  in which the human rotates 64-152 deg; the retarget follows the direction (correlation 0.767) but under-rotates
  (mean |net yaw| 45 deg in the source vs 18 deg here).

Whether those are acceptable is a modelling decision this branch does not make for you: filtering to the 16 clean
clips would discard 61% of the data to make a table look better, and re-grounding per clip would change the data the
platform run already trained on. If you need a clean gait library, rebuild with `--reject-*` tightened and a longer
source clip set (see the report's next steps).

`docs/X1_SUPERMIMIC_REPORT.md` has the full record: what was measured, what failed on the platform, and the
open items.

## Results

Training time is wall-clock hours of training on one RTX 4090 with `--preset recommended`, excluding evaluation.
Evaluation follows the protocol above.

| benchmark | result |
|---|---|
| LAFAN1 (40 clips, full length) | return 0.85 after 0.42 h; 0/40 failures after 0.83 h (return 0.91); return 0.94 after 2.0 h (0/40 failures) |
| BONES-SEED hard motions (306 clips, [list](lists/bones_hard.csv)), trained on these clips only | 0/306 failures after 0.68 h (return 0.88), flips 54/54, side rolls 22/22; return 0.91 after 1.5 h (0/306 failures) |
| BONES-SEED (full training split, 2 GPUs) | all 306 hard motions after 6.6 h (176k iterations); 303/308 flips, cartwheels and handstands; held-out BONES-SEED motions 100.0% / 0.884, LAFAN1 94.6% / 0.806 ([details](#bones-seed)) |

### BONES-SEED

**Fully reproduced.** Trained on the full BONES-SEED training split with `--preset recommended` on 2 RTX 4090 GPUs, the policy plays all 306 hard motions of the paper's failure table without a fall after 6.6 h of training from scratch (176k iterations), together with 303 of the 308 flips, cartwheels and handstands of the training split, while generalizing to held-out BONES-SEED motions (100.0% success) and to LAFAN1, AMASS and MotionDecode, which it never saw. The paper reports no failures on its hard groups after 9.8 h on 2 H100.

Test sets: LAFAN1, AMASS and MotionDecode (other datasets) and the held-out BONES-SEED splits, cut into 10 s
segments, without clips a G1 cannot perform on an empty floor. The right panel and the last four columns count
training clips played in full without a fall: the hard-motion list above and all flips, cartwheels and
handstands of the training split.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/bones_seed_curve_dark.png">
  <img src="assets/bones_seed_curve.png" width="100%" alt="BONES-SEED training curves: test-set return and success rate, and hard training motions solved, over training hours">
</picture>

Cells: success rate / mean return on the test sets, clips without a fall on the training motions.

| training (h) | iterations | LAFAN1 | AMASS | MotionDecode | BONES unseen motions | BONES unseen performers | hard motions | flips | cartwheels | handstands |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.6 | 16k | 93.2% / 0.786 | 96.0% / 0.817 | 99.3% / 0.887 | 99.8% / 0.916 | 99.8% / 0.937 | 218/306 | 0/54 | 6/108 | 0/146 |
| 1.2 | 32k | 92.7% / 0.801 | 97.3% / 0.833 | 99.4% / 0.906 | 99.8% / 0.923 | 99.9% / 0.949 | 244/306 | 0/54 | 31/108 | 0/146 |
| 1.8 | 48k | 93.1% / 0.805 | 97.3% / 0.821 | 99.5% / 0.901 | 99.8% / 0.924 | 99.9% / 0.941 | 249/306 | 0/54 | 101/108 | 0/146 |
| 2.4 | 64k | 95.1% / 0.846 | 97.7% / 0.845 | 99.5% / 0.901 | 99.8% / 0.929 | 99.9% / 0.943 | 251/306 | 0/54 | 100/108 | 0/146 |
| 3.0 | 80k | 94.4% / 0.828 | 97.5% / 0.838 | 99.5% / 0.891 | 99.8% / 0.921 | 100.0% / 0.937 | 245/306 | 0/54 | 104/108 | 0/146 |
| 3.6 | 96k | 95.1% / 0.815 | 97.3% / 0.820 | 99.5% / 0.879 | 99.8% / 0.913 | 100.0% / 0.922 | 258/306 | 7/54 | 105/108 | 0/146 |
| 4.2 | 112k | 94.8% / 0.781 | 97.7% / 0.792 | 99.4% / 0.849 | 99.8% / 0.884 | 100.0% / 0.904 | 260/306 | 14/54 | 102/108 | 0/146 |
| 4.8 | 128k | 94.1% / 0.821 | 97.6% / 0.826 | 99.5% / 0.886 | 99.8% / 0.902 | 99.9% / 0.920 | 268/306 | 18/54 | 105/108 | 0/146 |
| 5.4 | 144k | 94.8% / 0.804 | 98.1% / 0.821 | 99.6% / 0.862 | 99.9% / 0.846 | 100.0% / 0.883 | 298/306 | 51/54 | 107/108 | 79/146 |
| 6.0 | 160k | 93.3% / 0.755 | 96.7% / 0.795 | 99.1% / 0.860 | 99.9% / 0.830 | 99.5% / 0.890 | 303/306 | 53/54 | 108/108 | 120/146 |
| 6.6 | 176k | 94.6% / 0.806 | 97.5% / 0.823 | 99.6% / 0.886 | 100.0% / 0.884 | 100.0% / 0.919 | 306/306 | 54/54 | 108/108 | 141/146 |

Without the value clip, the same configuration diverged at 6.6 h (critic loss above 1e10); with it the critic
loss stays below 0.3. Without balanced sampling, no flip succeeded within 4.5 h.

## Citation

Please cite the SuperMimic paper:

```bibtex
@article{liao2026supermimic,
  title   = {SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model},
  author  = {Liao, Qiayuan and Shao, Yiyang and Long, Junfeng and Zakka, Kevin and Horowitz, Roberto and
             Abbeel, Pieter and Sreenath, Koushil},
  journal = {arXiv preprint arXiv:2610.28480},
  year    = {2026}
}
```

## Acknowledgements

- The method is by the SuperMimic authors.
- The base implementation is [dhmbb2/supermimic_vibe](https://github.com/dhmbb2/supermimic_vibe).
- The simulator is [mjlab](https://github.com/mujocolab/mjlab) on MuJoCo Warp, with the BeyondMimic tracking setup.
- The residual actions follow SuperTrack (Fussell et al., 2021).
- The world-model joint weights follow D. Holden's "Joint Error Propagation".
- Ideas taken from other open re-implementations are credited in [CHANGES.md](CHANGES.md).

## License

Not decided yet; see [LICENSE](LICENSE). The original implementation has no license file, so its terms must be
clarified first. The datasets are not included and keep their own licenses.
