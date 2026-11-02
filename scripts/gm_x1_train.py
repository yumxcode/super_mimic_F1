#!/usr/bin/env python3
"""Gradmotion entry point for the X1 allmimic run: build the environment, then train (task C).

The platform starts a task with `startScript = gm-run super_mimic_F1/scripts/gm_x1_train.py` inside a CUDA image that
has Python 3.11 but not this project's stack, so this wrapper does what a shell entry would do, in four steps:

  1. `pip install -r requirements.txt` (+ tensorboardX for the metric channel) and print the resolved versions;
  2. verify that mjlab / mujoco-warp / warp / torch import and that CUDA is visible;
  3. launch `python -m allmimic.train` with the flags from configs/x1_supermimic.json (`--hparams`), streaming its
     stdout to the task log (environment build, iter progress, return);
  4. while it runs, tail `<out>/log.jsonl` and republish every scalar to TensorBoard events under
     `logs/x1_supermimic/<timestamp>/`, which is the directory the platform's chart API reads (`gm task data get`);
     after it exits, copy the newest checkpoint to `output/x1_supermimic_iter<NNNN>.pt` (top level of output/, the
     channel `gm task model list` scans).

Only the standard library is imported before the pip step; everything else is imported lazily afterwards.

  gm-run super_mimic_F1/scripts/gm_x1_train.py                       # the task's startScript
  python scripts/gm_x1_train.py --hparams configs/x1_supermimic.json --dry-run   # print the plan, run nothing
  python scripts/gm_x1_train.py --skip-install                       # local rerun with the stack already installed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / 'requirements.txt'
EXTRA_PIP = ('tensorboardX',)  # metric channel only; not needed to train


def log(msg: str) -> None:
    print(f'[gm-entry] {msg}', flush=True)


def run(cmd, cwd=ROOT, check=True, **kw):
    log('$ ' + ' '.join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd), check=check, **kw)


def pip_install() -> None:
    base = [sys.executable, '-m', 'pip', 'install', '--no-input', '--disable-pip-version-check']
    # torch is pinned to a CUDA 12.6 wheel in requirements.txt's comment; PyPI's default wheel is used first and
    # the explicit index is only a fallback, so a mirror-configured image still works.
    r = run([*base, '-r', REQUIREMENTS, *EXTRA_PIP], check=False)
    if r.returncode != 0:
        log(f'pip install -r requirements.txt failed ({r.returncode}); retrying torch from the cu126 index')
        run([*base, 'torch==2.14.1', '--index-url', 'https://download.pytorch.org/whl/cu126'], check=False)
        run([*base, '-r', REQUIREMENTS, *EXTRA_PIP])


def verify_stack() -> None:
    code = (
        'import torch, mujoco, warp, mujoco_warp, mjlab, rsl_rl, numpy\n'
        'print(f"torch {torch.__version__} cuda {torch.version.cuda} available {torch.cuda.is_available()} '
        'count {torch.cuda.device_count()}") \n'
        'print(f"mujoco {mujoco.__version__} warp {warp.__version__} mjlab {mjlab.__file__} numpy {numpy.__version__}")\n'
        'print(f"gpu {torch.cuda.get_device_name(0)}")\n'
    )
    r = run([sys.executable, '-c', code], check=False)
    if r.returncode != 0:
        raise SystemExit('the X1 stack does not import; see the pip log above')


def trainer_argv(hp: dict) -> list[str]:
    argv = [sys.executable, '-m', 'allmimic.train']
    for key, value in hp.items():
        if key.startswith('_') or key == 'description':
            continue
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            argv += [flag, '1' if value else '0']
        else:
            argv += [flag, str(value)]
    return argv


class MetricMirror(threading.Thread):
    """tail <out>/log.jsonl -> TensorBoard events, so the platform's `gm task data get` has scalars"""

    def __init__(self, log_path: Path, logdir: Path, stop: threading.Event):
        super().__init__(daemon=True)
        self.log_path, self.logdir, self.stop = log_path, logdir, stop
        self.written = 0

    def run(self) -> None:
        try:
            from tensorboardX import SummaryWriter
        except Exception as e:  # noqa: BLE001
            log(f'metric mirror disabled ({type(e).__name__}: {e}); console log still has every number')
            return
        self.logdir.mkdir(parents=True, exist_ok=True)
        w = SummaryWriter(logdir=str(self.logdir), comment='')
        log(f'metric mirror -> {self.logdir}')
        while not self.stop.is_set():
            try:
                if self.log_path.exists():
                    with open(self.log_path) as f:
                        f.seek(self.written)
                        for line in f:
                            self.written = f.tell()
                            rec = json.loads(line)
                            step = int(rec.get('it', rec.get('updates', 0)))
                            for k, v in rec.items():
                                if isinstance(v, bool) or not isinstance(v, (int, float)):
                                    continue
                                w.add_scalar(f'Train/{k}', float(v), step)
                            w.flush()
            except Exception as e:  # noqa: BLE001
                log(f'metric mirror: {type(e).__name__}: {e}')
            self.stop.wait(5.0)
        w.close()


def publish_checkpoint(out_dir: Path, out_pub: Path) -> Path | None:
    ckpts = sorted(out_dir.glob('ckpt_*.pt'), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        log(f'no checkpoint under {out_dir}')
        return None
    src = ckpts[-1]
    m = re.search(r'ckpt_(\d+)\.pt$', src.name)
    dst = out_pub / f'x1_supermimic_iter{int(m.group(1)) if m else 0:06d}.pt'
    out_pub.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    log(f'checkpoint {src.name} ({src.stat().st_size / 1e6:.1f} MB) -> {dst}')
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--hparams', type=Path, default=ROOT / 'configs' / 'x1_supermimic.json')
    ap.add_argument('--skip-install', action='store_true', help='do not pip install (stack already present)')
    ap.add_argument('--dry-run', action='store_true', help='print the plan and exit')
    args, extra = ap.parse_known_args()
    hp = json.loads(args.hparams.read_text())
    argv = trainer_argv(hp)
    out_dir = ROOT / hp.get('out', 'runs/x1_cmu_walk')
    logdir = ROOT / 'logs' / 'x1_supermimic' / time.strftime('%Y-%m-%d_%H-%M-%S')
    log(f'repo {ROOT}')
    log(f'hparams {args.hparams}: {json.dumps(hp)}')
    log(f'trainer {argv}')
    log(f'metrics -> {logdir}, checkpoints -> output/')
    if extra:
        log(f'ignoring extra args: {extra}')
    if args.dry_run:
        return 0

    if not args.skip_install:
        pip_install()
    verify_stack()

    (ROOT / 'output').mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    mirror = MetricMirror(out_dir / 'log.jsonl', logdir, stop)
    mirror.start()
    t0 = time.time()
    try:
        code = subprocess.call(
            argv, cwd=str(ROOT), env={**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONUNBUFFERED': '1'}
        )
    finally:
        stop.set()
        mirror.join(timeout=30)
    log(f'trainer exited with {code} after {(time.time() - t0) / 60:.1f} min')
    published = publish_checkpoint(out_dir, ROOT / 'output')
    log(f'published {published}' if published else 'nothing published')
    return int(code)


if __name__ == '__main__':
    raise SystemExit(main())
