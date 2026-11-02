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
# The pins the wrapper does not force, and why (all three were measured on the platform, see the report):
#   torch     the CUDA images already ship a CUDA build (IsaacSim 5.1 has torch 2.7 and mjlab 1.2.0 only requires
#             >= 2.7); reinstalling costs a ~1 GB download and risks a wheel that does not match the image's driver.
#             Installed below from the cu126 index only if the image's torch is missing or CPU-only.
#   numpy     the IsaacSim images carry an Isaac Sim pip constraint "numpy<2" against requirements.txt's
#             numpy==2.4.6, and pip stops with ResolutionImpossible before installing anything. None of mjlab,
#             mujoco-warp or warp-lang pins numpy (their PyPI metadata only says "numpy"), so the image's numpy is
#             used and verify_stack() prints which version actually ran.
#   pytest / matplotlib   test-only and optional.
SKIP_PINS = ('torch', 'pytest', 'matplotlib', 'numpy')


def log(msg: str) -> None:
    print(f'[gm-entry] {msg}', flush=True)


def run(cmd, cwd=ROOT, check=True, **kw):
    log('$ ' + ' '.join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd), check=check, **kw)


# pip indexes, in the order they are tried. The platform images point pip at an internal mirror
# (mirrors.ivolces.com) that does not resolve from inside the task container -- measured: "Name or service not
# known" for every package while `git clone` from github.com in the same container works, so the container has
# outbound network and the mirror host is what is unreachable. pypi.org and the torch wheel index therefore come
# first, and the image's own default is the last resort.
PIP_INDEXES = ('https://pypi.org/simple', 'https://download.pytorch.org/whl/cu126', None)


def stack_ok() -> bool:
    code = ('import torch, mujoco, warp, mujoco_warp, mjlab, rsl_rl, numpy\n'
            'assert torch.cuda.is_available()\n')
    r = subprocess.run([sys.executable, '-c', code], cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode == 0:
        return True
    log('stack check failed: ' + (r.stderr.strip().splitlines() or ['?'])[-1])
    return False


def ensure_pip() -> list[str]:
    """`python -m pip` for this interpreter, installing pip first if the image ships a bare Python.

    The plain Ubuntu images have no pip unless python3-pip was baked in. Try ensurepip (offline, ships with CPython),
    then get-pip.py, then apt; if all fail, raise with what was tried.
    """
    if subprocess.run([sys.executable, '-m', 'pip', '--version'], capture_output=True).returncode == 0:
        return [sys.executable, '-m', 'pip']
    for cmd in ([sys.executable, '-m', 'ensurepip', '--upgrade'],
                ['bash', '-lc', 'curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3 -'],
                ['bash', '-lc', 'apt-get update -qq && apt-get install -y -qq python3-pip']):
        log('no pip; trying ' + ' '.join(str(c) for c in cmd))
        if subprocess.run([str(c) for c in cmd], cwd=str(ROOT)).returncode == 0:
            if subprocess.run([sys.executable, '-m', 'pip', '--version'], capture_output=True).returncode == 0:
                return [sys.executable, '-m', 'pip']
    raise SystemExit('pip is unavailable and could not be installed in this image')


def pip_install() -> None:
    base = [*ensure_pip(), 'install', '--no-input', '--disable-pip-version-check']
    req = ROOT / 'runs' / 'requirements.gm.txt'
    req.parent.mkdir(parents=True, exist_ok=True)
    req.write_text(''.join(
        line for line in REQUIREMENTS.read_text().splitlines(keepends=True)
        if not line.strip().startswith(SKIP_PINS) and not line.strip().startswith('#')
    ))
    kept = [ln.split('==')[0].split()[0] for ln in req.read_text().splitlines() if ln.strip()]
    log(f'filtered requirements -> {req}: {kept} (left to the image: {", ".join(SKIP_PINS)})')
    for index in PIP_INDEXES:
        cmd = [*base, '-r', req, *EXTRA_PIP]
        if index:
            cmd += ['--index-url', index]
        log(f'pip against {index or "the image default index"}')
        if subprocess.run([str(c) for c in cmd], cwd=str(ROOT)).returncode == 0:
            break
    probe = run([sys.executable, '-c', 'import torch; print(torch.cuda.is_available())'], check=False,
                capture_output=True, text=True)
    if probe.returncode != 0 or 'True' not in probe.stdout:
        log('the image has no CUDA torch; installing the pinned wheel from the cu126 index')
        run([*base, 'torch==2.14.1', '--index-url', 'https://download.pytorch.org/whl/cu126'], check=False)
    log('pip install done')


def verify_stack() -> str:
    # The driver/toolkit pair matters: mujoco_warp.put_model calls warp's is_conditional_graph_supported() and raises
    # "Minimum supported CUDA version: 12.4." if it returns False, which is about the *driver* warp sees, not the
    # CUDA version torch was built with. The driver line is printed before training so the log records it either way.
    code = (
        'import subprocess, torch, mujoco, warp, mujoco_warp, mjlab, rsl_rl, numpy\n'
        'print(f"torch {torch.__version__} cuda {torch.version.cuda} available {torch.cuda.is_available()} '
        'count {torch.cuda.device_count()}") \n'
        'print(f"mujoco {mujoco.__version__} warp {warp.__version__} mjlab {mjlab.__file__} numpy {numpy.__version__}")\n'
        'print(f"gpu {torch.cuda.get_device_name(0)}")\n'
        'print("driver:", subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], '
        'capture_output=True, text=True).stdout.strip())\n'
        'print("cond graph supported:", warp.is_conditional_graph_supported())\n'
    )
    r = run([sys.executable, '-c', code], check=False)
    if r.returncode != 0:
        raise SystemExit('the X1 stack does not import; see the pip log above')
    m = re.search(r'^driver: (\S+)', r.stdout or '', re.M)
    return m.group(1) if m else ''


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

    # The image may already have everything (IsaacSim 5.1 ships torch 2.7 + CUDA); only install what is missing, so a
    # dead package index cannot fail a run that did not need it.
    if stack_ok():
        log('the image already imports the whole stack; skipping pip')
    elif not args.skip_install:
        pip_install()
    driver = verify_stack()
    # mjlab disables CUDA graphs below driver 12.4 and prints why; mujoco_warp raises instead, so on an older driver
    # the training process gets scripts/gm_compat.py's no-op guard. Decided from the driver number, not assumed.
    try:
        old_driver = tuple(int(v) for v in driver.split('.')[:2]) < (12, 4)
    except ValueError:
        old_driver = False

    (ROOT / 'output').mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    mirror = MetricMirror(out_dir / 'log.jsonl', logdir, stop)
    mirror.start()
    t0 = time.time()
    try:
        # append, never replace: the platform's python is a hermetic kit install whose site-packages (torch among
        # them) is reachable through the image's own PYTHONPATH, and setting PYTHONPATH=ROOT alone made
        # `python -m allmimic.train` die with "No module named 'torch'" while the very same interpreter imported it
        # in the probe above (TASK_20261102_024).
        env = {**os.environ, 'PYTHONPATH': f'{ROOT}:{ROOT / "scripts"}:{os.environ.get("PYTHONPATH", "")}',
               'PYTHONUNBUFFERED': '1'}
        if old_driver:
            # scripts/gm_compat.py: bypass mujoco_warp's ">= driver 12.4" gate, which mjlab itself works around by
            # disabling CUDA graphs. Only set when the probe found the driver below 12.4.
            env['GM_ALLOW_OLD_CUDA_DRIVER'] = '1'
            preamble = ('import sys; sys.path.insert(0, {s!r}); import gm_compat; gm_compat.apply(); '
                        'sys.argv = {a!r}; import runpy; runpy.run_module("allmimic.train", run_name="__main__")')
            cmd = [sys.executable, '-c', preamble.format(s=str(ROOT / 'scripts'), a=argv[2:])]
        else:
            cmd = argv
        log('driver {} -> {}'.format(driver, 'bypassing mujoco_warp\'s >=12.4 gate (gm_compat)' if old_driver
                                     else 'supported natively'))
        code = subprocess.call(cmd, cwd=str(ROOT), env=env)
    finally:
        stop.set()
        mirror.join(timeout=30)
    log(f'trainer exited with {code} after {(time.time() - t0) / 60:.1f} min')
    published = publish_checkpoint(out_dir, ROOT / 'output')
    log(f'published {published}' if published else 'nothing published')
    return int(code)


if __name__ == '__main__':
    raise SystemExit(main())
