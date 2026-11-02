"""Render arm-study qpos sequences side by side (MuJoCo offscreen, CPU).

  ~/code/X1_29_AMP/_gmr_local/venv/bin/python tools/x1_render_arms.py \
      --study data/arm_study_v2 --clip 09_01 --variants stock,split --out data/arm_study_v2/renders

Also accepts plain qpos (N, 36) npy files via --npy for arbitrary sequences.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'data' / 'pylibs'))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from allmimic.robots import x1_29dof as x1  # noqa: E402


def build_model(width: int = 480, height: int = 640):
    spec = mujoco.MjSpec.from_file(str(x1.X1_XML))
    spec.visual.global_.offwidth = max(width, 640)
    spec.visual.global_.offheight = max(height, 480)
    spec.assets = {f'{spec.meshdir}/{p.name}': p.read_bytes() for p in sorted(x1.MESH_DIR.iterdir()) if p.is_file()}
    f = spec.worldbody.add_geom()
    f.name, f.type, f.size = 'floor', mujoco.mjtGeom.mjGEOM_PLANE, [0, 0, 0.05]
    f.rgba = [0.25, 0.27, 0.3, 1]
    return spec.compile()


def render_variant(model, qpos: np.ndarray, out_dir: Path, tag: str, fps_src: float, fps_out: float = 25.0,
                   every: int = 1, n_max: int = 0, width: int = 480, height: int = 640):
    import imageio.v2 as imageio

    r = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.45]
    cam.distance = 2.6
    cam.azimuth = 130
    cam.elevation = -12
    opt = mujoco.MjvOption()
    d = mujoco.MjData(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, int(round(fps_src / fps_out)))
    idx = list(range(0, len(qpos), step * every))[: n_max or None]
    frames = []
    for i in idx:
        d.qpos[:] = qpos[i]
        mujoco.mj_forward(model, d)
        r.update_scene(d, cam, opt)
        frames.append(r.render())
    try:
        path = out_dir / f'{tag}.mp4'
        imageio.mimwrite(path, frames, fps=fps_out, macro_block_size=None, quality=8)
    except Exception as e:  # noqa: BLE001 -- fall back to a PNG frame sequence (no extra deps)
        print(f'mp4 failed ({type(e).__name__}: {e}); writing PNG frames instead')
        path = out_dir / f'{tag}_frames'
        path.mkdir(parents=True, exist_ok=True)
        for i, fr in enumerate(frames):
            imageio.imwrite(path / f'{i:04d}.png', fr)
    r.close()
    return path, len(frames)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', type=Path, default=ROOT / 'data' / 'arm_study_v2')
    ap.add_argument('--clip', type=str, default='09_01')
    ap.add_argument('--variants', type=str, default='stock,split')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--fps-src', type=float, default=120.0)
    ap.add_argument('--n-max', type=int, default=0)
    args = ap.parse_args()
    out = args.out or (args.study / 'renders' / args.clip)
    model = build_model()
    for v in args.variants.split(','):
        f = args.study / f'{v}__{args.clip}.npy'
        if not f.exists():
            print(f'missing {f}')
            continue
        qpos = np.load(f)
        p, n = render_variant(model, qpos, out, v, args.fps_src, n_max=args.n_max)
        print(f'{v}: {n} frames -> {p}')


if __name__ == '__main__':
    raise SystemExit(main())
