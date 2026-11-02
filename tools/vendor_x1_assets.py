"""Vendor the X1 29-DoF MJCF into this repository as assets/x1_29dof/.

The source asset lives outside the repository (see --src). This script copies the meshes the robot XML
actually references plus the XML itself, and applies exactly four mechanical changes so the asset works
inside allmimic / mjlab on Linux:

  1. meshdir  "../meshes" -> "meshes"          (flat layout: assets/x1_29dof/{x1_29dof.xml,meshes/})
  2. case     mesh files are copied under the name the XML references, so a macOS-only case mismatch
              (XML `lumbar_pitch.STL` vs file `lumbar_pitch.stl`) cannot break the Linux container
  3. names    every unnamed <geom> gets a name derived from its body / role. Purely additive: MuJoCo
              resolves collision groups and the env's foot geoms by name.
  4. sole     each foot's four 2 mm corner spheres are replaced by one flat sole box with the same contact
              class and the same footprint. See foot_sole_box() -- this is the only change to the contact
              geometry, and it is not in the source asset.

No inertia, joint limit, actuator, contact parameter or body pose is modified. PROVENANCE.md (written next
to the XML) records the source path, its sha256 and the list of changes.

  PYTHONPATH=. python tools/vendor_x1_assets.py                     # default source, writes assets/x1_29dof/
  python tools/vendor_x1_assets.py --src ~/code/X1_29_AMP/X1_29DOF --check   # verify the vendored copy
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = Path.home() / 'code' / 'X1_29_AMP' / 'X1_29DOF'
SRC_XML = Path('mjcf') / 'robot' / 'xyber_x1' / 'xyber_x1_serial.xml'
OUT_DIR = ROOT / 'assets' / 'x1_29dof'
FOOT_BODIES = ('left_ankle_roll_link', 'right_ankle_roll_link')
SOLE_THICKNESS = 0.005  # m; a 5 mm plate whose underside is the plane the corner spheres touched

GEOM_RE = re.compile(r'<geom\b[^>]*?/?>')
BODY_RE = re.compile(r'<body name="([^"]+)"')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def attr(tag: str, key: str) -> str | None:
    m = re.search(rf'\b{key}="([^"]*)"', tag)
    return m.group(1) if m else None


def name_geoms(xml: str) -> tuple[str, list[str]]:
    """add a `name` to every unnamed <geom>; body-scoped, so the result is deterministic"""
    out, pos, changes = [], 0, []
    body = 'world'
    # separate counters per role, so the four contact spheres of a foot are foot_1..foot_4 (markers are named apart)
    foot_k = {(b, role): 0 for b in FOOT_BODIES for role in ('collision', 'marker')}
    while True:
        m = GEOM_RE.search(xml, pos)
        b = BODY_RE.search(xml, pos) if m else None
        if m is None:
            out.append(xml[pos:])
            break
        # the last <body name=...> opened before this geom is its parent body
        while b is not None and b.start() < m.start():
            body = b.group(1)
            b = BODY_RE.search(xml, b.end())
        out.append(xml[pos : m.start()])
        tag = m.group(0)
        # <geom> lines inside <default> classes are declarations, not model geoms: leave them alone
        in_default = xml.count('<default', 0, m.start()) > xml.count('</default>', 0, m.start())
        if attr(tag, 'name') is None and not in_default:
            cls, mesh = attr(tag, 'class'), attr(tag, 'mesh')
            if body in FOOT_BODIES and cls == 'collision':
                foot_k[(body, 'collision')] += 1
                name = f'{body.split("_")[0]}_foot_{foot_k[(body, "collision")]}_collision'
            elif body in FOOT_BODIES:
                foot_k[(body, 'marker')] += 1
                name = f'{body.split("_")[0]}_foot_{foot_k[(body, "marker")]}_marker'
            else:
                stem = re.sub(r'_link$', '', body)
                kind = 'collision' if cls == 'collision' else ('visual' if mesh else 'geom')
                name = f'{stem}_{kind}'
            tag = tag.replace('<geom', f'<geom name="{name}"', 1)
            changes.append(f'{body}: unnamed <geom> -> name="{name}"')
        out.append(tag)
        pos = m.end()
    return ''.join(out), changes


def referenced_meshes(xml: str) -> list[str]:
    return sorted(set(re.findall(r'<mesh\b[^>]*\bfile="([^"]+)"', xml)))


def foot_sole_box(xml: str) -> tuple[str, list[str]]:
    """replace each foot's four 2 mm corner spheres with one flat sole box.

    The source asset gives the foot exactly four collision geoms: 2 mm spheres at the four corners of the sole
    rectangle. A 2 mm point contact has to carry ~86 N of standing load, which at the 200 Hz contact stiffness
    sinks the sphere several radii into the floor (measured: up to 15 mm) -- the contact normal is then resolved
    on a degenerate, deeply penetrating point. A plate is how the G1 model is built and how a real sole behaves:
    the box keeps the same footprint (the four sphere centres become the box corners) and the same underside
    plane (the spheres' lowest point), so the model's standing height does not change.
    """
    per_foot: dict[str, list[tuple[int, re.Match]]] = {}
    for m in re.finditer(r'[ \t]*<geom\b[^>]*?/?>[ \t]*\n', xml):
        tag = m.group(0)
        name = attr(tag, 'name') or ''
        fm = re.fullmatch(r'(left|right)_foot_(\d)_collision', name)
        if fm and attr(tag, 'type') == 'sphere':
            per_foot.setdefault(fm.group(1), []).append(m)
    if not per_foot:
        return xml, []

    changes, spans = [], []
    for foot, hits in sorted(per_foot.items()):
        if len(hits) != 4:
            raise SystemExit(f'{foot} foot: expected 4 corner spheres, found {len(hits)}')
        pos = [np.array([float(v) for v in (attr(h.group(0), 'pos') or '0 0 0').split()]) for h in hits]
        radius = float((attr(hits[0].group(0), 'size') or '0').split()[0])
        pts = np.array(pos)
        spread = pts.max(0) - pts.min(0)
        normal = int(np.argmin(spread))  # the four spheres share one coordinate: that axis is the sole normal
        if spread[normal] > 1e-6:
            raise SystemExit(f'{foot} foot: the four spheres do not lie in one plane ({pts})')
        sign = float(np.sign(pts[0, normal]))  # the sole is on the far side of the body origin (the ankle)
        size = list(spread / 2)
        size[normal] = SOLE_THICKNESS / 2
        centre = pts.mean(0)
        centre[normal] = sign * (abs(pts[0, normal]) + radius - SOLE_THICKNESS / 2)
        box = (
            f'<geom name="{foot}_foot_sole_collision" type="box" '
            f'size="{" ".join(f"{v:.5g}" for v in size)}" '
            f'pos="{" ".join(f"{v:.5g}" for v in centre)}" class="collision" />'
        )
        first = hits[0]
        indent = first.group(0)[: len(first.group(0)) - len(first.group(0).lstrip(' \t'))]
        last = hits[-1]
        spans.append((first.start(), last.end(), f'{indent}{box}\n'))
        changes.append(
            f'{foot}_ankle_roll_link: 4 sphere collision geoms -> 1 sole box '
            f'(size {" ".join(f"{v:.5g}" for v in size)}, pos {" ".join(f"{v:.5g}" for v in centre)}, '
            f'sole plane {abs(pts[0, normal]) + radius:.5f} from the ankle)'
        )
    for start, end, repl in sorted(spans, reverse=True):
        xml = xml[:start] + repl + xml[end:]
    return xml, changes


def build(src: Path, out_dir: Path, keep_stl_bytes: bool) -> None:
    src_xml = src / SRC_XML
    xml = src_xml.read_text()
    refs = referenced_meshes(xml)
    if not refs:
        raise SystemExit(f'no meshes referenced in {src_xml}')

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'meshes').mkdir(parents=True, exist_ok=True)
    copied, missing, renamed = [], [], []
    for ref in refs:
        cand = src / 'meshes' / ref
        if not cand.exists():  # macOS case-insensitive source, case-sensitive target name
            alt = [p for p in (src / 'meshes').iterdir() if p.name.lower() == ref.lower()]
            if not alt:
                missing.append(ref)
                continue
            cand, renamed = alt[0], [*renamed, f'{alt[0].name} -> {ref}']
        dst = out_dir / 'meshes' / ref
        if keep_stl_bytes:
            shutil.copyfile(cand, dst)
        copied.append(ref)
    if missing:
        raise SystemExit(f'missing meshes in {src / "meshes"}: {missing}')

    xml = re.sub(r'<compiler\b[^>]*?>', lambda m: re.sub(r'meshdir="[^"]*"', 'meshdir="meshes"', m.group(0)), xml, count=1)
    xml, changes = name_geoms(xml)
    xml, sole = foot_sole_box(xml)
    (out_dir / 'x1_29dof.xml').write_text(xml)

    prov = [
        '# assets/x1_29dof provenance',
        '',
        f'source asset : `{src}`',
        f'source xml   : `{SRC_XML}` (sha256 `{sha256(src_xml)}`)',
        f'vendored xml : `assets/x1_29dof/x1_29dof.xml` (sha256 `{sha256(out_dir / "x1_29dof.xml")}`)',
        f'meshes       : {len(copied)} STL referenced by the xml, copied under the referenced name',
        '',
        '## Changes applied by tools/vendor_x1_assets.py',
        '',
        '1. `<compiler meshdir>`: `../meshes` -> `meshes` (flat layout).',
    ]
    prov += [f'2. mesh file renamed on copy (Linux case sensitivity): `{r}`' for r in renamed]
    prov += ['3. geom names added (collision selection / domain randomization by name):', '']
    prov += [f'   - {c}' for c in changes]
    prov += ['4. foot sole box (the only change to the collision geometry):', '']
    prov += [f'   - {c}' for c in sole]
    prov += [
        '',
        '   Why: the four 2 mm corner spheres put the whole standing load (~86 N each at 35 kg) on a point, and',
        '   at the 200 Hz contact stiffness (solref 0.02) the sphere sinks several radii into the floor. The box',
        '   keeps the same footprint (the sphere centres become its corners) and the same underside plane',
        '   (the spheres\' lowest point), so the standing height, the motion-library root heights and the',
        '   contact material (class="collision", inherited condim/friction/solref) are unchanged.',
    ]
    prov += [
        '',
        'Everything else (body poses, inertias, joint axes/limits, damping, friction loss, actuators, contact',
        'parameters) is byte-identical to the source XML.',
        '',
        f'`--check` recomputes the same transformation from the source and diffs it against the vendored copy.',
    ]
    (out_dir / 'PROVENANCE.md').write_text('\n'.join(prov) + '\n')

    size = sum(p.stat().st_size for p in (out_dir / 'meshes').iterdir())
    print(f'wrote {out_dir}/x1_29dof.xml + {len(copied)} meshes ({size / 1e6:.1f} MB)')
    if renamed:
        print(f'renamed on copy (case): {renamed}')
    print(f'geom names added: {len(changes)}')


def check(src: Path, out_dir: Path) -> int:
    xml = (src / SRC_XML).read_text()
    want, _ = name_geoms(
        re.sub(r'<compiler\b[^>]*?>', lambda m: re.sub(r'meshdir="[^"]*"', 'meshdir="meshes"', m.group(0)), xml, count=1)
    )
    want, _ = foot_sole_box(want)
    have = (out_dir / 'x1_29dof.xml').read_text()
    if want != have:
        (out_dir / 'x1_29dof.expected.xml').write_text(want)
        print(f'MISMATCH: {out_dir}/x1_29dof.xml differs from the transformation of {src / SRC_XML}')
        print(f'expected copy: {out_dir}/x1_29dof.expected.xml')
        return 1
    refs = referenced_meshes(have)
    missing = [r for r in refs if not (out_dir / 'meshes' / r).exists()]
    print(f'OK: xml matches the transformation of the source; {len(refs)} meshes present' if not missing
          else f'MISSING meshes: {missing}')
    return 0 if not missing else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--src', type=Path, default=DEFAULT_SRC, help='X1_29DOF asset root (contains mjcf/, meshes/)')
    ap.add_argument('--out', type=Path, default=OUT_DIR)
    ap.add_argument('--check', action='store_true', help='verify the vendored copy instead of writing it')
    args = ap.parse_args()
    if args.check:
        return check(args.src, args.out)
    build(args.src, args.out, keep_stl_bytes=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
