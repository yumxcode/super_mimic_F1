#!/usr/bin/env bash
# Phase-2 AMASS download: per-file mirror of realdream-ai/AMASS raw/CMU (SMPL-H npz).
# Direct huggingface.co is unreachable from this host; hf-mirror.com works (verified
# 2026-11-02). Raw files land in data/amass/CMU_raw (gitignored) and are DELETED after
# the stageii conversion + retargeting pass (storage red line: only training npz +
# meta + held-out npz are kept).
#
# Subject selection rationale (docs/X1_PHASE2_DATASET.md): walk families already
# local from phase 1 (07/08/16/35/37/38/49/82/91/103) + run (09/10) + locomotion
# variety (02/06/13/17) + dance families (05/60/131/135/136/137/143) + varied
# exercise (90/138). Clips are classified by measured kinematics AFTER conversion,
# so the subject list only needs to cover the motion families, not exact trials.
set -u
BASE="https://hf-mirror.com/datasets/realdream-ai/AMASS/resolve/main/raw/CMU"
DST="$(cd "$(dirname "$0")/.." && pwd)/data/amass/CMU_raw"
mkdir -p "$DST"
SUBJECTS="${*:-02 05 06 09 10 13 17 60 90 131 135 136 137 138 143}"
API="https://hf-mirror.com/api/datasets/realdream-ai/AMASS/tree/main/raw/CMU"

LIST="$DST/_urls.txt"
: > "$LIST"
for s in $SUBJECTS; do
  curl -sL -m 30 "$API/$s" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    for e in d:
        p = e['path']
        if p.endswith('_poses.npz'):
            name = p.split('/')[-1].replace('_poses.npz', '')
            print(f'$s {name} $BASE/$s/{name}_poses.npz')
except Exception:
    pass
" >> "$LIST"
done
TOTAL=$(wc -l < "$LIST" | tr -d ' ')
echo "candidates: $TOTAL"

ok=0; fail=0
while read -r sub name url; do
  f="$DST/$name.npz"
  if [ -s "$f" ]; then ok=$((ok+1)); continue; fi
  for attempt in 1 2 3; do
    curl -sL --fail -m 300 -o "$f.tmp" "$url" && mv "$f.tmp" "$f" && break
    sleep 2
  done
  if [ -s "$f" ]; then ok=$((ok+1)); else fail=$((fail+1)); echo "FAIL $name"; rm -f "$f" "$f.tmp"; fi
done < "$LIST"
echo "downloaded ok=$ok fail=$fail"
du -sh "$DST"
