#!/usr/bin/env bash
#
# fb-morph prune (global) on the two ConvMPM models: transitions come from the 1x1 morphological
# neuron's actual channel selection (max-plus join + min-plus meet winners, foreground-restricted),
# not co-activation. Same keep ratios / settings as the convmpm_small fbfg run so the three columns
# (fb, fbfg, fbmorph) line up. Runs concurrently with the random baseline (GPU has headroom).
# Idempotent via a per-run *_fbmorph.done marker.
#
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {  # tag fs keep
  local tag=$1 fs=$2 K=$3
  local kk; kk=$(printf "k%02d" "$(python3 -c "print(round($K*100))")")
  local stem="${tag}_prune-fbmorphg-${kk}_f0"
  if [[ -f "results/${stem}_fbmorph.done" ]]; then echo "== skip ${stem} (done) =="; return; fi
  echo "############ $stem  (fbmorph keep=$K, global max min-keep=2)  $(date '+%F %T') ############"
  if python3 prune.py --tag "$tag" --fold 0 --config heavy --impl convmpm --fs "$fs" \
       --method fbmorph --keep-ratio "$K" --alloc global --global-norm max --min-keep 2; then
    touch "results/${stem}_fbmorph.done"
  else
    echo "!!! $stem FAILED (exit $?) -- continuing"
  fi
}

for K in 0.5 0.3 0.1; do run convmpm_small   13 "$K"; done
for K in 0.5 0.3 0.1; do run convmpm_small2m 37 "$K"; done
echo "############ fb-morph prunes DONE $(date '+%F %T') ############"
