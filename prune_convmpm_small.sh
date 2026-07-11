#!/usr/bin/env bash
#
# Prune the trained convmpm_small (fs=13, /5 width) with the FIXED forward-backward at three global
# keep ratios, TWICE:
#   * fb    -- the fixed HMM forward-backward (residual add-one on stage boundaries, skip edges,
#              pi emission from E[morph(i)]); state stats over the whole feature map.
#   * fbfg  -- the SAME fixed forward-backward but with state stats restricted to foreground
#              receptive fields (the fbnew move, applied to the fixed fb). "fb-new based on fixed fb".
# Global allocation, per-layer max-norm, min-keep=2. Runs sequentially so it can share the card with
# the concurrent convmpm_small2m training (2 GPU jobs max).
#
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
sleep 20   # let the 2M training grab its memory first (avoid a startup OOM race)

TAG=convmpm_small
for M in fb fbfg; do
  for K in 0.5 0.3 0.1; do
    echo "############ prune $TAG  method=$M  keep=$K  (global,min-keep=2)  $(date '+%F %T') ############"
    python3 prune.py --tag "$TAG" --fold 0 --config heavy --impl convmpm --fs 13 \
        --method "$M" --keep-ratio "$K" --alloc global --global-norm max --min-keep 2 \
      || echo "!!! $TAG $M k=$K FAILED (exit $?) -- continuing"
  done
done
echo "############ convmpm_small fb/fbfg prunes DONE $(date '+%F %T') ############"
