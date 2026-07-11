#!/usr/bin/env bash
#
# RESUME the convmpm_small (fs=13) fb/fbfg prunes AFTER the 2M model finishes, so they run alone on
# the card. Waits on the exact 2M training PID (no pgrep -f -- avoids stale/wrong matches), then runs
# every fb/fbfg x {0.5,0.3,0.1} global prune whose _prune.json does NOT already exist (so the already-
# finished fb-k50 is skipped and a re-run is idempotent).
#
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_PID=417196
echo "############ resume-prunes: waiting for 2M training PID $TRAIN_PID to finish $(date '+%F %T') ############"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
echo "############ 2M training gone -> starting remaining prunes alone $(date '+%F %T') ############"
sleep 10

TAG=convmpm_small
kk() { printf "k%02d" "$(python3 -c "print(round($1*100))")"; }
for M in fb fbfg; do
  for K in 0.5 0.3 0.1; do
    stem="results/${TAG}_prune-${M}g-$(kk $K)_f0_prune.json"
    if [[ -f "$stem" ]]; then echo "== skip $M k=$K (already done: $stem) =="; continue; fi
    echo "############ prune $TAG method=$M keep=$K (global,min-keep=2) $(date '+%F %T') ############"
    python3 prune.py --tag "$TAG" --fold 0 --config heavy --impl convmpm --fs 13 \
        --method "$M" --keep-ratio "$K" --alloc global --global-norm max --min-keep 2 \
      || echo "!!! $TAG $M k=$K FAILED (exit $?) -- continuing"
  done
done
echo "############ convmpm_small fb/fbfg prunes DONE $(date '+%F %T') ############"
