#!/usr/bin/env bash
#
# LAST experiment: morphological-attention fold 2 with the skip gate WARM-STARTED half-active
# (--morph-attn-warm 0.5), the morphological analogue of the linear gate's gamma=0.5 init (which lifted
# lin-attn fold 0 from 0.455 -> 0.499). Distinct tag unet_morphattn_g05 so it never mixes with the
# existing 2-fold morphattn (which used the identity-init gate).
#
# Runs strictly LAST: waits until the g05 attention folds AND the convsep fold-0 reprune both finish, so
# it trains alone on the full card. HARD DEADLINE GUARD: if that point is reached after 06:00 it SKIPS
# (a morphattn fold needs ~1.5-2 h and the upload deadline is ~08:00) -- exactly the user's "only if it
# can run before 08:00" condition.
#
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=unet_morphattn_g05; FOLD=2
CUTOFF=$(date -d "2026-07-10 06:00:00" +%s)   # must START training before this to be safe for ~08:00

echo "############ $TAG f$FOLD (LAST): waiting for g05 folds + convsep reprune to finish $(date '+%F %T') ############"
while pgrep -f "queue_g05_folds.sh" >/dev/null 2>&1 || pgrep -f "convsep_f0_reprune.sh" >/dev/null 2>&1; do
  sleep 60
done

if [[ -f "results/${TAG}_f${FOLD}_scores.json" ]]; then
  echo "== $TAG f$FOLD already done -> nothing to do =="; exit 0
fi
now=$(date +%s)
if [[ "$now" -gt "$CUTOFF" ]]; then
  echo "############ SKIP: reached $(date '+%T') > 06:00 cutoff -- not enough time before 08:00 for a morphattn fold ############"
  exit 0
fi

echo "############ starting $TAG f$FOLD (warm=0.5) $(date '+%F %T') ############"
python3 train_eval.py --tag "$TAG" --fold "$FOLD" \
    --morph-unet none --morph-attn --morph-k 3 --morph-attn-warm 0.5 --batch-size 12 \
  || echo "!!! $TAG f$FOLD FAILED (exit $?) -- continuing"
echo "############ $TAG f$FOLD DONE $(date '+%F %T') ############"
