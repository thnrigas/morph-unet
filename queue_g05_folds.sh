#!/usr/bin/env bash
#
# Queue the remaining two folds of the lin-attention gamma=0.5 experiment (fold 0 already done:
# Vessel 0.499). Runs ONLY after BOTH prune sweeps finish (fbnew k03/k05 + random-global convsep),
# so the 16 GB card never holds more than the prune jobs OR this -- never a 3-way OOM. Batch-size 12
# to match fold 0 exactly (so the eventual 3-fold mean is apples-to-apples). When both folds land it
# regenerates the attention PNG and rebuilds the pptx, so the table goes to the full 3-fold gamma=0.5.
#
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VENV=/tmp/claude-1001/-home-Kasimatis-Documents-kasimat/f6662d64-bc71-456a-915b-ec84152c4c68/scratchpad/pptxvenv/bin/python

echo "############ g05 folds 1-2: waiting for BOTH prune sweeps to finish $(date '+%F %T') ############"
while pgrep -f "fbnew_extremes.sh" >/dev/null 2>&1 || pgrep -f "random_convsep.sh" >/dev/null 2>&1; do
  sleep 60
done
echo "############ prune sweeps done -> training g05 folds 1,2 $(date '+%F %T') ############"

for FOLD in 1 2; do
  if [[ -f "results/unet_linattn_g05_f${FOLD}_scores.json" ]]; then
    echo "== g05 fold $FOLD already done -> skip =="; continue
  fi
  echo "## TRAIN unet_linattn_g05 fold $FOLD (gamma=0.5) $(date '+%F %T')"
  python3 train_eval.py --tag unet_linattn_g05 --fold "$FOLD" \
      --lin-attn --lin-attn-heads 4 --lin-attn-gamma-init 0.5 --batch-size 12 \
    || echo "  !!! g05 fold $FOLD FAILED (exit $?) -- continuing"
done

echo "############ g05 3-fold done -> refresh attention PNG + pptx $(date '+%F %T') ############"
python3 slides/figs/tbl_attention.py || true
"$VENV" slides/figs/build_deck.py || true
python3 train_eval.py --fold-mean unet_linattn_g05 || true
echo "############ queue_g05_folds DONE $(date '+%F %T') ############"
