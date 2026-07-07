#!/usr/bin/env bash
#
# 3-fold training of the CONVSEP twin (depthwise 3x3 + 1x1, --morph-impl convsep):
# the plain-conv ablation control for morphunet_heavy. Same config (heavy), same conv
# stem, same k=3, same default schedule (400 epochs, patience 30) -> matched to the morph
# model so the only difference is morphology -> depthwise conv.
#
# Runs the folds SEQUENTIALLY (one at a time) so it never stacks two convsep jobs on the
# 16 GB GPU while the prune sweep and the mpm_full_l2 fold trainings are also running.
# Batch 24 matches the morph baseline and was probed to peak ~4 GB (fits current free mem);
# if a fold OOMs, drop BATCH to 16 and re-run -- finished folds are skipped.
#
set -uo pipefail
cd /home/Kasimatis/Documents/kasimat/morph-unet

TAG=convsep_heavy
BATCH=24
for FOLD in 0 1 2; do
  BEST="results/${TAG}_f${FOLD}_best.pth"
  if [[ -f "$BEST" ]]; then
    echo "=== fold $FOLD already trained ($BEST) -> skip ==="
    continue
  fi
  echo "############################################################"
  echo "## convsep_heavy  fold $FOLD  (batch=$BATCH)  $(date '+%F %T')"
  echo "############################################################"
  python train_eval.py \
      --tag "$TAG" --morph-unet heavy --morph-impl convsep --morph-k 3 \
      --fold "$FOLD" --batch-size "$BATCH" \
    || echo "!!! fold $FOLD FAILED (exit $?) -- continuing to next fold"
done
echo "=== convsep 3-fold training done $(date '+%F %T') ==="
