#!/usr/bin/env bash
#
# Train the two attention arms locally after the VM crash (unet_baseline already trained on
# the VM). Same plain-conv (config=none) backbone + default schedule as vm_run.sh PHASE 1b,
# so the only difference between the arms is the skip mechanism:
#   unet_morphattn -> morphological top-hat/bottom-hat gated skips
#   unet_linattn   -> linear cross-attention gated skips
#
# PRIORITY: fold 0 of EACH model first (so both fold-0 results land early), then the
# remaining folds 1,2. Runs one training at a time (never stacks two jobs on the 16 GB GPU
# while the extreme prune sweep is also using it). Resume-aware: finished folds are skipped.
# Logs to attn_train.log so the monitor can follow transitions.
#
set -uo pipefail
cd "$(dirname "$0")"
RESULTS=results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# tag|extra-train_eval-flags
declare -A FLAGS=(
  [unet_morphattn]="--morph-unet none --morph-attn --morph-k 3"
  [unet_linattn]="--lin-attn --lin-attn-heads 4"
)

train_one() {   # $1=tag  $2=fold
  local TAG=$1 FOLD=$2
  local BEST="$RESULTS/${TAG}_f${FOLD}_best.pth"
  if [[ -f "$BEST" ]]; then echo "== $TAG f$FOLD already trained -> skip =="; return 0; fi
  echo "############################################################"
  echo "## TRAIN $TAG  fold $FOLD   $(date '+%F %T')"
  echo "############################################################"
  python3 train_eval.py --tag "$TAG" --fold "$FOLD" ${FLAGS[$TAG]} \
    || echo "!!! $TAG f$FOLD FAILED (exit $?) -- continuing"
}

echo "############ ATTENTION training START $(date '+%F %T') ############"
# fold 0 of each first
train_one unet_morphattn 0
train_one unet_linattn   0
# then the remaining folds
train_one unet_morphattn 1
train_one unet_morphattn 2
train_one unet_linattn   1
train_one unet_linattn   2

echo "############ ATTENTION training DONE $(date '+%F %T') ############"
python3 train_eval.py --fold-mean unet_morphattn || true
python3 train_eval.py --fold-mean unet_linattn   || true
