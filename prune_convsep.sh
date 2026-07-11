#!/usr/bin/env bash
#
# Ablation prune: convsep_heavy (the SAME 3x3->1x1 block but LINEAR convolutions, no morphology).
# Mirrors morphunet_heavy's Phase-A global sweep so the two overlay directly on the pruning slide:
#   criteria lin/act/fb x global x keeps {0.1, 0.5}, fold 1 (convsep's best fold).
# l1x1 is skipped -- it's the morphological (SE-norm) criterion and does not apply to a linear block.
#
# Waits for the main finish_sweep.sh driver to exit first, so it never competes with the heavy
# random runs for the 16 GB GPU. Resume-aware (existing *_prune.json -> skip). Logs to finish_sweep.log
# so the existing monitor picks up the transitions.
#
set -uo pipefail
cd "$(dirname "$0")"
RESULTS=results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EPOCHS=80; LR=5e-5; MIN_KEEP=4; GLOBAL_NORM=max; PRUNE_TOL=0.015
TAG=convsep_heavy; CFG=heavy; IMPL=convsep; FOLD=1

echo "############ ABLATION: waiting for finish_sweep.sh to complete before pruning convsep ############"
while pgrep -f "finish_sweep[.]sh" >/dev/null 2>&1; do sleep 60; done
echo "############ ABLATION: convsep_heavy global prune (fold $FOLD, lin/act/fb, keeps 0.1/0.5) ############"

if [[ ! -f "$RESULTS/${TAG}_f${FOLD}_best.pth" ]]; then
  echo "!!! SKIP: base checkpoint $RESULTS/${TAG}_f${FOLD}_best.pth MISSING"; exit 1
fi

n=0
for METHOD in lin act fb; do
  for KEEP in 0.1 0.5; do
    KK=$(printf "k%02d" "$(python3 -c "print(round($KEEP*100))")")
    STEM="${TAG}_prune-${METHOD}g-${KK}_f${FOLD}"
    if [[ -f "$RESULTS/${STEM}_prune.json" ]]; then
      echo "  $STEM already done -> skip"; continue
    fi
    echo "== $STEM (global, keep=$KEEP) =="
    python3 prune.py --tag "$TAG" --config "$CFG" --impl "$IMPL" --fold "$FOLD" \
        --method "$METHOD" --keep-ratio "$KEEP" \
        --alloc global --global-norm "$GLOBAL_NORM" --min-keep "$MIN_KEEP" \
        --skip-ft-if-within "$PRUNE_TOL" --finetune-epochs "$EPOCHS" --lr "$LR" \
      || echo "  !!! $STEM FAILED (exit $?) -- continuing"
    n=$((n+1))
  done
done
echo "############ ABLATION DONE: $n convsep runs ############"
python3 collate_prune.py || true
