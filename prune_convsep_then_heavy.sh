#!/usr/bin/env bash
#
# Reordered remaining prune work: do the CHEAP convsep ablation FIRST, leave the heavy random
# (the expensive big-model fine-tunes) for LAST.
#
#   PHASE 1  convsep_heavy ablation -- same 3x3->1x1 block, LINEAR convs (no morphology).
#            Mirrors morphunet_heavy's global sweep: lin/act/fb x global x keeps {0.1,0.5}, fold 1.
#            (l1x1 = morphological SE-norm criterion, N/A to a linear block, so skipped.)
#   PHASE 2  morphunet_heavy random baseline -- local, keeps {0.1,0.3}, fold 0, batch 12.
#
# Resume-aware (existing *_prune.json -> skip). Logs to finish_sweep.log so the monitor sees it.
#
set -uo pipefail
cd "$(dirname "$0")"
RESULTS=results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EPOCHS=80; LR=5e-5; MIN_KEEP=4; GLOBAL_NORM=max; PRUNE_TOL=0.015; HEAVY_BATCH=12
n=0

echo "############ PHASE 1: convsep_heavy ablation (global, lin/act/fb, keeps 0.1/0.5, fold 1) ############"
CTAG=convsep_heavy; CCFG=heavy; CIMPL=convsep; CFOLD=1
if [[ ! -f "$RESULTS/${CTAG}_f${CFOLD}_best.pth" ]]; then
  echo "!!! SKIP convsep: base checkpoint $RESULTS/${CTAG}_f${CFOLD}_best.pth MISSING"
else
  for METHOD in lin act fb; do
    for KEEP in 0.1 0.5; do
      KK=$(printf "k%02d" "$(python3 -c "print(round($KEEP*100))")")
      STEM="${CTAG}_prune-${METHOD}g-${KK}_f${CFOLD}"
      if [[ -f "$RESULTS/${STEM}_prune.json" ]]; then echo "  $STEM already done -> skip"; continue; fi
      echo "== $STEM (global, keep=$KEEP) =="
      python3 prune.py --tag "$CTAG" --config "$CCFG" --impl "$CIMPL" --fold "$CFOLD" \
          --method "$METHOD" --keep-ratio "$KEEP" \
          --alloc global --global-norm "$GLOBAL_NORM" --min-keep "$MIN_KEEP" \
          --skip-ft-if-within "$PRUNE_TOL" --finetune-epochs "$EPOCHS" --lr "$LR" \
        || echo "  !!! $STEM FAILED (exit $?) -- continuing"
      n=$((n+1))
    done
  done
fi

echo "############ PHASE 2: morphunet_heavy random baseline (local, keeps 0.1/0.3, fold 0) ############"
HTAG=morphunet_heavy; HCFG=heavy; HFOLD=0
for KEEP in 0.1 0.3; do
  KK=$(printf "k%02d" "$(python3 -c "print(round($KEEP*100))")")
  STEM="${HTAG}_prune-random-${KK}_f${HFOLD}"
  if [[ -f "$RESULTS/${STEM}_prune.json" ]]; then echo "  $STEM already done -> skip"; continue; fi
  echo "== $STEM (random local, keep=$KEEP) =="
  python3 prune.py --tag "$HTAG" --config "$HCFG" --impl fast --fold "$HFOLD" \
      --method random --keep-ratio "$KEEP" --alloc local --min-keep "$MIN_KEEP" \
      --finetune-epochs "$EPOCHS" --lr "$LR" --batch-size "$HEAVY_BATCH" \
    || echo "  !!! $STEM FAILED (exit $?) -- continuing"
  n=$((n+1))
done

echo "############ ALL DONE: $n runs ############"
python3 collate_prune.py || true
