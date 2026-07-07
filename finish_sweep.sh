#!/usr/bin/env bash
#
# Remaining local pruning after the main sweep (deep/bottleneck/full_l2 are done).
# =================================================================================
# PHASE A -- HEAVY, GLOBAL ONLY, keeps {0.1, 0.5} only:
#   heavy has 18 morph layers, so GLOBAL allocation (reallocates a shared budget across
#   layers) clearly beats LOCAL there -- local is dropped for heavy. We probe just the two
#   informative regimes: k10 (extreme) and k50 (half), NO 0.3/0.7 (short on time), and NO
#   escalation (we WANT both keeps to see the per-layer channel distribution at each).
#   4 criteria (l1x1/lin/act/fb) x global x {0.1,0.5} = 8 runs. No local, no random for heavy.
#   Each run records per-layer surviving widths in results/<stem>_prune.json -> shows whether
#   the budget concentrates in early / bottleneck / late layers.
#
# PHASE B -- random sanity baseline for deep/bottleneck/full_l2 (local, 4 ratios each = 12 runs).
#   Heavy is intentionally excluded. Ctrl-C after Phase A if you only care about heavy.
#
set -uo pipefail
cd "$(dirname "$0")"
RESULTS=results
EPOCHS=80; LR=5e-5; MIN_KEEP=4; GLOBAL_NORM=max; PRUNE_TOL=0.015
t0=$(date +%s); n=0

echo "############ PHASE A: heavy global, keeps 0.1 & 0.5 (8 runs) ############"
for METHOD in l1x1 lin act fb; do
  for KEEP in 0.1 0.5; do
    KK=$(printf "k%02d" "$(python -c "print(round($KEEP*100))")")
    STEM="morphunet_heavy_prune-${METHOD}g-${KK}_f0"
    if [[ -f "$RESULTS/${STEM}_prune.json" ]]; then
      echo "  $STEM already done -> skip"; continue
    fi
    echo "== $STEM (global, keep=$KEEP) =="
    python prune.py --tag morphunet_heavy --config heavy --impl fast --fold 0 \
        --method "$METHOD" --keep-ratio "$KEEP" \
        --alloc global --global-norm "$GLOBAL_NORM" --min-keep "$MIN_KEEP" \
        --skip-ft-if-within "$PRUNE_TOL" --finetune-epochs "$EPOCHS" --lr "$LR" \
      || echo "  !!! $STEM FAILED (exit $?) -- continuing"
    n=$((n+1))
  done
done

echo "############ PHASE B: random baseline, deep/bottleneck/full_l2 (local) ############"
declare -A CFG=( [mpm_deep]=deep [mpm_bottleneck]=bottleneck [mpm_full_l2]=full_l2 )
declare -A FLD=( [mpm_deep]=2   [mpm_bottleneck]=2         [mpm_full_l2]=0 )
for TAG in mpm_deep mpm_bottleneck mpm_full_l2; do
  for KEEP in 0.1 0.3 0.5 0.7; do
    KK=$(printf "k%02d" "$(python -c "print(round($KEEP*100))")")
    STEM="${TAG}_prune-random-${KK}_f${FLD[$TAG]}"
    if [[ -f "$RESULTS/${STEM}_prune.json" ]]; then
      echo "  $STEM already done -> skip"; continue
    fi
    echo "== $STEM (random local, keep=$KEEP) =="
    python prune.py --tag "$TAG" --config "${CFG[$TAG]}" --impl fast --fold "${FLD[$TAG]}" \
        --method random --keep-ratio "$KEEP" --alloc local --min-keep "$MIN_KEEP" \
        --finetune-epochs "$EPOCHS" --lr "$LR" \
      || echo "  !!! $STEM FAILED (exit $?) -- continuing"
    n=$((n+1))
  done
done

echo "=================================================================="
echo "FINISH DONE: $n runs in $(( ($(date +%s)-t0)/60 )) min."
python collate_prune.py || true
