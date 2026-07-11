#!/usr/bin/env bash
#
# EXTREME global-allocation sweep: how far can each model be pruned before it truly breaks?
# criteria l1x1/lin/act/fb (NO morph, NO random) x global x keeps {0.01, 0.03, 0.05}.
# convsep is linear -> l1x1 (morphological SE-norm) does not apply, so lin/act/fb only.
#
# Ordered CHEAPEST-first (full_l2, convsep, deep) so usable low-keep curve points land early;
# the slow models (bottleneck 18M, heavy needs batch 12) run LAST.
# Waits for the current prune_convsep_then_heavy.sh driver to finish first. Resume-aware.
# Logs to finish_sweep.log so the existing monitor picks up transitions.
#
set -uo pipefail
cd "$(dirname "$0")"
RESULTS=results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EPOCHS=80; LR=5e-5; MIN_KEEP=4; GLOBAL_NORM=max; PRUNE_TOL=0.015
KEEPS="0.01 0.03 0.05"

# tag|config|impl|fold|criteria|extra-flags   (cheapest first)
JOBS=(
  "mpm_full_l2|full_l2|fast|0|l1x1 lin act fb|"
  "convsep_heavy|heavy|convsep|1|lin act fb|"
  "mpm_deep|deep|fast|2|l1x1 lin act fb|"
  "mpm_bottleneck|bottleneck|fast|2|l1x1 lin act fb|"
  "morphunet_heavy|heavy|fast|0|l1x1 lin act fb|--batch-size 12"
)

echo "############ EXTREME sweep: waiting for prune_convsep_then_heavy.sh to finish ############"
while pgrep -f "prune_convsep_then_heavy[.]sh" >/dev/null 2>&1; do sleep 60; done
echo "############ EXTREME global sweep START (keeps $KEEPS) ############"

n=0; t0=$(date +%s)
for J in "${JOBS[@]}"; do
  IFS='|' read -r TAG CFG IMPL FOLD CRITS EXTRA <<< "$J"
  if [[ ! -f "$RESULTS/${TAG}_f${FOLD}_best.pth" ]]; then
    echo "!!! SKIP $TAG f$FOLD: base checkpoint MISSING"; continue
  fi
  echo "==== $TAG (config=$CFG impl=$IMPL fold=$FOLD) ===="
  for METHOD in $CRITS; do
    for KEEP in $KEEPS; do
      KK=$(printf "k%02d" "$(python3 -c "print(round($KEEP*100))")")
      STEM="${TAG}_prune-${METHOD}g-${KK}_f${FOLD}"
      if [[ -f "$RESULTS/${STEM}_prune.json" ]]; then echo "  $STEM already done -> skip"; continue; fi
      echo "== $STEM (global, keep=$KEEP) =="
      python3 prune.py --tag "$TAG" --config "$CFG" --impl "$IMPL" --fold "$FOLD" \
          --method "$METHOD" --keep-ratio "$KEEP" \
          --alloc global --global-norm "$GLOBAL_NORM" --min-keep "$MIN_KEEP" \
          --skip-ft-if-within "$PRUNE_TOL" --finetune-epochs "$EPOCHS" --lr "$LR" $EXTRA \
        || echo "  !!! $STEM FAILED (exit $?) -- continuing"
      n=$((n+1))
    done
  done
done
echo "############ EXTREME sweep DONE: $n runs in $(( ($(date +%s)-t0)/60 )) min ############"
python3 collate_prune.py || true
