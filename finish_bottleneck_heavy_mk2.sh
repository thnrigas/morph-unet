#!/usr/bin/env bash
#
# Finish the EXTREME global sweep for the two models that had not started -- bottleneck and heavy --
# at min-keep=2. TRIMMED per the user (2026-07-09): keep-ratio 0.01 ONLY (0.03 and 0.05 dropped),
# because the smaller keeps are what matters and bottleneck isn't needed in depth. So each model
# runs its four criteria (l1x1/lin/act/fb) at keep 0.01 only -> 4 jobs each.
#
# The bottleneck l1x1g-k01 job was already IN FLIGHT under the previous driver and left running as
# an orphan when that driver was replaced, so we first WAIT for its _prune.json (never duplicate it),
# then the resume-aware loop skips it and does the rest. Appends to finish_sweep.log.
#
set -uo pipefail
cd "$(dirname "$0")"
RESULTS=results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EPOCHS=80; LR=5e-5; MIN_KEEP=2; GLOBAL_NORM=max; PRUNE_TOL=0.015
KEEPS="0.01"

# tag|config|impl|fold|criteria|extra-flags
JOBS=(
  "mpm_bottleneck|bottleneck|fast|2|l1x1 lin act fb|"
  "morphunet_heavy|heavy|fast|0|l1x1 lin act fb|--batch-size 12"
)

echo "############ mk2 k01 tail: waiting for the in-flight bottleneck l1x1g-k01 orphan $(date '+%F %T') ############"
while [[ ! -f "$RESULTS/mpm_bottleneck_prune-l1x1g-k01_f2_prune.json" ]]; do sleep 30; done
echo "############ orphan done -> bottleneck+heavy (l1x1/lin/act/fb) @ keep 0.01 mk=$MIN_KEEP START $(date '+%F %T') ############"

n=0; t0=$(date +%s)
for J in "${JOBS[@]}"; do
  IFS='|' read -r TAG CFG IMPL FOLD CRITS EXTRA <<< "$J"
  if [[ ! -f "$RESULTS/${TAG}_f${FOLD}_best.pth" ]]; then
    echo "!!! SKIP $TAG f$FOLD: base checkpoint MISSING"; continue
  fi
  echo "==== $TAG (config=$CFG impl=$IMPL fold=$FOLD) mk=$MIN_KEEP ===="
  for METHOD in $CRITS; do
    for KEEP in $KEEPS; do
      KK=$(printf "k%02d" "$(python3 -c "print(round($KEEP*100))")")
      STEM="${TAG}_prune-${METHOD}g-${KK}_f${FOLD}"
      if [[ -f "$RESULTS/${STEM}_prune.json" ]]; then echo "  $STEM already done -> skip"; continue; fi
      echo "== $STEM (global, keep=$KEEP, min_keep=$MIN_KEEP) =="
      python3 prune.py --tag "$TAG" --config "$CFG" --impl "$IMPL" --fold "$FOLD" \
          --method "$METHOD" --keep-ratio "$KEEP" \
          --alloc global --global-norm "$GLOBAL_NORM" --min-keep "$MIN_KEEP" \
          --skip-ft-if-within "$PRUNE_TOL" --finetune-epochs "$EPOCHS" --lr "$LR" $EXTRA \
        || echo "  !!! $STEM FAILED (exit $?) -- continuing"
      n=$((n+1))
    done
  done
done
echo "############ mk2 k01 tail DONE: $n runs in $(( ($(date +%s)-t0)/60 )) min $(date '+%F %T') ############"
python3 collate_prune.py || true
