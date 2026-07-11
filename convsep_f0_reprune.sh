#!/usr/bin/env bash
#
# Re-prune the linear Compact-CNN (convsep_heavy) on FOLD 0 as well as fold 1. Fold 1 was the weakest
# fold (macro 0.368; Tumour Dice only 0.253), so its prune column looks artificially bad. Fold 0
# (macro 0.451) matches the fold the other single-fold models were pruned on (mpm_full_l2, morphunet_heavy).
#
# We KEEP both: the existing fold-1 table (prune_convsep.png) is untouched, and this produces a SEPARATE
# fold-0 table (prune_convsep_f0.png) that build_deck.py adds as an extra slide -> the deck shows both.
#
# Scope: lin/act/fb global at keeps 0.01/0.03/0.05/0.1/0.5, min-keep 4 (identical to the fold-1 runs). 15
# runs. Writes *_f0_prune.json. OOM-safety: waits until BOTH current prune sweeps finish, then runs as a
# single job (may overlap the queued g05 attention folds -> at most 2 GPU jobs).
#
set -uo pipefail
cd "$(dirname "$0")"
RESULTS=results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VENV=/tmp/claude-1001/-home-Kasimatis-Documents-kasimat/f6662d64-bc71-456a-915b-ec84152c4c68/scratchpad/pptxvenv/bin/python
TAG=convsep_heavy; CFG=heavy; IMPL=convsep; FOLD=0
SCHEMES="lin act fb"; KEEPS="0.01 0.03 0.05 0.1 0.5"
MINKEEP=4; GNORM=max; EPOCHS=80; LR=5e-5; TOL=0.015
TBL=slides/figs/prune_tables.py

echo "############ convsep f0 reprune: waiting for current prune sweeps to finish $(date '+%F %T') ############"
while pgrep -f "fbnew_extremes.sh" >/dev/null 2>&1 || pgrep -f "random_convsep.sh" >/dev/null 2>&1; do
  sleep 60
done
echo "############ sweeps done -> convsep f0 reprune START $(date '+%F %T') ############"

n=0; t0=$(date +%s)
for METHOD in $SCHEMES; do
  for KEEP in $KEEPS; do
    KK=$(printf "k%02d" "$(python3 -c "print(round($KEEP*100))")")
    STEM="${TAG}_prune-${METHOD}g-${KK}_f${FOLD}"
    if [[ -f "$RESULTS/${STEM}_prune.json" ]]; then echo "== $STEM already done -> skip =="; continue; fi
    echo "## reprune $STEM (keep=$KEEP mk=$MINKEEP) $(date '+%F %T')"
    python3 prune.py --tag "$TAG" --config "$CFG" --impl "$IMPL" --fold "$FOLD" \
        --method "$METHOD" --keep-ratio "$KEEP" \
        --alloc global --global-norm "$GNORM" --min-keep "$MINKEEP" \
        --skip-ft-if-within "$TOL" --finetune-epochs "$EPOCHS" --lr "$LR" \
      || echo "  !!! $STEM FAILED (exit $?) -- continuing"
    n=$((n+1))
    python3 "$TBL" --convsep-fold0 >/dev/null 2>&1 || true   # refresh the SEPARATE fold-0 PNG after each run
  done
done
echo "############ convsep f0 reprune DONE: $n runs in $(( ($(date +%s)-t0)/60 )) min $(date '+%F %T') ############"
python3 collate_prune.py || true
python3 "$TBL" --convsep-fold0 >/dev/null 2>&1 || true
"$VENV" slides/figs/build_deck.py || true       # rebuild deck so the extra fold-0 convsep slide appears
echo "############ convsep f0 reprune: deck rebuilt $(date '+%F %T') ############"
