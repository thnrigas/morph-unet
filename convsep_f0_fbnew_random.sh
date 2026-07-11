#!/usr/bin/env bash
# Fill the fold-0 convsep table's missing columns so it matches fold 1: fbnew + random, GLOBAL, min-keep 2,
# keeps 0.01/0.03/0.05. Runs as a 2nd GPU slot alongside convmpm (memory-gated). Regenerates the separate
# fold-0 PNG + rebuilds the deck at the end.
set -uo pipefail
cd "$(dirname "$0")"; RESULTS=results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VENV=/tmp/claude-1001/-home-Kasimatis-Documents-kasimat/f6662d64-bc71-456a-915b-ec84152c4c68/scratchpad/pptxvenv/bin/python
NEED_FREE_MIB=5000
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
n=0; t0=$(date +%s)
for METHOD in fbnew random; do
  for KEEP in 0.01 0.03 0.05; do
    KK=$(printf "k%02d" "$(python3 -c "print(round($KEEP*100))")")
    STEM="convsep_heavy_prune-${METHOD}g-${KK}_f0"
    [[ -f "$RESULTS/${STEM}_prune.json" ]] && { echo "== $STEM done -> skip =="; continue; }
    while :; do f=$(free_mib); [[ "${f:-0}" -ge "$NEED_FREE_MIB" ]] && break
      echo "   [$STEM] wait GPU headroom (free ${f})"; sleep 30; done
    echo "## $STEM (keep=$KEEP mk=2) $(date '+%T')"
    python3 prune.py --tag convsep_heavy --config heavy --impl convsep --fold 0 \
        --method "$METHOD" --keep-ratio "$KEEP" --alloc global --global-norm max --min-keep 2 \
        --skip-ft-if-within 0.015 --finetune-epochs 80 --lr 5e-5 \
      || echo "  !!! $STEM FAILED -- continuing"
    n=$((n+1))
    python3 slides/figs/prune_tables.py --convsep-fold0 >/dev/null 2>&1 || true
  done
done
echo "############ convsep f0 fbnew+random DONE: $n runs in $(( ($(date +%s)-t0)/60 )) min ############"
python3 collate_prune.py || true
python3 slides/figs/prune_tables.py --convsep-fold0 >/dev/null 2>&1 || true
"$VENV" slides/figs/build_deck.py || true
echo "############ deck rebuilt $(date '+%F %T') ############"
