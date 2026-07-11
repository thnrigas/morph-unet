#!/usr/bin/env bash
#
# RANDOM-control pruning for the linear Compact-CNN (convsep_heavy) at the extreme keeps 0.01/0.03/0.05,
# GLOBAL allocation, min-keep = 2. The existing "random" column in the prune tables is LOCAL; this adds
# the GLOBAL random control so the fb/fbnew/lin/act global schemes have a matched random baseline at the
# extremes (global random can starve whole layers down to the min-keep=2 floor, unlike uniform local).
#
# Scheduling / OOM-safety: the 16 GB card already runs TWO jobs (the fbnew k03/k05 sweep + the
# lin-attn gamma=0.5 fold). A third heavy job would OOM (as happened on 2026-07-09). So we WAIT until the
# lin-attn fold finishes (its _scores.json appears), dropping the card to one job, then run these three as
# the second slot alongside the fbnew sweep -- each launches only when there is real headroom, and this
# driver runs them one at a time, so at most 2 GPU jobs coexist. Resume-aware: existing _prune.json skipped.
#
set -uo pipefail
cd "$(dirname "$0")"
RESULTS=results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
KEEPS="0.01 0.03 0.05"; MINKEEP=2; GNORM=max; EPOCHS=80; LR=5e-5; TOL=0.015
NEED_FREE_MIB=5000
TAG=convsep_heavy; CFG=heavy; IMPL=convsep; FOLD=1

free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

echo "############ random-global convsep: waiting for lin-attn g05 fold to finish $(date '+%F %T') ############"
while [[ ! -f "$RESULTS/unet_linattn_g05_f0_scores.json" ]]; do sleep 60; done
echo "############ lin-attn g05 done -> random-global runs eligible $(date '+%F %T') ############"

n=0; t0=$(date +%s)
for KEEP in $KEEPS; do
  KK=$(printf "k%02d" "$(python3 -c "print(round($KEEP*100))")")
  STEM="${TAG}_prune-randomg-${KK}_f${FOLD}"
  if [[ -f "$RESULTS/${STEM}_prune.json" ]]; then echo "== $STEM already done -> skip =="; continue; fi
  while :; do f=$(free_mib); [[ "${f:-0}" -ge "$NEED_FREE_MIB" ]] && break
    echo "   [$STEM] waiting for GPU headroom (free ${f} MiB < ${NEED_FREE_MIB}) $(date '+%T')"; sleep 30; done
  echo "############################################################"
  echo "## random-global $STEM  (keep=$KEEP min-keep=$MINKEEP)  $(date '+%F %T')"
  echo "############################################################"
  python3 prune.py --tag "$TAG" --config "$CFG" --impl "$IMPL" --fold "$FOLD" \
      --method random --keep-ratio "$KEEP" \
      --alloc global --global-norm "$GNORM" --min-keep "$MINKEEP" \
      --skip-ft-if-within "$TOL" --finetune-epochs "$EPOCHS" --lr "$LR" \
    || echo "  !!! $STEM FAILED (exit $?) -- continuing"
  n=$((n+1))
done
echo "############ random-global convsep DONE: $n runs in $(( ($(date +%s)-t0)/60 )) min $(date '+%F %T') ############"
python3 collate_prune.py || true
python3 slides/figs/prune_tables.py || true
