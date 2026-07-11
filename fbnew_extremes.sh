#!/usr/bin/env bash
#
# fb-new pruning at the two remaining extreme keeps (0.03 and 0.05), GLOBAL allocation, for all five
# models. fb-new already ran at keep 0.01; this fills the 0.03/0.05 cells so the fbnew column in the
# per-model prune tables matches the fb/act/lin columns across the whole extreme range.
#
# Config: min-keep = 2, the value that produced ALL the good fbnew results (fbnew k01 ran at mk=2:
# deep 0.432, bottleneck 0.414, heavy's best-at-extreme 0.169; deep@mk2 even edged the base schemes at
# mk=4). Keeping mk=2 across 0.01/0.03/0.05 makes the fbnew column a clean controlled series (only keep
# varies, floor fixed) and is the more aggressive setting. Norm=max, 80 fine-tune epochs, lr 5e-5,
# skip-ft-if-within 0.015 -- identical to the fbnew k01 run.
#
# Scheduling: the only other live job is linattn (batch-12, ~5 GB). These prunes fit a second slot on the
# 16 GB card, so each job just waits for real headroom (poll) and then launches; jobs run one at a time
# within THIS driver, giving at most 2 concurrent GPU jobs. Resume-aware: an existing _prune.json is skipped.
#
set -uo pipefail
cd "$(dirname "$0")"
RESULTS=results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
KEEPS="0.03 0.05"; MINKEEP=2; GNORM=max; EPOCHS=80; LR=5e-5; TOL=0.015
NEED_FREE_MIB=5000            # don't launch a job unless the GPU has this much free

# tag|config|impl|fold|extra
JOBS=(
  "mpm_full_l2|full_l2|fast|0|"
  "convsep_heavy|heavy|convsep|1|"
  "mpm_deep|deep|fast|2|"
  "mpm_bottleneck|bottleneck|fast|2|"
  "morphunet_heavy|heavy|fast|0|--batch-size 12"
)

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
acquire(){   # $1 = label -- wait for real headroom before taking the shared second slot
  while :; do
    f=$(free_mib)
    [[ "${f:-0}" -ge "$NEED_FREE_MIB" ]] && break
    echo "   [$1] waiting for GPU headroom (free ${f} MiB < ${NEED_FREE_MIB}) $(date '+%T')"
    sleep 30
  done
  echo "   [$1] launching (free ${f} MiB) $(date '+%T')"
}

echo "############ fb-new extremes (k03/k05): START $(date '+%F %T') ############"
n=0; t0=$(date +%s)
for KEEP in $KEEPS; do
  KK=$(printf "k%02d" "$(python3 -c "print(round($KEEP*100))")")
  for spec in "${JOBS[@]}"; do
    IFS='|' read -r TAG CFG IMPL FOLD EXTRA <<< "$spec"
    STEM="${TAG}_prune-fbnewg-${KK}_f${FOLD}"
    if [[ -f "$RESULTS/${STEM}_prune.json" ]]; then echo "== $STEM already done -> skip =="; continue; fi
    acquire "$STEM"
    echo "############################################################"
    echo "## fb-new $STEM  (keep=$KEEP min-keep=$MINKEEP)  $(date '+%F %T')"
    echo "############################################################"
    python3 prune.py --tag "$TAG" --config "$CFG" --impl "$IMPL" --fold "$FOLD" \
        --method fbnew --keep-ratio "$KEEP" \
        --alloc global --global-norm "$GNORM" --min-keep "$MINKEEP" \
        --skip-ft-if-within "$TOL" --finetune-epochs "$EPOCHS" --lr "$LR" $EXTRA \
      || echo "  !!! $STEM FAILED (exit $?) -- continuing"
    n=$((n+1))
  done
done
echo "############ fb-new extremes DONE: $n runs in $(( ($(date +%s)-t0)/60 )) min $(date '+%F %T') ############"
python3 collate_prune.py || true
python3 slides/figs/prune_tables.py || true
