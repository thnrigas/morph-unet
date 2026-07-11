#!/usr/bin/env bash
#
# fb-new pruning at keep 0.01, GLOBAL allocation, for all five models. fb-new = the data-driven
# "act" output-contribution BUT the per-channel activation mean is taken only over feature-map
# positions whose receptive field covers a foreground (vessel/tumour) voxel; the foreground mask is
# matched to each unit's resolution, so a decoder unit inherits its mirror encoder layer's field.
#
# Scheduling: the user asked to start this AFTER the bottleneck pruning finishes. But at that point
# the sweep's heavy tail and the resumed attention training may still hold the 16 GB GPU, so each job
# only launches once there is enough free memory (poll), and jobs run one at a time. Resume-aware:
# a model whose _prune.json already exists is skipped.
#
set -uo pipefail
cd "$(dirname "$0")"
RESULTS=results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
KEEP=0.01; KK=k01; MINKEEP=2; GNORM=max; EPOCHS=80; LR=5e-5; TOL=0.015
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

# --- GPU second-slot mutex (shared with attn_resume.sh) -----------------------------------
# heavy sweep is job-1 (unlocked); attention and fbnew share ONE second slot so we never run
# three heavy jobs at once. acquire also waits for real headroom before taking the slot.
# Heavy sweep is DONE, so there is no longer a "job-1" to reserve the card. The remaining jobs
# (batch-12 attention ~5 GB, fbnew prunes 2-7 GB) fit TWO at a time in 16 GB, and each driver runs
# one job at a time, so plain per-launch memory gating already caps concurrency at 2. Drop the
# single-slot mutex (it would needlessly serialize fbnew behind every attention fold) and just wait
# for real headroom before launching.
acquire(){   # $1 = label
  while :; do
    f=$(free_mib)
    [[ "${f:-0}" -ge "$NEED_FREE_MIB" ]] && break
    echo "   [$1] waiting for GPU headroom (free ${f} MiB < ${NEED_FREE_MIB}) $(date '+%T')"
    sleep 30
  done
  echo "   [$1] launching (free ${f} MiB) $(date '+%T')"
}
release(){ :; }

echo "############ fb-new k01: waiting for bottleneck sweep to finish $(date '+%F %T') ############"
while :; do
  n=$(ls "$RESULTS"/mpm_bottleneck_prune-*g-k01_f2_prune.json 2>/dev/null | wc -l)
  [[ "$n" -ge 4 ]] && break        # trimmed sweep: bottleneck now runs keep 0.01 only -> 4 jsons
  sleep 120
done
echo "############ bottleneck done ($n/12) -> fb-new runs eligible $(date '+%F %T') ############"

for spec in "${JOBS[@]}"; do
  IFS='|' read -r TAG CFG IMPL FOLD EXTRA <<< "$spec"
  STEM="${TAG}_prune-fbnewg-${KK}_f${FOLD}"
  if [[ -f "$RESULTS/${STEM}_prune.json" ]]; then echo "== $STEM already done -> skip =="; continue; fi
  acquire "$STEM"                    # take the shared second slot (waits for lock + headroom)
  echo "############################################################"
  echo "## fb-new $STEM   $(date '+%F %T')"
  echo "############################################################"
  python3 prune.py --tag "$TAG" --config "$CFG" --impl "$IMPL" --fold "$FOLD" \
      --method fbnew --keep-ratio "$KEEP" \
      --alloc global --global-norm "$GNORM" --min-keep "$MINKEEP" \
      --skip-ft-if-within "$TOL" --finetune-epochs "$EPOCHS" --lr "$LR" $EXTRA \
    || echo "  !!! $STEM FAILED (exit $?) -- continuing"
  release
done

echo "############ fb-new k01 DONE $(date '+%F %T') ############"
python3 collate_prune.py || true
