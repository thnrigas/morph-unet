#!/usr/bin/env bash
#
# Resume the attention training after it was paused so the prune sweep could finish deep +
# bottleneck on the full GPU. Waits until all 12 extreme-global bottleneck jobs (fold 2,
# keeps 0.01/0.03/0.05 x l1x1/lin/act/fb) have written their _prune.json, then trains the
# attention arms in the SAME priority order as before, overlapping the heavy prune tail.
#
# Correctness vs the old driver: a fold is DONE only when its _scores.json exists (written at
# the very end). _best.pth appears mid-training, so we do NOT use it as the skip marker.
# --resume continues morphattn f0 from its _last.pth (epoch ~143) with optimizer/scheduler
# state intact; folds that never started just train fresh (no _last.pth -> --resume is a no-op).
#
set -uo pipefail
cd "$(dirname "$0")"
RESULTS=results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- GPU second-slot mutex + headroom gate ------------------------------------------------
# The 16 GB card holds at most TWO of these jobs. The heavy sweep is always job-1 (unlocked);
# attention and fbnew share ONE second slot via this mkdir-mutex so we never get three heavy
# jobs at once (that OOM'd morphattn on 2026-07-09). acquire also waits for real headroom so a
# job never launches into a card the heavy fine-tune is mid-growing on.
LOCK="$(pwd)/gpu_slot2.lock"; WANT="$(pwd)/attn.want"; NEED_FREE_MIB=4000
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
acquire(){   # $1=label  $2=prio (1 -> raise the attn.want flag so fbnew yields the slot to us)
  [[ "${2:-0}" == "1" ]] && : > "$WANT"
  while :; do
    f=$(free_mib)
    if [[ "${f:-0}" -ge "$NEED_FREE_MIB" ]] && mkdir "$LOCK" 2>/dev/null; then break; fi
    echo "   [$1] waiting for GPU slot (free ${f} MiB, lock $( [[ -d $LOCK ]] && echo held || echo free)) $(date '+%T')"
    sleep 30
  done
  rm -f "$WANT"
  echo "   [$1] acquired GPU slot (free ${f} MiB) $(date '+%T')"
}
release(){ rmdir "$LOCK" 2>/dev/null || true; }
trap 'release; rm -f "$WANT"' EXIT

declare -A FLAGS=(
  # --batch-size 12 (down from the default 24) so an attention net co-fits with the heavy prune
  # fine-tune (~7 GB) on the 16 GB card; at 24 the pair OOM'd. morphattn f0 is a near-converged
  # RESUME (best kept), so the smaller-batch tail is harmless; linattn f0 trains fresh at 12.
  [unet_morphattn]="--morph-unet none --morph-attn --morph-k 3 --batch-size 12"
  [unet_linattn]="--lin-attn --lin-attn-heads 4 --batch-size 12"
)

echo "############ ATTENTION resume: waiting for bottleneck sweep to finish $(date '+%F %T') ############"
while :; do
  n=$(ls "$RESULTS"/mpm_bottleneck_prune-*g-k01_f2_prune.json 2>/dev/null | wc -l)
  [[ "$n" -ge 4 ]] && break        # trimmed sweep: bottleneck now runs keep 0.01 only -> 4 jsons
  sleep 120
done
echo "############ bottleneck done ($n/12) -> resuming attention $(date '+%F %T') ############"

train_one() {   # $1=tag  $2=fold  $3=prio (1 -> priority slot; used for the fold-0 arms)
  local TAG=$1 FOLD=$2 STEM
  STEM="${TAG}_f${FOLD}"
  if [[ -f "$RESULTS/${STEM}_scores.json" ]]; then echo "== $STEM already COMPLETE (scores.json) -> skip =="; return 0; fi
  acquire "$STEM" "${3:-0}"
  echo "############################################################"
  echo "## TRAIN $STEM  (--resume)   $(date '+%F %T')"
  echo "############################################################"
  python3 train_eval.py --tag "$TAG" --fold "$FOLD" --resume ${FLAGS[$TAG]} \
    || echo "!!! $STEM FAILED (exit $?) -- continuing"
  release
}

# fold 0 of each gets PRIORITY (prio=1 -> fbnew yields the shared slot); remaining folds share fairly
train_one unet_morphattn 0 1
train_one unet_linattn   0 1
train_one unet_morphattn 1
# morphattn f2 dropped per user (2026-07-09): only linear-attention folds remain
# train_one unet_morphattn 2
train_one unet_linattn   1
train_one unet_linattn   2

echo "############ ATTENTION training DONE $(date '+%F %T') ############"
python3 train_eval.py --fold-mean unet_morphattn || true
python3 train_eval.py --fold-mean unet_linattn   || true
