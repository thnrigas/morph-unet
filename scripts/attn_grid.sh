#!/usr/bin/env bash
#
# Skip-attention grid on MSD Task08: baseline + {linear, morph} x {cross, self}, 5 folds.
# =========================================================================================
# All arms share the residual U-Net backbone (--morph-unet none) and the config.py defaults
# (patch 128, batch 12, val-batch 4, seed 42; see the settings below). Only the skip gate differs:
#
#   grid_baseline     no gate
#   grid_lin_cross    linear attention, query = upsampled decoder feature, gamma0 = 0.5
#   grid_lin_self     linear attention, query = the skip itself,          gamma0 = 0.5
#   grid_morph_self   2*sigmoid(W[tophat(x); bothat(x)]) * x, identity init, k = 3
#   grid_morph_cross  2*sigmoid(W[tophat(g); bothat(g)]) * x, identity init, k = 3 (g = decoder)
#
# The morph gates use the fixed morphology (neutral border padding, reflected SE): the defaults.
#
# Usage (inside tmux, with `conda activate morph-unet`, from anywhere):
#   bash scripts/attn_grid.sh smoke <gpu-id>    # 3 epochs, fold 0, every arm; Dice-only test
#   bash scripts/attn_grid.sh grid  <gpu-id>    # 5 arms x 5 folds, fold by fold; full metrics
#   bash scripts/attn_grid.sh status            # which runs are done / partial / missing
#
# <gpu-id> is the nvidia-smi index of ONE free GPU (one GPU per student). The script refuses a GPU
# that already has more than 1 GB in use; FORCE=1 overrides that check.
#
# The grid is restartable: a run whose _scores.json exists is skipped, an interrupted run continues
# from its _last.pth (--resume), and a failed run is retried (up to MAX_TRIES attempts), then
# reported, and the grid moves on.
# Logs: logs/attn_grid/<tag>_f<fold>.log (one per run).
#
# Settings (environment variables):
#   PATIENCE=100     early-stopping patience in epochs (default: config.py, 30)
#   MIN_FREE_GB=5    a run starts only with at least this much free space on the results disk
#   MAX_TRIES=3      attempts per run; each waits for disk space and a free GPU first
#   POLL=300         seconds between checks while waiting
#   MORPH_NO_CKPT=1  morph arms train without gradient checkpointing (identical maths, faster, more
#                    VRAM); if such an attempt fails (e.g. out of memory), the run's next tries go
#                    back to checkpointing
#   ARMS="baseline morph_self"   run only these arms (names without "grid_"; default: all five)
#   SEED=1           passed as --seed; adds "_s1" to the tags (default: no flag, seed 42, no suffix)
#   VAL_CASES=0      passed as --val-cases (0 = all validation volumes); adds "_fv" (0) or "_v<N>"
#                    to the tags (default: no flag, config.py's 15, no suffix)
#   LAYOUT=axial     train and test on the axial data (data/<task>/preprocessed_axial, made once by
#                    `LAYOUT=axial python3 run_preprocessing.py`); adds "_ax" to the tags
#                    (default: the original x, y, z data, no suffix)
#   PATCH=512        passed as --patch-size; adds "_p<PATCH>" to the tags (default: config.py, 128)
#   BATCH=2          passed as --batch-size; adds "_b<BATCH>" to the tags (default: config.py, 12)
#   VAL_BATCH=2      passed as --val-batch; not in the tags, it changes no result (default: 4)
# LAYOUT, PATCH, BATCH, SEED and VAL_CASES are part of the tag, so such runs never collide with the
# seed-42 grid; PATIENCE is not (the patience-100 runs continued the patience-30 ones under the same
# tags). The smoke mode takes the same variables (tags smoke_<arm><suffix>). Examples:
#   ARMS="baseline morph_self" SEED=1 VAL_CASES=0 PATIENCE=50 MORPH_NO_CKPT=1 bash scripts/attn_grid.sh grid 1
#   -> grid_baseline_s1_fv_f0 ... ; status / fold-means / compare take the same variables.
#   LAYOUT=axial PATCH=512 BATCH=2 VAL_BATCH=2 ARMS="baseline morph_self" VAL_CASES=2 bash scripts/attn_grid.sh smoke 1
#   -> smoke_baseline_ax_p512_b2_v2_f0, smoke_morph_self_ax_p512_b2_v2_f0 (VRAM and s/epoch at full slices)
# Before every run the GPU is re-checked (no compute process, < 1 GB in use; FORCE=1 skips this),
# and the script waits until both the disk and the GPU are free. Inside a run, train_eval.py saves
# checkpoints atomically and waits while less than 1 GB is free (--min-free-gb).
#
set -uo pipefail
cd "$(dirname "$0")/.."

MODE=${1:-}
GPU=${2:-}
FOLDS=(0 1 2 3 4)
ALL_ARMS=(grid_baseline grid_lin_cross grid_lin_self grid_morph_self grid_morph_cross)
if [[ -n "${ARMS:-}" ]]; then
  read -r -a _sel <<< "$ARMS"; ARMS=()
  for a in "${_sel[@]}"; do
    [[ " ${ALL_ARMS[*]} " == *" grid_${a#grid_} "* ]] || { echo "!! unknown arm '$a' (choose from: ${ALL_ARMS[*]#grid_})"; exit 1; }
    ARMS+=("grid_${a#grid_}")
  done
else
  ARMS=("${ALL_ARMS[@]}")
fi
# tag suffix and extra flags from LAYOUT / PATCH / BATCH / VAL_BATCH / SEED / VAL_CASES (empty when
# unset: the seed-42 grid's tags). config.py reads LAYOUT from the environment itself, and
# train_eval.py refuses an axial run whose tag lacks "_ax" (and the reverse)
SUFFIX=""; EXTRA=""
case "${LAYOUT:-xyz}" in
  xyz)   ;;
  axial) SUFFIX+="_ax" ;;
  *)     echo "!! LAYOUT must be xyz or axial, not '$LAYOUT'"; exit 1 ;;
esac
if [[ -n "${PATCH:-}" ]]; then SUFFIX+="_p$PATCH"; EXTRA+=" --patch-size $PATCH"; fi
if [[ -n "${BATCH:-}" ]]; then SUFFIX+="_b$BATCH"; EXTRA+=" --batch-size $BATCH"; fi
if [[ -n "${VAL_BATCH:-}" ]]; then EXTRA+=" --val-batch $VAL_BATCH"; fi
if [[ -n "${SEED:-}" ]]; then SUFFIX+="_s$SEED"; EXTRA+=" --seed $SEED"; fi
if [[ -n "${VAL_CASES:-}" ]]; then
  [[ "$VAL_CASES" == "0" ]] && SUFFIX+="_fv" || SUFFIX+="_v$VAL_CASES"
  EXTRA+=" --val-cases $VAL_CASES"
fi
flags() {   # train_eval.py flags of an arm
  case "$1" in
    grid_baseline)    echo "--morph-unet none" ;;
    grid_lin_cross)   echo "--morph-unet none --lin-attn --lin-attn-heads 4 --lin-attn-gamma-init 0.5 --lin-attn-src cross" ;;
    grid_lin_self)    echo "--morph-unet none --lin-attn --lin-attn-heads 4 --lin-attn-gamma-init 0.5 --lin-attn-src self" ;;
    grid_morph_self)  echo "--morph-unet none --morph-attn --morph-attn-src self" ;;
    grid_morph_cross) echo "--morph-unet none --morph-attn --morph-attn-src cross" ;;
  esac
}

RESULTS="results/$(python3 -c 'import config; print(config.TASK)')"
LOGS=logs/attn_grid
mkdir -p "$LOGS" "$RESULTS"

MIN_FREE_GB=${MIN_FREE_GB:-5}
MAX_TRIES=${MAX_TRIES:-3}
POLL=${POLL:-300}
PATIENCE_FLAG=${PATIENCE:+--patience $PATIENCE}

usage() { awk 'NR == 1 { next } /^#/ { print; next } { exit }' "$0"; exit 1; }   # the header comment

free_gb() {   # whole GB free on the disk that holds the results
  df -Pk "$RESULTS" | awk 'NR == 2 { printf "%d", $4 / 1048576 }'
}

gpu_busy() {   # prints why GPU $GPU is taken (memory in use / compute processes); nothing if free
  [[ "${FORCE:-0}" == "1" ]] && return
  local used uuid nproc
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
  nproc=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")
  if [[ "$used" -gt 1000 || "$nproc" -gt 0 ]]; then
    echo "${used} MiB in use, ${nproc} compute process(es)"
  fi
}

wait_ready() {   # block until the disk has MIN_FREE_GB free and GPU $GPU is free
  local free why
  while :; do
    free=$(free_gb); why=$(gpu_busy)
    [[ "$free" -ge "$MIN_FREE_GB" && -z "$why" ]] && return
    echo "   waiting: disk ${free} GB free (need ${MIN_FREE_GB})${why:+; GPU $GPU taken: $why}  $(date '+%F %T')"
    sleep "$POLL"
  done
}

check_gpu() {
  [[ -n "$GPU" ]] || { echo "!! give a GPU index: bash scripts/attn_grid.sh $MODE <gpu-id>"; exit 1; }
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU") \
    || { echo "!! nvidia-smi does not know GPU $GPU"; exit 1; }
  echo "GPU $GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader -i "$GPU")"
  if [[ "$used" -gt 1000 && "${FORCE:-0}" != "1" ]]; then
    echo "!! GPU $GPU already has ${used} MiB in use (someone else's job?). Pick a free one, or FORCE=1."
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="$GPU"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python3 -c "import torch, sys; ok = torch.cuda.is_available(); \
print('torch', torch.__version__, '| CUDA visible:', ok, '|', torch.cuda.get_device_name(0) if ok else '-'); \
sys.exit(0 if ok else 1)" || { echo "!! torch sees no GPU: is the morph-unet env active?"; exit 1; }
}

case "$MODE" in
  smoke)
    check_gpu
    echo "############ SMOKE: 3 epochs, fold 0, ${#ARMS[@]} arms  $(date '+%F %T') ############"
    echo "   tag suffix: '${SUFFIX}' | extra flags:${EXTRA:- none}"
    SUMMARY=()          # one line per arm, same order as ARMS
    total=0             # sum of seconds/epoch over the arms that ran
    for ARM in "${ARMS[@]}"; do
      TAG="smoke_${ARM#grid_}${SUFFIX}"; LOG="$LOGS/${TAG}_f0.log"
      echo "== $TAG ($(date '+%T'))"
      python3 -u train_eval.py --tag "$TAG" --fold 0 $EXTRA $(flags "$ARM") \
          --epochs 3 --val-every 3 --fast-eval > "$LOG" 2>&1
      rc=$?
      if [[ $rc -ne 0 ]]; then
        SUMMARY+=("FAILED (exit $rc), see $LOG")
        echo "   FAILED (exit $rc). Last lines of $LOG:"; tail -n 15 "$LOG" | sed 's/^/   | /'
        continue
      fi
      sec=$(grep -oE '\([0-9.]+s/ep\)' "$LOG" | tail -1 | grep -oE '[0-9.]+')
      vram=$(grep -oE 'vram train=[^ ]+( val=[^ ]+)?' "$LOG" | tail -1)
      SUMMARY+=("${sec:-?} s/epoch | ${vram:-vram n/a}")
      [[ -n "$sec" ]] && total=$(python3 -c "print($total + $sec)")
      echo "   ok: ${SUMMARY[${#SUMMARY[@]}-1]}"
    done
    echo
    echo "############ SMOKE SUMMARY ############"
    for i in "${!ARMS[@]}"; do
      printf "  %-18s %s\n" "${ARMS[$i]}" "${SUMMARY[$i]}"
    done
    echo
    echo "  Time for ONE fold of all arms, per 100 epochs each: $(python3 -c "print(f'{$total*100/3600:.1f} h')")"
    echo "  Full grid (5 folds) at 100 / 200 / 400 epochs per run (early stopping usually ends sooner):"
    python3 -c "t=$total; print('    ' + ' / '.join(f'{t*e*5/3600/24:.1f} days' for e in (100, 200, 400)))"
    echo "  The smoke runs wrote results/*/smoke_* and $LOGS/smoke_*; delete them when done."
    ;;

  grid)
    check_gpu
    echo "############ GRID: ${#ARMS[@]} arms x ${#FOLDS[@]} folds, fold by fold  $(date '+%F %T') ############"
    echo "   patience: ${PATIENCE:-config default} | min free disk: ${MIN_FREE_GB} GB | tries per run: ${MAX_TRIES}"
    echo "   arms: ${ARMS[*]} | tag suffix: '${SUFFIX}' | extra flags:${EXTRA:- none}"
    FAILED=()
    for F in "${FOLDS[@]}"; do
      for ARM in "${ARMS[@]}"; do
        TAG="${ARM}${SUFFIX}"; STEM="${TAG}_f${F}"; LOG="$LOGS/${STEM}.log"
        if [[ -f "$RESULTS/${STEM}_scores.json" ]]; then
          echo "== $STEM done -> skip"; continue
        fi
        nockpt=""
        [[ "${MORPH_NO_CKPT:-0}" == "1" && "$ARM" == grid_morph_* ]] && nockpt="--morph-no-checkpoint"
        for (( try = 1; try <= MAX_TRIES; try++ )); do
          wait_ready
          [[ -f "$RESULTS/${STEM}_last.pth" ]] && note=" (resuming from _last.pth)" || note=""
          echo "############ $STEM$note  try $try/$MAX_TRIES${nockpt:+, no checkpointing}  $(date '+%F %T')"
          python3 -u train_eval.py --tag "$TAG" --fold "$F" --resume $PATIENCE_FLAG $EXTRA $nockpt $(flags "$ARM") \
            2>&1 | tee -a "$LOG"
          rc=${PIPESTATUS[0]}
          [[ $rc -eq 0 ]] && break
          echo "!!! $STEM failed (exit $rc), see $LOG"
          [[ -n "$nockpt" ]] && { nockpt=""; echo "   next tries use gradient checkpointing again"; }
        done
        if [[ $rc -ne 0 ]]; then
          echo "!!! $STEM FAILED $MAX_TRIES times -- continuing"; FAILED+=("$STEM")
        fi
      done
    done
    echo "############ GRID DONE  $(date '+%F %T') ############"
    [[ ${#FAILED[@]} -gt 0 ]] && echo "!!! failed runs: ${FAILED[*]}  (re-run the same command to retry them)"
    MEANS=()
    for ARM in "${ARMS[@]}"; do
      TAG="${ARM}${SUFFIX}"
      python3 train_eval.py --fold-mean "$TAG" > "$LOGS/${TAG}_mean.log" 2>&1 && MEANS+=("${TAG}_mean_scores.json")
    done
    [[ ${#MEANS[@]} -gt 0 ]] && python3 train_eval.py --compare "${MEANS[@]}" | tee "$LOGS/compare${SUFFIX}.log"
    ;;

  status)
    printf "%-36s" "arm \\ fold"; for F in "${FOLDS[@]}"; do printf " %-8s" "f$F"; done; echo
    for ARM in "${ARMS[@]}"; do
      TAG="${ARM}${SUFFIX}"
      printf "%-36s" "$TAG"
      for F in "${FOLDS[@]}"; do
        S="$RESULTS/${TAG}_f${F}"
        if   [[ -f "${S}_scores.json" ]]; then st="done"
        elif [[ -f "${S}_last.pth"    ]]; then st="partial"
        else st="-"; fi
        printf " %-8s" "$st"
      done; echo
    done
    ;;

  *) usage ;;
esac
