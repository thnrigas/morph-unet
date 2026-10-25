#!/usr/bin/env bash
#
# Skip-attention grid on MSD Task08: baseline + {linear, morph} x {cross, self}, 5 folds.
# =========================================================================================
# All arms share the residual U-Net backbone (--morph-unet none) and the config.py defaults
# (batch 12, val-batch 4, seed 42). Only the skip gate differs:
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
# from its _last.pth (--resume), and a failed run is reported and the grid moves on.
# Logs: logs/attn_grid/<tag>_f<fold>.log (one per run).
#
set -uo pipefail
cd "$(dirname "$0")/.."

MODE=${1:-}
GPU=${2:-}
FOLDS=(0 1 2 3 4)
ARMS=(grid_baseline grid_lin_cross grid_lin_self grid_morph_self grid_morph_cross)
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
mkdir -p "$LOGS"

usage() { sed -n '2,29p' "$0"; exit 1; }

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
    SUMMARY=()          # one line per arm, same order as ARMS
    total=0             # sum of seconds/epoch over the arms that ran
    for ARM in "${ARMS[@]}"; do
      TAG="smoke_${ARM#grid_}"; LOG="$LOGS/${TAG}_f0.log"
      echo "== $TAG ($(date '+%T'))"
      python3 -u train_eval.py --tag "$TAG" --fold 0 $(flags "$ARM") \
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
    FAILED=()
    for F in "${FOLDS[@]}"; do
      for ARM in "${ARMS[@]}"; do
        STEM="${ARM}_f${F}"; LOG="$LOGS/${STEM}.log"
        if [[ -f "$RESULTS/${STEM}_scores.json" ]]; then
          echo "== $STEM done -> skip"; continue
        fi
        [[ -f "$RESULTS/${STEM}_last.pth" ]] && note=" (resuming from _last.pth)" || note=""
        echo "############ $STEM$note  $(date '+%F %T')"
        python3 -u train_eval.py --tag "$ARM" --fold "$F" --resume $(flags "$ARM") 2>&1 | tee -a "$LOG"
        rc=${PIPESTATUS[0]}
        if [[ $rc -ne 0 ]]; then
          echo "!!! $STEM FAILED (exit $rc), see $LOG -- continuing"; FAILED+=("$STEM")
        fi
      done
    done
    echo "############ GRID DONE  $(date '+%F %T') ############"
    [[ ${#FAILED[@]} -gt 0 ]] && echo "!!! failed runs: ${FAILED[*]}  (re-run the same command to retry them)"
    MEANS=()
    for ARM in "${ARMS[@]}"; do
      python3 train_eval.py --fold-mean "$ARM" > "$LOGS/${ARM}_mean.log" 2>&1 && MEANS+=("${ARM}_mean_scores.json")
    done
    [[ ${#MEANS[@]} -gt 0 ]] && python3 train_eval.py --compare "${MEANS[@]}" | tee "$LOGS/compare.log"
    ;;

  status)
    printf "%-18s" "arm \\ fold"; for F in "${FOLDS[@]}"; do printf " %-8s" "f$F"; done; echo
    for ARM in "${ARMS[@]}"; do
      printf "%-18s" "$ARM"
      for F in "${FOLDS[@]}"; do
        S="$RESULTS/${ARM}_f${F}"
        if   [[ -f "${S}_scores.json" ]]; then st="done"
        elif [[ -f "${S}_last.pth"    ]]; then st="partial"
        else st="-"; fi
        printf " %-8s" "$st"
      done; echo
    done
    ;;

  *) usage ;;
esac
