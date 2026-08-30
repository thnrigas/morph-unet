#!/usr/bin/env bash
#
# Self-contained MC-Dropout UNCERTAINTY + CALIBRATION experiment on Task04
# Hippocampus (GPU VM, e.g. L4). Counterpart of run_all_spleen.sh.
#
# You ALREADY have the trained weights from a previous run, so this script SKIPS
# training when a checkpoint exists. Put your weights in a folder and pass it as
# WEIGHTS_DIR -> the script copies them into results/ before the (skipped) training.
# Expected filenames (per fold F): mcdropout_fF_best.pth, mcdropout_cal_fF_best.pth
#
# Bootstrap on the VM (grabs just this one script from the monorepo):
#   curl -O https://raw.githubusercontent.com/thnrigas/morph-unet/main/run/run_all_hippocampus.sh
#   WEIGHTS_DIR=$HOME/hippo-weights nohup bash run_all_hippocampus.sh &
#   tail -f ~/hippo-run/run_*.log
# Copy results off in the morning: ~/hippo-run/repo/results/
#
# torch/numpy are assumed preinstalled (DL VM) and are NOT reinstalled.

set -uo pipefail   # NOT -e: a single step failure must not throw away the rest.

# ============================ CONFIG (edit these) ============================
REPO_URL="${REPO_URL:-https://github.com/thnrigas/morph-unet.git}"
BRANCH="main"                               # branch that holds the code
WORKDIR="${WORKDIR:-$HOME/hippo-run}"       # where the repo + logs live
TASK="Task04_Hippocampus"
DATA_TAR_URL="https://msd-for-monai.s3-us-west-2.amazonaws.com/${TASK}.tar"

WEIGHTS_DIR="${WEIGHTS_DIR:-}"              # folder with your *_best.pth (copied into results/)
FOLDS="0"                                   # "0" ; set "0 1 2 3 4" if you have all folds' weights

# model / GPU knobs (Hippocampus is tiny 64x64)
PATCH=64
BATCH=8
WORKERS=0                 # 0 = single-process loading. Hippocampus mc_common applies workers to
                          # ALL loaders (incl. val/test); 0 avoids the CUDA-fork abort and is plenty
                          # fast at 64x64. (Only matters for the eval loaders here; training is skipped.)
DROPOUT=0.4
MC=30                     # MC stochastic passes T
# training knobs (used ONLY if a checkpoint is missing and we must retrain)
EPOCHS=150
PATIENCE=15
CALW=1.0
# ============================================================================

mkdir -p "$WORKDIR"
LOG="$WORKDIR/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "################ hippocampus uncertainty  $(date '+%F %T') ################"
echo "repo=$REPO_URL  branch=$BRANCH  workdir=$WORKDIR"
echo "task=$TASK folds='$FOLDS'  patch=$PATCH batch=$BATCH workers=$WORKERS mc=$MC  weights_dir='$WEIGHTS_DIR'"
echo "log -> $LOG"

run() {   # run() "label" cmd... : header + timing, CONTINUE on failure (overnight-safe)
    local label="$1"; shift
    echo ""; echo "===== [$(date '+%F %T')] $label ====="
    local t0=$SECONDS
    "$@"; local rc=$?
    echo "----- $label done in $((SECONDS - t0))s (exit $rc) -----"
    [ $rc -eq 0 ] || echo "!!! FAILED: $label (continuing) !!!"
    return 0
}

# resolve WEIGHTS_DIR to an absolute path NOW (we cd into the repo later)
if [ -n "$WEIGHTS_DIR" ]; then
    WEIGHTS_DIR="$(cd "$WEIGHTS_DIR" 2>/dev/null && pwd)" \
        || { echo "WEIGHTS_DIR not found -> aborting"; exit 1; }
fi

# ---- 1. clone (or force-update) the repo ----
#     The uncertainty code lives at the REPO ROOT (uncertainty/ + the shared packages),
#     so this is a plain clone -- no sparse checkout.
REPO_DIR="$WORKDIR/repo"
if [ -d "$REPO_DIR/.git" ]; then
    echo "repo present -> force-updating to origin/$BRANCH (code only; data/ & results/ untouched)"
    git -C "$REPO_DIR" fetch origin "$BRANCH" \
        && git -C "$REPO_DIR" checkout "$BRANCH" \
        && git -C "$REPO_DIR" reset --hard FETCH_HEAD \
        || echo "WARN: could not update; using existing checkout"
else
    git clone --filter=blob:none --branch "$BRANCH" --single-branch "$REPO_URL" "$REPO_DIR" \
        || { echo "git clone of branch '$BRANCH' failed -> aborting"; exit 1; }
fi
cd "$REPO_DIR" || { echo "cannot cd $REPO_DIR"; exit 1; }
echo "on branch: $(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD) @ $(git -C "$REPO_DIR" rev-parse --short HEAD)  |  cwd: $(pwd)"

[ -f "uncertainty/uncertainty_mc.py" ] || { echo "ERROR: uncertainty/uncertainty_mc.py not found in $(pwd)"; exit 1; }

# ---- 2. package dirs need __init__.py (may be .gitignored) ----
for d in datasets datasets/two_dim networks loss_functions evaluation utilities; do
    mkdir -p "$d"; touch "$d/__init__.py"
done

# ---- 3. dependencies (torch/numpy NOT reinstalled) ----
echo ""; echo "===== [$(date '+%F %T')] pip install deps ====="
PKGS="medpy nibabel SimpleITK batchgenerators==0.21 scipy scikit-image matplotlib pandas"
python3 -m pip install --break-system-packages $PKGS 2>/dev/null || python3 -m pip install $PKGS || \
    echo "WARN: pip install returned non-zero; continuing (deps may already be present)"
python3 - <<'PY' || { echo "FATAL: core deps import failed. Try: pip install 'numpy<2'  then re-run."; exit 1; }
import torch, importlib
for m in ("batchgenerators", "medpy", "nibabel", "scipy", "skimage", "matplotlib"):
    importlib.import_module(m)
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"))
print("core deps import OK")
PY

# ---- 4. data: reuse if already preprocessed, else download + preprocess ----
export TASK
export DATA_DIR="$PWD/data/$TASK"
PREP_DIR="$DATA_DIR/preprocessed"
SPLITS="$DATA_DIR/splits.pkl"

have_prep() {   # splits + a >=2-channel npy (image+label)
    [ -f "$SPLITS" ] || return 1
    local f
    f=$(find "$PREP_DIR" -maxdepth 1 -name '*.npy' -print -quit 2>/dev/null)
    [ -n "$f" ] || return 1
    python3 - "$f" <<'PY'
import sys, numpy as np
sys.exit(0 if np.load(sys.argv[1], mmap_mode="r").shape[0] >= 2 else 1)
PY
}

if have_prep; then
    echo "preprocessed data + splits present -> skipping download/preprocess"
else
    if [ ! -d "$DATA_DIR/imagesTr" ]; then
        echo ""; echo "===== [$(date '+%F %T')] download raw $TASK (~30 MB) ====="
        mkdir -p data
        ( cd data && curl -O "$DATA_TAR_URL" && tar -xf "${TASK}.tar" ) \
            || { echo "DOWNLOAD FAILED -> aborting"; exit 1; }
    fi
    run "preprocess (2-channel npy + splits)" python3 uncertainty/run_preprocessing_mc.py
    have_prep || { echo "PREPROCESS did not produce a 2-channel npy + splits -> aborting"; exit 1; }
fi

mkdir -p results

# ---- 5. bring in the pretrained weights so training is SKIPPED ----
if [ -n "$WEIGHTS_DIR" ]; then
    echo ""; echo "===== copy weights from $WEIGHTS_DIR -> results/ ====="
    cp "$WEIGHTS_DIR"/*_best.pth results/ 2>/dev/null && ls -1 results/*_best.pth | sed 's/^/  /' \
        || echo "WARN: no *_best.pth found in $WEIGHTS_DIR (will train from scratch if missing)"
fi

EVAL="--patch-size $PATCH --batch-size $BATCH --num-workers $WORKERS --dropout-p $DROPOUT --out-dir results"

# ---- 6. experiments: per fold, both models. skip-if-exists on the checkpoints ----
for FOLD in $FOLDS; do
    echo ""; echo "################  FOLD $FOLD  ################"

    # ===== A. baseline MC-Dropout (Dice + CE) =====
    if [ -f "results/mcdropout_f${FOLD}_best.pth" ]; then
        echo "===== using existing mcdropout f$FOLD weights (skip training) ====="
    else
        run "train mcdropout f$FOLD (no weights found)" python3 uncertainty/train_mc.py --fold "$FOLD" --tag mcdropout \
            --patch-size "$PATCH" --batch-size "$BATCH" --num-workers "$WORKERS" \
            --epochs "$EPOCHS" --patience "$PATIENCE" --dropout-p "$DROPOUT" --out-dir results
    fi
    run "test mcdropout f$FOLD"             python3 uncertainty/test_mc.py        --fold "$FOLD" --tag mcdropout $EVAL
    run "uncertainty raw mcdropout f$FOLD"  python3 uncertainty/uncertainty_mc.py --fold "$FOLD" --tag mcdropout $EVAL --mc-samples "$MC" --temperature 1.0 --save-volumes
    run "calibrate mcdropout f$FOLD"        python3 uncertainty/calibrate_mc.py   --fold "$FOLD" --tag mcdropout $EVAL
    run "uncertainty +T mcdropout f$FOLD"   python3 uncertainty/uncertainty_mc.py --fold "$FOLD" --tag mcdropout $EVAL --mc-samples "$MC" --save-volumes

    # ===== B. train-time calibrated MC-Dropout (Dice + CE + lambda*SB-ECE) =====
    if [ -f "results/mcdropout_cal_f${FOLD}_best.pth" ]; then
        echo "===== using existing mcdropout_cal f$FOLD weights (skip training) ====="
    else
        run "train mcdropout_cal f$FOLD (no weights found)" python3 uncertainty/train_mc_recalibrate.py --fold "$FOLD" --tag mcdropout_cal \
            --patch-size "$PATCH" --batch-size "$BATCH" --num-workers "$WORKERS" \
            --epochs "$EPOCHS" --patience "$PATIENCE" --dropout-p "$DROPOUT" --cal-weight "$CALW" --out-dir results
    fi
    run "test mcdropout_cal f$FOLD"             python3 uncertainty/test_mc.py        --fold "$FOLD" --tag mcdropout_cal $EVAL
    run "uncertainty raw mcdropout_cal f$FOLD"  python3 uncertainty/uncertainty_mc.py --fold "$FOLD" --tag mcdropout_cal $EVAL --mc-samples "$MC" --temperature 1.0 --save-volumes
    run "calibrate mcdropout_cal f$FOLD"        python3 uncertainty/calibrate_mc.py   --fold "$FOLD" --tag mcdropout_cal $EVAL
    run "uncertainty +T mcdropout_cal f$FOLD"   python3 uncertainty/uncertainty_mc.py --fold "$FOLD" --tag mcdropout_cal $EVAL --mc-samples "$MC" --save-volumes
done

# ---- 7. summary: Dice + foreground/macro ECE + temperature across the four settings ----
echo ""; echo "===== [$(date '+%F %T')] summary ====="
OUT_DIR="results" FOLDS="$FOLDS" python3 - <<'PY' || echo "WARN: summary failed"
import os, json
out_dir = os.environ.get("OUT_DIR", "results")
folds = os.environ.get("FOLDS", "0").split()
unc = os.path.join(out_dir, "uncertainty")

def dice_of(tag, f):
    p = os.path.join(out_dir, f"{tag}_f{f}_scores.json")
    if not os.path.exists(p):
        return None
    mean = json.load(open(p)).get("results", {}).get("mean", {})
    vals = [md["Dice"] for md in mean.values() if isinstance(md, dict) and md.get("Dice") is not None]
    return sum(vals) / len(vals) if vals else None

def ece_of(fn):
    p = os.path.join(unc, fn)
    if not os.path.exists(p):
        return None
    d = json.load(open(p)); c = d["calibration"]
    return c["foreground_ece"], c["macro_foreground_ece"], d.get("temperature", 1.0)

print(f"\n{'model / fold':<22}{'meanDice':>9}")
for f in folds:
    for tag in ("mcdropout", "mcdropout_cal"):
        d = dice_of(tag, f)
        print(f"{tag+' f'+f:<22}{('%.4f'%d) if d is not None else 'n/a':>9}")

print(f"\n{'setting':<26}{'fg_ECE':>9}{'macro_ECE':>11}{'T':>7}")
for f in folds:
    for label, fn in [
        (f"f{f} baseline raw",   f"mcdropout_f{f}_uncertainty.json"),
        (f"f{f} baseline +T",    f"mcdropout_f{f}_cal_uncertainty.json"),
        (f"f{f} calibrated raw", f"mcdropout_cal_f{f}_uncertainty.json"),
        (f"f{f} calibrated +T",  f"mcdropout_cal_f{f}_cal_uncertainty.json"),
    ]:
        r = ece_of(fn)
        if r:
            macro = f"{r[1]:.4f}" if r[1] is not None else "n/a"
            print(f"{label:<26}{r[0]:>9.4f}{macro:>11}{r[2]:>7.3f}")
PY

cp "$LOG" results/ 2>/dev/null || true
echo ""; echo "################ ALL DONE  $(date '+%F %T') ################"
echo "results in: $REPO_DIR/results/"
ls -1 results/ 2>/dev/null | sed 's/^/  /'
echo ""
echo "Copy off the VM, e.g.:"
echo "  gcloud compute scp --recurse <user>@<vm>:$REPO_DIR/results ./"
