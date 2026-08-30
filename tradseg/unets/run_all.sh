#!/usr/bin/env bash
#
# FULL U-Net + HYBRID experiment orchestrator (one script, all experiments).
#
#   1. clone the repo the first time, otherwise refresh it (git fetch + hard reset)
#   2. download the Spleen dataset (skip if already present)
#   3. preprocess Track A + build splits (skip if already done)
#   4. per fold, train three nets (SKIP-if-weights-exist inside train.py) and test them:
#        baseline   : pure U-Net              (dropout_p=0.0, full data)
#        mcdropout  : MC-Dropout U-Net        (dropout_p=0.4, full data)  [+ MC infer dumps]
#        weak       : weakened pure U-Net     (dropout_p=0.0, WEAK_FRAC of the train cases)
#   5. per fold, run the hybrids (both refiners each: random walker + level set):
#        Hybrid #1  : uncertainty-gated refinement of  mcdropout
#        Hybrid #2  : morphological anchored refinement of  weak
#   6. aggregate EVERYTHING into results/comparison/summary.{csv,json}
#
# Weights -> results/unets/ ; hybrid JSONs -> results/hybrid/ ; table -> results/comparison/.
# GPU recommended (L4); torch assumed PRE-INSTALLED. Does NOT touch my_unet-uncertainty/.
#
# Bootstrap on the VM:
#   curl -O https://raw.githubusercontent.com/thnrigas/morph-unet/main/tradseg/unets/run_all.sh
#   nohup bash run_all.sh &          # progress: tail -f ~/tradseg-run/run_all_*.log
#
# Env knobs (defaults: ONE fold, single training of each net):
#   FOLDS="0"  EPOCHS=150  WEAK_FRAC=0.2  WEAK_EPOCHS=$EPOCHS  MC=30
#   UNC=entropy  BRANCH=main  FORCE=  (set FORCE=1 to recompute test/hybrid JSONs)
#   Hybrids: seed-and-solve RW + edge-snapping GAC (balloon 0) on a Track-B image, with a
#   val-tuned uncertainty-selective safety net (reports always/selective/oracle).
#   For the full 5-fold CV later:  FOLDS="0 1 2 3 4" nohup bash run_all.sh &
#
set -uo pipefail

FOLDS="${FOLDS:-0}"
EPOCHS="${EPOCHS:-150}"
WEAK_FRAC="${WEAK_FRAC:-0.2}"          # genuinely weak base (frac 0.5 -> ~0.92, above the trad ceiling)
WEAK_EPOCHS="${WEAK_EPOCHS:-$EPOCHS}"
MC="${MC:-30}"
UNC="${UNC:-entropy}"          # predictive entropy: strong, boundary-localised (aleatoric) gate
FORCE="${FORCE:-}"

BASE_TAG="${BASE_TAG:-baseline}"
MC_TAG="${MC_TAG:-mcdropout}"
WEAK_TAG="${WEAK_TAG:-weak02}"          # tag encodes the frac so a new frac retrains (not skip)

REPO_URL="${REPO_URL:-https://github.com/thnrigas/morph-unet.git}"
BRANCH="${BRANCH:-main}"
SUBDIR="${SUBDIR:-tradseg}"      # this experiment's folder inside the monorepo
WORKDIR="${WORKDIR:-$HOME/tradseg-run}"
TASK="${TASK:-Task09_Spleen}"

mkdir -p "$WORKDIR"
LOG="$WORKDIR/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "############### U-Net + HYBRID  $(date '+%F %T') ###############"
echo "folds='$FOLDS' epochs=$EPOCHS weak_frac=$WEAK_FRAC weak_epochs=$WEAK_EPOCHS mc=$MC unc=$UNC branch=$BRANCH force='${FORCE}'"

run() { local l="$1"; shift; echo ""; echo "===== [$(date '+%F %T')] $l ====="; local t=$SECONDS
        "$@"; local rc=$?; echo "----- $l done in $((SECONDS-t))s (exit $rc) -----"
        [ $rc -eq 0 ] || echo "!!! FAILED: $l (continuing) !!!"; return 0; }
# have <file>  -> true if the file exists AND we are not forcing a recompute
have() { [ -f "$1" ] && [ -z "$FORCE" ]; }

# ---- 1. sparse-clone / refresh ONLY this experiment's subfolder -------------
#     The monorepo holds all three experiments; we check out just $SUBDIR via a
#     partial + cone sparse checkout so the other parts are never downloaded.
#     (needs git >= 2.25; falls back to a normal clone if sparse is unavailable.)
REPO_DIR="$WORKDIR/repo"
if [ -d "$REPO_DIR/.git" ]; then
    echo "refresh existing checkout ($SUBDIR)"
    git -C "$REPO_DIR" sparse-checkout set "$SUBDIR" 2>/dev/null || true
    git -C "$REPO_DIR" fetch origin "$BRANCH" && git -C "$REPO_DIR" checkout "$BRANCH" \
        && git -C "$REPO_DIR" reset --hard FETCH_HEAD || echo "WARN: refresh failed; using existing checkout"
else
    git clone --filter=blob:none --no-checkout --branch "$BRANCH" --single-branch "$REPO_URL" "$REPO_DIR" \
        || { echo "git clone failed -> aborting"; exit 1; }
    git -C "$REPO_DIR" sparse-checkout init --cone \
        && git -C "$REPO_DIR" sparse-checkout set "$SUBDIR" \
        && git -C "$REPO_DIR" checkout "$BRANCH" \
        || { echo "sparse checkout of '$SUBDIR' failed -> aborting"; exit 1; }
fi
cd "$REPO_DIR/$SUBDIR" || { echo "cannot cd $REPO_DIR/$SUBDIR"; exit 1; }
echo "branch $(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD) @ $(git -C "$REPO_DIR" rev-parse --short HEAD) | cwd $(pwd)"
[ -f "unets/train.py" ] || { echo "ERROR: unets/train.py not found in $SUBDIR/ (flat layout expected)."; exit 1; }
mkdir -p methods; touch methods/__init__.py

# ---- 2. deps ----------------------------------------------------------------
echo ""; echo "===== [$(date '+%F %T')] pip install deps ====="
python3 -m pip install --break-system-packages -q -r requirements.txt -r unets/requirements-unet.txt 2>/dev/null \
    || python3 -m pip install -q -r requirements.txt -r unets/requirements-unet.txt \
    || echo "WARN: pip install returned non-zero; continuing"
python3 - <<'PYCHK' || { echo "FATAL: core deps failed to import."; exit 1; }
import torch, numpy, scipy, skimage, medpy, batchgenerators
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"),
      "| numpy", numpy.__version__)
PYCHK

# ---- 3. data (once) ---------------------------------------------------------
export TASK
export DATA_DIR="$PWD/data/$TASK"
if [ ! -d "$DATA_DIR/imagesTr" ]; then
    echo ""; echo "===== [$(date '+%F %T')] download raw $TASK (~1.5 GB) ====="
    mkdir -p data
    ( cd data && curl -L -O "https://msd-for-monai.s3-us-west-2.amazonaws.com/${TASK}.tar" \
        && tar -xf "${TASK}.tar" ) || { echo "DOWNLOAD FAILED -> aborting"; exit 1; }
else
    echo "skip download ($DATA_DIR/imagesTr present)"
fi

# ---- 4. Track-A preprocessing + splits (once) -------------------------------
PREP="$DATA_DIR/preprocessed_A"
if [ -f "$DATA_DIR/splits.pkl" ] && ls "$PREP"/*.npy >/dev/null 2>&1; then
    echo "skip preprocess Track A (splits.pkl + npy already present)"
else
    run "preprocess Track A" python3 preprocessing.py --track A
fi

UOUT="results/unets"; HOUT="results/hybrid"; COUT="results/comparison"
mkdir -p "$UOUT" "$HOUT" "$COUT"

# ---- 5. per fold: train + test 3 nets, then 2 hybrids -----------------------
for fold in $FOLDS; do
    echo ""; echo "################  FOLD $fold  ################"

    # -- pure baseline (p=0, full data) --
    run "train $BASE_TAG f$fold" python3 unets/train.py --tag "$BASE_TAG" --fold "$fold" \
        --epochs "$EPOCHS" --frac 1.0 --dropout-p 0.0 --preprocessed-dir "$PREP" --out-dir "$UOUT"
    if have "$UOUT/unet_${BASE_TAG}_f${fold}.json"; then echo "skip test $BASE_TAG f$fold (json exists)"; else
        run "test $BASE_TAG f$fold" python3 unets/test.py --tag "$BASE_TAG" --fold "$fold" \
            --dropout-p 0.0 --preprocessed-dir "$PREP" --weights-dir "$UOUT" --out-dir "$UOUT" --advanced; fi

    # -- MC-Dropout net (p=0.4, full data) + MC infer dumps --
    run "train $MC_TAG f$fold" python3 unets/train.py --tag "$MC_TAG" --fold "$fold" \
        --epochs "$EPOCHS" --frac 1.0 --dropout-p 0.4 --preprocessed-dir "$PREP" --out-dir "$UOUT"
    if have "$UOUT/unet_${MC_TAG}_f${fold}.json"; then echo "skip test $MC_TAG f$fold (json exists)"; else
        run "test $MC_TAG f$fold" python3 unets/test.py --tag "$MC_TAG" --fold "$fold" \
            --dropout-p 0.4 --preprocessed-dir "$PREP" --weights-dir "$UOUT" --out-dir "$UOUT" --advanced; fi
    if have "$UOUT/unet_${MC_TAG}_f${fold}_mcinfer.json"; then echo "skip infer $MC_TAG f$fold (json exists)"; else
        run "infer $MC_TAG f$fold" python3 unets/infer.py --tag "$MC_TAG" --fold "$fold" \
            --mc-samples "$MC" --dropout-p 0.4 --preprocessed-dir "$PREP" --weights-dir "$UOUT" --out-dir "$UOUT"; fi

    # -- weakened pure net (p=0, WEAK_FRAC of train cases) --
    run "train $WEAK_TAG f$fold" python3 unets/train.py --tag "$WEAK_TAG" --fold "$fold" \
        --epochs "$WEAK_EPOCHS" --frac "$WEAK_FRAC" --dropout-p 0.0 --preprocessed-dir "$PREP" --out-dir "$UOUT"
    if have "$UOUT/unet_${WEAK_TAG}_f${fold}.json"; then echo "skip test $WEAK_TAG f$fold (json exists)"; else
        run "test $WEAK_TAG f$fold" python3 unets/test.py --tag "$WEAK_TAG" --fold "$fold" \
            --dropout-p 0.0 --preprocessed-dir "$PREP" --weights-dir "$UOUT" --out-dir "$UOUT" --advanced; fi

    # -- Hybrid #1: uncertainty-gated refinement of the MC-Dropout net (rw + ls) --
    if have "$HOUT/hybrid_uncertainty_rw_${MC_TAG}_f${fold}.json" \
       && have "$HOUT/hybrid_uncertainty_ls_${MC_TAG}_f${fold}.json"; then
        echo "skip hybrid#1 f$fold (jsons exist)"
    else
        run "hybrid#1 uncertainty $MC_TAG f$fold" python3 run_hybrid.py --tag "$MC_TAG" --fold "$fold" \
            --mode uncertainty --refiner both --uncertainty "$UNC" --mc-samples "$MC" --dropout-p 0.4 \
            --preprocessed-dir "$PREP" --weights-dir "$UOUT" --out-dir "$HOUT" --advanced
    fi

    # -- Hybrid #2: morphological anchored refinement of the weakened net (rw + ls) --
    if have "$HOUT/hybrid_morph_rw_${WEAK_TAG}_f${fold}.json" \
       && have "$HOUT/hybrid_morph_ls_${WEAK_TAG}_f${fold}.json"; then
        echo "skip hybrid#2 f$fold (jsons exist)"
    else
        run "hybrid#2 morph $WEAK_TAG f$fold" python3 run_hybrid.py --tag "$WEAK_TAG" --fold "$fold" \
            --mode morph --refiner both --dropout-p 0.0 \
            --preprocessed-dir "$PREP" --weights-dir "$UOUT" --out-dir "$HOUT" --advanced
    fi
done

# ---- 6. aggregate everything into one comparison table ----------------------
run "aggregate comparison" python3 agg_compare.py --unet-dir "$UOUT" --hybrid-dir "$HOUT" --out "$COUT"

cp "$LOG" "$COUT/" 2>/dev/null || true
echo ""; echo "############### DONE  $(date '+%F %T') ###############"
echo "weights -> $REPO_DIR/$SUBDIR/$UOUT/ | hybrids -> $REPO_DIR/$SUBDIR/$HOUT/ | table -> $REPO_DIR/$SUBDIR/$COUT/summary.csv"
[ -f "$COUT/summary.csv" ] && { echo ""; cat "$COUT/summary.csv"; }
