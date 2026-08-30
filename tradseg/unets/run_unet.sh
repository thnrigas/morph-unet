#!/usr/bin/env bash
#
# Self-contained U-Net runner (baseline + MC-Dropout uncertainty) for the HYBRID experiments.
# clone/update repo -> deps -> download Spleen (once) -> preprocess Track A (once) ->
# per fold: train (SKIP if <tag>_f<fold>_best.pth exists) -> test -> MC infer (seg+uncertainty
# for the traditional refiner). Weights + predictions land in results/unets/.
#
# GPU recommended (L4). torch is assumed PRE-INSTALLED (DL VM) and NOT reinstalled.
# Does NOT run the traditional tiers, and does NOT touch my_unet-uncertainty/.
#
# Bootstrap on the VM:
#   curl -O https://raw.githubusercontent.com/thnrigas/morph-unet/main/tradseg/unets/run_unet.sh
#   nohup bash run_unet.sh &        # progress: tail -f ~/tradseg-run/run_unet_*.log
#
# Env: TAG (mcdropout) FRAC (1.0) EPOCHS (150) MC (30) FOLDS ("0 1 2 3 4") BRANCH (main).
#   Weakened baseline (Idea 2):  TAG=mcdropout_f25 FRAC=0.25 nohup bash run_unet.sh &
#
set -uo pipefail

TAG="${TAG:-mcdropout}"; FRAC="${FRAC:-1.0}"; EPOCHS="${EPOCHS:-150}"
MC="${MC:-30}"; FOLDS="${FOLDS:-0 1 2 3 4}"

REPO_URL="${REPO_URL:-https://github.com/thnrigas/morph-unet.git}"
BRANCH="${BRANCH:-main}"
SUBDIR="${SUBDIR:-tradseg}"      # this experiment's folder inside the monorepo
WORKDIR="${WORKDIR:-$HOME/tradseg-run}"
TASK="${TASK:-Task09_Spleen}"

mkdir -p "$WORKDIR"
LOG="$WORKDIR/run_unet_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "############### U-Net (hybrid) $(date '+%F %T') ###############"
echo "tag=$TAG frac=$FRAC epochs=$EPOCHS mc=$MC folds='$FOLDS' branch=$BRANCH"

run() { local l="$1"; shift; echo ""; echo "===== [$(date '+%F %T')] $l ====="; local t=$SECONDS
        "$@"; local rc=$?; echo "----- $l done in $((SECONDS-t))s (exit $rc) -----"
        [ $rc -eq 0 ] || echo "!!! FAILED: $l (continuing) !!!"; return 0; }

# ---- 1. sparse-clone / update ONLY this experiment's subfolder ----
#     (needs git >= 2.25; other parts of the monorepo are never downloaded)
REPO_DIR="$WORKDIR/repo"
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" sparse-checkout set "$SUBDIR" 2>/dev/null || true
    git -C "$REPO_DIR" fetch origin "$BRANCH" && git -C "$REPO_DIR" checkout "$BRANCH" \
        && git -C "$REPO_DIR" reset --hard FETCH_HEAD || echo "WARN: update failed; using existing checkout"
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

# ---- 2. deps (torch assumed preinstalled; add base reqs + batchgenerators) ----
echo ""; echo "===== [$(date '+%F %T')] pip install deps ====="
python3 -m pip install --break-system-packages -q -r requirements.txt -r unets/requirements-unet.txt 2>/dev/null \
    || python3 -m pip install -q -r requirements.txt -r unets/requirements-unet.txt \
    || echo "WARN: pip install returned non-zero; continuing"
python3 - <<'PYCHK' || { echo "FATAL: core deps failed to import."; exit 1; }
import torch, numpy, scipy, skimage, medpy, nibabel, SimpleITK, batchgenerators
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"),
      "| numpy", numpy.__version__)
PYCHK

# ---- 3. data (once) ----
export TASK
export DATA_DIR="$PWD/data/$TASK"
if [ ! -d "$DATA_DIR/imagesTr" ]; then
    echo ""; echo "===== [$(date '+%F %T')] download raw $TASK (~1.5 GB) ====="
    mkdir -p data
    ( cd data && curl -L -O "https://msd-for-monai.s3-us-west-2.amazonaws.com/${TASK}.tar" \
        && tar -xf "${TASK}.tar" ) || { echo "DOWNLOAD FAILED -> aborting"; exit 1; }
fi

# ---- 4. Track-A preprocessing (once) -- reused by the U-Net (same contract) ----
PREP="$DATA_DIR/preprocessed_A"
if [ -f "$DATA_DIR/splits.pkl" ] && ls "$PREP"/*.npy >/dev/null 2>&1; then
    echo "skip preprocess Track A (already present)"
else
    run "preprocess Track A" python3 preprocessing.py --track A
fi

# ---- 5. per fold: train (skip-if-exists inside train.py) -> test -> MC infer ----
OUT="results/unets"; mkdir -p "$OUT"
for fold in $FOLDS; do
    echo ""; echo "################  $TAG  FOLD $fold  ################"
    run "train $TAG f$fold" python3 unets/train.py --tag "$TAG" --fold "$fold" --epochs "$EPOCHS" \
        --frac "$FRAC" --preprocessed-dir "$PREP" --out-dir "$OUT"
    run "test $TAG f$fold"  python3 unets/test.py  --tag "$TAG" --fold "$fold" \
        --preprocessed-dir "$PREP" --weights-dir "$OUT" --out-dir "$OUT" --advanced
    run "infer $TAG f$fold" python3 unets/infer.py --tag "$TAG" --fold "$fold" --mc-samples "$MC" \
        --preprocessed-dir "$PREP" --weights-dir "$OUT" --out-dir "$OUT"
done

cp "$LOG" "$OUT/" 2>/dev/null || true
echo ""; echo "############### U-Net DONE $(date '+%F %T') ###############"
echo "weights/preds/scores -> $REPO_DIR/$SUBDIR/$OUT/"
ls -1 "$OUT" 2>/dev/null | sed 's/^/  /'
