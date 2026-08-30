#!/usr/bin/env bash
#
# Self-contained CPU-only experiment runner for ONE tier of the traditional segmentation
# methods on MSD Task09 Spleen. The sibling scripts run_tier1.sh / run_tier2.sh /
# run_tier3.sh are IDENTICAL except the TIER default line below.
#
# Each does EVERYTHING itself and the SAME way: clone (or update) the repo -> install deps
# -> download Spleen (once) -> dual-track preprocess (once) -> run this tier's methods
# (method x regime x track x fold) -> results in results/tier<N>/<method>/ -> pooled-Dice
# summary. No GPU, no U-Net. Tier-3 methods are not implemented yet -> run_tier3.sh will
# report "nothing to run" until they are added.
#
# Bootstrap on the VM (survives an SSH disconnect); grabs just this one script:
#   curl -O https://raw.githubusercontent.com/thnrigas/morph-unet/main/tradseg/run_tier1.sh
#   nohup bash run_tier1.sh &        # progress: tail -f ~/tradseg-run/run_tier*_*.log
# Copy results off:  gcloud compute scp --recurse <user>@<vm>:~/tradseg-run/repo/tradseg/results ./
#
# Run the three tiers one at a time (they share the same repo checkout + data).
#
set -uo pipefail

TIER="${TIER:-1}"     # <== ONLY difference between run_tier1.sh / run_tier2.sh / run_tier3.sh

# ============================ CONFIG (override via env) =====================
REPO_URL="${REPO_URL:-https://github.com/thnrigas/morph-unet.git}"
BRANCH="${BRANCH:-main}"                                # set BRANCH=master if that is your default
SUBDIR="${SUBDIR:-tradseg}"                       # this experiment's folder inside the monorepo
WORKDIR="${WORKDIR:-$HOME/tradseg-run}"
TASK="${TASK:-Task09_Spleen}"
REGIMES="${REGIMES:-auto oracle}"
TRACKS="${TRACKS:-A B}"
FOLDS="${FOLDS:-0 1 2 3 4}"
ADVANCED="${ADVANCED:-1}"                               # 1 -> also HD95/ASSD (needs medpy)
# METHODS defaults to this tier's methods (resolved after clone via list_methods.py)
# ============================================================================

mkdir -p "$WORKDIR"
LOG="$WORKDIR/run_tier${TIER}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "############### TradSeg Tier-$TIER (CPU, no U-Net)  $(date '+%F %T') ###############"
echo "repo=$REPO_URL branch=$BRANCH workdir=$WORKDIR task=$TASK"
echo "regimes='$REGIMES' tracks='$TRACKS' folds='$FOLDS' advanced=$ADVANCED"

run() {   # run "label" cmd... : header + timing, CONTINUE on failure
    local label="$1"; shift
    echo ""; echo "===== [$(date '+%F %T')] $label ====="
    local t0=$SECONDS
    "$@"; local rc=$?
    echo "----- $label done in $((SECONDS - t0))s (exit $rc) -----"
    [ $rc -eq 0 ] || echo "!!! FAILED: $label (continuing) !!!"
    return 0
}

# ---- 1. sparse-clone (or update) ONLY this experiment's subfolder ----
#     The monorepo holds all three experiments; we check out just $SUBDIR via a
#     partial + cone sparse checkout so the other parts are never downloaded.
#     (needs git >= 2.25; falls back to a normal clone if sparse is unavailable.)
REPO_DIR="$WORKDIR/repo"
if [ -d "$REPO_DIR/.git" ]; then
    echo "repo present -> updating $SUBDIR to origin/$BRANCH (untracked data/ & results/ kept)"
    git -C "$REPO_DIR" sparse-checkout set "$SUBDIR" 2>/dev/null || true
    git -C "$REPO_DIR" fetch origin "$BRANCH" \
        && git -C "$REPO_DIR" checkout "$BRANCH" \
        && git -C "$REPO_DIR" reset --hard FETCH_HEAD \
        || echo "WARN: could not update; using existing checkout"
else
    git clone --filter=blob:none --no-checkout --branch "$BRANCH" --single-branch "$REPO_URL" "$REPO_DIR" \
        || { echo "git clone of branch '$BRANCH' failed -> aborting"; exit 1; }
    git -C "$REPO_DIR" sparse-checkout init --cone \
        && git -C "$REPO_DIR" sparse-checkout set "$SUBDIR" \
        && git -C "$REPO_DIR" checkout "$BRANCH" \
        || { echo "sparse checkout of '$SUBDIR' failed -> aborting"; exit 1; }
fi
cd "$REPO_DIR/$SUBDIR" || { echo "cannot cd $REPO_DIR/$SUBDIR"; exit 1; }
echo "on branch $(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD) @ $(git -C "$REPO_DIR" rev-parse --short HEAD)  |  cwd: $(pwd)"
[ -f "run_experiment.py" ] || { echo "ERROR: run_experiment.py not found in $SUBDIR/ (flat layout expected)."; exit 1; }
mkdir -p methods; touch methods/__init__.py

# ---- 2. dependencies (CPU-only; no torch needed) ----
echo ""; echo "===== [$(date '+%F %T')] pip install deps ====="
python3 -m pip install --break-system-packages -q -r requirements.txt 2>/dev/null \
    || python3 -m pip install -q -r requirements.txt \
    || echo "WARN: pip install returned non-zero; continuing (deps may already be present)"
python3 - <<'PYCHK' || { echo "FATAL: core deps failed to import."; exit 1; }
import numpy, scipy, skimage, sklearn, medpy, nibabel, SimpleITK
print("deps OK | numpy", numpy.__version__, "| skimage", skimage.__version__)
try:
    import maxflow; print("PyMaxflow OK")
except Exception as e:
    print("WARN: PyMaxflow not importable -> graphcut will be skipped:", e)
PYCHK

# ---- 3. data: download + extract once ----
export TASK
export DATA_DIR="$PWD/data/$TASK"
if [ ! -d "$DATA_DIR/imagesTr" ]; then
    echo ""; echo "===== [$(date '+%F %T')] download raw $TASK (~1.5 GB) ====="
    mkdir -p data
    ( cd data && curl -L -O "https://msd-for-monai.s3-us-west-2.amazonaws.com/${TASK}.tar" \
        && tar -xf "${TASK}.tar" ) \
        || { echo "DOWNLOAD FAILED -> aborting"; exit 1; }
fi

# ---- 4. dual-track preprocessing, once per track ----
for track in $TRACKS; do
    prep="$DATA_DIR/preprocessed_$track"
    if [ -f "$DATA_DIR/splits.pkl" ] && ls "$prep"/*.npy >/dev/null 2>&1; then
        echo "skip preprocess track $track (already present)"
    else
        run "preprocess track $track" python3 preprocessing.py --track "$track"
    fi
done

# ---- 5. resolve this tier's methods ----
METHODS="${METHODS:-$(python3 list_methods.py "$TIER")}"
if [ -z "${METHODS// /}" ]; then
    echo "no registered methods for tier $TIER (not implemented yet?) -> nothing to run"
    echo "################ Tier-$TIER DONE (empty)  $(date '+%F %T') ################"
    exit 0
fi
echo "tier $TIER methods: $METHODS"

# ---- 6. experiments -> results/tier<N>/<method>/ (skip-if-exists) ----
RESULTS="results/tier$TIER"
ADV=""; [ "$ADVANCED" = "1" ] && ADV="--advanced"
for method in $METHODS; do
    mkdir -p "$RESULTS/$method"
    for track in $TRACKS; do
        prep="$DATA_DIR/preprocessed_$track"
        ls "$prep"/*.npy >/dev/null 2>&1 || { echo "skip $method track $track (no data)"; continue; }
        for regime in $REGIMES; do
            for fold in $FOLDS; do
                stem="${method}_${regime}_track${track}_f${fold}"
                if [ -f "$RESULTS/$method/$stem.json" ]; then echo "skip $stem (exists)"; continue; fi
                run "$stem" python3 run_experiment.py --method "$method" --fold "$fold" \
                    --regime "$regime" --track "$track" --preprocessed-dir "$prep" \
                    --out-dir "$RESULTS/$method" $ADV
            done
        done
    done
done

# ---- 7. summary for this tier ----
echo ""; echo "===== [$(date '+%F %T')] tier $TIER summary ====="
RESULTS="$RESULTS" python3 - <<'PYSUM' || echo "WARN: summary failed"
import os, json, glob
from statistics import mean
root = os.environ["RESULTS"]
rows = {}
for jf in glob.glob(os.path.join(root, "*", "*.json")):
    try:
        d = json.load(open(jf))
    except Exception:
        continue
    m = d.get("meta", {})
    key = (m.get("method"), m.get("regime"), m.get("track"))
    vals = [c["Dice"] for c in d.get("per_case", {}).values()
            if c.get("Dice") is not None and c["Dice"] == c["Dice"]]
    rows.setdefault(key, []).extend(vals)
print(f"\n{'method':16s}{'regime':8s}{'track':6s}{'meanDice':>9}{'n':>5}")
for key in sorted(k for k in rows if all(x is not None for x in k)):
    v = rows[key]
    print(f"{key[0]:16s}{key[1]:8s}{key[2]:6s}{(mean(v) if v else float('nan')):>9.4f}{len(v):>5d}")
PYSUM

cp "$LOG" "$RESULTS/" 2>/dev/null || true
echo ""; echo "################ Tier-$TIER DONE  $(date '+%F %T') ################"
echo "results (one folder per method): $REPO_DIR/$SUBDIR/$RESULTS/<method>/"
ls -1 "$RESULTS" 2>/dev/null | sed 's/^/  /'
