# Part 3 — Traditional segmentation, U-Net comparison & hybrids (MSD Spleen)

Classical (non-neural) segmentation methods for MSD **Task09 Spleen**, scored **per-volume** in
the **same 256×256 space and 5-fold CV as the baseline U-Net**, so everything is directly
comparable. Three sub-experiments, along the axes **accuracy / time / memory**:

- **A. Standalone traditional methods** (Tiers 1–3) — CPU-only, no GPU.
- **B. U-Net baseline + MC-Dropout** — the neural reference to compare against (GPU).
- **C. Hybrids** — the U-Net refined by a traditional method (random walker / level set) inside
  an uncertainty- or morphology-anchored band.

The repo is **flat**: modules live at the folder root and are launched directly as
`python3 <script>.py` (absolute imports — do **not** use `python -m`). The `unets/` subfolder
holds the neural code + the U-Net↔refiner communication layer.

**Regimes** (for the traditional methods): `auto` = fully automatic — the spleen is localised
by a probabilistic **spatial + intensity prior built from the training fold** (fair vs the
automatic U-Net); `oracle` = seeds/component derived from the case GT → per-method upper bound.

**Tracks** (preprocessing): `A` = identical to the U-Net (window c40/w400, resize 256 — the fair
track); `B` = traditional-optimised (window c50/w150 + median denoise).

---

## Install
```bash
pip install -r requirements.txt                 # + unets/requirements-unet.txt for sub-exp B/C
```

## A. Standalone traditional methods — one command per tier
Each `run_tier<N>.sh` is self-contained and identical except the tier: sparse-clone this
subfolder → deps → download Spleen (once) → dual-track preprocess (once) → run that tier's
methods (method × regime × track × fold) → per-method JSON + pooled-Dice summary. Run them one
at a time (they share the repo checkout + data):
```bash
nohup bash run_tier1.sh &     # Tier 1: otsu, multiotsu, kmeans, gmm, region_growing, watershed
nohup bash run_tier2.sh &     # Tier 2: chanvese, morphgac, graphcut, random_walker
nohup bash run_tier3.sh &     # Tier 3: gabor, amfm, granulometry
# overrides: FOLDS="0"  TRACKS="A"  REGIMES="auto"  METHODS="graphcut random_walker"  ADVANCED=0
```
→ results in **`results/tier<N>/<method>/<method>_<regime>_track<T>_f<F>.json`**
(each has `meta`, mean metrics, and `per_case` Dice/Jaccard/HD95/ASSD).

**Manual (one method × fold):**
```bash
export DATA_DIR=$PWD/data/Task09_Spleen
python3 preprocessing.py --track A                       # once per track -> preprocessed_A/
python3 run_experiment.py --method multiotsu --fold 0 --regime auto \
    --preprocessed-dir data/Task09_Spleen/preprocessed_A --advanced --out-dir results/tier1/multiotsu
```

## B + C. U-Net baseline + hybrids — one command
`unets/run_all.sh` is the one-shot orchestrator: clone/refresh → data → preprocess Track A →
per fold train+test **3 nets** (`baseline` p=0, `mcdropout` p=0.4, `weak02` p=0 on a fraction of
train) + MC-infer → run **both hybrids** (each with random walker **and** level set) → aggregate
one comparison table. GPU recommended; torch assumed pre-installed.
```bash
nohup bash unets/run_all.sh &            # progress: tail -f ~/tradseg-run/run_all_*.log
nohup bash unets/run_5fold.sh &          # same, forced FOLDS="0 1 2 3 4" (fetches run_all.sh)
# knobs: FOLDS="0" EPOCHS=150 WEAK_FRAC=0.2 MC=30 UNC=entropy FORCE=1
```
Only the U-Net (no hybrids): `nohup bash unets/run_unet.sh &`.

**Manual (one fold):**
```bash
export DATA_DIR=$PWD/data/Task09_Spleen
python3 unets/train.py --tag mcdropout --fold 0 --dropout-p 0.4 --preprocessed-dir $DATA_DIR/preprocessed_A --out-dir results/unets
python3 unets/test.py  --tag mcdropout --fold 0 --dropout-p 0.4 --preprocessed-dir $DATA_DIR/preprocessed_A --weights-dir results/unets --out-dir results/unets --advanced
python3 unets/infer.py --tag mcdropout --fold 0 --mc-samples 30 --preprocessed-dir $DATA_DIR/preprocessed_A --weights-dir results/unets --out-dir results/unets
python3 run_hybrid.py  --tag mcdropout --fold 0 --mode uncertainty --refiner both --preprocessed-dir $DATA_DIR/preprocessed_A --weights-dir results/unets --out-dir results/hybrid
python3 agg_compare.py --unet-dir results/unets --hybrid-dir results/hybrid --out results/comparison
```

---

## What each file does, and where its output goes

**Core traditional pipeline (root):**

| File | Role | Produces → location |
|---|---|---|
| `config.py` | paths + fixed-window "data contract" (env-overridable) | (imported) |
| `preprocessing.py` | raw NIfTI → `(2,Z,256,256)` npy + splits, per `--track` | `data/<TASK>/preprocessed_<track>/*.npy`, `data/<TASK>/splits.pkl` |
| `io_utils.py` | load preprocessed npy + fold splits | (imported) |
| `seeding.py` | AUTO spatial+intensity prior (from train) / ORACLE markers & seeds | (imported) |
| `postprocess.py` | 3D largest-CC, small-object removal, hole fill, prior/overlap component selection | (imported) |
| `metrics.py` | per-volume Dice / Jaccard / HD95 / ASSD (baseline-matching NaN rules) | (imported) |
| `methods/` | one `Segmenter` class per method (see method table in the report) | (imported) |
| `texture_utils.py` | Tier-3 feature ops (Perona-Malik, Gabor bank, Teager energy, granulometry) | (imported) |
| `list_methods.py` | print a tier's registered method names (used by `run_tier*.sh`) | stdout |
| `run_experiment.py` | run ONE method × fold × regime × track → score per-volume | `results/tier<N>/<method>/<stem>.json` |

**Neural + hybrid pipeline:**

| File | Role | Produces → location |
|---|---|---|
| `unets/arch.py` | `MCDropoutUNet` architecture (torch-only) | (imported) |
| `unets/data.py`, `unets/engine.py`, `unets/losses.py` | loaders, train/eval loop (Dice+CE), losses | (imported) |
| `unets/mc_dropout.py` | enable-dropout / MC forward / uncertainty maps / temperature | (imported) |
| `unets/train.py` | train one net for a fold (skip if `_best.pth` exists) | `results/unets/<tag>_f<F>_best.pth`, `_last.pth` |
| `unets/test.py` | deterministic per-volume Dice | `results/unets/unet_<tag>_f<F>.json` |
| `unets/infer.py` | MC inference → seg + uncertainty maps for the refiner | `results/unets/<tag>_f<F>_mcinfer.json`, `results/unets/preds/*.npz` |
| `unets/predictor.py` | `UNetPredictor.predict()/predict_mc()` — the U-Net↔refiner API | (imported by `run_hybrid.py`) |
| `hybrid.py` | band construction + anchored refiners (random walker / GAC) | (imported) |
| `run_hybrid.py` | run a hybrid for one fold (reports `always`/`selective`/`oracle`) | `results/hybrid/hybrid_<mode>_<refiner>_<tag>_f<F>.json` |
| `agg_compare.py` | pool U-Net + hybrid results into one comparison table | `results/comparison/summary.{csv,json}` |

**Outputs at a glance:** traditional → `results/tier<N>/<method>/`, U-Net weights/scores →
`results/unets/`, hybrids → `results/hybrid/`, final comparison table → `results/comparison/`.

## Notes
- Evaluation is per-volume in the preprocessed 256×256 space **without voxel spacing** (HD95/ASSD
  in resized-voxel units) — identical to the U-Net, hence fair. Dice/Jaccard are the headline.
- The `auto` spatial prior needs a fixed in-plane size → use Track A (or resized B).
