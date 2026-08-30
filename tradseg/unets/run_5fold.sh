#!/usr/bin/env bash
#
# ONE-SHOT overnight FULL 5-FOLD run on a FRESH VM (nothing pre-run -> everything from scratch).
#
# Thin wrapper around unets/run_all.sh with FOLDS="0 1 2 3 4" baked in. It fetches the main
# orchestrator from the repo and runs the complete experiment for all 5 folds:
#   per fold: train+test  baseline (pure) + mcdropout (p=0.4) + weak02 (pure, frac 0.2)
#             + MC infer (mcdropout) + Hybrid #1 (uncertainty, rw+ls) + Hybrid #2 (morph, rw+ls)
#   then: aggregate everything -> results/comparison/summary.{csv,json}
# = 15 trainings (3 nets x 5 folds) + 20 hybrid runs. Idempotent / skip-if-exists, so if it dies
# mid-way you can just re-run it and it resumes.
#
# ASSUMES a GCP Deep Learning VM: torch (+CUDA) PRE-INSTALLED. Everything else is pip-installed.
#
# Bootstrap on the fresh VM (two lines):
#   curl -O https://raw.githubusercontent.com/thnrigas/morph-unet/main/tradseg/unets/run_5fold.sh
#   nohup bash run_5fold.sh &          # progress: tail -f ~/tradseg-run/run_all_*.log
#
# Env passthrough (all optional): EPOCHS WEAK_FRAC MC UNC BRANCH WORKDIR ...  (see run_all.sh)
#
set -uo pipefail

export FOLDS="${FOLDS:-0 1 2 3 4}"
BRANCH="${BRANCH:-main}"
RAW="https://raw.githubusercontent.com/thnrigas/morph-unet/${BRANCH}/tradseg/unets/run_all.sh"

echo "### 5-fold launcher: fetching run_all.sh from '$BRANCH', FOLDS='$FOLDS' ###"
curl -fsSL "$RAW" -o run_all.sh || { echo "FATAL: could not fetch run_all.sh from $RAW"; exit 1; }
exec bash run_all.sh
