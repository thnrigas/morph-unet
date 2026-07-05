#!/usr/bin/env bash
#
# Train the balanced morphological U-Net with 3-fold cross-validation, then aggregate
# the per-fold scores into a mean ± std. Defaults (conv stem, beta-warmup, InstanceNorm,
# max-pool, gradient checkpointing) are baked into the model, so nothing extra is needed.
#
# Usage:
#   ./run_balanced_3fold.sh                          # tag=mpm_balanced, folds 0 1 2
#   ./run_balanced_3fold.sh my_tag                   # custom run tag
#   ./run_balanced_3fold.sh my_tag --morph-half      # any extra flags are passed through
#   ./run_balanced_3fold.sh my_tag --resume          # resume interrupted folds
#
set -euo pipefail
cd "$(dirname "$0")"

TAG="${1:-mpm_balanced}"
shift || true                 # remaining args are forwarded verbatim to train_eval.py
EXTRA=("$@")
FOLDS=(0 1 2)

echo "==> balanced morph U-Net | tag=${TAG} | folds=${FOLDS[*]} | extra: ${EXTRA[*]:-none}"
START=$(date +%s)

for f in "${FOLDS[@]}"; do
  echo ""
  echo "==================== FOLD ${f} ===================="
  python3 train_eval.py --tag "${TAG}" --morph-unet balanced --morph-k 3 --fold "${f}" "${EXTRA[@]}"
done

echo ""
echo "==================== AGGREGATE ===================="
python3 train_eval.py --fold-mean "${TAG}"

echo ""
echo "==> done in $(( ($(date +%s) - START) / 60 )) min. Scores: results/${TAG}_f*_scores.json + fold-mean"
