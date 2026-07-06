#!/usr/bin/env bash
#
# Prune + fine-tune sweep for three trained MorphUNets.
# ----------------------------------------------------
# Models (tag  config  fold):
#     mpm_full_l2    full_l2     fold 0
#     mpm_deep       deep        fold 2
#     mpm_bottleneck bottleneck  fold 2
# Methods:  l1x1 (1x1-weighted L1)  and  morph (morphology-native)
# Keep ratios: 0.1 0.3 0.5 0.7
# => 3 models x 2 methods x 4 ratios = 24 prune+finetune runs (12 per method).
#
# One job at a time -> keeps the 16 GB GPU (and the desktop UI) sane.
# Usage:  ./prune_batch.sh            (foreground)
#         nohup ./prune_batch.sh > prune_batch.log 2>&1 &   (background, then: tail -f prune_batch.log)
#
set -euo pipefail
cd /home/Kasimatis/Documents/kasimat/morph-unet

EPOCHS=40
LR=5e-5

# "tag config fold" triples
MODELS=(
  "mpm_full_l2    full_l2     0"
  "mpm_deep       deep        2"
  "mpm_bottleneck bottleneck  2"
)
METHODS=(l1x1 morph)
KEEPS=(0.1 0.3 0.5 0.7)

n=0
total=$(( ${#MODELS[@]} * ${#METHODS[@]} * ${#KEEPS[@]} ))
t_start=$(date +%s)

for entry in "${MODELS[@]}"; do
  read -r TAG CFG FOLD <<< "$entry"
  for METHOD in "${METHODS[@]}"; do
    for KEEP in "${KEEPS[@]}"; do
      n=$((n + 1))
      echo "=================================================================="
      echo ">>> [$n/$total] $TAG (config=$CFG) fold=$FOLD  method=$METHOD  keep=$KEEP"
      echo "=================================================================="
      python prune.py \
          --tag "$TAG" --config "$CFG" --fold "$FOLD" \
          --method "$METHOD" --keep-ratio "$KEEP" \
          --finetune-epochs "$EPOCHS" --lr "$LR"
    done
  done
done

echo "=================================================================="
echo "ALL $total RUNS DONE in $(( ($(date +%s) - t_start) / 60 )) min. Results in results/*_prune.json"
