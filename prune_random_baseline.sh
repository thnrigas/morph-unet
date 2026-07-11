#!/usr/bin/env bash
#
# RANDOM-keep baseline on exactly the models / keep ratios where fbfg was run, so every fbfg (and
# fixed-fb) number has its uniform-random control. Global allocation, faithful to each model's fbfg
# settings: full_l2 & convsep use min-keep=4, convmpm_small (fs=13) uses min-keep=2. Random keep + the
# same fine-tune -- the honest "does an informed criterion beat random + finetune?" baseline (seeded).
# Idempotent via a per-run *_random.done marker.
#
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {  # tag config impl fs min_keep keep
  local tag=$1 cfg=$2 impl=$3 fs=$4 mk=$5 K=$6
  local kk; kk=$(printf "k%02d" "$(python3 -c "print(round($K*100))")")
  local stem="${tag}_prune-randomg-${kk}_f0"
  if [[ -f "results/${stem}_random.done" ]]; then echo "== skip ${stem} (done) =="; return; fi
  echo "############ $stem  (random keep=$K, global max min-keep=$mk)  $(date '+%F %T') ############"
  if python3 prune.py --tag "$tag" --fold 0 --config "$cfg" --impl "$impl" --fs "$fs" \
       --method random --keep-ratio "$K" --alloc global --global-norm max --min-keep "$mk"; then
    touch "results/${stem}_random.done"
  else
    echo "!!! $stem FAILED (exit $?) -- continuing"
  fi
}

for K in 0.01 0.03 0.05 0.10 0.30 0.50 0.70; do run mpm_full_l2   full_l2 fast    64 4 "$K"; done
for K in 0.01 0.03 0.05 0.10 0.50;           do run convsep_heavy heavy   convsep 64 4 "$K"; done
for K in 0.10 0.30 0.50;                     do run convmpm_small heavy   convmpm 13 2 "$K"; done
echo "############ random baseline DONE $(date '+%F %T') ############"
