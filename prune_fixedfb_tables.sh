#!/usr/bin/env bash
#
# Redo the GLOBAL fb prune column for the presentation models with the FIXED forward-backward, and
# add the fbfg (foreground-restricted fixed fb) column -- "so the numbers are more close to reality".
#   * fb   : overwrites the OLD buggy-fb results (beta-collapse / no skips / no residual / no pi).
#   * fbfg : new column.
# Faithful to each model's original prune settings: global alloc, max-norm, min-keep=4, 80 ft epochs.
#
# Chains AFTER the convmpm_small prunes: waits on the resume-watcher PID (which itself waits on the
# 2M training) so this heavy queue never competes with training. A per-run *_fixedfb.done marker makes
# it idempotent for the FIXED runs (a restart skips finished ones) while still replacing the old fb.
#
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WAIT_PID=495003   # prune_convmpm_small_resume.sh (waits on 2M training, then runs convmpm_small prunes)
echo "############ fixedfb-tables: waiting for convmpm_small prune chain PID $WAIT_PID $(date '+%F %T') ############"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "############ chain clear -> starting fixed-fb / fbfg table redo alone $(date '+%F %T') ############"
sleep 10

run() {  # tag config impl method keep
  local tag=$1 cfg=$2 impl=$3 M=$4 K=$5
  local kk; kk=$(printf "k%02d" "$(python3 -c "print(round($K*100))")")
  local stem="${tag}_prune-${M}g-${kk}_f0"
  if [[ -f "results/${stem}_fixedfb.done" ]]; then echo "== skip ${stem} (fixed already done) =="; return; fi
  echo "############ $stem  ($M keep=$K, global max min-keep=4)  $(date '+%F %T') ############"
  if python3 prune.py --tag "$tag" --fold 0 --config "$cfg" --impl "$impl" --fs 64 \
       --method "$M" --keep-ratio "$K" --alloc global --global-norm max --min-keep 4; then
    touch "results/${stem}_fixedfb.done"
  else
    echo "!!! $stem FAILED (exit $?) -- continuing"
  fi
}

for M in fb fbfg; do
  for K in 0.01 0.03 0.05 0.10 0.30 0.50 0.70; do run mpm_full_l2 full_l2 fast "$M" "$K"; done
  for K in 0.01 0.03 0.05 0.10 0.50;           do run convsep_heavy heavy convsep "$M" "$K"; done
done
echo "############ fixed-fb / fbfg table redo DONE $(date '+%F %T') ############"
