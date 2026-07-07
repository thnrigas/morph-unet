#!/usr/bin/env bash
#
# Full prune + fine-tune sweep: 5 criteria x 2 allocations x 3 models.
# =====================================================================
# Criteria : l1x1  morph  lin  act  fb          (5)
# Alloc    : local  global                       (2)   -> 10 schemes
# Models   : deep -> bottleneck -> full_l2       (in THIS order)
# Keeps    : 0.1 0.3 0.5 0.7                      (per scheme, with escalation)
#
# ESCALATION (the whole point):
#   Within one (model, scheme) we try keep=0.1 first (fewest channels kept =
#   most aggressive). If the fine-tune "recovers" (see GOOD_TOL) we STOP and
#   skip 0.3/0.5/0.7 -- if 0.1 already works, the milder ratios trivially do too.
#   Only if 0.1 falls short do we escalate to 0.3, then 0.5, then 0.7.
#   "good" is calibrated on the deep model: its l1x1 k10 finetuned to +0.0015
#   over the unpruned baseline, so GOOD_TOL=0.01 (within one Dice-point) reproduces
#   "stop at k10" for that reference case.
#
# RESUME / NO-REDO:
#   Any (model,scheme,keep) that already has results/<stem>_prune.json is skipped
#   and its recorded Dice still feeds the escalation decision. A crashed run leaves
#   no json -> it re-runs from scratch (there is no finer resume point than one run).
#
# MIN CHANNELS PER LAYER:
#   --min-keep 4 (floor guaranteed by the global allocator; inert for local here).
#
# One job at a time -> keeps the 16 GB GPU sane. No `set -e`: a failed run logs
# and the sweep continues.
#
# Usage:
#   ./prune_sweep.sh
#   nohup ./prune_sweep.sh > prune_sweep.log 2>&1 &   # then: tail -f prune_sweep.log
#
set -uo pipefail
cd /home/Kasimatis/Documents/kasimat/morph-unet
RESULTS=results

EPOCHS=80
LR=5e-5
MIN_KEEP=4
GLOBAL_NORM=max
GOOD_TOL=0.01          # finetuned >= unpruned - GOOD_TOL           ==> "good", stop escalating
PRUNE_TOL=0.015        # pruned (NO fine-tune) >= unpruned - PRUNE_TOL ==> free lunch: keep as-is,
                       #   skip fine-tune AND stop escalating (passed to prune.py --skip-ft-if-within)

# "tag config fold"   -- ORDER MATTERS: deep -> bottleneck -> full_l2 -> heavy (heaviest last).
# morphunet_heavy = "heavy" config: all 9 enc/dec stages + bottleneck are morphological (18 morph
# layers, ~double full_l2), so it is the slowest per run -> queued last.
MODELS=(
  "mpm_deep        deep        2"
  "mpm_bottleneck  bottleneck  2"
  "mpm_full_l2     full_l2     0"
  "morphunet_heavy heavy       0"
)

# "criterion alloc"
# NOTE: "morph" was DROPPED from the remaining runs -- it underperformed on the models already done,
# so we skip it on full_l2 (global) and heavy to save time. Its completed cells stay on disk / in the
# matrix; it is simply not scheduled again.
SCHEMES=(
  "l1x1  local"
  "lin   local"
  "act   local"
  "fb    local"
  "l1x1  global"
  "lin   global"
  "act   global"
  "fb    global"
)

KEEPS=(0.1 0.3 0.5 0.7)

# --- decide whether a finished run "recovered" ----------------------------
# echoes "good" (stop escalating) if EITHER:
#   free lunch : pruned Dice (no fine-tune) within PRUNE_TOL of the baseline, OR
#   recovered  : fine-tuned Dice within GOOD_TOL of the baseline.
is_good() {  # $1=json  $2=ft_tol  $3=prune_tol
  python - "$1" "$2" "$3" <<'PY'
import json, sys
path, ft_tol, prune_tol = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
try:
    d = json.load(open(path))
    base   = d.get("val_dice_unpruned")
    pruned = d.get("val_dice_pruned")
    ft     = d.get("val_dice_finetuned")
    free      = base is not None and pruned is not None and (base - pruned) <= prune_tol
    recovered = base is not None and ft     is not None and (ft - base)     >= -ft_tol
    print("good" if (free or recovered) else "bad")
except Exception:
    print("bad")   # missing/corrupt -> treat as not-good, keep escalating
PY
}

t_start=$(date +%s)
n_run=0
n_skip=0

for entry in "${MODELS[@]}"; do
  read -r TAG CFG FOLD <<< "$entry"
  echo "##################################################################"
  echo "## MODEL  $TAG  (config=$CFG  fold=$FOLD)"
  echo "##################################################################"

  for scheme in "${SCHEMES[@]}"; do
    read -r METHOD ALLOC <<< "$scheme"
    SUFFIX=""; [[ "$ALLOC" == "global" ]] && SUFFIX="g"
    MTAG="${METHOD}${SUFFIX}"                       # matches prune.py's method_tag

    echo "------------------------------------------------------------------"
    echo ">>> $TAG  scheme=$MTAG  ($METHOD / $ALLOC)"
    echo "------------------------------------------------------------------"

    for KEEP in "${KEEPS[@]}"; do
      KK=$(printf "k%02d" "$(python -c "print(round($KEEP*100))")")
      STEM="${TAG}_prune-${MTAG}-${KK}_f${FOLD}"
      JSON="$RESULTS/${STEM}_prune.json"

      if [[ -f "$JSON" ]]; then
        status=$(is_good "$JSON" "$GOOD_TOL" "$PRUNE_TOL")
        echo "    keep=$KEEP  already done ($status)  -> skip run"
        n_skip=$((n_skip + 1))
      else
        echo "    keep=$KEEP  RUNNING  ($STEM)"
        ALLOC_FLAGS=(--alloc "$ALLOC" --min-keep "$MIN_KEEP")
        [[ "$ALLOC" == "global" ]] && ALLOC_FLAGS+=(--global-norm "$GLOBAL_NORM")
        python prune.py \
            --tag "$TAG" --config "$CFG" --fold "$FOLD" \
            --method "$METHOD" --keep-ratio "$KEEP" \
            "${ALLOC_FLAGS[@]}" \
            --skip-ft-if-within "$PRUNE_TOL" \
            --finetune-epochs "$EPOCHS" --lr "$LR" \
          || echo "    !!! $STEM FAILED (exit $?) -- continuing"
        n_run=$((n_run + 1))
        status=$(is_good "$JSON" "$GOOD_TOL" "$PRUNE_TOL")
      fi

      if [[ "$status" == "good" ]]; then
        echo "    -> good at keep=$KEEP (free-lunch or recovered); skipping larger keep-ratios for $MTAG"
        break
      fi
    done
  done
done

# =====================================================================
# RANDOM sanity baseline -- local only, EVERY keep-ratio, NO escalation,
# ALWAYS fine-tune (no --skip-ft-if-within). If the informed criteria can't
# beat a random keep of the same size, they aren't buying anything.
# 3 models x 4 ratios = 12 runs.
# =====================================================================
echo "##################################################################"
echo "## RANDOM BASELINE  (local, all keep-ratios, always fine-tune)"
echo "##################################################################"
for entry in "${MODELS[@]}"; do
  read -r TAG CFG FOLD <<< "$entry"
  for KEEP in "${KEEPS[@]}"; do
    KK=$(printf "k%02d" "$(python -c "print(round($KEEP*100))")")
    STEM="${TAG}_prune-random-${KK}_f${FOLD}"
    JSON="$RESULTS/${STEM}_prune.json"
    if [[ -f "$JSON" ]]; then
      echo "    random $TAG keep=$KEEP  already done -> skip"
      n_skip=$((n_skip + 1))
      continue
    fi
    echo "    random $TAG keep=$KEEP  RUNNING  ($STEM)"
    python prune.py \
        --tag "$TAG" --config "$CFG" --fold "$FOLD" \
        --method random --keep-ratio "$KEEP" \
        --alloc local --min-keep "$MIN_KEEP" \
        --finetune-epochs "$EPOCHS" --lr "$LR" \
      || echo "    !!! $STEM FAILED (exit $?) -- continuing"
    n_run=$((n_run + 1))
  done
done

echo "=================================================================="
echo "SWEEP DONE: $n_run runs executed, $n_skip skipped, in $(( ($(date +%s) - t_start) / 60 )) min."
echo "Collating..."
python collate_prune.py || true
