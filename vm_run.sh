#!/usr/bin/env bash
#
# L4-VM job queue (Google Cloud). Runs, IN PRIORITY ORDER:
#   PHASE 1  prune the CONVSEP twin on its best fold (fold 1) -- the plain-conv model,
#            criteria lin/act/fb (local+global) + random(local), escalation + free-lunch skip.
#   PHASE 2  prune DEEP & BOTTLENECK on their remaining folds (1,2) -- morphological pruning,
#            same scheme set as the main sweep (l1x1/lin/act/fb local+global + random local).
#   PHASE 3  train DEEP & BOTTLENECK WITH DROPOUT (p=0.2) on folds 0,1,2 -- regularised variants.
#
# NOT here: pruning the full morphological (heavy) model -- that runs on the LOCAL machine's
# sweep already. Add it back (see the heavy block at the bottom, commented) only if you want it
# on the VM too.
#
# Everything is RESUME-AWARE (a finished run leaves results/<stem>_prune.json / _best.pth and is
# skipped) and CHECKPOINT-GUARDED (a prune step whose base <tag>_f<fold>_best.pth is missing is
# skipped with a warning instead of crashing -- so copy the base checkpoints first, see VM_SETUP.md).
#
# One job at a time. No `set -e`: a failed run logs and the queue continues.
#
# Usage:  nohup ./vm_run.sh > vm_run.log 2>&1 &   ;  tail -f vm_run.log
#
set -uo pipefail
cd "$(dirname "$0")"
RESULTS=results

EPOCHS=80
LR=5e-5
MIN_KEEP=4
GLOBAL_NORM=max
GOOD_TOL=0.01          # finetuned   >= unpruned - GOOD_TOL   -> "good", stop escalating
PRUNE_TOL=0.015        # pruned(noFT) >= unpruned - PRUNE_TOL  -> free lunch: keep as-is, stop
KEEPS=(0.1 0.3 0.5 0.7)

# ------------------------------------------------------------------ helpers
is_good() {  # $1=json  $2=ft_tol  $3=prune_tol   -> echoes "good"/"bad"
  python3 - "$1" "$2" "$3" <<'PY'
import json, sys
path, ft_tol, prune_tol = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
try:
    d = json.load(open(path))
    base, pruned, ft = d.get("val_dice_unpruned"), d.get("val_dice_pruned"), d.get("val_dice_finetuned")
    free      = base is not None and pruned is not None and (base - pruned) <= prune_tol
    recovered = base is not None and ft     is not None and (ft - base)     >= -ft_tol
    print("good" if (free or recovered) else "bad")
except Exception:
    print("bad")
PY
}

have_ckpt() { [[ -f "$RESULTS/${1}_f${2}_best.pth" ]]; }   # $1=tag $2=fold

n_run=0; n_skip=0; t_start=$(date +%s)

# prune one (tag,config,impl,fold) over a scheme list, with escalation + free-lunch skip.
# schemes are passed as "METHOD ALLOC" strings; random is handled separately (no escalation).
prune_model() {
  local TAG=$1 CFG=$2 IMPL=$3 FOLD=$4; shift 4
  local SCHEMES=("$@")
  if ! have_ckpt "$TAG" "$FOLD"; then
    echo "!!! SKIP prune $TAG f$FOLD: base checkpoint results/${TAG}_f${FOLD}_best.pth MISSING (copy it over)"
    return
  fi
  echo "=================================================================="
  echo "== PRUNE  $TAG  (config=$CFG impl=$IMPL fold=$FOLD)"
  echo "=================================================================="
  for scheme in "${SCHEMES[@]}"; do
    read -r METHOD ALLOC <<< "$scheme"
    SUFFIX=""; [[ "$ALLOC" == "global" ]] && SUFFIX="g"
    MTAG="${METHOD}${SUFFIX}"
    echo "------ $TAG f$FOLD  scheme=$MTAG ($METHOD/$ALLOC) ------"
    for KEEP in "${KEEPS[@]}"; do
      KK=$(printf "k%02d" "$(python3 -c "print(round($KEEP*100))")")
      STEM="${TAG}_prune-${MTAG}-${KK}_f${FOLD}"
      JSON="$RESULTS/${STEM}_prune.json"
      if [[ -f "$JSON" ]]; then
        status=$(is_good "$JSON" "$GOOD_TOL" "$PRUNE_TOL")
        echo "    keep=$KEEP already done ($status) -> skip"; n_skip=$((n_skip+1))
      else
        echo "    keep=$KEEP RUNNING ($STEM)"
        FLAGS=(--alloc "$ALLOC" --min-keep "$MIN_KEEP")
        [[ "$ALLOC" == "global" ]] && FLAGS+=(--global-norm "$GLOBAL_NORM")
        python3 prune.py --tag "$TAG" --config "$CFG" --impl "$IMPL" --fold "$FOLD" \
            --method "$METHOD" --keep-ratio "$KEEP" "${FLAGS[@]}" \
            --skip-ft-if-within "$PRUNE_TOL" --finetune-epochs "$EPOCHS" --lr "$LR" \
          || echo "    !!! $STEM FAILED (exit $?) -- continuing"
        n_run=$((n_run+1)); status=$(is_good "$JSON" "$GOOD_TOL" "$PRUNE_TOL")
      fi
      [[ "$status" == "good" ]] && { echo "    -> good at keep=$KEEP; skip larger ratios for $MTAG"; break; }
    done
  done
  # random baseline: local only, every keep-ratio, NO escalation, ALWAYS fine-tune
  for KEEP in "${KEEPS[@]}"; do
    KK=$(printf "k%02d" "$(python3 -c "print(round($KEEP*100))")")
    STEM="${TAG}_prune-random-${KK}_f${FOLD}"
    [[ -f "$RESULTS/${STEM}_prune.json" ]] && { echo "    random keep=$KEEP done -> skip"; n_skip=$((n_skip+1)); continue; }
    echo "    random keep=$KEEP RUNNING ($STEM)"
    python3 prune.py --tag "$TAG" --config "$CFG" --impl "$IMPL" --fold "$FOLD" \
        --method random --keep-ratio "$KEEP" --alloc local --min-keep "$MIN_KEEP" \
        --finetune-epochs "$EPOCHS" --lr "$LR" \
      || echo "    !!! $STEM FAILED (exit $?) -- continuing"
    n_run=$((n_run+1))
  done
}

# convsep has no morphology -> only the agnostic criteria transfer (l1x1 falls back to dw-norm).
CONVSEP_SCHEMES=("lin local" "act local" "fb local" "lin global" "act global" "fb global")
# morph models: the full scheme set (morph criterion dropped earlier as it underperformed).
MORPH_SCHEMES=("l1x1 local" "lin local" "act local" "fb local" \
               "l1x1 global" "lin global" "act global" "fb global")

# ================================================================== PHASE 1
echo "##### PHASE 1: prune convsep_heavy (best fold = f1) #####"
prune_model convsep_heavy heavy convsep 1 "${CONVSEP_SCHEMES[@]}"

# ================================================================== PHASE 1b
echo "##### PHASE 1b: MATCHED attention comparison -- same config=none backbone, folds 0,1,2 #####"
# Three arms, IDENTICAL plain-conv U-Net backbone + schedule (LR/epochs/patience defaults),
# so the ONLY difference is the skip mechanism. Both attention gates now start as IDENTITY
# (morph gate = 2*sigmoid; linear gate = ReZero gamma=0), i.e. from the same plain-U-Net point.
train_arm() {  # $1=tag  $2..=extra train_eval flags
  local TAG=$1; shift
  for FOLD in 0 1 2; do
    if [[ -f "$RESULTS/${TAG}_f${FOLD}_best.pth" ]]; then
      echo "    $TAG f$FOLD already trained -> skip"; n_skip=$((n_skip+1)); continue
    fi
    echo "== TRAIN $TAG f$FOLD =="
    python3 train_eval.py --tag "$TAG" --fold "$FOLD" "$@" \
      || echo "    !!! $TAG f$FOLD FAILED (exit $?) -- continuing"
    n_run=$((n_run+1))
  done
}
train_arm unet_baseline   --morph-unet none                          # no-attention reference
train_arm unet_morphattn  --morph-unet none --morph-attn --morph-k 3 # morphological attention
train_arm unet_linattn    --lin-attn --lin-attn-heads 4              # linear attention

# ================================================================== PHASE 2
echo "##### PHASE 2: prune deep & bottleneck, folds 1,2 #####"
for FOLD in 1 2; do
  prune_model mpm_deep       deep       fast "$FOLD" "${MORPH_SCHEMES[@]}"
  prune_model mpm_bottleneck bottleneck fast "$FOLD" "${MORPH_SCHEMES[@]}"
done

# ================================================================== PHASE 3
echo "##### PHASE 3: train deep & bottleneck WITH DROPOUT p=0.2, folds 0,1,2 #####"
for CFG in deep bottleneck; do
  for FOLD in 0 1 2; do
    TAG="mpm_${CFG}_do"                       # "_do" = dropout variant
    if [[ -f "$RESULTS/${TAG}_f${FOLD}_best.pth" ]]; then
      echo "    $TAG f$FOLD already trained -> skip"; n_skip=$((n_skip+1)); continue
    fi
    echo "== TRAIN $TAG f$FOLD (config=$CFG dropout=0.2) =="
    python3 train_eval.py --tag "$TAG" --morph-unet "$CFG" --morph-k 3 \
        --morph-dropout 0.2 --fold "$FOLD" \
      || echo "    !!! $TAG f$FOLD FAILED (exit $?) -- continuing"
    n_run=$((n_run+1))
  done
done

# ================================================================== (optional) heavy on the VM
# The local machine's sweep already prunes morphunet_heavy. Uncomment to ALSO do it here:
# prune_model morphunet_heavy heavy fast 0 "${MORPH_SCHEMES[@]}"

echo "=================================================================="
echo "VM QUEUE DONE: $n_run runs, $n_skip skipped, in $(( ($(date +%s)-t_start)/60 )) min."
python3 collate_prune.py || true
