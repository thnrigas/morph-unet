# Morphological U-Net

The project investigates whether morphological variations of the U-Net improve segmentation over the unaltered baseline, in the context of the Medical Segmentation Decathlon.

Morphology is injected at three levels. At the input, residuals are added as extra channels, either precomputed with a fixed structuring element or computed with a trainable structuring element by a morphological block, selected by a per task survey that ranks a library of classical filters (top/bottom-hat, gradient, reconstruction and h-/volume-dome, alternating-sequential and leveling residuals) and autoselects the top channels or custom selection. **Inside the network**, a morphological U-Net replaces the convolution itself with morphological layers, whose stages use depthwise soft erosion/dilation followed by a 1x1 projection, in configurations that morphologise the whole network, only the high-resolution stages or only the bottleneck. On the skip connections, attention gates are built either morphologically (top-hat/bottom-hat saliency) or linearly (kernel cross-attention), so the two can be compared on the same backbone.

All soft-morphology blocks approximate dilation/erosion via logsumexp with a temperature that is annealed during training to avoid the dead gradient problem. Every morphological block ships with a parameter-matched linear control (`convsep`, `convctrl`) so that gains can be attributed to morphology rather than to extra capacity. The pipeline reports segmentation and training-cost/convergence metrics, includes an MC-Dropout branch for predictive uncertainty, and a structural channel-pruning suite for measuring how much of a morphological network is actually load-bearing.

Three further experiment tracks share this codebase and are documented in sections 7-9: **uncertainty quantification and calibration** with MC-Dropout on Hippocampus and Spleen (`uncertainty/`), **loss-function experiments** on Hepatic Vessel (`train_losses.py`), and **traditional (non-neural) segmentation plus U-Net hybrids** on Spleen (`tradseg/`). They share the preprocessing utilities, the seeded split generation (`datasets/create_splits.py`) and the evaluator, so scores are computed the same way. Only the loss track runs on the same MSD task as the morphological experiments, and each track keeps its own training loop and sampling policy — so read each track's comparisons internally rather than against the morphological tables.

Splits are a seeded 5-fold cross validation (`datasets/create_splits.py`); the reported experiments use folds 0-2, and most of the block/pruning exploration ran on fold 0 only. The main task throughout is MSD **Task08 Hepatic Vessel** (vessel + tumour); select another with the `TASK` environment variable.

## Setup

Data available in http://medicaldecathlon.com. Place in `./data/` or add path to config.

```
pip install -r requirements.txt
```

```
python3 run_preprocessing.py
```

Training writes `results/<TASK>/<tag>_f<fold>_best.pth` and, after the final test pass,
`results/<TASK>/<tag>_f<fold>_scores.json`. Everything under `data/` and `results/` is gitignored
(the dataset and the checkpoints are large and stay local).

## 1. Input-level morphology

Baseline Model:
```
for f in 0 1 2; do
    python3 train_eval.py --tag baseline --fold $f
done
```

Static Filters:
```
for f in 0 1 2; do
    python3 train_eval.py --tag staticbank --static-auto --fold $f
done
```

Trainable Filters:
```
for f in 0 1 2; do
    python3 train_eval.py --tag morphbank --morph-bank auto --fold $f
done
```

Parameter-matched conv control for the trainable bank, which isolates "extra learned input channels"
from "morphology":
```
for f in 0 1 2; do
    python3 train_eval.py --tag convctrl --morph-bank auto --conv-control --fold $f
done
```

## 2. Morphological U-Net variants

All variants share the same training loop, loss (Dice + CE), and evaluation; the architecture is
selected with flags. `--morph-unet none` gives a plain-conv U-Net (the base for the attention nets);
any other value places morphology at that capacity `config`.

| flag | meaning |
|---|---|
| `--morph-unet {none,heavy,balanced,bottleneck,deep,full,full_l1,full_l2}` | which U-Net / where morphology sits |
| `--morph-impl {fast,paper,convsep,convmpm}` | the morphological block implementation |
| `--fs N` | base feature width (channels are `N,2N,4N,8N,16N` across levels; default 64) |
| `--morph-k K` | structuring-element / kernel size (we used 3) |
| `--morph-conv-stem` | 3x3 conv stem before any morphology (**off by default**, see note) |
| `--lin-attn`, `--lin-attn-gamma-init G` | linear cross-attention gates on the skips, ReZero init `G` |
| `--morph-attn`, `--morph-attn-warm W` | morphological self-gating on the skips, warm-start strength `W` |

> **Note on `--morph-conv-stem`.** The stem is opt-in here. Runs recorded in `results/results.md`
> predate it and were trained without it, so leaving it off keeps them reproducible. The banner printed
> at the start of each run records every switch that changes the model (`impl=`, `stem`, `fs=`, `attn`,
> `warm=`, `drop=`), which is what distinguishes two runs sharing a tag.

Configurations:
```
for cfg in bottleneck balanced deep; do
    for f in 0 1 2; do
        python3 train_eval.py --tag morphunet_$cfg --morph-unet $cfg --morph-k 3 --fold $f
    done
done
```

### MPM morphological block (`--morph-impl fast`)

The MPM neuron is a per-channel soft-morphology (max-plus dilation + min-plus erosion) followed by a
1x1 projection. `config` decides how many stages are morphological. `heavy` makes all nine stages
morphological; `full_l2` is the full variant with levels 1-2 kept linear.

```
python3 train_eval.py --tag mpm_full_l2 --fold 0 --morph-unet full_l2
```

### ConvSep twin (`--morph-impl convsep`)

The linear control for the MPM block: same per-channel-spatial then 1x1-mixer shape, but the spatial op
is a depthwise 3x3 convolution instead of morphology. This isolates what morphology buys over a plain
separable conv of identical structure.

```
python3 train_eval.py --tag convsep_heavy --fold 0 --morph-unet heavy --morph-impl convsep
```

### ConvMPM twin (`--morph-impl convmpm`)

ConvSep's other twin: keep the depthwise 3x3 spatial op, but make the 1x1 **channel mixer** a
morphological MPM neuron (a `StrictMorph2d` with k=1, max-plus join and min-plus meet). This puts the
morphology in the channel-mixing step. `--fs` is the width knob used to probe capacity.

```
# full width (fs=64, ~6M params)
python3 train_eval.py --tag convmpm_heavy   --fold 0 --morph-unet heavy --morph-impl convmpm --morph-k 3

# 1/5 width (fs=13, ~0.26M params) -- the capacity-floor ablation
python3 train_eval.py --tag convmpm_small   --fold 0 --morph-unet heavy --morph-impl convmpm --morph-k 3 --fs 13

# ~2M params (fs=37)
python3 train_eval.py --tag convmpm_small2m --fold 0 --morph-unet heavy --morph-impl convmpm --morph-k 3 --fs 37
```

Note: the tiny fs=13 ConvMPM does **not** collapse to zero Dice the way a same-width plain morph block
does. The morphological channel mixer is what keeps a floor-width net trainable.

## 3. Attention variants

Both gates start as the identity, so all three arms share the plain-U-Net starting point and the only
difference is the skip gate.

### Linear-attention U-Net (`--lin-attn`)

A plain-conv U-Net with linear (kernel) cross-attention gates on each skip. The gate is a ReZero
residual `y = x + gamma * attn(g, x)`; `--lin-attn-gamma-init` sets the initial `gamma`. `0.0` starts as
a plain U-Net (identity skips); `0.5` warm-starts the gate half-active.

```
# identity start (gamma = 0)
python3 train_eval.py --tag unet_linattn     --fold 0 --morph-unet none --lin-attn

# half-active warm start (gamma = 0.5)
python3 train_eval.py --tag unet_linattn_g05 --fold 0 --morph-unet none --lin-attn --lin-attn-gamma-init 0.5
```

### Morphological-attention U-Net (`--morph-attn`)

The morphological analogue: each skip is gated by its own top-hat / bottom-hat saliency.
`--morph-attn-warm` is the counterpart of the linear `gamma` init; `0.0` starts as identity, `0.5` reads
the top-hat minus bottom-hat contrast at half strength from step 0.

```
# identity-init gate
python3 train_eval.py --tag unet_morphattn     --fold 0 --morph-unet none --morph-attn --morph-k 3

# warm-started gate (analogue of gamma = 0.5)
python3 train_eval.py --tag unet_morphattn_g05 --fold 0 --morph-unet none --morph-attn --morph-k 3 --morph-attn-warm 0.5
```

## 4. Channel pruning (`prune.py`)

`prune.py` loads a trained checkpoint, structurally prunes the input channels of every morphological /
separable unit, reports the Dice drop, fine-tunes to recover, and writes a new pruned model plus test
scores. Pruning an input channel drops that channel's spatial filter and the matching 1x1 mixer column,
so it is local and cascade-free; the reported params/FLOP savings are real.

The load flags must match how the model was trained:

```
python3 prune.py --tag <model> --fold 0 --config <cfg> --impl <impl> --fs <width> \
    --method <criterion> --keep-ratio <k> --alloc global --global-norm max --min-keep <n>
```

Criteria (`--method`):

| method | idea |
|---|---|
| `l1x1` | `\|proj_col\|` x `\|alpha\|` x SE-spread (morph blocks) / depthwise-filter norm (conv twins) |
| `morph` | morphology-native saliency (+ off-centre win-rate); MPM only |
| `lin` | `\|proj_col\|` x `\|alpha\|`, morphology-agnostic output contribution |
| `act` | `\|proj_col\|` x mean\|activation\|, data-driven output contribution |
| `fb` | global HMM forward-backward posterior over the unit chain (details below) |
| `fbfg` | the same fixed `fb`, but statistics restricted to foreground receptive fields |
| `fbnew` | foreground-restricted `act` (the earlier foreground criterion) |
| `random` | uniform-random keep, sanity baseline |

Allocation: `--alloc local` keeps `keep_ratio` of each layer's own channels (uniform sparsity);
`--alloc global` shares one `keep_ratio * total` budget across layers (non-uniform, prunes redundant
layers harder) with a per-layer `--min-keep` floor and `--global-norm` for cross-layer comparability.

Two pruning back-ends exist, selected by which block implementation the checkpoint uses:
`networks/prune_morph.py` handles the `fast` / `convsep` / `convmpm` units (channel-level scoring), and
`networks/morph_prune.py` handles the strict `paper` max-plus layers (path/chain composition).

### The forward-backward criterion (`fb`) and its fixed form

`fb` treats each unit's input channels as HMM states and runs forward-backward over the U-Net's real
routing. The current (fixed) version differs from the first implementation in three ways, all on by
default:

- **Skip edges.** The U-Net skips `enc_k.sub2 -> dec_k.sub1` are added to the graph, so an encoder
  channel's importance reaches the decoder directly, not only through the bottleneck.
- **Emission in the backward pass.** Each state carries a prior/emission `pi_i` proportional to
  `E[morph(i)]`, applied in both passes. Without it the backward messages collapse to uniform and the
  posterior degenerates to the forward pass alone.
- **Residual smoothing where residuals exist.** Add-one (Laplace) smoothing is applied only on the
  stage-boundary edges `X.sub2 -> Y.sub1`, the exact edges the residual `out = sub1(x) + sub2(sub1(x))`
  feeds, and left off on within-stage and skip edges.

Flags to control or revert this:

```
--fb-no-skips           # drop the skip edges
--fb-no-emission        # do not carry pi through the backward pass
--fb-residual-smooth S  # residual add-one strength (default 1.0, 0 = off)
--fb-legacy             # reproduce the old linear-chain fb exactly (no skips, no emission, no smoothing)
```

`fbfg` is `fb` with the state statistics measured only over foreground positions (the `seg > 0` mask
max-pooled to each unit's resolution). It accepts the same `--fb-*` flags. On the floor-width ConvMPM
this holds real Dice at 38 to 50 percent of params removed where plain `fb` collapses to zero, since the
foreground restriction keeps the vessel-relevant channels.

### Example sweeps

Global sweep on the linear ConvSep (fold 0), with the fixed `fb` and its foreground form:

```
for m in fb fbfg; do
  for k in 0.01 0.03 0.05 0.10 0.50; do
    python3 prune.py --tag convsep_heavy --fold 0 --config heavy --impl convsep --fs 64 \
        --method $m --keep-ratio $k --alloc global --global-norm max --min-keep 4
  done
done
```

Global sweep on the MPM full_l2 model:

```
for m in fb fbfg; do
  for k in 0.01 0.03 0.05 0.10 0.30 0.50 0.70; do
    python3 prune.py --tag mpm_full_l2 --fold 0 --config full_l2 --impl fast --fs 64 \
        --method $m --keep-ratio $k --alloc global --global-norm max --min-keep 4
  done
done
```

Pruning the floor-width ConvMPM (fold 0), `fb` vs `fbfg` at three keep ratios:

```
for m in fb fbfg; do
  for k in 0.5 0.3 0.1; do
    python3 prune.py --tag convmpm_small --fold 0 --config heavy --impl convmpm --fs 13 \
        --method $m --keep-ratio $k --alloc global --global-norm max --min-keep 2
  done
done
```

## 5. Robustness and data efficiency

Eval-time stress tests reuse trained weights (no retraining) and write to a perturb-suffixed tag so the
clean run is not clobbered:

```
for s in 0.05 0.1 0.2 0.3; do
    python3 train_eval.py --tag baseline --fold 0 --test-only --fast-eval \
        --test-perturb noise --perturb-strength $s
done
```

`--test-perturb` also accepts `gamma` and `contrast`. Data-efficiency points retrain on a nested subset
of the training split:

```
for N in 5 15 30 60; do
    python3 train_eval.py --tag baseline --fold 0 --train-cases $N
done
```

## 6. Aggregating and comparing results

```
for t in baseline staticbank morphbank convctrl morphunet_bottleneck morphunet_balanced morphunet_deep; do
    python3 train_eval.py --fold-mean $t
done
```

```
python3 train_eval.py --compare baseline_mean_scores.json staticbank_mean_scores.json \
    morphbank_mean_scores.json convctrl_mean_scores.json \
    morphunet_bottleneck_mean_scores.json morphunet_balanced_mean_scores.json \
    morphunet_deep_mean_scores.json
```

## 7. Uncertainty quantification and calibration (`uncertainty/`)

Predictive uncertainty of a plain U-Net via **MC-Dropout**, and whether **calibration** improves it.
Two MSD datasets run independently: **Hippocampus** (Task04, the un-suffixed scripts) and
**Spleen** (Task09, the `_spleen` scripts). For each, two models are compared:

- `mcdropout` — baseline MC-Dropout U-Net (Dice + CE)
- `mcdropout_cal` — train-time calibrated variant (Dice + CE + λ·soft-binned-ECE)

Both are reported as **Dice** (deterministic) plus foreground/macro **ECE**, before and after
post-hoc **temperature scaling**. The network is `networks/UNET_mc.py` (baseline depth/filters,
`nn.Dropout2d` at the two deep encoder skips, the bottleneck and the two deep decoder blocks,
kept active at inference).

Training runs on this track's own loop (`uncertainty/mc_common.py`), not `train_eval.py`'s: no
gradient clipping and no class-balanced foreground oversampling, so the reported calibration
numbers are not perturbed by a training class prior that differs from the test one. That also
keeps the track free of any morphological import.

One command per dataset:

```
nohup bash run/run_all_hippocampus.sh &
nohup bash run/run_all_spleen.sh &
```

Manual, one fold (swap the `_spleen` scripts and `TASK` for spleen):

```
export TASK=Task04_Hippocampus

python3 uncertainty/run_preprocessing_mc.py

python3 uncertainty/train_mc.py --fold 0 --tag mcdropout --dropout-p 0.4 --out-dir results
python3 uncertainty/train_mc_recalibrate.py --fold 0 --tag mcdropout_cal --dropout-p 0.4 --cal-weight 1.0 --out-dir results

python3 uncertainty/test_mc.py --fold 0 --tag mcdropout --out-dir results
python3 uncertainty/uncertainty_mc.py --fold 0 --tag mcdropout --out-dir results --mc-samples 30 --temperature 1.0 --save-volumes
python3 uncertainty/calibrate_mc.py --fold 0 --tag mcdropout --out-dir results
python3 uncertainty/uncertainty_mc.py --fold 0 --tag mcdropout --out-dir results --mc-samples 30 --save-volumes
```

`calibrate_mc*.py` fits a single temperature on the val fold (with a `[0.5, 10]` safety rail,
`utilities/mc_calibration.py`) and writes `<tag>_f<fold>_temperature.json`; `uncertainty_mc*.py`
auto-loads it and writes a separate `_cal` output set, so raw and recalibrated sit side by side.
The spleen track adds `datasets/two_dim/NumpyDataLoader_spleen.py`, which trains only on
organ-bearing slices (widened by `--fg-margin` neighbours) — most abdominal CT slices contain no
spleen — while val/test still score every slice.

## 8. Loss-function experiments (`train_losses.py`)

Does swapping **only the loss** beat Dice + CE on the thin, tree-like structures of Task08
Hepatic Vessel? Same network (`networks/UNET.py`), budget, seed and data across combos.
Reported before and after 3D connected-component post-processing.

- `dice_ce` — baseline (Soft Dice + CE)
- `ftversky_ce_boundary` — Focal-Tversky + CE + λ·Boundary (SDF)
- `cldice_dice_ce` — clDice + Dice + CE (shipped; not in the default driver)

```
nohup bash run/run_losses.sh &
```

Manual, one fold:

```
export TASK=Task08_HepaticVessel

python3 run_preprocessing_losses.py

python3 train_losses.py --loss dice_ce --fold 0 --out-dir results
python3 train_losses.py --loss ftversky_ce_boundary --fold 0 --out-dir results

python3 test_losses.py --tag dice_ce --fold 0 --pp-min-size 50 --out-dir results
python3 test_losses.py --tag ftversky_ce_boundary --fold 0 --pp-min-size 50 --out-dir results

python3 losses_fold_eval.py --fold-mean dice_ce
python3 losses_fold_eval.py --fold-mean ftversky_ce_boundary
python3 losses_fold_eval.py --compare dice_ce_mean_scores.json ftversky_ce_boundary_mean_scores.json
```

Shared plumbing is `pipeline_loss.py` (loaders, train/val/test engine); the losses are
`loss_functions/{boundary,cldice,tversky}_loss.py`. `losses_fold_eval.py` is the aggregator for
this track — it reads the flat `results/`, unlike `train_eval.py --fold-mean`, which reads
`results/<TASK>/`.

This track shares the task, network, image preprocessing, splits and evaluator with the
morphological experiments, and `run_losses.sh` uses matching hyperparameters (patch 128,
batch 16, lr 2e-4, 400 epochs, patience 30, grad clip 5.0, fg-fraction 0.33). It still trains
on its own engine and its own loader fork, whose foreground crop puts the target at a random
position in the patch instead of centring it, so the training crops are not the same draws as
`train_eval.py`'s. Read the loss comparisons against this track's own `dice_ce` row rather
than against the morphological tables.

## 9. Traditional segmentation, U-Net comparison and hybrids (`tradseg/`)

Classical (non-neural) segmentation for MSD Task09 Spleen, scored per volume in the **same
256×256 space and 5-fold CV** as the baseline U-Net, so the numbers are directly comparable —
plus **hybrids** where the U-Net is refined by a traditional method (random walker / level set)
inside an uncertainty- or morphology-anchored band. Self-contained subfolder with its own
`config.py` and `requirements.txt`; run the scripts from inside it. Full detail in
[`tradseg/README.md`](tradseg/README.md).

```
pip install -r tradseg/requirements.txt # + tradseg/unets/requirements-unet.txt for B/C
cd tradseg

nohup bash run_tier1.sh & # otsu, multiotsu, kmeans, gmm, region_growing, watershed
nohup bash run_tier2.sh & # chanvese, morphgac, graphcut, random_walker
nohup bash run_tier3.sh & # gabor, amfm, granulometry

nohup bash unets/run_all.sh & # U-Net baseline + MC-Dropout + both hybrids, one comparison table
```

## Attribution & License

Derived from the MIC-DKFZ [`basic_unet_example`](https://github.com/MIC-DKFZ/basic_unet_example). 

Copyright © German Cancer Research Center (DKFZ), Division of Medical Image Computing (MIC). Licensed under the Apache License 2.0. 

This is a modified derivative work, original per-file copyright headers are retained. [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## References

[1] Antonelli, M., et al. "The Medical Segmentation Decathlon." Nature Communications, 2022.

[2] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation. " International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.

[3] Çiçek, Özgün, et al. "3D U-Net: learning dense volumetric segmentation from sparse annotation. "International conference on medical image computing and computer-assisted intervention. Springer, Cham, 2016.
