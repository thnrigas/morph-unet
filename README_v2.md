# Morphological U-Net, v2: compact blocks, attention, and channel pruning

This is the second-stage README. The [original Readme](Readme.md) covers the morphological-residual
experiments (baseline, static/trainable top-/bottom-hat residuals, morphological loss). This document
covers everything added on top of that: the compact morphological U-Net blocks (MPM, ConvSep, ConvMPM),
the two attention variants (linear and morphological), and the structural channel-pruning suite, with
the exact command used for every example we ran.

Target task throughout is MSD **Task08 Hepatic Vessel** (vessel + tumour), 3 classes, 5-fold cross
validation. Most of the exploration below ran on fold 0.

## Setup

Same as v1. Get the data from http://medicaldecathlon.com, put it under `./data/`, then:

```
pip install -r requirements.txt
python3 run_preprocessing.py
```

Training writes `results/<tag>_f<fold>_best.pth` and, after the final test pass,
`results/<tag>_f<fold>_scores.json`. Everything under `data/` and `results/` is gitignored (the
checkpoints and the dataset are large and stay local).

## 1. Architecture variants (`train_eval.py`)

All variants share the same training loop, loss (Dice + CE), and evaluation. You select an architecture
with a handful of flags. `--morph-unet none` gives a plain-conv U-Net (the base for the attention nets);
any other value selects a morphological U-Net at that capacity `config`.

Common flags:

| flag | meaning |
|---|---|
| `--morph-unet {none,heavy,balanced,bottleneck,deep,full,full_l1,full_l2}` | which U-Net / where morphology sits |
| `--morph-impl {fast,paper,convsep,convmpm}` | the morphological block implementation |
| `--fs N` | base feature width (channels are `N,2N,4N,8N,16N` across levels; default 64) |
| `--morph-k K` | structuring-element / kernel size (we used 3) |
| `--lin-attn`, `--lin-attn-gamma-init G` | linear cross-attention gates on the skips, ReZero init `G` |
| `--morph-attn`, `--morph-attn-warm W` | morphological self-gating on the skips, warm-start strength `W` |

### MPM morphological U-Net (`--morph-impl fast`)

The MPM neuron is a per-channel soft-morphology (max-plus dilation + min-plus erosion) followed by a
1x1 projection. `config` decides how many stages are morphological. `heavy` makes all nine stages
morphological; `full_l2` is the L2-regularised full variant.

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
morphology in the channel-mixing step. `--fs` is the width knob we used to probe capacity.

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

The morphological analogue: each skip is gated by its own top-hat / bottom-hat saliency. `--morph-attn-warm`
is the morphological counterpart of the linear `gamma` init; `0.0` starts as identity, `0.5` reads the
top-hat minus bottom-hat contrast at half strength from step 0.

```
# identity-init gate
python3 train_eval.py --tag unet_morphattn     --fold 0 --morph-unet none --morph-attn --morph-k 3

# warm-started gate (analogue of gamma = 0.5)
python3 train_eval.py --tag unet_morphattn_g05 --fold 0 --morph-unet none --morph-attn --morph-k 3 --morph-attn-warm 0.5
```

## 2. Channel pruning (`prune.py`)

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

### Example sweeps we ran

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

## 3. Aggregating and comparing results

Mean over folds and side-by-side comparison work exactly as in v1:

```
python3 train_eval.py --fold-mean convsep_heavy
python3 train_eval.py --compare mpm_full_l2_mean_scores.json convsep_heavy_mean_scores.json
```

The driver scripts we used for the batched sweeps (`prune_*.sh`) are checked in alongside this README;
each is a thin loop over the `prune.py` / `train_eval.py` commands above and records the exact
allocation, keep ratios, and min-keep used for that run.

## Attribution & License

Unchanged from v1: derived from the MIC-DKFZ
[`basic_unet_example`](https://github.com/MIC-DKFZ/basic_unet_example), copyright German Cancer Research
Center (DKFZ), Division of Medical Image Computing (MIC), Apache License 2.0. Original per-file copyright
headers are retained.
