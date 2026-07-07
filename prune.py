#
# Prune + fine-tune driver for the fast morphological U-Net.
# ----------------------------------------------------------
# Loads a trained MorphUNet checkpoint (e.g. results/mpm_full_l2_f0_best.pth), prunes it with the
# chosen scheme, reports the Dice drop, fine-tunes to recover, and writes a NEW pruned model +
# test scores (so it can go straight into train_eval.py --compare / --fold-mean).
#
# Schemes (see networks/prune_morph.py and networks/prune_tropnnc.py):
#   l1x1   : prune morph input channels by ||proj_col|| * |alpha| * spread(SE)   (1x1-weighted)
#   morph  : prune morph input channels by morphology-native saliency (+ off-centre win-rate)
#   lin    : prune by ||proj_col|| * |alpha|  -- morphology-AGNOSTIC output contribution; keeps
#            inert-but-informative linear channels that "morph" would discard
#   act    : prune by ||proj_col|| * E|morph_i(x)|  -- DATA-DRIVEN output contribution (real
#            activation replaces the spread(SE) proxy; needs calibration batches, no gradients)
#   fb     : GLOBAL importance via HMM forward-backward over the morph chain (L1-activation prior +
#            co-activation transitions); posterior gamma=alpha*beta, then keep-ratio (no gradients)
#   tropnnc: TropNNC structured merging of the conv/linear layers (tropical zonotope reduction)
#
# Allocation (--alloc): "local" = uniform keep_ratio per layer; "global" = one shared budget across
# all layers (non-uniform sparsity, prunes redundant layers harder) with a --min-keep floor/layer.
#
# Usage:
#   python prune.py --tag mpm_full_l2 --fold 0 --method l1x1 --keep-ratio 0.5
#   python prune.py --tag mpm_full_l2 --fold 0 --method lin  --keep-ratio 0.3
#   python prune.py --tag mpm_full_l2 --fold 0 --method morph --keep-ratio 0.5 --alloc global --min-keep 4
#   python prune.py --tag mpm_full_l2 --fold 0 --method tropnnc --keep-ratio 0.5
#   python prune.py --tag mpm_full_l2 --fold 0 --method l1x1 --keep-ratio 0.5 --no-finetune
#

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from networks.morph_unet import MorphUNet
from loss_functions.dice_loss import SoftDiceLoss
from train_eval import build_loaders, run_epoch, run_val_dice, evaluate_test, pick_device, set_seed
from networks.prune_morph import prune_morph_channels, count_params


def load_model(tag, fold, device, cfg, impl, conv_stem):
    """Rebuild the MorphUNet exactly as trained and load its best checkpoint (eval/prune mode)."""
    model = MorphUNet(num_classes=config.NUM_CLASSES, in_channels=1, k=3, beta=10.0,
                      config=cfg, impl=impl, conv_stem=conv_stem, checkpoint=False).to(device)
    ckpt = os.path.join(config.PROJECT_ROOT, "results", f"{tag}_f{fold}_best.pth")
    if not os.path.exists(ckpt):
        raise SystemExit(f"no checkpoint {ckpt}")
    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd)
    model.eval()
    return model


def make_calib(train_loader, device, n_batches, bs):
    """A few small calibration tensors (input patches) for the data-driven criteria."""
    calib = []
    for i, batch in enumerate(train_loader):
        if i >= n_batches:
            break
        calib.append(batch["data"][0][:bs].float())
    return calib


def finetune(model, args, train_loader, val_loader, device, out_stem):
    dice_loss = SoftDiceLoss(batch_dice=True, do_bg=False)
    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=args.sched_patience)
    results_dir = os.path.join(config.PROJECT_ROOT, "results")
    best_path = os.path.join(results_dir, f"{out_stem}_best.pth")
    best_dice, best_epoch, t0 = -1.0, 0, time.time()
    for epoch in range(1, args.finetune_epochs + 1):
        tr = run_epoch(model, train_loader, device, dice_loss, ce_loss, optimizer)
        if epoch % args.val_every == 0 or epoch == args.finetune_epochs:
            vd, per = run_val_dice(model, val_loader, device, config.NUM_CLASSES)
            scheduler.step(-vd)
            flag = ""
            if vd > best_dice:
                best_dice, best_epoch = vd, epoch
                torch.save(model, best_path)          # whole object: pruned architecture differs
                flag = "  *best"
            print(f"  ft epoch {epoch:3d}/{args.finetune_epochs}  train={tr:.4f}  "
                  f"val_fgDice={vd:.4f}  [per {' '.join(f'{d:.3f}' for d in per)}]{flag}")
        else:
            print(f"  ft epoch {epoch:3d}/{args.finetune_epochs}  train={tr:.4f}")
        if args.patience and (epoch - best_epoch) >= args.patience:
            print(f"  early stop (no val gain for {epoch - best_epoch} ep)")
            break
    print(f"  fine-tune done in {(time.time()-t0)/60:.1f} min | best fg-Dice {best_dice:.4f} "
          f"@ep{best_epoch} -> {best_path}")
    return best_path, best_dice, best_epoch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="base tag of the trained model (e.g. mpm_full_l2)")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--config", default="full_l2")
    p.add_argument("--impl", default="fast", choices=["fast", "paper", "convsep"],
                   help="'convsep' prunes the depthwise-conv twin (lin/act/fb/random criteria; "
                        "l1x1 falls back to depthwise-filter norm, 'morph' is not valid)")
    p.add_argument("--no-conv-stem", action="store_true", help="model was trained WITHOUT conv stem")
    p.add_argument("--method", required=True,
                   choices=["l1x1", "morph", "lin", "act", "fb", "random", "tropnnc"])
    p.add_argument("--keep-ratio", type=float, default=0.5, help="fraction of channels to KEEP")
    p.add_argument("--alloc", default="local", choices=["local", "global"],
                   help="local = uniform keep_ratio per morph layer; global = one shared budget "
                        "across layers (non-uniform sparsity) with a --min-keep floor per layer")
    p.add_argument("--global-norm", default="max", choices=["none", "max", "mean", "l2", "zscore"],
                   help="per-layer score normalisation before GLOBAL cross-layer ranking")
    p.add_argument("--min-keep", type=int, default=1,
                   help="minimum input channels guaranteed per morph layer (global alloc floor)")
    p.add_argument("--calib-batches", type=int, default=64)
    p.add_argument("--calib-bs", type=int, default=16)   # 64 x 16 = 1024 calibration patches
    p.add_argument("--finetune-epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--patience", type=int, default=24,
                   help="early-stop patience (epochs); must exceed sched-patience*val-every so the "
                        "LR scheduler can actually fire before we stop")
    p.add_argument("--sched-patience", type=int, default=3,
                   help="ReduceLROnPlateau patience in *validation steps* (i.e. sched-patience*"
                        "val-every epochs of no val gain -> halve LR)")
    p.add_argument("--val-every", type=int, default=3)
    p.add_argument("--no-finetune", action="store_true")
    p.add_argument("--skip-ft-if-within", type=float, default=0.0,
                   help="if the PRUNED net (before fine-tuning) is within this Dice of the unpruned "
                        "baseline, keep it as-is and skip fine-tuning entirely (0 = always fine-tune)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=config.HP["batch_size"])
    p.add_argument("--patch-size", type=int, default=config.HP["patch_size"])
    p.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 1, 6))
    p.add_argument("--iters-per-epoch", type=int, default=config.HP["iters_per_epoch"])
    p.add_argument("--fg-fraction", type=float, default=config.HP["fg_fraction"])
    p.add_argument("--val-batch", type=int, default=12)
    p.add_argument("--val-cases", type=int, default=15)
    p.add_argument("--test-only", action="store_true", help="skip fine-tune, just eval the pruned net")
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    train_loader, val_loader, test_loader, _ = build_loaders(args)

    kk = f"k{int(round(args.keep_ratio * 100)):02d}"
    # global allocation gets a "g" suffix on the method tag so its runs don't collide with / are
    # grouped separately from the local ones (e.g. mpm_full_l2_prune-morphg-k50_f0).
    method_tag = args.method + ("g" if args.alloc == "global" else "")
    out_stem = f"{args.tag}_prune-{method_tag}-{kk}_f{args.fold}"
    results_dir = os.path.join(config.PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    model = load_model(args.tag, args.fold, device, args.config, args.impl, not args.no_conv_stem)
    p_before = count_params(model)
    base_dice, _ = run_val_dice(model, val_loader, device, config.NUM_CLASSES)
    print(f"[{out_stem}] unpruned: {p_before/1e6:.3f}M params | val fg-Dice {base_dice:.4f}")

    calib = make_calib(train_loader, device, args.calib_batches, args.calib_bs)
    report = None
    if args.method == "tropnnc":
        from networks.prune_tropnnc import tropnnc_compress
        print(f"[prune] TropNNC structured merge, keep_ratio={args.keep_ratio}")
        tropnnc_compress(model, keep_ratio=args.keep_ratio, verbose=True)
    else:
        print(f"[prune] morph-channel {args.method}, keep_ratio={args.keep_ratio}, "
              f"alloc={args.alloc}" + (f" (norm={args.global_norm}, min_keep={args.min_keep})"
                                       if args.alloc == "global" else ""))
        report = prune_morph_channels(model, criterion=args.method, keep_ratio=args.keep_ratio,
                             calib_batches=calib, device=device, min_keep=args.min_keep,
                             alloc=args.alloc, global_norm=args.global_norm, verbose=True)
    p_after = count_params(model)
    pruned_dice, _ = run_val_dice(model, val_loader, device, config.NUM_CLASSES)
    print(f"[{out_stem}] pruned:   {p_after/1e6:.3f}M params "
          f"({100*(1-p_after/p_before):.1f}% off) | val fg-Dice {pruned_dice:.4f} "
          f"(drop {base_dice-pruned_dice:+.4f})")
    torch.save(model, os.path.join(results_dir, f"{out_stem}_pruned_init.pth"))

    # "free-lunch" prune: if pruning barely dented Dice, keep the pruned net untouched.
    ft_skipped = (args.skip_ft_if_within > 0.0
                  and (base_dice - pruned_dice) <= args.skip_ft_if_within)
    if ft_skipped:
        print(f"[{out_stem}] pruned Dice within {args.skip_ft_if_within:.3f} of baseline "
              f"(drop {base_dice-pruned_dice:+.4f}) -> skipping fine-tune, keeping pruned net")

    ft_dice, ft_best_epoch = None, None
    if not (args.no_finetune or args.test_only or ft_skipped):
        print(f"[{out_stem}] fine-tuning {args.finetune_epochs} ep @ lr={args.lr:g} ...")
        best_path, ft_dice, ft_best_epoch = finetune(model, args, train_loader, val_loader, device, out_stem)
        model = torch.load(best_path, map_location=device, weights_only=False)

    json_path = os.path.join(results_dir, f"{out_stem}_scores.json")
    scores = evaluate_test(model, test_loader, device, json_path, num_workers=args.num_workers)
    summary = {"tag": out_stem, "method": method_tag, "keep_ratio": args.keep_ratio,
               "alloc": args.alloc, "global_norm": args.global_norm, "min_keep": args.min_keep,
               "params_before": p_before, "params_after": p_after,
               "params_off_pct": round(100 * (1 - p_after / p_before), 2),
               "val_dice_unpruned": round(base_dice, 5), "val_dice_pruned": round(pruned_dice, 5),
               "val_dice_finetuned": (round(ft_dice, 5) if ft_dice is not None else None),
               "ft_skipped": ft_skipped,
               "best_epoch": ft_best_epoch, "finetune_epochs": args.finetune_epochs,
               # per-layer surviving input-channel width (which layers keep channels): with GLOBAL
               # allocation this is non-uniform -> shows whether the budget concentrates in early,
               # bottleneck, or late layers. Keyed by morph-unit name in forward order.
               "layer_widths": ({n: r["in_after"] for n, r in report.items()} if report else None),
               "layer_in_before": ({n: r["in_before"] for n, r in report.items()} if report else None)}
    with open(os.path.join(results_dir, f"{out_stem}_prune.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{out_stem}] DONE  params {p_before/1e6:.3f}M -> {p_after/1e6:.3f}M | "
          f"Dice {base_dice:.4f} -> pruned {pruned_dice:.4f}"
          + (f" -> finetuned {ft_dice:.4f}" if ft_dice is not None else "")
          + f"\n  scores: {json_path}")


if __name__ == "__main__":
    main()
