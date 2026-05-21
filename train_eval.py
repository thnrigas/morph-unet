#!/usr/bin/env python
"""
Train, Test and Eval for U-Net Segmentation.
"""

import argparse
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from networks.UNET import UNet
from loss_functions.dice_loss import SoftDiceLoss
from datasets.two_dim.NumpyDataLoader import NumpyDataSet
from evaluation.evaluator import aggregate_scores, Evaluator

LABELS = {1: "Anterior", 2: "Posterior"}   # Task04 Hippocampus


def set_seed(seed):
    """Fixed seed so runs share initialisation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loaders(args):
    with open(os.path.join(args.split_dir, "splits.pkl"), "rb") as f:
        splits = pickle.load(f)
    tr, vl, ts = (splits[args.fold]["train"], splits[args.fold]["val"], splits[args.fold]["test"])
    # npy is 4-channel (image, tophat, bottomhat, label)
    # the flags pick which morphological channels are fed to the network.
    input_slice = (0,)
    if args.tophat:
        input_slice += (1,)
    if args.bottomhat:
        input_slice += (2,)
    label_slice = 3
    common = dict(target_size=args.patch_size, batch_size=args.batch_size, input_slice=input_slice,
                  label_slice=label_slice, num_processes=args.num_workers)
    train = NumpyDataSet(args.data_dir, keys=tr, **common)
    val = NumpyDataSet(args.data_dir, keys=vl, mode="val",  do_reshuffle=False, **common)
    test = NumpyDataSet(args.data_dir, keys=ts, mode="test", do_reshuffle=False, **common)
    in_channels = len(input_slice)
    return train, val, test, in_channels


def run_epoch(model, loader, device, dice_loss, ce_loss, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    losses = []
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for batch in loader:
            data = batch["data"][0].float().to(device)       # [b, c, H, W]
            target = batch["seg"][0].long().to(device)       # [b, 1, H, W]
            if train_mode:
                optimizer.zero_grad()
            pred = model(data)
            pred_softmax = F.softmax(pred, dim=1)
            loss = dice_loss(pred_softmax, target.squeeze()) + ce_loss(pred, target.squeeze())
            if train_mode:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def evaluate_test(model, loader, device, json_path):
    model.eval()
    pred_dict, gt_dict = defaultdict(list), defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            data = batch["data"][0].float().to(device)
            target = batch["seg"][0].float().to(device)
            pred = model(data)
            pred_argmax = torch.argmax(pred.data.cpu(), dim=1, keepdim=True)
            for i, fname in enumerate(batch["fnames"]):
                pred_dict[fname[0]].append(pred_argmax[i].numpy())
                gt_dict[fname[0]].append(target[i].detach().cpu().numpy())
    pairs = [(np.stack(pred_dict[k]), np.stack(gt_dict[k])) for k in pred_dict]
    scores = aggregate_scores(
        pairs, evaluator=Evaluator, labels=LABELS,
        json_output_file=json_path, json_author="cv-project",
        json_task="Task04_Hippocampus", advanced=True,
    )
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/Task04_Hippocampus/preprocessed")
    p.add_argument("--split-dir", default="data/Task04_Hippocampus")
    p.add_argument("--tag", required=True)
    p.add_argument("--tophat", action="store_true")
    p.add_argument("--bottomhat", action="store_true")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--test-only", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device()
    train_loader, val_loader, test_loader, in_channels = build_loaders(args)
    model = UNet(num_classes=args.num_classes, in_channels=in_channels).to(device)

    # Output stem encodes the fold so a 5-fold sweep with the same --tag
    # does not overwrite itself (<tag>_f<fold>_{best.pth,last.pth,scores.json})
    stem = f"{args.tag}_f{args.fold}"
    parts = (["tophat"] if args.tophat else []) + (["bottomhat"] if args.bottomhat else [])
    mode = "+".join(parts) if parts else "baseline"
    print(f"[{stem}] device={device} mode={mode} seed={args.seed} "
          f"fold={args.fold} loader_in_ch={in_channels}")
    dice_loss = SoftDiceLoss(batch_dice=True)
    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, "min")

    best_path = f"{stem}_best.pth"
    if not args.test_only:
        best_val = float("inf")
        since_improved = 0
        for epoch in range(1, args.epochs + 1):
            tr = run_epoch(model, train_loader, device, dice_loss, ce_loss, optimizer)
            vl = run_epoch(model, val_loader, device, dice_loss, ce_loss)
            scheduler.step(vl)
            print(f"epoch {epoch:3d}/{args.epochs}  train={tr:.4f}  val={vl:.4f}")
            if vl < best_val:
                best_val = vl
                since_improved = 0
                torch.save(model.state_dict(), best_path)
            else:
                since_improved += 1
            torch.save(model.state_dict(), f"{stem}_last.pth")
            if args.patience and since_improved >= args.patience:
                print(f"early stop: no val improvement for {args.patience} "
                      f"epochs (best val={best_val:.4f} @ ep "
                      f"{epoch - since_improved})")
                break
    ckpt = best_path if os.path.exists(best_path) else f"{stem}_last.pth"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    json_path = f"{stem}_scores.json"
    scores = evaluate_test(model, test_loader, device, json_path)
    print(f"[{stem}] mean scores written to {json_path}")
    for label, md in scores["mean"].items():
        print(f"  label {label}: Dice={md.get('Dice')}")


if __name__ == "__main__":
    main()
