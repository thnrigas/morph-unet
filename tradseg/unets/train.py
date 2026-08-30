"""Train a (possibly weakened) MC-Dropout U-Net on one fold. Faithful recipe:
SoftDiceLoss(batch_dice) + CE, Adam(2e-4), ReduceLROnPlateau, early stopping, best ckpt.

SKIP-IF-EXISTS: if <out-dir>/<tag>_f<fold>_best.pth already exists it is NOT retrained
(pass --force to override). No weights exist yet; after this runs they will, and
predictor/infer/test load them.

Weakened baseline (Idea 2): use --epochs (fewer) and/or --frac (subsample TRAIN cases).

  python3 unets/train.py --tag mcdropout --fold 0 \
      --preprocessed-dir data/Task09_Spleen/preprocessed_A --out-dir results/unets
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))          # tradseg repo root (config)

import argparse

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from arch import MCDropoutUNet
from losses import SoftDiceLoss
from engine import build_loaders, run_epoch, pick_device
from data import set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="mcdropout")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--fg-margin", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--dropout-p", type=float, default=0.4)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--frac", type=float, default=1.0, help="fraction of TRAIN cases (weakened baseline)")
    p.add_argument("--preprocessed-dir", default=None, help="default: config.PREPROCESSED_DIR")
    p.add_argument("--splits", default=None, help="default: config.SPLITS_FILE")
    p.add_argument("--out-dir", default=None, help="default: config.RESULTS_DIR/unets")
    p.add_argument("--force", action="store_true", help="retrain even if the checkpoint exists")
    args = p.parse_args()

    if args.batch_size < 2:
        raise SystemExit("use --batch-size >= 2 (run_epoch squeezes the batch axis)")

    prep = str(args.preprocessed_dir or config.PREPROCESSED_DIR)
    splits = str(args.splits or config.SPLITS_FILE)
    out_dir = str(args.out_dir or (config.RESULTS_DIR / "unets"))
    os.makedirs(out_dir, exist_ok=True)

    stem = f"{args.tag}_f{args.fold}"
    best_path = os.path.join(out_dir, f"{stem}_best.pth")
    last_path = os.path.join(out_dir, f"{stem}_last.pth")
    if os.path.exists(best_path) and not args.force:
        print(f"[{stem}] checkpoint exists -> SKIP training ({best_path}). Use --force to retrain.")
        return

    set_seed(args.seed)
    device = pick_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader, _, in_channels = build_loaders(
        splits, prep, args.fold, patch_size=args.patch_size, batch_size=args.batch_size,
        num_workers=args.num_workers, fg_margin=args.fg_margin, train_frac=args.frac, seed=args.seed)
    model = MCDropoutUNet(num_classes=args.num_classes, in_channels=in_channels,
                          dropout_p=args.dropout_p).to(device)
    print(f"[{stem}] device={device} dropout={args.dropout_p} classes={args.num_classes} "
          f"patch={args.patch_size} bs={args.batch_size} frac={args.frac} epochs={args.epochs}")

    dice_loss = SoftDiceLoss(batch_dice=True)
    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=6)

    best_val, since_improved = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, dice_loss, ce_loss, optimizer)
        vl = run_epoch(model, val_loader, device, dice_loss, ce_loss, None)
        scheduler.step(vl)
        print(f"epoch {epoch:3d}/{args.epochs}  train={tr:.4f}  val={vl:.4f}", flush=True)
        if vl < best_val:
            best_val, since_improved = vl, 0
            torch.save(model.state_dict(), best_path)
        else:
            since_improved += 1
        torch.save(model.state_dict(), last_path)
        if args.patience and since_improved >= args.patience:
            print(f"early stop @ epoch {epoch} (best val={best_val:.4f})")
            break

    print(f"[{stem}] best -> {best_path}")


if __name__ == "__main__":
    main()
