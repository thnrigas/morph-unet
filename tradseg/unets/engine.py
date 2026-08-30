"""Train/val loaders + one-epoch engine (faithful to mc_common_spleen.build_plain_loaders_spleen
and run_epoch). Dice(softmax) + CE(logits), bf16 autocast on CUDA. TRAIN uses the
foreground-filtered spleen loader; VAL/TEST score every slice.
"""

import contextlib
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from data import NumpyDataSet, NumpyDataSetSpleen


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loaders(splits_file, preprocessed_dir, fold, patch_size=256, batch_size=8,
                  num_workers=4, fg_margin=3, train_frac=1.0, seed=42):
    """train_frac < 1.0 keeps a deterministic random subset of the TRAIN cases -- the
    'weakened baseline' (low-data regime) where a traditional refiner is most likely to help."""
    with open(splits_file, "rb") as f:
        splits = pickle.load(f)
    tr, vl, ts = splits[fold]["train"], splits[fold]["val"], splits[fold]["test"]
    if train_frac < 1.0:
        import random as _random
        rng = _random.Random(seed)
        tr = sorted(rng.sample(list(tr), max(1, int(round(len(tr) * train_frac)))))
    preprocessed_dir = str(preprocessed_dir)
    common = dict(target_size=patch_size, batch_size=batch_size, input_slice=(0,), label_slice=1)
    # workers ONLY for train (elastic aug); val/test single-process to avoid a 2nd forked pool next to CUDA
    train = NumpyDataSetSpleen(preprocessed_dir, keys=tr, foreground_only=True, margin=fg_margin,
                               num_processes=num_workers, **common)
    val = NumpyDataSet(preprocessed_dir, keys=vl, mode="val", do_reshuffle=False, num_processes=0, **common)
    test = NumpyDataSet(preprocessed_dir, keys=ts, mode="test", do_reshuffle=False, num_processes=0, **common)
    return train, val, test, 1


def run_epoch(model, loader, device, dice_loss, ce_loss, optimizer=None):
    """One Dice+CE epoch (bf16 autocast on CUDA). optimizer=None -> eval pass. Returns mean loss."""
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    losses = []
    amp = device.type == "cuda"
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for batch in loader:
            data = batch["data"][0].float().to(device, non_blocking=True)   # [b, c, H, W]
            target = batch["seg"][0].long().to(device, non_blocking=True)   # [b, 1, H, W]
            if train_mode:
                optimizer.zero_grad()
            with (torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()):
                pred = model(data)
                pred_softmax = F.softmax(pred, dim=1)
                loss = dice_loss(pred_softmax, target.squeeze()) + ce_loss(pred, target.squeeze())
            if train_mode:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")
