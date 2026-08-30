"""Load the 2-channel preprocessed npy + k-fold splits (baseline-compatible format).

The npy per case has shape (2, Z, H, W): channel 0 = CT image in [0,1], channel 1 =
integer label. This is exactly what the baseline produces/consumes, so pointing
DATA_DIR at either pipeline's output works. We never import baseline code.
"""

import pickle
from pathlib import Path

import numpy as np

import config


def load_splits(path=None):
    path = Path(path) if path else config.SPLITS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"splits file not found: {path}\n"
            "Generate it with the preprocessing step (create the (2,Z,H,W) npy + splits.pkl)."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def fold_cases(splits, fold, subset="test"):
    """Case names for one fold subset ('train' | 'val' | 'test')."""
    return list(splits[fold][subset])


def case_file(name, preprocessed_dir=None):
    d = Path(preprocessed_dir) if preprocessed_dir else config.PREPROCESSED_DIR
    return d / f"{name}.npy"


def load_case(name, preprocessed_dir=None):
    """Return (image (Z,H,W) float32 in [0,1], label (Z,H,W) int16)."""
    p = case_file(name, preprocessed_dir)
    if not p.exists():
        raise FileNotFoundError(f"preprocessed case not found: {p}")
    arr = np.load(p)
    if arr.ndim != 4 or arr.shape[0] < 2:
        raise ValueError(f"expected (2, Z, H, W) npy, got {arr.shape} for {p}")
    image = np.ascontiguousarray(arr[0], dtype=np.float32)
    label = np.rint(arr[1]).astype(np.int16)
    return image, label


def iter_cases(names, preprocessed_dir=None):
    for name in names:
        image, label = load_case(name, preprocessed_dir)
        yield name, image, label
