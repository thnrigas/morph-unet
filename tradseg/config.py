"""Paths and the preprocessing 'data contract' shared with the baseline.

Everything is overridable via environment variables so the exact same layout works
on the local machine and on the GCP Deep Learning VM. To reuse the U-Net's already
preprocessed volumes verbatim (maximum fairness), point DATA_DIR at its data dir:

    DATA_DIR=.../my_unet-uncertainty/data/Task09_Spleen python -m tradseg.run_tier1 ...
"""

import os
from pathlib import Path

# the directory holding the code modules (== repo root when the code lives flat)
PROJECT_ROOT = Path(__file__).resolve().parent

TASK = os.environ.get("TASK", "Task09_Spleen")


def _env_path(name: str, default: Path) -> Path:
    val = os.environ.get(name)
    return Path(val) if val else default


DATA_DIR = _env_path("DATA_DIR", PROJECT_ROOT / "data" / TASK)
IMAGES_DIR = _env_path("IMAGES_DIR", DATA_DIR / "imagesTr")     # raw NIfTI (for our own preprocessing)
LABELS_DIR = _env_path("LABELS_DIR", DATA_DIR / "labelsTr")
PREPROCESSED_DIR = _env_path("PREPROCESSED_DIR", DATA_DIR / "preprocessed")
SPLITS_FILE = _env_path("SPLITS_FILE", DATA_DIR / "splits.pkl")
RESULTS_DIR = _env_path("RESULTS_DIR", PROJECT_ROOT / "results")

# Foreground label of interest (MSD Task09 spleen == 1).
FOREGROUND_LABEL = int(os.environ.get("FOREGROUND_LABEL", "1"))

# ---- fixed-window 'data contract' (must match the fair-comparison npy) -------
# CT soft-tissue window mapped to [0, 1]: center 40, width 400 -> [-160, 240] HU.
CT_CENTER = float(os.environ.get("CT_CENTER", "40"))
CT_WIDTH = float(os.environ.get("CT_WIDTH", "400"))
PREPROC_SIZE = int(os.environ.get("PREPROC_SIZE", "256"))

# Typical spleen attenuation (HU); used only to seed intensity-band heuristics.
SPLEEN_HU = float(os.environ.get("SPLEEN_HU", "50"))


def spleen_target_intensity() -> float:
    """Spleen HU mapped into the fixed [0,1] window -> ~0.52 for center40/width400."""
    lo = CT_CENTER - CT_WIDTH / 2.0
    return float(min(1.0, max(0.0, (SPLEEN_HU - lo) / CT_WIDTH)))
