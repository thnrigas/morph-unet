import os
import json
from pathlib import Path

# file lives at project root
PROJECT_ROOT = Path(__file__).resolve().parent

# which MSD task to use
# Task01_BrainTumour Task02_Heart    Task03_Liver         Task04_Hippocampus Task05_Prostate
# Task06_Lung        Task07_Pancreas Task08_HepaticVessel Task09_Spleen      Task10_Colon
TASK = os.environ.get("TASK", "Task08_HepaticVessel")

CHANNEL = int(os.environ.get("CHANNEL", "0"))

# per task default training hyperparameters
_HP_DEFAULT = dict(epochs=400, patience=30, batch_size=16, patch_size=128, lr=2e-4, iters_per_epoch=250, fg_fraction=0.33, val_cases=15, val_batch=12, grad_clip=5.0)

# per task default custom hyperparameters
_HP = {
    "Task01_BrainTumour": dict(epochs=400, patience=40, iters_per_epoch=250, batch_size=16, patch_size=128, fg_fraction=0.40, val_cases=25, grad_clip=1.0, lr=1e-4),
    "Task02_Heart": dict(epochs=300, patience=30, iters_per_epoch=250, batch_size=16, patch_size=128, fg_fraction=0.33),
    "Task03_Liver": dict(epochs=400, patience=30, iters_per_epoch=250, batch_size=16, patch_size=128, fg_fraction=0.40),
    "Task04_Hippocampus": dict(epochs=150, patience=15, iters_per_epoch=0, batch_size=32, patch_size=64, fg_fraction=0.33),
    "Task05_Prostate": dict(epochs=300, patience=30, iters_per_epoch=250, batch_size=16, patch_size=128, fg_fraction=0.33),
    "Task06_Lung": dict(epochs=400, patience=30, iters_per_epoch=250, batch_size=16, patch_size=128, fg_fraction=0.66, grad_clip=2.0),
    "Task07_Pancreas": dict(epochs=400, patience=30, iters_per_epoch=250, batch_size=16, patch_size=128, fg_fraction=0.50, grad_clip=2.0),
    "Task08_HepaticVessel": dict(epochs=400, patience=30, iters_per_epoch=250, batch_size=24, patch_size=128, fg_fraction=0.33, grad_clip=5.0),
    "Task09_Spleen": dict(epochs=300, patience=30, iters_per_epoch=250, batch_size=16, patch_size=128, fg_fraction=0.33),
    "Task10_Colon": dict(epochs=400, patience=30, iters_per_epoch=250, batch_size=16, patch_size=128, fg_fraction=0.66, grad_clip=2.0),
}
HP = {**_HP_DEFAULT, **_HP.get(TASK, {})}

def resolve_dir(env_var: str, candidates: list[Path], default: Path) -> Path:
    # set manually to override
    val = os.environ.get(env_var)
    if val:
        return Path(val)
    # first existing candidate
    for path in candidates:
        if path.exists():
            return path
    # default fallback
    return default

DATA_DIR = resolve_dir(
    env_var="DATA_DIR",
    candidates=[
        PROJECT_ROOT / "data" / TASK,
    ],
    default=PROJECT_ROOT / "data" / TASK,
)

IMAGES_DIR = DATA_DIR / "imagesTr"
LABELS_DIR = DATA_DIR / "labelsTr"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
SPLITS_FILE = DATA_DIR / "splits.pkl"

def _load_meta(data_dir: Path):
    meta = data_dir / "dataset.json"
    if meta.exists():
        d = json.loads(meta.read_text())
        modality = d.get("modality", {"0": "MRI"})
        labels = {int(k): v for k, v in d.get("labels", {}).items() if int(k) != 0}
        return modality, labels
    return {"0": "MRI"}, {}

MODALITY, LABELS = _load_meta(DATA_DIR)
NUM_CLASSES = (max(LABELS) + 1) if LABELS else 3
