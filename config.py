import os
from pathlib import Path

# file lives at project root
PROJECT_ROOT = Path(__file__).resolve().parent

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
        PROJECT_ROOT / "data" / "Task04_Hippocampus",
    ],
    default=PROJECT_ROOT / "data" / "Task04_Hippocampus",
)

# set manually
# os.environ["DATA_DIR"] = PROJECT_ROOT / "data" / "Task04_Hippocampus"

IMAGES_DIR = DATA_DIR / "imagesTr"
LABELS_DIR = DATA_DIR / "labelsTr"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
SPLITS_FILE = DATA_DIR / "splits.pkl"
