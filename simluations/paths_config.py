import os
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

def _resolve(path) -> Path:
    p = Path(path) if not isinstance(path, Path) else path
    if not p.is_absolute():
        p = (_PROJECT_ROOT / p).resolve()
    return p.resolve()

# Базовые каталоги (переопределяются через env: DATASET_DIR, RESULTS_DIR, SETUP_DIR)
DATASET_DIR = _resolve(os.environ.get("DATASET_DIR", str(_PROJECT_ROOT / "dataset")))
RESULTS_DIR = _resolve(os.environ.get("RESULTS_DIR", str(_PROJECT_ROOT / "results")))
SETUP_DIR = _resolve(os.environ.get("SETUP_DIR", str(_PROJECT_ROOT / "setup")))

# Производные пути
MEAN_DISTANCE_TIME_PATH = DATASET_DIR / "mean_distance_time.csv"
DAILY_TRIPS_DIR = DATASET_DIR / "daily_trips"
