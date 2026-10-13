"""Paths, constants and small shared helpers for the local FEC pipeline.

Every path is derived from this file's location, so the same package works both
locally (``code/run/fec_common/config.py`` on this machine) and in Colab, where the
Drive-synced repo is imported from ``MyDrive/EXPERIMENTS/FEC_NeurIPS/code/run``.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# code/run/fec_common/config.py -> parents[2] == code/
CODE_DIR = Path(__file__).resolve().parents[2]
RUN_DIR = CODE_DIR / "run"
# Both overridable so the same package works in Colab, where the starter clone lives at
# /content/... and the Drive-synced repo sits somewhere else entirely.
DATA_ROOT = Path(os.environ.get("FEC_DATA_ROOT", CODE_DIR / "fusion-equilibrium-challenge" / "data"))
ARTIFACTS = Path(os.environ.get("FEC_ARTIFACTS", CODE_DIR / "artifacts"))
FEATURES_DIR = ARTIFACTS / "features"
MODELS_DIR = ARTIFACTS / "models"
SCORES_DIR = ARTIFACTS / "scores"
SUBMISSION_DIR = ARTIFACTS / "submission"
STARTER_DIR = Path(os.environ.get("FEC_STARTER_DIR", CODE_DIR / "starter"))
FUSION_SCORING_DIR = STARTER_DIR / "fusion_scoring"

REPO_ID = "Sophelio/fusion-equilibrium-challenge"
PINNED_COMMIT = "a67429165b09eb81c311d44db6ff11743f108b0e"
EFIT_GRID = 65
TRAIN_CONFIG = "diii_d_train"

# (hf config name, hf split name, source label) for the two graded test folds
TEST_CONFIGS = [
    ("diii_d_public_test", "public_test", "DIII-D"),
    ("mast_public_test", "public_test", "MAST"),
]

# Frozen artifact names -- 01 writes them, everything else reads them.
X_NPY = "X.f32.npy"
YPCA_NPY = "ypca.f32.npy"
YSCALAR_NPY = "y_scalar.f32.npy"
FRAME_INDEX_NPZ = "frame_index.npz"
SPLIT_JSON = "split.json"
META_JSON = "meta.json"
PCA_JOBLIB = "pca.joblib"
SCALER_JOBLIB = "scaler.joblib"
MANIFEST_JSON = "manifest.json"


def ensure_starter() -> Path:
    """Verify the pinned starter-kit clone exists and put it (plus fusion_scoring) on sys.path."""
    experiments_py = STARTER_DIR / "experiments.py"
    if not experiments_py.exists():
        raise SystemExit(
            f"Starter kit not found at {STARTER_DIR}\n"
            f"Run:  python {RUN_DIR / '00_setup.py'}   (clones the pinned commit)"
        )
    for p in (str(STARTER_DIR), str(FUSION_SCORING_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return STARTER_DIR


def ensure_dirs() -> None:
    for d in (ARTIFACTS, FEATURES_DIR, MODELS_DIR, SCORES_DIR, SUBMISSION_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    os.replace(tmp, path)


def load_json(path: Path):
    with open(path) as fh:
        return json.load(fh)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def rss_gb() -> float:
    """Current process peak RSS in GiB (Linux ru_maxrss is in KiB)."""
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_env() -> str | None:
    """Load HF_TOKEN from code/.env (or the environment). Returns the token or None."""
    from dotenv import load_dotenv

    load_dotenv(CODE_DIR / ".env")
    return os.getenv("HF_TOKEN") or None
