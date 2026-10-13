"""Model training and artifact IO for the local pipeline (notebook cells 16-18 glue)."""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np

from .config import (
    ARTIFACTS,
    FEATURES_DIR,
    FRAME_INDEX_NPZ,
    MANIFEST_JSON,
    MODELS_DIR,
    PCA_JOBLIB,
    SCALER_JOBLIB,
    SPLIT_JSON,
    META_JSON,
    X_NPY,
    YPCA_NPY,
    YSCALAR_NPY,
    ensure_starter,
    load_json,
    log,
    save_json,
)

ensure_starter()

from experiments import (  # noqa: E402
    EFIT_SCALAR_LABELS,
    ScalarTargetModels,
    SklearnModel,
    TargetPCA,
    compute_metrics,
    get_sklearn_models,
)

SCALAR_NAMES = list(EFIT_SCALAR_LABELS)


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def open_memmap(path: Path, mode: str = "r") -> np.memmap:
    # np.lib.format, not np.memmap: these are .npy files, and np.memmap would treat the
    # 128-byte .npy header as data.
    return np.lib.format.open_memmap(path, dtype=np.float32, mode=mode)


def load_cache(features_dir: Path = FEATURES_DIR) -> dict:
    """Open the 01 cache: X/ypca/y_scalar memmaps + per-shot index + split + meta."""
    features_dir = Path(features_dir)
    index = np.load(features_dir / FRAME_INDEX_NPZ, allow_pickle=True)
    meta = load_json(features_dir / META_JSON)
    split = load_json(features_dir / SPLIT_JSON)
    n_frames = int(meta["n_frames"])
    n_feat = int(meta["n_feat"])
    # The .npy headers keep the planned (pre-drop) length; slice to the real count.
    return {
        "dir": features_dir,
        "X": open_memmap(features_dir / X_NPY)[:n_frames],
        "Ypca": open_memmap(features_dir / YPCA_NPY)[:n_frames],
        "S": open_memmap(features_dir / YSCALAR_NPY)[:n_frames],
        "shot_id": index["shot_id"],  # per-frame shot index
        "raw_frame_idx": index["raw_frame_idx"],  # per-frame index within the shot's raw psi
        "frame_starts": index["frame_starts"],  # per-shot first frame offset
        "files": [str(f) for f in index["file_names"]],  # per-shot parquet path
        "n_frames_per_shot": index["n_frames_per_shot"],
        "n_frames": n_frames,
        "n_feat": n_feat,
        "meta": meta,
        "split": split,
    }


def shot_rows(cache: dict, shot_ids) -> np.ndarray:
    """Frame indices belonging to any of ``shot_ids`` (from the per-frame shot index)."""
    shot_id = np.asarray(cache["shot_id"])
    if np.isscalar(shot_ids):
        shot_ids = [shot_ids]
    mask = np.isin(shot_id, np.asarray(shot_ids))
    return np.flatnonzero(mask)


def load_split_arrays(cache: dict, which: str, scale: bool = True):
    """``(X, Ypca, S, frames)`` for one split ('train'|'val'|'test'|'all')."""
    rows = shot_rows(cache, cache["split"][which]) if which != "all" else slice(None)
    X = np.asarray(cache["X"][rows], dtype=np.float32)
    Ypca = np.asarray(cache["Ypca"][rows], dtype=np.float32)
    S = np.asarray(cache["S"][rows], dtype=np.float32)
    frames = np.arange(len(X)) if which == "all" else rows
    if scale:
        X = np.asarray(cache["scaler"].transform(X), dtype=np.float32)
    return X, Ypca, S, frames


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_scaler(cache: dict) -> "StandardScaler":
    """Fit the StandardScaler over the whole X memmap in batches and persist it."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    n = cache["X"].shape[0]
    batch = 200_000
    for start in range(0, n, batch):
        end = min(start + batch, n)
        scaler.partial_fit(np.asarray(cache["X"][start:end], dtype=np.float32))
        log(f"  scaler partial_fit {end}/{n}")
    joblib.dump(scaler, cache["dir"] / SCALER_JOBLIB)
    return scaler


def load_scaler(cache: dict):
    return joblib.load(cache["dir"] / SCALER_JOBLIB)


def save_pca(pca: TargetPCA, cache: dict) -> None:
    joblib.dump(pca, cache["dir"] / PCA_JOBLIB)


def load_pca(cache: dict) -> TargetPCA:
    return joblib.load(cache["dir"] / PCA_JOBLIB)


def available_flux_models(names=None) -> dict[str, SklearnModel]:
    """The starter's four sklearn candidates, optionally filtered by name substring."""
    models = {m.name: m for m in get_sklearn_models()}
    if names:
        keep = {}
        for want in names:
            matches = [m for k, m in models.items() if want.lower() in k.lower()]
            if not matches:
                raise SystemExit(
                    f"Unknown model '{want}'. Choose from: {', '.join(models)}"
                )
            keep.update({m.name: m for m in matches})
        return keep
    return models


def train_flux_model(cache: dict, model_name: str, score: bool = False) -> dict:
    """Fit one flux model on the train split, evaluate on val, persist under models/."""
    t0 = time.time()
    X_tr, Ypca_tr, _, _ = load_split_arrays(cache, "train")
    X_va, Ypca_va, _, frames_va = load_split_arrays(cache, "val")
    log(f"  {model_name}: train {X_tr.shape}  val {X_va.shape}  ({time.time()-t0:.0f}s load)")

    model = available_flux_models([model_name])[model_name]
    if hasattr(model, "verbose"):
        model.verbose = 1  # print progress every 10 trees
    t0 = time.time()
    model.fit(X_tr, Ypca_tr)
    fit_s = time.time() - t0

    pred = model.predict(X_va)
    val_r2 = float(_r2_flat(Ypca_va, pred))
    entry = {
        "name": model.name,
        "val_pca_r2": val_r2,
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "fit_seconds": round(fit_s, 1),
    }
    log(f"  {model.name}: val PCA R2={val_r2:.4f}  fit {fit_s:.0f}s")

    if score:
        entry["val_score"] = score_split(cache, model, frames_va, n_shots=None)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"flux_{_slug(model.name)}.joblib"
    joblib.dump(model, path)
    # Store just the filename, not an absolute path -- this manifest is read from both the
    # local machine and Drive-synced Colab, where the absolute prefix differs (see predict.py's
    # _load_flux_model, which re-joins it against the local MODELS_DIR).
    entry["path"] = path.name
    return entry


def _r2_flat(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics import r2_score

    return r2_score(np.asarray(a).reshape(len(a), -1), np.asarray(b).reshape(len(b), -1))


def score_split(cache: dict, model, frames, n_shots=None) -> dict:
    """Composite Challenge-1 score on a set of cached frame indices.

    Ground-truth psi is re-read from the per-shot parquets (column-projected to
    ``efit_psirz`` only), so nothing big has to be cached.
    """
    from .scoring import composite_from_predictions

    pca = load_pca(cache)
    scaler = cache.get("scaler") or load_scaler(cache)
    scalar_model = joblib.load(MODELS_DIR / "scalars.joblib")

    frames = np.asarray(frames)
    shot_id = np.asarray(cache["shot_id"])[frames]
    Y_true, Y_pred, S_true, S_pred = [], [], [], []

    for sid in np.unique(shot_id):
        idx = frames[shot_id == sid]
        psi = _read_shot_psi(cache, int(sid))
        raw = np.asarray(cache["raw_frame_idx"])[idx]
        Y_true.append(psi[raw])
        S_true.append(np.asarray(cache["S"][idx]))
        for start in range(0, len(idx), 20_000):  # keep the inverse-PCA peak small
            chunk = idx[start : start + 20_000]
            X = np.asarray(scaler.transform(np.asarray(cache["X"][chunk])), dtype=np.float32)
            Y_pred.append(pca.inverse_transform(model.predict(X)))
            S_pred.append(scalar_model.predict(X))

    return composite_from_predictions(
        np.concatenate(Y_true), np.concatenate(Y_pred),
        np.concatenate(S_true), np.concatenate(S_pred),
    )


def _read_shot_psi(cache: dict, shot_id: int) -> np.ndarray:
    """Ground-truth (T, 65, 65) psi for one cached shot, read straight from parquet."""
    from .features import _as_psirz_stack, read_parquet_row

    row = read_parquet_row(cache["files"][shot_id])
    return _as_psirz_stack(row["efit_psirz"])


def train_scalar_models(cache: dict) -> dict:
    """Fit the per-scalar RidgeCV heads on the train split; persist models/scalars.joblib."""
    X_tr, _, S_tr, _ = load_split_arrays(cache, "train")
    X_va, _, S_va, _ = load_split_arrays(cache, "val")

    scalar = ScalarTargetModels(SCALAR_NAMES)
    scalar.fit(X_tr, S_tr)
    log(f"  scalar heads fit on {len(X_tr)} train frames")
    pred = scalar.predict(X_va)

    per_scalar = {}
    from sklearn.metrics import r2_score

    for j, name in enumerate(SCALAR_NAMES):
        mask = np.isfinite(S_va[:, j]) & np.isfinite(pred[:, j])
        per_scalar[name] = (
            float(r2_score(S_va[mask, j], pred[mask, j])) if mask.sum() >= 2 else None
        )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scalar, MODELS_DIR / "scalars.joblib")
    entry = {"per_scalar_val_r2": per_scalar, "path": str(MODELS_DIR / "scalars.joblib")}
    log(f"  scalar heads val R2: {per_scalar}")
    return entry


# ---------------------------------------------------------------------------
# Manifest / best-model selection
# ---------------------------------------------------------------------------

def update_manifest(**kwargs) -> dict:
    """Merge kwargs into manifest.json. Dict-valued kwargs (flux_models, scalars, ...) are
    merged ONE LEVEL DEEP into any existing dict of the same key, not replaced wholesale --
    a caller adding one new flux model entry must not silently wipe out the others that were
    already there (this happened once: registering "Conv Decoder (full)" alone deleted Linear
    Regression/Ridge/Random Forest because plain dict.update() replaces the whole sub-dict)."""
    path = ARTIFACTS / MANIFEST_JSON
    manifest = load_json(path) if path.exists() else {}
    for key, value in kwargs.items():
        if isinstance(value, dict) and isinstance(manifest.get(key), dict):
            manifest[key].update(value)
        else:
            manifest[key] = value
    manifest["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(path, manifest)
    return manifest


def best_flux_model(manifest: dict | None = None) -> str:
    """Pick the flux model with the best validation PCA R2 (or 'S' if scored)."""
    manifest = manifest or load_json(ARTIFACTS / MANIFEST_JSON)
    flux = manifest.get("flux_models")
    if not flux:
        raise SystemExit("manifest.json has no flux_models entry -- run 02_train_flux.py")
    key = "val_score" if any("val_score" in e for e in flux.values()) else "val_pca_r2"
    ranked = sorted(flux.values(), key=lambda e: e.get(key, -np.inf), reverse=True)
    return ranked[0]["name"]


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")
