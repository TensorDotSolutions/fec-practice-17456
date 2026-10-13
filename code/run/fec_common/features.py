"""Feature engineering for the FEC pipeline.

Ported from notebook cells 6/16/20, plus the **Thomson repair**:

``experiments.load_shot_from_hf_row`` converts Thomson with
``np.asarray(row["thomson_core_Te"], dtype=np.float32)``.  That works for HuggingFace
rows (lists of lists) but raises ``ValueError: setting an array element with a
sequence`` on the local parquet rows (pandas gives an object array of per-timestep
1-D arrays), and the ``except Exception: pass`` around it silently drops BOTH Thomson
systems -- the trained models then see only the 21 magnetics channels.  The repair
below re-attaches the profiles using a ragged-safe conversion, which is what the
notebook's own Challenge-2 cell already does correctly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .config import EFIT_GRID, TRAIN_CONFIG, DATA_ROOT, ensure_starter

ensure_starter()

from experiments import (  # noqa: E402  (needs sys.path set up by ensure_starter)
    D3D_MAGNETICS_SIGNALS,
    EFIT_SCALAR_LABELS,
    TargetPCA,
    _as_psirz_stack,
    interpolate_magnetics_to_efit,
    interpolate_thomson_to_efit,
    load_shot_from_hf_row,
)
from data_fixes import align_mast_thomson_core, fix_d3d_ip_times  # noqa: E402

N_MAGNETICS = len(D3D_MAGNETICS_SIGNALS)  # 21
N_SCALARS = len(EFIT_SCALAR_LABELS)  # 6
THOMSON_SYSTEMS = ("thomson_core", "thomson_edge")


# ---------------------------------------------------------------------------
# Row loading
# ---------------------------------------------------------------------------

def ensure_efit_psirz(row: dict) -> dict:
    """Public-test rows ship WITHOUT efit_psirz (that is what we must predict).

    ``load_shot_from_hf_row`` reads it unconditionally, so inject a zero placeholder
    first.  Nothing downstream uses the placeholder for prediction (only the training
    path ever reads back shot["efit_psirz"]).
    """
    if "efit_psirz" not in row:
        try:
            t = len(np.asarray(row["efit_times"]))
            row["efit_psirz"] = np.zeros((t, EFIT_GRID, EFIT_GRID), dtype=np.float32)
        except Exception:
            row["efit_psirz"] = np.zeros((1, EFIT_GRID, EFIT_GRID), dtype=np.float32)
    return row


def read_parquet_row(path, columns=None, retries: int = 4, delay: float = 2.0) -> dict:
    """Robustly read the single row of a shot parquet with a few retries.

    Pass ``columns`` to project down to just the fields a step needs -- a full row is
    ~79 MB (21 magnetics traces dominate), while a Thomson-only projection is ~1 MB.
    """
    last = None
    for attempt in range(retries):
        try:
            df = pd.read_parquet(path, columns=columns)
            if len(df) != 1:
                raise ValueError(f"{path}: expected exactly one row, got {len(df)}")
            return df.iloc[0]
        except Exception as exc:  # transient local-IO / pyarrow hiccups
            last = exc
            import time as _time

            _time.sleep(delay * (attempt + 1))
    raise RuntimeError(f"Failed to read {path} after {retries} attempts: {last}")


def _as_channel_matrix(values, n_channels: int | None = None) -> np.ndarray:
    """Normalise one Thomson measure to a (T, C) float32 matrix.

    Handles the three layouts we see in practice:
      * (T, C) ndarray  (HuggingFace rows / healthy conversions)  -> unchanged
      * object array / list of per-timestep 1-D arrays            -> stacked, ragged
        rows padded with NaN to the widest row (or to ``n_channels``)
      * a bare 1-D array (single channel)                          -> (T, 1)
    """
    arr = np.asarray(values, dtype=object) if not isinstance(values, np.ndarray) else values
    if arr.dtype != object:
        a = np.asarray(arr, dtype=np.float32)
        return a[:, None] if a.ndim == 1 else a

    widths = [np.atleast_1d(r).size for r in arr if r is not None]
    width = n_channels or (max(widths) if widths else 1)
    width = max(int(width), 1)
    out = np.full((len(arr), width), np.nan, dtype=np.float32)
    for i, r in enumerate(arr):
        if r is None:
            continue
        row = np.asarray(r, dtype=np.float32).ravel()
        n = min(row.size, width)
        if n:
            out[i, :n] = row[:n]
    return out


def repair_thomson(shot: dict, row: dict) -> dict:
    """Attach thomson_core / thomson_edge to ``shot`` when the starter loader dropped them."""
    for system_key in THOMSON_SYSTEMS:
        if system_key in shot:
            continue
        times_col, te_col, ne_col = (
            f"{system_key}_times",
            f"{system_key}_Te",
            f"{system_key}_ne",
        )
        if times_col not in row or te_col not in row:
            continue
        try:
            Te = _as_channel_matrix(row[te_col])
            ne = (
                _as_channel_matrix(row[ne_col], n_channels=Te.shape[1])
                if ne_col in row
                else None
            )
            shot[system_key] = {
                "times": np.asarray(row[times_col], dtype=np.float64),
                "Te": Te,
                "ne": ne,
            }
        except Exception:
            pass  # a shot with an unreadable Thomson block simply loses that feature group
    return shot


def load_shot(row: dict) -> dict:
    """One parquet/HF row -> the internal shot dict, with the Thomson repair applied."""
    row = ensure_efit_psirz(row)
    shot = load_shot_from_hf_row(row)
    return repair_thomson(shot, row)


# ---------------------------------------------------------------------------
# Frame building
# ---------------------------------------------------------------------------

def build_frame(shot: dict, include_thomson: bool = True):
    """Per-shot features/targets.

    Returns ``(X, Y, S)`` where X is (T, F) float32, Y is (T, 65, 65) float32 and
    S is (T, 6) float32 with NaN where a scalar label is undefined.  Follows the
    starter's row-count rule (``n_times = min(features, psirz)``) and keeps scalar
    NaNs (they are masked per-scalar at fit time, never by dropping the row).
    """
    features = interpolate_magnetics_to_efit(shot)  # (T, 21)
    if include_thomson:
        thomson = interpolate_thomson_to_efit(shot)
        if thomson is not None:
            features = np.hstack([features, thomson])

    psirz = shot["efit_psirz"]
    n_times = min(len(features), len(psirz))
    X = np.asarray(features[:n_times], dtype=np.float32)
    Y = np.asarray(psirz[:n_times], dtype=np.float32)

    S = np.full((n_times, N_SCALARS), np.nan, dtype=np.float32)
    shot_scalars = shot.get("scalars", {})
    for j, name in enumerate(EFIT_SCALAR_LABELS):
        arr = shot_scalars.get(name)
        if arr is not None:
            m = min(n_times, len(arr))
            S[:m, j] = arr[:m]
    return X, Y, S


def valid_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """The starter's drop rule: rows with non-finite inputs or flux maps."""
    return np.isfinite(X).all(axis=1) & np.isfinite(Y.reshape(len(Y), -1)).all(axis=1)


def pad_or_truncate(feat: np.ndarray, n_feat: int) -> np.ndarray:
    """Align a feature matrix to the width the scaler/PCA were fit on.

    Thomson channel count is not fixed across shots (different probe configs), so a
    test shot can be wider or narrower than training.  Missing channels become zeros
    (matching what ``interpolate_thomson_to_efit`` itself emits for all-NaN channels).
    """
    if feat.shape[1] == n_feat:
        return feat
    if feat.shape[1] < n_feat:
        pad = np.zeros((feat.shape[0], n_feat - feat.shape[1]), dtype=np.float32)
        return np.hstack([feat, pad])
    return feat[:, :n_feat].copy()


# ---------------------------------------------------------------------------
# Scan helpers (pass A of 01 -- cheap column projection, no full materialisation)
# ---------------------------------------------------------------------------

def natural_width(shot: dict) -> int:
    """Feature width ``build_frame`` would produce for this shot, before pad/truncate."""
    width = N_MAGNETICS
    for system_key in THOMSON_SYSTEMS:
        system = shot.get(system_key)
        if not system:
            continue
        for measure in ("Te", "ne"):
            profiles = system.get(measure)
            if profiles is None:
                continue
            profiles = np.asarray(profiles)
            n_channels = profiles.shape[1] if profiles.ndim == 2 else 1
            width += int(n_channels)
    return width


def scan_shot(path) -> tuple[int, int]:
    """``(n_frames, natural_feature_width)`` for one parquet, reading only what it needs."""
    import pyarrow.parquet as pq

    file = pq.ParquetFile(path)
    names = set(file.schema_arrow.names)
    n_frames = 0
    efit_t = next(
        (c for c in ("efit_times",) if c in names), None
    )
    if efit_t is not None:
        col = file.read(columns=[efit_t]).column(efit_t)[0].as_py()
        n_frames = len(col) if col is not None else 0

    width = N_MAGNETICS
    for system_key in THOMSON_SYSTEMS:
        te_col = f"{system_key}_Te"
        times_col = f"{system_key}_times"
        if te_col not in names or times_col not in names:
            continue
        raw = file.read(columns=[te_col]).column(te_col)[0].as_py()
        if not raw:
            continue
        first = np.atleast_1d(raw[0])
        width += 2 * max(int(first.size), 1)  # Te and ne share the channel layout
    return n_frames, width


def train_files(n_shots: int = 0, seed: int = 42) -> list[Path]:
    """Sorted training parquets; with ``n_shots > 0`` take the starter's random subset."""
    files = sorted((DATA_ROOT / TRAIN_CONFIG).glob("*.parquet"))
    if not files:
        raise SystemExit(
            f"No training parquets under {DATA_ROOT / TRAIN_CONFIG}. Run 00_setup.py / "
            "check the dataset download."
        )
    if 0 < n_shots < len(files):
        rng = np.random.RandomState(seed)
        files = [files[i] for i in sorted(rng.choice(len(files), size=n_shots, replace=False))]
    return files


# ---------------------------------------------------------------------------
# Challenge-2 synthetic (machine-agnostic) features
# ---------------------------------------------------------------------------

def synthetic_features_for_shot(row: dict, source: str) -> np.ndarray:
    """Machine-agnostic features: Ip + Thomson pressure-proxy stats, on the efit_times base."""
    efit_times = np.asarray(row["efit_times"], dtype=np.float64)
    T = len(efit_times)

    if source == "DIII-D":
        ip_times = fix_d3d_ip_times(row)
        ip = np.asarray(row["magnetics_plasma_current"], dtype=np.float64)  # already A
    else:
        ip_times = np.asarray(row["magnetics_time"], dtype=np.float64)
        ip = np.asarray(row["magnetics_plasma_current"], dtype=np.float64) * 1000.0  # kA -> A
        ok = np.isfinite(ip_times) & np.isfinite(ip)
        ip_times, ip = ip_times[ok], ip[ok]
    f_ip = interp1d(ip_times, ip, kind="linear", fill_value="extrapolate", bounds_error=False)
    ip_feat = f_ip(efit_times)

    stats = []
    for system_key in THOMSON_SYSTEMS:
        times_key, te_key, ne_key = (
            f"{system_key}_times",
            f"{system_key}_Te",
            f"{system_key}_ne",
        )
        if times_key not in row or te_key not in row:
            stats.append(np.zeros((T, 3), dtype=np.float32))
            continue
        t_times = np.asarray(row[times_key], dtype=np.float64)
        try:
            if source == "MAST" and system_key == "thomson_core":
                _, Te, ne = align_mast_thomson_core(row)
                Te = np.asarray(Te, dtype=np.float64)
                ne = np.asarray(ne, dtype=np.float64)
            else:
                Te = _as_channel_matrix(row[te_key]).astype(np.float64)
                ne = (
                    _as_channel_matrix(row[ne_key]).astype(np.float64)
                    if ne_key in row
                    else np.full_like(Te, np.nan)
                )
            n = min(len(Te), len(ne))
            pe = Te[:n] * ne[:n]  # electron pressure proxy, channel x timestep
            t_times = t_times[:n]
        except Exception:
            stats.append(np.zeros((T, 3), dtype=np.float32))
            continue

        pe_mean = np.nanmean(pe, axis=1)
        pe_std = np.nanstd(pe, axis=1)
        pe_max = np.nanmax(np.nan_to_num(pe, nan=-np.inf), axis=1)
        sys_feat = []
        for series in (pe_mean, pe_std, pe_max):
            ok = np.isfinite(series)
            if ok.sum() < 2:
                sys_feat.append(np.zeros(T, dtype=np.float32))
                continue
            f = interp1d(
                t_times[ok], series[ok], kind="linear", fill_value=0.0, bounds_error=False
            )
            sys_feat.append(f(efit_times).astype(np.float32))
        stats.append(np.column_stack(sys_feat))

    feat = np.column_stack([ip_feat.astype(np.float32)] + stats)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


SYNTHETIC_N_FEATURES = 1 + 3 * len(THOMSON_SYSTEMS)  # Ip + (mean,std,max) per system
