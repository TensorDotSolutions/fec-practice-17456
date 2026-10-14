"""05_train_challenge2.py -- the zero-shot (MAST) predictor, fitted on DIII-D only.

    python code/run/05_train_challenge2.py                # all 7041 training shots
    python code/run/05_train_challenge2.py --n-shots 200

Uses only the machine-agnostic feature set (Ip + Thomson pressure-proxy stats, 7
features) so it can transfer to MAST, where there is no training data at all.  The
notebook's cell 20 left q95/betaN as zeros; this adds a RidgeCV scalar head on the
same shared features, which is worth ~0.02 of the composite score when it works.

Shape/amplitude split: the PCA is fit on each frame's psi *normalized by its own RMS*,
not raw psi -- so the PCA basis models machine-agnostic flux-map SHAPE only, and a
second RidgeCV head predicts the (log) amplitude scale separately from the same shared
features. Fitting PCA+Ridge on raw-scale psi directly (the original version of this
script) forced the shape basis to also absorb DIII-D's absolute psi magnitude, which
transfers poorly to MAST (different machine size/field -> different ψ scale) and was
part of why the first MAST submission scored catastrophically. Splitting the two lets
the shape fit transfer on its own merits even when the amplitude fit is still weak.

Writes ``code/artifacts/models/ch2.joblib`` =
    {scaler, pca, model, amp_model, scalar, meta}.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fec_common import features as F  # noqa: E402
from fec_common.config import MODELS_DIR, ensure_dirs, ensure_starter, log, rss_gb  # noqa: E402

ensure_starter()

from experiments import TargetPCA  # noqa: E402

COLUMNS = [
    "efit_times", "efit_psirz", "efit_q95", "efit_beta_n",
    "magnetics_time", "magnetics_plasma_current", "magnetics_plasma_current_times",
    "thomson_core_times", "thomson_core_Te", "thomson_core_ne",
    "thomson_edge_times", "thomson_edge_Te", "thomson_edge_ne",
]


_SCALE_EPS = 1e-6


def _rms_scale(psi: np.ndarray) -> np.ndarray:
    """Per-frame RMS amplitude, floored so normalization never divides by ~0."""
    return np.maximum(np.sqrt(np.mean(psi.reshape(len(psi), -1) ** 2, axis=1)), _SCALE_EPS)


def fit_pca_subsample(files, pca_components: int, pca_frames: int) -> TargetPCA:
    subsample = []
    for path in files:
        row = F.read_parquet_row(path, columns=["efit_psirz"])
        psi = F._as_psirz_stack(row["efit_psirz"])
        scale = _rms_scale(psi)
        subsample.append(psi / scale[:, None, None])  # shape-only, amplitude normalized out
        if sum(len(p) for p in subsample) >= pca_frames:
            break
        del row
        gc.collect()
    Y = np.concatenate(subsample, axis=0)
    n_components = min(pca_components, len(Y), 65 * 65)
    pca = TargetPCA(n_components=n_components)
    pca.fit(Y)
    log(f"  ch2 PCA on {len(Y)} normalized-shape frames -> {n_components} comps, "
        f"{float(np.cumsum(pca.explained_variance_ratio)[-1]) * 100:.1f}% var")
    return pca


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-shots", type=int, default=0, help="0 = all DIII-D training shots")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pca-components", type=int, default=30)
    ap.add_argument("--pca-frames", type=int, default=50_000)
    ap.add_argument("--alpha-pca", type=float, default=3.0,
                    help="RidgeCV alpha grid floor (logspace -3..3, as in the notebook)")
    args = ap.parse_args()

    ensure_dirs()
    log("=== 05_train_challenge2 ===")
    files = F.train_files(n_shots=args.n_shots, seed=args.seed)
    log(f"  {len(files)} DIII-D shots (shared-feature model for MAST)")

    pca = fit_pca_subsample(files, args.pca_components, args.pca_frames)

    X_parts, Ypca_parts, logscale_parts, scal_parts = [], [], [], []
    t0 = time.time()
    for i, path in enumerate(files):
        row = F.read_parquet_row(path, columns=COLUMNS)
        feat = F.synthetic_features_for_shot(row, "DIII-D")
        psi = F._as_psirz_stack(row["efit_psirz"])
        n = min(len(feat), len(psi))
        scale = _rms_scale(psi[:n])
        X_parts.append(feat[:n].astype(np.float32))
        Ypca_parts.append(pca.transform(psi[:n] / scale[:, None, None]))
        logscale_parts.append(np.log(scale).astype(np.float32))
        scal_parts.append(
            np.column_stack([
                np.asarray(row["efit_q95"], dtype=np.float32)[:n],
                np.asarray(row["efit_beta_n"], dtype=np.float32)[:n],
            ])
        )
        del row, feat, psi
        if (i + 1) % 200 == 0 or i + 1 == len(files):
            log(f"  features {i + 1}/{len(files)}  peak RSS {rss_gb():.2f} GiB  "
                f"{time.time() - t0:.0f}s")
        if (i + 1) % 500 == 0:
            gc.collect()

    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    X = np.concatenate(X_parts); Ypca = np.concatenate(Ypca_parts)
    logscale = np.concatenate(logscale_parts); S = np.concatenate(scal_parts)
    del X_parts, Ypca_parts, logscale_parts, scal_parts
    gc.collect()
    log(f"  {X.shape} frames x {X.shape[1]} shared features")

    scaler = StandardScaler().fit(X)
    Xs = np.asarray(scaler.transform(X), dtype=np.float32)
    model = RidgeCV(alphas=np.logspace(-3, 3, 10))
    model.fit(Xs, Ypca)

    amp_model = RidgeCV(alphas=np.logspace(-3, 3, 10))
    amp_model.fit(Xs, logscale)
    log(f"  ch2 amplitude head fit on {len(logscale)} frames "
        f"(log-scale R2 {amp_model.score(Xs, logscale):.3f})")

    scalar = RidgeCV(alphas=np.logspace(-3, 3, 10))
    ok = np.isfinite(S).all(axis=1)
    if ok.sum() >= 10:
        scalar.fit(Xs[ok], S[ok])
        log(f"  ch2 scalar head fit on {int(ok.sum())} frames "
            f"(q95/betaN val-proxy R2 {scalar.score(Xs[ok], S[ok]):.3f})")
    else:
        scalar = None
        log("  ch2 scalar head skipped (too few finite q95/betaN frames)")

    bundle = {
        "scaler": scaler,
        "pca": pca,
        "model": model,
        "amp_model": amp_model,
        "scalar": scalar,
        "meta": {
            "n_shots": len(files),
            "n_frames": int(len(X)),
            "n_features": int(X.shape[1]),
            "pca_components": int(pca.n_components),
            "seed": args.seed,
            "flux_r2": float(model.score(Xs, Ypca)),
            "amp_r2": float(amp_model.score(Xs, logscale)),
        },
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODELS_DIR / "ch2.joblib")
    log(f"  saved {MODELS_DIR / 'ch2.joblib'}  flux R2={bundle['meta']['flux_r2']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
