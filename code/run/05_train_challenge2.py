"""05_train_challenge2.py -- the zero-shot (MAST) predictor, fitted on DIII-D only.

    python code/run/05_train_challenge2.py                # all 7041 training shots
    python code/run/05_train_challenge2.py --n-shots 200

Uses the machine-agnostic feature set (Ip + Thomson pressure-proxy stats, 7 features) so
it can transfer to MAST, where there is no training data at all. The notebook's cell 20
left q95/betaN as zeros; this adds a RidgeCV scalar head on the same shared features,
which is worth ~0.02 of the composite score when it works.

Vacuum-field decomposition: before any statistical fitting, each frame's target psi has
a coil-only *vacuum* flux map subtracted off -- psi_vacuum, computed by
``fec_common.vacuum_field`` from a Green's-function/Biot-Savart calculation on the
shot's own coil geometry and currents (zero free parameters besides one global scale,
fit by least squares on DIII-D only, then applied identically to MAST -- the same
physics formula evaluated on MAST's own geometry, no MAST labels involved). This is the
"forward model" the dataset docs describe reaching R^2 ~ 0.94 on DIII-D by itself. The
statistical pipeline below (shape PCA + amplitude + scalar heads) then only has to learn
the *residual* -- the plasma's own contribution -- which is a much smaller, more
plasma-physics-driven signal than raw psi and should transfer zero-shot better than
fitting the whole flux map from 7 scalar stats.

Shape/amplitude split: the PCA is fit on each residual frame *normalized by its own
RMS*, not the raw residual -- so the PCA basis models machine-agnostic residual SHAPE
only, and a second RidgeCV head predicts the (log) amplitude scale separately from the
same shared features. Fitting PCA+Ridge on raw-scale values directly forces the shape
basis to also absorb DIII-D's absolute magnitude, which transfers poorly to MAST
(different machine size/field) -- this was part of why the first MAST submission scored
catastrophically, and is why both the amplitude split and the vacuum subtraction exist.

Writes ``code/artifacts/models/ch2.joblib`` =
    {scaler, pca, model, amp_model, scalar, vacuum_scale, meta}.
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
from fec_common import vacuum_field as VF  # noqa: E402
from fec_common.config import MODELS_DIR, ensure_dirs, ensure_starter, log, rss_gb  # noqa: E402

ensure_starter()

from experiments import D3D_MAGNETICS_SIGNALS, TargetPCA  # noqa: E402

# The 19 DIII-D coil-current columns (excludes bcoil [toroidal field, no poloidal
# rectangle] and plasma_current [the plasma itself, not a coil]) -- matches the 19 rows
# of coil_R/coil_Z 1:1 for this machine; see vacuum_field.py.
DIII_D_COIL_COLUMNS = [
    f"magnetics_{s}" for s in D3D_MAGNETICS_SIGNALS if s not in ("bcoil", "plasma_current")
]

COLUMNS = [
    "efit_times", "efit_psirz", "efit_q95", "efit_beta_n",
    "efit_grid_R", "efit_grid_Z", "coil_R", "coil_Z", "coil_input_column",
    "magnetics_time", "magnetics_plasma_current", "magnetics_plasma_current_times",
    "thomson_core_times", "thomson_core_Te", "thomson_core_ne",
    "thomson_edge_times", "thomson_edge_Te", "thomson_edge_ne",
] + DIII_D_COIL_COLUMNS


_SCALE_EPS = 1e-6


def _rms_scale(psi: np.ndarray) -> np.ndarray:
    """Per-frame RMS amplitude, floored so normalization never divides by ~0."""
    return np.maximum(np.sqrt(np.mean(psi.reshape(len(psi), -1) ** 2, axis=1)), _SCALE_EPS)


def fit_vacuum_and_pca(files, pca_components: int, pca_frames: int):
    """One pass over a DIII-D subsample: build the coil Green's-function basis, fit the
    global vacuum-field scale by least squares, then fit the shape PCA on the *residual*
    (psi - scale * psi_vacuum), normalized per-frame by its own RMS. Returns
    ``(pca, vacuum_names, vacuum_H, vacuum_scale)``.
    """
    psi_parts, vac_parts = [], []
    names = H = None
    for path in files:
        row = F.read_parquet_row(path, columns=COLUMNS)
        psi = F._as_psirz_stack(row["efit_psirz"])
        if names is None:
            names, H = VF.build_column_green(row, row["efit_grid_R"], row["efit_grid_Z"])
        efit_times = np.asarray(row["efit_times"], dtype=np.float64)
        vac = VF.vacuum_psi_for_shot(row, efit_times, names, H)
        n = min(len(psi), len(vac))
        psi_parts.append(psi[:n]); vac_parts.append(vac[:n])
        if sum(len(p) for p in psi_parts) >= pca_frames:
            break
        del row
        gc.collect()
    psi_all = np.concatenate(psi_parts, axis=0)
    vac_all = np.concatenate(vac_parts, axis=0)
    del psi_parts, vac_parts

    # global least-squares scale: psi ~ scale * psi_vacuum, fit on DIII-D only. The same
    # scalar is applied identically to MAST at inference (predict.py) -- no MAST data
    # anywhere in this fit.
    num = float(np.sum(psi_all.astype(np.float64) * vac_all.astype(np.float64)))
    den = float(np.sum(vac_all.astype(np.float64) ** 2))
    vacuum_scale = num / den if den > 0 else 0.0
    resid = psi_all - vacuum_scale * vac_all
    r2 = 1.0 - np.sum(resid.astype(np.float64) ** 2) / max(
        np.sum((psi_all.astype(np.float64) - psi_all.mean()) ** 2), 1e-12)
    log(f"  vacuum-field scale={vacuum_scale:.4g} fit on {len(psi_all)} frames, "
        f"{len(names)} coil columns, R2(psi vs scale*vacuum)={r2:.3f}")

    scale = _rms_scale(resid)
    Y = resid / scale[:, None, None]  # shape-only, amplitude normalized out
    n_components = min(pca_components, len(Y), 65 * 65)
    pca = TargetPCA(n_components=n_components)
    pca.fit(Y)
    log(f"  ch2 PCA on {len(Y)} normalized-residual-shape frames -> {n_components} comps, "
        f"{float(np.cumsum(pca.explained_variance_ratio)[-1]) * 100:.1f}% var")
    return pca, names, H, vacuum_scale


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

    pca, vacuum_names, vacuum_H, vacuum_scale = fit_vacuum_and_pca(
        files, args.pca_components, args.pca_frames)

    X_parts, Ypca_parts, logscale_parts, scal_parts = [], [], [], []
    t0 = time.time()
    for i, path in enumerate(files):
        row = F.read_parquet_row(path, columns=COLUMNS)
        feat = F.synthetic_features_for_shot(row, "DIII-D")
        psi = F._as_psirz_stack(row["efit_psirz"])
        efit_times = np.asarray(row["efit_times"], dtype=np.float64)
        vac = VF.vacuum_psi_for_shot(row, efit_times, vacuum_names, vacuum_H)
        n = min(len(feat), len(psi), len(vac))
        resid = psi[:n] - vacuum_scale * vac[:n]
        scale = _rms_scale(resid)
        X_parts.append(feat[:n].astype(np.float32))
        Ypca_parts.append(pca.transform(resid / scale[:, None, None]))
        logscale_parts.append(np.log(scale).astype(np.float32))
        scal_parts.append(
            np.column_stack([
                np.asarray(row["efit_q95"], dtype=np.float32)[:n],
                np.asarray(row["efit_beta_n"], dtype=np.float32)[:n],
            ])
        )
        del row, feat, psi, vac, resid
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
        "vacuum_scale": vacuum_scale,
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
