"""01_extract_features.py -- build the shared feature cache for everything downstream.

Reads the DIII-D training parquets once and writes, into ``code/artifacts/features``:

    X.f32.npy        (n_frames, n_feat) float32   raw features (scaled later per-split)
    ypca.f32.npy     (n_frames, n_pca)  float32   TargetPCA coefficients of psi
    y_scalar.f32.npy (n_frames, 6)      float32   EFIT scalar labels (NaN where undefined)
    frame_index.npz  per-frame shot id + per-frame raw index + file list
    split.json       train/val/test shot ids (split BY SHOT, starter rules)
    meta.json        n_frames / n_feat / pca info / drop counts
    pca.joblib       TargetPCA (fit on a subsample of the train split only)
    scaler.joblib    StandardScaler (fit on the train split only -- no val/test leakage)

Three passes, so nothing bigger than one shot is ever resident:

    A. scan    -- efit_times length + Thomson channel counts (column-projected)
    B. pca     -- efit_psirz only, subsample of train shots, until --pca-frames collected
    C. fill    -- one full row at a time into the memmaps

The Thomson repair in ``fec_common.features`` is what makes this cache usable at all:
the stock ``load_shot_from_hf_row`` silently drops both Thomson systems on parquet
rows, which reduced every training vector to the 21 magnetics channels.

Example:
    python code/run/01_extract_features.py --n-shots 20      # smoke
    python code/run/01_extract_features.py                   # full 7041-shot build
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fec_common import features as F  # noqa: E402
from fec_common.config import (  # noqa: E402
    FEATURES_DIR,
    FRAME_INDEX_NPZ,
    META_JSON,
    PCA_JOBLIB,
    SCALER_JOBLIB,
    SPLIT_JSON,
    X_NPY,
    YPCA_NPY,
    YSCALAR_NPY,
    ensure_dirs,
    load_json,
    log,
    rss_gb,
    save_json,
)

import joblib  # noqa: E402


def split_shots(n_shots: int, test_size: float, val_size: float, seed: int) -> dict:
    """The starter's shot-level split, reproduced exactly (train, then val, then test)."""
    shot_ids = np.arange(n_shots)
    n_test = max(1, int(n_shots * test_size))
    n_val = max(1, int(n_shots * val_size))
    n_train = max(1, n_shots - n_test - n_val)
    perm = np.random.RandomState(seed).permutation(shot_ids)
    return {
        "train": [int(i) for i in perm[:n_train]],
        "val": [int(i) for i in perm[n_train : n_train + n_val]],
        "test": [int(i) for i in perm[n_train + n_val :]],
        "random_state": seed,
        "test_size": test_size,
        "val_size": val_size,
    }


def pass_a(files) -> tuple[list[int], list[int]]:
    """Scan every shot for (frame count, natural feature width)."""
    t0 = time.time()
    counts, widths = [], []
    for i, path in enumerate(files):
        n_frames, width = F.scan_shot(path)
        counts.append(int(n_frames))
        widths.append(int(width))
        if (i + 1) % 100 == 0 or i + 1 == len(files):
            log(f"  scan {i + 1}/{len(files)}  T~{int(np.median(counts))}  "
                f"width max {max(widths)}  {time.time() - t0:.0f}s")
    return counts, widths


def pass_b(files, train_ids, pca_components: int, pca_frames: int):
    """Fit TargetPCA on a subsample of train-split psi maps (column-projected reads)."""
    from experiments import TargetPCA

    subsample = []
    n_shots_used = 0
    for sid in train_ids:
        row = F.read_parquet_row(files[sid])
        try:
            psi = F._as_psirz_stack(row["efit_psirz"])
        except Exception:
            continue
        n_shots_used += 1
        subsample.append(psi)
        n_frames = sum(len(p) for p in subsample)
        if n_frames >= pca_frames:
            break
        del row
        gc.collect()
    Y = np.concatenate(subsample, axis=0)
    n_components = min(pca_components, len(Y), 65 * 65)
    target_pca = TargetPCA(n_components=n_components)
    target_pca.fit(Y)
    cum = float(np.cumsum(target_pca.explained_variance_ratio)[-1])
    log(f"  PCA fit on {len(Y)} frames from {n_shots_used} train shots "
        f"-> {n_components} comps, {cum * 100:.1f}% variance")
    return target_pca, {"pca_frames": int(len(Y)), "pca_shots": n_shots_used,
                        "pca_variance": cum, "pca_components": int(n_components)}


def pass_c(files, out_dir: Path, counts, n_feat, target_pca):
    """Fill the memmaps one shot at a time; returns per-shot written counts."""
    total = int(sum(counts))
    X = np.lib.format.open_memmap(out_dir / X_NPY, mode="w+", dtype=np.float32,
                                  shape=(total, n_feat))
    Ypca = np.lib.format.open_memmap(out_dir / YPCA_NPY, mode="w+", dtype=np.float32,
                                     shape=(total, target_pca.n_components))
    S = np.lib.format.open_memmap(out_dir / YSCALAR_NPY, mode="w+", dtype=np.float32,
                                  shape=(total, F.N_SCALARS))

    written_per_shot = np.zeros(len(files), dtype=np.int32)
    raw_idx_parts = []
    cursor = 0
    t0 = time.time()
    n_dropped = 0
    n_padded = n_truncated = 0

    for sid, path in enumerate(files):
        row = F.read_parquet_row(path)
        shot = F.load_shot(row)
        Xs, Ys, Ss = F.build_frame(shot, include_thomson=True)

        width_before = Xs.shape[1]
        if width_before < n_feat:
            n_padded += 1
        elif width_before > n_feat:
            n_truncated += 1
        Xs = F.pad_or_truncate(Xs, n_feat)

        keep = F.valid_rows(Xs, Ys)
        if not keep.all():
            n_dropped += int((~keep).sum())
            Xs, Ys, Ss = Xs[keep], Ys[keep], Ss[keep]
        raw_idx = np.flatnonzero(keep)

        n = len(Xs)
        if n:
            X[cursor : cursor + n] = Xs
            Ypca[cursor : cursor + n] = target_pca.transform(Ys)
            S[cursor : cursor + n] = Ss
            raw_idx_parts.append(raw_idx.astype(np.int32))
        written_per_shot[sid] = n
        cursor += n

        del row, shot, Xs, Ys, Ss
        if (sid + 1) % 50 == 0 or sid + 1 == len(files):
            log(f"  fill {sid + 1}/{len(files)}  frames {cursor}/{total}  "
                f"peak RSS {rss_gb():.2f} GiB  {time.time() - t0:.0f}s")

    X.flush(); Ypca.flush(); S.flush()
    del X, Ypca, S
    gc.collect()

    # Files keep the planned (pre-drop) length in their .npy header; meta["n_frames"] is
    # the real count and every consumer slices to it, so we never rewrite headers.
    n_written = cursor

    frame_starts = np.concatenate([[0], np.cumsum(written_per_shot)]).astype(np.int64)
    shot_id = np.repeat(np.arange(len(files), dtype=np.int32), written_per_shot)
    raw_frame_idx = (
        np.concatenate(raw_idx_parts) if raw_idx_parts else np.zeros(0, dtype=np.int32)
    )
    np.savez_compressed(
        out_dir / FRAME_INDEX_NPZ,
        shot_id=shot_id,
        raw_frame_idx=raw_frame_idx,
        frame_starts=frame_starts,
        n_frames_per_shot=written_per_shot,
        file_names=np.array([str(f) for f in files]),
    )
    stats = {
        "n_written": int(n_written),
        "n_dropped": n_dropped,
        "n_padded": n_padded,
        "n_truncated": n_truncated,
    }
    return stats


def fit_scaler(out_dir: Path, frame_starts, n_frames_per_shot, train_ids) -> None:
    """Fit StandardScaler on the train split only, batched over the memmap."""
    from sklearn.preprocessing import StandardScaler

    X = np.lib.format.open_memmap(out_dir / X_NPY, mode="r", dtype=np.float32)
    n_feat = X.shape[1]
    scaler = StandardScaler()
    for i, sid in enumerate(sorted(train_ids)):
        start, end = frame_starts[sid], frame_starts[sid + 1]
        if end > start:
            scaler.partial_fit(np.asarray(X[start:end]).reshape(-1, n_feat))
        if (i + 1) % 500 == 0:
            log(f"  scaler {i + 1}/{len(train_ids)} shots")
    joblib.dump(scaler, out_dir / SCALER_JOBLIB)
    del X
    gc.collect()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-shots", type=int, default=0,
                    help="subsample of the 7041 training shots (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--pca-components", type=int, default=50)
    ap.add_argument("--pca-frames", type=int, default=50_000,
                    help="max psi frames used to fit TargetPCA (subsample of train split)")
    ap.add_argument("--redo", action="store_true", help="rebuild even if the cache exists")
    args = ap.parse_args()

    ensure_dirs()
    if (FEATURES_DIR / META_JSON).exists() and not args.redo:
        meta = load_json(FEATURES_DIR / META_JSON)
        log(f"cache already present ({meta.get('n_frames')} frames, "
            f"n_feat={meta.get('n_feat')}); use --redo to rebuild")
        return 0

    log("=== 01_extract_features ===")
    files = F.train_files(n_shots=args.n_shots, seed=args.seed)
    log(f"  {len(files)} training shots from {FEATURES_DIR.parent.parent / 'fusion-equilibrium-challenge'}")

    counts, widths = pass_a(files)
    n_feat = int(max(widths))
    log(f"  frames: {sum(counts)}   feature width: {n_feat} "
        f"(magnetics {F.N_MAGNETICS} + thomson {n_feat - F.N_MAGNETICS})")

    split = split_shots(len(files), args.test_size, args.val_size, args.seed)
    log(f"  split: train {len(split['train'])} / val {len(split['val'])} / test {len(split['test'])}")

    target_pca, pca_info = pass_b(files, split["train"], args.pca_components, args.pca_frames)
    joblib.dump(target_pca, FEATURES_DIR / PCA_JOBLIB)

    stats = pass_c(files, FEATURES_DIR, counts, n_feat, target_pca)
    frame_starts = np.load(FEATURES_DIR / FRAME_INDEX_NPZ)["frame_starts"]

    fit_scaler(FEATURES_DIR, frame_starts, None, split["train"])

    save_json(FEATURES_DIR / SPLIT_JSON, split)
    meta = {
        "n_shots": len(files),
        "n_frames": stats["n_written"],
        "n_feat": n_feat,
        "n_magnetics": F.N_MAGNETICS,
        "n_thomson": n_feat - F.N_MAGNETICS,
        "n_scalars": F.N_SCALARS,
        "include_thomson": True,
        "n_dropped": stats["n_dropped"],
        "n_padded_shots": stats["n_padded"],
        "n_truncated_shots": stats["n_truncated"],
        "natural_width_max": int(max(widths)),
        "natural_width_min": int(min(widths)),
        "seed": args.seed,
        "n_shots_requested": args.n_shots,
    }
    meta.update(pca_info)
    save_json(FEATURES_DIR / META_JSON, meta)

    from fec_common.models import update_manifest

    update_manifest(features=meta, features_dir=str(FEATURES_DIR))
    log(f"done: {stats['n_written']} frames x {n_feat} features, peak RSS {rss_gb():.2f} GiB")
    if n_feat - F.N_MAGNETICS <= 0:
        log("WARNING: no Thomson features in this cache -- models will be magnetics-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
