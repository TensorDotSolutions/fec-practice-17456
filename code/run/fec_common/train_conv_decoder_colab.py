"""Colab training loop for PsiConvDecoder -- paste this into a Colab cell (or !python this
file) AFTER the notebook's existing setup cell (drive mount + sys.path + ensure_starter()).

Needs torch (Colab ships it). Not part of the local CPU pipeline -- fec_common/__init__.py does
not import this module or conv_decoder.py, so nothing here affects 00..07.

Design notes
------------
* Reads real shots from the LOCAL parquet copy (synced via Drive), not HF streaming -- training
  needs ground-truth efit_psirz, which the public test streams don't have anyway, and shot-level
  random access to local files is much faster than re-streaming every epoch.
* frame_index.npz's file paths are absolute to whichever machine wrote them (this machine, not
  Colab's /content/drive/... mount) -- same portability issue 06/predict.py had with
  manifest.json's model paths. Resolved here by basename, joined against the Drive-synced
  DATA_ROOT, exactly like _load_flux_model() now does for the flux joblib paths.
* Split is BY SHOT (split.json's train/val/test shot ids), matching every other step in this
  pipeline -- never split by frame/timestep (see the project notes).
* One epoch = one pass over all frames of all train shots, shot order shuffled each epoch,
  frames within a shot kept together per batch draw (frames are cheap once a shot is loaded;
  reloading a 65x65xT parquet row is the expensive part, so this script batches across shots'
  concatenated frames rather than reopening files more than once per epoch).

Usage (inside Colab, after the setup cell has already run ensure_starter()):

    from fec_common.train_conv_decoder_colab import train
    model, history = train(n_shots_train=600, n_shots_val=100, epochs=15, batch_size=256)
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import ARTIFACTS, DATA_ROOT, FEATURES_DIR, MODELS_DIR, ensure_starter, load_json, log
from .conv_decoder import PsiConvDecoder, PsiConvDecoderV2, psi_loss, psi_loss_v2
from .features import build_frame, load_shot, pad_or_truncate

_ARCHS = {
    "v1": (PsiConvDecoder, psi_loss),
    "v2": (PsiConvDecoderV2, psi_loss_v2),
}

ensure_starter()
from experiments import EFIT_SCALAR_LABELS  # noqa: E402

_Q95_IDX = EFIT_SCALAR_LABELS.index("efit_q95")
_BETAN_IDX = EFIT_SCALAR_LABELS.index("efit_beta_n")


def _resolve_local_path(stored_path: str) -> Path:
    """frame_index.npz's file paths are absolute to the machine that wrote them -- re-join by
    basename against this machine's DATA_ROOT (local disk, or Drive-synced in Colab)."""
    name = Path(stored_path).name
    return DATA_ROOT / "diii_d_train" / name


def _load_one_shot(sid, files_by_id, n_feat):
    """Read + build one shot's frames. Standalone so it can run in a worker thread -- the
    actual file read (Drive FUSE, network-latency-bound) releases the GIL, so concurrent
    threads genuinely overlap their wait time instead of just taking turns."""
    path = _resolve_local_path(files_by_id[sid])
    if not path.exists():
        return sid, None
    import pandas as pd

    row = pd.read_parquet(path).iloc[0].to_dict()
    shot = load_shot(row)
    X, Y, S = build_frame(shot, include_thomson=True)
    X = pad_or_truncate(X, n_feat)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    good = np.isfinite(Y.reshape(len(Y), -1)).all(axis=1)
    # Cast Y to float16 HERE, before it ever reaches the results dict -- at full-split scale
    # (4929+704 shots) float32 psi alone is ~21GB; holding that plus a second concatenated
    # copy at once (the old list+np.concatenate pattern) is what pushed this past 52GB of host
    # RAM and crashed the kernel. float16 halves it and this avoids ever allocating the float32
    # version at full-dataset scale in the first place.
    return sid, (X[good].astype(np.float32, copy=False),
                Y[good].astype(np.float16, copy=False),
                S[good, _Q95_IDX].astype(np.float32, copy=False),
                S[good, _BETAN_IDX].astype(np.float32, copy=False))


class ShotFrameDataset(Dataset):
    """Loads+caches frames for a fixed set of shot ids. Built once per epoch's shot subset."""

    def __init__(self, shot_ids, files_by_id, n_feat, scaler, desc: str = "shots",
                max_workers: int = 16):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from tqdm.auto import tqdm

        self.n_feat = n_feat
        self.scaler = scaler
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_load_one_shot, sid, files_by_id, n_feat): sid
                      for sid in shot_ids}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="shot"):
                sid, result = fut.result()
                if result is None:
                    log(f"  WARNING: missing shot {sid}, skipping")
                    continue
                results[sid] = result

        # Keep shot_ids' original order for reproducibility (as_completed finishes out of order).
        # Pop from `results` as we go (not just iterate) so each shot's arrays drop to a single
        # live reference (in Xs/Ys/...) instead of two (dict + list) before concatenation.
        Xs, Ys, S_q95, S_betan = [], [], [], []
        for sid in shot_ids:
            if sid in results:
                X, Y, q95, betan = results.pop(sid)
                Xs.append(X)
                Ys.append(Y)
                S_q95.append(q95)
                S_betan.append(betan)
        del results
        self.X = np.concatenate(Xs).astype(np.float32)
        del Xs
        # Y stays float16 -- see _load_one_shot's comment. Cast up to float32 per-batch in
        # train(), not here, or this would re-create the exact OOM this avoids.
        self.Y = np.concatenate(Ys)
        del Ys
        self.q95 = np.nan_to_num(np.concatenate(S_q95)).astype(np.float32)
        self.betan = np.nan_to_num(np.concatenate(S_betan)).astype(np.float32)
        self.X = self.scaler.transform(self.X).astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i]),
                torch.tensor(self.q95[i]), torch.tensor(self.betan[i]))


def _load_shared_artifacts():
    meta = load_json(FEATURES_DIR / "meta.json")
    split = load_json(FEATURES_DIR / "split.json")
    scaler = joblib.load(FEATURES_DIR / "scaler.joblib")
    idx = np.load(FEATURES_DIR / "frame_index.npz", allow_pickle=True)
    # file_names is per-SHOT (one entry per shot, directly indexed by shot id) -- NOT
    # parallel to shot_id, which is per-FRAME (1.56M entries). Index it directly by shot id.
    file_names = idx["file_names"]
    files_by_id = {sid: file_names[sid] for sid in range(len(file_names))}
    return meta, split, scaler, files_by_id


def train(n_shots_train: int = 600, n_shots_val: int = 100, epochs: int = 15,
         batch_size: int = 256, lr: float = 1e-3, seed: int = 42,
         device: str | None = None, load_workers: int = 16,
         checkpoint_name: str = "flux_conv_decoder.pt", arch: str = "v1"):
    """arch: "v1" (original ConvTranspose2d stack, matches every checkpoint trained so far) or
    "v2" (CoordConv + bilinear-upsample + residual blocks + boundary-weighted loss -- targets
    the Consistency term's shape scalars specifically, see conv_decoder.py's V2 docstring).
    The checkpoint stores which one was used (`arch` key) so predict.py's TorchConvDecoderModel
    picks the matching class automatically -- old checkpoints without that key default to v1."""
    if arch not in _ARCHS:
        raise ValueError(f"arch must be one of {list(_ARCHS)}, got {arch!r}")
    model_cls, loss_fn = _ARCHS[arch]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        log("WARNING: device='cuda' requested but torch.cuda.is_available() is False "
            "-- falling back to cpu. Check Runtime > Change runtime type > GPU in Colab.")
        device = "cpu"
    if device == "cuda":
        log(f"device: cuda ({torch.cuda.get_device_name(0)}, "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        log("device: cpu")

    meta, split, scaler, files_by_id = _load_shared_artifacts()
    n_feat = int(meta["n_feat"])

    rng = np.random.default_rng(seed)
    train_ids = rng.choice(split["train"], size=min(n_shots_train, len(split["train"])),
                           replace=False)
    val_ids = rng.choice(split["val"], size=min(n_shots_val, len(split["val"])), replace=False)

    log(f"loading {len(train_ids)} train shots + {len(val_ids)} val shots from local parquet...")
    t0 = time.time()
    train_ds = ShotFrameDataset(train_ids, files_by_id, n_feat, scaler, desc="train shots",
                               max_workers=load_workers)
    val_ds = ShotFrameDataset(val_ids, files_by_id, n_feat, scaler, desc="val shots",
                             max_workers=load_workers)
    log(f"  {len(train_ds)} train frames / {len(val_ds)} val frames  ({time.time()-t0:.0f}s)")

    # Move the whole dataset onto `device` ONCE instead of batching through a CPU-bound
    # DataLoader every step -- with a model this small, per-step compute is fast enough that
    # DataLoader overhead was the actual bottleneck, leaving the GPU idle between steps
    # (reported: "GPU not used"). psi (Y) is kept float16 on-GPU -- at full scale (~4929 train
    # shots) it's ~18GB at float32, which doesn't fit even a 16GB T4; float16 halves that to
    # ~9GB, leaving headroom for activations/optimizer state. Cast back to float32 per-batch
    # for the loss (negligible cost, avoids float16 gradient underflow in the loss itself).
    X_tr = torch.from_numpy(train_ds.X).to(device)
    Y_tr = torch.from_numpy(train_ds.Y).to(device, dtype=torch.float16)
    q95_tr = torch.from_numpy(train_ds.q95).to(device)
    betan_tr = torch.from_numpy(train_ds.betan).to(device)
    X_va = torch.from_numpy(val_ds.X).to(device)
    Y_va = torch.from_numpy(val_ds.Y).to(device, dtype=torch.float16)
    q95_va = torch.from_numpy(val_ds.q95).to(device)
    betan_va = torch.from_numpy(val_ds.betan).to(device)

    log(f"arch: {arch} ({model_cls.__name__})")
    model = model_cls(n_feat=n_feat).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    # Precompute val-set variance once (R2 denominator) so each epoch only needs the residual
    # sum of squares -- avoids holding every prediction in memory just to call sklearn's r2_score.
    # dtype=np.float64 is required here: val_ds.Y is float16 (kept that way for memory -- see
    # ShotFrameDataset), and np.var's default accumulator on a float16 input over millions of
    # elements loses enough precision to silently produce a WRONG variance (observed: psi_R2
    # printing exactly 1.0000 from epoch 1, which is not real -- the denominator was off).
    psi_var = float(np.var(val_ds.Y, dtype=np.float64))
    q95_var = float(np.var(val_ds.q95))
    betan_var = float(np.var(val_ds.betan))

    history = []
    best_val = float("inf")
    best_state = None
    n_train = len(train_ds)
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train - batch_size + 1, batch_size):
            idx = perm[start:start + batch_size]
            X, Y, q95, betan = X_tr[idx], Y_tr[idx].float(), q95_tr[idx], betan_tr[idx]
            opt.zero_grad()
            psi_pred, q95_pred, betan_pred = model(X)
            loss = loss_fn(psi_pred, Y, q95_pred, q95, betan_pred, betan)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(X)
        train_loss /= (n_train - n_train % batch_size)

        model.eval()
        val_loss = 0.0
        psi_sse = q95_sse = betan_sse = 0.0
        n_val = len(val_ds)
        with torch.no_grad():
            for start in range(0, n_val, batch_size):
                X = X_va[start:start + batch_size]
                Y = Y_va[start:start + batch_size].float()
                q95 = q95_va[start:start + batch_size]
                betan = betan_va[start:start + batch_size]
                psi_pred, q95_pred, betan_pred = model(X)
                batch_loss = loss_fn(psi_pred, Y, q95_pred, q95, betan_pred, betan)
                val_loss += batch_loss.item() * len(X)
                psi_sse += ((psi_pred - Y) ** 2).sum().item()
                q95_sse += ((q95_pred - q95) ** 2).sum().item()
                betan_sse += ((betan_pred - betan) ** 2).sum().item()
        val_loss /= n_val
        n_psi = len(val_ds) * 65 * 65
        psi_r2 = 1.0 - psi_sse / n_psi / psi_var
        q95_r2 = 1.0 - q95_sse / len(val_ds) / q95_var
        betan_r2 = 1.0 - betan_sse / len(val_ds) / betan_var
        sched.step(val_loss)

        if not np.isfinite(val_loss):
            log(f"  WARNING: val_loss is {val_loss} (non-finite) at epoch {epoch+1} -- "
                f"keeping previous best checkpoint, not this one.")
        elif val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        log(f"  epoch {epoch+1:2d}/{epochs}  train_loss {train_loss:.5f}  val_loss {val_loss:.5f}"
            f"  psi_R2 {psi_r2:.4f}  q95_R2 {q95_r2:.4f}  betaN_R2 {betan_r2:.4f}"
            f"  ({time.time()-t0:.0f}s)")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                        "psi_r2": psi_r2, "q95_r2": q95_r2, "betan_r2": betan_r2})

    if best_state is None:
        log("WARNING: val_loss was non-finite every epoch -- no valid checkpoint to restore. "
            "Saving the LAST epoch's weights instead of crashing, but this run likely diverged.")
    else:
        model.load_state_dict(best_state)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / checkpoint_name
    torch.save({"state_dict": model.state_dict(), "n_feat": n_feat, "arch": arch}, out_path)
    log(f"saved best model (val_loss={best_val:.5f}, arch={arch}) -> {out_path}")
    return model, history


def evaluate(model, n_shots_val: int = 704, seed: int = 42, device: str | None = None,
            batch_size: int = 256, load_workers: int = 16) -> dict:
    """Recompute correct val-set R2 for an already-trained model -- no retraining needed.

    For re-scoring a model that trained under the pre-fix psi_var bug (np.var on a float16
    array silently miscomputing the R2 denominator, making psi_R2 print 1.0000 regardless of
    real accuracy). Pass the `model` object already sitting in your Colab session's memory
    after a `train(...)` call -- this just re-loads val shots fresh and scores it correctly.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    meta, split, scaler, files_by_id = _load_shared_artifacts()
    n_feat = int(meta["n_feat"])
    rng = np.random.default_rng(seed)
    val_ids = rng.choice(split["val"], size=min(n_shots_val, len(split["val"])), replace=False)

    log(f"loading {len(val_ids)} val shots for evaluation...")
    val_ds = ShotFrameDataset(val_ids, files_by_id, n_feat, scaler, desc="eval val shots",
                             max_workers=load_workers)
    log(f"  {len(val_ds)} val frames")

    X_va = torch.from_numpy(val_ds.X).to(device)
    Y_va = torch.from_numpy(val_ds.Y).to(device, dtype=torch.float16)
    q95_va = torch.from_numpy(val_ds.q95).to(device)
    betan_va = torch.from_numpy(val_ds.betan).to(device)

    # dtype=np.float64: see train()'s identical comment -- float16 input to np.var silently
    # gives a wrong (usually inflated) variance over millions of elements.
    psi_var = float(np.var(val_ds.Y, dtype=np.float64))
    q95_var = float(np.var(val_ds.q95))
    betan_var = float(np.var(val_ds.betan))

    psi_sse = q95_sse = betan_sse = 0.0
    n_val = len(val_ds)
    with torch.no_grad():
        for start in range(0, n_val, batch_size):
            X = X_va[start:start + batch_size]
            Y = Y_va[start:start + batch_size].float()
            q95 = q95_va[start:start + batch_size]
            betan = betan_va[start:start + batch_size]
            psi_pred, q95_pred, betan_pred = model(X)
            psi_sse += ((psi_pred - Y) ** 2).sum().item()
            q95_sse += ((q95_pred - q95) ** 2).sum().item()
            betan_sse += ((betan_pred - betan) ** 2).sum().item()

    n_psi = n_val * 65 * 65
    result = {
        "psi_r2": 1.0 - psi_sse / n_psi / psi_var,
        "q95_r2": 1.0 - q95_sse / n_val / q95_var,
        "betan_r2": 1.0 - betan_sse / n_val / betan_var,
        "n_val_shots": len(val_ids),
        "n_val_frames": n_val,
    }
    log(f"corrected eval: psi_R2={result['psi_r2']:.4f}  q95_R2={result['q95_r2']:.4f}  "
        f"betaN_R2={result['betan_r2']:.4f}  ({n_val} frames)")
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shots-train", type=int, default=600)
    ap.add_argument("--n-shots-val", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    train(n_shots_train=args.n_shots_train, n_shots_val=args.n_shots_val,
         epochs=args.epochs, batch_size=args.batch_size)
