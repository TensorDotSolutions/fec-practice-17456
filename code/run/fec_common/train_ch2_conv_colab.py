"""Colab training loop for Ch2PhysConvNet (Challenge-2 / MAST zero-shot conv model).

Paste into a Colab cell (or `!python` this file) after the notebook's setup cell (drive
mount + sys.path + ``ensure_starter()``). Needs torch (Colab ships it) -- not part of the
local CPU pipeline.

Why this exists: ``05_train_challenge2.py`` fits a flat 7-feature RidgeCV through a PCA
basis -- a linear model with no spatial structure, and it has run out of road (real
Codabench score: S_MAST ~0.057, Consistency pinned at 0.0). This trains
``conv_decoder.Ch2PhysConvNet`` on the 5-channel *image* representation from
``ch2_features.py`` instead -- a vacuum-field Green's-function map (real shared physics,
computed identically from each machine's own coil geometry), CoordConv channels, a
core-Thomson pressure profile, and a broadcast Ip channel. Same shape/amplitude-split
target as ``05_train_challenge2.py``: trains on ``psi_true - vacuum_scale * vacuum_psi``
(the plasma's own contribution, not the whole flux map), with ``vacuum_scale`` fit by
least squares on DIII-D only and reused unchanged on MAST.

Memory sizing: each frame's image stack is 5 x 65 x 65 x 2 bytes (float16) = ~42 KB --
~5x heavier per frame than the plain conv decoder's psi-only target. At the full DIII-D
train split (4929 shots, ~1.09M frames) that is ~46 GB, too large even for a 23GB-VRAM
GPU alongside everything else. This defaults to a bounded SUBSET of shots
(``n_shots_train``/``n_shots_val``) loaded fully onto the GPU, the same "whole dataset
resident, no DataLoader-per-step" strategy as ``train_conv_decoder_colab.py`` -- just
sized down to fit. Defaults (1500 train / 300 val shots) use roughly 12-15 GB on a
23GB GPU; raise them if your run has headroom, lower them if it OOMs.

Usage (inside Colab, after the setup cell has already run ``ensure_starter()``):

    from fec_common.train_ch2_conv_colab import train
    model, history, bundle = train(n_shots_train=1500, n_shots_val=300, epochs=20)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import ch2_features as CF
from . import vacuum_field as VF
from .config import DATA_ROOT, MODELS_DIR, ensure_starter, load_json, FEATURES_DIR, log
from .conv_decoder import Ch2PhysConvNet, ch2_loss

ensure_starter()
from experiments import D3D_MAGNETICS_SIGNALS  # noqa: E402

DIII_D_COIL_COLUMNS = [
    f"magnetics_{s}" for s in D3D_MAGNETICS_SIGNALS if s not in ("bcoil", "plasma_current")
]
COLUMNS = [
    "efit_psirz", "efit_times", "efit_q95", "efit_beta_n",
    "efit_grid_R", "efit_grid_Z", "coil_R", "coil_Z", "coil_input_column",
    "magnetics_time", "magnetics_plasma_current", "magnetics_plasma_current_times",
    "magnetics_bcoil",
    "thomson_core_R", "thomson_core_Te", "thomson_core_ne", "thomson_core_times",
] + DIII_D_COIL_COLUMNS

_EPS = 1e-6


def _resolve_local_path(stored_path: str) -> Path:
    return DATA_ROOT / "diii_d_train" / Path(stored_path).name


def _rms_norm(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x64 = x.astype(np.float64)
    scale = np.maximum(np.sqrt(np.mean(x64.reshape(len(x64), -1) ** 2, axis=1)), _EPS)
    return (x64 / scale[:, None, None]).astype(np.float32), scale


def _load_one_shot(sid, files_by_id, vacuum_names, vacuum_H):
    path = _resolve_local_path(files_by_id[sid])
    if not path.exists():
        return sid, None
    import pandas as pd

    row = pd.read_parquet(path, columns=COLUMNS).iloc[0]
    from .features import _as_psirz_stack

    psi = _as_psirz_stack(row["efit_psirz"])
    efit_times = np.asarray(row["efit_times"], dtype=np.float64)
    images, aux = CF.build_ch2_images(row, "DIII-D", vacuum_names, vacuum_H)
    # amplitude comes from the physics law, NOT from a learned head -- so the target the net
    # sees is already sigma-normalized and the net only ever learns a scale-free shape.
    sigma = CF.physics_sigma(row, "DIII-D", efit_times)
    n = min(len(psi), len(images), len(sigma))
    q95 = np.nan_to_num(np.asarray(row["efit_q95"], dtype=np.float32)[:n])
    betan = np.nan_to_num(np.asarray(row["efit_beta_n"], dtype=np.float32)[:n])
    return sid, (images[:n].astype(np.float16, copy=False), aux[:n], psi[:n],
                 sigma[:n].astype(np.float32), q95, betan)


class Ch2ShotDataset(Dataset):
    """Loads+caches a fixed set of shot ids' image/target arrays. Built once per call."""

    def __init__(self, shot_ids, files_by_id, vacuum_names, vacuum_H, desc: str = "shots",
                max_workers: int = 16):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from tqdm.auto import tqdm

        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_load_one_shot, sid, files_by_id, vacuum_names, vacuum_H): sid
                      for sid in shot_ids}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="shot"):
                sid, result = fut.result()
                if result is None:
                    log(f"  WARNING: missing shot {sid}, skipping")
                    continue
                results[sid] = result

        imgs, auxs, psis, sigs, q95s, betans = [], [], [], [], [], []
        for sid in shot_ids:
            if sid in results:
                im, ax, ps, sg, q, b = results.pop(sid)
                imgs.append(im); auxs.append(ax); psis.append(ps)
                sigs.append(sg); q95s.append(q); betans.append(b)
        del results

        self.images = np.concatenate(imgs); del imgs
        self.aux = np.concatenate(auxs).astype(np.float32); del auxs
        psi = np.concatenate(psis).astype(np.float64); del psis
        self.sigma = np.concatenate(sigs).astype(np.float64); del sigs
        self.q95 = np.concatenate(q95s)
        self.betan = np.concatenate(betans)

        # Target: psi / sigma_physics. NOT mean-subtracted -- psi's DC offset is real
        # (|mean|/std ~ 0.8), is scored by R2, and does not follow the Ip*R0 law, so the
        # network has to carry it. No vacuum_scale and no log-amplitude head are involved:
        # the vacuum field is now purely an INPUT channel (where only its relative
        # structure matters), and absolute amplitude is physics at inference time.
        self.shape = np.clip(psi / self.sigma[:, None, None],
                             -CF.CH2_SHAPE_CLIP, CF.CH2_SHAPE_CLIP).astype(np.float32)

    def __len__(self):
        return len(self.images)


def _load_shared(n_shots_train, n_shots_val, seed):
    split = load_json(FEATURES_DIR / "split.json")
    from .features import read_parquet_row, train_files

    rng = np.random.default_rng(seed)
    train_ids = rng.choice(split["train"], size=min(n_shots_train, len(split["train"])),
                           replace=False)
    val_ids = rng.choice(split["val"], size=min(n_shots_val, len(split["val"])), replace=False)

    idx = np.load(FEATURES_DIR / "frame_index.npz", allow_pickle=True)
    file_names = idx["file_names"]
    files_by_id = {sid: file_names[sid] for sid in range(len(file_names))}

    # geometry is constant across DIII-D shots -- build the Green's-function basis once
    # from whichever shot loads first.
    first_path = _resolve_local_path(files_by_id[int(train_ids[0])])
    first_row = read_parquet_row(first_path, columns=["efit_grid_R", "efit_grid_Z", "coil_R",
                                                       "coil_Z", "coil_input_column"])
    vacuum_names, vacuum_H = VF.build_column_green(first_row, first_row["efit_grid_R"],
                                                    first_row["efit_grid_Z"])
    return train_ids, val_ids, files_by_id, vacuum_names, vacuum_H


def train(n_shots_train: int = 1500, n_shots_val: int = 300, epochs: int = 20,
         batch_size: int = 128, lr: float = 1e-3, seed: int = 42,
         device: str | None = None, load_workers: int = 16,
         checkpoint_name: str = "ch2_conv.pt"):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        log("WARNING: cuda requested but unavailable -- falling back to cpu.")
        device = "cpu"
    if device == "cuda":
        log(f"device: cuda ({torch.cuda.get_device_name(0)}, "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        log("device: cpu")

    train_ids, val_ids, files_by_id, vacuum_names, vacuum_H = _load_shared(
        n_shots_train, n_shots_val, seed)

    log(f"loading {len(train_ids)} train shots + {len(val_ids)} val shots "
       f"({len(vacuum_names)} coil columns)...")
    t0 = time.time()
    train_ds = Ch2ShotDataset(train_ids, files_by_id, vacuum_names, vacuum_H,
                              desc="train shots", max_workers=load_workers)
    val_ds = Ch2ShotDataset(val_ids, files_by_id, vacuum_names, vacuum_H,
                            desc="val shots", max_workers=load_workers)
    log(f"  {len(train_ds)} train frames / {len(val_ds)} val frames  ({time.time()-t0:.0f}s)")
    log(f"  sigma_physics (train) median={np.median(train_ds.sigma):.4g} Wb  "
        f"target psi/sigma std={np.std(train_ds.shape):.3f}")

    img_tr = torch.from_numpy(train_ds.images).to(device, dtype=torch.float16)
    aux_tr = torch.from_numpy(train_ds.aux).to(device)
    shape_tr = torch.from_numpy(train_ds.shape).to(device)
    q95_tr = torch.from_numpy(train_ds.q95).to(device)
    betan_tr = torch.from_numpy(train_ds.betan).to(device)
    img_va = torch.from_numpy(val_ds.images).to(device, dtype=torch.float16)
    aux_va = torch.from_numpy(val_ds.aux).to(device)
    shape_va = torch.from_numpy(val_ds.shape).to(device)
    sig_va = torch.from_numpy(val_ds.sigma.astype(np.float32)).to(device)
    q95_va = torch.from_numpy(val_ds.q95).to(device)
    betan_va = torch.from_numpy(val_ds.betan).to(device)

    shape_var = float(np.var(val_ds.shape, dtype=np.float64))
    q95_var = float(np.var(val_ds.q95)) + 1e-12
    betan_var = float(np.var(val_ds.betan)) + 1e-12

    # Pooled psi-space stats -- psi_R2 below is the in-domain analogue of the metric the
    # official scorer actually reports (R2_psi), unlike shape_R2 which is measured on the
    # sigma-normalized target. Accumulated in chunks so the full psi stack is never resident.
    _n = _s = _s2 = 0.0
    for _i in range(0, len(val_ds), 4096):
        _p = (val_ds.shape[_i:_i + 4096].astype(np.float64)
              * val_ds.sigma[_i:_i + 4096, None, None])
        _n += _p.size; _s += float(_p.sum()); _s2 += float(np.sum(_p ** 2))
    psi_mean = _s / _n
    psi_sstot = _s2 - _n * psi_mean ** 2
    log(f"  val psi: mean={psi_mean:+.4f} Wb  SS_tot={psi_sstot:.4g}")

    model = Ch2PhysConvNet(in_channels=CF.N_CHANNELS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

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
            images, aux = img_tr[idx].float(), aux_tr[idx]
            shape = shape_tr[idx]
            q95, betan = q95_tr[idx], betan_tr[idx]
            opt.zero_grad()
            shape_pred, q95_pred, betan_pred = model(images, aux)
            loss = ch2_loss(shape_pred, shape, q95_pred, q95, betan_pred, betan)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(images)
        train_loss /= (n_train - n_train % batch_size)

        model.eval()
        val_loss = 0.0
        shape_sse = q95_sse = betan_sse = 0.0
        psi_sse = psi_sse_shrunk = 0.0
        n_val = len(val_ds)
        with torch.no_grad():
            for start in range(0, n_val, batch_size):
                images = img_va[start:start + batch_size].float()
                aux = aux_va[start:start + batch_size]
                shape = shape_va[start:start + batch_size]
                sig = sig_va[start:start + batch_size][:, None, None]
                q95 = q95_va[start:start + batch_size]
                betan = betan_va[start:start + batch_size]
                shape_pred, q95_pred, betan_pred = model(images, aux)
                bl = ch2_loss(shape_pred, shape, q95_pred, q95, betan_pred, betan)
                val_loss += bl.item() * len(images)
                shape_sse += ((shape_pred - shape) ** 2).sum().item()
                q95_sse += ((q95_pred - q95) ** 2).sum().item()
                betan_sse += ((betan_pred - betan) ** 2).sum().item()
                psi_true = sig * shape
                psi_sse += ((sig * shape_pred - psi_true) ** 2).sum().item()
                psi_sse_shrunk += ((CF.CH2_AMP_SHRINK * sig * shape_pred
                                    - psi_true) ** 2).sum().item()
        val_loss /= n_val
        n_shape = n_val * 65 * 65
        shape_r2 = 1.0 - shape_sse / n_shape / shape_var
        q95_r2 = 1.0 - q95_sse / n_val / q95_var
        betan_r2 = 1.0 - betan_sse / n_val / betan_var
        psi_r2 = 1.0 - psi_sse / psi_sstot
        psi_r2_shrunk = 1.0 - psi_sse_shrunk / psi_sstot
        sched.step(val_loss)

        if not np.isfinite(val_loss):
            log(f"  WARNING: val_loss non-finite at epoch {epoch+1} -- keeping previous best.")
        elif val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        log(f"  epoch {epoch+1:2d}/{epochs}  train_loss {train_loss:.5f}  val_loss {val_loss:.5f}"
            f"  shape_R2 {shape_r2:.4f}  psi_R2 {psi_r2:.4f}"
            f"  psi_R2@b={CF.CH2_AMP_SHRINK:g} {psi_r2_shrunk:.4f}"
            f"  q95_R2 {q95_r2:.4f}  betaN_R2 {betan_r2:.4f}  ({time.time()-t0:.0f}s)")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                        "shape_r2": shape_r2, "psi_r2": psi_r2,
                        "psi_r2_shrunk": psi_r2_shrunk, "q95_r2": q95_r2,
                        "betan_r2": betan_r2})

    if best_state is None:
        log("WARNING: val_loss non-finite every epoch -- saving LAST epoch's weights.")
    else:
        model.load_state_dict(best_state)

    bundle = {
        "state_dict": model.state_dict(),
        "arch": "ch2phys_sigma_qtf",   # sigma-normalized target + TF-coil q_proxy aux dim
        "in_channels": CF.N_CHANNELS,
        "ip_norm": CF.CH2_IP_NORM,
        "sigma_c": CF.CH2_SIGMA_C,
        "amp_shrink": CF.CH2_AMP_SHRINK,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / checkpoint_name
    torch.save(bundle, out_path)
    log(f"saved best model (val_loss={best_val:.5f}) -> {out_path}")
    return model, history, bundle


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shots-train", type=int, default=1500)
    ap.add_argument("--n-shots-val", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    train(n_shots_train=args.n_shots_train, n_shots_val=args.n_shots_val,
         epochs=args.epochs, batch_size=args.batch_size)
