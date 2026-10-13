"""Generates train_conv_decoder.ipynb -- the experimental conv-decoder training notebook.

Separate from train_score_submit.ipynb on purpose: that notebook only builds/validates/pushes
against already-trained artifacts and does zero training (see its own generator's docstring).
This one is the opposite -- it's where the GPU-hungry PsiConvDecoder experiment
(fec_common/conv_decoder.py + train_conv_decoder_colab.py) actually runs, kept out of the
submission notebook so a long/experimental training run can't block a routine build-and-push.

To change what this notebook does: edit fec_common/conv_decoder.py or
fec_common/train_conv_decoder_colab.py (the actual logic), or this file (which cells call them
/ the explanatory text), then regenerate:
    python notebook-build_train_conv_decoder.py
"""
import json
from pathlib import Path

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}

cells = []

cells.append(md("""# Fusion Equilibrium Challenge — Conv Decoder Training (GPU, experimental)

Trains `PsiConvDecoder` (`code/run/fec_common/conv_decoder.py`): a small conv-transpose network
that predicts ψ(65×65) directly from the 149-dim feature vector, instead of PCA coefficients.
Rationale is in that file's docstring — the PCA+linear/MLP baselines (see `02`/`02b`/`02c`) have
decent pointwise `R²ψ` but badly negative R² on geometric shape scalars (kappa, triangularity,
volume) derived from ψ's contour; a model that treats ψ as an image instead of 50 linear
coefficients should preserve that shape better.

This notebook is **separate from `train_score_submit.ipynb`** on purpose: that one only builds
and pushes a submission from already-trained artifacts and does no training at all. This one is
pure experimentation — long-running, GPU-hungry, nothing here touches the submission pipeline
until you decide the result is good enough to wire in (see the closing cell).

Needs the Drive-synced `code/run/` and `code/artifacts/features/` (from local `00`/`01`) to
already exist — this does NOT re-extract features, only trains a new model against the 01
cache's scaler/split/meta."""))

cells.append(code("""from google.colab import drive
drive.mount('/content/drive')
# torch ships with Colab already -- nothing to pip install for this notebook.
"""))

cells.append(code("""import sys
from pathlib import Path

FOLDER_NAME = "EXPERIMENTS/FEC_NeurIPS/code"
CODE_DIR = Path("/content/drive/MyDrive") / FOLDER_NAME
RUN_DIR = CODE_DIR / "run"
ARTIFACTS_DIR = CODE_DIR / "artifacts"

assert RUN_DIR.exists(), f"{RUN_DIR} not found -- check Drive sync and FOLDER_NAME above"
sys.path.insert(0, str(RUN_DIR))

from fec_common.config import ensure_starter, log  # noqa: E402

ensure_starter()
import torch
print("CUDA available:", torch.cuda.is_available(),
     "-", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
"""))

cells.append(md("""## 0b. Sanity-check the 01 cache synced correctly

Insync can lag on large binary files (`frame_index.npz` is tens of MB) even after a small `.py`
edit has already synced -- this caught exactly that: `train()` raising ``KeyError`` on a shot id
that should exist, because Drive's copy of `frame_index.npz` was still mid-sync and shorter than
the real file. Catch that here, with a clear message, instead of a confusing traceback three
calls deep."""))

cells.append(code("""import json
import numpy as np

_idx = np.load(ARTIFACTS_DIR / "features" / "frame_index.npz", allow_pickle=True)
_split = json.load(open(ARTIFACTS_DIR / "features" / "split.json"))
_n_files = len(_idx["file_names"])
_max_shot_id = int(_idx["shot_id"].max())
_max_split_id = max(max(_split["train"]), max(_split["val"]), max(_split["test"]))

print(f"file_names entries: {_n_files}")
print(f"max shot_id seen in frames: {_max_shot_id}")
print(f"max shot id referenced by split.json: {_max_split_id}")

assert _n_files > _max_split_id, (
    f"frame_index.npz only has {_n_files} file_names entries but split.json references shot id "
    f"{_max_split_id} -- Drive sync of frame_index.npz is likely still catching up (it's a large "
    f"binary file and can lag behind small .py edits). Wait a bit for Insync to finish, then "
    f"re-run this cell before continuing."
)
print("frame_index.npz looks fully synced.")
del _idx, _split, _n_files, _max_shot_id, _max_split_id
"""))

cells.append(md("""## 1. Smoke test the architecture (no data, just shapes)

Confirms `PsiConvDecoder` produces the right output shapes before spending time loading real
shots."""))

cells.append(code("""from fec_common.conv_decoder import PsiConvDecoder, count_params
import torch as _torch

_m = PsiConvDecoder(n_feat=149)
_x = _torch.randn(8, 149)
_psi, _q95, _betan = _m(_x)
assert _psi.shape == (8, 65, 65), _psi.shape
print(f"params: {count_params(_m):,}  psi {_psi.shape}  q95 {_q95.shape}  betaN {_betan.shape}")
del _m, _x, _psi, _q95, _betan
"""))

cells.append(md("""## 2. Train

Loads real shots from the **local parquet copy** (Drive-synced), not HF streaming — training
needs ground-truth `efit_psirz`. Start small (`n_shots_train=600`) to see per-epoch timing
before committing to more; each shot is a real parquet read, so load time scales with shot
count. Checkpoints the best-val-loss model to `code/artifacts/models/flux_conv_decoder.pt`."""))

cells.append(code("""from fec_common.train_conv_decoder_colab import train

model, history = train(n_shots_train=600, n_shots_val=100, epochs=15, batch_size=256)
"""))

cells.append(md("Looks like it's still improving? Train longer / on more shots:"))

cells.append(code("""# model, history = train(n_shots_train=2000, n_shots_val=300, epochs=30, batch_size=256)
"""))

cells.append(md("""## 3. Plot loss and accuracy

Left: the composite training loss (MSE + Sobel-gradient + scalar terms) -- useful for judging
convergence, not directly comparable to anything else in this repo. Right: `psi_R2`
(pointwise R² on ψ -- the same quantity as the PCA baselines' `val_pca_r2` in `manifest.json`,
though in raw-pixel space here rather than PCA-coefficient space) plus `q95_R2`/`betaN_R2`, so
you can actually compare this run against Ridge's 0.3967 / Random Forest's 0.7626."""))

cells.append(code("""import matplotlib.pyplot as plt

epochs_ = [h["epoch"] for h in history]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs_, [h["train_loss"] for h in history], label="train")
ax1.plot(epochs_, [h["val_loss"] for h in history], label="val")
ax1.set_xlabel("epoch"); ax1.set_ylabel("loss (MSE + grad + scalar)"); ax1.legend()
ax1.set_title("Training loss")

ax2.plot(epochs_, [h["psi_r2"] for h in history], label="psi R2", marker="o")
ax2.plot(epochs_, [h["q95_r2"] for h in history], label="q95 R2", marker="o")
ax2.plot(epochs_, [h["betan_r2"] for h in history], label="betaN R2", marker="o")
ax2.axhline(0.3967, color="gray", linestyle="--", linewidth=1, label="Ridge baseline (0.3967)")
ax2.axhline(0.7626, color="black", linestyle="--", linewidth=1, label="Random Forest (0.7626)")
ax2.set_xlabel("epoch"); ax2.set_ylabel("val R2"); ax2.legend(fontsize=8)
ax2.set_title("Validation accuracy")

plt.tight_layout(); plt.show()
print(f"best epoch: psi_R2={max(h['psi_r2'] for h in history):.4f}")
"""))

cells.append(md("""---

## Next steps

- `code/artifacts/models/flux_conv_decoder.pt` now exists (best val-loss checkpoint). It is
  **not yet wired into `Predictor`/`06_build_submission.py`** — that only knows how to load
  joblib flux models that output PCA coefficients, not a PyTorch state dict that outputs ψ
  directly. Ask for that bridge once this checkpoint's val loss looks competitive.
- To judge "competitive": compare this run's final `val_loss` against the PCA baselines'
  `val_pca_r2` isn't apples-to-apples (different loss space) — the real comparison is running
  `04_local_score.py`'s composite `S`, which needs the bridge above first.
- If GPU memory is tight (3.5GB-class cards), drop `batch_size` before touching the
  architecture — see `conv_decoder.py`'s docstring."""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "name": "train_conv_decoder.ipynb", "gpuType": "T4"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent.parent / "train_conv_decoder.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, out.stat().st_size, "bytes")
