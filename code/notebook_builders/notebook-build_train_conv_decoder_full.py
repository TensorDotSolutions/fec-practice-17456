"""Generates train_conv_decoder_full.ipynb -- the FULL-SPLIT conv-decoder training run (V1
architecture; see notebook-build_train_conv_decoder_v2.py for the V2 variant).

Trains on the entire train/val split (4929/704 shots, matching exactly what the Ridge/RF/MLP
baselines in manifest.json were trained on, so its val R2 is directly comparable to their
0.3967/0.7626/etc without a subsampling caveat).

Writes its checkpoint to flux_conv_decoder_full.pt (via train()'s checkpoint_name= param) --
V2's notebook writes to a different file, so both can run as separate Colab instances without
clobbering each other.

To change what this notebook does: edit fec_common/conv_decoder.py or
fec_common/train_conv_decoder_colab.py (the actual logic), or this file, then regenerate:
    python notebook-build_train_conv_decoder_full.py
"""
import json
from pathlib import Path

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}

cells = []

cells.append(md("""# Fusion Equilibrium Challenge — Conv Decoder Training (FULL split, GPU)

Trains `PsiConvDecoder` (V1 architecture) on the **entire** train/val split (4929 train shots /
704 val shots) -- matching exactly what the Ridge/Random Forest/MLP baselines in
`manifest.json` were trained on, so this run's `val R2` is directly comparable to their
0.3967 / 0.7626 without a "but it only saw 40% of the data" caveat.

See `train_conv_decoder_v2.ipynb` for the V2 architecture (CoordConv + bilinear-upsample +
residual blocks + boundary-weighted loss), which writes to a different checkpoint file, so both
can run as separate Colab instances without clobbering each other.

Needs a GPU with real headroom: ψ targets are stored float16 on-GPU (~9GB at this scale) --
a 16GB+ card (T4/L4/A100) is required; this will not fit a small (<12GB) GPU at full scale.

Needs the Drive-synced `code/run/` and `code/artifacts/features/` (from local `00`/`01`) to
already exist -- this does NOT re-extract features, only trains a new model against the 01
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
assert torch.cuda.is_available(), "No GPU attached -- Runtime > Change runtime type > GPU"
print("CUDA:", torch.cuda.get_device_name(0),
     f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
"""))

cells.append(md("""## 0b. Sanity-check the 01 cache synced correctly

Insync can lag on large binary files (`frame_index.npz` is tens of MB) even after a small `.py`
edit has already synced. Catch that here, with a clear message, instead of a confusing
traceback three calls deep."""))

cells.append(code("""import json
import numpy as np

_idx = np.load(ARTIFACTS_DIR / "features" / "frame_index.npz", allow_pickle=True)
_split = json.load(open(ARTIFACTS_DIR / "features" / "split.json"))
_n_files = len(_idx["file_names"])
_max_split_id = max(max(_split["train"]), max(_split["val"]), max(_split["test"]))
print(f"file_names entries: {_n_files}  (train {len(_split['train'])} / val {len(_split['val'])})")

assert _n_files > _max_split_id, (
    f"frame_index.npz only has {_n_files} file_names entries but split.json references shot id "
    f"{_max_split_id} -- Drive sync is likely still catching up. Wait, then re-run this cell."
)
print("frame_index.npz looks fully synced.")
del _idx, _split, _n_files, _max_split_id
"""))

cells.append(md("""## 1. Train on the full split

This loads ALL 4929 train shots + 704 val shots via the threaded parquet loader (16 workers --
drop `load_workers` if Drive FUSE starts throttling/erroring on concurrent reads). Expect the
loading phase itself to take a while even parallelized; `tqdm` will show real progress/ETA.
Checkpoints the best-val-loss model to `code/artifacts/models/flux_conv_decoder_full.pt`."""))

cells.append(code("""from fec_common.train_conv_decoder_colab import train

model, history = train(
    n_shots_train=4929, n_shots_val=704, epochs=30, batch_size=256,
    load_workers=16, checkpoint_name="flux_conv_decoder_full.pt",
)
"""))

cells.append(md("""## 2. Plot loss and accuracy"""))

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
ax2.set_title("Validation accuracy (full split -- directly comparable to baselines)")

plt.tight_layout(); plt.show()
print(f"best epoch: psi_R2={max(h['psi_r2'] for h in history):.4f}")
"""))

cells.append(md("""---

## Next steps

- `code/artifacts/models/flux_conv_decoder_full.pt` now exists (best val-loss checkpoint, full
  split). Still **not wired into `Predictor`/`06_build_submission.py`** -- that only loads
  joblib flux models outputting PCA coefficients, not a PyTorch state dict outputting ψ
  directly. Ask for that bridge once this checkpoint's val R2 looks competitive against the
  baselines plotted above.
- If this run OOMs even at float16 storage, the `n_shots_train` would need to drop, or the
  dataset would need to move off full-GPU-residency to a chunked/streaming load -- flag this
  if that happens."""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "name": "train_conv_decoder_full.ipynb", "gpuType": "T4"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent.parent / "train_conv_decoder_full.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, out.stat().st_size, "bytes")
