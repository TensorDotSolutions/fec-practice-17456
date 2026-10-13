"""Generates train_conv_decoder_<variant>.ipynb for every architecture variant in VARIANTS --
one shared template, parameterized per variant, instead of a near-duplicate generator script
per architecture (the previous arrangement: notebook-build_train_conv_decoder_full.py and
notebook-build_train_conv_decoder_v2.py were ~90% identical, differing only in arch/checkpoint
name/description text -- any shared-cell change had to be hand-applied twice).

Each variant trains on the FULL train/val split (4929/704 shots), matching exactly what the
Ridge/RF/MLP baselines in manifest.json were trained on, so every variant's val R2 is directly
comparable. Each writes to its OWN checkpoint file (checkpoint_name), so any subset of these
notebooks can run as separate, simultaneous Colab instances without clobbering each other.

To add a new architecture variant: add a model_cls to fec_common/conv_decoder.py + an "arch"
key to train_conv_decoder_colab.py's _ARCHS dict (the actual logic), then add one entry to
VARIANTS below and regenerate:
    python notebook-build_train_conv_decoder.py                 # regenerates every variant
    python notebook-build_train_conv_decoder.py v2               # regenerates one variant only

The --only-variant form matters when another variant's notebook is a live Colab session's
autosave (Insync-synced back down) -- regenerating ALL variants would clobber whatever that
session wrote, mid-run.
"""
import json
import sys
from pathlib import Path

VARIANTS = [
    {
        "key": "full",
        "arch": "v1",
        "checkpoint_name": "flux_conv_decoder_full.pt",
        "out_name": "train_conv_decoder_full.ipynb",
        "title": "Conv Decoder Training (FULL split, GPU)",
        "intro": """Trains `PsiConvDecoder` (V1 architecture) on the **entire** train/val split
(4929 train shots / 704 val shots) -- matching exactly what the Ridge/Random Forest/MLP
baselines in `manifest.json` were trained on, so this run's `val R2` is directly comparable to
their 0.3967 / 0.7626 without a "but it only saw 40% of the data" caveat.

See `train_conv_decoder_v2.ipynb` for the V2 architecture (CoordConv + bilinear-upsample +
residual blocks + boundary-weighted loss), which writes to a different checkpoint file, so both
can run as separate Colab instances without clobbering each other.""",
        "check_arch_synced": False,
        "baselines": [
            ("Ridge baseline (0.3967)", 0.3967, "gray"),
            ("Random Forest (0.7626)", 0.7626, "black"),
        ],
        "next_steps": """- `code/artifacts/models/flux_conv_decoder_full.pt` now exists (best val-loss checkpoint,
  full split). Still **not wired into `Predictor`/`06_build_submission.py`** -- that only
  loads joblib flux models outputting PCA coefficients, not a PyTorch state dict outputting ψ
  directly. Wire that bridge once this checkpoint's val R2 looks competitive against the
  baselines plotted above.
- If this run OOMs even at float16 storage, `n_shots_train` would need to drop, or the dataset
  would need to move off full-GPU-residency to a chunked/streaming load.""",
    },
    {
        "key": "v2",
        "arch": "v2",
        "checkpoint_name": "flux_conv_decoder_v2.pt",
        "out_name": "train_conv_decoder_v2.ipynb",
        "title": "Conv Decoder V2 Training (FULL split, GPU)",
        "intro": """Trains `PsiConvDecoderV2` (`fec_common/conv_decoder.py`) on the **entire**
train/val split (4929 train shots / 704 val shots) -- same split as
`train_conv_decoder_full.ipynb`'s V1 run, so results are directly comparable.

**Why V2**: the real composite scorer (`04_local_score.py`, not just pointwise R2) showed V1
hitting `psi_R2≈0.99` but the Consistency term's shape scalars -- kappa, tri_top/bot,
R_axis/Z_axis -- sitting at only 0.68-0.78. Those come from the scorer re-tracing the
**predicted** ψ's LCFS contour, so no auxiliary scalar head can shortcut it; the fix has to be
ψ's own spatial fidelity. V2 adds:
- **CoordConv** -- concatenates fixed (R,Z) coordinate channels into every decoder block, so
  the network has absolute-position awareness (helps R_axis/Z_axis specifically, which are
  defined by absolute grid position).
- **Bilinear-upsample decoder blocks instead of ConvTranspose2d** -- avoids the checkerboard
  artifacts transposed convs are known for, which corrupt the contour the scorer traces.
- **Residual blocks** at each resolution stage for more capacity without harder optimization.
- **Boundary-weighted loss** -- upweights pixel error near the separatrix (steep-gradient
  region), exactly where kappa/tri_top/tri_bot/volume are measured, instead of spending most
  gradient budget on the easy flat interior/exterior.

**Run this as a separate Colab instance** from `train_conv_decoder_full.ipynb` if that's still
running -- different checkpoint file, no collision.""",
        "check_arch_synced": True,
        "baselines": [
            ("V1 psi_R2 (0.9911)", 0.9911, "gray"),
        ],
        "next_steps": """The plot above is pointwise `psi_R2`/`q95_R2`/`betaN_R2` -- it does NOT show whether this
variant actually improved the Consistency term (kappa/tri_top/tri_bot/R_axis/Z_axis/volume/li),
since those only come from the real composite scorer, which runs **locally**, not in Colab:

```
uv run --python .venv code/run/04_local_score.py --model "Conv Decoder V2" --n-shots 50
```

`flux_conv_decoder_v2.pt` needs registering in `manifest.json` under a model name first (wire
this up once this checkpoint has synced back from Drive) -- then that command gives the real,
comparable `S` / `consistency` numbers against V1's existing `S=0.9429` / `consistency=0.766`
(n=50, in `code/artifacts/scores/local_score.json`).""",
    },
]


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def build(variant: dict) -> dict:
    cells = []

    cells.append(md(f"""# Fusion Equilibrium Challenge — {variant['title']}

{variant['intro']}

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

    if variant["check_arch_synced"]:
        cells.append(md(f"""## 0c. Sanity-check the {variant['arch']} code synced correctly

The `arch="{variant['arch']}"` support in `train_conv_decoder_colab.py` and its model class in
`conv_decoder.py` may have been added after the files above last synced -- confirm they're
actually here before a multi-hour run, instead of finding out via an ImportError or (worse)
silently training the wrong architecture if the arch param were ever ignored."""))

        cells.append(code(f"""import fec_common.conv_decoder as _cd
import fec_common.train_conv_decoder_colab as _m

assert "{variant['arch']}" in _m._ARCHS, (
    "train_conv_decoder_colab.py's _ARCHS dict is missing '{variant['arch']}' -- Drive sync "
    "hasn't caught up yet. Wait a bit and re-run this cell (or Runtime > Restart runtime if it "
    "still fails after waiting)."
)
print("{variant['arch']} code is present and synced.")
del _cd, _m
"""))

    cells.append(md(f"""## 1. Train on the full split

Loads ALL 4929 train shots + 704 val shots via the threaded parquet loader (16 workers -- drop
`load_workers` if Drive FUSE starts throttling/erroring on concurrent reads). Checkpoints the
best-val-loss model to `code/artifacts/models/{variant['checkpoint_name']}`."""))

    cells.append(code(f"""from fec_common.train_conv_decoder_colab import train

model, history = train(
    n_shots_train=4929, n_shots_val=704, epochs=30, batch_size=256,
    load_workers=16, arch="{variant['arch']}", checkpoint_name="{variant['checkpoint_name']}",
)
"""))

    cells.append(md("""## 2. Plot loss and accuracy"""))

    baseline_lines = "\n".join(
        f'ax2.axhline({value}, color="{color}", linestyle="--", linewidth=1, label="{label}")'
        for label, value, color in variant["baselines"]
    )
    cells.append(code(f"""import matplotlib.pyplot as plt

epochs_ = [h["epoch"] for h in history]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs_, [h["train_loss"] for h in history], label="train")
ax1.plot(epochs_, [h["val_loss"] for h in history], label="val")
ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend()
ax1.set_title("Training loss")

ax2.plot(epochs_, [h["psi_r2"] for h in history], label="psi R2", marker="o")
ax2.plot(epochs_, [h["q95_r2"] for h in history], label="q95 R2", marker="o")
ax2.plot(epochs_, [h["betan_r2"] for h in history], label="betaN R2", marker="o")
{baseline_lines}
ax2.set_xlabel("epoch"); ax2.set_ylabel("val R2"); ax2.legend(fontsize=8)
ax2.set_title("Validation accuracy (full split -- directly comparable to baselines)")

plt.tight_layout(); plt.show()
print(f"best epoch: psi_R2={{max(h['psi_r2'] for h in history):.4f}}")
"""))

    cells.append(md(f"""---

## Next steps

{variant['next_steps']}"""))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "name": variant["out_name"], "gpuType": "T4"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    variants = [v for v in VARIANTS if only is None or v["key"] == only]
    if only and not variants:
        raise SystemExit(f"no variant with key {only!r} -- choose from {[v['key'] for v in VARIANTS]}")
    for variant in variants:
        nb = build(variant)
        out = Path(__file__).resolve().parent.parent / variant["out_name"]
        out.write_text(json.dumps(nb, indent=1))
        print("wrote", out, out.stat().st_size, "bytes")
