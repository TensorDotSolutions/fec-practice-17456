"""Generates train_score_submit.ipynb.

Heavy preprocessing and training (00_setup.py .. 05_train_challenge2.py) run LOCALLY against
the fec_common package in code/run/ -- see the project notes. This repo folder is Insync-synced to
Google Drive, so everything those scripts write (code/artifacts/, code/fusion-equilibrium-
challenge/data/, code/starter/) shows up on Drive without any extra upload step.

This notebook's only job is the two steps that make sense to run from Colab against that
synced state: build the submission (06_build_submission.py -- streams the public test splits
from the Hub and predicts with the already-trained models) and validate/push it
(07_validate_push.py -- the official, unmodified validator/pusher). It does NOT retrain
anything, does NOT re-extract features, and does NOT duplicate fec_common's logic -- it just
puts the Drive-synced code/run/ on sys.path and calls straight into it, the same way you'd run
these scripts locally.

To change what the notebook does: edit the numbered scripts in code/run/ (or fec_common/) --
they're what actually runs, locally or in Colab. Edit this file only to change which cells call
them / the explanatory text, then regenerate:
    python notebook-build_notebook.py
"""
import json
from pathlib import Path

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}

cells = []

cells.append(md("""# Fusion Equilibrium Challenge — Build & Submit

All the heavy lifting (dataset download, feature extraction, model training, local scoring,
the Challenge-2 scaffold) happens **locally**, via `code/run/00_setup.py` .. `05_train_challenge2.py`
against the `fec_common` package. This directory is Insync-synced to Google Drive, so everything
those scripts write — `code/artifacts/` (trained models, feature cache, scores),
`code/fusion-equilibrium-challenge/data/` (the dataset), `code/starter/` (the pinned starter-kit
clone) — is already on Drive by the time you open this notebook. Nothing here retrains anything.

This notebook does exactly two things, both by calling straight into the synced
`code/run/*.py` scripts (not reimplementing them):

1. **`06_build_submission.py`** — streams the public test splits from Hugging Face (shot order
   has to come from the stream, not local files — see that script's docstring) and predicts with
   whatever model `code/artifacts/manifest.json` says is best, writing
   `code/artifacts/submission/*.npz`.
2. **`07_validate_push.py`** — runs the **official, unmodified** `validate_submission.py`, then
   (optionally) `push_predictions.py` to a private Hugging Face repo and writes
   `submission_pointer.zip`.

If `code/artifacts/manifest.json` or the model `.joblib` files aren't on Drive yet, go run the
local scripts first (`00` through at least `02`/`03`) — there is nothing for this notebook to
submit otherwise."""))

cells.append(code("""from google.colab import drive
drive.mount('/content/drive')

!pip install -q datasets huggingface_hub python-dotenv pyarrow scikit-learn scikit-image
# numpy, pandas, scipy, joblib already ship with Colab
"""))

cells.append(code("""import sys
from pathlib import Path

FOLDER_NAME = "EXPERIMENTS/FEC_NeurIPS/code"
CODE_DIR = Path("/content/drive/MyDrive") / FOLDER_NAME
RUN_DIR = CODE_DIR / "run"
ARTIFACTS_DIR = CODE_DIR / "artifacts"

assert RUN_DIR.exists(), f"{RUN_DIR} not found -- check Drive sync and FOLDER_NAME above"
sys.path.insert(0, str(RUN_DIR))

from fec_common.config import ensure_starter, load_json, log  # noqa: E402

ensure_starter()  # puts the Drive-synced code/starter/ (+ fusion_scoring/) on sys.path

manifest_path = ARTIFACTS_DIR / "manifest.json"
if manifest_path.exists():
    manifest = load_json(manifest_path)
    print("Best flux model so far:", manifest.get("val_score_model"),
          "  val S =", manifest.get("val_score"))
else:
    print(f"No manifest at {manifest_path} yet -- run the local 00..03 scripts first.")
"""))

cells.append(md("""## 1. Build the submission (quick format check, then the real thing)

Streams `diii_d_public_test` / `mast_public_test` from Hugging Face — not the local copy — so
shot order exactly matches what the scorer iterates. Writes to
`code/artifacts/submission/*.npz`. Start with `--max-shots 5`, confirm it runs clean, then build
every shot."""))

cells.append(code("!python {RUN_DIR}/06_build_submission.py --max-shots 5"))

cells.append(md("Looks right? Build the full submission (every shot, both configs):"))

cells.append(code("!python {RUN_DIR}/06_build_submission.py"))

cells.append(md("""## 2. Validate and push

Validates structure with the vendored, unmodified `validate_submission.py`. Push needs **two**
different Hugging Face tokens, easy to mix up (see the project notes' Credentials section):

- a **write** token, logged in once via `huggingface-cli login` (the cell below does this) —
  uploads the `.npz`, never leaves this runtime,
- a **fine-grained read-only** token scoped to your predictions repo, passed as `--read-token` —
  this one goes inside `submission_pointer.zip` and is what you actually submit."""))

cells.append(code("!python {RUN_DIR}/07_validate_push.py --validate-only"))

cells.append(code("""# One-time: log in with a WRITE token (https://huggingface.co/settings/tokens -> New token).
# Needed before --push; never used as --read-token below.
!huggingface-cli login
"""))

cells.append(code("""# First run (no --read-token): just creates the private repo -- Hugging Face can't scope
# a token to a repo that doesn't exist yet. Create a fine-grained READ token scoped to it,
# then re-run with --read-token to push for real.
!python {RUN_DIR}/07_validate_push.py --push --repo YOUR_USERNAME/fusion-eq-predictions
"""))

cells.append(code("""# Re-run after creating the fine-grained READ token scoped to the repo above:
!python {RUN_DIR}/07_validate_push.py --push --repo YOUR_USERNAME/fusion-eq-predictions --read-token hf_...
"""))

cells.append(md("""---

`submission_pointer.zip` (printed above, under `code/artifacts/submission/`) is what you upload
at [codabench.org/competitions/17456](https://www.codabench.org/competitions/17456/) → **Submit**.
Limits: 5/day, 100 total during Development (now → **Oct 18, 2026**); 3 total, blind, during
Final (**Oct 19–26**).

**To improve the score**, go back to the local pipeline, not this notebook:
- Raise `N_SHOTS` / retrain in `02_train_flux.py` / `03_train_scalars.py`, re-run `04_local_score.py`
  to see the real `S` before spending a submission slot.
- `05_train_challenge2.py` only has a zero scalar head for MAST `q95`/`betaN` — extending it with
  the same shared features is the natural next step.
- Check `code/artifacts/scores/local_score.json`'s per-scalar breakdown: whichever of
  `R_axis, Z_axis, kappa, tri_top, tri_bot, volume, li` scores worst tells you where ψ is
  geometrically wrong.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "name": "train_score_submit.ipynb"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent.parent / "train_score_submit.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, out.stat().st_size, "bytes")
