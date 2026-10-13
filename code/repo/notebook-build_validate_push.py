"""Generates validate_push.ipynb -- runs 07_validate_push.py's steps in Colab instead of
locally, on the theory that Colab's network path to the HF Hub is faster than the local
machine's for the ~10GB of reference data validate_submission.py streams per config.

Deliberately thin: every cell just shells out to the same vendored CLIs
(validate_submission.py / push_predictions.py) that 07_validate_push.py already wraps -- no
new logic here, so nothing can drift between the two environments. The submission .npz files
themselves are read straight from Drive (already synced there from the local or earlier-Colab
build), so this notebook needs no local artifacts beyond that Drive sync having caught up.

To change what this notebook does: edit this file, then regenerate:
    python notebook-build_validate_push.py
"""
import json
from pathlib import Path


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


cells = []

cells.append(md("""# Fusion Equilibrium Challenge — Validate & Push (Colab)

Runs the same two vendored CLIs `07_validate_push.py` wraps locally -- `validate_submission.py`
then `push_predictions.py` -- from Colab instead, in case Colab's network path to the HF Hub is
faster for the ~10GB of reference data each validate call streams (6.7GB for the 874-shot
DIII-D config, 3.1GB for the 1206-shot MAST config -- NOT the full 92GB dataset; only the one
config being checked, and nothing gets written to disk since it's streamed/discarded per shot).

Needs `code/artifacts/submission/*.npz` to already exist on Drive (built locally or by an
earlier Colab build-submission run) and `code/.env` to hold `HF_TOKEN` / `HF_READ_TOKEN` /
`FEC_HF_REPO`. Run cells top to bottom."""))

cells.append(code("""from google.colab import drive
drive.mount('/content/drive')
"""))

cells.append(code("""import sys
from pathlib import Path

FOLDER_NAME = "EXPERIMENTS/FEC_NeurIPS/code"
CODE_DIR = Path("/content/drive/MyDrive") / FOLDER_NAME
RUN_DIR = CODE_DIR / "run"
STARTER_DIR = CODE_DIR / "starter"
SUBMISSION_DIR = CODE_DIR / "artifacts" / "submission"

assert RUN_DIR.exists(), f"{RUN_DIR} not found -- check Drive sync and FOLDER_NAME above"
assert STARTER_DIR.exists(), f"{STARTER_DIR} not found -- run 00_setup.py locally at least once"
npzs = sorted(SUBMISSION_DIR.glob("*.npz"))
assert npzs, f"no .npz under {SUBMISSION_DIR} -- build the submission first (06_build_submission.py)"
for p in npzs:
    print(f"  {p.name}: {p.stat().st_size / 1e6:.0f} MB")
"""))

cells.append(code("""# python-dotenv ships with Colab's base image already; datasets/huggingface_hub too.
import os
from dotenv import dotenv_values

env = dotenv_values(CODE_DIR / ".env")
for k in ("HF_TOKEN", "HF_READ_TOKEN", "FEC_HF_REPO"):
    v = (env.get(k) or "").strip()
    if v:
        os.environ[k] = v
    print(f"  {k}: {'set' if v else 'MISSING -- add it to code/.env'}")
"""))

cells.append(code("""# Authenticate -- same as 07_validate_push.py: fewer rate-limit retries while streaming the
# reference dataset. Non-fatal if the token's stale; validation still works, just slower.
import huggingface_hub

try:
    huggingface_hub.login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
    print("authenticated to HF Hub")
except Exception as exc:
    print(f"WARNING: HF_TOKEN invalid, continuing unauthenticated ({exc})")
"""))

cells.append(md("""## Validate

Shells out to the unmodified `validate_submission.py`, one config at a time. No progress bar
inside (vendored script prints nothing mid-loop); this cell just waits. Each config streams only
its own reference data (not the other config, not the training split)."""))

cells.append(code("""import subprocess
import sys
import time

CFG_FOR = {"diii_d_public_test.npz": "diii_d_public_test", "mast_public_test.npz": "mast_public_test"}

for npz in npzs:
    config = CFG_FOR.get(npz.name)
    if config is None:
        print(f"skipping {npz.name} (unknown config)")
        continue
    print(f"=== validating {npz.name} (config={config}) ===")
    t0 = time.time()
    rc = subprocess.call([sys.executable, "validate_submission.py", str(npz),
                          "--config", config, "--max-shots", "0"], cwd=str(STARTER_DIR))
    print(f"  exit code {rc}  ({time.time()-t0:.0f}s)")
    assert rc == 0, f"validate_submission.py reported a problem for {npz.name}"
"""))

cells.append(md("""## Push

Only run this cell once both validations above printed exit code 0. Uploads the whole
`artifacts/submission/` directory to your private HF repo and writes `submission_pointer.zip` --
that zip is what you upload to Codabench (competition 17456), not the npz files directly."""))

cells.append(code("""repo = os.environ.get("FEC_HF_REPO")
read_token = os.environ.get("HF_READ_TOKEN")
assert repo, "FEC_HF_REPO missing from code/.env"
assert read_token, "HF_READ_TOKEN missing from code/.env"

confirm = input(f"Push {SUBMISSION_DIR} to {repo}? [y/N]: ").strip().lower()
assert confirm in ("y", "yes"), "not pushing"

out_zip = SUBMISSION_DIR / "submission_pointer.zip"
rc = subprocess.call([sys.executable, "push_predictions.py",
                      "--repo", repo, "--read-token", read_token,
                      "--dir", str(SUBMISSION_DIR), "--out", str(out_zip)], cwd=str(STARTER_DIR))
print(f"exit code {rc}")
assert rc == 0, "push_predictions.py reported a problem"
print(f"upload zip ready: {out_zip} -- submit it on Codabench (competition 17456)")
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent.parent / "validate_push.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}")
