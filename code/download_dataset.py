#!/usr/bin/env python3
"""Download the Sophelio/fusion-equilibrium-challenge dataset from Hugging Face.

Supports resuming interrupted downloads and shows per-file progress bars
(provided by huggingface_hub / tqdm out of the box).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

REPO_ID = "Sophelio/fusion-equilibrium-challenge"
REPO_TYPE = "dataset"

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_DIR = SCRIPT_DIR / "fusion-equilibrium-challenge"


def main() -> None:
    load_dotenv(SCRIPT_DIR / ".env")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not found in .env")

    print(f"Downloading {REPO_ID} -> {LOCAL_DIR}")
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(LOCAL_DIR),
        token=token,
        max_workers=4,
        # snapshot_download resumes partial files automatically (it uses the
        # same cache/.incomplete mechanism as hf_hub_download), so re-running
        # this script after an interruption will pick up where it left off.
    )
    print(f"Done. Dataset available at: {path}")


if __name__ == "__main__":
    main()
