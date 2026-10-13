"""Submission builder -- streams the public test splits from the Hub and writes the two
``.npz`` files Codabench expects (notebook cell 22).

Rows are streamed (not read from the local copy) so the shot order is exactly what the
official scorer iterates; psi is stored as float16 (never rounded), scalars as float32.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import SUBMISSION_DIR, load_env, log


def build_submission(config_name, split, source, predict_fn, out_dir, max_shots=0,
                     repo_id=None, token=None) -> Path:
    from datasets import load_dataset

    from .config import REPO_ID

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    token = load_env() if token is None else token
    ds = load_dataset(
        repo_id or REPO_ID, config_name, split=split, streaming=True, token=token or None
    )

    preds: dict = {}
    n = 0
    for i, row in enumerate(ds):
        if max_shots and i >= max_shots:
            break
        out = predict_fn(row, row.get("source", source))
        T = len(np.asarray(row["efit_times"]))
        if out["psirz"].shape[0] != T:
            raise AssertionError(
                f"{config_name} shot {i}: T mismatch {out['psirz'].shape[0]} != {T}"
            )
        preds[f"shot_{i:04d}_psirz"] = out["psirz"].astype(np.float16)
        for name in ("q95", "betaN"):
            preds[f"shot_{i:04d}_{name}"] = np.asarray(out[name], dtype=np.float32)
        n = i + 1
        if n % 25 == 0:
            log(f"  {config_name}: {n} shots")

    out_path = out_dir / f"{config_name}.npz"
    np.savez_compressed(out_path, **preds)
    log(f"  {config_name}: {n} shots -> {out_path}")
    return out_path


def build_all(test_configs, predict_fn, out_dir=None, max_shots=0) -> list[Path]:
    out_dir = Path(out_dir or SUBMISSION_DIR)
    return [
        build_submission(cfg, split, source, predict_fn, out_dir, max_shots=max_shots)
        for cfg, split, source in test_configs
    ]
