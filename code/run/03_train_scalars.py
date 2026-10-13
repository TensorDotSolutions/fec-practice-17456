"""03_train_scalars.py -- fit the per-scalar RidgeCV heads (q95 / betaN / ...).

    python code/run/03_train_scalars.py

Writes ``code/artifacts/models/scalars.joblib`` and a ``scalars`` block in
``artifacts/manifest.json`` (per-scalar val R2).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fec_common.config import ensure_dirs, log  # noqa: E402
from fec_common.models import load_cache, load_scaler, train_scalar_models, update_manifest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    ensure_dirs()
    log("=== 03_train_scalars ===")
    cache = load_cache()
    cache["scaler"] = load_scaler(cache)
    entry = train_scalar_models(cache)
    update_manifest(scalars=entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
