"""02_train_flux.py -- fit the psi models on the 01 cache and rank them on the val split.

    python code/run/02_train_flux.py                       # all four starter models
    python code/run/02_train_flux.py --models ridge mlp
    python code/run/02_train_flux.py --score               # also run the cheap scorer

Writes ``code/artifacts/models/flux_*.joblib`` and a ``flux_models`` block in
``artifacts/manifest.json`` (val PCA R2 per model, plus composite components if
``--score``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fec_common.config import ARTIFACTS, ensure_dirs, log  # noqa: E402
from fec_common.models import (  # noqa: E402
    available_flux_models,
    load_cache,
    load_scaler,
    train_flux_model,
    update_manifest,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None,
                    help="substrings of starter model names (default: all four)")
    ap.add_argument("--score", action="store_true",
                    help="also compute the cheap R2 components on the val split")
    args = ap.parse_args()

    ensure_dirs()
    log("=== 02_train_flux ===")
    cache = load_cache()
    cache["scaler"] = load_scaler(cache)
    log(f"  cache: {cache['n_frames']} frames x {cache['n_feat']} features "
        f"(train {len(cache['split']['train'])} / val {len(cache['split']['val'])} shots)")

    results = {}
    for name in available_flux_models(args.models):
        results[name] = train_flux_model(cache, name, score=args.score)

    update_manifest(flux_models=results)
    ranked = sorted(results.items(),
                    key=lambda kv: kv[1].get("val_pca_r2", float("-inf")), reverse=True)
    for name, entry in ranked:
        log(f"  {name:>20s}  val PCA R2 = {entry['val_pca_r2']:.4f}")
    log(f"best so far: {ranked[0][0]}  (04_local_score.py picks it unless --model is given)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
