"""06_build_submission.py -- stream the public test splits and write the two .npz files.

    python code/run/06_build_submission.py --max-shots 5     # format check
    python code/run/06_build_submission.py                   # the real submission

Streams from the Hub (not the local copy) so shot order matches what the official
scorer iterates.  DIII-D goes through the strong per-machine model, MAST through the
shared-feature Challenge-2 model.  Writes to ``code/artifacts/submission/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fec_common.build import build_all  # noqa: E402
from fec_common.config import SUBMISSION_DIR, TEST_CONFIGS, ensure_dirs, load_env, log  # noqa: E402
from fec_common.predict import Predictor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-shots", type=int, default=0, help="0 = full submission")
    ap.add_argument("--model", default=None, help="flux model (default: best in manifest)")
    ap.add_argument("--out-dir", default=str(SUBMISSION_DIR))
    ap.add_argument("--only", choices=[c for c, _, _ in TEST_CONFIGS], default=None,
                    help="build just one config")
    args = ap.parse_args()

    ensure_dirs()
    token = load_env()
    log(f"=== 06_build_submission  (max_shots={args.max_shots or 'ALL'}) ===")
    predictor = Predictor.from_artifacts(model_name=args.model)
    log(f"  flux model: {predictor.name}   n_feat={predictor.n_feat}   "
        f"ch2={'yes' if predictor.ch2 else 'NO (MAST will be zeros)'}")
    if token is None:
        log("  HF_TOKEN not set -- the dataset is public, so this still works")

    configs = [c for c in TEST_CONFIGS if args.only is None or c[0] == args.only]
    written = build_all(configs, predictor.full_predict, out_dir=args.out_dir,
                        max_shots=args.max_shots)
    for path in written:
        log(f"  wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    log("next: 07_validate_push.py --validate-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
