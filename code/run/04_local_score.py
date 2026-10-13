"""04_local_score.py -- run the real composite metric on held-out DIII-D shots.

    python code/run/04_local_score.py --n-shots 20
    python code/run/04_local_score.py --model "Ridge (CV)" --n-shots 50

The holdout is drawn from every training shot the model did NOT see (the train
split is excluded), so the number stays honest at any training scale -- unlike the
notebook's seed-999 holdout, which could overlap the seed-42 training subset.

Writes ``artifacts/scores/local_score.json``, appends to ``artifacts/leaderboard.csv``
and records the score in ``manifest.json`` under ``val_score`` for the scored model.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fec_common.config import ARTIFACTS, SCORES_DIR, ensure_dirs, load_json, log, save_json  # noqa: E402
from fec_common.models import best_flux_model, load_cache, update_manifest  # noqa: E402
from fec_common.predict import Predictor  # noqa: E402
from fec_common.scoring import compute_local_score, load_scoring_shots  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-shots", type=int, default=20)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--model", default=None, help="flux model name (default: best val PCA R2)")
    args = ap.parse_args()

    ensure_dirs()
    log("=== 04_local_score ===")
    cache = load_cache()
    model_name = args.model or best_flux_model()
    predictor = Predictor.from_artifacts(model_name=model_name)
    log(f"  model: {predictor.name}   n_feat={predictor.n_feat}   "
        f"ch2={'yes' if predictor.ch2 else 'no'}")

    train_files = {cache["files"][sid] for sid in cache["split"]["train"]}
    shots = load_scoring_shots(n_shots=args.n_shots, seed=args.seed, exclude=train_files)

    t0 = time.time()
    res = compute_local_score(shots, predictor.challenge1_predict)
    res["model"] = model_name
    res["seconds"] = round(time.time() - t0, 1)
    res["exclude_train_shots"] = len(train_files)

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    save_json(SCORES_DIR / "local_score.json", res)

    csv_path = ARTIFACTS / "leaderboard.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(["timestamp", "model", "S", "r2_psi", "r2_qb", "dlcfs",
                        "consistency", "n_shots", "n_feat"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), model_name, f"{res['S']:.4f}",
                    f"{res['r2_psi']:.4f}", f"{res['r2_qb']:.4f}",
                    f"{res['dlcfs']:.4f}", f"{res['consistency']:.4f}",
                    res["n_shots"], predictor.n_feat])

    update_manifest(val_score=float(res["S"]), val_score_model=model_name,
                    val_score_n_shots=int(res["n_shots"]))
    log(f"  gate for Challenge 2 is DIII-D S >= 0.85; this run: {res['S']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
