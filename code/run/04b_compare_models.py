"""04b_compare_models.py -- ranked comparison of every scored model, so comparing architectures
doesn't mean pasting numbers into chat by hand.

    python code/run/04b_compare_models.py              # every leaderboard.csv row, ranked by S
    python code/run/04b_compare_models.py --best-only   # best row per model name only

Reads artifacts/leaderboard.csv (every real composite-score run 04_local_score.py has ever
produced -- see its own docstring) and artifacts/manifest.json (every flux model trained, even
ones never composite-scored, ranked by val_pca_r2/val_psi_r2 instead). The two are different
metric spaces (composite S vs. a model's own training-time val metric) -- shown as two
separate tables, never mixed into one ranking.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fec_common.config import ARTIFACTS, load_json  # noqa: E402


def _print_table(rows: list[dict], columns: list[str]) -> None:
    if not rows:
        print("  (none)")
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    print("  " + "  ".join(c.ljust(widths[c]) for c in columns))
    print("  " + "  ".join("-" * widths[c] for c in columns))
    for r in rows:
        print("  " + "  ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--best-only", action="store_true",
                    help="collapse to the best-S run per model name")
    args = ap.parse_args()

    csv_path = ARTIFACTS / "leaderboard.csv"
    print("=== Composite score leaderboard (artifacts/leaderboard.csv) ===")
    if csv_path.exists():
        with open(csv_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if args.best_only:
            best: dict[str, dict] = {}
            for r in rows:
                name = r["model"]
                if name not in best or float(r["S"]) > float(best[name]["S"]):
                    best[name] = r
            rows = list(best.values())
        rows.sort(key=lambda r: float(r["S"]), reverse=True)
        _print_table(rows, ["model", "S", "r2_psi", "r2_qb", "consistency", "n_shots", "timestamp"])
        print(f"\n  Challenge-2 gate: DIII-D S >= 0.85")
    else:
        print("  no leaderboard.csv yet -- run 04_local_score.py at least once")

    manifest_path = ARTIFACTS / "manifest.json"
    print("\n=== Trained flux models (artifacts/manifest.json) ===")
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        flux = manifest.get("flux_models", {})
        rows = []
        for name, entry in flux.items():
            key = "val_psi_r2" if "val_psi_r2" in entry else "val_pca_r2"
            rows.append({
                "model": name,
                "metric": key,
                "value": entry.get(key, ""),
                "path": entry.get("path", ""),
                "note": entry.get("note", "")[:60],
            })
        rows.sort(key=lambda r: r["value"] if isinstance(r["value"], (int, float)) else -1,
                 reverse=True)
        _print_table(rows, ["model", "metric", "value", "path", "note"])
        print("\n  NOTE: val_pca_r2 (PCA-coefficient R2) and val_psi_r2 (pointwise ψ-image R2)\n"
              "  are DIFFERENT metric spaces -- not directly comparable to each other, only\n"
              "  within their own column. Use the composite leaderboard above for the real,\n"
              "  comparable score across every model.")
    else:
        print("  no manifest.json yet -- run 02_train_flux.py at least once")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
