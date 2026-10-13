"""02b_tune_flux_ridge.py -- Optuna search over Ridge hyperparameters for the flux (psi) model.

    python code/run/02b_tune_flux_ridge.py                  # 30 trials, default
    python code/run/02b_tune_flux_ridge.py --n-trials 100
    python code/run/02b_tune_flux_ridge.py --timeout 1800    # wall-clock cap instead/also

Why this exists alongside 02_train_flux.py's "Ridge (CV)" baseline: that one is
``RidgeCV(alphas=np.logspace(-3, 3, 10))`` -- it cross-validates *within* the training rows
using sklearn's generic K-fold, which mixes frames from different shots across folds. Every
other step in this pipeline (02's own val split, 04's scoring) deliberately splits by shot, not
by frame/timestep, because frames within one shot are highly correlated (see CLAUDE.md). This
script instead scores each Optuna trial against the *same* held-out val shots 02 and 04 use, so
the alpha it picks is validated the same way the final model will be judged -- and widens the
search beyond RidgeCV's fixed 10-point grid (alpha down to 1e-6, up to 1e4; both solvers that
support sparse/dense dense input here).

Writes code/artifacts/models/flux_ridge_tuned.joblib and a "Ridge (tuned)" entry in
manifest.json's flux_models block, alongside (not replacing) "Ridge (CV)" -- pass
``--model "Ridge (tuned)"`` to 06_build_submission.py / 04_local_score.py to use it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fec_common.config import ensure_dirs, log  # noqa: E402
from fec_common.models import (  # noqa: E402
    MODELS_DIR,
    _r2_flat,
    load_cache,
    load_scaler,
    load_split_arrays,
    update_manifest,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=None, help="wall-clock seconds, optional")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import optuna
    from sklearn.linear_model import Ridge

    ensure_dirs()
    log("=== 02b_tune_flux_ridge ===")
    cache = load_cache()
    cache["scaler"] = load_scaler(cache)

    X_tr, Ypca_tr, _, _ = load_split_arrays(cache, "train")
    X_va, Ypca_va, _, _ = load_split_arrays(cache, "val")
    log(f"  train {X_tr.shape}  val {X_va.shape}")

    def objective(trial: "optuna.Trial") -> float:
        alpha = trial.suggest_float("alpha", 1e-6, 1e4, log=True)
        solver = trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr"])
        model = Ridge(alpha=alpha, solver=solver, random_state=args.seed)
        model.fit(X_tr, Ypca_tr)
        pred = model.predict(X_va)
        return _r2_flat(Ypca_va, pred)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)
    log(f"  {len(study.trials)} trials in {time.time()-t0:.0f}s; best val PCA R2 = {study.best_value:.4f}")
    log(f"  best params: {study.best_params}")

    # Refit the winner once more and persist it, same as train_flux_model() does.
    best = Ridge(alpha=study.best_params["alpha"], solver=study.best_params["solver"],
                random_state=args.seed)
    t0 = time.time()
    best.fit(X_tr, Ypca_tr)
    fit_s = time.time() - t0
    val_r2 = _r2_flat(Ypca_va, best.predict(X_va))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "flux_ridge_tuned.joblib"
    import joblib
    joblib.dump(best, path)

    entry = {
        "name": "Ridge (tuned)",
        "val_pca_r2": val_r2,
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "fit_seconds": round(fit_s, 1),
        "path": path.name,
        "optuna_best_params": study.best_params,
        "optuna_n_trials": len(study.trials),
    }
    update_manifest(flux_models={"Ridge (tuned)": entry})
    log(f"  Ridge (tuned): val PCA R2={val_r2:.4f}  (vs RidgeCV baseline -- check manifest.json)")
    log("done -- pass --model \"Ridge (tuned)\" to 04_local_score.py / 06_build_submission.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
