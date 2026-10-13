"""02c_tune_flux_mlp.py -- Optuna search over MLPRegressor hyperparameters for the flux model.

    python code/run/02c_tune_flux_mlp.py                   # 20 trials, default
    python code/run/02c_tune_flux_mlp.py --n-trials 50 --trial-rows 150000
    python code/run/02c_tune_flux_mlp.py --timeout 3600     # wall-clock cap instead/also

Why MLP, not Random Forest, for this search: 02's "Ridge (tuned)" run (see 02b) showed Ridge is
already at its ceiling (~0.397 val PCA R2) regardless of alpha/solver -- the psi->PCA mapping is
non-linear, so a linear model can't close the gap to Random Forest's 0.76. But the RF joblib is
~20GB on disk and has hung this machine before; MLPRegressor is a much smaller artifact
(megabytes, not gigabytes) and can still capture non-linearity, so it's the safer next lever.

Memory/time guard: fitting MLP on the full ~1.09M-row train split for every trial would make a
30-trial search take hours. Each trial instead fits on a random ``--trial-rows`` subsample of
the train split (default 150,000 frames) and validates on the *full* val split (the same
held-out shots 02/04 use) -- same shot-split discipline as 02b, just with a cheaper per-trial fit.
Once the best config is found, the final model is refit on the *full* train split before being
persisted, so the saved model isn't undertrained relative to the baselines in manifest.json.

Writes code/artifacts/models/flux_mlp_tuned.joblib and a "MLP (tuned)" entry in
manifest.json's flux_models block, alongside the existing "MLP (sklearn)" baseline -- pass
``--model "MLP (tuned)"`` to 06_build_submission.py / 04_local_score.py to use it.
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


def fit_with_progress(model, X_tr, y_tr, X_va, y_va, max_iter: int, patience: int = 10,
                      label: str = "") -> int:
    """Drive MLPRegressor via partial_fit so we can print ONE overwriting progress line
    (carriage return) instead of either silence or sklearn's own one-line-per-iteration flood.
    Manual early stopping: stop if val R2 hasn't improved in `patience` iterations."""
    from fec_common.models import _r2_flat

    best_r2 = float("-inf")
    best_state = None
    no_improve = 0
    for it in range(1, max_iter + 1):
        model.partial_fit(X_tr, y_tr)
        val_r2 = _r2_flat(y_va, model.predict(X_va))
        sys.stdout.write(
            f"\r  {label} iter {it}/{max_iter}  train_loss={model.loss_:.5f}  "
            f"val_r2={val_r2:.4f}  best={best_r2:.4f}" + " " * 10
        )
        sys.stdout.flush()
        if val_r2 > best_r2:
            best_r2, no_improve = val_r2, 0
            best_state = [c.copy() for c in model.coefs_], [c.copy() for c in model.intercepts_]
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    sys.stdout.write("\n")
    if best_state is not None:
        model.coefs_, model.intercepts_ = best_state
    return it


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=None, help="wall-clock seconds, optional")
    ap.add_argument("--trial-rows", type=int, default=150_000,
                    help="train-split subsample size used per trial (0 = full train split)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import numpy as np
    import optuna
    from sklearn.neural_network import MLPRegressor

    # Optuna's own per-trial log line (study-create / "Trial N finished ... parameters: {...}")
    # is long enough to wrap across two terminal rows; we print our own shorter before/after
    # lines per trial instead (see fit_with_progress / objective below), so drop Optuna's.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ensure_dirs()
    log("=== 02c_tune_flux_mlp ===")
    cache = load_cache()
    cache["scaler"] = load_scaler(cache)

    X_tr, Ypca_tr, _, _ = load_split_arrays(cache, "train")
    X_va, Ypca_va, _, _ = load_split_arrays(cache, "val")
    log(f"  train {X_tr.shape}  val {X_va.shape}")

    rng = np.random.default_rng(args.seed)
    if args.trial_rows and args.trial_rows < len(X_tr):
        trial_idx = rng.choice(len(X_tr), size=args.trial_rows, replace=False)
        X_tr_trial, Ypca_tr_trial = X_tr[trial_idx], Ypca_tr[trial_idx]
    else:
        X_tr_trial, Ypca_tr_trial = X_tr, Ypca_tr
    log(f"  per-trial train subsample: {X_tr_trial.shape}")

    def objective(trial: "optuna.Trial") -> float:
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layer_size = trial.suggest_categorical("layer_size", [64, 128, 256, 512])
        hidden = tuple([layer_size] * n_layers)
        alpha = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)
        lr_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
        log(f"  trial {trial.number}: hidden={hidden} alpha={alpha:.2e} lr_init={lr_init:.2e} "
            f"-- fitting...")
        t_trial = time.time()
        model = MLPRegressor(
            hidden_layer_sizes=hidden, alpha=alpha, learning_rate_init=lr_init,
            random_state=args.seed,
        )
        n_iters = fit_with_progress(model, X_tr_trial, Ypca_tr_trial, X_va, Ypca_va,
                                    max_iter=300, label=f"trial {trial.number}")
        r2 = _r2_flat(Ypca_va, model.predict(X_va))
        log(f"  trial {trial.number}: val PCA R2={r2:.4f}  ({time.time()-t_trial:.0f}s, "
            f"{n_iters} iters)")
        return r2

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)
    log(f"  {len(study.trials)} trials in {time.time()-t0:.0f}s; best val PCA R2 = {study.best_value:.4f}")
    log(f"  best params: {study.best_params}")

    # Refit the winner on the FULL train split (not the trial subsample) before persisting.
    hidden = tuple([study.best_params["layer_size"]] * study.best_params["n_layers"])
    best = MLPRegressor(
        hidden_layer_sizes=hidden, alpha=study.best_params["alpha"],
        learning_rate_init=study.best_params["learning_rate_init"],
        random_state=args.seed,
    )
    t0 = time.time()
    fit_with_progress(best, X_tr, Ypca_tr, X_va, Ypca_va, max_iter=300, label="final refit")
    fit_s = time.time() - t0
    val_r2 = _r2_flat(Ypca_va, best.predict(X_va))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "flux_mlp_tuned.joblib"
    import joblib
    joblib.dump(best, path)

    entry = {
        "name": "MLP (tuned)",
        "val_pca_r2": val_r2,
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "fit_seconds": round(fit_s, 1),
        "path": path.name,
        "optuna_best_params": study.best_params,
        "optuna_n_trials": len(study.trials),
        "optuna_trial_rows": len(X_tr_trial),
    }
    update_manifest(flux_models={"MLP (tuned)": entry})
    log(f"  MLP (tuned): val PCA R2={val_r2:.4f}  (full-train refit, vs trial search's "
        f"{study.best_value:.4f} on the {len(X_tr_trial)}-row subsample)")
    log("done -- pass --model \"MLP (tuned)\" to 04_local_score.py / 06_build_submission.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
