# Fusion Equilibrium Challenge — Pipeline

Personal pipeline for the Fusion Equilibrium Challenge (NeurIPS competition, Codabench #17456):
reconstruct a tokamak's 2-D poloidal flux map ψ(R,Z) (65×65, per timestep) from coil currents +
Thomson scattering alone (no magnetic sensors), for DIII-D (Challenge 1) and zero-shot for MAST
(Challenge 2).

## How this was actually run

**Everything ran on a local machine, except model training, which ran on Google Colab (GPU).**
The local machine has no GPU with enough headroom for the conv-decoder architecture used here,
so training was always a Colab step; every other stage is CPU-bound and ran locally.

| stage | where | what |
|---|---|---|
| Setup | local | clone the pinned official starter kit, create the local venv |
| Feature extraction | local | stream all 7041 train shots once into on-disk memmaps (`01_extract_features.py`) |
| Baselines (Ridge/RF/MLP) | local | fit on PCA-compressed ψ targets (`02`/`02b`/`02c`) |
| **Conv-decoder training** | **Colab (GPU)** | train a PyTorch model that predicts ψ directly as an image, not PCA coefficients |
| Scalar heads (q95/βN) | local | per-scalar RidgeCV (`03_train_scalars.py`) |
| Local scoring | local | real composite scorer (pointwise R², LCFS consistency, etc.) against held-out shots (`04_local_score.py`, ranked across every model with `04b_compare_models.py`) |
| Challenge-2 (MAST, zero-shot) | local | shared-feature scalar head fit on DIII-D only |
| Build submission | local or Colab | stream the public test splits from the Hub, run every shot through the trained models, write `.npz` |
| Validate + push | local | run the official (vendored, unmodified) validator, then push to a private Hugging Face dataset repo |

Why training is the one GPU step: the first approach (PCA-compressed ψ + a linear/tree/MLP
regressor) trains fine on CPU but tops out around R²≈0.76 on the PCA coefficients, and the
flux-map's *shape* (contour-derived quantities like elongation/triangularity/volume) comes out
badly wrong even when the pointwise error looks fine — PCA linearizes the image away before any
model sees it. A small convolutional decoder that treats ψ as an actual 65×65 image preserves
that shape much better, but needs a GPU for the conv-transpose/upsample stack over the full
~1.1M-frame train split; Colab's free-tier GPUs (T4/L4) are enough.

## Repository layout

- `code/run/00_setup.py` … `07_validate_push.py` — the staged local pipeline, one script per
  step, run in order with the local venv.
- `code/run/fec_common/` — shared feature engineering, model training, scoring, and submission
  logic used by every local script.
- `code/run/fec_common/conv_decoder.py`, `train_conv_decoder_colab.py` — the GPU-only
  conv-decoder architecture and training loop, imported from Colab notebooks (needs `torch`,
  which the local CPU venv deliberately doesn't install for training).
- `code/train_conv_decoder_full.ipynb` / `code/train_conv_decoder_v2.ipynb` — Colab notebooks
  that train the conv decoder (V1: ConvTranspose2d stack; V2: CoordConv + bilinear-upsample +
  residual blocks + boundary-weighted loss, aimed at the LCFS-contour shape scalars V1 was
  weakest on) on the full train/val split. Both generated from the **same** template
  (`notebook_builders/notebook-build_train_conv_decoder.py`'s `VARIANTS` list) -- adding another
  architecture variant means adding one entry there, not writing a new near-duplicate notebook.
- `code/validate_push.ipynb` — Colab variant of the local validate/push step, for when Colab's
  network path to the Hugging Face Hub is faster than the local connection.
- `code/notebook_builders/notebook-build_*.py` — generators for every notebook above; notebooks are
  regenerated from these, never hand-edited.
- `docs/` — vendored copies of the official starter kit's README and modeling guide.

## Comparing models/architectures

`code/artifacts/manifest.json` is the single source of truth for every trained model (name,
checkpoint path, validation metric, timestamp); `code/artifacts/leaderboard.csv` is an
append-only log of every real composite-score run `04_local_score.py` has produced. Run
`04b_compare_models.py` for a ranked view of both instead of comparing numbers by hand:

```
uv run --python .venv code/run/04b_compare_models.py --best-only
```

## Known fixed bug: Challenge-2 (MAST) Ip unit mismatch

The first scored submission's Challenge-2 (MAST, zero-shot) result was not just "weak transfer"
— it was a genuine unit bug. `synthetic_features_for_shot` (in `fec_common/features.py`)
assumed DIII-D's `magnetics_plasma_current` was already in amps (no conversion) and multiplied
MAST's by 1000 assuming it was in kA needing conversion to A. Empirically, both machines' raw
values are already in kA (DIII-D tops out ~1000-1700, i.e. ~1-1.7 MA; MAST tops out ~400-1000,
matching MAST's known current range) — so the old code introduced a spurious ~1000x scale gap
between the two machines' dominant shared feature. Fed through a `StandardScaler` fit only on
DIII-D, MAST's Ip landed ~2500 standard deviations out of distribution, and a linear `RidgeCV`
has no mechanism to saturate on that — it just extrapolates, which is exactly what produced
R² in the hundreds-of-thousands-to-millions-negative range on the scored submission (not
ordinary "bad transfer" noise).

Fixed by removing both unit conversions (verified the gap collapses from ~2500σ to ~1.7σ after
the fix). `fec_common/predict.py`'s `challenge2_predict` also now clamps its output to the
physical range observed in DIII-D training data (`psi ∈ [-1.5, 1.5]`, `q95 ∈ [0, 40]`,
`betaN ∈ [-1, 6]`) as cheap insurance against any future out-of-distribution blow-up from a
linear model with no other guardrail — re-run `05_train_challenge2.py` after any change to
`synthetic_features_for_shot`.

## Credentials

Two different Hugging Face tokens are used and must not be confused: a **write** token (local
`huggingface-cli`/`hf auth login`, used to upload predictions) and a **fine-grained read-only**
token scoped to just the private predictions repo (shipped inside the submission pointer zip so
Codabench's backend can fetch the predictions on your behalf). Neither token is committed to
this repository — see `.gitignore`.
