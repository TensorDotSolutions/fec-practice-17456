# Reproduction package (Fusion Equilibrium Challenge, Codabench #17739)

This file is the reproduction-package reference required by competition rule 6 for
Final-phase submissions. It points at the exact weights, entrypoints, and data provenance
behind the submitted predictions.

## What's submitted

- **Challenge 1 (DIII-D)**: `flux_conv_decoder_v2.pt` — `PsiConvDecoderV2` (CoordConv +
  bilinear-upsample + residual blocks + boundary-weighted loss), trained in Colab.
- **Challenge 2 (MAST, zero-shot)**: `ch2.joblib` — a shape/amplitude-split RidgeCV on 7
  machine-agnostic shared features (plasma current + Thomson pressure-proxy stats per
  system), fit on DIII-D only, no MAST labels used anywhere in training or selection beyond
  the permitted Development-phase score comparison (see Disclosures below).

Weights actually used for this submission are committed under `code/release_weights/`:
`flux_conv_decoder_v2.pt`, `ch2.joblib`, `scalars.joblib`, `scaler.joblib`, `pca.joblib`,
`meta.json`. (`code/artifacts/` itself stays gitignored — it also holds superseded models,
e.g. an unused Random Forest checkpoint, ~19 GB, not part of this submission.)

## Training entrypoints (rebuild from scratch)

| Component | Entrypoint | Hardware | Wall-clock |
|---|---|---|---|
| Features (PCA, scaler) | `code/.venv/bin/python code/run/01_extract_features.py` | CPU | ~20 min |
| Flux model (Conv Decoder V2) | `code/train_conv_decoder_v2.ipynb` (Colab) | 1 GPU | ~3.7 h (load ~20 min + 30 epochs × ~7 min) |
| Scalar heads | `code/.venv/bin/python code/run/03_train_scalars.py` | CPU | a few min |
| Challenge-2 model | `code/.venv/bin/python code/run/05_train_challenge2.py` | CPU | ~20 min (7,041 DIII-D shots) |

Total declared compute budget: **~3.7 GPU-hours** (Challenge 1, Colab) — well under the
24 GPU-hour ceiling. Everything else is CPU-only local feature/model fitting, declared here
per rule 6's CPU-work disclosure requirement, uncapped.

## Inference entrypoint (regenerate the submitted `.npz`)

```
code/.venv/bin/python code/run/06_build_submission.py --model "Conv Decoder V2" \
    --repo TariqJamil/fusion-eq-predictions --read-token $HF_READ_TOKEN
code/.venv/bin/python code/run/07_validate_push.py --push \
    --repo TariqJamil/fusion-eq-predictions --read-token $HF_READ_TOKEN
```

Pinned predictions commit: `6c8cb6488bca515b04d6bebbcc9cf01ef8c51cd2` on
`TariqJamil/fusion-eq-predictions` (private HF dataset repo, read-scoped token submitted via
`manifest.json`).

## Environment

`code/requirements-local.txt` (installed into `code/.venv`, created with
`--system-site-packages`, so numpy/pandas/pyarrow/scikit-learn/torch come from the system
interpreter — see `CLAUDE.md`). Conv Decoder V2 trained in Colab's standard PyTorch/CUDA
image; no extra pinning beyond `torch`.

## Data provenance (per machine)

- **DIII-D**: `Sophelio/fusion-equilibrium-challenge` (`diii_d_train`, CC BY 4.0, Hugging
  Face), used for all Challenge-1 and Challenge-2 training. No external DIII-D data.
- **MAST**: `Sophelio/fusion-equilibrium-challenge` (`mast_public_test`, CC BY 4.0) — inputs
  only (coil currents, Thomson profiles, machine geometry). No MAST equilibrium/EFIT data
  from any source (including FAIR-MAST) was used to train, fine-tune, calibrate, or select
  any component producing MAST predictions, per rule 3.

No other external datasets or pre-trained models were used.

## Disclosures (rule 3 / rule 4)

- **Development-phase score comparison**: the Challenge-2 model variant that was ultimately
  submitted was chosen by comparing this team's own Development-phase S_MAST scores across
  two of its own prior submissions (0.0276 → 0.0575, after fixing a plasma-current unit bug),
  per rule 3's explicit permission to use the platform's own feedback for pipeline selection.
  No MAST equilibrium data was used to produce either candidate or to make this comparison.
- **AI coding assistance**: substantial parts of this entry's code (feature engineering, the
  conv-decoder architecture, training scripts, and this reproduction package) were written
  with Claude Code (Anthropic), under the entrant's direction and review, per rule 4.
