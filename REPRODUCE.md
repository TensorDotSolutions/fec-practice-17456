# Reproduction package (Fusion Equilibrium Challenge, Codabench #17739)

This file is the reproduction-package reference required by competition rule 6 for
Final-phase submissions. It points at the exact weights, entrypoints, and data provenance
behind the submitted predictions.

## What's submitted

- **Challenge 1 (DIII-D)**: `flux_conv_decoder_v2.pt` — `PsiConvDecoderV2` (CoordConv +
  bilinear-upsample + residual blocks + boundary-weighted loss), trained in Colab.
- **Challenge 2 (MAST, zero-shot)**: `ch2_conv.pt` — `Ch2PhysConvNet` (arch tag
  `ch2phys_sigma_qtf`), a CoordConv residual net over a 5-channel 65×65 *image*
  representation built from each machine's own geometry (coil-only vacuum flux map from a
  Green's-function/Biot–Savart calculation, CoordConv R/Z, a core-Thomson pressure profile,
  and a broadcast Ip channel), plus a 3-entry dimensionless aux vector
  (`log(vacuum_rms/σ)`, `Ip` z-score, `log(|I_tf|/|Ip|)`) feeding the q95/βN heads. It
  predicts only a **σ-normalized shape** (`ψ/σ`); absolute amplitude comes from a physics
  scaling law `σ = 6.003e-5·|Ip[kA]|·R0[m]` (fit on DIII-D, R² = 0.915) evaluated on each
  machine's own Ip and R0, times a deliberate shrink `β = 0.5`. Fit on DIII-D only; no MAST
  labels used anywhere in training or selection beyond the permitted Development-phase score
  comparison (see Disclosures below).

  Held-out DIII-D validation (20 epochs): shape R² 0.808, ψ R² 0.805, q95 R² 0.968,
  βN R² 0.945.

  Superseded: `ch2.joblib` (shape/amplitude-split RidgeCV on 7 flat shared features) scored
  `S_MAST = 0.0575`; a first conv version with a *learned* amplitude head scored 0.002; the
  physics-amplitude rework (no `q_proxy` aux) scored `S_MAST = 0.183` but left the q95/βN
  heads with no direct physics handle, transferring to MAST at `R²{q95,βN} = -0.27`. See
  "Challenge-2 corrections" below for why.

Weights actually used for this submission are committed under `code/release_weights/`:
`flux_conv_decoder_v2.pt`, `ch2_conv.pt`, `scalars.joblib`, `scaler.joblib`, `pca.joblib`,
`meta.json` (plus the superseded `ch2.joblib` for provenance). (`code/artifacts/` itself stays gitignored — it also holds superseded models,
e.g. an unused Random Forest checkpoint, ~19 GB, not part of this submission.)

## Training entrypoints (rebuild from scratch)

| Component | Entrypoint | Hardware | Wall-clock |
|---|---|---|---|
| Features (PCA, scaler) | `.venv/bin/python code/run/01_extract_features.py` | CPU | ~20 min |
| Flux model (Conv Decoder V2) | `code/train_conv_decoder_v2.ipynb` (Colab) | 1 GPU | ~3.7 h (load ~20 min + 30 epochs × ~7 min) |
| Scalar heads | `.venv/bin/python code/run/03_train_scalars.py` | CPU | a few min |
| Challenge-2 model (conv) | `code/train_ch2_conv.ipynb` (Colab) | 1 GPU | ~1.3 h (20 epochs × ~4 min + load) |
| Challenge-2 model (superseded RidgeCV) | `python code/run/05_train_challenge2.py` | CPU | ~20 min (7,041 DIII-D shots) |

Total declared compute budget: **~5 GPU-hours** (3.7 h Challenge 1 + 1.3 h Challenge 2, both
Colab) — well under the 24 GPU-hour ceiling. Everything else is CPU-only local feature/model
fitting, declared here per rule 6's CPU-work disclosure requirement, uncapped.

## Challenge-2 corrections (why the model changed)

Four defects were found and fixed across the `S_MAST = 0.002` → `0.183` submissions. All
are in `code/run/fec_common/`; none affect the Challenge-1 path, which never touches the
Green's-function code and is self-consistent with whatever index order the target uses.

1. **`efit_psirz` is stored `[iZ, iR]`, not `[iR, iZ]`** (`vacuum_field.py`,
   `ch2_features.py`). The Green's-function basis and the CoordConv/pressure channels were
   built transposed against the target. Verified against the shipped `efit_r_axis` /
   `efit_z_axis`: `argmin(ψ)` lands 0.017 m from the true magnetic axis under `[iZ, iR]`
   (sub-grid-cell; dR = 0.027 m) versus 0.105 m transposed. The grid is square, so a
   transpose raises no shape error, and DIII-D's near-up-down-symmetric equilibria make a
   transposed map look only mildly wrong — it has to be tested against the axis position.
2. **MAST P-coil turns were double-counted** (`vacuum_field.py`).
   `magnetics_p{2-6}{l,u}_current` ships already-turns-folded ampere-turns, but the basis
   summed each column's Green's function over all 20/8/23/23/4 conductor-element rows,
   an 8–23× overcount. Those columns are now averaged; `magnetics_sol_current` is the
   opposite case (a true per-turn current across 656 series elements) and is still summed.
3. **The learned amplitude head could not transfer.** It conditioned on `log(vacuum_rms)`,
   whose MAST distribution is out-of-sample, and overshot ~7× (predicted max\|ψ\| 1.12 Wb
   against a physics-implied 0.156 Wb) — matching the `k ≈ 6` overshoot implied by
   `R²ψ = −25.9` via `R² = 1−(k−1)²`. Since a *constant* prediction scores `R² = 0`, that
   head was strictly worse than predicting nothing. It was replaced by the dimensional
   scaling law above.
4. **q95/βN heads had no direct physics handle.** Fed only pooled shape features plus
   `vac_rel`/`ip_z`, they scored `R² = 0.88–0.92` on held-out DIII-D but `R² = -0.27` on
   MAST (`S_MAST = 0.183` submission). Added a third aux dimension,
   `q_proxy = log(|I_tf|/|Ip|)`, using the toroidal-field coil current
   (`magnetics_bcoil` DIII-D / `magnetics_tf_current` MAST — both genuine inputs, no
   EFIT/MAST labels) as a cross-machine proxy for the edge safety factor
   (`q ~ Bt/Bp ~ I_tf/Ip`). Held-out DIII-D improved to `R² = 0.97`/`0.95`; MAST transfer
   is unverified until the next score.

The scorer clamps negative component terms to zero, which was verified exactly against the
reported numbers (`0.10·(1−0.979969) = 0.002003` and `0.10·(1−0.425) = 0.0575`). Every MAST
point scored so far therefore came from the `D_LCFS` term alone, with `R²ψ`, `R²_qb` and
`Consistency` all contributing zero — which is why amplitude correctness, not shape
refinement, was the priority. The `β = 0.5` shrink follows from `R² = 2βρ − β²` under that
clamp: it stays positive for any shape-correlation `ρ > 0.25`, whereas `β = 1` scores zero
unless `ρ > 0.5`.

## Inference entrypoint (regenerate the submitted `.npz`)

```
.venv/bin/python code/run/06_build_submission.py --model "Conv Decoder V2"
.venv/bin/python code/run/07_validate_push.py --push
```

Pinned predictions commit: `6c8cb6488bca515b04d6bebbcc9cf01ef8c51cd2` on
`TariqJamil/fusion-eq-predictions` (private HF dataset repo, read-scoped token submitted via
`manifest.json`).

## Environment

`code/requirements-local.txt` (installed into `.venv` at the repo root, created with
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
