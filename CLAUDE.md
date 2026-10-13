# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Personal workspace for the **Fusion Equilibrium Challenge** (NeurIPS competition, Codabench
#17456, run by Sophelio/General Atomics/UT Austin). The task: reconstruct a tokamak's 2-D
poloidal flux map ψ(R,Z) (65×65, per timestep) from coil currents + Thomson scattering alone
(no magnetic sensors), for DIII-D (Challenge 1) and zero-shot for MAST (Challenge 2). See
`docs/starter-kit-README.md` and `docs/starter-kit-MODELING_GUIDE.md` for the full problem
spec, data dictionary, and scoring formula — read them before making modeling decisions, they
contain many non-obvious data-correctness gotchas (see below).

This repo is **not** the official starter kit — it's a personal layer on top of it. It runs two
ways, both sharing the same logic:

- **Google Colab against Google Drive** (the original design): one generated notebook, cells
  sharing in-memory state. Good for smoke runs and plots; `N_SHOTS` stays small because
  `FusionDataPipeline.prepare()` materialises every frame in RAM (~50 GB for all 7,041 shots).
- **Local staged pipeline** (`code/run/00_setup.py` … `07_validate_push.py`): one script per
  step, sharing intermediate artifacts on disk (`.npy` memmaps, `.npz`, `.joblib`, `.json`).
  This is how a full-scale run is done — `01` streams shot-by-shot into memmaps so nothing
  bigger than one shot (~79 MB) is ever resident. A local venv lives at `code/.venv`
  (created with `--system-site-packages`, so numpy/pandas/pyarrow/sklearn/torch come from the
  system interpreter; only `datasets`, `huggingface_hub`, `python-dotenv`, `scikit-image` are
  installed into it — see `code/requirements-local.txt`).

Both call into **`code/run/fec_common/`**, which is the single source of truth for feature
engineering, the composite scorer wrapper, the Challenge-2 shared features and the submission
builder. Keep it that way: change `fec_common`, not a copy of it in a cell or a script.

## Repository layout

- `docs/` — vendored copies of the official starter kit's README and modeling guide, plus the
  challenge brief/getting-started PDFs. Reference material; not code that runs.
- `code/run/fec_common/` — the shared module (`config.py`, `features.py`, `models.py`,
  `predict.py`, `scoring.py`, `build.py`). `config.py` resolves every path from its own file
  location, so the same package works locally and in Colab (where `FEC_STARTER_DIR` /
  `FEC_DATA_ROOT` env vars can override).
- `code/run/00_setup.py` … `07_validate_push.py` — the staged pipeline (see below). Run with
  `python code/.venv/bin/python code/run/<script>.py`.
- `code/starter/` — the **pinned** official starter-kit clone
  (`Sophelio/fusion-equilibrium-challenge-starter` @ `a674291`). Provides `experiments.py`,
  `data_fixes.py`, `fusion_scoring/*`, `validate_submission.py`, `push_predictions.py` — used
  **unmodified**; `00_setup.py` creates/repairs the clone.
- `code/fusion-equilibrium-challenge/data/` — the full local dataset
  (`diii_d_train` 7041 / `diii_d_public_test` 874 / `mast_public_test` 1206 parquets, ~98 GB).
- `code/artifacts/` — pipeline outputs (gitignored): `features/` memmaps + `split.json` +
  `meta.json` + `pca.joblib` + `scaler.joblib`, `models/*.joblib`, `manifest.json`,
  `scores/local_score.json`, `leaderboard.csv`, `submission/*.npz`.
- `code/download_dataset.py` + `code/requirements.txt` — standalone local script to pull the
  dataset via `snapshot_download` (reads `HF_TOKEN` from a local `.env`). Insync-syncs to
  Google Drive, so this is also how data gets onto Drive for Colab — but the notebook can
  download it itself.
- `code/train_score_submit.ipynb` — the **Colab** workflow (smoke runs, plots). It is
  **generated, not hand-edited** — see below. Its cells import from `fec_common`, so it cannot
  drift from the scripts.
- `code/repo/notebook-build_notebook.py` — the generator for `train_score_submit.ipynb`. Builds
  the notebook cell-by-cell from Python strings (`md(...)` / `code(...)` helpers appending to a
  `cells` list, dumped to ipynb JSON at the end). **To change the notebook, edit this script and
  regenerate (`python code/repo/notebook-build_notebook.py`) — never hand-edit the `.ipynb`
  directly**, since a regeneration overwrites it.

History, so this doesn't get "simplified" back: an earlier version of this project had numbered
scripts (`1_setup_env.py` … `7_validate_and_push.py`) *plus* glue modules that persisted state
via `joblib`, and that was rightly removed — Colab cells already share state, and the extra
files added failure modes. The scripts are back now for a different reason: full-scale local
runs do not fit in RAM without the memmap hand-off, and a notebook is a poor driver for a
multi-hour job that must be resumable. The difference now is that all logic lives in
`fec_common/` and the scripts are thin.

## How the staged pipeline is organized

Run top to bottom, locally, with the venv:

| script | reads | writes |
|---|---|---|
| `00_setup.py` | — | pinned starter clone, `.env` template, dep/data/memory report (optional `--bump-swap`) |
| `01_extract_features.py` | `diii_d_train/*.parquet` | `features/{X,ypca,y_scalar}.f32.npy`, `frame_index.npz`, `split.json`, `meta.json`, `pca.joblib`, `scaler.joblib` |
| `02_train_flux.py` | the 01 cache | `models/flux_*.joblib`, `manifest.json` (`flux_models`) |
| `03_train_scalars.py` | the 01 cache | `models/scalars.joblib`, `manifest.json` (`scalars`) |
| `04_local_score.py` | 01 cache + held-out parquets | `scores/local_score.json`, `leaderboard.csv`, `manifest.json` (`val_score`) |
| `05_train_challenge2.py` | `diii_d_train` (column-projected) | `models/ch2.joblib` |
| `06_build_submission.py` | Hub stream (public test splits) | `submission/*.npz` |
| `07_validate_push.py` | `submission/*.npz` | official validator output; with `--push`, the upload zip |

01 runs three passes (scan frame counts/widths → fit PCA on a train-split subsample → fill
memmaps one shot at a time). The scaler and the PCA are fit on the **train split only** —
no val/test leakage, matching `FusionDataPipeline`.

## How the notebook is organized

Run top to bottom in Colab, once per runtime (`code/train_score_submit.ipynb`):

0. Mount Drive, clone the pinned starter-kit commit, `pip install` deps, load `.env` for
   `HF_TOKEN`, then **download the dataset** to Drive if it isn't there yet
   (`snapshot_download`, idempotent — skips if the expected shot counts are already on disk,
   resumes if interrupted).
1. Import the starter kit **and `code/run/fec_common/`** (the Drive copy), then train flux (ψ)
   baselines (Linear/RidgeCV/RandomForest/sklearn MLP on PCA targets) on a small `N_SHOTS`
   subsample of `diii_d_train`; optional PyTorch conv decoder (section 1b, wants a GPU).
2. Train the `q95`/`betaN` scalar heads (`ScalarTargetModels`, per-scalar `RidgeCV`).
3. Wrap the trained models into a `your_model_predict`-shaped `challenge1_predict(row, source)`
   function, matching `submission_skeleton.py`'s contract — built on `fec_common.load_shot` /
   `build_frame` / `pad_or_truncate`, i.e. **with the Thomson repair**.
4. Score Challenge 1 locally with the **real vendored composite metric**
   `S = 0.55·R²ψ + 0.15·R²{q95,βN} + 0.10·(1−D_LCFS) + 0.20·Consistency`, via
   `fec_common.scoring.compute_local_score` (which drives the unmodified `fusion_scoring/`).
5. Train a best-effort Challenge 2 (MAST, zero-shot) predictor from machine-agnostic features
   (Ip + Thomson pressure proxies + a RidgeCV scalar head) fit on DIII-D, since there is no
   `mast_train` at all.
6. Build both submission `.npz` files by **streaming** `diii_d_public_test`/`mast_public_test`
   from Hugging Face (not the local copy — this is what guarantees shot order matches what the
   scorer expects); validate with the vendored, unmodified `validate_submission.py`.
7. Push via the vendored, unmodified `push_predictions.py` to a private HF repo and get the
   `submission_pointer.zip` to upload to Codabench.

Each cell's state (models, scalers, PCA objects, predict functions) lives in notebook variables
and is reused by later cells — there's no serialization between steps, so cells must run in
order, but can be re-run individually once earlier variables exist in the kernel.

## Thomson erratum (the bug that caps the score)

`experiments.load_shot_from_hf_row` converts the Thomson profiles with
`np.asarray(row["thomson_core_Te"], dtype=np.float32)`. That works for HuggingFace rows
(lists of lists) but **raises** on the local parquet rows pandas hands back (an object array of
per-timestep 1-D arrays) — and the `except Exception: pass` around it silently drops **both**
Thomson systems. Any model trained through that path therefore sees only the 21 magnetics
channels, whatever `include_thomson` says. The first full-scale Colab run hit exactly this
(`X: (32662, 21)`), and its `S = 0.8468` sits just under the 0.85 Challenge-2 gate.

`fec_common.features.repair_thomson` (applied inside `load_shot`) fixes it with a ragged-safe
channel conversion, and `01` records `n_feat` in `meta.json` — with the repair a shot yields
`21 + 2·n_core + 2·n_edge ≈ 129–133` features, not 21. Assert on this after any feature-pipeline
change.

## Critical data-correctness gotchas (from the starter-kit docs — easy to silently get wrong)

- **DIII-D Ip time axis is wrong on ~69% of shots.** `magnetics_plasma_current_times` ships one
  shared, often-incorrect axis; use `data_fixes.fix_d3d_ip_times(row)` for the corrected per-shot
  axis, never a blanket time shift. Already applied in the notebook's Challenge-2 feature
  extraction and the vendored `experiments.py` baseline.
- **MAST `thomson_core_R` has 130 entries vs 131 Te/ne channels** (channel 0 uncalibrated/NaN) —
  use `data_fixes.align_mast_thomson_core(row)`.
- **`magnetics_dsep` is EFIT-derived, not an input** — it's computed from the target equilibrium
  itself and withheld on test splits; never feed it to a model. It's also not scored, and DIII-D's
  `dsep` (separatrix↔limiter clearance) and MAST's (divertor balance) are different physical
  quantities with different sign semantics.
- **DIII-D and MAST store `efit_psirz` with opposite sign conventions** (axis = min on DIII-D, max
  on MAST). A correctly-transferred cross-machine model can look like it failed with a large
  negative R² — check `R²(−ψ_pred)` before concluding transfer failed. The official scorer
  normalizes sign per machine for you; amplitude is not normalized.
- **Never interpolate/resample the target flux maps** — only resample inputs (magnetics, Thomson)
  onto `efit_times`. Never round/truncate/decimate ψ before submission (`np.round` alone wrecks
  ~35% of the MAST Consistency term) — cast to float16 instead.
- **Always split train/test by shot, not by timestep** — timesteps within one shot are highly
  correlated. `FusionDataPipeline` does this already.
- **Current units differ by column, not by one global factor** between DIII-D (A, kA·turn) and
  MAST (kA; MAST P-coils are kA·turn with per-coil turn counts in `coil_*` geometry) — see the
  README's "Cross-machine convention notes" before combining machines' magnetics features.
- Thomson channel count varies per shot (different probe configs) — both the notebook's
  `challenge1_predict` and `fec_common.predict` call `pad_or_truncate` to the width in
  `meta.json`; don't assume a fixed feature width when extending the feature pipeline.
- **Local scoring needs `scikit-image`** — `fusion_scoring/lcfs.py` imports it for the contour
  extraction behind D_LCFS/Consistency (the scoring docker pins `0.25.2`, which is what
  `requirements-local.txt` pins too). Without it `04` dies with an ImportError from the scorer.
- **`04_local_score.py` excludes the train split from its holdout.** The notebook's seed-999
  holdout can overlap the seed-42 training subset once `N_SHOTS` grows (and must, at full
  scale), which inflates `S`. Keep any new scoring path doing the same exclusion.

## Credentials

Two different Hugging Face tokens are involved and must not be confused: a **write** token for
`huggingface-cli login` (uploads predictions, stays local, never submitted) and a **fine-grained
read-only** token scoped to your private predictions repo (goes inside the submission zip,
passed to `push_predictions.py --read-token`). The starter-kit scripts actively refuse a
write/classic-read token in the read-token slot. `.env` at the Drive-synced `UPW_DIR` (or
`code/` locally) holds `HF_TOKEN`, used both by `download_dataset.py` and the notebook's own
dataset-download cell. `07_validate_push.py --push` reads the *read* token from `--read-token`
or `HF_READ_TOKEN`; the *write* token comes from `huggingface-cli login`, and the dataset
itself is public, so `06` works with no token at all (warns only).
