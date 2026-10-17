"""Prediction wrappers matching the submission skeleton's ``your_model_predict`` contract.

``Predictor`` mirrors notebook cells 16/20 with two additions:

* Thomson profiles are always repaired before the features are built (see
  ``features.repair_thomson``) -- the stock ``load_shot_from_hf_row`` drops them on
  pandas rows, which is how the first full-scale run ended up with X width 21.
* A Challenge-2 scalar head (q95 / betaN RidgeCV on the shared features) is trained
  by ``05`` and used whenever present, instead of the zero scaffold.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from . import ch2_features
from . import vacuum_field as VF
from .config import ARTIFACTS, MODELS_DIR, load_json
from .features import build_frame, load_shot, pad_or_truncate, synthetic_features_for_shot


class TorchConvDecoderModel:
    """Wraps a PsiConvDecoder checkpoint so Predictor can detect it (isinstance check in
    challenge1_predict) and route around the PCA-inverse-transform / separate-scalar-model path
    that every joblib flux model uses -- the conv decoder outputs psi AND q95/betaN directly
    from one forward pass, there's no PCA coefficients or scalar model involved at all.
    """

    def __init__(self, checkpoint_path: Path):
        import torch

        from .conv_decoder import PsiConvDecoder, PsiConvDecoderV2

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # "arch" is missing on checkpoints saved before PsiConvDecoderV2 existed -- those are
        # all v1, so default to it rather than requiring every old checkpoint to be re-saved.
        model_cls = {"v1": PsiConvDecoder, "v2": PsiConvDecoderV2}[ckpt.get("arch", "v1")]
        self.model = model_cls(n_feat=int(ckpt["n_feat"]))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self._torch = torch

    def predict_full(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Xs: (T, n_feat) float32, already scaled. Returns (psi (T,65,65), q95 (T,), betaN (T,))."""
        torch = self._torch
        with torch.no_grad():
            X = torch.from_numpy(np.asarray(Xs, dtype=np.float32))
            psi, q95, beta_n = self.model(X)
        return (psi.numpy().astype(np.float32), q95.numpy().astype(np.float32),
                beta_n.numpy().astype(np.float32))


class Ch2TorchConvModel:
    """Wraps a ``ch2_conv.pt`` checkpoint (``Ch2PhysConvNet``, trained by
    ``train_ch2_conv_colab.py``) so ``Predictor.challenge2_predict`` can detect it and
    route around the RidgeCV/PCA bundle path that the ``ch2.joblib`` fallback uses.
    """

    ARCH = "ch2phys_sigma_qtf"

    def __init__(self, checkpoint_path: Path):
        import torch

        from .conv_decoder import Ch2PhysConvNet

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        arch = str(ckpt.get("arch", "?"))
        if arch != self.ARCH:
            # Pre-physics-amplitude checkpoints ("ch2phys") carry an amp_head and a
            # vacuum_scale and expect a transposed [iR, iZ] image stack; loading one now
            # would either raise an opaque state_dict error or silently mis-predict.
            raise RuntimeError(
                f"{checkpoint_path.name} has arch={arch!r}, expected {self.ARCH!r}. This "
                "checkpoint predates the [iZ, iR] orientation fix, the MAST coil-turn fix, "
                "the physics-amplitude rework, and/or the TF-coil q_proxy aux feature (aux "
                "dim 2 -> 3), so its weights are not usable. Retrain with "
                "code/train_ch2_conv.ipynb, or delete/rename the file to fall back to "
                "ch2.joblib."
            )
        self.model = Ch2PhysConvNet(in_channels=int(ckpt.get("in_channels", 5)))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.ip_norm = tuple(ckpt["ip_norm"])
        # amplitude is physics, not learned -- carried on the checkpoint so a submission is
        # always rebuilt with the same constants the weights were trained against.
        self.sigma_c = float(ckpt.get("sigma_c", ch2_features.CH2_SIGMA_C))
        self.amp_shrink = float(ckpt.get("amp_shrink", ch2_features.CH2_AMP_SHRINK))
        self._torch = torch

    def predict_full(self, images: np.ndarray, aux: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """images: (T, 5, 65, 65), aux: (T, 3) -- see ch2_features.build_ch2_images.
        Returns (shape, q95, betaN), all numpy. ``shape`` is psi/sigma, [iZ, iR]."""
        torch = self._torch
        with torch.no_grad():
            im = torch.from_numpy(np.asarray(images, dtype=np.float32))
            ax = torch.from_numpy(np.asarray(aux, dtype=np.float32))
            shape, q95, beta_n = self.model(im, ax)
        return (shape.numpy().astype(np.float32), q95.numpy().astype(np.float32),
                beta_n.numpy().astype(np.float32))


class Predictor:
    """Flux + scalar predictor for both machines."""

    def __init__(self, scaler, target_pca, flux_model, scalar_model, n_feat,
                 scalar_names, ch2=None, name: str = ""):
        self.scaler = scaler
        self.target_pca = target_pca
        self.flux_model = flux_model
        self.scalar_model = scalar_model
        self.n_feat = int(n_feat)
        self.scalar_names = list(scalar_names)
        self.ch2 = ch2  # dict: scaler, pca, model, scalar (Challenge-2 bundle)
        self.name = name

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_memory(cls, flux_model, scaler, target_pca, scalar_model, n_feat,
                    scalar_names=None, ch2=None, name: str = "") -> "Predictor":
        return cls(scaler, target_pca, flux_model, scalar_model, n_feat,
                   scalar_names or [], ch2=ch2, name=name or getattr(flux_model, "name", ""))

    @classmethod
    def from_artifacts(cls, model_name: str | None = None, artifacts: Path | None = None,
                       with_ch2: bool = True) -> "Predictor":
        artifacts = Path(artifacts or ARTIFACTS)
        features_dir = artifacts / "features"
        meta = load_json(features_dir / "manifest_meta.json") if (
            features_dir / "manifest_meta.json"
        ).exists() else load_json(features_dir / "meta.json")
        scaler = joblib.load(features_dir / "scaler.joblib")
        target_pca = joblib.load(features_dir / "pca.joblib")
        from experiments import EFIT_SCALAR_LABELS

        scalar_model = joblib.load(MODELS_DIR / "scalars.joblib")
        manifest = load_json(artifacts / "manifest.json")
        # Default to Ridge, not best_flux_model(manifest) -- that currently picks Random
        # Forest (val_pca_r2 0.76 vs Ridge's 0.40), but the RF joblib is ~20GB on disk and
        # has hung this machine before. Ridge loads in milliseconds. Pass --model explicitly
        # (06_build_submission.py --model "Random Forest") when you deliberately want it and
        # have the RAM headroom to spare.
        model_name = model_name or "Ridge (CV)"
        flux_model = _load_flux_model(model_name, manifest)
        ch2 = None
        if with_ch2 and (MODELS_DIR / "ch2_conv.pt").exists():
            # prefer the spatial conv model (train_ch2_conv_colab.py) over the flat-vector
            # RidgeCV fallback -- see ch2_features.py's module docstring for why.
            ch2 = Ch2TorchConvModel(MODELS_DIR / "ch2_conv.pt")
        elif with_ch2 and (MODELS_DIR / "ch2.joblib").exists():
            ch2 = joblib.load(MODELS_DIR / "ch2.joblib")
        return cls(scaler, target_pca, flux_model, scalar_model,
                   int(meta["n_feat"]), list(EFIT_SCALAR_LABELS), ch2=ch2, name=model_name)

    # -- Challenge 1 (DIII-D) ----------------------------------------------

    def challenge1_predict(self, row, source: str = "DIII-D") -> dict:
        """Coil currents + Thomson -> psi (65x65) + q95 + betaN, per efit timestep."""
        shot = load_shot(row)
        X, _, _ = build_frame(shot, include_thomson=True)
        T = len(X)
        X = pad_or_truncate(X, self.n_feat)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Xs = np.asarray(self.scaler.transform(X), dtype=np.float32)

        if isinstance(self.flux_model, TorchConvDecoderModel):
            # Conv decoder outputs psi AND q95/betaN directly from one forward pass -- no PCA
            # inverse_transform, no separate scalar model (see TorchConvDecoderModel docstring).
            psirz, q95, beta_n = self.flux_model.predict_full(Xs)
            return {"psirz": psirz.reshape(T, 65, 65), "q95": q95, "betaN": beta_n}

        psirz = self.target_pca.inverse_transform(self.flux_model.predict(Xs))
        scal = self._predict_scalars(Xs)

        return {
            "psirz": np.asarray(psirz, dtype=np.float32).reshape(T, 65, 65),
            "q95": scal[:, self.scalar_names.index("efit_q95")],
            "betaN": scal[:, self.scalar_names.index("efit_beta_n")],
        }

    # -- Challenge 2 (MAST, zero-shot) -------------------------------------

    # DIII-D's own training-data range (see 05_train_challenge2.py / features.py
    # comment on the Ip unit fix) with a generous margin -- a linear RidgeCV
    # transferred to MAST has no mechanism to saturate on out-of-distribution
    # input, so a clamp is cheap insurance against a repeat of the ~1000x Ip
    # unit mismatch that previously produced R2 in the hundreds-of-thousands
    # negative range on MAST.
    _CH2_PSI_CLAMP = (-1.5, 1.5)
    _CH2_Q95_CLAMP = (0.0, 40.0)
    _CH2_BETAN_CLAMP = (-1.0, 6.0)

    def challenge2_predict(self, row, source: str = "MAST") -> dict:
        """Shared Ip + Thomson pressure-proxy features fitted on DIII-D."""
        T = len(np.asarray(row["efit_times"]))
        if self.ch2 is None:
            return {
                "psirz": np.zeros((T, 65, 65), dtype=np.float32),
                "q95": np.zeros(T, dtype=np.float32),
                "betaN": np.zeros(T, dtype=np.float32),
            }
        try:
            if isinstance(self.ch2, Ch2TorchConvModel):
                # Spatial conv model (train_ch2_conv_colab.py) -- see ch2_features.py.
                # The net predicts only a SIGMA-NORMALIZED shape (psi/sigma); absolute
                # amplitude comes from the physics law sigma = C*|Ip|*R0 evaluated on THIS
                # machine's own Ip and R0, times a deliberate shrink. The previous learned
                # amplitude head had to extrapolate onto MAST and overshot ~7x, which is
                # what pinned the scored R2_psi at -25.9 (worse than predicting a constant).
                names, H = VF.get_column_green(source, row, row["efit_grid_R"], row["efit_grid_Z"])
                images, aux = ch2_features.build_ch2_images(
                    row, source, names, H, ip_norm=self.ch2.ip_norm)
                shape, q95_raw, beta_n_raw = self.ch2.predict_full(images[:T], aux[:T])
                efit_times = np.asarray(row["efit_times"], dtype=np.float64)
                sigma = ch2_features.physics_sigma(row, source, efit_times)[:T]
                psirz = (self.ch2.amp_shrink * sigma[:, None, None] * shape).astype(np.float32)
                # Clamp RELATIVE to each frame's own physics sigma, not the fixed +/-1.5 Wb
                # used by the RidgeCV path: MAST's sigma is ~0.038 Wb, so an absolute
                # 1.5 Wb bound is ~40 sigma and would never catch a blow-up on the machine
                # that actually needs catching. +/-8 sigma comfortably contains a real
                # equilibrium (DIII-D peaks at ~3.6 sigma) while still bounding outliers.
                lim = (ch2_features.CH2_SHAPE_CLIP * sigma[:, None, None]).astype(np.float32)
                psirz = np.clip(psirz, -lim, lim)
                q95 = np.clip(np.nan_to_num(q95_raw), *self._CH2_Q95_CLAMP).astype(np.float32)
                beta_n = np.clip(np.nan_to_num(beta_n_raw), *self._CH2_BETAN_CLAMP).astype(np.float32)
            else:
                feat = synthetic_features_for_shot(row, source)
                Xs = np.asarray(self.ch2["scaler"].transform(feat), dtype=np.float32)
                # shape (normalized-amplitude PCA basis) and amplitude are predicted by
                # two separate heads -- see 05_train_challenge2.py's docstring for why
                # that split transfers better zero-shot than fitting PCA on raw-scale
                # DIII-D psi directly. This predicts the RESIDUAL after the coil-only
                # vacuum field (below) is subtracted, not raw psi -- same reasoning.
                shape = self.ch2["pca"].inverse_transform(self.ch2["model"].predict(Xs))
                shape = np.asarray(shape, dtype=np.float32).reshape(T, 65, 65)
                if "amp_model" in self.ch2:
                    scale = np.exp(self.ch2["amp_model"].predict(Xs)).astype(np.float32)
                    resid_pred = shape * scale[:, None, None]
                else:
                    resid_pred = shape

                vacuum_scale = self.ch2.get("vacuum_scale")
                if vacuum_scale:
                    efit_times = np.asarray(row["efit_times"], dtype=np.float64)
                    names, H = VF.get_column_green(source, row, row["efit_grid_R"], row["efit_grid_Z"])
                    vac = VF.vacuum_psi_for_shot(row, efit_times, names, H)
                    psirz = vacuum_scale * vac[:T] + resid_pred
                else:
                    psirz = resid_pred
                psirz = np.clip(psirz, *self._CH2_PSI_CLAMP)
                if self.ch2.get("scalar") is not None:
                    scal = self.ch2["scalar"].predict(Xs)
                    q95 = np.clip(np.nan_to_num(scal[:, 0]), *self._CH2_Q95_CLAMP).astype(np.float32)
                    beta_n = np.clip(np.nan_to_num(scal[:, 1]), *self._CH2_BETAN_CLAMP).astype(np.float32)
                else:
                    q95 = np.zeros(T, dtype=np.float32)
                    beta_n = np.zeros(T, dtype=np.float32)
        except Exception as exc:  # never let one shot kill a submission build
            print(f"  challenge2_predict fallback to zeros for a {source} shot: {exc}")
            psirz = np.zeros((T, 65, 65), dtype=np.float32)
            q95 = np.zeros(T, dtype=np.float32)
            beta_n = np.zeros(T, dtype=np.float32)
        return {"psirz": psirz, "q95": q95, "betaN": beta_n}

    # -- dispatch -----------------------------------------------------------

    def full_predict(self, row, source: str) -> dict:
        return (
            self.challenge1_predict(row, source)
            if source == "DIII-D"
            else self.challenge2_predict(row, source)
        )

    def _predict_scalars(self, Xs: np.ndarray) -> np.ndarray:
        scal = self.scalar_model.predict(Xs)
        out = np.full((len(Xs), len(self.scalar_names)), np.nan, dtype=np.float32)
        for j, name in enumerate(self.scalar_names):
            model = getattr(self.scalar_model, "models", {}).get(name, None)
            if model is not None:
                out[:, j] = np.nan_to_num(scal[:, j]).astype(np.float32)
        return out


def _load_flux_model(model_name: str, manifest: dict):
    entry = manifest["flux_models"].get(model_name)
    if entry is None:
        raise SystemExit(f"No manifest entry for flux model '{model_name}'")
    # entry["path"] may be a bare filename (current) or a leftover absolute path from a
    # manifest written on a different machine (e.g. before a Drive-synced Colab run) -- take
    # just the filename and re-join against this machine's MODELS_DIR either way.
    path = MODELS_DIR / Path(entry["path"]).name
    if not path.exists():
        raise SystemExit(f"flux model file missing: {path} (re-run 02_train_flux.py?)")
    if path.suffix == ".pt":
        return TorchConvDecoderModel(path)
    return joblib.load(path)


_cached: Predictor | None = None


def get_predictor(model_name: str | None = None) -> Predictor:
    global _cached
    if _cached is None:
        _cached = Predictor.from_artifacts(model_name=model_name)
    return _cached
