"""Machine-agnostic SPATIAL (image) feature construction for the Challenge-2 conv model.

The existing Challenge-2 model (``05_train_challenge2.py``) is a RidgeCV on 7 flat scalar
stats (Ip + Thomson pressure-proxy mean/std/max per system) -- a linear model with no
spatial structure at all, predicting a 65x65 flux map through a 30-component PCA basis.
Both the full-grid vacuum-field check and the actual Codabench score (S_MAST ~ 0.057,
Consistency pinned at 0.0) show that representation has run out of road.

This module builds a fixed-channel-count *image* stack per efit timestep instead, so a
conv net (``conv_decoder.Ch2PhysConvNet``) can see real 2-D structure. Every channel is
re-derived on the fly from each machine's own geometry onto the *same* 65x65 grid, so the
channel count stays fixed regardless of DIII-D having 19 coils vs MAST's 812, or DIII-D's
2 Thomson systems having different chord layouts than MAST's:

  0. vacuum_psi    -- coil-only Green's-function flux map (``vacuum_field.py``), the one
                      genuinely shared-physics channel: same formula, each machine's own
                      coil geometry and shipped currents, zero MAST-specific fitting.
                      Self-normalized by its own per-frame RMS.
  1. R_norm        -- this shot's own ``efit_grid_R``, min-max normalized to [-1, 1],
                      broadcast along Z (CoordConv-style positional channel).
  2. Z_norm        -- as above, for ``efit_grid_Z``.
  3. core_pressure -- core-Thomson Te*ne, Gaussian-kernel-smoothed from each channel's own
                      radius (``data_fixes.align_mast_thomson_core``, which also handles
                      DIII-D unchanged) onto ``grid_R``, broadcast along Z -- core Thomson
                      is a near-midplane radial profile on both machines. Self-normalized.
  4. Ip_broadcast  -- plasma current interpolated to efit_times, z-scored with a fixed
                      constant fit on DIII-D only (``CH2_IP_NORM``), broadcast across the
                      grid.

All normalization is either self-normalizing (per-frame RMS, divides by a quantity
derived only from that frame's own data) or uses a constant fit on DIII-D and reused
unchanged on MAST -- no MAST-derived statistic anywhere, consistent with competition
rule 3.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from . import vacuum_field as VF
from . import features as F
from data_fixes import align_mast_thomson_core  # noqa: E402  (sys.path set up by `F` import)

N_CHANNELS = 5
_EPS = 1e-6

# (mean, std) of DIII-D plasma current in kA, fit once on the training set -- see
# train_ch2_conv_colab.py, which overwrites this via the saved checkpoint's own copy
# rather than mutating this module global.
CH2_IP_NORM = (1200.0, 450.0)

# --- physics amplitude law -------------------------------------------------------------
# std(psi) = CH2_SIGMA_C * |Ip[kA]| * R0[m], fit on DIII-D ONLY: C = 6.003e-5 with
# R2 = 0.915 (corr +0.96) over 761 frames. This replaces the old learned log-amplitude
# head, which had to EXTRAPOLATE its DIII-D-fit input distribution onto MAST and blew the
# amplitude up ~7x (measured max|psi| 1.12 Wb against a physics-implied 0.156 Wb, which is
# exactly the k ~ 6 overshoot implied by the scored R2_psi = -25.9). Amplitude now comes
# from dimensional analysis instead -- psi ~ mu0 Ip R0 -- evaluated on each machine's OWN
# Ip and R0, so nothing MAST-derived enters (rule 3).
CH2_SIGMA_C = 6.003e-05

# R0 proxy: midpoint of the Thomson-core chord radii. Chosen because it is an INPUT on both
# machines (DIII-D's vertical core laser pins it at ~1.94 m; MAST's horizontal midplane
# laser spans the plasma, midpoint ~0.876 m) -- unlike efit_r_axis, which is a withheld
# target quantity on the test splits. Refitting the law with this proxy instead of r_axis
# leaves R2 unchanged (0.9148 vs 0.9137), i.e. the law is insensitive to the choice.
#
# Deliberate amplitude shrink. With pred = beta*sigma*shape and shape-correlation rho,
# R2 = 2*beta*rho - beta^2, and the scorer CLAMPS negative terms to 0. beta = 0.5 stays
# positive for every rho > 0.25 and is near-optimal over rho in [0.5, 0.9], whereas beta = 1
# scores 0 unless rho > 0.5. Overshoot is unbounded-bad, undershoot is capped -- so shrink.
CH2_AMP_SHRINK = 0.5

# Floor on |Ip| (kA) inside the sigma law. The law was FIT only on frames with |Ip| > 100 kA,
# because during ramp-up/ramp-down there is no real equilibrium: Ip -> 0 while psi stays
# finite (coil fields dominate), so sigma -> 0 and psi/sigma diverges. Flooring sigma itself
# at a tiny epsilon is not enough -- it left the training target with std 82 instead of ~1,
# a handful of ramp frames dominating the entire MSE. Floor the DRIVER instead, which keeps
# sigma on the machine's own physical scale (DIII-D: 0.0116 Wb; MAST: 0.0053 Wb).
CH2_IP_FLOOR = 100.0

# Clip on the sigma-normalized target/prediction. A real equilibrium peaks near 3.6 sigma on
# DIII-D, so 8 sigma is loose enough to never touch a good frame while bounding the residual
# ramp-up outliers the Ip floor does not fully remove. Mirrored by the inference clamp.
CH2_SHAPE_CLIP = 8.0


def ip_on_efit_times(row, source: str, efit_times: np.ndarray) -> np.ndarray:
    """Plasma current (kA on both machines) interpolated onto ``efit_times``."""
    ip = np.asarray(row["magnetics_plasma_current"], dtype=np.float64)
    if source == "DIII-D":
        # the shipped Ip time axis is wrong on ~69% of DIII-D shots
        ip_times = np.asarray(F.fix_d3d_ip_times(row), dtype=np.float64)
    else:
        ip_times = np.asarray(row["magnetics_time"], dtype=np.float64)
    n = min(len(ip_times), len(ip))
    ip_times, ip = ip_times[:n], ip[:n]
    ok = np.isfinite(ip_times) & np.isfinite(ip)
    if ok.sum() < 2:
        return np.zeros(len(efit_times), dtype=np.float64)
    return interp1d(ip_times[ok], ip[ok], kind="linear", fill_value="extrapolate",
                    bounds_error=False)(efit_times)


def tf_on_efit_times(row, source: str, efit_times: np.ndarray) -> np.ndarray:
    """Toroidal-field coil current (kA) interpolated onto ``efit_times``.

    ``magnetics_bcoil`` (DIII-D) / ``magnetics_tf_current`` (MAST) -- an input on both
    machines, not part of the poloidal-flux Green's basis (the TF coil has no in-plane
    rectangle). Used only to build ``q_proxy`` below; never fed into the vacuum-field sum.
    """
    col = "magnetics_bcoil" if source == "DIII-D" else "magnetics_tf_current"
    tf = np.asarray(row[col], dtype=np.float64)
    tf_times = np.asarray(row["magnetics_time"], dtype=np.float64)
    n = min(len(tf_times), len(tf))
    tf_times, tf = tf_times[:n], tf[:n]
    ok = np.isfinite(tf_times) & np.isfinite(tf)
    if ok.sum() < 2:
        return np.zeros(len(efit_times), dtype=np.float64)
    return interp1d(tf_times[ok], tf[ok], kind="linear", fill_value="extrapolate",
                    bounds_error=False)(efit_times)


def machine_r0(row) -> float:
    """Plasma-scale radius proxy (m) from this shot's own Thomson-core chord radii."""
    try:
        R, _, _ = align_mast_thomson_core(row)
    except Exception:
        return np.nan
    R = np.asarray(R, dtype=np.float64)
    R = R[np.isfinite(R) & (R > 0)]
    return float(0.5 * (R.min() + R.max())) if len(R) else np.nan


def physics_sigma(row, source: str, efit_times: np.ndarray,
                  ip: np.ndarray | None = None) -> np.ndarray:
    """(T,) predicted per-frame ``std(psi)`` in Wb from the DIII-D-fit physics law.

    Used as the amplitude of the Challenge-2 prediction on BOTH machines, so the network
    only ever has to learn a scale-free shape (``psi / sigma``) -- the part that transfers.
    """
    if ip is None:
        ip = ip_on_efit_times(row, source, efit_times)
    r0 = machine_r0(row)
    if not np.isfinite(r0):
        r0 = float(np.median(np.asarray(row["efit_grid_R"], dtype=np.float64)))
    return CH2_SIGMA_C * np.maximum(np.abs(ip), CH2_IP_FLOOR) * r0


def _rms_norm(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # upcast before squaring -- raw electron pressure (Te*ne) runs ~1e21-1e24, which
    # overflows float32 once squared (float32 max ~3.4e38); float64 has headroom.
    x64 = x.astype(np.float64)
    scale = np.maximum(np.sqrt(np.mean(x64.reshape(len(x64), -1) ** 2, axis=1)), _EPS)
    return (x64 / scale[:, None, None]).astype(np.float32), scale


def _pressure_grid_map(row, efit_times: np.ndarray, grid_R: np.ndarray,
                       bandwidth: float = 0.08) -> np.ndarray:
    """(T, 65) core-Thomson radial pressure profile, Gaussian-smoothed onto grid_R."""
    try:
        R, Te, ne = align_mast_thomson_core(row)
    except Exception:
        return np.zeros((len(efit_times), len(grid_R)), dtype=np.float32)
    # scale down from raw SI (Te ~ eV-keV, ne ~ 1e19-1e20 m^-3, so Te*ne ~ 1e21-1e24) for
    # numerical conditioning -- an arbitrary constant, fine since this channel is
    # RMS-normalized again immediately after in build_ch2_images.
    pe = (Te.astype(np.float64) * ne.astype(np.float64)) * 1e-20  # (T_th, C)
    t_th = np.asarray(row["thomson_core_times"], dtype=np.float64)
    n = min(len(t_th), len(pe))
    t_th, pe = t_th[:n], pe[:n]
    if n < 2:
        return np.zeros((len(efit_times), len(grid_R)), dtype=np.float32)

    d2 = (grid_R[None, :] - R[:, None]) ** 2  # (C, 65)
    w = np.exp(-d2 / (2 * bandwidth ** 2))  # (C, 65)
    valid = np.isfinite(pe)
    pe0 = np.nan_to_num(pe, nan=0.0)
    num = pe0 @ w  # (T_th, 65)
    den = valid.astype(np.float64) @ w + _EPS
    profile_th = num / den

    f = interp1d(t_th, profile_th, axis=0, kind="linear", fill_value=0.0, bounds_error=False)
    return f(efit_times).astype(np.float32)


def build_ch2_images(row, source: str, vacuum_names: list[str], vacuum_H: np.ndarray,
                     ip_norm: tuple[float, float] = CH2_IP_NORM
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Returns ``(images, aux)``: images (T, 5, 65, 65) float32, aux (T, 3) float32 =
    [log(vacuum_rms/sigma), ip_z, log(|I_tf|/|Ip|)] -- small scalar side-features fed to
    the q95/betaN heads alongside the pooled conv features (see
    ``conv_decoder.Ch2PhysConvNet``).
    """
    efit_times = np.asarray(row["efit_times"], dtype=np.float64)
    grid_R = np.asarray(row["efit_grid_R"], dtype=np.float64)
    grid_Z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    T = len(efit_times)

    vac = VF.vacuum_psi_for_shot(row, efit_times, vacuum_names, vacuum_H)
    vac_norm, vac_scale = _rms_norm(vac)

    # Images are laid out [iZ, iR] to match efit_psirz's own storage order (see the note in
    # vacuum_field.build_column_green -- a transpose here is silent on a square grid and
    # misaligns every channel against both the target and the vacuum-field channel).
    r_norm = 2 * (grid_R - grid_R.min()) / max(grid_R.max() - grid_R.min(), _EPS) - 1
    z_norm = 2 * (grid_Z - grid_Z.min()) / max(grid_Z.max() - grid_Z.min(), _EPS) - 1
    R_chan = np.broadcast_to(r_norm[None, :].astype(np.float32), (65, 65))  # varies along iR
    Z_chan = np.broadcast_to(z_norm[:, None].astype(np.float32), (65, 65))  # varies along iZ

    pressure_1d = _pressure_grid_map(row, efit_times, grid_R)  # (T, 65) indexed by grid_R
    pressure_2d = np.ascontiguousarray(np.broadcast_to(pressure_1d[:, None, :], (T, 65, 65)))
    pressure_norm, _ = _rms_norm(pressure_2d)

    ip_feat = ip_on_efit_times(row, source, efit_times)
    ip_z = (ip_feat - ip_norm[0]) / ip_norm[1]
    Ip_chan = np.broadcast_to(ip_z[:, None, None].astype(np.float32), (T, 65, 65))

    images = np.stack([
        vac_norm,
        np.ascontiguousarray(np.broadcast_to(R_chan, (T, 65, 65))),
        np.ascontiguousarray(np.broadcast_to(Z_chan, (T, 65, 65))),
        pressure_norm,
        np.ascontiguousarray(Ip_chan),
    ], axis=1).astype(np.float32)  # (T, 5, 65, 65)

    # aux feeds the q95/betaN heads. Both entries are deliberately DIMENSIONLESS so they
    # occupy the same range on both machines: a raw log(vacuum_rms) is a machine-scale
    # quantity (MAST's coil field magnitude is nothing like DIII-D's), and feeding that to a
    # DIII-D-fit head is the same extrapolation trap that wrecked the old amplitude head --
    # MAST scored q95 R2 = -2.9 and betaN R2 = -9.1. Dividing by the physics sigma turns it
    # into "coil field strength relative to the expected flux scale", which is comparable.
    sigma = physics_sigma(row, source, efit_times, ip=ip_feat)
    vac_rel = np.log(np.maximum(vac_scale, _EPS) / sigma).astype(np.float32)

    # q_proxy: log(|TF coil current| / |Ip|), a dimensionless cross-machine proxy for the
    # edge safety factor (q ~ Bt/Bp ~ I_tf/Ip up to a machine-specific geometric constant the
    # head can absorb). Added because the learned q95/betaN heads, fed only pooled shape
    # features + vac_rel/ip_z, had no direct physics handle on q95 and transferred to MAST
    # with R2 = -0.27 (vs 0.88-0.92 on held-out DIII-D) -- a much smaller gap than the old
    # amplitude-head blowup, but still negative (clamped to 0 by the scorer either way).
    tf_feat = tf_on_efit_times(row, source, efit_times)
    q_proxy = np.log(np.maximum(np.abs(tf_feat), _EPS)
                     / np.maximum(np.abs(ip_feat), CH2_IP_FLOOR)).astype(np.float32)

    aux = np.column_stack([vac_rel, ip_z.astype(np.float32), q_proxy])
    return np.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0), np.nan_to_num(
        aux, nan=0.0, posinf=0.0, neginf=0.0)
