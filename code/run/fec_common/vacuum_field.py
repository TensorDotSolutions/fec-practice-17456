"""Vacuum (coil-only) poloidal flux via a Green's-function filament approximation.

psi_vacuum(R, Z, t) = sum_c I_c(t) * G(R, Z; R_c, Z_c) -- the field produced by the
shaping/field coils alone (no plasma current), computed from each shot's own coil
geometry (``coil_R``/``coil_Z``, constant per machine -- see dataset docs) and the
magnetics current columns named by ``coil_input_column``.

This is genuine shared physics: the *same* formula, applied to each machine's own
geometry, with zero per-machine fitting -- unlike the purely statistical Ip/Thomson
features in ``synthetic_features_for_shot``, so it is expected to transfer zero-shot far
better. The starter kit's own README notes that fitting psi outside the plasma envelope
to coil Green's functions alone reaches R^2 ~ 0.94 on DIII-D -- this module is that idea,
implemented, and reused here to give the Challenge-2 (MAST) model a physics-grounded
component instead of relying solely on 7 scalar stats.

Each coil is approximated as a single circular filament at its centroid (coil_R,
coil_Z), ignoring coil_width/height/angle1/angle2 (rectangular cross-section, and the
parallelogram skew on 6 DIII-D F-coils) -- a documented simplification: the filament
Green's function is a reasonable leading-order approximation to a compact coil's field
away from its own cross-section, which is the physically relevant regime at grid points
inside the vessel. Toroidal-field coils with no poloidal-plane rectangle (DIII-D
``bcoil``, MAST ``tf_current``/``efps_current``) never appear in ``coil_input_column``,
so they are naturally excluded, consistent with the dataset docs.

Only the *scale* of this field is fit (a single global least-squares scalar, on DIII-D
only -- see ``05_train_challenge2.py``); the geometry and Green's-function formula are
recomputed fresh per machine from that machine's own shipped ``coil_R``/``coil_Z``, so no
MAST-specific fitting happens anywhere in this module.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import ellipe, ellipk

MU0 = 4e-7 * np.pi

# MAST ships one coil_R/coil_Z row per physical conductor *turn* for these columns, but the
# current value itself is already turns-folded ampere-turns ("kA*turn", per the starter
# README's units table) -- summing the Green's function over all N filament rows and then
# multiplying by an already-N*I current double-counts turns (N^2 instead of N). These columns
# must be averaged across their filament rows, not summed. `magnetics_sol_current` is the
# opposite case (a true per-turn current shared by all 656 solenoid elements in series) and is
# deliberately excluded here -- summing its 656 filaments is the physically correct operation.
# DIII-D is unaffected: every DIII-D column has exactly one coil_R/coil_Z row, so sum == mean.
_AMPERE_TURN_ALREADY_FOLDED = {
    f"magnetics_p{n}{side}_current" for n in (2, 3, 4, 5, 6) for side in ("l", "u")
}


def _filament_green(R: np.ndarray, Z: np.ndarray, Rc: float, Zc: float) -> np.ndarray:
    """Green's function psi(R,Z) per ampere(-turn) of a circular filament at (Rc, Zc).

    Standard tokamak mutual-inductance form (e.g. Jardin, *Computational Methods in
    Plasma Physics*): psi = (mu0/2pi) sqrt(R Rc) [(2/k - k)K(k) - (2/k)E(k)],
    k^2 = 4 R Rc / ((R+Rc)^2 + (Z-Zc)^2). scipy's ellipk/ellipe take the parameter
    m = k^2, not the modulus k.
    """
    k2 = 4.0 * R * Rc / ((R + Rc) ** 2 + (Z - Zc) ** 2)
    k2 = np.clip(k2, 1e-12, 1.0 - 1e-12)
    k = np.sqrt(k2)
    K = ellipk(k2)
    E = ellipe(k2)
    # psi = 2 pi R A_phi with A_phi = (mu0 I / pi k) sqrt(Rc/R) [(1 - k^2/2) K - E], i.e.
    # psi = (mu0 / k) sqrt(R Rc) [(2 - k^2) K - 2 E]. An earlier version of this used a
    # (mu0 / 2pi) prefactor and was a factor of 4pi^2 low; harmless while the field was only
    # ever consumed RMS-normalized or behind a fitted scale, but wrong in absolute Wb.
    return MU0 * np.sqrt(R * Rc) * ((2.0 / k - k) * K - (2.0 / k) * E)


def build_column_green(row, grid_R: np.ndarray, grid_Z: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Per-machine Green's-function basis -- geometry is constant across shots (see
    dataset docs: ``coil_R``/``coil_Z`` do not vary shot to shot, unlike Thomson chords),
    so call this once per machine and cache the result.

    Returns ``(column_names, H)``, ``H`` shape ``(len(column_names), 65, 65)``: ``H[j]``
    is the Green's function of ALL coils that share magnetics current column
    ``column_names[j]``, summed (coils wired to the same current source add fields
    coherently -- e.g. MAST's 656 solenoid filaments all map to one current column).
    """
    coil_R = np.asarray(row["coil_R"], dtype=np.float64)
    coil_Z = np.asarray(row["coil_Z"], dtype=np.float64)
    coil_cols = [str(c) for c in np.asarray(row["coil_input_column"])]
    gR = np.asarray(grid_R, dtype=np.float64)
    gZ = np.asarray(grid_Z, dtype=np.float64)
    # ``efit_psirz`` is stored **[iZ, iR]**, not [iR, iZ) -- verified against the shipped
    # ``efit_r_axis``/``efit_z_axis``: argmin(psi) lands 0.017 m from the true magnetic axis
    # under this order (sub-grid-cell; dR = 0.027 m) vs 0.105 m under [iR, iZ]. The grid is
    # square (65x65) so a transpose raises no shape error, and DIII-D equilibria are nearly
    # up-down symmetric and grid-centred, so a transposed field still *looks* roughly right
    # and survives a full-grid fitted-scale R^2 check -- it has to be tested against the axis
    # position. Build the Green's functions in the target's own order so the two never
    # disagree.
    GZ, GR = np.meshgrid(gZ, gR, indexing="ij")  # both (nZ, nR)

    names = sorted(set(coil_cols))
    idx = {n: i for i, n in enumerate(names)}
    H = np.zeros((len(names), len(gZ), len(gR)), dtype=np.float64)
    counts = np.zeros(len(names), dtype=np.int64)
    for c in range(len(coil_R)):
        j = idx[coil_cols[c]]
        H[j] += _filament_green(GR, GZ, coil_R[c], coil_Z[c])
        counts[j] += 1
    for j, n in enumerate(names):
        if n in _AMPERE_TURN_ALREADY_FOLDED and counts[j] > 0:
            H[j] /= counts[j]
    return names, H.astype(np.float32)


def vacuum_psi_for_shot(row, efit_times: np.ndarray, names: list[str],
                        H: np.ndarray) -> np.ndarray:
    """psi_vacuum (T, 65, 65) at the shot's own efit_times, from its own coil currents.

    ``names``/``H`` come from ``build_column_green`` (same machine, cached once).
    Currents are read directly off the row by column name and interpolated onto
    efit_times; a column that doesn't exist on this row contributes zero.
    """
    mag_time = np.asarray(row.get("magnetics_time"), dtype=np.float64)
    T = len(efit_times)
    coeffs = np.zeros((T, len(names)), dtype=np.float32)
    for j, col in enumerate(names):
        if col not in row:
            continue
        vals = np.asarray(row[col], dtype=np.float64)
        n = min(len(mag_time), len(vals))
        if n < 2:
            continue
        ok = np.isfinite(mag_time[:n]) & np.isfinite(vals[:n])
        if ok.sum() < 2:
            continue
        # currents are kA(-turn) on both machines (see fec_common.features docstring on
        # the Ip unit fix) -> SI amperes for the Green's-function formula.
        f = interp1d(mag_time[:n][ok], vals[:n][ok] * 1000.0, kind="linear",
                    fill_value=0.0, bounds_error=False)
        coeffs[:, j] = f(efit_times).astype(np.float32)
    flat = H.reshape(len(names), -1)
    psi = coeffs @ flat
    return psi.reshape(T, H.shape[1], H.shape[2]).astype(np.float32)


_GEOM_CACHE: dict[str, tuple[list[str], np.ndarray]] = {}


def get_column_green(source: str, row, grid_R: np.ndarray, grid_Z: np.ndarray):
    """Lazily build + cache the per-machine Green's basis for inference (predict.py)."""
    if source not in _GEOM_CACHE:
        _GEOM_CACHE[source] = build_column_green(row, grid_R, grid_Z)
    return _GEOM_CACHE[source]
