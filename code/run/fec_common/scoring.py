"""Local scorer -- the vendored ``fusion_scoring`` package driven by our own predictors.

Ported from notebook cell 18.  The scorer itself is **not** reimplemented: it is the
official vendored code (``fusion_scoring/{common,contour,derive,lcfs,metrics}``), so
what we print here is the same formula Codabench runs:

    S = 0.55*R2_psi + 0.15*R2_{q95,betaN} + 0.10*(1 - D_LCFS) + 0.20*Consistency

Only the shot loader is ours, and it now excludes every shot the model was trained
on (the notebook scored on a seed-999 holdout that could overlap the seed-42 training
subset once N_SHOTS grew).
"""
from __future__ import annotations

import numpy as np

from .config import DATA_ROOT, FUSION_SCORING_DIR, ensure_starter, log

ensure_starter()

from common import (  # noqa: E402
    AXIS_SIGN,
    CONS_SCALARS,
    N_CONS,
    N_SCALARS,
    PSI_SIGNS,
    SCALARS,
    SCORING_VERSION,
)
from contour import symmetric_hausdorff  # noqa: E402
from derive import derive_frame  # noqa: E402
from lcfs import extract_lcfs, extract_lcfs_with_sign, major_radius  # noqa: E402
from metrics import Accum, finalize_machine  # noqa: E402

from .features import _as_psirz_stack, read_parquet_row  # noqa: E402

MACHINE = "DIII-D"
N_POINTS = 512
N_ITER = 22
LI_IDX = CONS_SCALARS.index("li")


def load_scoring_shots(n_shots: int, seed: int = 999, exclude: set | list | None = None):
    """Held-out DIII-D shots with ground truth, straight from the local parquet copy.

    ``exclude`` is a set of absolute file paths the model has already seen (the train
    split), so the holdout is genuinely unseen at any training scale.
    """
    files = sorted((DATA_ROOT / "diii_d_train").glob("*.parquet"))
    exclude = {str(p) for p in (exclude or set())}
    pool = [f for f in files if str(f) not in exclude]
    if not pool:
        raise SystemExit("load_scoring_shots: every training shot was excluded")
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(pool), size=min(n_shots, len(pool)), replace=False)
    shots = []
    for idx in sorted(chosen):
        row = read_parquet_row(pool[idx])
        shots.append(
            {
                "file": str(pool[idx]),
                "row": row,
                "psi": _as_psirz_stack(row["efit_psirz"]),
                "q95": np.asarray(row["efit_q95"], dtype=np.float64),
                "betaN": np.asarray(row["efit_beta_n"], dtype=np.float64),
            }
        )
    log(f"  scoring holdout: {len(shots)} shots from {len(pool)} candidates (seed={seed})")
    return shots


def build_reference(psi_gt, R, Z, mask_coarse, mask_f):
    T = psi_gt.shape[0]
    contours, rgeos = [], []
    cons = np.full((T, N_CONS), np.nan)
    for k in range(T):
        c = extract_lcfs(psi_gt[k], R, Z, MACHINE, mask_coarse, mask_f, n_points=N_POINTS)
        contours.append(c)
        if c is not None:
            rgeos.append(major_radius(c))
        vals = derive_frame(psi_gt[k], R, Z, MACHINE, mask_coarse, mask_f, contour=c)
        for j, name in enumerate(CONS_SCALARS):
            cons[k, j] = vals[name]
    return contours, cons, np.isfinite(cons), (float(np.mean(rgeos)) if rgeos else float("nan"))


def psi_residuals(psi_gt, pred, mean_psi):
    g = psi_gt.astype(np.float64)
    p = np.asarray(pred, dtype=np.float64)
    if p.shape != g.shape:
        r = g - mean_psi
        ss = float(np.sum(r * r))
        return {s: ss for s in PSI_SIGNS}, 0, True
    nf = ~np.isfinite(p)
    out = {}
    for s in PSI_SIGNS:
        q = np.where(nf, mean_psi, s * p) if nf.any() else s * p
        r = g - q
        out[s] = float(np.sum(r * r))
    return out, int(nf.sum()), False


def score_shot(gt, ref, pred, R, Z, mask_coarse, mask_f, psi_sign, means):
    mean_psi, mean_scal, mean_cons = means
    contours, cons_gt, cmask, rgeo = ref
    psi_gt = gt["psi"]
    T = psi_gt.shape[0]
    axis_sign = AXIS_SIGN[MACHINE] * psi_sign

    ss_res_psi, nonfinite, missing = psi_residuals(psi_gt, pred["psirz"], mean_psi)
    ss_tot_shot = float(np.sum((psi_gt.astype(np.float64) - mean_psi) ** 2))
    shot_r2 = {s: (1.0 - ss_res_psi[s] / ss_tot_shot if ss_tot_shot > 0 else 0.0) for s in PSI_SIGNS}

    ss_res_scal = np.zeros(N_SCALARS)
    for j, name in enumerate(SCALARS):
        g = np.asarray(gt[name], dtype=np.float64)
        defined = np.isfinite(g)
        if not defined.any():
            continue
        p = np.asarray(pred.get(name), dtype=np.float64) if pred.get(name) is not None else None
        if p is None or p.shape != (T,):
            ss_res_scal[j] = float(np.sum((g[defined] - mean_scal[j]) ** 2))
        else:
            pj = np.where(np.isfinite(p[defined]), p[defined], mean_scal[j])
            ss_res_scal[j] = float(np.sum((g[defined] - pj) ** 2))

    ss_res_cons = np.zeros(N_CONS)
    cons_frames = np.zeros(N_CONS, dtype=np.int64)
    cons_fails = np.zeros(N_CONS, dtype=np.int64)
    ds, n_fail, n_frames = [], 0, 0
    ref_dropped = sum(1 for c in contours if c is None)

    if missing:
        n_frames = T - ref_dropped
        n_fail = n_frames
        ds = [1.0] * n_frames
        for j in range(N_CONS):
            m = cmask[:, j]
            if m.any():
                ss_res_cons[j] = float(np.sum((cons_gt[m, j] - mean_cons[j]) ** 2))
                cons_frames[j] = cons_fails[j] = int(m.sum())
    else:
        psi_pred = np.asarray(pred["psirz"], dtype=np.float64)
        for k in range(T):
            ref_c, need = contours[k], cmask[k]
            if ref_c is None and not need.any():
                continue
            ex, _ = extract_lcfs_with_sign(
                psi_pred[k], R, Z, MACHINE, mask_coarse, mask_f, n_iter=N_ITER, axis_sign=axis_sign
            )
            if ref_c is not None:
                n_frames += 1
                if ex is None:
                    n_fail += 1
                    ds.append(1.0)
                else:
                    d = symmetric_hausdorff(ex, np.asarray(ref_c, dtype=np.float64))
                    ds.append(min(1.0, d / rgeo if rgeo > 0 else 1.0))
            if need.any():
                vals = derive_frame(
                    psi_pred[k], R, Z, MACHINE, mask_coarse, mask_f, contour=ex,
                    with_li=bool(need[LI_IDX]), axis_sign=axis_sign,
                )
                for j in np.nonzero(need)[0]:
                    v = vals[CONS_SCALARS[j]]
                    cons_frames[j] += 1
                    if not np.isfinite(v):
                        v = mean_cons[j]
                        cons_fails[j] += 1
                    dv = cons_gt[k, j] - v
                    ss_res_cons[j] += dv * dv

    return {
        "ss_res_psi": ss_res_psi, "psi_pixels": int(psi_gt.size), "psi_nonfinite": nonfinite,
        "ss_res_scal": ss_res_scal, "ss_res_cons": ss_res_cons,
        "cons_frames": cons_frames, "cons_fails": cons_fails,
        "shot_r2_psi": shot_r2, "shot_dlcfs": (float(np.mean(ds)) if ds else None),
        "lcfs_frames": n_frames, "lcfs_fails": n_fail, "ref_lcfs_dropped": ref_dropped,
        "missing": missing,
    }


def compute_local_score(shots, predict_fn, quiet: bool = False) -> dict:
    mask = np.load(FUSION_SCORING_DIR / "masks" / "d3d_envelope.npz")
    R, Z = mask["grid_R"], mask["grid_Z"]
    mask_coarse = mask["mask_coarse"].astype(bool)
    mask_f = mask_coarse.astype(np.float64)

    refs = []
    psi_sum = psi_sumsq = psi_n = 0.0
    scal_sum = np.zeros(N_SCALARS); scal_sumsq = np.zeros(N_SCALARS); scal_n = np.zeros(N_SCALARS)
    cons_sum = np.zeros(N_CONS); cons_sumsq = np.zeros(N_CONS); cons_n = np.zeros(N_CONS)
    for s in shots:
        ref = build_reference(s["psi"], R, Z, mask_coarse, mask_f)
        refs.append(ref)
        g = s["psi"].astype(np.float64)
        fin = np.isfinite(g)
        psi_sum += float(g[fin].sum()); psi_sumsq += float((g[fin] ** 2).sum()); psi_n += int(fin.sum())
        for j, name in enumerate(SCALARS):
            v = np.asarray(s[name], dtype=np.float64)
            v = v[np.isfinite(v)]
            scal_sum[j] += v.sum(); scal_sumsq[j] += (v ** 2).sum(); scal_n[j] += v.size
        cons_gt, cmask = ref[1], ref[2]
        for j in range(N_CONS):
            v = cons_gt[cmask[:, j], j]
            cons_sum[j] += v.sum(); cons_sumsq[j] += (v ** 2).sum(); cons_n[j] += v.size

    ref_stats = {
        "psi_sum": psi_sum, "psi_sumsq": psi_sumsq, "psi_n": psi_n,
        "scal_sum": scal_sum, "scal_sumsq": scal_sumsq, "scal_n": scal_n,
        "cons_sum": cons_sum, "cons_sumsq": cons_sumsq, "cons_n": cons_n,
    }
    mean_psi = psi_sum / psi_n if psi_n else 0.0
    means = (
        mean_psi,
        np.where(scal_n > 0, scal_sum / np.where(scal_n > 0, scal_n, 1), 0.0),
        np.where(cons_n > 0, cons_sum / np.where(cons_n > 0, cons_n, 1), 0.0),
    )

    preds = [predict_fn(s["row"], MACHINE) for s in shots]

    totals = {s: 0.0 for s in PSI_SIGNS}
    for s_, p in zip(shots, preds):
        rr, _, _ = psi_residuals(s_["psi"], p["psirz"], mean_psi)
        for sg in PSI_SIGNS:
            totals[sg] += rr[sg]
    psi_sign = min(PSI_SIGNS, key=lambda sg: totals[sg])

    acc = Accum()
    acc.psi_sign = psi_sign
    for s_, ref, p in zip(shots, refs, preds):
        acc.add(score_shot(s_, ref, p, R, Z, mask_coarse, mask_f, psi_sign, means))

    res = finalize_machine(acc, ref_stats)
    res["scoring_version"] = SCORING_VERSION
    res["n_shots"] = len(shots)
    res["shot_files"] = [s["file"] for s in shots]
    if not quiet:
        print(
            f"COMPOSITE S = {res['S']:.4f}   "
            f"({len(shots)} held-out {MACHINE} shots, psi_sign={res['psi_sign']:+d})"
        )
        for label, key, w in [
            ("R2_psi", "r2_psi", 0.55),
            ("R2_{q95,betaN}", "r2_qb", 0.15),
            ("1 - D_LCFS", "dlcfs", 0.10),
            ("Consistency", "consistency", 0.20),
        ]:
            v = (1.0 - min(1.0, res["dlcfs"])) if key == "dlcfs" else res[key]
            print(f"  {label:>16s}  {v:8.4f}   x {w:.2f}  =  {w * max(0.0, v):.4f}")
    return res


# ---------------------------------------------------------------------------
# Cheap component-only scorer (no LCFS contour work) -- used for model ranking
# ---------------------------------------------------------------------------

def composite_from_predictions(Y_true, Y_pred, S_true, S_pred) -> dict:
    """R2(psi) + R2(q95, betaN) on already-predicted arrays; D_LCFS/Consistency need
    contour extraction and are left as None (see ``compute_local_score`` for those)."""
    from sklearn.metrics import r2_score

    Y_true = np.asarray(Y_true, dtype=np.float64)
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    sign = AXIS_SIGN[MACHINE]
    r2_psi = float(
        r2_score(Y_true.reshape(len(Y_true), -1), (sign * Y_pred).reshape(len(Y_pred), -1))
    )
    idx = [SCALARS.index("q95"), SCALARS.index("betaN")] if "q95" in SCALARS else [0, 1]
    r2_qb = float(r2_score(S_true[:, idx], S_pred[:, idx]))
    return {"r2_psi": r2_psi, "r2_qb": r2_qb, "dlcfs": None, "consistency": None, "S": None}
