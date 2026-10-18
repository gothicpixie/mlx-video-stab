"""Per-frame motion from dense optical flow: robust global homography + spatially varying (mesh / per-row) residual.

Everything here consumes flow *samples* (a coarse grid of the dense flow), so it is cheap and testable without a flow net."""
from dataclasses import dataclass
import numpy as np
import mlx.core as mx
from .geometry import apply_h, ransac_homography, dlt, fit_model
from .warp import sample


# ``_fill`` diffuses under-observed mesh vertices / row bands from their observed neighbours. A location is
# *directly supported* when its support mass is at least this value and *filled* when it is below it.
SUPPORT_THRESHOLD = 3.0

# Stable, ordered fallback reasons for ``fit_motion``. ``NONE`` means the fit succeeded; the other three name
# the *first* gate that refused it, in the precedence order documented by :func:`fallback_reason`. These are
# diagnostics only and never change a fit or render.
FALLBACK_REASON_NONE = "none"
FALLBACK_REASON_INSUFFICIENT_VALID_SAMPLES = "insufficient_valid_samples"
FALLBACK_REASON_INSUFFICIENT_RANSAC_INLIERS = "insufficient_ransac_inliers"
FALLBACK_REASON_LOW_RANSAC_INLIER_FRACTION = "low_ransac_inlier_fraction"
# The cut itself is a deliberate observation boundary, not a failed geometric fit. It is recorded on the
# placeholder pair so the motion payload keeps one entry per analysed pair while plan() resets the path there.
FALLBACK_REASON_SEQUENCE_BOUNDARY = "sequence_boundary"
FALLBACK_REASONS = (FALLBACK_REASON_NONE, FALLBACK_REASON_INSUFFICIENT_VALID_SAMPLES,
                    FALLBACK_REASON_INSUFFICIENT_RANSAC_INLIERS, FALLBACK_REASON_LOW_RANSAC_INLIER_FRACTION,
                    FALLBACK_REASON_SEQUENCE_BOUNDARY)
# Stable numeric codes for the serialized payload (the names above remain the public identifiers). The order is
# the precedence order, so a smaller code is a stronger gate; -1 marks a fit whose metadata was not recorded.
FALLBACK_REASON_CODES = {name: code for code, name in enumerate(FALLBACK_REASONS)}
FALLBACK_REASON_BY_CODE = {code: name for code, name in enumerate(FALLBACK_REASONS)}
FALLBACK_REASON_UNAVAILABLE_CODE = -1
MIN_INLIER_FRACTION = 0.15   # the fraction gate ``fit_motion`` already applies; named for the diagnostics


@dataclass
class Obs:                       # flow samples for one frame pair (frame t -> t+1)
    pts: np.ndarray              # [N,2] pixel positions in frame t
    flow: np.ndarray             # [N,2]
    valid: np.ndarray            # [N] bool, forward-backward consistent and inside the frame
    W: int
    H: int
    # Per-sample forward-backward cycle diagnostics recorded by ``observe`` for *every* sampled grid point,
    # including points the consistency tolerance rejects (``valid`` is only a derived boolean). ``fb_err`` is
    # the cycle error ``||f + sampled_b||`` in pixels, ``fb_tol`` the tolerance ``fb_abs + fb_rel*||f||`` it is
    # compared against, and ``fb_inside`` whether the forward endpoint ``pts + flow`` lies inside the frame;
    # ``valid = (fb_err < fb_tol) & fb_inside``. Keeping the two components (rather than only ``valid``) lets a
    # spatial map count rejected-but-in-bounds samples too. This is a forward-backward **cycle-consistency**
    # diagnostic only: not a true correspondence error, not a probability, not calibrated uncertainty. ``None``
    # on observations built by hand.
    fb_err: np.ndarray | None = None      # [N] pixels
    fb_tol: np.ndarray | None = None      # [N] pixels
    fb_inside: np.ndarray | None = None   # [N] bool


@dataclass
class Fit:
    H: np.ndarray                # 3x3, frame t -> t+1 (global)
    mesh: np.ndarray             # [gr+1, gc+1, 2] residual displacement at mesh vertices (content motion beyond H)
    rows: np.ndarray             # [nr, 2] x-independent residual per row band (rolling shutter), content motion
    inlier_frac: float
    mover_frac: float
    fallback: bool
    # Fit-gate diagnostics recorded by ``fit_motion``: the number of valid flow samples feeding the initial
    # RANSAC solve, the number of RANSAC inliers it returned (both on the same all-valid solve whose
    # ``inlier_frac`` above is stored), and the stable ``fallback_reason`` (one of ``FALLBACK_REASONS``, or
    # ``FALLBACK_REASON_NONE`` for a successful fit). These are diagnostics only and never feed fitting,
    # stabilization or rendering. ``None`` means the value was not recorded (a manually constructed Fit, or a
    # legacy ``mlxstab.motion/1``/``/2``/``/3`` payload), not a count of zero.
    n_valid_samples: int | None = None
    n_ransac_inliers: int | None = None
    fallback_reason: str | None = None
    # Forward-backward **cycle-consistency** spatial maps on the mesh grid, built from every finite, in-bounds
    # forward endpoint independently of the fit's valid/non-mover mask (a sample rejected by the consistency
    # tolerance is still counted). ``cycle_err_rms`` is the Gaussian-spatially-weighted RMS of the per-sample
    # cycle error ``||f + sampled_b||`` in pixels; ``cycle_inconsistency_frac`` the spatially-weighted fraction
    # of in-bounds samples whose cycle error fails the existing tolerance ``fb_abs + fb_rel*||f||`` (in [0,1]);
    # ``cycle_sample_mass`` the spatial sample mass/coverage. A vertex with zero sample mass has zero maps. These
    # are diagnostics only -- NOT a true correspondence error, NOT a probability, NOT calibrated uncertainty --
    # and do not feed stabilization or fitting. ``None`` for manually constructed / legacy Fits.
    cycle_err_rms: np.ndarray | None = None            # [gr+1, gc+1] pixels
    cycle_inconsistency_frac: np.ndarray | None = None # [gr+1, gc+1] in [0,1]
    cycle_sample_mass: np.ndarray | None = None        # [gr+1, gc+1]
    # Observation coverage, NOT a probability and NOT flow confidence: the non-robust kernel-weighted count of
    # usable (valid, non-mover) flow samples contributing to each mesh vertex / row band, recorded *before*
    # ``_fill`` diffuses under-observed locations. ``value >= SUPPORT_THRESHOLD`` marks a directly observed
    # vertex/band, ``value < SUPPORT_THRESHOLD`` marks one filled by diffusion. ``None`` means the information
    # is unavailable (e.g. a legacy ``mlxstab.motion/1`` payload) rather than zero support.
    mesh_support_mass: np.ndarray | None = None   # [gr+1, gc+1]
    row_support_mass: np.ndarray | None = None    # [nr]


def observe(F, B, step=8, fb_abs=1.0, fb_rel=0.05):
    """Sample dense flows F (t->t+1) and B (t+1->t), [H,W,2] mx arrays, on a grid; forward-backward consistency gives `valid`."""
    H, W = F.shape[:2]
    ys = np.arange(step // 2, H, step); xs = np.arange(step // 2, W, step)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    f = F[mx.array(ys)][:, mx.array(xs)].reshape(-1, 2)
    tgt = mx.array(pts) + f
    b = sample(B[None], tgt.reshape(1, 1, -1, 2), "bilinear", "border").reshape(-1, 2)
    err = mx.sqrt(mx.sum((f + b) ** 2, -1)); mag = mx.sqrt(mx.sum(f ** 2, -1))
    inside = (tgt[:, 0] >= 0) & (tgt[:, 0] <= W - 1) & (tgt[:, 1] >= 0) & (tgt[:, 1] <= H - 1)
    valid = (err < fb_abs + fb_rel * mag) & inside
    mx.eval(f, valid, err, mag)
    return Obs(pts.astype(np.float64), np.array(f, np.float64), np.array(valid), W, H,
               fb_err=np.array(err, np.float64), fb_tol=np.array(fb_abs + fb_rel * mag, np.float64),
               fb_inside=np.array(inside, bool))


def _kernel_field(pts, vals, w, centers, sig, iters=3, huber=1.5):
    """Robust Gaussian-kernel regression of vals [N,k] at centers [V,d] (anisotropic sig [d]). Returns (est [V,k], mass [V])."""
    d2 = (((pts[:, None, :] - centers[None]) / sig) ** 2).sum(-1)
    K = np.exp(-0.5 * d2) * (d2 < 9.0)
    rw = np.ones(len(pts))
    for _ in range(iters):
        Kw = K * (w * rw)[:, None]
        mass = Kw.sum(0)
        est = (Kw.T @ vals) / np.maximum(mass, 1e-9)[:, None]
        at = (K @ est) / np.maximum(K.sum(1), 1e-9)[:, None]
        r = np.linalg.norm(vals - at, axis=1)
        rw = np.minimum(1.0, huber / np.maximum(r, 1e-9))
    return est, (K * w[:, None]).sum(0)


def _cycle_maps(obs, gc, gr):
    """Gaussian spatial aggregation of the per-sample forward-backward cycle diagnostics of ``obs``.

    Uses every sample whose forward endpoint ``pts + flow`` is finite and in-bounds, **independent of
    ``Obs.valid``**: a point rejected by the cycle tolerance must still be counted, otherwise the rejection
    rate would vanish exactly where it matters. At each mesh vertex the Gaussian kernel ``exp(-0.5 d2)``
    (``d2 < 9``) -- the same kernel/sigma ``_kernel_field`` uses -- weights the samples: ``cycle_err_rms`` is
    the weighted RMS of ``fb_err`` (pixels), ``cycle_inconsistency_frac`` the weighted fraction with
    ``fb_err >= fb_tol`` (in [0,1]) and ``cycle_sample_mass`` the weight sum. A vertex with zero sample mass
    gets zeroes rather than a 0/0. Returns ``(None, None, None)`` when ``obs`` carries no cycle diagnostics
    (an observation built by hand). This is a forward-backward **cycle-consistency** diagnostic only: it is
    not a true correspondence error, not a probability and not a calibrated uncertainty."""
    err = getattr(obs, "fb_err", None); tol = getattr(obs, "fb_tol", None)
    inside = getattr(obs, "fb_inside", None)
    if err is None or tol is None or inside is None:
        return None, None, None
    W, H = obs.W, obs.H
    pts = np.asarray(obs.pts, np.float64); flow = np.asarray(obs.flow, np.float64)
    endpoint = pts + flow
    err = np.asarray(err, np.float64); tol = np.asarray(tol, np.float64)
    sel = (np.isfinite(endpoint).all(1) & np.asarray(inside, bool) & np.isfinite(err) & np.isfinite(tol))
    rms = np.zeros((gr + 1, gc + 1)); frac = np.zeros((gr + 1, gc + 1)); mass = np.zeros((gr + 1, gc + 1))
    idx = np.flatnonzero(sel)
    if idx.size == 0:
        return rms, frac, mass
    vx = -.5 + np.arange(gc + 1) * W / gc; vy = -.5 + np.arange(gr + 1) * H / gr
    cx, cy = np.meshgrid(vx, vy)
    centers = np.stack([cx.ravel(), cy.ravel()], 1)
    sig = np.array([W / gc, H / gr])
    d2 = (((pts[idx][:, None, :] - centers[None]) / sig) ** 2).sum(-1)         # [S, V]
    K = np.exp(-0.5 * d2) * (d2 < 9.0)
    w = K.sum(0)                                                              # [V]
    e = err[idx]
    num = (K * (e ** 2)[:, None]).sum(0)
    bad = (K * (e >= tol[idx]).astype(float)[:, None]).sum(0)
    nz = w > 0
    rms.ravel()[nz] = np.sqrt(num[nz] / w[nz])
    frac.ravel()[nz] = bad[nz] / w[nz]
    mass.ravel()[nz] = w[nz]
    return rms, frac, mass


def _fill(field, mass, thr):
    """Replace under-observed vertices (mass<thr) by diffusion from observed neighbours (field [R,C,2])."""
    ok = mass >= thr
    f = np.where(ok[..., None], field, 0.0)
    if ok.all(): return f
    if not ok.any(): return np.zeros_like(f)
    cur = ok.copy()
    for _ in range(field.shape[0] + field.shape[1]):
        if cur.all(): break
        pad = np.pad(np.where(cur[..., None], f, 0.0), ((1, 1), (1, 1), (0, 0))); pc = np.pad(cur.astype(float), 1)
        nsum = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
        ncnt = pc[:-2, 1:-1] + pc[2:, 1:-1] + pc[1:-1, :-2] + pc[1:-1, 2:]
        new = (~cur) & (ncnt > 0)
        f[new] = nsum[new] / ncnt[new][:, None]; cur = cur | new
    return f


def _movers(obs, resid, inl, gate=6.0, cell=32):
    """Mask of points on independently moving things: coherent blobs whose flow disagrees with the dominant motion."""
    s = 1.4826 * np.median(resid[inl]) if inl.any() else 1.0
    out = obs.valid & (resid > max(gate, 4 * s))
    gy, gx = obs.H // cell + 1, obs.W // cell + 1
    cy = (obs.pts[:, 1] // cell).astype(int); cx = (obs.pts[:, 0] // cell).astype(int)
    tot = np.zeros((gy, gx)); bad = np.zeros((gy, gx))
    np.add.at(tot, (cy, cx), obs.valid); np.add.at(bad, (cy, cx), out)
    cellbad = (tot > 0) & (bad / np.maximum(tot, 1) >= 0.5)
    # keep blobs of >= 3 cells, dilate one cell
    from scipy.ndimage import label, binary_dilation
    lab, n = label(cellbad)
    keep = np.zeros_like(cellbad)
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() >= 3: keep |= m
    keep = binary_dilation(keep, iterations=1)
    return keep[cy, cx], keep


def fallback_reason(n_valid_samples, n_ransac_inliers, min_valid, min_inlier_frac=MIN_INLIER_FRACTION):
    """Stable reason for the ``fit_motion`` fallback gates, with an explicit precedence.

    Evaluated in order, the *first* failing gate names the reason, so a pair that fails several thresholds still
    gets one deterministic label:

    1. ``insufficient_valid_samples``: ``n_valid_samples < min_valid`` (the fit never reached RANSAC);
    2. ``insufficient_ransac_inliers``: ``n_ransac_inliers < min_valid`` (too few inliers in absolute count);
    3. ``low_ransac_inlier_fraction``: ``n_ransac_inliers / n_valid_samples < min_inlier_frac``.

    ``sequence_boundary`` is not produced by these gates; ``analyze`` assigns it to the skipped pair at an
    explicit cut so the placeholder is never mistaken for a failed geometric fit.

    Returns ``FALLBACK_REASON_NONE`` when every gate passes. This mirrors the existing
    ``inl.sum() < min_valid or inlier_frac < 0.15`` refusal exactly, so it only *labels* the decision and does
    not change it."""
    if n_valid_samples < min_valid:
        return FALLBACK_REASON_INSUFFICIENT_VALID_SAMPLES
    if n_ransac_inliers < min_valid:
        return FALLBACK_REASON_INSUFFICIENT_RANSAC_INLIERS
    if n_valid_samples > 0 and n_ransac_inliers / n_valid_samples < min_inlier_frac:
        return FALLBACK_REASON_LOW_RANSAC_INLIER_FRACTION
    return FALLBACK_REASON_NONE


def _validate_shape_strength(value):
    """Validate the opt-in mesh shape-regularization strength (finite, scalar, >= 0).

    Booleans, strings, arrays and complex values are rejected rather than coerced, so a typo cannot silently
    disable or distort the research-only regularizer. ``0`` is the production default."""
    name = "shape regularization strength"
    if (isinstance(value, (bool, np.bool_, str, bytes)) or not np.isscalar(value)):
        raise ValueError(f"{name} must be a finite non-negative scalar, got {value!r}")
    try:
        strength = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite non-negative scalar, got {value!r}") from None
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar, got {value!r}")
    return strength


def _regularize_mesh_shape(mesh, support_mass, W, H, strength):
    """Weighted least-squares shape-preserving regularization of the local residual mesh.

    The unknown is the per-vertex residual displacement ``u`` (pixels). The objective is

        sum_v m_v ||u_v - u0_v||^2  +  strength * sum_cells area_c * s_c ||dev sym(grad u)||_F^2

    where ``u0`` is the supplied local residual mesh, ``m_v`` is the data/support weight (the pre-fill
    observation coverage mass, normalized by its median positive value), ``s_c`` is the mean normalized support
    of the cell's four vertices, ``area_c`` is the cell area, and ``dev sym(grad u)`` is the trace-free
    symmetric part of the displacement gradient. Translation (constant ``u``), local rotation (skew gradient)
    and isotropic scale (trace) are in the nullspace of the energy, so they are not penalized. The global
    similarity and x-independent row field are not inputs to this solve.

    This is a direct sparse normal-equation solve, not a scalar shrink of ``u0``. At ``strength == 0`` the
    caller does not invoke it, preserving the historical mesh byte-for-byte. Degenerate grids, absent support,
    and non-finite solver results return a finite copy of ``u0`` unchanged."""
    mesh = np.asarray(mesh, np.float64)
    if strength <= 0.0 or mesh.ndim != 3 or mesh.shape[2] != 2:
        return mesh.copy()
    R, C, _ = mesh.shape
    if R < 2 or C < 2 or not np.isfinite(mesh).all():
        return mesh.copy()

    if support_mass is None:
        mass = np.ones((R, C), np.float64)
    else:
        mass = np.asarray(support_mass, np.float64)
        if mass.shape != (R, C):
            raise ValueError(f"support_mass must have shape {(R, C)}, got {mass.shape}")
        mass = np.where(np.isfinite(mass) & (mass > 0.0), mass, 0.0)
    positive = mass[mass > 0.0]
    if positive.size == 0:
        return mesh.copy()
    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        return mesh.copy()
    mass = mass / scale

    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    dx = float(W) / float(C - 1)
    dy = float(H) / float(R - 1)
    if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0.0 or dy <= 0.0:
        return mesh.copy()
    area = dx * dy

    # Assemble the two independent deviatoric strain components per cell. For e00 = (u_x,x - u_y,y)/2 and
    # e01 = (u_x,y + u_y,x)/2, ||dev sym(grad u)||_F^2 = 2 * (e00^2 + e01^2); the sqrt(2*area*s) row weight
    # makes the sparse normal equations equal that per-cell energy.
    rows, cols, vals = [], [], []
    row = 0
    for i in range(R - 1):
        for j in range(C - 1):
            n00 = i * C + j
            n01 = n00 + 1
            n10 = n00 + C
            n11 = n10 + 1
            cell_support = 0.25 * (mass[i, j] + mass[i, j + 1] + mass[i + 1, j] + mass[i + 1, j + 1])
            if not np.isfinite(cell_support) or cell_support <= 0.0:
                continue
            rw = np.sqrt(2.0 * area * cell_support)
            # e00 row.
            for node, comp, coef in (
                    (n00, 0, -1.0 / (4.0 * dx)), (n01, 0, 1.0 / (4.0 * dx)),
                    (n10, 0, -1.0 / (4.0 * dx)), (n11, 0, 1.0 / (4.0 * dx)),
                    (n00, 1, 1.0 / (4.0 * dy)), (n01, 1, 1.0 / (4.0 * dy)),
                    (n10, 1, -1.0 / (4.0 * dy)), (n11, 1, -1.0 / (4.0 * dy))):
                rows.append(row); cols.append(2 * node + comp); vals.append(rw * coef)
            row += 1
            # e01 row.
            for node, comp, coef in (
                    (n00, 1, -1.0 / (4.0 * dx)), (n01, 1, 1.0 / (4.0 * dx)),
                    (n10, 1, -1.0 / (4.0 * dx)), (n11, 1, 1.0 / (4.0 * dx)),
                    (n00, 0, -1.0 / (4.0 * dy)), (n01, 0, -1.0 / (4.0 * dy)),
                    (n10, 0, 1.0 / (4.0 * dy)), (n11, 0, 1.0 / (4.0 * dy))):
                rows.append(row); cols.append(2 * node + comp); vals.append(rw * coef)
            row += 1
    if row == 0:
        return mesh.copy()

    D = sparse.coo_matrix((vals, (rows, cols)), shape=(row, 2 * R * C)).tocsr()
    # A tiny floor on otherwise unsupported vertices keeps the normal equations non-singular and anchors them to
    # the finite filled target; it is negligible wherever real support exists.
    floor = max(1e-9 * float(mass.max()), 1e-12)
    data_w = np.repeat(np.where(mass > 0.0, mass, floor).ravel(), 2)
    A = sparse.diags(data_w, format="csr") + strength * (D.T @ D)
    u0 = mesh.reshape(-1)
    try:
        u = spsolve(A.tocsc(), data_w * u0)
    except Exception:
        return mesh.copy()
    if not np.isfinite(u).all():
        return mesh.copy()
    return np.asarray(u, np.float64).reshape(R, C, 2)


def fit_motion(obs, mesh=(16, 9), rs_rows=18, thr=1.5, rng=None, min_valid=60, model="similarity",
               shape_strength=0.0):
    """Global homography by RANSAC on valid flow samples, movers excluded, then row-wise and mesh residual fields.

    ``shape_strength`` is a research-only, opt-in strength for shape-preserving regularization of the local
    residual mesh (trace-free symmetric strain). It defaults to ``0``, which leaves the historical H/mesh/rows
    byte-for-byte unchanged. The regularizer never touches the global similarity or the x-independent row field."""
    strength = _validate_shape_strength(shape_strength)
    gc, gr = mesh
    W, H = obs.W, obs.H
    v = obs.valid
    nr = max(rs_rows, 1)
    # Cycle diagnostics are computed from the raw samples before any geometric decision, so the fallback path
    # below still carries them (the geometric fit may fail while the flow cycle diagnostics remain available).
    c_rms, c_frac, c_mass = _cycle_maps(obs, gc, gr)

    def _fallback(n_valid_samples, n_ransac_inliers, reason):
        return Fit(np.eye(3), np.zeros((gr + 1, gc + 1, 2)), np.zeros((nr, 2)), 0.0, 0.0, True,
                   n_valid_samples=int(n_valid_samples), n_ransac_inliers=int(n_ransac_inliers),
                   fallback_reason=reason,
                   mesh_support_mass=np.zeros((gr + 1, gc + 1)), row_support_mass=np.zeros(nr),
                   cycle_err_rms=c_rms, cycle_inconsistency_frac=c_frac, cycle_sample_mass=c_mass)

    n_valid = int(v.sum())
    if n_valid < min_valid:
        return _fallback(n_valid, 0, FALLBACK_REASON_INSUFFICIENT_VALID_SAMPLES)
    src = obs.pts[v]; dst = src + obs.flow[v]
    Hm, inl = ransac_homography(src, dst, thr=thr, rng=rng)
    inlier_frac = inl.mean()
    n_inliers = int(inl.sum())
    reason = fallback_reason(n_valid, n_inliers, min_valid)
    if reason != FALLBACK_REASON_NONE:
        return _fallback(n_valid, n_inliers, reason)
    mv_last = np.zeros(len(v), bool)
    for _ in range(2):                                                  # re-fit with moving subjects removed
        resid_all = np.linalg.norm(obs.pts + obs.flow - apply_h(Hm, obs.pts), axis=1)
        mv, _cells = _movers(obs, resid_all, (resid_all < thr) & v)
        use = v & ~mv; mv_last = mv
        if use.sum() < min_valid: break
        Hm, inl = ransac_homography(obs.pts[use], obs.pts[use] + obs.flow[use], thr=thr, rng=rng)
    if model != "homography":                                           # homography RANSAC picks inliers; the kept model is lower-dof
        e = np.linalg.norm(obs.pts + obs.flow - apply_h(Hm, obs.pts), axis=1)
        k = v & ~mv_last & (e < 2 * thr)
        if k.sum() >= min_valid: Hm = fit_model(obs.pts[k], obs.pts[k] + obs.flow[k], model)
    resid_vec = obs.flow - (apply_h(Hm, obs.pts) - obs.pts)
    resid_all = np.linalg.norm(resid_vec, axis=1)
    mv, _ = _movers(obs, resid_all, (resid_all < thr) & v)
    use = v & ~mv
    w = use.astype(float)
    # rolling shutter: x-independent residual per row band
    rows = np.zeros((nr, 2))
    row_support_mass = np.zeros(nr)                                 # zeros while rows are disabled (rs_rows<=0)
    if rs_rows > 0:
        cy = (np.arange(rs_rows) + .5) * H / rs_rows - .5
        est, mass = _kernel_field(obs.pts[:, 1:2], resid_vec, w, cy[:, None], np.array([H / rs_rows]))
        row_support_mass = mass                                     # preserved before _fill diffuses bands
        rows = _fill(est[:, None], mass[:, None], SUPPORT_THRESHOLD)[:, 0]
        row_at = np.stack([np.interp(obs.pts[:, 1], cy, rows[:, 0]), np.interp(obs.pts[:, 1], cy, rows[:, 1])], 1)
        resid_vec = resid_vec - row_at
    vx = -.5 + np.arange(gc + 1) * W / gc; vy = -.5 + np.arange(gr + 1) * H / gr
    cx, cy2 = np.meshgrid(vx, vy)
    centers = np.stack([cx.ravel(), cy2.ravel()], 1)
    est, mass = _kernel_field(obs.pts, resid_vec, w, centers, np.array([W / gc, H / gr]))
    mesh_support_mass = mass.reshape(gr + 1, gc + 1)                # preserved before _fill diffuses vertices
    mesh_f = _fill(est.reshape(gr + 1, gc + 1, 2), mesh_support_mass, SUPPORT_THRESHOLD)
    p = np.pad(mesh_f, ((1, 1), (1, 1), (0, 0)), mode="edge")                # light spatial regularisation [1 2 1]^2/16
    mesh_f = (p[:-2, 1:-1] + 2 * p[1:-1, 1:-1] + p[2:, 1:-1]) / 4
    p = np.pad(mesh_f, ((0, 0), (1, 1), (0, 0)), mode="edge"); mesh_f = (p[:, :-2] + 2 * p[:, 1:-1] + p[:, 2:]) / 4
    if strength > 0.0:
        mesh_f = _regularize_mesh_shape(mesh_f, mesh_support_mass, W, H, strength)
    return Fit(Hm, mesh_f, rows, float(inlier_frac), float(mv[v].mean()), False,
               n_valid_samples=n_valid, n_ransac_inliers=n_inliers, fallback_reason=FALLBACK_REASON_NONE,
               mesh_support_mass=mesh_support_mass, row_support_mass=row_support_mass,
               cycle_err_rms=c_rms, cycle_inconsistency_frac=c_frac, cycle_sample_mass=c_mass)


def _dilate(mask, pad):
    """Invalid-mask dilation: True within `pad` pixels (Chebyshev) of any invalid pixel. pad<=0 is a no-op."""
    mask = np.asarray(mask, bool)
    if pad <= 0: return mask
    from scipy.ndimage import maximum_filter
    return maximum_filter(mask, size=2 * pad + 1, mode="constant", cval=False)


def _sample_invalid(mask, px, py, H, W):
    """True where a correspondence must be rejected for one (already dilated) invalid mask.

    `px`, `py` are float pixel-centre coordinates of the observation in the mask's frame. A position is
    unsafe when it is non-finite, outside [0, W-1] x [0, H-1], or its bilinear support touches an invalid
    pixel with non-zero weight. At an exact pixel centre only that single tap has weight, so a valid pixel
    edge is not rejected by rounding alone. Conservative by construction: any non-zero weight on an invalid
    neighbour rejects the sample."""
    px = np.asarray(px, np.float64); py = np.asarray(py, np.float64)
    inside = np.isfinite(px) & np.isfinite(py) & (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)
    bad = ~inside
    idx = np.flatnonzero(inside)
    if idx.size:
        x = px[idx]; y = py[idx]
        x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
        fx = x - x0; fy = y - y0
        x1 = x0 + (fx > 0); y1 = y0 + (fy > 0)                     # duplicate taps at integer coords cancel via weights
        w00 = (1 - fx) * (1 - fy); w01 = (1 - fx) * fy; w10 = fx * (1 - fy); w11 = fx * fy
        acc = (mask[y0, x0] * w00 + mask[y1, x0] * w01 + mask[y0, x1] * w10 + mask[y1, x1] * w11)
        bad[idx] = acc > 0.0
    return bad


def exclude_masked(obs, m0, m1, pad=12):
    """Drop samples near invalid (no-source / border) pixels of the source or destination frame.

    `m0` and `m1` are bool [H,W] invalid masks (True = no source) of frame t (source) and t+1
    (destination), or None when a mask is unavailable. `m0` is sampled at `obs.pts`; `m1` is sampled at
    `obs.pts + obs.flow`, because the destination mask lives in the destination frame's coordinates. Each
    mask is first dilated by `pad` pixels, then sampled conservatively bilinearly (see `_sample_invalid`).
    Out-of-image and non-finite positions are rejected. `valid` is only ever ANDed, so an observation that
    was already invalid is never reactivated."""
    if m0 is None and m1 is None: return
    dst = obs.pts + obs.flow
    for m, pts in ((m0, obs.pts), (m1, dst)):
        if m is None: continue
        bad = _sample_invalid(_dilate(m, pad), pts[:, 0], pts[:, 1], obs.H, obs.W)
        obs.valid &= ~bad
