"""Complete-warp shape and coverage diagnostics.

These diagnostics measure the *complete* source-coordinate field actually used by rendering: the global
homography plus the residual mesh plus the rolling-shutter row correction. They are separate from the
historical ``distortion`` score, which only looks at the global homography and is kept unchanged.

Sampling and derivative convention
----------------------------------
The source-coordinate field ``src[y, x] = (sx, sy)`` maps an *output* (virtual) pixel centre ``(x, y)`` to a
source-image coordinate; it is evaluated once per output pixel centre on the integer grid. Local shape uses
**forward differences between adjacent pixel centres**. Each 1x1 output cell with corners
``(x,y) -> (x+1,y) -> (x+1,y+1) -> (x,y+1)`` is a bilinear patch, so its **four corner Jacobians** are
evaluated (the two edge differences on each axis are combined in all four pairings). Derivatives are in
source pixels per output pixel (pixel spacing is 1, not normalised further). Forward differences are used
deliberately: a centred difference cancels an alternating (checkerboard) fold, and a single top-left
Jacobian misses a fold, non-finite corner, or concave cell at the opposite corner. The signed determinant of
a corner Jacobian is the oriented area there (positive = orientation preserved, negative = folded, near zero
= collapsed). A bilinear patch's determinant is affine (the ``uv`` terms cancel), so its signed extrema lie on
the corners and the four corners bound the signed determinant; ``det_negative_frac`` flags any corner fold
(including mixed-sign cells, whose interior necessarily reaches zero), while ``det_collapse_frac`` is a
corner-collapse fraction and does not by itself bound interior collapse of a mixed-sign cell. Shape summaries
are per cell, combining its four corners (see the metric table in ``docs/warp_diagnostics.md``).

Coverage convention
-------------------
``center_valid`` is the existing pixel-edge convention used by :func:`mlxstab.warp.valid_mask`: a mapped point
is valid while ``-0.5 <= sx <= W-0.5`` and ``-0.5 <= sy <= H-0.5``. ``full_support`` additionally requires
that every interpolation tap with non-zero weight lies inside the source pixel array
``[0, W-1] x [0, H-1]``. Taps with ``|weight| <= tol`` (default ``1e-6``) are treated as zero, so an exact
identity warp is fully supported even though the nominal footprint of an edge pixel extends past the image.
Coverage fractions are always reported over *all* output pixels: invalid coverage is never dropped from the
denominator.

All returned values are JSON-safe: undefined or degenerate summaries are ``None``, never ``NaN``/``Inf``.
Cells whose finite coordinates overflow the determinant/singular-value computation are counted as unusable
(``overflow_cells``) and excluded from summaries rather than silently scoring as perfect.
"""
import numpy as np

KERNELS = ("bilinear", "bicubic", "lanczos3")
_TAPS = {"bilinear": 2, "bicubic": 4, "lanczos3": 6}


def tap_weights(kernel, d):
    """Interpolation tap weight for offset ``d``, matching the Metal sampler in :mod:`mlxstab.warp`."""
    a = np.abs(np.asarray(d, np.float64))
    if kernel == "bilinear":
        return np.where(a < 1.0, 1.0 - a, 0.0)
    if kernel == "bicubic":                                   # Catmull-Rom, a = -0.5
        return np.where(a < 1.0, 1.5 * a ** 3 - 2.5 * a ** 2 + 1.0,
                        np.where(a < 2.0, -0.5 * a ** 3 + 2.5 * a ** 2 - 4.0 * a + 2.0, 0.0))
    if kernel == "lanczos3":
        with np.errstate(invalid="ignore", divide="ignore"):
            s1 = np.where(a < 1e-12, 1.0, np.sin(np.pi * a) / (np.pi * a))
            s2 = np.where(a / 3.0 < 1e-12, 1.0, np.sin(np.pi * (a / 3.0)) / (np.pi * (a / 3.0)))
        w = np.where(a < 3.0, s1 * s2, 0.0)
        return np.where(np.isfinite(w), w, 0.0)
    raise ValueError(f"unknown kernel {kernel!r}")


def _axis_full_support(coord, n, kernel, tol):
    """True where every tap with ``|weight| > tol`` on one axis is inside ``[0, n-1]``."""
    taps = _TAPS[kernel]
    x0 = np.floor(coord) - (taps // 2 - 1)
    full = np.ones(coord.shape, bool)
    for t in range(taps):
        idx = x0 + t
        w = tap_weights(kernel, coord - idx)
        full &= ((idx >= 0) & (idx <= n - 1)) | (np.abs(w) <= tol)
    return full


def full_support_mask(src, H=None, W=None, kernel="lanczos3", tol=1e-6):
    """Per-output-pixel mask for complete interpolation-kernel support inside the source image.

    This uses the same tap locations and ``abs(weight) <= tol`` zero-tap rule as :func:`diagnose`. ``src`` must
    be an ``[H,W,2]`` source-coordinate field. The public mask lets offline candidate renderers act on the actual
    sampler footprint instead of approximating it with center-in-bounds checks.
    """
    a = np.asarray(src)
    if a.ndim != 3 or a.shape[-1] != 2:
        raise ValueError("src must have shape [H,W,2]")
    h, w = a.shape[:2]
    H = h if H is None else int(H)
    W = w if W is None else int(W)
    if H < 1 or W < 1:
        raise ValueError("source dimensions must be positive")
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; expected one of {KERNELS}")
    tolerance = float(tol)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tol must be finite and non-negative")
    finite = np.isfinite(a).all(axis=-1)
    sx, sy = a[..., 0], a[..., 1]
    return (finite & _axis_full_support(sx, W, kernel, tolerance)
            & _axis_full_support(sy, H, kernel, tolerance))


def _frac(mask):
    mask = np.asarray(mask, bool)
    return float(mask.mean()) if mask.size else None


def _scale_back(ratio, scale):
    """Undo a division by ``scale`` without overflowing: the true value lies in ``[-scale, scale]``."""
    if scale == 0.0: return 0.0
    return float(np.clip(ratio, -1.0, 1.0)) * scale


def _finite_mean(v):
    """Mean of finite values that cannot overflow: divide by the largest magnitude, average, scale back."""
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]
    if v.size == 0: return None
    scale = float(np.max(np.abs(v)))
    if scale == 0.0: return 0.0
    return _scale_back((v / scale).mean(), scale)


def _stats(vals):
    """mean/min/max/p50/p95 over finite entries; all ``None`` when there are none.

    Reductions are done after dividing by the largest magnitude, so a representable finite mean or quantile is
    not turned into ``inf``: interpolated quantiles subtract endpoints and can overflow for opposite-sign
    extremes, and a naive mean can overflow for large equal values."""
    v = np.asarray(vals, np.float64); v = v[np.isfinite(v)]
    if v.size == 0:
        return {k: None for k in ("mean", "min", "max", "p50", "p95")}
    mn, mx = float(v.min()), float(v.max())
    scale = max(abs(mn), abs(mx)); z = v / scale if scale else v
    return {"mean": _scale_back(z.mean(), scale), "min": mn, "max": mx,
            "p50": _scale_back(np.percentile(z, 50), scale), "p95": _scale_back(np.percentile(z, 95), scale)}


def _stable_svd2(a, b, c, d):
    """Elementwise singular values and determinant of 2x2 matrices ``J = [[a, b], [c, d]]``.

    A scale-normalised analytic 2x2 SVD is used: ``s2 = |det| / s1`` avoids the catastrophic cancellation in
    ``sqrt((E - disc)/2)``. The scaled intermediates are bounded, but the final determinant and singular values
    can still overflow for very large finite inputs (e.g. coordinates near 1e200); ``overflow`` marks those so
    the cell is excluded instead of silently scoring as perfect."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        m = np.maximum(np.maximum(np.abs(a), np.abs(b)), np.maximum(np.abs(c), np.abs(d)))
        ms = np.where(m > 0, m, 1.0)
        a_, b_, c_, d_ = a / ms, b / ms, c / ms, d / ms
        det_ = a_ * d_ - b_ * c_                               # |det_| <= 2 for finite inputs
        e_ = a_ * a_ + b_ * b_ + c_ * c_ + d_ * d_             # 0 <= e_ <= 4
        disc_ = np.sqrt(np.maximum(e_ * e_ - 4.0 * det_ * det_, 0.0))
        s1_ = np.sqrt(np.maximum((e_ + disc_) / 2.0, 0.0))
        s1 = m * s1_
        s2 = np.where(s1_ > 0, (np.abs(det_) / s1_) * m, 0.0)
        det = np.where(det_ == 0, 0.0, (m * m) * det_)
    finite = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & np.isfinite(d)
    overflow = finite & ~(np.isfinite(det) & np.isfinite(s1) & np.isfinite(s2))
    return s1, s2, det, overflow


def _shape(src, H, W, finite, s2_tol, det_tol):
    degenerate = bool(H < 2 or W < 2)
    out = {"cells": 0, "degenerate": degenerate, "nonfinite_frac": _frac(~finite),
           "nonfinite_cells": None, "nonfinite_cell_frac": None,
           "overflow_cells": None, "overflow_cell_frac": None, "usable_cells": None}
    for k in ("anisotropy", "stretch", "shear_deg"):
        for s in ("mean", "min", "max", "p50", "p95"):
            out[f"{k}_{s}"] = None
    for k in ("det_mean", "det_min", "det_abs_min", "det_negative_frac", "det_collapse_frac"):
        out[k] = None
    if degenerate:
        return out

    dxx = src[:, 1:] - src[:, :-1]                             # [H, W-1, 2] x-forward differences
    dyy = src[1:] - src[:-1]                                   # [H-1, W, 2] y-forward differences
    X_top, X_bot = dxx[:H - 1, :W - 1], dxx[1:H, :W - 1]       # x edges: top / bottom of each cell
    Y_left, Y_right = dyy[:H - 1, :W - 1], dyy[:H - 1, 1:W]    # y edges: left / right of each cell
    corners = ((X_top, Y_left), (X_top, Y_right), (X_bot, Y_left), (X_bot, Y_right))
    cell_finite = np.ones((H - 1, W - 1), bool)                # a cell needs all four source corners finite
    for u, v in corners:
        cell_finite &= np.isfinite(u).all(-1) & np.isfinite(v).all(-1)

    det_min = np.full((H - 1, W - 1), np.nan); det_abs_min = np.full((H - 1, W - 1), np.nan)
    det_mean_acc = np.zeros((H - 1, W - 1)); s1_max = np.full((H - 1, W - 1), np.nan)
    aniso_max = np.full((H - 1, W - 1), np.nan); shear_max = np.full((H - 1, W - 1), np.nan)
    neg = np.zeros((H - 1, W - 1), bool); col = np.zeros((H - 1, W - 1), bool)
    over = np.zeros((H - 1, W - 1), bool)
    for u, v in corners:
        a, b, c, d = u[..., 0], v[..., 0], u[..., 1], v[..., 1]
        s1, s2, det, ovf = _stable_svd2(a, b, c, d)
        fin = np.isfinite(det)
        det_min = np.fmin(det_min, det); det_abs_min = np.fmin(det_abs_min, np.abs(det))
        det_mean_acc = det_mean_acc + np.where(fin, det / 4.0, 0.0)   # average per corner, so the sum cannot overflow
        s1_max = np.fmax(s1_max, s1)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            ratio = np.where(s2 > s2_tol, s1 / s2, np.nan)
            n1 = np.hypot(a, c); n2 = np.hypot(b, d)
            cosang = np.where((n1 > s2_tol) & (n2 > s2_tol),
                              np.clip((a * b + c * d) / (n1 * n2), -1.0, 1.0), np.nan)
            sh = np.abs(np.degrees(np.arccos(cosang)) - 90.0)
        aniso_max = np.fmax(aniso_max, ratio); shear_max = np.fmax(shear_max, sh)
        neg |= fin & (det < -det_tol); col |= fin & (np.abs(det) <= det_tol)
        over |= ovf

    nonfinite = ~cell_finite
    overflow = cell_finite & over                              # usable/nonfinite/overflow partition the cells
    usable = cell_finite & ~over
    out["cells"] = int(cell_finite.size)
    out["nonfinite_cells"] = int(nonfinite.sum()); out["nonfinite_cell_frac"] = _frac(nonfinite)
    out["overflow_cells"] = int(overflow.sum()); out["overflow_cell_frac"] = _frac(overflow)
    out["usable_cells"] = int(usable.sum())
    if not usable.any():
        return out
    out["det_mean"] = _finite_mean(det_mean_acc[usable])       # robust across cells; corners already averaged
    out["det_min"] = float(det_min[usable].min())
    out["det_abs_min"] = float(det_abs_min[usable].min())
    out["det_negative_frac"] = _frac(neg[usable])
    out["det_collapse_frac"] = _frac(col[usable])
    for key, vals in (("anisotropy", aniso_max[usable]), ("stretch", s1_max[usable]), ("shear_deg", shear_max[usable])):
        for s, v in _stats(vals).items():
            out[f"{key}_{s}"] = v
    return out


def _coverage(src, H, W, finite, kernels, tol):
    sx, sy = src[..., 0], src[..., 1]
    center = finite & (sx >= -0.5) & (sx <= W - 0.5) & (sy >= -0.5) & (sy <= H - 0.5)
    out = {"pixels": int(H * W), "nonfinite_pixels": int((~finite).sum()),
           "nonfinite_frac": _frac(~finite), "center_valid_frac": _frac(center)}
    for k in kernels:
        full = finite & _axis_full_support(sx, W, k, tol) & _axis_full_support(sy, H, k, tol)
        out[f"full_support_{k}_frac"] = _frac(full)
        out[f"partial_support_{k}_frac"] = _frac(center & ~full)
    return out


def diagnose(src, H=None, W=None, kernel="lanczos3", kernels=KERNELS, tol=1e-6, s2_tol=1e-9, det_tol=1e-9):
    """Shape and coverage diagnostics for one complete source-coordinate field.

    ``src`` is ``[H,W,2]`` (source pixels, output pixel-centre sampling). ``H``/``W`` default to the array
    shape. Returns ``{"shape": {...}, "coverage": {...}}`` with JSON-safe scalar values (``None`` when
    undefined). ``kernel`` selects which kernel's ``partial_support`` macro is highlighted by callers;
    coverage is reported for every entry of ``kernels``."""
    src = np.asarray(src, np.float64)
    if src.ndim != 3 or src.shape[2] != 2:
        raise ValueError("src must have shape [H,W,2]")
    H = src.shape[0] if H is None else int(H)
    W = src.shape[1] if W is None else int(W)
    if src.shape[:2] != (H, W):
        raise ValueError(f"src shape {src.shape[:2]} does not match H,W=({H},{W})")
    if kernel not in kernels:
        raise ValueError(f"kernel {kernel!r} not in {kernels!r}")
    finite = np.isfinite(src).all(-1)
    return {"shape": _shape(src, H, W, finite, s2_tol, det_tol),
            "coverage": _coverage(src, H, W, finite, kernels, tol)}


def aggregate(diags):
    """Aggregate per-frame :func:`diagnose` results into ``{key: {mean, min, max}}`` for every scalar leaf.

    ``None`` leaves are ignored; non-finite values are ignored defensively. The result is JSON-safe."""
    leaves = {}

    def walk(node, prefix):
        for k, v in node.items():
            key = prefix + k
            if isinstance(v, dict):
                walk(v, key + ".")
            else:
                leaves.setdefault(key, []).append(v)

    for d in diags:
        walk(d, "")
    out = {}
    for key, vals in leaves.items():
        nums = [float(v) for v in vals
                if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v)]
        if not nums:
            out[key] = None
            continue
        agg = {"mean": _finite_mean(nums), "min": float(np.min(nums)), "max": float(np.max(nums))}
        out[key] = agg if all(v is not None and np.isfinite(v) for v in agg.values()) else None   # JSON-safe
    return out
