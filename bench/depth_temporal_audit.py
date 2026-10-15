"""Weight-free temporal-consistency audit for a cached monocular relative-depth map.

This is a small, **pure-NumPy** diagnostic. It asks one narrow question: is a cached *relative* depth map
more temporally consistent when the target frame's depth is sampled through candidate RAFT flow than when it
is read at the *same* pixel indices, once each comparison is allowed an independent per-pair affine
scale/offset? It does not run RAFT, decode media, download weights, read a manifest, render frames or import
production stabilization code. Nothing here is wired into a pipeline and no default is changed.

Why affine scale/offset, and what it discards
---------------------------------------------
Monocular relative depth is only defined up to an arbitrary positive scale and shift, so a raw depth
difference between two frames is meaningless. To compare *structure* rather than calibration, each method
(same-index or flow-warped) fits a robust affine map from the candidate samples of the **target** depth back
onto the **source** depth and reports the residual of that fitted map. The fitted scale/offset therefore
**discard all metric depth information**: the reported numbers say nothing about absolute depth, metric
accuracy, or which map is "correct". A lower flow-warped residual only means the transported target depth
is *more self-consistent with the source* under that affine model; systematic monocular bias, moving
objects, occlusion and depth-model error all survive this filter.

Support and normalization
--------------------------
Both methods are always evaluated on the **exact same support** so the comparison is paired. The support is
the set of feature cells where the forward flow is finite and stays inside the feature grid, both source and
target depth are finite, and the cell is not a false forward-cycle-valid cell. On top of that base support the
caller may pass thresholds; for each threshold only cells whose cycle residual is ``<= threshold`` are kept,
and the audit additionally always reports the ``all_cycle_valid`` tier. Support filters are **diagnostic
only**: they select a subset, they do not validate the depth map, and no threshold is chosen or recommended
here. Residuals are normalized by the source-depth ``p90 - p10`` span (a robust spread), so the reported
``*_normalized`` fields are in units of source-depth spread, not depth units.

Coordinate convention
---------------------
The depth map passed to :func:`depth_to_feature_grid` is the **full-resolution** depth map in **original**
frame coordinates, i.e. a per-pixel ``[H,W]`` map at the same resolution as the original frame -- not a
pre-downscaled feature-grid map and not the already-reduced content ``downscale`` was applied to. That is
required because ``downscale`` maps each feature-grid cell centre *back* to an original-image pixel centre. A
feature cell ``(fx, fy)`` sits at preprocessed pixel index ``(8*fx - pad_left, 8*fy - pad_top)`` -- the same
convention :func:`bench.flow_prediction_real.photometric_residual` uses -- and that index maps to the
original pixel centre via ``downscale * index + (downscale - 1) / 2``, the centre of the ``downscale`` block
the integer box reduction averaged. Sampling reuses :func:`bench.flow_prediction.sample_field` (bilinear).
Cells whose sample centre falls outside the original depth image are returned as ``nan`` rather than clamped
to the border: those are padded RAFT feature cells with no real depth evidence.

Robust affine fit
-----------------
The target-to-source affine fit is deterministic (no RANSAC, no random seeding): it starts from OLS on
``[x, 1]`` (via ``numpy.linalg.lstsq`` with ``rcond=None``) and, for at most ``MAD_MAX_ITERS`` iterations,
refits on the samples whose absolute residual is within ``MAD_OUTLIER_SIGMA`` times the robust residual sigma
(``1.4826 * median absolute deviation``) of the current fit -- a hard-rejection iteration, not a smooth
reweighting. The loop stops early when the robust sigma is at the residual noise floor (an exact fit) or when
no sample is rejected. This is robust to a **minority** of outliers; when outliers are a majority the
MAD scale itself is inflated by them and the fit can be pulled toward the outlier population, so the
reported scale/offset are only as trustworthy as the inlier fraction (``fit_inlier_count`` makes that
visible). Residual statistics (``median_abs_residual``/``p95_abs_residual`` and their normalized forms) are
always evaluated on the **full** support, outliers included, so discarded points remain visible in the
reported numbers. A unique OLS solution exists whenever the current fit samples' x-values are not all
identical (target depth spread ``TSS > 0``). Degenerate supports (fewer than the minimum number of samples,
a target depth with no span, or clipping leaving fewer than the minimum number of inliers) return explicit
``None`` metrics with a machine-readable reason instead of ``nan``/``inf`` or a divide-by-zero warning.

Finite behavior and limitations
-------------------------------
All array inputs pass through :func:`numpy.asarray(..., float64)`; non-finite depths/flows, out-of-bounds
mapped cells and non-finite samples are excluded, never clamped into the support. Empty or too-small
supports, and a depth field with zero span, yield null metrics with a reason. This is a **diagnostic**, not a
metric-depth estimator, not a depth-accuracy benchmark and not an occlusion/motion detector.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from . import flow_prediction as FP

MIN_SAMPLES = 3          # a scale+offset affine fit needs at least 3 samples
FEATURE_STRIDE = 8       # RAFT feature encoder stride
SOURCE_SPAN_EPS = 0.0    # p90-p10 must strictly exceed this to normalize
FIT_TARGET_SPAN_EPS = 0.0  # target TSS must strictly exceed this for the affine fit to be identifiable
MAD_OUTLIER_SIGMA = 2.5  # drop fit samples beyond this many robust sigmas of the residual
MAD_MAX_ITERS = 5        # fixed cap on MAD outlier-rejection iterations (deterministic)
_EXACT_FIT_REL_EPS = 1e-9  # a robust sigma this small relative to the data scale means an exact fit


def depth_to_feature_grid(depth: np.ndarray, feature_h: int, feature_w: int, pad: Sequence[int],
                          downscale: int) -> np.ndarray:
    """Sample a full-resolution 2D depth map at the exact RAFT feature-cell centres.

    Feature cell ``(fx, fy)`` corresponds to preprocessed pixel index ``(8*fx - pad_left, 8*fy - pad_top)``
    (``pad`` is the replicate pad ``(top, bottom, left, right)`` ``RaftRunner._prep_pad`` applied). For
    ``downscale == 1`` the preprocessed index *is* the pixel index, so only the pad shifts it; for
    ``downscale > 1`` the index is mapped to the original pixel centre with
    ``downscale * index + (downscale - 1) / 2``.

    The scalar depth map is repeated into the two channels :func:`bench.flow_prediction.sample_field`
    expects, so bilinear interpolation matches the flow field's sampler exactly. A sample whose centre lies
    outside the original depth image (``x < 0``, ``x > W-1``, ``y < 0`` or ``y > H-1``) is returned as
    ``nan`` instead of being border-clamped: those are padded RAFT feature cells with no real depth evidence.
    Returns a ``[feature_h, feature_w]`` ``float64`` array. Raises ``ValueError`` for a non-2D depth map,
    negative feature dimensions, a ``pad`` that is not four values, or ``downscale < 1``.
    """
    depth = np.asarray(depth, np.float64)
    if depth.ndim != 2:
        raise ValueError("depth must be a 2D array")
    fh, fw = int(feature_h), int(feature_w)
    if fh <= 0 or fw <= 0:
        raise ValueError("feature_h and feature_w must be positive")
    pad = np.asarray(pad, np.float64).reshape(-1)
    if pad.size != 4:
        raise ValueError("pad must be (top, bottom, left, right)")
    d = int(downscale)
    if d < 1:
        raise ValueError("downscale must be >= 1")
    H, W = depth.shape
    offset = (d - 1) / 2.0
    fxs, fys = np.meshgrid(np.arange(fw), np.arange(fh), indexing="xy")
    pre_x = FEATURE_STRIDE * fxs - pad[2]                       # pad_left
    pre_y = FEATURE_STRIDE * fys - pad[0]                       # pad_top
    px = d * pre_x + offset
    py = d * pre_y + offset
    pts = np.stack([px.ravel(), py.ravel()], axis=1)
    # A sample centre outside the original image is a padded feature cell with no depth evidence: keep the
    # bilinear sampler for real centres but force ``nan`` where the centre left the image instead of letting
    # ``sample_field`` border-clamp it onto an edge pixel.
    inside = FP.in_bounds(pts, W, H)
    field = np.repeat(depth[:, :, None], 2, axis=2)             # scalar depth in both channels
    sampled = FP.sample_field(field, pts, W, H)
    out = np.where(inside, sampled[:, 0], np.nan)
    return out.reshape(fh, fw)


def _source_span(values: np.ndarray) -> Optional[float]:
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return None
    p10, p90 = np.percentile(v, [10, 90])
    span = float(p90 - p10)
    return span if span > SOURCE_SPAN_EPS else None


def _ols_affine(src: np.ndarray, dst: np.ndarray) -> Optional[tuple]:
    """OLS affine ``dst ~ scale * src + offset``; ``None`` when unidentifiable.

    Returns ``(scale, offset)`` for the unique least-squares solution, or ``None`` when ``src``/``dst`` have
    fewer than ``MIN_SAMPLES`` points or the ``src`` samples have no span (their total sum of squares does not
    exceed ``FIT_TARGET_SPAN_EPS``), which would make the two-column design singular.
    """
    src = np.asarray(src, np.float64).ravel()
    dst = np.asarray(dst, np.float64).ravel()
    if src.size < MIN_SAMPLES:
        return None
    tss = float(np.sum((src - src.mean()) ** 2))
    if not (tss > FIT_TARGET_SPAN_EPS):
        return None
    design = np.stack([src, np.ones_like(src)], axis=1)
    coeff, *_ = np.linalg.lstsq(design, dst, rcond=None)
    return float(coeff[0]), float(coeff[1])


def _robust_affine_fit(src: np.ndarray, dst: np.ndarray) -> dict:
    """Deterministic MAD-rejection affine fit ``dst ~ scale * src + offset``.

    Starts from ordinary least squares on ``[src, 1]`` and, for at most ``MAD_MAX_ITERS`` iterations, refits
    on the samples whose absolute residual is within ``MAD_OUTLIER_SIGMA`` times the robust residual sigma
    (``1.4826 * median absolute deviation`` of the current residuals). The loop also stops early when that
    robust sigma collapses to the residual noise floor (an exact fit) or when no sample is rejected. There is
    no randomness and no seed, so the fit is reproducible. Robust to a **minority** of outliers; a majority of
    outliers can inflate the MAD and sway the fit, so ``fit_inlier_count`` is reported to expose the surviving
    support.

    ``src``/``dst`` are equal-length finite 1-D arrays over the full support. Returns ``scale``, ``offset``
    (both ``None`` when unidentifiable) and ``fit_inlier_count`` (the number of samples used by the final
    refit, ``None`` when the fit failed). Fails safely -- returning all-``None`` -- when the initial OLS is
    unidentifiable or when clipping leaves fewer than ``MIN_SAMPLES`` inliers.
    """
    src = np.asarray(src, np.float64).ravel()
    dst = np.asarray(dst, np.float64).ravel()
    failed = {"scale": None, "offset": None, "fit_inlier_count": None}
    fit = _ols_affine(src, dst)
    if fit is None:
        return failed
    scale, offset = fit
    inlier_count = int(src.size)
    for _ in range(MAD_MAX_ITERS):
        resid = np.abs(scale * src + offset - dst)
        mad = float(np.median(np.abs(resid - np.median(resid))))
        sigma = 1.4826 * mad
        # A numerically exact fit leaves sigma at floating-point noise. Treat a sigma that is negligible
        # relative to the *data* scale (~|scale*src+offset|) as "no outliers", so the loop stops instead of
        # trimming true inliers on noise; the threshold is far below any meaningful outlier residual.
        data_scale = float(np.median(np.abs(scale * src + offset)))
        sigma_floor = _EXACT_FIT_REL_EPS * max(1.0, data_scale)
        if not np.isfinite(sigma) or sigma <= sigma_floor:
            break                                   # exact fit or degenerate spread: keep current fit
        keep = resid <= MAD_OUTLIER_SIGMA * sigma
        if keep.all():
            break                                   # no outliers to drop
        if int(keep.sum()) < MIN_SAMPLES:
            return failed                           # clipping destroyed the support
        refit = _ols_affine(src[keep], dst[keep])
        if refit is None:
            return failed
        scale, offset = refit
        inlier_count = int(keep.sum())
    return {"scale": scale, "offset": offset, "fit_inlier_count": inlier_count}


def _method_metrics(src: np.ndarray, dst: np.ndarray, span: Optional[float]) -> dict:
    """Support count/fraction and normalized residual stats for one same-support method.

    ``src`` and ``dst`` are the source depth and the candidate target depth (same-index or flow-warped) over
    the common support. The residual is ``|(scale * dst + offset) - src|`` after the robust affine fit that
    maps the *candidate target* samples back onto the *source*. The fit may drop MAD outliers, but the residual
    statistics are evaluated on the **full** support so every outlier stays visible; ``fit_inlier_count``
    reports how many samples the final fit actually used. Normalized stats divide by ``span`` (the source-depth
    p90-p10), so they are unitless; raw median/p95 are also reported in depth units. Returns explicit ``None``
    metrics with a ``reason`` when the support is too small or the fit is unidentifiable.
    """
    n = int(src.size)
    base = {"n": n, "median_abs_residual": None, "p95_abs_residual": None,
            "median_normalized_residual": None, "p95_normalized_residual": None,
            "fit_scale": None, "fit_offset": None, "fit_inlier_count": None, "reason": None}
    if n < MIN_SAMPLES:
        base["reason"] = "insufficient_support"
        return base
    fit = _robust_affine_fit(dst, src)          # target -> source
    if fit["scale"] is None:
        base["reason"] = "unidentifiable_fit"
        return base
    # Residuals are always evaluated on the full support, not the fit's inlier subset, so rejected outliers
    # remain visible in the reported statistics.
    resid = np.abs(fit["scale"] * dst + fit["offset"] - src)
    median = float(np.median(resid))
    p95 = float(np.percentile(resid, 95))
    base["median_abs_residual"] = median
    base["p95_abs_residual"] = p95
    if span is not None:
        base["median_normalized_residual"] = median / span
        base["p95_normalized_residual"] = p95 / span
    base["fit_scale"] = fit["scale"]
    base["fit_offset"] = fit["offset"]
    base["fit_inlier_count"] = fit["fit_inlier_count"]
    return base


def _support_tier(src: np.ndarray, same: np.ndarray, warped: np.ndarray,
                  span: Optional[float]) -> dict:
    """One support tier: counts/fraction plus the same-index and flow-warped method metrics."""
    n = int(src.size)
    return {
        "support_count": n,
        "support_fraction": None,               # filled by the caller, which knows the total cell count
        "same_index": _method_metrics(src, same, span),
        "flow_warp": _method_metrics(src, warped, span),
    }


def relative_depth_pair_audit(depth_src: np.ndarray, depth_dst: np.ndarray, fwd_flow: np.ndarray,
                              cycle_resid: np.ndarray, cycle_valid: np.ndarray,
                              cycle_thresholds: Sequence[float] = ()) -> dict:
    """Compare flow-warped vs same-index target depth on one shared, affine-normalized support.

    All inputs are **feature-grid** shaped. ``depth_src``/``depth_dst`` are ``[H,W]`` relative-depth maps
    (already sampled onto the feature grid, e.g. by :func:`depth_to_feature_grid`); ``fwd_flow`` is the
    ``[H,W,2]`` (x, y) forward flow mapping cell ``p`` to ``q = p + fwd_flow[p]``; ``cycle_resid`` is the
    ``[H,W]`` forward cycle residual and ``cycle_valid`` the ``[H,W]`` forward cycle-valid mask (from
    :func:`bench.flow_prediction_real.cycle_residual_fields`).

    The base support excludes cells where the flow is non-finite, the mapped point ``q`` leaves the feature
    grid (edge rule ``[-0.5, W-0.5]`` / ``[-0.5, H-0.5]``, the same as ``flow_prediction.in_bounds``),
    either depth is non-finite, or ``cycle_valid`` is False. For each fixed caller threshold the audit keeps
    the subset with ``cycle_resid <= threshold`` and reports its count/fraction; it always also reports the
    ``all_cycle_valid`` tier (no residual gate). On the **same** support each tier compares, after an
    independent robust affine target-to-source fit, (a) target depth at the same grid index and (b) target
    depth bilinearly sampled at ``p + fwd_flow`` using :func:`bench.flow_prediction.sample_field`.

    Returns a JSON-safe dict with the base geometry, ``source_span`` (the p90-p10 normalizer), and
    ``tiers``: the ``all_cycle_valid`` tier first, then one per threshold in caller order (keys
    ``cycle_le_<threshold>``). ``cycle_thresholds`` must be finite and nonnegative, and no two may format to
    the same tier key; otherwise ``ValueError`` is raised so an invalid or duplicate key cannot silently
    overwrite another tier. Every method
    metric reports support count/fraction and raw + normalized median/p95 absolute residual, the fitted scale
    and offset plus ``fit_inlier_count`` (the samples the robust fit kept), or explicit ``None`` metrics with a
    ``reason``. ``cycle_valid`` on the source grid is the pairing mask; no threshold is selected or
    recommended and the fitted scale/offset discard depth scale, so nothing here is a metric-depth or
    accuracy claim.
    """
    src = np.asarray(depth_src, np.float64)
    dst = np.asarray(depth_dst, np.float64)
    flow = np.asarray(fwd_flow, np.float64)
    resid = np.asarray(cycle_resid, np.float64)
    valid = np.asarray(cycle_valid, bool)
    if src.ndim != 2:
        raise ValueError("depth_src must be a 2D feature-grid array")
    H, W = src.shape
    if dst.shape != (H, W):
        raise ValueError("depth_dst shape must match depth_src")
    if flow.shape != (H, W, 2):
        raise ValueError("fwd_flow must be [H, W, 2]")
    if resid.shape != (H, W):
        raise ValueError("cycle_resid shape must match depth_src")
    if valid.shape != (H, W):
        raise ValueError("cycle_valid shape must match depth_src")

    # Validate thresholds up front: a non-finite or negative gate is meaningless, and duplicate keys would
    # collide in ``tiers`` (``cycle_le_{t:g}``), silently overwriting an earlier tier. Detect collisions on
    # the formatted key, not the raw value, since that is what indexes ``tiers``. Preserve caller order.
    thresholds: list = []
    seen_keys = set()
    for th in cycle_thresholds:
        t = float(th)
        if not np.isfinite(t):
            raise ValueError("cycle_thresholds must be finite")
        if t < 0.0:
            raise ValueError("cycle_thresholds must be nonnegative")
        key = f"cycle_le_{t:g}"
        if key in seen_keys:
            raise ValueError(f"duplicate cycle_threshold: {t:g}")
        seen_keys.add(key)
        thresholds.append(t)

    n_total = H * W
    grid = FP.source_grid(W, H)
    finite_flow = np.isfinite(flow).reshape(n_total, 2).all(axis=1)
    safe_flow = np.where(finite_flow[:, None], flow.reshape(n_total, 2), 0.0)
    mapped = grid + safe_flow
    inb = finite_flow & FP.in_bounds(mapped, W, H)
    # A non-finite depth tap with zero bilinear weight would still poison the sample because ``0 * NaN`` is
    # NaN, so sanitize the field before sampling and reject any cell whose taps touch a non-finite depth via
    # the conservative mask (the same rule ``cycle_residual_fields`` uses for its sampled operand).
    dst_finite = np.isfinite(dst)
    safe_dst = np.where(dst_finite, dst, 0.0)
    sampled = FP.sample_field(np.repeat(safe_dst[:, :, None], 2, axis=2), mapped, W, H)[:, 0]
    tap_ok = FP.sample_mask_conservative(dst_finite, mapped, W, H)
    sampled = np.where(finite_flow & tap_ok, sampled, np.nan)
    same_flat = dst.reshape(n_total)
    src_flat = src.reshape(n_total)
    resid_flat = resid.reshape(n_total)
    valid_flat = valid.reshape(n_total)
    finite_base = (np.isfinite(src_flat) & np.isfinite(same_flat) & np.isfinite(sampled)
                   & np.isfinite(resid_flat))
    base_support = inb & valid_flat & finite_base

    span = _source_span(src_flat[base_support])

    def build_tier(mask: np.ndarray, key: str, threshold: Optional[float]) -> dict:
        s = src_flat[mask]
        tier = _support_tier(s, same_flat[mask], sampled[mask], span)
        tier["support_fraction"] = float(tier["support_count"] / n_total) if n_total else 0.0
        tier["cycle_threshold"] = threshold
        tier["key"] = key
        return tier

    tiers = {"all_cycle_valid": build_tier(base_support, "all_cycle_valid", None)}
    for t in thresholds:
        mask = base_support & (resid_flat <= t)
        tiers[f"cycle_le_{t:g}"] = build_tier(mask, f"cycle_le_{t:g}", t)

    return {
        "feature_h": H,
        "feature_w": W,
        "n_cells": n_total,
        "base_support_count": int(base_support.sum()),
        "base_support_fraction": float(base_support.sum() / n_total) if n_total else 0.0,
        "source_span": span,
        "tiers": tiers,
        "notes": [
            "affine scale/offset is fitted per method and discards metric depth information",
            "support filters (finite/bounds/cycle) are diagnostic only; no threshold is selected",
            "residuals are relative self-consistency, not depth accuracy",
        ],
    }
