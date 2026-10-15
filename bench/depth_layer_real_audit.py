"""Pure-NumPy downstream geometry audit for real-footage RAFT/depth observations.

This compares one pooled affine flow fit with neutral per-frame relative-depth-rank
fits. It measures implied affine paths and inverse-warp coverage; it does not render
images or establish camera motion, depth accuracy, or perceptual stabilization.

``audit_depth_layer_sweep`` runs the same per-window audit across a declared grid of
bin counts and smoothing sigmas and returns every variant. It scores and selects
nothing; the sweep is a diagnostic enumeration, not a tuning result.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

SCHEMA = "mlxstab.depth_layer_real_audit/1"
SWEEP_SCHEMA = "mlxstab.depth_layer_real_audit_sweep/1"
LIMITATIONS = [
    "flow and monocular depth are model-derived; neither is ground truth",
    "depth strata are per-frame relative-rank bins, not metric depth or near/far labels",
    "an affine fit to scene flow is only a camera-motion proxy and may include parallax or moving objects",
    "the Gaussian-smoothed path and its sigma are diagnostic choices, not a selected production policy",
    "cycle-valid means the bidirectional cycle sample is finite and in bounds; no residual threshold is selected",
    "coverage is geometric and uses no image rendering, texture, or perceptual assessment",
    "each result describes one supplied source window; windows must be audited independently",
]
MIN_FIT_SAMPLES = 6
SWEEP_LIMITATION = (
    "the sweep enumerates every requested bin-count/smoothing combination independently; "
    "no combination is scored, ranked or selected as a winner"
)


def _finite_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _fit(depth, flow, valid, *, rank_map=None, rank_id=None):
    h, w = depth.shape
    support = valid & np.isfinite(depth) & np.isfinite(flow).all(axis=2)
    if rank_map is not None:
        support &= rank_map == int(rank_id)
    yy, xx = np.mgrid[:h, :w]
    design = np.stack((xx[support] / max(w - 1, 1), yy[support] / max(h - 1, 1),
                       np.ones(int(support.sum()))), axis=1)
    if design.shape[0] < MIN_FIT_SAMPLES or np.linalg.matrix_rank(design) < 3:
        return None, int(support.sum()), None
    coeff, *_ = np.linalg.lstsq(design, flow[support], rcond=None)
    predicted = design @ coeff
    resid = np.linalg.norm(flow[support] - predicted, axis=1)
    # coeff rows are x, y, translation; columns are u, v.
    T = np.array([[1.0 + coeff[0, 0] / max(w - 1, 1),
                   coeff[1, 0] / max(h - 1, 1), coeff[2, 0]],
                  [coeff[0, 1] / max(w - 1, 1),
                   1.0 + coeff[1, 1] / max(h - 1, 1), coeff[2, 1]],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    if not np.isfinite(T).all():
        return None, int(support.sum()), None
    return T, int(support.sum()), _finite_float(np.median(resid))


def _rank_bins(depth, support, n_bins):
    """Assign stable equal-count quantile ranks; equal values stay in the same bin."""
    use = support & np.isfinite(depth)
    values = depth[use]
    assigned = np.full(depth.shape, -1, np.int64)
    if not values.size:
        return assigned, []
    ordered = np.sort(values, kind="stable")
    positions = np.ceil(np.arange(1, n_bins) * (len(ordered) / n_bins)).astype(np.int64)
    positions = np.clip(positions, 1, len(ordered) - 1)
    edges = np.unique(ordered[positions - 1]) if len(ordered) > 1 else np.empty(0)
    assigned[use] = np.searchsorted(edges, values, side="left")
    return assigned, sorted(int(i) for i in np.unique(assigned[use]))


def _path_summary(transforms, h, w, sigma):
    """Return component-smoothed path jitter and geometric inverse-warp coverage."""
    if any(t is None for t in transforms):
        return {"available": False, "reason": "at least one pair transform is unavailable",
                "jitter_px": None, "frames": []}
    path = [np.eye(3, dtype=np.float64)]
    for t in transforms:
        path.append(t @ path[-1])
    raw = np.asarray([[c[0, 0] - 1, c[0, 1], c[0, 2], c[1, 0], c[1, 1] - 1, c[1, 2]]
                      for c in path], np.float64)
    if sigma > 0:
        radius = max(1, int(np.ceil(4 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        smooth = np.stack([np.convolve(np.pad(raw[:, k], (radius, radius), mode="edge"), kernel,
                                       mode="valid") for k in range(6)], axis=1)
        smoothed_path = []
        for p in smooth:
            smoothed_path.append(np.array([[1 + p[0], p[1], p[2]], [p[3], 1 + p[4], p[5]],
                                           [0, 0, 1]], np.float64))
    else:
        # Preserve the exact matrices: reconstructing I + (C-I) can introduce tiny
        # round-off at image edges and falsely lower validity for an identity correction.
        smoothed_path = path
    points = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1],
                       [0, h - 1, 1], [(w - 1) / 2, (h - 1) / 2, 1]], np.float64).T
    deviations = []
    frames = []
    gy, gx = np.mgrid[:h, :w]
    target = np.stack((gx.ravel(), gy.ravel(), np.ones(h * w)), axis=0)
    for raw_c, smooth_c in zip(path, smoothed_path):
        deviations.append((raw_c @ points)[:2] - (smooth_c @ points)[:2])
        try:
            correction = np.eye(3, dtype=np.float64) if sigma == 0 else smooth_c @ np.linalg.inv(raw_c)
            source = np.linalg.inv(correction) @ target
            x, y = source[0], source[1]
            finite = np.isfinite(source).all(axis=0)
            inbounds = finite & (x >= -0.5) & (x <= w - 0.5) & (y >= -0.5) & (y <= h - 0.5)
            # A conservative bilinear render requires every contributing source tap in the image.
            rendered = finite & (x >= 0.0) & (x <= w - 1.0) & (y >= 0.0) & (y <= h - 1.0)
            frames.append({"inbounds_coverage": float(inbounds.mean()),
                           "rendered_valid_coverage": float(rendered.mean())})
        except np.linalg.LinAlgError:
            frames.append({"inbounds_coverage": None, "rendered_valid_coverage": None})
    if len(deviations) > 1:
        jitter = float(np.mean(np.abs(np.diff(np.stack(deviations), axis=0))))
    else:
        jitter = 0.0
    vals_in = [f["inbounds_coverage"] for f in frames if f["inbounds_coverage"] is not None]
    vals_r = [f["rendered_valid_coverage"] for f in frames if f["rendered_valid_coverage"] is not None]
    return {"available": True, "reason": None, "jitter_px": _finite_float(jitter), "frames": frames,
            "mean_inbounds_coverage": float(np.mean(vals_in)) if vals_in else None,
            "mean_rendered_valid_coverage": float(np.mean(vals_r)) if vals_r else None}


def audit_depth_layer_window(observations: Sequence[dict], *, n_bins: int = 4,
                             smooth_sigma: float = 1.0) -> dict:
    """Compare pooled and relative-depth-rank affine motion policies for one window.

    Each observation contains ``depth[H,W]``, ``flow[H,W,2]`` and
    ``cycle_valid[H,W]``; optional ``cycle_resid[H,W]`` is accepted for provenance
    but is not thresholded. The same finite-depth, cycle-valid cells support every
    policy. Rank bins are recomputed per pair, and rank 0 means the lowest numerical
    values under that pair's depth convention only.
    """
    if not observations:
        raise ValueError("observations must contain at least one pair")
    n_bins = int(n_bins)
    smooth_sigma = float(smooth_sigma)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if not np.isfinite(smooth_sigma) or smooth_sigma < 0:
        raise ValueError("smooth_sigma must be finite and >= 0")
    obs = []
    shape = None
    for i, item in enumerate(observations):
        try:
            depth = np.asarray(item["depth"], np.float64)
            flow = np.asarray(item["flow"], np.float64)
            valid = np.asarray(item["cycle_valid"], bool)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"observation {i} requires depth, flow, cycle_valid") from exc
        if depth.ndim != 2 or min(depth.shape) < 2:
            raise ValueError(f"observation {i} depth must be [H,W] with H,W >= 2")
        if flow.shape != (*depth.shape, 2) or valid.shape != depth.shape:
            raise ValueError(f"observation {i} flow/cycle_valid shapes must match depth")
        if shape is not None and depth.shape != shape:
            raise ValueError("all observations in a window must share one feature-grid shape")
        shape = depth.shape
        if "cycle_resid" in item and np.asarray(item["cycle_resid"]).shape != depth.shape:
            raise ValueError(f"observation {i} cycle_resid shape must match depth")
        obs.append((depth, flow, valid))

    h, w = shape
    policies = {"pooled": []}
    policies.update({f"relative_depth_rank_{rank_id}": [] for rank_id in range(n_bins)})
    rank_sets = []
    pair_details = []
    for depth, flow, valid in obs:
        rank_support = valid & np.isfinite(depth) & np.isfinite(flow).all(axis=2)
        ranks, ids = _rank_bins(depth, rank_support, n_bins)
        rank_sets.append(ids)
        pooled_t, pooled_n, pooled_resid = _fit(depth, flow, valid)
        policies["pooled"].append(pooled_t)
        details = {"pooled": {"support": pooled_n, "median_fit_residual_px": pooled_resid,
                              "available": pooled_t is not None}, "rank_bins": []}
        rank_results = {}
        for rank_id in range(n_bins):
            t, n, residual = _fit(depth, flow, valid, rank_map=ranks, rank_id=rank_id)
            policies[f"relative_depth_rank_{rank_id}"].append(t)
            if rank_id in ids:
                rank_results[rank_id] = (t, n, residual)
        for rank_id in ids:
            t, n, residual = rank_results[rank_id]
            details["rank_bins"].append({"rank_id": rank_id, "support": n,
                                         "median_fit_residual_px": residual, "available": t is not None})
        pair_details.append(details)
    # A rank policy missing from a given pair is explicitly unavailable for the whole path.
    summaries = {name: _path_summary(transforms, h, w, smooth_sigma)
                 for name, transforms in policies.items()}
    return {"schema": SCHEMA, "pair_count": len(obs), "feature_grid": [h, w], "n_bins_requested": n_bins,
            "rank_ids_per_pair": rank_sets, "smoothing": {"method": "component-wise Gaussian",
            "sigma_pairs": smooth_sigma, "boundary": "nearest"}, "limitations": list(LIMITATIONS),
            "pairs": pair_details, "policies": summaries}


def _sweep_bin_counts(bin_counts) -> list:
    """Normalise ``bin_counts`` to a nonempty list of positive ``int`` values or raise ``ValueError``."""
    if isinstance(bin_counts, (str, bytes)):
        raise ValueError("bin_counts must be a nonempty sequence of positive integers")
    try:
        values = list(bin_counts)
    except TypeError as exc:
        raise ValueError("bin_counts must be a nonempty sequence of positive integers") from exc
    if not values:
        raise ValueError("bin_counts must be a nonempty sequence of positive integers")
    checked = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
            raise ValueError("bin_counts must be a nonempty sequence of positive integers")
        checked.append(int(value))
    return list(dict.fromkeys(checked))


def _sweep_sigmas(smooth_sigmas) -> list:
    """Normalise ``smooth_sigmas`` to a nonempty list of finite nonnegative floats."""
    if isinstance(smooth_sigmas, (str, bytes)):
        raise ValueError("smooth_sigmas must be a nonempty sequence of finite nonnegative numbers")
    try:
        values = list(smooth_sigmas)
    except TypeError as exc:
        raise ValueError("smooth_sigmas must be a nonempty sequence of finite nonnegative numbers") from exc
    if not values:
        raise ValueError("smooth_sigmas must be a nonempty sequence of finite nonnegative numbers")
    checked = []
    for value in values:
        try:
            sigma = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "smooth_sigmas must be a nonempty sequence of finite nonnegative numbers") from exc
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("smooth_sigmas must be a nonempty sequence of finite nonnegative numbers")
        checked.append(sigma)
    return list(dict.fromkeys(checked))


def validate_sweep_options(bin_counts=(1, 2, 4, 8), smooth_sigmas=(0, 0.5, 1, 2)):
    """Validate a sweep grid and return ``(bin_counts, smooth_sigmas)`` as normalised lists.

    ``bin_counts`` must be a nonempty sequence of positive integers; ``smooth_sigmas`` must be
    a nonempty sequence of finite, nonnegative numbers. Raises ``ValueError`` otherwise.
    """
    return _sweep_bin_counts(bin_counts), _sweep_sigmas(smooth_sigmas)


def sweep_variant_key(n_bins, smooth_sigma) -> str:
    """Stable, filesystem/JSON-safe key for one ``(n_bins, smooth_sigma)`` variant."""
    return f"bins_{int(n_bins)}__sigma_{float(smooth_sigma):g}"


def audit_depth_layer_sweep(observations: Sequence[dict], *, bin_counts=(1, 2, 4, 8),
                            smooth_sigmas=(0, 0.5, 1, 2)) -> dict:
    """Run :func:`audit_depth_layer_window` independently for every grid combination.

    Each Cartesian ``(n_bins, smooth_sigma)`` pair is audited on the same supplied window
    and the full per-window report is kept verbatim under ``variants``, keyed by a stable
    string such as ``"bins_4__sigma_1"``. Every window (the single supplied observation
    sequence) is preserved as its own report; no pooling, scoring, ranking or winner
    selection is performed. The returned structure is JSON-safe (finite floats only).
    """
    bins, sigmas = validate_sweep_options(bin_counts, smooth_sigmas)
    variants = {}
    for n_bins in bins:
        for sigma in sigmas:
            variants[sweep_variant_key(n_bins, sigma)] = audit_depth_layer_window(
                observations, n_bins=n_bins, smooth_sigma=sigma)
    first = next(iter(variants.values()))
    return {"schema": SWEEP_SCHEMA, "pair_count": first["pair_count"],
            "feature_grid": list(first["feature_grid"]), "bin_counts": list(bins),
            "smooth_sigmas": list(sigmas), "variants": variants,
            "limitations": list(LIMITATIONS) + [SWEEP_LIMITATION]}
