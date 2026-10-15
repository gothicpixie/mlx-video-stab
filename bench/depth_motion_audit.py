"""Isolated, pure-NumPy diagnostic: does stratifying a RAFT flow field by *relative-depth rank* bins make
the motion field more explainable than one pooled affine model?

This is an **offline model-derived flow-structure diagnostic**. It asks one narrow, falsifiable question:

    On a checkerboard split of the feature grid, is a per-depth-bin 2-D affine displacement model fitted on
    the *train* cells a better predictor of the *held-out* flow cells than a single pooled affine model
    fitted on all train cells?

It does **not** run RAFT, decode media, download weights, touch the heavy lock, read a manifest, render
frames or import production stabilization code. It is deliberately **not**: a camera-motion ground truth
(RAFT flow is model-derived), a metric-depth or near/far assertion (the depth is per-frame *relative* rank
only), or production stabilization evidence. No threshold is selected and no default is changed.

Model
-----
Each cell ``p = (x, y)`` carries a source relative-depth value and a 2-D forward displacement
``(u, v) = fwd_flow[p]``. The displacement is modelled as a linear function of the cell's normalized source
coordinates ``(x / W, y / H, 1)``, with **independent** OLS coefficients for ``u`` and ``v`` (one design
matrix per direction, a plain ``numpy.linalg.lstsq`` with ``rcond=None`` -- no random seeding, no RANSAC).
The *global* model is fitted on every training-support cell; a *depth-bin* model is fitted on that bin's
training cells only. The global residual is evaluated on validation cells grouped by that same bin, so the
two are directly comparable on the identical validation cell set.

Split and comparability
-----------------------
The spatial split is deterministic: cell parity of ``x + y`` -- even cells train, odd cells validate -- so
train and validation cells are interleaved and each bin keeps a held-out subset. The comparison is asked
*beyond* the pooled model by reporting both the pooled-held-out residual within each bin and the
bin-specific-held-out residual, plus their delta (global minus bin, so a positive delta means the bin model
is lower/better). Residuals are Euclidean in feature-grid pixels.

Depth binning
-------------
Source relative depth is divided into equal-count quantile bins (default 4) over the eligible cells.
Quantile boundaries may be computed from **all eligible depth values**, including validation cells
(unlabeled, unsupervised split): that is a documented **leakage limitation** -- the bin edges see the
validation depth distribution. Constant or heavily tied depth collapses to fewer effective bins and is
handled safely (edges and effective bin count are reported; empty/1-value bins are skipped, never asserted
as near/far). The values are **per-frame relative-depth ranks only**; nothing here asserts a true metric
near/far.

Limitation: checkerboard neighbours are spatially correlated, so train/validation cells are not independent
samples; the held-out residual is a structural diagnostic, not a generalisation guarantee.

Support and thresholds
----------------------
Per-tier support is finite depth + finite bounded flow + ``cycle_valid``; on top of the base tier, each
caller threshold ``t`` keeps cells with ``cycle_resid <= t``. Defaults ``0.25, 0.5, 1, 2`` are diagnostic
gates only -- **no threshold is selected or recommended**. Degenerate supports (a tier with too few cells,
a fit whose design is singular, a bin with no held-out cells) return explicit ``None`` metrics with a
machine-readable reason rather than ``nan``/``inf``.

Runner
------
``python -m bench.depth_motion_audit`` rebuilds the same audit from three cached inputs (a decoded-frame
``.npz``, a cached relative-depth ``.npz`` and a local RAFT weights file). It encodes the sampled frames once
under the shared :class:`bench.heavylock.HeavyLock`, calls ``RaftRunner.reference_pair`` and
``cycle_residual_fields`` per adjacent pair, samples each depth map onto the feature grid with
``depth_to_feature_grid``, and audits **both** directions (the backward audit uses ``depth_dst`` as its
source). Like :mod:`bench.depth_temporal_pilot`, it never pools data across different source windows and
writes a **path-independent** (hashes only), atomic, finite-safe JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import flow_prediction_real as R
from . import manifest as MF
from .depth_temporal_audit import depth_to_feature_grid
from .depth_temporal_pilot import (DEFAULT_CYCLE_THRESHOLDS, DEFAULT_DOWNSCALE, DEFAULT_ITERS,
                                   load_depth_npz, load_frames_npz, validate_cache_correspondence)

SCHEMA = "mlxstab.depth_motion_audit/1"
DEFAULT_BINS = 4
DEFAULT_WEIGHTS = R.DEFAULT_WEIGHTS

# Reporting floor for any pooled or per-bin model. A 2-D affine fit is only *identifiable* with 3
# non-collinear cells, but a contract of 6 train cells plus 3 held-out validation cells is required before a
# fit/report is trusted, so a single-cell held-out block can never look like an accidental improvement.
MIN_FIT_SAMPLES = 6
MIN_EVAL_SAMPLES = 3

LIMITATIONS = (
    "one sampled source window and a fixed set of sampled frames; not clip-wide, held-out or multi-scene, "
    "and data are never pooled across different source windows",
    "RAFT flow is model-derived, so this measures how well a fitted displacement field explains RAFT output; "
    "it is not camera-motion ground truth and not optical-flow accuracy",
    "per-frame monocular depth is a relative-rank field only; depth bins are rank strata, not a metric "
    "near/far claim and not a depth-accuracy benchmark",
    "quantile bin edges may be computed over all eligible depth values including validation cells "
    "(unsupervised split), which leaks the validation depth distribution into the bin boundaries",
    "checkerboard train/validation cells are spatially correlated neighbours, so the held-out residual is a "
    "structure diagnostic, not an independent-sample generalisation guarantee",
    "support filters (finite/bounds/cycle) are diagnostic; no cycle threshold is selected or recommended and "
    "no stabilization output is compared; no speed or quality claim is made",
)


# --------------------------------------------------------------------------------------
# pure-NumPy core
# --------------------------------------------------------------------------------------

def _bin_assignments(depth_flat: np.ndarray, support: np.ndarray, n_bins: int):
    """Equal-count quantile bins over the eligible depth values (unsupervised split).

    Returns ``(assign, edges)`` where ``assign`` is an int array over all cells (``-1`` outside ``support`` or
    in an unused/empty bin) and ``edges`` are the interior quantile edges actually used. Quantile boundaries
    are computed from **all eligible (support) depth values**, validation cells included -- the documented
    leakage limitation. Constant/tied depth naturally yields fewer effective bins; empty quantile slices are
    dropped rather than merged, so an all-equal depth field reports one effective bin.
    """
    n = int(depth_flat.size)
    assign = np.full(n, -1, np.int64)
    elig_vals = depth_flat[support]
    elig_idx = np.flatnonzero(support)
    if elig_vals.size == 0 or n_bins < 1:
        return assign, []
    order = np.argsort(elig_vals, kind="stable")
    vals_sorted = elig_vals[order]
    # Interior quantile edge **positions** in the sorted array: slice ``s`` of the equal-count partition
    # holds ``vals_sorted[lo:hi]``; the edge value is the last value of each non-final slice. Using exact
    # integer positions (ceil of the cumulative boundaries) avoids any floating-point epsilon in the index.
    knots = np.ceil(np.arange(1, n_bins) * (vals_sorted.size / n_bins)).astype(np.int64)
    knots = np.clip(knots, 1, vals_sorted.size - 1)
    bounds = vals_sorted[knots - 1]
    if bounds.size == 0:
        assign[elig_idx] = 0
        return assign, []
    # searchsorted on the raw values gives the bin index; ``side="left"`` keeps a value exactly equal to an
    # interior edge in the lower bin (the first position whose bound is >= the value).
    bin_idx = np.searchsorted(bounds, elig_vals, side="left")
    assign[elig_idx] = bin_idx
    # Drop empty bins by relabeling the surviving ones, preserving depth order.
    used = sorted(set(int(b) for b in bin_idx))
    remap = {b: k for k, b in enumerate(used)}
    out = np.full(n, -1, np.int64)
    for b in used:
        out[elig_idx[bin_idx == b]] = remap[b]
    distinct_edges = sorted(set(float(v) for v in bounds))
    return out, distinct_edges


def _design(coords_flat: np.ndarray, mask: np.ndarray, W: int, H: int) -> Optional[np.ndarray]:
    """Design matrix ``[x/W, y/H, 1]`` for the cells selected by ``mask``; ``None`` if empty."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    pts = coords_flat[idx]
    x = pts[:, 0] / float(max(W, 1))
    y = pts[:, 1] / float(max(H, 1))
    return np.stack([x, y, np.ones_like(x)], axis=1)


def _affine_fit(design: np.ndarray, target: np.ndarray) -> Optional[tuple]:
    """OLS for one displacement component: returns ``coeff [3]`` or ``None`` when unidentifiable.

    ``design`` is ``[n,3]`` (normalized x, y, 1) and ``target`` is ``[n]``. Unidentifiable when there are
    fewer than :data:`MIN_FIT_SAMPLES` rows or the design is rank-deficient (e.g. all cells collinear, a
    constant-depth bin sitting on one checkerboard parity).
    """
    if design.shape[0] < MIN_FIT_SAMPLES:
        return None
    if np.linalg.matrix_rank(design) < 3:
        return None
    coeff, *_ = np.linalg.lstsq(design, target, rcond=None)
    return np.asarray(coeff, np.float64)


def _predicted(design: np.ndarray, cu: np.ndarray, cv: np.ndarray) -> np.ndarray:
    """Predicted ``(u, v)`` displacement ``[n,2]`` for ``design`` under coefficients ``cu``/``cv``."""
    return np.stack([design @ cu, design @ cv], axis=1)


def _resid_stats(resid: np.ndarray) -> dict:
    """Finite Euclidean-residual median/p95 plus count; ``None`` metrics when empty."""
    r = np.asarray(resid, np.float64).ravel()
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {"n": 0, "median": None, "p95": None}
    return {"n": int(r.size), "median": float(np.median(r)), "p95": float(np.percentile(r, 95))}


def _delta(global_stat, bin_stat) -> Optional[float]:
    """``global - bin`` for a residual statistic (positive = bin model lower/better); ``None`` if missing."""
    if global_stat is None or bin_stat is None:
        return None
    return float(global_stat - bin_stat)


def depth_motion_audit(source_depth: np.ndarray, fwd_flow: np.ndarray, cycle_resid: np.ndarray,
                       cycle_valid: np.ndarray, *, n_bins: int = DEFAULT_BINS,
                       cycle_thresholds: Sequence[float] = DEFAULT_CYCLE_THRESHOLDS) -> dict:
    """Audit whether depth-rank stratification explains the flow field beyond one pooled affine model.

    All inputs are **feature-grid** shaped. ``source_depth`` is the ``[H,W]`` relative-depth field (already
    sampled onto the feature grid, e.g. by :func:`depth_to_feature_grid`); ``fwd_flow`` is the ``[H,W,2]``
    ``(x, y)`` forward displacement; ``cycle_resid`` is the ``[H,W]`` forward cycle residual and
    ``cycle_valid`` the ``[H,W]`` cycle-valid mask (from ``cycle_residual_fields``).

    Support per tier is finite depth + finite flow + ``cycle_valid`` (base tier ``all_cycle_valid``); each
    threshold ``t`` keeps ``cycle_resid <= t``. A deterministic checkerboard split assigns even ``x + y``
    cells to train and odd cells to validation. A pooled affine displacement model is fitted on **all**
    train cells (per-component OLS on ``(x/W, y/H, 1)``); a per-bin model is fitted on the bin's train cells.
    Both are evaluated on the *same* held-out validation cells of each bin, and the pooled-vs-bin held-out
    median/p95 residual and their delta are reported per bin and aggregated.

    Returns a JSON-safe dict: source/bin geometry, ``tiers`` (``all_cycle_valid`` first, then
    ``cycle_le_<t>`` in caller order) each with pooled-model and per-bin held-out statistics, and notes.
    Raises ``ValueError`` for shape mismatches, non-finite/duplicate thresholds or ``n_bins < 1``. This asks
    whether depth stratification supports motion-field explanation; it is not an optical-flow accuracy claim.
    """
    src = np.asarray(source_depth, np.float64)
    flow = np.asarray(fwd_flow, np.float64)
    resid = np.asarray(cycle_resid, np.float64)
    valid = np.asarray(cycle_valid, bool)
    if src.ndim != 2:
        raise ValueError("source_depth must be a 2D feature-grid array")
    H, W = src.shape
    if flow.shape != (H, W, 2):
        raise ValueError("fwd_flow must be [H, W, 2]")
    if resid.shape != (H, W):
        raise ValueError("cycle_resid shape must match source_depth")
    if valid.shape != (H, W):
        raise ValueError("cycle_valid shape must match source_depth")
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

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
    src_flat = src.reshape(n_total)
    flow_flat = flow.reshape(n_total, 2)
    resid_flat = resid.reshape(n_total)
    valid_flat = valid.reshape(n_total)
    coords = R.FP.source_grid(W, H) if hasattr(R, "FP") else None
    if coords is None:  # pragma: no cover - R always exposes FP, keep a direct construction as a fallback
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        coords = np.stack([xs.ravel(), ys.ravel()], axis=1)
    finite_src = np.isfinite(src_flat)
    finite_flow = np.isfinite(flow_flat).all(axis=1)
    finite_resid = np.isfinite(resid_flat)
    # Base support mirrors the depth audit: finite source depth, finite flow, a finite cycle residual, and
    # cycle_valid. No residual gate here so this is the widest comparable set.
    base_support = finite_src & finite_flow & finite_resid & valid_flat
    # Checkerboard split on the source grid: even x+y train, odd x+y validate.
    parity = (coords[:, 0].astype(np.int64) + coords[:, 1].astype(np.int64)) % 2
    train_mask_all = parity == 0
    val_mask_all = parity == 1

    def build_tier(mask: np.ndarray, key: str, threshold: Optional[float]) -> dict:
        n_tier = int(mask.sum())
        train = mask & train_mask_all
        val = mask & val_mask_all
        assign, edges = _bin_assignments(src_flat, mask, n_bins)
        effective = int(assign.max()) + 1 if np.any(assign >= 0) else 0

        # Pooled model: all train cells of this tier.
        d_train = _design(coords, train, W, H)
        pooled_u = pooled_v = None
        pooled_ok = False
        if d_train is not None:
            pooled_u = _affine_fit(d_train, flow_flat[train, 0])
            pooled_v = _affine_fit(d_train, flow_flat[train, 1])
            pooled_ok = pooled_u is not None and pooled_v is not None

        bins = []
        for b in range(effective):
            bmask = assign == b
            b_train = bmask & train_mask_all
            b_val = bmask & val_mask_all
            entry = {
                "bin": int(b),
                "support_count": int(bmask.sum()),
                "train_count": int(b_train.sum()),
                "validation_count": int(b_val.sum()),
                "depth_min": None,
                "depth_max": None,
                "global_model": {"n": 0, "median": None, "p95": None, "fit_ok": False},
                "bin_model": {"n": 0, "median": None, "p95": None, "fit_ok": False},
                "median_delta": None,
                "p95_delta": None,
                "reason": None,
            }
            if entry["support_count"]:
                bvals = src_flat[bmask]
                entry["depth_min"] = float(np.min(bvals))
                entry["depth_max"] = float(np.max(bvals))

            # Both models are only scored on a bin whose held-out block reaches the validation floor.
            eval_ok = entry["validation_count"] >= MIN_EVAL_SAMPLES
            if not eval_ok:
                entry["reason"] = "insufficient_validation"

            # Global model evaluated on this bin's held-out cells.
            if pooled_ok and eval_ok:
                d_val = _design(coords, b_val, W, H)
                pred = _predicted(d_val, pooled_u, pooled_v)
                gstat = _resid_stats(np.linalg.norm(pred - flow_flat[b_val], axis=1))
                gstat["fit_ok"] = True
                entry["global_model"] = gstat

            # Bin-specific model fitted only on this bin's train cells, evaluated on its held-out cells.
            d_b_train = _design(coords, b_train, W, H)
            if d_b_train is not None:
                bu = _affine_fit(d_b_train, flow_flat[b_train, 0])
                bv = _affine_fit(d_b_train, flow_flat[b_train, 1])
                if bu is not None and bv is not None and eval_ok:
                    d_val = _design(coords, b_val, W, H)
                    pred = _predicted(d_val, bu, bv)
                    bstat = _resid_stats(np.linalg.norm(pred - flow_flat[b_val], axis=1))
                    bstat["fit_ok"] = True
                    entry["bin_model"] = bstat
                elif entry["reason"] is None:
                    entry["reason"] = "unidentifiable_bin_fit"
            elif entry["reason"] is None:
                entry["reason"] = "insufficient_bin_train"

            if entry["global_model"]["median"] is not None and entry["bin_model"]["median"] is not None:
                entry["median_delta"] = _delta(entry["global_model"]["median"],
                                               entry["bin_model"]["median"])
                entry["p95_delta"] = _delta(entry["global_model"]["p95"], entry["bin_model"]["p95"])
            bins.append(entry)

        # Aggregate over bins with both median statistics present (equal weight per bin, not per cell).
        g_med = [e["global_model"]["median"] for e in bins if e["global_model"]["median"] is not None]
        b_med = [e["bin_model"]["median"] for e in bins if e["bin_model"]["median"] is not None]
        deltas = [e["median_delta"] for e in bins if e["median_delta"] is not None]
        d_p95 = [e["p95_delta"] for e in bins if e["p95_delta"] is not None]
        return {
            "key": key,
            "cycle_threshold": threshold,
            "support_count": n_tier,
            "support_fraction": float(n_tier / n_total) if n_total else 0.0,
            "train_count": int(train.sum()),
            "validation_count": int(val.sum()),
            "effective_bins": effective,
            "quantile_edges": edges,
            "pooled_model_fit_ok": bool(pooled_ok),
            "pooled_median": float(np.median(g_med)) if g_med else None,
            "pooled_p95": float(np.median([e["global_model"]["p95"] for e in bins
                                           if e["global_model"]["p95"] is not None])) if g_med else None,
            "bin_median": float(np.median(b_med)) if b_med else None,
            "bin_p95": float(np.median([e["bin_model"]["p95"] for e in bins
                                        if e["bin_model"]["p95"] is not None])) if b_med else None,
            "bins_with_models": len(deltas),
            "median_delta": float(np.median(deltas)) if deltas else None,
            "p95_delta": float(np.median(d_p95)) if d_p95 else None,
            "improved_bins": int(sum(1 for d in deltas if d > 0.0)),
            "bins": bins,
        }

    tiers = {"all_cycle_valid": build_tier(base_support, "all_cycle_valid", None)}
    for t in thresholds:
        mask = base_support & (resid_flat <= t)
        tiers[f"cycle_le_{t:g}"] = build_tier(mask, f"cycle_le_{t:g}", t)

    return {
        "feature_h": H,
        "feature_w": W,
        "n_cells": n_total,
        "n_bins": n_bins,
        "cycle_thresholds": thresholds,
        "base_support_count": int(base_support.sum()),
        "base_support_fraction": float(base_support.sum() / n_total) if n_total else 0.0,
        "tiers": tiers,
        "notes": [
            "per-frame relative-depth rank bins only; no true metric near/far is asserted",
            "checkerboard even(x+y) train / odd(x+y) validate; neighbours are correlated, not independent",
            "quantile edges may use all eligible depth values including validation cells (leakage)",
            "positive delta = bin model held-out residual lower than the pooled model on the same cells",
            "this is a model-derived flow-structure diagnostic, not camera-motion ground truth",
            "no cycle threshold is selected; no accuracy, speed or stabilization claim is made",
        ],
    }


# --------------------------------------------------------------------------------------
# runner orchestration
# --------------------------------------------------------------------------------------

def _direction_audit(src_grid: np.ndarray, flow: np.ndarray, cycle_resid: np.ndarray,
                     cycle_valid: np.ndarray, n_bins: int, thresholds: Sequence[float]) -> dict:
    return depth_motion_audit(src_grid, flow, cycle_resid, cycle_valid, n_bins=n_bins,
                             cycle_thresholds=thresholds)


def summarize_pairs(pairs: Sequence[dict], cycle_thresholds: Sequence[float], n_bins: int) -> dict:
    """Aggregate per-pair/per-direction audits into a per-window summary, one block per tier.

    For each tier reports, over all pair-directions, the median support fraction, the count of usable
    (finite) pooled and bin medians, the median pooled and bin held-out median/p95 residual, the median
    delta, and how many bins improved. Nothing is collapsed into a single score and no threshold is selected.
    """
    keys = ["all_cycle_valid"] + [f"cycle_le_{float(t):g}" for t in cycle_thresholds]
    summary = {}
    for key in keys:
        entries = [d["audit"]["tiers"][key] for p in pairs for d in p["directions"]]
        n = len(entries)
        def _finite(vals):
            return [float(v) for v in vals if v is not None and np.isfinite(v)]
        pooled_med = _finite([e["pooled_median"] for e in entries])
        bin_med = _finite([e["bin_median"] for e in entries])
        pooled_p95 = _finite([e["pooled_p95"] for e in entries])
        bin_p95 = _finite([e["bin_p95"] for e in entries])
        deltas = _finite([e["median_delta"] for e in entries])
        summary[key] = {
            "pair_directions": n,
            "support_fraction_median": (float(np.median([e["support_fraction"] for e in entries]))
                                        if entries else None),
            "pooled_median_heldout_median": float(np.median(pooled_med)) if pooled_med else None,
            "bin_median_heldout_median": float(np.median(bin_med)) if bin_med else None,
            "median_delta_median": float(np.median(deltas)) if deltas else None,
            "usable_pooled": len(pooled_med),
            "usable_bin": len(bin_med),
            "median_pooled_p95": float(np.median(pooled_p95)) if pooled_p95 else None,
            "median_bin_p95": float(np.median(bin_p95)) if bin_p95 else None,
            "total_improved_bins": int(sum(e["improved_bins"] for e in entries)),
            "total_bins_with_models": int(sum(e["bins_with_models"] for e in entries)),
        }
    summary["_meta"] = {"n_bins": int(n_bins), "cycle_thresholds": [float(t) for t in cycle_thresholds]}
    return summary


def _source_block(frames_npz_sha256: str, depth_npz_sha256: str, weights_sha256: str,
                  meta: dict, model_id: str, normalization: str) -> dict:
    """Path-independent cache provenance (no absolute footage or model-cache paths)."""
    window = meta.get("window") or {}
    block = {
        "frames_npz_sha256": frames_npz_sha256,
        "depth_npz_sha256": depth_npz_sha256,
        "weights_sha256": weights_sha256,
        "frames_cache_schema": meta.get("schema"),
        "depth_model_id": model_id,
        "depth_normalization": normalization,
    }
    fps = meta.get("fps")
    if fps is not None:
        block["fps"] = float(fps)
    if window:
        block["window_s"] = [window.get("start_s"), window.get("duration_s")]
    return block


def run_audit(frames: np.ndarray, meta: dict, depth_info: dict, weights: str, out_path=None, *,
              model_id: Optional[str] = None, normalization: Optional[str] = None,
              iters: int = DEFAULT_ITERS, downscale: int = DEFAULT_DOWNSCALE,
              bins: int = DEFAULT_BINS, cycle_thresholds: Sequence[float] = DEFAULT_CYCLE_THRESHOLDS,
              flow_dtype: str = "float32", corr_dtype: str = "float32",
              lock_enabled: bool = True, lock_chunk: int = 0, lock_pause: float = 0.0,
              lock_factory=None, runner_factory=None, write: bool = True, log=print) -> dict:
    """Run the audit over adjacent sampled depth indices and (optionally) write the JSON atomically.

    ``lock_factory`` / ``runner_factory`` default to :class:`bench.heavylock.HeavyLock` and
    :class:`bench.flow_prediction_real.RaftRunner`; tests inject fakes so no MLX or weights are touched. Each
    ordered pair is audited in **both** directions; the backward direction uses ``depth_dst`` as its source
    and the backward flow/residual fields. Data are never pooled across different source windows.
    """
    iters = int(iters)
    downscale = int(downscale)
    bins = int(bins)
    if iters < 1:
        raise ValueError("--iters must be >= 1")
    if downscale < 1:
        raise ValueError("--downscale must be >= 1")
    if bins < 1:
        raise ValueError("--bins must be >= 1")
    thresholds = [float(t) for t in cycle_thresholds]
    if not thresholds:
        raise ValueError("--cycle-thresholds must list at least one value")
    for t in thresholds:
        if not np.isfinite(t) or t < 0.0:
            raise ValueError(f"--cycle-thresholds values must be finite and >= 0, got {t!r}")
    if int(lock_chunk) < 0:
        raise ValueError("--lock-chunk must be >= 0")
    if float(lock_pause) < 0.0:
        raise ValueError("--lock-pause must be >= 0")
    if not Path(weights).exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    lock_factory = lock_factory or R.HeavyLock
    runner_factory = runner_factory or R.RaftRunner

    indices = depth_info["indices"]
    depth = depth_info["depth"]
    sampled_frames = np.ascontiguousarray(frames[np.asarray(indices, np.int64)])

    result = _run(
        sampled_frames=sampled_frames, meta=meta, depth=depth, indices=indices, weights=weights,
        iters=iters, downscale=downscale, bins=bins, thresholds=thresholds, flow_dtype=flow_dtype,
        corr_dtype=corr_dtype, lock_enabled=lock_enabled, lock_chunk=int(lock_chunk),
        lock_pause=float(lock_pause), lock_factory=lock_factory, runner_factory=runner_factory, log=log,
        model_id=model_id if model_id is not None else depth_info.get("model_id"),
        normalization=normalization if normalization is not None else depth_info.get("normalization"))
    if write and out_path is not None:
        MF.write_manifest(out_path, result)
    return result


def _run(*, sampled_frames, meta, depth, indices, weights, iters, downscale, bins, thresholds,
         flow_dtype, corr_dtype, lock_enabled, lock_chunk, lock_pause, lock_factory, runner_factory,
         log, model_id=None, normalization=None) -> dict:
    """Core loop: encode frames once, then audit every adjacent pair in both directions under the lock."""
    n_pairs = int(indices.size) - 1
    lock = lock_factory(what="mlxstab depth_motion_audit", enabled=lock_enabled,
                        chunk=lock_chunk, pause_s=lock_pause)
    runner = None
    pairs = []
    prep_s = 0.0
    flow_total = 0.0
    audit_total = 0.0
    step = R.StepCounter(on_step=lock.on_step)
    t_start = time.perf_counter()
    with lock:
        runner = runner_factory(weights, downscale, flow_dtype, corr_dtype)
        t0 = time.perf_counter()
        runner.prepare(sampled_frames, on_step=step.tick)
        prep_s = time.perf_counter() - t0

        for i in range(n_pairs):
            t0 = time.perf_counter()
            fwd, bwd = runner.reference_pair(i, i + 1, iters)
            flow_s = time.perf_counter() - t0
            flow_total += flow_s

            fwd_flat = np.asarray(fwd, np.float64).reshape(runner.h * runner.w, 2)
            bwd_flat = np.asarray(bwd, np.float64).reshape(runner.h * runner.w, 2)
            cycle = R.cycle_residual_fields(fwd_flat, bwd_flat, runner.h, runner.w)

            src_grid = depth_to_feature_grid(depth[i], runner.h, runner.w, runner.pad, runner.downscale)
            dst_grid = depth_to_feature_grid(depth[i + 1], runner.h, runner.w, runner.pad,
                                             runner.downscale)

            t1 = time.perf_counter()
            directions = [
                {"direction": "forward",
                 "audit": _direction_audit(src_grid, fwd, cycle["fwd_resid"], cycle["fwd_valid"],
                                           bins, thresholds)},
                {"direction": "backward",
                 "audit": _direction_audit(dst_grid, bwd, cycle["bwd_resid"], cycle["bwd_valid"],
                                           bins, thresholds)},
            ]
            audit_total += time.perf_counter() - t1

            fps = meta.get("fps")
            if fps:
                time_s = [float(indices[i]) / float(fps), float(indices[i + 1]) / float(fps)]
            else:
                time_s = None
            fwd_valid_frac = float(np.mean(cycle["fwd_valid"])) if cycle["fwd_valid"].size else 0.0
            bwd_valid_frac = float(np.mean(cycle["bwd_valid"])) if cycle["bwd_valid"].size else 0.0
            pairs.append({
                "pair": i,
                "source_frame_indices": [int(indices[i]), int(indices[i + 1])],
                "time_s": time_s,
                "fwd_cycle_valid_frac": fwd_valid_frac,
                "bwd_cycle_valid_frac": bwd_valid_frac,
                "flow_s": flow_s,
                "directions": directions,
            })
            step.tick()

    duration_s = time.perf_counter() - t_start
    summary = summarize_pairs(pairs, thresholds, bins)
    grid = [int(runner.h), int(runner.w)] if runner is not None else None
    pad = [int(v) for v in runner.pad] if runner is not None and runner.pad is not None else None
    lock_status = lock.status()
    # HeavyLock reports its local lock-file path for interactive diagnostics. The artifact may be shared for
    # review, so keep the useful timing/state fields but never serialize that machine-specific path.
    lock_status.pop("path", None)
    return {
        "status": "success",
        "schema": SCHEMA,
        "limitations": list(LIMITATIONS),
        "pair_count": len(pairs),
        "cycle_thresholds": thresholds,
        "bins": bins,
        "depth": {
            "model_id": model_id,
            "normalization": normalization,
            "sampled_frames": int(indices.size),
        },
        "flow": {
            "model": "RAFT-things MLX",
            "downscale": downscale,
            "flow_dtype": flow_dtype,
            "corr_dtype": corr_dtype,
            "iters": iters,
            "feature_grid": grid,
            "pad": pad,
        },
        "split": {
            "scheme": "checkerboard_x_plus_y_parity",
            "train": "even",
            "validation": "odd",
        },
        "timing_s": {
            "feature_prepare": prep_s,
            "flow_pairs_total": flow_total,
            "flow_pair_mean": (flow_total / len(pairs)) if pairs else None,
            "audit_total": audit_total,
            "duration": duration_s,
            "lock": lock_status,
        },
        "pairs": pairs,
        "summary": summary,
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(prog="bench/depth_motion_audit.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames-npz", required=True, help="cached decoded-frame npz (frames/meta)")
    p.add_argument("--depth-npz", required=True, help="cached relative-depth npz (indices/depth/...)")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="local RAFT weights file")
    p.add_argument("--out", default=None, help="output JSON path (atomic write)")
    p.add_argument("--iters", type=int, default=DEFAULT_ITERS, help="cold RAFT iterations per pair")
    p.add_argument("--downscale", type=int, default=DEFAULT_DOWNSCALE, help="flow input downscale (>=1)")
    p.add_argument("--bins", type=int, default=DEFAULT_BINS, help="equal-count depth bins (>=1)")
    p.add_argument("--flow-dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--corr-dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--cycle-thresholds", default=",".join(f"{t:g}" for t in DEFAULT_CYCLE_THRESHOLDS),
                   help="comma-separated nonnegative cycle residual gates")
    p.add_argument("--lock", choices=["on", "off"], default="on", help="shared heavy lock")
    p.add_argument("--lock-chunk", type=int, default=0, help="release/re-acquire every N steps (0 = hold)")
    p.add_argument("--lock-pause", type=float, default=0.0, help="seconds to yield between lock chunks")
    p.add_argument("--dry-run", action="store_true", help="print resolved options and exit")
    return p


def parse_cycle_thresholds(text) -> list:
    """Parse ``"a,b,c"`` cycle gates: finite, >= 0, de-duplicated, caller order preserved."""
    if isinstance(text, (list, tuple)):
        vals = [float(v) for v in text]
    else:
        vals = [float(v) for v in str(text).split(",") if str(v).strip()]
    if not vals:
        raise ValueError("--cycle-thresholds must list at least one value")
    out = []
    for v in vals:
        if not np.isfinite(v) or v < 0.0:
            raise ValueError(f"--cycle-thresholds values must be finite and >= 0, got {v!r}")
        if v not in out:
            out.append(v)
    return out


def run(opts, log=print) -> dict:
    """Validate the caches, run the audit and write the atomic JSON. Returns the result."""
    thresholds = parse_cycle_thresholds(opts.cycle_thresholds)
    if opts.iters < 1:
        raise ValueError("--iters must be >= 1")
    if opts.downscale < 1:
        raise ValueError("--downscale must be >= 1")
    if opts.bins < 1:
        raise ValueError("--bins must be >= 1")
    if opts.lock_chunk < 0:
        raise ValueError("--lock-chunk must be >= 0")
    if opts.lock_pause < 0:
        raise ValueError("--lock-pause must be >= 0")
    for label, path in (("frames", opts.frames_npz), ("depth", opts.depth_npz)):
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} cache not found: {path}")
    if not Path(opts.weights).exists():
        raise FileNotFoundError(f"weights not found: {opts.weights}")

    frames, meta = load_frames_npz(opts.frames_npz)
    depth_info = load_depth_npz(opts.depth_npz)
    frames_npz_sha256 = MF.sha256_file(opts.frames_npz)
    validate_cache_correspondence(frames, meta, depth_info, frames_npz_sha256)

    result = run_audit(
        frames, meta, depth_info, opts.weights, None, model_id=depth_info["model_id"],
        normalization=depth_info["normalization"], iters=opts.iters, downscale=opts.downscale,
        bins=opts.bins, cycle_thresholds=thresholds, flow_dtype=opts.flow_dtype,
        corr_dtype=opts.corr_dtype, lock_enabled=(opts.lock == "on"), lock_chunk=opts.lock_chunk,
        lock_pause=opts.lock_pause, write=False, log=log)

    depth_npz_sha256 = MF.sha256_file(opts.depth_npz)
    weights_sha256 = MF.sha256_file(opts.weights)
    result["source"] = _source_block(frames_npz_sha256, depth_npz_sha256, weights_sha256, meta,
                                     depth_info["model_id"], depth_info["normalization"])
    result["depth"]["model_id"] = depth_info["model_id"]
    result["depth"]["normalization"] = depth_info["normalization"]
    result["flow"]["weights_sha256"] = weights_sha256
    result["updated"] = MF.now_iso()

    if opts.out:
        MF.write_manifest(opts.out, result)
    return result


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.dry_run:
        cfg = {
            "frames_npz": a.frames_npz, "depth_npz": a.depth_npz, "weights": a.weights, "out": a.out,
            "iters": a.iters, "downscale": a.downscale, "bins": a.bins, "flow_dtype": a.flow_dtype,
            "corr_dtype": a.corr_dtype, "cycle_thresholds": parse_cycle_thresholds(a.cycle_thresholds),
            "lock": a.lock, "lock_chunk": a.lock_chunk, "lock_pause": a.lock_pause,
        }
        print(json.dumps(cfg, indent=1, sort_keys=True))
        return 0
    try:
        result = run(a, log=lambda m: print(m, file=sys.stderr))
    except Exception as e:
        print(json.dumps({"status": "failure", "error": {"type": type(e).__name__, "message": str(e)}},
                         sort_keys=True))
        return 1
    print(json.dumps({"status": result["status"], "out": a.out,
                      "pair_count": result["pair_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
