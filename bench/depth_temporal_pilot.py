"""Cache-driven relative-depth temporal-consistency pilot (reproducible, offline from cached inputs).

This is the reproducible driver behind the one-off inline pilot that produced
``out/bench/flow_prediction_real/stream00000_t25_depth_temporal_audit.json``. It rebuilds the *same*
per-adjacent-sampled-pair audit from three cached inputs only -- a decoded-frame ``.npz``, a cached
per-frame monocular *relative* depth ``.npz`` and a local RAFT weights file -- with **no media decode, no
depth inference and no production-code change**. It runs RAFT only to obtain the flow that transports one
sampled depth map onto its neighbour, then hands the paired feature-grid depth maps to the pure-NumPy
:func:`bench.depth_temporal_audit.relative_depth_pair_audit`.

Scope and meaning
-----------------
Monocular relative depth is defined only up to a per-frame scale/shift, so each pair is compared with an
independent robust affine fit that **discards metric depth**: the reported residuals are *structure
self-consistency* under RAFT transport, not metric-depth accuracy, not a depth benchmark and not an
occlusion detector. Every support filter (finite/bounds/cycle) is diagnostic; no threshold is selected or
recommended. Nothing here is wired into a pipeline and no default is changed.

Cache provenance and privacy
----------------------------
The output records only **path-independent** provenance: content hashes of the two caches and of the
weights file, the depth model id/normalization, the sampled frame indices and the window metadata that the
frame cache itself carries (fps/window duration, if present). Absolute local footage paths, model-cache
paths and the raw source window text are deliberately **not** copied into the JSON; the frame cache's
``meta`` string is parsed only for non-identifying scalars and is never embedded verbatim.

Layering with the shared heavy lock
-----------------------------------
All MLX work runs under the project :class:`bench.heavylock.HeavyLock`. The sampled frames are encoded once
under the lock, then each adjacent pair is refined under the lock, with an ``on_step`` gate fired at every
completed feature (frame) and every completed pair boundary so ``--lock-chunk`` can release/re-acquire as
in :mod:`bench.flow_prediction_real`. The lock never covers pure-NumPy audit work, which is cheap and
device-independent.
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
from .depth_temporal_audit import depth_to_feature_grid, relative_depth_pair_audit

SCHEMA = "mlxstab.depth_temporal_pilot/1"
DEFAULT_ITERS = 24
DEFAULT_DOWNSCALE = 2
DEFAULT_CYCLE_THRESHOLDS = (0.25, 0.5, 1.0, 2.0)
DEFAULT_WEIGHTS = R.DEFAULT_WEIGHTS

LIMITATIONS = (
    "one sampled source window and a fixed set of sampled frames; not clip-wide, held-out or multi-scene",
    "per-frame monocular depth is percentile-normalized independently and affine scale/offset are fitted for "
    "each method/pair, so this is structure consistency only, not metric depth accuracy",
    "RAFT flow is model-derived and cycle filtering may select easy regions; no threshold selected",
    "no stabilization output is compared; this is an observation-quality pilot only",
)


# --------------------------------------------------------------------------------------
# cached-input loading and validation (no inference, no decode)
# --------------------------------------------------------------------------------------

def load_frames_npz(path) -> tuple:
    """Load a decoded-frame cache: returns ``(frames_uint8_NHW3, meta_dict)``.

    The frame cache is written by :func:`bench.flow_prediction_real.load_or_decode`. ``meta`` is a JSON
    scalar string; only non-identifying scalars are later surfaced. Raises ``ValueError`` on a missing array,
    a wrong dtype/shape or unparseable metadata.
    """
    with np.load(str(path)) as z:
        if "frames" not in z or "meta" not in z:
            raise ValueError("frames npz must contain 'frames' and 'meta'")
        frames = np.asarray(z["frames"])
        meta_raw = z["meta"]
    frames = np.ascontiguousarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must be [N,H,W,3]")
    if frames.dtype != np.uint8:
        raise ValueError("frames must be uint8")
    if frames.shape[0] < 2:
        raise ValueError("frames must contain at least 2 frames")
    meta = json.loads(str(meta_raw.item() if getattr(meta_raw, "ndim", 1) == 0 else meta_raw))
    if not isinstance(meta, dict):
        raise ValueError("frames meta must be a JSON object")
    return frames, meta


def load_depth_npz(path) -> dict:
    """Load a cached relative-depth ``.npz`` and validate its arrays and metadata.

    Requires ``indices`` (int, strictly increasing), ``depth`` (float ``[K,H,W]`` matching the frame cache
    original resolution), ``model_id``, ``normalization`` and ``frames_npz_sha256``. Returns a dict of the
    decoded arrays/scalars. Raises ``ValueError`` on any structural violation.
    """
    with np.load(str(path)) as z:
        missing = [k for k in ("indices", "depth", "model_id", "normalization", "frames_npz_sha256")
                   if k not in z]
        if missing:
            raise ValueError(f"depth npz missing fields: {','.join(missing)}")
        indices = np.asarray(z["indices"])
        depth = np.asarray(z["depth"])
        model_id = str(z["model_id"])
        normalization = str(z["normalization"])
        frames_npz_sha256 = str(z["frames_npz_sha256"])
    if indices.ndim != 1 or indices.size < 2:
        raise ValueError("depth indices must be a 1-D array with at least 2 entries")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("depth indices must be integers")
    if depth.ndim != 3:
        raise ValueError("depth must be [K,H,W]")
    if depth.shape[0] != indices.size:
        raise ValueError("depth count must match indices count")
    if not np.issubdtype(depth.dtype, np.floating):
        raise ValueError("depth must be floating point")
    if np.any(np.diff(indices.astype(np.int64)) <= 0):
        raise ValueError("depth indices must be strictly increasing")
    return {"indices": indices.astype(np.int64), "depth": depth, "model_id": model_id,
            "normalization": normalization, "frames_npz_sha256": frames_npz_sha256}


def validate_cache_correspondence(frames: np.ndarray, meta: dict, depth_info: dict,
                                  frames_npz_sha256: str) -> None:
    """Cross-check the frame cache, the depth cache and the recorded hashes.

    Ensures the depth cache was derived from exactly this frames file (``frames_npz_sha256``), that sampled
    indices stay inside the frame count, and that the depth maps are at the frame cache's original
    resolution. Raises ``ValueError`` on any mismatch.
    """
    if depth_info["frames_npz_sha256"] != frames_npz_sha256:
        raise ValueError(
            "depth cache frames_npz_sha256 does not match the frames npz content hash "
            "(depth was inferred from a different frame cache)")
    n = frames.shape[0]
    idx = depth_info["indices"]
    if int(idx[0]) < 0 or int(idx[-1]) >= n:
        raise ValueError(f"depth indices must be within [0, {n - 1}] for {n} frames")
    dH, dW = depth_info["depth"].shape[1], depth_info["depth"].shape[2]
    fH, fW = frames.shape[1], frames.shape[2]
    if (dH, dW) != (fH, fW):
        raise ValueError(
            f"depth original resolution {(dH, dW)} must match frames resolution {(fH, fW)}")
    # A frame cache carrying its own resolution should agree with the arrays.
    for key, got in (("height", fH), ("width", fW)):
        if meta.get(key) is not None and int(meta[key]) != int(got):
            raise ValueError(f"frames meta {key}={meta[key]} disagrees with array shape {got}")


# --------------------------------------------------------------------------------------
# json-safe helpers (reuse where available)
# --------------------------------------------------------------------------------------

def summarize_pairs(pairs: Sequence[dict], cycle_thresholds: Sequence[float]) -> dict:
    """Aggregate the per-pair audits into the pilot summary, one block per tier.

    For each tier (``all_cycle_valid`` first, then ``cycle_le_<t>``) reports the median support fraction,
    the pair count, and per-method ``mean/median`` of the pair median normalized residual plus the median
    pair p95 normalized residual over pairs where both are finite. This mirrors the inline pilot so the
    numbers stay comparable; nothing is collapsed into a single score and no threshold is selected.
    """
    n = len(pairs)
    keys = ["all_cycle_valid"] + [f"cycle_le_{float(t):g}" for t in cycle_thresholds]
    summary = {}
    for key in keys:
        tiers = [p["audit"]["tiers"][key] for p in pairs]
        block = {
            "pairs": n,
            "support_fraction_median": float(np.median([t["support_fraction"] for t in tiers])) if n else None,
        }
        for method in ("same_index", "flow_warp"):
            med = [t[method]["median_normalized_residual"] for t in tiers]
            p95 = [t[method]["p95_normalized_residual"] for t in tiers]
            ok = [(a, b) for a, b in zip(med, p95) if a is not None and b is not None
                  and np.isfinite(a) and np.isfinite(b)]
            block[method] = {
                "usable_pairs": len(ok),
                "mean_pair_median_norm_resid": float(np.mean([a for a, _ in ok])) if ok else None,
                "median_pair_median_norm_resid": float(np.median([a for a, _ in ok])) if ok else None,
                "median_pair_p95_norm_resid": float(np.median([b for _, b in ok])) if ok else None,
            }
        summary[key] = block
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


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------

def run_pilot(frames: np.ndarray, meta: dict, depth_info: dict, weights: str, out_path, *,
              model_id: Optional[str] = None, normalization: Optional[str] = None,
              iters: int = DEFAULT_ITERS, downscale: int = DEFAULT_DOWNSCALE,
              flow_dtype: str = "float32", corr_dtype: str = "float32",
              cycle_thresholds: Sequence[float] = DEFAULT_CYCLE_THRESHOLDS,
              lock_enabled: bool = True, lock_chunk: int = 0, lock_pause: float = 0.0,
              lock_factory=None, runner_factory=None, write: bool = True, log=print) -> dict:
    """Run the pilot over adjacent sampled depth indices and (optionally) write the JSON atomically.

    ``lock_factory`` / ``runner_factory`` default to :class:`bench.heavylock.HeavyLock` and
    :class:`bench.flow_prediction_real.RaftRunner`; tests inject fakes so no MLX or weights are touched.
    Returns the JSON-safe result dict. The depth maps are the **original-resolution** maps from the cache;
    they are sampled onto the feature grid with the runner's geometry (``h``/``w``/``pad``/``downscale``)
    exactly as the inline pilot did.
    """
    iters = int(iters)
    downscale = int(downscale)
    if iters < 1:
        raise ValueError("--iters must be >= 1")
    if downscale < 1:
        raise ValueError("--downscale must be >= 1")
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
        sampled_frames=sampled_frames, meta=meta, depth=depth, indices=indices,
        weights=weights, iters=iters, downscale=downscale, flow_dtype=flow_dtype,
        corr_dtype=corr_dtype, thresholds=thresholds, lock_enabled=lock_enabled,
        lock_chunk=int(lock_chunk), lock_pause=float(lock_pause), lock_factory=lock_factory,
        runner_factory=runner_factory, log=log,
        model_id=model_id if model_id is not None else depth_info.get("model_id"),
        normalization=normalization if normalization is not None else depth_info.get("normalization"))
    if write and out_path is not None:
        MF.write_manifest(out_path, result)
    return result


def _run(*, sampled_frames, meta, depth, indices, weights, iters, downscale, flow_dtype,
         corr_dtype, thresholds, lock_enabled, lock_chunk, lock_pause, lock_factory,
         runner_factory, log, model_id=None, normalization=None) -> dict:
    """Core loop: encode frames once, then audit every adjacent sampled pair under the lock."""
    n_pairs = int(indices.size) - 1
    lock = lock_factory(what=f"mlxstab depth_temporal_pilot", enabled=lock_enabled,
                        chunk=lock_chunk, pause_s=lock_pause)
    runner = None
    pairs = []
    prep_s = 0.0
    flow_total = 0.0
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
            dst_grid = depth_to_feature_grid(depth[i + 1], runner.h, runner.w, runner.pad, runner.downscale)

            audit = relative_depth_pair_audit(
                src_grid, dst_grid, fwd, cycle["fwd_resid"], cycle["fwd_valid"],
                cycle_thresholds=thresholds)

            valid_frac = float(np.mean(cycle["fwd_valid"])) if cycle["fwd_valid"].size else 0.0
            fps = meta.get("fps")
            if fps:
                time_s = [float(indices[i]) / float(fps), float(indices[i + 1]) / float(fps)]
            else:
                time_s = None
            pairs.append({
                "pair": i,
                "source_frame_indices": [int(indices[i]), int(indices[i + 1])],
                "time_s": time_s,
                "cycle_valid_frac": valid_frac,
                "flow_s": flow_s,
                "audit": audit,
            })
            step.tick()

    duration_s = time.perf_counter() - t_start
    summary = summarize_pairs(pairs, thresholds)
    grid = [int(runner.h), int(runner.w)] if runner is not None else None
    pad = [int(v) for v in runner.pad] if runner is not None and runner.pad is not None else None
    lock_status = lock.status()
    # HeavyLock reports its local lock-file path for interactive diagnostics. The pilot artifact may be
    # shared for review, so keep the useful timing/state fields but never serialize that machine-specific path.
    lock_status.pop("path", None)
    return {
        "status": "success",
        "schema": SCHEMA,
        "limitations": list(LIMITATIONS),
        "pair_count": len(pairs),
        "cycle_thresholds": thresholds,
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
        "timing_s": {
            "feature_prepare": prep_s,
            "flow_pairs_total": flow_total,
            "flow_pair_mean": (flow_total / len(pairs)) if pairs else None,
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
    p = argparse.ArgumentParser(prog="bench/depth_temporal_pilot.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames-npz", required=True, help="cached decoded-frame npz (frames/meta)")
    p.add_argument("--depth-npz", required=True, help="cached relative-depth npz (indices/depth/...)")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="local RAFT weights file")
    p.add_argument("--out", default=None, help="output JSON path (atomic write)")
    p.add_argument("--iters", type=int, default=DEFAULT_ITERS, help="cold RAFT iterations per pair")
    p.add_argument("--downscale", type=int, default=DEFAULT_DOWNSCALE, help="flow input downscale (>=1)")
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
    """Validate the caches, run the pilot and write the atomic JSON. Returns the result."""
    thresholds = parse_cycle_thresholds(opts.cycle_thresholds)
    if opts.iters < 1:
        raise ValueError("--iters must be >= 1")
    if opts.downscale < 1:
        raise ValueError("--downscale must be >= 1")
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

    result = run_pilot(
        frames, meta, depth_info, opts.weights, None, model_id=depth_info["model_id"],
        normalization=depth_info["normalization"], iters=opts.iters, downscale=opts.downscale,
        flow_dtype=opts.flow_dtype, corr_dtype=opts.corr_dtype, cycle_thresholds=thresholds,
        lock_enabled=(opts.lock == "on"), lock_chunk=opts.lock_chunk, lock_pause=opts.lock_pause,
        write=False, log=log)

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
            "iters": a.iters, "downscale": a.downscale, "flow_dtype": a.flow_dtype,
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
