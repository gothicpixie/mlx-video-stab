"""Cache/RAFT runner for :mod:`bench.depth_layer_real_audit`.

The audit core remains NumPy-only. This module validates one paired frame/depth
cache, runs bidirectional RAFT on its adjacent sampled frames, and writes one
path-free JSON report for that source window.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from . import depth_temporal_pilot as D
from . import flow_prediction_real as R
from . import manifest as MF
from .depth_layer_real_audit import audit_depth_layer_window


def run_cached_audit(frames_npz, depth_npz, weights, out_path=None, *, iters=12, downscale=1,
                     n_bins=4, smooth_sigma=1.0, flow_dtype="float32", corr_dtype="float32",
                     lock_enabled=True, lock_chunk=0, lock_pause=0.0, lock_factory=None,
                     runner_factory=None, write=True, log=print):
    """Run the matched downstream geometry audit on exactly one paired cache window."""
    if int(iters) < 1 or int(downscale) < 1 or int(n_bins) < 1:
        raise ValueError("iters, downscale and n_bins must be >= 1")
    if not np.isfinite(float(smooth_sigma)) or float(smooth_sigma) < 0:
        raise ValueError("smooth_sigma must be finite and >= 0")
    if int(lock_chunk) < 0 or float(lock_pause) < 0:
        raise ValueError("lock_chunk and lock_pause must be >= 0")
    frames_npz, depth_npz, weights = Path(frames_npz), Path(depth_npz), Path(weights)
    for label, path in (("frames", frames_npz), ("depth", depth_npz), ("weights", weights)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} input not found: {path}")
    frames, meta = D.load_frames_npz(frames_npz)
    depth_info = D.load_depth_npz(depth_npz)
    frames_sha = MF.sha256_file(frames_npz)
    D.validate_cache_correspondence(frames, meta, depth_info, frames_sha)
    frame_indices = depth_info["indices"]
    sampled_frames = np.ascontiguousarray(frames[frame_indices])
    depth = depth_info["depth"]

    lock_factory = lock_factory or R.HeavyLock
    runner_factory = runner_factory or R.RaftRunner
    lock = lock_factory(what="mlxstab depth_layer_real_audit", enabled=lock_enabled,
                        chunk=int(lock_chunk), pause_s=float(lock_pause))
    runner = None
    observations = []
    pair_meta = []
    start = time.perf_counter()
    step = R.StepCounter(on_step=lock.on_step)
    with lock:
        runner = runner_factory(str(weights), int(downscale), flow_dtype, corr_dtype)
        runner.prepare(sampled_frames, on_step=step.tick)
        for i in range(len(frame_indices) - 1):
            fwd, bwd = runner.reference_pair(i, i + 1, int(iters))
            fwd = np.asarray(fwd, np.float64)
            bwd = np.asarray(bwd, np.float64)
            cycle = R.cycle_residual_fields(
                fwd.reshape(runner.h * runner.w, 2), bwd.reshape(runner.h * runner.w, 2),
                runner.h, runner.w)
            dgrid = D.depth_to_feature_grid(depth[i], runner.h, runner.w, runner.pad, runner.downscale)
            observations.append({"depth": dgrid, "flow": fwd, "cycle_resid": cycle["fwd_resid"],
                                "cycle_valid": cycle["fwd_valid"]})
            fps = meta.get("fps")
            pair_meta.append({
                "source_frame_indices": [int(frame_indices[i]), int(frame_indices[i + 1])],
                "time_s": ([float(frame_indices[i]) / float(fps),
                            float(frame_indices[i + 1]) / float(fps)] if fps else None),
                "cycle_map_valid_fraction": float(np.mean(cycle["fwd_valid"])),
            })
            step.tick()
            if log:
                log(f"depth-layer real audit pair {i + 1}/{len(frame_indices) - 1}")

    report = audit_depth_layer_window(observations, n_bins=n_bins, smooth_sigma=smooth_sigma)
    for item, provenance in zip(report["pairs"], pair_meta):
        item.update(provenance)
    lock_status = lock.status()
    lock_status.pop("path", None)
    report.update({
        "inputs": {"frames_sha256": frames_sha, "depth_sha256": MF.sha256_file(depth_npz),
                   "weights_sha256": MF.sha256_file(weights),
                   "sampled_frame_count": int(len(frame_indices)),
                   "sampled_frame_indices": [int(v) for v in frame_indices]},
        "depth_model": {"model_id": depth_info["model_id"],
                        "normalization": depth_info["normalization"]},
        "flow": {"model": "RAFT-things MLX", "iters": int(iters), "downscale": int(downscale),
                 "flow_dtype": flow_dtype, "corr_dtype": corr_dtype,
                 "feature_grid": [int(runner.h), int(runner.w)],
                 "pad": [int(v) for v in runner.pad]},
        "timing_s": {"duration": float(time.perf_counter() - start), "lock": lock_status},
    })
    if write and out_path is not None:
        MF.write_manifest(out_path, report)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-npz", required=True, help="paired decoded-frame cache")
    parser.add_argument("--depth-npz", required=True, help="relative-depth cache derived from these frames")
    parser.add_argument("--weights", default=str(R.DEFAULT_WEIGHTS), help="local RAFT weights")
    parser.add_argument("--out", required=True, help="path for the path-free JSON report")
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--downscale", type=int, default=1)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--smooth-sigma", type=float, default=1.0, help="Gaussian sigma measured in pairs")
    parser.add_argument("--flow-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--corr-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--lock", choices=("on", "off"), default="on")
    parser.add_argument("--lock-chunk", type=int, default=0)
    parser.add_argument("--lock-pause", type=float, default=0.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run_cached_audit(args.frames_npz, args.depth_npz, args.weights, args.out,
                              iters=args.iters, downscale=args.downscale, n_bins=args.bins,
                              smooth_sigma=args.smooth_sigma, flow_dtype=args.flow_dtype,
                              corr_dtype=args.corr_dtype, lock_enabled=(args.lock == "on"),
                              lock_chunk=args.lock_chunk, lock_pause=args.lock_pause)
    print(MF.dumps_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
