"""Capture scale-4 fit-confidence signals from the source before stabilization.

This offline replay evaluates candidate diagnostics; it does not select a flow
scale or alter the stabilizer.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.flow_scale_localization import validate_matched_manifests
from bench.manifest import sha256_file
from mlxstab import media
from mlxstab.stabilize import Config, RaftFlow, analyze


def _block_summary(fits, cycle_records, fps, block_seconds=1.0):
    """Summarize existing scale-4 analyzer signals on fixed source-time blocks."""
    by_block = defaultdict(list)
    for pair, (fit, cycle) in enumerate(zip(fits, cycle_records)):
        block = int(math.floor((pair / fps) / block_seconds))
        by_block[block].append({
            "pair": pair,
            "fallback": bool(fit.fallback),
            "fallback_reason": fit.fallback_reason,
            "inlier_frac": float(fit.inlier_frac),
            "mover_frac": float(fit.mover_frac),
            "valid_samples": fit.n_valid_samples,
            "ransac_inliers": fit.n_ransac_inliers,
            **cycle,
        })
    blocks = []
    for block in sorted(by_block):
        rows = by_block[block]
        q = lambda key, p: float(np.quantile([r[key] for r in rows], p))
        blocks.append({
            "block": block,
            "start_s": float(block * block_seconds),
            "end_s": float((block + 1) * block_seconds),
            "pairs": len(rows),
            "fallback_pairs": int(sum(r["fallback"] for r in rows)),
            "fallback_reasons": dict(sorted({
                reason: sum(r["fallback"] and r["fallback_reason"] == reason for r in rows)
                for reason in {r["fallback_reason"] for r in rows
                               if r["fallback"] and r["fallback_reason"]}
            }.items())),
            "inlier_frac_median": q("inlier_frac", 0.5),
            "inlier_frac_min": q("inlier_frac", 0.0),
            "inlier_frac_p10": q("inlier_frac", 0.1),
            "mover_frac_median": q("mover_frac", 0.5),
            "valid_samples_median": q("valid_samples", 0.5),
            "fb_valid_fraction_mean": float(np.mean([r["fb_valid_fraction"] for r in rows])),
            "fb_error_over_tolerance_median": q("fb_error_over_tolerance_median", 0.5),
            "fb_error_over_tolerance_p90": q("fb_error_over_tolerance_p90", 0.9),
        })
    return blocks, by_block


def capture(manifest2, manifest4, block_seconds=1.0):
    provenance = validate_matched_manifests(manifest2, manifest4)
    m4 = json.loads(Path(manifest4).read_text())
    source = m4["source"]["path"]
    window = m4["source"]["window"]
    config = Config(**m4["config"])
    if config.flow_downscale != 4:
        raise ValueError("scale-4 manifest must describe the actual analyzer flow setting")

    info = media.probe(source)
    if m4["source"].get("sha256") != sha256_file(source):
        raise ValueError("source file no longer matches the benchmark manifest hash")
    frames, fps, (width, height) = media.decode(
        source, info, window.get("start_s"), window.get("duration_s"))
    weights = m4["weights"]["path"]
    flow = RaftFlow(weights, config.flow_iters, config.flow_downscale, config.warm_start,
                    getattr(mx, config.corr_dtype), getattr(mx, config.flow_dtype),
                    config.warm_iters or None)
    cycle_records = []

    def callback(prev_i, i, xs, ys, forward, backward, valid, fb_err, fb_tol, fb_inside):
        ratio = fb_err / np.maximum(fb_tol, np.finfo(np.float32).eps)
        cycle_records.append({
            "fb_valid_fraction": float(np.mean(valid)),
            "fb_error_over_tolerance_median": float(np.median(ratio)),
            "fb_error_over_tolerance_p90": float(np.quantile(ratio, 0.9)),
        })

    flow.reset()
    motion = analyze(frames, flow, config, width, height, fps,
                     flow_sample_callback=callback)
    if len(cycle_records) != len(motion.fits):
        raise RuntimeError("flow callback and motion fits have different pair counts")
    blocks, _ = _block_summary(motion.fits, cycle_records, fps, block_seconds)
    scale2 = json.loads(Path(manifest2).read_text())
    quality = {}
    for key, manifest in (("scale2", scale2), ("scale4", m4)):
        output = manifest["metrics"]["output"]
        quality[key] = {name: output.get(name) for name in (
            "jitter_px", "wobble_px", "fallback_frames", "inlier_frac_mean",
            "inlier_frac_min", "mover_frac_mean")}
    quality["scale4_minus_scale2"] = {
        name: quality["scale4"][name] - quality["scale2"][name]
        for name in ("jitter_px", "wobble_px")
    }
    return {
        "schema": "mlxstab.flow_scale_fit_diagnostics/1",
        "purpose": "Descriptive input-side scale-4 fit signals; not ground-truth confidence or a gate.",
        "provenance": provenance,
        "source": {"sha256": provenance["source_sha256"], "window": window,
                   "frames": motion.n_frames, "pairs": len(motion.fits),
                   "fps": float(fps), "width": width, "height": height},
        "analyzer": {"flow_downscale": 4, "flow_iters": config.flow_iters,
                     "flow_dtype": config.flow_dtype, "corr_dtype": config.corr_dtype,
                     "warm_start": config.warm_start, "model": config.model,
                     "step": config.step},
        "global": {
            "fallback_pairs": int(sum(f.fallback for f in motion.fits)),
            "inlier_frac_mean": float(np.mean([f.inlier_frac for f in motion.fits])),
            "inlier_frac_min": float(np.min([f.inlier_frac for f in motion.fits])),
            "mover_frac_mean": float(np.mean([f.mover_frac for f in motion.fits])),
            "valid_samples_mean": float(np.mean([f.n_valid_samples for f in motion.fits])),
        },
        "downstream_quality_reference": quality,
        "blocks": blocks,
        "interpretation_note": (
            "FB consistency measures cycle agreement, not correspondence truth. Fit inlier fraction is the "
            "analyzer's existing RANSAC statistic, not calibrated uncertainty. Three windows are a screening set only."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale2-manifest", required=True)
    parser.add_argument("--scale4-manifest", required=True)
    parser.add_argument("--block-seconds", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.block_seconds <= 0:
        parser.error("--block-seconds must be positive")
    result = capture(args.scale2_manifest, args.scale4_manifest, args.block_seconds)
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": args.out, "global": result["global"],
                      "blocks": len(result["blocks"])}, indent=2))


if __name__ == "__main__":
    main()
