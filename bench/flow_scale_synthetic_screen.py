"""Controlled known-translation screen for scale-4 fit-confidence diagnostics.

This checks flow-estimation error on synthetic texture pairs. It is not a
full stabilization or real-footage quality benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

from mlxstab.stabilize import Config, RaftFlow, analyze
from bench.manifest import sha256_file


CONDITIONS = (
    {"name": "sharp", "blur_sigma": 0.0, "contrast": 1.0},
    {"name": "blur_1p5", "blur_sigma": 1.5, "contrast": 1.0},
    {"name": "blur_4", "blur_sigma": 4.0, "contrast": 1.0},
    {"name": "low_contrast", "blur_sigma": 0.0, "contrast": 0.25},
)


def make_texture(width, height, margin, seed, blur_sigma=0.0, contrast=1.0):
    """Create deterministic multiscale luminance texture in a wider source canvas."""
    rng = np.random.default_rng(seed)
    shape = (height, width + 2 * margin)

    def norm_field(sigma):
        field = gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma=sigma)
        return (field - field.mean()) / (field.std() + 1e-6)

    texture = 128.0 + 26.0 * norm_field(0.6) + 20.0 * norm_field(2.5) + 12.0 * norm_field(8.0)
    if blur_sigma:
        texture = gaussian_filter(texture, sigma=blur_sigma)
    if contrast < 1.0:
        texture = 128.0 + (texture - 128.0) * contrast + rng.normal(0, 2.0, shape)
    texture = np.clip(texture, 0, 255).astype(np.uint8)
    return np.repeat(texture[..., None], 3, axis=2)


def translated_frames(canvas, width, height, frame_count, margin, shift_x):
    """Yield crops whose static scene translates by -shift_x pixels per frame."""
    for t in range(frame_count):
        x0 = margin + t * shift_x
        yield canvas[:, x0:x0 + width]


def rank_correlation(x, y):
    """Return Spearman rho, or None when the sample has no rank variation."""
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else None


def run_sequence(flow, canvas, width, height, frames, margin, shift_x, seed, iters, downscale):
    cfg = Config(flow_iters=iters, flow_downscale=downscale, warm_start=True,
                 mesh=(16, 9), rs_rows=18, model="similarity", seed=seed)
    pair_epe = []
    pair_valid = []

    def sample_callback(prev_i, i, xs, ys, forward, backward, valid, fb_err, fb_tol, fb_inside):
        xx, yy = np.meshgrid(xs, ys)
        inside = ((xx >= 64) & (xx < width - 64) &
                  (yy >= 64) & (yy < height - 64))
        gt = np.array([-shift_x, 0.0], dtype=np.float32)
        epe = np.linalg.norm(forward[inside] - gt, axis=-1)
        pair_epe.append({
            "pair": int(i - 1),
            "epe_mean_px": float(np.mean(epe)),
            "epe_median_px": float(np.median(epe)),
            "epe_p90_px": float(np.quantile(epe, 0.9)),
            "fb_valid_fraction": float(np.mean(valid)),
        })
        pair_valid.append(float(np.mean(valid)))

    flow.reset()
    motion = analyze(translated_frames(canvas, width, height, frames, margin, shift_x),
                     flow, cfg, width, height, fps=30.0,
                     flow_sample_callback=sample_callback)
    if len(motion.fits) != len(pair_epe):
        raise RuntimeError("flow sample callback and fit counts differ")
    inliers = np.array([fit.inlier_frac for fit in motion.fits], dtype=np.float64)
    fallbacks = int(sum(fit.fallback for fit in motion.fits))
    return {
        "flow_downscale": downscale,
        "pairs": len(motion.fits),
        "fit_fallback_pairs": fallbacks,
        "fit_inlier_frac_p10": float(np.quantile(inliers, 0.1)),
        "fit_inlier_frac_median": float(np.median(inliers)),
        "fit_inlier_frac_min": float(np.min(inliers)),
        "fb_valid_fraction_mean": float(np.mean(pair_valid)),
        "epe_mean_px": float(np.mean([x["epe_mean_px"] for x in pair_epe])),
        "epe_median_px": float(np.median([x["epe_median_px"] for x in pair_epe])),
        "epe_p90_px": float(np.mean([x["epe_p90_px"] for x in pair_epe])),
        "per_pair": pair_epe,
    }


def _fixed_effect_slope(rows, resamples=1000, seed=0):
    """Slope of EPE delta vs displacement with one intercept per texture-seed cell."""
    groups = sorted({(r["seed"], r["name"]) for r in rows})

    def slope(sampled_groups):
        selected = [r for group in sampled_groups for r in rows
                    if (r["seed"], r["name"]) == group]
        keys = [(r["seed"], r["name"]) for r in selected]
        unique = sorted(set(keys))
        X = np.zeros((len(selected), len(unique) + 1), dtype=np.float64)
        for i, (row, key) in enumerate(zip(selected, keys)):
            X[i, unique.index(key)] = 1.0
            X[i, -1] = row["shift_x"]
        y = np.array([r["delta_scale4_minus_scale2_epe_px"] for r in selected])
        return float(np.linalg.lstsq(X, y, rcond=None)[0][-1])

    rng = np.random.default_rng(seed)
    observed = slope(groups)
    boot = [slope([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            for _ in range(resamples)]
    return {"px_epe_per_px_displacement": observed,
            "cluster_bootstrap_ci95": [float(np.quantile(boot, 0.025)),
                                        float(np.quantile(boot, 0.975))],
            "bootstrap_resamples": resamples,
            "bootstrap_cluster": "texture-seed cell; all displacement levels kept together"}


def run_screen(weights, width=1920, height=1080, frames=30, shifts=(4, 8, 16, 32),
               seeds=(0, 1, 2), iters=20):
    shifts = tuple(int(s) for s in shifts)
    margin = frames * max(shifts) + 64
    flows = {
        scale: RaftFlow(weights, iters, scale, True, mx.float32, mx.float32, None)
        for scale in (2, 4)
    }
    cells = []
    for seed in seeds:
        for condition in CONDITIONS:
            canvas = make_texture(width, height, margin, seed,
                                  condition["blur_sigma"], condition["contrast"])
            for shift_x in shifts:
                row = {"seed": int(seed), "shift_x": shift_x, **condition}
                for scale in (2, 4):
                    row[f"scale{scale}"] = run_sequence(
                        flows[scale], canvas, width, height, frames, margin, shift_x,
                        seed, iters, scale)
                row["delta_scale4_minus_scale2_epe_px"] = (
                    row["scale4"]["epe_mean_px"] - row["scale2"]["epe_mean_px"])
                cells.append(row)

    fit_p10 = [r["scale4"]["fit_inlier_frac_p10"] for r in cells]
    epe_delta = [r["delta_scale4_minus_scale2_epe_px"] for r in cells]
    epe4 = [r["scale4"]["epe_mean_px"] for r in cells]
    rho_delta = rank_correlation(fit_p10, epe_delta)
    rho_absolute = rank_correlation(fit_p10, epe4)
    shift_summaries = []
    for shift in shifts:
        selected = [row for row in cells if row["shift_x"] == shift]
        shift_summaries.append({
            "shift_x_px_per_pair": shift,
            "cells": len(selected),
            "median_delta_epe_px": float(np.median([
                r["delta_scale4_minus_scale2_epe_px"] for r in selected])),
            "delta_epe_min_px": float(min(r["delta_scale4_minus_scale2_epe_px"] for r in selected)),
            "delta_epe_max_px": float(max(r["delta_scale4_minus_scale2_epe_px"] for r in selected)),
            "fit_inlier_p10_min": float(min(r["scale4"]["fit_inlier_frac_p10"] for r in selected)),
            "fit_inlier_p10_max": float(max(r["scale4"]["fit_inlier_frac_p10"] for r in selected)),
        })
    return {
        "schema": "mlxstab.flow_scale_synthetic_screen/1",
        "purpose": "Controlled dense-flow accuracy screen; not a full stabilization benchmark or calibrated confidence test.",
        "known_motion": {"kind": "constant_translation",
                         "apparent_dx_px_per_pair": [-s for s in shifts],
                         "apparent_dy_px_per_pair": 0.0, "fps": 30.0},
        "image": {"width": width, "height": height, "frames_per_sequence": frames,
                  "texture_conditions": list(CONDITIONS), "seeds": list(seeds)},
        "raft": {"weights_sha256": sha256_file(weights), "iters": iters,
                 "dtype": "float32", "corr_dtype": "float32", "warm_start": True},
        "summary": {
            "condition_seed_displacement_cells": len(cells),
            "condition_seed_clusters": len({(r["seed"], r["name"]) for r in cells}),
            "displacement_levels": shift_summaries,
            "spearman_fit_inlier_p10_vs_scale4_minus_scale2_epe": rho_delta,
            "spearman_fit_inlier_p10_vs_scale4_absolute_epe": rho_absolute,
            "fixed_effect_displacement_slope": _fixed_effect_slope(cells),
            "interpretation": "Descriptive association over synthetic condition-seed cells; no p-value, threshold, or real-video generalization claim.",
        },
        "cells": cells,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--shifts", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if min(args.width, args.height, args.frames, args.iters, *args.shifts) <= 0:
        parser.error("dimensions, frames, shift and iterations must be positive")
    result = run_screen(args.weights, args.width, args.height, args.frames,
                        tuple(args.shifts), tuple(args.seeds), args.iters)
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": args.out, "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
