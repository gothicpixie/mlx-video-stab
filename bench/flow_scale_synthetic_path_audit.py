"""Compare saved scale-2/scale-4 plans with a known synthetic camera path.

The supported scene is the far-texture-only ``synth.render`` control. The
``rho`` parameter enables or disables rolling shutter, and every sampled
pixel has one exact camera map in either case.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.flow_scale_localization import validate_matched_manifests
from bench.manifest import dumps_json
from bench.plan_shape_audit import load_plan_npz
from mlxstab import synth
from mlxstab.stabilize import source_coords


def _scene_map(points, poses, width, height):
    """Map camera pixels into the synthetic far layer's world coordinates."""
    points = np.asarray(points, np.float64)
    poses = np.asarray(poses, np.float64)
    x = points[:, 0] - (width - 1) / 2
    y = points[:, 1] - (height - 1) / 2
    theta = np.deg2rad(poses[:, 2])
    c, s = np.cos(theta), np.sin(theta)
    return np.stack((c * x - s * y + (width - 1) / 2 + poses[:, 0],
                     s * x + c * y + (height - 1) / 2 + poses[:, 1]), axis=1)


def error_series(plan, width, height, frames, fps, intended, actual, rho, grid_step):
    xs = np.arange(0, width, grid_step, dtype=np.float32)
    ys = np.arange(0, height, grid_step, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    output_points = np.stack((xx.ravel(), yy.ravel()), axis=1).astype(np.float64)
    errors = np.empty((frames, len(output_points), 2), dtype=np.float64)
    valid = np.empty((frames, len(output_points)), dtype=bool)
    for t in range(frames):
        if plan is None:
            source_points = output_points
        else:
            src = source_coords(plan, t, width, height,
                                grid=(mx.array(xx), mx.array(yy)))
            mx.eval(src)
            source_points = np.asarray(src[0]).reshape(-1, 2).astype(np.float64)
        valid[t] = ((source_points[:, 0] >= 0) & (source_points[:, 0] <= width - 1) &
                    (source_points[:, 1] >= 0) & (source_points[:, 1] <= height - 1))
        input_tau = t + rho * source_points[:, 1] / height
        target_tau = t + rho * output_points[:, 1] / height
        world_input = _scene_map(source_points, actual(input_tau), width, height)
        world_target = _scene_map(output_points, intended(target_tau), width, height)
        errors[t] = world_input - world_target
    return errors, valid


def summarize_errors(errors, valid, common_points, fps, sigma=15.0):
    e = errors[:, common_points, :]
    bias = e.mean(axis=(0, 1))
    frame_mean = e.mean(axis=1)
    frame_idx = np.arange(len(frame_mean), dtype=np.float64)
    linear = np.column_stack((frame_idx, np.ones_like(frame_idx)))
    drift = np.linalg.lstsq(linear, frame_mean, rcond=None)[0][0]
    lowpass_position = gaussian_filter1d(e, sigma, axis=0, mode="nearest")
    highpass_position = e - lowpass_position
    velocity_error = np.diff(e, axis=0)
    lowpass_velocity = gaussian_filter1d(velocity_error, sigma, axis=0, mode="nearest")
    highpass_velocity_error = velocity_error - gaussian_filter1d(
        velocity_error, sigma, axis=0, mode="nearest")
    return {
        "common_points": int(e.shape[1]),
        "coverage_fraction_mean": float(valid.mean()),
        "global_bias_xy_px": bias.tolist(),
        "total_path_rmse_px": float(np.sqrt(np.mean(np.sum(e ** 2, axis=2)))),
        "residual_path_rmse_after_global_bias_px": float(
            np.sqrt(np.mean(np.sum((e - bias) ** 2, axis=2)))),
        "linear_drift_xy_px_per_frame": drift.tolist(),
        "linear_drift_xy_px_over_clip": (drift * max(len(frame_mean) - 1, 0)).tolist(),
        "lowpass_position_error_rms_px": float(np.sqrt(np.mean(lowpass_position ** 2))),
        "highpass_position_error_rms_px": float(np.sqrt(np.mean(highpass_position ** 2))),
        "path_velocity_error_mean_abs_px": float(np.abs(velocity_error).mean()),
        "lowpass_path_velocity_error_rms_px": float(np.sqrt(np.mean(lowpass_velocity ** 2))),
        "highpass_path_velocity_error_mean_abs_px": float(np.abs(highpass_velocity_error).mean()),
        "highpass_path_velocity_error_rms_px": float(np.sqrt(np.mean(highpass_velocity_error ** 2))),
        "sigma_frames": sigma,
        "sigma_seconds": float(sigma / fps),
    }


def audit(manifest2, manifest4, plan2_path, plan4_path, motion2_path, motion4_path,
          seed=13, rho=0.8, jitter=6.0, rot_jit=0.6, grid_step=32):
    provenance = validate_matched_manifests(manifest2, manifest4)
    m2, m4 = json.loads(Path(manifest2).read_text()), json.loads(Path(manifest4).read_text())
    if m2["source"]["path"] != m4["source"]["path"]:
        raise ValueError("plans must use the same encoded synthetic source")
    p2, meta2 = load_plan_npz(plan2_path)
    p4, meta4 = load_plan_npz(plan4_path)
    dims = (meta2["width"], meta2["height"], meta2["frames"], meta2["fps"])
    if dims != (meta4["width"], meta4["height"], meta4["frames"], meta4["fps"]):
        raise ValueError("saved plans differ in dimensions, frames, or cadence")
    width, height, frames, fps = dims
    if m2["source"].get("sha256") != provenance["source_sha256"]:
        raise ValueError("source hashes do not match their manifests")
    intended, actual = synth.camera(frames, seed + 1, jitter=jitter, rot_jit=rot_jit)
    e2, v2 = error_series(p2, width, height, frames, fps, intended, actual, rho, grid_step)
    e4, v4 = error_series(p4, width, height, frames, fps, intended, actual, rho, grid_step)
    eraw, vraw = error_series(None, width, height, frames, fps, intended, actual, rho, grid_step)
    common = v2.all(axis=0) & v4.all(axis=0)
    if not common.any():
        raise ValueError("no output grid points remain valid in both plans for every frame")

    def add_run_metrics(summary, manifest, motion_path):
        output = manifest["metrics"]["output"]
        z = np.load(motion_path, allow_pickle=False)
        summary.update({
            "output_evaluator_jitter_px": output.get("jitter_px"),
            "output_evaluator_wobble_px": output.get("wobble_px"),
            "fit_inlier_frac_p10": float(np.quantile(z["inlier_frac"], 0.1)),
            "fit_inlier_frac_min": float(np.min(z["inlier_frac"])),
            "fit_fallback_pairs": int(np.asarray(z["fallback"]).sum()),
        })
        return summary

    return {
        "schema": "mlxstab.flow_scale_synthetic_path_audit/1",
        "scope": "Far-layer-only synthetic path audit with configurable rolling shutter; full source_coords map; not parallax or moving-subject quality.",
        "provenance": provenance,
        "scene": {"kind": "synth.render depths=(1.0,), subject=False",
                  "seed": seed, "rho": rho, "jitter": jitter, "rot_jit": rot_jit,
                  "frames": frames, "fps": fps, "width": width, "height": height,
                  "grid_step_px": grid_step, "compared_points": int(common.sum()),
                  "common_support": "same output-grid points valid in both scales for every frame"},
        "raw_input": summarize_errors(eraw, vraw, np.ones(vraw.shape[1], dtype=bool), fps),
        "scale2": add_run_metrics(summarize_errors(e2, v2, common, fps), m2, motion2_path),
        "scale4": add_run_metrics(summarize_errors(e4, v4, common, fps), m4, motion4_path),
        "metric_note": (
            "Each sampled stabilized output point is mapped through the full Plan source_coords field, then through "
            "the exact row-time camera transform to the far layer. The target uses the intended row-time camera. "
            "Global bias and high-pass path residual are reported separately; neither is a perceptual score."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale2-manifest", required=True)
    parser.add_argument("--scale4-manifest", required=True)
    parser.add_argument("--scale2-plan", required=True)
    parser.add_argument("--scale4-plan", required=True)
    parser.add_argument("--scale2-motion", required=True)
    parser.add_argument("--scale4-motion", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--rho", type=float, default=0.8)
    parser.add_argument("--jitter", type=float, default=6.0)
    parser.add_argument("--rot-jit", type=float, default=0.6)
    parser.add_argument("--grid-step", type=int, default=32)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.grid_step < 1:
        parser.error("--grid-step must be positive")
    payload = audit(args.scale2_manifest, args.scale4_manifest, args.scale2_plan, args.scale4_plan,
                    args.scale2_motion, args.scale4_motion, args.seed, args.rho,
                    args.jitter, args.rot_jit, args.grid_step)
    Path(args.out).write_text(dumps_json(payload) + "\n", encoding="utf-8")
    print(json.dumps({"out": args.out, "raw_input": payload["raw_input"],
                      "scale2": payload["scale2"], "scale4": payload["scale4"]}, indent=2))


if __name__ == "__main__":
    main()
