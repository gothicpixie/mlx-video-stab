"""Exact-motion audit of intended camera-path retention in the production planner.

This bypasses optical flow: adjacent camera poses are encoded as exact translation Fits, then passed through the
real ``plan`` implementation. It is intended to isolate path smoothing and planner conventions, not to score
rendered-video quality.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.manifest import dumps_json
from mlxstab.motion import Fit
from mlxstab.stabilize import Config, Motion, plan


def exact_motion(width, height, fps, intended_x, actual_x, config):
    """Build a real Plan from exact frame-to-frame camera translations."""
    intended_x = np.asarray(intended_x, np.float64)
    actual_x = np.asarray(actual_x, np.float64)
    if intended_x.ndim != 1 or actual_x.shape != intended_x.shape or len(actual_x) < 1:
        raise ValueError("intended_x and actual_x must be same-length 1-D paths")
    if not np.all(np.isfinite(intended_x)) or not np.all(np.isfinite(actual_x)):
        raise ValueError("camera paths must be finite")

    # In synth's image convention, a camera moving +x makes static scene texture move -x.
    mesh = np.zeros((config.mesh[1] + 1, config.mesh[0] + 1, 2), np.float64)
    rows = np.zeros((config.rs_rows, 2), np.float64)
    fits = []
    for camera_dx in np.diff(actual_x):
        H = np.eye(3, dtype=np.float64)
        H[0, 2] = -camera_dx
        fits.append(Fit(H=H, mesh=mesh, rows=rows, inlier_frac=1.0, mover_frac=0.0, fallback=False))
    motion = Motion(width, height, fps, fits=fits, n_frames=len(actual_x))
    return plan(motion, config, n_frames=len(actual_x))


def _path_metrics(plan_obj, intended_x, actual_x, fps, highpass_sigma=3.0):
    # path_quads are the planner's intended camera path in corner coordinates.
    base = np.asarray(plan_obj.path_quads[0, 0], np.float64)
    output_x = np.asarray(plan_obj.path_quads[:, 0, 0], np.float64) - base[0]
    intended_x = np.asarray(intended_x, np.float64)
    actual_x = np.asarray(actual_x, np.float64)
    idx = np.arange(len(output_x), dtype=np.float64)
    if np.ptp(intended_x) > 1e-9:
        path_gain = float(np.polyfit(intended_x, output_x, 1)[0])
    else:
        path_gain = None
    error = output_x - intended_x
    velocity_error = np.diff(error)
    hp_velocity_error = velocity_error - gaussian_filter1d(velocity_error, highpass_sigma, mode="nearest")
    raw_error = actual_x - intended_x
    raw_velocity_error = np.diff(raw_error)
    raw_hp_velocity_error = raw_velocity_error - gaussian_filter1d(
        raw_velocity_error, highpass_sigma, mode="nearest")
    slope = np.polyfit(idx, error, 1)[0] if len(idx) > 1 else 0.0
    return {
        "path_gain": path_gain,
        "output_endpoint_x_px": float(output_x[-1]),
        "intended_endpoint_x_px": float(intended_x[-1]),
        "endpoint_error_x_px": float(error[-1]),
        "output_path_rmse_px": float(np.sqrt(np.mean(error ** 2))),
        "output_path_drift_error_px_over_clip": float(slope * max(len(idx) - 1, 0)),
        "highpass_velocity_error_rms_px_per_frame": float(np.sqrt(np.mean(hp_velocity_error ** 2)))
        if len(hp_velocity_error) else 0.0,
        "raw_highpass_velocity_error_rms_px_per_frame": float(np.sqrt(np.mean(raw_hp_velocity_error ** 2)))
        if len(raw_hp_velocity_error) else 0.0,
        "static_output_drift_px_per_frame": float(slope) if np.ptp(intended_x) <= 1e-9 else None,
        "frames": len(output_x),
        "fps": float(fps),
    }


def run_case(width=1920, height=1080, frames=60, fps=30.0, pan_px_per_frame=0.5, jitter_px=0.0,
             seed=17, mode="l1", sigma=15.0, fidelity=0.1, budget_fraction=0.04):
    if width < 1 or height < 1 or frames < 5 or fps <= 0 or jitter_px < 0:
        raise ValueError("width/height must be positive, frames >= 5, fps positive, and jitter nonnegative")
    t = np.arange(frames, dtype=np.float64)
    intended_x = np.zeros(frames) if pan_px_per_frame == 0 else pan_px_per_frame * t
    rng = np.random.default_rng(seed)
    noise = gaussian_filter1d(rng.standard_normal(frames), 0.8)
    noise = noise / (noise.std() + 1e-12) * jitter_px
    actual_x = intended_x + noise
    config = Config(mesh=(0, 0), rs_rows=0, smooth=mode, sigma=sigma,
                    budget=budget_fraction, fidelity=fidelity)
    plan_obj = exact_motion(width, height, fps, intended_x, actual_x, config)
    metrics = _path_metrics(plan_obj, intended_x, actual_x, fps)
    return {
        "trajectory": "static" if pan_px_per_frame == 0 else ("pan_positive" if pan_px_per_frame > 0 else "pan_negative"),
        "pan_px_per_frame": float(pan_px_per_frame),
        "jitter_px_std": float(jitter_px),
        "smoother": {"mode": mode, "sigma_frames": float(sigma), "fidelity": float(fidelity),
                     "budget_fraction_of_width": float(budget_fraction),
                     "budget_px": float(budget_fraction * width)},
        "metrics": metrics,
        "scope": "Exact translation Fits through production plan(); bypasses flow, rendering, crop, parallax and rolling shutter.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--pan-px-per-frame", type=float, default=0.5)
    parser.add_argument("--jitter-px", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--mode", choices=("l1", "gauss"), default="l1")
    parser.add_argument("--sigma", type=float, default=15.0)
    parser.add_argument("--fidelity", type=float, default=0.1)
    parser.add_argument("--budget-fraction", type=float, default=0.04)
    parser.add_argument("--fidelities", type=str, help="comma-separated L1 fidelity sweep; runs one output per value")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    fidelities = ([float(v) for v in args.fidelities.split(",")] if args.fidelities else [args.fidelity])
    if not fidelities or any(not np.isfinite(v) or v < 0 for v in fidelities):
        parser.error("--fidelities must contain finite nonnegative values")
    results = [run_case(args.width, args.height, args.frames, args.fps, args.pan_px_per_frame,
                        args.jitter_px, args.seed, args.mode, args.sigma, fidelity, args.budget_fraction)
               for fidelity in fidelities]
    payload = {"schema": "mlxstab.path_smoother_gain_audit/1", "results": results}
    Path(args.out).write_text(dumps_json(payload) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
