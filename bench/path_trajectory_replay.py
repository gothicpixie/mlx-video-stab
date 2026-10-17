"""Replay one fixed camera trajectory through alternative path smoothers.

The real-video arm reuses one KLT Motion estimate, so smoother differences cannot be attributed to re-running
tracking. It remains descriptive because KLT has no ground-truth trajectory. The synthetic arm starts from a
known path and injects controlled pair-fit translation outliers, providing causal outlier-recovery controls.
Neither arm renders or selects a production default.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kurtosis

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.manifest import dumps_json, sha256_file
from mlxstab import geometry as G, media, metrics as M, tracks
from mlxstab.motion import Fit
from mlxstab.stabilize import Config, Motion, plan

SCHEMA = "mlxstab.path_trajectory_replay/1"
MODES = ("l1", "gauss", "butter")


def _relative_path(xy):
    xy = np.asarray(xy, np.float64)
    return xy - xy[0]


def _plan_xy(pl):
    return _relative_path(np.asarray(pl.path_quads, np.float64).mean(axis=1))


def _robust_residual_diagnostics(increments, sigma=3.0):
    d = np.asarray(increments, np.float64)
    smooth = gaussian_filter1d(d, sigma, axis=0, mode="nearest") if len(d) else d.copy()
    residual = d - smooth
    center = np.median(residual, axis=0) if len(residual) else np.zeros(d.shape[-1])
    radial = np.linalg.norm(residual - center, axis=-1) if len(residual) else np.zeros(0)
    med = float(np.median(radial)) if len(radial) else 0.0
    mad = float(np.median(np.abs(radial - med))) if len(radial) else 0.0
    scale = 1.4826 * mad
    threshold = med + 4.5 * scale
    kurt = float(kurtosis(radial, fisher=True, bias=False)) if len(radial) > 3 else None
    if kurt is not None and not np.isfinite(kurt):
        kurt = None
    return {
        "residual_kurtosis_excess": kurt,
        "residual_outlier_fraction_mad45": float(np.mean(radial > threshold)) if len(radial) else None,
        "residual_rms": float(np.sqrt(np.mean(residual ** 2))) if residual.size else None,
        "pairs": int(len(d)),
    }


def _path_metrics(path_xy, reference_xy, *, smooth_sigma=15.0):
    path = np.asarray(path_xy, np.float64)
    ref = np.asarray(reference_xy, np.float64)
    if path.shape != ref.shape or path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path_xy and reference_xy must have matching [T,2] shapes")
    err = path - ref
    low_path = gaussian_filter1d(path, smooth_sigma, axis=0, mode="nearest")
    low_ref = gaussian_filter1d(ref, smooth_sigma, axis=0, mode="nearest")
    high_path = path - low_path
    high_ref = ref - low_ref
    vel = np.diff(path, axis=0)
    vel_trend = gaussian_filter1d(vel, 3.0, axis=0, mode="nearest") if len(vel) else vel
    return {
        "path_rmse_px": float(np.sqrt(np.mean(err ** 2))),
        "low_frequency_path_rmse_px": float(np.sqrt(np.mean((low_path - low_ref) ** 2))),
        "high_frequency_path_rmse_px": float(np.sqrt(np.mean((high_path - high_ref) ** 2))),
        "highpass_velocity_rms_px_per_frame": float(np.sqrt(np.mean((vel - vel_trend) ** 2))) if len(vel) else 0.0,
    }


def _translation_motion(path_xy, fps=30.0):
    path = np.asarray(path_xy, np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 5 or not np.isfinite(path).all():
        raise ValueError("path_xy must be finite [T,2] with T >= 5")
    fits = []
    mesh = np.zeros((1, 1, 2), np.float64)
    rows = np.zeros((0, 2), np.float64)
    for dx, dy in np.diff(path, axis=0):
        H = np.eye(3, dtype=np.float64)
        H[:2, 2] = (-dx, -dy)
        fits.append(Fit(H=H, mesh=mesh.copy(), rows=rows.copy(), inlier_frac=1.0, mover_frac=0.0, fallback=False))
    return Motion(1920, 1080, float(fps), fits=fits, n_frames=len(path))


def _perturb_fit_translations(motion, *, fraction, amplitude_px, seed):
    if not 0 <= fraction <= 1 or amplitude_px < 0:
        raise ValueError("fraction must be in [0,1] and amplitude_px nonnegative")
    rng = np.random.default_rng(seed)
    out = Motion(motion.W, motion.H, motion.fps, fits=[], stride=motion.stride, n_frames=motion.n_frames)
    n = len(motion.fits)
    count = int(round(fraction * n))
    selected = np.sort(rng.choice(n, count, replace=False)) if count else np.zeros(0, dtype=int)
    selected_set = set(selected.tolist())
    for i, fit in enumerate(motion.fits):
        H = np.asarray(fit.H, np.float64).copy()
        if i in selected_set:
            # Corrupt a measured pair transform in a random 2-D direction. The corruption is pair-local in the
            # observations; integrating it into the path exposes the downstream persistence of a bad fit.
            theta = rng.uniform(0, 2 * np.pi)
            H[0, 2] += amplitude_px * np.cos(theta)
            H[1, 2] += amplitude_px * np.sin(theta)
        out.fits.append(Fit(H=H, mesh=np.asarray(fit.mesh).copy(), rows=np.asarray(fit.rows).copy(),
                            inlier_frac=fit.inlier_frac, mover_frac=fit.mover_frac, fallback=fit.fallback))
    return out, selected.tolist()


def run_synthetic(*, frames=600, fps=30.0, seed=0, profile="static", amplitude_px=0.0,
                  period_frames=120.0, jitter_px=0.0, jitter_correlation_frames=0.8,
                  outlier_fraction=0.0, outlier_amplitude_px=0.0, outlier_seed=101,
                  modes=MODES, gauss_sigma=10.0, butter_cutoff_period=45.0):
    if frames < 5 or fps <= 0 or jitter_px < 0 or jitter_correlation_frames < 0:
        raise ValueError("frames/fps must be positive and jitter settings nonnegative")
    t = np.arange(frames, dtype=np.float64)
    if profile == "static":
        intended = np.zeros((frames, 2), np.float64)
    elif profile == "sinusoid":
        if amplitude_px <= 0 or period_frames <= 0:
            raise ValueError("sinusoid amplitude and period must be positive")
        intended = np.column_stack((amplitude_px * np.sin(2 * np.pi * t / period_frames), np.zeros(frames)))
    elif profile == "linear":
        intended = np.column_stack((amplitude_px * t, np.zeros(frames)))
    else:
        raise ValueError("profile must be static, sinusoid, or linear")
    rng = np.random.default_rng(seed)
    jitter = gaussian_filter1d(rng.standard_normal((frames, 2)), jitter_correlation_frames, axis=0)
    jitter = jitter / (jitter.std(axis=0, keepdims=True) + 1e-12) * jitter_px
    clean_path = intended + jitter
    clean_motion = _translation_motion(clean_path, fps)
    corrupt_motion, selected = _perturb_fit_translations(clean_motion, fraction=outlier_fraction,
                                                          amplitude_px=outlier_amplitude_px, seed=outlier_seed)
    raw_path = _relative_path(M.path_series(corrupt_motion)[:, :2])
    clean_reference = _relative_path(intended)
    cfg_rows = []
    for mode in modes:
        sigma = gauss_sigma if mode == "gauss" else butter_cutoff_period
        cfg = Config(mesh=(0, 0), rs_rows=0, smooth=mode, sigma=sigma, budget=0.04, fidelity=0.1)
        pl = plan(corrupt_motion, cfg, n_frames=frames)
        output_path = _plan_xy(pl)
        cfg_rows.append({
            "mode": mode,
            "parameter": float(sigma) if mode != "l1" else {"fidelity": 0.1, "budget_fraction": 0.04},
            "metrics_vs_intended_path": _path_metrics(output_path, clean_reference),
            "raw_input_metrics_vs_intended_path": _path_metrics(raw_path, clean_reference),
            "input_high_frequency_residual": _robust_residual_diagnostics(np.diff(raw_path, axis=0)),
            "output_path_quads": int(len(pl.path_quads)),
        })
    raw_rmse = _path_metrics(raw_path, clean_reference)["path_rmse_px"]
    for row in cfg_rows:
        output_rmse = row["metrics_vs_intended_path"]["path_rmse_px"]
        row["output_error_ratio_vs_raw_error"] = output_rmse / raw_rmse if raw_rmse > 1e-12 else 0.0
        row["low_frequency_error_over_raw_error"] = (
            row["metrics_vs_intended_path"]["low_frequency_path_rmse_px"] / raw_rmse
            if raw_rmse > 1e-12 else 0.0
        )
    return {
        "schema": SCHEMA,
        "arm": "synthetic_known_trajectory_outlier_replay",
        "profile": profile,
        "frames": frames,
        "fps": fps,
        "seed": seed,
        "jitter_px": jitter_px,
        "jitter_correlation_frames": jitter_correlation_frames,
        "outliers": {"fraction_requested": outlier_fraction, "count": len(selected),
                     "amplitude_px": outlier_amplitude_px, "fit_indices": selected,
                     "outlier_seed": outlier_seed},
        "raw_error_rmse_px": raw_rmse,
        "results": cfg_rows,
        "scope": "Exact translation Fits through production plan(); synthetic ground truth is known. No flow, image rendering, parallax, rolling shutter or crop measurement.",
    }


def run_real(*, source, start, duration, modes=MODES, gauss_sigma=10.0, butter_cutoff_period=45.0):
    path = Path(source)
    info = media.probe(path)
    frames, fps, (width, height) = media.decode(path, info, start, duration)
    acc = tracks.TrackAccumulator(width, height, fps, seed=0)
    count = 0
    for frame in frames:
        acc.push(frame)
        count += 1
    motion = acc.motion()
    raw_series = M.path_series(motion)
    raw_xy = _relative_path(raw_series[:, :2])
    input_metrics = acc.metrics()
    result_rows = []
    for mode in modes:
        sigma = gauss_sigma if mode == "gauss" else butter_cutoff_period
        cfg = Config(mesh=(0, 0), rs_rows=0, smooth=mode, sigma=sigma, budget=0.04, fidelity=0.1)
        pl = plan(motion, cfg, n_frames=count)
        output_xy = _plan_xy(pl)
        result_rows.append({
            "mode": mode,
            "parameter": float(sigma) if mode != "l1" else {"fidelity": 0.1, "budget_fraction": 0.04},
            "path_spectrum_metrics": _path_metrics(output_xy, raw_xy),
            "input_fit_residual_diagnostics": _robust_residual_diagnostics(np.diff(raw_xy, axis=0)),
            "correction_rmse_px": _path_metrics(output_xy, raw_xy)["path_rmse_px"],
            "path_frames": len(pl.path_quads),
            "planner_fallbacks": int(pl.fallbacks),
        })
    return {
        "schema": SCHEMA,
        "arm": "real_klt_fixed_trajectory_replay_descriptive",
        "source_sha256": sha256_file(path),
        "source_name": path.name,
        "window": {"start_s": start, "duration_s": duration},
        "decode": {"width": width, "height": height, "fps": fps, "frames": count,
                   "interlaced": bool(info.get("interlaced"))},
        "klt_input_metrics": input_metrics,
        "raw_path_residual_diagnostics": _robust_residual_diagnostics(np.diff(raw_xy, axis=0)),
        "results": result_rows,
        "scope": "One KLT Motion estimate is reused for every planner configuration. Descriptive only: no ground-truth camera path, no rendering, no independent output tracker, no crop/distortion measurement.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="optional real video for fixed-KLT trajectory replay")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--profile", choices=("static", "sinusoid", "linear"), default="static")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--amplitude-px", type=float, default=0.0)
    parser.add_argument("--period-frames", type=float, default=120.0)
    parser.add_argument("--jitter-px", type=float, default=0.0)
    parser.add_argument("--jitter-correlation-frames", type=float, default=0.8)
    parser.add_argument("--outlier-fraction", type=float, default=0.0)
    parser.add_argument("--outlier-amplitude-px", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outlier-seed", type=int, default=101)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = (run_real(source=args.source, start=args.start, duration=args.duration)
               if args.source else run_synthetic(frames=args.frames, fps=args.fps, seed=args.seed,
                   profile=args.profile, amplitude_px=args.amplitude_px, period_frames=args.period_frames,
                   jitter_px=args.jitter_px, jitter_correlation_frames=args.jitter_correlation_frames,
                   outlier_fraction=args.outlier_fraction, outlier_amplitude_px=args.outlier_amplitude_px,
                   outlier_seed=args.outlier_seed))
    Path(args.out).write_text(dumps_json(payload) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
