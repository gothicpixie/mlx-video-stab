"""Ground-truth camera-pose recovery for rendered synthetic stabilization runs.

The saved stabilization plan is a backward map (output pixel -> input pixel). This module composes that map
with the known input camera-to-world map and fits the effective output-to-world similarity using only pixels
visible on one known scene layer. It reports fit residuals and support so nonrigid correction, row motion,
crop, or poor visibility cannot silently look like a clean camera pose.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from bench.manifest import sha256_file
from bench.plan_shape_audit import load_plan_npz
from mlxstab import synth
from mlxstab.stabilize import source_coords

SCHEMA = "mlxstab.synth_intent_metrics/1"
LAYER_IDS = {"far": 0, "mid": 1, "near": 2, "subject": 3}
ROOT = Path(__file__).resolve().parent.parent


def camera_to_world(pixel_xy, camera_pose, center, *, translation_factor=1.0, world_offset=(0.0, 0.0)):
    """Map input image pixels to the known layer's world coordinates.

    ``camera_pose`` is ``[tx_px, ty_px, angle_deg]`` per point or one shared pose. Translation is multiplied by
    the layer factor used by :func:`mlxstab.synth.render`; rotation is shared. ``world_offset`` is subtracted,
    which converts the moving subject's current world coordinates back to its object-local frame.
    """
    points = np.asarray(pixel_xy, np.float64)
    pose = np.asarray(camera_pose, np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("pixel_xy must be finite [N,2]")
    if pose.shape == (3,):
        pose = np.broadcast_to(pose, (len(points), 3))
    if pose.shape != (len(points), 3) or not np.isfinite(pose).all():
        raise ValueError("camera_pose must be finite [3] or [N,3]")
    center = np.asarray(center, np.float64)
    offset = np.asarray(world_offset, np.float64)
    if center.shape != (2,) or offset.shape != (2,):
        raise ValueError("center and world_offset must be [2]")
    if not np.isfinite(translation_factor) or not np.isfinite(center).all() or not np.isfinite(offset).all():
        raise ValueError("translation_factor, center and world_offset must be finite")
    theta = np.deg2rad(pose[:, 2])
    c, s = np.cos(theta), np.sin(theta)
    rel = points - center
    rotated = np.column_stack((c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]))
    return rotated + center + translation_factor * pose[:, :2] - offset


def fit_effective_camera_pose(output_xy, world_xy, center, *, robust=True):
    """Fit ``world = R_eff(output-center) + center + T_eff`` with optional scale.

    Returns the centered effective translation, angle, scale, support and residuals. The translation is in the
    camera-to-world direction; invert the fitted transform if the desired convention is world-to-camera.
    """
    output = np.asarray(output_xy, np.float64)
    world = np.asarray(world_xy, np.float64)
    center = np.asarray(center, np.float64)
    if output.ndim != 2 or output.shape[1] != 2 or world.shape != output.shape:
        raise ValueError("output_xy and world_xy must have matching [N,2] shapes")
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center must be finite [2]")
    valid = np.isfinite(output).all(axis=1) & np.isfinite(world).all(axis=1)
    output, world = output[valid], world[valid]
    if len(output) < 3:
        raise ValueError("at least three finite point correspondences are required")
    p = output - center
    w = world - center
    keep = np.ones(len(p), dtype=bool)

    def solve(mask):
        pm, wm = p[mask], w[mask]
        if len(pm) < 3 or np.linalg.matrix_rank(pm - pm.mean(axis=0)) < 2:
            raise ValueError("far-layer correspondences are degenerate")
        pc, wc = pm.mean(axis=0), wm.mean(axis=0)
        zp = pc[0] + 1j * pc[1]
        zw = wc[0] + 1j * wc[1]
        pz = pm[:, 0] + 1j * pm[:, 1]
        wz = wm[:, 0] + 1j * wm[:, 1]
        den = np.sum(np.abs(pz - zp) ** 2)
        if den <= 1e-12:
            raise ValueError("far-layer correspondences have no spatial extent")
        a = np.sum((wz - zw) * np.conj(pz - zp)) / den
        translation = zw - a * zp
        predicted = np.column_stack(((a * (pz - zp) + zw).real,
                                     (a * (pz - zp) + zw).imag))
        residual = np.linalg.norm(wm - predicted, axis=1)
        return a, translation, residual

    a, translation, residual = solve(keep)
    if robust and len(residual) >= 8:
        median = np.median(residual)
        mad = np.median(np.abs(residual - median))
        threshold = median + 4.5 * 1.4826 * mad
        candidate = residual <= max(threshold, median + 1e-6)
        if candidate.sum() >= 3 and candidate.sum() < len(keep):
            keep = candidate
            a, translation, residual = solve(keep)
    all_predicted = np.column_stack(((a * (p[:, 0] + 1j * p[:, 1])).real + translation.real,
                                     (a * (p[:, 0] + 1j * p[:, 1])).imag + translation.imag))
    all_residual = np.linalg.norm(w - all_predicted, axis=1)
    return {
        "translation_px": [float(translation.real), float(translation.imag)],
        "rotation_deg": float(np.rad2deg(np.angle(a))),
        "scale": float(np.abs(a)),
        "fit_rmse_px": float(np.sqrt(np.mean(residual ** 2))) if len(residual) else None,
        "all_support_rmse_px": float(np.sqrt(np.mean(all_residual ** 2))) if len(all_residual) else None,
        "points_total": int(len(output)),
        "points_inlier": int(keep.sum()),
        "inlier_fraction": float(keep.mean()),
    }


def recover_layer_pose_series(plan, *, actual_pose_fn, visible_layer_id, width, height, rho, layer_id=0,
                              translation_factor=1.0, world_speed_px_per_frame=0.0, sample_step=4):
    """Recover effective output camera poses on a known visible layer from a saved production ``Plan``.

    The plan's backward source coordinates are evaluated on a sparse regular output grid. They are kept only when
    the corresponding input pixel is in bounds and the ground-truth visible-layer ID matches ``layer_id``. The
    rolling-shutter input pose is evaluated at the sampled input row, matching the generator's row-time equation.
    """
    if width < 3 or height < 3 or sample_step < 1 or not 0.0 <= rho <= 1.0:
        raise ValueError("invalid dimensions, sample step, or rolling-shutter rho")
    vis = np.asarray(visible_layer_id)
    T = len(plan.G)
    if vis.shape != (T, height, width):
        raise ValueError(f"visible_layer_id must have shape {(T, height, width)}, got {vis.shape}")
    xs = np.arange(0, width, sample_step, dtype=np.float32)
    ys = np.arange(0, height, sample_step, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    output_xy = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float64)
    grid = (mx.array(grid_x), mx.array(grid_y))
    center = np.array([(width - 1.0) / 2.0, (height - 1.0) / 2.0])
    result = []
    for t in range(T):
        source = np.asarray(source_coords(plan, t, width, height, grid=grid))[0].astype(np.float64)
        sx, sy = source[..., 0].ravel(), source[..., 1].ravel()
        inside = np.isfinite(sx) & np.isfinite(sy) & (sx >= 0) & (sx <= width - 1) & (sy >= 0) & (sy <= height - 1)
        ix = np.clip(np.rint(sx).astype(np.int64), 0, width - 1)
        iy = np.clip(np.rint(sy).astype(np.int64), 0, height - 1)
        layer = inside & (vis[t, iy, ix] == int(layer_id))
        if layer.sum() < 3:
            result.append({"frame": t, "valid_fraction": float(inside.mean()), "visible_fraction": float(layer.mean()),
                           "pose": None, "unavailable_reason": "fewer than three visible in-bounds layer samples"})
            continue
        input_xy = np.column_stack((sx[layer], sy[layer]))
        pose_times = t + rho * input_xy[:, 1] / height
        camera_pose = actual_pose_fn(pose_times)
        world_offset = (float(world_speed_px_per_frame) * t, 0.0) if layer_id == LAYER_IDS["subject"] else (0.0, 0.0)
        world_xy = camera_to_world(input_xy, camera_pose, center,
                                   translation_factor=translation_factor, world_offset=world_offset)
        try:
            fitted = fit_effective_camera_pose(output_xy[layer], world_xy, center)
        except ValueError as exc:
            result.append({"frame": t, "valid_fraction": float(inside.mean()),
                           "visible_fraction": float(layer.mean()), "pose": None,
                           "unavailable_reason": str(exc)})
            continue
        result.append({"frame": t, "valid_fraction": float(inside.mean()),
                       "visible_fraction": float(layer.mean()), "pose": fitted})
    return result


def _band_component(values, bins):
    values = np.asarray(values, np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=0)
    selected = np.zeros(spectrum.shape[0], dtype=bool)
    selected[np.asarray(bins, dtype=np.int64)] = True
    return np.fft.irfft(spectrum * selected[:, None], n=len(values), axis=0)


def summarize_far_camera_path(rows, intended_pose, actual_pose, *, matched_profile, shake_band_min_cycles=0.12):
    """Summarize intent retention and shake rejection per translation/rotation channel.

    Phase-paired controls use harmonics 1..6 for intent and frequencies >=0.12 cycles/frame for shake. The two
    bands are disjoint at the recommended >=64-frame factorial length. Metrics are withheld unless all frames
    have a valid far-layer pose; per-frame failures remain explicit in the series.
    """
    intended = np.asarray(intended_pose, np.float64)
    actual = np.asarray(actual_pose, np.float64)
    T = len(rows)
    if intended.shape != (T, 3) or actual.shape != (T, 3):
        raise ValueError("intended_pose and actual_pose must be [T,3] and match the pose series")
    available = [row["pose"] is not None for row in rows]
    intent_bins = list(range(1, min(6, (T - 1) // 2) + 1)) if matched_profile else None
    max_intent_freq = (intent_bins[-1] / T) if intent_bins else 0.0
    resolved_shake_min = max(float(shake_band_min_cycles), (max_intent_freq + 1.0 / T))
    result = {"frames_expected": T, "frames_available": int(sum(available)),
              "complete_case": bool(all(available)), "shake_band_min_cycles_per_frame": float(shake_band_min_cycles),
              "resolved_shake_band_min_cycles_per_frame": resolved_shake_min,
              "recommended_minimum_frames": 64, "meets_recommended_minimum_frames": T >= 64,
              "intent_bins": intent_bins,
              "intent_retention": None, "shake_rejection": None}
    result["support"] = {
        "mean_input_valid_fraction": float(np.mean([row["valid_fraction"] for row in rows])) if rows else None,
        "mean_far_visible_fraction": float(np.mean([row["visible_fraction"] for row in rows])) if rows else None,
        "mean_pose_fit_rmse_px": float(np.mean([row["pose"]["fit_rmse_px"] for row in rows if row["pose"] is not None]))
        if any(available) else None,
        "unavailable_frames": [int(row["frame"]) for row in rows if row["pose"] is None],
    }
    if not all(available):
        result["unavailable_reason"] = "intent/shake summaries require an effective far-layer pose for every frame"
        return result

    output = np.asarray([[*row["pose"]["translation_px"], row["pose"]["rotation_deg"]] for row in rows], np.float64)
    # The matched pair has an identical non-periodic x drift. Remove that shared trend before comparing its
    # periodic intent component; report the preserved pan trend independently.
    intent_target = intended.copy()
    intent_output = output.copy()
    if matched_profile:
        t = np.arange(T, dtype=np.float64)
        intent_target[:, 0] -= 0.5 * t
        intent_output[:, 0] -= 0.5 * t
    bins = result["intent_bins"] if result["intent_bins"] is not None else list(range(1, min(6, (T - 1) // 2) + 1))
    target_band = _band_component(intent_target, bins)
    output_band = _band_component(intent_output, bins)
    axes = ("translation_x_px", "translation_y_px", "rotation_deg")
    retention = {}
    for axis, name in enumerate(axes):
        ref, pred = target_band[:, axis], output_band[:, axis]
        den = float(np.dot(ref, ref))
        corr_den = float(np.linalg.norm(ref) * np.linalg.norm(pred))
        retention[name] = {
            "gain": float(np.dot(pred, ref) / den) if den > 1e-12 else None,
            "correlation": float(np.dot(pred, ref) / corr_den) if corr_den > 1e-12 else None,
            "rmse": float(np.sqrt(np.mean((pred - ref) ** 2))),
            "target_band_rms": float(np.sqrt(np.mean(ref ** 2))),
            "output_band_rms": float(np.sqrt(np.mean(pred ** 2))),
        }
    t = np.arange(T, dtype=np.float64)
    target_drift = 0.5 if matched_profile else (np.polyfit(t, intended[:, 0], 1)[0] if T > 1 else 0.0)
    output_without_intent_band = output[:, 0] - output_band[:, 0]
    output_drift = np.polyfit(t, output_without_intent_band, 1)[0] if T > 1 else 0.0
    retention["shared_x_drift"] = {"target_px_per_frame": float(target_drift),
                                   "output_px_per_frame": float(output_drift),
                                   "gain": float(output_drift / target_drift) if abs(target_drift) > 1e-12 else None}
    shake = actual - intended
    output_error = output - intended
    freq = np.fft.rfftfreq(T)
    shake_bins = np.flatnonzero(freq >= resolved_shake_min).tolist()
    shake_rejection = {}
    if shake_bins:
        input_shake_band = _band_component(shake, shake_bins)
        output_shake_band = _band_component(output_error, shake_bins)
        for axis, name in enumerate(axes):
            in_rms = float(np.sqrt(np.mean(input_shake_band[:, axis] ** 2)))
            out_rms = float(np.sqrt(np.mean(output_shake_band[:, axis] ** 2)))
            shake_rejection[name] = {
                "input_shake_band_rms": in_rms,
                "output_residual_band_rms": out_rms,
                "attenuation_db": float(20.0 * np.log10(max(out_rms, 1e-12) / max(in_rms, 1e-12))),
            }
    result["shake_bins"] = shake_bins
    result["frequency_bands_disjoint"] = not bool(set(bins) & set(shake_bins))
    result["intent_retention"] = retention
    result["shake_rejection"] = shake_rejection if shake_bins else None
    return result


def evaluate_manifest(manifest_path, *, sample_step=4):
    """Evaluate a run manifest containing a saved plan and synthetic truth sidecar."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    source = manifest["source"]
    synthetic = source.get("synthetic")
    if not synthetic:
        raise ValueError("manifest has no synthetic scene provenance")
    plan_path = manifest.get("outputs", {}).get("plan")
    if not plan_path:
        raise ValueError("manifest has no dumped production plan; rerun with --dump-plan")
    plan_file = Path(plan_path)
    if not plan_file.is_absolute():
        plan_file = ROOT / plan_file
    plan, plan_meta = load_plan_npz(plan_file)
    truth_path = path.parent / synthetic["truth_file"]
    if sha256_file(truth_path) != synthetic["truth_sha256"]:
        raise ValueError("synthetic truth sidecar SHA-256 does not match manifest")
    with np.load(truth_path, allow_pickle=False) as z:
        visible = np.asarray(z["visible_layer_id"], np.uint8)

    intent_mode = synthetic["intent_mode"]
    intent_profile = synthetic.get("intent_profile", "historical")
    intended, actual = synth.camera(
        plan_meta["frames"], seed=int(manifest["seed_policy"]["synthetic_seed"]) + 1,
        jitter=float(synthetic["jitter_px"]), rot_jit=float(synthetic["rot_jitter_deg"]),
        intent_mode=intent_mode, intent_profile=intent_profile)
    dimensions = (int(synthetic["width"]), int(synthetic["height"]))
    if (plan_meta["frames"] != int(synthetic["frames_requested"])
            or plan_meta["width"] != dimensions[0] or plan_meta["height"] != dimensions[1]):
        raise ValueError("saved plan dimensions/frame count do not match synthetic scene provenance")
    layers = {}
    for name, layer_id in LAYER_IDS.items():
        if name == "subject" and not synthetic.get("subject_present"):
            continue
        factor = 1.0 if name in ("far", "subject") else float(synthetic["layer_motion_factors"][layer_id])
        speed = float(synthetic.get("subject_speed_px_per_frame") or 0.0) if name == "subject" else 0.0
        layers[name] = recover_layer_pose_series(
            plan, actual_pose_fn=actual, visible_layer_id=visible, width=dimensions[0], height=dimensions[1],
            rho=float(synthetic["rolling_shutter_rho"]), layer_id=layer_id, translation_factor=factor,
            world_speed_px_per_frame=speed, sample_step=sample_step)
    far_summary = summarize_far_camera_path(
        layers["far"], intended(np.arange(plan_meta["frames"])), actual(np.arange(plan_meta["frames"])),
        matched_profile=intent_profile == "matched-spectrum")
    layer_summary = {
        name: {
            "frames_expected": len(rows),
            "frames_available": int(sum(row["pose"] is not None for row in rows)),
            "mean_visible_fraction": float(np.mean([row["visible_fraction"] for row in rows])) if rows else None,
            "mean_valid_fraction": float(np.mean([row["valid_fraction"] for row in rows])) if rows else None,
            "mean_pose_fit_rmse_px": float(np.mean([row["pose"]["fit_rmse_px"] for row in rows if row["pose"] is not None]))
            if any(row["pose"] is not None for row in rows) else None,
            "unavailable_frames": [int(row["frame"]) for row in rows if row["pose"] is None],
        }
        for name, rows in layers.items()
    }
    return {"schema": SCHEMA, "run_id": manifest.get("run_id"), "plan": plan_meta,
            "scope": "Camera-to-world similarity fit of the saved output-to-input plan, using ground-truth visible-layer labels and row-time camera poses. It does not judge perceptual quality.",
            "far_camera_metrics": far_summary, "layer_support_and_distortion": layer_summary,
            "layers": layers}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-step", type=int, default=4)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = evaluate_manifest(args.manifest, sample_step=args.sample_step)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "success", "out": str(out), "layers": list(result["layers"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
