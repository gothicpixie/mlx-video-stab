"""Factorial audit of KLT feature-detection and tracking image domains.

This research-only tool crosses luminance/high-pass feature detection with luminance/high-pass LK tracking on
the exact same decoded frames. Luminance/high-pass are encoded to uint8 because OpenCV pyramidal LK requires
CV_8U. The high-pass scale is frozen across the whole supplied window (or passed from a calibration run), never
renormalized frame by frame. The tracking-domain contrast is paired within each detector's fixed feature set;
the detection-domain contrast changes the selected features and is therefore observational.

    python -m bench.track_domain_audit --frames-npz F.npz --out R.json
    python -m bench.track_domain_audit --frames-npz HELDOUT.npz --hp-scale 2.3 --out R-heldout.json

No RAFT inference, media decode, stabilization, or production default change occurs here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from bench import manifest as MF
from mlxstab import metrics as M, tracks

SCHEMA = "mlxstab.track_domain_audit/1"
DOMAINS = ("lum", "hp")
CELLS = {"LL": ("lum", "lum"), "LH": ("lum", "hp"),
         "HL": ("hp", "lum"), "HH": ("hp", "hp")}


def _representations(frames, *, hp_sigma=1.5, hp_scale=None):
    """Make same-size uint8 luminance/high-pass images; return the frozen scale and clipping fraction."""
    cv2 = tracks._require_cv2()
    lum, raw_hp = [], []
    for frame in frames:
        gray = tracks._gray(frame)
        smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), float(hp_sigma),
                                  borderType=cv2.BORDER_REFLECT)
        lum.append(gray)
        raw_hp.append(gray.astype(np.float32) - smooth)
    if not lum:
        raise ValueError("frames must contain at least one frame")
    if hp_scale is None:
        # Estimate once for this calibration window; output the chosen value so a separate held-out window can
        # reuse it unchanged. 99.5th percentile maps to +/-127 while preserving high-pass sign about 128.
        sample = np.concatenate([np.abs(x[::4, ::4]).ravel() for x in raw_hp])
        q = float(np.percentile(sample, 99.5))
        if not np.isfinite(q) or q <= 1e-8:
            raise ValueError("cannot calibrate high-pass encoding from a constant window")
        hp_scale = 127.0 / q
    hp_scale = float(hp_scale)
    if not np.isfinite(hp_scale) or hp_scale <= 0:
        raise ValueError("hp_scale must be a finite positive number")
    hp, clipped, count = [], 0, 0
    for image in raw_hp:
        scaled = image * hp_scale
        clipped += int(np.count_nonzero(np.abs(scaled) > 127.0))
        count += int(scaled.size)
        hp.append(np.clip(np.rint(128.0 + scaled), 0, 255).astype(np.uint8))
    return [{"lum": l, "hp": h} for l, h in zip(lum, hp, strict=True)], hp_scale, clipped / count


def _motion_metrics(transforms, width, height, fps, sigma):
    """Return the repository's path jitter and the aligned per-pair residual contributions."""
    fallbacks = sum(bool(x[1]) for x in transforms)
    n = len(transforms)
    if n < 2 or fallbacks:
        return {"jitter_px": None, "frame_motion_px": None, "stability": None,
                "jitter_contributions_px": None}
    fits = [tracks.MO.Fit(H, np.zeros((2, 2, 2)), np.zeros((1, 2)), float(inlier), 0.0, False)
            for H, _fallback, inlier, _ntracks, _detail in transforms]
    motion = tracks._Motion(width, height, fps, fits=fits, stride=1, n_frames=n + 1)
    jitter, frame_motion = M.jitter(motion, sigma=sigma)
    points = np.array([[width / 2, height / 2], [0, 0], [width, 0], [width, height], [0, height]], np.float64)
    velocity = np.asarray([tracks.G.apply_h(f.H, points) - points for f in fits]).reshape(n, -1)
    residual = velocity - gaussian_filter1d(velocity, sigma, axis=0, mode="nearest")
    contributions = np.abs(residual).mean(axis=1)
    return {"jitter_px": float(jitter), "frame_motion_px": float(frame_motion),
            "stability": min(M.stability(M.path_series(motion))),
            "jitter_contributions_px": contributions.tolist()}


def _run_cell(reps, detector, tracker, *, width, height, fps, seed, tracker_kwargs, sigma,
              invalid_masks=None):
    transforms, rows = [], []
    for i in range(len(reps) - 1):
        # Independent stream per cell and pair prevents a changed candidate count in one cell from shifting
        # the RANSAC samples used by another cell.
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), list(CELLS).index(
            next(k for k, v in CELLS.items() if v == (detector, tracker))), i]))
        result = tracks.pair_similarity_domains(
            reps[i][detector], reps[i][tracker], reps[i + 1][tracker],
            prev_invalid=None if invalid_masks is None else invalid_masks[i],
            next_invalid=None if invalid_masks is None else invalid_masks[i + 1],
            rng=rng, **tracker_kwargs)
        transforms.append(result)
        H, fallback, inlier_frac, n_tracks, detail = result
        rows.append({"pair": i, "fallback": bool(fallback), "inlier_frac": float(inlier_frac),
                     "n_detected": int(detail["n_detected"]), "n_tracks": int(n_tracks),
                     "n_inliers": int(detail["n_inliers"]), "failure_stage": detail["failure_stage"]})
    metrics = _motion_metrics(transforms, width, height, fps, sigma)
    n = len(transforms)
    return {"detector": detector, "tracker": tracker,
            "fallback_pairs": int(sum(r["fallback"] for r in rows)),
            "fallback_frac": float(sum(r["fallback"] for r in rows) / n) if n else None,
            "n_detected_mean": float(np.mean([r["n_detected"] for r in rows])) if rows else None,
            "n_tracks_mean": float(np.mean([r["n_tracks"] for r in rows])) if rows else None,
            "n_inliers_mean": float(np.mean([r["n_inliers"] for r in rows])) if rows else None,
            "inlier_frac_mean": float(np.mean([r["inlier_frac"] for r in rows])) if rows else None,
            **metrics, "pairs": rows, "_transforms": transforms}


def run_audit(frames, *, fps, hp_sigma=1.5, hp_scale=None, seed=0, sigma=15.0, tracker_kwargs=None,
              source_meta=None, invalid_masks=None):
    """Run all four cells and report paired main/interaction contrasts on common full support."""
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(f"frames must be [T,H,W,3] uint8, got {frames.shape} {frames.dtype}")
    if len(frames) < 4:
        raise ValueError("at least four frames are required")
    if invalid_masks is not None:
        invalid_masks = np.asarray(invalid_masks, bool)
        if invalid_masks.shape != frames.shape[:3]:
            raise ValueError(f"invalid_masks must have shape {frames.shape[:3]}, got {invalid_masks.shape}")
    reps, effective_scale, clip_frac = _representations(frames, hp_sigma=hp_sigma, hp_scale=hp_scale)
    h, w = frames.shape[1:3]
    kwargs = {**tracks.DEFAULTS, **(tracker_kwargs or {})}
    kwargs.pop("seed", None)
    cells = {}
    for name, (detector, tracker) in CELLS.items():
        cells[name] = _run_cell(reps, detector, tracker, width=w, height=h, fps=float(fps), seed=seed,
                                tracker_kwargs=kwargs, sigma=float(sigma), invalid_masks=invalid_masks)
    n_pairs = len(frames) - 1
    common = np.ones(n_pairs, dtype=bool)
    for cell in cells.values():
        common &= np.asarray([not row["fallback"] for row in cell["pairs"]], bool)
    valid_edges = np.flatnonzero(common)
    runs = np.split(valid_edges, np.flatnonzero(np.diff(valid_edges) != 1) + 1) if valid_edges.size else []
    best = max(runs, key=len) if runs else np.array([], dtype=int)
    start = int(best[0]) if best.size else 0
    stop = int(best[-1] + 1) if best.size else 0
    common_metrics = {}
    if len(best) >= 2:
        for name, cell in cells.items():
            common_metrics[name] = _motion_metrics(cell["_transforms"][start:stop], w, h,
                                                   float(fps), float(sigma))
            cell["common_contiguous_support"] = {
                "pair_start_in_window": start, "pair_stop_exclusive_in_window": stop,
                "pair_count": int(len(best)), **common_metrics[name]}
    contrasts = {"common_supported_pairs": int(common.sum()), "total_pairs": int(n_pairs),
                 "common_contiguous_pair_range": [start, stop] if best.size else None,
                 "common_contiguous_pairs": int(len(best)),
                 "track_main_effect_hp_minus_lum_px": None,
                 "detection_main_effect_hp_minus_lum_px": None,
                 "interaction_px": None}
    if len(best) >= 2:
        j = {k: np.asarray(common_metrics[k]["jitter_contributions_px"], np.float64) for k in cells}
        track = ((j["LH"] - j["LL"]) + (j["HH"] - j["HL"])) / 2.0
        detect = ((j["HL"] - j["LL"]) + (j["HH"] - j["LH"])) / 2.0
        interaction = (j["HH"] - j["HL"]) - (j["LH"] - j["LL"])
        contrasts.update({"track_main_effect_hp_minus_lum_px": float(track.mean()),
                          "detection_main_effect_hp_minus_lum_px": float(detect.mean()),
                          "interaction_px": float(interaction.mean()),
                          "track_main_effect_pair_values_px": track.tolist(),
                          "detection_main_effect_pair_values_px": detect.tolist(),
                          "interaction_pair_values_px": interaction.tolist()})
    for cell in cells.values():
        cell.pop("_transforms", None)
    return {"schema": SCHEMA, "status": "success", "window": {
                "frames": int(len(frames)), "pairs": int(n_pairs), "width": int(w), "height": int(h),
                "fps": float(fps), **(source_meta or {})},
            "representation": {"luminance": "decoded grayscale uint8",
                               "highpass": "gray - GaussianBlur(gray, sigma); signed offset encoding centered at 128",
                               "hp_sigma_px": float(hp_sigma), "hp_scale": float(effective_scale),
                               "hp_clip_frac": float(clip_frac), "scale_scope": "frozen for all frames in this run",
                               "highpass_scale_calibration": "derived from this window" if hp_scale is None else "provided frozen scale"},
            "validity_mask": (None if invalid_masks is None else {
                "semantics": "true means invalid/no source; same masks applied to every factorial cell",
                "invalid_fraction_mean": float(invalid_masks.mean()),
                "invalid_fraction_max": float(invalid_masks.reshape(len(frames), -1).mean(axis=1).max())}),
            "tracker_config": {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                               for k, v in kwargs.items()},
            "seed": int(seed), "jitter_sigma_frames": float(sigma), "cells": cells,
            "contrasts": contrasts,
            "interpretation": [
                "The tracking-domain effect is paired within a fixed detector and point set.",
                "The detection-domain effect changes the selected corners; treat it as observational and inspect counts.",
                "Any fallback makes that cell's full-path jitter unavailable; contrasts use only the longest contiguous segment supported by all four cells.",
                "KLT is an independent estimator, not ground truth; lower jitter alone does not prove better stabilization.",
                "This is one serially dependent window; it selects no production threshold or default."]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-npz", required=True, help="decoded-frame NPZ containing frames and optional meta")
    ap.add_argument("--out", required=True, help="atomic path-free JSON report")
    ap.add_argument("--fps", type=float, help="override cached frame rate")
    ap.add_argument("--hp-sigma", type=float, default=1.5)
    ap.add_argument("--hp-scale", type=float, help="reuse a frozen scale from a calibration report")
    ap.add_argument("--max-corners", type=int, default=tracks.DEFAULTS["max_corners"],
                    help="shared GFTT corner cap for all cells (lower to compare detectors at matched counts)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=15.0, help="jitter high-pass sigma in frames")
    a = ap.parse_args(argv)
    try:
        with np.load(a.frames_npz, allow_pickle=False) as z:
            if "frames" not in z.files:
                raise ValueError("frames NPZ missing 'frames'")
            frames = z["frames"]
            invalid_masks = z["invalid_masks"] if "invalid_masks" in z.files else None
            meta = json.loads(str(z["meta"].item())) if "meta" in z.files else {}
        fps = float(a.fps if a.fps is not None else meta.get("fps", 0.0))
        if fps <= 0 or not np.isfinite(fps):
            raise ValueError("positive --fps or cached meta.fps is required")
        window = meta.get("window") or {}
        source_meta = {"source_sha256": meta.get("source_sha256"),
                       "start_s": window.get("start_s"), "duration_s": window.get("duration_s"),
                       "frames_sha256": meta.get("frames_sha256")}
        result = run_audit(frames, fps=fps, hp_sigma=a.hp_sigma, hp_scale=a.hp_scale,
                           seed=a.seed, sigma=a.sigma, tracker_kwargs={"max_corners": a.max_corners},
                           source_meta=source_meta,
                           invalid_masks=invalid_masks)
        MF.write_manifest(a.out, result)
        print(json.dumps({"status": "success", "out": a.out, "hp_scale": result["representation"]["hp_scale"],
                          "common_supported_pairs": result["contrasts"]["common_supported_pairs"]}, sort_keys=True))
        return 0
    except Exception as exc:  # stable CLI failure, with no partial report
        print(json.dumps({"status": "error", "code": "track_domain_audit_failed", "message": str(exc)}),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
