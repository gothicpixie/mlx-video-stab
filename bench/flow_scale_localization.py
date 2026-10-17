"""Offline time-block audit of two already-rendered flow-scale benchmark outputs.

Re-evaluates only the stabilized videos with one fixed RAFT evaluator. This is a
descriptive localization tool, not a new benchmark score or a calibrated
uncertainty estimate.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench import manifest as MF
from mlxstab import media, metrics
from mlxstab.stabilize import Config, RaftFlow, analyze


def _wobble_per_fit(fit):
    if fit.fallback:
        return np.nan
    rows = fit.rows
    y = np.arange(len(rows))
    residuals = [rows[:, k] - np.polyval(np.polyfit(y, rows[:, k], 1), y)
                 for k in (0, 1)]
    return float(np.sqrt(np.mean(np.square(residuals))))


def block_summaries(mot, block_count=10):
    """Return whole-window-consistent jitter plus descriptive contiguous blocks."""
    W, H = mot.W, mot.H
    pts = np.array([[W / 2, H / 2], [0, 0], [W, 0], [W, H], [0, H]], np.float64)
    velocity = np.array([(metrics.G.apply_h(f.H, pts) - pts) for f in mot.fits])
    velocity = velocity.reshape(len(velocity), -1)
    from scipy.ndimage import gaussian_filter1d
    highpass = velocity - gaussian_filter1d(velocity, 15.0, axis=0, mode="nearest")
    jitter = np.abs(highpass).mean(axis=1)
    wobble = np.array([_wobble_per_fit(f) for f in mot.fits])
    edges = np.linspace(0, len(mot.fits), block_count + 1, dtype=int)
    blocks = []
    for index, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        usable = np.isfinite(wobble[a:b])
        blocks.append({
            "block": index,
            "first_pair": int(a),
            "last_pair_exclusive": int(b),
            "start_s": float(a / mot.fps),
            "end_s": float(b / mot.fps),
            "pairs": int(b - a),
            "jitter_px": float(jitter[a:b].mean()) if b > a else None,
            "wobble_px": float(wobble[a:b][usable].mean()) if usable.any() else None,
            "fallback_pairs": int(sum(f.fallback for f in mot.fits[a:b])),
        })
    return {
        "whole_window": {
            "jitter_px": float(jitter.mean()) if len(jitter) else None,
            "wobble_px": metrics.wobble(mot),
            "fallback_pairs": int(sum(f.fallback for f in mot.fits)),
            "support_pairs": len(mot.fits),
        },
        "blocks": blocks,
    }


def evaluate_video(path, mask_path, weights, fps_filter, fps, iters=12, downscale=2,
                   block_count=10):
    info = media.probe(path)
    frames, actual_fps, (W, H) = media.decode(path, info, vf=fps_filter)
    mask_info = media.probe(mask_path)
    masks, _, _ = media.decode(mask_path, mask_info, vf=fps_filter)

    def boolean_masks():
        for mask in masks:
            yield mask[..., 0] > 127

    flow = RaftFlow(weights, iters, downscale, True, mx.float32, mx.float32, None)
    flow.reset()
    cycle = []
    magnitude = []

    def sample_diagnostics(prev_i, i, xs, ys, forward, backward, valid, fb_err, fb_tol, fb_inside):
        mag = np.linalg.norm(forward, axis=-1)
        cycle.append({
            "pair": int(i - 1),
            "valid_fraction": float(np.mean(valid)),
            "inside_fraction": float(np.mean(fb_inside)),
            "fb_error_px_mean": float(np.mean(fb_err)),
            "fb_tolerance_px_mean": float(np.mean(fb_tol)),
            "flow_magnitude_px_mean": float(np.mean(mag)),
        })

    mot = analyze(frames, flow, Config(mesh=(0, 0), rs_rows=18), W, H, actual_fps,
                  masks=boolean_masks(), flow_sample_callback=sample_diagnostics)
    result = block_summaries(mot, block_count=block_count)
    # The callback diagnostics are compact per-pair records, then summarized over
    # the exact same contiguous block boundaries as the motion metrics.
    for block in result["blocks"]:
        records = [r for r in cycle if block["first_pair"] <= r["pair"] < block["last_pair_exclusive"]]
        for key in ("valid_fraction", "inside_fraction", "fb_error_px_mean",
                    "fb_tolerance_px_mean", "flow_magnitude_px_mean"):
            vals = [r[key] for r in records]
            block[key] = float(np.mean(vals)) if vals else None
    result["meta"] = {
        "path": str(path), "mask_path": str(mask_path),
        "frames": mot.n_frames, "pairs": len(mot.fits),
        "fps": float(actual_fps), "expected_fps": fps,
        "width": int(W), "height": int(H),
        "evaluator": {"backend": "mlx RAFT", "iters": iters,
                      "downscale": downscale, "flow_dtype": "float32",
                      "corr_dtype": "float32", "warm_start": True},
        "cycle_diagnostic_note": "Forward-backward cycle consistency is a diagnostic, not correspondence ground truth or calibrated uncertainty.",
    }
    return result


def validate_matched_manifests(path2, path4):
    """Reject comparisons whose source, evaluator, window, or timelines differ."""
    m2, m4 = json.loads(Path(path2).read_text()), json.loads(Path(path4).read_text())
    for label, manifest in (("scale2", m2), ("scale4", m4)):
        if manifest.get("status") != "success":
            raise ValueError(f"{label} manifest did not complete successfully")
        if not manifest.get("metrics", {}).get("cadence", {}).get("aligned"):
            raise ValueError(f"{label} output/mask/evaluation cadence is not aligned")
    if m2.get("source", {}).get("sha256") != m4.get("source", {}).get("sha256"):
        raise ValueError("manifests refer to different source content")
    if m2.get("source", {}).get("window") != m4.get("source", {}).get("window"):
        raise ValueError("manifests use different source windows")
    if m2.get("evaluation") != m4.get("evaluation"):
        raise ValueError("manifests use different evaluators or evaluation settings")
    c2, c4 = dict(m2.get("config") or {}), dict(m4.get("config") or {})
    c2.pop("flow_downscale", None)
    c4.pop("flow_downscale", None)
    if c2 != c4:
        raise ValueError("stabilization configs differ beyond flow_downscale")
    if m2.get("config", {}).get("flow_downscale") != 2 or m4.get("config", {}).get("flow_downscale") != 4:
        raise ValueError("expected scale2 and scale4 stabilization manifests")
    return {
        "scale2_manifest": str(path2), "scale4_manifest": str(path4),
        "source_sha256": m2["source"]["sha256"],
        "window": m2["source"]["window"],
        "matched_config_fields_except_flow_downscale": True,
        "evaluator_settings_matched": True,
        "cadence_aligned": True,
    }


def validate_replay_against_manifest(result, manifest_path, label):
    """Require the offline reduction to reproduce the benchmark's saved aggregate metrics."""
    manifest = json.loads(Path(manifest_path).read_text())
    saved = manifest["metrics"]["output"]
    got = result["whole_window"]
    for key, actual_key in (("jitter_px", "jitter_px"), ("wobble_px", "wobble_px")):
        expected = saved.get(key)
        actual = got.get(actual_key)
        if expected is None or actual is None or not math.isclose(
                float(expected), float(actual), rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(f"{label} replay {actual_key}={actual!r} does not match manifest {expected!r}")
    if saved.get("fallback_frames") != got["fallback_pairs"]:
        raise ValueError(f"{label} replay fallback count does not match manifest")
    return {
        "jitter_px_abs_delta": abs(float(saved["jitter_px"]) - got["jitter_px"]),
        "wobble_px_abs_delta": abs(float(saved["wobble_px"]) - got["wobble_px"]),
        "fallback_count_matched": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale2-manifest", required=True)
    parser.add_argument("--scale4-manifest", required=True)
    parser.add_argument("--scale2-video", required=True)
    parser.add_argument("--scale2-mask", required=True)
    parser.add_argument("--scale4-video", required=True)
    parser.add_argument("--scale4-mask", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--fps-filter", default="fps=60000/1001")
    parser.add_argument("--expected-fps", type=float, default=60000 / 1001)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.blocks < 1:
        parser.error("--blocks must be positive")

    # Retain decoded evidence only; all private input paths are kept under ignored out/.
    result2 = evaluate_video(args.scale2_video, args.scale2_mask, args.weights,
                             args.fps_filter, args.expected_fps, block_count=args.blocks)
    result4 = evaluate_video(args.scale4_video, args.scale4_mask, args.weights,
                             args.fps_filter, args.expected_fps, block_count=args.blocks)
    payload = {
        "schema": "mlxstab.flow_scale_localization/1",
        "purpose": "Descriptive time-block localization on existing rendered outputs; not an equivalence test.",
        "provenance": validate_matched_manifests(args.scale2_manifest, args.scale4_manifest),
        "scale2": result2,
        "scale4": result4,
        "replay_validation": {
            "scale2": validate_replay_against_manifest(result2, args.scale2_manifest, "scale2"),
            "scale4": validate_replay_against_manifest(result4, args.scale4_manifest, "scale4"),
        },
    }
    payload["blocks"] = args.blocks
    Path(args.out).write_text(MF.dumps_json(payload) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": args.out,
        "scale2": payload["scale2"]["whole_window"],
        "scale4": payload["scale4"]["whole_window"],
    }, indent=2))


if __name__ == "__main__":
    main()
