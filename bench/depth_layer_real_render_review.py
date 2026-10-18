"""Render real cached frames under pooled and relative-depth-rank affine paths.

This research-only diagnostic holds the observed frames and per-frame camera-path
policy fixed, renders RGB pixels, and builds blinded baseline-versus-rank reviews.
It does not composite depth layers or enter the production stabilizer.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from mlxstab import media
from . import depth_layer_real_runner as R
from . import depth_temporal_pilot as D
from . import manifest as MF
from .real_pair_review import build as build_review

SCHEMA = "mlxstab.depth_layer_real_render_review/1"
SWEEP_SCHEMA = "mlxstab.depth_layer_real_audit_sweep/2"
AUDIT_SCHEMA = "mlxstab.depth_layer_real_audit/2"
LIMITATIONS = [
    "the experiment renders only the sampled cached frames; intermediate video frames are not reconstructed",
    "each policy applies one global affine camera correction per frame; no depth compositing, occlusion or local warp is performed",
    "depth ranks are per-frame relative-value bins, not metric depth or semantic foreground/background labels",
    "the pooled path and rank paths are derived from the same model-estimated RAFT flow and monocular depth",
    "coverage is geometric sampler support; blind visual judgments are descriptive and do not prove general quality",
]


def pull_affine(rgb: np.ndarray, source_to_output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Backward-warp RGB by an affine map, returning uint8 pixels and conservative support.

    Pixel centers lie at integer ``(x,y)`` coordinates. ``source_to_output`` is
    the forward source-to-stabilized correction. The inverse maps each output
    center to its source position. A pixel is valid only when it is in the
    source image and every bilinear tap with nonzero weight is in bounds.
    """
    rgb = np.asarray(rgb)
    matrix = np.asarray(source_to_output, dtype=np.float64)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be uint8 [H,W,3]")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("source_to_output must be a finite 3x3 matrix")
    if not np.allclose(matrix[2], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-12):
        raise ValueError("source_to_output must be affine")
    h, w = rgb.shape[:2]
    out = np.zeros_like(rgb)
    valid = np.zeros((h, w), dtype=bool)
    if h < 2 or w < 2:
        return out, valid
    linear = matrix[:2, :2]
    try:
        inverse = np.linalg.inv(linear)
    except np.linalg.LinAlgError:
        return out, valid
    if not np.isfinite(inverse).all():
        return out, valid

    yy, xx = np.indices((h, w), dtype=np.float64)
    source = inverse @ (np.stack((xx.ravel(), yy.ravel())) - matrix[:2, 2:3])
    sx, sy = source[0], source[1]
    finite = np.isfinite(sx) & np.isfinite(sy)
    inside = finite & (sx >= 0.0) & (sx <= w - 1.0) & (sy >= 0.0) & (sy <= h - 1.0)
    safe_x, safe_y = np.where(inside, sx, 0.0), np.where(inside, sy, 0.0)
    x0, y0 = np.floor(safe_x).astype(np.intp), np.floor(safe_y).astype(np.intp)
    fx, fy = safe_x - x0, safe_y - y0
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    # Zero-weight neighbours do not contribute, including at exact edge pixels.
    valid_flat = inside & (((fx == 0.0) | (x0 + 1 < w)) & ((fy == 0.0) | (y0 + 1 < h)))
    src = rgb.astype(np.float32)
    a = src[y0, x0]
    b = src[y0, x1]
    c = src[y1, x0]
    d = src[y1, x1]
    wx, wy = fx[:, None], fy[:, None]
    sampled = (a * (1.0 - wx) * (1.0 - wy) + b * wx * (1.0 - wy)
               + c * (1.0 - wx) * wy + d * wx * wy)
    sampled[~valid_flat] = 0.0
    out[:] = np.rint(np.clip(sampled.reshape(h, w, 3), 0, 255)).astype(np.uint8)
    valid[:] = valid_flat.reshape(h, w)
    return out, valid


def _audit_variant(document: dict, variant: str | None) -> dict:
    schema = document.get("schema")
    if schema == SWEEP_SCHEMA:
        variants = document.get("variants")
        if not isinstance(variants, dict):
            raise ValueError("depth sweep report has no variants object")
        if variant is None:
            raise ValueError("--variant is required for a depth-layer sweep report")
        selected = variants.get(variant)
        if not isinstance(selected, dict):
            raise ValueError(f"requested variant {variant!r} is absent")
        return selected
    if schema == AUDIT_SCHEMA:
        if variant is not None:
            raise ValueError("--variant applies only to a depth-layer sweep report")
        return document
    raise ValueError(f"unsupported depth audit schema: {schema!r}")


def _load_inputs(frames_npz, report_json, variant=None):
    frames, meta = D.load_frames_npz(frames_npz)
    document = json.loads(Path(report_json).read_text(encoding="utf-8"))
    report = _audit_variant(document, variant)
    expected_hash = (document.get("inputs") or {}).get("frames_sha256")
    actual_hash = MF.sha256_file(frames_npz)
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError("frames NPZ hash does not match depth audit report")
    indices = np.asarray((document.get("inputs") or {}).get("sampled_frame_indices"), dtype=np.int64)
    if indices.ndim != 1 or len(indices) < 2 or np.any(np.diff(indices) <= 0):
        raise ValueError("depth audit sampled_frame_indices must be strictly increasing")
    if indices[0] < 0 or indices[-1] >= len(frames):
        raise ValueError("depth audit frame indices are outside the frame cache")
    policies = report.get("policies")
    if not isinstance(policies, dict) or "pooled" not in policies:
        raise ValueError("selected depth audit variant must contain a pooled policy")
    sampled = np.ascontiguousarray(frames[indices])
    for name, policy in policies.items():
        if not isinstance(policy, dict) or not policy.get("available"):
            continue
        entries = policy.get("frames")
        if not isinstance(entries, list) or len(entries) != len(sampled):
            raise ValueError(f"policy {name!r} frame count does not match cached samples")
        for i, item in enumerate(entries):
            corr = item.get("correction") if isinstance(item, dict) else None
            if corr is None:
                raise ValueError(f"policy {name!r} has no correction for sampled frame {i}")
            matrix = np.asarray(corr, dtype=np.float64)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                raise ValueError(f"policy {name!r} has an invalid correction at sampled frame {i}")
    if not meta.get("source_sha256") or not meta.get("window") or not meta.get("source_path"):
        raise ValueError("frame cache metadata must include source hash, path and window")
    return frames, meta, document, report, indices, sampled, actual_hash


def _render_policy(frames, entries, pad, downscale, path, mask_path, fps):
    h, w = frames.shape[1:3]
    video = media.Encoder(path, w, h, str(fps), crf=15, preset="fast")
    mask = media.Encoder(mask_path, w, h, str(fps), gray=True)
    valid_fracs = []
    try:
        for frame, item in zip(frames, entries):
            corr = R._correction_to_pixel_matrix(item["correction"], pad, downscale)
            rendered, valid = pull_affine(frame, corr)
            video.write(rendered)
            mask.write(np.where(valid, 0, 255).astype(np.uint8))
            valid_fracs.append(float(valid.mean()))
        media.close_all((video, mask))
    except BaseException:
        try:
            media.close_all((video, mask))
        except BaseException:
            pass
        raise
    return {"frames": len(frames), "mean_valid_fraction": float(np.mean(valid_fracs)),
            "min_frame_valid_fraction": float(np.min(valid_fracs)),
            "video_sha256": MF.sha256_file(path), "mask_sha256": MF.sha256_file(mask_path)}


def _render_manifest(meta, source_sha, video_path, fps, count, width, height):
    pts = media.frame_pts(video_path)
    return {
        "schema": "mlxstab.depth_layer_real_render_manifest/1", "status": "success",
        "run_id": "depth_layer_real_render", "case": {"name": "pooled_reference"},
        "source": {"path": meta["source_path"], "path_type": "file", "sha256": source_sha,
                   "window": meta["window"], "decoded": {"width": meta["width"],
                   "height": meta["height"], "fps": meta["fps"], "frames": meta["count"]}},
        "outputs": {"video": str(Path(video_path).resolve()),
                    "decoded": {"width": width, "height": height, "fps": float(fps), "frames": count},
                    "encoded_frames": count,
                    "cadence": {"measured_fps": float(fps), "measured_frames": count},
                    "encoded_pts": pts,
                    "timeline_alignment": {"aligned": True}},
    }


def build(frames_npz, report_json, out_dir, *, variant=None, panel_width=640, panel_height=360):
    """Render pooled/rank outputs and create counterbalanced baseline-vs-rank reviews."""
    frames_npz, report_json, out_dir = Path(frames_npz), Path(report_json), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    full_frames, meta, document, report, indices, sampled, frames_hash = _load_inputs(
        frames_npz, report_json, variant)
    steps = np.diff(indices)
    if not np.all(steps == steps[0]):
        raise ValueError("sampled frame cadence must be uniform for a review video")
    source_fps = Fraction(str(meta["fps"])).limit_denominator(1_000_000)
    effective_fps = source_fps / int(steps[0])
    policies = report["policies"]
    flow = document.get("flow") or {}
    pad, downscale = flow.get("pad"), flow.get("downscale")
    if pad is None or downscale is None:
        raise ValueError("depth audit must record feature-grid padding and downscale")

    baseline_video = out_dir / "pooled.mp4"
    baseline_mask = out_dir / "pooled_mask.mp4"
    pooled = _render_policy(sampled, policies["pooled"]["frames"], pad, downscale,
                            baseline_video, baseline_mask, effective_fps)
    h, w = sampled.shape[1:3]
    manifest_path = out_dir / "render_manifest.json"
    manifest = _render_manifest(meta, meta["source_sha256"], baseline_video,
                                effective_fps, len(sampled), w, h)
    MF.write_manifest(manifest_path, manifest)

    rank_outputs = {}
    for name, policy in policies.items():
        if name == "pooled" or not policy.get("available"):
            continue
        candidate_video = out_dir / f"{name}.mp4"
        candidate_mask = out_dir / f"{name}_mask.mp4"
        candidate = _render_policy(sampled, policy["frames"], pad, downscale,
                                   candidate_video, candidate_mask, effective_fps)
        review_dir = out_dir / f"review_{name}"
        review_dir.mkdir()
        review = build_review(
            manifest_path, candidate_video=candidate_video,
            out_video=review_dir / "baseline_vs_rank.mp4",
            out_reversed=review_dir / "baseline_vs_rank_posrev.mp4",
            out_report=review_dir / "review.json", panel_width=panel_width,
            panel_height=panel_height)
        rank_outputs[name] = {
            **candidate, "review_json": str((review_dir / "review.json").relative_to(out_dir)),
            "review_clips": review["output"]["clips"],
            "blind_group": review["position_reversal"]["group"],
        }
    result = {
        "schema": SCHEMA, "ok": True,
        "scope": "Fixed-path real-pixel diagnostic; no quality winner or production policy selected.",
        "source_sha256": meta["source_sha256"], "frames_npz_sha256": frames_hash,
        "depth_report_sha256": MF.sha256_file(report_json),
        "source_window": meta["window"], "sampled_frame_indices": indices.tolist(),
        "source_fps": float(source_fps), "effective_review_fps": [effective_fps.numerator,
                                                                      effective_fps.denominator],
        "sampled_frame_count": len(sampled), "dimensions": [w, h],
        "depth_report_schema": document.get("schema"), "variant": variant,
        "depth_model": document.get("depth_model"), "flow": flow,
        "policies": {"pooled": pooled, **rank_outputs},
        "limitations": LIMITATIONS,
    }
    MF.write_manifest(out_dir / "render_review.json", result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-npz", required=True)
    ap.add_argument("--depth-audit-json", required=True)
    ap.add_argument("--variant", help="required only when the audit JSON contains a sweep")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--panel-width", type=int, default=640)
    ap.add_argument("--panel-height", type=int, default=360)
    args = ap.parse_args(argv)
    result = build(args.frames_npz, args.depth_audit_json, args.out_dir, variant=args.variant,
                   panel_width=args.panel_width, panel_height=args.panel_height)
    print(json.dumps({"out_dir": str(args.out_dir), "sampled_frame_count": result["sampled_frame_count"],
                      "effective_review_fps": result["effective_review_fps"],
                      "policies": {name: {"mean_valid_fraction": item["mean_valid_fraction"],
                                           "review_json": item.get("review_json")}
                                   for name, item in result["policies"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
