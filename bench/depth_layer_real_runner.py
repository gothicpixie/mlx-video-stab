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
from .depth_layer_real_audit import (
    audit_depth_layer_sweep,
    audit_depth_layer_window,
    sweep_variant_key,
    validate_sweep_options,
)

FEATURE_STRIDE = 8
CONTACT_SHEET_NOTE = (
    "geometry-only preview: affine warps only; no depth-aware occlusion handling or image-quality score"
)


def _feature_to_pixel_matrix(pad, downscale):
    """Map original-image pixel coordinates to RAFT feature-grid coordinates (pixel -> feature).

    Mirrors :func:`bench.depth_temporal_audit.depth_to_feature_grid`: feature cell ``f`` samples
    original pixel ``(8*downscale) * f + (downscale - 1) / 2 - downscale * pad``, so the inverse
    pixel -> feature map is a scale ``8 * downscale`` plus a pad/downscale offset.
    """
    d = int(downscale)
    scale = float(FEATURE_STRIDE * d)
    pad = np.asarray(pad, np.float64).reshape(-1)
    if pad.size != 4:
        raise ValueError("pad must be (top, bottom, left, right)")
    offset_x = (d * pad[2] - (d - 1) / 2.0) / scale
    offset_y = (d * pad[0] - (d - 1) / 2.0) / scale
    return np.array([[1.0 / scale, 0.0, offset_x],
                     [0.0, 1.0 / scale, offset_y],
                     [0.0, 0.0, 1.0]], np.float64)


def _correction_to_pixel_matrix(correction, pad, downscale):
    """Conjugate a feature-grid source->stabilized correction into original-pixel coordinates."""
    feature_from_pixel = _feature_to_pixel_matrix(pad, downscale)
    pixel_from_feature = np.linalg.inv(feature_from_pixel)
    return pixel_from_feature @ np.asarray(correction, np.float64) @ feature_from_pixel


def _write_contact_sheet_preview(frames, report, pad, downscale, out_path, *, max_frames=6,
                                 cell_width=240):
    """Render an optional geometry-only contact-sheet preview and return the written path.

    Columns are the untouched original frames plus the pooled and every relative-depth-rank
    policy; up to ``max_frames`` frames are sampled evenly down each column. Feature-grid
    corrections are conjugated into original-pixel coordinates and inverted because Pillow's
    affine transform maps output coordinates back to source coordinates. Pillow is imported
    lazily so the audit core and the runner import stay NumPy-only. The input frames are only
    read; nothing here is written back into the JSON report.
    """
    from PIL import Image, ImageDraw, ImageFont

    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("frames must be uint8 [N,H,W,3] RGB")
    total = int(frames.shape[0])
    if total == 0:
        raise ValueError("frames must contain at least one frame")
    policies = report.get("policies") if isinstance(report, dict) else None
    if not isinstance(policies, dict):
        raise ValueError("report must contain per-window policy summaries")
    n_bins = int(report.get("n_bins_requested", 0))
    names = ["original", "pooled"] + [f"relative_depth_rank_{i}" for i in range(n_bins)]
    count = min(int(max_frames), total)
    sample = sorted({int(round(v)) for v in np.linspace(0, total - 1, count)})

    def corrections_for(name):
        policy = policies.get(name)
        if not isinstance(policy, dict) or not policy.get("available"):
            return [None] * total
        entries = policy.get("frames") or []
        out = []
        for index in range(total):
            entry = entries[index] if index < len(entries) else None
            corr = entry.get("correction") if isinstance(entry, dict) else None
            if corr is None:
                out.append(None)
                continue
            matrix = np.asarray(corr, np.float64)
            out.append(matrix if matrix.shape == (3, 3) and np.isfinite(matrix).all() else None)
        return out

    corrections = {name: corrections_for(name) for name in names}
    h, w = int(frames.shape[1]), int(frames.shape[2])
    cell_h = max(1, int(round(cell_width * h / w)))
    label_h = 18
    note_h = 20
    margin = 6
    label_font = ImageFont.load_default()
    sheet_w = margin + len(names) * (cell_width + margin)
    sheet_h = margin + label_h + len(sample) * (cell_h + margin) + note_h + margin
    sheet = Image.new("RGB", (sheet_w, sheet_h), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)

    for col, name in enumerate(names):
        x0 = margin + col * (cell_width + margin)
        draw.text((x0 + 2, margin), name, fill=(235, 235, 235), font=label_font)
        draw.line((x0, margin + label_h - 4, x0 + cell_width, margin + label_h - 4),
                  fill=(90, 90, 90))

    for row, frame_index in enumerate(sample):
        y0 = margin + label_h + row * (cell_h + margin)
        source = np.ascontiguousarray(frames[frame_index])
        for col, name in enumerate(names):
            x0 = margin + col * (cell_width + margin)
            if name == "original":
                tile = Image.fromarray(source)
            else:
                corr = corrections[name][frame_index]
                if corr is None:
                    draw.rectangle((x0, y0, x0 + cell_width, y0 + cell_h), fill=(0, 0, 0),
                                   outline=(120, 40, 40))
                    draw.text((x0 + 2, y0 + 2), "n/a", fill=(230, 130, 130), font=label_font)
                    continue
                try:
                    output_to_input = np.linalg.inv(
                        _correction_to_pixel_matrix(corr, pad, downscale))
                    data = (float(output_to_input[0, 0]), float(output_to_input[0, 1]),
                            float(output_to_input[0, 2]), float(output_to_input[1, 0]),
                            float(output_to_input[1, 1]), float(output_to_input[1, 2]))
                    tile = Image.fromarray(source).transform(
                        (w, h), Image.Transform.AFFINE, data,
                        resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
                except np.linalg.LinAlgError:
                    tile = Image.new("RGB", (w, h), (0, 0, 0))
            tile = tile.resize((cell_width, cell_h), Image.Resampling.LANCZOS)
            sheet.paste(tile, (x0, y0))
            draw.rectangle((x0, y0, x0 + cell_width, y0 + cell_h), outline=(70, 70, 70))
            draw.text((x0 + 2, y0 + 2), f"#{frame_index}", fill=(255, 230, 120), font=label_font)

    note_y = margin + label_h + len(sample) * (cell_h + margin) + 2
    draw.text((margin, note_y), CONTACT_SHEET_NOTE, fill=(200, 200, 200), font=label_font)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def run_cached_audit(frames_npz, depth_npz, weights, out_path=None, *, iters=12, downscale=1,
                     n_bins=4, smooth_sigma=1.0, sweep_bins=None, sweep_sigmas=None,
                     preview_out=None, preview_bins=4, preview_sigma=1.0,
                     flow_dtype="float32", corr_dtype="float32",
                     lock_enabled=True, lock_chunk=0, lock_pause=0.0, lock_factory=None,
                     runner_factory=None, write=True, log=print):
    """Run the matched downstream geometry audit on exactly one paired cache window.

    When ``sweep_bins`` or ``sweep_sigmas`` is supplied the RAFT pass still runs exactly
    once; the collected observations are then audited independently for every requested
    combination by :func:`bench.depth_layer_real_audit.audit_depth_layer_sweep`. A missing
    grid axis falls back to the single-run ``n_bins``/``smooth_sigma`` value. When both
    sweep arguments are omitted the existing single-run report is returned unchanged.

    ``preview_out`` optionally renders a geometry-only contact sheet for the pooled and
    relative-depth-rank policies. In sweep mode ``preview_bins``/``preview_sigma`` must
    resolve to an exact requested variant (validated before RAFT runs); in single mode they
    must match ``n_bins``/``smooth_sigma``. The preview path is never written into the JSON.
    """
    if int(iters) < 1 or int(downscale) < 1 or int(n_bins) < 1:
        raise ValueError("iters, downscale and n_bins must be >= 1")
    if not np.isfinite(float(smooth_sigma)) or float(smooth_sigma) < 0:
        raise ValueError("smooth_sigma must be finite and >= 0")
    sweep = sweep_bins is not None or sweep_sigmas is not None
    if sweep:
        bins = tuple(sweep_bins) if sweep_bins is not None else (int(n_bins),)
        sigmas = tuple(sweep_sigmas) if sweep_sigmas is not None else (float(smooth_sigma),)
        bins, sigmas = validate_sweep_options(bins, sigmas)
    preview = preview_out is not None
    preview_key = None
    if preview:
        preview_bins = int(preview_bins)
        preview_sigma = float(preview_sigma)
        if preview_bins < 1 or not np.isfinite(preview_sigma) or preview_sigma < 0:
            raise ValueError("preview_bins must be >= 1 and preview_sigma finite and >= 0")
        if sweep:
            preview_key = sweep_variant_key(preview_bins, preview_sigma)
            requested = {sweep_variant_key(b, s) for b in bins for s in sigmas}
            if preview_key not in requested:
                raise ValueError(
                    f"preview variant {preview_key!r} is not in the requested sweep grid")
        elif int(preview_bins) != int(n_bins) or float(preview_sigma) != float(smooth_sigma):
            raise ValueError(
                "preview_bins/preview_sigma must match n_bins/smooth_sigma in single-run mode")
    if int(lock_chunk) < 0 or float(lock_pause) < 0:
        raise ValueError("lock_chunk and lock_pause must be >= 0")
    frames_npz, depth_npz, weights = Path(frames_npz), Path(depth_npz), Path(weights)
    if preview:
        preview_path = Path(preview_out).expanduser().resolve()
        protected = {frames_npz.expanduser().resolve(), depth_npz.expanduser().resolve(),
                     weights.expanduser().resolve()}
        if out_path is not None:
            protected.add(Path(out_path).expanduser().resolve())
        if preview_path in protected:
            raise ValueError("preview_out must not overwrite a frame/depth/weight input or the JSON report")
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

    if sweep:
        report = audit_depth_layer_sweep(observations, bin_counts=bins, smooth_sigmas=sigmas)
        for variant in report["variants"].values():
            for item, provenance in zip(variant["pairs"], pair_meta):
                item.update(provenance)
    else:
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
    if preview:
        preview_report = report["variants"][preview_key] if sweep else report
        _write_contact_sheet_preview(sampled_frames, preview_report, runner.pad,
                                     int(downscale), preview_out)
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
    parser.add_argument("--sweep-bins", default=None,
                        help="comma-separated positive bin counts enabling the audit sweep")
    parser.add_argument("--sweep-sigmas", default=None,
                        help="comma-separated finite nonnegative sigmas enabling the audit sweep")
    parser.add_argument("--preview-out", default=None,
                        help="optional path for a geometry-only contact-sheet preview")
    parser.add_argument("--preview-bins", type=int, default=4,
                        help="rank bin count of the previewed policy (sweep variant / single run)")
    parser.add_argument("--preview-sigma", type=float, default=1.0,
                        help="smoothing sigma of the previewed policy (sweep variant / single run)")
    parser.add_argument("--flow-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--corr-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--lock", choices=("on", "off"), default="on")
    parser.add_argument("--lock-chunk", type=int, default=0)
    parser.add_argument("--lock-pause", type=float, default=0.0)
    return parser


def _parse_sweep_values(text, cast):
    """Parse a comma-separated CLI grid axis into a tuple, or ``None`` when the flag is absent."""
    if text is None:
        return None
    return tuple(cast(part.strip()) for part in text.split(",") if part.strip())


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run_cached_audit(args.frames_npz, args.depth_npz, args.weights, args.out,
                              iters=args.iters, downscale=args.downscale, n_bins=args.bins,
                              smooth_sigma=args.smooth_sigma,
                              sweep_bins=_parse_sweep_values(args.sweep_bins, int),
                              sweep_sigmas=_parse_sweep_values(args.sweep_sigmas, float),
                              preview_out=args.preview_out, preview_bins=args.preview_bins,
                              preview_sigma=args.preview_sigma,
                              flow_dtype=args.flow_dtype,
                              corr_dtype=args.corr_dtype, lock_enabled=(args.lock == "on"),
                              lock_chunk=args.lock_chunk, lock_pause=args.lock_pause)
    print(MF.dumps_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
