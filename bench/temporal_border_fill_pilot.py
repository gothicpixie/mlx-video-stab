"""Research-only temporal border-fill pilot for a stabilized clip's no-source pixels.

A successful ``bench/run_case.py`` run renders a video plus a co-timed gray **border mask** whose white
(``255``) pixels are "no source": the warp mapped them outside the frame, so ``stabilize`` zeroed them. This
pilot asks one narrow, purely visual question -- *what would it look like if those border pixels copied their
color from the nearest frame in time where the same pixel has a real source?* -- and writes one candidate MP4
plus one adjacent JSON report for later review. Nothing here is wired into the pipeline, changes a default or
selects a method, and no network call is made.

This is a **diagnostic heuristic, not a production default.** Copying a pixel forward or backward in time can
**ghost moving content**: any object that moves across the border region within ``radius`` frames may leave a
stale duplicate behind, and the fill can not distinguish a static background gap from a moving edge. The
report records only counts and fractions; it makes no quality claim and no threshold is selected.

Policy (explicit and deterministic)
-----------------------------------
For output frame ``t`` and each pixel marked invalid (mask true = no source) in that frame:

- search temporal distance ``d = 1..radius``;
- at the first ``d`` where at least one of the frames ``t-d`` / ``t+d`` has that pixel **valid**, copy it;
- if both are valid at that same distance, write the integer mean of the two colors;
- donors are read from the **original neighbor frames** and their **original masks only**, so a filled pixel
  is never used as a donor and no fill is propagated;
- pixels with no valid donor within ``radius`` are left **unchanged**, and pixels already valid in frame
  ``t`` are copied through untouched.

Memory is bounded by the search radius: frames are streamed through a sliding window that holds at most
``2*radius + 1`` decoded frames (never the whole clip), and only masks/counts are aggregated per frame.

Offline and safe by construction
--------------------------------
Inputs are validated up front: the manifest must be a successful ``run_case`` manifest, the output video and
mask must exist, match each other in dimensions/frame rate/frame count, follow their zero-origin CFR clock,
agree with the manifest's decoded metadata and recorded PTS counts, and the manifest's
``outputs.timeline_alignment.aligned`` must be true. ``--radius`` must be a positive integer no larger than
``MAX_RADIUS``. The candidate is encoded at the manifest's exact width/height/fps/frame count to a temporary
file, is refused if it would exceed Gemini's inline size cap (the clip is meant for the later local review
tooling), and is never allowed to overwrite an existing artifact. The JSON report is written atomically
beside the MP4 and carries the source manifest/video/mask hashes, the per-frame invalid/filled/unresolved
counts and fractions, the output hash and the recomputed output CFR verification.

Example
-------
::

    .venv/bin/python -m bench.temporal_border_fill_pilot \
        out/bench/RUN/manifest.json --radius 3 \
        --out-video out/bench/RUN/temporal_fill_r3.mp4
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from mlxstab import media

from . import gemini_blind_review as Gemini
from . import manifest as MF

SCHEMA = "mlxstab.temporal_border_fill_pilot/1"
DEFAULT_RADIUS = 3
#: Search-radius ceiling: keeps the sliding window bounded (``2*radius + 1`` frames).
MAX_RADIUS = 30
DEFAULT_CRF = 18
DEFAULT_PRESET = "medium"

FILL_POLICY = (
    "nearest_valid_temporal_neighbor(distance=1..radius); equidistant_previous_and_next_averaged; "
    "donors_from_original_neighbor_frames_and_masks_only; unresolved_left_unchanged"
)

LIMITATIONS = (
    "diagnostic heuristic only: copying a border pixel from another time can ghost moving content, and the "
    "fill cannot tell a static background gap from a moving foreground edge; not a production default",
    "the optional bilateral RGB-consistency gate is not motion alignment: a moving object with similar colors "
    "can still pass, while exposure changes or sensor noise can reject a static background donor",
    "one candidate clip per run; no stabilization, motion model, mask or default is changed",
    "counts and fractions only: no quality score, no winner and no threshold is selected",
    "the offline report is intended for a later local visual review; this tool makes no network call",
)


class TemporalFillError(ValueError):
    """The manifest, the source streams, the radius or the candidate evidence is not usable."""


# --------------------------------------------------------------------------------------
# pure fill core (unit-testable, no I/O)
# --------------------------------------------------------------------------------------

def fill_center(frames, masks, center, radius, max_bilateral_delta=None):
    """Fill one frame's invalid pixels from its temporal neighbors.

    ``frames`` is a sequence of original ``[H, W, 3]`` uint8 frames and ``masks`` a matching sequence of
    ``[H, W]`` boolean masks (``True`` = no source). ``center`` indexes the frame to fill and ``radius`` is
    the maximum temporal search distance. A pixel is filled from the nearest distance where a neighbor is
    valid; an equidistant valid previous **and** next pair is averaged (truncated integer mean); pixels with
    no valid donor are left unchanged and valid pixels are never modified. Returns
    ``(out_uint8, invalid_count, filled_count, unresolved_count)``.
    """
    radius = int(radius)
    if radius < 1:
        raise TemporalFillError("radius must be a positive integer")
    max_bilateral_delta = _validate_max_bilateral_delta(max_bilateral_delta)
    frames = [np.asarray(f) for f in frames]
    masks = [np.asarray(m).astype(bool) for m in masks]
    n = len(frames)
    if n == 0 or len(masks) != n:
        raise TemporalFillError("frames and masks must be non-empty and the same length")
    if not (0 <= int(center) < n):
        raise TemporalFillError("center is outside the supplied window")
    center = int(center)
    shape = frames[center].shape
    if frames[center].dtype != np.uint8 or len(shape) != 3 or shape[2] != 3:
        raise TemporalFillError("frames must be uint8 [H, W, 3]")
    for f, m in zip(frames, masks):
        if f.shape != shape or m.shape != shape[:2]:
            raise TemporalFillError("window frames/masks must share one shape")

    out = frames[center].copy()
    remaining = masks[center].copy()
    filled_any = np.zeros_like(remaining)
    blocked = np.zeros_like(remaining)
    for d in range(1, radius + 1):
        active = remaining & ~blocked
        if not active.any():
            break
        p, nx = center - d, center + d
        prev_ok = ~masks[p] if p >= 0 else np.zeros_like(remaining)
        next_ok = ~masks[nx] if nx < n else np.zeros_like(remaining)
        both = active & prev_ok & next_ok
        only_prev = active & prev_ok & ~next_ok
        only_next = active & next_ok & ~prev_ok
        if both.any():
            prev = frames[p][both].astype(np.uint16)
            nxt = frames[nx][both].astype(np.uint16)
            if max_bilateral_delta is None:
                out[both] = ((prev + nxt) // 2).astype(np.uint8)
                accepted = both
            else:
                agree = np.max(np.abs(prev.astype(np.int16) - nxt.astype(np.int16)), axis=1) <= max_bilateral_delta
                both_indices = np.flatnonzero(both)
                accepted = np.zeros_like(remaining)
                accepted.flat[both_indices[agree]] = True
                rejected = np.zeros_like(remaining)
                rejected.flat[both_indices[~agree]] = True
                if accepted.any():
                    out[accepted] = ((prev[agree] + nxt[agree]) // 2).astype(np.uint8)
                blocked |= rejected
        else:
            accepted = np.zeros_like(remaining)
        if only_prev.any() and max_bilateral_delta is None:
            out[only_prev] = frames[p][only_prev]
        if only_next.any() and max_bilateral_delta is None:
            out[only_next] = frames[nx][only_next]
        newly = accepted | (only_prev | only_next if max_bilateral_delta is None else False)
        filled_any |= newly
        remaining &= ~newly

    invalid = int(masks[center].sum())
    filled = int(filled_any.sum())
    unresolved = int(remaining.sum())
    return out, invalid, filled, unresolved


def _as_mask_bool(mask):
    """Normalize a decoded mask frame (uint8 gray/RGB) or a bool array to a ``[H, W]`` bool mask."""
    mask = np.asarray(mask)
    if mask.dtype == np.bool_:
        return mask
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 127


def stream_fill(pairs, count, radius, on_frame, window_probe=None, max_bilateral_delta=None):
    """Fill ``count`` frames from a ``(frame, mask)`` iterator with a bounded sliding window.

    ``pairs`` yields decoded ``(frame, mask)`` pairs. At most ``2*radius + 1`` pairs are ever buffered: frame
    ``t`` needs only ``t-radius .. t+radius``. ``on_frame(out_frame, record)`` is called once per frame in
    presentation order. ``window_probe(len(buf))``, if given, observes the live window size before each fill
    (used to prove the memory bound). Returns ``(records, totals)``. Raises :class:`TemporalFillError` if the
    decoded frame count differs from ``count`` or a frame/mask changes shape mid-stream.
    """
    count = int(count)
    radius = int(radius)
    pairs = iter(pairs)
    buf = deque()
    next_index = 0
    eof = False
    records = []
    totals = {"invalid": 0, "filled": 0, "unresolved": 0}
    expected_shape = None

    for t in range(count):
        # Drop frames older than the previous donor before pulling more, so the live window stays bounded.
        while buf and buf[0][0] < t - radius:
            buf.popleft()
        while next_index <= t + radius:
            item = next(pairs, None)
            if item is None:
                eof = True
                break
            frame, mask = item
            frame = np.asarray(frame)
            mask_bool = _as_mask_bool(mask)
            if expected_shape is None:
                expected_shape = (frame.shape, mask_bool.shape)
                if len(frame.shape) != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                    raise TemporalFillError("decoded video frames must be uint8 [H, W, 3]")
                if mask_bool.shape != frame.shape[:2]:
                    raise TemporalFillError("decoded mask shape disagrees with the video frame")
            elif (frame.shape, mask_bool.shape) != expected_shape:
                raise TemporalFillError("frame dimensions changed mid-stream")
            buf.append((next_index, frame, mask_bool))
            next_index += 1
        if not buf or buf[-1][0] < t:
            raise TemporalFillError(f"decoded frame count is short of the manifest frame count ({count})")
        if window_probe is not None:
            window_probe(len(buf))

        frames = [entry[1] for entry in buf]
        masks = [entry[2] for entry in buf]
        out, invalid, filled, unresolved = fill_center(
            frames, masks, t - buf[0][0], radius, max_bilateral_delta=max_bilateral_delta)
        record = {
            "index": int(t),
            "invalid": invalid,
            "filled": filled,
            "unresolved": unresolved,
            "fill_fraction": (filled / invalid) if invalid else 0.0,
            "invalid_fraction": invalid / float(out.shape[0] * out.shape[1]),
            "filled_fraction": filled / float(out.shape[0] * out.shape[1]),
        }
        on_frame(out, record)
        records.append(record)
        totals["invalid"] += invalid
        totals["filled"] += filled
        totals["unresolved"] += unresolved

    extra = 0
    if not eof:
        extra = sum(1 for _ in pairs)
    total_decoded = next_index + extra
    if total_decoded != count:
        raise TemporalFillError(
            f"decoded frame count {total_decoded} differs from the manifest frame count {count}")
    return records, totals


# --------------------------------------------------------------------------------------
# evidence validation (no decode of private footage beyond the already-encoded outputs)
# --------------------------------------------------------------------------------------

def _finite_rate(value, where):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or value <= 0:
        raise TemporalFillError(f"{where} must be a positive finite frame rate")
    return float(value)


def load_and_validate_manifest(manifest_path):
    """Load a successful ``run_case`` manifest and resolve its retained output video/mask paths."""
    path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TemporalFillError(f"cannot read run_case manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise TemporalFillError("manifest must be a JSON object")
    if manifest.get("schema") != MF.MANIFEST_SCHEMA:
        raise TemporalFillError(f"manifest schema must be {MF.MANIFEST_SCHEMA!r}")
    if manifest.get("status") != "success":
        raise TemporalFillError("manifest status must be success")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise TemporalFillError("manifest outputs are missing")
    video, mask = outputs.get("video"), outputs.get("mask")
    if not video or not mask:
        raise TemporalFillError("manifest must record an output video and mask")
    video_path, mask_path = Path(video), Path(mask)
    if not video_path.is_file():
        raise TemporalFillError("manifest output video is missing on disk")
    if not mask_path.is_file():
        raise TemporalFillError("manifest output mask is missing on disk")
    if video_path.resolve() == mask_path.resolve():
        raise TemporalFillError("manifest output video and mask must differ")
    if path in (video_path.resolve(), mask_path.resolve()):
        raise TemporalFillError("manifest path collides with an output stream")
    if outputs.get("timeline_alignment", {}).get("aligned") is not True:
        raise TemporalFillError("manifest output timeline alignment failed")
    return manifest, path, video_path, mask_path


def validate_evidence(manifest, video_path, mask_path):
    """Validate the encoded video/mask pair and return the resolved cadence metadata.

    Checks the two files agree in dimensions and frame rate, both follow their zero-origin constant-frame-rate
    clock with the manifest's encoded frame count, the manifest's own recorded PTS counts agree, and the video
    probe agrees with the manifest's decoded metadata.
    """
    outputs = manifest["outputs"]
    decoded = outputs.get("decoded")
    if not isinstance(decoded, dict):
        raise TemporalFillError("manifest decoded output metadata is missing")
    count = outputs.get("encoded_frames")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1 or decoded.get("frames") != count:
        raise TemporalFillError("manifest decoded/encoded frame counts are missing or inconsistent")

    video_info = media.probe(video_path)
    mask_info = media.probe(mask_path)
    dims = (video_info.get("width"), video_info.get("height"))
    if dims != (mask_info.get("width"), mask_info.get("height")):
        raise TemporalFillError("output video/mask dimensions differ")
    if None in dims:
        raise TemporalFillError("output dimensions are missing")
    if (decoded.get("width"), decoded.get("height")) != dims:
        raise TemporalFillError("output probe dimensions disagree with the manifest decoded metadata")
    video_fps = _finite_rate(video_info.get("fps"), "output video fps")
    mask_fps = _finite_rate(mask_info.get("fps"), "output mask fps")
    if abs(video_fps - mask_fps) > 1e-6:
        raise TemporalFillError("output video/mask frame rates differ")
    if abs(video_fps - _finite_rate(decoded.get("fps"), "manifest decoded fps")) > 1e-6:
        raise TemporalFillError("output frame rate disagrees with the manifest decoded metadata")

    video_pts = media.frame_pts(video_path)
    mask_pts = media.frame_pts(mask_path)
    cfr = MF.check_encoded_cfr(video_pts, mask_pts, video_fps, mask_fps)
    if not cfr["aligned"]:
        raise TemporalFillError("output video/mask CFR verification failed: " + "; ".join(cfr["reasons"]))
    if video_pts.get("count") != count or mask_pts.get("count") != count:
        raise TemporalFillError("output video/mask frame count disagrees with the manifest")
    for key in ("encoded_pts", "mask_pts"):
        recorded = outputs.get(key)
        if isinstance(recorded, dict) and recorded.get("count") is not None:
            if int(recorded["count"]) != count:
                raise TemporalFillError(f"manifest {key} count disagrees with encoded_frames")
    return {"count": int(count), "fps": video_fps, "width": int(dims[0]), "height": int(dims[1]),
            "cfr": cfr, "video_pts": video_pts, "mask_pts": mask_pts}


def verify_video_cfr(pts, count, fps, tol_s=1e-5):
    """Verify a single encoded stream follows the zero-origin ``i/fps`` CFR clock."""
    reasons = []
    if not isinstance(pts, dict) or pts.get("count") != count or pts.get("n_pts") != count:
        reasons.append("frame/PTS count does not match the expected frame count")
    else:
        values = pts.get("pts_s")
        if not pts.get("monotonic") or not isinstance(values, list) or len(values) != count:
            reasons.append("PTS are missing or non-monotonic")
        elif any(v is None or not np.isfinite(v) or abs(float(v) - i / float(fps)) > tol_s
                 for i, v in enumerate(values)):
            reasons.append("PTS do not follow the zero-origin CFR clock")
    return {"aligned": not reasons, "reasons": reasons, "count": int(count), "fps": float(fps),
            "tolerance_s": tol_s}


# --------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------

def _validate_radius(radius):
    if isinstance(radius, bool) or not isinstance(radius, int):
        raise TemporalFillError("radius must be a positive integer")
    if radius < 1:
        raise TemporalFillError("radius must be a positive integer")
    if radius > MAX_RADIUS:
        raise TemporalFillError(f"radius {radius} exceeds the bounded-window maximum {MAX_RADIUS}")
    return int(radius)


def _validate_max_bilateral_delta(value):
    """Validate an optional per-channel RGB agreement threshold in [0, 255]."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TemporalFillError("max_bilateral_delta must be finite and in [0, 255]")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise TemporalFillError("max_bilateral_delta must be finite and in [0, 255]") from exc
    if not np.isfinite(threshold) or not (0.0 <= threshold <= 255.0):
        raise TemporalFillError("max_bilateral_delta must be finite and in [0, 255]")
    return threshold


def _resolve_outputs(manifest_path, out_video, out_json, radius):
    run_dir = manifest_path.parent
    video_path = Path(out_video).resolve() if out_video else (run_dir / f"temporal_fill_r{radius}.mp4").resolve()
    json_path = Path(out_json).resolve() if out_json else video_path.with_suffix(".json")
    if video_path.suffix.lower() != ".mp4":
        raise TemporalFillError("candidate video must use the .mp4 suffix")
    if json_path.suffix.lower() != ".json":
        raise TemporalFillError("report must use the .json suffix")
    if video_path.parent != json_path.parent:
        raise TemporalFillError("candidate MP4 and report must be written in the same directory")
    return video_path, json_path


def build(manifest_path, *, out_video=None, out_json=None, radius=DEFAULT_RADIUS,
          crf=DEFAULT_CRF, preset=DEFAULT_PRESET, max_bilateral_delta=None, log=print) -> dict:
    """Validate the run, fill the border pixels and write the candidate MP4 + adjacent atomic JSON."""
    radius = _validate_radius(radius)
    max_bilateral_delta = _validate_max_bilateral_delta(max_bilateral_delta)
    manifest, manifest_path, video_path, mask_path = load_and_validate_manifest(manifest_path)
    evidence = validate_evidence(manifest, video_path, mask_path)
    out_video, out_json = _resolve_outputs(manifest_path, out_video, out_json, radius)
    for path in (out_video, out_json):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    for path in (out_video, out_json):
        if path in (manifest_path, video_path.resolve(), mask_path.resolve()):
            raise TemporalFillError("candidate output collides with an input artifact")
    if out_video == out_json:
        raise TemporalFillError("candidate MP4 and report paths must differ")
    out_video.parent.mkdir(parents=True, exist_ok=True)
    log(f"[temporal-border-fill] radius={radius} {evidence['width']}x{evidence['height']} "
        f"@{evidence['fps']:g} fps, {evidence['count']} frames")

    video_iter = media.decode(video_path, None, None, None)[0]
    mask_iter = media.decode(mask_path, None, None, None)[0]
    pairs = zip(video_iter, mask_iter)

    tmp_path = None
    encoder = None
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{out_video.stem}.", suffix=".mp4", dir=out_video.parent)
        os.close(fd)
        Path(tmp_name).unlink()
        tmp_path = Path(tmp_name)
        encoder = media.Encoder(tmp_path, evidence["width"], evidence["height"], evidence["fps"],
                                crf=crf, preset=preset)

        def on_frame(frame, _record):
            encoder.write(frame)

        records, totals = stream_fill(pairs, evidence["count"], radius, on_frame,
                                      max_bilateral_delta=max_bilateral_delta)
        encoder.close()
        encoder = None

        out_pts = media.frame_pts(tmp_path)
        out_cfr = verify_video_cfr(out_pts, evidence["count"], evidence["fps"])
        if not out_cfr["aligned"]:
            raise TemporalFillError("candidate MP4 CFR verification failed: " + "; ".join(out_cfr["reasons"]))
        size_bytes = tmp_path.stat().st_size
        if size_bytes > Gemini.MAX_INLINE_BYTES:
            raise TemporalFillError(
                f"candidate clip is {size_bytes} bytes, over Gemini's {Gemini.MAX_INLINE_BYTES}-byte inline "
                "cap; this clip could not be submitted for review")
        out_sha = MF.sha256_file(tmp_path)
        os.replace(tmp_path, out_video)
        tmp_path = None

        total_pixels = evidence["width"] * evidence["height"] * evidence["count"]
        report = {
            "status": "success",
            "schema": SCHEMA,
            "limitations": list(LIMITATIONS),
            "created": MF.now_iso(),
            "updated": MF.now_iso(),
            "radius": radius,
            "max_bilateral_delta": max_bilateral_delta,
            "policy": (FILL_POLICY if max_bilateral_delta is None else
                       FILL_POLICY + "; bilateral_only; nearest bilateral donor rejected when max per-channel "
                       f"RGB difference exceeds {max_bilateral_delta:g}; no farther fallback after rejection"),
            "max_window_frames": 2 * radius + 1,
            "source": {
                "manifest": {"path": str(manifest_path), "sha256": MF.sha256_file(manifest_path)},
                "video": {"path": str(video_path), "sha256": MF.sha256_file(video_path)},
                "mask": {"path": str(mask_path), "sha256": MF.sha256_file(mask_path)},
                "run_id": manifest.get("run_id"),
                "width": evidence["width"], "height": evidence["height"],
                "fps": evidence["fps"], "frames": evidence["count"],
                "timeline_alignment": manifest["outputs"].get("timeline_alignment"),
                "encoded_cfr": evidence["cfr"],
            },
            "fill": {
                "frames": records,
                "totals": {
                    **totals,
                    "fill_fraction": (totals["filled"] / totals["invalid"]) if totals["invalid"] else 0.0,
                    "invalid_fraction": totals["invalid"] / float(total_pixels) if total_pixels else 0.0,
                    "filled_fraction": totals["filled"] / float(total_pixels) if total_pixels else 0.0,
                },
            },
            "candidate": {
                "path": str(out_video), "report": str(out_json), "sha256": out_sha,
                "size_bytes": int(size_bytes),
                "width": evidence["width"], "height": evidence["height"], "fps": evidence["fps"],
                "frames": evidence["count"], "crf": crf, "preset": preset,
                "cfr": out_cfr, "within_inline_cap": True,
                "inline_cap_bytes": int(Gemini.MAX_INLINE_BYTES),
            },
            "command": [sys.executable, *sys.argv],
            "no_network": True,
        }
        MF.write_manifest(out_json, report)
        return report
    except BaseException:
        if encoder is not None:
            try:
                encoder.close()
            except Exception:
                pass
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)
        out_video.unlink(missing_ok=True)
        out_json.unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(prog="bench/temporal_border_fill_pilot.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", help="successful bench/run_case manifest.json")
    p.add_argument("--radius", type=int, default=DEFAULT_RADIUS,
                   help=f"maximum temporal donor distance in frames (default: %(default)s, max: {MAX_RADIUS})")
    p.add_argument("--max-bilateral-delta", type=float, default=None,
                   help="experimental: require valid neighbors on both sides and max RGB channel difference "
                        "<= this threshold (0..255); disagreements stay unresolved; default keeps the baseline policy")
    p.add_argument("--out-video", default=None,
                   help="candidate MP4 (default: <manifest-dir>/temporal_fill_r<radius>.mp4)")
    p.add_argument("--out-json", default=None, help="report JSON (default: <out-video stem>.json)")
    p.add_argument("--crf", type=int, default=DEFAULT_CRF, help="libx264 CRF for the candidate")
    p.add_argument("--preset", default=DEFAULT_PRESET, help="libx264 preset for the candidate")
    return p


def run(opts, log=print) -> dict:
    """Validate the options, run the pilot and write the artifacts. Returns the report dict."""
    return build(opts.manifest, out_video=opts.out_video, out_json=opts.out_json, radius=opts.radius,
                 crf=opts.crf, preset=opts.preset, max_bilateral_delta=opts.max_bilateral_delta, log=log)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args, log=lambda m: print(m, file=sys.stderr))
    except Exception as exc:  # structured failure, nonzero; no partial artifact is left behind
        print(json.dumps({"status": "failure", "error": {"type": type(exc).__name__, "message": str(exc)}},
                         sort_keys=True))
        return 1
    print(json.dumps({"status": "success", "video": report["candidate"]["path"],
                      "report": report["candidate"]["report"], "radius": report["radius"],
                      "frames": report["candidate"]["frames"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
