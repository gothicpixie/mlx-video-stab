"""Research-only rendered composition candidate from a captured plan + fitted motion.

This is the **rendered** companion to ``bench.real_composition_audit``. That audit is geometry-only: it probes
the captured plan and the Stage 1 composed candidate on four sparse straight lines and renders no pixels. The
next E02 composition-candidate gate is a perceptual check on **rendered candidates**, so this module turns the
same captured evidence into one explicit candidate MP4:

- the successful ``bench/run_case.py`` manifest is validated (status, source SHA, output cadence, captured
  plan/motion), the captured ``plan_r0.npz``/``motion_r0.npz`` are loaded through the same strict loader
  (``real_composition_audit.load_composition_inputs``: ``allow_pickle=False``, schema/shape/finiteness, stride-1
  composition scope, config-vs-array-shape checks);
- the output->source map is built with the chosen ``bench.composition_candidate`` mode (exact composition, or
  one of the two temporal-high-pass charts -- the chart is computed once for the whole clip with
  ``CC.smoothed_delta``). The composed reference map is **not** recomputed per output frame (the exact
  full-history composition is O(T^2) and was impractical); instead a **dense-map** is forward-propagated once
  per motion fit on a **padded world-pixel lattice** with ``CC.forward_apply`` and bilinearly interpolated at
  the candidate output->reference coordinates ``V_t(x) = inv(Psi_t) @ plan.G[t](x)``. The lattice is the
  output pixel-center rectangle's ``V_t`` image over every requested source frame (computed from the four
  output corners through ``V_t``), grown by a one-pixel interpolation guard so no query is border-clamped, and
  is stored in world pixel coordinates (negative origin allowed). A projective horizon or sign change of the
  map, or a non-finite mapping, is refused. This is a **dense-grid interpolation approximation** to the
  continuous exact composition, not mathematically exact; the grid, origin, padding, interpolation, resolution
  and exact-vs-dense error probes are recorded in the provenance JSON;
- each mapped source frame is rendered with the **production** ``stabilize.render_from_src`` sampler, padding
  and kernel, and the result is encoded at the manifest's exact output dimensions/fps/frame count.

Why this pipeline is special (source vs output cadence)
-------------------------------------------------------
This capture deinterlaces to the source processing rate (59.94 p) and encodes a 29.97 output. Output frame
``i`` therefore corresponds to the **nearest** source decoded frame ``indices[i]`` and the same plan/motion
frame index. The mapping is computed with ``bench.real_pair_review.source_indices`` (exact rational nearest,
half-frame ties/repeats/out-of-range refused); for the current 60-70 s capture 300 outputs pair to 600 source
frames as ``[0, 2, ..., 598]``. Only the mapped source frames are rendered; the source is streamed and the
decoded total is checked so an early or extra decode is refused.

Honesty scope
-------------
This is the **actual rendered version of the geometry-only candidate**. It is a research renderer: no
production code, default or output is changed, no mode is selected, and it makes no network call. Any visual
result is scoped to the one captured run/window; it is not clip-wide or held-out evidence and is not by itself
a quality claim. The MP4 is meant to be compared against the manifest's stabilized ``outputs.video`` with
``bench.real_pair_review --candidate-video`` and then reviewed (human or, separately, Gemini); this tool never
uploads anything.

Payload safety and refusals
---------------------------
Every input is validated before anything is written: the manifest is a successful ``run_case`` manifest with a
matching source SHA, existing baseline video/plan/motion; the plan/motion dimensions/fps/frame count must
independently agree with ``source.decoded`` and carry the expected source-frame timeline; the baseline video
probe/PTS must match the manifest's output dimensions/fps/``encoded_frames``. Unsupported modes, inconsistent
cadence, malformed evidence, early/extra source decode, a bad/non-finite field or render, and encode/CFR
failures are refused with a stable ``code``. No existing artifact (manifest, source, baseline video, plan,
motion) or output MP4/report is ever overwritten, and temporary outputs are removed on failure.

Run with::

    python -m bench.real_composition_render out/bench/RUN/manifest.json \
        --mode exact_composition --out-video out/bench/RUN/candidate_exact.mp4
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

import mlx.core as mx

from mlxstab import geometry as G
from mlxstab import media
from mlxstab import stabilize as S

from . import composition_candidate as CC
from .heavylock import HeavyLock
from . import manifest as MF
from . import real_composition_audit as RCA
from . import real_pair_review as RPR

SCHEMA = "mlxstab.real_composition_render/1"

#: The three explicit candidate modes, exactly the ones of ``bench.real_composition_audit``.
SUPPORTED_MODES = tuple(RCA.CANDIDATE_MODES)
DEFAULT_CRF = 18
DEFAULT_PRESET = "medium"

#: Shared heavy lock (``bench.heavylock.HeavyLock``) chunked-yield defaults: ~30 output frames per chunk with a
#: ~12 s gap, matching the anniversary-repo etiquette and ``bench.run_case --lock-chunk 30 --lock-pause 12``.
DEFAULT_LOCK_CHUNK = 30
DEFAULT_LOCK_PAUSE_S = 12.0

#: Scope note recorded in the report's ``lock`` block.
LOCK_SCOPE_NOTE = (
    "The candidate is rendered under the anniversary project's shared heavy lock (bench.heavylock.HeavyLock, "
    "reused from tools/heavy.py; a local ignored out/ fallback is used when unavailable). The lock is acquired "
    "before the precomputed smoothing chart and every source decode/render step and released in all paths; "
    "lock.on_step(rendered) is called after each output frame, so a long render yields for --lock-pause seconds "
    "at every --lock-chunk boundary. Immediately before each such yield the encoder writer queue is drained "
    "(mlxstab.media.Encoder.drain), so no output frame from the finished chunk is still queued for the writer; "
    "ffmpeg/x264 internal frame buffering and the source-decode subprocess/pipe buffers can still hold frames, "
    "so the yield does not fully quiesce all media work. Acquisition waits and chunk pauses are accounted "
    "separately and are not processing time. The lock filesystem path is intentionally omitted from this "
    "report."
)
DEINTERLACE_FILTER = "bwdif=mode=send_field:parity=auto:deint=all"
CFR_TOL_S = 1e-5

SCOPE_NOTE = (
    "Research-only rendered composition candidate. This is the rendered version of the "
    "bench.real_composition_audit geometry-only (no-media) candidate: the same captured plan and fitted motion "
    "are loaded through the strict loader, the chosen bench.composition_candidate candidate is built for each "
    "mapped source frame, and the frame is rendered through the production stabilize.render_from_src "
    "sampler/kernel. The output->source map is a dense-grid interpolation approximation to the continuous "
    "exact composition: the composed reference map Phi_t is forward-propagated once per motion fit on a padded "
    "world-pixel lattice (pixel-center identity at t=0) that covers every queried V_t(x) with a one-pixel guard "
    "and is bilinearly interpolated at the candidate output->reference coordinates V_t(x) = inv(Psi_t) @ "
    "plan.G[t](x). It is exact on lattice nodes and interpolated between them, so it is NOT mathematically "
    "exact; the grid, origin, padding, interpolation, resolution and direct exact-vs-dense error probes are "
    "recorded in the report. It writes one candidate MP4 plus one adjacent atomic JSON for a later local "
    "(human or Gemini) review and makes no network call. No mode is selected and no production code, default "
    "or output is changed."
)

#: Name of the dense-map approximation used for the output->source map (recorded in provenance).
APPROX_NAME = "dense_grid_interpolated_forward_composition"

#: One-line description of the approximation (recorded in provenance).
APPROX_NOTE = (
    "Dense-grid interpolation approximation to the continuous exact composition. Phi_t(z) = F_{t-1} o ... o "
    "F_0(z) is forward-propagated once per motion fit on a padded world-pixel lattice (identity at t=0) with "
    "composition_candidate.forward_apply, then bilinearly interpolated at the candidate output->reference "
    "coordinates V_t(x) = inv(Psi_t) @ plan.G[t](x). The lattice is the output pixel-center rectangle's V_t "
    "image over every requested source frame (the four output corners transformed through V_t), grown by one "
    "interpolation guard pixel, so every query is inside the lattice and no border clamp is applied. Exact "
    "full-history composition recomputes t fits per output frame (O(T^2) in the clip length); this propagates "
    "one fit per frame (O(T)). It is exact on lattice nodes and interpolated between them, so it is NOT "
    "mathematically exact at arbitrary reference coordinates."
)

#: Reference-lattice geometry/interpolation/resolution recorded in provenance.
APPROX_GRID = (
    "padded world-pixel lattice covering the output pixel-center rectangle's V_t image over every requested "
    "source frame (four corners through V_t = inv(Psi_t) @ plan.G[t]), plus one interpolation guard pixel on "
    "every side; world coordinates, negative origin allowed"
)
APPROX_INTERPOLATION = (
    "bilinear over the padded world-pixel lattice; the one-pixel guard keeps every query and bilinear tap "
    "inside the lattice, so edge clamping is not exercised"
)

#: Interpolation guard (integer pixels) grown on each side of the queried V_t(x) bounding box.
GRID_GUARD_PX = 1

#: Number of representative output frames and the sparse comparison grid for the exact-vs-dense probe.
ERROR_FRAME_COUNT = 5
COMPARISON_GRID = (9, 7)
COMPARISON_INSET_PX = 1.0

#: Fixed edge-strip width (pixels) for the invalid-support provenance report.
SUPPORT_STRIP_PX = 4

#: Definition of the invalid-support metric, recorded verbatim in the provenance report.
SUPPORT_DEFINITION = (
    "Invalid support is the boolean border mask returned by the production stabilize.render_from_src sampler: "
    "a rendered output pixel is invalid when its sampled source coordinate lies outside the pixel-edge interval "
    "[-0.5, W-0.5] x [-0.5, H-0.5] (the production warp.valid_mask convention), i.e. its center maps "
    "out-of-bounds of the source frame, and render_from_src fills it with zero. These are center-out-of-bounds "
    "pixels from S.render_from_src, a geometry/support diagnostic, and NOT a perceptual or visual-quality "
    "metric. No mask video or mask artifact is written; only per-frame counts/fractions and aggregates are "
    "recorded."
)

LIMITATIONS = (
    "research-only rendered geometry candidate; not a production integration and not a production default",
    "the output->source map is a dense-grid interpolation approximation to the continuous exact composition "
    "(forward-propagated once per motion fit on a padded world-pixel lattice that covers every queried V_t(x) "
    "with a one-pixel guard, then bilinearly sampled at V_t(x)), not mathematically exact; exact-vs-dense "
    "pixel errors on representative sparse points/frames are recorded in the report",
    "this renders the same composed candidate that bench.real_composition_audit measures geometrically; any "
    "visual result is the actual rendered version of one geometry-only candidate, not a quality claim",
    "one setup/window only: the render uses the captured plan/motion of a single run and source window, so any "
    "visual result is scoped to that capture and is not clip-wide or held-out evidence",
    "each explicit mode is a separate candidate and none is selected; the high-pass smoothing chart is a "
    "decision experiment, not a decided convention",
    "the candidate field is applied at the nearest mapped source frame; the production stabilized baseline "
    "uses the same source frames with its own map, so a baseline/candidate review isolates the map choice",
    "no network call and no upload; a later review (bench.real_pair_review --candidate-video, then "
    "bench.gemini_blind_review) happens separately",
    "the invalid-support diagnostics count center-out-of-bounds samples from stabilize.render_from_src "
    "(zero-filled pixels) and are not a perceptual metric; no mask video or mask artifact is written",
)


class RealCompositionRenderError(ValueError):
    """A rendered composition candidate was refused. ``code`` is a stable machine-readable reason."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = str(code)


def _require(cond, code, message):
    if not cond:
        raise RealCompositionRenderError(code, message)


def _close(a, b, tol=1e-5):
    try:
        return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _rate(value, where):
    """Positive rational frame rate from a manifest value (``"30000/1001"`` or a float)."""
    try:
        result = Fraction(str(value)).limit_denominator(1_000_000)
    except (ValueError, ZeroDivisionError, TypeError):
        raise RealCompositionRenderError("bad_rate", f"{where} must be a positive finite frame rate") from None
    if result <= 0:
        raise RealCompositionRenderError("bad_rate", f"{where} must be a positive finite frame rate")
    return result


def verify_cfr(pts, count, fps, where, tol_s=CFR_TOL_S):
    """Verify one encoded stream follows the zero-origin ``i/fps`` CFR clock for ``count`` frames."""
    reasons = []
    if not isinstance(pts, dict) or pts.get("count") != count or pts.get("n_pts") != count:
        reasons.append(f"{where} PTS count does not match the expected frame count")
    else:
        values = pts.get("pts_s")
        if not pts.get("monotonic") or not isinstance(values, list) or len(values) != count:
            reasons.append(f"{where} PTS are missing or non-monotonic")
        elif any(v is None or not np.isfinite(v) or abs(float(v) - i / float(fps)) > tol_s
                 for i, v in enumerate(values)):
            reasons.append(f"{where} is not zero-based constant-frame-rate")
    return {"ok": not reasons, "reasons": reasons, "count": int(count), "fps": float(fps),
            "tolerance_s": tol_s}


def lock_snapshot(lock):
    """Report block for the shared heavy lock; the private lock filesystem path is deliberately omitted.

    ``lock`` is an (optionally already acquired) ``bench.heavylock.HeavyLock`` or a compatible object. ``None``
    records the disabled state used by the Python test seam: unit tests inject no lock and never acquire a real
    one. Only the portable accounting is kept (backend/chunk/pause/release/acquire-wait/pause totals).
    """
    if lock is None:
        return {"backend": "disabled", "enabled": False, "chunk": None, "pause_s": None, "held": False,
                "releases": 0, "acquire_wait_s": 0.0, "pause_total_s": 0.0, "scope": LOCK_SCOPE_NOTE}
    st = lock.status()
    backend = st.get("backend")
    return {"backend": backend, "enabled": backend != "disabled", "chunk": st.get("chunk"),
            "pause_s": st.get("pause_s"), "held": bool(st.get("held")),
            "releases": int(st.get("releases") or 0),
            "acquire_wait_s": float(st.get("acquire_wait_s") or 0.0),
            "pause_total_s": float(st.get("pause_total_s") or 0.0), "scope": LOCK_SCOPE_NOTE}


# --------------------------------------------------------------------------------------
# evidence validation
# --------------------------------------------------------------------------------------

def load_manifest(manifest_path):
    """Read a successful ``run_case`` manifest; return ``(manifest, resolved_path)``."""
    path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RealCompositionRenderError("manifest_unreadable",
                                         f"cannot read run manifest {str(path)!r}: {exc}") from exc
    _require(isinstance(manifest, dict), "manifest_malformed", "manifest top level must be a JSON object")
    _require(manifest.get("schema") == MF.MANIFEST_SCHEMA, "manifest_schema",
             f"manifest schema must be {MF.MANIFEST_SCHEMA!r}, got {manifest.get('schema')!r}")
    _require(manifest.get("status") == "success", "manifest_status",
             f"manifest status must be 'success', got {manifest.get('status')!r}")
    return manifest, path


def validate_source(manifest, manifest_path):
    """Validate the manifest source identity and decoded metadata; return a plain dict of the evidence."""
    source = manifest.get("source")
    _require(isinstance(source, dict), "manifest_malformed", "manifest source object is missing")
    _require(source.get("path_type") == "file", "bad_source",
             "rendered composition candidate requires a local file source (path_type == 'file')")
    src_path = Path(source.get("path") or "").expanduser()
    _require(bool(str(source.get("path") or "")) and src_path.is_file(), "source_missing",
             "manifest source file is missing")
    src_path = src_path.resolve()
    _require(src_path != manifest_path, "source_missing", "manifest source collides with the manifest")
    sha = source.get("sha256")
    _require(isinstance(sha, str) and sha, "source_sha", "manifest source.sha256 is missing")
    _require(MF.sha256_file(src_path) == sha, "source_sha",
             "source SHA-256 does not match the benchmark manifest")

    window = source.get("window")
    _require(isinstance(window, dict), "bad_source_window", "manifest source.window is missing")
    start, duration = window.get("start_s"), window.get("duration_s")
    _require(_close(start, start) and float(start) >= 0, "bad_source_window",
             "manifest source.window.start_s is invalid")
    _require(_close(duration, duration) and float(duration) > 0, "bad_source_window",
             "manifest source.window.duration_s is invalid")

    preprocessing = source.get("preprocessing")
    _require(isinstance(preprocessing, dict), "bad_preprocessing",
             "manifest source.preprocessing is missing")
    if preprocessing.get("interlaced"):
        _require(preprocessing.get("deinterlace") == DEINTERLACE_FILTER, "bad_preprocessing",
                 "interlaced source must record the expected bwdif deinterlace filter")

    decoded = source.get("decoded")
    _require(isinstance(decoded, dict), "bad_source_decoded", "manifest source.decoded is missing")
    frames = decoded.get("frames")
    _require(isinstance(frames, int) and not isinstance(frames, bool) and frames >= 1, "bad_source_decoded",
             "manifest source.decoded.frames must be a positive integer")
    _require(decoded.get("width") and decoded.get("height"), "bad_source_decoded",
             "manifest source.decoded dimensions are missing")
    _rate(decoded.get("fps"), "manifest source processing fps")
    return {
        "path": src_path, "sha256": sha, "window": window, "preprocessing": preprocessing,
        "decoded": decoded,
    }


def validate_baseline(manifest, source_path):
    """Validate the manifest's stabilized output video against its own decoded metadata and CFR clock."""
    outputs = manifest.get("outputs")
    _require(isinstance(outputs, dict), "manifest_malformed", "manifest outputs object is missing")
    video = outputs.get("video")
    _require(isinstance(video, str) and video.strip(), "baseline_missing",
             "manifest outputs.video is missing or empty")
    baseline_path = Path(video).expanduser().resolve()
    _require(baseline_path.is_file(), "baseline_missing", "manifest output video is missing on disk")
    _require(baseline_path != source_path, "baseline_collision",
             "manifest output video collides with the raw source")

    decoded = outputs.get("decoded")
    _require(isinstance(decoded, dict), "manifest_malformed", "manifest outputs.decoded is missing")
    count = outputs.get("encoded_frames")
    _require(isinstance(count, int) and not isinstance(count, bool) and count >= 1, "baseline_frames",
             "manifest outputs.encoded_frames must be a positive integer")
    _require(decoded.get("frames") == count, "baseline_frames",
             "manifest outputs.decoded.frames disagrees with outputs.encoded_frames")
    out_fps = _rate(decoded.get("fps"), "manifest output fps")

    info = media.probe(baseline_path)
    _require((info.get("width"), info.get("height")) == (decoded.get("width"), decoded.get("height")),
             "baseline_dimensions", "baseline video probe disagrees with the manifest decoded dimensions")
    _require(_close(info.get("fps"), float(out_fps)), "baseline_fps",
             "baseline video probe disagrees with the manifest decoded frame rate")
    cfr = verify_cfr(media.frame_pts(baseline_path), count, out_fps, "baseline output")
    _require(cfr["ok"], "baseline_cfr", "baseline output CFR verification failed: " + "; ".join(cfr["reasons"]))

    recorded_pts = outputs.get("encoded_pts")
    if isinstance(recorded_pts, dict) and recorded_pts.get("count") is not None:
        _require(int(recorded_pts["count"]) == count, "baseline_pts",
                 "manifest outputs.encoded_pts count disagrees with encoded_frames")
    _require(outputs.get("timeline_alignment", {}).get("aligned") is True, "baseline_alignment",
             "manifest output timeline alignment failed")
    cadence = outputs.get("cadence") or {}
    _require(_close(cadence.get("measured_fps"), float(out_fps)) and cadence.get("measured_frames") == count,
             "baseline_cadence", "manifest measured cadence disagrees with the decoded output")
    return {"path": baseline_path, "sha256": MF.sha256_file(baseline_path), "count": int(count),
            "fps": out_fps, "width": int(info["width"]), "height": int(info["height"]), "cfr": cfr}


def validate_timeline(plan_meta, motion_meta, source_decoded):
    """Independently require the plan/motion to carry the expected source-frame timeline."""
    decoded_dims = (source_decoded.get("width"), source_decoded.get("height"))
    decoded_fps = source_decoded.get("fps")
    decoded_frames = source_decoded.get("frames")
    for label, meta, frames_key in (("plan", plan_meta, "frames"), ("motion", motion_meta, "n_frames")):
        _require((meta["width"], meta["height"]) == decoded_dims, "timeline_dimensions",
                 f"{label} is {meta['width']}x{meta['height']} but source.decoded is "
                 f"{decoded_dims[0]}x{decoded_dims[1]}")
        _require(_close(meta["fps"], decoded_fps), "timeline_fps",
                 f"{label} fps {meta['fps']!r} disagrees with source.decoded fps {decoded_fps!r}")
        _require(meta[frames_key] == decoded_frames, "timeline_frames",
                 f"{label} {frames_key}={meta[frames_key]} disagrees with source.decoded.frames="
                 f"{decoded_frames}")


def _resolve_input(base_dir, value):
    """Resolve a captured plan/motion artifact recorded in a run manifest.

    Absolute manifest paths are preserved verbatim. A relative value may be recorded either as a bare
    basename or as a repo-root-relative path (``out/bench/RUN/plan_r0.npz``), but the captured artifact is
    always a sibling of the benchmark manifest, so a relative value resolves against ``base_dir`` by
    basename. Prepending the whole relative path would duplicate the run directory (``RUN/out/bench/RUN/..``)
    and lose the sibling. ``real_composition_audit.load_composition_inputs`` still enforces sibling,
    capture-flag and basename provenance, so a wrong or foreign artifact is refused.
    """
    path = Path(str(value or "")).expanduser()
    return path if path.is_absolute() else (base_dir / path.name)


def _resolve_outputs(manifest_path, out_video, out_json, protected):
    video_path = Path(out_video).resolve()
    json_path = Path(out_json).resolve() if out_json else video_path.with_suffix(".json")
    _require(video_path.suffix.lower() == ".mp4", "output_suffix",
             "candidate video must use the .mp4 suffix")
    _require(json_path.suffix.lower() == ".json", "output_suffix",
             "provenance report must use the .json suffix")
    _require(video_path.parent == json_path.parent, "output_directory",
             "candidate MP4 and report must be written in the same directory")
    _require(video_path != json_path, "output_collision", "candidate MP4 and report paths must differ")
    _require(video_path not in protected and json_path not in protected, "output_collision",
             "candidate output collides with an input artifact")
    for path in (video_path, json_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    return video_path, json_path


# --------------------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------------------

def render_candidate_frame(frame, field, cfg):
    """Wrap a NumPy ``[H,W,2]`` candidate field and render it with the production sampler."""
    src = mx.array(np.asarray(field, np.float32))[None]
    img, bad = S.render_from_src(frame, src, cfg)
    return np.asarray(img), np.asarray(bad)


def _edge_strip_slices(shape, strip_px):
    """``(top, bottom, left, right)`` slice pairs for fixed-width edge strips of a ``(H, W)`` frame.

    The strip width is clamped to the frame dimensions so a frame narrower/shorter than ``strip_px`` still
    yields a well-defined (possibly full-frame) strip. The top/bottom strips span every column and the
    left/right strips span every row.
    """
    H, W = int(shape[0]), int(shape[1])
    if H < 1 or W < 1:
        raise ValueError("frame shape must be positive")
    s = min(int(strip_px), H, W)
    if s < 1:
        raise ValueError("strip_px must be >= 1")
    return {
        "top": (slice(0, s), slice(0, W)),
        "bottom": (slice(H - s, H), slice(0, W)),
        "left": (slice(0, H), slice(0, s)),
        "right": (slice(0, H), slice(W - s, W)),
    }


def support_frame_stats(mask, shape, *, strip_px=SUPPORT_STRIP_PX):
    """Per-frame invalid-support counts/fractions from one renderer border mask.

    ``mask`` is the ``bad`` mask returned by :func:`render_candidate_frame` (``stabilize.render_from_src``):
    ``True`` where the sampled source-coordinate center lies out-of-bounds. It must be a 2-D boolean-like
    array matching ``shape = (H, W)``; a wrong shape or an unsupported dtype is refused with a stable code
    (``bad_mask_shape``/``bad_mask_dtype``). Returns ``invalid_pixels``/``invalid_fraction`` plus per-strip
    ``edge_invalid_pixels``/``edge_fraction`` for the fixed top/bottom/left/right strips; it stores no mask.
    """
    H, W = int(shape[0]), int(shape[1])
    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.shape != (H, W):
        raise RealCompositionRenderError(
            "bad_mask_shape", f"renderer border mask has shape {arr.shape}, expected {(H, W)}")
    if arr.dtype.kind not in "biuf":
        raise RealCompositionRenderError(
            "bad_mask_dtype",
            f"renderer border mask has unsupported dtype {arr.dtype!r}; expected boolean/integer/float")
    bad = arr.astype(bool)
    invalid = int(np.count_nonzero(bad))
    total = H * W
    edge_invalid, edge_fraction = {}, {}
    for name, (ys, xs) in _edge_strip_slices((H, W), strip_px).items():
        sub = bad[ys, xs]
        n = int(sub.size)
        count = int(np.count_nonzero(sub))
        edge_invalid[name] = count
        edge_fraction[name] = (float(count) / n) if n else 0.0
    return {
        "invalid_pixels": invalid,
        "frame_pixels": total,
        "invalid_fraction": (float(invalid) / total) if total else 0.0,
        "edge_invalid_pixels": edge_invalid,
        "edge_fraction": edge_fraction,
    }


def aggregate_support(stats, shape, *, strip_px=SUPPORT_STRIP_PX):
    """Aggregate per-frame :func:`support_frame_stats` into a compact report section.

    Memory stays O(number of output frames): each entry is a small scalar/4-element dict and no mask or mask
    video is retained. Empty ``stats`` yields defined zero aggregates.
    """
    H, W = int(shape[0]), int(shape[1])
    frames = len(stats)
    total_pixels = frames * H * W
    total_invalid = int(sum(int(s["invalid_pixels"]) for s in stats))
    fractions = [float(s["invalid_fraction"]) for s in stats]
    edges = {}
    for name, (ys, xs) in _edge_strip_slices((H, W), strip_px).items():
        area = int((ys.stop - ys.start) * (xs.stop - xs.start))
        strip_total_pixels = frames * area
        strip_invalid = int(sum(int(s["edge_invalid_pixels"][name]) for s in stats))
        per = [float(s["edge_fraction"][name]) for s in stats]
        edges[name] = {
            "strip_px": int(ys.stop - ys.start) if name in ("top", "bottom") else int(xs.stop - xs.start),
            "per_frame_pixels": area,
            "total_pixels": strip_total_pixels,
            "total_invalid_pixels": strip_invalid,
            "pooled_fraction": (float(strip_invalid) / strip_total_pixels) if strip_total_pixels else 0.0,
            "mean_fraction": (sum(per) / frames) if frames else 0.0,
            "max_fraction": (max(per) if per else 0.0),
        }
    return {
        "frames": frames,
        "frame_pixels": H * W,
        "total_pixels": total_pixels,
        "total_invalid_pixels": total_invalid,
        "pooled_invalid_fraction": (float(total_invalid) / total_pixels) if total_pixels else 0.0,
        "mean_invalid_fraction": (sum(fractions) / frames) if frames else 0.0,
        "max_invalid_fraction": (max(fractions) if fractions else 0.0),
        "edge": edges,
    }


def _pixel_axis(coord, n):
    """Bilinear tap indices/weights over ``n`` pixel centres with border clamping.

    Mirrors ``composition_candidate._axis`` and the production ``warp.sample`` border convention: a query at an
    integer pixel centre is exact, between centres is interpolated, and outside ``[0, n-1]`` clamps to the edge.
    """
    base = np.floor(coord).astype(np.int64)
    f = coord - base
    i0 = np.clip(base, 0, n - 1)
    i1 = np.clip(base + 1, 0, n - 1)
    return i0, i1, 1.0 - f, f


def sample_pixel_lattice(field, pts, W, H):
    """Bilinear+border sample of a dense ``[H,W,2]`` (or ``[H*W,2]``) map at pixel coordinates ``pts [N,2]``."""
    fld = np.asarray(field, np.float64).reshape(H, W, 2)
    pts = np.asarray(pts, np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be [N,2] pixel coordinates")
    x0, x1, wx0, wx1 = _pixel_axis(pts[:, 0], W)
    y0, y1, wy0, wy1 = _pixel_axis(pts[:, 1], H)
    w00 = wy0 * wx0; w01 = wy0 * wx1; w10 = wy1 * wx0; w11 = wy1 * wx1
    return (fld[y0, x0] * w00[:, None] + fld[y0, x1] * w01[:, None]
            + fld[y1, x0] * w10[:, None] + fld[y1, x1] * w11[:, None])


def _output_corners(W, H):
    """The four output pixel-center rectangle corners whose ``V_t`` image bounds every output query."""
    return np.array([[0.0, 0.0], [W - 1.0, 0.0], [W - 1.0, H - 1.0], [0.0, H - 1.0]], np.float64)


def project_query_bounds(plan, mot, indices, *, guard_px=GRID_GUARD_PX):
    """World-pixel lattice bounds covering every queried ``V_t(x)`` over the requested source indices.

    For every requested source index ``t`` the recovered output->reference map
    ``V_t = inv(Psi_t) @ plan.G[t]`` is applied to the four output pixel-center rectangle corners. ``V_t`` is a
    projective map and the output query set is the axis-aligned pixel-center rectangle, so its image is a
    convex quadrilateral and the coordinate extrema are attained at those four corners; the corner bounding
    box therefore covers every interior query ``V_t(x)``. The homogeneous denominator ``w`` is checked at the
    corners: a zero, a non-finite value, or a sign change means the projective horizon crosses the output
    rectangle, so the image is not a bounded quadrilateral and the map is refused
    (``projective_horizon``). A non-finite mapped corner is refused (``nonfinite_mapping``). One guard pixel is
    added on each side so every bilinear tap of every query stays inside the lattice (no border clamping).

    Returns a dict ``{"origin", "upper", "min", "max"}`` of world-pixel bounds (``origin``/``upper`` are
    integer lattice endpoints inclusive; ``min``/``max`` are the raw floating query bounds).
    """
    W, H = int(mot.W), int(mot.H)
    corners = _output_corners(W, H)
    lo = np.array([np.inf, np.inf], np.float64)
    hi = np.array([-np.inf, -np.inf], np.float64)
    psi = np.eye(3)
    prev = 0
    requested = sorted({int(t) for t in indices})
    _require(bool(requested), "empty_query_bounds", "no source indices were requested for the dense lattice")
    for t in requested:
        _require(0 <= t < len(plan.G), "bad_frame_index",
                 f"requested source index {t} is outside the plan range [0, {len(plan.G)})")
        for k in range(prev, t):
            psi = np.asarray(mot.fits[k].H, np.float64) @ psi
        prev = t
        try:
            psi_inv = np.linalg.inv(psi)
        except np.linalg.LinAlgError:
            raise RealCompositionRenderError(
                "projective_horizon",
                f"global path Psi_{t} is singular; V_t cannot be bounded") from None
        V = psi_inv @ np.asarray(plan.G[t], np.float64)
        if not np.isfinite(V).all():
            raise RealCompositionRenderError(
                "nonfinite_mapping", f"V_{t} contains non-finite values; refusing to size the dense lattice")
        w = corners @ V[2, :2] + V[2, 2]
        if not np.isfinite(w).all() or np.any(w == 0.0) or not (bool(np.all(w > 0.0)) or bool(np.all(w < 0.0))):
            raise RealCompositionRenderError(
                "projective_horizon",
                f"V_{t} has a projective horizon/sign change across the output rectangle; the mapped image "
                f"is unbounded and the dense lattice cannot cover it")
        z = G.apply_h(V, corners)
        if not np.isfinite(z).all():
            raise RealCompositionRenderError(
                "nonfinite_mapping", f"V_{t} maps an output corner to a non-finite coordinate")
        lo = np.minimum(lo, z.min(0))
        hi = np.maximum(hi, z.max(0))
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()) or bool(np.any(hi < lo)):
        raise RealCompositionRenderError(
            "empty_query_bounds", "queried V_t(x) bounds are empty or non-finite; refusing the dense lattice")
    origin = (np.floor(lo) - float(guard_px)).astype(np.int64)
    upper = (np.ceil(hi) + float(guard_px)).astype(np.int64)
    return {"origin": origin, "upper": upper, "min": lo, "max": hi}


def forward_composed_map(plan, mot, indices, *, guard_px=GRID_GUARD_PX, check_finite=True):
    """Build a :class:`ForwardComposedMap` on the padded world-pixel lattice covering all queried ``V_t(x)``.

    ``indices`` are every source frame index that will be rendered (the dense map is advanced monotonically),
    so the padded lattice covers every ``V_t(x)`` those frames will query. The lattice is built on integer
    world pixel coordinates ``origin .. upper`` inclusive (a negative origin is allowed), with the requested
    guard pixel on each side.
    """
    bounds = project_query_bounds(plan, mot, indices, guard_px=guard_px)
    origin, upper = bounds["origin"], bounds["upper"]
    nx = int(upper[0] - origin[0] + 1)
    ny = int(upper[1] - origin[1] + 1)
    _require(nx >= 2 and ny >= 2, "degenerate_lattice",
             f"padded dense lattice {nx}x{ny} is too small to interpolate")
    xs = origin[0] + np.arange(nx)
    ys = origin[1] + np.arange(ny)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    lattice = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64)
    composed = ForwardComposedMap(mot, lattice=lattice, origin=origin.astype(np.float64),
                                  check_finite=check_finite)
    composed.query_min = np.array(bounds["min"], np.float64)
    composed.query_max = np.array(bounds["max"], np.float64)
    composed.guard_px = int(guard_px)
    return composed


class ForwardComposedMap:
    """Sequential forward-propagated dense composed source map (dense-grid interpolation approximation).

    Propagates ``Phi_t(z) = F_{t-1} o ... o F_0(z)`` **once per motion fit** on a world-pixel lattice
    (pixel-center identity at ``t=0``) with the existing ``composition_candidate.forward_apply`` and maintains
    the accumulated global ``Psi_t = H_{t-1} @ ... @ H_0``. The composed map is then bilinearly interpolated at
    query reference coordinates ``z``. Building every output frame's field is therefore O(T) forward passes
    over the lattice instead of the O(T^2) of recomputing the exact composition per output frame; the price is
    that the composition is interpolated between lattice nodes.

    The lattice is stored in **world pixel coordinates** with an integer ``origin`` (negative allowed) set by
    :func:`forward_composed_map` to cover every queried ``V_t(x)`` plus a one-pixel guard. Sampling subtracts
    ``origin`` to move into the lattice's own index space, so no query is border-clamped. ``forward_apply``
    still uses ``self.W``/``self.H`` for the mesh/row reference-grid conventions, independent of the lattice
    extent.
    """

    def __init__(self, mot, *, lattice=None, origin=None, check_finite=True):
        self.W = int(mot.W)
        self.H = int(mot.H)
        self.fits = list(mot.fits)
        if lattice is None:
            self.lattice = CC.output_grid(self.W, self.H)
            self.origin = np.zeros(2, np.float64)
        else:
            self.lattice = np.asarray(lattice, np.float64)
            self.origin = np.asarray(origin if origin is not None else np.zeros(2), np.float64).reshape(2)
        if self.lattice.ndim != 2 or self.lattice.shape[1] != 2:
            raise ValueError("dense lattice must be an [N,2] array of world pixel coordinates")
        xs = np.unique(self.lattice[:, 0])
        ys = np.unique(self.lattice[:, 1])
        if xs.size * ys.size != self.lattice.shape[0]:
            raise ValueError("dense lattice must be a complete rectangular mesh (x fastest, then y)")
        self.grid_shape = (int(ys.size), int(xs.size))     # (rows, cols)
        self.phi = self.lattice.copy()                     # Phi_0 = identity on the lattice
        self.psi = np.eye(3)                               # Psi_0 = identity
        self.t = 0
        self.check_finite = bool(check_finite)
        self.query_min = None                               # filled by forward_composed_map
        self.query_max = None
        self.guard_px = None

    def advance_to(self, t):
        """Apply each remaining fit once so the map state is ``Phi_t``/``Psi_t`` (monotonic forward only)."""
        t = int(t)
        if t < self.t:
            raise ValueError(f"forward composed map is at frame {self.t}; cannot rewind to {t}")
        while self.t < t:
            self.phi = CC.forward_apply(self.fits[self.t], self.phi, self.W, self.H)
            self.psi = np.asarray(self.fits[self.t].H, np.float64) @ self.psi
            self.t += 1
            if self.check_finite and not np.isfinite(self.phi).all():
                raise RealCompositionRenderError(
                    "nonfinite_map",
                    f"forward-propagated composed map is non-finite at source frame {self.t}")

    def sample(self, z):
        """Bilinear sample of the current composed map ``Phi_t`` at world reference coordinates ``z``.

        ``z`` are world pixel coordinates; the lattice's index space is ``z - origin`` (the lattice origin may
        be negative and extends beyond the frame, so every queried point has all four bilinear taps valid).
        """
        n = self.grid_shape
        return sample_pixel_lattice(self.phi, np.asarray(z, np.float64) - self.origin, n[1], n[0])

    def source_coords(self, plan, t, points=None, *, cells, highpass, smoothed, smoothing_basis):
        """Approximate output->source coordinates at output pixel points ``points`` (default ``output_grid``)."""
        self.advance_to(t)
        W, H = self.W, self.H
        if points is None:
            points = CC.output_grid(W, H)
        points = np.asarray(points, np.float64)
        psi = self.psi
        psi_inv = np.linalg.inv(psi)
        z = G.apply_h(psi_inv @ np.asarray(plan.G[int(t)], np.float64), points)   # output -> reference
        phi_z = self.sample(z)                                                   # ~= Phi_t(z)
        if not highpass:
            return phi_z                                                         # == base + (Phi_t(z) - Psi_t(z))
        if smoothed is None:
            raise ValueError("highpass dense-map sampling requires the precomputed smoothed chart")
        smooth_z = CC.sample_grid_bilinear(smoothed[int(t)], z, W, H, cells)
        if smoothing_basis == "source":
            return phi_z - smooth_z
        eta_hp = G.apply_h(psi_inv, phi_z) - z - smooth_z
        return G.apply_h(psi, z + eta_hp)                                       # exact projective push-forward

    def field(self, plan, t, *, cells, highpass, smoothed, smoothing_basis):
        """Full ``[H,W,2]`` output->source field for one output frame."""
        coords = self.source_coords(plan, t, None, cells=cells, highpass=highpass, smoothed=smoothed,
                                    smoothing_basis=smoothing_basis)
        return np.asarray(coords, np.float64).reshape(self.H, self.W, 2)


def build_candidate_field(composed_map, plan, cfg, t, *, cells, highpass, smoothed, smoothing_basis):
    """Build the ``[H,W,2]`` candidate field for one output frame from the forward-propagated dense map.

    This is the single per-frame seam over :class:`ForwardComposedMap` so the renderer never calls
    ``composition_candidate.candidate_field`` (the O(T^2)-per-clip exact composition) in its frame loop.
    """
    return composed_map.field(plan, t, cells=cells, highpass=highpass, smoothed=smoothed,
                              smoothing_basis=smoothing_basis)


def comparison_points(W, H, grid=COMPARISON_GRID, inset=COMPARISON_INSET_PX):
    """Representative sparse output-space points for the exact-vs-dense error probe.

    A ``grid`` of sub-pixel-offset points inset by ``inset`` pixels from the frame edges, so both bilinear taps
    of the dense lattice are valid (the degenerate last-node clamp is excluded).
    """
    nx, ny = int(grid[0]), int(grid[1])
    xs = np.linspace(float(inset), W - 1.0 - float(inset), nx) + 0.25
    ys = np.linspace(float(inset), H - 1.0 - float(inset), ny) + 0.25
    return np.stack(np.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2).astype(np.float64)


def representative_frames(count, k=ERROR_FRAME_COUNT):
    """Evenly spaced output-frame positions (distinct, inclusive of first and last) for the error probe."""
    n = int(count)
    if n <= 0:
        return []
    k = max(1, min(int(k), n))
    return sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)}) if k > 1 else [n // 2]


def dense_vs_exact_errors(composed_map, mot, plan, cfg, t, *, points, cells, highpass, smoothed,
                          smoothing_basis):
    """Direct exact comparison at ``points``/frame ``t``; return ``(stats, raw_errors)`` in pixels.

    Compares the dense forward-propagated map with ``CC.candidate_source_coords`` (the exact full-history
    composition) on the supplied output-space points and refuses a non-finite dense result or exact reference.
    """
    approx = composed_map.source_coords(plan, t, points, cells=cells, highpass=highpass, smoothed=smoothed,
                                        smoothing_basis=smoothing_basis)
    exact_kwargs = ({"smoothed": smoothed, "smoothing_basis": smoothing_basis} if highpass else {})
    exact = CC.candidate_source_coords(mot, plan, cfg, t, points=points, cells=cells, highpass=highpass,
                                       **exact_kwargs)
    approx = np.asarray(approx, np.float64)
    exact = np.asarray(exact, np.float64)
    if not (np.isfinite(approx).all() and np.isfinite(exact).all()):
        raise RealCompositionRenderError(
            "nonfinite_comparison",
            f"non-finite value in the exact-vs-dense comparison at source frame {int(t)}")
    err = np.linalg.norm(approx - exact, axis=1)
    stats = {"frame": int(t), "points": int(err.size), "max_px": float(err.max()),
             "mean_px": float(err.mean()), "p95_px": float(np.percentile(err, 95))}
    return stats, err


def at_lock_chunk_boundary(lock, step):
    """Mirror ``bench.heavylock.HeavyLock.on_step``'s yield condition without mutating the lock.

    ``chunk <= 0`` (or a compatible object without ``chunk``) never yields, so a barrier is never requested.
    """
    chunk = getattr(lock, "chunk", 0)
    return bool(chunk) and bool(step) and step % int(chunk) == 0


def render_candidate(manifest_path, *, mode, out_video, out_json=None, crf=DEFAULT_CRF,
                     preset=DEFAULT_PRESET, lock=None):
    """Render one explicit composition candidate MP4 from a captured run manifest.

    ``lock`` is an optional, already-constructed ``bench.heavylock.HeavyLock`` (or compatible object): the CLI
    builds it from ``--lock/--lock-chunk/--lock-pause`` and passes it in, while the Python seam defaults to
    ``None`` so unit tests never acquire a real shared lock. When supplied it is acquired before the smoothing
    chart and any source decode/render, ``on_step(rendered)`` is called after every output frame, and it is
    released on every path. At each ``chunk`` boundary the encoder writer queue is drained first so the yield
    does not overlap this render's own queued encoding work.
    """
    _require(mode in SUPPORTED_MODES, "unsupported_mode",
             f"mode must be one of {SUPPORTED_MODES}, got {mode!r}")
    manifest, man_path = load_manifest(manifest_path)
    source = validate_source(manifest, man_path)
    baseline = validate_baseline(manifest, source["path"])

    outputs = manifest["outputs"]
    plan_path = _resolve_input(man_path.parent, outputs.get("plan")).resolve()
    motion_path = _resolve_input(man_path.parent, outputs.get("motion")).resolve()
    _require(bool(str(outputs.get("plan") or "")) and plan_path.is_file(), "plan_missing",
             "manifest captured plan is missing on disk")
    _require(bool(str(outputs.get("motion") or "")) and motion_path.is_file(), "motion_missing",
             "manifest captured motion is missing on disk")
    try:
        plan, mot, cfg, meta = RCA.load_composition_inputs(plan_path, motion_path)
    except (RCA.CompositionAuditError, RCA.MotionPayloadError, RCA.PSA.PlanPayloadError) as exc:
        raise RealCompositionRenderError(getattr(exc, "code", "composition_inputs"), str(exc)) from exc
    _require(Path(meta["identity"]["manifest_path"]).resolve() == man_path, "manifest_mismatch",
             "the supplied manifest is not the sibling manifest recorded for the captured plan/motion")

    plan_meta, motion_meta = meta["plan"], meta["motion"]
    validate_timeline(plan_meta, motion_meta, source["decoded"])

    source_fps = _rate(source["decoded"]["fps"], "manifest source processing fps")
    out_fps = baseline["fps"]
    try:
        indices = RPR.source_indices(source_fps, out_fps, baseline["count"], source["decoded"]["frames"])
    except RPR.ReviewBuildError as exc:
        raise RealCompositionRenderError("cadence",
                                         f"cannot pair output frames to source frames: {exc}") from exc

    W, H = int(plan_meta["width"]), int(plan_meta["height"])
    protected = {man_path, source["path"], baseline["path"], plan_path, motion_path}
    video_path, json_path = _resolve_outputs(man_path, out_video, out_json, protected)

    highpass = bool(RCA.MODE_META[mode]["highpass"])
    smoothing_basis = RCA.MODE_META[mode]["smoothing_basis"]
    cells = CC.reference_cells(cfg)

    tmp_path = None
    encoder = None
    # Acquire the shared heavy lock before the precomputed smoothing chart and every source decode/render step,
    # so this candidate render excludes the anniversary repo's own heavy jobs; it is released on every path.
    if lock is not None:
        lock.acquire()
    try:
        smoothed = None
        if highpass:
            smoothed = CC.smoothed_delta(mot, plan, cfg, smoothing_basis=smoothing_basis)

        src_info = media.probe(source["path"])
        deinterlace = bool(source["preprocessing"].get("interlaced"))
        frames_iter, decoded_fps, decoded_size = media.decode(
            source["path"], src_info, source["window"]["start_s"], source["window"]["duration_s"],
            deinterlace=deinterlace, vf=source["preprocessing"].get("decode_vf"))
        _require(_close(decoded_fps, source["decoded"]["fps"]), "source_decode_fps",
                 f"decoded source fps {decoded_fps!r} disagrees with source.decoded fps "
                 f"{source['decoded']['fps']!r}")
        _require(tuple(decoded_size) == (source["decoded"]["width"], source["decoded"]["height"]),
                 "source_decode_dims", "decoded source dimensions disagree with the manifest")

        video_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{video_path.stem}.", suffix=".mp4", dir=video_path.parent)
        tmp_path = Path(tmp_name)
        os.close(fd)
        tmp_path.unlink()

        encoder = media.Encoder(tmp_path, W, H, out_fps, crf=crf, preset=preset)
        composed_map = forward_composed_map(plan, mot, indices)
        compare_positions = set(representative_frames(len(indices), ERROR_FRAME_COUNT))
        cmp_points = comparison_points(W, H)
        comparisons, raw_errors = [], []
        support_frames = []
        decoded_index, current, rendered = -1, None, 0
        for i, want in enumerate(indices):
            while decoded_index < want:
                nxt = next(frames_iter, None)
                _require(nxt is not None, "early_decode",
                         f"source decode ended before mapped frame {want} (output frame {i})")
                decoded_index += 1
                current = nxt
            frame = np.asarray(current)
            _require(frame.shape == (H, W, 3), "bad_source_frame",
                     f"decoded source frame {decoded_index} has shape {frame.shape}, expected {(H, W, 3)}")

            field = np.asarray(build_candidate_field(composed_map, plan, cfg, decoded_index, cells=cells,
                                                      highpass=highpass, smoothed=smoothed,
                                                      smoothing_basis=smoothing_basis))
            _require(field.shape == (H, W, 2), "bad_field_shape",
                     f"candidate field at source frame {decoded_index} has shape {field.shape}, expected "
                     f"{(H, W, 2)}")
            _require(bool(np.isfinite(field).all()), "nonfinite_field",
                     f"candidate field at source frame {decoded_index} contains non-finite values")
            if i in compare_positions:
                stats, err = dense_vs_exact_errors(
                    composed_map, mot, plan, cfg, decoded_index, points=cmp_points, cells=cells,
                    highpass=highpass, smoothed=smoothed, smoothing_basis=smoothing_basis)
                comparisons.append(stats)
                raw_errors.append(err)
            try:
                img, bad = render_candidate_frame(frame, field, cfg)
            except RealCompositionRenderError:
                raise
            except Exception as exc:  # the production sampler failed
                raise RealCompositionRenderError("render_failure",
                                                 f"render failed at source frame {decoded_index}: "
                                                 f"{type(exc).__name__}: {exc}") from exc
            support = support_frame_stats(bad, (H, W))
            support["frame"] = int(i)
            support["source_frame"] = int(decoded_index)
            support_frames.append(support)
            img = np.asarray(img)
            _require(img.shape == (H, W, 3), "bad_render_shape",
                     f"rendered frame at source frame {decoded_index} has shape {img.shape}, expected "
                     f"{(H, W, 3)}")
            _require(bool(np.isfinite(img).all()), "nonfinite_render",
                     f"rendered frame at source frame {decoded_index} contains non-finite values")
            try:
                encoder.write(img)
            except Exception as exc:
                raise RealCompositionRenderError("encode_failure",
                                                 f"encoder write failed at source frame {decoded_index}: "
                                                 f"{type(exc).__name__}: {exc}") from exc
            rendered += 1
            if lock is not None:
                # Before a chunk yield, wait for the writer thread to hand the finished chunk to ffmpeg so the
                # released shared lock is not still competing with our own queued encoding work.
                if at_lock_chunk_boundary(lock, rendered):
                    try:
                        encoder.drain()
                    except Exception as exc:
                        raise RealCompositionRenderError(
                            "encode_failure",
                            f"encoder drain failed before lock yield at output frame {rendered}: "
                            f"{type(exc).__name__}: {exc}") from exc
                lock.on_step(rendered)   # yield at chunk boundaries; no-op for chunk == 0

        extra = sum(1 for _ in frames_iter)
        decoded_total = decoded_index + 1 + extra
        _require(decoded_total == source["decoded"]["frames"], "extra_decode",
                 f"decoded source frame count {decoded_total} disagrees with source.decoded.frames "
                 f"{source['decoded']['frames']}")
        encoder.close()
        encoder = None

        probe = media.probe(tmp_path)
        _require((probe.get("width"), probe.get("height")) == (W, H), "candidate_probe",
                 "generated candidate dimensions disagree with the manifest")
        _require(_close(probe.get("fps"), float(out_fps)), "candidate_probe",
                 "generated candidate frame rate disagrees with the manifest")
        out_cfr = verify_cfr(media.frame_pts(tmp_path), baseline["count"], out_fps, "generated candidate")
        _require(out_cfr["ok"], "candidate_cfr",
                 "generated candidate CFR verification failed: " + "; ".join(out_cfr["reasons"]))
        size_bytes = int(tmp_path.stat().st_size)
        out_sha = MF.sha256_file(tmp_path)
        tmp_path.replace(video_path)
        tmp_path = None

        if raw_errors:
            all_errors = np.concatenate(raw_errors)
            error_aggregate = {"frames_compared": len(raw_errors), "points": int(all_errors.size),
                               "max_px": float(all_errors.max()), "mean_px": float(all_errors.mean()),
                               "p95_px": float(np.percentile(all_errors, 95))}
        else:
            error_aggregate = {"frames_compared": 0, "points": 0, "max_px": None, "mean_px": None,
                               "p95_px": None}

        support_aggregate = aggregate_support(support_frames, (H, W))

        report = {
            "schema": SCHEMA,
            "ok": True,
            "status": "success",
            "created": MF.now_iso(),
            "updated": MF.now_iso(),
            "scope": SCOPE_NOTE,
            "mode": mode,
            "mode_meta": dict(RCA.MODE_META[mode]),
            "chart": {"highpass": highpass, "smoothing_basis": smoothing_basis},
            "render_sampler": {
                "field_builder": "bench.real_composition_render.ForwardComposedMap (dense-grid interpolated "
                                 "forward composition)",
                "renderer": "mlxstab.stabilize.render_from_src",
                "kernel": cfg.kernel,
                "border": "zeros",
                "smoothed_precomputed_once": bool(smoothed is not None),
            },
            "invalid_support": {
                "definition": SUPPORT_DEFINITION,
                "metric": "center_out_of_bounds_invalid_fraction",
                "units": "fraction of output pixels",
                "strip_width_px": int(SUPPORT_STRIP_PX),
                "per_frame": support_frames,
                "aggregate": support_aggregate,
            },
            "approximation": {
                "name": APPROX_NAME,
                "description": APPROX_NOTE,
                "grid": APPROX_GRID,
                "grid_shape": [int(composed_map.grid_shape[0]), int(composed_map.grid_shape[1])],
                "grid_origin": [int(composed_map.origin[0]), int(composed_map.origin[1])],
                "padding": int(composed_map.guard_px),
                "query_bounds_px": {
                    "min_x": float(composed_map.query_min[0]),
                    "min_y": float(composed_map.query_min[1]),
                    "max_x": float(composed_map.query_max[0]),
                    "max_y": float(composed_map.query_max[1]),
                },
                "frame_shape": [int(H), int(W)],
                "interpolation": APPROX_INTERPOLATION,
                "reference_grid_cells": [int(cells[0]), int(cells[1])],
                "recurrence": ("phi_0 = pixel-center identity; psi_0 = I; "
                               "phi_{t+1} = composition_candidate.forward_apply(fit_t, phi_t); "
                               "psi_{t+1} = fit_t.H @ psi_t"),
                "complexity": "one forward_apply over the lattice per motion fit (O(T) total); the exact "
                              "full-history composition recomputes t fits per output frame (O(T^2))",
            },
            "source": {
                "manifest": {"path": str(man_path), "sha256": MF.sha256_file(man_path)},
                "run_id": manifest.get("run_id"),
                "case": (manifest.get("case") or {}).get("name"),
                "path": str(source["path"]), "sha256": source["sha256"],
                "window": source["window"], "preprocessing": source["preprocessing"],
                "decoded": source["decoded"],
            },
            "baseline": {
                "path": str(baseline["path"]), "sha256": baseline["sha256"],
                "width": baseline["width"], "height": baseline["height"],
                "fps": float(out_fps), "frames": baseline["count"], "cfr": baseline["cfr"],
            },
            "plan": {
                "path": str(plan_path), "sha256": MF.sha256_file(plan_path),
                "width": plan_meta["width"], "height": plan_meta["height"],
                "fps": plan_meta["fps"], "frames": plan_meta["frames"],
                "mesh_cells": plan_meta["mesh_cells"], "row_bands": plan_meta["row_bands"],
            },
            "motion": {
                "path": str(motion_path), "sha256": MF.sha256_file(motion_path),
                "schema": motion_meta["schema"], "width": motion_meta["width"],
                "height": motion_meta["height"], "fps": motion_meta["fps"],
                "n_frames": motion_meta["n_frames"], "n_fits": motion_meta["n_fits"],
                "stride": motion_meta["stride"],
            },
            "config": meta["config"],
            "pairing": {
                "rule": "exact nearest normalized presentation time; half-frame ties, repeats and "
                        "out-of-range indices are refused (bench.real_pair_review.source_indices)",
                "source_fps": float(source_fps), "output_fps": float(out_fps),
                "source_frames": source["decoded"]["frames"], "output_frames": baseline["count"],
                "rendered_source_frames": rendered, "decoded_source_frames": decoded_total,
                "frame_indices": indices, "unique": len(set(indices)) == len(indices),
            },
            "exact_comparison": {
                "method": "direct CC.candidate_source_coords on representative sparse points/frames",
                "point_grid": [int(COMPARISON_GRID[0]), int(COMPARISON_GRID[1])],
                "inset_px": COMPARISON_INSET_PX,
                "points_per_frame": int(cmp_points.shape[0]),
                "output_frame_positions": sorted(int(p) for p in compare_positions),
                "source_frames": [int(indices[p]) for p in sorted(compare_positions)],
                "per_frame": comparisons,
                "aggregate": error_aggregate,
            },
            "output": {
                "video": str(video_path), "report": str(json_path),
                "sha256": out_sha, "size_bytes": size_bytes,
                "width": W, "height": H, "fps": float(out_fps), "frames": baseline["count"],
                "crf": crf, "preset": preset, "cfr": out_cfr,
            },
            "no_network": True,
            "limitations": list(LIMITATIONS),
            "command": [sys.executable, *sys.argv],
        }
    except BaseException:
        if encoder is not None:
            try:
                encoder.close()
            except Exception:
                pass
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)
        video_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        raise
    finally:
        if lock is not None:
            lock.release()
    try:
        # Snapshot the lock only after release and write the report atomically; a report-building or write
        # failure here must not leave the finished video behind (same partial-output cleanup as a render
        # failure).
        report["lock"] = lock_snapshot(lock)
        MF.write_manifest(json_path, report)
    except BaseException:
        video_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        raise
    return report


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="bench/real_composition_render.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", help="successful bench/run_case manifest.json")
    p.add_argument("--mode", required=True, choices=list(SUPPORTED_MODES),
                   help="explicit composition candidate mode (no default is selected)")
    p.add_argument("--out-video", required=True, help="candidate MP4 (must not exist)")
    p.add_argument("--out-json", default=None,
                   help="adjacent atomic provenance JSON (default: <out-video stem>.json)")
    p.add_argument("--crf", type=int, default=DEFAULT_CRF, help="libx264 CRF for the candidate")
    p.add_argument("--preset", default=DEFAULT_PRESET, help="libx264 preset for the candidate")
    p.add_argument("--lock", choices=["auto", "on", "off"], default="auto",
                   help="shared heavy lock: auto=on for file-backed manifests (this renderer only accepts "
                        "local file sources), off disables it")
    p.add_argument("--lock-chunk", type=int, default=DEFAULT_LOCK_CHUNK,
                   help="release/re-acquire the shared lock every N output frames (default 30)")
    p.add_argument("--lock-pause", type=float, default=DEFAULT_LOCK_PAUSE_S,
                   help="seconds to yield between lock chunks (shared etiquette ~12 s)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    lock = None
    if args.lock != "off":
        # auto == on here: validate_source refuses anything but a local file-backed manifest, so every accepted
        # run is file-backed and must cooperate with the anniversary repo's shared heavy lock.
        lock = HeavyLock(what=f"mlxstab render {Path(args.manifest).parent.name}", enabled=True,
                         chunk=args.lock_chunk, pause_s=args.lock_pause)
    try:
        report = render_candidate(args.manifest, mode=args.mode, out_video=args.out_video,
                                  out_json=args.out_json, crf=args.crf, preset=args.preset, lock=lock)
    except Exception as exc:  # structured refusal, nonzero; no partial artifact is left behind
        code = getattr(exc, "code", type(exc).__name__)
        payload = {"schema": SCHEMA, "ok": False, "error": {"code": code, "message": str(exc)}}
        print(json.dumps(MF.json_safe(payload), sort_keys=True, allow_nan=False))
        return 2
    print(json.dumps({"schema": SCHEMA, "status": "success", "mode": report["mode"],
                      "video": report["output"]["video"], "report": report["output"]["report"],
                      "frames": report["output"]["frames"], "sha256": report["output"]["sha256"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
