"""Research-only rendered localized border-support candidate from a captured stabilization plan.

This is the **rendered** companion to the geometry-only localized-support idea in
``bench.local_support_candidate``. Given a captured Plan NPZ from an existing successful benchmark run (the
same strict, object-free payload ``bench.plan_shape_audit.load_plan_npz`` loads), it replays the run's source
window, deinterlace, output cadence and kernel **without rerunning RAFT** and renders one candidate MP4 plus
the matching validity mask:

- the successful ``bench/run_case`` manifest is validated (schema, ``status == "success"``), its recorded
  source hash is re-checked against the file on disk, and the captured plan's dimensions, frame count and fps
  must independently agree with ``source.decoded``; the resolved ``Config`` recovered from the manifest must
  agree with the captured plan's mesh/row shapes, so the composition rendered here is the one that was
  captured;
- each output frame is rendered from the production ``stabilize.source_coords`` /
  ``stabilize.render_from_src`` pipeline (no second warp pass, no re-fit). The **complete** output->source map
  (global homography + mesh + row corrections) and the **global-only** map (the same homography with the
  residual arrays zeroed) are both evaluated with the production sampler convention;
- :func:`bench.local_support_candidate.local_support_failure` finds the pixels where the complete map left the
  source but the global map is still supported (locally recoverable), and the **temporal union** of those
  masks over a centered ``[t - temporal_radius, t + temporal_radius]`` output-frame window is turned into a
  smooth output-space ``alpha`` taper with :func:`bench.local_support_candidate.local_support_alpha`. The two
  maps are blended with :func:`bench.local_support_candidate.blend_source_maps`; pixels the global motion also
  cannot recover are reported but never claimed fixed.

Streaming and bounded memory
----------------------------
Frames are decoded once and streamed in order. Because the alpha of an output frame needs its future temporal
neighbours, at most ``temporal_radius + 1`` frames' complete/global fields (and decoded frames) are buffered;
the mask history is bounded by ``bench.local_support_candidate.TemporalSupportWindow`` itself. The output is
encoded at the manifest's exact output dimensions, rational frame rate and frame count; after encoding the
generated video/mask clock and frame count are verified against the manifest and a mismatch is refused. Every
source decoder and output encoder is closed on every path (success or failure), and no partial output is left
behind.

Honesty scope
-------------
Research-only: no production code, default or output is changed, no mode is selected, and no network call is
made. The output is the rendered version of one geometry-only candidate for a single captured run/window; it
is not clip-wide or held-out evidence and is not by itself a quality claim. This is a geometry/support
diagnostic (center-out-of-bounds samples), **not** a perceptual metric, and it does not prevent foldover.

Run with::

    python -m bench.local_support_render --manifest out/bench/RUN/manifest.json \
        --output out/bench/RUN/local_support.mp4 --mask out/bench/RUN/local_support_mask.mp4
"""
from __future__ import annotations

import argparse
from collections import deque
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import mlx.core as mx

from mlxstab import media
from mlxstab import stabilize as S

from . import local_support_candidate as LSC
from . import manifest as MF
from . import plan_shape_audit as PSA
from . import real_pair_review as RPR
from .heavylock import HeavyLock

SCHEMA = "mlxstab.local_support_render/1"

DEFAULT_RADIUS_PX = 8.0
DEFAULT_TEMPORAL_RADIUS = 2
DEFAULT_CRF = 18
DEFAULT_PRESET = "medium"

#: Shared source-fps / plan-fps / output-fps agreement tolerance (mirrors the other bench renderers).
FPS_TOL = 1e-5
#: The bwdif filter a successful interlaced capture records; replayed verbatim for a file source.
DEINTERLACE_FILTER = "bwdif=mode=send_field:parity=auto:deint=all"

SUPPORT_DEFINITION = (
    "A source-map output pixel has full sampler support when every interpolation tap with absolute weight "
    "greater than 1e-6 lies inside the source image, using the configured kernel and the same tap convention as "
    "mlxstab.warpmetrics.full_support_mask. ``pre_invalid`` counts complete-map (global + mesh + rows) failures; "
    "``post_invalid`` counts blended-candidate failures; ``locally_unrecoverable`` counts pixels that the "
    "global-only map also cannot support, and these are never claimed fixed. ``affected`` is the fraction of "
    "output pixels with alpha < 1. The encoded validity mask separately follows ``warp.valid_mask`` center "
    "bounds. These are geometry/support diagnostics, NOT perceptual or visual-quality metrics."
)

CANDIDATE_NOTE = (
    "Blend the complete output->source map back toward the global-only map only near pixels the complete map "
    "loses but the global map still supports. alpha is a smooth output-space taper (0 at an unsupported pixel, "
    "1 at least radius_px away) over the temporal union of recoverable failures in a centered "
    "[t-temporal_radius, t+temporal_radius] output-frame window, so a transient invalid pixel does not switch "
    "the correction on and off on adjacent frames."
)

SCOPE_NOTE = (
    "Research-only rendered localized border-support candidate from a captured plan. Replays a successful "
    "bench/run_case source window/deinterlace/output cadence/kernel WITHOUT rerunning RAFT, evaluates the "
    "production source_coords (complete and global-only) and renders through the production render_from_src "
    "sampler. Writes one candidate MP4, one validity mask MP4 and one atomic JSON report; makes no network "
    "call. No mode is selected and no production code, default or output is changed."
)

LIMITATIONS = (
    "research-only rendered geometry candidate; not a production integration and not a production default",
    "renders the localized-support blend of the captured plan only; any visual result is scoped to that one "
    "run/window and is not clip-wide or held-out evidence",
    "full-kernel-support counts and alpha statistics are geometry/support diagnostics, not a "
    "perceptual or visual-quality metric, and the candidate does not prevent foldover",
    "the correction blends a locally-invalid complete map toward the global-only map with a spatial taper; it "
    "cannot recover a pixel the global model also leaves out of bounds, and those are reported separately",
    "alpha is evaluated on the temporal union over output frames, so the correction is temporally stable but "
    "is not a strict per-pixel guarantee",
    "no network call and no upload; a later review (bench.real_pair_review --candidate-video) happens "
    "separately",
)


class LocalSupportRenderError(ValueError):
    """The localized-support render refused its input or failed. ``code`` is a stable machine-readable reason."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = str(code)


def _require(cond, code, message):
    if not cond:
        raise LocalSupportRenderError(code, message)


def _close(a, b, tol=FPS_TOL):
    try:
        return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _rate(value, where):
    """Positive rational frame rate from a manifest value (``"30000/1001"`` or a float)."""
    try:
        result = Fraction(str(value)).limit_denominator(1_000_000)
    except (ValueError, ZeroDivisionError, TypeError):
        raise LocalSupportRenderError("bad_rate", f"{where} must be a positive finite frame rate") from None
    if result <= 0:
        raise LocalSupportRenderError("bad_rate", f"{where} must be a positive finite frame rate")
    return result


# --------------------------------------------------------------------------------------
# manifest / source validation and timeline
# --------------------------------------------------------------------------------------

def load_manifest(manifest_path):
    """Read a successful ``run_case`` manifest; return ``(manifest, resolved_path)``."""
    path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LocalSupportRenderError("manifest_unreadable",
                                      f"cannot read run manifest {str(path)!r}: {exc}") from exc
    _require(isinstance(manifest, dict), "manifest_malformed", "manifest top level must be a JSON object")
    _require(manifest.get("schema") == MF.MANIFEST_SCHEMA, "manifest_schema",
             f"manifest schema must be {MF.MANIFEST_SCHEMA!r}, got {manifest.get('schema')!r}")
    _require(manifest.get("status") == "success", "manifest_status",
             f"manifest status must be 'success', got {manifest.get('status')!r}")
    return manifest, path


def validate_source(manifest, source_override=None):
    """Validate the manifest source identity/hash/window/preprocessing/decoded metadata.

    ``source_override`` (the optional ``--source``) replaces the recorded path but must still hash to the
    manifest's ``source.sha256`` so the replayed evidence is the captured one."""
    source = manifest.get("source")
    _require(isinstance(source, dict), "manifest_malformed", "manifest source object is missing")
    raw = source_override if source_override is not None else source.get("path")
    _require(isinstance(raw, str) and raw.strip(), "source_missing", "manifest source path is missing")
    path = Path(raw).expanduser().resolve()
    _require(path.is_file(), "source_missing", f"source file is missing: {str(path)!r}")
    sha = source.get("sha256")
    _require(isinstance(sha, str) and sha, "source_sha", "manifest source.sha256 is missing")
    _require(MF.sha256_file(path) == sha, "source_sha",
             "source SHA-256 does not match the benchmark manifest")

    window = source.get("window")
    _require(isinstance(window, dict), "bad_source_window", "manifest source.window is missing")
    start, duration = window.get("start_s"), window.get("duration_s")
    _require(_close(start, start) and float(start) >= 0, "bad_source_window",
             "manifest source.window.start_s is invalid")
    _require(_close(duration, duration) and float(duration) > 0, "bad_source_window",
             "manifest source.window.duration_s is invalid")

    preprocessing = source.get("preprocessing")
    _require(isinstance(preprocessing, dict), "bad_preprocessing", "manifest source.preprocessing is missing")
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
    return {"path": path, "sha256": sha, "window": window, "preprocessing": preprocessing, "decoded": decoded}


def validate_timeline(plan_meta, decoded):
    """Require the captured plan to carry exactly the source timeline the manifest decoded."""
    _require((plan_meta["width"], plan_meta["height"]) == (decoded.get("width"), decoded.get("height")),
             "timeline_dimensions",
             f"plan is {plan_meta['width']}x{plan_meta['height']} but source.decoded is "
             f"{decoded.get('width')}x{decoded.get('height')}")
    _require(_close(plan_meta["fps"], decoded.get("fps")), "timeline_fps",
             f"plan fps {plan_meta['fps']!r} disagrees with source.decoded fps {decoded.get('fps')!r}")
    _require(plan_meta["frames"] == decoded.get("frames"), "timeline_frames",
             f"plan frames={plan_meta['frames']} disagrees with source.decoded.frames="
             f"{decoded.get('frames')}")


# --------------------------------------------------------------------------------------
# per-frame candidate helpers (pure NumPy; no MLX, no media)
# --------------------------------------------------------------------------------------

def blend_frame(full_src, global_src, alpha, width, height, kernel="lanczos3"):
    """Blend the complete/global maps for one frame and measure its support statistics.

    Returns ``(candidate [H,W,2] float32, stats)``. The stats count pre-blend (complete-map) invalid pixels,
    locally recoverable failures, locally unrecoverable failures, post-blend invalid pixels, the pixels where
    the local correction is active (``alpha < 1``) and the alpha summary. Support is the kernel-aware
    complete interpolation-kernel footprint (``mlxstab.warpmetrics.full_support_mask`` via
    :func:`bench.local_support_candidate.local_support_failure`), so the kernel must be the one the frame was
    rendered with and this stays a pure-NumPy seam over :mod:`bench.local_support_candidate`, unit-testable
    without MLX or media.
    """
    full = np.asarray(full_src, np.float32)
    glob = np.asarray(global_src, np.float32)
    weight = np.asarray(alpha, np.float32)
    recoverable, unrecoverable = LSC.local_support_failure(full, glob, width, height, kernel=kernel)
    candidate = LSC.blend_source_maps(full, glob, weight)
    post_bad = ~LSC._source_valid(candidate, width, height, kernel)
    pre_bad = ~LSC._source_valid(full, width, height, kernel)
    _require(np.array_equal(pre_bad, recoverable | unrecoverable), "support_accounting",
             "recoverable and unrecoverable counts do not partition complete-map support failures")
    affected = weight < 1.0
    total = int(full.shape[0] * full.shape[1])
    pre = int(np.count_nonzero(pre_bad))
    post = int(np.count_nonzero(post_bad))
    rec = int(np.count_nonzero(recoverable))
    unf = int(np.count_nonzero(unrecoverable))
    aff = int(np.count_nonzero(affected))
    frac = (lambda n: (float(n) / total) if total else 0.0)
    return candidate, {
        "frame_pixels": total,
        "pre_invalid_pixels": pre, "pre_invalid_fraction": frac(pre),
        "post_invalid_pixels": post, "post_invalid_fraction": frac(post),
        "recoverable_pixels": rec, "recoverable_fraction": frac(rec),
        "unrecoverable_pixels": unf, "unrecoverable_fraction": frac(unf),
        "repaired_pixels": int(np.count_nonzero(pre_bad & ~post_bad)),
        "regressed_pixels": int(np.count_nonzero(~pre_bad & post_bad)),
        "affected_pixels": aff, "affected_fraction": frac(aff),
        "alpha_mean": float(weight.mean()) if total else 0.0,
        "alpha_min": float(weight.min()) if total else 0.0,
        "alpha_max": float(weight.max()) if total else 0.0,
    }


def _fraction_block(per_frame, key, total_pixels):
    counts = [int(pf[f"{key}_pixels"]) for pf in per_frame]
    fracs = [float(pf[f"{key}_fraction"]) for pf in per_frame]
    total = int(sum(counts))
    frames = len(per_frame)
    return {
        "total_pixels": total,
        "pooled_fraction": (float(total) / total_pixels) if total_pixels else 0.0,
        "mean_fraction": (sum(fracs) / frames) if frames else 0.0,
        "max_fraction": (max(fracs) if fracs else 0.0),
    }


def aggregate_support(per_frame, width, height):
    """Aggregate per-frame :func:`blend_frame` statistics into the report's local-support section.

    Memory stays O(number of output frames): only small scalar dicts are retained, never a mask or field.
    Empty ``per_frame`` yields defined zero aggregates."""
    frames = len(per_frame)
    frame_pixels = int(width) * int(height)
    total_pixels = frames * frame_pixels
    alpha_means = [float(pf["alpha_mean"]) for pf in per_frame]
    render_center = (_fraction_block(per_frame, "render_center_invalid", total_pixels)
                     if frames and all("render_center_invalid_pixels" in pf for pf in per_frame) else None)
    return {
        "frames": frames,
        "frame_pixels": frame_pixels,
        "total_pixels": total_pixels,
        "pre_invalid": _fraction_block(per_frame, "pre_invalid", total_pixels),
        "post_invalid": _fraction_block(per_frame, "post_invalid", total_pixels),
        "recoverable": _fraction_block(per_frame, "recoverable", total_pixels),
        "locally_unrecoverable": _fraction_block(per_frame, "unrecoverable", total_pixels),
        "affected": _fraction_block(per_frame, "affected", total_pixels),
        "render_center_invalid": render_center,
        "repaired_pixels": int(sum(int(pf["repaired_pixels"]) for pf in per_frame)),
        "regressed_pixels": int(sum(int(pf["regressed_pixels"]) for pf in per_frame)),
        "alpha": {
            "mean": (sum(alpha_means) / frames) if frames else 0.0,
            "min": (min(float(pf["alpha_min"]) for pf in per_frame) if frames else 0.0),
            "max": (max(float(pf["alpha_max"]) for pf in per_frame) if frames else 0.0),
        },
    }


# --------------------------------------------------------------------------------------
# production-map evaluation / rendering
# --------------------------------------------------------------------------------------

def _field_np(plan, t, width, height, grid):
    """The production output->source field for one frame as a NumPy ``[H,W,2]`` float32 array."""
    src = S.source_coords(plan, t, width, height, grid)
    mx.eval(src)
    return np.array(src[0], np.float32)


def global_only_plan(plan):
    """A plan with the residual mesh/row corrections zeroed (same global homography and quads)."""
    return S.Plan(G=plan.G, mesh_corr=np.zeros_like(plan.mesh_corr), row_corr=np.zeros_like(plan.row_corr),
                  raw_quads=plan.raw_quads, path_quads=plan.path_quads, fallbacks=plan.fallbacks)


def render_field(frame, field, cfg):
    """Render a NumPy ``[H,W,2]`` output->source field with the production ``render_from_src`` sampler."""
    src = mx.array(np.asarray(field, np.float32))[None]
    img, bad = S.render_from_src(frame, src, cfg)
    return np.asarray(img), np.asarray(bad)


def _close_quietly(obj):
    """Best-effort close of a decode/sample iterator on a failure path (never masks the primary error)."""
    close = getattr(obj, "close", None)
    if callable(close):
        try:
            close()
        except BaseException:
            pass


def _temp_sibling(target):
    """A unique hidden sibling path of ``target`` for an encoder to publish into before the final replace."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=target.suffix or ".tmp", dir=target.parent)
    os.close(fd)
    Path(name).unlink()
    return Path(name)


def _resolve_outputs(man_path, source_path, plan_path, output, mask, report):
    """Resolve and validate the three output paths (distinct, non-input, non-existing)."""
    out_path = Path(output).resolve()
    mask_path = Path(mask).resolve()
    report_path = Path(report).resolve() if report else out_path.with_suffix(".json")
    _require(out_path != mask_path and out_path != report_path and mask_path != report_path,
             "output_collision", "output, mask and report paths must all differ")
    protected = {man_path, source_path, plan_path}
    for path in (out_path, mask_path, report_path):
        _require(path not in protected, "output_collision", "output collides with an input artifact")
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    return out_path, mask_path, report_path


# Path resolution: a relative recorded plan path is a sibling of the benchmark manifest (see
# ``bench.real_composition_render._resolve_input`` for the rationale).
def _resolve_input(base_dir, value):
    path = Path(str(value or "")).expanduser()
    return path if path.is_absolute() else (base_dir / path.name)


# --------------------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------------------

def render_local_support(manifest_path, *, source_path=None, output, mask, report=None,
                         radius_px=DEFAULT_RADIUS_PX, temporal_radius=DEFAULT_TEMPORAL_RADIUS,
                         crf=DEFAULT_CRF, preset=DEFAULT_PRESET, lock=None):
    """Render the localized-support candidate MP4 + mask MP4 from a successful captured run manifest."""
    radius_px = float(radius_px)
    _require(math.isfinite(radius_px) and radius_px > 0.0, "bad_radius",
             f"radius_px must be finite and > 0, got {radius_px!r}")
    _require(not isinstance(temporal_radius, bool) and isinstance(temporal_radius, (int, np.integer)),
             "bad_temporal_radius", "temporal_radius must be a non-negative integer")
    temporal_radius = int(temporal_radius)
    _require(temporal_radius >= 0, "bad_temporal_radius",
             f"temporal_radius must be >= 0, got {temporal_radius!r}")

    manifest, man_path = load_manifest(manifest_path)
    source = validate_source(manifest, source_path)
    outputs = manifest.get("outputs")
    _require(isinstance(outputs, dict), "manifest_malformed", "manifest outputs object is missing")

    plan_value = outputs.get("plan")
    _require(isinstance(plan_value, str) and plan_value.strip(), "plan_missing",
             "manifest outputs.plan is missing or empty; the run did not capture a plan")
    plan_path = _resolve_input(man_path.parent, plan_value).resolve()
    _require(plan_path.is_file(), "plan_missing", f"captured plan is missing: {str(plan_path)!r}")
    try:
        plan, plan_meta = PSA.load_plan_npz(plan_path)
    except PSA.PlanPayloadError as exc:
        raise LocalSupportRenderError(exc.code, str(exc)) from exc

    decoded = source["decoded"]
    validate_timeline(plan_meta, decoded)

    try:
        cfg = S.Config(**manifest["config"])
    except Exception as exc:
        raise LocalSupportRenderError("bad_config",
                                      f"cannot reconstruct Config from manifest config: "
                                      f"{type(exc).__name__}: {exc}") from exc
    gc, gr = int(cfg.mesh[0]), int(cfg.mesh[1])
    nr = max(int(cfg.rs_rows), 1)
    _require(plan_meta["mesh_cells"] == [gc, gr], "inconsistent_config",
             f"manifest config mesh={[gc, gr]} does not match the captured plan mesh cells "
             f"{plan_meta['mesh_cells']}")
    _require(plan_meta["row_bands"] == nr, "inconsistent_config",
             f"manifest config rs_rows={int(cfg.rs_rows)} implies {nr} row bands, but the captured plan "
             f"reports {plan_meta['row_bands']}")

    out_decoded = outputs.get("decoded")
    _require(isinstance(out_decoded, dict), "manifest_malformed", "manifest outputs.decoded is missing")
    out_fps = _rate(out_decoded.get("fps"), "manifest output fps")
    out_count = out_decoded.get("frames")
    _require(isinstance(out_count, int) and not isinstance(out_count, bool) and out_count >= 1,
             "bad_output_frames", "manifest outputs.decoded.frames must be a positive integer")
    if outputs.get("encoded_frames") is not None:
        _require(outputs["encoded_frames"] == out_count, "bad_output_frames",
                 "manifest outputs.encoded_frames disagrees with outputs.decoded.frames")

    source_fps = _rate(decoded["fps"], "manifest source processing fps")
    try:
        indices = RPR.source_indices(source_fps, out_fps, out_count, int(decoded["frames"]))
    except RPR.ReviewBuildError as exc:
        raise LocalSupportRenderError("cadence",
                                      f"cannot pair output frames to source frames: {exc}") from exc

    W, H = int(plan_meta["width"]), int(plan_meta["height"])
    out_path, mask_path, report_path = _resolve_outputs(man_path, source["path"], plan_path,
                                                        output, mask, report)

    decoded_iter = None
    video_encoder = mask_encoder = None
    video_tmp = _temp_sibling(out_path)
    mask_tmp = _temp_sibling(mask_path)
    try:
        src_info = media.probe(source["path"])
        deinterlace = bool(source["preprocessing"].get("interlaced"))
        decoded_iter, decoded_fps, decoded_size = media.decode(
            source["path"], src_info, source["window"]["start_s"], source["window"]["duration_s"],
            deinterlace=deinterlace, vf=source["preprocessing"].get("decode_vf"))
        _require(_close(decoded_fps, float(source_fps)), "source_decode_fps",
                 f"decoded source fps {decoded_fps!r} disagrees with source.decoded fps "
                 f"{decoded['fps']!r}")
        _require(tuple(decoded_size) == (W, H), "source_decode_dims",
                 f"decoded source dimensions {tuple(decoded_size)!r} disagree with the captured plan "
                 f"{(W, H)!r}")

        video_encoder = media.Encoder(video_tmp, W, H, out_fps, crf=crf, preset=preset)
        mask_encoder = media.Encoder(mask_tmp, W, H, out_fps, gray=True)

        ys, xs = mx.meshgrid(mx.arange(H), mx.arange(W), indexing="ij")
        grid = (xs.astype(mx.float32), ys.astype(mx.float32))
        gplan = global_only_plan(plan)
        window = LSC.TemporalSupportWindow(temporal_radius)
        buffer = deque()
        per_frame = []
        rendered = 0
        decoded_index = -1
        current = None

        def process(union):
            nonlocal rendered
            entry = buffer.popleft()
            alpha = LSC.local_support_alpha(union, radius_px)
            candidate, stats = blend_frame(entry["full"], entry["global"], alpha, W, H, kernel=cfg.kernel)
            stats["frame"] = int(entry["frame"])
            stats["source_frame"] = int(entry["plan_frame"])
            try:
                img, bad = render_field(entry["frame_img"], candidate, cfg)
            except LocalSupportRenderError:
                raise
            except Exception as exc:
                raise LocalSupportRenderError("render_failure",
                                              f"render failed at source frame {entry['plan_frame']}: "
                                              f"{type(exc).__name__}: {exc}") from exc
            _require(img.shape == (H, W, 3), "bad_render_shape",
                     f"rendered frame at source frame {entry['plan_frame']} has shape {img.shape}, "
                     f"expected {(H, W, 3)}")
            center_invalid = int(np.count_nonzero(bad))
            stats["render_center_invalid_pixels"] = center_invalid
            stats["render_center_invalid_fraction"] = center_invalid / float(W * H)
            try:
                video_encoder.write(img)
                mask_encoder.write(bad.astype(np.uint8) * 255)
            except Exception as exc:
                raise LocalSupportRenderError("encode_failure",
                                              f"encoder write failed at source frame "
                                              f"{entry['plan_frame']}: {type(exc).__name__}: {exc}") from exc
            per_frame.append(stats)
            rendered += 1
            if rendered % 30 == 0 or rendered == out_count:
                print(f"rendered {rendered}/{out_count} output frames", file=sys.stderr, flush=True)

        for i, t in enumerate(indices):
            if lock is not None:
                lock.on_step(i)
            while decoded_index < t:
                nxt = next(decoded_iter, None)
                _require(nxt is not None, "early_decode",
                         f"source decode ended before mapped frame {t} (output frame {i})")
                decoded_index += 1
                current = nxt
            frame = np.asarray(current)
            _require(frame.shape == (H, W, 3), "bad_source_frame",
                     f"decoded source frame {decoded_index} has shape {frame.shape}, expected {(H, W, 3)}")
            full = _field_np(plan, t, W, H, grid)
            glob = _field_np(gplan, t, W, H, grid)
            _require(bool(np.isfinite(full).all()) and bool(np.isfinite(glob).all()), "nonfinite_field",
                     f"source field at plan frame {t} contains non-finite values")
            recoverable, _unrecoverable = LSC.local_support_failure(full, glob, W, H, kernel=cfg.kernel)
            buffer.append({"frame": i, "plan_frame": t, "frame_img": frame, "full": full, "global": glob})
            emitted = window.push(recoverable)
            if emitted is not None:
                process(emitted[1])
        for _oldest, union in window.flush():
            process(union)

        extra = sum(1 for _ in decoded_iter)
        decoded_total = decoded_index + 1 + extra
        _require(decoded_total == int(decoded["frames"]), "extra_decode",
                 f"decoded source frame count {decoded_total} disagrees with source.decoded.frames "
                 f"{decoded['frames']}")
        _require(rendered == out_count, "frame_count",
                 f"rendered {rendered} output frames but outputs.decoded.frames is {out_count}")
        _require(not buffer, "internal_error", "temporal buffer was not drained")

        media.close_all((video_encoder, mask_encoder))
        video_encoder = mask_encoder = None

        probe = media.probe(video_tmp)
        _require((probe.get("width"), probe.get("height")) == (W, H), "output_probe",
                 "generated candidate dimensions disagree with the captured plan")
        _require(_close(probe.get("fps"), float(out_fps)), "output_probe",
                 "generated candidate frame rate disagrees with the manifest")
        video_pts = media.frame_pts(video_tmp)
        mask_pts = media.frame_pts(mask_tmp)
        _require(video_pts["count"] == out_count and mask_pts["count"] == out_count, "output_frames",
                 f"generated candidate has {video_pts['count']} video / {mask_pts['count']} mask frames, "
                 f"expected {out_count}")
        cfr = MF.check_encoded_cfr(video_pts, mask_pts, float(out_fps), float(out_fps),
                                   requested_fps=float(out_fps))
        _require(cfr["aligned"], "output_cfr",
                 "generated candidate CFR verification failed: " + "; ".join(cfr["reasons"]))
        out_sha = MF.sha256_file(video_tmp)
        out_size = int(video_tmp.stat().st_size)
        mask_sha = MF.sha256_file(mask_tmp)

        report_payload = {
            "schema": SCHEMA,
            "ok": True,
            "status": "success",
            "created": MF.now_iso(),
            "updated": MF.now_iso(),
            "scope": SCOPE_NOTE,
            "radius_px": radius_px,
            "temporal_radius": temporal_radius,
            "candidate": {"definition": CANDIDATE_NOTE,
                          "functions": ["bench.local_support_candidate.local_support_failure",
                                        "bench.local_support_candidate.local_support_alpha",
                                        "bench.local_support_candidate.blend_source_maps",
                                        "bench.local_support_candidate.TemporalSupportWindow"],
                          "buffer_frames": temporal_radius + 1},
            "alignment": {
                "rule": "exact nearest normalized presentation time; half-frame ties, repeats and "
                        "out-of-range indices are refused (bench.real_pair_review.source_indices)",
                "source_fps": float(source_fps), "output_fps": float(out_fps),
                "source_window": source["window"],
                "deinterlace": bool(source["preprocessing"].get("interlaced")),
                "deinterlace_filter": source["preprocessing"].get("deinterlace"),
                "decode_vf": source["preprocessing"].get("decode_vf"),
                "kernel": cfg.kernel,
                "source_frames": int(decoded["frames"]), "plan_frames": int(plan_meta["frames"]),
                "output_frames": int(out_count),
                "frame_indices": indices,
                "unique_source_frames": len(set(indices)) == len(indices),
            },
            "render_sampler": {"map": "mlxstab.stabilize.source_coords",
                               "renderer": "mlxstab.stabilize.render_from_src",
                               "kernel": cfg.kernel, "border": "zeros"},
            "source": {
                "manifest": {"path": str(man_path), "sha256": MF.sha256_file(man_path)},
                "run_id": manifest.get("run_id"),
                "case": (manifest.get("case") or {}).get("name"),
                "path": str(source["path"]), "sha256": source["sha256"],
                "window": source["window"], "preprocessing": source["preprocessing"], "decoded": decoded,
            },
            "plan": {
                "path": str(plan_path), "sha256": MF.sha256_file(plan_path),
                "schema": plan_meta["schema"], "width": plan_meta["width"], "height": plan_meta["height"],
                "fps": plan_meta["fps"], "frames": plan_meta["frames"],
                "mesh_cells": plan_meta["mesh_cells"], "row_bands": plan_meta["row_bands"],
            },
            "config": MF.json_safe(manifest.get("config")),
            "local_support": {
                "definition": SUPPORT_DEFINITION,
                "metric": "full_kernel_support_invalid_fraction",
                "units": "fraction of output pixels",
                "per_frame": per_frame,
                "aggregate": aggregate_support(per_frame, W, H),
            },
            "output": {
                "video": str(out_path), "mask": str(mask_path), "report": str(report_path),
                "sha256": out_sha, "mask_sha256": mask_sha, "size_bytes": out_size,
                "width": W, "height": H, "fps": float(out_fps), "frames": int(out_count),
                "cfr": cfr,
            },
            "lock": ({**lock.status(), "held": False} if lock is not None else
                     {"backend": "disabled", "enabled": False, "held": False}),
            "no_network": True,
            "limitations": list(LIMITATIONS),
            "command": [sys.executable, *sys.argv],
        }

        # Publish only after the encoded clocks were verified; a failure here removes the published files.
        video_tmp.replace(out_path)
        video_tmp = None
        mask_tmp.replace(mask_path)
        mask_tmp = None
        MF.write_manifest(report_path, report_payload)
    except BaseException:
        for encoder in (video_encoder, mask_encoder):
            if encoder is not None:
                try:
                    encoder.close()
                except BaseException:
                    pass
        if decoded_iter is not None:
            _close_quietly(decoded_iter)
        for path in (video_tmp, mask_tmp):
            if path is not None:
                Path(path).unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
        mask_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        raise
    return report_payload


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="bench/local_support_render.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="successful bench/run_case manifest.json")
    p.add_argument("--source", default=None,
                   help="source video path (default: the manifest's source.path); must still hash to the "
                        "manifest source.sha256")
    p.add_argument("--output", required=True, help="candidate MP4 (must not exist)")
    p.add_argument("--mask", required=True, help="validity mask MP4 (must not exist)")
    p.add_argument("--report", default=None,
                   help="atomic JSON report (default: <output stem>.json)")
    p.add_argument("--radius-px", type=float, default=DEFAULT_RADIUS_PX,
                   help="output-space taper radius in pixels (default: %(default)s)")
    p.add_argument("--temporal-radius", type=int, default=DEFAULT_TEMPORAL_RADIUS,
                   help="centered output-frame support window radius (default: %(default)s)")
    p.add_argument("--crf", type=int, default=DEFAULT_CRF, help="libx264 CRF for the candidate video")
    p.add_argument("--preset", default=DEFAULT_PRESET, help="libx264 preset for the candidate video")
    p.add_argument("--lock", choices=["auto", "on", "off"], default="auto",
                   help="shared heavy lock (auto is on for this file-backed renderer)")
    p.add_argument("--lock-chunk", type=int, default=30,
                   help="release/re-acquire the shared lock every N output frames (default 30)")
    p.add_argument("--lock-pause", type=float, default=12.0,
                   help="seconds to yield between shared-lock chunks (default 12)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    lock = None
    if args.lock != "off":
        lock = HeavyLock(what=f"mlxstab local-support render {Path(args.manifest).parent.name}",
                         enabled=True, chunk=args.lock_chunk, pause_s=args.lock_pause)
    try:
        if lock is not None:
            lock.acquire()
        report = render_local_support(args.manifest, source_path=args.source, output=args.output,
                                      mask=args.mask, report=args.report, radius_px=args.radius_px,
                                      temporal_radius=args.temporal_radius, crf=args.crf, preset=args.preset,
                                      lock=lock)
    except Exception as exc:  # structured refusal, nonzero; no partial artifact is left behind
        code = getattr(exc, "code", type(exc).__name__)
        payload = {"schema": SCHEMA, "ok": False, "error": {"code": code, "message": str(exc)}}
        print(json.dumps(MF.json_safe(payload), sort_keys=True, allow_nan=False))
        return 2
    finally:
        if lock is not None:
            lock.release()
    print(json.dumps({"schema": SCHEMA, "status": "success", "video": report["output"]["video"],
                      "mask": report["output"]["mask"], "report": report["output"]["report"],
                      "frames": report["output"]["frames"], "sha256": report["output"]["sha256"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
