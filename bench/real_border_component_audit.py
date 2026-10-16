"""Read-only CPU/NumPy component attribution of the production border composition.

Motivation
----------
The production output->source map composes three pieces (``mlxstab.stabilize.source_coords`` /
``_source_coords_ts_scales_np``): a global homography ``G``, a mesh residual correction and a rolling-shutter
row correction. When a captured run leaves pixels without a source, it is useful to know *which* component is
responsible. This module answers that offline, reproducibly, for one successful captured benchmark run.

Method
------
1. Load the strict, object-free Motion (``bench.real_composition_audit.load_motion_npz``) and the resolved
   ``config`` from a successful ``run_case`` manifest, and reconstruct the production ``Plan`` with
   ``mlxstab.stabilize.plan`` (deterministic; no fitting, no flow).
2. Decode ``mask_r0.mp4`` and reconstruct the full-plan invalid mask with the production NumPy sampler
   (``mlxstab.stabilize._source_coords_ts_scales_np`` at scale 1.0 + ``_valid_mask_np``) on every encoded
   output frame. The reconstruction must be **pixel-exact** against the encoded mask before any ablation is
   reported; a mismatch stops the comparison instead of reporting numbers from a reconstruction that does not
   match production.
3. Build three ablation plans from the reconstructed plan -- ``global_only`` (mesh and rows zeroed),
   ``global_mesh`` (rows zeroed) and ``global_rows`` (mesh zeroed) -- and measure every plan's full-resolution
   and bottom-N-row invalid counts/fractions, per output frame and in aggregate.
4. Report each ablation's per-frame and aggregate delta against ``global_only``, the sign counts of those
   deltas, and the non-additive interaction ``full - global_mesh - global_rows + global_only``.

The composition is sequential (the mesh correction is sampled at the global-mapped coordinate and the row
correction at the mesh-shifted ``y``), so the component deltas are **not** independent; the interaction term
reports exactly that non-additivity.

Guarantees / scope
------------------
Read-only and CPU/NumPy only: no source footage is decoded, no MLX graph is constructed or evaluated, nothing
is rendered or encoded, the heavy lock is never acquired, and no production default or artifact is modified.
The encoded validity mask is decoded with FFmpeg solely to validate the reconstructed full-composition mask.
The report is path-free and strictly finite, written atomically through :mod:`bench.manifest`. This is a
geometry/coverage diagnostic, not a perceptual, quality or distortion metric, and it does not prevent foldover.

Run with::

    python -m bench.real_border_component_audit MANIFEST.json --out REPORT.json [--bottom-rows N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from . import manifest as MF
from .real_composition_audit import load_motion_npz
from mlxstab import media
from mlxstab import stabilize as S

SCHEMA = "mlxstab.real_border_component_audit/1"
DEFAULT_MOTION_NAME = "motion_r0.npz"
DEFAULT_MASK_NAME = "mask_r0.mp4"
DEFAULT_BOTTOM_ROWS = 4
# Bounded temporal chunk so the NumPy [frames, 1, H, W, 2] source fields stay small (a 1080p plan frame is
# ~16.6 MB); four frames keeps the sampler/intermediate temporaries well under a few hundred MB.
MEASURE_CHUNK = 4
# Fps agreement tolerance, matching ``bench.manifest.check_encoded_cfr``'s 1e-6 comparison.
FPS_AGREEMENT_TOL = 1e-6
# Absolute tolerance for the recomputed float statistics (fractions and means) in the report validator. The
# values are float64 ratios of pixel counts, so 1e-9 dwarfs their rounding error while still catching any
# deliberate tampering; integer counts and pixel denominators are compared exactly.
STAT_TOL = 1e-9

#: The four plans compared. ``full`` is the reconstructed production composition; the other three zero one or
#: both residual correction arrays.
COMPONENTS = ("full", "global_only", "global_mesh", "global_rows")
COMPONENT_CONFIGS = {
    "full": "G + mesh + rows (production composition)",
    "global_only": "G; mesh and rows zero",
    "global_mesh": "G + mesh; rows zero",
    "global_rows": "G + rows; mesh zero",
}
#: Component deltas against ``global_only``. ``interaction`` is the non-additivity of the sequential
#: composition; every value is a per-frame invalid-pixel count difference before being summed/averaged.
DELTA_FORMULAS = {
    "mesh": "global_mesh - global_only (mesh contribution with rows zero)",
    "rows": "global_rows - global_only (rows contribution with mesh zero)",
    "interaction": "full - global_mesh - global_rows + global_only (= combined - mesh - rows)",
    "combined": "full - global_only (mesh and rows together)",
}
DELTA_NAMES = ("mesh", "rows", "interaction", "combined")

SCOPE_NOTE = (
    "Read-only CPU/NumPy geometry/coverage attribution of the production output->source map: no source "
    "footage decoded, no MLX/GPU flow, no render/encode, no heavy lock. The encoded validity mask is decoded "
    "only to validate the reconstructed full-composition mask. Component ablation zeroes one residual array "
    "while the production sampler still samples the other at its composed coordinate, so per-component deltas "
    "are not independent and the interaction term reports that non-additivity. NOT a perceptual, quality, "
    "distortion or foldover metric."
)


class BorderComponentAuditError(ValueError):
    """The audit refused its input or produced an inconsistent report. ``code`` is a stable reason."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = str(code)


def _require(cond, code, message):
    if not cond:
        raise BorderComponentAuditError(code, message)


def _require_positive_int(value, code, message):
    """Refuse anything that is not a true positive integer (rejects bools, floats, strings, <= 0)."""
    _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value >= 1,
             code, message)
    return int(value)


# --------------------------------------------------------------------------------------
# manifest / artifact identity
# --------------------------------------------------------------------------------------

def _manifest_identity(man, *, motion_name, mask_name):
    """Extract the path-free identity a component report needs, refusing a non-success or shapeless manifest."""
    _require(isinstance(man, dict), "manifest_malformed", "manifest top level must be a JSON object")
    _require(man.get("status") == "success", "not_success",
             f"manifest status is {man.get('status')!r}; only a successful capture is audited")
    source = man.get("source")
    _require(isinstance(source, dict) and isinstance(source.get("sha256"), str) and source["sha256"],
             "manifest_malformed", "manifest is missing source.sha256")
    sha = source["sha256"]
    _require(len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "bad_source_sha",
             f"source.sha256 must be 64 lowercase hex characters, got {sha!r}")
    window = source.get("window")
    _require(isinstance(window, dict) and "start_s" in window and "duration_s" in window,
             "manifest_malformed", "manifest is missing source.window.start_s/duration_s")
    weights = man.get("weights")
    _require(isinstance(weights, dict) and isinstance(weights.get("sha256"), str) and weights["sha256"],
             "manifest_malformed", "manifest is missing weights.sha256")
    config = man.get("config")
    _require(isinstance(config, dict) and config, "manifest_malformed",
             "manifest is missing the resolved config object needed to reconstruct the Plan")
    decoded = source.get("decoded")
    _require(isinstance(decoded, dict) and "fps" in decoded and "frames" in decoded,
             "manifest_malformed", "manifest is missing source.decoded.fps/frames (analysis cadence)")
    outputs = man.get("outputs")
    _require(isinstance(outputs, dict), "manifest_malformed", "manifest is missing outputs")
    out_decoded = outputs.get("decoded")
    _require(isinstance(out_decoded, dict) and "fps" in out_decoded and "frames" in out_decoded,
             "manifest_malformed", "manifest is missing outputs.decoded.fps/frames (output cadence)")
    case = man.get("case") if isinstance(man.get("case"), dict) else {}
    # Convert here so a non-numeric cadence is a stable malformed-manifest refusal rather than a bare
    # ``TypeError``/``ValueError``; finiteness/positivity is validated in ``run_audit`` as ``bad_cadence``.
    try:
        analysis_fps = float(decoded["fps"])
        output_fps = float(out_decoded["fps"])
    except (TypeError, ValueError):
        raise BorderComponentAuditError(
            "manifest_malformed", "source.decoded.fps and outputs.decoded.fps must be numeric")
    # Frame counts and dimensions must be genuine positive integers: ``int()`` would silently truncate a float
    # such as 2.5 and accept ``True`` as 1, hiding a mis-sized capture.
    analysis_frames = _require_positive_int(
        decoded["frames"], "bad_frame_count",
        f"source.decoded.frames must be a positive integer, got {decoded['frames']!r}")
    output_frames = _require_positive_int(
        out_decoded["frames"], "bad_frame_count",
        f"outputs.decoded.frames must be a positive integer, got {out_decoded['frames']!r}")
    source_width = _require_positive_int(
        decoded.get("width"), "bad_dimensions",
        f"source.decoded.width must be a positive integer, got {decoded.get('width')!r}")
    source_height = _require_positive_int(
        decoded.get("height"), "bad_dimensions",
        f"source.decoded.height must be a positive integer, got {decoded.get('height')!r}")
    return {
        "run_id": man.get("run_id"),
        "manifest_schema": man.get("schema"),
        "case": {"name": case.get("name"), "description": case.get("description"),
                 "output_fps": case.get("output_fps")},
        "source_sha256": sha,
        "source_window": {"start_s": window.get("start_s"), "duration_s": window.get("duration_s")},
        "source_preprocessing": MF.json_safe(source.get("preprocessing")),
        "weights_sha256": weights["sha256"],
        "config": MF.json_safe(config),
        "config_digest": man.get("config_digest"),
        "content_key": man.get("content_key"),
        "analysis_fps": analysis_fps,
        "analysis_frames": analysis_frames,
        "output_fps": output_fps,
        "output_frames": output_frames,
        "source_decoded_width": source_width,
        "source_decoded_height": source_height,
        "artifacts": {"motion": motion_name, "mask": mask_name},
        "manifest_outputs_plan_is_null": outputs.get("plan") is None,
    }


def _motion_identity(meta):
    """The path-free slice of ``load_motion_npz`` metadata (drop the private ``path``)."""
    return {
        "schema": meta.get("schema"),
        "width": meta.get("width"), "height": meta.get("height"),
        "n_frames": meta.get("n_frames"), "fps": meta.get("fps"), "stride": meta.get("stride"),
        "n_fits": meta.get("n_fits"), "mesh_cells": meta.get("mesh_cells"),
        "row_bands": meta.get("row_bands"), "fallback_fits": meta.get("fallback_fits"),
    }


def _require_artifact_fps(name, value):
    """Finite-positive fps for a loaded artifact (``motion``/``mask``), else a stable ``bad_cadence`` refusal."""
    try:
        fps = float(value)
    except (TypeError, ValueError):
        raise BorderComponentAuditError("bad_cadence", f"{name} fps must be numeric, got {value!r}")
    _require(np.isfinite(fps) and fps > 0.0, "bad_cadence", f"{name} fps must be finite and > 0, got {fps!r}")
    return fps


# --------------------------------------------------------------------------------------
# encoded mask decode + full-resolution NumPy reconstruction
# --------------------------------------------------------------------------------------

def _close_quietly(obj):
    """Best-effort close of a decode/sample iterator; used on every streaming exit path.

    Real :func:`mlxstab.media.decode` generators have a ``close``; test fakes may be plain iterators and
    are simply left to the garbage collector. Closing can only fail while already unwinding an error, so a
    secondary close failure is swallowed rather than masking the primary exception."""
    close = getattr(obj, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _iter_decode_mask_bool(path):
    """Streaming decode of the encoded gray validity mask to boolean ``[H,W]`` arrays (True = no source).

    Returns ``(masks, meta)`` where ``masks`` is a lazy iterator and ``meta`` is a mutable dict with the
    static ``width``/``height``/``fps`` plus a ``frames`` count that is incremented as frames are consumed.
    The underlying FFmpeg generator is closed when ``masks`` is closed or fully exhausted."""
    info = media.probe(str(path))
    gen, fps, (W, H) = media.decode(str(path), info)
    meta = {"width": int(W), "height": int(H), "fps": float(fps), "frames": 0}

    def masks():
        try:
            for f in gen:
                meta["frames"] += 1
                yield f[..., 0] > 127
        finally:
            _close_quietly(gen)

    return masks(), meta


def _decode_mask_bool(path):
    """Materialize :func:`_iter_decode_mask_bool` to a list of boolean ``[H,W]`` arrays (True = no source).

    Kept for callers/tests that want the whole mask; :func:`run_audit` uses the streaming iterator."""
    it, meta = _iter_decode_mask_bool(path)
    frames = list(it)
    return frames, {"width": meta["width"], "height": meta["height"], "fps": meta["fps"],
                    "frames": len(frames)}


def _iter_reconstructed_masks(pl, ts, W, H, grid, chunk=MEASURE_CHUNK):
    """Lazily yield NumPy invalid masks (True = out of source) for plan frames ``ts`` at full resolution.

    Uses the production-convention NumPy probe ``_source_coords_ts_scales_np`` at scale 1.0 and the local
    ``_valid_mask_np``; the yielded masks match the encoded ``mask_r0.mp4`` classification (verified). Only
    one bounded ``chunk`` of frames is held at a time, so aggregates never retain the whole sequence."""
    xx, yy = grid
    ts = list(ts)
    for i in range(0, len(ts), chunk):
        tsc = list(ts[i:i + chunk])
        src = S._source_coords_ts_scales_np(pl, tsc, W, H, [1.0], (xx, yy))[:, 0]   # [n,H,W,2]
        bad = ~S._valid_mask_np(src, H, W)                                          # [n,H,W]
        for k in range(len(tsc)):
            yield bad[k]


def _reconstructed_masks(pl, ts, W, H, grid, chunk=MEASURE_CHUNK):
    """Materialize :func:`_iter_reconstructed_masks`; kept for callers/tests that want the whole list."""
    return list(_iter_reconstructed_masks(pl, ts, W, H, grid, chunk))


def _frame_measure(mask, output_frame, plan_frame, bottom_rows):
    """One frame's invalid counts/fractions over every pixel and over the bottom ``bottom_rows`` rows."""
    H, W = mask.shape
    s = min(int(bottom_rows), H)
    n = int(mask.size)
    nb = int(np.count_nonzero(mask))
    bottom = mask[H - s:, :] if s > 0 else mask[0:0]
    bn = int(bottom.size)
    bb = int(np.count_nonzero(bottom))
    return {
        "output_frame": int(output_frame),
        "plan_frame": int(plan_frame),
        "invalid_pixels": nb, "frame_pixels": n, "invalid_fraction": (nb / n) if n else 0.0,
        "bottom_invalid_pixels": bb, "bottom_pixels": bn,
        "bottom_invalid_fraction": (bb / bn) if bn else 0.0,
    }


def _aggregate_measures(per_frame, H, W, bottom_rows):
    """Aggregate one component's per-frame counts/fractions (same numbers as the materialized version)."""
    s = min(int(bottom_rows), H) if H else 0
    full_px = sum(r["frame_pixels"] for r in per_frame)
    full_bad = sum(r["invalid_pixels"] for r in per_frame)
    bot_px = sum(r["bottom_pixels"] for r in per_frame)
    bot_bad = sum(r["bottom_invalid_pixels"] for r in per_frame)
    full_fractions = [r["invalid_fraction"] for r in per_frame]
    bot_fractions = [r["bottom_invalid_fraction"] for r in per_frame]
    return {
        "per_frame": per_frame,
        "full_resolution": {
            "frames": len(per_frame), "frame_pixels": int(full_px // max(len(per_frame), 1)),
            "total_pixels": int(full_px), "total_invalid_pixels": int(full_bad),
            "pooled_fraction": (full_bad / full_px) if full_px else 0.0,
            "mean_fraction": (sum(full_fractions) / len(full_fractions)) if full_fractions else 0.0,
            "max_fraction": max(full_fractions) if full_fractions else 0.0,
        },
        "bottom_rows": {
            "rows": int(s), "per_frame_pixels": int(bot_px // max(len(per_frame), 1)),
            "total_pixels": int(bot_px), "total_invalid_pixels": int(bot_bad),
            "pooled_fraction": (bot_bad / bot_px) if bot_px else 0.0,
            "mean_fraction": (sum(bot_fractions) / len(bot_fractions)) if bot_fractions else 0.0,
            "max_fraction": max(bot_fractions) if bot_fractions else 0.0,
        },
    }


#: Unique marker used to detect an over-long streaming mask source without relying on truthiness.
_SENTINEL = object()


def _measure_masks(masks, plan_indices, bottom_rows):
    """Per-frame and aggregate invalid fractions over every pixel and over the bottom ``bottom_rows`` rows.

    ``masks`` may be a bounded streaming iterator; only per-frame counts/fractions are retained, never the
    arrays. The mask iterator and ``plan_indices`` are consumed in lockstep and must contain exactly the same
    number of frames: an early EOF (short stream) or any extra mask frame is refused with
    ``frame_count_mismatch`` rather than silently truncated by ``zip``. The iterator is closed on every exit
    (success, refusal or any other exception)."""
    per_frame = []
    H = W = 0
    it = iter(masks)
    try:
        for j, t in enumerate(plan_indices):
            try:
                mask = next(it)
            except StopIteration:
                raise BorderComponentAuditError(
                    "frame_count_mismatch",
                    f"mask stream ended after {j} frames but {len(plan_indices)} plan output frames were expected")
            if not per_frame:
                H, W = mask.shape
            per_frame.append(_frame_measure(mask, len(per_frame), t, bottom_rows))
        if next(it, _SENTINEL) is not _SENTINEL:
            raise BorderComponentAuditError(
                "frame_count_mismatch",
                f"mask stream has more than the {len(plan_indices)} plan output frames")
    finally:
        _close_quietly(it)
    return _aggregate_measures(per_frame, H, W, bottom_rows)


def _verify_masks(reconstructed, encoded, plan_indices):
    """Exact comparison of the reconstructed and encoded invalid masks on every output frame."""
    diffs = []
    for mask, enc, t in zip(reconstructed, encoded, plan_indices):
        n = int(np.count_nonzero(mask != enc))
        diffs.append({"output_frame": len(diffs), "plan_frame": int(t), "differing_pixels": n,
                      "reconstructed_invalid_pixels": int(np.count_nonzero(mask)),
                      "encoded_invalid_pixels": int(np.count_nonzero(enc))})
    total = int(sum(d["differing_pixels"] for d in diffs))
    exact = total == 0 and len(reconstructed) == len(encoded)
    return {"exact": bool(exact), "frames_checked": len(diffs), "differing_pixels_total": total,
            "per_frame": diffs}


def _verify_and_measure_full(pl, plan_indices, decoded, W, H, grid, bottom_rows, chunk=MEASURE_CHUNK):
    """Stream decoded and reconstructed full-plan masks in lockstep, exactly verifying and measuring them.

    Returns ``(verification, measures_full)``. The decoded stream must contain exactly ``len(plan_indices)``
    frames: an early EOF (short stream) or any extra frame is refused with ``frame_count_mismatch`` before a
    report can be built. Only per-frame counts/diffs are retained; both iterators are closed on every exit
    (including an exception or interruption)."""
    reconstructed = _iter_reconstructed_masks(pl, plan_indices, W, H, grid, chunk)
    diffs = []
    per_frame = []
    try:
        for j, t in enumerate(plan_indices):
            try:
                enc = next(decoded)
            except StopIteration:
                raise BorderComponentAuditError(
                    "frame_count_mismatch",
                    f"decoded mask stream ended after {j} frames but the plan maps "
                    f"{len(plan_indices)} output frames")
            try:
                mask = next(reconstructed)
            except StopIteration:
                raise BorderComponentAuditError(
                    "frame_count_mismatch",
                    f"reconstructed mask stream ended after {j} of {len(plan_indices)} plan output frames")
            diffs.append({"output_frame": j, "plan_frame": int(t),
                          "differing_pixels": int(np.count_nonzero(mask != enc)),
                          "reconstructed_invalid_pixels": int(np.count_nonzero(mask)),
                          "encoded_invalid_pixels": int(np.count_nonzero(enc))})
            per_frame.append(_frame_measure(mask, j, t, bottom_rows))
        if next(decoded, _SENTINEL) is not _SENTINEL:
            raise BorderComponentAuditError(
                "frame_count_mismatch",
                f"decoded mask stream has more than the {len(plan_indices)} plan output frames")
    finally:
        _close_quietly(reconstructed)
        _close_quietly(decoded)
    total = int(sum(d["differing_pixels"] for d in diffs))
    verification = {"exact": bool(total == 0 and len(diffs) == len(plan_indices)),
                    "frames_checked": len(diffs), "differing_pixels_total": total, "per_frame": diffs}
    return verification, _aggregate_measures(per_frame, H, W, bottom_rows)


# --------------------------------------------------------------------------------------
# component plans and deltas
# --------------------------------------------------------------------------------------

def _component_plans(pl):
    """The four plans compared: ``full`` (the reconstruction) plus three residual ablations.

    The global homography and the corner quads are shared; only the mesh/row correction arrays are zeroed.
    ``np.zeros_like`` preserves each array's shape and dtype, so the sampler is exercised through exactly the
    same code path in every ablation."""
    def build(mesh, row):
        return S.Plan(G=pl.G, mesh_corr=mesh, row_corr=row, raw_quads=pl.raw_quads,
                      path_quads=pl.path_quads, fallbacks=pl.fallbacks)
    return {
        "full": pl,
        "global_only": build(np.zeros_like(pl.mesh_corr), np.zeros_like(pl.row_corr)),
        "global_mesh": build(pl.mesh_corr, np.zeros_like(pl.row_corr)),
        "global_rows": build(np.zeros_like(pl.mesh_corr), pl.row_corr),
    }


def _delta_values(counts):
    """Per-component per-frame invalid-pixel deltas from the four component counts (see ``DELTA_FORMULAS``)."""
    full, go = counts["full"], counts["global_only"]
    gm, gr = counts["global_mesh"], counts["global_rows"]
    return {
        "mesh": [gm[i] - go[i] for i in range(len(full))],
        "rows": [gr[i] - go[i] for i in range(len(full))],
        "interaction": [full[i] - gm[i] - gr[i] + go[i] for i in range(len(full))],
        "combined": [full[i] - go[i] for i in range(len(full))],
    }


def _delta_stats(values, region_pixels):
    """Aggregate statistics for one delta series over the output frames, plus its per-frame sign counts."""
    n = len(values)
    fractions = [(v / region_pixels) if region_pixels else 0.0 for v in values]
    pooled = (float(sum(values)) / (region_pixels * n)) if (region_pixels and n) else 0.0
    return {
        "total_pixels": int(sum(values)),
        "mean_pixels": (float(np.mean(values)) if n else 0.0),
        "min_pixels": (int(min(values)) if n else 0),
        "max_pixels": (int(max(values)) if n else 0),
        "pooled_fraction": pooled,
        "mean_fraction": (float(np.mean(fractions)) if n else 0.0),
        "min_fraction": (float(min(fractions)) if n else 0.0),
        "max_fraction": (float(max(fractions)) if n else 0.0),
        "sign_counts": {
            "positive": int(sum(1 for v in values if v > 0)),
            "negative": int(sum(1 for v in values if v < 0)),
            "zero": int(sum(1 for v in values if v == 0)),
        },
    }


def _build_deltas(measures, plan_indices):
    """Per-frame component deltas and their aggregate stats over the full image and the bottom row strip."""
    per_frame = []
    for j, t in enumerate(plan_indices):
        per_frame.append({"output_frame": j, "plan_frame": int(t), "deltas": {}})

    # Build the per-frame count series per component in one pass, then the deltas.
    counts_full = {c: [m["invalid_pixels"] for m in measures[c]["per_frame"]] for c in COMPONENTS}
    counts_bottom = {c: [m["bottom_invalid_pixels"] for m in measures[c]["per_frame"]] for c in COMPONENTS}
    deltas_full = _delta_values(counts_full)
    deltas_bottom = _delta_values(counts_bottom)
    for j, row in enumerate(per_frame):
        for name in DELTA_NAMES:
            row["deltas"][f"{name}_full_pixels"] = int(deltas_full[name][j])
            row["deltas"][f"{name}_bottom_pixels"] = int(deltas_bottom[name][j])

    full_px = measures["full"]["full_resolution"]["frame_pixels"]
    bottom_px = measures["full"]["bottom_rows"]["per_frame_pixels"]
    return {
        "definitions": DELTA_FORMULAS,
        "full_image": {name: _delta_stats(deltas_full[name], full_px) for name in DELTA_NAMES},
        "bottom_rows": {name: _delta_stats(deltas_bottom[name], bottom_px) for name in DELTA_NAMES},
        "per_frame": per_frame,
    }


# --------------------------------------------------------------------------------------
# report validation
# --------------------------------------------------------------------------------------

def _finite_fraction(value, code, where):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value),
             code, f"{where} must be a finite number")
    return float(value)


def _validate_component_block(name, blk):
    _require(isinstance(blk, dict), "bad_report", f"components.aggregate.{name} must be an object")
    for key in ("full_resolution", "bottom_rows"):
        b = blk.get(key)
        _require(isinstance(b, dict), "bad_report", f"{name}.{key} must be an object")
        for fld in ("pooled_fraction", "mean_fraction", "max_fraction"):
            v = _finite_fraction(b.get(fld), "bad_report", f"{name}.{key}.{fld}")
            _require(0.0 <= v <= 1.0, "bad_report", f"{name}.{key}.{fld} must be a fraction in [0,1]")
        _require(b.get("total_invalid_pixels", 0) <= b.get("total_pixels", 0), "bad_report",
                 f"{name}.{key} invalid count exceeds total")
        tot = b.get("total_pixels", 0)
        if tot:
            _require(abs(b["pooled_fraction"] - b["total_invalid_pixels"] / tot) < 1e-12, "bad_report",
                     f"{name}.{key}.pooled_fraction is inconsistent with its counts")


def _validate_delta_block(name, blk, region):
    _require(isinstance(blk, dict), "bad_report", f"components.deltas.{region}.{name} must be an object")
    for fld in ("pooled_fraction", "mean_fraction", "min_fraction", "max_fraction"):
        v = _finite_fraction(blk.get(fld), "bad_report", f"deltas.{region}.{name}.{fld}")
        _require(-1.0 <= v <= 1.0, "bad_report", f"deltas.{region}.{name}.{fld} out of [-1,1]")
    sc = blk.get("sign_counts")
    _require(isinstance(sc, dict) and set(sc) == {"positive", "negative", "zero"}, "bad_report",
             f"deltas.{region}.{name}.sign_counts must have positive/negative/zero")
    for k, v in sc.items():
        _require(isinstance(v, int) and v >= 0, "bad_report", f"deltas.{region}.{name}.sign_counts.{k} bad")


#: (aggregate region key, delta region key, aggregate denominator field, per-frame delta suffix,
#:  per-frame component count field, per-frame component fraction field)
_PER_FRAME_REGIONS = (
    ("full_resolution", "full_image", "frame_pixels", "full_pixels",
     "full_invalid_pixels", "full_invalid_fraction"),
    ("bottom_rows", "bottom_rows", "per_frame_pixels", "bottom_pixels",
     "bottom_invalid_pixels", "bottom_invalid_fraction"),
)


def _require_close(value, want, code, where):
    """Refuse ``value`` unless it is a finite number equal to ``want`` within :data:`STAT_TOL`."""
    _require(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)
             and abs(float(value) - float(want)) <= STAT_TOL, code,
             f"{where} must equal the recomputed value {want!r} (got {value!r})")
    return float(value)


def _reconcile_components(aggregate, deltas, per_frame):
    """Recompute every aggregate component/delta statistic from the per-frame series and refuse a mismatch.

    For each region (full resolution and bottom rows) the validator re-reads the four components' per-frame
    invalid counts and fractions plus the four delta series, then recomputes -- independently of the reported
    aggregates -- each component's ``frames``, per-frame pixel denominator, ``total_pixels``/
    ``total_invalid_pixels`` and pooled/mean/max fractions, and each delta's total/mean/min/max pixels,
    pooled/mean/min/max fractions and sign counts. Floating comparisons use :data:`STAT_TOL`; integer counts
    and denominators are compared exactly. The denominators must agree across the four components (they
    measure the same output pixels) and the ``mesh + rows + interaction == combined`` identity is checked per
    frame and in aggregate."""
    n = len(per_frame)
    count_series = {dk: {name: [] for name in COMPONENTS}
                    for _a, dk, _d, _s, _c, _f in _PER_FRAME_REGIONS}
    frac_series = {dk: {name: [] for name in COMPONENTS}
                   for _a, dk, _d, _s, _c, _f in _PER_FRAME_REGIONS}
    delta_series = {dk: {name: [] for name in DELTA_NAMES}
                    for _a, dk, _d, _s, _c, _f in _PER_FRAME_REGIONS}
    for j, row in enumerate(per_frame):
        _require(isinstance(row, dict) and row.get("output_frame") == j, "bad_report",
                 f"components.per_frame[{j}] must be an object with output_frame == {j}")
        cells = {}
        for name in COMPONENTS:
            cell = row.get(name)
            _require(isinstance(cell, dict), "bad_report",
                     f"components.per_frame[{j}].{name} must be an object")
            cells[name] = cell
        d = row.get("deltas")
        _require(isinstance(d, dict), "bad_report", f"components.per_frame[{j}].deltas must be an object")
        for _agg_key, delta_key, _denom_field, suffix, count_field, frac_field in _PER_FRAME_REGIONS:
            for name in COMPONENTS:
                v = cells[name].get(count_field)
                _require(isinstance(v, int) and not isinstance(v, bool) and v >= 0, "bad_report",
                         f"components.per_frame[{j}].{name}.{count_field} must be a non-negative integer")
                count_series[delta_key][name].append(v)
                frac_series[delta_key][name].append(
                    _finite_fraction(cells[name].get(frac_field), "bad_report",
                                     f"components.per_frame[{j}].{name}.{frac_field}"))
            f = cells["full"][count_field]
            go = cells["global_only"][count_field]
            gm = cells["global_mesh"][count_field]
            gr = cells["global_rows"][count_field]
            want = {"mesh": gm - go, "rows": gr - go, "interaction": f - gm - gr + go, "combined": f - go}
            for name in DELTA_NAMES:
                v = d.get(f"{name}_{suffix}")
                _require(isinstance(v, int) and not isinstance(v, bool) and v == want[name], "bad_report",
                         f"components.per_frame[{j}].deltas.{name}_{suffix} is inconsistent with the "
                         f"component counts")
                delta_series[delta_key][name].append(v)
            # ``interaction`` is defined so this identity must hold per frame and in aggregate.
            _require(d[f"mesh_{suffix}"] + d[f"rows_{suffix}"] + d[f"interaction_{suffix}"]
                     == d[f"combined_{suffix}"], "bad_report",
                     f"components.per_frame[{j}] mesh+rows+interaction must equal combined ({suffix})")
    for agg_key, delta_key, denom_field, _suffix, _count_field, frac_field in _PER_FRAME_REGIONS:
        # Every component samples the same output pixels, so the per-frame denominator must be identical.
        denominators = set()
        for name in COMPONENTS:
            den = aggregate[name][agg_key].get(denom_field)
            _require(isinstance(den, int) and not isinstance(den, bool) and den >= 0, "bad_report",
                     f"components.aggregate.{name}.{agg_key}.{denom_field} must be a non-negative integer")
            denominators.add(den)
        _require(len(denominators) == 1, "bad_report",
                 f"components.aggregate.*.{agg_key}.{denom_field} must agree across components")
        denom = denominators.pop()
        if n:
            _require(denom > 0, "bad_report",
                     f"components.aggregate.*.{agg_key}.{denom_field} must be positive for {n} frames")
        for name in COMPONENTS:
            blk = aggregate[name][agg_key]
            counts = count_series[delta_key][name]
            total = sum(counts)
            total_px = n * denom
            for fld in ("total_pixels", "total_invalid_pixels"):
                _require(isinstance(blk.get(fld), int) and not isinstance(blk.get(fld), bool)
                         and blk.get(fld) >= 0, "bad_report",
                         f"components.aggregate.{name}.{agg_key}.{fld} must be a non-negative integer")
            if agg_key == "full_resolution":
                _require(isinstance(blk.get("frames"), int) and not isinstance(blk.get("frames"), bool)
                         and blk.get("frames") >= 0, "bad_report",
                         f"components.aggregate.{name}.full_resolution.frames must be a non-negative integer")
            _require(blk["total_invalid_pixels"] == total, "bad_report",
                     f"components.aggregate.{name}.{agg_key}.total_invalid_pixels must equal the per-frame sum")
            _require(blk["total_pixels"] == total_px, "bad_report",
                     f"components.aggregate.{name}.{agg_key}.total_pixels must equal frames*{denom_field}")
            _require(denom == (blk["total_pixels"] // n if n else 0), "bad_report",
                     f"components.aggregate.{name}.{agg_key}.{denom_field} must equal total_pixels/frames")
            want_fracs = [(c / denom) if denom else 0.0 for c in counts]
            for j, got in enumerate(frac_series[delta_key][name]):
                _require_close(got, want_fracs[j], "bad_report",
                               f"components.per_frame[{j}].{name}.{frac_field}")
            _require_close(blk.get("pooled_fraction"), (total / total_px) if total_px else 0.0, "bad_report",
                           f"components.aggregate.{name}.{agg_key}.pooled_fraction")
            _require_close(blk.get("mean_fraction"), (sum(want_fracs) / n) if n else 0.0, "bad_report",
                           f"components.aggregate.{name}.{agg_key}.mean_fraction")
            _require_close(blk.get("max_fraction"), max(want_fracs) if want_fracs else 0.0, "bad_report",
                           f"components.aggregate.{name}.{agg_key}.max_fraction")
            if agg_key == "full_resolution":
                _require(blk.get("frames") == n, "bad_report",
                         f"components.aggregate.{name}.full_resolution.frames must equal the per-frame count")
        for name in DELTA_NAMES:
            stats = deltas[delta_key][name]
            values = delta_series[delta_key][name]
            total = sum(values)
            fracs = [(v / denom) if denom else 0.0 for v in values]
            want_signs = {
                "positive": sum(1 for v in values if v > 0),
                "negative": sum(1 for v in values if v < 0),
                "zero": sum(1 for v in values if v == 0),
            }
            _require(stats.get("total_pixels") == total, "bad_report",
                     f"deltas.{delta_key}.{name}.total_pixels must equal the per-frame sum")
            _require_close(stats.get("mean_pixels"), float(np.mean(values)) if n else 0.0, "bad_report",
                           f"deltas.{delta_key}.{name}.mean_pixels")
            _require(stats.get("min_pixels") == (min(values) if n else 0), "bad_report",
                     f"deltas.{delta_key}.{name}.min_pixels must equal the per-frame minimum")
            _require(stats.get("max_pixels") == (max(values) if n else 0), "bad_report",
                     f"deltas.{delta_key}.{name}.max_pixels must equal the per-frame maximum")
            _require_close(stats.get("pooled_fraction"),
                           (total / (denom * n)) if (denom and n) else 0.0, "bad_report",
                           f"deltas.{delta_key}.{name}.pooled_fraction")
            _require_close(stats.get("mean_fraction"), (sum(fracs) / n) if n else 0.0, "bad_report",
                           f"deltas.{delta_key}.{name}.mean_fraction")
            _require_close(stats.get("min_fraction"), (min(fracs) if fracs else 0.0), "bad_report",
                           f"deltas.{delta_key}.{name}.min_fraction")
            _require_close(stats.get("max_fraction"), (max(fracs) if fracs else 0.0), "bad_report",
                           f"deltas.{delta_key}.{name}.max_fraction")
            _require(stats.get("sign_counts") == want_signs, "bad_report",
                     f"deltas.{delta_key}.{name}.sign_counts must match the per-frame signs")
        # Aggregate identities, recomputed from the per-component totals.
        totals = {name: sum(count_series[delta_key][name]) for name in COMPONENTS}
        _require(deltas[delta_key]["mesh"]["total_pixels"] == totals["global_mesh"] - totals["global_only"],
                 "bad_report", f"deltas.{delta_key}.mesh.total_pixels is inconsistent with the components")
        _require(deltas[delta_key]["rows"]["total_pixels"] == totals["global_rows"] - totals["global_only"],
                 "bad_report", f"deltas.{delta_key}.rows.total_pixels is inconsistent with the components")
        _require(deltas[delta_key]["combined"]["total_pixels"] == totals["full"] - totals["global_only"],
                 "bad_report", f"deltas.{delta_key}.combined.total_pixels is inconsistent with the components")
        _require(deltas[delta_key]["interaction"]["total_pixels"]
                 == deltas[delta_key]["combined"]["total_pixels"] - deltas[delta_key]["mesh"]["total_pixels"]
                 - deltas[delta_key]["rows"]["total_pixels"], "bad_report",
                 f"deltas.{delta_key}.interaction.total_pixels breaks mesh+rows+interaction=combined")


def _validate_report(report):
    """Refuse an internally inconsistent report; stable codes for each invariant."""
    _require(report.get("schema") == SCHEMA, "bad_report", "wrong report schema")
    _require(report.get("status") in ("success", "verification_failed"), "bad_report", "bad report status")
    inp = report.get("input") or {}
    sha = inp.get("source_sha256")
    _require(isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha),
             "bad_report", "source_sha256 must be a 64-char lowercase hex digest")
    ver = report.get("verification") or {}
    _require(isinstance(ver.get("exact"), bool), "bad_report", "verification.exact must be a bool")
    if ver["exact"]:
        _require(ver.get("differing_pixels_total") == 0, "bad_report",
                 "exact verification must have zero differing pixels")
    comp = report.get("components")
    if report["status"] == "success":
        _require(ver["exact"] is True, "bad_report", "success report requires exact verification")
        _require(isinstance(comp, dict), "bad_report", "success report needs components")
        aggregate = comp.get("aggregate")
        _require(isinstance(aggregate, dict) and set(aggregate) == set(COMPONENTS), "bad_report",
                 "components.aggregate must contain exactly the four components")
        for name in COMPONENTS:
            _validate_component_block(name, aggregate[name])
        deltas = comp.get("deltas")
        _require(isinstance(deltas, dict), "bad_report", "success report needs components.deltas")
        _require(isinstance(deltas.get("definitions"), dict), "bad_report", "deltas needs definitions")
        for region in ("full_image", "bottom_rows"):
            _require(isinstance(deltas.get(region), dict) and set(deltas[region]) == set(DELTA_NAMES),
                     "bad_report", f"deltas.{region} must contain exactly {DELTA_NAMES}")
            for name in DELTA_NAMES:
                _validate_delta_block(name, deltas[region][name], region)
        per_frame = comp.get("per_frame")
        _require(isinstance(per_frame, list) and len(per_frame) == ver.get("frames_checked"),
                 "bad_report", "per_frame must have one entry per checked output frame")
        _reconcile_components(aggregate, deltas, per_frame)
    text = json.dumps(MF.json_safe(report), allow_nan=False)
    for bad in ("/Users/", "/home/", "\\", "out/"):
        _require(bad not in text, "not_path_free", f"report contains a private path fragment {bad!r}")
    return report


# --------------------------------------------------------------------------------------
# audit driver
# --------------------------------------------------------------------------------------

def run_audit(manifest_path, *, motion_name=DEFAULT_MOTION_NAME, mask_name=DEFAULT_MASK_NAME,
              bottom_rows=DEFAULT_BOTTOM_ROWS, expect_source_sha256=None, log=print):
    """Reconstruct the plan, verify it against the encoded mask, and report the component ablations.

    Returns the JSON-safe, path-free report dict (also the object written by :func:`main`). Raises
    :class:`BorderComponentAuditError` if the capture is unusable or the report would be inconsistent.
    """
    manifest_path = Path(manifest_path)
    run_dir = manifest_path.parent
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(int(bottom_rows) >= 1, "bad_bottom_rows", "bottom_rows must be >= 1")

    mot, mot_meta = load_motion_npz(run_dir / motion_name)
    identity = _manifest_identity(man, motion_name=motion_name, mask_name=mask_name)
    if expect_source_sha256 is not None:
        _require(identity["source_sha256"] == expect_source_sha256, "source_mismatch",
                 f"manifest source sha256 {identity['source_sha256']!r} != expected {expect_source_sha256!r}")
    identity["motion"] = _motion_identity(mot_meta)

    cfg = S.Config(**man["config"])
    pl = S.plan(mot, cfg, n_frames=mot.n_frames)
    W, H = int(mot.W), int(mot.H)
    # The strict motion artifact (``mot_meta`` and the reconstructed ``Motion``) must agree exactly with the
    # manifest's decoded source dimensions; a mismatch means the Plan was built for a different frame size.
    _require(identity["source_decoded_width"] == int(mot_meta.get("width")) == W, "shape_mismatch",
             f"source.decoded.width {identity['source_decoded_width']!r} != motion width "
             f"{mot_meta.get('width')!r} (Motion.W={W})")
    _require(identity["source_decoded_height"] == int(mot_meta.get("height")) == H, "shape_mismatch",
             f"source.decoded.height {identity['source_decoded_height']!r} != motion height "
             f"{mot_meta.get('height')!r} (Motion.H={H})")
    identity["plan_frames"] = len(pl.G)

    analysis_fps, output_fps = identity["analysis_fps"], identity["output_fps"]
    _require(np.isfinite(analysis_fps) and analysis_fps > 0.0
             and np.isfinite(output_fps) and output_fps > 0.0, "bad_cadence",
             f"analysis/output fps must be finite and > 0, got {analysis_fps!r}/{output_fps!r}")
    ratio = analysis_fps / output_fps
    step = int(round(ratio))
    _require(step >= 1 and abs(ratio - step) <= FPS_AGREEMENT_TOL, "bad_cadence",
             f"analysis/output fps ratio {ratio!r} is not an integer output decimation")
    plan_indices = list(range(0, len(pl.G), step))
    identity["output_frame_step"] = step

    motion_fps = _require_artifact_fps("motion", mot_meta.get("fps"))
    _require(abs(motion_fps - analysis_fps) <= FPS_AGREEMENT_TOL, "motion_cadence_mismatch",
             f"motion fps {motion_fps!r} != manifest analysis fps {analysis_fps!r} "
             f"(tolerance {FPS_AGREEMENT_TOL:g}); the plan indices would be misaligned")
    identity["motion_fps_matches_analysis"] = True
    _require(int(mot_meta.get("n_frames")) == identity["analysis_frames"], "motion_frame_mismatch",
             f"motion n_frames {mot_meta.get('n_frames')!r} != manifest analysis frames "
             f"{identity['analysis_frames']!r}")

    encoded, mask_info = _iter_decode_mask_bool(run_dir / mask_name)
    mask_fps = _require_artifact_fps("mask", mask_info.get("fps"))
    _require(abs(mask_fps - output_fps) <= FPS_AGREEMENT_TOL, "mask_cadence_mismatch",
             f"decoded mask fps {mask_fps!r} != manifest output fps {output_fps!r} "
             f"(tolerance {FPS_AGREEMENT_TOL:g}); the output frames would be misaligned")
    identity["mask_fps_matches_output"] = True
    # ``outputs.decoded.frames`` must agree with the plan's output-frame mapping; int()-truncation or a stale
    # count is refused with the stable ``frame_count_mismatch`` code. The decoded mask itself is checked
    # against both counts while it is streamed (early EOF/extra frames are refused).
    manifest_output_frames = identity["output_frames"]
    _require(manifest_output_frames == len(plan_indices), "frame_count_mismatch",
             f"manifest outputs.decoded.frames {manifest_output_frames} != the {len(plan_indices)} plan "
             f"output frames")
    _require(mask_info["width"] == W and mask_info["height"] == H, "shape_mismatch",
             f"decoded mask is {mask_info['width']}x{mask_info['height']}, motion is {W}x{H}")

    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    log(f"component audit: reconstructed {len(pl.G)}-frame plan from {identity['run_id']!r}; "
        f"{len(plan_indices)} output frames; verifying against {mask_name}")
    # One bounded streaming pass: the full-plan masks are reconstructed and compared against the decoded
    # mask frame by frame, retaining only per-frame counts/diffs. This both verifies and measures ``full``.
    verification, measures_full = _verify_and_measure_full(pl, plan_indices, encoded, W, H, (xx, yy),
                                                           bottom_rows)
    identity["output_frames"] = int(mask_info["frames"])
    verification.update({
        "mask_dimensions": [int(mask_info["width"]), int(mask_info["height"])],
        "mask_frames": int(mask_info["frames"]),
        "motion_frames": int(mot_meta.get("n_frames")),
        "motion_fps_matches_analysis": True,
        "mask_fps_matches_output": True,
        "output_frames_expected": len(plan_indices),
        "source_sha256_matches_expected": (None if expect_source_sha256 is None else True),
    })

    report = {
        "schema": SCHEMA,
        "status": "success",
        "input": identity,
        "method": {
            "motion_loader": "bench.real_composition_audit.load_motion_npz (np.load allow_pickle=False; "
                             "schema/required-keys/shape/dtype/finiteness validated)",
            "plan_reconstruction": "mlxstab.stabilize.plan(motion, Config(**manifest.config), "
                                   "n_frames=motion.n_frames)",
            "component_configs": COMPONENT_CONFIGS,
            "delta_formulas": DELTA_FORMULAS,
            "sampler": "mlxstab.stabilize._source_coords_ts_scales_np(scale=[1.0]) + _valid_mask_np "
                       "(production NumPy composition: global homography, then mesh bilinear-border at the "
                       "global-mapped coordinate, then row bilinear-border at the mesh-shifted y)",
            "full_resolution_measure": "count of _valid_mask_np == False over all H*W pixels per output frame",
            "bottom_rows_measured": int(bottom_rows),
            "output_frame_mapping": f"plan indices 0,{step},... (analysis {analysis_fps:g} fps -> output "
                                    f"{output_fps:g} fps)",
            "gpu_flow": False, "render_lock": False, "read_only": True, "images_rendered": False,
            "source_footage_decoded": False,
            "scope_note": SCOPE_NOTE,
        },
        "verification": verification,
        "components": None,
        "limitations": [
            "Component ablation zeroes one residual array while the production sampler still samples the other "
            "at its composed coordinate, so per-component deltas are not independent; the interaction term "
            "reports exactly this non-additivity.",
            "Full-resolution fractions come from the NumPy reconstruction of the production sampler and were "
            "verified pixel-identical to the encoded mask on this run's output frames only; other caches, "
            "unencoded analysis frames and other artifacts are not covered.",
            "manifest.outputs.plan is null for this cache, so the Plan is reconstructed from manifest.config "
            "and the Motion rather than read from a captured plan artifact; any ambiguity in those two "
            "artifacts is inherited.",
            "The encoded validity mask is decoded with FFmpeg solely to validate the reconstructed "
            "full-composition mask; the source footage itself is not decoded.",
            "Read-only CPU/NumPy geometry/coverage diagnostic: no rendering, no perceptual/quality/distortion "
            "metric, no foldover guard, and no production default is changed.",
        ],
    }

    if not verification["exact"]:
        report["status"] = "verification_failed"
        report["components"] = None
        report["stop_reason"] = (
            f"reconstructed invalid mask differs from {mask_name} on {verification['differing_pixels_total']} "
            f"pixel(s) across {verification['frames_checked']} frames; component comparison stopped")
        return _validate_report(report)

    plans = _component_plans(pl)
    # The full plan's per-frame counts were accumulated during the streaming verification pass above; the
    # three ablations stream their masks in bounded chunks, retaining only per-frame counts/fractions.
    measures = {"full": measures_full}
    for name in ("global_only", "global_mesh", "global_rows"):
        log(f"component audit: reconstructing {name} masks")
        masks = _iter_reconstructed_masks(plans[name], plan_indices, W, H, (xx, yy))
        measures[name] = _measure_masks(masks, plan_indices, bottom_rows)

    deltas = _build_deltas(measures, plan_indices)
    per_frame = []
    for j, t in enumerate(plan_indices):
        row = {"output_frame": j, "plan_frame": int(t)}
        for name in COMPONENTS:
            pf = measures[name]["per_frame"][j]
            row[name] = {
                "full_invalid_pixels": pf["invalid_pixels"],
                "full_invalid_fraction": pf["invalid_fraction"],
                "bottom_invalid_pixels": pf["bottom_invalid_pixels"],
                "bottom_invalid_fraction": pf["bottom_invalid_fraction"],
            }
        row["deltas"] = deltas["per_frame"][j]["deltas"]
        per_frame.append(row)
    report["components"] = {
        "aggregate": {name: {"full_resolution": measures[name]["full_resolution"],
                             "bottom_rows": measures[name]["bottom_rows"]} for name in COMPONENTS},
        "deltas": {"definitions": deltas["definitions"], "full_image": deltas["full_image"],
                   "bottom_rows": deltas["bottom_rows"]},
        "per_frame": per_frame,
    }
    return _validate_report(report)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(prog="bench/real_border_component_audit.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", help="successful bench/run_case manifest.json")
    p.add_argument("--out", required=True, help="path for the path-free JSON report")
    p.add_argument("--motion-name", default=DEFAULT_MOTION_NAME, help="motion NPZ basename beside the manifest")
    p.add_argument("--mask-name", default=DEFAULT_MASK_NAME, help="mask MP4 basename beside the manifest")
    p.add_argument("--bottom-rows", type=int, default=DEFAULT_BOTTOM_ROWS,
                   help="bottom row strip measured at full resolution (default: %(default)s)")
    p.add_argument("--expect-source-sha256", default=None, help="refuse unless source.sha256 matches this digest")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        report = run_audit(args.manifest, motion_name=args.motion_name, mask_name=args.mask_name,
                           bottom_rows=args.bottom_rows, expect_source_sha256=args.expect_source_sha256,
                           log=lambda m: print(m, file=sys.stderr))
        MF.write_manifest(args.out, report)
    except Exception as exc:
        print(json.dumps({"status": "failure", "error": {"type": type(exc).__name__, "message": str(exc)}},
                         sort_keys=True))
        return 1
    deltas = report["components"]["deltas"]["full_image"] if report["components"] else None
    print(json.dumps({
        "status": "success",
        "out": args.out,
        "run_id": report["input"]["run_id"],
        "verification_exact": report["verification"]["exact"],
        "output_frames": report["verification"]["frames_checked"],
        "full_invalid_pixels": (report["components"]["aggregate"]["full"]["full_resolution"]
                                ["total_invalid_pixels"] if report["components"] else None),
        "delta_total_pixels": ({k: deltas[k]["total_pixels"] for k in DELTA_NAMES} if deltas else None),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
