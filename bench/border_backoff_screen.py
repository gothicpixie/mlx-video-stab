"""Read-only CPU/NumPy screen of the opt-in sampled border backoff on one captured benchmark run.

A successful ``bench/run_case.py`` run can capture the fitted ``motion_r0.npz`` (schema ``mlxstab.motion/4``)
and the encoded validity ``mask_r0.mp4``, but (with ``--dump-plan`` off, the default) it does **not** capture the
``Plan``: the manifest records ``outputs.plan = null``. This module reconstructs that baseline Plan from the
manifest's resolved ``config`` plus the strictly loaded Motion (``mlxstab.stabilize.plan``), then screens the
opt-in ``mlxstab.stabilize.sampled_border_backoff`` over the **full** reconstructed plan.

The screen is entirely read-only and CPU/NumPy:

- the Motion is loaded through the strict, object-free loader ``bench.real_composition_audit.load_motion_npz``
  (``np.load(..., allow_pickle=False)`` with schema/shape/finiteness validation);
- the Plan is rebuilt with the existing production ``mlxstab.stabilize.plan`` (deterministic; no fitting);
- the validity mask is reconstructed with the NumPy production convention
  (``mlxstab.stabilize._source_coords_ts_scales_np`` at scale 1.0 + ``_valid_mask_np``), which mirrors the
  production ``source_coords``/``valid_mask`` sampler; it is checked to be **exactly** equal to the decoded
  ``mask_r0.mp4`` on every output frame before any candidate is compared;
- the backoff and the full-resolution measurements use only NumPy helpers from ``mlxstab.stabilize``.

No media is encoded, no GPU flow is run (RAFT is never constructed and no ``mlx`` graph is evaluated), no render
lock is acquired, and no production source, default or artifact is modified. The report is finite-safe and
path-free (atomic write through :mod:`bench.manifest`); it carries the source hash, the input manifest identity,
the method, the limitations and the per-frame and aggregate values.

Method in brief
---------------
1. Reconstruct the baseline Plan from ``manifest.config`` and the loaded Motion.
2. Verify: the reconstructed invalid mask (NumPy) must equal the decoded ``mask_r0.mp4`` boolean mask
   (gray ``> 127``) on all output frames (plan indices ``0, 2, 4, ...``, since the cache analyses at 59.94p but
   encodes output at 29.97p). If any pixel differs the candidate comparison is stopped and the mismatch is
   reported.
3. For the baseline and for each requested ``max_sampled_border_frac`` limit
   (:data:`DEFAULT_LIMITS`), run ``sampled_border_backoff`` over the full plan and measure, on every output
   frame, the full-resolution invalid fraction and the bottom-four-row invalid fraction. Candidate scales apply
   jointly to the global correction ``G -> I + s*(G-I)`` and to the mesh and row residuals.

Limitations
-----------
- The backoff probe grid is a fixed, coarse set (corners, edge samples and a small interior grid) that heavily
  over-weights the frame boundary; its worst sampled fraction is **not** the full-resolution fraction and the
  selected scale is driven by the probe worst, not by the reported full-resolution value.
- The reported full-resolution fractions come from the NumPy reconstruction of the production sampler. They were
  verified pixel-identical to the encoded mask on this run's output frames only; other caches, the queued
  10-second run and unencoded analysis frames are not covered.
- A single clip-wide scale is applied to ``G``, mesh and rows together, so a selected candidate is not a
  minimal-change correction; reducing the scale towards 0 drives the correction towards identity.
- This is a geometry/coverage diagnostic, not a perceptual or quality metric. It does not prevent foldover, does
  not render images, does not assess content distortion, and does not change any default.
- The manifest's ``outputs.plan`` is ``null`` for this cache; the Plan is reconstructed from the resolved
  ``config`` and Motion, so the screen inherits any ambiguity in those two artifacts.
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

SCHEMA = "mlxstab.border_backoff_screen/1"
DEFAULT_LIMITS = (0.005, 0.01, 0.02, 0.04)
DEFAULT_BOTTOM_ROWS = 4
DEFAULT_MOTION_NAME = "motion_r0.npz"
DEFAULT_MASK_NAME = "mask_r0.mp4"
# Bounded temporal chunk so the NumPy [frames, 1, H, W, 2] source fields stay small (a 1080p plan frame is
# ~16.6 MB); four frames keeps the probe/intermediate temporaries well under a few hundred MB.
MEASURE_CHUNK = 4
# Tight fps agreement tolerance, matching ``bench.manifest.check_encoded_cfr``'s 1e-6 comparison.
FPS_AGREEMENT_TOL = 1e-6


class BorderBackoffScreenError(ValueError):
    """The screen refused its input or produced an inconsistent report. ``code`` is a stable reason."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = str(code)


def _require(cond, code, message):
    if not cond:
        raise BorderBackoffScreenError(code, message)


def _manifest_identity(man, *, motion_name, mask_name):
    """Extract the path-free identity a screen report needs, refusing a non-success or shapeless manifest."""
    _require(isinstance(man, dict), "manifest_malformed", "manifest top level must be a JSON object")
    _require(man.get("status") == "success", "not_success",
             f"manifest status is {man.get('status')!r}; only a successful capture is screened")
    source = man.get("source")
    _require(isinstance(source, dict) and isinstance(source.get("sha256"), str) and source["sha256"],
             "manifest_malformed", "manifest is missing source.sha256")
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
    _require(isinstance(out_decoded, dict) and "fps" in out_decoded,
             "manifest_malformed", "manifest is missing outputs.decoded.fps (output cadence)")
    case = man.get("case") if isinstance(man.get("case"), dict) else {}
    # Convert here so a non-numeric cadence is a stable malformed-manifest refusal rather than a bare
    # ``TypeError``/``ValueError``; finiteness/positivity is validated in ``run_screen`` as ``bad_cadence``.
    try:
        analysis_fps = float(decoded["fps"])
        output_fps = float(out_decoded["fps"])
    except (TypeError, ValueError):
        raise BorderBackoffScreenError("manifest_malformed",
                                       "source.decoded.fps and outputs.decoded.fps must be numeric")
    return {
        "run_id": man.get("run_id"),
        "manifest_schema": man.get("schema"),
        "case": {"name": case.get("name"), "description": case.get("description"),
                 "output_fps": case.get("output_fps")},
        "source_sha256": source["sha256"],
        "source_window": {"start_s": window.get("start_s"), "duration_s": window.get("duration_s")},
        "source_preprocessing": MF.json_safe(source.get("preprocessing")),
        "weights_sha256": weights["sha256"],
        "config": MF.json_safe(config),
        "config_digest": man.get("config_digest"),
        "content_key": man.get("content_key"),
        "analysis_fps": analysis_fps,
        "analysis_frames": int(decoded["frames"]),
        "output_fps": output_fps,
        "output_frames": int(out_decoded.get("frames", 0)),
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
        raise BorderBackoffScreenError("bad_cadence", f"{name} fps must be numeric, got {value!r}")
    _require(np.isfinite(fps) and fps > 0.0, "bad_cadence",
             f"{name} fps must be finite and > 0, got {fps!r}")
    return fps


def _decode_mask_bool(path):
    """Decode the encoded gray validity mask to a list of boolean ``[H,W]`` arrays (True = no source)."""
    info = media.probe(str(path))
    gen, fps, (W, H) = media.decode(str(path), info)
    frames = [f[..., 0] > 127 for f in gen]
    return frames, {"width": int(W), "height": int(H), "fps": float(fps), "frames": len(frames)}


def _reconstructed_masks(pl, ts, W, H, grid, chunk=MEASURE_CHUNK):
    """NumPy invalid masks (True = out of source) for plan frames ``ts`` at full resolution.

    Uses the production-convention NumPy probe ``_source_coords_ts_scales_np`` at scale 1.0 and the local
    ``_valid_mask_np``; the returned masks match the encoded ``mask_r0.mp4`` classification (verified)."""
    xx, yy = grid
    out = []
    for i in range(0, len(ts), chunk):
        tsc = list(ts[i:i + chunk])
        src = S._source_coords_ts_scales_np(pl, tsc, W, H, [1.0], (xx, yy))[:, 0]   # [n,H,W,2]
        bad = ~S._valid_mask_np(src, H, W)                                          # [n,H,W]
        for k in range(len(tsc)):
            out.append(bad[k])
    return out


def _measure_masks(masks, plan_indices, bottom_rows):
    """Per-frame and aggregate invalid fractions over every pixel and over the bottom ``bottom_rows`` rows."""
    per_frame = []
    full_px = full_bad = bot_px = bot_bad = 0
    H, W = masks[0].shape if masks else (0, 0)
    s = min(int(bottom_rows), H)
    for mask, t in zip(masks, plan_indices):
        n = int(mask.size)
        nb = int(np.count_nonzero(mask))
        bottom = mask[H - s:, :] if s > 0 else mask[0:0]
        bn = int(bottom.size)
        bb = int(np.count_nonzero(bottom))
        full_px += n; full_bad += nb; bot_px += bn; bot_bad += bb
        per_frame.append({
            "output_frame": None,
            "plan_frame": int(t),
            "invalid_pixels": nb, "frame_pixels": n, "invalid_fraction": (nb / n) if n else 0.0,
            "bottom_invalid_pixels": bb, "bottom_pixels": bn,
            "bottom_invalid_fraction": (bb / bn) if bn else 0.0,
        })
    for j, row in enumerate(per_frame):
        row["output_frame"] = j
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


def run_screen(manifest_path, *, motion_name=DEFAULT_MOTION_NAME, mask_name=DEFAULT_MASK_NAME,
               limits=DEFAULT_LIMITS, bottom_rows=DEFAULT_BOTTOM_ROWS, expect_source_sha256=None, log=print):
    """Reconstruct the baseline Plan, verify it against the encoded mask, and screen the backoff limits.

    Returns the JSON-safe, path-free report dict (also the object written by :func:`main`). Raises
    :class:`BorderBackoffScreenError` if the capture is unusable or the report would be inconsistent.
    """
    manifest_path = Path(manifest_path)
    run_dir = manifest_path.parent
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    limits = [float(x) for x in limits]
    _require(limits, "bad_limit", "at least one max_sampled_border_frac limit is required")
    for x in limits:
        _require(np.isfinite(x) and 0.0 <= x < 1.0, "bad_limit",
                 f"limit must be finite and in [0, 1), got {x!r}")
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
    identity["plan_frames"] = len(pl.G)

    analysis_fps, output_fps = identity["analysis_fps"], identity["output_fps"]
    _require(np.isfinite(analysis_fps) and analysis_fps > 0.0
             and np.isfinite(output_fps) and output_fps > 0.0, "bad_cadence",
             f"analysis/output fps must be finite and > 0, got {analysis_fps!r}/{output_fps!r}")
    ratio = analysis_fps / output_fps
    step = int(round(ratio))
    _require(step >= 1 and abs(ratio - step) <= 1e-6, "bad_cadence",
             f"analysis/output fps ratio {ratio!r} is not an integer output decimation")
    plan_indices = list(range(0, len(pl.G), step))
    identity["output_frame_step"] = step

    motion_fps = _require_artifact_fps("motion", mot_meta.get("fps"))
    _require(abs(motion_fps - analysis_fps) <= FPS_AGREEMENT_TOL, "motion_cadence_mismatch",
             f"motion fps {motion_fps!r} != manifest analysis fps {analysis_fps!r} "
             f"(tolerance {FPS_AGREEMENT_TOL:g}); the plan indices would be misaligned")

    encoded, mask_info = _decode_mask_bool(run_dir / mask_name)
    mask_fps = _require_artifact_fps("mask", mask_info.get("fps"))
    _require(abs(mask_fps - output_fps) <= FPS_AGREEMENT_TOL, "mask_cadence_mismatch",
             f"decoded mask fps {mask_fps!r} != manifest output fps {output_fps!r} "
             f"(tolerance {FPS_AGREEMENT_TOL:g}); the output frames would be misaligned")
    _require(len(encoded) == len(plan_indices), "cadence_mismatch",
             f"decoded mask has {len(encoded)} frames but the plan maps {len(plan_indices)} output frames")
    _require(mask_info["width"] == W and mask_info["height"] == H, "shape_mismatch",
             f"decoded mask is {mask_info['width']}x{mask_info['height']}, motion is {W}x{H}")
    identity["output_frames"] = len(encoded)

    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    log(f"screen: reconstructed {len(pl.G)}-frame plan from {identity['run_id']!r}; "
        f"{len(plan_indices)} output frames; verifying against {mask_name}")
    reconstructed = _reconstructed_masks(pl, plan_indices, W, H, (xx, yy))
    verification = _verify_masks(reconstructed, encoded, plan_indices)

    baseline_worst = float(S._sampled_worst_border_frac_batch(
        pl, W, H, S._probe_grid(W, H), [1.0])[0])

    report = {
        "schema": SCHEMA,
        "status": "success",
        "input": identity,
        "method": {
            "motion_loader": "bench.real_composition_audit.load_motion_npz (np.load allow_pickle=False; "
                             "schema/required-keys/shape/dtype/finiteness validated)",
            "plan_reconstruction": "mlxstab.stabilize.plan(motion, Config(**manifest.config), "
                                   "n_frames=motion.n_frames)",
            "mask_reconstruction": "mlxstab.stabilize._source_coords_ts_scales_np(scale=[1.0]) + "
                                   "_valid_mask_np (NumPy production convention)",
            "output_frame_mapping": "plan indices 0,2,4,... (analysis 59.94p -> output 29.97p)",
            "full_resolution_measure": "count of _valid_mask_np == False over all H*W pixels per output frame",
            "bottom_rows_measured": int(bottom_rows),
            "backoff": "mlxstab.stabilize.sampled_border_backoff(plan, W, H, limit) over the full plan",
            "scale_application": "joint scale on G (I + s*(G-I)), mesh_corr and row_corr (mlxstab.stabilize."
                                 "_scaled_plan)",
            "sampled_note": S.SAMPLED_BORDER_NOTE,
            "limits": limits,
            "gpu_flow": False, "render_lock": False, "read_only": True, "images_rendered": False,
        },
        "verification": verification,
        "baseline": None,
        "candidates": [],
        "limitations": [
            "Backoff probe grid is coarse (corners, edge samples, small interior grid) and over-weights the "
            "frame boundary; its worst sampled fraction is not the full-resolution fraction and drives the "
            "selected scale.",
            "Full-resolution fractions come from the NumPy reconstruction of the production sampler and were "
            "verified pixel-identical to the encoded mask on this run's output frames only; other caches, the "
            "queued 10-second run and unencoded analysis frames are not covered.",
            "A single clip-wide scale is applied to G, mesh and rows jointly, so a candidate is not a "
            "minimal-change correction; scale 0 is the identity fallback.",
            "Geometry/coverage diagnostic only: no rendering, no perceptual or quality metric, no foldover "
            "guard, and no production default is changed.",
            "manifest.outputs.plan is null for this cache, so the Plan is reconstructed from manifest.config "
            "and the Motion rather than read from a captured plan artifact.",
        ],
    }

    if not verification["exact"]:
        report["status"] = "verification_failed"
        report["candidates"] = []
        report["baseline"] = None
        report["stop_reason"] = (
            f"reconstructed invalid mask differs from {mask_name} on {verification['differing_pixels_total']} "
            f"pixel(s) across {verification['frames_checked']} frames; candidate comparison stopped")
        return _validate_report(report)

    baseline_measure = _measure_masks(reconstructed, plan_indices, bottom_rows)
    report["baseline"] = {
        "limit": None, "selected_scale": 1.0, "sampled_worst_frac": baseline_worst,
        "sampled_border_measured": False, **baseline_measure,
    }

    for limit in limits:
        log(f"screen: sampled_border_backoff limit={limit}")
        cand = S.sampled_border_backoff(pl, W, H, limit)
        cand_masks = _reconstructed_masks(cand, plan_indices, W, H, (xx, yy))
        measure = _measure_masks(cand_masks, plan_indices, bottom_rows)
        report["candidates"].append({
            "limit": limit,
            "selected_scale": float(cand.sampled_border_scale),
            "sampled_worst_frac": float(cand.sampled_border_worst_frac),
            "sampled_border_measured": bool(cand.sampled_border_measured),
            **measure,
        })
    return _validate_report(report)


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
    def _frac_block(name, blk):
        _require(isinstance(blk, dict), "bad_report", f"{name} must be an object")
        for key in ("full_resolution", "bottom_rows"):
            b = blk.get(key)
            _require(isinstance(b, dict), "bad_report", f"{name}.{key} must be an object")
            for fld in ("pooled_fraction", "mean_fraction", "max_fraction"):
                v = b.get(fld)
                _require(isinstance(v, (int, float)) and np.isfinite(v) and 0.0 <= v <= 1.0,
                         "bad_report", f"{name}.{key}.{fld} must be a finite fraction")
            _require(b.get("total_invalid_pixels", 0) <= b.get("total_pixels", 0), "bad_report",
                     f"{name}.{key} invalid count exceeds total")
            tot = b.get("total_pixels", 0)
            if tot:
                _require(abs(b["pooled_fraction"] - b["total_invalid_pixels"] / tot) < 1e-12, "bad_report",
                         f"{name}.{key}.pooled_fraction is inconsistent with its counts")
        for row in blk["per_frame"]:
            _require(0.0 <= row["invalid_fraction"] <= 1.0 and 0.0 <= row["bottom_invalid_fraction"] <= 1.0,
                     "bad_report", f"{name} per-frame fraction out of range")
    if report["status"] == "success":
        _require(ver["exact"] is True, "bad_report", "success report requires exact verification")
        _require(isinstance(report.get("baseline"), dict), "bad_report", "success report needs a baseline")
        _frac_block("baseline", report["baseline"])
        _require(report.get("candidates"), "bad_report", "success report needs candidates")
        for i, cand in enumerate(report["candidates"]):
            _require(np.isfinite(cand["selected_scale"]) and 0.0 <= cand["selected_scale"] <= 1.0,
                     "bad_report", f"candidate {i} selected_scale out of [0,1]")
            _require(np.isfinite(cand["sampled_worst_frac"]) and 0.0 <= cand["sampled_worst_frac"] <= 1.0,
                     "bad_report", f"candidate {i} sampled_worst_frac out of [0,1]")
            _frac_block(f"candidate[{i}]", cand)
    text = json.dumps(MF.json_safe(report), allow_nan=False)
    for bad in ("/Users/", "/home/", "\\", "out/"):
        _require(bad not in text, "not_path_free", f"report contains a private path fragment {bad!r}")
    return report


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(prog="bench/border_backoff_screen.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", help="successful bench/run_case manifest.json")
    p.add_argument("--out", required=True, help="path for the path-free JSON report")
    p.add_argument("--motion-name", default=DEFAULT_MOTION_NAME, help="motion NPZ basename beside the manifest")
    p.add_argument("--mask-name", default=DEFAULT_MASK_NAME, help="mask MP4 basename beside the manifest")
    p.add_argument("--limits", default=",".join(str(x) for x in DEFAULT_LIMITS),
                   help="comma-separated max_sampled_border_frac limits (default: %(default)s)")
    p.add_argument("--bottom-rows", type=int, default=DEFAULT_BOTTOM_ROWS,
                   help="bottom row strip measured at full resolution (default: %(default)s)")
    p.add_argument("--expect-source-sha256", default=None, help="refuse unless source.sha256 matches this digest")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    limits = tuple(float(x) for x in args.limits.split(",") if x.strip())
    try:
        report = run_screen(args.manifest, motion_name=args.motion_name, mask_name=args.mask_name,
                            limits=limits, bottom_rows=args.bottom_rows,
                            expect_source_sha256=args.expect_source_sha256,
                            log=lambda m: print(m, file=sys.stderr))
        MF.write_manifest(args.out, report)
    except Exception as exc:
        print(json.dumps({"status": "failure", "error": {"type": type(exc).__name__, "message": str(exc)}},
                         sort_keys=True))
        return 1
    print(json.dumps({"status": "success", "out": args.out, "run_id": report["input"]["run_id"],
                      "verification_exact": report["verification"]["exact"],
                      "candidates": len(report["candidates"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
