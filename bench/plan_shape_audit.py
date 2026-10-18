"""Captured-plan geometry audit: production ``source_coords`` on a dumped Plan NPZ (CPU-orchestrated).

This is the *real-plan* companion to ``bench.shape_audit``. The analytic audit builds known fixtures and
compares the fixed-grid residual sum against the composed candidate; this module loads a **captured**
production plan (``mlxstab.stabilize.save_plan`` NPZ) and probes the *actual* production map
``mlxstab.stabilize.source_coords`` at sparse sampled coordinates. It reports how straight output-space
segments stay after mapping into the source frame and whether the mapped samples stay inside the source
bounds, for a strided selection of frames (the final frame is always included).

Scope and honesty
-----------------
This is a **warp-geometry proxy**: it measures the geometry of the production warp only. It is **not**
object/tree detection, scene understanding, or a perceptual quality evaluation, and it makes no claim about
what a viewer sees. No media is decoded, no frames are rendered, and no production code or default is
changed. The full frame is never materialised: ``source_coords`` is evaluated on an MLX grid shaped ``[1,N]``
(a single row of ``N`` sampled output coordinates), so the mesh/row corrections are still applied by the real
kernel but at ``N`` points instead of ``W*H``.

Payload safety
--------------
The NPZ is opened with ``np.load(..., allow_pickle=False)`` and only plain numeric/unicode arrays are
accepted; object arrays are rejected. ``PLAN_SCHEMA``, the required keys, the frame dimensions and every
array shape/dtype/finiteness are validated before a ``Plan`` is reconstructed. Malformed payloads raise
``PlanPayloadError`` with a stable ``code`` and a clear message; the CLI turns that into a JSON-safe error
object instead of a traceback.

Run with ``python -m bench.plan_shape_audit PLAN.npz [--every 30] [--samples 401] [--out file.json]``.

``--compare PLAN.npz PLAN.npz ...`` (two or more) is the matched multi-plan companion: it loads every plan
through the same strict loader, audits each with identical ``--every``/``--samples``, and only compares plans
whose sibling benchmark manifests prove the same source sha256, source window, preprocessing, weights sha256
and resolved stabilization ``config`` -- every config key/value must be exactly equal except the residual-
channel ablation variables ``mesh`` and ``rs_rows`` (and the payloads must agree exactly on dimensions, frame
count and fps). Each sibling manifest must also prove the supplied plan was actually captured:
``outputs.plan`` must be a nonempty path whose basename matches the plan file and
``experiment_identity.capture.dump_plan`` must be ``True``. It emits the distinct schema
``mlxstab.plan_shape_comparison/1`` and, on absent/malformed or mismatched provenance, a JSON error with exit
code ``2`` instead of comparing unproven windows.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import mlx.core as mx

from mlxstab import stabilize as S

from . import shape_audit as SA
from .manifest import json_safe

SCHEMA = "mlxstab.plan_shape_audit/1"

# Schema-common keys written by ``mlxstab.stabilize.save_plan``; every one must be present and well-formed.
REQUIRED_KEYS = ("schema", "width", "height", "frames", "fps", "G", "mesh_corr", "row_corr",
                 "raw_quads", "path_quads", "fallbacks")
# ``mlxstab.plan/2`` adds sampled border-backoff fields; ``/3`` adds shot boundaries.
BORDER_KEYS = ("sampled_border_scale", "sampled_border_worst_frac", "sampled_border_measured")
CUT_KEYS = ("cut_frames",)
SCHEMA_KEYS = {S.PLAN_SCHEMA_V1: REQUIRED_KEYS,
               S.PLAN_SCHEMA_V2: REQUIRED_KEYS + BORDER_KEYS,
               S.PLAN_SCHEMA: REQUIRED_KEYS + BORDER_KEYS + CUT_KEYS}

# Output-space probes. The first two are axis-aligned; the last two are the two diagonals. Diagonal (and
# horizontal/vertical) bending is a warp-geometry proxy only -- it is not object/tree detection or perception.
SEGMENT_ORDER = ("horizontal", "vertical", "diagonal_down", "diagonal_up")

COVERAGE_DOC = {
    "units": "fraction",
    "definition": "fraction of the mapped source samples inside the source bounds [-0.5, W-0.5] x [-0.5, H-0.5]",
    "reports": ["n", "in_bounds", "frac", "x_min", "x_max", "y_min", "y_max", "fully_in_bounds"],
}

SCOPE_NOTE = (
    "Real-plan warp-geometry proxy. Loads a captured Plan NPZ (plain arrays only, allow_pickle=False) and "
    "evaluates production stabilize.source_coords on MLX grids shaped [1,N] at sparse sampled output "
    "coordinates. It records how straight output-space segments stay after mapping to source space and how "
    "much of each mapped segment lands inside the source bounds. It is NOT object/tree detection, scene "
    "understanding, or a perceptual quality evaluation; no media is decoded, no frames are rendered, and no "
    "production code or default behavior is changed."
)


COMPARISON_SCHEMA = "mlxstab.plan_shape_comparison/1"

COMPARISON_SCOPE = (
    "Matched multi-plan warp-geometry comparison. Loads several captured Plan NPZs (plain arrays only, "
    "allow_pickle=False) together with their sibling benchmark manifests, verifies the plans share source "
    "identity (source sha256, source window, preprocessing, weights sha256) and a resolved stabilization config "
    "that is exactly equal except for the intended residual-channel ablation keys mesh and rs_rows, plus exact "
    "dimensions/frame count/fps. Each sibling manifest must prove the supplied plan was captured: "
    "outputs.plan must be a nonempty path whose basename matches the plan file and "
    "experiment_identity.capture.dump_plan must be True. It then audits each with identical "
    "--every/--samples and reports per-plan, per-segment, per-frame curvature "
    "and source-bounds coverage with deltas against the reference case. It is NOT object/tree detection, scene "
    "understanding, or a perceptual quality evaluation; no media is decoded, no frames are rendered, and no "
    "production code or default behavior is changed. Curvature and coverage are reported side by side and are "
    "never collapsed into a winner score."
)


class PlanPayloadError(ValueError):
    """A captured plan NPZ failed validation. ``code`` is a stable machine-readable reason."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = str(code)


class ComparisonError(ValueError):
    """A matched multi-plan comparison was refused. ``code`` is a stable machine-readable reason."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = str(code)


# --------------------------------------------------------------------------------------
# NPZ loading and validation
# --------------------------------------------------------------------------------------

def _arr(z, key):
    try:
        return np.asarray(z[key])       # lazy read: an object array raises here with allow_pickle=False
    except PlanPayloadError:
        raise
    except Exception as exc:
        raise PlanPayloadError("unreadable", f"cannot read array {key!r} with allow_pickle=False: {exc}")


def _require(cond, code, message):
    if not cond:
        raise PlanPayloadError(code, message)


def _unicode_scalar(z, key):
    a = _arr(z, key)
    _require(a.ndim == 0 and a.size == 1 and a.dtype.kind in "US", "bad_scalar",
             f"{key!r} must be a scalar string, got shape {a.shape} dtype {a.dtype}")
    return str(a.item())


def _int_scalar(z, key):
    a = _arr(z, key)
    _require(a.ndim == 0 and a.dtype.kind in "iu", "bad_scalar",
             f"{key!r} must be an integer scalar, got shape {a.shape} dtype {a.dtype}")
    return int(a.item())


def _float_scalar(z, key):
    a = _arr(z, key)
    _require(a.ndim == 0 and a.dtype.kind in "fiu", "bad_scalar",
             f"{key!r} must be a numeric scalar, got shape {a.shape} dtype {a.dtype}")
    return float(a.item())


def _bool_scalar(z, key):
    a = _arr(z, key)
    _require(a.ndim == 0 and a.size == 1 and a.dtype.kind == "b", "bad_scalar",
             f"{key!r} must be a boolean scalar, got shape {a.shape} dtype {a.dtype}")
    return bool(a.item())


def _numeric_array(z, key, shape=None):
    a = _arr(z, key)
    _require(a.dtype.kind in "fiu", "bad_shape", f"{key!r} must be a numeric array, got dtype {a.dtype}")
    if shape is not None:
        _require(a.shape == shape, "bad_shape", f"{key!r} must have shape {shape}, got {a.shape}")
    _require(bool(np.isfinite(a).all()), "nonfinite", f"{key!r} contains non-finite values")
    return a


def load_plan_npz(path):
    """Load and validate a ``save_plan`` NPZ, then reconstruct ``mlxstab.stabilize.Plan``.

    Accepts ``mlxstab.plan/3`` (current, with shot boundaries), ``/2`` (sampled border-backoff fields), and the
    legacy ``/1`` payload, which lacks those fields and loads with the disabled defaults.

    Returns ``(plan, meta)``. Malformed payloads raise ``PlanPayloadError`` with a stable ``code`` (one of
    ``unreadable``, ``missing_keys``, ``bad_schema``, ``bad_scalar``, ``bad_dims``, ``bad_shape``,
    ``bad_range``, ``nonfinite``). No pickle is ever loaded: ``np.load(..., allow_pickle=False)`` is used.
    """
    try:
        z = np.load(path, allow_pickle=False)
    except Exception as exc:  # bad zip, missing file, object array that would need unpickling, ...
        raise PlanPayloadError("unreadable", f"cannot load NPZ {str(path)!r} with allow_pickle=False: {exc}")
    try:
        files = set(z.files)
        missing = [k for k in REQUIRED_KEYS if k not in files]
        _require(not missing, "missing_keys", f"missing required arrays: {sorted(missing)}")
        schema = _unicode_scalar(z, "schema")
        _require(schema in S.PLAN_SCHEMAS, "bad_schema",
                 f"schema {schema!r} does not match any of {S.PLAN_SCHEMAS!r}")
        extra = [k for k in SCHEMA_KEYS[schema] if k not in files]
        _require(not extra, "missing_keys", f"missing required arrays: {sorted(extra)}")

        W, H = _int_scalar(z, "width"), _int_scalar(z, "height")
        T = _int_scalar(z, "frames")
        fps = _float_scalar(z, "fps")
        _require(W >= 2 and H >= 2, "bad_dims", f"width/height must be >= 2, got {W}x{H}")
        _require(T >= 1, "bad_dims", f"frames must be >= 1, got {T}")
        _require(math.isfinite(fps) and fps > 0.0, "bad_scalar", f"fps must be finite and > 0, got {fps!r}")
        fallbacks = _int_scalar(z, "fallbacks")

        G = _numeric_array(z, "G", (T, 3, 3))
        mc = _numeric_array(z, "mesh_corr", None)
        _require(mc.ndim == 4 and mc.shape[0] == T and mc.shape[3] == 2 and mc.shape[1] >= 1
                 and mc.shape[2] >= 1, "bad_shape",
                 f"'mesh_corr' must be [T, gr+1, gc+1, 2] with T={T} and gr,gc>=0, got {mc.shape}")
        # gr=0 with gc>0 is never produced by a valid config (only (0,0) disables the mesh) and would make
        # production ``source_coords`` divide by ``H / gr``; reject it here, before reconstructing the Plan.
        _require(not (mc.shape[1] == 1 and mc.shape[2] > 1), "bad_shape",
                 f"'mesh_corr' must not have zero row cells (gr=0) with positive column cells "
                 f"(gc={mc.shape[2] - 1}); source_coords divides by H/gr, got {mc.shape}")
        rc = _numeric_array(z, "row_corr", None)
        _require(rc.ndim == 3 and rc.shape[0] == T and rc.shape[2] == 2 and rc.shape[1] >= 1, "bad_shape",
                 f"'row_corr' must be [T, nr, 2] with T={T} and nr>=1, got {rc.shape}")
        rq = _numeric_array(z, "raw_quads", (T, 4, 2))
        pq = _numeric_array(z, "path_quads", (T, 4, 2))

        # Explicit shot boundaries are required by /3; older payloads have no key and load as one shot.
        cuts = ()
        if schema == S.PLAN_SCHEMA:
            _require("cut_frames" in files, "missing_key", "missing required plan key 'cut_frames'")
        if "cut_frames" in files:
            cf = _numeric_array(z, "cut_frames", None)
            _require(cf.ndim == 1 and cf.dtype.kind in "iu", "bad_shape",
                     f"'cut_frames' must be a 1-D integer array, got shape {cf.shape} dtype {cf.dtype}")
            cf = [int(v) for v in cf]
            _require(all(0 <= v < T for v in cf), "bad_range",
                     f"'cut_frames' must lie inside [0, {T}), got {cf}")
            _require(cf == sorted(set(cf)), "bad_range",
                     f"'cut_frames' must be sorted and unique, got {cf}")
            cuts = tuple(cf)

        # Sampled border-backoff result: present in /2 and /3. Legacy /1 loads as disabled defaults.
        if schema in (S.PLAN_SCHEMA_V2, S.PLAN_SCHEMA):
            scale = _float_scalar(z, "sampled_border_scale")
            worst = _float_scalar(z, "sampled_border_worst_frac")
            measured = _bool_scalar(z, "sampled_border_measured")
            _require(math.isfinite(scale) and 0.0 <= scale <= 1.0, "bad_range",
                     f"'sampled_border_scale' must be finite and in [0, 1], got {scale!r}")
            _require(math.isfinite(worst) and 0.0 <= worst <= 1.0, "bad_range",
                     f"'sampled_border_worst_frac' must be finite and in [0, 1], got {worst!r}")
        else:
            scale, worst, measured = 1.0, 0.0, False

        plan = S.Plan(
            G=[np.array(G[i], np.float64) for i in range(T)],
            mesh_corr=np.array(mc, np.float64),
            row_corr=np.array(rc, np.float64),
            raw_quads=np.array(rq, np.float64),
            path_quads=np.array(pq, np.float64),
            fallbacks=fallbacks,
            sampled_border_scale=scale, sampled_border_worst_frac=worst, sampled_border_measured=measured,
            cuts=cuts,
        )
        meta = {
            "path": str(path),
            "schema": schema,
            "width": W, "height": H, "frames": T, "fps": fps, "fallbacks": fallbacks,
            "mesh_cells": [int(mc.shape[2] - 1), int(mc.shape[1] - 1)],   # (gc, gr)
            "row_bands": int(rc.shape[1]),
            "sampled_border_scale": scale, "sampled_border_worst_frac": worst,
            "sampled_border_measured": measured,
            "cut_frames": list(cuts),
        }
        return plan, meta
    finally:
        z.close()


# --------------------------------------------------------------------------------------
# output-space probes
# --------------------------------------------------------------------------------------

def _margin(W, H):
    """Inset from the frame edge so probes stay strictly inside; at least 1 px on tiny frames."""
    return max(1, min(5, (min(W, H) - 1) // 3))


def output_segments(W, H):
    """The four output-space probes: horizontal, vertical and both diagonals, as ``{name: {p0, p1, ...}}``."""
    m = float(_margin(W, H))
    x0, x1 = m, W - 1.0 - m
    y0, y1 = m, H - 1.0 - m
    raw = {
        "horizontal": ((x0, H / 2.0), (x1, H / 2.0), "horizontal (left->right at mid height)"),
        "vertical": ((W / 2.0, y0), (W / 2.0, y1), "vertical (top->bottom at mid width)"),
        "diagonal_down": ((x0, y0), (x1, y1), "diagonal (top-left -> bottom-right)"),
        "diagonal_up": ((x1, y0), (x0, y1), "diagonal (top-right -> bottom-left)"),
    }
    segs = {}
    for name in SEGMENT_ORDER:
        p0, p1, desc = raw[name]
        p0, p1 = np.array(p0, np.float64), np.array(p1, np.float64)
        segs[name] = {"orientation": desc, "p0": p0.tolist(), "p1": p1.tolist(),
                      "length_px": float(np.linalg.norm(p1 - p0))}
    return segs


def frame_indices(T, every):
    """Strided frame indices ``range(0, T, every)`` with the final frame ``T-1`` always included."""
    if int(every) < 1:
        raise ValueError("every must be >= 1")
    if int(T) < 1:
        raise ValueError("T must be >= 1")
    idx = list(range(0, int(T), int(every)))
    if idx[-1] != int(T) - 1:
        idx.append(int(T) - 1)
    return idx


# --------------------------------------------------------------------------------------
# production-map evaluation
# --------------------------------------------------------------------------------------

def map_points(pl, t, W, H, p):
    """Evaluate production ``source_coords`` at numpy output points ``p [N,2]`` -> source points ``[N,2]``.

    The MLX grid is fed as ``[1, N]`` (one row of N samples) so no full ``H x W`` frame is ever built.
    """
    p = np.asarray(p, np.float64)
    x = mx.array(p[:, 0].astype(np.float32))[None]
    y = mx.array(p[:, 1].astype(np.float32))[None]
    src = S.source_coords(pl, t, W, H, (x, y))
    mx.eval(src)
    return np.array(src[0, 0], np.float64)


def coverage(mapped, W, H):
    """Source-bounds coverage of mapped samples, matching ``warp.valid_mask``'s pixel-edge convention."""
    mapped = np.asarray(mapped, np.float64)
    x, y = mapped[:, 0], mapped[:, 1]
    inside = (x >= -0.5) & (x <= W - 0.5) & (y >= -0.5) & (y <= H - 0.5)
    return {
        "n": int(len(mapped)),
        "in_bounds": int(inside.sum()),
        "frac": float(inside.mean()) if len(mapped) else None,
        "x_min": float(x.min()) if len(mapped) else None,
        "x_max": float(x.max()) if len(mapped) else None,
        "y_min": float(y.min()) if len(mapped) else None,
        "y_max": float(y.max()) if len(mapped) else None,
        "fully_in_bounds": bool(inside.all()),
    }


def evaluate_segment(pl, t, W, H, seg, samples):
    """One output-space segment: input line, mapped straightness (production map) and coverage."""
    p0, p1 = np.asarray(seg["p0"], np.float64), np.asarray(seg["p1"], np.float64)
    pts = SA.sample_line(p0, p1, samples)
    cache = {}

    def map_fn(p):
        out = map_points(pl, t, W, H, p)
        cache["mapped"] = out
        return out

    # Reuse the analytic audit's straightness metric verbatim; it also records ``input_span_px``.
    mapped_metrics = SA.map_line_deviation(map_fn, p0, p1, samples)
    return {
        "input_line": {"p0": p0.tolist(), "p1": p1.tolist(), "orientation": seg["orientation"]},
        "input": SA.best_fit_line(pts),
        "mapped": mapped_metrics,
        "coverage": coverage(cache["mapped"], W, H),
    }


def evaluate_frames(pl, W, H, frames, segments, samples):
    """Evaluate every probe for every requested frame, keeping the exact frame indices."""
    per_frame = []
    for t in frames:
        per_frame.append({
            "frame": int(t),
            "segments": {name: evaluate_segment(pl, t, W, H, segments[name], samples) for name in SEGMENT_ORDER},
        })
    return per_frame


def _summary(per_frame):
    per_segment = {}
    for name in SEGMENT_ORDER:
        devs = [pf["segments"][name]["mapped"]["dev_max_px"] for pf in per_frame]
        covs = [pf["segments"][name]["coverage"]["frac"] for pf in per_frame]
        valid = [d for d in devs if isinstance(d, (int, float))]
        per_segment[name] = {
            "frames": len(per_frame),
            "mapped_dev_max_px": devs,
            "mapped_dev_max_px_max": max(valid) if valid else None,
            "coverage_frac_min": min(covs) if covs else None,
            "invalid_lines": sum(1 for d in devs if d is None),
        }
    overall_dev = [v["mapped_dev_max_px_max"] for v in per_segment.values()
                   if v["mapped_dev_max_px_max"] is not None]
    overall_cov = [v["coverage_frac_min"] for v in per_segment.values()
                   if v["coverage_frac_min"] is not None]
    return {
        "per_segment": per_segment,
        "max_mapped_dev_max_px": max(overall_dev) if overall_dev else None,
        "min_coverage_frac": min(overall_cov) if overall_cov else None,
        "any_mapped_line_invalid": any(v["invalid_lines"] for v in per_segment.values()),
    }


# --------------------------------------------------------------------------------------
# report and CLI
# --------------------------------------------------------------------------------------

def _audit_plan(pl, meta, *, every, samples):
    """Audit an already-loaded plan with one selection; shared by the single and comparison entry points."""
    W, H, T = meta["width"], meta["height"], meta["frames"]
    segments = output_segments(W, H)
    frames = frame_indices(T, every)
    per_frame = evaluate_frames(pl, W, H, frames, segments, samples)
    return {
        "schema": SCHEMA,
        "ok": True,
        "scope": SCOPE_NOTE,
        "plan": meta,
        "plan_payload_safety": {
            "loader": "np.load(..., allow_pickle=False)",
            "required_keys": list(SCHEMA_KEYS[meta["schema"]]),
            "validated": ["schema", "required_keys", "dimensions", "array_shapes", "dtype_kind", "finiteness"],
        },
        "selection": {"every": every, "samples": samples, "frames": frames,
                      "final_frame_included": frames[-1] == T - 1},
        "segments": segments,
        "coverage": COVERAGE_DOC,
        "metric": SA.METRIC_DOC,
        "per_frame": per_frame,
        "summary": _summary(per_frame),
        "limitations": [
            "warp-geometry proxy only; not object/tree detection, scene understanding or perceptual evaluation",
            "probes four straight output-space segments, not the full field or real scene content",
            "mesh/row corrections are bilinear at the mesh resolution and sampled at the probe points",
            "no media is decoded and no frames are rendered; nothing about real-footage quality is claimed",
        ],
    }


def run_plan_audit(plan_path, *, every=30, samples=401):
    """Load a captured plan and audit production ``source_coords`` on the four probes; return a JSON-safe report."""
    every, samples = int(every), int(samples)
    if every < 1:
        raise ValueError("every must be >= 1")
    if samples < 2:
        raise ValueError("samples must be >= 2")
    pl, meta = load_plan_npz(plan_path)
    return _audit_plan(pl, meta, every=every, samples=samples)


# --------------------------------------------------------------------------------------
# matched multi-plan comparison
# --------------------------------------------------------------------------------------

def sibling_manifest_path(plan_path):
    """Benchmark manifest written beside a plan dump by ``bench.run_case`` (``<run_dir>/manifest.json``)."""
    return Path(plan_path).parent / "manifest.json"


def _manifest_identity(man, path, plan_path):
    """Pull the provenance identity out of a benchmark manifest, validating the fields the match needs."""
    def fail(message):
        raise ComparisonError("manifest_malformed", f"manifest {str(path)!r}: {message}")

    if not isinstance(man, dict):
        fail("top level must be a JSON object")
    case = man.get("case")
    if not isinstance(case, dict) or not isinstance(case.get("name"), str) or not case["name"].strip():
        fail("missing or empty case.name")
    source = man.get("source")
    if not isinstance(source, dict):
        fail("missing source object")
    sha = source.get("sha256")
    if not isinstance(sha, str) or not sha:
        fail("missing source.sha256")
    window = source.get("window")
    if not isinstance(window, dict) or "start_s" not in window or "duration_s" not in window:
        fail("missing source.window.start_s/duration_s")
    if not isinstance(source.get("preprocessing"), dict):
        fail("missing source.preprocessing")
    weights = man.get("weights")
    if not isinstance(weights, dict):
        fail("missing weights object")
    wsha = weights.get("sha256")
    if not isinstance(wsha, str) or not wsha:
        fail("missing weights.sha256")
    config = man.get("config")
    if not isinstance(config, dict) or not config:
        fail("missing resolved config object (required to prove the comparison is controlled)")

    # The manifest must prove *this* run captured the plan, and that the recorded output is the supplied file.
    # Compare basenames only: the plan is guaranteed to be a sibling of its manifest, so a relative
    # ``outputs.plan`` (written relative to the run dir or the caller's CWD) is never resolved against the CWD.
    outputs = man.get("outputs")
    if not isinstance(outputs, dict):
        fail("missing outputs object")
    outputs_plan = outputs.get("plan")
    if not isinstance(outputs_plan, str) or not outputs_plan.strip():
        fail("missing or empty outputs.plan (the manifest must record the captured plan path)")
    experiment = man.get("experiment_identity")
    if not isinstance(experiment, dict):
        fail("missing experiment_identity object")
    capture = experiment.get("capture")
    if not isinstance(capture, dict):
        fail("missing experiment_identity.capture object")
    if "dump_plan" not in capture:
        fail("missing experiment_identity.capture.dump_plan")
    if capture["dump_plan"] is not True:
        raise ComparisonError(
            "manifest_plan_not_captured",
            f"manifest {str(path)!r}: experiment_identity.capture.dump_plan is "
            f"{capture['dump_plan']!r}, not True; the run did not capture this plan, so its provenance cannot "
            f"be compared")
    want = Path(plan_path).name
    got = Path(outputs_plan).name
    if got != want:
        raise ComparisonError(
            "mismatched_plan_output",
            f"manifest {str(path)!r}: outputs.plan basename {got!r} does not match the supplied plan file "
            f"{want!r}; the manifest does not describe this plan")
    return {
        "case": case["name"], "run_id": man.get("run_id"), "manifest_schema": man.get("schema"),
        "source_sha256": sha,
        "source_window": {"start_s": window.get("start_s"), "duration_s": window.get("duration_s")},
        "preprocessing": json_safe(source.get("preprocessing")),
        "weights_sha256": wsha,
        "config": json_safe(config),
        "capture_dump_plan": True,
        "output_plan": outputs_plan,
    }


def load_manifest_identity(plan_path):
    """Load and validate the sibling benchmark manifest for ``plan_path``; return its provenance identity."""
    path = sibling_manifest_path(plan_path)
    if not path.exists():
        raise ComparisonError("manifest_missing",
                              f"no sibling benchmark manifest at {str(path)!r} for plan {str(plan_path)!r}")
    try:
        man = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # malformed JSON or unreadable file
        raise ComparisonError("manifest_unreadable",
                              f"cannot read/parse manifest {str(path)!r}: {exc}")
    ident = _manifest_identity(man, path, plan_path)
    ident["plan_path"] = str(plan_path)
    ident["manifest_path"] = str(path)
    return ident


# Manifest identity fields that must be exactly equal across compared plans, with the error code for each.
_MATCH_CHECKS = (
    ("source_sha256", "mismatched_source", "source sha256"),
    ("source_window", "mismatched_window", "source window"),
    ("preprocessing", "mismatched_preprocessing", "preprocessing"),
    ("weights_sha256", "mismatched_weights", "weights sha256"),
)

# Resolved Config keys that the residual-channel E02 comparison deliberately ablates. Every other key/value in
# the resolved Config must be exactly equal across the compared plans; a difference there is a mismatched
# experiment and is refused instead of being silently compared.
_ALLOWED_CONFIG_DIFF_KEYS = ("mesh", "rs_rows")
_CONFIG_MISMATCH_CODE = "mismatched_config"


def _canonical_value(value):
    """Canonical JSON for one config value so ``1``/``True`` and tuple/list spellings compare exactly."""
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _require_matched(identities):
    """Refuse plans whose sibling manifests do not prove identical source/weights provenance *and* config.

    Source sha256/window/preprocessing/weights sha256 must match exactly. In addition every resolved
    stabilization Config key/value must be exactly equal across the compared plans except the residual-channel
    ablation variables in :data:`_ALLOWED_CONFIG_DIFF_KEYS`; the comparison is refused (``mismatched_config``)
    otherwise, because a different seed, iteration count, smoothing or any other field would make the
    curvature/coverage deltas measure the wrong thing.
    """
    ref = identities[0]
    for ident in identities[1:]:
        for key, code, label in _MATCH_CHECKS:
            if ident[key] != ref[key]:
                raise ComparisonError(
                    code,
                    f"plan {ident['plan_path']!r} (case {ident['case']!r}) has a different {label} than "
                    f"plan {ref['plan_path']!r} (case {ref['case']!r}); refusing to compare unproven windows")
        ref_cfg, cfg = ref["config"], ident["config"]
        allowed = set(_ALLOWED_CONFIG_DIFF_KEYS)
        changed = sorted(
            k for k in set(ref_cfg) | set(cfg)
            if k not in allowed and (
                (k in ref_cfg) != (k in cfg)
                or _canonical_value(ref_cfg.get(k)) != _canonical_value(cfg.get(k))))
        if changed:
            detail = "; ".join(f"{k}: reference {ref_cfg.get(k)!r}, plan {cfg.get(k)!r}" for k in changed)
            raise ComparisonError(
                _CONFIG_MISMATCH_CODE,
                f"plan {ident['plan_path']!r} (case {ident['case']!r}) has a different stabilization config "
                f"than plan {ref['plan_path']!r} (case {ref['case']!r}): {detail}; this residual-channel "
                f"comparison permits differences only in {list(_ALLOWED_CONFIG_DIFF_KEYS)}")


def _per_segment_arrays(report):
    """Per-segment, per-frame absolute dev (max, p95) and coverage, in ``selection.frames`` order."""
    out = {}
    for name in SEGMENT_ORDER:
        dev, dev95, cov = [], [], []
        for pf in report["per_frame"]:
            seg = pf["segments"][name]
            dev.append(seg["mapped"]["dev_max_px"])
            dev95.append(seg["mapped"]["dev_p95_px"])
            cov.append(seg["coverage"]["frac"])
        out[name] = {"dev_max_px": dev, "dev_p95_px": dev95, "coverage_frac": cov}
    return out


def _delta(values, reference):
    """Element-wise ``values - reference``; a missing (``None``) side yields ``None`` rather than a guess."""
    return [None if (v is None or r is None) else v - r for v, r in zip(values, reference)]


def compare_plans(plan_paths, *, every=30, samples=401):
    """Matched multi-plan audit: verify shared provenance/selection, then report per-case curvature and coverage.

    Each plan is loaded with the strict :func:`load_plan_npz` loader and audited with the *same* selection. The
    sibling benchmark manifests must prove the supplied plan was captured (``outputs.plan`` basename matches and
    ``experiment_identity.capture.dump_plan is True``), agree on source sha256, source window, preprocessing and
    weights sha256, and their resolved stabilization ``config`` must be exactly equal except for the
    residual-channel ablation keys ``mesh`` and ``rs_rows``. The plans must agree exactly on dimensions, frame
    count and fps. Case names must be unique; ``baseline`` is the reference when present, otherwise the first
    plan. Curvature and coverage are reported separately for every case and frame, with deltas against the
    reference and no winner score.
    """
    every, samples = int(every), int(samples)
    if every < 1:
        raise ValueError("every must be >= 1")
    if samples < 2:
        raise ValueError("samples must be >= 2")
    paths = [Path(p) for p in plan_paths]
    if len(paths) < 2:
        raise ComparisonError("too_few_plans", "comparison needs at least two plan paths")

    identities = [load_manifest_identity(p) for p in paths]
    names = [i["case"] for i in identities]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ComparisonError("duplicate_case",
                              f"case names must be unique across compared plans; duplicated: {dupes}")
    _require_matched(identities)

    loaded = []
    for path, ident in zip(paths, identities):
        try:
            pl, meta = load_plan_npz(path)
        except PlanPayloadError as exc:
            raise ComparisonError(f"plan_{exc.code}",
                                  f"plan {str(path)!r} (case {ident['case']!r}): {exc}")
        loaded.append({"identity": ident, "meta": meta, "plan": pl})

    ref_meta = loaded[0]["meta"]
    for entry in loaded[1:]:
        meta, ident = entry["meta"], entry["identity"]
        tag = f"plan {ident['plan_path']!r} (case {ident['case']!r})"
        if (meta["width"], meta["height"]) != (ref_meta["width"], ref_meta["height"]):
            raise ComparisonError("mismatched_dimensions",
                                  f"{tag} is {meta['width']}x{meta['height']}, reference is "
                                  f"{ref_meta['width']}x{ref_meta['height']}")
        if meta["frames"] != ref_meta["frames"]:
            raise ComparisonError("mismatched_frame_count",
                                  f"{tag} has {meta['frames']} frames, reference has {ref_meta['frames']}")
        if meta["fps"] != ref_meta["fps"]:
            raise ComparisonError("mismatched_fps",
                                  f"{tag} has fps {meta['fps']!r}, reference has {ref_meta['fps']!r}")

    reports = [_audit_plan(e["plan"], e["meta"], every=every, samples=samples) for e in loaded]
    ref_index = names.index("baseline") if "baseline" in names else 0
    ref_arrays = _per_segment_arrays(reports[ref_index])
    frames = reports[ref_index]["selection"]["frames"]

    cases = []
    for i, (entry, report) in enumerate(zip(loaded, reports)):
        arrays = _per_segment_arrays(report)
        per_segment = {}
        for name in SEGMENT_ORDER:
            absolute = arrays[name]
            per_segment[name] = {
                "dev_max_px": absolute["dev_max_px"],
                "dev_p95_px": absolute["dev_p95_px"],
                "coverage_frac": absolute["coverage_frac"],
                "delta_vs_reference": {
                    "dev_max_px": _delta(absolute["dev_max_px"], ref_arrays[name]["dev_max_px"]),
                    "dev_p95_px": _delta(absolute["dev_p95_px"], ref_arrays[name]["dev_p95_px"]),
                    "coverage_frac": _delta(absolute["coverage_frac"], ref_arrays[name]["coverage_frac"]),
                },
            }
        cases.append({
            "case": entry["identity"]["case"],
            "is_reference": i == ref_index,
            "plan": report["plan"],
            "manifest_path": entry["identity"]["manifest_path"],
            "summary": report["summary"],
            "per_frame": report["per_frame"],
            "per_segment": per_segment,
        })

    return {
        "schema": COMPARISON_SCHEMA,
        "ok": True,
        "scope": COMPARISON_SCOPE,
        "reference": names[ref_index],
        "selection": {"every": every, "samples": samples, "frames": frames,
                      "final_frame_included": frames[-1] == ref_meta["frames"] - 1},
        "equality": {
            "manifests_matched": True,
            "checked_manifest_fields": [label for _, _, label in _MATCH_CHECKS],
            "checked_capture_fields": ["outputs.plan", "experiment_identity.capture.dump_plan"],
            "checked_config_keys": sorted({k for ident in identities for k in ident["config"]}),
            "allowed_config_diff_keys": list(_ALLOWED_CONFIG_DIFF_KEYS),
            "checked_plan_fields": ["dimensions", "frame_count", "fps"],
            "width": ref_meta["width"], "height": ref_meta["height"],
            "frames": ref_meta["frames"], "fps": ref_meta["fps"],
        },
        "provenance": {
            "source_sha256": identities[ref_index]["source_sha256"],
            "source_window": identities[ref_index]["source_window"],
            "preprocessing": identities[ref_index]["preprocessing"],
            "weights_sha256": identities[ref_index]["weights_sha256"],
            "config": identities[ref_index]["config"],
            "capture": {
                "checked_fields": ["outputs.plan", "experiment_identity.capture.dump_plan"],
                "dump_plan": True,
                "output_plans": {i["case"]: i["output_plan"] for i in identities},
            },
            "plans": [dict(e["identity"]) for e in loaded],
        },
        "segments": reports[ref_index]["segments"],
        "coverage": COVERAGE_DOC,
        "metric": SA.METRIC_DOC,
        "cases": cases,
        "not_a_winner_score": (
            "per-segment and per-frame curvature (deviation) and source-bounds coverage are reported "
            "separately for every case, with deltas against the reference; this report deliberately does not "
            "collapse them into a single winner or quality score"),
        "limitations": [
            "warp-geometry proxy only; not object/tree detection, scene understanding or perceptual evaluation",
            "probes four straight output-space segments per case at the shared frame selection",
            "only plans whose sibling manifests prove the plan was captured (outputs.plan basename matches, "
            "capture.dump_plan True) and the same source sha256, source window, preprocessing, "
            "weights sha256 and resolved config (all keys equal except mesh/rs_rows) are compared; anything "
            "else is refused, not assumed",
            "mesh/row corrections are bilinear at the mesh resolution and sampled at the probe points",
            "no media is decoded and no frames are rendered; nothing about real-footage quality is claimed",
        ],
    }


def _error_payload(plan_path, exc):
    return {
        "schema": SCHEMA,
        "ok": False,
        "plan": {"path": str(plan_path)},
        "error": {"code": exc.code, "message": str(exc)},
    }


def _run_compare(args):
    """Compare branch of the CLI: emit the comparison report, or a JSON error and exit ``2`` on refusal."""
    try:
        report = compare_plans(args.compare, every=args.every, samples=args.samples)
    except ComparisonError as exc:
        payload = {"schema": COMPARISON_SCHEMA, "ok": False, "plans": [str(p) for p in args.compare],
                   "error": {"code": exc.code, "message": str(exc)}}
        text = json.dumps(json_safe(payload), indent=1, sort_keys=True, allow_nan=False)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        else:
            print(text)
        return 2

    text = json.dumps(json_safe(report), indent=1, sort_keys=True, allow_nan=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"compare reference={report['reference']} cases={[c['case'] for c in report['cases']]} "
              f"frames={report['selection']['frames']}")
    else:
        print(text)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Captured-plan warp-geometry audit (production source_coords; no media, no rendering).")
    ap.add_argument("plan", nargs="?", help="path to a Plan NPZ written by mlxstab.stabilize.save_plan")
    ap.add_argument("--compare", nargs="+", metavar="PLAN.npz", default=None,
                    help="compare two or more matched Plan NPZs (each needs its sibling manifest.json; "
                         "baseline is the reference when present)")
    ap.add_argument("--every", type=int, default=30, help="sample every n-th frame (final frame always added)")
    ap.add_argument("--samples", type=int, default=401, help="samples per output-space segment")
    ap.add_argument("--out", default=None, help="write the JSON report here (default: stdout)")
    args = ap.parse_args(argv)
    if args.every < 1:
        ap.error("--every must be >= 1")
    if args.samples < 2:
        ap.error("--samples must be >= 2")

    if args.compare is not None:
        if args.plan is not None:
            ap.error("pass either a single PLAN or --compare PLAN.npz ..., not both")
        if len(args.compare) < 2:
            ap.error("--compare needs at least two PLAN.npz paths")
        return _run_compare(args)

    if args.plan is None:
        ap.error("a PLAN.npz path or --compare PLAN.npz ... is required")

    try:
        report = run_plan_audit(args.plan, every=args.every, samples=args.samples)
    except PlanPayloadError as exc:
        text = json.dumps(json_safe(_error_payload(args.plan, exc)), indent=1, sort_keys=True, allow_nan=False)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        else:
            print(text)
        return 2

    text = json.dumps(json_safe(report), indent=1, sort_keys=True, allow_nan=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        s = report["summary"]
        print(f"plan={report['plan']['path']} frames={report['selection']['frames']} "
              f"max mapped dev={s['max_mapped_dev_max_px']} px min coverage={s['min_coverage_frac']}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
