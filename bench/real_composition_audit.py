"""Captured-plan + captured-motion composition audit (CPU-orchestrated; real fitted motions).

This is the **real-motion** companion to ``bench/plan_shape_audit``. The plan audit probes production
``source_coords`` alone; the analytic audits (``bench/shape_audit``, ``bench/composition_audit``) build known
fixtures. This module loads a **captured** production plan (``mlxstab.stabilize.save_plan``) together with the
**captured** fitted motion (``mlxstab.stabilize.save_motion_npz``) from the same benchmark run and compares
production's actual output->source map against the Stage 1 composed candidate
(``bench/composition_candidate.candidate_source_coords``) in three modes:

* ``exact_composition`` -- ``highpass=False``: the exact composition ``Phi_t`` of the fitted per-pair maps.
* ``source_highpass``   -- ``highpass=True``, ``smoothing_basis="source"``.
* ``reference_highpass``-- ``highpass=True``, ``smoothing_basis="reference"``.

The same four sparse output-space line probes and the same strided frame selection as
``bench.plan_shape_audit`` are used, so the two audits are directly aligned. Both high-pass charts are
computed with one shared temporal smoothing per basis (never re-smoothed per frame). Curvature (best-fit-line
deviation), source-bounds coverage and the pointwise displacement from production are reported per frame,
per segment and per mode; there is deliberately **no** composite/winner score.

Honesty scope
-------------
This is an experiment on **real fitted motions**. It measures how three candidate maps differ from the
production map on four straight sparse probes. It is **not** object/tree detection, scene understanding, a
perceptual quality evaluation, or evidence that a tree looks straight to a viewer, and it is **not** a
production integration: no production code, default or output is changed. No media is decoded and no frames
are rendered.

Payload safety
--------------
Both NPZs are opened with ``np.load(..., allow_pickle=False)`` and only plain numeric/unicode arrays are
accepted; object arrays are rejected. The plan is validated through ``plan_shape_audit.load_plan_npz``; the
motion has its own strict schema/shape/finiteness validation. The sibling benchmark manifest must prove that
*this* run captured both supplied artifacts: ``outputs.plan``/``outputs.motion`` basenames must match the
supplied files and ``experiment_identity.capture.dump_plan``/``dump_motion`` must both be ``True``. Malformed
payloads raise ``MotionPayloadError``; a refused provenance/manifest raises ``CompositionAuditError`` with a
stable ``code``; the CLI turns either into a JSON-safe error object with exit code ``2``.

Run with::

    python -m bench.real_composition_audit PLAN.npz MOTION.npz [--every 30] [--samples 401] [--out file.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from mlxstab import stabilize as S
from mlxstab import motion as MO
from mlxstab.motion import Fit

from . import composition_candidate as CC
from . import plan_shape_audit as PSA
from . import shape_audit as SA
from .manifest import json_safe

SCHEMA = "mlxstab.real_composition_audit/1"

# Three candidate maps, exactly the ones named in the E02 composition experiment. ``exact_composition`` is the
# composed oracle (``highpass=False``); the other two are the two temporal-smoothing charts of the candidate.
CANDIDATE_MODES = ("exact_composition", "source_highpass", "reference_highpass")

MODE_META = {
    "exact_composition": {"highpass": False, "smoothing_basis": None,
                          "description": "exact composition Phi_t of the fitted per-pair maps (highpass=False)"},
    "source_highpass": {"highpass": True, "smoothing_basis": "source",
                        "description": "composed candidate with the source-chart temporal high-pass"},
    "reference_highpass": {"highpass": True, "smoothing_basis": "reference",
                           "description": "composed candidate with the reference-chart temporal high-pass"},
}

# Keys written by ``mlxstab.stabilize.save_motion_npz``; every one must be present and well-formed.
MOTION_REQUIRED_KEYS = ("schema", "width", "height", "n_frames", "fps", "stride", "n_fits",
                        "H", "mesh", "rows", "inlier_frac", "mover_frac", "fallback")
# Observation-coverage support masses added in ``mlxstab.motion/2``; required for /2 through /5, absent for /1.
MOTION_SUPPORT_KEYS = ("mesh_support_mass", "row_support_mass")
# Forward-backward cycle-consistency maps added in ``mlxstab.motion/3``; required for /3 through /5.
MOTION_CYCLE_KEYS = ("cycle_err_rms", "cycle_inconsistency_frac", "cycle_sample_mass", "cycle_map_available")
# Fit-gate diagnostics (valid/RANSAC-inlier counts and a stable numeric fallback reason code) added in
# ``mlxstab.motion/4``; required for /4 and /5, absent for /1, /2 and /3 (their Fits load with the three fields ``None``).
MOTION_FIT_META_KEYS = ("n_valid_samples", "n_ransac_inliers", "fallback_reason_code", "fit_meta_available")
# The just-captured, still-unpublished ``/4`` payloads named the integer code array ``fallback_reason`` before
# the standard field name ``fallback_reason_code``; read that pre-standardization key as a fallback so
# the preserved capture still loads. The writer only ever emits the canonical ``fallback_reason_code``.
MOTION_FIT_META_REASON_FALLBACK_KEY = "fallback_reason"

SCOPE_NOTE = (
    "Experiment on real fitted motions. Loads a captured Plan NPZ and the fitted Motion NPZ from the same "
    "benchmark run (plain arrays only, allow_pickle=False), reconstructs the Plan, Motion, Fit and Config, and "
    "compares the production source_coords map with the Stage 1 composed candidate in three modes "
    "(exact composition, source high-pass, reference high-pass) on the same four sparse straight output-space "
    "probes and frame selection as bench.plan_shape_audit. It reports per-frame curvature, source-bounds "
    "coverage and pointwise displacement from production, never a combined/winner score. It is NOT object/tree "
    "detection, scene understanding, a perceptual quality evaluation, or evidence of perceived tree "
    "straightness; no media is decoded, no frames are rendered, no production code, default or output is "
    "changed, and no production integration is proposed."
)


class MotionPayloadError(ValueError):
    """A captured motion NPZ failed validation. ``code`` is a stable machine-readable reason."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = str(code)


class CompositionAuditError(ValueError):
    """A captured plan/motion pair was refused. ``code`` is a stable machine-readable reason."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = str(code)


# --------------------------------------------------------------------------------------
# motion NPZ loading and validation
# --------------------------------------------------------------------------------------

def _arr(z, key):
    try:
        return np.asarray(z[key])       # lazy read: an object array raises here with allow_pickle=False
    except MotionPayloadError:
        raise
    except Exception as exc:
        raise MotionPayloadError("unreadable", f"cannot read array {key!r} with allow_pickle=False: {exc}")


def _require(cond, code, message):
    if not cond:
        raise MotionPayloadError(code, message)


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


def _numeric_array(z, key, shape=None):
    a = _arr(z, key)
    _require(a.dtype.kind in "fiu", "bad_shape", f"{key!r} must be a numeric array, got dtype {a.dtype}")
    if shape is not None:
        _require(a.shape == shape, "bad_shape", f"{key!r} must have shape {shape}, got {a.shape}")
    _require(bool(np.isfinite(a).all()), "nonfinite", f"{key!r} contains non-finite values")
    return a


def load_motion_npz(path):
    """Load and validate a ``save_motion_npz`` NPZ, then reconstruct ``mlxstab.stabilize.Motion``.

    Returns ``(motion, meta)``. ``mlxstab.motion/5`` (with explicit shot boundaries and the
    ``sequence_boundary`` reason), ``mlxstab.motion/4`` (with the ``mesh_support_mass``/``row_support_mass``
    observation-coverage arrays, the forward-backward cycle maps **and** the per-fit gate diagnostics
    ``n_valid_samples``/``n_ransac_inliers``/``fallback_reason_code``), ``mlxstab.motion/3`` (support + cycle, no fit
    metadata), ``mlxstab.motion/2`` (support only, cycle maps and fit metadata loaded as ``None``) and legacy
    ``mlxstab.motion/1`` (no support arrays, loaded with ``support=None`` and no cycle/fit metadata) are all
    accepted. A pre-standardization ``/4`` payload that still names the integer reason-code array
    ``fallback_reason`` is read through that alias so the preserved capture loads; the canonical key remains
    ``fallback_reason_code``. Malformed payloads raise :class:`MotionPayloadError` with a stable
    ``code`` (one of ``unreadable``, ``missing_keys``, ``bad_schema``, ``bad_scalar``, ``bad_dims``,
    ``bad_shape``, ``nonfinite``, ``negative``, ``bad_range``). No pickle is ever loaded: ``np.load(...,
    allow_pickle=False)`` is used.
    """
    try:
        z = np.load(path, allow_pickle=False)
    except Exception as exc:  # bad zip, missing file, object array that would need unpickling, ...
        raise MotionPayloadError("unreadable", f"cannot load NPZ {str(path)!r} with allow_pickle=False: {exc}")
    try:
        files = set(z.files)
        missing = [k for k in MOTION_REQUIRED_KEYS if k not in files]
        _require(not missing, "missing_keys", f"missing required arrays: {sorted(missing)}")

        schema = _unicode_scalar(z, "schema")
        _require(schema in S.MOTION_SCHEMAS, "bad_schema",
                 f"schema {schema!r} is not one of {S.MOTION_SCHEMAS!r}")
        legacy = schema == S.MOTION_SCHEMA_V1                    # /1 has no support-mass arrays (support=None)
        has_support = schema in S.MOTION_SCHEMAS_WITH_SUPPORT    # /2 through /5 carry support mass
        has_cycle = schema in S.MOTION_SCHEMAS_WITH_CYCLE        # /3 through /5 carry cycle maps
        has_fit_meta = schema in S.MOTION_SCHEMAS_WITH_FIT_META   # /4 and /5 carry fit-gate diagnostics

        W, H = _int_scalar(z, "width"), _int_scalar(z, "height")
        n_frames = _int_scalar(z, "n_frames")
        fps = _float_scalar(z, "fps")
        stride = _int_scalar(z, "stride")
        n_fits = _int_scalar(z, "n_fits")
        _require(W >= 2 and H >= 2, "bad_dims", f"width/height must be >= 2, got {W}x{H}")
        _require(n_frames >= 1, "bad_dims", f"n_frames must be >= 1, got {n_frames}")
        _require(stride >= 1, "bad_dims", f"stride must be >= 1, got {stride}")
        _require(n_fits >= 0, "bad_dims", f"n_fits must be >= 0, got {n_fits}")
        _require(math.isfinite(fps) and fps > 0.0, "bad_scalar", f"fps must be finite and > 0, got {fps!r}")

        Hm = _numeric_array(z, "H", (n_fits, 3, 3))
        mesh = _numeric_array(z, "mesh", None)
        _require(mesh.ndim == 4 and mesh.shape[0] == n_fits and mesh.shape[3] == 2
                 and mesh.shape[1] >= 1 and mesh.shape[2] >= 1, "bad_shape",
                 f"'mesh' must be [n_fits, gr+1, gc+1, 2] with n_fits={n_fits} and gr,gc>=0, got {mesh.shape}")
        # gr=0 with gc>0 is never produced by analyze (it fits with max(gc,1),max(gr,1) cells) and would make
        # the candidate's reference grid ill-defined; reject it here rather than compose a malformed mesh.
        _require(not (mesh.shape[1] == 1 and mesh.shape[2] > 1), "bad_shape",
                 f"'mesh' must not have zero row cells (gr=0) with positive column cells (gc={mesh.shape[2] - 1}); "
                 f"got {mesh.shape}")
        rows = _numeric_array(z, "rows", None)
        _require(rows.ndim == 3 and rows.shape[0] == n_fits and rows.shape[2] == 2 and rows.shape[1] >= 1,
                 "bad_shape", f"'rows' must be [n_fits, nr, 2] with n_fits={n_fits} and nr>=1, got {rows.shape}")
        inlier = _numeric_array(z, "inlier_frac", (n_fits,))
        mover = _numeric_array(z, "mover_frac", (n_fits,))
        fb = _arr(z, "fallback")
        _require(fb.shape == (n_fits,) and fb.dtype.kind in "biu", "bad_shape",
                 f"'fallback' must be a boolean/integer array of shape ({n_fits},), got shape {fb.shape} "
                 f"dtype {fb.dtype}")

        if legacy:
            mesh_mass = row_mass = None                          # /1 predates support, keep it loadable
        else:
            support_missing = [k for k in MOTION_SUPPORT_KEYS if k not in files]
            _require(not support_missing, "missing_keys",
                     f"schema {schema!r} requires support arrays; missing: {sorted(support_missing)}")
            mesh_mass = _numeric_array(z, "mesh_support_mass", (n_fits, mesh.shape[1], mesh.shape[2]))
            row_mass = _numeric_array(z, "row_support_mass", (n_fits, rows.shape[1]))
            # Support mass is a non-negative sample count: reject negatives rather than silently accept them.
            for _key, _mass in (("mesh_support_mass", mesh_mass), ("row_support_mass", row_mass)):
                _min = float(_mass.min()) if _mass.size else 0.0
                _require(_min >= 0.0, "negative", f"{_key!r} must be non-negative, got minimum {_min}")

        if has_cycle:
            cycle_missing = [k for k in MOTION_CYCLE_KEYS if k not in files]
            _require(not cycle_missing, "missing_keys",
                     f"schema {schema!r} requires cycle-map arrays; missing: {sorted(cycle_missing)}")
            cyc_rms = _numeric_array(z, "cycle_err_rms", (n_fits, mesh.shape[1], mesh.shape[2]))
            cyc_frac = _numeric_array(z, "cycle_inconsistency_frac", (n_fits, mesh.shape[1], mesh.shape[2]))
            cyc_mass = _numeric_array(z, "cycle_sample_mass", (n_fits, mesh.shape[1], mesh.shape[2]))
            cyc_avail = _arr(z, "cycle_map_available")
            _require(cyc_avail.shape == (n_fits,) and cyc_avail.dtype.kind in "biu", "bad_shape",
                     f"'cycle_map_available' must be a boolean/integer array of shape ({n_fits},), got shape "
                     f"{cyc_avail.shape} dtype {cyc_avail.dtype}")
            # Availability is a strict 0/1 flag; a nonzero integer must not silently become True.
            _a_min = int(cyc_avail.min()) if cyc_avail.size else 0
            _a_max = int(cyc_avail.max()) if cyc_avail.size else 0
            _require(_a_min >= 0 and _a_max <= 1, "bad_range",
                     f"'cycle_map_available' values must be 0 or 1, got [{_a_min}, {_a_max}]")
            # RMS and sample mass are non-negative; the inconsistency fraction is a fraction in [0, 1].
            for _key, _a in (("cycle_err_rms", cyc_rms), ("cycle_sample_mass", cyc_mass)):
                _min = float(_a.min()) if _a.size else 0.0
                _require(_min >= 0.0, "negative", f"{_key!r} must be non-negative, got minimum {_min}")
            _lo = float(cyc_frac.min()) if cyc_frac.size else 0.0
            _hi = float(cyc_frac.max()) if cyc_frac.size else 0.0
            _require(_lo >= 0.0 and _hi <= 1.0, "bad_range",
                     f"'cycle_inconsistency_frac' must lie in [0, 1], got [{_lo}, {_hi}]")
            # The writer contract serializes a Fit with None cycle fields as all-zero maps tagged unavailable
            # (False); an unavailable fit carrying any nonzero cycle data is malformed and would be dropped.
            # Checked after the per-map range rules so those stable codes keep precedence.
            _unavailable = ~cyc_avail.astype(bool)
            for _key, _a in (("cycle_err_rms", cyc_rms), ("cycle_inconsistency_frac", cyc_frac),
                             ("cycle_sample_mass", cyc_mass)):
                _nz = int(np.count_nonzero(_a[_unavailable]))
                _require(_nz == 0, "bad_range",
                         f"{_key!r} must be all-zero for fits with 'cycle_map_available'=0, "
                         f"got {_nz} nonzero entries")
        else:
            cyc_rms = cyc_frac = cyc_mass = cyc_avail = None   # /1 and /2 predate the cycle maps

        if has_fit_meta:
            meta_missing = [k for k in MOTION_FIT_META_KEYS if k not in files]
            # Tolerate the pre-standardization key for the reason code so the preserved /4 capture loads; the
            # canonical key is still required whenever neither name is present.
            if "fallback_reason_code" in meta_missing and MOTION_FIT_META_REASON_FALLBACK_KEY in files:
                meta_missing.remove("fallback_reason_code")
            _require(not meta_missing, "missing_keys",
                     f"schema {schema!r} requires fit-metadata arrays; missing: {sorted(meta_missing)}")
            reason_key = ("fallback_reason_code" if "fallback_reason_code" in files
                          else MOTION_FIT_META_REASON_FALLBACK_KEY)
            n_valid = _numeric_array(z, "n_valid_samples", (n_fits,))
            n_inliers = _numeric_array(z, "n_ransac_inliers", (n_fits,))
            _require(n_valid.dtype.kind in "iu" and n_inliers.dtype.kind in "iu", "bad_shape",
                     f"'n_valid_samples'/'n_ransac_inliers' must be integer arrays, got {n_valid.dtype}/"
                     f"{n_inliers.dtype}")
            for _key, _a in (("n_valid_samples", n_valid), ("n_ransac_inliers", n_inliers)):
                _min = int(_a.min()) if _a.size else 0
                _require(_min >= -1, "bad_range",
                         f"{_key!r} must be >= -1 (-1 = unavailable, 0 is a real count), got minimum {_min}")
            reason_code = _arr(z, reason_key)
            _require(reason_code.shape == (n_fits,) and reason_code.dtype.kind in "iu", "bad_shape",
                     f"{reason_key!r} must be an integer array of shape ({n_fits},), got shape "
                     f"{reason_code.shape} dtype {reason_code.dtype}")
            avail = _arr(z, "fit_meta_available")
            _require(avail.shape == (n_fits,) and avail.dtype.kind in "biu", "bad_shape",
                     f"'fit_meta_available' must be a boolean/integer array of shape ({n_fits},), got shape "
                     f"{avail.shape} dtype {avail.dtype}")
            _a_min = int(avail.min()) if avail.size else 0
            _a_max = int(avail.max()) if avail.size else 0
            _require(_a_min >= 0 and _a_max <= 1, "bad_range",
                     f"'fit_meta_available' values must be 0 or 1, got [{_a_min}, {_a_max}]")
            # Reason codes are schema-versioned: /4 permits 0..3 and /5 adds sequence_boundary as code 4.
            # ``FALLBACK_REASON_UNAVAILABLE_CODE`` (-1) marks a fit whose metadata was not recorded.
            _code_lo, _code_hi = min(MO.FALLBACK_REASON_BY_CODE), max(MO.FALLBACK_REASON_BY_CODE)
            _schema_reason_hi = (_code_hi if schema == S.MOTION_SCHEMA else
                                 MO.FALLBACK_REASON_CODES[MO.FALLBACK_REASON_LOW_RANSAC_INLIER_FRACTION])
            _c_min = int(reason_code.min()) if reason_code.size else 0
            _c_max = int(reason_code.max()) if reason_code.size else 0
            _require(_c_min >= MO.FALLBACK_REASON_UNAVAILABLE_CODE and _c_max <= _schema_reason_hi, "bad_range",
                     f"{reason_key!r} must be in [{MO.FALLBACK_REASON_UNAVAILABLE_CODE}, {_schema_reason_hi}] "
                     f"for schema {schema!r} (-1 = unavailable), got [{_c_min}, {_c_max}]")
            _unavail = ~avail.astype(bool)
            for _key, _a in (("n_valid_samples", n_valid), ("n_ransac_inliers", n_inliers)):
                _bad = int(np.count_nonzero(_a[_unavail] != -1))
                _require(_bad == 0, "bad_range",
                         f"{_key!r} must be -1 for fits with 'fit_meta_available'=0, got {_bad} others")
            _bad_unavail = int(np.count_nonzero(
                reason_code[_unavail] != MO.FALLBACK_REASON_UNAVAILABLE_CODE))
            _require(_bad_unavail == 0, "bad_range",
                     f"{reason_key!r} must be {MO.FALLBACK_REASON_UNAVAILABLE_CODE} for fits with "
                     f"'fit_meta_available'=0, got {_bad_unavail} others")
            _bad_avail = int(np.count_nonzero(
                reason_code[~_unavail] == MO.FALLBACK_REASON_UNAVAILABLE_CODE))
            _require(_bad_avail == 0, "bad_range",
                     f"{reason_key!r} must be one of {_code_lo}..{_code_hi} for fits with "
                     f"'fit_meta_available'=1, got {_bad_avail} unavailable codes")
        else:
            # /1, /2 and /3 predate the fit-gate diagnostics; their Fits load with all three fields None.
            n_valid = n_inliers = reason_code = avail = None

        # Explicit shot boundaries are required by /5; older payloads have no key and load as one shot.
        cuts = ()
        if schema == S.MOTION_SCHEMA:
            _require("cut_frames" in files, "missing_keys", "schema /5 requires 'cut_frames'")
        if "cut_frames" in files:
            cf = _numeric_array(z, "cut_frames", None)
            _require(cf.ndim == 1 and cf.dtype.kind in "iu", "bad_shape",
                     f"'cut_frames' must be a 1-D integer array, got shape {cf.shape} dtype {cf.dtype}")
            cf = [int(v) for v in cf]
            _require(all(0 <= v < n_frames for v in cf), "bad_range",
                     f"'cut_frames' must lie inside [0, {n_frames}), got {cf}")
            _require(cf == sorted(set(cf)), "bad_range",
                     f"'cut_frames' must be sorted and unique, got {cf}")
            cuts = tuple(cf)
        if schema == S.MOTION_SCHEMA:
            expected_boundary_fits = [c // stride - 1 for c in cuts]
            actual_boundary_fits = np.flatnonzero(
                reason_code == MO.FALLBACK_REASON_CODES[MO.FALLBACK_REASON_SEQUENCE_BOUNDARY]).tolist()
            _require(actual_boundary_fits == expected_boundary_fits, "bad_range",
                     "schema /5 sequence_boundary fit reasons must match cut_frames and stride; "
                     f"expected fit indices {expected_boundary_fits}, got {actual_boundary_fits}")

        def _cycle(i, arr):
            return None if cyc_avail is None or not bool(cyc_avail[i]) else np.array(arr[i], np.float64)

        def _fit_meta(i):
            if avail is None or not bool(avail[i]):
                return None, None, None
            return int(n_valid[i]), int(n_inliers[i]), MO.FALLBACK_REASON_BY_CODE[int(reason_code[i])]

        fits = [Fit(H=np.array(Hm[i], np.float64), mesh=np.array(mesh[i], np.float64),
                    rows=np.array(rows[i], np.float64), inlier_frac=float(inlier[i]),
                    mover_frac=float(mover[i]), fallback=bool(fb[i]),
                    n_valid_samples=_fit_meta(i)[0], n_ransac_inliers=_fit_meta(i)[1],
                    fallback_reason=_fit_meta(i)[2],
                    mesh_support_mass=(None if mesh_mass is None else np.array(mesh_mass[i], np.float64)),
                    row_support_mass=(None if row_mass is None else np.array(row_mass[i], np.float64)),
                    cycle_err_rms=_cycle(i, cyc_rms), cycle_inconsistency_frac=_cycle(i, cyc_frac),
                    cycle_sample_mass=_cycle(i, cyc_mass))
                for i in range(n_fits)]
        mot = S.Motion(W, H, fps, fits=fits, stride=stride, n_frames=n_frames, cuts=cuts)
        sequence_boundary_code = MO.FALLBACK_REASON_CODES[MO.FALLBACK_REASON_SEQUENCE_BOUNDARY]
        sequence_boundaries = (int(np.count_nonzero(reason_code == sequence_boundary_code))
                               if reason_code is not None else len(cuts))
        actual_fallbacks = (fb & (reason_code != sequence_boundary_code)
                            if reason_code is not None else fb)
        meta = {
            "path": str(path),
            "schema": schema,
            "width": W, "height": H, "n_frames": n_frames, "fps": fps, "stride": stride, "n_fits": n_fits,
            "mesh_cells": [int(mesh.shape[2] - 1), int(mesh.shape[1] - 1)],   # (gc, gr)
            "row_bands": int(rows.shape[1]),
            "fallback_fits": int(np.count_nonzero(actual_fallbacks)),
            "sequence_boundaries": sequence_boundaries,
            "cut_frames": list(cuts),
            "support_mass": {
                "present": has_support,
                "keys": list(MOTION_SUPPORT_KEYS),
                "threshold": float(MO.SUPPORT_THRESHOLD),
                "meaning": "non-robust kernel-weighted count of usable non-mover flow samples at each mesh "
                           "vertex / row band, recorded before diffusion; >= threshold = directly observed, "
                           "< threshold = filled. An observation-coverage diagnostic, not probability or "
                           "flow confidence.",
            },
            "cycle_maps": {
                "present": bool(has_cycle),
                "keys": list(MOTION_CYCLE_KEYS) if has_cycle else [],
                "available_fits": (None if cyc_avail is None else int(np.count_nonzero(cyc_avail))),
                "meaning": "forward-backward cycle-consistency maps on the mesh grid (cycle_err_rms px, "
                           "cycle_inconsistency_frac in [0,1], cycle_sample_mass); NOT a true correspondence "
                           "error, NOT a probability and NOT calibrated uncertainty, and never used by "
                           "stabilization or fitting.",
            },
        }
        return mot, meta
    finally:
        z.close()


# --------------------------------------------------------------------------------------
# sibling-manifest provenance (both artifacts captured)
# --------------------------------------------------------------------------------------

def sibling_manifest_path(plan_path):
    """Benchmark manifest written beside a plan dump by ``bench.run_case`` (``<run_dir>/manifest.json``)."""
    return Path(plan_path).parent / "manifest.json"


def _manifest_identity(man, path, plan_path, motion_path):
    def fail(message):
        raise CompositionAuditError("manifest_malformed", f"manifest {str(path)!r}: {message}")

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
    if not isinstance(weights, dict) or not isinstance(weights.get("sha256"), str) or not weights["sha256"]:
        fail("missing weights.sha256")
    config = man.get("config")
    if not isinstance(config, dict) or not config:
        fail("missing resolved config object (required to reconstruct the candidate Config)")

    outputs = man.get("outputs")
    if not isinstance(outputs, dict):
        fail("missing outputs object")
    outputs_plan = outputs.get("plan")
    outputs_motion = outputs.get("motion")
    if not isinstance(outputs_plan, str) or not outputs_plan.strip():
        fail("missing or empty outputs.plan (the manifest must record the captured plan path)")
    if not isinstance(outputs_motion, str) or not outputs_motion.strip():
        fail("missing or empty outputs.motion (the manifest must record the captured motion path)")
    experiment = man.get("experiment_identity")
    if not isinstance(experiment, dict):
        fail("missing experiment_identity object")
    capture = experiment.get("capture")
    if not isinstance(capture, dict):
        fail("missing experiment_identity.capture object")
    for key in ("dump_plan", "dump_motion"):
        if key not in capture:
            fail(f"missing experiment_identity.capture.{key}")
        if capture[key] is not True:
            raise CompositionAuditError(
                f"manifest_{key}_not_captured",
                f"manifest {str(path)!r}: experiment_identity.capture.{key} is {capture[key]!r}, not True; the "
                f"run did not capture this artifact, so its provenance cannot be compared")
    want_plan, got_plan = Path(plan_path).name, Path(outputs_plan).name
    if got_plan != want_plan:
        raise CompositionAuditError(
            "mismatched_plan_output",
            f"manifest {str(path)!r}: outputs.plan basename {got_plan!r} does not match the supplied plan file "
            f"{want_plan!r}; the manifest does not describe this plan")
    want_motion, got_motion = Path(motion_path).name, Path(outputs_motion).name
    if got_motion != want_motion:
        raise CompositionAuditError(
            "mismatched_motion_output",
            f"manifest {str(path)!r}: outputs.motion basename {got_motion!r} does not match the supplied motion "
            f"file {want_motion!r}; the manifest does not describe this motion")
    return {
        "case": case["name"], "run_id": man.get("run_id"), "manifest_schema": man.get("schema"),
        "source_sha256": sha,
        "source_window": {"start_s": window.get("start_s"), "duration_s": window.get("duration_s")},
        "preprocessing": json_safe(source.get("preprocessing")),
        "weights_sha256": weights["sha256"],
        "config": json_safe(config),
        "capture_dump_plan": True, "capture_dump_motion": True,
        "output_plan": outputs_plan, "output_motion": outputs_motion,
    }


def load_manifest_identity(plan_path, motion_path):
    """Load and validate the sibling benchmark manifest for the plan/motion pair; return its provenance."""
    path = sibling_manifest_path(plan_path)
    if Path(motion_path).parent.resolve() != Path(plan_path).parent.resolve():
        raise CompositionAuditError(
            "artifacts_not_siblings",
            f"plan {str(plan_path)!r} and motion {str(motion_path)!r} are not in the same run directory; "
            f"both must be siblings of the benchmark manifest")
    if not path.exists():
        raise CompositionAuditError("manifest_missing",
                                    f"no sibling benchmark manifest at {str(path)!r} for plan {str(plan_path)!r}")
    try:
        man = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # malformed JSON or unreadable file
        raise CompositionAuditError("manifest_unreadable",
                                    f"cannot read/parse manifest {str(path)!r}: {exc}")
    ident = _manifest_identity(man, path, plan_path, motion_path)
    ident["plan_path"] = str(plan_path)
    ident["motion_path"] = str(motion_path)
    ident["manifest_path"] = str(path)
    return ident


# --------------------------------------------------------------------------------------
# input loading, scope and config consistency
# --------------------------------------------------------------------------------------

def load_composition_inputs(plan_path, motion_path):
    """Load, validate and cross-check the plan, motion, manifest config and composition scope.

    Returns ``(plan, motion, cfg, meta)`` where ``meta`` carries ``plan``/``motion`` metadata and the
    manifest ``identity``. Enforces the candidate's composition scope (stride 1, ``len(plan.G) ==
    len(mot.fits)+1 == mot.n_frames``), matching dimensions/fps, and that the resolved Config's ``mesh``/
    ``rs_rows`` agree with the captured plan/motion array shapes.
    """
    ident = load_manifest_identity(plan_path, motion_path)
    plan, plan_meta = PSA.load_plan_npz(plan_path)          # raises PlanPayloadError (stable codes)
    mot, mot_meta = load_motion_npz(motion_path)            # raises MotionPayloadError (stable codes)
    try:
        cfg = S.Config(**ident["config"])
    except Exception as exc:
        raise CompositionAuditError("bad_config",
                                    f"cannot reconstruct Config from manifest config: {type(exc).__name__}: {exc}")
    gc, gr = int(cfg.mesh[0]), int(cfg.mesh[1])
    nr = max(int(cfg.rs_rows), 1)

    if (plan_meta["width"], plan_meta["height"]) != (mot_meta["width"], mot_meta["height"]):
        raise CompositionAuditError(
            "inconsistent_plan_motion",
            f"plan is {plan_meta['width']}x{plan_meta['height']} but motion is {mot_meta['width']}x"
            f"{mot_meta['height']}; the artifacts do not describe the same frame size")
    if plan_meta["fps"] != mot_meta["fps"]:
        raise CompositionAuditError(
            "inconsistent_plan_motion",
            f"plan fps {plan_meta['fps']!r} != motion fps {mot_meta['fps']!r}; the artifacts are not from the "
            f"same analysis")
    if int(mot.stride) != 1:
        raise CompositionAuditError(
            "unsupported_stride",
            f"the composed candidate requires stride=1, got motion stride {int(mot.stride)}; interpolated "
            f"paths are not composed here")
    if plan_meta["frames"] != int(mot.n_frames) or plan_meta["frames"] != len(mot.fits) + 1:
        raise CompositionAuditError(
            "mismatched_lengths",
            f"plan frames={plan_meta['frames']}, motion n_frames={int(mot.n_frames)} and "
            f"len(motion.fits)+1={len(mot.fits) + 1} must all agree (stride 1); refusing to compose mismatched "
            f"artifacts")
    if plan_meta["mesh_cells"] != [gc, gr]:
        raise CompositionAuditError(
            "inconsistent_config",
            f"manifest config mesh={[gc, gr]} does not match the plan mesh cells {plan_meta['mesh_cells']}")
    if mot_meta["mesh_cells"] != [max(gc, 1), max(gr, 1)]:
        raise CompositionAuditError(
            "inconsistent_config",
            f"manifest config mesh={[gc, gr]} implies fitted mesh cells {[max(gc, 1), max(gr, 1)]}, but the "
            f"motion mesh cells are {mot_meta['mesh_cells']}")
    if plan_meta["row_bands"] != nr or mot_meta["row_bands"] != nr:
        raise CompositionAuditError(
            "inconsistent_config",
            f"manifest config rs_rows={int(cfg.rs_rows)} implies {nr} row bands, but plan/motion report "
            f"{plan_meta['row_bands']}/{mot_meta['row_bands']}")
    meta = {"plan": plan_meta, "motion": mot_meta, "identity": ident, "config": ident["config"]}
    return plan, mot, cfg, meta


# --------------------------------------------------------------------------------------
# candidate evaluation on the plan_shape_audit probes
# --------------------------------------------------------------------------------------

def _line_metrics(mapped, p0, p1):
    """Best-fit-line straightness of already-mapped points plus the input line span (mirrors map_line_deviation)."""
    res = SA.best_fit_line(np.asarray(mapped, np.float64))
    res["input_span_px"] = float(np.linalg.norm(np.asarray(p1, np.float64) - np.asarray(p0, np.float64)))
    return res


def _displacement(a, b):
    """Pointwise Euclidean displacement of ``a`` from ``b`` (px); non-finite rows are dropped, not hidden."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = np.linalg.norm(a - b, axis=1)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"max_px": None, "mean_px": None, "n": 0}
    return {"max_px": float(d.max()), "mean_px": float(d.mean()), "n": int(d.size)}


def candidate_points(mot, plan, cfg, t, pts, mode, smoothed):
    """Sparse candidate output->source points for one mode (``smoothed`` keyed by high-pass mode)."""
    if mode == "exact_composition":
        return CC.candidate_source_coords(mot, plan, cfg, t, pts, highpass=False)
    if mode == "source_highpass":
        return CC.candidate_source_coords(mot, plan, cfg, t, pts, highpass=True,
                                          smoothed=smoothed["source_highpass"], smoothing_basis="source")
    if mode == "reference_highpass":
        return CC.candidate_source_coords(mot, plan, cfg, t, pts, highpass=True,
                                          smoothed=smoothed["reference_highpass"], smoothing_basis="reference")
    raise ValueError(f"unknown candidate mode {mode!r}")


def precompute_smoothed(mot, plan, cfg):
    """One temporal smoothing per high-pass basis for the whole clip (never recomputed per frame)."""
    return {
        "source_highpass": CC.smoothed_delta(mot, plan, cfg, smoothing_basis="source"),
        "reference_highpass": CC.smoothed_delta(mot, plan, cfg, smoothing_basis="reference"),
    }


def evaluate_segment(plan, mot, cfg, t, W, H, seg, samples, smoothed):
    """One output-space probe at one frame: production map vs the three candidate modes."""
    p0, p1 = np.asarray(seg["p0"], np.float64), np.asarray(seg["p1"], np.float64)
    pts = SA.sample_line(p0, p1, samples)
    prod = PSA.map_points(plan, t, W, H, pts)
    candidates = {}
    for mode in CANDIDATE_MODES:
        mapped = np.asarray(candidate_points(mot, plan, cfg, t, pts, mode, smoothed), np.float64)
        candidates[mode] = {
            "description": MODE_META[mode]["description"],
            "highpass": MODE_META[mode]["highpass"],
            "smoothing_basis": MODE_META[mode]["smoothing_basis"],
            "mapped": _line_metrics(mapped, p0, p1),
            "coverage": PSA.coverage(mapped, W, H),
            "displacement_from_production_px": _displacement(mapped, prod),
        }
    return {
        "input_line": {"p0": p0.tolist(), "p1": p1.tolist(), "orientation": seg["orientation"]},
        "input": SA.best_fit_line(pts),
        "production": {"mapped": _line_metrics(prod, p0, p1), "coverage": PSA.coverage(prod, W, H)},
        "candidates": candidates,
    }


def evaluate_frames(plan, mot, cfg, W, H, frames, segments, samples, smoothed):
    """Evaluate every probe and candidate mode for every requested frame, keeping the exact frame indices."""
    return [{
        "frame": int(t),
        "segments": {name: evaluate_segment(plan, mot, cfg, t, W, H, segments[name], samples, smoothed)
                     for name in PSA.SEGMENT_ORDER},
    } for t in frames]


def _mode_summary(per_frame, mode):
    """Per-segment, per-frame curvature/coverage/displacement series for production or one candidate mode."""
    per_segment = {}
    for name in PSA.SEGMENT_ORDER:
        if mode == "production":
            def block(pf):
                return pf["segments"][name]["production"]
        else:
            def block(pf):
                return pf["segments"][name]["candidates"][mode]
        dev = [block(pf)["mapped"]["dev_max_px"] for pf in per_frame]
        cov = [block(pf)["coverage"]["frac"] for pf in per_frame]
        valid = [d for d in dev if isinstance(d, (int, float))]
        entry = {
            "frames": len(per_frame),
            "mapped_dev_max_px": dev,
            "mapped_dev_max_px_max": max(valid) if valid else None,
            "coverage_frac_min": min(cov) if cov else None,
            "invalid_lines": sum(1 for d in dev if d is None),
        }
        if mode != "production":
            dm = [block(pf)["displacement_from_production_px"]["max_px"] for pf in per_frame]
            dmean = [block(pf)["displacement_from_production_px"]["mean_px"] for pf in per_frame]
            dvalid = [d for d in dm if isinstance(d, (int, float))]
            entry["displacement_from_production_max_px_max"] = max(dvalid) if dvalid else None
            entry["displacement_from_production_mean_px_max"] = max(
                [d for d in dmean if isinstance(d, (int, float))], default=None)
        per_segment[name] = entry
    return per_segment


def _summary(per_frame):
    production = _mode_summary(per_frame, "production")
    candidates = {mode: _mode_summary(per_frame, mode) for mode in CANDIDATE_MODES}
    overall_dev = [v["mapped_dev_max_px_max"] for v in production.values()
                   if v["mapped_dev_max_px_max"] is not None]
    overall_cov = [v["coverage_frac_min"] for v in production.values() if v["coverage_frac_min"] is not None]
    return {
        "production": production,
        "candidates": candidates,
        "production_max_mapped_dev_max_px": max(overall_dev) if overall_dev else None,
        "production_min_coverage_frac": min(overall_cov) if overall_cov else None,
        "any_production_line_invalid": any(v["invalid_lines"] for v in production.values()),
    }


# --------------------------------------------------------------------------------------
# report and CLI
# --------------------------------------------------------------------------------------

def run_real_composition_audit(plan_path, motion_path, *, every=30, samples=401):
    """Load a captured plan + motion, cross-check the manifest, and compare production with the candidate.

    Returns a JSON-safe report; see the module docstring for scope. Raises ``PlanPayloadError``,
    ``MotionPayloadError`` or ``CompositionAuditError`` on malformed/refused input.
    """
    every, samples = int(every), int(samples)
    if every < 1:
        raise ValueError("every must be >= 1")
    if samples < 2:
        raise ValueError("samples must be >= 2")
    plan, mot, cfg, meta = load_composition_inputs(plan_path, motion_path)
    W, H, T = meta["plan"]["width"], meta["plan"]["height"], meta["plan"]["frames"]
    segments = PSA.output_segments(W, H)
    frames = PSA.frame_indices(T, every)
    smoothed = precompute_smoothed(mot, plan, cfg)
    per_frame = evaluate_frames(plan, mot, cfg, W, H, frames, segments, samples, smoothed)
    return {
        "schema": SCHEMA,
        "ok": True,
        "scope": SCOPE_NOTE,
        "plan": meta["plan"],
        "motion": meta["motion"],
        "provenance": {
            "identity": meta["identity"],
            "config": meta["config"],
            "capture": {
                "checked_fields": ["outputs.plan", "outputs.motion",
                                   "experiment_identity.capture.dump_plan",
                                   "experiment_identity.capture.dump_motion"],
                "dump_plan": True, "dump_motion": True,
            },
        },
        "config": meta["config"],
        "candidate_modes": [{"mode": m, **MODE_META[m]} for m in CANDIDATE_MODES],
        "payload_safety": {
            "loader": "np.load(..., allow_pickle=False)",
            "plan_required_keys": list(PSA.REQUIRED_KEYS),
            "motion_required_keys": list(MOTION_REQUIRED_KEYS),
            "validated": ["schema", "required_keys", "dimensions", "array_shapes", "dtype_kind", "finiteness",
                          "stride_1", "matched_lengths", "config_vs_array_shapes"],
        },
        "selection": {"every": every, "samples": samples, "frames": frames,
                      "final_frame_included": frames[-1] == T - 1},
        "segments": segments,
        "coverage": PSA.COVERAGE_DOC,
        "metric": SA.METRIC_DOC,
        "per_frame": per_frame,
        "summary": _summary(per_frame),
        "no_composite_winner_score": (
            "production and the three candidate modes are reported side by side (per frame, segment and mode) "
            "as curvature, source-bounds coverage and pointwise displacement from production; this report "
            "deliberately does not collapse them into a single winner, ranking or quality score"),
        "limitations": [
            "experiment on real fitted motions only; not object/tree detection, scene understanding, perceptual "
            "evaluation, or evidence of perceived tree straightness",
            "no production integration: composition_candidate is not imported by stabilize/source_coords and no "
            "default or output is changed",
            "four straight output-space probes per frame, not the full field or real scene content",
            "mesh/row corrections are bilinear at the mesh resolution and sampled at the probe points",
            "high-pass modes use the candidate's own temporal smoothing; the smoothing chart is a decision "
            "experiment and no chart is selected",
            "no media is decoded and no frames are rendered; nothing about real-footage quality is claimed",
        ],
    }


def _error_payload(exc, plan_path=None, motion_path=None):
    payload = {
        "schema": SCHEMA,
        "ok": False,
        "error": {"code": exc.code, "message": str(exc)},
    }
    if plan_path is not None:
        payload["plan"] = {"path": str(plan_path)}
    if motion_path is not None:
        payload["motion"] = {"path": str(motion_path)}
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Captured plan+motion composition audit (production source_coords vs composed candidate; "
                    "no media, no rendering).")
    ap.add_argument("plan", nargs="?", help="path to a Plan NPZ written by mlxstab.stabilize.save_plan")
    ap.add_argument("motion", nargs="?", help="path to a Motion NPZ written by mlxstab.stabilize.save_motion_npz")
    ap.add_argument("--every", type=int, default=30, help="sample every n-th frame (final frame always added)")
    ap.add_argument("--samples", type=int, default=401, help="samples per output-space segment")
    ap.add_argument("--out", default=None, help="write the JSON report here (default: stdout)")
    args = ap.parse_args(argv)
    if args.every < 1:
        ap.error("--every must be >= 1")
    if args.samples < 2:
        ap.error("--samples must be >= 2")
    if args.plan is None or args.motion is None:
        ap.error("both a PLAN.npz and a MOTION.npz path are required")

    try:
        report = run_real_composition_audit(args.plan, args.motion, every=args.every, samples=args.samples)
    except (PSA.PlanPayloadError, MotionPayloadError, CompositionAuditError) as exc:
        text = json.dumps(json_safe(_error_payload(exc, args.plan, args.motion)), indent=1, sort_keys=True,
                          allow_nan=False)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        else:
            print(text)
        return 2

    text = json.dumps(json_safe(report), indent=1, sort_keys=True, allow_nan=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        s = report["summary"]
        print(f"plan={report['plan']['path']} motion={report['motion']['path']} frames={report['selection']['frames']} "
              f"production max dev={s['production_max_mapped_dev_max_px']} px "
              f"min coverage={s['production_min_coverage_frac']}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
