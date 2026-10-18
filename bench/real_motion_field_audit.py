"""Captured real motion-field audit: per-pair similarity rotation/scale and row/mesh residual magnitudes.

This is the **motion-only** companion to ``bench.real_composition_audit``. It loads one or more benchmark
``manifest.json`` files, proves each run succeeded and captured a motion NPZ (``outputs.motion``), checks that
the loaded motion matches the manifest's recorded source identity (decoded size/fps), refuses to mix manifests
from different sources, and then summarises the captured per-pair fitted motion (``mlxstab.motion.fit_motion``
fields serialized by ``mlxstab.stabilize.save_motion_npz``) without decoding any media or rendering any frame.

For every analysed frame pair it reports:

* the best-fit **image-y-down similarity** rotation angle (visual clockwise positive) and scale extracted from
  the global homography ``H`` (frame ``t`` -> ``t+1``);
* the **row residual** and **mesh residual** displacement magnitudes, each with RMS/median/p95/max;
* the mesh residual **vector and magnitude arrays** (the per-vertex grid) and the per-cell **spatial grid**
  (one averaged displacement vector and magnitude per mesh cell) with a cautious dispersion/neighbour-
  coherence proxy;
* the per-vertex and per-band **support mass** (observation coverage) and the residual magnitudes split into
  directly supported (mass >= 3.0) and diffusion-filled (mass < 3.0) groups;
* for schema-`/3` motion, the forward-backward **cycle-consistency maps** on the mesh grid
  (cycle-error RMS px, tolerance-failure fraction in [0,1] and sample mass), per pair and pooled per case,
  reported as null for manually constructed and legacy `/1`/`/2` fits;
* the per-pair **row-band line profile**: an unweighted least-squares line fit of both components of
  ``Fit.rows`` (an x-independent displacement per row band) against the row-band centre y positions
  ``(j + 0.5)*height/nr - 0.5``, reporting availability, band count, the slope vector (px per px row), the
  fitted first-to-last-band displacement delta, the pooled nonlinear residual RMS and a multivariate
  linear-explained fraction, with per-case descriptive statistics and per-pair series;
* the fit ``inlier_frac`` / ``mover_frac`` / ``fallback`` metadata and per-case aggregates, plus the
  ``mlxstab.motion/4`` and ``/5`` gate diagnostics ``n_valid_samples``, ``n_ransac_inliers`` and the numeric payload
  ``fallback_reason_code`` mapped back to the human-readable ``fallback_reason`` label per pair and their counts
  per case. Explicit ``sequence_boundary`` placeholders are reported separately from failed geometric fits. A
  legacy ``/1``-``/3`` payload predates
  these fields, so they are reported as ``null`` with ``fit_metadata_available=false`` rather than invented.

Honesty scope
-------------
Row and mesh residuals are **parallax-like spatial motion evidence only**: the displacement left after the
global (and row) model. They also contain model mismatch, optical-flow error and rolling-shutter leftovers, so
a large residual is *not* a depth measurement. The per-cell dispersion/neighbour-coherence proxy is
descriptive only: a coherent residual is equally consistent with parallax and with a smooth model mismatch, so
it is not depth evidence either. This audit reports no metric depth, no object semantics and no
perceptual quality: recovering depth would require camera intrinsics **plus** camera motion or an external
depth source, none of which is present here. The cycle-consistency maps are likewise a forward-backward
**consistency** diagnostic only: a consistent cycle does not prove the correspondence is correct and an
inconsistent cycle does not locate the error, so they are not true correspondence error, not ground-truth
error, not a probability and not calibrated uncertainty; no correlation with the residual maps is computed or
implied. The row-band line profile is **not** a rolling-shutter detector and **not** a physical readout
estimate: a smooth vertical row displacement is rolling-shutter-*like*, but it can equally arise from other
motion or model mismatch, and no camera readout time, focal calibration or ground-truth readout is inferred.
The source windows of different runs are typically inferred from
historical filenames and may be nested; this audit records them but never treats them as a formal selection.
The ``fallback_reason`` labels the first failing ``fit_motion`` gate, or ``sequence_boundary`` when a caller
explicitly skips a pair across a cut. Boundary placeholders are counted separately from failed geometric fits.
These labels are not quality judgements: they do not say whether the footage is stable or whether the pixels are
correct. No production code, default or
output is changed and no media is decoded.

Payload safety
--------------
Motion NPZs are opened with ``np.load(..., allow_pickle=False)`` and validated by
``bench.real_composition_audit.load_motion_npz`` (schema, required keys, shapes, dtype kinds, finiteness); no
pickle is ever loaded. Manifests are parsed as JSON only. Every report is passed through ``json_safe`` and
serialized with ``allow_nan=False`` so non-finite values never reach the JSON.

Run with::

    python -m bench.real_motion_field_audit MANIFEST.json [MANIFEST.json ...] [--out file.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from mlxstab.motion import (FALLBACK_REASONS, FALLBACK_REASON_SEQUENCE_BOUNDARY,
                            SUPPORT_THRESHOLD)

from . import real_composition_audit as RCA
from .manifest import json_safe

SCHEMA = "mlxstab.real_motion_field_audit/5"

# Manifest identity fields that must be exactly equal across manifests before their motion fields are reported
# together. The source window is deliberately *not* here: the t060/t800 windows are nested and inferred, and
# the audit records them per case instead of demanding they match. The decoded check uses the normalized
# ``decoded_identity`` (width/height/fps only), not the full ``decoded`` block: a longer source window yields
# a larger ``decoded.frames``/``decoded.duration_s`` for the *same* source and must not be refused as a
# different source. The full decoded block is still retained per manifest for provenance.
_MATCH_CHECKS = (
    ("source_sha256", "mismatched_source", "source sha256"),
    ("preprocessing", "mismatched_preprocessing", "preprocessing"),
    ("weights_sha256", "mismatched_weights", "weights sha256"),
    ("decoded_identity", "mismatched_decode", "decoded width/height/fps"),
)

SCOPE_NOTE = (
    "Isolated real motion-field audit over captured production motion NPZs. Loads each benchmark "
    "manifest.json, proves the run succeeded and captured outputs.motion, loads the motion with "
    "allow_pickle=False, and cross-checks it against the manifest's recorded source identity (decoded size/fps "
    "and source sha256). It reports, per analysed frame pair, the best-fit image-y-down similarity rotation "
    "angle and scale from the global homography H, the row and mesh residual displacement magnitude "
    "RMS/median/p95/max, the per-vertex and per-cell mesh residual vector/magnitude arrays, a cautious "
    "spatial dispersion/neighbour-coherence proxy, and the fit inlier/mover/fallback metadata. Row/mesh "
    "residuals are parallax-like spatial motion evidence only: they also contain model mismatch, optical-flow "
    "error and rolling-shutter leftovers, and they are NOT metric depth, object semantics or perceptual "
    "quality. The spatial proxy is descriptive and equally consistent with model mismatch as with parallax; "
    "recovering depth needs camera intrinsics plus camera motion or an external depth source, neither of which "
    "is present. It also records the per-vertex and per-band observation-coverage support mass and splits the "
    "residual magnitudes into directly supported (mass >= 3.0) and diffusion-filled (mass < 3.0) groups; "
    "support mass is a non-robust kernel-weighted count of usable non-mover flow samples, NOT a probability and "
    "NOT flow confidence. For schema-/3 motion it additionally reports forward-backward cycle-consistency maps "
    "(spatially weighted cycle-error RMS, tolerance-failure fraction and sample mass on the mesh grid), per "
    "pair and pooled per case; these are forward-backward consistency diagnostics only, NOT true correspondence "
    "error, NOT a probability and NOT calibrated uncertainty. It also reports a per-pair row-band line "
    "profile: an unweighted least-squares line fit of both components of Fit.rows against the row-band centre "
    "y positions, with availability, band count, slope vector (px per px row), fitted first-to-last-band "
    "delta, pooled nonlinear residual RMS and a multivariate linear-explained fraction, aggregated per case "
    "without forming a score. A smooth vertical row displacement is rolling-shutter-LIKE but can also arise "
    "from other motion or model mismatch, so this is NOT a rolling-shutter detector and NOT a physical readout "
    "estimate: no camera readout time, focal calibration or ground-truth readout is inferred. No media is "
    "decoded, no frames are rendered, no production code/default/output is changed, and no composite/winner "
    "score is produced. It also reports the mlxstab.motion/4 and /5 fit diagnostics: per-pair n_valid_samples and "
    "n_ransac_inliers and the numeric fallback_reason_code mapped back to the stable human-readable "
    "fallback_reason label for a passed fit, the first failing gate, or an explicit sequence boundary. Boundary "
    "placeholders are counted separately from failed fits. Per-case reason counts are descriptive and not a "
    "quality score; legacy /1-/3 payloads predate fit metadata and are reported as "
    "null with fit_metadata_available=false."
)

LIMITATIONS = [
    "row/mesh residuals are parallax-like spatial motion evidence only; they also contain model mismatch, "
    "optical-flow error and rolling-shutter leftovers",
    "no metric depth: a residual magnitude is not a distance and no depth ordering is inferred",
    "no object semantics and no scene understanding; residual blobs are not detections or segmentations",
    "the spatial dispersion/coherence proxy is descriptive and in-sample; a coherent residual is consistent "
    "with parallax AND with smooth model mismatch or rolling-shutter leftovers, so it is not depth evidence",
    "depth recovery would need camera intrinsics plus camera motion or an external depth source, absent here",
    "no perceptual quality evaluation and no claim about what a viewer sees",
    "one inferred (often nested) source window per run is not a clip-wide or held-out claim",
    "support mass is an observation-coverage diagnostic (a non-robust kernel-weighted count of usable "
    "non-mover flow samples), not a probability and not flow confidence, and the supported-vs-filled split "
    "says where the fit was directly observed, not how accurate the residual is",
    "the cycle-consistency maps are a forward-backward consistency diagnostic only: a consistent cycle does "
    "not prove the correspondence is correct and an inconsistent cycle does not locate the error, so they are "
    "not true correspondence error, not ground-truth error, not a probability and not calibrated uncertainty",
    "no correlation between the cycle maps and the residual maps is computed or implied; a visual or numeric "
    "association would not establish that the cycle is correct or that the residual is parallax",
    "the row-band line profile is rolling-shutter-like descriptive evidence only: a smooth vertical row "
    "displacement can also arise from other motion or model mismatch, so it is not a rolling-shutter detector "
    "and infers no camera readout time, focal calibration or ground-truth readout; it is not combined with the "
    "cycle maps or any other field into a score or correlation",
    "the row-band line fit is unweighted and does not use the support masses, so legacy /1 (no support) and "
    "/2 and /3 (with support) Fits share the same definition; slope/delta/explained-fraction are null when "
    "they are not identifiable (a constant field, or fewer than two row bands)",
    "no media is decoded and no frames are rendered; the audit is of the fitted motion arrays alone",
    "the per-pair fallback reason is a label for the existing fit_motion refusal gates (first failing gate wins); "
    "it says which sample/inlier threshold produced an identity fallback, not whether the fit is good or the "
    "pixels are correct, and it is null when the payload predates mlxstab.motion/4",
]


class MotionFieldAuditError(ValueError):
    """A manifest/motion pair was refused. ``code`` is a stable machine-readable reason."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = str(code)


# --------------------------------------------------------------------------------------
# manifest loading and provenance
# --------------------------------------------------------------------------------------

def _canonical(value):
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _manifest_identity(man, path):
    """Pull the provenance identity out of a benchmark manifest, validating the fields the audit needs."""
    def fail(message):
        raise MotionFieldAuditError("manifest_malformed", f"manifest {str(path)!r}: {message}")

    if not isinstance(man, dict):
        fail("top level must be a JSON object")
    case = man.get("case")
    if not isinstance(case, dict) or not isinstance(case.get("name"), str) or not case["name"].strip():
        fail("missing or empty case.name")
    status = man.get("status")
    if not isinstance(status, str) or not status:
        fail("missing status")
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
    decoded = source.get("decoded")
    if not isinstance(decoded, dict):
        fail("missing source.decoded object (needed to match the motion dimensions/fps)")
    for key in ("width", "height", "fps"):
        if not isinstance(decoded.get(key), (int, float)) or isinstance(decoded.get(key), bool):
            fail(f"missing numeric source.decoded.{key}")
    # Normalized identity used for cross-manifest matching: only the fields that describe the decoded frame
    # geometry/rate. ``decoded.frames``/``decoded.duration_s`` follow the *window* length for a given source
    # and therefore are provenance, not cross-run identity.
    decoded_identity = {"width": decoded["width"], "height": decoded["height"], "fps": decoded["fps"]}
    weights = man.get("weights")
    if not isinstance(weights, dict) or not isinstance(weights.get("sha256"), str) or not weights["sha256"]:
        fail("missing weights.sha256")
    outputs = man.get("outputs")
    if not isinstance(outputs, dict):
        fail("missing outputs object")
    outputs_motion = outputs.get("motion")
    if not isinstance(outputs_motion, str) or not outputs_motion.strip():
        fail("missing or empty outputs.motion (the manifest must record the captured motion path)")
    experiment = man.get("experiment_identity")
    capture = experiment.get("capture") if isinstance(experiment, dict) else None
    capture_dump_motion = bool(isinstance(capture, dict) and capture.get("dump_motion") is True)

    return {
        "case": case["name"],
        "run_id": man.get("run_id"),
        "manifest_schema": man.get("schema"),
        "status": status,
        "source_sha256": sha,
        "source_window": {"start_s": window.get("start_s"), "duration_s": window.get("duration_s")},
        "preprocessing": json_safe(source.get("preprocessing")),
        "decoded": json_safe(decoded),
        "decoded_identity": json_safe(decoded_identity),
        "weights_sha256": weights["sha256"],
        "outputs_motion": outputs_motion,
        "capture_dump_motion": capture_dump_motion,
        "manifest_path": str(path),
    }


def resolve_motion_path(manifest_path, outputs_motion):
    """Resolve a manifest ``outputs.motion`` entry to an existing file, or ``None``.

    The runner records the capture path relative to the repository root (its CWD at run time), so a relative
    entry is tried against the CWD first, then against the manifest directory, then as a sibling basename. An
    absolute entry is used as-is. This never *assumes* a path: a miss returns ``None`` and the caller refuses.
    """
    p = Path(outputs_motion)
    cands = []
    if p.is_absolute():
        cands.append(p)
    else:
        cands.append(Path.cwd() / p)
        cands.append(Path(manifest_path).parent / p)
        cands.append(Path(manifest_path).parent / p.name)
    for c in cands:
        if c.exists():
            return c
    return None


def load_manifest_motion(manifest_path):
    """Load, validate and cross-check one benchmark manifest and its captured motion.

    Returns ``(identity, motion, motion_meta)``. A malformed manifest raises
    :class:`MotionFieldAuditError` with a stable ``code``; the motion payload itself is validated by
    :func:`bench.real_composition_audit.load_motion_npz`, and a motion that does not match the manifest's
    recorded source size/fps is refused as ``motion_identity_mismatch``.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise MotionFieldAuditError("manifest_missing", f"no benchmark manifest at {str(path)!r}")
    try:
        man = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # malformed JSON or unreadable file
        raise MotionFieldAuditError("manifest_unreadable", f"cannot read/parse manifest {str(path)!r}: {exc}")

    ident = _manifest_identity(man, path)
    if ident["status"] != "success":
        raise MotionFieldAuditError(
            "manifest_not_success",
            f"manifest {str(path)!r} (case {ident['case']!r}) has status {ident['status']!r}, not 'success'; "
            f"refusing to audit a run that did not succeed")

    motion_path = resolve_motion_path(path, ident["outputs_motion"])
    if motion_path is None:
        raise MotionFieldAuditError(
            "motion_missing",
            f"manifest {str(path)!r} records outputs.motion {ident['outputs_motion']!r} but no such file exists "
            f"relative to the CWD or beside the manifest")
    ident["motion_path"] = str(motion_path)

    try:
        mot, meta = RCA.load_motion_npz(motion_path)       # raises MotionPayloadError (stable codes)
    except RCA.MotionPayloadError as exc:
        raise MotionFieldAuditError(f"motion_{exc.code}",
                                    f"motion {str(motion_path)!r} (case {ident['case']!r}): {exc}")

    dec = ident["decoded"]
    if meta["width"] != dec["width"] or meta["height"] != dec["height"]:
        raise MotionFieldAuditError(
            "motion_identity_mismatch",
            f"motion {str(motion_path)!r} is {meta['width']}x{meta['height']} but manifest "
            f"{str(path)!r} records source.decoded {dec['width']}x{dec['height']}; the motion does not describe "
            f"this run's source")
    if not math.isclose(float(meta["fps"]), float(dec["fps"]), rel_tol=1e-9, abs_tol=1e-6):
        raise MotionFieldAuditError(
            "motion_identity_mismatch",
            f"motion {str(motion_path)!r} has fps {meta['fps']!r} but manifest {str(path)!r} records "
            f"source.decoded.fps {dec['fps']!r}; the motion does not describe this run's source")
    return ident, mot, meta


def _require_matched(identities):
    """Refuse manifests whose recorded source identity does not match the reference, before comparing fields.

    Source sha256, preprocessing, weights sha256 and decoded size/fps must be exactly equal. The source window
    is deliberately excluded: the compared runs use different (often nested, inferred) windows and the report
    records each window per case rather than pretending they are the same selection. Case names are also
    deliberately not required to be unique: the t060/t800 baseline captures share the case name ``baseline``
    and stay distinguishable by ``run_id``, ``manifest_path`` and ``source_window``, so a repeated name is not
    a refusal.
    """
    ref = identities[0]
    for ident in identities[1:]:
        for key, code, label in _MATCH_CHECKS:
            if _canonical(ident[key]) != _canonical(ref[key]):
                raise MotionFieldAuditError(
                    code,
                    f"manifest {ident['manifest_path']!r} (case {ident['case']!r}) has a different {label} "
                    f"than manifest {ref['manifest_path']!r} (case {ref['case']!r}); refusing to report motion "
                    f"fields across unproven sources")


# --------------------------------------------------------------------------------------
# similarity extraction and residual statistics
# --------------------------------------------------------------------------------------

def similarity_from_h(H, width, height):
    """Best-fit image-y-down similarity ``(rotation_deg_y_down, scale)`` of a global homography ``H``.

    The homography is applied to an inset grid of source points spanning the frame and a least-squares
    similarity (``mlxstab.geometry.fit_model``) is fit to those mapped points, so a full projective ``H`` still
    yields a representative rotation/scale rather than a single-pixel Jacobian. The angle is extracted from the
    fitted linear part ``A = [[s cos, -s sin], [s sin, s cos]]`` as ``atan2(A[1, 0], A[0, 0])``: with image y
    pointing down this is the visual **clockwise** rotation (positive rotates right-of-centre content toward
    the bottom), matching ``bench.composition_audit.rotation_about_clockwise``. ``scale`` is the non-negative
    ``hypot(A[0, 0], A[1, 0])``.
    """
    Hm = np.asarray(H, np.float64)
    W, Hh = float(width), float(height)
    if W < 2.0 or Hh < 2.0:
        raise ValueError("width/height must be >= 2")
    m = max(1.0, min(W, Hh) / 8.0)
    xs = np.array([m, W - 1.0 - m, W / 2.0])
    ys = np.array([m, Hh - 1.0 - m, Hh / 2.0])
    gx, gy = np.meshgrid(xs, ys)
    src = np.stack([gx.ravel(), gy.ravel()], 1)
    from mlxstab import geometry as G
    dst = G.apply_h(Hm, src)
    sim = G.fit_model(src, dst, "similarity")
    a = sim[0, 0] + 1j * sim[1, 0]
    return float(np.degrees(np.angle(a))), float(abs(a))


def _stats(values):
    """Finite RMS/mean/median/p95/min/max/abs-max of a 1-D series; all ``None`` when empty."""
    a = np.asarray(values, np.float64).ravel()
    if a.size == 0:
        return {"n": 0, "mean": None, "rms": None, "median": None, "p95": None,
                "min": None, "max": None, "abs_max": None}
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "rms": float(np.sqrt(np.mean(a ** 2))),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "abs_max": float(np.max(np.abs(a))),
    }


def _magnitude_stats(vectors):
    """Magnitude RMS/median/p95/max of rows of a ``[..., 2]`` displacement field."""
    v = np.asarray(vectors, np.float64)
    mags = np.linalg.norm(v, axis=-1).ravel()
    return _stats(mags)


def _vector_magnitude(vec):
    """Euclidean magnitude of a 2-vector, or ``None`` when the vector itself is ``None``."""
    if vec is None:
        return None
    return float(math.hypot(float(vec[0]), float(vec[1])))


def row_line_profile(rows, height):
    """Unweighted least-squares line profile of the x-independent row-band displacement ``Fit.rows``.

    ``rows`` is ``[nr, 2]`` (the x-independent residual displacement per row band) and the band centres are
    ``(j + 0.5) * height / nr - 0.5``, the same y positions ``mlxstab.motion.fit_motion`` used. Each component
    is fit independently with an ordinary intercept-included least-squares line against the band centre. The
    fit deliberately **does not** use the support masses, so legacy ``mlxstab.motion/1`` Fits (no support) and
    ``/2``\ //``/3`` Fits (with support) keep exactly the same definition.

    Returns a JSON-safe dict with the availability (``nr >= 2``), the band count, the slope vector in px
    displacement per px row, the fitted first-to-last-band displacement delta in px, the pooled nonlinear
    residual RMS in px around the line, and the pooled multivariate linear-explained fraction
    (``1 - RSS/TSS`` with per-component means over the bands). Slope, delta and the fraction are ``None`` when
    they are not identifiable (fewer than 2 bands, or a constant field whose ``TSS = 0``); a constant or
    one-band field stays finite-safe (zero residual, null fraction) rather than emitting ``nan``/``inf``.

    A smooth vertical row displacement is rolling-shutter-*like*, but it can equally arise from other motion or
    model mismatch. This is **not** a rolling-shutter detector and **not** a physical readout estimate; no
    camera readout time, focal calibration or ground-truth readout is inferred.
    """
    R_ = np.asarray(rows, np.float64)
    if R_.ndim != 2 or R_.shape[1] != 2:
        raise ValueError("rows must be a [nr, 2] array")
    nr = int(R_.shape[0])
    profile = {
        "available": nr >= 2,
        "n_bands": nr,
        "slope_px_per_row": None,
        "delta_first_to_last_px": None,
        "nonlinear_residual_rms_px": 0.0,
        "linear_explained_fraction": None,
    }
    if nr < 2:
        return profile
    centers = (np.arange(nr, dtype=np.float64) + 0.5) * float(height) / float(nr) - 0.5
    design = np.stack([centers, np.ones(nr, np.float64)], axis=1)      # [nr, 2]: slope, intercept
    coeff, *_ = np.linalg.lstsq(design, R_, rcond=None)
    slope = coeff[0]
    resid = R_ - design @ coeff
    profile["slope_px_per_row"] = slope.tolist()
    profile["delta_first_to_last_px"] = (slope * float(centers[-1] - centers[0])).tolist()
    profile["nonlinear_residual_rms_px"] = float(np.sqrt(np.mean(resid ** 2)))
    tss = float(np.sum((R_ - R_.mean(axis=0, keepdims=True)) ** 2))
    if tss > 0.0:
        profile["linear_explained_fraction"] = float(1.0 - np.sum(resid ** 2) / tss)
    return profile


def support_split(magnitudes, support_mass, threshold=SUPPORT_THRESHOLD):
    """Split residual magnitudes into directly supported vs diffusion-filled groups by support mass.

    ``support_mass`` is the fitted observation-coverage diagnostic (a non-robust kernel-weighted count of
    usable non-mover flow samples at each vertex/band, recorded before diffusion), NOT a probability and NOT
    flow confidence. Locations with ``mass >= threshold`` were directly observed; ``mass < threshold`` were
    filled by ``mlxstab.motion._fill`` from their observed neighbours. Returns ``None`` when the mass is
    unavailable (a legacy ``mlxstab.motion/1`` payload).
    """
    if support_mass is None:
        return None
    mags = np.asarray(magnitudes, np.float64).ravel()
    mass = np.asarray(support_mass, np.float64).ravel()
    if mass.size != mags.size:
        raise ValueError(f"support_mass size {mass.size} does not match magnitudes size {mags.size}")
    ok = mass >= threshold
    return {
        "threshold": float(threshold),
        "n_supported": int(np.count_nonzero(ok)),
        "n_filled": int(np.count_nonzero(~ok)),
        "supported_fraction": float(np.count_nonzero(ok) / mass.size),
        "supported_residual_magnitude_px": _stats(mags[ok]),
        "filled_residual_magnitude_px": _stats(mags[~ok]),
    }


def _cycle_maps(fit):
    """The ``(err_rms, inconsistency_frac, sample_mass)`` cycle maps of a Fit, or ``(None, None, None)``.

    Manually constructed and legacy/2 Fits carry no cycle maps, so the caller must report them as unavailable
    rather than as zeros."""
    rms = getattr(fit, "cycle_err_rms", None)
    frac = getattr(fit, "cycle_inconsistency_frac", None)
    mass = getattr(fit, "cycle_sample_mass", None)
    if rms is None or frac is None or mass is None:
        return None, None, None
    return (np.asarray(rms, np.float64), np.asarray(frac, np.float64), np.asarray(mass, np.float64))


def _fit_metadata(fit):
    """``(n_valid_samples, n_ransac_inliers, fallback_reason, available)`` for one Fit.

    Returns ``(None, None, None, False)`` when the Fit carries no gate metadata (a manually constructed Fit, or
    a legacy ``mlxstab.motion/1``/``/2``/``/3`` payload), so the audit reports the values as null instead of
    inventing counts or a reason."""
    n_valid = getattr(fit, "n_valid_samples", None)
    n_inliers = getattr(fit, "n_ransac_inliers", None)
    reason = getattr(fit, "fallback_reason", None)
    if n_valid is None or n_inliers is None or reason is None:
        return None, None, None, False
    return int(n_valid), int(n_inliers), str(reason), True


def cycle_pair_summary(err_rms, inconsistency_frac, sample_mass):
    """Per-pair summary of the forward-backward cycle-consistency maps (all-null when unavailable).

    Vertices with zero sample mass carry no sample to judge and are excluded from the RMS/fraction statistics.
    ``inconsistency_fraction_mass_weighted`` is the sample-mass-weighted mean of the per-vertex fractions over
    vertices with samples (a single pooled scalar), and ``sample_mass_total`` sums the spatial weights. These
    are forward-backward **cycle-consistency** summaries only, not true correspondence error, not probability
    and not calibrated uncertainty."""
    if err_rms is None or inconsistency_frac is None or sample_mass is None:
        return {"available": False, "n_vertices_with_samples": 0,
                "error_rms_px": _stats([]), "inconsistency_fraction": _stats([]),
                "inconsistency_fraction_mass_weighted": None, "sample_mass_total": None,
                "sample_mass": _stats([])}
    rms = np.asarray(err_rms, np.float64).ravel()
    frac = np.asarray(inconsistency_frac, np.float64).ravel()
    mass = np.asarray(sample_mass, np.float64).ravel()
    sel = mass > 0.0
    total = float(mass.sum())
    weighted = float(np.sum(mass[sel] * frac[sel]) / total) if total > 0.0 else None
    return {"available": True, "n_vertices_with_samples": int(np.count_nonzero(sel)),
            "error_rms_px": _stats(rms[sel]), "inconsistency_fraction": _stats(frac[sel]),
            "inconsistency_fraction_mass_weighted": weighted, "sample_mass_total": total,
            "sample_mass": _stats(mass[sel])}


def mesh_cell_vectors(mesh):
    """Average residual vector of each mesh **cell** ``(gr, gc, 2)`` (mean of the cell's four vertices).

    The stored mesh is a per-vertex field ``[gr+1, gc+1, 2]``; a cell's vector is the mean of its four corner
    vertex residuals, i.e. the residual at the cell centre. This is a transparent spatial summary; it is a
    displacement, not a depth.
    """
    m = np.asarray(mesh, np.float64)
    return (m[:-1, :-1] + m[:-1, 1:] + m[1:, :-1] + m[1:, 1:]) / 4.0


def spatial_proxy(cell_vectors):
    """Cautious dispersion/coherence proxy of a per-cell residual field ``(gr, gc, 2)``.

    ``dispersion_rms_px`` is the RMS distance of the cell vectors from their spatial mean, so a global uniform
    shift is removed and only the *spatially varying* part contributes. ``neighbor_cosine_weighted_mean`` is
    the magnitude-product-weighted mean cosine between the centred cell vectors of 4-neighbour cells, over
    pairs whose centred magnitudes exceed a small epsilon (~1 = neighbouring cells move in the same centred
    direction; ~0 = incoherent; ``None`` when no such pair exists). Both are descriptive and finite-safe.

    This is **not** depth: a coherent residual is equally consistent with parallax, a smooth model mismatch and
    a rolling-shutter pattern; the proxy only says how non-uniform and how locally aligned the fitted field is.
    """
    v = np.asarray(cell_vectors, np.float64)
    if v.ndim != 3 or v.shape[-1] != 2:
        raise ValueError("cell_vectors must be [gr, gc, 2]")
    mean_vec = v.reshape(-1, 2).mean(0)
    centered = v - mean_vec
    flat = centered.reshape(-1, 2)
    dispersion = float(np.sqrt(np.mean(np.sum(flat ** 2, axis=1)))) if flat.size else 0.0

    num = 0.0
    den = 0.0
    pairs = 0
    for a, b in ((centered[:, :-1], centered[:, 1:]), (centered[:-1, :], centered[1:, :])):
        na = np.linalg.norm(a, axis=-1).ravel()
        nb = np.linalg.norm(b, axis=-1).ravel()
        dot = np.sum(a * b, axis=-1).ravel()
        ok = (na > 1e-12) & (nb > 1e-12)
        if np.any(ok):
            w = (na * nb)[ok]
            num += float(np.sum(w * (dot[ok] / (na[ok] * nb[ok]))))
            den += float(np.sum(w))
            pairs += int(np.count_nonzero(ok))
    coherence = (num / den) if den > 1e-12 else None
    return {
        "n_cells": int(v.shape[0] * v.shape[1]),
        "mean_vector_px": mean_vec.tolist(),
        "dispersion_rms_px": dispersion,
        "neighbor_cosine_weighted_mean": coherence,
        "neighbor_pairs": pairs,
    }


def summarize_pair(fit, *, pair, frame_index, width, height):
    """One analysed pair: similarity from ``H``, row/mesh magnitude stats, mesh arrays and fit metadata."""
    angle_deg, scale = similarity_from_h(fit.H, width, height)
    mesh_vec = np.asarray(fit.mesh, np.float64)
    row_vec = np.asarray(fit.rows, np.float64)
    cell_vec = mesh_cell_vectors(mesh_vec)
    mesh_mag = np.linalg.norm(mesh_vec, axis=-1)
    row_mag = np.linalg.norm(row_vec, axis=-1)
    mesh_mass = getattr(fit, "mesh_support_mass", None)
    row_mass = getattr(fit, "row_support_mass", None)
    cyc_rms, cyc_frac, cyc_mass = _cycle_maps(fit)
    n_valid_samples, n_ransac_inliers, fallback_reason, fit_meta_available = _fit_metadata(fit)
    return {
        "pair": int(pair),
        "frame_index": int(frame_index),
        "rotation_deg_y_down": angle_deg,
        "scale": scale,
        "inlier_frac": float(fit.inlier_frac),
        "mover_frac": float(fit.mover_frac),
        "fallback": bool(fit.fallback),
        "n_valid_samples": n_valid_samples,
        "n_ransac_inliers": n_ransac_inliers,
        "fallback_reason": fallback_reason,
        "fit_metadata_available": fit_meta_available,
        "mesh_vertex_residual_vectors_px": mesh_vec.tolist(),
        "mesh_vertex_residual_magnitudes_px": mesh_mag.tolist(),
        "mesh_vertex_support_mass": (None if mesh_mass is None
                                     else np.asarray(mesh_mass, np.float64).tolist()),
        "row_band_support_mass": (None if row_mass is None
                                  else np.asarray(row_mass, np.float64).tolist()),
        "mesh_cycle_error_rms_px": (None if cyc_rms is None else cyc_rms.tolist()),
        "mesh_cycle_inconsistency_fraction": (None if cyc_frac is None else cyc_frac.tolist()),
        "mesh_cycle_sample_mass": (None if cyc_mass is None else cyc_mass.tolist()),
        "cycle_consistency": cycle_pair_summary(cyc_rms, cyc_frac, cyc_mass),
        "support_threshold": float(SUPPORT_THRESHOLD),
        "mesh_supported_vs_filled": support_split(mesh_mag, mesh_mass),
        "row_supported_vs_filled": support_split(row_mag, row_mass),
        "mesh_residual_magnitude_px": _magnitude_stats(mesh_vec),
        "row_residual_magnitude_px": _magnitude_stats(row_vec),
        "row_line_profile": row_line_profile(row_vec, height),
        "spatial_grid": {
            "cell_vectors_px": cell_vec.tolist(),
            "cell_magnitudes_px": np.linalg.norm(cell_vec, axis=-1).tolist(),
            "proxy": spatial_proxy(cell_vec),
        },
    }


def _case_summary(pairs, mesh_mag, row_mag, mesh_supported=None, mesh_filled=None,
                  row_supported=None, row_filled=None):
    """Per-case aggregates over every pair: angle/scale/inlier/mover series and pooled residual magnitudes.

    ``mesh_mag``/``row_mag`` are already-flattened magnitude series pooled over every vertex/row of every pair.
    ``*_supported``/``*_filled`` (optional) are the pooled magnitudes of directly observed vs diffusion-filled
    vertices/bands, split at ``SUPPORT_THRESHOLD`` from each pair's observation-coverage support mass.
    """
    mesh_mag = np.asarray(mesh_mag, np.float64).ravel()
    row_mag = np.asarray(row_mag, np.float64).ravel()

    def _cat(chunks):
        return np.concatenate(chunks) if chunks else np.zeros(0)

    sequence_boundary_pairs = [p["pair"] for p in pairs
                               if p["fallback_reason"] == FALLBACK_REASON_SEQUENCE_BOUNDARY]
    fallback_pairs = [p["pair"] for p in pairs
                      if p["fallback"] and p["fallback_reason"] != FALLBACK_REASON_SEQUENCE_BOUNDARY]
    # Per-case counts of the stable fit-gate reasons. "unavailable" collects fits whose payload predates the
    # mlxstab.motion/4-/5 diagnostics (legacy /1-/3), so an absent reason is never silently counted as "none".
    fallback_reason_counts = {r: 0 for r in FALLBACK_REASONS}
    fallback_reason_counts["unavailable"] = 0
    fallback_reason_unavailable_pairs = []
    for p in pairs:
        if not p["fit_metadata_available"]:
            fallback_reason_counts["unavailable"] += 1
            fallback_reason_unavailable_pairs.append(p["pair"])
        else:
            fallback_reason_counts[p["fallback_reason"]] += 1
    mesh_fractions = [p["mesh_supported_vs_filled"]["supported_fraction"] for p in pairs
                      if p["mesh_supported_vs_filled"] is not None]
    row_fractions = [p["row_supported_vs_filled"]["supported_fraction"] for p in pairs
                     if p["row_supported_vs_filled"] is not None]
    # Cycle maps, pooled only over the pairs where they are available (schema /3); zero-sample vertices are
    # excluded, exactly as in the per-pair summary.
    cycle_available = [bool(p["cycle_consistency"]["available"]) for p in pairs]
    cyc_rms, cyc_frac, cyc_mass = [], [], []
    for p in pairs:
        mass = p["mesh_cycle_sample_mass"]
        if mass is None:
            continue
        mass = np.asarray(mass, np.float64).ravel()
        sel = mass > 0.0
        cyc_rms.append(np.asarray(p["mesh_cycle_error_rms_px"], np.float64).ravel()[sel])
        cyc_frac.append(np.asarray(p["mesh_cycle_inconsistency_fraction"], np.float64).ravel()[sel])
        cyc_mass.append(mass[sel])
    cyc_rms = _cat(cyc_rms); cyc_frac = _cat(cyc_frac); cyc_mass = _cat(cyc_mass)
    cyc_total = float(cyc_mass.sum()) if cyc_mass.size else None
    cyc_weighted = (float(np.sum(cyc_mass * cyc_frac) / cyc_total)
                    if cyc_total is not None and cyc_total > 0.0 else None)
    # Row-band line profile, pooled only over the pairs where the line is identifiable (>= 2 bands). The fit is
    # unweighted, so the slopes/deltas are directly comparable between legacy /1 and /2//3 Fits even though only
    # the latter carry support masses. This is a descriptive summary, never a score or a residual correlation.
    row_line_available = [bool(p["row_line_profile"]["available"]) for p in pairs]
    row_slope_mag = [_vector_magnitude(p["row_line_profile"]["slope_px_per_row"]) for p in pairs]
    row_delta_mag = [_vector_magnitude(p["row_line_profile"]["delta_first_to_last_px"]) for p in pairs]
    row_nonlinear_rms = [p["row_line_profile"]["nonlinear_residual_rms_px"] for p in pairs]
    row_explained = [p["row_line_profile"]["linear_explained_fraction"] for p in pairs]

    def _defined(vals):
        return [v for v in vals if v is not None]

    return {
        "n_pairs": len(pairs),
        "n_fallback": len(fallback_pairs),
        "fallback_pairs": fallback_pairs,
        "n_sequence_boundaries": len(sequence_boundary_pairs),
        "sequence_boundary_pairs": sequence_boundary_pairs,
        "fallback_reason_counts": fallback_reason_counts,
        "fallback_reason_unavailable_pairs": fallback_reason_unavailable_pairs,
        "rotation_deg_y_down": _stats([p["rotation_deg_y_down"] for p in pairs]),
        "scale": _stats([p["scale"] for p in pairs]),
        "inlier_frac": _stats([p["inlier_frac"] for p in pairs]),
        "mover_frac": _stats([p["mover_frac"] for p in pairs]),
        "mesh_residual_magnitude_px": _stats(mesh_mag),
        "row_residual_magnitude_px": _stats(row_mag),
        "mesh_supported_vs_filled": {
            "threshold": float(SUPPORT_THRESHOLD),
            "supported_residual_magnitude_px": _stats(_cat(mesh_supported)),
            "filled_residual_magnitude_px": _stats(_cat(mesh_filled)),
        },
        "row_supported_vs_filled": {
            "threshold": float(SUPPORT_THRESHOLD),
            "supported_residual_magnitude_px": _stats(_cat(row_supported)),
            "filled_residual_magnitude_px": _stats(_cat(row_filled)),
        },
        "row_line_profile": {
            "available_fits": int(sum(row_line_available)),
            "n_fits": len(pairs),
            "all_available": bool(all(row_line_available)) if pairs else True,
            "slope_magnitude_px_per_row": _stats(_defined(row_slope_mag)),
            "delta_magnitude_px": _stats(_defined(row_delta_mag)),
            "nonlinear_residual_rms_px": _stats(_defined(row_nonlinear_rms)),
            "linear_explained_fraction": _stats(_defined(row_explained)),
        },
        "cycle_consistency": {
            "available_fits": int(sum(cycle_available)),
            "n_fits": len(pairs),
            "all_available": bool(all(cycle_available)) if pairs else True,
            "error_rms_px": _stats(cyc_rms),
            "inconsistency_fraction": _stats(cyc_frac),
            "inconsistency_fraction_mass_weighted": cyc_weighted,
            "sample_mass": _stats(cyc_mass),
            "sample_mass_total": cyc_total,
        },
        "mesh_supported_fraction": _stats(mesh_fractions),
        "row_supported_fraction": _stats(row_fractions),
        "spatial_dispersion_rms_px": _stats([p["spatial_grid"]["proxy"]["dispersion_rms_px"]
                                             for p in pairs]),
        "per_pair_rotation_deg_y_down": [p["rotation_deg_y_down"] for p in pairs],
        "per_pair_scale": [p["scale"] for p in pairs],
        "per_pair_mesh_rms_px": [p["mesh_residual_magnitude_px"]["rms"] for p in pairs],
        "per_pair_row_rms_px": [p["row_residual_magnitude_px"]["rms"] for p in pairs],
        "per_pair_row_line_available": row_line_available,
        "per_pair_row_slope_magnitude_px_per_row": row_slope_mag,
        "per_pair_row_delta_magnitude_px": row_delta_mag,
        "per_pair_row_nonlinear_residual_rms_px": row_nonlinear_rms,
        "per_pair_row_linear_explained_fraction": row_explained,
        "per_pair_mesh_supported_fraction": [p["mesh_supported_vs_filled"]["supported_fraction"]
                                             if p["mesh_supported_vs_filled"] is not None else None
                                             for p in pairs],
        "per_pair_row_supported_fraction": [p["row_supported_vs_filled"]["supported_fraction"]
                                            if p["row_supported_vs_filled"] is not None else None
                                            for p in pairs],
        "per_pair_spatial_dispersion_rms_px": [p["spatial_grid"]["proxy"]["dispersion_rms_px"]
                                                for p in pairs],
        "per_pair_cycle_available": cycle_available,
        "per_pair_cycle_error_rms_px": [p["cycle_consistency"]["error_rms_px"]["rms"] for p in pairs],
        "per_pair_cycle_inconsistency_fraction_mean": \
            [p["cycle_consistency"]["inconsistency_fraction"]["mean"] for p in pairs],
        "per_pair_cycle_sample_mass_total": [p["cycle_consistency"]["sample_mass_total"] for p in pairs],
        "per_pair_fallback_reason": [p["fallback_reason"] for p in pairs],
        "per_pair_n_valid_samples": [p["n_valid_samples"] for p in pairs],
        "per_pair_n_ransac_inliers": [p["n_ransac_inliers"] for p in pairs],
        "per_pair_fit_metadata_available": [p["fit_metadata_available"] for p in pairs],
    }


# --------------------------------------------------------------------------------------
# per-case and top-level audit
# --------------------------------------------------------------------------------------

def _audit_one(ident, mot, meta):
    """Turn one loaded motion into a JSON-safe per-case report."""
    width, height = int(meta["width"]), int(meta["height"])
    stride = int(meta.get("stride", 1))
    pairs, mesh_mag, row_mag = [], [], []
    mesh_supported, mesh_filled, row_supported, row_filled = [], [], [], []
    for i, fit in enumerate(mot.fits):
        pairs.append(summarize_pair(fit, pair=i, frame_index=i * stride, width=width, height=height))
        mm = np.linalg.norm(np.asarray(fit.mesh, np.float64), axis=-1).ravel()
        rr = np.linalg.norm(np.asarray(fit.rows, np.float64), axis=-1).ravel()
        mesh_mag.append(mm)
        row_mag.append(rr)
        mesh_mass = getattr(fit, "mesh_support_mass", None)
        row_mass = getattr(fit, "row_support_mass", None)
        if mesh_mass is not None:
            ok = np.asarray(mesh_mass, np.float64).ravel() >= SUPPORT_THRESHOLD
            mesh_supported.append(mm[ok]); mesh_filled.append(mm[~ok])
        if row_mass is not None:
            ok = np.asarray(row_mass, np.float64).ravel() >= SUPPORT_THRESHOLD
            row_supported.append(rr[ok]); row_filled.append(rr[~ok])
    summary = _case_summary(pairs, np.concatenate(mesh_mag) if mesh_mag else np.zeros(0),
                            np.concatenate(row_mag) if row_mag else np.zeros(0),
                            mesh_supported=mesh_supported, mesh_filled=mesh_filled,
                            row_supported=row_supported, row_filled=row_filled)
    return {
        "case": ident["case"],
        "run_id": ident["run_id"],
        "manifest_path": ident["manifest_path"],
        "motion_path": ident["motion_path"],
        "source_window": ident["source_window"],
        "source_sha256": ident["source_sha256"],
        "weights_sha256": ident["weights_sha256"],
        "capture_dump_motion": ident["capture_dump_motion"],
        "motion": meta,
        "summary": summary,
        "pairs": pairs,
    }


def run_real_motion_field_audit(manifest_paths):
    """Audit one or more benchmark manifests and their captured motion; return a JSON-safe report.

    Each manifest must exist, parse as a JSON object, have ``status == "success"``, record a resolvable
    ``outputs.motion``, and carry a source identity (sha256, preprocessing, decoded size/fps, weights sha256)
    that every other manifest shares. The motion payload is loaded with ``allow_pickle=False`` and must match
    the manifest's recorded decoded size/fps.
    """
    paths = [Path(p) for p in manifest_paths]
    if not paths:
        raise MotionFieldAuditError("no_manifests", "at least one benchmark manifest.json path is required")

    loaded = [load_manifest_motion(p) for p in paths]
    identities = [ident for ident, _mot, _meta in loaded]
    _require_matched(identities)

    cases = [_audit_one(ident, mot, meta) for ident, mot, meta in loaded]
    ref = identities[0]
    return {
        "schema": SCHEMA,
        "ok": True,
        "scope": SCOPE_NOTE,
        "motion_payload_safety": {
            "loader": "bench.real_composition_audit.load_motion_npz (np.load(..., allow_pickle=False))",
            "validated": ["manifest json object", "status success", "outputs.motion", "source identity",
                          "motion schema (mlxstab.motion/1 through /5)", "motion required keys",
                          "motion array shapes", "motion support-mass shapes (v2 through v5)",
                          "motion cycle-map shapes/availability/ranges (v3 through v5)",
                          "motion fit-gate shapes/availability/reason values (v4/v5)",
                          "motion cut-boundary indices (v5)", "motion dtype kinds",
                          "motion finiteness", "no pickle"],
            "legacy_support": "mlxstab.motion/1 payloads load with support=None; support maps and "
                              "supported-vs-filled splits are then reported as null",
            "legacy_cycle": "mlxstab.motion/1 and /2 payloads carry no cycle maps; their cycle maps and "
                            "per-pair/pooled cycle summaries are then reported as null/unavailable",
            "legacy_fit_metadata": "mlxstab.motion/1, /2 and /3 payloads carry no fit-gate diagnostics; "
                                   "n_valid_samples, n_ransac_inliers and fallback_reason are then reported as "
                                   "null with fit_metadata_available=false rather than invented",
            "row_line_note": "the row-band line profile is computed from Fit.rows, which every motion schema "
                              "(/1 through /5) carries, so it is unaffected by support/cycle/fit metadata "
                              "availability differences; the unweighted fit never reads support masses",
        },
        "identity": {
            "matched_source_identity": True,
            "checked_manifest_fields": [label for _k, _c, label in _MATCH_CHECKS],
            "source_window_is_not_matched": True,
            "source_sha256": ref["source_sha256"],
            "preprocessing": ref["preprocessing"],
            "decoded": ref["decoded"],
            "decoded_identity": ref["decoded_identity"],
            "weights_sha256": ref["weights_sha256"],
            "manifests": [dict(ident) for ident in identities],
        },
        "similarity_definition": {
            "source": "best-fit similarity of the applied global homography H over an inset span grid",
            "angle_field": "rotation_deg_y_down",
            "angle_convention": "image y points down; positive is visual clockwise (atan2(A[1,0], A[0,0]))",
            "scale_field": "scale",
            "scale_definition": "non-negative hypot(A[0,0], A[1,0]) of the fitted linear part",
        },
        "residual_definition": {
            "units": "pixels (same frame units as mesh/rows in the motion NPZ)",
            "mesh_vertex_residual_vectors_px": "per-pair mesh residual vertex displacement [gr+1, gc+1, 2]",
            "mesh_vertex_residual_magnitudes_px": "per-pair mesh residual vertex displacement magnitude [gr+1, gc+1]",
            "stats": ["rms", "median", "p95", "max"],
            "scope": "parallax-like spatial motion evidence only; model mismatch/flow error/rolling-shutter "
                     "leftovers are included; not metric depth or object semantics",
        },
        "fallback_reason_definition": {
            "what": "per-pair diagnostics from mlxstab.motion/4 and /5: n_valid_samples, n_ransac_inliers "
                    "and the integer fallback_reason_code mapped back to a stable fallback_reason label; codes "
                    "0..3 name the first fit_motion gate that refused the fit, and /5 code 4 marks an explicit "
                    "sequence boundary where no fit was attempted",
            "none": "the fit passed every gate (a successful fit records the observed sample/inlier counts)",
            "insufficient_valid_samples": "n_valid_samples < min_valid; the fit never reached RANSAC",
            "insufficient_ransac_inliers": "n_ransac_inliers < min_valid",
            "low_ransac_inlier_fraction": "n_ransac_inliers / n_valid_samples < the inlier-fraction gate",
            "sequence_boundary": "the frame pair spans a caller-supplied cut; no flow or geometric fit was run",
            "precedence": "geometric reasons are tested in the order above; the first failing gate names the reason, "
                          "so a pair failing several thresholds still gets one deterministic label; a skipped "
                          "sequence-boundary pair is labelled separately and is not a failed fit",
            "is_not": "a label for the existing fit_motion refusal only: it says which sample/inlier threshold "
                      "produced an identity fallback, not whether the footage is stable or the pixels are "
                      "correct, and it is not a quality score or a combined metric",
            "availability": "null (with fit_metadata_available=false) for manually constructed and legacy "
                            "/1, /2 and /3 fits; /4 and /5 carry the counts and reason",
        },
        "spatial_proxy_definition": {
            "cell_vector": "mean of a mesh cell's four vertex residual vectors (the residual at the cell centre)",
            "dispersion_rms_px": "RMS distance of the per-cell vectors from their spatial mean; removes a "
                                 "global uniform shift and measures only the spatially varying part",
            "neighbor_cosine_weighted_mean": "magnitude-product-weighted mean cosine between the centred "
                                             "per-cell vectors of 4-neighbour cells; ~1 coherent, ~0 "
                                             "incoherent, None when undefined",
            "is_not_depth": "a coherent residual is equally consistent with parallax, smooth model mismatch "
                            "and rolling-shutter leftovers; the proxy is descriptive only",
        },
        "support_definition": {
            "support_mass": "per-pair, per-vertex (mesh_support_mass [gr+1,gc+1]) and per-band "
                            "(row_support_mass [nr]) observation-coverage diagnostic: the non-robust "
                            "kernel-weighted count of usable non-mover flow samples, recorded before "
                            "mlxstab.motion._fill diffuses under-observed locations",
            "threshold": float(SUPPORT_THRESHOLD),
            "meaning": "mass >= threshold = directly observed (supported); mass < threshold = filled by "
                       "diffusion from observed neighbours",
            "supported_vs_filled": "residual magnitude RMS/median/p95/max is reported separately for the "
                                   "supported and filled vertices/bands",
            "is_not": "NOT a probability and NOT flow confidence; an observation-coverage diagnostic only, "
                      "not a calibrated uncertainty estimate or a residual-accuracy measure",
            "legacy": "mlxstab.motion/1 payloads carry no support mass; the fields are then null",
        },
        "row_line_definition": {
            "what": "per-pair descriptive unweighted least-squares line fit of both components of Fit.rows "
                    "(an x-independent displacement per row band) against the row-band centre y positions",
            "band_centers": "(j + 0.5)*height/nr - 0.5 for j in [0, nr), the same y positions "
                            "mlxstab.motion.fit_motion used",
            "slope_px_per_row": "per-component line slope, in px displacement per px row; null when not "
                                 "identifiable (fewer than 2 bands)",
            "delta_first_to_last_px": "slope * (last_band_center - first_band_center): the fitted "
                                       "displacement change from the first to the last band centre; null when "
                                       "not identifiable",
            "nonlinear_residual_rms_px": "pooled RMS of the residual (rows - fitted line) over both "
                                          "components, in px; finite-safe 0 for constant and one-band fields",
            "linear_explained_fraction": "pooled multivariate R^2 = 1 - RSS/TSS with per-component means "
                                          "over the bands; null when the field is constant (TSS = 0) or has "
                                          "fewer than 2 bands",
            "unweighted": "the fit does not read the support masses, so legacy mlxstab.motion/1 (no support) "
                          "and /2 and /3 (with support) Fits share exactly the same definition",
            "aggregation": "per-case descriptive statistics and per-pair series of the slope magnitude, the "
                           "first-to-last-band delta magnitude, the nonlinear residual RMS and the "
                           "linear-explained fraction; they are not combined into a score and not correlated "
                           "with the cycle maps",
            "is_not": "NOT a rolling-shutter detector and NOT a physical readout estimate: a smooth vertical "
                      "row displacement is rolling-shutter-LIKE but can equally arise from other motion or "
                      "model mismatch, and no camera readout time, focal calibration or ground-truth readout "
                      "is inferred",
        },
        "cycle_consistency_definition": {
            "what": "spatial forward-backward cycle-consistency maps on the mesh grid, per pair and pooled "
                    "per case",
            "cycle_error": "fb_err = ||f + sampled_b||, the forward-backward (cycle) error at a sampled point",
            "tolerance": "fb_abs + fb_rel*||f||, the existing forward-backward consistency tolerance",
            "cycle_err_rms": "Gaussian-spatially-weighted RMS of fb_err at each mesh vertex, in pixels",
            "cycle_inconsistency_frac": "spatially-weighted fraction of finite, in-bounds samples whose "
                                         "fb_err >= the tolerance, in [0,1]",
            "cycle_sample_mass": "spatial sample weight sum (coverage) at each mesh vertex",
            "sample_set": "every finite, in-bounds forward endpoint, independent of the fit's valid/non-mover "
                          "mask; a sample rejected by the tolerance is still counted (zero-mass vertices are "
                          "zero maps with zero mass)",
            "is_not": "a forward-backward CYCLE-CONSISTENCY diagnostic only: NOT true correspondence error "
                      "(a consistent cycle does not prove the correspondence is right, and an inconsistent "
                      "cycle does not locate the error), NOT ground-truth error, NOT a probability and NOT "
                      "calibrated uncertainty. No correlation with the residual maps is computed or implied.",
            "availability": "null for manually constructed and legacy/1 and /2 fits; only mlxstab.motion/3 "
                            "and /4 carry the three cycle maps and the per-fit availability flag",
        },
        "cases": cases,
        "limitations": LIMITATIONS,
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _error_payload(manifest_paths, exc):
    return {
        "schema": SCHEMA,
        "ok": False,
        "manifests": [str(p) for p in manifest_paths],
        "error": {"code": exc.code, "message": str(exc)},
    }


def _emit(payload, out):
    text = json.dumps(json_safe(payload), indent=1, sort_keys=True, allow_nan=False)
    if out:
        Path(out).write_text(text, encoding="utf-8")
    else:
        print(text)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Captured real motion-field audit (rotation/scale + row/mesh residual magnitudes; "
                    "manifest-driven, no media, no rendering).")
    ap.add_argument("manifests", nargs="+", metavar="MANIFEST.json",
                    help="one or more benchmark manifest.json files whose outputs.motion to audit")
    ap.add_argument("--out", default=None, help="write the JSON report here (default: stdout)")
    args = ap.parse_args(argv)

    try:
        report = run_real_motion_field_audit(args.manifests)
    except MotionFieldAuditError as exc:
        _emit(_error_payload(args.manifests, exc), args.out)
        return 2

    _emit(report, args.out)
    if args.out:
        print(f"motion-field audit cases={[c['case'] for c in report['cases']]} "
              f"pairs={[c['summary']['n_pairs'] for c in report['cases']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
