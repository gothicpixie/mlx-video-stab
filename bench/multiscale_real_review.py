"""Research-only local review-clip builder: blinded A/B MP4s from a captured real-motion run directory.

What this does
--------------
A successful benchmark run directory contains a ``manifest.json`` and a captured motion payload
(``motion_r0.npz``, schema ``mlxstab.motion/4``). This module turns that pair into four short, **blinded A/B
review clips** so a human can eyeball the fixed E10 multiscale-retention candidates against their nearest
measured-crop L1 counterparts on real footage. It reads media but does **not** measure image quality and does
**not** pick a winner.

Pipeline (all fixed, nothing tuned):

1. Validate the motion payload with :func:`bench.real_composition_audit.load_motion_npz` (plain arrays only,
   ``allow_pickle=False``) and cross-check it against the sibling benchmark manifest
   (``source.sha256``/``source.window``/``source.decoded``/``source.preprocessing`` and ``outputs.motion``).
   A hash, window, dimension, fps or frame-count mismatch is refused with a stable ``code``.
2. Probe and decode the run's source at the manifest window with the manifest's recorded ``bwdif`` deinterlace,
   then downscale every frame to a fixed ``480x270`` panel before warping.
3. Compose the fitted global homographies with :func:`bench.composition_candidate.global_homography` into a
   **translation path** ``C_t = frame_center - Psi_t(frame_center)`` (see below), scaled into panel pixels.
4. Build the four fixed :data:`bench.multiscale_motion_audit.MULTISCALE_CANDIDATES` outputs and the fixed L1
   budget grid, pair each multiscale candidate with the L1 budget whose **measured** crop proxy is nearest
   (:func:`bench.multiscale_motion_audit.nearest_crop_index`), and backward-warp the observed panels one
   frame at a time with :func:`bench.multiscale_render_audit.backward_warp`.
5. Write one two-panel A/B MP4 per comparison through :class:`mlxstab.media.Encoder`. Which side is
   multiscale and which is L1 is randomised from a fixed recorded seed; the videos only carry ``A``/``B``
   bitmap labels and the hidden mapping is written to the JSON report.

Camera convention (explicit, and tested)
----------------------------------------
``Psi_t = H_{t-1} o ... o H_0`` maps reference-frame pixel coordinates to frame ``t`` (frame 0 identity).
The panel-space translation path is

    C_t = frame_center - Psi_t(frame_center)

with ``frame_center = ((W-1)/2, (H-1)/2)`` on the pixel-centre grid. With a pure forward translation
homography ``H = [[1,0,tx],[0,1,ty],[0,0,1]]`` this gives ``C_t = -t*(tx, ty)``: a camera that moves content
by ``+(tx, ty)`` yields a scene-shift path of ``-(tx, ty)``, which is exactly the sign
:func:`bench.multiscale_render_audit.backward_warp` expects for ``delta = P - C`` (``P = C`` reproduces the
observed panel). This is a **translation-only** reduction: only the frame-centre displacement of the composed
homography is used, so rotation, scale, shear, perspective and depth are ignored.

What this is not
----------------
It decodes real media and renders review clips, but it reports only path/crop/support numbers and writes no
image-quality metric (no MAE/RMS/PSNR/SSIM), no ranking, no winner and no aggregate score. It changes no
production default and imports no production stabilizer/renderer. The clips are a local human-review aid, not
evidence about the production pipeline.

Run ``python -m bench.multiscale_real_review RUN_DIR [--out-dir DIR] [--width 480] [--height 270]``. Outputs
go under the ignored ``out/`` tree by default (``RUN_DIR/review``).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

from mlxstab import geometry as G
from mlxstab import media

from . import composition_candidate as CC
from . import multiscale_motion_audit as A
from .manifest import json_safe, sha256_file, write_manifest
from .multiscale_render_audit import backward_warp
from .real_composition_audit import CompositionAuditError, MotionPayloadError, load_motion_npz

SCHEMA = "mlxstab.multiscale_real_review/1"

MANIFEST_NAME = "manifest.json"
DEFAULT_MOTION_NAME = "motion_r0.npz"
DEFAULT_OUT_WIDTH = 480
DEFAULT_OUT_HEIGHT = 270

# The one deinterlace filter this review knows about: it is exactly what ``stabilize`` records for an
# interlaced source, and exactly what :func:`mlxstab.media.decode` applies when ``deinterlace=True``.
BWDIF_VF = "bwdif=mode=send_field:parity=auto:deint=all"

# Fixed, recorded blinding seed. Side assignment is drawn once per comparison, in order, from this seed.
BLINDING_SEED = 20240517

# 5x7 bitmap font (only the labels a review clip needs). ``#`` marks an ink pixel.
FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
}
LABEL_ORIGIN = (4, 4)
LABEL_SCALE = 2
LABEL_PAD = 2
LABEL_INK = (255, 255, 255)
LABEL_BG = (0, 0, 0)

SCOPE_NOTE = (
    "Research-only local review-clip builder. From a successful benchmark run directory it validates the "
    "captured motion payload against the sibling manifest, decodes the source window with the recorded bwdif "
    "deinterlace, downscales to 480x270, reduces the composed global homographies to a translation-only "
    "frame-centre path, renders the four fixed multiscale-retention candidates against their nearest "
    "measured-crop L1 counterparts with a one-frame-at-a-time bilinear backward warp, and writes four "
    "blinded two-panel A/B MP4s. It reports path/crop/support provenance only: no image-quality metric, no "
    "ranking and no winner, and it changes no production default."
)

TRANSLATION_MODEL = {
    "definition": "C_t = frame_center - Psi_t(frame_center)",
    "composition": "Psi_t = H_{t-1} o ... o H_0 (bench.composition_candidate.global_homography; frame 0 identity)",
    "center_definition": "frame_center = ((W-1)/2, (H-1)/2) on the pixel-centre grid",
    "sign_convention": (
        "for a pure forward translation H with shift (tx, ty), C_t = -t*(tx, ty): the path follows the scene "
        "shift, so backward-warping by delta = P - C with P = C reproduces the observed panel"
    ),
    "path_channel_order": "channels are (x, y): index 0 horizontal, index 1 vertical",
    "limitation": (
        "translation-only: only the frame-centre displacement of the composed global homography is used; "
        "rotation, scale, shear, perspective and depth are ignored"
    ),
}

LIMITATIONS = (
    "translation-only reduction of the composed global homography: rotation, scale, shear, perspective and "
    "depth are ignored, so the clips are a translational approximation of the fitted motion",
    "single-layer warp of the real observed frames: no depth ordering, subject motion, parallax or occlusion "
    "handling",
    "downscaled to 480x270 before warping: the review is a small-panel visual aid, not a full-resolution render",
    "no image-quality metric is computed; the report has path/crop/support numbers only and selects no winner",
    "A/B side assignment is randomised from a fixed recorded seed and is hidden from the video (the mapping "
    "lives only in the JSON report)",
    "the fixed multiscale retention weights and the predeclared L1 budget grid are reported verbatim; nothing "
    "is tuned and no per-clip selection is made",
    "nearest-crop pairing is descriptive: it produces no winner, rank, score, aggregate or tuned selection",
    "review artifacts are local and are written under the ignored out/ tree; they are not published",
)

REVIEW_POLICY = (
    "Each comparison is a blinded two-panel A/B clip; the panel mapping is randomised from a fixed recorded "
    "seed and written only to the JSON report. Every path/crop/support number is reported separately per pair "
    "and per side. No aggregate winner, ranking, total score or image-quality metric is computed and no "
    "candidate or family is asserted superior."
)


class ReviewError(ValueError):
    """A run directory was refused. ``code`` is a stable machine-readable reason."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = str(code)


def _require(cond, code, message):
    if not cond:
        raise ReviewError(code, message)


def _finite_number(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _validate_dim(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


# --------------------------------------------------------------------------------------
# manifest provenance and motion loading
# --------------------------------------------------------------------------------------

def manifest_provenance(run_dir, motion_path):
    """Load and validate the sibling manifest for a captured motion payload.

    Returns a JSON-safe provenance dict. Requires a successful run whose manifest names this motion payload
    (``outputs.motion`` basename and ``experiment_identity.capture.dump_motion is True``) and records a
    non-empty ``source.sha256``, a finite positive ``source.window`` and a ``source.preprocessing`` /
    ``source.decoded`` description. Malformed or refused manifests raise :class:`ReviewError` with a stable
    ``code``.
    """
    run_dir = Path(run_dir)
    motion_path = Path(motion_path)
    manifest_path = run_dir / MANIFEST_NAME
    _require(manifest_path.exists(), "manifest_missing",
             f"no benchmark manifest at {manifest_path.name!r} in run directory {run_dir.name!r}")
    try:
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReviewError("manifest_unreadable", f"cannot read/parse {manifest_path.name!r}: {exc}")
    _require(isinstance(man, dict), "manifest_malformed", "manifest top level must be a JSON object")
    _require(man.get("status") == "success", "run_not_successful",
             f"manifest status is {man.get('status')!r}, not 'success'; refusing a non-successful run")

    case = man.get("case")
    _require(isinstance(case, dict) and isinstance(case.get("name"), str) and case["name"].strip(),
             "manifest_malformed", "manifest case.name is missing or empty")

    source = man.get("source")
    _require(isinstance(source, dict), "manifest_malformed", "manifest source object is missing")
    sha = source.get("sha256")
    _require(isinstance(sha, str) and sha.strip(), "manifest_malformed", "manifest source.sha256 is missing")
    window = source.get("window")
    _require(isinstance(window, dict) and "start_s" in window and "duration_s" in window,
             "manifest_malformed", "manifest source.window.start_s/duration_s are missing")
    _require(_finite_number(window.get("start_s")) and _finite_number(window.get("duration_s")),
             "manifest_malformed", "manifest source.window values must be finite")
    start_s, duration_s = float(window["start_s"]), float(window["duration_s"])
    _require(start_s >= 0.0 and duration_s > 0.0, "manifest_malformed",
             f"manifest source.window must have start_s >= 0 and duration_s > 0, got {start_s}, {duration_s}")
    preprocessing = source.get("preprocessing")
    _require(isinstance(preprocessing, dict), "manifest_malformed",
             "manifest source.preprocessing object is missing")
    _require("interlaced" in preprocessing and "deinterlace" in preprocessing, "manifest_malformed",
             "manifest source.preprocessing.interlaced/deinterlace are missing")
    interlaced = preprocessing.get("interlaced")
    _require(isinstance(interlaced, bool), "manifest_malformed",
             f"manifest source.preprocessing.interlaced must be a boolean, got {interlaced!r}")
    deinterlace = preprocessing.get("deinterlace")
    if interlaced:
        _require(isinstance(deinterlace, str) and deinterlace.strip(), "manifest_malformed",
                 "manifest source.preprocessing.deinterlace is missing for an interlaced source")
        _require(deinterlace == BWDIF_VF, "unsupported_deinterlace",
                 f"recorded deinterlace {deinterlace!r} is not the supported bwdif filter {BWDIF_VF!r}")
    decoded = source.get("decoded")
    _require(isinstance(decoded, dict), "manifest_malformed", "manifest source.decoded object is missing")
    for key in ("width", "height", "frames"):
        _require(isinstance(decoded.get(key), (int, np.integer)) and not isinstance(decoded.get(key), bool)
                 and int(decoded[key]) >= 1, "manifest_malformed",
                 f"manifest source.decoded.{key} must be a positive integer, got {decoded.get(key)!r}")
    _require(_finite_number(decoded.get("fps")) and float(decoded["fps"]) > 0.0, "manifest_malformed",
             f"manifest source.decoded.fps must be finite and > 0, got {decoded.get('fps')!r}")

    outputs = man.get("outputs")
    _require(isinstance(outputs, dict), "manifest_malformed", "manifest outputs object is missing")
    outputs_motion = outputs.get("motion")
    _require(isinstance(outputs_motion, str) and outputs_motion.strip(), "manifest_malformed",
             "manifest outputs.motion is missing")
    want = motion_path.name
    got = Path(outputs_motion).name
    _require(got == want, "mismatched_motion_output",
             f"manifest outputs.motion basename {got!r} does not match the supplied motion file {want!r}; the "
             f"manifest does not describe this motion")

    experiment = man.get("experiment_identity")
    _require(isinstance(experiment, dict), "manifest_malformed", "manifest experiment_identity is missing")
    capture = experiment.get("capture")
    _require(isinstance(capture, dict), "manifest_malformed",
             "manifest experiment_identity.capture is missing")
    _require(capture.get("dump_motion") is True, "manifest_dump_motion_not_captured",
             f"experiment_identity.capture.dump_motion is {capture.get('dump_motion')!r}, not True; the run "
             f"did not capture this motion payload")

    return {
        "manifest": MANIFEST_NAME,
        "run": run_dir.name,
        "case": case["name"],
        "run_id": man.get("run_id"),
        "manifest_schema": man.get("schema"),
        "source_sha256": sha,
        "source_window": {"start_s": start_s, "duration_s": duration_s},
        "source_decoded": {
            "width": int(decoded["width"]), "height": int(decoded["height"]),
            "fps": float(decoded["fps"]), "frames": int(decoded["frames"]),
        },
        "preprocessing": json_safe(preprocessing),
        "output_motion": outputs_motion,
        "capture_dump_motion": True,
        "source_path": source.get("path"),
    }


def load_review_inputs(run_dir, *, motion_name=DEFAULT_MOTION_NAME):
    """Load and cross-check the motion payload, manifest provenance and source hash for one run directory.

    Returns ``(motion, motion_meta, provenance)``. The motion payload goes through
    :func:`bench.real_composition_audit.load_motion_npz` (so malformed payloads raise ``MotionPayloadError``);
    its dimensions, fps and frame count must match ``source.decoded``. If the manifest source file is present
    its ``sha256`` is recomputed and a mismatch is refused. A refused manifest raises :class:`ReviewError`.
    """
    run_dir = Path(run_dir)
    _require(run_dir.is_dir(), "run_dir_missing",
             f"run directory {run_dir.name!r} does not exist or is not a directory")
    motion_path = run_dir / motion_name
    _require(motion_path.exists(), "motion_missing", f"no motion payload {motion_name!r} in {run_dir.name!r}")
    prov = manifest_provenance(run_dir, motion_path)
    mot, meta = load_motion_npz(motion_path)

    dec = prov["source_decoded"]
    _require((int(meta["width"]), int(meta["height"])) == (dec["width"], dec["height"]),
             "dimension_mismatch",
             f"motion is {int(meta['width'])}x{int(meta['height'])} but manifest source.decoded is "
             f"{dec['width']}x{dec['height']}")
    fps_tol = 1e-6 * max(1.0, abs(dec["fps"]))
    _require(abs(float(meta["fps"]) - dec["fps"]) <= fps_tol, "fps_mismatch",
             f"motion fps {float(meta['fps'])!r} != manifest source.decoded fps {dec['fps']!r}")
    _require(int(meta["n_frames"]) == dec["frames"], "frame_count_mismatch",
             f"motion n_frames {int(meta['n_frames'])} != manifest source.decoded frames {dec['frames']}")
    _require(int(meta["stride"]) == 1, "stride_not_supported",
             f"review requires stride=1 (contiguous composed frames), got stride {int(meta['stride'])}")

    source_path = prov.get("source_path")
    verified = False
    hash_note = "manifest source.path is absent; source.sha256 recorded but not recomputed"
    if isinstance(source_path, str) and source_path:
        sp = Path(source_path)
        if sp.is_file():
            actual = sha256_file(sp)
            _require(actual == prov["source_sha256"], "source_hash_mismatch",
                     f"source file sha256 {actual} != manifest source.sha256 {prov['source_sha256']}")
            verified = True
            hash_note = "source file present and sha256 matches the manifest"
        else:
            hash_note = f"source file {sp.name!r} is not present locally; recorded sha256 not recomputed"
    prov["source_sha256_verified"] = verified
    prov["source_hash_note"] = hash_note
    # NOTE: prov keeps the manifest source path for the decoder; run_review's report provenance lists only
    # explicit, path-free keys, so the path never reaches the JSON report.
    prov["motion"] = {"width": int(meta["width"]), "height": int(meta["height"]),
                      "n_frames": int(meta["n_frames"]), "fps": float(meta["fps"]),
                      "n_fits": int(meta["n_fits"]), "stride": int(meta["stride"]),
                      "schema": meta["schema"]}
    return mot, meta, prov


# --------------------------------------------------------------------------------------
# translation path, scaling and metrics
# --------------------------------------------------------------------------------------

def frame_center(width, height):
    """Pixel-centre-grid frame centre ``((W-1)/2, (H-1)/2)`` (x, y)."""
    width = _validate_dim(width, "width")
    height = _validate_dim(height, "height")
    return np.array([(width - 1) / 2.0, (height - 1) / 2.0], np.float64)


def composed_translation_path(fits, width, height, n_frames):
    """Compose the fitted global homographies into the translation path ``C_t = center - Psi_t(center)``.

    ``fits`` are the per-pair :class:`mlxstab.motion.Fit` objects (``.H``), ``width``/``height`` are the
    source pixel dimensions and ``n_frames`` the output frame count (``len(fits) + 1``). Returns ``[T, 2]``
    in source pixels, channels ``(x, y)``. Frame 0 is zero by construction. This is a **translation-only**
    reduction of ``Psi_t``: only the frame-centre displacement is used.
    """
    fits = list(fits)
    width = _validate_dim(width, "width")
    height = _validate_dim(height, "height")
    n_frames = _validate_dim(n_frames, "n_frames")
    if n_frames != len(fits) + 1:
        raise ValueError(
            f"n_frames {n_frames} != len(fits)+1 {len(fits) + 1}; the review requires stride-1 contiguous fits")
    center = frame_center(width, height)
    out = np.empty((n_frames, 2), np.float64)
    for t in range(n_frames):
        psi = CC.global_homography(fits, t)
        out[t] = center - G.apply_h(psi, center)
    return out


def scale_path(path, src_width, src_height, out_width, out_height):
    """Scale a source-pixel translation path into panel-pixel units ``(out/src)`` per channel."""
    path = np.asarray(path, np.float64)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError(f"path must be [T, 2], got shape {path.shape}")
    src_width = _validate_dim(src_width, "src_width")
    src_height = _validate_dim(src_height, "src_height")
    out_width = _validate_dim(out_width, "out_width")
    out_height = _validate_dim(out_height, "out_height")
    return path * np.array([out_width / src_width, out_height / src_height], np.float64)


def correction_metrics(delta):
    """Descriptive path metrics of a correction series ``delta = P - C`` ``[T, 2]`` (panel pixels).

    Reports the RMS correction, the RMS second difference of the correction (a path-level jitter proxy,
    ``None`` for fewer than three samples) and the maximum absolute correction (the translation crop proxy),
    overall and per channel. Nothing is pooled across candidates.
    """
    delta = np.asarray(delta, np.float64)
    if delta.ndim != 2 or delta.shape[1] != 2:
        raise ValueError(f"delta must be [T, 2], got shape {delta.shape}")
    if delta.shape[0] < 1:
        raise ValueError("delta must have at least one sample")
    if not bool(np.isfinite(delta).all()):
        raise ValueError("delta must be finite")
    second = delta[2:] - 2.0 * delta[1:-1] + delta[:-2] if delta.shape[0] >= 3 else None
    return {
        "n_samples": int(delta.shape[0]),
        "rms_correction_px": float(np.sqrt(np.mean(delta * delta))),
        "rms_second_difference_correction_px": (
            float(np.sqrt(np.mean(second * second))) if second is not None and second.size else None),
        "max_abs_correction_px": float(np.max(np.abs(delta))),
        "max_abs_correction_per_channel_px": [float(v) for v in np.max(np.abs(delta), axis=0)],
    }


def support_summary(valid_fractions):
    """Geometric support summary over per-frame valid fractions: mean/min/max and the min frame index."""
    frac = np.asarray(valid_fractions, np.float64)
    if frac.ndim != 1 or frac.size == 0:
        raise ValueError(f"valid_fractions must be a non-empty 1-D sequence, got shape {frac.shape}")
    if not bool(np.isfinite(frac).all()):
        raise ValueError("valid_fractions must be finite")
    return {
        "frames": int(frac.size),
        "mean": float(frac.mean()),
        "min": float(frac.min()),
        "max": float(frac.max()),
        "min_frame": int(np.argmin(frac)),
    }


# --------------------------------------------------------------------------------------
# downscale, warp and panel assembly
# --------------------------------------------------------------------------------------

def resize_bilinear(frame, out_width, out_height):
    """Deterministic bilinear downscale of a ``[H, W, 3]`` frame to ``out_height x out_width`` (uint8)."""
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be [H, W, 3], got shape {frame.shape}")
    out_width = _validate_dim(out_width, "out_width")
    out_height = _validate_dim(out_height, "out_height")
    height, width = int(frame.shape[0]), int(frame.shape[1])
    xs = (np.arange(out_width, dtype=np.float64) + 0.5) * (width / out_width) - 0.5
    ys = (np.arange(out_height, dtype=np.float64) + 0.5) * (height / out_height) - 0.5
    x0f, y0f = np.floor(xs), np.floor(ys)
    x0 = np.clip(x0f, 0, width - 1).astype(np.int64)
    x1 = np.clip(x0f + 1, 0, width - 1).astype(np.int64)
    y0 = np.clip(y0f, 0, height - 1).astype(np.int64)
    y1 = np.clip(y0f + 1, 0, height - 1).astype(np.int64)
    wx1 = (xs - x0f)[None, :, None]
    wx0 = 1.0 - wx1
    wy1 = (ys - y0f)[:, None, None]
    wy0 = 1.0 - wy1
    src = frame.astype(np.float64)
    out = (wy0 * wx0) * src[y0][:, x0] + (wy0 * wx1) * src[y0][:, x1] \
        + (wy1 * wx0) * src[y1][:, x0] + (wy1 * wx1) * src[y1][:, x1]
    return np.clip(np.rint(out), 0.0, 255.0).astype(np.uint8)


def warp_panel(observed_panel, delta):
    """One-frame backward warp of a panel by ``delta`` (x, y); returns ``(uint8 panel, valid mask)``."""
    observed_panel = np.asarray(observed_panel, np.float64)
    if observed_panel.ndim != 3:
        raise ValueError(f"observed_panel must be [H, W, C], got shape {observed_panel.shape}")
    delta = np.asarray(delta, np.float64)
    if delta.shape != (2,):
        raise ValueError(f"delta must be a single (x, y) pair, got shape {delta.shape}")
    frames, valid = backward_warp(observed_panel[None], delta[None])
    panel = np.clip(np.rint(frames[0]), 0.0, 255.0).astype(np.uint8)
    return panel, valid[0]


def draw_bitmap_label(panel, letter, *, origin=LABEL_ORIGIN, scale=LABEL_SCALE,
                      ink=LABEL_INK, background=LABEL_BG):
    """Draw a small bitmap letter onto ``panel`` ``[H, W, 3]`` in place; returns the panel."""
    glyph = FONT.get(letter)
    if glyph is None:
        raise ValueError(f"no bitmap label {letter!r}; known labels are {sorted(FONT)}")
    panel = np.asarray(panel)
    if panel.ndim != 3 or panel.shape[2] != 3:
        raise ValueError(f"panel must be [H, W, 3], got shape {panel.shape}")
    height, width = int(panel.shape[0]), int(panel.shape[1])
    oy, ox = int(origin[0]), int(origin[1])
    scale = int(scale)
    if scale < 1:
        raise ValueError("scale must be >= 1")
    gh, gw = len(glyph), len(glyph[0])
    box_h, box_w = gh * scale + 2 * LABEL_PAD, gw * scale + 2 * LABEL_PAD
    y0, y1 = max(0, oy), min(height, oy + box_h)
    x0, x1 = max(0, ox), min(width, ox + box_w)
    panel[y0:y1, x0:x1] = np.asarray(background, np.uint8)
    for gy in range(gh):
        for gx in range(gw):
            if glyph[gy][gx] != "1":
                continue
            py, px = oy + LABEL_PAD + gy * scale, ox + LABEL_PAD + gx * scale
            py1, px1 = min(height, py + scale), min(width, px + scale)
            if py < height and px < width:
                panel[py:py1, px:px1] = np.asarray(ink, np.uint8)
    return panel


def assemble_ab_panel(left_panel, right_panel):
    """Two-panel A/B frame: ``A`` labelled on the left, ``B`` on the right, side by side."""
    left = np.array(left_panel, np.uint8, copy=True)
    right = np.array(right_panel, np.uint8, copy=True)
    if left.shape != right.shape:
        raise ValueError(f"left {left.shape} and right {right.shape} panels must share one shape")
    draw_bitmap_label(left, "A")
    draw_bitmap_label(right, "B")
    return np.concatenate([left, right], axis=1)


# --------------------------------------------------------------------------------------
# blinding
# --------------------------------------------------------------------------------------

def blind_assignment(seed, count):
    """Deterministic per-comparison side assignment: ``left_is_multiscale[i]`` drawn from ``seed``.

    Uses a dedicated ``random.Random(seed)`` stream so the assignment depends only on the recorded seed and
    the comparison order. ``True`` means the multiscale candidate is the left panel; ``False`` means the L1
    counterpart is. The result is a plain list of bools.
    """
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or int(count) < 0:
        raise ValueError(f"count must be a non-negative integer, got {count!r}")
    rng = random.Random(int(seed))
    return [bool(rng.random() < 0.5) for _ in range(int(count))]


# --------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------

def _correction(path, observed):
    return np.asarray(path, np.float64) - np.asarray(observed, np.float64)


def _plan_pairs(observed_path, times):
    """Build the four multiscale candidates, their nearest-crop L1 partners and the path records.

    Returns a list of comparison records (without rendered support yet). Each record carries both sides'
    correction metrics and the crop-proxy pairing/tolerance, but no image-quality number.
    """
    multiscale = [(name, weights, A.build_multiscale_output(observed_path, times, weights))
                  for name, weights in A.MULTISCALE_CANDIDATES]
    l1_outputs = [A.build_l1_output(observed_path, budget) for budget in A.L1_BUDGETS]
    l1_crops = [float(np.max(np.abs(_correction(out, observed_path)))) for out in l1_outputs]

    records = []
    for index, (name, weights, ms_output) in enumerate(multiscale):
        ms_delta = _correction(ms_output, observed_path)
        ms_metrics = correction_metrics(ms_delta)
        idx = A.nearest_crop_index(ms_metrics["max_abs_correction_px"], l1_crops)
        l1_delta = _correction(l1_outputs[idx], observed_path)
        l1_metrics = correction_metrics(l1_delta)
        records.append({
            "pair_index": int(index),
            "multiscale_name": name,
            "retention_weights": [float(w) for w in weights],
            "multiscale_delta": ms_delta,
            "multiscale": {
                "retention_weights": [float(w) for w in weights],
                "crop_proxy_px": ms_metrics["max_abs_correction_px"],
                "path": ms_metrics,
            },
            "l1_name": f"budget_{A.L1_BUDGETS[idx]:g}",
            "l1_budget": float(A.L1_BUDGETS[idx]),
            "l1_delta": l1_delta,
            "l1": {
                "budget": float(A.L1_BUDGETS[idx]),
                "crop_proxy_px": l1_metrics["max_abs_correction_px"],
                "path": l1_metrics,
            },
            "pairing": {
                "rule": "multiscale candidate paired with the L1 budget whose measured crop proxy is nearest",
                "abs_crop_mismatch_px": float(abs(ms_metrics["max_abs_correction_px"]
                                                  - l1_metrics["max_abs_correction_px"])),
                "crop_match_tolerance_px": float(A.L1_CROP_MATCH_TOLERANCE),
            },
        })
    return records


def _decode_panels(source_path, prov, motion_fps, expected_frames, out_width, out_height):
    """Probe and decode the source window, downscaling each frame; returns the panel list and decode meta."""
    info = media.probe(source_path)
    dec = prov["source_decoded"]
    _require((int(info.get("width", -1)), int(info.get("height", -1))) == (dec["width"], dec["height"]),
             "decoded_dimension_mismatch",
             f"probed source is {info.get('width')}x{info.get('height')} but manifest source.decoded is "
             f"{dec['width']}x{dec['height']}")
    interlaced = bool(info.get("interlaced", False))
    _require(interlaced == bool(prov["preprocessing"].get("interlaced")), "interlace_mismatch",
             f"probed interlaced={interlaced} does not match manifest "
             f"preprocessing.interlaced={prov['preprocessing'].get('interlaced')!r}")
    window = prov["source_window"]
    gen, eff_fps, (dec_w, dec_h) = media.decode(
        source_path, info, start=window["start_s"], duration=window["duration_s"], deinterlace=interlaced)
    _require((int(dec_w), int(dec_h)) == (dec["width"], dec["height"]), "decoded_dimension_mismatch",
             f"decoded frame size {dec_w}x{dec_h} != manifest source.decoded {dec['width']}x{dec['height']}")
    fps_tol = 1e-6 * max(1.0, abs(float(motion_fps)))
    _require(abs(float(eff_fps) - float(motion_fps)) <= fps_tol, "decoded_fps_mismatch",
             f"decoded effective fps {float(eff_fps)!r} != motion fps {float(motion_fps)!r}")

    panels = []
    for frame in gen:
        panels.append(resize_bilinear(frame, out_width, out_height))
        if len(panels) > expected_frames:
            break
    _require(len(panels) == expected_frames, "decoded_frame_count_mismatch",
             f"decoded {len(panels)} frames but the motion has {expected_frames}")
    return panels, {"width": int(dec_w), "height": int(dec_h), "fps": float(eff_fps),
                    "frame_count": int(len(panels)),
                    "deinterlace": (BWDIF_VF if interlaced else None)}


def run_review(run_dir, *, out_dir=None, motion_name=DEFAULT_MOTION_NAME,
               out_width=DEFAULT_OUT_WIDTH, out_height=DEFAULT_OUT_HEIGHT, write=True):
    """Build blinded A/B review clips for one run directory and return a JSON-safe report.

    Loads and cross-checks the motion payload/manifest (see :func:`load_review_inputs`), reduces the composed
    global homographies to the translation path ``C_t``, scales it into panel pixels, builds the four fixed
    multiscale-vs-nearest-crop L1 comparisons and renders each as one two-panel A/B MP4. The side mapping is
    randomised from :data:`BLINDING_SEED` and recorded only in the report. No image-quality metric, winner or
    ranking is computed. Validation refusals raise :class:`ReviewError`; malformed motion raises
    ``MotionPayloadError``. ``write=False`` skips clip and report output (metrics are still computed).
    """
    out_width = _validate_dim(out_width, "out_width")
    out_height = _validate_dim(out_height, "out_height")
    run_dir = Path(run_dir)
    mot, meta, prov = load_review_inputs(run_dir, motion_name=motion_name)

    n_frames = int(meta["n_frames"])
    _require(n_frames >= A.MIN_SAMPLES, "clip_too_short",
             f"motion has {n_frames} frames; the fixed audit needs at least {A.MIN_SAMPLES}")
    _require(n_frames <= A.L1_WINDOW, "clip_too_long",
             f"motion has {n_frames} frames; the fixed single-window L1 audit caps at {A.L1_WINDOW}")

    src_w, src_h = int(meta["width"]), int(meta["height"])
    fps = float(meta["fps"])
    times = np.arange(n_frames, dtype=np.float64) / fps

    full_path = composed_translation_path(mot.fits, src_w, src_h, n_frames)
    observed_path = scale_path(full_path, src_w, src_h, out_width, out_height)

    records = _plan_pairs(observed_path, times)
    assignment = blind_assignment(BLINDING_SEED, len(records))

    source_path = prov.get("source_path")
    _require(isinstance(source_path, str) and source_path.strip(), "source_path_missing",
             "manifest source.path is missing; cannot decode the review source")
    panels, decode_meta = _decode_panels(source_path, prov, fps, n_frames, out_width, out_height)

    out_dir_path = Path(out_dir) if out_dir is not None else run_dir / "review"
    encoders = {}
    support = {}
    if write:
        out_dir_path.mkdir(parents=True, exist_ok=True)
    try:
        for record, left_is_multiscale in zip(records, assignment):
            record["left_panel"] = "multiscale" if left_is_multiscale else "l1"
            record["right_panel"] = "l1" if left_is_multiscale else "multiscale"
            record["clip"] = f"pair_{record['pair_index']:02d}_ab.mp4"
            support[record["pair_index"]] = {"multiscale": [], "l1": []}
            if write:
                encoders[record["pair_index"]] = media.Encoder(
                    str(out_dir_path / record["clip"]), 2 * out_width, out_height, fps)

        for frame_index, panel in enumerate(panels):
            for record in records:
                idx = record["pair_index"]
                ms_panel, ms_valid = warp_panel(panel, record["multiscale_delta"][frame_index])
                l1_panel, l1_valid = warp_panel(panel, record["l1_delta"][frame_index])
                support[idx]["multiscale"].append(float(ms_valid.mean()))
                support[idx]["l1"].append(float(l1_valid.mean()))
                left = ms_panel if record["left_panel"] == "multiscale" else l1_panel
                right = l1_panel if record["left_panel"] == "multiscale" else ms_panel
                if write:
                    encoders[idx].write(assemble_ab_panel(left, right))
    finally:
        for encoder in encoders.values():
            encoder.close()

    pairs_report = []
    for record, left_is_multiscale in zip(records, assignment):
        idx = record["pair_index"]
        record["multiscale"]["support"] = support_summary(support[idx]["multiscale"])
        record["l1"]["support"] = support_summary(support[idx]["l1"])
        pairs_report.append({
            "pair_index": idx,
            "multiscale": {
                "name": record["multiscale_name"],
                "retention_weights": record["retention_weights"],
                "crop_proxy_px": record["multiscale"]["crop_proxy_px"],
                "path": record["multiscale"]["path"],
                "geometric_support": record["multiscale"]["support"],
            },
            "l1": {
                "name": record["l1_name"],
                "budget": record["l1_budget"],
                "crop_proxy_px": record["l1"]["crop_proxy_px"],
                "path": record["l1"]["path"],
                "geometric_support": record["l1"]["support"],
            },
            "pairing": record["pairing"],
            "clip": record["clip"],
            "blind_mapping": {
                "left_panel": record["left_panel"],
                "right_panel": record["right_panel"],
                "video_labels": {"left": "A", "right": "B"},
            },
        })

    report = {
        "schema": SCHEMA,
        "ok": True,
        "scope_note": SCOPE_NOTE,
        "review_policy": REVIEW_POLICY,
        "no_quality_score_note": (
            "This report contains path, crop-proxy and geometric-support numbers plus provenance only. It "
            "computes no image-quality score, no ranking and no winner, and it does not select a candidate."
        ),
        "translation_model": dict(TRANSLATION_MODEL),
        "provenance": {
            "run": prov["run"],
            "manifest": prov["manifest"],
            "motion": motion_name,
            "case": prov["case"],
            "run_id": prov["run_id"],
            "manifest_schema": prov["manifest_schema"],
            "source_sha256": prov["source_sha256"],
            "source_sha256_verified": prov["source_sha256_verified"],
            "source_hash_note": prov["source_hash_note"],
            "source_window": prov["source_window"],
            "source_decoded": prov["source_decoded"],
            "preprocessing": prov["preprocessing"],
            "motion": prov["motion"],
            "capture_dump_motion": prov["capture_dump_motion"],
            "checks": ["status_success", "outputs_motion_basename", "capture_dump_motion",
                       "source_sha256", "source_window", "dimensions", "fps", "frame_count", "stride_1"],
        },
        "decode": decode_meta,
        "output": {
            "dir": out_dir_path.name,
            "panel_width": out_width,
            "panel_height": out_height,
            "frame_width": 2 * out_width,
            "frame_height": out_height,
            "fps": fps,
            "frames": n_frames,
            "comparisons": len(pairs_report),
            "clips": [record["clip"] for record in records],
        },
        "path_units": {
            "units": "panel pixels (downscaled review resolution)",
            "scale": {"x": out_width / src_w, "y": out_height / src_h},
        },
        "blinding": {
            "seed": int(BLINDING_SEED),
            "rule": "one random draw per comparison: True => multiscale is the left/A panel, False => L1 is",
            "assignment": ["multiscale" if left else "l1" for left in assignment],
            "hidden_mapping": "recorded per pair below and in each clip's blind_mapping; the MP4 carries only A/B",
        },
        "grids": {
            "multiscale_candidates": [
                {"name": name, "retention_weights": [float(w) for w in weights]}
                for name, weights in A.MULTISCALE_CANDIDATES
            ],
            "multiscale_scales_seconds": [float(s) for s in A.MULTISCALE_SCALES_SECONDS],
            "l1_budgets": [float(b) for b in A.L1_BUDGETS],
            "l1_crop_match_tolerance_px": float(A.L1_CROP_MATCH_TOLERANCE),
        },
        "pairs": pairs_report,
        "limitations": list(LIMITATIONS),
    }
    if write:
        write_manifest(out_dir_path / "review.json", report)
    return report


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="bench/multiscale_real_review.py",
        description="Build blinded A/B MP4 review clips for the fixed multiscale-vs-nearest-crop-L1 "
                    "comparison from a successful run directory (manifest.json + motion_r0.npz). Local, "
                    "translation-only, no image-quality score and no winner.")
    p.add_argument("run_dir", help="successful benchmark run directory containing manifest.json and motion_r0.npz")
    p.add_argument("--out-dir", default=None, help="output directory (default: RUN_DIR/review)")
    p.add_argument("--motion-name", default=DEFAULT_MOTION_NAME, help="motion payload filename (default: %(default)s)")
    p.add_argument("--width", type=int, default=DEFAULT_OUT_WIDTH, help="panel width (default: %(default)s)")
    p.add_argument("--height", type=int, default=DEFAULT_OUT_HEIGHT, help="panel height (default: %(default)s)")
    p.add_argument("--no-write", action="store_true", help="compute metrics only; write no clips or report")
    return p


def _error_payload(exc, run_dir=None):
    payload = {"schema": SCHEMA, "ok": False, "error": {"code": exc.code, "message": str(exc)}}
    if run_dir is not None:
        payload["run"] = Path(run_dir).name
    return payload


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_review(args.run_dir, out_dir=args.out_dir, motion_name=args.motion_name,
                            out_width=args.width, out_height=args.height, write=not args.no_write)
    except (ReviewError, MotionPayloadError, CompositionAuditError) as exc:
        payload = (_error_payload(exc, args.run_dir) if isinstance(exc, ReviewError)
                   else {"schema": SCHEMA, "ok": False, "error": {"code": exc.code, "message": str(exc)},
                         "run": Path(args.run_dir).name})
        print(json.dumps(json_safe(payload), indent=1, sort_keys=True, allow_nan=False))
        return 2
    except (ValueError, IndexError) as exc:
        payload = {"schema": SCHEMA, "ok": False,
                   "error": {"code": "invalid_input", "message": f"{type(exc).__name__}: {exc}"}}
        print(json.dumps(json_safe(payload), indent=1, sort_keys=True, allow_nan=False))
        return 2
    print(f"schema={report['schema']} comparisons={report['output']['comparisons']} "
          f"clips={report['output']['clips']} out={report['output']['dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA",
    "MANIFEST_NAME",
    "DEFAULT_MOTION_NAME",
    "DEFAULT_OUT_WIDTH",
    "DEFAULT_OUT_HEIGHT",
    "BWDIF_VF",
    "BLINDING_SEED",
    "FONT",
    "SCOPE_NOTE",
    "TRANSLATION_MODEL",
    "LIMITATIONS",
    "REVIEW_POLICY",
    "ReviewError",
    "manifest_provenance",
    "load_review_inputs",
    "frame_center",
    "composed_translation_path",
    "scale_path",
    "correction_metrics",
    "support_summary",
    "resize_bilinear",
    "warp_panel",
    "draw_bitmap_label",
    "assemble_ab_panel",
    "blind_assignment",
    "run_review",
    "build_parser",
    "main",
]
