"""Research-only E10.2 multi-layer rendered-pixel audit: fixed multiscale camera paths versus their
nearest-crop L1 counterparts on a deterministic three-layer parallax scene.

What this measures
------------------
E10 (``bench/multiscale_motion_audit.py``) measured translation *paths* only. E10.1
(``bench/multiscale_render_audit.py``) added a rendered-pixel stage but on a **single-layer, static,
zero-parallax** scene: a uniform camera warp could reproduce it exactly, so depth ordering, parallax and
independent subject motion were all out of scope.

E10.2 renders the *same* four fixed multiscale-retention camera paths and their nearest measured-crop L1
counterparts through a deterministic **three-layer** scene and compares rendered pixels:

- layer 0 **background**: a textured static far layer (a full-canvas procedural texture),
- layer 1 **foreground**: an opaque rectangular foreground/parallax geometry slot,
- layer 2 **subject**: an opaque rectangular subject that additionally moves on its own.

Every layer is displaced by a **depth-dependent** multiple of the camera path (``depth_scale[l] * camera``),
and the subject layer adds an independent, deterministic subject motion. The observed frames are the layered
render under the noisy observed camera path; the ideal frames are the layered render under the known intended
pan, with the same true subject motion. ``render_layered_scene`` resolves occlusion front-to-back (subject
over foreground over background) so each output pixel has one known **ideal semantic region**. This is the
layer/occlusion/parallax/subject-motion stress the E10.1 flat scene could not represent.

The candidate frames use the same rendered convention as E10.1: the candidate camera path is turned into a
view by backward-warping the observed pixels at ``candidate - observed`` (one uniform correction per frame).
That single uniform warp *cannot* represent per-layer parallax or independent subject motion, which is exactly
what the per-region metrics expose.

Camera / layer convention (explicit, and tested)
-----------------------------------------------
With output grid ``(x, y)`` on ``x in [0, W-1]``, ``y in [0, H-1]``, camera path ``C_t``, depth scale
``d_l`` and subject motion ``s_t`` (subject layer only):

    layer sample coordinate   = grid + margin + d_l * C_t          (plus s_t on the subject layer)
    observed frame            = layered composite under C_t
    ideal frame               = layered composite under pan_t (same s_t)
    candidate frame           = backward_warp(observed, P_t - C_t)

Two signs follow and are asserted: ``P = C`` reproduces the observed frame exactly, and if every
``d_l = 1`` with ``s_t = 0`` then a candidate with ``P = C + delta`` is the observed frame sampled at
``grid + delta`` (the E10.1 translation convention). A source coordinate is valid only when every bilinear
tap is in bounds; MAE/RMS are RGB-channel means over the intersection of candidate-valid and ideal-valid
pixels, so uncovered/zero pixels are never penalized.

Metrics
-------
Each of the four fixed multiscale candidates is paired, via
``bench.multiscale_motion_audit.nearest_crop_index`` (same deterministic tie rule: ties to the earlier grid
index), with the L1-budget path whose measured crop proxy is nearest. For each pair, **separately**:

- the E10 path metrics (RMS path deviation, residual second-difference RMS, maximum correction) for context,
- **whole-frame** rendered RGB MAE/RMS plus candidate/ideal/common valid coverage,
- the same common-support MAE/RMS/coverage computed **per ideal semantic region** (background, foreground,
  subject), each with explicit pixel counts (region pixels, candidate-valid, ideal-valid, common-valid) so a
  vanishing region cannot hide behind a pooled number.

Nothing is pooled across pairs, no winner/rank/score/aggregate is produced, no weights are tuned and no
candidate is asserted superior. All fixed parameters are recorded verbatim in the JSON-safe report.

Conditions (fixed isolation grid)
---------------------------------
- ``static_no_parallax``: zero camera motion, equal layer depths (no parallax) and no subject motion. This is
  the exact-identity control: the observed and ideal layered renders are identical and every candidate must
  reproduce them.
- ``parallax_shake``: the E10 ``pan_shake`` camera path plus depth-dependent parallax (background 0.5 vs
  foreground/subject 1.0) and no subject motion.
- ``parallax_subject_jolt``: the E10 ``pan_jolt`` camera path plus parallax plus an independent deterministic
  subject motion.

What this is not
----------------
It is a **deterministic synthetic three-layer caricature**, not real footage: fixed procedural textures, fixed
axis-aligned rectangular masks, hard (thresholded) opacity, scalar "depth" that is only a displacement
multiplier, translation-only pure-NumPy bilinear warps, and a single uniform candidate warp. It is not
perceptual quality, not a physical depth/segmentation model, not occlusion handling by any production method,
and not the production renderer. It changes no production default and imports no production module directly.
Reports are JSON-safe (finite-only) and path-free.

Run ``python -m bench.multiscale_layered_render_audit [--seeds 0,1,2]
[--conditions static_no_parallax,parallax_shake,parallax_subject_jolt] [--n 60] [--width 80] [--height 48]
[--margin 20] [--fps 60] [--out f.json]``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import multiscale_motion_audit as A
from . import multiscale_render_audit as R
from .manifest import json_safe, write_manifest

SCHEMA = "mlxstab.multiscale_layered_render_audit/1"

# --------------------------------------------------------------------------------------
# fixed layer model (the experiment, not tunables)
# --------------------------------------------------------------------------------------

BACKGROUND = 0
FOREGROUND = 1
SUBJECT = 2
LAYER_NAMES = ("background", "foreground", "subject")

# Fixed per-layer procedural-texture seeds. The three textures are distinct so a layer-specific displacement
# (parallax) and independent subject motion are visible in the rendered pixels. No media, asset or weight.
LAYER_TEXTURE_SEEDS = (1729, 2718, 3141)

# Fixed mask geometry in normalized output-grid coordinates ``(cx, cy, half_w, half_h)``. The foreground and
# subject rectangles are opaque; the subject is composited in front so the overlap tests occlusion ordering.
BACKGROUND_RECT = None  # the background covers the whole finite canvas
FOREGROUND_RECT = (0.42, 0.55, 0.24, 0.30)
SUBJECT_RECT = (0.60, 0.48, 0.14, 0.22)

# Fixed independent subject motion: a deterministic per-channel sinusoid whose phases depend on the run seed.
SUBJECT_PERIOD_SECONDS = 0.8
SUBJECT_AMPLITUDE = (1.0, 0.7)
SUBJECT_SEED_OFFSET = 101

# Fixed depth scales (displacement multipliers) per condition. Depth is a scalar displacement multiplier, not
# a physical depth: a larger scale means the layer responds more strongly to camera translation.
FLAT_DEPTH_SCALES = (1.0, 1.0, 1.0)
PARALLAX_DEPTH_SCALES = (0.5, 1.0, 1.0)

# Fixed condition specs chosen to isolate one effect at a time.
CONDITION_SPECS = {
    "static_no_parallax": {
        "camera_condition": "static",
        "depth_scales": FLAT_DEPTH_SCALES,
        "subject_motion": False,
        "parallax": False,
        "isolates": (
            "zero camera motion with equal layer depth scales and no subject motion: the exact-identity "
            "control (no camera shake, zero parallax, no independent subject motion)"
        ),
    },
    "parallax_shake": {
        "camera_condition": "pan_shake",
        "depth_scales": PARALLAX_DEPTH_SCALES,
        "subject_motion": False,
        "parallax": True,
        "isolates": (
            "depth-dependent parallax (background 0.5 vs foreground/subject 1.0) combined with the "
            "sinusoidal pan and periodic shake camera path, with no subject motion"
        ),
    },
    "parallax_subject_jolt": {
        "camera_condition": "pan_jolt",
        "depth_scales": PARALLAX_DEPTH_SCALES,
        "subject_motion": True,
        "parallax": True,
        "isolates": (
            "depth-dependent parallax combined with the one-sided transient jolt camera path and an "
            "independent deterministic subject motion"
        ),
    },
}
CONDITIONS = tuple(CONDITION_SPECS)
DEFAULT_CONDITIONS = CONDITIONS

DEFAULT_N = 60
DEFAULT_FPS = 60.0
DEFAULT_WIDTH = 80
DEFAULT_HEIGHT = 48
DEFAULT_MARGIN = 20
DEFAULT_SEEDS = A.DEFAULT_SEEDS

MIN_DIM = 1
MIN_MARGIN = 0

SCOPE_NOTE = (
    "Research-only rendered-pixel audit of a deterministic three-layer parallax scene (textured static "
    "background, opaque foreground geometry, independent subject) sampled with pure-NumPy bilinear "
    "translation-only warps. It compares four fixed multiscale-retention camera paths with their nearest "
    "measured-crop L1 counterparts on rendered RGB error against the known ideal layered render, reporting "
    "whole-frame and per-ideal-region metrics separately. It is not real footage, not perceptual quality, "
    "not physical depth, not a segmentation or occlusion model, and it makes no superiority claim."
)

REPORTING_POLICY = (
    "Every condition/seed/pair/region metric is reported separately. No aggregate winner, ranking, total "
    "score or weight optimization is computed, no per-seed tuning is performed and no candidate or family is "
    "asserted superior."
)

PIXEL_METRICS = [
    "whole-frame rgb_mae / rgb_rms: RGB-channel-mean absolute / root-mean-square error of the candidate "
    "against the ideal layered render on the common candidate-valid and ideal-valid support (normalized "
    "intensity [0, 1])",
    "whole-frame candidate_valid_fraction_of_total / ideal_valid_fraction_of_total: valid coverage over all "
    "output pixels",
    "whole-frame common_valid_fraction_of_total / common_valid_fraction_of_ideal_valid: common-support "
    "coverage over the total and over the ideal-valid support",
]

REGION_METRICS = [
    "region_pixels: output pixels assigned to this ideal semantic region",
    "candidate_valid_region_pixels / ideal_valid_region_pixels: region pixels valid for each side",
    "common_valid_region_pixels: region pixels valid for both sides (the MAE/RMS support)",
    "common_valid_fraction_of_region: common_valid_region_pixels / region_pixels",
    "common_valid_fraction_of_ideal_valid_region: common_valid_region_pixels / ideal_valid_region_pixels",
    "rgb_mae / rgb_rms: RGB-channel-mean MAE / RMS of the candidate against the ideal layered render on the "
    "common valid support inside this ideal region (None when that support is empty)",
]

LAYER_MODEL = {
    "layers": "background (textured static far layer), foreground (opaque geometry), subject (opaque mover)",
    "composition": "front-to-back: subject over foreground over background; the frontmost occupied layer wins",
    "opacity": "hard masks; a bilinear-sampled layer alpha in [0, 1] is occupied iff alpha >= 0.5",
    "depth": "scalar displacement multiplier per layer, not a physical depth; layer displacement = "
             "depth_scale * camera path (subject adds its independent motion)",
    "candidate_model": "one uniform camera correction per frame: backward-warp observed pixels at "
                       "candidate - observed camera path",
}

GROUND_TRUTH = {
    "observed": "layered scene under the noisy observed camera path C_t = pan_t + shake_t, with the true "
                "subject motion s_t",
    "ideal": "layered scene under the known intended pan_t, with the same true subject motion s_t",
    "candidate": "backward_warp(observed, candidate_path - observed_path)",
    "ideal_semantic_masks": "the per-pixel winning layer (background/foreground/subject) of the ideal render; "
                            "region metrics are computed on this mask",
}

LIMITATIONS = (
    "deterministic synthetic three-layer caricature: not real footage and not a physical scene",
    "layer depth is a fixed scalar displacement multiplier, not a physical depth map",
    "foreground and subject are axis-aligned rectangles with hard thresholded opacity, not real geometry, "
    "segmentation or soft alpha",
    "translation-only pure-NumPy bilinear warps: no rotation, scale, perspective, rolling shutter or lens",
    "the candidate is a single uniform camera warp per frame and cannot represent per-layer parallax or "
    "independent subject motion; the per-region metrics expose that residual rather than hiding it",
    "ideal semantic masks come from the ideal render's layer assignment; region metrics are on the common "
    "candidate-valid/ideal-valid support inside each ideal region",
    "rendered RGB MAE/RMS on a synthetic cartoon is not perceptual quality and not a human preference study",
    "coverage is a geometric source-frame validity fraction, not a real stabilizer crop measurement",
    "the four fixed multiscale weights and the predeclared uniform L1 budget grid are reported verbatim; "
    "nothing is tuned or swept",
    "nearest-crop pairing is descriptive: it produces no winner, rank, score, aggregate or tuned selection",
    "no production module is imported directly and no production default is changed; there is no media, MLX, "
    "torch or weight dependency",
)


# --------------------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------------------

def _validate_dim(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    v = int(value)
    if v < MIN_DIM:
        raise ValueError(f"{name} must be >= {MIN_DIM}, got {v}")
    return v


def _validate_margin(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"margin must be an integer, got {value!r}")
    v = int(value)
    if v < MIN_MARGIN:
        raise ValueError(f"margin must be >= {MIN_MARGIN}, got {v}")
    return v


def _validate_condition(condition):
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    return str(condition)


def _validate_conditions(conditions):
    seq = A._as_sequence(conditions, "conditions")
    if not seq:
        raise ValueError("conditions must be a non-empty sequence")
    out = []
    for c in seq:
        v = _validate_condition(c)
        if v in out:
            raise ValueError(f"duplicate condition {v!r}")
        out.append(v)
    return tuple(out)


def _validate_rect(rect, name):
    if rect is None:
        return None
    try:
        vals = [float(v) for v in rect]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a 4-tuple (cx, cy, half_w, half_h), got {rect!r}")
    if len(vals) != 4:
        raise ValueError(f"{name} must have exactly 4 entries, got {rect!r}")
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{name} must be finite, got {rect!r}")
    cx, cy, hw, hh = vals
    if hw <= 0.0 or hh <= 0.0:
        raise ValueError(f"{name} half-width and half-height must be > 0, got {rect!r}")
    if cx - hw < -1e-12 or cx + hw > 1.0 + 1e-12 or cy - hh < -1e-12 or cy + hh > 1.0 + 1e-12:
        raise ValueError(f"{name} must lie inside the normalized [0, 1] output grid, got {rect!r}")
    return (cx, cy, hw, hh)


# --------------------------------------------------------------------------------------
# deterministic layer canvases and subject motion
# --------------------------------------------------------------------------------------

def _rect_to_canvas(rect, width, height, margin):
    """Map a normalized ``(cx, cy, half_w, half_h)`` rect to a half-open canvas pixel rectangle."""
    cx, cy, hw, hh = [float(v) for v in rect]
    canvas_w = width + 2 * margin
    canvas_h = height + 2 * margin
    x0 = int(round(margin + (cx - hw) * width))
    x1 = int(round(margin + (cx + hw) * width))
    y0 = int(round(margin + (cy - hh) * height))
    y1 = int(round(margin + (cy + hh) * height))
    x0 = min(max(x0, 0), canvas_w - 1)
    x1 = min(max(x1, x0 + 1), canvas_w)
    y0 = min(max(y0, 0), canvas_h - 1)
    y1 = min(max(y1, y0 + 1), canvas_h)
    return x0, y0, x1, y1


def make_layer_canvases(width, height, margin, *, seeds=LAYER_TEXTURE_SEEDS,
                        foreground_rect=FOREGROUND_RECT, subject_rect=SUBJECT_RECT):
    """Deterministic list of three RGBA canvases of shape ``[H+2m, W+2m, 4]`` in ``[0, 1]``.

    Channel 3 is the binary layer opacity: the background covers the whole canvas, while the foreground and
    subject canvases are opaque only inside their fixed rectangles. Each layer's RGB is a distinct fixed
    procedural texture (``bench.multiscale_render_audit.make_scene``). With the same arguments the canvases
    are bit-identical: no media, asset, weight or run-seed dependence.
    """
    width = _validate_dim(width, "width")
    height = _validate_dim(height, "height")
    margin = _validate_margin(margin)
    try:
        seed_list = [int(s) for s in seeds]
    except (TypeError, ValueError):
        raise ValueError(f"seeds must be three integers, got {seeds!r}")
    if len(seed_list) != len(LAYER_NAMES):
        raise ValueError(f"seeds must have exactly {len(LAYER_NAMES)} entries, got {seeds!r}")
    foreground_rect = _validate_rect(foreground_rect, "foreground_rect")
    subject_rect = _validate_rect(subject_rect, "subject_rect")

    canvas_h = height + 2 * margin
    canvas_w = width + 2 * margin
    canvases = []
    for layer in range(len(LAYER_NAMES)):
        rgb = R.make_scene(canvas_h, canvas_w, seed=seed_list[layer])
        alpha = np.ones((canvas_h, canvas_w), dtype=np.float64)
        rect = None
        if layer == FOREGROUND:
            rect = foreground_rect
        elif layer == SUBJECT:
            rect = subject_rect
        if rect is not None:
            alpha[:] = 0.0
            x0, y0, x1, y1 = _rect_to_canvas(rect, width, height, margin)
            alpha[y0:y1, x0:x1] = 1.0
        canvases.append(np.concatenate([rgb, alpha[..., None]], axis=-1))
    return canvases


def subject_motion_path(times, *, seed=0, amplitude=SUBJECT_AMPLITUDE, period=SUBJECT_PERIOD_SECONDS):
    """Deterministic independent subject motion ``[T, 2]`` in ``(x, y)`` pixels.

    A fixed per-channel sinusoid ``amplitude[d] * sin(2*pi*t/period + phase[d])`` whose phases are drawn from
    ``numpy.random.default_rng(seed + SUBJECT_SEED_OFFSET)``. It is bounded by ``|amplitude[d]|`` and does not
    depend on the camera path: the subject moves on its own.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or times.shape[0] < 1 or not bool(np.isfinite(times).all()):
        raise ValueError(f"times must be a finite 1-D array, got shape {times.shape}")
    try:
        amp = [float(v) for v in amplitude]
    except (TypeError, ValueError):
        raise ValueError(f"amplitude must be a 2-tuple, got {amplitude!r}")
    if len(amp) != 2 or not all(math.isfinite(v) for v in amp):
        raise ValueError(f"amplitude must be two finite numbers, got {amplitude!r}")
    try:
        per = float(period)
    except (TypeError, ValueError):
        raise ValueError(f"period must be a finite positive number, got {period!r}")
    if not math.isfinite(per) or per <= 0.0:
        raise ValueError(f"period must be finite and > 0, got {period!r}")

    rng = np.random.default_rng(int(seed) + SUBJECT_SEED_OFFSET)
    two_pi = 2.0 * np.pi
    t = times - times[0]
    phase = rng.uniform(0.0, two_pi, size=2)
    return np.stack([amp[d] * np.sin(two_pi * t / per + phase[d]) for d in range(2)], axis=1)


def max_abs_layer_displacement(depth_scales, camera_path, subject_motion):
    """Largest absolute per-axis layer displacement implied by the depth scales, camera path and subject motion.

    Used to check the margin covers the motion (a source coordinate is valid only when every bilinear tap is
    in bounds, which requires ``max|displacement| <= margin``).
    """
    scales = [float(v) for v in depth_scales]
    camera_path = np.asarray(camera_path, dtype=np.float64)
    subject_motion = np.asarray(subject_motion, dtype=np.float64)
    if camera_path.ndim != 2 or camera_path.shape[1] != 2:
        raise ValueError(f"camera_path must be [T, 2], got shape {camera_path.shape}")
    if subject_motion.shape != camera_path.shape:
        raise ValueError(
            f"subject_motion must share the camera path shape {camera_path.shape}, got {subject_motion.shape}")
    biggest = 0.0
    for layer in range(len(LAYER_NAMES)):
        disp = scales[layer] * camera_path
        if layer == SUBJECT:
            disp = disp + subject_motion
        biggest = max(biggest, float(np.abs(disp).max(initial=0.0)))
    return biggest


# --------------------------------------------------------------------------------------
# layered renderer
# --------------------------------------------------------------------------------------

def render_layered_scene(canvases, depth_scales, camera_path, subject_motion, *, width, height, margin):
    """Render a deterministic three-layer composition.

    ``canvases`` is the list from :func:`make_layer_canvases` (RGBA, binary alpha). Layer ``l`` samples its
    canvas at ``grid + margin + depth_scales[l] * camera_path`` (the subject layer also adds its independent
    ``subject_motion``), front-to-back composited so the frontmost occupied layer wins. Returns
    ``(frames [T, H, W, 3], valid [T, H, W], region [T, H, W])`` where ``region`` holds the winning layer
    index (0 background, 1 foreground, 2 subject) or ``-1`` where no layer is valid. All returned values are
    finite NumPy arrays.
    """
    if not isinstance(canvases, (list, tuple)) or len(canvases) != len(LAYER_NAMES):
        raise ValueError(f"canvases must be a list of {len(LAYER_NAMES)} RGBA canvases, got {canvases!r}")
    width = _validate_dim(width, "width")
    height = _validate_dim(height, "height")
    margin = _validate_margin(margin)
    scales = [float(v) for v in depth_scales]
    if len(scales) != len(LAYER_NAMES):
        raise ValueError(f"depth_scales must have {len(LAYER_NAMES)} entries, got {depth_scales!r}")
    camera_path = np.asarray(camera_path, dtype=np.float64)
    subject_motion = np.asarray(subject_motion, dtype=np.float64)
    if camera_path.ndim != 2 or camera_path.shape[1] != 2:
        raise ValueError(f"camera_path must be [T, 2] (x, y), got shape {camera_path.shape}")
    if subject_motion.shape != camera_path.shape:
        raise ValueError(
            f"subject_motion must share the camera path shape {camera_path.shape}, got {subject_motion.shape}")
    if not bool(np.isfinite(camera_path).all()) or not bool(np.isfinite(subject_motion).all()):
        raise ValueError("camera_path and subject_motion must be finite")

    t_count = camera_path.shape[0]
    base = np.stack(np.meshgrid(np.arange(width, dtype=np.float64),
                                np.arange(height, dtype=np.float64), indexing="xy"), axis=-1)  # [H, W, 2]
    frames = np.zeros((t_count, height, width, 3), dtype=np.float64)
    region = np.full((t_count, height, width), -1, dtype=np.int64)
    for layer, canvas in enumerate(canvases):
        canvas = np.asarray(canvas, dtype=np.float64)
        if canvas.ndim != 3 or canvas.shape[2] != 4:
            raise ValueError(f"each canvas must be [Hc, Wc, 4], got shape {canvas.shape}")
        disp = scales[layer] * camera_path
        if layer == SUBJECT:
            disp = disp + subject_motion
        coords = base[None, :, :, :] + float(margin) + disp[:, None, None, :]
        images = np.broadcast_to(canvas, (t_count,) + canvas.shape)
        rgba, layer_valid = R.bilinear_sample(images, coords)
        occupied = layer_valid & (rgba[..., 3] >= 0.5)
        frames = np.where(occupied[..., None], rgba[..., :3], frames)
        region = np.where(occupied, layer, region)
    valid = region >= 0
    return frames, valid, region


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------

def region_pixel_metrics(candidate_frames, candidate_valid, ideal_frames, ideal_valid, region, region_id):
    """Common-support MAE/RMS plus coverage counts for one ideal semantic region.

    ``region`` is the ideal render's per-pixel winning-layer index. MAE/RMS are RGB-channel means over the
    intersection of ``candidate_valid``, ``ideal_valid`` and ``region == region_id`` (empty support yields
    ``None``). Every count is reported so a region that vanishes cannot hide behind a pooled number. All
    returned values are plain Python (finite floats or ints, or ``None``).
    """
    candidate_frames = np.asarray(candidate_frames, dtype=np.float64)
    ideal_frames = np.asarray(ideal_frames, dtype=np.float64)
    candidate_valid = np.asarray(candidate_valid, dtype=bool)
    ideal_valid = np.asarray(ideal_valid, dtype=bool)
    region = np.asarray(region)
    if candidate_frames.shape != ideal_frames.shape or candidate_frames.ndim != 4:
        raise ValueError(
            f"candidate_frames and ideal_frames must share one [T, H, W, C] shape, got "
            f"{candidate_frames.shape} and {ideal_frames.shape}")
    if candidate_valid.shape != candidate_frames.shape[:3] or ideal_valid.shape != candidate_frames.shape[:3]:
        raise ValueError("valid masks must match the frames' [T, H, W] shape")
    if region.shape != candidate_frames.shape[:3]:
        raise ValueError("region must match the frames' [T, H, W] shape")
    try:
        rid = int(region_id)
    except (TypeError, ValueError):
        raise ValueError(f"region_id must be an integer, got {region_id!r}")

    selected = region == rid
    n_region = int(np.count_nonzero(selected))
    candidate_region = candidate_valid & selected
    ideal_region = ideal_valid & selected
    common = candidate_region & ideal_region
    n_candidate = int(np.count_nonzero(candidate_region))
    n_ideal = int(np.count_nonzero(ideal_region))
    n_common = int(np.count_nonzero(common))
    channels = int(candidate_frames.shape[3])

    if n_common > 0:
        weight = common[..., None]
        diff = candidate_frames - ideal_frames
        rgb_mae = float(np.sum(np.abs(diff) * weight) / (n_common * channels))
        rgb_rms = float(np.sqrt(np.sum((diff * diff) * weight) / (n_common * channels)))
    else:
        rgb_mae = None
        rgb_rms = None

    return {
        "region_pixels": n_region,
        "candidate_valid_region_pixels": n_candidate,
        "ideal_valid_region_pixels": n_ideal,
        "common_valid_region_pixels": n_common,
        "common_valid_fraction_of_region": float(n_common / n_region) if n_region > 0 else 0.0,
        "common_valid_fraction_of_ideal_valid_region": float(n_common / n_ideal) if n_ideal > 0 else 0.0,
        "rgb_mae": rgb_mae,
        "rgb_rms": rgb_rms,
    }


def _region_report(candidate_frames, candidate_valid, ideal_frames, ideal_valid, region):
    return {
        name: region_pixel_metrics(candidate_frames, candidate_valid, ideal_frames, ideal_valid, region, idx)
        for idx, name in enumerate(LAYER_NAMES)
    }


# --------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------

def _path_report(pan, observed, output):
    """Small subset of the E10 path metrics used for context (plain Python, finite)."""
    m = A.translation_path_metrics(pan, observed, output)
    return {
        "rms_path_deviation": float(m["rms_path_deviation"]),
        "rms_second_difference_residual": float(m["rms_second_difference_residual"]),
        "max_abs_correction_from_observed": float(m["max_abs_correction_from_observed"]),
    }


def _render_candidate(observed_frames, observed_path, output_path):
    """Candidate virtual view: backward-warp the observed frames at ``output - observed``."""
    delta = np.asarray(output_path, dtype=np.float64) - np.asarray(observed_path, dtype=np.float64)
    return R.backward_warp(observed_frames, delta)


def _side_report(frames, valid, ideal_frames, ideal_valid, ideal_region):
    return {
        "whole_frame": R.rendered_pixel_metrics(frames, valid, ideal_frames, ideal_valid),
        "regions": _region_report(frames, valid, ideal_frames, ideal_valid, ideal_region),
    }


def _run_one(condition, seed, n, fps, width, height, margin, canvases):
    spec = CONDITION_SPECS[condition]
    fx = A.synthetic_translation_fixture(n=n, fps=fps, seed=seed, condition=spec["camera_condition"])
    times, pan, shake, observed = fx["times"], fx["pan"], fx["shake"], fx["observed"]
    depth_scales = [float(v) for v in spec["depth_scales"]]

    if spec["subject_motion"]:
        subject_motion = subject_motion_path(times, seed=seed)
    else:
        subject_motion = np.zeros((n, 2), dtype=np.float64)

    required = max(
        max_abs_layer_displacement(depth_scales, observed, subject_motion),
        max_abs_layer_displacement(depth_scales, pan, subject_motion),
    )
    if required > margin + 1e-9:
        raise ValueError(
            f"margin {margin} is too small for the {condition!r} motion (max layer displacement "
            f"{required:.3f}); use margin >= {int(math.ceil(required))} so the layered render has no holes")

    observed_frames, observed_valid, observed_region = render_layered_scene(
        canvases, depth_scales, observed, subject_motion, width=width, height=height, margin=margin)
    ideal_frames, ideal_valid, ideal_region = render_layered_scene(
        canvases, depth_scales, pan, subject_motion, width=width, height=height, margin=margin)
    if not bool(observed_valid.all()) or not bool(ideal_valid.all()):
        raise ValueError(
            f"layered render left holes for {condition!r} seed {seed} at margin {margin}; increase margin")

    multiscale = []
    for name, weights in A.MULTISCALE_CANDIDATES:
        output = A.build_multiscale_output(observed, times, weights)
        multiscale.append((name, weights, output))

    l1_outputs = [A.build_l1_output(observed, budget) for budget in A.L1_BUDGETS]
    l1_crops = [_path_report(pan, observed, out)["max_abs_correction_from_observed"] for out in l1_outputs]

    rendered_l1 = {}
    pairs = []
    for name, weights, ms_output in multiscale:
        ms_path = _path_report(pan, observed, ms_output)
        idx = A.nearest_crop_index(ms_path["max_abs_correction_from_observed"], l1_crops)
        if idx not in rendered_l1:
            rendered_l1[idx] = _render_candidate(observed_frames, observed, l1_outputs[idx])
        ms_frames, ms_valid = _render_candidate(observed_frames, observed, ms_output)
        l1_frames, l1_valid = rendered_l1[idx]
        pairs.append({
            "multiscale_name": name,
            "l1_name": f"budget_{A.L1_BUDGETS[idx]:g}",
            "multiscale_crop_proxy": float(ms_path["max_abs_correction_from_observed"]),
            "l1_crop_proxy": float(l1_crops[idx]),
            "abs_crop_mismatch": float(abs(ms_path["max_abs_correction_from_observed"] - l1_crops[idx])),
            "crop_match_tolerance": float(A.L1_CROP_MATCH_TOLERANCE),
            "multiscale": {
                "retention_weights": [float(w) for w in weights],
                "path": ms_path,
                **_side_report(ms_frames, ms_valid, ideal_frames, ideal_valid, ideal_region),
            },
            "l1": {
                "budget": float(A.L1_BUDGETS[idx]),
                "path": _path_report(pan, observed, l1_outputs[idx]),
                **_side_report(l1_frames, l1_valid, ideal_frames, ideal_valid, ideal_region),
            },
        })

    return {
        "condition": condition,
        "seed": int(seed),
        "fixture": {
            "n_samples": int(n),
            "fps": float(fps),
            "dt_seconds": float(1.0 / fps),
            "duration_seconds": float(times[-1] - times[0]),
            "timestamp_basis": "uniform",
            "camera_condition": spec["camera_condition"],
            "camera_condition_model": A.CONDITION_MODELS[spec["camera_condition"]],
            "camera_condition_kind": A.CONDITION_KIND[spec["camera_condition"]],
            "condition_isolates": spec["isolates"],
            "depth_scales": depth_scales,
            "parallax_present": bool(spec["parallax"]),
            "subject_motion_present": bool(spec["subject_motion"]),
            "width_px": int(width),
            "height_px": int(height),
            "margin_px": int(margin),
            "layer_texture_seeds": [int(s) for s in LAYER_TEXTURE_SEEDS],
            "foreground_rect": [float(v) for v in FOREGROUND_RECT],
            "subject_rect": [float(v) for v in SUBJECT_RECT],
            "subject_period_seconds": float(SUBJECT_PERIOD_SECONDS),
            "subject_amplitude": [float(v) for v in SUBJECT_AMPLITUDE],
            "subject_seed_offset": int(SUBJECT_SEED_OFFSET),
            "max_abs_layer_displacement": float(required),
            "pan_rms": float(np.sqrt(np.mean(pan * pan))),
            "shake_rms": float(np.sqrt(np.mean(shake * shake))),
            "observed_rms": float(np.sqrt(np.mean(observed * observed))),
            "subject_motion_rms": float(np.sqrt(np.mean(subject_motion * subject_motion))),
            "max_abs_observed_camera": float(np.max(np.abs(observed))),
            "max_abs_subject_motion": float(np.max(np.abs(subject_motion))),
            "observed_valid_fraction": float(np.count_nonzero(observed_valid) / observed_valid.size),
            "ideal_valid_fraction": float(np.count_nonzero(ideal_valid) / ideal_valid.size),
            "ideal_region_pixel_counts": {
                name: int(np.count_nonzero(ideal_region == idx)) for idx, name in enumerate(LAYER_NAMES)
            },
            "observed_region_pixel_counts": {
                name: int(np.count_nonzero(observed_region == idx)) for idx, name in enumerate(LAYER_NAMES)
            },
        },
        "pairs": pairs,
    }


def run_audit(*, seeds=DEFAULT_SEEDS, conditions=DEFAULT_CONDITIONS, n=DEFAULT_N, fps=DEFAULT_FPS,
              width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, margin=DEFAULT_MARGIN,
              layer_texture_seeds=LAYER_TEXTURE_SEEDS):
    """Run the fixed layered rendered-pixel audit and return a JSON-safe report.

    For every ``(condition, seed)`` the report lists the four fixed multiscale candidates, each paired with
    the L1-budget path whose **measured** crop proxy is nearest (``bench.multiscale_motion_audit
    .nearest_crop_index``, ties to the earlier grid index). Each pair carries both sides' E10 path metrics,
    whole-frame rendered RGB MAE/RMS/coverage against the known ideal layered render, and the same
    common-support metrics **per ideal semantic region** (background/foreground/subject) with explicit counts.
    Nothing is pooled into a cross-candidate number; no winner, rank, score or tuned selection is produced.
    ``n`` is capped at ``bench.multiscale_motion_audit.L1_WINDOW`` so the L1 solver stays on its single-window
    branch and both families keep matched offline whole-clip access. Validation errors raise ``ValueError``.
    """
    seeds = A._validate_seeds(seeds)
    conditions = _validate_conditions(conditions)
    n = A._validate_n(n)
    if n > A.L1_WINDOW:
        raise ValueError(
            f"n must be <= L1_WINDOW ({A.L1_WINDOW}) so l1_path_windowed keeps its single-window branch; "
            f"got n={n}")
    fps = A._validate_fps(fps)
    width = _validate_dim(width, "width")
    height = _validate_dim(height, "height")
    margin = _validate_margin(margin)
    canvases = make_layer_canvases(width, height, margin, seeds=layer_texture_seeds)

    runs = [
        _run_one(condition, seed, n, fps, width, height, margin, canvases)
        for condition in conditions for seed in seeds
    ]

    return {
        "schema": SCHEMA,
        "ok": True,
        "scope_note": SCOPE_NOTE,
        "scope": [
            "rendered-pixel comparison of camera paths on a three-layer parallax scene",
            "deterministic synthetic layer caricature: no media, physical depth, segmentation or occlusion model",
            "pure-NumPy bilinear translation-only rendering; not the production renderer",
            "each condition/seed/pair/region is reported separately and never pooled into one number",
        ],
        "reporting_policy": REPORTING_POLICY,
        "pixel_metrics": list(PIXEL_METRICS),
        "region_metrics": list(REGION_METRICS),
        "layer_model": dict(LAYER_MODEL),
        "ground_truth": dict(GROUND_TRUTH),
        "pairing": {
            "rule": (
                "each multiscale candidate is paired with the L1 variant whose measured crop proxy is nearest "
                "the multiscale candidate's own measured crop proxy; ties resolve to the earlier L1 grid index"
            ),
            "tolerance": float(A.L1_CROP_MATCH_TOLERANCE),
            "tolerance_definition": "half the uniform L1 budget grid step",
            "measured_by": "bench.multiscale_motion_audit.translation_path_metrics",
        },
        "temporal_access": {
            "kind": "offline_full_clip",
            "max_samples": int(A.L1_WINDOW),
            "status": "matched",
            "statement": (
                "Both families are evaluated with offline temporal access to the whole clip: n is capped at "
                "L1_WINDOW so l1_path_windowed solves one L1 problem over every sample, and the "
                "timestamp-aware Gaussian consumes the full clip too. These are offline path/render numbers, "
                "not online-latency measurements."
            ),
        },
        "grids": {
            "multiscale_scales_seconds": [float(s) for s in A.MULTISCALE_SCALES_SECONDS],
            "multiscale_candidates": [
                {"name": name, "retention_weights": [float(w) for w in weights]}
                for name, weights in A.MULTISCALE_CANDIDATES
            ],
            "l1_budgets": [float(b) for b in A.L1_BUDGETS],
            "l1_budget_min": float(A.L1_BUDGET_MIN),
            "l1_budget_max": float(A.L1_BUDGET_MAX),
            "l1_budget_step": float(A.L1_BUDGET_STEP),
            "l1_budget_count": int(A.L1_BUDGET_COUNT),
            "l1_crop_match_tolerance": float(A.L1_CROP_MATCH_TOLERANCE),
            "l1_weights": [float(w) for w in A.L1_WEIGHTS],
            "l1_fidelity": float(A.L1_FIDELITY),
            "l1_window": int(A.L1_WINDOW),
            "l1_overlap": int(A.L1_OVERLAP),
        },
        "scene": {
            "layer_names": list(LAYER_NAMES),
            "layer_texture_seeds": [int(s) for s in layer_texture_seeds],
            "background_rect": None,
            "background_scope": "full finite canvas",
            "foreground_rect": [float(v) for v in FOREGROUND_RECT],
            "subject_rect": [float(v) for v in SUBJECT_RECT],
            "subject_period_seconds": float(SUBJECT_PERIOD_SECONDS),
            "subject_amplitude": [float(v) for v in SUBJECT_AMPLITUDE],
            "subject_seed_offset": int(SUBJECT_SEED_OFFSET),
            "flat_depth_scales": [float(v) for v in FLAT_DEPTH_SCALES],
            "parallax_depth_scales": [float(v) for v in PARALLAX_DEPTH_SCALES],
            "conditions": {
                name: {
                    "camera_condition": CONDITION_SPECS[name]["camera_condition"],
                    "depth_scales": [float(v) for v in CONDITION_SPECS[name]["depth_scales"]],
                    "parallax_present": bool(CONDITION_SPECS[name]["parallax"]),
                    "subject_motion_present": bool(CONDITION_SPECS[name]["subject_motion"]),
                    "isolates": CONDITION_SPECS[name]["isolates"],
                }
                for name in CONDITIONS
            },
        },
        "fixture_defaults": {
            "n_samples": int(n),
            "fps": float(fps),
            "width_px": int(width),
            "height_px": int(height),
            "margin_px": int(margin),
            "seeds": list(seeds),
            "conditions": list(conditions),
            "conditions_supported": list(CONDITIONS),
        },
        "runs": runs,
        "limitations": list(LIMITATIONS),
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_conditions(text):
    """Parse e.g. ``"static_no_parallax,parallax_shake"`` into a validated condition tuple."""
    if isinstance(text, (list, tuple)):
        parts = list(text)
    else:
        parts = [p.strip() for p in str(text).split(",") if p.strip()]
    return _validate_conditions(parts)


def build_parser():
    p = argparse.ArgumentParser(
        prog="bench/multiscale_layered_render_audit.py",
        description="Research-only layered rendered-pixel audit of fixed multiscale-retention camera paths "
                    "against their nearest-crop L1 counterparts on a deterministic three-layer parallax "
                    "scene (background/foreground/subject). Pure NumPy; no media or production renderer.")
    p.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                   help="comma-separated integer seeds (default: %(default)s)")
    p.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS),
                   help=f"comma-separated conditions chosen from {CONDITIONS} (default: %(default)s)")
    p.add_argument("--n", type=int, default=DEFAULT_N,
                   help=f"number of uniform samples, {A.MIN_SAMPLES}..{A.L1_WINDOW} (default: %(default)s)")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS, help="uniform sample rate (default: %(default)s)")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="output width in pixels (default: %(default)s)")
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT,
                   help="output height in pixels (default: %(default)s)")
    p.add_argument("--margin", type=int, default=DEFAULT_MARGIN,
                   help="scene margin in pixels on every side (default: %(default)s)")
    p.add_argument("--out", default=None, help="write the JSON report here (default: stdout)")
    return p


def _emit(payload, out):
    text = json.dumps(json_safe(payload), indent=1, sort_keys=True, allow_nan=False)
    if out:
        write_manifest(out, payload)
    else:
        print(text)
    return text


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_audit(
            seeds=A.parse_seeds(args.seeds),
            conditions=parse_conditions(args.conditions),
            n=args.n,
            fps=args.fps,
            width=args.width,
            height=args.height,
            margin=args.margin,
        )
    except (ValueError, A.MM.MultiscaleError) as exc:
        payload = {"schema": SCHEMA, "ok": False,
                   "error": {"type": type(exc).__name__, "message": str(exc)}}
        _emit(payload, args.out)
        if args.out:
            print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.out:
        _emit(report, args.out)
        print(f"schema={report['schema']} runs={len(report['runs'])} out={Path(args.out)}")
    else:
        _emit(report, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA",
    "BACKGROUND",
    "FOREGROUND",
    "SUBJECT",
    "LAYER_NAMES",
    "LAYER_TEXTURE_SEEDS",
    "BACKGROUND_RECT",
    "FOREGROUND_RECT",
    "SUBJECT_RECT",
    "SUBJECT_PERIOD_SECONDS",
    "SUBJECT_AMPLITUDE",
    "SUBJECT_SEED_OFFSET",
    "FLAT_DEPTH_SCALES",
    "PARALLAX_DEPTH_SCALES",
    "CONDITION_SPECS",
    "CONDITIONS",
    "DEFAULT_CONDITIONS",
    "DEFAULT_N",
    "DEFAULT_FPS",
    "DEFAULT_WIDTH",
    "DEFAULT_HEIGHT",
    "DEFAULT_MARGIN",
    "DEFAULT_SEEDS",
    "SCOPE_NOTE",
    "REPORTING_POLICY",
    "PIXEL_METRICS",
    "REGION_METRICS",
    "LAYER_MODEL",
    "GROUND_TRUTH",
    "LIMITATIONS",
    "make_layer_canvases",
    "subject_motion_path",
    "max_abs_layer_displacement",
    "render_layered_scene",
    "region_pixel_metrics",
    "run_audit",
    "parse_conditions",
    "build_parser",
    "main",
]
