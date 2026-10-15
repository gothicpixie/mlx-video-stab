"""Research-only E10.1 rendered-pixel audit: fixed multiscale-retention paths vs their nearest-crop L1
counterparts on deterministic synthetic translation video.

What this measures
------------------
E10 (``bench/multiscale_motion_audit.py``) measured **translation-path** numbers only: RMS deviation from the
known pan, an RMS second-difference residual and a maximum-correction crop proxy. It never rasterized a pixel,
so its "coverage" was a geometric fraction, not a statement about which **rendered** pixels survive a warp.

This module closes that gap for the same fixed E10 candidates. It is an isolated, **pure-NumPy** research
runner that:

- builds a deterministic, procedurally textured RGB **static scene** on a finite array with a generous margin,
- renders the single-layer scene into moving frames by a **translation-only backward bilinear warp** along a
  known camera path, and
- compares the four fixed multiscale-retention candidates with the L1-budget path whose *measured* crop proxy
  is nearest (via ``bench.multiscale_motion_audit.nearest_crop_index``, same deterministic tie rule), using
  **rendered RGB MAE/RMS** against the known ideal frame path plus **geometric valid coverage**.

Camera convention (explicit, and tested)
----------------------------------------
Output pixel ``(row y, col x)`` lies on the grid ``x in [0, W-1]``, ``y in [0, H-1]``. With camera path
``C_t`` (the observed path), intended pan ``pan_t`` and candidate virtual path ``P_t``:

    observed frame   O_t(x, y) = scene(y + margin + C_t.y,   x + margin + C_t.x)
    ideal frame      I_t(x, y) = scene(y + margin + pan_t.y, x + margin + pan_t.x)
    candidate frame  V_t(x, y) = O_t(x + (P_t.x - C_t.x), y + (P_t.y - C_t.y))

so the candidate backward-samples the observed frame at ``grid + P_t - C_t``. Two signs follow directly and are
asserted in the tests: ``P = C`` reproduces the observed frame exactly, and with ``C = pan`` (so ``O = I``)
``P = pan`` gives the ideal view.

Renderer and validity
---------------------
The sampler is a locally implemented pure-NumPy **bilinear** sampler (no MLX, no torch, no media, no
production renderer), so the tests need only NumPy and bench modules. The warp is **translation-only**:
rotation, depth, subject motion, occlusion and perspective are all out of scope. A source coordinate is
**valid** only when ``0 <= x <= W-1`` and ``0 <= y <= H-1`` (every bilinear tap in bounds); invalid destination
pixels are written as zero but are **excluded from MAE/RMS**, so uncovered pixels never penalize a candidate.

Metrics
-------
For each of the four multiscale candidates and its paired L1 counterpart, reported **separately** (no pooled
score, ranking, winner or tuning):

- ``rgb_mae`` / ``rgb_rms``: RGB-channel-mean absolute / root-mean-square error of the candidate against the
  ideal frame path, computed on the intersection of candidate-valid and ideal-valid pixels. Units are
  normalized scene intensity in ``[0, 1]``.
- ``candidate_valid_fraction_of_total`` and ``ideal_valid_fraction_of_total``: valid coverage over **all**
  output pixels.
- ``common_valid_fraction_of_total`` and ``common_valid_fraction_of_ideal_valid``: the **common-support**
  coverage (intersection) over the total and over the ideal-valid support.
- the E10 path metrics (RMS path deviation, residual second-difference RMS, maximum correction) for context.

What this is not
----------------
It is a **single-layer, static-scene, translation-only, fixed-procedural-texture, bilinear-sampling** pixel
exercise. It is **not** perceptual quality, not depth/rotation/subject motion/occlusion, not real footage, not
the production renderer, not a quality score and not a superiority claim. It changes no production default and
imports no production module.

Run ``python -m bench.multiscale_render_audit [--seeds 0,1,2]
[--conditions static,pan_no_shake,pan_shake,pan_stochastic_shake,pan_jolt] [--n 60] [--width 80] [--height 48]
[--margin 16] [--fps 60] [--out f.json]``. Reports are JSON-safe (finite-only) and path-free.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import multiscale_motion_audit as A
from .manifest import json_safe, write_manifest

SCHEMA = "mlxstab.multiscale_render_audit/1"

# --------------------------------------------------------------------------------------
# fixed experiment constants (the experiment, not tunables)
# --------------------------------------------------------------------------------------

# Fixed procedural-texture seed. The scene is identical across conditions and run seeds so that only the
# camera path changes; it carries no media, weights or RNG-dependent run state.
SCENE_SEED = 1729

DEFAULT_N = 60
DEFAULT_FPS = 60.0
DEFAULT_WIDTH = 80
DEFAULT_HEIGHT = 48
DEFAULT_MARGIN = 16
DEFAULT_SEEDS = A.DEFAULT_SEEDS
DEFAULT_CONDITIONS = A.DEFAULT_CONDITIONS

MIN_DIM = 1
MIN_MARGIN = 0

SCOPE_NOTE = (
    "Research-only rendered-pixel audit of a single-layer, translation-only, fixed-procedural-texture static "
    "scene sampled with a pure-NumPy bilinear warp. It compares four fixed multiscale-retention candidates "
    "with their nearest measured-crop L1 counterparts on rendered RGB error against the known ideal frame "
    "path, reporting each pair separately. It is not perceptual quality, not depth, rotation, subject motion "
    "or occlusion, not real footage or the production renderer, and it makes no superiority claim."
)

REPORTING_POLICY = (
    "Every condition/seed/pair metric is reported separately. No aggregate winner, ranking, total score or "
    "weight optimization is computed, no per-seed tuning is performed and no candidate or family is asserted "
    "superior."
)

PIXEL_METRICS = [
    "rgb_mae: RGB-channel-mean absolute error of the candidate against the ideal frame path on the common "
    "candidate-valid and ideal-valid support; normalized intensity [0, 1]",
    "rgb_rms: RGB-channel-mean root-mean-square error on the same common support; normalized intensity [0, 1]",
    "candidate_valid_fraction_of_total: fraction of all output pixels where the candidate warp is valid",
    "ideal_valid_fraction_of_total: fraction of all output pixels where the ideal warp is valid",
    "common_valid_fraction_of_total: fraction of all output pixels valid for both candidate and ideal",
    "common_valid_fraction_of_ideal_valid: fraction of the ideal-valid support that the candidate also covers",
]

CAMERA_MODEL = {
    "output_grid": "row y in [0, H-1], column x in [0, W-1]; a pixel's source coordinate is grid + margin + path",
    "observed": "O_t(x, y) = scene(y + margin + C_t.y, x + margin + C_t.x)",
    "ideal": "I_t(x, y) = scene(y + margin + pan_t.y, x + margin + pan_t.x)",
    "candidate": "V_t(x, y) = O_t(x + (P_t.x - C_t.x), y + (P_t.y - C_t.y))",
    "path_order": "channels are (x, y): index 0 is horizontal, index 1 is vertical",
    "identity_check": "P = C reproduces the observed frame exactly",
    "ideal_check": "with C = pan (so observed = ideal), P = pan gives the ideal view",
}

RENDERER = {
    "sampler": "pure NumPy bilinear interpolation (translation-only warp)",
    "input": "finite static single-layer RGB scene with a margin on every side",
    "border_policy": "invalid destinations are written as zero",
    "validity_rule": "a source coordinate (x, y) is valid only if 0 <= x <= W-1 and 0 <= y <= H-1 (all "
                     "bilinear taps in bounds)",
    "not_included": "no rotation, depth, subject motion, occlusion, perspective, media decode or production "
                    "renderer",
}

LIMITATIONS = (
    "single-layer static scene only: no depth ordering, occlusion or parallax",
    "translation-only warps: no rotation, scale, perspective, rolling shutter or subject motion",
    "fixed procedural texture sampled by a pure-NumPy bilinear interpolator: not the production renderer",
    "rendered RGB MAE/RMS on a synthetic cartoon is not perceptual quality and not a human preference study",
    "coverage is a geometric source-frame validity fraction, not a real stabilizer crop measurement",
    "four fixed multiscale weights and the predeclared L1 budget grid are reported verbatim; nothing is tuned",
    "nearest-crop pairing is descriptive: it produces no winner, rank, score, aggregate or tuned selection",
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


def _validate_scene_seed(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"scene_seed must be an integer, got {value!r}")
    return int(value)


# --------------------------------------------------------------------------------------
# deterministic procedural scene and pure-NumPy bilinear warp
# --------------------------------------------------------------------------------------

def make_scene(height, width, *, seed=SCENE_SEED):
    """Deterministic procedurally textured RGB scene of shape ``[height, width, 3]`` in ``[0, 1]``.

    Every channel is a fixed sum of a handful of 2-D sinusoids whose frequencies, phases and amplitudes are
    drawn from ``numpy.random.default_rng(seed)``, then min-max normalized to ``[0, 1]``. There is no media,
    no asset, no weight and no run-seed dependence: with the same ``(height, width, seed)`` the scene is
    bit-identical. The texture is smooth, so the translation warp has no resampling holes inside the valid
    region.
    """
    height = _validate_dim(height, "height")
    width = _validate_dim(width, "width")
    seed = _validate_scene_seed(seed)

    yy, xx = np.meshgrid(np.arange(height, dtype=np.float64), np.arange(width, dtype=np.float64), indexing="ij")
    rng = np.random.default_rng(seed)
    scene = np.zeros((height, width, 3), dtype=np.float64)
    two_pi = 2.0 * np.pi
    for c in range(3):
        acc = np.zeros((height, width), dtype=np.float64)
        for _ in range(5):
            fx = float(rng.uniform(1.0, 5.0)) / max(width, 1)
            fy = float(rng.uniform(1.0, 5.0)) / max(height, 1)
            phase = float(rng.uniform(0.0, two_pi))
            amp = float(rng.uniform(0.4, 1.0))
            acc += amp * np.sin(two_pi * (fx * xx + fy * yy) + phase)
        lo = float(acc.min())
        hi = float(acc.max())
        span = hi - lo
        scene[..., c] = (acc - lo) / span if span > 0.0 else 0.0
    return scene


def bilinear_sample(images, coords):
    """Pure-NumPy bilinear sample of ``images [T, Hs, Ws, C]`` at float ``coords [T, Ho, Wo, 2]`` (x, y order).

    Only the batch axis ``T`` is shared: the coordinate spatial output axes ``(Ho, Wo)`` are independent of the
    source ``(Hs, Ws)``, so a larger finite scene can be sampled into a smaller (or larger) output grid.
    Returns ``(sampled [T, Ho, Wo, C], valid [T, Ho, Wo])``. A coordinate is **valid** only when
    ``0 <= x <= Ws-1`` and ``0 <= y <= Hs-1``, i.e. every contributing bilinear tap is in bounds; invalid
    destinations sample as zero and are flagged. Every gather index is clamped into the source image bounds so
    an out-of-range coordinate can never raise; clamping only guards the gather and never turns an
    out-of-bounds coordinate valid.
    """
    images = np.asarray(images, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    if images.ndim != 4:
        raise ValueError(f"images must be [T, H, W, C], got shape {images.shape}")
    if coords.ndim != 4 or coords.shape[0] != images.shape[0] or coords.shape[-1] != 2:
        raise ValueError(
            f"coords must be [T, Ho, Wo, 2] sharing the images batch axis T={images.shape[0]}, got shape "
            f"{coords.shape}")
    if not bool(np.isfinite(images).all()) or not bool(np.isfinite(coords).all()):
        raise ValueError("images and coords must be finite")

    t_count, height, width, channels = images.shape
    x = coords[..., 0]
    y = coords[..., 1]
    valid = (x >= 0.0) & (x <= width - 1) & (y >= 0.0) & (y <= height - 1)

    x0f = np.floor(x)
    y0f = np.floor(y)
    x0 = np.clip(x0f, 0.0, width - 1).astype(np.int64)
    x1 = np.clip(x0f + 1.0, 0.0, width - 1).astype(np.int64)
    y0 = np.clip(y0f, 0.0, height - 1).astype(np.int64)
    y1 = np.clip(y0f + 1.0, 0.0, height - 1).astype(np.int64)

    wx1 = x - x0f
    wx0 = 1.0 - wx1
    wy1 = y - y0f
    wy0 = 1.0 - wy1

    t = np.arange(t_count)[:, None, None]

    def gather(yi, xi):
        return images[t, yi, xi]

    out = (
        (wx0 * wy0)[..., None] * gather(y0, x0)
        + (wx1 * wy0)[..., None] * gather(y0, x1)
        + (wx0 * wy1)[..., None] * gather(y1, x0)
        + (wx1 * wy1)[..., None] * gather(y1, x1)
    )
    out = np.where(valid[..., None], out, 0.0)
    return out, valid


def sample_scene(scene, path, *, width, height, margin):
    """Render frames by sampling ``scene`` at ``grid + margin + path_t``.

    ``scene [Hs, Ws, C]`` with ``Hs >= height + 2*margin`` and ``Ws >= width + 2*margin``; ``path [T, 2]``
    is the camera path in ``(x, y)`` order. Returns ``(frames [T, height, width, C], valid [T, height, width])``.
    """
    scene = np.asarray(scene, dtype=np.float64)
    width = _validate_dim(width, "width")
    height = _validate_dim(height, "height")
    margin = _validate_margin(margin)
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError(f"path must be [T, 2] (x, y), got shape {path.shape}")
    if not bool(np.isfinite(path).all()):
        raise ValueError("path must be finite")
    if scene.ndim != 3:
        raise ValueError(f"scene must be [H, W, C], got shape {scene.shape}")
    if scene.shape[0] < height + 2 * margin or scene.shape[1] < width + 2 * margin:
        raise ValueError(
            f"scene {scene.shape[:2]} is too small for {height}x{width} with margin {margin} "
            f"(needs at least {height + 2 * margin}x{width + 2 * margin})")

    t_count = path.shape[0]
    base = np.stack(np.meshgrid(np.arange(width, dtype=np.float64),
                                np.arange(height, dtype=np.float64), indexing="xy"), axis=-1)  # [H, W, 2] (x, y)
    coords = base[None, :, :, :] + float(margin) + path[:, None, None, :]
    images = np.broadcast_to(scene, (t_count,) + scene.shape)
    return bilinear_sample(images, coords)


def backward_warp(observed_frames, path_delta):
    """Backward-sample observed frames at ``grid + path_delta_t`` (the candidate virtual view).

    ``observed_frames [T, H, W, C]`` and ``path_delta [T, 2]`` (x, y). For a camera path ``C`` and candidate
    path ``P`` the correct argument is ``P - C``; ``P = C`` therefore returns the observed frames unchanged.
    Returns ``(frames, valid)`` like :func:`bilinear_sample`.
    """
    observed_frames = np.asarray(observed_frames, dtype=np.float64)
    if observed_frames.ndim != 4:
        raise ValueError(f"observed_frames must be [T, H, W, C], got shape {observed_frames.shape}")
    delta = np.asarray(path_delta, dtype=np.float64)
    if delta.ndim != 2 or delta.shape[1] != 2 or delta.shape[0] != observed_frames.shape[0]:
        raise ValueError(
            f"path_delta must be [T, 2] with T={observed_frames.shape[0]}, got shape {delta.shape}")
    if not bool(np.isfinite(delta).all()):
        raise ValueError("path_delta must be finite")

    t_count, height, width, _ = observed_frames.shape
    base = np.stack(np.meshgrid(np.arange(width, dtype=np.float64),
                                np.arange(height, dtype=np.float64), indexing="xy"), axis=-1)
    coords = base[None, :, :, :] + delta[:, None, None, :]
    return bilinear_sample(observed_frames, coords)


# --------------------------------------------------------------------------------------
# pixel metrics
# --------------------------------------------------------------------------------------

def rendered_pixel_metrics(candidate_frames, candidate_valid, ideal_frames, ideal_valid):
    """Rendered RGB MAE/RMS against the ideal path plus valid-coverage fractions.

    MAE and RMS are **RGB-channel means** over the intersection of ``candidate_valid`` and ``ideal_valid``
    (uncovered/zero pixels are excluded, never penalized); ``rgb_mae``/``rgb_rms`` are ``None`` when the
    common support is empty. Coverage fractions are computed over all output pixels, and the common-support
    coverage is reported both over the total and over the ideal-valid support. Units are normalized scene
    intensity in ``[0, 1]``. All returned values are plain Python.
    """
    candidate_frames = np.asarray(candidate_frames, dtype=np.float64)
    ideal_frames = np.asarray(ideal_frames, dtype=np.float64)
    candidate_valid = np.asarray(candidate_valid, dtype=bool)
    ideal_valid = np.asarray(ideal_valid, dtype=bool)
    if candidate_frames.shape != ideal_frames.shape or candidate_frames.ndim != 4:
        raise ValueError(
            f"candidate_frames and ideal_frames must share one [T, H, W, C] shape, got "
            f"{candidate_frames.shape} and {ideal_frames.shape}")
    if candidate_valid.shape != candidate_frames.shape[:3] or ideal_valid.shape != candidate_frames.shape[:3]:
        raise ValueError("valid masks must match the frames' [T, H, W] shape")

    total = int(candidate_valid.size)
    common = candidate_valid & ideal_valid
    n_common = int(common.sum())
    n_ideal_valid = int(ideal_valid.sum())
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
        "rgb_mae": rgb_mae,
        "rgb_rms": rgb_rms,
        "common_valid_pixels": n_common,
        "candidate_valid_fraction_of_total": float(np.count_nonzero(candidate_valid) / total),
        "ideal_valid_fraction_of_total": float(np.count_nonzero(ideal_valid) / total),
        "common_valid_fraction_of_total": float(n_common / total),
        "common_valid_fraction_of_ideal_valid": (
            float(n_common / n_ideal_valid) if n_ideal_valid > 0 else 0.0),
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
    return backward_warp(observed_frames, delta)


def _run_one(condition, seed, n, fps, width, height, margin, scene_seed, scene):
    fx = A.synthetic_translation_fixture(n=n, fps=fps, seed=seed, condition=condition)
    times, pan, shake, observed = fx["times"], fx["pan"], fx["shake"], fx["observed"]

    observed_frames, observed_valid = sample_scene(
        scene, observed, width=width, height=height, margin=margin)
    ideal_frames, ideal_valid = sample_scene(scene, pan, width=width, height=height, margin=margin)
    if not bool(observed_valid.all()) or not bool(ideal_valid.all()):
        max_motion = float(max(np.max(np.abs(observed)), np.max(np.abs(pan))))
        raise ValueError(
            f"margin {margin} is too small for the {condition!r} motion (max |path| = {max_motion:.3f}); "
            f"use margin >= {int(np.ceil(max_motion)) + 1} so the scene sampling has no holes")

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
                "pixel": rendered_pixel_metrics(ms_frames, ms_valid, ideal_frames, ideal_valid),
            },
            "l1": {
                "budget": float(A.L1_BUDGETS[idx]),
                "path": _path_report(pan, observed, l1_outputs[idx]),
                "pixel": rendered_pixel_metrics(l1_frames, l1_valid, ideal_frames, ideal_valid),
            },
        })

    return {
        "condition": condition,
        "seed": seed,
        "fixture": {
            "n_samples": int(n),
            "fps": float(fps),
            "dt_seconds": float(1.0 / fps),
            "duration_seconds": float(times[-1] - times[0]),
            "timestamp_basis": "uniform",
            "condition_model": A.CONDITION_MODELS[condition],
            "condition_kind": A.CONDITION_KIND[condition],
            "width_px": int(width),
            "height_px": int(height),
            "margin_px": int(margin),
            "scene_seed": int(scene_seed),
            "scene_shape": [int(scene.shape[0]), int(scene.shape[1]), int(scene.shape[2])],
            "pan_rms": float(np.sqrt(np.mean(pan * pan))),
            "shake_rms": float(np.sqrt(np.mean(shake * shake))),
            "observed_rms": float(np.sqrt(np.mean(observed * observed))),
            "max_abs_pan": float(np.max(np.abs(pan))),
            "max_abs_observed": float(np.max(np.abs(observed))),
        },
        "pairs": pairs,
    }


def run_audit(*, seeds=DEFAULT_SEEDS, conditions=DEFAULT_CONDITIONS, n=DEFAULT_N, fps=DEFAULT_FPS,
              width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, margin=DEFAULT_MARGIN, scene_seed=SCENE_SEED):
    """Run the fixed rendered-pixel audit and return a JSON-safe report.

    For every ``(condition, seed)`` the report lists the four fixed multiscale candidates, each paired with
    the L1-budget path whose **measured** crop proxy is nearest (``bench.multiscale_motion_audit
    .nearest_crop_index``, ties to the earlier grid index). Each pair carries both sides' E10 path metrics and
    rendered RGB MAE/RMS/coverage against the known ideal frame path, computed on the common valid support.
    Nothing is pooled into a cross-candidate number; no winner, rank, score or tuned selection is produced.
    Only the four multiscale paths and their (cached) four paired L1 paths are rendered per run. ``n`` is
    capped at ``bench.multiscale_motion_audit.L1_WINDOW`` so the L1 solver stays on its single-window branch.
    Validation errors raise ``ValueError``.
    """
    seeds = A._validate_seeds(seeds)
    conditions = A._validate_conditions(conditions)
    n = A._validate_n(n)
    if n > A.L1_WINDOW:
        raise ValueError(
            f"n must be <= L1_WINDOW ({A.L1_WINDOW}) so l1_path_windowed keeps its single-window branch; "
            f"got n={n}")
    fps = A._validate_fps(fps)
    width = _validate_dim(width, "width")
    height = _validate_dim(height, "height")
    margin = _validate_margin(margin)
    scene_seed = _validate_scene_seed(scene_seed)

    scene = make_scene(height + 2 * margin, width + 2 * margin, seed=scene_seed)
    runs = [
        _run_one(condition, seed, n, fps, width, height, margin, scene_seed, scene)
        for condition in conditions for seed in seeds
    ]

    return {
        "schema": SCHEMA,
        "ok": True,
        "scope_note": SCOPE_NOTE,
        "scope": [
            "rendered-pixel comparison of translation paths only",
            "single-layer static deterministic procedural scene; no media, depth, occlusion or subject motion",
            "pure-NumPy bilinear translation-only rendering; not the production renderer",
            "each condition/seed/pair is reported separately and never pooled into one number",
        ],
        "reporting_policy": REPORTING_POLICY,
        "pixel_metrics": PIXEL_METRICS,
        "camera_model": dict(CAMERA_MODEL),
        "renderer": dict(RENDERER),
        "pairing": {
            "rule": (
                "each multiscale candidate is paired with the L1 variant whose measured crop proxy is nearest "
                "the multiscale candidate's own measured crop proxy; ties resolve to the earlier L1 grid index"
            ),
            "tolerance": float(A.L1_CROP_MATCH_TOLERANCE),
            "tolerance_definition": "half the uniform L1 budget grid step",
            "measured_by": "bench.multiscale_motion_audit.translation_path_metrics",
        },
        "grids": {
            "multiscale_scales_seconds": [float(s) for s in A.MULTISCALE_SCALES_SECONDS],
            "multiscale_candidates": [
                {"name": name, "retention_weights": [float(w) for w in weights]}
                for name, weights in A.MULTISCALE_CANDIDATES
            ],
            "l1_budgets": [float(b) for b in A.L1_BUDGETS],
            "l1_budget_count": int(A.L1_BUDGET_COUNT),
            "l1_crop_match_tolerance": float(A.L1_CROP_MATCH_TOLERANCE),
            "l1_weights": [float(w) for w in A.L1_WEIGHTS],
            "l1_fidelity": float(A.L1_FIDELITY),
            "l1_window": int(A.L1_WINDOW),
            "l1_overlap": int(A.L1_OVERLAP),
        },
        "fixture_defaults": {
            "n_samples": int(n),
            "fps": float(fps),
            "width_px": int(width),
            "height_px": int(height),
            "margin_px": int(margin),
            "scene_seed": int(scene_seed),
            "seeds": list(seeds),
            "conditions": list(conditions),
            "conditions_supported": list(A.CONDITIONS),
        },
        "runs": runs,
        "limitations": list(LIMITATIONS),
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="bench/multiscale_render_audit.py",
        description="Research-only rendered-pixel audit of fixed multiscale-retention paths against their "
                    "nearest-crop L1 counterparts on deterministic synthetic translation video. Single-layer "
                    "static scene, translation-only pure-NumPy bilinear warp; no media or production renderer.")
    p.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                   help="comma-separated integer seeds (default: %(default)s)")
    p.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS),
                   help=f"comma-separated conditions chosen from {A.CONDITIONS} (default: %(default)s)")
    p.add_argument("--n", type=int, default=DEFAULT_N,
                   help=f"number of uniform samples, {A.MIN_SAMPLES}..{A.L1_WINDOW} (default: %(default)s)")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS, help="uniform sample rate (default: %(default)s)")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="output width in pixels (default: %(default)s)")
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="output height in pixels (default: %(default)s)")
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
            conditions=A.parse_conditions(args.conditions),
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
    "SCENE_SEED",
    "DEFAULT_N",
    "DEFAULT_FPS",
    "DEFAULT_WIDTH",
    "DEFAULT_HEIGHT",
    "DEFAULT_MARGIN",
    "DEFAULT_SEEDS",
    "DEFAULT_CONDITIONS",
    "SCOPE_NOTE",
    "PIXEL_METRICS",
    "CAMERA_MODEL",
    "RENDERER",
    "LIMITATIONS",
    "make_scene",
    "bilinear_sample",
    "sample_scene",
    "backward_warp",
    "rendered_pixel_metrics",
    "run_audit",
    "build_parser",
    "main",
]
