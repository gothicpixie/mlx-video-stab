"""Research-only E10 comparison runner: fixed multiscale-retention candidates vs L1 budgets on synthetic
camera-translation paths.

What this measures
------------------
This module compares two *path-level* families on **deterministic known-truth synthetic camera translation**:

- fixed multiscale-retention candidates built from the committed research decomposition
  ``bench.multiscale_motion`` (``decompose`` + ``path_reconstruct`` with a fixed grid of detail-band
  retention weights), and
- L1-budget paths from the production smoothing primitive ``mlxstab.smooth.l1_path_windowed`` on a
  **predeclared uniform** budget grid ``0.0, 0.1, ..., 2.0`` (budget ``0.0`` is the observed-path identity).

Each multiscale candidate is additionally reported **paired** with the L1 variant whose *measured* crop proxy
(``max_abs_correction_from_observed``) is nearest its own measured crop proxy. Ties resolve to the earlier L1
grid index. The pairing is a descriptive matched-crop report: it is not a ranking, winner, score, aggregate or
tuned selection, and it never picks a candidate for downstream use.

Temporal access is **matched and offline** for both families: the audit requires ``n <= L1_WINDOW`` (600
samples), so ``l1_path_windowed`` takes its ``T <= window`` branch and solves **one** L1 problem over every
sample of the clip, and the timestamp-aware Gaussian low-pass also consumes the full clip. Both therefore see
the whole clip, so neither has a causal/streaming advantage; these are offline path-level numbers, **not**
online-latency measurements. The cap exists because a longer clip would push L1 into its overlapping-window
branch and the two families would no longer see the same entire clip.

The synthetic fixture is a deliberate slow pan (the known intended ground truth) plus one optional
perturbation. Timestamps are **uniform** on purpose: ``l1_path_windowed`` is frame-indexed, so uniform
spacing makes the frame clock and the elapsed-time clock agree and the fixed multiscale scales (seconds) and
L1 budgets (path units) are compared on one clock. Beyond the three original controls (``static``,
``pan_no_shake``, ``pan_shake``) the default report adds two deliberately **non-fractal** stress controls:
``pan_stochastic_shake`` (deterministic-seed stationary colored AR(1) shake with fixed per-channel
amplitudes and coefficients; not 1/f noise) and ``pan_jolt`` (a fixed-time one-sided transient displacement
with exponential recovery, placed at a fixed fraction of the clip duration). Both are stress controls only:
they exercise the fixed grids on non-periodic motion and make no superiority claim.

For every seed and every variant the report gives, **separately**:

- RMS path deviation from the known intended pan,
- RMS second difference of the residual (output minus intended pan) as a path-level jitter proxy, and
- maximum absolute correction from the observed path as a translation crop proxy,

plus the variant parameters and scope notes. Nothing is pooled into a cross-variant number.

What this is not
----------------
It measures a **1D/2D translation path only**. It does **not** measure rendered image quality, rotation,
depth, subject motion, border support or human preference. It does not decode media, render frames, import the
stabilizer pipeline or change any production default. It does **not** compute an aggregate winner or score,
does **not** optimize weights, does **not** tune per seed and does **not** assert that any family or candidate
is superior. Reports are JSON-serializable and path-free.

Run ``python -m bench.multiscale_motion_audit [--seeds 0,1,2]
[--conditions static,pan_no_shake,pan_shake,pan_stochastic_shake,pan_jolt] [--out f.json]`` (``n`` is capped
at ``L1_WINDOW``).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from mlxstab.smooth import l1_path_windowed

from . import multiscale_motion as MM
from .manifest import json_safe, write_manifest

SCHEMA = "mlxstab.multiscale_motion_audit/1"

# --------------------------------------------------------------------------------------
# fixed candidate grids (these are the experiment, not tunables)
# --------------------------------------------------------------------------------------

# Decomposition scales in seconds. Fine-to-coarse detail bands separate fast shake (< ~0.05 s) from mid shake
# (~0.25 s) from the slow deliberate pan; the coarsest trend carries the pan.
MULTISCALE_SCALES_SECONDS = (0.02, 0.12, 0.6)

# One retention weight per detail band (in scale order, finest -> coarsest). The trend is always retained.
# These are descriptive smoothing levels, deliberately ordered and never ranked.
MULTISCALE_CANDIDATES = (
    ("drop_all_detail", (0.0, 0.0, 0.0)),
    ("drop_finest", (0.0, 1.0, 1.0)),
    ("drop_finest_and_medium", (0.0, 0.0, 1.0)),
    ("keep_all_detail", (1.0, 1.0, 1.0)),
)

# L1 budget grid in path units (the box half-width |p - observed| <= budget) and the L1 objective weights.
# Predeclared uniform grid: 0.0, 0.1, ..., 2.0 (21 points). Budget 0.0 pins the solution to the observed
# path (identity); it is a fixed grid point, never a tuned or optimized value.
L1_BUDGET_MIN = 0.0
L1_BUDGET_MAX = 2.0
L1_BUDGET_STEP = 0.1
L1_BUDGET_COUNT = 21
L1_BUDGETS = tuple(round(L1_BUDGET_MIN + i * L1_BUDGET_STEP, 1) for i in range(L1_BUDGET_COUNT))
# Half a uniform grid step: the largest nearest-grid-point crop-proxy mismatch the grid can guarantee when the
# L1 measured crop proxy tracks its budget. Reported verbatim; it is descriptive, not a quality threshold.
L1_CROP_MATCH_TOLERANCE = L1_BUDGET_STEP / 2.0
L1_WEIGHTS = (10.0, 1.0, 100.0)
L1_FIDELITY = 0.1
L1_WINDOW = 600
L1_OVERLAP = 60

# Synthetic fixture constants. Amplitudes are per channel (x, y) in path units; periods are in seconds.
PAN_AMPLITUDE = (2.0, 1.4)
PAN_PERIOD_SECONDS = 12.0
PAN_PHASE_RADIANS = (0.0, 0.6)
MEDIUM_SHAKE_PERIOD_SECONDS = 0.25
MEDIUM_SHAKE_AMPLITUDE = (0.9, 0.7)
FAST_SHAKE_PERIOD_SECONDS = 0.05
FAST_SHAKE_AMPLITUDE = (0.4, 0.3)

# Stochastic stress control: deterministic-seed **stationary colored AR(1)** shake, one independent process
# per channel d. This is a fixed AR(1) process, explicitly **not fractal** and **not 1/f noise**:
#     x_t[d] = rho[d] * x_{t-1}[d] + sigma[d] * eps_t[d],  eps_t ~ N(0, 1),
#     sigma[d] = amplitude[d] * sqrt(1 - rho[d]^2)   (stationary per-channel std is amplitude[d]),
#     x_0[d] drawn from the stationary distribution N(0, amplitude[d]^2).
STOCHASTIC_AR1_AMPLITUDE = (0.6, 0.5)
STOCHASTIC_AR1_COEFFICIENT = (0.85, 0.72)

# Jolt stress control: a single fixed-time one-sided transient displacement with exponential recovery, one
# process per channel d. Explicitly non-periodic and non-fractal:
#     jolt_d(t) = 0                                                       for t < t_jolt
#     jolt_d(t) = JOLT_AMPLITUDE[d] * exp(-(t - t_jolt)/JOLT_DECAY_SECONDS)  for t >= t_jolt
#     t_jolt = JOLT_TIME_FRACTION * clip_duration
# The fixed fraction keeps the jolt inside any valid clip (MIN_SAMPLES <= n <= L1_WINDOW).
JOLT_AMPLITUDE = (1.6, 1.1)
JOLT_DECAY_SECONDS = 0.4
JOLT_TIME_FRACTION = 0.5

DEFAULT_N = 180
DEFAULT_FPS = 60.0
DEFAULT_SEEDS = (0, 1, 2)
# All five synthetic controls are reported by default.
DEFAULT_CONDITIONS = ("static", "pan_no_shake", "pan_shake", "pan_stochastic_shake", "pan_jolt")
CONDITIONS = ("static", "pan_no_shake", "pan_shake", "pan_stochastic_shake", "pan_jolt")
MIN_SAMPLES = 5

# Exact generation model per condition, recorded in the report so a reader does not have to re-derive it.
CONDITION_MODELS = {
    "static": "observed = 0 (zero pan and zero perturbation)",
    "pan_no_shake": (
        "observed = pan; pan[d](t) = PAN_AMPLITUDE[d] * sin(2*pi*t/PAN_PERIOD_SECONDS + "
        "PAN_PHASE_RADIANS[d])"
    ),
    "pan_shake": (
        "observed = pan + (medium + fast); medium[d] = MEDIUM_SHAKE_AMPLITUDE[d]*sin(2*pi*t/"
        "MEDIUM_SHAKE_PERIOD_SECONDS + medium_phase[d]); fast[d] = FAST_SHAKE_AMPLITUDE[d]*sin(2*pi*t/"
        "FAST_SHAKE_PERIOD_SECONDS + fast_phase[d]); medium_phase and fast_phase are deterministic-seed "
        "uniform draws"
    ),
    "pan_stochastic_shake": (
        "observed = pan + x; x_t[d] = STOCHASTIC_AR1_COEFFICIENT[d]*x_{t-1}[d] + sigma[d]*eps_t[d]; "
        "sigma[d] = STOCHASTIC_AR1_AMPLITUDE[d]*sqrt(1-STOCHASTIC_AR1_COEFFICIENT[d]^2); eps_t ~ N(0,1); "
        "x_0[d] ~ N(0, STOCHASTIC_AR1_AMPLITUDE[d]^2); deterministic per seed, stationary and colored, and "
        "explicitly NOT fractal and NOT 1/f noise"
    ),
    "pan_jolt": (
        "observed = pan + jolt; jolt_d(t) = 0 for t < t_jolt and JOLT_AMPLITUDE[d]*"
        "exp(-(t-t_jolt)/JOLT_DECAY_SECONDS) for t >= t_jolt; t_jolt = JOLT_TIME_FRACTION * clip_duration; "
        "a one-sided transient with exponential recovery: non-periodic and non-fractal"
    ),
}
CONDITION_KIND = {
    "static": "zero",
    "pan_no_shake": "sinusoidal_pan",
    "pan_shake": "sinusoidal_pan_plus_periodic_shake",
    "pan_stochastic_shake": "sinusoidal_pan_plus_stochastic_ar1_shake_non_fractal",
    "pan_jolt": "sinusoidal_pan_plus_transient_jolt_non_periodic",
}

SCOPE_NOTE = (
    "Measures a 1D/2D translation path only: it compares fixed multiscale-retention candidates against L1 "
    "budgets on deterministic synthetic camera-translation paths with a known intended pan, and reports RMS "
    "path deviation, an RMS second-difference residual jitter proxy and a maximum-correction crop proxy for "
    "each seed and variant separately. It does not measure rendered image quality, rotation, depth, subject "
    "motion, border support or human preference, and it makes no superiority claim."
)

VARIANT_SCOPE = (
    "translation-path numbers only; not rendered image quality, rotation, depth, subject motion, border "
    "support or human preference"
)

REPORTING_POLICY = (
    "Every seed/condition/variant metric is reported separately. No aggregate winner, ranking, total score or "
    "weight optimization is computed, no per-seed tuning is performed and no candidate or family is asserted "
    "superior."
)

LIMITATIONS = (
    "synthetic known-truth 1D/2D translation paths only; not real footage",
    "uniform timestamps are required so the frame-indexed L1 clock matches the elapsed-time multiscale clock",
    "a second difference of the residual is a path-level jitter proxy, not a perceptual jitter measurement",
    "maximum absolute correction is a translation crop proxy, not a rendered crop or border-support measurement",
    "the multiscale retention weights and L1 budgets are fixed grids reported verbatim; neither is optimized",
    "multiscale and L1 are compared with matched offline temporal access to the full clip: n is capped at "
    "L1_WINDOW so the L1 solver takes its single-window branch over every sample; nothing here measures "
    "online latency or causal-streaming behavior",
    "matched-crop pairing is descriptive: it produces no winner, rank, score, aggregate or tuned selection",
    "no rendered image quality, rotation, depth, subject motion, border support or human preference is measured",
    "the two added stress controls are non-fractal: pan_stochastic_shake is a stationary colored AR(1) "
    "process (not 1/f noise, not a Hurst/fractal estimate) and pan_jolt is a one-sided exponential-recovery "
    "transient; neither is periodic",
    "pan_stochastic_shake and pan_jolt are stress controls only: they run the same fixed grids on "
    "non-periodic perturbations and make no method-superiority claim",
)


# --------------------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------------------

def _validate_n(n):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError(f"n must be an integer, got {n!r}")
    n = int(n)
    if n < MIN_SAMPLES:
        raise ValueError(f"n must be >= {MIN_SAMPLES}, got {n}")
    return n


def _validate_fps(fps):
    try:
        f = float(fps)
    except (TypeError, ValueError):
        raise ValueError(f"fps must be a finite positive number, got {fps!r}")
    if not math.isfinite(f) or f <= 0.0:
        raise ValueError(f"fps must be finite and > 0, got {fps!r}")
    return f


def _validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    return int(seed)


def _validate_condition(condition):
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    return str(condition)


def _as_sequence(value, name):
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence, not a string")
    try:
        return list(value)
    except TypeError:
        raise ValueError(f"{name} must be a sequence, got {value!r}")


def _validate_seeds(seeds):
    seq = _as_sequence(seeds, "seeds")
    if not seq:
        raise ValueError("seeds must be a non-empty sequence")
    out = []
    for s in seq:
        v = _validate_seed(s)
        if v in out:
            raise ValueError(f"duplicate seed {v}")
        out.append(v)
    return tuple(out)


def _validate_conditions(conditions):
    seq = _as_sequence(conditions, "conditions")
    if not seq:
        raise ValueError("conditions must be a non-empty sequence")
    out = []
    for c in seq:
        v = _validate_condition(c)
        if v in out:
            raise ValueError(f"duplicate condition {v!r}")
        out.append(v)
    return tuple(out)


def _as_path(name, arr, *, min_t=3):
    try:
        a = np.asarray(arr, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a numeric array: {exc}")
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2-D [T, D] array, got shape {a.shape}")
    if a.shape[0] < min_t or a.shape[1] < 1:
        raise ValueError(f"{name} must have T >= {min_t} and D >= 1, got shape {a.shape}")
    if not bool(np.isfinite(a).all()):
        raise ValueError(f"{name} contains non-finite values")
    return a


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------

def translation_path_metrics(intended, observed, output):
    """The three path metrics, computed independently for ``output`` against the known paths.

    ``intended`` is the known ground-truth pan, ``observed`` the (pan + shake) input path and ``output`` a
    candidate path; all are ``[T, D]`` with the same shape, ``T >= 3``, ``D >= 1`` and finite.

    Returns plain-Python floats only:

    - ``rms_path_deviation``: RMS of ``output - intended`` (tracking error against ground truth),
    - ``rms_second_difference_residual``: RMS of the second difference of ``output - intended`` (a path-level
      jitter proxy for how much shake-like structure leaks into the residual), and
    - ``max_abs_correction_from_observed``: ``max |output - observed|`` (a translation crop proxy for the
      largest shift a renderer would have to apply relative to the observed path),

    each overall and per channel.
    """
    intended = _as_path("intended", intended)
    observed = _as_path("observed", observed)
    output = _as_path("output", output)
    if not (intended.shape == observed.shape == output.shape):
        raise ValueError(
            f"intended, observed and output must share one shape, got {intended.shape}, "
            f"{observed.shape}, {output.shape}")

    residual = output - intended
    correction = output - observed
    second = residual[2:] - 2.0 * residual[1:-1] + residual[:-2]
    return {
        "rms_path_deviation": float(np.sqrt(np.mean(residual * residual))),
        "rms_path_deviation_per_channel": [float(v) for v in np.sqrt(np.mean(residual * residual, axis=0))],
        "rms_second_difference_residual": float(np.sqrt(np.mean(second * second))),
        "rms_second_difference_residual_per_channel": [
            float(v) for v in np.sqrt(np.mean(second * second, axis=0))],
        "max_abs_correction_from_observed": float(np.max(np.abs(correction))),
        "max_abs_correction_from_observed_per_channel": [
            float(v) for v in np.max(np.abs(correction), axis=0)],
    }


# --------------------------------------------------------------------------------------
# synthetic known-truth fixture
# --------------------------------------------------------------------------------------

def _ar1_colored_shake(rng, n, *, coefficients, amplitudes):
    """Deterministic stationary colored AR(1) realization, one independent process per channel.

    ``x_t = rho * x_{t-1} + sigma * eps_t`` with ``eps_t ~ N(0, 1)`` and
    ``sigma = amplitude * sqrt(1 - rho^2)`` so the stationary per-channel standard deviation is
    ``amplitude``; the first sample is drawn from that stationary distribution. This is a plain colored AR(1)
    process: it is explicitly **not fractal** and **not 1/f noise**.
    """
    out = np.zeros((n, 2), np.float64)
    for d in range(2):
        rho = float(coefficients[d])
        amp = float(amplitudes[d])
        sigma = amp * math.sqrt(max(0.0, 1.0 - rho * rho))
        x = amp * float(rng.standard_normal())
        for i in range(n):
            x = rho * x + sigma * float(rng.standard_normal())
            out[i, d] = x
    return out


def synthetic_translation_fixture(*, n=DEFAULT_N, fps=DEFAULT_FPS, seed=0, condition="pan_shake"):
    """Deterministic known-truth camera translation: slow pan plus one optional perturbation.

    Timestamps are uniform (``arange(n) / fps``) so the frame-indexed L1 clock and the elapsed-time multiscale
    clock coincide. The intended pan is seed-independent ground truth; only the perturbation depends on
    ``seed``. ``condition`` is one of:

    - ``static``: zero pan and zero perturbation,
    - ``pan_no_shake``: the known sinusoidal pan only,
    - ``pan_shake``: pan plus deterministic-seed medium and fast periodic sinusoidal shake,
    - ``pan_stochastic_shake``: pan plus a deterministic-seed **stationary colored AR(1)** shake with fixed
      per-channel amplitudes and coefficients; explicitly a non-fractal stress control (not 1/f noise),
    - ``pan_jolt``: pan plus a fixed-time one-sided transient displacement (jolt) with exponential recovery,
      placed at a fixed fraction of the clip duration so it always falls inside any valid clip; explicitly a
      non-periodic, non-fractal stress control.

    Returns a dict of NumPy arrays ``times [T]``, ``pan [T, 2]``, ``shake [T, 2]`` and ``observed [T, 2]``
    (``observed = pan + shake`` exactly), plus the per-channel amplitudes and coefficients actually used.
    This is a synthetic cartoon, not real footage.
    """
    n = _validate_n(n)
    fps = _validate_fps(fps)
    seed = _validate_seed(seed)
    condition = _validate_condition(condition)

    times = np.arange(n, dtype=np.float64) / fps
    t = times - times[0]
    two_pi = 2.0 * math.pi

    pan = np.zeros((n, 2), np.float64)
    shake = np.zeros((n, 2), np.float64)
    pan_amplitude = (0.0, 0.0)
    medium_amplitude = (0.0, 0.0)
    fast_amplitude = (0.0, 0.0)
    stochastic_amplitude = (0.0, 0.0)
    jolt_amplitude = (0.0, 0.0)
    jolt_time_seconds = None

    if condition != "static":
        pan = np.stack([
            PAN_AMPLITUDE[d] * np.sin(two_pi * t / PAN_PERIOD_SECONDS + PAN_PHASE_RADIANS[d])
            for d in range(2)
        ], axis=1)
        pan_amplitude = PAN_AMPLITUDE

    if condition == "pan_shake":
        rng = np.random.default_rng(seed)
        medium_phase = rng.uniform(0.0, two_pi, size=2)
        fast_phase = rng.uniform(0.0, two_pi, size=2)
        medium = np.stack([
            MEDIUM_SHAKE_AMPLITUDE[d] * np.sin(two_pi * t / MEDIUM_SHAKE_PERIOD_SECONDS + medium_phase[d])
            for d in range(2)
        ], axis=1)
        fast = np.stack([
            FAST_SHAKE_AMPLITUDE[d] * np.sin(two_pi * t / FAST_SHAKE_PERIOD_SECONDS + fast_phase[d])
            for d in range(2)
        ], axis=1)
        shake = medium + fast
        medium_amplitude = MEDIUM_SHAKE_AMPLITUDE
        fast_amplitude = FAST_SHAKE_AMPLITUDE
    elif condition == "pan_stochastic_shake":
        shake = _ar1_colored_shake(
            np.random.default_rng(seed), n,
            coefficients=STOCHASTIC_AR1_COEFFICIENT, amplitudes=STOCHASTIC_AR1_AMPLITUDE)
        stochastic_amplitude = STOCHASTIC_AR1_AMPLITUDE
    elif condition == "pan_jolt":
        duration = float(times[-1] - times[0])
        jolt_time_seconds = JOLT_TIME_FRACTION * duration
        elapsed = t - jolt_time_seconds
        after = elapsed >= 0.0
        jolt = np.zeros((n, 2), np.float64)
        for d in range(2):
            jolt[after, d] = JOLT_AMPLITUDE[d] * np.exp(-elapsed[after] / JOLT_DECAY_SECONDS)
        shake = jolt
        jolt_amplitude = JOLT_AMPLITUDE

    return {
        "times": times,
        "pan": pan,
        "shake": shake,
        "observed": pan + shake,
        "pan_amplitude": [float(v) for v in pan_amplitude],
        "medium_shake_amplitude": [float(v) for v in medium_amplitude],
        "fast_shake_amplitude": [float(v) for v in fast_amplitude],
        "stochastic_shake_amplitude": [float(v) for v in stochastic_amplitude],
        "stochastic_ar1_coefficient": [float(v) for v in STOCHASTIC_AR1_COEFFICIENT],
        "jolt_amplitude": [float(v) for v in jolt_amplitude],
        "jolt_decay_seconds": float(JOLT_DECAY_SECONDS),
        "jolt_time_fraction": float(JOLT_TIME_FRACTION),
        "jolt_time_seconds": (None if jolt_time_seconds is None else float(jolt_time_seconds)),
    }


# --------------------------------------------------------------------------------------
# candidate paths
# --------------------------------------------------------------------------------------

def build_multiscale_output(observed, times, weights, *, scales=MULTISCALE_SCALES_SECONDS):
    """Reconstruct a path from the committed decomposition with one retention weight per detail band.

    Uses :func:`bench.multiscale_motion.decompose` with the fixed ``scales`` and
    :func:`bench.multiscale_motion.path_reconstruct`. The trend is always retained. Validation of the signal,
    timestamps, scales and weights is delegated to ``bench.multiscale_motion`` (its ``MultiscaleError`` is a
    ``ValueError``).
    """
    decomp = MM.decompose(observed, times, [float(s) for s in scales])
    return MM.path_reconstruct(decomp, np.asarray(list(weights), dtype=np.float64))


def build_l1_output(observed, budget, *, weights=L1_WEIGHTS, fidelity=L1_FIDELITY,
                    window=L1_WINDOW, overlap=L1_OVERLAP):
    """Smooth every channel of ``observed [T, D]`` with production ``l1_path_windowed`` at one ``budget``."""
    sig = _as_path("observed", observed)
    try:
        b = float(budget)
    except (TypeError, ValueError):
        raise ValueError(f"budget must be a finite non-negative number, got {budget!r}")
    if not math.isfinite(b) or b < 0.0:
        raise ValueError(f"budget must be finite and >= 0, got {budget!r}")
    try:
        f = float(fidelity)
    except (TypeError, ValueError):
        raise ValueError(f"fidelity must be finite and >= 0, got {fidelity!r}")
    if not math.isfinite(f) or f < 0.0:
        raise ValueError(f"fidelity must be finite and >= 0, got {fidelity!r}")
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or int(window) < 1:
        raise ValueError(f"window must be >= 1, got {window!r}")
    if isinstance(overlap, bool) or not isinstance(overlap, (int, np.integer)) or int(overlap) < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap!r}")
    if int(overlap) >= int(window):
        # Keeps the windowed solver strictly progressing: each step advances by window - overlap >= 1 samples,
        # so any accepted (window, overlap) terminates instead of re-solving the same window forever.
        raise ValueError(
            f"overlap must be strictly smaller than window so the windowed solver advances "
            f"(step = window - overlap >= 1), got overlap={overlap!r}, window={window!r}")

    out = np.empty_like(sig)
    for d in range(sig.shape[1]):
        out[:, d] = l1_path_windowed(sig[:, d], b, tuple(weights), int(window), int(overlap), fidelity=f)
    return out


# --------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------

def nearest_crop_index(target_crop, l1_crop_proxies):
    """Index into the fixed-order ``l1_crop_proxies`` whose value is nearest ``target_crop``.

    The search walks the sequence in its given (fixed grid) order using a strict improvement test, so a tie
    between two equally distant proxies resolves to the **earliest** index in grid order. All inputs must be
    finite; returns a plain int. This is a descriptive nearest-match over already-measured proxies, not a
    selection, ranking or optimization.
    """
    try:
        proxies = [float(v) for v in l1_crop_proxies]
    except (TypeError, ValueError):
        raise ValueError(f"l1_crop_proxies must be numeric, got {l1_crop_proxies!r}")
    if not proxies:
        raise ValueError("l1_crop_proxies must be non-empty")
    try:
        target = float(target_crop)
    except (TypeError, ValueError):
        raise ValueError(f"target_crop must be numeric, got {target_crop!r}")
    if not math.isfinite(target) or not all(math.isfinite(v) for v in proxies):
        raise ValueError("crop proxies must all be finite")
    best = 0
    best_dist = abs(proxies[0] - target)
    for i in range(1, len(proxies)):
        dist = abs(proxies[i] - target)
        if dist < best_dist:
            best_dist = dist
            best = i
    return best


def _variant(family, name, parameters, intended, observed, output):
    return {
        "family": family,
        "name": name,
        "parameters": parameters,
        "metrics": translation_path_metrics(intended, observed, output),
        "scope": VARIANT_SCOPE,
    }


def _run_one(condition, seed, n, fps, scales):
    fx = synthetic_translation_fixture(n=n, fps=fps, seed=seed, condition=condition)
    times, pan, shake, observed = fx["times"], fx["pan"], fx["shake"], fx["observed"]

    decom = MM.decompose(observed, times, list(scales))
    multiscale_variants = []
    for name, weights in MULTISCALE_CANDIDATES:
        out = MM.path_reconstruct(decom, np.asarray(list(weights), dtype=np.float64))
        multiscale_variants.append(_variant(
            "multiscale", name,
            {"scales_seconds": list(scales), "retention_weights": [float(w) for w in weights]},
            pan, observed, out))
    l1_variants = []
    for budget in L1_BUDGETS:
        out = build_l1_output(observed, budget)
        l1_variants.append(_variant(
            "l1_budget", f"budget_{budget:g}",
            {"budget": float(budget), "weights": [float(w) for w in L1_WEIGHTS], "fidelity": float(L1_FIDELITY),
             "window": int(L1_WINDOW), "overlap": int(L1_OVERLAP)},
            pan, observed, out))
    variants = multiscale_variants + l1_variants

    # Matched-crop pairing: each multiscale candidate is paired with the L1 variant whose measured crop proxy
    # is nearest the multiscale candidate's own crop proxy. Ties follow the fixed L1 grid order. No winner,
    # rank, score or aggregate is produced; both sides keep their own metrics in the record.
    l1_crops = [v["metrics"]["max_abs_correction_from_observed"] for v in l1_variants]
    pairs = []
    for mv in multiscale_variants:
        target = float(mv["metrics"]["max_abs_correction_from_observed"])
        idx = nearest_crop_index(target, l1_crops)
        lv = l1_variants[idx]
        l1_crop = float(l1_crops[idx])
        pairs.append({
            "multiscale_name": mv["name"],
            "l1_name": lv["name"],
            "multiscale_crop_proxy": target,
            "l1_crop_proxy": l1_crop,
            "abs_crop_mismatch": float(abs(target - l1_crop)),
            "multiscale_rms_path_deviation": float(mv["metrics"]["rms_path_deviation"]),
            "l1_rms_path_deviation": float(lv["metrics"]["rms_path_deviation"]),
            "multiscale_rms_second_difference_residual": float(
                mv["metrics"]["rms_second_difference_residual"]),
            "l1_rms_second_difference_residual": float(lv["metrics"]["rms_second_difference_residual"]),
            "crop_match_tolerance": float(L1_CROP_MATCH_TOLERANCE),
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
            "condition_model": CONDITION_MODELS[condition],
            "condition_kind": CONDITION_KIND[condition],
            "pan_amplitude": fx["pan_amplitude"],
            "medium_shake_amplitude": fx["medium_shake_amplitude"],
            "fast_shake_amplitude": fx["fast_shake_amplitude"],
            "stochastic_shake_amplitude": fx["stochastic_shake_amplitude"],
            "stochastic_ar1_coefficient": fx["stochastic_ar1_coefficient"],
            "jolt_amplitude": fx["jolt_amplitude"],
            "jolt_decay_seconds": fx["jolt_decay_seconds"],
            "jolt_time_fraction": fx["jolt_time_fraction"],
            "jolt_time_seconds": fx["jolt_time_seconds"],
            "pan_rms": float(np.sqrt(np.mean(pan * pan))),
            "shake_rms": float(np.sqrt(np.mean(shake * shake))),
            "observed_rms": float(np.sqrt(np.mean(observed * observed))),
        },
        "variants": variants,
        "pairs": pairs,
    }


def run_audit(*, seeds=DEFAULT_SEEDS, conditions=DEFAULT_CONDITIONS, n=DEFAULT_N, fps=DEFAULT_FPS):
    """Run the fixed-grid comparison over the given seeds/conditions and return a JSON-safe report.

    Every ``(condition, seed)`` run reports the same fixed multiscale candidate grid and the same predeclared
    uniform L1 budget grid (``0.0..2.0`` in ``0.1`` steps, budget ``0.0`` being the observed-path identity),
    with each variant's metrics and parameters listed separately. Each run also carries a ``pairs`` list that
    matches every multiscale candidate to the L1 variant with the nearest measured crop proxy (ties to the
    earlier grid index) and reports both sides' crop proxies, the absolute mismatch, both path-error and
    residual-second-difference metrics and the fixed-grid match tolerance. No cross-variant aggregate, winner,
    ranking, score or tuned selection is computed. ``n`` is capped at ``L1_WINDOW`` so the L1 solver covers
    the whole clip in a single window, which keeps its temporal access matched with the full-clip multiscale
    Gaussian; these are offline path-level numbers, not online latency. The returned dict contains only
    Python scalars, strings, lists and dicts (no NumPy objects and no filesystem paths); validation errors
    raise ``ValueError``.
    """
    seeds = _validate_seeds(seeds)
    conditions = _validate_conditions(conditions)
    n = _validate_n(n)
    if n > L1_WINDOW:
        # The full-clip comparison only holds while l1_path_windowed stays on its T <= window branch (one
        # global L1 solve); a longer clip would switch to overlapping windows and break the matched access.
        raise ValueError(
            f"n must be <= L1_WINDOW ({L1_WINDOW}) so l1_path_windowed solves one L1 problem over the whole "
            f"clip, matching the full-clip multiscale Gaussian; got n={n}")
    fps = _validate_fps(fps)
    scales = [float(s) for s in MULTISCALE_SCALES_SECONDS]

    runs = [_run_one(condition, seed, n, fps, scales) for condition in conditions for seed in seeds]

    return {
        "schema": SCHEMA,
        "ok": True,
        "scope_note": SCOPE_NOTE,
        "scope": [
            "1D/2D translation path comparison only",
            "known-truth synthetic camera translation; the intended pan is the ground truth",
            "metrics are reported per seed and per variant and are never pooled into one number",
        ],
        "reporting_policy": REPORTING_POLICY,
        "path_metrics": [
            "rms_path_deviation: RMS of output - intended pan",
            "rms_second_difference_residual: RMS of the second difference of output - intended pan "
            "(path-level jitter proxy)",
            "max_abs_correction_from_observed: max |output - observed| (translation crop proxy)",
        ],
        "timestamp_basis": {
            "kind": "uniform",
            "dt_seconds": float(1.0 / fps),
            "rationale": (
                "l1_path_windowed is frame-indexed, so uniform spacing makes the frame clock and the "
                "elapsed-time clock agree and the fixed multiscale scales (seconds) and L1 budgets share one "
                "clock"
            ),
        },
        "temporal_access": {
            "kind": "offline_full_clip",
            "max_samples": int(L1_WINDOW),
            "l1": (
                "mlxstab.smooth.l1_path_windowed is given the whole clip in a single window: the audit caps n at "
                "L1_WINDOW, so its T <= window branch solves a single L1 problem over every sample"
            ),
            "multiscale": (
                "bench.multiscale_motion.decompose applies a timestamp-aware Gaussian over every frame of the "
                "full clip"
            ),
            "status": "matched",
            "statement": (
                "Both families are evaluated with offline temporal access to the whole clip: the audit caps n "
                "at L1_WINDOW so the L1 solver covers every sample in a single window, and the timestamp-aware "
                "Gaussian consumes the full clip too, so their temporal access is matched. These are offline "
                "path-level numbers, not online latency or causal-streaming measurements."
            ),
        },
        "pairing": {
            "rule": (
                "each multiscale candidate is paired with the L1 variant whose measured crop proxy is nearest "
                "the multiscale candidate's own measured crop proxy; ties resolve to the earlier L1 grid index"
            ),
            "tolerance": float(L1_CROP_MATCH_TOLERANCE),
            "tolerance_definition": "half the uniform L1 budget grid step",
        },
        "grids": {
            "multiscale_scales_seconds": scales,
            "multiscale_candidates": [
                {"name": name, "retention_weights": [float(w) for w in weights]}
                for name, weights in MULTISCALE_CANDIDATES
            ],
            "l1_budgets": [float(b) for b in L1_BUDGETS],
            "l1_budget_min": float(L1_BUDGET_MIN),
            "l1_budget_max": float(L1_BUDGET_MAX),
            "l1_budget_step": float(L1_BUDGET_STEP),
            "l1_budget_count": int(L1_BUDGET_COUNT),
            "l1_crop_match_tolerance": float(L1_CROP_MATCH_TOLERANCE),
            "l1_weights": [float(w) for w in L1_WEIGHTS],
            "l1_fidelity": float(L1_FIDELITY),
            "l1_window": int(L1_WINDOW),
            "l1_overlap": int(L1_OVERLAP),
        },
        "decomposition": {
            "implementation": "bench.multiscale_motion.decompose + path_reconstruct",
            "kernel": dict(MM.KERNEL_DOC),
            "reconstruction": "trend + sum_k weights[k] * bands[k]; the trend is always retained",
        },
        "l1": {
            "implementation": "mlxstab.smooth.l1_path_windowed",
            "objective": "argmin_p w1*|D p| + w2*|D^2 p| + w3*|D^3 p| subject to |p - observed| <= budget",
        },
        "fixture_defaults": {
            "n_samples": int(n),
            "fps": float(fps),
            "seeds": list(seeds),
            "conditions": list(conditions),
            "conditions_supported": list(CONDITIONS),
            "condition_models": {c: CONDITION_MODELS[c] for c in CONDITIONS},
            "condition_kinds": {c: CONDITION_KIND[c] for c in CONDITIONS},
            "pan_period_seconds": float(PAN_PERIOD_SECONDS),
            "pan_phase_radians": [float(v) for v in PAN_PHASE_RADIANS],
            "pan_amplitude": [float(v) for v in PAN_AMPLITUDE],
            "medium_shake_period_seconds": float(MEDIUM_SHAKE_PERIOD_SECONDS),
            "medium_shake_amplitude": [float(v) for v in MEDIUM_SHAKE_AMPLITUDE],
            "fast_shake_period_seconds": float(FAST_SHAKE_PERIOD_SECONDS),
            "fast_shake_amplitude": [float(v) for v in FAST_SHAKE_AMPLITUDE],
            "stochastic_ar1_amplitude": [float(v) for v in STOCHASTIC_AR1_AMPLITUDE],
            "stochastic_ar1_coefficient": [float(v) for v in STOCHASTIC_AR1_COEFFICIENT],
            "jolt_amplitude": [float(v) for v in JOLT_AMPLITUDE],
            "jolt_decay_seconds": float(JOLT_DECAY_SECONDS),
            "jolt_time_fraction": float(JOLT_TIME_FRACTION),
        },
        "runs": runs,
        "limitations": list(LIMITATIONS),
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_seeds(text):
    """Parse ``"0,1,2"`` into a validated non-empty seed tuple."""
    if isinstance(text, (list, tuple)):
        parts = list(text)
    else:
        parts = [p for p in str(text).split(",") if str(p).strip()]
    out = []
    for p in parts:
        try:
            out.append(int(str(p).strip()))
        except (TypeError, ValueError):
            raise ValueError(f"invalid seed {p!r}")
    return _validate_seeds(out)


def parse_conditions(text):
    """Parse e.g. ``"static,pan_no_shake,pan_shake,pan_stochastic_shake,pan_jolt"`` into a tuple.

    Any subset of :data:`CONDITIONS` is accepted; the result is validated as a non-empty tuple of unique
    known conditions.
    """
    if isinstance(text, (list, tuple)):
        parts = list(text)
    else:
        parts = [p.strip() for p in str(text).split(",") if p.strip()]
    return _validate_conditions(parts)


def build_parser():
    p = argparse.ArgumentParser(
        prog="bench/multiscale_motion_audit.py",
        description="Research-only path-level comparison of fixed multiscale-retention candidates against L1 "
                    "budgets on synthetic camera translation (no media, no rendering). Conditions are "
                    "static, pan_no_shake, pan_shake, and two non-fractal stress controls "
                    "(pan_stochastic_shake: stationary colored AR(1); pan_jolt: fixed-time exponential-"
                    "recovery transient). Sample count is capped at L1_WINDOW so the L1 solver sees the "
                    "whole clip in one window.")
    p.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                   help="comma-separated integer seeds (default: %(default)s)")
    p.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS),
                   help=f"comma-separated conditions chosen from {CONDITIONS} (default: %(default)s)")
    p.add_argument("--n", type=int, default=DEFAULT_N,
                   help=f"number of uniform samples, {MIN_SAMPLES}..{L1_WINDOW} (default: %(default)s)")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS, help="uniform sample rate (default: %(default)s)")
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
            seeds=parse_seeds(args.seeds),
            conditions=parse_conditions(args.conditions),
            n=args.n,
            fps=args.fps,
        )
    except (ValueError, MM.MultiscaleError) as exc:
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
    "MULTISCALE_SCALES_SECONDS",
    "MULTISCALE_CANDIDATES",
    "L1_BUDGETS",
    "L1_BUDGET_MIN",
    "L1_BUDGET_MAX",
    "L1_BUDGET_STEP",
    "L1_BUDGET_COUNT",
    "L1_CROP_MATCH_TOLERANCE",
    "L1_WEIGHTS",
    "L1_WINDOW",
    "L1_OVERLAP",
    "STOCHASTIC_AR1_AMPLITUDE",
    "STOCHASTIC_AR1_COEFFICIENT",
    "JOLT_AMPLITUDE",
    "JOLT_DECAY_SECONDS",
    "JOLT_TIME_FRACTION",
    "CONDITIONS",
    "CONDITION_MODELS",
    "CONDITION_KIND",
    "_ar1_colored_shake",
    "DEFAULT_N",
    "DEFAULT_FPS",
    "DEFAULT_SEEDS",
    "DEFAULT_CONDITIONS",
    "SCOPE_NOTE",
    "LIMITATIONS",
    "translation_path_metrics",
    "nearest_crop_index",
    "synthetic_translation_fixture",
    "build_multiscale_output",
    "build_l1_output",
    "run_audit",
    "parse_seeds",
    "parse_conditions",
    "build_parser",
    "main",
]
