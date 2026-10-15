"""Stage 4 research baseline: timestamp-aware temporal multiscale decomposition (pure NumPy; not wired).

This is a **research-only** starting point for the Stage 4 / E10 question in ``PLAN.md``: decompose an
estimated camera-motion signal over *elapsed time* so that intentional motion and residual jitter can be
inspected band by band. It is deliberately isolated:

- it depends only on NumPy (plus the standard library),
- it does not import ``mlxstab``, does not run RAFT, does not decode media and does not change any
  production default,
- it makes **no** quality, ranking or stabilizer comparison claim, and
- it reports raw per-band energy and scale metadata, never a winner.

What it is not
--------------
The outputs are a descriptive multiscale decomposition. A detail band is **not** depth, **not** uncertainty,
**not** a fractal/Hurst estimate and **not** a selected "true motion" component. No band is chosen as a
winner and no comparison is made against the production L1 path planner. Turning these bands into a planner
policy is a later experiment (fixed/adaptive weights at matched crop and delay) and is out of scope here.

Decomposition
-------------
Given a motion signal ``x`` of shape ``[T, D]`` (``T`` time samples, ``D`` channels such as camera
translation/rotation parameters) and strictly increasing timestamps ``t`` in seconds, and given sorted
strictly increasing positive scales ``s_1 < ... < s_K`` in seconds:

    L_s(x)_i = sum_j w_ij x_j,     w_ij = exp(-0.5 ((t_i - t_j) / s)^2) / sum_j exp(-0.5 ((t_i - t_j) / s)^2)

is a normalized Gaussian low-pass evaluated on the **actual timestamp differences** ``t_i - t_j`` (seconds),
not on frame indices, so variable frame intervals are handled directly. Row normalization guarantees a
constant signal is reproduced exactly (partition of unity), and the kernel is symmetric with zero phase.

With ``S_0 = x`` (the unsmoothed signal) and ``S_k = L_{s_k}(x)``:

    band_k = S_{k-1} - S_k        (fine -> coarse; band_1 is the finest detail)
    trend  = S_K

so the bands telescope: ``sum_k band_k + trend = x`` exactly in exact arithmetic. ``path_reconstruct``
applies one retention weight in ``[0, 1]`` per detail band and always keeps the trend; all-ones weights
return the original signal (the implementation short-circuits that case so the stated contract is exact).

Bounded memory
--------------
The low-pass processes output time rows in blocks of at most ``chunk`` rows, so the largest kernel temporary
is ``[chunk, T]`` rather than a full ``[T, T]`` matrix. Bands and trend are the decomposition output and
necessarily occupy ``O(K * T * D)``; that is the returned result, not a scratch buffer.

Honesty boundary
----------------
- The Gaussian scale is a smoothing width in seconds, not a physical camera-response or depth prior.
- Band RMS is an energy-like summary of that band's samples; it is not variance explained of a known
  ground-truth motion, not a confidence and not an uncertainty.
- The synthetic fixture below is a deterministic cartoon (slow intentional pan plus two shake bands). It
  demonstrates exact reconstruction and per-band inspectability only. It is not real footage and says
  nothing about whether multiscale planning beats the L1 baseline.

Run: ``.venv/bin/python -c "import bench.multiscale_motion"`` (pure NumPy, no GPU, no heavy lock).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCHEMA = "mlxstab.multiscale_motion/1"

# Default output-row block size for the low-pass; only bounds the temporary, not the result.
DEFAULT_CHUNK = 256

# Description of the kernel, stored in the report so a reader does not have to re-derive it.
KERNEL_DOC = {
    "name": "normalized_gaussian_on_timestamp_differences",
    "kind": "low_pass",
    "argument_units": "seconds",
    "definition": "w_ij = exp(-0.5*((t_i - t_j)/scale)^2) / sum_j exp(-0.5*((t_i - t_j)/scale)^2)",
    "time_source": "explicit strictly increasing timestamps in seconds (not frame indices)",
    "row_normalized": True,
    "assumes_uniform_spacing": False,
}

SCOPE_NOTE = (
    "Research-only Stage 4 temporal multiscale decomposition of a [T, D] camera-motion signal. It returns "
    "fine-to-coarse Gaussian detail bands plus a coarsest trend that reconstruct the signal exactly, and "
    "reports per-band RMS energy and scale metadata. It is not depth, not uncertainty, not a fractal/Hurst "
    "estimate, and not a selected winner or quality score. It does not compare itself against the production "
    "L1 path planner and does not change any production code or default."
)


class MultiscaleError(ValueError):
    """Invalid input to the multiscale decomposition. ``code`` is a stable machine-readable reason."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class MultiscaleDecomposition:
    """Result of :func:`decompose`.

    ``bands`` is ``[K, T, D]`` ordered fine to coarse (``bands[0]`` is the finest detail), ``trend`` is
    ``[T, D]`` (the coarsest low-pass) and ``scales`` is the ``[K]`` strictly increasing scale array in
    seconds. ``signal`` and ``times`` are stored so the result is self-contained and auditable.
    """

    times: np.ndarray          # [T] seconds, strictly increasing
    signal: np.ndarray         # [T, D]
    bands: np.ndarray          # [K, T, D], fine -> coarse
    trend: np.ndarray          # [T, D]
    scales: np.ndarray         # [K] seconds, strictly increasing
    chunk: int


# --------------------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------------------

def _as_signal(signal):
    try:
        arr = np.asarray(signal, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MultiscaleError("bad_shape", f"signal is not a numeric array: {exc}")
    if arr.ndim != 2:
        raise MultiscaleError("bad_shape", f"signal must be 2-D [T, D], got shape {arr.shape}")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise MultiscaleError("bad_shape", f"signal must have T >= 1 and D >= 1, got shape {arr.shape}")
    if not bool(np.isfinite(arr).all()):
        raise MultiscaleError("nonfinite_signal", "signal contains non-finite values")
    return arr


def _as_times(times, T):
    try:
        arr = np.asarray(times, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MultiscaleError("bad_times", f"times is not a numeric array: {exc}")
    if arr.ndim != 1 or arr.shape[0] != T:
        raise MultiscaleError("bad_times", f"times must be 1-D of length T={T}, got shape {arr.shape}")
    if not bool(np.isfinite(arr).all()):
        raise MultiscaleError("bad_times", "times contains non-finite values")
    if T >= 2 and not bool(np.all(np.diff(arr) > 0.0)):
        raise MultiscaleError("nonmonotonic_times", "times must be strictly increasing (seconds)")
    return arr


def _as_scales(scales):
    try:
        arr = np.asarray(scales, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MultiscaleError("bad_scales", f"scales is not a numeric array: {exc}")
    if arr.ndim != 1 or arr.size < 1:
        raise MultiscaleError("bad_scales", f"scales must be a non-empty 1-D array, got shape {arr.shape}")
    if not bool(np.isfinite(arr).all()):
        raise MultiscaleError("bad_scales", "scales contains non-finite values")
    if not bool(np.all(arr > 0.0)):
        raise MultiscaleError("nonpositive_scale", f"scales must all be > 0 seconds, got {arr.tolist()}")
    if arr.size >= 2 and not bool(np.all(np.diff(arr) > 0.0)):
        raise MultiscaleError("unsorted_scales",
                              f"scales must be sorted strictly increasing (seconds), got {arr.tolist()}")
    return arr


def _as_weights(weights, K):
    try:
        arr = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MultiscaleError("bad_weights", f"weights is not a numeric array: {exc}")
    if arr.ndim != 1 or arr.shape[0] != K:
        raise MultiscaleError("bad_weights", f"weights must be 1-D of length K={K}, got shape {arr.shape}")
    if not bool(np.isfinite(arr).all()):
        raise MultiscaleError("bad_weights", "weights contains non-finite values")
    if bool((arr < 0.0).any()) or bool((arr > 1.0).any()):
        raise MultiscaleError("weights_out_of_range",
                              f"weights must lie in [0, 1], got min {float(arr.min())} max {float(arr.max())}")
    return arr


def _as_chunk(chunk):
    try:
        c = int(chunk)
    except (TypeError, ValueError):
        raise MultiscaleError("bad_chunk", f"chunk must be a positive integer, got {chunk!r}")
    if c != chunk or c < 1:
        raise MultiscaleError("bad_chunk", f"chunk must be a positive integer, got {chunk!r}")
    return c


# --------------------------------------------------------------------------------------
# low-pass
# --------------------------------------------------------------------------------------

def _gaussian_lowpass(signal, times, scale, chunk):
    """Normalized Gaussian low-pass on validated inputs; assumes shapes already checked."""
    T = signal.shape[0]
    out = np.empty_like(signal)
    with np.errstate(over="ignore", under="ignore"):
        for start in range(0, T, chunk):
            stop = min(start + chunk, T)
            dt = times[start:stop, None] - times[None, :]              # [<=chunk, T] seconds
            w = np.exp(-0.5 * (dt / scale) ** 2)
            denom = w.sum(axis=1, keepdims=True)                       # >= 1 (the self term has dt = 0)
            w = w / denom
            out[start:stop] = w @ signal
    return out


def gaussian_lowpass(signal, times, scale, *, chunk=DEFAULT_CHUNK):
    """Normalized Gaussian low-pass of ``signal [T, D]`` at one ``scale`` in seconds.

    The kernel uses the actual timestamp differences ``t_i - t_j`` and is row-normalized, so a constant
    signal is returned unchanged. Time rows are processed in blocks of at most ``chunk`` rows.
    """
    sig = _as_signal(signal)
    t = _as_times(times, sig.shape[0])
    try:
        sc = float(scale)
    except (TypeError, ValueError):
        raise MultiscaleError("bad_scales", f"scale is not a scalar: {scale!r}")
    if not np.isfinite(sc) or sc <= 0.0:
        raise MultiscaleError("nonpositive_scale", f"scale must be a finite value > 0 seconds, got {scale!r}")
    c = _as_chunk(chunk)
    return _gaussian_lowpass(sig, t, sc, c)


# --------------------------------------------------------------------------------------
# decomposition and reconstruction
# --------------------------------------------------------------------------------------

def decompose(signal, times, scales, *, chunk=DEFAULT_CHUNK):
    """Fine-to-coarse temporal multiscale decomposition of ``signal [T, D]``.

    ``times [T]`` are strictly increasing seconds and ``scales [K]`` are strictly increasing positive
    seconds. Returns a :class:`MultiscaleDecomposition` whose ``bands`` satisfy
    ``sum(bands, axis=0) + trend == signal`` (exact up to floating-point round-off) with ``bands[0]`` the
    finest detail and ``trend`` the coarsest low-pass.
    """
    sig = _as_signal(signal)
    t = _as_times(times, sig.shape[0])
    sc = _as_scales(scales)
    c = _as_chunk(chunk)

    K = int(sc.size)
    bands = np.empty((K,) + sig.shape, dtype=np.float64)
    prev = sig                                                            # S_0 = the raw signal
    for k in range(K):
        cur = _gaussian_lowpass(sig, t, float(sc[k]), c)                  # S_{k+1}
        bands[k] = prev - cur                                             # band_k = S_k - S_{k+1}
        prev = cur
    trend = prev.copy()                                                   # S_K
    return MultiscaleDecomposition(times=t.copy(), signal=sig.copy(), bands=bands, trend=trend,
                                   scales=sc.copy(), chunk=c)


def path_reconstruct(decomp, weights=None):
    """Apply one retention weight in ``[0, 1]`` per detail band; the trend is always retained.

    ``out = trend + sum_k weights[k] * bands[k]``. ``weights=None`` (or all ones) returns the original
    signal exactly: that case is short-circuited to the stored signal so the contract is bit-exact, while
    the general case uses the ordinary sum.
    """
    if not isinstance(decomp, MultiscaleDecomposition):
        raise TypeError("path_reconstruct expects a MultiscaleDecomposition")
    K = int(decomp.bands.shape[0])
    if weights is None:
        return decomp.signal.copy()
    w = _as_weights(weights, K)
    if bool(np.all(w == 1.0)):
        return decomp.signal.copy()                                       # exact all-ones fast path
    return decomp.trend + np.einsum("k,ktd->td", w, decomp.bands)


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------

def _dt_seconds(times):
    if times.shape[0] < 2:
        return {"min": None, "max": None, "mean": None}
    d = np.diff(times)
    return {"min": float(d.min()), "max": float(d.max()), "mean": float(d.mean())}


def band_energy_report(decomp):
    """Per-band RMS energy and scale metadata for a decomposition (plain Python, JSON-safe values).

    Every band entry carries its index, its ``scale_seconds`` (the coarser Gaussian width), the
    ``finer_scale_seconds`` it is differenced against (``0.0`` for the finest band, meaning the raw signal)
    and the RMS energy of the band overall and per channel. The trend entry carries the coarsest scale. The
    report also includes the reconstruction residual so the exactness statement is self-checking.
    """
    if not isinstance(decomp, MultiscaleDecomposition):
        raise TypeError("band_energy_report expects a MultiscaleDecomposition")
    bands, trend, sc = decomp.bands, decomp.trend, decomp.scales
    K, T, D = bands.shape

    band_rms = np.sqrt(np.mean(bands * bands, axis=(1, 2)))               # [K]
    band_rms_ch = np.sqrt(np.mean(bands * bands, axis=1))                 # [K, D]
    recon = trend + bands.sum(axis=0)
    recon_err = float(np.max(np.abs(recon - decomp.signal)))

    entries = []
    for k in range(K):
        entries.append({
            "index": k,
            "scale_seconds": float(sc[k]),
            "finer_scale_seconds": float(sc[k - 1]) if k > 0 else 0.0,
            "coarser_scale_seconds": float(sc[k]),
            "band_seconds": [float(sc[k - 1]) if k > 0 else 0.0, float(sc[k])],
            "rms": float(band_rms[k]),
            "rms_per_channel": [float(v) for v in band_rms_ch[k]],
        })

    trend_rms_ch = np.sqrt(np.mean(trend * trend, axis=0))                # [D]
    signal_rms_ch = np.sqrt(np.mean(decomp.signal * decomp.signal, axis=0))

    return {
        "schema": SCHEMA,
        "kernel": dict(KERNEL_DOC),
        "signal": {
            "n_samples": int(T),
            "n_channels": int(D),
            "duration_seconds": float(decomp.times[-1] - decomp.times[0]) if T >= 1 else 0.0,
            "dt_seconds": _dt_seconds(decomp.times),
        },
        "chunk": int(decomp.chunk),
        "n_bands": K,
        "scales_seconds": [float(s) for s in sc],
        "bands": entries,
        "trend": {
            "scale_seconds": float(sc[-1]),
            "rms": float(np.sqrt(np.mean(trend * trend))),
            "rms_per_channel": [float(v) for v in trend_rms_ch],
        },
        "signal_rms": float(np.sqrt(np.mean(decomp.signal * decomp.signal))),
        "signal_rms_per_channel": [float(v) for v in signal_rms_ch],
        "reconstruction_max_abs_error": recon_err,
        "scope": [
            "descriptive temporal multiscale decomposition of a [T, D] camera-motion signal",
            "band RMS is an energy-like summary of that band's samples, not a probability and not a "
            "variance-explained of a known ground-truth motion",
            "no band is depth, uncertainty, a fractal/Hurst estimate, a selected winner or a quality score",
            "no comparison against the L1 path planner is computed, implied or claimed",
        ],
        "scope_note": SCOPE_NOTE,
    }


# --------------------------------------------------------------------------------------
# synthetic inspectability fixture
# --------------------------------------------------------------------------------------

def synthetic_pan_plus_multiband_shake(*, n=300, dt=0.05, jitter=0.35, seed=0,
                                       pan_period=50.0, fast_period=0.3, medium_period=2.5):
    """Deliberate low-frequency pan plus two shake bands on irregular timestamps.

    Returns a dict with ``signal [T, 2]``, strictly increasing ``times`` (seconds, jittered intervals),
    the three known components (``pan``, ``shake_fast``, ``shake_medium``) and the sorted positive ``scales``
    recommended for inspecting them. The components are well separated in time scale, so a correct
    decomposition should put the fast shake in the finest band, the medium shake in the middle band and the
    pan in the trend. This is a deterministic cartoon used only to show exact reconstruction and that each
    band is inspectable; it is not real footage and implies no stabilizer comparison.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if dt <= 0.0 or jitter < 0.0 or jitter >= 1.0:
        raise ValueError("dt must be > 0 and jitter must lie in [0, 1)")

    rng = np.random.default_rng(seed)
    steps = dt * (1.0 + jitter * (2.0 * rng.random(n - 1) - 1.0))          # strictly positive intervals
    times = np.concatenate([[0.0], np.cumsum(steps)])
    t = times - times[0]
    two_pi = 2.0 * np.pi

    pan = np.stack([2.5 * np.sin(two_pi * t / pan_period),
                    1.8 * np.sin(two_pi * t / pan_period + 0.6)], axis=1)
    fast_phase = rng.uniform(0.0, two_pi, size=2)
    fast = np.stack([0.40 * np.sin(two_pi * t / fast_period + fast_phase[0]),
                     0.30 * np.sin(two_pi * t / fast_period + fast_phase[1])], axis=1)
    medium_phase = rng.uniform(0.0, two_pi, size=2)
    medium = np.stack([0.90 * np.sin(two_pi * t / medium_period + medium_phase[0]),
                       0.70 * np.sin(two_pi * t / medium_period + medium_phase[1])], axis=1)

    return {
        "signal": pan + fast + medium,
        "times": times,
        "pan": pan,
        "shake_fast": fast,
        "shake_medium": medium,
        "scales": np.array([0.15, 1.0, 4.0], dtype=np.float64),
    }


__all__ = [
    "SCHEMA",
    "DEFAULT_CHUNK",
    "KERNEL_DOC",
    "SCOPE_NOTE",
    "MultiscaleError",
    "MultiscaleDecomposition",
    "gaussian_lowpass",
    "decompose",
    "path_reconstruct",
    "band_energy_report",
    "synthetic_pan_plus_multiband_shake",
]
