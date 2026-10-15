"""Focused NumPy-only tests for the Stage 4 multiscale decomposition (``bench/multiscale_motion.py``).

Analytic scope: no media, no MLX, no production wiring, no changed defaults, and no comparison against the
production L1 path planner. The module under test imports only NumPy, so these tests need no shared heavy
lock. They check the timestamp-aware normalized Gaussian (against an independent brute-force sum), exact
reconstruction, the retention-weight contract, bounded-chunk invariance, validation errors and the synthetic
pan-plus-multiband-shake inspectability fixture.
"""
import numpy as np
import pytest

from bench import multiscale_motion as MM


def _valid():
    """A small well-formed input reused by the validation tests."""
    signal = np.arange(8, dtype=np.float64).reshape(4, 2)
    times = np.array([0.0, 0.1, 0.25, 0.5])
    scales = np.array([0.2, 1.0])
    return signal, times, scales


# --------------------------------------------------------------------------------------
# normalized Gaussian low-pass on timestamp differences
# --------------------------------------------------------------------------------------

def test_lowpass_preserves_constant():
    times = np.array([0.0, 0.037, 0.11, 0.12, 0.4])
    signal = np.full((5, 3), 2.75)
    out = MM.gaussian_lowpass(signal, times, 0.19)
    np.testing.assert_allclose(out, signal, atol=1e-12)


def test_lowpass_matches_bruteforce_on_irregular_times():
    rng = np.random.default_rng(3)
    times = np.cumsum(rng.uniform(0.01, 0.2, size=7))
    signal = rng.standard_normal((7, 2))
    scale = 0.13
    got = MM.gaussian_lowpass(signal, times, scale)

    expected = np.empty_like(signal)
    for i in range(times.size):
        w = np.exp(-0.5 * ((times[i] - times) / scale) ** 2)
        w = w / w.sum()
        expected[i] = w @ signal
    np.testing.assert_allclose(got, expected, atol=1e-12, rtol=0)


def test_lowpass_is_timestamp_based_not_frame_indexed():
    # Same scale and signal, but the two time axes have very different intervals.
    signal = np.array([[0.0], [1.0], [0.0], [1.0], [0.0]], dtype=np.float64)
    uniform = np.arange(5, dtype=np.float64) * 0.1
    irregular = np.array([0.0, 0.005, 0.9, 0.905, 1.0], dtype=np.float64)
    a = MM.gaussian_lowpass(signal, uniform, 0.02)
    b = MM.gaussian_lowpass(signal, irregular, 0.02)
    assert np.max(np.abs(a - b)) > 1e-3


def test_lowpass_tiny_scale_is_identity():
    times = np.array([0.0, 0.1, 0.2, 0.3])
    signal = np.array([[1.0, -1.0], [2.0, 0.5], [0.0, 3.0], [-2.0, 1.5]])
    out = MM.gaussian_lowpass(signal, times, 1e-6)
    np.testing.assert_allclose(out, signal, atol=1e-9)


def test_lowpass_single_sample_returns_signal():
    signal = np.array([[4.0, -1.5]])
    out = MM.gaussian_lowpass(signal, np.array([0.5]), 2.0)
    np.testing.assert_array_equal(out, signal)


# --------------------------------------------------------------------------------------
# decomposition, exact reconstruction and chunking
# --------------------------------------------------------------------------------------

def test_decompose_bands_reconstruct_signal():
    rng = np.random.default_rng(11)
    times = np.cumsum(rng.uniform(0.02, 0.09, size=40))
    signal = rng.standard_normal((40, 3))
    decom = MM.decompose(signal, times, [0.1, 0.5, 2.0])
    np.testing.assert_allclose(decom.trend + decom.bands.sum(axis=0), signal, atol=1e-9)
    assert decom.bands.shape == (3, 40, 3)
    assert decom.trend.shape == (40, 3)


def test_decompose_bands_are_successive_lowpass_differences():
    rng = np.random.default_rng(5)
    times = np.cumsum(rng.uniform(0.02, 0.08, size=25))
    signal = rng.standard_normal((25, 2))
    scales = [0.15, 0.6, 2.4]
    decom = MM.decompose(signal, times, scales)
    prev = signal
    for k, s in enumerate(scales):
        cur = MM.gaussian_lowpass(signal, times, s)
        np.testing.assert_allclose(decom.bands[k], prev - cur, atol=1e-12)
        prev = cur
    np.testing.assert_allclose(decom.trend, prev, atol=1e-12)


def test_decompose_chunk_invariance():
    rng = np.random.default_rng(7)
    times = np.cumsum(rng.uniform(0.01, 0.05, size=60))
    signal = rng.standard_normal((60, 2))
    scales = [0.1, 0.4, 1.6]
    a = MM.decompose(signal, times, scales, chunk=1)
    b = MM.decompose(signal, times, scales, chunk=7)
    c = MM.decompose(signal, times, scales, chunk=10_000)
    np.testing.assert_allclose(a.bands, b.bands, atol=1e-10)
    np.testing.assert_allclose(a.bands, c.bands, atol=1e-10)
    np.testing.assert_allclose(a.trend, c.trend, atol=1e-10)


def test_decompose_does_not_alias_input():
    signal, times, scales = _valid()
    decom = MM.decompose(signal, times, scales)
    decom.signal[0, 0] = 123.0
    assert signal[0, 0] != 123.0


# --------------------------------------------------------------------------------------
# retention-weight path reconstruction
# --------------------------------------------------------------------------------------

def test_path_reconstruct_all_ones_returns_original():
    rng = np.random.default_rng(2)
    times = np.cumsum(rng.uniform(0.02, 0.09, size=30))
    signal = rng.standard_normal((30, 2))
    decom = MM.decompose(signal, times, [0.2, 1.0, 4.0])
    np.testing.assert_array_equal(MM.path_reconstruct(decom, None), signal)
    np.testing.assert_array_equal(MM.path_reconstruct(decom, np.ones(3)), signal)


def test_path_reconstruct_applies_one_weight_per_band():
    rng = np.random.default_rng(4)
    times = np.cumsum(rng.uniform(0.02, 0.08, size=20))
    signal = rng.standard_normal((20, 2))
    decom = MM.decompose(signal, times, [0.1, 0.4])
    weights = np.array([0.5, 0.0])
    expected = decom.trend + 0.5 * decom.bands[0]
    np.testing.assert_allclose(MM.path_reconstruct(decom, weights), expected, atol=1e-12)
    # zero weights everywhere retain only the trend
    np.testing.assert_allclose(MM.path_reconstruct(decom, np.zeros(2)), decom.trend, atol=1e-12)


def test_path_reconstruct_removes_a_band():
    rng = np.random.default_rng(9)
    times = np.cumsum(rng.uniform(0.02, 0.08, size=20))
    signal = rng.standard_normal((20, 1))
    decom = MM.decompose(signal, times, [0.1, 0.4])
    with_first = MM.path_reconstruct(decom, np.array([1.0, 0.0]))
    without_first = MM.path_reconstruct(decom, np.array([0.0, 0.0]))
    np.testing.assert_allclose(with_first - without_first, decom.bands[0], atol=1e-12)


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------

def test_report_per_band_rms_matches_independent_recompute():
    rng = np.random.default_rng(13)
    times = np.cumsum(rng.uniform(0.02, 0.07, size=35))
    signal = rng.standard_normal((35, 2))
    scales = [0.15, 0.7, 3.0]
    decom = MM.decompose(signal, times, scales)
    report = MM.band_energy_report(decom)

    assert report["schema"] == MM.SCHEMA
    assert report["n_bands"] == 3
    assert report["scales_seconds"] == pytest.approx(scales)
    assert report["signal"]["n_samples"] == 35
    assert report["signal"]["n_channels"] == 2
    assert report["reconstruction_max_abs_error"] < 1e-9

    for k, entry in enumerate(report["bands"]):
        expected = float(np.sqrt(np.mean(decom.bands[k] ** 2)))
        assert entry["index"] == k
        assert entry["scale_seconds"] == pytest.approx(scales[k])
        assert entry["coarser_scale_seconds"] == pytest.approx(scales[k])
        assert entry["finer_scale_seconds"] == pytest.approx(0.0 if k == 0 else scales[k - 1])
        assert entry["rms"] == pytest.approx(expected)
    assert report["trend"]["scale_seconds"] == pytest.approx(scales[-1])
    # no field is named after a winner, depth, uncertainty or fractal notion
    keys = set(report) | set(report["trend"]) | {k for band in report["bands"] for k in band}
    for banned in ("winner", "depth", "uncertainty", "fractal", "hurst"):
        assert all(banned not in key.lower() for key in keys)


# --------------------------------------------------------------------------------------
# short sequences and edge behavior
# --------------------------------------------------------------------------------------

def test_single_sample_sequence():
    signal = np.array([[3.0, -4.0]])
    decom = MM.decompose(signal, np.array([0.0]), [0.5, 2.0])
    np.testing.assert_array_equal(decom.bands, np.zeros((2, 1, 2)))
    np.testing.assert_array_equal(decom.trend, signal)
    np.testing.assert_array_equal(MM.path_reconstruct(decom, np.ones(2)), signal)


def test_tiny_scale_bands_vanish():
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    signal = np.array([[1.0], [2.0], [3.0], [2.0], [0.0]])
    decom = MM.decompose(signal, times, [1e-6, 2e-6])
    np.testing.assert_allclose(decom.bands, 0.0, atol=1e-9)
    np.testing.assert_allclose(decom.trend, signal, atol=1e-9)
    np.testing.assert_allclose(decom.trend + decom.bands.sum(axis=0), signal, atol=1e-9)


def test_huge_scale_band_is_deviation_from_flat_trend():
    times = np.linspace(0.0, 1.0, 9)
    signal = np.array([[1.0], [-1.0], [2.0], [0.0], [-2.0], [1.0], [3.0], [-1.0], [0.0]])
    decom = MM.decompose(signal, times, [50.0])
    # A very wide kernel makes the trend nearly constant and band_1 the deviation from it.
    trend = decom.trend
    assert float(np.ptp(trend)) < 0.05
    np.testing.assert_allclose(decom.bands[0], signal - trend, atol=1e-12)
    np.testing.assert_allclose(decom.trend + decom.bands.sum(axis=0), signal, atol=1e-9)


# --------------------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------------------

def _code(fn, *args, **kwargs):
    with pytest.raises(MM.MultiscaleError) as info:
        fn(*args, **kwargs)
    return info.value.code


def test_validation_signal_shape():
    _, times, scales = _valid()
    assert _code(MM.decompose, np.zeros(4), times, scales) == "bad_shape"
    assert _code(MM.decompose, np.zeros((4, 2, 1)), times, scales) == "bad_shape"
    assert _code(MM.decompose, np.zeros((0, 2)), np.zeros(0), scales) == "bad_shape"


def test_validation_signal_finiteness():
    _, times, scales = _valid()
    bad = np.arange(8, dtype=np.float64).reshape(4, 2)
    bad[1, 0] = np.nan
    assert _code(MM.decompose, bad, times, scales) == "nonfinite_signal"
    bad[1, 0] = np.inf
    assert _code(MM.decompose, bad, times, scales) == "nonfinite_signal"


def test_validation_times():
    signal, _, scales = _valid()
    assert _code(MM.decompose, signal, np.array([0.0, 0.1, 0.2]), scales) == "bad_times"
    assert _code(MM.decompose, signal, np.zeros((4, 1)), scales) == "bad_times"
    assert _code(MM.decompose, signal, np.array([0.0, 0.1, np.nan, 0.5]), scales) == "bad_times"
    assert _code(MM.decompose, signal, np.array([0.0, 0.1, 0.1, 0.5]), scales) == "nonmonotonic_times"
    assert _code(MM.decompose, signal, np.array([0.0, 0.3, 0.2, 0.5]), scales) == "nonmonotonic_times"


def test_validation_scales():
    signal, times, _ = _valid()
    assert _code(MM.decompose, signal, times, []) == "bad_scales"
    assert _code(MM.decompose, signal, times, [1.0, np.inf]) == "bad_scales"
    assert _code(MM.decompose, signal, times, [1.0, 0.0]) == "nonpositive_scale"
    assert _code(MM.decompose, signal, times, [-1.0, 1.0]) == "nonpositive_scale"
    assert _code(MM.decompose, signal, times, [2.0, 1.0]) == "unsorted_scales"
    assert _code(MM.decompose, signal, times, [1.0, 1.0]) == "unsorted_scales"


def test_validation_weights():
    signal, times, scales = _valid()
    decom = MM.decompose(signal, times, scales)
    assert _code(MM.path_reconstruct, decom, [1.0, 1.0, 1.0]) == "bad_weights"
    assert _code(MM.path_reconstruct, decom, np.ones((2, 1))) == "bad_weights"
    assert _code(MM.path_reconstruct, decom, [np.nan, 0.5]) == "bad_weights"
    assert _code(MM.path_reconstruct, decom, [1.5, 0.5]) == "weights_out_of_range"
    assert _code(MM.path_reconstruct, decom, [-0.1, 0.5]) == "weights_out_of_range"


def test_validation_chunk():
    signal, times, scales = _valid()
    assert _code(MM.decompose, signal, times, scales, chunk=0) == "bad_chunk"
    assert _code(MM.decompose, signal, times, scales, chunk=2.5) == "bad_chunk"
    assert _code(MM.gaussian_lowpass, signal, times, 1.0, chunk=-3) == "bad_chunk"


def test_validation_lowpass_scale():
    signal, times, _ = _valid()
    assert _code(MM.gaussian_lowpass, signal, times, 0.0) == "nonpositive_scale"
    assert _code(MM.gaussian_lowpass, signal, times, -1.0) == "nonpositive_scale"
    assert _code(MM.gaussian_lowpass, signal, times, np.inf) == "nonpositive_scale"


def test_validation_type_errors_for_report_and_reconstruct():
    with pytest.raises(TypeError):
        MM.path_reconstruct(np.zeros((2, 3, 1)))
    with pytest.raises(TypeError):
        MM.band_energy_report(np.zeros((2, 3, 1)))


# --------------------------------------------------------------------------------------
# synthetic counterexample: deliberate pan plus multiband shake
# --------------------------------------------------------------------------------------

def _corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def test_counterexample_reconstructs_and_each_band_is_inspectable():
    # Deliberate low-frequency pan plus a fast and a medium shake band, on irregular timestamps.
    # This only demonstrates exact reconstruction and per-band inspectability. It does NOT compare
    # against the L1 stabilizer and makes no claim that decomposition outperforms it.
    fx = MM.synthetic_pan_plus_multiband_shake(n=300, dt=0.05, jitter=0.35, seed=0)
    signal, times, scales = fx["signal"], fx["times"], fx["scales"]
    assert np.all(np.diff(times) > 0)

    decom = MM.decompose(signal, times, scales)
    np.testing.assert_allclose(decom.trend + decom.bands.sum(axis=0), signal, atol=1e-9)
    np.testing.assert_array_equal(MM.path_reconstruct(decom, np.ones(3)), signal)
    assert np.isfinite(decom.bands).all()

    # Interior slice avoids the one-sided smoothing bias at the clip edges.
    sl = slice(80, signal.shape[0] - 80)
    fast = fx["shake_fast"][sl]
    medium = fx["shake_medium"][sl]
    pan = fx["pan"][sl]
    band_fast = decom.bands[0][sl]
    band_medium = decom.bands[1][sl]
    trend = decom.trend[sl]

    assert _corr(band_fast, fast) > 0.9
    assert _corr(band_medium, medium) > 0.8
    assert _corr(trend, pan) > 0.9
    # each band is dominated by its intended component
    assert _corr(band_fast, fast) > _corr(band_fast, medium) + 0.3
    assert _corr(band_medium, medium) > _corr(band_medium, fast) + 0.3

    report = MM.band_energy_report(decom)
    assert report["reconstruction_max_abs_error"] < 1e-9
    assert report["n_bands"] == 3
