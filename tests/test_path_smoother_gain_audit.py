import numpy as np

from bench.path_smoother_gain_audit import run_case


def test_exact_static_camera_has_no_path_drift():
    result = run_case(width=640, height=360, frames=30, profile="static", pan_px_per_frame=0, jitter_px=0,
                      mode="l1", fidelity=0.1)
    metrics = result["metrics"]
    assert abs(metrics["static_output_drift_px_per_frame"]) < 1e-10
    assert metrics["output_path_rmse_px"] < 1e-10


def test_no_smoothing_preserves_both_pan_directions():
    for pan in (0.5, -0.5):
        result = run_case(width=640, height=360, frames=30, pan_px_per_frame=pan, jitter_px=0,
                          mode="gauss", sigma=0)
        assert np.isclose(result["metrics"]["path_gain"], 1.0, atol=1e-9)
        assert result["metrics"]["output_path_rmse_px"] < 1e-9


def test_default_l1_smoother_suppresses_exact_linear_pan():
    result = run_case(width=1920, height=1080, frames=60, pan_px_per_frame=0.5, jitter_px=0,
                      mode="l1", sigma=15, fidelity=0.1, budget_fraction=0.04)
    assert abs(result["metrics"]["path_gain"]) < 1e-8
    assert np.isclose(result["metrics"]["output_endpoint_x_px"], 0.0, atol=1e-8)


def test_no_smoothing_preserves_sinusoid_amplitude_and_phase():
    result = run_case(width=640, height=360, frames=120, profile="sinusoid", amplitude_px=20,
                      period_frames=30, jitter_px=0, mode="gauss", sigma=0)
    metrics = result["metrics"]
    assert np.isclose(metrics["sinusoid_amplitude_gain"], 1.0, atol=1e-8)
    assert abs(metrics["sinusoid_phase_lag_frames"]) < 1e-8
    assert metrics["sinusoid_fit_rmse_px"] < 1e-8


def test_phase_is_unreported_when_l1_removes_sinusoid():
    result = run_case(width=640, height=360, frames=120, profile="sinusoid", amplitude_px=20,
                      period_frames=30, jitter_px=0, mode="l1", fidelity=0.1)
    metrics = result["metrics"]
    assert metrics["sinusoid_amplitude_gain"] < 1e-8
    assert metrics["sinusoid_phase_identifiable"] is False
    assert metrics["sinusoid_phase_lag_frames"] is None
