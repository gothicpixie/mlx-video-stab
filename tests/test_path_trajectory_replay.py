import numpy as np

from bench.path_trajectory_replay import run_synthetic


def test_clean_static_replay_has_zero_error_for_all_smoothers():
    result = run_synthetic(frames=120, profile="static", outlier_fraction=0, outlier_amplitude_px=0)
    assert result["raw_error_rmse_px"] < 1e-10
    assert result["results"][0]["input_high_frequency_residual"]["residual_kurtosis_excess"] is None
    for row in result["results"]:
        assert row["metrics_vs_intended_path"]["path_rmse_px"] < 1e-8
        assert row["output_error_ratio_vs_raw_error"] == 0


def test_pair_fit_outliers_are_replayed_against_known_camera_path():
    result = run_synthetic(frames=120, profile="static", outlier_fraction=0.05,
                           outlier_amplitude_px=12, outlier_seed=4)
    assert result["outliers"]["count"] == 6
    assert result["raw_error_rmse_px"] > 1
    modes = {row["mode"]: row for row in result["results"]}
    assert set(modes) == {"l1", "gauss", "butter"}
    assert modes["l1"]["output_error_ratio_vs_raw_error"] < 0.1
    assert modes["l1"]["metrics_vs_intended_path"]["path_rmse_px"] < modes["gauss"]["metrics_vs_intended_path"]["path_rmse_px"]
    assert modes["l1"]["metrics_vs_intended_path"]["path_rmse_px"] < modes["butter"]["metrics_vs_intended_path"]["path_rmse_px"]
    assert all(np.isfinite(row["metrics_vs_intended_path"]["path_rmse_px"]) for row in modes.values())


def test_sinusoidal_reference_is_measured_separately_from_shake():
    result = run_synthetic(frames=480, profile="sinusoid", amplitude_px=20, period_frames=120,
                           jitter_px=0, outlier_fraction=0.02, outlier_amplitude_px=8, outlier_seed=9)
    assert result["raw_error_rmse_px"] > 0
    for row in result["results"]:
        m = row["metrics_vs_intended_path"]
        assert m["path_rmse_px"] >= m["low_frequency_path_rmse_px"]
        assert m["path_rmse_px"] >= m["high_frequency_path_rmse_px"]
