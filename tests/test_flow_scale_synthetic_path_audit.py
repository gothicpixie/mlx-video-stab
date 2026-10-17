import numpy as np

from bench.flow_scale_synthetic_path_audit import _scene_map, summarize_errors


def test_scene_map_matches_renderer_translation_convention():
    width, height = 100, 80
    points = np.array([[0.0, 0.0], [50.0, 40.0], [99.0, 79.0]])
    actual = np.tile([7.0, -3.0, 0.0], (len(points), 1))
    intended = np.tile([2.0, 5.0, 0.0], (len(points), 1))
    source_points = points + (intended[0, :2] - actual[0, :2])
    actual_world = _scene_map(source_points, actual, width, height)
    intended_world = _scene_map(points, intended, width, height)
    assert np.allclose(actual_world, intended_world)


def test_path_summary_separates_global_bias_from_temporal_residual():
    errors = np.zeros((20, 8, 2), dtype=np.float64)
    errors[..., 0] = 3.0
    errors[..., 1] = -4.0
    valid = np.ones((20, 8), dtype=bool)
    result = summarize_errors(errors, valid, np.ones(8, dtype=bool), fps=30.0)
    assert np.allclose(result["global_bias_xy_px"], [3.0, -4.0])
    assert np.isclose(result["total_path_rmse_px"], 5.0)
    assert np.isclose(result["residual_path_rmse_after_global_bias_px"], 0.0)
    assert np.isclose(result["highpass_path_velocity_error_rms_px"], 0.0)


def test_path_summary_reports_linear_drift_separately_from_global_bias():
    errors = np.zeros((10, 4, 2), dtype=np.float64)
    errors[..., 0] = np.arange(10)[:, None] * 0.5
    errors[..., 1] = 2.0
    valid = np.ones((10, 4), dtype=bool)
    result = summarize_errors(errors, valid, np.ones(4, dtype=bool), fps=30.0)
    assert np.allclose(result["linear_drift_xy_px_per_frame"], [0.5, 0.0])
    assert np.allclose(result["linear_drift_xy_px_over_clip"], [4.5, 0.0])
