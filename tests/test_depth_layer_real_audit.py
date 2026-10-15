import json

import numpy as np
import pytest

from bench.depth_layer_real_audit import audit_depth_layer_window


def observation(depth, flow, valid=None):
    if valid is None:
        valid = np.ones(depth.shape, bool)
    return {"depth": depth, "flow": flow, "cycle_valid": valid}


def test_identity_has_full_warp_coverage_and_zero_jitter():
    h, w = 8, 10
    obs = observation(np.ones((h, w)), np.zeros((h, w, 2)))
    result = audit_depth_layer_window([obs], smooth_sigma=1)
    pooled = result["policies"]["pooled"]
    assert pooled["available"]
    assert pooled["jitter_px"] == 0
    assert pooled["mean_inbounds_coverage"] == 1
    assert pooled["mean_rendered_valid_coverage"] == 1
    json.dumps(result, allow_nan=False)


def test_translation_path_has_identity_correction_without_smoothing():
    h, w = 9, 11
    flow = np.zeros((h, w, 2), np.float64)
    flow[..., 0] = 1.25
    flow[..., 1] = -0.5
    obs = observation(np.ones((h, w)), flow)
    result = audit_depth_layer_window([obs], smooth_sigma=0)
    pooled = result["policies"]["pooled"]
    assert pooled["available"]
    assert pooled["jitter_px"] == 0
    assert pooled["mean_inbounds_coverage"] == 1
    assert pooled["mean_rendered_valid_coverage"] == 1


def test_smoothed_translation_path_reports_jitter_and_crop():
    h, w = 9, 11
    moving = np.zeros((h, w, 2), np.float64)
    moving[..., 0] = 1.25
    still = np.zeros_like(moving)
    depth = np.ones((h, w))
    result = audit_depth_layer_window([observation(depth, moving), observation(depth, still)],
                                      smooth_sigma=0.8)
    pooled = result["policies"]["pooled"]
    assert pooled["available"]
    assert pooled["jitter_px"] > 0
    assert pooled["mean_inbounds_coverage"] <= 1
    assert pooled["mean_rendered_valid_coverage"] < 1
    assert pooled["mean_rendered_valid_coverage"] <= pooled["mean_inbounds_coverage"]


def test_two_layer_affine_fields_produce_rank_and_pooled_models():
    h, w = 10, 12
    yy, xx = np.mgrid[:h, :w]
    depth = (xx >= w // 2).astype(np.float64)
    flow = np.zeros((h, w, 2), np.float64)
    flow[..., 0] = np.where(xx < w // 2, 1.0, -2.0) + 0.1 * yy
    flow[..., 1] = np.where(xx < w // 2, 0.25, 1.5)
    result = audit_depth_layer_window([observation(depth, flow)], n_bins=2, smooth_sigma=0)
    pair = result["pairs"][0]
    assert pair["pooled"]["available"]
    assert [b["rank_id"] for b in pair["rank_bins"]] == [0, 1]
    assert all(b["available"] for b in pair["rank_bins"])
    assert result["policies"]["relative_depth_rank_0"]["available"]
    assert result["policies"]["relative_depth_rank_1"]["available"]


def test_reversing_depth_reverses_neutral_rank_ids():
    h, w = 8, 10
    yy, xx = np.mgrid[:h, :w]
    depth = (xx >= w // 2).astype(np.float64)
    flow = np.zeros((h, w, 2), np.float64)
    flow[..., 0] = np.where(depth == 0, 1.0, -2.0)
    forward = audit_depth_layer_window([observation(depth, flow)], n_bins=2, smooth_sigma=0)
    reverse = audit_depth_layer_window([observation(1 - depth, flow)], n_bins=2, smooth_sigma=0)
    f = forward["pairs"][0]["rank_bins"]
    r = reverse["pairs"][0]["rank_bins"]
    assert f[0]["median_fit_residual_px"] == pytest.approx(r[1]["median_fit_residual_px"])
    assert f[1]["median_fit_residual_px"] == pytest.approx(r[0]["median_fit_residual_px"])


def test_tied_depth_collapses_to_one_rank_and_empty_rank_path_is_unavailable():
    h, w = 8, 8
    depth = np.ones((h, w))
    flow = np.zeros((h, w, 2))
    result = audit_depth_layer_window([observation(depth, flow)], n_bins=4)
    assert result["rank_ids_per_pair"] == [[0]]
    assert len(result["pairs"][0]["rank_bins"]) == 1
    assert result["policies"]["relative_depth_rank_0"]["available"]
    assert not result["policies"]["relative_depth_rank_1"]["available"]


def test_cycle_mask_and_nonfinite_values_are_excluded_from_fit():
    h, w = 8, 8
    depth = np.ones((h, w))
    flow = np.zeros((h, w, 2))
    valid = np.ones((h, w), bool)
    valid[:4] = False
    flow[4, 0] = np.nan
    valid[4, 0] = True
    result = audit_depth_layer_window([observation(depth, flow, valid)])
    assert result["pairs"][0]["pooled"]["support"] == 31
    assert result["pairs"][0]["pooled"]["available"]


@pytest.mark.parametrize("changes", [
    {"depth": np.ones((8, 8, 1))},
    {"flow": np.zeros((8, 8, 3))},
    {"cycle_valid": np.ones((8, 7), bool)},
    {"cycle_resid": np.ones((8, 7))},
])
def test_malformed_shapes_are_rejected(changes):
    base = observation(np.ones((8, 8)), np.zeros((8, 8, 2)))
    base.update(changes)
    with pytest.raises(ValueError):
        audit_depth_layer_window([base])


def test_invalid_options_and_window_shape_mismatch_are_rejected():
    obs = observation(np.ones((8, 8)), np.zeros((8, 8, 2)))
    with pytest.raises(ValueError):
        audit_depth_layer_window([obs], n_bins=0)
    with pytest.raises(ValueError):
        audit_depth_layer_window([obs], smooth_sigma=float("nan"))
    other = observation(np.ones((9, 8)), np.zeros((9, 8, 2)))
    with pytest.raises(ValueError):
        audit_depth_layer_window([obs, other])


def test_repeated_runs_are_deterministic():
    h, w = 8, 10
    yy, xx = np.mgrid[:h, :w]
    depth = (xx >= w // 2).astype(np.float64)
    flow = np.stack((0.2 * xx + depth, -0.1 * yy + 2 * depth), axis=-1)
    obs = observation(depth, flow)
    a = audit_depth_layer_window([obs, obs], n_bins=2, smooth_sigma=1.25)
    b = audit_depth_layer_window([obs, obs], n_bins=2, smooth_sigma=1.25)
    assert a == b
