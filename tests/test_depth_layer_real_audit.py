import json

import numpy as np
import pytest

from bench.depth_layer_real_audit import audit_depth_layer_sweep, audit_depth_layer_window


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


def test_sweep_one_bin_rank_matches_pooled():
    h, w = 8, 10
    yy, xx = np.mgrid[:h, :w]
    depth = (xx >= w // 2).astype(np.float64)
    flow = np.stack((0.2 * xx + depth, -0.1 * yy + 2 * depth), axis=-1)
    result = audit_depth_layer_sweep([observation(depth, flow)], bin_counts=(1,), smooth_sigmas=(1,))
    variant = result["variants"]["bins_1__sigma_1"]
    assert variant["n_bins_requested"] == 1
    assert len(variant["pairs"][0]["rank_bins"]) == 1
    assert variant["pairs"][0]["rank_bins"][0]["rank_id"] == 0
    # With a single bin every supported cell is rank 0, so the rank policy is the pooled fit.
    assert variant["policies"]["relative_depth_rank_0"] == variant["policies"]["pooled"]


def test_zero_sigma_is_exact_identity_correction_even_for_affine_motion():
    h, w = 9, 11
    yy, xx = np.mgrid[:h, :w]
    depth = np.ones((h, w))
    flow = np.stack((0.02 * xx - 0.01 * yy + 0.35,
                     0.01 * xx + 0.01 * yy - 0.6), axis=-1)
    result = audit_depth_layer_sweep([observation(depth, flow)], bin_counts=(1,), smooth_sigmas=(0,))
    pooled = result["variants"]["bins_1__sigma_0"]["policies"]["pooled"]
    assert pooled["jitter_px"] == 0
    assert pooled["mean_inbounds_coverage"] == 1
    assert pooled["mean_rendered_valid_coverage"] == 1


def test_sweep_enumerates_every_combination_without_selecting_one():
    h, w = 8, 8
    obs = observation(np.ones((h, w)), np.zeros((h, w, 2)))
    result = audit_depth_layer_sweep([obs], bin_counts=(1, 2), smooth_sigmas=(0, 0.5, 1))
    assert result["schema"] == "mlxstab.depth_layer_real_audit_sweep/1"
    assert result["bin_counts"] == [1, 2]
    assert result["smooth_sigmas"] == [0.0, 0.5, 1.0]
    assert result["feature_grid"] == [h, w]
    assert result["pair_count"] == 1
    assert set(result["variants"]) == {
        "bins_1__sigma_0", "bins_1__sigma_0.5", "bins_1__sigma_1",
        "bins_2__sigma_0", "bins_2__sigma_0.5", "bins_2__sigma_1"}
    assert "winner" not in result
    for variant in result["variants"].values():
        assert variant["pair_count"] == 1
        assert "policies" in variant and "pairs" in variant
    json.dumps(result, allow_nan=False)


def test_sweep_deduplicates_equivalent_grid_values():
    obs = observation(np.ones((8, 8)), np.zeros((8, 8, 2)))
    result = audit_depth_layer_sweep([obs], bin_counts=(1, 1, np.int64(2)),
                                     smooth_sigmas=(0, 0.0, 1))
    assert result["bin_counts"] == [1, 2]
    assert result["smooth_sigmas"] == [0.0, 1.0]
    assert len(result["variants"]) == 4


def test_sweep_keeps_each_supplied_window_separate():
    h, w = 8, 8
    yy, xx = np.mgrid[:h, :w]
    depth = np.ones((h, w))
    nonaffine = np.stack((np.sin(xx) + 0.1 * yy, np.cos(yy)), axis=-1)
    identity = np.zeros((h, w, 2))
    curved = audit_depth_layer_sweep([observation(depth, nonaffine)], bin_counts=(1,),
                                     smooth_sigmas=(0,))
    flat = audit_depth_layer_sweep([observation(depth, identity)], bin_counts=(1,),
                                   smooth_sigmas=(0,))
    assert curved["variants"]["bins_1__sigma_0"] != flat["variants"]["bins_1__sigma_0"]
    curved_resid = curved["variants"]["bins_1__sigma_0"]["pairs"][0]["pooled"]["median_fit_residual_px"]
    flat_resid = flat["variants"]["bins_1__sigma_0"]["pairs"][0]["pooled"]["median_fit_residual_px"]
    assert curved_resid > 0
    assert flat_resid == 0


@pytest.mark.parametrize("grid", [
    {"bin_counts": ()},
    {"bin_counts": (0,)},
    {"bin_counts": (-1,)},
    {"bin_counts": (1.5,)},
    {"bin_counts": (1, 0)},
    {"bin_counts": 4},
    {"smooth_sigmas": ()},
    {"smooth_sigmas": (-0.1,)},
    {"smooth_sigmas": (float("nan"),)},
    {"smooth_sigmas": (float("inf"),)},
])
def test_sweep_rejects_invalid_grids(grid):
    obs = observation(np.ones((8, 8)), np.zeros((8, 8, 2)))
    with pytest.raises(ValueError):
        audit_depth_layer_sweep([obs], **grid)


def test_sweep_propagates_window_validation():
    with pytest.raises(ValueError):
        audit_depth_layer_sweep([])
