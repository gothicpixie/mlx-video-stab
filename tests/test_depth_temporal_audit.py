"""Weight-free tests for the relative-depth temporal-consistency audit.

Focused on the pure-NumPy diagnostic in ``bench/depth_temporal_audit.py``: exact synthetic translation,
identity/affine-rescaled depth, exact feature-grid sampling, finite/degenerate-input behavior and the
threshold sweep. No RAFT weights, no media, no manifests, no production wiring, no changed defaults. Warnings
are escalated to errors so an unclean divide/invalid-cast path fails the test rather than passing silently.
"""
import warnings

import numpy as np
import pytest

from bench import depth_temporal_audit as D
from bench import flow_prediction_real as R


@pytest.fixture(autouse=True)
def no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def _translated_pair(h, w, dx, dy, seed=0):
    """Source, target and constant forward flow for an exact integer translation.

    Fills ``dst[p + flow] = src[p]`` with ``flow = (dx, dy)`` by placement (not ``np.roll``, whose wrapping
    would break the exactness) and marks cells where the mapped point left the grid as ``nan``. Returns
    ``(src, dst, fwd_flat, bwd_flat)`` with the flow fields flattened ``[h*w, 2]``.
    """
    rng = np.random.RandomState(seed)
    src = rng.rand(h, w)
    dst = np.full((h, w), np.nan)
    ys, xs = np.mgrid[0:h, 0:w]
    ty, tx = ys + dy, xs + dx
    inside = (ty >= 0) & (ty < h) & (tx >= 0) & (tx < w)
    dst[ty[inside], tx[inside]] = src[ys[inside], xs[inside]]
    fwd = np.zeros((h * w, 2))
    fwd[:, 0] = dx
    fwd[:, 1] = dy
    return src, dst, fwd, -fwd


# --------------------------------------------------------------------------------------
# depth_to_feature_grid
# --------------------------------------------------------------------------------------

def test_depth_to_feature_grid_exact_sampling_with_padding_and_downscale():
    depth = np.arange(1, 13, dtype=np.float64).reshape(3, 4)  # 3x4, values 1..12
    pad = (1, 2, 3, 4)                                        # top, bottom, left, right
    downscale = 2
    # feature cell (fx, fy) -> index (8*fx - left, 8*fy - top) -> downscale*index + (downscale-1)/2
    grid = D.depth_to_feature_grid(depth, feature_h=1, feature_w=1, pad=pad, downscale=downscale)
    assert grid.shape == (1, 1)
    # cell (0,0): index (0 - 3, 0 - 1) = (-3, -1); sample at 2*(-3) + 0.5 = -5.5, 2*(-1) + 0.5 = -1.5
    # both coordinates are off the 3x4 map, so the padded cell carries no depth evidence and is NaN.
    assert np.isnan(grid[0, 0])

    # A cell whose centre lands exactly on a real pixel: feature_w=1, pad left = 0 -> index 0 -> centre 0.5
    grid2 = D.depth_to_feature_grid(depth, feature_h=1, feature_w=1, pad=(0, 0, 0, 0), downscale=1)
    # downscale 1 -> centre = index = 0 -> depth[0,0]
    assert grid2[0, 0] == pytest.approx(float(depth[0, 0]))

    # A cell landing between two columns with downscale 2: index fx=1 -> pre 8 -> original x = 2*8+0.5 = 16.5
    # Beyond the 4-wide map, so NaN; instead check a wider synthetic.
    depth4 = np.tile(np.arange(20, dtype=np.float64), (3, 1))  # 3x20, col value == column index * 1
    g4 = D.depth_to_feature_grid(depth4, feature_h=1, feature_w=2, pad=(0, 0, 0, 0), downscale=1)
    # centres x = 0 and 8 map exactly to columns 0 and 8
    assert g4[0, 0] == pytest.approx(0.0)
    assert g4[0, 1] == pytest.approx(8.0)


def test_depth_to_feature_grid_padded_cells_are_nan():
    # Centres outside the original image must be NaN (padded RAFT feature cells), never border-clamped.
    depth = np.arange(1, 13, dtype=np.float64).reshape(3, 4)
    # pad left shifts fx=0 to a negative preprocessed index for both downscale settings.
    g1 = D.depth_to_feature_grid(depth, feature_h=4, feature_w=3, pad=(0, 0, 1, 0), downscale=1)
    # fx=0 -> pre x = -1 -> centre -1 (off image) -> NaN; fx=1 -> pre x = 7 -> off a 4-wide image -> NaN.
    assert np.isnan(g1[0, 0]) and np.isnan(g1[0, 1])
    # fy=1..3 -> pre y = 8,16,24 -> all far outside the 3-tall image -> NaN.
    assert np.isnan(g1[1, 0]) and np.isnan(g1[3, 2])


def test_depth_to_feature_grid_downscale_half_pixel_midpoint():
    # With downscale=2 the mapped centre is d*index + 0.5. Choose an index whose centre is a half pixel
    # between two columns, so a linear ramp interpolates exactly to the midpoint value.
    ramp = np.tile(np.arange(10, dtype=np.float64), (2, 1))  # value == column index
    # pad left = 7 -> pre index for fx=1 is 8 - 7 = 1 -> original x = 2*1 + 0.5 = 2.5 -> mean(col2, col3)=2.5
    grid = D.depth_to_feature_grid(ramp, feature_h=2, feature_w=2, pad=(0, 0, 7, 0), downscale=2)
    assert grid[0, 1] == pytest.approx(2.5)
    # fx=0 -> pre -7 -> original -13.5 -> outside the image -> NaN, not clamped to col 0.
    assert np.isnan(grid[0, 0])


def test_depth_to_feature_grid_padded_cells_emit_no_warnings():
    # NaN-ing padded cells must not warn (no invalid-cast / divide / all-NaN-slice warnings).
    depth = np.arange(1, 1 + 80 * 80, dtype=np.float64).reshape(80, 80)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        grid = D.depth_to_feature_grid(depth, feature_h=4, feature_w=3, pad=(1, 1, 1, 1), downscale=1)
    assert grid.shape == (4, 3)
    assert np.isnan(grid).any() and (~np.isnan(grid)).any()


# --------------------------------------------------------------------------------------
# robust affine fit
# --------------------------------------------------------------------------------------

def test_robust_affine_fit_rejects_minority_outliers():
    # Known affine relation y = 2*x + 3 with a small minority of extreme outliers; the robust fit must stay
    # near truth while the full-support residual stats still expose the outliers.
    rng = np.random.RandomState(11)
    x = np.linspace(0.0, 1.0, 200)
    y = 2.0 * x + 3.0
    n_out = 15                                             # 7.5% minority, above the 5% p95 tail
    out_idx = rng.choice(x.size, size=n_out, replace=False)
    y_out = y.copy()
    y_out[out_idx] += 50.0 * np.sign(rng.randn(n_out) + 1e-9)
    fit = D._robust_affine_fit(x, y_out)
    assert fit["scale"] == pytest.approx(2.0, abs=1e-6)
    assert fit["offset"] == pytest.approx(3.0, abs=1e-6)
    # The fitted scale/offset stay near truth, and the final refit trimmed the extreme outliers: the kept
    # count is a strong majority but strictly below the full support.
    assert 0.8 * x.size <= fit["fit_inlier_count"] < x.size

    # OLS on the same data is dragged toward the outliers, confirming the robustness actually matters.
    ols = D._ols_affine(x, y_out)
    assert ols is not None
    assert abs(ols[0] - 2.0) > abs(fit["scale"] - 2.0)

    # Full-support residual statistics still see the outliers that the fit discarded.
    # ``_method_metrics(src, dst)`` fits ``src ~ scale*dst + offset``, so pass src=y_out, dst=x to recover the
    # inverse (scale=2, offset=3) of the applied map while outliers stay in the residual support.
    metrics = D._method_metrics(y_out, x, span=1.0)
    assert metrics["fit_scale"] == pytest.approx(2.0, abs=1e-6)
    assert metrics["fit_offset"] == pytest.approx(3.0, abs=1e-6)
    assert metrics["fit_inlier_count"] is not None and metrics["fit_inlier_count"] < x.size
    assert metrics["p95_abs_residual"] > 1.0               # outliers remain visible


def test_robust_affine_fit_exact_no_outliers_keeps_full_support():
    x = np.linspace(0.0, 1.0, 40)
    y = 0.5 * x - 1.0
    fit = D._robust_affine_fit(x, y)
    assert fit["scale"] == pytest.approx(0.5)
    assert fit["offset"] == pytest.approx(-1.0)
    assert fit["fit_inlier_count"] == x.size


def test_robust_affine_fit_fails_safely_when_unidentifiable():
    fit = D._robust_affine_fit(np.zeros(10), np.zeros(10))
    assert fit == {"scale": None, "offset": None, "fit_inlier_count": None}


def test_depth_to_feature_grid_rejects_invalid_arguments():
    depth = np.zeros((4, 4), np.float64)
    with pytest.raises(ValueError):
        D.depth_to_feature_grid(depth[..., None], 2, 2, (0, 0, 0, 0), 1)   # 3D
    with pytest.raises(ValueError):
        D.depth_to_feature_grid(depth, 0, 2, (0, 0, 0, 0), 1)              # zero feature dim
    with pytest.raises(ValueError):
        D.depth_to_feature_grid(depth, 2, 2, (0, 0, 0), 1)                 # bad pad
    with pytest.raises(ValueError):
        D.depth_to_feature_grid(depth, 2, 2, (0, 0, 0, 0), 0)              # downscale < 1


# --------------------------------------------------------------------------------------
# exact synthetic translation
# --------------------------------------------------------------------------------------

def test_exact_translation_flow_warp_beats_same_index():
    h, w = 24, 28
    dx, dy = -2, 1
    src, dst, fwd, bwd = _translated_pair(h, w, dx, dy, seed=0)
    cyc = R.cycle_residual_fields(fwd, bwd, h, w)      # cycle residual from the real helper
    assert cyc["fwd_valid"].sum() > 0
    # The translation is exactly invertible on the valid interior, so the forward cycle residual is ~0 there.
    assert np.nanmax(cyc["fwd_resid"][cyc["fwd_valid"]]) < 1e-9

    out = D.relative_depth_pair_audit(src, dst, fwd.reshape(h, w, 2), cyc["fwd_resid"],
                                      cyc["fwd_valid"])
    tier = out["tiers"]["all_cycle_valid"]
    flow_med = tier["flow_warp"]["median_normalized_residual"]
    same_med = tier["same_index"]["median_normalized_residual"]
    assert flow_med is not None and same_med is not None
    assert flow_med < 1e-9
    assert same_med > 10.0 * flow_med
    assert same_med > 0.05


def test_exact_translation_over_textured_random_depth_map():
    h, w = 40, 48
    dx, dy = 3, -2
    src, dst, fwd, bwd = _translated_pair(h, w, dx, dy, seed=7)
    cyc = R.cycle_residual_fields(fwd, bwd, h, w)
    out = D.relative_depth_pair_audit(src, dst, fwd.reshape(h, w, 2), cyc["fwd_resid"],
                                      cyc["fwd_valid"])
    tier = out["tiers"]["all_cycle_valid"]
    assert tier["support_count"] >= 100
    # flow-warp is near zero; the same-index residual is the spread of the translated texture.
    assert tier["flow_warp"]["median_abs_residual"] < 1e-9
    assert tier["same_index"]["median_abs_residual"] > tier["flow_warp"]["median_abs_residual"]


# --------------------------------------------------------------------------------------
# identity / affine rescale and offset
# --------------------------------------------------------------------------------------

def test_identity_motion_both_methods_near_zero():
    rng = np.random.RandomState(1)
    src = rng.rand(12, 15)
    dst = src.copy()
    flow = np.zeros((12, 15, 2))
    zero = np.zeros((12, 15))
    out = D.relative_depth_pair_audit(src, dst, flow, zero, np.ones((12, 15), bool))
    tier = out["tiers"]["all_cycle_valid"]
    assert tier["flow_warp"]["median_normalized_residual"] == pytest.approx(0.0, abs=1e-12)
    assert tier["same_index"]["median_normalized_residual"] == pytest.approx(0.0, abs=1e-12)
    assert tier["flow_warp"]["fit_scale"] == pytest.approx(1.0)
    assert tier["same_index"]["fit_offset"] == pytest.approx(0.0, abs=1e-12)


def test_affine_rescale_and_offset_depth_absorbs_into_fit():
    rng = np.random.RandomState(2)
    src = rng.rand(16, 18)
    # target is an arbitrary affine copy of the source: the per-pair fit must absorb it for both methods.
    dst = 2.5 * src - 4.0
    flow = np.zeros((16, 18, 2))
    zero = np.zeros((16, 18))
    out = D.relative_depth_pair_audit(src, dst, flow, zero, np.ones((16, 18), bool))
    tier = out["tiers"]["all_cycle_valid"]
    assert tier["same_index"]["median_normalized_residual"] == pytest.approx(0.0, abs=1e-12)
    assert tier["flow_warp"]["median_normalized_residual"] == pytest.approx(0.0, abs=1e-12)
    # target -> source fit inverts the applied (2.5, -4.0), i.e. scale 1/2.5 and offset 4/2.5.
    assert tier["same_index"]["fit_scale"] == pytest.approx(1.0 / 2.5)
    assert tier["same_index"]["fit_offset"] == pytest.approx(4.0 / 2.5)


# --------------------------------------------------------------------------------------
# degenerate and invalid inputs
# --------------------------------------------------------------------------------------

def test_nonfinite_and_out_of_bounds_cells_are_excluded():
    h, w = 8, 9
    src = np.arange(h * w, dtype=np.float64).reshape(h, w)
    dst = src.copy()
    flow = np.zeros((h, w, 2))
    flow[0, 0, 0] = np.nan                     # non-finite flow -> excluded
    flow[1, 1, 0] = 1000.0                     # out-of-grid mapping -> excluded
    dst[2, 2] = np.nan                         # non-finite depth -> excluded
    valid = np.ones((h, w), bool)
    valid[3, 3] = False                        # false cycle-valid -> excluded
    out = D.relative_depth_pair_audit(src, dst, flow, np.zeros((h, w)), valid)
    assert out["base_support_count"] == h * w - 4


def test_insufficient_support_returns_explicit_null_metrics():
    # Support reduced to two cells: below MIN_SAMPLES, so all method metrics must be explicit nulls.
    h, w = 6, 7
    src = np.arange(h * w, dtype=np.float64).reshape(h, w)
    dst = src.copy()
    flow = np.zeros((h, w, 2))
    valid = np.zeros((h, w), bool)
    valid.flat[:2] = True
    out = D.relative_depth_pair_audit(src, dst, flow, np.zeros((h, w)), valid)
    tier = out["tiers"]["all_cycle_valid"]
    assert tier["support_count"] == 2
    for method in ("same_index", "flow_warp"):
        m = tier[method]
        assert m["median_abs_residual"] is None
        assert m["median_normalized_residual"] is None
        assert m["fit_scale"] is None
        assert m["reason"] == "insufficient_support"


def test_zero_depth_span_returns_explicit_null_metrics():
    # Constant depth with identity motion: the source span is zero (no normalizer) and the constant target
    # has no span, so the affine fit is unidentifiable. The audit must report explicit nulls with a reason
    # rather than nan/inf or a silent zero.
    src = np.full((10, 10), 3.0)
    dst = np.full((10, 10), 3.0)
    flow = np.zeros((10, 10, 2))
    out = D.relative_depth_pair_audit(src, dst, flow, np.zeros((10, 10)), np.ones((10, 10), bool))
    assert out["source_span"] is None
    tier = out["tiers"]["all_cycle_valid"]
    assert tier["same_index"]["median_abs_residual"] is None
    assert tier["same_index"]["median_normalized_residual"] is None
    assert tier["same_index"]["fit_scale"] is None
    assert tier["same_index"]["reason"] == "unidentifiable_fit"


def test_flat_target_depth_gives_unidentifiable_fit():
    # Source varies but the target is constant -> the target-to-source fit has zero target span.
    rng = np.random.RandomState(4)
    src = rng.rand(9, 9)
    dst = np.full((9, 9), 0.5)
    flow = np.zeros((9, 9, 2))
    out = D.relative_depth_pair_audit(src, dst, flow, np.zeros((9, 9)), np.ones((9, 9), bool))
    tier = out["tiers"]["all_cycle_valid"]
    assert tier["flow_warp"]["reason"] == "unidentifiable_fit"
    assert tier["flow_warp"]["fit_scale"] is None


def test_shape_mismatches_raise_value_error():
    src = np.zeros((5, 6))
    dst = np.zeros((5, 6))
    flow = np.zeros((5, 6, 2))
    resid = np.zeros((5, 6))
    valid = np.ones((5, 6), bool)
    with pytest.raises(ValueError):
        D.relative_depth_pair_audit(src[..., None], dst, flow, resid, valid)
    with pytest.raises(ValueError):
        D.relative_depth_pair_audit(src, np.zeros((5, 7)), flow, resid, valid)
    with pytest.raises(ValueError):
        D.relative_depth_pair_audit(src, dst, np.zeros((5, 6, 3)), resid, valid)
    with pytest.raises(ValueError):
        D.relative_depth_pair_audit(src, dst, flow, np.zeros((5, 7)), valid)
    with pytest.raises(ValueError):
        D.relative_depth_pair_audit(src, dst, flow, resid, np.ones((5, 7), bool))


# --------------------------------------------------------------------------------------
# threshold sweep
# --------------------------------------------------------------------------------------

def test_threshold_sweep_support_is_monotone_in_the_threshold():
    rng = np.random.RandomState(5)
    h, w = 20, 22
    src, dst, fwd, bwd = _translated_pair(h, w, -1, 1, seed=5)
    cyc = R.cycle_residual_fields(fwd, bwd, h, w)
    # Perturb the cycle residual so the thresholds actually split the support.
    resid = cyc["fwd_resid"] + rng.rand(h, w) * 0.2
    thresholds = (0.01, 0.05, 0.1, 0.2)
    out = D.relative_depth_pair_audit(src, dst, fwd.reshape(h, w, 2), resid, cyc["fwd_valid"],
                                      cycle_thresholds=thresholds)
    keys = ["all_cycle_valid"] + [f"cycle_le_{t:g}" for t in thresholds]
    counts = [out["tiers"][k]["support_count"] for k in keys]
    # Thresholds are ascending, so widening the gate can only add support: the counts grow monotonically
    # (equivalently shrink monotonically towards the strict end). No threshold is selected.
    assert counts[1:] == sorted(counts[1:])
    assert counts[0] == out["base_support_count"] == counts[-1]
    for k in keys[1:]:
        tier = out["tiers"][k]
        # Coverage is reported per tier, never merged or selected against.
        assert 0.0 <= tier["support_fraction"] <= out["tiers"]["all_cycle_valid"]["support_fraction"]
        assert tier["cycle_threshold"] is not None
    assert out["tiers"]["all_cycle_valid"]["cycle_threshold"] is None


def test_notes_state_scale_is_discarded_and_filters_are_diagnostic():
    src = np.ones((6, 6))
    flow = np.zeros((6, 6, 2))
    out = D.relative_depth_pair_audit(src, src, flow, np.zeros((6, 6)), np.ones((6, 6), bool))
    joined = " ".join(out["notes"]).lower()
    assert "scale" in joined and "diagnostic" in joined


def test_cycle_thresholds_preserve_caller_order():
    src, dst, fwd, bwd = _translated_pair(10, 10, 0, 0, seed=9)
    cyc = R.cycle_residual_fields(fwd, bwd, 10, 10)
    # Intentionally non-sorted valid thresholds: keys must follow caller order.
    thresholds = (0.2, 0.05, 0.1)
    out = D.relative_depth_pair_audit(src, dst, fwd.reshape(10, 10, 2), cyc["fwd_resid"],
                                      cyc["fwd_valid"], cycle_thresholds=thresholds)
    keys = list(out["tiers"].keys())
    assert keys == ["all_cycle_valid"] + [f"cycle_le_{t:g}" for t in thresholds]


def test_cycle_thresholds_reject_invalid_values():
    src = np.zeros((6, 6))
    flow = np.zeros((6, 6, 2))
    resid = np.zeros((6, 6))
    valid = np.ones((6, 6), bool)

    def run(thresholds):
        return D.relative_depth_pair_audit(src, src, flow, resid, valid, cycle_thresholds=thresholds)

    with pytest.raises(ValueError):
        run((np.nan,))
    with pytest.raises(ValueError):
        run((np.inf,))
    with pytest.raises(ValueError):
        run((-0.1,))
    # Duplicate keys (same ``:g`` formatting) would silently overwrite an earlier tier, so reject them.
    with pytest.raises(ValueError):
        run((0.1, 0.1))
    # Distinct values that format identically to the same key are also duplicates.
    with pytest.raises(ValueError):
        run((0.1, 0.10000000001))
