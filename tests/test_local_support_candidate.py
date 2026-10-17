"""Tests for the opt-in, full-resolution localized support candidate."""

import numpy as np
import pytest

from bench.local_support_candidate import (
    TemporalSupportWindow,
    blend_source_maps,
    local_support_alpha,
    local_support_failure,
)
from mlxstab.warpmetrics import full_support_mask


def _identity_map(height, width):
    y, x = np.mgrid[:height, :width]
    return np.stack([x, y], axis=-1).astype(np.float32)


def test_local_backoff_repairs_recoverable_border_and_preserves_distant_map():
    height, width = 12, 16
    global_src = _identity_map(height, width)
    full_src = global_src.copy()
    full_src[..., 0] -= 2.0

    unsupported, both_endpoint_maps_invalid = local_support_failure(
        full_src, global_src, width, height, kernel="bilinear")
    assert unsupported[:, :2].all()
    assert not unsupported[:, 2:].any()
    assert not both_endpoint_maps_invalid.any()

    alpha = local_support_alpha(unsupported, radius_px=3.0)
    candidate = blend_source_maps(full_src, global_src, alpha)
    _, candidate_both_endpoint_maps_invalid = local_support_failure(
        candidate, global_src, width, height, kernel="bilinear")
    assert not candidate_both_endpoint_maps_invalid.any()
    assert alpha[:, :2].max() == 0.0
    assert np.array_equal(candidate[:, 8:], full_src[:, 8:])
    assert np.all((candidate[..., 0] >= 0.0) & (candidate[..., 0] <= width - 1))


def test_local_support_failure_separates_both_endpoint_invalid_pixels():
    height, width = 5, 8
    global_src = _identity_map(height, width)
    global_src[..., 0] -= 1.0
    full_src = global_src.copy()
    full_src[..., 0] -= 2.0
    unsupported, both_endpoint_maps_invalid = local_support_failure(
        full_src, global_src, width, height, kernel="bilinear")
    assert unsupported[:, 1:3].all()
    assert not unsupported[:, 0].any()
    assert both_endpoint_maps_invalid[:, 0].all()
    assert not both_endpoint_maps_invalid[:, 1:].any()


def test_alpha_is_one_without_failures_and_has_a_bounded_spatial_taper():
    bad = np.zeros((9, 11), bool)
    assert np.all(local_support_alpha(bad, 2.0) == 1.0)
    bad[4, 5] = True
    alpha = local_support_alpha(bad, 2.0)
    assert alpha[4, 5] == 0.0
    assert 0.0 < alpha[4, 6] < 1.0
    assert alpha[4, 7] == 1.0
    assert np.all((0.0 <= alpha) & (alpha <= 1.0))


def test_temporal_support_window_uses_centered_context_with_clipped_edges():
    window = TemporalSupportWindow(radius=1)
    masks = []
    outputs = []
    for index in range(4):
        mask = np.zeros((2, 3), bool)
        if index == 1:
            mask[0, 0] = True
        if index == 3:
            mask[1, 2] = True
        masks.append(mask)
        emitted = window.push(mask)
        if emitted is not None:
            outputs.append(emitted)
    outputs.extend(window.flush())

    assert len(outputs) == len(masks)
    for index, (original, union) in enumerate(outputs):
        assert np.array_equal(original, masks[index])
        lo, hi = max(0, index - 1), min(len(masks), index + 2)
        assert np.array_equal(union, np.logical_or.reduce(masks[lo:hi]))


@pytest.mark.parametrize("radius", [-1, 1.5, True])
def test_temporal_support_window_rejects_invalid_radius(radius):
    with pytest.raises(ValueError):
        TemporalSupportWindow(radius)


@pytest.mark.parametrize("radius", [0.0, -1.0, float("nan"), float("inf")])
def test_local_support_alpha_rejects_invalid_radius(radius):
    with pytest.raises(ValueError):
        local_support_alpha(np.zeros((2, 3), bool), radius)


def test_blend_rejects_shape_and_weight_errors():
    src = _identity_map(4, 5)
    with pytest.raises(ValueError):
        blend_source_maps(src, src, np.ones((3, 5), np.float32))
    with pytest.raises(ValueError):
        blend_source_maps(src, src, np.full((4, 5), 1.1, np.float32))
    with pytest.raises(ValueError):
        local_support_failure(src, src[:, :-1], 5, 4)


def test_nonfinite_full_map_is_replaced_only_where_local_alpha_is_zero():
    global_src = _identity_map(3, 4)
    full_src = global_src.copy()
    full_src[1, 1, 0] = np.nan
    unsupported, both_endpoint_maps_invalid = local_support_failure(
        full_src, global_src, 4, 3, kernel="bilinear")
    assert unsupported[1, 1]
    assert not both_endpoint_maps_invalid.any()
    alpha = np.ones((3, 4), np.float32)
    alpha[1, 1] = 0.0
    candidate = blend_source_maps(full_src, global_src, alpha)
    assert np.isfinite(candidate).all()
    assert np.array_equal(candidate[1, 1], global_src[1, 1])


def test_full_support_mask_matches_the_reported_kernel_coverage():
    from mlxstab import warpmetrics

    src = _identity_map(13, 19)
    for kernel in warpmetrics.KERNELS:
        mask = full_support_mask(src, 13, 19, kernel)
        report = warpmetrics.diagnose(src, 13, 19, kernel=kernel)
        assert mask.mean() == pytest.approx(report["coverage"][f"full_support_{kernel}_frac"])
    fractional = src.copy()
    fractional[..., 0] += 0.25
    assert full_support_mask(fractional, 13, 19, "bilinear").sum() > \
        full_support_mask(fractional, 13, 19, "lanczos3").sum()


def test_full_support_uses_fractional_nonzero_taps_and_rejects_nonfinite():
    src = _identity_map(6, 8)
    src[2, 3, 0] = -0.25
    src[3, 4, 1] = np.nan
    mask = full_support_mask(src, 6, 8, "bilinear")
    assert not mask[2, 3]  # center is inside [-.5, W-.5], but the -1 tap carries weight
    assert not mask[3, 4]
