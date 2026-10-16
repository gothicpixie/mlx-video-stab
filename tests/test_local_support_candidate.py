"""Tests for the opt-in, full-resolution localized support candidate."""

import numpy as np
import pytest

from bench.local_support_candidate import (
    TemporalSupportWindow,
    blend_source_maps,
    local_support_alpha,
    local_support_failure,
)


def _identity_map(height, width):
    y, x = np.mgrid[:height, :width]
    return np.stack([x, y], axis=-1).astype(np.float32)


def test_local_backoff_repairs_recoverable_border_and_preserves_distant_map():
    height, width = 12, 16
    global_src = _identity_map(height, width)
    full_src = global_src.copy()
    full_src[..., 0] -= 2.0

    unsupported, unfixable = local_support_failure(full_src, global_src, width, height)
    assert unsupported[:, :2].all()
    assert not unsupported[:, 2:].any()
    assert not unfixable.any()

    alpha = local_support_alpha(unsupported, radius_px=3.0)
    candidate = blend_source_maps(full_src, global_src, alpha)
    _, candidate_unfixable = local_support_failure(candidate, global_src, width, height)
    assert not candidate_unfixable.any()
    assert alpha[:, :2].max() == 0.0
    assert np.array_equal(candidate[:, 8:], full_src[:, 8:])
    assert np.all((candidate[..., 0] >= 0.0) & (candidate[..., 0] <= width - 1))


def test_local_backoff_does_not_claim_to_fix_global_only_invalid_pixels():
    height, width = 5, 8
    global_src = _identity_map(height, width)
    global_src[..., 0] -= 1.0
    full_src = global_src.copy()
    full_src[..., 0] -= 2.0
    unsupported, unfixable = local_support_failure(full_src, global_src, width, height)
    assert unsupported[:, 1:3].all()
    assert not unsupported[:, 0].any()
    assert unfixable[:, 0].all()
    assert not unfixable[:, 1:].any()


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
    unsupported, unfixable = local_support_failure(full_src, global_src, 4, 3)
    assert unsupported[1, 1]
    assert not unfixable.any()
    alpha = np.ones((3, 4), np.float32)
    alpha[1, 1] = 0.0
    candidate = blend_source_maps(full_src, global_src, alpha)
    assert np.isfinite(candidate).all()
    assert np.array_equal(candidate[1, 1], global_src[1, 1])
