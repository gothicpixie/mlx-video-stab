import numpy as np

from bench.flow_scale_synthetic_screen import (
    _fixed_effect_slope,
    make_texture,
    rank_correlation,
    translated_frames,
)


def test_texture_is_deterministic_for_seed_and_condition():
    a = make_texture(32, 16, 8, seed=3, blur_sigma=1.5)
    b = make_texture(32, 16, 8, seed=3, blur_sigma=1.5)
    assert a.shape == (16, 48, 3)
    assert a.dtype == np.uint8
    assert np.array_equal(a, b)


def test_translation_generator_has_known_apparent_direction():
    canvas = np.tile(np.arange(30, dtype=np.uint8)[None, :, None], (4, 1, 3))
    frames = list(translated_frames(canvas, width=10, height=4,
                                    frame_count=2, margin=5, shift_x=2))
    # A fixed scene point appears two pixels to the left in the next crop.
    assert np.array_equal(frames[0][:, 2:], frames[1][:, :-2])


def test_rank_correlation_is_undefined_for_constant_signal():
    assert rank_correlation([1, 1, 1], [0, 1, 2]) is None
    assert rank_correlation([1, 2, 3], [0, 1, 2]) == 1.0


def test_fixed_effect_slope_uses_within_texture_displacement_changes():
    rows = []
    for seed, offset in ((0, 1.0), (1, -3.0)):
        for shift in (4, 8, 16):
            rows.append({"seed": seed, "name": "sharp", "shift_x": shift,
                         "delta_scale4_minus_scale2_epe_px": offset + 0.25 * shift})
    result = _fixed_effect_slope(rows, resamples=20, seed=1)
    assert np.isclose(result["px_epe_per_px_displacement"], 0.25)
    assert np.allclose(result["cluster_bootstrap_ci95"], [0.25, 0.25])
