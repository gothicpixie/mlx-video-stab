"""Tests for the research-only E10.1 rendered-pixel audit (``bench/multiscale_render_audit.py``).

Exercised on the fixed synthetic fixture only: pure NumPy, no MLX, no media, no production wiring and no
changed defaults. Coverage: the bilinear sampler (identity, integer shift, subpixel sign, constant, and sampling a larger image
into a smaller independent output grid with batched coordinates), the validity mask bounds/coverage and gather
clamping for out-of-bounds coordinates, the explicit camera/warp sign convention (``P=C`` reproduces the observed
frame; with ``C=pan`` and ``P=pan`` the candidate equals the ideal view), rendered metric recomputation and
the known zero-error identity case, crop pairing via the existing nearest-index rule, coverage/error on the
common valid support only, strict JSON safety and path-freedom, determinism, the absence of any ranking/winner
field, malformed-input rejection, and the CLI (stdout, ``--out`` and JSON error exits).
"""
import json
import math

import numpy as np
import pytest

from bench import multiscale_motion_audit as A
from bench import multiscale_render_audit as R


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _all_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _all_keys(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _all_keys(v)


def _all_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _all_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _all_strings(v)


def _assert_json_native(obj):
    if obj is None or isinstance(obj, (str, bool, int)):
        return
    if isinstance(obj, float):
        assert math.isfinite(obj), f"non-finite float in report: {obj!r}"
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert isinstance(k, str), f"non-string key {k!r}"
            _assert_json_native(v)
        return
    if isinstance(obj, list):
        for v in obj:
            _assert_json_native(v)
        return
    raise AssertionError(f"non-JSON-native type {type(obj)!r}: {obj!r}")


def _assert_pixel_close(got, want):
    assert set(got) == set(want)
    for key, value in want.items():
        if value is None:
            assert got[key] is None, key
        else:
            assert got[key] == pytest.approx(value, rel=1e-9, abs=1e-12), key


# --------------------------------------------------------------------------------------
# bilinear sampler
# --------------------------------------------------------------------------------------

def test_bilinear_sample_identity_integer_shift_and_constant():
    rng = np.random.default_rng(0)
    img = rng.random((1, 5, 6, 3))

    # Identity: integer coordinates on the grid reproduce the image exactly.
    ys, xs = np.mgrid[0:5, 0:6]
    coords = np.stack([xs, ys], axis=-1)[None].astype(np.float64)
    out, valid = R.bilinear_sample(img, coords)
    assert valid.all()
    np.testing.assert_allclose(out, img, rtol=0, atol=0)

    # Integer translation (+2 in x, +1 in y) reproduces the shifted sub-image exactly.
    ys, xs = np.mgrid[0:3, 0:4]
    coords = np.stack([xs + 2, ys + 1], axis=-1)[None].astype(np.float64)
    out, valid = R.bilinear_sample(img, coords)
    assert valid.all()
    np.testing.assert_allclose(out[0], img[0, ys + 1, xs + 2], rtol=0, atol=1e-12)

    # A constant image stays exact at any in-bounds fractional coordinate.
    const = np.full((1, 4, 4, 3), 0.37)
    out, valid = R.bilinear_sample(const, np.full((1, 4, 4, 2), 1.5))
    assert valid.all()
    np.testing.assert_allclose(out, 0.37, rtol=0, atol=1e-12)


def test_bilinear_sample_subpixel_sign_is_positive():
    # A horizontal ramp sampled at x = 1.5 must read 1.5, not 2.5 or 0.5.
    img = np.zeros((1, 3, 5, 1))
    img[0, :, :, 0] = np.arange(5, dtype=np.float64)[None, :]
    coords = np.zeros((1, 3, 5, 2))
    coords[..., 0] = 1.5
    coords[..., 1] = np.arange(3, dtype=np.float64)[:, None]
    out, valid = R.bilinear_sample(img, coords)
    assert valid.all()
    np.testing.assert_allclose(out[..., 0], 1.5, rtol=0, atol=1e-12)


def test_bilinear_sample_validity_mask_bounds_and_zero_fill():
    img = np.zeros((1, 4, 5, 1))
    coords = np.array([[
        [[-0.5, 0.0], [0.0, 0.0], [4.0, 0.0], [4.5, 0.0]],
        [[0.0, -0.5], [0.0, 0.0], [0.0, 3.0], [0.0, 3.5]],
    ]], dtype=np.float64)
    out, valid = R.bilinear_sample(img, coords)
    expected = np.array([[
        [False, True, True, False],
        [False, True, True, False],
    ]])
    np.testing.assert_array_equal(valid, expected)
    # Invalid destinations sample as exactly zero and never contribute to a metric.
    assert np.all(out[~valid] == 0.0)


def test_bilinear_sample_samples_larger_image_into_smaller_output_grid():
    # Only the batch axis must match: the coordinate spatial axes may be smaller than the source H/W, so a
    # larger finite scene can be rendered into smaller output frames.
    rng = np.random.default_rng(11)
    images = rng.random((2, 9, 11, 3))
    ys, xs = np.mgrid[0:5, 0:7]
    coords = np.stack([xs + 1.0, ys + 2.0], axis=-1)[None].astype(np.float64)
    coords = np.repeat(coords, 2, axis=0)                    # [2, 5, 7, 2] from a [2, 9, 11, 3] source
    out, valid = R.bilinear_sample(images, coords)
    assert out.shape == (2, 5, 7, 3)
    assert valid.shape == (2, 5, 7)
    assert valid.all()
    np.testing.assert_allclose(out, images[:, ys + 2, xs + 1], rtol=0, atol=1e-12)


def test_bilinear_sample_rejects_malformed_inputs():
    img = np.zeros((1, 4, 5, 1))
    with pytest.raises(ValueError):
        R.bilinear_sample(np.zeros((4, 5, 1)), np.zeros((1, 4, 5, 2)))          # wrong images ndim
    with pytest.raises(ValueError):
        R.bilinear_sample(img, np.zeros((4, 5, 2)))                             # wrong coords ndim
    with pytest.raises(ValueError):
        R.bilinear_sample(img, np.zeros((1, 4, 5, 3)))                          # coords last axis not 2
    with pytest.raises(ValueError):
        R.bilinear_sample(img, np.zeros((2, 4, 5, 2)))                          # batch axis mismatch
    with pytest.raises(ValueError):
        R.bilinear_sample(img, np.full((1, 4, 5, 2), np.nan))                   # non-finite coords
    with pytest.raises(ValueError):
        R.bilinear_sample(np.full((1, 4, 5, 1), np.inf), np.zeros((1, 4, 5, 2)))


# --------------------------------------------------------------------------------------
# camera / warp sign convention
# --------------------------------------------------------------------------------------

def test_p_equals_c_reproduces_the_observed_frame():
    scene = R.make_scene(24, 28, seed=3)
    camera = np.array([[0.3, -0.2], [1.1, 0.5], [-0.7, 0.9]])
    observed, _ = R.sample_scene(scene, camera, width=8, height=6, margin=8)
    candidate, valid = R.backward_warp(observed, camera - camera)
    assert valid.all()
    np.testing.assert_allclose(candidate, observed, rtol=0, atol=1e-12)


def test_c_equals_pan_and_p_equals_pan_gives_the_ideal_view():
    scene = R.make_scene(24, 28, seed=4)
    pan = np.array([[0.4, 0.1], [-0.2, 0.7], [1.3, -0.5]])
    observed, _ = R.sample_scene(scene, pan, width=8, height=6, margin=8)
    ideal, _ = R.sample_scene(scene, pan, width=8, height=6, margin=8)
    # With C = pan the observed frame is the ideal frame.
    np.testing.assert_allclose(observed, ideal, rtol=0, atol=1e-12)
    # Then P = pan (so P - C = 0) reproduces the ideal view.
    candidate, valid = R.backward_warp(observed, pan - pan)
    assert valid.all()
    np.testing.assert_allclose(candidate, ideal, rtol=0, atol=1e-12)


def test_backward_warp_integer_delta_shifts_by_exactly_that_delta():
    scene = R.make_scene(22, 24, seed=5)
    camera = np.zeros((3, 2))
    observed, _ = R.sample_scene(scene, camera, width=8, height=6, margin=8)
    delta = np.tile(np.array([1.0, 0.0]), (3, 1))
    candidate, valid = R.backward_warp(observed, delta)
    # x + 1 is valid only for x <= width - 2; the last column is out of bounds.
    assert valid[:, :, :7].all()
    assert not valid[:, :, 7].any()
    np.testing.assert_allclose(candidate[:, :, :7], observed[:, :, 1:], rtol=0, atol=1e-12)
    assert np.all(candidate[:, :, 7] == 0.0)


def test_backward_warp_rejects_bad_shapes():
    observed = np.zeros((3, 6, 8, 3))
    with pytest.raises(ValueError):
        R.backward_warp(observed, np.zeros((2, 2)))          # T mismatch
    with pytest.raises(ValueError):
        R.backward_warp(observed, np.zeros((3, 3)))          # not [T, 2]
    with pytest.raises(ValueError):
        R.backward_warp(observed, np.full((3, 2), np.nan))


def test_sample_scene_rejects_scene_too_small():
    scene = R.make_scene(10, 10, seed=0)
    with pytest.raises(ValueError):
        R.sample_scene(scene, np.zeros((2, 2)), width=8, height=6, margin=8)


def test_sample_scene_renders_smaller_grid_from_larger_scene():
    # A scene larger than the requested output grid: the render must produce [T, h, w, C] frames, mark every
    # pixel valid (the margin covers the motion) and value the pixels by the exact bilinear read of the scene.
    height, width, margin = 6, 8, 4
    scene_h, scene_w = height + 2 * margin, width + 2 * margin
    yy, xx = np.mgrid[0:scene_h, 0:scene_w]
    # A linear field is reproduced exactly by bilinear interpolation, so the expected values are analytic.
    scene = np.stack([xx + 2.0 * yy, 3.0 * xx, 0.5 * yy], axis=-1)
    path = np.array([[0.5, -0.25], [1.25, 0.75], [-0.75, 0.5]])

    frames, valid = R.sample_scene(scene, path, width=width, height=height, margin=margin)
    assert frames.shape == (3, height, width, 3)
    assert valid.shape == (3, height, width)
    assert valid.all()

    ys, xs = np.mgrid[0:height, 0:width]
    for t in range(path.shape[0]):
        sx = xs + margin + path[t, 0]
        sy = ys + margin + path[t, 1]
        # Every contributing tap is inside the source image, so the gather indices are in bounds.
        assert np.all(sx >= 0.0) and np.all(sx <= scene_w - 1)
        assert np.all(sy >= 0.0) and np.all(sy <= scene_h - 1)
        np.testing.assert_allclose(frames[t, ..., 0], sx + 2.0 * sy, rtol=0, atol=1e-12)
        np.testing.assert_allclose(frames[t, ..., 1], 3.0 * sx, rtol=0, atol=1e-12)
        np.testing.assert_allclose(frames[t, ..., 2], 0.5 * sy, rtol=0, atol=1e-12)


def test_bilinear_sample_clamps_gathers_far_outside_a_smaller_grid():
    # Output grid larger than the source: far out-of-bounds coordinates must be flagged invalid and zero-filled
    # without ever raising an index error, for both very large positive and very negative coordinates.
    images = np.arange(1 * 4 * 5 * 1, dtype=np.float64).reshape(1, 4, 5, 1)
    coords = np.empty((1, 3, 3, 2))
    coords[0, 0] = [1e6, 1e6]
    coords[0, 1] = [-1e6, -1e6]
    coords[0, 2] = [2.0, 1.0]                       # an actually in-bounds coordinate
    out, valid = R.bilinear_sample(images, coords)
    assert out.shape == (1, 3, 3, 1)
    assert valid.shape == (1, 3, 3)
    assert not valid[0, 0].any() and not valid[0, 1].any()
    assert valid[0, 2].all()
    assert np.all(out[0, 0] == 0.0) and np.all(out[0, 1] == 0.0)
    np.testing.assert_allclose(out[0, 2, :, 0], images[0, 1, 2, 0], rtol=0, atol=1e-12)


# --------------------------------------------------------------------------------------
# rendered pixel metrics
# --------------------------------------------------------------------------------------

def test_rendered_pixel_metrics_ignore_invalid_pixels():
    ideal = np.zeros((1, 2, 2, 3))
    ideal_valid = np.ones((1, 2, 2), dtype=bool)
    candidate = np.ones((1, 2, 2, 3))
    candidate[0, 1, 1] = 1e6                       # huge error at an invalid pixel
    candidate_valid = np.zeros((1, 2, 2), dtype=bool)
    candidate_valid[0, 0, 0] = True

    m = R.rendered_pixel_metrics(candidate, candidate_valid, ideal, ideal_valid)
    assert m["common_valid_pixels"] == 1
    assert m["rgb_mae"] == pytest.approx(1.0)
    assert m["rgb_rms"] == pytest.approx(1.0)
    assert m["candidate_valid_fraction_of_total"] == pytest.approx(1.0 / 4.0)
    assert m["ideal_valid_fraction_of_total"] == pytest.approx(1.0)
    assert m["common_valid_fraction_of_total"] == pytest.approx(1.0 / 4.0)
    assert m["common_valid_fraction_of_ideal_valid"] == pytest.approx(1.0 / 4.0)


def test_rendered_pixel_metrics_empty_common_support_is_none():
    ideal = np.zeros((1, 2, 2, 3))
    ideal_valid = np.ones((1, 2, 2), dtype=bool)
    candidate = np.ones((1, 2, 2, 3))
    candidate_valid = np.zeros((1, 2, 2), dtype=bool)
    m = R.rendered_pixel_metrics(candidate, candidate_valid, ideal, ideal_valid)
    assert m["common_valid_pixels"] == 0
    assert m["rgb_mae"] is None and m["rgb_rms"] is None
    assert m["common_valid_fraction_of_ideal_valid"] == pytest.approx(0.0)


def test_rendered_pixel_metrics_rejects_shape_mismatch():
    frames = np.zeros((1, 2, 2, 3))
    valid = np.ones((1, 2, 2), dtype=bool)
    with pytest.raises(ValueError):
        R.rendered_pixel_metrics(frames, valid, np.zeros((1, 2, 2, 1)), valid)
    with pytest.raises(ValueError):
        R.rendered_pixel_metrics(frames, np.ones((1, 2, 2, 1), dtype=bool), frames, valid)


# --------------------------------------------------------------------------------------
# report: determinism, JSON safety, path-freedom
# --------------------------------------------------------------------------------------

def _small_report(**overrides):
    kwargs = dict(seeds=(0,), conditions=("pan_shake",), n=12, width=16, height=10, margin=8)
    kwargs.update(overrides)
    return R.run_audit(**kwargs)


def test_report_is_deterministic_json_safe_and_path_free():
    a = _small_report()
    b = _small_report()
    assert json.dumps(a, allow_nan=False, sort_keys=True) == json.dumps(b, allow_nan=False, sort_keys=True)
    _assert_json_native(a)
    text = json.dumps(a, allow_nan=False, sort_keys=True)
    assert "NaN" not in text and "Infinity" not in text
    for s in _all_strings(a):
        assert not s.startswith("/")
        assert ".py" not in s
        assert ".json" not in s
        assert ".npz" not in s
        assert "bench/" not in s


def test_run_audit_reports_each_condition_and_seed_separately():
    rep = R.run_audit(seeds=(0, 1), conditions=("static", "pan_shake"), n=12, width=16, height=10, margin=8)
    combos = {(run["condition"], run["seed"]) for run in rep["runs"]}
    assert combos == {("static", 0), ("static", 1), ("pan_shake", 0), ("pan_shake", 1)}
    assert len(rep["runs"]) == 4
    for run in rep["runs"]:
        assert [p["multiscale_name"] for p in run["pairs"]] == [n for n, _ in A.MULTISCALE_CANDIDATES]
        assert len(run["pairs"]) == len(A.MULTISCALE_CANDIDATES) == 4


# --------------------------------------------------------------------------------------
# independent recomputation and known zero-error identity case
# --------------------------------------------------------------------------------------

def test_pair_pixel_metrics_recompute_independently():
    rep = _small_report()
    run = rep["runs"][0]
    fx_fixture = run["fixture"]
    fx = A.synthetic_translation_fixture(n=12, fps=60.0, seed=0, condition="pan_shake")

    scene = R.make_scene(fx_fixture["scene_shape"][0], fx_fixture["scene_shape"][1],
                         seed=fx_fixture["scene_seed"])
    observed, _ = R.sample_scene(scene, fx["observed"], width=16, height=10, margin=8)
    ideal, ideal_valid = R.sample_scene(scene, fx["pan"], width=16, height=10, margin=8)

    for pair in run["pairs"]:
        assert pair["crop_match_tolerance"] == pytest.approx(A.L1_CROP_MATCH_TOLERANCE)

        ms_out = A.build_multiscale_output(fx["observed"], fx["times"], pair["multiscale"]["retention_weights"])
        ms_path = A.translation_path_metrics(fx["pan"], fx["observed"], ms_out)
        assert pair["multiscale"]["path"]["rms_path_deviation"] == pytest.approx(
            ms_path["rms_path_deviation"])
        assert pair["multiscale"]["path"]["max_abs_correction_from_observed"] == pytest.approx(
            ms_path["max_abs_correction_from_observed"])
        ms_frames, ms_valid = R.backward_warp(observed, ms_out - fx["observed"])
        _assert_pixel_close(pair["multiscale"]["pixel"],
                            R.rendered_pixel_metrics(ms_frames, ms_valid, ideal, ideal_valid))

        l1_out = A.build_l1_output(fx["observed"], pair["l1"]["budget"])
        l1_frames, l1_valid = R.backward_warp(observed, l1_out - fx["observed"])
        _assert_pixel_close(pair["l1"]["pixel"],
                            R.rendered_pixel_metrics(l1_frames, l1_valid, ideal, ideal_valid))


def test_static_condition_has_exact_zero_rendered_error():
    rep = R.run_audit(seeds=(0, 1), conditions=("static",), n=12, width=16, height=10, margin=8)
    for run in rep["runs"]:
        assert run["fixture"]["observed_rms"] == pytest.approx(0.0, abs=1e-12)
        assert run["fixture"]["max_abs_observed"] == pytest.approx(0.0, abs=1e-12)
        for pair in run["pairs"]:
            for side in ("multiscale", "l1"):
                pixel = pair[side]["pixel"]
                assert pixel["rgb_mae"] == pytest.approx(0.0, abs=1e-12)
                assert pixel["rgb_rms"] == pytest.approx(0.0, abs=1e-12)
                assert pixel["candidate_valid_fraction_of_total"] == pytest.approx(1.0)
                assert pixel["common_valid_fraction_of_total"] == pytest.approx(1.0)
                assert pixel["common_valid_fraction_of_ideal_valid"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# crop pairing
# --------------------------------------------------------------------------------------

def test_each_pair_uses_the_nearest_measured_crop_proxy():
    rep = _small_report()
    fx = A.synthetic_translation_fixture(n=12, fps=60.0, seed=0, condition="pan_shake")
    crops = [
        A.translation_path_metrics(fx["pan"], fx["observed"], A.build_l1_output(fx["observed"], b))[
            "max_abs_correction_from_observed"]
        for b in A.L1_BUDGETS
    ]
    for pair in rep["runs"][0]["pairs"]:
        target = pair["multiscale_crop_proxy"]
        best = min(range(len(crops)), key=lambda i: (abs(crops[i] - target), i))
        assert pair["l1_name"] == f"budget_{A.L1_BUDGETS[best]:g}"
        assert pair["l1_crop_proxy"] == pytest.approx(crops[best])
        assert pair["abs_crop_mismatch"] == pytest.approx(abs(target - crops[best]))
        assert pair["l1"]["budget"] == pytest.approx(A.L1_BUDGETS[best])


def test_crop_pairing_ties_resolve_to_the_earliest_grid_index():
    assert A.nearest_crop_index(0.5, [0.4, 0.6, 0.6]) == 0
    assert A.nearest_crop_index(0.5, [0.0, 0.4, 0.6]) == 1


# --------------------------------------------------------------------------------------
# scene
# --------------------------------------------------------------------------------------

def test_scene_is_deterministic_and_bounded():
    a = R.make_scene(10, 12, seed=7)
    b = R.make_scene(10, 12, seed=7)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (10, 12, 3)
    assert float(a.min()) >= 0.0
    assert float(a.max()) <= 1.0 + 1e-12
    assert not np.allclose(a, R.make_scene(10, 12, seed=8))


# --------------------------------------------------------------------------------------
# no ranking / winner / score fields
# --------------------------------------------------------------------------------------

def test_report_has_no_ranking_or_winner_fields():
    rep = _small_report()
    banned = ("winner", "rank", "best", "score", "superior", "quality", "aggregate")
    for key in _all_keys(rep):
        low = key.lower()
        assert not any(b in low for b in banned), f"ranking-like key present: {key!r}"
    assert "reporting_policy" in rep
    assert "aggregate winner" in rep["reporting_policy"].lower()


def test_scope_note_states_the_boundaries():
    note = R.SCOPE_NOTE.lower()
    for term in ("single-layer", "translation-only", "bilinear", "not perceptual quality", "not depth",
                 "subject motion", "production renderer"):
        assert term in note
    assert "quality" not in R.RENDERER["sampler"].lower()
    assert "not_included" in R.RENDERER


# --------------------------------------------------------------------------------------
# malformed inputs
# --------------------------------------------------------------------------------------

def test_run_audit_rejects_malformed_parameters():
    with pytest.raises(ValueError):
        R.run_audit(width=0)
    with pytest.raises(ValueError):
        R.run_audit(height=-1)
    with pytest.raises(ValueError):
        R.run_audit(width=1.5)
    with pytest.raises(ValueError):
        R.run_audit(margin=-1)
    with pytest.raises(ValueError):
        R.run_audit(margin=1.5)
    with pytest.raises(ValueError):
        R.run_audit(n=0)
    with pytest.raises(ValueError):
        R.run_audit(n=A.L1_WINDOW + 1)
    with pytest.raises(ValueError):
        R.run_audit(fps=0.0)
    with pytest.raises(ValueError):
        R.run_audit(seeds=())
    with pytest.raises(ValueError):
        R.run_audit(conditions=("bogus",))
    # A margin smaller than the motion would leave holes in the sampled scene.
    with pytest.raises(ValueError):
        R.run_audit(seeds=(0,), conditions=("pan_shake",), n=12, width=16, height=10, margin=1)


def test_make_scene_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        R.make_scene(0, 5)
    with pytest.raises(ValueError):
        R.make_scene(5, -1)
    with pytest.raises(ValueError):
        R.make_scene(5, 5, seed=0.5)


# --------------------------------------------------------------------------------------
# default run and CLI
# --------------------------------------------------------------------------------------

def test_default_run_covers_all_conditions_and_pairs():
    assert R.DEFAULT_N == 60
    assert R.DEFAULT_FPS == 60.0
    assert R.DEFAULT_WIDTH == 80
    assert R.DEFAULT_HEIGHT == 48
    assert R.DEFAULT_MARGIN == 16
    assert R.DEFAULT_CONDITIONS == A.DEFAULT_CONDITIONS
    assert len(R.DEFAULT_CONDITIONS) == 5
    assert R.DEFAULT_SEEDS == (0, 1, 2)

    rep = R.run_audit()
    assert rep["ok"] is True
    assert len(rep["runs"]) == 5 * 3
    assert [run["condition"] for run in rep["runs"]] == [
        c for c in R.DEFAULT_CONDITIONS for _ in R.DEFAULT_SEEDS]
    for run in rep["runs"]:
        assert len(run["pairs"]) == len(A.MULTISCALE_CANDIDATES) == 4
        for pair in run["pairs"]:
            assert set(pair) >= {"multiscale_name", "l1_name", "multiscale", "l1"}
            for side in ("multiscale", "l1"):
                pixel = pair[side]["pixel"]
                assert 0.0 <= pixel["common_valid_fraction_of_total"] <= 1.0
                if pixel["rgb_mae"] is not None:
                    assert math.isfinite(pixel["rgb_mae"]) and pixel["rgb_mae"] >= 0.0
    _assert_json_native(rep)


def test_cli_prints_json_report_to_stdout(capsys):
    assert R.main(["--seeds", "0", "--conditions", "static", "--n", "12",
                   "--width", "16", "--height", "10", "--margin", "8"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == R.SCHEMA and report["ok"] is True
    assert len(report["runs"]) == 1
    assert report["runs"][0]["condition"] == "static"


def test_cli_writes_json_report_to_out_file(tmp_path):
    out = tmp_path / "render_audit.json"
    assert R.main(["--seeds", "0", "--conditions", "pan_shake", "--n", "12",
                   "--width", "16", "--height", "10", "--margin", "8", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema"] == R.SCHEMA and report["ok"] is True
    assert {run["condition"] for run in report["runs"]} == {"pan_shake"}


def test_cli_emits_json_safe_errors(capsys):
    assert R.main(["--seeds", "0", "--conditions", "bogus"]) == 2
    err = json.loads(capsys.readouterr().out)
    assert err["schema"] == R.SCHEMA and err["ok"] is False
    json.dumps(err, allow_nan=False)

    assert R.main(["--seeds", "0", "--conditions", "static", "--width", "0"]) == 2
    err = json.loads(capsys.readouterr().out)
    assert err["ok"] is False
