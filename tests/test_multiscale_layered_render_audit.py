"""Tests for the research-only E10.2 layered rendered-pixel audit (``bench/multiscale_layered_render_audit.py``).

Exercised on the fixed synthetic fixture only: pure NumPy, no MLX, no media, no production wiring and no
changed defaults. Coverage: deterministic layer canvases and mask geometry, the layered render sign/identity
convention (``P=C`` reproduces observed; with flat depth and no subject motion an integer camera shift equals
the E10.1 backward warp), depth-dependent parallax and independent subject motion actually altering the
known-truth render, per-region common-support metrics with explicit counts (including a vanishing region),
crop pairing via the existing nearest-index rule, independent recomputation of whole-frame and per-region
metrics, the exact zero-error identity condition, deterministic seeds, strict JSON safety and path-freedom,
the absence of any ranking/winner field, malformed-input rejection, and the CLI.
"""
import json
import math

import numpy as np
import pytest

from bench import multiscale_motion_audit as A
from bench import multiscale_render_audit as R
from bench import multiscale_layered_render_audit as L


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


def _assert_metrics_close(got, want):
    assert set(got) == set(want)
    for key, value in want.items():
        if value is None:
            assert got[key] is None, key
        else:
            assert got[key] == pytest.approx(value, rel=1e-9, abs=1e-12), key


def _small_report(**overrides):
    kwargs = dict(seeds=(0,), conditions=("parallax_shake",), n=12, width=16, height=10, margin=8)
    kwargs.update(overrides)
    return L.run_audit(**kwargs)


# --------------------------------------------------------------------------------------
# layer canvases
# --------------------------------------------------------------------------------------

def test_make_layer_canvases_deterministic_and_masked():
    a = L.make_layer_canvases(16, 10, 8)
    b = L.make_layer_canvases(16, 10, 8)
    assert len(a) == 3
    for x, y in zip(a, b):
        assert x.shape == (10 + 16, 16 + 16, 4)
        np.testing.assert_array_equal(x, y)

    background, foreground, subject = a
    # The background is opaque everywhere.
    np.testing.assert_array_equal(background[..., 3], 1.0)
    # The foreground and subject masks are binary with both classes present, and the subject is in front.
    for layer in (foreground, subject):
        alpha = layer[..., 3]
        assert set(np.unique(alpha).tolist()) <= {0.0, 1.0}
        assert 0 < np.count_nonzero(alpha) < alpha.size
    # Distinct textures per layer.
    assert not np.allclose(background[..., :3], foreground[..., :3])
    assert not np.allclose(foreground[..., :3], subject[..., :3])
    # Subject overlaps part of the foreground so occlusion ordering matters.
    overlap = (foreground[..., 3] > 0.5) & (subject[..., 3] > 0.5)
    assert np.count_nonzero(overlap) > 0

    # Changing the texture seeds changes the canvases.
    other = L.make_layer_canvases(16, 10, 8, seeds=(1, 2, 3))
    assert not np.allclose(other[0], background)


def test_make_layer_canvases_rejects_bad_inputs():
    with pytest.raises(ValueError):
        L.make_layer_canvases(0, 10, 8)
    with pytest.raises(ValueError):
        L.make_layer_canvases(16, -1, 8)
    with pytest.raises(ValueError):
        L.make_layer_canvases(16, 10, -1)
    with pytest.raises(ValueError):
        L.make_layer_canvases(16, 10, 8, seeds=(1, 2))                 # wrong count
    with pytest.raises(ValueError):
        L.make_layer_canvases(16, 10, 8, foreground_rect=(0.5, 0.5, 0.9, 0.2))   # outside [0, 1]
    with pytest.raises(ValueError):
        L.make_layer_canvases(16, 10, 8, subject_rect=(0.5, 0.5, 0.0, 0.2))       # zero half-width
    with pytest.raises(ValueError):
        L.make_layer_canvases(16, 10, 8, subject_rect=(0.5, 0.5, np.nan, 0.2))


# --------------------------------------------------------------------------------------
# subject motion
# --------------------------------------------------------------------------------------

def test_subject_motion_is_deterministic_bounded_and_seed_dependent():
    times = np.arange(20, dtype=np.float64) / 60.0
    m = L.subject_motion_path(times, seed=3)
    np.testing.assert_array_equal(m, L.subject_motion_path(times, seed=3))
    assert m.shape == (20, 2)
    assert np.max(np.abs(m[:, 0])) <= L.SUBJECT_AMPLITUDE[0] + 1e-12
    assert np.max(np.abs(m[:, 1])) <= L.SUBJECT_AMPLITUDE[1] + 1e-12
    assert np.max(np.abs(m)) > 0.0
    assert not np.allclose(m, L.subject_motion_path(times, seed=4))
    with pytest.raises(ValueError):
        L.subject_motion_path(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        L.subject_motion_path(times, period=0.0)
    with pytest.raises(ValueError):
        L.subject_motion_path(times, amplitude=(1.0,))


def test_max_abs_layer_displacement_respects_depth_and_subject_motion():
    camera = np.array([[3.0, 0.0]])
    subject = np.array([[1.5, 0.0]])
    assert L.max_abs_layer_displacement((1.0, 1.0, 1.0), camera, subject) == pytest.approx(4.5)
    assert L.max_abs_layer_displacement((0.5, 1.0, 1.0), camera, subject) == pytest.approx(4.5)
    assert L.max_abs_layer_displacement((0.5, 1.0, 1.0), camera, np.zeros((1, 2))) == pytest.approx(3.0)


# --------------------------------------------------------------------------------------
# layered render sign / identity conventions
# --------------------------------------------------------------------------------------

def _zero_subject(t):
    return np.zeros((t, 2), dtype=np.float64)


def test_render_layered_scene_reference_and_translation_sign():
    canvases = L.make_layer_canvases(16, 10, 8)
    reference, ref_valid, ref_region = L.render_layered_scene(
        canvases, (1.0, 1.0, 1.0), np.zeros((3, 2)), _zero_subject(3), width=16, height=10, margin=8)
    assert ref_valid.all()
    assert set(np.unique(ref_region).tolist()) == {L.BACKGROUND, L.FOREGROUND, L.SUBJECT}

    # With every depth scale 1 and no subject motion, an integer camera path shifts the flat composite.
    delta = np.tile(np.array([2.0, -1.0]), (3, 1))
    observed, observed_valid, _ = L.render_layered_scene(
        canvases, (1.0, 1.0, 1.0), delta, _zero_subject(3), width=16, height=10, margin=8)
    assert observed_valid.all()
    warped, warped_valid = R.backward_warp(reference, delta)
    assert warped_valid.any()
    np.testing.assert_allclose(observed[warped_valid], warped[warped_valid], rtol=0, atol=1e-12)

    # P = C reproduces the observed frame exactly (the general identity convention).
    same, same_valid = R.backward_warp(observed, delta - delta)
    assert same_valid.all()
    np.testing.assert_allclose(same, observed, rtol=0, atol=1e-12)


def test_render_layered_scene_rejects_bad_inputs():
    canvases = L.make_layer_canvases(16, 10, 8)
    with pytest.raises(ValueError):
        L.render_layered_scene(canvases[:2], (1.0, 1.0, 1.0), np.zeros((2, 2)), _zero_subject(2),
                               width=16, height=10, margin=8)
    with pytest.raises(ValueError):
        L.render_layered_scene(canvases, (1.0, 1.0), np.zeros((2, 2)), _zero_subject(2),
                               width=16, height=10, margin=8)
    with pytest.raises(ValueError):
        L.render_layered_scene(canvases, (1.0, 1.0, 1.0), np.zeros((2, 3)), _zero_subject(2),
                               width=16, height=10, margin=8)
    with pytest.raises(ValueError):
        L.render_layered_scene(canvases, (1.0, 1.0, 1.0), np.zeros((2, 2)), np.zeros((3, 2)),
                               width=16, height=10, margin=8)
    with pytest.raises(ValueError):
        L.render_layered_scene(canvases, (1.0, 1.0, 1.0), np.full((2, 2), np.nan), _zero_subject(2),
                               width=16, height=10, margin=8)


# --------------------------------------------------------------------------------------
# parallax and subject motion alter the known-truth render
# --------------------------------------------------------------------------------------

def test_parallax_depth_scales_alter_the_render():
    canvases = L.make_layer_canvases(16, 10, 8)
    camera = np.tile(np.array([3.0, 0.6]), (3, 1))
    flat, flat_valid, _ = L.render_layered_scene(
        canvases, L.FLAT_DEPTH_SCALES, camera, _zero_subject(3), width=16, height=10, margin=8)
    parallax, para_valid, _ = L.render_layered_scene(
        canvases, L.PARALLAX_DEPTH_SCALES, camera, _zero_subject(3), width=16, height=10, margin=8)
    assert flat_valid.all() and para_valid.all()
    assert not np.allclose(flat, parallax)
    # The difference is not only a global translation: the layers moved by different amounts.
    residual = np.abs(flat - parallax).mean(axis=(0, 3))
    assert np.max(residual) - np.min(residual) > 1e-3


def test_subject_motion_alters_the_render():
    canvases = L.make_layer_canvases(16, 10, 8)
    times = np.arange(5, dtype=np.float64) / 60.0
    subject = L.subject_motion_path(times, seed=2)
    still, still_valid, still_region = L.render_layered_scene(
        canvases, L.PARALLAX_DEPTH_SCALES, np.zeros((5, 2)), _zero_subject(5), width=16, height=10, margin=8)
    moving, moving_valid, moving_region = L.render_layered_scene(
        canvases, L.PARALLAX_DEPTH_SCALES, np.zeros((5, 2)), subject, width=16, height=10, margin=8)
    assert still_valid.all() and moving_valid.all()
    assert not np.allclose(still, moving)
    # The subject's own region must change.
    subject_pixels = (still_region == L.SUBJECT) | (moving_region == L.SUBJECT)
    assert np.count_nonzero(subject_pixels) > 0
    assert np.any(np.abs(still - moving)[subject_pixels] > 1e-6)


# --------------------------------------------------------------------------------------
# per-region metrics with counts
# --------------------------------------------------------------------------------------

def test_region_pixel_metrics_recompute_and_counts():
    ideal = np.zeros((1, 2, 3, 3))
    ideal[0, 0, 0] = 2.0                                # region 1 pixel with error 2
    ideal_valid = np.ones((1, 2, 3), dtype=bool)
    candidate = np.zeros((1, 2, 3, 3))
    candidate_valid = np.ones((1, 2, 3), dtype=bool)
    region = np.full((1, 2, 3), L.BACKGROUND, dtype=np.int64)
    region[0, 0, 0] = L.FOREGROUND

    fg = L.region_pixel_metrics(candidate, candidate_valid, ideal, ideal_valid, region, L.FOREGROUND)
    assert fg["region_pixels"] == 1
    assert fg["common_valid_region_pixels"] == 1
    assert fg["rgb_mae"] == pytest.approx(2.0)
    assert fg["rgb_rms"] == pytest.approx(2.0)
    assert fg["common_valid_fraction_of_region"] == pytest.approx(1.0)

    bg = L.region_pixel_metrics(candidate, candidate_valid, ideal, ideal_valid, region, L.BACKGROUND)
    assert bg["region_pixels"] == 5
    assert bg["rgb_mae"] == pytest.approx(0.0)


def test_region_pixel_metrics_expose_a_vanishing_region():
    ideal = np.zeros((1, 2, 2, 3))
    ideal_valid = np.ones((1, 2, 2), dtype=bool)
    candidate = np.ones((1, 2, 2, 3))
    candidate_valid = np.ones((1, 2, 2), dtype=bool)
    candidate_valid[..., 1] = False                     # the whole subject column is uncovered
    region = np.full((1, 2, 2), L.BACKGROUND, dtype=np.int64)
    region[:, :, 1] = L.SUBJECT

    subj = L.region_pixel_metrics(candidate, candidate_valid, ideal, ideal_valid, region, L.SUBJECT)
    assert subj["region_pixels"] == 2                   # the region exists in the ideal render
    assert subj["candidate_valid_region_pixels"] == 0
    assert subj["common_valid_region_pixels"] == 0
    assert subj["common_valid_fraction_of_region"] == pytest.approx(0.0)
    assert subj["rgb_mae"] is None and subj["rgb_rms"] is None

    absent = L.region_pixel_metrics(candidate, candidate_valid, ideal, ideal_valid, region, L.FOREGROUND)
    assert absent["region_pixels"] == 0
    assert absent["common_valid_fraction_of_ideal_valid_region"] == pytest.approx(0.0)
    assert absent["rgb_mae"] is None


def test_region_pixel_metrics_rejects_bad_shapes():
    frames = np.zeros((1, 2, 2, 3))
    valid = np.ones((1, 2, 2), dtype=bool)
    region = np.zeros((1, 2, 2), dtype=np.int64)
    with pytest.raises(ValueError):
        L.region_pixel_metrics(frames, valid, np.zeros((1, 2, 2, 1)), valid, region, 0)
    with pytest.raises(ValueError):
        L.region_pixel_metrics(frames, np.ones((1, 2, 2, 1), dtype=bool), frames, valid, region, 0)
    with pytest.raises(ValueError):
        L.region_pixel_metrics(frames, valid, frames, valid, np.zeros((1, 2), dtype=np.int64), 0)


# --------------------------------------------------------------------------------------
# report: determinism, JSON safety, path-freedom
# --------------------------------------------------------------------------------------

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
    rep = L.run_audit(seeds=(0, 1), conditions=("static_no_parallax", "parallax_shake"),
                     n=12, width=16, height=10, margin=8)
    combos = {(run["condition"], run["seed"]) for run in rep["runs"]}
    assert combos == {("static_no_parallax", 0), ("static_no_parallax", 1),
                      ("parallax_shake", 0), ("parallax_shake", 1)}
    assert len(rep["runs"]) == 4
    for run in rep["runs"]:
        assert [p["multiscale_name"] for p in run["pairs"]] == [n for n, _ in A.MULTISCALE_CANDIDATES]
        assert len(run["pairs"]) == len(A.MULTISCALE_CANDIDATES) == 4
        for pair in run["pairs"]:
            assert set(pair["multiscale"]["regions"]) == set(L.LAYER_NAMES)
            assert set(pair["l1"]["regions"]) == set(L.LAYER_NAMES)


# --------------------------------------------------------------------------------------
# identity control and isolated conditions
# --------------------------------------------------------------------------------------

def test_static_no_parallax_is_exact_identity_for_every_pair():
    rep = L.run_audit(seeds=(0, 1), conditions=("static_no_parallax",), n=12,
                     width=16, height=10, margin=8)
    for run in rep["runs"]:
        fixture = run["fixture"]
        assert fixture["parallax_present"] is False
        assert fixture["subject_motion_present"] is False
        assert fixture["subject_motion_rms"] == pytest.approx(0.0, abs=1e-12)
        assert fixture["max_abs_observed_camera"] == pytest.approx(0.0, abs=1e-12)
        assert fixture["observed_valid_fraction"] == pytest.approx(1.0)
        assert fixture["ideal_region_pixel_counts"]["background"] > 0
        assert fixture["ideal_region_pixel_counts"]["foreground"] > 0
        assert fixture["ideal_region_pixel_counts"]["subject"] > 0
        for pair in run["pairs"]:
            for side in ("multiscale", "l1"):
                whole = pair[side]["whole_frame"]
                assert whole["rgb_mae"] == pytest.approx(0.0, abs=1e-12)
                assert whole["rgb_rms"] == pytest.approx(0.0, abs=1e-12)
                assert whole["common_valid_fraction_of_total"] == pytest.approx(1.0)
                assert whole["common_valid_fraction_of_ideal_valid"] == pytest.approx(1.0)
                for region in pair[side]["regions"].values():
                    assert region["rgb_mae"] == pytest.approx(0.0, abs=1e-12)
                    assert region["common_valid_fraction_of_region"] == pytest.approx(1.0)


def test_conditions_isolate_parallax_and_subject_motion():
    assert L.CONDITION_SPECS["static_no_parallax"]["parallax"] is False
    assert L.CONDITION_SPECS["static_no_parallax"]["depth_scales"] == L.FLAT_DEPTH_SCALES
    assert L.CONDITION_SPECS["parallax_shake"]["parallax"] is True
    assert L.CONDITION_SPECS["parallax_shake"]["subject_motion"] is False
    assert L.CONDITION_SPECS["parallax_subject_jolt"]["parallax"] is True
    assert L.CONDITION_SPECS["parallax_subject_jolt"]["subject_motion"] is True

    rep = L.run_audit(seeds=(0,), conditions=L.CONDITIONS, n=12, width=16, height=10, margin=8)
    by_condition = {run["condition"]: run for run in rep["runs"]}
    assert by_condition["static_no_parallax"]["fixture"]["subject_motion_rms"] == pytest.approx(0.0)
    assert by_condition["parallax_shake"]["fixture"]["subject_motion_rms"] == pytest.approx(0.0)
    assert by_condition["parallax_shake"]["fixture"]["max_abs_observed_camera"] > 0.0
    assert by_condition["parallax_subject_jolt"]["fixture"]["subject_motion_rms"] > 0.0

    # The known-truth render must genuinely differ once parallax/subject motion are present.
    for condition in ("parallax_shake", "parallax_subject_jolt"):
        run = by_condition[condition]
        assert run["fixture"]["parallax_present"] is True
        # keep_all_detail reconstructs the observed camera path, so it renders the observed view exactly.
        identity_pair = next(p for p in run["pairs"] if p["multiscale_name"] == "keep_all_detail")
        assert identity_pair["multiscale"]["whole_frame"]["rgb_mae"] > 0.0
    # The subject region carries a nonzero identity baseline under jolt.
    jolt = by_condition["parallax_subject_jolt"]
    identity_pair = next(p for p in jolt["pairs"] if p["multiscale_name"] == "keep_all_detail")
    assert identity_pair["multiscale"]["regions"]["subject"]["rgb_mae"] > 0.0


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
# independent recomputation
# --------------------------------------------------------------------------------------

def test_pair_whole_frame_and_region_metrics_recompute_independently():
    rep = _small_report()
    run = rep["runs"][0]
    fixture = run["fixture"]
    fx = A.synthetic_translation_fixture(n=12, fps=60.0, seed=0,
                                         condition=L.CONDITION_SPECS["parallax_shake"]["camera_condition"])
    canvases = L.make_layer_canvases(fixture["width_px"], fixture["height_px"], fixture["margin_px"])
    subject = np.zeros((12, 2))
    observed, _, _ = L.render_layered_scene(
        canvases, fixture["depth_scales"], fx["observed"], subject,
        width=fixture["width_px"], height=fixture["height_px"], margin=fixture["margin_px"])
    ideal, ideal_valid, ideal_region = L.render_layered_scene(
        canvases, fixture["depth_scales"], fx["pan"], subject,
        width=fixture["width_px"], height=fixture["height_px"], margin=fixture["margin_px"])

    for pair in run["pairs"]:
        ms_output = A.build_multiscale_output(fx["observed"], fx["times"], pair["multiscale"]["retention_weights"])
        ms_path = A.translation_path_metrics(fx["pan"], fx["observed"], ms_output)
        assert pair["multiscale"]["path"]["rms_path_deviation"] == pytest.approx(
            ms_path["rms_path_deviation"])
        assert pair["multiscale"]["path"]["max_abs_correction_from_observed"] == pytest.approx(
            ms_path["max_abs_correction_from_observed"])
        ms_frames, ms_valid = R.backward_warp(observed, ms_output - fx["observed"])
        _assert_metrics_close(pair["multiscale"]["whole_frame"],
                              R.rendered_pixel_metrics(ms_frames, ms_valid, ideal, ideal_valid))
        for idx, name in enumerate(L.LAYER_NAMES):
            _assert_metrics_close(
                pair["multiscale"]["regions"][name],
                L.region_pixel_metrics(ms_frames, ms_valid, ideal, ideal_valid, ideal_region, idx))

        l1_output = A.build_l1_output(fx["observed"], pair["l1"]["budget"])
        l1_frames, l1_valid = R.backward_warp(observed, l1_output - fx["observed"])
        _assert_metrics_close(pair["l1"]["whole_frame"],
                              R.rendered_pixel_metrics(l1_frames, l1_valid, ideal, ideal_valid))
        for idx, name in enumerate(L.LAYER_NAMES):
            _assert_metrics_close(
                pair["l1"]["regions"][name],
                L.region_pixel_metrics(l1_frames, l1_valid, ideal, ideal_valid, ideal_region, idx))


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


def test_scope_note_and_limitations_state_boundaries():
    note = L.SCOPE_NOTE.lower()
    for term in ("three-layer", "parallax", "subject", "not real footage", "not perceptual quality",
                 "not physical depth", "no superiority claim"):
        assert term in note
    limitations = " ".join(L.LIMITATIONS).lower()
    for term in ("not real footage", "physical depth", "single uniform", "no production"):
        assert term in limitations


# --------------------------------------------------------------------------------------
# malformed inputs
# --------------------------------------------------------------------------------------

def test_run_audit_rejects_malformed_parameters():
    with pytest.raises(ValueError):
        L.run_audit(width=0)
    with pytest.raises(ValueError):
        L.run_audit(height=-1)
    with pytest.raises(ValueError):
        L.run_audit(width=1.5)
    with pytest.raises(ValueError):
        L.run_audit(margin=-1)
    with pytest.raises(ValueError):
        L.run_audit(margin=1.5)
    with pytest.raises(ValueError):
        L.run_audit(n=0)
    with pytest.raises(ValueError):
        L.run_audit(n=A.L1_WINDOW + 1)
    with pytest.raises(ValueError):
        L.run_audit(fps=0.0)
    with pytest.raises(ValueError):
        L.run_audit(seeds=())
    with pytest.raises(ValueError):
        L.run_audit(conditions=("bogus",))
    # A margin smaller than the layer displacement would leave holes in the layered render.
    with pytest.raises(ValueError):
        L.run_audit(seeds=(0,), conditions=("parallax_shake",), n=12, width=16, height=10, margin=1)


# --------------------------------------------------------------------------------------
# default run and CLI
# --------------------------------------------------------------------------------------

def test_default_run_covers_all_conditions_and_pairs():
    assert L.DEFAULT_N == 60
    assert L.DEFAULT_FPS == 60.0
    assert L.DEFAULT_WIDTH == 80
    assert L.DEFAULT_HEIGHT == 48
    assert L.DEFAULT_MARGIN == 20
    assert L.DEFAULT_CONDITIONS == L.CONDITIONS
    assert len(L.DEFAULT_CONDITIONS) == 3
    assert L.DEFAULT_SEEDS == (0, 1, 2)

    rep = L.run_audit()
    assert rep["ok"] is True
    assert len(rep["runs"]) == 3 * 3
    assert [run["condition"] for run in rep["runs"]] == [
        c for c in L.DEFAULT_CONDITIONS for _ in L.DEFAULT_SEEDS]
    for run in rep["runs"]:
        assert len(run["pairs"]) == len(A.MULTISCALE_CANDIDATES) == 4
        for pair in run["pairs"]:
            assert set(pair) >= {"multiscale_name", "l1_name", "multiscale", "l1"}
            for side in ("multiscale", "l1"):
                whole = pair[side]["whole_frame"]
                assert 0.0 <= whole["common_valid_fraction_of_total"] <= 1.0
                if whole["rgb_mae"] is not None:
                    assert math.isfinite(whole["rgb_mae"]) and whole["rgb_mae"] >= 0.0
                assert set(pair[side]["regions"]) == set(L.LAYER_NAMES)
    _assert_json_native(rep)


def test_cli_prints_json_report_to_stdout(capsys):
    assert L.main(["--seeds", "0", "--conditions", "static_no_parallax", "--n", "12",
                   "--width", "16", "--height", "10", "--margin", "8"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == L.SCHEMA and report["ok"] is True
    assert len(report["runs"]) == 1
    assert report["runs"][0]["condition"] == "static_no_parallax"


def test_cli_writes_json_report_to_out_file(tmp_path):
    out = tmp_path / "layered_audit.json"
    assert L.main(["--seeds", "0", "--conditions", "parallax_shake", "--n", "12",
                   "--width", "16", "--height", "10", "--margin", "8", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema"] == L.SCHEMA and report["ok"] is True
    assert {run["condition"] for run in report["runs"]} == {"parallax_shake"}


def test_cli_emits_json_safe_errors(capsys):
    assert L.main(["--seeds", "0", "--conditions", "bogus"]) == 2
    err = json.loads(capsys.readouterr().out)
    assert err["schema"] == L.SCHEMA and err["ok"] is False
    json.dumps(err, allow_nan=False)

    assert L.main(["--seeds", "0", "--conditions", "static_no_parallax", "--width", "0"]) == 2
    err = json.loads(capsys.readouterr().out)
    assert err["ok"] is False
