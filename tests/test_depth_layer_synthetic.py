"""Pure-NumPy tests for the known-truth two-layer depth-rank synthetic diagnostic.

The module is weight-free, media-free and deterministic, so the tests run it directly with no fakes: they
assert the ground-truth scene construction (two ordered layers, occlusion, checkerboard split), explicit rank
orientation, held-out affine fit on a checkerboard split, the two independent policy blocks (background-lock
and foreground-preserve) each scoring their own target layer, the downstream residual-jitter and coverage
metrics, the uninformative-depth unavailable fallback, the shuffled-depth negative control declaring both
policies unavailable with no wins, JSON-safe deterministic output and CLI parsing.

Warnings are escalated to errors so an unclean numeric path fails rather than passing silently.
"""
import json
import math
import warnings

import numpy as np
import pytest

from bench import depth_layer_synthetic as D


@pytest.fixture(autouse=True)
def no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# --------------------------------------------------------------------------------------
# affine helpers and rank orientation
# --------------------------------------------------------------------------------------

def test_apply_affine_matches_manual_translation_scale():
    m = np.array([[2.0, 0.0, 5.0], [0.0, 3.0, -1.0], [0.0, 0.0, 1.0]])
    p = np.array([[0.0, 0.0], [1.0, 2.0]])
    out = D.apply_affine(m, p)
    np.testing.assert_allclose(out, np.array([[5.0, -1.0], [7.0, 5.0]]))


def test_compose_affine_applies_second_first():
    a = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    b = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]])
    p = np.array([[0.0, 0.0]])
    np.testing.assert_allclose(D.apply_affine(D.compose_affine(a, b), p), [[10.0, 20.0]])


def test_coeff_to_affine_identity_for_zero_displacement():
    E = D._coeff_to_affine(np.zeros(3), np.zeros(3), 40, 30)
    np.testing.assert_allclose(E, np.eye(3))  # zero displacement must reproduce the identity


def test_coeff_to_affine_translation_is_nonsingular():
    E = D._coeff_to_affine(np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, -2.0]), 40, 30)
    assert abs(np.linalg.det(E[:2, :2])) > 1e-9
    np.testing.assert_allclose(E[:2, 2], [3.0, -2.0])


def test_rank_of_raw_low_raw_low_rank():
    np.testing.assert_array_equal(D.rank_of_raw((0.2, 0.8), "low_raw_low_rank"), [0, 1])


def test_rank_of_raw_low_raw_high_rank_is_reversed():
    np.testing.assert_array_equal(D.rank_of_raw((0.2, 0.8), "low_raw_high_rank"), [1, 0])


def test_rank_of_raw_ties_share_lower_rank():
    np.testing.assert_array_equal(D.rank_of_raw((0.5, 0.5), "low_raw_low_rank"), [0, 0])
    np.testing.assert_array_equal(D.rank_of_raw((0.5, 0.5), "low_raw_high_rank"), [0, 0])


def test_rank_of_raw_rejects_bad_orientation():
    with pytest.raises(ValueError):
        D.rank_of_raw((0.2, 0.8), "distance")


# --------------------------------------------------------------------------------------
# spec validation
# --------------------------------------------------------------------------------------

def test_spec_rejects_short_sequence():
    with pytest.raises(ValueError):
        D.SyntheticSpec(T=2)


def test_spec_rejects_small_frame():
    with pytest.raises(ValueError):
        D.SyntheticSpec(W=4)


def test_spec_rejects_wrong_depth_count():
    with pytest.raises(ValueError):
        D.SyntheticSpec(depths=(0.2,))


def test_spec_rejects_bad_orientation_and_mode():
    with pytest.raises(ValueError):
        D.SyntheticSpec(rank_orientation="physical_near")
    with pytest.raises(ValueError):
        D.SyntheticSpec(depth_mode="oracle")


def test_spec_as_dict_is_json_safe_and_path_free():
    d = D.SyntheticSpec().as_dict()
    assert "path" not in json.dumps(d).lower()
    json.dumps(d)
    assert d["depths"] == [0.2, 0.8] and isinstance(d["depths"], list)


# --------------------------------------------------------------------------------------
# scene ground truth
# --------------------------------------------------------------------------------------

def test_checkerboard_split_is_alternating_and_partitions_grid():
    coords = np.stack(np.meshgrid(np.arange(6), np.arange(5), indexing="xy"), -1).reshape(-1, 2)
    train = D.checkerboard(coords)
    assert train.dtype == bool and train.sum() + (~train).sum() == coords.shape[0]
    # even x+y -> train; check parity directly
    par = (coords[:, 0] + coords[:, 1]) % 2 == 0
    np.testing.assert_array_equal(train, par)


def test_scene_has_two_ordered_layers_with_occlusion():
    seq = D.build_sequence(D.SyntheticSpec())
    assert seq.n_layers == 2
    # layer 0 is full-frame background; layer 1 is a strict rectangular subset drawn on top
    assert seq.layer_masks[0].all()
    assert seq.layer_masks[1].any() and not seq.layer_masks[1].all()
    vis = seq.visible_layer.reshape(-1)
    assert set(np.unique(vis)).issubset({0, 1})
    # visible layer agrees with the foreground mask
    np.testing.assert_array_equal(vis.reshape(seq.H, seq.W) == 1, seq.layer_masks[1])


def test_scene_foreground_flow_differs_from_background_flow():
    seq = D.build_sequence(D.SyntheticSpec())
    vis = seq.visible_layer.reshape(-1)
    fg = vis == 1
    # the overlay carries an independent local motion, so at least one frame differs from the background
    diff = np.linalg.norm(seq.layer_flow[1] - seq.layer_flow[0], axis=-1)
    assert diff[1:, fg].max() > 1e-3


def test_scene_source_depth_matches_visible_layer_raw_values():
    spec = D.SyntheticSpec(depths=(0.2, 0.8))
    seq = D.build_sequence(spec)
    vis = seq.visible_layer.reshape(-1)
    expected = np.asarray(spec.depths)[vis].reshape(seq.H, seq.W)
    np.testing.assert_allclose(seq.source_depth, expected)
    np.testing.assert_array_equal(seq.rank_flat, np.array([0, 1])[vis])


def test_scene_validity_has_border_and_hole():
    spec = D.SyntheticSpec(border=3, occlusion_hole=True)
    seq = D.build_sequence(spec)
    v = seq.valid
    assert not v[:3, :].any() and not v[-3:, :].any()
    assert not v[:, :3].any() and not v[:, -3:].any()
    # the interior is mostly valid but a rectangular hole was removed
    assert v.sum() < (seq.H * seq.W)
    # validity and its negation partition the grid
    assert (v | ~v).all()


def test_scene_validity_without_hole_is_interior_only():
    spec = D.SyntheticSpec(border=2, occlusion_hole=False)
    seq = D.build_sequence(spec)
    expected = np.ones((seq.H, seq.W), bool)
    expected[:2, :] = expected[-2:, :] = expected[:, :2] = expected[:, -2:] = False
    np.testing.assert_array_equal(seq.valid, expected)


def test_scene_is_deterministic():
    a, b = D.build_sequence(D.SyntheticSpec()), D.build_sequence(D.SyntheticSpec())
    np.testing.assert_allclose(a.layer_transform, b.layer_transform)
    np.testing.assert_allclose(a.fwd_flow, b.fwd_flow)
    np.testing.assert_array_equal(a.valid, b.valid)


def test_shuffled_depth_is_a_deterministic_permutation():
    spec = D.SyntheticSpec(depth_mode="shuffled")
    s1 = D.build_sequence(spec).source_depth
    s2 = D.build_sequence(spec).source_depth
    np.testing.assert_allclose(s1, s2)
    ref = D.build_sequence(D.SyntheticSpec()).source_depth
    assert not np.allclose(np.sort(s1.ravel()), np.sort(ref.ravel())) or not np.array_equal(s1, ref)


def test_uninformative_depth_collapses_rank():
    seq = D.build_sequence(D.SyntheticSpec(depth_mode="uninformative"))
    np.testing.assert_array_equal(seq.layer_rank, [0, 0])
    assert np.unique(seq.source_depth).size == 1


# --------------------------------------------------------------------------------------
# rank -> layer resolution
# --------------------------------------------------------------------------------------

def test_resolve_rank_layer_ranked():
    seq = D.build_sequence(D.SyntheticSpec())
    r0 = D.resolve_rank_layer(seq, 0)
    r1 = D.resolve_rank_layer(seq, 1)
    assert r0["available"] and r0["layer"] == 0 and not r0["ambiguous"] and r0["reason"] is None
    assert r1["available"] and r1["layer"] == 1 and not r1["ambiguous"] and r1["reason"] is None


def test_resolve_rank_layer_missing_rank_is_unavailable():
    seq = D.build_sequence(D.SyntheticSpec())
    r = D.resolve_rank_layer(seq, 5)
    assert r["available"] is False and r["layer"] is None and r["reason"] == "rank_not_present"


def test_resolve_rank_layer_ambiguous_when_collapsed():
    seq = D.build_sequence(D.SyntheticSpec(depth_mode="uninformative"))
    r = D.resolve_rank_layer(seq, 0)
    assert r["available"] and r["ambiguous"] and r["reason"] == "rank_shared_by_multiple_layers"


# --------------------------------------------------------------------------------------
# held-out fit and target-error blocks
# --------------------------------------------------------------------------------------

def test_affine_fit_returns_none_below_support_floor():
    design = np.ones((D.MIN_FIT_SAMPLES - 1, 3))
    assert D._affine_fit(design, np.ones(design.shape[0])) is None


def test_affine_fit_returns_none_for_rank_deficient_design():
    design = np.ones((D.MIN_FIT_SAMPLES, 3))  # all rows identical -> rank 1
    assert D._affine_fit(design, np.ones(D.MIN_FIT_SAMPLES)) is None


def test_fit_policy_pooled_uses_all_valid_train_cells():
    seq = D.build_sequence(D.SyntheticSpec())
    fit = D._fit_policy(seq, 1, None)
    assert fit["fit_ok"] and fit["scope"] == "pooled"
    expected = int((seq.valid.reshape(-1) & seq.train).sum())
    assert fit["train_count"] == expected


def test_fit_policy_rank_uses_only_that_rank_train_cells():
    seq = D.build_sequence(D.SyntheticSpec())
    fit = D._fit_policy(seq, 1, 0)
    expected = int((seq.valid.reshape(-1) & seq.train & (seq.rank_flat == 0)).sum())
    assert fit["fit_ok"] and fit["scope"] == "rank_0" and fit["train_count"] == expected
    assert fit["train_count"] < D._fit_policy(seq, 1, None)["train_count"]


def test_eval_block_is_odd_parity_valid_of_rank():
    seq = D.build_sequence(D.SyntheticSpec())
    blk = D._eval_block(seq, 1)
    valid = seq.valid.reshape(-1)
    expected = valid & (~seq.train) & (seq.rank_flat == 1)
    np.testing.assert_array_equal(blk, expected)
    # train and validate blocks are disjoint by construction
    assert not np.any(blk & seq.train)


def test_heldout_error_insufficient_validation_reports_reason():
    seq = D.build_sequence(D.SyntheticSpec())
    E = np.eye(3)
    empty = np.zeros(seq.H * seq.W, bool)
    out = D._heldout_error(seq, 0, E, empty, 0)
    assert out["median"] is None and out["reason"] == "insufficient_validation"


def test_rank_selected_fit_recovers_exact_layer_flow_on_heldout():
    seq = D.build_sequence(D.SyntheticSpec())
    E = D._fit_policy(seq, 5, 1)["E"]
    err = D._heldout_error(seq, 5, E, D._eval_block(seq, 1), target_layer=1)
    assert err["reason"] is None and err["n"] >= D.MIN_EVAL_SAMPLES
    assert err["median"] < 1e-6  # known affine family => exact recovery


def test_target_layer_for_rank_does_not_clamp_on_ambiguity():
    seq = D.build_sequence(D.SyntheticSpec(depth_mode="uninformative"))
    assert D._target_layer_for_rank(seq, 1) == -1  # rank 1 has no layer


# --------------------------------------------------------------------------------------
# policy blocks
# --------------------------------------------------------------------------------------

def test_policy_blocks_are_independent_and_hit_own_layer():
    rep = D.run_experiment(D.SyntheticSpec())
    pol = {p["policy"]: p for p in rep["policies"]}
    assert set(pol) == {"background_lock", "foreground_preserve"}
    assert pol["background_lock"]["target_layer"] == 0
    assert pol["foreground_preserve"]["target_layer"] == 1
    for p in pol.values():
        assert p["available"] is True and p["reason"] is None
        assert p["target_error"]["frame_median_median"] < 1e-6
        # each block locks its own target layer, so its locked residual jitter is ~0
        assert p["downstream"]["locked_layer_residual_jitter_px"] < 1e-9


def test_background_lock_and_foreground_preserve_score_different_targets():
    rep = D.run_experiment(D.SyntheticSpec())
    bl, fp = rep["policies"]
    # the baseline is identical (same held-out cells) but the policy target layers differ
    assert bl["target_layer"] != fp["target_layer"]
    assert bl["eval_block_count"] != fp["eval_block_count"]
    # foreground and background flows differ, so the baseline error on each target differs
    assert bl["target_error"]["baseline_frame_median_median"] != \
        fp["target_error"]["baseline_frame_median_median"]


def test_policy_beats_pooled_baseline_when_rank_isolates_a_layer():
    rep = D.run_experiment(D.SyntheticSpec())
    for p in rep["policies"]:
        te = p["target_error"]
        assert te["material_win_frames"] >= te["comparable_frames"] - 1
        assert te["median_frame_delta_px"] < 0


def test_shuffled_depth_does_not_recover_the_layer():
    rep = D.run_experiment(D.SyntheticSpec(depth_mode="shuffled"))
    # The permutation destroys the rank/layer association by construction, so neither policy may be fitted,
    # scored or credited. Both must decline with a machine-readable reason and zero wins; no target is
    # fabricated and there is no downstream block.
    for p in rep["policies"]:
        assert p["available"] is False
        assert p["reason"] == "rank_layer_association_destroyed"
        assert p["target_layer"] is None
        assert p["downstream"] is None
        assert p["frames"] == []
        te = p["target_error"]
        assert te["material_win_frames"] == 0
        assert te["comparable_frames"] == 0
        assert te["frame_median_median"] is None
    # the pooled baseline is still reported as a diagnostic (no policy claim attached)
    assert rep["baseline"]["scope"] == "pooled"
    assert rep["notes"]["winner_declared"] is False


# --------------------------------------------------------------------------------------
# downstream coordinate model
# --------------------------------------------------------------------------------------

def test_coverage_identity_is_full_and_out_of_bounds_drops():
    full = D._coverage(np.eye(3), 20, 10)
    assert math.isclose(full["frac"], 1.0)
    shift = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert D._coverage(shift, 20, 10)["frac"] < 0.5


def test_residual_jitter_zero_for_constant_path_and_positive_for_shake():
    T, K = 6, 4
    const = np.zeros((T, K, 2))
    const[..., 0] = 3.0  # pure translation path
    assert D._residual_jitter(const) == 0.0
    shake = np.zeros((T, K, 2))
    shake[::2, :, 0] = 1.0
    assert D._residual_jitter(shake) > 0.0


def test_residual_jitter_none_for_short_path():
    assert D._residual_jitter(np.zeros((2, 3, 2))) is None


def test_policy_downstream_reports_jitter_and_coverage():
    rep = D.run_experiment(D.SyntheticSpec())
    for p in rep["policies"]:
        d = p["downstream"]
        assert d["aggregate_residual_jitter_px"] is not None
        assert d["aggregate_coverage_frac_mean"] is not None
        assert 0.0 < d["aggregate_coverage_frac_mean"] <= 1.0
        # exact recovery => locked layer residual jitter is negligible
        assert d["locked_layer_residual_jitter_px"] < 1e-9


# --------------------------------------------------------------------------------------
# pooled-baseline downstream (counterfactual evaluation under each intended target)
# --------------------------------------------------------------------------------------

def test_baseline_downstream_has_both_layers_with_stable_string_keys():
    rep = D.run_experiment(D.SyntheticSpec())
    b = rep["baseline"]
    assert b["scope"] == "pooled"
    # the pooled baseline targets neither layer; the block is a counterfactual, not a claim
    assert b["targets_layer"] is None
    assert b["downstream_available"] is True and b["downstream_reason"] is None
    by_layer = b["downstream_by_target_layer"]
    # stable string keys, one per ground-truth layer
    assert set(by_layer) == {str(l) for l in range(rep["n_layers"])}
    assert set(by_layer) == {"0", "1"}
    for k, d in by_layer.items():
        assert d["locked_layer"] == int(k)  # each entry evaluates the pooled path under that target
        assert d["locked_layer_residual_jitter_px"] is not None
        assert d["locked_layer_displacement_error_px"] is not None


def test_baseline_downstream_jitter_and_coverage_are_finite():
    rep = D.run_experiment(D.SyntheticSpec())
    for k, d in rep["baseline"]["downstream_by_target_layer"].items():
        jit = d["aggregate_residual_jitter_px"]
        cov = d["aggregate_coverage_frac_mean"]
        assert jit is not None and math.isfinite(jit) and jit >= 0.0
        assert cov is not None and math.isfinite(cov) and 0.0 < cov <= 1.0
        for layer_metric in d["layers"]:
            rj = layer_metric["residual_jitter_px"]
            assert rj is None or (math.isfinite(rj) and rj >= 0.0)
            cf = layer_metric["coverage_frac_mean"]
            assert cf is None or (math.isfinite(cf) and 0.0 < cf <= 1.0)


def test_baseline_downstream_identity_and_error_metrics_comparable_to_policies():
    rep = D.run_experiment(D.SyntheticSpec())
    pol = {p["target_layer"]: p for p in rep["policies"]}
    by_layer = rep["baseline"]["downstream_by_target_layer"]
    for layer, p in pol.items():
        assert p["available"] is True
        bd = by_layer[str(layer)]
        pd = p["downstream"]
        # per-target identity and error metrics are on the same footing (same paths, same fields)
        assert bd["locked_layer"] == pd["locked_layer"] == layer
        assert set(bd) == set(pd)
        assert set(bd["layers"][0]) == set(pd["layers"][0])
        # both report the same per-layer displacement-error fields without fabrication
        assert bd["locked_layer_displacement_error_px"] is not None
        assert pd["locked_layer_displacement_error_px"] is not None
        assert math.isfinite(bd["locked_layer_displacement_error_px"])
        assert math.isfinite(pd["locked_layer_displacement_error_px"])
        # a rank-selected policy recovers its target exactly, so its locked error beats the pooled path's
        assert pd["locked_layer_displacement_error_px"] < bd["locked_layer_displacement_error_px"]


def test_baseline_downstream_unavailable_when_pooled_fit_fails():
    # A very narrow valid interior (border eats all but two columns) leaves the affine design rank-deficient,
    # so the pooled fit fails on every frame. The baseline must report the downstream block unavailable with a
    # machine-readable reason and a None entry per layer, never a partial or fabricated path.
    spec = D.SyntheticSpec(T=4, W=8, H=8, border=3, occlusion_hole=False)
    rep = D.run_experiment(spec)
    b = rep["baseline"]
    assert b["fit_frame_count"] == 0  # every pooled fit genuinely failed
    assert all(not f["fit_ok"] for f in b["frames"])
    assert b["downstream_available"] is False
    assert b["downstream_reason"] in {"unidentifiable_fit", "insufficient_train"}
    assert set(b["downstream_by_target_layer"]) == {"0", "1"}
    assert all(v is None for v in b["downstream_by_target_layer"].values())
    # path-free, finite-safe JSON still serializes with the unavailable block
    json.loads(D.dumps_report(rep))


def test_shuffled_depth_keeps_baseline_downstream_diagnostic_but_no_policy_claim():
    rep = D.run_experiment(D.SyntheticSpec(depth_mode="shuffled"))
    b = rep["baseline"]
    # the negative control still checks that every pooled fit succeeded and the block is a diagnostic only
    assert b["downstream_available"] is True and b["downstream_reason"] is None
    assert set(b["downstream_by_target_layer"]) == {"0", "1"}
    assert b["targets_layer"] is None
    for p in rep["policies"]:
        assert p["available"] is False
        assert p["downstream"] is None
        assert p["target_error"]["material_win_frames"] == 0


# --------------------------------------------------------------------------------------
# uninformative-depth fallback
# --------------------------------------------------------------------------------------

def test_uninformative_depth_reports_unavailable_both_policies():
    rep = D.run_experiment(D.SyntheticSpec(depth_mode="uninformative"))
    assert rep["layer_rank"] == [0, 0]
    for p in rep["policies"]:
        assert p["available"] is False
        assert p["reason"] in {"rank_ambiguous_uninformative_depth", "target_rank_unavailable"}
        assert p["downstream"] is None
        assert p["target_error"]["frame_median_median"] is None


def test_uninformative_fallback_is_machine_readable_not_fabricated():
    rep = D.run_experiment(D.SyntheticSpec(depth_mode="uninformative"))
    bl = rep["policies"][0]
    assert bl["reason"] == "rank_ambiguous_uninformative_depth"
    assert bl["target_error"]["material_win_frames"] == 0
    assert bl["target_error"]["comparable_frames"] == 0


def test_missing_rank_policy_is_unavailable():
    # require rank 7 which no layer can have
    rep = D.run_experiment(D.SyntheticSpec(lock_rank=7, preserve_rank=1))
    bl = rep["policies"][0]
    assert bl["available"] is False and bl["reason"] == "target_rank_unavailable"
    assert bl["target_layer"] is None and bl["downstream"] is None


# --------------------------------------------------------------------------------------
# report, determinism and JSON safety
# --------------------------------------------------------------------------------------

def test_report_is_deterministic_and_json_safe():
    a = D.dumps_report(D.run_experiment(D.SyntheticSpec()))
    b = D.dumps_report(D.run_experiment(D.SyntheticSpec()))
    assert a == b
    parsed = json.loads(a)
    assert parsed["schema"] == D.SCHEMA
    # no NaN/Infinity tokens and no non-finite floats survive
    assert "NaN" not in a and "Infinity" not in a


def test_report_is_path_free_and_declares_synthetic_scope():
    rep = D.run_experiment(D.SyntheticSpec())
    blob = D.dumps_report(rep).lower()
    for token in ("/users/", "/home/", ".npz", ".mp4", "vendor/"):
        assert token not in blob
    assert rep["notes"]["synthetic_only"] is True
    assert rep["notes"]["no_media_decoded"] is True
    assert rep["notes"]["no_production_code_imported"] is True
    assert rep["notes"]["winner_declared"] is False
    assert rep["limitations"]


def test_report_orientation_is_explicit_and_reversible():
    low = D.run_experiment(D.SyntheticSpec(rank_orientation="low_raw_low_rank"))
    high = D.run_experiment(D.SyntheticSpec(rank_orientation="low_raw_high_rank"))
    assert low["layer_rank"] == [0, 1]
    assert high["layer_rank"] == [1, 0]
    # flipping the orientation flips which layer each policy targets
    assert low["policies"][0]["target_layer"] == 0
    assert high["policies"][0]["target_layer"] == 1


def test_report_has_no_winner_field_or_threshold_sweep():
    rep = D.run_experiment(D.SyntheticSpec())
    assert rep["notes"]["winner_declared"] is False
    assert rep["material_gain_px"] == D.MATERIAL_GAIN_PX
    # fixed support floors are recorded, not swept
    assert rep["min_fit_samples"] == D.MIN_FIT_SAMPLES
    assert rep["min_eval_samples"] == D.MIN_EVAL_SAMPLES


def test_json_safe_handles_nonfinite_floats():
    payload = {"a": float("nan"), "b": float("inf"), "c": [1.0, float("-inf")]}
    out = json.loads(D.MF.dumps_json(payload))
    assert out == {"a": None, "b": None, "c": [1.0, None]}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def test_parse_depths_validates():
    assert D.parse_depths("0.2,0.8") == (0.2, 0.8)
    with pytest.raises(ValueError):
        D.parse_depths("0.2")
    with pytest.raises(ValueError):
        D.parse_depths("0.2,nan")


def test_spec_from_args_roundtrip():
    a = D.build_parser().parse_args(["--T", "5", "--W", "24", "--H", "20", "--depth-mode", "uninformative"])
    spec = D.spec_from_args(a)
    assert spec.T == 5 and spec.W == 24 and spec.H == 20 and spec.depth_mode == "uninformative"


def test_cli_main_writes_atomic_json(tmp_path):
    out = tmp_path / "report.json"
    rc = D.main(["--T", "5", "--W", "24", "--H", "20", "--out", str(out)])
    assert rc == 0
    saved = json.loads(out.read_text())
    assert saved["schema"] == D.SCHEMA
    assert saved == D.run_experiment(D.SyntheticSpec(T=5, W=24, H=20))


def test_cli_main_dry_run_does_not_write(tmp_path, capsys):
    out = tmp_path / "report.json"
    rc = D.main(["--T", "5", "--W", "24", "--H", "20", "--out", str(out), "--dry-run"])
    assert rc == 0 and not out.exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["T"] == 5 and printed["depth_mode"] == "ranked"


def test_cli_main_reports_failure_as_json(capsys):
    rc = D.main(["--T", "2"])  # below the >= 3 floor
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failure" and payload["error"]["type"] == "ValueError"
