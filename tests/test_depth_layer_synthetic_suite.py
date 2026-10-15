"""Pure-NumPy tests for the exploratory multi-seed / foreground-geometry suite.

The suite is weight-free, media-free and deterministic, so the tests run it directly with no fakes. They cover
foreground-rectangle validation and the default mask, deterministic suite output, the expected 3x2x4 case grid,
the no-cross-geometry-pooling guarantee, agreement between the per-cell summaries and their constituent seed
results, path-free finite-safe JSON and atomic CLI output. Runs are kept cheap with a small T/W/H.
"""
import json
import math
import warnings

import numpy as np
import pytest

from bench import depth_layer_synthetic as D
from bench import depth_layer_synthetic_suite as S


@pytest.fixture(autouse=True)
def no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# A small fixture spec shared by the cheap suite runs (kept minimal so the grid is fast).
CHEAP = {"T": 5, "W": 24, "H": 20}


# --------------------------------------------------------------------------------------
# fg_rect geometry validation and default mask
# --------------------------------------------------------------------------------------

def test_fg_rect_default_preserves_the_e09_e10_geometry():
    spec = D.SyntheticSpec()
    assert spec.fg_rect == (0.56, 0.52, 0.20, 0.20)
    # the default equals the historical hard-coded rectangle exactly at any frame size
    seq = D.build_sequence(spec)
    expected = D._rect_mask(seq.W, seq.H, 0.56 * seq.W, 0.52 * seq.H,
                            max(2.0, 0.20 * seq.W), max(2.0, 0.20 * seq.H))
    np.testing.assert_array_equal(seq.layer_masks[1], expected)


def test_fg_rect_default_mask_is_unchanged_from_historical_construction():
    # a differently shaped frame confirms the mapping is normalized, not a fixed pixel box
    seq = D.build_sequence(D.SyntheticSpec(W=40, H=40))
    expected = D._rect_mask(40, 40, 0.56 * 40, 0.52 * 40, max(2.0, 0.20 * 40), max(2.0, 0.20 * 40))
    np.testing.assert_array_equal(seq.layer_masks[1], expected)


def test_fg_rect_changes_the_foreground_mask():
    a = D.build_sequence(D.SyntheticSpec(fg_rect=(0.56, 0.52, 0.20, 0.20)))
    b = D.build_sequence(D.SyntheticSpec(fg_rect=(0.32, 0.48, 0.12, 0.35)))
    assert not np.array_equal(a.layer_masks[1], b.layer_masks[1])


def test_fg_rect_is_in_as_dict_and_json_safe():
    d = D.SyntheticSpec(fg_rect=(0.64, 0.35, 0.30, 0.14)).as_dict()
    assert d["fg_rect"] == [0.64, 0.35, 0.30, 0.14]
    assert isinstance(d["fg_rect"], list)
    json.dumps(d)
    assert "path" not in json.dumps(d).lower()


def test_fg_rect_rejects_wrong_length_and_nonfinite():
    with pytest.raises(ValueError):
        D.SyntheticSpec(fg_rect=(0.5, 0.5, 0.2))
    with pytest.raises(ValueError):
        D.SyntheticSpec(fg_rect=(0.5, 0.5, 0.2, 0.2, 0.1))
    with pytest.raises(ValueError):
        D.SyntheticSpec(fg_rect=(0.5, 0.5, float("nan"), 0.2))
    with pytest.raises(ValueError):
        D.SyntheticSpec(fg_rect=(0.5, float("inf"), 0.2, 0.2))


def test_fg_rect_rejects_nonpositive_half_extents():
    with pytest.raises(ValueError):
        D.SyntheticSpec(fg_rect=(0.5, 0.5, 0.0, 0.2))
    with pytest.raises(ValueError):
        D.SyntheticSpec(fg_rect=(0.5, 0.5, 0.2, -0.1))


def test_fg_rect_rejects_rectangle_outside_unit_square():
    with pytest.raises(ValueError):
        D.SyntheticSpec(fg_rect=(0.05, 0.5, 0.10, 0.2))   # cx - half_w < 0
    with pytest.raises(ValueError):
        D.SyntheticSpec(fg_rect=(0.95, 0.5, 0.10, 0.2))   # cx + half_w > 1
    with pytest.raises(ValueError):
        D.SyntheticSpec(fg_rect=(0.5, 0.05, 0.2, 0.10))   # cy - half_h < 0
    with pytest.raises(ValueError):
        D.SyntheticSpec(fg_rect=(0.5, 0.95, 0.2, 0.10))   # cy + half_h > 1


def test_fg_rect_boundary_touching_unit_square_is_accepted():
    spec = D.SyntheticSpec(fg_rect=(0.5, 0.5, 0.5, 0.5))
    assert spec.fg_rect == (0.5, 0.5, 0.5, 0.5)


def test_parse_fg_rect_roundtrip_and_validation():
    assert D.parse_fg_rect("0.56,0.52,0.20,0.20") == (0.56, 0.52, 0.20, 0.20)
    with pytest.raises(ValueError):
        D.parse_fg_rect("0.56,0.52,0.20")
    with pytest.raises(ValueError):
        D.parse_fg_rect("0.56,0.52,0.20,nan")


def test_cli_fg_rect_roundtrips_through_spec():
    a = D.build_parser().parse_args(["--fg-rect", "0.32,0.48,0.12,0.35", "--T", "5", "--W", "24", "--H", "20"])
    spec = D.spec_from_args(a)
    assert spec.fg_rect == (0.32, 0.48, 0.12, 0.35)


def test_cli_default_fg_rect_matches_spec_default():
    spec = D.spec_from_args(D.build_parser().parse_args([]))
    assert spec.fg_rect == D.SyntheticSpec().fg_rect


# --------------------------------------------------------------------------------------
# grid shape
# --------------------------------------------------------------------------------------

def test_suite_grid_is_three_geometries_two_conditions_four_seeds():
    rep = S.run_suite(base=CHEAP)
    assert S.SEEDS == (0, 1, 7, 42)
    assert len(S.GEOMETRIES) == 3
    assert len(S.CONDITIONS) == 2
    g = rep["grid"]
    assert g["n_seeds"] == 4 and g["n_geometries"] == 3 and g["n_conditions"] == 2 and g["n_policies"] == 2
    assert g["n_cases"] == 3 * 2 * 4 == 24
    assert len(rep["cases"]) == 24


def test_suite_predeclared_geometries_and_default_match_e09_e10():
    names = [n for n, _ in S.GEOMETRIES]
    assert names == ["default_center", "offcenter_vertical", "upper_wide"]
    assert dict(S.GEOMETRIES)["default_center"] == (0.56, 0.52, 0.20, 0.20)
    assert dict(S.GEOMETRIES)["offcenter_vertical"] == (0.32, 0.48, 0.12, 0.35)
    assert dict(S.GEOMETRIES)["upper_wide"] == (0.64, 0.35, 0.30, 0.14)


def test_suite_conditions_are_fixed_ideal_and_stress():
    conds = dict(S.CONDITIONS)
    assert set(conds) == {"ideal", "stress"}
    assert conds["ideal"] == {"flow_noise_px": 0.0, "depth_error_frac": 0.0, "fg_nonaffine_px": 0.0}
    assert conds["stress"] == {"flow_noise_px": 0.5, "depth_error_frac": 0.10, "fg_nonaffine_px": 1.0}


def test_suite_preserves_every_individual_seed_result():
    rep = S.run_suite(base=CHEAP)
    keys = {(c["geometry"], c["condition"], c["seed"]) for c in rep["cases"]}
    expected = {(g, c, s) for g, _ in S.GEOMETRIES for c, _ in S.CONDITIONS for s in S.SEEDS}
    assert keys == expected
    for c in rep["cases"]:
        assert c["schema"] == D.SCHEMA
        assert len(c["policies"]) == 2
        assert c["spec"]["fg_rect"] == c["fg_rect"]


# --------------------------------------------------------------------------------------
# no cross-geometry pooling; summary matches constituent seeds
# --------------------------------------------------------------------------------------

def test_suite_summaries_are_keyed_per_geometry_condition_policy():
    rep = S.run_suite(base=CHEAP)
    expected = {f"{g}|{c}|{p}" for g, _ in S.GEOMETRIES for c, _ in S.CONDITIONS for p in S.POLICY_NAMES}
    assert set(rep["summaries"]) == expected
    for key, s in rep["summaries"].items():
        g, c, p = key.split("|")
        assert s["geometry"] == g and s["condition"] == c and s["policy"] == p
        assert s["total_cases"] == len(S.SEEDS)


def test_suite_does_not_pool_across_geometries():
    # two geometries with clearly different masks must be allowed to summarise independently; a shared key or a
    # geometry-agnostic bucket would collapse them. Assert each cell only ever sees its own geometry's cases.
    rep = S.run_suite(base=CHEAP)
    assert rep["notes"]["pooled_across_geometries"] is False
    # geometry-specific keys are distinct and each carries its own geometry label
    geoms = {s["geometry"] for s in rep["summaries"].values()}
    assert geoms == {n for n, _ in S.GEOMETRIES}
    # no summary is keyed or labelled as a combined/pooled geometry
    for key, s in rep["summaries"].items():
        assert key.startswith(s["geometry"] + "|")
        assert "pooled" not in key and "combined" not in key and "all" != s["geometry"]


def test_suite_summary_values_match_constituent_seed_results():
    rep = S.run_suite(base=CHEAP)
    # rebuild the per-cell metric lists directly from the preserved cases and check the summary matches
    from bench.depth_layer_synthetic_suite import _policy_target_metrics, _median_range
    for g, _ in S.GEOMETRIES:
        for c, _ in S.CONDITIONS:
            for p in S.POLICY_NAMES:
                cases = [x for x in rep["cases"] if x["geometry"] == g and x["condition"] == c]
                cases.sort(key=lambda x: x["seed"])
                metrics = [_policy_target_metrics(x, p) for x in cases]
                summary = rep["summaries"][f"{g}|{c}|{p}"]
                assert summary["total_cases"] == len(cases)
                avail = [m for m in metrics if m is not None and m["available"]]
                assert summary["available_cases"] == len(avail)
                te = _median_range([m["target_error_median_px"] for m in avail])
                jt = _median_range([m["target_layer_jitter_px"] for m in avail])
                cv = _median_range([m["target_layer_coverage_frac"] for m in avail])
                assert summary["target_error_median_px"] == te
                assert summary["target_layer_jitter_px"] == jt
                assert summary["target_layer_coverage_frac"] == cv


def test_suite_summary_median_min_max_are_consistent():
    rep = S.run_suite(base=CHEAP)
    for s in rep["summaries"].values():
        for metric in ("target_error_median_px", "target_layer_jitter_px", "target_layer_coverage_frac"):
            m = s[metric]
            if m["n"] == 0:
                assert m["median"] is None and m["min"] is None and m["max"] is None
                continue
            assert m["min"] <= m["median"] <= m["max"]
            assert math.isfinite(m["median"]) and math.isfinite(m["min"]) and math.isfinite(m["max"])
        assert 0 <= s["material_win_count"] <= s["available_cases"]
        assert s["comparable_frames_total"] >= s["material_win_frames_total"] >= 0


def test_suite_summary_of_one_seed_matches_that_seed_exactly():
    # a single-seed run's summary must equal the corresponding seed case number exactly (no smoothing)
    rep = S.run_suite(seed_list=(7,), base=CHEAP)
    for g, _ in S.GEOMETRIES:
        for c, _ in S.CONDITIONS:
            for p in S.POLICY_NAMES:
                s = rep["summaries"][f"{g}|{c}|{p}"]
                case = next(x for x in rep["cases"] if x["geometry"] == g and x["condition"] == c)
                pol = next(x for x in case["policies"] if x["policy"] == p)
                assert pol["available"] is True  # CHEAP ideal/stress both resolve the target ranks
                assert s["target_error_median_px"]["median"] == pytest.approx(
                    pol["target_error"]["frame_median_median"])
                assert s["target_error_median_px"]["min"] == pytest.approx(
                    pol["target_error"]["frame_median_median"])
                assert s["target_error_median_px"]["max"] == pytest.approx(
                    pol["target_error"]["frame_median_median"])


def test_suite_ideal_target_error_is_near_zero_and_stress_is_larger():
    # on the noiseless fixture a rank-selected affine fit recovers the known affine layer; the stress controls
    # break that exact recovery. This is a descriptive contrast, not a threshold or winner.
    rep = S.run_suite(base=CHEAP)
    for g, _ in S.GEOMETRIES:
        for p in S.POLICY_NAMES:
            ideal = rep["summaries"][f"{g}|ideal|{p}"]["target_error_median_px"]["median"]
            stress = rep["summaries"][f"{g}|stress|{p}"]["target_error_median_px"]["median"]
            assert ideal is not None and stress is not None
            assert ideal < 1e-6
            assert stress > ideal


# --------------------------------------------------------------------------------------
# determinism, JSON safety and scope flags
# --------------------------------------------------------------------------------------

def test_suite_is_deterministic():
    a = S.dumps_report(S.run_suite(base=CHEAP))
    b = S.dumps_report(S.run_suite(base=CHEAP))
    assert a == b


def test_suite_json_is_path_free_and_finite_safe():
    rep = S.run_suite(base=CHEAP)
    blob = S.dumps_report(rep).lower()
    for token in ("/users/", "/home/", ".npz", ".mp4", "vendor/"):
        assert token not in blob
    assert "nan" not in blob and "infinity" not in blob
    json.loads(S.dumps_report(rep))


def test_suite_notes_declare_exploratory_not_inferential_scope():
    rep = S.run_suite(base=CHEAP)
    n = rep["notes"]
    assert n["synthetic_only"] is True
    assert n["exploratory_only"] is True
    assert n["not_inferential"] is True
    assert n["winner_declared"] is False
    assert n["threshold_selected"] is False
    assert n["pooled_across_geometries"] is False
    assert rep["limitations"]


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def test_suite_cli_writes_atomic_json(tmp_path):
    out = tmp_path / "suite.json"
    rc = S.main(["--T", "5", "--W", "24", "--H", "20", "--out", str(out)])
    assert rc == 0
    saved = json.loads(out.read_text())
    assert saved["schema"] == S.SCHEMA
    assert saved == S.run_suite(base=CHEAP)
    # no stray temp files left behind by the atomic write
    assert list(tmp_path.glob("*.tmp")) == []


def test_suite_cli_dry_run_does_not_write(tmp_path, capsys):
    out = tmp_path / "suite.json"
    rc = S.main(["--T", "5", "--W", "24", "--H", "20", "--out", str(out), "--dry-run"])
    assert rc == 0 and not out.exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["grid"]["n_cases"] == 24
    assert printed["seeds"] == [0, 1, 7, 42]


def test_suite_cli_reports_failure_as_json(capsys):
    rc = S.main(["--T", "2"])  # below the SyntheticSpec >= 3 floor
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failure" and payload["error"]["type"] == "ValueError"
