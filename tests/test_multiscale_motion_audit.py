"""Tests for the research-only E10 path comparison runner (``bench/multiscale_motion_audit.py``).

The runner is exercised on the fixed synthetic fixture only: no media, no rendering, no MLX, no production
wiring and no changed defaults. Coverage: determinism, independent metric recomputation, strict JSON safety
and path-freedom, the explicit fixed parameter grids, the static no-shake control, malformed-input rejection,
the absence of any ranking/winner/score field, and the CLI (stdout, ``--out`` and JSON error exits).
"""
import json
import math

import numpy as np
import pytest

from bench import multiscale_motion_audit as A


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
        if isinstance(value, list):
            assert got[key] == pytest.approx(value, rel=1e-9, abs=1e-12)
        else:
            assert got[key] == pytest.approx(value, rel=1e-9, abs=1e-12)


# --------------------------------------------------------------------------------------
# metric recomputation
# --------------------------------------------------------------------------------------

def test_translation_path_metrics_match_manual_formula():
    intended = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [6.0, 4.0], [10.0, 5.0]])
    observed = intended + 0.25
    output = np.array([[0.1, -0.2], [0.9, 2.4], [2.7, 1.3], [6.4, 3.6], [9.5, 5.2]])

    m = A.translation_path_metrics(intended, observed, output)
    residual = output - intended
    second = residual[2:] - 2.0 * residual[1:-1] + residual[:-2]
    correction = output - observed

    assert m["rms_path_deviation"] == pytest.approx(float(np.sqrt(np.mean(residual ** 2))))
    assert m["rms_second_difference_residual"] == pytest.approx(float(np.sqrt(np.mean(second ** 2))))
    assert m["max_abs_correction_from_observed"] == pytest.approx(float(np.max(np.abs(correction))))
    assert m["rms_path_deviation_per_channel"] == pytest.approx(
        [float(v) for v in np.sqrt(np.mean(residual ** 2, axis=0))])
    assert m["rms_second_difference_residual_per_channel"] == pytest.approx(
        [float(v) for v in np.sqrt(np.mean(second ** 2, axis=0))])
    assert m["max_abs_correction_from_observed_per_channel"] == pytest.approx(
        [float(v) for v in np.max(np.abs(correction), axis=0)])


def test_report_metrics_recompute_from_the_documented_parameters():
    report = A.run_audit(seeds=(0,), conditions=("pan_shake",))
    run = report["runs"][0]
    fx = A.synthetic_translation_fixture(seed=0, condition="pan_shake")
    for variant in run["variants"]:
        params = variant["parameters"]
        if variant["family"] == "multiscale":
            out = A.build_multiscale_output(fx["observed"], fx["times"], params["retention_weights"])
        else:
            out = A.build_l1_output(fx["observed"], params["budget"])
        recomputed = A.translation_path_metrics(fx["pan"], fx["observed"], out)
        _assert_metrics_close(variant["metrics"], recomputed)


# --------------------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------------------

def test_run_audit_is_deterministic():
    a = A.run_audit(seeds=(0, 1), conditions=("pan_shake",))
    b = A.run_audit(seeds=(0, 1), conditions=("pan_shake",))
    assert json.dumps(a, allow_nan=False, sort_keys=True) == json.dumps(b, allow_nan=False, sort_keys=True)


def test_fixture_is_deterministic_per_seed_and_varies_across_seeds():
    one = A.synthetic_translation_fixture(seed=0, condition="pan_shake")
    again = A.synthetic_translation_fixture(seed=0, condition="pan_shake")
    np.testing.assert_array_equal(one["observed"], again["observed"])
    other = A.synthetic_translation_fixture(seed=1, condition="pan_shake")
    assert not np.allclose(one["observed"], other["observed"])
    # The intended pan is seed-independent ground truth; only the shake realization changes.
    np.testing.assert_array_equal(one["pan"], other["pan"])


# --------------------------------------------------------------------------------------
# JSON safety and path-freedom
# --------------------------------------------------------------------------------------

def test_report_is_strictly_json_native_and_path_free():
    report = A.run_audit(seeds=(0,), conditions=("static", "pan_shake"))
    _assert_json_native(report)
    text = json.dumps(report, allow_nan=False, sort_keys=True)
    assert "NaN" not in text and "Infinity" not in text
    for s in _all_strings(report):
        assert not s.startswith("/")
        assert "bench/" not in s
        assert ".py" not in s
        assert ".json" not in s
        assert ".npz" not in s


# --------------------------------------------------------------------------------------
# explicit fixed parameter grids
# --------------------------------------------------------------------------------------

def test_report_records_the_exact_fixed_grids():
    report = A.run_audit(seeds=(0,), conditions=("pan_shake",))
    grids = report["grids"]
    assert grids["multiscale_scales_seconds"] == list(A.MULTISCALE_SCALES_SECONDS)
    assert grids["multiscale_candidates"] == [
        {"name": name, "retention_weights": list(weights)}
        for name, weights in A.MULTISCALE_CANDIDATES
    ]
    assert grids["l1_budgets"] == list(A.L1_BUDGETS)
    assert grids["l1_weights"] == list(A.L1_WEIGHTS)
    assert grids["l1_fidelity"] == A.L1_FIDELITY
    assert grids["l1_window"] == A.L1_WINDOW
    assert grids["l1_overlap"] == A.L1_OVERLAP

    expected = [("multiscale", name) for name, _ in A.MULTISCALE_CANDIDATES]
    expected += [("l1_budget", f"budget_{budget:g}") for budget in A.L1_BUDGETS]
    for run in report["runs"]:
        assert [(v["family"], v["name"]) for v in run["variants"]] == expected
    for variant in report["runs"][0]["variants"]:
        if variant["family"] == "multiscale":
            assert variant["parameters"]["scales_seconds"] == list(A.MULTISCALE_SCALES_SECONDS)
        else:
            assert variant["parameters"]["budget"] in list(A.L1_BUDGETS)


# --------------------------------------------------------------------------------------
# controls: static / no shake
# --------------------------------------------------------------------------------------

def test_static_control_has_no_shake_and_no_spurious_jitter():
    report = A.run_audit(seeds=(0, 1), conditions=("static",))
    assert len(report["runs"]) == 2
    for run in report["runs"]:
        assert run["condition"] == "static"
        assert run["fixture"]["observed_rms"] == pytest.approx(0.0, abs=1e-12)
        assert run["fixture"]["shake_rms"] == pytest.approx(0.0, abs=1e-12)
        for variant in run["variants"]:
            m = variant["metrics"]
            # A static path has zero second difference of the residual for any constant output.
            assert m["rms_second_difference_residual"] == pytest.approx(0.0, abs=1e-9)
            if variant["family"] == "multiscale":
                assert m["rms_path_deviation"] == pytest.approx(0.0, abs=1e-9)
                assert m["max_abs_correction_from_observed"] == pytest.approx(0.0, abs=1e-9)
            else:
                # The L1 fidelity term anchors the otherwise unidentifiable constant offset to the observed path.
                assert m["rms_path_deviation"] == pytest.approx(0.0, abs=1e-9)
                assert m["max_abs_correction_from_observed"] == pytest.approx(0.0, abs=1e-9)


def test_pan_only_control_tracks_the_pan_within_budget():
    report = A.run_audit(seeds=(0,), conditions=("pan_no_shake",))
    run = report["runs"][0]
    assert run["fixture"]["shake_rms"] == pytest.approx(0.0, abs=1e-12)
    assert run["fixture"]["pan_rms"] > 0.0
    for variant in run["variants"]:
        budget = variant["parameters"].get("budget")
        if budget is not None:
            assert variant["metrics"]["max_abs_correction_from_observed"] <= budget + 1e-9


def test_multiple_seeds_and_conditions_are_reported_separately():
    rep = A.run_audit(seeds=(0, 1), conditions=("static", "pan_shake"))
    combos = {(run["condition"], run["seed"]) for run in rep["runs"]}
    assert combos == {("static", 0), ("static", 1), ("pan_shake", 0), ("pan_shake", 1)}
    assert len(rep["runs"]) == 4


# --------------------------------------------------------------------------------------
# malformed inputs
# --------------------------------------------------------------------------------------

def test_run_audit_rejects_malformed_parameters():
    with pytest.raises(ValueError):
        A.run_audit(n=0)
    with pytest.raises(ValueError):
        A.run_audit(n=4)
    with pytest.raises(ValueError):
        A.run_audit(n=180.5)
    with pytest.raises(ValueError):
        A.run_audit(fps=0.0)
    with pytest.raises(ValueError):
        A.run_audit(fps=float("nan"))
    with pytest.raises(ValueError):
        A.run_audit(seeds="0,1")
    with pytest.raises(ValueError):
        A.run_audit(seeds=())
    with pytest.raises(ValueError):
        A.run_audit(seeds=(0.5,))
    with pytest.raises(ValueError):
        A.run_audit(seeds=(0, 0))
    with pytest.raises(ValueError):
        A.run_audit(conditions=("bogus",))
    with pytest.raises(ValueError):
        A.run_audit(conditions=())


def test_fixture_rejects_malformed_arguments():
    with pytest.raises(ValueError):
        A.synthetic_translation_fixture(n=1)
    with pytest.raises(ValueError):
        A.synthetic_translation_fixture(fps=-1.0)
    with pytest.raises(ValueError):
        A.synthetic_translation_fixture(seed=0.5)
    with pytest.raises(ValueError):
        A.synthetic_translation_fixture(condition="not_a_condition")


def test_metric_rejects_malformed_paths():
    ok = np.zeros((6, 2))
    with pytest.raises(ValueError):
        A.translation_path_metrics(np.zeros(6), ok, ok)
    with pytest.raises(ValueError):
        A.translation_path_metrics(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError):
        A.translation_path_metrics(ok, ok, np.zeros((6, 3)))
    with pytest.raises(ValueError):
        A.translation_path_metrics(ok, ok, np.full((6, 2), np.nan))


def test_candidate_builders_reject_bad_parameters():
    signal = np.zeros((12, 2))
    times = np.arange(12) / 60.0
    with pytest.raises(ValueError):
        A.build_l1_output(signal, 0.0)
    with pytest.raises(ValueError):
        A.build_l1_output(signal, -1.0)
    with pytest.raises(ValueError):
        A.build_l1_output(signal, 0.1, window=0)
    with pytest.raises(ValueError):
        A.build_l1_output(signal, 0.1, window=2.5)
    with pytest.raises(ValueError):
        A.build_l1_output(signal, 0.1, window=5, overlap=5)
    with pytest.raises(ValueError):
        A.build_l1_output(signal, 0.1, fidelity=-0.1)
    with pytest.raises(ValueError):
        A.build_multiscale_output(signal, times, (0.0, 0.0))          # wrong number of weights
    with pytest.raises(ValueError):
        A.build_multiscale_output(signal, np.zeros(12), (0.0, 0.0, 0.0))   # non-monotonic times


# --------------------------------------------------------------------------------------
# no ranking / winner / score fields
# --------------------------------------------------------------------------------------

def test_report_has_no_ranking_or_winner_fields():
    report = A.run_audit(seeds=(0,), conditions=("pan_shake",))
    banned = ("winner", "rank", "best", "score", "superior", "quality", "aggregate")
    for key in _all_keys(report):
        low = key.lower()
        assert not any(b in low for b in banned), f"ranking-like key present: {key!r}"
    # No cross-variant ordering index either.
    for run in report["runs"]:
        for variant in run["variants"]:
            assert "index" not in variant and "position" not in variant
    assert "reporting_policy" in report
    assert "aggregate winner" in report["reporting_policy"].lower()


def test_scope_note_states_the_boundaries():
    note = A.SCOPE_NOTE.lower()
    for term in ("translation path", "rendered image quality", "rotation", "depth", "subject motion",
                 "border support", "human preference"):
        assert term in note
    assert "translation path" in (A.__doc__ or "").lower()


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def test_cli_prints_json_report_to_stdout(capsys):
    assert A.main(["--seeds", "0", "--conditions", "static"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == A.SCHEMA and report["ok"] is True
    assert len(report["runs"]) == 1


def test_cli_writes_json_report_to_out_file(tmp_path):
    out = tmp_path / "audit.json"
    assert A.main(["--seeds", "0", "--conditions", "pan_shake", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema"] == A.SCHEMA and report["ok"] is True
    assert {run["condition"] for run in report["runs"]} == {"pan_shake"}


def test_cli_emits_json_safe_error_for_bad_condition(capsys):
    assert A.main(["--seeds", "0", "--conditions", "bogus"]) == 2
    err = json.loads(capsys.readouterr().out)
    assert err["schema"] == A.SCHEMA and err["ok"] is False
    json.dumps(err, allow_nan=False)


def test_cli_emits_json_safe_error_for_bad_n(capsys):
    assert A.main(["--seeds", "0", "--n", "2"]) == 2
    err = json.loads(capsys.readouterr().out)
    assert err["ok"] is False


def test_parsers_validate_their_input():
    assert A.parse_seeds("0,1,2") == (0, 1, 2)
    assert A.parse_conditions("static,pan_shake") == ("static", "pan_shake")
    with pytest.raises(ValueError):
        A.parse_seeds("a,b")
    with pytest.raises(ValueError):
        A.parse_seeds("")
    with pytest.raises(ValueError):
        A.parse_conditions("bogus")
