"""Focused, deterministic tests for the offline E04 observable-signal audit.

Covers the dependency-free Spearman rank correlation (average ties, constant vectors, too-few points), the
manifest structure/success/budget-scan refusals, insufficient-history and null-signal skips, method/direction
grouping, the cross-sectional-by-iteration view, deterministic JSON-safe output, and the CLI (stdout and an
atomic ``--out``). No media, FFmpeg, MLX kernels, RAFT weights or production wiring; no default change.
"""
import json

import pytest

from bench import flow_prediction_observable_audit as A


# --------------------------------------------------------------------------------------
# rank correlation
# --------------------------------------------------------------------------------------

def test_average_ranks_handles_ties_and_preserves_order():
    assert A.average_ranks([1.0, 2.0, 2.0, 3.0]) == [1.0, 2.5, 2.5, 4.0]
    assert A.average_ranks([5.0, 5.0, 1.0]) == [2.5, 2.5, 1.0]
    # unsorted input is not mutated
    x = [3.0, 1.0, 2.0]
    assert A.average_ranks(x) == [3.0, 1.0, 2.0] and x == [3.0, 1.0, 2.0]


def test_spearman_matches_manual_value_with_average_ties():
    # x = [1,2,2,3] -> ranks [1, 2.5, 2.5, 4]; y = [1,2,3,4] -> ranks [1,2,3,4]
    value = A.spearman([1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    assert value == pytest.approx(4.5 / (4.5 * 5.0) ** 0.5, abs=1e-12)


def test_spearman_is_one_for_monotone_but_not_linear_pairs():
    assert A.spearman([1.0, 2.0, 3.0], [10.0, 20.0, 500.0]) == pytest.approx(1.0)
    assert A.spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    # both axes constant -> the correlation is undefined, not a perfect +1
    assert A.spearman([4.0, 4.0, 4.0, 4.0], [2.0, 2.0, 2.0, 2.0]) is None


def test_spearman_ties_that_are_monotone_are_one():
    assert A.spearman([1.0, 1.0, 2.0, 2.0], [5.0, 5.0, 9.0, 9.0]) == pytest.approx(1.0)


def test_spearman_constant_vectors_return_none_with_reason():
    assert A.spearman_with_reason([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == (None, A.STATUS_CONSTANT_SIGNAL)
    assert A.spearman_with_reason([1.0, 2.0, 3.0], [7.0, 7.0, 7.0]) == (None, A.STATUS_CONSTANT_EPE)
    assert A.spearman_with_reason([2.0, 2.0], [2.0, 2.0]) == (None, A.STATUS_CONSTANT_SIGNAL)


def test_spearman_too_few_points_and_length_mismatch():
    assert A.spearman_with_reason([], []) == (None, A.STATUS_INSUFFICIENT_POINTS)
    assert A.spearman_with_reason([1.0], [2.0]) == (None, A.STATUS_INSUFFICIENT_POINTS)
    with pytest.raises(ValueError):
        A.spearman([1.0, 2.0], [1.0])


def test_is_finite_number_excludes_bool_nan_and_inf():
    assert A._is_finite_number(1) and A._is_finite_number(0.5)
    assert not A._is_finite_number(None) and not A._is_finite_number(True)
    assert not A._is_finite_number(float("nan")) and not A._is_finite_number(float("inf"))


# --------------------------------------------------------------------------------------
# manifest fixtures
# --------------------------------------------------------------------------------------

def scan_row(it, epe, update, photo, *, valid_frac=1.0, texture_frac=0.5,
             photo_cost=0.002, signal_cost=0.0001):
    return {
        "iter": it,
        "epe_low": {"mean": epe},
        "flow_update_low": {"mean": update},
        "photometric_low": {"resid_textured": {"median": photo}, "valid_frac": valid_frac,
                            "texture_frac": texture_frac},
        "photometric_check_s": photo_cost,
        "update_mag_check_s": signal_cost,
    }


def trajectory(target, *, reason=None, rows=None):
    if rows is None:
        # Within a trajectory the signal and EPE decrease monotonically together (+1). The target offset also
        # spreads both axes across trajectories so the cross-sectional view is defined, not constant.
        off = 0.01 * float(target)
        rows = [scan_row(1, 0.30 + off, 0.40 + 2 * off, 3.0 + off),
                scan_row(2, 0.20 + off, 0.20 + 2 * off, 2.0 + off),
                scan_row(3, 0.10 + off, 0.05 + 2 * off, 1.0 + off)]
    return {"target": target, "reason": reason, "fallback": reason is not None, "scan": rows}


def make_manifest(results, *, status="success", enabled=True, schema=A.MANIFEST_SCHEMA):
    return {
        "schema": schema,
        "status": status,
        "stage": status,
        "run_id": "unit",
        "created": "2026-01-01T00:00:00-0000",
        "budget_scan": {"enabled": enabled, "max_iters": 3, "history_iters": 2,
                        "thresholds": [0.01], "oracle": True},
        "source": {"sha256": "sourcehash", "frames_sha256": "frameshash",
                   "window": {"start": 0.0, "duration": 1.0}, "fps": 30.0},
        "weights": {"sha256": "weightshash"},
        "config_digest": "digest",
        "config": {"iters": 2, "ref_iters": 3, "downscale": 2,
                   "methods": sorted(results.get("forward", {})), "directions": sorted(results),
                   "budget": {"max_iters": 3, "thresholds": [0.01], "scan": True}},
        "code": {"revision": "rev", "dirty_relevant": False},
        "results": results,
    }


def write_manifest(tmp_path, doc, name="manifest.json", *, allow_nan=False):
    path = tmp_path / name
    path.write_text(json.dumps(doc, allow_nan=allow_nan), encoding="utf-8")
    return path


def default_results():
    return {"forward": {"m1": {"per_pair": [trajectory(0), trajectory(1)]}},
            "backward": {"m1": {"per_pair": [trajectory(0), trajectory(1)]}}}


# --------------------------------------------------------------------------------------
# validation refusals
# --------------------------------------------------------------------------------------

def test_audit_ok_on_minimal_manifest(tmp_path):
    path = write_manifest(tmp_path, make_manifest(default_results()))
    case = A.audit_manifest(path)
    assert case["ok"] and case["coverage"]["trajectories_total"] == 4
    assert case["coverage"]["trajectories_usable"] == 4
    base = case["signals"]["flow_update_low.mean"]
    assert base["trajectories_valid"] == 4 and base["status_counts"] == {"ok": 4}


def test_missing_file_is_refused(tmp_path):
    with pytest.raises(A.ObservableAuditError) as exc:
        A.load_manifest(tmp_path / "nope.json")
    assert exc.value.code == "not_found"


def test_invalid_json_and_non_object_are_refused(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(A.ObservableAuditError) as exc:
        A.load_manifest(bad)
    assert exc.value.code == "not_json"
    arr = tmp_path / "arr.json"
    arr.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(A.ObservableAuditError) as exc:
        A.load_manifest(arr)
    assert exc.value.code == "not_object"


def test_wrong_schema_and_unsuccessful_status_are_refused(tmp_path):
    p = write_manifest(tmp_path, make_manifest(default_results(), schema="mlxstab.other/9"), name="s.json")
    with pytest.raises(A.ObservableAuditError) as exc:
        A.load_manifest(p)
    assert exc.value.code == "bad_schema"
    p2 = write_manifest(tmp_path, make_manifest(default_results(), status="failure"), name="f.json")
    with pytest.raises(A.ObservableAuditError) as exc:
        A.load_manifest(p2)
    assert exc.value.code == "not_success"


def test_disabled_budget_scan_is_refused(tmp_path):
    p = write_manifest(tmp_path, make_manifest(default_results(), enabled=False))
    with pytest.raises(A.ObservableAuditError) as exc:
        A.load_manifest(p)
    assert exc.value.code == "budget_scan_disabled"


def test_missing_results_and_per_pair_are_refused(tmp_path):
    doc = make_manifest(default_results())
    del doc["results"]
    with pytest.raises(A.ObservableAuditError) as exc:
        A.load_manifest(write_manifest(tmp_path, doc, name="r.json"))
    assert exc.value.code == "missing_results"

    doc2 = make_manifest({"forward": {"m1": {"summary": {}}}})
    with pytest.raises(A.ObservableAuditError) as exc:
        A.load_manifest(write_manifest(tmp_path, doc2, name="pp.json"))
    assert exc.value.code == "missing_per_pair"


def test_run_marks_refused_manifest_and_keeps_going(tmp_path):
    good = write_manifest(tmp_path, make_manifest(default_results()), name="good.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    report = A.run_observable_audit([str(good), str(bad)])
    assert report["ok"] is False and len(report["cases"]) == 2
    assert report["cases"][0]["ok"] is True
    assert report["cases"][1]["ok"] is False and report["cases"][1]["error"]["code"] == "bad_schema"


def test_unknown_signal_is_refused():
    with pytest.raises(ValueError):
        A.run_observable_audit([], signals=("nope",))


# --------------------------------------------------------------------------------------
# skips / null coverage
# --------------------------------------------------------------------------------------

def test_insufficient_history_is_excluded_and_counted(tmp_path):
    results = {"forward": {"m1": {"per_pair": [
        trajectory(0, reason="insufficient_history"),
        trajectory(1),
    ]}}}
    path = write_manifest(tmp_path, make_manifest(results))
    case = A.audit_manifest(path)
    assert case["coverage"]["trajectories_insufficient_history"] == 1
    assert case["coverage"]["trajectories_usable"] == 1
    base = case["signals"]["flow_update_low.mean"]
    assert base["status_counts"] == {"insufficient_history": 1, "ok": 1}
    assert base["trajectories_valid"] == 1 and base["trajectories_usable"] == 1


def test_all_null_textured_signal_marks_null_signal_but_update_still_ok(tmp_path):
    rows = [scan_row(1, 0.3, 0.4, None), scan_row(2, 0.2, 0.2, None), scan_row(3, 0.1, 0.05, None)]
    results = {"forward": {"m1": {"per_pair": [trajectory(0, rows=rows)]}}}
    case = A.audit_manifest(write_manifest(tmp_path, make_manifest(results)))
    photo = case["signals"]["photometric_low.resid_textured.median"]
    assert photo["status_counts"] == {"null_signal": 1}
    assert photo["n_null_signal_samples"] == 3 and photo["trajectories_valid"] == 0
    update = case["signals"]["flow_update_low.mean"]
    assert update["status_counts"] == {"ok": 1}


def test_non_finite_signal_values_are_treated_as_null(tmp_path):
    rows = [scan_row(1, 0.3, float("nan"), 3.0), scan_row(2, 0.2, float("inf"), 2.0)]
    case = A.audit_manifest(write_manifest(
        tmp_path, make_manifest({"forward": {"m1": {"per_pair": [trajectory(0, rows=rows)]}}}),
        allow_nan=True))
    update = case["signals"]["flow_update_low.mean"]
    assert update["status_counts"] == {"null_signal": 1} and update["n_null_signal_samples"] == 2


def test_constant_and_insufficient_trajectories_are_marked(tmp_path):
    constant = [scan_row(1, 0.3, 0.2, 5.0), scan_row(2, 0.2, 0.2, 4.0), scan_row(3, 0.1, 0.2, 3.0)]
    one_row = [scan_row(1, 0.3, 0.4, 3.0)]
    results = {"forward": {"m1": {"per_pair": [
        trajectory(0, rows=constant), trajectory(1, rows=one_row), trajectory(2, rows=[])]}}}
    case = A.audit_manifest(write_manifest(tmp_path, make_manifest(results)))
    update = case["signals"]["flow_update_low.mean"]
    assert update["status_counts"] == {"constant_signal": 1, "insufficient_points": 1, "no_scan_rows": 1}
    # A constant EPE axis is reported separately from a constant signal axis.
    flat_epe = [scan_row(1, 0.2, 0.3, 5.0), scan_row(2, 0.2, 0.2, 4.0)]
    case2 = A.audit_manifest(write_manifest(
        tmp_path, make_manifest({"forward": {"m1": {"per_pair": [trajectory(0, rows=flat_epe)]}}}),
        name="m2.json"))
    assert case2["signals"]["flow_update_low.mean"]["status_counts"] == {"constant_epe": 1}


def test_coverage_reports_median_fractions_and_costs(tmp_path):
    rows = [scan_row(1, 0.3, 0.4, 3.0, valid_frac=0.9, texture_frac=0.2, photo_cost=0.004, signal_cost=0.0002),
            scan_row(2, 0.2, 0.2, 2.0, valid_frac=1.0, texture_frac=0.4, photo_cost=0.006, signal_cost=0.0004)]
    case = A.audit_manifest(write_manifest(
        tmp_path, make_manifest({"forward": {"m1": {"per_pair": [trajectory(0, rows=rows)]}}})))
    cov = case["coverage"]
    assert cov["valid_frac_median"] == pytest.approx(0.95)
    assert cov["texture_frac_median"] == pytest.approx(0.3)
    assert cov["photometric_check_s_mean"] == pytest.approx(0.005)
    assert cov["signal_check_s_mean"] == pytest.approx(0.0003)
    assert cov["photometric_check_s_total"] == pytest.approx(0.010)
    # Budget rows carry no cycle fields, so the cycle-specific coverage entries stay explicit nulls.
    assert cov["cycle_valid_frac_median"] is None
    assert cov["cycle_valid_frac_min"] is None
    assert cov["cycle_check_s_mean"] is None
    assert cov["cycle_check_s_total"] is None


# --------------------------------------------------------------------------------------
# grouping, cross-sectional view, determinism
# --------------------------------------------------------------------------------------

def test_method_and_direction_stratification(tmp_path):
    results = {
        "forward": {"m1": {"per_pair": [trajectory(0)]}, "m2": {"per_pair": [trajectory(0)]}},
        "backward": {"m1": {"per_pair": [trajectory(0)]}},
    }
    case = A.audit_manifest(write_manifest(tmp_path, make_manifest(results)))
    base = case["signals"]["flow_update_low.mean"]
    assert set(base["by_method"]) == {"m1", "m2"}
    assert set(base["by_direction"]) == {"backward", "forward"}
    assert base["by_method"]["m1"]["n"] == 2 and base["by_direction"]["backward"]["n"] == 1
    order = [(r["direction"], r["method"], r["target"]) for r in base["trajectories"]]
    assert order == sorted(order)


def test_cross_sectional_by_iteration_is_reported(tmp_path):
    case = A.audit_manifest(write_manifest(tmp_path, make_manifest(default_results())))
    base = case["signals"]["flow_update_low.mean"]
    cs = base["cross_sectional_by_iteration"]
    assert [row["iter"] for row in cs] == [1, 2, 3]
    assert all(row["n"] == 4 and row["status"] == "ok" for row in cs)
    assert cs[0]["spearman"] == pytest.approx(1.0)


def test_positive_fraction_and_median_over_mixed_trajectories(tmp_path):
    up = [scan_row(1, 0.3, 0.1, 3.0), scan_row(2, 0.2, 0.2, 2.0), scan_row(3, 0.1, 0.3, 1.0)]  # negative
    down = [scan_row(1, 0.3, 0.3, 3.0), scan_row(2, 0.2, 0.2, 2.0), scan_row(3, 0.1, 0.1, 1.0)]  # positive
    results = {"forward": {"m1": {"per_pair": [trajectory(0, rows=up), trajectory(1, rows=down)]}}}
    base = A.audit_manifest(write_manifest(tmp_path, make_manifest(results)))["signals"]["flow_update_low.mean"]
    assert base["spearman_median"] == pytest.approx(0.0)
    assert base["positive_fraction"] == pytest.approx(0.5)


def test_report_is_deterministic_and_json_safe(tmp_path):
    path = write_manifest(tmp_path, make_manifest(default_results()))
    first = A.run_observable_audit([str(path)])
    second = A.run_observable_audit([str(path)])
    assert json.dumps(A.json_safe(first), sort_keys=True) == json.dumps(A.json_safe(second), sort_keys=True)
    text = json.dumps(A.json_safe(first), indent=1, sort_keys=True, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def test_cli_defaults_to_stdout(tmp_path, capsys):
    path = write_manifest(tmp_path, make_manifest(default_results()))
    assert A.main([str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == A.SCHEMA and payload["ok"] is True


def test_cli_writes_atomically_and_matches_in_memory(tmp_path):
    path = write_manifest(tmp_path, make_manifest(default_results()))
    out = tmp_path / "report.json"
    assert A.main([str(path), "--out", str(out)]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    expected = A.json_safe(A.run_observable_audit([str(path)]))
    assert written == expected
    assert not list(tmp_path.glob("report.json.*.tmp"))  # temp file was replaced, not left behind


def test_cli_reports_failure_with_exit_code_two(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert A.main([str(bad)]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False and payload["cases"][0]["error"]["code"] == "bad_schema"


def test_cli_signal_selection(tmp_path, capsys):
    path = write_manifest(tmp_path, make_manifest(default_results()))
    assert A.main([str(path), "--signal", "flow_update_low.mean"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["signals"] == ["flow_update_low.mean"]
    assert set(payload["cases"][0]["signals"]) == {"flow_update_low.mean"}


# --------------------------------------------------------------------------------------
# cold-only bidirectional cycle-consistency signal (cycle_scan manifests)
# --------------------------------------------------------------------------------------

def cycle_row(it, epe, cyc, valid_frac=1.0, cycle_cost=0.002):
    return {"iter": it, "epe_low": {"mean": epe},
            "cycle_low": {"resid": {"mean": cyc}, "valid_frac": valid_frac},
            "cycle_check_s": cycle_cost}


def cycle_results():
    rows = [cycle_row(1, 0.30, 0.9), cycle_row(2, 0.20, 0.5), cycle_row(3, 0.10, 0.2)]
    return {"bidirectional": {"cold": {"per_pair": [
        {"target": 0, "reason": None, "fallback": False, "scan": rows}]}}}


def make_cycle_manifest(results, *, enabled=True, status="success"):
    return {
        "schema": A.MANIFEST_SCHEMA, "status": status, "stage": status, "run_id": "cycle-unit",
        "created": "2026-01-01T00:00:00-0000",
        "cycle_scan": {"enabled": enabled, "kind": "cold_bidirectional_cycle_consistency",
                       "direction": "bidirectional", "method": "cold", "max_iters": 3,
                       "results": results},
        "source": {}, "weights": {}, "config_digest": "digest",
        "config": {"iters": 12, "ref_iters": 24, "downscale": 2, "methods": [], "directions": [],
                   "cycle": {"max_iters": 3, "scan": True}},
        "code": {"revision": "rev", "dirty_relevant": False},
        "results": {},
    }


def test_cycle_signal_registered_without_changing_defaults():
    assert "cycle_low.resid.mean" in A.SIGNALS
    # the default report stays the two budget-scan signals; cycle is explicit opt-in
    assert A.DEFAULT_SIGNALS == ("flow_update_low.mean", "photometric_low.resid_textured.median")


def test_cycle_scan_manifest_is_accepted_and_audited(tmp_path):
    path = write_manifest(tmp_path, make_cycle_manifest(cycle_results()))
    case = A.audit_manifest(path, signals=("cycle_low.resid.mean",))
    assert case["ok"] is True
    assert case["scan"]["kind"] == "cycle_scan"
    assert case["budget_scan"]["kind"] == "cycle_scan"       # backwards-compatible alias
    assert case["coverage"]["trajectories_total"] == 1
    assert case["coverage"]["trajectories_usable"] == 1
    sig = case["signals"]["cycle_low.resid.mean"]
    assert sig["trajectories_valid"] == 1
    assert sig["spearman_median"] == pytest.approx(1.0)      # cycle residual and oracle EPE fall together
    assert sig["status_counts"] == {"ok": 1}


def test_cycle_scan_manifest_reports_runs_without_a_default_signal(tmp_path):
    """Auditing a cycle manifest with the default budget signals yields explicit null-signal skips, not errors."""
    path = write_manifest(tmp_path, make_cycle_manifest(cycle_results()))
    case = A.audit_manifest(path)
    assert case["signals"]["flow_update_low.mean"]["status_counts"] == {"null_signal": 1}
    assert case["signals"]["photometric_low.resid_textured.median"]["status_counts"] == {"null_signal": 1}


def test_cycle_manifest_coverage_reports_cycle_fields_and_null_others(tmp_path):
    """A cycle-only manifest populates the cycle coverage entries and leaves the budget ones null."""
    rows = [cycle_row(1, 0.30, 0.9, valid_frac=0.9, cycle_cost=0.004),
            cycle_row(2, 0.20, 0.5, valid_frac=1.0, cycle_cost=0.006)]
    results = {"bidirectional": {"cold": {"per_pair": [
        {"target": 0, "reason": None, "fallback": False, "scan": rows}]}}}
    case = A.audit_manifest(write_manifest(tmp_path, make_cycle_manifest(results)),
                            signals=("cycle_low.resid.mean",))
    cov = case["coverage"]
    assert cov["scan_rows"] == 2
    assert cov["cycle_valid_frac_median"] == pytest.approx(0.95)
    assert cov["cycle_valid_frac_min"] == pytest.approx(0.9)
    assert cov["cycle_check_s_mean"] == pytest.approx(0.005)
    assert cov["cycle_check_s_total"] == pytest.approx(0.010)
    # Budget-only fields are absent from cycle rows and must stay null, not zero.
    assert cov["valid_frac_median"] is None
    assert cov["texture_frac_median"] is None
    assert cov["signal_check_s_mean"] is None
    assert cov["photometric_check_s_mean"] is None
    assert cov["photometric_check_s_total"] is None


def test_disabled_cycle_scan_without_budget_is_refused(tmp_path):
    doc = make_cycle_manifest(cycle_results(), enabled=False)
    with pytest.raises(A.ObservableAuditError) as exc:
        A.load_manifest(write_manifest(tmp_path, doc, name="c.json"))
    assert exc.value.code == "budget_scan_disabled"


def test_cli_accepts_cycle_signal(tmp_path, capsys):
    path = write_manifest(tmp_path, make_cycle_manifest(cycle_results()))
    assert A.main([str(path), "--signal", "cycle_low.resid.mean"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True and payload["signals"] == ["cycle_low.resid.mean"]
    assert payload["cases"][0]["scan"]["kind"] == "cycle_scan"
