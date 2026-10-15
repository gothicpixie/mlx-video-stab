"""Focused CPU-only tests for the E02 rigid-line shape audit (``bench/shape_audit.py``).

Analytic scope: no media, FFmpeg, MLX kernels, production wiring or changed defaults. This module imports the
audit/candidate modules (and thus ``mlxstab`` transitively) but exercises only NumPy geometry, so it can run
without holding the shared heavy lock.
"""
import json

import numpy as np
import pytest

from bench import composition_audit as CA
from bench import composition_candidate as CC
from bench import shape_audit as SA
from mlxstab import geometry as G

W, H = 64, 48


# --------------------------------------------------------------------------------------
# metric
# --------------------------------------------------------------------------------------

def test_metric_straight_line_is_zero():
    r = SA.best_fit_line(SA.sample_line([0.0, 0.0], [10.0, 0.0], 51))
    assert r["ok"] and r["dev_max_px"] < 1e-9 and r["dev_p95_px"] < 1e-9
    assert r["span_px"] == pytest.approx(10.0)


def test_metric_detects_bowed_line():
    pts = SA.sample_line([0.0, 0.0], [10.0, 0.0], 101)
    pts[:, 1] += 2.0 * np.sin(np.pi * np.arange(len(pts)) / (len(pts) - 1))
    r = SA.best_fit_line(pts)
    assert r["ok"] and r["dev_max_px"] > 0.5


def test_metric_rejects_nonfinite_coordinates():
    pts = SA.sample_line([0.0, 0.0], [10.0, 0.0], 21); pts[5, 1] = np.nan
    r = SA.best_fit_line(pts)
    assert not r["ok"] and "nonfinite_coordinates" in r["reasons"]
    assert r["dev_max_px"] is None and r["span_px"] is None and r["nonfinite_frac"] == pytest.approx(1 / 21)


def test_metric_rejects_degenerate_span():
    r = SA.best_fit_line(np.zeros((10, 2)))
    assert not r["ok"] and "degenerate_span" in r["reasons"]
    assert r["span_px"] == pytest.approx(0.0) and r["dev_max_px"] is None


def test_metric_rejects_too_few_points():
    r = SA.best_fit_line(np.zeros((1, 2)))
    assert not r["ok"] and "fewer_than_two_points" in r["reasons"]
    assert r["dev_max_px"] is None


def test_metric_rejects_malformed_input_shape():
    with pytest.raises(ValueError):
        SA.best_fit_line(np.zeros((5, 3)))
    with pytest.raises(ValueError):
        SA.map_line_deviation(lambda p: np.zeros((len(p), 3)), [0.0, 0.0], [10.0, 0.0], 11)


def test_metric_rejects_collapsed_mapped_line():
    r = SA.map_line_deviation(lambda p: np.zeros_like(p), [0.0, 0.0], [10.0, 0.0], 21)
    assert not r["ok"] and "degenerate_span" in r["reasons"]
    assert r["dev_max_px"] is None and r["dev_p95_px"] is None


def test_matrix_composition_equals_sequential_application():
    P0, P1 = SA.projective(7.5e-3, -4.5e-3), SA.projective(-6.0e-3, 6.0e-3)
    pts = SA.sample_line([1.0, 2.0], [50.0, 30.0], 37)
    np.testing.assert_allclose(G.apply_h(P1 @ P0, pts), G.apply_h(P1, G.apply_h(P0, pts)), atol=1e-9)


# --------------------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------------------

def test_identity_and_projective_controls_stay_straight():
    c = SA.metric_controls(n=401)
    assert c["identity"]["deviation"]["ok"] and c["identity"]["deviation"]["dev_max_px"] < 1e-9
    assert c["projective_homography"]["deviation"]["ok"]
    assert c["projective_homography"]["deviation"]["dev_max_px"] < 1e-9


def test_bowed_field_sensitivity_control_is_detected():
    c = SA.metric_controls(n=401)
    b = c["bowed_field"]["deviation"]
    assert b["ok"] and b["dev_max_px"] > 1.0  # measured ~3.26 px for a 6 px sine bow


def test_pipeline_projective_control_stays_straight():
    p = SA.pipeline_projective_control(n=401)
    assert p["fixed_grid"]["ok"] and p["fixed_grid"]["dev_max_px"] < 1e-9
    assert p["candidate"]["ok"] and p["candidate"]["dev_max_px"] < 1e-9


# --------------------------------------------------------------------------------------
# two-step analytic fixture: candidate composes, fixed grid sums
# --------------------------------------------------------------------------------------

def test_two_step_fixture_ground_truth_is_straight():
    f = SA.two_step_projective_fixture(n=401)
    assert f["ground_truth"]["ok"] and f["ground_truth"]["dev_max_px"] < 1e-9


def test_two_step_fixture_candidate_straight_and_fixed_grid_bends():
    f = SA.two_step_projective_fixture(n=401)
    cand, naive = f["candidate"], f["fixed_grid"]
    assert cand["ok"] and naive["ok"]
    # Candidate straightness is bounded by its measured mesh discretization vs the true composition.
    assert cand["dev_max_px"] < 0.01
    assert cand["dev_max_px"] <= f["candidate_vs_ground_truth_max_px"]
    # Fixed grid demonstrably bends more (measured ~1.03 px vs ~0.0018 px).
    assert naive["dev_max_px"] > 0.5
    assert naive["dev_max_px"] > 100.0 * cand["dev_max_px"]
    # And it is far from the true line pointwise, while the candidate is close.
    assert f["candidate_vs_ground_truth_max_px"] < 0.05
    assert f["fixed_grid_vs_ground_truth_max_px"] > 1.0
    assert f["fixed_grid_vs_ground_truth_max_px"] > 100.0 * f["candidate_vs_ground_truth_max_px"]


def test_two_step_time_series_temporal_variation():
    f = SA.two_step_projective_fixture(n=401)
    ts = f["time_series"]
    assert ts["frames"] == [0, 1, 2]
    for key in ("ground_truth_dev_max_px", "candidate_dev_max_px", "fixed_grid_dev_max_px"):
        assert len(ts[key]) == 3 and all(v is not None for v in ts[key])
    assert max(ts["ground_truth_dev_max_px"]) < 1e-9
    assert ts["candidate_dev_max_px"][0] < 1e-9 and ts["fixed_grid_dev_max_px"][0] < 1e-9
    # Candidate temporal variation is bounded by its measured mesh discretization error.
    assert ts["candidate_max_step_px"] <= ts["candidate_mesh_discretization_max_px"]
    # The fixed-grid jump appears only at frame 2 and exceeds the candidate control.
    assert ts["fixed_grid_dev_max_px"][1] < 0.01
    assert ts["fixed_grid_max_step_px"] > 0.5
    assert ts["fixed_grid_max_step_px"] > 100.0 * ts["candidate_max_step_px"]


def test_candidate_field_matches_point_form_on_the_grid():
    P0, P1 = SA.projective(7.5e-3, -4.5e-3), SA.projective(-6.0e-3, 6.0e-3)
    pairs = [CA.PairMotion(np.eye(3), SA._residual_mesh(P0, W, H, 16, 12)),
             CA.PairMotion(np.eye(3), SA._residual_mesh(P1, W, H, 16, 12))]
    _, plan, cfg, mot = SA._plan_and_motion(pairs, W, H, 3, 16, 12, "t", "t")
    field = CC.candidate_field(mot, plan, cfg, 2, highpass=False).reshape(-1, 2)
    point = CC.candidate_source_coords(mot, plan, cfg, 2, CC.output_grid(W, H), highpass=False)
    np.testing.assert_allclose(field, point, atol=0.0)


# --------------------------------------------------------------------------------------
# report and CLI
# --------------------------------------------------------------------------------------

def test_report_is_json_safe_and_self_describing(tmp_path):
    rep = SA.run_shape_audit(out_dir=tmp_path, n=201)
    json.dumps(SA.json_safe(rep), allow_nan=False)
    on_disk = json.loads((tmp_path / "shape_audit.json").read_text())
    assert on_disk["schema"] == SA.SCHEMA
    assert "analytic" in on_disk["scope"].lower()
    assert "media" in on_disk["scope"].lower()
    fi = on_disk["fixture"]
    assert fi["config"]["pairs"] == 2 and fi["config"]["mesh_cells"] == [48, 36]
    assert fi["ground_truth"]["ok"] is True
    assert rep["conclusion"]["candidate_straighter_than_fixed_grid"] is True


def test_cli_main_writes_json(tmp_path):
    assert SA.main(["--out-dir", str(tmp_path), "--samples", "101"]) == 0
    assert (tmp_path / "shape_audit.json").exists()


def test_report_is_json_safe_when_a_measured_line_is_invalid(monkeypatch):
    bad = {"ok": False, "dev_max_px": None, "dev_p95_px": None, "reasons": ["nonfinite_coordinates"]}
    stub = {
        "id": "stub", "description": "invalid", "config": {}, "globals": {}, "line": {},
        "time_series": {
            "frames": [0, 1, 2],
            "ground_truth_dev_max_px": [None, None, None],
            "candidate_dev_max_px": [None, None, None],
            "fixed_grid_dev_max_px": [None, None, None],
            "ground_truth_max_step_px": None, "candidate_max_step_px": None,
            "fixed_grid_max_step_px": None, "candidate_mesh_discretization_max_px": None,
        },
        "ground_truth": bad, "candidate": bad, "fixed_grid": bad,
        "candidate_vs_ground_truth_max_px": None, "fixed_grid_vs_ground_truth_max_px": None,
    }
    monkeypatch.setattr(SA, "two_step_projective_fixture", lambda **k: stub)
    rep = SA.run_shape_audit(n=51)
    c = rep["conclusion"]
    assert c["valid"] is False and c["separation"] == "unknown"
    assert c["candidate_straighter_than_fixed_grid"] is None
    assert c["candidate_temporal_variation_bounded"] is None
    assert c["fixed_grid_temporal_variation_exceeds_control"] is None
    assert "final_frame_line_invalid" in c["reasons"] and "time_series_line_invalid" in c["reasons"]
    json.dumps(SA.json_safe(rep), allow_nan=False)  # does not raise on None values
