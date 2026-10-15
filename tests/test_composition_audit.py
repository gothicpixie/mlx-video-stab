"""Focused tests for the Stage 0 residual-composition audit (bench/composition_audit.py).

These assert independently derived ground truth and controls, not the current production result: the
counterexample must show the fixed-grid sum is wrong, the rigid/translation-only controls must show it is
exact, and the production path must agree with the independent NumPy replication of ``source_coords``.
The actual ``source_coords`` test uses MLX/Metal and therefore runs under the shared heavy lock with the rest
of the suite.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from bench import composition_audit as CA
from mlxstab import geometry as G


@pytest.fixture(scope="module")
def fixtures():
    return CA.build_fixtures()


def _errors(fx, t):
    return CA.audit_fixture(fx)["errors"][f"frame_{t}"]


# --------------------------------------------------------------------------------------
# independent ground truth
# --------------------------------------------------------------------------------------

def test_rotation_helper_is_clockwise_in_image_coordinates():
    """A positive angle must rotate the helper clockwise (image y down): the point right of centre moves down."""
    W, H = 32, 24
    R = CA.rotation_about_clockwise(W, H, 90.0)
    c = np.array([W / 2 - 0.5, H / 2 - 0.5])
    q = G.apply_h(R, np.array([[c[0] + 4.0, c[1]]]))
    np.testing.assert_allclose(q, np.array([[c[0], c[1] + 4.0]]), atol=1e-9)


def test_oracle_matches_closed_form_rotation_plus_residual(fixtures):
    """Phi_2(x) must equal the closed form R(R(x)+u); a second independent implementation of the truth."""
    fx = fixtures["constant_residual_rotation"]
    W, H, gc, gr = fx.W, fx.H, *fx.mesh_cells
    R = CA.rotation_about_clockwise(W, H, 10.0)
    u = np.array([8.0, 0.0])
    pts = CA.output_grid(W, H)
    oracle = CA.compose_oracle(fx.pairs, 2, pts, W, H)
    closed = G.apply_h(R, G.apply_h(R, pts) + u)
    np.testing.assert_allclose(oracle, closed, atol=1e-9)


def test_transported_recursion_equals_composition_for_affine(fixtures):
    """The affine displacement transport delta recursion must reproduce the composition exactly."""
    for name in ("rigid_rotation_control", "constant_residual_rotation", "spatial_ramp_translation",
                 "parallax_two_layer", "rotation_translation_border"):
        fx = fixtures[name]
        pts = CA.output_grid(fx.W, fx.H)
        for t in range(fx.T):
            np.testing.assert_allclose(CA.transported_field(fx.pairs, t, pts, fx.W, fx.H),
                                       CA.compose_oracle(fx.pairs, t, pts, fx.W, fx.H), atol=1e-9,
                                       err_msg=f"{name} frame {t}")


def test_plan_globals_match_the_analytic_composition(fixtures):
    """The pinned production plan's global part is the true composition of the per-pair homographies."""
    for fx in fixtures.values():
        plan, _ = CA.build_plan(fx)
        check = CA._fixture_plan_check(fx, plan)
        assert max(check.values()) < 1e-9, (fx.name, check)


# --------------------------------------------------------------------------------------
# controls: when simple addition is exact
# --------------------------------------------------------------------------------------

def test_rigid_motion_without_residual_is_exact(fixtures):
    err = _errors(fixtures["rigid_rotation_control"], 2)["naive_vs_oracle"]
    assert err["max_px"] < 1e-9


def test_translation_only_globals_with_constant_residual_is_exact(fixtures):
    """A_k = I and a constant residual: the naive fixed-grid sum is the correct composition."""
    e1 = _errors(fixtures["translation_only_residual"], 1)["naive_vs_oracle"]
    e2 = _errors(fixtures["translation_only_residual"], 2)["naive_vs_oracle"]
    assert e1["max_px"] < 1e-9 and e2["max_px"] < 1e-9


# --------------------------------------------------------------------------------------
# the counterexample: vector transport
# --------------------------------------------------------------------------------------

def test_constant_residual_rotation_transport_error_matches_closed_form(fixtures):
    """Frame 2 must be wrong by exactly ||u - R_linear u||; frame 1 must be exact (constant field)."""
    fx = fixtures["constant_residual_rotation"]
    W, H = fx.W, fx.H
    R = CA.rotation_about_clockwise(W, H, 10.0)
    u = np.array([8.0, 0.0])
    expected = float(np.linalg.norm(u - R[:2, :2] @ u))
    assert expected == pytest.approx(1.3944918839625307, rel=1e-9)

    e1 = _errors(fx, 1)["naive_vs_oracle"]
    e2 = _errors(fx, 2)["naive_vs_oracle"]
    assert e1["max_px"] < 1e-9                       # constant residual: sampling location irrelevant
    assert e2["mean_px"] == pytest.approx(expected, rel=1e-9)
    assert e2["max_px"] == pytest.approx(expected, rel=1e-9)
    # and it is not a frozen "desired" value: the transported oracle is the one that matches the closed form
    pts = CA.output_grid(W, H)
    naive = CA.naive_from_plan(CA.build_plan(fx)[0], 2, pts, W, H)
    closed = G.apply_h(R, G.apply_h(R, pts) + u)
    assert np.linalg.norm(naive - closed, axis=1).max() == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------------------
# location / resampling error and spatial fields
# --------------------------------------------------------------------------------------

def test_spatial_ramp_translation_isolates_location_error(fixtures):
    """A linear residual sampled at the wrong preimage is wrong by an analytically known vector field.

    With identity linear parts (translation globals), ``r(p) = L p`` and translation ``T`` the naive sum is
    wrong by ``r(x+T) - r(x) = L T`` at frame 1 and ``3 L T - L^2 x`` at frame 2 (no vector transport).
    """
    fx = fixtures["spatial_ramp_translation"]
    W, H = fx.W, fx.H
    pts = CA.output_grid(W, H)
    plan, _ = CA.build_plan(fx)
    L = np.diag([0.02, 0.01]); T = np.array([6.0, 0.0])
    interior = (pts[:, 0] <= W - 14) & (pts[:, 1] > 1) & (pts[:, 1] < H - 2)
    expected = {1: np.broadcast_to(L @ T, pts.shape), 2: 3 * (L @ T) - pts @ (L @ L).T}
    for t in (1, 2):
        diff = CA.naive_from_plan(plan, t, pts, W, H) - CA.compose_oracle(fx.pairs, t, pts, W, H)
        np.testing.assert_allclose(diff[interior], expected[t][interior], atol=1e-9)


def test_parallax_error_is_localised_to_the_foreground(fixtures):
    fx = fixtures["parallax_two_layer"]
    res = CA.audit_fixture(fx)
    by = res["errors"]["frame_2"]["by_region"]
    assert by["foreground"]["mean_px"] > 0.1
    assert by["background"]["mean_px"] < 0.1
    assert by["foreground"]["n"] < by["background"]["n"]     # the foreground is the smaller supported region
    assert "displacement_px = (f_px/Z) * translation" in fx.notes


def test_rows_mesh_order_changes_the_result(fixtures):
    """Production applies the mesh shift before sampling rows; the model applies rows at the source row."""
    fx = fixtures["rows_mesh_order"]
    e = _errors(fx, 1)["naive_vs_oracle"]
    assert e["max_px"] == pytest.approx(0.06, abs=1e-9)


# --------------------------------------------------------------------------------------
# support / visibility
# --------------------------------------------------------------------------------------

def test_border_fixture_excludes_out_of_frame_output(fixtures):
    fx = fixtures["rotation_translation_border"]
    res = CA.audit_fixture(fx)
    assert res["support"]["final_excluded_frac"] > 0.1
    # the transport error is still present on the supported subset, so the fix is not just a border crop
    e = res["errors"]["frame_2"]["naive_vs_oracle_in_bounds"]
    assert e["max_px"] == pytest.approx(1.3944918839625307, rel=1e-6)


def test_support_fraction_is_reported_not_silently_dropped(fixtures):
    fx = fixtures["rotation_translation_border"]
    res = CA.audit_fixture(fx)
    for t in range(fx.T):
        frac = res["support"][f"frame_{t}"]["oracle_in_bounds_frac"]
        assert 0.0 <= frac <= 1.0
    assert res["errors"]["frame_2"]["naive_vs_oracle"]["n"] == fx.W * fx.H


# --------------------------------------------------------------------------------------
# the actual production path (MLX/Metal; run under the heavy lock)
# --------------------------------------------------------------------------------------

def test_production_path_matches_independent_numpy_replication(fixtures):
    pytest.importorskip("mlx.core")
    for name in ("rigid_rotation_control", "translation_only_residual", "constant_residual_rotation",
                 "spatial_ramp_translation", "parallax_two_layer", "rows_mesh_order"):
        fx = fixtures[name]
        plan, _ = CA.build_plan(fx)
        for t in range(fx.T):
            prod = CA.production_field(plan, t, fx.W, fx.H)
            naive = CA.naive_from_plan(plan, t, CA.output_grid(fx.W, fx.H), fx.W, fx.H)
            assert np.abs(prod - naive).max() < 1e-3, (name, t)


def test_production_counterexample_reproduces_the_transport_error(fixtures):
    pytest.importorskip("mlx.core")
    fx = fixtures["constant_residual_rotation"]
    plan, _ = CA.build_plan(fx)
    pts = CA.output_grid(fx.W, fx.H)
    prod = CA.production_field(plan, 2, fx.W, fx.H)
    oracle = CA.compose_oracle(fx.pairs, 2, pts, fx.W, fx.H)
    err = CA.error_stats(prod, oracle)
    assert err["mean_px"] == pytest.approx(1.39449, abs=2e-3)
    assert err["max_px"] == pytest.approx(1.39450, abs=2e-3)


# --------------------------------------------------------------------------------------
# inverse behaviour and JSON safety
# --------------------------------------------------------------------------------------

def test_inverse_audit_reports_convergence_and_failure():
    inv = CA.inverse_audit()
    assert inv["stable"]["converged"] is True and inv["stable"]["iterations"] < 50
    assert inv["zoom_fixed_point_divergence"]["converged"] is False
    assert inv["fold_over"]["negative_det_frac"] == 1.0


def test_run_audit_writes_json_safe_results_and_fixtures(tmp_path):
    out = tmp_path / "audit"
    res = CA.run_audit(out, with_production=False)
    payload = (out / "results.json").read_text()
    assert "NaN" not in payload and "Infinity" not in payload
    loaded = json.loads(payload)
    assert loaded["schema"] == CA.SCHEMA
    assert len(loaded["fixtures"]) == len(CA.build_fixtures())
    assert (out / "fixtures.npz").exists()
    saved = np.load(out / "fixtures.npz")
    assert "constant_residual_rotation.pair0.mesh" in saved
    assert res["fixtures"][0]["errors"]["frame_1"]["naive_vs_oracle"]["n"] > 0
