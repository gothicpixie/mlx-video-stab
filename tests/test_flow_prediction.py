"""Focused NumPy-only tests for the E04 flow-initialization experiment (``bench/flow_prediction.py``).

Analytic scope: no media, no MLX, no production wiring, no changed defaults. The module under test imports
only NumPy (plus the pure-Python manifest helper), so these tests can run without the shared heavy lock. They
assert independent closed forms -- that transport is *not* relabelling for spatially varying motion, that
constant-velocity/acceleration extrapolation is exact on matching translation sequences, and that the
two-layer fixture exposes occlusion/disocclusion with localised error.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bench import flow_prediction as FP

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fixtures():
    return FP.build_fixtures()


# --------------------------------------------------------------------------------------
# coordinate convention and resampling
# --------------------------------------------------------------------------------------

def test_in_bounds_pixel_edge_convention():
    assert FP.in_bounds(np.array([[-0.5, 0.0]]), 8, 6)[0]
    assert FP.in_bounds(np.array([[7.5, 5.5]]), 8, 6)[0]
    assert not FP.in_bounds(np.array([[-0.6, 0.0]]), 8, 6)[0]
    assert not FP.in_bounds(np.array([[7.6, 0.0]]), 8, 6)[0]


def test_bilinear_sampler_reproduces_linear_field():
    W, H = 16, 12
    xs, ys = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    field = np.stack([0.1 * xs + 0.2 * ys, -0.3 * xs + 0.05 * ys], -1).astype(np.float64)
    rng = np.random.default_rng(0)
    pts = np.stack([rng.uniform(1.0, W - 2.0, 200), rng.uniform(1.0, H - 2.0, 200)], 1)
    got = FP.sample_field(field, pts, W, H)
    expected = np.stack([0.1 * pts[:, 0] + 0.2 * pts[:, 1], -0.3 * pts[:, 0] + 0.05 * pts[:, 1]], 1)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_bilinear_sampler_border_clamps_to_edge_node():
    W, H = 5, 4
    field = np.zeros((H, W, 2))
    field[..., 0] = np.arange(W)[None, :]     # x component = column index
    # -0.5 clamps to node 0, W-0.5 clamps to node W-1
    got = FP.sample_field(field, np.array([[-0.5, 0.0], [W - 0.5, 0.0]]), W, H)
    assert got[0, 0] == pytest.approx(0.0)
    assert got[1, 0] == pytest.approx(W - 1.0)


def test_conservative_mask_sampling_rejects_invalid_centre_tap():
    W, H = 5, 4
    mask = np.ones((H, W), bool)
    mask[2, 2] = False
    got = FP.sample_mask_conservative(mask, np.array([[2.0, 2.0], [0.0, 0.0]]), W, H)
    assert not got[0]
    assert got[1]


def test_conservative_mask_sampling_ignores_zero_weight_neighbours():
    """At an exact pixel centre only one bilinear tap has weight; invalid adjacent pixels must not leak in
    (matches ``motion._sample_invalid``)."""
    W, H = 5, 4
    mask = np.ones((H, W), bool)
    mask[2, 3] = mask[3, 2] = mask[3, 3] = False          # all three zero-weight neighbours invalid
    got = FP.sample_mask_conservative(mask, np.array([[2.0, 2.0], [0.0, 0.0]]), W, H)
    assert got[0] and got[1]


def test_conservative_mask_sampling_rejects_invalid_fractional_neighbour():
    """A fractional position that puts non-zero weight on an invalid neighbour must reject the sample."""
    W, H = 5, 4
    mask = np.ones((H, W), bool)
    mask[2, 3] = False                                     # x=2.5 gives this tap weight 0.5
    got = FP.sample_mask_conservative(mask, np.array([[2.5, 2.0], [3.5, 3.0]]), W, H)
    assert not got[0]
    assert got[1]


def test_rotation_helper_is_clockwise_in_image_coordinates():
    """A positive angle must rotate the helper clockwise (image y down): the point right of centre moves
    down, which is the opposite sign to ``geometry.rotate_quad`` (visual counter-clockwise).
    """
    W, H = 32, 24
    R = FP.rotation_about_clockwise(W, H, 90.0)
    c = np.array([W / 2 - 0.5, H / 2 - 0.5])
    q = FP.apply_h(R, np.array([[c[0] + 4.0, c[1]]]))
    np.testing.assert_allclose(q, np.array([[c[0], c[1] + 4.0]]), atol=1e-9)
    assert q[0, 1] > c[1]


# --------------------------------------------------------------------------------------
# translation fixtures: exactness and extrapolation order
# --------------------------------------------------------------------------------------

def test_translation_constant_all_candidates_exact(fixtures):
    fx = fixtures["translation_constant"]
    grid = FP.source_grid(fx.W, fx.H)
    for target in (1, 2, 3):
        truth = fx.fields[target].reshape(-1, 2)
        truth_valid = fx.valids[target].reshape(-1)
        for method in FP.METHODS:
            if method == "cold":
                continue
            pred = FP.predict(fx.pairs, fx.fields, fx.valids, target, method, fx.W, fx.H, 0.0, grid)
            if pred.reason == "insufficient_history":
                continue
            ev = FP.evaluate(pred.field, pred.valid, truth, truth_valid,
                             raw_field=pred.raw_field, raw_valid=pred.raw_valid)
            assert ev["candidate_error"]["mean_px"] == pytest.approx(0.0, abs=1e-9), (target, method)
        cold = FP.predict(fx.pairs, fx.fields, fx.valids, target, "cold", fx.W, fx.H, 0.0, grid)
        assert np.linalg.norm(cold.raw_field[0] - truth[0]) == pytest.approx(np.sqrt(13.0))


def test_translation_linear_constant_velocity_and_acceleration_exact(fixtures):
    fx = fixtures["translation_linear"]
    grid = FP.source_grid(fx.W, fx.H)
    target = 3
    truth = fx.fields[target].reshape(-1, 2)
    truth_valid = fx.valids[target].reshape(-1)
    t = {k: fx.fields[k].reshape(-1, 2)[0] for k in range(4)}

    cv = FP.predict(fx.pairs, fx.fields, fx.valids, target, "constant_velocity", fx.W, fx.H, 0.0, grid)
    v = cv.raw_valid
    np.testing.assert_allclose(cv.raw_field[v], np.broadcast_to(2 * t[2] - t[1], truth.shape)[v], atol=1e-12)
    np.testing.assert_allclose(cv.raw_field[v], truth[v], atol=1e-12)

    ca = FP.predict(fx.pairs, fx.fields, fx.valids, target, "constant_acceleration", fx.W, fx.H, 0.0, grid)
    v = ca.raw_valid
    np.testing.assert_allclose(ca.raw_field[v], truth[v], atol=1e-12)

    sg = FP.predict(fx.pairs, fx.fields, fx.valids, target, "same_grid", fx.W, fx.H, 0.0, grid)
    ev = FP.evaluate(sg.field, sg.valid, truth, truth_valid)
    # same_grid holds the previous translation, so the error is exactly |t_3 - t_2| = sqrt(13).
    assert ev["candidate_error"]["mean_px"] == pytest.approx(np.sqrt(13.0), rel=1e-9)


def test_translation_quadratic_acceleration_exact_velocity_not(fixtures):
    fx = fixtures["translation_quadratic"]
    grid = FP.source_grid(fx.W, fx.H)
    target = 3
    truth = fx.fields[target].reshape(-1, 2)
    truth_valid = fx.valids[target].reshape(-1)
    t = {k: fx.fields[k].reshape(-1, 2)[0] for k in range(4)}

    ca = FP.predict(fx.pairs, fx.fields, fx.valids, target, "constant_acceleration", fx.W, fx.H, 0.0, grid)
    v = ca.raw_valid
    np.testing.assert_allclose(ca.raw_field[v], truth[v], atol=1e-10)

    cv = FP.predict(fx.pairs, fx.fields, fx.valids, target, "constant_velocity", fx.W, fx.H, 0.0, grid)
    ev = FP.evaluate(cv.field, cv.valid, truth, truth_valid)
    assert ev["candidate_error"]["mean_px"] == pytest.approx(1.0, rel=1e-9)   # |2t_2 - t_1 - t_3| = 1


def test_transport_uses_source_to_target_map_not_relabel(fixtures):
    """For a spatially varying (rotation) field, transport must differ from same-grid relabelling; for a
    constant translation field the two coincide."""
    rot = fixtures["rotation_const"]
    grid = FP.source_grid(rot.W, rot.H)
    target = 2
    sg = FP.predict(rot.pairs, rot.fields, rot.valids, target, "same_grid", rot.W, rot.H, 0.0, grid)
    tr = FP.predict(rot.pairs, rot.fields, rot.valids, target, "transported", rot.W, rot.H, 0.0, grid)
    assert np.abs(sg.raw_field - tr.raw_field).max() > 1e-3

    trans = fixtures["translation_constant"]
    g2 = FP.source_grid(trans.W, trans.H)
    sg2 = FP.predict(trans.pairs, trans.fields, trans.valids, target, "same_grid", trans.W, trans.H, 0.0, g2)
    tr2 = FP.predict(trans.pairs, trans.fields, trans.valids, target, "transported", trans.W, trans.H, 0.0, g2)
    common = sg2.raw_valid & tr2.raw_valid
    assert common.any()
    np.testing.assert_allclose(sg2.raw_field[common], tr2.raw_field[common], atol=1e-12)


# --------------------------------------------------------------------------------------
# rotation: steady image field and closed-form transport
# --------------------------------------------------------------------------------------

def test_rotation_steady_field_same_grid_exact_transport_closed_form(fixtures):
    fx = fixtures["rotation_const"]
    W, H = fx.W, fx.H
    grid = FP.source_grid(W, H)
    target = 2
    truth = fx.fields[target].reshape(-1, 2)
    truth_valid = fx.valids[target].reshape(-1)

    sg = FP.predict(fx.pairs, fx.fields, fx.valids, target, "same_grid", W, H, 0.0, grid)
    ev_sg = FP.evaluate(sg.field, sg.valid, truth, truth_valid)
    assert ev_sg["candidate_error"]["mean_px"] == pytest.approx(0.0, abs=1e-9)

    tr = FP.predict(fx.pairs, fx.fields, fx.valids, target, "transported", W, H, 0.0, grid)
    ev_tr = FP.evaluate(tr.field, tr.valid, truth, truth_valid)
    assert ev_tr["candidate_error"]["mean_px"] > 1e-3

    # independent closed form: transported_k(q) = f_k(Phi_k^{-1}(q)) = q - R^{-1} q
    R = FP.rotation_about_clockwise(W, H, 8.0)
    Ri = np.linalg.inv(R)
    q = np.array([[20.0, 6.0]])
    vals, valid = FP.transport(fx.fields[0], fx.valids[0], fx.pairs, 0, 1, q, W, H)
    assert valid[0]
    np.testing.assert_allclose(vals[0], (q - FP.apply_h(Ri, q))[0], atol=1e-12)


# --------------------------------------------------------------------------------------
# projective: inverse composition and honest resampling error
# --------------------------------------------------------------------------------------

def test_projective_inverse_composes_and_transport_is_reported(fixtures):
    fx = fixtures["projective"]
    grid = FP.source_grid(fx.W, fx.H)
    interior = (grid[:, 0] >= 2) & (grid[:, 0] <= fx.W - 3) & (grid[:, 1] >= 2) & (grid[:, 1] <= fx.H - 3)
    for pair in fx.pairs:
        p = grid[interior]
        q = pair.forward(p)
        np.testing.assert_allclose(pair.inverse(q), p, atol=1e-9)
        # forward validity agrees with inverse validity at the image of the point
        np.testing.assert_array_equal(pair.forward_valid(p), pair.inverse_valid(q))

    tr = FP.predict(fx.pairs, fx.fields, fx.valids, 1, "transported", fx.W, fx.H, 0.0, grid)
    assert tr.coverage > 0.5 and not tr.fallback
    truth = fx.fields[1].reshape(-1, 2)
    truth_valid = fx.valids[1].reshape(-1)

    # The repeated homography gives a stationary image-coordinate flow field, so relabelling is exact.
    sg = FP.predict(fx.pairs, fx.fields, fx.valids, 1, "same_grid", fx.W, fx.H, 0.0, grid)
    ev_sg = FP.evaluate(sg.field, sg.valid, truth, truth_valid)
    assert ev_sg["candidate_error"]["mean_px"] == pytest.approx(0.0, abs=1e-9)

    ev = FP.evaluate(tr.field, tr.valid, truth, truth_valid)
    # transport samples the stationary field at the wrong location: the error is chiefly model mismatch,
    # with bilinear discretization only a smaller additional term -- nonzero but small.
    assert 0.0 < ev["candidate_error"]["mean_px"] < 1.0

    ca = FP.predict(fx.pairs, fx.fields, fx.valids, 2, "constant_acceleration", fx.W, fx.H, 0.0, grid)
    assert ca.fallback and ca.reason == "insufficient_history"


# --------------------------------------------------------------------------------------
# fallback policy
# --------------------------------------------------------------------------------------

def test_low_coverage_falls_back_to_cold(fixtures):
    fx = fixtures["translation_out"]
    grid = FP.source_grid(fx.W, fx.H)
    pred = FP.predict(fx.pairs, fx.fields, fx.valids, 3, "transported", fx.W, fx.H, min_coverage=0.35, grid=grid)
    assert pred.fallback and pred.reason == "low_coverage"
    assert pred.coverage < 0.35
    assert np.all(pred.field == 0.0) and pred.valid.all()
    assert pred.raw_valid.mean() == pytest.approx(pred.coverage)

    truth = fx.fields[3].reshape(-1, 2)
    truth_valid = fx.valids[3].reshape(-1)
    ev = FP.evaluate(pred.field, pred.valid, truth, truth_valid,
                     raw_field=pred.raw_field, raw_valid=pred.raw_valid)
    assert ev["coverage"] < 0.35
    assert ev["effective_error"]["mean_px"] == pytest.approx(24.0)   # cold over the valid support


def test_insufficient_history_falls_back_to_cold(fixtures):
    fx = fixtures["translation_constant"]
    grid = FP.source_grid(fx.W, fx.H)
    for method, target in (("constant_velocity", 1), ("constant_acceleration", 2)):
        pred = FP.predict(fx.pairs, fx.fields, fx.valids, target, method, fx.W, fx.H, 0.0, grid)
        assert pred.fallback and pred.reason == "insufficient_history"
        assert pred.coverage == 0.0 and np.all(pred.field == 0.0)
        # no candidate exists: raw metrics are empty, effective cold validity stays all true.
        assert not pred.raw_valid.any()
        assert pred.valid.all()
        ev = FP.evaluate(pred.field, pred.valid, fx.fields[target].reshape(-1, 2),
                         fx.valids[target].reshape(-1),
                         raw_field=pred.raw_field, raw_valid=pred.raw_valid)
        assert ev["coverage"] == 0.0
        assert ev["support"] == 0.0
        assert ev["effective_coverage"] == 1.0
        assert ev["candidate_error"]["n"] == 0


def test_all_non_finite_transport_source_falls_back_zero_coverage(fixtures):
    """A fully non-finite transport source invalidates every sample, so there is no valid candidate
    support: the reason is ``zero_coverage`` (the masked raw candidate is finite) and the effective
    initialization is cold.
    """
    fx = fixtures["translation_constant"]
    grid = FP.source_grid(fx.W, fx.H)
    fields = fx.fields.copy()
    fields[2] = np.nan
    pred = FP.predict(fx.pairs, fields, fx.valids, 3, "transported", fx.W, fx.H, 0.0, grid)
    assert pred.fallback and pred.reason == "zero_coverage"
    assert not pred.raw_valid.any() and np.isfinite(pred.raw_field).all()
    assert pred.valid.all() and np.all(pred.field == 0.0)


def test_non_finite_on_invalid_cell_is_masked_and_does_not_fall_back(fixtures):
    """A non-finite value only forces a fallback when it sits on the candidate's valid support.

    ``same_grid`` reuses the previous field at the same indices; a NaN at an already-invalid cell is
    discarded by the validity mask (the finite check is on the masked raw candidate), while a NaN on a
    valid cell still triggers the ``non_finite`` fallback.
    """
    fx = fixtures["translation_constant"]
    grid = FP.source_grid(fx.W, fx.H)
    flat_valid = fx.valids[0].reshape(-1)
    invalid = np.flatnonzero(~flat_valid)
    valid = np.flatnonzero(flat_valid)
    assert invalid.size and valid.size

    # NaN only at an already-invalid cell: masked raw stays finite -> no fallback.
    masked = fx.fields.copy()
    masked[0].reshape(-1, 2)[invalid[0]] = np.nan
    pred = FP.predict(fx.pairs, masked, fx.valids, 1, "same_grid", fx.W, fx.H, 0.0, grid)
    assert not pred.fallback and pred.reason is None
    assert np.isfinite(pred.raw_field).all()

    # NaN on valid support still triggers the non_finite fallback.
    support = fx.fields.copy()
    support[0].reshape(-1, 2)[valid[0]] = np.nan
    pred2 = FP.predict(fx.pairs, support, fx.valids, 1, "same_grid", fx.W, fx.H, 0.0, grid)
    assert pred2.fallback and pred2.reason == "non_finite"
    assert np.all(pred2.field == 0.0) and pred2.valid.all()


def test_evaluate_reports_coverage_and_error_separately():
    N = 10
    truth = np.tile([2.0, 0.0], (N, 1))
    truth_valid = np.ones(N, bool)
    raw = np.zeros((N, 2))
    raw_valid = np.zeros(N, bool)
    raw_valid[:5] = True
    eff = np.zeros((N, 2))
    eff_valid = np.ones(N, bool)                     # cold replacement
    out = FP.evaluate(eff, eff_valid, truth, truth_valid, raw_field=raw, raw_valid=raw_valid)
    assert out["coverage"] == pytest.approx(0.5)
    assert out["effective_coverage"] == pytest.approx(1.0)
    assert out["support"] == pytest.approx(0.5)
    assert out["candidate_error"]["mean_px"] == pytest.approx(2.0)
    assert out["effective_error"]["mean_px"] == pytest.approx(2.0)


# --------------------------------------------------------------------------------------
# two-layer parallax
# --------------------------------------------------------------------------------------

def _layer_disocclusion_count(fixture, target):
    grid = FP.source_grid(fixture.W, fixture.H)
    labels = fixture.region(target, grid)
    inb = FP.in_bounds(grid, fixture.W, fixture.H)
    iv = fixture.pairs[target].inverse_valid(grid)
    return int((inb & (labels == 0) & ~iv).sum())


def test_layer_occlusion_and_disocclusion_regions_exist(fixtures):
    fx = fixtures["parallax_two_layer"]
    grid = FP.source_grid(fx.W, fx.H)
    total = fx.W * fx.H

    # explicit occluded source region: background source pixels hidden under the advanced foreground
    labels = fx.region(1, grid)
    occluded = int((labels == 3).sum())
    assert 0 < occluded < total

    # disocclusion: visible background targets with no valid preimage
    assert 0 < _layer_disocclusion_count(fx, 1) < total


def test_layer_transport_exact_on_visible_and_naive_error_localised(fixtures):
    fx = fixtures["parallax_two_layer"]
    grid = FP.source_grid(fx.W, fx.H)
    target = 2
    truth = fx.fields[target].reshape(-1, 2)
    truth_valid = fx.valids[target].reshape(-1)
    region = fx.region(target, grid)

    tr = FP.predict(fx.pairs, fx.fields, fx.valids, target, "transported", fx.W, fx.H, 0.0, grid)
    ev_tr = FP.evaluate(tr.field, tr.valid, truth, truth_valid, region)
    assert ev_tr["candidate_error"]["mean_px"] == pytest.approx(0.0, abs=1e-9)   # constant layered motion
    for stats in ev_tr["by_region"].values():
        if stats["mean_px"] is not None:
            assert stats["mean_px"] == pytest.approx(0.0, abs=1e-9)

    sg = FP.predict(fx.pairs, fx.fields, fx.valids, target, "same_grid", fx.W, fx.H, 0.0, grid)
    ev_sg = FP.evaluate(sg.field, sg.valid, truth, truth_valid, region)
    assert ev_sg["candidate_error"]["mean_px"] > 0.05
    bg = ev_sg["by_region"]["background"]
    assert bg["mean_px"] is not None and bg["mean_px"] > 0.0   # revealed/hidden background strips
    # transport excludes occluded/disoccluded samples rather than clamping them: it drops relabelled-valid
    # points. (Total coverage need not be smaller under the correct non-zero-weight tap rule.)
    assert tr.coverage < 1.0
    assert int((sg.valid & ~tr.valid).sum()) > 0


def test_layer_all_orders_exact_under_constant_layered_motion(fixtures):
    fx = fixtures["parallax_two_layer"]
    grid = FP.source_grid(fx.W, fx.H)
    target = 3
    truth = fx.fields[target].reshape(-1, 2)
    truth_valid = fx.valids[target].reshape(-1)
    for method in ("transported", "constant_velocity", "constant_acceleration"):
        pred = FP.predict(fx.pairs, fx.fields, fx.valids, target, method, fx.W, fx.H, 0.0, grid)
        ev = FP.evaluate(pred.field, pred.valid, truth, truth_valid)
        assert ev["candidate_error"]["mean_px"] == pytest.approx(0.0, abs=1e-9), method


def test_reversed_parallax_fixture_present_and_consistent(fixtures):
    """``build_fixtures`` must expose the backward parallax clip so the CLI/JSON cover both directions.

    The backward clip mirrors the forward one: transport is exact on its valid support under the constant
    layered motion, while ``same_grid`` relabels the moved layer and is not. Claims stay specific to this
    motion type, not universal.
    """
    assert "parallax_two_layer_backward" in fixtures
    fwd = fixtures["parallax_two_layer"]
    back = fixtures["parallax_two_layer_backward"]
    assert back.name == "parallax_two_layer_backward"
    assert (back.W, back.H, back.frames) == (fwd.W, fwd.H, fwd.frames)
    assert len(back.pairs) == len(fwd.pairs)

    grid = FP.source_grid(back.W, back.H)
    target = 2
    truth = back.fields[target].reshape(-1, 2)
    truth_valid = back.valids[target].reshape(-1)

    tr = FP.predict(back.pairs, back.fields, back.valids, target, "transported", back.W, back.H, 0.0, grid)
    ev_tr = FP.evaluate(tr.field, tr.valid, truth, truth_valid)
    assert ev_tr["candidate_error"]["mean_px"] == pytest.approx(0.0, abs=1e-9)

    sg = FP.predict(back.pairs, back.fields, back.valids, target, "same_grid", back.W, back.H, 0.0, grid)
    ev_sg = FP.evaluate(sg.field, sg.valid, truth, truth_valid)
    assert ev_sg["candidate_error"]["mean_px"] > 0.05

    # each backward pair is the inverse map of the corresponding forward pair (temporal order reversed)
    p = grid[(grid[:, 0] >= 2) & (grid[:, 0] <= back.W - 3)
             & (grid[:, 1] >= 2) & (grid[:, 1] <= back.H - 3)]
    for j, back_pair in enumerate(back.pairs):
        fwd_pair = fwd.pairs[len(fwd.pairs) - 1 - j]
        np.testing.assert_allclose(back_pair.forward(p), fwd_pair.inverse(p), atol=1e-9)


# --------------------------------------------------------------------------------------
# backward direction
# --------------------------------------------------------------------------------------

def test_backward_direction_is_handled_by_mirror(fixtures):
    fwd = fixtures["translation_constant"]
    back = FP.reversed_fixture(fwd)
    grid = FP.source_grid(back.W, back.H)
    t = fwd.fields[0].reshape(-1, 2)[0]                      # [3, -2]
    np.testing.assert_allclose(back.fields[0].reshape(-1, 2)[0], -t, atol=1e-12)

    pred = FP.predict(back.pairs, back.fields, back.valids, 2, "transported", back.W, back.H, 0.0, grid)
    truth = back.fields[2].reshape(-1, 2)
    truth_valid = back.valids[2].reshape(-1)
    ev = FP.evaluate(pred.field, pred.valid, truth, truth_valid)
    assert ev["candidate_error"]["mean_px"] == pytest.approx(0.0, abs=1e-9)


def _time_varying_translation_fixture():
    """Forward translations [1, 2, 4, 7] px/pair; exact constant-acceleration sequence."""
    W, H = 32, 24
    return FP._translation_fixture(
        "time_varying_translation", "Translation [1,2,4,7] px/pair.", W, H,
        [[1.0, 0.0], [2.0, 0.0], [4.0, 0.0], [7.0, 0.0]])


def test_reversed_fixture_reverses_temporal_order_and_inverts_each_pair():
    """The backward pass must invert each pair *and* reverse temporal order.

    Forward translations [1, 2, 4, 7] play backwards as [-7, -4, -2, -1]: the last forward pair (7) becomes
    the first backward pair and every map is inverted. Inverting without reversing order would instead give
    [-1, -2, -4, -7], so this assertion pins the temporal-order bug.
    """
    fwd = _time_varying_translation_fixture()
    back = FP.reversed_fixture(fwd)
    got = np.array([back.fields[k].reshape(-1, 2)[0, 0] for k in range(len(back.pairs))])
    np.testing.assert_allclose(got, [-7.0, -4.0, -2.0, -1.0], atol=1e-12)


def test_backward_constant_acceleration_exact_on_time_varying_translation():
    """Backward displacements [-7, -4, -2, -1] have a constant second difference of -1, so
    constant-acceleration prediction of the next backward pair is exact on the candidate's valid support."""
    fwd = _time_varying_translation_fixture()
    back = FP.reversed_fixture(fwd)
    grid = FP.source_grid(back.W, back.H)
    target = 3
    truth = back.fields[target].reshape(-1, 2)
    pred = FP.predict(back.pairs, back.fields, back.valids, target, "constant_acceleration",
                      back.W, back.H, 0.0, grid)
    valid = pred.raw_valid
    assert valid.any()
    assert not pred.fallback and pred.reason is None
    np.testing.assert_allclose(pred.raw_field[valid], truth[valid], atol=1e-12)
    # the predicted displacement itself is the next backward step -1 px on x, 0 on y
    np.testing.assert_allclose(pred.raw_field[valid][0], [-1.0, 0.0], atol=1e-12)


# --------------------------------------------------------------------------------------
# runner / JSON safety / NumPy-only import
# --------------------------------------------------------------------------------------

def test_run_experiment_is_json_safe_and_summarised(fixtures):
    res = FP.run_experiment(fixtures=fixtures)
    payload = json.dumps(FP.json_safe(res), allow_nan=False)
    assert res["schema"] == FP.SCHEMA
    assert set(res["summary"]) == set(fixtures)
    for per_method in res["summary"].values():
        assert set(per_method) == set(FP.METHODS)
        for s in per_method.values():
            assert "skipped_no_history" in s and "candidate_error_mean_px" in s
    assert "RAFT quality" in payload and "analytic" in payload


def test_main_reports_candidate_and_effective_epe_separately(capsys):
    """The CLI summary must not hide a cold fallback behind a low candidate error.

    ``translation_out`` leaves only 25% valid support, so ``same_grid`` reuses the correct previous
    displacement (candidate error 0) but its validity is below ``min_coverage`` and the effective
    initialization is cold zeros (effective error 24 px). Both numbers must be visible on the same row.
    """
    assert FP.main([]) == 0
    out = capsys.readouterr().out
    block = out.split("translation_out", 1)[1].split("\nrotation_const", 1)[0]
    row = next(line for line in block.splitlines() if line.strip().startswith("same_grid"))
    assert "candidate_epe=" in row and "effective_epe=" in row
    cand = re.search(r"candidate_epe=\s*(\S+)", row)
    eff = re.search(r"effective_epe=\s*(\S+)", row)
    assert cand is not None and eff is not None, row
    assert float(cand.group(1)) == pytest.approx(0.0)
    assert float(eff.group(1)) == pytest.approx(24.0)
    assert "coverage=" in row and "support=" in row and "fallbacks=3" in row


def test_module_imports_without_mlx():
    code = ("import sys, bench.flow_prediction as FP; "
            "assert 'mlx' not in sys.modules, [m for m in sys.modules if m.startswith('mlx')]")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
