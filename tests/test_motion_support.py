"""Focused tests for the motion support-mass observation-coverage diagnostic.

``Fit.mesh_support_mass`` / ``Fit.row_support_mass`` record the non-robust kernel-weighted count of usable
non-mover flow samples at each mesh vertex / row band, taken *before* ``_fill`` diffuses under-observed
locations. They are an observation-coverage diagnostic (NOT a probability and NOT flow confidence): a location
with mass >= ``SUPPORT_THRESHOLD`` (3.0) is directly observed, one below it is diffusion-filled.

Covers: the analytic kernel-weighted count, fit_motion preserving the pre-fill mass for mesh and row bands,
zero masses on the fallback path and when rows are disabled, and the stable threshold constant. No media,
FFmpeg, RAFT or changed production defaults.
"""
import numpy as np
import pytest

from mlxstab import motion as MO


def _grid_obs(W=64, H=48, step=4, xs=None, ys=None):
    if xs is None:
        xs = np.arange(step // 2, W, step)
    if ys is None:
        ys = np.arange(step // 2, H, step)
    gx, gy = np.meshgrid(np.asarray(xs, np.float64), np.asarray(ys, np.float64))
    pts = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64)
    flow = np.zeros_like(pts)                       # identity global motion -> no residual, no movers
    valid = np.ones(len(pts), bool)
    return MO.Obs(pts, flow, valid, W, H)


def _expected_mesh_mass(obs, W, H, gc, gr):
    """Independent recomputation of the raw (non-robust) kernel-weighted sample count at mesh vertices."""
    vx = -.5 + np.arange(gc + 1) * W / gc
    vy = -.5 + np.arange(gr + 1) * H / gr
    cx, cy = np.meshgrid(vx, vy)
    centers = np.stack([cx.ravel(), cy.ravel()], 1)
    sig = np.array([W / gc, H / gr])
    d2 = (((obs.pts[:, None, :] - centers[None]) / sig) ** 2).sum(-1)
    K = np.exp(-0.5 * d2) * (d2 < 9.0)
    return (K * obs.valid[:, None].astype(float)).sum(0).reshape(gr + 1, gc + 1)


def _expected_row_mass(obs, H, nr):
    cy = (np.arange(nr) + .5) * H / nr - .5
    d2 = ((obs.pts[:, 1:2] - cy[None, :]) / (H / nr)) ** 2
    K = np.exp(-0.5 * d2) * (d2 < 9.0)
    return (K * obs.valid[:, None].astype(float)).sum(0)


def test_support_threshold_is_three():
    assert MO.SUPPORT_THRESHOLD == 3.0


def test_kernel_field_mass_is_the_non_robust_weighted_count():
    """The mass returned by ``_kernel_field`` is exactly the kernel-weighted usable-sample count, not robust."""
    pts = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [50.0, 50.0]])
    vals = np.zeros((4, 2))
    w = np.array([1.0, 1.0, 0.0, 1.0])
    centers = np.array([[0.0, 0.0]])
    sig = np.array([4.0, 4.0])
    _, mass = MO._kernel_field(pts, vals, w, centers, sig)
    d2 = (((pts[:, None, :] - centers[None]) / sig) ** 2).sum(-1)
    expected = ((np.exp(-0.5 * d2) * (d2 < 9.0)) * w[:, None]).sum(0)
    np.testing.assert_allclose(mass, expected, rtol=1e-12)
    assert mass.shape == (1,)


def test_fit_motion_analytic_mesh_and_row_support_mass():
    gc, gr, nr = 4, 3, 4
    obs = _grid_obs(W=64, H=48, step=4)
    fit = MO.fit_motion(obs, mesh=(gc, gr), rs_rows=nr)
    assert fit.fallback is False
    np.testing.assert_allclose(fit.mesh_support_mass, _expected_mesh_mass(obs, 64, 48, gc, gr), rtol=1e-12)
    np.testing.assert_allclose(fit.row_support_mass, _expected_row_mass(obs, 48, nr), rtol=1e-12)
    assert fit.mesh_support_mass.shape == (gr + 1, gc + 1)
    assert fit.row_support_mass.shape == (nr,)


def test_fit_motion_preserves_pre_fill_support_mass_with_subthreshold_vertices():
    """A sparse corner leaves some vertices/bands below threshold; the raw mass (zeros included) is preserved."""
    gc, gr, nr = 4, 3, 4
    xs = np.arange(2, 22, 4)
    ys = np.arange(2, 22, 4)
    obs = _grid_obs(W=64, H=48, xs=xs, ys=ys)
    fit = MO.fit_motion(obs, mesh=(gc, gr), rs_rows=nr, min_valid=20)
    assert fit.fallback is False
    expected_mesh = _expected_mesh_mass(obs, 64, 48, gc, gr)
    expected_row = _expected_row_mass(obs, 48, nr)
    # Some vertices/bands are genuinely under-observed (they are what `_fill` would diffuse into); the stored
    # support mass is the raw pre-fill count, including those zero/sub-threshold values, not a filled field.
    assert (expected_mesh < MO.SUPPORT_THRESHOLD).any()
    assert (expected_row < MO.SUPPORT_THRESHOLD).any()
    np.testing.assert_allclose(fit.mesh_support_mass, expected_mesh, rtol=1e-12)
    np.testing.assert_allclose(fit.row_support_mass, expected_row, rtol=1e-12)
    assert np.any(fit.mesh_support_mass == 0.0)


def test_fit_motion_fallback_uses_zero_support_mass():
    obs = _grid_obs(W=64, H=48, step=4, xs=[2, 6, 10], ys=[2, 6, 10])   # fewer than min_valid samples
    for rs_rows in (0, 4):
        fit = MO.fit_motion(obs, mesh=(4, 3), rs_rows=rs_rows)
        assert fit.fallback is True
        assert fit.mesh_support_mass.shape == (4, 5) and np.all(fit.mesh_support_mass == 0.0)
        assert fit.row_support_mass.shape == (max(rs_rows, 1),) and np.all(fit.row_support_mass == 0.0)


def test_fit_motion_disabled_rows_have_zero_support_mass():
    obs = _grid_obs(W=64, H=48, step=4)
    fit = MO.fit_motion(obs, mesh=(4, 3), rs_rows=0)
    assert fit.fallback is False
    assert fit.rows.shape == (1, 2) and fit.row_support_mass.shape == (1,)
    assert np.all(fit.row_support_mass == 0.0)
    # Mesh support is still computed when rows are disabled.
    np.testing.assert_allclose(fit.mesh_support_mass, _expected_mesh_mass(obs, 64, 48, 4, 3), rtol=1e-12)


def test_fit_motion_does_not_change_mesh_rows_behavior():
    """Storing support mass must not perturb the production H/mesh/rows a caller sees."""
    obs = _grid_obs(W=64, H=48, step=4)
    fit = MO.fit_motion(obs, mesh=(4, 3), rs_rows=4)
    assert fit.mesh.shape == (4, 5, 2) and fit.rows.shape == (4, 2)
    assert np.isfinite(fit.mesh).all() and np.isfinite(fit.rows).all()
    # Identity motion means zero residual; the fields are the same values as before support was added.
    np.testing.assert_allclose(fit.mesh, 0.0, atol=1e-12)
    np.testing.assert_allclose(fit.rows, 0.0, atol=1e-12)
    np.testing.assert_array_equal(fit.H, np.eye(3))
