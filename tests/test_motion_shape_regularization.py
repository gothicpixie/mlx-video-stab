"""Analytic tests for the opt-in shape-preserving local-mesh regularizer in ``fit_motion``.

The regularizer penalizes the per-cell trace-free symmetric strain of the mesh displacement, so translation,
local rotation and isotropic scale are null modes. It is a weighted least-squares solve over the mesh nodes,
not a post-hoc scalar shrink. ``shape_strength`` defaults to zero and the regularizer is not invoked in that
case, so the historical H/mesh/rows remain byte-for-byte unchanged. The global similarity and the
x-independent row field are never regularized.

No media, FFmpeg, RAFT or changed production defaults.
"""
import numpy as np
import pytest

from mlxstab import motion as MO


def _grid(R, C, W, H):
    """Mesh vertex coordinates matching ``fit_motion``'s ``[-0.5, W-0.5] x [-0.5, H-0.5]`` grid."""
    x = -0.5 + np.arange(C) * (W / (C - 1))
    y = -0.5 + np.arange(R) * (H / (R - 1))
    return np.meshgrid(x, y)


def _strain_energy(mesh, W, H):
    """Per-cell area-weighted ``||dev sym(grad u)||_F^2`` (independent recomputation)."""
    R, C = mesh.shape[:2]
    dx, dy = W / (C - 1), H / (R - 1)
    ux, uy = mesh[..., 0], mesh[..., 1]
    dx_ux = ((ux[1:, 1:] - ux[1:, :-1]) + (ux[:-1, 1:] - ux[:-1, :-1])) / (2 * dx)
    dy_uy = ((uy[1:, :-1] - uy[:-1, :-1]) + (uy[1:, 1:] - uy[:-1, 1:])) / (2 * dy)
    dx_uy = ((uy[1:, 1:] - uy[1:, :-1]) + (uy[:-1, 1:] - uy[:-1, :-1])) / (2 * dx)
    dy_ux = ((ux[1:, :-1] - ux[:-1, :-1]) + (ux[1:, 1:] - ux[:-1, 1:])) / (2 * dy)
    e00 = 0.5 * (dx_ux - dy_uy)
    e01 = 0.5 * (dx_uy + dy_ux)
    return 2.0 * dx * dy * (e00 ** 2 + e01 ** 2)


def _localized_shear_obs(W=180, H=180, step=4):
    """Uniform samples with a translation plus a localized shear/anisotropic-strain bump."""
    xs = np.arange(step // 2, W, step)
    ys = np.arange(step // 2, H, step)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64)
    xc = yc = 90.0
    sig = 20.0
    g = np.exp(-((pts[:, 0] - xc) ** 2 + (pts[:, 1] - yc) ** 2) / (2 * sig ** 2))
    flow = np.stack([5.0 + 2.0 * g * (pts[:, 1] - yc) / sig,
                     -3.0 + 1.0 * g * (pts[:, 0] - xc) / sig], 1)
    return MO.Obs(pts, flow, np.ones(len(pts), bool), W, H)


def _assert_same_fit(a, b):
    for name in ("H", "mesh", "rows", "mesh_support_mass", "row_support_mass",
                 "cycle_err_rms", "cycle_inconsistency_frac", "cycle_sample_mass"):
        left, right = getattr(a, name), getattr(b, name)
        if left is None or right is None:
            assert left is None and right is None
        else:
            assert np.array_equal(left, right), name
            assert left.tobytes() == right.tobytes(), name


# --------------------------------------------------------------------------------------
# default parity and validation
# --------------------------------------------------------------------------------------

def test_strength_zero_is_byte_for_byte_parity_with_default():
    obs = _localized_shear_obs()
    default = MO.fit_motion(obs, mesh=(10, 10), rs_rows=4, rng=np.random.default_rng(7))
    explicit = MO.fit_motion(obs, mesh=(10, 10), rs_rows=4, rng=np.random.default_rng(7),
                             shape_strength=0.0)
    _assert_same_fit(default, explicit)
    assert default.fallback is False


@pytest.mark.parametrize("alias", ["strength", "shape_regularization_strength", "mesh_shape_strength"])
def test_fit_motion_does_not_accept_low_level_aliases(alias):
    obs = _localized_shear_obs(W=64, H=48, step=8)
    with pytest.raises(TypeError):
        MO.fit_motion(obs, mesh=(4, 3), rs_rows=2, **{alias: 2.0})


@pytest.mark.parametrize("bad", [-1.0, -1e-9, np.nan, np.inf, -np.inf, True, False, np.bool_(True),
                                 [1.0], np.array(1.0), "1", b"1", 1 + 2j, None])
def test_invalid_strength_is_rejected(bad):
    obs = _localized_shear_obs(W=64, H=48, step=8)
    with pytest.raises(ValueError):
        MO.fit_motion(obs, mesh=(4, 3), rs_rows=2, shape_strength=bad)


# --------------------------------------------------------------------------------------
# the energy's nullspace: translation, local rotation and isotropic scale
# --------------------------------------------------------------------------------------

def _affine_mesh(kind, R=9, C=9, W=144, H=144):
    X, Y = _grid(R, C, W, H)
    tx, ty = 3.25, -1.75
    scale, angle = 1.3, 0.2
    if kind == "translation":
        ux, uy = np.full_like(X, tx), np.full_like(X, ty)
    elif kind == "rotation":
        ux, uy = tx - angle * Y, ty + angle * X
    elif kind == "isotropic":
        ux, uy = tx + scale * X, ty + scale * Y
    elif kind == "combined":
        ux, uy = tx + scale * X - angle * Y, ty + angle * X + scale * Y
    else:
        raise AssertionError(kind)
    return np.stack([ux, uy], -1)


@pytest.mark.parametrize("kind", ["translation", "rotation", "isotropic", "combined"])
def test_regularizer_preserves_affine_similarity_modes_exactly(kind):
    R = C = 9
    mesh = _affine_mesh(kind, R, C)
    # Non-uniform data support must not break the exact nullspace.
    mass = np.linspace(0.5, 4.0, R * C).reshape(R, C)
    out = MO._regularize_mesh_shape(mesh, mass, 144, 144, strength=1e3)
    np.testing.assert_allclose(out, mesh, atol=1e-9, rtol=0)
    np.testing.assert_allclose(_strain_energy(out, 144, 144), 0.0, atol=1e-9)


# --------------------------------------------------------------------------------------
# localized shear / anisotropic strain reduction
# --------------------------------------------------------------------------------------

def test_direct_solve_reduces_local_shear_without_moving_far_nodes_or_mean():
    R = C = 11
    W = H = 180
    X, Y = _grid(R, C, W, H)
    xc = yc = 90.0
    sig = 20.0
    g = np.exp(-((X - xc) ** 2 + (Y - yc) ** 2) / (2 * sig ** 2))
    target = np.stack([5.0 + 2.0 * g * (Y - yc) / sig,
                       -3.0 + 1.0 * g * (X - xc) / sig], -1)
    mass = np.ones((R, C))
    out = MO._regularize_mesh_shape(target, mass, W, H, strength=3.0)

    before = _strain_energy(target, W, H)
    after = _strain_energy(out, W, H)
    central = (slice(4, 7), slice(4, 7))
    far = (slice(0, 2), slice(0, 2))
    assert after[central].sum() < 0.5 * before[central].sum()
    assert after.sum() < 0.5 * before.sum()
    np.testing.assert_allclose(out.mean((0, 1)), target.mean((0, 1)), atol=1e-10)
    displacement = np.linalg.norm(out - target, axis=-1)
    assert displacement[far].max() < 0.02
    assert displacement[central].max() > 5 * displacement[far].max()


def test_fit_motion_regularizer_reduces_mesh_strain_and_leaves_global_and_rows_alone():
    W = H = 180
    obs = _localized_shear_obs(W, H, step=4)
    gc = gr = 10
    baseline = MO.fit_motion(obs, mesh=(gc, gr), rs_rows=4)
    regularized = MO.fit_motion(obs, mesh=(gc, gr), rs_rows=4, shape_strength=3.0)
    assert baseline.fallback is False and regularized.fallback is False

    # The global similarity and x-independent row field are not regularized at all.
    assert np.array_equal(regularized.H, baseline.H)
    assert np.array_equal(regularized.rows, baseline.rows)
    assert np.array_equal(regularized.row_support_mass, baseline.row_support_mass)

    before = _strain_energy(baseline.mesh, W, H)
    after = _strain_energy(regularized.mesh, W, H)
    central = (slice(3, 7), slice(3, 7))
    far = (slice(0, 3), slice(0, 3))
    assert after[central].sum() < 0.5 * before[central].sum()
    assert after.sum() < before.sum()

    # Translation is anchored by the weighted data term: the weighted mean change is zero. The unweighted
    # mean also stays effectively fixed, and the far corner is not materially disturbed.
    change = regularized.mesh - baseline.mesh
    weighted_mean = np.average(change, axis=(0, 1), weights=baseline.mesh_support_mass)
    np.testing.assert_allclose(weighted_mean, 0.0, atol=1e-12)
    np.testing.assert_allclose(regularized.mesh.mean((0, 1)), baseline.mesh.mean((0, 1)), atol=1e-3)
    displacement = np.linalg.norm(change, axis=-1)
    assert displacement[far].max() < 0.05
    assert displacement[3:7, 3:7].max() > 3 * displacement[far].max()


def test_fallback_and_degenerate_grids_stay_finite_with_regularizer_enabled():
    # Too few valid samples -> the historical identity fallback, unchanged and finite.
    obs = MO.Obs(np.array([[2.0, 2.0], [6.0, 6.0]]), np.zeros((2, 2)), np.ones(2, bool), 64, 48)
    fit = MO.fit_motion(obs, mesh=(4, 3), rs_rows=4, shape_strength=5.0)
    assert fit.fallback is True
    assert np.array_equal(fit.H, np.eye(3))
    assert np.isfinite(fit.mesh).all() and np.isfinite(fit.rows).all()
    assert np.all(fit.mesh == 0.0) and np.all(fit.rows == 0.0)

    # A 1x1 mesh has no cells; the strength is harmless and the finite target is returned unchanged.
    mesh = np.stack([np.ones((1, 1)), np.zeros((1, 1))], -1)
    out = MO._regularize_mesh_shape(mesh, np.ones((1, 1)), 64, 48, strength=5.0)
    assert np.array_equal(out, mesh) and np.isfinite(out).all()

    # Sparse support cannot make the solve non-finite.
    R, C = 7, 8
    target = np.random.default_rng(0).normal(size=(R, C, 2))
    mass = np.zeros((R, C)); mass[0:2, 0:2] = 1.0
    out = MO._regularize_mesh_shape(target, mass, 80, 70, strength=3.0)
    assert np.isfinite(out).all() and out.shape == target.shape
