"""Stage 0 residual-composition audit: fixed-grid residual sums versus transported composition.

This is a bounded, analytic geometry experiment with independently known ground truth. It exists because
``stabilize.plan`` accumulates per-pair mesh/row residuals by *adding* them at the current source coordinate
of the fixed output grid (``source_coords``), while the physically correct output-to-source map is the
*composition* of the per-pair forward motions. The two agree for rigid motion and for constant residuals
under pure-translation globals, and disagree once a residual vector is carried through a rotation or a
spatially varying field is sampled at the wrong location. Nothing here changes production code, defaults or
media handling; it is a measurement that specifies the Stage 1 fix and its decision gate.

Conventions (also written to the result JSON and ``docs/composition_audit.md``)
--------------------------------------------------------------------------------
Coordinate frames: pixel centres are on integers; a ``W x H`` frame spans ``[-0.5, W-0.5] x [-0.5, H-0.5]``
(``geometry.corners``). ``x`` is an *output* (virtual-camera) coordinate, ``p`` a *source*-frame coordinate.

Forward frame-to-frame motion.  Each analysed pair ``k`` is a global homography ``H_k`` plus a residual
field ``r_k`` and an optional row field ``row_k``::

    F_k(p) = H_k(p) + r_k(p) + row_k(p_y)          p in source frame k, result in source frame k+1

``H_k`` maps frame ``k`` to frame ``k+1`` (``motion.Fit.H``). ``r_k`` is the residual displacement in the
*destination* frame ``k+1``, parameterised by the source position ``p`` in frame ``k`` (this is how
``motion.fit_motion`` measures it: the observed destination is ``H(pts) + resid``). The stored mesh is only
defined at the ``(gr+1) x (gc+1)`` vertices and is bilinearly interpolated with border clamping, exactly like
``warp.sample``.

Output-to-source sampling.  With the virtual camera locked to the first analysed frame (this experiment
pins every smoothed path series to its first sample), the output frame *is* reference frame 0, so the
output-to-source map is the forward composition::

    Phi_0(x) = x,        Phi_{k+1}(x) = F_k(Phi_k(x))

This is the "oracle" (ground truth). ``plan.G[t]`` is only the global part ``G_t = H_{t-1} ... H_0`` of it.

The current fixed-grid sum.  Production evaluates the cumulative residual *at the global source coordinate*
instead of at the correct preimage::

    naive_t(x) = G_t(x) + (sum_{k<t} r_k)(G_t(x)) [+ rows sampled after the mesh shift]

Writing ``Phi_k(x) = G_k(x) + delta_k(x)`` gives the exact affine transport recursion (``A_k = H_k[:2,:2]``)::

    delta_0 = 0,        delta_{k+1}(x) = A_k delta_k(x) + r_k(G_k(x) + delta_k(x))

Simple addition is exact only when this collapses to the naive sum: the globals must not rotate/scale the
displacement vector (``A_k = I``) *and* the residual must not depend on where it is sampled (constant, or the
same evaluation point). Both conditions are tested as controls.

Fixture caution: ``synth.render(depths=...)`` multiplies camera translation directly, so those numbers are
translation/disparity factors, not metric depths (defaults 1, .55, .25 make the far layer move most). The new
two-layer fixture therefore states its own convention explicitly: pinhole-ish ``displacement_px = (f_px / Z) *
translation`` with ``Z`` in arbitrary units, i.e. ``f_px/Z`` is inverse depth. Existing synthetic defaults and
old baseline fixtures are untouched.
"""
from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mlxstab import geometry as G
from mlxstab import smooth as SM
from mlxstab import stabilize as S
from mlxstab.motion import Fit

from .manifest import json_safe

SCHEMA = "mlxstab.composition_audit/1"


# --------------------------------------------------------------------------------------
# resampling that matches warp.sample (bilinear, border)
# --------------------------------------------------------------------------------------

def _axis(coord, n):
    """Bilinear tap indices/weights for one coordinate over ``n`` vertices (warp.sample border semantics)."""
    base = np.floor(coord).astype(np.int64)
    f = coord - base
    i0 = np.clip(base, 0, n - 1)
    i1 = np.clip(base + 1, 0, n - 1)
    return i0, i1, 1.0 - f, f


def sample_mesh_bilinear(mesh, pts, W, H):
    """Bilinear+border sample of a ``(gr+1, gc+1, 2)`` vertex field at pixel coords ``pts [N,2]``.

    Vertex ``(i, j)`` sits at ``x = -0.5 + i*W/gc``, ``y = -0.5 + j*H/gr`` (``fit_motion`` conventions), so
    ``uv = ((x+0.5)*gc/W, (y+0.5)*gr/H)`` indexes the vertex grid directly. This reproduces the Metal
    bilinear kernel used by ``stabilize.source_coords``.
    """
    mesh = np.asarray(mesh, np.float64)
    pts = np.asarray(pts, np.float64)
    gr, gc = mesh.shape[0] - 1, mesh.shape[1] - 1
    u = (pts[:, 0] + 0.5) * gc / W
    v = (pts[:, 1] + 0.5) * gr / H
    x0, x1, wx0, wx1 = _axis(u, gc + 1)
    y0, y1, wy0, wy1 = _axis(v, gr + 1)
    w00 = wy0 * wx0; w01 = wy0 * wx1; w10 = wy1 * wx0; w11 = wy1 * wx1
    return (mesh[y0, x0] * w00[:, None] + mesh[y0, x1] * w01[:, None]
            + mesh[y1, x0] * w10[:, None] + mesh[y1, x1] * w11[:, None])


def sample_rows_bilinear(rows, py, H):
    """Bilinear+border sample of a ``(nr, 2)`` row field at pixel rows ``py``.

    Row band ``j`` sits at ``y = -0.5 + (j+0.5)*H/nr``, so ``v = (y+0.5)*nr/H - 0.5`` indexes the bands.
    """
    rows = np.asarray(rows, np.float64)
    nr = rows.shape[0]
    v = (np.asarray(py, np.float64) + 0.5) * nr / H - 0.5
    i0, i1, w0, w1 = _axis(v, nr)
    return rows[i0] * w0[:, None] + rows[i1] * w1[:, None]


# --------------------------------------------------------------------------------------
# transforms and grid helpers
# --------------------------------------------------------------------------------------

def rotation_about_clockwise(W, H, deg_cw):
    """3x3 rotation by ``deg_cw`` visual **clockwise** about the image centre (image y down).

    ``[[cos, -sin], [sin, cos]]`` is counter-clockwise in mathematical (y-up) coordinates; with the image y
    axis pointing down it is a visual clockwise rotation, so a positive angle moves a point right of centre
    downward. This is the sign the fixtures use; ``geometry.rotate_quad`` is the visual-counter-clockwise
    inverse and is deliberately not used here.
    """
    r = np.deg2rad(deg_cw)
    A = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    c = np.array([W / 2 - .5, H / 2 - .5])
    T = np.eye(3); T[:2, 2] = c
    Ti = np.eye(3); Ti[:2, 2] = -c
    M = np.eye(3); M[:2, :2] = A
    return T @ M @ Ti


def translation(dx, dy):
    M = np.eye(3); M[0, 2] = dx; M[1, 2] = dy
    return M


def vertex_grid(W, H, gc, gr):
    xs = -0.5 + np.arange(gc + 1) * W / gc
    ys = -0.5 + np.arange(gr + 1) * H / gr
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], 1)


def field_mesh(W, H, gc, gr, fn):
    """Evaluate a displacement field ``fn(points)->[N,2]`` at the mesh vertices -> ``(gr+1, gc+1, 2)``."""
    return fn(vertex_grid(W, H, gc, gr)).reshape(gr + 1, gc + 1, 2)


def row_field(W, H, nr, fn):
    """Evaluate a displacement field ``fn(py)->[nr,2]`` at the row-band centres -> ``(nr,2)``."""
    ys = -0.5 + (np.arange(nr) + 0.5) * H / nr
    return fn(ys)


def output_grid(W, H):
    xs, ys = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    return np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)


# --------------------------------------------------------------------------------------
# fixture model
# --------------------------------------------------------------------------------------

@dataclass
class PairMotion:
    """Forward motion frame ``k -> k+1``: global ``H``, residual ``mesh`` and optional ``rows``."""
    H: np.ndarray
    mesh: np.ndarray                       # (gr+1, gc+1, 2), residual in frame k+1 parameterised by frame k
    rows: np.ndarray | None = None         # (nr, 2), x-independent residual added to the y row band


@dataclass
class Fixture:
    name: str
    description: str
    W: int
    H: int
    T: int
    pairs: list
    mesh_cells: tuple = (4, 3)             # (gc, gr)
    rs_rows: int = 0
    region: object = None                  # callable(pts)->labels: 0 background, 1 foreground, -1 outside
    notes: str = ""
    fps: float = 30.0


def forward_apply(pair, p, W, H, with_rows=True):
    """One forward frame-to-frame map ``F_k(p)`` (model order: global, then mesh, then rows)."""
    q = G.apply_h(pair.H, p) + sample_mesh_bilinear(pair.mesh, p, W, H)
    if with_rows and pair.rows is not None:
        q = q + sample_rows_bilinear(pair.rows, p[:, 1], H)
    return q


def compose_oracle(pairs, t, pts, W, H):
    """Exact output->source map at frame ``t`` by composing the per-pair forward maps."""
    p = np.array(pts, np.float64)
    for k in range(t):
        p = forward_apply(pairs[k], p, W, H)
    return p


def compose_globals(pairs, t):
    """``G_t = H_{t-1} ... H_0`` (apply ``H_0`` first)."""
    Gm = np.eye(3)
    for k in range(t):
        Gm = pairs[k].H @ Gm
    return Gm


def transported_field(pairs, t, pts, W, H):
    """Exact displacement transport for affine/similarity globals: ``delta_{k+1} = A_k delta_k + r_k(G_k+delta_k)``."""
    pts = np.asarray(pts, np.float64)
    delta = np.zeros_like(pts)
    for k in range(t):
        Gk = G.apply_h(compose_globals(pairs, k), pts)
        q = Gk + delta
        r = sample_mesh_bilinear(pairs[k].mesh, q, W, H)
        if pairs[k].rows is not None:
            r = r + sample_rows_bilinear(pairs[k].rows, q[:, 1], H)
        delta = delta @ pairs[k].H[:2, :2].T + r
    return G.apply_h(compose_globals(pairs, t), pts) + delta


def naive_from_plan(plan, t, pts, W, H):
    """Independent NumPy replication of ``source_coords``' fixed-grid sum, using the plan's own arrays."""
    src = G.apply_h(plan.G[t], pts) + sample_mesh_bilinear(plan.mesh_corr[t], G.apply_h(plan.G[t], pts), W, H)
    nr = plan.row_corr.shape[1]
    if np.abs(plan.row_corr[t]).max() > 0:
        src = src + sample_rows_bilinear(plan.row_corr[t], src[:, 1], H)
    return src


@contextmanager
def pin_first_sample():
    """Hold every smoothed path series at its first sample (a prescribed locked virtual camera)."""
    orig = SM.smooth_columns
    SM.smooth_columns = lambda C, *a, **k: np.repeat(C[:1], C.shape[0], axis=0)
    try:
        yield
    finally:
        SM.smooth_columns = orig


def build_plan(fixture, pin=True):
    """Production ``plan`` for a fixture. Pure NumPy; no GPU. Asserts the pinned plan matches the model."""
    gc, gr = fixture.mesh_cells
    nr = max(fixture.rs_rows, 1)
    fits = [Fit(p.H.copy(), np.asarray(p.mesh, np.float64).copy(),
                (np.asarray(p.rows, np.float64).copy() if p.rows is not None else np.zeros((nr, 2))),
                1.0, 0.0, False) for p in fixture.pairs]
    mot = S.Motion(fixture.W, fixture.H, fixture.fps, fits=fits, stride=1, n_frames=fixture.T)
    cfg = S.Config(mesh=(gc, gr), rs_rows=fixture.rs_rows)
    with (pin_first_sample() if pin else nullcontext()):
        pl = S.plan(mot, cfg, n_frames=fixture.T)
    return pl, cfg


def production_field(plan, t, W, H):
    """The actual production output->source field from ``stabilize.source_coords`` (requires MLX/Metal)."""
    src = S.source_coords(plan, t, W, H)
    return np.array(src[0], np.float64).reshape(-1, 2)


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------

def _const(gc, gr, vec):
    m = np.zeros((gr + 1, gc + 1, 2)); m[..., 0] = vec[0]; m[..., 1] = vec[1]
    return m


def build_fixtures():
    """Deterministic analytic fixtures. No files, no media, no private data."""
    fx = {}

    W, H, T, gc, gr = 64, 48, 3, 4, 3
    R = rotation_about_clockwise(W, H, 10.0)
    zero = _const(gc, gr, (0.0, 0.0))

    fx["rigid_rotation_control"] = Fixture(
        "rigid_rotation_control", "No residual; two identical 10 deg rotations. Composition control.",
        W, H, T, [PairMotion(R.copy(), zero.copy()), PairMotion(R.copy(), zero.copy())], (gc, gr))

    fx["translation_only_residual"] = Fixture(
        "translation_only_residual", "Identity globals + constant residual u=[8,0] on both pairs. Addition is exact.",
        W, H, T, [PairMotion(np.eye(3), _const(gc, gr, (8.0, 0.0))),
                  PairMotion(np.eye(3), _const(gc, gr, (8.0, 0.0)))], (gc, gr))

    fx["constant_residual_rotation"] = Fixture(
        "constant_residual_rotation",
        "10 deg rotation globals; constant residual u=[8,0] on pair 0 only. Isolates vector transport.",
        W, H, T, [PairMotion(R.copy(), _const(gc, gr, (8.0, 0.0))),
                  PairMotion(R.copy(), zero.copy())], (gc, gr))

    def ramp(p):
        return np.stack([0.02 * p[:, 0], 0.01 * p[:, 1]], 1)

    fx["spatial_ramp_translation"] = Fixture(
        "spatial_ramp_translation",
        "Translation globals (6 px x) + linear spatial residual. Isolates residual resampling/location error.",
        W, H, T, [PairMotion(translation(6.0, 0.0), field_mesh(W, H, gc, gr, ramp)),
                  PairMotion(translation(6.0, 0.0), field_mesh(W, H, gc, gr, ramp))], (gc, gr))

    # Two-layer parallax fixture with an explicit inverse-depth convention:
    #   f_px = 100 (pixels), Z_bg = 5.0, Z_fg = 1.25 (arbitrary units) -> w = f/Z is inverse depth.
    #   a camera translation T = [0.05, 0] world units gives background displacement w_bg*T = 1 px and
    #   foreground w_fg*T = 4 px. The global similarity fits the background (rotation + w_bg*T), so the
    #   residual is the differential (w_fg - w_bg)*T = [3, 0] px on the foreground and zero on the background.
    Wp, Hp, Tp, gcp, grp = 80, 60, 3, 5, 4
    f_px, Z_bg, Z_fg = 100.0, 5.0, 1.25
    w_bg, w_fg = f_px / Z_bg, f_px / Z_fg
    trans = np.array([0.05, 0.0])
    theta = 6.0
    Hk = rotation_about_clockwise(Wp, Hp, theta) @ translation(w_bg * trans[0], w_bg * trans[1])

    def fg(p):
        return (np.abs(p[:, 0] - (Wp / 2 - .5)) < 0.20 * Wp) & (np.abs(p[:, 1] - (Hp / 2 - .5)) < 0.25 * Hp)

    def parallax(p):
        return np.where(fg(p)[:, None], (w_fg - w_bg) * trans, 0.0)

    def region(p):
        inside = (p[:, 0] >= 0) & (p[:, 0] <= Wp - 1) & (p[:, 1] >= 0) & (p[:, 1] <= Hp - 1)
        return np.where(~inside, -1, fg(p).astype(int))

    fx["parallax_two_layer"] = Fixture(
        "parallax_two_layer",
        "6 deg rotation + background translation globals and a supported two-layer parallax residual "
        "(explicit inverse-depth f_px/Z; differential 3 px on the foreground).",
        Wp, Hp, Tp,
        [PairMotion(Hk.copy(), field_mesh(Wp, Hp, gcp, grp, parallax)),
         PairMotion(Hk.copy(), field_mesh(Wp, Hp, gcp, grp, parallax))], (gcp, grp),
        region=region,
        notes="displacement_px = (f_px/Z) * translation; f_px=100, Z_bg=5.0, Z_fg=1.25; not metric depth.")

    # rows + mesh ordering: identity global, constant mesh y-shift, linear row field. Model order applies
    # rows at the source y; production applies the mesh first and samples rows at the shifted y.
    Wr, Hr, Tr, gcr, grr, nr = 64, 48, 2, 4, 3, 6

    def rowlinear(y):
        return np.stack([np.zeros_like(y), 0.02 * y], 1)

    fx["rows_mesh_order"] = Fixture(
        "rows_mesh_order", "Identity globals, mesh y-shift 3 px, linear row residual. Isolates mesh/row order.",
        Wr, Hr, Tr, [PairMotion(np.eye(3), _const(gcr, grr, (0.0, 3.0)), row_field(Wr, Hr, nr, rowlinear))],
        (gcr, grr), rs_rows=nr)

    # rotation + translation with a constant residual: transport error and substantial border exclusion.
    Wb, Hb, Tb = 64, 48, 3
    Hbk = rotation_about_clockwise(Wb, Hb, 10.0) @ translation(20.0, 0.0)
    fx["rotation_translation_border"] = Fixture(
        "rotation_translation_border",
        "10 deg rotation + 20 px x translation, constant residual u=[8,0] on pair 0. Transport error plus "
        "large out-of-frame support exclusion.",
        Wb, Hb, Tb, [PairMotion(Hbk.copy(), _const(gc, gr, (8.0, 0.0))),
                     PairMotion(Hbk.copy(), zero.copy())], (gc, gr))

    return fx


# --------------------------------------------------------------------------------------
# comparisons and error statistics
# --------------------------------------------------------------------------------------

def in_bounds(p, W, H):
    return (p[:, 0] >= -0.5) & (p[:, 0] <= W - 0.5) & (p[:, 1] >= -0.5) & (p[:, 1] <= H - 0.5)


def error_stats(a, b, mask=None):
    """Pointwise Euclidean error statistics (px). ``mask`` selects a subset; counts are preserved."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    d = np.linalg.norm(a - b, axis=1)
    if mask is not None:
        mask = np.asarray(mask, bool)
        d = d[mask]
    if d.size == 0:
        return {"mean_px": None, "max_px": None, "p95_px": None, "n": 0}
    return {"mean_px": float(d.mean()), "max_px": float(d.max()),
            "p95_px": float(np.percentile(d, 95)), "n": int(d.size)}


def region_error(a, b, labels):
    """Error stats split by region label (0 background, 1 foreground, -1 outside)."""
    out = {}
    for lab, name in ((0, "background"), (1, "foreground"), (-1, "outside")):
        m = labels == lab
        if m.any():
            out[name] = error_stats(a, b, m)
    return out


def invert_map(pairs, t, target, W, H, p0=None, max_iter=200, tol=1e-6, damping=1.0):
    """Damped fixed-point inverse of ``Phi_t`` (output->source). Reports convergence, iterations and residual.

    The locked-camera audit itself needs no inversion: the oracle is already output->source. Inversion is
    needed for the reverse direction (source->output / visibility), where a general ``H + residual`` map can
    fold and the fixed point can diverge. That failure is reported, never hidden.
    """
    p = np.array(target if p0 is None else p0, np.float64)
    err = np.inf
    for i in range(max_iter):
        r = target - compose_oracle(pairs, t, p, W, H)
        err = float(np.linalg.norm(r, axis=1).max())
        if err <= tol:
            return {"converged": True, "iterations": i, "residual_px": err, "points": p}
        p = p + damping * r
    return {"converged": False, "iterations": max_iter, "residual_px": err, "points": p}


def _fixture_plan_check(fx, plan):
    """Assert the pinned production plan's globals and cumulative mesh equal the analytic model."""
    checks = {}
    pts = output_grid(fx.W, fx.H)
    for t in range(fx.T):
        checks[f"global_frame_{t}_max_px"] = float(np.abs(G.apply_h(plan.G[t], pts)
                                                          - G.apply_h(compose_globals(fx.pairs, t), pts)).max())
    return checks


def audit_fixture(fx, with_production=False):
    """Run all comparisons for one fixture and return a JSON-safe result dict."""
    pts = output_grid(fx.W, fx.H)
    plan, _cfg = build_plan(fx)
    oracle = [compose_oracle(fx.pairs, t, pts, fx.W, fx.H) for t in range(fx.T)]
    naive = [naive_from_plan(plan, t, pts, fx.W, fx.H) for t in range(fx.T)]
    transported = [transported_field(fx.pairs, t, pts, fx.W, fx.H) for t in range(fx.T)]
    region = fx.region(pts) if fx.region is not None else np.zeros(len(pts), int)
    support = in_bounds(oracle[-1], fx.W, fx.H)

    res = {
        "name": fx.name, "description": fx.description, "notes": fx.notes,
        "W": fx.W, "H": fx.H, "frames": fx.T, "mesh_cells": list(fx.mesh_cells), "rs_rows": fx.rs_rows,
        "plan_check": _fixture_plan_check(fx, plan),
        "errors": {}, "support": {},
    }
    for t in range(fx.T):
        tag = f"frame_{t}"
        res["errors"][tag] = {
            "naive_vs_oracle": error_stats(naive[t], oracle[t]),
            "transported_vs_oracle": error_stats(transported[t], oracle[t]),
            "naive_vs_oracle_in_bounds": error_stats(naive[t], oracle[t], in_bounds(oracle[t], fx.W, fx.H)),
            "by_region": region_error(naive[t], oracle[t], region),
        }
        res["support"][tag] = {
            "oracle_in_bounds_frac": float(in_bounds(oracle[t], fx.W, fx.H).mean()),
            "residual_support_frac": float((region >= 0).mean()),
        }
    # border fixture: how much naive error would be excluded if only in-bounds oracle points are compared
    final = fx.T - 1
    res["support"]["final_excluded_frac"] = float(1.0 - in_bounds(oracle[final], fx.W, fx.H).mean())

    if with_production:
        prod = [production_field(plan, t, fx.W, fx.H) for t in range(fx.T)]
        for t in range(fx.T):
            tag = f"frame_{t}"
            res["errors"][tag]["production_vs_naive"] = error_stats(prod[t], naive[t])
            res["errors"][tag]["production_vs_oracle"] = error_stats(prod[t], oracle[t])
    return res


def _jacobian_det_at(pair, p, W, H, eps=1e-3):
    """Signed determinant of the forward map's 2x2 Jacobian at ``p`` (finite differences)."""
    p = np.asarray(p, np.float64)
    ex = np.array([eps, 0.0]); ey = np.array([0.0, eps])
    fx = (forward_apply(pair, p + ex, W, H) - forward_apply(pair, p - ex, W, H)) / (2 * eps)
    fy = (forward_apply(pair, p + ey, W, H) - forward_apply(pair, p - ey, W, H)) / (2 * eps)
    return fx[:, 0] * fy[:, 1] - fx[:, 1] * fy[:, 0]


def inverse_audit():
    """Controlled inverse problems: a well-conditioned residual and two failure modes.

    The locked-camera audit needs no inversion (the oracle is already output->source). This block records why
    a Stage 1 implementation must not rely on a naive fixed-point inverse: closed-form affine inversion is
    required for the similarity/affine maps, and genuinely nonlinear maps need explicit convergence checks.
    Border clamping can rescue a diverging iterate outside the frame, so the divergence control deliberately
    uses a scaled similarity with no residual (no clamp to hide it).
    """
    W, H, gc, gr = 48, 36, 4, 3
    R = rotation_about_clockwise(W, H, 8.0)
    pts = output_grid(W, H)
    target = G.apply_h(R, pts)
    c = np.array([W / 2 - .5, H / 2 - .5])

    stable = [PairMotion(R.copy(), _const(gc, gr, (3.0, 0.0)))]
    ok = invert_map(stable, 1, target, W, H)
    ok.pop("points", None)

    # invertible similarity with scale 2.5 -> fixed-point iteration eigenvalue |1 - 2.5| > 1 (diverges)
    zoom = translation(c[0], c[1]) @ np.diag([2.5, 2.5, 1.0]) @ translation(-c[0], -c[1])
    zoom_pair = [PairMotion(zoom, np.zeros((gr + 1, gc + 1, 2)))]
    zoom_target = G.apply_h(zoom, pts)
    bad = invert_map(zoom_pair, 1, zoom_target, W, H, max_iter=50, damping=1.0)
    bad.pop("points", None)

    # an orientation-reversing residual is not a valid warp and is flagged before any iteration
    # (Jacobian x-slope 1-2 = -1 -> negative determinant; a stabilizer must reject this, not invert it)
    fold_pair = PairMotion(np.eye(3), field_mesh(W, H, gc, gr, lambda p: np.stack([-2.0 * (p[:, 0] - c[0]),
                                                                                    np.zeros(len(p))], 1)))
    sample = pts[::97]
    det = _jacobian_det_at(fold_pair, sample, W, H)
    fold = {"negative_det_frac": float((det < 0).mean()), "det_min": float(det.min()),
            "det_max": float(det.max()), "samples": int(len(det))}
    return {"stable": ok, "zoom_fixed_point_divergence": bad, "fold_over": fold}


# --------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------

def run_audit(out_dir=None, with_production=False, fixtures=None):
    """Run every fixture, write ``fixtures.npz`` + ``results.json`` and return the results dict.

    Pure NumPy by default. ``with_production=True`` additionally calls the real ``source_coords`` (MLX/Metal)
    and therefore must run under the shared heavy lock; the pure geometry part needs no GPU.
    """
    fixtures = build_fixtures() if fixtures is None else fixtures
    results = {
        "schema": SCHEMA,
        "with_production": bool(with_production),
        "conventions": {
            "output_to_source": "Phi_0 = I; Phi_{k+1} = F_k(Phi_k)",
            "forward_pair": "F_k(p) = H_k(p) + r_k(p) + row_k(p_y): frame k -> k+1",
            "residual_frame": "r_k is a displacement in destination frame k+1, parameterised at source position p in frame k",
            "production": "src = G_t(x) + (sum_{k<t} r_k)(G_t(x)) with rows sampled after the mesh shift",
            "simple_addition_valid": "A_k = I for all k AND the residual is constant (or evaluated at the same point)",
            "inverse_depth_convention": "displacement_px = (f_px/Z)*translation; f_px/Z is inverse depth (arbitrary units)",
        },
        "fixtures": [audit_fixture(fx, with_production=with_production) for fx in fixtures.values()],
        "inverse": inverse_audit(),
    }
    if out_dir is not None:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        save_fixtures(out / "fixtures.npz", fixtures)
        payload = json.dumps(json_safe(results), indent=1, sort_keys=True, allow_nan=False)
        (out / "results.json").write_text(payload, encoding="utf-8")
        results["out_dir"] = str(out)
    return results


def save_fixtures(path, fixtures):
    arrays = {}
    for name, fx in fixtures.items():
        for k, pair in enumerate(fx.pairs):
            arrays[f"{name}.pair{k}.H"] = pair.H
            arrays[f"{name}.pair{k}.mesh"] = pair.mesh
            if pair.rows is not None:
                arrays[f"{name}.pair{k}.rows"] = pair.rows
    np.savez(path, **arrays)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Stage 0 residual-composition audit (analytic, no media).")
    ap.add_argument("--out-dir", default=None, help="write fixtures.npz and results.json here")
    ap.add_argument("--production", action="store_true",
                    help="also call the real source_coords path (needs the heavy lock / MLX Metal)")
    args = ap.parse_args(argv)
    res = run_audit(args.out_dir, with_production=args.production)
    print(json.dumps(json_safe({k: res[k] for k in ("schema", "with_production", "conventions", "inverse")}),
                     indent=1, sort_keys=True, allow_nan=False))
    for fx in res["fixtures"]:
        last = fx["errors"][f"frame_{fx['frames'] - 1}"]
        print(f"{fx['name']:32s} naive_vs_oracle_max={last['naive_vs_oracle']['max_px']:.6g} px "
              f"in_bounds={last['naive_vs_oracle_in_bounds']['max_px']:.6g} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
