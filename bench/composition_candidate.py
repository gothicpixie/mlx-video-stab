"""Stage 1 candidate: frame-consistent residual composition (NumPy only; NOT wired into production).

This module is the next experiment after ``bench/composition_audit.py``. It keeps the existing smoothed
global path exactly and replaces the fixed-grid residual sum with a frame-consistent composition of the
per-pair forward motions.

Model and mapping
-----------------
Per analysed pair ``k`` (``stride == 1``), the fitted forward motion is

    F_k(p) = H_k(p) + rows_k(p_y) + mesh_k(p)          frame k -> k+1

with ``H_k`` the global homography (``Fit.H``), ``rows_k`` the x-independent row residual and ``mesh_k`` the
per-vertex residual. Both residuals are evaluated at their **source** position ``p`` (the fit-measurement
convention), matching ``composition_audit.forward_apply``.

For frame ``t`` (reference = frame 0):

    Psi_t   = H_{t-1} o ... o H_0                      (full projective composition, reference -> t)
    Phi_t   = F_{t-1} o ... o F_0                      (reference -> t, residuals included)
    delta_t = Phi_t - Psi_t                            (source-frame displacement indexed by reference z)
    eta_t   = inv(Psi_t)(Phi_t(z)) - z                 (reference-chart displacement; smoothing_basis="reference")

``delta_t`` (or, in the reference chart, ``eta_t``) is sampled on an explicit reference-coordinate grid and
smoothed over time with the existing ``smooth_columns`` machinery. The correction is the high-pass of the
chosen chart: ``delta_t - smooth(delta_t)`` (source), or the exact projective push-forward
``Psi_t(z + eta_hp) - Psi_t(z)`` with ``eta_hp = eta_t - smooth(eta_t)`` (reference). Both charts are
implemented; ``smoothing_basis`` defaults to ``"source"`` and no winner is selected.

The virtual map is recovered from the global-only production plan as

    V_t = inv(Psi_t) o Plan.G[t]                        (output -> reference)

and the returned output->source field is

    src_t(x) = Psi_t(V_t(x)) + highpass_delta_t(V_t(x))

with ``Psi_t(V_t(x))`` computed as ``Plan.G[t](x)`` so the smoothed global path is preserved **exactly**.
With ``highpass=False`` the function returns ``Phi_t(V_t(x))``, the exact composed oracle, which is what the
analytic validation compares against.

Reference grid, interpolation, numerical scope
----------------------------------------------
- The reference grid is the mesh vertex grid: ``x_i = -0.5 + i*W/gc``, ``y_j = -0.5 + j*H/gr`` for
  ``(gc, gr) = cfg.mesh``; if either is 0, a fallback ``(8, 6)`` grid is used. This is the same grid that
  parameterises ``Fit.mesh``, so a mesh-order-bilinear residual is represented exactly at vertices.
- ``delta_t`` is evaluated **pointwise** (exact projective composition) at any queried reference point; the
  *smoothed* term is defined on the grid and sampled bilinearly with border clamping (the same convention as
  ``warp.sample``/``composition_audit``). Sub-grid accuracy of the smoothed correction is therefore limited
  by the reference grid, while the composition itself is exact at the sampled points.
- The smoothing uses ``cfg.smooth``/``cfg.sigma``/``cfg.l1_weights``/``cfg.fidelity`` and a budget of
  ``cfg.mesh_budget * W`` (the existing residual-correction cap). Note that ``smooth_columns``' L1 solver is a
  no-op for series shorter than 5 frames, so for 2-3 frame fixtures the L1 high-pass is exactly zero and the
  candidate reduces to the smoothed global path; validate the residual channel with ``highpass=False`` or with
  ``smooth="gauss"`` on a longer series.
- **Temporal-smoothing chart is a decision experiment, not a decided convention.** ``smoothing_basis`` selects
  the chart smoothed across ``t``: ``"source"`` (default, current behavior) smooths ``delta_t = Phi_t - Psi_t``,
  a frame-``t`` source-pixel displacement indexed by the reference coordinate; ``"reference"`` smooths
  ``eta_t = inv(Psi_t)(Phi_t(z)) - z`` in the reference chart and maps the high-pass back exactly as
  ``Psi_t(z + eta_hp) - Psi_t(z)`` (no linear/Jacobian block, so projective homographies are supported). Both are
  implemented and neither is selected; see ``SMOOTHING_FRAME_NOTE``. The budget applies to the smoothed chart
  (source pixels vs reference pixels), so the reference-mode source correction is a nonlinear push-forward whose
  magnitude can differ by the local Jacobian of ``Psi_t``.
- All arithmetic is float64 NumPy. No MLX/Metal and no production import are used.

Limitations
-----------
- ``stride == 1`` and ``len(mot.fits) + 1 == len(plan.G)`` are required; anything else raises
  ``CompositionScopeError`` rather than silently composing interpolated maps. Stride>1 defines its global
  path by ``dlt`` of interpolated quads, so an exact composed analogue needs its own interpolation decision.
- The render order differs from production ``source_coords`` (which samples the mesh at the *source*
  coordinate and then samples rows after the mesh shift); this candidate applies the fit order
  global -> rows(p_y) -> mesh(p) at the source position.
- Support/fold reporting is finite-difference based on the rendered pixel grid. ``nonfinite_cells`` (a
  source-coordinate difference is non-finite) and ``overflow_cells`` (finite differences but a non-finite
  corner determinant) are reported separately and excluded from the finite summaries; it is a detector, not a
  proof of global invertibility.
- This is not a production patch and does not change defaults, ``stabilize``, or ``source_coords``.
"""
from __future__ import annotations

import numpy as np

from mlxstab import geometry as G
from mlxstab import smooth as SM

__all__ = [
    "CompositionScopeError", "WarpFoldError",
    "FALLBACK_CELLS", "output_grid", "reference_cells", "reference_points",
    "sample_mesh_bilinear", "sample_rows_bilinear", "sample_grid_bilinear",
    "forward_apply", "global_homography", "compose_reference", "delta_reference",
    "delta_grid", "eta_grid", "eta_reference", "smooth_delta_grid", "smoothed_delta", "virtual_map",
    "SMOOTHING_FRAME_NOTE", "SMOOTHING_BASES",
    "candidate_source_coords", "candidate_field", "global_component",
    "in_bounds", "fold_report", "candidate_report", "validate_scope",
]

FALLBACK_CELLS = (8, 6)

SMOOTHING_BASES = ("source", "reference")

SMOOTHING_FRAME_NOTE = (
    "DECISION EXPERIMENT (no winner selected): the temporal smoothing chart is selectable via "
    "smoothing_basis. 'source' (default, unchanged behavior) smooths delta_t(z) = Phi_t(z) - Psi_t(z), a frame-t "
    "source-pixel displacement vector indexed by the reference coordinate. 'reference' smooths "
    "eta_t(z) = inv(Psi_t)(Phi_t(z)) - z in the reference chart and pushes the high-pass correction back exactly "
    "as Psi_t(z + eta_hp) - Psi_t(z) (no linear/Jacobian block approximation, so projective Psi_t is supported). "
    "Both bases are implemented; neither is selected. The budget cfg.mesh_budget*W is in pixels of the smoothed "
    "chart: source pixels for 'source', reference pixels for 'reference'. The resulting source-space correction "
    "for 'reference' is the nonlinear push-forward and can differ in magnitude from the reference displacement "
    "by the local Jacobian of Psi_t."
)


class CompositionScopeError(ValueError):
    """Raised when a Motion/Plan pair is outside the supported (stride-1, contiguous) composition scope."""


class WarpFoldError(ArithmeticError):
    """Raised in strict mode for a folded/collapsed, non-finite, or determinant-overflowed composed field."""


# --------------------------------------------------------------------------------------
# resampling (bilinear + border, matching warp.sample / composition_audit)
# --------------------------------------------------------------------------------------

def _axis(coord, n):
    """Bilinear tap indices/weights for one coordinate over ``n`` vertices (border semantics)."""
    base = np.floor(coord).astype(np.int64)
    f = coord - base
    i0 = np.clip(base, 0, n - 1)
    i1 = np.clip(base + 1, 0, n - 1)
    return i0, i1, 1.0 - f, f


def _frac(mask):
    """Mean of a boolean mask, or ``None`` for an empty mask."""
    mask = np.asarray(mask, bool)
    return float(mask.mean()) if mask.size else None


def sample_mesh_bilinear(mesh, pts, W, H):
    """Bilinear+border sample of a ``(gr+1, gc+1, 2)`` vertex field at pixel coords ``pts [N,2]``."""
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
    """Bilinear+border sample of a ``(nr, 2)`` row field at pixel rows ``py``."""
    rows = np.asarray(rows, np.float64)
    nr = rows.shape[0]
    v = (np.asarray(py, np.float64) + 0.5) * nr / H - 0.5
    i0, i1, w0, w1 = _axis(v, nr)
    return rows[i0] * w0[:, None] + rows[i1] * w1[:, None]


def sample_grid_bilinear(field, pts, W, H, cells):
    """Bilinear+border sample of a reference-grid displacement field at reference points ``pts``.

    ``cells = (gc, gr)`` is the reference grid; ``field`` is a flattened ``[Nref, 2]`` (or
    ``(gr+1, gc+1, 2)``) displacement field. Queried points are reference coordinates.
    """
    gc, gr = int(cells[0]), int(cells[1])
    fld = np.asarray(field, np.float64).reshape(gr + 1, gc + 1, 2)
    pts = np.asarray(pts, np.float64)
    u = (pts[:, 0] + 0.5) * gc / W
    v = (pts[:, 1] + 0.5) * gr / H
    x0, x1, wx0, wx1 = _axis(u, gc + 1)
    y0, y1, wy0, wy1 = _axis(v, gr + 1)
    w00 = wy0 * wx0; w01 = wy0 * wx1; w10 = wy1 * wx0; w11 = wy1 * wx1
    return (fld[y0, x0] * w00[:, None] + fld[y0, x1] * w01[:, None]
            + fld[y1, x0] * w10[:, None] + fld[y1, x1] * w11[:, None])


# --------------------------------------------------------------------------------------
# grids and scope
# --------------------------------------------------------------------------------------

def output_grid(W, H):
    """Full output pixel-centre grid ``[H*W, 2]`` (x fastest, then y)."""
    xs, ys = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    return np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)


def reference_cells(cfg):
    """Reference-grid cell counts ``(gc, gr)``: ``cfg.mesh`` if enabled, else a documented fallback."""
    mesh = tuple(getattr(cfg, "mesh", (0, 0)))
    gc, gr = int(mesh[0]), int(mesh[1])
    if gc <= 0 or gr <= 0:
        return FALLBACK_CELLS
    return gc, gr


def reference_points(W, H, cells):
    """Reference-grid vertex coordinates ``[(gr+1)*(gc+1), 2]`` covering the frame extent."""
    gc, gr = int(cells[0]), int(cells[1])
    xs = -0.5 + np.arange(gc + 1) * W / gc
    ys = -0.5 + np.arange(gr + 1) * H / gr
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], 1)


def validate_scope(mot, plan=None):
    """Reject Motion/Plan pairs outside the straight-composition scope (stride 1, contiguous frames)."""
    stride = int(getattr(mot, "stride", 1))
    if stride != 1:
        raise CompositionScopeError(f"frame-consistent composition requires stride=1, got {stride!r}")
    n_frames = getattr(mot, "n_frames", None)
    if n_frames:                                     # 0/None means unset (Motion default), not a mismatch
        if int(n_frames) != len(mot.fits) + 1:
            raise CompositionScopeError(
                f"n_frames={int(n_frames)} but len(mot.fits)+1={len(mot.fits) + 1}; interpolation is not composed here")
    if plan is not None and len(plan.G) != len(mot.fits) + 1:
        raise CompositionScopeError(
            f"len(plan.G)={len(plan.G)} but len(mot.fits)+1={len(mot.fits) + 1}; plan and motion are inconsistent")


# --------------------------------------------------------------------------------------
# forward motion, composition, delta
# --------------------------------------------------------------------------------------

def forward_apply(fit, p, W, H):
    """One fitted forward map ``F_k(p) = H_k(p) + rows_k(p_y) + mesh_k(p)`` (residuals at source ``p``)."""
    p = np.asarray(p, np.float64)
    q = G.apply_h(fit.H, p)
    mesh = getattr(fit, "mesh", None)
    if mesh is not None:
        q = q + sample_mesh_bilinear(mesh, p, W, H)
    rows = getattr(fit, "rows", None)
    if rows is not None:
        q = q + sample_rows_bilinear(rows, p[:, 1], H)
    return q


def global_homography(fits, t):
    """``Psi_t = H_{t-1} o ... o H_0`` (apply ``H_0`` first); ``t=0`` is identity."""
    psi = np.eye(3)
    for k in range(t):
        psi = np.asarray(fits[k].H, np.float64) @ psi
    return psi


def compose_reference(fits, t, z, W, H):
    """``Phi_t(z) = F_{t-1} o ... o F_0(z)`` for reference points ``z``."""
    p = np.array(z, np.float64)
    for k in range(t):
        p = forward_apply(fits[k], p, W, H)
    return p


def delta_reference(fits, t, z, W, H):
    """``delta_t(z) = Phi_t(z) - Psi_t(z)`` (source-frame displacement, pointwise exact)."""
    phi = compose_reference(fits, t, z, W, H)
    psi = global_homography(fits, t)
    return phi - G.apply_h(psi, z)


def eta_reference(fits, t, z, W, H):
    """``eta_t(z) = inv(Psi_t)(Phi_t(z)) - z`` (reference-chart displacement, pointwise exact)."""
    phi = compose_reference(fits, t, z, W, H)
    psi_inv = np.linalg.inv(global_homography(fits, t))
    return G.apply_h(psi_inv, phi) - np.asarray(z, np.float64)


def delta_grid(fits, ref_pts, W, H, T):
    """``delta_t`` on the reference grid for every output frame -> ``[T, Nref, 2]``.

    ``ref_pts`` are reference coordinates (e.g. from :func:`reference_points`); ``T`` is the number of output
    frames. Frame 0 is zero by construction.
    """
    ref_pts = np.asarray(ref_pts, np.float64)
    out = np.zeros((int(T), len(ref_pts), 2), np.float64)
    phi = ref_pts.copy()
    psi = np.eye(3)
    for t in range(int(T)):
        out[t] = phi - G.apply_h(psi, ref_pts)
        if t < len(fits):
            phi = forward_apply(fits[t], phi, W, H)
            psi = np.asarray(fits[t].H, np.float64) @ psi
    return out


def eta_grid(fits, ref_pts, W, H, T):
    """``eta_t`` on the reference grid for every output frame -> ``[T, Nref, 2]`` (reference chart).

    ``eta_t(z) = inv(Psi_t)(Phi_t(z)) - z``. Frame 0 is zero by construction.
    """
    ref_pts = np.asarray(ref_pts, np.float64)
    out = np.zeros((int(T), len(ref_pts), 2), np.float64)
    phi = ref_pts.copy()
    psi_inv = np.eye(3)
    for t in range(int(T)):
        out[t] = G.apply_h(psi_inv, phi) - ref_pts
        if t < len(fits):
            phi = forward_apply(fits[t], phi, W, H)
            psi_inv = psi_inv @ np.linalg.inv(np.asarray(fits[t].H, np.float64))
    return out


def smooth_delta_grid(delta, cfg, W):
    """Smooth ``delta [T, Nref, 2]`` over time per component with the existing Config machinery.

    The budget is ``cfg.mesh_budget * W``, matching the residual-correction cap in ``plan``. For mode ``l1``
    and ``T < 5`` the solver is a no-op (returns the input), which is inherited behaviour, not an error.
    """
    delta = np.asarray(delta, np.float64)
    T = delta.shape[0]
    K = int(np.prod(delta.shape[1:]))
    budget = float(getattr(cfg, "mesh_budget", 0.012)) * float(W)
    weights = tuple(getattr(cfg, "l1_weights", (10.0, 1.0, 100.0)))
    fidelity = float(getattr(cfg, "fidelity", 0.0))
    sm = SM.smooth_columns(delta.reshape(T, K), getattr(cfg, "smooth", "l1"),
                           float(getattr(cfg, "sigma", 15.0)), budget, weights, fidelity)
    return np.asarray(sm, np.float64).reshape(delta.shape)


def virtual_map(fits, plan, t):
    """``V_t = inv(Psi_t) o Plan.G[t]`` (output -> reference), the recovered smoothed virtual camera."""
    psi = global_homography(fits, t)
    return np.linalg.inv(psi) @ np.asarray(plan.G[t], np.float64)


# --------------------------------------------------------------------------------------
# candidate field
# --------------------------------------------------------------------------------------

def global_component(plan, t, points):
    """``Plan.G[t](points)``: the preserved smoothed global path (independent of any residual)."""
    return G.apply_h(plan.G[t], np.asarray(points, np.float64))


def _reference_count(cells):
    gc, gr = int(cells[0]), int(cells[1])
    return (gr + 1) * (gc + 1)


def _check_smoothed(smoothed, T, cells):
    """Reject an explicitly supplied smoothed array whose grid size disagrees with ``cells``/the clip length."""
    arr = np.asarray(smoothed)
    nref = _reference_count(cells)
    if arr.ndim != 3 or arr.shape != (int(T), nref, 2):
        raise ValueError(
            f"smoothed delta shape {arr.shape} does not match [T={int(T)}, Nref={nref}, 2] for cells={tuple(int(c) for c in cells)}")


def _check_basis(smoothing_basis):
    """Reject an unknown temporal-smoothing chart."""
    if smoothing_basis not in SMOOTHING_BASES:
        raise ValueError(f"smoothing_basis must be one of {SMOOTHING_BASES}, got {smoothing_basis!r}")


def smoothed_delta(mot, plan, cfg, *, cells=None, smoothing_basis="source"):
    """Precompute the temporally smoothed residual once per clip -> ``[T, Nref, 2]`` in the chosen chart.

    ``smoothing_basis="source"`` (default) smooths ``delta_t``; ``"reference"`` smooths ``eta_t``. Call this
    once and pass the result as ``smoothed=`` to every :func:`candidate_source_coords` call for the clip;
    otherwise the ``T``-long smoothing is recomputed per frame. The budget is in pixels of the chosen chart.
    """
    _check_basis(smoothing_basis)
    validate_scope(mot, plan)
    W, H = int(mot.W), int(mot.H)
    if cells is None:
        cells = reference_cells(cfg)
    ref_pts = reference_points(W, H, cells)
    T = len(plan.G)
    grid = (delta_grid(list(mot.fits), ref_pts, W, H, T) if smoothing_basis == "source"
            else eta_grid(list(mot.fits), ref_pts, W, H, T))
    return smooth_delta_grid(grid, cfg, W)


def candidate_source_coords(mot, plan, cfg, t, points=None, *, cells=None, highpass=True, smoothed=None,
                            smoothing_basis="source"):
    """Frame-consistent output->source field.

    ``mot`` is a Motion (``.W``, ``.H``, ``.fits`` with ``.H/.mesh/.rows``, ``.stride``, ``.n_frames``),
    ``plan`` supplies ``.G``. ``t`` is the output frame index, ``points`` an optional ``[N,2]`` output grid
    (default: the full ``H x W`` grid).

    Returns ``Psi_t(V_t(x)) + correction``; with ``highpass=False`` returns the exact composed map
    ``Phi_t(V_t(x))`` in either chart. With ``highpass=True``, ``smoothing_basis`` selects the chart
    (``"source"``: ``delta - smooth(delta)``; ``"reference"``: exact push-forward of ``eta - smooth(eta)``).
    ``cells`` overrides the reference grid; pass a precomputed ``smoothed`` array ``[T, Nref, 2]`` from
    :func:`smoothed_delta` with the *same* basis to reuse the temporal smoothing across calls.
    """
    _check_basis(smoothing_basis)
    validate_scope(mot, plan)
    W, H = int(mot.W), int(mot.H)
    fits = list(mot.fits)
    T = len(plan.G)
    if not (0 <= int(t) < T):
        raise IndexError(f"frame index {t} outside plan range [0, {T})")
    if points is None:
        points = output_grid(W, H)
    points = np.asarray(points, np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be [N,2] pixel coordinates")

    if cells is None:
        cells = reference_cells(cfg)

    psi = global_homography(fits, int(t))
    V = np.linalg.inv(psi) @ np.asarray(plan.G[int(t)], np.float64)
    z = G.apply_h(V, points)                                     # output -> reference
    base = G.apply_h(plan.G[int(t)], points)                     # == Psi_t(V_t(x)), exactly Plan.G
    delta = delta_reference(fits, int(t), z, W, H)               # pointwise exact composition term

    if not highpass:
        return base + delta                                  # exact Phi_t(V_t(x)) in both charts

    if smoothed is None:
        smoothed = smoothed_delta(mot, plan, cfg, cells=cells, smoothing_basis=smoothing_basis)
    else:
        _check_smoothed(smoothed, T, cells)
    if smoothing_basis == "source":
        smooth_z = sample_grid_bilinear(smoothed[int(t)], z, W, H, cells)
        return base + (delta - smooth_z)
    # reference chart: high-pass eta, then push forward exactly with the full projective Psi_t (no linear block)
    eta = eta_reference(fits, int(t), z, W, H)
    eta_hp = eta - sample_grid_bilinear(smoothed[int(t)], z, W, H, cells)
    psi = global_homography(fits, int(t))
    return base + (G.apply_h(psi, z + eta_hp) - G.apply_h(psi, z))


def candidate_field(mot, plan, cfg, t, *, cells=None, highpass=True, smoothed=None, smoothing_basis="source",
                    strict=False):
    """Full ``[H, W, 2]`` candidate field; ``strict`` raises :class:`WarpFoldError` on any folded cell."""
    W, H = int(mot.W), int(mot.H)
    field = candidate_source_coords(mot, plan, cfg, t, points=None, cells=cells, highpass=highpass,
                                    smoothed=smoothed, smoothing_basis=smoothing_basis).reshape(H, W, 2)
    if strict:
        rep = fold_report(field, W, H)
        if rep["nonfinite_cells"]:
            raise WarpFoldError(f"composed map has {rep['nonfinite_cells']} non-finite cell(s)")
        if rep["overflow_cells"]:
            raise WarpFoldError(f"composed map has {rep['overflow_cells']} overflowed determinant cell(s)")
        if rep["fold_frac"] is not None and rep["fold_frac"] > 0.0:
            raise WarpFoldError(f"composed map folds on {rep['fold_frac']:.4f} of usable cells")
        if rep["collapse_frac"] is not None and rep["collapse_frac"] > 0.0:
            raise WarpFoldError(f"composed map collapses on {rep['collapse_frac']:.4f} of usable cells")
    return field


# --------------------------------------------------------------------------------------
# support / fold reporting
# --------------------------------------------------------------------------------------

def in_bounds(field, W, H):
    """Pixel-edge validity of a ``[H,W,2]`` field: ``-0.5 <= x <= W-0.5`` and likewise for ``y``."""
    field = np.asarray(field, np.float64)
    x, y = field[..., 0], field[..., 1]
    return (x >= -0.5) & (x <= W - 0.5) & (y >= -0.5) & (y <= H - 0.5)


def fold_report(field, W, H, det_tol=1e-9):
    """Four-corner Jacobian determinants of a ``[H,W,2]`` field, per bilinear output cell.

    Follows the documented ``warpmetrics`` convention: each 1x1 cell's four corner Jacobians are built from the
    x- and y-forward differences (``X_top``/``X_bot`` at rows ``[:-1]``/``[1:]``, ``Y_left``/``Y_right`` at
    columns ``[:-1]``/``[1:]``). A cell is folded when *any* corner reverses orientation (``det < -det_tol``)
    or collapses (``|det| <= det_tol``), so an opposite-corner fold that a top-left-only Jacobian would miss is
    detected. Two failure categories are reported separately and both are excluded from the finite determinant
    summaries: ``nonfinite_cells`` (a source-coordinate difference is non-finite) and ``overflow_cells`` (all
    four differences are finite but a corner determinant is non-finite, e.g. a finite huge-scale field whose
    products exceed float64). ``valid_cells`` counts cells with finite differences; ``usable_cells`` counts
    finite differences *and* finite determinants. All summaries are ``None`` when there are no usable cells.
    """
    field = np.asarray(field, np.float64).reshape(H, W, 2)
    empty = {"cells": 0, "valid_cells": 0, "nonfinite_cells": 0, "nonfinite_frac": None,
             "overflow_cells": 0, "overflow_frac": None, "usable_cells": 0,
             "fold_frac": None, "collapse_frac": None, "det_min": None, "det_max": None, "det_abs_min": None}
    if H < 2 or W < 2:
        return empty
    # Non-finite fields subtract/multiply to NaN, and huge-but-finite fields can overflow the determinant to
    # +-inf. Suppress the FP warnings (mirroring warpmetrics' overflow convention); those cells are reported
    # explicitly via ``nonfinite_cells``/``overflow_cells`` below, so the report is not silenced, only the noise.
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        dxx = field[:, 1:, :] - field[:, :-1, :]                       # [H, W-1, 2]
        dyy = field[1:, :, :] - field[:-1, :, :]                       # [H-1, W, 2]
        X_top, X_bot = dxx[:H - 1, :W - 1], dxx[1:H, :W - 1]
        Y_left, Y_right = dyy[:H - 1, :W - 1], dyy[:H - 1, 1:W]

        def _det(u, v):
            return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

        dets = np.stack([_det(X_top, Y_left), _det(X_top, Y_right),
                         _det(X_bot, Y_left), _det(X_bot, Y_right)], -1)   # [H-1, W-1, 4]
        finite = (np.isfinite(X_top).all(-1) & np.isfinite(X_bot).all(-1)
                  & np.isfinite(Y_left).all(-1) & np.isfinite(Y_right).all(-1))
        det_finite = np.isfinite(dets).all(-1)
    nonfinite = ~finite
    overflow = finite & ~det_finite
    usable = finite & det_finite
    use = dets[usable]                                             # [M, 4]
    out = {"cells": int(finite.size), "valid_cells": int(finite.sum()),
           "nonfinite_cells": int(nonfinite.sum()), "nonfinite_frac": _frac(nonfinite),
           "overflow_cells": int(overflow.sum()), "overflow_frac": _frac(overflow),
           "usable_cells": int(usable.sum())}
    if use.size == 0:
        out.update({"fold_frac": None, "collapse_frac": None, "det_min": None, "det_max": None,
                    "det_abs_min": None})
        return out
    out.update({"fold_frac": float((use < -det_tol).any(axis=1).mean()),
                "collapse_frac": float((np.abs(use) <= det_tol).any(axis=1).mean()),
                "det_min": float(use.min()), "det_max": float(use.max()),
                "det_abs_min": float(np.abs(use).min())})
    return out


def candidate_report(mot, plan, cfg, t, *, cells=None, highpass=True, smoothed=None, smoothing_basis="source"):
    """Geometric report for one candidate frame: support fraction, fold fraction and determinant range."""
    W, H = int(mot.W), int(mot.H)
    field = candidate_field(mot, plan, cfg, t, cells=cells, highpass=highpass, smoothed=smoothed,
                            smoothing_basis=smoothing_basis)
    rep = fold_report(field, W, H)
    rep.update({"frame": int(t), "W": W, "H": H,
                "in_bounds_frac": float(in_bounds(field, W, H).mean()),
                "residual_scope": "highpass" if highpass else "raw_composition",
                "smoothing_basis": smoothing_basis,
                "smoothing_frame_semantics": SMOOTHING_FRAME_NOTE})
    return rep
