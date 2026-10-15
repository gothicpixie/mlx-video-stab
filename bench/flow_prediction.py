"""E04 prediction experiment: coordinate-correct flow initialization (NumPy only; NOT wired into production).

This is the first *isolated* E04 experiment from ``PLAN.md``. It compares flow-initialization candidates on
**exact, analytically known** synthetic frame-to-frame fields so that the only thing under test is the
coordinate convention and the temporal extrapolation model. It does not call RAFT, does not touch
``mlxstab``, does not change any default, and makes **no RAFT-quality or real-footage claim**. It answers the
E04 sub-question "do transported / predicted initializations actually land closer to the true next flow than
cold or the current same-grid reuse?" *before* anyone wires prediction into the production flow estimator.

Coordinate convention (pixel centres on integers)
--------------------------------------------------
A ``W x H`` frame spans ``x in [-0.5, W-0.5]``, ``y in [-0.5, H-0.5]`` with ``x`` right and ``y`` down.
Pixel centres are the integer lattice ``(x, y)``. A field is stored as ``[H, W, 2]`` (component order
``(x, y)``); helper grids are flat ``[N, 2]`` with ``N = H*W``.

Per pair ``k`` (frame ``k -> k+1``) the forward map is

    Phi_k(p) = p + f_k(p)                                     frame k -> k+1

and the *flow* ``f_k`` is defined on the **source frame k grid**, i.e. at a frame-k pixel centre ``p`` it is
the displacement to the corresponding point in frame k+1. The backward direction is the inverse map

    Phi_k^{-1}(q) = q + b_k(q)                                frame k+1 -> k

with the backward flow ``b_k`` defined on the **frame k+1 source grid**. Both directions are handled by the
same machinery: a backward prediction is just a prediction over ``Pair.reversed()`` maps (see
:func:`reversed_fixture` and the backward test).

Why relabelling is wrong
------------------------
The production warm start reuses the previous pair's low-resolution flow array **at the same grid indices**
(``stabilize.RaftFlow.bidict`` / ``flow_bidir`` pass ``prev_f`` straight through as ``init_f``). Because the
network grid is identical for every pair, that is an array relabel. Semantically the previous flow lives on
frame ``k``'s grid while the next initialization must live on frame ``k+1``'s grid, so the field must be
carried by the previous source-to-target map:

    transported_k(q) = f_k( Phi_k^{-1}(q) )                   pull the previous field back to the new grid

:func:`transport` performs exactly this composition (through several pairs when needed) and tracks a validity
mask so that border / occluded / disoccluded samples are excluded rather than silently clamped into the
result. Validity is sampled conservatively (every bilinear tap with non-zero weight must be valid,
following ``motion._sample_invalid``; a zero-weight neighbour cannot reject an exact pixel centre).

Candidates
----------
All candidates produce a field on the **frame-``target`` source grid** (``target = k+1``) to initialize pair
``target``. Let ``g_j`` be the field of pair ``j`` pulled back onto the ``target`` grid by :func:`transport`:

    cold                  zero field
    same_grid             f_{target-1} taken at the same indices (the current production reuse)
    transported           g_{target-1}                                   (order 0)
    constant_velocity     2 g_{target-1} - g_{target-2}                  (order 1)
    constant_acceleration 3 g_{target-1} - 3 g_{target-2} + g_{target-3} (order 2)

The order-1/2 names follow E04 / ``PLAN.md`` Stage 2 ("constant velocity, then compare constant
acceleration"). The extrapolated quantity is the *transported displacement field* itself: order 1 holds its
first temporal difference constant and order 2 holds its second difference constant. This is the flow-domain
analogue of the camera-motion prediction in the plan; it is deliberately **not** a claim that a real camera
has constant velocity or acceleration, and the module reports the assumption rather than asserting physics.
The order-0 candidate ``transported`` is the "the tracked correspondence keeps its image displacement"
model.

Fallback
--------
:func:`predict` reports candidate validity separately from error and applies a safe fallback: if a candidate
needs more history than exists, produces non-finite values **on its valid support**, or covers less than
``min_coverage`` of the frame, the effective initialization is **cold (zeros)** and the prediction is flagged
``fallback=True`` with a machine-readable ``reason``. A non-finite value at an already-invalid cell is masked
out by the candidate validity and does not trigger the non-finite fallback. The pre-fallback coverage is kept
as ``raw_coverage`` so a low-coverage candidate is reported, not hidden. When there is no candidate at all
(``insufficient_history``), ``raw_valid`` is all false with ``coverage == 0`` while the effective cold
validity is all true.

Fixtures (exact, deterministic, in memory)
------------------------------------------
``translation_constant`` / ``translation_linear`` / ``translation_quadratic`` vary the global translation
between pairs (spatially constant fields; ``same_grid`` and ``transported`` agree in displacement values on
their shared valid support, although their validity masks differ). ``translation_out`` uses a translation
large enough to leave only 25% valid support, exercising the low-coverage fallback. ``rotation_const`` is a
constant per-pair rotation (a spatially varying field that is *steady in image coordinates*, so
``same_grid`` is exact and transport is not). ``rotation_rate`` increases the per-pair angle. ``projective``
repeats one well-conditioned non-affine homography, so its image-coordinate flow is stationary and
``same_grid`` is exact; transport's dominant error is model mismatch, with bilinear discretization only a
smaller extra. ``parallax_two_layer`` is a two-layer cartoon with an explicit occlusion / disocclusion
structure and per-region labels; ``parallax_two_layer_backward`` is the same clip traversed backwards.

Assumptions and limits
----------------------
- The experiment is **analytic**. It uses the exact pair maps and exact fields as the transport source; it
  isolates the coordinate/extrapolation question and does *not* include RAFT estimation error, feature
  resolution, correlation lookup, or any real imagery. It cannot establish a production speedup or a quality
  change; E04's timing/quality gate requires the production integration this module deliberately avoids.
- ``constant_acceleration`` needs three prior pairs, so it is only defined for ``target >= 3``. Its
  transport chains two/three inverse maps; compounding validity and any per-pair map error are reported, not
  hidden.
- The two-layer fixture is a 2D cartoon (two constant translations plus rectangle layers), not a physical
  occlusion model; it is exact for the correspondence/validity question but is not metric depth.
- ``rotation_const`` and the repeated ``projective`` homography show a real limitation: when the flow field
  is steady in *image* coordinates relabelling (``same_grid``) is already optimal and scene transport moves
  the sample to the wrong place, whereas for translation/parallax transport is the correct coordinate
  operation. Which wins is motion dependent; this module exposes the difference and does not crown a winner.

Run: ``.venv/bin/python -m bench.flow_prediction`` (pure NumPy, no GPU, no heavy lock).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .manifest import json_safe

SCHEMA = "mlxstab.flow_prediction/1"

METHODS = ("cold", "same_grid", "transported", "constant_velocity", "constant_acceleration")
_HISTORY = {"cold": 0, "same_grid": 1, "transported": 1, "constant_velocity": 2, "constant_acceleration": 3}
_REGION_NAMES = (-1, 0, 1, 3)
_REGION_LABELS = {-1: "outside", 0: "background", 1: "foreground", 3: "occluded_source"}
MIN_COVERAGE = 0.35

CONVENTIONS = {
    "pixels": "pixel centres on integers; a WxH frame spans [-0.5, W-0.5] x [-0.5, H-0.5], x right, y down",
    "field_layout": "[H, W, 2] with component order (x, y)",
    "forward_pair": "Phi_k(p) = p + f_k(p): frame k -> frame k+1; f_k is defined on the frame-k source grid",
    "backward_pair": "Phi_k^{-1}(q) = q + b_k(q): frame k+1 -> k; b_k is defined on the frame-k+1 source grid",
    "transport": "transported_k(q) = f_k(Phi_k^{-1}(q)); compose inverses for older pairs; validity is ANDed",
    "same_grid": "the current production reuse: f_{target-1} at identical grid indices (array relabel, no transport)",
    "orders": "transported=order0; constant_velocity=2g0-g1; constant_acceleration=3g0-3g1+g2 (g = transported)",
    "validity": "conservative: every bilinear tap with non-zero weight must be valid (a zero-weight "
                "neighbour cannot reject an exact pixel centre); out-of-frame and occluded/disoccluded "
                "samples excluded",
    "fallback": "insufficient history, non-finite, or coverage < min_coverage -> cold zeros with a reason flag",
    "scope": "analytic initialization only; no RAFT quality, no real footage, no production timing",
}


# --------------------------------------------------------------------------------------
# geometry and resampling (NumPy only, matching the repository pixel-centre convention)
# --------------------------------------------------------------------------------------

def apply_h(M: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography ``M`` to pixel points ``p [N,2]`` (pixel-centre convention)."""
    p = np.asarray(p, np.float64)
    q = np.concatenate([p, np.ones((len(p), 1))], 1) @ np.asarray(M, np.float64).T
    return q[:, :2] / q[:, 2:3]


def in_bounds(p: np.ndarray, W: int, H: int) -> np.ndarray:
    """Pixel-edge validity of points in a ``W x H`` frame."""
    p = np.asarray(p, np.float64)
    return (p[:, 0] >= -0.5) & (p[:, 0] <= W - 0.5) & (p[:, 1] >= -0.5) & (p[:, 1] <= H - 0.5)


def source_grid(W: int, H: int) -> np.ndarray:
    """Full source pixel-centre grid ``[H*W, 2]`` (row-major, x fastest)."""
    xs, ys = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    return np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)


def _taps(coord: np.ndarray, n: int):
    """Bilinear tap indices/weights for one coordinate over ``n`` samples (border clamping)."""
    base = np.floor(coord).astype(np.int64)
    f = coord - base
    i0 = np.clip(base, 0, n - 1)
    i1 = np.clip(base + 1, 0, n - 1)
    return i0, i1, 1.0 - f, f


def sample_field(field: np.ndarray, pts: np.ndarray, W: int, H: int) -> np.ndarray:
    """Bilinear + border sample of a ``[H,W,2]`` (or flat ``[H*W,2]``) field at pixel points ``pts``."""
    fld = np.asarray(field, np.float64).reshape(H, W, 2)
    pts = np.asarray(pts, np.float64)
    x0, x1, wx0, wx1 = _taps(pts[:, 0], W)
    y0, y1, wy0, wy1 = _taps(pts[:, 1], H)
    return (fld[y0, x0] * (wy0 * wx0)[:, None] + fld[y0, x1] * (wy0 * wx1)[:, None]
            + fld[y1, x0] * (wy1 * wx0)[:, None] + fld[y1, x1] * (wy1 * wx1)[:, None])


def sample_mask_conservative(mask: np.ndarray, pts: np.ndarray, W: int, H: int) -> np.ndarray:
    """Sample a boolean validity mask at ``pts``.

    True only when **every bilinear tap with non-zero weight** is valid, matching the repository rule in
    :func:`mlxstab.motion._sample_invalid`. At an exact pixel centre only one tap has weight, so an invalid
    zero-weight neighbour cannot reject the sample; a fractional position with weight on an invalid
    neighbour rejects it.
    """
    m = np.asarray(mask, bool).reshape(H, W)
    pts = np.asarray(pts, np.float64)
    x0, x1, wx0, wx1 = _taps(pts[:, 0], W)
    y0, y1, wy0, wy1 = _taps(pts[:, 1], H)
    ok = np.ones(len(pts), bool)
    for yi, wy in ((y0, wy0), (y1, wy1)):
        for xi, wx in ((x0, wx0), (x1, wx1)):
            w = wy * wx
            ok &= (w <= 0.0) | m[yi, xi]
    return ok


def rotation_about_clockwise(W: int, H: int, deg_cw: float) -> np.ndarray:
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


def translation(dx: float, dy: float) -> np.ndarray:
    """3x3 translation by ``(dx, dy)`` pixel units."""
    M = np.eye(3); M[0, 2] = dx; M[1, 2] = dy
    return M


# --------------------------------------------------------------------------------------
# pair maps
# --------------------------------------------------------------------------------------

@dataclass
class Pair:
    """One exact frame-to-frame map with forward/inverse callables and visibility predicates.

    ``forward(p)`` maps frame-``k`` points to frame-``k+1``; ``inverse(q)`` is its inverse.
    ``forward_valid(p)`` is True when a source point has an observable (in-frame, unoccluded) correspondence;
    ``inverse_valid(q)`` is True when a target point has a unique in-frame preimage (so it is not a
    disocclusion). Both take their own frame's coordinates.
    """
    name: str
    forward: Callable[[np.ndarray], np.ndarray]
    inverse: Callable[[np.ndarray], np.ndarray]
    forward_valid: Callable[[np.ndarray], np.ndarray]
    inverse_valid: Callable[[np.ndarray], np.ndarray]

    def reversed(self) -> "Pair":
        """The same map traversed backwards; the backward flow is predicted with this pair list."""
        return Pair(self.name + "_rev", self.inverse, self.forward,
                    self.inverse_valid, self.forward_valid)


def global_pair(M: np.ndarray, W: int, H: int, name: str = "global") -> Pair:
    """A global homography pair; visibility is in-frame source and destination."""
    Ms = np.asarray(M, np.float64)
    Mi = np.linalg.inv(Ms)

    def forward(p):
        return apply_h(Ms, p)

    def inverse(q):
        return apply_h(Mi, q)

    def fwd_valid(p):
        return in_bounds(p, W, H) & in_bounds(forward(p), W, H)

    def inv_valid(q):
        return in_bounds(q, W, H) & in_bounds(inverse(q), W, H)

    return Pair(name, forward, inverse, fwd_valid, inv_valid)


@dataclass
class Rect:
    """Axis-aligned rectangle (centre + half-extents) used for the two-layer fixture."""
    cx: float
    cy: float
    hx: float
    hy: float

    def contains(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, np.float64)
        return (np.abs(p[:, 0] - self.cx) <= self.hx) & (np.abs(p[:, 1] - self.cy) <= self.hy)

    def shifted(self, dx: float, dy: float) -> "Rect":
        return Rect(self.cx + dx, self.cy + dy, self.hx, self.hy)


def layer_pair(src_fg: Rect, dst_fg: Rect, bg: np.ndarray, fg: np.ndarray,
               W: int, H: int, name: str = "layer") -> Pair:
    """Two-layer parallax pair.

    The source frame's foreground is ``src_fg`` (background is its complement); the destination frame's
    foreground is ``dst_fg = src_fg + fg``. Foreground moves by ``fg``, background by ``bg`` and is occluded
    where it lands under the destination foreground. Inverse validity rejects disoccluded destination
    background (a visible background target whose preimage was itself foreground in the source frame).
    """
    bg = np.asarray(bg, np.float64)
    fg = np.asarray(fg, np.float64)

    def forward(p):
        infg = src_fg.contains(p)
        return np.where(infg[:, None], p + fg, p + bg)

    def inverse(q):
        indst = dst_fg.contains(q)
        return np.where(indst[:, None], q - fg, q - bg)

    def fwd_valid(p):
        infg = src_fg.contains(p)
        q_bg = p + bg
        occupied = (~infg) & in_bounds(q_bg, W, H) & dst_fg.contains(q_bg)
        return in_bounds(p, W, H) & in_bounds(forward(p), W, H) & ~occupied

    def inv_valid(q):
        indst = dst_fg.contains(q)
        p_fg = q - fg
        p_bg = q - bg
        fg_ok = indst & src_fg.contains(p_fg) & in_bounds(p_fg, W, H)
        bg_ok = (~indst) & (~src_fg.contains(p_bg)) & in_bounds(p_bg, W, H)
        return in_bounds(q, W, H) & (fg_ok | bg_ok)

    return Pair(name, forward, inverse, fwd_valid, inv_valid)


# --------------------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------------------

def transport(field: np.ndarray, field_valid: np.ndarray, pairs: Sequence[Pair],
              frm: int, target: int, grid: np.ndarray, W: int, H: int):
    """Express the discrete field of pair ``frm`` on the source grid of pair ``target``.

    ``field`` is the ``[H,W,2]`` flow of pair ``frm`` (defined on frame ``frm``); ``field_valid`` its
    ``[H,W]`` validity. For ``target > frm`` the frame-``target`` grid is pulled back through the inverse maps
    ``Phi_{target-1}^{-1} ... Phi_frm^{-1}`` (all inverses are ANDed); for ``target < frm`` the forward maps
    are composed instead. The sampled value is the previous field at the preimage, so the *value* is carried
    with the *point* rather than relabelled.

    Returns ``(values [N,2], valid [N] bool)``. Any non-finite coordinate, invalid intermediate map, or
    invalid bilinear tap invalidates that sample; invalid samples are zero-filled.
    """
    z = np.array(grid, np.float64)
    valid = np.ones(len(z), bool)
    if target > frm:
        for k in range(target - 1, frm - 1, -1):
            valid &= pairs[k].inverse_valid(z)
            z = pairs[k].inverse(z)
    elif target < frm:
        for k in range(target, frm):
            valid &= pairs[k].forward_valid(z)
            z = pairs[k].forward(z)
    valid &= np.isfinite(z).all(axis=1)

    vals = np.zeros_like(z)
    idx = np.flatnonzero(valid)
    if idx.size:
        zz = z[idx]
        vals[idx] = sample_field(field, zz, W, H)
        ok = sample_mask_conservative(field_valid, zz, W, H)
        ok &= np.isfinite(vals[idx]).all(axis=1)
        new_valid = np.zeros_like(valid)
        new_valid[idx] = ok
        valid = new_valid
    return vals, valid


# --------------------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------------------

@dataclass
class Prediction:
    """One candidate initialization plus its validity and fallback state.

    ``field``/``valid`` are the **effective** initialization after any fallback (cold zeros when a fallback
    fired); ``raw_field``/``raw_valid`` are the candidate before fallback, so a low-coverage candidate can
    still be scored on the support it actually had. ``coverage`` is the raw candidate valid fraction. When the
    fallback is ``insufficient_history`` there is no candidate at all, so ``raw_valid`` is all false and
    ``coverage`` is 0 while the effective ``valid`` (cold) is all true.
    """
    method: str
    field: np.ndarray          # [N,2] effective field (cold zeros when a fallback fired)
    valid: np.ndarray          # [N] effective validity
    raw_field: np.ndarray      # [N,2] candidate field before fallback
    raw_valid: np.ndarray      # [N] candidate validity before fallback
    coverage: float            # raw candidate valid fraction BEFORE any fallback
    fallback: bool
    reason: Optional[str]


def predict(pairs: Sequence[Pair], fields: np.ndarray, valids: np.ndarray, target: int,
            method: str, W: int, H: int, min_coverage: float = MIN_COVERAGE,
            grid: Optional[np.ndarray] = None) -> Prediction:
    """Build one initialization candidate for pair ``target`` on its source grid.

    ``fields``/``valids`` are the exact per-pair fields/validities for pairs ``0..target-1`` (and beyond; only
    the needed history is read). ``target`` is the pair index being initialized, so history indices are
    ``target-1, target-2, ...``. See the module docstring for the candidate definitions and fallback rules.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if grid is None:
        grid = source_grid(W, H)
    grid = np.asarray(grid, np.float64)
    N = len(grid)
    cold_field = np.zeros((N, 2), np.float64)
    cold_valid = np.ones(N, bool)

    def no_candidate(reason):
        """No candidate exists at all: empty raw metrics, effective initialization is cold."""
        return Prediction(method, cold_field, cold_valid, cold_field,
                          np.zeros(N, bool), 0.0, True, reason)

    history = _HISTORY[method]
    if method == "cold":
        return Prediction(method, cold_field, cold_valid, cold_field, cold_valid, 1.0, False, None)
    if int(target) < history:
        return no_candidate("insufficient_history")

    if method == "same_grid":
        pred = fields[target - 1].reshape(N, 2).copy()
        valid = valids[target - 1].reshape(N).astype(bool).copy()
    else:
        terms = [transport(fields[target - 1 - j], valids[target - 1 - j], pairs,
                           target - 1 - j, target, grid, W, H) for j in range(history)]
        if method == "transported":
            pred, valid = terms[0][0].copy(), terms[0][1].copy()
        elif method == "constant_velocity":
            (g0, v0), (g1, v1) = terms
            pred, valid = 2.0 * g0 - g1, v0 & v1
        else:  # constant_acceleration
            (g0, v0), (g1, v1), (g2, v2) = terms
            pred, valid = 3.0 * g0 - 3.0 * g1 + g2, v0 & v1 & v2

    coverage = float(valid.mean()) if valid.size else 0.0
    raw = np.where(valid[:, None], pred, 0.0)
    # Check finiteness of the *masked* candidate: a non-finite value at an already-invalid cell is
    # discarded by ``raw`` and must not trigger a fallback.
    if not np.isfinite(raw).all():
        return Prediction(method, cold_field, cold_valid, raw, valid, coverage, True, "non_finite")
    if coverage <= 0.0:
        return Prediction(method, cold_field, cold_valid, raw, valid, 0.0, True, "zero_coverage")
    if coverage < min_coverage:
        return Prediction(method, cold_field, cold_valid, raw, valid, coverage, True, "low_coverage")
    return Prediction(method, raw, valid, raw, valid, coverage, False, None)


# --------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------

def _stats(values: np.ndarray, mask: np.ndarray) -> dict:
    v = np.asarray(values, np.float64)[np.asarray(mask, bool)]
    if v.size == 0:
        return {"n": 0, "mean_px": None, "p95_px": None, "max_px": None}
    return {"n": int(v.size), "mean_px": float(v.mean()),
            "p95_px": float(np.percentile(v, 95)), "max_px": float(v.max())}


def evaluate(pred_field: np.ndarray, pred_valid: np.ndarray, truth_field: np.ndarray,
             truth_valid: np.ndarray, region: Optional[np.ndarray] = None,
             raw_field: Optional[np.ndarray] = None, raw_valid: Optional[np.ndarray] = None) -> dict:
    """Endpoint error of one candidate against the exact next field, with coverage reported separately.

    ``pred_field``/``pred_valid`` are the **effective** initialization (cold after a fallback);
    ``raw_field``/``raw_valid`` default to them and describe the candidate before fallback. Error is the
    Euclidean endpoint error (px). ``coverage`` and ``support`` use the raw candidate mask, so a candidate
    can score a small error on a tiny support -- both are always reported. ``effective_error`` is the
    post-fallback error actually realized by the policy (cold when a fallback fired).
    """
    raw_field = pred_field if raw_field is None else raw_field
    raw_valid = pred_valid if raw_valid is None else raw_valid
    tf = np.asarray(truth_field, np.float64)
    truth_valid = np.asarray(truth_valid, bool)
    raw_valid = np.asarray(raw_valid, bool)
    eff_valid = np.asarray(pred_valid, bool)
    raw_support = raw_valid & truth_valid
    eff_support = eff_valid & truth_valid
    cand_diff = np.linalg.norm(np.asarray(raw_field, np.float64) - tf, axis=1)
    eff_diff = np.linalg.norm(np.asarray(pred_field, np.float64) - tf, axis=1)
    out = {
        "coverage": float(raw_valid.mean()),
        "effective_coverage": float(eff_valid.mean()),
        "support": float(raw_support.mean()),
        "truth_valid_frac": float(truth_valid.mean()),
        "candidate_error": _stats(cand_diff, raw_support),
        "effective_error": _stats(eff_diff, eff_support),
    }
    if region is not None:
        region = np.asarray(region)
        out["by_region"] = {_REGION_LABELS[lab]: _stats(cand_diff, raw_support & (region == lab))
                            for lab in _REGION_NAMES if np.any(region == lab)}
    return out


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------

@dataclass
class Fixture:
    """A deterministic analytic clip: exact pair maps, source-grid fields/validities and optional regions."""
    name: str
    description: str
    W: int
    H: int
    pairs: list
    fields: np.ndarray          # [n_pairs, H, W, 2]
    valids: np.ndarray          # [n_pairs, H, W]
    region: Optional[Callable[[int, np.ndarray], np.ndarray]] = None
    notes: str = ""

    @property
    def frames(self) -> int:
        return len(self.pairs) + 1


def _fields_from_pairs(pairs: Sequence[Pair], W: int, H: int):
    """Exact source-grid fields ``f_k`` and forward validities for each pair."""
    grid = source_grid(W, H)
    n = len(pairs)
    fields = np.zeros((n, H, W, 2), np.float64)
    valids = np.zeros((n, H, W), bool)
    for k, pair in enumerate(pairs):
        q = pair.forward(grid)
        fields[k] = (q - grid).reshape(H, W, 2)
        valids[k] = pair.forward_valid(grid).reshape(H, W)
    return fields, valids


def _translation_fixture(name: str, desc: str, W: int, H: int, seq: Sequence[Sequence[float]],
                         notes: str = "") -> Fixture:
    pairs = [global_pair(translation(dx, dy), W, H, f"{name}{k}") for k, (dx, dy) in enumerate(seq)]
    fields, valids = _fields_from_pairs(pairs, W, H)
    return Fixture(name, desc, W, H, pairs, fields, valids, notes=notes)


def _layer_fixture(name: str, desc: str, W: int, H: int, rect0: Rect,
                   bg_seq: Sequence[Sequence[float]], fg_seq: Sequence[Sequence[float]],
                   notes: str = "") -> Fixture:
    n = len(bg_seq)
    src_fg = [rect0]
    for k in range(n):
        src_fg.append(src_fg[-1].shifted(fg_seq[k][0], fg_seq[k][1]))
    bg_arr = [np.asarray(b, np.float64) for b in bg_seq]
    fg_arr = [np.asarray(f, np.float64) for f in fg_seq]
    pairs = [layer_pair(src_fg[k], src_fg[k + 1], bg_arr[k], fg_arr[k], W, H, f"{name}{k}")
             for k in range(n)]
    fields, valids = _fields_from_pairs(pairs, W, H)

    def region(target: int, grid: np.ndarray) -> np.ndarray:
        p = np.asarray(grid, np.float64)
        in_fg = src_fg[target].contains(p)
        inb = in_bounds(p, W, H)
        q_bg = p + bg_arr[target]
        occluded = (~in_fg) & in_bounds(q_bg, W, H) & src_fg[target + 1].contains(q_bg)
        lab = np.where(in_fg, 1, 0).astype(int)
        lab[inb & occluded] = 3
        lab[~inb] = -1
        return lab

    return Fixture(name, desc, W, H, pairs, fields, valids, region=region, notes=notes)


def build_fixtures() -> dict:
    """Deterministic analytic fixtures (no files, no media, no private data)."""
    fx = {}

    # Spatially constant translation fields. same_grid and transported coincide; order-1/2 differ.
    fx["translation_constant"] = _translation_fixture(
        "translation_constant", "Constant translation [3,-2] every pair. All candidates should be exact.",
        32, 24, [[3.0, -2.0]] * 4)

    fx["translation_linear"] = _translation_fixture(
        "translation_linear", "Linearly growing translation [3k,-2k]. Constant-velocity extrapolation is exact.",
        32, 24, [[3.0 * k, -2.0 * k] for k in range(4)])

    fx["translation_quadratic"] = _translation_fixture(
        "translation_quadratic", "Quadratic translation [0.5k^2, 3k]. Constant-acceleration extrapolation is exact.",
        32, 24, [[0.5 * k * k, 3.0 * k] for k in range(4)])

    fx["translation_out"] = _translation_fixture(
        "translation_out", "Translation [24,0] leaves only 25% valid support: coverage falls below the "
        "threshold and falls back to cold.", 32, 24, [[24.0, 0.0]] * 4)

    # Spatially varying motion. A constant rotation is steady in image coordinates (same_grid wins);
    # an increasing rotation is not.
    Wr, Hr = 32, 24
    Rconst = rotation_about_clockwise(Wr, Hr, 8.0)
    pairs_const = [global_pair(Rconst, Wr, Hr, f"rot{k}") for k in range(4)]
    fields_const, valids_const = _fields_from_pairs(pairs_const, Wr, Hr)
    fx["rotation_const"] = Fixture(
        "rotation_const", "Constant 8 deg rotation every pair: the flow field is steady in image coords.",
        Wr, Hr, pairs_const, fields_const, valids_const,
        notes="same_grid should be exact; transport samples the field at the wrong location by design.")

    Rrate = [rotation_about_clockwise(Wr, Hr, 4.0 * (k + 1)) for k in range(4)]
    pairs_rate = [global_pair(Rrate[k], Wr, Hr, f"rotrate{k}") for k in range(4)]
    fields_rate, valids_rate = _fields_from_pairs(pairs_rate, Wr, Hr)
    fx["rotation_rate"] = Fixture(
        "rotation_rate", "Increasing rotation 4/8/12/16 deg: the field is not steady, so transport matters.",
        Wr, Hr, pairs_rate, fields_rate, valids_rate)

    # Well-conditioned non-affine projective global.
    Wp, Hp = 48, 36
    M = rotation_about_clockwise(Wp, Hp, 5.0) @ translation(2.0, -1.0)
    M[2, 0] = 2.0e-3
    M[2, 1] = -1.6e-3
    pairs_p = [global_pair(M, Wp, Hp, f"proj{k}") for k in range(3)]
    fields_p, valids_p = _fields_from_pairs(pairs_p, Wp, Hp)
    fx["projective"] = Fixture(
        "projective", "One repeated non-affine projective homography (rotation+translation+perspective): "
        "the image-coordinate flow is stationary, so same_grid is exact and transport is model-mismatched.",
        Wp, Hp, pairs_p, fields_p, valids_p,
        notes="same_grid is exact because the field is stationary in image coordinates; transported error is "
              "chiefly model mismatch (the correspondence displacement is not carried by the inverse map), "
              "with bilinear discretization only a smaller extra.")

    # Two-layer parallax: foreground [5,0] px/pair, background [1,0] px/pair, explicit occlusion.
    Wl, Hl = 48, 36
    rect0 = Rect(cx=Wl / 2 - .5, cy=Hl / 2 - .5, hx=8.0, hy=7.0)
    n = 4
    fx["parallax_two_layer"] = _layer_fixture(
        "parallax_two_layer", "Foreground [5,0] px/pair over background [1,0] px/pair with occlusions/disocclusions.",
        Wl, Hl, rect0, [[1.0, 0.0]] * n, [[5.0, 0.0]] * n,
        notes="two constant translations + rectangle layers; exact for correspondence/validity, not metric depth.")

    # The same clip traversed backwards: reverse temporal order and invert every pair. Included so the CLI
    # and JSON expose both directions; transport exactness here is again conditional on the layered motion.
    fx["parallax_two_layer_backward"] = reversed_fixture(fx["parallax_two_layer"])

    return fx


def reversed_fixture(fx: Fixture) -> Fixture:
    """The backward-direction fixture for ``fx``.

    The backward pass traverses the clip in reverse temporal order: its pair ``j`` maps frame ``n-j`` to
    ``n-j-1``, i.e. it is the inverse of the forward pair ``n-1-j``. Both the per-pair map (invert) and the
    temporal order (reverse) are flipped; inverting each pair without reversing the order would mislabel
    which field precedes which, so the extrapolation history would be reversed.
    """
    pairs = [p.reversed() for p in reversed(fx.pairs)]
    fields, valids = _fields_from_pairs(pairs, fx.W, fx.H)
    return Fixture(fx.name + "_backward", fx.description + " (backward direction)",
                   fx.W, fx.H, pairs, fields, valids, region=None, notes=fx.notes)


# --------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------

def compare_fixture(fx: Fixture, methods: Sequence[str] = METHODS,
                    targets: Optional[Sequence[int]] = None, min_coverage: float = MIN_COVERAGE) -> dict:
    """Run every method on every target pair of one fixture and return per-target rows."""
    if targets is None:
        targets = list(range(1, len(fx.pairs)))
    grid = source_grid(fx.W, fx.H)
    res = {"name": fx.name, "description": fx.description, "notes": fx.notes,
           "W": fx.W, "H": fx.H, "frames": fx.frames, "methods": {}}
    for method in methods:
        rows = []
        for target in targets:
            pred = predict(fx.pairs, fx.fields, fx.valids, target, method, fx.W, fx.H, min_coverage, grid)
            truth = fx.fields[target].reshape(-1, 2)
            truth_valid = fx.valids[target].reshape(-1)
            region = fx.region(target, grid) if fx.region is not None else None
            ev = evaluate(pred.field, pred.valid, truth, truth_valid, region,
                          raw_field=pred.raw_field, raw_valid=pred.raw_valid)
            ev.update({"target": int(target), "fallback": bool(pred.fallback), "reason": pred.reason,
                       "raw_coverage": float(pred.coverage)})
            rows.append(ev)
        res["methods"][method] = rows
    return res


def _method_summary(rows: Sequence[dict]) -> dict:
    """Mean coverage/support/error over target rows that had enough history to run.

    Rows that fell back only because the method needs more prior pairs
    (``reason == "insufficient_history"``) are excluded from the means and reported separately; otherwise a
    legitimate "not enough history yet" cold fallback would be averaged into the method's error.
    """
    usable = [r for r in rows if r["reason"] != "insufficient_history"]
    cov = [r["coverage"] for r in usable]
    sup = [r["support"] for r in usable]
    cerr = [r["candidate_error"]["mean_px"] for r in usable if r["candidate_error"]["mean_px"] is not None]
    eerr = [r["effective_error"]["mean_px"] for r in usable if r["effective_error"]["mean_px"] is not None]
    return {
        "targets": len(usable),
        "skipped_no_history": int(len(rows) - len(usable)),
        "coverage_mean": float(np.mean(cov)) if cov else None,
        "support_mean": float(np.mean(sup)) if sup else None,
        "candidate_error_mean_px": float(np.mean(cerr)) if cerr else None,
        "effective_error_mean_px": float(np.mean(eerr)) if eerr else None,
        "fallbacks": int(sum(1 for r in usable if r["fallback"])),
    }


def run_experiment(fixtures: Optional[dict] = None, methods: Sequence[str] = METHODS,
                   min_coverage: float = MIN_COVERAGE) -> dict:
    """Run every fixture, return a JSON-safe results dict and a per-fixture/method summary."""
    fixtures = build_fixtures() if fixtures is None else fixtures
    results = {
        "schema": SCHEMA,
        "methods": list(methods),
        "min_coverage": float(min_coverage),
        "conventions": CONVENTIONS,
        "scope": ("analytic initialization experiment only: exact synthetic fields and exact pair maps; "
                  "no RAFT quality, no real footage, no production timing, no production wiring"),
        "fixtures": [compare_fixture(fx, methods, min_coverage=min_coverage) for fx in fixtures.values()],
    }
    results["summary"] = {
        fx["name"]: {m: _method_summary(fx["methods"][m]) for m in methods} for fx in results["fixtures"]
    }
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="E04 coordinate-correct flow-initialization experiment (NumPy only).")
    ap.add_argument("--out-dir", default=None, help="write results.json here")
    ap.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    args = ap.parse_args(argv)

    res = run_experiment(min_coverage=args.min_coverage)
    if args.out_dir is not None:
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(
            json.dumps(json_safe(res), indent=1, sort_keys=True, allow_nan=False), encoding="utf-8")
        res["out_dir"] = str(out)

    print(json.dumps(json_safe({"schema": res["schema"], "min_coverage": res["min_coverage"],
                                "scope": res["scope"]}), indent=1, sort_keys=True, allow_nan=False))
    print("\nPer-fixture method means (targets with insufficient history are excluded; 'n/a' = no usable targets)")
    print("  candidate_epe = pre-fallback error of the candidate on shared support (valid AND truth-valid)")
    print("  effective_epe = post-fallback error actually realized (cold zeros when a fallback fired)")
    print("  coverage      = raw candidate valid fraction; support = shared candidate/truth valid fraction")
    for name, methods in res["summary"].items():
        print(f"\n{name}")
        for method in METHODS:
            s = methods.get(method)
            if s is None:
                continue
            cand = "n/a" if s["candidate_error_mean_px"] is None else f"{s['candidate_error_mean_px']:.4g}"
            eff = "n/a" if s["effective_error_mean_px"] is None else f"{s['effective_error_mean_px']:.4g}"
            cov = "n/a" if s["coverage_mean"] is None else f"{s['coverage_mean']:.3f}"
            sup = "n/a" if s["support_mean"] is None else f"{s['support_mean']:.3f}"
            print(f"  {method:22s} candidate_epe={cand:>8s} px  effective_epe={eff:>8s} px  "
                  f"coverage={cov:>6s}  support={sup:>6s}  fallbacks={s['fallbacks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
