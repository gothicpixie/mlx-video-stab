"""Isolated, pure-NumPy E08 gate: can an *explicit depth-rank layer policy* recover a requested coherent
motion layer on synthetic sequences with **known ground truth**, and what does each policy cost downstream?

This is a **falsifiable offline synthetic diagnostic**, not a claim that monocular depth understands semantic
foreground/background. It asks two narrow questions on a deterministic synthetic scene whose per-layer
frame transforms, per-pixel layer labels and validity are all known exactly:

1. **Layer recovery.** Given a source-depth field, a per-pixel forward-flow field and a validity mask, does
   selecting cells by an *explicit extreme-depth rank* (one policy locks one rank, a second policy preserves
   another rank) recover the requested layer's known flow better, on held-out cells, than a single pooled
   affine fit over all valid cells?
2. **Downstream tradeoff.** Applied to a well-defined coordinate model of the known rendered sequence, what
   are the **separate** background-lock and foreground-preserve tradeoffs in (a) residual jitter against the
   known intended target path, (b) valid warp/crop coverage, and (c) layer-specific displacement error?

Rank orientation, never physical near/far
-----------------------------------------
The scene defines layers by an **arbitrary raw depth value**. :data:`RANK_ORIENTATIONS` makes the mapping
from raw value to rank *explicit and caller-chosen*: ``"low_raw_low_rank"`` (the smallest raw value is rank
0) or ``"low_raw_high_rank"`` (the largest raw value is rank 0). The module **never** infers that a rank is
physical near or physical far, or that any rank is semantic foreground/background; a "lock rank 0" policy is
literally "select the cells whose depth rank is 0 under the configured orientation". A rank requested outside
the scene's layer count is reported as ``unavailable`` support (never silently clamped).

Synthetic scene (ground truth)
------------------------------
``build_sequence`` lays out ``n_layers`` layers as a deterministic stack: layer 0 is a full-frame plane, and
each higher layer is a rectangle (a "moving/deforming distractor") drawn on top in the scene's own
construction order. Per frame ``t`` every layer has a known 2x3 affine ``M[l][t]`` (reference -> frame-t
coordinates). The shared camera layers carry the intended smooth path composed with a deterministic
per-frame jitter; the foreground overlay carries the camera path composed with an **independent deforming
local motion** (an anisotropic scale that changes over time). Pixel visibility is by the scene's own z-order,
so the foreground rectangle **occludes** the layers beneath it. The observed field is

    fwd[p] = M[visible_layer(p)][t] (p) - p          (a reference-coordinate displacement)

with a caller-configurable validity mask (border band and a rectangular occlusion/"invalid support" hole).
The source-depth field is the raw depth value of the visible layer at ``p``; ``depth_mode`` may make it
**uninformative** (every layer shares one raw value, collapsing to one rank) or **shuffled** (a deterministic
permutation of the raw values, so ranks carry no layer information), for the degenerate fallback checks.

Estimation and held-out evaluation
----------------------------------
A deterministic checkerboard split (even ``x + y`` train, odd validate) is used, exactly as in
:mod:`bench.depth_motion_audit`. The **pooled baseline** fits one affine per component on all valid train
cells. Each **policy** fits the same affine model on only its selected layer's valid train cells, fitted
**from source-depth/flow/validity alone** (the true per-pixel layer id is never used for fitting). Both are
scored against the **known target-layer flow** on held-out cells of the selected layer, and the pooled
baseline is scored on the *same* held-out block, so the comparison is on identical cells. Support floors
(:data:`MIN_FIT_SAMPLES`, :data:`MIN_EVAL_SAMPLES`) make an accidental single-cell "win" impossible; a policy
whose selection collapses (e.g. under uninformative depth) reports a machine-readable reason and declines to
claim an improvement.

The two policies are **independent blocks** with their own selected rank, their own fitted model and their
own held-out statistics; nothing is pooled between them, and no threshold is selected or swept.

Downstream coordinate model
---------------------------
Stabilization by an estimated per-frame path ``E[t]`` is modelled as composing ``E[t]⁻¹`` with the known
layer transform ``M[l][t]``. The residual path ``R_l[t] = E[t]⁻¹ ∘ M[l][t]`` is evaluated on deterministic
per-layer reference sample points, against a per-layer **intended residual path** ``G_l[t]``:

* residual **jitter** = RMS of the discrete second difference of ``R_l[t]`` over ``t`` (a motion-smoothness
  proxy; the second difference of the residual path point displacement), reported per layer;
* **valid warp / crop coverage** = fraction of the output grid whose ``E[t]⁻¹`` maps inside the frame
  bounds ``[-0.5, W-0.5] x [-0.5, H-0.5]`` (matching ``warp.valid_mask``'s pixel-edge convention);
* **layer-specific displacement error** = mean over sample points and frames of
  ``|R_l[t](q) - G_l[t](q)|``.

``G_l[t]`` is the intended residual: identity for the layer a policy locks (that layer should become static)
and the layer's intended in-frame motion for a layer a policy preserves. These are reported separately for
each policy; there is no combined winner score.

Determinism, JSON and scope
---------------------------
Everything is float64 NumPy with fixed seeds/permutations; :func:`run_audit` is a pure function of the spec.
The report is path-free, finite-safe and written atomically through :mod:`bench.manifest`. No media is
decoded, no RAFT/MLX/production code is imported, no weights are read, the heavy lock is untouched, and no
production default is changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

import numpy as np

from . import manifest as MF

SCHEMA = "mlxstab.depth_rank_layer_audit/1"

RANK_ORIENTATIONS = ("low_raw_low_rank", "low_raw_high_rank")

DEPTH_MODES = ("ranked", "uninformative", "shuffled")

# Support floors. A 2-D affine per component is identifiable with 3 non-collinear cells, but a contract of
# 6 valid train cells and 3 held-out cells is required before a fit/report is trusted, so a single lucky
# held-out cell can never read as an improvement. These are fixed constants, not swept thresholds.
MIN_FIT_SAMPLES = 6
MIN_EVAL_SAMPLES = 3

# A policy's held-out median residual must beat the pooled baseline by at least this margin (in synthetic
# pixels) to be recorded as an improvement; equality or noise below this is "no material gain". Fixed.
MATERIAL_GAIN_PX = 1e-6

LIMITATIONS = (
    "all results are SYNTHETIC: a deterministic coordinate model with known ground-truth layer transforms; "
    "they say nothing about real footage, RAFT accuracy, or metric depth",
    "the scene's raw depth values are arbitrary labels; rank orientation is explicit and the module never "
    "asserts physical near/far or semantic foreground/background",
    "the foreground distractor is a rectangular overlay with a known affine local motion, so occlusion and "
    "deformation are synthetic constructions, not realistic object behaviour",
    "held-out fit error uses a checkerboard split whose train/validation neighbours are spatially correlated, "
    "so it is a structural diagnostic, not an independent-sample generalisation guarantee",
    "downstream jitter/coverage/displacement are coordinate-model metrics, not rendered-pixel or perceptual "
    "quality measurements; no frame is rendered and no production stabilizer is run",
    "no threshold is selected or swept, no policy is declared a winner, and no production wiring or default "
    "change is implied",
)


# --------------------------------------------------------------------------------------
# small affine helpers (3x3 with bottom row [0,0,1]); affine-only, applied to pixel points
# --------------------------------------------------------------------------------------

def _affine_identity() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def _affine_from_2x3(a: Sequence[Sequence[float]]) -> np.ndarray:
    """Build a 3x3 affine from a 2x3 ``[[a,b,tx],[c,d,ty]]`` block."""
    a = np.asarray(a, np.float64)
    if a.shape != (2, 3):
        raise ValueError("affine block must be 2x3")
    m = np.eye(3, dtype=np.float64)
    m[:2, :] = a
    return m


def apply_affine(m: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply a 3x3 affine ``m`` to points ``p [N,2]`` -> ``[N,2]`` (pure translation/linear, no perspective)."""
    m = np.asarray(m, np.float64)
    p = np.asarray(p, np.float64)
    return p @ m[:2, :2].T + m[:2, 2]


def invert_affine(m: np.ndarray) -> np.ndarray:
    """Inverse of a 3x3 affine; raises ``LinAlgError`` for a singular linear block."""
    m = np.asarray(m, np.float64)
    lin = m[:2, :2]
    return np.linalg.inv(m)


def compose_affine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a ∘ b`` (apply ``b`` first) for two 3x3 affines."""
    return np.asarray(a, np.float64) @ np.asarray(b, np.float64)


# --------------------------------------------------------------------------------------
# synthetic scene
# --------------------------------------------------------------------------------------

def _checkerboard(coords: np.ndarray) -> np.ndarray:
    """Deterministic split: even ``x + y`` -> train (True), odd -> validate (False)."""
    return ((coords[:, 0].astype(np.int64) + coords[:, 1].astype(np.int64)) % 2) == 0


def _rect_mask(W: int, H: int, cx: float, cy: float, hw: float, hh: float) -> np.ndarray:
    """Boolean ``[H,W]`` mask of a rectangle centred at ``(cx, cy)`` with half-extents ``(hw, hh)``."""
    xs = np.arange(W)[None, :]
    ys = np.arange(H)[:, None]
    return (np.abs(xs - cx) <= hw) & (np.abs(ys - cy) <= hh)


def _camera_transforms(T: int, W: int, H: int, *, jitter_px: float, drift_px: float, rot_deg: float,
                       seed: int) -> tuple:
    """Known camera transforms ``C[t]`` (reference -> frame t) and the intended smooth path ``P[t]``.

    ``P[t]`` is a smooth pan + slow zoom (the *intended* target path). ``C[t] = P[t] ∘ J[t]`` adds a
    deterministic zero-mean per-frame jitter ``J[t]`` (small translation plus rotation). Both are returned as
    ``[T,3,3]``. The centred jitter is removed from ``P`` by construction (``P`` is the smooth trend).
    """
    t = np.arange(T, dtype=np.float64)
    tau = np.linspace(0.0, 1.0, max(T, 2))[:T]
    # Smooth intended trend: gentle horizontal pan and a slow scale about the frame centre.
    pan_x = drift_px * tau
    scale = 1.0 + 0.02 * tau
    cx, cy = (W - 1.0) / 2.0, (H - 1.0) / 2.0
    # Deterministic jitter: a fixed low-discrepancy-looking pattern (sinusoids with incommensurate rates).
    jx = jitter_px * np.sin(2.399 * t + 0.7 * seed)
    jy = jitter_px * np.cos(3.117 * t + 1.3 * seed)
    jr = np.deg2rad(rot_deg) * np.sin(1.771 * t + 0.3 * seed)

    P = np.zeros((T, 3, 3), np.float64)
    C = np.zeros((T, 3, 3), np.float64)
    for i in range(T):
        # Intended path: scale about centre, then pan.
        s, a = scale[i], pan_x[i]
        p = np.array([[s, 0.0, a + (1.0 - s) * cx],
                      [0.0, s, (1.0 - s) * cy],
                      [0.0, 0.0, 1.0]], np.float64)
        P[i] = p
        # Jitter about the frame centre: rotation + small translation.
        ca, sa = np.cos(jr[i]), np.sin(jr[i])
        j = np.array([[ca, -sa, cx - ca * cx + sa * cy + jx[i]],
                      [sa, ca, cy - sa * cx - ca * cy + jy[i]],
                      [0.0, 0.0, 1.0]], np.float64)
        C[i] = p @ j
    return C, P


def _deforming_local(T: int, W: int, H: int, *, amp: float) -> np.ndarray:
    """Foreground overlay's independent local motion ``L[t]`` (reference -> local coords), an anisotropic,
    time-varying scale about the frame centre plus a small slide; the "moving/deforming distractor".

    The deformation is a known non-rigid *affine* scale (so it stays inside the affine model family used for
    both the ground truth and the estimator; non-affine deformation is deliberately out of scope)."""
    tau = np.linspace(0.0, 1.0, max(T, 2))[:T]
    cx, cy = (W - 1.0) / 2.0, (H - 1.0) / 2.0
    L = np.zeros((T, 3, 3), np.float64)
    for i in range(T):
        sx = 1.0 + amp * tau[i]
        sy = 1.0 - 0.5 * amp * tau[i]
        dx = amp * 12.0 * np.sin(0.9 * i)
        L[i] = np.array([[sx, 0.0, dx + (1.0 - sx) * cx],
                         [0.0, sy, (1.0 - sy) * cy],
                         [0.0, 0.0, 1.0]], np.float64)
    return L


class SyntheticSpec:
    """Immutable description of the synthetic sequence. Defaults are the documented, fixed test scene."""

    def __init__(self, *, T: int = 10, W: int = 48, H: int = 36, n_layers: int = 3,
                 depths: Sequence[float] = (0.0, 0.5, 1.0), rank_orientation: str = "low_raw_low_rank",
                 depth_mode: str = "ranked", lock_rank: int = 0, preserve_rank: int = 2,
                 jitter_px: float = 3.0, drift_px: float = 10.0, rot_deg: float = 0.4,
                 fg_amp: float = 0.25, border: int = 3, occlusion_hole: bool = True,
                 seed: int = 0):
        if T < 3:
            raise ValueError("T must be >= 3 so a residual second difference is defined")
        if W < 8 or H < 8:
            raise ValueError("W and H must be >= 8")
        if n_layers < 2:
            raise ValueError("n_layers must be >= 2")
        depths = tuple(float(d) for d in depths)
        if len(depths) != n_layers:
            raise ValueError("len(depths) must equal n_layers")
        if rank_orientation not in RANK_ORIENTATIONS:
            raise ValueError(f"rank_orientation must be one of {RANK_ORIENTATIONS}")
        if depth_mode not in DEPTH_MODES:
            raise ValueError(f"depth_mode must be one of {DEPTH_MODES}")
        if border < 0:
            raise ValueError("border must be >= 0")
        self.T, self.W, self.H = int(T), int(W), int(H)
        self.n_layers = int(n_layers)
        self.depths = depths
        self.rank_orientation = rank_orientation
        self.depth_mode = depth_mode
        self.lock_rank = int(lock_rank)
        self.preserve_rank = int(preserve_rank)
        self.jitter_px = float(jitter_px)
        self.drift_px = float(drift_px)
        self.rot_deg = float(rot_deg)
        self.fg_amp = float(fg_amp)
        self.border = int(border)
        self.occlusion_hole = bool(occlusion_hole)
        self.seed = int(seed)

    def as_dict(self) -> dict:
        """JSON-safe, path-free description of the spec (for the report's ``spec`` block)."""
        return {
            "T": self.T, "W": self.W, "H": self.H, "n_layers": self.n_layers,
            "depths": list(self.depths), "rank_orientation": self.rank_orientation,
            "depth_mode": self.depth_mode, "lock_rank": self.lock_rank,
            "preserve_rank": self.preserve_rank, "jitter_px": self.jitter_px,
            "drift_px": self.drift_px, "rot_deg": self.rot_deg, "fg_amp": self.fg_amp,
            "border": self.border, "occlusion_hole": self.occlusion_hole, "seed": self.seed,
        }


def rank_of_raw(depths: Sequence[float], orientation: str) -> np.ndarray:
    """Rank of each layer's **raw** depth value under the explicit ``orientation``.

    ``"low_raw_low_rank"``: smallest raw value -> rank 0. ``"low_raw_high_rank"``: largest raw value -> rank
    0. Ties (equal raw values, as in uninformative mode) share the lower rank so a collapsed field yields one
    effective rank rather than an arbitrary order. No physical near/far meaning is attached.
    """
    if orientation not in RANK_ORIENTATIONS:
        raise ValueError(f"rank_orientation must be one of {RANK_ORIENTATIONS}")
    vals = np.asarray(depths, np.float64)
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(len(vals), np.int64)
    if orientation == "low_raw_low_rank":
        ranks[order] = np.arange(len(vals))
    else:
        ranks[order] = np.arange(len(vals))[::-1]
    # Collapse ties to the minimum rank of the tied group so equal raw values share a rank.
    uniq = np.unique(vals)
    for u in uniq:
        grp = vals == u
        if grp.sum() > 1:
            ranks[grp] = ranks[grp].min()
    return ranks


class Sequence:
    """Known-ground-truth synthetic sequence: transforms, per-pixel layer labels, depth, flow and validity."""

    def __init__(self, spec: SyntheticSpec):
        self.spec = spec
        W, H, T, L = spec.W, spec.H, spec.T, spec.n_layers
        self.W, self.H, self.T, self.n_layers = W, H, T, L
        self.coords = np.stack(np.meshgrid(np.arange(W), np.arange(H), indexing="xy"), -1).reshape(-1, 2)
        self.train = _checkerboard(self.coords)

        # Scene construction order: layer 0 is the full-frame plane, higher layers are rectangles drawn on
        # top (occluding). This is a synthetic z-order, not a semantic foreground claim.
        layer_masks = np.zeros((L, H, W), bool)
        layer_masks[0] = True
        for l in range(1, L):
            # Deterministic rectangle placement that varies with layer index only (frame-independent layout).
            frac = (l + 1) / (L + 1.0)
            cx = 0.2 * W + 0.55 * W * frac
            cy = 0.25 * H + 0.5 * H * (1.0 - frac)
            hw = max(2.0, 0.16 * W)
            hh = max(2.0, 0.16 * H)
            layer_masks[l] = _rect_mask(W, H, cx, cy, hw, hh)

        # Visible layer id per pixel: highest scene-order layer whose mask covers the pixel.
        visible = np.zeros((H, W), np.int64)
        for l in range(L):
            visible[layer_masks[l]] = l
        self.layer_masks = layer_masks
        self.visible_layer = visible

        # Known per-layer frame transforms: shared camera for layer 0 (and any non-overlay layer), and the
        # camera composed with an independent deforming local motion for the single overlay layer (last).
        C, P = _camera_transforms(T, W, H, jitter_px=spec.jitter_px, drift_px=spec.drift_px,
                                  rot_deg=spec.rot_deg, seed=spec.seed)
        self.camera = C
        self.intended_path = P
        self.layer_transform = np.zeros((L, T, 3, 3), np.float64)
        self.intended_layer = np.zeros((L, T, 3, 3), np.float64)
        self.layer_transform[0] = C
        self.intended_layer[0] = P
        if L >= 2:
            Lm = _deforming_local(T, W, H, amp=spec.fg_amp)
            for l in range(1, L):
                # Each overlay shares the camera but has its own (here identical) local deformation; keep the
                # distractor independent from the camera so background-lock and foreground-preserve differ.
                local = Lm if l == L - 1 else _deforming_local(T, W, H, amp=spec.fg_amp * 0.5 * l)
                for t in range(T):
                    self.layer_transform[l, t] = compose_affine(C[t], local[t])
                self.intended_layer[l, t] = P[t]  # intended residue for an overlay is the camera trend

        # Observed flow at reference coords p for each frame: displacement of p's visible layer.
        # fwd[t,p] = M[visible(p)][t](p) - p  (reference-coordinate displacement).
        flat_coords = self.coords
        vis_flat = visible.reshape(-1)
        self.fwd_flow = np.zeros((T, H * W, 2), np.float64)
        for t in range(T):
            for l in range(L):
                sel = vis_flat == l
                if sel.any():
                    self.fwd_flow[t, sel] = apply_affine(self.layer_transform[l, t], flat_coords[sel]) - flat_coords[sel]

        # Source depth per pixel = raw depth value of the visible layer (collapsed/shuffled per depth_mode).
        raw = np.asarray(spec.depths, np.float64)
        if spec.depth_mode == "uninformative":
            raw = np.full(L, raw[0] if L else 0.0, np.float64)
        self.source_depth_layer = raw
        depth_flat = raw[vis_flat]
        if spec.depth_mode == "shuffled":
            # Deterministic permutation of per-pixel depths so ranks carry no layer information.
            rng = np.random.RandomState(spec.seed + 7919)
            perm = rng.permutation(H * W)
            depth_flat = depth_flat[perm]
        self.source_depth = depth_flat.reshape(H, W)

        # Per-pixel rank under the explicit orientation.
        self.layer_rank = rank_of_raw(raw, spec.rank_orientation)
        self.rank_flat = self.layer_rank[vis_flat]

        # Validity: border band + optional rectangular invalid-support hole (fractional positions dropped).
        valid = np.ones((H, W), bool)
        b = spec.border
        if b > 0:
            valid[:b, :] = False
            valid[-b:, :] = False
            valid[:, :b] = False
            valid[:, -b:] = False
        if spec.occlusion_hole and W >= 12 and H >= 12:
            hole = _rect_mask(W, H, 0.78 * W, 0.72 * H, 0.06 * W, 0.06 * H)
            valid &= ~hole
        self.valid = valid

        # Known target layer flow (reference-coordinate displacement) on the full grid for every layer.
        self.layer_flow = np.zeros((L, T, H * W, 2), np.float64)
        for l in range(L):
            for t in range(T):
                self.layer_flow[l, t] = apply_affine(self.layer_transform[l, t], flat_coords) - flat_coords

    # -- sampling helpers -----------------------------------------------------------------

    def layer_points(self, layer: int, n: int = 25) -> np.ndarray:
        """Deterministic reference-frame sample points inside a layer's own mask (for displacement checks)."""
        mask = self.layer_masks[layer].reshape(-1)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return np.zeros((0, 2), np.float64)
        take = np.linspace(0, idx.size - 1, min(n, idx.size)).astype(np.int64)
        return self.coords[idx[take]]


def build_sequence(spec: SyntheticSpec) -> Sequence:
    """Construct the deterministic known-ground-truth sequence for ``spec``."""
    return Sequence(spec)


# --------------------------------------------------------------------------------------
# estimation core
# --------------------------------------------------------------------------------------

def _design(coords: np.ndarray, mask: np.ndarray, W: int, H: int) -> Optional[np.ndarray]:
    """Design matrix ``[x/W, y/H, 1]`` for cells selected by ``mask``; ``None`` if empty."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    pts = coords[idx]
    x = pts[:, 0] / float(max(W, 1))
    y = pts[:, 1] / float(max(H, 1))
    return np.stack([x, y, np.ones_like(x)], axis=1)


def _affine_fit(design: np.ndarray, target: np.ndarray) -> Optional[np.ndarray]:
    """OLS for one displacement component: ``coeff [3]`` or ``None`` when unidentifiable."""
    if design.shape[0] < MIN_FIT_SAMPLES:
        return None
    if np.linalg.matrix_rank(design) < 3:
        return None
    coeff, *_ = np.linalg.lstsq(design, target, rcond=None)
    return np.asarray(coeff, np.float64)


def _fit_displacement(design: np.ndarray, disp: np.ndarray):
    """Fit ``u``/``v`` affine coefficients from a displacement field; returns ``(cu, cv)`` or ``(None, None)``."""
    cu = _affine_fit(design, disp[:, 0])
    cv = _affine_fit(design, disp[:, 1])
    if cu is None or cv is None:
        return None, None
    return cu, cv


def _coeff_to_affine(cu: np.ndarray, cv: np.ndarray, W: int, H: int) -> np.ndarray:
    """Affine transform ``E[t]`` (reference -> frame t) from displacement coefficients ``disp = A(x/W,y/H,1)``.

    ``E = I + D`` where ``D`` is the affine displacement with linear block ``[[cu0/W, cu1/H],[cv0/W, cv1/H]]``
    and translation ``[cu2, cv2]``.
    """
    E = np.eye(3, dtype=np.float64)
    E[0, 0] = cu[0] / float(W)
    E[0, 1] = cu[1] / float(H)
    E[0, 2] = cu[2]
    E[1, 0] = cv[0] / float(W)
    E[1, 1] = cv[1] / float(H)
    E[1, 2] = cv[2]
    return E


def _resid_stats(resid: np.ndarray) -> dict:
    """Finite Euclidean-residual median/p95 plus count; ``None`` metrics when empty."""
    r = np.asarray(resid, np.float64).ravel()
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {"n": 0, "median": None, "p95": None}
    return {"n": int(r.size), "median": float(np.median(r)), "p95": float(np.percentile(r, 95))}


def _select_mask(seq: Sequence, valid_flat: np.ndarray, *, rank: int) -> np.ndarray:
    """Cells whose **visible-layer depth rank** equals ``rank`` (from source depth only, not ground truth)."""
    return valid_flat & (seq.rank_flat == int(rank))


def _fit_policy(seq: Sequence, t: int, select_rank: Optional[int]):
    """Fit the affine displacement model for one frame.

    ``select_rank=None`` is the pooled baseline (all valid train cells). Otherwise the fit uses only valid
    train cells whose visible-layer depth rank equals ``select_rank``. Returns a dict with the fitted
    coefficients, the affine estimate, train/held-out masks and a machine-readable ``reason`` when the fit or
    support is unavailable.
    """
    H, W = seq.H, seq.W
    valid_flat = seq.valid.reshape(-1)
    train_all = valid_flat & seq.train
    if select_rank is None:
        train_mask = train_all
        scope = "pooled"
    else:
        rank_mask = _select_mask(seq, valid_flat, rank=select_rank)
        train_mask = rank_mask & seq.train
        scope = f"rank_{int(select_rank)}"
    # Held-out block is the same selected set's odd-parity cells. The pooled baseline's block is all valid
    # odd cells; the caller scores every estimate against a specific known target layer by intersecting this
    # block with that layer's visible cells (see :func:`_heldout_error`), so the comparison is apples-to-apples.
    valid_odd = valid_flat & ~seq.train
    if select_rank is None:
        eval_mask = valid_odd
    else:
        eval_mask = _select_mask(seq, valid_odd, rank=select_rank)

    design = _design(seq.coords, train_mask, W, H)
    if design is None:
        return {"scope": scope, "fit_ok": False, "reason": "insufficient_train",
                "cu": None, "cv": None, "E": None, "train_count": int(train_mask.sum()),
                "eval_count": int(eval_mask.sum())}
    cu, cv = _fit_displacement(design, seq.fwd_flow[t][train_mask])
    if cu is None:
        return {"scope": scope, "fit_ok": False, "reason": "unidentifiable_fit",
                "cu": None, "cv": None, "E": None, "train_count": int(train_mask.sum()),
                "eval_count": int(eval_mask.sum())}
    E = _coeff_to_affine(cu, cv, W, H)
    # Reject a degenerate/degenerate-adjacent estimate (singular linear block or non-finite entries) rather
    # than letting a downstream inversion blow up; the caller substitutes identity and records the reason.
    if not np.isfinite(E).all() or np.linalg.matrix_rank(E[:2, :2]) < 2 or \
            abs(np.linalg.det(E[:2, :2])) < 1e-9:
        return {"scope": scope, "fit_ok": False, "reason": "degenerate_transform",
                "cu": None, "cv": None, "E": None, "train_count": int(train_mask.sum()),
                "eval_count": int(eval_mask.sum())}
    return {"scope": scope, "fit_ok": True, "reason": None, "cu": cu, "cv": cv, "E": E,
            "train_count": int(train_mask.sum()), "eval_count": int(eval_mask.sum())}


def _heldout_error(seq: Sequence, t: int, E: np.ndarray, eval_mask: np.ndarray, target_layer: int) -> dict:
    """Held-out error of an estimated transform against the **known** target-layer flow at ``eval_mask``."""
    idx = np.flatnonzero(eval_mask)
    if idx.size < MIN_EVAL_SAMPLES:
        return {"n": int(idx.size), "median": None, "p95": None, "reason": "insufficient_validation"}
    pts = seq.coords[idx]
    est = apply_affine(E, pts) - pts
    true = seq.layer_flow[target_layer, t][idx]
    stat = _resid_stats(np.linalg.norm(est - true, axis=1))
    stat["reason"] = None
    return stat


# --------------------------------------------------------------------------------------
# downstream coordinate model
# --------------------------------------------------------------------------------------

def _coverage(E: np.ndarray, W: int, H: int) -> dict:
    """Fraction of the output grid that ``E⁻¹`` maps inside ``[-0.5, W-0.5] x [-0.5, H-0.5]``."""
    pts = np.stack(np.meshgrid(np.arange(W), np.arange(H), indexing="xy"), -1).reshape(-1, 2).astype(np.float64)
    mapped = apply_affine(np.linalg.inv(E), pts)
    x, y = mapped[:, 0], mapped[:, 1]
    inside = (x >= -0.5) & (x <= W - 0.5) & (y >= -0.5) & (y <= H - 0.5)
    return {"n": int(pts.shape[0]), "in_bounds": int(inside.sum()),
            "frac": float(inside.mean()) if pts.shape[0] else None}


def _residual_jitter(residual_path: np.ndarray) -> float:
    """RMS of the discrete second difference of a ``[T, K, 2]`` residual-path point displacement over ``t``.

    ``residual_path`` holds, for each frame, the residual point set; the metric is a translation-invariant
    motion-smoothness proxy (frame-to-frame acceleration), matching the plan's "residual jitter vs the known
    intended target path" language. ``None`` when ``T < 3``.
    """
    rp = np.asarray(residual_path, np.float64)
    if rp.shape[0] < 3:
        return None
    acc = rp[2:] - 2.0 * rp[1:-1] + rp[:-2]
    per_frame = np.linalg.norm(acc, axis=-1).mean(axis=1)
    return float(np.sqrt(np.mean(per_frame ** 2)))


def _layer_downstream(seq: Sequence, E_by_frame: np.ndarray, layer: int,
                      target_layer: int, *, n_points: int = 25) -> dict:
    """Downstream metrics for one layer under per-frame estimated transforms ``E_by_frame [T,3,3]``.

    The residual path is ``R_l[t] = E[t]^-1 . M[l][t]`` evaluated at ``layer``'s reference sample points, and
    the intended residual ``G_l[t]`` (identity for a locked layer, the layer's intended in-frame motion
    otherwise) is subtracted for the displacement error. Reported metrics are the residual jitter (RMS second
    difference of ``R_l``), the mean coverage, and the layer displacement error vs ``G_l``.
    """
    pts = seq.layer_points(layer, n_points)
    if pts.shape[0] == 0:
        return {"layer": int(layer), "n_points": 0, "residual_jitter_px": None,
                "displacement_error_px": None, "displacement_error_p95_px": None,
                "coverage_frac_min": None, "coverage_frac_mean": None, "reason": "empty_layer_mask"}
    T = seq.T
    residual = np.zeros((T, pts.shape[0], 2), np.float64)
    intended_resid = np.zeros((T, pts.shape[0], 2), np.float64)
    covs = []
    for t in range(T):
        Einv = np.linalg.inv(E_by_frame[t])
        moved = apply_affine(compose_affine(Einv, seq.layer_transform[layer, t]), pts)
        residual[t] = moved - pts
        # Intended residual: identity (the target layer should become static) for the locked layer; the
        # intended in-frame motion relative to the lock target otherwise.
        if layer == target_layer:
            intended_resid[t] = 0.0
        else:
            intended_resid[t] = apply_affine(seq.intended_layer[layer, t], pts) - pts
        covs.append(_coverage(E_by_frame[t], seq.W, seq.H)["frac"])
    err = np.linalg.norm(residual - intended_resid, axis=-1).ravel()
    err = err[np.isfinite(err)]
    cvalid = [c for c in covs if c is not None]
    return {
        "layer": int(layer), "n_points": int(pts.shape[0]),
        "residual_jitter_px": _residual_jitter(residual),
        "displacement_error_px": float(np.mean(err)) if err.size else None,
        "displacement_error_p95_px": float(np.percentile(err, 95)) if err.size else None,
        "coverage_frac_min": float(min(cvalid)) if cvalid else None,
        "coverage_frac_mean": float(np.mean(cvalid)) if cvalid else None,
        "reason": None,
    }


def _policy_downstream(seq: Sequence, E_by_frame: np.ndarray, locked_layer: int,
                       target_layer: int) -> dict:
    """Independent downstream block for one policy: per-layer metrics plus aggregate jitter/coverage.

    ``locked_layer`` is the layer the policy locks (its intended residual is identity); ``target_layer`` is
    the layer whose held-out flow fit the policy requested. Jitter is reported for every layer; the aggregate
    is the mean over layers. Nothing is pooled across policies.
    """
    layers = [_layer_downstream(seq, E_by_frame, l, locked_layer) for l in range(seq.n_layers)]
    jitters = [d["residual_jitter_px"] for d in layers if d["residual_jitter_px"] is not None]
    covs = [d["coverage_frac_mean"] for d in layers if d["coverage_frac_mean"] is not None]
    locked = layers[locked_layer] if 0 <= locked_layer < len(layers) else None
    return {
        "locked_layer": int(locked_layer),
        "target_layer": int(target_layer),
        "layers": layers,
        "aggregate_residual_jitter_px": float(np.mean(jitters)) if jitters else None,
        "aggregate_coverage_frac_mean": float(np.mean(covs)) if covs else None,
        "locked_layer_residual_jitter_px": locked["residual_jitter_px"] if locked else None,
        "locked_layer_displacement_error_px": locked["displacement_error_px"] if locked else None,
    }


# --------------------------------------------------------------------------------------
# audit assembly
# --------------------------------------------------------------------------------------

def resolve_rank_layer(seq: Sequence, rank: int) -> dict:
    """Map an explicit depth ``rank`` to a target layer under the spec's orientation, without near/far claims.

    Returns ``{"rank", "layer", "available", "ambiguous", "reason"}``. ``layer`` is the ground-truth layer
    whose rank equals ``rank``; when several layers share that rank (e.g. uninformative depth collapses every
    layer to rank 0), the *first* (lowest scene-order) is reported and ``ambiguous`` is set. When no layer has
    that rank, ``available`` is False and a machine-readable reason is returned (never silently clamped).
    """
    ranks = seq.layer_rank
    matches = np.flatnonzero(ranks == int(rank))
    if matches.size == 0:
        return {"rank": int(rank), "layer": None, "available": False, "ambiguous": False,
                "reason": "rank_not_present"}
    return {"rank": int(rank), "layer": int(matches[0]), "available": True,
            "ambiguous": bool(matches.size > 1), "reason": None}


def _policy_block(seq: Sequence, *, name: str, select_rank: int, locked_layer: int,
                  target_rank: int, pooled: bool) -> dict:
    """One independent policy block: per-frame fits, held-out error vs the known target layer, downstream.

    ``select_rank`` is the explicit depth rank whose cells the fit uses; ``locked_layer`` is the layer the
    downstream coordinate model locks; ``target_rank`` is the rank whose layer the held-out error is measured
    against (equal to ``select_rank`` for lock/preserve selection; separate so they can be checked to agree).
    Nothing is pooled between policy blocks.
    """
    H, W, T = seq.H, seq.W, seq.T
    valid_flat = seq.valid.reshape(-1)
    resolved = resolve_rank_layer(seq, target_rank)
    target_layer = resolved["layer"]

    # Known ground-truth scoring block: odd-parity valid cells of the target layer's visible pixels. Used
    # only for evaluation; never for fitting.
    if target_layer is None:
        eval_block = np.zeros(H * W, bool)
    else:
        eval_block = valid_flat & ~seq.train & (seq.visible_layer.reshape(-1) == target_layer)

    per_frame = []
    E_by_frame = np.zeros((T, 3, 3), np.float64)
    fit_reasons = []
    heldouts = []
    for t in range(T):
        fit = _fit_policy(seq, t, select_rank)
        if fit["fit_ok"]:
            E_by_frame[t] = fit["E"]
        else:
            E_by_frame[t] = _affine_identity()
            fit_reasons.append(fit["reason"])
        # Pooled baseline fit for this frame (all valid train cells), scored on the same block.
        pooled_fit = _fit_policy(seq, t, None)
        pooled_E = pooled_fit["E"] if pooled_fit["fit_ok"] else _affine_identity()

        ph = _heldout_error(seq, t, E_by_frame[t], eval_block, target_layer) if target_layer is not None \
            else {"n": 0, "median": None, "p95": None, "reason": "target_unavailable"}
        gh = _heldout_error(seq, t, pooled_E, eval_block, target_layer) if target_layer is not None \
            else {"n": 0, "median": None, "p95": None, "reason": "target_unavailable"}
        per_frame.append({
            "frame": int(t),
            "policy_scope": fit["scope"], "policy_fit_ok": fit["fit_ok"],
            "policy_reason": fit["reason"],
            "policy_train_count": fit["train_count"], "policy_eval_count": fit["eval_count"],
            "eval_block_count": int(eval_block.sum()),
            "policy_heldout": ph, "pooled_heldout": gh,
        })
        if ph["median"] is not None and gh["median"] is not None:
            heldouts.append((gh["median"], ph["median"]))

    policy_med = [p for _, p in heldouts]
    pooled_med = [g for g, _ in heldouts]
    delta = [g - p for g, p in heldouts]
    downstream = _policy_downstream(seq, E_by_frame, locked_layer, target_layer) if target_layer is not None \
        else {"locked_layer": int(locked_layer), "target_layer": None, "layers": [],
              "aggregate_residual_jitter_px": None, "aggregate_coverage_frac_mean": None,
              "locked_layer_residual_jitter_px": None, "locked_layer_displacement_error_px": None}

    # A "material win" requires the policy's held-out median to be lower than the pooled baseline's on the
    # identical block by at least MATERIAL_GAIN_PX, in the majority of comparable frames. Otherwise it is
    # recorded as "no_material_gain" and never claimed as an improvement.
    wins = int(sum(1 for d in delta if d > MATERIAL_GAIN_PX))
    usable = len(delta)
    if target_layer is None:
        decision = "unavailable"
    elif usable < max(1, T // 2):
        decision = "insufficient_support"
    elif wins > usable / 2.0:
        decision = "material_gain"
    else:
        decision = "no_material_gain"

    return {
        "name": name,
        "select_rank": int(select_rank),
        "target_rank": int(target_rank),
        "target_rank_available": bool(resolved["available"]),
        "target_rank_ambiguous": bool(resolved["ambiguous"]),
        "target_layer": None if target_layer is None else int(target_layer),
        "used_pooled_fit": bool(pooled),
        "policy_reasons": sorted(set(r for r in fit_reasons if r)),
        "usable_frames": int(usable),
        "frames_with_gain": int(wins),
        "heldout_policy_median_px": float(np.median(policy_med)) if policy_med else None,
        "heldout_pooled_median_px": float(np.median(pooled_med)) if pooled_med else None,
        "heldout_median_delta_px": float(np.median(delta)) if delta else None,
        "decision": decision,
        "material_gain_px": MATERIAL_GAIN_PX,
        "per_frame": per_frame,
        "downstream": downstream,
    }


def run_audit(spec: Optional[SyntheticSpec] = None) -> dict:
    """Build the sequence and run the two independent policies plus the pooled baseline. Pure and deterministic.

    Returns a JSON-safe report (no absolute paths): the spec, the sequence geometry, rank resolution for the
    lock/preserve policies, the two independent policy blocks (nothing pooled), and limitations. Does not
    select a threshold, declare a winner, or touch production code.
    """
    if spec is None:
        spec = SyntheticSpec()
    seq = build_sequence(spec)
    lock = _policy_block(seq, name="background_lock", select_rank=spec.lock_rank,
                         locked_layer=resolve_rank_layer(seq, spec.lock_rank)["layer"] or 0,
                         target_rank=spec.lock_rank, pooled=False)
    preserve = _policy_block(seq, name="foreground_preserve", select_rank=spec.preserve_rank,
                             locked_layer=lock["target_layer"] if lock["target_layer"] is not None else 0,
                             target_rank=spec.preserve_rank, pooled=False)

    valid_frac = float(seq.valid.mean())
    return {
        "schema": SCHEMA,
        "spec": spec.as_dict(),
        "geometry": {"W": seq.W, "H": seq.H, "T": seq.T, "n_layers": seq.n_layers,
                     "valid_fraction": valid_frac,
                     "layer_ranks": [int(r) for r in seq.layer_rank],
                     "train_cells": int(seq.train.sum()),
                     "validate_cells": int((~seq.train).sum())},
        "policies": [lock, preserve],
        "policy_note": ("the two policies are independent blocks with their own selected rank, fit and "
                        "held-out statistics; nothing is pooled between them and no winner is selected"),
        "metric_doc": {
            "heldout_median_px": "median over frames of the held-out Euclidean error between the estimated "
                                 "transform's displacement and the known target-layer flow, on the target "
                                 "layer's odd-parity valid cells",
            "heldout_median_delta_px": "pooled minus policy held-out median (positive = policy lower/better)",
            "residual_jitter_px": "RMS of the discrete second difference of the residual path "
                                  "R_l[t] = E[t]^-1 . M[l][t] over t, a coordinate-model motion-smoothness "
                                  "proxy (not rendered pixels)",
            "coverage_frac": "fraction of the output grid whose E[t]^-1 maps inside [-0.5,W-0.5]x[-0.5,H-0.5]",
            "displacement_error_px": "mean |R_l[t](q) - G_l[t](q)| over the layer's reference sample points",
        },
        "limitations": list(LIMITATIONS),
    }


# --------------------------------------------------------------------------------------
# report / CLI
# --------------------------------------------------------------------------------------

def build_report(spec: Optional[SyntheticSpec] = None) -> dict:
    """Alias of :func:`run_audit` returning the JSON-safe, path-free, finite-safe report dict."""
    return MF.json_safe(run_audit(spec))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench/depth_rank_layer_audit.py", description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="output JSON path (atomic write)")
    p.add_argument("--frames", type=int, default=10, help="T, number of frames (>= 3)")
    p.add_argument("--width", type=int, default=48, help="W (>= 8)")
    p.add_argument("--height", type=int, default=36, help="H (>= 8)")
    p.add_argument("--layers", type=int, default=3, help="number of synthetic layers (>= 2)")
    p.add_argument("--depths", default="0,0.5,1", help="comma-separated raw depth value per layer")
    p.add_argument("--rank-orientation", choices=list(RANK_ORIENTATIONS), default="low_raw_low_rank",
                   help="explicit raw-value -> rank mapping (never a physical near/far claim)")
    p.add_argument("--depth-mode", choices=list(DEPTH_MODES), default="ranked",
                   help="ranked, uninformative (one rank), or shuffled (ranks carry no layer info)")
    p.add_argument("--lock-rank", type=int, default=0, help="depth rank the background-lock policy selects")
    p.add_argument("--preserve-rank", type=int, default=2, help="depth rank the foreground-preserve policy selects")
    p.add_argument("--seed", type=int, default=0, help="deterministic scene seed")
    p.add_argument("--dry-run", action="store_true", help="print resolved options and exit")
    return p


def spec_from_args(a) -> SyntheticSpec:
    """Build a :class:`SyntheticSpec` from parsed CLI options, validating the depth list length."""
    depths = tuple(float(v) for v in str(a.depths).split(",") if str(v).strip())
    if len(depths) != a.layers:
        raise ValueError(f"--depths must list exactly --layers={a.layers} values, got {len(depths)}")
    return SyntheticSpec(T=a.frames, W=a.width, H=a.height, n_layers=a.layers, depths=depths,
                         rank_orientation=a.rank_orientation, depth_mode=a.depth_mode,
                         lock_rank=a.lock_rank, preserve_rank=a.preserve_rank, seed=a.seed)


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        spec = spec_from_args(a)
    except Exception as e:
        print(json.dumps({"status": "failure", "error": {"type": type(e).__name__, "message": str(e)}},
                         sort_keys=True))
        return 1
    if a.dry_run:
        print(json.dumps({"spec": spec.as_dict(), "out": a.out}, indent=1, sort_keys=True))
        return 0
    try:
        report = build_report(spec)
        if a.out:
            MF.write_manifest(a.out, report)
    except Exception as e:
        print(json.dumps({"status": "failure", "error": {"type": type(e).__name__, "message": str(e)}},
                         sort_keys=True))
        return 1
    decisions = {p["name"]: p["decision"] for p in report["policies"]}
    print(json.dumps({"status": "ok", "out": a.out, "decisions": decisions}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
