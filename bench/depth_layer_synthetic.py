"""Isolated, pure-NumPy known-truth experiment: does selecting cells by an *explicit depth rank* recover a
requested coherent motion layer on a deterministic two-layer synthetic scene, and what do the separate
background-lock and foreground-preserve policies cost downstream?

This is a **falsifiable offline synthetic diagnostic**, not a claim that monocular depth understands semantic
foreground/background. Every quantity is a coordinate-model construction with known ground truth, so a wrong
answer is detectable rather than plausible.

Scope caution (read this)
-------------------------
The scene's raw depth values are **arbitrary labels**. :data:`RANK_ORIENTATIONS` makes the mapping from raw
value to rank *explicit and caller-chosen* (``low_raw_low_rank``: the smallest raw value is rank 0;
``low_raw_high_rank``: the largest raw value is rank 0). The module **never** infers that a rank is physical
near or physical far, and never claims a rank is semantic foreground/background. A "lock rank 0" policy is
literally "select the cells whose depth rank is 0 under the configured orientation". A rank with no layer is
reported ``unavailable`` and is never silently clamped.

Synthetic scene (ground truth)
------------------------------
:func:`build_sequence` lays out exactly two layers in a deterministic z-order: a full-frame **background**
plane (layer 0) and a rectangular **foreground distractor** overlay (layer 1) drawn on top, so the overlay
occludes the background. Per frame ``t`` each layer ``l`` has a known 2x3 affine ``M[l][t]`` (reference ->
frame-``t``):

* both layers share the camera transform ``C[t] = P[t] . J[t]`` (an intended smooth pan/zoom ``P[t]``
  composed with a deterministic per-frame jitter ``J[t]``);
* the foreground overlay additionally carries an **independent deforming local motion** (an anisotropic,
  time-varying scale plus slide) composed with the camera, so foreground and background genuinely differ.

The observed forward flow at reference coordinate ``p`` is ``fwd[t, p] = M[visible(p)][t](p) - p``. The
source depth per pixel is the raw depth value of the visible layer; ``depth_mode`` can make it
**uninformative** (both layers share one raw value, collapsing to a single rank) or **shuffled** (a
deterministic permutation that destroys the rank/layer association), for the degenerate checks. Validity is a
border band plus an optional rectangular invalid-support hole.

Estimation and held-out evaluation
----------------------------------
A deterministic checkerboard split (even ``x + y`` train, odd validate; see :func:`checkerboard`) is used.
The **pooled** baseline fits one affine displacement field on all valid train cells. Each **policy** fits the
same affine family on only its selected rank's valid train cells, fitted **from source-depth, flow and
validity alone** (the true per-pixel layer id is never used for fitting). Both are scored against the
**known layer flow** of the policy's target layer on the identical held-out block, so the comparison is on
identical cells. Support floors (:data:`MIN_FIT_SAMPLES`, :data:`MIN_EVAL_SAMPLES`) make an accidental
single-cell "win" impossible: a policy whose selection collapses (e.g. uninformative depth) reports a
machine-readable reason and declines to claim an improvement instead of fabricating one.

Under the ``"shuffled"`` negative control the observed rank field is a permutation that carries no layer
evidence by construction, so a rank-selected fit can nevertheless score an accidental frame-wise "win"
against the pooled baseline. Because the generator knows the regime, both policies are instead declared
unavailable (reason ``"rank_layer_association_destroyed"``) with a ``None`` downstream block and zero
material/comparable wins; no fit is evaluated and no target is fabricated. The pooled baseline is still
reported as a diagnostic for that regime, but it supports no policy claim.

Downstream coordinate model
---------------------------
Stabilization by an estimated per-frame path ``E[t]`` is modelled as composing ``E[t]^-1`` with the known
layer transform ``M[l][t]``. On deterministic per-layer reference sample points the residual path is
``R_l[t] = E[t]^-1 . M[l][t]`` and the metrics are:

* **residual jitter** = RMS of the discrete second difference of ``R_l[t]`` over ``t`` (a motion-smoothness
  proxy, reported per layer and aggregated);
* **valid warp coverage** = fraction of the output grid whose ``E[t]^-1`` maps inside
  ``[-0.5, W-0.5] x [-0.5, H-0.5]`` (matching the pixel-edge convention used elsewhere in the repo);
* **target-flow error** = held-out Euclidean error of the estimated transform's displacement against the
  known target-layer flow (the primary accuracy number).

The two policies are **independent blocks** with their own selected rank, their own fitted model and their
own statistics; nothing is pooled between them, no threshold is selected or swept, and no winner is declared.

Pooled-baseline downstream (counterfactual)
-------------------------------------------
The pooled baseline targets **neither** layer: it fits one affine displacement field on all valid train
cells. To let its downstream cost be compared with both policies on identical footing, ``baseline`` reports
``downstream_by_target_layer``: the same pooled path evaluated under each ground-truth layer as the
*counterfactual* intended target, keyed by the stable string layer id (``"0"``, ``"1"``). Entry ``k`` is
:func:`_policy_downstream` on the pooled per-frame estimates with ``locked_layer=k`` -- the identity intended
residual for layer ``k``, its intended in-frame motion otherwise. This is a counterfactual evaluation of the
same pooled estimate, not a claim that the pooled baseline knows or selects a target layer
(``baseline.targets_layer`` is ``None``). If any pooled fit fails the block is reported unavailable
(``downstream_available=False`` with a machine-readable ``downstream_reason``) and every entry is ``None``;
no partial or fabricated path is substituted.

Robustness controls (E10)
-------------------------
Three opt-in, default-off robustness controls stress the same fixture without changing the default
(all-zero) behaviour:

* ``flow_noise_px`` adds zero-mean i.i.d. Gaussian **measurement noise** (sigma in pixels) to the *observed*
  ``fwd_flow`` only, under a fixed named-offset seeded RNG; the known ``layer_flow`` ground truth is never
  perturbed, so noisy observations yet exact scoring remain separable.
* ``depth_error_frac`` deterministically flips that fraction of the **visible ranked** pixel labels between the
  two raw depth classes (fixed seeded permutation). It applies to ``depth_mode="ranked"`` only: the
  uninformative and shuffled regimes are left untouched. The realized flipped count/fraction is reported. It
  changes the *observed* per-pixel rank (what fitting consumes) but never the ground-truth layer labels, the
  layer transforms or the known flow.
* ``fg_nonaffine_px`` adds a deterministic **smooth, spatially varying** (sinusoidal, frame-varying) foreground
  displacement. It is folded into the known ``layer_flow`` ground truth and the observed ``fwd_flow``, but the
  affine ``layer_transform`` is kept separately as the affine component. Downstream coordinate evaluation
  applies the true sampled non-affine displacement to the warped point *before* the inverse estimated ``E``, so
  residual jitter and displacement-vs-intent error stay meaningfully distinct from coverage.

All three default to zero, so :func:`run_experiment` on the default spec is bit-for-bit the E09 fixture. The
truth layer labels are still used only to define the corruption/synthetic generation and to score; they are
never used to fit.

Determinism, JSON and scope
---------------------------
Everything is float64 NumPy with fixed seeds/permutations; :func:`run_experiment` is a pure function of its
:class:`SyntheticSpec`. The report is path-free and finite-safe (written atomically through
:mod:`bench.manifest`). No media is decoded, no RAFT/MLX/production code is imported, no weights are read,
the heavy lock is untouched, and no production default is changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

import numpy as np

from . import manifest as MF

SCHEMA = "mlxstab.depth_layer_synthetic/1"

# Explicit raw-value -> rank orientations. Never a physical near/far or semantic claim.
RANK_ORIENTATIONS = ("low_raw_low_rank", "low_raw_high_rank")

# Depth regimes. "ranked" is the informative case; the other two are deliberate degeneracies.
DEPTH_MODES = ("ranked", "uninformative", "shuffled")

POLICY_NAMES = ("background_lock", "foreground_preserve")

# Support floors. A 2-D affine per component needs 3 non-collinear cells to be identifiable, but a contract
# of 6 valid train cells and 3 held-out cells is required before a fit/report is trusted, so a single lucky
# held-out cell can never read as an improvement. Fixed constants, never swept.
MIN_FIT_SAMPLES = 6
MIN_EVAL_SAMPLES = 3

# A policy's held-out median error must beat the pooled baseline's on the identical block by at least this
# margin (synthetic pixels) in the majority of comparable frames to be recorded as a material gain. Fixed.
MATERIAL_GAIN_PX = 1e-6

# Named seed offsets: independent deterministic RNG streams derived from ``spec.seed`` so the flow-noise and
# depth-corruption draws never alias each other or the existing shuffled-permutation stream (``seed + 7919``).
SEED_FLOW_NOISE = 104729
SEED_DEPTH_ERROR = 130363

LIMITATIONS = (
    "all results are SYNTHETIC: a deterministic coordinate model with known ground-truth layer transforms; "
    "they say nothing about real footage, RAFT accuracy, or metric depth",
    "the scene's raw depth values are arbitrary labels; rank orientation is explicit and the module never "
    "asserts physical near/far or semantic foreground/background",
    "monocular depth is NOT claimed to predict semantics; the foreground overlay is a synthetic rectangle "
    "with a known affine local motion and occlusion is a construction, not realistic object behaviour",
    "held-out error uses a checkerboard split whose train/validation neighbours are spatially correlated, so "
    "it is a structural diagnostic, not an independent-sample generalisation guarantee",
    "downstream jitter and coverage are coordinate-model metrics, not rendered-pixel or perceptual quality; no "
    "frame is rendered and no production stabilizer is run",
    "no threshold is selected or swept, no policy is declared a winner, and no production wiring or default "
    "change is implied",
    "the E10 robustness controls (flow_noise_px, depth_error_frac, fg_nonaffine_px) are synthetic corruptions "
    "of the coordinate model, not calibrated models of real sensor noise, depth errors or non-rigid motion",
)


# --------------------------------------------------------------------------------------
# affine helpers (3x3 with bottom row [0,0,1]); affine-only, applied to pixel points
# --------------------------------------------------------------------------------------

def _affine_identity() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def apply_affine(m: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply a 3x3 affine ``m`` to points ``p [N,2]`` -> ``[N,2]`` (pure affine, no perspective)."""
    m = np.asarray(m, np.float64)
    p = np.asarray(p, np.float64)
    return p @ m[:2, :2].T + m[:2, 2]


def compose_affine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a . b`` (apply ``b`` first) for two 3x3 affines."""
    return np.asarray(a, np.float64) @ np.asarray(b, np.float64)


_ORIGIN = np.array([0.0, 0.0, 1.0], np.float64)


def affine_translation(m: np.ndarray) -> np.ndarray:
    """The translation component of a 3x3 affine."""
    return np.asarray(m, np.float64)[:2, 2]


# --------------------------------------------------------------------------------------
# synthetic scene
# --------------------------------------------------------------------------------------

def checkerboard(coords: np.ndarray) -> np.ndarray:
    """Deterministic split: even ``x + y`` -> train (True), odd -> validate (False)."""
    c = np.asarray(coords)
    return ((c[:, 0].astype(np.int64) + c[:, 1].astype(np.int64)) % 2) == 0


def _rect_mask(W: int, H: int, cx: float, cy: float, hw: float, hh: float) -> np.ndarray:
    """Boolean ``[H,W]`` mask of a rectangle centred at ``(cx, cy)`` with half-extents ``(hw, hh)``."""
    xs = np.arange(W)[None, :]
    ys = np.arange(H)[:, None]
    return (np.abs(xs - cx) <= hw) & (np.abs(ys - cy) <= hh)


def _camera_smooth(T: int, W: int, H: int, *, drift_px: float, zoom: float) -> np.ndarray:
    """Intended smooth camera path ``P[t]`` (reference -> frame t): gentle pan + slow zoom about centre."""
    tau = np.linspace(0.0, 1.0, max(T, 2))[:T]
    cx, cy = (W - 1.0) / 2.0, (H - 1.0) / 2.0
    P = np.zeros((T, 3, 3), np.float64)
    for i in range(T):
        s = 1.0 + zoom * tau[i]
        a = drift_px * tau[i]
        P[i] = np.array([[s, 0.0, a + (1.0 - s) * cx],
                         [0.0, s, (1.0 - s) * cy],
                         [0.0, 0.0, 1.0]], np.float64)
    return P


def _camera_jitter(T: int, W: int, H: int, *, jitter_px: float, rot_deg: float, seed: int) -> np.ndarray:
    """Deterministic zero-mean per-frame jitter ``J[t]`` (small translation + rotation about centre)."""
    t = np.arange(T, dtype=np.float64)
    jx = jitter_px * np.sin(2.399 * t + 0.7 * seed)
    jy = jitter_px * np.cos(3.117 * t + 1.3 * seed)
    jr = np.deg2rad(rot_deg) * np.sin(1.771 * t + 0.3 * seed)
    cx, cy = (W - 1.0) / 2.0, (H - 1.0) / 2.0
    J = np.zeros((T, 3, 3), np.float64)
    for i in range(T):
        ca, sa = np.cos(jr[i]), np.sin(jr[i])
        J[i] = np.array([[ca, -sa, cx - ca * cx + sa * cy + jx[i]],
                         [sa, ca, cy - sa * cx - ca * cy + jy[i]],
                         [0.0, 0.0, 1.0]], np.float64)
    return J


def _fg_local(T: int, W: int, H: int, *, amp: float) -> np.ndarray:
    """Foreground overlay's independent local motion ``L[t]``: anisotropic, time-varying scale + slide.

    The deformation is a known non-rigid *affine* scale, so the ground truth stays inside the affine model
    family both policies fit (deliberately, so a difference in error is attributable to selection, not to a
    model-family mismatch)."""
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


# Fixed spatial frequencies / temporal rate for the deterministic non-affine foreground field. Smooth by
# construction (a low-order sinusoidal basis), so the local deformation is genuinely non-affine yet bounded.
_NONAFFINE_KX = 2.0 * np.pi * 1.5
_NONAFFINE_KY = 2.0 * np.pi * 1.0
_NONAFFINE_OMEGA = 0.7


def fg_nonaffine_displacement(pts: np.ndarray, t: int, W: int, H: int, *, amp: float,
                              seed: int) -> np.ndarray:
    """Deterministic smooth foreground displacement field ``[K,2]`` at reference points ``pts [K,2]``.

    A fixed sinusoidal x/y field that also varies with frame ``t`` (phase seeded by ``seed``). ``amp`` is the
    peak displacement in pixels; ``amp == 0`` returns exact zeros, so the default fixture is unchanged. The
    field is spatially varying (not an affine map), which is why it lives outside ``layer_transform``.
    """
    pts = np.asarray(pts, np.float64)
    if amp == 0.0:
        return np.zeros((pts.shape[0], 2), np.float64)
    x = pts[:, 0] / float(max(W, 1))
    y = pts[:, 1] / float(max(H, 1))
    ph = 0.13 * float(seed)
    u = amp * np.sin(_NONAFFINE_KX * x + _NONAFFINE_OMEGA * float(t) + ph) * np.cos(_NONAFFINE_KY * y)
    v = amp * np.cos(_NONAFFINE_KY * y + _NONAFFINE_OMEGA * float(t)) * np.sin(_NONAFFINE_KX * x + ph)
    return np.stack([u, v], axis=-1)


class SyntheticSpec:
    """Immutable description of the deterministic two-layer synthetic sequence."""

    def __init__(self, *, T: int = 10, W: int = 48, H: int = 36, depths: Sequence[float] = (0.2, 0.8),
                 rank_orientation: str = "low_raw_low_rank", depth_mode: str = "ranked",
                 lock_rank: int = 0, preserve_rank: int = 1, jitter_px: float = 3.0, drift_px: float = 10.0,
                 rot_deg: float = 0.4, zoom: float = 0.02, fg_amp: float = 0.25, border: int = 3,
                 occlusion_hole: bool = True, seed: int = 0, flow_noise_px: float = 0.0,
                 depth_error_frac: float = 0.0, fg_nonaffine_px: float = 0.0,
                 fg_rect: Sequence[float] = (0.56, 0.52, 0.20, 0.20)):
        if T < 3:
            raise ValueError("T must be >= 3 so a residual second difference is defined")
        if W < 8 or H < 8:
            raise ValueError("W and H must be >= 8")
        depths = tuple(float(d) for d in depths)
        if len(depths) != 2:
            raise ValueError("depths must list exactly the two layer raw values (background, foreground)")
        if rank_orientation not in RANK_ORIENTATIONS:
            raise ValueError(f"rank_orientation must be one of {RANK_ORIENTATIONS}")
        if depth_mode not in DEPTH_MODES:
            raise ValueError(f"depth_mode must be one of {DEPTH_MODES}")
        if border < 0:
            raise ValueError("border must be >= 0")
        flow_noise_px = float(flow_noise_px)
        depth_error_frac = float(depth_error_frac)
        fg_nonaffine_px = float(fg_nonaffine_px)
        if not np.isfinite(flow_noise_px) or flow_noise_px < 0.0:
            raise ValueError("flow_noise_px must be finite and >= 0")
        if not np.isfinite(depth_error_frac) or not 0.0 <= depth_error_frac <= 1.0:
            raise ValueError("depth_error_frac must be finite and in [0, 1]")
        if not np.isfinite(fg_nonaffine_px) or fg_nonaffine_px < 0.0:
            raise ValueError("fg_nonaffine_px must be finite and >= 0")
        # Foreground rectangle as normalized fractions of the frame: (cx, cy, half_w, half_h). Validated as
        # four finite values with positive half-extents whose rectangle lies fully inside [0, 1] in both axes.
        fg_rect = tuple(float(v) for v in fg_rect)
        if len(fg_rect) != 4:
            raise ValueError("fg_rect must list exactly four values (cx, cy, half_w, half_h)")
        if not all(np.isfinite(v) for v in fg_rect):
            raise ValueError("fg_rect values must be finite")
        fcx, fcy, fhw, fhh = fg_rect
        if fhw <= 0.0 or fhh <= 0.0:
            raise ValueError("fg_rect half-extents (half_w, half_h) must be positive")
        if not (0.0 <= fcx - fhw and fcx + fhw <= 1.0):
            raise ValueError("fg_rect must lie fully inside [0, 1] horizontally (cx +/- half_w)")
        if not (0.0 <= fcy - fhh and fcy + fhh <= 1.0):
            raise ValueError("fg_rect must lie fully inside [0, 1] vertically (cy +/- half_h)")
        self.fg_rect = fg_rect
        self.T, self.W, self.H = int(T), int(W), int(H)
        self.depths = depths
        self.rank_orientation = rank_orientation
        self.depth_mode = depth_mode
        self.lock_rank = int(lock_rank)
        self.preserve_rank = int(preserve_rank)
        self.jitter_px = float(jitter_px)
        self.drift_px = float(drift_px)
        self.rot_deg = float(rot_deg)
        self.zoom = float(zoom)
        self.fg_amp = float(fg_amp)
        self.border = int(border)
        self.occlusion_hole = bool(occlusion_hole)
        self.seed = int(seed)
        self.flow_noise_px = flow_noise_px
        self.depth_error_frac = depth_error_frac
        self.fg_nonaffine_px = fg_nonaffine_px

    def as_dict(self) -> dict:
        """JSON-safe, path-free description of the spec (for the report's ``spec`` block)."""
        return {
            "T": self.T, "W": self.W, "H": self.H, "depths": list(self.depths),
            "rank_orientation": self.rank_orientation, "depth_mode": self.depth_mode,
            "lock_rank": self.lock_rank, "preserve_rank": self.preserve_rank,
            "jitter_px": self.jitter_px, "drift_px": self.drift_px, "rot_deg": self.rot_deg,
            "zoom": self.zoom, "fg_amp": self.fg_amp, "border": self.border,
            "occlusion_hole": self.occlusion_hole, "seed": self.seed,
            "flow_noise_px": self.flow_noise_px, "depth_error_frac": self.depth_error_frac,
            "fg_nonaffine_px": self.fg_nonaffine_px, "fg_rect": list(self.fg_rect),
        }


def rank_of_raw(depths: Sequence[float], orientation: str) -> np.ndarray:
    """Rank of each layer's **raw** depth value under the explicit ``orientation``.

    ``"low_raw_low_rank"``: smallest raw value -> rank 0. ``"low_raw_high_rank"``: largest raw value -> rank
    0. Ties (equal raw values, as when depth is uninformative) share the lower rank so a collapsed field
    yields one effective rank rather than an arbitrary order. No physical near/far meaning is attached.
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
    for u in np.unique(vals):
        grp = vals == u
        if grp.sum() > 1:
            ranks[grp] = ranks[grp].min()
    return ranks


def _rank_observed(values: np.ndarray, orientation: str) -> np.ndarray:
    """Per-cell rank of an *observed* depth field under ``orientation``.

    Distinct raw values are ranked densely (ties share the lower rank) so the observed field — which may be
    uninformative (one value) or shuffled (value/layer association destroyed) — is ranked exactly as those
    values earn, never by ground-truth layer identity. This is what the fits and held-out blocks consume.
    """
    if orientation not in RANK_ORIENTATIONS:
        raise ValueError(f"rank_orientation must be one of {RANK_ORIENTATIONS}")
    vals = np.asarray(values, np.float64).reshape(-1)
    uniq = np.unique(vals)
    ranks = np.empty(vals.shape, np.int64)
    if orientation == "low_raw_low_rank":
        order = uniq
    else:
        order = uniq[::-1]
    dense = {float(v): i for i, v in enumerate(order)}
    for i, v in enumerate(vals):
        ranks[i] = dense[float(v)]
    return ranks


class Sequence:
    """Known-ground-truth two-layer synthetic sequence: transforms, layer labels, depth, flow and validity."""

    def __init__(self, spec: SyntheticSpec):
        self.spec = spec
        W, H, T = spec.W, spec.H, spec.T
        self.W, self.H, self.T = W, H, T
        self.n_layers = 2
        self.coords = np.stack(np.meshgrid(np.arange(W), np.arange(H), indexing="xy"), -1).reshape(-1, 2)
        self.train = checkerboard(self.coords)

        # Scene z-order: layer 0 full-frame background, layer 1 a rectangular foreground overlay drawn on
        # top (occluding). This is a synthetic z-order, not a semantic foreground claim.
        fcx, fcy, fhw, fhh = spec.fg_rect
        fg = _rect_mask(W, H, fcx * W, fcy * H, max(2.0, fhw * W), max(2.0, fhh * H))
        self.layer_masks = np.stack([np.ones((H, W), bool), fg], 0)
        visible = np.zeros((H, W), np.int64)
        visible[fg] = 1
        self.visible_layer = visible

        # Known per-layer frame transforms: shared camera (intended path + jitter); the overlay additionally
        # composes an independent deforming local motion, so foreground and background genuinely differ.
        P = _camera_smooth(T, W, H, drift_px=spec.drift_px, zoom=spec.zoom)
        J = _camera_jitter(T, W, H, jitter_px=spec.jitter_px, rot_deg=spec.rot_deg, seed=spec.seed)
        C = np.stack([compose_affine(P[t], J[t]) for t in range(T)])
        Lm = _fg_local(T, W, H, amp=spec.fg_amp)
        self.camera = C
        self.intended_path = P
        self.layer_transform = np.zeros((2, T, 3, 3), np.float64)
        self.layer_transform[0] = C
        for t in range(T):
            self.layer_transform[1, t] = compose_affine(C[t], Lm[t])
        # Intended residual for a locked layer is identity; the overlay's intended in-frame motion is the
        # camera trend. Kept for downstream displacement-vs-intent reporting.
        self.intended_layer = np.zeros((2, T, 3, 3), np.float64)
        self.intended_layer[0] = P
        self.intended_layer[1] = P

        # Observed flow at reference coords (displacement of the visible layer), and known per-layer flow.
        # Known flow folds in the true (non-affine) foreground displacement, but the affine ``layer_transform``
        # above is kept separately as the affine component of that motion.
        vis_flat = visible.reshape(-1)
        self.layer_nonaffine = np.zeros((2, T, H * W, 2), np.float64)
        self.fwd_flow = np.zeros((T, H * W, 2), np.float64)
        self.layer_flow = np.zeros((2, T, H * W, 2), np.float64)
        for t in range(T):
            for l in range(2):
                self.layer_flow[l, t] = apply_affine(self.layer_transform[l, t], self.coords) - self.coords
                if l == 1:
                    d = fg_nonaffine_displacement(self.coords, t, W, H, amp=spec.fg_nonaffine_px,
                                                  seed=spec.seed)
                    self.layer_nonaffine[l, t] = d
                    self.layer_flow[l, t] = self.layer_flow[l, t] + d
                sel = vis_flat == l
                if sel.any():
                    self.fwd_flow[t, sel] = self.layer_flow[l, t][sel]

        # Observed-flow measurement noise: fixed-offset seeded Gaussian applied to ``fwd_flow`` only. The known
        # ``layer_flow`` ground truth above is never perturbed, and a zero sigma adds nothing (bit-exact).
        if spec.flow_noise_px > 0.0:
            rng = np.random.RandomState(spec.seed + SEED_FLOW_NOISE)
            self.fwd_flow = self.fwd_flow + spec.flow_noise_px * rng.standard_normal(self.fwd_flow.shape)

        # Source depth per pixel = raw depth value of the visible layer (collapsed/shuffled per depth_mode).
        raw = np.asarray(spec.depths, np.float64)
        if spec.depth_mode == "uninformative":
            raw = np.full(2, raw[0], np.float64)
        self.source_depth_layer = raw
        depth_flat = raw[vis_flat]
        if spec.depth_mode == "shuffled":
            rng = np.random.RandomState(spec.seed + 7919)
            depth_flat = depth_flat[rng.permutation(H * W)]
        # Depth-rank corruption (ranked mode only): deterministically flip a fraction of the visible pixel
        # labels between the two raw depth classes. Fixed offset seeded permutation; never touches the ground
        # truth (layer labels/transforms/flow) and is skipped entirely for zero fraction or degenerate modes.
        self.depth_error_flipped_count = 0
        self.depth_error_flipped_frac = 0.0
        if spec.depth_error_frac > 0.0 and spec.depth_mode == "ranked":
            n_pix = H * W
            n_flip = int(round(spec.depth_error_frac * n_pix))
            n_flip = max(0, min(n_flip, n_pix))
            if n_flip > 0:
                rng = np.random.RandomState(spec.seed + SEED_DEPTH_ERROR)
                flip_idx = rng.permutation(n_pix)[:n_flip]
                # Flip between the two raw classes via the nominal raw-value ordering of the two layers.
                two = np.asarray(spec.depths, np.float64)
                lo, hi = float(min(two)), float(max(two))
                cur = depth_flat[flip_idx]
                depth_flat[flip_idx] = np.where(cur >= hi, lo, hi)
                self.depth_error_flipped_count = int(n_flip)
                self.depth_error_flipped_frac = float(n_flip) / float(n_pix)
        self.source_depth = depth_flat.reshape(H, W)

        # Per-layer rank of the *nominal* layer raw values (used only for rank->layer resolution claims), and
        # the per-pixel rank read from the *observed* source depth under the explicit orientation. The latter
        # is what fitting uses, so an uninformative/shuffled field informs it exactly as much (or as little)
        # as the depth actually does; it is never silently replaced by the ground-truth layer rank.
        self.layer_rank = rank_of_raw(raw, spec.rank_orientation)
        self.rank_flat = _rank_observed(self.source_depth.reshape(-1), spec.rank_orientation)

        # Validity: border band + optional rectangular invalid-support hole.
        valid = np.ones((H, W), bool)
        b = spec.border
        if b > 0:
            valid[:b, :] = False
            valid[-b:, :] = False
            valid[:, :b] = False
            valid[:, -b:] = False
        if spec.occlusion_hole and W >= 12 and H >= 12:
            hole = _rect_mask(W, H, 0.80 * W, 0.74 * H, 0.06 * W, 0.06 * H)
            valid &= ~hole
        self.valid = valid

    def layer_points(self, layer: int, n: int = 25) -> np.ndarray:
        """Deterministic reference-frame sample points inside a layer's own mask."""
        idx = np.flatnonzero(self.layer_masks[layer].reshape(-1))
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
    """OLS for one displacement component: ``coeff [3]`` or ``None`` when unidentifiable/under-floored."""
    if design.shape[0] < MIN_FIT_SAMPLES:
        return None
    if np.linalg.matrix_rank(design) < 3:
        return None
    coeff, *_ = np.linalg.lstsq(design, target, rcond=None)
    return np.asarray(coeff, np.float64)


def _coeff_to_affine(cu: np.ndarray, cv: np.ndarray, W: int, H: int) -> np.ndarray:
    """Affine ``E[t]`` (reference -> frame t) from displacement coefficients ``disp = A(x/W, y/H, 1)``.

    ``E = I + D`` with linear block ``1 + [[cu0/W, cu1/H],[cv0/W, cv1/H]]`` and translation ``[cu2, cv2]``,
    so a zero displacement reproduces the identity and a pure translation stays nonsingular."""
    E = np.eye(3, dtype=np.float64)
    E[0, 0] = 1.0 + cu[0] / float(W)
    E[0, 1] = cu[1] / float(H)
    E[0, 2] = cu[2]
    E[1, 0] = cv[0] / float(W)
    E[1, 1] = 1.0 + cv[1] / float(H)
    E[1, 2] = cv[2]
    return E


def _resid_stats(resid: np.ndarray) -> dict:
    """Finite Euclidean-residual count/median/p95; ``None`` metrics when empty."""
    r = np.asarray(resid, np.float64).ravel()
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {"n": 0, "median": None, "p95": None}
    return {"n": int(r.size), "median": float(np.median(r)), "p95": float(np.percentile(r, 95))}


def _fit_policy(seq: Sequence, t: int, select_rank: Optional[int]) -> dict:
    """Fit the affine displacement model for one frame from source depth/flow/validity alone.

    ``select_rank=None`` is the pooled baseline (all valid train cells). Otherwise the fit uses only valid
    train cells whose visible-layer depth rank equals ``select_rank``. Returns fitted coefficients, the affine
    estimate, support counts and a machine-readable ``reason`` when fit/support is unavailable.
    """
    H, W = seq.H, seq.W
    valid_flat = seq.valid.reshape(-1)
    if select_rank is None:
        train_mask = valid_flat & seq.train
        scope = "pooled"
    else:
        train_mask = valid_flat & seq.train & (seq.rank_flat == int(select_rank))
        scope = f"rank_{int(select_rank)}"
    design = _design(seq.coords, train_mask, W, H)
    base = {"scope": scope, "fit_ok": False, "cu": None, "cv": None, "E": None,
            "train_count": int(train_mask.sum())}
    if design is None:
        return {**base, "reason": "insufficient_train"}
    cu = _affine_fit(design, seq.fwd_flow[t][train_mask][:, 0])
    cv = _affine_fit(design, seq.fwd_flow[t][train_mask][:, 1])
    if cu is None or cv is None:
        return {**base, "reason": "unidentifiable_fit"}
    E = _coeff_to_affine(cu, cv, W, H)
    if (not np.isfinite(E).all()) or np.linalg.matrix_rank(E[:2, :2]) < 2 or \
            abs(np.linalg.det(E[:2, :2])) < 1e-9:
        return {**base, "reason": "degenerate_transform"}
    return {"scope": scope, "fit_ok": True, "reason": None, "cu": cu, "cv": cv, "E": E,
            "train_count": int(train_mask.sum())}


def _eval_block(seq: Sequence, rank: int) -> np.ndarray:
    """Held-out scoring block: odd-parity valid cells whose visible-layer rank equals ``rank``."""
    valid_flat = seq.valid.reshape(-1)
    return valid_flat & (~seq.train) & (seq.rank_flat == int(rank))


def _heldout_error(seq: Sequence, t: int, E: np.ndarray, eval_mask: np.ndarray, target_layer: int) -> dict:
    """Held-out error of an estimated transform against the **known** target-layer flow on ``eval_mask``."""
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
    """Fraction of the output grid whose ``E^-1`` maps inside ``[-0.5, W-0.5] x [-0.5, H-0.5]``."""
    pts = np.stack(np.meshgrid(np.arange(W), np.arange(H), indexing="xy"), -1).reshape(-1, 2).astype(np.float64)
    mapped = apply_affine(np.linalg.inv(E), pts)
    x, y = mapped[:, 0], mapped[:, 1]
    inside = (x >= -0.5) & (x <= W - 0.5) & (y >= -0.5) & (y <= H - 0.5)
    return {"n": int(pts.shape[0]), "in_bounds": int(inside.sum()),
            "frac": float(inside.mean()) if pts.shape[0] else None}


def _residual_jitter(residual_path: np.ndarray) -> Optional[float]:
    """RMS of the discrete second difference of a ``[T, K, 2]`` residual-path displacement over ``t``."""
    rp = np.asarray(residual_path, np.float64)
    if rp.shape[0] < 3:
        return None
    acc = rp[2:] - 2.0 * rp[1:-1] + rp[:-2]
    per_frame = np.linalg.norm(acc, axis=-1).mean(axis=1)
    return float(np.sqrt(np.mean(per_frame ** 2)))


def _layer_downstream(seq: Sequence, E_by_frame: np.ndarray, layer: int, locked_layer: int,
                      *, n_points: int = 25) -> dict:
    """Downstream metrics for one layer under per-frame estimated transforms ``E_by_frame [T,3,3]``.

    Residual path ``R_l[t] = E[t]^-1(warp_true(p)) - p`` at ``layer``'s reference sample points, where
    ``warp_true(p) = M[l][t](p) + d_l(p)`` is the affine layer transform composed with the known sampled
    **non-affine** displacement ``d_l`` (the true displacement is applied *before* the inverse estimated
    ``E``). Intended residual is identity for a locked layer and the layer's intended in-frame motion
    otherwise. Reports residual jitter, displacement error vs intent and mean valid warp coverage."""
    pts = seq.layer_points(layer, n_points)
    if pts.shape[0] == 0:
        return {"layer": int(layer), "n_points": 0, "residual_jitter_px": None,
                "displacement_error_px": None, "displacement_error_p95_px": None,
                "coverage_frac_min": None, "coverage_frac_mean": None, "reason": "empty_layer_mask"}
    T = seq.T
    residual = np.zeros((T, pts.shape[0], 2), np.float64)
    intended = np.zeros((T, pts.shape[0], 2), np.float64)
    covs = []
    for t in range(T):
        Einv = np.linalg.inv(E_by_frame[t])
        warped = apply_affine(seq.layer_transform[layer, t], pts)
        if layer == 1:
            warped = warped + fg_nonaffine_displacement(
                pts, t, seq.W, seq.H, amp=seq.spec.fg_nonaffine_px, seed=seq.spec.seed)
        moved = apply_affine(Einv, warped)
        residual[t] = moved - pts
        if layer == locked_layer:
            intended[t] = 0.0
        else:
            intended[t] = apply_affine(seq.intended_layer[layer, t], pts) - pts
        covs.append(_coverage(E_by_frame[t], seq.W, seq.H)["frac"])
    err = np.linalg.norm(residual - intended, axis=-1).ravel()
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


def _policy_downstream(seq: Sequence, E_by_frame: np.ndarray, locked_layer: int) -> dict:
    """Independent downstream block for one policy: per-layer metrics plus aggregate jitter/coverage."""
    layers = [_layer_downstream(seq, E_by_frame, l, locked_layer) for l in range(seq.n_layers)]
    jitters = [d["residual_jitter_px"] for d in layers if d["residual_jitter_px"] is not None]
    covs = [d["coverage_frac_mean"] for d in layers if d["coverage_frac_mean"] is not None]
    locked = layers[locked_layer] if 0 <= locked_layer < len(layers) else None
    return {
        "locked_layer": int(locked_layer),
        "layers": layers,
        "aggregate_residual_jitter_px": float(np.mean(jitters)) if jitters else None,
        "aggregate_coverage_frac_mean": float(np.mean(covs)) if covs else None,
        "locked_layer_residual_jitter_px": locked["residual_jitter_px"] if locked else None,
        "locked_layer_displacement_error_px": locked["displacement_error_px"] if locked else None,
    }


# --------------------------------------------------------------------------------------
# rank -> layer resolution and policy blocks
# --------------------------------------------------------------------------------------

def resolve_rank_layer(seq: Sequence, rank: int) -> dict:
    """Map an explicit depth ``rank`` to a target layer without any near/far claim.

    Returns ``{"rank","layer","available","ambiguous","reason"}``. ``layer`` is the ground-truth layer whose
    rank equals ``rank``; when several layers share that rank (e.g. uninformative depth collapses both to rank
    0) the lowest scene-order layer is reported and ``ambiguous`` is set. When no layer has that rank,
    ``available`` is False with a machine-readable reason (never silently clamped)."""
    matches = np.flatnonzero(seq.layer_rank == int(rank))
    if matches.size == 0:
        return {"rank": int(rank), "layer": None, "available": False, "ambiguous": False,
                "reason": "rank_not_present"}
    return {"rank": int(rank), "layer": int(matches[0]), "available": True,
            "ambiguous": bool(matches.size > 1),
            "reason": "rank_shared_by_multiple_layers" if matches.size > 1 else None}


def _target_layer_for_rank(seq: Sequence, rank: int) -> int:
    """The ground-truth layer a policy must preserve: the layer whose rank equals ``rank``.

    Unlike :func:`resolve_rank_layer` this is *not* clamped to the lowest layer when the rank is shared:
    scoring always uses the layer the caller asked to preserve, so an uninformative field cannot make a
    policy look good by silently retargeting it onto a layer it did not select."""
    matches = np.flatnonzero(seq.layer_rank == int(rank))
    return int(matches[0]) if matches.size else -1

# --------------------------------------------------------------------------------------
# per-policy blocks: held-out target-flow error against the known target-layer flow
# --------------------------------------------------------------------------------------

def _summarize_target_error(per_frame: Sequence[dict], baseline: Sequence[dict]) -> dict:
    """Aggregate held-out target-flow error across frames and compare against the pooled baseline.

    ``per_frame``/``baseline`` are per-frame dicts with a finite ``median`` or ``None``. Identical frames are
    compared (both blocks score the same held-out cells), so a count of frame-wise wins is meaningful and an
    absent baseline frame is never counted as a win. No threshold is swept: the only fixed gate is
    :data:`MATERIAL_GAIN_PX`.
    """
    meds = [d["median"] for d in per_frame if d.get("median") is not None]
    base_meds = [d["median"] for d in baseline if d.get("median") is not None]
    p95s = [d["p95"] for d in per_frame if d.get("p95") is not None]
    n_samples = int(sum(int(d.get("n", 0)) for d in per_frame))
    wins = 0
    comparable = 0
    deltas = []
    for pol, base in zip(per_frame, baseline):
        pm, bm = pol.get("median"), base.get("median")
        if pm is None or bm is None:
            continue
        comparable += 1
        deltas.append(pm - bm)
        if pm < bm - MATERIAL_GAIN_PX:
            wins += 1
    return {
        "frame_median_median": float(np.median(meds)) if meds else None,
        "frame_median_p95": float(np.percentile(p95s, 95)) if p95s else None,
        "total_eval_samples": n_samples,
        "baseline_frame_median_median": float(np.median(base_meds)) if base_meds else None,
        "comparable_frames": comparable,
        "material_win_frames": wins,
        "median_frame_delta_px": float(np.median(deltas)) if deltas else None,
        "material_gain_px": MATERIAL_GAIN_PX,
    }


def _policy_block(seq: Sequence, spec: SyntheticSpec, *, policy: str, select_rank: int) -> dict:
    """One independent policy block: fit from source depth+flow+validity, score on held-out layer flow.

    The target layer scored against is the ground-truth layer whose rank equals ``select_rank`` under the
    configured orientation (:func:`_target_layer_for_rank`); when no layer carries that rank the policy is
    reported ``available=False`` and declines to claim anything. When the rank is present the fit uses only
    valid train cells of that rank and is scored on the held-out (odd-parity) cells of the same rank against
    the *known* flow of the target layer.

    Under the ``"shuffled"`` negative control the per-pixel depth field is a permutation that destroys the
    rank/layer association by construction, so neither layer-selection policy may be fitted, scored or
    credited: both are reported ``available=False`` with reason ``"rank_layer_association_destroyed"`` and a
    ``None`` downstream block, and every win/comparable counter is left at zero. The generator knows the
    regime (:data:`DEPTH_MODES`), so this is a refusal to report a policy claim, not a silently-failed fit.
    """
    if spec.depth_mode == "shuffled":
        return {
            "policy": policy, "target_rank": int(select_rank), "target_layer": None,
            "available": False, "reason": "rank_layer_association_destroyed",
            "resolution": {"rank": int(select_rank), "layer": None, "available": False,
                            "reason": "rank_layer_association_destroyed"},
            "eval_block_count": 0, "frames": [],
            "target_error": _summarize_target_error([], []), "downstream": None,
        }
    target_layer = _target_layer_for_rank(seq, select_rank)
    matches = int(np.count_nonzero(seq.layer_rank == int(select_rank)))
    resolution = {"rank": int(select_rank), "layer": (int(target_layer) if target_layer >= 0 else None),
                  "available": bool(target_layer >= 0 and matches == 1)}
    if target_layer < 0:
        return {
            "policy": policy, "target_rank": int(select_rank), "target_layer": None,
            "available": False, "reason": "target_rank_unavailable",
            "resolution": {**resolution, "reason": "rank_not_present"},
            "eval_block_count": 0, "frames": [], "target_error": _summarize_target_error([], []),
            "downstream": None,
        }
    if matches > 1:
        # Uninformative depth: both layers collapse onto one rank, so the explicit rank no longer isolates a
        # single layer. The policy declines (reason is machine-readable) instead of silently scoring the
        # pooled field or clamping to the lowest layer.
        return {
            "policy": policy, "target_rank": int(select_rank), "target_layer": None,
            "available": False, "reason": "rank_ambiguous_uninformative_depth",
            "resolution": {**resolution, "reason": "rank_shared_by_multiple_layers"},
            "eval_block_count": int(_eval_block(seq, select_rank).sum()), "frames": [],
            "target_error": _summarize_target_error([], []), "downstream": None,
        }

    eval_mask = _eval_block(seq, select_rank)
    E_by_frame = np.zeros((spec.T, 3, 3), np.float64)
    base_by_frame = np.zeros((spec.T, 3, 3), np.float64)
    frames = []
    fc = 0
    avail = True
    reason = None
    for t in range(spec.T):
        fit = _fit_policy(seq, t, select_rank)
        base = _fit_policy(seq, t, None)
        frame = {"frame": t, "fit_ok": fit["fit_ok"], "fit_reason": fit["reason"],
                 "train_count": fit["train_count"], "target_error": None, "baseline_error": None,
                 "baseline_fit_ok": base["fit_ok"]}
        if not fit["fit_ok"]:
            avail = False
            reason = reason or fit["reason"]
            frames.append(frame)
            continue
        E_by_frame[t] = fit["E"]
        err = _heldout_error(seq, t, fit["E"], eval_mask, target_layer)
        frame["target_error"] = {"n": err["n"], "median": err["median"], "p95": err["p95"],
                                 "reason": err["reason"]}
        if base["fit_ok"]:
            base_by_frame[t] = base["E"]
            berr = _heldout_error(seq, t, base["E"], eval_mask, target_layer)
            frame["baseline_error"] = {"n": berr["n"], "median": berr["median"], "p95": berr["p95"],
                                       "reason": berr["reason"]}
        fc += 1
        frames.append(frame)

    error_empty = not any(f["target_error"] is not None and
                          f["target_error"].get("median") is not None for f in frames)
    target_error = _summarize_target_error(
        [f["target_error"] or {} for f in frames],
        [f["baseline_error"] or {} for f in frames])

    if avail:
        downstream = _policy_downstream(seq, E_by_frame, locked_layer=target_layer)
    else:
        downstream = None

    return {
        "policy": policy, "target_rank": int(select_rank), "target_layer": int(target_layer),
        "available": bool(avail and not error_empty),
        "reason": None if (avail and not error_empty) else (reason or "insufficient_validation"),
        "resolution": {**resolution, "reason": None},
        "eval_block_count": int(eval_mask.sum()),
        "fit_frame_count": fc,
        "frames": frames,
        "target_error": target_error,
        "downstream": downstream,
    }


# --------------------------------------------------------------------------------------
# experiment
# --------------------------------------------------------------------------------------

def run_experiment(spec: Optional[SyntheticSpec] = None) -> dict:
    """Pure function of ``spec``: build the sequence, run both policies and return a JSON-safe report.

    Both policy blocks are independent: the background-lock policy selects ``spec.lock_rank`` and the
    foreground-preserve policy selects ``spec.preserve_rank`` under the configured explicit orientation.
    Nothing is pooled between them, no threshold is selected and no winner is declared. Under uninformative
    depth both policies report ``available=False`` (the rank collapses or is absent) rather than fabricating
    an improvement; under shuffled depth the rank/layer association is destroyed by construction, so both
    policies report ``available=False`` with reason ``"rank_layer_association_destroyed"``, a ``None``
    downstream block and zero material/comparable wins. The pooled baseline is retained as a diagnostic in
    every regime but carries no policy claim; its ``downstream_by_target_layer`` block is a counterfactual
    evaluation of the same pooled path under each ground-truth layer as intended target (never a claim that
    the baseline targets a layer).
    """
    spec = spec or SyntheticSpec()
    seq = build_sequence(spec)

    baseline_frames = []
    baseline_E = np.zeros((spec.T, 3, 3), np.float64)
    pooled_fit_ok = True
    pooled_fail_reason = None
    for t in range(spec.T):
        fit = _fit_policy(seq, t, None)
        baseline_frames.append({"frame": t, "fit_ok": fit["fit_ok"], "fit_reason": fit["reason"],
                                "train_count": fit["train_count"]})
        if fit["fit_ok"]:
            baseline_E[t] = fit["E"]
        else:
            pooled_fit_ok = False
            pooled_fail_reason = pooled_fail_reason or fit["reason"]

    # Counterfactual downstream evaluation of the SAME pooled path under each intended target. The pooled
    # baseline targets neither layer: here we only ask what a stabilizer driven by the pooled estimate would
    # cost on each ground-truth layer if that layer were the intended target (identity intended residual for
    # the indexed layer, its intended in-frame motion otherwise). Stable string keys are the layer ids. When
    # any pooled fit fails the block is reported unavailable with a machine-readable reason and no metrics,
    # rather than substituting a partial path.
    if pooled_fit_ok:
        baseline_downstream = {
            str(l): _policy_downstream(seq, baseline_E, locked_layer=l) for l in range(seq.n_layers)
        }
        baseline_downstream_reason = None
    else:
        baseline_downstream = {str(l): None for l in range(seq.n_layers)}
        baseline_downstream_reason = pooled_fail_reason or "pooled_fit_unavailable"

    policies = [
        _policy_block(seq, spec, policy="background_lock", select_rank=spec.lock_rank),
        _policy_block(seq, spec, policy="foreground_preserve", select_rank=spec.preserve_rank),
    ]

    return {
        "schema": SCHEMA,
        "spec": spec.as_dict(),
        "orientation": {
            "rank_orientation": spec.rank_orientation,
            "rank_by_layer": [int(r) for r in seq.layer_rank],
            "claim": "raw values are arbitrary labels; rank orientation is explicit and never physical",
        },
        "depth_mode": spec.depth_mode,
        "n_layers": int(seq.n_layers),
        "layer_rank": [int(r) for r in seq.layer_rank],
        "corruption": {
            "flow_noise_px": spec.flow_noise_px,
            "depth_error_frac": spec.depth_error_frac,
            "depth_error_applied": bool(spec.depth_error_frac > 0.0 and spec.depth_mode == "ranked"),
            "depth_error_realized_count": int(seq.depth_error_flipped_count),
            "depth_error_realized_frac": float(seq.depth_error_flipped_frac),
            "fg_nonaffine_px": spec.fg_nonaffine_px,
        },
        "baseline": {
            "scope": "pooled",
            "frames": baseline_frames,
            "fit_frame_count": int(sum(1 for f in baseline_frames if f["fit_ok"])),
            "targets_layer": None,
            "downstream_available": bool(pooled_fit_ok),
            "downstream_reason": baseline_downstream_reason,
            "downstream_by_target_layer": baseline_downstream,
        },
        "policies": policies,
        "min_fit_samples": MIN_FIT_SAMPLES,
        "min_eval_samples": MIN_EVAL_SAMPLES,
        "material_gain_px": MATERIAL_GAIN_PX,
        "limitations": list(LIMITATIONS),
        "notes": {
            "synthetic_only": True,
            "no_media_decoded": True,
            "no_production_code_imported": True,
            "winner_declared": False,
        },
    }


def dumps_report(report: dict) -> str:
    """Deterministic JSON serialization of a report, finite-safe and path-free."""
    return MF.dumps_json(report)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench/depth_layer_synthetic.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--T", type=int, default=10, help="number of frames (>= 3)")
    p.add_argument("--W", type=int, default=48, help="frame width (>= 8)")
    p.add_argument("--H", type=int, default=36, help="frame height (>= 8)")
    p.add_argument("--depths", default="0.2,0.8",
                   help="comma-separated two raw layer depth values (background,foreground)")
    p.add_argument("--rank-orientation", choices=list(RANK_ORIENTATIONS), default="low_raw_low_rank")
    p.add_argument("--depth-mode", choices=list(DEPTH_MODES), default="ranked")
    p.add_argument("--lock-rank", type=int, default=0)
    p.add_argument("--preserve-rank", type=int, default=1)
    p.add_argument("--jitter-px", type=float, default=3.0)
    p.add_argument("--drift-px", type=float, default=10.0)
    p.add_argument("--rot-deg", type=float, default=0.4)
    p.add_argument("--zoom", type=float, default=0.02)
    p.add_argument("--fg-amp", type=float, default=0.25)
    p.add_argument("--flow-noise-px", type=float, default=0.0,
                   help="sigma of Gaussian measurement noise added to observed fwd_flow only (>= 0)")
    p.add_argument("--depth-error-frac", type=float, default=0.0,
                   help="fraction in [0,1] of visible ranked pixel labels flipped between raw depth classes")
    p.add_argument("--fg-nonaffine-px", type=float, default=0.0,
                   help="peak amplitude of the deterministic non-affine foreground displacement (>= 0)")
    p.add_argument("--fg-rect", default="0.56,0.52,0.20,0.20",
                   help="normalized foreground rectangle cx,cy,half_w,half_h (fractions in [0,1])")
    p.add_argument("--border", type=int, default=3)
    p.add_argument("--no-occlusion-hole", action="store_true", help="omit the rectangular invalid-support hole")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="output JSON path (atomic write)")
    p.add_argument("--dry-run", action="store_true", help="print resolved spec and exit")
    return p


def parse_depths(text) -> tuple:
    """Parse ``"a,b"`` into exactly two finite raw depth values."""
    if isinstance(text, (list, tuple)):
        vals = [float(v) for v in text]
    else:
        vals = [float(v) for v in str(text).split(",") if str(v).strip()]
    if len(vals) != 2:
        raise ValueError("--depths must list exactly two values (background,foreground)")
    if not all(np.isfinite(v) for v in vals):
        raise ValueError("--depths values must be finite")
    return tuple(vals)


def parse_fg_rect(text) -> tuple:
    """Parse ``"cx,cy,half_w,half_h"`` into four finite normalized foreground-rectangle values."""
    if isinstance(text, (list, tuple)):
        vals = [float(v) for v in text]
    else:
        vals = [float(v) for v in str(text).split(",") if str(v).strip()]
    if len(vals) != 4:
        raise ValueError("--fg-rect must list exactly four values (cx,cy,half_w,half_h)")
    if not all(np.isfinite(v) for v in vals):
        raise ValueError("--fg-rect values must be finite")
    return tuple(vals)


def spec_from_args(a: argparse.Namespace) -> SyntheticSpec:
    """Build a validated :class:`SyntheticSpec` from parsed CLI arguments."""
    return SyntheticSpec(
        T=a.T, W=a.W, H=a.H, depths=parse_depths(a.depths), rank_orientation=a.rank_orientation,
        depth_mode=a.depth_mode, lock_rank=a.lock_rank, preserve_rank=a.preserve_rank,
        jitter_px=a.jitter_px, drift_px=a.drift_px, rot_deg=a.rot_deg, zoom=a.zoom, fg_amp=a.fg_amp,
        border=a.border, occlusion_hole=not a.no_occlusion_hole, seed=a.seed,
        flow_noise_px=a.flow_noise_px, depth_error_frac=a.depth_error_frac,
        fg_nonaffine_px=a.fg_nonaffine_px, fg_rect=parse_fg_rect(a.fg_rect))


def run(opts: argparse.Namespace, log=print) -> dict:
    """Build the spec, run the experiment and optionally write the atomic JSON report."""
    spec = spec_from_args(opts)
    report = run_experiment(spec)
    if opts.out:
        MF.write_manifest(opts.out, report)
    return report


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.dry_run:
        print(json.dumps(spec_from_args(a).as_dict(), indent=1, sort_keys=True))
        return 0
    try:
        report = run(a, log=lambda m: print(m, file=sys.stderr))
    except Exception as e:
        print(json.dumps({"status": "failure", "error": {"type": type(e).__name__, "message": str(e)}},
                         sort_keys=True))
        return 1
    print(json.dumps({"status": "ok", "out": a.out, "schema": report["schema"],
                      "policy_available": {p["policy"]: p["available"] for p in report["policies"]}},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
