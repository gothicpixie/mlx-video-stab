"""Isolated, pure-NumPy rendered-validity audit on the deterministic two-layer synthetic scene.

This module closes one narrow gap left by the coordinate-model diagnostics in
:mod:`bench.depth_layer_synthetic` and :mod:`bench.depth_rank_layer_audit`: those measure a *coordinate model*
(a residual path, a crop-coverage fraction, a flow error) but never rasterize a single pixel, so "valid
coverage" there is a geometric fraction, not a statement about which *rendered* pixels actually survived the
warp. Here we build deterministic RGB textures (no assets, no media, no weights), forward-splat them through
each synthetic layer's known affine transform plus ``seq.layer_nonaffine``, composite them, and then *pull*-
warp the rendered stack through the inverse estimated ``E[t]`` with a pure-NumPy bilinear sampler and a
conservative validity mask. The report records per-frame and per-policy in-bounds coverage, *actual rendered-
valid* coverage, finite RGB fraction and RGB variance.

Scope caution (read this)
-------------------------
Everything here is **synthetic pixel bookkeeping**, not perceptual or production evidence:

* the textures are procedurally generated RGB fields, not photographs; a "valid rendered pixel" means a
  deterministic splat landed there, nothing about visual quality;
* forward *splatting* with nearest-pixel rasterization leaves **holes** where the forward map is not surjective
  (and can collapse several sources onto one destination, resolved by an explicit collision policy). A real
  renderer would use a backward sampler; the hole rate here is an artefact of the synthetic splat, not a
  property of stabilization;
* the pull-warp uses the *estimated* ``E[t]`` from :func:`bench.depth_layer_synthetic._fit_policy`, and the
  estimated ``E`` on shuffled/uninformative depth is not a policy claim (see below);
* no threshold is selected, no policy is declared a winner, and no production stabilizer, renderer or default
  is touched.

Rendering model
---------------
Textures. :func:`build_textures` returns a ``[2, H, W, 3]`` float64 RGB stack, one deterministic texture per
layer, built from fixed sinusoids of the reference coordinates (background) and a distinct sinusoid plus a
domain-separating offset (foreground). No RNG, no files.

Forward splat per frame. For frame ``t`` and layer ``l`` the reference pixel ``p`` maps to
``q = M[l][t](p) + d_l(p)`` where ``M[l][t] = seq.layer_transform[l, t]`` is the known affine and
``d_l = seq.layer_nonaffine[l, t]`` is the known sampled non-affine displacement (exact zeros for the
background and for the default zero non-affine amplitude). ``q`` is rounded to the nearest integer pixel;
destination pixels outside ``[0, W-1] x [0, H-1]`` are dropped (this is the source of splat holes). The scene
composites layer 0 first, then layer 1 on top, so the explicit collision policy is **later layer wins**: a
destination pixel written by layer 1 overrides layer 0, and within one layer the last reference pixel in
deterministic row-major order wins. A per-pixel render-validity mask marks destinations that received at least
one splat.

Pull-warp. :func:`pull_warp` maps every *output* pixel ``o`` back to a source location ``E[t]^-1(o)`` and
bilinearly samples the rendered frame there with a **conservative** validity mask: a pixel is valid only when
the mapped sample is strictly inside the pixel-edge box ``[0, W-1] x [0, H-1]`` (so all four bilinear taps are
in bounds) **and** all four contributing source-mask taps are valid. The conservative rule makes the warped
valid mask a subset of the source mask's interior; an output pixel adjacent to a hole is dropped rather than
mixed with an invalid neighbour.

Policies and the pooled baseline
--------------------------------
The two policies are the same explicit rank selections as the coordinate-model audit: **background_lock**
selects ``spec.lock_rank`` and **foreground_preserve** selects ``spec.preserve_rank``. Each has its own
estimated ``E[t]`` per frame (:func:`bench.depth_layer_synthetic._fit_policy`) and therefore its own rendered-
validity numbers; nothing is pooled between them. The **pooled baseline** fits one affine on all valid train
cells and is reported as a single common diagnostic path. When a policy's rank is unavailable -- under
uninformative depth the rank collapses, under shuffled depth the rank/layer association is destroyed, and a
rank outside the scene is simply absent -- the policy is reported ``available=False`` with a machine-readable
reason and **no rendered frames are fabricated** (its render block is ``None``). The pooled baseline remains a
diagnostic in those regimes.

Determinism, JSON and scope
---------------------------
Everything is float64 NumPy with fixed seeds; :func:`run_audit` is a pure function of its
:class:`bench.depth_layer_synthetic.SyntheticSpec`. The report contains **no image pixels**, is path-free and
finite-safe (written atomically through :mod:`bench.manifest`). No media is decoded, no RAFT/MLX/production
code is imported, no weights are read, the heavy lock is untouched, and no production default is changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np

from . import manifest as MF
from . import depth_layer_synthetic as DLS
from .depth_layer_synthetic import SyntheticSpec

SCHEMA = "mlxstab.depth_layer_render_audit/1"

# Composite and collision policy identifiers, recorded in the report so the synthetic construction is explicit.
COLLISION_POLICY = "later_layer_wins_then_row_major_last"
RASTERIZATION = "nearest_forward_splat"

LIMITATIONS = (
    "all results are SYNTHETIC pixel bookkeeping on procedurally generated RGB textures; they say nothing "
    "about real footage, perceptual quality, or production stabilization",
    "textures are deterministic sinusoids, not photographs; a rendered-valid pixel means a splat landed there, "
    "not that the pixel is visually meaningful",
    "forward splatting with nearest-pixel rasterization leaves HOLES where the forward map is not surjective "
    "and can collapse several sources onto one destination; the hole rate is an artefact of the synthetic "
    "splat, not a property of any stabilizer",
    "the pull-warp uses the estimated E[t] from the depth-layer fit; its validity mask is a conservative "
    "sampling of the rendered mask, not a perceptual or edge-aware measure",
    "the pooled baseline is a single diagnostic path that targets neither layer; it supports no policy claim",
    "no threshold is selected or swept, no policy is declared a winner, and no production wiring, renderer or "
    "default change is implied",
)


# --------------------------------------------------------------------------------------
# deterministic textures
# --------------------------------------------------------------------------------------

def build_textures(W: int, H: int, *, n_layers: int = 2) -> np.ndarray:
    """Deterministic ``[n_layers, H, W, 3]`` float64 RGB textures (no RNG, no assets, no media).

    Layer 0 is a two-frequency background gradient; each higher layer adds a distinct frequency set and a
    per-layer offset in both value and hue so neighbouring layers never coincide and foreground compositing is
    observable. Values are finite and bounded to ``[0, 1]``; a non-constant texture has non-zero variance.
    """
    xs = np.arange(W, dtype=np.float64)[None, :]
    ys = np.arange(H, dtype=np.float64)[:, None]
    u = xs / float(max(W - 1, 1))
    v = ys / float(max(H - 1, 1))
    out = np.zeros((n_layers, H, W, 3), np.float64)
    for l in range(n_layers):
        f = 1.0 + 0.75 * l
        base = 0.5 + 0.5 * np.sin(2.0 * np.pi * f * u + 0.31 * l) * np.cos(2.0 * np.pi * f * v)
        r = 0.5 + 0.5 * np.sin(2.0 * np.pi * (f + 0.25) * u + 0.7 * l)
        g = 0.5 + 0.5 * np.cos(2.0 * np.pi * (f + 0.5) * v + 0.9 * l)
        b = 0.5 + 0.5 * np.sin(2.0 * np.pi * (f + 0.75) * (u + v) + 1.1 * l)
        out[l, :, :, 0] = 0.20 + 0.60 * r
        out[l, :, :, 1] = 0.20 + 0.60 * g
        out[l, :, :, 2] = base * 0.6 + 0.2 * b
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------------------
# forward splat rendering
# --------------------------------------------------------------------------------------

def _fg_nonaffine_for(seq: DLS.Sequence, layer: int, t: int) -> np.ndarray:
    """Known non-affine displacement ``[H*W, 2]`` for ``layer`` at ``t`` (exact zeros off layer 1)."""
    if layer == 1:
        return DLS.fg_nonaffine_displacement(seq.coords, t, seq.W, seq.H,
                                             amp=seq.spec.fg_nonaffine_px, seed=seq.spec.seed)
    return np.zeros((seq.coords.shape[0], 2), np.float64)


def render_frame(seq: DLS.Sequence, textures: np.ndarray, t: int) -> dict:
    """Forward-splat every layer's visible reference pixels into one composited frame.

    Layer 0 is splatted first, layer 1 on top; the collision policy is :data:`COLLISION_POLICY` (a destination
    pixel written by a later layer overrides an earlier one, and within a layer the last reference pixel in
    deterministic row-major order wins). Returns the composited ``[H, W, 3]`` frame, the per-pixel splat-
    validity mask and per-layer splat pixel counts. No pixel is invented: an unwritten destination stays at
    zero RGB with ``valid=False`` (a hole).
    """
    H, W = seq.H, seq.W
    tex_flat = textures.reshape(textures.shape[0], H * W, 3)
    frame = np.zeros((H * W, 3), np.float64)
    valid = np.zeros(H * W, bool)
    per_layer_splats = []
    for l in range(min(seq.n_layers, textures.shape[0])):
        pts = DLS.apply_affine(seq.layer_transform[l, t], seq.coords) + _fg_nonaffine_for(seq, l, t)
        qx = np.rint(pts[:, 0]).astype(np.int64)
        qy = np.rint(pts[:, 1]).astype(np.int64)
        inside = (qx >= 0) & (qx < W) & (qy >= 0) & (qy < H)
        # Composites only where the source pixel is actually visible (z-order), matching the observed scene.
        vis = seq.visible_layer.reshape(-1) == l
        sel = np.flatnonzero(inside & vis)
        n_splat = 0
        if sel.size:
            # Deterministic row-major last-writer-wins within the layer: ascending flat index order.
            order = sel[np.argsort(sel, kind="stable")]
            dest = qy[order] * W + qx[order]
            frame[dest] = tex_flat[l][order]  # later index overwrites earlier: explicit collision policy
            valid[dest] = True
            n_splat = int(np.unique(dest).size)
        per_layer_splats.append({"layer": int(l), "visible_sources": int(vis.sum()),
                                 "splatted_pixels": n_splat})
    return {"frame": frame.reshape(H, W, 3), "valid": valid.reshape(H, W),
            "per_layer_splats": per_layer_splats, "splatted_pixels": int(valid.sum())}


# --------------------------------------------------------------------------------------
# pull-warp with bilinear sampling and conservative mask
# --------------------------------------------------------------------------------------

def _bilinear_sample(img: np.ndarray, x: np.ndarray, y: np.ndarray, m: np.ndarray):
    """Bilinear sample ``img [H,W,C]`` at float coords ``(x, y)``; returns ``[..., C]`` and tap-validity.

    Uses floor/ceil taps so a coordinate exactly on an integer pixel samples only that pixel. ``m`` is the
    source validity mask ``[H,W]``; a sample is valid only when all four contributing taps are in bounds and
    valid. For simplicity this returns the per-tap valid mask ``[N,4]`` and the sampled values.
    """
    H, W, C = img.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = x - x0
    wy = y - y0
    # In-bounds test for each of the four taps.
    inb = [(x0 >= 0), (x1 >= 0), (x0 <= W - 1), (x1 <= W - 1)]
    inb_y = [(y0 >= 0), (y1 >= 0), (y0 <= H - 1), (y1 <= H - 1)]
    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)
    taps = np.stack([
        img[y0c, x0c],  # (x0, y0)
        img[y0c, x1c],  # (x1, y0)
        img[y1c, x0c],  # (x0, y1)
        img[y1c, x1c],  # (x1, y1)
    ], axis=1)  # [N,4,C]
    valid_taps = np.stack([
        m[y0c, x0c] & inb[0] & inb_y[0],
        m[y0c, x1c] & inb[1] & inb_y[0],
        m[y1c, x0c] & inb[0] & inb_y[1],
        m[y1c, x1c] & inb[1] & inb_y[1],
    ], axis=1)  # [N,4]
    w00 = (1.0 - wx) * (1.0 - wy)
    w10 = wx * (1.0 - wy)
    w01 = (1.0 - wx) * wy
    w11 = wx * wy
    w = np.stack([w00, w10, w01, w11], axis=1)[..., None]  # [N,4,1]
    return taps, valid_taps, w


def pull_warp(frame: np.ndarray, source_valid: np.ndarray, E: np.ndarray) -> dict:
    """Pull-warp a rendered frame through ``E[t]^-1`` with bilinear sampling and a conservative mask.

    Output pixel ``o`` samples the frame at ``s = E^-1(o)``. A pixel is marked valid only when the bilinear
    sample is **strictly inside** the pixel-edge box ``[0, W-1] x [0, H-1]`` (all four taps in bounds) **and**
    all four contributing source-validity taps are valid. Returns the warped image ``[H,W,3]``, the warped
    validity mask ``[H,W]``, the geometric in-bounds mask (pixel edges only) and finite-pixel counts.
    """
    H, W, _ = frame.shape
    pts = np.stack(np.meshgrid(np.arange(W), np.arange(H), indexing="xy"), -1).reshape(-1, 2).astype(np.float64)
    mapped = DLS.apply_affine(np.linalg.inv(E), pts)
    x, y = mapped[:, 0], mapped[:, 1]
    in_bounds = (x >= 0.0) & (x <= W - 1.0) & (y >= 0.0) & (y <= H - 1.0)
    taps, valid_taps, w = _bilinear_sample(frame, x, y, source_valid)
    warped = (taps * w).sum(axis=1)  # [N,3]
    all_taps_valid = valid_taps.all(axis=1) & in_bounds
    out = np.zeros((H, W, 3), np.float64)
    out_v = np.zeros((H, W), bool)
    finite = np.isfinite(warped).all(axis=1)
    npix = pts.shape[0]
    out.reshape(-1, 3)[:] = np.where(finite[:, None], warped, 0.0)
    out_v.reshape(-1)[:] = all_taps_valid & finite
    vals = out.reshape(-1, 3)[out_v.reshape(-1)]
    var = float(np.var(vals)) if vals.size else None
    return {
        "image": out,
        "valid": out_v,
        "in_bounds": in_bounds.reshape(H, W),
        "in_bounds_count": int(in_bounds.sum()),
        "valid_count": int(out_v.sum()),
        "total_pixels": int(npix),
        "finite_count": int(finite.sum()),
        "finite_frac": float(finite.mean()) if npix else None,
        "rgb_variance": var,
    }


# --------------------------------------------------------------------------------------
# fit blocks and policy rendering
# --------------------------------------------------------------------------------------

def _summarize_coverage(per_frame: list) -> dict:
    """Aggregate per-frame coverage fractions into finite min/mean/max plus pixel counts."""
    def _fracs(key):
        return [f[key] for f in per_frame if f.get(key) is not None]
    def _agg(vals):
        return {"min": float(min(vals)), "mean": float(np.mean(vals)), "max": float(max(vals))} \
            if vals else {"min": None, "mean": None, "max": None}
    ib = _fracs("in_bounds_frac")
    rv = _fracs("rendered_valid_frac")
    fr = _fracs("finite_frac")
    return {
        "in_bounds_frac": _agg(ib),
        "rendered_valid_frac": _agg(rv),
        "finite_rgb_frac": _agg(fr),
        "total_out_pixels": int(sum(int(f.get("total_pixels", 0)) for f in per_frame)),
        "total_in_bounds": int(sum(int(f.get("in_bounds_count", 0)) for f in per_frame)),
        "total_rendered_valid": int(sum(int(f.get("valid_count", 0)) for f in per_frame)),
    }


def _render_policy(seq: DLS.Sequence, textures: np.ndarray, E_by_frame: np.ndarray) -> dict:
    """Render each frame, pull-warp it with ``E_by_frame[t]`` and aggregate rendered-validity statistics.

    Per frame the raw composited frame is rendered once from the known layer transforms (independent of the
    policy) and only the *warp* uses the policy's estimated ``E[t]``. Per-frame and aggregate in-bounds
    coverage, actual rendered-valid coverage and finite RGB fractions are reported; the actual rendered-valid
    count is always ``<=`` the in-bounds count because the conservative mask is a subset of the in-bounds set.
    """
    frames = []
    for t in range(seq.T):
        rendered = render_frame(seq, textures, t)
        warped = pull_warp(rendered["frame"], rendered["valid"], E_by_frame[t])
        total = max(warped["total_pixels"], 1)
        frames.append({
            "frame": int(t),
            "splatted_pixels": int(rendered["splatted_pixels"]),
            "splat_holes": int(total - rendered["splatted_pixels"]),
            "in_bounds_count": int(warped["in_bounds_count"]),
            "in_bounds_frac": float(warped["in_bounds_count"]) / total,
            "valid_count": int(warped["valid_count"]),
            "rendered_valid_frac": float(warped["valid_count"]) / total,
            "finite_count": int(warped["finite_count"]),
            "finite_frac": warped["finite_frac"],
            "rgb_variance": warped["rgb_variance"],
            "per_layer_splats": rendered["per_layer_splats"],
        })
    return {"frames": frames, "aggregate": _summarize_coverage(frames)}


def _policy_path(seq: DLS.Sequence, spec: SyntheticSpec, *, select_rank: Optional[int]) -> dict:
    """Fit the per-frame estimated ``E[t]`` for one policy (or the pooled baseline when ``select_rank`` None).

    Returns the fit availability, the stacked ``E`` array, per-frame fit records and a machine-readable reason.
    """
    E_by_frame = np.zeros((spec.T, 3, 3), np.float64)
    frames = []
    ok = True
    reason = None
    for t in range(spec.T):
        fit = DLS._fit_policy(seq, t, select_rank)
        frames.append({"frame": int(t), "fit_ok": bool(fit["fit_ok"]), "fit_reason": fit["reason"],
                       "train_count": int(fit["train_count"])})
        if fit["fit_ok"]:
            E_by_frame[t] = fit["E"]
        else:
            ok = False
            reason = reason or fit["reason"]
    return {"fit_ok": bool(ok), "reason": reason, "E_by_frame": E_by_frame, "frames": frames}


def _resolve_target_layer(seq: DLS.Sequence, select_rank: int) -> dict:
    """Ground-truth layer targeted by an explicit rank; unavailable when absent or shared (ambiguous)."""
    matches = np.flatnonzero(seq.layer_rank == int(select_rank))
    if matches.size == 0:
        return {"layer": None, "available": False, "reason": "rank_not_present"}
    if matches.size > 1:
        return {"layer": None, "available": False, "reason": "rank_shared_by_multiple_layers"}
    return {"layer": int(matches[0]), "available": True, "reason": None}


def _policy_block(seq: DLS.Sequence, spec: SyntheticSpec, textures: np.ndarray, *,
                  policy: str, select_rank: int) -> dict:
    """One independent policy render block.

    Under ``"shuffled"`` depth the rank/layer association is destroyed by construction, so the policy is
    reported ``available=False`` with reason ``"rank_layer_association_destroyed"`` and a ``None`` render block
    -- no frames are fabricated. Under uninformative depth the rank collapses; a rank outside the scene is
    absent. In every unavailable case no render is produced.
    """
    base = {"policy": policy, "select_rank": int(select_rank), "target_layer": None,
            "available": False, "reason": None, "select_rank_count": int(np.count_nonzero(
                seq.rank_flat == int(select_rank))), "render": None}
    if spec.depth_mode == "shuffled":
        return {**base, "reason": "rank_layer_association_destroyed"}
    res = _resolve_target_layer(seq, select_rank)
    if not res["available"]:
        return {**base, "reason": res["reason"]}
    path = _policy_path(seq, spec, select_rank=select_rank)
    if not path["fit_ok"]:
        return {**base, "target_layer": int(res["layer"]), "reason": path["reason"] or "fit_unavailable",
                "fit_frames": path["frames"]}
    render = _render_policy(seq, textures, path["E_by_frame"])
    return {**base, "target_layer": int(res["layer"]), "available": True, "reason": None,
            "fit_frames": path["frames"], "render": render}


# --------------------------------------------------------------------------------------
# experiment
# --------------------------------------------------------------------------------------

def run_audit(spec: Optional[SyntheticSpec] = None) -> dict:
    """Pure function of ``spec``: render, warp and report rendered-validity statistics.

    The pooled baseline is a single common diagnostic path (one fit on all valid train cells, rendered once).
    The two policies are separate render blocks, each driven by its own estimated ``E[t]``; nothing is pooled
    between them. When a policy is unavailable no render is fabricated and its ``render`` block is ``None``;
    the pooled baseline remains a diagnostic but supports no policy claim. No threshold is selected and no
    winner is declared.
    """
    spec = spec or SyntheticSpec()
    seq = DLS.build_sequence(spec)
    textures = build_textures(spec.W, spec.H, n_layers=seq.n_layers)

    pooled = _policy_path(seq, spec, select_rank=None)
    if pooled["fit_ok"]:
        pooled_render = _render_policy(seq, textures, pooled["E_by_frame"])
        pooled_reason = None
    else:
        pooled_render = None
        pooled_reason = pooled["reason"] or "pooled_fit_unavailable"

    policies = [
        _policy_block(seq, spec, textures, policy="background_lock", select_rank=spec.lock_rank),
        _policy_block(seq, spec, textures, policy="foreground_preserve", select_rank=spec.preserve_rank),
    ]

    return {
        "schema": SCHEMA,
        "spec": spec.as_dict(),
        "n_layers": int(seq.n_layers),
        "rasterization": RASTERIZATION,
        "collision_policy": COLLISION_POLICY,
        "textures": {
            "kind": "procedural_rgb_sinusoids",
            "shape": [int(seq.n_layers), int(spec.H), int(spec.W), 3],
            "finite": bool(np.isfinite(textures).all()),
            "variance": float(np.var(textures)),
            "pixels_in_report": False,
        },
        "baseline": {
            "scope": "pooled",
            "targets_layer": None,
            "available": bool(pooled["fit_ok"]),
            "reason": pooled_reason,
            "fit_frames": pooled["frames"],
            "render": pooled_render,
        },
        "policies": policies,
        "limitations": list(LIMITATIONS),
        "notes": {
            "synthetic_only": True,
            "no_media_decoded": True,
            "no_assets_used": True,
            "no_production_code_imported": True,
            "no_image_pixels_in_report": True,
            "winner_declared": False,
            "threshold_selected": False,
        },
    }


def dumps_report(report: dict) -> str:
    """Deterministic JSON serialization of a report, finite-safe and path-free."""
    return MF.dumps_json(report)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench/depth_layer_render_audit.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--T", type=int, default=10, help="number of frames (>= 3)")
    p.add_argument("--W", type=int, default=48, help="frame width (>= 8)")
    p.add_argument("--H", type=int, default=36, help="frame height (>= 8)")
    p.add_argument("--depths", default="0.2,0.8",
                   help="comma-separated two raw layer depth values (background,foreground)")
    p.add_argument("--rank-orientation", choices=list(DLS.RANK_ORIENTATIONS), default="low_raw_low_rank")
    p.add_argument("--depth-mode", choices=list(DLS.DEPTH_MODES), default="ranked")
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


def spec_from_args(a: argparse.Namespace) -> SyntheticSpec:
    """Build a validated :class:`SyntheticSpec` from parsed CLI arguments."""
    return SyntheticSpec(
        T=a.T, W=a.W, H=a.H, depths=DLS.parse_depths(a.depths),
        rank_orientation=a.rank_orientation, depth_mode=a.depth_mode, lock_rank=a.lock_rank,
        preserve_rank=a.preserve_rank, jitter_px=a.jitter_px, drift_px=a.drift_px, rot_deg=a.rot_deg,
        zoom=a.zoom, fg_amp=a.fg_amp, border=a.border, occlusion_hole=not a.no_occlusion_hole, seed=a.seed,
        flow_noise_px=a.flow_noise_px, depth_error_frac=a.depth_error_frac,
        fg_nonaffine_px=a.fg_nonaffine_px, fg_rect=DLS.parse_fg_rect(a.fg_rect))


def run(opts: argparse.Namespace, log=print) -> dict:
    """Build the spec, run the audit and optionally write the atomic JSON report."""
    spec = spec_from_args(opts)
    report = run_audit(spec)
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
