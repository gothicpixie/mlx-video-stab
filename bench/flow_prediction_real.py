"""E04 real-footage flow-initialization benchmark (isolated; NOT wired into production, no default change).

This is the second, **real-footage** E04 gate. The first gate (``bench/flow_prediction.py``) isolated the
prediction mathematics on exact *analytic* pair maps. This module replays the same prediction mathematics on
**real consecutive frame pairs** decoded from a local video (or a tiny in-memory synthetic clip) and measures
the initialization candidates against a documented high-iteration cold reference.

What is compared
----------------
For each pair of consecutive frames and each direction (forward ``k -> k+1`` and, where practical, backward
``k+1 -> k``) the following initializations are evaluated with the *same* RAFT network, the *same* cached
per-frame features, the *same* refinement iteration count and the *same* decoded/preprocessed frames:

    cold                  zero initialization
    same_grid             the previous pair's low-resolution flow reused at the same grid indices
                          (this is exactly what ``stabilize.RaftFlow`` does today via ``prev_f``/``prev_b``)
    transported           the previous pair's low-resolution flow pulled back onto this pair's grid through
                          the approximate inverse map (order 0)
    constant_velocity     2 g0 - g1                      (order 1; g = transported history)
    constant_acceleration 3 g0 - 3 g1 + g2               (order 2)

The candidate fields, their validity masks and the fallback policy reuse ``bench.flow_prediction.predict`` /
``transport`` (the same code that the analytic gate tests). The **only** changed assumption is the pair map:
on analytic fixtures it was an exact map; here the map is RAFT's own forward/backward flow, sampled
bilinearly, which is an *approximate* inverse with **no occlusion/disocclusion truth**. Validity is therefore
only "the mapped point stayed inside the grid and is finite".

Reference and what the error means
----------------------------------
The reference for every pair/direction is a **cold, high-iteration RAFT run** (default 24 iterations) computed
by the same runner from the same cached features. Reported errors are the Euclidean endpoint error (EPE) of
a candidate's *refined* low-resolution flow against that reference, in **feature-grid units** (see "grid"
below), plus the EPE of the initialization itself before refinement. This is a *distance to a model-derived
reference*, **not** true optical-flow error: the reference shares RAFT's systematic errors and the candidates
share its weights, so this experiment answers "does a predicted initialization land closer to the converged
cold solution at a fixed iteration count?" and nothing more.

Reference per-pair EPE is never reported as ground truth and the reference is never called "the true flow" in
the manifest.

Pair maps are reference-derived (offline assumption)
----------------------------------------------------
Every transport ``Pair`` map (both directions) is built from the **full-window cold high-iteration reference
flows** (``--ref-iters``), not from the candidate's own low-iteration outputs. This is an **offline,
reference-derived-map assumption**: in a real pipeline the map the candidate transports through would have to
come from the candidate's own run, whereas here it is exactly the expensive converged flow the benchmark is
trying to avoid. The candidate never sees or consumes its own low-iteration map outputs. The measured transport
behaviour therefore bounds an idealized offline replay, not an online/production implementation, and nothing
here implies online or production readiness.

Fairness
--------
The window is decoded once and every pair (and the reference) is replayed from the same in-memory
``uint8`` frames, so no candidate sees a different decode. Per-frame network features are computed once and
shared by the reference and every candidate; only the initialization differs. Candidate initialization cost
(NumPy transport) and GPU refinement wall time are recorded **separately**, per pair.

Directions
----------
The backward direction is a separate traversal of the same pairs in reverse temporal order with every map
inverted (matching ``bench.flow_prediction.reversed_fixture``): traversal pair ``t`` is original pair
``i = n_pairs-1-t`` reversed, so its "previous" field is genuinely the next original pair. This is the
explicit backward handling the plan asks for and is not the same as production's ``prev_b`` relabel.

Grid
----
All fields live on the network's **feature grid** (padded image / 8). The runner records ``feature_h`` /
``feature_w``, the replicate pad and the conversion ``units_to_source_px = 8 * flow_downscale``. EPE is in
feature-grid units; ``*_source_px`` fields multiply it by that factor. The replicate-padded border rows/cols
are included in the grid and in the aggregates; they are a small fraction of the frame.

Budget scan (opt-in offline oracle)
-----------------------------------
``--budget-scan`` adds an offline per-iteration scan on top of the same replay. For every method and direction
each target pair still starts from the initialization the normal policy would produce, and history is still
built by the normal fixed ``--iters`` replay; but the pair is then refined one iteration at a time up to
``--budget-max-iters`` (default 24) and the mean feature-grid EPE against that pair's cold reference is
recorded after every iteration. For each ``--budget-thresholds`` value (default 0.005,0.01,0.02 feature-grid
px) the manifest records the first iteration whose EPE crosses the threshold, or that it was censored (not
reached by the max). **History invariant:** the flow written for the following pair's initialization is
always the state after exactly ``--iters`` iterations, so scan iterations beyond ``--iters`` never leak into
later initializations; ``--budget-max-iters`` must be >= ``--iters``. Per-iteration device refinement time
(the stepper evaluates before returning) is recorded separately from host EPE/threshold check time. For each
threshold the summary also reports **prefix-summed measured costs**: every usable pair consumes its first
crossing iteration when reached, otherwise its number of measured scan steps (normally ``--budget-max-iters``,
less only if output went non-finite early), and the per-step ``update_s``/``epe_check_s`` from those measured
steps are summed -- so censored pairs still contribute their work. These prefixes are computed from the single
full trace (which deliberately runs to the scan end to measure the whole curve), not from separate
early-stopped executions; the top-level ``full_scan_*`` totals cover that full trace. This is
an **offline oracle scan** -- it is not a deployable stopping rule and it does not model an online pipeline or
prove a speedup.

Candidate-observable update magnitude (diagnostic only)
-------------------------------------------------------
Each per-iteration row also records ``flow_update_low``: the mean/p95/max per-pixel magnitude
``|low - previous_low|`` in feature-grid px, where ``previous_low`` is the *effective initialization* on the
first iteration and the previous step's flow afterwards. It is computed **solely from the candidate's own
initialization/current/previous prediction** -- the cold reference never participates -- so it is a
*candidate-observable* signal, deliberately distinct from the oracle ``epe_low``. Its host computation time is
measured separately as ``update_mag_check_s`` (never mixed into ``update_s`` or ``epe_check_s``), and the
summary carries prefix-summed ``signal_check_s_*`` and full-trace ``full_scan_signal_check_s_*`` timings.
These diagnostics **could** support a future convergence-based stopping rule, but **no stop rule is
implemented or selected**: the threshold scan remains an offline oracle comparison to the cold reference.

Candidate-observable photometric consistency (diagnostic only)
--------------------------------------------------------------
When ``--budget-scan`` is enabled every per-iteration row *also* records ``photometric_low``: robust absolute
luminance-residual statistics after warping the **source** preprocessed frame toward the **target**
preprocessed frame at the candidate's current low-resolution flow. It uses only the two preprocessed frames
and the candidate flow -- the cold reference flow never participates -- so it is a candidate-observable signal
distinct from both ``epe_low`` and ``flow_update_low``.

Geometry. The candidate flow lives on the network feature grid (padded image / 8). Feature cell ``(fx, fy)``
corresponds to preprocessed (integer box-downscaled) pixel ``(8*fx - pad_left, 8*fy - pad_top)`` -- the same
``8 * downscale`` feature-to-source scaling the EPE uses (``units_to_source_px``), with the replicate-pad
offset removed so only feature cells whose centre lands on real, unpadded preprocessed content are used. The
warped target location is the source location plus ``8 * flow``. This adopts RAFT's own integer
feature-coordinate convention (the coordinate grid plus the x8 convex upsampling). RAFT's convex upsampling
actually spreads a feature cell's flow over its 8x8 output block with a learned 3x3 neighbour mixing; this
diagnostic deliberately uses the simpler convention (cell -> top-left preprocessed pixel of the block,
displacement ``8 * flow``) and does not reconstruct that sub-cell weighting, so a small alignment residual
floor is possible and is recorded as a limitation rather than silently corrected.

Robustness. Luminance is Rec.601 (``0.299 R + 0.587 G + 0.114 B``) of the same box downscale
``RaftRunner._prep_pad`` applies, so the diagnostic sees exactly the preprocessed content. The residual is
``|(I_target(p + 8f) - offset) - I_source(p)|`` where ``offset`` is the per-pair difference of frame
luminance medians, which removes a global exposure/brightness shift. Samples whose source cell lies in the
replicate pad, whose target location is out of the preprocessed frame, or whose flow/value is non-finite are
excluded (``valid_frac`` reports the retained fraction). Because flat regions carry no photometric signal (the
aperture problem), ``resid_textured`` restricts the same statistics to source cells whose central luminance
gradient exceeds ``PHOTOMETRIC_TEXTURE_MIN`` and ``texture_frac`` reports how many samples those are; a low
``texture_frac`` or ``valid_frac`` means the signal is only weakly observed on that pair. A global brightness
offset is removed, but brightness *changes* beyond a global offset, rolling shutter, motion blur, compression
and non-Lambertian content can still dominate the residual, so this remains a diagnostic and not a stopping
rule. Its host computation cost is measured separately as ``photometric_check_s`` and never mixed into
``update_s`` (device), ``epe_check_s`` (oracle EPE) or ``update_mag_check_s`` (the other candidate-observable
signal); the summary adds prefix-summed ``photometric_check_s_*`` and full-trace
``full_scan_photometric_check_s_*`` totals plus ``total_with_observables_s_*``.

Candidate-observable forward-backward cycle consistency (opt-in pilot; diagnostic only)
----------------------------------------------------------------------------------------
``--cycle-scan`` (with a bounded ``--cycle-max-iters``, default 24) is a separate, **cold-only bidirectional**
pilot. For each pair it starts the forward (``k -> k+1``) and backward (``k+1 -> k``) candidates from the
cold zero initialization and refines them **together, one iteration at a time**, through the production
batch-2 path (mirroring ``RAFT.flow_bidir``). After every iteration the two **same-iteration, same-pair,
same-model** fields are used to compute a forward-backward cycle residual in feature-grid pixels:
``|fwd(p) + bwd(p + fwd(p))|`` and ``|bwd(q) + fwd(q + bwd(q))|``, with an in-bounds/finite validity filter
and robust ``mean``/``median``/``p95``/``max`` plus ``valid_frac`` (see :func:`cycle_residual`). The residual
uses only the two candidate fields, the feature-grid geometry and optional validity masks -- **never** a cold
reference field or EPE. Host cycle-check cost is measured separately as ``cycle_check_s`` from the device
``update_s`` and the oracle ``epe_check_s``.

This signal is a **cycle-consistency / visibility diagnostic only**: cycle errors can also mean wrong flow,
blur or independent motion; consistency does **not** prove visibility or correspondence correctness; and a
low cycle residual is **not** true-flow accuracy. The cold high-iteration references are used solely for the
separately reported oracle EPE, which is post-hoc evaluation and never enters the cycle statistics. No
threshold is selected, no stop rule is implemented, and the run is isolated from production (no ``mlxstab``
import, no default change, no ``RaftFlow`` wiring).

Scope and limits
----------------
Isolated experiment only. It does not import or modify ``stabilize``, does not change ``Config`` or any
default, does not wire prediction into ``RaftFlow``, and makes no speedup or quality claim for the
stabilizer. Outputs are local-only under the ignored ``out/``; private footage never becomes a tracked
artifact. See ``LIMITATIONS`` below and the ``scope``/``limitations`` fields of every manifest.

Run
---
    # tiny public synthetic smoke (no private footage; heavy lock off)
    python -m bench.flow_prediction_real --source synthetic --synth-frames 6 --synth-w 96 --synth-h 64 \
        --out-dir out/bench/flow_prediction_real/smoke

    # local real footage window (footage stays local; use the shared heavy lock)
    python -m bench.flow_prediction_real --source /path/to/local/clip.MTS --start 60 --duration 0.25 \
        --downscale 2 --iters 12 --ref-iters 24 --lock on \
        --out-dir out/bench/flow_prediction_real/t060

    # longer real window with shared-lock chunking: one monotonic step per completed frame/pair across
    # feature preparation, reference generation and every candidate replay; release/re-acquire every 30
    # steps and pause 12 s for waiting jobs (the shared-project etiquette)
    python -m bench.flow_prediction_real --source /path/to/local/clip.MTS --start 800 --duration 1 \
        --downscale 2 --iters 12 --ref-iters 24 --lock on --lock-chunk 30 --lock-pause 12 \
        --cache-frames out/bench/flow_prediction_real/t800_frames.npz \
        --out-dir out/bench/flow_prediction_real/t800_iters12

    # opt-in cold-only bidirectional cycle-consistency pilot (cannot be combined with --budget-scan): both
    # directions start cold, are advanced together one iteration at a time to --cycle-max-iters, and the
    # per-iteration cycle residual is the candidate-observable signal while oracle EPE stays separate
    python -m bench.flow_prediction_real --source /path/to/local/clip.MTS --start 63 --duration 1 \
        --downscale 2 --ref-iters 24 --cycle-scan --cycle-max-iters 24 --lock on --lock-chunk 30 \
        --lock-pause 12 --cache-frames out/bench/flow_prediction_real/t063_photo_frames.npz \
        --out-dir out/bench/flow_prediction_real/t063_cycle_scan_iters24
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from . import flow_prediction as FP
from . import manifest as MF
from .heavylock import HeavyLock

SCHEMA = "mlxstab.flow_prediction_real/1"
FRAME_CACHE_SCHEMA = "mlxstab.flow_prediction_real.frames/1"
METHODS = tuple(FP.METHODS)
DIRECTIONS = ("forward", "backward")
DEFAULT_ITERS = 12
DEFAULT_REF_ITERS = 24
DEFAULT_BUDGET_MAX_ITERS = 24
DEFAULT_BUDGET_THRESHOLDS = (0.005, 0.01, 0.02)
DEFAULT_CYCLE_MAX_ITERS = 24
# Central luminance-gradient magnitude (0..255 levels) below which a source cell is treated as textureless for
# the candidate-observable photometric residual's ``resid_textured`` statistic. Not a user knob.
PHOTOMETRIC_TEXTURE_MIN = 2.0
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "out" / "bench" / "flow_prediction_real"
DEFAULT_WEIGHTS = ROOT / "vendor" / "weights" / "conv" / "raft-things.safetensors"

SCOPE = ("isolated E04 real-footage flow-initialization benchmark: same decoded/preprocessed real frame pairs, "
         "cold vs same-grid reuse vs transported vs constant-velocity vs constant-acceleration initializations, "
         "measured against a documented high-iteration cold RAFT reference; no production wiring, no default "
         "change, no optical-flow ground-truth claim")

LIMITATIONS = (
    "The reference is a cold high-iteration RAFT output with the same weights and features as the candidates, "
    "not ground truth. Reported EPE is a distance to that reference and cannot establish absolute flow accuracy.",
    "Transport uses RAFT's own forward/backward flows as approximate inverse maps sampled bilinearly. RAFT "
    "does not provide occlusion/disocclusion validity, so map validity is only in-bounds and finite; the "
    "analytic gate's exact-map behaviour does not carry over unchanged.",
    "Every transport Pair map is built from the full-window cold high-iteration reference flows (ref_iters), "
    "not from the candidate's own low-iteration outputs; this is an offline, reference-derived-map assumption. "
    "The candidate never consumes its own map outputs, so these numbers do not model an online/production "
    "pipeline and imply no online or production readiness.",
    "Field grids are the network feature grid (padded image / 8, after integer box downscale); EPE units are "
    "feature-grid pixels. Multiply by units_to_source_px = 8 * flow_downscale for source-pixel units. "
    "Replicate-padded border rows/cols are included.",
    "Candidate refinement runs one direction at a time (batch 1) while the reference runs the production "
    "batch-2 path; per-pair wall times are therefore compared within candidates, and the reference timing is "
    "reported separately.",
    "Only the refinement and initialization are timed. Frame load/decode/cache restore is reported separately "
    "as timing_s['frames_load'], which excludes the shared-lock acquire wait (recorded as "
    "lock.acquire_wait_s); feature extraction and evaluation are reported separately. None is part of an "
    "end-to-end throughput claim.",
    "With --lock-chunk, one monotonic step counter is shared by feature preparation, reference generation and "
    "candidate replay; the lock is released/re-acquired at every chunk-th completed frame/pair boundary and "
    "the yielded time is reported in lock.pause_total_s. Chunking changes throughput timing only and not the "
    "decoded frames, cached features, references or reported errors.",
    "A single inferred/selected window is not a clip-wide, held-out, perceptual or multi-scene result.",
    "flow_update_low is a candidate-observable per-iteration update magnitude and nothing more: it could "
    "support a future convergence-based stopping rule, but no stop rule is implemented or selected, and the "
    "budget scan still compares against a cold reference offline.",
    "photometric_low is a candidate-observable residual after warping the source preprocessed frame toward the "
    "target at the candidate's own flow; it uses no reference flow, but a global brightness offset is only "
    "removed heuristically and non-global exposure change, blur, compression and flat/low-texture regions can "
    "dominate it, so it is a diagnostic and not a validated stopping signal.",
    "photometric_low maps each candidate feature cell to the top-left preprocessed pixel of its 8x8 upsampled "
    "block and applies 8*flow directly; it does not reconstruct RAFT's learned convex-upsampling sub-cell "
    "weighting, so an alignment-dependent residual floor is possible even at the true flow.",
    "cycle_low is a candidate-observable forward-backward cycle-consistency/visibility diagnostic computed "
    "from the two same-iteration candidate fields only. A large cycle residual can also mean wrong flow, "
    "motion blur or independent motion rather than occlusion; a small cycle residual does not prove the "
    "correspondence is visible/correct and is not true-flow accuracy. No threshold or stop rule is selected.",
    "The --cycle-scan pilot is cold-only and bidirectional: both candidate directions start at zero and are "
    "advanced together, and no flow from another pair participates, so it is an independent diagnostic run "
    "rather than a warm-start or production policy. Its oracle EPE is the same model-derived cold-reference "
    "distance as the rest of E04, not optical-flow ground truth.",
)


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------

def frames_sha256(frames: Sequence[np.ndarray]) -> str:
    """SHA-256 over the exact decoded/preprocessed frame bytes replayed by every candidate."""
    h = hashlib.sha256()
    for f in frames:
        h.update(np.ascontiguousarray(f).tobytes())
    return h.hexdigest()


def load_under_lock(lock, load: Callable, clock: Callable[[], float] = time.perf_counter):
    """Acquire ``lock``, then time only the load/decode that follows -- never the acquire wait.

    The shared heavy lock can block for an unbounded time behind another job. That wait is accounted by
    ``HeavyLock.acquire_wait_s`` and recorded in the manifest's ``lock`` block, so it must not inflate the
    frame load/decode figure. The returned duration covers only the ``load()`` call and is accurate whether
    frames are freshly decoded or restored from a cache. ``load`` may return anything; this returns
    ``(result, seconds)``.
    """
    lock.acquire()
    t0 = clock()
    result = load()
    return result, clock() - t0


def _stats(values: np.ndarray) -> dict:
    v = np.asarray(values, np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "mean": None, "p95": None, "max": None}
    return {"n": int(v.size), "mean": float(v.mean()), "p95": float(np.percentile(v, 95)), "max": float(v.max())}


# --------------------------------------------------------------------------------------
# candidate-observable photometric consistency (host-side; no reference flow, no RAFT)
# --------------------------------------------------------------------------------------

def preprocess_luminance(frame: np.ndarray, downscale: int) -> np.ndarray:
    """Rec.601 luminance of the exact integer box-downscaled frame the network saw.

    The block mean (``[H2,d,W2,d,3].mean(axis=(1,3))``) is the same downscale ``RaftRunner._prep_pad`` applies
    before the network's 8x feature encoder, so the diagnostic works on the true preprocessed content.
    Rec.601 (``0.299 R + 0.587 G + 0.114 B``) is a fixed, documented luminance choice; residual units are
    0..255 luminance levels. Returns a ``[H2,W2]`` ``float32`` array.
    """
    x = np.asarray(frame, np.float32)
    d = int(downscale)
    if d > 1:
        H, W = x.shape[:2]
        H2, W2 = H // d, W // d
        x = x[:H2 * d, :W2 * d].reshape(H2, d, W2, d, 3).mean(axis=(1, 3))
    lum = (np.float32(0.299) * x[..., 0] + np.float32(0.587) * x[..., 1]
           + np.float32(0.114) * x[..., 2])
    return lum.astype(np.float32)


def _bilinear_luminance(img: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    """Bilinear sample of a 2-D luminance image; returns ``(values, valid)``.

    ``valid`` uses the same pixel-edge in-bounds test as ``flow_prediction.in_bounds`` (``[-0.5, W-0.5]``)
    and additionally requires finite coordinates and a finite result. Tap indices are clamped to the border,
    but a sample whose *centre* is out of frame is marked invalid rather than silently clamped, so an
    out-of-bounds warp is excluded instead of scoring a clamped value.
    """
    img = np.asarray(img, np.float64)
    H, W = img.shape
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    finite = np.isfinite(xs) & np.isfinite(ys)
    x0 = np.floor(np.where(finite, xs, 0.0)).astype(np.int64)
    y0 = np.floor(np.where(finite, ys, 0.0)).astype(np.int64)
    fx = np.where(finite, xs - x0, 0.0)
    fy = np.where(finite, ys - y0, 0.0)
    x0c = np.clip(x0, 0, W - 1); x1c = np.clip(x0 + 1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1); y1c = np.clip(y0 + 1, 0, H - 1)
    vals = (img[y0c, x0c] * (1 - fy) * (1 - fx) + img[y0c, x1c] * (1 - fy) * fx
            + img[y1c, x0c] * fy * (1 - fx) + img[y1c, x1c] * fy * fx)
    inb = finite & (xs >= -0.5) & (xs <= W - 0.5) & (ys >= -0.5) & (ys <= H - 0.5)
    return vals, inb & np.isfinite(vals)


def _resid_stats(values: np.ndarray) -> dict:
    """Finite photometric-residual statistics: count, mean, median, p95 and max (``None`` when empty)."""
    v = np.asarray(values, np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def photometric_residual(L_src: np.ndarray, L_dst: np.ndarray, luma_offset: float,
                         flow_flat: np.ndarray, pad: Sequence[int], h: int, w: int,
                         texture_min: float = PHOTOMETRIC_TEXTURE_MIN) -> dict:
    """Robust photometric residual of warping ``L_src`` toward ``L_dst`` at a feature-grid flow.

    ``L_src``/``L_dst`` are preprocessed luminance images ``[H2,W2]`` (see :func:`preprocess_luminance`).
    ``pad`` is the replicate pad ``(top, bottom, left, right)`` ``RaftRunner._prep_pad`` applied and
    ``flow_flat`` is the candidate ``[h*w,2]`` feature-grid flow (x, y). Feature cell ``(fx, fy)`` sits at
    preprocessed coordinate ``(8*fx - left, 8*fy - top)`` and warps to ``source + 8*flow`` (RAFT's integer
    feature-coordinate convention). The residual is ``|(I_target(warp) - luma_offset) - I_source|`` where
    ``luma_offset`` removes a global exposure/brightness shift (the caller passes the frame-median difference).

    Returns a JSON-safe dict: ``valid_frac`` (retained fraction of all feature cells), ``texture_frac``
    (textured fraction among retained samples), the applied ``luma_offset``, and ``resid``/``resid_textured``
    (:func:`_resid_stats` over all retained samples / only samples whose central source-luminance gradient
    exceeds ``texture_min``). Flat/low-texture regions carry little photometric signal, so ``resid_textured``
    is the more informative statistic; non-global brightness change, blur, rolling shutter and compression can
    still dominate the residual.
    """
    L_src = np.asarray(L_src, np.float64)
    L_dst = np.asarray(L_dst, np.float64)
    H2, W2 = L_src.shape
    top = int(pad[0]); left = int(pad[2])          # only the near-edge offsets shift cells onto real content
    f = np.asarray(flow_flat, np.float64).reshape(int(h), int(w), 2)
    fxs, fys = np.meshgrid(np.arange(int(w)), np.arange(int(h)), indexing="xy")
    sx = 8.0 * fxs - float(left)
    sy = 8.0 * fys - float(top)
    src_in = (sx >= 0.0) & (sx <= W2 - 1.0) & (sy >= 0.0) & (sy <= H2 - 1.0)
    finite_flow = np.isfinite(f).all(axis=-1)
    src_vals, src_ok = _bilinear_luminance(L_src, sx, sy)
    dst_vals, dst_ok = _bilinear_luminance(L_dst, sx + 8.0 * f[..., 0], sy + 8.0 * f[..., 1])
    valid = src_in & finite_flow & src_ok & dst_ok
    resid = np.abs((dst_vals - float(luma_offset)) - src_vals)
    # Central luminance gradient at the source cell, for the texture mask (indices clamped at the border).
    sxi = np.clip(np.rint(sx), 0, W2 - 1).astype(np.int64)
    syi = np.clip(np.rint(sy), 0, H2 - 1).astype(np.int64)
    gx = L_src[syi, np.clip(sxi + 1, 0, W2 - 1)] - L_src[syi, np.clip(sxi - 1, 0, W2 - 1)]
    gy = L_src[np.clip(syi + 1, 0, H2 - 1), sxi] - L_src[np.clip(syi - 1, 0, H2 - 1), sxi]
    textured = valid & (np.sqrt(gx * gx + gy * gy) > float(texture_min))
    n_valid = int(valid.sum())
    return {
        "valid_frac": float(valid.mean()) if valid.size else 0.0,
        "texture_frac": (float(textured.sum()) / n_valid) if n_valid else 0.0,
        "luma_offset": float(luma_offset),
        "resid": _resid_stats(resid[valid]),
        "resid_textured": _resid_stats(resid[textured]),
    }


# ---------------------------------------------------------------------------------
# candidate-observable forward-backward cycle consistency (host-side; no reference flow, no RAFT)
# ---------------------------------------------------------------------------------

def cycle_residual_fields(fwd_flat: np.ndarray, bwd_flat: np.ndarray, h: int, w: int,
                          fwd_valid: Optional[np.ndarray] = None,
                          bwd_valid: Optional[np.ndarray] = None) -> dict:
    """Per-cell forward/backward cycle residual fields: the shared pure-NumPy core of :func:`cycle_residual`.

    ``fwd_flat`` maps feature cell ``p`` to ``p + fwd(p)`` (frame ``k -> k+1``) and ``bwd_flat`` maps ``q`` to
    ``q + bwd(q)`` (frame ``k+1 -> k``), both ``[h*w,2]`` in feature-grid pixels ``(x, y)``. The round-trip
    residual magnitude is ``|fwd(p) + bwd(p + fwd(p))|`` on the frame-``k`` grid and ``|bwd(q) + fwd(q +
    bwd(q))|`` on the frame-``k+1`` grid. ``fwd_valid``/``bwd_valid`` are optional per-cell boolean masks; a
    cell is excluded when its own mask is False and a sampled mask tap is False. A sample is also excluded when
    either field value is non-finite, the mapped point leaves the feature grid (edge rule ``[-0.5, W-0.5]``,
    the same as :func:`flow_prediction.in_bounds`), or the sampled bilinear value is non-finite.

    This returns the *unaggregated* spatial fields so a caller can inspect where the residual lives (the
    aggregated, JSON-safe statistics stay in :func:`cycle_residual`). The dict has four ``[h, w]`` float64/
    bool arrays: ``fwd_resid``/``fwd_valid`` on the frame-``k`` source grid and ``bwd_resid``/``bwd_valid`` on
    the frame-``k+1`` source grid. These arrays are diagnostics only: they are never written to a manifest or
    used by the replay/production paths. Only the two same-pair candidate fields, the grid geometry and the
    optional masks participate; no reference field, EPE or cold flow is involved.
    """
    f = np.asarray(fwd_flat, np.float64).reshape(int(h), int(w), 2)
    b = np.asarray(bwd_flat, np.float64).reshape(int(h), int(w), 2)
    hh, ww = int(h), int(w)
    n = hh * ww
    grid = FP.source_grid(ww, hh)

    def compose(first, second, first_valid, second_valid):
        first2 = np.asarray(first, np.float64).reshape(n, 2)
        second2 = np.asarray(second, np.float64).reshape(hh, ww, 2)
        finite = np.isfinite(first2).all(axis=1)
        # Keep invalid vectors out of coordinate helpers: sample_field floors coordinates before applying
        # its finite mask, which would otherwise emit an invalid-cast warning for NaN/Inf. The sanitized
        # sample is discarded by ``finite`` below, so it cannot affect the reported residuals.
        mapped = grid + np.where(finite[:, None], first2, 0.0)
        inb = finite & FP.in_bounds(mapped, ww, hh)  # out-of-bounds mapped points are excluded, not clamped
        src_ok = np.ones(n, bool)
        if first_valid is not None:
            src_ok = np.asarray(first_valid, bool).reshape(n).copy()
        sampled = FP.sample_field(second2, mapped, ww, hh)
        tap_ok = np.ones(n, bool)
        if second_valid is not None:
            tap_ok = FP.sample_mask_conservative(np.asarray(second_valid, bool).reshape(hh, ww),
                                                 mapped, ww, hh)
        resid = np.linalg.norm(first2 + sampled, axis=1)
        valid = finite & inb & src_ok & tap_ok & np.isfinite(sampled).all(axis=1) & np.isfinite(resid)
        return resid.reshape(hh, ww), valid.reshape(hh, ww)

    resid_f, valid_f = compose(f, b, fwd_valid, bwd_valid)
    resid_b, valid_b = compose(b, f, bwd_valid, fwd_valid)
    return {"fwd_resid": resid_f, "fwd_valid": valid_f,
            "bwd_resid": resid_b, "bwd_valid": valid_b}


def cycle_residual(fwd_flat: np.ndarray, bwd_flat: np.ndarray, h: int, w: int,
                   fwd_valid: Optional[np.ndarray] = None,
                   bwd_valid: Optional[np.ndarray] = None) -> dict:
    """Forward-backward cycle residual between two **same-pair** candidate fields on the feature grid.

    ``fwd_flat`` maps feature cell ``p`` to ``p + fwd(p)`` (frame ``k -> k+1``) and ``bwd_flat`` maps ``q`` to
    ``q + bwd(q)`` (frame ``k+1 -> k``), both ``[h*w,2]`` in feature-grid pixels ``(x, y)``. For every cell the
    round-trip residual is the magnitude of the composition::

        forward  : |fwd(p) + bwd(p + fwd(p))|
        backward : |bwd(q) + fwd(q + bwd(q))|

    ``fwd_valid``/``bwd_valid`` are optional per-cell boolean masks; a cell is excluded when its own mask is
    False and a sampled mask tap is False. A sample is also excluded when either field value is non-finite, the
    mapped point leaves the feature grid (edge rule ``[-0.5, W-0.5]``, the same as
    :func:`flow_prediction.in_bounds`), or the sampled bilinear value is non-finite.

    Only the two candidate fields, the feature-grid geometry ``(h, w)`` and the optional masks participate.
    Reference fields, EPE and the cold high-iteration flow never enter this calculation, so the result is a
    candidate-observable forward-backward **cycle-consistency / visibility** diagnostic. It is **not** a
    true-flow error: an inconsistent cycle can also mean wrong flow, blur or independent motion; a consistent
    cycle does not prove the correspondence is visible/correct; and a low cycle residual is **not** evidence of
    true-flow accuracy. Returns a JSON-safe dict with ``valid_frac`` (retained fraction of the ``2*h*w``
    directional checks) and pooled/per-direction ``n``/``mean``/``median``/``p95``/``max`` residual statistics
    in feature-grid pixels. Non-finite vectors and out-of-bounds mapped points are handled explicitly.
    :func:`cycle_residual_fields` performs the per-cell calculation; the per-cell arrays are aggregated here
    and never enter the returned JSON.
    """
    fields = cycle_residual_fields(fwd_flat, bwd_flat, h, w, fwd_valid, bwd_valid)
    resid_f = fields["fwd_resid"].reshape(-1)
    valid_f = fields["fwd_valid"].reshape(-1)
    resid_b = fields["bwd_resid"].reshape(-1)
    valid_b = fields["bwd_valid"].reshape(-1)
    pooled = np.concatenate([resid_f[valid_f], resid_b[valid_b]])
    denom = 2 * int(h) * int(w)
    return {
        "valid_frac": float((int(valid_f.sum()) + int(valid_b.sum())) / denom) if denom else 0.0,
        "resid": _resid_stats(pooled),
        "fwd": _resid_stats(resid_f[valid_f]),
        "bwd": _resid_stats(resid_b[valid_b]),
        "units": "feature-grid px",
        "note": ("candidate-observable forward-backward cycle-consistency/visibility diagnostic; uses only the "
                 "two same-iteration candidate fields, grid geometry and validity, never the cold reference or "
                 "EPE; consistency does not prove visibility and low residual is not true-flow accuracy"),
    }


class PhotometricWarper:
    """Candidate-observable photometric consistency for the budget scan (no reference flow).

    Precomputes each frame's Rec.601 luminance once with the same integer box downscale ``RaftRunner`` uses
    (plus a per-frame median for the global brightness offset), so the per-iteration ``residual`` only does
    the warp and statistics and its measured host cost reflects that work. ``closure(order)`` binds the
    traversal's ``(src, dst)`` frame order so the scan calls ``photo(target, flow_flat)``. The cold reference
    flow and the oracle EPE never enter this class.
    """

    def __init__(self, frames: Sequence[np.ndarray], downscale: int, pad: Sequence[int], h: int, w: int,
                 texture_min: float = PHOTOMETRIC_TEXTURE_MIN):
        self.downscale = int(downscale)
        self.pad = tuple(int(v) for v in pad)
        self.h, self.w = int(h), int(w)
        self.texture_min = float(texture_min)
        self.lum = [preprocess_luminance(f, self.downscale) for f in frames]
        self.medians = [float(np.median(L)) for L in self.lum]

    def residual(self, src: int, dst: int, flow_flat: np.ndarray) -> dict:
        return photometric_residual(self.lum[src], self.lum[dst],
                                    self.medians[dst] - self.medians[src],
                                    flow_flat, self.pad, self.h, self.w, self.texture_min)

    def closure(self, order: Sequence[tuple]):
        def photo(target: int, flow_flat: np.ndarray) -> dict:
            s, d = order[int(target)]
            return self.residual(int(s), int(d), flow_flat)
        return photo


class StepCounter:
    """One shared monotonic step counter for every phase of a run.

    The harness calls :meth:`tick` exactly once at each completed frame/pair boundary: feature encoding always
    ticks once per frame, reference generation once per pair, and every candidate replay once per method/pair.
    The counter is a single object threaded through all three phases so a ``--lock-chunk`` boundary is hit on
    the same monotonic sequence no matter which phase produced the step. ``on_step`` (typically
    ``HeavyLock.on_step``) receives the new step; chunk size and pause duration live in that callback, so the
    harness never needs to know whether a shared lock is held.
    """

    def __init__(self, on_step: Optional[Callable[[int], None]] = None):
        self.on_step = on_step
        self.step = 0
        self.calls = 0

    def tick(self) -> int:
        """Advance the shared counter by one and hand the new step to the callback (if any)."""
        self.step += 1
        self.calls += 1
        if self.on_step is not None:
            self.on_step(self.step)
        return self.step


def lock_enabled(source: str, lock: str) -> bool:
    """Resolve ``--lock auto|on|off``: ``auto`` is on only for file sources, never synthetic by default."""
    if lock not in ("auto", "on", "off"):
        raise ValueError(f"unknown lock mode {lock!r}; expected auto|on|off")
    return lock == "on" or (lock == "auto" and source != "synthetic")


# --------------------------------------------------------------------------------------
# approximate pair maps and traversal construction (reuses flow_prediction.Pair/transport)
# --------------------------------------------------------------------------------------

def flow_pair(name: str, fwd_field: np.ndarray, bwd_field: np.ndarray, W: int, H: int) -> FP.Pair:
    """An approximate RAFT pair map on the feature grid.

    ``fwd_field`` maps frame ``k -> k+1`` and ``bwd_field`` maps ``k+1 -> k``, both ``[H,W,2]``. Forward and
    inverse are ``p + flow(p)`` sampled bilinearly with border clamping (``flow_prediction.sample_field``);
    validity is in-bounds plus finite. No occlusion/disocclusion information exists in RAFT flow, so this is
    deliberately weaker than the analytic gate's exact pair.
    """
    fwd = np.asarray(fwd_field, np.float64)
    bwd = np.asarray(bwd_field, np.float64)

    def forward(p):
        p = np.asarray(p, np.float64)
        return p + FP.sample_field(fwd, p, W, H)

    def inverse(q):
        q = np.asarray(q, np.float64)
        return q + FP.sample_field(bwd, q, W, H)

    def fwd_valid(p):
        q = forward(p)
        return FP.in_bounds(p, W, H) & FP.in_bounds(q, W, H) & np.isfinite(q).all(1)

    def inv_valid(q):
        p = inverse(q)
        return FP.in_bounds(q, W, H) & FP.in_bounds(p, W, H) & np.isfinite(p).all(1)

    return FP.Pair(name, forward, inverse, fwd_valid, inv_valid)


def build_traversal(ref_fwd: np.ndarray, ref_bwd: np.ndarray, direction: str):
    """Build the pair list and reference fields for one direction's temporal traversal.

    ``forward``: pair ``k`` is the original pair ``k``; its map is ``(F_k, B_k)`` and its reference field is
    ``F_k``. ``backward``: traversal pair ``t`` is the original pair ``i = n-1-t`` traversed backwards, so its
    map is ``(B_i, F_i)`` (forward = the original backward flow) and its reference field is ``B_i``.
    """
    ref_fwd = np.asarray(ref_fwd, np.float64)
    ref_bwd = np.asarray(ref_bwd, np.float64)
    if ref_fwd.shape != ref_bwd.shape or ref_fwd.ndim != 4:
        raise ValueError("reference forward/backward fields must share shape [n_pairs,H,W,2]")
    n, H, W, _ = ref_fwd.shape
    if direction == "forward":
        pairs = [flow_pair(f"fwd{k}", ref_fwd[k], ref_bwd[k], W, H) for k in range(n)]
        return pairs, ref_fwd.copy()
    if direction == "backward":
        pairs, fields = [], np.zeros_like(ref_bwd)
        for k in range(n):
            i = n - 1 - k
            pairs.append(flow_pair(f"bwd{k}", ref_bwd[i], ref_fwd[i], W, H))
            fields[k] = ref_bwd[i]
        return pairs, fields
    raise ValueError(f"unknown direction {direction!r}; expected one of {DIRECTIONS}")


def traversal_order(n_pairs: int, direction: str) -> list:
    """Frame-index ``(src, dst)`` for every traversal pair (used to pick the cached features)."""
    if direction == "forward":
        return [(k, k + 1) for k in range(n_pairs)]
    if direction == "backward":
        return [(k + 1, k) for k in range(n_pairs - 1, -1, -1)]
    raise ValueError(f"unknown direction {direction!r}; expected one of {DIRECTIONS}")


# --------------------------------------------------------------------------------------
# model-agnostic deterministic replay (unit-testable with a fake refine)
# --------------------------------------------------------------------------------------

def replay(pairs: Sequence[FP.Pair], ref_fields: np.ndarray, method: str,
           refine: Callable[[np.ndarray, int], np.ndarray],
           min_coverage: float = FP.MIN_COVERAGE,
           on_step: Optional[Callable[[], None]] = None) -> list:
    """Replay one initialization policy over one traversal, one pair at a time.

    ``refine(init_flat, target) -> [N,2]`` is the refinement step (real RAFT refine, or a deterministic fake
    in tests). The policy's own refined outputs form its history for later pairs; ``ref_fields`` are the
    shared high-iteration cold references. Returns one JSON-safe row per pair.

    ``on_step`` is the shared monotonic gate, invoked once per completed pair (after the row is recorded) so a
    ``--lock-chunk`` boundary can yield here as well as during feature preparation and reference generation.

    Deterministic: the only inputs are ``pairs``/``ref_fields``/``method``/``refine``, and pairs are processed
    in traversal order with no randomness or time-dependent branching. Wall times are recorded but never
    influence the result; ``on_step`` is a timing/gating side effect only.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    ref_fields = np.asarray(ref_fields, np.float64)
    n, H, W, _ = ref_fields.shape
    grid = FP.source_grid(W, H)
    fields = np.zeros((n, H, W, 2), np.float64)   # this policy's own produced history
    valids = np.zeros((n, H, W), bool)
    rows = []
    for target in range(n):
        t0 = time.perf_counter()
        pred = FP.predict(pairs, fields, valids, target, method, W, H, min_coverage, grid)
        init_s = time.perf_counter() - t0
        eff = np.where(np.isfinite(pred.field), pred.field, 0.0)      # never hand a non-finite value to RAFT
        t1 = time.perf_counter()
        out = np.asarray(refine(eff.copy(), target), np.float64).reshape(-1, 2)
        refine_s = time.perf_counter() - t1
        out2 = out.reshape(H, W, 2)
        fields[target] = out2
        valids[target] = np.isfinite(out2).all(-1)
        ref = ref_fields[target].reshape(-1, 2)
        refined_epe = np.linalg.norm(out - ref, axis=1)
        eff2 = eff.reshape(H, W, 2)
        init_epe = np.linalg.norm(eff2 - ref_fields[target], axis=-1)[pred.valid.reshape(H, W)]
        st = _stats(refined_epe)
        rows.append({
            "target": int(target),
            "fallback": bool(pred.fallback),
            "reason": pred.reason,
            "coverage": float(pred.coverage),
            "effective_coverage": float(pred.valid.mean()),
            "validity_frac": float(valids[target].mean()),
            "init_s": float(init_s),
            "refine_s": float(refine_s),
            "init_epe_low": _stats(init_epe),
            "refined_epe_low": st,
        })
        if on_step is not None:
            on_step()          # completed pair boundary for the shared lock gate
    return rows


def summarize(rows: Sequence[dict], iters: int, units_to_source_px: float,
              targets: Optional[Sequence[int]] = None) -> dict:
    """Aggregate one method/direction's per-pair rows without collapsing quality into a single score.

    ``targets`` restricts the aggregate to an explicit pair set so methods with different history
    requirements can be compared on the same pairs (see ``summary_common``).
    """
    if targets is not None:
        keep = {int(t) for t in targets}
        rows = [r for r in rows if r["target"] in keep]
    usable = [r for r in rows if r["reason"] != "insufficient_history"]
    cov = [r["coverage"] for r in usable]
    epe = [r["refined_epe_low"]["mean"] for r in usable if r["refined_epe_low"]["mean"] is not None]
    init_epe = [r["init_epe_low"]["mean"] for r in usable if r["init_epe_low"]["mean"] is not None]
    return {
        "pairs": len(rows),
        "usable_pairs": len(usable),
        "skipped_no_history": int(len(rows) - len(usable)),
        "targets": [int(r["target"]) for r in rows],
        "fallbacks": int(sum(1 for r in usable if r["fallback"])),
        "coverage_mean": float(np.mean(cov)) if cov else None,
        "coverage_min": float(np.min(cov)) if cov else None,
        "init_epe_mean_low": float(np.mean(init_epe)) if init_epe else None,
        "refined_epe_mean_low": float(np.mean(epe)) if epe else None,
        "refined_epe_mean_source_px": (float(np.mean(epe)) * units_to_source_px) if epe else None,
        "init_s_mean": float(np.mean([r["init_s"] for r in usable])) if usable else None,
        "refine_s_mean": float(np.mean([r["refine_s"] for r in usable])) if usable else None,
        "refine_s_total": float(np.sum([r["refine_s"] for r in rows])),
        "iters": int(iters),
    }


def common_targets(results_direction: dict) -> list:
    """Targets where every method in a direction produced a real candidate (no insufficient history).

    Used for the fair ``summary_common`` aggregate: ``cold`` has a candidate from pair 0 while order-1/2
    extrapolation needs two/three prior pairs, so a plain mean over each method's own usable pairs would
    compare different pair sets.
    """
    sets = []
    for payload in results_direction.values():
        sets.append({int(r["target"]) for r in payload["per_pair"] if r["reason"] != "insufficient_history"})
    if not sets:
        return []
    return sorted(set.intersection(*sets))


# --------------------------------------------------------------------------------------
# opt-in per-iteration budget scan (offline oracle; history fixed at --iters)
# --------------------------------------------------------------------------------------

def parse_thresholds(text) -> list:
    """Parse ``"a,b,c"`` feature-grid EPE thresholds: finite, > 0, de-duplicated and ascending."""
    if isinstance(text, (list, tuple)):
        vals = [float(v) for v in text]
    else:
        vals = [float(v) for v in str(text).split(",") if str(v).strip()]
    if not vals:
        raise ValueError("--budget-thresholds must list at least one positive number")
    for v in vals:
        if not np.isfinite(v) or v <= 0.0:
            raise ValueError(f"--budget-thresholds values must be finite and > 0, got {v!r}")
    return sorted(set(vals))


class _ScanStepper:
    """One-iteration-at-a-time refinement state: ``step()`` is a zero-arg callable returning ``[N,2]``.

    The real (RAFT) stepper evaluates the device result before returning, so the harness can time the device
    refinement separately from the host EPE/oracle check. Deterministic fakes implement the same tiny surface.
    """

    def __init__(self, step_fn: Callable[[], np.ndarray]):
        self._step = step_fn

    def step(self) -> np.ndarray:
        return self._step()


class _BidirScanStepper:
    """Paired one-iteration refinement state: ``step()`` returns ``(fwd_flat, bwd_flat)`` for one pair.

    Mirrors ``RaftRunner.begin_scan`` for the production batch-2 bidirectional path: the forward (``k -> k+1``)
    and backward (``k+1 -> k``) fields are advanced for the *same pair* by the *same model* in the *same*
    iteration, so a cycle residual computed from one ``step()`` is genuinely same-iteration/same-pair/
    same-model. The real stepper evaluates the device result before returning; deterministic fakes implement
    the same tiny surface.
    """

    def __init__(self, step_fn: Callable[[], tuple]):
        self._step = step_fn

    def step(self) -> tuple:
        return self._step()


def replay_budget_scan(pairs: Sequence[FP.Pair], ref_fields: np.ndarray, method: str,
                       make_stepper: Callable[[np.ndarray, int], "_ScanStepper"],
                       iters: int, max_iters: int, thresholds: Sequence[float],
                       min_coverage: float = FP.MIN_COVERAGE,
                       on_step: Optional[Callable[[], None]] = None,
                       photometric: Optional[Callable[[int, np.ndarray], dict]] = None) -> list:
    """Offline oracle scan of one initialization policy over one traversal, one iteration at a time.

    Candidate construction, fallback and insufficient-history semantics are exactly :func:`replay`'s. The
    difference is refinement: every pair starts from the policy's own initialization and is advanced one
    iteration at a time up to ``max_iters``; after each iteration the mean (and p95/max) low-grid EPE against
    that pair's cold reference is recorded, plus the first iteration at or below each threshold. The history
    written for the following pair is **always the state after exactly ``iters`` iterations** -- the normal
    fixed-budget replay policy -- so scan iterations beyond ``iters`` never leak into a later initialization.
    ``max_iters`` must be ``>= iters`` so that state is always reached.

    Each row additionally records ``flow_update_low``: the mean/p95/max per-pixel magnitude of
    ``low - previous_low``, where ``previous_low`` is the effective initialization on the first step and the
    previous step's flow afterwards. It is computed **only** from the candidate's own initialization/current/
    previous prediction -- never from the cold reference -- so it is a candidate-observable signal distinct
    from the oracle ``epe_low``.

    When ``photometric`` is supplied it is called once per iteration as ``photometric(target, low_flat)`` and
    must return a JSON-safe photometric-residual dict (see :class:`PhotometricWarper`). It likewise uses only
    the candidate flow and the preprocessed frame pair, never the reference.

    Timing stays separated: ``update_s`` measures only ``stepper.step()`` (which must have evaluated the
    device result before returning), ``update_mag_check_s`` measures only the host update-magnitude/statistics
    calculation, ``photometric_check_s`` measures only the host photometric-warp/statistics calculation, and
    ``epe_check_s`` measures only the host EPE/threshold check. This is an offline oracle
    scan, not a deployable stopping rule: it evaluates against a reference the pipeline would not have, and
    the candidate-observable signals could support a future convergence stopping rule but **no stop
    rule is implemented or selected**; nothing here implies a speedup or readiness.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if int(max_iters) < int(iters):
        raise ValueError("max_iters must be >= iters so the fixed-budget history state is reached")
    if not list(thresholds):
        raise ValueError("thresholds must list at least one positive number")
    thresholds = [float(t) for t in thresholds]
    ref_fields = np.asarray(ref_fields, np.float64)
    n, H, W, _ = ref_fields.shape
    grid = FP.source_grid(W, H)
    fields = np.zeros((n, H, W, 2), np.float64)   # this policy's own produced history (at ``iters``)
    valids = np.zeros((n, H, W), bool)
    rows = []
    for target in range(n):
        t0 = time.perf_counter()
        pred = FP.predict(pairs, fields, valids, target, method, W, H, min_coverage, grid)
        init_s = time.perf_counter() - t0
        eff = np.where(np.isfinite(pred.field), pred.field, 0.0)      # never hand a non-finite value to RAFT
        stepper = make_stepper(eff.copy(), target)
        ref = ref_fields[target].reshape(-1, 2)
        scan = []
        crossings = [{"threshold": float(th), "iter": None} for th in thresholds]
        history_field = None
        last_low = None
        prev_low = eff.reshape(-1, 2).copy()   # candidate-observable baseline: the effective initialization
        min_epe = None
        min_iter = None
        stopped = None
        update_total = epe_total = signal_total = photometric_total = 0.0
        for it in range(1, int(max_iters) + 1):
            t1 = time.perf_counter()
            low = np.asarray(stepper.step(), np.float64).reshape(-1, 2)   # already evaluated on device
            update_s = time.perf_counter() - t1
            # Candidate-observable update magnitude: how far this step moved the candidate's own flow from
            # the previous prediction (or the effective initialization). Only the candidate's current/previous
            # state participates -- the cold reference is never used here -- so this is deliberately distinct
            # from the oracle ``epe_low``. Timed separately from the device update and the host EPE check.
            tm = time.perf_counter()
            update_stat = _stats(np.linalg.norm(low - prev_low, axis=1))
            update_mag_s = time.perf_counter() - tm
            # Candidate-observable photometric residual: warp the preprocessed source frame toward the target
            # at this iteration's own candidate flow. It never sees the reference flow; timed separately again.
            photometric_stat = None
            photometric_s = 0.0
            if photometric is not None:
                tp = time.perf_counter()
                photometric_stat = photometric(int(target), low)
                photometric_s = time.perf_counter() - tp
            t2 = time.perf_counter()
            finite = bool(np.isfinite(low).all())
            st = _stats(np.linalg.norm(low - ref, axis=1))
            if st["mean"] is not None:
                if min_epe is None or st["mean"] < min_epe:
                    min_epe, min_iter = st["mean"], it
                for c in crossings:
                    if c["iter"] is None and st["mean"] <= c["threshold"]:
                        c["iter"] = int(it)
            if it == int(iters):
                history_field = low.reshape(H, W, 2)
            epe_s = time.perf_counter() - t2
            update_total += update_s
            epe_total += epe_s
            signal_total += update_mag_s
            photometric_total += photometric_s
            last_low = low
            prev_low = low.copy()
            scan.append({"iter": int(it), "epe_low": st, "flow_update_low": update_stat,
                         "photometric_low": photometric_stat,
                         "update_s": float(update_s), "update_mag_check_s": float(update_mag_s),
                         "photometric_check_s": float(photometric_s),
                         "epe_check_s": float(epe_s)})
            if not finite:
                stopped = "non_finite"
                break
        if history_field is None:
            # Only reachable if a non-finite step stopped the scan before ``iters``; fall back to the last
            # produced field so later pairs still see a finite-safe (but flagged) history, as in ``replay``.
            history_field = last_low.reshape(H, W, 2)
        fields[target] = history_field
        valids[target] = np.isfinite(history_field).all(-1)
        at_iters = next((e for e in scan if e["iter"] == int(iters)), scan[-1] if scan else None)
        refined_epe = at_iters["epe_low"] if at_iters is not None else _stats(np.array([]))
        history_update_s = float(sum(e["update_s"] for e in scan if e["iter"] <= int(iters)))
        eff2 = eff.reshape(H, W, 2)
        init_epe = np.linalg.norm(eff2 - ref_fields[target], axis=-1)[pred.valid.reshape(H, W)]
        rows.append({
            "target": int(target),
            "fallback": bool(pred.fallback),
            "reason": pred.reason,
            "coverage": float(pred.coverage),
            "effective_coverage": float(pred.valid.mean()),
            "validity_frac": float(valids[target].mean()),
            "init_s": float(init_s),
            "refine_s": history_update_s,             # fixed-budget-at-iters device time (compat with replay)
            "init_epe_low": _stats(init_epe),
            "refined_epe_low": refined_epe,           # EPE at ``iters`` (compat with summarize)
            "history_iters": int(iters),
            "max_iters": int(max_iters),
            "scan": scan,
            "crossings": crossings,
            "crossed": bool(any(c["iter"] is not None for c in crossings)),
            "min_epe_low": min_epe,
            "min_epe_iter": int(min_iter) if min_iter is not None else None,
            "stopped_reason": stopped,
            "update_s_total": float(update_total),
            "epe_check_s_total": float(epe_total),
            "update_mag_check_s_total": float(signal_total),
            "photometric_check_s_total": float(photometric_total),
        })
        if on_step is not None:
            on_step()          # completed pair boundary for the shared lock gate
    return rows


def _prefix_costs(row: dict, steps_consumed: int) -> tuple:
    """Sum the measured device-update / signal-check / EPE-check / photometric times of a scan-step prefix.

    ``steps_consumed`` is clamped to the actually recorded ``row["scan"]`` prefix, so a scan that stopped
    early on non-finite output contributes only the work it really performed. The timings are kept separate:
    device ``update_s``; host candidate-observable ``update_mag_check_s``; host candidate-observable
    ``photometric_check_s``; and host oracle ``epe_check_s``.
    """
    scan = row["scan"]
    n = max(0, min(int(steps_consumed), len(scan)))
    update_s = float(sum(float(e["update_s"]) for e in scan[:n]))
    epe_check_s = float(sum(float(e["epe_check_s"]) for e in scan[:n]))
    signal_check_s = float(sum(float(e["update_mag_check_s"]) for e in scan[:n]))
    photometric_check_s = float(sum(float(e.get("photometric_check_s", 0.0)) for e in scan[:n]))
    return update_s, epe_check_s, signal_check_s, photometric_check_s


def summarize_budget_scan(rows: Sequence[dict], iters: int, max_iters: int, thresholds: Sequence[float],
                          units_to_source_px: float, targets: Optional[Sequence[int]] = None) -> dict:
    """Aggregate a budget scan: per-threshold reach rates/iterations plus measured prefix and full-scan times.

    Rows without enough history (``insufficient_history``) are excluded exactly as in :func:`summarize` so the
    scan aggregate matches the fixed-budget common-pair semantics. ``thresholds`` must be the same ordered list
    used to build the rows (the per-pair ``crossings`` are aligned to it).

    The trace deliberately runs every pair to the scan end so the full EPE curve is measured. The top-level
    ``full_scan_*`` totals therefore sum that whole trace (over all recorded rows), whereas each
    ``per_threshold`` entry reports **prefix-summed measured costs**: for every usable pair the prefix is the
    first crossing iteration when the threshold was reached and otherwise the number of measured scan steps
    (normally ``max_iters``, less only if output went non-finite early). Censored pairs therefore still
    contribute their full measured work. These prefix sums are not separate early-stopped executions.
    """
    if targets is not None:
        keep = {int(t) for t in targets}
        rows = [r for r in rows if r["target"] in keep]
    usable = [r for r in rows if r["reason"] != "insufficient_history"]
    thresholds = [float(t) for t in thresholds]
    per_threshold = []
    for i, th in enumerate(thresholds):
        reached = [r["crossings"][i]["iter"] for r in usable if r["crossings"][i]["iter"] is not None]
        consumed = []
        update_s_sum = epe_check_s_sum = signal_check_s_sum = photometric_check_s_sum = 0.0
        for r in usable:
            crossing = r["crossings"][i]["iter"]
            # reached -> first crossing iteration; censored -> all measured scan steps (incl. max_iters)
            steps_consumed = int(crossing) if crossing is not None else len(r["scan"])
            consumed.append(steps_consumed)
            update_s, epe_check_s, signal_check_s, photometric_check_s = _prefix_costs(r, steps_consumed)
            update_s_sum += update_s
            epe_check_s_sum += epe_check_s
            signal_check_s_sum += signal_check_s
            photometric_check_s_sum += photometric_check_s
        total_s_sum = update_s_sum + epe_check_s_sum
        total_with_signal_s_sum = total_s_sum + signal_check_s_sum
        total_with_observables_s_sum = total_with_signal_s_sum + photometric_check_s_sum
        npairs = len(usable)
        per_threshold.append({
            "threshold": float(th),
            "threshold_source_px": float(th) * float(units_to_source_px),
            "reached": len(reached),
            "censored": len(usable) - len(reached),
            "reach_rate": (len(reached) / len(usable)) if usable else None,
            "iter_mean": float(np.mean(reached)) if reached else None,
            "iter_median": float(np.median(reached)) if reached else None,
            "iter_min": int(min(reached)) if reached else None,
            "iter_max": int(max(reached)) if reached else None,
            # iterations consumed across all usable pairs; censored pairs consume the measured scan cap
            "steps_consumed_mean": float(np.mean(consumed)) if consumed else None,
            "steps_consumed_median": float(np.median(consumed)) if consumed else None,
            "steps_consumed_min": int(min(consumed)) if consumed else None,
            "steps_consumed_max": int(max(consumed)) if consumed else None,
            # prefix-summed measured costs over the consumed steps (reached + censored work); the
            # candidate-observable signal check is reported separately and never folded into ``total_s_*``
            "update_s_total": update_s_sum,
            "epe_check_s_total": epe_check_s_sum,
            "signal_check_s_total": signal_check_s_sum,
            "photometric_check_s_total": photometric_check_s_sum,
            "total_s_total": total_s_sum,
            "total_with_signal_s_total": total_with_signal_s_sum,
            "total_with_observables_s_total": total_with_observables_s_sum,
            "update_s_mean": (update_s_sum / npairs) if npairs else None,
            "epe_check_s_mean": (epe_check_s_sum / npairs) if npairs else None,
            "signal_check_s_mean": (signal_check_s_sum / npairs) if npairs else None,
            "photometric_check_s_mean": (photometric_check_s_sum / npairs) if npairs else None,
            "total_s_mean": (total_s_sum / npairs) if npairs else None,
            "total_with_signal_s_mean": (total_with_signal_s_sum / npairs) if npairs else None,
            "total_with_observables_s_mean": (total_with_observables_s_sum / npairs) if npairs else None,
        })
    reached_any = int(sum(1 for r in usable if r["crossed"]))
    min_epes = [r["min_epe_low"] for r in usable if r["min_epe_low"] is not None]
    return {
        "pairs": len(rows),
        "usable_pairs": len(usable),
        "skipped_no_history": int(len(rows) - len(usable)),
        "targets": [int(r["target"]) for r in rows],
        "fallbacks": int(sum(1 for r in usable if r["fallback"])),
        "iters": int(iters),
        "max_iters": int(max_iters),
        "thresholds": thresholds,
        "per_threshold": per_threshold,
        "reached_any": reached_any,
        "censored_any": int(len(usable) - reached_any),
        "reach_rate_any": (reached_any / len(usable)) if usable else None,
        "min_epe_low_mean": float(np.mean(min_epes)) if min_epes else None,
        # full-scan totals: the trace runs to the scan end to measure the whole EPE curve, so these sum every
        # recorded row, not the per-threshold prefixes; see ``per_threshold`` for the prefix-summed costs.
        "full_scan_update_s_total": float(sum(r["update_s_total"] for r in rows)),
        "full_scan_epe_check_s_total": float(sum(r["epe_check_s_total"] for r in rows)),
        "full_scan_signal_check_s_total": float(sum(r["update_mag_check_s_total"] for r in rows)),
        "full_scan_photometric_check_s_total": float(sum(r["photometric_check_s_total"] for r in rows)),
        "full_scan_total_s": float(sum(r["update_s_total"] + r["epe_check_s_total"] for r in rows)),
        "full_scan_total_with_signal_s": float(sum(r["update_s_total"] + r["epe_check_s_total"]
                                                   + r["update_mag_check_s_total"] for r in rows)),
        "full_scan_total_with_observables_s": float(sum(r["update_s_total"] + r["epe_check_s_total"]
                                                         + r["update_mag_check_s_total"]
                                                         + r["photometric_check_s_total"] for r in rows)),
        "full_scan_update_s_mean": float(np.mean([r["update_s_total"] for r in usable])) if usable else None,
        "full_scan_epe_check_s_mean": float(np.mean([r["epe_check_s_total"] for r in usable])) if usable else None,
        "full_scan_signal_check_s_mean": (float(np.mean([r["update_mag_check_s_total"] for r in usable]))
                                          if usable else None),
        "full_scan_photometric_check_s_mean": (float(np.mean([r["photometric_check_s_total"] for r in usable]))
                                               if usable else None),
        "full_scan_total_s_mean": (float(np.mean([r["update_s_total"] + r["epe_check_s_total"] for r in usable]))
                                   if usable else None),
        "full_scan_total_with_signal_s_mean": (
            float(np.mean([r["update_s_total"] + r["epe_check_s_total"] + r["update_mag_check_s_total"]
                           for r in usable])) if usable else None),
        "full_scan_total_with_observables_s_mean": (
            float(np.mean([r["update_s_total"] + r["epe_check_s_total"] + r["update_mag_check_s_total"]
                           + r["photometric_check_s_total"] for r in usable])) if usable else None),
        "history_refine_s_total": float(sum(r["refine_s"] for r in usable)),
        "stopped_non_finite": int(sum(1 for r in usable if r["stopped_reason"] == "non_finite")),
        "oracle": True,
        "candidate_observable": {
            "flow_update_low": ("per-iteration mean/p95/max of the per-pixel magnitude |low - previous_low| "
                                "(feature-grid px); previous_low is the effective initialization at "
                                "iteration 1 and the previous step afterwards; computed only from the "
                                "candidate's own current/previous prediction, so the cold reference never "
                                "participates and it is distinct from the oracle epe_low"),
            "update_mag_check_s": ("host time to compute flow_update_low, measured separately from device "
                                   "update_s and oracle epe_check_s; prefix sums live in signal_check_s_* "
                                   "and full_scan_signal_check_s_*"),
            "photometric_low": ("per-iteration robust absolute luminance-residual statistics after warping "
                                "the preprocessed source frame toward the target frame at the candidate's own "
                                "flow (Rec.601 luminance, global-median brightness offset removed, pad/OOB/"
                                "non-finite samples excluded; resid_textured restricts to source cells above "
                                "the gradient threshold); uses no reference flow, so it is distinct from epe_low"),
            "photometric_check_s": ("host time to compute photometric_low, measured separately from device "
                                    "update_s, signal update_mag_check_s and oracle epe_check_s; prefix sums "
                                    "live in photometric_check_s_* and full_scan_photometric_check_s_*"),
            "stopping_rule": ("none implemented or selected; these candidate-observable diagnostics could "
                              "support a future convergence-based stopping rule, but the current threshold "
                              "scan remains an offline oracle comparison to the cold reference"),
        },
        "note": ("offline oracle scan: per-iteration mean feature-grid EPE against the cold high-iteration "
                 "reference; history is always the fixed-budget state at iters; per_threshold costs are "
                 "prefix sums of the measured steps (censored pairs run to the measured scan end) while "
                 "full_scan_* totals cover the whole trace; flow_update_low and photometric_low are "
                 "candidate-observable diagnostics only with no stopping rule implemented or selected; "
                 "total_with_observables_s_* adds both signal costs to the unchanged total_s_*; not a "
                 "deployable stopping rule and no speedup, online or production-readiness claim"),
    }


# --------------------------------------------------------------------------------------
# opt-in cold-only bidirectional cycle-consistency pilot (candidate-observable; no reference in the signal)
# --------------------------------------------------------------------------------------

def _finite_mean(values) -> Optional[float]:
    v = [float(x) for x in values if x is not None and np.isfinite(float(x))]
    return float(np.mean(v)) if v else None


def _finite_median(values) -> Optional[float]:
    v = [float(x) for x in values if x is not None and np.isfinite(float(x))]
    return float(np.median(v)) if v else None


def replay_cycle_scan(ref_fwd: np.ndarray, ref_bwd: np.ndarray,
                      make_stepper: Callable[[int], "_BidirScanStepper"], max_iters: int,
                      on_step: Optional[Callable[[], None]] = None) -> list:
    """Cold-only bidirectional per-iteration cycle-consistency scan over every pair.

    For each pair the forward and backward candidates are always the **cold zero initialization** refined
    together, one iteration at a time, by ``make_stepper(target)`` (which must return the two fields from the
    same model/pair/iteration). No flow from another pair is reused, so there is no cross-pair history
    invariant to preserve. Every iteration records the candidate-observable ``cycle_low`` computed only from
    that iteration's two fields (see :func:`cycle_residual`) and, separately, the oracle
    ``epe_low``/``epe_low_fwd``/``epe_low_bwd`` against the cold high-iteration reference fields. The two are
    deliberately kept apart: ``cycle_low`` never consumes a reference field or EPE.

    Timing stays separated: ``update_s`` is the device refinement, ``cycle_check_s`` the host cycle-residual
    calculation and ``epe_check_s`` the host oracle EPE check. A pair whose fields go non-finite stops early
    with ``stopped_reason='non_finite'`` (the cycle residual already excludes such samples). ``on_step`` is
    invoked once per completed pair for the shared lock gate; wall times are recorded but never influence the
    result, so the scan is deterministic given ``make_stepper``.
    """
    ref_fwd = np.asarray(ref_fwd, np.float64)
    ref_bwd = np.asarray(ref_bwd, np.float64)
    if ref_fwd.shape != ref_bwd.shape or ref_fwd.ndim != 4:
        raise ValueError("reference forward/backward fields must share shape [n_pairs,H,W,2]")
    if int(max_iters) < 1:
        raise ValueError("max_iters must be >= 1")
    n, h, w, _ = ref_fwd.shape
    rows = []
    for target in range(n):
        stepper = make_stepper(target)
        rf = ref_fwd[target].reshape(-1, 2)
        rb = ref_bwd[target].reshape(-1, 2)
        scan = []
        stopped = None
        update_total = cycle_total = epe_total = 0.0
        for it in range(1, int(max_iters) + 1):
            t1 = time.perf_counter()
            fwd, bwd = stepper.step()          # same model, same pair, same iteration (device-evaluated)
            update_s = time.perf_counter() - t1
            fwd = np.asarray(fwd, np.float64).reshape(-1, 2)
            bwd = np.asarray(bwd, np.float64).reshape(-1, 2)
            # Candidate-observable cycle residual: only the two candidate fields and the grid geometry. The
            # cold reference and EPE never enter this call; its host cost is timed separately below.
            tc = time.perf_counter()
            cyc = cycle_residual(fwd, bwd, h, w)
            cycle_s = time.perf_counter() - tc
            # Oracle EPE against the cold high-iteration reference, reported separately for post-hoc eval.
            te = time.perf_counter()
            epe_f = np.linalg.norm(fwd - rf, axis=1)
            epe_b = np.linalg.norm(bwd - rb, axis=1)
            epe_stats = _stats(np.concatenate([epe_f, epe_b]))
            epe_s = time.perf_counter() - te
            finite = bool(np.isfinite(fwd).all() and np.isfinite(bwd).all())
            scan.append({
                "iter": int(it),
                "cycle_low": cyc,
                "epe_low": epe_stats,
                "epe_low_fwd": _stats(epe_f),
                "epe_low_bwd": _stats(epe_b),
                "update_s": float(update_s),
                "cycle_check_s": float(cycle_s),
                "epe_check_s": float(epe_s),
            })
            update_total += update_s
            cycle_total += cycle_s
            epe_total += epe_s
            if not finite:
                stopped = "non_finite"
                break
        rows.append({
            "target": int(target),
            "reason": None,
            "fallback": False,
            "scan": scan,
            "stopped_reason": stopped,
            "update_s_total": float(update_total),
            "cycle_check_s_total": float(cycle_total),
            "epe_check_s_total": float(epe_total),
        })
        if on_step is not None:
            on_step()          # completed pair boundary for the shared lock gate
    return rows


def summarize_cycle_scan(rows: Sequence[dict], max_iters: int, units_to_source_px: float) -> dict:
    """Aggregate the cold bidirectional cycle scan; oracle EPE and its check cost stay separate.

    ``cycle_low`` is the candidate-observable signal and is summarized on its own; the oracle EPE statistics
    and the host ``epe_check_s`` cost are reported separately and never folded into the cycle totals. Timing is
    separated into device ``update_s``, host candidate-observable ``cycle_check_s`` and host oracle
    ``epe_check_s``. Cold has no history requirement, so every pair is usable.
    """
    usable = [r for r in rows]
    per_iter = {}
    for r in usable:
        for e in r["scan"]:
            per_iter.setdefault(int(e["iter"]), []).append(e)
    by_iter = []
    for it in sorted(per_iter):
        entries = per_iter[it]
        cyc_means = [e["cycle_low"]["resid"]["mean"] for e in entries]
        vf = [e["cycle_low"]["valid_frac"] for e in entries]
        finite_means = [float(v) for v in cyc_means if v is not None and np.isfinite(float(v))]
        by_iter.append({
            "iter": int(it),
            "trajectories": len(entries),
            "cycle_low_resid_mean_median": _finite_median(cyc_means),
            "cycle_low_resid_mean_min": float(min(finite_means)) if finite_means else None,
            "cycle_low_resid_mean_max": float(max(finite_means)) if finite_means else None,
            "valid_frac_mean": _finite_mean(vf),
        })
    all_cyc = [e["cycle_low"]["resid"]["mean"] for r in usable for e in r["scan"]]
    all_vf = [e["cycle_low"]["valid_frac"] for r in usable for e in r["scan"]]
    update_total = float(sum(r["update_s_total"] for r in usable))
    cycle_total = float(sum(r["cycle_check_s_total"] for r in usable))
    epe_total = float(sum(r["epe_check_s_total"] for r in usable))
    npairs = len(usable)
    return {
        "pairs": len(rows),
        "usable_pairs": npairs,
        "targets": [int(r["target"]) for r in rows],
        "max_iters": int(max_iters),
        "units_to_source_px": float(units_to_source_px),
        "stopped_non_finite": int(sum(1 for r in usable if r["stopped_reason"] == "non_finite")),
        "cycle_low_resid_mean": _finite_mean(all_cyc),
        "cycle_low_resid_median_of_pair_iteration_means": _finite_median(all_cyc),
        "valid_frac_mean": _finite_mean(all_vf),
        "cycle_low_by_iter": by_iter,
        # device refinement, host candidate-observable check and host oracle check are separate costs
        "full_scan_update_s_total": update_total,
        "full_scan_cycle_check_s_total": cycle_total,
        "full_scan_epe_check_s_total": epe_total,
        "full_scan_candidate_total_s": update_total + cycle_total,
        "full_scan_total_with_oracle_s": update_total + cycle_total + epe_total,
        "full_scan_update_s_mean": (update_total / npairs) if npairs else None,
        "full_scan_cycle_check_s_mean": (cycle_total / npairs) if npairs else None,
        "full_scan_epe_check_s_mean": (epe_total / npairs) if npairs else None,
        "candidate_observable": {
            "cycle_low": ("per-iteration forward-backward cycle-consistency/visibility residual from the two "
                          "same-iteration cold candidate fields (mean/median/p95/max in feature-grid px and "
                          "valid_frac); never uses a reference field or EPE"),
            "cycle_check_s": ("host time to compute cycle_low, kept separate from device update_s and oracle "
                              "epe_check_s"),
            "stopping_rule": ("none implemented or selected; cycle_low is a diagnostic, not a stopping rule"),
        },
        "oracle": {
            "epe_low": ("mean feature-grid EPE of the two candidate fields against the cold high-iteration "
                        "reference; reported separately for post-hoc evaluation only, not ground truth"),
            "epe_check_s": "host time to compute the oracle EPE, kept separate from cycle_low and its cost",
        },
        "note": ("cold-only bidirectional cycle-consistency pilot: every pair starts from zero initialization "
                 "in both directions and is advanced together one iteration at a time; cycle_low is a "
                 "candidate-observable consistency/visibility diagnostic, not true-flow accuracy, and no stop "
                 "rule is implemented or selected"),
    }


# --------------------------------------------------------------------------------------
# real RAFT runner (features cached once; shared by reference and every candidate)
# --------------------------------------------------------------------------------------

class RaftRunner:
    """Thin, isolated wrapper around the existing RAFT model that caches per-frame features.

    It deliberately reuses ``RAFT.frame_features`` / ``CorrBlock`` / ``RAFTCore._iterate`` and
    ``RAFT.flow_bidir`` but does not touch ``RaftFlow`` or any production default.
    """

    def __init__(self, weights, downscale: int = 1, flow_dtype: str = "float32", corr_dtype: str = "float32"):
        import mlx.core as mx
        from mlxstab.raft.model import load_raft
        self.mx = mx
        self.downscale = int(downscale)
        self.flow_dtype = getattr(mx, flow_dtype)
        self.corr_dtype = getattr(mx, corr_dtype)
        self.model = load_raft(weights)
        if self.flow_dtype != mx.float32:
            self.model.to_dtype(self.flow_dtype)
        self.feats = []
        self.h = self.w = None
        self.pad = None
        self.prep_s = 0.0

    def _prep_pad(self, im):
        from mlxstab.raft.model import pad8
        mx = self.mx
        x = mx.array(im).astype(mx.float32)
        if self.downscale > 1:
            H, W = x.shape[:2]
            H2, W2 = H // self.downscale, W // self.downscale
            x = x[:H2 * self.downscale, :W2 * self.downscale].reshape(
                H2, self.downscale, W2, self.downscale, 3).mean(axis=(1, 3))
        return pad8(x[None])

    def prepare(self, frames: Sequence[np.ndarray], on_step: Optional[Callable[[], None]] = None) -> None:
        """Encode every frame once (fnet/cnet) and cache the features for the whole run.

        ``on_step`` is the shared monotonic gate, invoked once per completed frame so lock chunking can yield
        between frame encodings as well as between reference/candidate pairs.
        """
        t0 = time.perf_counter()
        self.feats = []
        for im in frames:
            x, p = self._prep_pad(im)
            self.feats.append(self.model.frame_features(x))
            self.pad = tuple(int(v) for v in p)
            if on_step is not None:
                on_step()      # completed frame boundary for the shared lock gate
        self.mx.eval([f.fmap for f in self.feats])
        _, self.h, self.w, _ = self.feats[0].fmap.shape
        self.prep_s = time.perf_counter() - t0

    def refine_pair(self, src: int, dst: int, iters: int, init_flat: np.ndarray) -> np.ndarray:
        """Refine one direction with a given low-resolution initialization using cached features."""
        from mlxstab.raft.model import CorrBlock, coords_grid
        mx = self.mx
        fa, fb = self.feats[src], self.feats[dst]
        cf = CorrBlock(fa.fmap, fb.fmap, self.model.levels, self.model.radius, self.corr_dtype)
        c0 = coords_grid(self.h, self.w)
        init = mx.array(np.asarray(init_flat, np.float32).reshape(1, self.h, self.w, 2))
        c1 = c0 + init
        net, c1 = self.model._iterate([cf], fa.net, fa.inp, c0, c1, int(iters))
        low = c1 - c0
        mx.eval(low)
        return np.array(low[0], np.float64)

    def reference_pair(self, src: int, dst: int, iters: int):
        """Cold high-iteration reference for both directions in one production batch-2 call."""
        mx = self.mx
        fa, fb = self.feats[src], self.feats[dst]
        lf, uf, lb, ub = self.model.flow_bidir(fa, fb, int(iters), None, None, self.corr_dtype)
        mx.eval(lf, lb)
        return np.array(lf[0], np.float64), np.array(lb[0], np.float64)

    def refine_closure(self, order: Sequence[tuple], iters: int):
        def refine(init_flat, target):
            s, d = order[target]
            return self.refine_pair(s, d, iters, init_flat)
        return refine

    def begin_scan(self, src: int, dst: int, init_flat: np.ndarray) -> "_ScanStepper":
        """Start one-iteration-at-a-time refinement of one pair from an initialization.

        Returns a stepper whose ``step()`` performs exactly one ``_iterate(..., iters=1)``, evaluates the
        device result (``mx.eval``) and only then returns the numpy low-resolution flow. Calling ``step()``
        ``iters`` times from the policy initialization reproduces :meth:`refine_pair` exactly, which is what
        makes the fixed-history-at-``iters`` invariant hold; the harness times each ``step()`` as the device
        refinement time and the returned flow as the input to the host EPE check.
        """
        from mlxstab.raft.model import CorrBlock, coords_grid
        mx = self.mx
        fa, fb = self.feats[src], self.feats[dst]
        cf = CorrBlock(fa.fmap, fb.fmap, self.model.levels, self.model.radius, self.corr_dtype)
        c0 = coords_grid(self.h, self.w)
        init = mx.array(np.asarray(init_flat, np.float32).reshape(1, self.h, self.w, 2))
        state = {"c1": c0 + init, "net": fa.net, "inp": fa.inp}

        def one_step():
            net, c1 = self.model._iterate([cf], state["net"], state["inp"], c0, state["c1"], 1)
            state["net"], state["c1"] = net, c1
            low = c1 - c0
            mx.eval(low)
            return np.array(low[0], np.float64)

        return _ScanStepper(one_step)

    def scan_closure(self, order: Sequence[tuple]):
        def make_stepper(init_flat, target):
            s, d = order[target]
            return self.begin_scan(s, d, init_flat)
        return make_stepper

    def begin_bidir_scan(self, src: int, dst: int,
                         init_f_flat: Optional[np.ndarray] = None,
                         init_b_flat: Optional[np.ndarray] = None) -> "_BidirScanStepper":
        """Start one-iteration-at-a-time refinement of the forward and backward pair together (batch 2).

        Mirrors :meth:`reference_pair` / ``RAFT.flow_bidir``: the forward (``src -> dst``) and backward
        (``dst -> src``) correlation pyramids are built once and both states are advanced by the same
        ``_iterate(..., iters=1)`` call, so ``step()`` returns the two fields from the same model, the same
        pair and the same iteration. Calling ``step()`` ``iters`` times from zero initializations reproduces
        :meth:`reference_pair` (the parity test pins this). Both initializations default to cold zeros.
        """
        from mlxstab.raft.model import CorrBlock, coords_grid
        mx = self.mx
        fa, fb = self.feats[src], self.feats[dst]
        cf = CorrBlock(fa.fmap, fb.fmap, self.model.levels, self.model.radius, self.corr_dtype)
        cb = CorrBlock(fb.fmap, fa.fmap, self.model.levels, self.model.radius, self.corr_dtype)
        c0 = mx.concatenate([coords_grid(self.h, self.w)] * 2, 0)
        z = np.zeros((self.h * self.w, 2), np.float64)
        f0 = z if init_f_flat is None else np.asarray(init_f_flat, np.float64)
        b0 = z if init_b_flat is None else np.asarray(init_b_flat, np.float64)
        init = mx.concatenate([
            mx.array(np.asarray(f0, np.float32).reshape(1, self.h, self.w, 2)),
            mx.array(np.asarray(b0, np.float32).reshape(1, self.h, self.w, 2)),
        ], 0)
        state = {"c1": c0 + init, "net": mx.concatenate([fa.net, fb.net], 0),
                 "inp": mx.concatenate([fa.inp, fb.inp], 0)}

        def one_step():
            net, c1 = self.model._iterate([cf, cb], state["net"], state["inp"], c0, state["c1"], 1)
            state["net"], state["c1"] = net, c1
            low = c1 - c0
            mx.eval(low)
            arr = np.array(low, np.float64)
            return arr[0].reshape(-1, 2), arr[1].reshape(-1, 2)

        return _BidirScanStepper(one_step)

    def bidir_scan_closure(self):
        """``make_stepper(target)`` for :func:`replay_cycle_scan`: cold paired refinement of pair ``target``."""
        def make_stepper(target):
            return self.begin_bidir_scan(int(target), int(target) + 1)
        return make_stepper


def compute_references(runner: RaftRunner, n_pairs: int, ref_iters: int,
                       on_step: Optional[Callable[[], None]] = None):
    """Cold high-iteration reference fields for every pair, in both directions, plus per-pair wall time.

    ``on_step`` is the shared monotonic gate, invoked once per completed reference pair.
    """
    ref_fwd = np.zeros((n_pairs, runner.h, runner.w, 2), np.float64)
    ref_bwd = np.zeros((n_pairs, runner.h, runner.w, 2), np.float64)
    times = []
    for i in range(n_pairs):
        t0 = time.perf_counter()
        F, B = runner.reference_pair(i, i + 1, ref_iters)
        times.append(time.perf_counter() - t0)
        ref_fwd[i], ref_bwd[i] = F, B
        if on_step is not None:
            on_step()          # completed reference-pair boundary for the shared lock gate
    return ref_fwd, ref_bwd, times


def run_cycle_scan_block(runner: RaftRunner, ref_fwd: np.ndarray, ref_bwd: np.ndarray, max_iters: int,
                         on_step: Optional[Callable[[], None]], units_to_source_px: float,
                         log: Callable[[str], None] = print) -> dict:
    """Run the cold bidirectional cycle-consistency pilot and return the manifest ``cycle_scan`` block.

    The oracle EPE against the (already computed) cold references is recorded next to, but separately from,
    the candidate-observable ``cycle_low``; the reference fields never enter :func:`cycle_residual`.
    """
    make_stepper = runner.bidir_scan_closure()
    n_pairs = int(ref_fwd.shape[0])
    log(f"[flow-init] bidirectional cold cycle scan over {n_pairs} pairs to {int(max_iters)} iters")
    rows = replay_cycle_scan(ref_fwd, ref_bwd, make_stepper, int(max_iters), on_step=on_step)
    summary = summarize_cycle_scan(rows, int(max_iters), units_to_source_px)
    return {
        "enabled": True,
        "kind": "cold_bidirectional_cycle_consistency",
        "direction": "bidirectional",
        "method": "cold",
        "max_iters": int(max_iters),
        "results": {"bidirectional": {"cold": {"per_pair": rows, "summary": summary}}},
        "candidate_observable": {
            "cycle_low": ("per-iteration forward-backward cycle-consistency/visibility residual from the two "
                           "same-iteration cold candidate fields (mean/median/p95/max in feature-grid px and "
                           "valid_frac); computed only from the candidate fields and grid geometry, never "
                           "from the cold reference or EPE"),
            "cycle_check_s": ("host time to compute cycle_low, kept separate from device update_s and oracle "
                              "epe_check_s"),
            "stopping_rule": ("none implemented or selected; cycle_low is a diagnostic and not a stopping "
                              "rule"),
        },
        "oracle": {
            "epe_low": ("per-iteration mean feature-grid EPE of the two candidate fields against the cold "
                        "high-iteration reference; reported separately for post-hoc evaluation only and not "
                        "ground truth"),
            "epe_check_s": "host time to compute the oracle EPE, kept separate from cycle_low and its cost",
        },
        "note": ("cold-only bidirectional cycle-consistency pilot: both directions start at zero and are "
                 "advanced together; cycle_low is a candidate-observable consistency/visibility diagnostic, "
                 "not true-flow accuracy; no threshold is selected and no stop rule is implemented"),
    }


# --------------------------------------------------------------------------------------
# benchmark orchestration
# --------------------------------------------------------------------------------------

def run_benchmark(runner: RaftRunner, frames: Sequence[np.ndarray], methods: Sequence[str] = METHODS,
                  directions: Sequence[str] = DIRECTIONS, iters: int = DEFAULT_ITERS,
                  ref_iters: int = DEFAULT_REF_ITERS, min_coverage: float = FP.MIN_COVERAGE,
                  max_pairs: Optional[int] = None, log: Callable[[str], None] = print,
                  on_step: Optional[Callable[[int], None]] = None, budget_scan: bool = False,
                  budget_max_iters: int = DEFAULT_BUDGET_MAX_ITERS,
                  budget_thresholds: Sequence[float] = DEFAULT_BUDGET_THRESHOLDS,
                  cycle_scan: bool = False,
                  cycle_max_iters: int = DEFAULT_CYCLE_MAX_ITERS) -> dict:
    """Reference + every method/direction on the shared cached features. Returns a JSON-safe result dict.

    ``on_step`` (typically ``HeavyLock.on_step``) is threaded through one :class:`StepCounter` so feature
    preparation, reference generation and every candidate replay share a single monotonic sequence of
    completed frame/pair boundaries. The returned ``steps`` is the total number of boundaries reached.

    With ``budget_scan`` the candidate replay is replaced by :func:`replay_budget_scan`: every pair is refined
    one iteration at a time up to ``budget_max_iters`` while the fixed-budget history is still the state at
    ``iters``. The fixed-budget aggregates (``summary``/``summary_common``) remain the iters-iteration
    equivalents and the scan adds a separate ``budget_scan`` block per method. ``budget_max_iters`` must be
    ``>= iters`` so the state the history invariant depends on is always reached.

    With ``cycle_scan`` the run is a **cycle-only pilot**: the standard method/direction replay is skipped and
    no ``budget_scan`` runs. The cold high-iteration references are still computed (for the separately
    reported oracle EPE) and the paired forward/backward candidates are scanned one iteration at a time to
    ``cycle_max_iters`` (see :func:`replay_cycle_scan`). The block is returned under ``out["cycle_scan"]``.
    """
    n_pairs = len(frames) - 1
    if max_pairs is not None:
        n_pairs = min(n_pairs, int(max_pairs))
    if n_pairs < 1:
        raise ValueError("need at least two frames (one pair)")
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"unknown method {m!r}")
    for d in directions:
        if d not in DIRECTIONS:
            raise ValueError(f"unknown direction {d!r}")
    if cycle_scan and budget_scan:
        raise ValueError("cycle_scan is a cycle-only pilot and cannot be combined with budget_scan")
    requested_directions = list(directions)
    if cycle_scan:
        if int(cycle_max_iters) < 1:
            raise ValueError("cycle_max_iters must be >= 1")
        methods, directions = (), ()      # cycle-only: no standard candidate replay
    if budget_scan:
        if int(budget_max_iters) < int(iters):
            raise ValueError("--budget-max-iters must be >= --iters to preserve the fixed-budget history")
        if not list(budget_thresholds):
            raise ValueError("--budget-thresholds must list at least one threshold")

    steps = StepCounter(on_step)
    runner.prepare(frames[:n_pairs + 1], on_step=steps.tick)
    ref_fwd, ref_bwd, ref_times = compute_references(runner, n_pairs, ref_iters, on_step=steps.tick)
    units_to_source_px = 8.0 * runner.downscale
    # Candidate-observable photometric warper: built once from the same decoded frames (no reference flow).
    # The luminance cache is preprocessed once per frame, so the per-iteration measured host cost is the
    # warp/statistics work only. Only the budget scan uses it.
    photo_closures = {}
    if budget_scan:
        warper = PhotometricWarper(frames[:n_pairs + 1], runner.downscale, runner.pad, runner.h, runner.w)
        for direction in directions:
            photo_closures[direction] = warper.closure(traversal_order(n_pairs, direction))

    results = {d: {} for d in directions}
    common_by_direction = {}
    for direction in directions:
        pairs, ref_fields = build_traversal(ref_fwd, ref_bwd, direction)
        order = traversal_order(n_pairs, direction)
        for method in methods:
            if budget_scan:
                make_stepper = runner.scan_closure(order)
                log(f"[flow-init] {direction}/{method} budget scan over {n_pairs} pairs "
                    f"(history {iters} iters, scan to {budget_max_iters})")
                rows = replay_budget_scan(pairs, ref_fields, method, make_stepper, iters, budget_max_iters,
                                          budget_thresholds, min_coverage, on_step=steps.tick,
                                          photometric=photo_closures.get(direction))
            else:
                refine = runner.refine_closure(order, iters)
                log(f"[flow-init] {direction}/{method} over {n_pairs} pairs ({iters} iters)")
                rows = replay(pairs, ref_fields, method, refine, min_coverage, on_step=steps.tick)
            payload = {
                "per_pair": rows,
                "summary": summarize(rows, iters, units_to_source_px),
            }
            if budget_scan:
                payload["budget_scan"] = {
                    "oracle": True, "history_iters": int(iters), "max_iters": int(budget_max_iters),
                    "thresholds": [float(t) for t in budget_thresholds],
                    "summary": summarize_budget_scan(rows, iters, budget_max_iters, budget_thresholds,
                                                     units_to_source_px),
                    "candidate_observable": ("per-iteration flow_update_low (|low - previous_low| mean/p95/"
                                             "max in feature-grid px) and photometric_low (robust luminance"
                                             " residual after warping the preprocessed source toward the "
                                             "target at the candidate's own flow) computed only from the "
                                             "candidate's own state and the decoded/preprocessed frames; no "
                                             "stop rule implemented or selected yet"),
                    "note": ("offline oracle scan: per-iteration low-grid EPE against the cold high-iteration "
                             "reference; history is always the state at history_iters; flow_update_low and "
                             "photometric_low are candidate-observable diagnostics, not stopping rules; no "
                             "deployable stopping rule and no online/production readiness claim"),
                }
            results[direction][method] = payload
        common = common_targets(results[direction])
        common_by_direction[direction] = common
        for method in methods:
            rows = results[direction][method]["per_pair"]
            results[direction][method]["summary_common"] = summarize(rows, iters, units_to_source_px, common)
            if budget_scan:
                results[direction][method]["budget_scan"]["summary_common"] = summarize_budget_scan(
                    rows, iters, budget_max_iters, budget_thresholds, units_to_source_px, common)
    out = {
        "n_pairs": int(n_pairs),
        "feature_h": int(runner.h),
        "feature_w": int(runner.w),
        "pad": list(runner.pad) if runner.pad is not None else None,
        "units_to_source_px": float(units_to_source_px),
        "feature_encode_s": float(runner.prep_s),
        "steps": int(steps.step),
        "reference": {
            "kind": "cold_high_iteration",
            "iters": int(ref_iters),
            "directions": requested_directions,
            "per_pair_refine_s": [float(t) for t in ref_times],
            "refine_s_mean": float(np.mean(ref_times)) if ref_times else None,
            "note": ("cold high-iteration RAFT with the same weights/features as the candidates; used as the "
                     "endpoint-error target, not as optical-flow ground truth"),
        },
        "common_targets": common_by_direction,
        "results": results,
    }
    if cycle_scan:
        out["cycle_scan"] = run_cycle_scan_block(runner, ref_fwd, ref_bwd, cycle_max_iters,
                                                 steps.tick, units_to_source_px, log=log)
    if budget_scan:
        out["budget_scan"] = {
            "enabled": True, "oracle": True, "history_iters": int(iters),
            "max_iters": int(budget_max_iters), "thresholds": [float(t) for t in budget_thresholds],
            "metric": "mean feature-grid EPE of the refined low-resolution flow against the cold reference",
            "candidate_observable": {
                "flow_update_low": ("per-iteration |low - previous_low| mean/p95/max in feature-grid px, "
                                    "computed only from the candidate's own initialization/current/previous "
                                    "prediction (the cold reference does not participate)"),
                "photometric_low": ("per-iteration robust absolute luminance-residual statistics after "
                                    "warping the preprocessed source frame toward the target at the "
                                    "candidate's own flow; pad/out-of-bounds/non-finite samples are "
                                    "excluded and a global-median brightness offset is removed, but the "
                                    "cold reference does not participate"),
                "stopping_rule": ("none implemented or selected; these diagnostics could support future "
                                  "convergence stopping but the scan is still an offline oracle comparison to "
                                  "the cold reference"),
            },
            "note": ("oracle-based, reference-derived and offline: it scans against a cold reference the "
                     "pipeline would not have, is not a deployable stopping rule, and claims no speedup or "
                     "readiness; flow_update_low and photometric_low are candidate-observable diagnostics only"),
        }
    return out


# --------------------------------------------------------------------------------------
# frames, caching and provenance
# --------------------------------------------------------------------------------------

def decode_frames(source: str, start: Optional[float], duration: Optional[float], vf: Optional[str],
                  synth: Optional[dict] = None):
    """Decode (or synthesize) the exact frames replayed by every candidate. Returns (frames, fps, info)."""
    from mlxstab import media
    if source == "synthetic":
        from mlxstab import synth as synth_mod
        s = dict(synth or {})
        W, H = int(s.get("w", 96)), int(s.get("h", 64))
        T = int(s.get("frames", 6))
        vid, _ = synth_mod.render(T=T, W=W, H=H, seed=int(s.get("seed", 0)))
        frames = [np.ascontiguousarray(f) for f in vid]
        return frames, float(s.get("fps", 30.0)), {
            "path_type": "synthetic", "sha256": None, "probe": None,
            "synthetic": {"frames": T, "width": W, "height": H, "seed": int(s.get("seed", 0)), "fps": float(s.get("fps", 30.0))},
            "window": {"start_s": None, "duration_s": None},
        }
    path = Path(source)
    info = media.probe(path)
    gen, fps, (W, H) = media.decode(path, info, start, duration, vf=vf)
    frames = [np.ascontiguousarray(f) for f in gen]
    return frames, float(fps), {
        "path": str(path), "path_type": "file", "sha256": MF.sha256_file(path), "probe": info,
        "window": {"start_s": start, "duration_s": duration}, "vf": vf,
    }


def _probe_frames_shape(frames):
    a = frames[0]
    return {"count": int(len(frames)), "height": int(a.shape[0]), "width": int(a.shape[1]), "channels": int(a.shape[2])}


def _request_identity(source: str, start, duration, vf, synth):
    """The request fingerprint a frame cache must match before it is trusted."""
    if source == "synthetic":
        return {"source_path_type": "synthetic", "source_sha256": None, "window": None, "vf": None,
                "synth": dict(synth or {})}
    p = Path(source)
    return {"source_path_type": "file", "source_path": str(p), "source_sha256": MF.sha256_file(p),
            "window": {"start_s": start, "duration_s": duration}, "vf": vf, "synth": None}


def load_or_decode(source: str, start, duration, vf, synth, cache_path: Optional[str], log=print):
    """Decode once, optionally persist/restore the exact uint8 frame stack under ``out/`` for replay.

    A cache is only trusted when its recorded request fingerprint (source hash/path, window, filter, synth
    parameters) matches the current request *and* its frame digest matches its own contents; otherwise it is
    refused with an error (never silently mixed). The digest is over the exact bytes every candidate replays.
    """
    request = _request_identity(source, start, duration, vf, synth)
    if cache_path and Path(cache_path).exists():
        with np.load(cache_path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"].item()))
            if meta.get("schema") != FRAME_CACHE_SCHEMA:
                raise ValueError(f"frame cache {cache_path} has schema {meta.get('schema')!r}, expected {FRAME_CACHE_SCHEMA!r}")
            cache_request = {k: meta.get(k) for k in request}
            if MF.json_safe(cache_request) != MF.json_safe(request):
                raise ValueError(f"frame cache {cache_path} was built for a different request; refusing to mix it")
            frames = [np.ascontiguousarray(x) for x in z["frames"]]
        if meta.get("frames_sha256") != frames_sha256(frames):
            raise ValueError(f"frame cache {cache_path} digest does not match its contents")
        return frames, float(meta["fps"]), {"path_type": "file" if request["source_path_type"] == "file" else "synthetic",
                                            "cache_path": str(cache_path), "cache_meta": meta,
                                            "sha256": meta.get("source_sha256"), "probe": meta.get("probe"),
                                            "window": meta.get("window"), "vf": meta.get("vf"),
                                            "synthetic": meta.get("synth"),
                                            "frames_cached": int(meta.get("count", len(frames)))}

    frames, fps, info = decode_frames(source, start, duration, vf, synth)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        meta = _probe_frames_shape(frames)
        meta.update(request)
        meta.update({"schema": FRAME_CACHE_SCHEMA, "probe": info.get("probe"), "fps": float(fps),
                     "frames_sha256": frames_sha256(frames)})
        np.savez(cache_path, meta=np.array(json.dumps(meta, sort_keys=True)), frames=np.stack(frames))
    return frames, fps, info


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _default_out_dir(source: str, start, duration):
    from datetime import datetime
    tag = "synthetic" if source == "synthetic" else Path(source).stem[:40]
    win = f"_t{start:g}+{duration:g}" if (start is not None or duration is not None) else ""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    import re
    tag = re.sub(r"[^0-9A-Za-z._-]+", "_", tag)
    return DEFAULT_OUT / f"{stamp}_{tag}{win}"


def build_parser():
    p = argparse.ArgumentParser(prog="bench/flow_prediction_real.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="synthetic", help="'synthetic' or a local video path")
    p.add_argument("--start", type=float, help="source window start (seconds)")
    p.add_argument("--duration", type=float, help="source window duration (seconds)")
    p.add_argument("--vf", help="extra ffmpeg decode filter (e.g. fps=30000/1001)")
    p.add_argument("--out-dir", help="output directory (default under ignored out/bench/flow_prediction_real)")
    p.add_argument("--overwrite", action="store_true", help="allow reusing an existing out-dir")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--iters", type=int, default=DEFAULT_ITERS, help="refinement iterations for candidates")
    p.add_argument("--ref-iters", type=int, default=DEFAULT_REF_ITERS, help="cold high-iteration reference iterations")
    p.add_argument("--budget-scan", action="store_true",
                   help="opt-in offline oracle scan: refine each pair one iteration at a time from the policy "
                        "initialization and record first-threshold crossings against the cold reference; each "
                        "iteration also records candidate-observable flow_update_low and photometric_low "
                        "diagnostics (no stop rule is implemented or selected)")
    p.add_argument("--budget-max-iters", type=int, default=DEFAULT_BUDGET_MAX_ITERS,
                   help="max scan iterations per pair (>= --iters); no crossing by then is censored")
    p.add_argument("--budget-thresholds", default=",".join(f"{t:g}" for t in DEFAULT_BUDGET_THRESHOLDS),
                   help="comma-separated feature-grid EPE thresholds (default 0.005,0.01,0.02)")
    p.add_argument("--cycle-scan", action="store_true",
                   help="opt-in cold-only bidirectional cycle-consistency pilot: refine the forward and "
                        "backward zero-initialized candidates of each pair together one iteration at a time "
                        "and record the candidate-observable cycle residual (no reference flow) plus separate "
                        "oracle EPE; cycle-only, cannot be combined with --budget-scan")
    p.add_argument("--cycle-max-iters", type=int, default=DEFAULT_CYCLE_MAX_ITERS,
                   help="max paired cycle-scan iterations per pair (default 24)")
    p.add_argument("--downscale", type=int, default=2, help="flow input downscale (1 = full resolution)")
    p.add_argument("--flow-dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--corr-dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--methods", default=",".join(METHODS), help="comma-separated initialization policies")
    p.add_argument("--directions", default=",".join(DIRECTIONS), help="forward,backward (subset allowed)")
    p.add_argument("--min-coverage", type=float, default=FP.MIN_COVERAGE)
    p.add_argument("--max-pairs", type=int, default=None, help="cap the number of consecutive pairs")
    p.add_argument("--cache-frames", default=None, help="persist/restore the decoded uint8 frames (npz)")
    p.add_argument("--lock", choices=["auto", "on", "off"], default="auto",
                   help="shared heavy lock: auto=on for file sources, off for synthetic (held for the whole run)")
    p.add_argument("--lock-chunk", type=int, default=0,
                   help="release/re-acquire the lock every N completed frame/pair steps (0 = hold throughout)")
    p.add_argument("--lock-pause", type=float, default=0.0,
                   help="seconds to yield for waiting jobs between lock chunks (etiquette ~12 s)")
    p.add_argument("--synth-frames", type=int, default=6)
    p.add_argument("--synth-w", type=int, default=96)
    p.add_argument("--synth-h", type=int, default=64)
    p.add_argument("--synth-fps", type=float, default=30.0)
    p.add_argument("--synth-seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="print resolved options and exit")
    return p


def _parse_list(text):
    return [v.strip() for v in str(text).split(",") if v.strip()]


def _experiment_identity(source_info, frames_sha, fps, cfg, weights_sha, code):
    ident = {
        "source": {"sha256": source_info.get("sha256"), "path_type": source_info.get("path_type"),
                   "window": source_info.get("window"), "vf": source_info.get("vf"),
                   "frames_sha256": frames_sha, "fps": fps},
        "weights_sha256": weights_sha,
        "preprocessing": {"downscale": cfg["downscale"], "flow_dtype": cfg["flow_dtype"],
                          "corr_dtype": cfg["corr_dtype"], "iters": cfg["iters"], "ref_iters": cfg["ref_iters"]},
        "methods": cfg["methods"], "directions": cfg["directions"], "min_coverage": cfg["min_coverage"],
        "code": {"revision": (code or {}).get("revision"), "files": (code or {}).get("files"),
                 "dirty_relevant": (code or {}).get("dirty_relevant")},
    }
    if cfg.get("budget") is not None:
        ident["budget"] = cfg["budget"]
    if cfg.get("cycle") is not None:
        ident["cycle"] = cfg["cycle"]
    return ident


def run(opts, log=print) -> dict:
    """Run one benchmark and write an atomic ``manifest.json``. Returns the manifest."""
    methods = _parse_list(opts.methods)
    directions = _parse_list(opts.directions)
    synth = {"frames": opts.synth_frames, "w": opts.synth_w, "h": opts.synth_h,
             "fps": opts.synth_fps, "seed": opts.synth_seed}
    cfg = {"iters": opts.iters, "ref_iters": opts.ref_iters, "downscale": opts.downscale,
           "flow_dtype": opts.flow_dtype, "corr_dtype": opts.corr_dtype, "methods": methods,
           "directions": directions, "min_coverage": opts.min_coverage,
           "lock_chunk": int(opts.lock_chunk), "lock_pause_s": float(opts.lock_pause)}
    thresholds = parse_thresholds(opts.budget_thresholds)
    if opts.budget_scan and opts.cycle_scan:
        raise ValueError("--cycle-scan is a cycle-only mode and cannot be combined with --budget-scan")
    if opts.budget_scan:
        cfg["budget"] = {"scan": True, "max_iters": int(opts.budget_max_iters),
                         "thresholds": [float(t) for t in thresholds]}
    if opts.cycle_scan:
        if int(opts.cycle_max_iters) < 1:
            raise ValueError("--cycle-max-iters must be >= 1")
        cfg["cycle"] = {"scan": True, "max_iters": int(opts.cycle_max_iters)}

    if opts.source != "synthetic" and not Path(opts.source).exists():
        raise FileNotFoundError(f"source not found: {opts.source}")
    if not Path(opts.weights).exists():
        raise FileNotFoundError(f"weights not found: {opts.weights}")
    if opts.iters < 1 or opts.ref_iters < 1:
        raise ValueError("--iters and --ref-iters must be >= 1")
    if opts.downscale < 1:
        raise ValueError("--downscale must be >= 1")
    if opts.lock_chunk < 0:
        raise ValueError("--lock-chunk must be >= 0")
    if opts.lock_pause < 0:
        raise ValueError("--lock-pause must be >= 0")
    if opts.budget_scan:
        if opts.budget_max_iters < 1:
            raise ValueError("--budget-max-iters must be >= 1")
        if opts.budget_max_iters < opts.iters:
            raise ValueError("--budget-max-iters must be >= --iters to preserve the fixed-budget history")
    if opts.cycle_scan and opts.cycle_max_iters < 1:
        raise ValueError("--cycle-max-iters must be >= 1")

    run_dir = Path(opts.out_dir) if opts.out_dir else _default_out_dir(opts.source, opts.start, opts.duration)
    man_path = run_dir / "manifest.json"
    if not opts.overwrite and man_path.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence {man_path}; pass --overwrite to reuse it")
    run_dir.mkdir(parents=True, exist_ok=True)

    man = {
        "schema": SCHEMA, "created": MF.now_iso(), "status": "running", "stage": "validate", "error": None,
        "run_id": run_dir.name, "scope": SCOPE,
        "source": None, "weights": {"path": str(opts.weights), "sha256": None},
        "config": cfg, "config_digest": None, "experiment_identity": None, "content_key": None,
        "grid": None, "reference": None, "results": None, "timing_s": {}, "lock": None, "steps": None,
        "code": None, "environment": None, "hardware": None, "limitations": list(LIMITATIONS),
        "command": [sys.executable, *sys.argv], "manifest": str(man_path),
    }
    _write(man, man_path, required=True)

    lock = None
    try:
        enabled_lock = lock_enabled(opts.source, opts.lock)
        lock = HeavyLock(what=f"mlxstab flow_prediction_real {Path(opts.source).name}", enabled=enabled_lock,
                         chunk=opts.lock_chunk, pause_s=opts.lock_pause)
        # One monotonic gate shared by feature preparation, reference generation and candidate replay. The
        # lock is only gated when chunking is requested, so a default synthetic run never touches it.
        gate = lock.on_step if opts.lock_chunk else None
        man["stage"] = "provenance"
        man["code"] = MF.code_fingerprint(ROOT)
        man["environment"] = MF.env_versions()
        man["hardware"] = MF.hardware_info()
        _write(man, man_path)

        man["stage"] = "decode"
        load = lambda: load_or_decode(opts.source, opts.start, opts.duration, opts.vf, synth,  # noqa: E731
                                      opts.cache_frames, log=log)
        (frames, fps, source_info), frames_load_s = load_under_lock(lock, load)
        frames_sha = frames_sha256(frames)
        if len(frames) < 2:
            raise ValueError(f"need at least two frames, decoded {len(frames)}")
        man["timing_s"]["frames_load"] = frames_load_s
        man["source"] = {**source_info, "frames_sha256": frames_sha, "decoded_frames": len(frames),
                         "fps": fps, "height": int(frames[0].shape[0]), "width": int(frames[0].shape[1])}
        man["weights"]["sha256"] = MF.sha256_file(opts.weights)
        man["config_digest"] = hashlib.sha256(json.dumps(MF.json_safe(cfg), sort_keys=True,
                                                        separators=(",", ":")).encode()).hexdigest()
        identity = _experiment_identity(man["source"], frames_sha, fps, cfg, man["weights"]["sha256"], man["code"])
        man["experiment_identity"] = identity
        man["content_key"] = MF.content_key(identity)
        man["lock"] = lock.status()
        _write(man, man_path)

        man["stage"] = "model"
        from mlxstab.raft.model import load_raft  # noqa: F401  (fail early with a clear stage)
        runner = RaftRunner(opts.weights, opts.downscale, opts.flow_dtype, opts.corr_dtype)
        log(f"[flow-init] model loaded; {len(frames)} frames, {opts.downscale}x downscale")
        man["stage"] = "benchmark"
        t0 = time.perf_counter()
        res = run_benchmark(runner, frames, methods=methods, directions=directions, iters=opts.iters,
                            ref_iters=opts.ref_iters, min_coverage=opts.min_coverage, max_pairs=opts.max_pairs,
                            log=log, on_step=gate, budget_scan=opts.budget_scan,
                            budget_max_iters=opts.budget_max_iters, budget_thresholds=thresholds,
                            cycle_scan=opts.cycle_scan, cycle_max_iters=opts.cycle_max_iters)
        man["timing_s"]["benchmark"] = time.perf_counter() - t0
        man["steps"] = res.get("steps")
        if res.get("budget_scan") is not None:
            man["budget_scan"] = res["budget_scan"]
        if res.get("cycle_scan") is not None:
            man["cycle_scan"] = res["cycle_scan"]
        man["grid"] = {"feature_h": res["feature_h"], "feature_w": res["feature_w"], "pad": res["pad"],
                       "units_to_source_px": res["units_to_source_px"],
                       "feature_encode_s": res["feature_encode_s"],
                       "note": "network feature grid (padded image / 8); EPE is in feature-grid units"}
        man["reference"] = res["reference"]
        man["common_targets"] = res["common_targets"]
        man["results"] = res["results"]
        man["n_pairs"] = res["n_pairs"]
        man["status"] = "success"
        man["stage"] = "success"
    except BaseException as e:  # noqa: BLE001 - every failure must be reviewable
        man["status"] = "cancelled" if isinstance(e, (KeyboardInterrupt, SystemExit)) else "failure"
        man["error"] = {"type": type(e).__name__, "message": str(e), "stage": man.get("stage")}
        log(f"[flow-init] {man['status']} at {man['stage']}: {type(e).__name__}: {e}")
    finally:
        if lock is not None:
            lock.release()
    if lock is not None:
        man["lock"] = lock.status()
    _write(man, man_path, required=True)
    return man


def _write(man, path, required=False):
    man["updated"] = MF.now_iso()
    try:
        MF.write_manifest(path, man)
    except Exception:
        if required:
            raise


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.dry_run:
        cfg = {"iters": a.iters, "ref_iters": a.ref_iters, "downscale": a.downscale,
               "flow_dtype": a.flow_dtype, "corr_dtype": a.corr_dtype,
               "methods": _parse_list(a.methods), "directions": _parse_list(a.directions),
               "min_coverage": a.min_coverage, "lock": a.lock, "lock_chunk": a.lock_chunk,
               "lock_pause_s": a.lock_pause, "budget_scan": a.budget_scan,
               "budget_max_iters": a.budget_max_iters, "budget_thresholds": _parse_list(a.budget_thresholds),
               "cycle_scan": a.cycle_scan, "cycle_max_iters": a.cycle_max_iters,
               "source": a.source, "start": a.start, "duration": a.duration,
               "out_dir": a.out_dir or str(_default_out_dir(a.source, a.start, a.duration))}
        print(json.dumps(cfg, indent=1, sort_keys=True))
        return 0
    try:
        man = run(a, log=lambda m: print(m, file=sys.stderr))
    except Exception as e:  # overwrite refusal, missing source, option error
        print(json.dumps({"status": "failure", "error": {"type": type(e).__name__, "message": str(e)}},
                         sort_keys=True))
        return 1
    if man["status"] == "success":
        for direction, methods in man["results"].items():
            common = man.get("common_targets", {}).get(direction, [])
            print(f"\n{direction} (common pairs: {common[0] if common else '-'}..{common[-1] if common else '-'})")
            for method, payload in methods.items():
                s = payload["summary_common"]
                mean = "n/a" if s["refined_epe_mean_low"] is None else f"{s['refined_epe_mean_low']:.4g}"
                cov = "n/a" if s["coverage_mean"] is None else f"{s['coverage_mean']:.3f}"
                print(f"  {method:22s} common_pairs={s['usable_pairs']:>3d} refined_epe_low={mean:>8s} "
                      f"coverage={cov:>6s} fallbacks={s['fallbacks']}")
                bs = payload.get("budget_scan")
                if bs is not None:
                    sc = bs["summary_common"]
                    reach = " ".join(
                        f"<={p['threshold']:g}:{'n/a' if p['reach_rate'] is None else round(p['reach_rate'], 2)}"
                        for p in sc["per_threshold"])
                    print(f"  {'':22s} budget_scan(oracle) reach {reach} "
                          f"full_scan update_s={sc['full_scan_update_s_total']:.3f} "
                          f"epe_check_s={sc['full_scan_epe_check_s_total']:.3f} "
                          f"photometric_s={sc.get('full_scan_photometric_check_s_total', 0.0):.3f}")
                    for p in sc["per_threshold"]:
                        if p["total_s_mean"] is None:
                            continue
                        print(f"  {'':22s}   prefix <={p['threshold']:g}: "
                              f"steps_mean={p['steps_consumed_mean']:.2f} "
                              f"update_s={p['update_s_total']:.3f} "
                              f"epe_check_s={p['epe_check_s_total']:.3f} "
                              f"total_s={p['total_s_total']:.3f}")
        cs = man.get("cycle_scan")
        if cs is not None:
            s = cs["results"]["bidirectional"]["cold"]["summary"]
            print(f"\nbidirectional cold cycle scan ({cs['max_iters']} iters, {s['usable_pairs']} pairs)")
            cyc = "n/a" if s["cycle_low_resid_mean"] is None else f"{s['cycle_low_resid_mean']:.4g}"
            vf = "n/a" if s["valid_frac_mean"] is None else f"{s['valid_frac_mean']:.3f}"
            print(f"  cycle_low_resid_mean={cyc} valid_frac={vf} "
                  f"stopped_non_finite={s['stopped_non_finite']}")
            print(f"  full_scan update_s={s['full_scan_update_s_total']:.3f} "
                  f"cycle_check_s={s['full_scan_cycle_check_s_total']:.3f} "
                  f"oracle_epe_check_s={s['full_scan_epe_check_s_total']:.3f}")
    print(json.dumps({"status": man["status"], "manifest": man["manifest"], "error": man["error"]}, sort_keys=True))
    return 0 if man["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
