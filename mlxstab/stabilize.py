"""No-crop video stabilizer: RAFT flow -> robust global homography + mesh/row residuals -> L1-optimal camera paths -> one
Lanczos resample to the full original frame, plus a per-frame mask of pixels that have no source."""
import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import mlx.core as mx
from . import geometry as G, motion as MO, smooth as SM, media, warpmetrics
from .warp import sample, valid_mask
from .raft.model import load_raft, raft_flow


def _validate_sampled_border_frac(value):
    """Validate ``Config.max_sampled_border_frac``: ``None`` disables; otherwise finite and in ``[0, 1)``."""
    if value is None:
        return None
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"max_sampled_border_frac must be finite, got {value!r}")
    if not (0.0 <= v < 1.0):
        raise ValueError(f"max_sampled_border_frac must be in [0, 1), got {value!r}")
    return v


def _validate_crop_zoom(value):
    """Validate ``Config.crop_zoom``: finite and ``>= 1`` (``1.0`` is the exact no-op identity)."""
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"crop_zoom must be finite, got {value!r}")
    if v < 1.0:
        raise ValueError(f"crop_zoom must be >= 1, got {value!r}")
    return v


def _crop_zoom_matrix(W, H, crop_zoom):
    """Homogeneous output-space zoom ``Z`` mapping ``x -> c + (x - c)/crop_zoom`` with ``c = (W/2-.5, H/2-.5)``.

    Composed on the right of the global map (``G @ Z``) so the isotropic center zoom acts on output coordinates
    before the output-to-source homography. ``crop_zoom == 1.0`` yields the exact identity matrix."""
    s = float(crop_zoom)
    if s == 1.0:
        return np.eye(3)
    cx, cy = W / 2 - .5, H / 2 - .5
    k = 1.0 - 1.0 / s
    return np.array([[1.0 / s, 0.0, cx * k],
                     [0.0, 1.0 / s, cy * k],
                     [0.0, 0.0, 1.0]], np.float64)


@dataclass
class Config:
    smooth: str = "l1"             # "l1" (Grundmann) or "gauss"
    sigma: float = 15.0            # gaussian sigma in frames (smooth="gauss")
    l1_weights: tuple = (10.0, 1.0, 100.0)
    budget: float = 0.04           # max path deviation as a fraction of frame width = the border budget
    mesh: tuple = (16, 9)          # cells (cols, rows) of the residual mesh; (0, 0) disables
    mesh_budget: float = 0.012     # residual correction cap, fraction of width
    rs_rows: int = 18              # row bands for rolling-shutter correction; 0 disables
    flow_iters: int = 24
    flow_downscale: int = 1        # 1 = full resolution flow
    warm_start: bool = True
    step: int = 8                  # flow sampling grid for motion fitting
    kernel: str = "lanczos3"       # final resampling kernel
    zoom: float = 1.0              # optional output zoom composed into the same warp
    seed: int = 0
    analysis_stride: int = 1       # analyse every n-th frame and interpolate the path (2 = source frame rate for 59.94p fields)
    flow_dtype: str = "float32"    # "float16" runs encoders + update block in half precision (opt-in)
    corr_dtype: str = "float32"    # "float16" stores the correlation pyramid in half precision (opt-in)
    warm_iters: int = 0            # iterations once warm-started (0 = same as flow_iters)
    model: str = "similarity"      # global camera model: similarity | affine | homography (residuals go to the mesh)
    fidelity: float = 0.1          # L1 pull of the path towards the raw path (keeps the border small; 1.0 leaked jitter, 0 -> big border)
    warp_diag: bool = False        # opt-in: complete-warp shape/coverage diagnostics in the report (extra host math per frame)
    max_sampled_border_frac: float | None = None   # opt-in: cap the sampled out-of-bounds fraction (finite, [0,1)); None disables
    crop_zoom: float = 1.0         # opt-in: isotropic center zoom on output coords, composed before the output->source map (finite, >=1)

    def __post_init__(self):
        self.max_sampled_border_frac = _validate_sampled_border_frac(self.max_sampled_border_frac)
        self.crop_zoom = _validate_crop_zoom(self.crop_zoom)


def load_roll(path):
    """Roll series (degrees, positive = counter-clockwise lean). JSON: [deg per frame] | [[t, deg(, conf)], ...] | {"series": ...}."""
    d = json.loads(Path(path).read_text()); d = d.get("series", d) if isinstance(d, dict) else d
    a = np.array(d, np.float64)
    return ("frames", a) if a.ndim == 1 else ("timed", a[:, :2])


def roll_per_frame(roll, T, fps):
    if roll is None: return np.zeros(T)
    kind, a = roll
    if kind == "frames":
        return np.interp(np.arange(T), np.arange(len(a)), a)
    return np.interp(np.arange(T) / fps, a[:, 0], a[:, 1])


class RaftFlow:
    """RAFT flow for the stabilizer.

    `bidir(a, b)` gives forward and backward flow of one frame pair as a single batch-2 refinement, with the encoders run once
    per frame (the previous call's features of `b` are reused as `a`), one host sync per pair, warm start from the previous pair.
    Options (all default to the exact fp32 path of the parity tests; see the README for their measured cost):
      dtype=mx.float16     encoders + update block in half precision (correlation volume still built from fp32 features)
      corr_dtype=float16   store the correlation pyramid in half precision
      warm_iters=N         iterations used once a warm start exists (first pair uses `iters`)
      downscale=2          estimate flow on a box-filtered half-resolution image, flow returned at full resolution."""
    def __init__(self, weights, iters=24, downscale=1, warm=True, corr_dtype=mx.float32, dtype=mx.float32, warm_iters=None):
        self.m = load_raft(weights)
        if dtype != mx.float32: self.m.to_dtype(dtype)
        self.iters, self.ds, self.warm, self.cd, self.warm_iters = iters, downscale, warm, corr_dtype, warm_iters
        self.prev = {}; self.prev_f = self.prev_b = None; self._last = None; self.syncs = 0

    def _prep(self, im):
        x = mx.array(im).astype(mx.float32)
        if self.ds > 1:
            H, W = x.shape[:2]; H2, W2 = H // self.ds, W // self.ds
            x = x[:H2 * self.ds, :W2 * self.ds].reshape(H2, self.ds, W2, self.ds, 3).mean(axis=(1, 3))
        return x

    def _restore(self, up, p, H, W):
        from .raft.model import unpad
        f = unpad(up, p)[0]
        if self.ds > 1:
            ys, xs = mx.meshgrid(mx.arange(H), mx.arange(W), indexing="ij")
            c = mx.stack([(xs + .5) / self.ds - .5, (ys + .5) / self.ds - .5], -1)[None].astype(mx.float32)
            f = sample(f[None], c, "bilinear", "border")[0] * self.ds
        return f

    def reset(self):
        """Start a new independent sequence (new clip, cut, or changed shape/preprocessing).

        Drops every piece of temporal state that could leak across a boundary: the legacy warm-start
        cache (`prev`), the bidirectional low-resolution warm-start flows (`prev_f`, `prev_b`) and the
        cached frame features (`_last`) that let consecutive pairs of one sequence share encoders. Call
        this exactly at sequence boundaries; never between frames of one sequence, which would discard
        feature reuse and warm starts. The diagnostic `syncs` counter is reset too, so it reports host
        syncs for the new sequence rather than the process lifetime.
        """
        self.prev = {}
        self.prev_f = self.prev_b = None
        self._last = None
        self.syncs = 0

    def __call__(self, a, b, key="f"):
        """One direction (legacy API): flow a -> b at full resolution, lazily evaluated by the caller."""
        from .raft.model import pad8
        H, W = a.shape[:2]
        A, p = pad8(self._prep(a)[None]); B, _ = pad8(self._prep(b)[None])
        init = self.prev.get(key) if self.warm else None
        low, up = self.m(A, B, self.iters, init, corr_dtype=self.cd)
        self.prev[key] = low; mx.eval(low, up); self.syncs += 1
        return self._restore(up, p, H, W)

    def bidir(self, a, b):
        from .raft.model import pad8
        H, W = a.shape[:2]
        if self._last is not None and self._last[0] is a: fa, p = self._last[1], self._last[2]
        else:
            x, p = pad8(self._prep(a)[None]); fa = self.m.frame_features(x)
        x, _ = pad8(self._prep(b)[None]); fb = self.m.frame_features(x)
        warm = self.warm and self.prev_f is not None
        it = (self.warm_iters or self.iters) if warm else self.iters
        lf, uf, lb, ub = self.m.flow_bidir(fa, fb, it, self.prev_f if warm else None, self.prev_b if warm else None, self.cd)
        self.prev_f, self.prev_b = lf, lb; self._last = (b, fb, p)
        return self._restore(uf, p, H, W), self._restore(ub, p, H, W)           # lazy: observe() evaluates them


@dataclass
class Motion:
    W: int
    H: int
    fps: float
    fits: list = field(default_factory=list)
    stride: int = 1            # frames between analysed pairs
    n_frames: int = 0          # frames seen


def analyze(frames, flow_fn, cfg, W, H, fps, log=None, masks=None, on_step=None, stride=1, threads=True):
    """Pass 1: motion fit for each consecutive analysed pair (every `stride`-th frame). `frames` yields uint8 [H,W,3];
    flow_fn(a, b, key) -> [H,W,2] mx, or flow_fn.bidir(a, b) -> (forward, backward). While the GPU computes the next flow, a worker
    thread fits the previous pair on the CPU."""
    from concurrent.futures import ThreadPoolExecutor
    mot = Motion(W, H, fps, stride=stride); prev = None; rng = np.random.default_rng(cfg.seed)
    gc, gr = cfg.mesh
    pool = ThreadPoolExecutor(1) if threads else None; pend = []
    fit = lambda obs: MO.fit_motion(obs, (max(gc, 1), max(gr, 1)), cfg.rs_rows, rng=rng, model=cfg.model)
    pm = None
    for i, fr in enumerate(frames):
        cm = next(masks) if masks is not None else None
        mot.n_frames = i + 1
        if i % stride: continue
        if on_step: on_step(i)                 # hook for callers that share the GPU (e.g. release/re-acquire a lock between chunks)
        if prev is not None:
            if hasattr(flow_fn, "bidir"): Ff, Bf = flow_fn.bidir(prev, fr)
            else: Ff = flow_fn(prev, fr, "f"); Bf = flow_fn(fr, prev, "b")
            obs = MO.observe(Ff, Bf, cfg.step)
            if cm is not None or pm is not None: MO.exclude_masked(obs, pm, cm)
            if pool: pend.append(pool.submit(fit, obs))
            else: mot.fits.append(fit(obs))
            while len(pend) > 2: mot.fits.append(pend.pop(0).result())
            if log and i % 20 == 0: log(f"flow {i}")
        prev = fr; pm = cm
    for f in pend: mot.fits.append(f.result())
    if pool: pool.shutdown()
    return mot


@dataclass
class Plan:
    G: list                    # per frame: 3x3 virtual-frame pixel -> source-frame pixel (global part)
    mesh_corr: np.ndarray      # [T, gr+1, gc+1, 2] source displacement correction
    row_corr: np.ndarray       # [T, nr, 2]
    raw_quads: np.ndarray      # [T,4,2] in reference coords
    path_quads: np.ndarray     # smoothed
    fallbacks: int = 0
    sampled_border_scale: float = 1.0        # clip-wide scale applied by the opt-in sampled border backoff (1.0 = untouched)
    sampled_border_worst_frac: float = 0.0   # worst sampled out-of-bounds fraction over frames at that scale
    sampled_border_measured: bool = False    # whether the sampled border probe ran (Config.max_sampled_border_frac was set)


PLAN_SCHEMA = "mlxstab.plan/2"   # schema tag stored inside every dumped Plan NPZ (bump on layout change)
PLAN_SCHEMA_V1 = "mlxstab.plan/1"   # legacy layout (no sampled border-backoff fields); still loadable, read-only
PLAN_SCHEMAS = (PLAN_SCHEMA_V1, PLAN_SCHEMA)
MOTION_SCHEMA = "mlxstab.motion/4"   # schema tag stored inside every dumped Motion NPZ (bump on layout change)
MOTION_SCHEMA_V3 = "mlxstab.motion/3"   # cycle maps, no fit-gate metadata; still loadable, read-only
MOTION_SCHEMA_V2 = "mlxstab.motion/2"   # support mass, no cycle maps; still loadable, read-only
MOTION_SCHEMA_V1 = "mlxstab.motion/1"   # legacy layout (no support-mass / cycle / fit-gate arrays); still loadable, read-only
MOTION_SCHEMAS = (MOTION_SCHEMA_V1, MOTION_SCHEMA_V2, MOTION_SCHEMA_V3, MOTION_SCHEMA)
MOTION_SCHEMAS_WITH_SUPPORT = (MOTION_SCHEMA_V2, MOTION_SCHEMA_V3, MOTION_SCHEMA)   # /1 predates support mass
MOTION_SCHEMAS_WITH_CYCLE = (MOTION_SCHEMA_V3, MOTION_SCHEMA)                        # /1 and /2 predate cycle maps


def _support_mass_or_zeros(mass, shape):
    """Shape-correct support mass for one Fit: the preserved array, or zeros when the Fit carries none.

    ``None`` means the field was not recorded (a manually constructed Fit, or a legacy payload), not that
    support is zero; for a fresh dump the zero array is the honest "no usable samples recorded" placeholder."""
    if mass is None:
        return np.zeros(shape, np.float64)
    return np.asarray(mass, np.float64).reshape(shape)


def _cycle_maps_or_zeros(fit, shape):
    """The ``(err_rms, inconsistency_frac, sample_mass, available)`` cycle maps for one Fit.

    A Fit with any of the three map fields unset (a manually constructed Fit) serializes shape-correct zero
    maps with ``available=False``, so availability is explicit instead of being inferred from an all-zero map.
    The maps are a forward-backward **cycle-consistency** diagnostic only: not a true correspondence error,
    not a probability and not a calibrated uncertainty."""
    rms = getattr(fit, "cycle_err_rms", None)
    frac = getattr(fit, "cycle_inconsistency_frac", None)
    mass = getattr(fit, "cycle_sample_mass", None)
    if rms is None or frac is None or mass is None:
        return np.zeros(shape, np.float64), np.zeros(shape, np.float64), np.zeros(shape, np.float64), False
    return (np.asarray(rms, np.float64).reshape(shape), np.asarray(frac, np.float64).reshape(shape),
            np.asarray(mass, np.float64).reshape(shape), True)


def _fit_meta_or_unavailable(fit):
    """The ``(n_valid_samples, n_ransac_inliers, reason_code, available)`` fit-gate metadata for one Fit.

    The three fields are recorded together by ``mlxstab.motion.fit_motion``. The reason is serialized as its
    stable numeric code (``mlxstab.motion.FALLBACK_REASON_CODES``); all payload fields are non-object numeric
    arrays. A Fit that carries none of them (a manually constructed Fit) serializes the numeric sentinels
    ``-1, -1, -1`` and ``available=False``, so the payload says "not recorded" instead of inventing counts or a
    reason. Legacy ``/1``-``/3`` payloads have no such keys and load with these fields ``None``."""
    n_valid = getattr(fit, "n_valid_samples", None)
    n_inliers = getattr(fit, "n_ransac_inliers", None)
    reason = getattr(fit, "fallback_reason", None)
    if n_valid is None or n_inliers is None or reason is None:
        return (np.int64(-1), np.int64(-1), np.int64(MO.FALLBACK_REASON_UNAVAILABLE_CODE), False)
    return (np.int64(int(n_valid)), np.int64(int(n_inliers)),
            np.int64(MO.FALLBACK_REASON_CODES[str(reason)]), True)


def save_motion_npz(mot, path, cfg=None):
    """Write a computed ``Motion`` (per-pair fitted motion) to a compressed, non-pickle NPZ.

    Only plain numeric arrays are stored (no object dtype), so the file loads with
    ``np.load(path, allow_pickle=False)``. Contents: schema tag, frame dimensions, total frame count
    (``n_frames``), analysis fps and stride, the fit count, and the stacked per-pair fit fields
    (``H [n_fits,3,3]``, ``mesh [n_fits,gr+1,gc+1,2]``, ``rows [n_fits,nr,2]``, ``inlier_frac [n_fits]``,
    ``mover_frac [n_fits]``, ``fallback [n_fits]``) plus the fit-gate diagnostics
    (``n_valid_samples [n_fits]``, ``n_ransac_inliers [n_fits]``, the numeric ``fallback_reason_code [n_fits]``,
    and the per-fit ``fit_meta_available [n_fits]`` flag), the observation-coverage support masses
    (``mesh_support_mass [n_fits,gr+1,gc+1]``, ``row_support_mass [n_fits,nr]``) and the forward-backward
    cycle-consistency maps (``cycle_err_rms``, ``cycle_inconsistency_frac``, ``cycle_sample_mass``, each
    ``[n_fits,gr+1,gc+1]``, plus the per-fit ``cycle_map_available [n_fits]`` flag).

    The fit-gate diagnostics name *why* ``fit_motion`` returned an identity fallback. ``n_valid_samples`` and
    ``n_ransac_inliers`` are the valid-sample count and RANSAC-inlier count of the initial all-valid solve;
    ``fallback_reason_code`` is the stable numeric code of one of ``mlxstab.motion.FALLBACK_REASONS`` (0 =
    ``none`` for a successful fit, 1-3 the failing gate, ``-1`` unavailable). A Fit
    built without them (a manually constructed Fit) serializes ``-1, -1, -1`` tagged
    ``fit_meta_available=False``, so the payload never invents counts or a reason. These fields are diagnostics
    only and do not feed fitting, stabilization or rendering.

    Support mass is the non-robust kernel-weighted count of usable non-mover flow samples at each mesh
    vertex / row band, recorded before diffusion; it is an observation-coverage diagnostic, **not** a
    probability and **not** flow confidence. A Fit built without one (``None``) serializes a shape-correct
    **zero** array, so every dumped ``mlxstab.motion/4`` payload carries both arrays.

    The cycle maps are a forward-backward **cycle-consistency** diagnostic only: ``cycle_err_rms`` is the
    spatially weighted RMS of ``||f + sampled_b||``, ``cycle_inconsistency_frac`` the weighted fraction of
    in-bounds samples failing the existing forward-backward tolerance and ``cycle_sample_mass`` the spatial
    sample coverage. They are NOT a true correspondence error, NOT a probability and NOT calibrated
    uncertainty. A Fit with ``None`` cycle fields serializes zero maps tagged ``cycle_map_available=False``.

    The mesh/row shapes come from the fits. When there are **no** fits, they are taken from
    ``cfg.mesh``/``cfg.rs_rows`` (the same ``max(..., 1)`` shapes ``analyze`` would have produced), so a
    zero-fit Motion still dumps a well-formed payload rather than an undefined shape."""
    fits = list(mot.fits)
    n = len(fits)
    if fits:
        gr, gc = np.asarray(fits[0].mesh).shape[0] - 1, np.asarray(fits[0].mesh).shape[1] - 1
        nr = np.asarray(fits[0].rows).shape[0]
    else:
        mgc, mgr = (getattr(cfg, "mesh", (0, 0)) if cfg is not None else (0, 0))
        gc, gr = max(int(mgc), 1), max(int(mgr), 1)
        nr = max(int(getattr(cfg, "rs_rows", 0)) if cfg is not None else 0, 1)
    cycle = [_cycle_maps_or_zeros(f, (gr + 1, gc + 1)) for f in fits]
    fit_meta = [_fit_meta_or_unavailable(f) for f in fits]
    np.savez_compressed(
        path,
        schema=np.asarray(MOTION_SCHEMA), width=np.int64(mot.W), height=np.int64(mot.H),
        n_frames=np.int64(mot.n_frames), fps=np.float64(mot.fps), stride=np.int64(getattr(mot, "stride", 1)),
        n_fits=np.int64(n),
        H=np.asarray([np.asarray(f.H, np.float64) for f in fits], np.float64).reshape(n, 3, 3),
        mesh=np.asarray([np.asarray(f.mesh, np.float64) for f in fits], np.float64).reshape(n, gr + 1, gc + 1, 2),
        rows=np.asarray([np.asarray(f.rows, np.float64) for f in fits], np.float64).reshape(n, nr, 2),
        inlier_frac=np.asarray([f.inlier_frac for f in fits], np.float64),
        mover_frac=np.asarray([f.mover_frac for f in fits], np.float64),
        fallback=np.asarray([bool(f.fallback) for f in fits], np.bool_),
        n_valid_samples=np.asarray([m[0] for m in fit_meta], np.int64).reshape(n),
        n_ransac_inliers=np.asarray([m[1] for m in fit_meta], np.int64).reshape(n),
        fallback_reason_code=np.asarray([m[2] for m in fit_meta], np.int64).reshape(n),
        fit_meta_available=np.asarray([m[3] for m in fit_meta], np.bool_).reshape(n),
        mesh_support_mass=np.asarray([_support_mass_or_zeros(getattr(f, "mesh_support_mass", None),
                                                             (gr + 1, gc + 1)) for f in fits],
                                     np.float64).reshape(n, gr + 1, gc + 1),
        row_support_mass=np.asarray([_support_mass_or_zeros(getattr(f, "row_support_mass", None), (nr,))
                                     for f in fits], np.float64).reshape(n, nr),
        cycle_err_rms=np.asarray([c[0] for c in cycle], np.float64).reshape(n, gr + 1, gc + 1),
        cycle_inconsistency_frac=np.asarray([c[1] for c in cycle], np.float64).reshape(n, gr + 1, gc + 1),
        cycle_sample_mass=np.asarray([c[2] for c in cycle], np.float64).reshape(n, gr + 1, gc + 1),
        cycle_map_available=np.asarray([c[3] for c in cycle], np.bool_),
    )


def save_plan(pl, path, W, H, fps):
    """Write a computed ``Plan`` to a compressed, non-pickle NPZ for later shape analysis.

    Only plain numeric/unicode/boolean arrays are stored (no object dtype), so the file loads with
    ``np.load(path, allow_pickle=False)``. Contents: schema tag, frame dimensions, frame count, analysis fps,
    the per-frame global homographies, residual mesh/row corrections, raw/smoothed corner quads, the
    fallback count, and the opt-in sampled border-backoff result (applied clip-wide ``scale``, the worst
    sampled out-of-bounds fraction at that scale, and whether the probe ran). Schema ``mlxstab.plan/1`` is the
    legacy layout without those three fields and is still loadable read-only."""
    np.savez_compressed(
        path,
        schema=np.asarray(PLAN_SCHEMA), width=np.int64(W), height=np.int64(H),
        frames=np.int64(len(pl.G)), fps=np.float64(fps),
        G=np.asarray(pl.G, np.float64), mesh_corr=np.asarray(pl.mesh_corr), row_corr=np.asarray(pl.row_corr),
        raw_quads=np.asarray(pl.raw_quads), path_quads=np.asarray(pl.path_quads),
        fallbacks=np.int64(pl.fallbacks),
        sampled_border_scale=np.float64(pl.sampled_border_scale),
        sampled_border_worst_frac=np.float64(pl.sampled_border_worst_frac),
        sampled_border_measured=np.bool_(pl.sampled_border_measured),
    )


def _upsample_time(X, stride, T):
    """Linear interpolation of a per-analysed-index series X [Ta,...] to T frames (hold after the last analysed one)."""
    Ta = len(X); t = np.arange(T) / stride; i0 = np.minimum(t.astype(int), Ta - 1); i1 = np.minimum(i0 + 1, Ta - 1)
    w = np.where(t >= Ta - 1, 0.0, t - i0).reshape((-1,) + (1,) * (X.ndim - 1))
    return X[i0] * (1 - w) + X[i1] * w


# --- opt-in sampled complete-warp border backoff -------------------------------------------------------------
# The probe grid is fixed and deterministic: the four frame corners, regular samples along each edge and a
# regular interior grid. Feasibility is measured with the production ``source_coords``/``valid_mask`` convention:
# the mesh/row corrections are applied exactly as they are rendered (bilinear, border-clamped, mesh shift first)
# at the probes. The many-scale scan, however, runs as a bounded, vectorized NumPy pass rather than one tiny MLX
# graph per temporal chunk: the probe grid is so small that on this workload the MLX launch/synchronisation cost
# dominated the arithmetic. ``_source_coords_ts_scales_mlx`` keeps the previous MLX batched helper as the
# reference the NumPy probe is checked against.
_PROBE_EDGE_SAMPLES = 17       # samples along each frame edge (endpoints = corners)
_PROBE_INTERIOR_GRID = (9, 5)  # interior samples (cols, rows)
# Descending candidate scales (1.000, 0.999, ..., 0.000): 0.001 spacing, so a useful correction that a coarser
# (0.01) grid silently discards is still found. Scanned in order because feasibility need not be monotonic in
# the scale; 0.0 is the identity fallback and is always feasible for a valid limit in [0, 1).
_SAMPLED_BORDER_SCALES = tuple(round(1.0 - i * 0.001, 4) for i in range(1001))
# Temporal chunking of the sampled probe: a bounded number of frames is evaluated together with all currently
# alive candidate scales in one vectorized NumPy pass, so temporaries stay bounded and the whole clip is never
# materialised at once.
_SAMPLED_BORDER_TIME_CHUNK = 8          # frames per batched temporal chunk
_SAMPLED_BORDER_CHUNK_ELEMS = 1 << 23   # hard cap on a chunk's (frame * scale * probe) elements, bounds temporaries


def _sampled_border_chunk_frames(n_scales, n_probe):
    """Frames to evaluate per batched temporal chunk: never the whole clip, never more than the element cap.

    This is ``min(_SAMPLED_BORDER_TIME_CHUNK, cap // (alive scales * probes))`` clamped to at least one, so the
    ``[frames, scales, probes]`` temporaries stay explicitly bounded even if the probe grid or the candidate set
    grows. A scale can only be pruned at a chunk boundary, which leaves its running worst a running maximum; the
    selected scale and reported worst are therefore independent of the chunk length."""
    per_frame = max(int(n_scales), 1) * max(int(n_probe), 1)
    return int(min(_SAMPLED_BORDER_TIME_CHUNK, max(1, _SAMPLED_BORDER_CHUNK_ELEMS // per_frame)))


SAMPLED_BORDER_NOTE = (
    "Sampled border backoff: the complete composed output-to-source field is measured on a fixed probe grid "
    "(frame corners, edge samples and a regular interior grid) for every frame, and one clip-wide scale in "
    "[0,1] is applied to the global correction (I + scale*(G-I)) and to the mesh/row residuals (scale*array). "
    "This is a sampled estimate of the out-of-bounds fraction, not a strict per-pixel guarantee, and it does "
    "not prevent foldover."
)


def _probe_grid(W, H):
    """Deterministic output-space probe points ``[N,2]``: corners, edge samples and a regular interior grid."""
    ex = np.linspace(0.0, float(W - 1), _PROBE_EDGE_SAMPLES)
    ey = np.linspace(0.0, float(H - 1), _PROBE_EDGE_SAMPLES)
    edge_x = np.concatenate([ex, ex, np.zeros_like(ey), np.full_like(ey, float(W - 1))])
    edge_y = np.concatenate([np.zeros_like(ex), np.full_like(ex, float(H - 1)), ey, ey])
    ix = np.linspace(0.0, float(W - 1), _PROBE_INTERIOR_GRID[0])
    iy = np.linspace(0.0, float(H - 1), _PROBE_INTERIOR_GRID[1])
    gx, gy = np.meshgrid(ix, iy)
    return np.unique(np.stack([np.concatenate([edge_x, gx.ravel()]),
                               np.concatenate([edge_y, gy.ravel()])], axis=1), axis=0)


def _scaled_plan(pl, scale):
    """A Plan with corrections scaled by ``scale``: ``G -> I + scale*(G-I)``, mesh/row -> ``scale*array``.

    ``scale == 1.0`` returns the plan itself so its arrays are never rebuilt or modified."""
    if scale == 1.0:
        return pl
    eye = np.eye(3)
    Gs = [eye + scale * (np.asarray(g, np.float64) - eye) for g in pl.G]
    return Plan(G=Gs, mesh_corr=pl.mesh_corr * scale, row_corr=pl.row_corr * scale,
                raw_quads=pl.raw_quads, path_quads=pl.path_quads, fallbacks=pl.fallbacks)


def _valid_mask_np(coords, H, W):
    """True where a NumPy sample position lies inside the source image (same convention as ``warp.valid_mask``).

    Kept local so the sampled probe never imports a benchmark module; non-finite coordinates compare ``False``
    and are therefore out of bounds, exactly like the MLX predicate."""
    x, y = coords[..., 0], coords[..., 1]
    return (x >= -0.5) & (x <= W - 0.5) & (y >= -0.5) & (y <= H - 0.5)


def _bilinear_border_sample_np(img, coords):
    """NumPy bilinear-border sampler mirroring ``warp.sample(img, coords, "bilinear", "border")``.

    ``img`` is ``[N,H,W,C]`` and ``coords`` ``[N,Ho,Wo,2]`` (x, y) in pixel coordinates; the result is
    ``[N,Ho,Wo,C]`` float32. Tap positions, border index clamping, the per-axis weight sum and the final
    ``1/(sx*sy)`` renormalisation reproduce the Metal kernel's float32 arithmetic and term order, so the sampled
    mesh/row corrections -- and the resulting in-bounds classification -- agree with the production renderer
    instead of merely approximating the bilinear interpolation. Non-finite coordinates produce a NaN sample
    (treated as out of bounds by :func:`_valid_mask_np`), matching the kernel's clamped taps for NaN inputs."""
    img = np.asarray(img, np.float32); coords = np.asarray(coords, np.float32)
    N, H, W, C = img.shape
    px, py = coords[..., 0], coords[..., 1]
    fx, fy = np.floor(px), np.floor(py)
    ax0, ax1 = px - fx, px - (fx + np.float32(1.0))
    ay0, ay1 = py - fy, py - (fy + np.float32(1.0))
    one, zero = np.float32(1.0), np.float32(0.0)
    wx0 = np.where(np.abs(ax0) < one, one - np.abs(ax0), zero).astype(np.float32)
    wx1 = np.where(np.abs(ax1) < one, one - np.abs(ax1), zero).astype(np.float32)
    wy0 = np.where(np.abs(ay0) < one, one - np.abs(ay0), zero).astype(np.float32)
    wy1 = np.where(np.abs(ay1) < one, one - np.abs(ay1), zero).astype(np.float32)
    sx, sy = (wx0 + wx1).astype(np.float32), (wy0 + wy1).astype(np.float32)
    # integer taps: floor of a finite coordinate, border-clamped; non-finite floors collapse to the first pixel
    fxi = np.nan_to_num(fx, nan=0.0, posinf=0.0, neginf=0.0)
    fyi = np.nan_to_num(fy, nan=0.0, posinf=0.0, neginf=0.0)
    x0 = np.clip(fxi, 0.0, float(W - 1)).astype(np.intp); x1 = np.clip(fxi + one, 0.0, float(W - 1)).astype(np.intp)
    y0 = np.clip(fyi, 0.0, float(H - 1)).astype(np.intp); y1 = np.clip(fyi + one, 0.0, float(H - 1)).astype(np.intp)
    b = np.arange(N).reshape((N,) + (1,) * (px.ndim - 1))
    t00, t10 = img[b, y0, x0], img[b, y0, x1]
    t01, t11 = img[b, y1, x0], img[b, y1, x1]
    acc = (wx0 * wy0)[..., None] * t00                     # same tap/accumulation order as the Metal kernel
    acc = acc + (wx1 * wy0)[..., None] * t10
    acc = acc + (wx0 * wy1)[..., None] * t01
    acc = acc + (wx1 * wy1)[..., None] * t11
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = one / (sx * sy)
    return (acc * norm[..., None]).astype(np.float32)


def _source_coords_ts_scales_np(pl, ts, W, H, scales, grid):
    """NumPy source-coordinate fields for several frames and candidate backoff scales at once.

    The NumPy counterpart of ``_source_coords_ts_scales_mlx`` used by the many-scale probe; entry ``[f, k]``
    reproduces ``source_coords(_scaled_plan(pl, scales[k]), ts[f], W, H, grid)[0]``: the global part is
    ``I + s*(G-I)`` evaluated in float32, the mesh correction is bilinear-border sampled at the global-mapped
    source coordinate and added, then the row correction is bilinear-border sampled at the shifted ``y`` and
    added (exactly the render order). Returns ``[len(ts), len(scales), Ho, Wo, 2]``. ``grid`` is an explicit
    ``(x, y)`` pair of 2-D NumPy arrays. This is a sampled evaluation; it makes no per-pixel guarantee and does
    not prevent foldover."""
    x, y = grid
    x = np.asarray(x, np.float32); y = np.asarray(y, np.float32)
    ts = np.asarray(ts, np.int64).reshape(-1); s = np.asarray(scales, np.float64).reshape(-1)
    F, S = len(ts), len(s); Ho, Wo = x.shape; FS = F * S
    eye = np.eye(3)
    Gt = np.asarray(pl.G, np.float64)[ts]
    # float64 I + s*(G-I) then float32, matching _scaled_plan's matrix and source_coords' float32 entries
    Gm = (eye[None, None] + s[None, :, None, None] * (Gt[:, None] - eye[None, None])).astype(np.float32)
    shp = (F, S) + (1,) * x.ndim
    def col(i, j): return Gm[:, :, i, j].reshape(shp)
    a00, a01, a02 = col(0, 0), col(0, 1), col(0, 2)
    a10, a11, a12 = col(1, 0), col(1, 1), col(1, 2)
    a20, a21, a22 = col(2, 0), col(2, 1), col(2, 2)
    w = a20 * x + a21 * y + a22
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = (a00 * x + a01 * y + a02) / w; sy = (a10 * x + a11 * y + a12) / w
    src = np.stack([sx, sy], -1)                           # [F,S,Ho,Wo,2]
    gr, gc = pl.mesh_corr.shape[1] - 1, pl.mesh_corr.shape[2] - 1
    if gc > 0:
        u = (src[..., 0] + np.float32(.5)) / np.float32(W / gc)
        v = (src[..., 1] + np.float32(.5)) / np.float32(H / gr)
        mesh = np.asarray(pl.mesh_corr, np.float64)[ts][:, None] * s[None, :, None, None, None]
        disp = _bilinear_border_sample_np(mesh.astype(np.float32).reshape(FS, gr + 1, gc + 1, 2),
                                          np.stack([u, v], -1).reshape(FS, Ho, Wo, 2))
        src = src + disp.reshape(F, S, Ho, Wo, 2)
    nr = pl.row_corr.shape[1]
    row_t = np.asarray(pl.row_corr, np.float64)[ts]
    if np.abs(row_t).max() > 0:
        v = (src[..., 1] + np.float32(.5)) / np.float32(H / nr) - np.float32(.5)
        rows = (row_t[:, None] * s[None, :, None, None]).astype(np.float32)   # [F,S,nr,2]
        disp = _bilinear_border_sample_np(rows.reshape(FS, nr, 1, 2),
                                          np.stack([np.zeros((FS, Ho, Wo), np.float32), v.reshape(FS, Ho, Wo)], -1))
        src = src + disp.reshape(F, S, Ho, Wo, 2)
    return src


def _sampled_worst_border_frac(pl, W, H, pts):
    """Worst sampled output-out-of-bounds fraction over every frame of ``pl`` (production sampler convention)."""
    x = mx.array(pts[:, 0].astype(np.float32))[None]
    y = mx.array(pts[:, 1].astype(np.float32))[None]
    worst = 0.0
    for t in range(len(pl.G)):
        src = source_coords(pl, t, W, H, (x, y))
        bad = valid_mask(src, H, W)
        mx.eval(bad)
        frac = 1.0 - float(np.array(bad[0, 0]).mean())
        if frac > worst:
            worst = frac
    return worst


def _sampled_worst_border_frac_batch(pl, W, H, pts, scales, limit=None, frames_per_chunk=None):
    """Worst sampled out-of-bounds fraction for every candidate scale, batched over short temporal chunks.

    Returns ``[len(scales)]``; entry ``k`` is the same quantity :func:`_sampled_worst_border_frac` measures on
    ``_scaled_plan(pl, scales[k])`` (each candidate scale evaluated with the production mesh/row bilinear-border
    convention through the vectorized NumPy probe :func:`_source_coords_ts_scales_np`). A bounded chunk of frames
    (see :func:`_sampled_border_chunk_frames`) is evaluated together with all currently alive candidate scales in
    one NumPy pass, so temporaries stay bounded and the whole clip is never materialised at once.

    When ``limit`` is given, a scale whose running worst already exceeds it is dropped at the next chunk
    boundary: the worst is a running maximum, so a scale that is infeasible on one frame can never become
    feasible again. This prunes work without assuming any monotonicity of feasibility across scales, and it
    leaves the value of every feasible scale (worst ``<= limit`` over the whole clip) identical to the scalar
    reference."""
    pts = np.asarray(pts, np.float64)
    x = pts[:, 0].astype(np.float32).reshape(1, -1)
    y = pts[:, 1].astype(np.float32).reshape(1, -1)
    scales = np.asarray(scales, np.float64).reshape(-1)
    worst = np.zeros(len(scales), np.float64)
    alive = np.ones(len(scales), bool)
    T = len(pl.G); t = 0
    while t < T:
        idx = np.nonzero(alive)[0]
        if idx.size == 0:
            break
        n = frames_per_chunk or _sampled_border_chunk_frames(idx.size, pts.shape[0])
        ts = np.arange(t, min(t + n, T))
        src = _source_coords_ts_scales_np(pl, ts, W, H, scales[idx], (x, y))
        bad = _valid_mask_np(src, H, W)
        frac = 1.0 - bad.reshape(len(ts), idx.size, -1).mean(axis=2)
        worst[idx] = np.maximum(worst[idx], frac.max(axis=0))
        if limit is not None:
            alive[idx] = worst[idx] <= limit
        t += len(ts)
    return worst


def _with_sampled_border_meta(pl, scale, worst):
    """A Plan sharing ``pl``'s arrays, tagged with the sampled border-backoff result (arrays untouched)."""
    return Plan(G=pl.G, mesh_corr=pl.mesh_corr, row_corr=pl.row_corr, raw_quads=pl.raw_quads,
                path_quads=pl.path_quads, fallbacks=pl.fallbacks,
                sampled_border_scale=float(scale), sampled_border_worst_frac=float(worst),
                sampled_border_measured=True)


def sampled_border_backoff(pl, W, H, limit):
    """Opt-in clip-wide border backoff measured on the deterministic probe grid.

    Scans the fixed descending scale grid :data:`_SAMPLED_BORDER_SCALES` (0.001 spacing) from 1.0 to 0.0 and
    returns the first (largest) scale whose worst per-frame sampled out-of-bounds fraction is ``<= limit``.
    Feasibility is *not* assumed monotonic, so the grid is scanned in order rather than binary-searched. Scale 0
    is the identity fallback (``G = I``, mesh/row zero), which is always feasible. If scale 1 is already feasible
    the returned Plan shares the input's correction arrays unchanged (no copy, no modification). ``limit`` must
    already be validated (``None`` disables). This is a sampled estimate, not a strict per-pixel guarantee, and
    it does not prevent foldover."""
    if limit is None:
        return pl
    pts = _probe_grid(W, H)
    scales = np.asarray(_SAMPLED_BORDER_SCALES, np.float64)
    # Scale 1 first (chunked like everything else): when it is already feasible the plan's arrays are returned
    # untouched, preserving the no-op identity exactly, and the other ~1000 candidates are never evaluated.
    worst1 = float(_sampled_worst_border_frac_batch(pl, W, H, pts, scales[:1], limit=limit)[0])
    if worst1 <= limit:
        return _with_sampled_border_meta(pl, 1.0, worst1)
    worst = _sampled_worst_border_frac_batch(pl, W, H, pts, scales, limit)
    feasible = np.nonzero(worst <= limit)[0]
    k = int(feasible[0]) if feasible.size else len(scales) - 1   # scale 0 is always feasible
    scale = float(scales[k])
    return _with_sampled_border_meta(_scaled_plan(pl, scale), scale, float(worst[k]))


def plan(mot, cfg, roll=None, n_frames=None):
    W, H = mot.W, mot.H; stride = getattr(mot, "stride", 1)
    Ta = len(mot.fits) + 1; T = n_frames or (Ta - 1) * stride + 1
    cs = G.corners(W, H)
    Cs = [np.eye(3)]
    for f in mot.fits: Cs.append(Cs[-1] @ np.linalg.inv(f.H))                 # analysed frame -> reference
    quads = np.stack([G.apply_h(C, cs) for C in Cs])                           # camera path in corner space
    if stride > 1 or T != Ta:                                                  # path analysed at a lower rate: interpolate it
        quads = _upsample_time(quads, stride, T); Cs = [G.dlt(cs, q) for q in quads]
    rolls = roll_per_frame(roll, T, mot.fps) if roll is not None else None
    target = np.stack([G.rotate_quad(q, r) for q, r in zip(quads, rolls)]) if rolls is not None else quads.copy()
    centre = np.array([W / 2 - .5, H / 2 - .5]); base = cs - centre
    a_l, t_l, res_l = [], [], []
    for q in target:
        a, t, r = G.similarity_fit(base, q); a_l.append(a); t_l.append(t); res_l.append(r)
    a = np.array(a_l); t = np.array(t_l); res = np.stack(res_l)
    th = np.unwrap(np.angle(a)); ls = np.log(np.abs(a))
    R = 0.5 * np.hypot(W, H); b = cfg.budget * W
    P = np.stack([t.real, t.imag], 1)
    P = SM.smooth_columns(P, cfg.smooth, cfg.sigma, b, cfg.l1_weights, cfg.fidelity)
    ang = SM.smooth_columns(np.stack([th, ls], 1) * R, cfg.smooth, cfg.sigma, b, cfg.l1_weights, cfg.fidelity) / R      # in corner-pixel units
    rs = SM.smooth_columns(res.reshape(T, 8), cfg.smooth, cfg.sigma, b, cfg.l1_weights, cfg.fidelity).reshape(T, 4, 2)
    path = np.zeros_like(quads)
    for i in range(T):
        path[i] = G.similarity_apply(base, np.exp(ang[i, 1] + 1j * ang[i, 0]), P[i, 0] + 1j * P[i, 1]) + rs[i]
        if cfg.zoom != 1.0: c = path[i].mean(0); path[i] = c + (path[i] - c) / cfg.zoom
    Gs = [np.linalg.inv(Cs[i]) @ G.dlt(cs, path[i]) for i in range(T)]
    if cfg.crop_zoom != 1.0:
        # Opt-in isotropic center zoom on output coordinates, composed before the output->source map:
        # G_new = G @ Z. The mesh/row residual arrays below are left untouched, so production re-evaluates
        # them at the new G-mapped source points rather than pre-composing the zoom into the residuals.
        Z = _crop_zoom_matrix(W, H, cfg.crop_zoom)
        Gs = [np.asarray(g, np.float64) @ Z for g in Gs]
    # mesh / rolling-shutter residual paths (content motion beyond the global homography), accumulated and smoothed
    gc, gr = cfg.mesh
    mesh_corr = np.zeros((T, gr + 1, gc + 1, 2)); row_corr = np.zeros((T, max(cfg.rs_rows, 1), 2))
    mb = cfg.mesh_budget * W
    if gc > 0 and mot.fits:
        M = np.concatenate([np.zeros((1, gr + 1, gc + 1, 2)), np.cumsum([f.mesh for f in mot.fits], 0)])
        if stride > 1 or T != Ta: M = _upsample_time(M, stride, T)
        Ms = SM.smooth_columns(M.reshape(T, -1), cfg.smooth, cfg.sigma, mb, cfg.l1_weights, cfg.fidelity).reshape(M.shape)
        mesh_corr = M - Ms
    if cfg.rs_rows > 0 and mot.fits:
        Rw = np.concatenate([np.zeros((1, cfg.rs_rows, 2)), np.cumsum([f.rows for f in mot.fits], 0)])
        if stride > 1 or T != Ta: Rw = _upsample_time(Rw, stride, T)
        Rs = SM.smooth_columns(Rw.reshape(T, -1), cfg.smooth, cfg.sigma, mb, cfg.l1_weights, cfg.fidelity).reshape(Rw.shape)
        row_corr = Rw - Rs
    pl = Plan(Gs, mesh_corr, row_corr, quads, path, sum(f.fallback for f in mot.fits))
    limit = getattr(cfg, "max_sampled_border_frac", None)
    if limit is not None:
        pl = sampled_border_backoff(pl, W, H, limit)
    return pl


def source_coords(pl, t, W, H, grid=None):
    """Virtual-frame pixel -> source pixel field [H,W,2] (global homography + mesh + row corrections)."""
    if grid is None:
        ys, xs = mx.meshgrid(mx.arange(H), mx.arange(W), indexing="ij"); grid = (xs.astype(mx.float32), ys.astype(mx.float32))
    x, y = grid
    Gm = [[float(v) for v in r] for r in pl.G[t]]
    # reference matrix entries as float64 python floats; mx computes in float32 (|coords| ~ 1e3 -> ~1e-4 px)
    w = Gm[2][0] * x + Gm[2][1] * y + Gm[2][2]
    sx = (Gm[0][0] * x + Gm[0][1] * y + Gm[0][2]) / w; sy = (Gm[1][0] * x + Gm[1][1] * y + Gm[1][2]) / w
    src = mx.stack([sx, sy], -1)[None]
    gr, gc = pl.mesh_corr.shape[1] - 1, pl.mesh_corr.shape[2] - 1
    if gc > 0:
        u = (src[..., 0] + .5) / (W / gc); v = (src[..., 1] + .5) / (H / gr)
        mesh = mx.array(pl.mesh_corr[t].astype(np.float32))[None]
        src = src + sample(mesh, mx.stack([u, v], -1), "bilinear", "border")
    nr = pl.row_corr.shape[1]
    if np.abs(pl.row_corr[t]).max() > 0:
        v = (src[..., 1] + .5) / (H / nr) - .5
        rows = mx.array(pl.row_corr[t].astype(np.float32)).reshape(1, nr, 1, 2)
        src = src + sample(rows, mx.stack([mx.zeros_like(v), v], -1), "bilinear", "border")
    return src


def _source_coords_ts_scales_mlx(pl, ts, W, H, scales, grid):
    """Reference MLX implementation: source pixel fields for several frames and candidate backoff scales at once.

    Kept as the reference the NumPy probe :func:`_source_coords_ts_scales_np` is checked against (and used by
    :func:`source_coords_batch`). ``ts`` is a 1-D sequence of frame indices and ``scales`` a 1-D sequence; the
    result is ``[len(ts), len(scales), Ho, Wo, 2]`` whose entry ``[f, k]`` is
    ``source_coords(_scaled_plan(pl, scales[k]), ts[f], W, H, grid)[0]``: the global part is ``I + s*(G-I)`` and
    the residual mesh/row corrections are ``s * mesh_corr`` / ``s * row_corr``, sampled with the exact
    bilinear-border mesh/row convention of :func:`source_coords` (the mesh shift is applied first, then the row
    correction is evaluated on the shifted coordinate, as in rendering). The ``(frame, scale)`` axes are
    flattened into one leading axis only for the ``sample`` calls, which require 4-D inputs. One MLX graph covers
    the whole chunk, so callers pay a single synchronisation per chunk. ``grid`` must be an explicit ``(x, y)``
    pair of 2-D coordinate arrays. This is a sampled evaluation; it makes no per-pixel guarantee and does not
    prevent foldover."""
    x, y = grid
    ts = np.asarray(ts, np.int64).reshape(-1); s = np.asarray(scales, np.float64).reshape(-1)
    F, S = len(ts), len(s); FS = F * S; Ho, Wo = x.shape
    eye = np.eye(3)
    # float64 I + s*(G-I) then float32, matching _scaled_plan's matrix and source_coords' float32 entries
    Gt = np.asarray(pl.G, np.float64)[ts]
    Gm = (eye[None, None] + s[None, :, None, None] * (Gt[:, None] - eye[None, None])).astype(np.float32)
    shp = (F, S) + (1,) * x.ndim
    def col(i, j): return mx.array(Gm[:, :, i, j]).reshape(shp)
    a00 = col(0, 0); a01 = col(0, 1); a02 = col(0, 2)
    a10 = col(1, 0); a11 = col(1, 1); a12 = col(1, 2)
    a20 = col(2, 0); a21 = col(2, 1); a22 = col(2, 2)
    w = a20 * x + a21 * y + a22
    sx = (a00 * x + a01 * y + a02) / w; sy = (a10 * x + a11 * y + a12) / w
    src = mx.stack([sx, sy], -1).reshape(FS, Ho, Wo, 2)
    gr, gc = pl.mesh_corr.shape[1] - 1, pl.mesh_corr.shape[2] - 1
    if gc > 0:
        u = (src[..., 0] + .5) / (W / gc); v = (src[..., 1] + .5) / (H / gr)
        mesh = np.asarray(pl.mesh_corr, np.float64)[ts][:, None] * s[None, :, None, None, None]
        mesh = mx.array(mesh.astype(np.float32)).reshape(FS, gr + 1, gc + 1, 2)
        src = src + sample(mesh, mx.stack([u, v], -1), "bilinear", "border")
    nr = pl.row_corr.shape[1]
    row_t = np.asarray(pl.row_corr, np.float64)[ts]
    if np.abs(row_t).max() > 0:
        v = (src[..., 1] + .5) / (H / nr) - .5
        rows = (row_t[:, None] * s[None, :, None, None]).astype(np.float32)
        rows = mx.array(rows).reshape(FS, nr, 1, 2)
        src = src + sample(rows, mx.stack([mx.zeros_like(v), v], -1), "bilinear", "border")
    return src.reshape(F, S, Ho, Wo, 2)


def source_coords_batch(pl, t, W, H, scales, grid=None):
    """Virtual-frame pixel -> source pixel fields for several candidate backoff scales of one frame.

    ``scales`` is a 1-D sequence; the result is ``[S, Ho, Wo, 2]`` (with ``S = len(scales)``) whose row ``k`` is
    ``source_coords(_scaled_plan(pl, scales[k]), t, W, H, grid)[0]``: the global part is ``I + s*(G-I)`` and the
    residual mesh/row corrections are ``s * mesh_corr`` / ``s * row_corr``, sampled with the exact bilinear-border
    mesh/row convention of :func:`source_coords` (the row correction is evaluated after the mesh shift, as in
    rendering). Every candidate scale shares one lazily-built MLX graph, so the caller materialises all of them
    with a single synchronisation. This is the single-frame special case of
    :func:`_source_coords_ts_scales_mlx`, which additionally batches a bounded chunk of frames.

    ``grid`` is an explicit ``(x, y)`` coordinate pair of matching shape; the returned leading axis enumerates
    the candidate scales. This is a sampled evaluation; it makes no per-pixel guarantee and does not prevent
    foldover."""
    if grid is None:
        raise ValueError("source_coords_batch requires an explicit grid")
    return _source_coords_ts_scales_mlx(pl, [t], W, H, scales, grid)[0]


def render_from_src(frame, src, cfg):
    """Render from an already-computed source-coordinate field -> (uint8 [H,W,3], border mask bool [H,W])."""
    H, W = frame.shape[:2]
    img = sample(mx.array(frame).astype(mx.float32)[None], src, cfg.kernel, "zeros")
    bad = ~valid_mask(src, H, W)
    img = mx.where(bad[..., None], 0.0, mx.clip(img, 0, 255))
    mx.eval(img, bad, src)
    return np.array(img[0]).round().astype(np.uint8), np.array(bad[0])


def render_frame(frame, pl, t, cfg, grid=None, diagnose=False, src=None):
    """-> (stabilized uint8 [H,W,3], border mask bool [H,W] (True = no source), distortion[, warp_diag]).

    ``distortion`` is the historical global-homography-only score. With ``diagnose=True`` a fourth item
    holds complete-warp diagnostics computed from the very source field used for rendering (no second warp pass).
    Pass ``src`` to reuse a field already computed by the caller."""
    H, W = frame.shape[:2]
    if src is None: src = source_coords(pl, t, W, H, grid)
    img, bad = render_from_src(frame, src, cfg)
    out = (img, bad, _distortion(pl.G[t], W, H))
    if diagnose:
        return (*out, warpmetrics.diagnose(np.array(src[0]), H, W, kernel=cfg.kernel))
    return out


def warp_diagnostics(pl, cfg, W, H, frames=None, grid=None):
    """Aggregated complete-warp diagnostics for a Plan (standalone API/benchmark entry point).

    ``stabilize`` reuses the source field already computed while rendering; this helper recomputes and
    aggregates it for callers that only want diagnostics."""
    diags = []
    for t in (range(len(pl.G)) if frames is None else frames):
        src = source_coords(pl, t, W, H, grid); mx.eval(src)
        diags.append(warpmetrics.diagnose(np.array(src[0]), H, W, kernel=cfg.kernel))
    return warpmetrics.aggregate(diags)


def _distortion(Gm, W, H):
    """Liu et al. 2013 distortion: ratio of the two singular values of the affine part of the global output->input homography
    (1 = none). The residual mesh is local detail and is not counted. Returns None for a degenerate or non-finite mapping
    (checked before least squares/SVD), so an invalid global map cannot be reported as a perfect score."""
    from .geometry import apply_h
    if H < 1 or W < 1: return None
    Gm = np.asarray(Gm, np.float64)
    if Gm.shape != (3, 3) or not np.isfinite(Gm).all(): return None
    ys, xs = np.mgrid[0:H:H // 8 or 1, 0:W:W // 8 or 1]; q = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)
    if len(q) < 3: return None
    mapped = apply_h(Gm, q)
    if not np.isfinite(mapped).all(): return None               # invalid mapping must not reach lstsq/SVD
    A = np.c_[q, np.ones(len(q))]
    try:
        sol = np.linalg.lstsq(A, mapped, rcond=None)[0]
        sv = np.linalg.svd(sol[:2].T, compute_uv=False)
    except np.linalg.LinAlgError:
        return None
    if sv.size < 2 or not np.isfinite(sv).all() or sv[0] <= 0: return None
    r = float(sv[1] / sv[0])
    return r if np.isfinite(r) else None


def stabilize(src, out, mask=None, weights=None, cfg=None, roll=None, start=None, duration=None, fps_out=None,
              flow_fn=None, log=print, report=None, on_step=None, dump_motion=None, dump_motion_npz=None,
              dump_plan=None, diag_every=0, diag_callback=None):
    cfg = cfg or Config()
    if cfg.warp_diag and diag_callback is not None and diag_every:
        raise ValueError("choose all-frame warp_diag or sampled diagnostics, not both")
    info = media.probe(src)
    if flow_fn is None:
        flow_fn = RaftFlow(weights, cfg.flow_iters, cfg.flow_downscale, cfg.warm_start, getattr(mx, cfg.corr_dtype), getattr(mx, cfg.flow_dtype), cfg.warm_iters or None)
    frames, fps, (W, H) = media.decode(src, info, start, duration); frames = media.prefetch(frames)
    if info["interlaced"]: log(f"interlaced ({info['field_order']}): bwdif send_field -> {fps:.3f} fps")
    import time; t0 = time.perf_counter()
    mot = analyze(frames, flow_fn, cfg, W, H, fps, log, on_step=on_step, stride=cfg.analysis_stride); t1 = time.perf_counter()
    if dump_motion:
        import pickle; Path(dump_motion).write_bytes(pickle.dumps(mot))
    if dump_motion_npz:
        save_motion_npz(mot, dump_motion_npz, cfg)
    pl = plan(mot, cfg, load_roll(roll) if isinstance(roll, (str, Path)) else roll, n_frames=mot.n_frames)
    if dump_plan:
        save_plan(pl, dump_plan, W, H, mot.fps)
    t2 = time.perf_counter()
    frames, fps, _ = media.decode(src, info, start, duration); frames = media.prefetch(frames)
    enc = media.Encoder(out, W, H, fps, out_fps=fps_out)
    menc = None
    try:
        menc = media.Encoder(mask, W, H, fps, gray=True, out_fps=fps_out) if mask else None
        grid = None; borders = []; dist = []; diags = []; rendered = 0; diag_total = 0.0
        ys, xs = mx.meshgrid(mx.arange(H), mx.arange(W), indexing="ij"); grid = (xs.astype(mx.float32), ys.astype(mx.float32))
        for t, fr in enumerate(frames):
            if t >= len(pl.G): break
            if on_step: on_step(t)
            if diag_callback is not None and diag_every and t % diag_every == 0:
                src = source_coords(pl, t, W, H, grid)
                ts = time.perf_counter(); img, bad = render_from_src(fr, src, cfg); t_render = time.perf_counter() - ts
                ts = time.perf_counter(); dg = warpmetrics.diagnose(np.array(src[0]), H, W, kernel=cfg.kernel); t_diag = time.perf_counter() - ts
                diag_total += t_diag
                diag_callback(t, t_render, t_diag, dg)
                d = _distortion(pl.G[t], W, H)
            elif cfg.warp_diag:
                img, bad, d, dg = render_frame(fr, pl, t, cfg, grid, diagnose=True); diags.append(dg)
            else:
                img, bad, d = render_frame(fr, pl, t, cfg, grid)
            enc.write(img); borders.append(float(bad.mean())); dist.append(d); rendered += 1
            if menc: menc.write(bad.astype(np.uint8) * 255)
            if t % 20 == 0: log(f"render {t}/{len(pl.G)}")
    except BaseException:
        # A render failure must not skip closing the owned video/mask encoders, and neither close may mask it.
        try:
            media.close_all((enc, menc))
        except BaseException:
            pass
        raise
    media.close_all((enc, menc))
    t3 = time.perf_counter(); n = len(pl.G); dv = [d for d in dist if d is not None]
    rep = {"frames": n, "decoded_frames": int(mot.n_frames), "rendered_frames": int(rendered),
           "output_fps": fps_out or fps, "analysis_fps": fps / max(int(cfg.analysis_stride), 1),
           "diagnostics_total_s": diag_total, "observed_nondiag_render_s": (t3 - t2) - diag_total,
           "timing_s": {"analysis": t1 - t0, "plan": t2 - t1, "render": t3 - t2},
           "fps_analysis": n / (t1 - t0), "fps_render": n / (t3 - t2), "fps_total": n / (t3 - t0), "fps": fps, "interlaced": info["interlaced"], "fallback_frames": pl.fallbacks,
           "mean_border_frac": float(np.mean(borders)), "max_border_frac": float(np.max(borders)),
           "cropping_ratio": float(1 - np.mean(borders)), "distortion_scope": "global_homography_only",
           "distortion_min": float(np.min(dv)) if dv else None, "distortion_mean": float(np.mean(dv)) if dv else None,
           "warp_diag": warpmetrics.aggregate(diags) if cfg.warp_diag else None,
           "max_sampled_border_frac": getattr(cfg, "max_sampled_border_frac", None),
           "crop_zoom": cfg.crop_zoom,
           "sampled_border_scale": pl.sampled_border_scale if pl.sampled_border_measured else None,
           "sampled_border_worst_frac": pl.sampled_border_worst_frac if pl.sampled_border_measured else None,
           "sampled_border_measured": bool(pl.sampled_border_measured),
           "sampled_border_note": SAMPLED_BORDER_NOTE}
    if report: Path(report).write_text(json.dumps(rep, indent=1))
    return rep
