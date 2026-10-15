"""Build the relative-depth cache consumed by :mod:`bench.depth_temporal_pilot` (optional, offline-capable).

This module turns an existing **already-decoded, already-verified** frame cache (the
``frames``/``meta`` ``.npz`` written by :func:`bench.flow_prediction_real.load_or_decode`) into exactly the
``.npz`` contract that :func:`bench.depth_temporal_pilot.load_depth_npz` validates and
:func:`bench.depth_temporal_pilot.validate_cache_correspondence` cross-checks: the keys ``indices`` (int),
``depth`` (float ``[K,H,W]`` at the frame cache's original resolution), ``model_id``, ``normalization`` and
``frames_npz_sha256`` (the content hash of the input frame cache).

Scope and meaning
-----------------
The maps are **monocular relative depth**, defined only up to a per-frame scale/shift. Each sampled map is
independently percentile-normalized (1st/99th percentile, clipped to ``[0, 1]``), so a value of ``1`` is
"relatively near within this frame" and ``0`` is "relatively far", with no metric meaning whatsoever. The
pilot compares such maps only through a robust affine fit, so this cache is sufficient for *structure
self-consistency* under RAFT transport; it is **not** metric depth, not a depth benchmark, and not
comparable across frames without a fitted affine map.

Optional heavy dependencies
---------------------------
``torch``, ``transformers`` and ``PIL`` are imported **lazily**, only when the real inference path actually
runs. The module imports cleanly, the CLI parses and the index/validation helpers work in an environment
with none of them installed; requesting real inference without them raises a clear ``RuntimeError``.

Shared heavy lock
-----------------
Real MPS inference is a heavy job and must not race a film render, so by default (``--lock on``) the
predictor is constructed and every frame is inferred while holding the project's shared
:class:`bench.heavylock.HeavyLock` (imported lazily). Cheap local work -- loading the frame npz, hashing it,
generating indices and writing the output -- is done **outside** the lock. ``--lock-chunk N`` releases and
re-acquires the lock after every ``N`` completed frames (``--lock-pause`` seconds of yielded time), which is
how a long run stays polite to other jobs.

Preprocessing (reproduces the local reference exactly)
------------------------------------------------------
For each sampled frame: RGB uint8 array -> PIL ``RGB`` image; the longest side is resized toward
``--max-side`` (default 1036) with the resulting dimensions rounded to the nearest **multiple of 14** and
floored at 14, using bicubic resampling; pixels are scaled to ``float32`` in ``[0, 1]`` and ImageNet
mean/std normalized; a single-image batch is run through ``AutoModelForDepthEstimation`` under ``eval()``
and ``no_grad()`` on the chosen device; the predicted map is restored to the original ``H, W`` with torch
bicubic interpolation (``align_corners=False``); finally it is percentile-normalized and clipped. Frames are
inferred **one at a time** to bound memory on MPS/CPU.

Output portability
------------------
The cache records only path-independent provenance: the frame content hash and the model id/normalization.
No absolute paths, source file names or machine details are written. The output is written atomically
(temp file + ``os.replace``) and refuses to overwrite the input frame cache.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from . import manifest as MF
from .depth_temporal_pilot import load_frames_npz

SCHEMA = "mlxstab.depth_cache/1"
DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"
DEFAULT_EVERY = 4
DEFAULT_MAX_SIDE = 1036
DEFAULT_DEVICE = "auto"
# Shared heavy lock (see bench/heavylock.py). On by default: real MPS inference is a heavy job and must not
# race a film render. Chunking follows the project etiquette (~30 steps per chunk, ~12 s yield).
DEFAULT_LOCK = "on"
DEFAULT_LOCK_CHUNK = 0
DEFAULT_LOCK_PAUSE = 0.0
# The reference preprocessing rounds the resized longest side to a multiple of 14 (the Depth Anything V2
# patch size) and never below one patch.
PATCH_MULTIPLE = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NORMALIZATION = "percentile_1_99_clip_0_1"
DEPTH_KEYS = ("indices", "depth", "model_id", "normalization", "frames_npz_sha256")


# --------------------------------------------------------------------------------------
# index selection
# --------------------------------------------------------------------------------------

def generate_indices(n_frames: int, *, every: int = DEFAULT_EVERY,
                     indices: Optional[Sequence[int]] = None) -> np.ndarray:
    """Return the strictly increasing sampled frame indices for a frame cache of ``n_frames`` frames.

    With an explicit ``indices`` list those values are validated and used verbatim. Otherwise a ``every``
    stride is expanded to ``0, every, 2*every, ...`` up to the **last eligible** frame, which is always
    included so the final frame is never silently dropped. At least two indices are required by the depth
    cache contract. Raises ``ValueError`` on a stride below 1, an out-of-range/duplicate/unsorted list, or
    fewer than two resulting samples.
    """
    n = int(n_frames)
    if n < 2:
        raise ValueError(f"frame cache must contain at least 2 frames to sample, got {n}")
    if indices is not None:
        idx = np.asarray([int(v) for v in indices], dtype=np.int64)
        if idx.ndim != 1 or idx.size < 2:
            raise ValueError("explicit indices must list at least 2 values")
        if np.any(np.diff(idx) <= 0):
            raise ValueError("explicit indices must be strictly increasing and unique")
        if int(idx[0]) < 0 or int(idx[-1]) >= n:
            raise ValueError(f"explicit indices must lie within [0, {n - 1}]")
        return idx
    step = int(every)
    if step < 1:
        raise ValueError("--every must be >= 1")
    last = ((n - 1) // step) * step  # last eligible multiple of the stride
    idx = np.arange(0, last + 1, step, dtype=np.int64)
    if idx.size < 2:
        # A stride larger than the clip must still yield the "first and last eligible frame" pair.
        idx = np.array([0, last if last > 0 else n - 1], dtype=np.int64)
    if idx.size < 2:
        raise ValueError(f"stride {step} yields fewer than 2 samples for {n} frames")
    return idx


def parse_indices(text) -> list:
    """Parse a ``"0,4,8"`` index list into ints, rejecting empty/garbage tokens."""
    if text is None:
        return None
    if isinstance(text, (list, tuple)):
        raw = [str(v) for v in text]
    else:
        raw = [t for t in str(text).split(",")]
    if len(raw) == 1 and not raw[0].strip():
        raise ValueError("--indices was given but is empty")
    out = []
    for tok in raw:
        tok = tok.strip()
        if not tok:
            raise ValueError(f"--indices contains an empty token in {text!r}")
        out.append(int(tok))
    return out


# --------------------------------------------------------------------------------------
# normalization and preprocessing (pure, testable, no heavy imports)
# --------------------------------------------------------------------------------------

def normalize_percentile(depth: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Percentile-normalize a single map to ``[0, 1]`` and clip.

    Values are mapped from the ``[lo, hi]`` percentile band of the finite entries; a degenerate (constant
    or all-non-finite) map yields zeros rather than a division blow-up. Returns ``float32`` with no
    absolute scale preserved -- this is a purely relative, per-frame normalization.
    """
    d = np.asarray(depth, dtype=np.float64)
    finite = np.isfinite(d)
    if not finite.any():
        return np.zeros(d.shape, dtype=np.float32)
    vals = d[finite]
    p_lo = float(np.percentile(vals, lo))
    p_hi = float(np.percentile(vals, hi))
    if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi - p_lo <= 0.0:
        return np.zeros(d.shape, dtype=np.float32)
    out = (d - p_lo) / (p_hi - p_lo)
    out = np.clip(out, 0.0, 1.0)
    out[~finite] = 0.0
    return out.astype(np.float32)


def target_resize_dims(height: int, width: int, max_side: int) -> tuple:
    """Return the ``(new_h, new_w)`` preprocessing size for a ``height x width`` frame.

    The longest side is scaled toward ``max_side``, each result is rounded to the nearest multiple of 14 and
    floored at 14 (never zero, never below one patch). Matches the local reference preprocessing.
    """
    h, w = int(height), int(width)
    if h <= 0 or w <= 0:
        raise ValueError(f"frame dimensions must be positive, got {h}x{w}")
    ms = int(max_side)
    if ms <= 0:
        raise ValueError("--max-side must be >= 1")
    scale = float(ms) / float(max(h, w))
    nh = max(PATCH_MULTIPLE, int(round(h * scale / PATCH_MULTIPLE)) * PATCH_MULTIPLE)
    nw = max(PATCH_MULTIPLE, int(round(w * scale / PATCH_MULTIPLE)) * PATCH_MULTIPLE)
    return nh, nw


def imagenet_normalize(rgb_float: np.ndarray) -> np.ndarray:
    """Apply ImageNet mean/std normalization to a ``float32`` RGB array already scaled to ``[0, 1]``."""
    a = np.asarray(rgb_float, dtype=np.float32)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    return (a - mean) / std


# --------------------------------------------------------------------------------------
# lazy heavy imports and the real predictor
# --------------------------------------------------------------------------------------

def _require_heavy_deps():
    """Import torch/transformers/PIL lazily; raise a clear error when they are absent."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without the optional stack
        raise RuntimeError(
            "real depth inference requires torch, transformers and pillow; install them (e.g. in the "
            "side '.venv-vox' used by the local reference) or pass an injected predictor for tests"
        ) from e
    import torch
    from PIL import Image
    from transformers import AutoModelForDepthEstimation
    return torch, Image, AutoModelForDepthEstimation


def resolve_device(device: str):
    """Resolve ``auto|mps|cpu`` to a torch device string; ``auto`` prefers MPS when available."""
    torch, _, _ = _require_heavy_deps()
    dev = str(device).lower()
    if dev == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if dev not in ("mps", "cpu"):
        raise ValueError(f"--device must be one of auto|mps|cpu, got {device!r}")
    if dev == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but torch reports MPS is unavailable")
    return dev


class DepthAnythingPredictor:
    """Real Depth Anything V2 predictor: lazy heavy imports, ``eval()`` + ``no_grad()``, one frame at a time.

    Reproduces the local reference preprocessing (bicubic resize to a multiple-of-14 longest side, ImageNet
    normalization, ``predicted_depth``, bicubic restore to the original resolution) and percentile-normalizes
    each map independently. This is relative depth, not metric depth. Construction downloads weights unless
    ``local_files_only`` is set (no network in that mode).
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, *, device: str = DEFAULT_DEVICE,
                 max_side: int = DEFAULT_MAX_SIDE, local_files_only: bool = False):
        torch, Image, AutoModelForDepthEstimation = _require_heavy_deps()
        self._torch = torch
        self._Image = Image
        self.model_id = str(model_id)
        self.device = resolve_device(device)
        self.max_side = int(max_side)
        if self.max_side <= 0:
            raise ValueError("--max-side must be >= 1")
        model = AutoModelForDepthEstimation.from_pretrained(
            self.model_id, local_files_only=bool(local_files_only))
        self.model = model.to(self.device).eval()

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Infer one relative-depth map (``float32 [H,W]`` in ``[0, 1]``) for an RGB uint8 array."""
        torch = self._torch
        Image = self._Image
        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError("predictor input must be an [H,W,3] RGB array")
        height, width = int(arr.shape[0]), int(arr.shape[1])
        im = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        nh, nw = target_resize_dims(height, width, self.max_side)
        resized = im.resize((nw, nh), Image.BICUBIC)
        a = np.asarray(resized, dtype=np.float32) / 255.0
        a = imagenet_normalize(a)
        x = torch.from_numpy(a).permute(2, 0, 1)[None].to(self.device)
        with torch.no_grad():
            d = self.model(pixel_values=x).predicted_depth
            d = torch.nn.functional.interpolate(
                d[:, None], size=(height, width), mode="bicubic", align_corners=False)[0, 0]
            d = d.float().cpu().numpy()
        return normalize_percentile(d)


# --------------------------------------------------------------------------------------
# inference driver (predictor-injectable so tests need no weights)
# --------------------------------------------------------------------------------------

def build_depth_array(frames: np.ndarray, indices: np.ndarray, predictor,
                      *, log: Callable[[str], None] = lambda _: None,
                      on_step: Optional[Callable[[int], None]] = None) -> np.ndarray:
    """Run ``predictor`` on each sampled frame and stack ``float32 [K,H,W]`` maps.

    Inference is done **one frame at a time** (the predictor owns its own batching) to bound peak memory.
    Every map is checked for the expected ``[H, W]`` shape and finite values before being stored, so a broken
    predictor fails loudly instead of writing a silently wrong cache.

    ``on_step`` is invoked once per **completed, validated** sample with a monotonically increasing one-based
    count (1, 2, ...). Passing ``HeavyLock.on_step`` lets a ``--lock-chunk`` boundary yield between frames;
    the callback is a timing/gating side effect only and never affects the returned maps. Failed samples do
    not tick, so a partial run never advances the shared step counter past the last good frame.
    """
    n = frames.shape[0]
    h, w = frames.shape[1], frames.shape[2]
    depth = np.empty((len(indices), h, w), dtype=np.float32)
    for k, fi in enumerate(np.asarray(indices)):
        i = int(fi)
        if i < 0 or i >= n:
            raise ValueError(f"index {i} out of range for {n} frames")
        d = np.asarray(predictor.predict(frames[i]), dtype=np.float32)
        if d.shape != (h, w):
            raise ValueError(f"predictor returned shape {d.shape}, expected {(h, w)}")
        if not np.all(np.isfinite(d)):
            raise ValueError(f"predictor returned non-finite depth for frame {i}")
        depth[k] = d
        step = k + 1
        log(f"depth {step}/{len(indices)} frame {i}")
        if on_step is not None:
            on_step(step)
    return depth


# --------------------------------------------------------------------------------------
# atomic, portable output
# --------------------------------------------------------------------------------------

def _same_path(a, b) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return os.path.abspath(str(a)) == os.path.abspath(str(b))


def write_depth_npz(path, *, indices: np.ndarray, depth: np.ndarray, model_id: str,
                    normalization: str, frames_npz_sha256: str) -> str:
    """Atomically write the depth cache with exactly the keys :func:`load_depth_npz` requires.

    The arrays are written to a sibling temp file, flushed and ``os.replace``d into place. Only
    path-independent scalars are stored: ``indices`` (int32), ``depth`` (float32), ``model_id``,
    ``normalization`` and ``frames_npz_sha256`` (scalars). No absolute paths or machine details.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = np.asarray(indices, dtype=np.int32)
    dep = np.asarray(depth, dtype=np.float32)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp.npz")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            np.savez(f, indices=idx, depth=dep,
                     model_id=np.array(str(model_id)),
                     normalization=np.array(str(normalization)),
                     frames_npz_sha256=np.array(str(frames_npz_sha256)))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return str(path)


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------

def resolve_lock_enabled(lock) -> bool:
    """Map the ``--lock`` CLI value (``on``/``off``) to ``HeavyLock(enabled=...)``.

    Only ``on``/``off`` are accepted: unlike ``flow_prediction_real`` there is no synthetic source, so an
    ``auto`` mode would be ambiguous. Any other value is a configuration error.
    """
    if lock not in ("on", "off"):
        raise ValueError(f"--lock must be 'on' or 'off', got {lock!r}")
    return lock == "on"


def run(opts, *, predictor=None, log: Callable[[str], None] = lambda _: None,
        lock_factory: Optional[Callable[..., object]] = None) -> dict:
    """Validate the frame cache, infer the sampled maps and write the portable depth cache.

    The frame cache is validated through :func:`bench.depth_temporal_pilot.load_frames_npz` and hashed with
    :func:`bench.manifest.sha256_file`. ``predictor`` defaults to a real :class:`DepthAnythingPredictor`
    (built *under the shared heavy lock*, after validation so a bad request fails before any weights are
    touched); tests inject a fake.

    The project's shared :class:`bench.heavylock.HeavyLock` is held while the predictor is constructed and
    while every frame is inferred, because both are heavy MPS work that must not race a film render. It is
    *not* held while loading the frame npz, hashing it, generating indices or writing the output -- those are
    cheap, local operations. With ``lock_chunk > 0`` the lock yields between completed frames via
    ``lock.on_step`` (one gate per validated frame). ``lock_factory`` exists so weight-free tests can inject
    a fake or a disabled lock; it defaults to the lazily imported :class:`bench.heavylock.HeavyLock`.

    Returns a JSON-safe, path-independent summary. Raises ``ValueError`` if ``--out`` would overwrite the
    input frame cache.
    """
    lock_chunk = int(getattr(opts, "lock_chunk", DEFAULT_LOCK_CHUNK) or 0)
    lock_pause = float(getattr(opts, "lock_pause", DEFAULT_LOCK_PAUSE) or 0.0)
    # Validate lock options before any model loading: a bad config must fail before weights are touched.
    if lock_chunk < 0:
        raise ValueError("--lock-chunk must be >= 0")
    if lock_pause < 0:
        raise ValueError("--lock-pause must be >= 0")
    lock_enabled = resolve_lock_enabled(getattr(opts, "lock", DEFAULT_LOCK))

    frames_path = Path(opts.frames_npz)
    out_path = Path(opts.out)
    if not frames_path.exists():
        raise FileNotFoundError(f"frames cache not found: {frames_path}")
    if _same_path(frames_path, out_path):
        raise ValueError("--out must not overwrite the input frames npz")

    # Everything up to the lock: cheap local I/O that must not hold the heavy lock.
    frames, _meta = load_frames_npz(frames_path)
    n = int(frames.shape[0])
    explicit = parse_indices(getattr(opts, "indices", None))
    indices = generate_indices(n, every=getattr(opts, "every", DEFAULT_EVERY), indices=explicit)
    frames_npz_sha256 = MF.sha256_file(frames_path)

    lock = _make_lock(lock_factory, lock_enabled, lock_chunk, lock_pause)
    # Hold the lock across predictor construction and every prediction. Real MPS inference is the heavy
    # work; construction downloads/loads weights, which is equally heavy.
    lock.acquire()
    try:
        if predictor is None:
            predictor = DepthAnythingPredictor(
                getattr(opts, "model_id", DEFAULT_MODEL_ID),
                device=getattr(opts, "device", DEFAULT_DEVICE),
                max_side=getattr(opts, "max_side", DEFAULT_MAX_SIDE),
                local_files_only=bool(getattr(opts, "local_files_only", False)))
        gate = lock.on_step if lock_chunk else None
        depth = build_depth_array(frames, indices, predictor, log=log, on_step=gate)
    finally:
        lock.release()

    # Writing output is cheap local I/O and happens outside the lock.
    write_depth_npz(out_path, indices=indices, depth=depth,
                    model_id=getattr(opts, "model_id", DEFAULT_MODEL_ID),
                    normalization=NORMALIZATION, frames_npz_sha256=frames_npz_sha256)

    result = {
        "status": "success",
        "schema": SCHEMA,
        "out": str(out_path),
        "sampled_frames": int(indices.size),
        "frames_npz_sha256": frames_npz_sha256,
        "model_id": str(getattr(opts, "model_id", DEFAULT_MODEL_ID)),
        "normalization": NORMALIZATION,
    }
    status = getattr(lock, "status", None)
    if callable(status):
        # Drop the machine-specific lock path so the returned summary stays path-independent.
        st = dict(status())
        st.pop("path", None)
        result["lock"] = st
    return result


def _make_lock(lock_factory, enabled: bool, chunk: int, pause: float):
    """Build the heavy lock, importing :mod:`bench.heavylock` lazily so this module stays light-importable."""
    if lock_factory is None:
        from .heavylock import HeavyLock as lock_factory
    return lock_factory(what="mlxstab depth_cache", enabled=enabled, chunk=chunk, pause_s=pause)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="bench/depth_cache.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames-npz", required=True, help="cached decoded-frame npz (frames/meta)")
    p.add_argument("--out", required=True, help="output relative-depth npz (atomic write)")
    p.add_argument("--every", type=int, default=DEFAULT_EVERY,
                   help="sample stride from frame 0 to the last eligible frame (>= 1)")
    p.add_argument("--indices", default=None,
                   help="explicit comma-separated strictly increasing frame indices (overrides --every)")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="transformers depth model id")
    p.add_argument("--device", choices=["auto", "mps", "cpu"], default=DEFAULT_DEVICE,
                   help="inference device; auto prefers MPS when available")
    p.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE,
                   help="resize longest side before inference (rounded to a multiple of 14)")
    p.add_argument("--local-files-only", action="store_true",
                   help="never hit the network; use only already-cached model files")
    p.add_argument("--lock", choices=["on", "off"], default=DEFAULT_LOCK,
                   help="shared heavy lock around MPS inference (default on); off for tests/synthetic-only")
    p.add_argument("--lock-chunk", type=int, default=DEFAULT_LOCK_CHUNK,
                   help="release/re-acquire the lock every N completed frames (0 = hold throughout)")
    p.add_argument("--lock-pause", type=float, default=DEFAULT_LOCK_PAUSE,
                   help="seconds to yield for waiting jobs between lock chunks (etiquette ~12 s)")
    p.add_argument("--dry-run", action="store_true", help="print resolved options and exit")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.dry_run:
        cfg = {
            "frames_npz": a.frames_npz, "out": a.out, "every": a.every,
            "indices": parse_indices(a.indices), "model_id": a.model_id, "device": a.device,
            "max_side": a.max_side, "local_files_only": a.local_files_only,
            "lock": a.lock, "lock_chunk": a.lock_chunk, "lock_pause": a.lock_pause,
        }
        print(json.dumps(cfg, indent=1, sort_keys=True))
        return 0
    try:
        result = run(a, log=lambda m: print(m, file=sys.stderr))
    except Exception as e:
        print(json.dumps({"status": "failure", "error": {"type": type(e).__name__, "message": str(e)}},
                         sort_keys=True))
        return 1
    print(json.dumps({"status": result["status"], "out": result["out"],
                      "sampled_frames": result["sampled_frames"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
