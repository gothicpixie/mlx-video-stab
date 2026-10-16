"""Research-only spatial border-support correction for captured stabilization plans.

The candidate blends the rendered source map back toward its global-only map only near pixels where the complete
mesh/row map leaves the source image while the global map remains supported. It does not change production defaults
or the captured plan. Temporal support union keeps a transient invalid pixel from switching the local correction
on and off on adjacent frames.
"""

from collections import deque

import numpy as np
from scipy.ndimage import distance_transform_edt


def _source_valid(src, width, height):
    src = np.asarray(src)
    if src.ndim != 3 or src.shape[-1] != 2:
        raise ValueError("source map must have shape [H,W,2]")
    x, y = src[..., 0], src[..., 1]
    return (np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (x <= width - 1)
            & (y >= 0.0) & (y <= height - 1))


def local_support_failure(full_src, global_src, width, height):
    """Return pixels locally recoverable by dropping residuals, and pixels global motion cannot recover."""
    full_src = np.asarray(full_src)
    global_src = np.asarray(global_src)
    if full_src.shape != global_src.shape or full_src.ndim != 3 or full_src.shape[-1] != 2:
        raise ValueError("full and global maps must share shape [H,W,2]")
    if full_src.shape[:2] != (height, width):
        raise ValueError("source-map dimensions disagree with width/height")
    full_valid = _source_valid(full_src, width, height)
    global_valid = _source_valid(global_src, width, height)
    return ~full_valid & global_valid, ~full_valid & ~global_valid


def local_support_alpha(unsupported, radius_px):
    """Build a smooth output-space taper: zero at unsupported pixels, one at least radius_px away.

    ``unsupported`` is the temporal union of recoverable support failures for the current frame and its temporal
    neighbours. The Euclidean distance field makes the correction spatially continuous at the mask boundary.
    """
    bad = np.asarray(unsupported, dtype=bool)
    if bad.ndim != 2 or not bad.size:
        raise ValueError("unsupported mask must be a non-empty [H,W] array")
    radius = float(radius_px)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_px must be finite and > 0")
    if not bad.any():
        return np.ones(bad.shape, np.float32)
    distance = distance_transform_edt(~bad)
    return np.clip(distance / radius, 0.0, 1.0).astype(np.float32)


def blend_source_maps(full_src, global_src, alpha):
    """Blend complete and global-only maps using an ``[H,W]`` local correction weight.

    Non-finite complete-map entries are replaced by the global map before blending. If the global map is also
    invalid, the resulting point stays invalid and is reported by :func:`local_support_failure`.
    """
    full = np.asarray(full_src, dtype=np.float32)
    glob = np.asarray(global_src, dtype=np.float32)
    weight = np.asarray(alpha, dtype=np.float32)
    if full.shape != glob.shape or full.ndim != 3 or full.shape[-1] != 2:
        raise ValueError("full and global maps must share shape [H,W,2]")
    if weight.shape != full.shape[:2] or not np.isfinite(weight).all():
        raise ValueError("alpha must be a finite [H,W] array matching the maps")
    if np.any((weight < 0.0) | (weight > 1.0)):
        raise ValueError("alpha values must be in [0,1]")
    full_safe = np.where(np.isfinite(full), full, glob)
    return glob + weight[..., None] * (full_safe - glob)


class TemporalSupportWindow:
    """Bounded-memory centered OR window for per-frame support-failure masks.

    Push masks in frame order. Once a frame has ``radius`` future masks, ``push`` returns the oldest frame and the
    union covering its clipped ``[t-radius, t+radius]`` temporal neighbourhood. Call :meth:`flush` at end of input.
    The class retains only ``2*radius+1`` masks, independent of clip length.
    """

    def __init__(self, radius=0):
        if isinstance(radius, bool) or not isinstance(radius, (int, np.integer)) or radius < 0:
            raise ValueError("temporal radius must be a non-negative integer")
        self.radius = int(radius)
        self._history = deque(maxlen=self.radius)
        self._pending = deque()
        self._shape = None

    def push(self, item):
        mask = np.asarray(item, dtype=bool)
        if mask.ndim != 2 or not mask.size:
            raise ValueError("support mask must be a non-empty [H,W] array")
        if self._shape is None:
            self._shape = mask.shape
        elif mask.shape != self._shape:
            raise ValueError("support mask dimensions changed within a sequence")
        self._pending.append(mask.copy())
        if len(self._pending) <= self.radius:
            return None
        return self._emit_oldest(self.radius)

    def _emit_oldest(self, future_count):
        masks = list(self._history) + list(self._pending)[:future_count + 1]
        union = np.logical_or.reduce(masks)
        oldest = self._pending.popleft()
        if self.radius:
            self._history.append(oldest)
        return oldest, union

    def flush(self):
        """Emit all remaining masks using the available (clipped) future context."""
        while self._pending:
            yield self._emit_oldest(min(self.radius, len(self._pending) - 1))
