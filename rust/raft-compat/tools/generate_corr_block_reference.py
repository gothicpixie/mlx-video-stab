"""Regenerate the Python MLX golden for the Rust correlation-pyramid parity test.

Run from the repository root with `.venv/bin/python
rust/raft-compat/tools/generate_corr_block_reference.py`.

This fixture is checkpoint-independent: it uses deterministic feature maps and
exercises `mlxstab.raft.model.CorrBlock` construction, every stored pyramid
level, and all lookup values for fractional/out-of-bounds coordinates. The
feature maps are deliberately small and odd-sized so floor-sized 2x2 pooling
drops a trailing row and column. Both supported correlation storage dtypes
(float32 and float16) are recorded.
"""
import json
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlxstab.raft.model import CorrBlock


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "rust/raft-compat/tests/fixtures/corr_block_reference.json"
HEIGHT, WIDTH, CHANNELS = 7, 9, 5
LEVELS, RADIUS = 3, 2
# Rust side expected by the checked-in fixture (mlx-rs 0.32.0 pins MLX 0.32.2).
RUST_MLX_RS = "0.32.0"
RUST_MLX = "0.32.2"
CASES = (("float32", mx.float32), ("float16", mx.float16))


def feature(seed, shape):
    """Small deterministic NHWC feature map with varied positive/negative values."""
    values = np.arange(np.prod(shape), dtype=np.float32)
    return ((np.mod(values * (seed + 3) + 7 * seed, 31) - 15) / 8).reshape(shape)


def flat(array):
    return np.asarray(array, dtype=np.float32).reshape(-1).tolist()


def case(name, dtype, f1, f2, coords):
    """Build one CorrBlock case and materialize every output the Rust test checks."""
    block = CorrBlock(mx.array(f1), mx.array(f2), levels=LEVELS, radius=RADIUS, dtype=dtype)
    lookup = block(mx.array(coords))
    mx.eval(*block.pyr, lookup)
    return {
        "dtype": name,
        "pyramid_shapes": [list(level.shape) for level in block.pyr],
        "pyramid": [flat(level) for level in block.pyr],
        "lookup_shape": list(lookup.shape),
        "lookup": flat(lookup),
    }


def main():
    f1 = feature(1, (1, HEIGHT, WIDTH, CHANNELS))
    f2 = feature(2, (1, HEIGHT, WIDTH, CHANNELS))
    # A full 7x9 coordinate grid, matching CorrBlock's requirement that the
    # lookup count equals the feature-grid pair count. Deterministic fractional
    # jitter pushes several neighborhoods across the top/left/right/bottom edges.
    ys, xs = np.meshgrid(
        np.arange(HEIGHT, dtype=np.float32),
        np.arange(WIDTH, dtype=np.float32),
        indexing="ij",
    )
    jitter_x = np.array([0.25, -0.5, 0.75, -0.25, 0.5, 0.0, 0.125, 0.375, 0.625])
    jitter_y = np.array([0.75, -0.25, 0.5, 0.0, -0.5, 0.25, 0.125])
    coords = np.stack(
        [xs + jitter_x[None, :], ys + jitter_y[:, None]], axis=-1
    ).astype(np.float32)[None]

    payload = {
        "schema": "mlxstab.raft_compat.corr_block/2",
        "source": "rust/raft-compat/tools/generate_corr_block_reference.py",
        "python_mlx": version("mlx"),
        "rust_mlx_rs": RUST_MLX_RS,
        "rust_mlx": RUST_MLX,
        "checkpoint_independent": True,
        "notes": (
            "Deterministic NHWC feature maps; covers all-pairs correlation, "
            "sqrt(C) scaling, floor-sized 2x2 pooling over odd dimensions, and "
            "XY-order bilinear zero-padded lookups in both stored correlation dtypes."
        ),
        "feature_shape": list(f1.shape),
        "feature_dtype": "float32",
        "levels": LEVELS,
        "radius": RADIUS,
        "f1": flat(f1),
        "f2": flat(f2),
        "coords_shape": list(coords.shape),
        "coords": flat(coords),
        "cases": [case(name, dtype, f1, f2, coords) for name, dtype in CASES],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
