"""Regenerate the Python MLX golden for the Rust bidirectional RAFT flow parity test.

Run from the repository root with `.venv/bin/python
rust/raft-compat/tools/generate_flow_bidir_reference.py`.

The fixture exercises the complete fp32 path used by the compatibility API:
`RAFT.frame_features` on two deterministic 64x64 images, followed by
`RAFT.flow_bidir` for one shared refinement iteration. The feature grid is
8x8, so all four correlation levels remain non-empty. It records both images,
the forward and backward low-resolution flow, and both full-resolution flows
after the mask head and convex 8x upsampling. This is an uncompiled host-lookup
compatibility reference, not a performance measurement.
"""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlxstab.raft.model import load_raft


ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "vendor/weights/conv/raft-things.safetensors"
OUTPUT = ROOT / "rust/raft-compat/tests/fixtures/flow_bidir_reference.json"
HEIGHT = WIDTH = 64
ITERS = 1
LEVELS, RADIUS = 4, 4
# Synthetic subpixel translation used to make the flow non-zero without any
# random state. The second image samples the same analytic pattern shifted by
# this amount, so forward flow is approximately (+1.25, -0.75) pixels.
SHIFT_X, SHIFT_Y = 1.25, -0.75
# Rust side expected by the checked-in fixture (mlx-rs 0.32.0 pins MLX 0.32.2).
RUST_MLX_RS = "0.32.0"
RUST_MLX = "0.32.2"


def pattern(xs, ys):
    """Deterministic smooth 0..255 RGB pattern evaluated at fractional coordinates."""
    red = 127.5 + 70.0 * np.sin(0.21 * xs + 0.13 * ys) + 35.0 * np.cos(0.07 * xs - 0.17 * ys)
    green = 127.5 + 60.0 * np.cos(0.11 * xs - 0.19 * ys) + 30.0 * np.sin(0.23 * xs + 0.05 * ys)
    blue = 127.5 + 55.0 * np.sin(0.17 * xs + 0.29 * ys) + 40.0 * np.cos(0.09 * xs + 0.21 * ys)
    return np.stack([red, green, blue], axis=-1).clip(0.0, 255.0).astype(np.float32)


def flat(array):
    return np.asarray(array, dtype=np.float32).reshape(-1).tolist()


def main():
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    first = pattern(xs, ys)
    second = pattern(xs - SHIFT_X, ys - SHIFT_Y)

    model = load_raft(CHECKPOINT)
    a = model.frame_features(mx.array(first[None]))
    b = model.frame_features(mx.array(second[None]))
    low_f, up_f, low_b, up_b = model.flow_bidir(a, b, iters=ITERS)
    mx.eval(low_f, up_f, low_b, up_b)

    low_f = np.asarray(low_f, dtype=np.float32)
    up_f = np.asarray(up_f, dtype=np.float32)
    low_b = np.asarray(low_b, dtype=np.float32)
    up_b = np.asarray(up_b, dtype=np.float32)
    assert a.fmap.shape == (1, HEIGHT // 8, WIDTH // 8, 256), a.fmap.shape
    assert low_f.shape == (1, HEIGHT // 8, WIDTH // 8, 2), low_f.shape
    assert up_f.shape == (1, HEIGHT, WIDTH, 2), up_f.shape
    assert low_b.shape == low_f.shape
    assert up_b.shape == up_f.shape

    payload = {
        "schema": "mlxstab.raft_compat.flow_bidir/1",
        "source": "rust/raft-compat/tools/generate_flow_bidir_reference.py",
        "python_mlx": version("mlx"),
        "rust_mlx_rs": RUST_MLX_RS,
        "rust_mlx": RUST_MLX,
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "height": HEIGHT,
        "width": WIDTH,
        "iters": ITERS,
        "corr_levels": LEVELS,
        "corr_radius": RADIUS,
        "corr_dtype": "float32",
        "image_dtype": "float32 (0..255)",
        "synthetic_shift_xy": [SHIFT_X, SHIFT_Y],
        "notes": (
            "Deterministic analytic images; exercises RAFT.frame_features and "
            "RAFT.flow_bidir for one iteration, including separate forward/backward "
            "correlation pyramids, batch-2 update-block refinement, the mask head, "
            "and convex 8x upsampling. Host-side lookup reference; not compiled."
        ),
        "images": {
            "first_shape": list(first[None].shape),
            "first": flat(first[None]),
            "second_shape": list(second[None].shape),
            "second": flat(second[None]),
        },
        "low_forward_shape": list(low_f.shape),
        "low_forward": flat(low_f),
        "up_forward_shape": list(up_f.shape),
        "up_forward": flat(up_f),
        "low_backward_shape": list(low_b.shape),
        "low_backward": flat(low_b),
        "up_backward_shape": list(up_b.shape),
        "up_backward": flat(up_b),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
