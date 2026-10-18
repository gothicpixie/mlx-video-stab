"""Regenerate the Python MLX float16-update bidirectional flow golden.

The encoders, frame features, correlation pyramids, coordinate grid, and flow
accumulator stay float32. Only the update block's convolution parameters and
its runtime inputs are float16. Per iteration, the FP32 correlation lookup and
FP32 flow are cast to float16, the returned delta is cast to float32 before it
is added to the coordinates, and the final mask is cast to float32 before
convex upsampling. Run from the repository root with
`.venv/bin/python rust/raft-compat/tools/generate_flow_bidir_fp16_reference.py`.
"""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlxstab.raft.model import CorrBlock, coords_grid, load_raft, upsample_flow_batch


ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "vendor/weights/conv/raft-things.safetensors"
OUTPUT = ROOT / "rust/raft-compat/tests/fixtures/flow_bidir_fp16_reference.json"
HEIGHT = WIDTH = 64
ITERS_CASES = (1, 24)
LEVELS, RADIUS = 4, 4
SHIFT_X, SHIFT_Y = 1.25, -0.75
RUST_MLX_RS = "0.32.0"
RUST_MLX = "0.32.2"


def pattern(xs, ys):
    red = 127.5 + 70.0 * np.sin(0.21 * xs + 0.13 * ys) + 35.0 * np.cos(0.07 * xs - 0.17 * ys)
    green = 127.5 + 60.0 * np.cos(0.11 * xs - 0.19 * ys) + 30.0 * np.sin(0.23 * xs + 0.05 * ys)
    blue = 127.5 + 55.0 * np.sin(0.17 * xs + 0.29 * ys) + 40.0 * np.cos(0.09 * xs + 0.21 * ys)
    return np.stack([red, green, blue], axis=-1).clip(0.0, 255.0).astype(np.float32)


def flat(array):
    return np.asarray(array, dtype=np.float32).reshape(-1).tolist()


def flow_bidir_fp16_update(model, a, b, iters):
    """Float32 encoders/correlation/coordinates, float16 update block, float32 outputs."""
    cf = CorrBlock(a.fmap.astype(mx.float32), b.fmap.astype(mx.float32), LEVELS, RADIUS, mx.float32)
    cb = CorrBlock(b.fmap.astype(mx.float32), a.fmap.astype(mx.float32), LEVELS, RADIUS, mx.float32)
    _, h, w, _ = a.fmap.shape
    c0 = mx.concatenate([coords_grid(h, w)] * 2, 0)
    c1 = c0
    net = mx.concatenate([a.net, b.net], 0).astype(mx.float16)
    inp = mx.concatenate([a.inp, b.inp], 0).astype(mx.float16)
    for _ in range(iters):
        corr = mx.concatenate([f(c1[i:i + 1]) for i, f in enumerate([cf, cb])], 0).astype(mx.float16)
        net, delta = model.update_block.core(net, inp, corr, (c1 - c0).astype(mx.float16))
        c1 = c1 + delta.astype(mx.float32)
    mask = model.update_block.mask_head(net).astype(mx.float32)
    low = c1 - c0
    up = upsample_flow_batch(low, mask)
    return low[:1], up[:1], low[1:], up[1:]


def main():
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    first = pattern(xs, ys)
    second = pattern(xs - SHIFT_X, ys - SHIFT_Y)

    model = load_raft(CHECKPOINT)
    a = model.frame_features(mx.array(first[None]))
    b = model.frame_features(mx.array(second[None]))
    assert a.fmap.dtype == mx.float32 and a.net.dtype == mx.float32 and a.inp.dtype == mx.float32
    model.update_block.set_dtype(mx.float16)

    cases = []
    for iters in ITERS_CASES:
        low_f, up_f, low_b, up_b = flow_bidir_fp16_update(model, a, b, iters)
        mx.eval(low_f, up_f, low_b, up_b)
        assert low_f.dtype == mx.float32 and up_f.dtype == mx.float32
        low_f = np.asarray(low_f, dtype=np.float32)
        up_f = np.asarray(up_f, dtype=np.float32)
        low_b = np.asarray(low_b, dtype=np.float32)
        up_b = np.asarray(up_b, dtype=np.float32)
        assert low_f.shape == (1, HEIGHT // 8, WIDTH // 8, 2), low_f.shape
        assert up_f.shape == (1, HEIGHT, WIDTH, 2), up_f.shape
        cases.append({
            "iters": iters,
            "low_forward_shape": list(low_f.shape),
            "low_forward": flat(low_f),
            "up_forward_shape": list(up_f.shape),
            "up_forward": flat(up_f),
            "low_backward_shape": list(low_b.shape),
            "low_backward": flat(low_b),
            "up_backward_shape": list(up_b.shape),
            "up_backward": flat(up_b),
        })

    payload = {
        "schema": "mlxstab.raft_compat.flow_bidir_fp16/1",
        "source": "rust/raft-compat/tools/generate_flow_bidir_fp16_reference.py",
        "python_mlx": version("mlx"),
        "rust_mlx_rs": RUST_MLX_RS,
        "rust_mlx": RUST_MLX,
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "height": HEIGHT,
        "width": WIDTH,
        "verified_iters": list(ITERS_CASES),
        "corr_levels": LEVELS,
        "corr_radius": RADIUS,
        "corr_dtype": "float32",
        "coordinate_dtype": "float32",
        "flow_accumulator_dtype": "float32",
        "update_weight_dtype": "float16",
        "update_input_dtype": "float16",
        "returned_dtype": "float32",
        "image_dtype": "float32 (0..255)",
        "synthetic_shift_xy": [SHIFT_X, SHIFT_Y],
        "notes": (
            "Deterministic analytic images. FP32 encoders, features, correlation and flow "
            "accumulator; FP16 update-block weights/inputs; delta and final mask cast back "
            "to FP32. Host-side uncompiled reference for 1 and 24 iterations."
        ),
        "images": {
            "first_shape": list(first[None].shape),
            "first": flat(first[None]),
            "second_shape": list(second[None].shape),
            "second": flat(second[None]),
        },
        "cases": cases,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
