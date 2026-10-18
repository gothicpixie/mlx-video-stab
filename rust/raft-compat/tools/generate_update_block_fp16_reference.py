"""Regenerate the Python MLX float16 update-block golden.

Only the update block's convolution weights/biases and its four runtime inputs
(net, inp, corr, flow) are float16. Inputs are cast in the same place as the
Rust opt-in path: after the FP32 features/correlation lookup and before the
first Conv2d. The mask head also runs in float16 and its output is cast back to
float32 for the fixture. Run from the repository root with
`.venv/bin/python rust/raft-compat/tools/generate_update_block_fp16_reference.py`.
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
OUTPUT = ROOT / "rust/raft-compat/tests/fixtures/update_block_fp16_reference.json"
HEIGHT, WIDTH = 3, 4
HIDDEN_CHANNELS = 128
CORR_CHANNELS = 4 * (2 * 4 + 1) ** 2
MASK_CHANNELS = 64 * 9
RUST_MLX_RS = "0.32.0"
RUST_MLX = "0.32.2"


def signal(count, scale, phase):
    i = np.arange(count, dtype=np.float32)
    return (scale * (np.sin(i * 0.37 + phase) + 0.5 * np.cos(i * 0.11 + 2 * phase))).astype(
        np.float32
    )


def flat(array):
    return np.asarray(array, dtype=np.float32).reshape(-1).tolist()


def tensor(shape, scale, phase):
    return signal(int(np.prod(shape)), scale, phase).reshape(shape)


def main():
    model = load_raft(CHECKPOINT)
    update_block = model.update_block
    update_block.set_dtype(mx.float16)

    net = tensor((1, HEIGHT, WIDTH, HIDDEN_CHANNELS), 0.6, 0.3)
    inp = tensor((1, HEIGHT, WIDTH, HIDDEN_CHANNELS), 0.6, 1.1)
    corr = tensor((1, HEIGHT, WIDTH, CORR_CHANNELS), 1.5, 2.0)
    flow = tensor((1, HEIGHT, WIDTH, 2), 0.4, 0.7)

    net16 = mx.array(net).astype(mx.float16)
    inp16 = mx.array(inp).astype(mx.float16)
    corr16 = mx.array(corr).astype(mx.float16)
    flow16 = mx.array(flow).astype(mx.float16)

    core_net, delta = update_block.core(net16, inp16, corr16, flow16)
    mask = update_block.mask_head(core_net).astype(mx.float32)
    full_net, full_mask, full_delta = update_block(net16, inp16, corr16, flow16)
    full_mask = full_mask.astype(mx.float32)
    mx.eval(core_net, delta, mask, full_net, full_mask, full_delta)
    np.testing.assert_array_equal(np.asarray(full_net), np.asarray(core_net))
    np.testing.assert_array_equal(np.asarray(full_mask), np.asarray(mask))
    np.testing.assert_array_equal(np.asarray(full_delta), np.asarray(delta))
    assert core_net.dtype == mx.float16 and delta.dtype == mx.float16

    payload = {
        "schema": "mlxstab.raft_compat.update_block_fp16/1",
        "python_mlx": version("mlx"),
        "rust_mlx_rs": RUST_MLX_RS,
        "rust_mlx": RUST_MLX,
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "height": HEIGHT,
        "width": WIDTH,
        "hidden_channels": HIDDEN_CHANNELS,
        "corr_channels": CORR_CHANNELS,
        "mask_channels": MASK_CHANNELS,
        "update_weight_dtype": "float16",
        "update_input_dtype": "float16",
        "returned_dtype": "float32",
        "inputs": {
            "net_shape": list(net.shape),
            "net": flat(net),
            "inp_shape": list(inp.shape),
            "inp": flat(inp),
            "corr_shape": list(corr.shape),
            "corr": flat(corr),
            "flow_shape": list(flow.shape),
            "flow": flat(flow),
        },
        "core": {
            "net_shape": list(core_net.shape),
            "net": flat(core_net),
            "delta_shape": list(delta.shape),
            "delta": flat(delta),
        },
        "mask_shape": list(mask.shape),
        "mask": flat(mask),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
