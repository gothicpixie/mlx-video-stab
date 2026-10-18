"""Regenerate the Python MLX golden for the Rust full BasicEncoder parity test.

Run from the repository root with `.venv/bin/python
rust/raft-compat/tools/generate_basic_encoder_reference.py`.

The input is an already normalized NHWC tensor, exactly what `RAFT.encode`
passes to `fnet`/`cnet` after the 0..255 -> -1..1 transform. Both encoders run
with the model in evaluation mode, so `cnet` uses its checkpointed running
statistics. This fixture covers the two encoders only; it is not full RAFT
parity.
"""
import hashlib
import json
from pathlib import Path
from importlib.metadata import version

import mlx.core as mx
import numpy as np

from mlxstab.raft.model import load_raft


ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "vendor/weights/conv/raft-things.safetensors"
OUTPUT = ROOT / "rust/raft-compat/tests/fixtures/basic_encoder_reference.json"
HEIGHT, WIDTH = 32, 40
# Versions of the Rust side that generated the current expected fixture.
# mlx-rs 0.32.0 is built against mlx-sys 0.6.0, which pins MLX 0.32.2.
RUST_MLX_RS = "0.32.0"
RUST_MLX = "0.32.2"


def main():
    model = load_raft(CHECKPOINT)
    values = (np.arange(HEIGHT * WIDTH * 3, dtype=np.float32) % 29 - 14) / 14
    image = values.reshape(1, HEIGHT, WIDTH, 3)
    pair = np.concatenate((image, np.flip(image, axis=2).copy()), axis=0)
    x = mx.array(pair)
    # Production RAFT encodes both frames together for fnet, then runs cnet on
    # the first frame only. Preserve those batch shapes in the parity fixture.
    fnet = model.fnet(x)
    cnet = model.cnet(x[:1])
    mx.eval(fnet, cnet)
    fnet = np.array(fnet).astype(np.float32)
    cnet = np.array(cnet).astype(np.float32)
    assert fnet.shape == (2, HEIGHT // 8, WIDTH // 8, 256), fnet.shape
    assert cnet.shape == (1, HEIGHT // 8, WIDTH // 8, 256), cnet.shape
    payload = {
        "schema": "mlxstab.raft_compat.basic_encoder/1",
        "python_mlx": version("mlx"),
        "rust_mlx_rs": RUST_MLX_RS,
        "rust_mlx": RUST_MLX,
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "input_shape": list(pair.shape),
        "input_normalized": True,
        "input": pair.reshape(-1).tolist(),
        "encoders": {
            "fnet": {"norm": "instance", "output_shape": list(fnet.shape), "output": fnet.reshape(-1).tolist()},
            "cnet": {"norm": "batch", "output_shape": list(cnet.shape), "output": cnet.reshape(-1).tolist()},
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
