"""Regenerate the Python MLX golden used by the Rust cnet.conv1 parity test.

Run from the repository root with `.venv/bin/python
rust/raft-compat/tools/generate_cnet_conv1_reference.py`.
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
OUTPUT = ROOT / "rust/raft-compat/tests/fixtures/cnet_conv1_reference.json"
POSITIONS = ((0, 0), (2, 3), (4, 5))


def main():
    model = load_raft(CHECKPOINT)
    values = np.arange(9 * 11 * 3, dtype=np.float32)
    image = (((values % 29) - 14) / 14).reshape(1, 9, 11, 3)
    output = model.cnet.conv1(mx.array(image))
    mx.eval(output)
    samples = [np.array(output[0, y, x]).astype(np.float32).tolist() for y, x in POSITIONS]
    payload = {
        "schema": "mlxstab.raft_compat.cnet_conv1/1",
        "python_mlx": version("mlx"),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "input_shape": [1, 9, 11, 3],
        "positions_yx": POSITIONS,
        "outputs": samples,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
