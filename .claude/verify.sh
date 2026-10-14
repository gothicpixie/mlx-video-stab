#!/usr/bin/env bash
# Fast, offline, weight-free checks: warp parity vs torch, path/geometry maths, stabilizer logic on a tiny synthetic clip
# (Farneback flow, no RAFT weights), RAFT parity on a random-init net when the reference clone is present.
# Weight-dependent tests (real raft-things EPE) run only if the weights exist, otherwise they skip.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
echo "verify: $($PY -c 'import mlx.core as mx;print("mlx", mx.default_device())')"
exec "$PY" -I -W ignore -m pytest -q -rs tests
