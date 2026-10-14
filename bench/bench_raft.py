"""RAFT timing, one backend per process:
    python -I bench/bench_raft.py {mlx|mps|cpu} H W [iters] [reps] [mode]
modes: pair    one direction, fp32, exactly the reference algorithm (default)
       bidir   forward + backward of the pair as one batch of 2, fp32
       bidir16 same, fp16 network (mlx: also fp16 correlation pyramid; torch: model.half())
Prints one JSON line. torch backends run the unmodified reference code (vendor/reference) with the real raft-things weights.
The torch reference computes the upsampling mask on every iteration and cannot be changed; the MLX path computes it once."""
import json, sys, time
import numpy as np

backend, H, W = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
iters = int(sys.argv[4]) if len(sys.argv) > 4 else 12; reps = int(sys.argv[5]) if len(sys.argv) > 5 else 3
mode = sys.argv[6] if len(sys.argv) > 6 else "pair"
rng = np.random.default_rng(0)
a = (rng.random((H, W, 3)) * 255).astype(np.float32); b = np.roll(a, (3, -5), (0, 1))
if backend == "mlx":
    import mlx.core as mx
    from mlxstab.raft.model import load_raft
    from mlxstab.__main__ import DEFAULT_WEIGHTS
    m = load_raft(DEFAULT_WEIGHTS); A = mx.array(a)[None]; B = mx.array(b)[None]
    if mode == "pair": run = lambda: mx.eval(m(A, B, iters)[1])
    else:
        if mode == "bidir16": m.to_dtype(mx.float16)
        cd = mx.float16 if mode == "bidir16" else mx.float32
        def run():
            x = m.frame_features(A); y = m.frame_features(B)
            mx.eval(*m.flow_bidir(x, y, iters, corr_dtype=cd))
else:
    import torch
    tests_ref = __import__("tests.ref", fromlist=["x"])
    tm = tests_ref.torch_raft()
    dev = "mps" if backend == "mps" else "cpu"
    half = mode == "bidir16"
    if half:
        tm = tm.half()
        import torch.nn as nn                                # the reference mixes fp32 flow/coords with an fp16 net: cast conv inputs
        _f = nn.Conv2d.forward
        nn.Conv2d.forward = lambda self, x: _f(self, x.to(self.weight.dtype))
    tm = tm.to(dev)
    dt = torch.float16 if half else torch.float32
    ta = torch.from_numpy(a.transpose(2, 0, 1).copy())[None]; tb = torch.from_numpy(b.transpose(2, 0, 1).copy())[None]
    I1 = (torch.cat([ta, tb]) if mode != "pair" else ta).to(dev, dt); I2 = (torch.cat([tb, ta]) if mode != "pair" else tb).to(dev, dt)
    def run():
        with torch.no_grad(): tm(I1, I2, iters=iters, test_mode=True)
        if dev == "mps": torch.mps.synchronize()
run()
t = []
for _ in range(reps):
    s = time.perf_counter(); run(); t.append(time.perf_counter() - s)
print(json.dumps({"backend": backend, "mode": mode, "H": H, "W": W, "iters": iters, "sec_median": float(np.median(t)), "sec_all": t}))
