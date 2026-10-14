"""RAFT timing, one backend per process:  python -I bench/bench_raft.py {mlx|mps|cpu} H W [iters] [reps]
Prints one JSON line. torch backends use the unmodified reference code (vendor/reference) with the real raft-things weights."""
import json, sys, time
import numpy as np

backend, H, W = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
iters = int(sys.argv[4]) if len(sys.argv) > 4 else 12; reps = int(sys.argv[5]) if len(sys.argv) > 5 else 3
rng = np.random.default_rng(0)
a = (rng.random((H, W, 3)) * 255).astype(np.float32); b = np.roll(a, (3, -5), (0, 1))
if backend == "mlx":
    import mlx.core as mx
    from mlxstab.raft.model import load_raft
    from mlxstab.__main__ import DEFAULT_WEIGHTS
    m = load_raft(DEFAULT_WEIGHTS); A = mx.array(a)[None]; B = mx.array(b)[None]
    run = lambda: mx.eval(m(A, B, iters)[1])
else:
    import torch, sys as _s
    sys_path = __import__("tests.ref", fromlist=["x"])
    tm = sys_path.torch_raft()
    dev = "mps" if backend == "mps" else "cpu"
    tm = tm.to(dev); A = torch.from_numpy(a.transpose(2, 0, 1).copy())[None].to(dev); B = torch.from_numpy(b.transpose(2, 0, 1).copy())[None].to(dev)
    def run():
        with torch.no_grad(): tm(A, B, iters=iters, test_mode=True)
        if dev == "mps": torch.mps.synchronize()
run()                                           # warm-up (kernel compilation, allocator)
t = []
for _ in range(reps):
    s = time.perf_counter(); run(); t.append(time.perf_counter() - s)
print(json.dumps({"backend": backend, "H": H, "W": W, "iters": iters, "sec_median": float(np.median(t)), "sec_all": t}))
