"""RAFT timing, one backend per process:
    python -I bench/bench_raft.py {mlx|mps|cpu} H W [iters] [reps] [mode] [mask_schedule]
modes: pair    one direction, fp32, exactly the reference algorithm (default)
       bidir   forward + backward of the pair as one batch of 2, fp32
       bidir16 same, fp16 network (mlx: also fp16 correlation pyramid; torch: model.half())
mask_schedule:
       every_iteration  reference path (default for torch): the vendored RAFT forward computes the
                        upsampling mask and upsampled flow on every refinement iteration
       final_only       compute them once, after the last iteration; mlx always uses this schedule
Prints one JSON line. torch backends run the unmodified reference code (vendor/reference) with the real raft-things weights.
The torch reference computes the upsampling mask on every iteration and cannot be changed; final_only reproduces
the same forward in this file and asserts parity with the reference before timing."""
import json, sys, time
import numpy as np

backend, H, W = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
iters = int(sys.argv[4]) if len(sys.argv) > 4 else 12; reps = int(sys.argv[5]) if len(sys.argv) > 5 else 3
mode = sys.argv[6] if len(sys.argv) > 6 else "pair"
mask_arg = sys.argv[7] if len(sys.argv) > 7 else None
if mask_arg not in (None, "every_iteration", "final_only"):
    raise SystemExit(f"mask_schedule must be every_iteration or final_only, got {mask_arg!r}")
if backend == "mlx":
    if mask_arg == "every_iteration":
        raise SystemExit("mlx only implements the final_only mask schedule")
    mask_schedule = "final_only"
else:
    mask_schedule = mask_arg or "every_iteration"
rng = np.random.default_rng(0)
a = (rng.random((H, W, 3)) * 255).astype(np.float32); b = np.roll(a, (3, -5), (0, 1))


def torch_final_mask(tm, image1, image2, iters):
    """Vendored RAFT.forward(test_mode=True), but the mask head and 8x upsampling run only once."""
    from corr import CorrBlock
    image1 = (2 * (image1 / 255.0) - 1.0).contiguous(); image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
    fmap1, fmap2 = tm.fnet([image1, image2])
    corr_fn = CorrBlock(fmap1.float(), fmap2.float(), radius=tm.args.corr_radius)
    cnet = tm.cnet(image1)
    net, inp = torch.split(cnet, [tm.hidden_dim, tm.context_dim], dim=1)
    net, inp = torch.tanh(net), torch.relu(inp)
    coords0, coords1 = tm.initialize_flow(image1)
    ub = tm.update_block
    for _ in range(iters):
        coords1 = coords1.detach()
        motion_features = ub.encoder(coords1 - coords0, corr_fn(coords1))
        net = ub.gru(net, torch.cat([inp, motion_features], dim=1))
        coords1 = coords1 + ub.flow_head(net)
    low = coords1 - coords0
    return low, tm.upsample_flow(low, .25 * ub.mask(net))


parity = None
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
    def sync():
        if dev == "mps": torch.mps.synchronize()
    def run_reference():
        with torch.no_grad(): out = tm(I1, I2, iters=iters, test_mode=True)
        sync(); return out
    def run_final():
        with torch.no_grad(): out = torch_final_mask(tm, I1, I2, iters)
        sync(); return out
    if mask_schedule == "final_only":
        reference, got = run_reference(), run_final()   # untimed parity check on the exact benchmark inputs
        atol = rtol = 1e-5 if dt == torch.float32 else 1e-3
        parity = {}
        for name, expected, actual in zip(("low_max_abs", "high_max_abs"), reference, got):
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=lambda m: f"final_only {name}: {m}")
            parity[name] = float((actual - expected).abs().max())
        del reference, got
        run = run_final
    else:
        run = run_reference
run()
t = []
for _ in range(reps):
    s = time.perf_counter(); run(); t.append(time.perf_counter() - s)
result = {"backend": backend, "mode": mode, "mask_schedule": mask_schedule, "H": H, "W": W, "iters": iters,
          "sec_median": float(np.median(t)), "sec_all": t}
if parity is not None: result["parity"] = parity
print(json.dumps(result))
