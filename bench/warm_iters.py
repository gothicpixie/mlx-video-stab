"""Smallest iteration count that keeps the warm-started MLX flow within tolerance of the 12-iteration PyTorch reference.

    python -I bench/warm_iters.py ref      # run the torch reference on the demo frame pairs (CPU) and cache it in out/
    python -I bench/warm_iters.py mlx      # warm-started MLX flows for several iteration counts / dtypes, EPE vs the cached reference"""
import sys, json
from pathlib import Path
import numpy as np
from tests import ref as R

OUT = R.ROOT / "out/warm_ref.npz"
FRAMES = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]


def load():
    from PIL import Image
    return [np.array(Image.open(R.DEMO / f"frame_00{n}.png").convert("RGB"))[:432, :1024].astype(np.float32) for n in FRAMES]


if sys.argv[1] == "ref":
    import torch
    tm = R.torch_raft(); ims = load(); out = {}
    for k in range(len(ims) - 1):
        a = torch.from_numpy(ims[k].transpose(2, 0, 1).copy())[None]; b = torch.from_numpy(ims[k + 1].transpose(2, 0, 1).copy())[None]
        with torch.no_grad():
            _, f = tm(a, b, iters=12, test_mode=True); _, g = tm(b, a, iters=12, test_mode=True)
        out[f"f{k}"] = f[0].numpy().transpose(1, 2, 0); out[f"b{k}"] = g[0].numpy().transpose(1, 2, 0)
    OUT.parent.mkdir(exist_ok=True); np.savez(OUT, **out); print("saved", OUT)
else:
    import mlx.core as mx
    from mlxstab.raft.model import load_raft, pad8, unpad
    ref = np.load(OUT); ims = load()
    base = load_raft(R.ROOT / "vendor/weights/conv/raft-things.safetensors")
    res = {}
    for dtype in ("float32", "float16"):
        m = load_raft(R.ROOT / "vendor/weights/conv/raft-things.safetensors")
        if dtype == "float16": m.to_dtype(mx.float16)
        for n in (2, 3, 4, 6, 8, 12):
            pf = pb = None; epes = []
            feats = [m.frame_features(pad8(mx.array(im)[None])[0]) for im in ims]
            for k in range(len(ims) - 1):
                it = 12 if pf is None else n                                  # first pair has no warm start
                lf, uf, lb, ub = m.flow_bidir(feats[k], feats[k + 1], it, pf, pb)
                pf, pb = lf, lb; mx.eval(uf, ub)
                epes.append([float(np.sqrt(((np.array(uf[0]) - ref[f"f{k}"]) ** 2).sum(-1)).mean()), float(np.sqrt(((np.array(ub[0]) - ref[f"b{k}"]) ** 2).sum(-1)).mean())])
            e = np.array(epes[1:])                                             # warm-started pairs only
            res[f"{dtype}_it{n}"] = {"mean_epe": float(e.mean()), "max_epe": float(e.max())}
            print(dtype, n, res[f"{dtype}_it{n}"], flush=True)
    json.dump(res, open(R.ROOT / "out/warm_iters.json", "w"), indent=1)
