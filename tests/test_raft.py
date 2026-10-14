import numpy as np, pytest
import mlx.core as mx
torch = pytest.importorskip("torch")
from . import ref
pytestmark = pytest.mark.skipif(not ref.available, reason="vendor/reference/RAFT not present")
from mlxstab.raft import model as M


def nhwc(t): return mx.array(t.detach().numpy().transpose(0, 2, 3, 1))
def err(a, t):
    """max abs error, relative to max(1, max|reference|) so large-valued tensors (correlations, flows) aren't judged in absolute terms"""
    t = t.detach().numpy().transpose(0, 2, 3, 1)
    return float(np.abs(np.array(a) - t).max() / max(1.0, np.abs(t).max()))


def stages(tm, mm, a, b, tol):
    """Per-stage max abs error between torch (NCHW) and MLX (NHWC)."""
    ta = torch.from_numpy(a.transpose(0, 3, 1, 2).copy()); tb = torch.from_numpy(b.transpose(0, 3, 1, 2).copy())
    res = {}
    with torch.no_grad():
        n1 = 2 * (ta / 255) - 1; n2 = 2 * (tb / 255) - 1
        f1, f2 = tm.fnet([n1, n2]); c = tm.cnet(n1)
        net_t, inp_t = torch.tanh(c[:, :128]), torch.relu(c[:, 128:])
        m1, m2, net_m, inp_m = mm.encode(mx.array(a), mx.array(b))
        res["fnet"] = max(err(m1, f1), err(m2, f2)); res["cnet_net"] = err(net_m, net_t); res["cnet_inp"] = err(inp_m, inp_t)
        from corr import CorrBlock
        ct = CorrBlock(f1, f2, radius=4); cm = M.CorrBlock(nhwc(f1), nhwc(f2))
        for i in (0, 3):
            r_ = ct.corr_pyramid[i].numpy().reshape(-1)
            res[f"corr_pyr{i}"] = float(np.abs(np.array(cm.pyr[i]).reshape(-1) - r_).max() / max(1.0, np.abs(r_).max()))
        h, w = f1.shape[2:]
        co = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")[::-1], 0).float()[None] + torch.randn(1, 2, h, w) * 2
        res["corr_lookup"] = err(cm(nhwc(co)), ct(co))
        # one update step from identical inputs
        flow = torch.randn(1, 2, h, w)
        corr_t = ct(co)
        nt, mt, dt = tm.update_block(net_t, inp_t, corr_t, flow)
        nm, mm_, dm = mm.update_block(net_m, inp_m, nhwc(corr_t), nhwc(flow))
        res["update_net"] = err(nm, nt); res["update_delta"] = err(dm, dt); res["update_mask"] = err(mm_, mt)
        res["upsample"] = err(M.upsample_flow(nhwc(flow), mm_), tm.upsample_flow(flow, mt))
        low_t, up_t = tm(ta, tb, iters=12, test_mode=True)
    low_m, up_m = mm(mx.array(a), mx.array(b), 12)
    res["final_low"] = err(low_m, low_t); res["final_flow"] = err(up_m, up_t)
    epe = float(np.sqrt(((np.array(up_m) - up_t.numpy().transpose(0, 2, 3, 1)) ** 2).sum(-1)).mean())
    res["EPE_vs_torch"] = epe
    return res


def synth(h=136, w=160, seed=0):  # >=16 cells at 1/8 so the coarsest corr level is >1 px (the reference NaNs on 1-px levels)
    rng = np.random.default_rng(seed)
    base = rng.random((h + 16, w + 16, 3)).astype(np.float32)
    import scipy.ndimage as nd
    base = np.stack([nd.gaussian_filter(base[..., i], 2.0) for i in range(3)], -1); base = (base - base.min()) / np.ptp(base) * 255
    a = base[8:8 + h, 8:8 + w]; b = base[8 + 2:8 + 2 + h, 8 - 3:8 - 3 + w]            # shift (+3, -2) px
    return a[None].astype(np.float32), b[None].astype(np.float32)


def test_random_init_stages():
    tm = ref.torch_raft(random_init_seed=0); mm = ref.convert_in_memory(tm)
    a, b = synth()
    r = stages(tm, mm, a, b, 0)
    print({k: f"{v:.2e}" for k, v in r.items()})
    for k, v in r.items(): assert v < 2e-3, (k, v)


@pytest.mark.skipif(not ref.WEIGHTS.exists(), reason="raft-things weights absent")
def test_real_weights_synthetic_and_demo():
    from PIL import Image
    tm = ref.torch_raft(); mm = ref.convert_in_memory(tm)
    a, b = synth(160, 192)
    r = stages(tm, mm, a, b, 0); print("synthetic", {k: f"{v:.2e}" for k, v in r.items()})
    assert r["EPE_vs_torch"] < 0.01
    ims = [np.array(Image.open(ref.DEMO / f"frame_00{n}.png").convert("RGB"))[None].astype(np.float32) for n in (16, 17)]
    ims = [x[:, :432, :1024] for x in ims]
    r = stages(tm, mm, *ims, 0); print("demo", {k: f"{v:.2e}" for k, v in r.items()})
    assert r["EPE_vs_torch"] < 0.01


def _epe(a, b): return float(np.sqrt(((np.array(a) - np.array(b)) ** 2).sum(-1)).mean())


def test_fast_paths_match_reference_path_random_init():
    """New lookup kernel == grid_sample lookup; batch-2 bidirectional compiled path == two independent fp32 calls."""
    tm = ref.torch_raft(random_init_seed=3); mm = ref.convert_in_memory(tm)
    a, b = synth(); A, B = mx.array(a), mx.array(b)
    f1, f2, net, inp = mm.encode(A, B)
    cb = M.CorrBlock(f1, f2); c = M.coords_grid(*f1.shape[1:3]) + mx.array(np.random.default_rng(0).normal(0, 3, (1, *f1.shape[1:3], 2)).astype(np.float32))
    d = float(mx.abs(cb(c) - cb.lookup_reference(c)).max()); assert d < 1e-3, d
    lf, uf = mm(A, B, 6); lb, ub = mm(B, A, 6)
    x, y = mm.frame_features(A), mm.frame_features(B)
    blf, buf, blb, bub = mm.flow_bidir(x, y, 6)
    assert _epe(uf[0], buf[0]) < 1e-3 and _epe(ub[0], bub[0]) < 1e-3, (_epe(uf[0], buf[0]), _epe(ub[0], bub[0]))
    # compiled and eager update steps agree
    l2, u2 = mm(A, B, 6, compiled=False); assert _epe(uf[0], u2[0]) < 1e-3


@pytest.mark.skipif(not ref.WEIGHTS.exists(), reason="raft-things weights absent")
def test_real_weights_bidir_and_fp16():
    from PIL import Image
    tm = ref.torch_raft(); mm = ref.convert_in_memory(tm)
    ims = [np.array(Image.open(ref.DEMO / f"frame_00{n}.png").convert("RGB"))[:432, :1024].astype(np.float32) for n in (16, 17)]
    ta = [torch.from_numpy(i.transpose(2, 0, 1).copy())[None] for i in ims]
    with torch.no_grad():
        _, tf = tm(ta[0], ta[1], iters=12, test_mode=True); _, tb = tm(ta[1], ta[0], iters=12, test_mode=True)
    tf, tb = tf[0].numpy().transpose(1, 2, 0), tb[0].numpy().transpose(1, 2, 0)
    A, B = mx.array(ims[0])[None], mx.array(ims[1])[None]
    x, y = mm.frame_features(A), mm.frame_features(B)
    _, uf, _, ub = mm.flow_bidir(x, y, 12)
    e32 = max(_epe(uf[0], tf), _epe(ub[0], tb)); print("bidir fp32 EPE vs torch", e32)
    assert e32 < 0.01
    mh = ref.convert_in_memory(tm).to_dtype(mx.float16)
    x, y = mh.frame_features(A), mh.frame_features(B)
    _, uf, _, ub = mh.flow_bidir(x, y, 12)
    e16 = max(_epe(uf[0], tf), _epe(ub[0], tb)); print("bidir fp16 EPE vs torch", e16)
    assert e16 < 0.05
