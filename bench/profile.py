"""Where does the time go?   python -I bench/profile.py raft H W [iters]     per-stage GPU time of one RAFT pair
                              python -I bench/profile.py pipeline W H T [downscale] [iters] [--chunk N]   full stabilizer on a synthetic clip

raft: every stage is evaluated separately (mx.eval) to get its GPU time; 'build' is the Python time to construct the lazy graph.
pipeline: wall time split into GPU flow, host sync/conversion, CPU fitting, planning (LP), render, decode/encode."""
import json, sys, time
import numpy as np
import mlx.core as mx


def T(fn):
    t = time.perf_counter(); r = fn(); return r, time.perf_counter() - t


def prof_raft(H, W, iters=12):
    from mlxstab.raft import model as M
    from mlxstab.__main__ import DEFAULT_WEIGHTS
    m = M.load_raft(DEFAULT_WEIGHTS)
    rng = np.random.default_rng(0)
    a = mx.array((rng.random((1, H, W, 3)) * 255).astype(np.float32)); b = mx.array(np.roll(np.array(a), (3, -5), (1, 2)))
    def once(prof):
        r = {}; ev = {"n": 0}
        def E(*x):
            ev["n"] += 1; mx.eval(*x)
        t0 = time.perf_counter()
        (f1, f2, net, inp), r["encode_build"] = T(lambda: m.encode(a, b)); _, r["encoders"] = T(lambda: E(f1, f2, net, inp))
        cb, r["corr_build_py"] = T(lambda: M.CorrBlock(f1, f2)); _, r["corr_pyramid"] = T(lambda: E(*cb.pyr))
        c0 = M.coords_grid(f1.shape[1], f1.shape[2]); c1 = c0
        look = upd = 0.0; lb = ub = 0.0
        for _ in range(iters):
            corr, tb = T(lambda: cb(c1)); _, te = T(lambda: E(corr)); lb += tb; look += te
            (net, mask, d), tb = T(lambda: m.update_block(net, inp, corr, c1 - c0)); _, te = T(lambda: E(net, mask, d)); ub += tb; upd += te
            c1 = c1 + d
        r["lookup_total"] = look; r["update_total"] = upd; r["lookup_build_py"] = lb; r["update_build_py"] = ub
        up, tb = T(lambda: M.upsample_flow(c1 - c0, mask)); _, r["upsample"] = T(lambda: E(up))
        r["wall"] = time.perf_counter() - t0; r["evals"] = ev["n"]
        return r
    once(0); r = once(1)
    # the shipped path: one mx.eval per iteration
    (m(a, b, iters)); t = time.perf_counter(); mx.eval(m(a, b, iters)[1]); shipped = time.perf_counter() - t
    # fully lazy: no per-iteration eval at all
    def lazy():
        f1, f2, net, inp = m.encode(a, b); cb = M.CorrBlock(f1, f2); c0 = M.coords_grid(f1.shape[1], f1.shape[2]); c1 = c0
        for _ in range(iters):
            net, mask, d = m.update_block(net, inp, cb(c1), c1 - c0); c1 = c1 + d
        return M.upsample_flow(c1 - c0, mask)
    mx.eval(lazy()); t = time.perf_counter(); mx.eval(lazy()); lz = time.perf_counter() - t
    r["it_lookup_ms"] = 1e3 * r["lookup_total"] / iters; r["it_update_ms"] = 1e3 * r["update_total"] / iters
    r["shipped_total"] = shipped; r["lazy_total"] = lz
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}


def prof_raft_fast(H, W, iters=12, dtype="float32", corr_dtype="float32"):
    """Same breakdown for the optimized bidirectional path (forward + backward of one pair as one batch-2 refinement)."""
    from mlxstab.raft import model as M
    from mlxstab.__main__ import DEFAULT_WEIGHTS
    dt, cd = getattr(mx, dtype), getattr(mx, corr_dtype)
    m = M.load_raft(DEFAULT_WEIGHTS)
    if dt != mx.float32: m.to_dtype(dt)
    rng = np.random.default_rng(0)
    a = mx.array((rng.random((1, H, W, 3)) * 255).astype(np.float32)); b = mx.array(np.roll(np.array(a), (3, -5), (1, 2)))
    def once():
        r = {}
        fa, r["frame_features(a)"] = T(lambda: (lambda x: (mx.eval(x.fmap, x.net, x.inp), x)[1])(m.frame_features(a)))
        fb, r["frame_features(b)"] = T(lambda: (lambda x: (mx.eval(x.fmap, x.net, x.inp), x)[1])(m.frame_features(b)))
        cf, _ = T(lambda: M.CorrBlock(fa.fmap.astype(mx.float32), fb.fmap.astype(mx.float32), 4, 4, cd)); _, r["corr_pyramid(fwd)"] = T(lambda: mx.eval(*cf.pyr))
        cb, _ = T(lambda: M.CorrBlock(fb.fmap.astype(mx.float32), fa.fmap.astype(mx.float32), 4, 4, cd)); _, r["corr_pyramid(bwd)"] = T(lambda: mx.eval(*cb.pyr))
        h, w = fa.fmap.shape[1:3]; c0 = mx.concatenate([M.coords_grid(h, w)] * 2, 0); c1 = c0 + 0.5
        net = mx.concatenate([fa.net, fb.net], 0); inp = mx.concatenate([fa.inp, fb.inp], 0)
        look = upd = 0.0
        step = m._core()
        for _ in range(iters):
            corr = mx.concatenate([cf(c1[:1]), cb(c1[1:])], 0).astype(dt); _, te = T(lambda: mx.eval(corr)); look += te
            (net, d), _ = T(lambda: step(net, inp, corr, (c1 - c0).astype(dt))); _, te = T(lambda: mx.eval(net, d)); upd += te
            c1 = c1 + d.astype(mx.float32)
        r["lookup (12 it, batch 2)"] = look; r["GRU update (12 it, batch 2, compiled)"] = upd
        up, _ = T(lambda: M.upsample_flow_batch(c1 - c0, m.update_block.mask_head(net).astype(mx.float32))); _, r["mask head + upsample"] = T(lambda: mx.eval(up))
        return r
    once(); r = once()
    def lazy():
        x = m.frame_features(a); y = m.frame_features(b)
        return m.flow_bidir(x, y, iters, corr_dtype=cd)
    mx.eval(*lazy()); t = time.perf_counter(); mx.eval(*lazy()); lz = time.perf_counter() - t
    r["TOTAL fwd+bwd, one lazy graph, one sync"] = lz
    return {k: round(v, 4) for k, v in r.items()}


class Timers(dict):
    def wrap(self, mod, name, label):
        f = getattr(mod, name)
        def g(*a, **k):
            t = time.perf_counter()
            try: return f(*a, **k)
            finally: self[label] = self.get(label, 0.0) + time.perf_counter() - t
        setattr(mod, name, g)


def prof_pipeline(W, H, Tn, ds=1, iters=12, chunk=0, gate=None, cfgkw=None):
    import os, tempfile
    from mlxstab import synth, media, motion as MO, stabilize as S
    from mlxstab.__main__ import DEFAULT_WEIGHTS
    d = tempfile.mkdtemp(); src = os.path.join(d, "in.mp4")
    vid, _ = synth.render(T=Tn, W=W, H=H); e = media.Encoder(src, W, H, 30, crf=12, preset="veryfast")
    for f in vid: e.write(f)
    e.close(); del vid
    tm = Timers()
    tm.wrap(MO, "observe", "observe (GPU sample + host copy)"); tm.wrap(MO, "fit_motion", "CPU fit (RANSAC, mesh, rows)")
    tm.wrap(S, "plan", "CPU plan (path LPs)"); tm.wrap(S, "render_frame", "render (GPU warp + host copy)")
    tm.wrap(media.Encoder, "write", "encode write (pipe)")
    kw = cfgkw or {}
    flow = S.RaftFlow(DEFAULT_WEIGHTS, iters, ds, True, getattr(mx, kw.get('corr_dtype', 'float32')), getattr(mx, kw.get('flow_dtype', 'float32')), kw.get('warm_iters') or None)
    orig = flow.__call__ if False else None
    cls_call = S.RaftFlow.__call__; bi = S.RaftFlow.bidir
    def timed_bi(self, a, b):
        t = time.perf_counter(); r = bi(self, a, b); tm["flow graph build (python, lazy)"] = tm.get("flow graph build (python, lazy)", 0.0) + time.perf_counter() - t; return r
    S.RaftFlow.bidir = timed_bi
    def timed(self, a, b, key="f"):
        t = time.perf_counter(); r = cls_call(self, a, b, key); tm["flow (RAFT, GPU, incl. its eval)"] = tm.get("flow (RAFT, GPU, incl. its eval)", 0.0) + time.perf_counter() - t; return r
    S.RaftFlow.__call__ = timed
    t0 = time.perf_counter()
    rep = S.stabilize(src, os.path.join(d, "o.mp4"), os.path.join(d, "m.mp4"), DEFAULT_WEIGHTS, S.Config(flow_iters=iters, flow_downscale=ds, **(cfgkw or {})), flow_fn=flow, log=lambda m: None, on_step=gate)
    wall = time.perf_counter() - t0
    acc = sum(tm.values())
    out = {k: round(v, 2) for k, v in tm.items()}; out["other (decode, python glue)"] = round(wall - acc, 2); out["wall"] = round(wall, 2)
    out["frames"] = rep["frames"]; out["fps_total"] = round(rep["frames"] / wall, 3)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "raftfast":
        print(json.dumps(prof_raft_fast(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), *(sys.argv[5:7]))))
    elif sys.argv[1] == "raft":
        print(json.dumps(prof_raft(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 12)))
    else:
        a = [x for x in sys.argv[2:] if not x.startswith("--")]
        print(json.dumps(prof_pipeline(int(a[0]), int(a[1]), int(a[2]), int(a[3]) if len(a) > 3 else 1, int(a[4]) if len(a) > 4 else 12, cfgkw=json.loads(a[5]) if len(a) > 5 else None)))
