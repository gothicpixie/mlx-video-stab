import numpy as np, pytest
cv2 = pytest.importorskip("cv2")
import mlx.core as mx
from mlxstab import synth, stabilize as S, motion as MO


def farneback(a, b, key=None):
    g = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    f = cv2.calcOpticalFlowFarneback(g(a), g(b), None, 0.5, 4, 21, 5, 7, 1.5, 0)
    return mx.array(f.astype(np.float32))


def jitter_px(vid, cfg):
    """mean |frame-to-frame residual motion| of the global translation (high-pass of per-frame velocity)."""
    from scipy.ndimage import gaussian_filter1d
    mot = S.analyze(iter(vid), farneback, cfg, vid.shape[2], vid.shape[1], 30.0)
    v = np.array([f.H[:2, 2] for f in mot.fits])
    return float(np.abs(v - gaussian_filter1d(v, 8, axis=0, mode="nearest")).mean())


def test_stabilizer_reduces_jitter_and_masks_border():
    vid, _ = synth.render(T=80, W=256, H=144, jitter=6.0)
    cfg = S.Config(budget=0.06, step=8, mesh=(8, 4), rs_rows=8)
    mot = S.analyze(iter(vid), farneback, cfg, 256, 144, 30.0)
    assert sum(f.fallback for f in mot.fits) < 10
    pl = S.plan(mot, cfg)
    out = []; masks = []
    for t in range(len(pl.G)):
        im, bad, d = S.render_frame(vid[t], pl, t, cfg); out.append(im); masks.append(bad)
        assert d > 0.9
    out = np.stack(out)
    jin, jout = jitter_px(vid[:len(out)], cfg), jitter_px(out, cfg)
    print("jitter in/out", jin, jout, "border", np.mean(masks))
    assert jout < 0.5 * jin
    assert 0 < np.mean(masks) < 0.2                                           # a border exists, and it is bounded by the budget
    assert np.all(out[np.array(masks)] == 0)                                  # masked pixels carry no content


def test_roll_levelling_sign():
    from mlxstab import metrics
    vid, _ = synth.render(T=30, W=256, H=144, jitter=0.0, rot_jit=0.0, subject=False)
    a = np.deg2rad(5.0); H_, W_ = 144, 256
    ys, xs = np.mgrid[0:H_, 0:W_].astype(np.float32); dx, dy = xs - W_ / 2, ys - H_ / 2
    c = np.stack([dx * np.cos(a) - dy * np.sin(a) + W_ / 2, dx * np.sin(a) + dy * np.cos(a) + H_ / 2], -1)      # content rotated visually CCW by 5 deg
    from mlxstab.warp import sample
    tilted = np.stack([np.array(sample(mx.array(f.astype(np.float32))[None], mx.array(c)[None], "bilinear", "border")[0]).astype(np.uint8) for f in vid])
    lean_in = metrics.vline_wobble  # noqa: just to make sure cv2 path imports
    def lean(frames):
        import cv2
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD); r = []
        for fr in frames[::5]:
            ls = lsd.detect(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY))[0]; a_ = []
            for x1, y1, x2, y2 in ls[:, 0]:
                ddx, ddy = float(x2 - x1), float(y1 - y2)
                if ddy < 0: ddx, ddy = -ddx, -ddy
                if np.hypot(ddx, ddy) > 35 and abs(np.degrees(np.arctan2(-ddx, ddy))) < 20: a_.append(np.degrees(np.arctan2(-ddx, ddy)))
            r.append(np.median(a_))
        return float(np.median(r))
    cfg = S.Config(budget=0.12, mesh=(0, 0), rs_rows=0)
    mot = S.analyze(iter(tilted), farneback, cfg, W_, H_, 30.0)
    pl = S.plan(mot, cfg, ("frames", np.full(30, 5.0)))
    out = np.stack([S.render_frame(tilted[t], pl, t, cfg)[0] for t in range(len(pl.G))])
    li, lo = lean(tilted), lean(out)
    print("lean in/out", li, lo)
    assert li > 3.5 and abs(lo) < 1.5


def test_analysis_stride_interpolates_path():
    vid, _ = synth.render(T=61, W=256, H=144, jitter=6.0)
    cfg = S.Config(budget=0.06, mesh=(8, 4), rs_rows=8, analysis_stride=2)
    mot = S.analyze(iter(vid), farneback, cfg, 256, 144, 30.0, stride=2)
    assert len(mot.fits) == 30 and mot.n_frames == 61
    pl = S.plan(mot, cfg, n_frames=mot.n_frames)
    assert len(pl.G) == 61 and pl.mesh_corr.shape[0] == 61
    out = np.stack([S.render_frame(vid[t], pl, t, cfg)[0] for t in range(61)])
    jin, jout = jitter_px(vid, cfg), jitter_px(out, cfg)
    print("stride-2 jitter in/out", jin, jout)
    assert jout < 0.6 * jin


def test_stabilize_report_includes_warp_diag(monkeypatch):
    """The stabilize report path emits complete-warp diagnostics when opt-in, preserving the old global metric."""
    import json
    from mlxstab import media
    vid, _ = synth.render(T=8, W=96, H=64, jitter=3.0)
    H, W = 64, 96
    monkeypatch.setattr(media, "probe", lambda p: {"width": W, "height": H, "fps": 30.0,
                                                    "field_order": "unknown", "interlaced": False, "duration": 0.0})

    def fake_decode(path, info=None, start=None, duration=None, deinterlace=True, vf=None):
        return iter(vid), 30.0, (W, H)

    monkeypatch.setattr(media, "decode", fake_decode)
    monkeypatch.setattr(media, "prefetch", lambda gen, n=4: gen)

    class FakeEncoder:
        def __init__(self, *a, **k): pass
        def write(self, frame): pass
        def close(self): pass

    monkeypatch.setattr(media, "Encoder", FakeEncoder)
    cfg = S.Config(mesh=(4, 3), rs_rows=4, budget=0.06, warp_diag=True)
    rep = S.stabilize("in.mp4", "out.mp4", None, None, cfg, flow_fn=farneback, log=lambda m: None)
    assert rep["distortion_scope"] == "global_homography_only"          # old field explicitly labelled, not redefined
    assert rep["distortion_min"] is not None and rep["distortion_mean"] is not None
    assert rep["warp_diag"] is not None
    assert "coverage.center_valid_frac" in rep["warp_diag"]
    assert "shape.anisotropy_mean" in rep["warp_diag"]
    json.dumps(rep, allow_nan=False)                                    # JSON-safe, including the diagnostics


def test_stabilize_dumps_plan_npz_without_pickle(tmp_path, monkeypatch):
    """``dump_plan`` writes the computed Plan as a compressed, non-pickle NPZ through stabilize."""
    from mlxstab import media
    vid, _ = synth.render(T=8, W=96, H=64, jitter=3.0)
    H, W = 64, 96
    monkeypatch.setattr(media, "probe", lambda p: {"width": W, "height": H, "fps": 30.0,
                                                    "field_order": "unknown", "interlaced": False, "duration": 0.0})
    monkeypatch.setattr(media, "decode", lambda *a, **k: (iter(vid), 30.0, (W, H)))
    monkeypatch.setattr(media, "prefetch", lambda gen, n=4: gen)

    class FakeEncoder:
        def __init__(self, *a, **k): pass
        def write(self, frame): pass
        def close(self): pass

    monkeypatch.setattr(media, "Encoder", FakeEncoder)
    cfg = S.Config(mesh=(4, 3), rs_rows=4, budget=0.06)
    out = tmp_path / "plan.npz"
    S.stabilize("in.mp4", "out.mp4", None, None, cfg, flow_fn=farneback, log=lambda m: None, dump_plan=str(out))

    assert out.exists() and out.stat().st_size > 0
    with np.load(out, allow_pickle=False) as z:                        # loads without unpickling any object
        assert set(z.files) == {"schema", "width", "height", "frames", "fps", "G",
                                "mesh_corr", "row_corr", "raw_quads", "path_quads", "fallbacks",
                                "sampled_border_scale", "sampled_border_worst_frac", "sampled_border_measured"}
        assert str(z["schema"]) == S.PLAN_SCHEMA
        assert float(z["sampled_border_scale"]) == 1.0 and bool(z["sampled_border_measured"]) is False
        assert int(z["width"]) == W and int(z["height"]) == H
        T = int(z["frames"])
        assert T == len(vid)
        assert float(z["fps"]) == pytest.approx(30.0)
        assert z["G"].shape == (T, 3, 3) and np.isfinite(z["G"]).all()
        assert z["mesh_corr"].shape == (T, 4, 5, 2)                   # gr+1 rows, gc+1 cols
        assert z["row_corr"].shape == (T, 4, 2)
        assert z["raw_quads"].shape == (T, 4, 2) and z["path_quads"].shape == (T, 4, 2)
        assert int(z["fallbacks"]) == 0
        assert all(z[k].dtype.kind in "fiuUb" for k in z.files)        # numeric/boolean/unicode only, no object arrays


def test_stabilize_dumps_motion_npz_without_pickle(tmp_path, monkeypatch):
    """``dump_motion_npz`` writes the fitted Motion as a compressed, non-pickle NPZ through stabilize."""
    from mlxstab import media
    vid, _ = synth.render(T=8, W=96, H=64, jitter=3.0)
    H, W = 64, 96
    monkeypatch.setattr(media, "probe", lambda p: {"width": W, "height": H, "fps": 30.0,
                                                    "field_order": "unknown", "interlaced": False, "duration": 0.0})
    monkeypatch.setattr(media, "decode", lambda *a, **k: (iter(vid), 30.0, (W, H)))
    monkeypatch.setattr(media, "prefetch", lambda gen, n=4: gen)

    class FakeEncoder:
        def __init__(self, *a, **k): pass
        def write(self, frame): pass
        def close(self): pass

    monkeypatch.setattr(media, "Encoder", FakeEncoder)
    cfg = S.Config(mesh=(4, 3), rs_rows=4, budget=0.06)
    out = tmp_path / "motion.npz"
    S.stabilize("in.mp4", "out.mp4", None, None, cfg, flow_fn=farneback, log=lambda m: None,
                dump_motion_npz=str(out))

    assert out.exists() and out.stat().st_size > 0
    with np.load(out, allow_pickle=False) as z:                        # loads without unpickling any object
        assert set(z.files) == {"schema", "width", "height", "n_frames", "fps", "stride", "n_fits",
                                "H", "mesh", "rows", "inlier_frac", "mover_frac", "fallback",
                                "n_valid_samples", "n_ransac_inliers", "fallback_reason_code", "fit_meta_available",
                                "mesh_support_mass", "row_support_mass",
                                "cycle_err_rms", "cycle_inconsistency_frac", "cycle_sample_mass",
                                "cycle_map_available"}
        assert str(z["schema"]) == S.MOTION_SCHEMA == "mlxstab.motion/4"
        assert int(z["width"]) == W and int(z["height"]) == H
        assert int(z["n_frames"]) == len(vid)
        assert int(z["stride"]) == 1
        assert float(z["fps"]) == pytest.approx(30.0)
        n = int(z["n_fits"])
        assert n == len(vid) - 1
        assert z["H"].shape == (n, 3, 3) and np.isfinite(z["H"]).all()
        assert z["mesh"].shape == (n, 4, 5, 2)                       # gr+1 rows, gc+1 cols
        assert z["rows"].shape == (n, 4, 2)
        assert z["inlier_frac"].shape == (n,) and z["mover_frac"].shape == (n,)
        assert z["fallback"].shape == (n,)
        assert z["n_valid_samples"].shape == (n,) and z["n_ransac_inliers"].shape == (n,)
        assert z["fallback_reason_code"].shape == (n,) and z["fit_meta_available"].shape == (n,)
        assert z["n_valid_samples"].dtype.kind in "iu" and z["n_ransac_inliers"].dtype.kind in "iu"
        assert z["fallback_reason_code"].dtype.kind in "iu" and z["fit_meta_available"].dtype.kind == "b"
        assert z["fit_meta_available"].all()                       # observe() records the fit-gate metadata
        assert z["mesh_support_mass"].shape == (n, 4, 5) and np.isfinite(z["mesh_support_mass"]).all()
        assert z["row_support_mass"].shape == (n, 4) and np.isfinite(z["row_support_mass"]).all()
        # stabilize() obtains Obs from observe(), so the cycle maps are available and correctly shaped.
        assert z["cycle_err_rms"].shape == (n, 4, 5) and np.isfinite(z["cycle_err_rms"]).all()
        assert z["cycle_inconsistency_frac"].shape == (n, 4, 5)
        assert np.all((z["cycle_inconsistency_frac"] >= 0) & (z["cycle_inconsistency_frac"] <= 1))
        assert z["cycle_sample_mass"].shape == (n, 4, 5) and np.all(z["cycle_sample_mass"] >= 0)
        assert z["cycle_map_available"].shape == (n,) and z["cycle_map_available"].all()
        assert all(z[k].dtype.kind in "fiuUb" for k in z.files)       # numeric/boolean/unicode only, no objects


def test_stabilize_rejects_ambiguous_diagnostic_scope_before_loading_media():
    with pytest.raises(ValueError, match="not both"):
        S.stabilize("unused", "unused", cfg=S.Config(warp_diag=True), diag_every=2,
                    diag_callback=lambda *args: None)


def test_distortion_guards_invalid_mappings():
    """The legacy global distortion must return None, not raise or report a perfect score, for invalid maps."""
    assert S._distortion(np.eye(3), 8, 8) == pytest.approx(1.0)
    assert S._distortion(np.full((3, 3), np.nan), 8, 8) is None
    assert S._distortion(np.full((3, 3), np.inf), 8, 8) is None
    assert S._distortion(np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]]), 8, 8) is None   # w = 0
    assert S._distortion(np.eye(3), 0, 0) is None
    assert S._distortion(np.eye(3), 1, 1) is None                        # fewer than 3 sample points


# ----------------------------------------------------------------------------------------
# opt-in sampled complete-warp border backoff
# ----------------------------------------------------------------------------------------

class _Fit:
    """Minimal Motion fit carrying only the fields ``plan`` reads."""

    def __init__(self, H=None, mesh=None, rows=None, fallback=False):
        self.H = np.eye(3) if H is None else np.asarray(H, np.float64)
        self.mesh = np.zeros((2, 2, 2)) if mesh is None else np.asarray(mesh, np.float64)
        self.rows = np.zeros((2, 2)) if rows is None else np.asarray(rows, np.float64)
        self.fallback = fallback


def _motion(fits, w=32, h=24, fps=30.0):
    return S.Motion(w, h, fps, fits=list(fits), n_frames=len(fits) + 1)


def _plan(*, G=None, mesh=None, row=None, t=2, w=32, h=24):
    if G is None: G = [np.eye(3) for _ in range(t)]
    if mesh is None: mesh = np.zeros((t, 2, 2, 2))
    if row is None: row = np.zeros((t, 2, 2))
    return S.Plan(G=[np.asarray(g, np.float64) for g in G], mesh_corr=np.asarray(mesh, np.float64),
                  row_corr=np.asarray(row, np.float64), raw_quads=np.zeros((t, 4, 2)),
                  path_quads=np.zeros((t, 4, 2)))


def _shift_plan(tx, ty=0.0, channel="global", t=2, w=32, h=24):
    """A plan whose selected channel is a constant pixel shift, so scale 1 clearly overshoots the frame."""
    if channel == "global":
        G = np.eye(3); G[0, 2] = tx; G[1, 2] = ty
        return _plan(G=[G.copy() for _ in range(t)], t=t, w=w, h=h)
    if channel == "mesh":
        mesh = np.zeros((t, 2, 2, 2)); mesh[:, :, :, 0] = tx; mesh[:, :, :, 1] = ty
        return _plan(mesh=mesh, t=t, w=w, h=h)
    row = np.zeros((t, 2, 2)); row[:, :, 0] = tx; row[:, :, 1] = ty
    return _plan(row=row, t=t, w=w, h=h)


def test_sampled_border_backoff_disabled_parity():
    """With the option off (default/None) the plan is the pre-existing result and no probe runs."""
    mot = _motion([_Fit(H=np.array([[1, 0, 1.5], [0, 1, -0.5], [0, 0, 1]], float)), _Fit()])
    cfg = S.Config(mesh=(1, 1), rs_rows=2, budget=0.1)
    pl = S.plan(mot, cfg)
    assert pl.sampled_border_measured is False and pl.sampled_border_scale == 1.0
    pl2 = S.plan(mot, S.Config(mesh=(1, 1), rs_rows=2, budget=0.1, max_sampled_border_frac=None))
    np.testing.assert_array_equal(np.asarray(pl.G), np.asarray(pl2.G))
    np.testing.assert_array_equal(pl.mesh_corr, pl2.mesh_corr)
    np.testing.assert_array_equal(pl.row_corr, pl2.row_corr)
    # ``sampled_border_backoff(..., None)`` is a pure identity passthrough.
    assert S.sampled_border_backoff(pl, 32, 24, None) is pl


def test_sampled_border_backoff_feasible_scale_one_leaves_arrays_untouched():
    pl = _plan()
    out = S.sampled_border_backoff(pl, 32, 24, 0.5)
    assert out.sampled_border_measured is True
    assert out.sampled_border_scale == 1.0 and out.sampled_border_worst_frac == 0.0
    # The correction arrays are the same objects, not copies: scale 1 is a no-op.
    assert out.G[0] is pl.G[0] and out.mesh_corr is pl.mesh_corr and out.row_corr is pl.row_corr


@pytest.mark.parametrize("channel", ["global", "mesh", "row"])
def test_sampled_border_backoff_clips_constructed_channels(channel):
    pl = _shift_plan(8.0, channel=channel)
    base_worst = S._sampled_worst_border_frac(pl, 32, 24, S._probe_grid(32, 24))
    assert base_worst > 0.05                                        # scale 1 clearly overshoots
    out = S.sampled_border_backoff(pl, 32, 24, 0.01)
    assert 0.0 <= out.sampled_border_scale < 1.0
    assert out.sampled_border_worst_frac <= 0.01                    # every frame stays under the requested cap
    assert out.sampled_border_worst_frac < base_worst


def test_sampled_border_backoff_uses_one_shared_scale_over_all_frames():
    G0 = np.eye(3); G1 = np.eye(3); G1[0, 2] = 8.0                   # frame 0 fine, frame 1 overshoots
    pl = _plan(G=[G0, G1])
    out = S.sampled_border_backoff(pl, 32, 24, 0.01)
    s = out.sampled_border_scale
    assert 0.0 < s < 1.0
    eye = np.eye(3)
    np.testing.assert_allclose(np.asarray(out.G[0]), eye + s * (G0 - eye))
    np.testing.assert_allclose(np.asarray(out.G[1]), eye + s * (G1 - eye))
    assert out.sampled_border_worst_frac <= 0.01


def test_sampled_border_backoff_identity_fallback():
    # A shift large enough that every positive candidate scale still overshoots: only scale 0 is feasible.
    pl = _shift_plan(10000.0, ty=10000.0)
    out = S.sampled_border_backoff(pl, 32, 24, 0.0)
    assert out.sampled_border_scale == 0.0 and out.sampled_border_worst_frac == 0.0
    for g in out.G:
        np.testing.assert_array_equal(g, np.eye(3))
    np.testing.assert_array_equal(out.mesh_corr, np.zeros_like(out.mesh_corr))
    np.testing.assert_array_equal(out.row_corr, np.zeros_like(out.row_corr))


def _mixed_plan(t=3, w=32, h=24):
    """Projective global + nonzero 2x2 mesh + 2-band row corrections over a few frames."""
    Gs = []
    for i in range(t):
        g = np.eye(3)
        g[0, 0] = 1.0 + 0.01 * (i + 1); g[1, 1] = 1.0 - 0.008 * (i + 1)
        g[0, 2] = 1.3 * (i + 1); g[1, 2] = -0.8 * (i + 1)                      # translation
        g[2, 0] = 0.0012 * (i + 1); g[2, 1] = -0.0009 * (i + 1)               # genuine projective row
        Gs.append(g)
    mesh = np.zeros((t, 2, 2, 2))
    mesh[:, 0, 0, 0] = 2.0; mesh[:, 0, 1, 0] = 0.5
    mesh[:, 1, 0, 1] = -1.5; mesh[:, 1, 1, 1] = 0.75
    row = np.zeros((t, 2, 2))
    row[:, 0, 0] = 1.2; row[:, 1, 1] = -0.7
    return _plan(G=Gs, mesh=mesh, row=row, t=t, w=w, h=h)


def _probe_xy(W, H):
    pts = S._probe_grid(W, H)
    return mx.array(pts[:, 0].astype(np.float32))[None], mx.array(pts[:, 1].astype(np.float32))[None]


def test_source_coords_batch_matches_source_coords_at_scale_one():
    """The batched helper must reproduce ``source_coords`` exactly at scale 1 on a mixed projective+mesh+row plan."""
    pl = _mixed_plan()
    W, H = 32, 24
    x, y = _probe_xy(W, H)
    for t in range(len(pl.G)):
        ref = np.array(S.source_coords(pl, t, W, H, (x, y)))
        got = np.array(S.source_coords_batch(pl, t, W, H, [1.0], (x, y)))
        assert got[0].shape == ref[0].shape                      # [1,N,2], same layout as source_coords
        np.testing.assert_allclose(got[0], ref[0], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(np.array(S.valid_mask(got[0], H, W)),
                                      np.array(S.valid_mask(ref[0], H, W)))


def test_source_coords_batch_matches_scaled_source_coords():
    """Batch row ``k`` equals ``source_coords`` of ``_scaled_plan(pl, scales[k])`` for several scales."""
    pl = _mixed_plan()
    W, H = 32, 24
    x, y = _probe_xy(W, H)
    scales = [1.0, 0.5, 0.13, 0.004, 0.0]
    got = np.array(S.source_coords_batch(pl, 1, W, H, scales, (x, y)))
    assert got.shape[0] == len(scales)
    for k, s in enumerate(scales):
        ref = np.array(S.source_coords(S._scaled_plan(pl, s), 1, W, H, (x, y)))
        np.testing.assert_allclose(got[k], ref[0], rtol=0, atol=1e-6)


def test_sampled_worst_border_frac_batch_matches_reference():
    """Batched worst fractions equal the per-scale reference, and pruning only drops already-infeasible scales."""
    pl = _mixed_plan()
    pts = S._probe_grid(32, 24)
    scales = [1.0, 0.5, 0.13, 0.004, 0.0]
    got = S._sampled_worst_border_frac_batch(pl, 32, 24, pts, scales)
    for k, s in enumerate(scales):
        ref = S._sampled_worst_border_frac(S._scaled_plan(pl, s), 32, 24, pts)
        assert got[k] == pytest.approx(ref, abs=1e-12)
    limit = 0.02
    pruned = S._sampled_worst_border_frac_batch(pl, 32, 24, pts, scales, limit=limit)
    keep = got <= limit
    np.testing.assert_array_equal(pruned[keep], got[keep])       # feasible scales accumulated over every frame
    assert np.all(pruned[~keep] > limit)                          # dropped scales stay infeasible


def test_source_coords_ts_scales_mlx_matches_per_frame_and_scalar():
    """The MLX reference batched (frames x scales) chunk reproduces both the per-frame batch and per-frame
    ``source_coords`` on a mixed projective + mesh + row plan, for a chunk that straddles the 8-frame boundary."""
    pl = _mixed_plan(t=17)
    W, H = 32, 24
    x, y = _probe_xy(W, H)
    scales = [1.0, 0.5, 0.13, 0.004, 0.0]
    ts = [0, 3, 7, 8, 15, 16]                                    # crosses the 8-frame chunk boundary
    got = np.array(S._source_coords_ts_scales_mlx(pl, ts, W, H, scales, (x, y)))
    assert got.shape == (len(ts), len(scales), 1, S._probe_grid(W, H).shape[0], 2)
    for f, t in enumerate(ts):
        ref = np.array(S.source_coords_batch(pl, t, W, H, scales, (x, y)))
        np.testing.assert_allclose(got[f], ref, rtol=0, atol=1e-6)
        for k, s in enumerate(scales):
            one = np.array(S.source_coords(S._scaled_plan(pl, s), t, W, H, (x, y)))
            np.testing.assert_allclose(got[f, k], one[0], rtol=0, atol=1e-6)
            np.testing.assert_array_equal(np.array(S.valid_mask(got[f, k], H, W)),
                                          np.array(S.valid_mask(one, H, W)[0]))


def test_bilinear_border_sample_np_matches_warp_sample():
    """The local NumPy bilinear-border sampler reproduces ``warp.sample`` (bilinear, border) for in-range,
    exactly on-tap, sub-pixel and heavily out-of-range coordinates."""
    from mlxstab.warp import sample as warp_sample
    rng = np.random.default_rng(0)
    img = rng.standard_normal((3, 11, 7, 2)).astype(np.float32)
    coords = np.concatenate([rng.uniform(-5.0, 11.0, (3, 5, 6, 1)),
                             rng.uniform(-5.0, 15.0, (3, 5, 6, 1))], -1).astype(np.float32)
    coords[0, 0] = np.array([0.0, 0.0], np.float32)              # exact tap (no interpolation)
    coords[0, 1] = np.array([6.0, 10.0], np.float32)            # last pixel of a W=7, H=11 image
    coords[0, 2] = np.array([-0.5, 10.5], np.float32)           # on the valid-mask boundary
    coords[0, 3] = np.array([2.25, 3.75], np.float32)           # generic sub-pixel interior point
    ref = np.array(warp_sample(mx.array(img), mx.array(coords), "bilinear", "border"))
    got = S._bilinear_border_sample_np(img, coords)
    assert got.shape == ref.shape and got.dtype == np.float32
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-5)


def test_source_coords_ts_scales_np_matches_production_source_coords():
    """The NumPy many-scale probe reproduces production ``source_coords`` (values and valid-mask classes) on a
    mixed projective + mesh + row plan over several frames, scales and a chunk boundary."""
    pl = _mixed_plan(t=17)
    W, H = 32, 24
    x, y = _probe_xy(W, H)
    scales = [1.0, 0.5, 0.13, 0.004, 0.0]
    ts = [0, 3, 7, 8, 15, 16]                                    # crosses the 8-frame chunk boundary
    got = S._source_coords_ts_scales_np(pl, ts, W, H, scales, (np.array(x), np.array(y)))
    assert got.shape == (len(ts), len(scales), 1, S._probe_grid(W, H).shape[0], 2)
    for f, t in enumerate(ts):
        for k, s in enumerate(scales):
            ref_src = S.source_coords(S._scaled_plan(pl, s), t, W, H, (x, y))
            one = np.array(ref_src)
            np.testing.assert_allclose(got[f, k], one[0], rtol=0, atol=1e-4)
            np.testing.assert_array_equal(S._valid_mask_np(got[f, k], H, W),
                                          np.array(S.valid_mask(ref_src, H, W))[0])


def test_source_coords_ts_scales_np_matches_mlx_reference():
    """The NumPy probe agrees (values and valid-mask classes) with the retained MLX reference helper."""
    pl = _mixed_plan(t=17)
    W, H = 32, 24
    x, y = _probe_xy(W, H)
    scales = [1.0, 0.5, 0.13, 0.004, 0.0]
    ts = [0, 3, 7, 8, 15, 16]
    ref = np.array(S._source_coords_ts_scales_mlx(pl, ts, W, H, scales, (x, y)))
    got = S._source_coords_ts_scales_np(pl, ts, W, H, scales, (np.array(x), np.array(y)))
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-4)
    np.testing.assert_array_equal(S._valid_mask_np(got, H, W), np.array(S.valid_mask(mx.array(ref), H, W)))


def test_sampled_worst_border_frac_batch_np_resolves_real_0004_regression():
    """The real 0.004 regression through the NumPy probe: with a 111 px x-shift the right-edge probes at ``x = 31``
    sit at 31.555 at ``s = 0.005`` (out of frame) and at 31.444 at ``s = 0.004`` (valid), so a limit of 0 keeps
    only 0.004 feasible, matching the production ``source_coords``/``valid_mask`` classification."""
    pl = _shift_plan(111.0, channel="global", t=2)
    pts = S._probe_grid(32, 24)
    worst = S._sampled_worst_border_frac_batch(pl, 32, 24, pts, [0.005, 0.004], limit=0.0, frames_per_chunk=1)
    assert worst[0] > 0.0 and worst[1] == 0.0
    x, y = _probe_xy(32, 24)
    for k, s in enumerate([0.005, 0.004]):
        one = S.source_coords(S._scaled_plan(pl, s), 0, 32, 24, (x, y))
        frac = 1.0 - np.array(S.valid_mask(one, 24, 32))[0].mean()
        assert frac == pytest.approx(worst[k], abs=1e-12)


def test_sampled_worst_border_frac_batch_chunking_matches_reference_and_prunes():
    """The chunked batched worst is chunk-length independent, matches the scalar reference, and its pruning
    only removes scales whose running worst already exceeds the cap."""
    pl = _mixed_plan(t=17)                                       # 17 frames: several chunks at 8 frames/chunk
    pts = S._probe_grid(32, 24)
    scales = [1.0, 0.5, 0.13, 0.004, 0.0]
    ref = np.array([S._sampled_worst_border_frac(S._scaled_plan(pl, s), 32, 24, pts) for s in scales])
    for fpc in (1, 2, 8, 100):                                   # 100 >= clip length: one chunk
        got = S._sampled_worst_border_frac_batch(pl, 32, 24, pts, scales, frames_per_chunk=fpc)
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)
    limit = 0.02
    keep = ref <= limit
    assert keep.any() and not keep.all()                         # the scenario both keeps and prunes scales
    for fpc in (1, 2, 8, 100):
        pruned = S._sampled_worst_border_frac_batch(pl, 32, 24, pts, scales, limit=limit, frames_per_chunk=fpc)
        np.testing.assert_allclose(pruned[keep], ref[keep], rtol=0, atol=1e-12)
        assert np.all(pruned[~keep] > limit)                     # dropped scales stay infeasible at any chunk size


def test_sampled_border_backoff_selected_scale_matches_scalar_reference():
    """On a multi-frame plan the selected scale and reported worst equal the per-scale scalar reference.

    A global x-shift of 0.5003 px leaves the right-edge probes at ``x = 31`` out of frame at scale 1
    (31.5003 > 31.5) but inside at scale 0.999 (31.4998 <= 31.5), so the scalar reference resolves exactly the
    same first-feasible candidate without scanning the whole 0.001 grid."""
    pl = _shift_plan(0.5003, channel="global", t=12)            # multi-frame plan (12 frames, one shift)
    W, H = 32, 24
    pts = S._probe_grid(W, H)
    limit = 0.0
    assert S._sampled_worst_border_frac(pl, W, H, pts) > limit   # scale 1 is infeasible on the clip
    assert S._sampled_worst_border_frac(S._scaled_plan(pl, 0.999), W, H, pts) <= limit
    got = S.sampled_border_backoff(pl, W, H, limit)
    ref_scale, ref_worst = 0.0, 0.0
    for s in S._SAMPLED_BORDER_SCALES:                           # existing scalar reference: first feasible scale
        w = S._sampled_worst_border_frac(S._scaled_plan(pl, s), W, H, pts)
        if w <= limit:
            ref_scale, ref_worst = float(s), float(w)
            break
    assert ref_scale == pytest.approx(0.999)
    assert got.sampled_border_scale == pytest.approx(ref_scale)
    assert got.sampled_border_worst_frac == pytest.approx(ref_worst, abs=1e-12)
    assert got.sampled_border_measured is True


def test_sampled_border_backoff_finer_grid_resolves_sub_percent_scale():
    """A 0.001 grid finds the useful 0.004 correction a 0.01 grid silently rounds down to the 0.0 fallback.

    With a 111 px x-shift the right-edge probes sit at ``x + 111*s``: at ``s = 0.005`` they leave the frame
    (31.555 > 31.5) but at ``s = 0.004`` they are still valid (31.444), so a limit of 0 must select 0.004."""
    pl = _shift_plan(111.0, channel="global", t=2)
    out = S.sampled_border_backoff(pl, 32, 24, 0.0)
    assert out.sampled_border_scale == pytest.approx(0.004)
    assert out.sampled_border_worst_frac == 0.0
    # The deterministic grid is at least 0.001-spaced and actually contains the resolved scale.
    grid = S._SAMPLED_BORDER_SCALES
    assert grid[0] == 1.0 and grid[-1] == 0.0 and len(grid) == 1001
    assert 0.004 in grid and 0.005 in grid
    assert max(abs(a - b) for a, b in zip(grid, grid[1:])) < 0.0011
    assert 0.004 not in [round(1.0 - i * 0.01, 4) for i in range(101)]   # the old coarse grid could not land here


@pytest.mark.parametrize("bad", [1.0, -0.01, 1.5, np.nan, np.inf])
def test_max_sampled_border_frac_rejects_invalid_config(bad):
    with pytest.raises(ValueError):
        S.Config(max_sampled_border_frac=bad)


def test_stabilize_report_reports_sampled_border_backoff(monkeypatch):
    """The stabilize report exposes the requested cap, applied scale and measured worst sampled fraction."""
    import json
    from mlxstab import media
    vid, _ = synth.render(T=8, W=96, H=64, jitter=3.0)
    H, W = 64, 96
    monkeypatch.setattr(media, "probe", lambda p: {"width": W, "height": H, "fps": 30.0,
                                                    "field_order": "unknown", "interlaced": False, "duration": 0.0})
    monkeypatch.setattr(media, "decode", lambda *a, **k: (iter(vid), 30.0, (W, H)))
    monkeypatch.setattr(media, "prefetch", lambda gen, n=4: gen)

    class FakeEncoder:
        def __init__(self, *a, **k): pass
        def write(self, frame): pass
        def close(self): pass

    monkeypatch.setattr(media, "Encoder", FakeEncoder)
    cfg = S.Config(mesh=(4, 3), rs_rows=4, budget=0.06, max_sampled_border_frac=0.02)
    rep = S.stabilize("in.mp4", "out.mp4", None, None, cfg, flow_fn=farneback, log=lambda m: None)
    assert rep["max_sampled_border_frac"] == 0.02
    assert rep["sampled_border_scale"] is not None and 0.0 <= rep["sampled_border_scale"] <= 1.0
    assert rep["sampled_border_worst_frac"] is not None and rep["sampled_border_worst_frac"] <= 0.02
    assert "sampled" in rep["sampled_border_note"].lower() and "per-pixel" in rep["sampled_border_note"]
    json.dumps(rep, allow_nan=False)

    rep_off = S.stabilize("in.mp4", "out.mp4", None, None, S.Config(mesh=(4, 3), rs_rows=4, budget=0.06),
                          flow_fn=farneback, log=lambda m: None)
    assert rep_off["sampled_border_scale"] is None and rep_off["sampled_border_worst_frac"] is None
    assert rep_off["sampled_border_measured"] is False


def test_main_wires_max_sampled_border_frac(monkeypatch):
    import mlxstab.__main__ as M
    captured = {}

    def fake_stabilize(*args, **kwargs):
        captured["cfg"] = args[4]
        return {"ok": True}

    monkeypatch.setattr(M, "stabilize", fake_stabilize)
    M.main(["in.mp4", "out.mp4", "--max-sampled-border-frac", "0.03"])
    assert captured["cfg"].max_sampled_border_frac == 0.03
