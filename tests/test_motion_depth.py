"""Focused tests for opt-in relative-depth weighting of the local residual mesh.

Depth is a per-frame *relative* signal, so the weighting uses similarity in a robustly normalized depth span.
It affects only the local mesh estimate, its under-observed fill and the separable mesh smoothing in
``fit_motion``. The global H, RANSAC/mover masks, row field, plan and renderer are untouched. The default
``Config.mesh_depth_strength == 0`` and ``fit_motion(depth_strength=0)`` keep the historical result byte for
byte, and the observation-coverage support mass remains the raw pre-fill sample count.

No media, FFmpeg, RAFT or changed production defaults.
"""
import numpy as np
import mlx.core as mx

from mlxstab import motion as MO
from mlxstab import stabilize as S


def _grid_obs(W=96, H=64, step=4):
    xs = np.arange(step // 2, W, step)
    ys = np.arange(step // 2, H, step)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64), W, H


def _two_layer_obs(W=96, H=64, step=4):
    """Piecewise horizontal residual: +1 px left of the vertical boundary, -1 px right of it."""
    pts, W, H = _grid_obs(W, H, step)
    flow = np.stack([np.where(pts[:, 0] < W / 2, 1.0, -1.0), np.zeros(len(pts))], 1)
    return MO.Obs(pts, flow, np.ones(len(pts), bool), W, H)


def _two_layer_depth(W=96, H=64):
    return (np.arange(W)[None, :] >= W / 2).astype(np.float64) * np.ones((H, 1))


def _assert_same_fit(a, b):
    for name in ("H", "mesh", "rows", "mesh_support_mass", "row_support_mass",
                 "cycle_err_rms", "cycle_inconsistency_frac", "cycle_sample_mass"):
        left, right = getattr(a, name), getattr(b, name)
        if left is None or right is None:
            assert left is None and right is None
        else:
            assert np.array_equal(left, right), name
            assert left.tobytes() == right.tobytes(), name


# --------------------------------------------------------------------------------------
# validation and the depth-off parity contract
# --------------------------------------------------------------------------------------

def test_depth_strength_zero_is_byte_for_byte_parity_with_default():
    obs = _two_layer_obs()
    depth = _two_layer_depth()
    default = MO.fit_motion(obs, mesh=(8, 4), rs_rows=4, rng=np.random.default_rng(3))
    explicit = MO.fit_motion(obs, mesh=(8, 4), rs_rows=4, rng=np.random.default_rng(3),
                             depth=depth, depth_strength=0.0)
    _assert_same_fit(default, explicit)
    assert default.fallback is False


def test_config_mesh_depth_strength_defaults_off_and_validates():
    assert S.Config().mesh_depth_strength == 0.0
    assert isinstance(S.Config().mesh_depth_strength, float)
    for value in (0, 0.0, 1, 2.5, np.float32(0.25), np.float64(3.0)):
        assert S.Config(mesh_depth_strength=value).mesh_depth_strength == float(value)
    for bad in (-1.0, -1e-9, np.nan, np.inf, -np.inf, True, False, np.bool_(True),
                "1", b"1", [1.0], np.array(1.0), 1 + 2j, None):
        try:
            S.Config(mesh_depth_strength=bad)
        except ValueError as e:
            assert "mesh_depth_strength" in str(e)
        else:
            raise AssertionError(f"mesh_depth_strength accepted {bad!r}")


def test_fit_motion_validates_depth_strength_and_requires_depth():
    obs = _two_layer_obs(W=64, H=48)
    for bad in (-1.0, np.nan, np.inf, True, "1", [1.0]):
        try:
            MO.fit_motion(obs, mesh=(4, 3), rs_rows=2, depth=np.ones((48, 64)), depth_strength=bad)
        except ValueError as e:
            assert "depth strength" in str(e)
        else:
            raise AssertionError(f"depth_strength accepted {bad!r}")
    try:
        MO.fit_motion(obs, mesh=(4, 3), rs_rows=2, depth_strength=1.0)
    except ValueError as e:
        assert "requires a source relative-depth map" in str(e)
    else:
        raise AssertionError("positive depth_strength without depth was accepted")
    try:
        MO.fit_motion(obs, mesh=(4, 3), rs_rows=2, depth=np.ones((3, 4)), depth_strength=1.0)
    except ValueError as e:
        assert "shape" in str(e)
    else:
        raise AssertionError("wrong-shaped depth was accepted")


# --------------------------------------------------------------------------------------
# a strong depth boundary is not averaged by the local estimate, fill or smoothing
# --------------------------------------------------------------------------------------

def test_two_layer_residual_is_not_averaged_at_the_depth_boundary():
    obs = _two_layer_obs()
    depth = _two_layer_depth()
    baseline = MO.fit_motion(obs, mesh=(8, 4), rs_rows=0, thr=1.5)
    depth_fit = MO.fit_motion(obs, mesh=(8, 4), rs_rows=0, thr=1.5, depth=depth, depth_strength=1.0)
    assert baseline.fallback is False and depth_fit.fallback is False

    # Without depth the boundary vertex is pulled toward the two-layer average; with depth it remains on its
    # own layer instead. Strong boundaries separate the columns on either side.
    assert np.abs(baseline.mesh[..., 0]).min() < 0.1
    assert np.abs(depth_fit.mesh[..., 0]).min() > 0.1
    assert depth_fit.mesh[2, 4, 0] < -0.2
    assert depth_fit.mesh[2, 3, 0] > 0.2

    # Support mass is still the raw, non-robust pre-fill observation-coverage count.
    assert np.array_equal(depth_fit.mesh_support_mass, baseline.mesh_support_mass)
    assert np.array_equal(depth_fit.row_support_mass, baseline.row_support_mass)


def test_depth_aware_fill_does_not_diffuse_across_a_strong_boundary():
    """An unsupported vertex next to two layers takes its own layer's value, not their average."""
    field = np.array([[[1.0, 0.0], [9.0, 9.0], [-1.0, 0.0]]])   # [1,3,2]; centre is unsupported
    mass = np.array([[4.0, 0.0, 4.0]])
    depth = np.array([[0.0, 1.0, 1.0]])                           # strong boundary between columns 0 and 1
    old = MO._fill(field, mass, MO.SUPPORT_THRESHOLD)
    new = MO._fill(field, mass, MO.SUPPORT_THRESHOLD, depth=depth, depth_strength=1.0)
    np.testing.assert_allclose(old[0, 1], [0.0, 0.0])             # historical fill averages across the layers
    np.testing.assert_allclose(new[0, 1], [-1.0, 0.0])            # depth-aware fill stays on the centre layer


# --------------------------------------------------------------------------------------
# analyze/stabilize callback threading and downstream plan/render effect
# --------------------------------------------------------------------------------------

class _FakeFlow:
    """Constant bidirectional flow for one pair; the forward/backward pair is cycle-consistent."""
    def __init__(self, flow):
        self.forward = mx.array(np.asarray(flow, np.float32))
        self.backward = -self.forward

    def bidir(self, a, b):
        return self.forward, self.backward


def test_analyze_calls_depth_fn_with_the_source_frame_index():
    W, H, step = 96, 64, 8
    frames = [np.zeros((H, W, 3), np.uint8) for _ in range(5)]
    depth = np.ones((H, W), np.float64)
    seen = []

    def depth_fn(i):
        seen.append(i)
        return depth

    cfg = S.Config(mesh=(4, 3), rs_rows=2, step=step, mesh_depth_strength=1.0)
    mot = S.analyze(iter(frames), _FakeFlow(np.zeros((H, W, 2))), cfg, W, H, 30.0,
                    threads=False, stride=2, depth_fn=depth_fn)
    assert seen == [0, 2]                       # source frames of pairs (0->2) and (2->4), never destinations
    assert len(mot.fits) == 2 and not any(f.fallback for f in mot.fits)

    # With the default strength the callback is not consulted at all.
    seen.clear()
    S.analyze(iter(frames), _FakeFlow(np.zeros((H, W, 2))), S.Config(mesh=(4, 3), rs_rows=2, step=step),
              W, H, 30.0, threads=False, stride=2, depth_fn=depth_fn)
    assert seen == []


def test_depth_changes_the_planned_mesh_and_the_rendered_frame():
    W, H, step, T = 96, 64, 8, 6
    base_flow = np.zeros((H, W, 2), np.float32)
    base_flow[..., 0] = np.where(np.arange(W)[None, :] < W / 2, 1.0, -1.0)
    amplitudes = (1.0, 2.0, 0.5, 1.5, 0.75, 1.25)
    depth = _two_layer_depth(W, H)
    frames = [(np.add.outer(np.arange(H), np.arange(W))[..., None] * np.array([1, 2, 3])) % 251
              for _ in range(T)]
    frames = [f.astype(np.uint8) for f in frames]

    class VaryingFlow(_FakeFlow):
        def __init__(self):
            self.k = 0

        def bidir(self, a, b):
            f = mx.array(base_flow * np.float32(amplitudes[self.k % len(amplitudes)]))
            self.k += 1
            return f, -f

    def run(cfg, depth_fn=None):
        mot = S.analyze(iter(frames), VaryingFlow(), cfg, W, H, 30.0, threads=False, depth_fn=depth_fn)
        pl = S.plan(mot, cfg)
        return pl, [S.render_frame(frames[t], pl, t, cfg)[0] for t in range(len(pl.G))]

    off_cfg = S.Config(mesh=(8, 4), rs_rows=0, step=step)
    on_cfg = S.Config(mesh=(8, 4), rs_rows=0, step=step, mesh_depth_strength=1.0)
    off_plan, off_frames = run(off_cfg)
    on_plan, on_frames = run(on_cfg, lambda i: depth)

    assert all(np.array_equal(a, b) for a, b in zip(off_plan.G, on_plan.G))    # global path unchanged
    assert not np.array_equal(off_plan.mesh_corr, on_plan.mesh_corr)           # local mesh changed
    assert not all(np.array_equal(a, b) for a, b in zip(off_frames, on_frames))  # render consumes that mesh


def test_stabilize_forwards_depth_fn_to_analyze(monkeypatch):
    """The public stabilize path hands the optional callback to analysis without changing its own outputs."""
    from mlxstab import media
    H, W, T = 32, 48, 3
    frames = [np.zeros((H, W, 3), np.uint8) for _ in range(T)]
    monkeypatch.setattr(media, "probe", lambda p: {"width": W, "height": H, "fps": 30.0,
                                                    "field_order": "unknown", "interlaced": False, "duration": 0.0})
    monkeypatch.setattr(media, "decode", lambda *a, **k: (iter(frames), 30.0, (W, H)))
    monkeypatch.setattr(media, "prefetch", lambda gen, n=4: gen)

    class FakeEncoder:
        def __init__(self, *a, **k): pass
        def write(self, frame): pass
        def close(self): pass

    monkeypatch.setattr(media, "Encoder", FakeEncoder)
    captured = {}

    def fake_analyze(*args, **kwargs):
        captured["depth_fn"] = kwargs.get("depth_fn")
        return S.Motion(W, H, 30.0, fits=[], stride=1, n_frames=T)

    monkeypatch.setattr(S, "analyze", fake_analyze)
    depth_fn = lambda i: np.ones((H, W))
    S.stabilize("in.mp4", "out.mp4", cfg=S.Config(mesh=(4, 3), rs_rows=2, mesh_depth_strength=1.0),
                flow_fn=_FakeFlow(np.zeros((H, W, 2))), log=lambda m: None, depth_fn=depth_fn)
    assert captured["depth_fn"] is depth_fn
