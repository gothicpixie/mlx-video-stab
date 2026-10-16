"""Opt-in isotropic output-space crop zoom (``Config.crop_zoom``).

``crop_zoom > 1`` composes ``Z`` on the right of the global output-to-source map (``G_new = G @ Z``),
where ``Z`` maps ``x -> c + (x - c)/crop_zoom`` and ``c = (W/2 - .5, H/2 - .5)``. The mesh/row residual
arrays are deliberately left unchanged so production re-evaluates them at the new G-mapped source points.
"""

import numpy as np
import pytest
import mlx.core as mx

from mlxstab import stabilize as S
from mlxstab.warp import valid_mask


class _Fit:
    """Minimal Motion fit carrying only the fields ``plan`` reads."""

    def __init__(self, H=None, mesh=None, rows=None, fallback=False):
        self.H = np.eye(3) if H is None else np.asarray(H, np.float64)
        self.mesh = np.zeros((2, 2, 2)) if mesh is None else np.asarray(mesh, np.float64)
        self.rows = np.zeros((2, 2)) if rows is None else np.asarray(rows, np.float64)
        self.fallback = fallback


def _motion(fits, w=32, h=24, fps=30.0):
    return S.Motion(w, h, fps, fits=list(fits), n_frames=len(fits) + 1)


def _translation_motion(tx, ty, t=6, w=32, h=24):
    H = np.eye(3); H[0, 2] = tx; H[1, 2] = ty
    return _motion([_Fit(H=H.copy()) for _ in range(t - 1)], w=w, h=h)


def _grid(W, H):
    ys, xs = mx.meshgrid(mx.arange(H), mx.arange(W), indexing="ij")
    return xs.astype(mx.float32), ys.astype(mx.float32)


def _cfg(**kw):
    return S.Config(mesh=(0, 0), rs_rows=0, budget=0.1, **kw)


def test_crop_zoom_config_default_and_validation():
    assert S.Config().crop_zoom == 1.0                      # opt-in: default preserves the existing warp exactly
    for ok in (1.0, 1.0000001, 1.5, 3.0):
        assert S.Config(crop_zoom=ok).crop_zoom == pytest.approx(ok)
    for bad in (0.5, 0.0, -1.0, 0.999, np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            S.Config(crop_zoom=bad)


def test_crop_zoom_default_and_explicit_identity_are_exact():
    """``crop_zoom == 1.0`` must be a bit-exact no-op on the plan and the composed source field."""
    mot = _translation_motion(3.0, -2.0)
    W, H = mot.W, mot.H
    default = S.plan(mot, _cfg())
    explicit = S.plan(mot, _cfg(crop_zoom=1.0))
    np.testing.assert_array_equal(np.asarray(default.G), np.asarray(explicit.G))
    np.testing.assert_array_equal(default.mesh_corr, explicit.mesh_corr)
    np.testing.assert_array_equal(default.row_corr, explicit.row_corr)
    grid = _grid(W, H)
    for t in range(len(default.G)):
        np.testing.assert_array_equal(np.array(S.source_coords(default, t, W, H, grid)),
                                      np.array(S.source_coords(explicit, t, W, H, grid)))


def test_crop_zoom_is_composed_after_the_global_map():
    """``G_new == G @ Z`` with ``Z`` mapping output ``x -> c + (x - c)/crop_zoom``."""
    mot = _translation_motion(3.0, -2.0)
    W, H = mot.W, mot.H
    base = S.plan(mot, _cfg())
    zoom = S.plan(mot, _cfg(crop_zoom=2.5))
    Z = S._crop_zoom_matrix(W, H, 2.5)
    cx, cy = W / 2 - .5, H / 2 - .5
    for t in range(len(base.G)):
        np.testing.assert_allclose(np.asarray(zoom.G[t]), np.asarray(base.G[t]) @ Z, rtol=0, atol=1e-12)
        # the isotropic center zoom is on the output side: it scales the linear part and shifts the translation
        np.testing.assert_allclose(np.asarray(zoom.G[t])[:2, :2], np.asarray(base.G[t])[:2, :2] / 2.5, atol=1e-12)
        np.testing.assert_allclose(float(zoom.G[t][0, 2]),
                                   float(base.G[t][0, 2]) + cx * (1 - 1 / 2.5), atol=1e-9)
        np.testing.assert_allclose(float(zoom.G[t][1, 2]),
                                   float(base.G[t][1, 2]) + cy * (1 - 1 / 2.5), atol=1e-9)


def test_crop_zoom_pure_translation_preserves_translation_and_moves_edges_inward():
    """A pure translation keeps its translation term and its edge samples move toward the frame center."""
    mot = _translation_motion(3.0, -2.0)
    W, H, s = mot.W, mot.H, 2.0
    base = S.plan(mot, _cfg())
    zoom = S.plan(mot, _cfg(crop_zoom=s))
    cx, cy = W / 2 - .5, H / 2 - .5
    tx, ty = float(base.G[0][0, 2]), float(base.G[0][1, 2])
    np.testing.assert_allclose(np.asarray(base.G[0])[:2, :2], np.eye(2), atol=1e-12)   # the base map is a pure translation
    grid = _grid(W, H)
    src = np.array(S.source_coords(zoom, 0, W, H, grid)[0])
    x, y = np.array(grid[0]), np.array(grid[1])
    # source = zoomed output coordinate + the original translation term (the term is preserved, not rescaled)
    np.testing.assert_allclose(src[..., 0], (x - cx) / s + cx + tx, rtol=0, atol=1e-4)
    np.testing.assert_allclose(src[..., 1], (y - cy) / s + cy + ty, rtol=0, atol=1e-4)
    base_src = np.array(S.source_coords(base, 0, W, H, grid)[0])
    for (px, py) in ((0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)):
        assert abs(src[py, px, 0] - cx) <= abs(base_src[py, px, 0] - cx) + 1e-9   # x close to the center
        assert abs(src[py, px, 1] - cy) <= abs(base_src[py, px, 1] - cy) + 1e-9


def test_crop_zoom_does_not_mutate_inputs_or_residual_arrays():
    """The zoom rewrites only ``G``: fitted motion and residual arrays are byte-identical across plans."""
    mot = _translation_motion(3.0, -2.0, w=32, h=24)
    before_H = [np.array(f.H) for f in mot.fits]
    before_mesh = [np.array(f.mesh) for f in mot.fits]
    before_rows = [np.array(f.rows) for f in mot.fits]
    base = S.plan(mot, _cfg())
    zoom = S.plan(mot, _cfg(crop_zoom=2.0))
    # the shared Motion inputs were not mutated by either plan
    for f, h0, m0, r0 in zip(mot.fits, before_H, before_mesh, before_rows):
        np.testing.assert_array_equal(np.asarray(f.H), h0)
        np.testing.assert_array_equal(np.asarray(f.mesh), m0)
        np.testing.assert_array_equal(np.asarray(f.rows), r0)
    # residual arrays are unchanged by the crop zoom (production re-evaluates them at the new G-mapped points)
    np.testing.assert_array_equal(np.asarray(zoom.mesh_corr), np.asarray(base.mesh_corr))
    np.testing.assert_array_equal(np.asarray(zoom.row_corr), np.asarray(base.row_corr))
    # G entries are fresh arrays, not aliases of the default plan
    for t in range(len(base.G)):
        assert not np.shares_memory(np.asarray(zoom.G[t]), np.asarray(base.G[t]))
        assert not np.array_equal(np.asarray(zoom.G[t]), np.asarray(base.G[t]))


def test_crop_zoom_validity_mask_is_recomputed_from_changed_source_coords():
    """The rendered border mask must come from the crop-zoomed source field, not a stale default one."""
    mot = _translation_motion(18.0, 0.0, w=32, h=24)          # large shift: the default plan clips the right edge
    W, H = mot.W, mot.H
    base = S.plan(mot, _cfg())
    zoom = S.plan(mot, _cfg(crop_zoom=2.0))
    grid = _grid(W, H)
    frame = np.zeros((H, W, 3), np.uint8)
    _, bad_base, _ = S.render_frame(frame, base, 0, _cfg(), grid)
    _, bad_zoom, _ = S.render_frame(frame, zoom, 0, _cfg(crop_zoom=2.0), grid)
    expected = ~np.array(valid_mask(S.source_coords(zoom, 0, W, H, grid), H, W))[0]
    np.testing.assert_array_equal(bad_zoom, expected)         # recomputed from the changed source_coords
    assert bad_zoom.mean() != bad_base.mean()                 # the crop-zoomed mask is not the default mask


def test_stabilize_report_reports_crop_zoom(monkeypatch):
    """The stabilize report exposes the configured crop zoom alongside the stabilization config."""
    import json
    from mlxstab import media, synth

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
    rep = S.stabilize("in.mp4", "out.mp4", None, None, S.Config(mesh=(4, 3), rs_rows=4, crop_zoom=1.7),
                      flow_fn=lambda a, b, key=None: mx.zeros((a.shape[0], a.shape[1], 2), mx.float32),
                      log=lambda m: None)
    assert rep["crop_zoom"] == pytest.approx(1.7)
    json.dumps(rep, allow_nan=False)
    rep_off = S.stabilize("in.mp4", "out.mp4", None, None, S.Config(mesh=(4, 3), rs_rows=4),
                          flow_fn=lambda a, b, key=None: mx.zeros((a.shape[0], a.shape[1], 2), mx.float32),
                          log=lambda m: None)
    assert rep_off["crop_zoom"] == 1.0


def test_main_wires_crop_zoom(monkeypatch):
    import mlxstab.__main__ as M
    captured = {}

    def fake_stabilize(*args, **kwargs):
        captured["cfg"] = args[4]
        return {"ok": True}

    monkeypatch.setattr(M, "stabilize", fake_stabilize)
    M.main(["in.mp4", "out.mp4", "--crop-zoom", "1.5"])
    assert captured["cfg"].crop_zoom == 1.5
