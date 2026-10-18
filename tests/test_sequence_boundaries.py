"""Focused tests for opt-in shot-boundary (cut) handling.

Cuts are an explicit production boundary, not detected content semantics: the caller supplies internal frame
indices that begin a new shot. The contract under test is:

* no optical-flow pair is measured across a cut;
* the flow backend's temporal state is reset at the cut;
* one identity placeholder fit keeps the one-fit-per-analysed-pair layout;
* planning restarts the global path and the residual accumulation at the cut;
* the placeholder is not counted as a failed fit; and
* the default (no cuts) path is unchanged.
"""
import numpy as np
import mlx.core as mx
import pytest

from mlxstab import media, motion as MO, stabilize as S


def _fit(Hm=None, mesh=None, rows=None, fallback=False, reason=None,
         n_valid_samples=None, n_ransac_inliers=None):
    Hm = np.eye(3) if Hm is None else np.asarray(Hm, np.float64)
    mesh = np.zeros((3, 3, 2)) if mesh is None else np.asarray(mesh, np.float64)
    rows = np.zeros((2, 2)) if rows is None else np.asarray(rows, np.float64)
    return MO.Fit(Hm, mesh, rows, 1.0, 0.0, fallback, fallback_reason=reason,
                  n_valid_samples=n_valid_samples, n_ransac_inliers=n_ransac_inliers)


def _translation(dx, dy):
    Hm = np.eye(3)
    Hm[0, 2] = dx
    Hm[1, 2] = dy
    return Hm


class _FakeFlow:
    """Zero flow with an explicit call log and a resettable temporal state."""

    def __init__(self, H, W):
        self.H, self.W = H, W
        self.calls = []
        self.resets = 0

    def bidir(self, a, b):
        self.calls.append((a.shape, b.shape))
        z = mx.zeros((self.H, self.W, 2), dtype=mx.float32)
        return z, z

    def reset(self):
        self.resets += 1


def test_analyze_skips_pair_across_cut_and_resets_flow_state():
    H, W, T = 32, 48, 8
    frames = [np.zeros((H, W, 3), np.uint8) for _ in range(T)]
    flow = _FakeFlow(H, W)
    cfg = S.Config(step=4, mesh=(4, 3), rs_rows=4)

    mot = S.analyze(iter(frames), flow, cfg, W, H, 30.0, stride=1, threads=False, cuts=[3])

    assert mot.cuts == (3,)
    assert mot.n_frames == T
    assert len(mot.fits) == T - 1                       # one slot per analysed pair
    assert len(flow.calls) == T - 2                     # the 2 -> 3 pair was never measured
    assert flow.resets == 1
    boundary = mot.fits[2]
    assert boundary.fallback and boundary.fallback_reason == MO.FALLBACK_REASON_SEQUENCE_BOUNDARY
    np.testing.assert_array_equal(boundary.H, np.eye(3))
    assert not boundary.mesh.any() and not boundary.rows.any()
    # The immediately following pair is measured normally again (from the new shot's first frame).
    assert not mot.fits[3].fallback


def test_plan_restarts_global_path_and_residuals_at_cut_and_does_not_count_boundary_fallback():
    W, H, T, cut = 48, 32, 12, 6
    fits = []
    for k in range(T - 1):
        if k + 1 == cut:                               # placeholder for the skipped pair 5 -> 6
            fits.append(_fit(fallback=True, reason=MO.FALLBACK_REASON_SEQUENCE_BOUNDARY))
        elif k < cut - 1:
            fits.append(_fit(_translation(4.0 if k % 2 else -4.0, 2.0),
                             mesh=np.full((3, 3, 2), [0.4, -0.2]), rows=np.full((2, 2), [0.3, -0.1])))
        else:
            fits.append(_fit())
    cfg = S.Config(budget=0.02, mesh=(2, 2), rs_rows=2, smooth="l1", fidelity=0.1)
    corners = S.G.corners(W, H)

    cut_plan = S.plan(S.Motion(W, H, 30.0, fits=fits, stride=1, n_frames=T, cuts=(cut,)), cfg)
    continuous = S.plan(S.Motion(W, H, 30.0, fits=fits, stride=1, n_frames=T), cfg)

    assert cut_plan.cuts == (cut,) and continuous.cuts == ()
    assert cut_plan.fallbacks == 0                      # a cut is not a failed motion fit
    np.testing.assert_allclose(cut_plan.raw_quads[cut], corners, atol=1e-12)
    np.testing.assert_allclose(cut_plan.path_quads[cut], corners, atol=1e-9)
    np.testing.assert_allclose(cut_plan.G[cut], np.eye(3), atol=1e-9)
    # Without the boundary the pre-cut camera offset leaks into the new shot.
    assert np.abs(np.asarray(cut_plan.G[cut]) - np.asarray(continuous.G[cut])).max() > 1e-3
    assert not np.allclose(continuous.path_quads[cut], corners, atol=1e-3)
    # Residual accumulation also restarts: the first frame of the new shot has no carried mesh correction.
    assert not cut_plan.mesh_corr[cut].any()


def test_plan_rejects_cut_not_aligned_with_analysis_stride():
    W, H, T = 32, 24, 8
    mot = S.Motion(W, H, 30.0, fits=[_fit() for _ in range(T // 2)],
                   stride=2, n_frames=T, cuts=(3,))
    with pytest.raises(ValueError, match="analysis_stride"):
        S.plan(mot, S.Config(mesh=(2, 2), rs_rows=2))


@pytest.mark.parametrize("cuts", [[-1], [1.5], [True], [8], "3"])
def test_analyze_rejects_invalid_cut_indices(cuts):
    H, W, T = 16, 24, 8
    frames = [np.zeros((H, W, 3), np.uint8) for _ in range(T)]
    with pytest.raises(ValueError):
        S.analyze(iter(frames), _FakeFlow(H, W), S.Config(step=8, mesh=(2, 2), rs_rows=2),
                  W, H, 30.0, stride=1, threads=False, cuts=cuts)


def test_analyze_rejects_cut_beyond_decoded_clip():
    H, W, T = 16, 24, 8
    frames = [np.zeros((H, W, 3), np.uint8) for _ in range(T)]
    with pytest.raises(ValueError, match="inside the clip"):
        S.analyze(iter(frames), _FakeFlow(H, W), S.Config(step=8, mesh=(2, 2), rs_rows=2),
                  W, H, 30.0, stride=1, threads=False, cuts=[T])


def test_cut_at_first_frame_is_canonicalized_away():
    H, W, T = 16, 24, 4
    frames = [np.zeros((H, W, 3), np.uint8) for _ in range(T)]
    flow = _FakeFlow(H, W)
    mot = S.analyze(iter(frames), flow, S.Config(step=8, mesh=(2, 2), rs_rows=2),
                    W, H, 30.0, stride=1, threads=False, cuts=[0])
    assert mot.cuts == ()
    assert flow.resets == 0 and len(flow.calls) == T - 1


def test_cli_wires_explicit_cuts_and_analysis_stride(monkeypatch, capsys):
    import mlxstab.__main__ as M
    captured = {}

    def fake_stabilize(*args, **kwargs):
        captured["cfg"] = args[4]
        captured["cuts"] = kwargs["cuts"]
        return {"ok": True}

    monkeypatch.setattr(M, "stabilize", fake_stabilize)
    M.main(["in.mp4", "out.mp4", "--cut-frames", "6,3,6", "--analysis-stride", "3"])
    assert captured["cfg"].analysis_stride == 3
    assert captured["cuts"] == (3, 6)
    assert '"ok": true' in capsys.readouterr().out


@pytest.mark.parametrize("stride", ["0", "-1", "1.5", "bad"])
def test_cli_rejects_invalid_analysis_stride(stride):
    import mlxstab.__main__ as M
    with pytest.raises(SystemExit):
        M.main(["in.mp4", "out.mp4", "--analysis-stride", stride])


def test_motion_and_plan_payloads_preserve_cut_frames(tmp_path):
    from bench import plan_shape_audit as PSA
    from bench import real_composition_audit as RCA

    W, H, T, cut = 32, 24, 8, 4
    fits = [_fit() for _ in range(T - 1)]
    fits[cut - 1] = _fit(fallback=True, reason=MO.FALLBACK_REASON_SEQUENCE_BOUNDARY,
                         n_valid_samples=0, n_ransac_inliers=0)
    mot = S.Motion(W, H, 30.0, fits=fits, stride=1, n_frames=T, cuts=(cut,))
    cfg = S.Config(mesh=(2, 2), rs_rows=2)
    motion_path = tmp_path / "motion.npz"
    plan_path = tmp_path / "plan.npz"
    S.save_motion_npz(mot, motion_path, cfg)
    S.save_plan(S.plan(mot, cfg), plan_path, W, H, 30.0)

    loaded_motion, motion_meta = RCA.load_motion_npz(motion_path)
    assert loaded_motion.cuts == (cut,) and motion_meta["cut_frames"] == [cut]
    assert motion_meta["sequence_boundaries"] == 1 and motion_meta["fallback_fits"] == 0
    loaded_plan, plan_meta = PSA.load_plan_npz(plan_path)
    assert loaded_plan.cuts == (cut,) and plan_meta["cut_frames"] == [cut]


def test_stabilize_threads_cuts_into_analysis_plan_and_report(monkeypatch, tmp_path):
    H, W, T = 32, 48, 8
    vid = [np.zeros((H, W, 3), np.uint8) for _ in range(T)]
    monkeypatch.setattr(media, "probe", lambda p: {"width": W, "height": H, "fps": 30.0,
                                                    "field_order": "unknown", "interlaced": False,
                                                    "duration": 0.0})
    monkeypatch.setattr(media, "decode", lambda *a, **k: (iter(vid), 30.0, (W, H)))
    monkeypatch.setattr(media, "prefetch", lambda gen, n=4: gen)

    class FakeEncoder:
        def __init__(self, *a, **k): pass
        def write(self, frame): pass
        def close(self): pass

    monkeypatch.setattr(media, "Encoder", FakeEncoder)
    flow = _FakeFlow(H, W)
    cfg = S.Config(step=4, mesh=(4, 3), rs_rows=4, budget=0.04)
    plan_path = tmp_path / "plan.npz"
    rep = S.stabilize("in.mp4", "out.mp4", None, None, cfg, flow_fn=flow, log=lambda m: None,
                      dump_plan=str(plan_path), cuts=[4])

    assert flow.resets == 1 and len(flow.calls) == T - 2
    assert rep["cut_frames"] == [4] and rep["sequence_boundaries"] == 1
    assert rep["fallback_frames"] == 0
    with np.load(plan_path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["cut_frames"], [4])
