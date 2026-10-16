"""Focused tests for the research-only rendered composition candidate
(``bench/real_composition_render.py``).

All media and render dependencies are faked (no weights, no real footage, no MLX compute): a tiny captured
plan/motion pair is built with the ``bench.composition_audit`` analytic fixtures, and ``media.probe``/
``media.decode``/``media.Encoder``/``media.frame_pts`` plus ``real_composition_render.render_candidate_frame``
are monkeypatched. Covered: the 600-source -> 300-output skip mapping, the chosen mode and its precomputed
smoothing chart, frame render pairing, manifest/provenance/output hashes, early/extra decode, malformed
timeline and unsupported cadence, non-finite field/render and encode/CFR failures with cleanup, output
collision/overwrite refusal, and the CLI.
"""
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from mlxstab import media
from mlxstab import stabilize as S
from mlxstab.motion import Fit

from bench import composition_audit as CA
from bench import manifest as MF
from bench import real_composition_render as R

W, H = 16, 12
GC, GR = 4, 3
SOURCE_FPS = 60.0
OUT_FPS = 30.0

CONFIG_FIELDS = ("smooth", "sigma", "l1_weights", "budget", "mesh", "mesh_budget", "rs_rows", "flow_iters",
                 "flow_downscale", "warm_start", "step", "kernel", "zoom", "seed", "analysis_stride",
                 "flow_dtype", "corr_dtype", "warm_iters", "model", "fidelity", "warp_diag")

RENDER_CALLS = []


# --------------------------------------------------------------------------------------
# tiny analytic fixture (identity pairs -> identity candidate field)
# --------------------------------------------------------------------------------------

def _config_dict(cfg):
    d = {k: getattr(cfg, k) for k in CONFIG_FIELDS}
    d["l1_weights"] = list(d["l1_weights"])
    d["mesh"] = list(d["mesh"])
    return d


def _fits(pairs):
    return [Fit(np.asarray(p.H, np.float64).copy(), np.asarray(p.mesh, np.float64).copy(),
                np.zeros((1, 2)), 0.95, 0.01, False) for p in pairs]


def _identity_pairs(T):
    return [CA.PairMotion(np.eye(3), np.zeros((GR + 1, GC + 1, 2))) for _ in range(T - 1)]


def _pts(count, fps):
    return {"count": count, "n_pts": count, "monotonic": True,
            "pts_s": [i / float(fps) for i in range(count)]}


def save_run(root, *, name="run", source_frames=6, output_frames=3, fps_source=SOURCE_FPS,
             fps_out=OUT_FPS):
    run = Path(root) / name
    run.mkdir(parents=True, exist_ok=True)
    T = int(source_frames)
    pairs = _identity_pairs(T)
    mot = S.Motion(W, H, float(fps_source), fits=_fits(pairs), stride=1, n_frames=T)
    fx = CA.Fixture("fx", "fixture", W, H, T, pairs, (GC, GR), rs_rows=0, fps=float(fps_source))
    plan, cfg = CA.build_plan(fx, pin=False)
    plan_path, motion_path = run / "plan_r0.npz", run / "motion_r0.npz"
    S.save_plan(plan, str(plan_path), W, H, float(fps_source))
    S.save_motion_npz(mot, str(motion_path), cfg)

    src = run / "source.mts"
    src.write_bytes(b"source-bytes")
    baseline = run / "stabilized.mp4"
    baseline.write_bytes(b"fake-mp4:%d" % int(output_frames))
    manifest = {
        "schema": MF.MANIFEST_SCHEMA,
        "status": "success",
        "run_id": name,
        "case": {"name": "baseline"},
        "source": {
            "path": str(src), "path_type": "file", "sha256": MF.sha256_file(src),
            "window": {"start_s": 60.0, "duration_s": 10.0},
            "preprocessing": {"interlaced": True, "field_order": "tb",
                              "deinterlace": R.DEINTERLACE_FILTER, "processing_fps": float(fps_source),
                              "decode_vf": None},
            "decoded": {"frames": T, "fps": float(fps_source), "width": W, "height": H,
                        "duration_s": T / float(fps_source)},
        },
        "weights": {"sha256": "w" * 64},
        "config": _config_dict(cfg),
        "outputs": {
            "video": str(baseline), "plan": str(plan_path), "motion": str(motion_path),
            "decoded": {"frames": int(output_frames), "fps": float(fps_out), "width": W, "height": H,
                        "duration_s": int(output_frames) / float(fps_out)},
            "encoded_frames": int(output_frames),
            "encoded_pts": _pts(int(output_frames), fps_out),
            "cadence": {"measured_fps": float(fps_out), "measured_frames": int(output_frames)},
            "timeline_alignment": {"aligned": True},
        },
        "experiment_identity": {"capture": {"dump_plan": True, "dump_motion": True}},
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run, manifest


class FakeLock:
    """Records the ``HeavyLock`` lifecycle without touching the filesystem or the anniversary repo.

    Unit tests inject this through ``render_candidate(..., lock=fake)`` or by patching ``R.HeavyLock`` for the
    CLI, so no real shared lock is ever acquired.
    """

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.acquires = 0
        self.releases = 0
        self.steps = []
        self.held = False
        FakeLock.instances.append(self)

    def acquire(self):
        self.acquires += 1
        self.held = True

    def release(self):
        self.releases += 1
        self.held = False

    def on_step(self, step):
        self.steps.append(int(step))

    def status(self):
        return {"backend": "fake", "path": "/private/anniversary/media/.heavy.lock",
                "chunk": self.kwargs.get("chunk", 0), "pause_s": self.kwargs.get("pause_s", 0.0),
                "held": self.held, "releases": self.releases,
                "acquire_wait_s": 0.0, "pause_total_s": 0.0}


class FakeEncoder:
    instances = []

    def __init__(self, path, width, height, fps, **kwargs):
        self.path, self.width, self.height, self.fps = str(path), width, height, fps
        self.kwargs = kwargs
        self.frames = []
        FakeEncoder.instances.append(self)

    def write(self, frame):
        self.frames.append(np.array(frame, copy=True))

    def close(self):
        Path(self.path).write_bytes(b"fake-mp4:%d" % len(self.frames))


def _setup(monkeypatch, tmp_path, *, source_frames=6, output_frames=3, fps_source=SOURCE_FPS,
           fps_out=OUT_FPS, decode_extra=0):
    run, manifest = save_run(tmp_path, source_frames=source_frames, output_frames=output_frames,
                             fps_source=fps_source, fps_out=fps_out)
    FakeEncoder.instances = []
    RENDER_CALLS.clear()

    def fake_probe(path):
        p = Path(path)
        if p.name == "source.mts":
            return {"width": W, "height": H, "fps": float(fps_source), "interlaced": True,
                    "field_order": "tb"}
        return {"width": W, "height": H, "fps": float(fps_out), "interlaced": False}

    def fake_decode(path, info=None, start=None, duration=None, deinterlace=True, vf=None):
        if Path(path).name == "source.mts":
            n = int(source_frames) + int(decode_extra)
            return iter([np.full((H, W, 3), (i * 30) % 256, np.uint8) for i in range(n)]), \
                float(fps_source), (W, H)
        return iter([np.full((H, W, 3), 200, np.uint8) for _ in range(int(output_frames))]), \
            float(fps_out), (W, H)

    def fake_frame_pts(path):
        p = Path(path)
        if p.name == "stabilized.mp4":
            return _pts(int(output_frames), fps_out)
        data = p.read_bytes()
        count = int(data.split(b":")[1])
        return _pts(count, fps_out)

    def fake_render(frame, field, cfg):
        field = np.asarray(field)
        RENDER_CALLS.append({"frame": np.array(frame, copy=True), "field_shape": field.shape,
                             "finite": bool(np.isfinite(field).all())})
        return np.array(frame, copy=True), np.zeros(frame.shape[:2], bool)

    monkeypatch.setattr(media, "probe", fake_probe)
    monkeypatch.setattr(media, "decode", fake_decode)
    monkeypatch.setattr(media, "Encoder", FakeEncoder)
    monkeypatch.setattr(media, "frame_pts", fake_frame_pts)
    monkeypatch.setattr(R, "render_candidate_frame", fake_render)
    return run, manifest


def _out(tmp_path, name="candidate.mp4"):
    return tmp_path / name, tmp_path / "candidate.json"


# --------------------------------------------------------------------------------------
# dense forward-propagated map: small smooth synthetic motion
# --------------------------------------------------------------------------------------

# Justified tolerance for the dense-grid bilinear interpolation approximation of the exact composition on the
# small, smooth synthetic motion below. The composed map is smooth with a small curvature, so the error is
# O(h^2 * f'') with h = 1 px; measured worst case on these fixtures is < 0.1 px and this bound leaves margin.
DENSE_INTERP_TOL_PX = 0.25


def _smooth_fits(W, H, gc, gr, nr, T, amp=2.0, lam=18.0, amp_row=1.5, lam_row=15.0):
    """Identity-scale globals with a small, smooth, curvilinear mesh/row residual (time-invariant)."""
    fits = []
    for _k in range(T - 1):
        Hk = (CA.rotation_about_clockwise(W, H, 0.25) @ CA.translation(0.3, -0.2)).copy()
        mesh = CA.field_mesh(W, H, gc, gr, lambda p: np.stack(
            [amp * np.sin(p[:, 1] / lam), amp * np.cos(p[:, 0] / lam)], 1))
        rows = CA.row_field(W, H, nr, lambda y: np.stack(
            [np.zeros_like(y), amp_row * np.sin(y / lam_row)], 1))
        fits.append(Fit(Hk.copy(), mesh, rows, 1.0, 0.0, False))
    return fits


class _PlanLike:
    def __init__(self, G):
        self.G = list(G)


def _smooth_fixture(W=96, H=72, gc=12, gr=9, nr=18, T=5, shift=(0.37, 0.21)):
    """Motion with a sub-pixel virtual path so ``V_t = inv(Psi_t) @ plan.G[t]`` is a non-identity sub-pixel map.

    ``plan.G[t] = Psi_t @ Shift`` keeps the reference coordinates ``z = V_t(x)`` off the lattice nodes, so the
    dense-grid interpolation is genuinely exercised (with a pinned/identity path the probe would be exact).
    """
    fits = _smooth_fits(W, H, gc, gr, nr, T)
    shift_m = np.eye(3); shift_m[0, 2] = shift[0]; shift_m[1, 2] = shift[1]
    psi = np.eye(3)
    Gts = [psi @ shift_m]
    for k in range(T - 1):
        psi = fits[k].H @ psi
        Gts.append(psi @ shift_m)
    mot = S.Motion(W, H, 30.0, fits=fits, stride=1, n_frames=T)
    return mot, _PlanLike(Gts), S.Config(mesh=(gc, gr), rs_rows=nr)


def test_forward_map_recurrence_matches_exact_compose_reference():
    """The sequential recurrence must equal the direct per-frame composition at every lattice node."""
    mot, plan, _cfg = _smooth_fixture()
    W, H = int(mot.W), int(mot.H)
    fmap = R.forward_composed_map(plan, mot, range(mot.n_frames))
    lattice = fmap.lattice
    assert np.isfinite(fmap.phi).all()
    for t in range(mot.n_frames):
        fmap.advance_to(t)
        np.testing.assert_allclose(fmap.phi, R.CC.compose_reference(list(mot.fits), t, lattice, W, H),
                                   atol=0.0, err_msg=f"phi recurrence frame {t}")
        np.testing.assert_allclose(fmap.psi, R.CC.global_homography(list(mot.fits), t), atol=0.0,
                                   err_msg=f"psi recurrence frame {t}")


def test_dense_map_sampling_matches_direct_exact_within_tolerance():
    """Bilinear sampling of the dense map must track the exact composition within the interpolation bound."""
    mot, plan, cfg = _smooth_fixture()
    W, H = int(mot.W), int(mot.H)
    cells = R.CC.reference_cells(cfg)
    points = R.comparison_points(W, H)
    fmap = R.forward_composed_map(plan, mot, range(mot.n_frames))
    worst = 0.0
    for t in range(mot.n_frames):
        approx = fmap.source_coords(plan, t, points, cells=cells, highpass=False, smoothed=None,
                                    smoothing_basis=None)
        exact = R.CC.candidate_source_coords(mot, plan, cfg, t, points=points, cells=cells, highpass=False)
        err = np.linalg.norm(np.asarray(approx) - np.asarray(exact), axis=1)
        worst = max(worst, float(err.max()))
        stats, raw = R.dense_vs_exact_errors(fmap, mot, plan, cfg, t, points=points, cells=cells,
                                             highpass=False, smoothed=None, smoothing_basis=None)
        assert set(stats) == {"frame", "points", "max_px", "mean_px", "p95_px"}
        assert stats["frame"] == t and stats["points"] == points.shape[0]
        assert np.isfinite([stats["max_px"], stats["mean_px"], stats["p95_px"]]).all()
        np.testing.assert_allclose(raw, err, atol=0.0)
    assert 0.0 < worst < DENSE_INTERP_TOL_PX


def test_dense_map_highpass_sampling_matches_direct_exact():
    """Both temporal-high-pass charts sample the dense map within the same interpolation bound."""
    mot, plan, cfg = _smooth_fixture()
    W, H = int(mot.W), int(mot.H)
    cells = R.CC.reference_cells(cfg)
    points = R.comparison_points(W, H)
    fmap = R.forward_composed_map(plan, mot, range(mot.n_frames))
    for basis in ("source", "reference"):
        smoothed = R.CC.smoothed_delta(mot, plan, cfg, cells=cells, smoothing_basis=basis)
        fmap = R.forward_composed_map(plan, mot, range(mot.n_frames))
        for t in range(mot.n_frames):
            approx = fmap.source_coords(plan, t, points, cells=cells, highpass=True, smoothed=smoothed,
                                        smoothing_basis=basis)
            exact = R.CC.candidate_source_coords(mot, plan, cfg, t, points=points, cells=cells, highpass=True,
                                                 smoothed=smoothed, smoothing_basis=basis)
            err = np.linalg.norm(np.asarray(approx) - np.asarray(exact), axis=1)
            assert err.max() < DENSE_INTERP_TOL_PX, f"{basis} frame {t}: {err.max():.4g} px"


# A shift large enough that ``V_t(x) = x + shift`` leaves the reference frame on both axes, so the padded
# lattice must extend into negative y and past H-1; the old unpadded/border-clamped lattice reported large
# errors at those output points.
OFFFRAME_SHIFT = (-37.0, 41.0)


def _offframe_fixture(W=96, H=72, gc=12, gr=9, nr=18, T=5, shift=OFFFRAME_SHIFT):
    """Sub-pixel-curved motion with a constant off-frame virtual path ``V_t = Translate(shift)``."""
    fits = _smooth_fits(W, H, gc, gr, nr, T)
    translate = np.eye(3); translate[0, 2] = shift[0]; translate[1, 2] = shift[1]
    psi = np.eye(3)
    Gts = [psi @ translate]
    for k in range(T - 1):
        psi = fits[k].H @ psi
        Gts.append(psi @ translate)
    mot = S.Motion(W, H, 30.0, fits=fits, stride=1, n_frames=T)
    return mot, _PlanLike(Gts), S.Config(mesh=(gc, gr), rs_rows=nr)


def test_padded_map_covers_offframe_v_and_matches_exact():
    """An off-frame ``V_t(x)`` must be covered by the padded lattice, with exact dense interpolation."""
    mot, plan, cfg = _offframe_fixture()
    W, H = int(mot.W), int(mot.H)
    cells = R.CC.reference_cells(cfg)
    # Output-space corners/edge points: V_t(x) = x + shift leaves the reference frame on both axes.
    points = np.array([[0.0, 0.0], [W - 1.0, 0.0], [W - 1.0, H - 1.0], [0.0, H - 1.0],
                       [W // 2.0, H - 1.0], [0.0, H // 2.0], [W - 1.0, H // 2.0]], np.float64)
    z = points + np.array(OFFFRAME_SHIFT)
    assert z[:, 0].min() < 0.0                                   # leaves the frame on the left
    assert z[:, 1].max() > H - 1.0                               # and past the bottom edge
    fmap = R.forward_composed_map(plan, mot, range(mot.n_frames))
    # One guard pixel around the raw query bounds, so every bilinear tap stays inside the lattice.
    assert fmap.origin[0] <= z[:, 0].min() - 0.0 and fmap.origin[1] <= z[:, 1].min() - 0.0
    assert fmap.origin[0] + fmap.grid_shape[1] - 1 >= z[:, 0].max() + 1.0
    assert fmap.origin[1] + fmap.grid_shape[0] - 1 >= z[:, 1].max() + 1.0
    for t in range(mot.n_frames):
        approx = fmap.source_coords(plan, t, points, cells=cells, highpass=False, smoothed=None,
                                    smoothing_basis=None)
        exact = R.CC.candidate_source_coords(mot, plan, cfg, t, points=points, cells=cells, highpass=False)
        err = np.linalg.norm(np.asarray(approx) - np.asarray(exact), axis=1)
        assert err.max() < DENSE_INTERP_TOL_PX, f"off-frame frame {t}: {err.max():.4g} px"


def test_offframe_border_clamp_would_be_large_unpadded():
    """Document the boundary bug: the unpadded lattice border-clamps off-frame queries and reports large errors."""
    mot, plan, cfg = _offframe_fixture()
    W, H = int(mot.W), int(mot.H)
    cells = R.CC.reference_cells(cfg)
    points = np.array([[W - 1.0, H - 1.0], [0.0, H - 1.0]], np.float64)
    exact = R.CC.candidate_source_coords(mot, plan, cfg, 0, points=points, cells=cells, highpass=False)
    unpadded = R.ForwardComposedMap(mot)
    clamped = unpadded.source_coords(plan, 0, points, cells=cells, highpass=False, smoothed=None,
                                     smoothing_basis=None)
    padded = R.forward_composed_map(plan, mot, range(mot.n_frames))
    interpolated = padded.source_coords(plan, 0, points, cells=cells, highpass=False, smoothed=None,
                                        smoothing_basis=None)
    clamp_err = np.linalg.norm(np.asarray(clamped) - np.asarray(exact), axis=1)
    padded_err = np.linalg.norm(np.asarray(interpolated) - np.asarray(exact), axis=1)
    assert clamp_err.max() > padded_err.max()
    assert padded_err.max() < DENSE_INTERP_TOL_PX


def test_padded_map_refuses_projective_horizon():
    """A ``V_t`` whose denominator changes sign over the output rectangle is refused."""
    mot, _plan, _cfg = _smooth_fixture(T=2)
    W = int(mot.W)
    crossing = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, -(W - 1.0) / 2.0]])
    G = [crossing, mot.fits[0].H @ crossing]
    with pytest.raises(R.RealCompositionRenderError) as ei:
        R.forward_composed_map(_PlanLike(G), mot, range(2))
    assert ei.value.code == "projective_horizon"


def test_padded_map_refuses_nonfinite_mapping():
    """A non-finite ``V_t`` (NaN plan entry) is refused before any lattice is built."""
    mot, _plan, _cfg = _smooth_fixture(T=2)
    bad = np.eye(3); bad[0, 2] = np.nan
    G = [bad, np.asarray(mot.fits[0].H, np.float64) @ bad]
    with pytest.raises(R.RealCompositionRenderError) as ei:
        R.forward_composed_map(_PlanLike(G), mot, range(2))
    assert ei.value.code == "nonfinite_mapping"


def test_padded_map_refuses_singular_global_path():
    """A singular accumulated ``Psi_t`` cannot bound ``V_t`` and is refused."""
    mot, _plan, _cfg = _smooth_fixture(T=2)
    mot.fits[0].H = np.zeros((3, 3))
    with pytest.raises(R.RealCompositionRenderError) as ei:
        R.forward_composed_map(_PlanLike([np.eye(3), np.eye(3)]), mot, range(2))
    assert ei.value.code == "projective_horizon"


def test_forward_map_rejects_nonfinite_lattice():
    """A non-finite propagated lattice is refused (map rejection), not silently rendered."""
    mot, _plan, _cfg = _smooth_fixture(T=2)
    mot.fits[0].mesh = np.full_like(mot.fits[0].mesh, np.inf)
    fmap = R.ForwardComposedMap(mot)
    with pytest.raises(R.RealCompositionRenderError) as ei:
        fmap.advance_to(1)
    assert ei.value.code == "nonfinite_map"


# --------------------------------------------------------------------------------------
# pairing / mapping
# --------------------------------------------------------------------------------------

def test_full_capture_300_outputs_pair_to_600_source_frames(tmp_path, monkeypatch):
    run, _ = _setup(monkeypatch, tmp_path, source_frames=600, output_frames=300,
                    fps_source=60000 / 1001, fps_out=30000 / 1001)
    video, report = _out(tmp_path)
    rep = R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video,
                             out_json=report)
    assert rep["pairing"]["frame_indices"] == list(range(0, 600, 2))
    assert rep["pairing"]["rendered_source_frames"] == 300
    assert rep["pairing"]["source_frames"] == 600 and rep["pairing"]["output_frames"] == 300
    assert len(FakeEncoder.instances[0].frames) == 300


def test_frame_render_pairing_uses_mapped_indices(tmp_path, monkeypatch):
    run, _ = _setup(monkeypatch, tmp_path, source_frames=6, output_frames=3)
    video, report = _out(tmp_path)
    R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video, out_json=report)
    # Only the mapped source frames 0, 2, 4 are decoded into the renderer; the encoder sees exactly those.
    assert [call["field_shape"] for call in RENDER_CALLS] == [(H, W, 2)] * 3
    assert all(call["finite"] for call in RENDER_CALLS)
    written = [int(f[0, 0, 0]) for f in FakeEncoder.instances[0].frames]
    assert written == [0, 60, 120]


def test_encoder_uses_exact_rational_output_rate(tmp_path, monkeypatch):
    run, _ = _setup(monkeypatch, tmp_path, source_frames=6, output_frames=3,
                    fps_source=60000 / 1001, fps_out=30000 / 1001)
    video, report = _out(tmp_path)
    rep = R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video, out_json=report)
    assert FakeEncoder.instances[0].fps == Fraction(30000, 1001)
    assert rep["output"]["cfr"]["ok"] is True


# --------------------------------------------------------------------------------------
# mode and smoothing chart
# --------------------------------------------------------------------------------------

def test_exact_composition_needs_no_smoothing(tmp_path, monkeypatch):
    run, _ = _setup(monkeypatch, tmp_path)
    calls = []
    original = R.CC.smoothed_delta

    def spy(mot, plan, cfg, *, cells=None, smoothing_basis="source"):
        calls.append(smoothing_basis)
        return original(mot, plan, cfg, cells=cells, smoothing_basis=smoothing_basis)

    monkeypatch.setattr(R.CC, "smoothed_delta", spy)
    field_kwargs = []
    original_field = R.build_candidate_field

    def spy_field(*args, **kwargs):
        field_kwargs.append(dict(kwargs))
        return original_field(*args, **kwargs)

    monkeypatch.setattr(R, "build_candidate_field", spy_field)
    video, report = _out(tmp_path)
    R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video, out_json=report)
    assert calls == []
    assert field_kwargs and all(kw["highpass"] is False and kw["smoothed"] is None
                                for kw in field_kwargs)


@pytest.mark.parametrize("mode,basis", [("source_highpass", "source"),
                                         ("reference_highpass", "reference")])
def test_highpass_mode_precomputes_matching_chart_once(tmp_path, monkeypatch, mode, basis):
    run, _ = _setup(monkeypatch, tmp_path)
    calls = []
    original = R.CC.smoothed_delta

    def spy(mot, plan, cfg, *, cells=None, smoothing_basis="source"):
        calls.append(smoothing_basis)
        return original(mot, plan, cfg, cells=cells, smoothing_basis=smoothing_basis)

    monkeypatch.setattr(R.CC, "smoothed_delta", spy)
    field_kwargs = []
    original_field = R.build_candidate_field

    def spy_field(*args, **kwargs):
        field_kwargs.append(dict(kwargs))
        return original_field(*args, **kwargs)

    monkeypatch.setattr(R, "build_candidate_field", spy_field)
    video, report = _out(tmp_path)
    R.render_candidate(run / "manifest.json", mode=mode, out_video=video, out_json=report)
    assert calls == [basis]                                   # computed once, not per frame
    assert field_kwargs and all(kw["highpass"] is True and kw["smoothing_basis"] == basis
                                for kw in field_kwargs)
    assert all(kw["smoothed"] is not None for kw in field_kwargs)


def test_unsupported_mode_refused_before_any_write(tmp_path, monkeypatch):
    run, _ = _setup(monkeypatch, tmp_path)
    video, report = _out(tmp_path)
    with pytest.raises(R.RealCompositionRenderError) as ei:
        R.render_candidate(run / "manifest.json", mode="bogus", out_video=video, out_json=report)
    assert ei.value.code == "unsupported_mode"
    assert not video.exists() and not report.exists()


# --------------------------------------------------------------------------------------
# provenance / hashes
# --------------------------------------------------------------------------------------

def test_repo_relative_plan_motion_resolve_to_manifest_siblings(tmp_path, monkeypatch):
    """A manifest may record repo-root-relative outputs.plan/motion while the captured files are siblings.

    Regression: ``_resolve_input`` prepended the manifest directory to the whole relative string
    (``RUN/out/bench/RUN/plan_r0.npz``), so the sibling capture was never found and the render refused
    with ``plan_missing``. The relative value must resolve to the manifest's sibling by basename, with
    ``load_composition_inputs`` still validating sibling/capture/basename provenance.
    """
    run, manifest = _setup(monkeypatch, tmp_path)
    rel_run = Path("out") / "bench" / "current_source_t060_10s_baseline"
    manifest["outputs"]["plan"] = str(rel_run / "plan_r0.npz")
    manifest["outputs"]["motion"] = str(rel_run / "motion_r0.npz")
    (run / "manifest.json").write_text(json.dumps(manifest))

    video, report = _out(tmp_path)
    rep = R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video,
                             out_json=report)
    assert Path(rep["plan"]["path"]) == (run / "plan_r0.npz").resolve()
    assert Path(rep["motion"]["path"]) == (run / "motion_r0.npz").resolve()
    assert rep["plan"]["sha256"] == MF.sha256_file(run / "plan_r0.npz")
    assert rep["motion"]["sha256"] == MF.sha256_file(run / "motion_r0.npz")
    assert rep["output"]["frames"] == 3


def test_absolute_plan_motion_paths_preserved(tmp_path, monkeypatch):
    """Absolute manifest artifact paths are used verbatim, not rebased against the manifest directory."""
    run, manifest = _setup(monkeypatch, tmp_path)
    other = tmp_path / "relocated"
    other.mkdir()
    for name in ("plan_r0.npz", "motion_r0.npz"):
        (other / name).write_bytes((run / name).read_bytes())
    manifest["outputs"]["plan"] = str((other / "plan_r0.npz").resolve())
    manifest["outputs"]["motion"] = str((other / "motion_r0.npz").resolve())
    (run / "manifest.json").write_text(json.dumps(manifest))

    # Non-sibling artifacts are refused: the absolute path is preserved and the strict loader rejects it.
    video, report = _out(tmp_path)
    with pytest.raises(R.RealCompositionRenderError) as ei:
        R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video,
                           out_json=report)
    assert ei.value.code == "manifest_missing"
    assert not video.exists() and not report.exists()


def test_report_records_dense_approximation_and_error_fields(tmp_path, monkeypatch):
    """The report names the dense-grid approximation and records the direct exact-vs-dense error fields."""
    run, _ = _setup(monkeypatch, tmp_path)
    video, report = _out(tmp_path)
    rep = R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video, out_json=report)
    approx = rep["approximation"]
    assert approx["name"] == R.APPROX_NAME
    assert approx["grid"] == R.APPROX_GRID
    assert approx["frame_shape"] == [H, W]
    # The lattice is padded past the frame on every side by at least the guard pixel.
    assert approx["grid_shape"][0] >= H + 2 and approx["grid_shape"][1] >= W + 2
    assert approx["padding"] == R.GRID_GUARD_PX
    ox, oy = approx["grid_origin"]
    ny, nx = approx["grid_shape"]
    assert ox <= 0 and oy <= 0
    assert ox + nx - 1 >= W - 1 and oy + ny - 1 >= H - 1
    # The recorded corner query bounds cover the frame and drive the padded lattice.
    qb = approx["query_bounds_px"]
    assert qb["min_x"] <= 1e-9 and qb["min_y"] <= 1e-9
    assert qb["max_x"] >= W - 1 - 1e-6 and qb["max_y"] >= H - 1 - 1e-6
    assert np.floor(qb["min_x"]) - approx["padding"] == ox
    assert np.floor(qb["min_y"]) - approx["padding"] == oy
    assert approx["interpolation"] == R.APPROX_INTERPOLATION
    assert approx["recurrence"].startswith("phi_0 = pixel-center identity")
    assert rep["render_sampler"]["field_builder"].startswith("bench.real_composition_render.ForwardComposedMap")
    cmp = rep["exact_comparison"]
    assert cmp["points_per_frame"] == R.comparison_points(W, H).shape[0]
    assert len(cmp["per_frame"]) == len(cmp["output_frame_positions"]) == len(cmp["source_frames"])
    assert cmp["per_frame"]
    for stats in cmp["per_frame"]:
        assert {"frame", "points", "max_px", "mean_px", "p95_px"} <= set(stats)
        assert stats["points"] == cmp["points_per_frame"]
    agg = cmp["aggregate"]
    assert agg["frames_compared"] == len(cmp["per_frame"])
    assert agg["points"] == agg["frames_compared"] * cmp["points_per_frame"]
    assert np.isfinite([agg["max_px"], agg["mean_px"], agg["p95_px"]]).all()
    on_disk = json.loads(report.read_text())
    assert on_disk["approximation"]["name"] == R.APPROX_NAME
    assert on_disk["approximation"]["grid_origin"] == approx["grid_origin"]
    assert on_disk["approximation"]["padding"] == approx["padding"]
    assert on_disk["exact_comparison"]["aggregate"] == agg


def test_report_hashes_and_adjacency(tmp_path, monkeypatch):
    run, manifest = _setup(monkeypatch, tmp_path)
    video, report = _out(tmp_path)
    rep = R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video, out_json=report)
    assert rep["schema"] == R.SCHEMA and rep["ok"] is True and rep["no_network"] is True
    assert rep["source"]["sha256"] == manifest["source"]["sha256"]
    assert rep["source"]["manifest"]["sha256"] == MF.sha256_file(run / "manifest.json")
    assert rep["baseline"]["sha256"] == MF.sha256_file(run / "stabilized.mp4")
    assert rep["plan"]["sha256"] == MF.sha256_file(run / "plan_r0.npz")
    assert rep["motion"]["sha256"] == MF.sha256_file(run / "motion_r0.npz")
    assert rep["output"]["sha256"] == MF.sha256_file(video)
    assert Path(rep["output"]["report"]).parent == Path(rep["output"]["video"]).parent
    on_disk = json.loads(report.read_text())
    assert on_disk["output"]["sha256"] == rep["output"]["sha256"]
    assert "render_sampler" in on_disk and on_disk["render_sampler"]["renderer"] == "mlxstab.stabilize.render_from_src"


# --------------------------------------------------------------------------------------
# refusals, failures and cleanup
# --------------------------------------------------------------------------------------

def test_early_decode_refused_and_cleaned(tmp_path, monkeypatch):
    run, _ = _setup(monkeypatch, tmp_path, source_frames=6, output_frames=3, decode_extra=-2)
    video, report = _out(tmp_path)
    with pytest.raises(R.RealCompositionRenderError) as ei:
        R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video, out_json=report)
    assert ei.value.code == "early_decode"
    assert not video.exists() and not report.exists()
    assert not list(tmp_path.glob(".candidate.*.mp4"))


def test_extra_decode_refused_and_cleaned(tmp_path, monkeypatch):
    run, _ = _setup(monkeypatch, tmp_path, source_frames=6, output_frames=3, decode_extra=1)
    video, report = _out(tmp_path)
    with pytest.raises(R.RealCompositionRenderError) as ei:
        R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video, out_json=report)
    assert ei.value.code == "extra_decode"
    assert not video.exists() and not report.exists()


def test_inconsistent_timeline_refused(tmp_path, monkeypatch):
    run, manifest = _setup(monkeypatch, tmp_path)
    manifest["source"]["decoded"]["frames"] = 5
    (run / "manifest.json").write_text(json.dumps(manifest))
    video, report = _out(tmp_path)
    with pytest.raises(R.RealCompositionRenderError) as ei:
        R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video, out_json=report)
    assert ei.value.code == "timeline_frames"


def test_malformed_cadence_tie_refused(tmp_path, monkeypatch):
    run, _ = _setup(monkeypatch, tmp_path, source_frames=4, output_frames=3,
                    fps_source=60.0, fps_out=40.0)
    video, report = _out(tmp_path)
    with pytest.raises(R.RealCompositionRenderError) as ei:
        R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video, out_json=report)
    assert ei.value.code == "cadence"


def test_nonfinite_field_refused_and_cleaned(tmp_path, monkeypatch):
    run, _ = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(R, "build_candidate_field",
                        lambda *a, **k: np.full((H, W, 2), np.nan))
    video, report = _out(tmp_path)
    with pytest.raises(R.RealCompositionRenderError) as ei:
        R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video, out_json=report)
    assert ei.value.code == "nonfinite_field"
    assert not video.exists() and not report.exists()


def test_bad_field_shape_refused(tmp_path, monkeypatch):
    run, _ = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(R, "build_candidate_field", lambda *a, **k: np.zeros((H, W, 3)))
    video, report = _out(tmp_path)
    with pytest.raises(R.RealCompositionRenderError) as ei:
        R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video, out_json=report)
    assert ei.value.code == "bad_field_shape"


# --------------------------------------------------------------------------------------
# invalid-support provenance diagnostics (center-out-of-bounds mask from render_from_src)
# --------------------------------------------------------------------------------------

def _support_masks():
    """Three known masks over the 12x16 test frame exercising the fixed 4-px edge strips.

    Strip geometry: top/bottom = 4 rows x 16 cols (area 64), left/right = 12 rows x 4 cols (area 48).
    Per frame invalid counts are 2 (top only), 4 (left only) and 1 (interior only), so the per-frame and
    aggregate mean/max and pooled edge fractions all differ and can be pinned exactly.
    """
    m0 = np.zeros((H, W), bool)
    m0[0, 6] = m0[0, 7] = True              # top strip only (cols 6,7 are inside the horizontal interior)
    m1 = np.zeros((H, W), bool)
    m1[5, 0] = m1[6, 0] = m1[5, 1] = m1[6, 1] = True   # left strip only (rows 5,6 are inside the vertical interior)
    m2 = np.zeros((H, W), bool)
    m2[6, 6] = True                         # interior only, no edge strip
    return [m0, m1, m2]


def test_invalid_support_per_frame_and_aggregate_values(tmp_path, monkeypatch):
    """The border mask is kept (not discarded) and yields exact per-frame/aggregate invalid-support values."""
    run, _ = _setup(monkeypatch, tmp_path, source_frames=6, output_frames=3)
    masks = _support_masks()
    calls = {"n": 0}

    def fake_known_mask(frame, field, cfg):
        mask = masks[calls["n"] % len(masks)]
        calls["n"] += 1
        return np.array(frame, copy=True), mask.copy()

    monkeypatch.setattr(R, "render_candidate_frame", fake_known_mask)
    video, report = _out(tmp_path)
    rep = R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video, out_json=report)

    sup = rep["invalid_support"]
    assert sup["strip_width_px"] == R.SUPPORT_STRIP_PX == 4
    assert sup["definition"] == R.SUPPORT_DEFINITION
    assert "NOT a perceptual" in sup["definition"]
    assert "center-out-of-bounds" in sup["definition"]

    per = sup["per_frame"]
    assert len(per) == 3
    assert [p["frame"] for p in per] == [0, 1, 2]
    assert [p["source_frame"] for p in per] == rep["pairing"]["frame_indices"]
    assert [p["invalid_pixels"] for p in per] == [2, 4, 1]
    frame_px = H * W
    assert [p["invalid_fraction"] for p in per] == pytest.approx([2 / frame_px, 4 / frame_px, 1 / frame_px])
    # frame 0: top strip only; frame 1: left strip only; frame 2: interior, so no edge strip.
    assert per[0]["edge_fraction"]["top"] == pytest.approx(2 / 64)
    assert per[0]["edge_fraction"]["left"] == 0.0
    assert per[1]["edge_fraction"]["left"] == pytest.approx(4 / 48)
    assert per[2]["edge_fraction"] == {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}
    assert per[0]["edge_invalid_pixels"]["top"] == 2

    agg = sup["aggregate"]
    assert agg["frames"] == 3 and agg["frame_pixels"] == frame_px
    assert agg["total_pixels"] == 3 * frame_px
    assert agg["total_invalid_pixels"] == 7
    assert agg["pooled_invalid_fraction"] == pytest.approx(7 / (3 * frame_px))
    assert agg["mean_invalid_fraction"] == pytest.approx(7 / (3 * frame_px))
    assert agg["max_invalid_fraction"] == pytest.approx(4 / frame_px)
    # mean and max differ, so both are genuinely reported.
    assert agg["max_invalid_fraction"] > agg["mean_invalid_fraction"]
    left = agg["edge"]["left"]
    assert left["strip_px"] == 4 and left["per_frame_pixels"] == 48
    assert left["total_pixels"] == 3 * 48
    assert left["total_invalid_pixels"] == 4
    assert left["pooled_fraction"] == pytest.approx(4 / (3 * 48))
    assert left["mean_fraction"] == pytest.approx(4 / (3 * 48))
    assert left["max_fraction"] == pytest.approx(4 / 48)
    assert agg["edge"]["bottom"]["total_invalid_pixels"] == 0
    assert agg["edge"]["top"]["total_invalid_pixels"] == 2
    assert agg["edge"]["right"]["total_invalid_pixels"] == 0

    # The report is JSON on disk, and no mask video/artifact was produced.
    on_disk = json.loads(report.read_text())
    assert on_disk["invalid_support"]["aggregate"] == agg
    assert on_disk["invalid_support"]["per_frame"] == per
    assert not list(tmp_path.glob("*mask*"))


def test_bad_mask_shape_refused_and_cleaned(tmp_path, monkeypatch):
    run, _ = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(R, "render_candidate_frame",
                        lambda frame, field, cfg: (np.array(frame, copy=True), np.zeros((H, W + 1), bool)))
    video, report = _out(tmp_path)
    with pytest.raises(R.RealCompositionRenderError) as ei:
        R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video, out_json=report)
    assert ei.value.code == "bad_mask_shape"
    assert not video.exists() and not report.exists()
    assert not list(tmp_path.glob(".candidate.*.mp4"))


def test_bad_mask_dtype_refused_and_cleaned(tmp_path, monkeypatch):
    run, _ = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(R, "render_candidate_frame",
                        lambda frame, field, cfg: (np.array(frame, copy=True), np.full((H, W), "x")))
    video, report = _out(tmp_path)
    with pytest.raises(R.RealCompositionRenderError) as ei:
        R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video, out_json=report)
    assert ei.value.code == "bad_mask_dtype"
    assert not video.exists() and not report.exists()


def test_nonfinite_render_refused_and_cleaned(tmp_path, monkeypatch):
    run, _ = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(R, "render_candidate_frame",
                        lambda frame, field, cfg: (np.full(frame.shape, np.nan), np.zeros(frame.shape[:2], bool)))
    video, report = _out(tmp_path)
    with pytest.raises(R.RealCompositionRenderError) as ei:
        R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video, out_json=report)
    assert ei.value.code == "nonfinite_render"
    assert not video.exists() and not report.exists()


def test_encode_failure_refused_and_cleaned(tmp_path, monkeypatch):
    run, _ = _setup(monkeypatch, tmp_path)

    class BoomEncoder(FakeEncoder):
        def write(self, frame):
            raise RuntimeError("encoder exploded")

    monkeypatch.setattr(media, "Encoder", BoomEncoder)
    video, report = _out(tmp_path)
    with pytest.raises(R.RealCompositionRenderError) as ei:
        R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video, out_json=report)
    assert ei.value.code == "encode_failure"
    assert not video.exists() and not report.exists()


def test_output_cfr_failure_refused_and_cleaned(tmp_path, monkeypatch):
    run, _ = _setup(monkeypatch, tmp_path)

    def bad_pts(path):
        p = Path(path)
        if p.name == "stabilized.mp4":
            return _pts(3, OUT_FPS)
        return _pts(3, 25.0)                       # wrong clock for the generated candidate

    monkeypatch.setattr(media, "frame_pts", bad_pts)
    video, report = _out(tmp_path)
    with pytest.raises(R.RealCompositionRenderError) as ei:
        R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video, out_json=report)
    assert ei.value.code == "candidate_cfr"
    assert not video.exists() and not report.exists()


def test_source_sha_mismatch_refused(tmp_path, monkeypatch):
    run, manifest = _setup(monkeypatch, tmp_path)
    (run / "source.mts").write_bytes(b"tampered")
    video, report = _out(tmp_path)
    with pytest.raises(R.RealCompositionRenderError) as ei:
        R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video, out_json=report)
    assert ei.value.code == "source_sha"


@pytest.mark.parametrize("field,value,code", [("status", "failure", "manifest_status"),
                                              ("schema", "wrong", "manifest_schema")])
def test_manifest_status_and_schema_refused(tmp_path, monkeypatch, field, value, code):
    run, manifest = _setup(monkeypatch, tmp_path)
    manifest[field] = value
    (run / "manifest.json").write_text(json.dumps(manifest))
    video, report = _out(tmp_path)
    with pytest.raises(R.RealCompositionRenderError) as ei:
        R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video, out_json=report)
    assert ei.value.code == code


def test_output_overwrite_refused(tmp_path, monkeypatch):
    run, _ = _setup(monkeypatch, tmp_path)
    video, report = _out(tmp_path)
    video.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video, out_json=report)


def test_output_collision_with_inputs_refused(tmp_path, monkeypatch):
    run, _ = _setup(monkeypatch, tmp_path)
    video = run / "candidate.mp4"
    with pytest.raises(R.RealCompositionRenderError) as ei:
        R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video,
                           out_json=run / "manifest.json")
    assert ei.value.code == "output_collision"


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def test_cli_success(tmp_path, monkeypatch, capsys):
    run, _ = _setup(monkeypatch, tmp_path)
    video, report = _out(tmp_path)
    rc = R.main([str(run / "manifest.json"), "--mode", "exact_composition",
                 "--out-video", str(video), "--out-json", str(report), "--lock", "off"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "success" and out["mode"] == "exact_composition"
    assert out["frames"] == 3 and Path(out["video"]) == video


def test_cli_error_is_json_safe(tmp_path, monkeypatch, capsys):
    run, manifest = _setup(monkeypatch, tmp_path)
    manifest["status"] = "failure"
    (run / "manifest.json").write_text(json.dumps(manifest))
    video, report = _out(tmp_path)
    rc = R.main([str(run / "manifest.json"), "--mode", "exact_composition",
                 "--out-video", str(video), "--out-json", str(report), "--lock", "off"])
    assert rc == 2
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False and out["error"]["code"] == "manifest_status"


def test_cli_requires_mode_and_out_video():
    with pytest.raises(SystemExit):
        R.main(["manifest.json"])
    with pytest.raises(SystemExit):
        R.main(["manifest.json", "--mode", "exact_composition"])


# --------------------------------------------------------------------------------------
# shared heavy lock (injected fake; unit tests never acquire a real lock)
# --------------------------------------------------------------------------------------

def test_function_seam_defaults_to_no_lock(tmp_path, monkeypatch):
    """The Python seam defaults to ``lock=None``: nothing is acquired and the report records disabled."""
    run, _ = _setup(monkeypatch, tmp_path)
    FakeLock.instances = []
    video, report = _out(tmp_path)
    rep = R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video,
                             out_json=report)
    assert FakeLock.instances == []                       # no real lock constructed/acquired
    lock = rep["lock"]
    assert lock["backend"] == "disabled" and lock["enabled"] is False
    assert lock["chunk"] is None and lock["pause_s"] is None
    assert lock["releases"] == 0 and lock["acquire_wait_s"] == 0.0 and lock["pause_total_s"] == 0.0
    assert "path" not in lock                             # private lock paths are never published
    assert "scope" in lock


def test_lock_on_step_after_every_output_frame_and_release_on_success(tmp_path, monkeypatch):
    run, _ = _setup(monkeypatch, tmp_path, source_frames=6, output_frames=3)
    lock = FakeLock(chunk=30, pause_s=12.0)
    video, report = _out(tmp_path)
    rep = R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video,
                             out_json=report, lock=lock)
    assert lock.acquires == 1
    assert lock.steps == [1, 2, 3]                        # after each output frame, 1-based
    assert lock.releases == 1                             # final release (chunk=30 > 3 frames)
    snap = rep["lock"]
    assert snap["backend"] == "fake" and snap["enabled"] is True
    assert snap["chunk"] == 30 and snap["pause_s"] == 12.0
    assert snap["held"] is False and snap["releases"] == 1
    assert "path" not in snap                             # FakeLock.status() has a path; it is dropped
    assert json.loads(report.read_text())["lock"] == snap


def test_lock_on_step_release_reacquire_at_chunk_boundaries(tmp_path, monkeypatch):
    """A real HeavyLock-style chunk gate is exercised when the fake forwards on_step to release/acquire."""
    run, _ = _setup(monkeypatch, tmp_path, source_frames=6, output_frames=3)

    class ChunkyLock(FakeLock):
        def on_step(self, step):
            super().on_step(step)
            if step % 2 == 0:
                self.release()
                self.acquire()

    lock = ChunkyLock(chunk=2, pause_s=0.0)
    video, report = _out(tmp_path)
    R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video,
                       out_json=report, lock=lock)
    assert lock.steps == [1, 2, 3]
    assert lock.releases == 2                             # chunk yield at frame 2 + final release
    assert lock.acquires == 2                             # initial + re-acquire after the yield


def test_lock_released_on_failure(tmp_path, monkeypatch):
    run, _ = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(R, "build_candidate_field", lambda *a, **k: np.full((H, W, 2), np.nan))
    lock = FakeLock(chunk=2, pause_s=0.0)
    video, report = _out(tmp_path)
    with pytest.raises(R.RealCompositionRenderError) as ei:
        R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video,
                           out_json=report, lock=lock)
    assert ei.value.code == "nonfinite_field"
    assert lock.acquires == 1 and lock.releases == 1 and lock.held is False
    assert not video.exists() and not report.exists()


def test_lock_released_when_source_decode_fails(tmp_path, monkeypatch):
    """A failure between acquire and the render loop (decode validation) still releases the lock."""
    run, _ = _setup(monkeypatch, tmp_path, decode_extra=-2)
    lock = FakeLock(chunk=30, pause_s=0.0)
    video, report = _out(tmp_path)
    with pytest.raises(R.RealCompositionRenderError) as ei:
        R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video,
                           out_json=report, lock=lock)
    assert ei.value.code == "early_decode"
    assert lock.acquires == 1 and lock.releases == 1 and lock.held is False
    assert not video.exists() and not report.exists()


def test_lock_acquire_failure_leaves_no_partial_outputs(tmp_path, monkeypatch):
    """If ``acquire()`` raises, the error surfaces before any output exists and nothing is released."""
    run, _ = _setup(monkeypatch, tmp_path)

    class RefusingLock(FakeLock):
        def acquire(self):
            self.acquires += 1
            raise OSError("lock acquisition failed")

    lock = RefusingLock(chunk=30, pause_s=0.0)
    video, report = _out(tmp_path)
    with pytest.raises(OSError):
        R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video,
                           out_json=report, lock=lock)
    assert lock.acquires == 1 and lock.releases == 0   # a failed acquire was never held, so nothing to release
    assert not video.exists() and not report.exists()
    assert not list(tmp_path.glob(".candidate.*.mp4"))


def test_lock_snapshot_failure_cleans_partial_outputs(tmp_path, monkeypatch):
    """A report-building failure after a successful render still removes the video (no partial artifact)."""
    run, _ = _setup(monkeypatch, tmp_path)

    class BadStatusLock(FakeLock):
        def status(self):
            raise RuntimeError("lock status unavailable")

    lock = BadStatusLock(chunk=30, pause_s=0.0)
    video, report = _out(tmp_path)
    with pytest.raises(RuntimeError, match="lock status unavailable"):
        R.render_candidate(run / "manifest.json", mode="exact_composition", out_video=video,
                           out_json=report, lock=lock)
    assert lock.acquires == 1 and lock.releases == 1 and lock.held is False
    assert not video.exists() and not report.exists()


def test_cli_lock_defaults(tmp_path, monkeypatch):
    run, _ = _setup(monkeypatch, tmp_path)
    video, report = _out(tmp_path)
    FakeLock.instances = []
    monkeypatch.setattr(R, "HeavyLock", FakeLock)
    rc = R.main([str(run / "manifest.json"), "--mode", "exact_composition",
                 "--out-video", str(video), "--out-json", str(report)])
    assert rc == 0
    assert len(FakeLock.instances) == 1
    built = FakeLock.instances[0]
    assert built.kwargs["enabled"] is True                # auto == on for file-backed manifests
    assert built.kwargs["chunk"] == R.DEFAULT_LOCK_CHUNK == 30
    assert built.kwargs["pause_s"] == R.DEFAULT_LOCK_PAUSE_S == 12.0
    assert built.kwargs["what"].startswith("mlxstab render ")
    assert built.acquires == 1 and built.releases == 1
    assert json.loads(report.read_text())["lock"]["backend"] == "fake"


def test_cli_lock_chunk_pause_overrides(tmp_path, monkeypatch):
    run, _ = _setup(monkeypatch, tmp_path)
    video, report = _out(tmp_path)
    FakeLock.instances = []
    monkeypatch.setattr(R, "HeavyLock", FakeLock)
    rc = R.main([str(run / "manifest.json"), "--mode", "exact_composition",
                 "--out-video", str(video), "--out-json", str(report),
                 "--lock", "on", "--lock-chunk", "2", "--lock-pause", "0.5"])
    assert rc == 0
    built = FakeLock.instances[0]
    assert built.kwargs["chunk"] == 2 and built.kwargs["pause_s"] == 0.5


def test_cli_lock_off_constructs_no_lock(tmp_path, monkeypatch):
    run, _ = _setup(monkeypatch, tmp_path)
    video, report = _out(tmp_path)
    FakeLock.instances = []
    monkeypatch.setattr(R, "HeavyLock", FakeLock)
    rc = R.main([str(run / "manifest.json"), "--mode", "exact_composition",
                 "--out-video", str(video), "--out-json", str(report), "--lock", "off"])
    assert rc == 0
    assert FakeLock.instances == []
    assert json.loads(report.read_text())["lock"]["backend"] == "disabled"
