"""Weight-free tests for the cache-driven relative-depth temporal pilot.

The pilot is exercised with injected fakes for the RAFT runner and the heavy lock, so no weights, no MLX,
no media and no private footage are touched. Coverage: cache shape/hash/index validation, correspondence
guards, CLI parsing, deterministic aggregation, adjacent frame/depth mapping and cycle-valid wiring, and that
invalid inputs fail before any runner is constructed. Warnings are escalated to errors so an unclean numeric
path fails rather than passing silently.
"""
import json
import warnings

import numpy as np
import pytest

from bench import depth_temporal_pilot as P


@pytest.fixture(autouse=True)
def no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# --------------------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------------------

class FakeRunner:
    """Deterministic stand-in for :class:`bench.flow_prediction_real.RaftRunner` (no MLX, no weights)."""

    def __init__(self, weights, downscale=2, flow_dtype="float32", corr_dtype="float32", h=4, w=5):
        self.weights = weights
        self.downscale = int(downscale)
        self.flow_dtype = flow_dtype
        self.corr_dtype = corr_dtype
        self.h = int(h)
        self.w = int(w)
        self.pad = (1, 1, 0, 0)
        self.prep_s = 0.0
        self.prepared = None
        self.calls = []

    def prepare(self, frames, on_step=None):
        self.prepared = [np.asarray(f) for f in frames]
        if on_step is not None:
            for _ in self.prepared:
                on_step()

    def reference_pair(self, src, dst, iters):
        self.calls.append((int(src), int(dst), int(iters)))
        # Constant unit forward flow, zero backward flow: fully cycle-valid, so the audit's forward
        # cycle-valid mask is all True.
        fwd = np.ones((self.h, self.w, 2), np.float64)
        bwd = np.zeros((self.h, self.w, 2), np.float64)
        return fwd, bwd


class RaisingRunner:
    """Runner whose construction fails: used to prove invalid inputs never reach runner creation."""

    def __init__(self, *a, **k):
        raise AssertionError("runner must not be constructed for invalid inputs")


class RecordingLock:
    """Fake ``HeavyLock`` recording construction, steps and holding state without any real lock."""

    instances = []

    def __init__(self, what="", enabled=True, chunk=0, poll=10, pause_s=0.0, sleep=None):
        self.what = what
        self.enabled = enabled
        self.chunk = int(chunk or 0)
        self.pause_s = float(pause_s or 0.0)
        self.steps = []
        self.held = False
        self.releases = 0
        RecordingLock.instances.append(self)

    def on_step(self, step):
        self.steps.append(int(step))

    def status(self):
        return {"backend": "fake", "enabled": self.enabled, "chunk": self.chunk, "pause_s": self.pause_s,
                "held": self.held, "releases": self.releases, "steps": len(self.steps)}

    def __enter__(self):
        self.held = True
        return self

    def __exit__(self, *exc):
        self.held = False
        return False


# --------------------------------------------------------------------------------------
# fixtures / builders
# --------------------------------------------------------------------------------------

def _frames(n=6, h=8, w=10, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(n, h, w, 3), dtype=np.uint8)


def _meta(n=6, h=8, w=10, fps=30.0):
    return {"schema": "mlxstab.flow_prediction_real.frames/1", "count": n, "height": h, "width": w,
            "channels": 3, "fps": fps, "window": {"start_s": 1.0, "duration_s": 2.0}}


def _write_frames_npz(tmp_path, frames, meta):
    path = tmp_path / "frames.npz"
    np.savez(path, frames=frames, meta=np.array(json.dumps(meta)))
    return path


def _depth_info(indices, h=8, w=10, frames_sha="x"):
    depth = np.zeros((len(indices), h, w), np.float32)
    for k in range(len(indices)):
        depth[k] = k  # distinct maps so per-pair source/target are identifiable
    return {"indices": np.asarray(indices, np.int64), "depth": depth,
            "model_id": "fake-model", "normalization": "fake-norm", "frames_npz_sha256": frames_sha}


@pytest.fixture
def weights_file(tmp_path):
    path = tmp_path / "fake.safetensors"
    path.write_bytes(b"fake-weights")
    return path


def _run(frames, meta, depth_info, weights=None, tmp_weights=None, **kw):
    kw.setdefault("lock_factory", RecordingLock)
    kw.setdefault("runner_factory", FakeRunner)
    kw.setdefault("write", False)
    kw.setdefault("log", lambda m: None)
    if weights is None:
        weights = str(tmp_weights)
    return P.run_pilot(frames, meta, depth_info, weights, None, **kw)


# --------------------------------------------------------------------------------------
# cache loading / validation
# --------------------------------------------------------------------------------------

def test_load_frames_npz_round_trip(tmp_path):
    frames = _frames()
    meta = _meta()
    path = _write_frames_npz(tmp_path, frames, meta)
    got, got_meta = P.load_frames_npz(path)
    assert got.shape == frames.shape and got.dtype == np.uint8
    assert np.array_equal(got, frames)
    assert got_meta["fps"] == meta["fps"]


def test_load_frames_npz_rejects_wrong_dtype_and_shape(tmp_path):
    path = tmp_path / "f.npz"
    np.savez(path, frames=np.zeros((2, 4, 4, 3), np.float32), meta=np.array("{}"))
    with pytest.raises(ValueError, match="uint8"):
        P.load_frames_npz(path)
    np.savez(path, frames=np.zeros((2, 4, 4), np.uint8), meta=np.array("{}"))
    with pytest.raises(ValueError, match=r"\[N,H,W,3\]"):
        P.load_frames_npz(path)


def test_load_frames_npz_rejects_missing_fields(tmp_path):
    path = tmp_path / "f.npz"
    np.savez(path, frames=_frames())
    with pytest.raises(ValueError, match="must contain"):
        P.load_frames_npz(path)


def test_load_depth_npz_round_trip(tmp_path):
    info = _depth_info([0, 2, 4])
    path = tmp_path / "d.npz"
    np.savez(path, indices=info["indices"].astype(np.int32), depth=info["depth"],
             model_id=np.array(info["model_id"]), normalization=np.array(info["normalization"]),
             frames_npz_sha256=np.array(info["frames_npz_sha256"]))
    got = P.load_depth_npz(path)
    assert list(got["indices"]) == [0, 2, 4]
    assert got["depth"].shape == (3, 8, 10)
    assert got["model_id"] == "fake-model"
    assert got["frames_npz_sha256"] == "x"


def test_load_depth_npz_rejects_missing_fields(tmp_path):
    path = tmp_path / "d.npz"
    np.savez(path, indices=np.array([0, 1]), depth=np.zeros((2, 4, 4), np.float32))
    with pytest.raises(ValueError, match="missing fields"):
        P.load_depth_npz(path)


def test_load_depth_npz_rejects_non_increasing_indices(tmp_path):
    path = tmp_path / "d.npz"
    np.savez(path, indices=np.array([0, 2, 2]), depth=np.zeros((3, 4, 4), np.float32),
             model_id=np.array("m"), normalization=np.array("n"), frames_npz_sha256=np.array("h"))
    with pytest.raises(ValueError, match="strictly increasing"):
        P.load_depth_npz(path)


def test_load_depth_npz_rejects_count_mismatch(tmp_path):
    path = tmp_path / "d.npz"
    np.savez(path, indices=np.array([0, 2, 4]), depth=np.zeros((2, 4, 4), np.float32),
             model_id=np.array("m"), normalization=np.array("n"), frames_npz_sha256=np.array("h"))
    with pytest.raises(ValueError, match="count"):
        P.load_depth_npz(path)


def test_validate_cache_correspondence_hash_mismatch():
    with pytest.raises(ValueError, match="frames_npz_sha256"):
        P.validate_cache_correspondence(_frames(), _meta(), _depth_info([0, 2], frames_sha="other"), "real")


def test_validate_cache_correspondence_index_bounds():
    info = _depth_info([0, 12], frames_sha="h")
    with pytest.raises(ValueError, match="within"):
        P.validate_cache_correspondence(_frames(n=6), _meta(n=6), info, "h")


def test_validate_cache_correspondence_resolution_mismatch():
    info = _depth_info([0, 2], h=9, w=10, frames_sha="h")
    with pytest.raises(ValueError, match="resolution"):
        P.validate_cache_correspondence(_frames(h=8, w=10), _meta(h=8, w=10), info, "h")


def test_validate_cache_correspondence_meta_resolution_disagrees():
    info = _depth_info([0, 2], frames_sha="h")
    meta = _meta(h=7, w=10)
    with pytest.raises(ValueError, match="disagrees"):
        P.validate_cache_correspondence(_frames(h=8, w=10), meta, info, "h")


# --------------------------------------------------------------------------------------
# CLI parsing
# --------------------------------------------------------------------------------------

def test_parser_requires_cache_paths():
    with pytest.raises(SystemExit):
        P.build_parser().parse_args([])


def test_parser_defaults():
    a = P.build_parser().parse_args(["--frames-npz", "f.npz", "--depth-npz", "d.npz"])
    assert a.iters == P.DEFAULT_ITERS
    assert a.downscale == P.DEFAULT_DOWNSCALE
    assert a.flow_dtype == "float32" and a.corr_dtype == "float32"
    assert a.lock == "on"


def test_parse_cycle_thresholds_validates_dedupes_and_preserves_order():
    assert P.parse_cycle_thresholds("0.5,0.25,0.5,1") == [0.5, 0.25, 1.0]
    with pytest.raises(ValueError, match="finite"):
        P.parse_cycle_thresholds("nan")
    with pytest.raises(ValueError, match=">= 0"):
        P.parse_cycle_thresholds("-1")
    with pytest.raises(ValueError, match="at least one"):
        P.parse_cycle_thresholds("")


def test_parser_flow_dtype_choices():
    with pytest.raises(SystemExit):
        P.build_parser().parse_args(["--frames-npz", "f", "--depth-npz", "d", "--flow-dtype", "int8"])


# --------------------------------------------------------------------------------------
# run: validation before runner creation
# --------------------------------------------------------------------------------------

def test_run_pilot_rejects_bad_iters_before_runner(weights_file):
    with pytest.raises(ValueError, match="iters"):
        _run(_frames(), _meta(), _depth_info([0, 2]), iters=0, runner_factory=RaisingRunner,
             tmp_weights=weights_file)


def test_run_pilot_rejects_bad_downscale_before_runner(weights_file):
    with pytest.raises(ValueError, match="downscale"):
        _run(_frames(), _meta(), _depth_info([0, 2]), downscale=0, runner_factory=RaisingRunner,
             tmp_weights=weights_file)


def test_run_pilot_does_not_serialize_machine_specific_lock_path(weights_file):
    result = _run(_frames(), _meta(), _depth_info([0, 2]), tmp_weights=weights_file)
    assert "path" not in result["timing_s"]["lock"]


def test_run_pilot_rejects_bad_thresholds_before_runner(weights_file):
    with pytest.raises(ValueError, match="thresholds"):
        _run(_frames(), _meta(), _depth_info([0, 2]), cycle_thresholds=[-0.5], runner_factory=RaisingRunner,
             tmp_weights=weights_file)


def test_run_pilot_rejects_missing_weights_before_runner(tmp_path):
    with pytest.raises(FileNotFoundError, match="weights"):
        _run(_frames(), _meta(), _depth_info([0, 2]), weights=str(tmp_path / "nope.safetensors"),
             runner_factory=RaisingRunner)


# --------------------------------------------------------------------------------------
# run: adjacent mapping, cycle-valid wiring, determinism
# --------------------------------------------------------------------------------------

def test_run_pilot_maps_adjacent_frames_and_depth_under_lock(weights_file):
    RecordingLock.instances.clear()
    frames = _frames(n=7)
    meta = _meta(n=7)
    indices = [0, 2, 4, 6]
    info = _depth_info(indices)
    res = _run(frames, meta, info, iters=7, tmp_weights=weights_file)
    assert res["pair_count"] == 3
    # adjacent sampled pairs only, and the exact requested iteration count
    assert [(p["source_frame_indices"]) for p in res["pairs"]] == [[0, 2], [2, 4], [4, 6]]
    assert list(res["cycle_thresholds"]) == [0.25, 0.5, 1.0, 2.0]
    # time in source seconds uses fps
    assert res["pairs"][0]["time_s"] == [0.0, 2 / 30.0]
    assert res["pairs"][1]["time_s"] == [2 / 30.0, 4 / 30.0]


def test_run_pilot_uses_sampled_frames_in_order(weights_file):
    calls = {}

    class SpyRunner(FakeRunner):
        def prepare(self, frames, on_step=None):
            calls["frames"] = [np.array(f) for f in frames]
            super().prepare(frames, on_step)

    frames = _frames(n=5)
    info = _depth_info([0, 2, 4])
    _run(frames, _meta(n=5), info, runner_factory=SpyRunner, tmp_weights=weights_file)
    assert len(calls["frames"]) == 3
    for k, idx in enumerate([0, 2, 4]):
        assert np.array_equal(calls["frames"][k], frames[idx])


def test_run_pilot_reference_pairs_are_adjacent_on_sampled_sequence(weights_file):
    made = {}

    class SpyRunner(FakeRunner):
        def reference_pair(self, src, dst, iters):
            made.setdefault("pairs", []).append((int(src), int(dst), int(iters)))
            return super().reference_pair(src, dst, iters)

    _run(_frames(n=5), _meta(n=5), _depth_info([0, 2, 4]), iters=5, runner_factory=SpyRunner,
         tmp_weights=weights_file)
    assert made["pairs"] == [(0, 1, 5), (1, 2, 5)]


def test_run_pilot_lock_covers_mlx_work_and_gates_per_frame_and_pair(weights_file):
    RecordingLock.instances.clear()
    res = _run(_frames(n=5), _meta(n=5), _depth_info([0, 2, 4]), tmp_weights=weights_file)
    lock = RecordingLock.instances[-1]
    # 3 sampled frames prepared + 2 completed pairs = 5 gates, one monotonic sequence
    assert lock.steps == [1, 2, 3, 4, 5]
    assert lock.status()["held"] is False  # released on exit
    assert res["timing_s"]["lock"]["backend"] == "fake"


def test_run_pilot_cycle_valid_frac_is_all_true_for_zero_flow(weights_file):
    class ZeroFlowRunner(FakeRunner):
        def reference_pair(self, src, dst, iters):
            self.calls.append((int(src), int(dst), int(iters)))
            return (np.zeros((self.h, self.w, 2)), np.zeros((self.h, self.w, 2)))

    res = _run(_frames(n=3), _meta(n=3), _depth_info([0, 1, 2]), runner_factory=ZeroFlowRunner,
               tmp_weights=weights_file)
    for p in res["pairs"]:
        assert p["cycle_valid_frac"] == pytest.approx(1.0)


def test_run_pilot_cycle_valid_frac_drops_for_edge_leaving_flow(weights_file):
    res = _run(_frames(n=3), _meta(n=3), _depth_info([0, 1, 2]), tmp_weights=weights_file)
    for p in res["pairs"]:
        # constant +1 flow pushes the last column out of bounds, so some cells are invalid
        assert 0.0 < p["cycle_valid_frac"] < 1.0


def test_run_pilot_is_deterministic(weights_file):
    a = _run(_frames(n=5), _meta(n=5), _depth_info([0, 2, 4]), tmp_weights=weights_file)
    b = _run(_frames(n=5), _meta(n=5), _depth_info([0, 2, 4]), tmp_weights=weights_file)
    # timing fields differ; the audits, summary and geometry must not.
    assert a["pairs"] == b["pairs"] or _strip_flow_s(a["pairs"]) == _strip_flow_s(b["pairs"])
    assert a["summary"] == b["summary"]
    assert a["flow"]["feature_grid"] == b["flow"]["feature_grid"]
    assert a["pair_count"] == b["pair_count"]


def _strip_flow_s(pairs):
    return [{k: v for k, v in p.items() if k != "flow_s"} for p in pairs]


# --------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------

def _fake_pair(tier_med, support_fraction, keys=("all_cycle_valid",)):
    def method(v):
        return {"median_normalized_residual": v, "p95_normalized_residual": v + 1.0}
    return {"audit": {"tiers": {k: {"support_fraction": support_fraction,
                                      "same_index": method(tier_med), "flow_warp": method(tier_med)}
                                for k in keys}}}


def test_summarize_pairs_matches_manual_aggregation():
    pairs = [_fake_pair(0.1, 0.5, keys=("all_cycle_valid", "cycle_le_0.25")),
             _fake_pair(0.3, 0.7, keys=("all_cycle_valid", "cycle_le_0.25"))]
    s = P.summarize_pairs(pairs, [0.25])
    assert set(s) == {"all_cycle_valid", "cycle_le_0.25"}
    block = s["all_cycle_valid"]
    assert block["pairs"] == 2
    assert block["support_fraction_median"] == pytest.approx(0.6)
    assert block["same_index"]["mean_pair_median_norm_resid"] == pytest.approx(0.2)
    assert block["same_index"]["median_pair_median_norm_resid"] == pytest.approx(0.2)
    assert block["same_index"]["usable_pairs"] == 2
    assert block["same_index"]["median_pair_p95_norm_resid"] == pytest.approx(1.2)


def test_summarize_pairs_reports_all_none_when_no_usable_pairs():
    pair = {"audit": {"tiers": {"all_cycle_valid": {
        "support_fraction": 0.0,
        "same_index": {"median_normalized_residual": None, "p95_normalized_residual": None},
        "flow_warp": {"median_normalized_residual": None, "p95_normalized_residual": None}}}}}
    block = P.summarize_pairs([pair], [])["all_cycle_valid"]
    assert block["support_fraction_median"] == 0.0
    assert block["same_index"]["usable_pairs"] == 0
    assert block["same_index"]["mean_pair_median_norm_resid"] is None


# --------------------------------------------------------------------------------------
# end-to-end run() with real cache files (fake runner, no weights)
# --------------------------------------------------------------------------------------

def test_run_end_to_end_writes_atomic_json_and_path_independent_provenance(tmp_path, monkeypatch):
    frames = _frames(n=6, h=8, w=10)
    meta = _meta(n=6, h=8, w=10)
    fpath = _write_frames_npz(tmp_path, frames, meta)
    import bench.manifest as MF
    fsha = MF.sha256_file(fpath)

    info = _depth_info([0, 2, 4], h=8, w=10, frames_sha=fsha)
    dpath = tmp_path / "depth.npz"
    np.savez(dpath, indices=info["indices"].astype(np.int32), depth=info["depth"],
             model_id=np.array(info["model_id"]), normalization=np.array(info["normalization"]),
             frames_npz_sha256=np.array(info["frames_npz_sha256"]))
    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"fake-weights")
    out = tmp_path / "audit.json"

    monkeypatch.setattr(P.R, "RaftRunner", FakeRunner)
    monkeypatch.setattr(P.R, "HeavyLock", RecordingLock)

    args = ["--frames-npz", str(fpath), "--depth-npz", str(dpath), "--weights", str(weights),
            "--out", str(out), "--lock", "off", "--iters", "3"]
    assert P.main(args) == 0
    result = json.loads(out.read_text())
    assert result["status"] == "success"
    assert result["schema"] == P.SCHEMA
    assert result["pair_count"] == 2
    # no absolute local footage / model-cache path leaked
    blob = json.dumps(result)
    assert str(tmp_path) not in blob
    assert "source_path" not in blob
    assert result["source"]["frames_npz_sha256"] == fsha
    assert result["depth"]["model_id"] == "fake-model"
    assert result["flow"]["weights_sha256"] == MF.sha256_file(weights)


def test_run_rejects_depth_from_other_frame_cache(tmp_path):
    frames = _frames(n=6, h=8, w=10)
    meta = _meta(n=6, h=8, w=10)
    fpath = _write_frames_npz(tmp_path, frames, meta)
    info = _depth_info([0, 2, 4], h=8, w=10, frames_sha="deadbeef")
    dpath = tmp_path / "depth.npz"
    np.savez(dpath, indices=info["indices"].astype(np.int32), depth=info["depth"],
             model_id=np.array(info["model_id"]), normalization=np.array(info["normalization"]),
             frames_npz_sha256=np.array(info["frames_npz_sha256"]))
    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"fake")
    args = ["--frames-npz", str(fpath), "--depth-npz", str(dpath), "--weights", str(weights),
            "--out", str(tmp_path / "o.json")]
    assert P.main(args) == 1
