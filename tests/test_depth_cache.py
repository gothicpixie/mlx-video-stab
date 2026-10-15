"""Weight-free tests for the optional relative-depth cache builder.

Everything here runs with injected fakes: no torch, no transformers, no PIL, no model download, no MPS and
no real inference. Coverage: stride/explicit index generation and its validation, percentile normalization
and resize-dimension maths, predictor-driven shapes and finite checks, the exact portable output contract
(keys, dtypes, hash binding, reload via the pilot loader), atomic-write and input-clobber guards, and -- the
point of the optional-dependency design -- that importing the module, parsing the CLI and running the
non-inference paths all work with the heavy stack absent. Warnings are escalated to errors.
"""
import json
import os
import warnings

import numpy as np
import pytest

from bench import depth_cache as D
from bench import depth_temporal_pilot as P


@pytest.fixture(autouse=True)
def no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# --------------------------------------------------------------------------------------
# fakes / builders
# --------------------------------------------------------------------------------------

class FakePredictor:
    """Deterministic stand-in for the real predictor: records frames, returns a fixed-pattern map."""

    def __init__(self, hw=None, *, pattern="ramp", constant=None):
        self.hw = hw
        self.pattern = pattern
        self.constant = constant
        self.calls = []

    def predict(self, rgb):
        arr = np.asarray(rgb)
        self.calls.append(arr.copy())
        h, w = (self.hw if self.hw is not None else (arr.shape[0], arr.shape[1]))
        if self.constant is not None:
            return np.full((h, w), float(self.constant), np.float32)
        # A frame-content-dependent ramp so different sampled frames get distinguishable maps.
        base = float(arr.reshape(-1)[0]) if arr.size else 0.0
        return (np.arange(h * w, dtype=np.float32).reshape(h, w) / float(h * w) + base * 1e-6).astype(np.float32)


def _frames(n=8, h=10, w=14, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(n, h, w, 3), dtype=np.uint8)


def _write_frames_npz(tmp_path, frames, name="frames.npz"):
    meta = {"schema": "mlxstab.flow_prediction_real.frames/1", "count": int(frames.shape[0]),
            "height": int(frames.shape[1]), "width": int(frames.shape[2]), "channels": 3, "fps": 30.0}
    path = tmp_path / name
    np.savez(path, frames=frames, meta=np.array(json.dumps(meta)))
    return path


class _Opts:
    def __init__(self, **kw):
        self.frames_npz = kw.get("frames_npz")
        self.out = kw.get("out")
        self.every = kw.get("every", D.DEFAULT_EVERY)
        self.indices = kw.get("indices", None)
        self.model_id = kw.get("model_id", D.DEFAULT_MODEL_ID)
        self.device = kw.get("device", "cpu")
        self.max_side = kw.get("max_side", D.DEFAULT_MAX_SIDE)
        self.local_files_only = kw.get("local_files_only", True)
        # Weight-free runs default to the lock disabled so they never touch a real shared lock.
        self.lock = kw.get("lock", "off")
        self.lock_chunk = kw.get("lock_chunk", 0)
        self.lock_pause = kw.get("lock_pause", 0.0)


class _FakeLock:
    """Records acquire/release and exposes ``on_step``; ``status`` stays available but test-only."""

    def __init__(self, *a, **kw):
        self.what = kw.get("what", "")
        self.enabled = kw.get("enabled", True)
        self.chunk = int(kw.get("chunk", 0) or 0)
        self.pause_s = float(kw.get("pause_s", 0.0) or 0.0)
        self.held = False
        self.acquires = 0
        self.releases = 0
        self.steps = []

    def acquire(self):
        if self.held:
            return
        self.held = True
        self.acquires += 1

    def release(self):
        if not self.held:
            return
        self.held = False
        self.releases += 1

    def on_step(self, step):
        self.steps.append(int(step))

    def status(self):
        return {"enabled": self.enabled, "chunk": self.chunk, "pause_s": self.pause_s,
                "held": self.held, "acquires": self.acquires, "releases": self.releases}

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


class _LockRecordingPredictor(FakePredictor):
    """Asserts the shared lock is held during every prediction."""

    def __init__(self, lock, **kw):
        super().__init__(**kw)
        self.lock = lock

    def predict(self, rgb):
        assert self.lock.held, "prediction ran without holding the heavy lock"
        return super().predict(rgb)


def _run(tmp_path, frames, predictor=None, **kw):
    path = kw.pop("frames_path", None) or _write_frames_npz(tmp_path, frames)
    out = kw.pop("out", None) or (tmp_path / "depth.npz")
    lock_factory = kw.pop("lock_factory", None)
    opts = _Opts(frames_npz=str(path), out=str(out), **kw)
    predictor = predictor if predictor is not None else FakePredictor()
    result = D.run(opts, predictor=predictor, log=lambda _m: None, lock_factory=lock_factory)
    return result, np.load(out), path, predictor


# --------------------------------------------------------------------------------------
# index generation
# --------------------------------------------------------------------------------------

def test_generate_indices_stride_starts_at_zero_ends_at_last_eligible():
    assert list(D.generate_indices(10, every=4)) == [0, 4, 8]
    assert list(D.generate_indices(9, every=4)) == [0, 4, 8]  # 8 is the last eligible multiple of 4
    assert list(D.generate_indices(8, every=1)) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_generate_indices_stride_larger_than_clip_still_yields_pair():
    idx = D.generate_indices(5, every=100)
    assert idx.size == 2 and int(idx[0]) == 0 and int(idx[-1]) == 4


def test_generate_indices_explicit_validated():
    assert list(D.generate_indices(10, indices=[0, 3, 9])) == [0, 3, 9]
    with pytest.raises(ValueError, match="strictly increasing"):
        D.generate_indices(10, indices=[0, 3, 3])
    with pytest.raises(ValueError, match="strictly increasing"):
        D.generate_indices(10, indices=[3, 1])
    with pytest.raises(ValueError, match="within"):
        D.generate_indices(10, indices=[0, 10])
    with pytest.raises(ValueError, match="at least 2"):
        D.generate_indices(10, indices=[0])
    with pytest.raises(ValueError, match="at least 2"):
        D.generate_indices(1, every=1)
    with pytest.raises(ValueError, match="--every must be >= 1"):
        D.generate_indices(10, every=0)


def test_parse_indices():
    assert D.parse_indices("0,4,8") == [0, 4, 8]
    assert D.parse_indices(None) is None
    with pytest.raises(ValueError, match="empty"):
        D.parse_indices("")
    with pytest.raises(ValueError, match="empty token"):
        D.parse_indices("0,,4")


# --------------------------------------------------------------------------------------
# normalization / preprocessing maths
# --------------------------------------------------------------------------------------

def test_normalize_percentile_clips_and_is_relative():
    d = np.linspace(0.0, 100.0, 101, dtype=np.float32)
    out = D.normalize_percentile(d)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert out[1] == 0.0 and out[99] == 1.0  # the 1st/99th percentile band maps to the endpoints
    assert out[0] == 0.0 and out[100] == 1.0  # tails are clipped


def test_normalize_percentile_degenerate_and_nonfinite():
    assert np.array_equal(D.normalize_percentile(np.zeros((4, 4), np.float32)), np.zeros((4, 4), np.float32))
    bad = np.array([[np.nan, np.inf], [1.0, 2.0]], np.float32)
    out = D.normalize_percentile(bad)
    assert np.all(np.isfinite(out))
    allbad = np.full((3, 3), np.nan, np.float32)
    assert np.array_equal(D.normalize_percentile(allbad), np.zeros((3, 3), np.float32))


def test_target_resize_dims_multiple_of_14_and_minimum():
    nh, nw = D.target_resize_dims(1080, 1920, 1036)
    assert nh % 14 == 0 and nw % 14 == 0
    assert max(nh, nw) >= 1036 - D.PATCH_MULTIPLE  # close to max_side after rounding
    # A very small max_side still floors each side at one patch (never 0, never < 14).
    assert D.target_resize_dims(10, 5000, 14) == (14, 14)
    assert all(v >= 14 and v % 14 == 0 for v in D.target_resize_dims(1, 1, 1036))
    with pytest.raises(ValueError, match="positive"):
        D.target_resize_dims(0, 10, 1036)
    with pytest.raises(ValueError, match="--max-side"):
        D.target_resize_dims(10, 10, 0)


def test_imagenet_normalize_mean_zero_for_mean_pixel():
    mean_px = np.array(D.IMAGENET_MEAN, np.float32).reshape(1, 1, 3)
    out = D.imagenet_normalize(mean_px)
    assert np.allclose(out, 0.0, atol=1e-6)


# --------------------------------------------------------------------------------------
# inference driver shapes / finite checks
# --------------------------------------------------------------------------------------

def test_build_depth_array_checks_shape_and_finiteness():
    frames = _frames(n=4, h=10, w=14)
    idx = np.array([0, 2], np.int64)
    depth = D.build_depth_array(frames, idx, FakePredictor(), log=lambda _m: None)
    assert depth.shape == (2, 10, 14) and depth.dtype == np.float32
    assert np.all(np.isfinite(depth))

    with pytest.raises(ValueError, match="expected"):
        D.build_depth_array(frames, idx, FakePredictor(hw=(3, 3)), log=lambda _m: None)


def test_build_depth_array_rejects_nonfinite():
    class BadPredictor:
        def predict(self, rgb):
            return np.array([[np.nan, 1.0], [1.0, 1.0]], np.float32)

    frames = _frames(n=2, h=2, w=2)
    with pytest.raises(ValueError, match="non-finite"):
        D.build_depth_array(frames, np.array([0, 1]), BadPredictor(), log=lambda _m: None)


def test_build_depth_array_is_one_frame_at_a_time():
    frames = _frames(n=6, h=4, w=4)
    pred = FakePredictor()
    D.build_depth_array(frames, np.array([0, 2, 4]), pred, log=lambda _m: None)
    assert len(pred.calls) == 3  # one predict() per sampled frame, not batched


# --------------------------------------------------------------------------------------
# end-to-end: exact portable output, reload, provenance
# --------------------------------------------------------------------------------------

def test_run_writes_exact_depth_npz_contract(tmp_path):
    frames = _frames(n=10, h=8, w=12, seed=1)
    result, z, frames_path, _ = _run(tmp_path, frames, every=4)

    assert set(z.files) == set(D.DEPTH_KEYS)
    assert z["indices"].dtype == np.int32
    assert list(z["indices"]) == [0, 4, 8]
    assert z["depth"].dtype == np.float32 and z["depth"].shape == (3, 8, 12)
    assert np.all(np.isfinite(z["depth"])) and z["depth"].min() >= 0.0 and z["depth"].max() <= 1.0
    assert str(z["model_id"]) == D.DEFAULT_MODEL_ID
    assert str(z["normalization"]) == D.NORMALIZATION
    assert str(z["frames_npz_sha256"]) == result["frames_npz_sha256"]

    # Bound to the *content* hash of the input frame cache.
    from bench import manifest as MF
    assert str(z["frames_npz_sha256"]) == MF.sha256_file(frames_path)


def test_output_reloads_with_pilot_loader_and_matches_frames(tmp_path):
    frames = _frames(n=9, h=6, w=14, seed=2)
    _result, _z, frames_path, _ = _run(tmp_path, frames, every=4)
    depth_path = tmp_path / "depth.npz"

    info = P.load_depth_npz(depth_path)
    got_frames, meta = P.load_frames_npz(frames_path)
    from bench import manifest as MF
    P.validate_cache_correspondence(got_frames, meta, info, MF.sha256_file(frames_path))
    assert list(info["indices"]) == [0, 4, 8]


def test_output_is_path_independent(tmp_path):
    frames = _frames(n=6, h=4, w=4, seed=3)
    _result, z, _frames_path, _ = _run(tmp_path, frames, every=4)
    # No scalar field may contain a path or the frames file name.
    for key in D.DEPTH_KEYS:
        val = z[key]
        if val.dtype.kind in ("U", "S", "O"):
            text = str(val)
            assert tmp_path.name not in text
            assert "frames.npz" not in text
            assert not os.path.isabs(text)


def test_explicit_indices_override_every(tmp_path):
    frames = _frames(n=10, h=4, w=4, seed=4)
    _result, z, _p, _ = _run(tmp_path, frames, every=2, indices="1,5,9")
    assert list(z["indices"]) == [1, 5, 9]


def test_run_refuses_to_clobber_input(tmp_path):
    frames = _frames(n=6, h=4, w=4)
    path = _write_frames_npz(tmp_path, frames)
    opts = _Opts(frames_npz=str(path), out=str(path))
    with pytest.raises(ValueError, match="must not overwrite"):
        D.run(opts, predictor=FakePredictor(), log=lambda _m: None)


def test_run_atomic_no_temp_left_behind(tmp_path):
    frames = _frames(n=6, h=4, w=4)
    _run(tmp_path, frames, every=4)
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp" in p]
    assert leftovers == []


def test_run_missing_frames_file(tmp_path):
    opts = _Opts(frames_npz=str(tmp_path / "nope.npz"), out=str(tmp_path / "d.npz"))
    with pytest.raises(FileNotFoundError):
        D.run(opts, predictor=FakePredictor(), log=lambda _m: None)


def test_run_passes_predictor_output_through_untouched(tmp_path):
    # The injectable predictor owns normalization: run() stores whatever it returns (shape/finite checked).
    frames = _frames(n=6, h=6, w=6, seed=5)
    _result, z, _p, _ = _run(tmp_path, frames, predictor=FakePredictor(constant=3.0), every=2)
    assert np.all(z["depth"] == 3.0)

    # The real predictor's own percentile normalization turns the same constant map into zeros.
    assert np.all(D.normalize_percentile(np.full((6, 6), 3.0, np.float32)) == 0.0)


# --------------------------------------------------------------------------------------
# shared heavy lock coordination
# --------------------------------------------------------------------------------------

def test_cli_defaults_lock_on():
    a = D.build_parser().parse_args(["--frames-npz", "f.npz", "--out", "d.npz"])
    assert a.lock == "on"
    assert D.resolve_lock_enabled(a.lock) is True
    assert D.resolve_lock_enabled("off") is False
    with pytest.raises(ValueError, match="'on' or 'off'"):
        D.resolve_lock_enabled("auto")


def test_lock_held_during_construction_and_inference(tmp_path, monkeypatch):
    frames = _frames(n=8, h=6, w=6, seed=7)
    lock = _FakeLock(enabled=True)

    class _RecordingPredictor(FakePredictor):
        def __init__(self, *a, **kw):
            super().__init__()
            # run() must have acquired the lock before constructing the real predictor.
            assert lock.held, "predictor constructed without holding the heavy lock"
            self.constructed_under_lock = True

        def predict(self, rgb):
            assert lock.held, "prediction ran without holding the heavy lock"
            return super().predict(rgb)

    monkeypatch.setattr(D, "DepthAnythingPredictor", _RecordingPredictor)

    def factory(**kw):
        lock.enabled = kw["enabled"]
        lock.chunk = kw["chunk"]
        lock.pause_s = kw["pause_s"]
        return lock

    path = _write_frames_npz(tmp_path, frames)
    out = tmp_path / "depth.npz"
    opts = _Opts(frames_npz=str(path), out=str(out), every=2, lock="on")
    result = D.run(opts, predictor=None, log=lambda _m: None, lock_factory=factory)
    assert result["status"] == "success"
    assert lock.acquires == 1 and lock.releases == 1
    assert lock.held is False  # released after the run


def test_lock_gate_receives_one_step_per_completed_frame(tmp_path):
    frames = _frames(n=10, h=6, w=6, seed=8)
    lock = _FakeLock(enabled=True, chunk=2)
    pred = _LockRecordingPredictor(lock)

    opts = _Opts(frames_npz=str(_write_frames_npz(tmp_path, frames)), out=str(tmp_path / "d.npz"),
                 every=2, lock="on", lock_chunk=2)

    def factory(**kw):
        lock.enabled = kw["enabled"]
        lock.chunk = kw["chunk"]
        lock.pause_s = kw["pause_s"]
        return lock

    D.run(opts, predictor=pred, log=lambda _m: None, lock_factory=factory)
    # indices 0,2,4,6,8 -> 5 completed frames -> steps 1..5, monotonically increasing
    assert lock.steps == [1, 2, 3, 4, 5]
    assert lock.acquires == 1 and lock.held is False


def test_lock_no_gate_without_chunking(tmp_path):
    frames = _frames(n=6, h=4, w=4, seed=9)
    lock = _FakeLock(enabled=True, chunk=0)
    D.run(_Opts(frames_npz=str(_write_frames_npz(tmp_path, frames)), out=str(tmp_path / "d.npz"),
               every=2, lock="on", lock_chunk=0),
          predictor=FakePredictor(), log=lambda _m: None, lock_factory=lambda **kw: lock)
    assert lock.steps == []  # no chunk -> no gate ticks, but the lock is still held and released
    assert lock.held is False


def test_lock_released_on_predictor_failure(tmp_path):
    frames = _frames(n=6, h=4, w=4, seed=10)
    lock = _FakeLock(enabled=True, chunk=2)

    class Boom:
        def predict(self, rgb):
            assert lock.held, "prediction ran without holding the heavy lock"
            raise RuntimeError("actor down")

    opts = _Opts(frames_npz=str(_write_frames_npz(tmp_path, frames)), out=str(tmp_path / "d.npz"),
                 every=2, lock="on", lock_chunk=2)
    with pytest.raises(RuntimeError, match="actor down"):
        D.run(opts, predictor=Boom(), log=lambda _m: None, lock_factory=lambda **kw: lock)
    assert lock.held is False and lock.releases >= 1  # always released, even on failure


def test_invalid_lock_options_fail_before_predictor_construction(tmp_path):
    frames = _frames(n=6, h=4, w=4, seed=11)
    path = _write_frames_npz(tmp_path, frames)
    built = []

    def factory(**kw):
        built.append(kw)
        return _FakeLock(**kw)

    # A predictor factory that must never run for the invalid configs.
    class Sentinel:
        def predict(self, rgb):  # pragma: no cover - must not be called
            raise AssertionError("predictor ran despite invalid lock options")

    for kw, msg in (({"lock_chunk": -1}, "lock-chunk"), ({"lock_pause": -1.0}, "lock-pause"),
                    ({"lock": "auto"}, "'on' or 'off'")):
        opts = _Opts(frames_npz=str(path), out=str(tmp_path / "d.npz"), **kw)
        with pytest.raises(ValueError, match=msg):
            D.run(opts, predictor=Sentinel(), log=lambda _m: None, lock_factory=factory)
    assert built == []  # rejected before the lock (and therefore the predictor) was constructed


def test_build_depth_array_on_step_is_one_based_and_monotonic():
    frames = _frames(n=6, h=4, w=4)
    steps = []
    D.build_depth_array(frames, np.array([0, 2, 4]), FakePredictor(), log=lambda _m: None,
                        on_step=steps.append)
    assert steps == [1, 2, 3]


def test_build_depth_array_failed_sample_does_not_tick():
    class FailSecond(FakePredictor):
        def predict(self, rgb):
            if len(self.calls) == 1:
                raise RuntimeError("nope")
            return super().predict(rgb)

    steps = []
    with pytest.raises(RuntimeError, match="nope"):
        D.build_depth_array(_frames(n=4, h=4, w=4), np.array([0, 1, 2]), FailSecond(),
                            log=lambda _m: None, on_step=steps.append)
    assert steps == [1]  # only the first completed sample ticked


# --------------------------------------------------------------------------------------
# optional-dependency absence and CLI
# --------------------------------------------------------------------------------------

def test_module_importable_and_cli_parses():
    assert "depth_cache" in D.__name__
    a = D.build_parser().parse_args(["--frames-npz", "f.npz", "--out", "d.npz"])
    assert a.every == D.DEFAULT_EVERY and a.device == "auto" and a.max_side == D.DEFAULT_MAX_SIDE
    assert a.local_files_only is False
    assert a.lock == "on" and a.lock_chunk == 0 and a.lock_pause == 0.0
    a2 = D.build_parser().parse_args(
        ["--frames-npz", "f.npz", "--out", "d.npz", "--indices", "0,4", "--device", "cpu",
         "--local-files-only", "--max-side", "518"])
    assert D.parse_indices(a2.indices) == [0, 4] and a2.device == "cpu" and a2.local_files_only


def test_non_inference_path_works_without_heavy_deps(tmp_path, monkeypatch):
    """Deleting torch/transformers/PIL from the import namespace must not break the non-inference path."""
    import sys
    for mod in ("torch", "transformers", "PIL"):
        monkeypatch.setitem(sys.modules, mod, None)

    frames = _frames(n=8, h=4, w=4, seed=6)
    # Index maths, normalization and the injected-predictor run must all work regardless.
    assert list(D.generate_indices(8, every=3)) == [0, 3, 6]
    result, z, _p, _ = _run(tmp_path, frames, every=4)
    assert result["status"] == "success" and list(z["indices"]) == [0, 4]


def test_real_predictor_reports_missing_deps(monkeypatch):
    import sys
    for mod in ("torch", "transformers", "PIL"):
        monkeypatch.setitem(sys.modules, mod, None)
    with pytest.raises(RuntimeError, match="requires torch, transformers and pillow"):
        D._require_heavy_deps()


def test_main_dry_run_and_failure_paths(tmp_path, capsys):
    frames = _frames(n=6, h=4, w=4)
    path = _write_frames_npz(tmp_path, frames)
    code = D.main(["--frames-npz", str(path), "--out", str(tmp_path / "d.npz"), "--every", "4",
                   "--indices", "0,4", "--dry-run"])
    assert code == 0
    cfg = json.loads(capsys.readouterr().out)
    assert cfg["indices"] == [0, 4] and cfg["every"] == 4

    # The failure path serializes an error object rather than raising.
    code = D.main(["--frames-npz", str(tmp_path / "missing.npz"), "--out", str(tmp_path / "d.npz")])
    assert code == 1
    err = json.loads(capsys.readouterr().out)
    assert err["status"] == "failure" and err["error"]["type"] == "FileNotFoundError"
