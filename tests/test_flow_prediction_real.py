"""Focused, weight-free tests for the E04 real-footage flow-initialization harness.

They pin the deterministic orchestration, the candidate mapping (transport direction, identity-map
equivalence, fallback and insufficient history), the frame-cache request guard and JSON/manifest safety.
No RAFT weights, no private footage and no timing assertions: the model is replaced by a deterministic fake so
the same checks run offline. The real end-to-end run is exercised separately under the shared heavy lock.
"""
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from bench import flow_prediction as FP
from bench import flow_prediction_real as R
from bench import manifest as MF


# --------------------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------------------

class FakeScanStepper:
    """Deterministic stand-in for the real one-iteration-at-a-time scan stepper.

    Each ``step()`` adds ``0.5`` (to both components), so exactly ``iters`` steps reproduce
    :meth:`FakeRunner.refine_closure`; this is what lets a test assert the fixed-history invariant directly.
    """

    def __init__(self, init, delta=0.5):
        self.cur = np.asarray(init, np.float64).copy()
        self.delta = float(delta)

    def step(self):
        self.cur = self.cur + self.delta
        return self.cur.copy()


class FakeRunner:
    """Deterministic stand-in for :class:`RaftRunner` (no MLX, no weights)."""

    def __init__(self, h=4, w=5, downscale=2):
        self.h, self.w = h, w
        self.pad = (0, 0, 0, 0)
        self.downscale = downscale
        self.prep_s = 0.0
        self.prepared = None

    def prepare(self, frames, on_step=None):
        self.prepared = len(frames)
        if on_step is not None:
            for _ in frames:
                on_step()

    def reference_pair(self, src, dst, iters):
        f = np.full((self.h, self.w, 2), float(dst - src), np.float64)
        b = np.full((self.h, self.w, 2), -float(dst - src), np.float64)
        return f, b

    def refine_closure(self, order, iters):
        def refine(init, target):
            return np.asarray(init, np.float64) + 0.5 * float(iters)
        return refine

    def scan_closure(self, order):
        def make_stepper(init, target):
            return FakeScanStepper(init)
        return make_stepper

    def bidir_scan_closure(self):
        def make_stepper(target):
            z = np.zeros((self.h * self.w, 2), np.float64)
            return FakeBidirStepper(z, z)
        return make_stepper


class FakeBidirStepper:
    """Paired stand-in for the real batch-2 cycle scan stepper: advances both directions together."""

    def __init__(self, init_f, init_b, delta=0.5):
        self.f = np.asarray(init_f, np.float64).copy()
        self.b = np.asarray(init_b, np.float64).copy()
        self.delta = float(delta)
        self.iters = 0

    def step(self):
        self.iters += 1
        self.f = self.f + self.delta
        self.b = self.b - self.delta
        return self.f.copy(), self.b.copy()


def make_refs(n, H=4, W=5, fwd=(2.0, 0.0), bwd=(-2.0, 0.0)):
    rf = np.zeros((n, H, W, 2), np.float64); rf[..., 0] = fwd[0]; rf[..., 1] = fwd[1]
    rb = np.zeros((n, H, W, 2), np.float64); rb[..., 0] = bwd[0]; rb[..., 1] = bwd[1]
    return rf, rb


def strip_times(rows):
    return [{k: v for k, v in r.items() if not k.endswith("_s")} for r in rows]


class _FakeClock:
    """Deterministic perf_counter stand-in so the timing test never depends on wall-clock jitter."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class _WaitingLock:
    """Fake heavy lock whose ``acquire`` blocks for a fixed time before returning."""

    def __init__(self, clock, wait_s):
        self.clock = clock
        self.wait_s = wait_s
        self.acquired = False

    def acquire(self):
        self.acquired = True
        self.clock.advance(self.wait_s)      # the wait elapses *inside* acquire, as in HeavyLock.acquire


# --------------------------------------------------------------------------------------
# frame-load timing excludes the shared-lock wait
# --------------------------------------------------------------------------------------

def test_frames_load_excludes_lock_acquire_wait():
    """``timing_s['frames_load']`` must measure only the load, not the ``lock.acquire`` wait."""
    clock = _FakeClock()
    lock = _WaitingLock(clock, wait_s=100.0)
    payload = (["frame0", "frame1"], 30.0, {"path_type": "synthetic"})

    def load():
        clock.advance(2.5)                    # decode or cache restore, not the lock wait
        return payload

    result, load_s = R.load_under_lock(lock, load, clock=clock)
    assert lock.acquired
    assert result == payload
    assert load_s == 2.5                      # the 100 s acquire wait is excluded


# --------------------------------------------------------------------------------------
# traversal construction and candidate mapping
# --------------------------------------------------------------------------------------

def test_traversal_order_reverses_for_backward():
    assert R.traversal_order(4, "forward") == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert R.traversal_order(4, "backward") == [(4, 3), (3, 2), (2, 1), (1, 0)]


def test_build_traversal_forward_uses_fwd_ref_backward_uses_bwd_ref():
    n = 3
    rf, rb = make_refs(n, fwd=(2.0, 0.0), bwd=(-2.0, 0.0))
    f_pairs, f_fields = R.build_traversal(rf, rb, "forward")
    b_pairs, b_fields = R.build_traversal(rf, rb, "backward")
    assert len(f_pairs) == len(b_pairs) == n
    np.testing.assert_array_equal(f_fields, rf)
    # backward traversal visits original pairs in reverse and carries the backward reference
    for t in range(n):
        np.testing.assert_array_equal(b_fields[t], rb[n - 1 - t])
    # a constant map makes the pair callables agree with a plain translation
    p = np.array([[2.0, 2.0]])
    np.testing.assert_allclose(f_pairs[0].forward(p), p + [2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(f_pairs[0].inverse(p), p + [-2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(b_pairs[0].forward(p), p + [-2.0, 0.0], atol=1e-12)   # original backward
    np.testing.assert_allclose(b_pairs[0].inverse(p), p + [2.0, 0.0], atol=1e-12)    # original forward


def test_transport_uses_approximate_inverse_map_not_relabel():
    """With a spatially varying field and a non-zero map, transport shifts the sample by the inverse map."""
    H, W = 6, 7
    bwd = np.zeros((H, W, 2)); bwd[..., 0] = 2.0          # frame k+1 -> k: +2 in x
    fwd = np.zeros((H, W, 2))
    pair = R.flow_pair("p", fwd, bwd, W, H)
    grid = FP.source_grid(W, H)
    field = np.zeros((H, W, 2)); field[..., 0] = grid[:, 0].reshape(H, W)
    vals, valid = FP.transport(field, np.ones((H, W), bool), [pair], 0, 1, grid, W, H)
    # interior points only (sampling clamps at the border)
    interior = ((grid[:, 0] >= 1) & (grid[:, 0] <= W - 3) & (grid[:, 1] >= 1) & (grid[:, 1] <= H - 2))
    np.testing.assert_allclose(vals[interior, 0], grid[interior, 0] + 2.0, atol=1e-9)
    # same-grid relabel would leave the field unchanged
    assert np.all(np.abs(vals[interior, 0] - grid[interior, 0]) > 1.0)
    assert valid[interior].all()


def test_identity_maps_make_transported_equal_same_grid():
    """Zero flow maps are the identity, so transported must reproduce same-grid reuse exactly."""
    ref_fwd, ref_bwd = make_refs(4, fwd=(0.0, 0.0), bwd=(0.0, 0.0))
    pairs, ref_fields = R.build_traversal(ref_fwd, ref_bwd, "forward")

    def record():
        seen = {}
        def refine(init, target):
            seen[target] = init.copy()
            return init + 1.0
        return seen, refine

    tr_seen, tr_refine = record()
    R.replay(pairs, ref_fields, "transported", tr_refine)
    sg_seen, sg_refine = record()
    R.replay(pairs, ref_fields, "same_grid", sg_refine)
    for target in tr_seen:
        np.testing.assert_allclose(tr_seen[target], sg_seen[target], atol=1e-12)


def test_fallback_and_insufficient_history_reported():
    n = 4
    rf, rb = make_refs(n)
    rb[:, :, :, 0] = 1000.0                       # inverse map throws every sample out of frame
    pairs, ref_fields = R.build_traversal(rf, rb, "forward")
    rows = R.replay(pairs, ref_fields, "transported", lambda init, t: init)
    assert rows[0]["reason"] == "insufficient_history"
    assert rows[1]["fallback"] and rows[1]["reason"] in ("zero_coverage", "low_coverage")
    assert rows[1]["coverage"] < FP.MIN_COVERAGE

    cv = R.replay(pairs, ref_fields, "constant_velocity", lambda init, t: init)
    assert cv[0]["reason"] == "insufficient_history"
    assert cv[1]["reason"] == "insufficient_history"
    ca = R.replay(pairs, ref_fields, "constant_acceleration", lambda init, t: init)
    assert ca[2]["reason"] == "insufficient_history"


def test_replay_rejects_unknown_method():
    pairs, ref_fields = R.build_traversal(*make_refs(2), "forward")
    with pytest.raises(ValueError):
        R.replay(pairs, ref_fields, "teleport", lambda init, t: init)


# --------------------------------------------------------------------------------------
# deterministic orchestration
# --------------------------------------------------------------------------------------

def test_replay_is_deterministic():
    pairs, ref_fields = R.build_traversal(*make_refs(5), "forward")
    refine = lambda init, t: init + np.array([0.25, -0.5])
    a = R.replay(pairs, ref_fields, "constant_acceleration", refine)
    b = R.replay(pairs, ref_fields, "constant_acceleration", refine)
    assert strip_times(a) == strip_times(b)


def test_run_benchmark_orchestration_is_deterministic_and_json_safe():
    frames = [np.zeros((64, 96, 3), np.uint8) for _ in range(5)]
    a = R.run_benchmark(FakeRunner(), frames, methods=R.METHODS, directions=R.DIRECTIONS, log=lambda m: None)
    b = R.run_benchmark(FakeRunner(), frames, methods=R.METHODS, directions=R.DIRECTIONS, log=lambda m: None)
    assert a["n_pairs"] == 4
    for direction in R.DIRECTIONS:
        for method in R.METHODS:
            assert len(a["results"][direction][method]["per_pair"]) == 4
            s = a["results"][direction][method]["summary"]
            assert s["pairs"] == 4 and s["iters"] == R.DEFAULT_ITERS
    # exact structure/values (ignoring wall times) reproduce
    def clean(res):
        return {d: {m: strip_times(p["per_pair"]) for m, p in res["results"][d].items()} for d in res["results"]}
    assert clean(a) == clean(b)
    payload = MF.dumps_json(a)
    assert "NaN" not in payload and "Infinity" not in payload
    json.loads(payload)


def test_run_benchmark_summary_flags_and_units():
    frames = [np.zeros((64, 96, 3), np.uint8) for _ in range(4)]
    res = R.run_benchmark(FakeRunner(downscale=2), frames, methods=("cold",), directions=("forward",), log=lambda m: None)
    assert res["units_to_source_px"] == 16.0
    s = res["results"]["forward"]["cold"]["summary"]
    assert s["iters"] == R.DEFAULT_ITERS
    assert s["skipped_no_history"] == 0          # cold always has a valid (zero) candidate
    assert s["fallbacks"] == 0
    assert s["coverage_mean"] == pytest.approx(1.0)


def test_run_benchmark_rejects_bad_inputs():
    frames = [np.zeros((8, 8, 3), np.uint8) for _ in range(3)]
    with pytest.raises(ValueError):
        R.run_benchmark(FakeRunner(), frames, methods=("nope",), log=lambda m: None)
    with pytest.raises(ValueError):
        R.run_benchmark(FakeRunner(), frames, directions=("sideways",), log=lambda m: None)
    with pytest.raises(ValueError):
        R.run_benchmark(FakeRunner(), [np.zeros((8, 8, 3), np.uint8)], log=lambda m: None)


def test_max_pairs_caps_the_window():
    frames = [np.zeros((64, 96, 3), np.uint8) for _ in range(9)]
    res = R.run_benchmark(FakeRunner(), frames, methods=("cold",), directions=("forward",), max_pairs=3, log=lambda m: None)
    assert res["n_pairs"] == 3
    assert res["results"]["forward"]["cold"]["summary"]["pairs"] == 3


def test_common_targets_intersect_history_requirements():
    """Constant acceleration needs three prior pairs, so the common pair set starts at target 3."""
    pairs, ref_fields = R.build_traversal(*make_refs(6), "forward")
    results = {m: {"per_pair": R.replay(pairs, ref_fields, m, lambda init, t: init)} for m in R.METHODS}
    common = R.common_targets(results)
    assert common == [3, 4, 5]
    for method in R.METHODS:
        s = R.summarize(results[method]["per_pair"], R.DEFAULT_ITERS, 8.0, common)
        assert s["usable_pairs"] == 3 and s["targets"] == [3, 4, 5]


def test_run_benchmark_reports_common_target_summary():
    frames = [np.zeros((64, 96, 3), np.uint8) for _ in range(6)]
    res = R.run_benchmark(FakeRunner(), frames, methods=R.METHODS, directions=("forward",), log=lambda m: None)
    assert res["common_targets"]["forward"] == [3, 4]
    for method in R.METHODS:
        assert "summary_common" in res["results"]["forward"][method]
        assert res["results"]["forward"][method]["summary_common"]["usable_pairs"] == 2


# --------------------------------------------------------------------------------------
# shared-lock chunking: one monotonic step gate across every phase
# --------------------------------------------------------------------------------------

class _RecordingGate:
    """Sink that records the step numbers the harness hands to ``on_step``."""

    def __init__(self):
        self.steps = []

    def __call__(self, step):
        self.steps.append(int(step))


class _RunRaftRunner(FakeRunner):
    """Stand-in for ``RaftRunner(weights, downscale, flow_dtype, corr_dtype)`` in a full ``run()``."""

    def __init__(self, weights, downscale=1, flow_dtype="float32", corr_dtype="float32"):
        super().__init__(downscale=downscale)


class _RecordingLock:
    """Fake ``HeavyLock`` capturing construction, steps and chunk/pause without any real lock."""

    instances = []

    def __init__(self, what="", enabled=True, chunk=0, poll=10, pause_s=0.0, sleep=None):
        self.what = what
        self.enabled = enabled
        self.chunk = int(chunk)
        self.pause_s = float(pause_s)
        self.steps = []
        self.acquires = 0
        self.releases = 0
        _RecordingLock.instances.append(self)

    def acquire(self):
        self.acquires += 1

    def release(self):
        self.releases += 1

    def on_step(self, step):
        self.steps.append(int(step))

    def status(self):
        return {"backend": "fake", "chunk": self.chunk, "pause_s": self.pause_s,
                "enabled": self.enabled, "steps": len(self.steps)}


def _fake_frames(n=4, h=64, w=96):
    return [np.zeros((h, w, 3), np.uint8) for _ in range(n)]


def test_replay_and_references_tick_once_per_completed_pair():
    pairs, ref_fields = R.build_traversal(*make_refs(4), "forward")
    replay_calls = []
    rows = R.replay(pairs, ref_fields, "transported", lambda init, t: init,
                    on_step=lambda: replay_calls.append(1))
    assert len(rows) == 4 and len(replay_calls) == 4
    ref_calls = []
    R.compute_references(FakeRunner(), 3, 24, on_step=lambda: ref_calls.append(1))
    assert len(ref_calls) == 3


def test_step_gate_is_one_monotonic_sequence_across_all_phases():
    gate = _RecordingGate()
    res = R.run_benchmark(FakeRunner(), _fake_frames(5), methods=R.METHODS, directions=R.DIRECTIONS,
                          log=lambda m: None, on_step=gate)
    n_pairs = 4
    expected = 5 + n_pairs + len(R.METHODS) * len(R.DIRECTIONS) * n_pairs
    assert gate.steps == list(range(1, expected + 1))     # contiguous, monotonic, exactly once each
    assert res["steps"] == expected


def test_step_counter_counts_without_a_callback():
    res = R.run_benchmark(FakeRunner(), _fake_frames(3), methods=("cold",), directions=("forward",),
                          log=lambda m: None)
    assert res["steps"] == 3 + 2 + 2                       # frames + references + 1x1 method/pairs


def test_parser_lock_chunk_and_pause_defaults_and_values():
    a = R.build_parser().parse_args([])
    assert a.lock_chunk == 0 and a.lock_pause == 0.0
    b = R.build_parser().parse_args(["--lock-chunk", "30", "--lock-pause", "12"])
    assert b.lock_chunk == 30 and b.lock_pause == 12.0


def test_lock_enabled_resolution():
    assert R.lock_enabled("synthetic", "auto") is False
    assert R.lock_enabled("clip.MTS", "auto") is True
    assert R.lock_enabled("synthetic", "on") is True
    assert R.lock_enabled("clip.MTS", "off") is False
    with pytest.raises(ValueError):
        R.lock_enabled("clip.MTS", "sometimes")


def test_run_wires_and_persists_lock_chunking(monkeypatch, tmp_path):
    _RecordingLock.instances.clear()
    monkeypatch.setattr(R, "HeavyLock", _RecordingLock)
    monkeypatch.setattr(R, "RaftRunner", _RunRaftRunner)
    monkeypatch.setattr(R, "load_or_decode", lambda *a, **k: (
        _fake_frames(), 30.0, {"path_type": "synthetic", "sha256": None, "window": None,
                               "probe": None, "synthetic": None}))
    out = tmp_path / "run-chunk"
    rc = R.main(["--source", "synthetic", "--lock", "on", "--lock-chunk", "30", "--lock-pause", "12",
                 "--out-dir", str(out)])
    assert rc == 0
    man = json.loads((out / "manifest.json").read_text())
    assert man["status"] == "success"
    assert man["config"]["lock_chunk"] == 30 and man["config"]["lock_pause_s"] == 12.0
    assert man["lock"]["chunk"] == 30 and man["lock"]["pause_s"] == 12.0
    lock = _RecordingLock.instances[-1]
    assert lock.enabled is True and lock.chunk == 30 and lock.pause_s == 12.0
    assert lock.steps and lock.steps == list(range(1, len(lock.steps) + 1))
    assert man["steps"] == len(lock.steps)


def test_run_does_not_gate_synthetic_by_default(monkeypatch, tmp_path):
    _RecordingLock.instances.clear()
    monkeypatch.setattr(R, "HeavyLock", _RecordingLock)
    monkeypatch.setattr(R, "RaftRunner", _RunRaftRunner)
    monkeypatch.setattr(R, "load_or_decode", lambda *a, **k: (
        _fake_frames(3), 30.0, {"path_type": "synthetic"}))
    out = tmp_path / "run-default"
    rc = R.main(["--source", "synthetic", "--out-dir", str(out)])
    assert rc == 0
    lock = _RecordingLock.instances[-1]
    assert lock.enabled is False           # auto -> off for synthetic
    assert lock.steps == []                # no chunk configured -> the gate is never invoked
    man = json.loads((out / "manifest.json").read_text())
    assert man["config"]["lock_chunk"] == 0 and man["config"]["lock_pause_s"] == 0.0


# --------------------------------------------------------------------------------------
# cache and manifest/schema safety
# --------------------------------------------------------------------------------------

def test_frames_sha256_is_content_sensitive():
    a = [np.zeros((4, 4, 3), np.uint8)]
    b = [np.zeros((4, 4, 3), np.uint8)]; b[0][0, 0, 0] = 1
    assert R.frames_sha256(a) == R.frames_sha256([a[0].copy()])
    assert R.frames_sha256(a) != R.frames_sha256(b)


def test_synthetic_frame_cache_round_trip_and_request_guard(tmp_path):
    synth = {"frames": 4, "w": 32, "h": 24, "fps": 30.0, "seed": 0}
    cache = tmp_path / "frames.npz"
    frames, fps, info = R.load_or_decode("synthetic", None, None, None, synth, str(cache))
    assert info["path_type"] == "synthetic" and cache.exists()
    again, fps2, info2 = R.load_or_decode("synthetic", None, None, None, synth, str(cache))
    assert info2.get("cache_path") == str(cache)
    assert R.frames_sha256(frames) == R.frames_sha256(again)
    assert fps == fps2
    # a different request (different synth parameters) must refuse the cache rather than mix frames
    with pytest.raises(ValueError):
        R.load_or_decode("synthetic", None, None, None, dict(synth, frames=5), str(cache))


def test_frame_cache_detects_content_tampering(tmp_path):
    synth = {"frames": 4, "w": 32, "h": 24, "fps": 30.0, "seed": 0}
    cache = tmp_path / "frames.npz"
    R.load_or_decode("synthetic", None, None, None, synth, str(cache))
    z = dict(np.load(cache, allow_pickle=False))
    z["frames"] = np.zeros_like(z["frames"])
    np.savez(cache, **z)
    with pytest.raises(ValueError):
        R.load_or_decode("synthetic", None, None, None, synth, str(cache))


def test_manifest_identity_is_json_safe():
    code = MF.code_fingerprint(R.ROOT)
    source_info = {"sha256": "a" * 64, "path_type": "file", "window": {"start_s": 60.0, "duration_s": 0.25},
                   "vf": None}
    cfg = {"iters": 12, "ref_iters": 24, "downscale": 2, "flow_dtype": "float32", "corr_dtype": "float32",
           "methods": list(R.METHODS), "directions": list(R.DIRECTIONS), "min_coverage": FP.MIN_COVERAGE}
    ident = R._experiment_identity(source_info, "b" * 64, 59.94, cfg, "c" * 64, code)
    payload = MF.dumps_json({"identity": ident, "content_key": MF.content_key(ident),
                             "schema": R.SCHEMA, "scope": R.SCOPE, "limitations": list(R.LIMITATIONS)})
    obj = json.loads(payload)
    assert obj["schema"] == R.SCHEMA and obj["content_key"] == MF.content_key(ident)
    assert len(obj["limitations"]) == len(R.LIMITATIONS)
    assert "NaN" not in payload and "Infinity" not in payload


def test_schema_version_and_write_are_null_safe(tmp_path):
    assert R.SCHEMA == "mlxstab.flow_prediction_real/1"
    man = {"schema": R.SCHEMA, "scope": R.SCOPE, "limitations": list(R.LIMITATIONS),
           "results": {"forward": {"cold": {"summary": {"refined_epe_mean_low": float("nan"),
                                                          "coverage_mean": 1.0}}}}}
    path = tmp_path / "manifest.json"
    R._write(man, path, required=True)
    obj = json.loads(path.read_text())
    assert obj["results"]["forward"]["cold"]["summary"]["refined_epe_mean_low"] is None
    assert obj["results"]["forward"]["cold"]["summary"]["coverage_mean"] == 1.0
    assert list(tmp_path.glob("*.tmp")) == []          # atomic replace left no temp file behind


def test_main_reports_structured_failure_for_missing_source(tmp_path, capsys):
    out = tmp_path / "run"
    rc = R.main(["--source", str(tmp_path / "nope.mp4"), "--out-dir", str(out)])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["status"] == "failure" and payload["error"]["type"] == "FileNotFoundError"


def test_overwrite_protection(tmp_path, capsys):
    out = tmp_path / "run"
    out.mkdir()
    (out / "manifest.json").write_text("{}")
    rc = R.main(["--source", "synthetic", "--out-dir", str(out)])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["error"]["type"] == "FileExistsError"


# --------------------------------------------------------------------------------------
# opt-in per-iteration budget scan (offline oracle; history fixed at --iters)
# --------------------------------------------------------------------------------------

class _TickClock:
    """Deterministic perf_counter stand-in: every call advances the clock by ``dt``."""

    def __init__(self, dt=0.01):
        self.t = 0.0
        self.dt = float(dt)

    def __call__(self):
        v = self.t
        self.t += self.dt
        return v


class _WorkingStepper:
    """Adds a fixed delta per step and burns clock ticks to simulate device work in ``step()``."""

    def __init__(self, init, clock, delta=0.25, work_ticks=2):
        self.cur = np.asarray(init, np.float64).copy()
        self.clock = clock
        self.delta = float(delta)
        self.work_ticks = int(work_ticks)

    def step(self):
        for _ in range(self.work_ticks):
            self.clock()
        self.cur = self.cur + self.delta
        return self.cur.copy()


class _ScheduledStepper:
    """Stepper whose EPE against a constant-x reference is exactly ``epe_seq[k]`` on the whole grid."""

    def __init__(self, N, ref_x, epe_seq):
        self.N, self.ref_x, self.epe_seq, self.k = int(N), float(ref_x), list(epe_seq), 0

    def step(self):
        e = self.epe_seq[self.k]
        self.k += 1
        out = np.zeros((self.N, 2), np.float64)
        out[:, 0] = self.ref_x - e
        return out


class _ScheduledTickStepper:
    """Scheduled EPE sequence that also burns deterministic device-work clock ticks per step."""

    def __init__(self, N, ref_x, epe_seq, clock, work_ticks=2):
        self.N, self.ref_x, self.epe_seq, self.k = int(N), float(ref_x), list(epe_seq), 0
        self.clock = clock
        self.work_ticks = int(work_ticks)

    def step(self):
        for _ in range(self.work_ticks):
            self.clock()
        e = self.epe_seq[self.k]
        self.k += 1
        out = np.zeros((self.N, 2), np.float64)
        out[:, 0] = self.ref_x - e
        return out


class _TrajectoryStepper:
    """Returns a precomputed flat ``[N,2]`` flow for each step, for exact update-magnitude checks."""

    def __init__(self, flows):
        self.flows = [np.asarray(f, np.float64) for f in flows]
        self.k = 0

    def step(self):
        f = self.flows[self.k]
        self.k += 1
        return f.copy()


def test_parse_thresholds_validates_sorts_and_dedupes():
    assert R.parse_thresholds("0.02,0.005,0.01,0.01") == [0.005, 0.01, 0.02]
    assert R.parse_thresholds([0.1, 0.05]) == [0.05, 0.1]
    for bad in ("", "0", "-1", "nan", "inf", "0.1,x"):
        with pytest.raises(ValueError):
            R.parse_thresholds(bad)


def test_budget_scan_rejects_max_iters_below_iters():
    pairs, refs = R.build_traversal(*make_refs(2), "forward")
    make = lambda init, target: _ScheduledStepper(20, 2.0, [0.0])
    with pytest.raises(ValueError):
        R.replay_budget_scan(pairs, refs, "cold", make, iters=5, max_iters=3, thresholds=[0.1])
    with pytest.raises(ValueError):
        R.replay_budget_scan(pairs, refs, "cold", make, iters=1, max_iters=1, thresholds=[])
    with pytest.raises(ValueError):
        R.replay_budget_scan(pairs, refs, "teleport", make, iters=1, max_iters=2, thresholds=[0.1])


def test_budget_scan_crossings_censor_and_reach_rates():
    H, W, n = 4, 5, 2
    rf, rb = make_refs(n, H, W, fwd=(2.0, 0.0), bwd=(-2.0, 0.0))
    pairs, refs = R.build_traversal(rf, rb, "forward")
    N = H * W
    epe_by_target = {0: [1.8, 1.2, 0.8, 0.5, 0.5], 1: [1.8, 1.2, 0.3, 0.3, 0.3]}
    make = lambda init, target: _ScheduledStepper(N, 2.0, epe_by_target[target])
    thresholds = [1.0, 0.4]
    rows = R.replay_budget_scan(pairs, refs, "cold", make, iters=2, max_iters=5, thresholds=thresholds)
    assert rows[0]["crossings"][0]["iter"] == 3 and rows[0]["min_epe_iter"] == 4
    assert rows[0]["crossings"][1]["iter"] is None and rows[0]["crossed"] is True
    assert rows[1]["crossings"][1]["iter"] == 3 and rows[1]["min_epe_iter"] == 3
    s = R.summarize_budget_scan(rows, 2, 5, thresholds, units_to_source_px=16.0)
    assert s["usable_pairs"] == 2 and s["reached_any"] == 2 and s["censored_any"] == 0
    assert s["per_threshold"][0]["reach_rate"] == 1.0 and s["per_threshold"][0]["iter_mean"] == 3.0
    assert s["per_threshold"][1]["reached"] == 1 and s["per_threshold"][1]["censored"] == 1
    assert s["per_threshold"][1]["reach_rate"] == pytest.approx(0.5)
    assert s["per_threshold"][1]["iter_mean"] == pytest.approx(3.0)
    assert s["per_threshold"][0]["threshold_source_px"] == pytest.approx(16.0)
    # prefix consumption: threshold 0 crosses both at iter 3, threshold 1 censors target 0 at the cap
    assert s["per_threshold"][0]["steps_consumed_mean"] == pytest.approx(3.0)
    assert s["per_threshold"][1]["steps_consumed_mean"] == pytest.approx(4.0)
    assert s["per_threshold"][1]["steps_consumed_max"] == 5
    for p in s["per_threshold"]:
        assert p["total_s_total"] == pytest.approx(p["update_s_total"] + p["epe_check_s_total"])
        assert p["total_s_mean"] == pytest.approx(p["update_s_mean"] + p["epe_check_s_mean"])
    assert s["oracle"] is True


def test_budget_scan_history_is_the_state_at_iters_not_max():
    """The following pair's prediction must see the flow after ``iters``, never after ``max_iters``."""
    H, W, n = 4, 5, 3
    rf, rb = make_refs(n, H, W, fwd=(2.0, 0.0), bwd=(-2.0, 0.0))
    pairs, refs = R.build_traversal(rf, rb, "forward")
    seen = []

    def make(init, target):
        seen.append((int(target), np.asarray(init, np.float64).copy()))
        return _WorkingStepper(init, clock=lambda: None, delta=0.25, work_ticks=0)

    rows = R.replay_budget_scan(pairs, refs, "same_grid", make, iters=2, max_iters=5, thresholds=[0.1])
    by_target = dict(seen)
    np.testing.assert_allclose(by_target[0], 0.0)      # cold start
    np.testing.assert_allclose(by_target[1], 0.5)      # 2-step state, not the 5-step 1.25
    np.testing.assert_allclose(by_target[2], 1.0)      # 2-step state of target 1
    assert all(r["history_iters"] == 2 and r["max_iters"] == 5 for r in rows)
    # the whole grid carries the iters state and only the iters count is recorded as history
    np.testing.assert_allclose(rows[0]["refined_epe_low"]["n"], H * W)
    assert rows[0]["scan"][-1]["iter"] == 5            # the scan itself still ran to the max


def test_budget_scan_separates_update_and_epe_check_time(monkeypatch):
    H, W = 4, 5
    rf, rb = make_refs(1, H, W, fwd=(2.0, 0.0), bwd=(-2.0, 0.0))
    pairs, refs = R.build_traversal(rf, rb, "forward")
    clock = _TickClock(dt=0.01)
    monkeypatch.setattr(R, "time", SimpleNamespace(perf_counter=clock))
    work_ticks, max_iters = 3, 3
    make = lambda init, target: _WorkingStepper(init, clock, delta=0.25, work_ticks=work_ticks)
    rows = R.replay_budget_scan(pairs, refs, "cold", make, iters=2, max_iters=max_iters, thresholds=[0.1])
    r = rows[0]
    # update times only the stepper call (work_ticks + the one closing clock tick); EPE check is 1 tick/iter
    assert r["update_s_total"] == pytest.approx((work_ticks + 1) * 0.01 * max_iters)
    assert r["epe_check_s_total"] == pytest.approx(0.01 * max_iters)
    assert r["update_s_total"] > r["epe_check_s_total"]
    assert r["refine_s"] == pytest.approx((work_ticks + 1) * 0.01 * 2)   # history at iters=2
    s = R.summarize_budget_scan(rows, 2, max_iters, [0.1], units_to_source_px=1.0)
    assert s["full_scan_update_s_total"] == pytest.approx(r["update_s_total"])
    assert s["full_scan_epe_check_s_total"] == pytest.approx(r["epe_check_s_total"])
    assert s["full_scan_total_s"] == pytest.approx(r["update_s_total"] + r["epe_check_s_total"])


def test_budget_scan_flow_update_low_expected_for_known_trajectory():
    """``flow_update_low`` is the per-pixel |low - previous_low| of the candidate's own trajectory."""
    H, W, n = 2, 2, 1
    N = H * W
    rf, rb = make_refs(n, H, W, fwd=(2.0, 0.0), bwd=(-2.0, 0.0))
    pairs, refs = R.build_traversal(rf, rb, "forward")
    step1 = np.zeros((N, 2))
    step1[:, 0] = [0.1, 0.2, 0.3, 0.4]                 # per-pixel magnitudes 0.1, 0.2, 0.3, 0.4
    step2 = step1 + np.array([0.0, 0.5])               # adds a flat 0.5 in y
    traj = [step1.copy(), step2.copy(), step2.copy(), step2.copy()]
    make = lambda init, target: _TrajectoryStepper(traj)
    rows = R.replay_budget_scan(pairs, refs, "cold", make, iters=1, max_iters=4, thresholds=[10.0])
    scan = rows[0]["scan"]
    # iteration 1: baseline is the effective (cold) initialization, so the update is exactly the trajectory's
    # first-step per-pixel magnitude; iteration 2 adds a flat 0.5; iterations 3-4 move by zero.
    first, second, third, fourth = (e["flow_update_low"] for e in scan)
    assert first["n"] == N
    assert first["mean"] == pytest.approx(0.25)
    assert first["p95"] == pytest.approx(0.385)
    assert first["max"] == pytest.approx(0.4)
    assert second["mean"] == pytest.approx(0.5)
    assert second["p95"] == pytest.approx(0.5)
    assert second["max"] == pytest.approx(0.5)
    assert third["mean"] == pytest.approx(0.0) and third["max"] == pytest.approx(0.0)
    assert fourth["n"] == N and fourth["max"] == pytest.approx(0.0)


def test_budget_scan_flow_update_low_ignores_the_cold_reference():
    """The signal uses only candidate state: changing the reference changes EPE but not flow_update_low."""
    H, W, n = 2, 2, 1
    N = H * W
    step1 = np.zeros((N, 2))
    step1[:, 0] = [0.1, 0.2, 0.3, 0.4]
    traj = [step1.copy(), (step1 + np.array([0.0, 0.5]))]
    pairs_a, refs_a = R.build_traversal(*make_refs(n, H, W, fwd=(2.0, 0.0), bwd=(-2.0, 0.0)), "forward")
    pairs_b, refs_b = R.build_traversal(*make_refs(n, H, W, fwd=(5.0, 0.0), bwd=(-5.0, 0.0)), "forward")
    rows_a = R.replay_budget_scan(pairs_a, refs_a, "cold", lambda i, t: _TrajectoryStepper(traj),
                                  iters=1, max_iters=2, thresholds=[10.0])
    rows_b = R.replay_budget_scan(pairs_b, refs_b, "cold", lambda i, t: _TrajectoryStepper(traj),
                                  iters=1, max_iters=2, thresholds=[10.0])
    assert [e["flow_update_low"] for e in rows_a[0]["scan"]] == [e["flow_update_low"] for e in rows_b[0]["scan"]]
    assert [e["epe_low"]["mean"] for e in rows_a[0]["scan"]] != [e["epe_low"]["mean"] for e in rows_b[0]["scan"]]


def test_budget_scan_flow_update_signal_timed_separately_from_epe(monkeypatch):
    """Device update, candidate-observable signal check and oracle EPE check are three separate costs."""
    H, W = 4, 5
    rf, rb = make_refs(1, H, W, fwd=(2.0, 0.0), bwd=(-2.0, 0.0))
    pairs, refs = R.build_traversal(rf, rb, "forward")
    clock = _TickClock(dt=0.01)
    monkeypatch.setattr(R, "time", SimpleNamespace(perf_counter=clock))
    work_ticks, max_iters = 3, 3
    make = lambda init, target: _WorkingStepper(init, clock, delta=0.25, work_ticks=work_ticks)
    rows = R.replay_budget_scan(pairs, refs, "cold", make, iters=2, max_iters=max_iters, thresholds=[0.1])
    r = rows[0]
    # the flat 0.25 step moves both components, so the per-pixel magnitude is 0.25*sqrt(2) every iteration
    expected = 0.25 * np.sqrt(2.0)
    for e in r["scan"]:
        assert e["flow_update_low"]["n"] == H * W
        assert e["flow_update_low"]["mean"] == pytest.approx(expected)
        assert e["flow_update_low"]["max"] == pytest.approx(expected)
    # three independent measured costs: device update, host signal check, host oracle EPE check
    assert r["update_s_total"] == pytest.approx((work_ticks + 1) * 0.01 * max_iters)
    assert r["update_mag_check_s_total"] == pytest.approx(0.01 * max_iters)
    assert r["epe_check_s_total"] == pytest.approx(0.01 * max_iters)
    assert r["update_mag_check_s_total"] > 0.0 and r["epe_check_s_total"] > 0.0
    s = R.summarize_budget_scan(rows, 2, max_iters, [0.1], units_to_source_px=1.0)
    assert s["full_scan_signal_check_s_total"] == pytest.approx(r["update_mag_check_s_total"])
    # the signal time is reported separately and never folded into the existing oracle totals
    assert s["full_scan_total_s"] == pytest.approx(r["update_s_total"] + r["epe_check_s_total"])
    assert s["full_scan_total_with_signal_s"] == pytest.approx(
        s["full_scan_total_s"] + s["full_scan_signal_check_s_total"])
    p = s["per_threshold"][0]
    assert p["signal_check_s_total"] == pytest.approx(r["update_mag_check_s_total"])
    assert p["signal_check_s_mean"] == pytest.approx(r["update_mag_check_s_total"])
    assert p["total_s_total"] == pytest.approx(p["update_s_total"] + p["epe_check_s_total"])
    assert p["total_with_signal_s_total"] == pytest.approx(p["total_s_total"] + p["signal_check_s_total"])
    assert p["total_with_signal_s_mean"] == pytest.approx(p["total_s_mean"] + p["signal_check_s_mean"])


def test_budget_scan_prefix_costs_cover_reached_and_censored_pairs(monkeypatch):
    """Per-threshold prefix costs sum the measured steps: crossing iteration if reached, scan cap if censored."""
    H, W, n = 4, 5, 2
    rf, rb = make_refs(n, H, W, fwd=(2.0, 0.0), bwd=(-2.0, 0.0))
    pairs, refs = R.build_traversal(rf, rb, "forward")
    clock = _TickClock(dt=0.01)
    monkeypatch.setattr(R, "time", SimpleNamespace(perf_counter=clock))
    max_iters, work_ticks = 4, 2
    # target 0 reaches 0.5 at iter 2; target 1 never reaches it (censored -> all max_iters steps)
    epe_by_target = {0: [1.0, 0.4, 0.4, 0.4], 1: [1.0, 0.9, 0.8, 0.7]}
    make = lambda init, target: _ScheduledTickStepper(H * W, 2.0, epe_by_target[target], clock, work_ticks)
    thresholds = [0.5, 0.05]
    rows = R.replay_budget_scan(pairs, refs, "cold", make, iters=2, max_iters=max_iters, thresholds=thresholds)
    assert [r["crossings"][0]["iter"] for r in rows] == [2, None]
    assert [len(r["scan"]) for r in rows] == [max_iters, max_iters]
    s = R.summarize_budget_scan(rows, 2, max_iters, thresholds, units_to_source_px=1.0)
    # per-step measured cost: update = (work_ticks + 1) ticks, epe_check = 1 tick
    update_step, epe_step = (work_ticks + 1) * 0.01, 0.01
    # threshold 0.5: pair 0 consumes 2 steps, pair 1 consumes the 4-step cap
    p0 = s["per_threshold"][0]
    assert p0["reached"] == 1 and p0["censored"] == 1
    assert p0["steps_consumed_mean"] == pytest.approx(3.0)
    assert p0["update_s_total"] == pytest.approx(6 * update_step)
    assert p0["epe_check_s_total"] == pytest.approx(6 * epe_step)
    assert p0["total_s_total"] == pytest.approx(6 * (update_step + epe_step))
    assert p0["update_s_mean"] == pytest.approx(3 * update_step)
    assert p0["epe_check_s_mean"] == pytest.approx(3 * epe_step)
    assert p0["total_s_mean"] == pytest.approx(3 * (update_step + epe_step))
    assert p0["update_s_total"] == pytest.approx(p0["update_s_mean"] * 2)
    # threshold 0.05: both censored, each consuming all max_iters measured steps
    p1 = s["per_threshold"][1]
    assert p1["reached"] == 0 and p1["censored"] == 2
    assert p1["steps_consumed_mean"] == pytest.approx(4.0)
    assert p1["update_s_total"] == pytest.approx(8 * update_step)
    assert p1["epe_check_s_total"] == pytest.approx(8 * epe_step)
    assert p1["total_s_total"] == pytest.approx(8 * (update_step + epe_step))
    assert p1["total_s_mean"] == pytest.approx(4 * (update_step + epe_step))
    assert p1["total_s_total"] == pytest.approx(p1["update_s_total"] + p1["epe_check_s_total"])
    # the full trace equals the censored-both prefix here; full-scan totals sum every recorded row
    assert s["full_scan_update_s_total"] == pytest.approx(8 * update_step)
    assert s["full_scan_epe_check_s_total"] == pytest.approx(8 * epe_step)
    assert s["full_scan_total_s"] == pytest.approx(8 * (update_step + epe_step))


def test_budget_scan_non_finite_step_stops_and_is_flagged():
    H, W, n = 4, 5, 2
    rf, rb = make_refs(n, H, W, fwd=(2.0, 0.0), bwd=(-2.0, 0.0))
    pairs, refs = R.build_traversal(rf, rb, "forward")
    N = H * W

    class _NaNHitter:
        def __init__(self):
            self.k = 0

        def step(self):
            self.k += 1
            out = np.zeros((N, 2), np.float64)
            if self.k >= 1:
                out[0, 0] = np.nan
            return out

    rows = R.replay_budget_scan(pairs, refs, "cold", lambda init, target: _NaNHitter(),
                                iters=2, max_iters=4, thresholds=[0.5])
    assert rows[0]["stopped_reason"] == "non_finite"
    assert len(rows[0]["scan"]) == 1                       # stopped at the first non-finite step
    assert rows[0]["validity_frac"] == pytest.approx((N - 1) / N)
    s = R.summarize_budget_scan(rows, 2, 4, [0.5], units_to_source_px=1.0)
    assert s["stopped_non_finite"] == 2
    # censored non-finite rows consume only their measured (early-stopped) steps, not max_iters
    assert s["per_threshold"][0]["reached"] == 0 and s["per_threshold"][0]["censored"] == 2
    assert s["per_threshold"][0]["steps_consumed_mean"] == pytest.approx(1.0)
    assert s["per_threshold"][0]["steps_consumed_max"] == 1
    assert s["full_scan_update_s_total"] == pytest.approx(
        sum(r["update_s_total"] for r in rows))


def test_budget_scan_matches_insufficient_history_semantics():
    H, W, n = 4, 5, 6
    rf, rb = make_refs(n, H, W, fwd=(2.0, 0.0), bwd=(-2.0, 0.0))
    pairs, refs = R.build_traversal(rf, rb, "forward")
    make = lambda init, target: _ScheduledStepper(H * W, 2.0, [0.1] * 4)
    rows = R.replay_budget_scan(pairs, refs, "constant_acceleration", make,
                                iters=1, max_iters=4, thresholds=[0.05])
    assert [r["reason"] for r in rows[:3]] == ["insufficient_history"] * 3
    s = R.summarize_budget_scan(rows, 1, 4, [0.05], units_to_source_px=8.0)
    assert s["skipped_no_history"] == 3 and s["usable_pairs"] == 3
    assert R.common_targets({"constant_acceleration": {"per_pair": rows}}) == [3, 4, 5]


def test_run_benchmark_budget_scan_preserves_fixed_budget_and_marks_oracle():
    frames = [np.zeros((64, 96, 3), np.uint8) for _ in range(5)]
    fixed = R.run_benchmark(FakeRunner(), frames, methods=R.METHODS, directions=("forward",), log=lambda m: None)
    scan = R.run_benchmark(FakeRunner(), frames, methods=R.METHODS, directions=("forward",),
                           log=lambda m: None, budget_scan=True, budget_max_iters=R.DEFAULT_ITERS + 4,
                           budget_thresholds=[0.01, 0.1])
    assert scan["budget_scan"]["enabled"] is True and scan["budget_scan"]["oracle"] is True
    assert scan["budget_scan"]["history_iters"] == R.DEFAULT_ITERS
    assert scan["budget_scan"]["max_iters"] == R.DEFAULT_ITERS + 4
    for method in R.METHODS:
        s_fixed = fixed["results"]["forward"][method]["summary_common"]
        s_scan = scan["results"]["forward"][method]["summary_common"]
        assert s_scan["usable_pairs"] == s_fixed["usable_pairs"]
        # the scan's fixed-budget-at-iters EPE is the same as the plain replay (FakeScanStepper == refine)
        assert s_scan["refined_epe_mean_low"] == pytest.approx(s_fixed["refined_epe_mean_low"])
        bs = scan["results"]["forward"][method]["budget_scan"]
        assert bs["oracle"] is True and bs["history_iters"] == R.DEFAULT_ITERS
        assert bs["summary"]["oracle"] is True
        assert bs["summary_common"]["usable_pairs"] == s_fixed["usable_pairs"]
        assert len(scan["results"]["forward"][method]["per_pair"]) == 4
        assert "budget_scan" not in fixed["results"]["forward"][method]
    payload = MF.dumps_json(scan)
    assert "NaN" not in payload and "Infinity" not in payload
    json.loads(payload)


def test_run_wires_budget_scan_into_manifest(monkeypatch, tmp_path):
    _RecordingLock.instances.clear()
    monkeypatch.setattr(R, "HeavyLock", _RecordingLock)
    monkeypatch.setattr(R, "RaftRunner", _RunRaftRunner)
    monkeypatch.setattr(R, "load_or_decode", lambda *a, **k: (
        _fake_frames(4), 30.0, {"path_type": "synthetic", "sha256": None, "window": None,
                                "probe": None, "synthetic": None}))
    out = tmp_path / "run-budget"
    rc = R.main(["--source", "synthetic", "--out-dir", str(out), "--budget-scan",
                 "--budget-max-iters", "14", "--budget-thresholds", "0.01,0.5"])
    assert rc == 0
    man = json.loads((out / "manifest.json").read_text())
    assert man["status"] == "success"
    assert man["budget_scan"]["enabled"] is True and man["budget_scan"]["max_iters"] == 14
    assert man["config"]["budget"]["thresholds"] == [0.01, 0.5]
    assert "candidate_observable" in man["budget_scan"]
    assert "stopping_rule" in man["budget_scan"]["candidate_observable"]
    for method in R.METHODS:
        bs = man["results"]["forward"][method]["budget_scan"]
        assert bs["oracle"] is True and "summary_common" in bs
        assert "candidate_observable" in bs
        first = man["results"]["forward"][method]["per_pair"][0]
        assert "flow_update_low" in first["scan"][0]
        assert "update_mag_check_s" in first["scan"][0]
        assert first["scan"][0]["flow_update_low"]["n"] == first["scan"][0]["epe_low"]["n"]
    payload = json.dumps(man, sort_keys=True)
    assert "NaN" not in payload and "Infinity" not in payload


def test_run_off_mode_has_no_budget_block(monkeypatch, tmp_path):
    _RecordingLock.instances.clear()
    monkeypatch.setattr(R, "HeavyLock", _RecordingLock)
    monkeypatch.setattr(R, "RaftRunner", _RunRaftRunner)
    monkeypatch.setattr(R, "load_or_decode", lambda *a, **k: (
        _fake_frames(3), 30.0, {"path_type": "synthetic", "sha256": None, "window": None,
                                "probe": None, "synthetic": None}))
    out = tmp_path / "run-off"
    rc = R.main(["--source", "synthetic", "--out-dir", str(out)])
    assert rc == 0
    man = json.loads((out / "manifest.json").read_text())
    assert man["status"] == "success"
    assert "budget_scan" not in man and "budget" not in man["config"]
    assert all("budget_scan" not in payload for payload in man["results"]["forward"].values())


def test_run_budget_scan_rejects_short_max_iters(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(R, "RaftRunner", _RunRaftRunner)
    monkeypatch.setattr(R, "load_or_decode", lambda *a, **k: (
        _fake_frames(3), 30.0, {"path_type": "synthetic"}))
    out = tmp_path / "run-bad-budget"
    rc = R.main(["--source", "synthetic", "--out-dir", str(out), "--budget-scan",
                 "--iters", "12", "--budget-max-iters", "3"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["status"] == "failure" and payload["error"]["type"] == "ValueError"


# --------------------------------------------------------------------------------------
# candidate-observable photometric consistency (host-side; no reference flow, no RAFT)
# --------------------------------------------------------------------------------------

def _luminance_ramp(H2, W2, ax=3.0, ay=2.0):
    """A linear luminance ramp ``ax*x + ay*y``; the gradient magnitude ``sqrt(ax^2+ay^2)`` is texture."""
    ys, xs = np.mgrid[0:H2, 0:W2]
    return (ax * xs + ay * ys).astype(np.float64)


def _luminance_periodic(H2, W2, amp=50.0, px=32.0, py=24.0):
    """A separable periodic sinusoid, so an integer x shift is exact under ``np.roll``."""
    xs = np.arange(W2)
    ys = np.arange(H2)
    return (amp * np.sin(2.0 * np.pi * xs[None, :] / px)
            + amp * np.sin(2.0 * np.pi * ys[:, None] / py)).astype(np.float64)


def test_preprocess_luminance_matches_integer_box_downscale():
    rng = np.random.default_rng(7)
    frame = (rng.random((8, 8, 3)) * 255).astype(np.uint8)
    L = R.preprocess_luminance(frame, 2)
    block = frame[:8, :8].reshape(4, 2, 4, 2, 3).astype(np.float32).mean(axis=(1, 3))
    manual = 0.299 * block[..., 0] + 0.587 * block[..., 1] + 0.114 * block[..., 2]
    assert L.shape == (4, 4) and L.dtype == np.float32
    np.testing.assert_allclose(L, manual, atol=1e-4)
    # downscale 1 keeps the input shape (the network's ``_prep_pad`` does no averaging then either)
    assert R.preprocess_luminance(frame, 1).shape == (8, 8)


def test_bilinear_luminance_edge_in_bounds_and_non_finite():
    img = np.arange(12, dtype=np.float64).reshape(3, 4)
    vals, ok = R._bilinear_luminance(img, np.array([-0.5, 3.5, 4.0, np.nan]),
                                     np.array([0.0, 0.0, 0.0, 0.0]))
    assert ok[0] and ok[1] and not ok[2] and not ok[3]      # centre in [-0.5, W-0.5] and finite only
    assert vals[1] == pytest.approx(3.0)                    # clamped second tap on the border is exact


def test_photometric_residual_recovers_known_translation():
    """A feature-unit flow of 1.0 must cancel a real 8-preprocessed-pixel translation."""
    H2, W2, h, w = 24, 32, 3, 4
    L_src = _luminance_periodic(H2, W2)
    L_dst = np.roll(L_src, 8, axis=1)                       # L_dst(x) = L_src(x - 8), exact for the period
    flow = np.zeros((h * w, 2)); flow[:, 0] = 1.0
    res = R.photometric_residual(L_src, L_dst, 0.0, flow, (0, 0, 0, 0), h, w)
    # fx=3 (sx=24) warps to 32, one past the 32-wide preprocessed frame, so only 3 of 4 columns are valid
    assert res["resid"]["n"] == 9
    assert res["resid"]["max"] == pytest.approx(0.0, abs=1e-9)
    assert res["resid_textured"]["n"] == 9
    assert res["texture_frac"] == pytest.approx(1.0)


def test_photometric_residual_scales_feature_flow_by_eight():
    """With identical frames the residual is exactly ``|8 * flow_x|`` luminance levels on a ramp."""
    H2, W2, h, w = 24, 32, 3, 4
    L = _luminance_ramp(H2, W2, ax=3.0, ay=2.0)

    def run(flow_x):
        flow = np.zeros((h * w, 2)); flow[:, 0] = flow_x
        return R.photometric_residual(L, L, 0.0, flow, (0, 0, 0, 0), h, w)

    assert run(0.0)["resid"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert run(0.5)["resid"]["mean"] == pytest.approx(12.0, abs=1e-6)   # 8 * 0.5 px * 3 levels/px
    assert run(1.0)["resid"]["mean"] == pytest.approx(24.0, abs=1e-6)


def test_photometric_residual_applies_pad_offset_to_feature_coordinates():
    """Feature cell (fx, fy) sits at ``8*fx - left``, ``8*fy - top``, so pad shifts validity."""
    H2, W2, h, w = 24, 32, 3, 4
    L = _luminance_ramp(H2, W2)
    flow = np.zeros((h * w, 2))
    unpadded = R.photometric_residual(L, L, 0.0, flow, (0, 0, 0, 0), h, w)
    padded = R.photometric_residual(L, L, 0.0, flow, (8, 0, 8, 0), h, w)   # top=8, left=8
    assert unpadded["valid_frac"] == pytest.approx(1.0)
    # top pad drops fy=0 (sy=-8) and left pad drops fx=0 (sx=-8): 2 of 3 rows by 3 of 4 columns remain
    assert padded["valid_frac"] == pytest.approx(6.0 / 12.0)


def test_photometric_residual_marks_out_of_bounds_invalid():
    H2, W2, h, w = 24, 32, 3, 4
    L = _luminance_ramp(H2, W2)
    far = np.full((h * w, 2), 100.0)                        # every target far past the right edge
    res = R.photometric_residual(L, L, 0.0, far, (0, 0, 0, 0), h, w)
    assert res["valid_frac"] == 0.0
    assert res["resid"]["n"] == 0 and res["resid"]["mean"] is None and res["resid"]["p95"] is None
    assert res["resid_textured"]["n"] == 0 and res["texture_frac"] == 0.0


def test_photometric_residual_excludes_non_finite_flow_and_luminance():
    H2, W2, h, w = 24, 32, 3, 4
    L = _luminance_ramp(H2, W2)
    flow = np.zeros((h * w, 2)); flow[:, 0] = 1.0
    assert R.photometric_residual(L, L, 0.0, flow, (0, 0, 0, 0), h, w)["resid"]["n"] == 9
    nan_flow = flow.copy(); nan_flow[0, 0] = np.nan          # cell (fx=0, fy=0) was valid
    r_nan = R.photometric_residual(L, L, 0.0, nan_flow, (0, 0, 0, 0), h, w)
    assert r_nan["resid"]["n"] == 8 and r_nan["valid_frac"] == pytest.approx(8.0 / 12.0)
    inf_flow = flow.copy(); inf_flow[0, 1] = np.inf
    assert R.photometric_residual(L, L, 0.0, inf_flow, (0, 0, 0, 0), h, w)["resid"]["n"] == 8
    # a non-finite source sample invalidates its cell instead of scoring the clamped tap
    L_bad = L.copy(); L_bad[0, 0] = np.nan
    assert R.photometric_residual(L_bad, L, 0.0, flow, (0, 0, 0, 0), h, w)["resid"]["n"] == 8
    all_nan = np.full((h * w, 2), np.nan)
    empty = R.photometric_residual(L, L, 0.0, all_nan, (0, 0, 0, 0), h, w)
    assert empty["resid"]["n"] == 0 and empty["resid"]["max"] is None and empty["valid_frac"] == 0.0


def test_photometric_residual_removes_global_brightness_offset():
    H2, W2, h, w = 24, 32, 3, 4
    L_src = _luminance_periodic(H2, W2)
    L_dst = np.roll(L_src, 8, axis=1) + 40.0                 # translated AND globally brightened by 40
    flow = np.zeros((h * w, 2)); flow[:, 0] = 1.0
    removed = R.photometric_residual(L_src, L_dst, 40.0, flow, (0, 0, 0, 0), h, w)
    kept = R.photometric_residual(L_src, L_dst, 0.0, flow, (0, 0, 0, 0), h, w)
    assert removed["resid"]["max"] == pytest.approx(0.0, abs=1e-9)
    assert kept["resid"]["mean"] == pytest.approx(40.0, abs=1e-6)
    assert removed["luma_offset"] == pytest.approx(40.0)


def test_photometric_residual_texture_mask_excludes_flat_regions():
    H2, W2, h, w = 24, 32, 3, 4
    flow = np.zeros((h * w, 2)); flow[:, 0] = 0.5
    flat = np.full((H2, W2), 100.0)
    r_flat = R.photometric_residual(flat, flat, 0.0, flow, (0, 0, 0, 0), h, w)
    assert r_flat["resid"]["n"] > 0 and r_flat["texture_frac"] == 0.0 and r_flat["resid_textured"]["n"] == 0
    # a gradient below PHOTOMETRIC_TEXTURE_MIN is still treated as textureless
    gentle = (0.5 * np.mgrid[0:H2, 0:W2][1]).astype(np.float64)
    assert R.photometric_residual(gentle, gentle, 0.0, flow, (0, 0, 0, 0), h, w)["texture_frac"] == 0.0
    steep = (3.0 * np.mgrid[0:H2, 0:W2][1]).astype(np.float64)
    assert R.photometric_residual(steep, steep, 0.0, flow, (0, 0, 0, 0), h, w)["texture_frac"] == pytest.approx(1.0)


def test_photometric_warper_preprocesses_frames_and_closure_maps_traversal():
    H, W, h, w = 24, 32, 3, 4
    rng = np.random.default_rng(3)
    frame0 = (rng.random((H, W, 3)) * 255).astype(np.uint8)
    frame1 = np.roll(frame0, 8, axis=1)                      # frame1 is frame0 shifted right by 8 px
    warper = R.PhotometricWarper([frame0, frame1], 1, (0, 0, 0, 0), h, w)
    assert [L.shape for L in warper.lum] == [(H, W), (H, W)]
    assert all(np.isfinite(warper.medians))
    flow = np.zeros((h * w, 2)); flow[:, 0] = 1.0
    forward = warper.closure([(0, 1)])
    backward = warper.closure([(1, 0)])
    # the closure routes target through the traversal order, never a fixed src/dst
    assert forward(0, flow) == warper.residual(0, 1, flow)
    assert backward(0, flow) == warper.residual(1, 0, flow)
    # the same +x flow aligns the forward traversal but not the reversed one
    assert forward(0, flow)["resid"]["mean"] < backward(0, flow)["resid"]["mean"]


def test_photometric_warper_downscale_shapes_and_medians():
    frames = [np.zeros((64, 96, 3), np.uint8), np.full((64, 96, 3), 255, np.uint8)]
    w2 = R.PhotometricWarper(frames, 2, (0, 0, 0, 0), 4, 5)
    assert [L.shape for L in w2.lum] == [(32, 48), (32, 48)]
    assert w2.medians == pytest.approx([0.0, 255.0])
    w1 = R.PhotometricWarper(frames, 1, (0, 0, 0, 0), 8, 12)
    assert [L.shape for L in w1.lum] == [(64, 96), (64, 96)]


def test_budget_scan_records_photometric_low_and_times_it_separately(monkeypatch):
    H, W = 4, 5
    rf, rb = make_refs(1, H, W, fwd=(2.0, 0.0), bwd=(-2.0, 0.0))
    pairs, refs = R.build_traversal(rf, rb, "forward")
    clock = _TickClock(dt=0.01)
    monkeypatch.setattr(R, "time", SimpleNamespace(perf_counter=clock))
    work_ticks, max_iters, photo_ticks = 3, 3, 4
    make = lambda init, target: _WorkingStepper(init, clock, delta=0.25, work_ticks=work_ticks)
    stat = {"valid_frac": 0.5, "texture_frac": 0.25, "luma_offset": 1.5,
            "resid": {"n": 10, "mean": 2.0, "median": 1.5, "p95": 4.0, "max": 6.0},
            "resid_textured": {"n": 2, "mean": 1.0, "median": 1.0, "p95": 1.5, "max": 2.0}}

    def photo(target, flow_flat):
        for _ in range(photo_ticks):
            clock()
        return dict(stat)

    rows = R.replay_budget_scan(pairs, refs, "cold", make, iters=2, max_iters=max_iters,
                                thresholds=[0.1], photometric=photo)
    r = rows[0]
    assert all(e["photometric_low"] == stat for e in r["scan"])
    # the host photometric check burns exactly (photo_ticks + 1) clock ticks per iteration
    assert r["photometric_check_s_total"] == pytest.approx((photo_ticks + 1) * 0.01 * max_iters)
    # device update, update-magnitude check, photometric check and oracle EPE check stay four separate costs
    assert r["update_s_total"] == pytest.approx((work_ticks + 1) * 0.01 * max_iters)
    assert r["update_mag_check_s_total"] == pytest.approx(0.01 * max_iters)
    assert r["epe_check_s_total"] == pytest.approx(0.01 * max_iters)
    assert r["photometric_check_s_total"] > 0.0 and r["update_s_total"] > 0.0
    s = R.summarize_budget_scan(rows, 2, max_iters, [0.1], units_to_source_px=1.0)
    assert s["full_scan_photometric_check_s_total"] == pytest.approx(r["photometric_check_s_total"])
    assert s["full_scan_total_with_observables_s"] == pytest.approx(
        s["full_scan_total_with_signal_s"] + s["full_scan_photometric_check_s_total"])
    p = s["per_threshold"][0]
    assert p["photometric_check_s_total"] == pytest.approx(r["photometric_check_s_total"])
    assert p["photometric_check_s_mean"] == pytest.approx(r["photometric_check_s_total"])
    assert p["total_with_observables_s_total"] == pytest.approx(
        p["total_with_signal_s_total"] + p["photometric_check_s_total"])
    assert p["total_with_observables_s_mean"] == pytest.approx(
        p["total_with_signal_s_mean"] + p["photometric_check_s_mean"])


def test_budget_scan_photometric_low_ignores_the_cold_reference():
    """The photometric signal is a function of the candidate flow and frames only, never the reference."""
    H, W, n = 2, 2, 1
    N = H * W
    flows = [np.full((N, 2), [0.1, 0.2]), np.full((N, 2), [0.3, 0.4])]

    def photo(target, flow_flat):
        return {"resid": {"n": N, "mean": float(flow_flat[0, 0])}}

    pairs_a, refs_a = R.build_traversal(*make_refs(n, H, W, fwd=(2.0, 0.0), bwd=(-2.0, 0.0)), "forward")
    pairs_b, refs_b = R.build_traversal(*make_refs(n, H, W, fwd=(5.0, 0.0), bwd=(-5.0, 0.0)), "forward")
    make = lambda init, target: _TrajectoryStepper(flows)
    rows_a = R.replay_budget_scan(pairs_a, refs_a, "cold", make, iters=1, max_iters=2,
                                  thresholds=[10.0], photometric=photo)
    rows_b = R.replay_budget_scan(pairs_b, refs_b, "cold", make, iters=1, max_iters=2,
                                  thresholds=[10.0], photometric=photo)
    assert [e["photometric_low"] for e in rows_a[0]["scan"]] == [e["photometric_low"] for e in rows_b[0]["scan"]]
    assert [e["epe_low"]["mean"] for e in rows_a[0]["scan"]] != [e["epe_low"]["mean"] for e in rows_b[0]["scan"]]


class _RecordingPhotometricWarper:
    """Records the traversal order each direction binds, without touching frames or the reference."""

    instances = []

    def __init__(self, frames, downscale, pad, h, w, texture_min=None):
        self.orders = []
        _RecordingPhotometricWarper.instances.append(self)

    def closure(self, order):
        self.orders.append([tuple(int(v) for v in o) for o in order])

        def photo(target, flow_flat):
            return {"valid_frac": 1.0, "texture_frac": 1.0, "luma_offset": 0.0,
                    "resid": {"n": int(flow_flat.shape[0]), "mean": 0.0, "median": 0.0,
                              "p95": 0.0, "max": 0.0},
                    "resid_textured": {"n": int(flow_flat.shape[0]), "mean": 0.0, "median": 0.0,
                                       "p95": 0.0, "max": 0.0}}

        return photo


def test_run_benchmark_binds_photometric_closure_to_each_traversal_direction(monkeypatch):
    _RecordingPhotometricWarper.instances.clear()
    monkeypatch.setattr(R, "PhotometricWarper", _RecordingPhotometricWarper)
    frames = [np.zeros((64, 96, 3), np.uint8) for _ in range(5)]
    res = R.run_benchmark(FakeRunner(), frames, methods=("cold",), directions=R.DIRECTIONS,
                          log=lambda m: None, budget_scan=True, budget_max_iters=R.DEFAULT_ITERS + 1,
                          budget_thresholds=[0.1])
    warper = _RecordingPhotometricWarper.instances[-1]
    assert warper.orders == [R.traversal_order(4, "forward"), R.traversal_order(4, "backward")]
    for direction in R.DIRECTIONS:
        first = res["results"][direction]["cold"]["per_pair"][0]["scan"][0]
        assert first["photometric_low"] is not None and "photometric_check_s" in first


def test_run_benchmark_budget_scan_records_photometric_manifest_fields():
    frames = [np.zeros((64, 96, 3), np.uint8) for _ in range(5)]
    res = R.run_benchmark(FakeRunner(), frames, methods=("cold",), directions=("forward",), log=lambda m: None,
                          budget_scan=True, budget_max_iters=R.DEFAULT_ITERS + 2, budget_thresholds=[0.1])
    bs = res["results"]["forward"]["cold"]["budget_scan"]
    # the per-method block carries a prose candidate_observable note; the dicts live at top level and summary
    assert "photometric_low" in bs["candidate_observable"]
    assert "stopping_rule" in res["budget_scan"]["candidate_observable"]
    assert "photometric_low" in res["budget_scan"]["candidate_observable"]
    first = res["results"]["forward"]["cold"]["per_pair"][0]["scan"][0]
    assert first["photometric_low"] is not None and first["photometric_check_s"] >= 0.0
    for key in ("valid_frac", "texture_frac", "luma_offset", "resid", "resid_textured"):
        assert key in first["photometric_low"]
    for key in ("mean", "p95"):
        assert key in first["photometric_low"]["resid"]
        assert key in first["photometric_low"]["resid_textured"]
    summary = bs["summary"]
    assert "full_scan_photometric_check_s_total" in summary
    assert "photometric_low" in summary["candidate_observable"]
    payload = MF.dumps_json(res)
    assert "NaN" not in payload and "Infinity" not in payload
    json.loads(payload)


def test_photometric_limitation_records_alignment_boundary():
    text = " ".join(R.LIMITATIONS)
    assert "photometric_low" in text
    assert "convex-upsampling" in text          # the documented feature-cell -> pixel alignment boundary
    assert "no stop rule" in text.lower() or "not a validated stopping signal" in text


# --------------------------------------------------------------------------------------
# opt-in cold-only bidirectional cycle-consistency pilot
# --------------------------------------------------------------------------------------

class _WorkingBidirStepper:
    """Advances both directions together and burns deterministic clock ticks as device work."""

    def __init__(self, init_f, init_b, clock, delta=0.25, work_ticks=2):
        self.f = np.asarray(init_f, np.float64).copy()
        self.b = np.asarray(init_b, np.float64).copy()
        self.clock = clock
        self.delta = float(delta)
        self.work_ticks = int(work_ticks)

    def step(self):
        for _ in range(self.work_ticks):
            self.clock()
        self.f = self.f + self.delta
        self.b = self.b - self.delta
        return self.f.copy(), self.b.copy()


class _SeqBidirStepper:
    """Returns a precomputed `(fwd_flat, bwd_flat)` pair per step, for exact paired-state checks."""

    def __init__(self, fwd_seq, bwd_seq):
        self.fwd = [np.asarray(f, np.float64) for f in fwd_seq]
        self.bwd = [np.asarray(b, np.float64) for b in bwd_seq]
        self.k = 0

    def step(self):
        f, b = self.fwd[self.k], self.bwd[self.k]
        self.k += 1
        return f.copy(), b.copy()


def _const_field(h, w, x, y):
    f = np.zeros((h * w, 2), np.float64)
    f[:, 0] = x
    f[:, 1] = y
    return f


def test_cycle_residual_zero_for_inverse_consistent_constant_fields():
    h, w = 3, 4
    cyc = R.cycle_residual(_const_field(h, w, 1.0, 0.0), _const_field(h, w, -1.0, 0.0), h, w)
    # the rightmost forward column and leftmost backward column map off-grid and are excluded explicitly
    assert cyc["resid"]["n"] == 2 * (w - 1) * h
    assert cyc["resid"]["mean"] == pytest.approx(0.0)
    assert cyc["resid"]["median"] == pytest.approx(0.0)
    assert cyc["resid"]["p95"] == pytest.approx(0.0)
    assert cyc["resid"]["max"] == pytest.approx(0.0)
    assert cyc["valid_frac"] == pytest.approx((w - 1) / w)
    assert cyc["fwd"]["n"] == (w - 1) * h and cyc["bwd"]["n"] == (w - 1) * h


def test_cycle_residual_measures_analytic_translation_disagreement():
    h, w = 2, 6
    cyc = R.cycle_residual(_const_field(h, w, 1.0, 0.0), _const_field(h, w, -2.0, 0.0), h, w)
    # both round trips sum to (1,0) + (-2,0), so every valid sample has residual exactly 1 px
    for key in ("mean", "median", "p95", "max"):
        assert cyc["resid"][key] == pytest.approx(1.0)
    assert cyc["fwd"]["n"] == 5 * h and cyc["bwd"]["n"] == 4 * h
    assert cyc["resid"]["n"] == cyc["fwd"]["n"] + cyc["bwd"]["n"]


def test_cycle_residual_border_out_of_bounds_and_non_finite_coverage():
    h, w = 3, 5
    # both directions map every point off-grid -> every directional check is excluded, not clamped
    cyc = R.cycle_residual(_const_field(h, w, 100.0, 0.0), _const_field(h, w, 100.0, 0.0), h, w)
    assert cyc["fwd"]["n"] == 0 and cyc["bwd"]["n"] == 0
    assert cyc["resid"]["n"] == 0 and cyc["valid_frac"] == 0.0 and cyc["resid"]["mean"] is None
    # a non-finite vector is dropped explicitly: the NaN forward cell fails the forward check and, sampled
    # at weight 1, the backward check too -> two fewer valid directional samples, no NaN leaks to the stats
    fwd = _const_field(h, w, 1.0, 0.0)
    fwd[0, 0] = np.nan
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        cyc2 = R.cycle_residual(fwd, _const_field(h, w, -1.0, 0.0), h, w)
    assert cyc2["resid"]["n"] == 2 * (w - 1) * h - 2
    for key in ("mean", "median", "p95", "max"):
        assert np.isfinite(cyc2["resid"][key])


def test_cycle_residual_respects_validity_masks():
    h, w = 3, 4
    fwd = np.zeros((h * w, 2))
    bwd = np.zeros((h * w, 2))
    full = R.cycle_residual(fwd, bwd, h, w)
    mask = np.ones((h * w,), bool)
    mask[0] = False
    masked = R.cycle_residual(fwd, bwd, h, w, fwd_valid=mask, bwd_valid=mask)
    assert full["resid"]["n"] == 2 * h * w
    assert masked["resid"]["n"] == 2 * (h * w - 1)
    assert masked["valid_frac"] < full["valid_frac"]
    # the argument surface is candidate fields + grid geometry + validity only (no reference/EPE parameter)
    import inspect
    params = list(inspect.signature(R.cycle_residual).parameters)
    assert params == ["fwd_flat", "bwd_flat", "h", "w", "fwd_valid", "bwd_valid"]


def test_cycle_low_is_independent_of_the_cold_reference():
    """Changing the references changes oracle EPE but never the candidate-observable cycle residual."""
    h, w, n = 2, 2, 1
    fwd = [_const_field(h, w, 0.5, 0.25)]
    bwd = [_const_field(h, w, -0.5, -0.25)]
    make = lambda target: _SeqBidirStepper(fwd, bwd)
    rf_a, rb_a = make_refs(n, h, w, fwd=(2.0, 0.0), bwd=(-2.0, 0.0))
    rf_b, rb_b = make_refs(n, h, w, fwd=(5.0, 0.0), bwd=(-5.0, 0.0))
    rows_a = R.replay_cycle_scan(rf_a, rb_a, make, 1)
    rows_b = R.replay_cycle_scan(rf_b, rb_b, make, 1)
    assert rows_a[0]["scan"][0]["cycle_low"] == rows_b[0]["scan"][0]["cycle_low"]
    assert rows_a[0]["scan"][0]["epe_low"]["mean"] != rows_b[0]["scan"][0]["epe_low"]["mean"]


def test_replay_cycle_scan_uses_each_pair_and_same_iteration_vectors():
    h, w, n = 2, 2, 2
    rf, rb = make_refs(n, h, w, fwd=(1.0, 0.0), bwd=(-1.0, 0.0))
    calls = []

    def make(target):
        calls.append(int(target))
        f = _const_field(h, w, float(target), 0.0)
        b = _const_field(h, w, -float(target), 0.0)
        return _SeqBidirStepper([f, f], [b, b])

    rows = R.replay_cycle_scan(rf, rb, make, 2)
    assert calls == [0, 1]
    assert [r["target"] for r in rows] == [0, 1]
    # each pair is scored against its own reference with its own stepper output (target 0 -> |0-1| = 1,
    # target 1 -> |1-1| = 0), so the fields are genuinely per-pair/same-iteration
    assert rows[0]["scan"][0]["iter"] == 1
    assert rows[0]["scan"][0]["epe_low"]["mean"] == pytest.approx(1.0)
    assert rows[1]["scan"][0]["epe_low"]["mean"] == pytest.approx(0.0)
    assert rows[0]["scan"][0]["cycle_low"]["valid_frac"] > 0.0


def test_replay_cycle_scan_separates_update_cycle_and_oracle_times(monkeypatch):
    h, w, n = 3, 4, 1
    rf, rb = make_refs(n, h, w, fwd=(1.0, 0.0), bwd=(-1.0, 0.0))
    clock = _TickClock(dt=0.01)
    monkeypatch.setattr(R, "time", SimpleNamespace(perf_counter=clock))
    work, iters = 2, 3
    make = lambda target: _WorkingBidirStepper(np.zeros((h * w, 2)), np.zeros((h * w, 2)),
                                               clock, delta=0.25, work_ticks=work)
    rows = R.replay_cycle_scan(rf, rb, make, iters)
    r = rows[0]
    assert r["update_s_total"] == pytest.approx((work + 1) * 0.01 * iters)
    assert r["cycle_check_s_total"] == pytest.approx(0.01 * iters)
    assert r["epe_check_s_total"] == pytest.approx(0.01 * iters)
    assert r["update_s_total"] > r["cycle_check_s_total"]
    s = R.summarize_cycle_scan(rows, iters, units_to_source_px=16.0)
    assert s["full_scan_update_s_total"] == pytest.approx(r["update_s_total"])
    assert s["full_scan_cycle_check_s_total"] == pytest.approx(r["cycle_check_s_total"])
    assert s["full_scan_epe_check_s_total"] == pytest.approx(r["epe_check_s_total"])
    assert s["full_scan_candidate_total_s"] == pytest.approx(r["update_s_total"] + r["cycle_check_s_total"])
    assert s["full_scan_total_with_oracle_s"] == pytest.approx(
        r["update_s_total"] + r["cycle_check_s_total"] + r["epe_check_s_total"])
    assert "cycle_low" in s["candidate_observable"] and "stopping_rule" in s["candidate_observable"]
    assert "epe_low" in s["oracle"]
    assert [row["iter"] for row in s["cycle_low_by_iter"]] == [1, 2, 3]
    assert s["cycle_low_by_iter"][0]["trajectories"] == 1


def test_replay_cycle_scan_stops_on_non_finite():
    h, w, n = 2, 2, 1
    rf, rb = make_refs(n, h, w, fwd=(1.0, 0.0), bwd=(-1.0, 0.0))
    good_f, good_b = _const_field(h, w, 0.1, 0.0), _const_field(h, w, -0.1, 0.0)
    bad_f = good_f.copy()
    bad_f[0, 0] = np.nan
    make = lambda target: _SeqBidirStepper([good_f, bad_f, good_f], [good_b, good_b, good_b])
    rows = R.replay_cycle_scan(rf, rb, make, 3)
    r = rows[0]
    assert r["stopped_reason"] == "non_finite" and len(r["scan"]) == 2
    s = R.summarize_cycle_scan(rows, 3, units_to_source_px=1.0)
    assert s["stopped_non_finite"] == 1 and s["usable_pairs"] == 1


def test_run_benchmark_cycle_scan_is_json_safe_and_skips_standard_methods():
    frames = [np.zeros((64, 96, 3), np.uint8) for _ in range(5)]
    res = R.run_benchmark(FakeRunner(), frames, log=lambda m: None, cycle_scan=True, cycle_max_iters=3)
    assert res["results"] == {} and res["common_targets"] == {}
    cs = res["cycle_scan"]
    assert cs["enabled"] is True and cs["kind"] == "cold_bidirectional_cycle_consistency"
    assert cs["max_iters"] == 3 and cs["direction"] == "bidirectional" and cs["method"] == "cold"
    per_pair = cs["results"]["bidirectional"]["cold"]["per_pair"]
    assert len(per_pair) == 4
    assert "cycle_low" in per_pair[0]["scan"][0] and "epe_low" in per_pair[0]["scan"][0]
    assert cs["results"]["bidirectional"]["cold"]["summary"]["usable_pairs"] == 4
    payload = MF.dumps_json(res)
    assert "NaN" not in payload and "Infinity" not in payload
    json.loads(payload)


def test_run_benchmark_rejects_cycle_and_budget_combination():
    frames = [np.zeros((64, 96, 3), np.uint8) for _ in range(4)]
    with pytest.raises(ValueError):
        R.run_benchmark(FakeRunner(), frames, log=lambda m: None, budget_scan=True, cycle_scan=True)
    with pytest.raises(ValueError):
        R.run_benchmark(FakeRunner(), frames, log=lambda m: None, cycle_scan=True, cycle_max_iters=0)


def test_run_wires_cycle_scan_into_manifest(monkeypatch, tmp_path):
    _RecordingLock.instances.clear()
    monkeypatch.setattr(R, "HeavyLock", _RecordingLock)
    monkeypatch.setattr(R, "RaftRunner", _RunRaftRunner)
    monkeypatch.setattr(R, "load_or_decode", lambda *a, **k: (
        _fake_frames(4), 30.0, {"path_type": "synthetic", "sha256": None, "window": None,
                                "probe": None, "synthetic": None}))
    out = tmp_path / "run-cycle"
    rc = R.main(["--source", "synthetic", "--out-dir", str(out), "--cycle-scan", "--cycle-max-iters", "5",
                 "--lock", "on", "--lock-chunk", "4"])
    assert rc == 0
    man = json.loads((out / "manifest.json").read_text())
    assert man["status"] == "success"
    assert man["config"]["cycle"] == {"scan": True, "max_iters": 5}
    assert man["cycle_scan"]["enabled"] is True and man["cycle_scan"]["max_iters"] == 5
    assert "budget_scan" not in man and man["results"] == {}
    assert man["cycle_scan"]["results"]["bidirectional"]["cold"]["summary"]["usable_pairs"] == 3
    # feature prep (4 frames) + 3 references + 3 cycle pairs share one monotonic lock gate
    lock = _RecordingLock.instances[-1]
    assert lock.steps == list(range(1, 4 + 3 + 3 + 1))
    payload = json.dumps(man, sort_keys=True)
    assert "NaN" not in payload and "Infinity" not in payload


def test_run_cycle_scan_rejects_budget_combination(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(R, "RaftRunner", _RunRaftRunner)
    monkeypatch.setattr(R, "load_or_decode", lambda *a, **k: (
        _fake_frames(3), 30.0, {"path_type": "synthetic"}))
    out = tmp_path / "run-bad-cycle"
    rc = R.main(["--source", "synthetic", "--out-dir", str(out), "--cycle-scan", "--budget-scan"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["status"] == "failure" and payload["error"]["type"] == "ValueError"


def _average_tie_auroc(scores, labels):
    """Weight-free AUROC with average ranks for ties (the standard rank-statistic definition).

    ``scores`` is a 1-D numeric array and ``labels`` a same-length boolean array whose True entries are the
    positives; returns ``(auc, n_pos, n_neg)`` or ``(None, n_pos, n_neg)`` when one class is empty. Ties in
    ``scores`` share the mean of their ranks, so a constant score gives exactly 0.5 and no arbitrary ordering
    is imposed.
    """
    scores = np.asarray(scores, np.float64).ravel()
    labels = np.asarray(labels, bool).ravel()
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None, n_pos, n_neg
    order = np.argsort(scores, kind="mergesort")
    ranked = np.empty(scores.size, np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < scores.size:
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranked[i:j + 1] = (i + j) / 2.0 + 1.0      # average of the 1-based ranks the tie spans
        i = j + 1
    ranks = np.empty(scores.size, np.float64)
    ranks[order] = ranked
    auc = (ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc), n_pos, n_neg


def test_average_tie_auroc_reference_cases():
    """The AUROC helper itself: perfect, inverted, constant (tie -> 0.5) and single-class cases."""
    assert _average_tie_auroc([0.1, 0.2, 0.3, 0.4], [False, False, True, True])[0] == pytest.approx(1.0)
    assert _average_tie_auroc([0.4, 0.3, 0.2, 0.1], [False, False, True, True])[0] == pytest.approx(0.0)
    assert _average_tie_auroc([1.0, 1.0, 1.0, 1.0], [False, True, False, True])[0] == pytest.approx(0.5)
    auc, n_pos, n_neg = _average_tie_auroc([0.1, 0.2], [True, True])
    assert auc is None and (n_pos, n_neg) == (2, 0)


def test_cycle_residual_forward_occlusion_and_backward_disocclusion_auroc():
    """Exact-analytic visibility check: the cycle residual flags forward occlusion and reverse disocclusion.

    Weight-free and reference-free. For each exact ``parallax_two_layer`` analytic :class:`flow_prediction.Pair`
    (48x36, four pairs) the forward field is ``pair.forward(grid) - grid`` and the backward field is
    ``pair.inverse(grid) - grid``; :func:`cycle_residual_fields` is called with no ground-truth masks. On the
    helper's ``fwd_valid`` support the score is the forward residual and the label is the pair's own occlusion
    ``~pair.forward_valid(grid)``, so a good direction gives AUROC ~1.0. The reversed pair (the original clip
    traversed backwards) is evaluated on its own ``fwd_valid`` support with the reversed pair's *forward*
    residual scored against ``~rev.forward_valid(grid)``. The reversed pair's forward source is original frame
    k+1, so this label is exactly the original *disocclusion*; the score and label must be taken on the same
    source grid, giving AUROC ~1.0. This is a visibility sanity check on one exact 48x36 toy, not semantic or
    depth understanding and not a real-footage result.
    """
    fx = FP.build_fixtures()["parallax_two_layer"]
    H, W = fx.H, fx.W
    assert (W, H) == (48, 36) and len(fx.pairs) == 4
    grid = FP.source_grid(W, H)

    fwd_scores, fwd_labels = [], []
    rev_scores, rev_labels = [], []
    for pair in fx.pairs:
        fwd = (pair.forward(grid) - grid).reshape(H * W, 2)
        bwd = (pair.inverse(grid) - grid).reshape(H * W, 2)
        fields = R.cycle_residual_fields(fwd, bwd, H, W)     # no fwd_valid/bwd_valid GT masks passed in
        keep = fields["fwd_valid"].reshape(-1)
        fwd_scores.append(fields["fwd_resid"].reshape(-1)[keep])
        fwd_labels.append((~pair.forward_valid(grid))[keep])

        rev = pair.reversed()                                # forward = inverse, inverse = forward
        rfwd = (rev.forward(grid) - grid).reshape(H * W, 2)
        rbwd = (rev.inverse(grid) - grid).reshape(H * W, 2)
        rfields = R.cycle_residual_fields(rfwd, rbwd, H, W)
        keep_r = rfields["fwd_valid"].reshape(-1)
        # reversed forward residual vs reversed forward-valid label, both on the reversed fwd source grid
        rev_scores.append(rfields["fwd_resid"].reshape(-1)[keep_r])
        rev_labels.append((~rev.forward_valid(grid))[keep_r])   # reversed forward occlusion == original disocclusion

    fwd_auc, fwd_pos, _ = _average_tie_auroc(np.concatenate(fwd_scores), np.concatenate(fwd_labels))
    rev_auc, rev_pos, _ = _average_tie_auroc(np.concatenate(rev_scores), np.concatenate(rev_labels))
    assert fwd_pos > 0 and rev_pos > 0                          # both label sets are non-degenerate
    assert fwd_auc == pytest.approx(1.0), f"forward occlusion AUROC {fwd_auc}"
    assert rev_auc == pytest.approx(1.0), f"reverse disocclusion AUROC {rev_auc}"
    # both analytic AUROCs are exactly 1.0 on this toy: forward occlusion and reverse disocclusion
    assert fwd_auc == pytest.approx(1.0, abs=1e-12)
    assert rev_auc == pytest.approx(1.0, abs=1e-12)


def test_cycle_limitation_states_diagnostic_boundary():
    text = " ".join(R.LIMITATIONS)
    assert "cycle_low" in text
    assert "not true-flow accuracy" in text
    assert "does not prove" in text
    # the observable function itself never accepts a reference field
    import inspect
    assert "ref" not in ",".join(inspect.signature(R.cycle_residual).parameters)


@pytest.mark.skipif(not R.DEFAULT_WEIGHTS.exists(), reason="raft-things weights absent")
def test_bidir_stepper_parity_with_reference_pair():
    """A full paired scan to ref_iters reproduces the production batch-2 ``reference_pair`` within 1e-3 EPE."""
    rng = np.random.default_rng(0)
    base = (rng.random((96, 128, 3)) * 255).astype(np.uint8)
    frames = [base, np.roll(base, 3, axis=1)]
    runner = R.RaftRunner(str(R.DEFAULT_WEIGHTS), downscale=2)
    runner.prepare(frames)
    iters = 4
    ref_f, ref_b = runner.reference_pair(0, 1, iters)
    stepper = runner.begin_bidir_scan(0, 1)
    low_f = low_b = None
    for _ in range(iters):
        low_f, low_b = stepper.step()
    epe_f = float(np.sqrt(((np.asarray(low_f).reshape(ref_f.shape) - ref_f) ** 2).sum(-1)).mean())
    epe_b = float(np.sqrt(((np.asarray(low_b).reshape(ref_b.shape) - ref_b) ** 2).sum(-1)).mean())
    assert epe_f < 1e-3 and epe_b < 1e-3
