import json

import numpy as np
import pytest

from bench import depth_layer_real_runner as runner
from bench.manifest import sha256_file


class FakeLock:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def on_step(self, step):
        pass

    def status(self):
        return {"path": "/private/lock", "enabled": self.kwargs["enabled"]}


class FakeRunner:
    def __init__(self, weights, downscale, flow_dtype, corr_dtype):
        self.pad = (0, 0, 0, 0)
        self.downscale = downscale
        self.prepared = None

    def prepare(self, frames, on_step=None):
        self.prepared = frames
        self.h, self.w = 8, 10
        for _ in frames:
            on_step()

    def reference_pair(self, src, dst, iters):
        return np.zeros((self.h, self.w, 2)), np.zeros((self.h, self.w, 2))


class CountingRunner(FakeRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pair_calls = 0

    def reference_pair(self, src, dst, iters):
        self.pair_calls += 1
        return super().reference_pair(src, dst, iters)


def write_caches(tmp_path, *, n=3, h=64, w=80):
    frames_path = tmp_path / "frames.npz"
    frames = np.zeros((n, h, w, 3), np.uint8)
    frames[:, :, :, 0] = np.arange(n, dtype=np.uint8)[:, None, None]
    meta = np.array(json.dumps({"height": h, "width": w, "fps": 2.0}))
    np.savez_compressed(frames_path, frames=frames, meta=meta)
    indices = np.arange(n, dtype=np.int64)
    yy, xx = np.mgrid[:h, :w]
    depth = np.stack([xx + yy + i for i in indices]).astype(np.float32)
    depth_path = tmp_path / "depth.npz"
    np.savez_compressed(depth_path, indices=indices, depth=depth, model_id=np.array("fake-depth"),
                        normalization=np.array("fixture"),
                        frames_npz_sha256=np.array(sha256_file(frames_path)))
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"fake")
    return frames_path, depth_path, weights


def test_cached_runner_pairs_flow_and_writes_path_free_report(tmp_path):
    frames, depth, weights = write_caches(tmp_path)
    out = tmp_path / "report.json"
    result = runner.run_cached_audit(frames, depth, weights, out, runner_factory=FakeRunner,
                                     lock_factory=FakeLock, n_bins=2, write=True, log=None)
    assert result["pair_count"] == 2
    assert len(result["pairs"]) == 2
    assert result["pairs"][0]["source_frame_indices"] == [0, 1]
    assert result["pairs"][0]["time_s"] == [0.0, 0.5]
    assert result["pairs"][0]["cycle_map_valid_fraction"] == 1.0
    assert result["flow"]["feature_grid"] == [8, 10]
    assert result["inputs"]["weights_sha256"]
    assert "path" not in result["timing_s"]["lock"]
    payload = out.read_text()
    assert str(tmp_path) not in payload
    json.loads(payload)


def test_cached_runner_rejects_mismatched_frame_and_depth_cache(tmp_path):
    frames, depth, weights = write_caches(tmp_path)
    np.savez_compressed(depth, indices=np.arange(3), depth=np.zeros((3, 64, 80), np.float32),
                        model_id=np.array("fake-depth"), normalization=np.array("fixture"),
                        frames_npz_sha256=np.array("0" * 64))
    with pytest.raises(ValueError, match="does not match"):
        runner.run_cached_audit(frames, depth, weights, write=False, runner_factory=FakeRunner,
                                lock_factory=FakeLock, log=None)


def test_runner_rejects_bad_parameters_before_loading_model(tmp_path):
    frames, depth, weights = write_caches(tmp_path)
    with pytest.raises(ValueError, match="iters"):
        runner.run_cached_audit(frames, depth, weights, iters=0, runner_factory=FakeRunner,
                                lock_factory=FakeLock, log=None)


def test_sweep_runner_runs_raft_once_and_reuses_observations(tmp_path):
    frames, depth, weights = write_caches(tmp_path)
    created = []

    def factory(*args):
        instance = CountingRunner(*args)
        created.append(instance)
        return instance

    result = runner.run_cached_audit(frames, depth, weights, None, runner_factory=factory,
                                     lock_factory=FakeLock, sweep_bins=(1, 2),
                                     sweep_sigmas=(0, 1), write=False, log=None)
    assert len(created) == 1
    assert created[0].pair_calls == 2  # one RAFT pass over the two adjacent frame pairs
    assert set(result["variants"]) == {
        "bins_1__sigma_0", "bins_1__sigma_1", "bins_2__sigma_0", "bins_2__sigma_1"}
    for variant in result["variants"].values():
        assert variant["pair_count"] == 2
        assert len(variant["pairs"]) == 2
        assert variant["pairs"][0]["source_frame_indices"] == [0, 1]
        assert variant["pairs"][0]["time_s"] == [0.0, 0.5]
    assert "pairs" not in result  # the sweep nests each variant instead of a single-run report
    assert result["flow"]["feature_grid"] == [8, 10]
    json.dumps(result, allow_nan=False)


def test_sweep_runner_falls_back_to_single_run_axis(tmp_path):
    frames, depth, weights = write_caches(tmp_path)
    result = runner.run_cached_audit(frames, depth, weights, None, runner_factory=FakeRunner,
                                     lock_factory=FakeLock, sweep_bins=(1,), smooth_sigma=0.5,
                                     write=False, log=None)
    assert set(result["variants"]) == {"bins_1__sigma_0.5"}


def test_sweep_runner_rejects_bad_grid_before_loading_model(tmp_path):
    frames, depth, weights = write_caches(tmp_path)
    with pytest.raises(ValueError, match="bin_counts"):
        runner.run_cached_audit(frames, depth, weights, None, runner_factory=FakeRunner,
                                lock_factory=FakeLock, sweep_bins=(0,), write=False, log=None)
