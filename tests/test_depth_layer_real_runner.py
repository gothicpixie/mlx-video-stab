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


def test_feature_grid_correction_conjugates_into_pixel_translation():
    pad = (3, 5, 7, 11)  # top, bottom, left, right
    downscale = 2
    scale = runner.FEATURE_STRIDE * downscale
    correction = np.array([[1.0, 0.0, 2.5], [0.0, 1.0, -1.25], [0.0, 0.0, 1.0]])
    pixel = runner._correction_to_pixel_matrix(correction, pad, downscale)
    assert pixel[0, 0] == pytest.approx(1.0)
    assert pixel[1, 1] == pytest.approx(1.0)
    assert pixel[0, 2] == pytest.approx(scale * 2.5)
    assert pixel[1, 2] == pytest.approx(scale * -1.25)
    feature_from_pixel = runner._feature_to_pixel_matrix(pad, downscale)
    expected = np.linalg.inv(feature_from_pixel) @ correction @ feature_from_pixel
    np.testing.assert_allclose(pixel, expected)


def test_preview_renders_contact_sheet_without_leaking_paths(tmp_path):
    frames, depth, weights = write_caches(tmp_path)
    before = sha256_file(frames)
    preview = tmp_path / "preview dir" / "sheet.png"
    result = runner.run_cached_audit(frames, depth, weights, None, runner_factory=FakeRunner,
                                     lock_factory=FakeLock, n_bins=2, smooth_sigma=1.0,
                                     preview_out=preview, preview_bins=2, preview_sigma=1.0,
                                     write=False, log=None)
    assert preview.is_file()
    assert preview.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image
    with Image.open(preview) as sheet:
        assert sheet.width > 0 and sheet.height > 0
    # The input frame cache is only read, and no output/input path is written into the report.
    assert sha256_file(frames) == before
    payload = json.dumps(result, allow_nan=False)
    assert str(preview) not in payload
    assert str(tmp_path) not in payload
    assert "preview_out" not in result


def test_preview_renders_requested_sweep_variant(tmp_path):
    frames, depth, weights = write_caches(tmp_path)
    preview = tmp_path / "sweep.png"
    result = runner.run_cached_audit(frames, depth, weights, None, runner_factory=FakeRunner,
                                     lock_factory=FakeLock, sweep_bins=(1, 2),
                                     sweep_sigmas=(0, 1), preview_out=preview,
                                     preview_bins=2, preview_sigma=1.0, write=False, log=None)
    assert preview.is_file()
    assert set(result["variants"]) == {
        "bins_1__sigma_0", "bins_1__sigma_1", "bins_2__sigma_0", "bins_2__sigma_1"}
    json.dumps(result, allow_nan=False)


def test_preview_rejects_missing_sweep_variant_before_raft(tmp_path):
    frames, depth, weights = write_caches(tmp_path)
    created = []

    def factory(*args):
        created.append(1)
        return FakeRunner(*args)

    preview = tmp_path / "missing.png"
    with pytest.raises(ValueError, match="preview"):
        runner.run_cached_audit(frames, depth, weights, None, runner_factory=factory,
                                lock_factory=FakeLock, sweep_bins=(1, 2), sweep_sigmas=(0, 1),
                                preview_out=preview, preview_bins=4, preview_sigma=1.0,
                                write=False, log=None)
    assert created == []
    assert not preview.exists()


def test_preview_rejects_ambiguous_single_mode_before_raft(tmp_path):
    frames, depth, weights = write_caches(tmp_path)
    created = []

    def factory(*args):
        created.append(1)
        return FakeRunner(*args)

    with pytest.raises(ValueError, match="preview"):
        runner.run_cached_audit(frames, depth, weights, None, runner_factory=factory,
                                lock_factory=FakeLock, n_bins=4, smooth_sigma=1.0,
                                preview_out=tmp_path / "mismatch.png", preview_bins=2,
                                preview_sigma=1.0, write=False, log=None)
    assert created == []


@pytest.mark.parametrize("collision", ["frames", "depth", "weights", "report"])
def test_preview_cannot_overwrite_an_input_or_json_report(tmp_path, collision):
    frames, depth, weights = write_caches(tmp_path)
    paths = {"frames": frames, "depth": depth, "weights": weights,
             "report": tmp_path / "report.json"}
    created = []

    def factory(*args):
        created.append(1)
        return FakeRunner(*args)

    report_path = paths["report"] if collision == "report" else None
    with pytest.raises(ValueError, match="preview_out must not overwrite"):
        runner.run_cached_audit(frames, depth, weights, report_path, runner_factory=factory,
                                lock_factory=FakeLock, n_bins=4, smooth_sigma=1.0,
                                preview_out=paths[collision], preview_bins=4, preview_sigma=1.0,
                                write=False, log=None)
    assert created == []
