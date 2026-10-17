"""Tests for the shared-pipeline backend comparison's reporting safeguards."""
import json

import numpy as np

from bench import bench_stabilizer as BS


def test_backend_order_is_explicit_and_validated():
    assert BS._backend_order("mlx-first") == ("mlx", "torch-mps")
    assert BS._backend_order("torch-first") == ("torch-mps", "mlx")
    try:
        BS._backend_order("mps-first")
    except ValueError as exc:
        assert "backend_order" in str(exc)
    else:
        raise AssertionError("unsupported backend order was accepted")


def test_plan_delta_reports_global_mesh_and_row_changes(tmp_path):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    np.savez(a, frames=np.int64(2), G=np.zeros((2, 3, 3)), mesh_corr=np.zeros((2, 2)),
             row_corr=np.zeros((2, 2)))
    np.savez(b, frames=np.int64(2), G=np.ones((2, 3, 3)), mesh_corr=np.full((2, 2), 2),
             row_corr=np.full((2, 2), 3))
    result = BS._plan_delta(a, b)
    assert result["available"] and result["frames"] == 2
    assert result["G"] == {"shape_match": True, "max_abs": 1.0, "mean_abs": 1.0}
    assert result["mesh_corr"]["max_abs"] == 2.0
    assert result["row_corr"]["max_abs"] == 3.0


def test_encoded_cadence_comparison_reads_timestamp_samples():
    left = {"encoded_pts": {"pts_s": [0.0, 0.033367]},
            "cadence": {"measured_fps": 30000 / 1001}}
    right = {"encoded_pts": {"pts_s": [0.0, 0.033367]},
             "cadence": {"measured_fps": 30000 / 1001}}
    assert BS._cadence_matches(left, right)
    right["encoded_pts"]["pts_s"][-1] += 0.001
    assert not BS._cadence_matches(left, right)


def test_manifest_command_replays_single_backend_runner(tmp_path):
    command = BS._runner_command("input.mp4", tmp_path / "torch-mps", "torch-mps",
                                 "mlx.safetensors", "torch.pth", 1.5, 2.0,
                                 12, 2, 12, 2, 3, 7, True, "30000/1001", "off", True)
    assert command[1:3] == ["-I", str(BS.ROOT / "bench" / "run_case.py")]
    assert command[command.index("--backend") + 1] == "torch-mps"
    assert command[command.index("--source") + 1] == "input.mp4"
    assert command[command.index("--start") + 1] == "1.5"
    assert "--eval-tracks" in command
    assert "--dump-motion" in command


def test_failed_backend_writes_failure_json_without_comparison_table(tmp_path, monkeypatch):
    root = tmp_path / "comparison"
    def make_source(path, *args):
        path.write_bytes(b"synthetic")
    monkeypatch.setattr(BS.runner, "make_synthetic", make_source)
    calls = iter([
        {"status": "success", "flow_backend": {"name": "mlx"}, "outputs": {"manifest": "mlx/manifest.json"}},
        {"status": "failure", "flow_backend": {"name": "torch-mps"},
         "error": {"stage": "model_construct", "message": "MPS unavailable"},
         "outputs": {"manifest": "torch/manifest.json"}},
    ])
    monkeypatch.setattr(BS.runner, "run_case", lambda opts: next(calls))
    result = BS.compare(out_dir=str(root), synth_frames=2, synth_w=32, synth_h=32, warm_repeats=1)
    assert result["status"] == "failure"
    assert result["failed_backend"] == "torch-mps"
    assert json.loads((root / "comparison.json").read_text())["status"] == "failure"
    assert not (root / "comparison.md").exists()


def test_compare_honors_torch_first_order(tmp_path, monkeypatch):
    root = tmp_path / "comparison"
    def make_source(path, *args):
        path.write_bytes(b"synthetic")
    monkeypatch.setattr(BS.runner, "make_synthetic", make_source)
    calls = []
    def fail_first(opts):
        calls.append((opts.backend, opts.dump_motion))
        return {"status": "failure", "flow_backend": {"name": opts.backend},
                "error": {"stage": "model_construct", "message": "stop after first backend"}}
    monkeypatch.setattr(BS.runner, "run_case", fail_first)
    result = BS.compare(out_dir=str(root), synth_frames=2, synth_w=32, synth_h=32,
                        warm_repeats=1, backend_order="torch-first", dump_motion=True)
    assert calls == [("torch-mps", True)]
    assert result["failed_backend"] == "torch-mps"
