import numpy as np

from bench import run_case, runner


def test_layer_motion_factor_parser():
    assert run_case._parse_layer_motion_factors("1,.8,.6") == (1.0, 0.8, 0.6)
    try:
        run_case._parse_layer_motion_factors("1,.6")
    except Exception as exc:
        assert "three" in str(exc)
    else:
        raise AssertionError("accepted a factor tuple with the wrong length")


def test_run_options_reject_a_moving_far_layer_for_parallax_screens():
    opts = runner.RunOptions(synth_layer_motion_factors=(0.8, 0.6, 0.4))
    assert any("far-layer" in error for error in runner.validate_options(opts))
    opts = runner.RunOptions(synth_intent_profile="not-a-profile")
    assert any("synth-intent-profile" in error for error in runner.validate_options(opts))
    opts = runner.RunOptions(synth_frames=15, synth_intent_profile="matched-spectrum")
    assert any("synth-frames >= 16" in error for error in runner.validate_options(opts))
    assert any("synth-fps" in error for error in runner.validate_options(runner.RunOptions(synth_fps=0)))


def test_synthetic_source_records_exact_scene_factors_and_camera_truth(tmp_path, monkeypatch):
    calls = {}

    def fake_render(**kwargs):
        calls.update(kwargs)
        return (np.zeros((kwargs["T"], kwargs["H"], kwargs["W"], 3), np.uint8),
                np.zeros((kwargs["T"], 3)),
                {"visible_layer_id": np.zeros((kwargs["T"], kwargs["H"], kwargs["W"]), np.uint8)})

    class FakeEncoder:
        def __init__(self, *args, **kwargs):
            self.frames = []

        def write(self, frame):
            self.frames.append(frame)

        def close(self):
            pass

    monkeypatch.setattr(runner.synth, "render", fake_render)
    monkeypatch.setattr(runner.media, "Encoder", FakeEncoder)
    meta = runner.make_synthetic(tmp_path / "scene.mp4", 64, 48, 30, 13, 30,
        intent_mode="stochastic", layer_motion_factors=(1.0, 0.85, 0.7), subject_mode="static",
        subject_speed=1.6, jitter_px=5.0, rot_jitter_deg=0.4, rho=0.25,
        intent_profile="matched-spectrum")

    assert calls["intent_mode"] == "stochastic"
    assert calls["depths"] == (1.0, 0.85, 0.7)
    assert calls["subject"] is True and calls["subject_speed"] == 0
    assert calls["intent_profile"] == "matched-spectrum"
    assert meta["intent_mode"] == "stochastic"
    assert meta["subject_mode"] == "static"
    assert meta["subject_present"] is True
    assert meta["intent_profile"] == "matched-spectrum"
    assert meta["layer_motion_factors"] == [1.0, 0.85, 0.7]
    assert len(meta["intended_camera_pose_px_px_deg"]) == 30
    assert len(meta["actual_camera_pose_px_px_deg"]) == 30
    assert np.all(np.asarray(meta["camera_shake_rms_px_px_deg"]) > 0)
    assert meta["truth_file"] == "synthetic_truth.npz"
    assert len(meta["truth_sha256"]) == 64


def test_parser_exposes_scene_factor_options():
    args = run_case.build_parser().parse_args([
        "--source", "synthetic", "--synth-intent-mode", "stochastic",
        "--synth-layer-motion-factors", "1,.9,.8", "--synth-subject-mode", "absent",
        "--synth-jitter", "4", "--synth-rho", ".2",
    ])
    assert args.synth_intent_mode == "stochastic"
    assert args.synth_layer_motion_factors == (1.0, 0.9, 0.8)
    assert args.synth_subject_mode == "absent"
    assert args.synth_jitter == 4
    assert args.synth_rho == 0.2
