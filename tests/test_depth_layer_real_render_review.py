import json

import numpy as np
import pytest

from bench import depth_layer_real_render_review as review
from bench.manifest import sha256_file


def test_pull_affine_identity_preserves_pixels_and_edge_support():
    rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)

    out, valid = review.pull_affine(rgb, np.eye(3))

    np.testing.assert_array_equal(out, rgb)
    assert valid.all()


def test_pull_affine_translation_uses_inverse_map_and_marks_exposed_edge():
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(5, dtype=np.uint8)[None, :]
    correction = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    out, valid = review.pull_affine(rgb, correction)

    assert not valid[:, 0].any()
    assert valid[:, 1:].all()
    np.testing.assert_array_equal(out[:, 1:, 0], rgb[:, :-1, 0])
    assert np.all(out[:, 0] == 0)


def _small_frames(path, count=2):
    frames = np.zeros((count, 4, 5, 3), np.uint8)
    meta = {"source_sha256": "source", "source_path": "unused", "window": {"start_s": 0, "duration_s": 1},
            "fps": 2.0, "height": 4, "width": 5, "count": count}
    np.savez_compressed(path, frames=frames, meta=np.asarray(json.dumps(meta)))
    return frames


def test_load_inputs_refuses_frames_npz_hash_mismatch(tmp_path):
    path = tmp_path / "frames.npz"
    _small_frames(path)
    report = tmp_path / "audit.json"
    report.write_text(json.dumps({"schema": review.AUDIT_SCHEMA,
                                  "inputs": {"frames_sha256": "wrong",
                                             "sampled_frame_indices": [0, 1]},
                                  "policies": {"pooled": {"available": True, "frames": []}}}))

    with pytest.raises(ValueError, match="hash does not match"):
        review._load_inputs(path, report)


def test_load_inputs_refuses_policy_frame_count_mismatch(tmp_path):
    path = tmp_path / "frames.npz"
    _small_frames(path)
    audit = {"schema": review.AUDIT_SCHEMA,
             "inputs": {"frames_sha256": sha256_file(path), "sampled_frame_indices": [0, 1]},
             "policies": {"pooled": {"available": True, "frames": []}}}
    report = tmp_path / "audit.json"
    report.write_text(json.dumps(audit))

    with pytest.raises(ValueError, match="frame count does not match"):
        review._load_inputs(path, report)


def test_load_inputs_refuses_nonfinite_policy_correction(tmp_path):
    path = tmp_path / "frames.npz"
    _small_frames(path)
    entry = {"correction": np.eye(3).tolist()}
    audit = {"schema": review.AUDIT_SCHEMA,
             "inputs": {"frames_sha256": sha256_file(path), "sampled_frame_indices": [0, 1]},
             "policies": {"pooled": {"available": True, "frames": [entry, entry]}}}
    audit["policies"]["pooled"]["frames"][1]["correction"][0][0] = float("nan")
    report = tmp_path / "audit.json"
    report.write_text(json.dumps(audit))

    with pytest.raises(ValueError, match="invalid correction"):
        review._load_inputs(path, report)
