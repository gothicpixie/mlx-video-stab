"""Tests for the experimental luminance/high-pass KLT factorial audit."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from bench import track_domain_audit as audit
from mlxstab import tracks


def _texture(size=128, seed=4):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (size, size), dtype=np.uint8)
    return np.repeat(base[..., None], 3, axis=2)


def test_pair_similarity_domains_separates_detection_and_lk_images(monkeypatch):
    detect = np.full((96, 96), 37, np.uint8)
    prev_track = np.full((96, 96), 91, np.uint8)
    next_track = np.full((96, 96), 143, np.uint8)
    points = np.array([[[x, y]] for y in range(15, 81, 15) for x in range(15, 81, 15)], np.float32)
    calls = []

    def gftt(image, **_kwargs):
        calls.append(("detect", image.copy()))
        return points.copy()

    def lk(src, dst, p, *_args, **_kwargs):
        calls.append(("lk", src.copy(), dst.copy()))
        return p.copy(), np.ones((len(p), 1), np.uint8), None

    monkeypatch.setattr(cv2, "goodFeaturesToTrack", gftt)
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", lk)
    H, fallback, inlier_frac, n_tracks, detail = tracks.pair_similarity_domains(
        detect, prev_track, next_track, rng=np.random.default_rng(0))

    assert not fallback
    assert inlier_frac == 1.0
    assert n_tracks == len(points)
    assert detail["n_detected"] == len(points)
    assert detail["n_inliers"] == len(points)
    assert np.allclose(H, np.eye(3))
    assert np.array_equal(calls[0][1], detect)
    assert np.array_equal(calls[1][1], prev_track)
    assert np.array_equal(calls[1][2], next_track)
    assert np.array_equal(calls[2][1], next_track)
    assert np.array_equal(calls[2][2], prev_track)


def test_pair_similarity_luminance_wrapper_preserves_domain_pair_behavior():
    prev = tracks._gray(_texture())
    nxt = np.roll(prev, 2, axis=1)
    legacy = tracks.pair_similarity(prev, nxt, rng=np.random.default_rng(7))
    domain = tracks.pair_similarity_domains(prev, prev, nxt, rng=np.random.default_rng(7))[:4]
    assert legacy[1:] == domain[1:]
    assert np.array_equal(legacy[0], domain[0])


def test_factorial_report_has_four_cells_and_paired_contrasts_on_common_support():
    base = _texture(size=160, seed=8)
    frames = []
    for i in range(12):
        matrix = np.float32([[1, 0, 1.2 * i], [0, 1, 0.4 * i]])
        frames.append(cv2.warpAffine(base, matrix, (160, 160), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT))
    result = audit.run_audit(np.asarray(frames), fps=30.0, hp_sigma=1.2, seed=3,
                             tracker_kwargs={"max_corners": 120, "min_distance": 5.0})

    assert set(result["cells"]) == {"LL", "LH", "HL", "HH"}
    assert result["window"]["pairs"] == 11
    assert result["representation"]["hp_scale"] > 0
    assert result["contrasts"]["common_supported_pairs"] <= 11
    if result["contrasts"]["common_supported_pairs"] == 11:
        assert len(result["contrasts"]["track_main_effect_pair_values_px"]) == 11
        assert len(result["contrasts"]["interaction_pair_values_px"]) == 11
    assert any("observational" in line for line in result["interpretation"])


def test_invalid_support_is_applied_to_every_domain_and_never_scored_as_identity():
    frames = np.stack([_texture(size=96, seed=i) for i in range(5)])
    masks = np.ones((5, 96, 96), bool)
    result = audit.run_audit(frames, fps=30.0, invalid_masks=masks)
    assert result["validity_mask"]["invalid_fraction_mean"] == 1.0
    for cell in result["cells"].values():
        assert cell["fallback_pairs"] == 4
        assert cell["jitter_px"] is None
    assert result["contrasts"]["common_supported_pairs"] == 0
    assert result["contrasts"]["track_main_effect_hp_minus_lum_px"] is None
