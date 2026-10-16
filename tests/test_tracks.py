"""Focused tests for the independent sparse-KLT evaluation estimator (``mlxstab.tracks``).

These are offline, deterministic and weight-free. Synthetic frames carry a *known* integer camera path, so
the recovered transforms can be checked directly. The point of the estimator is that it is a different optical
flow algorithm from RAFT; the tests therefore also pin the estimator's own failure accounting (a pair with too
few corners must be reported as a fallback, not scored as perfectly stable).
"""
import numpy as np
import pytest
import warnings

from mlxstab import geometry as G
from mlxstab import tracks

cv2 = pytest.importorskip("cv2")


# --------------------------------------------------------------------------------------
# fixtures: textured frames with a known integer camera path
# --------------------------------------------------------------------------------------

def _base(W, H, seed=0, pad=64):
    """A large high-texture canvas; cropping a window lets integer shifts avoid wrap seams."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (H + 2 * pad, W + 2 * pad, 3), dtype=np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0.6)


def _shifted(base, W, H, dx=0, dy=0):
    """Crop a W x H window whose content is shifted by (dx, dy) pixels (positive = right/down)."""
    pad_y = (base.shape[0] - H) // 2
    pad_x = (base.shape[1] - W) // 2
    y0, x0 = pad_y - int(dy), pad_x - int(dx)
    return np.ascontiguousarray(base[y0:y0 + H, x0:x0 + W])


def _gray(f):
    return cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)


def _disp(H, p):
    p = np.asarray(p, np.float64)
    return G.apply_h(H, p) - p


# --------------------------------------------------------------------------------------
# pair_similarity: direction, exactness, fallback
# --------------------------------------------------------------------------------------

def test_identity_pair_is_identity_and_supported():
    b = _base(96, 96, seed=1)
    f = _shifted(b, 96, 96)
    H, fb, inl, n = tracks.pair_similarity(_gray(f), _gray(f), rng=np.random.default_rng(0))
    assert not fb and n >= 8
    assert np.allclose(H, np.eye(3), atol=1e-6)
    assert inl > 0.9


def test_integer_translation_recovered_in_prev_to_next_direction():
    """A similarity must map a previous-frame point to its next-frame location (positive shift = right/down)."""
    b = _base(120, 120, seed=2)
    prev = _shifted(b, 120, 120)
    nxt = _shifted(b, 120, 120, dx=7, dy=-4)
    H, fb, inl, n = tracks.pair_similarity(_gray(prev), _gray(nxt), rng=np.random.default_rng(0))
    assert not fb
    d = _disp(H, [[60.0, 60.0]])[0]
    assert np.allclose(d, [7.0, -4.0], atol=0.5), d
    assert inl > 0.5 and n >= 8


def test_subpixel_translation_is_recovered_within_half_a_pixel():
    b = _base(140, 140, seed=3)
    prev = _shifted(b, 140, 140)
    M = np.float32([[1, 0, 3.4], [0, 1, -2.6]])
    nxt = cv2.warpAffine(prev, M, (140, 140), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    H, fb, inl, n = tracks.pair_similarity(_gray(prev), _gray(nxt), rng=np.random.default_rng(0))
    assert not fb
    d = _disp(H, [[70.0, 70.0]])[0]
    assert np.allclose(d, [3.4, -2.6], atol=0.5), d


def test_blank_frames_are_reported_as_a_fallback_not_static_success():
    blank = np.zeros((96, 96, 3), np.uint8)
    H, fb, inl, n = tracks.pair_similarity(_gray(blank), _gray(blank), rng=np.random.default_rng(0))
    assert fb and n == 0 and inl == 0.0
    assert np.allclose(H, np.eye(3))


def test_empty_ransac_support_returns_fallback_without_nan_fit_warning(monkeypatch):
    b = _base(120, 120, seed=12)
    f = _shifted(b, 120, 120)
    monkeypatch.setattr(tracks, "_ransac_similarity",
                        lambda src, dst, *a, **k: (np.eye(3), np.zeros(len(src), bool)))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        H, fb, inl, n = tracks.pair_similarity(_gray(f), _gray(f), rng=np.random.default_rng(0))
    assert fb and inl == 0.0 and n >= 8
    assert np.allclose(H, np.eye(3))


def test_invalid_destination_mask_rejects_track_and_masks_use_true_for_no_source():
    b = _base(120, 120, seed=9)
    f = _shifted(b, 120, 120)
    invalid = np.ones((120, 120), bool)
    H, fb, inl, n = tracks.pair_similarity(_gray(f), _gray(f), next_invalid=invalid,
                                           rng=np.random.default_rng(0))
    assert fb and inl == 0.0 and n == 0


def test_out_of_bounds_lk_endpoints_are_rejected_even_when_fb_cycle_closes(monkeypatch):
    b = _base(120, 120, seed=10)
    f = _shifted(b, 120, 120)
    points = np.array([[[x, y]] for y in range(20, 101, 20) for x in range(20, 101, 20)], np.float32)
    monkeypatch.setattr(cv2, "goodFeaturesToTrack", lambda *a, **k: points)
    calls = {"n": 0}
    def fake_lk(_src, _dst, p, *_a, **_k):
        calls["n"] += 1
        q = p + 500 if calls["n"] == 1 else p - 500
        return q, np.ones((len(p), 1), np.uint8), None
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fake_lk)
    H, fb, inl, n = tracks.pair_similarity(_gray(f), _gray(f), rng=np.random.default_rng(0))
    assert fb and inl == 0.0 and n == 0
    assert np.allclose(H, np.eye(3))


def test_tracker_rejects_invalid_mask_shape():
    f = np.zeros((96, 96, 3), np.uint8)
    acc = tracks.TrackAccumulator(96, 96, 30.0)
    with pytest.raises(ValueError, match="invalid_mask must have shape"):
        acc.push(f, np.zeros((95, 96), bool))


def test_ransac_similarity_isolates_outliers_and_keeps_inliers_empty_below_floor():
    rng = np.random.default_rng(11)
    src = rng.uniform(0, 200, (60, 2))
    truth = np.array([[0.98, -0.1, 5.0], [0.1, 0.98, -3.0], [0.0, 0.0, 1.0]])
    dst = G.apply_h(truth, src) + rng.normal(0, 0.1, src.shape)
    bad = rng.choice(60, 15, replace=False)
    dst[bad] += rng.normal(0, 40.0, (15, 2))
    H, inl = tracks._ransac_similarity(src, dst, thr=1.0, rng=np.random.default_rng(0), min_inliers=8)
    assert inl.sum() >= 40
    d = _disp(H, [[100.0, 100.0]])[0] - _disp(truth, [[100.0, 100.0]])[0]
    assert np.allclose(d, 0.0, atol=0.5), d
    # Too few points to satisfy the floor: identity + empty mask, never a spurious fit.
    H2, inl2 = tracks._ransac_similarity(src[:3], dst[:3], thr=0.1, rng=np.random.default_rng(0), min_inliers=8)
    assert inl2.sum() == 0 and np.allclose(H2, np.eye(3))


# --------------------------------------------------------------------------------------
# TrackAccumulator: single-pass plumbing and honest failure accounting
# --------------------------------------------------------------------------------------

def test_accumulator_wrap_yields_frames_unchanged_single_pass():
    b = _base(96, 96, seed=4)
    frames = [_shifted(b, 96, 96, dx=t) for t in range(4)]
    acc = tracks.TrackAccumulator(96, 96, 30.0)
    out = list(acc.wrap(frames))
    assert len(out) == 4 and all(o is f for o, f in zip(out, frames))
    assert acc.n_frames == 4 and len(acc.transforms) == 3


def test_accumulator_wrap_keeps_frame_and_invalid_mask_streams_in_lockstep():
    b = _base(96, 96, seed=14)
    frames = [_shifted(b, 96, 96) for _ in range(4)]
    masks = [np.ones((96, 96), bool)] * 4
    acc = tracks.TrackAccumulator(96, 96, 30.0)
    assert len(list(acc.wrap(frames, masks))) == 4
    assert acc.n_frames == 4 and len(acc.transforms) == 3
    assert all(fb for _, fb, _, _ in acc.transforms)


def test_smooth_pan_has_near_zero_jitter_while_shake_does_not():
    b = _base(160, 120, seed=5)
    W = H = 120
    pan = [_shifted(b, W, H, dx=t * 2) for t in range(24)]
    shake = [_shifted(b, W, H, dx=t * 2 + (2 if t % 2 else -2)) for t in range(24)]
    j_smooth = tracks.track_metrics(pan, W, H, 30.0)["jitter_px"]
    j_shake = tracks.track_metrics(shake, W, H, 30.0)["jitter_px"]
    assert j_smooth < 0.05 and j_shake > 1.0 and j_shake > 5 * max(j_smooth, 1e-3)


def test_static_scene_is_reported_perfectly_stable():
    b = _base(120, 120, seed=6)
    f = _shifted(b, 120, 120)
    m = tracks.track_metrics([f] * 12, 120, 120, 30.0)
    assert m["fallback_frames"] == 0
    assert m["jitter_px"] == pytest.approx(0.0, abs=1e-6)
    assert m["stability"] == pytest.approx(1.0)


def test_blank_sequence_metrics_are_unavailable_instead_of_scoring_fallbacks_as_stable():
    blank = np.zeros((96, 96, 3), np.uint8)
    m = tracks.track_metrics([blank] * 5, 96, 96, 30.0)
    assert m["fallback_frames"] == 4 and m["support_frames"] == 4
    assert m["supported_frames"] == 0
    assert m["jitter_px"] is None and m["stability"] is None
    assert m["jitter_px_supported"] is None
    assert "failed pair" in m["unavailable_reason"]


def test_partial_tracking_support_does_not_join_across_a_failed_pair(monkeypatch):
    """Dropping a failed transform would falsely make its neighbors look temporally adjacent."""
    base = _base(96, 96, seed=8)
    frames = [_shifted(base, 96, 96, dx=i) for i in range(5)]
    H = np.eye(3)
    calls = iter([(H, False, 1.0, 20), (H, True, 0.0, 0),
                  (H, False, 1.0, 20), (H, False, 1.0, 20)])
    monkeypatch.setattr(tracks, "pair_similarity", lambda *a, **k: next(calls))
    acc = tracks.TrackAccumulator(96, 96, 30.0)
    for frame in frames:
        acc.push(frame)
    m = acc.metrics()
    assert m["fallback_frames"] == 1
    assert m["supported_frames"] == 3
    assert m["jitter_px"] is None and m["jitter_px_supported"] is None


def test_too_short_sequence_is_explicitly_unavailable():
    b = _base(96, 96, seed=7)
    f = _shifted(b, 96, 96)
    m = tracks.track_metrics([f], 96, 96, 30.0)
    assert m["jitter_px"] is None and m["stability"] is None
    assert m["unavailable_reason"]


# --------------------------------------------------------------------------------------
# agreement: two estimators, direction of the input -> output change
# --------------------------------------------------------------------------------------

def _m(jitter, fallback=0):
    return {"jitter_px": jitter, "fallback_frames": fallback, "support_frames": 10, "inlier_frac_mean": 0.9}


def test_agreement_is_reliable_when_both_estimators_see_the_same_improvement():
    a = tracks.agreement(_m(2.0), _m(1.0), _m(2.0), _m(1.1))
    assert a["raft_direction"] == "improves" and a["track_direction"] == "improves"
    assert a["agree_on_direction"] is True and a["reliable"] is True
    assert a["raft_jitter_ratio"] == pytest.approx(0.5)

def test_agreement_reports_direction_disagreement():
    a = tracks.agreement(_m(2.0), _m(1.0), _m(2.0), _m(3.0))
    assert a["raft_direction"] == "improves" and a["track_direction"] == "worsens"
    assert a["agree_on_direction"] is False and a["reliable"] is True   # disagreement is an interpretable finding


def test_agreement_treats_small_changes_as_unchanged():
    a = tracks.agreement(_m(1.0), _m(1.02), _m(1.0), _m(0.99))
    assert a["raft_direction"] == "unchanged" and a["track_direction"] == "unchanged"
    assert a["agree_on_direction"] is True


def test_agreement_flags_failed_fits_low_input_jitter_and_missing_track_metrics():
    fb = tracks.agreement(_m(2.0, fallback=1), _m(1.0), _m(2.0), _m(1.0))
    assert fb["reliable"] is False and any("failed fit" in r for r in fb["reasons"])

    low = tracks.agreement(_m(0.01), _m(0.005), _m(0.01), _m(0.005))
    assert low["raft_jitter_ratio"] is None and any("ratio floor" in r for r in low["reasons"])

    missing = tracks.agreement(_m(2.0), _m(1.0), None, None)
    assert missing["track_jitter_ratio"] is None and missing["reliable"] is False

    nan = tracks.agreement(_m(2.0), {"jitter_px": float("nan"), "fallback_frames": 0}, _m(2.0), _m(1.0))
    assert nan["raft_jitter_ratio"] is None and any("non-finite" in r for r in nan["reasons"])


def test_agreement_does_not_import_the_stabilizer_flow_stack():
    """The estimator must stay independent of the production RAFT/stabilize import path."""
    import mlxstab.tracks as t
    src = open(t.__file__).read()
    assert "RaftFlow" not in src and "from .stabilize" not in src and "import stabilize" not in src
