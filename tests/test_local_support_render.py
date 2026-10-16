"""Focused tests for the research-only localized-support renderer (``bench/local_support_render.py``).

These tests exercise only the pure-NumPy / pure-Python seams: the manifest timeline guard, the per-frame
blend/measure helper, the aggregate support math, the rational rate parser and the output-path guard. No MLX
compute, no media probe/decode/encode and no real render is touched, so the suite needs neither weights nor
footage. The full render path is intentionally left to integration/visual review.
"""
from fractions import Fraction

import numpy as np
import pytest

from bench import local_support_candidate as LSC
from bench import local_support_render as R


def _identity_map(height, width):
    y, x = np.mgrid[:height, :width]
    return np.stack([x, y], axis=-1).astype(np.float32)


# --------------------------------------------------------------------------------------
# timeline validation
# --------------------------------------------------------------------------------------

def test_validate_timeline_accepts_matching_metadata():
    R.validate_timeline({"width": 16, "height": 12, "fps": 30.0, "frames": 5},
                        {"width": 16, "height": 12, "fps": 30.0, "frames": 5})


@pytest.mark.parametrize("plan,decoded,code", [
    ({"width": 16, "height": 12, "fps": 30.0, "frames": 5},
     {"width": 8, "height": 12, "fps": 30.0, "frames": 5}, "timeline_dimensions"),
    ({"width": 16, "height": 12, "fps": 30.0, "frames": 5},
     {"width": 16, "height": 12, "fps": 25.0, "frames": 5}, "timeline_fps"),
    ({"width": 16, "height": 12, "fps": 30.0, "frames": 5},
     {"width": 16, "height": 12, "fps": 30.0, "frames": 4}, "timeline_frames"),
])
def test_validate_timeline_refuses_mismatch(plan, decoded, code):
    with pytest.raises(R.LocalSupportRenderError) as ei:
        R.validate_timeline(plan, decoded)
    assert ei.value.code == code


# --------------------------------------------------------------------------------------
# per-frame blend/measure helper
# --------------------------------------------------------------------------------------

def test_blend_frame_repairs_recoverable_failures_and_preserves_distant_map():
    H, W = 6, 10
    global_src = _identity_map(H, W)
    full_src = global_src.copy()
    full_src[..., 0] -= 2.0
    recoverable, unrecoverable = LSC.local_support_failure(full_src, global_src, W, H, kernel="bilinear")
    alpha = np.ones((H, W), np.float32)
    alpha[recoverable] = 0.0
    candidate, stats = R.blend_frame(full_src, global_src, alpha, W, H, kernel="bilinear")

    assert stats["frame_pixels"] == H * W
    assert stats["pre_invalid_pixels"] == int((recoverable | unrecoverable).sum()) > 0
    assert stats["recoverable_pixels"] == int(recoverable.sum())
    assert stats["unrecoverable_pixels"] == 0
    assert stats["post_invalid_pixels"] == 0
    assert stats["repaired_pixels"] == stats["pre_invalid_pixels"]
    assert stats["regressed_pixels"] == 0
    assert stats["affected_pixels"] == int(recoverable.sum())
    assert 0.0 <= stats["alpha_mean"] < 1.0
    # Distant pixels keep the complete map; repaired pixels fall back to the global-only map.
    np.testing.assert_array_equal(candidate[:, 6:], full_src[:, 6:])
    np.testing.assert_array_equal(candidate[recoverable], global_src[recoverable])


def test_blend_frame_kernel_changes_the_support_footprint():
    """The kernel-aware mask must reflect the sampler footprint, so a stricter kernel flags more pixels."""
    H, W = 8, 12
    global_src = _identity_map(H, W)
    full_src = global_src.copy()
    full_src[..., 0] -= 2.25                       # fractional shift: genuinely exercises the tap footprint
    alpha = np.ones((H, W), np.float32)
    _, bilinear = R.blend_frame(full_src, global_src, alpha, W, H, kernel="bilinear")
    _, lanczos = R.blend_frame(full_src, global_src, alpha, W, H, kernel="lanczos3")
    assert lanczos["pre_invalid_pixels"] >= bilinear["pre_invalid_pixels"]
    assert lanczos["pre_invalid_pixels"] > bilinear["pre_invalid_pixels"]


def test_blend_frame_counts_locally_unrecoverable_and_leaves_them_invalid():
    H, W = 5, 8
    global_src = _identity_map(H, W)
    global_src[..., 0] -= 1.0                      # global map itself loses column 0
    full_src = global_src.copy()
    full_src[..., 0] -= 2.0                        # complete map loses columns 0 and 1
    alpha = np.ones((H, W), np.float32)            # no correction: candidate == complete map
    candidate, stats = R.blend_frame(full_src, global_src, alpha, W, H, kernel="bilinear")

    expected_unfixable = int((~LSC._source_valid(global_src, W, H, "bilinear")).sum())
    assert stats["unrecoverable_pixels"] == expected_unfixable > 0
    assert stats["post_invalid_pixels"] >= stats["unrecoverable_pixels"]
    assert stats["affected_pixels"] == 0           # alpha is all ones, so nothing is locally active
    np.testing.assert_array_equal(candidate, full_src)


def test_blend_frame_replaces_nonfinite_with_global_and_reports_alpha():
    H, W = 4, 5
    global_src = _identity_map(H, W)
    full_src = global_src.copy()
    full_src[1, 2, 0] = np.nan
    alpha = np.ones((H, W), np.float32)
    alpha[1, 2] = 0.0
    candidate, stats = R.blend_frame(full_src, global_src, alpha, W, H, kernel="bilinear")

    assert np.isfinite(candidate).all()
    np.testing.assert_array_equal(candidate[1, 2], global_src[1, 2])
    assert stats["affected_pixels"] == 1
    assert stats["affected_fraction"] == pytest.approx(1 / (H * W))
    assert stats["alpha_min"] == 0.0 and stats["alpha_max"] == 1.0


def test_blend_frame_rejects_shape_mismatch():
    src = _identity_map(4, 5)
    with pytest.raises(ValueError):
        R.blend_frame(src, src[:, :-1], np.ones((4, 5), np.float32), 5, 4, kernel="bilinear")


# --------------------------------------------------------------------------------------
# aggregate support math
# --------------------------------------------------------------------------------------

def test_aggregate_support_recomputes_pooled_mean_max_and_alpha():
    H, W = 4, 6
    global_src = _identity_map(H, W)
    shifted = global_src.copy()
    shifted[..., 0] -= 2.0
    ok = global_src.copy()

    _, frame0 = R.blend_frame(shifted, global_src, np.zeros((H, W), np.float32), W, H, kernel="bilinear")
    _, frame1 = R.blend_frame(ok, global_src, np.ones((H, W), np.float32), W, H, kernel="bilinear")
    agg = R.aggregate_support([frame0, frame1], W, H)

    total = 2 * H * W
    assert agg["frames"] == 2 and agg["frame_pixels"] == H * W and agg["total_pixels"] == total
    assert agg["pre_invalid"]["total_pixels"] == frame0["pre_invalid_pixels"] + frame1["pre_invalid_pixels"]
    assert agg["pre_invalid"]["pooled_fraction"] == pytest.approx(agg["pre_invalid"]["total_pixels"] / total)
    assert agg["post_invalid"]["total_pixels"] == 0
    assert agg["recoverable"]["total_pixels"] == frame0["recoverable_pixels"]
    assert agg["affected"]["total_pixels"] == frame0["affected_pixels"]
    assert agg["repaired_pixels"] == frame0["repaired_pixels"] + frame1["repaired_pixels"]
    assert agg["regressed_pixels"] == 0
    assert agg["render_center_invalid"] is None  # pure map helper has no rendered validity mask
    assert agg["alpha"]["mean"] == pytest.approx((frame0["alpha_mean"] + frame1["alpha_mean"]) / 2)
    assert agg["alpha"]["max"] == pytest.approx(1.0)
    assert agg["pre_invalid"]["max_fraction"] >= agg["pre_invalid"]["mean_fraction"]


def test_aggregate_support_empty_is_defined():
    agg = R.aggregate_support([], 4, 6)
    assert agg["frames"] == 0 and agg["total_pixels"] == 0
    assert agg["pre_invalid"]["pooled_fraction"] == 0.0
    assert agg["alpha"] == {"mean": 0.0, "min": 0.0, "max": 0.0}


# --------------------------------------------------------------------------------------
# rate parsing and output-path guard
# --------------------------------------------------------------------------------------

def test_rate_parses_rational_exactly():
    assert R._rate("30000/1001", "output fps") == Fraction(30000, 1001)
    assert R._rate(30.0, "output fps") == Fraction(30, 1)


@pytest.mark.parametrize("bad", [0, -1, "0/1", "nan", None, "abc"])
def test_rate_refuses_nonpositive_or_unparseable(bad):
    with pytest.raises(R.LocalSupportRenderError) as ei:
        R._rate(bad, "output fps")
    assert ei.value.code == "bad_rate"


def test_resolve_outputs_defaults_report_and_refuses_collisions(tmp_path):
    man = tmp_path / "manifest.json"; man.write_text("{}", encoding="utf-8")
    src = tmp_path / "source.mp4"; src.write_bytes(b"src")
    plan = tmp_path / "plan_r0.npz"; plan.write_bytes(b"plan")
    out = tmp_path / "candidate.mp4"
    mask = tmp_path / "candidate_mask.mp4"

    resolved = R._resolve_outputs(man, src, plan, out, mask, None)
    assert resolved == (out.resolve(), mask.resolve(), out.resolve().with_suffix(".json"))

    with pytest.raises(R.LocalSupportRenderError) as ei:
        R._resolve_outputs(man, src, plan, out, out, None)
    assert ei.value.code == "output_collision"

    with pytest.raises(R.LocalSupportRenderError) as ei:
        R._resolve_outputs(man, src, plan, man, mask, None)
    assert ei.value.code == "output_collision"

    out.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        R._resolve_outputs(man, src, plan, out, mask, None)


# --------------------------------------------------------------------------------------
# CLI surface
# --------------------------------------------------------------------------------------

def test_cli_help_lists_the_documented_flags():
    text = R.build_parser().format_help()
    for token in ("--manifest", "--source", "--output", "--mask", "--report",
                  "--radius-px", "--temporal-radius", "--lock", "--lock-chunk", "--lock-pause"):
        assert token in text


@pytest.mark.parametrize("argv", [
    [],
    ["--manifest", "m.json"],                                   # missing --output and --mask
    ["--manifest", "m.json", "--mask", "m.mp4"],                # missing --output
    ["--manifest", "m.json", "--output", "o.mp4"],              # missing --mask
])
def test_cli_requires_manifest_output_and_mask(argv):
    with pytest.raises(SystemExit):
        R.main(argv)
