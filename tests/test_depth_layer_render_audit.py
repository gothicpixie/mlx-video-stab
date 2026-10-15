"""Pure-NumPy tests for the isolated rendered-validity audit.

The module is weight-free, media-free, asset-free and deterministic, so these tests run it directly: they
assert the deterministic textured render dimensions and finite variance, the foreground compositing z-order,
per-pixel render-validity masks, in-bounds and rendered-valid coverage in [0, 1] with rendered-valid never
exceeding in-bounds, deterministic output, separate policy versus pooled paths, inclusion of the non-affine
foreground term, the unavailable-policy fallbacks (uninformative/shuffled/missing rank) fabricating no render,
path-free JSON-safe reports and atomic CLI output.

Warnings are escalated to errors so an unclean numeric path fails rather than passing silently.
"""
import json
import warnings

import numpy as np
import pytest

from bench import depth_layer_render_audit as A
from bench import depth_layer_synthetic as D


@pytest.fixture(autouse=True)
def no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def _default_seq(tmp_path=None):
    return D.build_sequence(D.SyntheticSpec(W=48, H=36, T=6))


# --------------------------------------------------------------------------------------
# textures
# --------------------------------------------------------------------------------------

def test_textures_have_expected_shape_and_are_finite():
    tex = A.build_textures(48, 36)
    assert tex.shape == (2, 36, 48, 3)
    assert np.isfinite(tex).all()
    assert tex.min() >= 0.0 and tex.max() <= 1.0


def test_textures_are_nonconstant_with_positive_variance():
    tex = A.build_textures(48, 36)
    assert np.var(tex) > 0.0
    # each layer's own texture is non-constant too, so compositing is observable
    for l in range(2):
        assert np.var(tex[l]) > 0.0


def test_textures_are_deterministic():
    a = A.build_textures(48, 36)
    b = A.build_textures(48, 36)
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------------------
# forward splat rendering
# --------------------------------------------------------------------------------------

def test_render_frame_shape_and_validity_mask_dtype():
    seq = _default_seq()
    tex = A.build_textures(seq.W, seq.H)
    out = A.render_frame(seq, tex, 0)
    assert out["frame"].shape == (seq.H, seq.W, 3)
    assert out["valid"].shape == (seq.H, seq.W)
    assert out["valid"].dtype == bool
    assert out["frame"].dtype == np.float64


def test_render_frame_is_deterministic_and_finite():
    seq = _default_seq()
    tex = A.build_textures(seq.W, seq.H)
    a = A.render_frame(seq, tex, 2)
    b = A.render_frame(seq, tex, 2)
    np.testing.assert_array_equal(a["frame"], b["frame"])
    np.testing.assert_array_equal(a["valid"], b["valid"])
    assert np.isfinite(a["frame"]).all()


def test_render_frame_splat_count_matches_valid_mask():
    seq = _default_seq()
    tex = A.build_textures(seq.W, seq.H)
    out = A.render_frame(seq, tex, 1)
    assert out["splatted_pixels"] == int(out["valid"].sum())
    assert len(out["per_layer_splats"]) == seq.n_layers


def test_render_frame_leaves_holes_at_zero_rgb():
    seq = _default_seq()
    tex = A.build_textures(seq.W, seq.H)
    out = A.render_frame(seq, tex, 0)
    holes = ~out["valid"]
    assert holes.any()  # a non-surjective splat must leave holes at the default fixture
    assert np.all(out["frame"][holes] == 0.0)


def test_foreground_occludes_background_in_composite():
    # The foreground overlay is drawn on top: a destination pixel splatted by layer 1 must carry layer 1's
    # texture value, not layer 0's. Compare against a layer-1-only splat on those pixels.
    seq = _default_seq()
    tex = A.build_textures(seq.W, seq.H)
    out = A.render_frame(seq, tex, 0)
    tex_flat = tex.reshape(2, seq.H * seq.W, 3)
    pts = D.apply_affine(seq.layer_transform[1, 0], seq.coords)
    qx = np.rint(pts[:, 0]).astype(np.int64)
    qy = np.rint(pts[:, 1]).astype(np.int64)
    inside = (qx >= 0) & (qx < seq.W) & (qy >= 0) & (qy < seq.H)
    vis = seq.visible_layer.reshape(-1) == 1
    sel = np.flatnonzero(inside & vis)
    dest = qy[sel] * seq.W + qx[sel]
    # the last writer wins within the layer; check every destination carries some layer-1 value
    got = out["frame"].reshape(-1, 3)[dest]
    layer1_vals = tex_flat[1][sel]
    assert np.isclose(got, layer1_vals).all(axis=1).mean() > 0.99


# --------------------------------------------------------------------------------------
# pull-warp
# --------------------------------------------------------------------------------------

def test_pull_warp_identity_recovers_frame_where_valid():
    seq = _default_seq()
    tex = A.build_textures(seq.W, seq.H)
    out = A.render_frame(seq, tex, 0)
    warped = A.pull_warp(out["frame"], out["valid"], np.eye(3))
    # identity warp must sample the same integer pixel: wherever a pixel is valid in the source and the
    # conservative 4-tap neighbourhood is valid, the warped value equals the source value.
    got = warped["image"][warped["valid"]]
    src = out["frame"][warped["valid"]]
    np.testing.assert_allclose(got, src)


def test_pull_warp_valid_is_subset_of_in_bounds():
    seq = _default_seq()
    tex = A.build_textures(seq.W, seq.H)
    out = A.render_frame(seq, tex, 0)
    E = D._fit_policy(seq, 0, None)["E"]
    warped = A.pull_warp(out["frame"], out["valid"], E)
    assert warped["valid_count"] <= warped["in_bounds_count"]
    assert not np.any(warped["valid"] & ~warped["in_bounds"])


def test_pull_warp_counts_and_finite_fraction_are_bounded():
    seq = _default_seq()
    tex = A.build_textures(seq.W, seq.H)
    out = A.render_frame(seq, tex, 3)
    E = D._fit_policy(seq, 3, None)["E"]
    warped = A.pull_warp(out["frame"], out["valid"], E)
    assert 0 <= warped["in_bounds_count"] <= warped["total_pixels"]
    assert 0.0 <= warped["finite_frac"] <= 1.0
    assert np.isfinite(warped["image"]).all()


def test_pull_warp_out_of_bounds_transform_is_empty():
    seq = _default_seq()
    tex = A.build_textures(seq.W, seq.H)
    out = A.render_frame(seq, tex, 0)
    # a large translation pushes every sample out of bounds
    E = np.array([[1.0, 0.0, 1e6], [0.0, 1.0, 1e6], [0.0, 0.0, 1.0]])
    warped = A.pull_warp(out["frame"], out["valid"], E)
    assert warped["in_bounds_count"] == 0
    assert warped["valid_count"] == 0


# --------------------------------------------------------------------------------------
# report structure and coverage
# --------------------------------------------------------------------------------------

def test_report_policy_and_pool_are_separate_paths():
    rep = A.run_audit(D.SyntheticSpec(W=48, H=36, T=6))
    assert rep["baseline"]["scope"] == "pooled"
    assert rep["baseline"]["targets_layer"] is None
    assert [p["policy"] for p in rep["policies"]] == ["background_lock", "foreground_preserve"]
    # the pooled baseline estimates the same E[t] as a rank-selected policy only by coincidence; their
    # rendered-valid coverage is computed independently and both blocks are present.
    assert rep["baseline"]["render"] is not None
    p0, p1 = rep["policies"]
    assert p0["target_layer"] == 0 and p1["target_layer"] == 1
    assert p0["render"] is not None and p1["render"] is not None


def test_coverage_fractions_are_in_unit_interval_and_valid_le_inbounds():
    rep = A.run_audit(D.SyntheticSpec(W=48, H=36, T=6))
    blocks = [rep["baseline"]["render"]] + [p["render"] for p in rep["policies"] if p["render"]]
    for block in blocks:
        for f in block["frames"]:
            assert 0.0 <= f["in_bounds_frac"] <= 1.0
            assert 0.0 <= f["rendered_valid_frac"] <= 1.0
            assert f["valid_count"] <= f["in_bounds_count"]
            assert f["rendered_valid_frac"] <= f["in_bounds_frac"]
            assert f["finite_frac"] is not None and 0.0 <= f["finite_frac"] <= 1.0
        agg = block["aggregate"]
        assert agg["total_rendered_valid"] <= agg["total_in_bounds"]
        assert agg["in_bounds_frac"]["mean"] is not None


def test_report_is_deterministic_and_json_safe():
    a = A.dumps_report(A.run_audit(D.SyntheticSpec(W=48, H=36, T=6)))
    b = A.dumps_report(A.run_audit(D.SyntheticSpec(W=48, H=36, T=6)))
    assert a == b
    parsed = json.loads(a)
    assert parsed["schema"] == A.SCHEMA
    assert "NaN" not in a and "Infinity" not in a
    assert parsed["textures"]["finite"] is True
    assert parsed["textures"]["variance"] > 0.0


def test_report_contains_no_image_pixels():
    rep = A.run_audit(D.SyntheticSpec(W=48, H=36, T=6))
    assert rep["notes"]["no_image_pixels_in_report"] is True
    assert rep["textures"]["pixels_in_report"] is False
    blob = A.dumps_report(rep)
    low = blob.lower()
    assert "ndarray" not in low
    # per-frame records are scalar summaries only: no pixel buffer is stored under any frame record
    for f in rep["baseline"]["render"]["frames"]:
        assert isinstance(f["rgb_variance"], (float, type(None)))
        assert "image" not in f
        assert not isinstance(f.get("frame"), (list, tuple))


def test_report_is_path_free_and_declares_scope():
    blob = A.dumps_report(A.run_audit(D.SyntheticSpec(W=48, H=36, T=6))).lower()
    for token in ("/users/", "/home/", ".npz", ".mp4", ".png", "vendor/"):
        assert token not in blob
    rep = A.run_audit(D.SyntheticSpec(W=48, H=36, T=6))
    assert rep["notes"]["synthetic_only"] is True
    assert rep["notes"]["no_media_decoded"] is True
    assert rep["notes"]["no_assets_used"] is True
    assert rep["notes"]["no_production_code_imported"] is True
    assert rep["notes"]["winner_declared"] is False
    assert len(rep["limitations"]) >= 4


def test_report_records_rasterization_and_collision_policy():
    rep = A.run_audit(D.SyntheticSpec(W=48, H=36, T=6))
    assert rep["rasterization"] == A.RASTERIZATION
    assert rep["collision_policy"] == A.COLLISION_POLICY


def test_pooled_baseline_render_is_finite_and_has_variance():
    rep = A.run_audit(D.SyntheticSpec(W=48, H=36, T=6))
    block = rep["baseline"]["render"]
    variances = [f["rgb_variance"] for f in block["frames"] if f["rgb_variance"] is not None]
    assert variances and all(v >= 0.0 for v in variances)
    assert any(v > 0.0 for v in variances)


# --------------------------------------------------------------------------------------
# non-affine stress and unavailability
# --------------------------------------------------------------------------------------

def test_nonaffine_foreground_is_included_in_render():
    base = A.run_audit(D.SyntheticSpec(W=48, H=36, T=6))
    stress = A.run_audit(D.SyntheticSpec(W=48, H=36, T=6, fg_nonaffine_px=2.0))
    # the non-affine displacement moves foreground splats, so the composited/valid frames differ
    assert stress["spec"]["fg_nonaffine_px"] == 2.0
    assert base["baseline"]["render"]["aggregate"]["total_rendered_valid"] != \
        stress["baseline"]["render"]["aggregate"]["total_rendered_valid"] or \
        base["policies"][1]["render"]["aggregate"]["total_rendered_valid"] != \
        stress["policies"][1]["render"]["aggregate"]["total_rendered_valid"]


def test_stress_combined_settings_are_available_and_finite():
    rep = A.run_audit(D.SyntheticSpec(W=48, H=36, T=6, flow_noise_px=0.5, depth_error_frac=0.2,
                                      fg_nonaffine_px=2.0))
    assert rep["baseline"]["available"] is True
    for p in rep["policies"]:
        assert p["available"] is True and p["render"] is not None
        assert p["render"]["aggregate"]["rendered_valid_frac"]["mean"] is not None
    # JSON-serializable and finite
    assert json.loads(A.dumps_report(rep))["schema"] == A.SCHEMA


def test_uninformative_depth_policies_unavailable_without_render():
    rep = A.run_audit(D.SyntheticSpec(W=48, H=36, T=6, depth_mode="uninformative"))
    for p in rep["policies"]:
        assert p["available"] is False
        assert p["render"] is None
        assert p["reason"] in {"rank_shared_by_multiple_layers", "rank_not_present",
                               "rank_layer_association_destroyed"}
    # the pooled baseline remains a diagnostic
    assert rep["baseline"]["available"] is True
    assert rep["baseline"]["render"] is not None


def test_shuffled_depth_policies_unavailable_without_render():
    rep = A.run_audit(D.SyntheticSpec(W=48, H=36, T=6, depth_mode="shuffled"))
    for p in rep["policies"]:
        assert p["available"] is False
        assert p["reason"] == "rank_layer_association_destroyed"
        assert p["render"] is None


def test_missing_rank_policy_unavailable_without_render():
    rep = A.run_audit(D.SyntheticSpec(W=48, H=36, T=6, lock_rank=7, preserve_rank=1))
    bl = rep["policies"][0]
    assert bl["available"] is False and bl["reason"] == "rank_not_present"
    assert bl["render"] is None
    assert rep["policies"][1]["available"] is True


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def test_cli_main_writes_atomic_json(tmp_path):
    out = tmp_path / "report.json"
    rc = A.main(["--T", "6", "--W", "48", "--H", "36", "--out", str(out)])
    assert rc == 0
    saved = json.loads(out.read_text())
    assert saved["schema"] == A.SCHEMA
    assert saved == A.run_audit(D.SyntheticSpec(T=6, W=48, H=36))


def test_cli_main_dry_run_does_not_write(tmp_path, capsys):
    out = tmp_path / "report.json"
    rc = A.main(["--T", "6", "--W", "48", "--H", "36", "--out", str(out), "--dry-run"])
    assert rc == 0 and not out.exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["T"] == 6 and printed["depth_mode"] == "ranked"


def test_cli_main_reports_failure_as_json(capsys):
    rc = A.main(["--T", "2"])  # below the >= 3 floor
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failure" and payload["error"]["type"] == "ValueError"


def test_cli_roundtrips_nonaffine_and_stress_flags():
    a = A.build_parser().parse_args(["--fg-nonaffine-px", "2.0", "--flow-noise-px", "0.5",
                                     "--depth-error-frac", "0.2", "--fg-rect", "0.5,0.5,0.2,0.2"])
    spec = A.spec_from_args(a)
    assert spec.fg_nonaffine_px == 2.0
    assert spec.flow_noise_px == 0.5
    assert spec.depth_error_frac == 0.2
    assert spec.fg_rect == (0.5, 0.5, 0.2, 0.2)
    rep = A.run_audit(spec)
    assert rep["spec"]["fg_nonaffine_px"] == 2.0
