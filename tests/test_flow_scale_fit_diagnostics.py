from types import SimpleNamespace

from bench.flow_scale_fit_diagnostics import _block_summary


def test_fit_diagnostics_use_fixed_source_time_blocks_and_keep_fallbacks_visible():
    fits = [
        SimpleNamespace(fallback=(i == 60), fallback_reason="insufficient_valid_samples" if i == 60 else "none",
                        inlier_frac=0.0 if i == 60 else 0.8, mover_frac=0.1,
                        n_valid_samples=100, n_ransac_inliers=80)
        for i in range(61)
    ]
    cycles = [{"fb_valid_fraction": 0.9,
               "fb_error_over_tolerance_median": 0.5,
               "fb_error_over_tolerance_p90": 1.2} for _ in fits]

    blocks, by_block = _block_summary(fits, cycles, fps=60.0, block_seconds=1.0)
    assert [b["pairs"] for b in blocks] == [60, 1]
    assert blocks[0]["fallback_pairs"] == 0
    assert blocks[1]["fallback_pairs"] == 1
    assert blocks[1]["inlier_frac_min"] == 0.0
    assert by_block[1][0]["pair"] == 60
