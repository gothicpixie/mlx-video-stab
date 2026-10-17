from types import SimpleNamespace

import numpy as np

from bench.flow_scale_localization import block_summaries


def test_blocks_partition_pairs_and_reconstruct_whole_window_jitter():
    fits = []
    for i in range(8):
        H = np.eye(3)
        H[0, 2] = float((i % 3) - 1)
        H[1, 2] = float(i % 2)
        rows = np.stack([np.arange(4) * 0.1, np.arange(4) * -0.2], axis=1)
        fits.append(SimpleNamespace(H=H, rows=rows, fallback=False))
    mot = SimpleNamespace(W=640, H=360, fps=30.0, fits=fits)

    result = block_summaries(mot, block_count=4)
    assert [b["pairs"] for b in result["blocks"]] == [2, 2, 2, 2]
    assert result["whole_window"]["support_pairs"] == 8
    weighted_block_mean = np.average(
        [b["jitter_px"] for b in result["blocks"]],
        weights=[b["pairs"] for b in result["blocks"]],
    )
    assert np.isclose(weighted_block_mean, result["whole_window"]["jitter_px"])


def test_wobble_blocks_exclude_fallback_fits_like_whole_window_metric():
    rows = np.stack([np.arange(4) * 0.1, np.arange(4) * -0.2], axis=1)
    fits = [SimpleNamespace(H=np.eye(3), rows=rows, fallback=False),
            SimpleNamespace(H=np.eye(3), rows=rows, fallback=True)]
    mot = SimpleNamespace(W=640, H=360, fps=30.0, fits=fits)

    result = block_summaries(mot, block_count=2)
    assert result["whole_window"]["fallback_pairs"] == 1
    assert result["blocks"][0]["wobble_px"] is not None
    assert result["blocks"][1]["wobble_px"] is None
