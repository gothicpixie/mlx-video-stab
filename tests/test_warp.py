import numpy as np, pytest
import mlx.core as mx
torch = pytest.importorskip("torch")
import torch.nn.functional as F
from mlxstab.warp import grid_sample


@pytest.mark.parametrize("pad", ["zeros", "border"])
@pytest.mark.parametrize("ac", [True, False])
def test_grid_sample_parity(pad, ac):
    rng = np.random.default_rng(0)
    img = rng.standard_normal((2, 3, 17, 23)).astype(np.float32)          # torch NCHW
    grid = rng.uniform(-1.3, 1.3, (2, 11, 13, 2)).astype(np.float32)      # includes out-of-range samples
    ref = F.grid_sample(torch.from_numpy(img), torch.from_numpy(grid), mode="bilinear", padding_mode=pad, align_corners=ac).numpy()
    got = np.array(grid_sample(mx.array(img.transpose(0, 2, 3, 1)), mx.array(grid), pad, ac)).transpose(0, 3, 1, 2)
    assert np.abs(got - ref).max() < 1e-4


def test_identity_and_kernels():
    from mlxstab.warp import sample
    rng = np.random.default_rng(1); img = mx.array(rng.random((1, 16, 20, 3), dtype=np.float32))
    ys, xs = mx.meshgrid(mx.arange(16), mx.arange(20), indexing="ij")
    c = mx.stack([xs, ys], -1)[None].astype(mx.float32)
    for m in ("bilinear", "bicubic", "lanczos3"):
        assert np.abs(np.array(sample(img, c, m, "border") - img)).max() < 1e-5      # integer coords reproduce the image
