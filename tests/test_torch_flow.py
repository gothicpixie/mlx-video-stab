"""PyTorch MPS flow adapter contract for the shared-pipeline backend comparison."""
import numpy as np
import pytest

from bench.torch_flow import DEFAULT_TORCH_WEIGHTS, TorchMPSFlow, verify_weight_equivalence


def test_weight_equivalence_requires_exact_converted_tensor_match(tmp_path):
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.numpy").save_file
    pth = tmp_path / "weights.pth"
    safe = tmp_path / "weights.safetensors"
    state = {"layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
             "layer.num_batches_tracked": torch.tensor(0)}
    torch.save(state, pth)
    save_file({"layer.weight": state["layer.weight"].numpy()}, safe)
    result = verify_weight_equivalence(safe, pth)
    assert result["exact_tensor_match"] and result["tensor_count"] == 1
    assert len(result["canonical_sha256"]) == 64

    save_file({"layer.weight": np.zeros((2, 2), np.float32)}, safe)
    with pytest.raises(ValueError, match="tensor differs"):
        verify_weight_equivalence(safe, pth)


def _mps_ready():
    try:
        import torch
        return torch.backends.mps.is_available()
    except ImportError:
        return False


@pytest.mark.skipif(not _mps_ready() or not DEFAULT_TORCH_WEIGHTS.is_file(),
                    reason="upstream RAFT weights and PyTorch MPS are required")
def test_adapter_matches_upstream_bidirectional_raft_and_resets_state():
    import torch

    flow = TorchMPSFlow(DEFAULT_TORCH_WEIGHTS, iters=2, downscale=1, warm=False)
    rng = np.random.default_rng(21)
    a = rng.integers(0, 256, (128, 160, 3), dtype=np.uint8)
    b = np.roll(a, shift=(1, -2), axis=(0, 1))
    got_f, got_b = flow.bidir(a, b)

    A, pad = flow._pad(flow._prep(a))
    B, _ = flow._pad(flow._prep(b))
    with torch.no_grad():
        _, reference = flow.model(torch.cat([A, B]), torch.cat([B, A]), iters=2, test_mode=True)
    ref_f = flow._restore(reference[:1], pad, *a.shape[:2])
    ref_b = flow._restore(reference[1:], pad, *a.shape[:2])
    np.testing.assert_allclose(np.asarray(got_f), np.asarray(ref_f), rtol=1e-3, atol=2e-2)
    np.testing.assert_allclose(np.asarray(got_b), np.asarray(ref_b), rtol=1e-3, atol=2e-2)

    assert flow.prev_f is not None and flow.prev_b is not None
    flow.reset()
    assert flow.prev_f is None and flow.prev_b is None and flow._last is None
