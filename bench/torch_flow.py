"""PyTorch MPS RAFT adapter for the shared stabilizer benchmark.

This swaps only the bidirectional flow backend. Planning, rendering, media I/O and
evaluation continue through the same mlxstab pipeline as the MLX backend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RAFT_CORE = ROOT / "vendor" / "reference" / "RAFT" / "core"
DEFAULT_TORCH_WEIGHTS = ROOT / "vendor" / "weights" / "things" / "raft-things.pth"


def verify_weight_equivalence(mlx_weights, torch_weights):
    """Require the MLX and PyTorch checkpoint files to contain exactly the same RAFT tensors."""
    import torch
    from safetensors.numpy import load_file
    from mlxstab.raft.convert import convert_state_dict

    torch_state = torch.load(str(torch_weights), map_location="cpu", weights_only=True)
    converted = convert_state_dict(torch_state)
    mlx_state = load_file(str(mlx_weights))
    left, right = set(converted), set(mlx_state)
    if left != right:
        missing = sorted(left - right)[:8]
        extra = sorted(right - left)[:8]
        raise ValueError(f"RAFT checkpoints contain different tensor keys (missing={missing}, extra={extra})")

    digest = hashlib.sha256()
    for key in sorted(left):
        a, b = np.asarray(converted[key]), np.asarray(mlx_state[key])
        if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
            raise ValueError(f"RAFT checkpoint tensor differs after conversion: {key}")
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(json.dumps(list(a.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(a.dtype.str.encode("ascii") + b"\0")
        digest.update(np.ascontiguousarray(a).view(np.uint8))
    return {"exact_tensor_match": True, "tensor_count": len(left), "canonical_sha256": digest.hexdigest()}


class TorchMPSFlow:
    """Flow-function adapter matching ``RaftFlow.bidir`` and ``reset``.

    It uses the upstream RAFT model and the same 8-pixel padding, average
    downscale, bidirectional batch, warm-start policy, and shared stabilizer
    stages. Results cross back to MLX because the rest of this pipeline is MLX.
    """

    def __init__(self, weights=DEFAULT_TORCH_WEIGHTS, iters=24, downscale=1,
                 warm=True, corr_dtype=None, dtype=None, warm_iters=None):
        del corr_dtype, dtype  # the upstream MPS reference path is fp32
        if not RAFT_CORE.is_dir():
            raise FileNotFoundError(f"vendored PyTorch RAFT is missing: {RAFT_CORE}")
        import torch

        if not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS is unavailable on this machine")
        if str(RAFT_CORE) not in sys.path:
            sys.path.insert(0, str(RAFT_CORE))
        from raft import RAFT
        from corr import CorrBlock

        self.torch = torch
        self.CorrBlock = CorrBlock
        self.device = torch.device("mps")
        self.iters = int(iters)
        self.ds = int(downscale)
        self.warm = bool(warm)
        self.warm_iters = warm_iters
        args = argparse.Namespace(small=False, mixed_precision=False, alternate_corr=False, dropout=0)
        self.model = RAFT(args)
        state = torch.load(str(weights), map_location="cpu", weights_only=True)
        self.model.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in state.items()})
        self.model = self.model.to(self.device).eval()
        self.reset()

    def reset(self):
        self.prev_f = self.prev_b = None
        self._last = None

    def _prep(self, image):
        x = np.asarray(image, dtype=np.float32)
        if self.ds > 1:
            h, w = x.shape[:2]
            h2, w2 = h // self.ds, w // self.ds
            x = x[:h2 * self.ds, :w2 * self.ds].reshape(
                h2, self.ds, w2, self.ds, 3
            ).mean(axis=(1, 3))
        return self.torch.from_numpy(x.transpose(2, 0, 1).copy())[None].to(self.device)

    def _pad(self, image):
        torch = self.torch
        _, _, h, w = image.shape
        ph = (((h // 8) + 1) * 8 - h) % 8
        pw = (((w // 8) + 1) * 8 - w) % 8
        pad = (ph // 2, ph - ph // 2, pw // 2, pw - pw // 2)  # top,bottom,left,right
        if any(pad):
            image = torch.nn.functional.pad(image, (pad[2], pad[3], pad[0], pad[1]), mode="replicate")
        return image, pad

    def _features(self, image):
        torch = self.torch
        normalized = (2.0 * (image / 255.0) - 1.0).contiguous()
        with torch.no_grad():
            fmap = self.model.fnet(normalized).float()
            context = self.model.cnet(normalized)
            net, inp = torch.split(context, [self.model.hidden_dim, self.model.context_dim], dim=1)
            return fmap, torch.tanh(net), torch.relu(inp)

    def _restore(self, flow_nchw, pad, height, width):
        from mlxstab.stabilize import RaftFlow

        nhwc = mx.array(flow_nchw.detach().to("cpu").numpy().transpose(0, 2, 3, 1))
        return RaftFlow._restore(self, nhwc, pad, height, width)

    def bidir(self, a, b):
        torch = self.torch
        height, width = a.shape[:2]
        A, pad = self._pad(self._prep(a))
        B, _ = self._pad(self._prep(b))
        if self._last is not None and self._last[0] is a:
            fa, na, ia = self._last[1]
        else:
            fa = na = ia = None
        if fa is None:
            fa, na, ia = self._features(A)
        fb, nb, ib = self._features(B)

        with torch.no_grad():
            corr = self.CorrBlock(torch.cat([fa, fb], dim=0), torch.cat([fb, fa], dim=0),
                                  num_levels=self.model.args.corr_levels,
                                  radius=self.model.args.corr_radius)
            image_pair = torch.cat([A, B], dim=0)
            coords0, coords1 = self.model.initialize_flow(image_pair)
            if self.warm and self.prev_f is not None and self.prev_b is not None:
                init = torch.cat([self.prev_f, self.prev_b], dim=0)
                coords1 = coords1 + init
            n = torch.cat([na, nb], dim=0)
            inp = torch.cat([ia, ib], dim=0)
            iterations = (self.warm_iters or self.iters) if self.warm and self.prev_f is not None else self.iters
            mask = None
            for _ in range(iterations):
                coords1 = coords1.detach()
                corr_now = corr(coords1)
                flow = coords1 - coords0
                n, mask, delta = self.model.update_block(n, inp, corr_now, flow)
                coords1 = coords1 + delta
            low = coords1 - coords0
            if mask is None:
                raise ValueError("RAFT requires at least one refinement iteration")
            up = self.model.upsample_flow(low, mask)
            self.prev_f, self.prev_b = low[:1].detach(), low[1:].detach()
            self._last = (b, (fb, nb, ib))

        return self._restore(up[:1], pad, height, width), self._restore(up[1:], pad, height, width)
