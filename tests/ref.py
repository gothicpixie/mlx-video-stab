"""Loads the PyTorch RAFT reference from vendor/reference (untracked download). Tests skip when it is absent."""
import sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "vendor/reference/RAFT/core"
WEIGHTS = ROOT / "vendor/weights/things/raft-things.pth"
DEMO = ROOT / "vendor/reference/RAFT/demo-frames"
available = CORE.exists()


def torch_raft(random_init_seed=None):
    import torch
    sys.path.insert(0, str(CORE))
    from raft import RAFT
    m = RAFT(argparse.Namespace(small=False, mixed_precision=False))
    if random_init_seed is None:
        sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        m.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in sd.items()})
    else:
        torch.manual_seed(random_init_seed)
        for mod in m.modules():                                   # non-trivial BN statistics
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.running_mean.normal_(0, .1); mod.running_var.uniform_(.5, 1.5); mod.weight.data.uniform_(.5, 1.5); mod.bias.data.normal_(0, .1)
        for mod in m.modules():                                   # keep the GRU well-conditioned
            if isinstance(mod, torch.nn.Conv2d) and mod.out_channels in (2,): mod.weight.data *= .1
    return m.eval()


def convert_in_memory(tm):
    import numpy as np, mlx.core as mx
    from mlxstab.raft.convert import convert_state_dict
    from mlxstab.raft.model import RAFT
    m = RAFT(); m.load_weights(list({k: mx.array(v) for k, v in convert_state_dict(tm.state_dict()).items()}.items()), strict=True)
    m.eval(); return m
