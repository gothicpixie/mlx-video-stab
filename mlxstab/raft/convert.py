"""Convert a RAFT .pth checkpoint (torch) to safetensors in this package's layout.

    python -I -m mlxstab.raft.convert raft-things.pth raft-things.safetensors

The .pth is a pickle: it is loaded with weights_only=True. Conv weights go [O,I,kh,kw] -> [O,kh,kw,I]."""
import sys


def convert_state_dict(sd):
    import numpy as np
    out = {}
    for k, v in sd.items():
        k = k[7:] if k.startswith("module.") else k
        if k.endswith("num_batches_tracked"): continue
        if ".norm3." in k and ".downsample." not in k: continue      # alias of downsample.1 (same module object in torch)
        a = v.detach().cpu().float().numpy()
        if a.ndim == 4: a = np.ascontiguousarray(a.transpose(0, 2, 3, 1))
        out[k] = a
    return out


def main(src, dst):
    import torch
    from safetensors.numpy import save_file
    sd = torch.load(src, map_location="cpu", weights_only=True)
    save_file(convert_state_dict(sd), dst)
    print(f"wrote {dst} ({len(sd)} tensors)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
