"""RAFT (full 'things' architecture, Teed & Deng 2020, BSD-3) in MLX. NHWC layout; inference only.

Parameter names mirror the PyTorch reference so a converted state dict loads 1:1 (see convert.py)."""
import math
import mlx.core as mx
import mlx.nn as nn
from ..warp import sample


class InstanceNorm(nn.Module):
    def __init__(self, eps=1e-5): super().__init__(); self.eps = eps
    def __call__(self, x):
        m = mx.mean(x, axis=(1, 2), keepdims=True); v = mx.var(x, axis=(1, 2), keepdims=True)
        return (x - m) * mx.rsqrt(v + self.eps)


def _norm(kind, c):
    if kind == "instance": return InstanceNorm()
    if kind == "batch": return nn.BatchNorm(c)
    return nn.Identity()


class ResidualBlock(nn.Module):
    def __init__(self, cin, c, norm, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, c, 3, stride=stride, padding=1); self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.norm1 = _norm(norm, c); self.norm2 = _norm(norm, c)
        self.downsample = None if stride == 1 else [nn.Conv2d(cin, c, 1, stride=stride), _norm(norm, c)]
    def __call__(self, x):
        y = nn.relu(self.norm1(self.conv1(x))); y = nn.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None: x = self.downsample[1](self.downsample[0](x))
        return nn.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, out_dim, norm):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3); self.norm1 = _norm(norm, 64)
        self.layer1 = [ResidualBlock(64, 64, norm), ResidualBlock(64, 64, norm)]
        self.layer2 = [ResidualBlock(64, 96, norm, 2), ResidualBlock(96, 96, norm)]
        self.layer3 = [ResidualBlock(96, 128, norm, 2), ResidualBlock(128, 128, norm)]
        self.conv2 = nn.Conv2d(128, out_dim, 1)
    def __call__(self, x):
        x = nn.relu(self.norm1(self.conv1(x)))
        for l in self.layer1 + self.layer2 + self.layer3: x = l(x)
        return self.conv2(x)


class MotionEncoder(nn.Module):
    def __init__(self, cor_planes):
        super().__init__()
        self.convc1 = nn.Conv2d(cor_planes, 256, 1); self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3); self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 126, 3, padding=1)
    def __call__(self, flow, corr):
        cor = nn.relu(self.convc2(nn.relu(self.convc1(corr))))
        flo = nn.relu(self.convf2(nn.relu(self.convf1(flow))))
        out = nn.relu(self.conv(mx.concatenate([cor, flo], -1)))
        return mx.concatenate([out, flow], -1)


class SepConvGRU(nn.Module):
    def __init__(self, hd=128, idim=256):
        super().__init__()
        c = hd + idim
        self.convz1 = nn.Conv2d(c, hd, (1, 5), padding=(0, 2)); self.convr1 = nn.Conv2d(c, hd, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(c, hd, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(c, hd, (5, 1), padding=(2, 0)); self.convr2 = nn.Conv2d(c, hd, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(c, hd, (5, 1), padding=(2, 0))
    def __call__(self, h, x):
        for z_, r_, q_ in ((self.convz1, self.convr1, self.convq1), (self.convz2, self.convr2, self.convq2)):
            hx = mx.concatenate([h, x], -1)
            z = mx.sigmoid(z_(hx)); r = mx.sigmoid(r_(hx))
            q = mx.tanh(q_(mx.concatenate([r * h, x], -1)))
            h = (1 - z) * h + z * q
        return h


class FlowHead(nn.Module):
    def __init__(self, i=128, h=256):
        super().__init__(); self.conv1 = nn.Conv2d(i, h, 3, padding=1); self.conv2 = nn.Conv2d(h, 2, 3, padding=1)
    def __call__(self, x): return self.conv2(nn.relu(self.conv1(x)))


class BasicUpdateBlock(nn.Module):
    def __init__(self, cor_planes, hd=128):
        super().__init__()
        self.encoder = MotionEncoder(cor_planes); self.gru = SepConvGRU(hd, 128 + hd); self.flow_head = FlowHead(hd, 256)
        self.mask = [nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 64 * 9, 1)]
    def __call__(self, net, inp, corr, flow):
        mf = self.encoder(flow, corr)
        net = self.gru(net, mx.concatenate([inp, mf], -1))
        m = .25 * self.mask[2](self.mask[1](self.mask[0](net)))
        return net, m, self.flow_head(net)


def _pool2(c):
    """avg_pool2d(2) on dims 1,2 of [B,h,w,1] (floor like torch)."""
    B, h, w, _ = c.shape; h2, w2 = h // 2, w // 2
    return c[:, :h2 * 2, :w2 * 2].reshape(B, h2, 2, w2, 2, 1).mean(axis=(2, 4))


class CorrBlock:
    """All-pairs correlation pyramid. corr_pyramid[i]: [h*w, h/2^i, w/2^i, 1]."""
    def __init__(self, f1, f2, levels=4, radius=4, dtype=mx.float32):
        self.levels, self.radius = levels, radius
        B, h, w, C = f1.shape; assert B == 1
        a = f1.reshape(h * w, C).astype(dtype); b = f2.reshape(h * w, C).astype(dtype)
        corr = ((a @ b.T) / math.sqrt(C)).reshape(h * w, h, w, 1)
        self.pyr = [corr]
        for _ in range(levels - 1): corr = _pool2(corr); self.pyr.append(corr)
        d = mx.arange(-radius, radius + 1).astype(mx.float32)
        # reference quirk kept: meshgrid(dy, dx) is added to (x, y), so offset index a is applied to x and b to y
        self.delta = mx.stack([mx.broadcast_to(d[:, None], (2 * radius + 1,) * 2), mx.broadcast_to(d[None, :], (2 * radius + 1,) * 2)], -1)

    def __call__(self, coords):                       # coords [1,h,w,2] (x,y)
        _, h, w, _ = coords.shape; out = []
        cen = coords.reshape(h * w, 1, 1, 2)
        for i in range(self.levels):
            g = cen / 2 ** i + self.delta[None]
            s = sample(self.pyr[i].astype(mx.float32), g, "bilinear", "zeros")           # [hw, 9, 9, 1]
            out.append(s.reshape(1, h, w, -1))
        return mx.concatenate(out, -1)


def coords_grid(h, w):
    ys, xs = mx.meshgrid(mx.arange(h), mx.arange(w), indexing="ij")
    return mx.stack([xs, ys], -1).astype(mx.float32)[None]


def upsample_flow(flow, mask):
    """Convex 8x upsampling. flow [1,h,w,2], mask [1,h,w,576] -> [1,8h,8w,2]."""
    _, h, w, _ = flow.shape
    m = mx.softmax(mask.reshape(h, w, 9, 8, 8), axis=2)
    f = mx.pad(8 * flow[0], [(1, 1), (1, 1), (0, 0)])
    nb = mx.stack([f[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)], 2)      # [h,w,9,2]
    up = mx.einsum("hwkij,hwkc->hiwjc", m, nb)
    return up.reshape(1, 8 * h, 8 * w, 2)


class RAFT(nn.Module):
    def __init__(self, corr_levels=4, corr_radius=4):
        super().__init__()
        self.hdim = self.cdim = 128; self.levels, self.radius = corr_levels, corr_radius
        self.fnet = BasicEncoder(256, "instance"); self.cnet = BasicEncoder(256, "batch")
        self.update_block = BasicUpdateBlock(corr_levels * (2 * corr_radius + 1) ** 2, self.hdim)

    def encode(self, im1, im2):
        """im 0..255 float [1,H,W,3], H,W multiples of 8. Returns fmap1, fmap2, net, inp."""
        im1 = 2 * (im1 / 255.0) - 1; im2 = 2 * (im2 / 255.0) - 1
        f = self.fnet(mx.concatenate([im1, im2], 0))
        c = self.cnet(im1)
        return f[:1], f[1:], mx.tanh(c[..., :self.hdim]), nn.relu(c[..., self.hdim:])

    def __call__(self, im1, im2, iters=12, flow_init=None, corr_dtype=mx.float32, return_all=False):
        """Returns (low-res flow [1,h,w,2], full-res flow [1,H,W,2]) in pixels (x, y). flow_init is low-res."""
        f1, f2, net, inp = self.encode(im1, im2)
        corr_fn = CorrBlock(f1, f2, self.levels, self.radius, corr_dtype)
        _, h, w, _ = f1.shape
        c0 = coords_grid(h, w); c1 = c0 if flow_init is None else c0 + flow_init
        preds = []
        for _ in range(iters):
            corr = corr_fn(c1)
            net, up_mask, delta = self.update_block(net, inp, corr, c1 - c0)
            c1 = c1 + delta
            mx.eval(c1, net)
            if return_all: preds.append(upsample_flow(c1 - c0, up_mask))
        flow_up = upsample_flow(c1 - c0, up_mask)
        return c1 - c0, flow_up


def pad8(x):
    """Replicate-pad [1,H,W,C] to multiples of 8 (symmetric, like the reference InputPadder). Returns (x, (top, bottom, left, right))."""
    H, W = x.shape[1:3]; ph = (((H // 8) + 1) * 8 - H) % 8; pw = (((W // 8) + 1) * 8 - W) % 8
    p = (ph // 2, ph - ph // 2, pw // 2, pw - pw // 2)
    if sum(p): x = mx.pad(x, [(0, 0), (p[0], p[1]), (p[2], p[3]), (0, 0)], mode="edge")
    return x, p


def unpad(x, p):
    H, W = x.shape[1:3]
    return x[:, p[0]:H - p[1], p[2]:W - p[3]]


def load_raft(path):
    m = RAFT(); m.load_weights(str(path), strict=True); m.eval(); return m


def raft_flow(model, im1, im2, iters=12, flow_init=None, **kw):
    """Full-resolution flow im1 -> im2 for uint8/float [H,W,3] arrays (0..255). Returns (H,W,2) mx.array."""
    a, p = pad8(mx.array(im1).astype(mx.float32)[None]); b, _ = pad8(mx.array(im2).astype(mx.float32)[None])
    low, up = model(a, b, iters, flow_init, **kw)
    return unpad(up, p)[0]
