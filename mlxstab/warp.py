"""Image resampling as an MLX Metal kernel.

``sample`` works in pixel coordinates (pixel centres on integers) and supports bilinear (taps=2), Catmull-Rom bicubic
(taps=4) and Lanczos-3 (taps=6) kernels with zeros or border padding. ``grid_sample`` is a torch.nn.functional.grid_sample
look-alike (NHWC layout) on top of it, bilinear only, used for the parity test. Forward only: no backward pass.
"""
import mlx.core as mx

_HEADER = """
constant float PI_F = 3.14159265358979f;
inline float sinc_f(float x) { x = fabs(x); if (x < 1e-6f) return 1.0f; float t = PI_F * x; return metal::precise::sin(t) / t; }
inline float tap_w(int mode, float d) {
    float a = fabs(d);
    if (mode == 0) return a < 1.0f ? 1.0f - a : 0.0f;
    if (mode == 1) {                       // Catmull-Rom, a = -0.5
        if (a < 1.0f) return 1.5f * a * a * a - 2.5f * a * a + 1.0f;
        if (a < 2.0f) return -0.5f * a * a * a + 2.5f * a * a - 4.0f * a + 2.0f;
        return 0.0f;
    }
    return a < 3.0f ? sinc_f(a) * sinc_f(a / 3.0f) : 0.0f;   // Lanczos-3
}
"""

_SRC = """
    uint x = thread_position_in_grid.x, y = thread_position_in_grid.y, n = thread_position_in_grid.z;
    int H = img_shape[1], W = img_shape[2], C = img_shape[3];
    int Ho = coords_shape[1], Wo = coords_shape[2];
    if ((int)x >= Wo || (int)y >= Ho) return;
    size_t ci = (((size_t)n * Ho + y) * Wo + x) * 2;
    float px = coords[ci], py = coords[ci + 1];
    const int T = MODE == 0 ? 2 : (MODE == 1 ? 4 : 6);
    float fx = metal::floor(px), fy = metal::floor(py);
    int x0 = (int)fx - (T / 2 - 1), y0 = (int)fy - (T / 2 - 1);
    float wx[6], wy[6]; int ix[6], iy[6]; float sx = 0.0f, sy = 0.0f;
    for (int t = 0; t < T; t++) {
        wx[t] = tap_w(MODE, px - (float)(x0 + t)); sx += wx[t];
        wy[t] = tap_w(MODE, py - (float)(y0 + t)); sy += wy[t];
        ix[t] = x0 + t; iy[t] = y0 + t;
    }
    float norm = 1.0f / (sx * sy);
    size_t obase = (((size_t)n * Ho + y) * Wo + x) * C;
    for (int c = 0; c < C; c++) {
        float acc = 0.0f;
        for (int j = 0; j < T; j++) {
            int yy = iy[j];
            if (PAD == 1) yy = metal::clamp(yy, 0, H - 1); else if (yy < 0 || yy >= H) continue;
            for (int i = 0; i < T; i++) {
                int xx = ix[i];
                if (PAD == 1) xx = metal::clamp(xx, 0, W - 1); else if (xx < 0 || xx >= W) continue;
                acc += wx[i] * wy[j] * img[(((size_t)n * H + yy) * W + xx) * C + c];
            }
        }
        out[obase + c] = acc * norm;
    }
"""

_kernel = mx.fast.metal_kernel(name="mlxstab_sample", input_names=["img", "coords"], output_names=["out"],
                               header=_HEADER, source=_SRC)
_MODES = {"bilinear": 0, "bicubic": 1, "lanczos3": 2}
_PADS = {"zeros": 0, "border": 1}


def sample(img, coords, mode="bilinear", padding="zeros"):
    """img [N,H,W,C], coords [N,Ho,Wo,2] (x, y) in pixels -> [N,Ho,Wo,C] float32."""
    img = img.astype(mx.float32); coords = coords.astype(mx.float32)
    N, H, W, C = img.shape; _, Ho, Wo, _ = coords.shape
    if coords.shape[0] != N: raise ValueError("batch mismatch")
    return _kernel(inputs=[img, coords], template=[("MODE", _MODES[mode]), ("PAD", _PADS[padding])],
                   grid=(Wo, Ho, N), threadgroup=(32, 8, 1), output_shapes=[(N, Ho, Wo, C)], output_dtypes=[mx.float32])[0]


def grid_sample(img, grid, padding_mode="zeros", align_corners=False, mode="bilinear"):
    """Like torch grid_sample but NHWC: img [N,H,W,C], grid [N,Ho,Wo,2] in [-1,1] (x, y)."""
    H, W = img.shape[1:3]
    if align_corners: px = (grid[..., 0] + 1) * 0.5 * (W - 1); py = (grid[..., 1] + 1) * 0.5 * (H - 1)
    else: px = ((grid[..., 0] + 1) * W - 1) * 0.5; py = ((grid[..., 1] + 1) * H - 1) * 0.5
    return sample(img, mx.stack([px, py], -1), mode, padding_mode)


def valid_mask(coords, H, W):
    """True where a sample position lies inside the source image (pixel-edge convention), [N,Ho,Wo]."""
    x, y = coords[..., 0], coords[..., 1]
    return (x >= -0.5) & (x <= W - 0.5) & (y >= -0.5) & (y <= H - 0.5)
