//! E20 compatibility spike for RAFT in Rust/MLX.
//!
//! The crate now covers the checkpoint-backed encoders, correlation pyramid,
//! update block, and a bidirectional flow loop. It remains a correctness
//! prototype: correlation lookup runs as a cached MLX Metal kernel on-device,
//! and the update-block core runs through a cached compiled step by default.
//! The API and numerical behavior mirror Python's `RAFT.frame_features` /
//! `RAFT.flow_bidir` so outputs can be compared directly; performance work is
//! still outstanding.

use std::cell::RefCell;
use std::ffi::{CString, c_char};
use std::sync::Once;

use mlx_rs::error::Exception;
use mlx_rs::module::{Module, ModuleParameters};
use mlx_rs::nn::{BatchNorm, Conv2d, relu, sigmoid};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{concatenate, einsum, pad, softmax_axis, stack, tanh};
use mlx_rs::transforms::compile::compile_with_state;
use mlx_rs::{Array, Dtype, Stream};
use mlx_sys as sys;

/// Verbatim port of Python `mlxstab.raft.model._LOOK_SRC`.
const LOOK_SOURCE: &str = r#"
    uint gid = thread_position_in_grid.x;
    const int D = 2 * R + 1, K = D * D;
    int n = gid / K, k = gid % K;
    if (n >= (int)coords_shape[0]) return;
    int a = k / D, b = k % D;                                   // a offsets x, b offsets y (reference quirk, see CorrBlock)
    const float sc = metal::exp2(-(float)LV);
    float px = coords[n * 2] * sc + (float)(a - R), py = coords[n * 2 + 1] * sc + (float)(b - R);
    int H = corr_shape[1], W = corr_shape[2];
    float fx = metal::floor(px), fy = metal::floor(py);
    int x0 = (int)fx, y0 = (int)fy; float tx = px - fx, ty = py - fy;
    size_t base = (size_t)n * H * W;
    float acc = 0.0f;
    for (int j = 0; j < 2; j++) {
        int yy = y0 + j; if (yy < 0 || yy >= H) continue;
        float wy = j ? ty : 1.0f - ty;
        for (int i = 0; i < 2; i++) {
            int xx = x0 + i; if (xx < 0 || xx >= W) continue;
            acc += (i ? tx : 1.0f - tx) * wy * (float)corr[base + (size_t)yy * W + xx];
        }
    }
    out[(size_t)n * K + k] = acc;
"#;

const MLX_SUCCESS: i32 = 0;

fn check_status(status: i32, operation: &str) -> Result<(), String> {
    if status == MLX_SUCCESS {
        Ok(())
    } else {
        Err(format!("{operation} failed with MLX status {status}"))
    }
}

/// Ensure `mlx-rs` has installed its non-exiting process-wide error handler.
///
/// The raw FFI calls below do not go through `mlx-rs`'s guarded operation path.
/// Without this initialization, an MLX kernel error would reach mlx-c's default
/// handler, which terminates the process.
fn ensure_mlx_error_handler() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = mlx_rs::ops::zeros_dtype(&[1], Dtype::Float32);
    });
}

/// A compiled MLX Metal kernel. The handle is immutable after construction and
/// can be reused for every lookup, matching Python's module-level `_look_kernel`.
struct LookKernel {
    raw: sys::mlx_fast_metal_kernel,
}

impl LookKernel {
    fn compile() -> Result<Self, String> {
        let input_names = RawVectorString::new(&["corr", "coords"])?;
        let output_names = RawVectorString::new(&["out"])?;
        let source = CString::new(LOOK_SOURCE).map_err(|e| e.to_string())?;
        let raw = unsafe {
            sys::mlx_fast_metal_kernel_new(
                c"mlxstab_corr_lookup".as_ptr(),
                input_names.0,
                output_names.0,
                source.as_ptr(),
                c"".as_ptr(),
                true,
                false,
            )
        };
        if raw.ctx.is_null() {
            return Err("MLX failed to create the correlation lookup kernel".into());
        }
        Ok(Self { raw })
    }
}

impl Drop for LookKernel {
    fn drop(&mut self) {
        unsafe { sys::mlx_fast_metal_kernel_free(self.raw) };
    }
}

thread_local! {
    // MLX streams are thread-local. Keep the cached kernel on the same thread
    // too, so its opaque C handle never needs an unsafe Send/Sync assertion.
    static LOOK_KERNEL: RefCell<Option<LookKernel>> = const { RefCell::new(None) };
}

fn with_look_kernel<T>(run: impl FnOnce(&LookKernel) -> Result<T, String>) -> Result<T, String> {
    ensure_mlx_error_handler();
    LOOK_KERNEL.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(LookKernel::compile()?);
        }
        run(slot.as_ref().expect("kernel initialized above"))
    })
}

struct RawVectorString(sys::mlx_vector_string);

impl RawVectorString {
    fn new(names: &[&str]) -> Result<Self, String> {
        let names: Vec<CString> = names
            .iter()
            .map(|name| CString::new(*name).map_err(|e| e.to_string()))
            .collect::<Result<_, _>>()?;
        let mut pointers: Vec<*const c_char> = names.iter().map(|name| name.as_ptr()).collect();
        let raw = unsafe { sys::mlx_vector_string_new_data(pointers.as_mut_ptr(), pointers.len()) };
        if raw.ctx.is_null() {
            return Err("MLX failed to allocate a string vector".into());
        }
        Ok(Self(raw))
    }
}

impl Drop for RawVectorString {
    fn drop(&mut self) {
        unsafe { sys::mlx_vector_string_free(self.0) };
    }
}

struct RawVectorArray(sys::mlx_vector_array);

impl RawVectorArray {
    fn from_arrays(arrays: &[&Array]) -> Result<Self, String> {
        let handles: Vec<sys::mlx_array> = arrays.iter().map(|array| array.as_ptr()).collect();
        let raw = unsafe { sys::mlx_vector_array_new_data(handles.as_ptr(), handles.len()) };
        if raw.ctx.is_null() {
            return Err("MLX failed to allocate an array vector".into());
        }
        Ok(Self(raw))
    }

    fn empty() -> Result<Self, String> {
        let raw = unsafe { sys::mlx_vector_array_new() };
        if raw.ctx.is_null() {
            return Err("MLX failed to allocate an output array vector".into());
        }
        Ok(Self(raw))
    }
}

impl Drop for RawVectorArray {
    fn drop(&mut self) {
        unsafe { sys::mlx_vector_array_free(self.0) };
    }
}

struct RawConfig(sys::mlx_fast_metal_kernel_config);

impl RawConfig {
    fn new() -> Result<Self, String> {
        let raw = unsafe { sys::mlx_fast_metal_kernel_config_new() };
        if raw.ctx.is_null() {
            return Err("MLX failed to allocate a Metal kernel config".into());
        }
        Ok(Self(raw))
    }
}

impl Drop for RawConfig {
    fn drop(&mut self) {
        unsafe { sys::mlx_fast_metal_kernel_config_free(self.0) };
    }
}

/// Bilinearly sample one RAFT correlation-pyramid level.
///
/// `corr` is NHWC `[N,H,W,1]` in float16 or float32, and `coords` is `[N,2]` in
/// level-zero `(x,y)` coordinates. Neighborhood index `a` offsets x and `b`
/// offsets y, matching the reference implementation's meshgrid ordering.
/// Out-of-bounds taps are zero, and output has shape `[N,(2*radius+1)^2]` and
/// dtype float32.
///
/// The lookup runs in a cached MLX Metal kernel, so `corr` and `coords` are
/// consumed on-device instead of being materialized into host slices.
pub fn lookup_level(
    corr: &Array,
    coords: &Array,
    level: usize,
    radius: usize,
) -> Result<Array, String> {
    let cs = corr.shape();
    let qs = coords.shape();
    if cs.len() != 4 || cs[3] != 1 {
        return Err(format!("corr must have shape [N,H,W,1], got {cs:?}"));
    }
    if qs.len() != 2 || qs[1] != 2 || qs[0] != cs[0] {
        return Err(format!(
            "coords must have shape [N,2] matching corr, got {qs:?}"
        ));
    }
    if cs.iter().any(|&d| d <= 0) {
        return Err(format!("corr dimensions must be positive, got {cs:?}"));
    }
    if !matches!(corr.dtype(), Dtype::Float16 | Dtype::Float32) {
        return Err(format!(
            "corr dtype must be float16 or float32, got {:?}",
            corr.dtype()
        ));
    }
    if !matches!(coords.dtype(), Dtype::Float16 | Dtype::Float32) {
        return Err(format!(
            "coords dtype must be float16 or float32, got {:?}",
            coords.dtype()
        ));
    }
    if level > i32::MAX as usize {
        return Err("pyramid level exceeds supported range".into());
    }
    if radius > i32::MAX as usize {
        return Err("radius exceeds supported range".into());
    }
    let n = cs[0] as usize;
    let diameter = radius
        .checked_mul(2)
        .and_then(|x| x.checked_add(1))
        .ok_or_else(|| "radius is too large".to_string())?;
    let k = diameter
        .checked_mul(diameter)
        .ok_or_else(|| "radius is too large".to_string())?;
    let out_len = n
        .checked_mul(k)
        .ok_or_else(|| "output is too large".to_string())?;
    if out_len > i32::MAX as usize {
        return Err("output dimensions exceed MLX limits".into());
    }

    with_look_kernel(|kernel| {
        let stream = Stream::thread_local_or_default();
        let inputs = RawVectorArray::from_arrays(&[corr, coords])?;
        let mut outputs = RawVectorArray::empty()?;
        let config = RawConfig::new()?;
        let output_shape = [n as i32, k as i32];

        check_status(
            unsafe {
                sys::mlx_fast_metal_kernel_config_add_output_arg(
                    config.0,
                    output_shape.as_ptr(),
                    output_shape.len(),
                    sys::mlx_dtype__MLX_FLOAT32,
                )
            },
            "configuring the correlation lookup output",
        )?;
        check_status(
            unsafe { sys::mlx_fast_metal_kernel_config_set_grid(config.0, out_len as i32, 1, 1) },
            "configuring the correlation lookup grid",
        )?;
        check_status(
            unsafe { sys::mlx_fast_metal_kernel_config_set_thread_group(config.0, 256, 1, 1) },
            "configuring the correlation lookup thread group",
        )?;
        check_status(
            unsafe {
                sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                    config.0,
                    c"R".as_ptr(),
                    radius as i32,
                )
            },
            "configuring the correlation lookup radius",
        )?;
        check_status(
            unsafe {
                sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                    config.0,
                    c"LV".as_ptr(),
                    level as i32,
                )
            },
            "configuring the correlation lookup level",
        )?;

        check_status(
            unsafe {
                sys::mlx_fast_metal_kernel_apply(
                    &mut outputs.0,
                    kernel.raw,
                    inputs.0,
                    config.0,
                    stream.as_ptr(),
                )
            },
            "running the correlation lookup kernel",
        )?;

        let mut output = unsafe { sys::mlx_array_new() };
        let status = unsafe { sys::mlx_vector_array_get(&mut output, outputs.0, 0) };
        if status != MLX_SUCCESS {
            unsafe { sys::mlx_array_free(output) };
            return Err(format!(
                "reading the correlation lookup output failed with MLX status {status}"
            ));
        }
        // SAFETY: `mlx_vector_array_get` copies a retained handle into `output`;
        // ownership transfers to `Array`, which frees it on drop. The owning vector
        // (with its own reference) is released when `outputs` drops.
        Ok(unsafe { Array::from_ptr(output) })
    })
}

/// Floor-sized 2x2 average pooling on NHWC `[N,H,W,1]`.
///
/// Odd trailing rows and columns are dropped before pooling, matching Python's
/// `_pool2` and PyTorch's `avg_pool2d(2)` rather than padding them. The pooled
/// result has shape `[N,H/2,W/2,1]` and keeps the input dtype.
fn pool2(corr: &Array) -> Result<Array, String> {
    let shape = corr.shape();
    if shape.len() != 4 || shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 || shape[3] != 1 {
        return Err(format!(
            "pool2 expects non-empty NHWC [N,H,W,1], got {shape:?}"
        ));
    }
    let (batch, height, width) = (shape[0], shape[1], shape[2]);
    let pooled_height = height / 2;
    let pooled_width = width / 2;
    if pooled_height == 0 || pooled_width == 0 {
        return Err(format!(
            "pool2 would drop the whole spatial extent of {height}x{width}"
        ));
    }
    // Slice off the trailing odd row/column before reshaping into 2x2 cells.
    let cropped = corr.index((.., 0..pooled_height * 2, 0..pooled_width * 2, ..));
    let cells = cropped
        .reshape(&[batch, pooled_height, 2, pooled_width, 2, 1])
        .map_err(|e| e.to_string())?;
    cells.mean_axes(&[2, 4], false).map_err(|e| e.to_string())
}

/// RAFT all-pairs correlation pyramid over two NHWC feature maps.
///
/// `f1` and `f2` each have shape `[1,H,W,C]`. Level zero stores
/// `[H*W,H,W,1]` values from `(f1 @ f2^T) / sqrt(C)`; each subsequent level is
/// the floor-sized 2x2 average of the previous one. Lookups use level-zero
/// `(x,y)` pixel coordinates and the same XY-order, bilinear, zero-padding
/// contract as [`lookup_level`].
#[derive(Debug)]
pub struct CorrBlock {
    pub levels: usize,
    pub radius: usize,
    pub pyr: Vec<Array>,
}

impl CorrBlock {
    /// Build the correlation pyramid from paired NHWC feature maps.
    ///
    /// `dtype` controls the stored correlation values and must be float16 or
    /// float32, matching Python's `CorrBlock(..., dtype=...)`. The result is
    /// lazy, like the Python graph. Every requested pyramid level must retain a
    /// non-empty spatial extent.
    pub fn new(
        f1: &Array,
        f2: &Array,
        levels: usize,
        radius: usize,
        dtype: Dtype,
    ) -> Result<Self, String> {
        if levels == 0 {
            return Err("levels must be at least 1".into());
        }
        if !matches!(dtype, Dtype::Float16 | Dtype::Float32) {
            return Err(format!(
                "correlation dtype must be float16 or float32, got {dtype:?}"
            ));
        }
        let shape = f1.shape();
        if shape.len() != 4 || shape[0] != 1 || shape[1] <= 0 || shape[2] <= 0 || shape[3] <= 0 {
            return Err(format!(
                "f1 must be non-empty NHWC [1,H,W,C], got {shape:?}"
            ));
        }
        if f2.shape() != shape {
            return Err(format!(
                "f1 and f2 must have the same shape, got {:?} and {:?}",
                shape,
                f2.shape()
            ));
        }
        let (height, width, channels) = (shape[1], shape[2], shape[3]);
        let pair = height * width;

        let a = f1
            .reshape(&[pair, channels])
            .and_then(|v| v.as_dtype(dtype))
            .map_err(|e| e.to_string())?;
        let b_t = f2
            .reshape(&[pair, channels])
            .and_then(|v| v.as_dtype(dtype))
            .and_then(|v| v.transpose_axes(&[1, 0]))
            .map_err(|e| e.to_string())?;
        // Python computes the divisor with math.sqrt (f64), then MLX converts the
        // scalar to the stored dtype. Keep the divisor in that dtype: dividing by
        // a float32 array would promote a float16 correlation pyramid to float32.
        let divisor = Array::from_f32((channels as f64).sqrt() as f32)
            .as_dtype(dtype)
            .map_err(|e| e.to_string())?;
        let corr = a
            .matmul(&b_t)
            .and_then(|v| v.divide(&divisor))
            .and_then(|v| v.reshape(&[pair, height, width, 1]))
            .map_err(|e| e.to_string())?;

        let mut pyr = Vec::with_capacity(levels);
        pyr.push(corr);
        for level in 1..levels {
            let previous = pyr.last().unwrap();
            let (h, w) = (previous.shape()[1], previous.shape()[2]);
            if h < 2 || w < 2 {
                return Err(format!(
                    "correlation level {level} would be empty from level {} {h}x{w}",
                    level - 1
                ));
            }
            pyr.push(pool2(previous)?);
        }
        Ok(Self {
            levels,
            radius,
            pyr,
        })
    }

    /// Look up all pyramid levels at NHWC `[1,h,w,2]` level-zero `(x,y)` coordinates.
    ///
    /// Returns `[1,h,w,levels*(2*radius+1)^2]` float32 values, matching Python's
    /// `CorrBlock.__call__`.
    pub fn lookup(&self, coords: &Array) -> Result<Array, String> {
        let shape = coords.shape();
        if shape.len() != 4 || shape[0] != 1 || shape[1] <= 0 || shape[2] <= 0 || shape[3] != 2 {
            return Err(format!(
                "coords must be non-empty NHWC [1,H,W,2], got {shape:?}"
            ));
        }
        let (height, width) = (shape[1], shape[2]);
        let pair = self.pyr[0].shape()[0];
        if height * width != pair {
            return Err(format!(
                "coords cover {} positions but the correlation pyramid stores {pair} pairs",
                height * width
            ));
        }
        let flat = coords
            .reshape(&[height * width, 2])
            .map_err(|e| e.to_string())?;
        let mut levels = Vec::with_capacity(self.levels);
        for (level, corr) in self.pyr.iter().enumerate() {
            levels.push(lookup_level(corr, &flat, level, self.radius)?);
        }
        let combined = mlx_rs::ops::concatenate(&levels, 1).map_err(|e| e.to_string())?;
        let lookup_width = combined.shape()[1];
        combined
            .reshape(&[1, height, width, lookup_width])
            .map_err(|e| e.to_string())
    }
}

/// Read a safetensors file through MLX and validate a caller-provided exact
/// parameter schema before a Rust model accepts it.
pub fn load_safetensors_checked(
    path: impl AsRef<std::path::Path>,
    expected: &[(&str, &[i32])],
) -> Result<std::collections::HashMap<String, Array>, String> {
    let loaded = Array::load_safetensors(path).map_err(|e| e.to_string())?;
    let expected_keys: std::collections::HashSet<&str> = expected.iter().map(|(k, _)| *k).collect();
    let loaded_keys: std::collections::HashSet<&str> = loaded.keys().map(String::as_str).collect();
    let missing: Vec<_> = expected_keys.difference(&loaded_keys).copied().collect();
    let extra: Vec<_> = loaded_keys.difference(&expected_keys).copied().collect();
    if !missing.is_empty() || !extra.is_empty() {
        return Err(format!(
            "safetensors key mismatch; missing={missing:?}, extra={extra:?}"
        ));
    }
    for (key, shape) in expected {
        let actual = loaded[*key].shape();
        if actual != *shape {
            return Err(format!(
                "safetensors shape mismatch for {key}: expected {shape:?}, got {actual:?}"
            ));
        }
    }
    Ok(loaded)
}

/// Run the checkpoint-backed RAFT context encoder's first 7x7 convolution.
///
/// This is a real MLX convolution with the converted RAFT weight layout
/// `[out,kh,kw,in]`; the remaining normalization/residual encoder stages are
/// not ported yet.
pub fn cnet_conv1(
    input: &Array,
    params: &std::collections::HashMap<String, Array>,
) -> Result<Array, String> {
    if input.shape().len() != 4 || input.shape()[0] != 1 || input.shape()[3] != 3 {
        return Err(format!(
            "input must be NHWC [1,H,W,3], got {:?}",
            input.shape()
        ));
    }
    let weight = params
        .get("cnet.conv1.weight")
        .ok_or_else(|| "missing cnet.conv1.weight".to_string())?;
    let bias = params
        .get("cnet.conv1.bias")
        .ok_or_else(|| "missing cnet.conv1.bias".to_string())?;
    if weight.shape() != [64, 7, 7, 3] || bias.shape() != [64] {
        return Err(format!(
            "unexpected cnet.conv1 parameter shapes: weight={:?}, bias={:?}",
            weight.shape(),
            bias.shape()
        ));
    }
    let mut conv = Conv2d::new(3, 64, (7, 7)).map_err(|e| e.to_string())?;
    conv.stride = (2, 2);
    conv.padding = (3, 3);
    conv.weight.value = weight.clone();
    conv.bias.value = Some(bias.clone());
    conv.forward(input).map_err(|e| e.to_string())
}

// --- Full RAFT BasicEncoder (encoders only; the update block is not ported yet) -----------------------------

use std::collections::HashSet;

/// Normalization used by a RAFT `BasicEncoder`: instance norm for `fnet`, batch norm for `cnet`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderNorm {
    Instance,
    Batch,
}

/// The normalization layer paired with an encoder convolution.
#[derive(Debug)]
enum NormLayer {
    /// Parameter-free instance norm with statistics accumulated in f32, matching the Python reference.
    Instance { eps: f32 },
    /// Evaluation-mode batch norm using the checkpoint's running statistics.
    Batch(BatchNorm),
}

impl NormLayer {
    fn instance() -> Self {
        NormLayer::Instance { eps: 1e-5 }
    }

    fn batch(reader: &mut ParamReader<'_>, prefix: &str, channels: i32) -> Result<Self, String> {
        let width = [channels];
        let weight = reader.get(&format!("{prefix}.weight"), &width)?;
        let bias = reader.get(&format!("{prefix}.bias"), &width)?;
        let running_mean = reader.get(&format!("{prefix}.running_mean"), &width)?;
        let running_var = reader.get(&format!("{prefix}.running_var"), &width)?;
        let mut norm = BatchNorm::new(channels).map_err(|e| e.to_string())?;
        norm.training = false;
        norm.weight.value = Some(weight);
        norm.bias.value = Some(bias);
        norm.running_mean.value = Some(running_mean);
        norm.running_var.value = Some(running_var);
        Ok(NormLayer::Batch(norm))
    }

    fn forward(&mut self, x: &Array) -> Result<Array, String> {
        match self {
            NormLayer::Instance { eps } => {
                // Match Python's InstanceNorm: force f32 statistics even if the caller uses fp16.
                let dtype = x.dtype();
                let xf = x.as_dtype(Dtype::Float32).map_err(|e| e.to_string())?;
                let mean = xf.mean_axes(&[1, 2], true).map_err(|e| e.to_string())?;
                let variance = xf
                    .var_axes(&[1, 2], true, None)
                    .map_err(|e| e.to_string())?;
                let denom = variance
                    .add(Array::from_f32(*eps))
                    .and_then(|v| mlx_rs::ops::rsqrt(&v))
                    .map_err(|e| e.to_string())?;
                xf.subtract(&mean)
                    .and_then(|v| v.multiply(&denom))
                    .and_then(|v| v.as_dtype(dtype))
                    .map_err(|e| e.to_string())
            }
            NormLayer::Batch(norm) => norm.forward(x).map_err(|e| e.to_string()),
        }
    }
}

/// Loads and verifies parameters by their converted safetensors names.
struct ParamReader<'a> {
    params: &'a std::collections::HashMap<String, Array>,
    used: HashSet<String>,
}

impl<'a> ParamReader<'a> {
    fn new(params: &'a std::collections::HashMap<String, Array>) -> Self {
        Self {
            params,
            used: HashSet::new(),
        }
    }

    fn get(&mut self, name: &str, shape: &[i32]) -> Result<Array, String> {
        if !self.used.insert(name.to_owned()) {
            return Err(format!("parameter {name} requested more than once"));
        }
        let value = self
            .params
            .get(name)
            .ok_or_else(|| format!("missing parameter {name}"))?;
        if value.shape() != shape {
            return Err(format!(
                "parameter shape mismatch for {name}: expected {shape:?}, got {:?}",
                value.shape()
            ));
        }
        Ok(value.clone())
    }

    fn conv(
        &mut self,
        prefix: &str,
        input_channels: i32,
        output_channels: i32,
        kernel: i32,
        stride: i32,
        padding: i32,
    ) -> Result<Conv2d, String> {
        self.conv_rect(
            prefix,
            input_channels,
            output_channels,
            (kernel, kernel),
            (stride, stride),
            (padding, padding),
        )
    }

    /// Load a convolution whose spatial kernel, stride and padding may differ per axis.
    ///
    /// Weights use the converted NHWC layout `[out, kernel_h, kernel_w, in]`.
    #[allow(clippy::too_many_arguments)]
    fn conv_rect(
        &mut self,
        prefix: &str,
        input_channels: i32,
        output_channels: i32,
        kernel: (i32, i32),
        stride: (i32, i32),
        padding: (i32, i32),
    ) -> Result<Conv2d, String> {
        let weight = self.get(
            &format!("{prefix}.weight"),
            &[output_channels, kernel.0, kernel.1, input_channels],
        )?;
        let bias = self.get(&format!("{prefix}.bias"), &[output_channels])?;
        let mut conv =
            Conv2d::new(input_channels, output_channels, kernel).map_err(|e| e.to_string())?;
        conv.stride = stride;
        conv.padding = padding;
        conv.weight.value = weight;
        conv.bias.value = Some(bias);
        Ok(conv)
    }

    fn finish(&self, prefix: &str) -> Result<(), String> {
        let scope = format!("{prefix}.");
        let mut extra: Vec<_> = self
            .params
            .keys()
            .filter(|key| key.starts_with(&scope) && !self.used.contains(*key))
            .cloned()
            .collect();
        extra.sort();
        if extra.is_empty() {
            Ok(())
        } else {
            Err(format!("unexpected parameters under {prefix}: {extra:?}"))
        }
    }
}

/// One RAFT residual stage with its optional 1x1 downsample projection.
#[derive(Debug)]
pub struct ResidualBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    norm1: NormLayer,
    norm2: NormLayer,
    downsample: Option<(Conv2d, NormLayer)>,
}

impl ResidualBlock {
    fn load(
        reader: &mut ParamReader<'_>,
        prefix: &str,
        input_channels: i32,
        channels: i32,
        norm: EncoderNorm,
        stride: i32,
    ) -> Result<Self, String> {
        let conv1 = reader.conv(
            &format!("{prefix}.conv1"),
            input_channels,
            channels,
            3,
            stride,
            1,
        )?;
        let conv2 = reader.conv(&format!("{prefix}.conv2"), channels, channels, 3, 1, 1)?;
        let norm1 = match norm {
            EncoderNorm::Instance => NormLayer::instance(),
            EncoderNorm::Batch => NormLayer::batch(reader, &format!("{prefix}.norm1"), channels)?,
        };
        let norm2 = match norm {
            EncoderNorm::Instance => NormLayer::instance(),
            EncoderNorm::Batch => NormLayer::batch(reader, &format!("{prefix}.norm2"), channels)?,
        };
        let downsample = if stride == 1 {
            None
        } else {
            let conv = reader.conv(
                &format!("{prefix}.downsample.0"),
                input_channels,
                channels,
                1,
                stride,
                0,
            )?;
            let norm = match norm {
                EncoderNorm::Instance => NormLayer::instance(),
                EncoderNorm::Batch => {
                    NormLayer::batch(reader, &format!("{prefix}.downsample.1"), channels)?
                }
            };
            Some((conv, norm))
        };
        Ok(Self {
            conv1,
            conv2,
            norm1,
            norm2,
            downsample,
        })
    }

    /// Run the residual block on an NHWC f32 tensor.
    pub fn forward(&mut self, x: &Array) -> Result<Array, String> {
        let mut y = self.conv1.forward(x).map_err(|e| e.to_string())?;
        y = self.norm1.forward(&y)?;
        y = relu(&y).map_err(|e| e.to_string())?;
        y = self.conv2.forward(&y).map_err(|e| e.to_string())?;
        y = self.norm2.forward(&y)?;
        y = relu(&y).map_err(|e| e.to_string())?;

        let shortcut = match &mut self.downsample {
            Some((conv, norm)) => {
                let projected = conv.forward(x).map_err(|e| e.to_string())?;
                norm.forward(&projected)?
            }
            None => x.clone(),
        };
        let sum = shortcut.add(&y).map_err(|e| e.to_string())?;
        relu(&sum).map_err(|e| e.to_string())
    }
}

/// RAFT's BasicEncoder. `fnet` uses instance norm and `cnet` uses batch norm in the reference model.
#[derive(Debug)]
pub struct BasicEncoder {
    pub norm: EncoderNorm,
    pub out_dim: i32,
    conv1: Conv2d,
    norm1: NormLayer,
    layer1: Vec<ResidualBlock>,
    layer2: Vec<ResidualBlock>,
    layer3: Vec<ResidualBlock>,
    conv2: Conv2d,
}

impl BasicEncoder {
    /// Load one encoder from converted RAFT parameters, validating every expected name and shape.
    ///
    /// `params` is the converted safetensors map; `prefix` is `"fnet"` or `"cnet"`. Any unconsumed
    /// parameter under that prefix is rejected so an architecture change cannot silently ignore a
    /// checkpoint tensor.
    pub fn load(
        params: &std::collections::HashMap<String, Array>,
        prefix: &str,
        norm: EncoderNorm,
    ) -> Result<Self, String> {
        let mut reader = ParamReader::new(params);
        let encoder = Self::load_with(&mut reader, prefix, norm)?;
        reader.finish(prefix)?;
        Ok(encoder)
    }

    fn load_with(
        reader: &mut ParamReader<'_>,
        prefix: &str,
        norm: EncoderNorm,
    ) -> Result<Self, String> {
        let out_dim = 256;
        let conv1 = reader.conv(&format!("{prefix}.conv1"), 3, 64, 7, 2, 3)?;
        let norm1 = match norm {
            EncoderNorm::Instance => NormLayer::instance(),
            EncoderNorm::Batch => NormLayer::batch(reader, &format!("{prefix}.norm1"), 64)?,
        };
        let layer1 = vec![
            ResidualBlock::load(reader, &format!("{prefix}.layer1.0"), 64, 64, norm, 1)?,
            ResidualBlock::load(reader, &format!("{prefix}.layer1.1"), 64, 64, norm, 1)?,
        ];
        let layer2 = vec![
            ResidualBlock::load(reader, &format!("{prefix}.layer2.0"), 64, 96, norm, 2)?,
            ResidualBlock::load(reader, &format!("{prefix}.layer2.1"), 96, 96, norm, 1)?,
        ];
        let layer3 = vec![
            ResidualBlock::load(reader, &format!("{prefix}.layer3.0"), 96, 128, norm, 2)?,
            ResidualBlock::load(reader, &format!("{prefix}.layer3.1"), 128, 128, norm, 1)?,
        ];
        let conv2 = reader.conv(&format!("{prefix}.conv2"), 128, out_dim, 1, 1, 0)?;
        Ok(Self {
            norm,
            out_dim,
            conv1,
            norm1,
            layer1,
            layer2,
            layer3,
            conv2,
        })
    }

    /// Run the encoder on an already normalized NHWC f32 input `[N,H,W,3]`.
    ///
    /// This matches the tensor the Python `RAFT.encode` path passes to `fnet`/`cnet`; image scaling
    /// from 0..255 to -1..1 is the caller's responsibility. H and W must be multiples of 8 (the
    /// reference `pad8` step pads them before calling the encoder).
    pub fn forward(&mut self, x: &Array) -> Result<Array, String> {
        let shape = x.shape();
        if shape.len() != 4 || shape[3] != 3 {
            return Err(format!("input must be NHWC [N,H,W,3], got {shape:?}"));
        }
        if shape[1] % 8 != 0 || shape[2] % 8 != 0 {
            return Err(format!(
                "input height and width must be multiples of 8, got {shape:?}"
            ));
        }
        let mut y = self.conv1.forward(x).map_err(|e| e.to_string())?;
        y = self.norm1.forward(&y)?;
        y = relu(&y).map_err(|e| e.to_string())?;
        for block in self
            .layer1
            .iter_mut()
            .chain(self.layer2.iter_mut())
            .chain(self.layer3.iter_mut())
        {
            y = block.forward(&y)?;
        }
        self.conv2.forward(&y).map_err(|e| e.to_string())
    }
}

/// The two RAFT feature/context encoders, loaded from the same converted checkpoint.
#[derive(Debug)]
pub struct RaftEncoders {
    pub fnet: BasicEncoder,
    pub cnet: BasicEncoder,
}

impl RaftEncoders {
    /// Load `fnet` (instance norm) and `cnet` (batch norm) from a converted RAFT safetensors file.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let params = Array::load_safetensors(path).map_err(|e| e.to_string())?;
        Self::from_params(&params)
    }

    /// Build both encoders from an already-loaded parameter map.
    pub fn from_params(params: &std::collections::HashMap<String, Array>) -> Result<Self, String> {
        Ok(Self {
            fnet: BasicEncoder::load(params, "fnet", EncoderNorm::Instance)?,
            cnet: BasicEncoder::load(params, "cnet", EncoderNorm::Batch)?,
        })
    }
}

// --- Bidirectional flow API ---------------------------------------------------------------------------------

/// Per-frame RAFT features, reusable across the forward and backward pairs it joins.
#[derive(Debug)]
pub struct FrameFeat {
    pub fmap: Array,
    pub net: Array,
    pub inp: Array,
}

/// Forward/backward flow in low-resolution feature-grid pixels and full-resolution image pixels.
///
/// Field names mirror Python's `RAFT.flow_bidir` return order: `(low_f, up_f, low_b, up_b)`.
#[derive(Debug)]
pub struct BidirectionalFlow {
    pub low_forward: Array,
    pub up_forward: Array,
    pub low_backward: Array,
    pub up_backward: Array,
}

fn split_frame_features(
    fmap: &Array,
    context: &Array,
    batch_index: i32,
) -> Result<FrameFeat, String> {
    let batch = batch_index..batch_index + 1;
    let fmap = fmap.index((batch.clone(), .., .., ..));
    let net = tanh(context.index((batch.clone(), .., .., 0..RAFT_HIDDEN_CHANNELS)))
        .map_err(|e| e.to_string())?;
    let inp = relu(context.index((
        batch,
        ..,
        ..,
        RAFT_HIDDEN_CHANNELS..2 * RAFT_HIDDEN_CHANNELS,
    )))
    .map_err(|e| e.to_string())?;
    Ok(FrameFeat { fmap, net, inp })
}

/// Checkpoint-backed RAFT encoders plus update block for fp32 inference and Python parity checks.
///
/// The bidirectional loop mirrors `mlxstab.raft.model.RAFT`: features are encoded once per frame, the
/// forward and backward correlation pyramids are kept separate, and both directions share one
/// batch-2 update-block call per iteration. Coordinate and flow updates stay float32. By default the
/// update-block core runs through one cached `compile_with_state` step whose state is the mutable
/// update block, so no model `Array` is captured by the compiled closure. `set_compiled_update(false)`
/// selects the direct `BasicUpdateBlock::core` reference path instead.
#[derive(Debug)]
pub struct Raft {
    pub encoders: RaftEncoders,
    pub update_block: BasicUpdateBlock,
    compiled_update: bool,
    compiled_core: Option<CompiledCore>,
    batched_encoding: bool,
}

impl Raft {
    /// Load all RAFT parameters from a converted safetensors checkpoint.
    ///
    /// The update block defaults to float32.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let params = Array::load_safetensors(path).map_err(|e| e.to_string())?;
        Self::from_params(&params)
    }

    /// Load a converted safetensors checkpoint and select the update-block parameter dtype.
    pub fn load_with_update_dtype(
        path: impl AsRef<std::path::Path>,
        dtype: Dtype,
    ) -> Result<Self, String> {
        let params = Array::load_safetensors(path).map_err(|e| e.to_string())?;
        Self::from_params_with_update_dtype(&params, dtype)
    }

    /// Build the complete model from an already-loaded converted parameter map.
    ///
    /// The update block defaults to float32.
    pub fn from_params(params: &std::collections::HashMap<String, Array>) -> Result<Self, String> {
        Self::from_params_with_update_dtype(params, Dtype::Float32)
    }

    /// Build the complete model and select the update-block parameter dtype.
    pub fn from_params_with_update_dtype(
        params: &std::collections::HashMap<String, Array>,
        dtype: Dtype,
    ) -> Result<Self, String> {
        let mut update_block = BasicUpdateBlock::load(params)?;
        update_block.set_dtype(dtype)?;
        Ok(Self {
            encoders: RaftEncoders::from_params(params)?,
            update_block,
            compiled_update: true,
            compiled_core: None,
            batched_encoding: false,
        })
    }

    /// Select float32 (default) or float16 update-block convolution parameters.
    ///
    /// Only the update block's Conv2d weights/biases are cast. Encoders, feature maps,
    /// correlation pyramids, coordinates, the flow accumulator, and returned flows stay float32.
    /// Changing this invalidates the cached compiled update core so it is rebuilt against the new
    /// parameter dtype on the next call.
    pub fn set_update_dtype(&mut self, dtype: Dtype) -> Result<(), String> {
        self.update_block.set_dtype(dtype)?;
        self.compiled_core = None;
        Ok(())
    }

    /// Return the current update-block parameter dtype.
    pub fn update_dtype(&self) -> Dtype {
        self.update_block.dtype()
    }

    /// Select the update-block core used by `flow_bidir` and `flow_bidir_features`.
    ///
    /// `true` (the default) runs the cached compiled core, matching Python's `mx.compile` path.
    /// `false` calls `BasicUpdateBlock::core` directly as the uncompiled reference.
    pub fn set_compiled_update(&mut self, compiled: bool) {
        self.compiled_update = compiled;
    }

    /// Return the update-block core mode selected by `set_compiled_update`.
    pub fn compiled_update(&self) -> bool {
        self.compiled_update
    }

    /// Select whether `flow_bidir` encodes both input frames as one batch of two.
    ///
    /// `false` (the default) encodes the frames separately as the reference. `true` runs each
    /// encoder once on the concatenated pair, then splits the per-frame features.
    pub fn set_batched_encoding(&mut self, batched: bool) {
        self.batched_encoding = batched;
    }

    /// Return the frame-encoder batching mode selected by `set_batched_encoding`.
    pub fn batched_encoding(&self) -> bool {
        self.batched_encoding
    }

    /// Encode one `[1,H,W,3]` 0..255 image into fnet/cnet features.
    ///
    /// This matches Python's `RAFT.frame_features`: the image is normalized to -1..1, `fnet` (instance
    /// norm) produces the correlation features, and `cnet` (batch norm) produces the hidden state and
    /// context split by `tanh` and `relu`. Height and width must be positive multiples of 8.
    pub fn frame_features(&mut self, im: &Array) -> Result<FrameFeat, String> {
        let x = normalize_image(im)?;
        let fmap = self.encoders.fnet.forward(&x)?;
        let c = self.encoders.cnet.forward(&x)?;
        split_frame_features(&fmap, &c, 0)
    }

    /// Encode two `[1,H,W,3]` images together, then return one `FrameFeat` per image.
    ///
    /// Instance normalization is per sample and cnet batch normalization uses the checkpoint's
    /// evaluation statistics, so this preserves the separate `frame_features` semantics.
    pub fn frame_features_pair(
        &mut self,
        im1: &Array,
        im2: &Array,
    ) -> Result<(FrameFeat, FrameFeat), String> {
        let first = normalize_image(im1)?;
        let second = normalize_image(im2)?;
        if first.shape() != second.shape() {
            return Err(format!(
                "paired images must have matching shapes, got {:?} and {:?}",
                first.shape(),
                second.shape()
            ));
        }
        let pair = concatenate(&[&first, &second], 0).map_err(|e| e.to_string())?;
        let fmap = self.encoders.fnet.forward(&pair)?;
        let context = self.encoders.cnet.forward(&pair)?;
        Ok((
            split_frame_features(&fmap, &context, 0)?,
            split_frame_features(&fmap, &context, 1)?,
        ))
    }

    /// Bidirectional flow between two `[1,H,W,3]` 0..255 images.
    ///
    /// `iters` is the shared GRU refinement count for both directions; zero returns the initialized
    /// (zero) flow with the encoder mask head. Outputs mirror Python's `(low_f, up_f, low_b, up_b)`.
    pub fn flow_bidir(
        &mut self,
        im1: &Array,
        im2: &Array,
        iters: usize,
    ) -> Result<BidirectionalFlow, String> {
        let (a, b) = if self.batched_encoding {
            self.frame_features_pair(im1, im2)?
        } else {
            (self.frame_features(im1)?, self.frame_features(im2)?)
        };
        self.flow_bidir_features(&a, &b, iters)
    }

    /// Bidirectional flow from precomputed frame features, matching Python's `flow_bidir`.
    ///
    /// Both features must come from `frame_features` on the same feature grid. The forward pyramid is
    /// `a @ b^T`, the backward pyramid is `b @ a^T`, and each iteration looks up both directions before
    /// running one batch-2 update-block core step in the mode selected by `set_compiled_update`.
    pub fn flow_bidir_features(
        &mut self,
        a: &FrameFeat,
        b: &FrameFeat,
        iters: usize,
    ) -> Result<BidirectionalFlow, String> {
        let shape = a.fmap.shape();
        if shape.len() != 4 || shape[0] != 1 || shape[1] <= 0 || shape[2] <= 0 || shape[3] <= 0 {
            return Err(format!(
                "fmap must be non-empty NHWC [1,H,W,C], got {shape:?}"
            ));
        }
        if b.fmap.shape() != shape {
            return Err(format!(
                "forward and backward feature grids must match, got {:?} and {:?}",
                shape,
                b.fmap.shape()
            ));
        }
        let feature_shape = [1, shape[1], shape[2], RAFT_HIDDEN_CHANNELS];
        for (name, array) in [("forward net", &a.net), ("forward inp", &a.inp)] {
            if array.shape() != feature_shape.as_slice() {
                return Err(format!(
                    "{name} must have shape {feature_shape:?}, got {:?}",
                    array.shape()
                ));
            }
        }
        for (name, array) in [("backward net", &b.net), ("backward inp", &b.inp)] {
            if array.shape() != feature_shape.as_slice() {
                return Err(format!(
                    "{name} must have shape {feature_shape:?}, got {:?}",
                    array.shape()
                ));
            }
        }

        let (height, width) = (shape[1], shape[2]);
        let corr_forward = CorrBlock::new(
            &a.fmap,
            &b.fmap,
            RAFT_CORR_LEVELS,
            RAFT_CORR_RADIUS,
            Dtype::Float32,
        )?;
        let corr_backward = CorrBlock::new(
            &b.fmap,
            &a.fmap,
            RAFT_CORR_LEVELS,
            RAFT_CORR_RADIUS,
            Dtype::Float32,
        )?;

        let c0 = coords_grid(2, height, width);
        let mut c1 = c0.clone();
        let update_dtype = self.update_dtype();
        let cast = |array: &Array| -> Result<Array, String> {
            if array.dtype() == update_dtype {
                Ok(array.clone())
            } else {
                array.as_dtype(update_dtype).map_err(|e| e.to_string())
            }
        };
        let mut net = concatenate(&[&a.net, &b.net], 0).map_err(|e| e.to_string())?;
        net = cast(&net)?;
        let inp = cast(&concatenate(&[&a.inp, &b.inp], 0).map_err(|e| e.to_string())?)?;

        for _ in 0..iters {
            let forward_coords = c1.index((0..1, .., .., ..));
            let backward_coords = c1.index((1..2, .., .., ..));
            let forward_corr = corr_forward.lookup(&forward_coords)?;
            let backward_corr = corr_backward.lookup(&backward_coords)?;
            let corr =
                concatenate(&[&forward_corr, &backward_corr], 0).map_err(|e| e.to_string())?;
            let corr = cast(&corr)?;
            let flow = c1.subtract(&c0).map_err(|e| e.to_string())?;
            let flow = cast(&flow)?;
            let core = if self.compiled_update {
                let compiled_core = self.compiled_core.get_or_insert_with(CompiledCore::new);
                compiled_core.core(&mut self.update_block, &net, &inp, &corr, &flow)?
            } else {
                self.update_block.core(&net, &inp, &corr, &flow)?
            };
            net = core.net;
            let delta = core
                .delta
                .as_dtype(Dtype::Float32)
                .map_err(|e| e.to_string())?;
            c1 = c1.add(&delta).map_err(|e| e.to_string())?;
        }

        let mask = self.update_block.mask_head(&net)?;
        let mask = mask.as_dtype(Dtype::Float32).map_err(|e| e.to_string())?;
        let low = c1.subtract(&c0).map_err(|e| e.to_string())?;
        let up = upsample_flow_batch(&low, &mask)?;
        Ok(BidirectionalFlow {
            low_forward: low.index((0..1, .., .., ..)),
            up_forward: up.index((0..1, .., .., ..)),
            low_backward: low.index((1..2, .., .., ..)),
            up_backward: up.index((1..2, .., .., ..)),
        })
    }
}

/// Normalize a `[1,H,W,3]` 0..255 image exactly like Python's `RAFT.frame_features`.
fn normalize_image(im: &Array) -> Result<Array, String> {
    let shape = im.shape();
    if shape.len() != 4 || shape[0] != 1 || shape[1] <= 0 || shape[2] <= 0 || shape[3] != 3 {
        return Err(format!(
            "image must have shape [1,H,W,3] with positive spatial dimensions, got {shape:?}"
        ));
    }
    if shape[1] % 8 != 0 || shape[2] % 8 != 0 {
        return Err(format!(
            "image height and width must be multiples of 8 (pad first), got {shape:?}"
        ));
    }
    let x = im.as_dtype(Dtype::Float32).map_err(|e| e.to_string())?;
    x.divide(Array::from_f32(255.0))
        .and_then(|v| v.multiply(Array::from_f32(2.0)))
        .and_then(|v| v.subtract(Array::from_f32(1.0)))
        .map_err(|e| e.to_string())
}

/// XY coordinate grid `[batch,H,W,2]`, matching Python's `coords_grid`.
fn coords_grid(batch: i32, height: i32, width: i32) -> Array {
    let mut values = Vec::with_capacity((batch * height * width * 2) as usize);
    for _ in 0..batch {
        for y in 0..height {
            for x in 0..width {
                values.push(x as f32);
                values.push(y as f32);
            }
        }
    }
    Array::from_slice(&values, &[batch, height, width, 2])
}

/// Convex 8x upsampling for one direction: `flow [1,h,w,2]`, `mask [1,h,w,576]` -> `[1,8h,8w,2]`.
fn upsample_flow(flow: &Array, mask: &Array) -> Result<Array, String> {
    let shape = flow.shape();
    if shape.len() != 4 || shape[0] != 1 || shape[1] <= 0 || shape[2] <= 0 || shape[3] != 2 {
        return Err(format!(
            "flow must have shape [1,h,w,2] with positive spatial dimensions, got {shape:?}"
        ));
    }
    let (height, width) = (shape[1], shape[2]);
    let expected_mask = [1, height, width, RAFT_MASK_CHANNELS];
    if mask.shape() != expected_mask.as_slice() {
        return Err(format!(
            "mask must have shape {expected_mask:?}, got {:?}",
            mask.shape()
        ));
    }

    let m = mask
        .reshape(&[height, width, 9, 8, 8])
        .and_then(|v| softmax_axis(&v, 2, None))
        .map_err(|e| e.to_string())?;
    let f = flow
        .index(0)
        .multiply(Array::from_f32(8.0))
        .map_err(|e| e.to_string())?;
    let pads = [(1, 1), (1, 1), (0, 0)];
    let padded = pad(&f, &pads, None, None).map_err(|e| e.to_string())?;
    let mut neighbors = Vec::with_capacity(9);
    for dy in 0..3 {
        for dx in 0..3 {
            neighbors.push(padded.index((dy..dy + height, dx..dx + width, ..)));
        }
    }
    let nb = stack(&neighbors[..], 2).map_err(|e| e.to_string())?;
    // Python: mx.einsum("hwkij,hwkc->hiwjc", m, nb).reshape(1, 8*h, 8*w, 2)
    let up = einsum("hwkij,hwkc->hiwjc", [&m, &nb]).map_err(|e| e.to_string())?;
    up.reshape(&[1, 8 * height, 8 * width, 2])
        .map_err(|e| e.to_string())
}

/// Batch wrapper for [`upsample_flow`], matching Python's `upsample_flow_batch`.
fn upsample_flow_batch(flow: &Array, mask: &Array) -> Result<Array, String> {
    let shape = flow.shape();
    if shape.len() != 4 || shape[0] <= 0 {
        return Err(format!(
            "flow must have shape [B,h,w,2] with positive batch, got {shape:?}"
        ));
    }
    if mask.shape().len() != 4 || mask.shape()[0] != shape[0] {
        return Err(format!(
            "mask batch must match flow batch {}, got {:?}",
            shape[0],
            mask.shape()
        ));
    }
    let mut parts = Vec::with_capacity(shape[0] as usize);
    for i in 0..shape[0] {
        parts.push(upsample_flow(
            &flow.index((i..i + 1, .., .., ..)),
            &mask.index((i..i + 1, .., .., ..)),
        )?);
    }
    concatenate(&parts[..], 0).map_err(|e| e.to_string())
}

// --- RAFT BasicUpdateBlock (checkpoint-backed, inference only) ---------------------------------------------

/// Hidden channels in the RAFT GRU and the update-block feature/context tensors.
pub const RAFT_HIDDEN_CHANNELS: i32 = 128;
/// Correlation channels produced by the production RAFT configuration (`levels=4`, `radius=4`).
pub const RAFT_CORR_CHANNELS: i32 = 324;
/// Correlation pyramid levels used by the production checkpoint and the Python `RAFT` defaults.
pub const RAFT_CORR_LEVELS: usize = 4;
/// Correlation lookup radius used by the production checkpoint and the Python `RAFT` defaults.
pub const RAFT_CORR_RADIUS: usize = 4;
/// Channels emitted by the mask head (the 9x9 convex-upsampling weights).
pub const RAFT_MASK_CHANNELS: i32 = 576;

/// The correlation/flow feature encoder inside the RAFT update block.
#[derive(Debug, mlx_rs::macros::ModuleParameters)]
pub struct MotionEncoder {
    #[param]
    convc1: Conv2d,
    #[param]
    convc2: Conv2d,
    #[param]
    convf1: Conv2d,
    #[param]
    convf2: Conv2d,
    #[param]
    conv: Conv2d,
}

impl MotionEncoder {
    fn load(reader: &mut ParamReader<'_>, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            convc1: reader.conv(
                &format!("{prefix}.convc1"),
                RAFT_CORR_CHANNELS,
                256,
                1,
                1,
                0,
            )?,
            convc2: reader.conv(&format!("{prefix}.convc2"), 256, 192, 3, 1, 1)?,
            convf1: reader.conv(&format!("{prefix}.convf1"), 2, 128, 7, 1, 3)?,
            convf2: reader.conv(&format!("{prefix}.convf2"), 128, 64, 3, 1, 1)?,
            conv: reader.conv(&format!("{prefix}.conv"), 256, 126, 3, 1, 1)?,
        })
    }

    /// Encode correlation and current flow, returning 128 channels: 126 learned plus the 2 flow channels.
    pub fn forward(&mut self, flow: &Array, corr: &Array) -> Result<Array, String> {
        let cor = self.convc1.forward(corr).map_err(|e| e.to_string())?;
        let cor = relu(&cor).map_err(|e| e.to_string())?;
        let cor = self.convc2.forward(&cor).map_err(|e| e.to_string())?;
        let cor = relu(&cor).map_err(|e| e.to_string())?;

        let flo = self.convf1.forward(flow).map_err(|e| e.to_string())?;
        let flo = relu(&flo).map_err(|e| e.to_string())?;
        let flo = self.convf2.forward(&flo).map_err(|e| e.to_string())?;
        let flo = relu(&flo).map_err(|e| e.to_string())?;

        let combined = concatenate(&[&cor, &flo], -1).map_err(|e| e.to_string())?;
        let out = self.conv.forward(&combined).map_err(|e| e.to_string())?;
        let out = relu(&out).map_err(|e| e.to_string())?;
        concatenate(&[&out, flow], -1).map_err(|e| e.to_string())
    }
}

/// RAFT's separable convolution GRU: one horizontal pass followed by one vertical pass.
#[derive(Debug, mlx_rs::macros::ModuleParameters)]
pub struct SepConvGRU {
    #[param]
    convz1: Conv2d,
    #[param]
    convr1: Conv2d,
    #[param]
    convq1: Conv2d,
    #[param]
    convz2: Conv2d,
    #[param]
    convr2: Conv2d,
    #[param]
    convq2: Conv2d,
}

impl SepConvGRU {
    fn load(reader: &mut ParamReader<'_>, prefix: &str) -> Result<Self, String> {
        let hidden = RAFT_HIDDEN_CHANNELS;
        let input = 2 * hidden; // concatenation of context (128) and motion features (128)
        let channels = hidden + input;
        Ok(Self {
            convz1: reader.conv_rect(
                &format!("{prefix}.convz1"),
                channels,
                hidden,
                (1, 5),
                (1, 1),
                (0, 2),
            )?,
            convr1: reader.conv_rect(
                &format!("{prefix}.convr1"),
                channels,
                hidden,
                (1, 5),
                (1, 1),
                (0, 2),
            )?,
            convq1: reader.conv_rect(
                &format!("{prefix}.convq1"),
                channels,
                hidden,
                (1, 5),
                (1, 1),
                (0, 2),
            )?,
            convz2: reader.conv_rect(
                &format!("{prefix}.convz2"),
                channels,
                hidden,
                (5, 1),
                (1, 1),
                (2, 0),
            )?,
            convr2: reader.conv_rect(
                &format!("{prefix}.convr2"),
                channels,
                hidden,
                (5, 1),
                (1, 1),
                (2, 0),
            )?,
            convq2: reader.conv_rect(
                &format!("{prefix}.convq2"),
                channels,
                hidden,
                (5, 1),
                (1, 1),
                (2, 0),
            )?,
        })
    }

    /// One gated recurrent update of the hidden state with the concatenated context/motion input.
    pub fn forward(&mut self, h: &Array, x: &Array) -> Result<Array, String> {
        let mut h = h.clone();
        h = sep_gru_step(&mut self.convz1, &mut self.convr1, &mut self.convq1, &h, x)?;
        sep_gru_step(&mut self.convz2, &mut self.convr2, &mut self.convq2, &h, x)
    }
}

fn sep_gru_step(
    z_conv: &mut Conv2d,
    r_conv: &mut Conv2d,
    q_conv: &mut Conv2d,
    h: &Array,
    x: &Array,
) -> Result<Array, String> {
    let hx = concatenate(&[h, x], -1).map_err(|e| e.to_string())?;
    let z = z_conv.forward(&hx).map_err(|e| e.to_string())?;
    let z = sigmoid(&z).map_err(|e| e.to_string())?;
    let r = r_conv.forward(&hx).map_err(|e| e.to_string())?;
    let r = sigmoid(&r).map_err(|e| e.to_string())?;

    let rh = &r * h;
    let q_input = concatenate(&[&rh, x], -1).map_err(|e| e.to_string())?;
    let q = q_conv.forward(&q_input).map_err(|e| e.to_string())?;
    let q = tanh(&q).map_err(|e| e.to_string())?;

    // Keep the scalar in the hidden-state dtype. A float32 scalar would promote
    // a float16 GRU to float32 and defeat the opt-in update-block path.
    let one = Array::from_f32(1.0)
        .as_dtype(h.dtype())
        .map_err(|e| e.to_string())?;
    Ok((&one - &z) * h + &z * &q)
}

/// Two-layer 3x3 flow head producing the per-iteration flow delta.
#[derive(Debug, mlx_rs::macros::ModuleParameters)]
pub struct FlowHead {
    #[param]
    conv1: Conv2d,
    #[param]
    conv2: Conv2d,
}

impl FlowHead {
    fn load(reader: &mut ParamReader<'_>, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            conv1: reader.conv(
                &format!("{prefix}.conv1"),
                RAFT_HIDDEN_CHANNELS,
                256,
                3,
                1,
                1,
            )?,
            conv2: reader.conv(&format!("{prefix}.conv2"), 256, 2, 3, 1, 1)?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, String> {
        let y = self.conv1.forward(x).map_err(|e| e.to_string())?;
        let y = relu(&y).map_err(|e| e.to_string())?;
        self.conv2.forward(&y).map_err(|e| e.to_string())
    }
}

/// Convex-upsampling mask head: 3x3 convolution, ReLU, then a 1x1 576-channel projection scaled by 0.25.
#[derive(Debug, mlx_rs::macros::ModuleParameters)]
pub struct MaskHead {
    #[param]
    conv1: Conv2d,
    #[param]
    conv2: Conv2d,
}

impl MaskHead {
    fn load(
        reader: &mut ParamReader<'_>,
        conv1_prefix: &str,
        conv2_prefix: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            conv1: reader.conv(conv1_prefix, RAFT_HIDDEN_CHANNELS, 256, 3, 1, 1)?,
            conv2: reader.conv(conv2_prefix, 256, RAFT_MASK_CHANNELS, 1, 1, 0)?,
        })
    }

    pub fn forward(&mut self, net: &Array) -> Result<Array, String> {
        let y = self.conv1.forward(net).map_err(|e| e.to_string())?;
        let y = relu(&y).map_err(|e| e.to_string())?;
        let y = self.conv2.forward(&y).map_err(|e| e.to_string())?;
        let scale = Array::from_f32(0.25)
            .as_dtype(y.dtype())
            .map_err(|e| e.to_string())?;
        Ok(&y * &scale)
    }
}

/// Output of one RAFT update-block core step.
#[derive(Debug)]
pub struct CoreOutput {
    pub net: Array,
    pub delta: Array,
}

/// Output of a complete RAFT update-block call, including the mask head.
#[derive(Debug)]
pub struct UpdateBlockOutput {
    pub net: Array,
    pub delta: Array,
    pub mask: Array,
}

/// Checkpoint-backed RAFT `BasicUpdateBlock` for the production 4-level, radius-4 correlation pyramid.
///
/// Its `ModuleParameters` projection is the compile state: exactly the update-block convolution
/// weights and biases, with no runtime tensors.
#[derive(Debug, mlx_rs::macros::ModuleParameters)]
pub struct BasicUpdateBlock {
    #[param]
    pub encoder: MotionEncoder,
    #[param]
    pub gru: SepConvGRU,
    #[param]
    pub flow_head: FlowHead,
    #[param]
    pub mask: MaskHead,
    // Pristine float32 weights, kept out of the compile-state projection so switching back to
    // float32 restores the checkpoint exactly instead of round-tripping through float16.
    fp32_parameters: std::collections::HashMap<String, Array>,
    parameter_dtype: Dtype,
}

impl BasicUpdateBlock {
    /// Load and validate the complete update block from a converted RAFT parameter map.
    ///
    /// Every `update_block.*` key is consumed exactly once by name and shape. Missing or
    /// unexpected parameters under that prefix are rejected; unrelated `fnet`/`cnet` keys
    /// in the same checkpoint are ignored. Parameters start in float32.
    pub fn load(params: &std::collections::HashMap<String, Array>) -> Result<Self, String> {
        let mut reader = ParamReader::new(params);
        let mut block = Self {
            encoder: MotionEncoder::load(&mut reader, "update_block.encoder")?,
            gru: SepConvGRU::load(&mut reader, "update_block.gru")?,
            flow_head: FlowHead::load(&mut reader, "update_block.flow_head")?,
            mask: MaskHead::load(&mut reader, "update_block.mask.0", "update_block.mask.2")?,
            fp32_parameters: std::collections::HashMap::new(),
            parameter_dtype: Dtype::Float32,
        };
        reader.finish("update_block")?;
        block.fp32_parameters = block
            .parameters()
            .flatten()
            .into_iter()
            .map(|(name, value)| (name.to_string(), value.clone()))
            .collect();
        Ok(block)
    }

    /// Return the dtype currently used by every update-block convolution weight and bias.
    pub fn dtype(&self) -> Dtype {
        self.parameter_dtype
    }

    /// Select float32 (default) or float16 update-block convolution parameters.
    ///
    /// Only this block's weights and biases are cast. Switching back to float32 restores the
    /// pristine checkpoint values captured at load time.
    pub fn set_dtype(&mut self, dtype: Dtype) -> Result<(), String> {
        if !matches!(dtype, Dtype::Float16 | Dtype::Float32) {
            return Err(format!(
                "update-block dtype must be float16 or float32, got {dtype:?}"
            ));
        }
        if dtype == self.parameter_dtype {
            return Ok(());
        }
        let sources = self.fp32_parameters.clone();
        for (name, value) in self.parameters_mut().flatten() {
            let source = sources
                .get(name.as_ref())
                .ok_or_else(|| format!("missing pristine float32 update-block parameter {name}"))?;
            *value = source.as_dtype(dtype).map_err(|e| e.to_string())?;
        }
        self.parameter_dtype = dtype;
        Ok(())
    }

    fn cast_input(&self, array: &Array) -> Result<Array, String> {
        if array.dtype() == self.parameter_dtype {
            Ok(array.clone())
        } else {
            array
                .as_dtype(self.parameter_dtype)
                .map_err(|e| e.to_string())
        }
    }

    /// Load the update block directly from a converted safetensors checkpoint.
    pub fn load_checkpoint(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let params = Array::load_safetensors(path).map_err(|e| e.to_string())?;
        Self::load(&params)
    }

    /// Run one GRU/flow-head step and return the updated hidden state and flow delta.
    ///
    /// `net` and `inp` are `[B,H,W,128]`, `corr` is `[B,H,W,324]`, and `flow` is `[B,H,W,2]`.
    /// `B` may be 1 (single pair) or 2 (batched forward/backward refinement).
    pub fn core(
        &mut self,
        net: &Array,
        inp: &Array,
        corr: &Array,
        flow: &Array,
    ) -> Result<CoreOutput, String> {
        validate_update_inputs(net, inp, corr, flow)?;
        let net = self.cast_input(net)?;
        let inp = self.cast_input(inp)?;
        let corr = self.cast_input(corr)?;
        let flow = self.cast_input(flow)?;
        let motion = self.encoder.forward(&flow, &corr)?;
        let gru_input = concatenate(&[&inp, &motion], -1).map_err(|e| e.to_string())?;
        let net = self.gru.forward(&net, &gru_input)?;
        let delta = self.flow_head.forward(&net)?;
        Ok(CoreOutput { net, delta })
    }

    /// Run the mask head on a `[B,H,W,128]` hidden state, returning `[B,H,W,576]` scaled by 0.25.
    ///
    /// The result uses the update-block parameter dtype; callers choose when to cast it back.
    pub fn mask_head(&mut self, net: &Array) -> Result<Array, String> {
        validate_hidden(net)?;
        let net = self.cast_input(net)?;
        self.mask.forward(&net)
    }

    /// Run the complete update block: core step followed by the mask head on the updated state.
    pub fn forward(
        &mut self,
        net: &Array,
        inp: &Array,
        corr: &Array,
        flow: &Array,
    ) -> Result<UpdateBlockOutput, String> {
        let core = self.core(net, inp, corr, flow)?;
        let mask = self.mask_head(&core.net)?;
        Ok(UpdateBlockOutput {
            net: core.net,
            delta: core.delta,
            mask,
        })
    }
}

type CompiledCoreFn =
    Box<dyn for<'a> FnMut(&mut BasicUpdateBlock, &'a [Array]) -> Result<Vec<Array>, Exception>>;

/// One cached `compile_with_state` update-block core step.
///
/// The compiled closure captures no model arrays. Every runtime tensor is a call input: net, inp,
/// corr and flow. The update block is passed as mutable compile state, so MLX supplies its
/// convolution weights and biases as graph inputs instead of treating captured arrays as constants.
struct CompiledCore {
    run: CompiledCoreFn,
}

impl std::fmt::Debug for CompiledCore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("CompiledCore")
    }
}

impl CompiledCore {
    fn new() -> Self {
        let run = compile_with_state(
            |state: &mut BasicUpdateBlock, args: &[Array]| -> Result<Vec<Array>, Exception> {
                if args.len() != 4 {
                    return Err(Exception::custom(format!(
                        "compiled update-block core expects 4 inputs (net, inp, corr, flow), got {}",
                        args.len()
                    )));
                }
                let core = state
                    .core(&args[0], &args[1], &args[2], &args[3])
                    .map_err(|error| Exception::custom(error))?;
                Ok(vec![core.net, core.delta])
            },
            None,
        );
        Self { run: Box::new(run) }
    }

    fn core(
        &mut self,
        state: &mut BasicUpdateBlock,
        net: &Array,
        inp: &Array,
        corr: &Array,
        flow: &Array,
    ) -> Result<CoreOutput, String> {
        let inputs = [net.clone(), inp.clone(), corr.clone(), flow.clone()];
        let mut outputs = (self.run)(state, &inputs).map_err(|error| error.to_string())?;
        if outputs.len() != 2 {
            return Err(format!(
                "compiled update-block core returned {} outputs, expected 2",
                outputs.len()
            ));
        }
        let delta = outputs.pop().expect("checked output count");
        let net = outputs.pop().expect("checked output count");
        Ok(CoreOutput { net, delta })
    }
}

fn validate_hidden(net: &Array) -> Result<(i32, i32), String> {
    let shape = net.shape();
    if shape.len() != 4 || shape[0] <= 0 || shape[3] != RAFT_HIDDEN_CHANNELS {
        return Err(format!(
            "net must have shape [B,H,W,{RAFT_HIDDEN_CHANNELS}] with positive batch, got {shape:?}"
        ));
    }
    if shape[1] <= 0 || shape[2] <= 0 {
        return Err(format!(
            "net spatial dimensions must be positive, got {shape:?}"
        ));
    }
    Ok((shape[1], shape[2]))
}

fn validate_update_inputs(
    net: &Array,
    inp: &Array,
    corr: &Array,
    flow: &Array,
) -> Result<(i32, i32), String> {
    let (height, width) = validate_hidden(net)?;
    let batch = net.shape()[0];
    for (name, array, channels) in [
        ("net", net, RAFT_HIDDEN_CHANNELS),
        ("inp", inp, RAFT_HIDDEN_CHANNELS),
        ("corr", corr, RAFT_CORR_CHANNELS),
        ("flow", flow, 2),
    ] {
        let expected = [batch, height, width, channels];
        if array.shape() != expected.as_slice() {
            return Err(format!(
                "{name} must have shape {expected:?} to match net, got {:?}",
                array.shape()
            ));
        }
    }
    Ok((height, width))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest;

    fn json_shape(value: &serde_json::Value) -> Vec<i32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect()
    }

    fn json_f32(value: &serde_json::Value, shape: &[i32]) -> Array {
        let data: Vec<f32> = value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let expected_len = shape.iter().map(|&d| d as usize).product::<usize>();
        assert_eq!(
            data.len(),
            expected_len,
            "fixture count for shape {shape:?}"
        );
        Array::from_slice(&data, shape)
    }

    fn max_abs_error(actual: &Array, expected: &Array) -> f32 {
        assert_eq!(actual.shape(), expected.shape());
        let actual = actual
            .as_dtype(Dtype::Float32)
            .unwrap()
            .contiguous()
            .unwrap();
        let expected = expected
            .as_dtype(Dtype::Float32)
            .unwrap()
            .contiguous()
            .unwrap();
        let actual = actual.as_slice::<f32>();
        let expected = expected.as_slice::<f32>();
        assert_eq!(actual.len(), expected.len());
        actual
            .iter()
            .zip(expected)
            .map(|(got, want)| (got - want).abs())
            .fold(0.0, f32::max)
    }

    fn max_normalized_error(actual: &Array, expected: &Array) -> f32 {
        assert_eq!(actual.shape(), expected.shape());
        let actual = actual
            .as_dtype(Dtype::Float32)
            .unwrap()
            .contiguous()
            .unwrap();
        let expected = expected.contiguous().unwrap();
        let actual = actual.as_slice::<f32>();
        let expected = expected.as_slice::<f32>();
        assert_eq!(actual.len(), expected.len());
        let scale = expected.iter().map(|value| value.abs()).fold(1.0, f32::max);
        actual
            .iter()
            .zip(expected)
            .map(|(got, want)| (got - want).abs())
            .fold(0.0, f32::max)
            / scale
    }

    #[test]
    fn corr_block_matches_python_fixture() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/corr_block_reference.json"))
                .unwrap();
        assert_eq!(fixture["schema"], "mlxstab.raft_compat.corr_block/2");
        assert_eq!(
            fixture["source"],
            "rust/raft-compat/tools/generate_corr_block_reference.py"
        );
        assert_eq!(fixture["rust_mlx_rs"], "0.32.0");
        assert_eq!(fixture["rust_mlx"], "0.32.2");
        assert_eq!(fixture["checkpoint_independent"].as_bool(), Some(true));
        assert!(fixture.get("checkpoint_sha256").is_none());
        assert!(fixture["python_mlx"].is_string());

        let feature_shape = json_shape(&fixture["feature_shape"]);
        let f1 = json_f32(&fixture["f1"], &feature_shape);
        let f2 = json_f32(&fixture["f2"], &feature_shape);
        let levels = fixture["levels"].as_u64().unwrap() as usize;
        let radius = fixture["radius"].as_u64().unwrap() as usize;
        let coords_shape = json_shape(&fixture["coords_shape"]);
        let coords = json_f32(&fixture["coords"], &coords_shape);

        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 2);
        for case in cases {
            let name = case["dtype"].as_str().unwrap();
            let dtype = match name {
                "float32" => Dtype::Float32,
                "float16" => Dtype::Float16,
                other => panic!("unexpected fixture dtype {other}"),
            };
            // fp16 matmul/pooling may round differently across MLX builds; fp32
            // is expected to match the Python graph almost exactly. Errors are
            // normalized the same way as tests/test_raft.py's torch comparison.
            let tolerance = if dtype == Dtype::Float32 { 1e-5 } else { 1e-3 };
            let block = CorrBlock::new(&f1, &f2, levels, radius, dtype).unwrap();
            assert_eq!(block.levels, levels);
            assert_eq!(block.radius, radius);
            assert_eq!(block.pyr.len(), levels);
            assert!(block.pyr.iter().all(|level| level.dtype() == dtype));

            let shapes = case["pyramid_shapes"].as_array().unwrap();
            let pyramid = case["pyramid"].as_array().unwrap();
            assert_eq!(block.pyr.len(), shapes.len());
            assert_eq!(block.pyr.len(), pyramid.len());
            let mut worst_pyramid = 0.0f32;
            for (level, ((actual, shape), expected)) in
                block.pyr.iter().zip(shapes).zip(pyramid).enumerate()
            {
                let shape = json_shape(shape);
                assert_eq!(actual.shape(), shape.as_slice(), "level {level} shape");
                let expected = json_f32(expected, &shape);
                let error = max_normalized_error(actual, &expected);
                worst_pyramid = worst_pyramid.max(error);
                assert!(
                    error < tolerance,
                    "{name} correlation level {level} max normalized error {error}"
                );
            }

            let lookup = block.lookup(&coords).unwrap();
            let lookup_shape = json_shape(&case["lookup_shape"]);
            assert_eq!(lookup.shape(), lookup_shape.as_slice());
            assert_eq!(lookup.dtype(), Dtype::Float32);
            let expected = json_f32(&case["lookup"], &lookup_shape);
            let lookup_error = max_normalized_error(&lookup, &expected);
            assert!(
                lookup_error < tolerance,
                "{name} lookup max normalized error {lookup_error}"
            );
            println!(
                "corr block {name}: worst pyramid error {worst_pyramid:e}, lookup error {lookup_error:e}"
            );
        }
    }

    #[test]
    fn pool2_drops_odd_trailing_rows_and_columns() {
        // The 3x3 grid loses row 2 and column 2; the kept cell is mean(1,2,4,5).
        let values: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let corr = Array::from_slice(&values, &[1, 3, 3, 1]);
        let pooled = pool2(&corr).unwrap();
        assert_eq!(pooled.shape(), &[1, 1, 1, 1]);
        assert_eq!(pooled.contiguous().unwrap().as_slice::<f32>(), &[3.0]);
    }

    #[test]
    fn corr_block_rejects_invalid_inputs() {
        let one = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 1, 2]);
        let error = CorrBlock::new(&one, &one, 0, 1, Dtype::Float32).unwrap_err();
        assert!(error.contains("levels"), "unexpected error: {error}");
        let error = CorrBlock::new(&one, &one, 1, 1, Dtype::Int32).unwrap_err();
        assert!(
            error.contains("float16 or float32"),
            "unexpected error: {error}"
        );
        let error = CorrBlock::new(&one, &one, 2, 1, Dtype::Float32).unwrap_err();
        assert!(
            error.contains("level 1 would be empty"),
            "unexpected error: {error}"
        );

        let wrong_rank = Array::from_slice(&[1.0f32, 2.0], &[2, 1]);
        assert!(CorrBlock::new(&wrong_rank, &wrong_rank, 1, 1, Dtype::Float32).is_err());

        let features = Array::from_slice(&[0.0f32; 8], &[1, 2, 2, 2]);
        let block = CorrBlock::new(&features, &features, 1, 1, Dtype::Float32).unwrap();
        let coords = Array::from_slice(&[0.0f32, 0.0], &[1, 1, 1, 2]);
        let error = block.lookup(&coords).unwrap_err();
        assert!(
            error.contains("coords cover 1 positions"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn lookup_matches_xy_order_bilinear_and_zero_padding() {
        let corr = Array::from_slice(&[0.0f32, 10., 20., 30., 40., 50.], &[1, 2, 3, 1]);
        // x=1.25, y=0.5 -> average of 12.5 and 42.5 = 27.5
        let coords = Array::from_slice(&[1.25f32, 0.5], &[1, 2]);
        let center = lookup_level(&corr, &coords, 0, 0).unwrap();
        assert_eq!(center.shape(), &[1, 1]);
        assert!((center.as_slice::<f32>()[0] - 27.5).abs() < 1e-6);

        // At (x=1,y=0), a is x and b is y: the 3x3 flattened grid is
        // [0,0,30, 0,10,40, 0,20,50] with out-of-frame taps zeroed.
        let integer = Array::from_slice(&[1.0f32, 0.0], &[1, 2]);
        let grid = lookup_level(&corr, &integer, 0, 1).unwrap();
        let expected = [0., 0., 30., 0., 10., 40., 0., 20., 50.];
        for (got, want) in grid.as_slice::<f32>().iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, expected {want}");
        }

        // Fractional x=-0.25 straddles the left edge: the valid x=0 column
        // contributes 0.75 of its bilinear weight (no clamp/renormalize).
        let edge = Array::from_slice(&[-0.25f32, 0.5], &[1, 2]);
        let edge_value = lookup_level(&corr, &edge, 0, 0).unwrap();
        assert!((edge_value.as_slice::<f32>()[0] - 11.25).abs() < 1e-6);
    }

    #[test]
    fn lookup_scales_level_zero_coordinates() {
        let corr = Array::from_slice(
            &(0..16).map(|x| x as f32).collect::<Vec<_>>(),
            &[1, 4, 4, 1],
        );
        // (2,2) in level-zero coordinates samples (1,1) at level 1 -> value 5.
        let coords = Array::from_slice(&[2.0f32, 2.0], &[1, 2]);
        let got = lookup_level(&corr, &coords, 1, 0).unwrap();
        assert_eq!(got.as_slice::<f32>(), &[5.0]);
    }

    #[test]
    fn lookup_rejects_invalid_shapes() {
        let corr = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 2, 1]);
        let coords = Array::from_slice(&[0.0f32, 0.0, 1.0, 1.0], &[2, 2]);
        assert!(
            lookup_level(&corr, &coords, 0, 0)
                .unwrap_err()
                .contains("matching corr")
        );
    }

    #[test]
    fn lookup_materializes_noncontiguous_and_fp16_inputs() {
        // Transpose H/W to make a non-contiguous view, then restore its shape.
        let base = Array::from_slice(&[0.0f32, 30., 10., 40., 20., 50.], &[1, 3, 2, 1]);
        let corr = base.transpose_axes(&[0, 2, 1, 3]).unwrap();
        assert_eq!(corr.shape(), &[1, 2, 3, 1]);
        let half = corr.as_dtype(Dtype::Float16).unwrap();
        let coords = Array::from_slice(&[1.25f32, 0.5], &[1, 2]);
        let got = lookup_level(&half, &coords, 0, 0).unwrap();
        assert_eq!(got.dtype(), Dtype::Float32);
        assert!((got.as_slice::<f32>()[0] - 27.5).abs() < 1e-6);
    }

    #[test]
    fn reads_checked_in_raft_safetensors() {
        let checkpoint = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../vendor/weights/conv/raft-things.safetensors");
        let checkpoint_hash = format!(
            "{:x}",
            sha2::Sha256::digest(std::fs::read(&checkpoint).unwrap())
        );
        assert_eq!(
            checkpoint_hash,
            "84c1f730d4620ef4dd0a2eeeef97db7a729fc4190ee05cdd0169030fc92fccc1"
        );
        let params = Array::load_safetensors(checkpoint).unwrap();
        assert_eq!(params.len(), 154);
        assert_eq!(params["cnet.conv1.weight"].shape(), &[64, 7, 7, 3]);
    }

    #[test]
    fn encoder_loader_rejects_missing_and_unexpected_parameters() {
        let checkpoint = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../vendor/weights/conv/raft-things.safetensors");
        let params = Array::load_safetensors(checkpoint).unwrap();

        let mut missing = params.clone();
        missing.remove("fnet.layer3.1.conv2.weight");
        let error = BasicEncoder::load(&missing, "fnet", EncoderNorm::Instance).unwrap_err();
        assert!(
            error.contains("missing parameter fnet.layer3.1.conv2.weight"),
            "unexpected error: {error}"
        );

        let mut extra = params.clone();
        extra.insert(
            "fnet.unexpected.weight".to_string(),
            Array::from_slice(&[0.0f32], &[1]),
        );
        let error = BasicEncoder::load(&extra, "fnet", EncoderNorm::Instance).unwrap_err();
        assert!(
            error.contains("unexpected parameters under fnet"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn basic_encoders_match_python_mlx_fixture() {
        let checkpoint = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../vendor/weights/conv/raft-things.safetensors");
        let checkpoint_hash = format!(
            "{:x}",
            sha2::Sha256::digest(std::fs::read(&checkpoint).unwrap())
        );
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/basic_encoder_reference.json"
        ))
        .unwrap();
        assert_eq!(fixture["schema"], "mlxstab.raft_compat.basic_encoder/1");
        assert_eq!(fixture["checkpoint_sha256"], checkpoint_hash);
        assert_eq!(fixture["rust_mlx_rs"], "0.32.0");
        assert_eq!(fixture["rust_mlx"], "0.32.2");

        let params = Array::load_safetensors(checkpoint).unwrap();
        let mut encoders = RaftEncoders::from_params(&params).unwrap();

        let input_shape: Vec<i32> = fixture["input_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let input: Vec<f32> = fixture["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(
            input.len(),
            input_shape.iter().map(|&d| d as usize).product::<usize>()
        );
        assert_eq!(input_shape, [2, 32, 40, 3]);
        let paired_input = Array::from_slice(&input, &input_shape);
        let cnet_input = Array::from_slice(&input[..32 * 40 * 3], &[1, 32, 40, 3]);

        for (name, norm) in [
            ("fnet", EncoderNorm::Instance),
            ("cnet", EncoderNorm::Batch),
        ] {
            let encoder_fixture = &fixture["encoders"][name];
            let expected: Vec<f32> = encoder_fixture["output"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let expected_shape: Vec<i32> = encoder_fixture["output_shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect();
            let expected_len = expected_shape
                .iter()
                .map(|&d| d as usize)
                .product::<usize>();
            assert_eq!(
                encoder_fixture["norm"],
                match norm {
                    EncoderNorm::Instance => "instance",
                    EncoderNorm::Batch => "batch",
                }
            );
            assert_eq!(expected.len(), expected_len);
            let output = match name {
                "fnet" => encoders.fnet.forward(&paired_input).unwrap(),
                "cnet" => encoders.cnet.forward(&cnet_input).unwrap(),
                _ => unreachable!(),
            };
            assert_eq!(output.shape(), expected_shape);
            let actual = output.contiguous().unwrap();
            let actual = actual.as_slice::<f32>();
            assert_eq!(actual.len(), expected.len());

            let mut max_abs_error = 0.0f32;
            let mut max_rel_error = 0.0f32;
            for (got, want) in actual.iter().zip(expected.iter()) {
                let abs = (got - want).abs();
                max_abs_error = max_abs_error.max(abs);
                if want.abs() > 1e-5 {
                    max_rel_error = max_rel_error.max(abs / want.abs());
                }
            }
            println!("{name}: max abs error {max_abs_error:e}, max rel error {max_rel_error:e}");
            assert!(
                max_abs_error < 1e-5,
                "{name} max abs error {max_abs_error} exceeds tolerance"
            );
        }
    }

    #[test]
    fn update_block_matches_python_mlx_fixture() {
        let checkpoint = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../vendor/weights/conv/raft-things.safetensors");
        let checkpoint_hash = format!(
            "{:x}",
            sha2::Sha256::digest(std::fs::read(&checkpoint).unwrap())
        );
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/update_block_reference.json"
        ))
        .unwrap();
        assert_eq!(fixture["schema"], "mlxstab.raft_compat.update_block/1");
        assert_eq!(fixture["checkpoint_sha256"], checkpoint_hash);
        assert_eq!(fixture["rust_mlx_rs"], "0.32.0");
        assert_eq!(fixture["rust_mlx"], "0.32.2");
        assert!(fixture["python_mlx"].is_string());
        assert_eq!(fixture["height"], 3);
        assert_eq!(fixture["width"], 4);
        assert_eq!(fixture["hidden_channels"], RAFT_HIDDEN_CHANNELS);
        assert_eq!(fixture["corr_channels"], RAFT_CORR_CHANNELS);
        assert_eq!(fixture["mask_channels"], RAFT_MASK_CHANNELS);

        let inputs = &fixture["inputs"];
        let net = json_f32(&inputs["net"], &json_shape(&inputs["net_shape"]));
        let inp = json_f32(&inputs["inp"], &json_shape(&inputs["inp_shape"]));
        let corr = json_f32(&inputs["corr"], &json_shape(&inputs["corr_shape"]));
        let flow = json_f32(&inputs["flow"], &json_shape(&inputs["flow_shape"]));
        assert_eq!(net.shape(), &[1, 3, 4, RAFT_HIDDEN_CHANNELS]);
        assert_eq!(inp.shape(), &[1, 3, 4, RAFT_HIDDEN_CHANNELS]);
        assert_eq!(corr.shape(), &[1, 3, 4, RAFT_CORR_CHANNELS]);
        assert_eq!(flow.shape(), &[1, 3, 4, 2]);

        let mut block = BasicUpdateBlock::load_checkpoint(&checkpoint).unwrap();
        let core = block.core(&net, &inp, &corr, &flow).unwrap();
        let mask = block.mask_head(&core.net).unwrap();
        let combined = block.forward(&net, &inp, &corr, &flow).unwrap();

        let expected_core = &fixture["core"];
        let expected_net = json_f32(
            &expected_core["net"],
            &json_shape(&expected_core["net_shape"]),
        );
        let expected_delta = json_f32(
            &expected_core["delta"],
            &json_shape(&expected_core["delta_shape"]),
        );
        let expected_mask = json_f32(&fixture["mask"], &json_shape(&fixture["mask_shape"]));
        assert_eq!(expected_net.shape(), &[1, 3, 4, RAFT_HIDDEN_CHANNELS]);
        assert_eq!(expected_delta.shape(), &[1, 3, 4, 2]);
        assert_eq!(expected_mask.shape(), &[1, 3, 4, RAFT_MASK_CHANNELS]);

        let tolerance = 1e-5f32;
        for (label, actual, expected) in [
            ("updated net", &core.net, &expected_net),
            ("delta flow", &core.delta, &expected_delta),
            ("mask", &mask, &expected_mask),
            ("combined net", &combined.net, &expected_net),
            ("combined delta", &combined.delta, &expected_delta),
            ("combined mask", &combined.mask, &expected_mask),
        ] {
            let error = max_abs_error(actual, expected);
            println!("update block {label}: max abs error {error:e}");
            assert!(
                error < tolerance,
                "update block {label} max abs error {error} exceeds {tolerance}"
            );
        }
    }

    #[test]
    fn update_block_fp16_matches_python_mlx_fixture() {
        let checkpoint = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../vendor/weights/conv/raft-things.safetensors");
        let checkpoint_hash = format!(
            "{:x}",
            sha2::Sha256::digest(std::fs::read(&checkpoint).unwrap())
        );
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/update_block_fp16_reference.json"
        ))
        .unwrap();
        assert_eq!(fixture["schema"], "mlxstab.raft_compat.update_block_fp16/1");
        assert_eq!(fixture["checkpoint_sha256"], checkpoint_hash);
        assert_eq!(fixture["rust_mlx_rs"], "0.32.0");
        assert_eq!(fixture["rust_mlx"], "0.32.2");
        assert!(fixture["python_mlx"].is_string());
        assert_eq!(fixture["update_weight_dtype"], "float16");
        assert_eq!(fixture["update_input_dtype"], "float16");
        assert_eq!(fixture["returned_dtype"], "float32");

        let inputs = &fixture["inputs"];
        let net = json_f32(&inputs["net"], &json_shape(&inputs["net_shape"]));
        let inp = json_f32(&inputs["inp"], &json_shape(&inputs["inp_shape"]));
        let corr = json_f32(&inputs["corr"], &json_shape(&inputs["corr_shape"]));
        let flow = json_f32(&inputs["flow"], &json_shape(&inputs["flow_shape"]));

        let mut block = BasicUpdateBlock::load_checkpoint(&checkpoint).unwrap();
        assert_eq!(block.dtype(), Dtype::Float32);
        block.set_dtype(Dtype::Float16).unwrap();
        assert_eq!(block.dtype(), Dtype::Float16);
        assert!(
            block
                .parameters()
                .flatten()
                .values()
                .all(|value| value.dtype() == Dtype::Float16),
            "all update-block parameters must be float16"
        );

        let core = block.core(&net, &inp, &corr, &flow).unwrap();
        let mask = block.mask_head(&core.net).unwrap();
        assert_eq!(core.net.dtype(), Dtype::Float16);
        assert_eq!(core.delta.dtype(), Dtype::Float16);
        assert_eq!(mask.dtype(), Dtype::Float16);
        let mask = mask.as_dtype(Dtype::Float32).unwrap();

        let expected_core = &fixture["core"];
        let expected_net = json_f32(
            &expected_core["net"],
            &json_shape(&expected_core["net_shape"]),
        );
        let expected_delta = json_f32(
            &expected_core["delta"],
            &json_shape(&expected_core["delta_shape"]),
        );
        let expected_mask = json_f32(&fixture["mask"], &json_shape(&fixture["mask_shape"]));

        for (label, actual, expected) in [
            ("updated net", &core.net, &expected_net),
            ("delta flow", &core.delta, &expected_delta),
            ("mask", &mask, &expected_mask),
        ] {
            // Measured against the Python reference: 9.765625e-4 for net/delta and
            // 1.953125e-3 for the mask (one float16 quantum at these magnitudes).
            let tolerance = if label == "mask" {
                3.0e-3f32
            } else {
                1.5e-3f32
            };
            let error = max_abs_error(actual, expected);
            println!("update block fp16 {label}: max abs error {error:e}");
            assert!(
                error < tolerance,
                "update block fp16 {label} max abs error {error} exceeds {tolerance}"
            );
        }

        block.set_dtype(Dtype::Float32).unwrap();
        assert_eq!(block.dtype(), Dtype::Float32);
        assert!(
            block
                .parameters()
                .flatten()
                .values()
                .all(|value| value.dtype() == Dtype::Float32),
            "switching back must restore float32 parameters"
        );
        assert!(block.set_dtype(Dtype::Int32).is_err());
    }

    #[test]
    fn update_block_loader_rejects_missing_extra_and_misshapen_parameters() {
        let checkpoint = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../vendor/weights/conv/raft-things.safetensors");
        let params = Array::load_safetensors(checkpoint).unwrap();

        let mut missing = params.clone();
        missing.remove("update_block.gru.convz2.weight");
        let error = BasicUpdateBlock::load(&missing).unwrap_err();
        assert!(
            error.contains("missing parameter update_block.gru.convz2.weight"),
            "unexpected error: {error}"
        );

        let mut extra = params.clone();
        extra.insert(
            "update_block.unexpected.weight".to_string(),
            Array::from_slice(&[0.0f32], &[1]),
        );
        let error = BasicUpdateBlock::load(&extra).unwrap_err();
        assert!(
            error.contains("unexpected parameters under update_block"),
            "unexpected error: {error}"
        );

        let mut misshapen = params.clone();
        misshapen.insert(
            "update_block.flow_head.conv1.bias".to_string(),
            Array::from_slice(&[0.0f32], &[1]),
        );
        let error = BasicUpdateBlock::load(&misshapen).unwrap_err();
        assert!(
            error.contains("parameter shape mismatch for update_block.flow_head.conv1.bias"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn update_block_rejects_invalid_input_shapes() {
        let checkpoint = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../vendor/weights/conv/raft-things.safetensors");
        let mut block = BasicUpdateBlock::load_checkpoint(checkpoint).unwrap();
        let net = mlx_rs::ops::zeros::<f32>(&[1, 3, 4, RAFT_HIDDEN_CHANNELS]).unwrap();
        let inp = mlx_rs::ops::zeros::<f32>(&[1, 3, 4, RAFT_HIDDEN_CHANNELS]).unwrap();
        let corr = mlx_rs::ops::zeros::<f32>(&[1, 3, 4, RAFT_CORR_CHANNELS]).unwrap();
        let flow = mlx_rs::ops::zeros::<f32>(&[1, 3, 4, 2]).unwrap();

        let wrong_corr = mlx_rs::ops::zeros::<f32>(&[1, 3, 4, 323]).unwrap();
        let error = block.core(&net, &inp, &wrong_corr, &flow).unwrap_err();
        assert!(
            error.contains("corr must have shape [1, 3, 4, 324]"),
            "unexpected error: {error}"
        );

        let wrong_spatial = mlx_rs::ops::zeros::<f32>(&[1, 2, 4, 2]).unwrap();
        let error = block.core(&net, &inp, &corr, &wrong_spatial).unwrap_err();
        assert!(
            error.contains("flow must have shape [1, 3, 4, 2]"),
            "unexpected error: {error}"
        );

        let wrong_hidden = mlx_rs::ops::zeros::<f32>(&[1, 3, 4, 127]).unwrap();
        let error = block.mask_head(&wrong_hidden).unwrap_err();
        assert!(
            error.contains("net must have shape [B,H,W,128]"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn update_block_accepts_batch_two() {
        let checkpoint = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../vendor/weights/conv/raft-things.safetensors");
        let mut block = BasicUpdateBlock::load_checkpoint(checkpoint).unwrap();
        let net = mlx_rs::ops::zeros::<f32>(&[2, 3, 4, RAFT_HIDDEN_CHANNELS]).unwrap();
        let inp = mlx_rs::ops::zeros::<f32>(&[2, 3, 4, RAFT_HIDDEN_CHANNELS]).unwrap();
        let corr = mlx_rs::ops::zeros::<f32>(&[2, 3, 4, RAFT_CORR_CHANNELS]).unwrap();
        let flow = mlx_rs::ops::zeros::<f32>(&[2, 3, 4, 2]).unwrap();
        let out = block.forward(&net, &inp, &corr, &flow).unwrap();
        assert_eq!(out.net.shape(), &[2, 3, 4, RAFT_HIDDEN_CHANNELS]);
        assert_eq!(out.delta.shape(), &[2, 3, 4, 2]);
        assert_eq!(out.mask.shape(), &[2, 3, 4, RAFT_MASK_CHANNELS]);
    }

    #[test]
    fn compiled_core_matches_direct_core() {
        let checkpoint = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../vendor/weights/conv/raft-things.safetensors");
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/update_block_reference.json"
        ))
        .unwrap();
        let inputs = &fixture["inputs"];
        let net = json_f32(&inputs["net"], &json_shape(&inputs["net_shape"]));
        let inp = json_f32(&inputs["inp"], &json_shape(&inputs["inp_shape"]));
        let corr = json_f32(&inputs["corr"], &json_shape(&inputs["corr_shape"]));
        let flow = json_f32(&inputs["flow"], &json_shape(&inputs["flow_shape"]));

        let mut block = BasicUpdateBlock::load_checkpoint(&checkpoint).unwrap();
        let direct = block.core(&net, &inp, &corr, &flow).unwrap();
        let mut compiled = CompiledCore::new();
        let compiled = compiled.core(&mut block, &net, &inp, &corr, &flow).unwrap();

        // A missing state projection would surface as an MLX compile error or as
        // captured-weight behavior rather than these matching outputs.
        let tolerance = 1e-5f32;
        for (label, actual, expected) in [
            ("net", &compiled.net, &direct.net),
            ("delta", &compiled.delta, &direct.delta),
        ] {
            let error = max_abs_error(actual, expected);
            println!("compiled vs direct core {label}: max abs error {error:e}");
            assert!(
                error < tolerance,
                "compiled vs direct core {label} max abs error {error} exceeds {tolerance}"
            );
        }
    }

    #[test]
    fn flow_bidir_matches_python_mlx_fixture() {
        let checkpoint = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../vendor/weights/conv/raft-things.safetensors");
        let checkpoint_hash = format!(
            "{:x}",
            sha2::Sha256::digest(std::fs::read(&checkpoint).unwrap())
        );
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/flow_bidir_reference.json"))
                .unwrap();
        assert_eq!(fixture["schema"], "mlxstab.raft_compat.flow_bidir/2");
        assert_eq!(fixture["checkpoint_sha256"], checkpoint_hash);
        assert_eq!(fixture["rust_mlx_rs"], "0.32.0");
        assert_eq!(fixture["rust_mlx"], "0.32.2");
        assert!(fixture["python_mlx"].is_string());
        assert_eq!(fixture["height"], 64);
        assert_eq!(fixture["width"], 64);
        let verified_iters: Vec<u64> = fixture["verified_iters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap())
            .collect();
        assert_eq!(verified_iters, [1, 24]);
        assert_eq!(fixture["corr_levels"], RAFT_CORR_LEVELS);
        assert_eq!(fixture["corr_radius"], RAFT_CORR_RADIUS);

        let images = &fixture["images"];
        let first = json_f32(&images["first"], &json_shape(&images["first_shape"]));
        let second = json_f32(&images["second"], &json_shape(&images["second_shape"]));
        assert_eq!(first.shape(), &[1, 64, 64, 3]);
        assert_eq!(second.shape(), &[1, 64, 64, 3]);

        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), verified_iters.len());
        let mut model = Raft::load(checkpoint).unwrap();
        assert!(
            model.compiled_update(),
            "compiled update must default to true"
        );
        for (case, expected_iters) in cases.iter().zip(&verified_iters) {
            let iters = case["iters"].as_u64().unwrap();
            assert_eq!(iters, *expected_iters);
            model.set_compiled_update(true);
            let compiled = model.flow_bidir(&first, &second, iters as usize).unwrap();
            model.set_compiled_update(false);
            assert!(!model.compiled_update());
            let direct = model.flow_bidir(&first, &second, iters as usize).unwrap();
            model.set_batched_encoding(true);
            let batched = model.flow_bidir(&first, &second, iters as usize).unwrap();
            model.set_batched_encoding(false);

            for (label, shape_key, compiled_output, direct_output, batched_output, expected) in [
                (
                    "low forward",
                    "low_forward_shape",
                    &compiled.low_forward,
                    &direct.low_forward,
                    &batched.low_forward,
                    &case["low_forward"],
                ),
                (
                    "full forward",
                    "up_forward_shape",
                    &compiled.up_forward,
                    &direct.up_forward,
                    &batched.up_forward,
                    &case["up_forward"],
                ),
                (
                    "low backward",
                    "low_backward_shape",
                    &compiled.low_backward,
                    &direct.low_backward,
                    &batched.low_backward,
                    &case["low_backward"],
                ),
                (
                    "full backward",
                    "up_backward_shape",
                    &compiled.up_backward,
                    &direct.up_backward,
                    &batched.up_backward,
                    &case["up_backward"],
                ),
            ] {
                let expected = json_f32(expected, &json_shape(&case[shape_key]));
                // The 1-iteration direct case is the original regression and stays
                // exact. The compiled graph fuses/reorders operations, so its 1-iteration
                // full-resolution output differs by ~2.7e-4 px after convex upsampling.
                // At 24 iterations small per-step differences accumulate; the observed
                // worst output is under 3e-3 px for both paths.
                let (direct_tolerance, compiled_tolerance) = if iters == 1 {
                    (1e-4, 1e-3)
                } else {
                    (5e-3, 5e-3)
                };
                let compiled_error = max_abs_error(compiled_output, &expected);
                let direct_error = max_abs_error(direct_output, &expected);
                let batched_error = max_abs_error(batched_output, &expected);
                let split_error = max_abs_error(compiled_output, direct_output);
                println!(
                    "flow_bidir {iters} iters {label}: compiled error {compiled_error:e}, \
                     direct error {direct_error:e}, batched error {batched_error:e}, \
                     compiled vs direct {split_error:e}"
                );
                assert!(
                    compiled_error < compiled_tolerance,
                    "compiled flow_bidir {iters} iters {label} max abs error {compiled_error} \
                     exceeds {compiled_tolerance}"
                );
                assert!(
                    direct_error < direct_tolerance,
                    "direct flow_bidir {iters} iters {label} max abs error {direct_error} \
                     exceeds {direct_tolerance}"
                );
                assert!(
                    batched_error < compiled_tolerance,
                    "batched flow_bidir {iters} iters {label} max abs error {batched_error} \
                     exceeds {compiled_tolerance}"
                );
            }
        }
    }

    #[test]
    fn flow_bidir_fp16_update_matches_python_mlx_fixture() {
        let checkpoint = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../vendor/weights/conv/raft-things.safetensors");
        let checkpoint_hash = format!(
            "{:x}",
            sha2::Sha256::digest(std::fs::read(&checkpoint).unwrap())
        );
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/flow_bidir_fp16_reference.json"
        ))
        .unwrap();
        assert_eq!(fixture["schema"], "mlxstab.raft_compat.flow_bidir_fp16/1");
        assert_eq!(fixture["checkpoint_sha256"], checkpoint_hash);
        assert_eq!(fixture["rust_mlx_rs"], "0.32.0");
        assert_eq!(fixture["rust_mlx"], "0.32.2");
        assert!(fixture["python_mlx"].is_string());
        assert_eq!(fixture["height"], 64);
        assert_eq!(fixture["width"], 64);
        assert_eq!(fixture["corr_dtype"], "float32");
        assert_eq!(fixture["coordinate_dtype"], "float32");
        assert_eq!(fixture["flow_accumulator_dtype"], "float32");
        assert_eq!(fixture["update_weight_dtype"], "float16");
        assert_eq!(fixture["update_input_dtype"], "float16");
        assert_eq!(fixture["returned_dtype"], "float32");
        let verified_iters: Vec<u64> = fixture["verified_iters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap())
            .collect();
        assert_eq!(verified_iters, [1, 24]);

        let images = &fixture["images"];
        let first = json_f32(&images["first"], &json_shape(&images["first_shape"]));
        let second = json_f32(&images["second"], &json_shape(&images["second_shape"]));

        let mut model = Raft::load(checkpoint).unwrap();
        assert_eq!(model.update_dtype(), Dtype::Float32);
        assert!(model.compiled_core.is_none());
        model.set_update_dtype(Dtype::Float16).unwrap();
        assert_eq!(model.update_dtype(), Dtype::Float16);
        assert!(
            model.compiled_core.is_none(),
            "changing update dtype must invalidate the compiled core"
        );

        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), verified_iters.len());
        for (case, expected_iters) in cases.iter().zip(&verified_iters) {
            let iters = case["iters"].as_u64().unwrap();
            assert_eq!(iters, *expected_iters);
            model.set_compiled_update(true);
            let compiled = model.flow_bidir(&first, &second, iters as usize).unwrap();
            assert!(
                model.compiled_core.is_some(),
                "compiled core should be rebuilt for the current update dtype"
            );
            model.set_compiled_update(false);
            let direct = model.flow_bidir(&first, &second, iters as usize).unwrap();

            for (label, shape_key, compiled_output, direct_output, expected) in [
                (
                    "low forward",
                    "low_forward_shape",
                    &compiled.low_forward,
                    &direct.low_forward,
                    &case["low_forward"],
                ),
                (
                    "full forward",
                    "up_forward_shape",
                    &compiled.up_forward,
                    &direct.up_forward,
                    &case["up_forward"],
                ),
                (
                    "low backward",
                    "low_backward_shape",
                    &compiled.low_backward,
                    &direct.low_backward,
                    &case["low_backward"],
                ),
                (
                    "full backward",
                    "up_backward_shape",
                    &compiled.up_backward,
                    &direct.up_backward,
                    &case["up_backward"],
                ),
            ] {
                let expected = json_f32(expected, &json_shape(&case[shape_key]));
                assert_eq!(compiled_output.dtype(), Dtype::Float32);
                assert_eq!(direct_output.dtype(), Dtype::Float32);
                let compiled_error = max_abs_error(compiled_output, &expected);
                let direct_error = max_abs_error(direct_output, &expected);
                println!(
                    "flow_bidir fp16 update {iters} iters {label}: compiled error \
                     {compiled_error:e}, direct error {direct_error:e}"
                );
                // Measured against the uncompiled Python reference: worst 1.775e-3 at
                // 1 iteration and 7.752e-3 at 24 iterations (full backward). The
                // compiled and direct Rust paths matched to the printed precision.
                let tolerance = if iters == 1 { 2.5e-3f32 } else { 1.0e-2f32 };
                assert!(
                    direct_error < tolerance,
                    "direct fp16-update flow_bidir {iters} iters {label} max abs error \
                     {direct_error} exceeds {tolerance}"
                );
                assert!(
                    compiled_error < tolerance,
                    "compiled fp16-update flow_bidir {iters} iters {label} max abs error \
                     {compiled_error} exceeds {tolerance}"
                );
            }
        }

        model.set_update_dtype(Dtype::Float32).unwrap();
        assert_eq!(model.update_dtype(), Dtype::Float32);
        assert!(model.compiled_core.is_none());
    }

    #[test]
    fn flow_bidir_rejects_invalid_images() {
        let checkpoint = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../vendor/weights/conv/raft-things.safetensors");
        let mut model = Raft::load(checkpoint).unwrap();
        let valid = mlx_rs::ops::zeros::<f32>(&[1, 64, 64, 3]).unwrap();
        let wrong_channels = mlx_rs::ops::zeros::<f32>(&[1, 64, 64, 2]).unwrap();
        let wrong_batch = mlx_rs::ops::zeros::<f32>(&[2, 64, 64, 3]).unwrap();
        let unaligned = mlx_rs::ops::zeros::<f32>(&[1, 60, 64, 3]).unwrap();

        assert!(model.flow_bidir(&valid, &wrong_channels, 1).is_err());
        assert!(model.flow_bidir(&wrong_batch, &valid, 1).is_err());
        let error = model.flow_bidir(&valid, &unaligned, 1).unwrap_err();
        assert!(
            error.contains("multiples of 8"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn cnet_conv1_matches_python_mlx_fixture() {
        let checkpoint = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../vendor/weights/conv/raft-things.safetensors");
        let params = Array::load_safetensors(checkpoint).unwrap();
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/cnet_conv1_reference.json"))
                .unwrap();
        assert_eq!(fixture["schema"], "mlxstab.raft_compat.cnet_conv1/1");
        let checkpoint = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../vendor/weights/conv/raft-things.safetensors");
        let checkpoint_hash = format!(
            "{:x}",
            sha2::Sha256::digest(std::fs::read(checkpoint).unwrap())
        );
        assert_eq!(fixture["checkpoint_sha256"], checkpoint_hash);

        let input: Vec<f32> = (0..9 * 11 * 3)
            .map(|i| (((i % 29) as f32) - 14.0) / 14.0)
            .collect();
        let input = Array::from_slice(&input, &[1, 9, 11, 3]);
        let output = cnet_conv1(&input, &params).unwrap().contiguous().unwrap();
        assert_eq!(output.shape(), &[1, 5, 6, 64]);
        let actual = output.as_slice::<f32>();
        let positions = fixture["positions_yx"].as_array().unwrap();
        let expected_rows = fixture["outputs"].as_array().unwrap();
        assert_eq!(positions.len(), expected_rows.len());
        let mut max_abs_error = 0.0f32;
        for (position, expected) in positions.iter().zip(expected_rows) {
            let y = position[0].as_u64().unwrap() as usize;
            let x = position[1].as_u64().unwrap() as usize;
            let expected = expected.as_array().unwrap();
            assert_eq!(expected.len(), 64);
            for channel in 0..64 {
                let got = actual[(y * 6 + x) * 64 + channel];
                let want = expected[channel].as_f64().unwrap() as f32;
                max_abs_error = max_abs_error.max((got - want).abs());
            }
        }
        assert!(max_abs_error < 2e-4, "max abs error {max_abs_error}");
    }
}
