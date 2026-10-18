//! Initial E20 compatibility spike for the MLX RAFT correlation lookup.
//!
//! This is a correctness prototype, not a production lookup kernel: it materializes
//! MLX inputs on the host and loops in Rust. The API and numerical behavior are
//! intentionally small so it can be compared against Python's `lookup_level`.

use mlx_rs::module::Module;
use mlx_rs::nn::{BatchNorm, Conv2d, relu};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype};

/// Bilinearly sample one RAFT correlation-pyramid level.
///
/// `corr` is NHWC `[N,H,W,1]`, and `coords` is `[N,2]` in level-zero `(x,y)`
/// coordinates. Neighborhood index `a` offsets x and `b` offsets y, matching
/// the reference implementation's meshgrid ordering. Out-of-bounds taps are
/// zero, and output has shape `[N,(2*radius+1)^2]` and dtype float32.
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
    if level > i32::MAX as usize {
        return Err("pyramid level exceeds supported range".into());
    }
    let n = cs[0] as usize;
    let h = cs[1] as usize;
    let w = cs[2] as usize;
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
    if n > i32::MAX as usize || k > i32::MAX as usize {
        return Err("output dimensions exceed MLX limits".into());
    }

    // Explicit materialization handles fp16 input and non-contiguous views. It
    // also makes this prototype's synchronization/copy cost visible.
    let corr_f32 = corr
        .as_dtype(Dtype::Float32)
        .map_err(|e| e.to_string())?
        .contiguous()
        .map_err(|e| e.to_string())?;
    let coords_f32 = coords
        .as_dtype(Dtype::Float32)
        .map_err(|e| e.to_string())?
        .contiguous()
        .map_err(|e| e.to_string())?;
    let cv = corr_f32.as_slice::<f32>();
    let xy = coords_f32.as_slice::<f32>();
    let mut out = vec![0.0f32; out_len];
    let scale = 2.0f32.powi(-(level as i32));
    let r = radius as isize;

    for batch in 0..n {
        let px = xy[batch * 2] * scale;
        let py = xy[batch * 2 + 1] * scale;
        for a in 0..diameter {
            for b in 0..diameter {
                let x = px + a as f32 - r as f32;
                let y = py + b as f32 - r as f32;
                let x0 = x.floor() as isize;
                let y0 = y.floor() as isize;
                let tx = x - x.floor();
                let ty = y - y.floor();
                let mut sum = 0.0f32;
                for j in 0..2 {
                    let yy = y0 + j;
                    if yy < 0 || yy >= h as isize {
                        continue;
                    }
                    let wy = if j == 0 { 1.0 - ty } else { ty };
                    for i in 0..2 {
                        let xx = x0 + i;
                        if xx < 0 || xx >= w as isize {
                            continue;
                        }
                        let wx = if i == 0 { 1.0 - tx } else { tx };
                        let index = ((batch * h + yy as usize) * w) + xx as usize;
                        sum += wx * wy * cv[index];
                    }
                }
                out[batch * k + a * diameter + b] = sum;
            }
        }
    }
    Ok(Array::from_slice(&out, &[n as i32, k as i32]))
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
        let weight = self.get(
            &format!("{prefix}.weight"),
            &[output_channels, kernel, kernel, input_channels],
        )?;
        let bias = self.get(&format!("{prefix}.bias"), &[output_channels])?;
        let mut conv = Conv2d::new(input_channels, output_channels, (kernel, kernel))
            .map_err(|e| e.to_string())?;
        conv.stride = (stride, stride);
        conv.padding = (padding, padding);
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
