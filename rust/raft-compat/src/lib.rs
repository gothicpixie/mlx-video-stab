//! Initial E20 compatibility spike for the MLX RAFT correlation lookup.
//!
//! This is a correctness prototype, not a production lookup kernel: it materializes
//! MLX inputs on the host and loops in Rust. The API and numerical behavior are
//! intentionally small so it can be compared against Python's `lookup_level`.

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

#[cfg(test)]
mod tests {
    use super::*;

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
        let params = Array::load_safetensors(checkpoint).unwrap();
        assert_eq!(params.len(), 154);
        assert_eq!(params["cnet.conv1.weight"].shape(), &[64, 7, 7, 3]);
    }
}
