# RAFT Rust/MLX compatibility spike

This crate starts E20 from `PLAN.md`. It pins the community `mlx-rs` binding to
0.32.0 and exercises parts of the Python RAFT contract: correlation lookup,
the all-pairs correlation pyramid, loading the checked-in converted RAFT
safetensors file, the checkpoint-backed `cnet.conv1` layer, and the complete
`fnet` and `cnet` `BasicEncoder` stacks.

`CorrBlock` builds level-zero correlation from paired NHWC features as
`(f1 @ f2^T) / sqrt(C)`. Each coarser level uses floor-sized 2x2 averages,
dropping any odd trailing row or column exactly like Python's `_pool2`.
`CorrBlock::lookup` flattens an `[1,H,W,2]` coordinate grid, samples every
level through the tested `lookup_level` XY-order/zero-padding contract, and
returns `[1,H,W,levels*(2*radius+1)^2]` float32 values.

`RaftEncoders` loads both encoders from the converted checkpoint by their exact
safetensors names. It validates every expected shape and rejects unconsumed keys
under the encoder prefix. `fnet` uses parameter-free instance norm and `cnet`
uses the checkpoint's evaluation-mode batch-norm running statistics. Both
include the six residual blocks, all three stride-2 downsampling stages (the
first 7x7 convolution plus two downsample projections), and the final 1x1
projection. Inputs are normalized NHWC floats with height and width
already padded to multiples of 8, matching what `RAFT.encode` passes to the
Python encoders.

Three goldens cover deterministic inputs:

- `tests/fixtures/cnet_conv1_reference.json` for the first context convolution.
- `tests/fixtures/basic_encoder_reference.json` for all values on a normalized
  32x40 frame pair. `fnet` runs with the production pair batch (`[2,4,5,256]`);
  `cnet` runs on the first frame (`[1,4,5,256]`). It records the Python MLX
  version, Rust `mlx-rs`/MLX versions, and checkpoint SHA-256.
- `tests/fixtures/corr_block_reference.json` for every correlation-pyramid value
  and every lookup value on deterministic 7x9x5 feature maps, in both float32
  and float16 correlation storage. It is checkpoint-independent, uses three
  levels with radius two, and records the Python MLX/Rust version metadata
  without a checkpoint hash.

Regenerate them with:

```sh
.venv/bin/python rust/raft-compat/tools/generate_cnet_conv1_reference.py
.venv/bin/python rust/raft-compat/tools/generate_basic_encoder_reference.py
.venv/bin/python rust/raft-compat/tools/generate_corr_block_reference.py
```

The lookup is deliberately a host-side correctness reference: it materializes
MLX inputs into contiguous f32 slices and builds a new MLX output array.
`CorrBlock` construction and pooling use ordinary MLX operations on-device.
Neither path is a fused Metal kernel or a production inference path, and neither
runs the RAFT update block or full flow pipeline. That makes an eventual
fused/device-side port easier to validate without hiding copies or
synchronization.

Run on Apple Silicon with Rust and Xcode installed:

```sh
cargo test --manifest-path rust/raft-compat/Cargo.toml
```

`mlx-rs` 0.32.0 currently targets MLX 0.32.2; the Python environment in this
checkout has MLX 0.32.3. The full encoder outputs in the checked-in fixture match
exactly on this setup, but that is not full-model parity. E20 still requires an
exact version pairing, strict validation of the complete RAFT parameter schema,
the update block, compiled bidirectional RAFT updates, final-flow comparisons,
cancellation/error-recovery tests, and repeated end-to-end runtime and memory
measurements before any migration decision.
