# RAFT Rust/MLX compatibility spike

This crate starts E20 from `PLAN.md`. It pins the community `mlx-rs` binding to
0.32.0 and exercises parts of the Python RAFT contract: correlation lookup,
loading the checked-in converted RAFT safetensors file, the checkpoint-backed
`cnet.conv1` layer, and the complete `fnet` and `cnet` `BasicEncoder` stacks.

`RaftEncoders` loads both encoders from the converted checkpoint by their exact
safetensors names. It validates every expected shape and rejects unconsumed keys
under the encoder prefix. `fnet` uses parameter-free instance norm and `cnet`
uses the checkpoint's evaluation-mode batch-norm running statistics. Both
include the six residual blocks, all three stride-2 downsampling stages (the
first 7x7 convolution plus two downsample projections), and the final 1x1
projection. Inputs are normalized NHWC floats with height and width
already padded to multiples of 8, matching what `RAFT.encode` passes to the
Python encoders.

Two goldens cover deterministic inputs:

- `tests/fixtures/cnet_conv1_reference.json` for the first context convolution.
- `tests/fixtures/basic_encoder_reference.json` for all values on a normalized
  32x40 frame pair. `fnet` runs with the production pair batch (`[2,4,5,256]`);
  `cnet` runs on the first frame (`[1,4,5,256]`). It records the Python MLX
  version, Rust `mlx-rs`/MLX versions, and checkpoint SHA-256.

Regenerate them with:

```sh
.venv/bin/python rust/raft-compat/tools/generate_cnet_conv1_reference.py
.venv/bin/python rust/raft-compat/tools/generate_basic_encoder_reference.py
```

The lookup is deliberately a host-side correctness reference. It materializes
MLX inputs into contiguous f32 slices and builds a new MLX output array. It is
not a Metal kernel, does not run the RAFT update block or full flow pipeline, and
is not a production inference path. It makes an eventual fused/device-side port easier
to validate without hiding copies or synchronization.

Run on Apple Silicon with Rust and Xcode installed:

```sh
cargo test --manifest-path rust/raft-compat/Cargo.toml
```

`mlx-rs` 0.32.0 currently targets MLX 0.32.2; the Python environment in this
checkout has MLX 0.32.3. The full encoder outputs in the checked-in fixture match
exactly on this setup, but that is not full-model parity. E20 still requires an
exact version pairing, strict validation of the complete RAFT parameter schema,
the update block and correlation integration, compiled bidirectional RAFT
updates, final-flow comparisons, cancellation/error-recovery tests, and repeated
end-to-end runtime and memory measurements before any migration decision.
