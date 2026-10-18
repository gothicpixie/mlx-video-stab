# RAFT Rust/MLX compatibility spike

This crate starts E20 from `PLAN.md`. It pins the community `mlx-rs` binding to
0.32.0 and exercises parts of the Python RAFT contract: correlation lookup,
loading the checked-in converted RAFT safetensors file, and the checkpoint-backed
`cnet.conv1` layer compared with a Python MLX golden. Regenerate that golden
with `.venv/bin/python rust/raft-compat/tools/generate_cnet_conv1_reference.py`.

The lookup is deliberately a host-side correctness reference. It materializes
MLX inputs into contiguous f32 slices and builds a new MLX output array. It is
not a Metal kernel, does not run RAFT's encoders or update block, and is not a
production inference path. It makes an eventual fused/device-side port easier
to validate without hiding copies or synchronization.

Run on Apple Silicon with Rust and Xcode installed:

```sh
cargo test --manifest-path rust/raft-compat/Cargo.toml
```

`mlx-rs` 0.32.0 currently targets MLX 0.32.2; the Python environment in this
checkout has MLX 0.32.3. The sampled `cnet.conv1` fixture passes across those
versions, but this is not full-model parity. E20 still requires an exact version
pairing, strict full-model parameter-schema validation, the rest of both
encoders, compiled bidirectional RAFT updates, final-flow comparisons,
cancellation/error-recovery tests, and repeated end-to-end runtime and memory
measurements before any migration decision.
