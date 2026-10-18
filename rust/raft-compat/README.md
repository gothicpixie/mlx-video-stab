# RAFT Rust/MLX compatibility spike

This crate starts E20 from `PLAN.md`. It pins the community `mlx-rs` binding to
0.32.0 and exercises a small part of the Python RAFT contract: correlation
lookup and loading the checked-in converted RAFT safetensors file.

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
checkout has MLX 0.32.3. The present tests establish API/build feasibility and
lookup semantics, not cross-version model parity. E20 still requires an exact
version pairing, strict full-model parameter-schema validation, compiled
bidirectional RAFT updates, Python-vs-Rust intermediate/final comparisons,
cancellation/error-recovery tests, and repeated end-to-end runtime and memory
measurements before any migration decision.
