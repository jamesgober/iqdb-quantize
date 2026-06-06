<h1 align="center">
    <img width="99" alt="Rust logo" src="https://raw.githubusercontent.com/jamesgober/rust-collection/72baabd71f00e14aa9184efcb16fa3deddda3a0a/assets/rust-logo.svg">
    <br>
    <b>iqdb-quantize</b>
    <br>
    <sub><sup>iQDB VECTOR QUANTIZATION</sup></sub>
</h1>

<div align="center">
    <a href="https://crates.io/crates/iqdb-quantize"><img alt="Crates.io" src="https://img.shields.io/crates/v/iqdb-quantize"></a>
    <a href="https://crates.io/crates/iqdb-quantize"><img alt="Downloads" src="https://img.shields.io/crates/d/iqdb-quantize?color=%230099ff"></a>
    <a href="https://docs.rs/iqdb-quantize"><img alt="docs.rs" src="https://img.shields.io/docsrs/iqdb-quantize"></a>
    <a href="https://github.com/jamesgober/iqdb-quantize/actions"><img alt="CI" src="https://github.com/jamesgober/iqdb-quantize/actions/workflows/ci.yml/badge.svg"></a>
    <a href="https://github.com/rust-lang/rfcs/blob/master/text/2495-min-rust-version.md"><img alt="MSRV" src="https://img.shields.io/badge/MSRV-1.87%2B-blue"></a>
</div>

<br>

<div align="left">
    <p>
        <strong>iqdb-quantize</strong> compresses f32 vectors into smaller representations while preserving search quality. A million 768-dim vectors drop from ~3 GB to as little as ~96 MB, trading a controlled amount of recall for memory.
    </p>
    <p>
        It implements the three standard schemes (scalar, product, binary) behind a common `Quantizer` trait, and reuses `iqdb-distance` for distance on quantized codes.
    </p>
    <br>
    <hr>
    <p>
        <strong>MSRV is 1.87+</strong> (Rust 2024 edition). Scalar, product, and binary quantization. Quality/space dial.
    </p>
    <blockquote>
        <strong>Status: pre-1.0, in active development.</strong> The public API is being designed across the 0.x series and frozen at <code>1.0.0</code>. See <a href="./CHANGELOG.md"><code>CHANGELOG.md</code></a>.
    </blockquote>
</div>

<hr>
<br>

<h2>What it does</h2>

- **Scalar quantization** &mdash; SQ8: f32 to int8, ~4x compression
- **Product quantization** &mdash; PQ: subvector codebooks, 8x-16x compression
- **Binary quantization** &mdash; BQ: sign-based, 32x compression with Hamming distance
- **Train / quantize / distance** &mdash; compute distance directly on the compressed form where possible
- **Asymmetric distance** &mdash; quantize the database, keep the query in f32 for better recall


<br>

## Installation

```toml
[dependencies]
iqdb-quantize = "0.1"
```

<br>

## Example

```rust
use iqdb_quantize::{Quantizer, ScalarQuantizer};
use iqdb_types::DistanceMetric;

let training: Vec<Vec<f32>> = vec![
    vec![0.10, 0.20, 0.30],
    vec![0.15, 0.18, 0.32],
    vec![0.12, 0.22, 0.28],
];
let refs: Vec<&[f32]> = training.iter().map(Vec::as_slice).collect();

let mut sq = ScalarQuantizer::new();
sq.train(&refs).unwrap();

let candidate = [0.11_f32, 0.21, 0.29];
let code = sq.quantize(&candidate).unwrap();

let query = [0.10_f32, 0.20, 0.30];
let d = sq.distance(&query, &code, DistanceMetric::Cosine).unwrap();
assert!(d.is_finite());
```

Two rules to use quantization correctly: **train on representative data**, and **search quantized but rerank with full `f32`**. Skipping the rerank step is the most common cause of "quantization broke recall" reports.

<br>

## How to use it

Every method of the `Quantizer` trait is fallible and returns `iqdb_types::Result`. The library never panics on bad input.

- **`ScalarQuantizer` (SQ8)** &mdash; per-dimension affine calibration; codes are `u8`. Supports every `DistanceMetric` via asymmetric distance through `iqdb-distance`.
- **`BinaryQuantizer` (BQ)** &mdash; one bit per dimension, packed into `Vec<u64>`. Supports `DistanceMetric::Hamming` only; other metrics return `IqdbError::InvalidMetric`.
- **`ProductQuantizer` (PQ)** &mdash; `M`-byte codes via deterministic k-means codebooks (`PqAdcTables` precomputes per-query lookup tables for batch ADC scoring). Supports `Euclidean`, `DotProduct`, `Manhattan`; `Cosine` (no global norm) and `Hamming` (wrong code space) return `IqdbError::InvalidMetric`.

<br>

## Status

This is the <code>v0.1.0</code> release: SQ8, BQ, and PQ quantization land behind a single `Quantizer` trait, with the `PqAdcTables` batch-ADC primitive, deterministic seeded k-means, property tests for round-trip and distance invariants, recall integration tests, and a criterion bench harness. The public API stabilises across the 0.x series and freezes at <code>1.0.0</code> &mdash; see the <a href="./dev/ROADMAP.md"><code>ROADMAP</code></a> and <a href="./docs/API.md"><code>docs/API.md</code></a>.

<hr>
<br>

## Where It Fits

`iqdb-quantize` is a Phase-2 crate, independent of the index layer. It is used by:

- `iqdb-types` &mdash; vector and metric types
- `iqdb-distance` &mdash; distance on quantized codes
- `iqdb-ivf` &mdash; IVF-PQ consumes this for in-cluster compression

It is an optimization, not a requirement: iQDB runs without it.

<br>

## Contributing

See <a href="./dev/DIRECTIVES.md"><code>dev/DIRECTIVES.md</code></a> for engineering standards and the definition of done. Before a PR: `cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, and `cargo test --all-features` must be clean.

<br>

<div id="license">
    <h2>License</h2>
    <p>Licensed under either of</p>
    <ul>
        <li><b>Apache License, Version 2.0</b> &mdash; <a href="./LICENSE-APACHE">LICENSE-APACHE</a></li>
        <li><b>MIT License</b> &mdash; <a href="./LICENSE-MIT">LICENSE-MIT</a></li>
    </ul>
    <p>at your option.</p>
</div>

<div align="center">
  <h2></h2>
  <sup>COPYRIGHT <small>&copy;</small> 2026 <strong>JAMES GOBER.</strong></sup>
</div>
