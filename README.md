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
        It implements the three standard schemes &mdash; scalar (SQ8), product (PQ), and binary (BQ) &mdash; behind a single `Quantizer` trait, and reuses `iqdb-distance` for distance on the compressed codes.
    </p>
    <br>
    <hr>
    <p>
        <strong>MSRV is 1.87+</strong> (Rust 2024 edition). Scalar, product, and binary quantization. One trait. A quality/space dial.
    </p>
    <blockquote>
        <strong>Status: pre-1.0, public API frozen at 0.5.0.</strong> The surface is locked for the 1.x series; only additive, non-breaking changes land before <code>1.0.0</code>. See <a href="./CHANGELOG.md"><code>CHANGELOG.md</code></a>.
    </blockquote>
</div>

<hr>
<br>

<h2>What it does</h2>

- **Scalar quantization (SQ8)** &mdash; f32 → `u8` per dimension, ~4× compression; asymmetric distance under **every** metric
- **Product quantization (PQ)** &mdash; subvector k-means codebooks, up to ~192× compression, with batch-ADC scoring for IVF-PQ
- **Binary quantization (BQ)** &mdash; one bit per dimension, 32× compression, Hamming distance on packed `u64` words
- **One trait** &mdash; `train` → `quantize` → `distance`, every method fallible and returning a typed error
- **Asymmetric distance** &mdash; compress the database, keep the query in f32 for better recall
- **Deterministic** &mdash; same seed + same data ⇒ byte-identical PQ codebooks on every platform
- **Never panics on bad input** &mdash; empty, non-finite, mismatched, untrained, and unsupported-metric inputs return a typed `IqdbError`

<br>

## Installation

```toml
[dependencies]
iqdb-quantize = "0.5"
```

<br>

## Quick start

Train on a representative sample, then quantize and score. **Scalar (SQ8)** supports every metric:

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

let code = sq.quantize(&[0.11_f32, 0.21, 0.29]).unwrap();   // 3 bytes
let d = sq.distance(&[0.10_f32, 0.20, 0.30], &code, DistanceMetric::Cosine).unwrap();
assert!(d.is_finite());
```

**Product (PQ)** trades a little recall for big compression, and precomputes a query table for batch scoring:

```rust
use iqdb_quantize::{ProductQuantizer, Quantizer};
use iqdb_types::DistanceMetric;

let mut pq = ProductQuantizer::with_config(2, 4, 7); // M = 2 subvectors, K = 4, seed = 7
let training: Vec<Vec<f32>> = (0..16)
    .map(|i| { let f = i as f32; vec![f, f + 1.0, f + 2.0, f + 3.0] })
    .collect();
let refs: Vec<&[f32]> = training.iter().map(Vec::as_slice).collect();
pq.train(&refs).unwrap();

// Build the ADC table once per query, then score many codes against it.
let query = [1.0_f32, 2.0, 3.0, 4.0];
let tables = pq.build_query_tables(&query, DistanceMetric::Euclidean).unwrap();
let code = pq.quantize(&query).unwrap();                    // 2 bytes
let d = tables.distance(&code).unwrap();
assert!(d.is_finite());
```

**Binary (BQ)** is the highest-compression scheme, scored with Hamming distance:

```rust
use iqdb_quantize::{BinaryQuantizer, Quantizer};
use iqdb_types::DistanceMetric;

let mut bq = BinaryQuantizer::new();
bq.train(&[&[0.0_f32, 1.0, 2.0][..], &[2.0_f32, 1.0, 0.0][..]]).unwrap();

let code = bq.quantize(&[0.5_f32, 1.5, 2.5]).unwrap();       // packed u64 words
let d = bq.distance(&[0.5_f32, 1.5, 2.5], &code, DistanceMetric::Hamming).unwrap();
assert_eq!(d, 0.0); // self-distance is zero
```

Two rules to use quantization well: **train on representative data**, and **search quantized but rerank with full `f32`**. Skipping the rerank is the most common cause of "quantization broke recall" reports.

<br>

## How to use it

Every method of the `Quantizer` trait is fallible and returns `iqdb_types::Result`. The library never panics on bad input.

- **`ScalarQuantizer` (SQ8)** &mdash; per-dimension affine calibration; codes are `u8`. Supports every `DistanceMetric` via asymmetric distance through `iqdb-distance`.
- **`ProductQuantizer` (PQ)** &mdash; `M`-byte codes via deterministic k-means codebooks. `PqAdcTables` precomputes per-query lookup tables for batch scoring. Supports `Euclidean`, `DotProduct`, `Manhattan`; `Cosine` (no global norm — L2-normalize and use `DotProduct`) and `Hamming` (wrong code space) return `IqdbError::InvalidMetric`.
- **`BinaryQuantizer` (BQ)** &mdash; one bit per dimension, packed into `Vec<u64>`. Supports `DistanceMetric::Hamming` only; other metrics return `IqdbError::InvalidMetric`.

The full per-item reference, including the metric-support matrix and the error variants, is in <a href="./docs/API.md"><code>docs/API.md</code></a>.

<br>

## Status

<code>v0.5.0</code> &mdash; **feature-complete, API frozen.** SQ8, PQ, and BQ all ship behind a single `Quantizer` trait, with the `PqAdcTables` batch-ADC primitive, deterministic seeded k-means, property tests for round-trip and distance invariants, recall integration tests against full-`f32` baselines, `tracing` instrumentation, and a criterion bench harness. The public surface is locked for the 1.x series (the frozen item list is in the <a href="./dev/ROADMAP.md"><code>ROADMAP</code></a>); <code>1.0.0</code> adds the stability guarantee and real-consumer integration without changing the API. Verified on Windows, macOS, and Linux across stable and the 1.87 MSRV.

<hr>
<br>

## Where It Fits

`iqdb-quantize` is a Phase-2 crate, independent of the index layer. It builds on:

- `iqdb-types` &mdash; the `DistanceMetric`, `IqdbError`, and `Result` vocabulary
- `iqdb-distance` &mdash; the f32 distance kernels SQ8 and PQ delegate to

and is consumed by:

- `iqdb-ivf` &mdash; IVF-PQ scores in-cluster codes through `PqAdcTables`

It is an optimization, not a requirement: iQDB runs without it.

<br>

## Standards

Built to the iQDB Rust standard. See <a href="./REPS.md"><code>REPS.md</code></a> (Rust Efficiency &amp; Performance Standards) and <a href="./dev/DIRECTIVES.md"><code>dev/DIRECTIVES.md</code></a> for the engineering law and the definition of done. Before a PR: `cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, and `cargo test --all-features` must be clean.

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
