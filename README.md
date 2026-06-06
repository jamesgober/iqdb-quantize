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

## Status

This is the <code>v0.1.0</code> scaffold: structure, tooling, and quality gates are in place; the implementation lands across the 0.x series per the <a href="./dev/ROADMAP.md"><code>ROADMAP</code></a> and <a href="./docs/API.md"><code>docs/API.md</code></a>.

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
