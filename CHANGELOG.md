# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

### Changed

### Fixed

### Security

---

## [1.0.0] - 2026-06-06

First stable release. The public API frozen at 0.5.0 is now committed under
SemVer for the 1.x series: no breaking changes until 2.0. Every
Definition-of-Done criterion (`dev/DIRECTIVES.md` §7) is satisfied, and the
surface is verified on Windows and Linux across the stable and 1.87 MSRV
toolchains. Nothing in the public surface changed since 0.5.0; this release adds
the consumer-simulation soak, runnable examples, and the stability commitment.

### Added

- Consumer-simulation suite (`tests/consumer_simulation.rs`): a mini IVF-PQ
  index built **only** on the public surface, reproducing the real consumer's
  pipeline — partition a corpus into coarse clusters, store each member as a
  `PqCode`, build the ADC tables once per query with `build_query_tables`, then
  scan the probed clusters through `PqAdcTables::distance`. It asserts batch ADC
  equals the single-shot `distance` bit-for-bit for every code and metric, that
  the PQ index recovers the correct cluster (purity &ge; 0.9) and that a
  PQ-shortlist + `f32`-rerank recovers the exact top-10 (overlap &ge; 0.9), that
  an SQ8 flat index preserves the exact top-10 (overlap &ge; 0.9), and that
  foreign code shapes and unsupported metrics are rejected, not panicked.
- `examples/`: five runnable, documented examples — `scalar_quantization`,
  `product_quantization` (with batch ADC), `binary_quantization`, `rerank` (the
  search-quantized-then-rerank quality path), and `compression` (the three
  schemes' code sizes side by side).

### Changed

- Declared **1.0 stable**: the frozen surface (recorded in `dev/ROADMAP.md`) is
  now under the SemVer 1.x compatibility guarantee. The **public API is unchanged
  from 0.5.0** — this release adds the consumer-simulation, examples, and the
  stability commitment. Compression is exact and deterministic: SQ8 4&times;, BQ
  32&times;, and PQ up to ~192&times; (`M = 16` on a 768-dim vector). Per-vector
  throughput benchmarked on Windows x86_64 at 768 dims (criterion medians): SQ8
  quantize ~1.56 µs, SQ8 asymmetric Cosine distance ~0.95 µs, BQ quantize
  ~0.64 µs, BQ Hamming ~0.68 µs.

---

## [0.5.0] - 2026-06-05

Training-stability, recall validation, and **API freeze**. The three schemes
from 0.2&ndash;0.4 are now measured end to end against full-precision baselines
on synthetic corpora, the training paths are instrumented through `tracing`, and
the public surface is locked for the 1.x series.

### Added

- Recall integration suite (`tests/recall.rs`) on Gaussian-cluster synthetic
  data: SQ8 asserts top-10 index overlap &ge; 0.9 against the full-`f32`
  baseline, BQ asserts top-10 cluster purity &ge; 0.7, and PQ asserts top-_k_
  overlap against the Euclidean `f32` baseline. Thresholds are taken from
  measured values on the seeded corpus with margin, not aspirational targets.
- Criterion bench harness (`benches/quantize.rs`) covering SQ8 quantize, SQ8
  asymmetric distance versus raw `f32` distance, BQ quantize, and BQ Hamming
  throughput, so any hot-path regression is caught against a tracked baseline.
- Tracing emission test (`tests/tracing.rs`): verifies that error-level events on
  failure paths and structured fields (e.g. `error.kind`) flow through the
  library's `tracing` instrumentation, using an inlined recording subscriber that
  captures events without installing a global one.

### Changed

- **Public API frozen for 1.x.** The surface declared complete at the 0.4.0
  feature freeze is now locked; only additive, non-breaking changes land before
  2.0. The frozen item list is recorded in `dev/ROADMAP.md`. `cargo audit` and
  `cargo deny check` are clean.
- Training boundaries across all three quantizers are instrumented with
  `tracing` at `info` level (per-vector encode/decode/distance stays hot-path
  only), emitting structured error events through `error-forge`'s `kind()` /
  `caption()` when a fallible training call fails.

---

## [0.4.0] - 2026-06-05

Binary quantization and **feature freeze**. The third and last scheme lands, the
asymmetric-distance story is complete across every quantizer, and the public
surface is declared frozen &mdash; no new public items before 1.0.

### Added

- `BinaryQuantizer` &mdash; binary quantization (BQ, 32&times; compression). One
  bit per dimension thresholded against a trained per-dimension mean; codes are
  packed into `Vec<u64>` words with the padding bits in the trailing word zeroed
  so they cannot contribute to Hamming distance. Supports
  `DistanceMetric::Hamming` only; every other metric returns
  `IqdbError::InvalidMetric`.
- `BqCode` &mdash; owned, immutable BQ code (`Debug`, `Clone`, `PartialEq`,
  `Eq`) with `dim`, `is_empty`, and `as_words` accessors and no public mutators;
  produced only by `BinaryQuantizer::quantize`.
- Property tests (`proptest`) for BQ: Hamming distance on the packed `u64` words
  matches a naive per-dimension popcount reference, and distance is finite and
  non-negative for every valid pair.

### Changed

- **Feature freeze declared.** With SQ8, PQ, and BQ all present and the
  asymmetric-distance path implemented for each, the public surface is complete.
  No new public items land before 1.0; there is no `todo!` or `unimplemented!`
  anywhere in shipping code.

---

## [0.3.0] - 2026-06-05

Product quantization. Each vector splits into `M` subvectors, each subvector
gets a learned `K`-centroid codebook, and asymmetric distance computation (ADC)
scores a stored code by table lookup rather than reconstruction &mdash; the
primitive IVF-PQ is built on.

### Added

- `ProductQuantizer` &mdash; product quantization (PQ, `M` bytes per code; at
  `M = 16, K = 256` a 768-dim `f32` vector compresses from 3072 B to 16 B).
  Build it with `ProductQuantizer::new` (standard `M = 8, K = 256, seed = 0`) or
  `ProductQuantizer::with_config(n_subvectors, n_centroids, seed)`; the
  `dim`, `n_subvectors`, `n_centroids`, and `seed` accessors report its
  configuration. Splits each input into `M` equal-length subvectors and learns a
  `K`-centroid codebook per position via hand-rolled k-means (k-means++ seeding,
  Lloyd's iterations, `MAX_ITERS = 25`, relative shift tolerance `1e-4`, `f64`
  accumulators downcast on commit, deterministic empty-cluster recovery).
  Supports `DistanceMetric::Euclidean`, `DistanceMetric::DotProduct`, and
  `DistanceMetric::Manhattan`; `Cosine` (needs a global norm &mdash; the
  documented path is to L2-normalize and use `DotProduct`) and `Hamming` (wrong
  code space) return `IqdbError::InvalidMetric`.
- `PqCode` &mdash; owned, immutable PQ code (`Debug`, `Clone`, `PartialEq`,
  `Eq`) with `dim`, `n_subvectors`, `len`, `is_empty`, and `as_bytes`
  accessors; produced only by `ProductQuantizer::quantize`.
- `PqAdcTables` + `ProductQuantizer::build_query_tables(query, metric)` &mdash;
  the query-side batch-ADC primitive. It builds the `M &times; K` lookup table
  once per `(query, metric)` and scores many `PqCode`s through
  `PqAdcTables::distance`, amortizing the table cost across an arbitrary set of
  codes. This is the path IVF-PQ's intra-cluster scan uses, scoring every code in
  every probed cluster against a single query. `ProductQuantizer::distance` is a
  thin wrapper around `build_query_tables` + `PqAdcTables::distance`, so the
  single-shot result is byte-for-byte identical to the batch path (proptested).
- Determinism contract: the same `seed` + the same training data produce
  byte-identical codebooks and codes on every supported platform, covered by
  `tests/determinism.rs` &mdash; two `ProductQuantizer`s trained alike produce
  identical codes for every probe input.
- Property tests (`proptest`) for PQ: distance finiteness across every supported
  metric, and a `pq_adc_matches_dequantize_then_compute` invariant proving ADC
  equals `dequantize` + `iqdb_distance::compute` within floating-point reduction
  tolerance for all subvector-decomposable metrics.

---

## [0.2.0] - 2026-06-05

Scalar quantization (SQ8) and the `Quantizer` trait. The first scheme lands
behind the trait every quantizer implements, with per-dimension affine
calibration, asymmetric distance through `iqdb-distance`, and typed-error input
validation.

### Added

- `Quantizer` trait with associated `train`, `quantize`, `dequantize`, and
  `distance` methods, every one fallible and returning `iqdb_types::Result` so
  bad input becomes a typed `IqdbError` rather than a panic.
- `ScalarQuantizer` &mdash; scalar quantization (SQ8, 4&times; compression).
  Per-dimension affine calibration with `u8` codes; the zero-range
  (`max == min`) lane is guarded so encoding never divides by zero. Asymmetric
  distance keeps the query in `f32`, dequantizes the candidate to a temporary
  buffer, and routes through `iqdb_distance::compute` for every
  `DistanceMetric`.
- `Sq8Code` &mdash; owned, immutable SQ8 code (`Debug`, `Clone`, `PartialEq`,
  `Eq`) with `len`, `is_empty`, and `as_bytes` accessors and no public mutators;
  produced only by `ScalarQuantizer::quantize`.
- `VERSION` constant exposing the crate's compile-time `CARGO_PKG_VERSION` for
  diagnostics and version-skew checks across the iqdb crate family.
- Input validation surfacing typed `iqdb_types::IqdbError`: empty or non-finite
  vectors as `InvalidVector`, dimension drift as `DimensionMismatch`, an empty
  training set or a call before `train` as `InvalidConfig`. The library never
  panics on bad input.
- Property tests (`proptest`) for SQ8 round-trip error bounds, distance
  finiteness, and metric-aware non-negativity (skipping `DotProduct`, which is
  stored as `-dot` and is legitimately negative).
- Edge-case coverage: empty / single-vector training, zero-range dimensions,
  dimension mismatch, quantize/distance before train, NaN / infinite inputs, and
  boundary clamp behaviour outside the trained range.

### Changed

- Now depends on `iqdb-types` 1.0 (shared `DistanceMetric` / `IqdbError` /
  `Result` vocabulary) and `iqdb-distance` 1.0 (the f32 distance kernels the SQ8
  asymmetric path delegates to).

---

## [0.1.0] - 2026-05-30

Initial scaffold and repository bootstrap. No domain logic yet &mdash; this
release establishes the structure, tooling, and quality gates the quantization
layer is built on.

### Added

- `Cargo.toml` with crate metadata, Rust 2024 edition, MSRV 1.87.
- Dual `Apache-2.0 OR MIT` license files.
- `README.md`, `CHANGELOG.md`, and a documentation skeleton (`docs/API.md`).
- `REPS.md` compliance baseline.
- `.github/workflows/ci.yml` CI matrix; `deny.toml`, `clippy.toml`, `rustfmt.toml`.
- `dev/DIRECTIVES.md` and `dev/ROADMAP.md` (committed engineering standards + plan).

[Unreleased]: https://github.com/jamesgober/iqdb-quantize/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/jamesgober/iqdb-quantize/compare/v0.5.0...v1.0.0
[0.5.0]: https://github.com/jamesgober/iqdb-quantize/releases/tag/v0.5.0
