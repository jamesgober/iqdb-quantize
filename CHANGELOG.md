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

## [0.1.0] - 2026-06-05

Initial public surface for the **iqdb** quantization layer. Three quantization schemes &mdash; scalar (SQ8), binary (BQ), and product (PQ) &mdash; land together behind a single `Quantizer` trait, with the `PqAdcTables` batch-ADC primitive, deterministic seeded k-means, property and recall coverage, and a criterion bench harness.

### Added

<<<<<<< HEAD
<<<<<<< HEAD
- `Cargo.toml` with crate metadata, Rust 2024 edition, MSRV 1.87.
- Dual `Apache-2.0 OR MIT` license files.
- `README.md`, `CHANGELOG.md`, and a documentation skeleton.
- `REPS.md` compliance baseline.
- `.github/workflows/ci.yml` CI matrix; `deny.toml`, `clippy.toml`, `rustfmt.toml`.
- `dev/DIRECTIVES.md` and `dev/ROADMAP.md` (committed engineering standards + plan).
=======
=======
>>>>>>> a62852ef3ed25f477b97eef1f7cbec854963f422
- `Quantizer` trait with associated `train`, `quantize`, `dequantize`, and `distance` methods, all returning `iqdb_types::Result`.
- `ScalarQuantizer` &mdash; scalar quantization (SQ8, 4&times; compression). Per-dimension affine calibration with `u8` codes; the zero-range (`max == min`) case is guarded to avoid division by zero. Asymmetric distance dequantizes the candidate and routes through `iqdb_distance::compute`; supports every `DistanceMetric` variant.
- `BinaryQuantizer` &mdash; binary quantization (BQ, 32&times; compression). One bit per dimension thresholded against a trained per-dimension mean; codes are packed into `Vec<u64>` words with padding bits zeroed in the trailing word so they cannot contribute to Hamming. Supports `DistanceMetric::Hamming` only; other metrics return `IqdbError::InvalidMetric`.
- `Sq8Code` and `BqCode` &mdash; owned, immutable code types with `Debug`, `Clone`, `PartialEq`, `Eq`. No public mutators.
- `ProductQuantizer` &mdash; product quantization (PQ, `M` bytes per code; at `M = 16, K = 256` that compresses a 768-dim `f32` vector from 3072 B to 16 B). Splits each input into `M` equal-length subvectors and learns a `K`-centroid codebook per position via hand-rolled k-means (k-means++ seeding, Lloyd's iterations, `MAX_ITERS = 25`, relative shift tolerance `1e-4`, f64 accumulators downcast on commit, deterministic empty-cluster recovery). Asymmetric distance computation (ADC) precomputes a per-subvector distance table from the query to all `K` centroids, then scores each stored code with `M` table lookups + a single sum (Euclidean takes one final `sqrt`). Supports `DistanceMetric::Euclidean`, `DistanceMetric::DotProduct`, and `DistanceMetric::Manhattan`; `Cosine` (needs a global norm &mdash; workaround: L2-normalize and use `DotProduct`) and `Hamming` (wrong code space) return `IqdbError::InvalidMetric`. Determinism contract: same `seed` + same training data &rArr; byte-identical codebooks and codes on every supported platform.
- `PqCode` &mdash; owned, immutable PQ code with `Debug`, `Clone`, `PartialEq`, `Eq`. No public mutators; produced only by `ProductQuantizer::quantize`.
- `PqAdcTables` + `ProductQuantizer::build_query_tables(query, metric)` &mdash; query-side batch ADC scoring primitive. Builds the `M &times; K` lookup table once per `(query, metric)` and scores many `PqCode`s against it via `PqAdcTables::distance`, amortizing the table cost across an arbitrary set of codes. Designed for IVF-PQ's intra-cluster scan, which scores every code in every probed cluster against a single query. `ProductQuantizer::distance` is a thin wrapper around this path &mdash; `build_query_tables` + `PqAdcTables::distance` &mdash; so the per-code single-shot result is unchanged byte-for-byte (verified by a proptest).
- Property tests (proptest) for PQ: distance finiteness across every supported metric and a `pq_adc_matches_dequantize_then_compute` invariant proving ADC equals `dequantize` + `iqdb_distance::compute` within floating-point reduction tolerance for all subvector-decomposable metrics.
- Recall integration test on Gaussian-cluster synthetic data: SQ8 asserts top-10 index overlap &ge; 0.9 against the full-`f32` baseline; BQ asserts top-10 cluster purity &ge; 0.7; PQ asserts top-_k_ overlap against the Euclidean `f32` baseline. Thresholds are taken from actually measured values on the seeded corpus with margin, not aspirational targets.
- Determinism integration test (`tests/determinism.rs`): two `ProductQuantizer` instances trained with the same seed + same data produce byte-identical codes for every probe input.
- Edge-case coverage: empty / single-vector training, zero-range dimensions, dimension mismatch, quantize/distance before train, NaN / infinite inputs, boundary clamp behaviour outside the trained range.
- Property tests (proptest) for round-trip error bounds, distance finiteness, metric-aware non-negativity (skipping `DotProduct`, which is stored as `-dot` and is legitimately negative), and a naive popcount reference for BQ Hamming.
- Tracing emission test (`tests/tracing.rs`): verifies that error-level events on failure paths and structured fields (e.g., `error.kind`) flow through the library's `tracing` instrumentation. Uses an inlined recording-subscriber helper to capture events without installing a global subscriber.
- Criterion bench harness (`benches/quantize.rs`) covering SQ8 quantize, SQ8 asymmetric distance versus raw `f32` distance, BQ quantize, and BQ Hamming throughput.

### Notes

- `src/rng.rs` and `src/train.rs` are near-verbatim copies of the SplitMix64 PRNG and k-means used by the broader iqdb spine. The duplication is intentional for v0.1.0 because lifting them into shared utility crates (`iqdb-rand`, `iqdb-cluster`) is tracked separately; the duplication has no consumer-visible effect.

<<<<<<< HEAD
>>>>>>> 90cf5f804c31c4b137e8d23e9e00b7b4f56f10b2
=======
>>>>>>> a62852ef3ed25f477b97eef1f7cbec854963f422
[Unreleased]: https://github.com/jamesgober/iqdb-quantize/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jamesgober/iqdb-quantize/releases/tag/v0.1.0
