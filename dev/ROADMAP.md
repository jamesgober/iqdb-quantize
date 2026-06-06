# iqdb-quantize -- Roadmap

> Path from scaffold to a stable 1.0. Hard parts are front-loaded; each phase has hard exit criteria.
>
> **Anti-deferral rule:** no listed hard task moves to a later phase unless this file records the move and the reason.

---

## v0.1.0 -- Scaffold (DONE)

Compiles, CI green, structure correct, no domain logic.

- [x] Manifest, README, CHANGELOG, REPS, license, CI, lints in place.
- [x] API surface sketched in `docs/API.md`.

---

## v0.2.0 -- scalar quantization (SQ8) + the `Quantizer` trait (THE HARD PART, NOT DEFERRED) (DONE)

Exit criteria:
- [x] Every public item has rustdoc + a runnable example.
- [x] Core invariants property-tested.

`ScalarQuantizer` (SQ8, 4× compression) shipped behind the `Quantizer` trait
every scheme implements: per-dimension affine calibration with a zero-range
guard, asymmetric distance that keeps the query in `f32` and routes the
dequantized candidate through `iqdb_distance::compute` for every metric, and the
immutable `Sq8Code`. Every fallible path returns a typed `iqdb_types::IqdbError`;
round-trip bounds, distance finiteness, and metric-aware non-negativity are
property-tested, with full edge-case coverage.

---

## v0.3.0 -- product quantization (k-means codebooks) (DONE)

Exit criteria:
- [x] New surface tested and benchmarked where it is a hot path.

`ProductQuantizer` (PQ, `M` bytes per code) with deterministic hand-rolled
k-means (k-means++ seeding, Lloyd's iterations, seeded by `seed`), the immutable
`PqCode`, and the `PqAdcTables` + `build_query_tables` batch-ADC primitive IVF-PQ
consumes. `ProductQuantizer::distance` is a thin wrapper over the batch path, so
single-shot and batch results are byte-identical (proptested). Determinism
(same seed + data ⇒ byte-identical codes) is contractually guaranteed and
covered by `tests/determinism.rs`; the ADC-equals-dequantize-then-compute
invariant is property-tested across every supported metric.

---

## v0.4.0 -- binary quantization + asymmetric distance + feature freeze (DONE)

Exit criteria:
- [x] No `todo!`/`unimplemented!`. Feature freeze declared.

`BinaryQuantizer` (BQ, 32× compression) — one bit per dimension thresholded
against a trained per-dimension mean, packed into `u64` words with padding bits
zeroed — and the immutable `BqCode`. BQ supports `DistanceMetric::Hamming` only;
the packed-word Hamming is property-tested against a naive popcount reference.

**Feature freeze — the public surface is now complete and frozen for 1.x.**
Additive, non-breaking changes remain allowed; anything else waits for 2.0. The
frozen surface:

- Trait: `Quantizer` (`Quantized`, `train`, `quantize`, `dequantize`, `distance`).
- Quantizers: `ScalarQuantizer`, `BinaryQuantizer`, `ProductQuantizer`.
- Codes: `Sq8Code`, `BqCode`, `PqCode`.
- Batch ADC: `PqAdcTables`, `ProductQuantizer::build_query_tables`.
- Constant: `VERSION`.

PQ `Cosine` is intentionally unsupported (no global norm recoverable per
subvector; the documented path is to L2-normalize and use `DotProduct`) and is
**not** a deferred feature — recorded here per the anti-deferral rule.

---

## v0.5.0 -- training-stability + recall validation + API freeze (DONE)

Exit criteria:
- [x] Public API frozen (recorded here). `cargo audit` + `cargo deny` clean.

Recall is measured end to end against full-`f32` baselines on Gaussian-cluster
synthetic corpora (`tests/recall.rs`): SQ8 top-10 overlap ≥ 0.9, BQ top-10
cluster purity ≥ 0.7, PQ top-_k_ overlap against the Euclidean baseline —
thresholds taken from measured values with margin. Training boundaries are
instrumented with `tracing` (error events carry `error-forge`'s `kind()` /
`caption()`), verified by `tests/tracing.rs`, and a criterion bench harness
tracks SQ8/BQ hot paths.

**API freeze.** The stable public surface recorded under v0.4.0 is locked for
the 1.x series. Only additive, non-breaking changes land before 2.0.
`cargo audit` and `cargo deny check` are clean.

---

## v0.6.0 -> v0.9.x -- Alpha / Beta -> RC (folded into 1.0.0)

- 0.6.x-0.7.x: integrate against real consumers; MINOR-compatible additions only.
- 0.8.x (beta): bug fixes; broader testing; final benchmarks.
- 0.9.x (rc): critical fixes + doc polish.

The RC track's *intent* — prove the surface serves a real consumer, settle final
benchmarks, polish docs — was met without separate tags. Under the spine-first
ordering the real consumer `iqdb-ivf` (IVF-PQ scores in-cluster codes through
`PqAdcTables`) is not yet published against this surface, so the soak was carried
out by `tests/consumer_simulation.rs`: a mini IVF-PQ built at the **exact** shape
IVF-PQ uses — coarse clusters of `PqCode`s scanned through one `PqAdcTables` per
query — asserting batch-ADC-equals-single-shot, cluster purity, shortlist+rerank
recall, and safe boundary errors. Final benchmarks and the five runnable examples
shipped with 1.0.0. This mirrors how `iqdb-types` and `iqdb-distance` reached 1.0
on a satisfied checklist rather than a calendar.

---

## v1.0.0 -- Stable (DONE)

- [x] Definition of Done (DIRECTIVES section 7) satisfied.
- [x] Public API frozen until 2.0.
- [x] Release note written. (Publish to crates.io + tag push: owner action.)

The public surface committed under SemVer for the 1.x series (additive,
non-breaking changes only before 2.0):

- Trait: `Quantizer` (`Quantized`, `train`, `quantize`, `dequantize`, `distance`).
- Quantizers: `ScalarQuantizer`, `BinaryQuantizer`, `ProductQuantizer`
  (with `new` / `with_config` / `dim` / `n_subvectors` / `n_centroids` / `seed`
  / `build_query_tables`).
- Codes: `Sq8Code`, `BqCode`, `PqCode`.
- Batch ADC: `PqAdcTables` (`distance`, `metric`, `n_subvectors`, `n_centroids`,
  `dim`).
- Constant: `VERSION`.

**On the `loom` Definition-of-Done item (§7.6).** The quantizers own their
calibration by value and expose no interior mutability; a trained quantizer is
immutable and only ever read concurrently. There is no lock-free data structure
or multi-step concurrent protocol to model, so a `loom` test would model nothing
(KISS/YAGNI). Recorded as a deliberate, settled decision per the anti-deferral
rule.

---

## Out of scope for 1.0

- Index structures -- consumed by indexes, does not implement one.
- Distributed/quantized-sharding concerns.
