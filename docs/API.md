# iqdb-quantize &mdash; API Reference

> Complete reference for **every** public item in `iqdb-quantize` as of
> **v1.0.0**: what it is, its parameters and return shape, the traits it
> implements, and worked examples for each use case.
>
> **Status: stable (1.0).** The public API is committed under SemVer for the 1.x
> series — no breaking changes until 2.0. The frozen surface is recorded in
> `dev/ROADMAP.md`; only additive, non-breaking changes are made within 1.x.

## Table of Contents

- [Overview](#overview)
- [Crate constants](#crate-constants)
  - [`VERSION`](#version)
- [The `Quantizer` trait](#the-quantizer-trait)
  - [`train`](#quantizertrain)
  - [`quantize`](#quantizerquantize)
  - [`dequantize`](#quantizerdequantize)
  - [`distance`](#quantizerdistance)
- [The quantizers](#the-quantizers)
  - [`ScalarQuantizer` (SQ8)](#scalarquantizer-sq8)
  - [`BinaryQuantizer` (BQ)](#binaryquantizer-bq)
  - [`ProductQuantizer` (PQ)](#productquantizer-pq)
- [The code types](#the-code-types)
  - [`Sq8Code`](#sq8code)
  - [`BqCode`](#bqcode)
  - [`PqCode`](#pqcode)
- [Batch ADC](#batch-adc)
  - [`PqAdcTables`](#pqadctables)
- [Metric support matrix](#metric-support-matrix)
- [Errors](#errors)
- [Feature flags](#feature-flags)
- [Trait implementation matrix](#trait-implementation-matrix)

---

## Overview

`iqdb-quantize` compresses `f32` embedding vectors into compact codes that
preserve similarity-search quality. A million 768-dim vectors drop from ~3 GB to
as little as ~96 MB, trading a controlled amount of recall for memory. Three
schemes share one [`Quantizer`](#the-quantizer-trait) trait:

| Scheme | Type | Code | Compression | Metrics |
|---|---|---|---|---|
| Scalar (SQ8) | [`ScalarQuantizer`](#scalarquantizer-sq8) | one `u8` / dim | ~4× | every metric (asymmetric) |
| Product (PQ) | [`ProductQuantizer`](#productquantizer-pq) | `M` bytes | up to ~192× | Euclidean, DotProduct, Manhattan |
| Binary (BQ) | [`BinaryQuantizer`](#binaryquantizer-bq) | one bit / dim | ~32× | Hamming only |

Two rules make quantization behave:

1. **Train on representative data.** Per-dimension calibration is only as good as
   the sample it was learned from — train on the embeddings you intend to index.
2. **Search quantized, then rerank with full `f32`.** Quantized distance narrows
   the candidate set cheaply; the final ranking should use the original vectors.
   Skipping the rerank is the most common cause of "quantization broke recall".

**No panics.** Every fallible method returns `iqdb_types::Result`. Empty,
non-finite, dimension-mismatched, untrained, and unsupported-metric inputs all
surface as a typed [`IqdbError`](#errors).

```rust
use iqdb_quantize::{Quantizer, ScalarQuantizer};
use iqdb_types::DistanceMetric;

let training = [
    vec![0.10_f32, 0.20, 0.30],
    vec![0.15, 0.18, 0.32],
    vec![0.12, 0.22, 0.28],
];
let refs: Vec<&[f32]> = training.iter().map(Vec::as_slice).collect();

let mut sq = ScalarQuantizer::new();
sq.train(&refs).expect("non-empty, consistent dims, finite values");

let code = sq.quantize(&[0.11_f32, 0.21, 0.29]).expect("dim matches training");
let d = sq
    .distance(&[0.10_f32, 0.20, 0.30], &code, DistanceMetric::Cosine)
    .expect("dim matches");
assert!(d.is_finite());
```

---

## Crate constants

### `VERSION`

```rust
pub const VERSION: &str;
```

The crate's compile-time version (`CARGO_PKG_VERSION`), a `major.minor.patch`
SemVer core. Use it to report the exact `iqdb-quantize` build a binary links
against — useful in diagnostics and version-skew checks across the iQDB crate
family.

```rust
let v = iqdb_quantize::VERSION;
assert_eq!(v.split('.').count(), 3);
assert!(v.split('.').all(|part| !part.is_empty()));
```

---

## The `Quantizer` trait

```rust
pub trait Quantizer {
    type Quantized;

    fn train(&mut self, vectors: &[&[f32]]) -> Result<()>;
    fn quantize(&self, vector: &[f32]) -> Result<Self::Quantized>;
    fn dequantize(&self, quantized: &Self::Quantized) -> Result<Vec<f32>>;
    fn distance(
        &self,
        query: &[f32],
        quantized: &Self::Quantized,
        metric: DistanceMetric,
    ) -> Result<f32>;
}
```

The single contract every scheme implements. The associated type `Quantized` is
the scheme's code: [`Sq8Code`](#sq8code) for [`ScalarQuantizer`](#scalarquantizer-sq8),
[`BqCode`](#bqcode) for [`BinaryQuantizer`](#binaryquantizer-bq),
[`PqCode`](#pqcode) for [`ProductQuantizer`](#productquantizer-pq).

> **Calibration contract.** A quantizer **must** be trained before any of
> `quantize`, `dequantize`, or `distance` is called. Calling a hot method on an
> untrained quantizer returns [`IqdbError::InvalidConfig`](#errors) rather than
> panicking. A trained quantizer is immutable and `Send + Sync` — it owns its
> calibration by value and exposes no interior mutability, so it can be shared
> across threads.

### `Quantizer::train`

```rust
fn train(&mut self, vectors: &[&[f32]]) -> Result<()>;
```

Learn the scheme's calibration from a representative sample — per-dimension
`(min, max)` for SQ8, per-dimension means for BQ, per-subvector k-means
codebooks for PQ.

- **`vectors`** — the training sample, as a slice of `f32` slices. Must be
  non-empty, every vector non-empty and finite, and all the same length.
- **Returns** `Ok(())`, or:
  - [`Err(IqdbError::InvalidConfig)`](#errors) if `vectors` is empty (or, for PQ,
    the configured shape is invalid for the data — see
    [`with_config`](#productquantizer-pq)).
  - [`Err(IqdbError::InvalidVector)`](#errors) if any training vector is empty or
    has a `NaN`/`±∞` component.
  - [`Err(IqdbError::DimensionMismatch)`](#errors) if the training vectors
    disagree on length.

```rust
use iqdb_quantize::{Quantizer, ScalarQuantizer};

let mut sq = ScalarQuantizer::new();
sq.train(&[&[0.0_f32, 1.0, 2.0][..], &[1.0_f32, 0.0, 1.0][..]])
    .expect("two non-empty, finite vectors of equal dim");
assert_eq!(sq.dim(), Some(3));
```

### `Quantizer::quantize`

```rust
fn quantize(&self, vector: &[f32]) -> Result<Self::Quantized>;
```

Encode `vector` into the scheme's compact code.

- **`vector`** — the vector to compress; must be non-empty, finite, and match the
  trained dimension.
- **Returns** `Ok(Self::Quantized)`, or [`InvalidConfig`](#errors) if untrained,
  [`InvalidVector`](#errors) if empty/non-finite, [`DimensionMismatch`](#errors)
  if the length differs from training.

```rust
use iqdb_quantize::{Quantizer, ScalarQuantizer};

let mut sq = ScalarQuantizer::new();
sq.train(&[&[0.0_f32, 10.0][..], &[10.0_f32, 0.0][..]]).expect("ok");
let code = sq.quantize(&[5.0_f32, 5.0]).expect("dim matches");
assert_eq!(code.len(), 2);
```

### `Quantizer::dequantize`

```rust
fn dequantize(&self, quantized: &Self::Quantized) -> Result<Vec<f32>>;
```

Decode a code back to an `f32` vector. The result is an approximation —
quantization is lossy.

- **`quantized`** — a code produced by this scheme.
- **Returns** `Ok(Vec<f32>)` of the trained dimension, or [`InvalidConfig`](#errors)
  if untrained / [`DimensionMismatch`](#errors) if the code was produced under a
  different trained dimension.

```rust
use iqdb_quantize::{Quantizer, ScalarQuantizer};

let mut sq = ScalarQuantizer::new();
sq.train(&[&[0.0_f32, 10.0][..], &[10.0_f32, 0.0][..]]).expect("ok");
let code = sq.quantize(&[5.0_f32, 5.0]).expect("ok");
let approx = sq.dequantize(&code).expect("ok");
assert_eq!(approx.len(), 2);
assert!((approx[0] - 5.0).abs() < 0.1); // close, not exact — lossy
```

### `Quantizer::distance`

```rust
fn distance(
    &self,
    query: &[f32],
    quantized: &Self::Quantized,
    metric: DistanceMetric,
) -> Result<f32>;
```

Compute the **asymmetric** distance between a raw `f32` query and a stored code:
the query stays full precision, only the candidate is compressed. "Smaller is
nearer", matching the rest of the iQDB spine.

- **`query`** — the full-precision query vector (non-empty, finite, trained dim).
- **`quantized`** — the stored code to score against.
- **`metric`** — which [`DistanceMetric`] to use. Support is scheme-specific —
  see the [metric support matrix](#metric-support-matrix). An unsupported metric
  returns [`InvalidMetric`](#errors).
- **Returns** `Ok(f32)`, or the typed errors above.

```rust
use iqdb_quantize::{Quantizer, ScalarQuantizer};
use iqdb_types::DistanceMetric;

let mut sq = ScalarQuantizer::new();
sq.train(&[&[0.0_f32, 1.0][..], &[1.0_f32, 0.0][..]]).expect("ok");
let code = sq.quantize(&[0.5_f32, 0.5]).expect("ok");
let d = sq
    .distance(&[0.5_f32, 0.5], &code, DistanceMetric::Euclidean)
    .expect("supported");
assert!(d.is_finite());
```

[`DistanceMetric`]: iqdb_types::DistanceMetric

---

## The quantizers

### `ScalarQuantizer` (SQ8)

```rust
pub struct ScalarQuantizer { /* … */ }

impl ScalarQuantizer {
    pub fn new() -> Self;
    pub fn dim(&self) -> Option<usize>;
}
impl Default for ScalarQuantizer { /* = new() */ }
impl Quantizer for ScalarQuantizer { type Quantized = Sq8Code; }
```

Scalar quantization: one `u8` per dimension, ~4× compression. The calibration is
a per-dimension affine map — each dimension stores its trained `min` and a
`scale = (max - min) / 255`. Encoding clamps the input into `[min, max]`, scales
onto `[0, 255]`, and rounds; decoding reverses it. A zero-range dimension
(`max == min`) collapses to a `scale = 0` lane: every code byte there is `0` and
`dequantize` returns `min`, so there is no division by zero. Distance is
**asymmetric** and supports **every** [`DistanceMetric`] — the candidate is
dequantized to a temporary buffer and routed through
[`iqdb_distance::compute`](iqdb_distance::compute).

- **`new()`** — build an untrained quantizer. `#[must_use]`. Equivalent to
  [`Default`].
- **`dim()`** — the trained dimension, or `None` before training.

```rust
use iqdb_quantize::{Quantizer, ScalarQuantizer};
use iqdb_types::DistanceMetric;

let mut sq = ScalarQuantizer::new();
assert_eq!(sq.dim(), None);
sq.train(&[&[0.0_f32, 1.0, 2.0][..], &[1.0_f32, 0.0, 1.0][..]]).expect("ok");
assert_eq!(sq.dim(), Some(3));

let code = sq.quantize(&[0.5_f32, 0.5, 1.5]).expect("dim matches");
let d = sq.distance(&[0.5_f32, 0.5, 1.5], &code, DistanceMetric::Cosine).expect("ok");
assert!(d.is_finite());
```

### `BinaryQuantizer` (BQ)

```rust
pub struct BinaryQuantizer { /* … */ }

impl BinaryQuantizer {
    pub fn new() -> Self;
    pub fn dim(&self) -> Option<usize>;
}
impl Default for BinaryQuantizer { /* = new() */ }
impl Quantizer for BinaryQuantizer { type Quantized = BqCode; }
```

Binary quantization: one bit per dimension, ~32× compression. Bit `i` is `1` when
`vector[i] >= mean[i]` (the trained per-dimension mean), `0` otherwise; bits pack
into `u64` words with the trailing word's unused high bits zeroed so they cannot
contribute to Hamming. The query path binarizes against the **same** trained
means, so query and code bits share one space.

BQ supports [`DistanceMetric::Hamming`] **only** — every other metric returns
[`InvalidMetric`](#errors). A one-bit code carries no magnitude, so a cosine or
Euclidean comparison over ±1 codes would be a roundabout Hamming in misleading
units; the contract rejects that rather than mislead (matching the Faiss
`IndexBinary` convention).

- **`new()`** — build an untrained quantizer. `#[must_use]`. Equivalent to
  [`Default`].
- **`dim()`** — the trained dimension, or `None` before training.

```rust
use iqdb_quantize::{BinaryQuantizer, Quantizer};
use iqdb_types::DistanceMetric;

let mut bq = BinaryQuantizer::new();
bq.train(&[&[0.0_f32, 1.0, 2.0][..], &[2.0_f32, 1.0, 0.0][..]]).expect("ok");

let code = bq.quantize(&[0.5_f32, 1.5, 2.5]).expect("dim matches");
assert_eq!(code.dim(), 3);
let d = bq.distance(&[0.5_f32, 1.5, 2.5], &code, DistanceMetric::Hamming).expect("ok");
assert_eq!(d, 0.0); // self-distance is zero

// Any non-Hamming metric is rejected.
use iqdb_types::IqdbError;
let err = bq.distance(&[0.5_f32, 1.5, 2.5], &code, DistanceMetric::Cosine).unwrap_err();
assert_eq!(err, IqdbError::InvalidMetric);
```

[`DistanceMetric::Hamming`]: iqdb_types::DistanceMetric::Hamming

### `ProductQuantizer` (PQ)

```rust
pub struct ProductQuantizer { /* … */ }

impl ProductQuantizer {
    pub fn new() -> Self;                                          // M = 8, K = 256, seed = 0
    pub fn with_config(n_subvectors: usize, n_centroids: usize, seed: u64) -> Self;
    pub fn dim(&self) -> Option<usize>;
    pub fn n_subvectors(&self) -> usize;                          // M
    pub fn n_centroids(&self) -> usize;                           // K
    pub fn seed(&self) -> u64;
    pub fn build_query_tables(&self, query: &[f32], metric: DistanceMetric) -> Result<PqAdcTables>;
}
impl Default for ProductQuantizer { /* = new() */ }
impl Quantizer for ProductQuantizer { type Quantized = PqCode; }
```

Product quantization: split each vector into `M = n_subvectors` equal-length
subvectors, learn a `K = n_centroids`-centroid codebook per subvector via
k-means (k-means++ seeding, Lloyd's iterations), and store one `u8` centroid
index per subvector — `M` bytes total. At `M = 16, K = 256` a 768-dim vector
shrinks from 3072 bytes to 16 (192×). Distance uses **asymmetric distance
computation (ADC)**: the query stays `f32`, a per-subvector query-to-centroid
table is precomputed, and a stored code is scored by `M` lookups plus a sum.

PQ supports [`DistanceMetric::Euclidean`], [`DistanceMetric::DotProduct`], and
[`DistanceMetric::Manhattan`] — each decomposes into a per-subvector sum.
[`DistanceMetric::Cosine`] (no global norm recoverable per subvector;
L2-normalize and use `DotProduct`) and [`DistanceMetric::Hamming`] (wrong code
space) return [`InvalidMetric`](#errors).

- **`new()`** — standard `M = 8, K = 256, seed = 0`. `M = 8` divides the common
  embedding dims (128, 256, 384, 512, 768, 1024). `#[must_use]`.
- **`with_config(n_subvectors, n_centroids, seed)`** — pick `M`, `K` (`K ≤ 256`,
  codes are `u8`), and the training seed. The constructor is infallible; invalid
  combinations (`n_centroids == 0` or `> 256`, training dim not divisible by `M`)
  surface as [`InvalidConfig`](#errors) from [`train`](#quantizertrain).
  `#[must_use]`.
- **`dim()` / `n_subvectors()` / `n_centroids()` / `seed()`** — report the
  trained dimension (or `None`) and the configured `M`, `K`, and seed.
- **`build_query_tables(query, metric)`** — see [Batch ADC](#batch-adc).

> **Determinism.** The same `seed` + the same training data produce
> byte-identical codebooks and codes on every supported platform. ADC is exact:
> [`distance`](#quantizerdistance) equals [`dequantize`](#quantizerdequantize) +
> [`iqdb_distance::compute`](iqdb_distance::compute) within floating-point
> reduction tolerance — both are property-tested.

```rust
use iqdb_quantize::{ProductQuantizer, Quantizer};
use iqdb_types::DistanceMetric;

let mut pq = ProductQuantizer::with_config(2, 4, 7); // M = 2, K = 4, seed = 7
let training: Vec<Vec<f32>> = (0..16)
    .map(|i| { let f = i as f32; vec![f, f + 1.0, f + 2.0, f + 3.0] })
    .collect();
let refs: Vec<&[f32]> = training.iter().map(Vec::as_slice).collect();
pq.train(&refs).expect("dim divisible by M, K <= 256");

let code = pq.quantize(&[1.0_f32, 2.0, 3.0, 4.0]).expect("quantize");
assert_eq!(code.n_subvectors(), 2);
let d = pq.distance(&[1.0_f32, 2.0, 3.0, 4.0], &code, DistanceMetric::Euclidean).expect("ok");
assert!(d.is_finite());
```

[`DistanceMetric::Euclidean`]: iqdb_types::DistanceMetric::Euclidean
[`DistanceMetric::DotProduct`]: iqdb_types::DistanceMetric::DotProduct
[`DistanceMetric::Manhattan`]: iqdb_types::DistanceMetric::Manhattan
[`DistanceMetric::Cosine`]: iqdb_types::DistanceMetric::Cosine

---

## The code types

All three codes are owned, immutable newtypes — `Debug`, `Clone`, `PartialEq`,
`Eq`, no public mutators. Each is produced **only** by its owning quantizer, so a
code's contents always match the calibrated quantizer that made it; a caller
cannot fabricate one.

### `Sq8Code`

```rust
pub struct Sq8Code { /* … */ }

impl Sq8Code {
    pub fn len(&self) -> usize;       // one byte per dimension
    pub fn is_empty(&self) -> bool;
    pub fn as_bytes(&self) -> &[u8];
}
```

A scalar-quantized code: one `u8` per dimension. Byte `i` is the linear `u8`
encoding of component `i` under that dimension's affine calibration — not useful
on its own; decode with [`dequantize`](#quantizerdequantize) or compare via
[`distance`](#quantizerdistance).

```rust
use iqdb_quantize::{Quantizer, ScalarQuantizer};

let mut sq = ScalarQuantizer::new();
sq.train(&[&[0.0_f32, 1.0, 2.0][..]]).expect("ok");
let code = sq.quantize(&[0.5_f32, 0.5, 0.5]).expect("ok");
assert_eq!(code.len(), 3);
assert!(!code.is_empty());
assert_eq!(code.as_bytes().len(), 3);
```

### `BqCode`

```rust
pub struct BqCode { /* … */ }

impl BqCode {
    pub fn dim(&self) -> usize;       // original vector dimension
    pub fn is_empty(&self) -> bool;
    pub fn as_words(&self) -> &[u64]; // packed bits
}
```

A binary-quantized code: one bit per dimension, packed into `u64` words. `dim` is
the number of meaningful bits; the trailing word's unused high bits are always
zero. The word count is `dim.div_ceil(64)`.

```rust
use iqdb_quantize::{BinaryQuantizer, Quantizer};

let mut bq = BinaryQuantizer::new();
bq.train(&[&[0.0_f32; 65][..], &[1.0_f32; 65][..]]).expect("ok");
let code = bq.quantize(&[0.5_f32; 65]).expect("ok");
assert_eq!(code.dim(), 65);
assert_eq!(code.as_words().len(), 2); // 65 bits → two u64 words
```

### `PqCode`

```rust
pub struct PqCode { /* … */ }

impl PqCode {
    pub fn dim(&self) -> usize;          // original vector dimension
    pub fn n_subvectors(&self) -> usize; // M
    pub fn len(&self) -> usize;          // == n_subvectors
    pub fn is_empty(&self) -> bool;
    pub fn as_bytes(&self) -> &[u8];     // one centroid index per subvector
}
```

A product-quantized code: one `u8` centroid index per subvector. Byte `m` is the
index (in `0..n_centroids`, `≤ 256`) of the centroid in codebook `m` that best
approximates the `m`-th subvector. `len()` equals `n_subvectors()`.

```rust
use iqdb_quantize::{ProductQuantizer, Quantizer};

let mut pq = ProductQuantizer::with_config(2, 4, 42);
let training: Vec<Vec<f32>> = (0..8)
    .map(|i| vec![i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32])
    .collect();
let refs: Vec<&[f32]> = training.iter().map(Vec::as_slice).collect();
pq.train(&refs).expect("ok");

let code = pq.quantize(&[1.0_f32, 2.0, 3.0, 4.0]).expect("ok");
assert_eq!(code.n_subvectors(), 2);
assert_eq!(code.dim(), 4);
assert_eq!(code.as_bytes().len(), 2);
```

---

## Batch ADC

### `PqAdcTables`

```rust
pub struct PqAdcTables { /* … */ } // Debug, Clone

impl PqAdcTables {
    pub fn distance(&self, code: &PqCode) -> Result<f32>;
    pub fn metric(&self) -> DistanceMetric;
    pub fn n_subvectors(&self) -> usize;
    pub fn n_centroids(&self) -> usize;
    pub fn dim(&self) -> usize;
}
```

Per-`(query, metric)` precomputed ADC lookup tables, built once with
[`ProductQuantizer::build_query_tables`](#productquantizer-pq) and reused to score
many [`PqCode`](#pqcode)s. Row `m` holds the distances from query subvector `q_m`
to each of the `K` centroids of codebook `m`. For
[`DistanceMetric::Euclidean`](iqdb_types::DistanceMetric::Euclidean) the rows hold
**squared L2** values (so they sum decomposably) and `distance` takes `sqrt` of
the total exactly once; `DotProduct` and `Manhattan` sum directly.

This is the primitive `iqdb-ivf`'s IVF-PQ intra-cluster scan is built on: build
the table once per query, then score every code in every probed cluster against
it. [`ProductQuantizer::distance`](#quantizerdistance) is itself a thin wrapper
over `build_query_tables` + `distance`, so the single-shot result is
byte-for-byte identical to the batch path.

- **`distance(code)`** — score one code against the prepared tables. Returns
  [`DimensionMismatch`](#errors) if `code` came from a quantizer with a different
  `M` or trained `dim`.
- **`metric()` / `n_subvectors()` / `n_centroids()` / `dim()`** — the metric and
  geometry the tables were built for.

```rust
use iqdb_quantize::{ProductQuantizer, Quantizer};
use iqdb_types::DistanceMetric;

let mut pq = ProductQuantizer::with_config(2, 4, 7);
let training: Vec<Vec<f32>> = (0..16)
    .map(|i| { let f = i as f32; vec![f, f + 1.0, f + 2.0, f + 3.0] })
    .collect();
let refs: Vec<&[f32]> = training.iter().map(Vec::as_slice).collect();
pq.train(&refs).expect("ok");

let code_a = pq.quantize(&[1.0_f32, 2.0, 3.0, 4.0]).expect("ok");
let code_b = pq.quantize(&[5.0_f32, 6.0, 7.0, 8.0]).expect("ok");

// Build the table ONCE, then score many codes.
let query = [1.0_f32, 2.0, 3.0, 4.0];
let tables = pq.build_query_tables(&query, DistanceMetric::Euclidean).expect("supported");
let d_a = tables.distance(&code_a).expect("matching shape");
let d_b = tables.distance(&code_b).expect("matching shape");
assert!(d_a.is_finite() && d_b.is_finite());

// Identical to the single-shot path.
let single = pq.distance(&query, &code_a, DistanceMetric::Euclidean).expect("ok");
assert_eq!(d_a.to_bits(), single.to_bits());
```

---

## Metric support matrix

`distance` and `build_query_tables` accept the metric at runtime; what each
scheme supports differs. An unsupported metric returns
[`IqdbError::InvalidMetric`](#errors) — never a panic — which keeps callers
working as `iqdb-types` adds `#[non_exhaustive]` `DistanceMetric` variants.

| Metric | `ScalarQuantizer` | `ProductQuantizer` | `BinaryQuantizer` |
|---|:---:|:---:|:---:|
| `Euclidean` | ✅ | ✅ | ❌ |
| `DotProduct` | ✅ | ✅ | ❌ |
| `Manhattan` | ✅ | ✅ | ❌ |
| `Cosine` | ✅ | ❌¹ | ❌ |
| `Hamming` | ✅² | ❌ | ✅ |

<sub>¹ PQ needs a global norm it cannot recover per subvector — L2-normalize and
use `DotProduct`. ² SQ8 routes through `iqdb-distance`, which defines Hamming on
the dequantized `f32` components.</sub>

---

## Errors

`iqdb-quantize` returns the shared [`iqdb_types::IqdbError`] / `Result`
vocabulary — it adds no error type of its own. The variants it produces:

| Variant | When |
|---|---|
| `InvalidConfig { reason }` | A hot method (`quantize` / `dequantize` / `distance` / `build_query_tables`) called before `train`; an empty training set; or a PQ shape invalid for the data (`n_centroids` 0 or > 256, dim not divisible by `M`). |
| `InvalidVector` | An input (training vector, query, or candidate) is empty or has a `NaN`/`±∞` component. |
| `DimensionMismatch { expected, found }` | Training vectors disagree on length, or a query / code does not match the trained dimension. |
| `InvalidMetric` | A metric the scheme does not support — see the [matrix](#metric-support-matrix) — including unimplemented `#[non_exhaustive]` `DistanceMetric` variants. |

`IqdbError` is `Copy` and `#[non_exhaustive]`; match it with a wildcard arm. See
the `iqdb-types` API reference for `Display`, `kind()`, and `caption()`.

```rust
use iqdb_quantize::{Quantizer, ScalarQuantizer};
use iqdb_types::{DistanceMetric, IqdbError};

let sq = ScalarQuantizer::new(); // untrained

// Quantizing before training is a typed error, not a panic.
assert!(matches!(sq.quantize(&[1.0, 2.0]), Err(IqdbError::InvalidConfig { .. })));

let mut trained = ScalarQuantizer::new();
trained.train(&[&[0.0_f32, 1.0][..]]).expect("ok");
let code = trained.quantize(&[0.5_f32, 0.5]).expect("ok");

// Wrong query dimension.
let err = trained.distance(&[0.5_f32, 0.5, 0.5], &code, DistanceMetric::Euclidean).unwrap_err();
assert_eq!(err, IqdbError::DimensionMismatch { expected: 2, found: 3 });
```

[`iqdb_types::IqdbError`]: iqdb_types::IqdbError

---

## Feature flags

The crate has **no optional features** — `default = []`. It is `std`-only and
always pulls its four dependencies: `iqdb-types` (the shared `DistanceMetric` /
`IqdbError` / `Result` vocabulary), `iqdb-distance` (the f32 distance kernels SQ8
and PQ delegate to), `error-forge` (the `ForgeError` trait behind `IqdbError`'s
`kind()` / `caption()`), and `tracing` (instrumentation at the training
boundary). SIMD acceleration (AVX2 / NEON) lands transparently through
`iqdb-distance`, so there is no SIMD feature to toggle here.

---

## Trait implementation matrix

| Type | `Debug` | `Clone` | `Default` | `PartialEq` | `Eq` | `Quantizer` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `ScalarQuantizer` | ✅ | ✅ | ✅ | ✅ | — | ✅ (`= Sq8Code`) |
| `BinaryQuantizer` | ✅ | ✅ | ✅ | ✅ | — | ✅ (`= BqCode`) |
| `ProductQuantizer` | ✅ | ✅ | ✅ | ✅ | — | ✅ (`= PqCode`) |
| `Sq8Code` | ✅ | ✅ | — | ✅ | ✅ | — |
| `BqCode` | ✅ | ✅ | — | ✅ | ✅ | — |
| `PqCode` | ✅ | ✅ | — | ✅ | ✅ | — |
| `PqAdcTables` | ✅ | ✅ | — | — | — | — |

The quantizers hold `f32` calibration, so they are `PartialEq` but not `Eq`. The
code types hold only integer storage, so they are fully `Eq`.

---

<sub>Copyright &copy; 2026 <strong>James Gober</strong>.</sub>
