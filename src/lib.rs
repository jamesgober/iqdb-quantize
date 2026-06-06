//! # iqdb-quantize
//!
//! Vector quantization for the **iqdb** embedded vector-database spine. The
//! crate compresses `f32` embedding vectors into compact codes that preserve
//! similarity-search quality. It ships three schemes behind one trait:
//!
//! - [`ScalarQuantizer`] — scalar quantization (SQ8, 4× compression).
//!   Per-dimension affine calibration learned from a training sample; codes
//!   are `u8`. Asymmetric distance dequantizes the candidate to a temporary
//!   buffer and routes through [`iqdb_distance::compute`] for every metric.
//! - [`BinaryQuantizer`] — binary quantization (BQ, 32× compression). One
//!   bit per dimension thresholded against a trained per-dimension mean;
//!   codes are packed into [`u64`] words. Hamming distance is computed
//!   directly on the packed codes via XOR + popcount. **BQ supports
//!   [`DistanceMetric::Hamming`] only**; other metrics return
//!   [`IqdbError::InvalidMetric`].
//! - [`ProductQuantizer`] — product quantization (PQ, configurable
//!   compression — `M` bytes per code, e.g. `M = 16` shrinks a 768-dim
//!   `f32` vector from 3072 bytes to 16). Splits each vector into `M`
//!   subvectors and learns a `K`-centroid codebook per position via
//!   deterministic k-means (k-means++ seeding, Lloyd's iterations, seeded
//!   by [`ProductQuantizer::seed`]). Asymmetric distance computation (ADC)
//!   precomputes per-subvector distance tables and scores codes by table
//!   lookup + summation. **PQ supports [`DistanceMetric::Euclidean`],
//!   [`DistanceMetric::DotProduct`], and [`DistanceMetric::Manhattan`]**;
//!   [`DistanceMetric::Cosine`] (no global norm) and
//!   [`DistanceMetric::Hamming`] (wrong code space) return
//!   [`IqdbError::InvalidMetric`].
//!
//! Every method of the [`Quantizer`] trait is fallible and returns
//! [`iqdb_types::Result`]. The library never panics on bad input.
//!
//! ## How to use quantization correctly
//!
//! Quantization is lossy by design. Two rules:
//!
//! 1. **Train on representative data.** Per-dimension calibration is only
//!    as good as the sample it was learned from. Train on the embeddings
//!    you intend to index, not a synthetic placeholder.
//! 2. **Search quantized, rerank with full `f32`.** Quantized distance
//!    narrows the candidate set cheaply; the final ranking should use the
//!    original `f32` vectors. Skipping the rerank step is the most common
//!    cause of "quantization broke recall" reports.
//!
//! ## Example
//!
//! ```
//! use iqdb_quantize::{Quantizer, ScalarQuantizer};
//! use iqdb_types::DistanceMetric;
//!
//! let training = [
//!     vec![0.10_f32, 0.20, 0.30],
//!     vec![0.15, 0.18, 0.32],
//!     vec![0.12, 0.22, 0.28],
//! ];
//! let refs: Vec<&[f32]> = training.iter().map(Vec::as_slice).collect();
//!
//! let mut sq = ScalarQuantizer::new();
//! sq.train(&refs).expect("non-empty, consistent dims, finite values");
//!
//! let code = sq.quantize(&[0.11, 0.21, 0.29]).expect("dim matches training");
//! let d = sq
//!     .distance(&[0.10, 0.20, 0.30], &code, DistanceMetric::Cosine)
//!     .expect("dim matches");
//! assert!(d.is_finite());
//! ```
//!
//! ## Errors
//!
//! Every fallible call returns [`iqdb_types::Result`]. Empty or non-finite
//! inputs surface as [`IqdbError::InvalidVector`]; dimension drift as
//! [`IqdbError::DimensionMismatch`]; calling a hot method before
//! [`Quantizer::train`] returns [`IqdbError::InvalidConfig`]; a non-Hamming
//! metric against [`BinaryQuantizer`] or an unsupported metric
//! ([`DistanceMetric::Cosine`], [`DistanceMetric::Hamming`]) against
//! [`ProductQuantizer`] returns [`IqdbError::InvalidMetric`].
//!
//! [`DistanceMetric::Cosine`]: iqdb_types::DistanceMetric::Cosine
//! [`DistanceMetric::DotProduct`]: iqdb_types::DistanceMetric::DotProduct
//! [`DistanceMetric::Euclidean`]: iqdb_types::DistanceMetric::Euclidean
//! [`DistanceMetric::Hamming`]: iqdb_types::DistanceMetric::Hamming
//! [`DistanceMetric::Manhattan`]: iqdb_types::DistanceMetric::Manhattan
//! [`IqdbError`]: iqdb_types::IqdbError
//! [`IqdbError::InvalidConfig`]: iqdb_types::IqdbError::InvalidConfig
//! [`IqdbError::InvalidMetric`]: iqdb_types::IqdbError::InvalidMetric
//! [`IqdbError::InvalidVector`]: iqdb_types::IqdbError::InvalidVector
//! [`IqdbError::DimensionMismatch`]: iqdb_types::IqdbError::DimensionMismatch

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(warnings)]
#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(unused_must_use)]
#![deny(unused_results)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![deny(clippy::print_stdout)]
#![deny(clippy::print_stderr)]
#![deny(clippy::dbg_macro)]
#![deny(clippy::unreachable)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod binary;
mod code;
mod product;
mod rng;
mod scalar;
mod train;
mod traits;
mod validate;

pub use crate::binary::BinaryQuantizer;
pub use crate::code::{BqCode, PqCode, Sq8Code};
pub use crate::product::{PqAdcTables, ProductQuantizer};
pub use crate::scalar::ScalarQuantizer;
pub use crate::traits::Quantizer;

/// The version of this crate, taken from `Cargo.toml` at compile time.
///
/// Exposed so a consumer can report the exact `iqdb-quantize` build it links
/// against — useful in diagnostics and version-skew checks across the iqdb
/// crate family.
///
/// # Examples
///
/// ```
/// // Carries a `major.minor.patch` SemVer core.
/// let version = iqdb_quantize::VERSION;
/// assert_eq!(version.split('.').count(), 3);
/// assert!(version.split('.').all(|part| !part.is_empty()));
/// ```
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
