//! The shared [`Quantizer`] trait.
//!
//! Every quantization scheme in this crate — scalar (SQ8) and binary (BQ)
//! today, product quantization (PQ) eventually — implements this trait.
//! The shape mirrors the iqdb specification, with one deviation:
//! [`Quantizer::quantize`], [`Quantizer::dequantize`], and
//! [`Quantizer::distance`] are fallible (each returns
//! [`iqdb_types::Result`]) so that bad input becomes a typed
//! [`iqdb_types::IqdbError`] instead of a panic.

use iqdb_types::{DistanceMetric, Result};

/// A vector quantizer.
///
/// Implementations compress `f32` vectors into a compact [`Self::Quantized`]
/// representation and provide an asymmetric distance function that takes a
/// raw `f32` query against a stored code.
///
/// All methods returning [`Result`] surface failure as
/// [`iqdb_types::IqdbError`]. The library never panics on bad input.
///
/// # Calibration
///
/// Quantizers MUST be trained before any hot method is called. Calling
/// [`Quantizer::quantize`], [`Quantizer::dequantize`], or
/// [`Quantizer::distance`] before [`Quantizer::train`] returns
/// [`iqdb_types::IqdbError::InvalidConfig`].
///
/// # Examples
///
/// ```
/// use iqdb_quantize::{Quantizer, ScalarQuantizer};
/// use iqdb_types::DistanceMetric;
///
/// let mut sq = ScalarQuantizer::new();
/// sq.train(&[&[0.0_f32, 1.0][..], &[1.0_f32, 0.0][..]])
///     .expect("two non-empty, finite vectors of equal dim");
///
/// let code = sq.quantize(&[0.5_f32, 0.5]).expect("dim matches");
/// let d = sq
///     .distance(&[0.5_f32, 0.5], &code, DistanceMetric::Euclidean)
///     .expect("dim matches");
/// assert!(d.is_finite());
/// ```
pub trait Quantizer {
    /// The compact code produced by [`Quantizer::quantize`].
    type Quantized;

    /// Train the quantizer from a sample of representative vectors.
    ///
    /// The implementation derives whatever calibration the scheme needs —
    /// per-dimension `(min, max)` for SQ8, per-dimension means for BQ —
    /// from `vectors`.
    ///
    /// # Errors
    ///
    /// Returns [`iqdb_types::IqdbError::InvalidConfig`] if `vectors` is
    /// empty. Returns [`iqdb_types::IqdbError::InvalidVector`] if any
    /// training vector is empty or contains non-finite components. Returns
    /// [`iqdb_types::IqdbError::DimensionMismatch`] if the training
    /// vectors disagree on dimension.
    fn train(&mut self, vectors: &[&[f32]]) -> Result<()>;

    /// Encode `vector` as a compact code.
    ///
    /// # Errors
    ///
    /// Returns [`iqdb_types::IqdbError::InvalidConfig`] if the quantizer
    /// has not been trained. Returns [`iqdb_types::IqdbError::InvalidVector`]
    /// if `vector` is empty or contains non-finite components. Returns
    /// [`iqdb_types::IqdbError::DimensionMismatch`] if `vector` does not
    /// match the dimension the quantizer was trained on.
    fn quantize(&self, vector: &[f32]) -> Result<Self::Quantized>;

    /// Decode `quantized` back to an `f32` vector.
    ///
    /// The result is an approximation of the original input — quantization
    /// is lossy.
    ///
    /// # Errors
    ///
    /// Returns [`iqdb_types::IqdbError::InvalidConfig`] if the quantizer
    /// has not been trained, or [`iqdb_types::IqdbError::DimensionMismatch`]
    /// if `quantized` was produced by a quantizer with a different trained
    /// dimension.
    fn dequantize(&self, quantized: &Self::Quantized) -> Result<Vec<f32>>;

    /// Compute the asymmetric distance between a raw `f32` query and a
    /// stored code under `metric`.
    ///
    /// "Smaller is nearer", matching the convention in `iqdb-distance` and
    /// [`iqdb_types::Hit`](iqdb_types::Hit).
    ///
    /// # Errors
    ///
    /// Returns [`iqdb_types::IqdbError::InvalidConfig`] if the quantizer
    /// has not been trained. Returns [`iqdb_types::IqdbError::InvalidVector`]
    /// if `query` is empty or contains non-finite components. Returns
    /// [`iqdb_types::IqdbError::DimensionMismatch`] if `query` does not
    /// match the trained dimension. Implementations MAY return
    /// [`iqdb_types::IqdbError::InvalidMetric`] for metrics they do not
    /// support — for example, [`BinaryQuantizer`](crate::BinaryQuantizer)
    /// supports [`DistanceMetric::Hamming`] only.
    fn distance(
        &self,
        query: &[f32],
        quantized: &Self::Quantized,
        metric: DistanceMetric,
    ) -> Result<f32>;
}
