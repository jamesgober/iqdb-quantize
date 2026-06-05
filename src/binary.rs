//! [`BinaryQuantizer`] — binary quantization (BQ, 32× compression).
//!
//! Codes are one bit per dimension, packed into [`u64`] words. The threshold
//! used at encode time is the per-dimension mean learned from the training
//! sample: bit `i` is `1` when `vector[i] >= mean[i]`, `0` otherwise. When
//! the dimension is not a multiple of 64 the trailing word has unused high
//! bits; those are zeroed at encode time so they cannot contribute to
//! Hamming distance.
//!
//! Distance is supported under [`DistanceMetric::Hamming`] only — any other
//! metric returns [`IqdbError::InvalidMetric`]. BQ discards magnitude
//! entirely, so a cosine or Euclidean distance over ±1 codes would be a
//! roundabout Hamming dressed in misleading units. Restricting the contract
//! prevents that silent misuse and matches the public Faiss `IndexBinary`
//! convention. The query path inside [`BinaryQuantizer::distance`]
//! binarizes the query against the **same trained per-dimension means**
//! used during [`BinaryQuantizer::quantize`], so the query bits live in
//! the same space as the stored code bits.

use error_forge::ForgeError;
use iqdb_types::{DistanceMetric, IqdbError, Result};

use crate::code::BqCode;
use crate::traits::Quantizer;
use crate::validate::{dim_eq, finite_non_empty, training_set};

const BITS_PER_WORD: usize = u64::BITS as usize;

/// Calibration learned during [`BinaryQuantizer::train`].
#[derive(Debug, Clone, PartialEq)]
struct BqCalibration {
    /// Per-dimension mean from the training sample.
    means: Vec<f32>,
}

/// Binary quantizer (BQ): one bit per dimension, 32× compression.
///
/// Build one with [`BinaryQuantizer::new`] (or [`Default`]), train it once
/// with a representative sample, then quantize and compare. BQ supports
/// [`DistanceMetric::Hamming`] only; other metrics return
/// [`IqdbError::InvalidMetric`].
///
/// # Examples
///
/// ```
/// use iqdb_quantize::{BinaryQuantizer, Quantizer};
/// use iqdb_types::DistanceMetric;
///
/// let mut bq = BinaryQuantizer::new();
/// bq.train(&[&[0.0_f32, 1.0, 2.0][..], &[2.0_f32, 1.0, 0.0][..]])
///     .expect("two non-empty, finite vectors of equal dim");
///
/// let code = bq.quantize(&[0.5_f32, 1.5, 2.5]).expect("dim matches");
/// let d = bq
///     .distance(&[0.5_f32, 1.5, 2.5], &code, DistanceMetric::Hamming)
///     .expect("dim matches");
/// // Self-distance is zero.
/// assert_eq!(d, 0.0);
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BinaryQuantizer {
    calibration: Option<BqCalibration>,
}

impl BinaryQuantizer {
    /// Build an untrained binary quantizer.
    ///
    /// Every hot method returns [`IqdbError::InvalidConfig`] until
    /// [`Quantizer::train`] succeeds.
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_quantize::BinaryQuantizer;
    /// let _bq = BinaryQuantizer::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self { calibration: None }
    }

    /// The trained dimension, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_quantize::{BinaryQuantizer, Quantizer};
    ///
    /// let mut bq = BinaryQuantizer::new();
    /// assert_eq!(bq.dim(), None);
    /// bq.train(&[&[0.0_f32, 1.0][..]]).expect("ok");
    /// assert_eq!(bq.dim(), Some(2));
    /// ```
    #[must_use]
    pub fn dim(&self) -> Option<usize> {
        self.calibration.as_ref().map(|c| c.means.len())
    }

    fn calibration(&self) -> Result<&BqCalibration> {
        self.calibration.as_ref().ok_or(IqdbError::InvalidConfig {
            reason: "BinaryQuantizer has not been trained",
        })
    }
}

impl Quantizer for BinaryQuantizer {
    type Quantized = BqCode;

    #[tracing::instrument(
        level = "info",
        skip_all,
        fields(quantizer = "bq", training_size = vectors.len()),
    )]
    fn train(&mut self, vectors: &[&[f32]]) -> Result<()> {
        let dim = training_set(vectors).inspect_err(|err: &IqdbError| {
            tracing::error!(
                error.kind = err.kind(),
                error.reason = err.caption(),
                "binary quantizer training failed",
            );
        })?;
        let mut sums = vec![0.0_f64; dim];
        for v in vectors {
            for (i, &x) in v.iter().enumerate() {
                sums[i] += f64::from(x);
            }
        }
        let n = vectors.len() as f64;
        let means: Vec<f32> = sums.iter().map(|s| (s / n) as f32).collect();
        // Defensive: a finite f64 average MUST cast to a finite f32 given
        // finite inputs (training_set rejected non-finite). Belt-and-braces
        // guard avoids storing a NaN threshold if a future change weakens
        // that invariant.
        if means.iter().any(|m| !m.is_finite()) {
            let err = IqdbError::InvalidVector;
            tracing::error!(
                error.kind = err.kind(),
                error.reason = err.caption(),
                "binary quantizer training failed: non-finite mean",
            );
            return Err(err);
        }
        self.calibration = Some(BqCalibration { means });
        Ok(())
    }

    fn quantize(&self, vector: &[f32]) -> Result<Self::Quantized> {
        let cal = self.calibration()?;
        finite_non_empty(vector)?;
        dim_eq(cal.means.len(), vector.len())?;
        Ok(BqCode {
            words: pack_bits(vector, &cal.means),
            dim: vector.len(),
        })
    }

    fn dequantize(&self, quantized: &Self::Quantized) -> Result<Vec<f32>> {
        let cal = self.calibration()?;
        dim_eq(cal.means.len(), quantized.dim)?;
        let mut out = Vec::with_capacity(quantized.dim);
        for i in 0..quantized.dim {
            let word = quantized.words[i / BITS_PER_WORD];
            let bit = (word >> (i % BITS_PER_WORD)) & 1;
            out.push(if bit == 1 { 1.0_f32 } else { -1.0_f32 });
        }
        Ok(out)
    }

    fn distance(
        &self,
        query: &[f32],
        quantized: &Self::Quantized,
        metric: DistanceMetric,
    ) -> Result<f32> {
        let cal = self.calibration()?;
        finite_non_empty(query)?;
        dim_eq(cal.means.len(), query.len())?;
        dim_eq(cal.means.len(), quantized.dim)?;
        if metric != DistanceMetric::Hamming {
            return Err(IqdbError::InvalidMetric);
        }
        // Binarize the query against the same trained thresholds the stored
        // code was built from, then Hamming via packed XOR + popcount.
        let query_words = pack_bits(query, &cal.means);
        let mut diff: u32 = 0;
        for (q, c) in query_words.iter().zip(quantized.words.iter()) {
            diff = diff.saturating_add((q ^ c).count_ones());
        }
        Ok(diff as f32)
    }
}

/// Pack one bit per component of `vector` into `u64` words, with the bit
/// set when `vector[i] >= means[i]`. The trailing word's unused high bits
/// are zero so they cannot contribute to Hamming distance.
fn pack_bits(vector: &[f32], means: &[f32]) -> Vec<u64> {
    let dim = vector.len();
    let words = dim.div_ceil(BITS_PER_WORD);
    let mut out = vec![0_u64; words];
    for i in 0..dim {
        if vector[i] >= means[i] {
            out[i / BITS_PER_WORD] |= 1_u64 << (i % BITS_PER_WORD);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use iqdb_types::{DistanceMetric, IqdbError};

    fn trained_unit() -> BinaryQuantizer {
        let mut bq = BinaryQuantizer::new();
        bq.train(&[&[0.0_f32, 1.0, 2.0][..], &[2.0_f32, 1.0, 0.0][..]])
            .unwrap();
        bq
    }

    #[test]
    fn quantize_before_train_returns_invalid_config() {
        let bq = BinaryQuantizer::new();
        let err = bq.quantize(&[0.5_f32, 0.5]).unwrap_err();
        assert!(
            matches!(err, IqdbError::InvalidConfig { .. }),
            "expected InvalidConfig, got {err:?}",
        );
    }

    #[test]
    fn distance_before_train_returns_invalid_config() {
        let bq = BinaryQuantizer::new();
        let code = BqCode {
            words: vec![0],
            dim: 3,
        };
        let err = bq
            .distance(&[0.0_f32, 0.0, 0.0], &code, DistanceMetric::Hamming)
            .unwrap_err();
        assert!(
            matches!(err, IqdbError::InvalidConfig { .. }),
            "expected InvalidConfig, got {err:?}",
        );
    }

    #[test]
    fn dequantize_before_train_returns_invalid_config() {
        let bq = BinaryQuantizer::new();
        let code = BqCode {
            words: vec![0],
            dim: 3,
        };
        let err = bq.dequantize(&code).unwrap_err();
        assert!(
            matches!(err, IqdbError::InvalidConfig { .. }),
            "expected InvalidConfig, got {err:?}",
        );
    }

    #[test]
    fn train_empty_set_returns_invalid_config() {
        let mut bq = BinaryQuantizer::new();
        let empty: [&[f32]; 0] = [];
        let err = bq.train(&empty).unwrap_err();
        assert!(
            matches!(err, IqdbError::InvalidConfig { .. }),
            "expected InvalidConfig, got {err:?}",
        );
    }

    #[test]
    fn train_inconsistent_dim_returns_dimension_mismatch() {
        let mut bq = BinaryQuantizer::new();
        let a = [0.0_f32, 1.0, 2.0];
        let b = [1.0_f32, 0.0];
        let err = bq.train(&[&a[..], &b[..]]).unwrap_err();
        assert_eq!(
            err,
            IqdbError::DimensionMismatch {
                expected: 3,
                found: 2,
            },
        );
    }

    #[test]
    fn train_non_finite_returns_invalid_vector() {
        let mut bq = BinaryQuantizer::new();
        let v = [1.0_f32, f32::NAN];
        assert_eq!(bq.train(&[&v[..]]).unwrap_err(), IqdbError::InvalidVector,);
    }

    #[test]
    fn quantize_dim_mismatch_returns_dimension_mismatch() {
        let bq = trained_unit();
        let err = bq.quantize(&[0.5_f32, 0.5]).unwrap_err();
        assert_eq!(
            err,
            IqdbError::DimensionMismatch {
                expected: 3,
                found: 2,
            },
        );
    }

    #[test]
    fn quantize_non_finite_returns_invalid_vector() {
        let bq = trained_unit();
        let err = bq.quantize(&[0.5_f32, f32::NEG_INFINITY, 0.5]).unwrap_err();
        assert_eq!(err, IqdbError::InvalidVector);
    }

    #[test]
    fn distance_rejects_non_hamming_metrics() {
        let bq = trained_unit();
        let code = bq.quantize(&[0.5_f32, 0.5, 0.5]).unwrap();
        let q = [0.5_f32, 0.5, 0.5];
        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::DotProduct,
            DistanceMetric::Euclidean,
            DistanceMetric::Manhattan,
        ] {
            assert_eq!(
                bq.distance(&q, &code, metric).unwrap_err(),
                IqdbError::InvalidMetric,
                "metric {metric:?} must be rejected",
            );
        }
    }

    #[test]
    fn distance_self_consistency_is_zero() {
        // The query path MUST binarize against the same trained means used
        // to build the stored code. If it didn't, `distance(v, code(v))`
        // would generally be > 0.
        let bq = trained_unit();
        let v = [0.4_f32, 1.1, 1.9];
        let code = bq.quantize(&v).unwrap();
        let d = bq.distance(&v, &code, DistanceMetric::Hamming).unwrap();
        assert_eq!(d, 0.0);
    }

    fn naive_hamming(a: &[u64], b: &[u64]) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum()
    }

    #[test]
    fn hamming_matches_naive_popcount_reference() {
        let mut bq = BinaryQuantizer::new();
        // 70-dim training -> two words per code, with 6 padding bits in
        // the trailing word that MUST stay zero.
        let dim = 70;
        let a: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();
        bq.train(&[&a[..], &b[..]]).unwrap();

        let query: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.5).sin()).collect();
        let code = bq.quantize(&b).unwrap();
        let d = bq.distance(&query, &code, DistanceMetric::Hamming).unwrap();

        let cal = bq.calibration.as_ref().unwrap();
        let query_words = pack_bits(&query, &cal.means);
        let expected = naive_hamming(&query_words, &code.words);
        assert_eq!(d as u32, expected);
    }

    #[test]
    fn quantize_zeros_padding_bits_for_dim_not_multiple_of_64() {
        let dims = [63_usize, 64, 65, 127, 128, 129];
        for &dim in &dims {
            // Train with all-zeros and all-ones so the per-dim mean is 0.5;
            // an all-ones query then sets every meaningful bit, leaving the
            // padding bits in the trailing word as the only source of 0s
            // above `dim`.
            let zeros = vec![0.0_f32; dim];
            let ones = vec![1.0_f32; dim];
            let mut bq = BinaryQuantizer::new();
            bq.train(&[&zeros[..], &ones[..]]).unwrap();

            let code = bq.quantize(&ones).unwrap();
            assert_eq!(code.dim, dim);
            assert_eq!(code.words.len(), dim.div_ceil(BITS_PER_WORD));

            let used_in_last = dim % BITS_PER_WORD;
            if used_in_last != 0 {
                let last = *code.words.last().unwrap();
                let padding_mask = !0_u64 << used_in_last;
                assert_eq!(
                    last & padding_mask,
                    0,
                    "dim={dim}: padding bits must be zero",
                );
            }
        }
    }
}
