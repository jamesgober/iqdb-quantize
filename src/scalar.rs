//! [`ScalarQuantizer`] — scalar quantization (SQ8, 4× compression).
//!
//! Codes are `u8` per dimension. The calibration is a per-dimension affine
//! map: each dimension stores its trained `min` and a `scale` defined as
//! `(max - min) / 255`. Encoding clamps the input into `[min, max]`, scales
//! it onto `[0, 255]`, and rounds to the nearest integer. Decoding reverses
//! the affine map. A dimension with `max == min` collapses to a `scale = 0`
//! lane: every code byte there is `0` and `dequantize` returns `min` —
//! there is no division by zero.
//!
//! Asymmetric distance keeps the query as `f32`, dequantizes the candidate
//! code into a temporary `Vec<f32>`, and delegates to
//! [`iqdb_distance::compute`] for every metric. The result honours the
//! "smaller is nearer" convention used by the rest of the iqdb spine.

use error_forge::ForgeError;
use iqdb_distance::compute;
use iqdb_types::{DistanceMetric, IqdbError, Result};

use crate::code::Sq8Code;
use crate::traits::Quantizer;
use crate::validate::{dim_eq, finite_non_empty, training_set};

/// Number of `u8` code levels above zero.
const LEVELS: f32 = 255.0;

/// Calibration learned during [`ScalarQuantizer::train`].
#[derive(Debug, Clone, PartialEq)]
struct Sq8Calibration {
    /// Per-dimension minimum from the training sample.
    mins: Vec<f32>,
    /// Per-dimension `(max - min) / 255`. Zero for any zero-range dimension
    /// (`max == min`); see [`ScalarQuantizer::quantize`] for the guard.
    scales: Vec<f32>,
}

/// Scalar quantizer (SQ8): one `u8` per dimension, 4× compression.
///
/// Build one with [`ScalarQuantizer::new`] (or [`Default`]), train it once
/// with a representative sample, then quantize and compare. The trained
/// quantizer is callable from multiple threads — it owns its calibration
/// by value and exposes no interior mutability.
///
/// # Examples
///
/// ```
/// use iqdb_quantize::{Quantizer, ScalarQuantizer};
/// use iqdb_types::DistanceMetric;
///
/// let mut sq = ScalarQuantizer::new();
/// sq.train(&[&[0.0_f32, 1.0, 2.0][..], &[1.0_f32, 0.0, 1.0][..]])
///     .expect("two non-empty, finite vectors of equal dim");
///
/// let code = sq.quantize(&[0.5_f32, 0.5, 1.5]).expect("dim matches");
/// let d = sq
///     .distance(&[0.5_f32, 0.5, 1.5], &code, DistanceMetric::Euclidean)
///     .expect("dim matches");
/// assert!(d.is_finite());
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ScalarQuantizer {
    calibration: Option<Sq8Calibration>,
}

impl ScalarQuantizer {
    /// Build an untrained scalar quantizer.
    ///
    /// Every hot method returns [`IqdbError::InvalidConfig`] until
    /// [`Quantizer::train`] succeeds.
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_quantize::ScalarQuantizer;
    /// let _sq = ScalarQuantizer::new();
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
    /// use iqdb_quantize::{Quantizer, ScalarQuantizer};
    ///
    /// let mut sq = ScalarQuantizer::new();
    /// assert_eq!(sq.dim(), None);
    /// sq.train(&[&[0.0_f32, 1.0][..]]).expect("ok");
    /// assert_eq!(sq.dim(), Some(2));
    /// ```
    #[must_use]
    pub fn dim(&self) -> Option<usize> {
        self.calibration.as_ref().map(|c| c.mins.len())
    }

    fn calibration(&self) -> Result<&Sq8Calibration> {
        self.calibration.as_ref().ok_or(IqdbError::InvalidConfig {
            reason: "ScalarQuantizer has not been trained",
        })
    }
}

impl Quantizer for ScalarQuantizer {
    type Quantized = Sq8Code;

    #[tracing::instrument(
        level = "info",
        skip_all,
        fields(quantizer = "sq8", training_size = vectors.len()),
    )]
    fn train(&mut self, vectors: &[&[f32]]) -> Result<()> {
        let dim = training_set(vectors).inspect_err(|err: &IqdbError| {
            tracing::error!(
                error.kind = err.kind(),
                error.reason = err.caption(),
                "scalar quantizer training failed",
            );
        })?;
        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];
        for v in vectors {
            for (i, &x) in v.iter().enumerate() {
                if x < mins[i] {
                    mins[i] = x;
                }
                if x > maxs[i] {
                    maxs[i] = x;
                }
            }
        }
        let mut scales = vec![0.0_f32; dim];
        for i in 0..dim {
            let range = maxs[i] - mins[i];
            scales[i] = if range > 0.0 { range / LEVELS } else { 0.0 };
        }
        self.calibration = Some(Sq8Calibration { mins, scales });
        Ok(())
    }

    fn quantize(&self, vector: &[f32]) -> Result<Self::Quantized> {
        let cal = self.calibration()?;
        finite_non_empty(vector)?;
        dim_eq(cal.mins.len(), vector.len())?;
        let mut bytes = Vec::with_capacity(vector.len());
        for (i, &x) in vector.iter().enumerate() {
            bytes.push(encode_scalar(x, cal.mins[i], cal.scales[i]));
        }
        Ok(Sq8Code { bytes })
    }

    fn dequantize(&self, quantized: &Self::Quantized) -> Result<Vec<f32>> {
        let cal = self.calibration()?;
        dim_eq(cal.mins.len(), quantized.bytes.len())?;
        let mut out = Vec::with_capacity(quantized.bytes.len());
        for (i, &b) in quantized.bytes.iter().enumerate() {
            out.push(decode_scalar(b, cal.mins[i], cal.scales[i]));
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
        dim_eq(cal.mins.len(), query.len())?;
        dim_eq(cal.mins.len(), quantized.bytes.len())?;
        let decoded = self.dequantize(quantized)?;
        compute(metric, query, &decoded)
    }
}

/// Encode one `f32` component into a `u8` code under an affine calibration.
///
/// Clamps the input into the trained range before the cast, so the `as u8`
/// step cannot trip release-mode out-of-range UB even on inputs well
/// outside `[min, max]`. A zero-`scale` lane (the `max == min` case in
/// training) always encodes to `0`.
fn encode_scalar(value: f32, min: f32, scale: f32) -> u8 {
    if scale <= 0.0 {
        return 0;
    }
    let normalised = ((value - min) / scale).round();
    if normalised <= 0.0 {
        0
    } else if normalised >= LEVELS {
        u8::MAX
    } else {
        normalised as u8
    }
}

/// Decode one `u8` code byte back to `f32` under an affine calibration.
fn decode_scalar(byte: u8, min: f32, scale: f32) -> f32 {
    if scale <= 0.0 {
        return min;
    }
    min + f32::from(byte) * scale
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use iqdb_types::{DistanceMetric, IqdbError};

    fn trained_unit() -> ScalarQuantizer {
        let mut sq = ScalarQuantizer::new();
        sq.train(&[&[0.0_f32, 1.0, 2.0][..], &[1.0_f32, 0.0, 1.0][..]])
            .unwrap();
        sq
    }

    #[test]
    fn quantize_before_train_returns_invalid_config() {
        let sq = ScalarQuantizer::new();
        let err = sq.quantize(&[0.5_f32, 0.5]).unwrap_err();
        assert!(
            matches!(err, IqdbError::InvalidConfig { .. }),
            "expected InvalidConfig, got {err:?}",
        );
    }

    #[test]
    fn distance_before_train_returns_invalid_config() {
        let sq = ScalarQuantizer::new();
        let code = Sq8Code {
            bytes: vec![0, 0, 0],
        };
        let err = sq
            .distance(&[0.5_f32, 0.5, 0.5], &code, DistanceMetric::Euclidean)
            .unwrap_err();
        assert!(
            matches!(err, IqdbError::InvalidConfig { .. }),
            "expected InvalidConfig, got {err:?}",
        );
    }

    #[test]
    fn dequantize_before_train_returns_invalid_config() {
        let sq = ScalarQuantizer::new();
        let code = Sq8Code { bytes: vec![0, 0] };
        let err = sq.dequantize(&code).unwrap_err();
        assert!(
            matches!(err, IqdbError::InvalidConfig { .. }),
            "expected InvalidConfig, got {err:?}",
        );
    }

    #[test]
    fn train_empty_set_returns_invalid_config() {
        let mut sq = ScalarQuantizer::new();
        let empty: [&[f32]; 0] = [];
        let err = sq.train(&empty).unwrap_err();
        assert!(
            matches!(err, IqdbError::InvalidConfig { .. }),
            "expected InvalidConfig, got {err:?}",
        );
    }

    #[test]
    fn train_inconsistent_dim_returns_dimension_mismatch() {
        let mut sq = ScalarQuantizer::new();
        let a = [0.0_f32, 1.0, 2.0];
        let b = [1.0_f32, 0.0];
        let err = sq.train(&[&a[..], &b[..]]).unwrap_err();
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
        let mut sq = ScalarQuantizer::new();
        let v = [1.0_f32, f32::NAN];
        assert_eq!(sq.train(&[&v[..]]).unwrap_err(), IqdbError::InvalidVector,);
    }

    #[test]
    fn quantize_dim_mismatch_returns_dimension_mismatch() {
        let sq = trained_unit();
        let err = sq.quantize(&[0.5_f32, 0.5]).unwrap_err();
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
        let sq = trained_unit();
        let err = sq.quantize(&[0.5_f32, f32::INFINITY, 0.5]).unwrap_err();
        assert_eq!(err, IqdbError::InvalidVector);
    }

    #[test]
    fn round_trip_within_per_dim_bound() {
        let sq = trained_unit();
        // Per-dim trained ranges: dim0=[0,1], dim1=[0,1], dim2=[1,2].
        // Per-dim max round-trip error <= scale = range/255 <= 1/255 here.
        let inputs = [0.1_f32, 0.5, 1.5];
        let code = sq.quantize(&inputs).unwrap();
        let decoded = sq.dequantize(&code).unwrap();
        for (i, (&expected, &got)) in inputs.iter().zip(decoded.iter()).enumerate() {
            let err = (expected - got).abs();
            // 1.0 / 255.0 plus a tiny rounding cushion.
            assert!(
                err <= 1.0 / 255.0 + 1e-6,
                "dim {i}: |{expected} - {got}| = {err}",
            );
        }
    }

    #[test]
    fn zero_range_dimension_does_not_panic_and_round_trips_to_min() {
        // All training vectors share dim0 = 7.0 -> max == min on that lane.
        let mut sq = ScalarQuantizer::new();
        sq.train(&[&[7.0_f32, 0.0][..], &[7.0_f32, 1.0][..]])
            .unwrap();

        let code = sq.quantize(&[7.0_f32, 0.5]).unwrap();
        let decoded = sq.dequantize(&code).unwrap();
        assert!((decoded[0] - 7.0).abs() < 1e-6);

        // Even an out-of-range value on dim0 cannot escape the lane.
        let code = sq.quantize(&[42.0_f32, 0.5]).unwrap();
        let decoded = sq.dequantize(&code).unwrap();
        assert!((decoded[0] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn distance_smaller_is_nearer_for_euclidean() {
        let sq = trained_unit();
        let near = sq.quantize(&[0.5_f32, 0.5, 1.5]).unwrap();
        let far = sq.quantize(&[1.0_f32, 0.0, 1.0]).unwrap();
        let q = [0.5_f32, 0.5, 1.5];
        let d_near = sq.distance(&q, &near, DistanceMetric::Euclidean).unwrap();
        let d_far = sq.distance(&q, &far, DistanceMetric::Euclidean).unwrap();
        assert!(d_near < d_far);
    }

    #[test]
    fn distance_matches_iqdb_distance_on_dequantized() {
        let sq = trained_unit();
        let q = [0.5_f32, 0.5, 1.5];
        let code = sq.quantize(&[0.4_f32, 0.6, 1.4]).unwrap();
        let decoded = sq.dequantize(&code).unwrap();
        let via_quant = sq.distance(&q, &code, DistanceMetric::Cosine).unwrap();
        let direct = compute(DistanceMetric::Cosine, &q, &decoded).unwrap();
        assert_eq!(via_quant.to_bits(), direct.to_bits());
    }

    #[test]
    fn encode_clamps_below_range() {
        // scale > 0, value far below min: byte is 0, no UB on the cast.
        assert_eq!(encode_scalar(-1e9, 0.0, 1.0 / 255.0), 0);
    }

    #[test]
    fn encode_clamps_above_range() {
        // scale > 0, value far above max: byte is 255, no UB on the cast.
        assert_eq!(encode_scalar(1e9, 0.0, 1.0 / 255.0), u8::MAX);
    }
}
