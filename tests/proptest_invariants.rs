//! Property-based invariants for both quantizers.
//!
//! All round-trip inputs are drawn from inside the trained range so the
//! per-dimension SQ8 error bound `(max - min) / 255` actually applies;
//! out-of-range / clamp behaviour lives in `edge_cases.rs`. Distance
//! finiteness is asserted for every metric. Non-negativity is asserted
//! only for Cosine, Euclidean, Manhattan, and Hamming — `DotProduct`
//! distance is stored as `-dot` in `iqdb-distance` to honour "smaller is
//! nearer", so it is legitimately negative for any pair of positively
//! correlated vectors.

#![allow(clippy::unwrap_used)]

use iqdb_quantize::{BinaryQuantizer, ProductQuantizer, Quantizer, ScalarQuantizer};
use iqdb_types::DistanceMetric;
use proptest::prelude::*;

/// Metrics for which "smaller is nearer" coincides with "non-negative".
const NON_NEGATIVE_METRICS: [DistanceMetric; 4] = [
    DistanceMetric::Cosine,
    DistanceMetric::Euclidean,
    DistanceMetric::Manhattan,
    DistanceMetric::Hamming,
];

const ALL_METRICS: [DistanceMetric; 5] = [
    DistanceMetric::Cosine,
    DistanceMetric::DotProduct,
    DistanceMetric::Euclidean,
    DistanceMetric::Manhattan,
    DistanceMetric::Hamming,
];

/// Generate a vector of `dim` components drawn from a fixed, bounded
/// range. The trained range covers `[-1.0, 1.0]`, so every drawn input is
/// in-range for the SQ8 round-trip bound.
fn vector_strategy(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-1.0_f32..=1.0_f32, dim..=dim)
}

proptest! {
    /// SQ8 round-trip error per dimension is at most one quantization step,
    /// given inputs drawn from within the trained range.
    #[test]
    fn sq8_round_trip_within_step(v in vector_strategy(8)) {
        let mut sq = ScalarQuantizer::new();
        // Train on the corners of the box; this puts the input strictly
        // inside the trained range so the affine encoding is well-defined.
        let lows = [-1.0_f32; 8];
        let highs = [1.0_f32; 8];
        sq.train(&[&lows[..], &highs[..]]).unwrap();

        let code = sq.quantize(&v).unwrap();
        let decoded = sq.dequantize(&code).unwrap();
        // step = range / 255 = 2.0 / 255; tolerate one full step plus a
        // tiny rounding cushion.
        let step = 2.0_f32 / 255.0 + 1e-6;
        for (original, got) in v.iter().zip(decoded.iter()) {
            prop_assert!((original - got).abs() <= step);
        }
    }

    /// SQ8 asymmetric distance is finite for every metric, and non-negative
    /// for every metric whose distance has that contract.
    #[test]
    fn sq8_distance_is_finite_and_metric_aware(v in vector_strategy(8), q in vector_strategy(8)) {
        let mut sq = ScalarQuantizer::new();
        let lows = [-1.0_f32; 8];
        let highs = [1.0_f32; 8];
        sq.train(&[&lows[..], &highs[..]]).unwrap();
        let code = sq.quantize(&v).unwrap();

        for metric in ALL_METRICS {
            let d = sq.distance(&q, &code, metric).unwrap();
            prop_assert!(d.is_finite(), "metric {:?} returned non-finite {}", metric, d);
            if NON_NEGATIVE_METRICS.contains(&metric) {
                prop_assert!(d >= 0.0, "metric {:?} returned negative {}", metric, d);
            }
        }
    }

    /// BQ Hamming distance is finite, non-negative, and bounded by the dim.
    #[test]
    fn bq_hamming_is_in_bounds(v in vector_strategy(70), q in vector_strategy(70)) {
        let mut bq = BinaryQuantizer::new();
        let lows = [-1.0_f32; 70];
        let highs = [1.0_f32; 70];
        bq.train(&[&lows[..], &highs[..]]).unwrap();

        let code = bq.quantize(&v).unwrap();
        let d = bq.distance(&q, &code, DistanceMetric::Hamming).unwrap();
        prop_assert!(d.is_finite());
        prop_assert!(d >= 0.0);
        prop_assert!(d <= 70.0);
    }

    /// BQ self-distance is always zero: the query path uses the same trained
    /// thresholds as the stored code path.
    #[test]
    fn bq_self_distance_is_zero(v in vector_strategy(70)) {
        let mut bq = BinaryQuantizer::new();
        let lows = [-1.0_f32; 70];
        let highs = [1.0_f32; 70];
        bq.train(&[&lows[..], &highs[..]]).unwrap();
        let code = bq.quantize(&v).unwrap();
        let d = bq.distance(&v, &code, DistanceMetric::Hamming).unwrap();
        prop_assert_eq!(d, 0.0);
    }
}

/// PQ-specific metrics: Cosine and Hamming are rejected, the rest are
/// supported. Used by the PQ property block below.
const PQ_SUPPORTED_METRICS: [DistanceMetric; 3] = [
    DistanceMetric::Euclidean,
    DistanceMetric::DotProduct,
    DistanceMetric::Manhattan,
];

/// Train a PQ with deterministic seed `7`, `M = 2, K = 4`, dim 8.
///
/// Training set spans the corners of the `[-1, 1]` hypercube enough to
/// give k-means real geometry to learn without depending on `proptest`
/// inputs (which we want to remain free-form for the ADC properties).
fn pq_for_proptest() -> ProductQuantizer {
    let mut pq = ProductQuantizer::with_config(2, 4, 7);
    let training: Vec<Vec<f32>> = vec![
        vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        vec![-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
        vec![-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
        vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        vec![0.0, 0.5, -0.5, 0.0, 0.5, 0.0, -0.5, 0.5],
        vec![-0.5, 0.0, 0.5, -0.5, 0.0, 0.5, 0.0, -0.5],
    ];
    let refs: Vec<&[f32]> = training.iter().map(Vec::as_slice).collect();
    pq.train(&refs).unwrap();
    pq
}

proptest! {
    /// PQ ADC distance is finite for every supported metric.
    #[test]
    fn pq_distance_is_finite_for_supported_metrics(
        v in vector_strategy(8),
        q in vector_strategy(8),
    ) {
        let pq = pq_for_proptest();
        let code = pq.quantize(&v).unwrap();
        for metric in PQ_SUPPORTED_METRICS {
            let d = pq.distance(&q, &code, metric).unwrap();
            prop_assert!(d.is_finite(), "metric {:?} returned non-finite {}", metric, d);
            if matches!(metric, DistanceMetric::Euclidean | DistanceMetric::Manhattan) {
                prop_assert!(d >= 0.0, "metric {:?} returned negative {}", metric, d);
            }
        }
    }

    /// PQ ADC equals dequantize-then-`iqdb_distance::compute` (within
    /// floating-point reduction tolerance) for every supported metric.
    /// This is the "asymmetric distance is mathematically exact for
    /// subvector-decomposable metrics" invariant from the design doc.
    #[test]
    fn pq_adc_matches_dequantize_then_compute(
        v in vector_strategy(8),
        q in vector_strategy(8),
    ) {
        let pq = pq_for_proptest();
        let code = pq.quantize(&v).unwrap();
        let decoded = pq.dequantize(&code).unwrap();
        for metric in PQ_SUPPORTED_METRICS {
            let adc = pq.distance(&q, &code, metric).unwrap();
            let reference = iqdb_distance::compute(metric, &q, &decoded).unwrap();
            // f32 reductions in different orders can disagree by a few
            // ULPs even when the math is identical; a relative+absolute
            // tolerance covers both small and large totals.
            let tol = 1e-4_f32 + 1e-4_f32 * reference.abs().max(adc.abs());
            prop_assert!(
                (adc - reference).abs() <= tol,
                "metric {:?}: ADC {} vs reference {} (tol {})",
                metric, adc, reference, tol,
            );
        }
    }

    /// `ProductQuantizer::distance` and the build-once / score-many
    /// path via `PqAdcTables::distance` produce **byte-identical**
    /// floats. `distance` is now a thin wrapper around
    /// `build_query_tables` + `PqAdcTables::distance`, so any
    /// disagreement is a refactor regression.
    #[test]
    fn pq_adc_tables_distance_equals_pq_distance_byte_for_byte(
        v in vector_strategy(8),
        q in vector_strategy(8),
    ) {
        let pq = pq_for_proptest();
        let code = pq.quantize(&v).unwrap();
        for metric in PQ_SUPPORTED_METRICS {
            let tables = pq.build_query_tables(&q, metric).unwrap();
            let batched = tables.distance(&code).unwrap();
            let direct = pq.distance(&q, &code, metric).unwrap();
            prop_assert_eq!(
                batched.to_bits(),
                direct.to_bits(),
                "metric {:?}: tables.distance != pq.distance",
                metric,
            );
        }
    }
}
