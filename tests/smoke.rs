//! End-to-end smoke coverage for the `iqdb-quantize` public surface.
//!
//! Exercises both quantizers through the [`Quantizer`] trait, the owned
//! code types, and the trained-dimension accessor on each scheme.

#![allow(clippy::unwrap_used)]

use iqdb_quantize::{
    BinaryQuantizer, BqCode, PqAdcTables, PqCode, ProductQuantizer, Quantizer, ScalarQuantizer,
    Sq8Code, VERSION,
};
use iqdb_types::DistanceMetric;

#[test]
fn version_is_semver_triplet() {
    assert_eq!(VERSION.split('.').count(), 3);
    assert!(VERSION.split('.').all(|part| !part.is_empty()));
}

#[test]
fn sq8_round_trip_smoke() {
    let mut sq = ScalarQuantizer::new();
    sq.train(&[&[0.0_f32, 1.0, 2.0][..], &[1.0_f32, 0.0, 1.0][..]])
        .unwrap();
    assert_eq!(sq.dim(), Some(3));

    let code: Sq8Code = sq.quantize(&[0.5_f32, 0.5, 1.5]).unwrap();
    assert_eq!(code.len(), 3);
    assert_eq!(code.as_bytes().len(), 3);

    let decoded = sq.dequantize(&code).unwrap();
    assert_eq!(decoded.len(), 3);
    for v in &decoded {
        assert!(v.is_finite());
    }
}

#[test]
fn sq8_distance_through_every_metric() {
    let mut sq = ScalarQuantizer::new();
    sq.train(&[&[0.0_f32, 1.0, 2.0][..], &[2.0_f32, 0.0, 1.0][..]])
        .unwrap();
    let code = sq.quantize(&[1.0_f32, 0.5, 1.5]).unwrap();
    let q = [1.0_f32, 0.5, 1.5];
    for metric in [
        DistanceMetric::Cosine,
        DistanceMetric::DotProduct,
        DistanceMetric::Euclidean,
        DistanceMetric::Manhattan,
        DistanceMetric::Hamming,
    ] {
        let d = sq.distance(&q, &code, metric).unwrap();
        assert!(d.is_finite(), "metric {metric:?} returned non-finite {d}");
    }
}

#[test]
fn bq_round_trip_smoke() {
    let mut bq = BinaryQuantizer::new();
    bq.train(&[&[0.0_f32, 1.0, 2.0][..], &[2.0_f32, 1.0, 0.0][..]])
        .unwrap();
    assert_eq!(bq.dim(), Some(3));

    let code: BqCode = bq.quantize(&[0.5_f32, 1.5, 2.5]).unwrap();
    assert_eq!(code.dim(), 3);
    assert_eq!(code.as_words().len(), 1);

    let decoded = bq.dequantize(&code).unwrap();
    assert_eq!(decoded.len(), 3);
    for v in &decoded {
        assert!(*v == 1.0 || *v == -1.0);
    }
}

#[test]
fn bq_hamming_self_distance_is_zero() {
    let mut bq = BinaryQuantizer::new();
    bq.train(&[&[0.0_f32, 1.0, 2.0][..], &[2.0_f32, 1.0, 0.0][..]])
        .unwrap();
    let v = [0.4_f32, 1.1, 1.9];
    let code = bq.quantize(&v).unwrap();
    let d = bq.distance(&v, &code, DistanceMetric::Hamming).unwrap();
    assert_eq!(d, 0.0);
}

/// Build 32 training vectors of `dim` 8 with 4 visually-separable
/// clusters at 0, 5, 10, 15 along each component, so PQ has plenty of
/// material to learn 4 centroids per subvector.
fn pq_smoke_training_data() -> Vec<Vec<f32>> {
    let mut data: Vec<Vec<f32>> = Vec::with_capacity(32);
    for centre in &[0.0_f32, 5.0, 10.0, 15.0] {
        for j in 0..8 {
            let jitter = (j as f32) * 0.1 - 0.35;
            let v = (0..8)
                .map(|k| centre + jitter + (k as f32) * 0.01)
                .collect();
            data.push(v);
        }
    }
    data
}

#[test]
fn pq_round_trip_smoke() {
    let mut pq = ProductQuantizer::with_config(4, 4, 7);
    let training = pq_smoke_training_data();
    let refs: Vec<&[f32]> = training.iter().map(Vec::as_slice).collect();
    pq.train(&refs).unwrap();
    assert_eq!(pq.dim(), Some(8));

    let v = vec![5.0_f32, 5.01, 5.02, 5.03, 5.04, 5.05, 5.06, 5.07];
    let code: PqCode = pq.quantize(&v).unwrap();
    assert_eq!(code.dim(), 8);
    assert_eq!(code.n_subvectors(), 4);
    assert_eq!(code.len(), 4);
    assert_eq!(code.as_bytes().len(), 4);

    let decoded = pq.dequantize(&code).unwrap();
    assert_eq!(decoded.len(), 8);
    for x in &decoded {
        assert!(x.is_finite());
    }
    // The recovered vector should land near the input (within one
    // codebook step on this very tight cluster).
    let l1: f32 = v
        .iter()
        .zip(decoded.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(l1 < 8.0, "PQ round-trip L1 error {l1} too large");
}

#[test]
fn pq_distance_through_supported_metrics() {
    let mut pq = ProductQuantizer::with_config(4, 4, 11);
    let training = pq_smoke_training_data();
    let refs: Vec<&[f32]> = training.iter().map(Vec::as_slice).collect();
    pq.train(&refs).unwrap();

    let v = vec![10.0_f32, 10.01, 10.02, 10.03, 10.04, 10.05, 10.06, 10.07];
    let code = pq.quantize(&v).unwrap();
    let q = v.clone();
    for metric in [
        DistanceMetric::Euclidean,
        DistanceMetric::DotProduct,
        DistanceMetric::Manhattan,
    ] {
        let d = pq.distance(&q, &code, metric).unwrap();
        assert!(d.is_finite(), "metric {metric:?} returned non-finite {d}");
    }
}

#[test]
fn pq_build_query_tables_then_score_matches_distance() {
    // Build the table once for a (query, metric), score many codes
    // through it, and compare each score against the single-shot
    // `ProductQuantizer::distance`. They must agree byte-for-byte —
    // `distance` now delegates through `build_query_tables`.
    let mut pq = ProductQuantizer::with_config(4, 4, 13);
    let training = pq_smoke_training_data();
    let refs: Vec<&[f32]> = training.iter().map(Vec::as_slice).collect();
    pq.train(&refs).unwrap();

    // A handful of distinct vectors → distinct codes.
    let inputs: Vec<Vec<f32>> = vec![
        vec![1.0_f32, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
        vec![5.0_f32, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7],
        vec![9.0_f32, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7],
        vec![3.5_f32, 4.2, 8.1, 0.9, 6.4, 2.2, 7.7, 1.1],
    ];
    let codes: Vec<PqCode> = inputs.iter().map(|v| pq.quantize(v).unwrap()).collect();

    let query = vec![4.0_f32, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7];

    for metric in [
        DistanceMetric::Euclidean,
        DistanceMetric::DotProduct,
        DistanceMetric::Manhattan,
    ] {
        let tables: PqAdcTables = pq.build_query_tables(&query, metric).unwrap();
        assert_eq!(tables.metric(), metric);
        assert_eq!(tables.n_subvectors(), pq.n_subvectors());
        assert_eq!(tables.n_centroids(), pq.n_centroids());
        assert_eq!(tables.dim(), pq.dim().unwrap());

        for code in &codes {
            let batched = tables.distance(code).unwrap();
            let direct = pq.distance(&query, code, metric).unwrap();
            assert_eq!(
                batched.to_bits(),
                direct.to_bits(),
                "metric {metric:?}: tables.distance != pq.distance ({batched} vs {direct})"
            );
        }
    }
}

#[test]
fn pq_build_query_tables_rejects_unsupported_metrics() {
    let mut pq = ProductQuantizer::with_config(4, 4, 17);
    let training = pq_smoke_training_data();
    let refs: Vec<&[f32]> = training.iter().map(Vec::as_slice).collect();
    pq.train(&refs).unwrap();

    let q = vec![1.0_f32, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7];
    for metric in [DistanceMetric::Cosine, DistanceMetric::Hamming] {
        let err = pq.build_query_tables(&q, metric).unwrap_err();
        assert!(matches!(err, iqdb_types::IqdbError::InvalidMetric));
    }
}
