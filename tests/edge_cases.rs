//! Edge-case coverage for the public surface.
//!
//! Boundary inputs that every method MUST tolerate without panicking:
//! empty / single-vector training, mismatched dimensions, quantize before
//! train, NaN / infinite components, out-of-range values clamped at encode
//! time, and a non-Hamming metric against [`BinaryQuantizer`].

#![allow(clippy::unwrap_used)]

use iqdb_quantize::{BinaryQuantizer, ProductQuantizer, Quantizer, ScalarQuantizer};
use iqdb_types::{DistanceMetric, IqdbError};

#[test]
fn sq8_train_rejects_empty_set() {
    let mut sq = ScalarQuantizer::new();
    let empty: [&[f32]; 0] = [];
    let err = sq.train(&empty).unwrap_err();
    assert!(
        matches!(err, IqdbError::InvalidConfig { .. }),
        "expected InvalidConfig, got {err:?}",
    );
}

#[test]
fn sq8_train_accepts_single_vector() {
    let mut sq = ScalarQuantizer::new();
    sq.train(&[&[1.0_f32, 2.0, 3.0][..]]).unwrap();
    // Every dim has max == min -> zero-range lane, every code byte is 0.
    let code = sq.quantize(&[1.0_f32, 2.0, 3.0]).unwrap();
    assert!(code.as_bytes().iter().all(|&b| b == 0));
    let decoded = sq.dequantize(&code).unwrap();
    assert_eq!(decoded, vec![1.0, 2.0, 3.0]);
}

#[test]
fn sq8_quantize_empty_vector_rejected() {
    let mut sq = ScalarQuantizer::new();
    sq.train(&[&[0.0_f32, 1.0][..]]).unwrap();
    let empty: [f32; 0] = [];
    // The emptiness check fires before the dim check, so an empty input
    // surfaces as InvalidVector regardless of trained dim.
    assert_eq!(sq.quantize(&empty).unwrap_err(), IqdbError::InvalidVector);
}

#[test]
fn sq8_quantize_nan_inf_rejected() {
    let mut sq = ScalarQuantizer::new();
    sq.train(&[&[0.0_f32, 1.0][..]]).unwrap();
    assert_eq!(
        sq.quantize(&[1.0, f32::NAN]).unwrap_err(),
        IqdbError::InvalidVector,
    );
    assert_eq!(
        sq.quantize(&[1.0, f32::INFINITY]).unwrap_err(),
        IqdbError::InvalidVector,
    );
    assert_eq!(
        sq.quantize(&[1.0, f32::NEG_INFINITY]).unwrap_err(),
        IqdbError::InvalidVector,
    );
}

#[test]
fn sq8_quantize_clamps_out_of_range_values() {
    let mut sq = ScalarQuantizer::new();
    sq.train(&[&[0.0_f32, 1.0][..], &[1.0_f32, 0.0][..]])
        .unwrap();
    // Way below min -> byte 0; way above max -> byte 255.
    let code = sq.quantize(&[-1.0e6_f32, 1.0e6_f32]).unwrap();
    assert_eq!(code.as_bytes()[0], 0);
    assert_eq!(code.as_bytes()[1], u8::MAX);
}

#[test]
fn sq8_pre_train_methods_return_invalid_config() {
    let sq = ScalarQuantizer::new();
    let err = sq.quantize(&[0.5_f32, 0.5]).unwrap_err();
    assert!(
        matches!(err, IqdbError::InvalidConfig { .. }),
        "expected InvalidConfig, got {err:?}",
    );
}

#[test]
fn bq_train_rejects_empty_set() {
    let mut bq = BinaryQuantizer::new();
    let empty: [&[f32]; 0] = [];
    let err = bq.train(&empty).unwrap_err();
    assert!(
        matches!(err, IqdbError::InvalidConfig { .. }),
        "expected InvalidConfig, got {err:?}",
    );
}

#[test]
fn bq_train_accepts_single_vector() {
    let mut bq = BinaryQuantizer::new();
    bq.train(&[&[1.0_f32, 2.0, 3.0][..]]).unwrap();
    // mean == vector -> every component >= mean -> every bit is 1.
    let code = bq.quantize(&[1.0_f32, 2.0, 3.0]).unwrap();
    assert_eq!(code.as_words()[0] & 0b111, 0b111);
}

#[test]
fn bq_quantize_nan_inf_rejected() {
    let mut bq = BinaryQuantizer::new();
    bq.train(&[&[0.0_f32, 1.0][..]]).unwrap();
    assert_eq!(
        bq.quantize(&[1.0, f32::NAN]).unwrap_err(),
        IqdbError::InvalidVector,
    );
    assert_eq!(
        bq.quantize(&[1.0, f32::INFINITY]).unwrap_err(),
        IqdbError::InvalidVector,
    );
}

#[test]
fn bq_distance_rejects_non_hamming_metrics() {
    let mut bq = BinaryQuantizer::new();
    bq.train(&[&[0.0_f32, 1.0, 2.0][..], &[2.0_f32, 1.0, 0.0][..]])
        .unwrap();
    let code = bq.quantize(&[0.5_f32, 1.5, 2.5]).unwrap();
    let q = [0.5_f32, 1.5, 2.5];
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
fn bq_pre_train_methods_return_invalid_config() {
    let bq = BinaryQuantizer::new();
    let err = bq.quantize(&[0.5_f32, 0.5]).unwrap_err();
    assert!(
        matches!(err, IqdbError::InvalidConfig { .. }),
        "expected InvalidConfig, got {err:?}",
    );
}

// -- ProductQuantizer edge cases ------------------------------------

/// 16 vectors of dim 8 spread across 4 broad clusters — enough for a
/// `M=4, K=4` PQ to train without hitting "sample.len() < K" anywhere.
fn pq_edge_training_data() -> Vec<Vec<f32>> {
    let mut data: Vec<Vec<f32>> = Vec::with_capacity(16);
    for centre in &[-2.0_f32, 0.0, 2.0, 4.0] {
        for j in 0..4 {
            let jitter = (j as f32) * 0.05;
            data.push(
                (0..8)
                    .map(|k| centre + jitter + (k as f32) * 0.01)
                    .collect(),
            );
        }
    }
    data
}

#[test]
fn pq_train_rejects_empty_set() {
    let mut pq = ProductQuantizer::with_config(4, 4, 1);
    let empty: [&[f32]; 0] = [];
    let err = pq.train(&empty).unwrap_err();
    assert!(
        matches!(err, IqdbError::InvalidConfig { .. }),
        "expected InvalidConfig, got {err:?}",
    );
}

#[test]
fn pq_quantize_nan_inf_rejected() {
    let mut pq = ProductQuantizer::with_config(4, 4, 1);
    let data = pq_edge_training_data();
    let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
    pq.train(&refs).unwrap();

    let mut nan = vec![0.0_f32; 8];
    nan[3] = f32::NAN;
    assert_eq!(pq.quantize(&nan).unwrap_err(), IqdbError::InvalidVector);

    let mut inf = vec![0.0_f32; 8];
    inf[0] = f32::INFINITY;
    assert_eq!(pq.quantize(&inf).unwrap_err(), IqdbError::InvalidVector);

    let mut ninf = vec![0.0_f32; 8];
    ninf[7] = f32::NEG_INFINITY;
    assert_eq!(pq.quantize(&ninf).unwrap_err(), IqdbError::InvalidVector);
}

#[test]
fn pq_pre_train_methods_return_invalid_config() {
    let pq = ProductQuantizer::with_config(4, 4, 1);
    let err = pq.quantize(&[0.5_f32; 8]).unwrap_err();
    assert!(
        matches!(err, IqdbError::InvalidConfig { .. }),
        "expected InvalidConfig, got {err:?}",
    );
}

#[test]
fn pq_distance_rejects_cosine_and_hamming() {
    let mut pq = ProductQuantizer::with_config(4, 4, 1);
    let data = pq_edge_training_data();
    let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
    pq.train(&refs).unwrap();
    let v = vec![0.5_f32; 8];
    let code = pq.quantize(&v).unwrap();
    for metric in [DistanceMetric::Cosine, DistanceMetric::Hamming] {
        assert_eq!(
            pq.distance(&v, &code, metric).unwrap_err(),
            IqdbError::InvalidMetric,
            "metric {metric:?} must be rejected",
        );
    }
}

#[test]
fn pq_rejects_non_divisible_dim() {
    let mut pq = ProductQuantizer::with_config(3, 4, 1);
    // 16 vectors of dim 8 — 8 is not divisible by 3.
    let data = pq_edge_training_data();
    let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
    let err = pq.train(&refs).unwrap_err();
    assert!(
        matches!(err, IqdbError::InvalidConfig { .. }),
        "expected InvalidConfig, got {err:?}",
    );
}

#[test]
fn pq_rejects_too_few_training_vectors() {
    // K = 32 centroids but only 16 training vectors.
    let mut pq = ProductQuantizer::with_config(4, 32, 1);
    let data = pq_edge_training_data();
    let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
    let err = pq.train(&refs).unwrap_err();
    assert!(
        matches!(err, IqdbError::InvalidConfig { .. }),
        "expected InvalidConfig, got {err:?}",
    );
}

#[test]
fn pq_rejects_n_centroids_zero_or_over_256() {
    let data = pq_edge_training_data();
    let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();

    let mut zero = ProductQuantizer::with_config(4, 0, 1);
    assert!(matches!(
        zero.train(&refs).unwrap_err(),
        IqdbError::InvalidConfig { .. }
    ));

    let mut too_many = ProductQuantizer::with_config(4, 257, 1);
    assert!(matches!(
        too_many.train(&refs).unwrap_err(),
        IqdbError::InvalidConfig { .. }
    ));
}

#[test]
fn pq_rejects_n_subvectors_zero() {
    let mut pq = ProductQuantizer::with_config(0, 4, 1);
    let data = pq_edge_training_data();
    let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
    let err = pq.train(&refs).unwrap_err();
    assert!(
        matches!(err, IqdbError::InvalidConfig { .. }),
        "expected InvalidConfig, got {err:?}",
    );
}

#[test]
fn pq_n_subvectors_one_ok() {
    // M = 1 collapses PQ to a single full-vector codebook — a degenerate
    // but valid shape.
    let mut pq = ProductQuantizer::with_config(1, 4, 1);
    let data = pq_edge_training_data();
    let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
    pq.train(&refs).unwrap();
    let code = pq.quantize(&[1.0_f32; 8]).unwrap();
    assert_eq!(code.n_subvectors(), 1);
    assert_eq!(code.len(), 1);
    let decoded = pq.dequantize(&code).unwrap();
    assert_eq!(decoded.len(), 8);
}

#[test]
fn pq_n_subvectors_equals_dim_ok() {
    // M = dim → each subvector is a scalar; each codebook is K scalars.
    let mut pq = ProductQuantizer::with_config(8, 4, 1);
    let data = pq_edge_training_data();
    let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
    pq.train(&refs).unwrap();
    let code = pq.quantize(&[1.0_f32; 8]).unwrap();
    assert_eq!(code.n_subvectors(), 8);
    assert_eq!(code.len(), 8);
    let decoded = pq.dequantize(&code).unwrap();
    assert_eq!(decoded.len(), 8);
}
