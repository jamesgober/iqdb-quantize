//! Determinism guarantees for [`ProductQuantizer`].
//!
//! The PQ training pipeline is seeded end-to-end (SplitMix64 PRNG,
//! fixed-order reductions, f64 accumulators). Same seed + same training
//! data ⇒ byte-identical codebooks and byte-identical codes.

#![allow(clippy::unwrap_used)]

use iqdb_quantize::{ProductQuantizer, Quantizer};

fn training_data() -> Vec<Vec<f32>> {
    // 64 vectors of dim 12 across 8 broad clusters — gives PQ room to
    // learn a meaningful K=8 codebook per subvector at M=4.
    let mut data: Vec<Vec<f32>> = Vec::with_capacity(64);
    for cluster in 0..8 {
        let centre = (cluster as f32) * 3.0 - 10.0;
        for j in 0..8 {
            let jitter = (j as f32) * 0.05;
            data.push(
                (0..12)
                    .map(|k| centre + jitter + (k as f32) * 0.02)
                    .collect(),
            );
        }
    }
    data
}

#[test]
fn pq_train_is_deterministic_under_seed() {
    let data = training_data();
    let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();

    let mut a = ProductQuantizer::with_config(4, 8, 0xDEAD_BEEF);
    a.train(&refs).unwrap();
    let mut b = ProductQuantizer::with_config(4, 8, 0xDEAD_BEEF);
    b.train(&refs).unwrap();

    // Two quantizers trained with the same seed + data must produce the
    // same codes on every input.
    let probes: [[f32; 12]; 4] = [
        [-8.0; 12],
        [0.0; 12],
        [7.5; 12],
        [
            -10.0, -7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, -10.0, -7.0, -4.0, -1.0,
        ],
    ];
    for probe in &probes {
        let code_a = a.quantize(probe).unwrap();
        let code_b = b.quantize(probe).unwrap();
        assert_eq!(
            code_a.as_bytes(),
            code_b.as_bytes(),
            "PQ codes must match across two same-seeded trainings (probe {probe:?})",
        );
        // And dequantizing the two codes must reconstruct the same
        // approximate vectors.
        let dec_a = a.dequantize(&code_a).unwrap();
        let dec_b = b.dequantize(&code_b).unwrap();
        assert_eq!(dec_a, dec_b, "dequantized PQ vectors must match");
    }
}

#[test]
fn pq_train_with_different_seeds_can_diverge() {
    let data = training_data();
    let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();

    let mut a = ProductQuantizer::with_config(4, 8, 1);
    a.train(&refs).unwrap();
    let mut b = ProductQuantizer::with_config(4, 8, 2);
    b.train(&refs).unwrap();

    // We expect at least one probe code to differ between the two
    // seeds — k-means++ initial centroid picks diverge under different
    // seeds and propagate through Lloyd's iterations.
    let probes: [[f32; 12]; 4] = [
        [-8.0; 12],
        [0.0; 12],
        [7.5; 12],
        [
            -10.0, -7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, -10.0, -7.0, -4.0, -1.0,
        ],
    ];
    let any_differ = probes.iter().any(|probe| {
        a.quantize(probe).unwrap().as_bytes() != b.quantize(probe).unwrap().as_bytes()
    });
    assert!(
        any_differ,
        "different seeds should produce different codebooks on at least one probe",
    );
}
