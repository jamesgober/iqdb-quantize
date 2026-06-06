//! Binary quantization (BQ): the highest-compression scheme — one bit per
//! dimension, 32x smaller, scored with Hamming distance on packed `u64` words.
//! Each bit records whether a component is at or above its trained per-dimension
//! mean. BQ supports `Hamming` only; any other metric is a typed error.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example binary_quantization
//! ```

use iqdb_quantize::{BinaryQuantizer, Quantizer};
use iqdb_types::{DistanceMetric, IqdbError};

fn main() -> Result<(), IqdbError> {
    // Two clusters of 4-dimensional vectors: one "low-high" pattern, one "high-low".
    let corpus: [&[f32]; 4] = [
        &[0.1, 0.2, 0.9, 0.8], // low, low, high, high
        &[0.2, 0.1, 0.8, 0.9], // same pattern
        &[0.9, 0.8, 0.1, 0.2], // high, high, low, low
        &[0.8, 0.9, 0.2, 0.1], // same pattern
    ];

    let mut bq = BinaryQuantizer::new();
    bq.train(&corpus)?;

    let codes = corpus
        .iter()
        .map(|v| bq.quantize(v))
        .collect::<Result<Vec<_>, _>>()?;

    let f32_bytes = size_of_val(corpus[0]);
    let code_bytes = size_of_val(codes[0].as_words());
    println!(
        "dim {} packs into {} u64 word(s) = {code_bytes} bytes (from {f32_bytes} f32 bytes; \
         one bit per dimension, 32x at realistic dims)",
        codes[0].dim(),
        codes[0].as_words().len(),
    );

    // Hamming distance between every pair: same-pattern pairs are near (0),
    // opposite-pattern pairs are far (every bit differs).
    println!("\npairwise Hamming distance:");
    for code in &codes {
        for other in &corpus {
            let d = bq.distance(other, code, DistanceMetric::Hamming)?;
            print!("  {d:>3.0}");
        }
        println!();
    }

    // A non-Hamming metric is rejected rather than silently misused.
    let err = bq
        .distance(&[0.5_f32, 0.5, 0.5, 0.5], &codes[0], DistanceMetric::Cosine)
        .unwrap_err();
    println!("\nCosine against a binary code is rejected: {err:?}");

    Ok(())
}
