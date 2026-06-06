//! Scalar quantization (SQ8): the simplest scheme, and the one that supports
//! every distance metric. Train on a representative sample, compress each
//! vector to one byte per dimension (4x smaller), then score a raw `f32` query
//! against a stored code with asymmetric distance.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example scalar_quantization
//! ```

use iqdb_quantize::{Quantizer, ScalarQuantizer};
use iqdb_types::{DistanceMetric, IqdbError};

fn main() -> Result<(), IqdbError> {
    // A tiny 8-dimensional corpus standing in for real embeddings.
    let corpus: [&[f32]; 4] = [
        &[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        &[0.15, 0.18, 0.32, 0.41, 0.49, 0.58, 0.72, 0.79],
        &[0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20],
        &[0.05, 0.95, 0.05, 0.95, 0.05, 0.95, 0.05, 0.95],
    ];

    // Train once on the sample you intend to index.
    let mut sq = ScalarQuantizer::new();
    sq.train(&corpus)?;
    println!(
        "trained SQ8 on {} vectors, dim = {:?}",
        corpus.len(),
        sq.dim()
    );

    // Compress the corpus. Each code is one byte per dimension.
    let codes = corpus
        .iter()
        .map(|v| sq.quantize(v))
        .collect::<Result<Vec<_>, _>>()?;

    let f32_bytes = size_of_val(corpus[0]);
    let code_bytes = codes[0].len();
    println!(
        "each vector: {f32_bytes} bytes of f32 -> {code_bytes} bytes of code ({:.0}x smaller)",
        f32_bytes as f32 / code_bytes as f32,
    );

    // Score a raw f32 query against every stored code (asymmetric distance).
    let query = [0.11_f32, 0.21, 0.29, 0.39, 0.51, 0.61, 0.69, 0.81];
    println!("\nquery scored against each code (Cosine, smaller is nearer):");
    for (i, code) in codes.iter().enumerate() {
        let d = sq.distance(&query, code, DistanceMetric::Cosine)?;
        println!("  vector {i}: {d:.4}");
    }

    // Decoding is lossy but close — useful for a full-precision rerank step.
    let approx = sq.dequantize(&codes[0])?;
    println!("\noriginal[0] = {:?}", corpus[0]);
    println!("decoded[0]  = {approx:.3?}");

    Ok(())
}
