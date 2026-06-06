//! Product quantization (PQ): the high-compression scheme. Each vector splits
//! into `M` subvectors, each subvector gets a learned `K`-centroid codebook,
//! and a vector compresses to `M` bytes. Distance uses asymmetric distance
//! computation (ADC): build a per-query lookup table once, then score many
//! codes against it cheaply — exactly what IVF-PQ does inside each cluster.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example product_quantization
//! ```

use iqdb_quantize::{ProductQuantizer, Quantizer};
use iqdb_types::{DistanceMetric, IqdbError};

fn main() -> Result<(), IqdbError> {
    // A 16-dimensional corpus. With M = 4 each vector becomes 4 bytes.
    let corpus: Vec<Vec<f32>> = (0..32)
        .map(|i| {
            let f = i as f32 * 0.1;
            (0..16).map(|d| f + d as f32 * 0.05).collect()
        })
        .collect();
    let refs: Vec<&[f32]> = corpus.iter().map(Vec::as_slice).collect();

    // M = 4 subvectors, K = 16 centroids each, fixed seed for reproducibility.
    let mut pq = ProductQuantizer::with_config(4, 16, 1234);
    pq.train(&refs)?;
    println!(
        "trained PQ: M = {}, K = {}, dim = {:?}",
        pq.n_subvectors(),
        pq.n_centroids(),
        pq.dim(),
    );

    let codes = refs
        .iter()
        .map(|v| pq.quantize(v))
        .collect::<Result<Vec<_>, _>>()?;

    let f32_bytes = size_of_val(refs[0]);
    let code_bytes = codes[0].len();
    println!(
        "each vector: {f32_bytes} bytes of f32 -> {code_bytes} bytes of code ({:.0}x smaller)",
        f32_bytes as f32 / code_bytes as f32,
    );

    // Build the ADC table ONCE for this query, then score every code with it.
    let query: Vec<f32> = (0..16).map(|d| 0.5 + d as f32 * 0.05).collect();
    let tables = pq.build_query_tables(&query, DistanceMetric::Euclidean)?;

    let mut scored: Vec<(usize, f32)> = codes
        .iter()
        .enumerate()
        .map(|(i, c)| tables.distance(c).map(|d| (i, d)))
        .collect::<Result<_, _>>()?;
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));

    println!("\ntop 5 nearest by batch ADC (Euclidean):");
    for (rank, (i, d)) in scored.iter().take(5).enumerate() {
        println!("  #{}: vector {i} at distance {d:.4}", rank + 1);
    }

    // The single-shot `distance` is the same value as the batch path.
    let (best, best_d) = scored[0];
    let single = pq.distance(&query, &codes[best], DistanceMetric::Euclidean)?;
    println!(
        "\nbatch ADC and single-shot distance agree: {}",
        best_d.to_bits() == single.to_bits()
    );

    Ok(())
}
