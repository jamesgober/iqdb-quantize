//! The space/quality dial: the three schemes side by side on the same 768-dim
//! vector, showing how many bytes each code occupies and the compression ratio
//! over the raw `f32` representation.
//!
//! - SQ8 — one byte per dimension (~4x), every metric, best recall.
//! - PQ  — `M` bytes per vector (here 16, so ~192x), Euclidean / DotProduct / Manhattan.
//! - BQ  — one bit per dimension (~32x), Hamming only, highest compression.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example compression
//! ```

use iqdb_quantize::{BinaryQuantizer, ProductQuantizer, Quantizer, ScalarQuantizer};
use iqdb_types::IqdbError;

const DIM: usize = 768;

fn main() -> Result<(), IqdbError> {
    // A small deterministic corpus of 768-dim vectors to train on.
    let corpus: Vec<Vec<f32>> = (0..128)
        .map(|i| {
            (0..DIM)
                .map(|d| (((i * 31 + d * 17) % 251) as f32 / 251.0) - 0.5)
                .collect()
        })
        .collect();
    let refs: Vec<&[f32]> = corpus.iter().map(Vec::as_slice).collect();
    let sample = &corpus[0];

    let raw_bytes = DIM * size_of::<f32>();
    println!("raw f32 vector: {DIM} dims = {raw_bytes} bytes\n");
    println!("{:<8} {:>8} {:>14}", "scheme", "bytes", "compression");
    println!("{:-<32}", "");

    // SQ8 — one u8 per dimension.
    let mut sq = ScalarQuantizer::new();
    sq.train(&refs)?;
    let sq_bytes = sq.quantize(sample)?.len();
    report("SQ8", sq_bytes, raw_bytes);

    // PQ — M bytes per vector (M = 16 here).
    let mut pq = ProductQuantizer::with_config(16, 64, 7);
    pq.train(&refs)?;
    let pq_bytes = pq.quantize(sample)?.len();
    report("PQ(16)", pq_bytes, raw_bytes);

    // BQ — one bit per dimension, packed into u64 words.
    let mut bq = BinaryQuantizer::new();
    bq.train(&refs)?;
    let bq_code = bq.quantize(sample)?;
    let bq_bytes = size_of_val(bq_code.as_words());
    report("BQ", bq_bytes, raw_bytes);

    Ok(())
}

fn report(name: &str, bytes: usize, raw_bytes: usize) {
    let ratio = raw_bytes as f32 / bytes as f32;
    println!("{name:<8} {bytes:>8} {ratio:>13.0}x");
}
