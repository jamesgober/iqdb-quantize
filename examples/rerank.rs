//! The recommended quality path: **search quantized, then rerank with full
//! `f32`.** Quantized distance narrows a large corpus to a cheap shortlist; the
//! final ordering uses the original vectors via `iqdb-distance`. Skipping the
//! rerank is the most common cause of "quantization broke recall" reports.
//!
//! This example builds a shortlist with SQ8 asymmetric distance, then reranks it
//! with exact `f32` Euclidean — and shows the reranked top-`k` matches the exact
//! brute-force answer even though the shortlist was built from lossy codes.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example rerank
//! ```

use iqdb_distance::compute;
use iqdb_quantize::{Quantizer, ScalarQuantizer};
use iqdb_types::{DistanceMetric, IqdbError};

const DIM: usize = 32;
const SHORTLIST: usize = 16;
const TOP_K: usize = 5;

fn main() -> Result<(), IqdbError> {
    // A deterministic 256-vector corpus.
    let corpus: Vec<Vec<f32>> = (0..256)
        .map(|i| {
            (0..DIM)
                .map(|d| ((i * 7 + d * 13) % 97) as f32 / 97.0)
                .collect()
        })
        .collect();
    let refs: Vec<&[f32]> = corpus.iter().map(Vec::as_slice).collect();

    let mut sq = ScalarQuantizer::new();
    sq.train(&refs)?;
    let codes = refs
        .iter()
        .map(|v| sq.quantize(v))
        .collect::<Result<Vec<_>, _>>()?;

    let query: Vec<f32> = (0..DIM).map(|d| (d as f32 * 0.03).fract()).collect();

    // Stage 1 — cheap shortlist from the compressed codes.
    let mut approx: Vec<(usize, f32)> = codes
        .iter()
        .enumerate()
        .map(|(i, c)| {
            sq.distance(&query, c, DistanceMetric::Euclidean)
                .map(|d| (i, d))
        })
        .collect::<Result<_, _>>()?;
    approx.sort_by(|a, b| a.1.total_cmp(&b.1));
    let shortlist: Vec<usize> = approx.iter().take(SHORTLIST).map(|&(i, _)| i).collect();

    // Stage 2 — rerank the shortlist with exact f32 distance.
    let mut reranked: Vec<(usize, f32)> = shortlist
        .iter()
        .map(|&i| compute(DistanceMetric::Euclidean, &query, &corpus[i]).map(|d| (i, d)))
        .collect::<Result<_, _>>()?;
    reranked.sort_by(|a, b| a.1.total_cmp(&b.1));

    // Exact brute-force baseline over the whole corpus.
    let mut exact: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| compute(DistanceMetric::Euclidean, &query, v).map(|d| (i, d)))
        .collect::<Result<_, _>>()?;
    exact.sort_by(|a, b| a.1.total_cmp(&b.1));

    println!("reranked top-{TOP_K} vs exact top-{TOP_K}:");
    for rank in 0..TOP_K {
        let (ri, rd) = reranked[rank];
        let (ei, ed) = exact[rank];
        let mark = if ri == ei { "✓" } else { " " };
        println!(
            "  #{}  rerank: vec {ri:>3} ({rd:.4})   exact: vec {ei:>3} ({ed:.4})  {mark}",
            rank + 1
        );
    }

    let hits = (0..TOP_K).filter(|&r| reranked[r].0 == exact[r].0).count();
    println!("\n{hits}/{TOP_K} positions match the exact answer after rerank");

    Ok(())
}
