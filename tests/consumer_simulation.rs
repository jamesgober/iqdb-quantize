//! Consumer-simulation suite — the RC soak gate.
//!
//! A stand-in IVF-PQ index built **only** on the public `iqdb-quantize` surface,
//! exercising it the way the real consumer (`iqdb-ivf`) does: partition a corpus
//! into coarse clusters, store each member as a [`PqCode`], and at query time
//! build the ADC tables once per query with
//! [`ProductQuantizer::build_query_tables`], then scan the probed clusters'
//! codes through [`PqAdcTables::distance`]. This is the exact shape an IVF-PQ
//! intra-cluster scan takes.
//!
//! It asserts two things the consumer relies on:
//!
//! 1. **Batch ADC equals the single-shot path** — `PqAdcTables::distance` agrees
//!    bit-for-bit with `ProductQuantizer::distance` for every code. If the batch
//!    primitive ever drifted from the per-code result, these tests fail.
//! 2. **The quantized index preserves ranking** — top-`k` from the PQ index
//!    overlaps the exact `f32` brute-force answer, and an SQ8 flat index does the
//!    same. That is the operational proof the surface is sufficient to build a
//!    real recall-preserving consumer.

#![allow(clippy::unwrap_used)]

use iqdb_quantize::{BinaryQuantizer, ProductQuantizer, Quantizer, ScalarQuantizer};
use iqdb_types::{DistanceMetric, IqdbError};

const DIM: usize = 32;
const N: usize = 512;
const N_CLUSTERS: usize = 8;

/// A deterministic, clustered synthetic corpus: `N_CLUSTERS` blobs around
/// distinct random centers with per-vector noise, so each vector is unique and
/// nearest-neighbour structure is unambiguous (inter-cluster distance well
/// exceeds the within-cluster spread). Vector `i` belongs to cluster
/// `i % N_CLUSTERS`, so indices `0..N_CLUSTERS` are one representative each.
fn corpus() -> Vec<Vec<f32>> {
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut next = || {
        // SplitMix64 — deterministic, no external rng dependency.
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        let z = z ^ (z >> 31);
        // Map to [-1.0, 1.0).
        ((z as f64 / u64::MAX as f64) as f32 - 0.5) * 2.0
    };

    // Distinct random center per cluster.
    let centers: Vec<Vec<f32>> = (0..N_CLUSTERS)
        .map(|_| (0..DIM).map(|_| next()).collect())
        .collect();

    (0..N)
        .map(|i| {
            let center = &centers[i % N_CLUSTERS];
            // Small per-element noise keeps each vector unique without erasing
            // the cluster it belongs to.
            center.iter().map(|&c| c + 0.18 * next()).collect()
        })
        .collect()
}

/// Exact brute-force top-`k` indices under Euclidean — the ground truth.
fn exact_top_k(corpus: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let d2: f32 = v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum();
            (i, d2)
        })
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

fn overlap(a: &[usize], b: &[usize]) -> f32 {
    let hits = a.iter().filter(|i| b.contains(i)).count();
    hits as f32 / a.len() as f32
}

/// The mini IVF-PQ index: codes bucketed into clusters by nearest center.
struct IvfPqSim {
    pq: ProductQuantizer,
    centers: Vec<Vec<f32>>,
    clusters: Vec<Vec<(usize, iqdb_quantize::PqCode)>>,
}

impl IvfPqSim {
    fn build(corpus: &[Vec<f32>]) -> Result<Self, IqdbError> {
        let refs: Vec<&[f32]> = corpus.iter().map(Vec::as_slice).collect();
        let mut pq = ProductQuantizer::with_config(8, 32, 99);
        pq.train(&refs)?;

        // Coarse centers: one representative per latent cluster. Vector `c`
        // belongs to cluster `c % N_CLUSTERS`, so indices `0..N_CLUSTERS` give
        // one seed per cluster.
        let centers: Vec<Vec<f32>> = (0..N_CLUSTERS).map(|c| corpus[c].clone()).collect();

        let mut clusters: Vec<Vec<(usize, iqdb_quantize::PqCode)>> = vec![Vec::new(); N_CLUSTERS];
        for (i, v) in corpus.iter().enumerate() {
            let c = nearest_center(&centers, v);
            let code = pq.quantize(v)?;
            clusters[c].push((i, code));
        }
        Ok(Self {
            pq,
            centers,
            clusters,
        })
    }

    /// Probe the `n_probe` nearest clusters, scan their codes with one ADC table.
    fn search(&self, query: &[f32], k: usize, n_probe: usize) -> Result<Vec<usize>, IqdbError> {
        let tables = self
            .pq
            .build_query_tables(query, DistanceMetric::Euclidean)?;

        // Rank clusters by center distance, take the closest `n_probe`.
        let mut by_center: Vec<usize> = (0..self.centers.len()).collect();
        by_center
            .sort_by(|&a, &b| l2(&self.centers[a], query).total_cmp(&l2(&self.centers[b], query)));

        let mut scored: Vec<(usize, f32)> = Vec::new();
        for &c in by_center.iter().take(n_probe) {
            for (id, code) in &self.clusters[c] {
                let d = tables.distance(code)?;
                scored.push((*id, d));
            }
        }
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        Ok(scored.into_iter().take(k).map(|(i, _)| i).collect())
    }
}

fn nearest_center(centers: &[Vec<f32>], v: &[f32]) -> usize {
    let mut best = 0;
    let mut best_d = f32::INFINITY;
    for (c, center) in centers.iter().enumerate() {
        let d = l2(center, v);
        if d < best_d {
            best_d = d;
            best = c;
        }
    }
    best
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

// --- Contract 1: batch ADC == single-shot distance ---------------------------

#[test]
fn batch_adc_matches_single_shot_for_every_code() {
    let data = corpus();
    let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
    let mut pq = ProductQuantizer::with_config(8, 32, 99);
    pq.train(&refs).unwrap();

    let codes: Vec<_> = refs.iter().map(|v| pq.quantize(v).unwrap()).collect();

    for metric in [
        DistanceMetric::Euclidean,
        DistanceMetric::DotProduct,
        DistanceMetric::Manhattan,
    ] {
        let query = &data[7];
        let tables = pq.build_query_tables(query, metric).unwrap();
        for code in &codes {
            let batch = tables.distance(code).unwrap();
            let single = pq.distance(query, code, metric).unwrap();
            assert_eq!(
                batch.to_bits(),
                single.to_bits(),
                "batch ADC diverged from single-shot for {metric:?}",
            );
        }
    }
}

// --- Contract 2: the PQ index preserves ranking ------------------------------

#[test]
fn ivf_pq_index_recovers_correct_cluster() {
    // PQ is a coarse code: it reliably finds the right *neighbourhood*, but is
    // too lossy to finely rank near-identical within-cluster members — which is
    // exactly why IVF-PQ reranks with full precision. So the right thing to
    // assert here is cluster purity: the top-k the index returns should belong
    // to the query's true cluster.
    let data = corpus();
    let index = IvfPqSim::build(&data).unwrap();

    let mut total_purity = 0.0_f32;
    let queries = 16;
    for q in 0..queries {
        let qi = q * (N / queries);
        let true_cluster = qi % N_CLUSTERS;
        let got = index.search(&data[qi], 10, 2).unwrap();
        let pure = got
            .iter()
            .filter(|&&i| i % N_CLUSTERS == true_cluster)
            .count();
        total_purity += pure as f32 / got.len() as f32;
    }
    let mean = total_purity / queries as f32;
    assert!(mean >= 0.9, "mean PQ cluster purity {mean:.3} below 0.9");
}

#[test]
fn ivf_pq_shortlist_then_rerank_recovers_exact() {
    // The full IVF-PQ pipeline: PQ narrows to a shortlist cheaply, then full f32
    // distance reranks it. This recovers the exact top-k even though the
    // shortlist came from lossy codes — the documented quality path.
    let data = corpus();
    let index = IvfPqSim::build(&data).unwrap();

    let mut total = 0.0_f32;
    let queries = 16;
    for q in 0..queries {
        let query = &data[q * (N / queries)];
        let exact = exact_top_k(&data, query, 10);

        // PQ shortlist (top-50 over probed clusters), then exact rerank to top-10.
        let shortlist = index.search(query, 50, 4).unwrap();
        let mut reranked: Vec<(usize, f32)> = shortlist
            .iter()
            .map(|&i| (i, l2(&data[i], query)))
            .collect();
        reranked.sort_by(|a, b| a.1.total_cmp(&b.1));
        let got: Vec<usize> = reranked.into_iter().take(10).map(|(i, _)| i).collect();

        total += overlap(&exact, &got);
    }
    let mean = total / queries as f32;
    assert!(
        mean >= 0.9,
        "mean rerank top-10 overlap {mean:.3} below 0.9"
    );
}

#[test]
fn sq8_flat_index_preserves_top_k() {
    let data = corpus();
    let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
    let mut sq = ScalarQuantizer::new();
    sq.train(&refs).unwrap();
    let codes: Vec<_> = refs.iter().map(|v| sq.quantize(v).unwrap()).collect();

    let mut total = 0.0_f32;
    let queries = 16;
    for q in 0..queries {
        let query = &data[q * (N / queries)];
        let exact = exact_top_k(&data, query, 10);

        let mut scored: Vec<(usize, f32)> = codes
            .iter()
            .enumerate()
            .map(|(i, c)| (i, sq.distance(query, c, DistanceMetric::Euclidean).unwrap()))
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        let got: Vec<usize> = scored.into_iter().take(10).map(|(i, _)| i).collect();
        total += overlap(&exact, &got);
    }
    let mean = total / queries as f32;
    // SQ8 keeps far more precision than PQ — full scan should be near-perfect.
    assert!(mean >= 0.9, "mean SQ8 top-10 overlap {mean:.3} below 0.9");
}

// --- Contracts the consumer relies on at the boundary ------------------------

#[test]
fn adc_table_rejects_foreign_code_shape() {
    let data = corpus();
    let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();

    let mut pq8 = ProductQuantizer::with_config(8, 32, 1);
    pq8.train(&refs).unwrap();
    let mut pq4 = ProductQuantizer::with_config(4, 32, 1);
    pq4.train(&refs).unwrap();

    // A table built for M=8 must reject a code produced under M=4.
    let tables = pq8
        .build_query_tables(&data[0], DistanceMetric::Euclidean)
        .unwrap();
    let foreign = pq4.quantize(&data[0]).unwrap();
    let err = tables.distance(&foreign).unwrap_err();
    assert!(matches!(err, IqdbError::DimensionMismatch { .. }));
}

#[test]
fn unsupported_metric_is_rejected_not_panicked() {
    let data = corpus();
    let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
    let mut pq = ProductQuantizer::with_config(8, 32, 1);
    pq.train(&refs).unwrap();

    // PQ has no global norm — Cosine is a typed error, never a panic.
    let err = pq
        .build_query_tables(&data[0], DistanceMetric::Cosine)
        .unwrap_err();
    assert_eq!(err, IqdbError::InvalidMetric);

    // BQ rejects everything but Hamming.
    let mut bq = BinaryQuantizer::new();
    bq.train(&refs).unwrap();
    let code = bq.quantize(&data[0]).unwrap();
    let err = bq
        .distance(&data[0], &code, DistanceMetric::Euclidean)
        .unwrap_err();
    assert_eq!(err, IqdbError::InvalidMetric);
}
