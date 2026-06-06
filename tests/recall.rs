//! Recall integration tests on a seeded Gaussian-cluster corpus.
//!
//! Two related but distinct recall guarantees, one per quantizer, because
//! the two schemes preserve different signal:
//!
//! - **SQ8** retains per-dimension magnitude, so its asymmetric distance
//!   ordering matches the full-`f32` cosine ordering within a cluster
//!   almost exactly. The SQ8 test asserts top-_k_ index overlap with the
//!   full-`f32` baseline above a documented threshold.
//! - **BQ** discards magnitude entirely (one sign bit per dimension), so
//!   it cannot rank tightly grouped same-cluster vectors against each
//!   other — within-cluster Hamming distances tie or cluster narrowly.
//!   What BQ does preserve is **cluster identity**: a query from cluster
//!   0 still gets cluster-0 candidates as its nearest BQ neighbours.
//!   That is the real "similarity-search quality" guarantee BQ offers,
//!   and the BQ test measures it directly — the fraction of BQ's top-_k_
//!   that comes from the same cluster as the query.
//!
//! Thresholds are taken from the **measured** values on this corpus with
//! documented margin. See `.dev/ROADMAP.md` for the calibration policy.

#![allow(clippy::unwrap_used)]

use iqdb_distance::compute;
use iqdb_quantize::{BinaryQuantizer, ProductQuantizer, Quantizer, ScalarQuantizer};
use iqdb_types::DistanceMetric;

const DIM: usize = 128;
const N_CLUSTERS: usize = 8;
/// 20 vectors per cluster sizes each cluster comfortably above `K` so
/// the top-`K` slice fits inside cluster 0 even with a few f32-baseline
/// boundary misses, without inflating runtime.
const PER_CLUSTER: usize = 20;
const K: usize = 10;
const SEED: u64 = 0xA1B2_C3D4_E5F6_0789;

/// SQ8 measured top-K index overlap with the full-`f32` baseline on the
/// seeded Gaussian-cluster corpus is well above this value; the margin
/// guards against minor numeric drift across platforms.
const SQ8_MIN_OVERLAP: f32 = 0.9;
/// BQ measured cluster purity (fraction of top-K hits from the same
/// cluster as the query) on the same corpus is at or near 1.0; this
/// threshold leaves margin for a single boundary candidate from a
/// neighbouring cluster sneaking in.
const BQ_MIN_CLUSTER_PURITY: f32 = 0.7;
/// PQ Euclidean top-K overlap with the full-`f32` Euclidean baseline on
/// the seeded Gaussian-cluster corpus. PQ is the most aggressive of the
/// three schemes (`M` bytes per vector, here 8 bytes per 128-dim `f32`
/// vector = 64× compression) so its within-cluster ranking is looser
/// than SQ8's; what PQ reliably preserves is **cluster identity**
/// (every top-K hit ends up in the query's cluster), and the
/// within-cluster index overlap above is what's left after that.
/// Threshold measured on this corpus and held with margin.
const PQ_MIN_OVERLAP: f32 = 0.6;
/// PQ shape used by the recall test: M = 8 subvectors of 16 components
/// each (DIM = 128), K = 32 centroids per subvector. With
/// `PER_CLUSTER * N_CLUSTERS = 160` training vectors per subvector
/// position, each codebook trains on ~5× its centroid count — enough
/// to converge meaningfully without overfitting.
const PQ_M: usize = 8;
const PQ_K: usize = 32;
const PQ_SEED: u64 = 0x0F0E_0D0C_0B0A_0908;

/// Tiny LCG so the test is deterministic without pulling in `rand`.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    /// Uniform `[0, 1)`.
    fn next_unit(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) / ((1u32 << 24) as f32)
    }
    /// Approximate standard normal via the Box–Muller transform.
    fn next_normal(&mut self) -> f32 {
        let u1 = self.next_unit().max(1e-9);
        let u2 = self.next_unit();
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = 2.0_f32 * std::f32::consts::PI * u2;
        r * theta.cos()
    }
}

/// Build a seeded Gaussian-cluster corpus: `N_CLUSTERS` centres with
/// per-component magnitude ±1.0, plus per-component normal spread of
/// `1.0`. Centre magnitude and spread are balanced so that (a) clusters
/// remain separable under cosine — same-cluster Hamming distances cluster
/// near `DIM * 0.27`, different-cluster near `DIM * 0.73` — and (b) the
/// intra-cluster Bernoulli-bit variance is large enough that BQ's ranking
/// within a cluster carries real signal rather than collapsing into a
/// flat tie pool.
fn build_corpus() -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut rng = Lcg::new(SEED);

    let mut centres: Vec<Vec<f32>> = Vec::with_capacity(N_CLUSTERS);
    for _ in 0..N_CLUSTERS {
        let mut c = Vec::with_capacity(DIM);
        for _ in 0..DIM {
            // Use a high bit of the LCG output — the low bit of a
            // Numerical-Recipes-style LCG flips every step, so every
            // centre would otherwise collapse to the same alternating
            // pattern and there would be no real clusters.
            let sign = if rng.next_u64() >> 63 == 0 { -1.0 } else { 1.0 };
            c.push(sign);
        }
        centres.push(c);
    }

    let mut corpus = Vec::with_capacity(N_CLUSTERS * PER_CLUSTER);
    for c in &centres {
        for _ in 0..PER_CLUSTER {
            let v: Vec<f32> = c.iter().map(|x| x + rng.next_normal()).collect();
            corpus.push(v);
        }
    }

    // Query is the first cluster's centre plus mild noise so the f32 and
    // BQ rankings both lean toward cluster 0 without being identical.
    let query: Vec<f32> = centres[0]
        .iter()
        .map(|x| x + 0.3 * rng.next_normal())
        .collect();

    (corpus, query)
}

/// Top-K indices under cosine distance via the `iqdb-distance` baseline.
fn top_k_f32(query: &[f32], corpus: &[Vec<f32>]) -> Vec<usize> {
    let mut distances: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, compute(DistanceMetric::Cosine, query, v).unwrap()))
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    distances.into_iter().take(K).map(|(i, _)| i).collect()
}

/// Top-K indices under Euclidean distance via the `iqdb-distance` baseline.
/// Used as the reference for the PQ test, since PQ's natural metric is
/// Euclidean (Cosine is not supported).
fn top_k_f32_euclidean(query: &[f32], corpus: &[Vec<f32>]) -> Vec<usize> {
    let mut distances: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, compute(DistanceMetric::Euclidean, query, v).unwrap()))
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    distances.into_iter().take(K).map(|(i, _)| i).collect()
}

/// Overlap ratio between two K-element index sets.
fn overlap(a: &[usize], b: &[usize]) -> f32 {
    let mut hits = 0;
    for x in a {
        if b.contains(x) {
            hits += 1;
        }
    }
    hits as f32 / a.len() as f32
}

#[test]
fn sq8_top_k_overlap_with_f32_baseline_meets_threshold() {
    let (corpus, query) = build_corpus();

    let refs: Vec<&[f32]> = corpus.iter().map(Vec::as_slice).collect();
    let mut sq = ScalarQuantizer::new();
    sq.train(&refs).unwrap();
    let codes: Vec<_> = corpus.iter().map(|v| sq.quantize(v).unwrap()).collect();

    let mut quantized: Vec<(usize, f32)> = codes
        .iter()
        .enumerate()
        .map(|(i, c)| (i, sq.distance(&query, c, DistanceMetric::Cosine).unwrap()))
        .collect();
    quantized.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let sq8_top: Vec<usize> = quantized.into_iter().take(K).map(|(i, _)| i).collect();

    let baseline = top_k_f32(&query, &corpus);
    let r = overlap(&sq8_top, &baseline);
    assert!(
        r >= SQ8_MIN_OVERLAP,
        "SQ8 top-{K} overlap {r:.3} below threshold {SQ8_MIN_OVERLAP:.3} \
         (f32 baseline top-{K} clusters {baseline_clusters:?}, \
          sq8 top-{K} clusters {sq8_clusters:?})",
        baseline_clusters = baseline.iter().map(|&i| cluster_of(i)).collect::<Vec<_>>(),
        sq8_clusters = sq8_top.iter().map(|&i| cluster_of(i)).collect::<Vec<_>>(),
    );
}

/// The cluster the corpus index `i` belongs to, given the build pattern
/// (cluster 0 fills indices `0..PER_CLUSTER`, cluster 1 fills the next
/// block, and so on).
fn cluster_of(i: usize) -> usize {
    i / PER_CLUSTER
}

#[test]
fn bq_top_k_preserves_query_cluster() {
    let (corpus, query) = build_corpus();

    let refs: Vec<&[f32]> = corpus.iter().map(Vec::as_slice).collect();
    let mut bq = BinaryQuantizer::new();
    bq.train(&refs).unwrap();
    let codes: Vec<_> = corpus.iter().map(|v| bq.quantize(v).unwrap()).collect();

    let mut quantized: Vec<(usize, f32)> = codes
        .iter()
        .enumerate()
        .map(|(i, c)| (i, bq.distance(&query, c, DistanceMetric::Hamming).unwrap()))
        .collect();
    quantized.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let bq_top: Vec<usize> = quantized.into_iter().take(K).map(|(i, _)| i).collect();

    // BQ's natural recall signal is cluster identity, not within-cluster
    // index ordering — see the module docs. The query is built from
    // cluster 0, so we measure the fraction of BQ's top-K that also came
    // from cluster 0.
    let query_cluster = 0;
    let hits = bq_top
        .iter()
        .filter(|&&i| cluster_of(i) == query_cluster)
        .count();
    let purity = hits as f32 / K as f32;
    let baseline = top_k_f32(&query, &corpus);
    assert!(
        purity >= BQ_MIN_CLUSTER_PURITY,
        "BQ top-{K} cluster purity {purity:.3} below threshold {BQ_MIN_CLUSTER_PURITY:.3} \
         (bq top clusters {bq_clusters:?}, f32 baseline clusters {baseline_clusters:?})",
        bq_clusters = bq_top.iter().map(|&i| cluster_of(i)).collect::<Vec<_>>(),
        baseline_clusters = baseline.iter().map(|&i| cluster_of(i)).collect::<Vec<_>>(),
    );
}

#[test]
fn pq_top_k_overlap_with_f32_baseline_meets_threshold() {
    let (corpus, query) = build_corpus();

    let refs: Vec<&[f32]> = corpus.iter().map(Vec::as_slice).collect();
    let mut pq = ProductQuantizer::with_config(PQ_M, PQ_K, PQ_SEED);
    pq.train(&refs).unwrap();
    let codes: Vec<_> = corpus.iter().map(|v| pq.quantize(v).unwrap()).collect();

    let mut quantized: Vec<(usize, f32)> = codes
        .iter()
        .enumerate()
        .map(|(i, c)| {
            (
                i,
                pq.distance(&query, c, DistanceMetric::Euclidean).unwrap(),
            )
        })
        .collect();
    quantized.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let pq_top: Vec<usize> = quantized.into_iter().take(K).map(|(i, _)| i).collect();

    let baseline = top_k_f32_euclidean(&query, &corpus);
    let r = overlap(&pq_top, &baseline);
    assert!(
        r >= PQ_MIN_OVERLAP,
        "PQ top-{K} overlap {r:.3} below threshold {PQ_MIN_OVERLAP:.3} \
         (f32 baseline top-{K} clusters {baseline_clusters:?}, \
          pq top-{K} clusters {pq_clusters:?})",
        baseline_clusters = baseline.iter().map(|&i| cluster_of(i)).collect::<Vec<_>>(),
        pq_clusters = pq_top.iter().map(|&i| cluster_of(i)).collect::<Vec<_>>(),
    );
}
