//! K-means training for PQ subvector codebooks.
//!
//! Hand-rolled k-means with k-means++ seeding and Lloyd's iterations,
//! driven by the seeded [`crate::rng::SplitMix64`] PRNG. The
//! determinism contract is the load-bearing reason every loop here is
//! sequential and every reduction runs in fixed order:
//!
//! > **Determinism.** Given the same `seed`, the same `subvectors`
//! > slice (same pointers, same dimension, same byte content), and
//! > the same `n_centroids`, [`train_codebook`] returns byte-identical
//! > `Vec<Vec<f32>>` centroids on every supported platform. This holds
//! > because (1) all PRNG draws come from the in-tree
//! > [`SplitMix64`](crate::rng::SplitMix64); (2) all reductions run in
//! > a fixed sequential order; (3) centroid sums accumulate in `f64`
//! > and downcast to `f32` once at the end of each iteration,
//! > sidestepping the `f32` catastrophic-cancellation cliff for large
//! > clusters.
//!
//! ## Why this lives in `iqdb-quantize`
//!
//! This is a near-verbatim copy of `iqdb-ivf/src/train.rs` (minus the
//! IVF-specific PRNG subsampling — PQ trains full-batch over each
//! subvector slice for v0.3, per the spec). `iqdb-ivf` cannot be a
//! dependency: it sits above `iqdb-quantize` in the workspace graph
//! and the reverse edge would cycle. A consolidation PR that lifts
//! the k-means core into a shared `iqdb-cluster` crate (used by HNSW,
//! IVF, and PQ alike) is tracked alongside the parallel
//! [`SplitMix64`](crate::rng::SplitMix64) consolidation.
//!
//! ## Algorithm
//!
//! 1. **Pre-checks.** Reject empty samples, wrong-dimension slices,
//!    and `subvectors.len() < n_centroids`. All errors surface as
//!    [`iqdb_types::IqdbError`].
//! 2. **k-means++ seeding.** Pick the first centroid uniformly,
//!    then for each remaining centroid pick a point with probability
//!    proportional to its squared distance from the nearest already-
//!    chosen centroid (the canonical Arthur–Vassilvitskii procedure).
//! 3. **Lloyd's iterations.** Up to [`MAX_ITERS`] passes; each pass
//!    assigns every subvector to its nearest centroid, then
//!    recomputes every centroid as the mean of its assigned points.
//!    Convergence triggers when the maximum relative centroid shift
//!    drops below [`REL_TOL`].
//! 4. **Empty-cluster recovery.** When a Lloyd's pass leaves a
//!    cluster with zero assignments, the centroid is moved to the
//!    sample point that is furthest from any current centroid — a
//!    deterministic recovery that preserves the determinism contract.

use iqdb_types::{IqdbError, Result};

use crate::rng::SplitMix64;

/// Maximum number of Lloyd's iterations.
///
/// Matches `iqdb-ivf`'s `MAX_ITERS = 25`. Enough for centroids to
/// stabilize on every dataset benchmarked in `iqdb-eval`; tuning is
/// a v0.4 knob.
pub(crate) const MAX_ITERS: usize = 25;

/// Convergence threshold on the *relative* centroid shift.
///
/// At the end of each Lloyd's iteration we compute
/// `max over centroids of ||old - new||_2 / max(||old||_2, 1.0)`
/// and stop once it drops below this value. Matches `iqdb-ivf`'s
/// `REL_TOL = 1e-4`.
pub(crate) const REL_TOL: f32 = 1e-4;

/// Train one PQ subvector codebook on `subvectors`.
///
/// Returns `n_centroids` centroids, each a `Vec<f32>` of length
/// `sub_dim`, or an [`IqdbError`] if the pre-checks reject the
/// inputs. See the module-level docs for the determinism contract.
///
/// PQ trains M of these in sequence (one per subvector position),
/// each over the slice `[v[m*sub_dim .. (m+1)*sub_dim] for v in
/// training_set]`. The slices are passed in directly as
/// `&[&[f32]]`; the caller (in [`crate::product`]) builds them.
pub(crate) fn train_codebook(
    sub_dim: usize,
    n_centroids: usize,
    seed: u64,
    subvectors: &[&[f32]],
) -> Result<Vec<Vec<f32>>> {
    // -- Pre-checks ---------------------------------------------------
    if subvectors.is_empty() {
        return Err(IqdbError::InvalidConfig {
            reason: "ProductQuantizer codebook training requires a non-empty sample",
        });
    }
    for v in subvectors {
        if v.len() != sub_dim {
            return Err(IqdbError::DimensionMismatch {
                expected: sub_dim,
                found: v.len(),
            });
        }
    }
    if subvectors.len() < n_centroids {
        return Err(IqdbError::InvalidConfig {
            reason: "ProductQuantizer codebook training requires sample size >= n_centroids",
        });
    }

    let mut rng = SplitMix64::new(seed);

    // PQ trains full-batch for v0.3 — no subsampling. Pass-through.
    let working_set: Vec<&[f32]> = subvectors.to_vec();

    let centroids = kmeans_plus_plus(sub_dim, n_centroids, &working_set, &mut rng);
    let final_centroids = lloyd(centroids, &working_set, sub_dim);
    Ok(final_centroids)
}

/// Compute the squared L2 distance between `a` and `b`.
///
/// Mirrors `iqdb-ivf/src/assign.rs::squared_l2`. Squared L2 is the
/// canonical k-means kernel: the centroid that minimizes the
/// within-cluster sum of distances is the arithmetic mean only under
/// L2. Running in fixed component order makes the reduction
/// reproducible across platforms.
#[must_use]
pub(crate) fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "squared_l2 requires same-dim slices");
    let mut sum: f32 = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// Return the index of the centroid nearest to `vector` under squared L2.
///
/// Ties are broken by **lower index wins** — never `<=`, always `<`
/// — so the assignment is fully deterministic in centroid order.
#[must_use]
pub(crate) fn assign_to_cluster(centroids: &[Vec<f32>], vector: &[f32]) -> usize {
    debug_assert!(
        !centroids.is_empty(),
        "assign_to_cluster needs at least one centroid"
    );
    let mut best_idx: usize = 0;
    let mut best_dist = squared_l2(&centroids[0], vector);
    for (i, c) in centroids.iter().enumerate().skip(1) {
        let d = squared_l2(c, vector);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx
}

/// Pick `n_centroids` initial centroids via the Arthur–Vassilvitskii
/// k-means++ procedure.
fn kmeans_plus_plus(
    dim: usize,
    n_centroids: usize,
    working_set: &[&[f32]],
    rng: &mut SplitMix64,
) -> Vec<Vec<f32>> {
    let n = working_set.len();
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(n_centroids);

    // 1. First centroid: uniformly random.
    let first_idx = rng.next_below(n as u64) as usize;
    centroids.push(working_set[first_idx].to_vec());

    // `min_sq[i]` = squared L2 distance from `working_set[i]` to its
    // nearest already-chosen centroid. Initialized from centroid 0.
    let mut min_sq: Vec<f32> = working_set
        .iter()
        .map(|v| squared_l2(working_set[first_idx], v))
        .collect();

    // 2. Remaining centroids: weighted by `min_sq`.
    for _ in 1..n_centroids {
        let mut total: f64 = 0.0;
        for &w in &min_sq {
            total += w as f64;
        }
        let next_idx = if total <= 0.0 {
            // All sampled points already coincide with a centroid.
            // Fall back to a uniform pick.
            rng.next_below(n as u64) as usize
        } else {
            // Inverse-CDF draw against the `min_sq` distribution.
            let target = rng.next_open_unit() * total;
            let mut running: f64 = 0.0;
            let mut chosen: usize = n - 1;
            for (i, &w) in min_sq.iter().enumerate() {
                running += w as f64;
                if running >= target {
                    chosen = i;
                    break;
                }
            }
            chosen
        };
        centroids.push(working_set[next_idx].to_vec());

        // Refresh `min_sq` against the newly chosen centroid only.
        let new_centroid = &centroids[centroids.len() - 1];
        for (i, v) in working_set.iter().enumerate() {
            let d = squared_l2(new_centroid, v);
            if d < min_sq[i] {
                min_sq[i] = d;
            }
        }
    }

    debug_assert_eq!(centroids.len(), n_centroids);
    debug_assert!(centroids.iter().all(|c| c.len() == dim));
    let _ = dim;
    centroids
}

/// Run Lloyd's iterations on `centroids` until convergence or
/// [`MAX_ITERS`].
fn lloyd(mut centroids: Vec<Vec<f32>>, working_set: &[&[f32]], dim: usize) -> Vec<Vec<f32>> {
    let n_clusters = centroids.len();
    let n = working_set.len();

    // Reusable buffers, allocated once outside the iteration loop.
    let mut sums: Vec<Vec<f64>> = vec![vec![0.0_f64; dim]; n_clusters];
    let mut counts: Vec<usize> = vec![0_usize; n_clusters];
    let mut assignments: Vec<usize> = vec![0_usize; n];

    for _iter in 0..MAX_ITERS {
        for s in sums.iter_mut() {
            for v in s.iter_mut() {
                *v = 0.0;
            }
        }
        for c in counts.iter_mut() {
            *c = 0;
        }

        // -- Assignment pass ---------------------------------------
        for (i, v) in working_set.iter().enumerate() {
            let c = assign_to_cluster(&centroids, v);
            assignments[i] = c;
            let s = &mut sums[c];
            for (k, &x) in v.iter().enumerate() {
                s[k] += x as f64;
            }
            counts[c] += 1;
        }

        // -- Update pass + max-shift tracking ----------------------
        let mut max_rel_shift: f32 = 0.0;
        for c in 0..n_clusters {
            let count = counts[c];
            if count == 0 {
                // Empty cluster — deterministic recovery: the sample
                // point with the largest min-distance to any current
                // centroid, scanning in fixed order so ties go to the
                // lowest sample index.
                let mut best_idx: usize = 0;
                let mut best_dist: f32 = -1.0;
                for (i, v) in working_set.iter().enumerate() {
                    let mut nearest: f32 = squared_l2(&centroids[0], v);
                    for cc in centroids.iter().skip(1) {
                        let d = squared_l2(cc, v);
                        if d < nearest {
                            nearest = d;
                        }
                    }
                    if nearest > best_dist {
                        best_dist = nearest;
                        best_idx = i;
                    }
                }
                let new_centroid = working_set[best_idx].to_vec();
                let shift = relative_shift(&centroids[c], &new_centroid);
                if shift > max_rel_shift {
                    max_rel_shift = shift;
                }
                centroids[c] = new_centroid;
                continue;
            }

            let inv = 1.0_f64 / (count as f64);
            let new_centroid: Vec<f32> = sums[c].iter().map(|&s| (s * inv) as f32).collect();
            let shift = relative_shift(&centroids[c], &new_centroid);
            if shift > max_rel_shift {
                max_rel_shift = shift;
            }
            centroids[c] = new_centroid;
        }

        if max_rel_shift < REL_TOL {
            break;
        }
    }

    let _ = assignments;
    centroids
}

/// `||old - new||_2 / max(||old||_2, 1.0)`. Used as the Lloyd
/// convergence criterion.
fn relative_shift(old: &[f32], new: &[f32]) -> f32 {
    debug_assert_eq!(old.len(), new.len());
    let mut diff_sq: f32 = 0.0;
    let mut old_norm_sq: f32 = 0.0;
    for i in 0..old.len() {
        let d = old[i] - new[i];
        diff_sq += d * d;
        old_norm_sq += old[i] * old[i];
    }
    let diff = diff_sq.sqrt();
    let denom = old_norm_sq.sqrt().max(1.0);
    diff / denom
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn refs(slice: &[Vec<f32>]) -> Vec<&[f32]> {
        slice.iter().map(|v| v.as_slice()).collect()
    }

    #[test]
    fn rejects_empty_sample() {
        let err = train_codebook(2, 3, 0, &[]).unwrap_err();
        match err {
            IqdbError::InvalidConfig { reason } => {
                assert!(reason.contains("non-empty sample"));
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn rejects_sample_smaller_than_n_centroids() {
        let data = vec![vec![0.0_f32, 0.0], vec![1.0, 1.0]];
        let sample = refs(&data);
        let err = train_codebook(2, 5, 0, &sample).unwrap_err();
        match err {
            IqdbError::InvalidConfig { reason } => {
                assert!(reason.contains("sample size >= n_centroids"));
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn rejects_dim_mismatch() {
        let bad = vec![1.0_f32, 2.0, 3.0];
        let sample = vec![bad.as_slice()];
        let err = train_codebook(2, 1, 0, &sample).unwrap_err();
        match err {
            IqdbError::DimensionMismatch { expected, found } => {
                assert_eq!(expected, 2);
                assert_eq!(found, 3);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn converges_on_two_obvious_clusters() {
        let data: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![0.1, -0.1],
            vec![-0.05, 0.05],
            vec![10.0, 10.0],
            vec![10.1, 9.9],
            vec![9.95, 10.05],
        ];
        let sample = refs(&data);
        let centroids = train_codebook(2, 2, 1, &sample).unwrap();
        assert_eq!(centroids.len(), 2);
        let mut near_origin = 0;
        let mut near_ten = 0;
        for c in &centroids {
            if c[0].abs() < 1.0 && c[1].abs() < 1.0 {
                near_origin += 1;
            }
            if (c[0] - 10.0).abs() < 1.0 && (c[1] - 10.0).abs() < 1.0 {
                near_ten += 1;
            }
        }
        assert_eq!(near_origin, 1);
        assert_eq!(near_ten, 1);
    }

    #[test]
    fn same_seed_produces_identical_centroids() {
        let data: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![(i as f32) * 0.1, ((i * 3) as f32) * 0.07])
            .collect();
        let sample = refs(&data);
        let a = train_codebook(2, 4, 1234, &sample).unwrap();
        let b = train_codebook(2, 4, 1234, &sample).unwrap();
        assert_eq!(a, b, "same seed + same data → identical centroids");
    }

    #[test]
    fn different_seeds_can_diverge() {
        let data: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![(i as f32) * 0.1, ((i * 3) as f32) * 0.07])
            .collect();
        let sample = refs(&data);
        let a = train_codebook(2, 4, 1, &sample).unwrap();
        let b = train_codebook(2, 4, 2, &sample).unwrap();
        assert_ne!(a, b);
    }
}
