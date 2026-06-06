//! A small seeded deterministic PRNG.
//!
//! [`SplitMix64`] is the standard 64-bit `SplitMix` generator from Vigna,
//! used here as the seed source for `ProductQuantizer`'s k-means++
//! initialization and for deterministic subsampling of subvector
//! training slices. It is intentionally hand-rolled and dependency-free
//! so the determinism contract — same `seed` + same training sample →
//! identical centroids — does not depend on an external crate's version
//! drift.
//!
//! This is a verbatim copy of `iqdb-ivf/src/rng.rs` (which is itself a
//! verbatim copy of `iqdb-hnsw/src/rng.rs`). Keeping a local copy means
//! `iqdb-quantize` does not depend on `iqdb-ivf` or `iqdb-hnsw` — both
//! of which sit above `iqdb-quantize` in the dependency graph and would
//! cycle. A consolidation PR that lifts `SplitMix64` into a shared
//! utility crate (e.g. `iqdb-rand`) is tracked alongside the parallel
//! k-means consolidation in [`crate::train`]; see the workspace
//! `.dev/ROADMAP.md`.
//!
//! The generator is not cryptographically secure and is not intended to
//! be. It is fast, has good enough distribution for the small floating
//! draws k-means++ makes, and produces the same byte sequence on every
//! platform.

/// A deterministic 64-bit PRNG seeded once at construction.
///
/// State is a single `u64` updated by Vigna's SplitMix64 mix function
/// on each [`next_u64`](SplitMix64::next_u64) call. Cloning the
/// generator copies the state — two clones at the same point produce
/// identical subsequent draws.
#[derive(Debug, Clone)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Build a generator seeded with `seed`.
    ///
    /// A `seed` of `0` is fine — the SplitMix mix function does not
    /// degenerate on a zero state.
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance the generator and return the next `u64`.
    ///
    /// The mix constants are from Vigna's reference implementation.
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Sample an `f64` in `(0, 1]`.
    ///
    /// The k-means++ weighted draw is `u * total_weight` where
    /// `total_weight > 0`; `u = 0` would always pick the very first
    /// candidate even when its weight is tiny, biasing seeding. We
    /// re-roll if the top-53-bit mantissa draw is zero, and fall back
    /// to `2^-53` after a bounded number of attempts so termination is
    /// provable from the source.
    pub(crate) fn next_open_unit(&mut self) -> f64 {
        for _ in 0..4 {
            let bits = self.next_u64() >> 11;
            if bits != 0 {
                return (bits as f64) * (1.0 / ((1_u64 << 53) as f64));
            }
        }
        1.0 / ((1_u64 << 53) as f64)
    }

    /// Uniform integer draw in `0..n`.
    ///
    /// Implemented as the standard rejection-on-bias loop so the
    /// distribution is exact even when `n` is not a power of two.
    /// Callers MUST ensure `n >= 1`; the only call sites are inside
    /// [`crate::train`] where that has already been validated.
    pub(crate) fn next_below(&mut self, n: u64) -> u64 {
        debug_assert!(n >= 1, "next_below requires n >= 1");
        let zone = u64::MAX - (u64::MAX % n);
        loop {
            let v = self.next_u64();
            if v < zone {
                return v % n;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn same_seed_produces_same_sequence() {
        let mut a = SplitMix64::new(0xCAFE_F00D);
        let mut b = SplitMix64::new(0xCAFE_F00D);
        for _ in 0..1024 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(2);
        let mut equal = 0;
        for _ in 0..1024 {
            if a.next_u64() == b.next_u64() {
                equal += 1;
            }
        }
        assert!(equal < 4, "too much agreement: {equal}/1024");
    }

    #[test]
    fn open_unit_is_in_unit_interval_exclusive_zero() {
        let mut rng = SplitMix64::new(7);
        for _ in 0..10_000 {
            let u = rng.next_open_unit();
            assert!(u > 0.0 && u <= 1.0, "u = {u}");
        }
    }

    #[test]
    fn open_unit_never_returns_zero_from_zero_seed() {
        let mut rng = SplitMix64::new(0);
        for _ in 0..10_000 {
            assert!(rng.next_open_unit() > 0.0);
        }
    }

    #[test]
    fn next_below_stays_in_range() {
        let mut rng = SplitMix64::new(42);
        for _ in 0..10_000 {
            let v = rng.next_below(7);
            assert!(v < 7);
        }
    }

    #[test]
    fn next_below_one_always_returns_zero() {
        let mut rng = SplitMix64::new(0);
        for _ in 0..1_000 {
            assert_eq!(rng.next_below(1), 0);
        }
    }
}
