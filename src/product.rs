//! [`ProductQuantizer`] — product quantization (PQ).
//!
//! PQ splits each input vector into `M = n_subvectors` equal-length
//! chunks and learns a small codebook of `K = n_centroids` centroids
//! (with `K <= 256`) for each chunk via k-means. A vector compresses
//! to `M` bytes — one centroid index per chunk — for a compression
//! ratio of `(dim * 4) / M` (e.g. 768 dims at `M = 16` → 16 bytes,
//! 192×). Reconstruction error trades off cleanly against `M` and `K`.
//!
//! Asymmetric distance computation (ADC) keeps the query in `f32`,
//! precomputes a per-subvector distance table from the query to each
//! of the `K` centroids, and scores a stored code with `M` table
//! lookups plus a single summation pass. The math is decomposable —
//! and so **PQ ADC returns the same value as
//! [`Quantizer::distance`](crate::Quantizer::distance) would after
//! [`Quantizer::dequantize`](crate::Quantizer::dequantize) +
//! [`iqdb_distance::compute`]** — for every metric where it's
//! supported.
//!
//! ## Supported metrics
//!
//! | Metric                                  | Supported | Why                                                              |
//! |-----------------------------------------|-----------|------------------------------------------------------------------|
//! | [`DistanceMetric::Euclidean`]           | yes       | `L2² = Σ_m L2²(q_m, c_m)`; take `sqrt` once at the end.          |
//! | [`DistanceMetric::DotProduct`]          | yes       | `dot = Σ_m dot(q_m, c_m)`; raw inner product (matches SQ8).      |
//! | [`DistanceMetric::Manhattan`]           | yes       | `L1 = Σ_m L1(q_m, c_m)`.                                         |
//! | [`DistanceMetric::Cosine`]              | **no**    | Requires global `‖c‖`; deferred to v0.4. Returns `InvalidMetric`.|
//! | [`DistanceMetric::Hamming`]             | **no**    | Meaningless on `f32` codes. Returns `InvalidMetric`.             |
//!
//! Production practice: L2-normalize vectors before training and use
//! [`DistanceMetric::DotProduct`] when you want cosine semantics.
//!
//! [`DistanceMetric::Euclidean`]: iqdb_types::DistanceMetric::Euclidean
//! [`DistanceMetric::DotProduct`]: iqdb_types::DistanceMetric::DotProduct
//! [`DistanceMetric::Manhattan`]: iqdb_types::DistanceMetric::Manhattan
//! [`DistanceMetric::Cosine`]: iqdb_types::DistanceMetric::Cosine
//! [`DistanceMetric::Hamming`]: iqdb_types::DistanceMetric::Hamming

use error_forge::ForgeError;
use iqdb_distance::compute_batch;
use iqdb_types::{DistanceMetric, IqdbError, Result};

use crate::code::PqCode;
use crate::train::{assign_to_cluster, squared_l2, train_codebook};
use crate::traits::Quantizer;
use crate::validate::{dim_eq, finite_non_empty, training_set};

/// Default number of subvectors used by [`ProductQuantizer::new`].
const DEFAULT_N_SUBVECTORS: usize = 8;
/// Default number of centroids per subvector used by [`ProductQuantizer::new`].
const DEFAULT_N_CENTROIDS: usize = 256;
/// Upper bound on `n_centroids`: codes are stored as `u8`.
const MAX_N_CENTROIDS: usize = 256;
/// Default seed used by [`ProductQuantizer::new`].
const DEFAULT_SEED: u64 = 0;

/// Calibration learned during [`ProductQuantizer::train`].
#[derive(Debug, Clone, PartialEq)]
struct PqCalibration {
    /// The trained input dimension; equals `n_subvectors * sub_dim`.
    dim: usize,
    /// `M`, the number of subvectors.
    n_subvectors: usize,
    /// `dim / n_subvectors`.
    sub_dim: usize,
    /// `K`, the number of centroids per subvector codebook.
    n_centroids: usize,
    /// `codebooks[m][k]` is the `k`-th centroid of subvector `m`,
    /// stored as a `Vec<f32>` of length `sub_dim`.
    codebooks: Vec<Vec<Vec<f32>>>,
}

/// Product quantizer: `M` subvectors × `K` centroids per subvector.
///
/// Build one with [`ProductQuantizer::new`] for the standard
/// `M = 8, K = 256` shape, or [`ProductQuantizer::with_config`] to
/// pick `M`, `K`, and the training `seed` explicitly. Train it once
/// with a representative sample, then quantize and compare. The
/// trained quantizer is callable from multiple threads — it owns its
/// calibration by value and exposes no interior mutability.
///
/// # Examples
///
/// ```
/// use iqdb_quantize::{ProductQuantizer, Quantizer};
/// use iqdb_types::DistanceMetric;
///
/// let mut pq = ProductQuantizer::with_config(2, 4, 7);
/// let training: Vec<Vec<f32>> = (0..16)
///     .map(|i| {
///         let f = i as f32;
///         vec![f, f + 1.0, f + 2.0, f + 3.0]
///     })
///     .collect();
/// let refs: Vec<&[f32]> = training.iter().map(Vec::as_slice).collect();
/// pq.train(&refs).expect("training succeeds");
///
/// let code = pq.quantize(&[1.0_f32, 2.0, 3.0, 4.0]).expect("quantize");
/// let d = pq
///     .distance(&[1.0_f32, 2.0, 3.0, 4.0], &code, DistanceMetric::Euclidean)
///     .expect("supported metric");
/// assert!(d.is_finite());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ProductQuantizer {
    n_subvectors: usize,
    n_centroids: usize,
    seed: u64,
    calibration: Option<PqCalibration>,
}

impl Default for ProductQuantizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ProductQuantizer {
    /// Build an untrained PQ with the standard shape (`M = 8`,
    /// `K = 256`, `seed = 0`).
    ///
    /// Every hot method returns [`IqdbError::InvalidConfig`] until
    /// [`Quantizer::train`] succeeds. The trained dimension must be a
    /// multiple of `M`, so `new()`'s `M = 8` works for the common
    /// embedding dimensions (128, 256, 384, 512, 768, 1024, …) but
    /// not for, say, dim 50; use [`ProductQuantizer::with_config`]
    /// when that matters.
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_quantize::ProductQuantizer;
    /// let pq = ProductQuantizer::new();
    /// assert_eq!(pq.n_subvectors(), 8);
    /// assert_eq!(pq.n_centroids(), 256);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(DEFAULT_N_SUBVECTORS, DEFAULT_N_CENTROIDS, DEFAULT_SEED)
    }

    /// Build an untrained PQ with the given shape and training seed.
    ///
    /// All three parameters take effect at [`Quantizer::train`] time;
    /// invalid combinations (e.g. `n_centroids == 0`, `n_centroids >
    /// 256`, training dim not divisible by `n_subvectors`) surface as
    /// [`IqdbError::InvalidConfig`] from `train`. The constructor
    /// itself is infallible — it just stores the configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_quantize::ProductQuantizer;
    /// let pq = ProductQuantizer::with_config(16, 256, 42);
    /// assert_eq!(pq.n_subvectors(), 16);
    /// assert_eq!(pq.n_centroids(), 256);
    /// assert_eq!(pq.seed(), 42);
    /// ```
    #[must_use]
    pub fn with_config(n_subvectors: usize, n_centroids: usize, seed: u64) -> Self {
        Self {
            n_subvectors,
            n_centroids,
            seed,
            calibration: None,
        }
    }

    /// The trained dimension, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_quantize::{ProductQuantizer, Quantizer};
    /// let mut pq = ProductQuantizer::with_config(2, 4, 7);
    /// assert_eq!(pq.dim(), None);
    /// let data: Vec<Vec<f32>> = (0..8).map(|i| vec![i as f32; 4]).collect();
    /// let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
    /// pq.train(&refs).expect("ok");
    /// assert_eq!(pq.dim(), Some(4));
    /// ```
    #[must_use]
    pub fn dim(&self) -> Option<usize> {
        self.calibration.as_ref().map(|c| c.dim)
    }

    /// The configured number of subvectors `M`.
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_quantize::ProductQuantizer;
    /// assert_eq!(ProductQuantizer::with_config(4, 16, 1).n_subvectors(), 4);
    /// ```
    #[must_use]
    pub fn n_subvectors(&self) -> usize {
        self.n_subvectors
    }

    /// The configured number of centroids per subvector codebook `K`.
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_quantize::ProductQuantizer;
    /// assert_eq!(ProductQuantizer::with_config(4, 16, 1).n_centroids(), 16);
    /// ```
    #[must_use]
    pub fn n_centroids(&self) -> usize {
        self.n_centroids
    }

    /// The configured training seed.
    ///
    /// Same seed + same training data ⇒ byte-identical codebooks.
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_quantize::ProductQuantizer;
    /// assert_eq!(ProductQuantizer::with_config(4, 16, 99).seed(), 99);
    /// ```
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    fn calibration(&self) -> Result<&PqCalibration> {
        self.calibration.as_ref().ok_or(IqdbError::InvalidConfig {
            reason: "ProductQuantizer has not been trained",
        })
    }

    /// Validate the configured shape against the training-set dimension.
    /// Returns `sub_dim = dim / n_subvectors` on success.
    fn validate_shape(&self, dim: usize, training_count: usize) -> Result<usize> {
        if self.n_subvectors == 0 {
            return Err(IqdbError::InvalidConfig {
                reason: "ProductQuantizer requires n_subvectors >= 1",
            });
        }
        if self.n_centroids == 0 {
            return Err(IqdbError::InvalidConfig {
                reason: "ProductQuantizer requires n_centroids >= 1",
            });
        }
        if self.n_centroids > MAX_N_CENTROIDS {
            return Err(IqdbError::InvalidConfig {
                reason: "ProductQuantizer requires n_centroids <= 256 (one byte per code)",
            });
        }
        if dim == 0 || !dim.is_multiple_of(self.n_subvectors) {
            return Err(IqdbError::InvalidConfig {
                reason: "ProductQuantizer requires training dim to be a positive multiple of n_subvectors",
            });
        }
        if training_count < self.n_centroids {
            return Err(IqdbError::InvalidConfig {
                reason: "ProductQuantizer requires training_set.len() >= n_centroids",
            });
        }
        Ok(dim / self.n_subvectors)
    }
}

impl Quantizer for ProductQuantizer {
    type Quantized = PqCode;

    #[tracing::instrument(
        level = "info",
        skip_all,
        fields(
            quantizer = "pq",
            training_size = vectors.len(),
            n_subvectors = self.n_subvectors,
            n_centroids = self.n_centroids,
        ),
    )]
    fn train(&mut self, vectors: &[&[f32]]) -> Result<()> {
        let dim = training_set(vectors).inspect_err(|err: &IqdbError| {
            tracing::error!(
                error.kind = err.kind(),
                error.reason = err.caption(),
                "product quantizer training failed",
            );
        })?;
        let sub_dim = self
            .validate_shape(dim, vectors.len())
            .inspect_err(|err: &IqdbError| {
                tracing::error!(
                    error.kind = err.kind(),
                    error.reason = err.caption(),
                    "product quantizer training failed",
                );
            })?;

        // Build the per-subvector training slices and train one
        // codebook per subvector position. The seed is per-subvector
        // (`base_seed.wrapping_add(m as u64)`) so the M k-means runs
        // don't all draw from the same PRNG state.
        let mut codebooks: Vec<Vec<Vec<f32>>> = Vec::with_capacity(self.n_subvectors);
        for m in 0..self.n_subvectors {
            let start = m * sub_dim;
            let end = start + sub_dim;
            let slices: Vec<&[f32]> = vectors.iter().map(|v| &v[start..end]).collect();
            let centroids = train_codebook(
                sub_dim,
                self.n_centroids,
                self.seed.wrapping_add(m as u64),
                &slices,
            )
            .inspect_err(|err: &IqdbError| {
                tracing::error!(
                    error.kind = err.kind(),
                    error.reason = err.caption(),
                    subvector = m,
                    "product quantizer codebook training failed",
                );
            })?;
            codebooks.push(centroids);
        }

        self.calibration = Some(PqCalibration {
            dim,
            n_subvectors: self.n_subvectors,
            sub_dim,
            n_centroids: self.n_centroids,
            codebooks,
        });
        Ok(())
    }

    fn quantize(&self, vector: &[f32]) -> Result<Self::Quantized> {
        let cal = self.calibration()?;
        finite_non_empty(vector)?;
        dim_eq(cal.dim, vector.len())?;
        let mut codes: Vec<u8> = Vec::with_capacity(cal.n_subvectors);
        for m in 0..cal.n_subvectors {
            let start = m * cal.sub_dim;
            let end = start + cal.sub_dim;
            let idx = assign_to_cluster(&cal.codebooks[m], &vector[start..end]);
            // `assign_to_cluster` returns an index in `0..n_centroids`,
            // and `n_centroids <= 256` (enforced in `validate_shape`),
            // so this cast cannot lose information.
            codes.push(idx as u8);
        }
        Ok(PqCode {
            codes,
            dim: cal.dim,
            n_subvectors: cal.n_subvectors,
        })
    }

    fn dequantize(&self, quantized: &Self::Quantized) -> Result<Vec<f32>> {
        let cal = self.calibration()?;
        dim_eq(cal.dim, quantized.dim)?;
        if quantized.n_subvectors != cal.n_subvectors {
            return Err(IqdbError::DimensionMismatch {
                expected: cal.n_subvectors,
                found: quantized.n_subvectors,
            });
        }
        let mut out: Vec<f32> = Vec::with_capacity(cal.dim);
        for (m, &code) in quantized.codes.iter().enumerate() {
            let centroid = &cal.codebooks[m][code as usize];
            out.extend_from_slice(centroid);
        }
        Ok(out)
    }

    fn distance(
        &self,
        query: &[f32],
        quantized: &Self::Quantized,
        metric: DistanceMetric,
    ) -> Result<f32> {
        let tables = self.build_query_tables(query, metric)?;
        tables.distance(quantized)
    }
}

impl ProductQuantizer {
    /// Build the ADC lookup tables for `(query, metric)` once so the
    /// caller can score many [`PqCode`]s against the same query
    /// without rebuilding the `M × K` table per call.
    ///
    /// This is the primitive that
    /// [`Quantizer::distance`](crate::Quantizer::distance) is built
    /// on; callers scoring a single code can keep using `distance`
    /// directly. Use this method when scoring a batch — e.g.
    /// IVF-PQ's intra-cluster scan, which builds the table once per
    /// query and then scores every code in every probed cluster.
    ///
    /// # Errors
    ///
    /// Returns [`IqdbError::InvalidConfig`] if the quantizer is
    /// untrained, [`IqdbError::InvalidVector`] if `query` is empty or
    /// non-finite, [`IqdbError::DimensionMismatch`] if `query.len()`
    /// doesn't match the trained dim, or [`IqdbError::InvalidMetric`]
    /// for [`DistanceMetric::Cosine`] / [`DistanceMetric::Hamming`].
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_quantize::{ProductQuantizer, Quantizer};
    /// use iqdb_types::DistanceMetric;
    ///
    /// let mut pq = ProductQuantizer::with_config(2, 4, 7);
    /// let training: Vec<Vec<f32>> = (0..16)
    ///     .map(|i| {
    ///         let f = i as f32;
    ///         vec![f, f + 1.0, f + 2.0, f + 3.0]
    ///     })
    ///     .collect();
    /// let refs: Vec<&[f32]> = training.iter().map(Vec::as_slice).collect();
    /// pq.train(&refs).expect("training succeeds");
    ///
    /// let code_a = pq.quantize(&[1.0_f32, 2.0, 3.0, 4.0]).expect("quantize");
    /// let code_b = pq.quantize(&[5.0_f32, 6.0, 7.0, 8.0]).expect("quantize");
    ///
    /// // Build the table ONCE for this (query, metric), then score many codes.
    /// let query = [1.0_f32, 2.0, 3.0, 4.0];
    /// let tables = pq
    ///     .build_query_tables(&query, DistanceMetric::Euclidean)
    ///     .expect("supported metric");
    /// let d_a = tables.distance(&code_a).expect("matching code shape");
    /// let d_b = tables.distance(&code_b).expect("matching code shape");
    /// assert!(d_a.is_finite() && d_b.is_finite());
    /// ```
    pub fn build_query_tables(&self, query: &[f32], metric: DistanceMetric) -> Result<PqAdcTables> {
        let cal = self.calibration()?;
        finite_non_empty(query)?;
        dim_eq(cal.dim, query.len())?;
        match metric {
            DistanceMetric::Euclidean | DistanceMetric::DotProduct | DistanceMetric::Manhattan => {}
            DistanceMetric::Cosine | DistanceMetric::Hamming => {
                return Err(IqdbError::InvalidMetric);
            }
            // `DistanceMetric` is `#[non_exhaustive]` in published iqdb-types
            // v1.0.0; any future variant defaults to InvalidMetric until PQ
            // explicitly opts in. Behavior on the five existing variants is
            // unchanged.
            _ => return Err(IqdbError::InvalidMetric),
        }
        let table = build_adc_table_rows(query, metric, cal)?;
        Ok(PqAdcTables {
            table,
            metric,
            n_subvectors: cal.n_subvectors,
            n_centroids: cal.n_centroids,
            dim: cal.dim,
        })
    }
}

/// Per-`(query, metric)` precomputed ADC lookup tables built from a
/// [`ProductQuantizer`].
///
/// Build once with [`ProductQuantizer::build_query_tables`], then
/// score many [`PqCode`]s against it via [`PqAdcTables::distance`]
/// without rebuilding the `M × K` table per call.
///
/// Row `m` of the internal table holds the distances from query
/// subvector `q_m` to each of the `K` centroids of codebook `m`,
/// packed row-major. For [`DistanceMetric::Euclidean`] the row holds
/// **squared L2** values (so they sum decomposably across
/// subvectors); [`PqAdcTables::distance`] takes `sqrt` of the total
/// exactly once for Euclidean.
#[derive(Debug, Clone)]
pub struct PqAdcTables {
    /// `n_subvectors * n_centroids` entries, row-major.
    table: Vec<f32>,
    metric: DistanceMetric,
    n_subvectors: usize,
    n_centroids: usize,
    dim: usize,
}

impl PqAdcTables {
    /// Score a single [`PqCode`] against the prepared tables.
    ///
    /// The returned value matches
    /// [`Quantizer::distance`](crate::Quantizer::distance) for the
    /// same `(query, code, metric)` — for [`DistanceMetric::Euclidean`]
    /// the table holds squared L2 per subvector and this method
    /// `sqrt`s the sum exactly once; the other supported metrics
    /// (`DotProduct`, `Manhattan`) sum directly.
    ///
    /// # Errors
    ///
    /// Returns [`IqdbError::DimensionMismatch`] if `code` was produced
    /// by a [`ProductQuantizer`] with a different `M` or trained `dim`
    /// — typically the same quantizer that built the tables.
    pub fn distance(&self, code: &PqCode) -> Result<f32> {
        if code.n_subvectors != self.n_subvectors {
            return Err(IqdbError::DimensionMismatch {
                expected: self.n_subvectors,
                found: code.n_subvectors,
            });
        }
        if code.dim != self.dim {
            return Err(IqdbError::DimensionMismatch {
                expected: self.dim,
                found: code.dim,
            });
        }
        let total = score_code_rows(&self.table, code, self.n_centroids);
        Ok(if self.metric == DistanceMetric::Euclidean {
            total.sqrt()
        } else {
            total
        })
    }

    /// The metric these tables were built for.
    #[must_use]
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// The number of subvectors `M`.
    #[must_use]
    pub fn n_subvectors(&self) -> usize {
        self.n_subvectors
    }

    /// The number of centroids per subvector codebook `K`.
    #[must_use]
    pub fn n_centroids(&self) -> usize {
        self.n_centroids
    }

    /// The trained dimension these tables were built against.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

fn build_adc_table_rows(
    query: &[f32],
    metric: DistanceMetric,
    cal: &PqCalibration,
) -> Result<Vec<f32>> {
    let total_entries = cal.n_subvectors * cal.n_centroids;
    let mut table: Vec<f32> = vec![0.0; total_entries];
    let mut centroid_refs: Vec<&[f32]> = Vec::with_capacity(cal.n_centroids);
    for m in 0..cal.n_subvectors {
        let start = m * cal.sub_dim;
        let end = start + cal.sub_dim;
        let q_sub = &query[start..end];
        let row_start = m * cal.n_centroids;
        let row_end = row_start + cal.n_centroids;
        let row = &mut table[row_start..row_end];

        match metric {
            DistanceMetric::Euclidean => {
                // Squared L2 per centroid, summed decomposably across
                // subvectors. The caller takes `sqrt` of the total in
                // `PqAdcTables::distance`.
                for (k, centroid) in cal.codebooks[m].iter().enumerate() {
                    row[k] = squared_l2(q_sub, centroid);
                }
            }
            DistanceMetric::DotProduct | DistanceMetric::Manhattan => {
                centroid_refs.clear();
                for centroid in &cal.codebooks[m] {
                    centroid_refs.push(centroid.as_slice());
                }
                compute_batch(metric, q_sub, &centroid_refs, row)?;
            }
            DistanceMetric::Cosine | DistanceMetric::Hamming => {
                // Rejected earlier in `build_query_tables` — expressing
                // it as an error here keeps the match total without a
                // panic if the upstream guard is ever relaxed.
                return Err(IqdbError::InvalidMetric);
            }
            // `DistanceMetric` is `#[non_exhaustive]` in published iqdb-types
            // v1.0.0; same defensive treatment as `build_query_tables`.
            _ => return Err(IqdbError::InvalidMetric),
        }
    }
    Ok(table)
}

fn score_code_rows(table: &[f32], code: &PqCode, n_centroids: usize) -> f32 {
    let mut sum: f32 = 0.0;
    for (m, &c) in code.codes.iter().enumerate() {
        let row_start = m * n_centroids;
        sum += table[row_start + c as usize];
    }
    sum
}
