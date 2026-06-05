//! Owned, immutable code types: [`Sq8Code`], [`BqCode`], and [`PqCode`].
//!
//! Each is a thin newtype over the underlying storage with no public
//! mutators. Codes are produced exclusively by their owning quantizer; a
//! caller cannot construct a `Sq8Code`, `BqCode`, or `PqCode` outside this
//! crate, which keeps the contract that a code's contents match the
//! calibrated quantizer.

/// A scalar-quantized (SQ8) code: one `u8` per dimension of the trained
/// vector space.
///
/// Produced by [`Quantizer::quantize`](crate::Quantizer::quantize) on a
/// trained [`ScalarQuantizer`](crate::ScalarQuantizer). The byte at
/// position `i` is the linear `u8` encoding of the original `f32`
/// component under that dimension's affine calibration; it is not useful
/// on its own. Decode it with
/// [`Quantizer::dequantize`](crate::Quantizer::dequantize) or compare it
/// against a query through
/// [`Quantizer::distance`](crate::Quantizer::distance).
///
/// # Examples
///
/// ```
/// use iqdb_quantize::{Quantizer, ScalarQuantizer};
///
/// let mut sq = ScalarQuantizer::new();
/// sq.train(&[&[0.0_f32, 1.0][..], &[1.0_f32, 0.0][..]]).expect("ok");
/// let code = sq.quantize(&[0.5_f32, 0.5]).expect("ok");
/// assert_eq!(code.len(), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8Code {
    pub(crate) bytes: Vec<u8>,
}

impl Sq8Code {
    /// Returns the dimension of the encoded vector (one byte per dimension).
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_quantize::{Quantizer, ScalarQuantizer};
    ///
    /// let mut sq = ScalarQuantizer::new();
    /// sq.train(&[&[0.0_f32, 1.0, 2.0][..]]).expect("ok");
    /// let code = sq.quantize(&[0.5_f32, 0.5, 0.5]).expect("ok");
    /// assert_eq!(code.len(), 3);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Returns `true` if the code holds no bytes.
    ///
    /// A `Sq8Code` produced by [`Quantizer::quantize`](crate::Quantizer::quantize)
    /// on a trained [`ScalarQuantizer`](crate::ScalarQuantizer) is never
    /// empty (empty inputs are rejected at the boundary); this method
    /// exists for API symmetry with [`Sq8Code::len`].
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_quantize::{Quantizer, ScalarQuantizer};
    ///
    /// let mut sq = ScalarQuantizer::new();
    /// sq.train(&[&[0.0_f32, 1.0][..]]).expect("ok");
    /// let code = sq.quantize(&[0.5_f32, 0.5]).expect("ok");
    /// assert!(!code.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Borrow the raw `u8` code bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_quantize::{Quantizer, ScalarQuantizer};
    ///
    /// let mut sq = ScalarQuantizer::new();
    /// sq.train(&[&[0.0_f32, 1.0][..]]).expect("ok");
    /// let code = sq.quantize(&[0.5_f32, 0.5]).expect("ok");
    /// assert_eq!(code.as_bytes().len(), 2);
    /// ```
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// A binary-quantized (BQ) code: one bit per dimension, packed into
/// `u64` words.
///
/// Produced by [`Quantizer::quantize`](crate::Quantizer::quantize) on a
/// trained [`BinaryQuantizer`](crate::BinaryQuantizer). Each bit is `1`
/// when the corresponding `f32` component is at or above the trained
/// per-dimension mean, `0` otherwise. When the dimension is not a
/// multiple of 64 the trailing word has unused high bits; those bits
/// are always `0`, so they cannot contribute to Hamming distance.
///
/// # Examples
///
/// ```
/// use iqdb_quantize::{BinaryQuantizer, Quantizer};
///
/// let mut bq = BinaryQuantizer::new();
/// bq.train(&[&[0.0_f32, 1.0, 2.0][..], &[2.0_f32, 1.0, 0.0][..]]).expect("ok");
/// let code = bq.quantize(&[0.5_f32, 1.5, 2.5]).expect("ok");
/// assert_eq!(code.dim(), 3);
/// // dim 3 fits in a single u64 word.
/// assert_eq!(code.as_words().len(), 1);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BqCode {
    pub(crate) words: Vec<u64>,
    pub(crate) dim: usize,
}

impl BqCode {
    /// Returns the original vector dimension this code was produced from.
    ///
    /// This is the number of meaningful bits in the packed words; the
    /// trailing word may have unused high bits, which are always zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_quantize::{BinaryQuantizer, Quantizer};
    ///
    /// let mut bq = BinaryQuantizer::new();
    /// bq.train(&[&[0.0_f32; 65][..], &[1.0_f32; 65][..]]).expect("ok");
    /// let code = bq.quantize(&[0.5_f32; 65]).expect("ok");
    /// assert_eq!(code.dim(), 65);
    /// // 65 bits requires two u64 words.
    /// assert_eq!(code.as_words().len(), 2);
    /// ```
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns `true` if the code encodes a zero-dimensional vector.
    ///
    /// A `BqCode` produced by [`Quantizer::quantize`](crate::Quantizer::quantize)
    /// on a trained [`BinaryQuantizer`](crate::BinaryQuantizer) is never
    /// empty (empty inputs are rejected at the boundary); this method
    /// exists for API symmetry with [`BqCode::dim`].
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_quantize::{BinaryQuantizer, Quantizer};
    ///
    /// let mut bq = BinaryQuantizer::new();
    /// bq.train(&[&[0.0_f32, 1.0][..]]).expect("ok");
    /// let code = bq.quantize(&[0.5_f32, 0.5]).expect("ok");
    /// assert!(!code.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dim == 0
    }

    /// Borrow the raw packed `u64` words.
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_quantize::{BinaryQuantizer, Quantizer};
    ///
    /// let mut bq = BinaryQuantizer::new();
    /// bq.train(&[&[0.0_f32; 4][..], &[1.0_f32; 4][..]]).expect("ok");
    /// let code = bq.quantize(&[0.5_f32; 4]).expect("ok");
    /// assert_eq!(code.as_words().len(), 1);
    /// ```
    #[must_use]
    pub fn as_words(&self) -> &[u64] {
        &self.words
    }
}

/// A product-quantized (PQ) code: one `u8` centroid index per subvector.
///
/// Produced by [`Quantizer::quantize`](crate::Quantizer::quantize) on a
/// trained [`ProductQuantizer`](crate::ProductQuantizer). The byte at
/// position `m` is the index (in `0..n_centroids`, where `n_centroids
/// <= 256`) of the centroid in subvector codebook `m` that best
/// approximates the `m`-th subvector of the encoded vector. Decode it
/// with [`Quantizer::dequantize`](crate::Quantizer::dequantize) (lossy)
/// or compare it against a query through
/// [`Quantizer::distance`](crate::Quantizer::distance).
///
/// # Examples
///
/// ```
/// use iqdb_quantize::{ProductQuantizer, Quantizer};
///
/// let mut pq = ProductQuantizer::with_config(2, 4, 42);
/// let training: Vec<Vec<f32>> = (0..8)
///     .map(|i| vec![i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32])
///     .collect();
/// let refs: Vec<&[f32]> = training.iter().map(Vec::as_slice).collect();
/// pq.train(&refs).expect("training succeeds");
///
/// let code = pq.quantize(&[1.0_f32, 2.0, 3.0, 4.0]).expect("quantize");
/// assert_eq!(code.n_subvectors(), 2);
/// assert_eq!(code.dim(), 4);
/// assert_eq!(code.as_bytes().len(), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PqCode {
    pub(crate) codes: Vec<u8>,
    pub(crate) dim: usize,
    pub(crate) n_subvectors: usize,
}

impl PqCode {
    /// Returns the original vector dimension this code was produced from.
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_quantize::{ProductQuantizer, Quantizer};
    ///
    /// let mut pq = ProductQuantizer::with_config(2, 4, 7);
    /// let training: Vec<Vec<f32>> = (0..8)
    ///     .map(|i| vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32])
    ///     .collect();
    /// let refs: Vec<&[f32]> = training.iter().map(Vec::as_slice).collect();
    /// pq.train(&refs).expect("training succeeds");
    /// let code = pq.quantize(&[1.0_f32, 2.0, 3.0, 4.0]).expect("quantize");
    /// assert_eq!(code.dim(), 4);
    /// ```
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the number of subvectors `M` this code was produced under.
    ///
    /// Equal to `self.as_bytes().len()` and to the
    /// [`ProductQuantizer::n_subvectors`](crate::ProductQuantizer::n_subvectors)
    /// the code was produced with.
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_quantize::{ProductQuantizer, Quantizer};
    ///
    /// let mut pq = ProductQuantizer::with_config(2, 4, 7);
    /// let training: Vec<Vec<f32>> = (0..8)
    ///     .map(|i| vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32])
    ///     .collect();
    /// let refs: Vec<&[f32]> = training.iter().map(Vec::as_slice).collect();
    /// pq.train(&refs).expect("training succeeds");
    /// let code = pq.quantize(&[1.0_f32, 2.0, 3.0, 4.0]).expect("quantize");
    /// assert_eq!(code.n_subvectors(), 2);
    /// ```
    #[must_use]
    pub fn n_subvectors(&self) -> usize {
        self.n_subvectors
    }

    /// Returns the number of bytes in the code (equal to
    /// [`PqCode::n_subvectors`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_quantize::{ProductQuantizer, Quantizer};
    ///
    /// let mut pq = ProductQuantizer::with_config(2, 4, 7);
    /// let training: Vec<Vec<f32>> = (0..8)
    ///     .map(|i| vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32])
    ///     .collect();
    /// let refs: Vec<&[f32]> = training.iter().map(Vec::as_slice).collect();
    /// pq.train(&refs).expect("training succeeds");
    /// let code = pq.quantize(&[1.0_f32, 2.0, 3.0, 4.0]).expect("quantize");
    /// assert_eq!(code.len(), 2);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Returns `true` if the code holds no centroid indices.
    ///
    /// A `PqCode` produced by [`Quantizer::quantize`](crate::Quantizer::quantize)
    /// on a trained [`ProductQuantizer`](crate::ProductQuantizer) is never
    /// empty (empty inputs and `n_subvectors == 0` are rejected at the
    /// boundary); this method exists for API symmetry with [`PqCode::len`].
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_quantize::{ProductQuantizer, Quantizer};
    ///
    /// let mut pq = ProductQuantizer::with_config(2, 4, 7);
    /// let training: Vec<Vec<f32>> = (0..8)
    ///     .map(|i| vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32])
    ///     .collect();
    /// let refs: Vec<&[f32]> = training.iter().map(Vec::as_slice).collect();
    /// pq.train(&refs).expect("training succeeds");
    /// let code = pq.quantize(&[1.0_f32, 2.0, 3.0, 4.0]).expect("quantize");
    /// assert!(!code.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Borrow the raw centroid-index bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_quantize::{ProductQuantizer, Quantizer};
    ///
    /// let mut pq = ProductQuantizer::with_config(2, 4, 7);
    /// let training: Vec<Vec<f32>> = (0..8)
    ///     .map(|i| vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32])
    ///     .collect();
    /// let refs: Vec<&[f32]> = training.iter().map(Vec::as_slice).collect();
    /// pq.train(&refs).expect("training succeeds");
    /// let code = pq.quantize(&[1.0_f32, 2.0, 3.0, 4.0]).expect("quantize");
    /// assert_eq!(code.as_bytes().len(), 2);
    /// ```
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.codes
    }
}
