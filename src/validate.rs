//! Shared input-validation helpers.
//!
//! Surface bad input as typed [`iqdb_types::IqdbError`] variants so the
//! quantizer hot paths never panic on empty, non-finite, or
//! dimension-mismatched inputs.

use iqdb_types::{IqdbError, Result};

/// Reject empty vectors and vectors with NaN / infinite components.
pub(crate) fn finite_non_empty(vector: &[f32]) -> Result<()> {
    if vector.is_empty() {
        return Err(IqdbError::InvalidVector);
    }
    if vector.iter().any(|v| !v.is_finite()) {
        return Err(IqdbError::InvalidVector);
    }
    Ok(())
}

/// Check that `actual == expected`, returning
/// [`IqdbError::DimensionMismatch`] when they disagree.
pub(crate) fn dim_eq(expected: usize, actual: usize) -> Result<()> {
    if actual != expected {
        return Err(IqdbError::DimensionMismatch {
            expected,
            found: actual,
        });
    }
    Ok(())
}

/// Validate a training set as a unit: non-empty, every vector valid, every
/// vector the same dimension. Returns the shared dimension on success.
pub(crate) fn training_set(vectors: &[&[f32]]) -> Result<usize> {
    let first = match vectors.first() {
        Some(first) => first,
        None => {
            return Err(IqdbError::InvalidConfig {
                reason: "quantizer training set is empty",
            });
        }
    };
    finite_non_empty(first)?;
    let dim = first.len();
    for v in &vectors[1..] {
        finite_non_empty(v)?;
        dim_eq(dim, v.len())?;
    }
    Ok(dim)
}
