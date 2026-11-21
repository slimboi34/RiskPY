//! The core's error type. The bindings map each kind onto the Python exception the
//! C++ core raised for it (pybind11's translation of the C++ exception):
//!
//! | C++ exception            | here               | Python         |
//! |--------------------------|--------------------|----------------|
//! | `std::invalid_argument`  | [`Error::Value`]   | `ValueError`   |
//! | `std::runtime_error`     | [`Error::Runtime`] | `RuntimeError` |
//! | `std::bad_alloc`         | [`Error::Memory`]  | `MemoryError`  |

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A refused argument (C++ `std::invalid_argument`).
    Value(String),
    /// A failure while doing the work, such as writing a file (C++ `std::runtime_error`).
    Runtime(String),
    /// An allocation that could not be made (C++ `std::bad_alloc`).
    Memory(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Value(m) | Error::Runtime(m) | Error::Memory(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Error {}

/// `Err(Error::Value(msg))`: the `throw std::invalid_argument(msg)` of the C++ core.
pub(crate) fn invalid<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Value(msg.into()))
}

/// A vector with room for `n` elements, or [`Error::Memory`] instead of the abort a
/// failed `Vec::with_capacity` would be.
pub(crate) fn vec_with_capacity<T>(n: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| Error::Memory(format!("cannot allocate {n} elements")))?;
    Ok(v)
}

/// A vector of `n` copies of `value`, or [`Error::Memory`].
pub(crate) fn vec_filled<T: Clone>(n: usize, value: T) -> Result<Vec<T>> {
    let mut v = vec_with_capacity(n)?;
    v.resize(n, value);
    Ok(v)
}
