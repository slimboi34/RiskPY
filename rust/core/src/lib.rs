//! RiskPY's compiled core, in plain Rust.
//!
//! This crate has no Python in it. `riskpy-python` (the sibling crate) wraps it with
//! PyO3 as the extension module `riskpy.cpp_underwriter`; the module keeps the name the
//! C++ core had, so nothing that imports it changes.
//!
//! Every class of the old C++ core has a counterpart here with the same checks, the
//! same error messages and the same arithmetic, operation for operation, so the
//! deterministic methods return the numbers the C++ returned. Two things differ on
//! purpose:
//!
//! * the random numbers ([`rng`], [`dist`]): a seeded PCG64-DXSM stream with samplers
//!   written here, identical on every platform, where the C++ used the standard
//!   library's distributions, which differ between libc++, libstdc++ and MSVC;
//! * [`actuarial::present_value`] / [`actuarial::future_value`]: `expm1`/`log1p`
//!   forms that keep their digits for tiny rates, and a refusal of negative periods
//!   and non-finite payments.
//!
//! Transcendental functions all come from the `libm` crate (see [`m`]) so results do
//! not depend on the platform's C library.

// The argument checks are written `!(x >= 0.0)` and `!(a <= x && x <= b)` on purpose:
// a NaN fails every comparison, so these forms refuse NaN where `x < 0.0` would let it
// through (the bug the C++ core's 0.4.1 hardening fixed). They mirror the C++ checks
// one for one, which is what makes the port reviewable against them. For the same
// reason the clamps and compound conditions keep the C++ core's shape.
#![allow(
    clippy::neg_cmp_op_on_partial_ord,
    clippy::manual_range_contains,
    clippy::manual_clamp,
    clippy::nonminimal_bool
)]

pub mod actuarial;
pub mod error;
pub mod experience;
pub mod exposure;
pub mod factor;
pub mod fourier;
pub mod m;
pub mod montecarlo;
pub mod rate;
pub mod rng;
pub mod dist;
pub mod triangle;
pub mod value;
pub mod xlsx;

pub use error::{Error, Result};
pub use value::{Inputs, Value};
