//! The transcendental functions the core uses, all from the pure-Rust `libm` crate.
//!
//! `f64::exp` and friends call the platform's C library, and Apple's libm, glibc and
//! the MSVC runtime do not agree in the last bit. That made the C++ core's results
//! platform-dependent; routing every call through `libm` makes them the same
//! everywhere. Square roots, `floor` and the basic arithmetic are exact under IEEE 754
//! on every target, so they use the standard operators.

#[inline]
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

#[inline]
pub fn expm1(x: f64) -> f64 {
    libm::expm1(x)
}

#[inline]
pub fn ln(x: f64) -> f64 {
    libm::log(x)
}

#[inline]
pub fn ln1p(x: f64) -> f64 {
    libm::log1p(x)
}

#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}

#[inline]
pub fn sin(x: f64) -> f64 {
    libm::sin(x)
}

#[inline]
pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}

#[inline]
pub fn erfc(x: f64) -> f64 {
    libm::erfc(x)
}

/// Correctly rounded under IEEE 754, so the same on every platform.
#[inline]
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

/// Exact on every platform.
#[inline]
pub fn floor(x: f64) -> f64 {
    x.floor()
}

/// `std::to_string(double)`: `printf("%f")`, six decimals.
pub fn cpp_to_string(x: f64) -> String {
    if x.is_nan() {
        if x.is_sign_negative() { "-nan".into() } else { "nan".into() }
    } else if x.is_infinite() {
        if x > 0.0 { "inf".into() } else { "-inf".into() }
    } else {
        format!("{x:.6}")
    }
}
