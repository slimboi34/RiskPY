//! `ActuarialMath`: annuity present and future values, loss ratio, a mortality lookup.

use crate::error::{Result, invalid};
use crate::m;

fn check_annuity_args(rate: f64, periods: i32, payment: f64) -> Result<()> {
    if !(rate > -1.0) || !rate.is_finite() {
        return invalid("Rate must be finite and > -1.0");
    }
    if periods < 0 {
        return invalid("Periods must be non-negative");
    }
    if !payment.is_finite() {
        return invalid("Payment must be finite");
    }
    Ok(())
}

/// Present value of an annuity-immediate: payment · (1 − (1+r)^−n) / r.
///
/// Written as `−expm1(−n·log1p(r)) / r`, which keeps full precision for tiny rates;
/// the C++ core's `(1 − pow(1+r, −n)) / r` lost about half the digits at r = 1e-8
/// (0.8.0's known limit). Negative periods and non-finite payments are refused (they
/// were accepted before 0.9.0).
pub fn present_value(rate: f64, periods: i32, payment: f64) -> Result<f64> {
    check_annuity_args(rate, periods, payment)?;
    if rate == 0.0 {
        return Ok(payment * f64::from(periods));
    }
    if periods == 0 {
        // The old formula's (1 − 1) / r, keeping its sign of zero.
        return Ok(payment * (0.0 / rate));
    }
    let n = f64::from(periods);
    Ok(payment * (-m::expm1(-n * m::ln1p(rate)) / rate))
}

/// Future value of an annuity-immediate: payment · ((1+r)^n − 1) / r, as
/// `expm1(n·log1p(r)) / r` (see [`present_value`]).
pub fn future_value(rate: f64, periods: i32, payment: f64) -> Result<f64> {
    check_annuity_args(rate, periods, payment)?;
    if rate == 0.0 {
        return Ok(payment * f64::from(periods));
    }
    if periods == 0 {
        return Ok(payment * (0.0 / rate));
    }
    let n = f64::from(periods);
    Ok(payment * (m::expm1(n * m::ln1p(rate)) / rate))
}

pub fn calculate_loss_ratio(incurred_losses: f64, earned_premium: f64) -> Result<f64> {
    if !(earned_premium > 0.0) || !earned_premium.is_finite() {
        return invalid("Earned premium must be finite and strictly positive");
    }
    if !(incurred_losses >= 0.0) || !incurred_losses.is_finite() {
        return invalid("Incurred losses must be finite and non-negative");
    }
    Ok(incurred_losses / earned_premium)
}

/// A simplified mock of the CSO 2001 mortality table.
pub fn lookup_mortality_rate(age: i32) -> f64 {
    match age {
        i32::MIN..=-1 => 0.0,
        0..=18 => 0.0015,
        19..=30 => 0.0020,
        31..=40 => 0.0025,
        41..=50 => 0.0040,
        51..=60 => 0.0090,
        61..=70 => 0.0250,
        71..=80 => 0.0600,
        81..=90 => 0.1500,
        _ => 0.3000,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annuity_values() {
        let pv = present_value(0.05, 10, 100.0).unwrap();
        assert!((pv - 772.173_492_918_482_3).abs() < 1e-9);
        let fv = future_value(0.05, 10, 100.0).unwrap();
        assert!((fv - 1_257.789_253_554_883_7).abs() < 1e-9);
        assert_eq!(present_value(0.0, 7, 3.0).unwrap(), 21.0);
        assert_eq!(present_value(0.05, 0, 3.0).unwrap(), 0.0);
    }

    #[test]
    fn tiny_rates_keep_their_digits() {
        // n = 10, r = 1e-12: PV = n - n(n+1)/2 r + ... = 10 - 5.5e-11.
        let pv = present_value(1e-12, 10, 1.0).unwrap();
        assert!((pv - (10.0 - 5.5e-11)).abs() < 1e-14, "{pv}");
        let fv = future_value(1e-12, 10, 1.0).unwrap();
        assert!((fv - (10.0 + 4.5e-11)).abs() < 1e-14, "{fv}");
    }

    #[test]
    fn refusals() {
        assert!(present_value(f64::NAN, 1, 1.0).is_err());
        assert!(present_value(-1.0, 1, 1.0).is_err());
        assert!(present_value(0.05, -1, 1.0).is_err());
        assert!(future_value(0.05, 1, f64::INFINITY).is_err());
        assert!(future_value(0.05, 1, f64::NAN).is_err());
        assert!(calculate_loss_ratio(1.0, 0.0).is_err());
        assert!(calculate_loss_ratio(-1.0, 1.0).is_err());
    }

    #[test]
    fn mortality_bands() {
        assert_eq!(lookup_mortality_rate(-5), 0.0);
        assert_eq!(lookup_mortality_rate(18), 0.0015);
        assert_eq!(lookup_mortality_rate(19), 0.0020);
        assert_eq!(lookup_mortality_rate(91), 0.3);
    }
}
