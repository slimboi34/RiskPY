//! `ExperienceRating`: credibility and experience modification.

use crate::error::{Result, invalid};
use crate::m;

#[derive(Debug, Clone)]
pub struct ExperienceRating {
    k_parameter: f64,
    ballast: f64,
}

impl ExperienceRating {
    pub fn new(k_parameter: f64, ballast: f64) -> Result<Self> {
        if !(k_parameter > 0.0) || !k_parameter.is_finite() {
            return invalid("k_parameter must be finite and > 0");
        }
        if !(ballast >= 0.0) || !ballast.is_finite() {
            return invalid("ballast must be finite and non-negative");
        }
        Ok(ExperienceRating { k_parameter, ballast })
    }

    /// Bühlmann credibility-weighted rate.
    pub fn credibility_weighted_rate(&self, observed_rate: f64, expected_rate: f64, credibility_z: f64) -> Result<f64> {
        if !(credibility_z >= 0.0 && credibility_z <= 1.0) {
            return invalid("Credibility Z must be between 0 and 1");
        }
        if !(observed_rate >= 0.0)
            || !(expected_rate >= 0.0)
            || !observed_rate.is_finite()
            || !expected_rate.is_finite()
        {
            return invalid("Rates must be finite and non-negative");
        }
        Ok((credibility_z * observed_rate) + ((1.0 - credibility_z) * expected_rate))
    }

    /// Classical limited-fluctuation credibility, min(1, √(n / k)).
    pub fn calculate_credibility(&self, claim_count: i32) -> Result<f64> {
        if claim_count < 0 {
            return invalid("Claim count cannot be negative");
        }
        let z = m::sqrt(f64::from(claim_count) / self.k_parameter);
        // std::min(1.0, z): returns 1.0 unless z < 1.0 (so a NaN z would give 1.0).
        Ok(if z < 1.0 { z } else { 1.0 })
    }

    /// NCCI-style experience modification factor.
    pub fn experience_mod_factor(&self, actual_losses: f64, expected_losses: f64) -> Result<f64> {
        if !(actual_losses >= 0.0)
            || !(expected_losses >= 0.0)
            || !actual_losses.is_finite()
            || !expected_losses.is_finite()
        {
            return invalid("Losses must be finite and non-negative");
        }
        if expected_losses == 0.0 {
            return invalid("Expected losses cannot be zero");
        }
        Ok((actual_losses + self.ballast * expected_losses) / (expected_losses * (1.0 + self.ballast)))
    }

    /// Bühlmann–Straub credibility factor, P / (P + K).
    pub fn buhlmann_straub_credibility(&self, exposure: f64, variance_ratio: f64) -> Result<f64> {
        if !(exposure >= 0.0) || !exposure.is_finite() {
            return invalid("Exposure must be finite and non-negative");
        }
        if !(variance_ratio >= 0.0) || !variance_ratio.is_finite() {
            return invalid("Variance ratio must be finite and non-negative");
        }
        if exposure + variance_ratio == 0.0 {
            return invalid("Exposure and variance ratio cannot both be zero (credibility undefined)");
        }
        Ok(exposure / (exposure + variance_ratio))
    }

    /// Expected losses from exposure, frequency and severity.
    pub fn expected_losses(&self, exposure: f64, frequency: f64, severity: f64) -> Result<f64> {
        if !(exposure >= 0.0)
            || !(frequency >= 0.0)
            || !(severity >= 0.0)
            || !exposure.is_finite()
            || !frequency.is_finite()
            || !severity.is_finite()
        {
            return invalid("Inputs must be finite and non-negative");
        }
        Ok(exposure * frequency * severity)
    }
}
