//! `ExposureRating`: increased limits, layers, burning cost, lognormal LER.

use crate::error::{Result, invalid};
use crate::m;

#[derive(Debug, Clone)]
pub struct ExposureRating {
    base_limit: f64,
    b_parameter: f64,
}

fn std_normal_cdf(x: f64) -> f64 {
    0.5 * m::erfc(-x / std::f64::consts::SQRT_2)
}

/// `std::min(a, b)`: `b < a ? b : a`.
#[inline]
fn cpp_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// `std::max(a, b)`: `a < b ? b : a`.
#[inline]
fn cpp_max(a: f64, b: f64) -> f64 {
    if a < b { b } else { a }
}

impl ExposureRating {
    pub fn new(base_limit: f64, b_parameter: f64) -> Result<Self> {
        if !(base_limit > 0.0) || !base_limit.is_finite() {
            return invalid("Base limit must be finite and > 0");
        }
        // Power-curve ILF (L/B)^(1-b): b in [0, 1] keeps it increasing and concave
        // (b = 0 linear, b = 1 flat); outside that range layer premiums turn negative.
        if !(b_parameter >= 0.0 && b_parameter <= 1.0) {
            return invalid("b_parameter must be in [0, 1]");
        }
        Ok(ExposureRating { base_limit, b_parameter })
    }

    pub fn increased_limits_factor(&self, target_limit: f64) -> Result<f64> {
        if !(target_limit > 0.0) || !target_limit.is_finite() {
            return invalid("Target limit must be finite and strictly positive");
        }
        Ok(m::pow(target_limit / self.base_limit, 1.0 - self.b_parameter))
    }

    pub fn layer_premium(&self, ground_up_premium: f64, attachment: f64, limit: f64) -> Result<f64> {
        if !(ground_up_premium >= 0.0) || !ground_up_premium.is_finite() {
            return invalid("Premium must be finite and non-negative");
        }
        if !(attachment >= 0.0) || !(limit > 0.0) || !attachment.is_finite() || !limit.is_finite() {
            return invalid("Invalid attachment or limit (must be finite, attachment >= 0, limit > 0)");
        }
        let ilf_at_base = self.increased_limits_factor(self.base_limit)?;
        // The power-curve ILF is only defined for limits > 0, but the limited expected
        // value at zero is zero for any severity curve, so a ground-up layer uses 0.
        let ilf_at_attachment = if attachment > 0.0 { self.increased_limits_factor(attachment)? } else { 0.0 };
        let ilf_at_limit = self.increased_limits_factor(attachment + limit)?;
        if ilf_at_base == 0.0 {
            return invalid("ILF Base cannot be zero");
        }
        Ok(ground_up_premium * (ilf_at_limit - ilf_at_attachment) / ilf_at_base)
    }

    pub fn burning_cost(&self, historical_losses: &[f64], attachment: f64, limit: f64, exposure_years: i32) -> Result<f64> {
        if exposure_years <= 0 {
            return invalid("Exposure years must be strictly positive");
        }
        if !(attachment >= 0.0) || !(limit > 0.0) || !attachment.is_finite() || !limit.is_finite() {
            return invalid("Invalid attachment or limit (must be finite, attachment >= 0, limit > 0)");
        }
        let mut total_layer_losses = 0.0;
        for &loss in historical_losses {
            if !(loss >= 0.0) || !loss.is_finite() {
                return invalid("Loss amounts must be finite and non-negative");
            }
            let capped = cpp_min(loss, attachment + limit);
            let layer_loss = cpp_max(capped - attachment, 0.0);
            total_layer_losses += layer_loss;
        }
        Ok(total_layer_losses / f64::from(exposure_years))
    }

    pub fn loss_elimination_ratio(&self, deductible: f64, expected_loss: f64, severity_cv: f64) -> Result<f64> {
        if !(expected_loss > 0.0) || !expected_loss.is_finite() {
            return invalid("Expected loss must be finite and positive");
        }
        if !(deductible >= 0.0) || deductible.is_nan() {
            return invalid("Deductible cannot be negative or NaN");
        }
        if !(severity_cv > 0.0) || !severity_cv.is_finite() {
            return invalid("Severity CV must be finite and > 0");
        }
        // Lognormal severity with mean E and coefficient of variation CV:
        //   sigma^2 = ln(1 + CV^2),  mu = ln E - sigma^2 / 2.
        // LER(d) = E[min(X, d)] / E[X], with the limited expected value
        //   LEV(d) = e^(mu + sigma^2/2) Phi((ln d - mu - sigma^2)/sigma) + d (1 - Phi((ln d - mu)/sigma)).
        if deductible == 0.0 {
            return Ok(0.0);
        }
        if deductible.is_infinite() {
            return Ok(1.0);
        }
        let sigma2 = m::ln1p(severity_cv * severity_cv);
        let sigma = m::sqrt(sigma2);
        let mu = m::ln(expected_loss) - 0.5 * sigma2;
        let z = (m::ln(deductible) - mu) / sigma;
        // LEV/E = Phi(z - sigma) + (d/E) (1 - Phi(z)); 1 - Phi(z) = Phi(-z) avoids cancellation.
        let mut ler = std_normal_cdf(z - sigma) + (deductible / expected_loss) * std_normal_cdf(-z);
        if ler > 1.0 {
            ler = 1.0;
        }
        if ler < 0.0 {
            ler = 0.0;
        }
        Ok(ler)
    }

    pub fn excess_risk_load(&self, layer_std_dev: f64, risk_load_multiple: f64) -> Result<f64> {
        if !(layer_std_dev >= 0.0) || !layer_std_dev.is_finite() {
            return invalid("Layer std_dev must be finite and non-negative");
        }
        if !risk_load_multiple.is_finite() {
            return invalid("Risk load multiple must be finite");
        }
        Ok(layer_std_dev * risk_load_multiple)
    }
}
