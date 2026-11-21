//! `RateAnalyzer`: on-levelling, trend, rate change indications.

use crate::error::{Result, invalid, vec_filled};
use crate::m;

#[derive(Debug, Clone)]
pub struct RateAnalyzer {
    target_combined_ratio: f64,
    expense_ratio: f64,
}

impl RateAnalyzer {
    pub fn new(target_combined_ratio: f64, expense_ratio: f64) -> Result<Self> {
        if !target_combined_ratio.is_finite() || target_combined_ratio <= 0.0 {
            return invalid("Target combined ratio must be a finite positive number");
        }
        if !expense_ratio.is_finite() || expense_ratio < 0.0 || expense_ratio >= 1.0 {
            return invalid("Expense ratio must be finite and in [0, 1)");
        }
        Ok(RateAnalyzer { target_combined_ratio, expense_ratio })
    }

    /// Product of (1 + change) over a history of rate changes.
    pub fn on_level_factor(&self, rate_changes: &[f64]) -> Result<f64> {
        let mut factor = 1.0;
        for &change in rate_changes {
            if !(change > -1.0) || !change.is_finite() {
                return invalid("Rate changes must be finite and > -100%");
            }
            factor *= 1.0 + change;
        }
        Ok(factor)
    }

    pub fn rate_change_impact(&self, old_premium: f64, new_premium: f64) -> Result<f64> {
        if !(old_premium > 0.0) || !old_premium.is_finite() {
            return invalid("Old premium must be finite and > 0");
        }
        if !(new_premium >= 0.0) || !new_premium.is_finite() {
            return invalid("New premium must be finite and non-negative");
        }
        Ok((new_premium / old_premium) - 1.0)
    }

    pub fn combined_ratio(&self, loss_ratio: f64) -> Result<f64> {
        if !(loss_ratio >= 0.0) || !loss_ratio.is_finite() {
            return invalid("Loss ratio must be finite and non-negative");
        }
        Ok(loss_ratio + self.expense_ratio)
    }

    pub fn required_rate_change(&self, current_loss_ratio: f64) -> Result<f64> {
        if !(current_loss_ratio >= 0.0) || !current_loss_ratio.is_finite() {
            return invalid("Loss ratio must be finite and non-negative");
        }
        let plr = self.target_combined_ratio - self.expense_ratio;
        if plr <= 0.0 {
            return invalid("Permissible loss ratio would mathematically result in division by zero/negatives");
        }
        Ok((current_loss_ratio / plr) - 1.0)
    }

    /// Earned premium brought to current rate level, latest period first.
    pub fn on_level_premiums(&self, earned_premiums: &[f64], rate_changes: &[f64]) -> Result<Vec<f64>> {
        if earned_premiums.len() != rate_changes.len() {
            return invalid("Mismatched arrays: Prem and Rate vectors must be same size");
        }
        let n = earned_premiums.len();
        let mut on_leveled = vec_filled(n, 0.0)?;
        let mut cumulative_factor = 1.0;
        for i in (0..n).rev() {
            if !(rate_changes[i] > -1.0) || !rate_changes[i].is_finite() {
                return invalid("Rate change must be finite and > -100%");
            }
            if !(earned_premiums[i] >= 0.0) || !earned_premiums[i].is_finite() {
                return invalid("Premium must be finite and non-negative");
            }
            on_leveled[i] = earned_premiums[i] * cumulative_factor;
            cumulative_factor *= 1.0 + rate_changes[i];
        }
        Ok(on_leveled)
    }

    pub fn permissible_loss_ratio(&self) -> f64 {
        self.target_combined_ratio - self.expense_ratio
    }

    /// (1 + annual_trend)^years.
    pub fn trend_factor(&self, annual_trend: f64, years: f64) -> Result<f64> {
        if !(annual_trend > -1.0) || !annual_trend.is_finite() {
            return invalid("Trend must be finite and > -100%");
        }
        if !(years >= 0.0) || !years.is_finite() {
            return invalid("Years must be finite and non-negative");
        }
        Ok(m::pow(1.0 + annual_trend, years))
    }

    pub fn loss_cost_rate(&self, incurred_losses: f64, earned_exposure: f64) -> Result<f64> {
        if !(earned_exposure > 0.0) || !earned_exposure.is_finite() {
            return invalid("Exposure must be finite and > 0");
        }
        if !(incurred_losses >= 0.0) || !incurred_losses.is_finite() {
            return invalid("Losses must be finite and non-negative");
        }
        Ok(incurred_losses / earned_exposure)
    }
}
