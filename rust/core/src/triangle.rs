//! `LossTriangle`: chain-ladder development factors, ultimates and IBNR.

use crate::error::{Result, invalid, vec_filled, vec_with_capacity};

#[derive(Debug, Clone, Default)]
pub struct LossTriangle {
    origin_years: Vec<i32>,
    // rows = origin years, columns = development periods; rows may be ragged.
    triangle_data: Vec<Vec<f64>>,
}

impl LossTriangle {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a row of cumulative payments for an origin year.
    pub fn add_origin_year(&mut self, year: i32, cumulative_payments: Vec<f64>) -> Result<()> {
        if year < 0 {
            return invalid("Origin year cannot be negative");
        }
        if cumulative_payments.is_empty() {
            return invalid("Cumulative payments cannot be empty");
        }
        for &p in &cumulative_payments {
            if !p.is_finite() || p < 0.0 {
                return invalid("Cumulative payments must be finite and non-negative");
            }
        }
        self.origin_years.push(year);
        self.triangle_data.push(cumulative_payments);
        Ok(())
    }

    pub fn origin_year_count(&self) -> usize {
        self.origin_years.len()
    }

    pub fn development_period_count(&self) -> usize {
        self.triangle_data.iter().map(Vec::len).max().unwrap_or(0)
    }

    /// Volume-weighted age-to-age factors: for each age d, the sum of C(i, d+1) over the
    /// sum of C(i, d), across the origin years that have both.
    pub fn development_factors(&self) -> Result<Vec<f64>> {
        let n_periods = self.development_period_count();
        if n_periods <= 1 {
            return Ok(Vec::new());
        }
        let mut factors = vec_with_capacity(n_periods - 1)?;
        for d in 0..n_periods - 1 {
            let mut sum_next = 0.0;
            let mut sum_curr = 0.0;
            for row in &self.triangle_data {
                if row.len() > d + 1 {
                    sum_curr += row[d];
                    sum_next += row[d + 1];
                }
            }
            if sum_curr > 0.0 {
                factors.push(sum_next / sum_curr);
            } else if sum_next == 0.0 {
                // Nothing paid at either age: no development to project.
                factors.push(1.0);
            } else {
                // Zero at age d but positive at d+1: the age-to-age factor is infinite.
                return invalid(format!(
                    "Development factor undefined: column {d} sums to zero but column {} is positive \
                     (start the triangle at the first non-zero age)",
                    d + 1
                ));
            }
        }
        Ok(factors)
    }

    /// Ultimate losses: each origin year's latest value times the cumulative factor
    /// from its latest age to ultimate.
    pub fn ultimate_losses(&self) -> Result<Vec<f64>> {
        let dev_factors = self.development_factors()?;
        let n_periods = self.development_period_count();
        let n_years = self.triangle_data.len();
        if n_years == 0 {
            return Ok(Vec::new());
        }
        // CDF(d) = LDF(d) · LDF(d+1) ⋯ LDF(n−2), built from the end.
        let mut cum_factors = vec_filled(n_periods, 1.0)?;
        if n_periods >= 2 {
            for d in (0..=n_periods - 2).rev() {
                if d < dev_factors.len() {
                    cum_factors[d] = dev_factors[d] * cum_factors[d + 1];
                }
            }
        }
        let mut ultimates = vec_with_capacity(n_years)?;
        for row in &self.triangle_data {
            let latest_col = row.len() - 1;
            let latest_value = row[latest_col];
            let cdf_remaining = if latest_col < n_periods { cum_factors[latest_col] } else { 1.0 };
            ultimates.push(latest_value * cdf_remaining);
        }
        Ok(ultimates)
    }

    /// IBNR = ultimate − latest diagonal, floored at zero.
    pub fn ibnr_reserves(&self) -> Result<Vec<f64>> {
        let ultimates = self.ultimate_losses()?;
        let mut ibnr = vec_with_capacity(self.triangle_data.len())?;
        for (row, ult) in self.triangle_data.iter().zip(&ultimates) {
            let latest = *row.last().expect("rows are never empty");
            let mut r = ult - latest;
            if r < 0.0 {
                r = 0.0;
            }
            ibnr.push(r);
        }
        Ok(ibnr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_ladder_on_a_small_triangle() {
        let mut t = LossTriangle::new();
        t.add_origin_year(2020, vec![100.0, 150.0, 175.0]).unwrap();
        t.add_origin_year(2021, vec![110.0, 165.0]).unwrap();
        t.add_origin_year(2022, vec![120.0]).unwrap();
        let f = t.development_factors().unwrap();
        assert_eq!(f, vec![315.0 / 210.0, 175.0 / 150.0]);
        let u = t.ultimate_losses().unwrap();
        assert_eq!(u[0], 175.0);
        assert_eq!(u[1], 165.0 * (175.0 / 150.0));
        assert_eq!(u[2], 120.0 * ((315.0 / 210.0) * ((175.0 / 150.0) * 1.0)));
        let ibnr = t.ibnr_reserves().unwrap();
        assert_eq!(ibnr[0], 0.0);
    }

    #[test]
    fn zero_then_positive_column_is_refused() {
        let mut t = LossTriangle::new();
        t.add_origin_year(1, vec![0.0, 5.0]).unwrap();
        let err = t.development_factors().unwrap_err();
        assert!(err.to_string().contains("column 0 sums to zero but column 1 is positive"));
    }
}
