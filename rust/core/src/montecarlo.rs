//! `MonteCarloSimulator`: five fixed actuarial simulations.
//!
//! The parameter checks are the C++ core's after its 0.4.1 / 0.6.0 hardening, in the
//! same order and with the same messages (NaN refused everywhere, the NaN-frequency
//! infinite loop impossible, frequencies above 1e9 refused, negative-binomial counts as
//! a true gamma–Poisson mixture). The draws come from [`crate::dist`], so a seed gives
//! different numbers from 0.8.0, but the same numbers on every platform.

use crate::dist::Stream;
use crate::error::{Result, invalid, vec_with_capacity};
use crate::m;

#[derive(Debug, Clone)]
pub struct MonteCarloSimulator {
    trials: i32,
    seed: u32,
}

impl MonteCarloSimulator {
    pub fn new(trials: i32, seed: u32) -> Result<Self> {
        if trials <= 0 {
            return invalid("Trials must be strictly positive");
        }
        Ok(MonteCarloSimulator { trials, seed })
    }

    pub fn trials(&self) -> i32 {
        self.trials
    }

    pub fn seed(&self) -> u32 {
        self.seed
    }

    fn buffer(&self) -> Result<Vec<f64>> {
        vec_with_capacity(self.trials as usize)
    }

    /// P&C aggregate loss: Poisson claim count, lognormal claim sizes.
    pub fn simulate_aggregate_loss(&self, expected_frequency: f64, expected_severity_mu: f64, severity_sigma: f64) -> Result<Vec<f64>> {
        // Written as !(x >= 0) so NaN is refused too: a NaN Poisson mean never terminates.
        if !(expected_frequency >= 0.0) {
            return invalid("Expected frequency cannot be negative");
        }
        // Past this the Poisson count no longer fits in an int.
        if expected_frequency > 1e9 {
            return invalid("Expected frequency must be at most 1e9");
        }
        if !expected_severity_mu.is_finite() {
            return invalid("Severity mu must be finite");
        }
        if !(severity_sigma > 0.0) || !severity_sigma.is_finite() {
            return invalid("Severity sigma must be strictly positive");
        }
        let mut results = self.buffer()?;
        let mut s = Stream::new(self.seed);
        for _ in 0..self.trials {
            let claim_count = s.poisson(expected_frequency);
            let mut aggregate_loss = 0.0;
            for _ in 0..claim_count {
                aggregate_loss += s.lognormal(expected_severity_mu, severity_sigma);
            }
            results.push(aggregate_loss);
        }
        Ok(results)
    }

    /// Life: deaths in a pool of policies under a systemic mortality shock.
    pub fn simulate_life_portfolio(&self, policy_count: i32, base_mortality_rate: f64, shock_volatility: f64, death_benefit: f64) -> Result<Vec<f64>> {
        if policy_count < 0 {
            return invalid("Policy count cannot be negative");
        }
        if !(base_mortality_rate >= 0.0 && base_mortality_rate <= 1.0) {
            return invalid("Mortality probability must be between 0 and 1");
        }
        if !(shock_volatility >= 0.0) || !shock_volatility.is_finite() {
            return invalid("Shock volatility cannot be negative");
        }
        if !(death_benefit >= 0.0) || !death_benefit.is_finite() {
            return invalid("Death benefit must be finite and non-negative");
        }
        let mut results = self.buffer()?;
        let mut s = Stream::new(self.seed);
        let n = policy_count as u64;
        for _ in 0..self.trials {
            // Normal systemic shock with mean 1, floored at 0.
            let mut shock = s.standard_normal() * shock_volatility + 1.0;
            if shock < 0.0 {
                shock = 0.0;
            }
            let mut actual_prob = base_mortality_rate * shock;
            if actual_prob > 1.0 {
                actual_prob = 1.0;
            }
            // 0 · ∞ (a zero base rate and an overflowing shock) is no deaths, not NaN.
            if actual_prob.is_nan() {
                actual_prob = 0.0;
            }
            let deaths = s.binomial(n, actual_prob);
            results.push(deaths as f64 * death_benefit);
        }
        Ok(results)
    }

    /// Economic scenario: terminal value of a geometric Brownian motion, unit time steps.
    pub fn simulate_economic_path(&self, initial_price: f64, drift: f64, volatility: f64, time_steps: i32) -> Result<Vec<f64>> {
        if !(initial_price > 0.0) {
            return invalid("Initial price must be strictly positive");
        }
        if !(volatility >= 0.0) || !volatility.is_finite() {
            return invalid("Volatility cannot be negative");
        }
        if time_steps <= 0 {
            return invalid("Time steps must be strictly positive");
        }
        if !drift.is_finite() {
            return invalid("Drift must be finite");
        }
        if !initial_price.is_finite() {
            return invalid("Initial price must be finite");
        }
        let mut results = self.buffer()?;
        let mut s = Stream::new(self.seed);
        let dt = 1.0; // Time step size (e.g. 1 year)
        let det = (drift - 0.5 * volatility * volatility) * dt;
        let vol_sqrt_dt = volatility * m::sqrt(dt);
        for _ in 0..self.trials {
            let mut current_price = initial_price;
            for _ in 0..time_steps {
                let z = s.standard_normal();
                current_price *= m::exp(det + vol_sqrt_dt * z);
            }
            results.push(current_price);
        }
        Ok(results)
    }

    /// Health: negative-binomial claim counts (mean m, variance m + d·m²) as the
    /// gamma–Poisson mixture λ ~ Gamma(1/d, d·m), N ~ Poisson(λ); gamma claim sizes.
    pub fn simulate_health_claims(&self, expected_frequency: f64, dispersion: f64, severity_shape: f64, severity_scale: f64) -> Result<Vec<f64>> {
        if !(expected_frequency >= 0.0) {
            return invalid("Frequency cannot be negative");
        }
        if expected_frequency > 1e9 {
            return invalid("Frequency must be at most 1e9");
        }
        if !(dispersion > 0.0) || !dispersion.is_finite() {
            return invalid("Dispersion must be strictly positive");
        }
        if !(severity_shape > 0.0 && severity_scale > 0.0) || !severity_shape.is_finite() || !severity_scale.is_finite() {
            return invalid("Gamma parameters must be strictly positive");
        }
        let mut results = self.buffer()?;
        let mut s = Stream::new(self.seed);
        let mix_shape = 1.0 / dispersion;
        let mix_scale = dispersion * expected_frequency;
        // A vanishing dispersion is the Poisson limit; 1/d would overflow the gamma shape.
        let poisson_limit = !mix_shape.is_finite() || mix_shape > 1e12;
        let mix_shape = if poisson_limit { 1.0 } else { mix_shape };
        let mix_scale = if mix_scale > 0.0 { mix_scale } else { 1.0 };
        // Counts beyond this no longer fit in an int (or in memory as a loop).
        let max_count = 2e9;
        for _ in 0..self.trials {
            let mut lambda = expected_frequency;
            if expected_frequency > 0.0 && !poisson_limit {
                lambda = s.gamma(mix_shape, mix_scale);
            }
            if !(lambda > 0.0) {
                results.push(0.0);
                continue;
            }
            if lambda > 1e9 {
                return invalid("Dispersion too large for this frequency: a gamma-mixed Poisson mean exceeded 1e9 claims");
            }
            let claims = s.poisson(lambda);
            if claims as f64 > max_count {
                return invalid("Simulated claim count exceeds 2e9");
            }
            let mut total_loss = 0.0;
            for _ in 0..claims {
                total_loss += s.gamma(severity_shape, severity_scale);
            }
            results.push(total_loss);
        }
        Ok(results)
    }

    /// Catastrophe: an event with probability p, Pareto(xm, alpha) loss when it happens.
    pub fn simulate_catastrophe_loss(&self, probability_of_event: f64, pareto_xm: f64, pareto_alpha: f64) -> Result<Vec<f64>> {
        if !(probability_of_event >= 0.0 && probability_of_event <= 1.0) {
            return invalid("Probability must be [0, 1]");
        }
        if !(pareto_xm > 0.0) || !pareto_xm.is_finite() {
            return invalid("Pareto xm must be finite and strictly positive");
        }
        if !(pareto_alpha > 0.0) || !pareto_alpha.is_finite() {
            return invalid("Pareto alpha must be finite and strictly positive");
        }
        let mut results = self.buffer()?;
        let mut s = Stream::new(self.seed);
        let inv_alpha = 1.0 / pareto_alpha;
        for _ in 0..self.trials {
            if s.bernoulli(probability_of_event) {
                // Inversion: xm / (1 − U)^(1/alpha), with 1 − U in (0, 1].
                let u = s.uniform();
                results.push(pareto_xm / m::pow(1.0 - u, inv_alpha));
            } else {
                results.push(0.0);
            }
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_match_the_cpp_messages() {
        let sim = MonteCarloSimulator::new(3, 1).unwrap();
        let msg = |r: Result<Vec<f64>>| r.unwrap_err().to_string();
        assert_eq!(msg(sim.simulate_aggregate_loss(f64::NAN, 0.0, 1.0)), "Expected frequency cannot be negative");
        assert_eq!(msg(sim.simulate_aggregate_loss(1e12, 0.0, 1.0)), "Expected frequency must be at most 1e9");
        assert_eq!(msg(sim.simulate_health_claims(10.0, 0.0, 1.0, 1.0)), "Dispersion must be strictly positive");
        assert_eq!(msg(sim.simulate_economic_path(f64::INFINITY, 0.0, 0.2, 5)), "Initial price must be finite");
        assert!(MonteCarloSimulator::new(0, 1).is_err());
    }

    #[test]
    fn degenerate_cases_are_exact() {
        let sim = MonteCarloSimulator::new(5, 9).unwrap();
        assert_eq!(sim.simulate_economic_path(1.0, 0.0, 0.0, 1).unwrap(), vec![1.0; 5]);
        assert_eq!(sim.simulate_catastrophe_loss(0.0, 1.0, 1.0).unwrap(), vec![0.0; 5]);
        assert_eq!(sim.simulate_life_portfolio(1, 1.0, 0.0, 1.0).unwrap(), vec![1.0; 5]);
        assert_eq!(sim.simulate_aggregate_loss(0.0, 0.0, 1.0).unwrap(), vec![0.0; 5]);
    }

    #[test]
    fn seeded_runs_repeat() {
        let a = MonteCarloSimulator::new(1000, 77).unwrap();
        let b = MonteCarloSimulator::new(1000, 77).unwrap();
        assert_eq!(a.simulate_health_claims(3.0, 0.5, 2.0, 10.0).unwrap(), b.simulate_health_claims(3.0, 0.5, 2.0, 10.0).unwrap());
    }
}
