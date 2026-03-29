#include "MonteCarlo.hpp"
#include <random>
#include <cmath>
#include <stdexcept>

MonteCarloSimulator::MonteCarloSimulator(int trials) : trials(trials) {
    if (trials <= 0) throw std::invalid_argument("Trials must be strictly positive");
}

std::vector<double> MonteCarloSimulator::simulate_aggregate_loss(double expected_frequency, double expected_severity_mu, double severity_sigma) const {
    if (expected_frequency < 0.0) throw std::invalid_argument("Expected frequency cannot be negative");
    if (severity_sigma <= 0.0) throw std::invalid_argument("Severity sigma must be strictly positive");

    std::vector<double> results(trials, 0.0);
    
    // Seed and generators
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::poisson_distribution<int> freq_dist(expected_frequency);
    std::lognormal_distribution<double> sev_dist(expected_severity_mu, severity_sigma);
    
    for (int i = 0; i < trials; ++i) {
        int claim_count = freq_dist(gen);
        double aggregate_loss = 0.0;
        for (int c = 0; c < claim_count; ++c) {
            aggregate_loss += sev_dist(gen);
        }
        results[i] = aggregate_loss;
    }
    
    return results;
}

std::vector<double> MonteCarloSimulator::simulate_life_portfolio(int policy_count, double base_mortality_rate, double shock_volatility, double death_benefit) const {
    if (policy_count < 0) throw std::invalid_argument("Policy count cannot be negative");
    if (base_mortality_rate < 0.0 || base_mortality_rate > 1.0) throw std::invalid_argument("Mortality probability must be between 0 and 1");
    if (shock_volatility < 0.0) throw std::invalid_argument("Shock volatility cannot be negative");
    if (death_benefit < 0.0) throw std::invalid_argument("Death benefit cannot be negative");

    std::vector<double> results(trials, 0.0);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Normal distribution for systemic mortality shock (mean=1.0)
    std::normal_distribution<double> shock_dist(1.0, shock_volatility);
    
    for (int i = 0; i < trials; ++i) {
        double shock = shock_dist(gen);
        if (shock < 0.0) shock = 0.0; // Cannot have negative shock
        
        double actual_prob = base_mortality_rate * shock;
        if (actual_prob > 1.0) actual_prob = 1.0;
        
        std::binomial_distribution<int> claims_dist(policy_count, actual_prob);
        int deaths = claims_dist(gen);
        
        results[i] = deaths * death_benefit;
    }
    
    return results;
}
