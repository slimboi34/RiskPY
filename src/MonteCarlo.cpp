#include "MonteCarlo.hpp"
#include <random>
#include <numeric>
#include <stdexcept>

std::vector<double> MonteCarloSimulator::simulate_aggregate_loss(int trials, double expected_frequency, double expected_severity_mu, double severity_sigma) {
    if (trials <= 0) throw std::invalid_argument("Trials must be positive");
    if (expected_frequency < 0.0) throw std::invalid_argument("Expected frequency cannot be negative");
    if (severity_sigma <= 0.0) throw std::invalid_argument("Severity sigma must be strictly positive");
    
    std::vector<double> results(trials, 0.0);
    
    // Seed the random number engine
    std::mt19937 generator(1337); // Fixed seed for reproducibility in testing, use std::random_device otherwise
    
    std::poisson_distribution<int> freq_dist(expected_frequency);
    std::lognormal_distribution<double> sev_dist(expected_severity_mu, severity_sigma);
    
    for (int i = 0; i < trials; ++i) {
        int num_claims = freq_dist(generator);
        double total_loss = 0.0;
        
        for (int c = 0; c < num_claims; ++c) {
            total_loss += sev_dist(generator);
        }
        
        results[i] = total_loss;
    }
    
    return results;
}

std::vector<double> MonteCarloSimulator::simulate_life_portfolio(int trials, int policy_count, double base_mortality_rate, double shock_volatility, double death_benefit) {
    if (trials <= 0) throw std::invalid_argument("Trials must be positive");
    if (policy_count < 0) throw std::invalid_argument("Policy count cannot be negative");
    if (base_mortality_rate < 0.0 || base_mortality_rate > 1.0) throw std::invalid_argument("Base mortality rate must be between 0 and 1");
    if (shock_volatility < 0.0) throw std::invalid_argument("Shock volatility cannot be negative");
    if (death_benefit < 0.0) throw std::invalid_argument("Death benefit cannot be negative");
    std::vector<double> results(trials, 0.0);
    
    std::mt19937 generator(1337);
    std::normal_distribution<double> shock_dist(1.0, shock_volatility);
    
    for (int i = 0; i < trials; ++i) {
        // Apply a random systemic shock to the overall mortality rate for this simulated year
        double current_mortality = base_mortality_rate * shock_dist(generator);
        if (current_mortality < 0.0) current_mortality = 0.0;
        if (current_mortality > 1.0) current_mortality = 1.0;
        
        std::binomial_distribution<int> claim_dist(policy_count, current_mortality);
        int deaths = claim_dist(generator);
        
        results[i] = deaths * death_benefit;
    }
    
    return results;
}
