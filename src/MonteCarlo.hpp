#pragma once
#include <vector>

class MonteCarloSimulator {
public:
    // P&C: Simulate aggregate losses over N trials. 
    // Frequency follows Poisson(lambda)
    // Severity follows Lognormal(mu, sigma)
    static std::vector<double> simulate_aggregate_loss(int trials, double expected_frequency, double expected_severity_mu, double severity_sigma);

    // Life: Simulate total claims out of a pool of policies given a base mortality probability
    // and a stochastic mortality shock factor (Normal(1.0, shock_volatility))
    static std::vector<double> simulate_life_portfolio(int trials, int policy_count, double base_mortality_rate, double shock_volatility, double death_benefit);
};
