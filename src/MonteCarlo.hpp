#pragma once
#include <vector>

class MonteCarloSimulator {
public:
    MonteCarloSimulator(int trials = 10000);

    // P&C: Simulate aggregate losses over N trials. 
    std::vector<double> simulate_aggregate_loss(double expected_frequency, double expected_severity_mu, double severity_sigma) const;

    // Life: Simulate total claims out of a pool of policies
    std::vector<double> simulate_life_portfolio(int policy_count, double base_mortality_rate, double shock_volatility, double death_benefit) const;

private:
    int trials;
};
