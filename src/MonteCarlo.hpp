#pragma once
#include <vector>

class MonteCarloSimulator {
public:
    MonteCarloSimulator(int trials = 10000, unsigned int seed = 0);

    // P&C: Simulate aggregate losses over N trials. 
    std::vector<double> simulate_aggregate_loss(double expected_frequency, double expected_severity_mu, double severity_sigma) const;

    // Life: Simulate total claims out of a pool of policies
    std::vector<double> simulate_life_portfolio(int policy_count, double base_mortality_rate, double shock_volatility, double death_benefit) const;

    // Economic: Geometric Brownian Motion asset paths
    std::vector<double> simulate_economic_path(double initial_price, double drift, double volatility, int time_steps) const;

    // Health: Negative binomial frequency + Gamma severity
    std::vector<double> simulate_health_claims(double expected_frequency, double dispersion, double severity_shape, double severity_scale) const;

    // CAT / Extreme Events: Bernoulli + Pareto severity
    std::vector<double> simulate_catastrophe_loss(double probability_of_event, double pareto_xm, double pareto_alpha) const;

private:
    int trials;
    unsigned int seed;
};
