#include "MonteCarlo.hpp"
#include <random>
#include <cmath>
#include <stdexcept>

MonteCarloSimulator::MonteCarloSimulator(int trials, unsigned int seed) : trials(trials), seed(seed) {
    if (trials <= 0) throw std::invalid_argument("Trials must be strictly positive");
}

std::vector<double> MonteCarloSimulator::simulate_aggregate_loss(double expected_frequency, double expected_severity_mu, double severity_sigma) const {
    if (expected_frequency < 0.0) throw std::invalid_argument("Expected frequency cannot be negative");
    if (severity_sigma <= 0.0) throw std::invalid_argument("Severity sigma must be strictly positive");

    std::vector<double> results(trials, 0.0);
    
    // Seed and generators
    std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
    
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
    
    std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
    
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

std::vector<double> MonteCarloSimulator::simulate_economic_path(double initial_price, double drift, double volatility, int time_steps) const {
    if (initial_price <= 0.0) throw std::invalid_argument("Initial price must be strictly positive");
    if (volatility < 0.0) throw std::invalid_argument("Volatility cannot be negative");
    if (time_steps <= 0) throw std::invalid_argument("Time steps must be strictly positive");

    std::vector<double> results(trials, 0.0);
    std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    double dt = 1.0; // Time step size (e.g. 1 year)
    
    for (int i = 0; i < trials; ++i) {
        double current_price = initial_price;
        for (int t = 0; t < time_steps; ++t) {
            double z = normal_dist(gen);
            // Geometric Brownian Motion formula
            current_price *= std::exp((drift - 0.5 * volatility * volatility) * dt + volatility * std::sqrt(dt) * z);
        }
        results[i] = current_price;
    }
    return results;
}

std::vector<double> MonteCarloSimulator::simulate_health_claims(double expected_frequency, double dispersion, double severity_shape, double severity_scale) const {
    if (expected_frequency < 0.0) throw std::invalid_argument("Frequency cannot be negative");
    if (dispersion <= 0.0) throw std::invalid_argument("Dispersion must be strictly positive");
    if (severity_shape <= 0.0 || severity_scale <= 0.0) throw std::invalid_argument("Gamma parameters must be strictly positive");

    std::vector<double> results(trials, 0.0);
    std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
    
    // Convert (mean, dispersion) to Negative Binomial (r, p)
    // p = mean / variance = 1 / (1 + mean * dispersion)
    // r = 1 / dispersion
    
    double p = 1.0 / (1.0 + expected_frequency * dispersion);
    double r = 1.0 / dispersion;
    
    std::negative_binomial_distribution<int> freq_dist(r, p);
    std::gamma_distribution<double> sev_dist(severity_shape, severity_scale);
    
    for(int i = 0; i < trials; ++i) {
        int claims = freq_dist(gen);
        double total_loss = 0.0;
        for(int c = 0; c < claims; ++c) {
            total_loss += sev_dist(gen);
        }
        results[i] = total_loss;
    }
    return results;
}

std::vector<double> MonteCarloSimulator::simulate_catastrophe_loss(double probability_of_event, double pareto_xm, double pareto_alpha) const {
    if (probability_of_event < 0.0 || probability_of_event > 1.0) throw std::invalid_argument("Probability must be [0, 1]");
    if (pareto_xm <= 0.0) throw std::invalid_argument("Pareto xm must be strictly positive");
    if (pareto_alpha <= 0.0) throw std::invalid_argument("Pareto alpha must be strictly positive");

    std::vector<double> results(trials, 0.0);
    std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
    
    std::bernoulli_distribution event_dist(probability_of_event);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    
    for (int i = 0; i < trials; ++i) {
        if (event_dist(gen)) {
            // Event happens, draw from Pareto
            double u = uniform_dist(gen);
            // avoid 1.0 boundary for division by zero exception effectively
            if (u == 1.0) u = 0.99999999;
            double loss = pareto_xm / std::pow(1.0 - u, 1.0 / pareto_alpha);
            results[i] = loss;
        } else {
            results[i] = 0.0;
        }
    }
    
    return results;
}
