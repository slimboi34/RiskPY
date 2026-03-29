#include "ExperienceRating.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

ExperienceRating::ExperienceRating(double k_parameter, double ballast)
    : k_parameter(k_parameter), ballast(ballast) {
    if (k_parameter <= 0.0) throw std::invalid_argument("k_parameter must be > 0");
    if (ballast < 0.0) throw std::invalid_argument("ballast cannot be negative");
}

double ExperienceRating::credibility_weighted_rate(double observed_rate, double expected_rate, double credibility_z) const {
    if (credibility_z < 0.0 || credibility_z > 1.0) throw std::invalid_argument("Credibility Z must be between 0 and 1");
    if (observed_rate < 0.0 || expected_rate < 0.0) throw std::invalid_argument("Rates cannot be negative");
    
    return (credibility_z * observed_rate) + ((1.0 - credibility_z) * expected_rate);
}

double ExperienceRating::calculate_credibility(int claim_count) const {
    if (claim_count < 0) throw std::invalid_argument("Claim count cannot be negative");
    
    double z = std::sqrt(static_cast<double>(claim_count) / k_parameter);
    return std::min(1.0, z);
}

double ExperienceRating::experience_mod_factor(double actual_losses, double expected_losses) const {
    if (actual_losses < 0.0 || expected_losses < 0.0) throw std::invalid_argument("Losses cannot be negative");
    if (expected_losses == 0.0) throw std::invalid_argument("Expected losses cannot be zero");

    return (actual_losses + ballast * expected_losses) / (expected_losses * (1.0 + ballast));
}

double ExperienceRating::buhlmann_straub_credibility(double exposure, double variance_ratio) const {
    if (exposure < 0.0) throw std::invalid_argument("Exposure cannot be negative");
    if (variance_ratio < 0.0) throw std::invalid_argument("Variance ratio cannot be negative");
    
    return exposure / (exposure + variance_ratio);
}

double ExperienceRating::expected_losses(double exposure, double frequency, double severity) const {
    if (exposure < 0.0 || frequency < 0.0 || severity < 0.0) throw std::invalid_argument("Inputs cannot be negative");
    
    return exposure * frequency * severity;
}
