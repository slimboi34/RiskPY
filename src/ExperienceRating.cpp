#include "ExperienceRating.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

ExperienceRating::ExperienceRating(double k_parameter, double ballast)
    : k_parameter(k_parameter), ballast(ballast) {
    if (!(k_parameter > 0.0) || !std::isfinite(k_parameter)) throw std::invalid_argument("k_parameter must be finite and > 0");
    if (!(ballast >= 0.0) || !std::isfinite(ballast)) throw std::invalid_argument("ballast must be finite and non-negative");
}

double ExperienceRating::credibility_weighted_rate(double observed_rate, double expected_rate, double credibility_z) const {
    if (!(credibility_z >= 0.0 && credibility_z <= 1.0)) throw std::invalid_argument("Credibility Z must be between 0 and 1");
    if (!(observed_rate >= 0.0) || !(expected_rate >= 0.0) || !std::isfinite(observed_rate) || !std::isfinite(expected_rate)) throw std::invalid_argument("Rates must be finite and non-negative");
    
    return (credibility_z * observed_rate) + ((1.0 - credibility_z) * expected_rate);
}

double ExperienceRating::calculate_credibility(int claim_count) const {
    if (claim_count < 0) throw std::invalid_argument("Claim count cannot be negative");
    
    double z = std::sqrt(static_cast<double>(claim_count) / k_parameter);
    return std::min(1.0, z);
}

double ExperienceRating::experience_mod_factor(double actual_losses, double expected_losses) const {
    if (!(actual_losses >= 0.0) || !(expected_losses >= 0.0) || !std::isfinite(actual_losses) || !std::isfinite(expected_losses)) throw std::invalid_argument("Losses must be finite and non-negative");
    if (expected_losses == 0.0) throw std::invalid_argument("Expected losses cannot be zero");

    return (actual_losses + ballast * expected_losses) / (expected_losses * (1.0 + ballast));
}

double ExperienceRating::buhlmann_straub_credibility(double exposure, double variance_ratio) const {
    if (!(exposure >= 0.0) || !std::isfinite(exposure)) throw std::invalid_argument("Exposure must be finite and non-negative");
    if (!(variance_ratio >= 0.0) || !std::isfinite(variance_ratio)) throw std::invalid_argument("Variance ratio must be finite and non-negative");
    if (exposure + variance_ratio == 0.0) throw std::invalid_argument("Exposure and variance ratio cannot both be zero (credibility undefined)");
    
    return exposure / (exposure + variance_ratio);
}

double ExperienceRating::expected_losses(double exposure, double frequency, double severity) const {
    if (!(exposure >= 0.0) || !(frequency >= 0.0) || !(severity >= 0.0) || !std::isfinite(exposure) || !std::isfinite(frequency) || !std::isfinite(severity)) throw std::invalid_argument("Inputs must be finite and non-negative");
    
    return exposure * frequency * severity;
}
