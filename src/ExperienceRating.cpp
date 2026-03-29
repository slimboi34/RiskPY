#include "ExperienceRating.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

double ExperienceRating::credibility_weighted_rate(double observed_rate, double expected_rate, double credibility_z) {
    if (observed_rate < 0.0) throw std::invalid_argument("Observed rate cannot be negative");
    if (expected_rate < 0.0) throw std::invalid_argument("Expected rate cannot be negative");
    
    // Clamp Z to [0, 1]
    double z = std::max(0.0, std::min(1.0, credibility_z));
    return z * observed_rate + (1.0 - z) * expected_rate;
}

double ExperienceRating::calculate_credibility(int claim_count, double k_parameter) {
    if (claim_count < 0) throw std::invalid_argument("Claim count cannot be negative");
    if (k_parameter <= 0.0) return 1.0;
    
    double z = std::sqrt(static_cast<double>(claim_count) / k_parameter);
    return std::min(1.0, z);
}

double ExperienceRating::experience_mod_factor(double actual_losses, double expected_losses, double ballast) {
    if (actual_losses < 0.0) throw std::invalid_argument("Actual losses cannot be negative");
    if (expected_losses < 0.0) throw std::invalid_argument("Expected losses cannot be negative");
    if (ballast < 0.0) throw std::invalid_argument("Ballast cannot be negative");
    
    if (expected_losses == 0.0) return 1.0;
    // NCCI formula: EMF = (A + B*E) / (E * (1 + B))
    return (actual_losses + ballast * expected_losses) / (expected_losses * (1.0 + ballast));
}

double ExperienceRating::buhlmann_straub_credibility(double exposure, double variance_ratio) {
    if (exposure < 0.0) throw std::invalid_argument("Exposure cannot be negative");
    if (variance_ratio < 0.0) throw std::invalid_argument("Variance ratio cannot be negative");
    
    if (variance_ratio == 0.0) return 1.0;
    return exposure / (exposure + variance_ratio);
}

double ExperienceRating::expected_losses(double exposure, double frequency, double severity) {
    if (exposure < 0.0) throw std::invalid_argument("Exposure cannot be negative");
    if (frequency < 0.0) throw std::invalid_argument("Frequency cannot be negative");
    if (severity < 0.0) throw std::invalid_argument("Severity cannot be negative");
    return exposure * frequency * severity;
}
