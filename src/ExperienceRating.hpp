#pragma once

class ExperienceRating {
public:
    // Bühlmann credibility-weighted rate
    // Blends observed experience with the class expected rate
    // Result = Z * observed + (1 - Z) * expected
    static double credibility_weighted_rate(double observed_rate, double expected_rate, double credibility_z);

    // Classical limited fluctuation credibility
    // Z = min(1.0, sqrt(claim_count / k_parameter))
    // k = full credibility standard (commonly 1082 for Poisson frequency)
    static double calculate_credibility(int claim_count, double k_parameter);

    // NCCI-style Experience Modification Factor (EMF)
    // EMF = (actual_losses + ballast * expected_losses) / (expected_losses * (1 + ballast))
    static double experience_mod_factor(double actual_losses, double expected_losses, double ballast);

    // Bühlmann-Straub credibility factor
    // Z = n / (n + k) where n = exposure, k = variance ratio
    static double buhlmann_straub_credibility(double exposure, double variance_ratio);

    // Calculate expected losses from frequency and severity
    static double expected_losses(double exposure, double frequency, double severity);
};
