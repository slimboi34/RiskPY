#pragma once

class ExperienceRating {
public:
    ExperienceRating(double k_parameter = 1082.0, double ballast = 0.0);

    // Bühlmann credibility-weighted rate
    double credibility_weighted_rate(double observed_rate, double expected_rate, double credibility_z) const;

    // Classical limited fluctuation credibility
    double calculate_credibility(int claim_count) const;

    // NCCI-style Experience Modification Factor (EMF)
    double experience_mod_factor(double actual_losses, double expected_losses) const;

    // Bühlmann-Straub credibility factor
    double buhlmann_straub_credibility(double exposure, double variance_ratio) const;

    // Calculate expected losses from frequency and severity
    double expected_losses(double exposure, double frequency, double severity) const;

private:
    double k_parameter;
    double ballast;
};
