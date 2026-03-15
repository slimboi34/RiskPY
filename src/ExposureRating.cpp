#include "ExposureRating.hpp"
#include <cmath>
#include <algorithm>

double ExposureRating::increased_limits_factor(double base_limit, double target_limit, double b_parameter) {
    // ILF using a truncated Pareto severity model
    // ILF(L) = [1 - (base/L)^(b-1)] / [1 - (base/base)^(b-1)]
    // For b > 1, the ILF increases with limit
    if (base_limit <= 0.0 || target_limit <= 0.0) return 1.0;
    if (target_limit <= base_limit) return 1.0;
    
    // Using the standard actuarial ILF formula:
    // ILF = (target_limit / base_limit)^(1 - b)  for single-parameter Pareto
    // When b < 1: severity is very heavy-tailed
    // When b = 0.5: moderate tail
    return std::pow(target_limit / base_limit, 1.0 - b_parameter);
}

double ExposureRating::layer_premium(double ground_up_premium, double attachment, double limit,
                                      double ilf_at_base, double ilf_at_attachment, double ilf_at_limit) {
    // Layer premium = ground_up * (ILF(att + limit) - ILF(att)) / ILF(base)
    if (ilf_at_base <= 0.0) return 0.0;
    return ground_up_premium * (ilf_at_limit - ilf_at_attachment) / ilf_at_base;
}

double ExposureRating::burning_cost(const std::vector<double>& historical_losses,
                                     double attachment, double limit, int exposure_years) {
    if (exposure_years <= 0) return 0.0;
    
    double total_layer_loss = 0.0;
    double cap = attachment + limit;
    
    for (double loss : historical_losses) {
        if (loss > attachment) {
            // Loss in the layer = min(loss, att + lim) - att
            double layer_loss = std::min(loss, cap) - attachment;
            total_layer_loss += layer_loss;
        }
    }
    
    return total_layer_loss / static_cast<double>(exposure_years);
}

double ExposureRating::loss_elimination_ratio(double deductible, double expected_loss, double severity_cv) {
    // Simplified LER using exponential severity assumption
    // LER(d) ≈ 1 - exp(-d / mean_severity)
    // For more sophisticated models, use severity_cv to adjust
    if (expected_loss <= 0.0) return 0.0;
    if (deductible <= 0.0) return 0.0;
    
    double mean_severity = expected_loss;
    // Adjust using CV (coefficient of variation)
    double adjusted_mean = mean_severity / (1.0 + severity_cv * severity_cv);
    
    double ler = 1.0 - std::exp(-deductible / adjusted_mean);
    return std::max(0.0, std::min(1.0, ler));
}

double ExposureRating::excess_risk_load(double layer_std_dev, double risk_load_multiple) {
    return layer_std_dev * risk_load_multiple;
}
