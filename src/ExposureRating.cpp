#include "ExposureRating.hpp"
#include <cmath>
#include <algorithm>

ExposureRating::ExposureRating(double base_limit, double b_parameter)
    : base_limit(base_limit), b_parameter(b_parameter) {
    if (base_limit <= 0.0) throw std::invalid_argument("Base limit must be > 0");
}

double ExposureRating::increased_limits_factor(double target_limit) const {
    if (target_limit <= 0.0) throw std::invalid_argument("Target limit must be strictly positive");
    return std::pow(target_limit / base_limit, 1.0 - b_parameter);
}

double ExposureRating::layer_premium(double ground_up_premium, double attachment, double limit) const {
    if (ground_up_premium < 0.0) throw std::invalid_argument("Premium cannot be negative");
    if (attachment < 0.0 || limit <= 0.0) throw std::invalid_argument("Invalid attachment or limit");
    
    double ilf_at_base = increased_limits_factor(base_limit);
    double ilf_at_attachment = increased_limits_factor(attachment);
    double ilf_at_limit = increased_limits_factor(attachment + limit);
    
    if (ilf_at_base == 0.0) throw std::invalid_argument("ILF Base cannot be zero");

    return ground_up_premium * (ilf_at_limit - ilf_at_attachment) / ilf_at_base;
}

double ExposureRating::burning_cost(const std::vector<double>& historical_losses, 
                                    double attachment, double limit, int exposure_years) const {
    
    if (exposure_years <= 0) throw std::invalid_argument("Exposure years must be strictly positive");
    if (attachment < 0.0 || limit <= 0.0) throw std::invalid_argument("Invalid attachment or limit");

    double total_layer_losses = 0.0;
    for (double loss : historical_losses) {
        if (loss < 0.0) throw std::invalid_argument("Loss amounts cannot be negative");
        double capped = std::min(loss, attachment + limit);
        double layer_loss = std::max(capped - attachment, 0.0);
        total_layer_losses += layer_loss;
    }
    return total_layer_losses / exposure_years;
}

double ExposureRating::loss_elimination_ratio(double deductible, double expected_loss, double severity_cv) const {
    if (expected_loss <= 0.0) throw std::invalid_argument("Expected loss must be positive");
    if (deductible < 0.0) throw std::invalid_argument("Deductible cannot be negative");
    if (severity_cv <= 0.0) throw std::invalid_argument("Severity CV must be > 0");

    double sigma2 = std::log(1.0 + severity_cv * severity_cv);
    double std_dev = std::sqrt(sigma2);
    if (std_dev == 0.0) throw std::invalid_argument("Calculated std_dev is zero");

    return 1.0 - std::exp(-deductible / (expected_loss * std_dev));
}

double ExposureRating::excess_risk_load(double layer_std_dev, double risk_load_multiple) const {
    if (layer_std_dev < 0.0) throw std::invalid_argument("Layer std_dev cannot be negative");
    return layer_std_dev * risk_load_multiple;
}
