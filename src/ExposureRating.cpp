#include "ExposureRating.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

double ExposureRating::increased_limits_factor(double base_limit, double target_limit, double b_parameter) {
    if (base_limit <= 0.0) throw std::invalid_argument("Base limit must be strictly positive");
    if (target_limit <= 0.0) throw std::invalid_argument("Target limit must be strictly positive");
    if (target_limit <= base_limit) return 1.0;
    
    return std::pow(target_limit / base_limit, 1.0 - b_parameter);
}

double ExposureRating::layer_premium(double ground_up_premium, double attachment, double limit,
                                      double ilf_at_base, double ilf_at_attachment, double ilf_at_limit) {
    if (ilf_at_base <= 0.0) throw std::invalid_argument("ILF at base must be > 0");
    if (ground_up_premium < 0.0) throw std::invalid_argument("Ground up premium cannot be negative");
    if (limit <= 0.0) throw std::invalid_argument("Layer limit must be > 0");
    if (attachment < 0.0) throw std::invalid_argument("Attachment point cannot be negative");
    
    return ground_up_premium * (ilf_at_limit - ilf_at_attachment) / ilf_at_base;
}

double ExposureRating::burning_cost(const std::vector<double>& historical_losses,
                                     double attachment, double limit, int exposure_years) {
    if (exposure_years <= 0) throw std::invalid_argument("Exposure years must be > 0");
    if (limit <= 0.0) throw std::invalid_argument("Layer limit must be > 0");
    if (attachment < 0.0) throw std::invalid_argument("Attachment point cannot be negative");
    
    double total_layer_loss = 0.0;
    double cap = attachment + limit;
    
    for (double loss : historical_losses) {
        if (loss < 0.0) throw std::invalid_argument("Historical losses cannot be negative");
        if (loss > attachment) {
            double layer_loss = std::min(loss, cap) - attachment;
            total_layer_loss += layer_loss;
        }
    }
    
    return total_layer_loss / static_cast<double>(exposure_years);
}

double ExposureRating::loss_elimination_ratio(double deductible, double expected_loss, double severity_cv) {
    if (expected_loss <= 0.0) throw std::invalid_argument("Expected loss must be > 0");
    if (deductible <= 0.0) return 0.0;
    if (severity_cv <= 0.0) throw std::invalid_argument("Severity CV must be > 0");
    
    double mean_severity = expected_loss;
    double adjusted_mean = mean_severity / (1.0 + severity_cv * severity_cv);
    
    double ler = 1.0 - std::exp(-deductible / adjusted_mean);
    return std::max(0.0, std::min(1.0, ler));
}

double ExposureRating::excess_risk_load(double layer_std_dev, double risk_load_multiple) {
    if (layer_std_dev < 0.0) throw std::invalid_argument("Layer std dev cannot be negative");
    return layer_std_dev * risk_load_multiple;
}
