#include "ExposureRating.hpp"
#include <cmath>
#include <algorithm>

ExposureRating::ExposureRating(double base_limit, double b_parameter)
    : base_limit(base_limit), b_parameter(b_parameter) {
    if (!(base_limit > 0.0) || !std::isfinite(base_limit)) throw std::invalid_argument("Base limit must be finite and > 0");
    // Power-curve ILF (L/B)^(1-b): b in [0, 1] keeps it increasing and concave
    // (b = 0 linear, b = 1 flat); outside that range layer premiums turn negative.
    if (!(b_parameter >= 0.0 && b_parameter <= 1.0)) throw std::invalid_argument("b_parameter must be in [0, 1]");
}

namespace {
double std_normal_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }
}

double ExposureRating::increased_limits_factor(double target_limit) const {
    if (!(target_limit > 0.0) || !std::isfinite(target_limit)) throw std::invalid_argument("Target limit must be finite and strictly positive");
    return std::pow(target_limit / base_limit, 1.0 - b_parameter);
}

double ExposureRating::layer_premium(double ground_up_premium, double attachment, double limit) const {
    if (!(ground_up_premium >= 0.0) || !std::isfinite(ground_up_premium)) throw std::invalid_argument("Premium must be finite and non-negative");
    if (!(attachment >= 0.0) || !(limit > 0.0) || !std::isfinite(attachment) || !std::isfinite(limit))
        throw std::invalid_argument("Invalid attachment or limit (must be finite, attachment >= 0, limit > 0)");

    double ilf_at_base = increased_limits_factor(base_limit);
    // The power-curve ILF is only defined for limits > 0, but the limited expected
    // value at zero is zero for any severity curve, so a ground-up layer uses 0.
    double ilf_at_attachment = attachment > 0.0 ? increased_limits_factor(attachment) : 0.0;
    double ilf_at_limit = increased_limits_factor(attachment + limit);
    
    if (ilf_at_base == 0.0) throw std::invalid_argument("ILF Base cannot be zero");

    return ground_up_premium * (ilf_at_limit - ilf_at_attachment) / ilf_at_base;
}

double ExposureRating::burning_cost(const std::vector<double>& historical_losses, 
                                    double attachment, double limit, int exposure_years) const {
    
    if (exposure_years <= 0) throw std::invalid_argument("Exposure years must be strictly positive");
    if (!(attachment >= 0.0) || !(limit > 0.0) || !std::isfinite(attachment) || !std::isfinite(limit))
        throw std::invalid_argument("Invalid attachment or limit (must be finite, attachment >= 0, limit > 0)");

    double total_layer_losses = 0.0;
    for (double loss : historical_losses) {
        if (!(loss >= 0.0) || !std::isfinite(loss)) throw std::invalid_argument("Loss amounts must be finite and non-negative");
        double capped = std::min(loss, attachment + limit);
        double layer_loss = std::max(capped - attachment, 0.0);
        total_layer_losses += layer_loss;
    }
    return total_layer_losses / exposure_years;
}

double ExposureRating::loss_elimination_ratio(double deductible, double expected_loss, double severity_cv) const {
    if (!(expected_loss > 0.0) || !std::isfinite(expected_loss)) throw std::invalid_argument("Expected loss must be finite and positive");
    if (!(deductible >= 0.0) || std::isnan(deductible)) throw std::invalid_argument("Deductible cannot be negative or NaN");
    if (!(severity_cv > 0.0) || !std::isfinite(severity_cv)) throw std::invalid_argument("Severity CV must be finite and > 0");

    // Lognormal severity with mean E and coefficient of variation CV:
    //   sigma^2 = ln(1 + CV^2),  mu = ln E - sigma^2 / 2.
    // LER(d) = E[min(X, d)] / E[X], with the limited expected value
    //   LEV(d) = e^(mu + sigma^2/2) Phi((ln d - mu - sigma^2)/sigma) + d (1 - Phi((ln d - mu)/sigma)).
    if (deductible == 0.0) return 0.0;
    if (std::isinf(deductible)) return 1.0;
    const double sigma2 = std::log1p(severity_cv * severity_cv);
    const double sigma = std::sqrt(sigma2);
    const double mu = std::log(expected_loss) - 0.5 * sigma2;
    const double z = (std::log(deductible) - mu) / sigma;
    // LEV/E = Phi(z - sigma) + (d/E) (1 - Phi(z)); 1 - Phi(z) = Phi(-z) avoids cancellation.
    double ler = std_normal_cdf(z - sigma) + (deductible / expected_loss) * std_normal_cdf(-z);
    if (ler > 1.0) ler = 1.0;
    if (ler < 0.0) ler = 0.0;
    return ler;
}

double ExposureRating::excess_risk_load(double layer_std_dev, double risk_load_multiple) const {
    if (!(layer_std_dev >= 0.0) || !std::isfinite(layer_std_dev)) throw std::invalid_argument("Layer std_dev must be finite and non-negative");
    if (!std::isfinite(risk_load_multiple)) throw std::invalid_argument("Risk load multiple must be finite");
    return layer_std_dev * risk_load_multiple;
}
