#pragma once
#include <vector>
#include <stdexcept>

class ExposureRating {
public:
    // Initialize with standard parameters
    ExposureRating(double base_limit = 100000.0, double b_parameter = 1.0);

    double increased_limits_factor(double target_limit) const;

    double layer_premium(double ground_up_premium, double attachment, double limit) const;

    double burning_cost(const std::vector<double>& historical_losses, 
                                double attachment, double limit, int exposure_years) const;

    double loss_elimination_ratio(double deductible, double expected_loss, double severity_cv) const;

    double excess_risk_load(double layer_std_dev, double risk_load_multiple) const;

private:
    double base_limit;
    double b_parameter;
};
