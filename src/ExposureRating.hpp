#pragma once
#include <vector>

class ExposureRating {
public:
    // Calculate Increased Limits Factor using a Pareto severity curve
    // ILF(limit) = 1 - (base_limit / limit)^b  /  1 - (base_limit / base_limit)^b
    // Simplified: (limit / base_limit)^(1-b) for practical use
    static double increased_limits_factor(double base_limit, double target_limit, double b_parameter);

    // Price an excess layer using ILF differencing
    // Layer premium = ground_up_premium * (ILF(attachment + limit) - ILF(attachment)) / ILF(base)
    static double layer_premium(double ground_up_premium, double attachment, double limit, 
                                 double ilf_at_base, double ilf_at_attachment, double ilf_at_limit);

    // Historical burning cost for a specific layer
    // Sum of (min(loss, attachment + limit) - attachment)+ / exposure_years
    static double burning_cost(const std::vector<double>& historical_losses, 
                                double attachment, double limit, int exposure_years);

    // Calculate the Lee diagram (loss elimination ratio)
    // LER(d) = E[min(X, d)] / E[X] — proportion of losses eliminated by deductible d
    static double loss_elimination_ratio(double deductible, double expected_loss, double severity_cv);

    // Swiss Re style risk load for an excess layer
    // Risk load = std_dev_of_layer_losses * risk_load_multiple
    static double excess_risk_load(double layer_std_dev, double risk_load_multiple);
};
