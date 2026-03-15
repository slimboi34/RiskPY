#include "RateAnalyzer.hpp"
#include <cmath>

double RateAnalyzer::on_level_factor(const std::vector<double>& rate_changes) {
    // On-level factor brings historical premiums to the CURRENT rate level
    // If rate changes are [+5%, +3%, +10%] (oldest to newest):
    //   Premiums earned during period 1 need to be multiplied by (1.05)(1.03)(1.10)
    //   Premiums earned during period 2 need to be multiplied by (1.03)(1.10)
    //   Premiums earned during period 3 need to be multiplied by (1.10)
    //   Current period premiums = 1.0
    // This function returns the CUMULATIVE on-level factor for the OLDEST period
    
    double factor = 1.0;
    for (const auto& change : rate_changes) {
        factor *= (1.0 + change);
    }
    return factor;
}

double RateAnalyzer::rate_change_impact(double old_premium, double new_premium) {
    if (old_premium == 0.0) return 0.0;
    return (new_premium - old_premium) / old_premium;
}

double RateAnalyzer::combined_ratio(double loss_ratio, double expense_ratio) {
    return loss_ratio + expense_ratio;
}

double RateAnalyzer::required_rate_change(double target_combined_ratio, double current_loss_ratio, double expense_ratio) {
    // Target CR = target_LR + expense_ratio
    // target_LR = target_CR - expense_ratio  
    // If current LR is above target LR, we need a rate increase
    // Required rate change = (current_LR / target_LR) - 1.0
    double target_lr = target_combined_ratio - expense_ratio;
    if (target_lr <= 0.0) return 0.0;
    return (current_loss_ratio / target_lr) - 1.0;
}

std::vector<double> RateAnalyzer::on_level_premiums(const std::vector<double>& earned_premiums,
                                                     const std::vector<double>& rate_changes) {
    // Bring all historical premiums to the current rate level
    // For period i, the on-level factor = product of (1 + rate_change_j) for j = i to n-1
    
    int n = static_cast<int>(earned_premiums.size());
    std::vector<double> result(n, 0.0);
    
    // Build cumulative factors from right to left
    std::vector<double> cum_factors(n, 1.0);
    for (int i = n - 2; i >= 0; --i) {
        if (i + 1 < static_cast<int>(rate_changes.size())) {
            cum_factors[i] = cum_factors[i + 1] * (1.0 + rate_changes[i + 1]);
        } else {
            cum_factors[i] = cum_factors[i + 1];
        }
    }
    
    for (int i = 0; i < n; ++i) {
        result[i] = earned_premiums[i] * cum_factors[i];
    }
    
    return result;
}

double RateAnalyzer::permissible_loss_ratio(double target_combined_ratio, double expense_ratio) {
    return target_combined_ratio - expense_ratio;
}

double RateAnalyzer::trend_factor(double annual_trend, double years) {
    return std::pow(1.0 + annual_trend, years);
}

double RateAnalyzer::loss_cost_rate(double incurred_losses, double earned_exposure) {
    if (earned_exposure <= 0.0) return 0.0;
    return incurred_losses / earned_exposure;
}
