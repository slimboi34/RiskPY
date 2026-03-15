#pragma once
#include <vector>

class RateAnalyzer {
public:
    // Compute on-level factor from a history of rate changes
    // Each rate change is a decimal (e.g., 0.05 = +5%, -0.03 = -3%)
    // On-level factor = product of (1 / (1 + rate_change_i)) for all periods after the target
    static double on_level_factor(const std::vector<double>& rate_changes);

    // Simple percentage rate change between two premiums
    static double rate_change_impact(double old_premium, double new_premium);

    // Combined ratio = loss ratio + expense ratio
    static double combined_ratio(double loss_ratio, double expense_ratio);

    // Required rate change to achieve a target combined ratio
    // Assumes expense ratio stays constant
    static double required_rate_change(double target_combined_ratio, double current_loss_ratio, double expense_ratio);

    // Earned premium on-leveling for a book of business
    // Takes historical earned premiums and corresponding rate changes
    // Returns on-leveled premiums at current rate level
    static std::vector<double> on_level_premiums(const std::vector<double>& earned_premiums, 
                                                  const std::vector<double>& rate_changes);

    // Calculate permissible loss ratio given target combined ratio and expense ratio
    static double permissible_loss_ratio(double target_combined_ratio, double expense_ratio);

    // Trend factor projection: (1 + annual_trend)^years
    static double trend_factor(double annual_trend, double years);

    // Loss cost rate = pure premium / exposure
    static double loss_cost_rate(double incurred_losses, double earned_exposure);
};
