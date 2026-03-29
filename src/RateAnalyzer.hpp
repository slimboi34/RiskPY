#pragma once
#include <vector>

class RateAnalyzer {
public:
    RateAnalyzer(double target_combined_ratio = 1.0, double expense_ratio = 0.30);

    // Compute on-level factor from a history of rate changes
    double on_level_factor(const std::vector<double>& rate_changes) const;

    // Simple percentage rate change between two premiums
    double rate_change_impact(double old_premium, double new_premium) const;

    // Combined ratio = loss ratio + expense ratio
    double combined_ratio(double loss_ratio) const;

    // Required rate change to achieve a target combined ratio
    double required_rate_change(double current_loss_ratio) const;

    // Earned premium on-leveling for a book of business
    std::vector<double> on_level_premiums(const std::vector<double>& earned_premiums, 
                                          const std::vector<double>& rate_changes) const;

    // Calculate permissible loss ratio given target combined ratio and expense ratio
    double permissible_loss_ratio() const;

    // Trend factor projection: (1 + annual_trend)^years
    double trend_factor(double annual_trend, double years) const;

    // Loss cost rate = pure premium / exposure
    double loss_cost_rate(double incurred_losses, double earned_exposure) const;

private:
    double target_combined_ratio;
    double expense_ratio;
};
