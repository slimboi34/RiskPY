#include "RateAnalyzer.hpp"
#include <cmath>
#include <stdexcept>

RateAnalyzer::RateAnalyzer(double target_combined_ratio, double expense_ratio)
    : target_combined_ratio(target_combined_ratio), expense_ratio(expense_ratio) {
    if (target_combined_ratio <= 0.0) throw std::invalid_argument("Target combined ratio must be positive");
    if (expense_ratio < 0.0 || expense_ratio >= 1.0) throw std::invalid_argument("Expense ratio must be between 0 and 1");
}

double RateAnalyzer::on_level_factor(const std::vector<double>& rate_changes) const {
    double factor = 1.0;
    for (double change : rate_changes) {
        if (change <= -1.0) throw std::invalid_argument("Rate changes cannot be <= -100%");
        factor *= (1.0 + change);
    }
    return factor;
}

double RateAnalyzer::rate_change_impact(double old_premium, double new_premium) const {
    if (old_premium <= 0.0) throw std::invalid_argument("Old premium must be > 0");
    if (new_premium < 0.0) throw std::invalid_argument("New premium cannot be negative");
    return (new_premium / old_premium) - 1.0;
}

double RateAnalyzer::combined_ratio(double loss_ratio) const {
    if (loss_ratio < 0.0) throw std::invalid_argument("Loss ratio cannot be negative");
    return loss_ratio + expense_ratio;
}

double RateAnalyzer::required_rate_change(double current_loss_ratio) const {
    if (current_loss_ratio < 0.0) throw std::invalid_argument("Loss ratio cannot be negative");
    
    double plr = target_combined_ratio - expense_ratio;
    if (plr <= 0.0) throw std::invalid_argument("Permissible loss ratio would mathematically result in division by zero/negatives");
    
    return (current_loss_ratio / plr) - 1.0;
}

std::vector<double> RateAnalyzer::on_level_premiums(const std::vector<double>& earned_premiums, 
                                                    const std::vector<double>& rate_changes) const {
    if (earned_premiums.size() != rate_changes.size()) {
        throw std::invalid_argument("Mismatched arrays: Prem and Rate vectors must be same size");
    }

    std::vector<double> on_leveled;
    on_leveled.reserve(earned_premiums.size());

    double cumulative_factor = 1.0;
    for (int i = rate_changes.size() - 1; i >= 0; --i) {
        if (rate_changes[i] <= -1.0) throw std::invalid_argument("Rate change cannot be <= -100%");
        if (earned_premiums[i] < 0.0) throw std::invalid_argument("Premium cannot be negative");
        
        on_leveled.insert(on_leveled.begin(), earned_premiums[i] * cumulative_factor);
        cumulative_factor *= (1.0 + rate_changes[i]);
    }
    
    return on_leveled;
}

double RateAnalyzer::permissible_loss_ratio() const {
    return target_combined_ratio - expense_ratio;
}

double RateAnalyzer::trend_factor(double annual_trend, double years) const {
    if (annual_trend <= -1.0) throw std::invalid_argument("Trend cannot be <= -100%");
    if (years < 0.0) throw std::invalid_argument("Years cannot be negative");
    
    return std::pow(1.0 + annual_trend, years);
}

double RateAnalyzer::loss_cost_rate(double incurred_losses, double earned_exposure) const {
    if (earned_exposure <= 0.0) throw std::invalid_argument("Exposure must be > 0");
    if (incurred_losses < 0.0) throw std::invalid_argument("Losses cannot be negative");
    
    return incurred_losses / earned_exposure;
}
