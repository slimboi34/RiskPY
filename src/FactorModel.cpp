#include "FactorModel.hpp"

FactorModel::FactorModel(double initial_base_rate) : base_rate(initial_base_rate) {}

void FactorModel::set_base_rate(double rate) {
    base_rate = rate;
}

void FactorModel::add_multiplier(const std::string& field_name, const std::string& exact_value, double multiplier) {
    exact_rules.push_back({field_name, exact_value, multiplier});
}

void FactorModel::add_numeric_band_multiplier(const std::string& field_name, double min_val, double max_val, double multiplier) {
    band_rules.push_back({field_name, min_val, max_val, multiplier});
}

double FactorModel::calculate(const std::map<std::string, DynamicField>& inputs) const {
    double final_rate = base_rate;
    
    // Apply Exact Match string rules
    for (const auto& rule : exact_rules) {
        auto it = inputs.find(rule.field_name);
        if (it != inputs.end() && std::holds_alternative<std::string>(it->second)) {
            if (std::get<std::string>(it->second) == rule.exact_value) {
                final_rate *= rule.multiplier;
            }
        }
    }
    
    // Apply Numeric Band rules
    for (const auto& rule : band_rules) {
        auto it = inputs.find(rule.field_name);
        if (it != inputs.end() && std::holds_alternative<double>(it->second)) {
            double val = std::get<double>(it->second);
            if (val >= rule.min_val && val <= rule.max_val) {
                final_rate *= rule.multiplier;
            }
        }
    }
    
    return final_rate;
}
