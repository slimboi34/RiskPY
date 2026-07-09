#include "FactorModel.hpp"
#include <cmath>
#include <stdexcept>

FactorModel::FactorModel(double initial_base_rate) : base_rate(initial_base_rate) {
    if (!std::isfinite(initial_base_rate)) {
        throw std::invalid_argument("base_rate must be finite");
    }
}

void FactorModel::set_base_rate(double rate) {
    if (!std::isfinite(rate)) {
        throw std::invalid_argument("base_rate must be finite");
    }
    base_rate = rate;
}

void FactorModel::add_multiplier(const std::string& field_name, const std::string& exact_value, double multiplier) {
    if (field_name.empty()) {
        throw std::invalid_argument("field_name cannot be empty");
    }
    if (!std::isfinite(multiplier)) {
        throw std::invalid_argument("multiplier must be finite");
    }
    exact_rules[field_name].push_back({exact_value, multiplier});
}

void FactorModel::add_numeric_band_multiplier(const std::string& field_name, double min_val, double max_val, double multiplier) {
    if (field_name.empty()) {
        throw std::invalid_argument("field_name cannot be empty");
    }
    if (!std::isfinite(min_val) || !std::isfinite(max_val) || !std::isfinite(multiplier)) {
        throw std::invalid_argument("band bounds and multiplier must be finite");
    }
    if (min_val > max_val) {
        throw std::invalid_argument("band min_val cannot exceed max_val");
    }
    band_rules[field_name].push_back({min_val, max_val, multiplier});
}

double FactorModel::calculate(const std::map<std::string, DynamicField>& inputs) const {
    double final_rate = base_rate;

    // Iterate only the input fields and look up rules for each (preserves multiplicative stacking)
    for (const auto& [field_name, value] : inputs) {
        auto exact_it = exact_rules.find(field_name);
        if (exact_it != exact_rules.end() && std::holds_alternative<std::string>(value)) {
            const std::string& str_val = std::get<std::string>(value);
            for (const auto& rule : exact_it->second) {
                if (str_val == rule.exact_value) {
                    final_rate *= rule.multiplier;
                }
            }
        }

        auto band_it = band_rules.find(field_name);
        if (band_it != band_rules.end() && std::holds_alternative<double>(value)) {
            double val = std::get<double>(value);
            for (const auto& rule : band_it->second) {
                if (val >= rule.min_val && val <= rule.max_val) {
                    final_rate *= rule.multiplier;
                }
            }
        }
    }

    return final_rate;
}
