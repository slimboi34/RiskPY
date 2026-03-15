#pragma once
#include <map>
#include <string>
#include <variant>
#include <vector>

using DynamicField = std::variant<double, std::string>;

class FactorModel {
public:
    FactorModel(double initial_base_rate = 0.0);
    
    void set_base_rate(double rate);
    void add_multiplier(const std::string& field_name, const std::string& exact_value, double multiplier);
    void add_numeric_band_multiplier(const std::string& field_name, double min_val, double max_val, double multiplier);
    
    double calculate(const std::map<std::string, DynamicField>& inputs) const;

private:
    double base_rate;
    
    struct ExactMatchRule {
        std::string field_name;
        std::string exact_value;
        double multiplier;
    };
    
    struct NumericBandRule {
        std::string field_name;
        double min_val;
        double max_val;
        double multiplier;
    };
    
    std::vector<ExactMatchRule> exact_rules;
    std::vector<NumericBandRule> band_rules;
};
