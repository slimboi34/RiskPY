#pragma once
#include <vector>
#include <string>

class LossTriangle {
public:
    LossTriangle() = default;

    // Add a row of cumulative payments for an origin year
    void add_origin_year(int year, const std::vector<double>& cumulative_payments);

    // Calculate age-to-age (link ratio) development factors
    std::vector<double> get_development_factors() const;

    // Project ultimate losses using the chain ladder method
    std::vector<double> get_ultimate_losses() const;

    // Calculate IBNR reserves (ultimate - latest diagonal)
    std::vector<double> get_ibnr_reserves() const;

    // Get the number of origin years
    int get_origin_year_count() const;

    // Get the number of development periods
    int get_development_period_count() const;

private:
    std::vector<int> origin_years;
    std::vector<std::vector<double>> triangle_data; // rows = origin years, cols = development periods
};
