#include "LossTriangle.hpp"
#include <stdexcept>
#include <numeric>

void LossTriangle::add_origin_year(int year, const std::vector<double>& cumulative_payments) {
    if (year < 0) throw std::invalid_argument("Origin year cannot be negative");
    if (cumulative_payments.empty()) throw std::invalid_argument("Cumulative payments cannot be empty");
    for (double p : cumulative_payments) {
        if (p < 0.0) throw std::invalid_argument("Cumulative payments cannot be negative");
    }
    origin_years.push_back(year);
    triangle_data.push_back(cumulative_payments);
}

int LossTriangle::get_origin_year_count() const {
    return static_cast<int>(origin_years.size());
}

int LossTriangle::get_development_period_count() const {
    if (triangle_data.empty()) return 0;
    int max_cols = 0;
    for (const auto& row : triangle_data) {
        if (static_cast<int>(row.size()) > max_cols)
            max_cols = static_cast<int>(row.size());
    }
    return max_cols;
}

std::vector<double> LossTriangle::get_development_factors() const {
    // Chain ladder age-to-age factors
    // For each development period d, compute the weighted average
    // sum(C(i,d+1)) / sum(C(i,d)) across all origin years with data at both d and d+1
    
    int n_periods = get_development_period_count();
    if (n_periods <= 1) return {};
    
    std::vector<double> factors;
    
    for (int d = 0; d < n_periods - 1; ++d) {
        double sum_next = 0.0;
        double sum_curr = 0.0;
        
        for (const auto& row : triangle_data) {
            if (static_cast<int>(row.size()) > d + 1) {
                sum_curr += row[d];
                sum_next += row[d + 1];
            }
        }
        
        if (sum_curr > 0.0) {
            factors.push_back(sum_next / sum_curr);
        } else if (sum_curr == 0.0 && sum_next == 0.0) {
            factors.push_back(1.0);
        } else {
            // If sum_curr is 0 but sum_next > 0, mathematically infinite. Protect against it.
            factors.push_back(1.0);
        }
    }
    
    return factors;
}

std::vector<double> LossTriangle::get_ultimate_losses() const {
    // Project ultimate losses by applying remaining development factors
    // to each origin year's latest known value
    
    std::vector<double> dev_factors = get_development_factors();
    int n_periods = get_development_period_count();
    int n_years = static_cast<int>(triangle_data.size());
    
    if (n_years == 0) return {};
    
    // Compute cumulative development factors from each period to ultimate
    // CDF(d) = prod(LDF(d), LDF(d+1), ..., LDF(n-1))
    std::vector<double> cum_factors(n_periods, 1.0);
    for (int d = n_periods - 2; d >= 0; --d) {
        if (d < static_cast<int>(dev_factors.size())) {
            cum_factors[d] = dev_factors[d] * cum_factors[d + 1];
        }
    }
    
    std::vector<double> ultimates(n_years, 0.0);
    
    for (int i = 0; i < n_years; ++i) {
        int latest_col = static_cast<int>(triangle_data[i].size()) - 1;
        double latest_value = triangle_data[i][latest_col];
        
        // Apply cumulative development factor from latest column to ultimate
        double cdf_remaining = 1.0;
        for (int d = latest_col; d < n_periods - 1 && d < static_cast<int>(dev_factors.size()); ++d) {
            cdf_remaining *= dev_factors[d];
        }
        
        ultimates[i] = latest_value * cdf_remaining;
    }
    
    return ultimates;
}

std::vector<double> LossTriangle::get_ibnr_reserves() const {
    std::vector<double> ultimates = get_ultimate_losses();
    int n_years = static_cast<int>(triangle_data.size());
    
    std::vector<double> ibnr(n_years, 0.0);
    
    for (int i = 0; i < n_years; ++i) {
        double latest = triangle_data[i].back();
        ibnr[i] = ultimates[i] - latest;
        if (ibnr[i] < 0.0) ibnr[i] = 0.0;
    }
    
    return ibnr;
}
