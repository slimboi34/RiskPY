#include "ActuarialMath.hpp"
#include <cmath>
#include <stdexcept>

double ActuarialMath::present_value(double rate, int periods, double payment) {
    if (!(rate > -1.0) || !std::isfinite(rate)) throw std::invalid_argument("Rate must be finite and > -1.0");
    if (rate == 0.0) return payment * periods;
    return payment * ((1.0 - std::pow(1.0 + rate, -periods)) / rate);
}

double ActuarialMath::future_value(double rate, int periods, double payment) {
    if (!(rate > -1.0) || !std::isfinite(rate)) throw std::invalid_argument("Rate must be finite and > -1.0");
    if (rate == 0.0) return payment * periods;
    return payment * ((std::pow(1.0 + rate, periods) - 1.0) / rate);
}

double ActuarialMath::calculate_loss_ratio(double incurred_losses, double earned_premium) {
    if (!(earned_premium > 0.0) || !std::isfinite(earned_premium)) throw std::invalid_argument("Earned premium must be finite and strictly positive");
    if (!(incurred_losses >= 0.0) || !std::isfinite(incurred_losses)) throw std::invalid_argument("Incurred losses must be finite and non-negative");
    return incurred_losses / earned_premium;
}

double ActuarialMath::lookup_mortality_rate(int age) {
    // A simplified mock of the CSO 2001 mortality table
    if (age < 0) return 0.0;
    if (age <= 18) return 0.0015;
    if (age <= 30) return 0.0020;
    if (age <= 40) return 0.0025;
    if (age <= 50) return 0.0040;
    if (age <= 60) return 0.0090;
    if (age <= 70) return 0.0250;
    if (age <= 80) return 0.0600;
    if (age <= 90) return 0.1500;
    return 0.3000;
}
