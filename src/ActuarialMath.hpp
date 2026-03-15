#pragma once

class ActuarialMath {
public:
    static double present_value(double rate, int periods, double payment);
    static double future_value(double rate, int periods, double payment);
    static double calculate_loss_ratio(double incurred_losses, double earned_premium);
    static double lookup_mortality_rate(int age);
};
