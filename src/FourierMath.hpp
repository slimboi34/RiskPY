#pragma once
#include <utility>
#include <vector>

// Analytic Fourier / FFT utilities for actuarial aggregate-loss modeling.
// Iterative radix-2 Cooley–Tukey transform with auto zero-padding to the next power of 2.
class FourierTransform {
public:
    // Forward DFT: returns (real, imag) component pairs, length next power of 2 ≥ input.
    static std::pair<std::vector<double>, std::vector<double>>
    fft(const std::vector<double>& real, const std::vector<double>& imag);

    // Inverse DFT (normalized by 1/N).
    static std::pair<std::vector<double>, std::vector<double>>
    ifft(const std::vector<double>& real, const std::vector<double>& imag);

    // Fast linear convolution; result length = a.size() + b.size() - 1 (empty if either empty).
    static std::vector<double>
    convolve(const std::vector<double>& a, const std::vector<double>& b);

    // Compound-Poisson aggregate-loss PMF via FFT identity:
    //   Ŝ(t) = exp(λ (X̂(t) − 1))
    // severity_pmf is a discrete severity mass on {0, 1, …, n-1}; auto-normalized
    // to a PMF if total mass > 0. Output is renormalized to sum ≈ 1.
    // grid_size must be a power of 2 (or 0 to auto-select next power of 2 ≥ 2 * severity size).
    static std::vector<double>
    compound_poisson_pmf(const std::vector<double>& severity_pmf,
                         double expected_frequency,
                         int grid_size = 0);
};
