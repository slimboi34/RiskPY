#include "FourierMath.hpp"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

// Portable π (avoids non-standard M_PI); not constexpr because acos is not.
const double PI = std::acos(-1.0);

std::size_t next_power_of_two(std::size_t n) {
    if (n == 0) return 1;
    std::size_t p = 1;
    while (p < n) {
        if (p > (static_cast<std::size_t>(-1) >> 1)) {
            throw std::invalid_argument("Input size too large for FFT zero-padding");
        }
        p <<= 1;
    }
    return p;
}

bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// In-place iterative radix-2 Cooley–Tukey. inverse=false → forward, true → inverse (no 1/N).
void fft_inplace(std::vector<double>& re, std::vector<double>& im, bool inverse) {
    const std::size_t n = re.size();
    if (n != im.size()) {
        throw std::invalid_argument("Real and imag vectors must have equal length");
    }
    if (n == 0) return;
    if ((n & (n - 1)) != 0) {
        throw std::invalid_argument("FFT length must be a power of 2");
    }

    // Bit-reversal permutation
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }

    const double sign = inverse ? 1.0 : -1.0;
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double ang = sign * 2.0 * PI / static_cast<double>(len);
        const double wlen_re = std::cos(ang);
        const double wlen_im = std::sin(ang);
        for (std::size_t i = 0; i < n; i += len) {
            double w_re = 1.0;
            double w_im = 0.0;
            const std::size_t half = len >> 1;
            for (std::size_t j = 0; j < half; ++j) {
                const std::size_t u = i + j;
                const std::size_t v = u + half;
                const double t_re = w_re * re[v] - w_im * im[v];
                const double t_im = w_re * im[v] + w_im * re[v];
                re[v] = re[u] - t_re;
                im[v] = im[u] - t_im;
                re[u] += t_re;
                im[u] += t_im;
                const double next_w_re = w_re * wlen_re - w_im * wlen_im;
                w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = next_w_re;
            }
        }
    }
}

void require_finite_pair(const std::vector<double>& real, const std::vector<double>& imag) {
    if (real.size() != imag.size()) {
        throw std::invalid_argument("Real and imag vectors must have equal length");
    }
    for (std::size_t i = 0; i < real.size(); ++i) {
        if (!std::isfinite(real[i]) || !std::isfinite(imag[i])) {
            throw std::invalid_argument("FFT inputs must be finite");
        }
    }
}

std::pair<std::vector<double>, std::vector<double>>
pad_and_transform(const std::vector<double>& real,
                  const std::vector<double>& imag,
                  bool inverse) {
    require_finite_pair(real, imag);
    // Empty input → empty output (no phantom length-1 zero-pad surprise)
    if (real.empty()) {
        return {{}, {}};
    }
    const std::size_t n = next_power_of_two(real.size());
    std::vector<double> re(n, 0.0);
    std::vector<double> im(n, 0.0);
    for (std::size_t i = 0; i < real.size(); ++i) {
        re[i] = real[i];
        im[i] = imag[i];
    }
    fft_inplace(re, im, inverse);
    if (inverse) {
        const double inv_n = 1.0 / static_cast<double>(n);
        for (std::size_t i = 0; i < n; ++i) {
            re[i] *= inv_n;
            im[i] *= inv_n;
        }
    }
    return {std::move(re), std::move(im)};
}

}  // namespace

std::pair<std::vector<double>, std::vector<double>>
FourierTransform::fft(const std::vector<double>& real, const std::vector<double>& imag) {
    return pad_and_transform(real, imag, false);
}

std::pair<std::vector<double>, std::vector<double>>
FourierTransform::ifft(const std::vector<double>& real, const std::vector<double>& imag) {
    return pad_and_transform(real, imag, true);
}

std::vector<double>
FourierTransform::convolve(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.empty() || b.empty()) {
        return {};
    }
    for (double x : a) {
        if (!std::isfinite(x)) throw std::invalid_argument("convolve inputs must be finite");
    }
    for (double x : b) {
        if (!std::isfinite(x)) throw std::invalid_argument("convolve inputs must be finite");
    }
    const std::size_t out_len = a.size() + b.size() - 1;
    const std::size_t n = next_power_of_two(out_len);

    std::vector<double> a_re(n, 0.0), a_im(n, 0.0);
    std::vector<double> b_re(n, 0.0), b_im(n, 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) a_re[i] = a[i];
    for (std::size_t i = 0; i < b.size(); ++i) b_re[i] = b[i];

    fft_inplace(a_re, a_im, false);
    fft_inplace(b_re, b_im, false);

    for (std::size_t i = 0; i < n; ++i) {
        const double re = a_re[i] * b_re[i] - a_im[i] * b_im[i];
        const double im = a_re[i] * b_im[i] + a_im[i] * b_re[i];
        a_re[i] = re;
        a_im[i] = im;
    }

    fft_inplace(a_re, a_im, true);
    const double inv_n = 1.0 / static_cast<double>(n);
    std::vector<double> result(out_len);
    for (std::size_t i = 0; i < out_len; ++i) {
        result[i] = a_re[i] * inv_n;
    }
    return result;
}

std::vector<double>
FourierTransform::compound_poisson_pmf(const std::vector<double>& severity_pmf,
                                       double expected_frequency,
                                       int grid_size) {
    if (severity_pmf.empty()) {
        throw std::invalid_argument("severity_pmf cannot be empty");
    }
    if (!(expected_frequency >= 0.0) || !std::isfinite(expected_frequency)) {
        throw std::invalid_argument("expected_frequency must be a finite non-negative number");
    }

    double sev_sum = 0.0;
    for (double p : severity_pmf) {
        if (!(p >= 0.0) || !std::isfinite(p)) {
            throw std::invalid_argument("severity_pmf entries must be finite and non-negative");
        }
        sev_sum += p;
    }
    if (!(sev_sum > 0.0)) {
        throw std::invalid_argument("severity_pmf must have positive total mass");
    }
    // Auto-normalize so unscaled histograms still yield a valid probability measure.
    // For a true PMF, |φ_X(t)| ≤ 1 ⇒ Re(λ(φ−1)) ≤ 0 ⇒ exp is stable (no overflow).
    const double inv_sev = 1.0 / sev_sum;

    std::size_t n;
    if (grid_size == 0) {
        n = next_power_of_two(severity_pmf.size() * 2);
        if (n < 16) n = 16;
    } else {
        if (!is_power_of_two(grid_size)) {
            throw std::invalid_argument("grid_size must be a power of 2 (or 0 for auto)");
        }
        n = static_cast<std::size_t>(grid_size);
        if (n < severity_pmf.size()) {
            throw std::invalid_argument("grid_size must be >= severity_pmf length");
        }
    }

    std::vector<double> re(n, 0.0), im(n, 0.0);
    for (std::size_t i = 0; i < severity_pmf.size(); ++i) {
        re[i] = severity_pmf[i] * inv_sev;
    }

    fft_inplace(re, im, false);

    // Ŝ(t) = exp(λ (X̂(t) − 1))
    for (std::size_t i = 0; i < n; ++i) {
        const double xr = re[i] - 1.0;
        const double xi = im[i];
        // Clamp real exponent for pathological float noise (should be ≤ 0 for valid PMFs)
        double mag = expected_frequency * xr;
        if (mag > 0.0) mag = 0.0;
        if (mag < -700.0) mag = -700.0;  // underflows to ~0, avoids denorm churn
        const double phase = expected_frequency * xi;
        const double scale = std::exp(mag);
        re[i] = scale * std::cos(phase);
        im[i] = scale * std::sin(phase);
    }

    fft_inplace(re, im, true);
    const double inv_n = 1.0 / static_cast<double>(n);
    std::vector<double> pmf(n);
    double mass = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        // Clip tiny negative round-off from floating point
        double p = re[i] * inv_n;
        if (p < 0.0) p = 0.0;
        pmf[i] = p;
        mass += p;
    }
    // Renormalize residual FFT mass error so callers can treat the result as a PMF
    if (mass > 0.0 && std::isfinite(mass)) {
        const double inv_mass = 1.0 / mass;
        for (double& p : pmf) {
            p *= inv_mass;
        }
    }
    return pmf;
}
