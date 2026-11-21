//! `FourierTransform`: radix-2 FFT, convolution and the compound-Poisson aggregate PMF.
//!
//! The transform is the C++ core's iterative Cooley–Tukey, operation for operation
//! (same bit reversal, same twiddle recurrence), so results agree with 0.8.0 to the
//! last bit wherever `libm`'s sin and cos agree with the platform's.

use crate::error::{Error, Result, invalid, vec_filled};
use crate::m;

/// Largest grid `compound_poisson_pmf` allocates (16M points, ~400 MB of work arrays).
pub const MAX_GRID: usize = 1 << 24;

fn next_power_of_two(n: usize) -> Result<usize> {
    if n == 0 {
        return Ok(1);
    }
    let mut p: usize = 1;
    while p < n {
        if p > (usize::MAX >> 1) {
            return invalid("Input size too large for FFT zero-padding");
        }
        p <<= 1;
    }
    Ok(p)
}

/// In-place iterative radix-2 Cooley–Tukey; `inverse` flips the sign (no 1/N).
fn fft_inplace(re: &mut [f64], im: &mut [f64], inverse: bool) -> Result<()> {
    let n = re.len();
    if n != im.len() {
        return invalid("Real and imag vectors must have equal length");
    }
    if n == 0 {
        return Ok(());
    }
    if n & (n - 1) != 0 {
        return invalid("FFT length must be a power of 2");
    }

    // Bit-reversal permutation
    let mut j: usize = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let ang = sign * 2.0 * std::f64::consts::PI / len as f64;
        let wlen_re = m::cos(ang);
        let wlen_im = m::sin(ang);
        let half = len >> 1;
        let mut i = 0;
        while i < n {
            let mut w_re = 1.0;
            let mut w_im = 0.0;
            for jj in 0..half {
                let u = i + jj;
                let v = u + half;
                let t_re = w_re * re[v] - w_im * im[v];
                let t_im = w_re * im[v] + w_im * re[v];
                re[v] = re[u] - t_re;
                im[v] = im[u] - t_im;
                re[u] += t_re;
                im[u] += t_im;
                let next_w_re = w_re * wlen_re - w_im * wlen_im;
                w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = next_w_re;
            }
            i += len;
        }
        len <<= 1;
    }
    Ok(())
}

fn require_finite_pair(real: &[f64], imag: &[f64]) -> Result<()> {
    if real.len() != imag.len() {
        return invalid("Real and imag vectors must have equal length");
    }
    for (r, i) in real.iter().zip(imag) {
        if !r.is_finite() || !i.is_finite() {
            return invalid("FFT inputs must be finite");
        }
    }
    Ok(())
}

fn pad_and_transform(real: &[f64], imag: &[f64], inverse: bool) -> Result<(Vec<f64>, Vec<f64>)> {
    require_finite_pair(real, imag)?;
    // Empty input → empty output (no phantom length-1 zero-pad surprise)
    if real.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let n = next_power_of_two(real.len())?;
    let mut re = vec_filled(n, 0.0)?;
    let mut im = vec_filled(n, 0.0)?;
    re[..real.len()].copy_from_slice(real);
    im[..imag.len()].copy_from_slice(imag);
    fft_inplace(&mut re, &mut im, inverse)?;
    if inverse {
        let inv_n = 1.0 / n as f64;
        for k in 0..n {
            re[k] *= inv_n;
            im[k] *= inv_n;
        }
    }
    Ok((re, im))
}

/// Forward DFT, zero-padded to the next power of two.
pub fn fft(real: &[f64], imag: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
    pad_and_transform(real, imag, false)
}

/// Inverse DFT (normalised by 1/N), zero-padded to the next power of two.
pub fn ifft(real: &[f64], imag: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
    pad_and_transform(real, imag, true)
}

/// Linear convolution via FFT; length a.len() + b.len() − 1, empty if either is empty.
pub fn convolve(a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
    if a.is_empty() || b.is_empty() {
        return Ok(Vec::new());
    }
    if a.iter().chain(b).any(|x| !x.is_finite()) {
        return invalid("convolve inputs must be finite");
    }
    let out_len = a.len() + b.len() - 1;
    let n = next_power_of_two(out_len)?;
    let mut a_re = vec_filled(n, 0.0)?;
    let mut a_im = vec_filled(n, 0.0)?;
    let mut b_re = vec_filled(n, 0.0)?;
    let mut b_im = vec_filled(n, 0.0)?;
    a_re[..a.len()].copy_from_slice(a);
    b_re[..b.len()].copy_from_slice(b);
    fft_inplace(&mut a_re, &mut a_im, false)?;
    fft_inplace(&mut b_re, &mut b_im, false)?;
    for k in 0..n {
        let re = a_re[k] * b_re[k] - a_im[k] * b_im[k];
        let im = a_re[k] * b_im[k] + a_im[k] * b_re[k];
        a_re[k] = re;
        a_im[k] = im;
    }
    fft_inplace(&mut a_re, &mut a_im, true)?;
    let inv_n = 1.0 / n as f64;
    Ok(a_re[..out_len].iter().map(|x| x * inv_n).collect())
}

/// `std::max(a, b)` on doubles: `a < b ? b : a`.
#[inline]
fn cpp_max(a: f64, b: f64) -> f64 {
    if a < b { b } else { a }
}

/// `std::min(a, b)` on doubles: `b < a ? b : a`.
#[inline]
fn cpp_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// Compound-Poisson aggregate-loss PMF on the severity's grid, via
/// Ŝ(t) = exp(λ (X̂(t) − 1)).
///
/// `severity_pmf` is a mass on {0, 1, …, n−1}, normalised if it does not sum to 1.
/// `grid_size` is a power of two up to 2^24, or 0 to size the grid from λ and the
/// severity (support and moments); a grid on which a material share of the aggregate
/// wraps around is refused.
pub fn compound_poisson_pmf(severity_pmf: &[f64], expected_frequency: f64, grid_size: i32) -> Result<Vec<f64>> {
    if severity_pmf.is_empty() {
        return invalid("severity_pmf cannot be empty");
    }
    if !(expected_frequency >= 0.0) || !expected_frequency.is_finite() {
        return invalid("expected_frequency must be a finite non-negative number");
    }
    let mut sev_sum = 0.0;
    for &p in severity_pmf {
        if !(p >= 0.0) || !p.is_finite() {
            return invalid("severity_pmf entries must be finite and non-negative");
        }
        sev_sum += p;
    }
    if !(sev_sum > 0.0) {
        return invalid("severity_pmf must have positive total mass");
    }
    // Auto-normalize so unscaled histograms still yield a valid probability measure.
    // For a true PMF, |φ_X(t)| ≤ 1 ⇒ Re(λ(φ−1)) ≤ 0 ⇒ exp is stable (no overflow).
    let inv_sev = 1.0 / sev_sum;

    // Severity moments in grid units; they size the grid and check it afterwards.
    let mut sev_mean = 0.0;
    let mut sev_m2 = 0.0;
    let mut support: usize = 0; // index of the last non-zero severity point
    for (i, &raw) in severity_pmf.iter().enumerate() {
        let p = raw * inv_sev;
        if p > 0.0 {
            support = i;
        }
        let fi = i as f64;
        sev_mean += fi * p;
        sev_m2 += fi * fi * p;
    }
    let lam = expected_frequency;
    let target_mean = lam * sev_mean;

    let auto_grid = grid_size == 0;
    let mut n: usize;
    if auto_grid {
        // The aggregate lives on [0, support * N]. Size the grid from the claim count
        // (N <= lambda + 10 sqrt(lambda) + 10 essentially surely) and from the aggregate
        // moments, whichever is smaller; a grid that is still too small (circular
        // wrap-around) is detected below and doubled.
        let n_max = lam + 10.0 * m::sqrt(lam) + 10.0;
        let by_support = support as f64 * n_max + 1.0;
        let by_moments = target_mean + 12.0 * m::sqrt(lam * sev_m2) + 2.0 * support as f64 + 16.0;
        let mut want = cpp_min(by_support, by_moments);
        want = cpp_max(want, 2.0 * severity_pmf.len() as f64);
        want = cpp_max(want, 16.0);
        if !(want <= MAX_GRID as f64) {
            n = MAX_GRID;
        } else {
            n = next_power_of_two(want.ceil() as usize)?;
        }
        if n < severity_pmf.len() {
            return invalid("severity_pmf is longer than the largest supported FFT grid (2^24)");
        }
    } else {
        if !(grid_size > 0 && (grid_size & (grid_size - 1)) == 0) {
            return invalid("grid_size must be a power of 2 (or 0 for auto)");
        }
        n = grid_size as usize;
        if n > MAX_GRID {
            return invalid("grid_size must be at most 2^24 (16777216)");
        }
        if n < severity_pmf.len() {
            return invalid("grid_size must be >= severity_pmf length");
        }
    }

    loop {
        let mut re = vec_filled(n, 0.0)?;
        let mut im = vec_filled(n, 0.0)?;
        for (dst, &p) in re.iter_mut().zip(severity_pmf) {
            *dst = p * inv_sev;
        }

        fft_inplace(&mut re, &mut im, false)?;

        // Ŝ(t) = exp(λ (X̂(t) − 1))
        for k in 0..n {
            let xr = re[k] - 1.0;
            let xi = im[k];
            // Clamp real exponent for pathological float noise (should be ≤ 0 for valid PMFs)
            let mut mag = lam * xr;
            if mag > 0.0 {
                mag = 0.0;
            }
            if mag < -700.0 {
                mag = -700.0; // underflows to ~0, avoids denorm churn
            }
            let phase = lam * xi;
            let scale = m::exp(mag);
            re[k] = scale * m::cos(phase);
            im[k] = scale * m::sin(phase);
        }

        fft_inplace(&mut re, &mut im, true)?;
        let inv_n = 1.0 / n as f64;

        // Circular convolution folds any aggregate mass beyond the grid back onto the
        // start: total mass is kept, but the mean drops. Compare the raw mean with λ·E[X].
        let mut raw_mean = 0.0;
        for (k, &v) in re.iter().enumerate() {
            raw_mean += k as f64 * v * inv_n;
        }
        let mean_err = (raw_mean - target_mean).abs();
        let noise = 1e-9 * cpp_max(target_mean, 1.0) + 1e-12 * n as f64;
        let wrapped = mean_err > noise;
        if wrapped {
            if auto_grid && n < MAX_GRID {
                n <<= 1;
                continue;
            }
            // An explicit grid keeps the caller's choice unless the fold is material.
            if auto_grid || mean_err > 1e-3 * cpp_max(target_mean, 1.0) {
                return Err(Error::Value(format!(
                    "FFT grid too small: aggregate mass wrapped around the grid (mean {} vs lambda*E[X] = {}); \
                     pass a larger power-of-two grid_size (max 2^24) or 0 for auto",
                    m::cpp_to_string(raw_mean),
                    m::cpp_to_string(target_mean)
                )));
            }
        }

        let mut mass = 0.0;
        for v in re.iter_mut() {
            // Clip tiny negative round-off from floating point
            let mut p = *v * inv_n;
            if p < 0.0 {
                p = 0.0;
            }
            *v = p;
            mass += p;
        }
        // Renormalize residual FFT mass error so callers can treat the result as a PMF
        if mass > 0.0 && mass.is_finite() {
            let inv_mass = 1.0 / mass;
            for p in re.iter_mut() {
                *p *= inv_mass;
            }
        }
        return Ok(re);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_of_two_dice() {
        let die = vec![1.0 / 6.0; 6];
        let two = convolve(&die, &die).unwrap();
        assert_eq!(two.len(), 11);
        assert!((two[5] - 6.0 / 36.0).abs() < 1e-15);
    }

    #[test]
    fn fft_round_trip() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (re, im) = fft(&x, &[0.0; 5]).unwrap();
        assert_eq!(re.len(), 8);
        let (back, _) = ifft(&re, &im).unwrap();
        for (a, b) in back.iter().zip(x.iter().chain([0.0, 0.0, 0.0].iter())) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn compound_poisson_mean() {
        let sev = [0.0, 0.5, 0.5];
        let pmf = compound_poisson_pmf(&sev, 4.0, 0).unwrap();
        let mean: f64 = pmf.iter().enumerate().map(|(k, p)| k as f64 * p).sum();
        assert!((mean - 6.0).abs() < 1e-9);
    }

    #[test]
    fn to_string_matches_printf() {
        assert_eq!(m::cpp_to_string(1.0 / 3.0), "0.333333");
        assert_eq!(m::cpp_to_string(2.5e-7), "0.000000");
        assert_eq!(m::cpp_to_string(123456.0), "123456.000000");
    }
}
