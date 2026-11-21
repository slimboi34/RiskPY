//! Samplers for the distributions `MonteCarloSimulator` draws from, on top of the
//! PCG64-DXSM stream in [`crate::rng`].
//!
//! All are exact methods (no normal approximations), use only IEEE arithmetic and the
//! pure-Rust `libm` functions, and so give the same draws on every platform:
//!
//! | distribution | method |
//! |---|---|
//! | uniform      | top 53 bits of a 64-bit draw |
//! | normal       | Marsaglia's polar method (the second variate is kept for the next call) |
//! | lognormal    | `exp(mu + sigma·Z)` |
//! | gamma        | Marsaglia & Tsang (2000); shape < 1 by `G(a+1)·U^(1/a)` |
//! | Poisson      | inversion below λ = 10; Hörmann's PTRS (1993) from λ = 10, accepting with Loader's saddle-point log-pmf (2000) so large λ keeps its digits |
//! | binomial     | inversion when n·min(p, 1−p) ≤ 30; Kachitvichyanukul & Schmeiser's BTPE (1988) otherwise, as NumPy implements it |
//! | Bernoulli    | `U < p` |
//!
//! `tests/test_rust_core.py` checks every one against its closed form (moments and a
//! Kolmogorov–Smirnov or chi-square test) at several parameter points.

use crate::m;
use crate::rng::Pcg64Dxsm;

const TWO_POW_M53: f64 = 1.0 / 9_007_199_254_740_992.0; // 2^-53
const TWO_POW_M52: f64 = 1.0 / 4_503_599_627_370_496.0; // 2^-52

/// A random stream with the samplers attached. Each `simulate_*` call makes one from
/// its seed, as the C++ core made a fresh `std::mt19937` per call.
#[derive(Debug, Clone)]
pub struct Stream {
    rng: Pcg64Dxsm,
    spare_normal: Option<f64>,
}

impl Stream {
    pub fn new(seed: u32) -> Self {
        Stream { rng: Pcg64Dxsm::from_seed(seed), spare_normal: None }
    }

    pub fn from_generator(rng: Pcg64Dxsm) -> Self {
        Stream { rng, spare_normal: None }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }

    /// Uniform on [0, 1), multiples of 2^-53.
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.rng.next_u64() >> 11) as f64 * TWO_POW_M53
    }

    /// Uniform on (0, 1), never 0 or 1: odd multiples of 2^-53.
    #[inline]
    pub fn uniform_open(&mut self) -> f64 {
        ((self.rng.next_u64() >> 12) as f64 + 0.5) * TWO_POW_M52
    }

    /// Uniform on (-1, 1), symmetric about 0.
    #[inline]
    fn uniform_signed(&mut self) -> f64 {
        ((self.rng.next_u64() >> 11) as f64 + 0.5) * TWO_POW_M52 - 1.0
    }

    /// Standard normal, Marsaglia's polar method.
    pub fn standard_normal(&mut self) -> f64 {
        if let Some(z) = self.spare_normal.take() {
            return z;
        }
        loop {
            let u = self.uniform_signed();
            let v = self.uniform_signed();
            let s = u * u + v * v;
            if s < 1.0 && s > 0.0 {
                let f = m::sqrt(-2.0 * m::ln(s) / s);
                self.spare_normal = Some(v * f);
                return u * f;
            }
        }
    }

    /// exp(mu + sigma·Z).
    #[inline]
    pub fn lognormal(&mut self, mu: f64, sigma: f64) -> f64 {
        m::exp(mu + sigma * self.standard_normal())
    }

    /// Bernoulli(p): true with probability p (p in [0, 1]).
    #[inline]
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.uniform() < p
    }

    /// Gamma(shape, 1), shape > 0 and finite.
    pub fn standard_gamma(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            // G(a) = G(a + 1) · U^(1/a) (Marsaglia & Tsang, section 6), in log form.
            let g = self.gamma_marsaglia_tsang(shape + 1.0);
            let u = self.uniform_open();
            return g * m::exp(m::ln(u) / shape);
        }
        self.gamma_marsaglia_tsang(shape)
    }

    /// Gamma(shape, scale) = scale · Gamma(shape, 1), as `std::gamma_distribution`.
    #[inline]
    pub fn gamma(&mut self, shape: f64, scale: f64) -> f64 {
        self.standard_gamma(shape) * scale
    }

    /// Marsaglia & Tsang (2000), shape >= 1.
    ///
    /// The log test is `ln U < x²/2 + d(1 − v + ln v)` with `v = (1 + c x)³`. For large
    /// shapes `1 − v + ln v` cancels badly when computed from v, so it is computed from
    /// `t = c x` as `3(ln(1+t) − t) − 3t² − t³` with a series for `ln(1+t) − t`.
    fn gamma_marsaglia_tsang(&mut self, shape: f64) -> f64 {
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / m::sqrt(9.0 * d);
        loop {
            let (x, t) = loop {
                let x = self.standard_normal();
                let t = c * x;
                if t > -1.0 {
                    break (x, t);
                }
            };
            let v1 = 1.0 + t;
            let v = v1 * v1 * v1;
            let u = self.uniform_open();
            let x2 = x * x;
            if u < 1.0 - 0.0331 * x2 * x2 {
                return d * v;
            }
            let q = 3.0 * ln1p_minus_x(t) - 3.0 * t * t - t * t * t;
            if m::ln(u) < 0.5 * x2 + d * q {
                return d * v;
            }
        }
    }

    /// Poisson(lam), lam >= 0 and finite.
    pub fn poisson(&mut self, lam: f64) -> u64 {
        if lam >= 10.0 {
            self.poisson_ptrs(lam)
        } else if lam > 0.0 {
            self.poisson_inversion(lam)
        } else {
            0
        }
    }

    /// Sequential-search inversion for 0 < lam < 10.
    fn poisson_inversion(&mut self, lam: f64) -> u64 {
        let p0 = m::exp(-lam);
        loop {
            let mut u = self.uniform();
            let mut k = 0u64;
            let mut p = p0;
            // P(N > 200) is below 1e-200 for lam < 10; the cap only guards against the
            // rounding residue of the subtractions leaving u above the last term.
            while k < 200 {
                if u < p {
                    return k;
                }
                u -= p;
                k += 1;
                p *= lam / k as f64;
            }
        }
    }

    /// Hörmann's PTRS (transformed rejection with squeeze), lam >= 10.
    fn poisson_ptrs(&mut self, lam: f64) -> u64 {
        let slam = m::sqrt(lam);
        let b = 0.931 + 2.53 * slam;
        let a = -0.059 + 0.02483 * b;
        let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
        let vr = 0.9277 - 3.6224 / (b - 2.0);
        let ln_inv_alpha = m::ln(inv_alpha);
        loop {
            let u = self.uniform() - 0.5;
            let v = self.uniform();
            let us = 0.5 - u.abs();
            let kf = m::floor((2.0 * a / us + b) * u + lam + 0.43);
            if us >= 0.07 && v <= vr {
                return kf as u64;
            }
            if kf < 0.0 || (us < 0.013 && v > us) {
                continue;
            }
            // ln(v) of 0 is -inf, which accepts; us == 0 only when u == ±0.5, which the
            // squeeze above has already taken.
            if m::ln(v) + ln_inv_alpha - m::ln(a / (us * us) + b) <= poisson_ln_pmf(kf, lam) {
                return kf as u64;
            }
        }
    }

    /// Binomial(n, p), p in [0, 1].
    pub fn binomial(&mut self, n: u64, p: f64) -> u64 {
        if n == 0 || p <= 0.0 {
            return 0;
        }
        if p <= 0.5 {
            if p * n as f64 <= 30.0 {
                self.binomial_inversion(n, p)
            } else {
                self.binomial_btpe(n, p)
            }
        } else {
            let q = 1.0 - p;
            if q * n as f64 <= 30.0 {
                n - self.binomial_inversion(n, q)
            } else {
                n - self.binomial_btpe(n, q)
            }
        }
    }

    /// Inversion for n·p <= 30, p <= 0.5 (NumPy's `random_binomial_inversion`).
    fn binomial_inversion(&mut self, n: u64, p: f64) -> u64 {
        let nf = n as f64;
        let q = 1.0 - p;
        let qn = m::exp(nf * m::ln1p(-p));
        let np = nf * p;
        let bound = nf.min(np + 10.0 * m::sqrt(np * q + 1.0)) as u64;
        let mut x = 0u64;
        let mut px = qn;
        let mut u = self.uniform();
        while u > px {
            x += 1;
            if x > bound {
                x = 0;
                px = qn;
                u = self.uniform();
            } else {
                u -= px;
                px = ((nf - x as f64 + 1.0) * p * px) / (x as f64 * q);
            }
        }
        x
    }

    /// BTPE for n·p > 30, p <= 0.5: Kachitvichyanukul & Schmeiser (1988), with the
    /// correction to the Stirling terms NumPy applies (two of the four are subtracted).
    fn binomial_btpe(&mut self, n: u64, p: f64) -> u64 {
        let nf = n as f64;
        let r = p.min(1.0 - p);
        let q = 1.0 - r;
        let fm = nf * r + r;
        let mi = m::floor(fm) as i64;
        let mf = mi as f64;
        let p1 = m::floor(2.195 * m::sqrt(nf * r * q) - 4.6 * q) + 0.5;
        let xm = mf + 0.5;
        let xl = xm - p1;
        let xr = xm + p1;
        let c = 0.134 + 20.5 / (15.3 + mf);
        let a = (fm - xl) / (fm - xl * r);
        let laml = a * (1.0 + a / 2.0);
        let a = (xr - fm) / (xr * q);
        let lamr = a * (1.0 + a / 2.0);
        let p2 = p1 * (1.0 + 2.0 * c);
        let p3 = p2 + c / laml;
        let p4 = p3 + c / lamr;
        let nrq = nf * r * q;
        let ni = n as i64;

        loop {
            // Step 10
            let u = self.uniform() * p4;
            let mut v = self.uniform();
            let y: i64;
            if u <= p1 {
                y = m::floor(xm - p1 * v + u) as i64;
                return y as u64;
            }
            if u <= p2 {
                // Step 20: parallelogram
                let x = xl + (u - p1) / c;
                v = v * c + 1.0 - (mf - x + 0.5).abs() / p1;
                if v > 1.0 {
                    continue;
                }
                y = m::floor(x) as i64;
            } else if u <= p3 {
                // Step 30: left exponential tail
                let yf = m::floor(xl + m::ln(v) / laml);
                if yf < 0.0 || v == 0.0 {
                    continue;
                }
                y = yf as i64;
                v *= (u - p2) * laml;
            } else {
                // Step 40: right exponential tail
                let yf = m::floor(xr - m::ln(v) / lamr);
                if yf > nf || v == 0.0 {
                    continue;
                }
                y = yf as i64;
                v *= (u - p3) * lamr;
            }

            // Step 50: acceptance
            let k = (y - mi).abs();
            let kf = k as f64;
            if !(k > 20 && kf < nrq / 2.0 - 1.0) {
                // Evaluate f(y)/f(m) recursively.
                let s = r / q;
                let aa = s * (nf + 1.0);
                let mut f = 1.0;
                if mi < y {
                    let mut i = mi + 1;
                    while i <= y {
                        f *= aa / i as f64 - s;
                        i += 1;
                    }
                } else if mi > y {
                    let mut i = y + 1;
                    while i <= mi {
                        f /= aa / i as f64 - s;
                        i += 1;
                    }
                }
                if v > f {
                    continue;
                }
                return y as u64;
            }

            // Step 52: squeeze on ln f(y)/f(m), then the Stirling bound.
            let rho = (kf / nrq) * ((kf * (kf / 3.0 + 0.625) + 0.166_666_666_666_666_66) / nrq + 0.5);
            let t = -kf * kf / (2.0 * nrq);
            let big_a = m::ln(v);
            if big_a < t - rho {
                return y as u64;
            }
            if big_a > t + rho {
                continue;
            }
            let yf = y as f64;
            let x1 = yf + 1.0;
            let f1 = mf + 1.0;
            let z = nf + 1.0 - mf;
            let w = nf - yf + 1.0;
            let x2 = x1 * x1;
            let f2 = f1 * f1;
            let z2 = z * z;
            let w2 = w * w;
            let bound = xm * m::ln(f1 / x1)
                + ((ni - mi) as f64 + 0.5) * m::ln(z / w)
                + (y - mi) as f64 * m::ln(w * r / (x1 * q))
                + stirling_tail(f2) / f1 / 166_320.0
                + stirling_tail(z2) / z / 166_320.0
                - stirling_tail(x2) / x1 / 166_320.0
                - stirling_tail(w2) / w / 166_320.0;
            if big_a > bound {
                continue;
            }
            return y as u64;
        }
    }
}

#[inline]
fn stirling_tail(x2: f64) -> f64 {
    13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / x2) / x2) / x2) / x2
}

/// ln(1 + t) − t without cancellation for small t.
#[inline]
fn ln1p_minus_x(t: f64) -> f64 {
    if t.abs() < 1e-2 {
        // −t²/2 + t³/3 − … − t¹²/12 by Horner; the first omitted term is below
        // 1e-22 of the sum for |t| < 0.01.
        let mut acc = 0.0;
        for k in (2..=12).rev() {
            let c = if k % 2 == 0 { -1.0 } else { 1.0 } / f64::from(k);
            acc = acc * t + c;
        }
        acc * t * t
    } else {
        m::ln1p(t) - t
    }
}

/// Loader's Stirling-series error, ln(n!) − ln(√(2πn)(n/e)^n), for whole n >= 0.
fn stirlerr(n: f64) -> f64 {
    // n = 0..15, computed at 40 digits with mpmath and rounded to double.
    const TABLE: [f64; 16] = [
        0.0,
        0.08106146679532726,
        0.0413406959554093,
        0.02767792568499834,
        0.020790672103765093,
        0.016644691189821193,
        0.013876128823070748,
        0.01189670994589177,
        0.010411265261972096,
        0.009255462182712733,
        0.00833056343336287,
        0.007573675487951841,
        0.00694284010720953,
        0.006408994188004207,
        0.0059513701127588475,
        0.005554733551962801,
    ];
    const S0: f64 = 1.0 / 12.0;
    const S1: f64 = 1.0 / 360.0;
    const S2: f64 = 1.0 / 1260.0;
    const S3: f64 = 1.0 / 1680.0;
    const S4: f64 = 1.0 / 1188.0;
    if n <= 15.0 {
        return TABLE[n as usize];
    }
    let nn = n * n;
    if n > 500.0 {
        return (S0 - S1 / nn) / n;
    }
    if n > 80.0 {
        return (S0 - (S1 - S2 / nn) / nn) / n;
    }
    if n > 35.0 {
        return (S0 - (S1 - (S2 - S3 / nn) / nn) / nn) / n;
    }
    (S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / n
}

/// Loader's deviance term x·ln(x/np) + np − x, accurate when x is close to np.
fn bd0(x: f64, np: f64) -> f64 {
    if (x - np).abs() < 0.1 * (x + np) {
        let mut v = (x - np) / (x + np);
        let mut s = (x - np) * v;
        if s.abs() < f64::MIN_POSITIVE {
            return s;
        }
        let mut ej = 2.0 * x * v;
        v *= v;
        for j in 1..1000 {
            ej *= v;
            let s1 = s + ej / f64::from(2 * j + 1);
            if s1 == s {
                return s1;
            }
            s = s1;
        }
    }
    x * m::ln(x / np) + np - x
}

const LN_2PI: f64 = 1.837_877_066_409_345_6; // ln(2π)

/// ln P(N = k) for N ~ Poisson(lam): Loader's saddle-point form,
/// −stirlerr(k) − bd0(k, λ) − ½ ln(2πk), accurate to a few ulps even at λ = 1e9
/// where −λ + k ln λ − ln k! cancels to about 1e-6.
pub fn poisson_ln_pmf(k: f64, lam: f64) -> f64 {
    if k == 0.0 {
        return -lam;
    }
    -stirlerr(k) - bd0(k, lam) - 0.5 * (LN_2PI + m::ln(k))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniforms_stay_in_range() {
        let mut s = Stream::new(7);
        for _ in 0..100_000 {
            let u = s.uniform();
            assert!((0.0..1.0).contains(&u));
            let o = s.uniform_open();
            assert!(o > 0.0 && o < 1.0);
            let w = s.uniform_signed();
            assert!(w > -1.0 && w < 1.0);
        }
    }

    #[test]
    fn poisson_ln_pmf_matches_reference_values() {
        // -lam + k ln lam - ln k!, evaluated with mpmath at 50 digits.
        let cases = [
            (0.0, 3.5, -3.5),
            (3.0, 3.5, -1.533470563741951),
            (10.0, 10.0, -2.0785616431350586),
            (25.0, 10.0, -10.438977898129378),
            (1000.0, 1000.0, -4.372899506026297),
            (1_000_000_000.0, 1_000_000_000.0, -11.280571451761212),
            (1_000_030_000.0, 1_000_000_000.0, -11.730581951603712),
            (16.0, 17.5, -2.376646011209179),
            (40.0, 30.0, -4.2727444482711805),
            (81.0, 90.0, -3.5829901436146727),
            (501.0, 480.0, -4.480226506370706),
            (7.0, 12.25, -3.2364798021302623),
        ];
        for &(k, lam, want) in &cases {
            let got = poisson_ln_pmf(k, lam);
            assert!((got - want).abs() <= 2e-14 * want.abs(), "k={k} lam={lam}: {got} vs {want}");
        }
    }

    #[test]
    fn ln1p_minus_x_is_continuous_at_the_switch() {
        for &t in &[0.009_999_999, 0.01, -0.009_999_999, -0.01] {
            let series_side = ln1p_minus_x(t * 0.999_999_9);
            let direct_side = m::ln1p(t * 0.999_999_9) - t * 0.999_999_9;
            assert!((series_side - direct_side).abs() <= 1e-12 * direct_side.abs());
        }
    }

    #[test]
    fn samplers_are_deterministic_per_seed() {
        let draw = |seed| {
            let mut s = Stream::new(seed);
            (s.poisson(3.0), s.poisson(1e6), s.binomial(1000, 0.3), s.standard_gamma(0.5), s.standard_normal())
        };
        assert_eq!(draw(5), draw(5));
    }

    #[test]
    fn binomial_edges() {
        let mut s = Stream::new(3);
        for _ in 0..1000 {
            assert_eq!(s.binomial(10, 0.0), 0);
            assert_eq!(s.binomial(10, 1.0), 10);
            assert_eq!(s.binomial(0, 0.5), 0);
            assert!(s.binomial(2_000_000_000, 0.5) <= 2_000_000_000);
        }
    }
}
