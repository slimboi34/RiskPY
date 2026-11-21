"""Check every sampler in the Rust core against its closed form.

The draws come straight from the core (``rust/core/examples/sample.rs``), not through a
``MonteCarloSimulator`` method, so each distribution is tested on its own:

* continuous ones by a one-sample Kolmogorov–Smirnov test against SciPy's CDF;
* discrete ones by a chi-square test against SciPy's exact PMF (equiprobable bins
  for large means, single values with pooled tails for small ones);
* all of them by the mean and variance, as z-scores against the exact moments.

    cargo build --release --manifest-path rust/Cargo.toml -p riskpy-core --example sample
    python benchmarks/rust_core/samplers.py rust/target/release/examples/sample

Needs NumPy and SciPy. A p-value below 1e-4 or a moment more than 5 standard errors
off is a failure; with ~70 tests, one p below 0.01 is expected by chance.
"""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
from scipy import stats

N = 400_000
SEED = 20261121


_calls = [0]


def draw(binary, kind, *params, n=N, seed=None):
    # A different seed for every draw, so the tests are independent of each other.
    if seed is None:
        _calls[0] += 1
        seed = SEED + _calls[0]
    out = subprocess.run([binary, kind, str(seed), str(n), *map(str, params)], check=True,
                         capture_output=True, text=True).stdout
    return np.array(out.split(), dtype=float)


def moment_z(x, mean, var):
    """z-scores of the sample mean and the sample variance against the exact values."""
    n = len(x)
    z_mean = (x.mean() - mean) / math.sqrt(var / n) if var > 0 else 0.0
    # Var of the sample variance ≈ (mu4 - var^2) / n; estimate mu4 from the sample.
    mu4 = np.mean((x - x.mean()) ** 4)
    se_var = math.sqrt(max(mu4 - var * var, 1e-300) / n)
    z_var = (x.var() - var) / se_var
    return z_mean, z_var


def chi_square(x, dist):
    """Chi-square goodness of fit of integer draws to a SciPy discrete distribution."""
    lo, hi = dist.ppf(1e-9), dist.ppf(1 - 1e-9)
    n = len(x)
    if hi - lo <= 200:
        values = np.arange(lo, hi + 1)
        expected = dist.pmf(values) * n
        observed = np.array([(x == v).sum() for v in values], dtype=float)
        # mass outside [lo, hi] goes to the end bins
        expected[0] += dist.cdf(lo - 1) * n
        expected[-1] += dist.sf(hi) * n
        observed[0] += (x < lo).sum()
        observed[-1] += (x > hi).sum()
        # pool bins with small expectations into their neighbours
        e, o, acc_e, acc_o = [], [], 0.0, 0.0
        for ee, oo in zip(expected, observed):
            acc_e += ee
            acc_o += oo
            if acc_e >= 20:
                e.append(acc_e)
                o.append(acc_o)
                acc_e = acc_o = 0.0
        if acc_e > 0:
            e[-1] += acc_e
            o[-1] += acc_o
    else:
        # equiprobable bins through the exact CDF
        qs = np.linspace(0, 1, 101)[1:-1]
        edges = np.unique(dist.ppf(qs))
        cdf = np.concatenate([[0.0], dist.cdf(edges), [1.0]])
        e = np.diff(cdf) * n
        idx = np.searchsorted(edges, x, side="left")  # bin k holds (edge[k-1], edge[k]]
        o = np.bincount(idx, minlength=len(e)).astype(float)
    e, o = np.asarray(e), np.asarray(o)
    stat = ((o - e) ** 2 / e).sum()
    dof = len(e) - 1
    return stat, dof, stats.chi2.sf(stat, dof)


def main(binary):
    rows = []

    def record(name, test, stat, p, zm, zv):
        ok = p > 1e-4 and abs(zm) < 5 and abs(zv) < 5
        rows.append((name, test, stat, p, zm, zv, ok))

    x = draw(binary, "uniform")
    ks = stats.kstest(x, stats.uniform.cdf)
    record("uniform[0,1)", "KS", ks.statistic, ks.pvalue, *moment_z(x, 0.5, 1 / 12))

    x = draw(binary, "normal")
    ks = stats.kstest(x, stats.norm.cdf)
    record("normal(0,1)", "KS", ks.statistic, ks.pvalue, *moment_z(x, 0.0, 1.0))

    for mu, sigma in ((0.0, 1.0), (8.0, 0.5), (-3.0, 2.0)):
        x = draw(binary, "lognormal", mu, sigma)
        ks = stats.kstest(np.log(x), stats.norm(mu, sigma).cdf)
        zm, zv = moment_z(np.log(x), mu, sigma * sigma)
        record(f"lognormal({mu},{sigma}) [log scale]", "KS", ks.statistic, ks.pvalue, zm, zv)

    for shape in (0.01, 0.1, 0.5, 1.0, 2.5, 30.0, 1e4, 1e8, 1e12):
        x = draw(binary, "gamma", shape)
        if shape >= 1e8:
            # Gamma(a) = a + sqrt(a) Z + O(1): standardise; skewness 2/sqrt(a) is below the KS resolution
            z = (x - shape) / math.sqrt(shape)
            ks = stats.kstest(z, stats.norm.cdf)
            zm, zv = moment_z(z, 0.0, 1.0)
        else:
            ks = stats.kstest(x, stats.gamma(shape).cdf)
            zm, zv = moment_z(x, shape, shape)
        record(f"gamma({shape:g})", "KS", ks.statistic, ks.pvalue, zm, zv)

    for lam in (0.5, 3.0, 9.99, 10.0, 25.0, 1e3, 1e6, 1e9):
        x = draw(binary, "poisson", lam)
        stat, dof, p = chi_square(x, stats.poisson(lam))
        record(f"poisson({lam:g})", f"chi2[{dof}]", stat, p, *moment_z(x, lam, lam))

    for trials, p in ((10, 0.3), (100, 0.2), (50, 0.9), (1000, 0.3), (1000, 0.95), (1_000_000, 0.5),
                      (2_000_000_000, 1e-6), (2_147_483_647, 0.5), (5, 1.0), (7, 0.0)):
        x = draw(binary, "binomial", trials, p)
        var = trials * p * (1 - p)
        if p in (0.0, 1.0):
            ok = bool(np.all(x == trials * p))
            rows.append((f"binomial({trials},{p:g})", "exact", 0.0, 1.0 if ok else 0.0, 0.0, 0.0, ok))
            continue
        stat, dof, pv = chi_square(x, stats.binom(trials, p))
        record(f"binomial({trials},{p:g})", f"chi2[{dof}]", stat, pv, *moment_z(x, trials * p, var))

    width = max(len(r[0]) for r in rows)
    print(f"{'distribution':<{width}}  test       statistic    p-value   z(mean)  z(var)  ok")
    for name, test, stat, p, zm, zv, ok in rows:
        print(f"{name:<{width}}  {test:<9} {stat:10.4g}  {p:9.3g}  {zm:7.2f}  {zv:6.2f}  {'yes' if ok else 'NO'}")
    failed = [r[0] for r in rows if not r[-1]]
    print(f"\n{len(rows) - len(failed)} of {len(rows)} pass" + (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
