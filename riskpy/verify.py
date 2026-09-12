"""The verification suite — every identity the library claims, checked.

A test suite tells you whether the code still does what it did yesterday. This
module answers a different question: whether what it does is *right*, by
computing a value with the library and comparing it to something the library
had no hand in — a closed form, a published table, a mathematical identity, or
the same quantity by an unrelated route.

    from riskpy import verify

    report = verify.run()
    print(report.summary())
    assert report          # False if any check failed

    python -m riskpy.verify              # the same from the command line
    python -m riskpy.verify --bench      # plus timings, for speed regressions
    python -m riskpy.verify --json out.json

Every check is a :class:`Check` — a name, the value the library produced, the
reference it should match, and an **absolute** tolerance. Tolerances are
chosen per check and stated: machine precision for exact identities, the
printed precision for values transcribed from a book, and a few standard
errors for anything Monte Carlo (every stochastic check runs on a fixed seed,
so it is reproducible, and it is loose enough never to fail by chance).

The suite runs in CI on every push and nightly, and the documentation site
publishes the latest report, so a regression in the mathematics shows up as a
red row rather than as a wrong number in someone's report six months later.

Modules contribute checks through a private ``_verification_checks()`` hook
returning ``[(name, value, reference, tolerance), …]``; this module adds its
own for the compiled core, the special functions, the simulation engine, the
pricing layer, and the chart palette. SciPy is never required, but if it is
installed the special functions are also checked against it as an independent
oracle.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["Check", "Report", "run", "checks", "benchmark", "MODULES", "main"]

# The order the report lists them in.
MODULES: Tuple[str, ...] = (
    "core",
    "special",
    "mc",
    "quant",
    "life",
    "reserving",
    "rates",
    "credit",
    "viz",
)


@dataclass(frozen=True)
class Check:
    """One verified quantity: what the library said, what it should have said."""

    module: str
    name: str
    value: float
    reference: float
    tolerance: float

    @property
    def error(self) -> float:
        if math.isnan(self.value) or math.isnan(self.reference):
            return math.inf
        if math.isinf(self.value) or math.isinf(self.reference):
            return 0.0 if self.value == self.reference else math.inf
        return abs(self.value - self.reference)

    @property
    def passed(self) -> bool:
        return self.error <= self.tolerance

    @property
    def margin(self) -> float:
        """``error / tolerance`` — below 1 passes; how far below is the headroom."""
        if self.tolerance <= 0.0:
            return 0.0 if self.error == 0.0 else math.inf
        return self.error / self.tolerance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "name": self.name,
            "value": self.value,
            "reference": self.reference,
            "tolerance": self.tolerance,
            "error": self.error,
            "passed": self.passed,
        }


@dataclass
class Report:
    """The outcome of a run — checks, timing, and the ways people read it."""

    checks: List[Check]
    seconds: float
    version: str
    benchmarks: Dict[str, float] = field(default_factory=dict)
    oracle: bool = False

    @property
    def passed(self) -> List[Check]:
        return [c for c in self.checks if c.passed]

    @property
    def failed(self) -> List[Check]:
        return [c for c in self.checks if not c.passed]

    @property
    def ok(self) -> bool:
        return not self.failed and bool(self.checks)

    def __bool__(self) -> bool:
        return self.ok

    def by_module(self) -> Dict[str, List[Check]]:
        grouped: Dict[str, List[Check]] = {}
        for check in self.checks:
            grouped.setdefault(check.module, []).append(check)
        return grouped

    def summary(self, verbose: bool = False) -> str:
        """A text table: one line per module, every failure spelled out, and
        every check listed if ``verbose``."""
        lines = [
            f"RiskPY {self.version} verification — {len(self.passed)}/{len(self.checks)} checks pass "
            f"in {self.seconds:.2f}s" + ("  (SciPy oracle on)" if self.oracle else ""),
            "",
            f"  {'module':<10} {'pass':>5} {'fail':>5}   worst margin",
        ]
        for module, group in self.by_module().items():
            fails = [c for c in group if not c.passed]
            worst = max(group, key=lambda c: c.margin)
            lines.append(
                f"  {module:<10} {len(group) - len(fails):>5} {len(fails):>5}   "
                f"{worst.margin:.2g}  ({worst.name[:48]})"
            )
        if self.failed:
            lines += ["", "FAILED:"]
            for c in self.failed:
                lines.append(
                    f"  [{c.module}] {c.name}\n"
                    f"      value {c.value!r}  reference {c.reference!r}  "
                    f"error {c.error:.3g} > tolerance {c.tolerance:.3g}"
                )
        if verbose:
            lines += ["", "ALL CHECKS:"]
            for c in self.checks:
                flag = "ok " if c.passed else "BAD"
                lines.append(f"  {flag} [{c.module}] {c.name:<62} err {c.error:.2e}  tol {c.tolerance:.0e}")
        if self.benchmarks:
            lines += ["", "BENCHMARKS (seconds):"]
            for name, secs in self.benchmarks.items():
                lines.append(f"  {name:<58} {secs:8.4f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "seconds": self.seconds,
            "ok": self.ok,
            "passed": len(self.passed),
            "failed": len(self.failed),
            "oracle": self.oracle,
            "checks": [c.to_dict() for c in self.checks],
            "benchmarks": self.benchmarks,
        }

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent)
        if path:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text + "\n")
        return text

    def to_markdown(self) -> str:
        """A Markdown table, the form the documentation site publishes."""
        rows = [
            f"**RiskPY {self.version}** — {len(self.passed)} of {len(self.checks)} checks pass "
            f"({self.seconds:.1f}s)" + (", SciPy oracle on" if self.oracle else ""),
            "",
            "| module | check | value | reference | error | tolerance | |",
            "|---|---|---:|---:|---:|---:|:-:|",
        ]
        for c in self.checks:
            rows.append(
                f"| `{c.module}` | {c.name} | {_fmt(c.value)} | {_fmt(c.reference)} | "
                f"{c.error:.1e} | {c.tolerance:.0e} | {'✅' if c.passed else '❌'} |"
            )
        if self.benchmarks:
            rows += ["", "| benchmark | seconds |", "|---|---:|"]
            for name, secs in self.benchmarks.items():
                rows.append(f"| {name} | {secs:.4f} |")
        return "\n".join(rows)


def _fmt(x: float) -> str:
    if x == 0.0:
        return "0"
    if abs(x) >= 1e5 or abs(x) < 1e-3:
        return f"{x:.6g}"
    return f"{x:.6f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# Check sources
# ---------------------------------------------------------------------------

Raw = Tuple[str, float, float, float]


def _core_checks() -> List[Raw]:
    """The compiled engine, against hand arithmetic and the Python layer."""
    import riskpy

    out: List[Raw] = []

    model = riskpy.FactorModel(1000.0)
    model.add_multiplier("state", "FL", 3.0)
    model.add_numeric_band_multiplier("age", 16, 25, 2.0)
    out.append(("FactorModel: 1000 × 3.0 (FL) × 2.0 (age 19)",
                model.calculate({"state": "FL", "age": 19.0}), 6000.0, 1e-9))
    out.append(("FactorModel: unmatched inputs leave the base rate alone",
                model.calculate({"state": "NY", "age": 40.0}), 1000.0, 1e-9))

    pv = riskpy.ActuarialMath.present_value(0.05, 20, 50_000.0)
    out.append(("ActuarialMath: PV of 20-year annuity-immediate at 5%",
                pv, 50_000.0 * (1 - 1.05 ** -20) / 0.05, 1e-6))
    fv = riskpy.ActuarialMath.future_value(0.06, 30, 5_000.0)
    out.append(("ActuarialMath: FV of 30 payments at 6%",
                fv, 5_000.0 * (1.06 ** 30 - 1) / 0.06, 1e-6))

    tri = riskpy.LossTriangle()
    tri.add_origin_year(2019, [1000.0, 1500.0, 1800.0, 2000.0])
    tri.add_origin_year(2020, [1200.0, 1800.0, 2160.0])
    tri.add_origin_year(2021, [1400.0, 2100.0])
    tri.add_origin_year(2022, [1600.0])
    # LDFs 1.5, 1.2, 10/9 → ultimates 2000, 2400, 2800, 3200 → IBNR 2540.
    out.append(("LossTriangle: hand-checked chain ladder IBNR", sum(tri.get_ibnr_reserves()), 2540.0, 1e-6))

    # Severity always 1 → aggregate is Poisson(λ) exactly.
    lam = 2.0
    pmf = riskpy.FourierTransform.compound_poisson_pmf([0.0, 1.0], lam, 64)
    poisson = [math.exp(-lam) * lam ** k / math.factorial(k) for k in range(12)]
    out.append(("FourierTransform: compound Poisson with unit severity = Poisson(2)",
                max(abs(pmf[k] - poisson[k]) for k in range(12)), 0.0, 1e-12))
    out.append(("FourierTransform: PMF has unit mass", sum(pmf), 1.0, 1e-12))
    conv = riskpy.FourierTransform.convolve([1.0, 2.0, 3.0], [4.0, 5.0])
    out.append(("FourierTransform: convolution [1,2,3]*[4,5]",
                max(abs(a - b) for a, b in zip(conv, [4.0, 13.0, 22.0, 15.0])), 0.0, 1e-10))

    exp_rating = riskpy.ExperienceRating(1082.0, 0.0)
    out.append(("ExperienceRating: full credibility at k claims", exp_rating.calculate_credibility(1082), 1.0, 1e-12))
    out.append(("ExperienceRating: √(n/k) below the standard",
                exp_rating.calculate_credibility(270), math.sqrt(270.0 / 1082.0), 1e-12))
    exposure = riskpy.ExposureRating(100_000.0, 0.7)
    out.append(("ExposureRating: ILF at the base limit is 1", exposure.increased_limits_factor(100_000.0), 1.0, 1e-12))
    out.append(("ExposureRating: burning cost of a 100 xs 100 layer",
                exposure.burning_cost([50.0, 150.0, 250.0, 400.0], 100.0, 100.0, 4), (0 + 50 + 100 + 100) / 4.0, 1e-9))
    analyzer = riskpy.RateAnalyzer(1.0, 0.30)
    out.append(("RateAnalyzer: indicated change at a 77% loss ratio",
                analyzer.required_rate_change(0.77), 0.77 / 0.70 - 1.0, 1e-12))
    out.append(("RateAnalyzer: on-level factor compounds rate changes",
                analyzer.on_level_factor([0.05, 0.10, -0.02]), 1.05 * 1.10 * 0.98, 1e-12))

    sim = riskpy.MonteCarloSimulator(trials=200_000, seed=7)
    losses = sim.simulate_aggregate_loss(5.0, 8.0, 0.5)
    expected = 5.0 * math.exp(8.0 + 0.125)
    out.append(("MonteCarloSimulator: aggregate mean = λ·E[severity] (200k trials, ±3 s.e.)",
                sum(losses) / len(losses), expected, 3.0 * _agg_se(5.0, 8.0, 0.5, 200_000)))
    return out


def _agg_se(lam: float, mu: float, sigma: float, n: int) -> float:
    """Standard error of the compound Poisson sample mean, for the tolerance."""
    second = math.exp(2 * mu + 2 * sigma * sigma)
    return math.sqrt(lam * second / n)


def _special_checks() -> List[Raw]:
    """Special functions against closed forms — no oracle needed."""
    from . import _special as sp

    out: List[Raw] = []
    out.append(("norm_cdf(0) = 1/2", sp.norm_cdf(0.0), 0.5, 1e-16))
    out.append(("norm_cdf(-37) keeps its tail (erf-based forms give 0)", float(sp.norm_cdf(-37.0) > 0.0), 1.0, 0.0))
    out.append(("norm_ppf(norm_cdf(x)) = x at x = -3.7", sp.norm_ppf(sp.norm_cdf(-3.7)), -3.7, 1e-13))
    out.append(("norm_ppf(0.975) = 1.959964", sp.norm_ppf(0.975), 1.959963984540054, 1e-14))
    out.append(("gammainc(1, x) = 1 - e^-x", sp.gammainc(1.0, 2.3), 1.0 - math.exp(-2.3), 1e-14))
    out.append(("gammainc + gammaincc = 1", sp.gammainc(3.7, 2.2) + sp.gammaincc(3.7, 2.2), 1.0, 1e-14))
    out.append(("gammainc(n, x) = Poisson survival (n=4, x=2.5)",
                sp.gammainc(4.0, 2.5), 1.0 - sum(math.exp(-2.5) * 2.5 ** k / math.factorial(k) for k in range(4)), 1e-14))
    out.append(("betainc(1, 1, x) = x", sp.betainc(1.0, 1.0, 0.37), 0.37, 1e-14))
    out.append(("betainc symmetry I_x(a,b) = 1 - I_{1-x}(b,a)",
                sp.betainc(2.5, 4.0, 0.3) + sp.betainc(4.0, 2.5, 0.7), 1.0, 1e-14))
    out.append(("betainc(a, 1, x) = x^a", sp.betainc(3.0, 1.0, 0.6), 0.6 ** 3, 1e-14))
    out.append(("t_cdf with 1 d.o.f. is Cauchy: ½ + atan(x)/π", sp.t_cdf(1.3, 1.0), 0.5 + math.atan(1.3) / math.pi, 1e-13))
    out.append(("t_ppf(t_cdf(x)) = x, 4 d.o.f.", sp.t_ppf(sp.t_cdf(2.1, 4.0), 4.0), 2.1, 1e-9))
    out.append(("chi2_cdf with 2 d.o.f. = 1 - e^{-x/2}", sp.chi2_cdf(3.0, 2.0), 1.0 - math.exp(-1.5), 1e-14))
    out.append(("digamma(1) = -Euler–Mascheroni", sp.digamma(1.0), -0.5772156649015329, 1e-13))
    out.append(("trigamma(1) = π²/6", sp.trigamma(1.0), math.pi ** 2 / 6.0, 1e-13))
    out.append(("brent solves x³ = 2", sp.brent(lambda x: x ** 3 - 2.0, 0.0, 2.0), 2.0 ** (1.0 / 3.0), 1e-12))
    out.append(("simpson ∫₀^π sin = 2", sp.simpson(math.sin, 0.0, math.pi, 200), 2.0, 1e-9))
    xm, _ = sp.nelder_mead(lambda v: (v[0] - 1.0) ** 2 + 10.0 * (v[1] + 2.0) ** 2, [0.0, 0.0])
    out.append(("nelder_mead finds (1, -2)", abs(xm[0] - 1.0) + abs(xm[1] + 2.0), 0.0, 1e-5))
    out.append(("kolmogorov_sf(1) = 0.26999967", sp.kolmogorov_sf(1.0), 0.26999967167735456, 1e-12))
    return out


def _oracle_checks() -> List[Raw]:
    """The special functions against SciPy, when SciPy happens to be present."""
    from . import _special as sp

    try:
        from scipy import special, stats  # noqa: WPS433
    except ImportError:
        return []

    out: List[Raw] = []
    for p in (1e-12, 1e-6, 0.001, 0.3, 0.9, 1.0 - 1e-9):
        out.append((f"oracle: norm_ppf({p:g}) vs scipy", sp.norm_ppf(p), float(stats.norm.ppf(p)), 1e-13 * max(1.0, abs(stats.norm.ppf(p)))))
    for a, x in ((0.3, 0.2), (2.5, 4.0), (30.0, 25.0)):
        out.append((f"oracle: gammainc({a}, {x}) vs scipy", sp.gammainc(a, x), float(special.gammainc(a, x)), 1e-13))
    for a, b, x in ((0.5, 0.5, 0.3), (2.0, 5.0, 0.6), (10.0, 0.7, 0.95)):
        out.append((f"oracle: betainc({a}, {b}, {x}) vs scipy", sp.betainc(a, b, x), float(special.betainc(a, b, x)), 1e-13))
    for p, df in ((0.01, 3.0), (0.975, 12.0)):
        out.append((f"oracle: t_ppf({p}, {df}) vs scipy", sp.t_ppf(p, df), float(stats.t.ppf(p, df)), 1e-8))
    return out


def _mc_checks() -> List[Raw]:
    """The simulation engine: analytic layer, sampling, dependence."""
    from . import mc

    np = mc._numpy()
    out: List[Raw] = []

    families = [
        mc.Normal(2.0, 3.0), mc.LogNormal(1.2, 0.7), mc.Gamma(2.5, 1.7), mc.Beta(2.0, 5.0),
        mc.Weibull(1.5, 2.0), mc.StudentT(5.0, 1.0, 2.0), mc.Pareto(2.0, 3.5), mc.PERT(0.0, 2.0, 10.0),
        mc.Triangular(0.0, 1.0, 5.0), mc.Exponential(3.0), mc.Uniform(-1.0, 4.0),
    ]
    q = np.array([0.001, 0.05, 0.5, 0.95, 0.999])
    for dist in families:
        back = dist.cdf(dist.ppf(q))
        out.append((f"{type(dist).__name__}: cdf(ppf(q)) = q", float(np.max(np.abs(back - q))), 0.0, 1e-9))

    rng = np.random.default_rng(2026)
    for dist in (mc.Gamma(2.5, 1.7), mc.LogNormal(1.2, 0.7), mc.Poisson(140.0),
                 mc.NegativeBinomial.from_mean_dispersion(140.0, 2.4), mc.Weibull(1.5, 2.0)):
        m = dist.moments()
        sample = dist.sample(200_000, rng)
        out.append((f"{type(dist).__name__}: sample mean = analytic mean (200k, ±4 s.e.)",
                    float(sample.mean()), m.mean, 4.0 * m.sd / math.sqrt(200_000)))
        lhs = dist.sample_lhs(200_000, rng)
        out.append((f"{type(dist).__name__}: LHS mean within 1 s.e. (variance reduction)",
                    float(lhs.mean()), m.mean, 1.0 * m.sd / math.sqrt(200_000)))

    # Rank correlation is imposed exactly on the ranks, so the recovered
    # Spearman is off only by the score-matrix noise, well under 0.01 at 100k.
    model = mc.Model(a=mc.LogNormal(8.0, 1.0), b=mc.Gamma(2.0, 3.0), c=mc.Poisson(30.0))
    model.correlate("a", "b", 0.7).correlate("a", "c", -0.4)
    drawn = model.sample(100_000, seed=11)
    rank = lambda x: mc._rankdata(np.asarray(x), np)  # noqa: E731
    out.append(("Iman–Conover: Spearman(a, b) = 0.7", float(np.corrcoef(rank(drawn["a"]), rank(drawn["b"]))[0, 1]), 0.7, 0.01))
    out.append(("Iman–Conover: Spearman(a, c) = -0.4", float(np.corrcoef(rank(drawn["a"]), rank(drawn["c"]))[0, 1]), -0.4, 0.01))
    out.append(("Iman–Conover: unrelated pair stays at 0", float(np.corrcoef(rank(drawn["b"]), rank(drawn["c"]))[0, 1]), 0.0, 0.01))
    independent = mc.LogNormal(8.0, 1.0).sample(100_000, np.random.default_rng(11))
    out.append(("Iman–Conover: marginal untouched (same draws, only re-paired)",
                float(np.abs(np.sort(drawn["a"]) - np.sort(independent)).max()), 0.0, 1e-9))

    trunc = mc.Truncated(mc.Normal(0.0, 1.0), low=-1.0, high=2.0)
    grid = np.linspace(-1.0, 2.0, 20_001)
    out.append(("Truncated: density integrates to 1", float(getattr(np, 'trapezoid', getattr(np, 'trapz', None))(trunc.pdf(grid), grid)), 1.0, 1e-6))
    mix = mc.Mixture([mc.LogNormal(8.0, 1.0), mc.Pareto(50_000.0, 2.5)], [0.9, 0.1])
    parts = [c.moments().mean for c in mix.components]
    out.append(("Mixture: mean is the weighted mean", mix.moments().mean, 0.9 * parts[0] + 0.1 * parts[1], 1e-9))

    result = mc.simulate(lambda x: x, trials=50_000, seed=3, x=mc.Normal(0.0, 1.0))
    out.append(("Result.var(0.99) = np.percentile(99)", result.var(0.99), float(np.percentile(result.values, 99.0)), 1e-12))
    out.append(("Result.tvar ≥ Result.var", float(result.tvar(0.99) >= result.var(0.99)), 1.0, 0.0))
    return out


def _quant_checks() -> List[Raw]:
    from . import quant

    out: List[Raw] = []
    out.append(("Black–Scholes ATM call, S=K=100, T=1, r=5%, σ=20%",
                quant.black_scholes(100, 100, 1.0, 0.05, 0.2), 10.450583572185565, 1e-9))
    S, K, T, r, sigma, q = 103.0, 95.0, 0.75, 0.031, 0.27, 0.012
    call = quant.black_scholes(S, K, T, r, sigma, q, "call")
    put = quant.black_scholes(S, K, T, r, sigma, q, "put")
    out.append(("put–call parity", call - put, S * math.exp(-q * T) - K * math.exp(-r * T), 1e-10))
    h = 1e-5
    bumped = (quant.black_scholes(S + h, K, T, r, sigma, q) - quant.black_scholes(S - h, K, T, r, sigma, q)) / (2 * h)
    out.append(("delta = ∂price/∂S (central difference)", quant.greeks(S, K, T, r, sigma, q).delta, bumped, 1e-6))
    iv = quant.implied_vol(call, S, K, T, r, q, "call")
    out.append(("implied_vol(price(σ)) = σ", iv, sigma, 1e-7))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        heston = quant.heston_price(100, 100, 1.0, 0.03, v0=0.04, kappa=2.0, theta=0.04, xi=1e-6, rho=0.0)
    out.append(("Heston → Black–Scholes as vol-of-vol → 0", heston, quant.black_scholes(100, 100, 1.0, 0.03, 0.2), 1e-3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hc = quant.heston_price(100, 90, 0.9, 0.025, 0.05, 1.4, 0.045, 0.45, -0.65, kind="call")
        hp = quant.heston_price(100, 90, 0.9, 0.025, 0.05, 1.4, 0.045, 0.45, -0.65, kind="put")
    out.append(("Heston put–call parity", hc - hp, 100.0 - 90.0 * math.exp(-0.025 * 0.9), 1e-6))
    out.append(("parametric_var at 99% = 2.326348 σ", quant.parametric_var(level=0.99, mean=0.0, sd=1.0), 2.3263478740408408, 1e-9))
    try:
        np = quant._numpy()
        paths = quant.gbm_paths(100.0, 0.07, 0.22, 1.0, steps=50, trials=100_000, seed=5)
        out.append(("GBM terminal mean = S₀e^{μT} (100k paths, ±4 s.e.)",
                    float(paths[:, -1].mean()), 100.0 * math.exp(0.07),
                    4.0 * 100.0 * math.exp(0.07) * math.sqrt(math.exp(0.22 ** 2) - 1.0) / math.sqrt(100_000)))
    except ImportError:  # pragma: no cover - NumPy absent
        pass
    return out


def _viz_checks() -> List[Raw]:
    """The palette's accessibility claims, computed rather than asserted."""
    from . import viz

    out: List[Raw] = []
    for mode in ("dark", "light"):
        palette = viz.PALETTE[mode]
        worst = min(_contrast(colour, palette["surface"]) for colour in palette["series"])
        out.append((f"{mode} palette: every series colour ≥ 3:1 on its surface (worst)", float(worst >= 3.0), 1.0, 0.0))
        lum = [_luminance(c) for c in palette["sequential"]]
        monotone = all(b > a for a, b in zip(lum, lum[1:])) or all(b < a for a, b in zip(lum, lum[1:]))
        out.append((f"{mode} sequential ramp: lightness monotone", float(monotone), 1.0, 0.0))
        out.append((f"{mode} palette: eight categorical slots, no more", float(len(palette["series"])), 8.0, 0.0))
    return out


def _luminance(hex_colour: str) -> float:
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255.0 for i in (1, 3, 5))

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _contrast(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _module_checks(name: str) -> List[Raw]:
    module = importlib.import_module(f"riskpy.{name}")
    return list(module._verification_checks())


_SOURCES: Dict[str, Callable[[], List[Raw]]] = {
    "core": _core_checks,
    "special": _special_checks,
    "mc": _mc_checks,
    "quant": _quant_checks,
    "life": lambda: _module_checks("life"),
    "reserving": lambda: _module_checks("reserving"),
    "rates": lambda: _module_checks("rates"),
    "credit": lambda: _module_checks("credit"),
    "viz": _viz_checks,
}


def checks(modules: Optional[Iterable[str]] = None, oracle: bool = True) -> List[Check]:
    """Compute every check without judging it — the raw material of a report."""
    wanted = list(MODULES) if modules is None else list(modules)
    unknown = [m for m in wanted if m not in _SOURCES]
    if unknown:
        raise ValueError(f"unknown module(s) {unknown}; choose from {list(_SOURCES)}")

    collected: List[Check] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for module in wanted:
            try:
                raw = _SOURCES[module]()
            except ImportError as exc:
                # An optional dependency is missing: record it as a single
                # failed check so the report says so, rather than vanishing.
                collected.append(Check(module, f"import failed: {exc}", math.nan, 0.0, 0.0))
                continue
            for name, value, reference, tolerance in raw:
                collected.append(Check(module, name, _as_float(value), _as_float(reference), float(tolerance)))
        if oracle and "special" in wanted:
            for name, value, reference, tolerance in _oracle_checks():
                collected.append(Check("special", name, _as_float(value), _as_float(reference), float(tolerance)))
    return collected


def _as_float(x: Any) -> float:
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    return float(x)


def run(modules: Optional[Iterable[str]] = None, bench: bool = False, oracle: bool = True) -> Report:
    """Run the suite and return a :class:`Report`.

    ``modules`` restricts the run (``["life", "reserving"]``); ``bench`` adds
    the timing benchmarks; ``oracle=False`` skips the SciPy cross-checks even
    when SciPy is installed.
    """
    from . import __version__

    started = time.perf_counter()
    collected = checks(modules, oracle=oracle)
    has_oracle = oracle and any(c.name.startswith("oracle:") for c in collected)
    timings = benchmark() if bench else {}
    return Report(collected, time.perf_counter() - started, __version__, timings, has_oracle)


# ---------------------------------------------------------------------------
# Benchmarks — the speed half of "constantly tested"
# ---------------------------------------------------------------------------


def benchmark(repeat: int = 3) -> Dict[str, float]:
    """Time a fixed set of representative operations; best of ``repeat``.

    The numbers are for comparison between commits on the same machine, not
    for comparison between machines.
    """
    import riskpy
    from . import mc, quant, reserving

    np = mc._numpy()

    def best(fn: Callable[[], Any]) -> float:
        times = []
        for _ in range(repeat):
            t = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t)
        return min(times)

    model = mc.Model(
        claim_count=mc.Poisson(140.0),
        severity=mc.LogNormal.from_moments(18_000.0, 42_000.0),
        inflation=mc.Normal(0.043, 0.012),
    )
    model.formula(lambda claim_count, severity, inflation: claim_count * severity * (1 + inflation))
    result = model.run(200_000, seed=1)
    rng = np.random.default_rng(1)
    tri = reserving.genins()
    sim = riskpy.MonteCarloSimulator(trials=200_000, seed=1)

    return {
        "mc.simulate 200k trials, 3 inputs": best(lambda: model.run(200_000, seed=1)),
        "mc.simulate 200k trials, LHS + rank correlation": best(
            lambda: mc.simulate(lambda a, b: a * b, trials=200_000, seed=1, sampling="lhs",
                                correlation={("a", "b"): 0.5}, a=mc.Gamma(2.0, 3.0), b=mc.LogNormal(8.0, 1.0))),
        "Gamma.ppf on 200k points": best(lambda: mc.Gamma(2.5, 1.7).ppf(rng.random(200_000))),
        "Result.var + tvar at 6 levels (one sort, then cached)": best(
            lambda: [(fresh.var(l), fresh.tvar(l))
                     for fresh in [mc.Result(result.values, {}, result.trials, 1)]
                     for l in (0.5, 0.75, 0.9, 0.95, 0.99, 0.995)]),
        "Result.sensitivity (Spearman, 3 inputs × 200k)": best(result.sensitivity),
        "quant.black_scholes × 10,000": best(lambda: [quant.black_scholes(100, 90 + i * 0.002, 1.0, 0.05, 0.2) for i in range(10_000)]),
        "quant.heston_price (4096 nodes)": best(lambda: quant.heston_price(100, 100, 1.0, 0.03, 0.04, 2.0, 0.04, 0.3, -0.7)),
        "reserving.chain_ladder GenIns × 100": best(lambda: [reserving.chain_ladder(tri) for _ in range(100)]),
        "reserving.mack_chain_ladder GenIns × 100": best(lambda: [reserving.mack_chain_ladder(tri) for _ in range(100)]),
        "reserving.bootstrap_chain_ladder n=1000": best(lambda: reserving.bootstrap_chain_ladder(tri, n=1000, seed=1)),
        "C++ MonteCarloSimulator aggregate loss 200k": best(lambda: sim.simulate_aggregate_loss(5.0, 8.0, 0.5)),
        "C++ FourierTransform compound Poisson 4096": best(
            lambda: riskpy.FourierTransform.compound_poisson_pmf([0.0] + [1.0 / 200.0] * 200, 20.0, 4096)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="riskpy-verify",
        description="Run the RiskPY verification suite: every identity the library claims, checked.",
    )
    parser.add_argument("-m", "--module", action="append", dest="modules", metavar="NAME",
                        help=f"restrict to a module (repeatable): {', '.join(MODULES)}")
    parser.add_argument("--bench", action="store_true", help="also run the timing benchmarks")
    parser.add_argument("--no-oracle", action="store_true", help="skip the SciPy cross-checks")
    parser.add_argument("--json", metavar="PATH", help="write the full report as JSON")
    parser.add_argument("--markdown", metavar="PATH", help="write the report as a Markdown table")
    parser.add_argument("-v", "--verbose", action="store_true", help="list every check, not only failures")
    parser.add_argument("-q", "--quiet", action="store_true", help="print nothing; exit status only")
    args = parser.parse_args(argv)

    # Check names carry λ, Φ, ä and friends. A Windows console in a legacy
    # code page raises on those, and a verifier that crashes while reporting
    # a pass is worse than useless — so degrade the glyphs, never the run.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="backslashreplace")
            except (ValueError, OSError):  # pragma: no cover - closed or odd streams
                pass

    report = run(args.modules, bench=args.bench, oracle=not args.no_oracle)
    if args.json:
        report.to_json(args.json)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write(report.to_markdown() + "\n")
    if not args.quiet:
        print(report.summary(verbose=args.verbose))
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
