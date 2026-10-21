"""Real-use scenarios and edge cases, runnable against any installed RiskPY.

Each case is a snippet a user would actually write (it must leave its answer in
``result``) plus what the answer should be: a number from an independent
reference (SciPy, a closed form, a published value), a structured error code,
or just "runs and returns something finite".

The same cases drive three things:

* ``regression.ipynb`` runs them against the published 0.5.2 wheel and the
  development build side by side, to show what changed;
* ``tests/test_user_workflows.py`` runs them against the installed package;
* ``python cases.py`` prints a plain table for a terminal.

References are computed here, in this file, from first principles — never by
calling RiskPY — so a case cannot agree with the library by construction.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Independent references
# ---------------------------------------------------------------------------


def _norm():
    from scipy import stats

    return stats.norm


def bs_ref(S, K, T, r, sigma, q=0.0, kind="call"):
    """Black–Scholes with SciPy's normal, in the numerically safe arrangement."""
    N = _norm()
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S * math.exp(-q * T) * N.cdf(d1) - K * math.exp(-r * T) * N.cdf(d2)
    return K * math.exp(-r * T) * N.cdf(-d2) - S * math.exp(-q * T) * N.cdf(-d1)


def cds_ref(h, R, r, M, freq=4):
    """Par spread with default at any instant: protection and accrued premium integrated exactly."""
    from scipy.integrate import quad

    dt = 1.0 / freq
    protection = (1 - R) * quad(lambda t: h * math.exp(-(h + r) * t), 0, M, limit=200)[0]
    premium = 0.0
    for k in range(1, int(round(M * freq)) + 1):
        a, b = (k - 1) * dt, k * dt
        premium += dt * math.exp(-(h + r) * b)
        premium += quad(lambda t: (t - a) * h * math.exp(-(h + r) * t), a, b)[0]
    return protection / premium


def merton_spread_ref(V, D, T, r, sigma):
    from scipy.special import log_ndtr

    d1 = (math.log(V / D) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    import numpy as np

    log_debt = np.logaddexp(math.log(D) - r * T + log_ndtr(d2), math.log(V) + log_ndtr(-d1))
    return -(float(log_debt) - math.log(D)) / T - r


def basel_rw_ref(pd, lgd, M=2.5):
    """BCBS corporate IRB risk weight, written out by hand."""
    N = _norm()
    w = (1 - math.exp(-50 * pd)) / (1 - math.exp(-50))
    R = 0.12 * w + 0.24 * (1 - w)
    b = (0.11852 - 0.05478 * math.log(pd)) ** 2
    K = lgd * (N.cdf((N.ppf(pd) + math.sqrt(R) * N.ppf(0.999)) / math.sqrt(1 - R)) - pd)
    K *= (1 + (M - 2.5) * b) / (1 - 1.5 * b)
    return 12.5 * K


def ler_ref(d, mean, cv):
    """Lognormal loss elimination ratio E[min(X, d)] / E[X] by quadrature."""
    from scipy import integrate, stats

    s2 = math.log1p(cv * cv)
    dist = stats.lognorm(s=math.sqrt(s2), scale=math.exp(math.log(mean) - 0.5 * s2))
    return integrate.quad(dist.sf, 0, d, limit=200)[0] / mean


def references() -> Dict[str, float]:
    """Every reference value, computed once. Needs SciPy."""
    from scipy import special, stats

    return {
        "bs_atm": bs_ref(100, 100, 1, 0.05, 0.2),
        "bs_far_otm_put": bs_ref(100, 30, 0.5, 0.0, 0.2, kind="put"),
        "bs_3x_call": bs_ref(100, 300, 0.5, 0.0, 0.2),
        "cds_ig": cds_ref(0.02, 0.4, 0.03, 5),
        "cds_distressed": cds_ref(5.0, 0.4, 0.03, 5),
        "cds_defaulting": cds_ref(50.0, 0.4, 0.0, 5),
        "merton_mid": merton_spread_ref(100, 100, 5, 0.05, 0.10),
        "merton_long": merton_spread_ref(100, 100, 100, 0.05, 10.0),
        "basel_1pct": basel_rw_ref(0.01, 0.45),
        "ler_cv2": ler_ref(2000, 1000, 2.0),
        "ler_cv05": ler_ref(1000, 1000, 0.5),
        "chi2_cdf_big": float(stats.chi2.cdf(1e6, 1e6)),
        "t_ppf_tail": float(stats.t.ppf(1e-20, 5)),
        "gammainc_big": float(special.gammainc(1e5, 1e5)),
        "norm_ppf_sub": float(stats.norm.ppf(1e-320)),
    }


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@dataclass
class Case:
    section: str
    label: str
    code: str
    expect: Dict[str, Any] = field(default_factory=dict)
    why: str = ""

    # expect is one of:
    #   {"ref": key or number, "rel": tol, "abs": tol}
    #   {"raises": "RP1"}               — a RiskPyError whose code starts with this
    #   {"finite": True}                — runs, and every number in the result is finite
    #   {"true": True}                  — result is True


CASES: List[Case] = [
    # -- Options desk -------------------------------------------------------------------
    Case("Options desk", "ATM call, the textbook number",
         "from riskpy.quant import black_scholes\nresult = black_scholes(100, 100, 1, 0.05, 0.2)",
         {"ref": "bs_atm", "rel": 1e-12}),
    Case("Options desk", "Far OTM put (30% strike, 6m) — a tail-hedging wing",
         "from riskpy.quant import black_scholes\nresult = black_scholes(100, 30, 0.5, 0.0, 0.2, kind='put')",
         {"ref": "bs_far_otm_put", "rel": 1e-6},
         "erf-based N(x) rounded the price to exactly 0"),
    Case("Options desk", "Deep OTM call (3× strike)",
         "from riskpy.quant import black_scholes\nresult = black_scholes(100, 300, 0.5, 0.0, 0.2)",
         {"ref": "bs_3x_call", "rel": 1e-6}),
    Case("Options desk", "Implied vol of a cheap 2× strike call",
         "from riskpy.quant import black_scholes, implied_vol\n"
         "p = black_scholes(100, 200, 0.25, 0.0, 0.2)\nresult = implied_vol(p, 100, 200, 0.25, 0.0)",
         {"ref": 0.2, "rel": 1e-8}, "absolute 1e-8 price tolerance stopped at the first midpoint"),
    Case("Options desk", "Implied vol at 2000% vol (price ≡ S)",
         "from riskpy.quant import black_scholes, implied_vol\n"
         "result = implied_vol(black_scholes(100, 100, 1, 0.05, 20), 100, 100, 1, 0.05)",
         {"raises": "RP30"}, "returned 15.0 with no error"),
    Case("Options desk", "Zero-vol forward: delta of a call whose forward is OTM",
         "from riskpy.quant import greeks\nresult = greeks(100, 95, 1, 0.0, 0.0, q=0.10).delta",
         {"ref": 0.0, "abs": 1e-15}, "delta decided by S vs K, not the forward"),
    Case("Options desk", "Zero-vol forward: theta per day",
         "from riskpy.quant import greeks\nresult = greeks(100, 99, 1, 0.05, 0.0).theta",
         {"ref": -0.05 * 99 * math.exp(-0.05) / 365, "rel": 1e-10}),
    Case("Options desk", "Heston at expiry is intrinsic",
         "from riskpy.quant import heston_price\nresult = heston_price(100, 110, 0, 0.03, 0.04, 1.5, 0.04, 0.3, -0.7)",
         {"ref": 0.0, "abs": 1e-12}, "quadrature error gave a negative price"),
    Case("Options desk", "NaN spot from a bad market-data feed",
         "from riskpy.quant import black_scholes\nresult = black_scholes(float('nan'), 100, 1, 0.05, 0.2)",
         {"raises": "RP101"}, "returned nan silently"),

    # -- Market risk ---------------------------------------------------------------------
    Case("Market risk", "ES of an empty P&L vector",
         "from riskpy.quant import expected_shortfall\nresult = expected_shortfall([])",
         {"raises": "RP111"}, "raw IndexError"),
    Case("Market risk", "ES with level given as a percentage-ish 1.5",
         "from riskpy.quant import expected_shortfall\nresult = expected_shortfall([0.01, -0.02, 0.005], 1.5)",
         {"raises": "RP110"}),
    Case("Market risk", "Parametric VaR with a negative sd",
         "from riskpy.quant import parametric_var\nresult = parametric_var(mean=0.0, sd=-0.1)",
         {"raises": "RP102"}, "returned a negative VaR"),
    Case("Market risk", "Historical ES ≥ VaR on fat-tailed returns",
         "import numpy as np\nfrom riskpy.quant import expected_shortfall, historical_var\n"
         "x = np.random.default_rng(1).standard_t(3, 2000) * 0.01\n"
         "result = bool(expected_shortfall(x, 0.99) >= historical_var(x, 0.99))",
         {"true": True}),

    # -- Rates & ALM -----------------------------------------------------------------------
    Case("Rates & ALM", "Nelson–Siegel +100bp: short end",
         "from riskpy.rates import YieldCurve\nresult = YieldCurve.nelson_siegel(0.04, -0.02, 0.01, 2).shift(100).zero(0)",
         {"ref": 0.03, "abs": 1e-14}, "shift dropped the formula: 0.030614"),
    Case("Rates & ALM", "Nelson–Siegel +100bp: 60y (beyond the knots)",
         "from riskpy.rates import YieldCurve\nc = YieldCurve.nelson_siegel(0.04, -0.02, 0.01, 2)\n"
         "result = c.shift(100).zero(60) - c.zero(60)",
         {"ref": 0.01, "abs": 1e-14}),
    Case("Rates & ALM", "Vasicek curve short rate",
         "from riskpy.rates import vasicek_curve\nresult = vasicek_curve(0.03, 0.5, 0.05, 0.01).zero(0)",
         {"ref": 0.03, "abs": 1e-14}, "interpolated sample gave 0.03041"),
    Case("Rates & ALM", "Vasicek bond with almost no mean reversion",
         "from riskpy.rates import vasicek_zero_bond\nresult = vasicek_zero_bond(0.03, 0, 5, 1e-10, 0.05, 0.01)",
         {"ref": math.exp(-0.03 * 5 + 0.01 ** 2 * 125 / 6), "rel": 1e-8}, "cancellation gave a price of 1.45"),
    Case("Rates & ALM", "Discount factor that overflows",
         "from riskpy.rates import discount_factor\nresult = discount_factor(-10, 1000)",
         {"raises": "RP304"}, "raw OverflowError"),
    Case("Rates & ALM", "5% daily → continuous",
         "from riskpy.rates import convert_rate\nresult = convert_rate(0.05, 'daily', 'continuous')",
         {"ref": 365 * math.log1p(0.05 / 365), "rel": 1e-15}),
    Case("Rates & ALM", "Bootstrap a 1–10y par curve",
         "from riskpy.rates import YieldCurve\n"
         "c = YieldCurve.bootstrap([0.030, 0.032, 0.034, 0.035, 0.036, 0.037, 0.0375, 0.038, 0.0385, 0.039],"
         " list(range(1, 11)), frequency=1)\nresult = [c.par_rate(t) for t in range(1, 11)]",
         {"ref": [0.030, 0.032, 0.034, 0.035, 0.036, 0.037, 0.0375, 0.038, 0.0385, 0.039], "abs": 1e-10}),

    # -- Credit ------------------------------------------------------------------------------
    Case("Credit", "CDS par spread, investment grade (h = 2%)",
         "from riskpy.credit import cds_par_spread\nresult = cds_par_spread(0.02, 0.4, 0.03, 5)",
         {"ref": "cds_ig", "rel": 1e-6}),
    Case("Credit", "CDS par spread, distressed (h = 500%)",
         "from riskpy.credit import cds_par_spread\nresult = cds_par_spread(5.0, 0.4, 0.03, 5)",
         {"ref": "cds_distressed", "rel": 1e-6}, "midpoint accrual was 11% low"),
    Case("Credit", "CDS par spread, defaulting name (h = 5000%)",
         "from riskpy.credit import cds_par_spread\nresult = cds_par_spread(50.0, 0.4, 0.0, 5)",
         {"ref": "cds_defaulting", "rel": 1e-6}, "capped at 4.80 for a true 30.0"),
    Case("Credit", "Merton spread, mid leverage",
         "from riskpy.credit import merton\nresult = merton(100, 100, 5, 0.05, 0.10).credit_spread",
         {"ref": "merton_mid", "rel": 1e-8}),
    Case("Credit", "Merton, 100y at 1000% asset vol (debt value cancels to 0)",
         "from riskpy.credit import merton\nresult = merton(100, 100, 100, 0.05, 10.0).credit_spread",
         {"ref": "merton_long", "rel": 1e-6}, "math domain error from log(0)"),
    Case("Credit", "Merton, cash-rich firm: spread ≈ 0",
         "from riskpy.credit import merton\nresult = merton(1e12, 1, 1, 0.05, 0.2).credit_spread",
         {"ref": 0.0, "abs": 1e-12}, "60bp of spurious spread"),
    Case("Credit", "Basel IRB, PD 1% LGD 45% — the textbook exposure",
         "from riskpy.credit import basel_irb_capital\nresult = basel_irb_capital(0.01, 0.45, 1.0).risk_weight",
         {"ref": "basel_1pct", "rel": 1e-12}),
    Case("Credit", "Basel IRB at PD 2.93e-6 (below the formula's pole)",
         "from riskpy.credit import basel_irb_capital\nresult = basel_irb_capital(2.93e-6, 0.45, 1.0).risk_weight",
         {"raises": "RP1"}, "returned a 1184% risk weight"),
    Case("Credit", "Rating migration: AAA cumulative PD over 10⁶ years",
         "from riskpy.credit import sp_transition_matrix\nresult = sp_transition_matrix().cumulative_pd('AAA', 10**6)",
         {"ref": 1.0, "abs": 1e-9}, "re-validation failed on 1.0000000000000002"),

    # -- Life & pensions ----------------------------------------------------------------------
    Case("Life & pensions", "Annuity-certain at a 1e-12 rate (rates near zero)",
         "from riskpy.life import annuity_certain\nresult = annuity_certain(1e-12, 30)",
         {"ref": 30 - 465e-12, "rel": 1e-14}, "8.9e-5 relative error"),
    Case("Life & pensions", "Gompertz table out to age 2000",
         "import warnings\nfrom riskpy.life import LifeTable\nwarnings.simplefilter('ignore')\n"
         "result = LifeTable.gompertz(1e-5, 1.5, max_age=2000).curtate_expectation(0)",
         {"finite": True}, "raw OverflowError from c**x"),
    Case("Life & pensions", "Table from survivors with a near-total die-off",
         "from riskpy.life import LifeTable\nresult = LifeTable.from_lx([1, 1e-20, 0]).qx(0)",
         {"ref": 1.0, "abs": 1e-15}),
    Case("Life & pensions", "SULT ä₄₀ at 5% (AMLCR Table D.3)",
         "from riskpy.life import LifeTable, whole_life_annuity_due\n"
         "result = whole_life_annuity_due(LifeTable.sult(), 40, 0.05)",
         {"ref": 18.4578, "abs": 5e-5}),

    # -- Reserving -----------------------------------------------------------------------------
    Case("Reserving", "Taylor–Ashe chain-ladder reserve (Mack 1993)",
         "from riskpy.reserving import chain_ladder, genins\nresult = float(chain_ladder(genins()).reserve.sum())",
         {"ref": 18_680_856, "abs": 1.0}),
    Case("Reserving", "Taylor–Ashe Mack total standard error",
         "from riskpy.reserving import mack_chain_ladder, genins\nresult = float(mack_chain_ladder(genins()).total_se)",
         {"ref": 2_447_095, "abs": 1.0}),
    Case("Reserving", "Bootstrap on a perfectly proportional triangle",
         "from riskpy.reserving import bootstrap_chain_ladder\nimport warnings\nwarnings.simplefilter('ignore')\n"
         "b = bootstrap_chain_ladder([[10,20,30,40],[20,40,60],[30,60],[40]], n=50, seed=1)\n"
         "result = float(b.total.mean())",
         {"ref": 20 + 60 + 120, "rel": 1e-9}, "all-NaN when φ = 0"),
    Case("Reserving", "C++ LossTriangle with an all-zero first column",
         "import riskpy\nt = riskpy.LossTriangle()\nt.add_origin_year(2020, [0., 10., 20.])\n"
         "t.add_origin_year(2021, [0., 5.])\nt.add_origin_year(2022, [7.])\nresult = t.get_development_factors()",
         {"raises": "core"}, "factor silently 1.0"),

    # -- Capital modelling ------------------------------------------------------------------------
    Case("Capital modelling", "ES₉₅ of a 1%-likely loss of 100 (Acerbi–Tasche = 20)",
         "from riskpy.mc import simulate, Bernoulli, Constant\n"
         "r = simulate(lambda ev, sev: ev * sev, 200_000, seed=1, ev=Bernoulli(0.01), sev=Constant(100.0))\n"
         "result = r.tvar(0.95)",
         {"ref": 20.0, "rel": 0.05}, "averaged the tied zeros: 1.0"),
    Case("Capital modelling", "Health claims, NB dispersion 0.3: mean count",
         "import numpy as np, riskpy\nx = riskpy.MonteCarloSimulator(100_000, 3).simulate_health_claims(10, 0.3, 1e6, 1e-6)\n"
         "result = float(np.mean(x))",
         {"ref": 10.0, "rel": 0.02}, "r = 1/d truncated to an int: 8.97"),
    Case("Capital modelling", "Health claims, NB dispersion 2: mean count",
         "import numpy as np, riskpy\nx = riskpy.MonteCarloSimulator(100_000, 3).simulate_health_claims(10, 2.0, 1e6, 1e-6)\n"
         "result = float(np.mean(x))",
         {"ref": 10.0, "rel": 0.05}, "all zeros"),
    Case("Capital modelling", "FFT compound Poisson λ=50, default grid: mean",
         "import riskpy\npmf = riskpy.FourierTransform.compound_poisson_pmf([0.0, 1.0], 50.0)\n"
         "result = sum(k * p for k, p in enumerate(pmf))",
         {"ref": 50.0, "rel": 1e-6}, "wrapped around: 7.45"),
    Case("Capital modelling", "Loss elimination ratio, CV 2, deductible 2× mean",
         "import riskpy\nresult = riskpy.ExposureRating().loss_elimination_ratio(2000, 1000, 2.0)",
         {"ref": "ler_cv2", "rel": 1e-8}, "not a lognormal LER: 0.793"),
    Case("Capital modelling", "Loss elimination ratio, CV 0.5, deductible = mean",
         "import riskpy\nresult = riskpy.ExposureRating().loss_elimination_ratio(1000, 1000, 0.5)",
         {"ref": "ler_cv05", "rel": 1e-8}),
    Case("Capital modelling", "Credibility with a NaN k (bad config)",
         "import riskpy\nresult = riskpy.ExperienceRating(float('nan')).calculate_credibility(5)",
         {"raises": "core"}, "returned full credibility 1.0"),

    # -- Statistics ---------------------------------------------------------------------------------
    Case("Statistics", "χ² CDF at 10⁶ degrees of freedom",
         "from riskpy import _special as sp\nresult = sp.chi2_cdf(1e6, 1e6)", {"ref": "chi2_cdf_big", "rel": 1e-9},
         "iteration cap: 0.42 for 0.50"),
    Case("Statistics", "Student-t 1e-20 quantile, df=5",
         "from riskpy import _special as sp\nresult = sp.t_ppf(1e-20, 5)", {"ref": "t_ppf_tail", "rel": 1e-9},
         "-4096 for -15684"),
    Case("Statistics", "Incomplete gamma P(1e5, 1e5)",
         "from riskpy import _special as sp\nresult = sp.gammainc(1e5, 1e5)", {"ref": "gammainc_big", "rel": 1e-9}),
    Case("Statistics", "Normal quantile at p = 1e-320",
         "from riskpy import _special as sp\nresult = sp.norm_ppf(1e-320)", {"ref": "norm_ppf_sub", "rel": 1e-12}),

    # -- AI agent tools ---------------------------------------------------------------------------------
    Case("AI agent tools", "aggregate_loss with trials sent as 1000.0 (JSON numbers)",
         "from riskpy import tools\nresult = tools.call('aggregate_loss', {'expected_frequency': 2, 'severity_mean': 1000,"
         " 'severity_sd': 500, 'trials': 1000.0, 'seed': 1})['mean']",
         {"finite": True}, "pybind refused 1000.0"),
    Case("AI agent tools", "bootstrap_yield_curve with frequency 100000",
         "from riskpy import tools\nresult = tools.call('bootstrap_yield_curve', {'par_rates': [0.03, 0.035, 0.04],"
         " 'maturities': [1, 2, 3], 'frequency': 100000})",
         {"raises": "RP405"}, "ran for minutes"),
    Case("AI agent tools", "convert_rate with 10¹⁶ periods a year",
         "from riskpy import tools\nresult = tools.call('convert_rate', {'rate': 0.05, 'from_compounding':"
         " '10000000000000000', 'to_compounding': 'annual'})",
         {"raises": "RP405"}, "returned 0.0"),
    Case("AI agent tools", "bond_analytics on a 1e308 face value",
         "from riskpy import tools\nresult = tools.call('bond_analytics', {'face': 1e308, 'coupon': 0.05, 'maturity': 10,"
         " 'ytm': 0.05})",
         {"raises": "RP304"}, "'ok' with null durations"),
]


# Everyday inputs: nothing here is an edge case, so 0.5.2 and the development build
# must agree. Each line is one number; the notebook compares them pairwise.
EVERYDAY = """
import random, warnings
warnings.simplefilter('ignore')
from riskpy import quant, rates, credit, life, reserving
rng = random.Random(2026)
out = {k: [] for k in ('black_scholes + greeks', 'implied_vol error (iv - true sigma)', 'discount_factor', 'basel_irb_capital',
                       'merton pd', 'cds_par_spread', 'life annuities & reserves', 'chain ladder & Mack')}
for _ in range(60):
    S, K, T = 100.0, rng.uniform(70, 130), rng.uniform(0.1, 3)
    r, sig = rng.uniform(0.0, 0.06), rng.uniform(0.1, 0.5)
    kind = rng.choice(['call', 'put'])
    p = quant.black_scholes(S, K, T, r, sig, kind=kind)
    g = quant.greeks(S, K, T, r, sig, kind=kind)
    out['black_scholes + greeks'] += [p, g.delta, g.gamma, g.vega, g.theta, g.rho]
    out['implied_vol error (iv - true sigma)'].append(quant.implied_vol(p, S, K, T, r, kind=kind) - sig)
for _ in range(30):
    out['discount_factor'].append(rates.discount_factor(rng.uniform(0, 0.08), rng.uniform(0.5, 30),
                                                        rng.choice(['annual', 2, 12, 'continuous'])))
    out['basel_irb_capital'].append(credit.basel_irb_capital(rng.uniform(0.0005, 0.2), rng.uniform(0.1, 0.6),
                                                             rng.uniform(1, 5)).risk_weight)
    out['merton pd'].append(credit.merton(rng.uniform(80, 200), 100, rng.uniform(0.5, 5), 0.03,
                                          rng.uniform(0.1, 0.4)).pd)
    out['cds_par_spread'].append(credit.cds_par_spread(rng.uniform(0.001, 0.05), 0.4, 0.03, 5))
sult = life.LifeTable.sult()
for x in range(20, 90, 5):
    out['life annuities & reserves'] += [life.whole_life_annuity_due(sult, x, 0.05),
                                         life.whole_life_insurance(sult, x, 0.04),
                                         life.net_premium_reserve(sult, x, 10, 0.05),
                                         life.annuity_certain(0.04, x - 15)]
cl = reserving.chain_ladder(reserving.genins())
out['chain ladder & Mack'] += [float(v) for v in cl.factors] + [float(v) for v in cl.reserve]
out['chain ladder & Mack'] += [float(v) for v in reserving.mack_chain_ladder(reserving.genins()).se]
result = out
"""

# The same calls timed, so a fix that is slower than it should be shows up.
TIMING = """
import time, warnings
warnings.simplefilter('ignore')
from riskpy import quant, credit, life, reserving, _special as sp
from riskpy.mc import simulate, Poisson, LogNormal
def t(fn, n):
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - start) / n * 1e6
sult = life.LifeTable.sult()
tri = reserving.genins()
result = {
    'black_scholes': t(lambda: quant.black_scholes(100, 105, 1, 0.03, 0.2), 20000),
    'greeks': t(lambda: quant.greeks(100, 105, 1, 0.03, 0.2), 20000),
    'implied_vol': t(lambda: quant.implied_vol(8.0, 100, 105, 1, 0.03), 2000),
    'cds_par_spread': t(lambda: credit.cds_par_spread(0.02, 0.4, 0.03, 5), 2000),
    'merton': t(lambda: credit.merton(120, 100, 2, 0.03, 0.25), 20000),
    'basel_irb_capital': t(lambda: credit.basel_irb_capital(0.01, 0.45, 2.5), 20000),
    'whole_life_annuity_due': t(lambda: life.whole_life_annuity_due(sult, 40, 0.05), 2000),
    'gammainc(50, 49)': t(lambda: sp.gammainc(50.0, 49.0), 20000),
    't_ppf(0.01, 5)': t(lambda: sp.t_ppf(0.01, 5), 2000),
    'chi2_ppf(0.95, 10)': t(lambda: sp.chi2_ppf(0.95, 10), 2000),
    'mack_chain_ladder': t(lambda: reserving.mack_chain_ladder(tri), 200),
    'simulate 100k (compound Poisson)': t(lambda: simulate(lambda n, s: n * s, 100_000, seed=1,
                                         n=Poisson(2.0), s=LogNormal.from_moments(1000, 500)), 5),
}
"""


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


class _Timeout(Exception):
    pass


def _alarm(signum, frame):  # pragma: no cover - signal handler
    raise _Timeout()


def run_one(code: str, timeout: float = 8.0) -> Dict[str, Any]:
    """Execute a case snippet here and describe the outcome as plain JSON-able data."""
    ns: Dict[str, Any] = {}
    use_alarm = hasattr(signal, "SIGALRM")
    if use_alarm:
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        exec(compile(code, "<case>", "exec"), ns)
        return {"value": _plain(ns.get("result"))}
    except _Timeout:
        return {"timeout": timeout}
    except BaseException as exc:  # noqa: BLE001 - a case reports whatever it hits
        return {"error": type(exc).__name__, "code": getattr(exc, "code", None), "message": str(exc)[:160]}
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)


def _plain(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else repr(f)
    return repr(value)


def run_in(pythonpath: Optional[str], cases: List[Case] = CASES, python: str = sys.executable,
           timeout: float = 8.0) -> List[Dict[str, Any]]:
    """Run every case in a fresh interpreter with ``pythonpath`` first on sys.path.

    One process per case, so a crash, a hang the alarm cannot interrupt (inside the C++
    core) or a leaked warning filter in one case cannot touch the next.
    """
    env = dict(os.environ)
    env.pop("RISKPY_TRACE_FILE", None)
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    else:
        env.pop("PYTHONPATH", None)
    here = os.path.dirname(os.path.abspath(__file__))
    outcomes = []
    for case in cases:
        driver = textwrap.dedent(f"""
            import json, sys
            sys.path.insert(0, {here!r})
            import cases, riskpy
            out = cases.run_one({case.code!r}, timeout={timeout!r})
            out["version"] = riskpy.__version__
            print("@@" + json.dumps(out))
        """)
        try:
            proc = subprocess.run([python, "-c", driver], capture_output=True, text=True, env=env,
                                  timeout=timeout + 20, cwd=os.path.expanduser("~"))
            line = next((l for l in proc.stdout.splitlines() if l.startswith("@@")), None)
            if line is None:
                outcomes.append({"error": "crash", "code": None, "message": (proc.stderr or "")[-160:]})
            else:
                outcomes.append(json.loads(line[2:]))
        except subprocess.TimeoutExpired:
            outcomes.append({"timeout": timeout + 20})
    return outcomes


def value_in(pythonpath: Optional[str], code: str, timeout: float = 120.0) -> Any:
    """Run one snippet in a fresh interpreter (``None`` = the installed build) and return ``result``."""
    out = run_in(pythonpath, [Case("", "", code)], timeout=timeout)[0]
    if "value" not in out:
        raise RuntimeError(f"snippet failed under {pythonpath or 'the installed build'}: {out}")
    return out["value"]


def judge(case: Case, outcome: Dict[str, Any], refs: Dict[str, float]) -> str:
    """'pass' or a short reason it does not."""
    exp = case.expect
    if "timeout" in outcome:
        return f"timed out after {outcome['timeout']:.0f}s"
    if "raises" in exp:
        want = exp["raises"]
        if "error" not in outcome:
            return f"no error (returned {_short(outcome.get('value'))})"
        if want == "core":   # the C++ core raises plain ValueError/RuntimeError
            return "pass" if outcome["error"] in ("ValueError", "RuntimeError", "InputError") else f"{outcome['error']}"
        code = outcome.get("code") or ""
        return "pass" if str(code).startswith(want) else f"{outcome['error']} {code or ''}".strip()
    if "error" in outcome:
        return f"{outcome['error']}: {outcome['message'][:60]}"
    value = outcome.get("value")
    if exp.get("true"):
        return "pass" if value is True else f"got {value!r}"
    if exp.get("finite"):
        nums = _numbers(value)
        return "pass" if nums and all(isinstance(v, float) or isinstance(v, int) for v in nums) else f"got {_short(value)}"
    ref = refs[exp["ref"]] if isinstance(exp["ref"], str) else exp["ref"]
    rel, ab = exp.get("rel", 0.0), exp.get("abs", 0.0)
    pairs = list(zip(value, ref)) if isinstance(ref, list) else [(value, ref)]
    if isinstance(ref, list) and (not isinstance(value, list) or len(value) != len(ref)):
        return f"got {_short(value)}"
    for got, want in pairs:
        if not isinstance(got, (int, float)):
            return f"got {_short(got)}"
        if abs(got - want) > max(ab, rel * abs(want)):
            err = abs(got - want) / abs(want) if want else abs(got - want)
            return f"off by {err:.1e}{' rel' if want else ''}"
    return "pass"


def _numbers(value):
    if isinstance(value, (list, tuple)):
        return [n for v in value for n in _numbers(v)]
    if isinstance(value, dict):
        return [n for v in value.values() for n in _numbers(v)]
    return [value] if isinstance(value, (int, float)) and not isinstance(value, bool) else []


def _short(value) -> str:
    text = repr(value)
    return text if len(text) <= 40 else text[:37] + "..."


def baseline_path(version: str = "0.5.2") -> str:
    """A ``--target`` install of a published wheel (no dependencies), made on first use."""
    path = os.environ.get("RISKPY_BASELINE") or os.path.expanduser(f"~/.cache/riskpy-baseline/{version}")
    if not os.path.isdir(os.path.join(path, "riskpy")):
        cmd = ["uv", "pip", "install", "--python", sys.executable, "--no-deps", "--target", path,
               f"open-riskpy=={version}"]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except (OSError, subprocess.CalledProcessError):
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", path,
                            f"open-riskpy=={version}"], check=True, capture_output=True)
    return path


if __name__ == "__main__":
    refs = references()
    for case, out in zip(CASES, run_in(None)):
        print(f"{judge(case, out, refs):<28} {case.section:<18} {case.label}")
