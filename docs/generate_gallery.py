"""Render the chart gallery and the verification report for the docs site.

Run from the repo root:

    python docs/generate_gallery.py

Writes PNGs and ``verification.md`` into ``docs/assets/``. The docs workflow
runs this before building the site, which means the gallery and the report
can never drift from the code — and a broken module fails the docs build
rather than shipping a stale picture of a version that no longer exists.

Every figure here is also a worked example. If you want to know how to produce
one of the charts on the site, this file is the answer.
"""

from __future__ import annotations

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")  # no display in CI

import matplotlib.pyplot as plt  # noqa: E402

from riskpy import credit, life, quant, rates, reserving, verify, viz  # noqa: E402
from riskpy.mc import (  # noqa: E402
    Constant,
    LogNormal,
    Model,
    NegativeBinomial,
    Normal,
    PERT,
    Poisson,
    simulate,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
os.makedirs(ASSETS, exist_ok=True)

SEED = 20260912
started = time.perf_counter()


def out(name: str) -> str:
    return os.path.join(ASSETS, name)


def emit(name: str, fig) -> None:
    viz.save(fig, out(name + ".png"))
    plt.close(fig)
    print(f"  {name}.png")


# ---------------------------------------------------------------------------
# 1. The headline example: an aggregate loss model with a correlated input.
# ---------------------------------------------------------------------------

model = Model(
    claim_count=Poisson(mean=140),
    severity=LogNormal.from_moments(mean=18_000, sd=42_000),
    inflation=Normal(mean=0.043, sd=0.012),
    reinsurance_recovery=PERT(low=0.0, mode=0.15, high=0.45),
)
model.correlate("claim_count", "inflation", 0.35)


@model.formula
def annual_loss(claim_count, severity, inflation, reinsurance_recovery):
    gross = claim_count * severity * (1.0 + inflation)
    return gross * (1.0 - reinsurance_recovery)


result = model.run(200_000, seed=SEED, sampling="lhs")
print(result.summary())
print()

emit("distribution", viz.distribution(result))
emit("density", viz.density(result))
emit("cdf", viz.cdf(result))
emit("exceedance", viz.exceedance(result))
emit("convergence", viz.convergence(result))
emit("tornado", viz.tornado(result))
emit("qq", viz.qq(result, dist=LogNormal.from_moments(result.mean, result.std),
                  title="Q–Q against a log-normal with the same moments"))
emit("dashboard", viz.dashboard(result))
emit("correlation", viz.correlation(result))
emit("scatter", viz.scatter(result, "claim_count", "inflation",
                            title="Correlated inputs — claim count against inflation"))

# ---------------------------------------------------------------------------
# 2. Scenario comparison — same model, three reinsurance structures.
# ---------------------------------------------------------------------------

scenarios = model.sweep(
    "reinsurance_recovery",
    {
        "no cover": Constant(0.0),
        "quota share 15%": Constant(0.15),
        "negotiated": PERT(low=0.05, mode=0.25, high=0.5),
    },
    trials=100_000,
    seed=SEED,
)
emit("compare", viz.compare(scenarios, title="Annual loss by reinsurance structure"))
emit("spread", viz.spread(scenarios, title="Annual loss by reinsurance structure"))

# ---------------------------------------------------------------------------
# 3. Over-dispersed frequency — Poisson is usually too tidy for real claims.
# ---------------------------------------------------------------------------

overdispersed = simulate(
    lambda claim_count, severity: claim_count * severity,
    trials=100_000, seed=SEED, label="annual loss",
    claim_count=NegativeBinomial.from_mean_dispersion(mean=140, dispersion=2.4),
    severity=LogNormal.from_moments(mean=18_000, sd=42_000),
)
emit("overdispersion", viz.compare({"Poisson frequency": result, "Negative binomial": overdispersed},
                                   title="What over-dispersion does to the tail"))

# ---------------------------------------------------------------------------
# 4. Paths — GBM and Merton jump-diffusion, same parameters otherwise.
# ---------------------------------------------------------------------------

paths = quant.gbm_paths(S0=100.0, mu=0.07, sigma=0.22, T=1.0, steps=252, trials=20_000, seed=SEED)
emit("gbm_fan", viz.fan(paths, title="GBM — 20,000 paths, one year", ylabel="price"))

jumps = quant.merton_jump_paths(S0=100.0, mu=0.07, sigma=0.18, T=1.0, jump_intensity=1.2,
                                jump_mean=-0.05, jump_sd=0.12, steps=252, trials=20_000, seed=SEED)
emit("jump_fan", viz.fan(jumps, title="Merton jump-diffusion — same drift, fatter tails", ylabel="price"))

# ---------------------------------------------------------------------------
# 5. Reserving — the GenIns triangle.
# ---------------------------------------------------------------------------

tri = reserving.genins()
cl = reserving.chain_ladder(tri)
mack = reserving.mack_chain_ladder(tri)
print(mack.summary())
print()
emit("triangle", viz.triangle(tri))
emit("development", viz.development(cl, triangle=tri))
emit("reserve_range", viz.reserve_range(mack))

# ---------------------------------------------------------------------------
# 6. Life — the standard table and a 20-year endowment.
# ---------------------------------------------------------------------------

sult = life.LifeTable.sult()
emit("survival", viz.survival(sult, title="Survivorship — Standard Ultimate Life Table"))
emit("mortality", viz.mortality(sult, title="Mortality — Standard Ultimate Life Table (log scale)"))
emit("reserve_profile", viz.reserve_profile(life.reserve_profile(sult, 40, 0.05, "endowment", n=20),
                                            title="Net premium reserve — 20-year endowment issued at 40"))

# ---------------------------------------------------------------------------
# 7. Rates and capital.
# ---------------------------------------------------------------------------

emit("curve", viz.curve(rates.YieldCurve.nelson_siegel(0.042, -0.018, 0.028, 2.0),
                        title="Nelson–Siegel curve — zero and one-year forward"))

portfolio = credit.credit_portfolio_loss(
    pds=[0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
    lgds=[0.4, 0.45, 0.45, 0.5, 0.55, 0.6],
    eads=[5e6, 4e6, 3e6, 2e6, 1e6, 5e5],
    rho=0.2, trials=100_000, seed=SEED,
)
emit("allocation", viz.allocation({"Motor": 41_200_000, "Property": 28_900_000,
                                   "Liability": 63_400_000, "Marine": 11_050_000}))
emit("waterfall", viz.waterfall({"gross loss": 53_000_000, "reinsurance": -12_400_000,
                                 "expenses": 8_900_000, "diversification": -6_300_000},
                                title="From gross loss to net capital"))

# ---------------------------------------------------------------------------
# 8. The light theme, once, so the page can show it.
# ---------------------------------------------------------------------------

viz.theme("light")
emit("distribution_light", viz.distribution(result))
viz.theme("dark")

# ---------------------------------------------------------------------------
# 9. The verification report the site publishes.
# ---------------------------------------------------------------------------

report = verify.run()
with open(out("verification.md"), "w", encoding="utf-8") as handle:
    handle.write(report.to_markdown() + "\n")
print(report.summary())
print(f"\nWrote gallery and verification report to {ASSETS} in {time.perf_counter() - started:.1f}s")
if not report:
    sys.exit("verification failed — refusing to publish a site that says otherwise")
