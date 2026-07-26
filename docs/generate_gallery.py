"""Render the chart gallery used by the documentation site.

Run from the repo root:

    python docs/generate_gallery.py

Writes PNGs into ``docs/assets/``. The docs workflow runs this before building
the site, which means the gallery can never drift from the code — and, usefully,
that a broken ``riskpy.mc`` or ``riskpy.viz`` fails the docs build rather than
shipping a stale picture of a version that no longer exists.

Every figure here is also a worked example. If you want to know how to produce
one of the charts on the site, this file is the answer.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # no display in CI

from riskpy import quant, viz
from riskpy.mc import (
    Constant,
    LogNormal,
    Model,
    Normal,
    NegativeBinomial,
    PERT,
    Poisson,
    simulate,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
os.makedirs(ASSETS, exist_ok=True)

SEED = 20260726


def out(name: str) -> str:
    return os.path.join(ASSETS, name)


# ---------------------------------------------------------------------------
# 1. The headline example: an aggregate loss model in six lines.
# ---------------------------------------------------------------------------

model = Model(
    claim_count=Poisson(mean=140),
    severity=LogNormal.from_moments(mean=18_000, sd=42_000),
    inflation=Normal(mean=0.043, sd=0.012),
    reinsurance_recovery=PERT(low=0.0, mode=0.15, high=0.45),
)


@model.formula
def annual_loss(claim_count, severity, inflation, reinsurance_recovery):
    gross = claim_count * severity * (1.0 + inflation)
    return gross * (1.0 - reinsurance_recovery)


result = model.run(200_000, seed=SEED)
print(result.summary())

viz.save(viz.distribution(result), out("distribution.png"))
viz.save(viz.exceedance(result), out("exceedance.png"))
viz.save(viz.convergence(result), out("convergence.png"))
viz.save(viz.tornado(result), out("tornado.png"))

# ---------------------------------------------------------------------------
# 2. Scenario comparison — same model, three reinsurance structures.
# ---------------------------------------------------------------------------

scenarios = {}
for name, recovery in (
    ("no cover", Constant(0.0)),
    ("quota share 15%", Constant(0.15)),
    ("negotiated", PERT(low=0.05, mode=0.25, high=0.5)),
):
    scenarios[name] = simulate(
        annual_loss,
        trials=100_000,
        seed=SEED,
        label="annual loss",
        claim_count=Poisson(mean=140),
        severity=LogNormal.from_moments(mean=18_000, sd=42_000),
        inflation=Normal(mean=0.043, sd=0.012),
        reinsurance_recovery=recovery,
    )

viz.save(viz.compare(scenarios, title="Annual loss by reinsurance structure"), out("compare.png"))

# ---------------------------------------------------------------------------
# 3. Over-dispersed frequency — Poisson is usually too tidy for real claims.
# ---------------------------------------------------------------------------

overdispersed = simulate(
    lambda claim_count, severity: claim_count * severity,
    trials=100_000,
    seed=SEED,
    label="annual loss",
    claim_count=NegativeBinomial.from_mean_dispersion(mean=140, dispersion=2.4),
    severity=LogNormal.from_moments(mean=18_000, sd=42_000),
)
viz.save(
    viz.compare({"Poisson frequency": result, "Negative binomial": overdispersed},
                title="What over-dispersion does to the tail"),
    out("overdispersion.png"),
)

# ---------------------------------------------------------------------------
# 4. Paths — GBM and Merton jump-diffusion, same parameters otherwise.
# ---------------------------------------------------------------------------

paths = quant.gbm_paths(S0=100.0, mu=0.07, sigma=0.22, T=1.0, steps=252,
                        trials=20_000, seed=SEED)
viz.save(viz.fan(paths, title="GBM — 20,000 paths, one year", ylabel="price"),
         out("gbm_fan.png"))

jumps = quant.merton_jump_paths(S0=100.0, mu=0.07, sigma=0.18, T=1.0,
                                jump_intensity=1.2, jump_mean=-0.05, jump_sd=0.12,
                                steps=252, trials=20_000, seed=SEED)
viz.save(viz.fan(jumps, title="Merton jump-diffusion — same drift, fatter tails",
                 ylabel="price"),
         out("jump_fan.png"))

# ---------------------------------------------------------------------------
# 5. A sanity check worth printing: Heston should converge to Black-Scholes
#    when vol-of-vol goes to zero and variance starts at its long-run level.
# ---------------------------------------------------------------------------

bs = quant.black_scholes(S=100, K=100, T=1.0, r=0.03, sigma=0.2)
heston_flat = quant.heston_price(
    S=100, K=100, T=1.0, r=0.03,
    v0=0.04, kappa=2.0, theta=0.04, xi=1e-4, rho=0.0,
)
print(f"\nBlack-Scholes      {bs:.6f}")
print(f"Heston (xi -> 0)   {heston_flat:.6f}")
print(f"difference         {abs(bs - heston_flat):.2e}")

print(f"\nWrote gallery to {ASSETS}")
