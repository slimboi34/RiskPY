"""Measure the 0.7.0 capital, stress and audit features against closed forms, and time them.

    python benchmarks/capital/run.py --markdown benchmarks/capital/RESULTS.md

Run it from outside the checkout (or with the package installed) so the installed RiskPY is
what is measured. Needs NumPy and SciPy. Everything is seeded; the timings are the median of
several repeats on whatever machine runs it, so they move from machine to machine while the
accuracy figures do not.

Four sections:

1. **Copula samplers** — for each family at Kendall's tau 0.5, one million draws: the sample
   Kendall's tau (SciPy) against the closed form, and the observed share of draws with both
   coordinates above 0.99 against ``joint_exceedance(0.99)``, in binomial standard errors,
   for seed 1 and pooled over seeds 1 to 10.
2. **Euler allocation** — two jointly normal lines, where the TVaR contributions have a
   closed form, μᵢ + Cov(Xᵢ, S)/σ_S · φ(z)/(1 − α): the worst absolute error of a
   contribution over 10 seeds, at four trial counts.
3. **Shared draws in a stress test** — the change in the mean loss when inflation moves by
   one point, estimated from two independent runs and from ``Model.stress``, over 20 seeds:
   the standard deviation of the estimate each way.
4. **Timings** — sampling, allocation, the meta-tools, and what recording a call costs.
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import tempfile
import time


def _median_ms(fn, repeats: int = 5) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--markdown", metavar="FILE", help="write the results as Markdown")
    args = parser.parse_args()

    import numpy as np
    from scipy import stats

    import riskpy
    from riskpy import audit, tools
    from riskpy.capital import (ClaytonCopula, FrankCopula, GaussianCopula, GumbelCopula, StudentTCopula, allocate)
    from riskpy.mc import Compound, LogNormal, Model, Normal, Poisson

    out = [f"# Capital, stress and audit benchmark — RiskPY {riskpy.__version__}", "",
           "Produced by `benchmarks/capital/run.py` (its docstring says what each section measures). "
           f"NumPy {np.__version__}, SciPy {__import__('scipy').__version__}, Python "
           f"{__import__('platform').python_version()}.", ""]

    # 1. samplers ------------------------------------------------------------------------
    n, q = 1_000_000, 0.99
    out += ["## Copula samplers against their closed forms", "",
            f"{n:,} draws each, seed 1, every family at Kendall's tau = 0.5. The Kendall's tau column uses a "
            "200,000-draw subsample (SciPy's estimator). `z` is the gap between observed and exact joint "
            "exceedance in binomial standard errors; `z, 10 seeds` pools seeds 1 to 10 (ten million draws).", "",
            "| copula | tau (exact) | tau (sample) | P(both > 0.99) exact | observed | z | z, 10 seeds | "
            "ms per million draws |",
            "|---|---|---|---|---|---|---|---|"]
    families = [
        ("Gaussian", GaussianCopula.from_kendall_tau(0.5)),
        ("Student t, df = 4", StudentTCopula.from_kendall_tau(0.5, df=4)),
        ("Clayton (survival)", ClaytonCopula.from_kendall_tau(0.5, survival=True)),
        ("Gumbel", GumbelCopula.from_kendall_tau(0.5)),
        ("Frank", FrankCopula.from_kendall_tau(0.5)),
    ]
    for label, cop in families:
        u = cop.sample(n, seed=1)
        tau = stats.kendalltau(u[:200_000, 0], u[:200_000, 1]).statistic
        exact = cop.joint_exceedance(q)
        observed = float(np.mean((u[:, 0] > q) & (u[:, 1] > q)))
        z = (observed - exact) / math.sqrt(exact * (1.0 - exact) / n)
        hits = int(np.sum((u[:, 0] > q) & (u[:, 1] > q)))
        for seed in range(2, 11):
            more = cop.sample(n, seed=seed)
            hits += int(np.sum((more[:, 0] > q) & (more[:, 1] > q)))
        pooled = (hits / (10.0 * n) - exact) / math.sqrt(exact * (1.0 - exact) / (10.0 * n))
        ms = _median_ms(lambda cop=cop: cop.sample(n, seed=2), repeats=3)
        out.append(f"| {label} | {cop.kendall_tau():.4f} | {tau:.4f} | {exact:.6f} | {observed:.6f} | {z:+.2f} | "
                   f"{pooled:+.2f} | {ms:.0f} |")
    out.append("")

    # 2. Euler allocation ----------------------------------------------------------------
    rho, level = 0.5, 0.99
    sd_s = math.sqrt(4.0 + 1.0 + 2.0 * rho * 2.0)
    ratio = stats.norm.pdf(stats.norm.ppf(level)) / (1.0 - level)
    exact = {"x": 10.0 + (4.0 + rho * 2.0) / sd_s * ratio, "y": 5.0 + (1.0 + rho * 2.0) / sd_s * ratio}
    out += ["## Euler TVaR contributions against the normal closed form", "",
            f"X ~ N(10, 2²), Y ~ N(5, 1²), correlation {rho}, level {level}. Exact contributions: "
            f"x = {exact['x']:.4f}, y = {exact['y']:.4f} (total {exact['x'] + exact['y']:.4f}). "
            "Worst absolute error of either contribution over 10 seeds, and the largest gap between the sum "
            "of the contributions and the total.", "",
            "| trials | worst error | as % of the smaller contribution | max ∣Σ contributions − total∣ | ms per allocation |",
            "|---|---|---|---|---|"]
    for trials in (10_000, 100_000, 1_000_000, 4_000_000):
        worst = gap = 0.0
        sample = None
        for seed in range(1, 11):
            u = GaussianCopula(rho).sample(trials, seed=seed)
            losses = {"x": 10.0 + 2.0 * stats.norm.ppf(u[:, 0]), "y": 5.0 + 1.0 * stats.norm.ppf(u[:, 1])}
            alloc = allocate(losses, level=level)
            sample = losses
            for name in ("x", "y"):
                worst = max(worst, abs(alloc.contributions[name] - exact[name]))
            gap = max(gap, abs(sum(alloc.contributions.values()) - alloc.total))
        ms = _median_ms(lambda: allocate(sample, level=level), repeats=3)
        out.append(f"| {trials:,} | {worst:.4f} | {100.0 * worst / exact['y']:.2f}% | {gap:.1e} | {ms:.1f} |")
    out.append("")

    # 3. shared draws --------------------------------------------------------------------
    def book(inflation_mean):
        model = Model(claims=Compound(Poisson(mean=120), LogNormal(mu=8.5, sigma=1.2)),
                      inflation=Normal(mean=inflation_mean, sd=0.01))
        model.correlate("claims", "inflation", 0.3)
        return model

    loss = lambda claims, inflation: claims * (1.0 + inflation)  # noqa: E731
    separate, shared = [], []
    for seed in range(20):
        base = book(0.04).run(100_000, seed=100 + seed, fn=loss).mean
        moved = book(0.05).run(100_000, seed=500 + seed, fn=loss).mean
        separate.append(moved - base)
        stress = book(0.04).stress({"+1pt": {"inflation": lambda x: x + 0.01}}, 100_000, seed=100 + seed, fn=loss)
        shared.append(stress.delta("+1pt"))
    out += ["## Shared draws in a stress test", "",
            "Compound Poisson(120) × LogNormal(8.5, 1.2) claims, inflation N(4%, 1%), rank correlation 0.3; "
            "loss = claims × (1 + inflation). The effect on the mean loss of inflation one point higher, "
            "100,000 trials, 20 seeds.", "",
            "| how the difference is estimated | mean of the estimate | standard deviation across seeds |",
            "|---|---|---|",
            f"| two independent runs | {statistics.mean(separate):,.0f} | {statistics.stdev(separate):,.1f} |",
            f"| `Model.stress` (shared draws) | {statistics.mean(shared):,.0f} | {statistics.stdev(shared):,.1f} |",
            "", f"Noise ratio: {statistics.stdev(separate) / statistics.stdev(shared):.0f}×.", ""]

    # 4. timings -------------------------------------------------------------------------
    bs = {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2}
    plain = _median_ms(lambda: [tools.call("black_scholes", bs) for _ in range(2000)]) / 2000.0
    with audit.record():
        in_memory = _median_ms(lambda: [tools.call("black_scholes", bs) for _ in range(2000)]) / 2000.0
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ledger.jsonl")
        with audit.record(path):
            on_disk = _median_ms(lambda: [tools.call("black_scholes", bs) for _ in range(2000)]) / 2000.0
        entries = len(audit.load(path))
        verify_ms = _median_ms(lambda: audit.verify_chain(path), repeats=3)
        replay_ms = _median_ms(lambda: audit.replay(path), repeats=3)
    sens = _median_ms(lambda: tools.call("sensitivity", {"tool": "black_scholes", "arguments": bs, "parameter": "S"}),
                      repeats=50)
    table = _median_ms(lambda: tools.call("scenario_table", {
        "tool": "black_scholes", "arguments": bs,
        "scenarios": {f"s{k}": {"sigma": 0.1 + 0.01 * k} for k in range(20)}}), repeats=20)
    alloc_tool = _median_ms(lambda: tools.call("capital_allocation", {
        "means": [100, 60, 40], "sds": [20, 30, 25], "copula": "gumbel", "copula_theta": 1.5, "level": 0.995,
        "trials": 200_000, "seed": 1}), repeats=5)
    out += ["## Timings", "",
            "Median of repeated runs on the machine that produced this file.", "",
            "| operation | time |", "|---|---|",
            f"| `tools.call(\"black_scholes\")`, no ledger | {plain * 1000.0:.1f} µs |",
            f"| the same, recorded to an in-memory ledger | {in_memory * 1000.0:.1f} µs |",
            f"| the same, recorded to a ledger file | {on_disk * 1000.0:.1f} µs |",
            f"| `audit.verify_chain` on {entries:,} entries | {verify_ms:.0f} ms |",
            f"| `audit.replay` on {entries:,} entries | {replay_ms:.0f} ms |",
            f"| `sensitivity` of `black_scholes` (three valuations) | {sens * 1000.0:.0f} µs |",
            f"| `scenario_table` of `black_scholes`, 20 scenarios | {table * 1000.0:.0f} µs |",
            f"| `capital_allocation`, 3 lognormal lines, Gumbel copula, 200,000 trials | {alloc_tool:.0f} ms |",
            ""]

    text = "\n".join(out)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.markdown}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
