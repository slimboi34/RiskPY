"""Generic Monte Carlo — any formula, any distributions.

The C++ ``MonteCarloSimulator`` covers five specific actuarial scenarios very
fast. This module covers the other case: you have a formula that is too awkward
to solve analytically, so you feed it random inputs a hundred thousand times and
look at the distribution that falls out.

    from riskpy.mc import Model, LogNormal, Normal, Poisson

    model = Model(
        claims   = Poisson(mean=120),
        severity = LogNormal(mu=8.5, sigma=1.2),
        inflation= Normal(mean=0.04, sd=0.01),
    )

    @model.formula
    def annual_loss(claims, severity, inflation):
        return claims * severity * (1 + inflation)

    result = model.run(100_000, seed=42)
    print(result.summary())
    result.plot()

Everything is vectorised: your formula is called **once** with arrays, not once
per trial, so a hundred thousand trials is one NumPy expression. If your formula
cannot be written that way, pass ``vectorised=False`` and it will be looped
instead — correct, just slower.

NumPy is required here and is an optional extra for the package as a whole
(``pip install open-riskpy[sim]``). The C++ core keeps its zero-dependency
install; this module is opt-in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = [
    "Model",
    "Result",
    "simulate",
    "Distribution",
    "Constant",
    "Uniform",
    "Normal",
    "LogNormal",
    "Triangular",
    "PERT",
    "Exponential",
    "Gamma",
    "Beta",
    "Pareto",
    "Poisson",
    "NegativeBinomial",
    "Bernoulli",
    "Binomial",
    "Empirical",
]


def _numpy():
    """Import NumPy on use, with an error that says how to fix it."""
    try:
        import numpy  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "riskpy.mc needs NumPy, which is not installed. The C++ core is "
            "deliberately dependency-free, so the simulation layer ships as an "
            "extra:\n"
            "    pip install open-riskpy[sim]\n"
            "or just:\n"
            "    pip install numpy"
        ) from exc
    return numpy


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


class Distribution:
    """Base class. A distribution only has to know how to draw ``n`` samples.

    Parameters are validated in ``__init__`` rather than at sample time. A
    negative standard deviation is a typo, and finding out about it after a
    ten-second simulation has already run is worse than useless.
    """

    def sample(self, n: int, rng: Any):  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in sorted(vars(self).items()))
        return f"{type(self).__name__}({args})"


def _check_positive(name: str, value: float) -> float:
    if not (value > 0):
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return float(value)


def _check_non_negative(name: str, value: float) -> float:
    if not (value >= 0):
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return float(value)


class Constant(Distribution):
    """A fixed value. Useful for pinning one input while others vary."""

    def __init__(self, value: float):
        self.value = float(value)

    def sample(self, n, rng):
        return _numpy().full(n, self.value)


class Uniform(Distribution):
    def __init__(self, low: float, high: float):
        if not (high > low):
            raise ValueError(f"high must exceed low, got low={low!r} high={high!r}")
        self.low, self.high = float(low), float(high)

    def sample(self, n, rng):
        return rng.uniform(self.low, self.high, n)


class Normal(Distribution):
    def __init__(self, mean: float, sd: float):
        self.mean = float(mean)
        self.sd = _check_non_negative("sd", sd)

    def sample(self, n, rng):
        return rng.normal(self.mean, self.sd, n)


class LogNormal(Distribution):
    """Log-normal in terms of the **underlying normal's** mu and sigma.

    This is the convention the C++ ``simulate_aggregate_loss`` uses, so the two
    agree. If you have a target mean and standard deviation on the natural
    scale, use :meth:`from_moments` instead — getting this backwards is the most
    common severity-modelling mistake there is.
    """

    def __init__(self, mu: float, sigma: float):
        self.mu = float(mu)
        self.sigma = _check_non_negative("sigma", sigma)

    @classmethod
    def from_moments(cls, mean: float, sd: float) -> "LogNormal":
        mean = _check_positive("mean", mean)
        sd = _check_non_negative("sd", sd)
        variance = sd * sd
        sigma_sq = math.log(1.0 + variance / (mean * mean))
        return cls(mu=math.log(mean) - 0.5 * sigma_sq, sigma=math.sqrt(sigma_sq))

    def sample(self, n, rng):
        return rng.lognormal(self.mu, self.sigma, n)


class Triangular(Distribution):
    """Low / most-likely / high. The estimator's distribution when all you have
    is a best guess and two bounds."""

    def __init__(self, low: float, mode: float, high: float):
        if not (low <= mode <= high):
            raise ValueError(f"need low <= mode <= high, got {low!r}, {mode!r}, {high!r}")
        if low == high:
            raise ValueError("low and high must differ")
        self.low, self.mode, self.high = float(low), float(mode), float(high)

    def sample(self, n, rng):
        return rng.triangular(self.low, self.mode, self.high, n)


class PERT(Distribution):
    """Beta-PERT — a smoother Triangular, standard in project and reserve risk.

    ``lam`` is the confidence placed on the mode; 4 is the classic PERT value.
    """

    def __init__(self, low: float, mode: float, high: float, lam: float = 4.0):
        if not (low <= mode <= high):
            raise ValueError(f"need low <= mode <= high, got {low!r}, {mode!r}, {high!r}")
        if low == high:
            raise ValueError("low and high must differ")
        self.low, self.mode, self.high = float(low), float(mode), float(high)
        self.lam = _check_non_negative("lam", lam)

    def sample(self, n, rng):
        span = self.high - self.low
        a = 1.0 + self.lam * (self.mode - self.low) / span
        b = 1.0 + self.lam * (self.high - self.mode) / span
        return self.low + rng.beta(a, b, n) * span


class Exponential(Distribution):
    def __init__(self, scale: float):
        self.scale = _check_positive("scale", scale)

    def sample(self, n, rng):
        return rng.exponential(self.scale, n)


class Gamma(Distribution):
    def __init__(self, shape: float, scale: float):
        self.shape = _check_positive("shape", shape)
        self.scale = _check_positive("scale", scale)

    def sample(self, n, rng):
        return rng.gamma(self.shape, self.scale, n)


class Beta(Distribution):
    def __init__(self, a: float, b: float):
        self.a = _check_positive("a", a)
        self.b = _check_positive("b", b)

    def sample(self, n, rng):
        return rng.beta(self.a, self.b, n)


class Pareto(Distribution):
    """Pareto Type I with scale ``xm`` and tail index ``alpha``.

    Matches the C++ catastrophe simulator. Note the mean is infinite for
    ``alpha <= 1`` and the variance for ``alpha <= 2`` — a sample mean will
    still print a number, and it will be meaningless.
    """

    def __init__(self, xm: float, alpha: float):
        self.xm = _check_positive("xm", xm)
        self.alpha = _check_positive("alpha", alpha)

    def sample(self, n, rng):
        return self.xm * (1.0 + rng.pareto(self.alpha, n))


class Poisson(Distribution):
    def __init__(self, mean: float):
        self.mean = _check_non_negative("mean", mean)

    def sample(self, n, rng):
        return rng.poisson(self.mean, n)


class NegativeBinomial(Distribution):
    """Over-dispersed count model — claim frequency when Poisson is too tidy."""

    def __init__(self, n_success: float, p: float):
        self.n_success = _check_positive("n_success", n_success)
        if not (0.0 < p <= 1.0):
            raise ValueError(f"p must be in (0, 1], got {p!r}")
        self.p = float(p)

    @classmethod
    def from_mean_dispersion(cls, mean: float, dispersion: float) -> "NegativeBinomial":
        """Parameterise by mean and variance-to-mean ratio (must exceed 1)."""
        mean = _check_positive("mean", mean)
        if not (dispersion > 1.0):
            raise ValueError(f"dispersion must be > 1 (else use Poisson), got {dispersion!r}")
        p = 1.0 / dispersion
        return cls(n_success=mean * p / (1.0 - p), p=p)

    def sample(self, n, rng):
        return rng.negative_binomial(self.n_success, self.p, n)


class Bernoulli(Distribution):
    def __init__(self, p: float):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p!r}")
        self.p = float(p)

    def sample(self, n, rng):
        return (rng.random(n) < self.p).astype(float)


class Binomial(Distribution):
    def __init__(self, trials: int, p: float):
        if trials < 0:
            raise ValueError(f"trials must be >= 0, got {trials!r}")
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p!r}")
        self.trials, self.p = int(trials), float(p)

    def sample(self, n, rng):
        return rng.binomial(self.trials, self.p, n)


class Empirical(Distribution):
    """Resample your own history with replacement — the bootstrap.

    Often the honest choice: it makes no distributional assumption at all, at
    the cost of never producing a loss larger than the worst one you have seen.
    """

    def __init__(self, data: Sequence[float]):
        np = _numpy()
        values = np.asarray(list(data), dtype=float)
        if values.size == 0:
            raise ValueError("Empirical needs at least one observation")
        self.data = values

    def sample(self, n, rng):
        return rng.choice(self.data, size=n, replace=True)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class Result:
    """The output of a run, plus the questions you actually ask of it.

    ``values`` is the simulated output, one entry per trial. ``inputs`` holds
    the sampled inputs (unless the run was told not to keep them), which is what
    makes sensitivity analysis possible after the fact.
    """

    values: Any
    inputs: Dict[str, Any]
    trials: int
    seed: Optional[int]
    label: str = "output"

    # -- summary statistics -------------------------------------------------

    @property
    def mean(self) -> float:
        return float(self.values.mean())

    @property
    def std(self) -> float:
        # Sample standard deviation (ddof=1): these are draws, not a population.
        return float(self.values.std(ddof=1))

    @property
    def standard_error(self) -> float:
        """How much the *mean* would wobble if you re-ran with a new seed.

        The number that tells you whether you ran enough trials. Shrinks with
        the square root of ``trials``, so ten times the precision costs a
        hundred times the work.
        """
        return self.std / math.sqrt(self.trials)

    def percentile(self, q: float) -> float:
        """``q`` in percent, e.g. ``percentile(99.5)``."""
        return float(_numpy().percentile(self.values, q))

    def var(self, level: float = 0.995) -> float:
        """Value at Risk — the loss exceeded with probability ``1 - level``."""
        _check_level(level)
        return self.percentile(level * 100.0)

    def tvar(self, level: float = 0.995) -> float:
        """Tail VaR / Expected Shortfall — the mean of the losses beyond VaR.

        Preferred to VaR for the same reason regulators prefer it: VaR tells you
        the threshold and nothing about how bad it gets past it.
        """
        _check_level(level)
        np = _numpy()
        threshold = self.var(level)
        tail = self.values[self.values >= threshold]
        if tail.size == 0:  # pragma: no cover - only if every value is identical
            return threshold
        return float(np.mean(tail))

    def prob_above(self, threshold: float) -> float:
        return float((self.values > threshold).mean())

    def prob_below(self, threshold: float) -> float:
        return float((self.values < threshold).mean())

    # -- diagnostics --------------------------------------------------------

    def convergence(self, points: int = 200):
        """Running mean and a ±1.96 standard-error band, for a convergence plot.

        Returns ``(n, running_mean, half_width)``. If the band is still visibly
        narrowing at the right-hand edge, you have not run enough trials.
        """
        np = _numpy()
        points = max(2, min(int(points), self.trials))
        idx = np.unique(np.linspace(1, self.trials, points).astype(int))
        cumulative = np.cumsum(self.values)
        running = cumulative[idx - 1] / idx
        # Running variance via the sum of squares — one pass, no Python loop.
        cumulative_sq = np.cumsum(self.values ** 2)
        mean_sq = cumulative_sq[idx - 1] / idx
        variance = np.maximum(mean_sq - running ** 2, 0.0)
        half_width = 1.96 * np.sqrt(variance / idx)
        return idx, running, half_width

    def sensitivity(self) -> List[tuple]:
        """Rank inputs by how much they move the output — the tornado data.

        Spearman rank correlation, not Pearson: the relationship between an
        input and a total is usually monotonic but rarely linear, and rank
        correlation does not care about the shape.

        Returns ``[(name, correlation), …]``, strongest first.
        """
        if not self.inputs:
            raise ValueError(
                "This run kept no inputs, so sensitivity cannot be computed. "
                "Re-run with keep_inputs=True (the default)."
            )
        np = _numpy()
        out_rank = _rankdata(self.values, np)
        scores = []
        for name, column in self.inputs.items():
            if float(np.std(column)) == 0.0:
                scores.append((name, 0.0))  # a Constant moves nothing
                continue
            corr = float(np.corrcoef(_rankdata(column, np), out_rank)[0, 1])
            scores.append((name, 0.0 if math.isnan(corr) else corr))
        scores.sort(key=lambda pair: abs(pair[1]), reverse=True)
        return scores

    # -- presentation -------------------------------------------------------

    def summary(self, levels: Iterable[float] = (0.5, 0.75, 0.9, 0.95, 0.99, 0.995)) -> str:
        rows = [
            f"Monte Carlo — {self.label}",
            f"  trials       {self.trials:,}" + (f"   seed {self.seed}" if self.seed is not None else ""),
            f"  mean         {self.mean:,.2f}   ± {1.96 * self.standard_error:,.2f} (95% CI)",
            f"  std dev      {self.std:,.2f}",
            f"  min / max    {float(self.values.min()):,.2f} / {float(self.values.max()):,.2f}",
            "",
            "  level        VaR              TVaR",
        ]
        for level in levels:
            rows.append(f"  {level:<12.3%} {self.var(level):>14,.2f}   {self.tvar(level):>14,.2f}")
        return "\n".join(rows)

    def to_frame(self):
        """A pandas DataFrame of inputs plus the output, if pandas is around."""
        try:
            import pandas  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("to_frame() needs pandas: pip install pandas") from exc
        data = dict(self.inputs)
        data[self.label] = self.values
        return pandas.DataFrame(data)

    def plot(self, kind: str = "distribution", **kwargs):
        """Shortcut into :mod:`riskpy.viz`.

        ``kind`` is one of ``distribution``, ``convergence``, ``tornado`` or
        ``exceedance``.
        """
        from . import viz

        plots = {
            "distribution": viz.distribution,
            "convergence": viz.convergence,
            "tornado": viz.tornado,
            "exceedance": viz.exceedance,
        }
        if kind not in plots:
            raise ValueError(f"kind must be one of {sorted(plots)}, got {kind!r}")
        return plots[kind](self, **kwargs)

    def __repr__(self) -> str:
        return (
            f"<Result {self.label}: {self.trials:,} trials, "
            f"mean {self.mean:,.2f}, p99.5 {self.var(0.995):,.2f}>"
        )


def _check_level(level: float) -> None:
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be a probability in (0, 1), e.g. 0.995 — got {level!r}")


def _rankdata(values, np):
    """Ranks with ties averaged. Small enough not to be worth a SciPy dependency."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)

    sorted_values = values[order]
    # Average the ranks within each run of equal values.
    start = 0
    for i in range(1, len(sorted_values) + 1):
        if i == len(sorted_values) or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


class Model:
    """A set of named random inputs plus a formula that combines them."""

    def __init__(self, **variables: Distribution):
        for name, dist in variables.items():
            if not isinstance(dist, Distribution):
                raise TypeError(
                    f"{name!r} must be a Distribution, got {type(dist).__name__}. "
                    f"Wrap a fixed number in Constant({dist!r})."
                )
        self.variables: Dict[str, Distribution] = dict(variables)
        self._formula: Optional[Callable[..., Any]] = None

    def formula(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register the formula. Usable as a decorator; returns ``fn`` unchanged."""
        self._formula = fn
        return fn

    def run(
        self,
        trials: int = 10_000,
        seed: Optional[int] = None,
        fn: Optional[Callable[..., Any]] = None,
        vectorised: bool = True,
        keep_inputs: bool = True,
        label: Optional[str] = None,
    ) -> Result:
        formula = fn or self._formula
        if formula is None:
            raise ValueError(
                "No formula registered. Decorate one with @model.formula, or "
                "pass fn= to run()."
            )
        return simulate(
            formula,
            trials=trials,
            seed=seed,
            vectorised=vectorised,
            keep_inputs=keep_inputs,
            label=label or getattr(formula, "__name__", "output"),
            **self.variables,
        )

    def __repr__(self) -> str:
        names = ", ".join(self.variables)
        return f"<Model({names})>"


def simulate(
    fn: Callable[..., Any],
    trials: int = 10_000,
    seed: Optional[int] = None,
    vectorised: bool = True,
    keep_inputs: bool = True,
    label: str = "output",
    **variables: Distribution,
) -> Result:
    """Run ``fn`` against ``trials`` draws of each named distribution.

    ``fn`` receives one keyword argument per variable. Vectorised (the default)
    means it is called once with arrays of length ``trials``; set
    ``vectorised=False`` if the formula contains branching that NumPy cannot
    express, and it will be called once per trial instead.
    """
    np = _numpy()

    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials!r}")
    if not variables:
        raise ValueError("Nothing to simulate — pass at least one distribution.")

    rng = np.random.default_rng(seed)
    samples = {name: dist.sample(trials, rng) for name, dist in variables.items()}

    if vectorised:
        values = np.asarray(fn(**samples), dtype=float)
        if values.shape != (trials,):
            raise ValueError(
                f"The formula returned shape {values.shape}, expected ({trials},). "
                f"A vectorised formula must operate elementwise on its inputs — "
                f"use np.where instead of if/else, np.minimum instead of min(), "
                f"and so on. Or pass vectorised=False to loop instead."
            )
    else:
        values = np.empty(trials, dtype=float)
        for i in range(trials):
            values[i] = fn(**{name: column[i] for name, column in samples.items()})

    if not np.all(np.isfinite(values)):
        bad = int((~np.isfinite(values)).sum())
        raise ValueError(
            f"{bad:,} of {trials:,} trials produced NaN or infinity. Usually a "
            f"division by a variable that can reach zero, or a log of a variable "
            f"that can go negative."
        )

    return Result(
        values=values,
        inputs=samples if keep_inputs else {},
        trials=trials,
        seed=seed,
        label=label,
    )
