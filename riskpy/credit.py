"""Credit risk — structural and reduced-form default models, Basel, portfolios.

Two layers, split by what they cost to install:

* **Scalar maths** — expected and unexpected loss, the :func:`merton`
  structural model, hazard rates and CDS legs, the Vasicek/ASRF portfolio
  formulae, the Basel IRB risk weights and :class:`TransitionMatrix` — is pure
  Python and needs nothing beyond the standard library, so it works in the same
  lean install as the C++ core.
* **Portfolio simulation** (:func:`credit_portfolio_loss`) needs NumPy
  (``pip install open-riskpy[sim]``), imported lazily on first use, and hands
  back a :class:`riskpy.mc.Result` so every VaR, TVaR and convergence
  diagnostic in that class applies unchanged.

Conventions throughout. ``pd`` is a probability in ``[0, 1]`` over the stated
horizon, not a percentage. ``lgd`` is loss *given* default as a fraction of
exposure and ``recovery`` is ``1 - lgd``. Hazard rates, credit spreads and
risk-free rates are **annualised continuously compounded decimals**
(``0.0125``, not ``125`` basis points), and times are in years.

The one convention worth stating loudly: every ``rho`` here is an **asset**
correlation — the correlation of the latent creditworthiness variables, not of
the default indicators. The two are not close. A 20% asset correlation between
two 1% names implies a default correlation of about 2%, and quietly passing one
where the other is expected is the most common error in this corner of finance.

Nothing here plots. :func:`vasicek_loss_cdf` on a grid is the loss-distribution
chart, :meth:`TransitionMatrix.to_frame` is the heat map, and
:func:`credit_portfolio_loss` returns a :class:`riskpy.mc.Result` that
:mod:`riskpy.viz` already knows how to draw.

References
----------
Merton (1974), "On the Pricing of Corporate Debt"; Vasicek (2002), "The
Distribution of Loan Portfolio Value"; BCBS (2005), "An Explanatory Note on the
Basel II IRB Risk Weight Functions"; Hull, *Options, Futures and Other
Derivatives*, chapters on credit risk and credit derivatives.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

from . import _special
from .mc import _numpy
from .quant import black_scholes

__all__ = [
    # Single exposure
    "expected_loss",
    "unexpected_loss",
    # Structural
    "MertonResult",
    "merton",
    # Reduced form
    "survival_probability",
    "hazard_from_spread",
    "spread_from_hazard",
    "CDSLegs",
    "cds_legs",
    "cds_par_spread",
    # Portfolio, analytic
    "asrf_conditional_pd",
    "vasicek_loss_cdf",
    "vasicek_loss_quantile",
    "basel_correlation",
    "IRBCapital",
    "basel_irb_capital",
    # Portfolio, simulated
    "credit_portfolio_loss",
    # Ratings
    "TransitionMatrix",
    "sp_transition_matrix",
]

_SUPERVISORY_LEVEL = 0.999  # Basel's confidence level, fixed by the Accord.


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_finite(name: str, value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _check_positive(name: str, value: float) -> float:
    value = _check_finite(name, value)
    if not value > 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return value


def _check_non_negative(name: str, value: float) -> float:
    value = _check_finite(name, value)
    if not value >= 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return value


def _check_probability(name: str, value: float) -> float:
    """A probability in the closed unit interval. Percentages are the usual slip."""
    value = _check_finite(name, value)
    if not 0.0 <= value <= 1.0:
        hint = " — probabilities here are decimals, so 1% is 0.01" if value > 1.0 else ""
        raise ValueError(f"{name} must be in [0, 1], got {value!r}{hint}")
    return value


def _check_level(name: str, value: float) -> float:
    """A confidence level strictly inside (0, 1); 0 and 1 have no finite quantile."""
    value = _check_finite(name, value)
    if not 0.0 < value < 1.0:
        raise ValueError(
            f"{name} must be strictly inside (0, 1), got {value!r} — "
            f"0.999 means the 99.9th percentile"
        )
    return value


def _check_frequency(value: object, name: str = "frequency") -> int:
    """A positive integer number of payments per year. Rejects bools and 2.5."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if isinstance(value, numbers.Integral):
        m = int(value)
    elif isinstance(value, numbers.Real) and float(value).is_integer():
        m = int(value)
    else:
        raise ValueError(f"{name} must be a positive integer (payments per year), got {value!r}")
    if m < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")
    return m


def _check_count(value: object, name: str = "n") -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        if not (isinstance(value, numbers.Real) and float(value).is_integer()):
            raise ValueError(f"{name} must be a whole number of periods, got {value!r}")
    n = int(value)
    if n < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return n


# ---------------------------------------------------------------------------
# One exposure
# ---------------------------------------------------------------------------


def expected_loss(pd: float, lgd: float, ead: float) -> float:
    """Expected loss on a single exposure: ``PD · LGD · EAD``.

    The mean of the loss over the horizon that ``pd`` refers to. It is a cost
    of doing business, not a risk measure — it is what provisions and the
    credit spread in the price are supposed to cover. Capital exists for the
    *unexpected* part, which is why :func:`basel_irb_capital` subtracts this
    number out.

    ``lgd`` is a fraction of ``ead``, so a 45% loss given default is ``0.45``.
    Values above 1 are permitted (workout costs can exceed the exposure) but
    are unusual enough to be worth a second look.
    """
    pd = _check_probability("pd", pd)
    lgd = _check_non_negative("lgd", lgd)
    ead = _check_non_negative("ead", ead)
    return pd * lgd * ead


def unexpected_loss(pd: float, lgd: float, ead: float) -> float:
    """Standard deviation of the loss on a single exposure.

    ``sqrt(PD·(1 − PD)) · LGD · EAD``: the default indicator is Bernoulli, so
    its standard deviation is ``sqrt(p(1−p))``, and a **deterministic** LGD
    simply scales it.

    That assumption is the whole content of the formula and it is optimistic.
    Realised LGD is itself a random variable with a standard deviation of
    20–30 percentage points, and it is correlated with the default rate — bad
    years are bad twice over. With a stochastic LGD of mean ``m`` and variance
    ``v`` the exposure standard deviation becomes
    ``EAD·sqrt(PD·v + PD·(1−PD)·m²)``, which is strictly larger. Use this
    function as a floor, not as an answer.

    Note the shape: it peaks at ``PD = 0.5`` and vanishes at both ends, so a
    near-certain default carries large expected loss and almost no *risk* in
    this sense.
    """
    pd = _check_probability("pd", pd)
    lgd = _check_non_negative("lgd", lgd)
    ead = _check_non_negative("ead", ead)
    return math.sqrt(pd * (1.0 - pd)) * lgd * ead


# ---------------------------------------------------------------------------
# Structural: Merton
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MertonResult:
    """The output of :func:`merton`, in one place.

    ``distance_to_default`` is the number of asset-volatility standard
    deviations between the expected log asset value at ``T`` and the default
    point — Merton's ``d2``. ``pd`` is ``N(-d2)``.

    ``equity``, ``debt_value`` and ``credit_spread`` are **prices**, and prices
    are always risk-neutral: they do not change when you supply a physical
    drift ``mu``. Only ``distance_to_default`` and ``pd`` switch measure. See
    :func:`merton` for why that is the right behaviour and not an oversight.
    """

    distance_to_default: float
    pd: float
    equity: float
    debt_value: float
    credit_spread: float
    leverage: float

    def __str__(self) -> str:
        return (
            f"distance to default {self.distance_to_default:,.4f}   PD {self.pd:.4%}\n"
            f"equity {self.equity:,.4f}   debt {self.debt_value:,.4f}   "
            f"leverage {self.leverage:.4f}\n"
            f"credit spread {self.credit_spread * 1e4:,.2f} bp"
        )


def merton(
    assets: float,
    debt: float,
    T: float,
    r: float,
    sigma: float,
    mu: Optional[float] = None,
) -> MertonResult:
    """Merton's (1974) structural model of a firm financed by one zero-coupon bond.

    The firm's asset value follows geometric Brownian motion and defaults at
    ``T`` if the assets are worth less than the face value ``debt``. Equity is
    then exactly a European call on the assets struck at the debt, so the
    equity value here *is*
    :func:`riskpy.quant.black_scholes` — the model is not a new formula, it is
    a change of what the underlying means.

    Parameters: ``assets`` is the market value of the firm's assets, ``debt``
    the face value of the zero-coupon debt maturing at ``T`` years, ``r`` the
    continuously compounded risk-free rate, ``sigma`` the annualised **asset**
    volatility, and ``mu`` the optional physical (real-world) drift of the
    assets.

    **Measure.** Without ``mu`` the distance to default is the risk-neutral
    ``d2 = (ln(V/D) + (r − σ²/2)T) / (σ√T)`` and ``pd = N(−d2)`` is the
    risk-neutral default probability — the one embedded in bond and CDS
    prices. Supply ``mu`` and the distance to default is computed under the
    physical measure instead, giving the *actual* probability of default, which
    is the number a KMV/Moody's-style EDF reports and is materially smaller
    when ``mu > r``. The prices in the result stay risk-neutral either way,
    because a physical drift cannot change what something costs.

    **What people get wrong.** ``sigma`` is asset volatility, which is not
    observable; the observed *equity* volatility is levered and much larger.
    Feeding equity volatility straight in overstates the default probability,
    often by a lot. The industry fix is to solve the two equations
    ``E = BS(V, D, …)`` and ``σ_E·E = N(d1)·σ_A·V`` jointly for ``V`` and
    ``σ_A``; that calibration is deliberately not in this function, so that
    what this function does stays unambiguous.

    ``sigma = 0`` is allowed and gives the deterministic limit: a solvent firm
    has zero credit spread and zero default probability, and an insolvent one
    has probability 1. That limit is the sharpest available test of the code.
    """
    assets = _check_positive("assets", assets)
    debt = _check_positive("debt", debt)
    T = _check_positive("T", T)
    r = _check_finite("r", r)
    sigma = _check_non_negative("sigma", sigma)
    drift = r if mu is None else _check_finite("mu", mu)

    # Equity is a call on the assets. Delegating keeps one implementation of
    # Black-Scholes in the package, degenerate cases included.
    equity = black_scholes(S=assets, K=debt, T=T, r=r, sigma=sigma, kind="call")
    debt_value = assets - equity

    moneyness = math.log(assets / debt)
    if sigma == 0.0:
        # Deterministic assets: default is a certainty or an impossibility.
        gap = moneyness + drift * T
        distance = math.inf if gap > 0.0 else (-math.inf if gap < 0.0 else 0.0)
        pd = 0.0 if gap > 0.0 else (1.0 if gap < 0.0 else 0.5)
    else:
        distance = (moneyness + (drift - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        pd = _special.norm_cdf(-distance)

    # Continuously compounded yield on the risky debt, minus the risk-free rate.
    # debt_value can only reach 0 if the assets do, which the validation forbids.
    credit_spread = -math.log(debt_value / debt) / T - r

    return MertonResult(
        distance_to_default=distance,
        pd=pd,
        equity=equity,
        debt_value=debt_value,
        credit_spread=credit_spread,
        # Merton's quasi-debt ratio: the debt discounted at the risk-free rate,
        # per unit of assets. Above 1 the firm is already underwater on a
        # present-value basis.
        leverage=debt * math.exp(-r * T) / assets,
    )


# ---------------------------------------------------------------------------
# Reduced form: hazard rates and CDS
# ---------------------------------------------------------------------------

HazardLike = Union[float, Sequence[Tuple[float, float]]]


def _hazard_curve(hazard: HazardLike, name: str = "hazard") -> List[Tuple[float, float]]:
    """Normalise ``hazard`` to ``[(end_time, rate), …]`` with a final ``inf`` knot.

    A float becomes a single flat segment. A sequence of ``(time, rate)`` pairs
    is a piecewise-constant term structure: ``rate`` applies from the previous
    knot up to ``time``, and the last rate continues beyond the last knot.
    """
    if isinstance(hazard, numbers.Real) and not isinstance(hazard, bool):
        rate = _check_non_negative(name, hazard)
        return [(math.inf, rate)]

    try:
        pairs = [tuple(item) for item in hazard]
    except TypeError:
        raise ValueError(
            f"{name} must be a rate, or a sequence of (time, rate) pairs giving a "
            f"piecewise-constant term structure, got {hazard!r}"
        ) from None
    if not pairs:
        raise ValueError(f"{name} term structure is empty")

    knots: List[Tuple[float, float]] = []
    previous = 0.0
    for index, item in enumerate(pairs):
        if len(item) != 2:
            raise ValueError(f"{name}[{index}] must be a (time, rate) pair, got {item!r}")
        t = _check_positive(f"{name}[{index}] time", item[0])
        rate = _check_non_negative(f"{name}[{index}] rate", item[1])
        if not t > previous:
            raise ValueError(
                f"{name} times must be strictly increasing, got {previous!r} then {t!r}"
            )
        knots.append((t, rate))
        previous = t
    knots[-1] = (math.inf, knots[-1][1])
    return knots


def _cumulative_hazard(knots: Sequence[Tuple[float, float]], t: float) -> float:
    total = 0.0
    start = 0.0
    for end, rate in knots:
        if t <= start:
            break
        total += rate * (min(t, end) - start)
        start = end
    return total


def survival_probability(hazard: HazardLike, t: float) -> float:
    """Probability of surviving to ``t`` years: ``exp(-∫₀ᵗ h(u) du)``.

    ``hazard`` is either a flat annualised default intensity, giving
    ``exp(-h·t)``, or a sequence of ``(time, rate)`` pairs describing a
    **piecewise-constant** term structure — ``[(1.0, 0.01), (5.0, 0.02)]`` means
    1% for the first year and 2% from year 1 to year 5, with the last rate
    continuing beyond the final knot.

    The hazard is an *instantaneous* rate of default conditional on having
    survived, not a probability. That is why a 5% hazard over one year gives a
    4.88% default probability, not 5%: ``1 − e^{−0.05}``. The gap grows fast
    for distressed names, where the difference between 50% and ``1 − e^{−0.5}``
    = 39% matters.
    """
    t = _check_non_negative("t", t)
    return math.exp(-_cumulative_hazard(_hazard_curve(hazard), t))


def hazard_from_spread(spread: float, recovery: float) -> float:
    """Hazard rate implied by a credit spread — the "credit triangle" ``h ≈ s/(1−R)``.

    The identity is *exact* in the continuous-payment limit with a flat hazard:
    the protection leg is ``(1−R)·h·∫e^{−(r+h)t}dt`` and the risky annuity is
    ``∫e^{−(r+h)t}dt``, so the ratio is ``h(1−R)`` with the discount factor and
    the maturity cancelling completely. That cancellation is why the rule of
    thumb survives contact with reality at all.

    Against the quarterly-premium :func:`cds_par_spread` it is an
    approximation, accurate to better than 0.1% of the spread for hazards
    under 5% and to about 1% at a hazard of 20%.

    The one thing people get wrong: ``recovery``, not ``lgd``. A 40% recovery is
    ``0.4``, and the spread is divided by ``0.6``.
    """
    spread = _check_non_negative("spread", spread)
    recovery = _check_probability("recovery", recovery)
    if recovery == 1.0:
        raise ValueError(
            "recovery of 1 means default costs nothing, so no hazard rate can produce "
            "a positive spread — use a recovery below 1"
        )
    return spread / (1.0 - recovery)


def spread_from_hazard(hazard: float, recovery: float) -> float:
    """Credit spread implied by a hazard rate: ``s = h·(1−R)``.

    The inverse of :func:`hazard_from_spread`, with the same standing and the
    same accuracy. Returned as a decimal, so multiply by 10,000 for basis
    points.
    """
    hazard = _check_non_negative("hazard", hazard)
    recovery = _check_probability("recovery", recovery)
    return hazard * (1.0 - recovery)


@dataclass(frozen=True)
class CDSLegs:
    """The two sides of a credit default swap, valued per :func:`cds_legs`.

    ``risky_annuity`` is the present value of a **unit annual spread** paid on
    ``notional`` until default or maturity, accrual included — the desk's
    "risky PV01" times 10,000. ``protection_leg`` is the present value of the
    contingent ``(1−R)·notional`` payment. ``premium_leg`` is
    ``spread · risky_annuity``.

    ``upfront`` is what the protection *buyer* pays at inception:
    ``protection_leg − premium_leg``. It is zero at the par spread by
    construction, positive when the contract's fixed coupon is below fair, and
    negative when it is above — which is the normal case for the standardised
    100bp and 500bp coupons traded today.
    """

    par_spread: float
    risky_annuity: float
    protection_leg: float
    premium_leg: float
    upfront: float
    spread: float

    def __str__(self) -> str:
        return (
            f"par spread {self.par_spread * 1e4:,.2f} bp   "
            f"contract spread {self.spread * 1e4:,.2f} bp\n"
            f"risky annuity {self.risky_annuity:,.6f}   "
            f"protection {self.protection_leg:,.6f}   premium {self.premium_leg:,.6f}\n"
            f"upfront (buyer pays) {self.upfront:,.6f}"
        )


def _cds_schedule(maturity: float, frequency: int) -> List[float]:
    """Payment times, counted back from ``maturity`` so any stub comes first.

    Same convention as :class:`riskpy.rates.Bond`: a 2.5-year quarterly
    schedule has ten regular periods, and a 2.6-year one has a short first
    period followed by ten regular ones.
    """
    periods = int(math.ceil(maturity * frequency - 1e-9))
    return [maturity - (periods - k) / frequency for k in range(1, periods + 1)]


def cds_legs(
    hazard: HazardLike,
    recovery: float,
    r: float,
    maturity: float,
    spread: Optional[float] = None,
    frequency: int = 4,
    notional: float = 1.0,
) -> CDSLegs:
    """Value both legs of a CDS under a piecewise-constant hazard and a flat curve.

    ``hazard`` is a flat intensity or a ``(time, rate)`` term structure, exactly
    as in :func:`survival_probability`. ``r`` is a flat continuously compounded
    discount rate, ``maturity`` is in years and ``frequency`` the premium
    payments per year (4 is the market standard). ``spread`` is the contract's
    fixed coupon; leave it ``None`` to value at the par spread, which makes the
    legs equal and the upfront zero.

    **Accrual convention**, since every implementation picks one and they
    disagree in the third decimal place:

    * premium periods run between the dates of :func:`_cds_schedule`, counted
      back from maturity so a stub falls first, with an accrual factor equal to
      the period's actual length in years (ACT/ACT on a continuous timeline —
      no ACT/360 day counting, no business-day rolls, no holiday calendars);
    * a surviving premium is paid in full at the period end and discounted
      there;
    * **default is assumed to happen at the midpoint of the period it falls
      in**. The protection payment ``(1−R)`` and the accrued premium (half a
      period) are both discounted to that midpoint.

    The midpoint rule is the standard first-order fix for the fact that a
    period is not a point. Paying protection at the period *end* instead would
    understate the leg by roughly ``r·Δ/2``; assuming it at the *start* would
    overstate it by the same. The midpoint is accurate to ``O(Δ²)`` and is what
    makes the par spread reproduce the credit triangle ``h(1−R)`` to a fraction
    of a basis point.

    Not modelled: a term structure of interest rates (pass the flat equivalent),
    counterparty risk on the protection seller, and the standardised
    fixed-coupon/upfront quoting mechanics beyond the ``upfront`` field.
    """
    knots = _hazard_curve(hazard)
    recovery = _check_probability("recovery", recovery)
    r = _check_finite("r", r)
    maturity = _check_positive("maturity", maturity)
    frequency = _check_frequency(frequency)
    notional = _check_non_negative("notional", notional)

    annuity = 0.0
    protection = 0.0
    start = 0.0
    survival_start = 1.0
    for end in _cds_schedule(maturity, frequency):
        length = end - start
        survival_end = math.exp(-_cumulative_hazard(knots, end))
        default_probability = survival_start - survival_end
        middle = 0.5 * (start + end)
        df_end = math.exp(-r * end)
        df_mid = math.exp(-r * middle)

        annuity += length * df_end * survival_end          # paid in full if it survives
        annuity += 0.5 * length * df_mid * default_probability  # accrued if it does not
        protection += (1.0 - recovery) * df_mid * default_probability

        start, survival_start = end, survival_end

    annuity *= notional
    protection *= notional

    if annuity <= 0.0:  # pragma: no cover - needs a zero notional or maturity
        raise ValueError(
            "the risky annuity is zero, so no par spread exists — check notional and maturity"
        )
    par = protection / annuity
    contract = par if spread is None else _check_finite("spread", spread)
    premium = contract * annuity
    return CDSLegs(
        par_spread=par,
        risky_annuity=annuity,
        protection_leg=protection,
        premium_leg=premium,
        upfront=protection - premium,
        spread=contract,
    )


def cds_par_spread(
    hazard: HazardLike,
    recovery: float,
    r: float,
    maturity: float,
    frequency: int = 4,
) -> float:
    """The fixed spread that makes a CDS worth zero at inception, as a decimal.

    ``protection leg / risky annuity`` from :func:`cds_legs`, whose docstring
    carries the accrual convention. Multiply by 10,000 for basis points.

    Notice what it does *not* depend on, to first order: the discount rate and
    the maturity nearly cancel between the two legs, which is why the credit
    triangle ``s ≈ h(1−R)`` works at all and why a CDS quote is a far cleaner
    read on default risk than a bond spread, which carries funding and
    liquidity along with it.
    """
    return cds_legs(hazard, recovery, r, maturity, frequency=frequency).par_spread


# ---------------------------------------------------------------------------
# Portfolio: the one-factor Gaussian model, analytically
# ---------------------------------------------------------------------------


def asrf_conditional_pd(pd: float, rho: float, level: float) -> float:
    """Default probability conditional on a bad draw of the systematic factor.

    ``Φ((Φ⁻¹(PD) + √ρ · Φ⁻¹(level)) / √(1−ρ))`` — the Vasicek/Basel
    *asymptotic single risk factor* formula, and the engine inside every IRB
    risk weight.

    The model: a borrower's latent creditworthiness is
    ``A = √ρ·Z + √(1−ρ)·ε`` with ``Z`` a systematic factor common to everyone
    and ``ε`` idiosyncratic, both standard normal, and it defaults when
    ``A < Φ⁻¹(PD)``. Conditioning on the ``level`` quantile of *bad* outcomes —
    ``Z = −Φ⁻¹(level)``, hence the plus sign in the formula — gives the default
    rate in a stress of that severity.

    ``level`` is the severity of the stress, so 0.999 means "the worst year in
    a thousand" and gives a number far above ``pd``. ``rho`` is an **asset**
    correlation. At ``rho = 0`` the answer is ``pd`` at every level, because
    with no common factor a large portfolio has no uncertainty left to stress.

    The one thing people get wrong: reading the output as a probability of the
    *portfolio* defaulting. It is the fraction of a large, fine-grained
    portfolio expected to default in a stress that severe — a loss *rate*, not
    an event probability.
    """
    pd = _check_probability("pd", pd)
    rho = _check_probability("rho", rho)
    level = _check_level("level", level)

    if pd == 0.0 or pd == 1.0:
        return pd
    if rho == 0.0:
        return pd
    z_pd = _special.norm_ppf(pd)
    z_level = _special.norm_ppf(level)
    if rho == 1.0:
        # Perfect correlation: the whole portfolio defaults together, so the
        # conditional rate collapses to 0 or 1 as the stress passes the barrier.
        combined = z_pd + z_level
        return 1.0 if combined > 0.0 else (0.0 if combined < 0.0 else 0.5)
    return _special.norm_cdf((z_pd + math.sqrt(rho) * z_level) / math.sqrt(1.0 - rho))


def vasicek_loss_cdf(x: float, pd: float, rho: float) -> float:
    """``P(L ≤ x)`` for the loss fraction of a large homogeneous portfolio.

    Vasicek's (2002) limiting distribution:
    ``F(x) = Φ((√(1−ρ)·Φ⁻¹(x) − Φ⁻¹(PD)) / √ρ)``, valid as the number of
    equally weighted exposures goes to infinity so that all idiosyncratic risk
    diversifies away and only the systematic factor is left.

    ``x`` is a **fraction of the portfolio in default**, in ``[0, 1]``, and the
    distribution assumes ``LGD = 1``; for a deterministic LGD apply
    ``F(x/LGD)``. The shape is the point: strongly right-skewed with a mode
    below the mean whenever ``ρ`` is small and ``PD`` is small, so a typical
    year looks better than average and the bad years are very bad.

    Evaluate it on a grid of ``x`` to get the data behind a loss-distribution
    chart. The degenerate ends are handled: ``ρ = 0`` puts all mass at ``PD``
    and ``ρ = 1`` splits it between 0 and 1.

    The one thing people get wrong: applying it to a portfolio of 40 names.
    The limit ignores granularity, and a real portfolio's tail is fatter than
    this by a granularity adjustment that grows like ``1/n``. Compare against
    :func:`credit_portfolio_loss` before trusting it on a concentrated book.
    """
    x = _check_finite("x", x)
    pd = _check_probability("pd", pd)
    rho = _check_probability("rho", rho)

    if pd == 0.0:
        return 1.0 if x >= 0.0 else 0.0
    if pd == 1.0:
        return 1.0 if x >= 1.0 else 0.0
    if rho == 0.0:
        return 1.0 if x >= pd else 0.0
    if rho == 1.0:
        return 1.0 if x >= 1.0 else (1.0 - pd if x >= 0.0 else 0.0)
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    numerator = math.sqrt(1.0 - rho) * _special.norm_ppf(x) - _special.norm_ppf(pd)
    return _special.norm_cdf(numerator / math.sqrt(rho))


def vasicek_loss_quantile(level: float, pd: float, rho: float) -> float:
    """The ``level`` quantile of the Vasicek loss fraction — the inverse of
    :func:`vasicek_loss_cdf`.

    Inverting the CDF gives ``Φ((Φ⁻¹(PD) + √ρ·Φ⁻¹(level)) / √(1−ρ))``, which is
    *exactly* :func:`asrf_conditional_pd`. That coincidence is not a
    coincidence: Basel's capital formula is the 99.9% quantile of this loss
    distribution, and the reason it can be written as a per-exposure conditional
    default probability is that in the asymptotic limit the portfolio quantile
    decomposes into a sum of contributions, one per name. Both functions are
    kept because the two readings are genuinely different questions.

    ``level`` here is a quantile of the *loss*, so 0.999 is a severe outcome.
    """
    level = _check_level("level", level)
    pd = _check_probability("pd", pd)
    rho = _check_probability("rho", rho)
    if rho == 1.0:
        # Bernoulli(pd) on {0, 1}: everything or nothing.
        return 1.0 if level > 1.0 - pd else 0.0
    return asrf_conditional_pd(pd, rho, level)


# ---------------------------------------------------------------------------
# Basel II / III internal ratings based approach
# ---------------------------------------------------------------------------

_CORPORATE_CLASSES = ("corporate", "sovereign", "bank")
_RETAIL_CLASSES = ("retail_mortgage", "retail_revolving", "retail_other")
_ASSET_CLASSES = _CORPORATE_CLASSES + _RETAIL_CLASSES


def _check_asset_class(asset_class: str) -> str:
    if not isinstance(asset_class, str):
        raise ValueError(f"asset_class must be a string, got {asset_class!r}")
    key = asset_class.strip().lower()
    if key not in _ASSET_CLASSES:
        raise ValueError(
            f"asset_class must be one of {', '.join(_ASSET_CLASSES)}, got {asset_class!r}"
        )
    return key


def _exposure_weight(pd: float, k: float) -> float:
    """Basel's exponential blend ``(1 − e^{−k·PD}) / (1 − e^{−k})``, in [0, 1]."""
    return -math.expm1(-k * pd) / -math.expm1(-k)


def basel_correlation(pd: float, asset_class: str = "corporate") -> float:
    """Supervisory asset correlation ``R`` for the Basel IRB risk-weight functions.

    ==================  ==========================================================
    ``asset_class``     correlation
    ==================  ==========================================================
    ``corporate``       ``0.12·w + 0.24·(1−w)``, ``w = (1−e^{−50·PD})/(1−e^{−50})``
    ``sovereign``       as corporate
    ``bank``            as corporate
    ``retail_mortgage`` 0.15, flat
    ``retail_revolving`` 0.04, flat
    ``retail_other``    ``0.03·w + 0.16·(1−w)``, ``w = (1−e^{−35·PD})/(1−e^{−35})``
    ==================  ==========================================================

    The corporate curve runs from 0.24 at ``PD → 0`` down to 0.12 at
    ``PD → 1``, and the ``e^{−50·PD}`` decay means it has already fallen to
    0.19 by a 1% PD. That direction surprises people every time: *riskier
    borrowers are less correlated*. The supervisory story is that a
    high-PD firm is in trouble for its own reasons, so its default carries
    more idiosyncratic and less systematic information — and the empirical
    evidence behind it is thin. These are negotiated constants, fixed by the
    Accord, not parameters you estimate; that is why they take no data.

    Deliberately not implemented: the firm-size adjustment for SMEs
    (``−0.04·(1 − (S−5)/45)`` for annual sales ``S`` between €5m and €50m) and
    the 1.25 multiplier on large financial institutions, both of which are
    jurisdiction-specific overlays rather than part of the core formula.
    """
    pd = _check_probability("pd", pd)
    key = _check_asset_class(asset_class)

    if key in _CORPORATE_CLASSES:
        w = _exposure_weight(pd, 50.0)
        return 0.12 * w + 0.24 * (1.0 - w)
    if key == "retail_mortgage":
        return 0.15
    if key == "retail_revolving":
        return 0.04
    w = _exposure_weight(pd, 35.0)
    return 0.03 * w + 0.16 * (1.0 - w)


@dataclass(frozen=True)
class IRBCapital:
    """The output of :func:`basel_irb_capital`.

    ``capital_requirement`` is Basel's ``K``: capital per unit of EAD, covering
    **unexpected** loss only. ``capital`` is ``K · EAD`` in currency, ``rwa`` is
    ``12.5 · K · EAD`` — the 12.5 being ``1/0.08``, which converts a capital
    requirement back into the risk-weighted assets a bank reports against an 8%
    minimum ratio.
    """

    capital_requirement: float
    capital: float
    rwa: float
    correlation: float
    conditional_pd: float
    maturity_adjustment: float

    @property
    def risk_weight(self) -> float:
        """``RWA / EAD`` — ``12.5 · K``, the number quoted as "a 92% risk weight"."""
        return 12.5 * self.capital_requirement

    def __str__(self) -> str:
        return (
            f"K {self.capital_requirement:.4%} of EAD   risk weight {self.risk_weight:.2%}\n"
            f"capital {self.capital:,.2f}   RWA {self.rwa:,.2f}\n"
            f"correlation {self.correlation:.4f}   conditional PD {self.conditional_pd:.4%}   "
            f"maturity adj {self.maturity_adjustment:.4f}"
        )


def basel_irb_capital(
    pd: float,
    lgd: float,
    ead: float,
    maturity: float = 2.5,
    asset_class: str = "corporate",
) -> IRBCapital:
    """Basel II/III internal-ratings-based capital for one exposure.

    ``K = [LGD·Φ((Φ⁻¹(PD) + √R·Φ⁻¹(0.999)) / √(1−R)) − PD·LGD] ·
    (1 + (M − 2.5)·b) / (1 − 1.5·b)`` with ``b = (0.11852 − 0.05478·ln PD)²``
    and ``R`` from :func:`basel_correlation`. ``RWA = 12.5 · K · EAD``.

    Read it in three pieces. The first bracket is the 99.9% loss rate from
    :func:`asrf_conditional_pd` — a one-in-a-thousand-years systematic stress.
    Subtracting ``PD·LGD`` removes the *expected* loss, which provisions are
    supposed to cover already, leaving capital to absorb only the unexpected
    part; forgetting this subtraction is the classic double count. The final
    factor is the maturity adjustment, calibrated so that ``M = 1`` leaves the
    number untouched and longer exposures attract more capital because they
    have time to be downgraded, not just to default.

    ``maturity`` is the effective maturity ``M`` in years, floored at 1 and
    capped at 5 by the Accord — passing something outside that range raises,
    rather than being silently clamped into compliance. The maturity adjustment
    does **not** apply to the retail classes, where ``maturity`` is ignored and
    reported as an adjustment of 1.0.

    ``pd`` must be strictly positive because ``ln PD`` appears in ``b``.
    Supervisors additionally impose a 0.03% floor on PD for corporate and bank
    exposures; that floor is a policy choice, so it is documented here and not
    applied for you.

    The one thing people get wrong: this is a *regulatory* capital number, not
    an economic one. It fixes the correlation by decree, the confidence level
    at 99.9%, and the LGD as deterministic; a portfolio model such as
    :func:`credit_portfolio_loss` will disagree with it, and neither of you is
    making an arithmetic error.
    """
    pd = _check_probability("pd", pd)
    lgd = _check_non_negative("lgd", lgd)
    ead = _check_non_negative("ead", ead)
    key = _check_asset_class(asset_class)
    maturity = _check_finite("maturity", maturity)

    if pd <= 0.0:
        raise ValueError(
            "pd must be > 0: the maturity adjustment contains ln(PD), which is "
            "undefined at zero. Use the supervisory floor of 0.0003 (3bp) if the "
            "obligor has never defaulted."
        )

    correlation = basel_correlation(pd, key)
    conditional = asrf_conditional_pd(pd, correlation, _SUPERVISORY_LEVEL)
    unexpected = lgd * conditional - pd * lgd

    if key in _RETAIL_CLASSES:
        adjustment = 1.0
    else:
        if not 1.0 <= maturity <= 5.0:
            raise ValueError(
                f"maturity must be between 1 and 5 years for {key} exposures "
                f"(the Accord's floor and cap), got {maturity!r}"
            )
        b = (0.11852 - 0.05478 * math.log(pd)) ** 2
        adjustment = (1.0 + (maturity - 2.5) * b) / (1.0 - 1.5 * b)

    k = unexpected * adjustment
    return IRBCapital(
        capital_requirement=k,
        capital=k * ead,
        rwa=12.5 * k * ead,
        correlation=correlation,
        conditional_pd=conditional,
        maturity_adjustment=adjustment,
    )


# ---------------------------------------------------------------------------
# Portfolio: the one-factor Gaussian copula, simulated
# ---------------------------------------------------------------------------

# Cells of the (trials x exposures) shock matrix held at once. 4 million doubles
# is 32MB, which keeps a thousand-name book at fifty thousand trials inside a
# laptop's cache budget instead of asking for two gigabytes in one allocation.
_BLOCK_CELLS = 4_000_000


def credit_portfolio_loss(
    pds: Sequence[float],
    lgds: Sequence[float],
    eads: Sequence[float],
    rho: Union[float, Sequence[float]],
    trials: int = 50_000,
    seed: Optional[int] = None,
):
    """Simulate portfolio credit loss under a one-factor Gaussian copula.

    Each obligor ``i`` carries a latent asset variable
    ``A_i = √ρ_i·Z + √(1−ρ_i)·ε_i`` with one systematic ``Z`` shared by the
    whole portfolio and independent idiosyncratic ``ε_i``, all standard normal.
    Obligor ``i`` defaults when ``A_i < Φ⁻¹(PD_i)``, which reproduces its
    marginal ``PD_i`` exactly, and the portfolio loss on that trial is
    ``Σ LGD_i · EAD_i`` over the defaulters. This is the CreditMetrics /
    Basel model, simulated instead of taken to its asymptotic limit — so unlike
    :func:`vasicek_loss_cdf` it sees concentration and granularity.

    ``rho`` is a scalar asset correlation applied to everyone, or one per
    exposure. LGD is **deterministic** here; stochastic and downturn-correlated
    LGD would fatten the tail further, and is the first thing to add if this
    number has to be defended.

    Returns a :class:`riskpy.mc.Result` labelled ``"portfolio loss"`` with
    ``values`` the loss per trial in the units of ``eads``, so
    ``result.var(0.999)``, ``result.tvar(0.99)`` and ``result.summary()`` work
    directly. ``result.inputs["systematic_factor"]`` holds the ``Z`` draw for
    each trial, which is what makes ``result.sensitivity()`` and a loss-versus-
    factor scatter possible after the fact.

    The one thing people get wrong: the Gaussian copula has **no tail
    dependence**. Two names with ``ρ = 0.9`` still become asymptotically
    independent deep in the tail, which is precisely the criticism levelled at
    it after 2008. If joint extreme defaults are the question, use a t-copula
    from :mod:`riskpy.capital` instead of raising ``rho`` until the answer
    looks scary enough.
    """
    np = _numpy()

    pd_list = [_check_probability(f"pds[{i}]", p) for i, p in enumerate(pds)]
    lgd_list = [_check_non_negative(f"lgds[{i}]", v) for i, v in enumerate(lgds)]
    ead_list = [_check_non_negative(f"eads[{i}]", v) for i, v in enumerate(eads)]
    n = len(pd_list)
    if n == 0:
        raise ValueError("the portfolio is empty — pds must have at least one exposure")
    if not len(lgd_list) == len(ead_list) == n:
        raise ValueError(
            f"pds, lgds and eads must have the same length, got "
            f"{n}, {len(lgd_list)} and {len(ead_list)}"
        )

    if isinstance(rho, numbers.Real) and not isinstance(rho, bool):
        rho_list = [_check_probability("rho", rho)] * n
    else:
        rho_list = [_check_probability(f"rho[{i}]", v) for i, v in enumerate(rho)]
        if len(rho_list) != n:
            raise ValueError(
                f"rho must be a scalar or one correlation per exposure, got "
                f"{len(rho_list)} for {n} exposures"
            )

    if not isinstance(trials, numbers.Integral) or isinstance(trials, bool) or trials < 1:
        raise ValueError(f"trials must be a positive integer, got {trials!r}")

    thresholds = _special.norm_ppf_vec(np.asarray(pd_list, dtype=float))
    rho_array = np.asarray(rho_list, dtype=float)
    systematic_weight = np.sqrt(rho_array)
    idiosyncratic_weight = np.sqrt(1.0 - rho_array)
    severity = np.asarray(lgd_list, dtype=float) * np.asarray(ead_list, dtype=float)

    rng = np.random.default_rng(seed)
    losses = np.empty(trials, dtype=float)
    factor = np.empty(trials, dtype=float)

    # Blocked so a large book does not demand one enormous allocation. The
    # blocking is a deterministic function of (n, trials), so a repeated seed
    # still reproduces the run exactly.
    block = max(1, min(trials, _BLOCK_CELLS // n))
    start = 0
    while start < trials:
        size = min(block, trials - start)
        z = rng.standard_normal(size)
        shocks = rng.standard_normal((size, n))
        assets = systematic_weight * shocks + idiosyncratic_weight * shocks
        # Recompute properly: the systematic term is common across the row.
        assets = systematic_weight * z[:, None] + idiosyncratic_weight * shocks
        defaulted = assets < thresholds
        losses[start:start + size] = defaulted.astype(float) @ severity
        factor[start:start + size] = z
        start += size

    from .mc import Result

    return Result(
        values=losses,
        inputs={"systematic_factor": factor},
        trials=trials,
        seed=seed,
        label="portfolio loss",
    )


# ---------------------------------------------------------------------------
# Rating transitions
# ---------------------------------------------------------------------------


def _identity(n: int) -> List[List[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    n = len(a)
    columns = [[b[k][j] for k in range(n)] for j in range(n)]
    return [[math.fsum(row[k] * column[k] for k in range(n)) for column in columns] for row in a]


class TransitionMatrix:
    """A one-period credit rating migration matrix, with the default state absorbing.

    ``matrix[i][j]`` is the probability that an obligor rated ``ratings[i]``
    today is rated ``ratings[j]`` one period later. Ratings run from best to
    worst and **the last one must be** ``"D"``: default is the model's
    absorbing state and everything in :meth:`cumulative_pd` reads the last
    column.

    Validated on construction — square, entries in ``[0, 1]``, every row summing
    to 1 within ``1e-8``, and the default row exactly ``(0, …, 0, 1)``. A matrix
    that fails any of these is not a Markov chain, and finding that out after a
    ten-year projection is worse than useless.

    The model assumes migration is **Markov and time-homogeneous**: where you
    came from does not matter and neither does the year. Both are false —
    downgrade momentum is real and default rates are strongly cyclical — so
    ``power(n)`` gives a through-the-cycle average, not a forecast for a
    particular ``n`` years.
    """

    def __init__(self, matrix: Sequence[Sequence[float]], ratings: Sequence[str]):
        rows = [list(row) for row in matrix]
        labels = [str(label) for label in ratings]
        n = len(rows)
        if n == 0:
            raise ValueError("a TransitionMatrix needs at least one rating")
        if len(labels) != n:
            raise ValueError(
                f"ratings has {len(labels)} labels for a {n}x{n} matrix — they must match"
            )
        if len(set(labels)) != n:
            raise ValueError(f"ratings must be unique, got {labels!r}")
        if labels[-1] != "D":
            raise ValueError(
                f"the last rating must be 'D', the absorbing default state, got "
                f"{labels[-1]!r} — order the ratings from best to worst"
            )

        for i, row in enumerate(rows):
            if len(row) != n:
                raise ValueError(
                    f"the matrix must be square: row {i} has {len(row)} entries, expected {n}"
                )
            for j, value in enumerate(row):
                value = _check_finite(f"matrix[{i}][{j}]", value)
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"matrix[{i}][{j}] must be a probability in [0, 1], got {value!r}"
                    )
                rows[i][j] = value
            total = math.fsum(row)
            if abs(total - 1.0) > 1e-8:
                raise ValueError(
                    f"row {i} ({labels[i]}) sums to {total!r}, not 1 — every obligor has to "
                    f"end up somewhere. If a 'not rated' column was dropped, renormalise "
                    f"the row or fold it into the diagonal."
                )

        default_row = rows[-1]
        if any(abs(v) > 1e-12 for v in default_row[:-1]) or abs(default_row[-1] - 1.0) > 1e-12:
            raise ValueError(
                f"the default row must be absorbing, i.e. (0, …, 0, 1), got {default_row!r} — "
                f"an issuer that has defaulted does not migrate back"
            )

        self._matrix: Tuple[Tuple[float, ...], ...] = tuple(tuple(row) for row in rows)
        self._ratings: Tuple[str, ...] = tuple(labels)

    # -- accessors ----------------------------------------------------------

    @property
    def matrix(self) -> Tuple[Tuple[float, ...], ...]:
        """The probabilities as a tuple of tuples — immutable, and the data a
        migration heat map needs."""
        return self._matrix

    @property
    def ratings(self) -> Tuple[str, ...]:
        return self._ratings

    @property
    def size(self) -> int:
        return len(self._ratings)

    def _index(self, rating: str) -> int:
        try:
            return self._ratings.index(rating)
        except ValueError:
            raise ValueError(
                f"unknown rating {rating!r}; this matrix has {', '.join(self._ratings)}"
            ) from None

    def probability(self, from_rating: str, to_rating: str) -> float:
        """One-period probability of migrating from one rating to another."""
        return self._matrix[self._index(from_rating)][self._index(to_rating)]

    # -- powers -------------------------------------------------------------

    def power(self, n: int) -> "TransitionMatrix":
        """The ``n``-period matrix ``P^n``, by repeated squaring.

        ``n = 0`` is the identity. Row sums stay at 1 to machine precision
        because each product is a convex combination of rows, and the
        accumulation uses ``math.fsum`` so the drift does not compound.

        ``n`` must be a non-negative integer. Fractional powers — "what is the
        six-month matrix?" — need the matrix logarithm, and the embedding
        problem bites: most empirical annual matrices have **no** valid
        generator, and forcing one produces negative transition probabilities.
        That is a genuinely different calculation and is deliberately not
        hidden behind this method.
        """
        n = _check_count(n, "n")
        size = self.size
        result = _identity(size)
        base = [list(row) for row in self._matrix]
        while n:
            if n & 1:
                result = _matmul(result, base)
            n >>= 1
            if n:
                base = _matmul(base, base)
        return TransitionMatrix(result, self._ratings)

    # -- default probabilities ---------------------------------------------

    def cumulative_pd(self, rating: str, n: int) -> float:
        """Probability of having defaulted at any point within ``n`` periods.

        ``P^n[rating, "D"]``. Non-decreasing in ``n`` because ``"D"`` absorbs,
        and ``0`` at ``n = 0`` for every non-defaulted rating.
        """
        n = _check_count(n, "n")
        index = self._index(rating)
        if n == 0:
            return 1.0 if index == self.size - 1 else 0.0
        return self.power(n).matrix[index][-1]

    def marginal_pd(self, rating: str, n: int) -> float:
        """Probability of defaulting **in** period ``n``, having survived to ``n−1``
        — the unconditional marginal, ``cumulative_pd(n) − cumulative_pd(n−1)``.

        Not the same as the *conditional* (forward) default rate, which divides
        this by the survival probability to ``n−1``. For investment-grade names
        the marginal rises with ``n`` for a decade or so, as ratings drift down;
        for CCC it falls sharply, because the survivors are the ones that got
        better. Mixing the two up is how a credit curve ends up inverted by
        accident.
        """
        n = _check_count(n, "n")
        if n < 1:
            raise ValueError(f"n must be >= 1 for a marginal default probability, got {n!r}")
        return self.cumulative_pd(rating, n) - self.cumulative_pd(rating, n - 1)

    # -- presentation -------------------------------------------------------

    def to_frame(self):
        """The matrix as a pandas DataFrame, rows and columns labelled by rating."""
        try:
            import pandas  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("to_frame() needs pandas: pip install pandas") from exc
        frame = pandas.DataFrame(
            [list(row) for row in self._matrix],
            index=list(self._ratings),
            columns=list(self._ratings),
        )
        frame.index.name = "from"
        frame.columns.name = "to"
        return frame

    def __repr__(self) -> str:
        width = max(6, max(len(label) for label in self._ratings) + 1)
        header = " " * width + "".join(f"{label:>{width}}" for label in self._ratings)
        rows = [
            f"{label:<{width}}" + "".join(f"{value * 100:>{width}.2f}" for value in row)
            for label, row in zip(self._ratings, self._matrix)
        ]
        return (
            f"<TransitionMatrix {self.size} ratings, one period, %>\n"
            + header
            + "\n"
            + "\n".join(rows)
        )


# S&P-style average one-year global corporate migration rates, in per cent,
# NR-adjusted and rounded. Off-diagonal entries only: the diagonal is filled in
# as one minus the row's off-diagonal total, so every row sums to exactly 1 and
# the published default and migration rates survive intact rather than being
# smeared by a renormalisation.
_SP_RATINGS = ("AAA", "AA", "A", "BBB", "BB", "B", "CCC/C", "D")
_SP_OFF_DIAGONAL = (
    (None, 9.09, 0.55, 0.05, 0.11, 0.03, 0.05, 0.00),
    (0.53, None, 8.60, 0.58, 0.06, 0.07, 0.02, 0.02),
    (0.03, 1.87, None, 5.61, 0.39, 0.16, 0.02, 0.06),
    (0.01, 0.12, 3.63, None, 3.96, 0.63, 0.13, 0.21),
    (0.01, 0.04, 0.15, 5.31, None, 4.94, 0.61, 0.70),
    (0.00, 0.03, 0.11, 0.22, 5.62, None, 3.72, 3.11),
    (0.00, 0.00, 0.15, 0.23, 0.68, 12.24, None, 35.14),
)


def sp_transition_matrix() -> TransitionMatrix:
    """A realistic S&P-style average one-year corporate rating transition matrix.

    Eight states, ``AAA`` through ``CCC/C`` and ``D``, with the published
    average one-year global corporate migration rates rounded to two decimal
    places and adjusted for withdrawn ratings. The diagonal carries the rounding
    residual so each row sums to exactly one; the off-diagonal entries, and in
    particular the default column, are the published figures.

    It is here so the rest of this section has something real to run against.
    Treat the numbers as illustrative, not as a citable data source: the true
    figures move with the cohort window and the NR treatment, and a real
    application should load the current study rather than a rounded copy of an
    old one.

        >>> sp_transition_matrix().cumulative_pd("BBB", 5) > 0.01
        True
    """
    size = len(_SP_RATINGS)
    rows: List[List[float]] = []
    for i, row in enumerate(_SP_OFF_DIAGONAL):
        values = [0.0 if value is None else value / 100.0 for value in row]
        values[i] = 1.0 - math.fsum(values[:i] + values[i + 1:])
        rows.append(values)
    rows.append([0.0] * (size - 1) + [1.0])
    return TransitionMatrix(rows, _SP_RATINGS)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _verification_checks():
    """Return a list of (name, value, reference, tolerance) tuples."""
    checks = []

    # -- Merton: the accounting identity and the option it is built on -------
    m = merton(assets=120.0, debt=100.0, T=2.0, r=0.03, sigma=0.25)
    checks.append(("Merton: equity + debt value = assets", m.equity + m.debt_value, 120.0, 1e-12))
    checks.append(
        (
            "Merton: equity = Black-Scholes call on the assets",
            m.equity,
            black_scholes(S=120.0, K=100.0, T=2.0, r=0.03, sigma=0.25, kind="call"),
            1e-14,
        )
    )
    solvent = merton(assets=120.0, debt=100.0, T=2.0, r=0.03, sigma=0.0)
    checks.append(("Merton: credit spread = 0 at zero asset volatility", solvent.credit_spread, 0.0, 1e-14))

    # -- ASRF / Vasicek ------------------------------------------------------
    checks.append(
        ("ASRF: conditional PD at rho = 0 is the unconditional PD",
         asrf_conditional_pd(0.02, 0.0, 0.999), 0.02, 1e-15)
    )
    checks.append(
        ("Vasicek loss quantile = ASRF conditional PD",
         vasicek_loss_quantile(0.999, 0.02, 0.18),
         asrf_conditional_pd(0.02, 0.18, 0.999), 1e-15)
    )

    # -- CDS: the credit triangle --------------------------------------------
    # With r = 0 the discounting cancels and the only error left is the
    # trapezium rule on the risky annuity, which is O(dt^2 h^2).
    hazard, recovery = 0.01, 0.4
    checks.append(
        ("CDS par spread = credit triangle h(1-R)",
         cds_par_spread(hazard, recovery, r=0.0, maturity=5.0, frequency=12),
         spread_from_hazard(hazard, recovery), 1e-8)
    )

    # -- Basel ---------------------------------------------------------------
    checks.append(
        ("Basel corporate correlation -> 0.12 as PD -> 1", basel_correlation(1.0), 0.12, 1e-15)
    )
    # BCBS (2005), Explanatory Note, benchmark exposure: PD 1%, LGD 45%, M 2.5
    # carries a 92.32% risk weight.
    benchmark = basel_irb_capital(pd=0.01, lgd=0.45, ead=1.0, maturity=2.5)
    checks.append(
        ("Basel IRB risk weight, PD 1% / LGD 45% / M 2.5", benchmark.risk_weight, 0.9232, 1e-3)
    )
    return checks
