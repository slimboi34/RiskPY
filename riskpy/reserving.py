"""P&C claims reserving — run-off triangles and the methods that complete them.

A run-off triangle records, for each origin period (accident or underwriting
year), how the claims from that period have paid out over successive
development periods. The bottom-right half is the future, and estimating it is
the reserving problem. This module is the full toolkit that sits above the
compiled ``riskpy.LossTriangle`` (which does volume-weighted chain ladder, no
tail, no uncertainty):

* :class:`Triangle` — the data structure, from a NaN-padded array or a ragged
  list of lists, cumulative or incremental.
* :func:`chain_ladder` — the deterministic point estimate, with a choice of
  factor averaging, a tail and a look-back window.
* :func:`mack_chain_ladder` — the same point estimate plus Mack's
  distribution-free standard errors, split into process and parameter risk.
* :func:`bornhuetter_ferguson`, :func:`cape_cod`, :func:`expected_claims` —
  the expected-loss-ratio family, for when the chain ladder is too responsive
  to a thin latest diagonal.
* :func:`bootstrap_chain_ladder` — England & Verrall's over-dispersed Poisson
  bootstrap, giving a full predictive distribution of the reserve that plugs
  straight into :mod:`riskpy.viz`.
* :func:`fit_tail`, :func:`cdf_from_factors`, :func:`development_pattern` —
  the small pieces you need around them.

Conventions used throughout: rows are origins (oldest first), columns are
development periods, and development period ``k = 0`` is the origin period
itself. ``factors[k]`` takes column ``k`` to ``k + 1``; ``cdf[k]`` takes column
``k`` to ultimate (so ``cdf[-1]`` is the tail). A *reserve* is ultimate minus
the latest diagonal — IBNR plus IBNER for a paid triangle, pure IBNR for an
incurred one. "Level" means a probability such as ``0.995``, as in
:meth:`riskpy.mc.Result.var`.

This is the **NumPy layer**: every function imports NumPy lazily through
:func:`riskpy.mc._numpy` (``pip install open-riskpy[sim]``). Nothing here plots;
results carry the arrays a chart needs (``full_triangle``, ``link_ratios()``,
``BootstrapResult.result()``) and :mod:`riskpy.viz` draws them.

    from riskpy import reserving
    tri = reserving.genins()                     # the Taylor–Ashe triangle
    print(reserving.chain_ladder(tri).summary())
    mack = reserving.mack_chain_ladder(tri)
    mack.reserve_percentile(0.995)               # lognormal-matched percentile
    boot = reserving.bootstrap_chain_ladder(tri, n=5000, seed=1)
    boot.result().var(0.995)                     # a riskpy.mc.Result

References: Mack (1993) *Distribution-free calculation of the standard error of
chain ladder reserve estimates*, ASTIN Bulletin 23(2); Mack (1999) *The standard
error of chain ladder reserve estimates: recursive calculation and inclusion of
a tail factor*, ASTIN Bulletin 29(2); England & Verrall (2002) *Stochastic
claims reserving in general insurance*, British Actuarial Journal 8(3);
Bornhuetter & Ferguson (1972) *The actuary and IBNR*, PCAS LIX; Stanard (1985)
and Bühlmann (1983) for Cape Cod; Sherman (1984) *Extrapolating, smoothing and
interpolating development factors*, PCAS LXXI, for the inverse-power tail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import _special
from .mc import _numpy
from .quant import _normal_quantile

__all__ = [
    "Triangle",
    "chain_ladder",
    "ChainLadderResult",
    "mack_chain_ladder",
    "MackResult",
    "bornhuetter_ferguson",
    "cape_cod",
    "expected_claims",
    "ReserveResult",
    "bootstrap_chain_ladder",
    "BootstrapResult",
    "fit_tail",
    "cdf_from_factors",
    "development_pattern",
    "genins",
]


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _check_level(level: float) -> float:
    if not (isinstance(level, (int, float)) and 0.0 < level < 1.0):
        raise ValueError(
            f"level must be a probability in (0, 1), e.g. 0.995 — got {level!r}"
        )
    return float(level)


def _check_tail(tail: float) -> float:
    if not isinstance(tail, (int, float)) or not math.isfinite(tail) or tail <= 0.0:
        raise ValueError(
            f"tail must be a finite positive factor (1.0 means no tail), got {tail!r}. "
            f"Use fit_tail() to estimate one from the development factors."
        )
    return float(tail)


def _as_triangle(triangle: Any) -> "Triangle":
    return triangle if isinstance(triangle, Triangle) else Triangle(triangle)


def _safe_ratio(numerator, denominator):
    """``numerator / denominator`` with NaN, not a warning, where the denominator is 0."""
    np = _numpy()
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.full(np.broadcast(num, den).shape, np.nan)
    ok = den != 0.0
    np.divide(num, den, out=out, where=ok)
    return out


def _number_spec(values) -> str:
    """Pick a format for a block of money-like numbers from their magnitude."""
    np = _numpy()
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return ",.2f"
    top = float(np.abs(finite).max())
    if top >= 10_000:
        return ",.0f"
    if top >= 1:
        return ",.2f"
    return ".4f"


def _fmt(value, spec: str) -> str:
    if value is None:
        return ""
    value = float(value)
    if math.isnan(value):
        return ""
    return format(value, spec)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]],
           footer: Optional[Sequence[Sequence[str]]] = None) -> str:
    """Monospace table: first column left-aligned, the rest right-aligned."""
    everything = list(rows) + list(footer or [])
    widths = [
        max([len(str(h))] + [len(r[j]) for r in everything])
        for j, h in enumerate(headers)
    ]

    def line(cells: Sequence[str]) -> str:
        first = str(cells[0]).ljust(widths[0])
        rest = [str(c).rjust(w) for c, w in zip(cells[1:], widths[1:])]
        return "  ".join([first] + rest)

    out = [line(headers)] + [line(r) for r in rows]
    if footer:
        out.append("-" * len(out[0]))
        out.extend(line(r) for r in footer)
    return "\n".join(out)


def _pad_rows(data: Any, np) -> Any:
    """Turn a 2-D array-like or a ragged list of rows into a NaN-padded float array."""
    if isinstance(data, Triangle):
        return data.cumulative
    if hasattr(data, "ndim"):  # ndarray, pandas DataFrame, …
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 2:
            raise ValueError(
                f"data must be 2-D (origins × development periods), got {arr.ndim}-D"
            )
        return arr
    try:
        rows = [list(r) for r in data]
    except TypeError as exc:
        raise ValueError(
            "data must be a 2-D array (NaN for future cells) or a list of rows, "
            "one per origin, each row as long as its history"
        ) from exc
    if not rows:
        raise ValueError("data is empty — a triangle needs at least one origin")
    width = max(len(r) for r in rows)
    out = np.full((len(rows), width), np.nan)
    for i, row in enumerate(rows):
        try:
            out[i, : len(row)] = np.asarray(row, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"origin row {i} contains non-numeric values: {row!r}") from exc
    return out


# ---------------------------------------------------------------------------
# Triangle
# ---------------------------------------------------------------------------


class Triangle:
    """A run-off triangle: rows are origin periods, columns development periods.

    ``data`` is either a 2-D array with ``NaN`` in the cells that have not
    happened yet, or a ragged list of lists — each row as long as its own
    history, oldest origin first. These two are the same triangle::

        Triangle([[100, 150, 180], [120, 170], [140]])
        Triangle([[100, 150, 180], [120, 170, nan], [140, nan, nan]])

    Convention: ``cumulative=True`` (the default) means the cells are
    cumulative paid or incurred amounts. Pass ``cumulative=False`` — or use
    :meth:`from_incremental` — for incremental amounts, which are summed along
    each row. Development period ``k = 0`` is the origin period itself, so the
    ``factors[k]`` of every result below takes column ``k`` to ``k + 1``.
    ``origin`` labels the rows (accident years, say) and defaults to
    ``0, 1, 2, …``; the ``origin=`` argument of the result methods refers to
    these labels, not to positions.

    What the constructor refuses, and why: infinities; a NaN *inside* a row's
    history (a run-off has no holes — that is a data problem, not a future
    cell); a column no origin has reached; negative cumulatives; and, unless
    ``allow_decreasing=True``, a row that goes down. Paid triangles never
    decrease. Incurred ones can, when case reserves are released, and the flag
    exists for them — but the decreasing-row check is on by default because
    the commonest mistake in reserving is feeding incremental data in as
    cumulative, and this check catches it almost every time.
    """

    def __init__(
        self,
        data: Any,
        origin: Optional[Sequence[Any]] = None,
        cumulative: bool = True,
        allow_decreasing: bool = False,
    ):
        np = _numpy()
        values = _pad_rows(data, np)
        n_origin, n_dev = values.shape
        if n_origin == 0 or n_dev == 0:
            raise ValueError("data is empty — a triangle needs at least one origin and one column")
        if np.isinf(values).any():
            raise ValueError("triangle contains infinite values; only finite numbers or NaN are allowed")

        observed = ~np.isnan(values)
        counts = observed.sum(axis=1)
        for i in range(n_origin):
            if counts[i] == 0:
                raise ValueError(
                    f"origin row {i} has no observations at all — drop it, or give it "
                    f"at least its first development period"
                )
            if not observed[i, : counts[i]].all():
                first_gap = int(np.argmin(observed[i]))
                raise ValueError(
                    f"origin row {i} has a NaN at development period {first_gap} followed "
                    f"by data — a run-off has no holes. Fill the missing cell or drop "
                    f"the row; NaN is only for periods that have not happened yet."
                )
        empty_columns = np.flatnonzero(~observed.any(axis=0))
        if empty_columns.size:
            raise ValueError(
                f"development period {int(empty_columns[0])} has no observations in any "
                f"origin, so nothing can be estimated for it — drop the column"
            )

        if not cumulative:
            summed = np.cumsum(np.where(observed, values, 0.0), axis=1)
            values = np.where(observed, summed, np.nan)

        negatives = observed & (values < 0.0)
        if negatives.any():
            i, k = (int(x) for x in np.argwhere(negatives)[0])
            raise ValueError(
                f"cumulative value at origin row {i}, development period {k} is negative "
                f"({values[i, k]!r}); cumulative claims cannot be below zero. If these are "
                f"incremental amounts pass cumulative=False."
            )
        if not allow_decreasing:
            steps = values[:, 1:] - values[:, :-1]
            decreasing = np.isfinite(steps) & (steps < 0.0)
            if decreasing.any():
                i, k = (int(x) for x in np.argwhere(decreasing)[0])
                raise ValueError(
                    f"cumulative values decrease at origin row {i} between development "
                    f"periods {k} and {k + 1} ({values[i, k]:g} -> {values[i, k + 1]:g}). "
                    f"If these are incremental amounts pass cumulative=False; if this is an "
                    f"incurred triangle with genuine downward development pass "
                    f"allow_decreasing=True."
                )

        if origin is None:
            labels: List[Any] = list(range(n_origin))
        else:
            labels = list(origin)
            if len(labels) != n_origin:
                raise ValueError(
                    f"origin has {len(labels)} labels but the triangle has {n_origin} rows"
                )
            if len(set(labels)) != len(labels):
                raise ValueError("origin labels must be unique")

        self._values = values
        self._observed = observed
        self._latest_index = (counts - 1).astype(int)
        self._origin = labels

    @classmethod
    def from_incremental(
        cls,
        data: Any,
        origin: Optional[Sequence[Any]] = None,
        allow_decreasing: bool = False,
    ) -> "Triangle":
        """Build from incremental amounts (summed along each row)."""
        return cls(data, origin=origin, cumulative=False, allow_decreasing=allow_decreasing)

    # -- shape ------------------------------------------------------------------

    @property
    def n_origin(self) -> int:
        return int(self._values.shape[0])

    @property
    def n_dev(self) -> int:
        return int(self._values.shape[1])

    @property
    def origin(self) -> List[Any]:
        return list(self._origin)

    # -- data ---------------------------------------------------------------------

    @property
    def cumulative(self):
        """``(n_origin, n_dev)`` cumulative array, NaN in the future cells. A copy."""
        return self._values.copy()

    @property
    def incremental(self):
        """``(n_origin, n_dev)`` incremental array, NaN in the future cells."""
        inc = self._values.copy()
        inc[:, 1:] = self._values[:, 1:] - self._values[:, :-1]
        return inc

    @property
    def observed(self):
        """Boolean ``(n_origin, n_dev)`` mask of the cells that have happened."""
        return self._observed.copy()

    @property
    def latest_index(self):
        """Column index of each origin's latest observation, length ``n_origin``."""
        return self._latest_index.copy()

    @property
    def latest_diagonal(self):
        """The most recent cumulative value of each origin, length ``n_origin``."""
        np = _numpy()
        return self._values[np.arange(self.n_origin), self._latest_index].copy()

    def link_ratios(self):
        """Individual age-to-age factors ``C[i, k+1] / C[i, k]``, shape ``(n_origin, n_dev - 1)``.

        NaN wherever either cell is unknown or the denominator is zero. This is
        the raw material of every chain-ladder average and the natural input to
        a heat map: an outlying ratio in a single cell is the first thing a
        reviewer wants to see.
        """
        np = _numpy()
        both = self._observed[:, :-1] & self._observed[:, 1:]
        ratios = _safe_ratio(self._values[:, 1:], self._values[:, :-1])
        return np.where(both, ratios, np.nan)

    def to_frame(self):
        """The cumulative triangle as a pandas DataFrame (origin index, dev columns)."""
        try:
            import pandas  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("to_frame() needs pandas: pip install pandas") from exc
        frame = pandas.DataFrame(self._values, index=self._origin, columns=list(range(self.n_dev)))
        frame.index.name = "origin"
        frame.columns.name = "dev"
        return frame

    def __repr__(self) -> str:
        spec = _number_spec(self._values)
        headers = ["origin"] + [str(k) for k in range(self.n_dev)]
        rows = [
            [str(label)] + [_fmt(v, spec) for v in self._values[i]]
            for i, label in enumerate(self._origin)
        ]
        title = f"Triangle: {self.n_origin} origins x {self.n_dev} development periods (cumulative)"
        return title + "\n" + _table(headers, rows)


# ---------------------------------------------------------------------------
# Factors, tails and patterns
# ---------------------------------------------------------------------------


def cdf_from_factors(factors: Sequence[float], tail: float = 1.0):
    """Cumulative-to-ultimate factors from age-to-age factors, length ``len(factors) + 1``.

    ``cdf[k] = f[k] · f[k+1] · … · f[-1] · tail``, so ``cdf[-1] == tail`` and
    ``latest · cdf[latest_index]`` is the ultimate. The extra element is the
    thing people trip on: a triangle with ``n_dev`` columns has ``n_dev - 1``
    factors and ``n_dev`` cumulative factors.
    """
    np = _numpy()
    tail = _check_tail(tail)
    f = np.asarray(list(factors), dtype=float)
    if f.ndim != 1:
        raise ValueError("factors must be a 1-D sequence of age-to-age factors")
    if not np.all(np.isfinite(f)) or (f <= 0).any():
        raise ValueError(f"factors must be finite and positive, got {f!r}")
    out = np.empty(f.size + 1)
    out[-1] = tail
    for k in range(f.size - 1, -1, -1):
        out[k] = f[k] * out[k + 1]
    return out


def development_pattern(cdf: Sequence[float]):
    """Percentage developed at each column: ``1 / cdf``.

    ``development_pattern(result.cdf)`` is the payment pattern implied by a
    chain ladder — the x-axis-to-y-axis data of a development curve, and the
    exposure weights that Bornhuetter–Ferguson and Cape Cod use.
    """
    np = _numpy()
    c = np.asarray(list(cdf), dtype=float)
    if c.ndim != 1 or not np.all(np.isfinite(c)) or (c <= 0).any():
        raise ValueError(f"cdf must be a 1-D sequence of finite positive factors, got {c!r}")
    return 1.0 / c


def fit_tail(
    factors: Sequence[float],
    method: str = "exponential",
    n_extrapolate: int = 50,
) -> float:
    """A tail factor from the trend in the age-to-age factors.

    Fits ``ln(f_k - 1)`` against the development age ``k = 1, 2, …`` (the
    first factor is age 1) and multiplies up the extrapolated factors for the
    next ``n_extrapolate`` periods. ``method="exponential"`` regresses on
    ``k`` (McClenahan's exponential decay); ``"inverse_power"`` regresses on
    ``ln k`` (Sherman 1984), which decays more slowly and gives a heavier
    tail — the usual choice for liability lines whose factors refuse to reach
    one. Only factors above 1 enter the regression, because the log of a
    non-positive excess does not exist.

    The one thing to know: the exponential product converges, so 50 periods is
    effectively infinity; the inverse-power product does **not** converge for
    a fitted exponent above -1, so its result depends on ``n_extrapolate`` and
    you should say which horizon you used. Both methods raise if the factors
    are not decaying, because a tail from a rising trend is not a number
    anyone should book.
    """
    np = _numpy()
    method = method.lower()
    if method not in ("exponential", "inverse_power"):
        raise ValueError(f"method must be 'exponential' or 'inverse_power', got {method!r}")
    if not isinstance(n_extrapolate, int) or n_extrapolate < 1:
        raise ValueError(f"n_extrapolate must be a positive integer, got {n_extrapolate!r}")
    f = np.asarray(list(factors), dtype=float)
    if f.ndim != 1 or not np.all(np.isfinite(f)):
        raise ValueError("factors must be a 1-D sequence of finite numbers")

    ages = np.arange(1, f.size + 1, dtype=float)
    use = f > 1.0
    if use.sum() < 2:
        raise ValueError(
            "fit_tail needs at least two factors above 1 to see a trend; "
            f"got {int(use.sum())}. If development is genuinely complete use tail=1.0."
        )
    x = ages[use] if method == "exponential" else np.log(ages[use])
    y = np.log(f[use] - 1.0)
    x_mean, y_mean = x.mean(), y.mean()
    slope = float(((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum())
    intercept = float(y_mean - slope * x_mean)
    if slope >= 0.0:
        raise ValueError(
            f"the factors are not decaying (fitted slope {slope:+.4f}), so an extrapolated "
            f"tail would grow without bound. Check the later factors, or set the tail by judgement."
        )
    future = np.arange(f.size + 1, f.size + 1 + n_extrapolate, dtype=float)
    xf = future if method == "exponential" else np.log(future)
    return float(np.prod(1.0 + np.exp(intercept + slope * xf)))


def _development_factors(tri: Triangle, average: str, n_periods: Optional[int], np):
    """Age-to-age factors under the chosen averaging rule. Returns ``(factors, weights)``.

    ``weights[k]`` is the volume ``Σ C[i, k]`` behind factor ``k`` (Mack needs it).
    """
    C, obs = tri._values, tri._observed
    n_dev = tri.n_dev
    factors = np.empty(max(n_dev - 1, 0))
    volume = np.empty(max(n_dev - 1, 0))
    for k in range(n_dev - 1):
        rows = np.flatnonzero(obs[:, k] & obs[:, k + 1])
        if rows.size == 0:
            raise ValueError(
                f"no origin has both development periods {k} and {k + 1}, so factor {k} "
                f"cannot be estimated — drop the last column or supply the factors yourself"
            )
        if n_periods is not None:
            rows = rows[-n_periods:]
        current, following = C[rows, k], C[rows, k + 1]
        volume[k] = current.sum()
        if average == "volume":
            if volume[k] <= 0.0:
                raise ValueError(
                    f"development period {k} sums to zero over the origins used, so the "
                    f"volume-weighted factor {k} is undefined"
                )
            factors[k] = following.sum() / volume[k]
        else:
            if (current <= 0.0).any():
                raise ValueError(
                    f"development period {k} contains a zero, so the individual link ratios "
                    f"are undefined; use average='volume', which only needs the column total"
                )
            if average == "simple":
                factors[k] = (following / current).mean()
            else:  # regression through the origin, equal weights
                factors[k] = (current * following).sum() / (current * current).sum()
    return factors, volume


def _complete_square(C, obs, factors, np):
    """Fill the future cells column by column: ``C[i, k] = C[i, k-1] · f[k-1]``."""
    full = C.copy()
    for k in range(1, C.shape[1]):
        future = ~obs[:, k]
        full[future, k] = full[future, k - 1] * factors[k - 1]
    return full


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


class _ReserveTable:
    """What every reserve result shares: totals, percentages, a table, a DataFrame.

    Subclasses are dataclasses holding at least ``ultimate``, ``latest``,
    ``reserve``, ``origin`` and ``method``; they extend ``_columns`` /
    ``_totals`` / ``_notes`` to add their own columns to the summary.
    """

    @property
    def total_reserve(self) -> float:
        return float(self.reserve.sum())

    @property
    def total_ultimate(self) -> float:
        return float(self.ultimate.sum())

    @property
    def total_latest(self) -> float:
        return float(self.latest.sum())

    @property
    def percent_developed(self):
        """``latest / ultimate`` per origin — how much of each year is already in the books."""
        return _safe_ratio(self.latest, self.ultimate)

    def _money_spec(self) -> str:
        return _number_spec(self.ultimate)

    def _columns(self) -> List[Tuple[str, Any, str]]:
        money = self._money_spec()
        return [
            ("latest", self.latest, money),
            ("developed", self.percent_developed, ".1%"),
            ("ultimate", self.ultimate, money),
            ("reserve", self.reserve, money),
        ]

    def _totals(self) -> List[Tuple[Optional[float], str]]:
        money = self._money_spec()
        latest, ultimate = self.total_latest, self.total_ultimate
        return [
            (latest, money),
            (latest / ultimate if ultimate else None, ".1%"),
            (ultimate, money),
            (self.total_reserve, money),
        ]

    def _title(self) -> str:
        method = self.method
        return f"{method[0].upper()}{method[1:]} — {len(self.origin)} origins"

    def _notes(self) -> List[str]:
        return []

    def summary(self) -> str:
        """A table with one row per origin and a totals line, ready to print."""
        cols = self._columns()
        headers = ["origin"] + [name for name, _, _ in cols]
        rows = [
            [str(label)] + [_fmt(values[i], spec) for _, values, spec in cols]
            for i, label in enumerate(self.origin)
        ]
        footer = [["total"] + [_fmt(value, spec) for value, spec in self._totals()]]
        text = self._title() + "\n" + _table(headers, rows, footer)
        notes = self._notes()
        return text + ("\n" + "\n".join(notes) if notes else "")

    def to_frame(self):
        """One row per origin as a pandas DataFrame, if pandas is around."""
        try:
            import pandas  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("to_frame() needs pandas: pip install pandas") from exc
        data = {name: values for name, values, _ in self._columns()}
        frame = pandas.DataFrame(data, index=list(self.origin))
        frame.index.name = "origin"
        return frame

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} {self.method}: {len(self.origin)} origins, "
            f"total reserve {self.total_reserve:,.2f}>"
        )


@dataclass(repr=False)
class ReserveResult(_ReserveTable):
    """Reserve estimates from an expected-loss-ratio method.

    ``method`` names the method; ``extra`` carries whatever it derived on the
    way (Cape Cod puts its implied ``expected_loss_ratio`` there, as one value
    per origin). Any per-origin array in ``extra`` is shown as a column of
    :meth:`summary`; scalars are printed below the table.
    """

    ultimate: Any
    latest: Any
    reserve: Any
    origin: List[Any]
    method: str
    extra: Dict[str, Any]

    def _columns(self):
        cols = super()._columns()
        n = len(self.origin)
        for name, value in self.extra.items():
            if hasattr(value, "shape") and getattr(value, "shape", None) == (n,):
                cols.append((name, value, _number_spec(value)))
        return cols

    def _totals(self):
        totals = super()._totals()
        n = len(self.origin)
        for value in self.extra.values():
            if hasattr(value, "shape") and getattr(value, "shape", None) == (n,):
                totals.append((None, ""))
        return totals

    def _notes(self):
        return [
            f"{name} {value:.4f}" if isinstance(value, float) else f"{name} {value!r}"
            for name, value in self.extra.items()
            if not hasattr(value, "shape")
        ]


@dataclass(repr=False)
class ChainLadderResult(_ReserveTable):
    """The chain ladder's output, with the arrays a development chart needs.

    ``factors`` are the age-to-age factors ``f_k`` (length ``n_dev - 1``),
    ``cdf`` the cumulative factors to ultimate (length ``n_dev``, last one the
    tail). ``full_triangle`` is the observed triangle with every future cell
    filled in by the factors — the tail is *not* in it, so
    ``ultimate == full_triangle[:, -1] * tail``; plot its rows against the
    development index and the original cells against the same axis for the
    classic "actual vs projected" fan. ``latest`` and ``reserve`` are one per
    origin, ``reserve = ultimate - latest``.
    """

    factors: Any
    tail: float
    cdf: Any
    ultimate: Any
    latest: Any
    reserve: Any
    full_triangle: Any
    origin: List[Any]
    method: str

    def _columns(self):
        money = self._money_spec()
        return [
            ("latest", self.latest, money),
            ("cdf", _safe_ratio(self.ultimate, self.latest), ".4f"),
            ("developed", self.percent_developed, ".1%"),
            ("ultimate", self.ultimate, money),
            ("reserve", self.reserve, money),
        ]

    def _totals(self):
        money = self._money_spec()
        latest, ultimate = self.total_latest, self.total_ultimate
        return [
            (latest, money),
            (ultimate / latest if latest else None, ".4f"),
            (latest / ultimate if ultimate else None, ".1%"),
            (ultimate, money),
            (self.total_reserve, money),
        ]

    def _title(self):
        return f"{super()._title()}, tail {self.tail:.4f}"

    def _notes(self):
        return ["factors  " + "  ".join(f"{f:.4f}" for f in self.factors)]


@dataclass(repr=False)
class MackResult(ChainLadderResult):
    """Chain ladder plus Mack's standard errors.

    ``sigma`` holds Mack's **variance** parameters ``σ_k²`` (one per
    age-to-age step, the last extrapolated) — the name follows the usual
    notation, so take a square root before comparing with R's ``sigma``.
    ``process_se``, ``parameter_se`` and ``se`` are per origin, with
    ``se² = process_se² + parameter_se²``; ``total_se`` is the standard error
    of the *total* reserve, which is more than the root-sum-square of the
    origins because they share the same estimated factors (``total_se² =
    total_process_se² + total_parameter_se²``). ``cv`` is ``se / reserve``.

    Process error is what remains if the factors were known exactly;
    parameter error is the price of having estimated them from the same
    triangle. Old origins are dominated by the former, young ones by the latter.
    """

    sigma: Any
    process_se: Any
    parameter_se: Any
    se: Any
    total_se: float
    total_process_se: float
    total_parameter_se: float
    tail_sigma: float
    tail_se: float

    @property
    def cv(self):
        """Coefficient of variation ``se / reserve`` per origin (NaN where the reserve is 0)."""
        return _safe_ratio(self.se, self.reserve)

    @property
    def total_cv(self) -> float:
        total = self.total_reserve
        return self.total_se / total if total else float("nan")

    def reserve_percentile(self, level: float, origin: Optional[Any] = None) -> float:
        """The reserve at probability ``level``, e.g. ``0.75`` or ``0.995``.

        Mack gives a mean and a standard error but no distribution. The
        percentile here comes from a **lognormal matched to those two moments**
        — the conventional bridge from Mack to a risk margin, chosen over the
        normal because reserves cannot go negative and their distribution is
        right-skewed. It is an approximation: it says nothing about the true
        shape of the tail, and for a young origin with a CV above about 0.5 it
        can be markedly heavier than a bootstrap would suggest. Use
        :func:`bootstrap_chain_ladder` when the shape matters.

        ``origin=None`` gives the total reserve; otherwise pass an origin label.
        """
        level = _check_level(level)
        if origin is None:
            mean, se = self.total_reserve, self.total_se
        else:
            idx = _origin_index(self.origin, origin)
            mean, se = float(self.reserve[idx]), float(self.se[idx])
        if se == 0.0:
            return mean
        if mean <= 0.0:
            raise ValueError(
                f"the lognormal approximation needs a positive reserve; origin "
                f"{origin!r} has reserve {mean:,.2f} with s.e. {se:,.2f}"
            )
        sigma_sq = math.log(1.0 + (se / mean) ** 2)
        mu = math.log(mean) - 0.5 * sigma_sq
        return math.exp(mu + math.sqrt(sigma_sq) * _normal_quantile(level))

    def _columns(self):
        cols = super()._columns()
        money = self._money_spec()
        cols.append(("se", self.se, money))
        cols.append(("cv", self.cv, ".3f"))
        return cols

    def _totals(self):
        return super()._totals() + [(self.total_se, self._money_spec()), (self.total_cv, ".3f")]

    def _notes(self):
        np = _numpy()
        notes = super()._notes()
        notes.append("sigma    " + "  ".join(f"{s:.2f}" for s in np.sqrt(self.sigma)))
        notes.append(
            f"total s.e. {self.total_se:,.0f} = process {self.total_process_se:,.0f} "
            f"+ parameter {self.total_parameter_se:,.0f} (in quadrature)"
        )
        return notes


def _origin_index(labels: Sequence[Any], origin: Any) -> int:
    try:
        return list(labels).index(origin)
    except ValueError:
        raise ValueError(
            f"origin {origin!r} is not one of the triangle's origin labels {list(labels)!r}"
        ) from None


# ---------------------------------------------------------------------------
# Chain ladder
# ---------------------------------------------------------------------------


def chain_ladder(
    triangle: Any,
    tail: float = 1.0,
    average: str = "volume",
    n_periods: Optional[int] = None,
) -> ChainLadderResult:
    """The chain ladder: project each origin forward with averaged age-to-age factors.

    ``average`` chooses how the individual link ratios are combined into one
    factor per column:

    * ``"volume"`` — ``Σ C[i, k+1] / Σ C[i, k]``, the weighted average with
      weights ``C[i, k]``. The default, the C++ ``LossTriangle``'s method and
      the one Mack's standard errors assume. Big origins count more, which is
      right when a big origin's ratio is better information.
    * ``"simple"`` — the plain mean of the link ratios. Every origin counts the
      same; more sensitive to a small origin with a wild ratio.
    * ``"regression"`` — least squares of ``C[i, k+1]`` on ``C[i, k]`` through
      the origin, ``Σ C[i,k]·C[i,k+1] / Σ C[i,k]²``. Weights the big origins
      even harder than volume does.

    ``n_periods`` restricts each factor to the most recent ``n`` link ratios
    (the last ``n`` calendar diagonals for a regular triangle) — the standard
    response to a change in settlement speed. ``tail`` multiplies every
    ultimate for development beyond the triangle's last column; see
    :func:`fit_tail`.

    The thing to remember: the chain ladder multiplies the latest diagonal,
    so a young origin with a small or unusual first cell gets that oddity
    magnified by the whole product of factors. That is the case for
    :func:`bornhuetter_ferguson`.
    """
    np = _numpy()
    tri = _as_triangle(triangle)
    tail = _check_tail(tail)
    average = average.lower() if isinstance(average, str) else average
    if average not in ("volume", "simple", "regression"):
        raise ValueError(
            f"average must be 'volume', 'simple' or 'regression', got {average!r}"
        )
    if n_periods is not None and (not isinstance(n_periods, int) or n_periods < 1):
        raise ValueError(f"n_periods must be a positive integer or None, got {n_periods!r}")

    factors, _ = _development_factors(tri, average, n_periods, np)
    cdf = cdf_from_factors(factors, tail)
    full = _complete_square(tri._values, tri._observed, factors, np)
    latest = tri.latest_diagonal
    ultimate = full[:, -1] * tail
    return ChainLadderResult(
        factors=factors,
        tail=tail,
        cdf=cdf,
        ultimate=ultimate,
        latest=latest,
        reserve=ultimate - latest,
        full_triangle=full,
        origin=tri.origin,
        method=f"chain ladder ({average})",
    )


# ---------------------------------------------------------------------------
# Mack
# ---------------------------------------------------------------------------


def _loglinear_extrapolate(values, at: float, np) -> float:
    """Fit ``ln(values)`` on the index and evaluate at ``at``. Needs two positive points."""
    y_all = np.asarray(values, dtype=float)
    x_all = np.arange(y_all.size, dtype=float)
    use = np.isfinite(y_all) & (y_all > 0.0)
    if use.sum() < 2:
        return float("nan")
    x, y = x_all[use], np.log(y_all[use])
    x_mean, y_mean = x.mean(), y.mean()
    slope = float(((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum())
    return float(math.exp(y_mean + slope * (at - x_mean)))


def mack_chain_ladder(
    triangle: Any,
    tail: float = 1.0,
    tail_sigma: Optional[float] = None,
    tail_se: Optional[float] = None,
) -> MackResult:
    """Chain ladder with Mack's (1993) distribution-free standard errors.

    Always volume-weighted: that is the model whose assumptions Mack's
    formulae rest on (``E[C_{k+1} | C_k] = f_k C_k`` and
    ``Var[C_{k+1} | C_k] = σ_k² C_k``). The variance parameters are

        σ_k² = 1/(n_k - 1) · Σ_i C[i,k] (C[i,k+1]/C[i,k] - f_k)²

    over the ``n_k`` origins that have both columns, and the last one — which
    only one origin sees — is extrapolated by Mack's rule
    ``σ²_{K} = min(σ⁴_{K-1}/σ²_{K-2}, min(σ²_{K-2}, σ²_{K-1}))``: a log-linear
    step down, floored so it can never exceed its predecessors. Standard errors
    then follow Mack's equation (3) for each origin and his total-reserve
    formula, which adds the covariance between origins that share factors.

    With a ``tail`` above 1 the tail is treated as one more factor with its
    own ``σ²`` and standard error (Mack 1999). Unless you pass ``tail_sigma``
    (a σ², same units as ``result.sigma``) and ``tail_se`` (a standard error of
    the tail factor) they are extrapolated by log-linear regression of the
    fitted ``σ_k`` and ``s.e.(f_k)`` on ``k``, which is R's ChainLadder
    convention and a heuristic: judge it, do not just accept it.

    The one thing people get wrong is reading ``se`` as a confidence interval.
    It is a root mean square prediction error — it contains the process
    variance of the future claims as well as the estimation error, and it
    is only a standard deviation, with no distribution attached. See
    :meth:`MackResult.reserve_percentile` for the usual way round that.
    """
    np = _numpy()
    tri = _as_triangle(triangle)
    tail = _check_tail(tail)
    if tri.n_dev < 2:
        raise ValueError("Mack's method needs at least two development periods")
    if tail_sigma is not None and (not math.isfinite(tail_sigma) or tail_sigma < 0):
        raise ValueError(f"tail_sigma must be a finite non-negative variance, got {tail_sigma!r}")
    if tail_se is not None and (not math.isfinite(tail_se) or tail_se < 0):
        raise ValueError(f"tail_se must be a finite non-negative standard error, got {tail_se!r}")

    base = chain_ladder(tri, tail=tail, average="volume")
    C, obs = tri._values, tri._observed
    n_origin, n_dev = tri.n_origin, tri.n_dev
    f = base.factors
    n_steps = n_dev - 1

    # -- σ_k² and the volume S_k behind each factor ---------------------------------
    volume = np.zeros(n_steps)
    sigma_sq = np.full(n_steps, np.nan)
    for k in range(n_steps):
        rows = np.flatnonzero(obs[:, k] & obs[:, k + 1])
        current, following = C[rows, k], C[rows, k + 1]
        volume[k] = current.sum()
        if rows.size >= 2:
            positive = current > 0.0
            ratios = following[positive] / current[positive]
            sigma_sq[k] = float((current[positive] * (ratios - f[k]) ** 2).sum() / (rows.size - 1))
    for k in range(n_steps):
        if np.isnan(sigma_sq[k]):
            if k >= 2 and np.isfinite(sigma_sq[k - 1]) and np.isfinite(sigma_sq[k - 2]):
                a, b = sigma_sq[k - 2], sigma_sq[k - 1]
                sigma_sq[k] = min(b * b / a, min(a, b)) if a > 0.0 else 0.0
            elif k >= 1 and np.isfinite(sigma_sq[k - 1]):
                sigma_sq[k] = sigma_sq[k - 1]
            else:
                raise ValueError(
                    f"σ² for development step {k} cannot be estimated: only one origin has "
                    f"both periods and there is no earlier step to extrapolate from. Mack's "
                    f"method needs at least two origins on the first step."
                )
    se_f_sq = sigma_sq / volume  # (s.e. of f_k)², Mack's σ_k² / Σ_i C[i,k]

    # -- the tail as one more step ------------------------------------------------------
    if tail != 1.0:
        if tail_sigma is None:
            sd = _loglinear_extrapolate(np.sqrt(sigma_sq), float(n_steps), np)
            if math.isnan(sd):
                raise ValueError(
                    "cannot extrapolate σ for the tail (fewer than two positive σ_k); "
                    "pass tail_sigma explicitly"
                )
            tail_sigma = sd * sd
        if tail_se is None:
            tail_se = _loglinear_extrapolate(np.sqrt(se_f_sq), float(n_steps), np)
            if math.isnan(tail_se):
                raise ValueError(
                    "cannot extrapolate the tail factor's standard error (fewer than two "
                    "positive s.e.(f_k)); pass tail_se explicitly"
                )
        f_ext = np.append(f, tail)
        sigma_ext = np.append(sigma_sq, tail_sigma)
        se_f_ext = np.append(se_f_sq, tail_se * tail_se)
    else:
        tail_sigma, tail_se = 0.0, 0.0
        f_ext, sigma_ext, se_f_ext = f, sigma_sq, se_f_sq
    n_ext = f_ext.size

    # -- Mack (1993) eq. (3), per origin ------------------------------------------------
    latest_index = tri._latest_index
    ultimate = base.ultimate
    start = base.full_triangle[:, :n_ext]                 # Ĉ[i,k] at the start of step k
    uses = np.arange(n_ext)[None, :] >= latest_index[:, None]   # origin i uses step k
    process_terms = np.where(uses, _safe_ratio(sigma_ext / f_ext ** 2, start), 0.0)
    process_terms = np.where(np.isnan(process_terms), 0.0, process_terms)
    parameter_terms = np.where(uses, (se_f_ext / f_ext ** 2)[None, :], 0.0)
    process_var = ultimate ** 2 * process_terms.sum(axis=1)
    parameter_var = ultimate ** 2 * parameter_terms.sum(axis=1)
    se = np.sqrt(process_var + parameter_var)

    # -- the total: process variances add, parameter errors are shared ----------------
    # Σ_k (s.e. f_k / f_k)² · (Σ_{i using k} U_i)² — Mack's total formula written so that
    # the covariance between every pair of origins that share a factor is counted once.
    shared_ultimate = (uses * ultimate[:, None]).sum(axis=0)
    total_parameter_var = float(((se_f_ext / f_ext ** 2) * shared_ultimate ** 2).sum())
    total_process_var = float(process_var.sum())

    return MackResult(
        factors=f,
        tail=tail,
        cdf=base.cdf,
        ultimate=ultimate,
        latest=base.latest,
        reserve=base.reserve,
        full_triangle=base.full_triangle,
        origin=tri.origin,
        method="Mack chain ladder",
        sigma=sigma_sq,
        process_se=np.sqrt(process_var),
        parameter_se=np.sqrt(parameter_var),
        se=se,
        total_se=math.sqrt(total_process_var + total_parameter_var),
        total_process_se=math.sqrt(total_process_var),
        total_parameter_se=math.sqrt(total_parameter_var),
        tail_sigma=float(tail_sigma),
        tail_se=float(tail_se),
    )


# ---------------------------------------------------------------------------
# Expected-loss-ratio methods
# ---------------------------------------------------------------------------


def _resolve_cdf(tri: Triangle, factors: Optional[Sequence[float]], tail: float, np):
    """The cumulative-to-ultimate factors an ELR method should use."""
    if factors is None:
        return chain_ladder(tri, tail=tail).cdf
    cdf = np.asarray(list(factors), dtype=float)
    if cdf.ndim != 1 or cdf.size != tri.n_dev:
        hint = (
            " — that looks like age-to-age factors; convert them with "
            "cdf_from_factors(factors, tail)"
            if cdf.ndim == 1 and cdf.size == tri.n_dev - 1
            else ""
        )
        raise ValueError(
            f"factors must be the {tri.n_dev} cumulative-to-ultimate factors "
            f"(a ChainLadderResult.cdf), one per development period; got "
            f"{cdf.size if cdf.ndim == 1 else cdf.shape}{hint}"
        )
    if not np.all(np.isfinite(cdf)) or (cdf <= 0).any():
        raise ValueError(f"factors must be finite and positive, got {cdf!r}")
    return cdf


def _per_origin(name: str, value: Any, n: int, np, non_negative: bool = True):
    """Broadcast a scalar, or check a length-``n`` sequence, of finite numbers."""
    if isinstance(value, (int, float)):
        arr = np.full(n, float(value))
    else:
        arr = np.asarray(list(value), dtype=float)
        if arr.ndim != 1 or arr.size != n:
            raise ValueError(
                f"{name} must be a scalar or one value per origin ({n}), got "
                f"{arr.size if arr.ndim == 1 else arr.shape}"
            )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {arr!r}")
    if non_negative and (arr < 0).any():
        raise ValueError(f"{name} must be non-negative, got {arr!r}")
    return arr


def _elr_result(tri: Triangle, cdf, expected_ultimate, method: str, extra: Dict[str, Any], np):
    """Ultimate = latest + expected ultimate × (share still to come)."""
    unreported = 1.0 - 1.0 / cdf[tri._latest_index]
    latest = tri.latest_diagonal
    reserve = expected_ultimate * unreported
    return ReserveResult(
        ultimate=latest + reserve,
        latest=latest,
        reserve=reserve,
        origin=tri.origin,
        method=method,
        extra=extra,
    )


def bornhuetter_ferguson(
    triangle: Any,
    premiums: Sequence[float],
    expected_loss_ratio: Any,
    factors: Optional[Sequence[float]] = None,
    tail: float = 1.0,
) -> ReserveResult:
    """Bornhuetter–Ferguson (1972): reserve = premium × ELR × (1 − 1/cdf).

    Each origin's expected ultimate is ``premium × expected_loss_ratio``, and
    the reserve is the share of it the development pattern says is still to
    come. The claims paid so far do not enter the reserve at all — that is the
    point. The chain ladder multiplies the latest diagonal, so one unusual
    early cell drives the whole projection; BF trusts the prior instead and
    only uses the triangle for the *pattern*. It is the method of choice for
    the youngest, least-developed origins, and it converges to the chain
    ladder as an origin matures (the unreported share goes to zero).

    ``expected_loss_ratio`` is a decimal (``0.65``), scalar or per origin.
    ``factors`` are the cumulative-to-ultimate factors (``result.cdf``) and
    default to a volume-weighted chain ladder with ``tail``; ``tail`` is
    ignored when ``factors`` is given, since a cdf already contains its tail.

    The thing people get wrong: choosing the ELR *from the triangle* and then
    calling the result independent of it. If the ELR is the chain-ladder
    ultimate over premium, BF simply reproduces the chain ladder.
    """
    np = _numpy()
    tri = _as_triangle(triangle)
    tail = _check_tail(tail)
    premium = _per_origin("premiums", premiums, tri.n_origin, np)
    elr = _per_origin("expected_loss_ratio", expected_loss_ratio, tri.n_origin, np)
    cdf = _resolve_cdf(tri, factors, tail, np)
    return _elr_result(
        tri, cdf, premium * elr, "Bornhuetter–Ferguson",
        {"expected_loss_ratio": elr, "premium": premium}, np,
    )


def cape_cod(
    triangle: Any,
    premiums: Sequence[float],
    factors: Optional[Sequence[float]] = None,
    tail: float = 1.0,
    decay: float = 1.0,
) -> ReserveResult:
    """Cape Cod (Stanard–Bühlmann): Bornhuetter–Ferguson with the ELR taken from the data.

    The expected loss ratio is estimated from the triangle itself as

        ELR = Σ_i latest_i / Σ_i premium_i · (1 / cdf_i)

    — total claims to date over total *used-up* premium, where each origin's
    premium is weighted by the fraction of its development the pattern says
    is complete. That is the honest way to pool: an origin that is 10 %
    developed contributes 10 % of its premium to the denominator. The reserve
    is then the BF reserve at that ELR.

    ``decay`` in ``(0, 1]`` gives the *generalised* Cape Cod (Gluck 1997):
    origin ``i``'s ELR is estimated from every origin ``j`` with weight
    ``decay^|i-j|``, so nearer years count more. ``decay=1`` is the classic
    method with one pooled ELR; ``decay=0.75`` is a common compromise between
    "one ratio for all" and "chain ladder for each". The implied ratios are
    in ``result.extra["expected_loss_ratio"]``, one per origin.

    Use it instead of BF when you have no credible prior ELR — but remember
    the ELR now depends on the triangle, so the result is *not* independent
    of a bad diagonal, only less sensitive to it than the chain ladder.
    """
    np = _numpy()
    tri = _as_triangle(triangle)
    tail = _check_tail(tail)
    if not isinstance(decay, (int, float)) or not (0.0 < decay <= 1.0):
        raise ValueError(f"decay must be in (0, 1] (1 = classic Cape Cod), got {decay!r}")
    premium = _per_origin("premiums", premiums, tri.n_origin, np)
    cdf = _resolve_cdf(tri, factors, tail, np)

    latest = tri.latest_diagonal
    used_up = premium / cdf[tri._latest_index]          # premium × percent developed
    idx = np.arange(tri.n_origin)
    weights = float(decay) ** np.abs(idx[:, None] - idx[None, :])
    denominators = weights @ used_up
    if (denominators <= 0.0).any():
        raise ValueError(
            "Cape Cod needs positive used-up premium (premium × percent developed) "
            "in the origins it pools; at least one weighted total is zero"
        )
    elr = (weights @ latest) / denominators
    return _elr_result(
        tri, cdf, premium * elr, "Cape Cod",
        {"expected_loss_ratio": elr, "premium": premium, "decay": float(decay)}, np,
    )


def expected_claims(
    triangle: Any,
    ultimates_prior: Sequence[float],
    factors: Optional[Sequence[float]] = None,
    tail: float = 1.0,
) -> ReserveResult:
    """The a-priori (expected claims) method: BF with the expected ultimate given directly.

    ``ultimates_prior`` is your prior estimate of each origin's ultimate
    claims, from a pricing model, an exposure rating or last year's reserving
    exercise. The reserve is the share of it still to come under the
    development pattern; premiums never enter. Identical to
    :func:`bornhuetter_ferguson` with ``premium × ELR`` replaced by the prior,
    and provided separately because it is what you actually have when the
    business is not priced on a loss ratio.
    """
    np = _numpy()
    tri = _as_triangle(triangle)
    tail = _check_tail(tail)
    prior = _per_origin("ultimates_prior", ultimates_prior, tri.n_origin, np)
    cdf = _resolve_cdf(tri, factors, tail, np)
    return _elr_result(tri, cdf, prior, "expected claims", {"ultimates_prior": prior}, np)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


@dataclass(repr=False)
class BootstrapResult:
    """The predictive distribution of the reserve from a bootstrap.

    ``reserves`` is ``(n, n_origin)`` — one simulated reserve per origin per
    pseudo-triangle — and ``total`` its row sums. ``mean`` / ``se`` are per
    origin, ``total_mean`` / ``total_se`` for the whole, all sample statistics
    of those arrays. ``cl_reserve`` is the deterministic chain-ladder reserve
    for comparison: the bootstrap mean sits a little above it, because the
    bootstrap includes the estimation bias the point estimate ignores.
    ``phi`` is the fitted ODP scale parameter.
    """

    reserves: Any
    total: Any
    mean: Any
    se: Any
    total_mean: float
    total_se: float
    origin: List[Any]
    latest: Any
    cl_reserve: Any
    n: int
    seed: Optional[int]
    process: str
    phi: float

    def percentile(self, level: float, origin: Optional[Any] = None) -> float:
        """Reserve at probability ``level`` (``0.995`` for the 99.5th), from the sample.

        ``origin=None`` gives the total reserve; otherwise pass an origin label.
        Empirical, so it never exceeds the largest simulated value — run more
        trials before you quote a 99.9th percentile.
        """
        np = _numpy()
        level = _check_level(level)
        values = self.total if origin is None else self.reserves[:, _origin_index(self.origin, origin)]
        return float(np.percentile(values, level * 100.0))

    def result(self):
        """The total reserve as a :class:`riskpy.mc.Result`, for :mod:`riskpy.viz`.

        ``viz.distribution``, ``viz.exceedance`` and ``viz.convergence`` work on
        it unchanged, and so do ``var``, ``tvar`` and ``summary``. ``inputs``
        is empty (a bootstrap has no named inputs), so ``sensitivity`` will
        tell you so.
        """
        from .mc import Result

        return Result(
            values=self.total.copy(), inputs={}, trials=int(self.n),
            seed=self.seed, label="total reserve",
        )

    @property
    def cv(self):
        return _safe_ratio(self.se, self.mean)

    @property
    def total_cv(self) -> float:
        return self.total_se / self.total_mean if self.total_mean else float("nan")

    def summary(self, levels: Sequence[float] = (0.75, 0.95, 0.995)) -> str:
        np = _numpy()
        money = _number_spec(self.latest + self.cl_reserve)
        headers = ["origin", "latest", "CL reserve", "mean", "se", "cv"] + [f"{lv:.1%}" for lv in levels]
        rows = []
        for i, label in enumerate(self.origin):
            cells = [str(label), _fmt(self.latest[i], money), _fmt(self.cl_reserve[i], money),
                     _fmt(self.mean[i], money), _fmt(self.se[i], money), _fmt(self.cv[i], ".3f")]
            cells += [_fmt(np.percentile(self.reserves[:, i], lv * 100.0), money) for lv in levels]
            rows.append(cells)
        footer = [[
            "total", _fmt(self.latest.sum(), money), _fmt(self.cl_reserve.sum(), money),
            _fmt(self.total_mean, money), _fmt(self.total_se, money), _fmt(self.total_cv, ".3f"),
        ] + [_fmt(np.percentile(self.total, lv * 100.0), money) for lv in levels]]
        title = (
            f"Bootstrap chain ladder (ODP, process={self.process}) — {self.n:,} samples"
            + (f", seed {self.seed}" if self.seed is not None else "")
            + f", phi {self.phi:,.2f}"
        )
        return title + "\n" + _table(headers, rows, footer)

    def to_frame(self):
        """One row per origin as a pandas DataFrame, if pandas is around."""
        try:
            import pandas  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("to_frame() needs pandas: pip install pandas") from exc
        frame = pandas.DataFrame(
            {"latest": self.latest, "cl_reserve": self.cl_reserve, "mean": self.mean,
             "se": self.se, "cv": self.cv},
            index=list(self.origin),
        )
        frame.index.name = "origin"
        return frame

    def __repr__(self) -> str:
        return (
            f"<BootstrapResult {self.n:,} samples: total reserve mean "
            f"{self.total_mean:,.0f}, s.e. {self.total_se:,.0f}>"
        )


def bootstrap_chain_ladder(
    triangle: Any,
    n: int = 1000,
    seed: Optional[int] = None,
    process: str = "odp",
    tail: float = 1.0,
) -> BootstrapResult:
    """England & Verrall's (2002) bootstrap of the over-dispersed Poisson chain ladder.

    The chain ladder is the maximum-likelihood fit of an over-dispersed
    Poisson GLM to the *incremental* triangle (log link, origin and
    development effects) — that equivalence is what makes this bootstrap
    cheap, because refitting the GLM to a pseudo-triangle is just re-running
    the chain ladder. The steps, per Appendix 3 of England & Verrall:

    1. Fit the volume-weighted chain ladder; back-fit the incremental
       triangle by dividing the latest diagonal down the factors.
    2. Unscaled Pearson residuals ``r = (q − m) / √m`` on the observed cells,
       scale ``φ = Σr² / (N − p)`` with ``p = n_origin + n_dev − 1``
       parameters, and residuals inflated by ``√(N / (N − p))`` for the
       degrees of freedom. The two corner cells — the first origin's last
       column and the last origin's first — are structurally zero (one
       observation fits one parameter) and are left out of the pool.
    3. ``n`` times: resample the residuals with replacement, form
       pseudo-incrementals ``m + r*·√m``, cumulate, refit the chain ladder,
       project the future incrementals — that is the *parameter* error — and
       then draw each future cell from the process distribution around the
       projection.

    ``process`` chooses that distribution: ``"odp"`` — mean ``m``, variance
    ``φ·m``, drawn from a gamma with those moments (E&V's recommended
    continuous stand-in for a scaled Poisson); ``"gamma"`` — a constant
    coefficient of variation, variance ``φ_γ·m²`` with ``φ_γ`` estimated from
    the same fit's relative residuals (the gamma model of Mack 1991, heavier
    for large cells); ``"none"`` — parameter error only, which is what you
    compare to Mack's ``parameter_se``. A ``tail`` above 1 adds one final
    future cell per origin, ``Ĉ_last · (tail − 1)``, with process error like
    the rest.

    What people get wrong: quoting the bootstrap *mean* as the reserve. The
    point estimate is the chain ladder's; the bootstrap's job is the spread
    around it, and its mean sits above the chain ladder by the resampling
    bias. Use :meth:`BootstrapResult.result` for VaR/TVaR and the
    :mod:`riskpy.viz` charts.
    """
    np = _numpy()
    tri = _as_triangle(triangle)
    tail = _check_tail(tail)
    if not isinstance(n, int) or n < 2:
        raise ValueError(f"n must be an integer >= 2 (1000 is a sensible default), got {n!r}")
    process = process.lower() if isinstance(process, str) else process
    if process not in ("odp", "gamma", "none"):
        raise ValueError(f"process must be 'odp', 'gamma' or 'none', got {process!r}")

    base = chain_ladder(tri, tail=tail)
    C, obs = tri._values, tri._observed
    latest_index = tri._latest_index
    n_origin, n_dev = tri.n_origin, tri.n_dev
    f = base.factors
    rows_all = np.arange(n_origin)

    # -- 1. back-fit the incremental triangle -----------------------------------------
    fitted = np.full((n_origin, n_dev), np.nan)
    fitted[rows_all, latest_index] = C[rows_all, latest_index]
    for k in range(n_dev - 2, -1, -1):
        earlier = latest_index > k
        fitted[earlier, k] = fitted[earlier, k + 1] / f[k]
    m = fitted.copy()
    m[:, 1:] = fitted[:, 1:] - fitted[:, :-1]
    if (m[obs] <= 0.0).any():
        raise ValueError(
            "the fitted incremental triangle has a non-positive cell (a development "
            "factor below 1, or a zero column). The ODP bootstrap needs positive fitted "
            "incrementals; this triangle is not an ODP triangle."
        )

    # -- 2. residuals, scale, degrees of freedom ------------------------------------------
    q = tri.incremental
    n_obs = int(obs.sum())
    p = n_origin + n_dev - 1
    dof = n_obs - p
    if dof <= 0:
        raise ValueError(
            f"the triangle has {n_obs} observed cells but the chain ladder fits {p} "
            f"parameters; the bootstrap needs more cells than parameters"
        )
    m_obs, q_obs = m[obs], q[obs]
    residuals = (q_obs - m_obs) / np.sqrt(m_obs)
    phi = float((residuals ** 2).sum() / dof)
    adjusted = residuals * math.sqrt(n_obs / dof)
    single_row = (obs.sum(axis=1) == 1)[:, None]
    single_col = (obs.sum(axis=0) == 1)[None, :]
    structural_zero = (single_row | single_col)[obs]
    pool = adjusted[~structural_zero]
    if pool.size < 2:
        raise ValueError("too few non-degenerate residuals to resample from")

    # -- 3. resample, refit, project ------------------------------------------------------
    rng = np.random.default_rng(seed)
    rows_idx, cols_idx = np.nonzero(obs)
    draws = rng.choice(pool, size=(n, n_obs), replace=True)
    pseudo_inc = np.zeros((n, n_origin, n_dev))
    pseudo_inc[:, rows_idx, cols_idx] = m_obs[None, :] + draws * np.sqrt(m_obs)[None, :]
    pseudo = np.cumsum(pseudo_inc, axis=2)                 # only meaningful on observed cells

    f_star = np.empty((n, n_dev - 1))
    for k in range(n_dev - 1):
        both = obs[:, k] & obs[:, k + 1]
        f_star[:, k] = pseudo[:, both, k + 1].sum(axis=1) / pseudo[:, both, k].sum(axis=1)
    if not np.all(np.isfinite(f_star)) or (f_star <= 0.0).any():
        raise ValueError(
            "a resampled triangle produced a non-positive column total, so its chain "
            "ladder is undefined. The cells are too small relative to their noise for "
            "the ODP bootstrap; use mack_chain_ladder for this triangle."
        )

    projected = np.full((n, n_origin, n_dev), np.nan)
    projected[:, rows_all, latest_index] = pseudo[:, rows_all, latest_index]
    for k in range(1, n_dev):
        future = latest_index < k
        projected[:, future, k] = projected[:, future, k - 1] * f_star[:, k - 1][:, None]
    future_mask = ~obs
    means = np.zeros((n, n_origin, n_dev + 1))
    means[:, :, 1:n_dev] = np.where(
        future_mask[None, :, 1:], projected[:, :, 1:] - projected[:, :, :-1], 0.0
    )
    means[:, :, n_dev] = projected[:, :, n_dev - 1] * (tail - 1.0)
    means = np.where(np.isnan(means), 0.0, means)

    # -- 4. process error around the projection ------------------------------------------
    simulated = means.copy()
    positive = means > 0.0
    if process == "odp" and positive.any():
        simulated[positive] = rng.gamma(means[positive] / phi, phi)
    elif process == "gamma" and positive.any():
        phi_gamma = float((((q_obs - m_obs) / m_obs) ** 2).sum() / dof)
        simulated[positive] = rng.gamma(1.0 / phi_gamma, phi_gamma * means[positive])

    reserves = simulated.sum(axis=2)
    total = reserves.sum(axis=1)
    return BootstrapResult(
        reserves=reserves,
        total=total,
        mean=reserves.mean(axis=0),
        se=reserves.std(axis=0, ddof=1),
        total_mean=float(total.mean()),
        total_se=float(total.std(ddof=1)),
        origin=tri.origin,
        latest=base.latest,
        cl_reserve=base.reserve,
        n=n,
        seed=seed,
        process=process,
        phi=phi,
    )


# ---------------------------------------------------------------------------
# Reference data and verification
# ---------------------------------------------------------------------------


_GENINS_INCREMENTAL = [
    [357848, 766940, 610542, 482940, 527326, 574398, 146342, 139950, 227229, 67948],
    [352118, 884021, 933894, 1183289, 445745, 320996, 527804, 266172, 425046],
    [290507, 1001799, 926219, 1016654, 750816, 146923, 495992, 280405],
    [310608, 1108250, 776189, 1562400, 272482, 352053, 206286],
    [443160, 693190, 991983, 769488, 504851, 470639],
    [396132, 937085, 847498, 805037, 705960],
    [440832, 847631, 1131398, 1063269],
    [359480, 1061648, 1443370],
    [376686, 986608],
    [344014],
]


def genins() -> Triangle:
    """The Taylor & Ashe (1983) triangle — ``GenIns`` in R's ChainLadder.

    Ten accident years of incremental paid claims, the data set Mack (1993)
    used, so every published number for it is a test of this module: factors
    3.4906, 1.7473, …, total reserve 18,680,856, Mack total s.e. 2,447,095.
    """
    return Triangle.from_incremental(_GENINS_INCREMENTAL)


def _verification_checks():
    """Return a list of (name, value, reference, tolerance) tuples."""
    np = _numpy()
    tri = genins()
    cl = chain_ladder(tri)
    mack = mack_chain_ladder(tri)

    # The compiled chain ladder on the same rows.
    from . import LossTriangle

    compiled = LossTriangle()
    C, obs = tri.cumulative, tri.observed
    for i in range(tri.n_origin):
        compiled.add_origin_year(i, [float(v) for v in C[i, obs[i]]])
    compiled_ultimate = np.asarray(compiled.get_ultimate_losses())
    cpp_gap = float(np.max(np.abs(compiled_ultimate / cl.ultimate - 1.0)))

    # Bornhuetter–Ferguson collapses to the chain ladder at the CL loss ratio.
    premium = np.full(tri.n_origin, 10_000_000.0)
    bf = bornhuetter_ferguson(tri, premium, cl.ultimate / premium)

    # Cape Cod's pooled ELR identity: Σ latest = ELR · Σ premium · %developed.
    cc = cape_cod(tri, premium)
    elr = float(cc.extra["expected_loss_ratio"][0])
    used_up = float((premium / cl.cdf[tri.latest_index]).sum())

    youngest = tri.origin[-1]
    return [
        ("GenIns first volume-weighted factor (Mack 1993)", float(cl.factors[0]), 3.4906, 5e-5),
        ("GenIns chain ladder total reserve (Mack 1993)", cl.total_reserve, 18_680_856.0, 1.0),
        ("GenIns Mack total s.e. (Mack 1993 Table)", mack.total_se, 2_447_095.0, 2_447.0),
        ("GenIns Mack s.e. of the youngest origin", float(mack.se[-1]), 1_363_155.0, 1_363.0),
        (f"Mack se² = process² + parameter² (origin {youngest})",
         float(mack.se[-1] ** 2), float(mack.process_se[-1] ** 2 + mack.parameter_se[-1] ** 2),
         1e-6 * float(mack.se[-1] ** 2)),
        ("chain ladder vs compiled LossTriangle (max relative gap)", cpp_gap, 0.0, 1e-6),
        ("Bornhuetter–Ferguson at ELR = CL ultimate / premium equals CL", bf.total_reserve,
         cl.total_reserve, 1e-6 * cl.total_reserve),
        ("Cape Cod: Σ latest = ELR · Σ premium · %developed", elr * used_up, tri.latest_diagonal.sum(),
         1e-6 * float(tri.latest_diagonal.sum())),
    ]
