"""Visualisation for risk output — one theme, one set of marks, many charts.

Every function takes data (a :class:`riskpy.mc.Result`, an array, a life table, a
reserving result), returns the Matplotlib ``Figure``, and never calls ``show()``
— so the same call works in a notebook, in a script that saves a PNG, and in a
test.

    from riskpy import viz

    fig = viz.distribution(result)
    viz.save(fig, "loss.png")

Matplotlib is an optional extra: ``pip install open-riskpy[viz]``.

The charts, by the question they answer
---------------------------------------

============================  =================================================
``distribution``              what does the spread look like, and where is the tail?
``density``                   the same, smoothed, when the histogram bins fight you
``cdf``                       what is the probability of being under X?
``exceedance``                how likely is a loss bigger than X?
``convergence``               did I run enough trials?
``tornado``                   which input is driving the answer?
``qq``                        does this distribution actually fit?
``compare``                   how do scenarios sit against each other?
``spread``                    the same, as quantile ranges, when there are many
``fan``                       what do the paths look like over time?
``dashboard``                 all four of the usual ones at once
``correlation``               what depends on what?
``scatter``                   what does that dependence look like?
``triangle``                  where is the run-off pattern unstable?
``development``               how does each origin year develop to ultimate?
``reserve_range``             how uncertain is each origin's reserve?
``survival``                  how many survive to each age?
``mortality``                 what is the force of mortality doing?
``reserve_profile``           how does a policy reserve build and release?
``curve``                     what do the zero and forward curves look like?
``allocation``                who is using the capital?
``waterfall``                 how do the pieces add up to the total?
============================  =================================================

Design rules worth not undoing by accident
------------------------------------------
* **One series, one colour.** Bars in a single-series chart are all the same
  hue — colouring them by height double-encodes a value the length already
  shows. Colour is reserved for identity (which scenario) and for status
  (VaR, TVaR).
* **Direction never rests on colour alone.** VaR and TVaR markers carry a
  label as well as a hue, so the chart survives greyscale printing and the
  most common forms of colour vision deficiency.
* **Thin marks, hairline grid.** Gridlines are one step off the surface and
  solid — dashed grid reads as a threshold when it is just a grid.
* **Never two y-axes.** If you want two quantities of different scale, that is
  two charts, and ``compare`` will index them for you.
* **Sequential means one hue, light to dark.** Diverging means two opposed
  hues with a neutral grey midpoint, never a hue in the middle.

Both palettes are validated: every categorical step clears 3:1 contrast on its
own background, adjacent pairs stay separable under protanopia, deuteranopia
and tritanopia, and the lightness band and chroma floor hold. Swap the hexes
for a house palette if you have one, but keep the slot ORDER — that ordering is
what makes adjacent series distinguishable, and re-sorting it quietly breaks
the property.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .mc import _numpy  # single source of the "install NumPy" error message

__all__ = [
    "theme",
    "PALETTE",
    "distribution",
    "density",
    "cdf",
    "exceedance",
    "convergence",
    "tornado",
    "qq",
    "compare",
    "spread",
    "fan",
    "dashboard",
    "correlation",
    "scatter",
    "triangle",
    "development",
    "reserve_range",
    "survival",
    "mortality",
    "reserve_profile",
    "curve",
    "allocation",
    "waterfall",
    "save",
]


PALETTE: Dict[str, Dict[str, Any]] = {
    "dark": {
        "page": "#0d0d0d",
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
        # Sequential: one hue, low to high. Diverging: two opposed hues with a
        # neutral midpoint that reads as "nothing".
        "sequential": ["#16243a", "#1d4a80", "#2a68b8", "#3987e5", "#7db2ee", "#c3dbf8"],
        "diverging": ["#3987e5", "#7db2ee", "#4a4a46", "#e69a70", "#d95926"],
        "good": "#0ca30c",
        "critical": "#d03b3b",
        "warning": "#fab219",
    },
    "light": {
        "page": "#f9f9f7",
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#6b6a65",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#d1541c", "#00816c", "#9c7100", "#c9527f", "#2f7d32", "#4a3aa7", "#cf3f3e"],
        "sequential": ["#eaf1fb", "#c3dbf8", "#7db2ee", "#3987e5", "#2a68b8", "#14375f"],
        "diverging": ["#2a78d6", "#7db2ee", "#b9b8b0", "#e69a70", "#d1541c"],
        "good": "#006300",
        "critical": "#c22f2f",
        "warning": "#8a5a00",
    },
}

_MODE = "dark"
_themed = False


def _plt():
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "riskpy.viz needs Matplotlib, which is not installed:\n"
            "    pip install open-riskpy[viz]\n"
            "or just:\n"
            "    pip install matplotlib"
        ) from exc
    return plt


def _numpy():
    from .mc import _numpy as _np  # single source of the error message

    return _np()


def _colours() -> Dict[str, Any]:
    return PALETTE[_MODE]


def theme(mode: str = "dark") -> None:
    """Apply the RiskPY look to every subsequent plot.

    Called automatically the first time you plot, so you only need this to
    switch modes:

        riskpy.viz.theme("light")
    """
    global _MODE, _themed
    if mode not in PALETTE:
        raise ValueError(f"mode must be 'dark' or 'light', got {mode!r}")
    _MODE = mode
    _themed = True

    plt = _plt()
    c = _colours()
    plt.rcParams.update(
        {
            "figure.facecolor": c["page"],
            "axes.facecolor": c["surface"],
            "savefig.facecolor": c["page"],
            "text.color": c["text"],
            "axes.labelcolor": c["muted"],
            "axes.edgecolor": c["axis"],
            "axes.titlecolor": c["text"],
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 14,
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": c["grid"],
            "grid.linewidth": 1.0,
            "grid.linestyle": "-",  # never dashed: a dashed grid reads as a threshold
            "xtick.color": c["muted"],
            "ytick.color": c["muted"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "font.size": 10,
            "figure.autolayout": True,
        }
    )


def _ensure_theme() -> None:
    global _themed
    if not _themed:
        theme(_MODE)


def _figure(figsize):
    _ensure_theme()
    plt = _plt()
    return plt.subplots(figsize=figsize)


def _colormap(kind: str = "sequential"):
    """Build a Matplotlib colormap from the palette's ramp stops."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(f"riskpy_{kind}_{_MODE}", _colours()[kind])


def _compact(value: float) -> str:
    """Short money label: 8.1m rather than 8,091,985.

    Loss figures run to nine digits, and nine-digit tick labels on a 900-pixel
    axis overlap into a grey smear. Ticks get the compact form; annotations,
    where the exact number is the point, keep every digit.
    """
    magnitude = abs(value)
    for cutoff, suffix in ((1e12, "tn"), (1e9, "bn"), (1e6, "m"), (1e3, "k")):
        if magnitude >= cutoff:
            scaled = value / cutoff
            text = f"{scaled:.1f}".rstrip("0").rstrip(".")
            return f"{text}{suffix}"
    if magnitude >= 1.0 or value == 0.0:
        return f"{value:,.0f}"
    return f"{value:,.3g}"


def _money(ax, axis: str = "x", ticks: int = 6) -> None:
    """Compact, sparse tick labels. Raw floats are unreadable at loss scale."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(FuncFormatter(lambda v, _pos: _compact(v)))
    target.set_major_locator(MaxNLocator(nbins=ticks, prune=None))


def _percent(ax, axis: str = "y") -> None:
    """Percent ticks with as many decimals as the tick spacing needs — a curve
    spanning 3.2% to 4.4% must not print every tick as "4%"."""
    from matplotlib.ticker import PercentFormatter

    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(PercentFormatter(xmax=1.0, decimals=None))


def _spread_labels(positions: Sequence[float], min_gap: float) -> List[float]:
    """Nudge label positions apart so none overlap, preserving their order.

    Direct labels at a series' end point stack on top of each other whenever
    two series finish close together, which in a development chart is the
    normal case. A single upward sweep pushing each label clear of the one
    below it is enough; the leader line back to the data keeps it honest.
    """
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    adjusted = list(positions)
    for k in range(1, len(order)):
        prev, cur = order[k - 1], order[k]
        if adjusted[cur] - adjusted[prev] < min_gap:
            adjusted[cur] = adjusted[prev] + min_gap
    return adjusted


def _values(data):
    """Accept a Result or a raw array everywhere a sample is wanted."""
    np = _numpy()
    if hasattr(data, "values") and hasattr(data, "trials"):
        return np.asarray(data.values, dtype=float)
    return np.asarray(data, dtype=float)


def _label_of(data, default: str = "value") -> str:
    return getattr(data, "label", default) or default


def _kde(values, grid_size: int = 512, bandwidth: Optional[float] = None):
    """Gaussian kernel density on a grid, by binning and convolving.

    Evaluating a kernel at every one of a million points against every grid
    node is a million-by-512 operation nobody needs. Binning first and
    convolving the histogram with the kernel is the same estimate to well
    inside plotting accuracy, and it is linear in the sample size.

    Bandwidth defaults to Silverman's rule using the smaller of the standard
    deviation and the interquartile range, which keeps a heavy tail from
    over-smoothing the body.
    """
    np = _numpy()
    values = np.asarray(values, dtype=float)
    n = values.size
    if n < 2:
        raise ValueError("a density estimate needs at least two observations")

    sd = float(values.std(ddof=1))
    iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
    spread_est = min(sd, iqr / 1.349) if iqr > 0 else sd
    if spread_est <= 0:
        spread_est = abs(float(values.mean())) * 1e-6 or 1.0
    if bandwidth is None:
        bandwidth = 0.9 * spread_est * n ** (-0.2)

    lo = float(values.min()) - 3.0 * bandwidth
    hi = float(values.max()) + 3.0 * bandwidth
    grid = np.linspace(lo, hi, grid_size)
    step = grid[1] - grid[0]

    counts, _ = np.histogram(values, bins=grid_size, range=(lo, hi))
    # Kernel out to four bandwidths; beyond that it contributes nothing visible.
    reach = max(1, int(math.ceil(4.0 * bandwidth / step)))
    offsets = np.arange(-reach, reach + 1) * step
    kernel = np.exp(-0.5 * (offsets / bandwidth) ** 2)
    kernel /= kernel.sum() * step

    density = np.convolve(counts.astype(float), kernel, mode="same") / n
    return grid, density


def _finish(ax, title, xlabel=None, ylabel=None):
    c = _colours()
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


# ---------------------------------------------------------------------------
# Distribution shape
# ---------------------------------------------------------------------------


def _draw_distribution(ax, result, bins=60, levels=(0.95, 0.995), title=None, clip=0.999):
    np = _numpy()
    c = _colours()
    values = _values(result)

    # A loss distribution is heavy-tailed by nature: one trial in a hundred
    # thousand can be fifty times the mean, and a histogram drawn to that trial
    # puts every other trial in the first bin. Bin to a high quantile instead
    # and say how many trials ran off the end — hiding them silently would be
    # worse than the unreadable chart.
    upper = float(np.percentile(values, clip * 100.0)) if clip else float(values.max())
    lower = float(values.min())
    if not (upper > lower):
        upper = lower + 1.0
    beyond = int((values > upper).sum())
    ax.hist(np.clip(values, lower, upper), bins=bins, range=(lower, upper),
            color=c["series"][0], edgecolor="none", alpha=0.9)
    ax.set_xlim(lower, upper)

    ymax = ax.get_ylim()[1]
    for i, level in enumerate(levels):
        var = result.var(level)
        tvar = result.tvar(level)
        ax.axvline(var, color=c["critical"], linewidth=2.0)
        ax.text(var, ymax * (0.97 - 0.09 * i), f" VaR {level:.1%}  {_compact(var)}",
                color=c["text"], fontsize=9, va="top", ha="left")
        if tvar <= upper:
            ax.axvline(tvar, color=c["warning"], linewidth=1.5)
            ax.text(tvar, ymax * (0.97 - 0.09 * i) - ymax * 0.045,
                    f" TVaR {_compact(tvar)}", color=c["muted"], fontsize=8.5,
                    va="top", ha="left")

    ax.axvline(float(values.mean()), color=c["muted"], linewidth=1.0)
    label = _label_of(result)
    _finish(ax, title or f"Distribution of {label} — {len(values):,} trials", label, "trials")
    if beyond:
        ax.annotate(f"{beyond:,} trials beyond {_compact(upper)}  (max {_compact(float(values.max()))})",
                    xy=(0.995, 0.02), xycoords="axes fraction", ha="right", va="bottom",
                    color=c["muted"], fontsize=8.5)
    ax.set_yticks([])  # the counts carry no decision; the shape and the rules do
    _money(ax, "x")
    return ax


def distribution(result, bins: int = 60, levels: Sequence[float] = (0.95, 0.995),
                 title: Optional[str] = None, clip: Optional[float] = 0.999, figsize=(9, 5)):
    """Histogram of the simulated output, with VaR and TVaR called out.

    The markers are the point of the chart: the shape tells you the story, but
    the tail statistics are what anyone acts on.

    ``clip`` truncates the axis at that quantile so the body of a heavy-tailed
    distribution is visible; the trials past it are counted in a note rather
    than dropped. Pass ``clip=None`` to draw the full range.
    """
    fig, ax = _figure(figsize)
    _draw_distribution(ax, result, bins, levels, title, clip)
    return fig


def density(result, levels: Sequence[float] = (0.95, 0.995), title: Optional[str] = None,
            bandwidth: Optional[float] = None, figsize=(9, 5)):
    """Smoothed density instead of bars.

    Worth reaching for when the bin width is doing the arguing — a histogram of
    a heavy-tailed loss either buries the body or loses the tail, and the
    reader cannot tell which. The shaded tail beyond VaR is the region the
    capital number is about.
    """
    np = _numpy()
    c = _colours()
    values = _values(result)
    fig, ax = _figure(figsize)

    grid, dens = _kde(values, bandwidth=bandwidth)
    ax.plot(grid, dens, color=c["series"][0])
    ax.set_xlim(float(values.min()), float(np.percentile(values, 99.9)))
    ax.fill_between(grid, dens, color=c["series"][0], alpha=0.18, linewidth=0)

    if levels:
        var = result.var(max(levels))
        tail = grid >= var
        ax.fill_between(grid[tail], dens[tail], color=c["critical"], alpha=0.35, linewidth=0)
        ax.axvline(var, color=c["critical"], linewidth=2.0)
        ax.text(var, ax.get_ylim()[1] * 0.94, f"  VaR {max(levels):.1%}  {var:,.0f}",
                color=c["text"], fontsize=9, va="top")

    label = _label_of(result)
    _finish(ax, title or f"Density of {label}", label, "density")
    ax.set_yticks([])
    _money(ax, "x")
    return fig


def cdf(result, levels: Sequence[float] = (0.5, 0.95, 0.995), title: Optional[str] = None,
        figsize=(9, 5)):
    """Empirical cumulative distribution — P(outcome <= x).

    The chart to read a probability off directly. Where :func:`exceedance`
    stretches the far tail on a log axis, this keeps the whole distribution on
    one linear scale, which is the better view when the question is about the
    body rather than the extreme.
    """
    np = _numpy()
    c = _colours()
    values = np.sort(_values(result))
    fig, ax = _figure(figsize)

    probs = np.arange(1, values.size + 1) / values.size
    ax.plot(values, probs, color=c["series"][0])

    for level in levels:
        x = result.var(level)
        ax.plot([x], [level], marker="o", markersize=8, color=c["series"][0],
                markeredgecolor=c["surface"], markeredgewidth=2)
        ax.annotate(f"{level:.1%}  {x:,.0f}", xy=(x, level), xytext=(8, -4),
                    textcoords="offset points", color=c["text"], fontsize=9)

    ax.set_ylim(0, 1.02)
    _percent(ax, "y")
    label = _label_of(result)
    _finish(ax, title or f"Cumulative distribution — {label}", label, "P(outcome ≤ x)")
    _money(ax, "x")
    return fig


def _draw_exceedance(ax, result, log_y=True, title=None):
    np = _numpy()
    c = _colours()
    values = _values(result)

    ordered = np.sort(values)
    # P(X > x) at each observed value: 1 - empirical CDF, without the zero at
    # the top end that would vanish on a log axis.
    survival = 1.0 - np.arange(1, ordered.size + 1) / (ordered.size + 1)

    ax.plot(ordered, survival, color=c["series"][0])
    if log_y:
        ax.set_yscale("log")

    for level in (0.95, 0.995):
        var = result.var(level)
        ax.plot([var], [1.0 - level], marker="o", markersize=7,
                color=c["critical"], markeredgecolor=c["surface"], markeredgewidth=2)
        ax.annotate(f"{level:.1%}  {var:,.0f}", xy=(var, 1.0 - level), xytext=(8, 6),
                    textcoords="offset points", color=c["text"], fontsize=9)

    label = _label_of(result)
    _finish(ax, title or f"Exceedance probability — {label}", label, "P(loss exceeds x)")
    _money(ax, "x")
    return ax


def exceedance(result, log_y: bool = True, title: Optional[str] = None, figsize=(9, 5)):
    """P(loss > x) against x — the survival curve.

    The natural way to answer "how likely is something worse than this?", and
    far easier to read off than a histogram tail. Log scale by default because
    the interesting probabilities are all near zero.
    """
    fig, ax = _figure(figsize)
    _draw_exceedance(ax, result, log_y, title)
    return fig


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _draw_convergence(ax, result, points=200, title=None):
    c = _colours()
    n, running, half = result.convergence(points=points)

    ax.fill_between(n, running - half, running + half, color=c["series"][0], alpha=0.16, linewidth=0)
    ax.plot(n, running, color=c["series"][0])
    ax.axhline(result.mean, color=c["muted"], linewidth=1.0)

    ax.annotate(f"mean {result.mean:,.2f}  ± {1.96 * result.standard_error:,.2f}",
                xy=(n[-1], running[-1]), xytext=(-4, 10), textcoords="offset points",
                ha="right", color=c["text"], fontsize=9)

    _finish(ax, title or f"Convergence of the mean — {_label_of(result)}", "trials", "running mean")
    _money(ax, "x")
    _money(ax, "y")
    return ax


def convergence(result, points: int = 200, title: Optional[str] = None, figsize=(9, 4.5)):
    """Running mean with a 95% band — the "have I run enough trials?" chart.

    If the band is still narrowing noticeably at the right-hand edge, the answer
    is no.
    """
    fig, ax = _figure(figsize)
    _draw_convergence(ax, result, points, title)
    return fig


def _draw_tornado(ax, result, top=12, method="spearman", title=None):
    np = _numpy()
    c = _colours()
    scores = result.sensitivity(method=method)[:top]
    if not scores:
        raise ValueError("No inputs to rank.")

    names = [name for name, _ in scores][::-1]
    values = [value for _, value in scores][::-1]
    y = np.arange(len(names))

    ax.barh(y, values, height=0.55, color=c["series"][0], edgecolor="none")
    ax.axvline(0, color=c["axis"], linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=c["text"], fontsize=10)

    for yi, value in zip(y, values):
        offset = 0.02 if value >= 0 else -0.02
        ax.text(value + offset, yi, f"{value:+.2f}", va="center",
                ha="left" if value >= 0 else "right", color=c["text"], fontsize=9)

    limit = max(0.25, min(1.0, max(abs(v) for v in values) * 1.35))
    ax.set_xlim(-limit, limit)
    xlabel = {"spearman": "rank correlation with the output",
              "pearson": "linear correlation with the output",
              "contribution": "share of explained variance"}[method]
    _finish(ax, title or f"What drives {_label_of(result)}", xlabel)
    ax.grid(axis="y", visible=False)
    return ax


def tornado(result, top: int = 12, method: str = "spearman", title: Optional[str] = None,
            figsize=(9, 5)):
    """Which inputs move the output, strongest first.

    Bars are rank correlation by default, so they run from -1 to 1 and the sign
    means direction. One colour for every bar: the length is the value, and
    tinting by magnitude would say the same thing twice.
    """
    fig, ax = _figure(figsize)
    _draw_tornado(ax, result, top, method, title)
    return fig


def qq(result, dist=None, title: Optional[str] = None, figsize=(6.5, 6.5)):
    """Quantile–quantile plot against a fitted distribution.

    Points on the line mean the distribution fits. The informative part is
    always the ends: severity curves that look fine in the body routinely miss
    by a factor of two in the tail, and that is the part capital depends on.

    ``dist`` is anything with a ``ppf`` — a :mod:`riskpy.mc` distribution, or a
    :class:`riskpy.fit.Fit`'s ``distribution``. Omit it and the sample is
    checked against a normal with the same mean and standard deviation, which
    is the usual first question.
    """
    np = _numpy()
    c = _colours()
    values = np.sort(_values(result))
    n = values.size
    if n < 2:
        raise ValueError("a Q–Q plot needs at least two observations")

    if dist is None:
        from .mc import Normal

        dist = Normal(float(values.mean()), float(values.std(ddof=1)))
    if not hasattr(dist, "ppf"):
        raise TypeError(f"dist must have a ppf method, got {type(dist).__name__}")

    # Hazen plotting positions: (i - 0.5) / n is unbiased for the median of the
    # order statistic and, unlike i/n, does not ask for the infinite quantile.
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = np.asarray(dist.ppf(probs), dtype=float)

    fig, ax = _figure(figsize)
    ax.plot(theoretical, values, linestyle="none", marker="o", markersize=4,
            color=c["series"][0], alpha=0.55, markeredgecolor="none")

    lo = float(min(theoretical.min(), values.min()))
    hi = float(max(theoretical.max(), values.max()))
    ax.plot([lo, hi], [lo, hi], color=c["muted"], linewidth=1.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    _finish(ax, title or f"Q–Q against {type(dist).__name__}", "theoretical quantile", "observed quantile")
    _money(ax, "x")
    _money(ax, "y")
    return fig


def dashboard(result, title: Optional[str] = None, figsize=(13, 9)):
    """The four standard diagnostics on one page.

    Distribution, exceedance, convergence and tornado — what you would look at
    anyway before believing a simulation. Falls back gracefully when a run kept
    no inputs, since then there is nothing to rank.
    """
    _ensure_theme()
    plt = _plt()
    c = _colours()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.set_layout_engine("constrained")

    _draw_distribution(axes[0][0], result, title="Distribution")
    _draw_exceedance(axes[0][1], result, title="Exceedance")
    _draw_convergence(axes[1][0], result, title="Convergence of the mean")
    try:
        _draw_tornado(axes[1][1], result, title="What drives it")
    except ValueError:
        axes[1][1].text(0.5, 0.5, "no inputs kept\n(run with keep_inputs=True)",
                        ha="center", va="center", color=c["muted"], fontsize=10,
                        transform=axes[1][1].transAxes)
        axes[1][1].set_axis_off()

    fig.suptitle(title or f"{_label_of(result)} — {result.trials:,} trials",
                 color=c["text"], fontsize=14, fontweight="semibold", x=0.02, ha="left")
    return fig


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare(results: Mapping[str, Any], bins: int = 60, title: str = "Scenario comparison",
            figsize=(9, 5)):
    """Overlay several runs. Colour is identity here, so a legend is mandatory.

    ``results`` maps a scenario name to a :class:`~riskpy.mc.Result`. Past eight
    scenarios the palette runs out on purpose — nine colours cannot be told
    apart reliably, so group the tail into an "other" scenario, or use
    :func:`spread`, which stays readable for dozens.
    """
    c = _colours()
    if not results:
        raise ValueError("Nothing to compare.")
    if len(results) > len(c["series"]):
        raise ValueError(
            f"{len(results)} scenarios exceeds the {len(c['series'])}-colour palette. "
            f"Beyond that the colours stop being distinguishable — use viz.spread() "
            f"for many scenarios, or facet into separate charts."
        )

    fig, ax = _figure(figsize)

    for i, (name, result) in enumerate(results.items()):
        colour = c["series"][i]
        ax.hist(_values(result), bins=bins, color=colour, alpha=0.45,
                edgecolor="none", label=name)
        ax.axvline(result.var(0.995), color=colour, linewidth=1.5)

    _finish(ax, title, "outcome", "trials")
    ax.set_yticks([])
    ax.legend(loc="upper right")
    _money(ax, "x")
    return fig


def spread(results: Mapping[str, Any], levels: Sequence[float] = (0.5, 0.9, 0.99),
           title: str = "Scenario ranges", figsize=(9, 5)):
    """Quantile ranges, one row per scenario — the many-scenario alternative.

    Nested bands around the median, with the mean marked. Reads at a glance for
    dozens of scenarios where overlaid histograms would be mud, and it uses one
    hue throughout because the row label already carries the identity.
    """
    np = _numpy()
    c = _colours()
    if not results:
        raise ValueError("Nothing to compare.")
    levels = sorted(levels)

    fig, ax = _figure(figsize)
    names = list(results)
    for row, name in enumerate(names):
        result = results[name]
        values = _values(result)
        for depth, level in enumerate(reversed(levels)):
            lo = float(np.percentile(values, (1 - level) / 2 * 100))
            hi = float(np.percentile(values, (1 + level) / 2 * 100))
            height = 0.16 + 0.11 * depth
            ax.barh(row, hi - lo, left=lo, height=height, color=c["series"][0],
                    alpha=0.20 + 0.22 * depth, edgecolor="none")
        median = float(np.percentile(values, 50))
        ax.plot([median], [row], marker="|", markersize=18, color=c["text"], markeredgewidth=2)
        ax.plot([float(values.mean())], [row], marker="o", markersize=6,
                color=c["series"][0], markeredgecolor=c["surface"], markeredgewidth=2)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color=c["text"], fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    bands = ", ".join(f"{l:.0%}" for l in levels)
    _finish(ax, title, f"outcome — bands at {bands}, bar is the median, dot the mean")
    _money(ax, "x")
    return fig


def fan(paths, levels: Sequence[float] = (0.5, 0.8, 0.95), title: str = "Simulated paths",
        ylabel: str = "value", times: Optional[Sequence[float]] = None, figsize=(9, 5)):
    """Percentile fan for path simulations — GBM, reserve run-off, projections.

    ``paths`` is ``(trials, steps)``. Shows the median as a line and nested
    bands for each level, which reads far better than 10,000 spaghetti lines.
    """
    np = _numpy()
    c = _colours()
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2:
        raise ValueError(f"paths must be 2-D (trials, steps), got shape {paths.shape}")

    fig, ax = _figure(figsize)
    steps = np.arange(paths.shape[1]) if times is None else np.asarray(times, dtype=float)
    if steps.size != paths.shape[1]:
        raise ValueError(f"times has {steps.size} entries but paths have {paths.shape[1]} steps")

    # Widest band first so the narrower, darker ones sit on top.
    for i, level in enumerate(sorted(levels, reverse=True)):
        lower = np.percentile(paths, (1 - level) / 2 * 100, axis=0)
        upper = np.percentile(paths, (1 + level) / 2 * 100, axis=0)
        ax.fill_between(steps, lower, upper, color=c["series"][0],
                        alpha=0.12 + 0.10 * i, linewidth=0, label=f"{level:.0%} band")

    ax.plot(steps, np.percentile(paths, 50, axis=0), color=c["series"][0], label="median")

    _finish(ax, title, "step" if times is None else "time", ylabel)
    ax.legend(loc="upper left")
    _money(ax, "y")
    return fig


# ---------------------------------------------------------------------------
# Dependence
# ---------------------------------------------------------------------------


def correlation(data, names: Optional[Sequence[str]] = None, method: str = "spearman",
                title: Optional[str] = None, figsize=(7.5, 6.5)):
    """Correlation matrix as a heatmap, on a diverging scale centred at zero.

    ``data`` is a :class:`~riskpy.mc.Result` (its sampled inputs), a mapping of
    name to array, or a square correlation matrix. Two opposed hues with a
    neutral grey midpoint, because the midpoint here means "no relationship"
    and has to read as nothing.
    """
    np = _numpy()
    c = _colours()

    if hasattr(data, "inputs") and getattr(data, "inputs", None):
        columns = dict(data.inputs)
    elif isinstance(data, Mapping):
        columns = dict(data)
    else:
        matrix = np.asarray(data, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"expected a square matrix, a mapping or a Result, got shape {matrix.shape}")
        columns = None

    if columns is not None:
        if len(columns) < 2:
            raise ValueError("a correlation matrix needs at least two variables")
        names = list(columns) if names is None else list(names)
        stacked = np.vstack([np.asarray(columns[n], dtype=float) for n in names])
        if method == "spearman":
            from .mc import _rankdata

            stacked = np.vstack([_rankdata(row, np) for row in stacked])
        elif method != "pearson":
            raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")
        matrix = np.corrcoef(stacked)
    elif names is None:
        names = [str(i) for i in range(matrix.shape[0])]

    fig, ax = _figure(figsize)
    image = ax.imshow(matrix, cmap=_colormap("diverging"), vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", color=c["text"], fontsize=9)
    ax.set_yticklabels(names, color=c["text"], fontsize=9)
    ax.grid(visible=False)

    # Every cell labelled: a heatmap without numbers makes the reader guess a
    # value off a colour ramp, which nobody can do to better than a decimal.
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color=c["text"], fontsize=8)

    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=c["muted"], labelsize=8)
    _finish(ax, title or f"{method.title()} correlation")
    return fig


def scatter(result, x: str, y: str, sample: int = 5000, title: Optional[str] = None,
            figsize=(6.5, 6)):
    """Two sampled inputs against each other — what the dependence looks like.

    Correlation is one number and hides shape: a rank correlation of 0.6 can be
    an even cloud or a pair of arms that only meet in the tail, and those price
    very differently. Points are thinned to ``sample`` because a hundred
    thousand overlapping dots is a filled rectangle.
    """
    np = _numpy()
    c = _colours()
    inputs = getattr(result, "inputs", None)
    if not inputs:
        raise ValueError("this run kept no inputs; re-run with keep_inputs=True")
    for name in (x, y):
        if name not in inputs:
            raise ValueError(f"unknown variable {name!r}; known: {list(inputs)}")

    xs = np.asarray(inputs[x], dtype=float)
    ys = np.asarray(inputs[y], dtype=float)
    if xs.size > sample:
        rng = np.random.default_rng(0)
        pick = rng.choice(xs.size, size=sample, replace=False)
        xs, ys = xs[pick], ys[pick]

    fig, ax = _figure(figsize)
    ax.plot(xs, ys, linestyle="none", marker="o", markersize=3.5,
            color=c["series"][0], alpha=0.35, markeredgecolor="none")
    _finish(ax, title or f"{y} against {x}", x, y)
    _money(ax, "x")
    _money(ax, "y")
    return fig


# ---------------------------------------------------------------------------
# Reserving
# ---------------------------------------------------------------------------


def triangle(tri, kind: str = "link", title: Optional[str] = None, figsize=(10, 5.5)):
    """The run-off triangle as a heatmap.

    ``kind="link"`` shows individual age-to-age factors, which is the diagnostic
    view: the chain ladder assumes each column's factor is the same for every
    origin, and this is where you see that it is not. ``kind="cumulative"`` and
    ``kind="incremental"`` show the amounts.

    One hue, light to dark — the value has a natural order, so a sequential
    ramp is right and a rainbow would invent categories that are not there.
    """
    np = _numpy()
    c = _colours()

    if kind == "link":
        matrix = np.asarray(tri.link_ratios(), dtype=float)
        fmt = "{:.3f}"
        caption = "age-to-age factor"
    elif kind == "cumulative":
        matrix = np.asarray(tri.cumulative, dtype=float)
        fmt = "{:,.0f}"
        caption = "cumulative paid"
    elif kind == "incremental":
        matrix = np.asarray(tri.incremental, dtype=float)
        fmt = "{:,.0f}"
        caption = "incremental paid"
    else:
        raise ValueError(f"kind must be 'link', 'cumulative' or 'incremental', got {kind!r}")

    fig, ax = _figure(figsize)
    masked = np.ma.masked_invalid(matrix)
    cmap = _colormap("sequential").with_extremes(bad=c["surface"])
    image = ax.imshow(masked, cmap=cmap, aspect="auto")

    origins = [str(o) for o in tri.origin]
    ax.set_yticks(range(len(origins)))
    ax.set_yticklabels(origins, color=c["text"], fontsize=9)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(range(1, matrix.shape[1] + 1), color=c["muted"], fontsize=9)
    ax.grid(visible=False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isfinite(matrix[i, j]):
                continue
            ax.text(j, i, fmt.format(matrix[i, j]), ha="center", va="center",
                    color=c["text"], fontsize=7.5)

    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=c["muted"], labelsize=8)
    _finish(ax, title or f"Run-off triangle — {caption}", "development period", "origin")
    return fig


def development(result, triangle=None, title: Optional[str] = None, figsize=(9, 5.5)):
    """Each origin's cumulative development, with the projection dashed.

    Takes a :class:`riskpy.reserving.ChainLadderResult`. The solid part is what
    has been paid; the dashed continuation is what the factors say is still to
    come. Dashing is doing real work here — it separates fact from projection,
    which is the one distinction a reserving chart has to make.

    Pass the original ``triangle`` when the data is not a standard staircase
    (each origin one period less developed than the one before); without it
    that shape is assumed, which is right for the overwhelming majority of
    triangles and wrong silently for the rest.
    """
    np = _numpy()
    c = _colours()
    full = np.asarray(result.full_triangle, dtype=float)
    origins = list(result.origin)
    n_dev = full.shape[1]
    observed = None
    if triangle is not None:
        observed = np.isfinite(np.asarray(triangle.cumulative, dtype=float))

    fig, ax = _figure(figsize)
    ramp = _colormap("sequential")
    ends = []
    for i, origin in enumerate(origins):
        shade = ramp(0.25 + 0.7 * (i / max(len(origins) - 1, 1)))
        row = full[i]
        known = int(observed[i].sum()) if observed is not None else n_dev - i
        known = max(1, min(known, n_dev))
        x = np.arange(1, n_dev + 1)
        ax.plot(x[:known], row[:known], color=shade, linewidth=1.8)
        if known < n_dev:
            ax.plot(x[known - 1:], row[known - 1:], color=shade, linewidth=1.4, linestyle=(0, (4, 3)))
        ends.append(float(row[-1]))

    span = float(np.nanmax(full) - np.nanmin(full)) or 1.0
    label_y = _spread_labels(ends, 0.035 * span)
    for origin, end, y in zip(origins, ends, label_y):
        if abs(y - end) > 1e-9 * span:
            ax.plot([n_dev, n_dev + 0.35], [end, y], color=c["muted"], linewidth=0.8)
        ax.annotate(str(origin), xy=(n_dev + 0.35, y), xytext=(4, 0), textcoords="offset points",
                    color=c["text"], fontsize=8, va="center")

    _finish(ax, title or "Development to ultimate — solid is paid, dashed is projected",
            "development period", "cumulative")
    _money(ax, "y")
    ax.set_xlim(1, n_dev + 1.0)
    return fig


def reserve_range(result, level: float = 0.95, title: Optional[str] = None, figsize=(9, 5.5)):
    """Reserve by origin with an uncertainty range.

    Takes a :class:`riskpy.reserving.MackResult` (which carries a standard
    error per origin) or a :class:`~riskpy.reserving.BootstrapResult`. The bar
    is the estimate; the line is the range. The total is shown apart from the
    origins because it is not on the same scale and does not compete with them.
    """
    np = _numpy()
    c = _colours()
    if not hasattr(result, "origin") or not (hasattr(result, "se") or hasattr(result, "reserves")):
        raise TypeError(
            f"expected a MackResult or a BootstrapResult from riskpy.reserving, got {type(result).__name__}"
        )

    origins = [str(o) for o in result.origin]
    reserves = np.asarray(result.reserve if hasattr(result, "reserve") else result.mean, dtype=float)

    if hasattr(result, "se"):
        errors = np.asarray(result.se, dtype=float)
        band = f"± 1 standard error"
    elif hasattr(result, "reserves"):
        lo = np.percentile(result.reserves, (1 - level) / 2 * 100, axis=0)
        hi = np.percentile(result.reserves, (1 + level) / 2 * 100, axis=0)
        errors = np.vstack([reserves - lo, hi - reserves])
        band = f"{level:.0%} bootstrap interval"
    else:
        raise TypeError("expected a MackResult or a BootstrapResult")

    y = np.arange(len(origins))
    fig, ax = _figure(figsize)
    ax.barh(y, reserves, height=0.55, color=c["series"][0], edgecolor="none")
    ax.errorbar(reserves, y, xerr=errors, fmt="none", ecolor=c["text"],
                elinewidth=1.4, capsize=4, capthick=1.4)
    ax.set_yticks(y)
    ax.set_yticklabels(origins, color=c["text"], fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)

    total = float(np.sum(reserves))
    total_se = getattr(result, "total_se", None)
    caption = f"total {total:,.0f}"
    if total_se:
        caption += f" ± {total_se:,.0f}"
    _finish(ax, title or f"Reserve by origin — {caption}", f"reserve ({band})")
    _money(ax, "x")
    return fig


# ---------------------------------------------------------------------------
# Life
# ---------------------------------------------------------------------------


def survival(table, title: Optional[str] = None, figsize=(9, 5)):
    """The survivorship curve ``l_x`` — how many of the original cohort remain.

    Takes a :class:`riskpy.life.LifeTable`. The steepest part of the curve is
    where the annuity value is decided, which is rarely where people expect.
    """
    np = _numpy()
    c = _colours()
    ages, lx = table.survival_curve()

    fig, ax = _figure(figsize)
    ax.plot(ages, lx, color=c["series"][0])
    ax.fill_between(ages, lx, color=c["series"][0], alpha=0.15, linewidth=0)

    # Median future lifetime: the age where half the cohort is gone.
    radix = lx[0]
    half = next((a for a, l in zip(ages, lx) if l <= radix / 2), None)
    if half is not None:
        ax.axvline(half, color=c["muted"], linewidth=1.0)
        ax.annotate(f"half gone by {half}", xy=(half, radix / 2), xytext=(8, 8),
                    textcoords="offset points", color=c["text"], fontsize=9)

    _finish(ax, title or "Survivorship", "age", "lives remaining")
    _money(ax, "y")
    return fig


def mortality(table, log_y: bool = True, title: Optional[str] = None, figsize=(9, 5)):
    """Mortality rates ``q_x`` by age, on a log scale.

    Log by default because mortality is close to exponential in age over most
    of the range — that is Gompertz's law — and on a log axis it is close to a
    straight line, so departures from it are visible instead of buried.
    """
    c = _colours()
    ages, qx = table.mortality_curve()

    fig, ax = _figure(figsize)
    ax.plot(ages, qx, color=c["series"][0])
    if log_y:
        ax.set_yscale("log")
    _finish(ax, title or "Mortality rate by age", "age", "q(x)")
    return fig


def reserve_profile(profile, title: Optional[str] = None, figsize=(9, 5)):
    """A policy's net premium reserve over its life.

    Takes the ``[(t, reserve), …]`` from :func:`riskpy.life.reserve_profile`.
    The shape is the story: a reserve builds while premiums exceed the cost of
    cover and releases as the risk runs off, and an endowment's reserve has to
    reach the sum assured exactly at maturity.
    """
    np = _numpy()
    c = _colours()
    times = [t for t, _ in profile]
    values = [v for _, v in profile]

    fig, ax = _figure(figsize)
    ax.plot(times, values, color=c["series"][0])
    ax.fill_between(times, values, color=c["series"][0], alpha=0.15, linewidth=0)
    ax.axhline(0.0, color=c["axis"], linewidth=1.0)

    peak = int(np.argmax(values))
    ax.plot([times[peak]], [values[peak]], marker="o", markersize=8, color=c["series"][0],
            markeredgecolor=c["surface"], markeredgewidth=2)
    ax.annotate(f"peak {values[peak]:,.4f} at t={times[peak]}", xy=(times[peak], values[peak]),
                xytext=(8, -14), textcoords="offset points", color=c["text"], fontsize=9)

    _finish(ax, title or "Net premium reserve", "policy year", "reserve per unit sum assured")
    return fig


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------


def curve(yield_curve, maturities: Optional[Sequence[float]] = None, forwards: bool = True,
          title: Optional[str] = None, figsize=(9, 5)):
    """Zero and one-year forward rates against maturity.

    Two rates on one axis, which is legitimate because they are the same
    quantity in the same units — unlike the dual-axis charts this module
    refuses to draw. The forward curve is the more informative of the two: it
    is what the zero curve is implicitly forecasting, and it shows kinks the
    zero curve smooths over.
    """
    np = _numpy()
    c = _colours()
    if maturities is None:
        maturities = np.linspace(0.25, 30.0, 120)
    grid = np.asarray(maturities, dtype=float)

    fig, ax = _figure(figsize)
    zeros = np.array([yield_curve.zero(float(t)) for t in grid])
    ax.plot(grid, zeros, color=c["series"][0], label="zero rate")

    if forwards:
        fwd = np.array([yield_curve.forward(float(t), float(t) + 1.0) for t in grid])
        ax.plot(grid, fwd, color=c["series"][1], label="1y forward")
        ax.legend(loc="best")

    _percent(ax, "y")
    _finish(ax, title or "Yield curve", "maturity (years)", "rate")
    return fig


# ---------------------------------------------------------------------------
# Capital
# ---------------------------------------------------------------------------


def allocation(shares: Mapping[str, float], total: Optional[float] = None,
               title: str = "Capital allocation", figsize=(9, 5)):
    """Allocated capital by business unit.

    One hue: the unit names carry the identity, so tinting the bars by size
    would spend the colour channel restating the length. Shares are labelled
    directly because the whole point of an allocation is the number.
    """
    np = _numpy()
    c = _colours()
    if not shares:
        raise ValueError("nothing to allocate")

    names = list(shares)
    values = np.array([float(shares[n]) for n in names], dtype=float)
    order = np.argsort(values)
    names = [names[i] for i in order]
    values = values[order]
    grand = float(values.sum()) if total is None else float(total)

    y = np.arange(len(names))
    fig, ax = _figure(figsize)
    ax.barh(y, values, height=0.6, color=c["series"][0], edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=c["text"], fontsize=10)
    ax.grid(axis="y", visible=False)

    span = float(values.max()) if values.size else 1.0
    for yi, value in zip(y, values):
        share = value / grand if grand else 0.0
        ax.text(value + 0.015 * span, yi, f"{value:,.0f}   {share:.1%}",
                va="center", ha="left", color=c["text"], fontsize=9)
    ax.set_xlim(min(0.0, float(values.min()) * 1.1), span * 1.28)

    _finish(ax, f"{title} — total {grand:,.0f}", "allocated capital")
    _money(ax, "x")
    return fig


def waterfall(items: Mapping[str, float], start: float = 0.0, title: str = "Contributions",
              figsize=(9, 5.5)):
    """How a set of increments builds to a total.

    Increases and decreases are opposed hues from the diverging ramp *and* are
    signed in the label, so the direction never rests on colour alone. The
    final bar is the total, drawn from the baseline.
    """
    np = _numpy()
    c = _colours()
    if not items:
        raise ValueError("nothing to chart")

    names = list(items) + ["total"]
    deltas = np.array([float(items[n]) for n in items], dtype=float)
    bottoms = start + np.concatenate([[0.0], np.cumsum(deltas)[:-1]])
    total = start + float(deltas.sum())

    fig, ax = _figure(figsize)
    up, down = c["diverging"][-1], c["diverging"][0]
    for i, (delta, bottom) in enumerate(zip(deltas, bottoms)):
        ax.bar(i, delta, bottom=bottom, width=0.62,
               color=up if delta >= 0 else down, edgecolor="none")
        ax.text(i, bottom + delta + (0.01 if delta >= 0 else -0.01) * abs(total or 1.0),
                f"{delta:+,.0f}", ha="center",
                va="bottom" if delta >= 0 else "top", color=c["text"], fontsize=9)
    ax.bar(len(deltas), total, width=0.62, color=c["series"][0], edgecolor="none")
    ax.text(len(deltas), total, f"{total:,.0f}", ha="center", va="bottom",
            color=c["text"], fontsize=9, fontweight="semibold")

    ax.axhline(start, color=c["axis"], linewidth=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", color=c["text"], fontsize=9)
    ax.grid(axis="x", visible=False)
    _finish(ax, title, None, "amount")
    _money(ax, "y")
    return fig


# ---------------------------------------------------------------------------


def save(fig, path: str, dpi: int = 160) -> str:
    """Write a figure to disk and return the path."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
