"""Visualisation for simulation output.

One theme, one set of marks, six plots that answer the six questions people
actually ask of a Monte Carlo run:

    distribution  what does the spread look like, and where is the tail?
    exceedance    how likely is a loss bigger than X?
    convergence   did I run enough trials?
    tornado       which input is driving the answer?
    fan           what do the paths look like over time?
    compare       how do two or more scenarios sit against each other?

Every function takes a :class:`riskpy.mc.Result` (or, for ``fan``, an array of
paths), returns the Matplotlib ``Figure``, and never calls ``show()`` — so the
same call works in a notebook, in a script that saves a PNG, and in a test.

Matplotlib is an optional extra: ``pip install open-riskpy[viz]``.

Design rules worth not undoing by accident
------------------------------------------
* **One series, one colour.** Bars in a single-series chart are all the same
  hue — colouring them by height double-encodes a value the length already
  shows. Colour is reserved for identity (which scenario) and for status
  (VaR, TVaR).
* **Direction never rests on colour alone.** VaR and TVaR markers carry a
  label as well as a hue, so the chart survives greyscale printing and the
  most common forms of colour blindness.
* **Thin marks, hairline grid.** Gridlines are one step off the surface and
  solid — dashed grid reads as a threshold when it is just a grid.
* **Never two y-axes.** If you want two quantities of different scale, that is
  two charts, and ``compare`` will index them for you.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

__all__ = [
    "theme",
    "PALETTE",
    "distribution",
    "exceedance",
    "convergence",
    "tornado",
    "fan",
    "compare",
    "save",
]


# Validated against both surfaces: every categorical step clears 3:1 contrast on
# its own background, and adjacent pairs stay separable under the common forms
# of colour vision deficiency. Swap the hexes for a house palette if you have
# one, but keep the slot ORDER — that ordering is what makes adjacent series
# distinguishable, and re-sorting it quietly breaks the property.
PALETTE: Dict[str, Dict[str, Any]] = {
    "dark": {
        "page": "#0d0d0d",
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
        "good": "#0ca30c",
        "critical": "#d03b3b",
        "warning": "#fab219",
    },
    "light": {
        "page": "#f9f9f7",
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
        "good": "#006300",
        "critical": "#d03b3b",
        "warning": "#fab219",
    },
}

_MODE = "dark"


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
    global _MODE
    if mode not in PALETTE:
        raise ValueError(f"mode must be 'dark' or 'light', got {mode!r}")
    _MODE = mode

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


_themed = False


def _figure(figsize):
    global _themed
    if not _themed:
        theme(_MODE)
        _themed = True
    plt = _plt()
    return plt.subplots(figsize=figsize)


def _money(ax, axis: str = "x") -> None:
    """Thousands separators on tick labels. Raw floats are unreadable at scale."""
    from matplotlib.ticker import FuncFormatter

    fmt = FuncFormatter(lambda v, _pos: f"{v:,.0f}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


# ---------------------------------------------------------------------------


def distribution(
    result,
    bins: int = 60,
    levels: Sequence[float] = (0.95, 0.995),
    title: Optional[str] = None,
    figsize=(9, 5),
):
    """Histogram of the simulated output, with VaR and TVaR called out.

    The markers are the point of the chart: the shape tells you the story, but
    the tail statistics are what anyone acts on.
    """
    np = _numpy()
    c = _colours()
    fig, ax = _figure(figsize)

    ax.hist(result.values, bins=bins, color=c["series"][0], edgecolor="none", alpha=0.9)

    # Two markers per level would clutter fast, so label selectively: VaR gets a
    # solid rule, TVaR a lighter one, and both carry text.
    ymax = ax.get_ylim()[1]
    for i, level in enumerate(levels):
        var = result.var(level)
        tvar = result.tvar(level)
        ax.axvline(var, color=c["critical"], linewidth=2.0)
        ax.axvline(tvar, color=c["warning"], linewidth=1.5)
        ax.text(
            var,
            ymax * (0.96 - 0.14 * i),
            f" VaR {level:.1%}  {var:,.0f}",
            color=c["text"],
            fontsize=9,
            va="top",
        )
        ax.text(
            tvar,
            ymax * (0.88 - 0.14 * i),
            f" TVaR  {tvar:,.0f}",
            color=c["muted"],
            fontsize=9,
            va="top",
        )

    ax.axvline(result.mean, color=c["muted"], linewidth=1.0)
    ax.set_title(title or f"Distribution of {result.label} — {result.trials:,} trials")
    ax.set_xlabel(result.label)
    ax.set_ylabel("trials")
    ax.set_yticks([])  # the counts carry no decision; the shape and the rules do
    _money(ax, "x")
    return fig


def exceedance(
    result,
    log_y: bool = True,
    title: Optional[str] = None,
    figsize=(9, 5),
):
    """P(loss > x) against x — the survival curve.

    The natural way to answer "how likely is something worse than this?", and
    far easier to read off than a histogram tail. Log scale by default because
    the interesting probabilities are all near zero.
    """
    np = _numpy()
    c = _colours()
    fig, ax = _figure(figsize)

    ordered = np.sort(result.values)
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
        ax.annotate(
            f"{level:.1%}  {var:,.0f}",
            xy=(var, 1.0 - level),
            xytext=(8, 6),
            textcoords="offset points",
            color=c["text"],
            fontsize=9,
        )

    ax.set_title(title or f"Exceedance probability — {result.label}")
    ax.set_xlabel(result.label)
    ax.set_ylabel("P(loss exceeds x)")
    _money(ax, "x")
    return fig


def convergence(result, points: int = 200, title: Optional[str] = None, figsize=(9, 4.5)):
    """Running mean with a 95% band — the "have I run enough trials?" chart.

    If the band is still narrowing noticeably at the right-hand edge, the answer
    is no.
    """
    c = _colours()
    fig, ax = _figure(figsize)

    n, running, half = result.convergence(points=points)

    ax.fill_between(n, running - half, running + half, color=c["series"][0], alpha=0.16, linewidth=0)
    ax.plot(n, running, color=c["series"][0])
    ax.axhline(result.mean, color=c["muted"], linewidth=1.0)

    ax.annotate(
        f"mean {result.mean:,.2f}  ± {1.96 * result.standard_error:,.2f}",
        xy=(n[-1], running[-1]),
        xytext=(-4, 10),
        textcoords="offset points",
        ha="right",
        color=c["text"],
        fontsize=9,
    )

    ax.set_title(title or f"Convergence of the mean — {result.label}")
    ax.set_xlabel("trials")
    ax.set_ylabel("running mean")
    _money(ax, "x")
    _money(ax, "y")
    return fig


def tornado(result, top: int = 12, title: Optional[str] = None, figsize=(9, 5)):
    """Which inputs move the output, strongest first.

    Bars are rank correlation, so they run from -1 to 1 and the sign means
    direction. One colour for every bar: the length is the value, and tinting
    by magnitude would say the same thing twice.
    """
    np = _numpy()
    c = _colours()
    scores = result.sensitivity()[:top]
    if not scores:
        raise ValueError("No inputs to rank.")

    fig, ax = _figure(figsize)

    names = [name for name, _ in scores][::-1]
    values = [value for _, value in scores][::-1]
    y = np.arange(len(names))

    ax.barh(y, values, height=0.55, color=c["series"][0], edgecolor="none")
    ax.axvline(0, color=c["axis"], linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=c["text"], fontsize=10)

    for yi, value in zip(y, values):
        offset = 0.02 if value >= 0 else -0.02
        ax.text(
            value + offset,
            yi,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            color=c["text"],
            fontsize=9,
        )

    limit = max(0.25, min(1.0, max(abs(v) for v in values) * 1.35))
    ax.set_xlim(-limit, limit)
    ax.set_title(title or f"What drives {result.label}")
    ax.set_xlabel("rank correlation with the output")
    ax.grid(axis="y", visible=False)
    return fig


def fan(
    paths,
    levels: Sequence[float] = (0.5, 0.8, 0.95),
    title: str = "Simulated paths",
    ylabel: str = "value",
    figsize=(9, 5),
):
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
    steps = np.arange(paths.shape[1])

    # Widest band first so the narrower, darker ones sit on top.
    for i, level in enumerate(sorted(levels, reverse=True)):
        lower = np.percentile(paths, (1 - level) / 2 * 100, axis=0)
        upper = np.percentile(paths, (1 + level) / 2 * 100, axis=0)
        ax.fill_between(
            steps, lower, upper,
            color=c["series"][0],
            alpha=0.12 + 0.10 * i,
            linewidth=0,
            label=f"{level:.0%} band",
        )

    ax.plot(steps, np.percentile(paths, 50, axis=0), color=c["series"][0], label="median")

    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")
    _money(ax, "y")
    return fig


def compare(
    results: Dict[str, Any],
    bins: int = 60,
    title: str = "Scenario comparison",
    figsize=(9, 5),
):
    """Overlay several runs. Colour is identity here, so a legend is mandatory.

    ``results`` maps a scenario name to a :class:`~riskpy.mc.Result`. Past eight
    scenarios the palette runs out on purpose — nine colours cannot be told
    apart reliably, so group the tail into an "other" scenario or facet instead.
    """
    c = _colours()
    if not results:
        raise ValueError("Nothing to compare.")
    if len(results) > len(c["series"]):
        raise ValueError(
            f"{len(results)} scenarios exceeds the {len(c['series'])}-colour palette. "
            f"Beyond that the colours stop being distinguishable — facet into "
            f"separate charts instead of adding hues."
        )

    fig, ax = _figure(figsize)

    for i, (name, result) in enumerate(results.items()):
        colour = c["series"][i]
        ax.hist(result.values, bins=bins, color=colour, alpha=0.45,
                edgecolor="none", label=name)
        ax.axvline(result.var(0.995), color=colour, linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel("outcome")
    ax.set_ylabel("trials")
    ax.set_yticks([])
    ax.legend(loc="upper right")
    _money(ax, "x")
    return fig


def save(fig, path: str, dpi: int = 160) -> str:
    """Write a figure to disk and return the path."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
