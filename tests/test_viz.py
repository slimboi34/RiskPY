"""Tests for the visualisation layer.

Charts are not checked pixel by pixel — that pins the test to a Matplotlib
version. They are checked for the things that matter and can be asserted: that
each function returns a Figure without calling show(), that the data contracts
hold, that the guard rails raise, and that the palettes keep the accessibility
properties the module claims.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

from riskpy import life, quant, rates, reserving, viz  # noqa: E402
from riskpy.mc import PERT, Constant, LogNormal, Model, Normal, Poisson  # noqa: E402

SEED = 99


@pytest.fixture(scope="module")
def result():
    model = Model(
        claim_count=Poisson(mean=40),
        severity=LogNormal.from_moments(mean=10_000, sd=20_000),
        inflation=Normal(mean=0.03, sd=0.01),
        recovery=PERT(0.0, 0.1, 0.3),
    )
    model.formula(lambda claim_count, severity, inflation, recovery: claim_count * severity * (1 + inflation) * (1 - recovery))
    return model.run(20_000, seed=SEED)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _is_figure(fig):
    assert isinstance(fig, Figure)
    return True


def test_simulation_charts(result):
    assert _is_figure(viz.distribution(result))
    assert _is_figure(viz.distribution(result, clip=None))
    assert _is_figure(viz.density(result))
    assert _is_figure(viz.cdf(result))
    assert _is_figure(viz.exceedance(result))
    assert _is_figure(viz.exceedance(result, log_y=False))
    assert _is_figure(viz.convergence(result))
    assert _is_figure(viz.tornado(result))
    assert _is_figure(viz.tornado(result, method="contribution"))
    assert _is_figure(viz.qq(result))
    assert _is_figure(viz.qq(result, dist=LogNormal.from_moments(result.mean, result.std)))
    assert _is_figure(viz.dashboard(result))
    assert _is_figure(viz.correlation(result))
    assert _is_figure(viz.correlation(np.eye(3), names=["a", "b", "c"]))
    assert _is_figure(viz.scatter(result, "claim_count", "severity"))


def test_result_plot_shortcut_reaches_every_kind(result):
    for kind in ("distribution", "exceedance", "convergence", "tornado", "cdf", "density", "qq", "dashboard"):
        assert _is_figure(result.plot(kind))


def test_comparison_charts(result):
    other = Model(x=Normal(result.mean, result.std)).run(5_000, seed=SEED, fn=lambda x: x)
    assert _is_figure(viz.compare({"simulated": result, "normal": other}))
    assert _is_figure(viz.spread({"simulated": result, "normal": other}))
    paths = quant.gbm_paths(100.0, 0.05, 0.2, 1.0, steps=50, trials=2_000, seed=SEED)
    assert _is_figure(viz.fan(paths))
    assert _is_figure(viz.fan(paths, times=np.linspace(0, 1, 51)))


def test_reserving_charts():
    tri = reserving.genins()
    assert _is_figure(viz.triangle(tri))
    assert _is_figure(viz.triangle(tri, kind="cumulative"))
    assert _is_figure(viz.triangle(tri, kind="incremental"))
    cl = reserving.chain_ladder(tri)
    assert _is_figure(viz.development(cl))
    assert _is_figure(viz.development(cl, triangle=tri))
    assert _is_figure(viz.reserve_range(reserving.mack_chain_ladder(tri)))
    assert _is_figure(viz.reserve_range(reserving.bootstrap_chain_ladder(tri, n=100, seed=SEED)))


def test_life_and_rates_charts():
    table = life.LifeTable.sult()
    assert _is_figure(viz.survival(table))
    assert _is_figure(viz.mortality(table))
    assert _is_figure(viz.reserve_profile(life.reserve_profile(table, 45, 0.05, "endowment", n=20)))
    assert _is_figure(viz.curve(rates.YieldCurve.nelson_siegel(0.04, -0.02, 0.03, 2.0)))
    assert _is_figure(viz.curve(rates.YieldCurve.flat(0.03), forwards=False))


def test_capital_charts():
    assert _is_figure(viz.allocation({"motor": 3.0, "property": 5.0, "liability": 2.0}))
    assert _is_figure(viz.waterfall({"gross": 100.0, "reinsurance": -30.0, "expenses": 12.0}))


def test_guard_rails_raise(result):
    with pytest.raises(ValueError, match="palette"):
        viz.compare({f"s{i}": result for i in range(9)})
    with pytest.raises(ValueError):
        viz.compare({})
    with pytest.raises(ValueError):
        viz.spread({})
    with pytest.raises(ValueError, match="2-D"):
        viz.fan(np.arange(10.0))
    with pytest.raises(ValueError):
        viz.fan(np.zeros((5, 4)), times=[0, 1])
    with pytest.raises(ValueError, match="mode"):
        viz.theme("sepia")
    with pytest.raises(ValueError, match="kind"):
        viz.triangle(reserving.genins(), kind="ratios")
    with pytest.raises(ValueError):
        viz.allocation({})
    with pytest.raises(ValueError):
        viz.waterfall({})
    with pytest.raises(ValueError, match="unknown variable"):
        viz.scatter(result, "claim_count", "nope")
    with pytest.raises(TypeError):
        viz.qq(result, dist=object())
    with pytest.raises(TypeError):
        viz.reserve_range(object())
    with pytest.raises(ValueError):
        viz.correlation(np.zeros((2, 3)))


def test_theme_switches_and_persists(result):
    viz.theme("light")
    try:
        fig = viz.distribution(result)
        assert fig.get_facecolor()[:3] == pytest.approx(matplotlib.colors.to_rgb(viz.PALETTE["light"]["page"]))
    finally:
        viz.theme("dark")


def test_palettes_keep_their_accessibility_claims():
    def luminance(hex_colour):
        r, g, b = (int(hex_colour[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
        lin = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
        return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)

    for mode, palette in viz.PALETTE.items():
        assert len(palette["series"]) == 8
        surface = luminance(palette["surface"])
        for colour in palette["series"]:
            hi, lo = sorted((luminance(colour), surface), reverse=True)
            assert (hi + 0.05) / (lo + 0.05) >= 3.0, (mode, colour)
        lum = [luminance(c) for c in palette["sequential"]]
        assert all(b > a for a, b in zip(lum, lum[1:])) or all(b < a for a, b in zip(lum, lum[1:]))
        assert len(palette["diverging"]) == 5


def test_compact_labels():
    assert viz._compact(0.0) == "0"
    assert viz._compact(950.0) == "950"
    assert viz._compact(8_091_985.0) == "8.1m"
    assert viz._compact(29_215_865.0) == "29.2m"
    assert viz._compact(1_500.0) == "1.5k"
    assert viz._compact(2_000_000_000.0) == "2bn"
    assert viz._compact(-4_200.0) == "-4.2k"
    assert viz._compact(0.25) == "0.25"


def test_kde_integrates_to_one_and_spread_labels_separate():
    rng = np.random.default_rng(SEED)
    grid, density = viz._kde(rng.normal(size=5_000))
    assert getattr(np, 'trapezoid', getattr(np, 'trapz', None))(density, grid) == pytest.approx(1.0, abs=0.02)
    with pytest.raises(ValueError):
        viz._kde(np.array([1.0]))
    moved = viz._spread_labels([1.0, 1.01, 1.02, 5.0], min_gap=0.5)
    assert all(b - a >= 0.5 - 1e-12 for a, b in zip(sorted(moved), sorted(moved)[1:]))
    assert moved[3] == 5.0


def test_save_writes_a_file(result, tmp_path):
    path = viz.save(viz.distribution(result), str(tmp_path / "chart.png"), dpi=60)
    assert (tmp_path / "chart.png").stat().st_size > 1_000
    assert path.endswith("chart.png")
