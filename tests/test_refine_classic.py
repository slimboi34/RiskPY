"""Regression tests for the 0.8.0 refinement of the classic modules.

Each test pins a defect found by the 0.7.0 stress test: Heston and CIR at a
vanishing vol-of-vol, uncoded overflows and log-domain errors, validators that
coerced strings and bools, unvalidated trial counts and seeds, an additive
``set_premium_column``, tutorial blocks that did not run, and a publish job that
installed packages next to its OIDC token.
"""

import decimal
import fractions
import math
import os
import pathlib
import re
import subprocess
import sys
import textwrap
import warnings
import zipfile
import xml.etree.ElementTree as ET

import pytest

from riskpy import credit, life, quant, rates, tools
from riskpy.errors import InputError, RiskPyError

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _quiet_heston(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # Feller, irrelevant here
        return quant.heston_price(*args, **kwargs)


# ---------------------------------------------------------------------------
# Heston as the vol-of-vol vanishes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("xi", [1e-7, 1e-8, 1e-9, 1e-10, 1e-12, 1e-170])
def test_heston_reaches_black_scholes_at_any_small_vol_of_vol(xi):
    pytest.importorskip("numpy")
    # 0.7.0 formed b − d by subtraction and divided it by xi²: 9.84 at 1e-8, 38.65 at 1e-9,
    # 4.67 at 1e-10, against Black–Scholes' 9.4134. xi = 1e-170 squares to zero.
    bs = quant.black_scholes(S=100, K=100, T=1, r=0.03, sigma=0.2)
    h = _quiet_heston(S=100, K=100, T=1, r=0.03, v0=0.04, kappa=2.0, theta=0.04, xi=xi, rho=0.0)
    assert h == pytest.approx(bs, abs=1e-8)


@pytest.mark.parametrize("rho", [-0.9, 0.5])
def test_heston_small_vol_of_vol_with_correlation_is_black_scholes_too(rho):
    pytest.importorskip("numpy")
    # The skew is first order in xi, so at xi = 1e-10 it is invisible.
    for K in (70.0, 100.0, 140.0):
        bs = quant.black_scholes(S=100, K=K, T=2, r=0.02, sigma=0.3, q=0.01, kind="put")
        h = _quiet_heston(S=100, K=K, T=2, r=0.02, v0=0.09, kappa=1.5, theta=0.09, xi=1e-10, rho=rho, q=0.01,
                          kind="put")
        assert h == pytest.approx(bs, abs=1e-8)


@pytest.mark.parametrize("args, expected", [
    # 0.7.0 values at ordinary vol-of-vol (nodes=1024): the rewrite keeps them to rounding.
    ((100.0, 100.0, 1.0, 0.03, 0.04, 2.0, 0.04, 0.5, -0.9, 0.0, "call"), 8.881114916920616),
    ((100.0, 125.0, 1.0, 0.0, 0.04, 0.3, 0.2, 0.5, 0.0, 0.02, "call"), 2.375062467457715),
    ((100.0, 90.0, 1.0, 0.0, 0.04, 2.0, 0.2, 1.5, -0.9, 0.0, "put"), 7.838496209300899),
    ((100.0, 200.0, 0.05, 0.0, 0.04, 2.0, 0.04, 1.5, 0.7, 0.02, "call"), 0.0003491434601644007),
    ((100.0, 100.0, 0.05, 0.0, 0.04, 8.0, 0.04, 1.5, 0.0, 0.02, "put"), 1.6932342809473369),
])
def test_heston_values_at_ordinary_vol_of_vol_are_unchanged(args, expected):
    pytest.importorskip("numpy")
    assert _quiet_heston(*args, nodes=1024) == pytest.approx(expected, abs=1e-10)


def test_heston_small_vol_of_vol_matches_a_high_precision_evaluation():
    pytest.importorskip("numpy")
    # The same Simpson grid with the transform evaluated in 60-digit mpmath gives
    # 165.92698256519637; 0.7.0 returned 165.92711624 (1.3e-4 off) from cancellation.
    args = (100.0, 200.0, 20.0, 0.0, 0.09, 8.0, 0.2, 1e-4, 0.7, 0.02, "put")
    assert _quiet_heston(*args, nodes=1024) == pytest.approx(165.92698256519637, abs=1e-9)


@pytest.mark.parametrize("kappa, rho, xi", [(0.3, 0.7, 0.5), (0.3, 0.7, 1.5), (1.0, 0.5, 2.0)])
def test_heston_refuses_kappa_at_or_below_rho_xi(kappa, rho, xi):
    pytest.importorskip("numpy")
    # 0.7.0 divided 0 by 0 evaluating the transform at u = −i whenever kappa <= rho·xi.
    with pytest.raises(InputError) as info:
        _quiet_heston(S=100, K=100, T=1, r=0.03, v0=0.04, kappa=kappa, theta=0.04, xi=xi, rho=rho)
    assert info.value.code == "RP113" and info.value.param == "kappa"
    # At expiry the price is intrinsic whatever the dynamics.
    assert _quiet_heston(S=120, K=100, T=0, r=0.03, v0=0.04, kappa=kappa, theta=0.04, xi=xi, rho=rho) == 20.0


def test_heston_nodes_and_upper_are_validated():
    pytest.importorskip("numpy")
    base = dict(S=100, K=100, T=1, r=0.03, v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
    for bad, code in ((dict(nodes=2.5), "RP107"), (dict(nodes=True), "RP107"), (dict(nodes=1), "RP107"),
                      (dict(upper=0.0), "RP102"), (dict(upper="200"), "RP101")):
        with pytest.raises(InputError) as info:
            quant.heston_price(**base, **bad)
        assert info.value.code == code


# ---------------------------------------------------------------------------
# Short-rate closed forms: tiny sigma and overflow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [1e-155, 1e-160, 1e-162, 1e-170, 1e-200, 5e-324])
def test_cir_tiny_sigma_is_the_deterministic_limit(sigma):
    # 0.7.0: 0.0 at sigma = 1e-155 and 1e-160 (sigma² subnormal), ZeroDivisionError below.
    limit = rates.cir_zero_bond(0.03, 0, 5, 0.5, 0.04, 0.0)
    assert rates.cir_zero_bond(0.03, 0, 5, 0.5, 0.04, sigma) == pytest.approx(limit, rel=1e-14)
    assert limit == pytest.approx(rates.vasicek_zero_bond(0.03, 0, 5, 0.5, 0.04, 0.0), rel=1e-14)


def test_cir_is_continuous_across_the_small_sigma_switch():
    values = [rates.cir_zero_bond(0.03, 0, 30, 0.5, 0.04, s) for s in (1e-100, 1e-140, 1e-149, 1e-151, 1e-160)]
    for v in values[1:]:
        assert v == pytest.approx(values[0], rel=1e-12)


def test_cir_with_huge_kappa_and_small_sigma_reverts_instantly():
    # gap/total underflows long before sigma² does; the rate snaps to theta: P = e^(−θτ).
    assert rates.cir_zero_bond(0.03, 0, 30, 1e150, 0.04, 1e-100) == pytest.approx(math.exp(-1.2), rel=1e-14)


def test_vasicek_overflow_is_coded_and_names_sigma():
    # A 29-point absolute short-rate vol over 48.6 years: e^(σ²τ³/6) is past a double.
    with pytest.raises(InputError) as info:
        rates.vasicek_zero_bond(0.0021195672601045158, 0, 48.56649001034031, 0.0005128612544522537,
                                0.05919708236539828, 0.2941683476575762)
    assert info.value.code == "RP304" and info.value.param == "sigma"
    assert "absolute" in info.value.hint


def test_vasicek_overflow_from_a_hugely_negative_rate_names_r():
    with pytest.raises(InputError) as info:
        rates.vasicek_zero_bond(-1000.0, 0, 5, 0.5, 0.04, 0.01)
    assert info.value.code == "RP304" and info.value.param == "r"


# ---------------------------------------------------------------------------
# log(S / K) underflow
# ---------------------------------------------------------------------------


def test_black_scholes_survives_a_spot_strike_ratio_that_underflows():
    # 1e-200 / 1e200 is 0.0, and log(0) was a bare "math domain error".
    assert quant.black_scholes(1e-200, 1e200, 1, 0.05, 0.2) == 0.0
    put = quant.black_scholes(1e-200, 1e200, 1, 0.05, 0.2, kind="put")
    assert put == pytest.approx(1e200 * math.exp(-0.05), rel=1e-12)
    g = quant.greeks(5e-324, 100, 1, 0.05, 0.2)
    assert (g.price, g.delta, g.gamma) == (0.0, 0.0, 0.0)


def test_merton_survives_an_assets_debt_ratio_that_underflows():
    m = credit.merton(assets=1e-200, debt=1e200, T=1, r=0.05, sigma=0.25)
    assert m.pd == 1.0 and m.equity == 0.0
    assert math.isfinite(m.distance_to_default)


@pytest.mark.parametrize("name, args", [
    ("black_scholes", {"S": 1e-200, "K": 1e200, "T": 1, "r": 0.05, "sigma": 0.2}),
    ("black_scholes", {"S": 1e-200, "K": 1e200, "T": 1, "r": 0.05, "sigma": 0.2, "explain": True}),
    ("option_greeks", {"S": 5e-324, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2}),
    ("implied_vol", {"price": 1.0, "S": 5e-324, "K": 100, "T": 1, "r": 0.05}),
    ("merton", {"assets": 1e-200, "debt": 1e200, "T": 1, "r": 0.05, "sigma": 0.25}),
])
def test_tools_no_longer_leak_math_domain_error(name, args):
    out = tools.call(name, args, errors="return")
    error = out.get("error") if isinstance(out, dict) else None
    assert not (error and error["message"] == "math domain error")
    assert not (error and error["code"] in ("RP100", "RP900", "RP304"))


# ---------------------------------------------------------------------------
# Numbers must be numbers
# ---------------------------------------------------------------------------

_LEAN_CALLS = [
    (credit.expected_loss, dict(pd=0.02, lgd=0.45, ead=1e6)),
    (credit.unexpected_loss, dict(pd=0.02, lgd=0.45, ead=1e6)),
    (credit.merton, dict(assets=150.0, debt=100.0, T=1.0, r=0.03, sigma=0.25)),
    (credit.hazard_from_spread, dict(spread=0.01, recovery=0.4)),
    (credit.spread_from_hazard, dict(hazard=0.02, recovery=0.4)),
    (credit.cds_par_spread, dict(hazard=0.02, recovery=0.4, r=0.03, maturity=5.0)),
    (credit.asrf_conditional_pd, dict(pd=0.01, rho=0.12, level=0.999)),
    (credit.vasicek_loss_quantile, dict(level=0.999, pd=0.01, rho=0.12)),
    (credit.basel_irb_capital, dict(pd=0.01, lgd=0.45, ead=1e6, maturity=2.5)),
    (quant.black_scholes, dict(S=100.0, K=100.0, T=1.0, r=0.03, sigma=0.2, q=0.01)),
    (quant.greeks, dict(S=100.0, K=100.0, T=1.0, r=0.03, sigma=0.2, q=0.01)),
    (quant.implied_vol, dict(price=10.0, S=100.0, K=100.0, T=1.0, r=0.03, q=0.0, tolerance=1e-8)),
    (quant.norm_cdf, dict(x=0.5)),
    (quant.norm_pdf, dict(x=0.5)),
    (rates.discount_factor, dict(rate=0.05, t=2.0)),
    (rates.convert_rate, dict(rate=0.05, from_compounding="annual", to_compounding="continuous")),
    (rates.bond_price, dict(face=100.0, coupon=0.05, maturity=5.0, ytm=0.04)),
    (rates.vasicek_zero_bond, dict(r=0.03, t=0.0, T=5.0, kappa=0.5, theta=0.04, sigma=0.01)),
    (rates.cir_zero_bond, dict(r=0.03, t=0.0, T=5.0, kappa=0.5, theta=0.04, sigma=0.01)),
    (life.discount_factor, dict(i=0.05)),
    (life.annuity_certain, dict(i=0.05, n=10)),
    (life.perpetuity, dict(i=0.05)),
    (life.net_premium, dict(benefit_apv=0.2, annuity_apv=15.0)),
]
# hazard also takes a curve of (time, rate) pairs, so a bool there is RP503 (bad term structure).
_FLOAT_PARAMS = [(fn, base, p) for fn, base in _LEAN_CALLS for p in base
                 if p not in ("n", "from_compounding", "to_compounding")
                 and not (fn is credit.cds_par_spread and p == "hazard")]


@pytest.mark.parametrize("fn, base, param", _FLOAT_PARAMS,
                         ids=[f"{fn.__module__.split('.')[-1]}.{fn.__name__}-{p}" for fn, base, p in _FLOAT_PARAMS])
@pytest.mark.parametrize("bad", [True, "100", b"1", 10 ** 400, None], ids=["True", "str", "bytes", "bigint", "None"])
def test_float_parameters_refuse_non_numbers(fn, base, param, bad):
    # 0.7.0 priced True as 1.0 and "100" as 100.0, and leaked OverflowError for 10**400.
    fn(**base)
    with pytest.raises(RiskPyError) as info:
        fn(**{**base, param: bad})
    assert info.value.code == "RP101" and info.value.param == param


def test_numpy_bools_and_arrays_with_a_dimension_are_refused():
    np = pytest.importorskip("numpy")
    for bad in (np.True_, np.array(True), np.array([100.0]), np.array("100"), np.complex128(100)):
        with pytest.raises(InputError) as info:
            quant.black_scholes(bad, 100, 1, 0.03, 0.2)
        assert info.value.code == "RP101"


def test_real_numbers_of_other_types_are_still_accepted():
    np = pytest.importorskip("numpy")
    plain = quant.black_scholes(100.0, 100.0, 1.0, 0.03, 0.2)
    for spot in (np.float64(100), np.float32(100), np.int64(100), np.array(100.0), fractions.Fraction(100),
                 decimal.Decimal("100")):
        assert quant.black_scholes(spot, 100, 1, 0.03, 0.2) == plain
    assert credit.expected_loss(fractions.Fraction(1, 50), 0.45, np.int64(1_000_000)) == pytest.approx(9000.0)
    g = quant.greeks(decimal.Decimal("100"), 100, 1, 0.03, 0.2)
    assert g.price == plain


def test_norm_cdf_checks_its_argument_but_keeps_float_semantics():
    for bad in (None, "0.5", True):
        with pytest.raises(InputError) as info:
            quant.norm_cdf(bad)
        assert info.value.code == "RP101" and info.value.param == "x"
    assert quant.norm_cdf(math.inf) == 1.0 and quant.norm_cdf(-math.inf) == 0.0
    assert math.isnan(quant.norm_cdf(math.nan))
    assert quant.norm_pdf(0) == pytest.approx(1 / math.sqrt(2 * math.pi), rel=1e-15)


@pytest.mark.parametrize("bad, code", [(dict(T=None), "RP101"), (dict(T="1"), "RP101"), (dict(S=b"1"), "RP101"),
                                       (dict(tolerance=None), "RP101"), (dict(tolerance=0.0), "RP102"),
                                       (dict(max_iterations=2.5), "RP107"), (dict(max_iterations=0), "RP107")])
def test_implied_vol_validates_before_comparing(bad, code):
    # T was compared with 0 before it was checked: None gave "'<=' not supported".
    args = dict(price=10.0, S=100.0, K=100.0, T=1.0, r=0.03) | bad
    with pytest.raises(InputError) as info:
        quant.implied_vol(**args)
    assert info.value.code == code


# ---------------------------------------------------------------------------
# VaR helpers: one series, numbers only
# ---------------------------------------------------------------------------

_VAR_FUNCTIONS = [quant.historical_var, quant.parametric_var, quant.expected_shortfall]


@pytest.mark.parametrize("fn", _VAR_FUNCTIONS)
def test_var_helpers_refuse_two_dimensional_returns(fn):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(1)
    matrix = rng.normal(0, 0.01, size=(250, 3))
    for bad in (matrix, matrix[:, :1], matrix.tolist()):
        with pytest.raises(InputError) as info:
            fn(bad, 0.99)
        assert info.value.code == "RP100" and info.value.param == "returns"
    # One column of it is fine, as a 1-D array, a list or a generator.
    column = matrix[:, 0]
    assert fn(column, 0.99) == fn(column.tolist(), 0.99) == fn((x for x in column), 0.99)


@pytest.mark.parametrize("fn", _VAR_FUNCTIONS)
def test_var_helpers_refuse_text_and_text_levels(fn):
    pytest.importorskip("numpy")
    for bad in (["a", "b"], ["0.01", "-0.02"], [True, False], [None, 0.01], "0.01", 0.01):
        with pytest.raises(InputError) as info:
            fn(bad, 0.99)
        assert info.value.code == "RP101"
    for level in ("x", None, True):
        with pytest.raises(InputError) as info:
            fn([0.01, -0.02, 0.03], level)
        assert info.value.code == "RP101" and info.value.param == "level"


def test_var_helpers_accept_a_pandas_series():
    pd = pytest.importorskip("pandas")
    series = pd.Series([0.01, -0.02, 0.005, 0.03, -0.01, 0.02, -0.04, 0.0])
    assert quant.historical_var(series, 0.9) == quant.historical_var(series.tolist(), 0.9)


# ---------------------------------------------------------------------------
# Trial counts and seeds
# ---------------------------------------------------------------------------


def _simulators():
    from riskpy import reserving

    return {
        "gbm_paths": lambda **kw: quant.gbm_paths(100, 0.05, 0.2, 1, **{"steps": 4, "trials": 16, **kw}),
        "merton_jump_paths": lambda **kw: quant.merton_jump_paths(100, 0.05, 0.2, 1, 1.0, -0.05, 0.1,
                                                                  **{"steps": 4, "trials": 16, **kw}),
        "vasicek_paths": lambda **kw: rates.vasicek_paths(0.03, 0.5, 0.04, 0.01, 1, **{"steps": 4, "trials": 16, **kw}),
        "cir_paths": lambda **kw: rates.cir_paths(0.03, 0.5, 0.04, 0.01, 1, **{"steps": 4, "trials": 16, **kw}),
        "credit_portfolio_loss": lambda **kw: credit.credit_portfolio_loss(
            [0.02, 0.05], [0.4, 0.5], [1.0, 2.0], 0.1, **{"trials": 16, **kw}).values,
        "bootstrap_chain_ladder": lambda **kw: reserving.bootstrap_chain_ladder(
            reserving.genins(), **{"n": 16, **kw}).total,
    }


_SIMULATOR_NAMES = ["gbm_paths", "merton_jump_paths", "vasicek_paths", "cir_paths", "credit_portfolio_loss",
                    "bootstrap_chain_ladder"]


@pytest.mark.parametrize("name", _SIMULATOR_NAMES)
@pytest.mark.parametrize("seed, code", [(-1, "RP102"), (1.5, "RP107"), ("1", "RP107"), (True, "RP107"),
                                        ([1, -2], "RP102"), (object(), "RP107")])
def test_invalid_seeds_are_coded(name, seed, code):
    pytest.importorskip("numpy")
    with pytest.raises(InputError) as info:
        _simulators()[name](seed=seed)
    assert info.value.code == code and info.value.param == "seed"


@pytest.mark.parametrize("name", _SIMULATOR_NAMES)
def test_valid_seeds_still_reproduce(name):
    np = pytest.importorskip("numpy")
    run = _simulators()[name]
    a = run(seed=7)
    assert np.array_equal(a, run(seed=np.int64(7)))
    assert np.array_equal(a, run(seed=np.random.default_rng(7)))
    assert np.array_equal(run(seed=[1, 2]), run(seed=np.random.SeedSequence([1, 2])))


@pytest.mark.parametrize("name", ["gbm_paths", "merton_jump_paths", "vasicek_paths", "cir_paths",
                                  "credit_portfolio_loss"])
@pytest.mark.parametrize("trials", [1.5, True, None, 0, "16", 16.0])
def test_fractional_bool_or_missing_trials_are_coded(name, trials):
    pytest.importorskip("numpy")
    with pytest.raises(InputError) as info:
        _simulators()[name](trials=trials)
    assert info.value.code == "RP107" and info.value.param == "trials"


@pytest.mark.parametrize("name", ["gbm_paths", "merton_jump_paths", "vasicek_paths", "cir_paths"])
def test_fractional_steps_are_coded(name):
    np = pytest.importorskip("numpy")
    for steps in (2.5, True, None):
        with pytest.raises(InputError) as info:
            _simulators()[name](steps=steps)
        assert info.value.code == "RP107" and info.value.param == "steps"
    assert _simulators()[name](steps=np.int64(4)).shape == (16, 5)


# ---------------------------------------------------------------------------
# UnderwritingApp.set_premium_column moves the column
# ---------------------------------------------------------------------------

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _cells(path):
    with zipfile.ZipFile(path) as z:
        try:
            shared = ["".join(t.itertext())
                      for t in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS)]
        except KeyError:
            shared = []
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    out = {}
    for cell in sheet.iter("{%s}c" % _NS["m"]):
        value = cell.find("m:v", _NS)
        if value is None:
            inline = cell.find("m:is", _NS)
            if inline is None:
                continue
            out[cell.get("r")] = "".join(inline.itertext())
            continue
        out[cell.get("r")] = shared[int(value.text)] if cell.get("t") == "s" else value.text
    return out


def _app():
    try:
        from riskpy import UnderwritingApp
    except ImportError as exc:  # headless CI without Tkinter or matplotlib
        pytest.skip(f"UnderwritingApp unavailable: {exc}")
    return UnderwritingApp(title="t")


def test_set_premium_column_twice_moves_the_premium(tmp_path, capsys):
    app = _app()
    app.add_field("x", "X", "A")
    app.set_logic(lambda inputs: inputs.get("x", 0.0) * 2)
    app.set_premium_column("B", "Premium")
    app.set_premium_column("C", "Premium")
    app.add_field("y", "Y", "D")  # mapped after the move: still exported

    app.calculate_headless({"x": 5.0, "y": 1.0})
    quote = tmp_path / "quote.xlsx"
    app.export_excel_headless(str(quote))
    cells = _cells(quote)
    headers = {k: v for k, v in cells.items() if k.endswith("1") and len(k) == 2}
    assert headers == {"A1": "X", "C1": "Premium", "D1": "Y"}
    assert list(headers.values()).count("Premium") == 1
    assert cells["C2"] == "10"

    book = tmp_path / "book.xlsx"
    source = tmp_path / "b.csv"
    source.write_text("x,y\n1,0\n2,0\n")
    app.calculate_batch(str(source), str(book))
    cells = _cells(book)
    assert "B1" not in cells and cells["C1"] == "Premium" and (cells["C2"], cells["C3"]) == ("2", "4")
    capsys.readouterr()


def test_set_premium_column_same_column_relabels(tmp_path, capsys):
    app = _app()
    app.add_field("x", "X", "A")
    app.set_logic(lambda inputs: 1.0)
    exporter = app.exporter
    app.set_premium_column("B", "Premium")
    app.set_premium_column("B", "Final Premium")
    assert app.exporter is exporter  # nothing to drop, so no rebuild
    app.calculate_headless({"x": 1.0})
    quote = tmp_path / "q.xlsx"
    app.export_excel_headless(str(quote))
    cells = _cells(quote)
    assert cells["B1"] == "Final Premium" and "C1" not in cells
    capsys.readouterr()


# ---------------------------------------------------------------------------
# Every tutorial runs as written
# ---------------------------------------------------------------------------


def _tutorials():
    text = (ROOT / "docs" / "TUTORIALS.md").read_text(encoding="utf-8")
    sections = re.split(r"^## (Tutorial \d+[^\n]*)$", text, flags=re.M)
    out = []
    for title, body in zip(sections[1::2], sections[2::2]):
        blocks = re.findall(r"^```(\w*)\n(.*?)^```", body, flags=re.M | re.S)
        out.append((title.split(":")[0], blocks))
    return out


# What each tutorial prints that pins its numbers (Tutorial 4: 1,080 + 13,500 + 2,160).
_TUTORIAL_OUTPUT = {"Tutorial 4": "Processed 3 policies. Total: $16,740.00"}


@pytest.mark.parametrize("title, blocks", _tutorials(), ids=[t for t, _ in _tutorials()])
def test_tutorial_blocks_run_in_a_fresh_process(title, blocks, tmp_path):
    pytest.importorskip("numpy")
    python = [code for lang, code in blocks if lang == "python"]
    assert python, f"{title} has no Python blocks"
    for lang, code in blocks:
        if lang == "csv":
            (tmp_path / "submissions.csv").write_text(code, encoding="utf-8")
    if any("UnderwritingApp" in code for code in python):
        _app()  # skip where Tkinter is missing
    # The window is the one thing a test cannot open: run() becomes a no-op.
    prelude = textwrap.dedent("""
        try:
            import riskpy.app
            riskpy.app.UnderwritingApp.run = lambda self, *a, **k: None
        except ImportError:
            pass
    """)
    script = tmp_path / "tutorial.py"
    script.write_text(prelude + "\n".join(python), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONIOENCODING"] = "utf-8"            # the tutorials print ≈ and —; a Windows pipe is cp1252
    done = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=300, env=env)
    assert done.returncode == 0, f"{title} failed:\n{done.stderr[-2000:]}"
    assert _TUTORIAL_OUTPUT.get(title, "") in done.stdout


# ---------------------------------------------------------------------------
# The publishing job installs nothing
# ---------------------------------------------------------------------------


def test_publish_job_with_the_oidc_token_installs_nothing():
    path = ROOT / ".github" / "workflows" / "publish.yml"
    if not path.exists():
        pytest.skip("workflow files are not shipped in the sdist")
    jobs, current = {}, None
    in_jobs = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("jobs:"):
            in_jobs = True
            continue
        match = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if in_jobs and match:
            current = match.group(1)
            jobs[current] = []
        elif in_jobs and current:
            jobs[current].append(line)
    privileged = [name for name, body in jobs.items() if any("id-token: write" in l for l in body)]
    assert privileged == ["publish"]
    body = "\n".join(jobs["publish"])
    assert "pip install" not in body and "setup-python" not in body
    assert "pypa/gh-action-pypi-publish@" in body
    # The metadata check still happens, in a job that cannot mint a token.
    assert "twine check --strict" in "\n".join(jobs["check-dist"])
    assert "check-dist" in body
