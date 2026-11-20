"""Stress testing: Model.stress, Result.reverse_stress, and the sensitivity / scenario_table tools.

References are closed forms: a pinned or scaled input in a linear formula moves the output
by an amount you can write down; the conditional mean of a normal input given a normal
total is a textbook formula; the derivative of a Black–Scholes price is its delta.
"""
import math

import pytest

np = pytest.importorskip("numpy")

from riskpy import tools  # noqa: E402
from riskpy.errors import InputError, RiskPyError  # noqa: E402
from riskpy.mc import (  # noqa: E402
    Compound, Constant, LogNormal, Model, Normal, Poisson, ReverseStress, StressResult, Uniform,
)


def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def Phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _book():
    model = Model(claims=LogNormal.from_moments(1000.0, 300.0), inflation=Normal(0.04, 0.01))

    @model.formula
    def loss(claims, inflation):
        return claims * (1.0 + inflation)

    return model


# ---------------------------------------------------------------------------
# Model.stress


def test_stress_shares_the_base_draws_so_differences_are_exact():
    model = _book()
    stress = model.stress({
        "inflation 8%": {"inflation": 0.08},
        "claims +20%": {"claims": lambda x: 1.2 * x},
        "both": {"inflation": 0.08, "claims": lambda x: 1.2 * x},
    }, trials=20_000, seed=11)
    assert isinstance(stress, StressResult) and list(stress.scenarios) == ["inflation 8%", "claims +20%", "both"]
    base = stress.base
    # the base is the ordinary run on that seed
    assert np.array_equal(base.values, model.run(20_000, seed=11).values)
    claims = base.inputs["claims"]
    # pinned inflation: loss = claims · 1.08, trial for trial
    assert np.allclose(stress.scenarios["inflation 8%"].values, claims * 1.08, rtol=1e-14)
    # scaled claims: every trial is exactly 1.2× the base, so every statistic is too
    assert np.allclose(stress.scenarios["claims +20%"].values, 1.2 * base.values, rtol=1e-14)
    assert stress.delta("claims +20%", "mean") == pytest.approx(0.2 * base.mean, rel=1e-12)
    assert stress.delta("claims +20%", "var", 0.995) == pytest.approx(0.2 * base.var(0.995), rel=1e-12)
    assert stress.delta("claims +20%", "tvar", 0.99) == pytest.approx(0.2 * base.tvar(0.99), rel=1e-12)
    assert stress.delta("claims +20%", "std") == pytest.approx(0.2 * base.std, rel=1e-12)
    assert np.allclose(stress.scenarios["both"].values, 1.2 * claims * 1.08, rtol=1e-14)
    assert stress.worst("mean") == "both"
    # the inputs of a scenario are the shocked ones, the others untouched
    assert np.all(stress.scenarios["inflation 8%"].inputs["inflation"] == 0.08)
    assert stress.scenarios["inflation 8%"].inputs["claims"] is claims


def test_stress_mean_shift_against_the_closed_form():
    # E[claims·(1 + inflation)] with independent inputs is 1000·1.04; pinning inflation at 8% makes it 1000·1.08.
    # The difference in the *sample* is mean(claims)·0.08 − mean(claims·inflation): about 40, with almost no noise.
    stress = _book().stress({"inflation 8%": {"inflation": 0.08}}, trials=200_000, seed=3)
    assert stress.base.mean == pytest.approx(1040.0, abs=5 * 300.0 * 1.04 / math.sqrt(200_000))
    assert stress.delta("inflation 8%") == pytest.approx(40.0, abs=0.2)
    # two separate runs would carry sampling noise of about ±1 on that difference; the shared draws do not
    data = stress.to_dict(level=0.99)
    row = data["scenarios"]["inflation 8%"]
    assert row["change"]["mean"] == pytest.approx(stress.delta("inflation 8%"))
    assert row["relative_change"]["mean"] == pytest.approx(row["change"]["mean"] / data["base"]["mean"])
    assert data["level"] == 0.99 and data["trials"] == 200_000 and data["seed"] == 3
    assert row["var"] == pytest.approx(stress.scenarios["inflation 8%"].var(0.99))


def test_a_distribution_shock_keeps_the_ranks_and_changes_the_marginal():
    stats = pytest.importorskip("scipy.stats")
    model = _book()
    model.correlate("claims", "inflation", 0.5)
    heavier = LogNormal.from_moments(1000.0, 600.0)
    stress = model.stress({"heavier tail": {"claims": heavier}}, trials=50_000, seed=8)
    base, shocked = stress.base.inputs["claims"], stress.scenarios["heavier tail"].inputs["claims"]
    # same ranks: the worst base year is the worst stressed year, so the correlation with inflation survives
    assert stats.spearmanr(base, shocked).statistic == pytest.approx(1.0, abs=1e-12)
    assert stats.spearmanr(shocked, stress.base.inputs["inflation"]).statistic == pytest.approx(
        stats.spearmanr(base, stress.base.inputs["inflation"]).statistic, abs=1e-12)
    # new marginal: the stressed claims follow the heavier lognormal
    sigma = math.sqrt(math.log(1.0 + 0.36))
    mu = math.log(1000.0) - 0.5 * sigma * sigma
    assert stats.kstest(np.log(shocked), stats.norm(mu, sigma).cdf).statistic < 2.5 / math.sqrt(50_000)
    assert stress.delta("heavier tail", "var", 0.995) > 0.0
    # a Compound (no quantile function) works as a shock too
    comp = model.stress({"compound": {"claims": Compound(Poisson(20), LogNormal(3.0, 0.5))}}, trials=5_000, seed=1)
    assert comp.scenarios["compound"].mean == pytest.approx(20.0 * math.exp(3.125) * 1.04, rel=0.02)


def test_stress_tables_and_reproducibility():
    pd = pytest.importorskip("pandas")
    model = _book()
    spec = {"up": {"inflation": 0.08}, "down": {"inflation": 0.0}}
    one, two = model.stress(spec, 5_000, seed=5), model.stress(spec, 5_000, seed=5)
    assert one.to_dict() == two.to_dict()
    frame = one.to_frame(level=0.99)
    assert isinstance(frame, pd.DataFrame) and list(frame.index) == ["base", "up", "down"]
    assert frame.loc["up", "mean"] > frame.loc["base", "mean"] > frame.loc["down", "mean"]
    text = one.summary(level=0.99)
    assert "Stress test" in text and "up" in text and "Δ VaR" in text
    assert one.worst("var", 0.99) == "up"
    # lhs sampling and a formula passed as fn= work as for run()
    lhs = Model(x=Uniform(0.0, 1.0)).stress({"half": {"x": 0.5}}, 1_000, seed=1, fn=lambda x: 2.0 * x, sampling="lhs")
    assert lhs.base.mean == pytest.approx(1.0, abs=1e-3) and lhs.scenarios["half"].mean == 1.0


def test_stress_refuses_bad_scenarios():
    model = _book()
    cases = [
        (lambda: model.stress({}, 100), "RP111", "scenarios"),
        (lambda: model.stress({"s": {"rates": 0.1}}, 100), "RP108", "scenarios"),
        (lambda: model.stress({"s": {"inflation": "8%"}}, 100), "RP404", "scenarios"),
        (lambda: model.stress({"s": 0.08}, 100), "RP404", "scenarios"),
        (lambda: model.stress({"s": {"inflation": lambda x: x[:10]}}, 100), "RP100", "scenarios"),
        (lambda: model.stress({"s": {"inflation": float("inf")}}, 100), "RP101", None),
        (lambda: model.stress({"s": {"inflation": 0.1}}, 0), "RP107", "trials"),
        (lambda: Model(x=Constant(1.0)).stress({"s": {"x": 2.0}}, 100), "RP113", "fn"),
        (lambda: model.stress({"s": {"inflation": 0.1}}, 100).delta("nope"), "RP108", "scenario"),
        (lambda: model.stress({"s": {"inflation": 0.1}}, 100).delta("s", "median"), "RP108", "statistic"),
        (lambda: model.stress({"s": {"claims": -5.0}}, 100, fn=lambda claims, inflation: np.where(claims < 0.0, np.inf, claims)),
         "RP305", "fn"),
    ]
    for make, code, param in cases:
        with pytest.raises(RiskPyError) as exc:
            make()
        assert exc.value.code == code and (param is None or exc.value.param == param), (code, param, exc.value)


# ---------------------------------------------------------------------------
# Result.reverse_stress


def test_reverse_stress_matches_the_normal_conditional_mean():
    # X ~ N(10, 2²), Y ~ N(5, 1²), independent; S = X + Y. E[X | S ≥ VaR_α] = 10 + 4/√5 · φ(z)/(1 − α).
    model = Model(x=Normal(10.0, 2.0), y=Normal(5.0, 1.0))
    result = model.run(1_000_000, seed=20261120, fn=lambda x, y: x + y)
    level = 0.99
    rs = result.reverse_stress(level)
    assert isinstance(rs, ReverseStress) and rs.tail == "upper" and rs.level == level
    ratio = phi(2.3263478740408408) / (1.0 - level)
    sd_s = math.sqrt(5.0)
    assert rs.inputs["x"]["tail_mean"] == pytest.approx(10.0 + 4.0 / sd_s * ratio, abs=0.06)
    assert rs.inputs["y"]["tail_mean"] == pytest.approx(5.0 + 1.0 / sd_s * ratio, abs=0.05)
    assert rs.scenario == {"x": rs.inputs["x"]["tail_mean"], "y": rs.inputs["y"]["tail_mean"]}
    # shift in sd units: (Cov/σ_S · ratio)/σ_i — x moves 2/√5·ratio sds, y moves 1/√5·ratio
    assert rs.inputs["x"]["shift_sd"] == pytest.approx(2.0 / sd_s * ratio, abs=0.03)
    assert rs.inputs["y"]["shift_sd"] == pytest.approx(1.0 / sd_s * ratio, abs=0.05)
    assert [name for name, _ in rs.drivers] == ["x", "y"]
    # the percentile of the tail mean in the input's own distribution: Φ(shift)
    assert rs.inputs["x"]["percentile"] == pytest.approx(Phi(2.0 / sd_s * ratio), abs=0.003)
    # the tail is the worst 1%: its mean output is the TVaR, and the scenario adds up to it
    assert rs.probability == pytest.approx(0.01, abs=2e-6) and rs.tail_trials == 10_001
    assert rs.output_mean == pytest.approx(result.tvar(level), rel=1e-4)
    assert sum(rs.scenario.values()) == pytest.approx(rs.output_mean, rel=1e-12)
    assert rs.threshold == pytest.approx(15.0 + sd_s * 2.3263478740408408, abs=0.045)


def test_reverse_stress_by_threshold_and_in_the_lower_tail():
    model = Model(x=Normal(0.0, 1.0), y=Normal(0.0, 1.0))
    result = model.run(400_000, seed=2, fn=lambda x, y: x + y)
    # P(S ≥ 2) = 1 − Φ(2/√2); E[X | S ≥ t] = ½·E[S | S ≥ t] = (1/√2)·φ(t/√2)/(1 − Φ(t/√2))
    rs = result.reverse_stress(threshold=2.0)
    p = 1.0 - Phi(math.sqrt(2.0))
    assert rs.level is None and rs.threshold == 2.0
    assert rs.probability == pytest.approx(p, abs=5 * math.sqrt(p / 400_000))
    assert rs.inputs["x"]["tail_mean"] == pytest.approx(phi(math.sqrt(2.0)) / p / math.sqrt(2.0), abs=0.02)
    # by symmetry the lower tail is the mirror image
    low = result.reverse_stress(threshold=-2.0, tail="lower")
    assert low.inputs["x"]["tail_mean"] == pytest.approx(-phi(math.sqrt(2.0)) / p / math.sqrt(2.0), abs=0.02)
    assert low.inputs["x"]["shift_sd"] < 0.0
    by_level = result.reverse_stress(0.99, tail="lower")
    assert by_level.threshold == pytest.approx(-math.sqrt(2.0) * 2.3263478740408408, abs=0.045)
    assert by_level.probability == pytest.approx(0.01, abs=1e-5)
    data = rs.to_dict()
    assert data["drivers"][0][0] in ("x", "y") and data["tail_trials"] == rs.tail_trials
    assert "Reverse stress" in rs.summary() and "x" in rs.summary()


def test_reverse_stress_finds_the_driver_and_refuses_what_it_cannot_do():
    model = _book()
    result = model.run(100_000, seed=4)
    rs = result.reverse_stress(0.995)
    # claims (cv 30%) drive the loss; inflation (±1 point on a 1.04 factor) barely matters
    assert rs.drivers[0][0] == "claims" and rs.inputs["claims"]["shift_sd"] > 2.5
    assert abs(rs.inputs["inflation"]["shift_sd"]) < 0.5
    assert rs.inputs["claims"]["percentile"] > 0.99
    for make, code, param in (
        (lambda: result.reverse_stress(99.5), "RP110", None),
        (lambda: result.reverse_stress(threshold=1e12), "RP111", "threshold"),
        (lambda: result.reverse_stress(tail="both"), "RP108", "tail"),
        (lambda: result.reverse_stress(threshold=float("nan")), "RP101", "threshold"),
        (lambda: model.run(100, seed=1, keep_inputs=False).reverse_stress(), "RP113", "keep_inputs"),
    ):
        with pytest.raises(InputError) as exc:
            make()
        assert exc.value.code == code and (param is None or exc.value.param == param), (code, exc.value)


# ---------------------------------------------------------------------------
# The sensitivity tool: derivatives against the analytic ones


BS = {"S": 100.0, "K": 105.0, "T": 0.75, "r": 0.04, "sigma": 0.25, "q": 0.01}


def _bs_greeks(S, K, T, r, sigma, q):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    price = S * math.exp(-q * T) * Phi(d1) - K * math.exp(-r * T) * Phi(d2)
    return {
        "price": price,
        "delta": math.exp(-q * T) * Phi(d1),
        "gamma": math.exp(-q * T) * phi(d1) / (S * sigma * math.sqrt(T)),
        "vega": S * math.exp(-q * T) * phi(d1) * math.sqrt(T),
        "rho": K * T * math.exp(-r * T) * Phi(d2),
        "volga": S * math.exp(-q * T) * phi(d1) * math.sqrt(T) * d1 * d2 / sigma,
    }


def test_sensitivity_of_black_scholes_is_its_greeks():
    g = _bs_greeks(**BS)
    out = tools.call("sensitivity", {"tool": "black_scholes", "arguments": BS, "parameter": "S"})
    row = out["outputs"]["result"]
    assert out["scheme"] == "central" and out["value"] == 100.0 and out["bump"] == pytest.approx(0.01)
    assert row["base"] == pytest.approx(g["price"], abs=1e-12)
    assert row["derivative"] == pytest.approx(g["delta"], abs=1e-8)
    assert row["second_derivative"] == pytest.approx(g["gamma"], rel=1e-4)
    assert row["elasticity"] == pytest.approx(g["delta"] * 100.0 / g["price"], rel=1e-7)
    assert row["up"] > row["base"] > row["down"]
    vol = tools.call("sensitivity", {"tool": "black_scholes", "arguments": BS, "parameter": "sigma"})
    assert vol["outputs"]["result"]["derivative"] == pytest.approx(g["vega"], rel=1e-7)        # per 1.00 of vol
    assert vol["outputs"]["result"]["second_derivative"] == pytest.approx(g["volga"], rel=1e-4)
    rate = tools.call("sensitivity", {"tool": "black_scholes", "arguments": BS, "parameter": "r", "bump": 1e-4})
    assert rate["outputs"]["result"]["derivative"] == pytest.approx(g["rho"], rel=1e-7)        # per 1.00 of rate
    assert rate["bump"] == 1e-4
    # a relative bump: 1% of S
    rel = tools.call("sensitivity", {"tool": "black_scholes", "arguments": BS, "parameter": "S", "bump": 0.01,
                                     "relative": True})
    assert rel["bump"] == pytest.approx(1.0) and rel["outputs"]["result"]["derivative"] == pytest.approx(
        g["delta"], abs=2e-4)        # a bump of 1.00 carries a truncation error of h²/6·f‴
    # q was not in a second call's arguments: its default (0) is bumped
    no_q = {k: v for k, v in BS.items() if k != "q"}
    dq = tools.call("sensitivity", {"tool": "black_scholes", "arguments": no_q, "parameter": "q"})
    d1 = (math.log(100.0 / 105.0) + (0.04 + 0.5 * 0.0625) * 0.75) / (0.25 * math.sqrt(0.75))
    assert dq["value"] == 0.0 and dq["outputs"]["result"]["derivative"] == pytest.approx(-0.75 * 100.0 * Phi(d1), rel=1e-6)
    assert dq["outputs"]["result"]["elasticity"] == 0.0


def test_sensitivity_of_a_bond_price_is_its_duration_and_convexity():
    face, coupon, maturity, ytm, m = 100.0, 0.05, 10.0, 0.04, 2
    out = tools.call("sensitivity", {"tool": "bond_analytics", "parameter": "ytm",
                                     "arguments": {"face": face, "coupon": coupon, "maturity": maturity, "ytm": ytm}})
    # the cash flows, discounted and differentiated by hand
    flows = [(k / m, face * coupon / m + (face if k == maturity * m else 0.0)) for k in range(1, int(maturity * m) + 1)]
    price = sum(cf * (1.0 + ytm / m) ** (-m * t) for t, cf in flows)
    d_price = sum(-t * cf * (1.0 + ytm / m) ** (-m * t - 1.0) for t, cf in flows)
    d2_price = sum(t * (t + 1.0 / m) * cf * (1.0 + ytm / m) ** (-m * t - 2.0) for t, cf in flows)
    row = out["outputs"]["price"]
    assert row["base"] == pytest.approx(price, rel=1e-12)
    assert row["derivative"] == pytest.approx(d_price, rel=1e-6)
    assert row["second_derivative"] == pytest.approx(d2_price, rel=1e-4)
    # every numeric output is differentiated, each under its own key
    assert set(out["outputs"]) == {"price", "macaulay_duration", "modified_duration", "convexity", "dv01"}
    assert out["outputs"]["modified_duration"]["base"] == pytest.approx(-d_price / price, rel=1e-9)
    assert out["outputs"]["modified_duration"]["derivative"] < 0.0        # duration falls as the yield rises


def test_sensitivity_handles_whole_numbers_edges_and_simulations():
    # a whole-number parameter is bumped by 1: ä-certain for 9, 10, 11 years at 5%
    out = tools.call("sensitivity", {"tool": "annuity_certain", "arguments": {"i": 0.05, "n": 10}, "parameter": "n"})
    a = lambda n: (1.0 - 1.05 ** -n) / 0.05  # noqa: E731
    row = out["outputs"]["result"]
    assert out["bump"] == 1 and row["down"] == pytest.approx(a(9)) and row["up"] == pytest.approx(a(11))
    assert row["derivative"] == pytest.approx((a(11) - a(9)) / 2.0)
    assert row["second_derivative"] == pytest.approx(a(11) - 2.0 * a(10) + a(9))
    # at sigma = 0 the down-bump is refused, so a one-sided three-point difference is used
    edge = tools.call("sensitivity", {"tool": "black_scholes", "parameter": "sigma",
                                      "arguments": {"S": 100, "K": 120, "T": 1, "r": 0.0, "sigma": 0.0}})
    assert edge["scheme"] == "forward" and edge["outputs"]["result"]["down"] is None
    assert edge["outputs"]["result"]["base"] == 0.0 and edge["outputs"]["result"]["elasticity"] is None
    # a simulation with a seed: both bumps share the draws, so the derivative of the mean wrt frequency is
    # close to E[severity] = 100
    sim = {"expected_frequency": 50, "severity_mean": 100, "severity_sd": 40, "trials": 20_000, "seed": 5}
    out = tools.call("sensitivity", {"tool": "aggregate_loss", "arguments": sim, "parameter": "severity_mean",
                                     "bump": 1.0})
    assert "var.0.995" in out["outputs"] and out["outputs"]["trials"]["derivative"] == 0.0
    assert out["outputs"]["mean"]["derivative"] == pytest.approx(50.0, rel=0.05)     # d(λ·m)/dm = λ


def test_sensitivity_refusals_name_the_argument():
    cases = [
        ({"tool": "blackscholes", "arguments": BS, "parameter": "S"}, "RP401", "tool"),
        ({"tool": "riskpy_status", "arguments": {}, "parameter": "recent"}, "RP108", "tool"),
        ({"tool": "sensitivity", "arguments": {}, "parameter": "bump"}, "RP108", "tool"),
        ({"tool": "black_scholes", "arguments": BS, "parameter": "spot"}, "RP108", "parameter"),
        ({"tool": "black_scholes", "arguments": BS, "parameter": "kind"}, "RP108", "parameter"),
        ({"tool": "black_scholes", "arguments": {"S": 100}, "parameter": "S"}, "RP403", "K"),
        ({"tool": "black_scholes", "arguments": {**BS, "vol": 0.2}, "parameter": "S"}, "RP402", "vol"),
        ({"tool": "black_scholes", "arguments": {**BS, "sigma": 25}, "parameter": "S"}, "RP105", "sigma"),
        ({"tool": "black_scholes", "arguments": BS, "parameter": "S", "bump": -1}, "RP102", "bump"),
        ({"tool": "black_scholes", "arguments": BS, "parameter": "S", "relative": True}, "RP113", "bump"),
        ({"tool": "black_scholes", "arguments": "S=100", "parameter": "S"}, "RP404", "arguments"),
        ({"tool": "annuity_certain", "arguments": {"i": 0.05, "n": 10}, "parameter": "n", "bump": 0.5}, "RP107", "bump"),
        ({"tool": "aggregate_loss", "parameter": "severity_mean",
          "arguments": {"expected_frequency": 5, "severity_mean": 10, "severity_sd": 4}}, "RP113", "arguments"),
    ]
    for args, code, param in cases:
        err = tools.call("sensitivity", args, errors="return")["error"]
        assert (err["code"], err["param"]) == (code, param), (args, err)
    hint = tools.call("sensitivity", {"tool": "black_scholes", "arguments": BS, "parameter": "spot"},
                      errors="return")["error"]["hint"]
    assert "'S'" in hint


# ---------------------------------------------------------------------------
# The scenario_table tool


def test_scenario_table_revalues_each_scenario_against_the_base():
    g = _bs_greeks(**BS)
    out = tools.call("scenario_table", {"tool": "black_scholes", "arguments": BS, "scenarios": {
        "rates +100bp": {"r": 0.05},
        "vol 35%": {"sigma": 0.35},
        "crash": {"S": 80.0, "sigma": 0.45},
        "put": {"kind": "put"},
    }})
    assert out["tool"] == "black_scholes" and out["failed"] == 0 and list(out["scenarios"]) == [
        "rates +100bp", "vol 35%", "crash", "put"]
    assert out["base"]["result"] == pytest.approx(g["price"], abs=1e-12)
    crash = out["scenarios"]["crash"]
    reference = _bs_greeks(**{**BS, "S": 80.0, "sigma": 0.45})["price"]
    assert crash["result"] == pytest.approx(reference, abs=1e-12) and crash["overrides"] == {"S": 80.0, "sigma": 0.45}
    assert crash["change"]["result"] == pytest.approx(reference - g["price"], abs=1e-12)
    assert crash["relative_change"]["result"] == pytest.approx((reference - g["price"]) / g["price"], abs=1e-12)
    # put–call parity: put − call = K·e^(−rT) − S·e^(−qT)
    assert out["scenarios"]["put"]["change"]["result"] == pytest.approx(
        105.0 * math.exp(-0.03) - 100.0 * math.exp(-0.0075), abs=1e-10)
    # a 100bp move revalued in full differs from rho × 0.01 by the convexity in r
    full = out["scenarios"]["rates +100bp"]["change"]["result"]
    assert full == pytest.approx(g["rho"] * 0.01, rel=0.02) and full != pytest.approx(g["rho"] * 0.01, rel=1e-6)


def test_scenario_table_on_a_dict_result_and_with_a_refused_scenario():
    args = {"pd": 0.01, "lgd": 0.45, "ead": 1_000_000.0}
    out = tools.call("scenario_table", {"tool": "basel_irb_capital", "arguments": args, "scenarios": {
        "downturn": {"pd": 0.03, "lgd": 0.6}, "typo": {"pd": 3}, "retail": {"asset_class": "retail_other"}}})
    direct = tools.call("basel_irb_capital", {**args, "pd": 0.03, "lgd": 0.6})
    base = tools.call("basel_irb_capital", args)
    assert out["scenarios"]["downturn"]["result"] == direct and out["base"]["result"] == base
    assert out["scenarios"]["downturn"]["change"]["capital"] == pytest.approx(direct["capital"] - base["capital"])
    assert out["scenarios"]["downturn"]["relative_change"]["rwa"] == pytest.approx(direct["rwa"] / base["rwa"] - 1.0)
    # the bad scenario is reported in place, with the same coded error a direct call gives; the rest still ran
    assert out["failed"] == 1 and out["scenarios"]["typo"]["error"]["code"] == "RP103"
    assert out["scenarios"]["typo"]["error"]["param"] == "pd" and "result" not in out["scenarios"]["typo"]
    assert out["scenarios"]["retail"]["result"]["capital"] < base["capital"]
    for call_args, code, param in (
        ({"tool": "black_scholes", "arguments": BS, "scenarios": {}}, "RP111", "scenarios"),
        ({"tool": "black_scholes", "arguments": BS, "scenarios": {"a": {"vol": 0.3}}}, "RP402", "scenarios"),
        ({"tool": "black_scholes", "arguments": BS, "scenarios": {"a": 0.3}}, "RP404", "scenarios"),
        ({"tool": "black_scholes", "arguments": BS, "scenarios": [{"r": 0.05}]}, "RP404", "scenarios"),
        ({"tool": "nope", "arguments": BS, "scenarios": {"a": {}}}, "RP401", "tool"),
        ({"tool": "black_scholes", "arguments": BS, "scenarios": {str(k): {"r": 0.01} for k in range(101)}},
         "RP405", "scenarios"),
    ):
        err = tools.call("scenario_table", call_args, errors="return")["error"]
        assert (err["code"], err["param"]) == (code, param), (call_args, err)
    hint = tools.call("scenario_table", {"tool": "black_scholes", "arguments": BS, "scenarios": {"a": {"vol": 0.3}}},
                      errors="return")["error"]["hint"]
    assert "'sigma'" in hint


def test_the_meta_tools_have_object_schemas_and_explanations():
    for name in ("sensitivity", "scenario_table"):
        schema = tools.TOOLS[name].schema()
        assert schema["properties"]["arguments"]["type"] == "object" and "tool" in schema["required"]
        assert all("description" in prop for prop in schema["properties"].values())
        assert tools.call("riskpy_explain", {"topic": name})["formula"]
    assert tools.TOOLS["scenario_table"].schema()["properties"]["scenarios"]["type"] == "object"
    assert tools.TOOLS["sensitivity"].requires == "" and tools.TOOLS["sensitivity"].kind == "calculator"
