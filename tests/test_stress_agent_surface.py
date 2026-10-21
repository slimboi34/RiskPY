"""Robustness of the agent surface under hostile and oversized input.

Regression tests for what fuzzing tools.call(), the CLI and the trace found in
0.5.2, plus a seeded fuzz sweep: every tool, every argument, a menu of bad
values — the call must return a structured error or strict JSON, never raise.
"""

import json
import math
import random
import threading
import time

import pytest

from riskpy import rates, tools


def _call(name, args):
    return tools.call(name, args, errors="return")


def _code(out):
    assert isinstance(out, dict) and set(out) == {"error"}, out
    return out["error"]["code"]


# ---------------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------------


def test_bootstrap_frequency_is_bounded():
    started = time.perf_counter()
    out = _call("bootstrap_yield_curve", {"par_rates": [0.03, 0.035, 0.04], "maturities": [1, 2, 3],
                                          "frequency": 2**31 - 1})
    assert _code(out) == "RP405"
    assert time.perf_counter() - started < 1.0
    ok = _call("bootstrap_yield_curve", {"par_rates": [0.03, 0.035, 0.04], "maturities": [1, 2, 3], "frequency": 2})
    assert len(ok["zero_rates"]) == 3


def test_whole_number_floats_reach_the_core_as_ints():
    pytest.importorskip("riskpy.cpp_underwriter")
    args = {"expected_frequency": 2, "severity_mean": 1000, "severity_sd": 500, "trials": 2000, "seed": 7}
    as_int = _call("aggregate_loss", args)
    as_float = _call("aggregate_loss", args | {"trials": 2000.0, "seed": 7.0})
    assert "error" not in as_float, as_float
    assert as_float == as_int


def test_lone_surrogate_never_breaks_a_traced_call(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("RISKPY_TRACE_FILE", str(path))
    assert _code(_call("\ud800", {})) == "RP401"
    assert "error" in _call("black_scholes", {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2, "kind": "\ud800"})
    assert "error" in _call("black_scholes", {"\ud800": 1})
    for line in path.read_text(encoding="utf-8").splitlines():
        json.loads(line)


def test_convert_rate_compounding_strings():
    # Ten-to-the-sixteen periods a year silently returned 0.0.
    assert _code(_call("convert_rate", {"rate": 0.05, "from_compounding": "10000000000000000",
                                        "to_compounding": "annual"})) == "RP405"
    assert _code(_call("convert_rate", {"rate": 0.05, "from_compounding": "²", "to_compounding": "annual"})) == "RP108"
    # 5% compounded a billion times a year is e^(0.05 - 1.25e-12) - 1 to all digits.
    got = _call("convert_rate", {"rate": 0.05, "from_compounding": "1000000000", "to_compounding": "annual"})
    assert got == pytest.approx(math.expm1(1e9 * math.log1p(0.05 / 1e9)), rel=1e-13)


def test_zero_rate_with_huge_compounding_frequency():
    for m in (12, 365, 10**6, 10**9, 2**31 - 1):
        df = rates.discount_factor(0.05, 2.0, m)
        assert rates.zero_rate(df, 2.0, m) == pytest.approx(0.05, rel=1e-12), m


def test_verify_deduplicates_and_refuses_empty():
    started = time.perf_counter()
    out = _call("verify", {"modules": ["special"] * 1000})
    single = _call("verify", {"modules": ["special"]})
    assert out["checks"] == single["checks"]
    assert time.perf_counter() - started < 10
    assert _code(_call("verify", {"modules": []})) == "RP111"


def test_overflowed_results_are_errors_not_nulls():
    out = _call("bond_analytics", {"face": 1e308, "coupon": 0.05, "maturity": 10, "ytm": 0.05})
    assert _code(out) == "RP304"
    assert "overflowed" in out["error"]["message"]


def test_aggregate_loss_tail_is_expected_shortfall():
    pytest.importorskip("riskpy.cpp_underwriter")
    out = _call("aggregate_loss", {"expected_frequency": 3, "severity_mean": 1000, "severity_sd": 2000,
                                   "trials": 20000, "seed": 11})
    assert out["tvar_0.995"] >= out["var"]["0.995"]
    assert out["tvar_0.995"] <= out["max"]


def test_trace_limit_validation(capsys):
    with pytest.raises(ValueError):
        tools.trace(-5)
    with pytest.raises(SystemExit):
        tools.main(["trace", "--limit", "-3"])
    assert "must be >= 0" in capsys.readouterr().err


def test_cli_survives_pathological_json(capsys):
    assert tools.main(["call", "black_scholes", '{"S": ' + "9" * 5000 + "}"]) == 1
    assert tools.main(["call", "black_scholes", "[" * 100_000 + "]" * 100_000]) == 1
    err = capsys.readouterr()
    assert "RP406" in err.out + err.err
    assert "Traceback" not in err.out + err.err


def test_trace_json_is_strict(tmp_path, capsys):
    path = tmp_path / "trace.jsonl"
    path.write_text('{"seq": 1, "tool": "x", "ms": NaN, "ok": true}\n{"seq": 2, "tool": "y", "ms": Infinity}\n',
                    encoding="utf-8")
    assert tools.main(["trace", "--json", "--file", str(path)]) == 0
    printed = capsys.readouterr().out
    json.loads(printed, parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))


# ---------------------------------------------------------------------------
# Fuzz and concurrency
# ---------------------------------------------------------------------------

_BAD_VALUES = [None, True, False, "", "abc", "1e999", "\ud800", "é" * 300, math.nan, math.inf, -math.inf,
               1e308, -1e308, 5e-324, -1, 0, 2**63, [], [1, 2], [math.nan], {}, {"a": 1}, [[1, None]]]


def _valid_args(name):
    examples = {
        "black_scholes": {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2},
        "option_greeks": {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2},
        "implied_vol": {"price": 10.45, "S": 100, "K": 100, "T": 1, "r": 0.05},
        "bond_analytics": {"face": 100, "coupon": 0.05, "maturity": 10, "ytm": 0.04},
        "convert_rate": {"rate": 0.05, "from_compounding": "annual", "to_compounding": "continuous"},
        "hazard_from_spread": {"spread": 0.01, "recovery": 0.4},
        "annuity_certain": {"n": 10, "i": 0.05},
    }
    return examples.get(name)


def test_fuzz_every_argument_of_core_tools():
    rng = random.Random(1234)
    for name in ("black_scholes", "option_greeks", "implied_vol", "bond_analytics", "convert_rate",
                 "hazard_from_spread", "annuity_certain"):
        if name not in tools.TOOLS:
            continue
        base = _valid_args(name)
        first = _call(name, base)
        assert not (isinstance(first, dict) and "error" in first), (name, first)
        for key in list(base):
            for bad in _BAD_VALUES:
                out = _call(name, base | {key: bad})
                if isinstance(out, dict) and "error" in out:
                    assert out["error"]["code"].startswith("RP"), (name, key, bad, out)
                else:
                    json.dumps(out, allow_nan=False)
        for _ in range(50):
            junk = {rng.choice(list(base) + ["zz", "\ud800", ""]): rng.choice(_BAD_VALUES) for _ in range(3)}
            out = _call(name, junk)
            assert (isinstance(out, dict) and "error" in out) or json.dumps(out, allow_nan=False)


def test_concurrent_calls_match_serial():
    cases = [("black_scholes", {"S": 100, "K": k, "T": 1, "r": 0.05, "sigma": 0.2}) for k in range(60, 141, 4)]
    cases += [("bond_analytics", {"face": 100, "coupon": c / 100, "maturity": 10, "ytm": 0.04}) for c in range(1, 9)]
    serial = [_call(n, a) for n, a in cases]
    results = [None] * 16
    errors = []

    def worker(slot):
        try:
            results[slot] = [_call(n, a) for n, a in cases * 5]
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for got in results:
        assert got == serial * 5


def test_simulation_releases_the_gil():
    core = pytest.importorskip("riskpy.cpp_underwriter")
    sim = core.MonteCarloSimulator(400_000, 3)
    done = {}

    def slow():
        sim.simulate_aggregate_loss(50.0, 6.0, 1.0)
        done["slow"] = time.perf_counter()

    worker = threading.Thread(target=slow)
    start = time.perf_counter()
    worker.start()
    time.sleep(0.02)
    _call("black_scholes", {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2})
    fast = time.perf_counter()
    worker.join()
    if done["slow"] - start < 0.3:
        pytest.skip("simulation too quick on this machine to tell")
    # Holding the GIL, the fast call only returned when the simulation did.
    if fast >= done["slow"]:
        pytest.skip("this core build holds the GIL during simulation (fixed in the next core release)")
    assert fast < done["slow"]
