"""0.5.0: error codes, structured tool errors, the trace, riskpy-doctor, and the new tools.

The new tools are pinned to numbers computed here by an independent route — closed forms,
the Makeham law behind the SULT written out by hand, the chain-ladder definition applied to
the raw GenIns data, the standard library's NormalDist — never by calling RiskPY twice.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import pickle
import re
import subprocess
import sys
from statistics import NormalDist

import pytest

import riskpy
from riskpy import diagnostics, errors, tools
from riskpy.errors import CATALOGUE, ArgumentTypeError, ConvergenceError, InputError, RiskPyError, UnknownNameError

BS = {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2}
N = NormalDist()


# ---------------------------------------------------------------- the catalogue


def test_every_code_raised_in_the_package_is_in_the_catalogue():
    root = pathlib.Path(riskpy.__file__).parent
    used = set()
    for path in root.glob("*.py"):
        used |= set(re.findall(r"\bRP\d{3}\b", path.read_text(encoding="utf-8")))
    assert used, "no codes found — the scan is broken"
    missing = sorted(used - set(CATALOGUE))
    assert not missing, f"codes used in the source but missing from the catalogue: {missing}"


def test_catalogue_entries_are_complete_and_in_their_family():
    for code, entry in CATALOGUE.items():
        assert re.fullmatch(r"RP\d{3}", code)
        assert code[:3] in errors.FAMILIES, code
        assert entry["title"] and entry["meaning"] and entry["hint"], code
    listed = errors.catalogue()
    assert [e["code"] for e in listed] == sorted(CATALOGUE)
    assert errors.explain("rp105")["title"] == "Volatility out of range"
    with pytest.raises(UnknownNameError) as info:
        errors.explain("RP999")
    assert info.value.code == "RP407"


def test_the_error_codes_page_documents_every_code():
    page = pathlib.Path(__file__).resolve().parents[1] / "docs" / "errors.md"
    if not page.exists():
        pytest.skip("docs/ not shipped here")
    text = page.read_text(encoding="utf-8")
    missing = [c for c in CATALOGUE if f"{{#{c.lower()}}}" not in text]
    assert not missing, f"docs/errors.md lacks {missing}"


def test_errors_keep_their_builtin_types_and_serialise():
    err = InputError("sigma must be >= 0, got -0.2", code="RP105", param="sigma")
    assert isinstance(err, ValueError) and isinstance(err, RiskPyError)
    assert str(err) == "sigma must be >= 0, got -0.2"
    d = err.to_dict()
    assert d["code"] == "RP105" and d["param"] == "sigma" and d["hint"] == CATALOGUE["RP105"]["hint"]
    assert d["docs"].endswith("#rp105")
    again = pickle.loads(pickle.dumps(err))
    assert type(again) is InputError and again.code == "RP105" and again.param == "sigma"
    assert isinstance(ArgumentTypeError("x"), TypeError) and isinstance(UnknownNameError("x"), KeyError)
    assert str(UnknownNameError("unknown tool 'x'")) == "unknown tool 'x'"       # not KeyError's quoted repr
    assert isinstance(ConvergenceError("x"), ValueError)


def test_from_exception_classifies_anything_without_a_traceback():
    assert errors.from_exception(ValueError("tail must be finite"))["param"] == "tail"
    assert errors.from_exception(ValueError("tail must be finite"))["code"] == "RP100"
    assert errors.from_exception(OverflowError("(34, 'Result too large')"))["code"] == "RP304"
    assert errors.from_exception(ImportError("No module named 'matplotlib'"))["code"] == "RP202"
    weird = errors.from_exception(RuntimeError("boom"))
    assert weird["code"] == "RP900" and weird["message"] == "RuntimeError: boom"


# ---------------------------------------------------------------- the library raises coded errors


@pytest.mark.parametrize("fn, code, param", [
    (lambda: riskpy.quant.black_scholes(100, 100, 1, 0.05, -0.2), "RP105", "sigma"),
    (lambda: riskpy.quant.black_scholes(100, -1, 1, 0.05, 0.2), "RP102", "K"),
    (lambda: riskpy.quant.black_scholes(100, 100, -1, 0.05, 0.2), "RP106", "T"),
    (lambda: riskpy.quant.black_scholes(100, 100, 1, 0.05, 0.2, kind="straddle"), "RP108", "kind"),
    (lambda: riskpy.quant.implied_vol(1.0, 100, 50, 1, 0.05), "RP301", "price"),
    (lambda: riskpy.quant.implied_vol(150.0, 100, 100, 1, 0.05), "RP302", "price"),
    (lambda: riskpy.quant.implied_vol(10.0, 100, 100, 0, 0.05), "RP106", "T"),
    (lambda: riskpy.credit.expected_loss(1.5, 0.45, 1e6), "RP103", "pd"),
    (lambda: riskpy.credit.basel_irb_capital(pd=0.01, lgd=0.45, ead=1, asset_class="mortgage"), "RP108", "asset_class"),
    (lambda: riskpy.credit.basel_irb_capital(pd=0.01, lgd=0.45, ead=1, maturity=7), "RP106", "maturity"),
    (lambda: riskpy.rates.discount_factor(0.05, 1, "fortnightly"), "RP108", "compounding"),
    (lambda: riskpy.rates.YieldCurve([1, 2], [0.02]), "RP109", "zero_rates"),
    (lambda: riskpy.rates.YieldCurve([2, 1], [0.02, 0.03]), "RP112", "times"),
    (lambda: riskpy.rates.Bond(100, 0.05, 0, 2), "RP106", "maturity"),
    (lambda: riskpy.life.whole_life_annuity_due(riskpy.life.LifeTable.sult(), 10, 0.05), "RP113", "x"),
    (lambda: riskpy.life.annuity_certain(-2, 10), "RP104", "i"),
    (lambda: riskpy.life.LifeTable([0.1, 0.2]), "RP501", "qx"),
])
def test_library_errors_name_the_code_and_the_parameter(fn, code, param):
    with pytest.raises(RiskPyError) as info:
        fn()
    assert (info.value.code, info.value.param) == (code, param), info.value.to_dict()
    assert info.value.hint


def test_messages_are_unchanged_so_existing_matches_still_work():
    with pytest.raises(ValueError, match=r"pd must be in \[0, 1\], got 1.5 — probabilities here are decimals"):
        riskpy.credit.expected_loss(1.5, 0.45, 1e6)
    with pytest.raises(ValueError, match="is below intrinsic"):
        riskpy.quant.implied_vol(1.0, 100, 50, 1, 0.05)


def test_missing_extra_is_a_coded_import_error(monkeypatch):
    import builtins

    from riskpy import mc

    real = builtins.__import__

    def no_numpy(name, *a, **k):
        if name == "numpy":
            raise ImportError("No module named 'numpy'")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_numpy)
    with pytest.raises(ImportError) as info:
        mc._numpy()
    assert info.value.code == "RP201" and "open-riskpy[sim]" in info.value.hint


# ---------------------------------------------------------------- structured tool errors


@pytest.mark.parametrize("name, args, code, param", [
    ("black_scholes", {**BS, "sigma": -0.2}, "RP105", "sigma"),
    ("black_scholes", {**BS, "sigma": 20}, "RP105", "sigma"),
    ("black_scholes", {**BS, "r": 5}, "RP104", "r"),
    ("black_scholes", {**BS, "S": "100"}, "RP404", "S"),
    ("black_scholes", {**BS, "vol": 0.2}, "RP402", "vol"),
    ("black_scholes", {"S": 100, "K": 100, "T": 1, "r": 0.05}, "RP403", "sigma"),
    ("implied_vol", {"price": 200, "S": 100, "K": 100, "T": 1, "r": 0.05}, "RP302", "price"),
    ("implied_vol", {"price": 10, "S": 100, "K": 100, "T": 0, "r": 0.05}, "RP106", "T"),
    ("bond_analytics", {"face": 100, "coupon": 5, "maturity": 10, "ytm": 0.04}, "RP104", "coupon"),
    ("bond_analytics", {"face": 100, "coupon": 0.05, "maturity": 0, "ytm": 0.04}, "RP106", "maturity"),
    ("expected_loss", {"pd": 2, "lgd": 0.45, "ead": 1e6}, "RP103", "pd"),
    ("expected_loss", {"pd": 0.02, "lgd": 45, "ead": 1e6}, "RP103", "lgd"),
    ("basel_irb_capital", {"pd": 0.01, "lgd": 0.45, "ead": 1e6, "asset_class": "mortgage"}, "RP108", "asset_class"),
    ("bootstrap_yield_curve", {"par_rates": [0.02, 0.03], "maturities": [1]}, "RP109", "maturities"),
    ("aggregate_loss", {"expected_frequency": 1, "severity_mean": 1, "severity_sd": 1, "trials": 10**8}, "RP405", "trials"),
    ("astrology", {}, "RP401", "name"),
    pytest.param("chain_ladder", {"triangle": [[100.0, 150.0], [110.0, 170.0, 180.0]]}, "RP502", "triangle",
                 marks=pytest.mark.skipif(not tools._extra_installed("sim"), reason="needs NumPy")),
])
def test_tool_failures_come_back_as_structured_errors(name, args, code, param):
    out = tools.call(name, args, errors="return")
    assert set(out) == {"error"}, out
    e = out["error"]
    assert (e["code"], e["param"]) == (code, param), e
    assert e["message"] and e["hint"] and e["code"] in CATALOGUE
    json.dumps(out, allow_nan=False)


def test_structured_errors_never_raise_whatever_is_sent():
    for name, args in [(None, {}), (42, None), ("black_scholes", [1, 2]), ("black_scholes", "S=100"),
                       ("annuity_certain", {"i": -0.5, "n": 100000})]:
        out = tools.call(name, args, errors="return")
        assert set(out) == {"error"} and out["error"]["code"] in CATALOGUE, (name, args, out)
    assert tools.call("black_scholes", BS, errors="return") == pytest.approx(10.4506, abs=1e-4)


def test_unavailable_tool_names_the_package_and_the_pip_line(monkeypatch):
    monkeypatch.setattr(tools, "_extra_installed", lambda extra: False)
    e = tools.call("chain_ladder", {"triangle": [[1.0]]}, errors="return")["error"]
    assert e["code"] == "RP201" and "numpy" in e["message"] and e["hint"] == 'pip install "open-riskpy[sim]"'


def test_hints_suggest_the_right_name():
    out = tools.call("black_scholes", {**BS, "volatility": 0.2}, errors="return")["error"]
    assert "'sigma'" in out["hint"]
    assert "'black_scholes'" in tools.call("black_schole", BS, errors="return")["error"]["hint"]
    assert "spot price" in tools.call("black_scholes", {"K": 100, "T": 1, "r": 0.05, "sigma": 0.2},
                                      errors="return")["error"]["hint"]


def test_plausibility_guards_leave_real_inputs_alone():
    assert tools.call("black_scholes", {**BS, "r": -0.005, "sigma": 1.5}) > 0         # negative rates, 150% vol
    assert tools.call("annuity_certain", {"i": 1.0, "n": 3}) == pytest.approx(1 / 2 + 1 / 4 + 1 / 8)
    with pytest.raises(ValueError, match="looks like a percentage"):
        tools.black_scholes(100, 100, 1, 5, 0.2)                                      # direct calls too
    assert tools.PLAUSIBLE == {"rate": 1.0, "volatility": 5.0, "loss_ratio": 5.0}


def test_raise_mode_keeps_the_old_exception_types():
    with pytest.raises(KeyError):
        tools.call("astrology", {})
    with pytest.raises(TypeError):
        tools.call("black_scholes", {**BS, "S": "100"})
    with pytest.raises(ValueError) as info:
        tools.call("black_scholes", {**BS, "sigma": -1})
    assert info.value.code == "RP105"


# ---------------------------------------------------------------- the trace


def test_trace_records_calls_with_codes_and_durations():
    tools.trace(clear=True)
    tools.call("black_scholes", BS)
    tools.call("black_scholes", {**BS, "sigma": -1}, errors="return")
    tools.call("astrology", {}, errors="return")
    tools.call("riskpy_explain", {"topic": "RP105"})                 # introspection is not traced
    entries = tools.trace()
    assert [e["tool"] for e in entries] == ["black_scholes", "black_scholes", "astrology"]
    ok, bad, unknown = entries
    assert ok["ok"] and ok["code"] is None and ok["ms"] >= 0 and ok["args"]["sigma"] == 0.2 and ok["via"] == "python"
    assert (bad["ok"], bad["code"], bad["param"]) == (False, "RP105", "sigma")
    assert unknown["code"] == "RP401"
    assert entries[0]["seq"] < entries[1]["seq"] < entries[2]["seq"]
    s = tools.status()
    assert s["trace"]["calls"] == 3 and s["trace"]["errors"] == 2 and s["trace"]["by_code"] == {"RP105": 1, "RP401": 1}
    assert tools.trace(clear=True) and tools.trace() == []


def test_trace_summarises_big_arguments():
    tools.trace(clear=True)
    tools.call("chain_ladder", {"triangle": [[1.0, 2.0]] * 50}, errors="return")
    assert tools.trace()[-1]["args"]["triangle"] == "<array of 50>"


def test_trace_file_and_cli(tmp_path, monkeypatch, capsys):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("RISKPY_TRACE_FILE", str(path))
    tools.trace(clear=True)
    assert tools.main(["call", "black_scholes", json.dumps({**BS, "sigma": -1})]) == 1
    capsys.readouterr()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["code"] == "RP105" and json.loads(lines[-1])["via"] == "cli"
    tools.trace(clear=True)
    assert tools.main(["trace", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[-1]["code"] == "RP105"
    assert tools.main(["trace"]) == 0
    assert "RP105" in capsys.readouterr().out


# ---------------------------------------------------------------- CLIs


def test_cli_prints_the_error_as_json_and_a_line_on_stderr(capsys):
    assert tools.main(["call", "black_scholes", json.dumps({**BS, "r": 5})]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["error"]["code"] == "RP104"
    assert "riskpy-tools: RP104 (r)" in captured.err and "hint:" in captured.err
    assert tools.main(["call", "black_scholes", "{S: 100}"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "RP406"
    assert tools.main(["schema", "astrology"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "RP401"


def test_explain_cli(capsys):
    assert errors.main(["RP105"]) == 0
    assert "Volatility out of range" in capsys.readouterr().out
    assert errors.main(["RP999"]) == 1
    assert "RP407" in capsys.readouterr().err
    assert errors.main(["--list", "--json"]) == 0
    assert {e["code"] for e in json.loads(capsys.readouterr().out)} == set(CATALOGUE)
    assert errors.main(["--list"]) == 0
    assert "RP1xx" in capsys.readouterr().out


# ---------------------------------------------------------------- doctor


def test_doctor_reports_every_check_and_passes_here():
    report = riskpy.doctor()
    assert report["ok"] is True, [c for c in report["checks"] if c["status"] == "fail"]
    ids = {c["id"] for c in report["checks"]}
    assert {"python", "riskpy_version", "core_import", "core_compute", "extra_numpy", "extra_mcp", "selfcheck",
            "tools", "mcp_server", "platform"} <= ids
    for c in report["checks"]:
        assert set(c) >= {"id", "status", "message", "hint"} and c["status"] in ("ok", "warn", "fail")
        assert c["status"] == "ok" or c["hint"], c
    assert report["summary"]["fail"] == 0
    json.dumps(report)


def test_doctor_fails_and_exits_1_when_the_self_check_fails(monkeypatch, capsys):
    from riskpy import verify

    real = verify.run

    def broken(modules=None, **kw):
        report = real(modules, **kw)
        bad = verify.Check("quant", "planted failure", 1.0, 2.0, 0.0)
        return verify.Report(checks=report.checks + [bad], seconds=report.seconds, version=report.version)

    monkeypatch.setattr(verify, "run", broken)
    assert diagnostics.main(["--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    selfcheck = next(c for c in out["checks"] if c["id"] == "selfcheck")
    assert out["ok"] is False and selfcheck["status"] == "fail" and "planted failure" in selfcheck["message"]


def test_doctor_names_the_pip_line_for_a_missing_extra(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "pandas" else real(name, *a))
    check = next(c for c in diagnostics.checks() if c["id"] == "extra_pandas")
    assert check["status"] == "warn" and check["hint"] == "pip install pandas"


def test_doctor_human_output(capsys):
    assert diagnostics.main([]) == 0
    out = capsys.readouterr().out
    assert out.startswith("riskpy-doctor — RiskPY") and "selfcheck" in out and " fail — " in out


# ---------------------------------------------------------------- introspection tools


def test_introspection_tools():
    names = {t["name"]: t for t in tools.manifest()}
    assert {"riskpy_status", "riskpy_doctor", "riskpy_explain"} <= set(names)
    assert names["riskpy_status"]["kind"] == "introspection" and names["black_scholes"]["kind"] == "calculator"
    s = tools.call("riskpy_status", {"recent": 2})
    assert s["version"] == riskpy.__version__ and s["tools"]["calculators"] >= 27 and s["limits"] == tools.LIMITS
    assert tools.call("riskpy_explain", {"topic": "RP104"})["title"] == "Rate looks like a percentage"
    bs = tools.call("riskpy_explain", {"topic": "black_scholes"})
    assert "d1" in bs["formula"] and bs["input_schema"]["required"] == ["S", "K", "T", "r", "sigma"]
    assert tools.call("riskpy_explain", {"topic": "bogus"}, errors="return")["error"]["code"] == "RP401"
    assert tools.call("riskpy_doctor", {})["ok"] is True


def test_every_tool_parameter_is_documented():
    for t in tools.manifest():
        for pname, prop in t["input_schema"]["properties"].items():
            assert prop.get("description"), f"{t['name']}.{pname} has no description"


# ---------------------------------------------------------------- explain=True


def test_black_scholes_explain_gives_d1_d2_and_the_convention():
    out = tools.call("black_scholes", {**BS, "explain": True})
    assert out["price"] == pytest.approx(10.450583572185565, abs=1e-12)
    ex = out["explain"]
    assert ex["d1"] == pytest.approx(0.35, abs=1e-15) and ex["d2"] == pytest.approx(0.15, abs=1e-15)
    assert ex["N(d1)"] == pytest.approx(N.cdf(0.35), abs=1e-15)
    assert any("continuously compounded" in c for c in ex["conventions"])
    assert "degenerate" in tools.call("black_scholes", {**BS, "T": 0, "explain": True})["explain"]
    assert tools.call("black_scholes", BS) == out["price"]             # default unchanged: a bare number


def test_bond_analytics_explain_is_additive():
    plain = tools.call("bond_analytics", {"face": 100, "coupon": 0.05, "maturity": 4, "ytm": 0.045})
    rich = tools.call("bond_analytics", {"face": 100, "coupon": 0.05, "maturity": 4, "ytm": 0.045, "explain": True})
    assert {k: v for k, v in rich.items() if k != "explain"} == plain and "formula" in rich["explain"]


# ---------------------------------------------------------------- the new tools, independently pinned

V = 1 / 1.05
_A, _B, _C = 0.00022, 2.7e-6, 1.124            # the SULT's Makeham law (AMLCR), written out here


def _tpx(t, x):
    return math.exp(-_A * t - _B * _C ** x * (_C ** t - 1) / math.log(_C))


def _a_due(x, n):
    return sum(V ** k * _tpx(k, x) for k in range(n))


def _term(x, n):
    return sum(V ** (k + 1) * _tpx(k, x) * (1 - _tpx(1, x + k)) for k in range(n))


def test_life_tools_match_the_makeham_law_by_hand():
    x, n = 40, 20
    assert tools.call("temporary_annuity_due", {"x": x, "n": n, "i": 0.05}) == pytest.approx(_a_due(x, n), abs=1e-9)
    assert tools.call("term_insurance", {"x": x, "n": n, "i": 0.05}) == pytest.approx(_term(x, n), abs=1e-9)
    endow = _term(x, n) + V ** n * _tpx(n, x)
    assert tools.call("endowment_insurance", {"x": x, "n": n, "i": 0.05}) == pytest.approx(endow, abs=1e-9)
    p = tools.call("life_premium", {"x": x, "i": 0.05, "kind": "endowment", "n": n})
    assert p["premium"] == pytest.approx(endow / _a_due(x, n), abs=1e-10)
    assert p["premium"] == pytest.approx(1 / _a_due(x, n) - 0.05 / 1.05, abs=1e-10)   # P = 1/ä − d
    term = tools.call("life_premium", {"x": x, "i": 0.05, "kind": "term", "n": n})
    assert term["premium"] == pytest.approx(_term(x, n) / _a_due(x, n), abs=1e-10)


def test_whole_life_premium_matches_amlcr_table_d3():
    # AMLCR Table D.3 at 5%: A_40 = 0.12106, ä_40 = 18.4578 — the premium to the book's precision.
    out = tools.call("life_premium", {"x": 40, "i": 0.05})
    assert out["premium"] == pytest.approx(0.12106 / 18.4578, abs=5e-7)
    assert out["benefit_apv"] == pytest.approx(0.12106, abs=5e-6) and out["annuity_apv"] == pytest.approx(18.4578, abs=5e-5)
    assert tools.call("life_premium", {"x": 40, "i": 0.05, "kind": "term"}, errors="return")["error"]["code"] == "RP113"


def test_convert_rate_closed_forms():
    assert tools.call("convert_rate", {"rate": 0.05, "from_compounding": "semiannual", "to_compounding": "continuous"}) \
        == pytest.approx(2 * math.log(1.025), abs=1e-14)
    assert tools.call("convert_rate", {"rate": 0.05, "from_compounding": "annual", "to_compounding": "continuous"}) \
        == pytest.approx(math.log(1.05), abs=1e-14)
    assert tools.call("convert_rate", {"rate": 0.05, "from_compounding": "continuous", "to_compounding": "12"}) \
        == pytest.approx(12 * (math.exp(0.05 / 12) - 1), abs=1e-14)
    bad = tools.call("convert_rate", {"rate": 0.05, "from_compounding": "weekly-ish", "to_compounding": "annual"},
                     errors="return")
    assert bad["error"]["code"] == "RP108"


def test_bootstrap_yield_curve_by_hand():
    out = tools.call("bootstrap_yield_curve", {"par_rates": [0.02, 0.03], "maturities": [1, 2], "at": [1.5]})
    df1 = 1 / 1.02
    df2 = (1 - 0.03 * df1) / 1.03
    assert out["discount_factors"] == pytest.approx([df1, df2], abs=1e-12)
    assert out["zero_rates"] == pytest.approx([-math.log(df1), -math.log(df2 ) / 2], abs=1e-12)
    z15 = 0.5 * (-math.log(df1) - math.log(df2) / 2)           # linear in zero rates between the knots
    assert out["at"][0]["zero_rate"] == pytest.approx(z15, abs=1e-12)


def test_short_rate_zero_bond_closed_forms():
    r, T, k, th, s = 0.03, 5.0, 0.5, 0.05, 0.02
    B = (1 - math.exp(-k * T)) / k
    vas = math.exp((th - s * s / (2 * k * k)) * (B - T) - s * s * B * B / (4 * k) - B * r)
    out = tools.call("short_rate_zero_bond", {"model": "vasicek", "r": r, "T": T, "kappa": k, "theta": th, "sigma": s})
    assert out["price"] == pytest.approx(vas, abs=1e-14) and out["yield"] == pytest.approx(-math.log(vas) / T, abs=1e-14)
    s = 0.1
    h = math.sqrt(k * k + 2 * s * s)
    den = (h + k) * (math.exp(h * T) - 1) + 2 * h
    cir = (2 * h * math.exp((k + h) * T / 2) / den) ** (2 * k * th / (s * s)) * math.exp(-2 * (math.exp(h * T) - 1) / den * r)
    out = tools.call("short_rate_zero_bond", {"model": "CIR", "r": r, "T": T, "kappa": k, "theta": th, "sigma": s})
    assert out["price"] == pytest.approx(cir, abs=1e-13)


def test_credit_triangle_and_vasicek_quantile():
    out = tools.call("hazard_from_spread", {"spread": 0.015, "recovery": 0.4, "horizon": 5})
    assert out["hazard"] == pytest.approx(0.025, abs=1e-15)
    assert out["default_probability"] == pytest.approx(1 - math.exp(-0.125), abs=1e-15)
    expected = N.cdf((N.inv_cdf(0.02) + math.sqrt(0.18) * N.inv_cdf(0.999)) / math.sqrt(1 - 0.18))
    assert tools.call("vasicek_loss_quantile", {"level": 0.999, "pd": 0.02, "rho": 0.18}) == pytest.approx(expected, abs=1e-12)
    assert tools.call("hazard_from_spread", {"spread": 150, "recovery": 0.4}, errors="return")["error"]["code"] == "RP104"


def test_value_at_risk_by_hand():
    pytest.importorskip("numpy")
    returns = [math.sin(7.3 * k) * 0.02 + 0.0005 for k in range(250)]
    level = 0.99
    xs = sorted(returns)
    pos = (1 - level) * (len(xs) - 1)                          # numpy's default 'linear' percentile
    lo = int(math.floor(pos))
    cutoff = xs[lo] + (pos - lo) * (xs[lo + 1] - xs[lo])
    tail = [r for r in returns if r <= cutoff]
    hist = tools.call("value_at_risk", {"returns": returns, "level": level})
    assert hist["var"] == pytest.approx(-cutoff, abs=1e-15)
    assert hist["expected_shortfall"] == pytest.approx(-sum(tail) / len(tail), abs=1e-15)
    mean = sum(returns) / len(returns)
    sd = math.sqrt(sum((r - mean) ** 2 for r in returns) / (len(returns) - 1))
    para = tools.call("value_at_risk", {"returns": returns, "level": level, "method": "parametric"})
    assert para["var"] == pytest.approx(-(mean + N.inv_cdf(1 - level) * sd), abs=1e-12)
    assert tools.call("value_at_risk", {"returns": [0.01]}, errors="return")["error"]["code"] == "RP111"


_GENINS = [
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


def test_bornhuetter_ferguson_on_genins_by_hand():
    pytest.importorskip("numpy")
    cum = [[float(sum(row[: j + 1])) for j in range(len(row))] for row in _GENINS]
    n = len(cum)
    factors = []
    for k in range(n - 1):
        rows = [r for r in cum if len(r) > k + 1]
        factors.append(sum(r[k + 1] for r in rows) / sum(r[k] for r in rows))
    assert factors[0] == pytest.approx(3.4906, abs=1e-4) and factors[1] == pytest.approx(1.7473, abs=1e-4)  # Mack (1993)
    premiums = [10_000_000.0] * n
    elr = 0.65
    expected = []
    for row in cum:
        cdf = math.prod(factors[len(row) - 1:])
        expected.append(10_000_000.0 * elr * (1 - 1 / cdf))
    triangle = [row + [None] * (n - len(row)) for row in cum]
    out = tools.call("bornhuetter_ferguson", {"triangle": triangle, "premiums": premiums, "expected_loss_ratio": elr})
    assert out["reserve"] == pytest.approx(expected, abs=1e-6)
    assert out["total_reserve"] == pytest.approx(sum(expected), abs=1e-5)
    short = tools.call("bornhuetter_ferguson", {"triangle": triangle, "premiums": premiums[:3], "expected_loss_ratio": elr},
                       errors="return")
    assert short["error"]["code"] == "RP109" and short["error"]["param"] == "premiums"


# ---------------------------------------------------------------- MCP: structured errors over real stdio


def test_mcp_returns_structured_errors_and_the_status_over_stdio(tmp_path):
    pytest.importorskip("mcp")
    from test_agents import _Stdio

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen([sys.executable, "-m", "riskpy.mcp_server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1, env=env, cwd=tmp_path)
    try:
        io = _Stdio(proc)
        io.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                  "clientInfo": {"name": "riskpy-tests", "version": "0"}})
        io.notify("notifications/initialized")
        listed = {t["name"]: t for t in io.request("tools/list")["result"]["tools"]}
        assert {"riskpy_status", "riskpy_doctor", "riskpy_explain", "convert_rate", "life_premium"} <= set(listed)
        assert listed["black_scholes"]["inputSchema"]["properties"]["sigma"]["description"].startswith("annual volatility")

        def call(name, arguments):
            reply = io.request("tools/call", {"name": name, "arguments": arguments})["result"]
            return reply.get("isError") is True, json.loads(reply["content"][0]["text"])

        failed, body = call("black_scholes", {**BS, "S": "100"})            # refused, not coerced
        assert failed and body["error"]["code"] == "RP404" and body["error"]["param"] == "S"
        failed, body = call("astrology", {})
        assert failed and body["error"]["code"] == "RP401"
        failed, body = call("black_scholes", {**BS, "sigma": 20})
        assert failed and body["error"]["code"] == "RP105" and "Divide by 100" in body["error"]["hint"]
        failed, body = call("black_scholes", BS)
        assert not failed and body == pytest.approx(10.450583572185565)
        failed, status = call("riskpy_status", {"recent": 5})
        assert not failed and status["trace"]["calls"] == 4 and status["trace"]["errors"] == 3
        assert [e["via"] for e in status["trace"]["recent"]] == ["mcp"] * 4
        failed, explained = call("riskpy_explain", {"topic": "RP404"})
        assert not failed and explained["code"] == "RP404"
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_doctor_cli_runs_as_a_subprocess(tmp_path):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    out = subprocess.run([sys.executable, "-m", "riskpy.diagnostics", "--json"], capture_output=True, text=True,
                         encoding="utf-8", env=env, timeout=120, cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["ok"] is True


# ---------------------------------------------------------------- 0.5.1: found by black-box testing 0.5.0


def test_parametric_value_at_risk_reports_the_normal_expected_shortfall():
    # 0.5.0 paired the normal VaR with the *historical* expected shortfall.
    pytest.importorskip("numpy")
    returns = [math.sin(7.3 * k) * 0.02 + 0.0005 for k in range(250)]
    mean = sum(returns) / len(returns)
    sd = math.sqrt(sum((r - mean) ** 2 for r in returns) / (len(returns) - 1))
    for level in (0.95, 0.99, 0.995):
        z = N.inv_cdf(1 - level)
        out = tools.call("value_at_risk", {"returns": returns, "level": level, "method": "parametric"})
        assert out["var"] == pytest.approx(-(mean + z * sd), abs=1e-12)
        assert out["expected_shortfall"] == pytest.approx(-mean + sd * N.pdf(z) / (1 - level), rel=1e-9)
        assert out["expected_shortfall"] > out["var"]
    hist = tools.call("value_at_risk", {"returns": returns, "level": 0.99})          # unchanged
    assert hist["expected_shortfall"] == pytest.approx(riskpy.quant.expected_shortfall(returns, 0.99), abs=0)


def test_bornhuetter_ferguson_refuses_a_loss_ratio_typed_as_a_percentage():
    pytest.importorskip("numpy")
    args = {"triangle": [[100, 150, 160], [110, 170, None], [120, None, None]], "premiums": [200, 210, 220]}
    bad = tools.call("bornhuetter_ferguson", {**args, "expected_loss_ratio": 70}, errors="return")["error"]
    assert bad["code"] == "RP104" and bad["param"] == "expected_loss_ratio" and "0.7" in bad["hint"]
    for elr in (0.0, 0.65, 1.4, 5.0):                    # a 140% loss ratio is a bad year, not a typo
        assert "error" not in tools.call("bornhuetter_ferguson", {**args, "expected_loss_ratio": elr}, errors="return")


@pytest.mark.parametrize("name, args, param", [
    ("chain_ladder", {"triangle": []}, "triangle"),
    ("bootstrap_yield_curve", {"par_rates": [0.03, 0.035], "maturities": [0.75, 2]}, "maturities"),
    ("bootstrap_yield_curve", {"par_rates": [0.03, 0.035], "maturities": [1, 2], "at": [1.5, -1]}, "at"),
    ("convert_rate", {"rate": 0.05, "from_compounding": "annual", "to_compounding": "fortnightly"}, "to_compounding"),
    ("convert_rate", {"rate": 0.05, "from_compounding": "nope", "to_compounding": "annual"}, "from_compounding"),
    ("convert_rate", {"rate": 0.05, "from_compounding": "annual", "to_compounding": "0"}, "to_compounding"),
    ("riskpy_explain", {"topic": "RP999"}, "topic"),
])
def test_errors_name_the_tools_own_argument(name, args, param):
    if name in ("chain_ladder",):
        pytest.importorskip("numpy")
    err = tools.call(name, args, errors="return")["error"]
    assert err["param"] == param, err
    if name == "convert_rate":
        assert err["message"].startswith(f"{param} must")


def test_aggregate_loss_explains_mu_and_sigma_instead_of_guessing_a_name():
    err = tools.call("aggregate_loss", {"expected_frequency": 10, "severity_mu": 7, "severity_sigma": 0.5},
                     errors="return")["error"]
    assert err["code"] == "RP402" and "money scale" in err["hint"] and "exp(mu + sigma²/2)" in err["hint"]
    assert "'severity_sigma' → 'severity_mean'" not in err["hint"]
    # an ordinary misspelling still gets its suggestion
    err = tools.call("aggregate_loss", {"expected_frequency": 10, "severity_mean": 1000, "severity_sd": 500,
                                        "n_trials": 10}, errors="return")["error"]
    assert "'n_trials' → 'trials'" in err["hint"]


def test_convert_rate_accepts_the_usual_spellings():
    r = 0.05
    cases = {"daily": 365, "weekly": 52, "annually": 1, "yearly": 1, "semi_annual": 2, "semi-annually": 2}
    for word, m in cases.items():
        got = tools.call("convert_rate", {"rate": r, "from_compounding": word, "to_compounding": "continuous"})
        assert got == pytest.approx(m * math.log(1 + r / m), abs=1e-14), word
        assert riskpy.rates.convert_rate(r, word, m) == pytest.approx(r, abs=1e-12)
    assert tools.call("convert_rate", {"rate": r, "from_compounding": "continuously", "to_compounding": "annual"}) \
        == pytest.approx(math.exp(r) - 1, abs=1e-14)


def test_doctor_selfcheck_names_only_the_modules_it_ran(monkeypatch):
    from riskpy import verify

    real = verify.run

    def without_quant(modules=None, oracle=False):
        report = real([m for m in modules if m != "quant"], oracle=oracle)
        report.skipped.append(("quant", "riskpy.mc needs NumPy, which is not installed."))
        return report

    monkeypatch.setattr(verify, "run", without_quant)
    check = next(c for c in diagnostics.checks() if c["id"] == "selfcheck")
    assert check["status"] == "ok" and check["skipped"] == ["quant"]
    ran = check["message"].split("(")[1].split(")")[0]
    assert "quant" not in ran and "core" in ran and "skipped quant" in check["message"]


def test_missing_numpy_message_puts_the_quoted_pip_line_first(monkeypatch):
    import builtins

    from riskpy import mc

    real = builtins.__import__

    def no_numpy(name, *a, **k):
        if name == "numpy":
            raise ImportError("No module named 'numpy'")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_numpy)
    with pytest.raises(ImportError) as info:
        mc._numpy()
    first = str(info.value).splitlines()[0]
    # riskpy-verify shows only the first line of a skip reason; zsh needs the quotes
    assert first.startswith("riskpy.mc needs NumPy, which is not installed") and 'pip install "open-riskpy[sim]"' in first


def test_the_wheel_installs_only_the_extension_and_the_upload_waits_for_every_wheel():
    root = pathlib.Path(__file__).resolve().parents[1]
    cmake, pyproject = root / "CMakeLists.txt", root / "pyproject.toml"
    workflow = root / ".github" / "workflows" / "publish.yml"
    if not (cmake.exists() and pyproject.exists() and workflow.exists()):
        pytest.skip("not a source checkout")
    assert re.search(r"install\(TARGETS cpp_underwriter DESTINATION riskpy COMPONENT riskpy\)", cmake.read_text())
    assert 'install.components = ["riskpy"]' in pyproject.read_text()
    needs = re.search(r"\n  publish:\n(?:.*\n)*?    needs: \[([^\]]*)\]", workflow.read_text()).group(1)
    assert "build-wheels-extra" in needs
