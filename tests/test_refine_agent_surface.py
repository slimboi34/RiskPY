"""The agent surface after the 0.8.0 stress test: budgets, bounded names, coded failures, CLIs.

Regression tests for what a planned stress test of 0.7.0 found in the tool layer, the MCP
server, the error catalogue and the riskpy-context / riskpy-verify commands. Each test says
what used to happen.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import re
import subprocess
import sys
import threading
import time

import pytest

from riskpy import context, errors, mcp_server, tools, verify

BS = {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2}
AGG = {"expected_frequency": 100, "severity_mean": 1000, "severity_sd": 500, "trials": 1_000_000, "seed": 1}


def _err(name, args):
    out = tools.call(name, args, errors="return")
    assert isinstance(out, dict) and set(out) == {"error"}, out
    return out["error"]


def _nested(depth):
    value = 1.0
    for _ in range(depth):
        value = [value]
    return value


# ---------------------------------------------------------------------------
# sensitivity and scenario_table: one call is one call's worth of work


def test_a_scenario_table_of_full_size_simulations_is_refused_before_it_runs():
    # 0.7.0 ran aggregate_loss 101 times at 10^8 simulated claims each: minutes from one call.
    started = time.perf_counter()
    err = _err("scenario_table", {"tool": "aggregate_loss", "arguments": AGG,
                                  "scenarios": {f"seed {k}": {"seed": k + 2} for k in range(100)}})
    assert (err["code"], err["param"]) == ("RP405", "scenarios")
    assert "101 times" in err["message"] and "simulated claims" in err["message"]
    assert "split the scenarios" in err["hint"]
    # one scenario that alone asks for a full-size run is counted too
    small = {**AGG, "trials": 10_000}
    err = _err("scenario_table", {"tool": "aggregate_loss", "arguments": small,
                                  "scenarios": {"big": {"trials": 1_000_000}}})
    assert err["code"] == "RP405"
    assert time.perf_counter() - started < 1.0


def test_sensitivity_counts_its_four_runs():
    err = _err("sensitivity", {"tool": "aggregate_loss", "arguments": AGG, "parameter": "expected_frequency"})
    assert (err["code"], err["param"]) == ("RP405", "arguments") and "4 times" in err["message"]
    pytest.importorskip("numpy")
    big = {"means": [10.0] * 20, "sds": [5.0] * 20, "trials": 1_000_000, "seed": 1}
    err = _err("sensitivity", {"tool": "capital_allocation", "arguments": big, "parameter": "level"})
    assert err["code"] == "RP405" and "simulated cells" in err["message"]


def test_meta_tools_within_the_budget_still_run():
    pytest.importorskip("riskpy.cpp_underwriter")
    small = {**AGG, "trials": 2_000}
    out = tools.call("scenario_table", {"tool": "aggregate_loss", "arguments": small,
                                        "scenarios": {f"s{k}": {"seed": k + 2} for k in range(20)}})
    assert out["failed"] == 0 and len(out["scenarios"]) == 20 and "not_run" not in out
    many = tools.call("scenario_table", {"tool": "black_scholes", "arguments": BS,
                                         "scenarios": {f"r{k}": {"r": k / 1000} for k in range(100)}})
    assert many["failed"] == 0 and len(many["scenarios"]) == 100       # closed forms cost nothing


def test_verify_cannot_be_driven_by_a_meta_tool():
    # verify was a "calculator" to scenario_table: 100 runs of the whole verification suite.
    for name, args in (("scenario_table", {"tool": "verify", "arguments": {}, "scenarios": {"a": {}}}),
                       ("sensitivity", {"tool": "verify", "arguments": {}, "parameter": "oracle"})):
        err = _err(name, args)
        assert (err["code"], err["param"]) == ("RP108", "tool") and "prices nothing" in err["message"]
    assert tools.call("verify", {"modules": ["special"]})["ok"]       # still a tool in its own right


def test_a_scenario_table_past_its_time_allowance_returns_what_ran(monkeypatch):
    monkeypatch.setitem(tools.LIMITS, "meta_seconds", 0)
    out = tools.call("scenario_table", {"tool": "black_scholes", "arguments": BS,
                                        "scenarios": {"up": {"S": 110}, "down": {"S": 90}}})
    assert out["base"]["result"] == pytest.approx(10.450583572185565)
    assert out["not_run"] == ["up", "down"] and out["failed"] == 2
    row = out["scenarios"]["down"]
    assert row["overrides"] == {"S": 90} and row["error"]["code"] == "RP405" and "not run" in row["error"]["message"]


def test_a_cancelled_meta_tool_stops_between_runs(monkeypatch):
    runs = []
    original = tools._run_inner
    cancel = threading.Event()

    def counting(t, arguments):
        runs.append(arguments)
        if len(runs) == 2:
            cancel.set()          # the client cancels while the second run is under way
        return original(t, arguments)

    monkeypatch.setattr(tools, "_run_inner", counting)
    tools.trace(clear=True)
    with tools._cancellable(cancel):
        out = tools.call("scenario_table", {"tool": "black_scholes", "arguments": BS,
                                            "scenarios": {f"s{k}": {"S": 90 + k} for k in range(10)}},
                         errors="return")
    assert out["error"]["code"] == "RP408" and len(runs) == 2
    assert tools.trace()[-1]["code"] == "RP408"
    with tools._cancellable(cancel):
        assert _err("sensitivity", {"tool": "black_scholes", "arguments": BS, "parameter": "S"})["code"] == "RP408"
    # outside the scope the flag is gone: the same call runs
    assert tools.call("sensitivity", {"tool": "black_scholes", "arguments": BS, "parameter": "S"})["scheme"] == "central"


# ---------------------------------------------------------------------------
# sensitivity's default bump for a simulation


def test_a_simulation_is_bumped_by_one_per_cent_by_default():
    pytest.importorskip("riskpy.cpp_underwriter")
    sim = {"expected_frequency": 10, "severity_mean": 1000, "severity_sd": 500, "trials": 50_000, "seed": 5}
    out = tools.call("sensitivity", {"tool": "aggregate_loss", "arguments": sim, "parameter": "expected_frequency"})
    assert out["bump"] == pytest.approx(0.1)                    # 1% of 10, stated in the result
    assert out["outputs"]["mean"]["derivative"] == pytest.approx(1000.0, rel=0.15)   # d(λ·E[X])/dλ = E[X]
    # 0.7.0's default, 0.0001 × λ, differenced sampling noise: about −10,000 on macOS. How far a tiny
    # bump drifts is sampling noise (and, before 0.9.0, depended on the C++ standard library's
    # Poisson sampler), so only the 1% default is pinned here.
    # a closed form keeps the fine default
    assert tools.call("sensitivity", {"tool": "black_scholes", "arguments": BS, "parameter": "S"})["bump"] == 0.01


def test_a_simulated_confidence_level_is_bumped_by_its_distance_to_one():
    pytest.importorskip("numpy")
    args = {"means": [100, 50], "sds": [30, 20], "correlation": [[1, 0.3], [0.3, 1]], "trials": 20_000, "seed": 3}
    out = tools.call("sensitivity", {"tool": "capital_allocation", "arguments": args, "parameter": "level"})
    assert out["bump"] == pytest.approx(1e-4) and out["scheme"] == "central"     # not 0.0099 → level 0.9999
    assert "1% of" in tools.TOOLS["sensitivity"].schema()["properties"]["bump"]["description"]


# ---------------------------------------------------------------------------
# names, keys and labels are bounded like string values


def test_long_names_are_refused_and_never_echoed():
    big = "一" * 100_000
    started = time.perf_counter()
    name = _err(big, {})
    assert (name["code"], name["param"]) == ("RP405", "name") and len(name["message"]) < 200
    key = _err("black_scholes", {**BS, big: 1})
    assert key["code"] == "RP405" and len(key["param"]) <= 80 and len(key["message"]) < 200
    label = _err("scenario_table", {"tool": "black_scholes", "arguments": BS, "scenarios": {big: {"S": 110}}})
    assert (label["code"], label["param"]) == ("RP405", "scenarios") and len(label["message"]) < 200
    override = _err("scenario_table", {"tool": "black_scholes", "arguments": BS, "scenarios": {"a": {big: 1}}})
    assert (override["code"], override["param"]) == ("RP405", "scenarios") and len(override["message"]) < 200
    assert time.perf_counter() - started < 1.0          # no difflib over 100,000 characters


def test_unknown_names_are_listed_briefly():
    err = _err("black_scholes", {**BS, **{f"extra_{k}": 1 for k in range(500)}})
    assert err["code"] == "RP402" and "and 495 more" in err["message"] and len(err["message"]) < 400
    assert err["param"] == "extra_0"
    assert _err("black_scholes", {**BS, "vol": 0.2})["param"] == "vol"        # short names unchanged
    assert "'vol' → 'sigma'?" in _err("black_scholes", {**BS, "vol": 0.2})["hint"]
    odd = _err(["x"] * 10_000, {})
    assert odd["code"] == "RP401" and len(odd["message"]) < 120


def test_the_trace_bounds_the_parameter_and_the_cli_escapes_control_characters(tmp_path, monkeypatch, capsys):
    # A 2 MB argument name went into every riskpy_status reply for the next 200 calls.
    monkeypatch.setenv("RISKPY_TRACE_FILE", str(tmp_path / "trace.jsonl"))
    tools.trace(clear=True)
    el = {"pd": 0.02, "lgd": 0.45, "ead": 1e6}
    tools.call("expected_loss", {**el, "k" * 150: 1}, errors="return")
    assert len(tools.trace(1)[0]["param"]) == 80
    escape = "\x1b[1A\x1b[2K\x1b]0;pwned\x07"
    tools.call("expected_loss", {**el, escape: 1}, errors="return")
    tools.call(escape + "black_scholes", {}, errors="return")
    # and a foreign entry in the trace file, written by anything at all
    with open(tmp_path / "trace.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"tool": escape, "ok": False, "code": "RP100", "param": "p" * 100_000,
                             "message": escape * 1000, "via": "\x9b31m"}) + "\n")
    assert tools.main(["trace", "--limit", "0"]) == 0
    out = capsys.readouterr().out
    assert len(out.splitlines()) == 7                     # three calls, in the file and in this process, + the foreign one
    assert not re.search(r"[\x00-\x09\x0b-\x1f\x7f-\x9f]", out), repr(out[:300])
    assert "\\x1b[1A" in out and "\\x9b31m" in out and max(len(line) for line in out.splitlines()) < 1000
    assert tools.main(["trace", "--json", "--limit", "0"]) == 0
    text = capsys.readouterr().out
    assert "\x9b" not in text and "\x1b" not in text
    assert any(entry.get("via") == "\x9b31m" for entry in json.loads(text))     # escaped, not lost


# ---------------------------------------------------------------------------
# deep nesting


def test_deeply_nested_objects_are_refused_with_a_code():
    # 0.7.0 answered RP900 RecursionError. 500 levels: json.loads on Python 3.10/3.11 stops near 1,000.
    deep = json.loads('{"tool": "black_scholes", "arguments": {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2},'
                      ' "scenarios": {"up": {"S": 110}, "junk": {"S": ' + "[" * 500 + "]" * 500 + "}}}")
    err = _err("scenario_table", deep)
    assert (err["code"], err["param"]) == ("RP405", "scenarios") and "32 levels" in err["message"]
    err = _err("sensitivity", {"tool": "black_scholes", "arguments": {**BS, "K": _nested(40)}, "parameter": "S"})
    assert (err["code"], err["param"]) == ("RP405", "arguments")
    # 32 levels in all is accepted as far as nesting goes; the calculator then refuses the list itself
    ok = tools.call("scenario_table", {"tool": "black_scholes", "arguments": BS,
                                       "scenarios": {"junk": {"S": _nested(30)}}})
    assert ok["scenarios"]["junk"]["error"]["code"] == "RP404"
    # the check is iterative: nesting far past the interpreter's recursion limit is just "too deep"
    assert tools._nested_deeper_than(_nested(100_000), 32)
    assert not tools._nested_deeper_than({"a": [[1.0] * 300] * 300}, 32)


# ---------------------------------------------------------------------------
# coded numerical failures


def test_math_domain_error_is_rp304_with_a_hint():
    detail = errors.from_exception(ValueError("math domain error"))
    assert detail["code"] == "RP304" and detail["message"] != "math domain error"
    assert "realistic size" in detail["hint"] and detail["docs"].endswith("#rp304")
    assert errors.from_exception(ValueError("x must be positive"))["code"] == "RP100"     # others unchanged
    for model, args in (("vasicek", {"r": 0.03, "T": 1e6, "kappa": 0.5, "theta": 0.04, "sigma": 0.01}),
                        ("cir", {"r": 0.5, "T": 5000, "kappa": 0.5, "theta": 0.5, "sigma": 0.1})):
        err = _err("short_rate_zero_bond", {"model": model, **args})
        assert (err["code"], err["param"]) == ("RP304", "T"), err
    fine = tools.call("short_rate_zero_bond", {"model": "vasicek", "r": 0.03, "T": 10, "kappa": 0.5, "theta": 0.04,
                                               "sigma": 0.01})
    assert 0.0 < fine["price"] < 1.0 and fine["yield"] > 0.0


def test_capital_allocation_correlation_shape_errors_fit_their_hint():
    pytest.importorskip("numpy")
    base = {"means": [60, 36, 24], "sds": [40, 24, 16]}
    err = _err("capital_allocation", {**base, "correlation": [[1, 0.5], [0.5, 1]]})
    assert (err["code"], err["param"]) == ("RP109", "correlation") and "got 2 rows" in err["message"]
    assert err["hint"] == "One row and one column per line, in the order of means."
    err = _err("capital_allocation", {**base, "correlation": [[1, 0.5], [0.5, 1], [0.2, 0.2]]})
    assert (err["code"], err["param"]) == ("RP109", "correlation[0]") and "row 0 has 2 entries" in err["message"]


def test_option_greeks_states_its_units():
    # The units were only in riskpy_explain; the description is what an MCP client shows the model.
    description = tools.TOOLS["option_greeks"].description
    for words in ("delta per unit of S", "gamma per unit of S squared", "vega and rho per 1 percentage point",
                  "theta per calendar day"):
        assert words in description, words
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf, cdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi), 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
    g = tools.call("option_greeks", BS)
    assert g["vega"] == pytest.approx(S * pdf * math.sqrt(T) / 100, rel=1e-12)
    assert g["rho"] == pytest.approx(K * T * math.exp(-r * T) * cdf / 100, rel=1e-12)
    assert g["theta"] == pytest.approx((-S * pdf * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * cdf) / 365,
                                       rel=1e-12)


# ---------------------------------------------------------------------------
# the audit ledger on the CLI and MCP paths


def test_riskpy_tools_call_warns_when_the_ledger_cannot_record(tmp_path, monkeypatch, capsys):
    bad = tmp_path / "no-such-folder" / "q4.jsonl"
    monkeypatch.setenv("RISKPY_AUDIT_FILE", str(bad))
    assert tools.main(["call", "annuity_certain", '{"i": 0.05, "n": 10}']) == 0      # the answer still comes
    captured = capsys.readouterr()
    assert json.loads(captured.out) == pytest.approx(7.721734929184812)
    assert "RP505" in captured.err and str(bad) in captured.err and not bad.exists()
    good = tmp_path / "q4.jsonl"
    monkeypatch.setenv("RISKPY_AUDIT_FILE", str(good))
    assert tools.main(["call", "annuity_certain", '{"i": 0.05, "n": 10}']) == 0
    assert capsys.readouterr().err == "" and good.exists()


def test_riskpy_mcp_refuses_to_start_with_a_ledger_it_cannot_write(tmp_path, monkeypatch, capsys):
    bad = str(tmp_path / "no-such-folder" / "session.jsonl")
    monkeypatch.setenv("RISKPY_AUDIT_FILE", "placeholder")       # restored after the test
    assert mcp_server.main(["--audit", bad]) == 1
    err = capsys.readouterr().err
    assert "RP505" in err and "--audit" in err and bad in err and "Traceback" not in err
    monkeypatch.setenv("RISKPY_AUDIT_FILE", bad)
    assert mcp_server.main([]) == 1
    assert "RISKPY_AUDIT_FILE" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# MCP over real stdio: cancellation and a long tool name


class _Stdio:
    def __init__(self, proc):
        self.proc, self.lines, self.next_id = proc, [], 1
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.proc.stdout:
            self.lines.append(line)

    def send(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def start(self, method, params):
        rid, self.next_id = self.next_id, self.next_id + 1
        self.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        return rid

    def wait(self, rid, timeout=30.0):
        deadline, seen = time.time() + timeout, 0
        while time.time() < deadline:
            while seen < len(self.lines):
                line = self.lines[seen]
                seen += 1
                try:
                    reply = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if reply.get("id") == rid:
                    return reply, len(line)
            time.sleep(0.02)
        return None, 0

    def request(self, method, params=None, timeout=30.0):
        reply, size = self.wait(self.start(method, params or {}), timeout)
        assert reply is not None, f"no reply to {method}"
        return reply, size


def test_mcp_cancellation_stops_a_scenario_table_and_long_names_stay_small(tmp_path):
    pytest.importorskip("mcp")
    pytest.importorskip("numpy")
    proc = subprocess.Popen([sys.executable, "-m", "riskpy.mcp_server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1, cwd=str(tmp_path))
    try:
        io = _Stdio(proc)
        io.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                  "clientInfo": {"name": "riskpy-tests", "version": "0"}})
        io.send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        # a tool name of 100,000 characters: refused, and the reply does not carry it back
        reply, size = io.request("tools/call", {"name": "x" * 100_000, "arguments": {}})
        assert reply["result"]["isError"] is True and reply["result"]["structuredContent"]["error"]["code"] == "RP405"
        assert size < 4_000
        # ten capital allocations of 2 million cells each: within the budget, about a second apiece
        args = {"tool": "capital_allocation", "scenarios": {f"s{k}": {"seed": k + 2} for k in range(9)},
                "arguments": {"means": [10, 20], "sds": [5, 5], "correlation": [[1, 0.3], [0.3, 1]],
                              "trials": 1_000_000, "seed": 1}}
        long_call = io.start("tools/call", {"name": "scenario_table", "arguments": args})
        time.sleep(1.0)
        io.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": long_call}})
        cancelled = time.time()
        quick, _ = io.request("tools/call", {"name": "black_scholes", "arguments": BS})
        assert "10.45" in quick["result"]["content"][0]["text"]
        entry = None
        while entry is None and time.time() - cancelled < 30:
            status, _ = io.request("tools/call", {"name": "riskpy_status", "arguments": {"recent": 10}})
            entry = next((e for e in status["result"]["structuredContent"]["trace"]["recent"]
                          if e["tool"] == "scenario_table"), None)
            time.sleep(0.1)
        # the server stopped at the next gap between runs and recorded why
        assert entry is not None and entry["code"] == "RP408", entry
        assert io.wait(long_call, timeout=0.5)[0] is None           # a cancelled request gets no reply
    finally:
        proc.kill()
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# riskpy-context


def _class_line(text, name):
    return next(line for line in text.splitlines() if line.startswith(f"- class `{name}("))


def test_context_shows_constructors_own_summaries_and_every_method():
    pytest.importorskip("numpy")
    text = context.build()
    assert _class_line(text, "Model").startswith("- class `Model(**variables: Distribution)`")
    assert "dim: int = 2" in _class_line(text, "GumbelCopula")
    beta = _class_line(text, "Beta")
    assert "Base class" not in beta and beta.startswith("- class `Beta(a: float, b: float)`")   # nothing inherited
    result = _class_line(text, "Result")
    for method in ("var", "tvar", "summary", "describe", "reverse_stress"):
        assert f"`{method}`" in result, method
    assert "_sorted" not in result                                  # private fields stay out
    # a summary is a whole first paragraph, not a first line cut mid-sentence
    assert context._summary("Counting distributions on {0, 1, 2, …}. CDF and quantiles come from the\n"
                            "probability mass function.\n\nMore.") == (
        "Counting distributions on {0, 1, 2, …}. CDF and quantiles come from the probability mass function.")
    assert "its own base run" in text and "share the base run's draws" not in text


def test_the_worked_example_in_the_conventions_runs():
    pytest.importorskip("numpy")
    bullet = next(c for c in context.CONVENTIONS if c.startswith("**A model end to end**"))
    code = re.search(r"```python\n(.*?)```", bullet, re.S).group(1)
    namespace: dict = {}
    exec(compile("\n".join(line[2:] for line in code.splitlines()), "<conventions>", "exec"), namespace)
    res = namespace["res"]
    from riskpy import capital

    allocation = capital.allocate(res.inputs, level=0.995)
    assert sum(allocation.contributions.values()) == pytest.approx(res.tvar(0.995), rel=1e-12)
    assert namespace["stress"].base.tvar(0.995) == pytest.approx(res.tvar(0.995), rel=1e-12)   # its own base
    assert "```python" in mcp_server.read_resource("riskpy://conventions")


def _run_module(module, *args, env=None, cwd=None):
    return subprocess.run([sys.executable, "-m", module, *args], capture_output=True, env=env, cwd=cwd)


def _main_into_a_closed_pipe(cwd, module, *args):
    """Run ``python -m module args`` with stdout a pipe whose reader has gone (`riskpy-context | head -0`):
    the read end is closed as soon as the process starts, long before it has imported riskpy."""
    proc = subprocess.Popen([sys.executable, "-m", module, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            cwd=cwd)
    proc.stdout.close()
    err = proc.stderr.read()
    proc.stderr.close()
    return subprocess.CompletedProcess(proc.args, proc.wait(timeout=120), None, err)


def test_riskpy_context_survives_a_cp1252_console_a_closed_pipe_and_a_bad_path(tmp_path, capsys):
    env = dict(os.environ, PYTHONIOENCODING="cp1252", PYTHONUTF8="0")
    run = _run_module("riskpy.context", env=env, cwd=str(tmp_path))
    assert run.returncode == 0 and b"UnicodeEncodeError" not in run.stderr and b"Traceback" not in run.stderr
    # `riskpy-context --json | head -1`
    for args in (("--json",), ()):
        run = _main_into_a_closed_pipe(str(tmp_path), "riskpy.context", *args)
        assert run.returncode == 0 and b"Traceback" not in run.stderr and b"BrokenPipe" not in run.stderr, run.stderr
    # an output file it cannot write
    missing = tmp_path / "no-such-folder" / "AGENTS.md"
    assert context.main(["-o", str(missing)]) == 1
    message = capsys.readouterr().err
    assert message.startswith("riskpy-context: RP505") and str(missing) in message


# ---------------------------------------------------------------------------
# riskpy-verify


def test_riskpy_verify_refuses_bad_arguments_before_running(tmp_path, capsys):
    with pytest.raises(SystemExit) as info:
        verify.main(["-m", "nosuch"])
    assert info.value.code == 2 and "invalid choice: 'nosuch'" in capsys.readouterr().err
    started = time.perf_counter()
    for flag in ("--json", "--markdown"):
        assert verify.main([flag, str(tmp_path / "no-such-folder" / "out")]) == 1
        assert capsys.readouterr().err.startswith("riskpy-verify: RP505")
    assert time.perf_counter() - started < 2.0             # found before the suite ran, not after
    out, md = tmp_path / "report.json", tmp_path / "report.md"
    assert verify.main(["-m", "special", "--no-oracle", "-q", "--json", str(out), "--markdown", str(md)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["ok"] is True and md.read_text(encoding="utf-8").startswith("**")


def test_riskpy_verify_is_quiet_on_a_closed_pipe(tmp_path):
    run = _main_into_a_closed_pipe(str(tmp_path), "riskpy.verify", "-m", "special", "--no-oracle", "-v")
    assert run.returncode == 0 and b"Traceback" not in run.stderr and b"BrokenPipe" not in run.stderr, run.stderr


# ---------------------------------------------------------------------------
# the docs


def test_every_python_block_on_the_agents_page_runs_on_its_own(tmp_path):
    page = pathlib.Path(__file__).resolve().parents[1] / "docs" / "agents.md"
    if not page.exists():
        pytest.skip("docs/ not shipped here")
    pytest.importorskip("numpy")
    blocks = re.findall(r"```python\n(.*?)```", page.read_text(encoding="utf-8"), re.S)
    assert len(blocks) >= 3
    for block in blocks:
        run = subprocess.run([sys.executable, "-c", block], capture_output=True, text=True, cwd=str(tmp_path))
        assert run.returncode == 0, (block, run.stderr[-500:])


def test_a_closed_pipe_is_recognised_on_windows_too(monkeypatch):
    import errno
    from riskpy.errors import _closed_pipe, _quote_path

    assert _closed_pipe(BrokenPipeError(errno.EPIPE, "Broken pipe"))
    invalid = OSError(errno.EINVAL, "Invalid argument")
    monkeypatch.setattr(os, "name", "nt")
    assert _closed_pipe(invalid)                   # what Windows raises for `riskpy-context | head -0`
    monkeypatch.setattr(os, "name", "posix")
    assert not _closed_pipe(invalid) and not _closed_pipe(OSError(errno.ENOSPC, "No space left"))
    # paths read as typed: backslashes are not doubled, control characters cannot start a new line
    assert _quote_path("C:\\Users\\me\\run.jsonl") == "'C:\\Users\\me\\run.jsonl'"
    assert _quote_path("a\nb") == "'a\\nb'" and "\n" not in _quote_path("a\nb")
