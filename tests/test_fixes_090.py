"""Regression tests for the fixes made in 0.9.0 after the machine-driven testing of 0.8.0.

tests/test_properties_090.py and tests/test_mutation_killers_090.py come from that campaign and pin each
finding; the tests here cover the parts of the fixes those files do not reach: the notes beside a null that
is null by construction, the refusals' codes and parameters, the audit ledger's thread semantics, replay of
ledgers written by 0.8.0, the MCP server's answer to a line its SDK cannot parse, and the formula's
read-only inputs. Reference values were computed with mpmath at 50 digits and are hard-coded.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import math
import os
import subprocess
import sys
import threading
import time

import pytest

from riskpy import _special, audit, quant, rates, tools
from riskpy.errors import RiskPyError

EL = {"pd": 0.02, "lgd": 0.45, "ead": 1000.0}
BS = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2}
AGG = {"expected_frequency": 2.0, "severity_mean": 100.0, "severity_sd": 50.0, "trials": 500, "seed": 5}


def _error(out):
    assert isinstance(out, dict) and set(out) == {"error"}, out
    return out["error"]


# ---------------------------------------------------------------------------
# Finding 1: overflowed numbers are refused, and a null is always explained


def test_a_result_dataclass_or_array_holding_inf_is_refused_not_nulled():
    np = pytest.importorskip("numpy")
    with pytest.raises(RiskPyError) as info:
        tools._check_result_finite(quant.Greeks(1.0, 0.5, math.inf, 0.1, -0.01, 0.2))
    assert info.value.code == "RP304" and "result.gamma" in str(info.value)
    with pytest.raises(RiskPyError):
        tools._check_result_finite({"x": np.array([1.0, np.nan])})
    with pytest.raises(RiskPyError):
        tools._check_result_finite(np.float32("inf"))
    tools._check_result_finite({"n": 10 ** 400, "ok": [1.0, 2], "s": "inf", "none": None})     # nothing to refuse


def test_merton_deterministic_distance_is_null_with_a_flag_and_a_note():
    out = tools.call("merton", {"assets": 120.0, "debt": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.0})
    assert out["distance_to_default"] is None and out["deterministic"] is True and out["pd"] == 0.0
    assert "infinite" in out["note"]
    regular = tools.call("merton", {"assets": 120.0, "debt": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.25})
    assert math.isfinite(regular["distance_to_default"]) and "deterministic" not in regular and "note" not in regular


def test_mack_note_explains_the_null_total_cv():
    out = tools.call("mack_chain_ladder", {"triangle": [[100, 100, 100], [100, 100, None], [100, None, None]]})
    assert out["total_reserve"] == 0.0 and out["total_cv"] is None and "total reserve is 0" in out["note"]
    regular = tools.call("mack_chain_ladder", {"triangle": [[100, 150, 160], [110, 170, None], [120, None, None]]})
    assert regular["total_cv"] > 0 and "note" not in regular


def test_basel_overflow_is_rp304_and_ordinary_results_keep_risk_weight():
    out = tools.call("basel_irb_capital", {"pd": 0.01, "lgd": 0.45, "ead": 1_000_000.0})
    assert out["risk_weight"] == pytest.approx(12.5 * out["capital_requirement"], rel=1e-15)
    assert _error(tools.call("basel_irb_capital", {"pd": 0.5, "lgd": 0.75, "ead": 1.7976931348623157e308},
                             errors="return"))["code"] == "RP304"


# ---------------------------------------------------------------------------
# Finding 2: no meta-tool run ever draws a fresh seed


def test_sensitivity_refuses_to_bump_the_seed():
    err = _error(tools.call("sensitivity", {"tool": "aggregate_loss", "arguments": AGG, "parameter": "seed"},
                            errors="return"))
    assert err["code"] == "RP108" and err["param"] == "parameter" and "seed" in err["message"]


@pytest.mark.parametrize("seed", [0, 0.0, None])
def test_scenario_table_refuses_a_scenario_that_reseeds_with_zero(seed):
    out = tools.call("scenario_table", {"tool": "aggregate_loss", "arguments": AGG,
                                        "scenarios": {"ok": {"seed": 9}, "fresh": {"seed": seed}}}, errors="return")
    err = _error(out)
    assert err["code"] == "RP113" and err["param"] == "scenarios" and "'fresh'" in err["message"]


def test_scenario_table_still_accepts_other_seeds_and_closed_forms_with_a_seed_override_key():
    out = tools.call("scenario_table", {"tool": "aggregate_loss", "arguments": AGG, "scenarios": {"s": {"seed": 9}}})
    assert "error" not in out["scenarios"]["s"]


def test_replay_of_a_0_8_ledger_skips_a_seed_zero_scenario_and_a_seed_bump():
    # 0.8.0 ran these and recorded results; 0.9.0 refuses them, so replay must skip, not report drift.
    scenario = {"tool": "aggregate_loss", "arguments": AGG, "scenarios": {"fresh": {"seed": 0}}}
    bump = {"tool": "aggregate_loss", "arguments": {**AGG, "seed": 1}, "parameter": "seed"}
    entries = []
    for name, args in (("scenario_table", scenario), ("sensitivity", bump)):
        ledger_entry = {"seq": len(entries) + 1, "tool": name, "arguments": args, "result": {"tool": "x"},
                        "prev": entries[-1]["digest"] if entries else audit.GENESIS}
        ledger_entry["digest"] = audit.digest(ledger_entry)
        entries.append(ledger_entry)
    rep = audit.replay(entries)
    assert rep.chain.ok and not rep.drifted and len(rep.skipped) == 2
    assert all("seed=0" in s["reason"] for s in rep.skipped)


# ---------------------------------------------------------------------------
# Finding 4: integers compare exactly, and one odd entry never stops a replay


def test_replay_compares_integers_exactly_whatever_their_size():
    out = []
    audit._differences(10 ** 400, 10 ** 400, 1e-9, 1e-12, "result", (), out)
    assert out == []
    audit._differences(10 ** 400, 10 ** 400 + 1, 1e-9, 1e-12, "result", (), out)
    assert len(out) == 1 and out[0]["recorded"] == 10 ** 400 and out[0]["replayed"] == 10 ** 400 + 1
    out.clear()
    audit._differences(10 ** 400, 1.0, 1e-9, 1e-12, "result", (), out)      # int beyond float range vs a float
    assert len(out) == 1 and out[0]["relative_difference"] is None
    out.clear()
    audit._differences(2 ** 60, 2 ** 60 + 1, 1e-9, 1e-12, "result", (), out)  # exact, not within rel_tol
    assert len(out) == 1


def test_an_entry_that_cannot_be_replayed_is_skipped_with_its_reason(monkeypatch):
    with audit.record() as led:
        tools.call("expected_loss", EL)
        tools.call("expected_loss", dict(EL, ead=2000.0))

    real = audit._replay_one

    def boom(tools_module, position, entry, rel_tol, abs_tol):
        if position == 1:
            raise OverflowError("int too large to convert to float")
        return real(tools_module, position, entry, rel_tol, abs_tol)

    monkeypatch.setattr(audit, "_replay_one", boom)
    rep = audit.replay(led)
    assert rep.matched == 1 and len(rep.skipped) == 1 and "OverflowError" in rep.skipped[0]["reason"]


# ---------------------------------------------------------------------------
# Finding 5: which ledger a call goes to, per thread and task


def test_nested_record_blocks_restore_the_outer_one_and_start_resumes_after():
    started = audit.start()
    try:
        with audit.record() as outer:
            with audit.record() as inner:
                tools.call("expected_loss", dict(EL, ead=1.0))
            tools.call("expected_loss", dict(EL, ead=2.0))
        tools.call("expected_loss", dict(EL, ead=3.0))
    finally:
        audit.stop()
    assert [e["arguments"]["ead"] for e in inner.entries] == [1.0]
    assert [e["arguments"]["ead"] for e in outer.entries] == [2.0]
    assert [e["arguments"]["ead"] for e in started.entries] == [3.0]
    assert audit.active() is None


def test_a_worker_thread_started_inside_a_block_records_to_it():
    with audit.record() as led:
        t = threading.Thread(target=tools.call, args=("expected_loss", EL))
        t.start()
        t.join()
    assert len(led.entries) == 1 and audit.active() is None


def test_a_copied_context_ties_a_worker_to_its_block_while_another_thread_has_a_newer_one():
    other_in, other_done, worker_done = threading.Event(), threading.Event(), threading.Event()
    got = {}

    def other():
        with audit.record() as lb:
            other_in.set()
            worker_done.wait()
        got["B"] = lb
        other_done.set()

    with audit.record() as la:
        context = contextvars.copy_context()
        ob = threading.Thread(target=other)
        ob.start()
        other_in.wait()                 # B's block is now the most recently entered one
        w = threading.Thread(target=context.run, args=(tools.call, "expected_loss", EL))
        w.start()
        w.join()
        worker_done.set()
        ob.join()
    assert len(la.entries) == 1 and got["B"].entries == []


def test_asyncio_tasks_each_keep_their_own_block():
    async def task(ead, delay):
        with audit.record() as led:
            await asyncio.sleep(delay)
            tools.call("expected_loss", dict(EL, ead=ead))
            await asyncio.sleep(delay)
        return led

    async def main():
        return await asyncio.gather(task(1.0, 0.02), task(2.0, 0.01))

    a, b = asyncio.run(main())
    assert [e["arguments"]["ead"] for e in a.entries] == [1.0]
    assert [e["arguments"]["ead"] for e in b.entries] == [2.0]


# ---------------------------------------------------------------------------
# Findings 16, 17, 19: replay and verify


def test_bytearray_and_bytes_echo_the_same_way():
    assert tools._jsonable(bytearray(b"ab")) == tools._jsonable(b"ab") == "b'ab'"


def test_a_refused_scenario_is_compared_by_code_on_replay():
    recorded = {"scenarios": {"t": {"overrides": {"K": [1.0, 2.0]}, "error": {"code": "RP404", "param": "K",
                                                                           "message": "got (1.0, 2.0)", "hint": "h"}}}}
    replayed = {"scenarios": {"t": {"overrides": {"K": [1.0, 2.0]}, "error": {"code": "RP404", "param": "K",
                                                                           "message": "got [1.0, 2.0]", "hint": "g"}}}}
    assert audit._scenario_errors_by_code(recorded) == audit._scenario_errors_by_code(replayed)
    changed = {"scenarios": {"t": {**replayed["scenarios"]["t"], "error": {"code": "RP101", "param": "K"}}}}
    assert audit._scenario_errors_by_code(recorded) != audit._scenario_errors_by_code(changed)


def test_a_missing_ledger_named_by_path_is_still_an_error(tmp_path):
    with pytest.raises(RiskPyError) as info:
        audit.verify_chain(str(tmp_path / "typo.jsonl"))
    assert info.value.code == "RP505"
    led = audit.Ledger(str(tmp_path / "unwritten.jsonl"))
    assert audit.verify_chain(led).ok and audit.replay(led).ok and "intact" in audit.report(led)


# ---------------------------------------------------------------------------
# Findings 3, 10, 11, 12, 13: numerics and coded errors


def test_formula_inputs_are_read_only_and_a_mutating_formula_reruns_on_copies():
    np = pytest.importorskip("numpy")
    from riskpy import mc

    calls = []

    def mutating(x, y):
        calls.append(1)
        x *= 2.0
        return x + y

    res = mc.Model(x=mc.Normal(0, 1), y=mc.Normal(0, 1)).run(1000, seed=4, fn=mutating)
    draws = mc.Model(x=mc.Normal(0, 1), y=mc.Normal(0, 1)).sample(1000, seed=4)
    assert len(calls) == 2                                    # refused on the read-only view, then run on copies
    assert np.array_equal(res.inputs["x"], draws["x"])        # the draws are untouched
    assert np.allclose(res.values, 2.0 * draws["x"] + draws["y"])
    identity = mc.Model(x=mc.Normal(0, 1)).run(100, seed=1, fn=lambda x: x)
    assert identity.values.flags.writeable and not np.shares_memory(identity.values, identity.inputs["x"])

    def other_error(x):
        raise ValueError("something else went wrong")

    with pytest.raises(ValueError, match="something else"):
        mc.Model(x=mc.Normal(0, 1)).run(10, seed=1, fn=other_error)


def test_tiny_severity_sd_is_rp102_naming_severity_sd():
    err = _error(tools.call("aggregate_loss", {**AGG, "severity_sd": 1e-200}, errors="return"))
    assert err["code"] == "RP102" and err["param"] == "severity_sd"


def test_subnormal_volatility_prices_the_discounted_intrinsic_value():
    price = quant.black_scholes(100.0, 90.0, 1e-4, 0.05, 5e-324)
    assert price == pytest.approx(100.0 - 90.0 * math.exp(-0.05e-4), rel=1e-15)
    g = quant.greeks(100.0, 90.0, 1e-4, 0.05, 5e-324)
    assert g.price == price and g.gamma == 0.0 and g.delta == 1.0
    out = tools.call("black_scholes", {"S": 100.0, "K": 90.0, "T": 1e-4, "r": 0.05, "sigma": 5e-324, "explain": True})
    assert out["price"] == price and "degenerate" in out["explain"]


def test_explain_names_the_diagnostics_that_are_not_finite():
    args = {"S": 1.0, "K": 2.0, "T": 2.220446049250313e-16, "r": 0.0, "sigma": 6.8416506183402196e-304}
    out = tools.call("black_scholes", {**args, "explain": True})
    assert out["price"] == 0.0 and out["explain"]["d1"] is None and out["explain"]["d2"] is None
    assert out["explain"]["N(d1)"] == 0.0 and "d1, d2" in out["explain"]["degenerate"]


def test_sensitivity_names_a_base_value_too_large_for_a_float():
    err = _error(tools.call("sensitivity", {"tool": "vasicek_loss_quantile", "parameter": "rho",
                                            "arguments": {"level": 0.99, "pd": 0.01, "rho": 10 ** 400}},
                            errors="return"))
    assert err["code"] == "RP405" and err["param"] == "rho"


@pytest.mark.parametrize("r, tau, kappa, theta, sigma, expected", [
    (0.05, 10.0, 1e-8, 0.04, 1e-9, 0.60653066274528661688),      # mpmath, 50 digits
    (0.05, 10.0, 0.3, 0.04, 0.1, 0.65756024637614246772),
    (0.03, 30.0, 1e-6, 0.05, 0.02, 0.42757147432556942663),
    (0.05, 1.0, 1e-170, 0.04, 1e-170, math.exp(-0.05)),          # κ, σ → 0: dr = 0, so P = e^(−rτ)
    (0.05, 1e-5, 1e-320, 0.04, 0.0, math.exp(-0.05e-5)),         # γτ underflows: B is τ, not 0
])
def test_cir_bond_with_vanishing_kappa_and_sigma(r, tau, kappa, theta, sigma, expected):
    assert rates.cir_zero_bond(r, 0.0, tau, kappa, theta, sigma) == pytest.approx(expected, rel=2e-15)


def test_a_bond_price_beyond_double_precision_is_rp304_and_the_yield_still_solves():
    bond = rates.Bond(face=100.0, coupon=0.04, maturity=160.0, frequency=1)
    with pytest.raises(RiskPyError) as info:
        bond.price(-0.99)
    assert info.value.code == "RP304" and info.value.param == "ytm"
    y = bond.yield_to_maturity(90.0)
    assert bond.price(y) == pytest.approx(90.0, rel=1e-12)


@pytest.mark.parametrize("p, expected", [
    (0.5 - 1e-12, -2.506572823701860466856097e-12),              # mpmath, 50 digits
    (0.5 + 1e-10, 2.506628482030353902220789e-10),
    (0.5 + 2 ** -52, 5.565832849343533844467847e-16),
    (0.30000000000000004, -0.5244005127080406563136292),
    (0.7, 0.5244005127080406563136292),
    (0.2500000001, -0.6744897498813952091422359),
    (0.7499999999, 0.6744897498813952091422359),
])
def test_norm_ppf_keeps_full_relative_precision_in_the_central_band(p, expected):
    assert _special.norm_ppf(p) == pytest.approx(expected, rel=1e-15)
    assert _special.norm_ppf(0.5) == 0.0


# ---------------------------------------------------------------------------
# Finding 15: the capital_allocation limit


def test_capital_allocation_is_refused_past_five_million_cells():
    assert tools.LIMITS["capital_cells"] == 5_000_000
    err = _error(tools.call("capital_allocation", {"means": [1.0] * 6, "sds": [1.0] * 6, "trials": 1_000_000,
                                                   "seed": 1}, errors="return"))
    assert err["code"] == "RP405" and err["param"] == "trials"


# ---------------------------------------------------------------------------
# Finding 21: riskpy-mcp answers a request line its SDK's parser drops


def test_unparsable_reply_recovers_the_request_id():
    pytest.importorskip("mcp")
    from riskpy import mcp_server

    class Fake(Exception):
        def __init__(self, item):
            self.item = item

        def errors(self):
            return [self.item]

    line = '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"x","arguments":{"k":"\\ud800"}}}'
    reply = mcp_server._unparsable_reply(Fake({"type": "json_invalid", "input": line, "msg": "Invalid JSON"}))
    assert reply.id == 7 and reply.error.code == -32700
    reply = mcp_server._unparsable_reply(Fake({"type": "missing", "input": {"jsonrpc": "2.0", "id": "a",
                                                                             "method": 5}}))
    assert reply is None                                       # method is not a string: not a request we can name
    reply = mcp_server._unparsable_reply(Fake({"type": "missing", "input": {"jsonrpc": "2.0", "id": "a",
                                                                             "method": "ping", "params": 3}}))
    assert reply.id == "a" and reply.error.code == -32600
    assert mcp_server._unparsable_reply(Fake({"input": '{"jsonrpc":"2.0","method":"notifications/x"}'})) is None
    assert mcp_server._unparsable_reply(Fake({"input": "{not json"})) is None
    assert mcp_server._unparsable_reply(Fake({"input": "[" * 100_000})) is None
    assert mcp_server._unparsable_reply(ValueError("not pydantic")) is None


def test_riskpy_mcp_answers_lines_its_parser_rejects(tmp_path):
    pytest.importorskip("mcp")
    # cwd outside the checkout: from the repo root `-m riskpy.mcp_server` imports the uncompiled tree
    proc = subprocess.Popen([sys.executable, "-m", "riskpy.mcp_server"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
                            cwd=str(tmp_path))
    replies = {}
    reader = threading.Thread(target=lambda: [replies.setdefault(m.get("id"), m)
                                              for m in map(json.loads, proc.stdout)], daemon=True)
    reader.start()

    def send(text):
        proc.stdin.write(text + "\n")
        proc.stdin.flush()

    def wait(rid, seconds=20.0):
        end = time.time() + seconds
        while rid not in replies and time.time() < end:
            time.sleep(0.02)
        return replies.get(rid)

    try:
        send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}))
        assert wait(1) is not None
        send(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        send('{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"black_scholes","arguments":'
             '{"S":100,"K":100,"T":1,"r":0.05,"sigma":0.2,"kind":"\\ud800"}}}')
        deep = 1
        for _ in range(220):
            deep = [deep]
        send(json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                         "params": {"name": "black_scholes", "arguments": {"S": deep}}}))
        send(json.dumps({"jsonrpc": "2.0", "id": 4, "method": "ping"}))
        for rid in (2, 3):
            reply = wait(rid)
            assert reply is not None and reply["error"]["code"] == -32700, reply
        assert wait(4) is not None and "result" in replies[4]            # and the server carries on
    finally:
        proc.kill()
        proc.wait(timeout=10)
