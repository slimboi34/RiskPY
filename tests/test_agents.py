"""The agent surface: tools with schemas, the context pack, and the MCP server.

Every tool is a wrapper over a verified function, so these tests pin the wrappers to the
same numbers the verification suite checks — and the plumbing an agent depends on: the
manifest is data, unavailable tools say what to install, errors are clear.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time

import pytest

from riskpy import context, tools

BS = {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2}


def test_manifest_lists_every_tool_with_a_schema():
    m = tools.manifest()
    names = {t["name"] for t in m}
    assert {"black_scholes", "option_greeks", "implied_vol", "bond_analytics", "basel_irb_capital", "expected_loss",
            "merton", "cds_par_spread", "annuity_certain", "life_annuity_due", "life_insurance",
            "net_premium_reserve", "aggregate_loss", "chain_ladder", "mack_chain_ladder", "verify"} <= names
    bs = next(t for t in m if t["name"] == "black_scholes")
    assert bs["description"].startswith("Black–Scholes–Merton price")
    assert bs["input_schema"]["required"] == ["S", "K", "T", "r", "sigma"]
    assert bs["input_schema"]["properties"]["kind"] == {"type": "string", "default": "call", "description": '"call" or "put"'}
    assert bs["input_schema"]["properties"]["r"]["type"] == "number"
    assert bs["input_schema"]["additionalProperties"] is False
    assert bs["requires"] == "" and bs["available"] and bs["install"] is None
    reserve = next(t for t in m if t["name"] == "net_premium_reserve")
    assert reserve["input_schema"]["properties"]["n"]["type"] == ["integer", "null"]
    cl = next(t for t in m if t["name"] == "chain_ladder")
    assert cl["requires"] == "sim" and cl["install"] == 'pip install "open-riskpy[sim]"'
    assert cl["input_schema"]["properties"]["triangle"] == {
        "type": "array", "items": {"type": "array", "items": {"type": ["number", "null"]}},
        "description": "rows are origin periods, columns are development periods, cumulative amounts; null where not yet observed",
    }
    json.dumps(m)


def test_calls_reproduce_the_verified_numbers():
    assert tools.call("black_scholes", BS) == pytest.approx(10.4506, abs=1e-4)
    greeks = tools.call("option_greeks", BS)
    assert greeks["delta"] == pytest.approx(0.6368, abs=1e-4) and set(greeks) >= {"price", "gamma", "vega", "theta", "rho"}
    assert tools.call("implied_vol", {"price": 10.4506, **{k: v for k, v in BS.items() if k != "sigma"}}) == pytest.approx(0.2, abs=1e-4)
    bond = tools.call("bond_analytics", {"face": 100, "coupon": 0.05, "maturity": 4, "ytm": 0.045})
    assert bond["price"] == pytest.approx(101.8118, abs=1e-4) and bond["modified_duration"] > 0 and bond["dv01"] > 0
    irb = tools.call("basel_irb_capital", {"pd": 0.01, "lgd": 0.45, "ead": 1e6})
    assert irb["risk_weight"] == pytest.approx(0.9232, abs=1e-4) and irb["capital"] == pytest.approx(73853.44, abs=0.01)
    assert tools.call("expected_loss", {"pd": 0.01, "lgd": 0.45, "ead": 1e6}) == pytest.approx(4500.0)
    assert tools.call("merton", {"assets": 100, "debt": 80, "T": 1, "r": 0.03, "sigma": 0.25})["pd"] == pytest.approx(0.18738, abs=1e-4)
    assert 0 < tools.call("cds_par_spread", {"hazard": 0.02, "recovery": 0.4, "r": 0.03, "maturity": 5}) < 0.05
    assert tools.call("annuity_certain", {"i": 0.05, "n": 10, "due": True}) == pytest.approx(8.107822, abs=1e-6)
    assert tools.call("life_annuity_due", {"x": 40, "i": 0.05}) == pytest.approx(18.457757, abs=1e-6)
    assert tools.call("life_insurance", {"x": 40, "i": 0.05}) == pytest.approx(0.121059, abs=1e-6)
    assert tools.call("net_premium_reserve", {"x": 40, "t": 10, "i": 0.05, "kind": "endowment", "n": 20}) == pytest.approx(0.380073, abs=1e-6)


def test_aggregate_loss_uses_the_core_and_converts_moments():
    out = tools.call("aggregate_loss", {"expected_frequency": 140, "severity_mean": 18_000, "severity_sd": 42_000,
                                        "trials": 20_000, "seed": 42})
    assert out["trials"] == 20_000
    assert out["severity_mu"] == pytest.approx(8.8665, abs=1e-3) and out["severity_sigma"] == pytest.approx(1.3650, abs=1e-3)
    assert 2.0e6 < out["mean"] < 3.2e6            # 140 claims × 18,000 ≈ 2.52m
    assert out["var"]["0.5"] < out["mean"] < out["var"]["0.995"] < out["tvar_0.995"] <= out["max"]
    again = tools.call("aggregate_loss", {"expected_frequency": 140, "severity_mean": 18_000, "severity_sd": 42_000,
                                          "trials": 20_000, "seed": 42})
    assert again == out                            # seeded, so reproducible
    with pytest.raises(ValueError):
        tools.call("aggregate_loss", {"expected_frequency": 1, "severity_mean": -5, "severity_sd": 1})


def test_verify_tool_returns_the_report_as_data():
    out = tools.call("verify", {"modules": ["core", "life"]})
    assert out["ok"] is True and out["failed"] == 0 and out["passed"] == out["checks"] > 0
    assert out["failures"] == [] and isinstance(out["skipped"], list)


def test_reserving_tools_reproduce_genins_and_say_what_they_need(monkeypatch):
    pytest.importorskip("numpy")
    from riskpy import reserving

    tri = [[None if math.isnan(v) else float(v) for v in row] for row in reserving.genins().cumulative.tolist()]
    cl = tools.call("chain_ladder", {"triangle": tri})
    assert cl["total_reserve"] == pytest.approx(18_680_856, abs=1)
    assert len(cl["reserve"]) == 10 and len(cl["factors"]) == 9
    mack = tools.call("mack_chain_ladder", {"triangle": tri})
    assert mack["total_se"] == pytest.approx(2_447_095, abs=1)
    json.dumps(mack)

    monkeypatch.setattr(tools, "_extra_installed", lambda extra: False)
    assert next(t for t in tools.manifest() if t["name"] == "chain_ladder")["available"] is False
    with pytest.raises(tools.ToolUnavailable, match=r"open-riskpy\[sim\]"):
        tools.call("chain_ladder", {"triangle": [[1.0]]})
    assert tools.call("expected_loss", {"pd": 0.5, "lgd": 0.5, "ead": 4}) == 1.0   # the core keeps working


def test_unknown_tool_and_unexpected_argument_are_clear_errors():
    with pytest.raises(KeyError, match="unknown tool"):
        tools.call("astrology", {})
    with pytest.raises(TypeError, match="unexpected argument"):
        tools.call("expected_loss", {"pd": 0.01, "lgd": 0.4, "ead": 1, "foo": 1})


def test_cli(capsys):
    assert tools.main([]) == 0
    text = capsys.readouterr().out
    assert "black_scholes" in text and "tools." in text
    assert tools.main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["input_schema"]["type"] == "object"
    assert tools.main(["call", "expected_loss", '{"pd": 0.01, "lgd": 0.45, "ead": 1000000}']) == 0
    assert json.loads(capsys.readouterr().out) == pytest.approx(4500.0)
    assert tools.main(["schema", "annuity_certain"]) == 0
    assert json.loads(capsys.readouterr().out)["required"] == ["i", "n"]
    assert tools.main(["call", "astrology", "{}"]) == 1
    assert "unknown tool" in capsys.readouterr().err


def test_context_pack_is_the_rules_then_the_map(capsys):
    text = context.build()
    for heading in ("# RiskPY", "## Install", "## Conventions that bite", "## Tools for agents", "## The compiled core",
                    "## Modules", "### `riskpy.quant`", "## Verification"):
        assert heading in text, heading
    assert "from_moments" in text and "Rates are decimals" in text
    assert "`black_scholes(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, kind: str = call)" in text
    assert "| `black_scholes` |" in text
    d = context.describe()
    assert d["version"] and "quant" in d["modules"] and d["modules"]["quant"]["available"]
    assert {t["name"] for t in d["tools"]} >= {"black_scholes", "verify"}
    json.dumps(d)
    assert context.main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out)["conventions"]


# ---------------------------------------------------------------- MCP, over real stdio


class _Stdio:
    """Speak newline-delimited JSON-RPC to a subprocess, with a timeout."""

    def __init__(self, proc):
        self.proc, self.lines, self.next_id = proc, [], 1
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.proc.stdout:
            self.lines.append(line)

    def request(self, method, params=None, timeout=30.0):
        rid, self.next_id = self.next_id, self.next_id + 1
        msg = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        deadline = time.time() + timeout
        seen = 0
        while time.time() < deadline:
            while seen < len(self.lines):
                line = self.lines[seen]
                seen += 1
                try:
                    reply = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if reply.get("id") == rid:
                    return reply
            time.sleep(0.02)
        raise TimeoutError(f"no reply to {method}")

    def notify(self, method, params=None):
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method, "params": params or {}}) + "\n")
        self.proc.stdin.flush()


def test_mcp_server_lists_and_calls_the_tools_over_stdio():
    pytest.importorskip("mcp")
    # utf-8 explicitly: the descriptions carry en dashes, and Windows' locale codec is not utf-8
    proc = subprocess.Popen([sys.executable, "-m", "riskpy.mcp_server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
    try:
        io = _Stdio(proc)
        init = io.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                         "clientInfo": {"name": "riskpy-tests", "version": "0"}})
        assert init["result"]["serverInfo"]["name"] == "riskpy"
        io.notify("notifications/initialized")
        listed = io.request("tools/list")
        names = {t["name"] for t in listed["result"]["tools"]}
        assert {"black_scholes", "basel_irb_capital", "life_annuity_due", "aggregate_loss", "verify"} <= names
        bs = next(t for t in listed["result"]["tools"] if t["name"] == "black_scholes")
        assert bs["description"].startswith("Black") and "Merton price of a European option" in bs["description"]
        assert "sigma" in bs["inputSchema"]["properties"]
        called = io.request("tools/call", {"name": "black_scholes", "arguments": BS})
        assert called["result"].get("isError") is not True, called
        text = " ".join(c.get("text", "") for c in called["result"]["content"])
        assert "10.45" in text
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_mcp_client_config_points_at_the_command():
    pytest.importorskip("mcp")
    from riskpy import mcp_server

    cfg = mcp_server.client_config()
    command = cfg["mcpServers"]["riskpy"]["command"]
    assert os.path.basename(command).lower().startswith("riskpy-mcp")   # riskpy-mcp, or riskpy-mcp.EXE on Windows
