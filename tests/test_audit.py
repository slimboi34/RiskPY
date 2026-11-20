"""riskpy.audit: the ledger, its hash chain, replay, the report — and the MCP resources and prompts.

The digests are recomputed here with hashlib from the documented canonical form, not with
the library's own function, so a change to how an entry is hashed breaks these tests.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import pytest

from riskpy import audit, mcp_server, tools
from riskpy.errors import InputError, RiskPyError

BS = {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2}
SIM = {"expected_frequency": 20, "severity_mean": 100, "severity_sd": 50, "trials": 2000}


def canonical_digest(entry):
    body = {k: v for k, v in entry.items() if k != "digest"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


@pytest.fixture(autouse=True)
def no_ambient_ledger(monkeypatch):
    monkeypatch.delenv(audit.ENV, raising=False)
    audit.stop()
    yield
    audit.stop()


def _run(path=None, note=None):
    with audit.record(path, note=note) as ledger:
        tools.call("black_scholes", BS)
        tools.call("expected_loss", {"pd": 0.02, "lgd": 0.45, "ead": 1_000_000})
        tools.call("black_scholes", {**BS, "sigma": 20}, errors="return")           # refused: RP105
        tools.call("aggregate_loss", {**SIM, "seed": 7})
    return ledger


# ---------------------------------------------------------------------------
# Recording


def test_a_ledger_records_calls_in_full_and_chains_them():
    ledger = _run(note="unit test")
    assert len(ledger) == 4 and [e["seq"] for e in ledger.entries] == [1, 2, 3, 4]
    first, second, refused, sim = ledger.entries
    assert first["tool"] == "black_scholes" and first["arguments"] == BS
    assert first["result"] == pytest.approx(10.450583572185565, abs=1e-9)            # the verified BS value
    assert second["result"] == pytest.approx(9000.0)                                 # 0.02 × 0.45 × 1,000,000
    assert "result" not in refused and refused["error"]["code"] == "RP105" and refused["error"]["param"] == "sigma"
    assert sim["arguments"]["seed"] == 7 and sim["result"]["trials"] == 2000
    for entry in ledger.entries:
        assert entry["note"] == "unit test" and entry["via"] == "python" and entry["ms"] >= 0.0
        assert entry["riskpy"] == __import__("riskpy").__version__
    # the chain, recomputed independently
    assert first["prev"] == "0" * 64 == audit.GENESIS
    for previous, entry in zip([None, *ledger.entries], ledger.entries):
        assert entry["digest"] == canonical_digest(entry) == audit.digest(entry)
        if previous is not None:
            assert entry["prev"] == previous["digest"]
    assert ledger.head == sim["digest"] and len(ledger.head) == 64
    assert audit.Ledger().head == audit.GENESIS


def test_only_calculator_calls_inside_the_block_are_recorded():
    tools.call("black_scholes", BS)                                  # before: not recorded
    with audit.record() as ledger:
        tools.call("riskpy_status", {})                              # introspection: not recorded
        tools.call("riskpy_explain", {"topic": "RP105"})
        tools.call("no_such_tool", {}, errors="return")              # an unknown tool is a recorded failure
        tools.black_scholes(**BS)                                    # a direct Python call bypasses tools.call
        assert audit.active() is ledger
    tools.call("black_scholes", BS)                                  # after: not recorded
    assert audit.active() is None
    assert [e["tool"] for e in ledger.entries] == ["no_such_tool"] and ledger.entries[0]["error"]["code"] == "RP401"
    # nested blocks: the inner one records, then the outer one resumes
    with audit.record() as outer:
        tools.call("black_scholes", BS)
        with audit.record() as inner:
            tools.call("expected_loss", {"pd": 0.1, "lgd": 0.5, "ead": 10})
        tools.call("black_scholes", BS)
    assert len(outer) == 2 and len(inner) == 1
    # start/stop
    started = audit.start(note="s")
    tools.call("black_scholes", BS)
    assert audit.stop() is started and len(started) == 1 and audit.stop() is None


def test_a_file_ledger_is_json_lines_and_continues_across_ledgers(tmp_path):
    path = str(tmp_path / "run.jsonl")
    first = _run(path)
    lines = open(path, encoding="utf-8").read().splitlines()
    assert len(lines) == 4 and [json.loads(line) for line in lines] == first.entries
    assert all(ord(ch) < 128 for line in lines for ch in line)        # ASCII on disk, whatever the arguments held
    # a second ledger on the same file picks the chain up where the first left it
    with audit.record(path) as second:
        tools.call("black_scholes", {**BS, "kind": "put"})
    entry = second.entries[0]
    assert entry["seq"] == 5 and entry["prev"] == first.entries[-1]["digest"]
    assert audit.verify_chain(path).ok and audit.verify_chain(path).entries == 5
    assert audit.load(path)[-1] == entry and second.head == entry["digest"]
    assert "run.jsonl" in repr(second)


def test_the_environment_variable_records_cli_calls_in_separate_processes(tmp_path):
    path = str(tmp_path / "cli.jsonl")
    env = {**os.environ, audit.ENV: path}
    for arguments in (BS, {**BS, "sigma": 0.3}, {**BS, "sigma": -1}):
        subprocess.run([sys.executable, "-m", "riskpy.tools", "call", "black_scholes", json.dumps(arguments)],
                       env=env, cwd=str(tmp_path), capture_output=True, text=True, encoding="utf-8", timeout=120)
    entries = audit.load(path)
    assert [e["seq"] for e in entries] == [1, 2, 3] and all(e["via"] == "cli" for e in entries)
    assert entries[1]["arguments"]["sigma"] == 0.3 and entries[2]["error"]["code"] == "RP105"
    chain = audit.verify_chain(path)
    assert chain.ok and chain.head == entries[-1]["digest"] == canonical_digest(entries[-1])
    # the CLI: verify, replay, report
    run = lambda *argv: subprocess.run([sys.executable, "-m", "riskpy.audit", *argv], cwd=str(tmp_path),  # noqa: E731
                                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    verified = run("verify", path)
    assert verified.returncode == 0 and chain.head in verified.stdout and "3 entries" in verified.stdout
    replayed = run("replay", path, "--json")
    assert replayed.returncode == 0 and json.loads(replayed.stdout)["matched"] == 3
    out = str(tmp_path / "record.md")
    assert run("report", path, "-o", out, "--title", "Test record", "--check").returncode == 0
    text = open(out, encoding="utf-8").read()
    assert text.startswith("# Test record") and chain.head in text and "3 matched, 0 drifted" in text
    # tamper with the file: verify and replay both exit 1 and say where
    lines = open(path, encoding="utf-8").read().splitlines()
    lines[1] = lines[1].replace('"sigma": 0.3', '"sigma": 0.25')
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    broken = run("verify", path)
    assert broken.returncode == 1 and "BROKEN at entry 2" in broken.stdout
    assert run("replay", path).returncode == 1
    assert run("verify", str(tmp_path / "missing.jsonl")).returncode == 1


# ---------------------------------------------------------------------------
# The chain


def test_verify_chain_catches_every_kind_of_tampering():
    entries = audit.load(_run())
    intact = audit.verify_chain(entries)
    assert intact.ok and bool(intact) and intact.entries == 4 and intact.broken_at is None
    assert intact.head == entries[-1]["digest"] and intact.to_dict()["ok"] is True

    def tampered(change):
        copy = json.loads(json.dumps(entries))
        change(copy)
        return audit.verify_chain(copy)

    edited = tampered(lambda e: e[1].__setitem__("result", 9000.5))                 # a result changed
    assert not edited.ok and edited.broken_at == 2 and "changed after it was recorded" in edited.reason
    argument = tampered(lambda e: e[0]["arguments"].__setitem__("sigma", 0.21))      # an argument changed
    assert not argument.ok and argument.broken_at == 1
    removed = tampered(lambda e: e.pop(1))                                           # an entry dropped
    assert not removed.ok and removed.broken_at == 2 and "removed" in removed.reason
    swapped = tampered(lambda e: e.insert(0, e.pop(1)))                              # reordered
    assert not swapped.ok and swapped.broken_at == 1
    truncated = tampered(lambda e: e.pop())                                          # the tail cut off: still a valid
    assert truncated.ok and truncated.head != intact.head                            # chain, but a different head
    no_digest = tampered(lambda e: e[2].pop("digest"))
    assert not no_digest.ok and no_digest.broken_at == 3

    # Re-hashing one edited entry is not enough: the next entry still points at the old digest.
    def rehash_one(e):
        e[1]["result"] = 9000.5
        e[1]["digest"] = canonical_digest(e[1])

    relinked = tampered(rehash_one)
    assert not relinked.ok and relinked.broken_at == 3 and "does not link" in relinked.reason
    assert audit.verify_chain([]).ok and audit.verify_chain([]).head == audit.GENESIS


def test_load_refuses_a_damaged_ledger(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"seq": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(InputError) as exc:
        audit.load(str(path))
    assert exc.value.code == "RP100" and "line 2" in str(exc.value)
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(InputError):
        audit.load(str(path))
    with pytest.raises(InputError) as exc:
        audit.load(str(tmp_path / "absent.jsonl"))
    assert exc.value.code == "RP505"
    with pytest.raises(InputError):
        audit.load(42)
    with pytest.raises(InputError) as exc:
        audit.digest({"x": float("nan")})
    assert exc.value.code == "RP100"


# ---------------------------------------------------------------------------
# Replay


def test_replay_recomputes_every_call():
    ledger = _run()
    with audit.record() as watching:
        outcome = audit.replay(ledger)
    assert outcome.ok and bool(outcome) and outcome.entries == 4 and outcome.matched == 4
    assert outcome.drifted == [] and outcome.skipped == [] and outcome.chain.ok
    assert outcome.recorded_versions == [outcome.version]
    assert len(watching) == 0                         # replayed calls are not themselves recorded
    assert len(ledger) == 4
    assert "4 matched, 0 drifted, 0 skipped" in outcome.summary()


def _chained(entries):
    """Re-chain hand-made entries with hashlib, as a ledger written by another version would be."""
    prev, out = "0" * 64, []
    for seq, entry in enumerate(entries, 1):
        entry = {**entry, "seq": seq, "prev": prev}
        entry["digest"] = canonical_digest(entry)
        prev = entry["digest"]
        out.append(entry)
    return out


def test_replay_reports_drift_against_a_ledger_from_another_version():
    base = {"at": "2026-01-01T00:00:00.000+00:00", "ms": 1.0, "via": "python", "riskpy": "0.0.1", "python": "3.12.0"}
    entries = _chained([
        {**base, "tool": "black_scholes", "arguments": BS, "result": 10.45},                  # a rounded old value
        {**base, "tool": "black_scholes", "arguments": BS, "result": 10.450583572185565},     # the right one
        {**base, "tool": "bond_analytics", "arguments": {"face": 100, "coupon": 0.05, "maturity": 10, "ytm": 0.05},
         "result": {"price": 100.0, "macaulay_duration": 7.0, "old_field": 1}},
        {**base, "tool": "black_scholes", "arguments": {**BS, "sigma": 20},
         "error": {"code": "RP100", "message": "old message", "param": "sigma"}},              # the code changed since
        {**base, "tool": "black_scholes", "arguments": {**BS, "sigma": 20},
         "error": {"code": "RP105", "message": "old wording", "param": "sigma"}},              # same code: a match
        {**base, "tool": "black_scholes", "arguments": {**BS, "sigma": 20}, "result": 12.0},    # now refused
        {**base, "tool": "retired_tool", "arguments": {}, "result": 1.0},
        {**base, "tool": "aggregate_loss", "arguments": SIM, "result": {"mean": 2000.0}},      # no seed
        {**base, "tool": "sensitivity", "arguments": {"tool": "aggregate_loss", "arguments": {**SIM, "seed": 0},
                                                      "parameter": "severity_mean"}, "result": {}},
    ])
    outcome = audit.replay(entries)
    assert outcome.chain.ok and not outcome.ok and outcome.recorded_versions == ["0.0.1"]
    assert outcome.matched == 2 and [d["seq"] for d in outcome.drifted] == [1, 3, 4, 6]
    first = outcome.drifted[0]["differences"][0]
    assert first["path"] == "result" and first["recorded"] == 10.45
    assert first["relative_difference"] == pytest.approx(0.000583572185565 / 10.450583572185565, rel=1e-6)
    paths = {d["path"] for d in outcome.drifted[1]["differences"]}
    assert "result.macaulay_duration" in paths and "result.old_field" in paths and "result.price" not in paths
    assert outcome.drifted[2]["differences"][0] == {"path": "error.code", "recorded": "RP100", "replayed": "RP105",
                                                    "relative_difference": None}
    assert "error RP105" in outcome.drifted[3]["differences"][0]["replayed"]
    assert [(s["seq"], s["tool"]) for s in outcome.skipped] == [(7, "retired_tool"), (8, "aggregate_loss"),
                                                                  (9, "sensitivity")]
    assert "no such tool" in outcome.skipped[0]["reason"] and "seed=0" in outcome.skipped[1]["reason"]
    # a looser tolerance accepts the rounded value
    assert [d["seq"] for d in audit.replay(entries, rel_tol=1e-3).drifted] == [3, 4, 6]
    text = outcome.summary()
    assert "drift  #1 black_scholes" in text and "skip   #8 aggregate_loss" in text and "0.0.1" in text
    assert outcome.to_dict()["matched"] == 2
    with pytest.raises(InputError) as exc:
        audit.replay(entries, rel_tol=-1)
    assert exc.value.code == "RP102"


def test_replay_of_a_tampered_ledger_fails_on_the_chain_and_on_the_value():
    entries = json.loads(json.dumps(audit.load(_run())))
    entries[1]["result"] = 9500.0
    outcome = audit.replay(entries)
    assert not outcome.ok and not outcome.chain.ok and outcome.chain.broken_at == 2
    assert [d["seq"] for d in outcome.drifted] == [2] and "BROKEN at entry 2" in outcome.summary()


# ---------------------------------------------------------------------------
# The report and the status


def test_report_is_a_readable_record(tmp_path):
    path = str(tmp_path / "run.jsonl")
    ledger = _run(path, note="Q4 capital run")
    text = audit.report(path, title="Q4 capital — calculation record", check=True)
    assert text.startswith("# Q4 capital — calculation record\n")
    assert f"`{ledger.head}`" in text and "**Chain:** intact" in text
    assert "**Calls:** 4 (3 succeeded, 1 refused)" in text and "Q4 capital run" in text
    assert "4 matched, 0 drifted, 0 skipped" in text
    assert "| 3 | " in text and "refused: RP105" in text
    assert '"sigma": 0.2' in text and "10.450583572185565" in text          # full arguments and results
    for entry in ledger.entries:
        assert entry["digest"] in text
    assert "riskpy-audit replay" in text and "Rates, yields, volatilities" in text
    # without check there is no replay section; a broken chain is said plainly
    assert "## Reproducibility" not in audit.report(ledger.entries)
    broken = json.loads(json.dumps(ledger.entries))
    broken[0]["result"] = 1.0
    assert "**BROKEN** at entry 1" in audit.report(broken)
    with pytest.raises(InputError):
        audit.report(ledger.entries, title="  ")
    # a long result is cut, with a pointer to the ledger
    with audit.record() as big:
        tools.call("bootstrap_yield_curve", {"par_rates": [0.03 + 0.0001 * k for k in range(60)],
                                             "maturities": [float(k) for k in range(1, 61)]})
    assert "the ledger has the rest" in audit.report(big)


def test_status_and_the_riskpy_audit_tool(tmp_path, monkeypatch):
    off = tools.call("riskpy_audit", {})
    assert off["recording"] is False and off["head"] is None and "RISKPY_AUDIT_FILE" in off["how"]
    path = str(tmp_path / "env.jsonl")
    monkeypatch.setenv(audit.ENV, path)
    tools.call("black_scholes", BS)
    tools.call("aggregate_loss", SIM)                                   # seed 0: recorded, not replayable
    on = tools.call("riskpy_audit", {"check": True})
    entries = audit.load(path)
    assert on["recording"] is True and on["file"] == path and on["entries"] == 2 and on["chain_ok"] is True
    assert on["head"] == entries[-1]["digest"] == canonical_digest(entries[-1])
    assert on["replay"]["matched"] == 1 and on["replay"]["skipped"][0]["tool"] == "aggregate_loss"
    assert on["write_failures"] == 0
    assert tools.status()["audit_file"] == path
    assert len(audit.load(path)) == 2                                   # asking about the ledger did not add to it
    assert tools.TOOLS["riskpy_audit"].kind == "introspection"
    # a ledger that cannot be written never breaks the calculation; the failure is counted
    monkeypatch.setenv(audit.ENV, str(tmp_path / "no" / "such" / "dir" / "x.jsonl"))
    assert tools.call("black_scholes", BS) == pytest.approx(10.450583572185565)
    status = audit.status()
    assert audit.active().failures == 1 and status["write_failures"] == 1 and status["entries"] == 0
    assert "could not be written" in status["how"]
    # a ledger file that has not been written to yet is empty, not broken
    monkeypatch.setenv(audit.ENV, str(tmp_path / "fresh.jsonl"))
    fresh = audit.status()
    assert fresh["recording"] is True and fresh["entries"] == 0 and fresh["chain_ok"] is True
    assert fresh["head"] == audit.GENESIS and fresh["write_failures"] == 0


# ---------------------------------------------------------------------------
# MCP: resources and prompts


def test_resources_are_readable_without_the_sdk():
    assert set(mcp_server.RESOURCES) == {"riskpy://context", "riskpy://conventions", "riskpy://errors",
                                         "riskpy://tools", "riskpy://audit"}
    conventions = mcp_server.read_resource("riskpy://conventions")
    assert conventions.startswith("# RiskPY conventions") and "Rates are decimals" in conventions
    assert "Losses are positive" in conventions and "Quote the ledger head" in conventions
    assert mcp_server.read_resource("riskpy://context").startswith("# RiskPY")
    codes = {e["code"] for e in json.loads(mcp_server.read_resource("riskpy://errors"))}
    assert {"RP104", "RP105", "RP401"} <= codes
    names = {t["name"] for t in json.loads(mcp_server.read_resource("riskpy://tools"))}
    assert names == set(tools.TOOLS)
    assert json.loads(mcp_server.read_resource("riskpy://audit"))["recording"] is False
    assert json.loads(mcp_server.read_resource("riskpy://explain/RP104"))["code"] == "RP104"
    assert json.loads(mcp_server.read_resource("riskpy://explain/capital_aggregate"))["tool"] == "capital_aggregate"
    with pytest.raises(RiskPyError) as exc:
        mcp_server.read_resource("riskpy://nope")
    assert exc.value.code == "RP401"


def test_prompts_name_real_tools_only():
    assert set(mcp_server.PROMPTS) == {"capital_model", "reserve_review", "stress_test", "audit_trail"}
    text = mcp_server.render_prompt("capital_model", lines="motor, property", level="0.99")
    assert "motor, property" in text and "0.99 confidence level" in text
    assert "stress_test" in mcp_server.PROMPTS and "black_scholes" in mcp_server.render_prompt(
        "stress_test", tool="black_scholes", base_case="S=100")
    # every snake_case name a prompt mentions is a real tool, so an agent following it cannot be sent to a
    # tool that does not exist
    import re

    for name in mcp_server.PROMPTS:
        mentioned = set(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", mcp_server.render_prompt(name)))
        assert mentioned and mentioned <= set(tools.TOOLS), (name, mentioned - set(tools.TOOLS))
    with pytest.raises(RiskPyError) as exc:
        mcp_server.render_prompt("nope")
    assert exc.value.code == "RP401"
    with pytest.raises(RiskPyError) as exc:
        mcp_server.render_prompt("audit_trail", lines="x")
    assert exc.value.code == "RP402" and exc.value.param == "lines"


class _Stdio:
    """Newline-delimited JSON-RPC to a subprocess, with a timeout."""

    def __init__(self, proc):
        self.proc, self.next_id = proc, 0

    def request(self, method, params=None, timeout=60.0):
        self.next_id += 1
        message = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
        if params is not None:
            message["params"] = params
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"server closed the pipe: {self.proc.stderr.read()}")
            reply = json.loads(line)
            if reply.get("id") == self.next_id:
                return reply
        raise TimeoutError(f"no reply to {method}")

    def notify(self, method):
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method, "params": {}}) + "\n")
        self.proc.stdin.flush()


def test_mcp_server_serves_resources_prompts_and_the_ledger_over_stdio(tmp_path):
    pytest.importorskip("mcp")
    pytest.importorskip("numpy")
    ledger = str(tmp_path / "mcp.jsonl")
    proc = subprocess.Popen([sys.executable, "-m", "riskpy.mcp_server", "--audit", ledger], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
                            cwd=str(tmp_path))
    try:
        io = _Stdio(proc)
        init = io.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                         "clientInfo": {"name": "riskpy-tests", "version": "0"}})
        capabilities = init["result"]["capabilities"]
        assert "resources" in capabilities and "prompts" in capabilities and "tools" in capabilities
        io.notify("notifications/initialized")

        listed = {r["uri"]: r for r in io.request("resources/list")["result"]["resources"]}
        assert set(listed) == set(mcp_server.RESOURCES)
        assert listed["riskpy://conventions"]["mimeType"] == "text/markdown"
        templates = io.request("resources/templates/list")["result"]["resourceTemplates"]
        assert [t["uriTemplate"] for t in templates] == ["riskpy://explain/{topic}"]
        read = io.request("resources/read", {"uri": "riskpy://conventions"})["result"]["contents"][0]
        assert read["text"] == mcp_server.read_resource("riskpy://conventions")
        explained = io.request("resources/read", {"uri": "riskpy://explain/RP105"})["result"]["contents"][0]
        assert json.loads(explained["text"])["code"] == "RP105"

        prompts = {p["name"]: p for p in io.request("prompts/list")["result"]["prompts"]}
        assert set(prompts) == set(mcp_server.PROMPTS)
        assert [a["name"] for a in prompts["capital_model"]["arguments"]] == ["lines", "level"]
        got = io.request("prompts/get", {"name": "capital_model", "arguments": {"lines": "motor, property"}})
        text = got["result"]["messages"][0]["content"]["text"]
        assert text == mcp_server.render_prompt("capital_model", lines="motor, property")

        # the new tools, over the wire, and the ledger they were recorded to
        names = {t["name"] for t in io.request("tools/list")["result"]["tools"]}
        assert {"capital_aggregate", "capital_allocation", "copula_dependence", "nearest_correlation", "sensitivity",
                "scenario_table", "riskpy_audit"} <= names
        agg = io.request("tools/call", {"name": "capital_aggregate", "arguments": {
            "standalone": [3, 4], "correlation": [[1, 0], [0, 1]]}})["result"]
        assert agg["structuredContent"]["total"] == pytest.approx(5.0)
        sens = io.request("tools/call", {"name": "sensitivity", "arguments": {
            "tool": "black_scholes", "arguments": BS, "parameter": "S"}})["result"]
        assert sens["structuredContent"]["outputs"]["result"]["derivative"] == pytest.approx(0.6368306511756191, abs=1e-7)
        refused = io.request("tools/call", {"name": "copula_dependence", "arguments": {"family": "gumbel"}})["result"]
        assert refused["isError"] is True and refused["structuredContent"]["error"]["code"] == "RP113"
        status = io.request("tools/call", {"name": "riskpy_audit", "arguments": {"check": True}})["result"]
        status = status["structuredContent"]
        assert status["recording"] is True and status["entries"] == 3 and status["chain_ok"] is True
        assert status["replay"]["matched"] == 3
        resource = json.loads(io.request("resources/read", {"uri": "riskpy://audit"})["result"]["contents"][0]["text"])
        assert resource["head"] == status["head"]
    finally:
        proc.kill()
        proc.wait(timeout=10)
    entries = audit.load(ledger)
    assert [e["tool"] for e in entries] == ["capital_aggregate", "sensitivity", "copula_dependence"]
    assert all(e["via"] == "mcp" for e in entries) and entries[-1]["digest"] == status["head"]
    assert audit.verify_chain(ledger).ok
