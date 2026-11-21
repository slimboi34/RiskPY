"""riskpy.audit, 0.8.0 refinements: the ledger survives torn and hostile lines, stores arguments
exactly, keeps the report inert, and checks the head that was quoted.

Each test names the defect it guards against. As in test_audit.py, digests that matter are
recomputed here with hashlib from the documented canonical form.
"""
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import pytest

from riskpy import audit, tools
from riskpy.errors import InputError

BS = {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2}


def canonical_digest(entry):
    body = {k: v for k, v in entry.items() if k != "digest"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def chained(entries):
    """Hand-made entries chained with hashlib, as another writer (or an attacker) would."""
    prev, out = "0" * 64, []
    for seq, entry in enumerate(entries, 1):
        entry = {"seq": seq, **entry, "prev": prev}
        entry["digest"] = prev = canonical_digest(entry)
        out.append(entry)
    return out


def write_lines(path, entries):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(e) + "\n" for e in entries))


def cli(*argv, cwd, env=None):
    return subprocess.run([sys.executable, "-m", "riskpy.audit", *argv], cwd=str(cwd), capture_output=True,
                          text=True, encoding="utf-8", timeout=120, env=env)


@pytest.fixture(autouse=True)
def no_ambient_ledger(monkeypatch):
    monkeypatch.delenv(audit.ENV, raising=False)
    audit.stop()
    yield
    audit.stop()


# ---------------------------------------------------------------------------
# The tail of the ledger: read in linear time, cached, never glued to a torn line


def test_a_huge_last_line_is_read_once_in_linear_time(tmp_path, monkeypatch):
    path = str(tmp_path / "big.jsonl")
    writer = audit.Ledger(path)
    writer.append("black_scholes", BS, result=10.45)
    writer.append("scenario_table", {"scenarios": {"x" * 20_000_000: {"S": 110}}}, result={"ok": True})
    size = os.path.getsize(path)
    assert size > 20_000_000

    # A new writer (another process) must find the last entry by reading the line once, not
    # re-splitting a growing buffer for every block (which was quadratic: minutes for 20 MB).
    read = {"bytes": 0}
    real_open = open

    class Counting:
        def __init__(self, fh):
            self.fh = fh

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.fh.close()

        def read(self, n=-1):
            data = self.fh.read(n)
            read["bytes"] += len(data)
            return data

        def seek(self, *args):
            return self.fh.seek(*args)

    with monkeypatch.context() as patch:
        patch.setattr(audit, "open", lambda p, mode="r", *a, **k: Counting(real_open(p, mode, *a, **k)),
                      raising=False)
        started = time.perf_counter()
        continuing = audit.Ledger(path)
        entry = continuing.append("black_scholes", BS, result=10.45)
        elapsed = time.perf_counter() - started
    assert read["bytes"] <= size + 1                  # the last line (and the final byte) read once
    assert elapsed < 5.0                              # 0.1 s here; the old reader took minutes
    assert entry["seq"] == 3 and entry["prev"] == writer.entries[-1]["digest"]

    # The writer remembers the tail it wrote, so its next append reads nothing at all.
    with monkeypatch.context() as patch:
        patch.setattr(audit, "_scan_tail", lambda p: pytest.fail("re-read a ledger only this object wrote"))
        started = time.perf_counter()
        continuing.append("black_scholes", BS, result=10.45)
        assert continuing.head == continuing.entries[-1]["digest"]
        assert time.perf_counter() - started < 1.0
    # ... but notices when someone else writes to the file.
    writer.append("black_scholes", BS, result=10.45)
    assert continuing.append("black_scholes", BS, result=10.45)["seq"] == 6
    assert audit.verify_chain(path).ok and audit.verify_chain(path).entries == 6


def test_error_messages_and_parameters_are_capped_in_the_ledger():
    ledger = audit.Ledger()
    entry = ledger.append("black_scholes", BS, error={"code": "RP402", "message": "m" * 50_000, "param": "p" * 5_000})
    assert len(entry["error"]["message"]) == 1001 and entry["error"]["message"].endswith("…")
    assert len(entry["error"]["param"]) == 81 and entry["error"]["code"] == "RP402"
    with pytest.raises(InputError) as exc:
        ledger.append("black_scholes", BS, error="refused")
    assert exc.value.code == "RP100" and exc.value.param == "error"
    # through the tool layer: an 8 MB argument name is kept in the arguments, not three times over
    with audit.record() as recorded:
        tools.call("annuity_certain", {"i": 0.05, "n": 10, "x" * 8_000_000: 1}, errors="return")
    stored = recorded.entries[0]
    assert len(stored["error"]["message"]) <= 1001 and len(stored["error"]["param"] or "") <= 81


def test_a_torn_last_line_is_not_glued_to_the_next_entry(tmp_path):
    path = str(tmp_path / "torn.jsonl")
    with audit.record(path):
        tools.call("annuity_certain", {"i": 0.05, "n": 10})
        tools.call("annuity_certain", {"i": 0.05, "n": 11})
    intact = open(path, encoding="utf-8").read()
    with open(path, "a", encoding="utf-8") as fh:            # a write cut short by a crash or a full disk
        fh.write(intact.splitlines()[-1][:60])
    env = {**os.environ, audit.ENV: path}
    for n in (12, 13):                                       # later processes continue the ledger, as documented
        done = subprocess.run([sys.executable, "-m", "riskpy.tools", "call", "annuity_certain",
                               json.dumps({"i": 0.05, "n": n})], env=env, cwd=str(tmp_path), capture_output=True,
                              text=True, encoding="utf-8", timeout=120)
        assert done.returncode == 0, done.stderr
    lines = open(path, encoding="utf-8").read().splitlines()
    assert len(lines) == 5 and all(line.count('"seq"') == 1 for line in lines)
    parsed = [json.loads(line) for k, line in enumerate(lines) if k != 2]
    assert [e["seq"] for e in parsed] == [1, 2, 3, 4]               # seq continues from the last good entry
    assert [e["arguments"]["n"] for e in parsed] == [10, 11, 12, 13]  # nothing lost inside the torn line
    torn = hashlib.sha256(lines[2].encode("utf-8")).hexdigest()
    assert parsed[2]["prev"] == torn                                  # chained to the damage, so it stays visible
    assert parsed[3]["prev"] == parsed[2]["digest"]

    # verify names the damaged line instead of refusing the whole ledger
    check = audit.verify_chain(path)
    assert not check.ok and check.broken_at == 3 and check.reason.startswith("damaged line")
    assert check.entries == 5 and check.head == parsed[1]["digest"]
    verified = cli("verify", path, cwd=tmp_path)
    assert verified.returncode == 1 and "BROKEN at entry 3: damaged line: line 3 is not valid JSON" in verified.stdout
    assert "Traceback" not in verified.stderr
    # replay and report still read the intact entries around it
    outcome = audit.replay(path)
    assert not outcome.ok and outcome.matched == 4 and outcome.skipped[0]["reason"].startswith("damaged line")
    text = audit.report(path)
    assert "*damaged line*" in text and "### 3. Damaged line" in text and "**BROKEN** at entry 3" in text
    # load() still refuses: a ledger is evidence
    with pytest.raises(InputError) as exc:
        audit.load(path)
    assert exc.value.code == "RP100" and "line 3 is not valid JSON" in str(exc.value)


def test_a_foreign_last_line_does_not_stop_the_recording(tmp_path):
    path = str(tmp_path / "foreign.jsonl")
    with audit.record(path):
        tools.call("annuity_certain", {"i": 0.05, "n": 10})
    for foreign in ('{"seq": Infinity, "digest": "x"}', '{"seq": 1e400}', '{"seq": "2", "digest": 5}',
                    "[" * 100_000 + "]" * 100_000, '{"seq": 3, "digest": "not hex"}'):
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(foreign + "\n")
        with audit.record(path) as ledger:
            tools.call("annuity_certain", {"i": 0.05, "n": 20})
        assert ledger.failures == 0 and len(ledger) == 1          # 0.7.0: every write failed, head raised
        last = ledger.entries[-1]
        assert ledger.head == last["digest"] == canonical_digest(last)
        assert last["prev"] == hashlib.sha256(foreign.encode()).hexdigest()
    seqs = [json.loads(line)["seq"] for line in open(path, encoding="utf-8").read().splitlines()[::2]]
    assert seqs == [1, 2, 3, 4, 5, 6]
    with open(path, "ab") as fh:                                   # bytes that are not UTF-8 at all
        fh.write(b"\xff\xfe garbage")                               # ... and no newline either
    with audit.record(path) as ledger:
        tools.call("annuity_certain", {"i": 0.05, "n": 30})
    assert ledger.failures == 0 and ledger.entries[0]["seq"] == 7
    assert open(path, "rb").read().count(b"\n") == 13              # the garbage got its own line


def test_each_entry_is_one_write(tmp_path, monkeypatch):
    path = str(tmp_path / "writes.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{"seq": 1, "torn')                               # no newline: it must be added in the same write
    calls = []
    real_write = os.write

    def counting(fd, data):
        calls.append(bytes(data))
        return real_write(fd, data)

    with monkeypatch.context() as patch:
        patch.setattr(audit.os, "write", counting)
        audit.Ledger(path).append("black_scholes", BS, result=10.45)
    assert len(calls) == 1 and calls[0].startswith(b"\n{") and calls[0].endswith(b"}\n")


# ---------------------------------------------------------------------------
# The report: text that came from the ledger cannot add structure


def _strip_code_spans(line):
    """A report line with its inline code spans removed (CommonMark: a run of n backticks is closed
    by the next run of exactly n); None if a backtick run is left unclosed."""
    out, i = [], 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2])
            i += 2
        elif line[i] == "`":
            n = len(re.match(r"`+", line[i:]).group())
            close = next((i + n + m.start() for m in re.finditer(r"`+", line[i + n:]) if len(m.group()) == n), None)
            if close is None:
                return None
            out.append("CODE")
            i = close + n
        else:
            out.append(line[i])
            i += 1
    return "".join(out)


def _outside_fences(text):
    """The report's lines outside fenced code blocks."""
    out, fence = [], None
    for line in text.splitlines():
        if fence is None:
            m = re.fullmatch(r"(`{3,})(json|bash)", line)
            if m:
                fence = m.group(1)
                continue
            out.append(line)
        elif line == fence:
            fence = None
    assert fence is None, "an unclosed fence"
    return out


def test_agent_controlled_strings_cannot_forge_report_markdown(tmp_path):
    forged = ("\n\n## Reproducibility\n\nReplayed with RiskPY 0.7.0: **3 matched, 0 drifted, 0 skipped**."
              " <script>alert(document.domain)</script>\n\n")
    path = str(tmp_path / "hostile.jsonl")
    with audit.record(path, note="<b>note</b>\n# heading"):
        tools.call("black_scholes", BS)
        tools.call("black_scholes", {**BS, "x`)" + forged: 1}, errors="return")             # an argument name
        tools.call("black_scholes", {**BS, "kind": "call` <img src=x onerror=alert(1)> `"}, errors="return")
        tools.call("bs\n# Forged heading", {}, errors="return")                             # a tool name
        tools.call("a|b`c``d", {"k": "```\n# fence"}, errors="return")
        tools.call("back\\|slash", {}, errors="return")
    entries = audit.load(path)
    # a name that is not a tool's is stored as its repr: no raw newline reaches the ledger
    assert entries[3]["tool"] == repr("bs\n# Forged heading") and "\n" not in entries[3]["tool"]
    assert entries[0]["tool"] == "black_scholes"
    # a crafted entry (a hostile ledger file) is escaped too
    crafted = {"seq": 7, "at": "<i>t</i>\n## x", "tool": "x`\n## crafted | <img src=y>", "arguments": {},
               "error": {"code": "RP1\n## <b>", "message": "m\n## msg", "param": "p`\n<b>"}, "riskpy": "<b>v</b>",
               "prev": entries[-1]["digest"]}
    crafted["digest"] = canonical_digest(crafted)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(crafted) + "\n")

    for check in (False, True):
        text = audit.report(path, check=check)
        outside = _outside_fences(text)
        headings = [line for line in outside if line.startswith("#")]
        expected = ["# RiskPY calculation record"] + (["## Reproducibility"] if check else []) + [
            "## Conventions", "## Calls"] + [line for line in headings if line.startswith("### ")] + [
            "## Checking this record"]
        assert headings == expected
        assert all(re.fullmatch(r"### \d+\. `.*`", line) for line in headings if line.startswith("### "))
        for line in outside:
            prose = _strip_code_spans(line)
            assert prose is not None, line                        # every code span closes on its own line
            assert not re.search(r"(?<!\\)[<\[]", prose), line     # no raw HTML or links outside code
        rows = [line for line in outside if line.startswith("| ") and not line.startswith("| #")]
        assert len(rows) == 7
        for row in rows:                                          # every row of the calls table has five cells
            assert len(re.split(r"(?<!\\)\|", row.replace("\\\\", ""))) == 7, row
    assert "riskpy-audit verify --head " + audit.verify_chain(path).head in text


def test_report_code_spans_and_fences_are_longer_than_any_backtick_run():
    assert audit._code("a`b") == "``a`b``" and audit._code("`x`") == "`` `x` ``"
    assert audit._code("a_b", cell=True) == "`a_b`" and audit._code("line\nbreak") == "`line\\nbreak`"
    assert audit._code("a|b", cell=True) == "a\\|b" and audit._code("a\\|b", cell=True) == "a\\\\\\|b"
    assert audit._md("<b>*x*</b> [l](u) | `c`") == "\\<b\\>\\*x\\*\\</b\\> \\[l\\](u) \\| \\`c\\`"
    assert audit._fenced('{"k": "````"}') == ["`````json", '{"k": "````"}', "`````"]
    entries = chained([{"tool": "black_scholes", "arguments": {"s": "\udc80"}, "result": 1.0}])
    assert "\\udc80" in audit.report(entries)               # a lone surrogate cannot break writing the report


# ---------------------------------------------------------------------------
# Arguments are stored exactly, so replay does not report false drift


def test_nan_inf_and_bytes_arguments_replay_without_false_drift():
    cases = [
        {**BS, "sigma": float("nan")}, {**BS, "S": float("inf")}, {**BS, "r": float("-inf")},
        {**BS, "kind": b"put"}, {**BS, "kind": bytearray(b"call")},
    ]
    for arguments in cases:
        with audit.record() as ledger:
            live = tools.call("black_scholes", arguments, errors="return")
        assert "error" in live
        outcome = audit.replay(ledger)
        assert outcome.ok and outcome.matched == 1 and outcome.skipped == [], (arguments, outcome.summary())
    with audit.record() as ledger:
        tools.call("capital_aggregate", {"standalone": [100.0, float("nan")], "correlation": [[1, 0.25], [0.25, 1]]},
                   errors="return")
    assert ledger.entries[0]["arguments"]["standalone"] == [100.0, {"$riskpy": "float", "value": "nan"}]
    assert audit.replay(ledger).matched == 1
    stored = audit.Ledger().append("t", {"b": b"\x00\xff", "inf": float("-inf")}, result=None)["arguments"]
    assert stored == {"b": {"$riskpy": "bytes", "base64": base64.b64encode(b"\x00\xff").decode()},
                      "inf": {"$riskpy": "float", "value": "-inf"}}


def test_values_round_trip_through_the_ledger_encoding():
    huge = 7 ** 3000
    for value in [None, True, 0, -3, 1.5, "s", "\udc80", [1, [2, [3]]], {"a": {"b": 1}}, float("inf"),
                  float("-inf"), b"", b"\x00\x01", {1: "int key", (2): "x"}, {"$riskpy": "literal key"},
                  {3, 4}, frozenset({"a"}), huge, -huge]:
        stored = audit._encode(value)
        json.dumps(stored, allow_nan=False)                  # always plain, finite JSON
        assert audit._decode(json.loads(json.dumps(stored))) == value, value
    nan = audit._decode(audit._encode(float("nan")))
    assert nan != nan
    assert audit._decode(audit._encode((1, 2))) == [1, 2]    # tuples come back as lists, as tools treat them
    # anything else is marked, and replay skips it with the reason instead of reporting drift
    marked = audit._encode(object())
    assert marked["$riskpy"] == "unrepresentable" and marked["type"] == "object"
    with pytest.raises(audit._CannotRebuild):
        audit._decode(marked)
    for malformed in ({"$riskpy": "float", "value": "1"}, {"$riskpy": "bytes", "base64": "!!"},
                      {"$riskpy": "nope"}, {"$riskpy": "ndarray", "dtype": "complex128", "shape": [], "data": []}):
        with pytest.raises(audit._CannotRebuild):
            audit._decode(malformed)


def test_numpy_arguments_replay_without_false_drift():
    np = pytest.importorskip("numpy")
    cases = [
        ("capital_aggregate", {"standalone": np.array([100, 60]), "correlation": np.array([[1, 0.25], [0.25, 1]])}),
        ("capital_aggregate", {"standalone": np.array([[100.0, np.nan]]), "correlation": [[1, 0.25], [0.25, 1]]}),
        ("annuity_certain", {"i": np.float32(0.05), "n": np.int64(10)}),
        ("black_scholes", {**BS, "explain": np.bool_(True)}),
    ]
    for name, arguments in cases:
        with audit.record() as ledger:
            tools.call(name, arguments, errors="return")
        outcome = audit.replay(ledger)
        assert outcome.ok and outcome.matched == 1, (name, outcome.summary())
    stored = ledger.entries[0]["arguments"]["explain"]
    assert stored == {"$riskpy": "numpy", "dtype": "bool", "value": True}
    for value in (np.arange(6, dtype=np.int32).reshape(2, 3), np.array([1.5, np.inf]), np.array(["a", "bc"]),
                  np.array([b"x"]), np.array([[1, "a"], [None, 2.5]], dtype=object), np.float16(0.5),
                  np.uint8(255), np.array(3.0)):
        back = audit._decode(json.loads(json.dumps(audit._encode(value))))
        assert type(back) is type(value) and back.dtype == value.dtype and np.array_equal(back, value)
        assert np.shape(back) == np.shape(value)


def test_ledgers_written_by_0_7_0_still_verify_and_replay():
    # 0.7.0 stored NaN as null and arrays as lists; those entries replay exactly as before.
    base = {"at": "2026-11-20T09:00:00.000+00:00", "ms": 1.0, "via": "mcp", "riskpy": "0.7.0", "python": "3.12.13"}
    entries = chained([
        {**base, "tool": "black_scholes", "arguments": BS, "result": 10.450583572185565},
        {**base, "tool": "black_scholes", "arguments": {**BS, "sigma": None},
         "error": {"code": "RP404", "message": "sigma must be a number", "param": "sigma"}},
        {**base, "tool": "capital_aggregate", "arguments": {"standalone": [3, 4], "correlation": [[1, 0], [0, 1]]},
         "result": {"total": 5.0, "undiversified": 7.0, "diversification_benefit": 2.0,
                    "diversification_ratio": 2 / 7, "standalone": {"risk_1": 3.0, "risk_2": 4.0},
                    "contributions": {"risk_1": 1.8, "risk_2": 3.2}, "shares": {"risk_1": 0.36, "risk_2": 0.64}}},
        {**base, "tool": "a_tool_since_retired", "arguments": {}, "result": 1.0},
    ])
    assert audit.verify_chain(entries).ok
    outcome = audit.replay(entries)
    assert outcome.ok and outcome.matched == 3 and outcome.recorded_versions == ["0.7.0"]
    assert [s["reason"] for s in outcome.skipped] == ["this installation has no such tool"]


def test_arguments_too_deep_to_store_leave_a_chained_placeholder():
    deep = {"S": 110}
    node = deep
    for _ in range(400):
        node["x"] = {}
        node = node["x"]
    ledger = audit.Ledger()
    entry = ledger.append("scenario_table", {"tool": "black_scholes", "scenarios": {"deep": deep}}, result={"ok": 1})
    placeholder = entry["arguments"]
    assert placeholder["$riskpy"] == "unserialisable" and placeholder["type"] == "dict"
    assert "nested more than" in placeholder["reason"] and len(placeholder["sha256"]) == 64
    assert entry["result"] == {"ok": 1} and entry["digest"] == canonical_digest(entry) and ledger.failures == 0
    outcome = audit.replay(ledger)
    assert outcome.ok and outcome.skipped[0]["reason"].startswith("the call's arguments could not be serialised")
    # through the tool layer: a successful call is in the chain even when its arguments are not storable
    with audit.record() as recorded:
        tools.call("black_scholes", BS, errors="return")
        audit._record("black_scholes", {"deep": deep}, 10.45, None, 1.0, "python")     # as tools.call hands it over
    assert len(recorded) == 2 and recorded.failures == 0
    assert recorded.entries[1]["arguments"]["$riskpy"] == "unserialisable" and recorded.entries[1]["result"] == 10.45


def test_write_failures_say_why(tmp_path, monkeypatch):
    monkeypatch.setenv(audit.ENV, str(tmp_path / "no" / "such" / "dir" / "x.jsonl"))
    tools.call("black_scholes", BS)
    tools.call("black_scholes", BS)
    status = audit.status()
    assert status["write_failures"] == 2
    [(reason, count)] = status["write_failure_reasons"].items()
    assert reason.startswith("cannot write the file") and count == 2
    assert "could not be written" in status["how"] and "×2" in status["how"]
    # a failure that is not about the file is reported as what it was
    ledger = audit.start()
    monkeypatch.setattr(ledger, "append", lambda *a, **k: (_ for _ in ()).throw(TypeError("boom")))
    tools.call("black_scholes", BS)
    assert ledger.failure_reasons == {"TypeError: boom": 1}


# ---------------------------------------------------------------------------
# riskpy_audit(check=true) does a bounded amount of work


def test_the_status_check_replays_only_the_most_recent_entries(tmp_path, monkeypatch):
    path = str(tmp_path / "long.jsonl")
    monkeypatch.setenv(audit.ENV, path)
    for n in range(1, 61):
        tools.call("annuity_certain", {"i": 0.05, "n": n})
    status = tools.call("riskpy_audit", {"check": True})              # what an agent gets
    assert status["entries"] == 60 and status["replay"]["entries"] == 60
    assert status["replay"]["replayed"] == 50 and status["replay"]["matched"] == 50
    assert "most recent 50 of 60" in status["how"] and "riskpy-audit replay FILE" in status["how"]
    full = audit.status(check=True, replay_last=None)
    assert full["replay"]["replayed"] == 60 and full["replay"]["matched"] == 60 and "most recent" not in full["how"]
    assert audit.replay(path).matched == 60                           # the library and the CLI stay uncapped
    # the time budget: at least one entry, the newest, then it stops
    quick = audit.status(check=True, replay_last=None, replay_seconds=1e-9)
    assert quick["replay"]["replayed"] >= 1 and quick["replay"]["matched"] == quick["replay"]["replayed"]
    assert quick["replay"]["replayed"] < 60
    for bad in ({"replay_last": 0}, {"replay_last": 2.5}, {"replay_last": True}, {"replay_seconds": 0},
                {"replay_seconds": "1"}):
        with pytest.raises(InputError) as exc:
            audit.status(check=True, **bad)
        assert exc.value.code == "RP102"
    report = audit.ReplayReport(chain=audit.verify_chain([]), entries=60, matched=50, replayed=50)
    assert "the most recent 50 of 60 entries" in report.summary()


def test_the_status_check_lists_drift_in_ledger_order(tmp_path, monkeypatch):
    base = {"at": "2026-01-01T00:00:00.000+00:00", "ms": 1.0, "via": "python", "riskpy": "0.0.1", "python": "3.12.0"}
    entries = chained([{**base, "tool": "annuity_certain", "arguments": {"i": 0.05, "n": n}, "result": 1.0}
                       for n in range(1, 8)])
    path = str(tmp_path / "drift.jsonl")
    write_lines(path, entries)
    monkeypatch.setenv(audit.ENV, path)
    replayed = audit.status(check=True, replay_last=3)["replay"]
    assert [d["seq"] for d in replayed["drifted"]] == [5, 6, 7] and replayed["replayed"] == 3


# ---------------------------------------------------------------------------
# verify against the head that was quoted


def test_verify_catches_a_cut_off_tail_and_forged_appends_with_the_expected_head(tmp_path):
    path = str(tmp_path / "q4.jsonl")
    with audit.record(path) as ledger:
        tools.call("black_scholes", BS)
        tools.call("expected_loss", {"pd": 0.02, "lgd": 0.45, "ead": 1_000_000})
        tools.call("basel_irb_capital", {"pd": 0.01, "lgd": 0.45, "ead": 1_000_000})
    quoted = ledger.head
    lines = open(path, encoding="utf-8").read().splitlines()
    assert audit.verify_chain(path, expected_head=quoted, expected_entries=3).ok
    assert audit.verify_chain(path, expected_head=quoted.upper()).ok             # hex in any case

    dropped = str(tmp_path / "dropped.jsonl")
    open(dropped, "w", encoding="utf-8").write("\n".join(lines[:-1]) + "\n")
    assert audit.verify_chain(dropped).ok                                         # still a valid chain ...
    cut = audit.verify_chain(dropped, expected_head=quoted)                       # ... but not the one quoted
    assert not cut.ok and cut.broken_at is None and "removed from the end" in cut.reason
    counted = audit.verify_chain(dropped, expected_entries=3)
    assert not counted.ok and counted.broken_at == 3 and "the last 1 entry was removed" in counted.reason
    empty = str(tmp_path / "empty.jsonl")
    open(empty, "w").close()
    assert not audit.verify_chain(empty, expected_head=quoted).ok

    last = json.loads(lines[-1])
    forged = {"seq": 4, "at": last["at"], "tool": "expected_loss", "arguments": {"pd": 0.5, "lgd": 1, "ead": 1},
              "ms": 0.1, "via": "mcp", "riskpy": last["riskpy"], "python": last["python"], "result": 0.5,
              "prev": last["digest"]}
    forged["digest"] = canonical_digest(forged)
    appended = str(tmp_path / "appended.jsonl")
    open(appended, "w", encoding="utf-8").write("\n".join(lines + [json.dumps(forged)]) + "\n")
    extra = audit.verify_chain(appended, expected_head=quoted)
    assert not extra.ok and extra.broken_at == 4 and "1 entry was added after it" in extra.reason
    assert not audit.verify_chain(appended, expected_entries=3).ok

    for bad in ("abc", 42, "g" * 64):
        with pytest.raises(InputError) as exc:
            audit.verify_chain(path, expected_head=bad)
        assert exc.value.code == "RP100" and exc.value.param == "expected_head"
    for bad in (-1, 1.5, True):
        with pytest.raises(InputError):
            audit.verify_chain(path, expected_entries=bad)

    # the CLI
    assert cli("verify", "--head", quoted, path, cwd=tmp_path).returncode == 0
    ok = cli("verify", "--head", quoted, "--entries", "3", path, cwd=tmp_path)
    assert ok.returncode == 0 and "(the expected head)" in ok.stdout
    broken = cli("verify", "--head", quoted, dropped, cwd=tmp_path)
    assert broken.returncode == 1 and broken.stdout.startswith("BROKEN: the chain is intact but ends at head")
    assert cli("verify", "--entries", "3", dropped, cwd=tmp_path).returncode == 1
    tail = cli("verify", "--head", quoted, appended, cwd=tmp_path)
    assert tail.returncode == 1 and "BROKEN at entry 4" in tail.stdout
    garbage = cli("verify", "--head", "nothex", path, cwd=tmp_path)
    assert garbage.returncode == 1 and garbage.stderr.startswith("riskpy-audit: RP100")
    assert "Traceback" not in garbage.stderr
    assert "--head HEX" in cli("verify", "--help", cwd=tmp_path).stdout
    # the report prints the command with the real head
    assert f"riskpy-audit verify --head {quoted} LEDGER.jsonl" in audit.report(path)


# ---------------------------------------------------------------------------
# replay: no re-entry, no foreign tools


def test_replay_never_records_and_skips_tools_a_ledger_never_records(tmp_path, monkeypatch):
    entries = chained([
        {"tool": "riskpy_audit", "arguments": {"check": True}, "result": {}},
        {"tool": "riskpy_status", "arguments": {}, "result": {}},
        {"tool": "annuity_certain", "arguments": {"i": 0.05, "n": 10}, "result": 7.721734929184812},
    ])
    received = str(tmp_path / "received.jsonl")
    write_lines(received, entries)
    # replaying while recording to the very file being replayed: nothing is added to the evidence
    monkeypatch.setenv(audit.ENV, received)
    outcome = audit.replay(received)
    assert outcome.matched == 1 and [s["seq"] for s in outcome.skipped] == [1, 2]
    assert all(s["reason"].startswith("foreign entry") for s in outcome.skipped)
    assert len(audit.load(received)) == 3
    done = cli("replay", received, cwd=tmp_path, env={**os.environ, audit.ENV: received})
    assert done.returncode == 0 and len(audit.load(received)) == 3
    # a nested replay restores the flag it found instead of switching recording back on
    audit._replaying.on = True
    try:
        audit.replay(entries)
        assert audit._replaying.on is True
    finally:
        audit._replaying.on = False
    with audit.record() as watching:
        audit.replay(entries)
        tools.call("black_scholes", BS)                  # after the replay, recording works again
    assert len(watching) == 1


# ---------------------------------------------------------------------------
# Duplicate keys, damaged bytes and the CLI's errors


def test_a_line_that_repeats_a_key_is_damaged(tmp_path):
    path = str(tmp_path / "dup.jsonl")
    with audit.record(path):
        tools.call("expected_loss", {"pd": 0.02, "lgd": 0.45, "ead": 1_000_000})
    line = open(path, encoding="utf-8").read().rstrip("\n")
    tampered = '{"result": 90.0, ' + line[1:]               # a first-wins reader sees 90.0; json.loads the original
    open(path, "w", encoding="utf-8").write(tampered + "\n")
    check = audit.verify_chain(path)
    assert not check.ok and check.broken_at == 1 and "repeats the key 'result'" in check.reason
    with pytest.raises(InputError) as exc:
        audit.load(path)
    assert exc.value.code == "RP100" and "repeats the key" in str(exc.value)
    assert "repeats the key 'b'" in audit._parse_line(b'{"a": {"b": 1, "b": 2}}')     # at any depth
    # formatting is not evidence: the same entry re-serialised with other spacing and key order still verifies
    entry = json.loads(line)
    open(path, "w", encoding="utf-8").write(json.dumps(dict(reversed(list(entry.items()))), indent=None,
                                                       separators=(" ,  ", " : ")) + "\n")
    assert audit.verify_chain(path).ok


def test_riskpy_audit_answers_damaged_ledgers_with_codes_not_tracebacks(tmp_path):
    deep = str(tmp_path / "deep.jsonl")
    with audit.record(deep):
        tools.call("annuity_certain", {"i": 0.05, "n": 10})
    with open(deep, "a", encoding="utf-8") as fh:
        fh.write("[" * 100_000 + "]" * 100_000 + "\n")
    bad_error = {"seq": 1, "tool": "annuity_certain", "arguments": {"i": 0.05, "n": 10}, "error": "refused",
                 "prev": audit.GENESIS}
    bad_error["digest"] = canonical_digest(bad_error)
    error_string = str(tmp_path / "error-string.jsonl")
    write_lines(error_string, [bad_error])
    binary = str(tmp_path / "binary.jsonl")
    open(binary, "wb").write(b'\xff\xfe{"seq":1}\n')
    for path, commands in ((deep, (["verify"], ["replay"], ["report"])),
                           (error_string, (["verify"], ["replay"], ["report", "--check"])),
                           (binary, (["verify"], ["replay"], ["report"]))):
        for command in commands:
            done = cli(*command, path, cwd=tmp_path)
            assert "Traceback" not in done.stderr, (path, command, done.stderr)
    assert "line 2 is nested too deeply" in cli("verify", deep, cwd=tmp_path).stdout
    assert "line 1 is not UTF-8 text" in cli("verify", binary, cwd=tmp_path).stdout
    skipped = audit.replay(error_string).skipped
    assert skipped[0]["reason"] == "the entry's error is not an object {code, message, param}"
    for path in (deep, binary):
        with pytest.raises(InputError) as exc:
            audit.load(path)
        assert exc.value.code == "RP100"
    # an unwritable output file is RP505, exit 1
    out = cli("report", error_string, "-o", str(tmp_path / "no" / "dir" / "record.md"), cwd=tmp_path)
    assert out.returncode == 1 and out.stderr.startswith("riskpy-audit: RP505 cannot write")
    assert "Traceback" not in out.stderr


def test_an_unexpected_error_in_the_cli_is_rp900_not_a_traceback(tmp_path, monkeypatch, capsys):
    path = str(tmp_path / "x.jsonl")
    write_lines(path, chained([{"tool": "annuity_certain", "arguments": {"i": 0.05, "n": 10}, "result": 7.7}]))
    monkeypatch.setattr(audit, "report", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("surprise")))
    assert audit.main(["report", path]) == 1
    assert capsys.readouterr().err.strip() == "riskpy-audit: RP900 RuntimeError: surprise"


def test_the_status_reports_a_damaged_or_unreadable_ledger(tmp_path, monkeypatch):
    path = str(tmp_path / "bad.jsonl")
    monkeypatch.setenv(audit.ENV, path)
    tools.call("black_scholes", BS)
    with open(path, "ab") as fh:
        fh.write(b"\xff\xfe garbage\n")
    status = tools.call("riskpy_audit", {})
    assert status["recording"] is True and status["chain_ok"] is False and status["entries"] == 2
    assert status["broken_at"] == 2 and "not UTF-8" in status["reason"]
    folder = tmp_path / "a-directory"
    folder.mkdir()
    monkeypatch.setenv(audit.ENV, str(folder))
    unreadable = audit.status()
    assert unreadable["chain_ok"] is False and unreadable["how"].startswith("The ledger cannot be read (RP505)")
