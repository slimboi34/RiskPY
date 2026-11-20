"""An audit ledger for tool calls: record them in full, prove the record is intact, replay it.

A number in a board paper, a reserve report or an agent's answer raises the same three
questions: what exactly was calculated, has the record been altered since, and does the
library still give that answer? :func:`riskpy.tools.trace` shows recent calls with their
arguments abbreviated; this module keeps the whole call and makes it checkable.

    from riskpy import audit, tools

    with audit.record("q4-capital.jsonl", note="Q4 capital run") as ledger:
        tools.call("black_scholes", {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2})
        tools.call("capital_aggregate", {"standalone": [100, 60], "correlation": [[1, 0.25], [0.25, 1]]})

    ledger.head                                # the SHA-256 digest that pins the whole run
    audit.verify_chain("q4-capital.jsonl")     # has any entry been edited, dropped or reordered?
    audit.replay("q4-capital.jsonl")           # recompute every call and compare
    print(audit.report("q4-capital.jsonl"))    # a Markdown calculation record

    RISKPY_AUDIT_FILE=run.jsonl riskpy-mcp     # the same ledger for an agent's MCP session
    riskpy-audit verify run.jsonl
    riskpy-audit replay run.jsonl
    riskpy-audit report run.jsonl -o record.md

Each ledger line is one JSON object: ``{seq, at, tool, arguments, result | error, ms, via,
riskpy, python, note, prev, digest}``. ``digest`` is the SHA-256 of the entry's canonical
JSON (keys sorted, no whitespace, ASCII) without the digest itself, and ``prev`` is the
digest of the entry before it, so the entries form a hash chain: changing, removing or
reordering any of them breaks every digest after it. The last digest — the *head* — is one
short string that identifies the whole run; quote it next to the numbers.

The chain shows that a ledger is internally consistent. It cannot show that the whole file
was not regenerated: that needs the head to be kept somewhere else (the report, an email, a
commit). The ledger assumes one writer at a time.

Introspection tools (``riskpy_status`` and friends) are not recorded. A simulation called
with ``seed=0`` draws a fresh seed, so it is recorded but cannot be replayed; pass a seed
for any number that will be quoted.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import inspect
import json
import math
import os
import platform
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from .errors import InputError

__all__ = ["Ledger", "ChainReport", "ReplayReport", "ENV", "GENESIS", "start", "stop", "active", "record", "load",
           "digest", "verify_chain", "replay", "report", "status", "main"]

ENV = "RISKPY_AUDIT_FILE"
GENESIS = "0" * 64          # the "previous digest" of the first entry

# Result fields that legitimately differ between two runs of the same call.
_VOLATILE_FIELDS: Dict[str, Tuple[str, ...]] = {"verify": ("seconds", "version")}

_lock = threading.RLock()
_active: Optional["Ledger"] = None
_env_ledgers: Dict[str, "Ledger"] = {}
_replaying = threading.local()


def digest(entry: Dict[str, Any]) -> str:
    """The SHA-256 hex digest of a ledger entry, ignoring any ``digest`` key it already has.

    entry: a ledger entry (a dict of JSON values)

    The entry is serialised canonically — keys sorted, no whitespace, non-ASCII escaped —
    so the same entry gives the same digest on any machine and after any JSON round trip.

    Example::

        audit.digest({"seq": 1, "tool": "expected_loss"})     # 'a3c1…' (64 hex characters)
    """
    body = {k: v for k, v in entry.items() if k != "digest"}
    try:
        text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise InputError(f"a ledger entry must hold only finite JSON values: {exc}", code="RP100",
                         param="entry") from None
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _last_line(path: str) -> Optional[str]:
    """The last non-empty line of a file, read from the end (the ledger may be large)."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            block, tail = 4096, b""
            while size > 0:
                step = min(block, size)
                size -= step
                fh.seek(size)
                tail = fh.read(step) + tail
                lines = [ln for ln in tail.splitlines() if ln.strip()]
                if len(lines) > 1 or (lines and size == 0):
                    return lines[-1].decode("utf-8", errors="replace")
            return None
    except OSError:
        return None


class Ledger:
    """An append-only, hash-chained record of tool calls.

    path: a JSON-lines file to append to (created if missing; an existing ledger is
        continued from its last entry), or None to keep the entries in memory only
    note: a free-text label written into every entry — the run, the report, the author

    Attributes: ``path``, ``note``, ``entries`` (the entries this object has written, oldest
    first), ``failures`` (entries that could not be written to the file). Usually made by
    :func:`start` or :func:`record` rather than directly.

    Example::

        ledger = audit.start("run.jsonl", note="pricing review")
        tools.call("expected_loss", {"pd": 0.02, "lgd": 0.45, "ead": 1_000_000})
        ledger.head, len(ledger)
        audit.stop()
    """

    def __init__(self, path: Optional[str] = None, note: Optional[str] = None):
        if path is not None and not isinstance(path, (str, os.PathLike)):
            raise InputError(f"path must be a file path or None, got {path!r}", code="RP100", param="path")
        if note is not None and not isinstance(note, str):
            raise InputError(f"note must be a string, got {note!r}", code="RP100", param="note")
        self.path: Optional[str] = os.fspath(path) if path is not None else None
        self.note = note
        self.entries: List[dict] = []
        self.failures = 0

    def _tail(self) -> Tuple[int, str]:
        """(sequence number, digest) of the last entry: from the file when there is one, so
        that several processes appending one after another keep a single chain."""
        if self.path is not None:
            line = _last_line(self.path)
            if line is not None:
                try:
                    last = json.loads(line)
                    return int(last["seq"]), str(last["digest"])
                except (ValueError, KeyError, TypeError):
                    # A foreign or torn last line: chain from its hash so the break is visible to verify_chain.
                    return len(self.entries), hashlib.sha256(line.encode("utf-8", "replace")).hexdigest()
            return 0, GENESIS
        if self.entries:
            return self.entries[-1]["seq"], self.entries[-1]["digest"]
        return 0, GENESIS

    @property
    def head(self) -> str:
        """The digest of the last entry (64 zeros for an empty ledger): one string that pins the whole run."""
        with _lock:
            return self._tail()[1]

    def append(self, tool: str, arguments: Any, result: Any = None, error: Optional[dict] = None,
               ms: float = 0.0, via: str = "python") -> dict:
        """Add one call to the ledger and return the entry written.

        tool: the tool's name
        arguments: the arguments exactly as sent (JSON values)
        result: the JSON result of a successful call
        error: ``{code, message, param}`` for a failed call, in place of ``result``
        ms: how long the call took, in milliseconds
        via: "python", "cli" or "mcp"

        :func:`riskpy.tools.call` does this for every calculator call while a ledger is
        active; call it yourself only to record a calculation made outside the tool layer.
        Raises ``OSError`` if the file cannot be written.

        Example::

            ledger.append("my_model", {"scenario": "base"}, result={"reserve": 1.25e6})
        """
        from . import __version__
        from .tools import _jsonable

        entry: Dict[str, Any] = {
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
            "tool": str(tool),
            "arguments": _jsonable(arguments),
            "ms": round(float(ms), 3) if math.isfinite(float(ms)) else None,
            "via": str(via),
            "riskpy": __version__,
            "python": platform.python_version(),
        }
        if self.note:
            entry["note"] = self.note
        if error is not None:
            entry["error"] = {"code": error.get("code"), "message": error.get("message"), "param": error.get("param")}
        else:
            entry["result"] = _jsonable(result)
        with _lock:
            seq, prev = self._tail()
            entry = {"seq": seq + 1, **entry, "prev": prev}
            entry["digest"] = digest(entry)
            if self.path is not None:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=True, allow_nan=False) + "\n")
            self.entries.append(entry)
        return entry

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        where = self.path or "memory"
        return f"<Ledger {where} entries={len(self.entries)} head={self.head[:12]}>"


def start(path: Optional[str] = None, note: Optional[str] = None) -> Ledger:
    """Start recording every calculator call made through :func:`riskpy.tools.call` in this process.

    path: a JSON-lines file to append to, or None to record in memory
    note: a label written into every entry

    Returns the :class:`Ledger`. Recording continues until :func:`stop`; starting again
    replaces the active ledger. Setting the environment variable ``RISKPY_AUDIT_FILE`` does
    the same for a whole process (the CLI, the MCP server) without any code.

    Example::

        ledger = audit.start("run.jsonl")
        tools.call("annuity_certain", {"i": 0.05, "n": 10})
        audit.stop()
    """
    global _active
    ledger = Ledger(path, note)
    with _lock:
        _active = ledger
    return ledger


def stop() -> Optional[Ledger]:
    """Stop the recording that :func:`start` began; returns its ledger (None if none was active).

    A ledger named by ``RISKPY_AUDIT_FILE`` is not affected: unset the variable to stop it.
    """
    global _active
    with _lock:
        ledger, _active = _active, None
    return ledger


def active() -> Optional[Ledger]:
    """The ledger calls are being recorded to: the one from :func:`start`, else the file
    named by ``RISKPY_AUDIT_FILE``, else None."""
    with _lock:
        if _active is not None:
            return _active
        path = os.environ.get(ENV)
        if not path:
            return None
        if path not in _env_ledgers:
            _env_ledgers[path] = Ledger(path)
        return _env_ledgers[path]


@contextlib.contextmanager
def record(path: Optional[str] = None, note: Optional[str] = None) -> Iterator[Ledger]:
    """Record the tool calls made inside a ``with`` block.

    path: a JSON-lines file to append to, or None to record in memory
    note: a label written into every entry

    Yields the :class:`Ledger`; the previously active ledger (if any) resumes afterwards.

    Example::

        with audit.record(note="IRB check") as ledger:
            tools.call("basel_irb_capital", {"pd": 0.01, "lgd": 0.45, "ead": 1_000_000})
        ledger.entries[0]["result"]["risk_weight"]     # 0.9232…
    """
    global _active
    ledger = Ledger(path, note)
    with _lock:
        previous, _active = _active, ledger
    try:
        yield ledger
    finally:
        with _lock:
            _active = previous


def _record(name: Any, arguments: Any, result: Any, error: Optional[dict], ms: float, via: str) -> None:
    """Called by tools.call after every calculator call. Never raises: an audit trail that
    cannot be written is counted (``Ledger.failures``), not allowed to break the calculation."""
    if getattr(_replaying, "on", False):
        return
    ledger = active()
    if ledger is None:
        return
    try:
        ledger.append(str(name)[:80], arguments, result, error, ms, via)
    except Exception:  # noqa: BLE001
        ledger.failures += 1


# ---------------------------------------------------------------------------
# Reading, checking and replaying a ledger


Source = Union[str, "os.PathLike[str]", Ledger, Iterable[dict]]


def load(source: Source) -> List[dict]:
    """The entries of a ledger as a list of dicts, oldest first.

    source: a ledger file path, a :class:`Ledger`, or an iterable of entries

    Raises RP505 if the file cannot be read and RP100 if a line is not a JSON object — a
    ledger is evidence, so a damaged one is reported rather than skipped over.

    Example::

        for entry in audit.load("run.jsonl"):
            print(entry["seq"], entry["tool"], entry.get("result"))
    """
    if isinstance(source, Ledger):
        if source.path is None:
            return [dict(e) for e in source.entries]
        source = source.path
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        entries: List[dict] = []
        try:
            with open(path, encoding="utf-8") as fh:
                for number, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        raise InputError(f"{path}: line {number} is not valid JSON", code="RP100", param="source",
                                         hint="A ledger is one JSON object per line; this one is damaged.") from None
                    if not isinstance(entry, dict):
                        raise InputError(f"{path}: line {number} is not a JSON object", code="RP100", param="source")
                    entries.append(entry)
        except OSError as exc:
            raise InputError(f"cannot read the ledger {path!r}: {exc.strerror or exc}", code="RP505",
                             param="source") from None
        return entries
    try:
        entries = [dict(e) for e in source]
    except (TypeError, ValueError):
        raise InputError("source must be a ledger file path, a Ledger or a list of entries", code="RP100",
                         param="source") from None
    return entries


@dataclass(frozen=True)
class ChainReport:
    """The outcome of :func:`verify_chain`.

    Fields: ``ok``; ``entries`` (how many were checked); ``head`` (the last digest, or 64
    zeros for an empty ledger); ``broken_at`` (the ``seq`` — or 1-based position — of the
    first bad entry, None when intact); ``reason`` (what is wrong with it).
    """

    ok: bool
    entries: int
    head: str
    broken_at: Optional[int] = None
    reason: Optional[str] = None

    def __bool__(self) -> bool:
        return self.ok

    def to_dict(self) -> Dict[str, Any]:
        """The report as plain JSON-able data."""
        return {"ok": self.ok, "entries": self.entries, "head": self.head, "broken_at": self.broken_at,
                "reason": self.reason}


def verify_chain(source: Source) -> ChainReport:
    """Check that a ledger is intact: every digest matches its entry and links to the one before.

    source: a ledger file path, a :class:`Ledger`, or a list of entries

    Returns a :class:`ChainReport`, truthy when the chain is intact. Any edit to a recorded
    argument or result, a deleted or inserted entry, or a reordering breaks it, and
    ``broken_at`` names the first entry affected.

    Example::

        check = audit.verify_chain("run.jsonl")
        check.ok, check.entries, check.head
    """
    entries = load(source)
    prev = GENESIS
    for position, entry in enumerate(entries, 1):
        seq = entry.get("seq", position)
        where = seq if isinstance(seq, int) and not isinstance(seq, bool) else position
        if not isinstance(entry.get("digest"), str):
            return ChainReport(False, len(entries), prev, where, "the entry has no digest")
        if seq != position:
            return ChainReport(False, len(entries), prev, position,
                               f"sequence number {seq!r} where {position} was expected: an entry was removed, "
                               "inserted or reordered")
        if entry.get("prev") != prev:
            return ChainReport(False, len(entries), prev, where,
                               "the entry does not link to the one before it: an entry was removed, inserted or "
                               "reordered")
        try:
            expected = digest(entry)
        except InputError:
            return ChainReport(False, len(entries), prev, where, "the entry holds a value that is not finite JSON")
        if expected != entry["digest"]:
            return ChainReport(False, len(entries), prev, where,
                               "the digest does not match the entry: its content was changed after it was recorded")
        prev = entry["digest"]
    return ChainReport(True, len(entries), prev)


def _differences(recorded: Any, replayed: Any, rel_tol: float, abs_tol: float, path: str, skip: Tuple[str, ...],
                 out: List[dict]) -> None:
    def is_num(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    if is_num(recorded) and is_num(replayed):
        gap = abs(float(recorded) - float(replayed))
        scale = max(abs(float(recorded)), abs(float(replayed)))
        if gap > max(abs_tol, rel_tol * scale):
            out.append({"path": path, "recorded": recorded, "replayed": replayed,
                        "relative_difference": gap / scale if scale else None})
    elif isinstance(recorded, dict) and isinstance(replayed, dict):
        for key in sorted(set(recorded) | set(replayed)):
            if path == "result" and key in skip:
                continue
            if key not in recorded or key not in replayed:
                out.append({"path": f"{path}.{key}", "recorded": recorded.get(key), "replayed": replayed.get(key),
                            "relative_difference": None})
            else:
                _differences(recorded[key], replayed[key], rel_tol, abs_tol, f"{path}.{key}", skip, out)
    elif isinstance(recorded, list) and isinstance(replayed, list):
        if len(recorded) != len(replayed):
            out.append({"path": path, "recorded": f"<array of {len(recorded)}>",
                        "replayed": f"<array of {len(replayed)}>", "relative_difference": None})
        else:
            for k, (a, b) in enumerate(zip(recorded, replayed)):
                _differences(a, b, rel_tol, abs_tol, f"{path}[{k}]", skip, out)
    elif recorded != replayed:
        out.append({"path": path, "recorded": recorded, "replayed": replayed, "relative_difference": None})


def _unseeded(tool_name: Any, arguments: Any) -> bool:
    """True when the call drew a fresh random seed, so recomputing it cannot give the same numbers."""
    from .tools import TOOLS

    if not isinstance(arguments, dict):
        return False
    if isinstance(arguments.get("tool"), str) and isinstance(arguments.get("arguments"), dict):
        return _unseeded(arguments["tool"], arguments["arguments"])       # sensitivity / scenario_table
    t = TOOLS.get(tool_name) if isinstance(tool_name, str) else None
    if t is None:
        return False
    parameter = inspect.signature(t.fn).parameters.get("seed")
    if parameter is None:
        return False
    seed = arguments.get("seed", parameter.default)
    return seed in (0, None)


@dataclass
class ReplayReport:
    """The outcome of :func:`replay`.

    Fields: ``chain`` (the :class:`ChainReport` of the ledger); ``entries`` (how many it
    holds); ``matched`` (calls recomputed to the same result or the same error code);
    ``drifted`` (calls whose result differs — ``[{seq, tool, differences: [{path, recorded,
    replayed, relative_difference}]}]``); ``skipped`` (calls that cannot be recomputed —
    ``[{seq, tool, reason}]``); ``recorded_versions`` (the RiskPY versions in the ledger);
    ``version`` (the version that replayed it); ``rel_tol`` / ``abs_tol``.
    """

    chain: ChainReport
    entries: int
    matched: int
    drifted: List[dict] = field(default_factory=list)
    skipped: List[dict] = field(default_factory=list)
    recorded_versions: List[str] = field(default_factory=list)
    version: str = ""
    rel_tol: float = 1e-9
    abs_tol: float = 1e-12

    @property
    def ok(self) -> bool:
        """True when the chain is intact and no replayed call drifted."""
        return self.chain.ok and not self.drifted

    def __bool__(self) -> bool:
        return self.ok

    def to_dict(self) -> Dict[str, Any]:
        """The report as plain JSON-able data."""
        return {"ok": self.ok, "chain": self.chain.to_dict(), "entries": self.entries, "matched": self.matched,
                "drifted": self.drifted, "skipped": self.skipped, "recorded_versions": self.recorded_versions,
                "version": self.version, "rel_tol": self.rel_tol, "abs_tol": self.abs_tol}

    def summary(self) -> str:
        """A few lines of text: the chain, the counts, and each drifted or skipped call."""
        lines = [
            f"chain     {'intact' if self.chain.ok else 'BROKEN at entry ' + str(self.chain.broken_at)}"
            f" ({self.chain.entries} entries, head {self.chain.head[:16]}…)",
            f"replayed  {self.matched} matched, {len(self.drifted)} drifted, {len(self.skipped)} skipped "
            f"(recorded with RiskPY {', '.join(self.recorded_versions) or '?'}; replayed with {self.version})",
        ]
        if not self.chain.ok:
            lines.append(f"  entry {self.chain.broken_at}: {self.chain.reason}")
        for item in self.drifted:
            first = item["differences"][0]
            lines.append(f"  drift  #{item['seq']} {item['tool']}: {first['path']} recorded {first['recorded']!r}, "
                         f"now {first['replayed']!r}" + (f" (+{len(item['differences']) - 1} more)"
                                                         if len(item["differences"]) > 1 else ""))
        for item in self.skipped:
            lines.append(f"  skip   #{item['seq']} {item['tool']}: {item['reason']}")
        return "\n".join(lines)


def replay(source: Source, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> ReplayReport:
    """Recompute every call in a ledger with the installed library and compare the results.

    source: a ledger file path, a :class:`Ledger`, or a list of entries
    rel_tol: relative tolerance on each number (default 1e-9; platforms differ in the last digits)
    abs_tol: absolute tolerance, for numbers near zero

    Returns a :class:`ReplayReport`, truthy when the chain is intact and nothing drifted. A
    recorded failure matches when the call fails again with the same error code. Calls that
    cannot be recomputed are listed in ``skipped`` with the reason: a simulation run with
    ``seed=0`` (a fresh seed each time), or a tool this installation does not have. Replayed
    calls are not themselves recorded.

    Use it after upgrading RiskPY to see which reported numbers the new version changes,
    and before signing off a report to confirm its numbers are reproducible.

    Example::

        outcome = audit.replay("run.jsonl")
        outcome.ok, outcome.matched, outcome.drifted
        print(outcome.summary())
    """
    from . import __version__
    from . import tools

    for name, value in (("rel_tol", rel_tol), ("abs_tol", abs_tol)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value >= 0.0:
            raise InputError(f"{name} must be a number >= 0, got {value!r}", code="RP102", param=name)
    entries = load(source)
    out = ReplayReport(chain=verify_chain(entries), entries=len(entries), matched=0, version=__version__,
                       rel_tol=float(rel_tol), abs_tol=float(abs_tol),
                       recorded_versions=sorted({str(e.get("riskpy")) for e in entries if e.get("riskpy")}))
    _replaying.on = True
    try:
        for position, entry in enumerate(entries, 1):
            seq, name, arguments = entry.get("seq", position), entry.get("tool"), entry.get("arguments")
            if "result" not in entry and "error" not in entry:
                out.skipped.append({"seq": seq, "tool": name, "reason": "the entry has neither a result nor an error"})
                continue
            t = tools.TOOLS.get(name) if isinstance(name, str) else None
            if t is None and "error" not in entry:
                out.skipped.append({"seq": seq, "tool": name, "reason": "this installation has no such tool"})
                continue
            if t is not None and not t.available:
                out.skipped.append({"seq": seq, "tool": name, "reason": f"the tool needs {t.install}"})
                continue
            if "error" not in entry and _unseeded(name, arguments):
                out.skipped.append({"seq": seq, "tool": name,
                                    "reason": "run with seed=0 (a fresh seed), so it cannot be reproduced"})
                continue
            got = tools.call(name, arguments, errors="return", via="replay")
            failed = isinstance(got, dict) and set(got) == {"error"}
            differences: List[dict] = []
            if "error" in entry:
                recorded_code = (entry["error"] or {}).get("code")
                replayed_code = got["error"]["code"] if failed else None
                if recorded_code != replayed_code:
                    differences.append({"path": "error.code", "recorded": recorded_code,
                                        "replayed": replayed_code or "a result", "relative_difference": None})
            elif failed:
                differences.append({"path": "result", "recorded": "a result",
                                    "replayed": f"error {got['error']['code']}: {got['error']['message']}",
                                    "relative_difference": None})
            else:
                _differences(entry["result"], got, float(rel_tol), float(abs_tol), "result",
                             _VOLATILE_FIELDS.get(str(name), ()), differences)
            if differences:
                out.drifted.append({"seq": seq, "tool": name, "differences": differences})
            else:
                out.matched += 1
    finally:
        _replaying.on = False
    return out


def _compact(value: Any, limit: int = 1200) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=False)
    return text if len(text) <= limit else text[:limit] + f" … ({len(text):,} characters; the ledger has the rest)"


def report(source: Source, title: str = "RiskPY calculation record", check: bool = False) -> str:
    """A Markdown calculation record of a ledger: what was computed, with what, and the digest that pins it.

    source: a ledger file path, a :class:`Ledger`, or a list of entries
    title: the document's heading
    check: True to replay the ledger as well and include the outcome

    Returns the Markdown text: the head digest and chain status, the versions and period
    covered, the conventions the numbers follow, a table of the calls, then every call's
    full arguments and result (results longer than about 1,200 characters are cut, with a
    pointer to the ledger).

    Example::

        with open("record.md", "w", encoding="utf-8") as fh:
            fh.write(audit.report("run.jsonl", title="Q4 capital — calculation record", check=True))
    """
    from . import __version__

    if not isinstance(title, str) or not title.strip():
        raise InputError("title must be a non-empty string", code="RP111", param="title")
    entries = load(source)
    chain = verify_chain(entries)
    versions = sorted({str(e.get("riskpy")) for e in entries if e.get("riskpy")})
    notes = list(dict.fromkeys(str(e["note"]) for e in entries if e.get("note")))
    failed = sum("error" in e for e in entries)
    lines = [f"# {title.strip()}", ""]
    lines += [
        f"- **Ledger head (SHA-256):** `{chain.head}`",
        f"- **Chain:** {'intact' if chain.ok else f'**BROKEN** at entry {chain.broken_at} — {chain.reason}'}",
        f"- **Calls:** {len(entries)} ({len(entries) - failed} succeeded, {failed} refused)",
        f"- **Recorded with:** RiskPY {', '.join(versions) or 'unknown'}",
    ]
    if entries:
        lines.append(f"- **Period:** {entries[0].get('at', '?')} to {entries[-1].get('at', '?')} (UTC)")
    if notes:
        lines.append(f"- **Note:** {'; '.join(notes)}")
    lines.append(f"- **Report generated with:** RiskPY {__version__}")
    if check:
        outcome = replay(entries)
        lines += ["", "## Reproducibility", "",
                  f"Replayed with RiskPY {outcome.version}: **{outcome.matched} matched, {len(outcome.drifted)} "
                  f"drifted, {len(outcome.skipped)} skipped** (relative tolerance {outcome.rel_tol:g})."]
        for item in outcome.drifted:
            for d in item["differences"][:5]:
                lines.append(f"- drift, call {item['seq']} `{item['tool']}`: `{d['path']}` recorded "
                             f"`{d['recorded']!r}`, now `{d['replayed']!r}`")
        for item in outcome.skipped:
            lines.append(f"- skipped, call {item['seq']} `{item['tool']}`: {item['reason']}")
    lines += [
        "", "## Conventions", "",
        "Rates, yields, volatilities, probabilities and confidence levels are decimals (0.05 is 5%). Times are in "
        "years. `r` is continuously compounded, `i` is an effective annual rate, a bond's `ytm` is compounded "
        "`frequency` times a year. Losses are positive numbers. Every tool wraps a function that `riskpy-verify` "
        "checks against published tables and closed forms.",
        "", "## Calls", "",
        "| # | time (UTC) | tool | outcome | ms |", "|---|---|---|---|---|",
    ]
    for position, e in enumerate(entries, 1):
        outcome_text = f"refused: {e['error'].get('code')}" if isinstance(e.get("error"), dict) else "ok"
        lines.append(f"| {e.get('seq', position)} | {str(e.get('at', ''))[:23]} | `{e.get('tool')}` | {outcome_text} "
                     f"| {e.get('ms', '')} |")
    for position, e in enumerate(entries, 1):
        lines += ["", f"### {e.get('seq', position)}. `{e.get('tool')}`", "",
                  f"Arguments: `{_compact(e.get('arguments'))}`", ""]
        if isinstance(e.get("error"), dict):
            err = e["error"]
            lines.append(f"Refused — **{err.get('code')}**"
                         + (f" (`{err.get('param')}`)" if err.get("param") else "") + f": {err.get('message')}")
        else:
            lines.append(f"Result: `{_compact(e.get('result'))}`")
        lines += ["", f"Digest: `{e.get('digest')}`"]
    lines += [
        "", "## Checking this record", "",
        "```bash",
        "riskpy-audit verify LEDGER.jsonl     # the chain is intact and ends at the head above",
        "riskpy-audit replay LEDGER.jsonl     # every call recomputes to the recorded result",
        "```",
        "",
        "The head digest changes if any argument, result, or the order of the calls is altered. Keep it with the "
        "figures it supports.",
    ]
    return "\n".join(lines) + "\n"


def status(check: bool = False) -> Dict[str, Any]:
    """Whether calls are being recorded, where, and the head digest to quote.

    check: True to also replay the ledger and include the outcome

    Returns ``{"recording", "file", "entries", "head", "chain_ok", "write_failures",
    "how"}`` (plus ``"replay"`` when ``check`` is true). This is what the ``riskpy_audit``
    tool returns to an agent.

    Example::

        audit.status()["head"]     # quote this next to the numbers
    """
    ledger = active()
    if ledger is None:
        return {"recording": False, "file": None, "entries": 0, "head": None, "chain_ok": None, "write_failures": 0,
                "how": f"Set {ENV}=/path/ledger.jsonl before starting the process (or run `riskpy-mcp --audit "
                       "FILE`), or call riskpy.audit.start(path) in Python. Every calculator call is then recorded "
                       "in full and hash-chained."}
    try:
        # A file ledger that has not been written yet is an empty ledger, not a damaged one.
        entries = [] if ledger.path is not None and not os.path.exists(ledger.path) else load(ledger)
        chain = verify_chain(entries)
    except InputError as exc:
        return {"recording": True, "file": ledger.path, "entries": None, "head": None, "chain_ok": False,
                "write_failures": ledger.failures, "how": f"The ledger cannot be read: {exc}"}
    out: Dict[str, Any] = {
        "recording": True, "file": ledger.path, "entries": chain.entries, "head": chain.head, "chain_ok": chain.ok,
        "write_failures": ledger.failures,
        "how": "Quote `head` with your numbers: it pins every recorded call. `riskpy-audit replay FILE` recomputes "
               "them; `riskpy-audit report FILE` writes the calculation record.",
    }
    if not chain.ok:
        out["broken_at"], out["reason"] = chain.broken_at, chain.reason
    if ledger.failures:
        out["how"] = (f"{ledger.failures} call(s) could not be written to {ledger.path!r} and are missing from the "
                      "ledger: check that the folder exists and is writable. " + out["how"])
    if check:
        out["replay"] = replay(entries).to_dict()
    return out


# ---------------------------------------------------------------------------
# CLI


def main(argv: Optional[List[str]] = None) -> int:
    from . import errors as _errors

    _errors._safe_streams()
    parser = argparse.ArgumentParser(
        prog="riskpy-audit",
        description="Check, replay and report a RiskPY audit ledger (written when RISKPY_AUDIT_FILE is set).")
    sub = parser.add_subparsers(dest="command", required=True)
    p_verify = sub.add_parser("verify", help="check the hash chain: exit 1 if any entry was altered")
    p_replay = sub.add_parser("replay", help="recompute every call and compare: exit 1 on drift or a broken chain")
    p_replay.add_argument("--rel-tol", type=float, default=1e-9, help="relative tolerance (default 1e-9)")
    p_report = sub.add_parser("report", help="write a Markdown calculation record")
    p_report.add_argument("-o", "--output", metavar="FILE", help="write to FILE instead of stdout")
    p_report.add_argument("--title", default="RiskPY calculation record")
    p_report.add_argument("--check", action="store_true", help="replay the ledger and include the outcome")
    for p in (p_verify, p_replay, p_report):
        p.add_argument("ledger", help="the JSON-lines ledger file")
    for p in (p_verify, p_replay):
        p.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args(argv)

    try:
        if args.command == "verify":
            chain = verify_chain(args.ledger)
            if args.json:
                print(json.dumps(chain.to_dict(), indent=2))
            elif chain.ok:
                print(f"intact: {chain.entries} entries, head {chain.head}")
            else:
                print(f"BROKEN at entry {chain.broken_at}: {chain.reason}")
            return 0 if chain.ok else 1
        if args.command == "replay":
            outcome = replay(args.ledger, rel_tol=args.rel_tol)
            print(json.dumps(outcome.to_dict(), indent=2, ensure_ascii=False) if args.json else outcome.summary())
            return 0 if outcome.ok else 1
        text = report(args.ledger, title=args.title, check=args.check)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"wrote {args.output}")
        else:
            sys.stdout.write(text)
        return 0
    except InputError as exc:
        print(f"riskpy-audit: {exc.code} {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
