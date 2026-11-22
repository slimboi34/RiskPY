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
    audit.verify_chain("q4-capital.jsonl", expected_head=ledger.head)   # ... or cut off, or added to?
    audit.replay("q4-capital.jsonl")           # recompute every call and compare
    print(audit.report("q4-capital.jsonl"))    # a Markdown calculation record

    RISKPY_AUDIT_FILE=run.jsonl riskpy-mcp     # the same ledger for an agent's MCP session
    riskpy-audit verify run.jsonl
    riskpy-audit verify --head 9c1f0b52… run.jsonl
    riskpy-audit replay run.jsonl
    riskpy-audit report run.jsonl -o record.md

Each ledger line is one JSON object: ``{seq, at, tool, arguments, result | error, ms, via,
riskpy, python, note, prev, digest}``. ``digest`` is the SHA-256 of the entry's canonical
JSON (keys sorted, no whitespace, ASCII) without the digest itself, and ``prev`` is the
digest of the entry before it, so the entries form a hash chain: changing, removing,
inserting or reordering any of them breaks every digest after it. The last digest — the
*head* — is one short string that identifies the whole run; quote it next to the numbers.

The chain shows that a ledger is internally consistent. Entries cut off the end, or new
ones appended after it, leave a valid chain with a different head, and the whole file could
be regenerated: catching those needs the head to be kept somewhere else (the report, an
email, a commit) and passed back to :func:`verify_chain`. The ledger assumes one writer at
a time.

Arguments are stored exactly: NaN and infinities, bytes and NumPy arrays and scalars are
written as small tagged objects (``{"$riskpy": "float", "value": "nan"}``) that
:func:`replay` turns back into the original values.

Introspection tools (``riskpy_status`` and friends) are not recorded. A simulation called
with ``seed=0`` draws a fresh seed, so it is recorded but cannot be replayed; pass a seed
for any number that will be quoted.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import contextvars
import dataclasses
import datetime
import hashlib
import inspect
import json
import math
import os
import platform
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from .errors import InputError, _quote_path

__all__ = ["Ledger", "ChainReport", "ReplayReport", "ENV", "GENESIS", "start", "stop", "active", "record", "load",
           "digest", "verify_chain", "replay", "report", "status", "main"]

ENV = "RISKPY_AUDIT_FILE"
GENESIS = "0" * 64          # the "previous digest" of the first entry

# Result fields that legitimately differ between two runs of the same call.
_VOLATILE_FIELDS: Dict[str, Tuple[str, ...]] = {"verify": ("seconds", "version")}

_lock = threading.RLock()
# Where a call is recorded, in order (see active()):
# 1. the innermost record() block of the thread, or asyncio task, that makes the call: a contextvars
#    variable, so two threads each inside their own block keep their own calls, and leaving a block
#    restores what that thread had before, whatever other threads are doing;
# 2. for a thread with no block of its own (a worker started inside a block), the most recently
#    entered record() block that is still open anywhere in the process, as in 0.8.x;
# 3. the ledger start() set, process-wide;
# 4. RISKPY_AUDIT_FILE.
_active: Optional["Ledger"] = None
_scoped: "contextvars.ContextVar[Optional[Ledger]]" = contextvars.ContextVar("riskpy_audit_ledger", default=None)
_open_blocks: List["Ledger"] = []          # every record() block now open, oldest first (under _lock)
_env_ledgers: Dict[str, "Ledger"] = {}
_replaying = threading.local()

_HEX64 = re.compile(r"[0-9a-f]{64}")
_TOOL_NAME = re.compile(r"[A-Za-z0-9_]{1,64}")
_MESSAGE_LIMIT = 1000       # characters of an error message kept in an entry
_PARAM_LIMIT = 80           # characters of an error's parameter name kept in an entry
_BLOCK = 1 << 16            # bytes read at a time when looking for the last entry
_MAX_SEQ = 2 ** 53          # a "seq" at or above this is not a ledger's


# ---------------------------------------------------------------------------
# Storing arguments exactly
#
# The ledger is JSON, which has no NaN, infinity, bytes or arrays. Plain JSON values are
# stored as they are, so an ordinary entry reads exactly as it did in 0.7.0; anything else
# becomes a small object tagged with "$riskpy" that replay turns back into the original.

_TAG = "$riskpy"
_MAX_DEPTH = 256            # deeper arguments are stored as an "unserialisable" placeholder
_BIG_INT_BITS = 2000        # wider integers are stored in hex (JSON readers refuse > 4,300 digits)
_ARRAY_KINDS = "biufUSO"    # NumPy dtype kinds that round-trip: bool, ints, floats, strings, bytes, objects


class _Unstorable(Exception):
    """A value that cannot be written to the ledger at all (it is nested too deeply)."""


class _CannotRebuild(Exception):
    """A stored value that replay cannot turn back into the original call's argument."""


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception:  # noqa: BLE001 - a broken __repr__ must not stop the recording
        return f"<{type(value).__name__} object>"


def _fingerprint(value: Any) -> Optional[str]:
    """The SHA-256 of a value's repr, so a placeholder can still be matched to the original."""
    try:
        return hashlib.sha256(repr(value).encode("utf-8", "backslashreplace")).hexdigest()
    except Exception:  # noqa: BLE001 - RecursionError, a broken __repr__
        return None


def _encode(value: Any, depth: int = 0) -> Any:
    """A JSON value that :func:`_decode` turns back into ``value`` (tuples come back as lists)."""
    if depth > _MAX_DEPTH:
        raise _Unstorable(f"nested more than {_MAX_DEPTH} levels deep")
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return {_TAG: "int", "hex": hex(value)} if value.bit_length() > _BIG_INT_BITS else int(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        return {_TAG: "float", "value": "nan" if value != value else ("inf" if value > 0 else "-inf")}
    if isinstance(value, (bytes, bytearray)):
        return {_TAG: "bytes", "base64": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            items = {k: _encode(v, depth + 1) for k, v in value.items()}
            return {_TAG: "object", "value": items} if _TAG in value else items
        return {_TAG: "dict", "items": [[_encode(k, depth + 1), _encode(v, depth + 1)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return [_encode(v, depth + 1) for v in value]
    if isinstance(value, (set, frozenset)):
        return {_TAG: "frozenset" if isinstance(value, frozenset) else "set",
                "items": [_encode(v, depth + 1) for v in value]}
    np = sys.modules.get("numpy")
    if np is not None:
        if isinstance(value, np.ndarray) and value.dtype.kind in _ARRAY_KINDS and value.dtype.names is None:
            return {_TAG: "ndarray", "dtype": str(value.dtype), "shape": list(value.shape),
                    "data": [_encode(v, depth + 1) for v in value.ravel().tolist()]}
        if isinstance(value, np.generic) and value.dtype.kind in _ARRAY_KINDS and value.dtype.kind != "O":
            return {_TAG: "numpy", "dtype": str(value.dtype), "value": _encode(value.item(), depth + 1)}
    return {_TAG: "unrepresentable", "type": type(value).__name__, "repr": _clip(_safe_repr(value), 200)}


def _placeholder(value: Any, why: str) -> dict:
    return {_TAG: "unserialisable", "type": type(value).__name__, "reason": why, "sha256": _fingerprint(value)}


def _stored(value: Any) -> Any:
    """``value`` as it goes into an entry; a placeholder when it cannot be written at all."""
    try:
        return _encode(value)
    except _Unstorable as exc:
        return _placeholder(value, str(exc))
    except RecursionError:
        return _placeholder(value, "nested too deeply")


def _decode(value: Any, depth: int = 0) -> Any:
    """The original value of something :func:`_encode` stored."""
    if depth > _MAX_DEPTH + 2:
        raise _CannotRebuild("an argument is nested too deeply to rebuild")
    if isinstance(value, list):
        return [_decode(v, depth + 1) for v in value]
    if not isinstance(value, dict):
        return value
    if _TAG not in value:
        return {k: _decode(v, depth + 1) for k, v in value.items()}
    tag = value[_TAG]
    if tag == "unserialisable":
        raise _CannotRebuild(f"the call's arguments could not be serialised ({value.get('reason')})")
    if tag == "unrepresentable":
        raise _CannotRebuild(f"an argument of type {value.get('type')} could not be stored exactly")
    try:
        if tag == "float" and value["value"] in ("nan", "inf", "-inf"):
            return float(value["value"])
        if tag == "int" and isinstance(value["hex"], str):
            return int(value["hex"], 16)
        if tag == "bytes" and isinstance(value["base64"], str):
            return base64.b64decode(value["base64"], validate=True)
        if tag == "object" and isinstance(value["value"], dict):
            return {k: _decode(v, depth + 1) for k, v in value["value"].items()}
        if tag == "dict" and isinstance(value["items"], list):
            return {_decode(k, depth + 1): _decode(v, depth + 1) for k, v in value["items"]}
        if tag in ("set", "frozenset") and isinstance(value["items"], list):
            items = [_decode(v, depth + 1) for v in value["items"]]
            return frozenset(items) if tag == "frozenset" else set(items)
        if tag in ("ndarray", "numpy"):
            try:
                import numpy as np
            except ImportError:
                raise _CannotRebuild("an argument is a NumPy value and NumPy is not installed") from None
            dtype = np.dtype(value["dtype"])
            if dtype.kind not in _ARRAY_KINDS or dtype.names is not None:
                raise ValueError(dtype)
            if tag == "numpy":
                return dtype.type(_decode(value["value"], depth + 1))
            data = [_decode(v, depth + 1) for v in value["data"]]
            shape = tuple(int(n) for n in value["shape"])
            if dtype.kind == "O":
                array = np.empty(len(data), dtype=object)
                for k, item in enumerate(data):
                    array[k] = item
            else:
                array = np.array(data, dtype=dtype)
            return array.reshape(shape)
    except _CannotRebuild:
        raise
    except Exception:  # noqa: BLE001 - a crafted or damaged tag
        pass
    raise _CannotRebuild(f"an argument holds a malformed {_TAG!r} value")


# ---------------------------------------------------------------------------
# Hashing and reading ledger lines


def _canonical(entry: Dict[str, Any]) -> str:
    body = {k: v for k, v in entry.items() if k != "digest"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(entry: Dict[str, Any]) -> str:
    """The SHA-256 hex digest of a ledger entry, ignoring any ``digest`` key it already has.

    entry: a ledger entry (a dict of JSON values)

    The entry is serialised canonically — keys sorted, no whitespace, non-ASCII escaped —
    so the same entry gives the same digest on any machine and after any JSON round trip.

    Example::

        audit.digest({"seq": 1, "tool": "expected_loss"})     # 'a3c1…' (64 hex characters)
    """
    try:
        text = _canonical(entry)
    except (TypeError, ValueError) as exc:
        raise InputError(f"a ledger entry must hold only finite JSON values: {exc}", code="RP100",
                         param="entry") from None
    except RecursionError:
        raise InputError("a ledger entry is nested too deeply to hash", code="RP100", param="entry") from None
    return hashlib.sha256(text.encode("ascii")).hexdigest()


class _DuplicateKey(ValueError):
    pass


def _unique_keys(pairs: List[Tuple[str, Any]]) -> dict:
    out = dict(pairs)
    if len(out) != len(pairs):
        seen = set()
        for key, _ in pairs:
            if key in seen:
                raise _DuplicateKey(key)
            seen.add(key)
    return out


def _parse_line(raw: bytes) -> Union[dict, str]:
    """A ledger line as an entry, or what is wrong with it ("is not valid JSON")."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "is not UTF-8 text"
    try:
        entry = json.loads(text, object_pairs_hook=_unique_keys)
    except _DuplicateKey as exc:
        return f"repeats the key {_clip(str(exc.args[0]), 40)!r}, so readers disagree on what it says"
    except RecursionError:
        return "is nested too deeply to read"
    except ValueError:
        return "is not valid JSON"
    if not isinstance(entry, dict):
        return "is not a JSON object"
    return entry


class _Damaged:
    """A ledger line that is not an entry: torn by a crash, or not written by a ledger at all."""

    __slots__ = ("line", "problem")

    def __init__(self, line: int, problem: str):
        self.line, self.problem = line, problem

    @property
    def reason(self) -> str:
        return f"damaged line: line {self.line} {self.problem}"


Item = Union[dict, _Damaged]


def _chain_link(raw: bytes) -> Optional[Tuple[int, str]]:
    """(seq, digest) of a line that is a ledger entry, None for a damaged or foreign line."""
    entry = _parse_line(raw)
    if not isinstance(entry, dict):
        return None
    seq, link = entry.get("seq"), entry.get("digest")
    if (isinstance(seq, int) and not isinstance(seq, bool) and 0 <= seq < _MAX_SEQ
            and isinstance(link, str) and _HEX64.fullmatch(link)):
        return seq, link
    return None


def _lines_backwards(fh: Any, end: int) -> Iterator[bytes]:
    """The non-blank lines of a binary file, last first, reading each byte once."""
    pending: List[bytes] = []        # pieces of the current line, last piece first
    position = end
    while position > 0:
        step = min(_BLOCK, position)
        position -= step
        fh.seek(position)
        block = fh.read(step)
        cut = len(block)
        while True:
            newline = block.rfind(b"\n", 0, cut)
            if newline < 0:
                pending.append(block[:cut])
                break
            pending.append(block[newline + 1:cut])
            line = b"".join(reversed(pending))
            pending = []
            if line.strip():
                yield line
            cut = newline
    line = b"".join(reversed(pending))
    if line.strip():
        yield line


def _scan_tail(path: str) -> Tuple[int, str, bool]:
    """(seq, digest to chain from, whether the file ends with a newline) for continuing a ledger.

    The sequence number comes from the last line that is a ledger entry. When the last line
    is damaged (a torn write) or foreign, the next entry chains from that line's SHA-256, so
    verify_chain reports the damage instead of the chain quietly skipping over it.
    """
    try:
        fh = open(path, "rb")
    except FileNotFoundError:
        return 0, GENESIS, True
    with fh:
        size = fh.seek(0, os.SEEK_END)
        if size == 0:
            return 0, GENESIS, True
        fh.seek(size - 1)
        ends_with_newline = fh.read(1) == b"\n"
        prev: Optional[str] = None
        for line in _lines_backwards(fh, size):
            link = _chain_link(line)
            if link is not None:
                return link[0], (link[1] if prev is None else prev), ends_with_newline
            if prev is None:
                prev = hashlib.sha256(line).hexdigest()
        return 0, (prev or GENESIS), ends_with_newline


def _file_key(st: os.stat_result) -> Tuple[int, int, int, int]:
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


class Ledger:
    """An append-only, hash-chained record of tool calls.

    path: a JSON-lines file to append to (created if missing; an existing ledger is
        continued from its last entry), or None to keep the entries in memory only
    note: a free-text label written into every entry — the run, the report, the author

    Attributes: ``path``, ``note``, ``entries`` (the entries this object has written, oldest
    first), ``failures`` (calls that could not be written to the file) and
    ``failure_reasons`` (``{reason: count}`` for those). Usually made by :func:`start` or
    :func:`record` rather than directly.

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
        self.failure_reasons: Dict[str, int] = {}
        # (file identity, seq, digest, ends with newline) as of the last read or write, so an
        # append does not re-read a file nobody else has touched.
        self._tail_cache: Optional[Tuple[Tuple[int, int, int, int], int, str, bool]] = None

    def _tail_info(self) -> Tuple[int, str, bool]:
        """(sequence number, digest, file ends with a newline) of the last entry."""
        if self.path is None:
            if self.entries:
                return self.entries[-1]["seq"], self.entries[-1]["digest"], True
            return 0, GENESIS, True
        try:
            key = _file_key(os.stat(self.path))
        except OSError:
            self._tail_cache = None
            return 0, GENESIS, True
        cached = self._tail_cache
        if cached is not None and cached[0] == key:
            return cached[1], cached[2], cached[3]
        seq, prev, ends_with_newline = _scan_tail(self.path)
        self._tail_cache = (key, seq, prev, ends_with_newline)
        return seq, prev, ends_with_newline

    def _tail(self) -> Tuple[int, str]:
        """(sequence number, digest) of the last entry: from the file when there is one, so
        that several processes appending one after another keep a single chain."""
        seq, prev, _ = self._tail_info()
        return seq, prev

    @property
    def head(self) -> str:
        """The digest of the last entry (64 zeros for an empty ledger): one string that pins the whole run."""
        with _lock:
            return self._tail()[1]

    def append(self, tool: str, arguments: Any, result: Any = None, error: Optional[dict] = None,
               ms: float = 0.0, via: str = "python") -> dict:
        """Add one call to the ledger and return the entry written.

        tool: the tool's name
        arguments: the arguments exactly as sent
        result: the JSON result of a successful call
        error: ``{code, message, param}`` for a failed call, in place of ``result``
        ms: how long the call took, in milliseconds
        via: "python", "cli" or "mcp"

        :func:`riskpy.tools.call` does this for every calculator call while a ledger is
        active; call it yourself only to record a calculation made outside the tool layer.
        Arguments are stored exactly (NaN, infinities, bytes and NumPy values as tagged
        objects); arguments nested too deeply to store are replaced by a placeholder holding
        the SHA-256 of their repr, so the call is still in the chain. An error message is
        kept to 1,000 characters and its parameter to 80. Raises ``OSError`` if the file
        cannot be written.

        Example::

            ledger.append("my_model", {"scenario": "base"}, result={"reserve": 1.25e6})
        """
        from . import __version__
        from .tools import _jsonable

        if error is not None and not isinstance(error, dict):
            raise InputError(f"error must be a dict {{code, message, param}}, got {error!r:.80}", code="RP100",
                             param="error")
        entry: Dict[str, Any] = {
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
            "tool": str(tool),
            "arguments": _stored(arguments),
            "ms": round(float(ms), 3) if math.isfinite(float(ms)) else None,
            "via": str(via),
            "riskpy": __version__,
            "python": platform.python_version(),
        }
        if self.note:
            entry["note"] = self.note
        if error is not None:
            entry["error"] = {"code": _error_field(error.get("code"), 40),
                              "message": _error_field(error.get("message"), _MESSAGE_LIMIT),
                              "param": _error_field(error.get("param"), _PARAM_LIMIT)}
        else:
            try:
                entry["result"] = _stored(_jsonable(result))
            except RecursionError:
                entry["result"] = _placeholder(result, "nested too deeply")
        with _lock:
            if self.path is None:
                seq, prev, _ = self._tail_info()
                entry = {"seq": seq + 1, **entry, "prev": prev}
                entry["digest"] = digest(entry)
                self.entries.append(entry)
                return entry
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
            fd = os.open(self.path, flags, 0o666)
            try:
                seq, prev, ends_with_newline = self._tail_info()
                entry = {"seq": seq + 1, **entry, "prev": prev}
                entry["digest"] = digest(entry)
                line = (json.dumps(entry, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
                # A torn last line (a crash or a full disk mid-write) gets its newline back first, so this
                # entry stays a line of its own. Newline and entry go in one O_APPEND write (the loop only
                # continues a short write), so the line lands whole at the end of the file.
                _write_all(fd, line if ends_with_newline else b"\n" + line)
                self._tail_cache = (_file_key(os.fstat(fd)), entry["seq"], entry["digest"], True)
            finally:
                os.close(fd)
            self.entries.append(entry)
        return entry

    def _failed(self, reason: str) -> None:
        with _lock:
            self.failures += 1
            self.failure_reasons[reason] = self.failure_reasons.get(reason, 0) + 1

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        where = self.path or "memory"
        return f"<Ledger {where} entries={len(self.entries)} head={self.head[:12]}>"


def _error_field(value: Any, limit: int) -> Any:
    """One field of a recorded error: a short JSON scalar, or text cut to ``limit`` characters."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and value.bit_length() <= 64:
        return int(value)
    if isinstance(value, float) and math.isfinite(value):
        return float(value)
    return _clip(value if isinstance(value, str) else _safe_repr(value)[:limit + 1], limit)


def start(path: Optional[str] = None, note: Optional[str] = None) -> Ledger:
    """Start recording every calculator call made through :func:`riskpy.tools.call` in this process.

    path: a JSON-lines file to append to, or None to record in memory
    note: a label written into every entry

    Returns the :class:`Ledger`. Recording continues until :func:`stop`; starting again
    replaces the active ledger. It is process-wide: calls from every thread go to it while no
    :func:`record` block is open (an open block takes precedence; see :func:`record`).
    Setting the environment variable ``RISKPY_AUDIT_FILE`` does the same for a whole process
    (the CLI, the MCP server) without any code.

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
    Nor is an open :func:`record` block: it records until it ends.
    """
    global _active
    with _lock:
        ledger, _active = _active, None
    return ledger


def active() -> Optional[Ledger]:
    """The ledger this thread's calls are recorded to: the innermost :func:`record` block open
    in this thread (or asyncio task); for a thread without one, the most recently entered
    block still open in another thread; else the one from :func:`start`, else the file named
    by ``RISKPY_AUDIT_FILE``, else None."""
    scoped = _scoped.get()
    if scoped is not None:
        return scoped
    with _lock:
        if _open_blocks:
            return _open_blocks[-1]
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

    Yields the :class:`Ledger`. The calls the entering thread (or asyncio task) makes inside
    the block go to this ledger, and when the block ends that thread's previous ledger (an
    enclosing block's, :func:`start`'s, or none) resumes, whatever other threads are doing:
    two threads can each record into their own block at the same time. A thread with no
    block of its own, such as a worker started inside the block, records to the most
    recently entered block still open in the process. When blocks are open in several
    threads at once, a worker that must record to one particular block should run in a copy
    of that block's context (``contextvars.copy_context().run``, or ``asyncio.to_thread``,
    which copies it).

    Example::

        with audit.record(note="IRB check") as ledger:
            tools.call("basel_irb_capital", {"pd": 0.01, "lgd": 0.45, "ead": 1_000_000})
        ledger.entries[0]["result"]["risk_weight"]     # 0.9232…
    """
    ledger = Ledger(path, note)
    previous = _scoped.get()
    token = _scoped.set(ledger)
    with _lock:
        _open_blocks.append(ledger)
    try:
        yield ledger
    finally:
        with _lock:
            # This block's own ledger, wherever it is in the list: blocks in different threads need
            # not end in the order they began.
            for k in range(len(_open_blocks) - 1, -1, -1):
                if _open_blocks[k] is ledger:
                    del _open_blocks[k]
                    break
        try:
            _scoped.reset(token)
        except ValueError:          # left in a different context than it was entered in
            _scoped.set(previous)


def _record(name: Any, arguments: Any, result: Any, error: Optional[dict], ms: float, via: str) -> None:
    """Called by tools.call after every calculator call. Never raises: an audit trail that
    cannot be written is counted (``Ledger.failures``), not allowed to break the calculation."""
    if getattr(_replaying, "on", False):
        return
    ledger = active()
    if ledger is None:
        return
    # A name that is not a tool's is kept as its repr, so newlines or Markdown in it stay inert.
    if isinstance(name, str) and _TOOL_NAME.fullmatch(name):
        tool = name
    else:
        tool = _safe_repr(name[:200] if isinstance(name, str) else name)[:80]
    try:
        ledger.append(tool, arguments, result, error, ms, via)
    except OSError as exc:
        ledger._failed(f"cannot write the file: {exc.strerror or type(exc).__name__}")
    except Exception as exc:  # noqa: BLE001
        ledger._failed(_clip(f"{type(exc).__name__}: {exc}", 200))


# ---------------------------------------------------------------------------
# Reading, checking and replaying a ledger


Source = Union[str, "os.PathLike[str]", Ledger, Iterable[dict]]


def _read(source: Source) -> List[Item]:
    """The entries of a ledger, with a :class:`_Damaged` in place of each line that is not one."""
    if isinstance(source, Ledger):
        if source.path is None:
            return [dict(e) for e in source.entries]
        if not os.path.exists(source.path):
            # Nothing recorded yet (the file is created by the first append): an empty, intact
            # ledger, as ledger.head (64 zeros) and status() already say. A path given as a string
            # is still RP505 when missing: a typo must not pass for an empty ledger.
            return []
        source = source.path
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        items: List[Item] = []
        try:
            with open(path, "rb") as fh:
                for number, raw in enumerate(fh, 1):
                    if not raw.strip():
                        continue
                    entry = _parse_line(raw)
                    items.append(entry if isinstance(entry, dict) else _Damaged(number, entry))
        except OSError as exc:
            raise InputError(f"cannot read the ledger {_quote_path(path)}: {exc.strerror or exc}", code="RP505",
                             param="source") from None
        return items
    try:
        return [dict(e) for e in source]
    except (TypeError, ValueError):
        raise InputError("source must be a ledger file path, a Ledger or a list of entries", code="RP100",
                         param="source") from None


def load(source: Source) -> List[dict]:
    """The entries of a ledger as a list of dicts, oldest first.

    source: a ledger file path, a :class:`Ledger`, or an iterable of entries

    Raises RP505 if the file cannot be read and RP100 if a line is not a JSON object (not
    UTF-8, not JSON, nested too deeply, or repeating a key) — a ledger is evidence, so a
    damaged one is reported rather than skipped over. :func:`verify_chain`, :func:`replay`
    and :func:`report` read a damaged ledger anyway and say which line is damaged.

    Example::

        for entry in audit.load("run.jsonl"):
            print(entry["seq"], entry["tool"], entry.get("result"))
    """
    items = _read(source)
    for item in items:
        if isinstance(item, _Damaged):
            where = os.fspath(source.path if isinstance(source, Ledger) else source)  # type: ignore[arg-type]
            raise InputError(f"{where}: line {item.line} {item.problem}", code="RP100", param="source",
                             hint="A ledger is one JSON object per line; this one is damaged. "
                                  "riskpy-audit verify names the first damaged line.")
    return items  # type: ignore[return-value]


@dataclass(frozen=True)
class ChainReport:
    """The outcome of :func:`verify_chain`.

    Fields: ``ok``; ``entries`` (how many lines were checked); ``head`` (the last digest, or
    64 zeros for an empty ledger; on a broken chain, the last digest before the break);
    ``broken_at`` (the ``seq`` — or 1-based position — of the first bad entry, None when
    intact or when the break cannot be placed); ``reason`` (what is wrong).
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


def _verify(items: Sequence[Item]) -> Tuple[ChainReport, List[str]]:
    """The chain report and the digests of the entries before any break."""
    prev, digests = GENESIS, []
    for position, entry in enumerate(items, 1):
        if isinstance(entry, _Damaged):
            return ChainReport(False, len(items), prev, position, entry.reason), digests
        seq = entry.get("seq", position)
        # The position, never the entry's own claim: a foreign line saying "seq": 999 is still where it is.
        # (Once the sequence check below has passed, the two are equal anyway.)
        where = position
        if not isinstance(entry.get("digest"), str):
            return ChainReport(False, len(items), prev, where, "the entry has no digest"), digests
        if seq != position:
            return ChainReport(False, len(items), prev, position,
                               f"sequence number {_clip(_safe_repr(seq), 40)} where {position} was expected: an "
                               "entry was removed, inserted or reordered"), digests
        if entry.get("prev") != prev:
            return ChainReport(False, len(items), prev, where,
                               "the entry does not link to the one before it: an entry was removed, inserted or "
                               "reordered"), digests
        try:
            expected = hashlib.sha256(_canonical(entry).encode("ascii")).hexdigest()
        except RecursionError:
            return ChainReport(False, len(items), prev, where, "the entry is nested too deeply to hash"), digests
        except (TypeError, ValueError):
            return ChainReport(False, len(items), prev, where,
                               "the entry holds a value that is not finite JSON"), digests
        if expected != entry["digest"]:
            return ChainReport(False, len(items), prev, where,
                               "the digest does not match the entry: its content was changed after it was "
                               "recorded"), digests
        prev = entry["digest"]
        digests.append(prev)
    return ChainReport(True, len(items), prev), digests


def _expected(chain: ChainReport, digests: List[str], expected_head: Optional[str],
              expected_entries: Optional[int]) -> ChainReport:
    """An intact chain checked against the head and length quoted when the numbers were reported."""
    if not chain.ok:
        return chain
    def entries(count: int) -> str:
        return f"{count} entr{'y was' if count == 1 else 'ies were'}"

    n = chain.entries
    if expected_entries is not None and n != expected_entries:
        if n < expected_entries:
            return dataclasses.replace(chain, ok=False, broken_at=n + 1, reason=(
                f"the chain is intact but has {n} entries where {expected_entries} were expected: the last "
                f"{entries(expected_entries - n)} removed"))
        return dataclasses.replace(chain, ok=False, broken_at=expected_entries + 1, reason=(
            f"the chain is intact but has {n} entries where {expected_entries} were expected: "
            f"{entries(n - expected_entries)} added after the end"))
    if expected_head is not None and chain.head != expected_head:
        if expected_head in digests:
            k = digests.index(expected_head) + 1
            return dataclasses.replace(chain, ok=False, broken_at=k + 1, reason=(
                f"the chain continues past the expected head (entry {k}): {entries(n - k)} added after it"))
        return dataclasses.replace(chain, ok=False, broken_at=None, reason=(
            f"the chain is intact but ends at head {chain.head}, not the expected {expected_head}, and does not "
            "contain it: entries were removed from the end, or this is not the ledger the head was quoted from"))
    return chain


def verify_chain(source: Source, expected_head: Optional[str] = None,
                 expected_entries: Optional[int] = None) -> ChainReport:
    """Check that a ledger is intact: every digest matches its entry and links to the one before.

    source: a ledger file path, a :class:`Ledger`, or a list of entries
    expected_head: the head digest quoted when the numbers were reported (from the report,
        an email, a commit); the chain must end exactly there
    expected_entries: the number of entries the ledger should hold

    Returns a :class:`ChainReport`, truthy when the chain is intact. An edited argument or
    result, an entry deleted, inserted or moved inside the chain, or a damaged line (a torn
    write, a foreign line) breaks it, and ``broken_at`` names the first entry affected.
    Entries cut off the end, or new ones appended after it, leave a valid chain with a
    different head: only ``expected_head`` (or ``expected_entries``) catches those, which is
    why the head has to be kept outside the ledger.

    Whitespace, key order and string escapes on a line do not matter: the digest is taken over
    the canonical form of the parsed entry. Number spelling does: the canonical form writes
    ``9.0`` and ``9`` differently, as Python's ``json`` does, so a ledger re-serialised by a
    tool that keeps number spelling (Python's ``json``, jq 1.7 or later) still verifies, but
    one passed through a tool that rewrites ``9.0`` as ``9`` (JavaScript's ``JSON.parse`` /
    ``JSON.stringify``, Go's ``interface{}`` decoding, jq 1.6) does not. A line that repeats
    a key is damaged, because readers disagree on which value it holds.

    Example::

        check = audit.verify_chain("run.jsonl")
        check.ok, check.entries, check.head
        audit.verify_chain("run.jsonl", expected_head="9c1f0b52…").ok     # nothing cut off or added
    """
    if expected_head is not None:
        if not isinstance(expected_head, str) or not _HEX64.fullmatch(expected_head.strip().lower()):
            raise InputError(f"expected_head must be a SHA-256 digest (64 hex characters), got "
                             f"{_clip(_safe_repr(expected_head), 80)}", code="RP100", param="expected_head",
                             hint="Pass the head exactly as it was quoted, e.g. from the report or riskpy_audit.")
        expected_head = expected_head.strip().lower()
    if expected_entries is not None and (isinstance(expected_entries, bool) or not isinstance(expected_entries, int)
                                         or expected_entries < 0):
        raise InputError(f"expected_entries must be a whole number >= 0, got {expected_entries!r:.80}",
                         code="RP100", param="expected_entries")
    chain, digests = _verify(_read(source))
    return _expected(chain, digests, expected_head, expected_entries)


def _differences(recorded: Any, replayed: Any, rel_tol: float, abs_tol: float, path: str, skip: Tuple[str, ...],
                 out: List[dict]) -> None:
    def is_num(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    if is_num(recorded) and is_num(replayed):
        if isinstance(recorded, int) and isinstance(replayed, int):
            # Exactly: an int may be far beyond float range (10**400 echoed from a refused scenario).
            if recorded != replayed:
                try:
                    relative: Optional[float] = abs(recorded - replayed) / max(abs(recorded), abs(replayed))
                except OverflowError:
                    relative = None
                out.append({"path": path, "recorded": recorded, "replayed": replayed,
                            "relative_difference": relative})
            return
        try:
            gap = abs(float(recorded) - float(replayed))
            scale = max(abs(float(recorded)), abs(float(replayed)))
        except OverflowError:           # an int beyond float range against a float: they cannot be equal
            out.append({"path": path, "recorded": recorded, "replayed": replayed, "relative_difference": None})
            return
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
    """True when the call drew a fresh random seed somewhere, so recomputing it cannot give the same
    numbers: its own seed is 0, or (for sensitivity and scenario_table) the base arguments', a
    scenario's override's, or a bumped seed's. Since 0.9.0 the meta-tools refuse the last two, so
    this matters for ledgers recorded by earlier versions."""
    from .tools import TOOLS, _fresh_seed

    if not isinstance(arguments, dict):
        return False
    if isinstance(arguments.get("tool"), str) and isinstance(arguments.get("arguments"), dict):
        inner, base = arguments["tool"], arguments["arguments"]          # sensitivity / scenario_table
        if _unseeded(inner, base):
            return True
        if tool_name == "sensitivity" and arguments.get("parameter") == "seed":
            return True                       # seed ± bump: unrelated streams, and seed 0 among them
        scenarios = arguments.get("scenarios") if tool_name == "scenario_table" else None
        if isinstance(scenarios, dict):
            return any(isinstance(o, dict) and "seed" in o and _unseeded(inner, {**base, **o})
                       for o in scenarios.values())
        return False
    t = TOOLS.get(tool_name) if isinstance(tool_name, str) else None
    if t is None:
        return False
    parameter = inspect.signature(t.fn).parameters.get("seed")
    if parameter is None:
        return False
    return _fresh_seed(arguments.get("seed", parameter.default))


@dataclass
class ReplayReport:
    """The outcome of :func:`replay`.

    Fields: ``chain`` (the :class:`ChainReport` of the ledger); ``entries`` (how many it
    holds); ``matched`` (calls recomputed to the same result or the same error code);
    ``drifted`` (calls whose result differs — ``[{seq, tool, differences: [{path, recorded,
    replayed, relative_difference}]}]``); ``skipped`` (calls that cannot be recomputed —
    ``[{seq, tool, reason}]``); ``recorded_versions`` (the RiskPY versions in the ledger);
    ``version`` (the version that replayed it); ``rel_tol`` / ``abs_tol``; ``replayed``
    (how many entries were looked at: all of them, except in a capped
    :func:`status` check, which replays only the most recent).
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
    replayed: Optional[int] = None

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
                "version": self.version, "rel_tol": self.rel_tol, "abs_tol": self.abs_tol,
                "replayed": self.entries if self.replayed is None else self.replayed}

    def summary(self) -> str:
        """A few lines of text: the chain, the counts, and each drifted or skipped call."""
        partial = (f" — the most recent {self.replayed} of {self.entries} entries"
                   if self.replayed is not None and self.replayed < self.entries else "")
        lines = [
            f"chain     {'intact' if self.chain.ok else 'BROKEN at entry ' + str(self.chain.broken_at)}"
            f" ({self.chain.entries} entries, head {self.chain.head[:16]}…)",
            f"replayed  {self.matched} matched, {len(self.drifted)} drifted, {len(self.skipped)} skipped{partial} "
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
            lines.append(f"  skip   #{item['seq']} {item['tool'] if item['tool'] is not None else '-'}: "
                         f"{item['reason']}")
        return "\n".join(lines)


def _check_tolerances(rel_tol: Any, abs_tol: Any) -> None:
    for name, value in (("rel_tol", rel_tol), ("abs_tol", abs_tol)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value >= 0.0:
            raise InputError(f"{name} must be a number >= 0, got {value!r}", code="RP102", param=name)


def _replay_one(tools: Any, position: int, entry: Item, rel_tol: float, abs_tol: float) -> Tuple[str, dict]:
    """("matched" | "drifted" | "skipped", detail) for one ledger item."""
    if isinstance(entry, _Damaged):
        return "skipped", {"seq": position, "tool": None, "reason": entry.reason}
    seq, name, arguments = entry.get("seq", position), entry.get("tool"), entry.get("arguments")

    def skip(reason: str) -> Tuple[str, dict]:
        return "skipped", {"seq": seq, "tool": name, "reason": reason}

    if "result" not in entry and "error" not in entry:
        return skip("the entry has neither a result nor an error")
    if "error" in entry and not isinstance(entry["error"], dict):
        return skip("the entry's error is not an object {code, message, param}")
    error = entry.get("error")
    if (isinstance(error, dict) and error.get("code") == "RP405" and error.get("param") == "name"
            and not (isinstance(name, str) and name in tools.TOOLS)):
        # Refused for a tool name over LIMITS["string"] characters, which the ledger keeps only as a clipped
        # repr: calling that short repr would be an unknown name (RP401), not the same call.
        return skip("the call was refused for an over-long tool name, of which the ledger keeps only the start, "
                    "so it cannot be recomputed")
    t = tools.TOOLS.get(name) if isinstance(name, str) else None
    if t is not None and t.kind != "calculator":
        # Introspection tools are never recorded; replaying one (riskpy_audit with check=true) would start
        # a replay inside the replay.
        return skip(f"foreign entry: {name} is not a calculator, so a ledger never records it")
    if t is None and "error" not in entry:
        return skip("this installation has no such tool")
    if t is not None and not t.available:
        return skip(f"the tool needs {t.install}")
    # Ledgers from 0.7.0 hold no tagged values (they stored NaN as null), so this reads them unchanged.
    try:
        arguments = _decode(arguments)
    except _CannotRebuild as exc:
        return skip(f"{exc}, so the call cannot be recomputed")
    try:
        recorded = _decode(entry.get("result"))
    except _CannotRebuild:
        return skip("the recorded result could not be stored, so there is nothing to compare it with")
    if "error" not in entry and _unseeded(name, arguments):
        return skip("run with seed=0 (a fresh seed) in its arguments, a scenario or a bumped seed, so it cannot be "
                    "reproduced")
    got = tools.call(name, arguments, errors="return", via="replay")
    failed = isinstance(got, dict) and set(got) == {"error"}
    differences: List[dict] = []
    if "error" in entry:
        recorded_code = entry["error"].get("code")
        replayed_code = got["error"]["code"] if failed else None
        if recorded_code != replayed_code:
            differences.append({"path": "error.code", "recorded": recorded_code,
                                "replayed": replayed_code or "a result", "relative_difference": None})
    elif failed:
        differences.append({"path": "result", "recorded": "a result",
                            "replayed": f"error {got['error']['code']}: {got['error']['message']}",
                            "relative_difference": None})
    else:
        if name == "scenario_table":
            recorded, got = _scenario_errors_by_code(recorded), _scenario_errors_by_code(got)
        _differences(recorded, got, rel_tol, abs_tol, "result", _VOLATILE_FIELDS.get(str(name), ()), differences)
    if differences:
        return "drifted", {"seq": seq, "tool": name, "differences": differences}
    return "matched", {}


def _scenario_errors_by_code(result: Any) -> Any:
    """A scenario_table result with each refused scenario's error cut to its code and param.

    A refused call matches on replay when it fails again with the same code; a scenario refused
    inside a successful scenario_table is compared the same way. Its message and hint quote the
    value the agent sent, whose repr the ledger cannot always keep (a tuple is stored as a list:
    "got (1.0, 2.0)" would replay as "got [1.0, 2.0]"), and a scenario not run in time says how
    many seconds had passed."""
    if not isinstance(result, dict) or not isinstance(result.get("scenarios"), dict):
        return result
    scenarios = {}
    for label, row in result["scenarios"].items():
        if isinstance(row, dict) and isinstance(row.get("error"), dict):
            row = {**row, "error": {k: row["error"].get(k) for k in ("code", "param")}}
        scenarios[label] = row
    return {**result, "scenarios": scenarios}


def _replay_items(items: Sequence[Item], chain: ChainReport, rel_tol: float, abs_tol: float,
                  positions: Optional[Sequence[int]] = None, seconds: Optional[float] = None) -> ReplayReport:
    """Replay ``items`` (all of them, or those at ``positions``, in that order, stopping once
    ``seconds`` have passed) against an already-computed chain report."""
    from . import __version__
    from . import tools

    out = ReplayReport(chain=chain, entries=len(items), matched=0, version=__version__,
                       rel_tol=float(rel_tol), abs_tol=float(abs_tol),
                       recorded_versions=sorted({str(e.get("riskpy")) for e in items
                                                 if isinstance(e, dict) and e.get("riskpy")}))
    order = list(range(len(items))) if positions is None else list(positions)
    deadline = None if seconds is None else time.perf_counter() + seconds
    drifted: List[Tuple[int, dict]] = []
    skipped: List[Tuple[int, dict]] = []
    looked = 0
    previous = getattr(_replaying, "on", False)
    _replaying.on = True
    try:
        for index in order:
            if deadline is not None and looked and time.perf_counter() > deadline:
                break
            looked += 1
            entry = items[index]
            try:
                outcome, detail = _replay_one(tools, index + 1, entry, float(rel_tol), float(abs_tol))
            except RecursionError:          # a crafted entry nested just shallowly enough to parse
                assert isinstance(entry, dict)
                outcome, detail = "skipped", {"seq": entry.get("seq", index + 1), "tool": entry.get("tool"),
                                              "reason": "the entry is nested too deeply to replay"}
            except Exception as exc:  # noqa: BLE001 - one odd entry must not stop the replay of the rest
                where = entry if isinstance(entry, dict) else {}
                outcome, detail = "skipped", {"seq": where.get("seq", index + 1), "tool": where.get("tool"),
                                              "reason": _clip(f"the entry could not be replayed ({type(exc).__name__}: "
                                                              f"{exc})", 300)}
            if outcome == "matched":
                out.matched += 1
            elif outcome == "drifted":
                drifted.append((index, detail))
            else:
                skipped.append((index, detail))
    finally:
        _replaying.on = previous
    out.drifted = [d for _, d in sorted(drifted, key=lambda pair: pair[0])]
    out.skipped = [s for _, s in sorted(skipped, key=lambda pair: pair[0])]
    if positions is not None:
        out.replayed = looked
    return out


def replay(source: Source, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> ReplayReport:
    """Recompute every call in a ledger with the installed library and compare the results.

    source: a ledger file path, a :class:`Ledger`, or a list of entries
    rel_tol: relative tolerance on each number (default 1e-9; platforms differ in the last digits)
    abs_tol: absolute tolerance, for numbers near zero

    Returns a :class:`ReplayReport`, truthy when the chain is intact and nothing drifted. A
    recorded failure matches when the call fails again with the same error code. Calls that
    cannot be recomputed are listed in ``skipped`` with the reason: a simulation run with
    ``seed=0`` (a fresh seed each time), a tool this installation does not have, an argument
    that could not be stored exactly, a damaged line, or an entry for a tool that is never
    recorded (``riskpy_audit`` and the other introspection tools). Replayed calls are not
    themselves recorded.

    Use it after upgrading RiskPY to see which reported numbers the new version changes,
    and before signing off a report to confirm its numbers are reproducible.

    Example::

        outcome = audit.replay("run.jsonl")
        outcome.ok, outcome.matched, outcome.drifted
        print(outcome.summary())
    """
    _check_tolerances(rel_tol, abs_tol)
    items = _read(source)
    chain, _ = _verify(items)
    return _replay_items(items, chain, rel_tol, abs_tol)


# ---------------------------------------------------------------------------
# The report


_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f  ]")
_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>|])")


def _one_line(value: Any) -> str:
    """Text with every line break and control character shown as an escape (\\n, \\x1b)."""
    return _CONTROL.sub(lambda m: m.group().encode("unicode_escape").decode("ascii"), str(value))


def _md(value: Any) -> str:
    """Untrusted text as inert Markdown: one line, no HTML, no code spans, links or table cells."""
    return _MD_SPECIAL.sub(r"\\\1", _one_line(value))


def _code(value: Any, cell: bool = False) -> str:
    """Untrusted text as one inline code span, whatever backticks it holds."""
    text = _one_line(value)
    if cell and ("|" in text or "\\" in text):
        # GFM splits table cells on | even inside code spans, and reads \\| as a backslash then a split:
        # such a name goes in the cell as escaped plain text instead.
        return _md(text)
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if (not text or text.startswith("`") or text.endswith("`")) else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def _fenced(text: str) -> List[str]:
    """A fenced json block that nothing inside it can close."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return [fence + "json", text, fence]


def _compact(value: Any, limit: int = 1200) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=False, default=_safe_repr)
    except (RecursionError, ValueError):
        return "(nested too deeply to show; the ledger has it)"
    return text if len(text) <= limit else text[:limit] + f" … ({len(text):,} characters; the ledger has the rest)"


def _report_lines(items: Sequence[Item], chain: ChainReport, title: str, check: bool) -> List[str]:
    from . import __version__

    entries = [e for e in items if isinstance(e, dict)]
    versions = sorted({str(e.get("riskpy")) for e in entries if e.get("riskpy")})
    notes = list(dict.fromkeys(str(e["note"]) for e in entries if e.get("note")))
    failed = sum("error" in e for e in entries)
    damaged = len(items) - len(entries)
    lines = [f"# {' '.join(title.split())}", ""]
    lines += [
        f"- **Ledger head (SHA-256):** {_code(chain.head)}",
        f"- **Chain:** {'intact' if chain.ok else f'**BROKEN** at entry {chain.broken_at} — {_md(chain.reason)}'}",
        f"- **Calls:** {len(entries)} ({len(entries) - failed} succeeded, {failed} refused)"
        + (f"; {damaged} damaged line{'s' if damaged != 1 else ''}" if damaged else ""),
        f"- **Recorded with:** RiskPY {_md(', '.join(versions)) or 'unknown'}",
    ]
    if entries:
        lines.append(f"- **Period:** {_md(entries[0].get('at', '?'))} to {_md(entries[-1].get('at', '?'))} (UTC)")
    if notes:
        lines.append(f"- **Note:** {_md('; '.join(notes))}")
    lines.append(f"- **Report generated with:** RiskPY {__version__}")
    if check:
        outcome = _replay_items(items, chain, 1e-9, 1e-12)
        lines += ["", "## Reproducibility", "",
                  f"Replayed with RiskPY {outcome.version}: **{outcome.matched} matched, {len(outcome.drifted)} "
                  f"drifted, {len(outcome.skipped)} skipped** (relative tolerance {outcome.rel_tol:g})."]
        for item in outcome.drifted:
            for d in item["differences"][:5]:
                lines.append(f"- drift, call {_md(item['seq'])} {_code(item['tool'])}: {_code(d['path'])} recorded "
                             f"{_code(_safe_repr(d['recorded']))}, now {_code(_safe_repr(d['replayed']))}")
        for item in outcome.skipped:
            lines.append(f"- skipped, call {_md(item['seq'])} {_code(item['tool'])}: {_md(item['reason'])}")
    lines += [
        "", "## Conventions", "",
        "Rates, yields, volatilities, probabilities and confidence levels are decimals (0.05 is 5%). Times are in "
        "years. `r` is continuously compounded, `i` is an effective annual rate, a bond's `ytm` is compounded "
        "`frequency` times a year. Losses are positive numbers. Every tool wraps a function that `riskpy-verify` "
        "checks against published tables and closed forms.",
        "", "## Calls", "",
        "| # | time (UTC) | tool | outcome | ms |", "|---|---|---|---|---|",
    ]
    for position, e in enumerate(items, 1):
        if isinstance(e, _Damaged):
            lines.append(f"| {position} |  | *damaged line* | line {e.line} {_md(e.problem)} |  |")
            continue
        error = e.get("error")
        outcome_text = f"refused: {_md(error.get('code'))}" if isinstance(error, dict) else (
            "refused" if "error" in e else "ok")
        lines.append(f"| {_md(e.get('seq', position))} | {_md(str(e.get('at', ''))[:23])} | "
                     f"{_code(e.get('tool'), cell=True)} | {outcome_text} | {_md(e.get('ms', ''))} |")
    for position, e in enumerate(items, 1):
        if isinstance(e, _Damaged):
            lines += ["", f"### {position}. Damaged line", "",
                      f"Line {e.line} of the file {_md(e.problem)}. It is not an entry, so it is not shown; "
                      "the chain is broken here."]
            continue
        lines += ["", f"### {_md(e.get('seq', position))}. {_code(e.get('tool'))}", "", "Arguments:", ""]
        lines += _fenced(_compact(e.get("arguments")))
        lines.append("")
        if "error" in e:
            error = e["error"]
            if isinstance(error, dict):
                lines.append(f"Refused — **{_md(error.get('code'))}**"
                             + (f" ({_code(error.get('param'))})" if error.get("param") else "") + ":")
            else:
                lines.append("Refused:")
            lines.append("")
            lines += _fenced(_compact(error))
        else:
            lines += ["Result:", ""]
            lines += _fenced(_compact(e.get("result")))
        lines += ["", f"Digest: {_code(e.get('digest'))}"]
    lines += [
        "", "## Checking this record", "",
        "```bash",
        f"riskpy-audit verify --head {chain.head} LEDGER.jsonl",
        "riskpy-audit replay LEDGER.jsonl     # every call recomputes to the recorded result",
        "```",
        "",
        "The first command checks that the chain is intact and ends at the head above, so no entry was changed, "
        "removed or added. The head digest changes if any argument, result, or the order of the calls is altered. "
        "Keep it with the figures it supports.",
    ]
    return lines


def report(source: Source, title: str = "RiskPY calculation record", check: bool = False) -> str:
    """A Markdown calculation record of a ledger: what was computed, with what, and the digest that pins it.

    source: a ledger file path, a :class:`Ledger`, or a list of entries
    title: the document's heading
    check: True to replay the ledger as well and include the outcome

    Returns the Markdown text: the head digest and chain status, the versions and period
    covered, the conventions the numbers follow, a table of the calls, then every call's
    full arguments and result in fenced JSON blocks (results longer than about 1,200
    characters are cut, with a pointer to the ledger), and the ``riskpy-audit verify
    --head`` command that checks the ledger against this record. Text that came from the
    ledger (tool names, error messages, notes) is escaped, so a crafted call cannot add
    headings, links or HTML to the record.

    Example::

        with open("record.md", "w", encoding="utf-8") as fh:
            fh.write(audit.report("run.jsonl", title="Q4 capital — calculation record", check=True))
    """
    if not isinstance(title, str) or not title.strip():
        raise InputError("title must be a non-empty string", code="RP111", param="title")
    items = _read(source)
    chain, _ = _verify(items)
    text = "\n".join(_report_lines(items, chain, title, check)) + "\n"
    return text.encode("utf-8", "backslashreplace").decode("utf-8")      # lone surrogates cannot be written


# ---------------------------------------------------------------------------
# The status, for an agent

_STATUS_REPLAY_LAST = 50
_STATUS_REPLAY_SECONDS = 10.0


def _failure_text(ledger: Ledger) -> str:
    reasons = "; ".join(f"{reason} ×{count}" if count > 1 else reason
                        for reason, count in ledger.failure_reasons.items()) or "reason not recorded"
    advice = (" Check that the folder exists, is writable and has space."
              if any(r.startswith("cannot write the file") for r in ledger.failure_reasons) else "")
    return (f"{ledger.failures} call(s) could not be written to {ledger.path or 'the ledger'!r} and are missing from "
            f"the ledger ({reasons}).{advice} ")


def status(check: bool = False, replay_last: Optional[int] = _STATUS_REPLAY_LAST,
           replay_seconds: Optional[float] = _STATUS_REPLAY_SECONDS) -> Dict[str, Any]:
    """Whether calls are being recorded, where, and the head digest to quote.

    check: True to also replay the most recent entries and include the outcome
    replay_last: how many of the most recent entries ``check`` replays (default 50; None for all)
    replay_seconds: stop replaying after about this long (default 10 s; None for no limit)

    Returns ``{"recording", "file", "entries", "head", "chain_ok", "write_failures",
    "write_failure_reasons", "how"}`` (plus ``"broken_at"`` and ``"reason"`` for a broken
    chain, and ``"replay"`` when ``check`` is true, whose ``replayed`` says how many entries
    it covered). This is what the ``riskpy_audit`` tool returns to an agent, so one call
    does a bounded amount of work however long the ledger grows; ``riskpy-audit replay
    FILE`` replays every entry.

    Example::

        audit.status()["head"]     # quote this next to the numbers
    """
    if replay_last is not None and (isinstance(replay_last, bool) or not isinstance(replay_last, int)
                                    or replay_last < 1):
        raise InputError(f"replay_last must be a whole number >= 1 or None, got {replay_last!r:.80}", code="RP102",
                         param="replay_last")
    if replay_seconds is not None and (isinstance(replay_seconds, bool) or not isinstance(replay_seconds, (int, float))
                                       or not replay_seconds > 0):
        raise InputError(f"replay_seconds must be a number > 0 or None, got {replay_seconds!r:.80}", code="RP102",
                         param="replay_seconds")
    ledger = active()
    if ledger is None:
        return {"recording": False, "file": None, "entries": 0, "head": None, "chain_ok": None, "write_failures": 0,
                "write_failure_reasons": {},
                "how": f"Set {ENV}=/path/ledger.jsonl before starting the process (or run `riskpy-mcp --audit "
                       "FILE`), or call riskpy.audit.start(path) in Python. Every calculator call is then recorded "
                       "in full and hash-chained."}
    try:
        # A file ledger that has not been written yet is an empty ledger, not a damaged one.
        items = [] if ledger.path is not None and not os.path.exists(ledger.path) else _read(ledger)
        chain, _ = _verify(items)
    except Exception as exc:  # noqa: BLE001 - the status must answer, whatever the file holds
        code = getattr(exc, "code", "RP505")
        return {"recording": True, "file": ledger.path, "entries": None, "head": None, "chain_ok": False,
                "write_failures": ledger.failures, "write_failure_reasons": dict(ledger.failure_reasons),
                "how": f"The ledger cannot be read ({code}): {exc}"}
    out: Dict[str, Any] = {
        "recording": True, "file": ledger.path, "entries": chain.entries, "head": chain.head, "chain_ok": chain.ok,
        "write_failures": ledger.failures, "write_failure_reasons": dict(ledger.failure_reasons),
        "how": "Quote `head` with your numbers: it pins every recorded call. `riskpy-audit verify --head HEAD FILE` "
               "checks the ledger against it later; `riskpy-audit replay FILE` recomputes the calls; "
               "`riskpy-audit report FILE` writes the calculation record.",
    }
    if not chain.ok:
        out["broken_at"], out["reason"] = chain.broken_at, chain.reason
    if ledger.failures:
        out["how"] = _failure_text(ledger) + out["how"]
    if check:
        n = len(items)
        first = 0 if replay_last is None else max(0, n - replay_last)
        outcome = _replay_items(items, chain, 1e-9, 1e-12, positions=range(n - 1, first - 1, -1),
                                seconds=replay_seconds)
        out["replay"] = outcome.to_dict()
        if (outcome.replayed or 0) < n:
            out["how"] += (f" The check replayed the most recent {outcome.replayed} of {n} entries (riskpy_audit "
                           f"replays at most {replay_last if replay_last is not None else n}"
                           + (f", for about {replay_seconds:g} s" if replay_seconds is not None else "")
                           + "); `riskpy-audit replay FILE` replays them all.")
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
    p_verify.add_argument("--head", metavar="HEX",
                          help="the head digest quoted with the numbers: exit 1 unless the chain ends exactly there "
                               "(catches entries cut off the end or appended)")
    p_verify.add_argument("--entries", metavar="N", type=int, help="exit 1 unless the ledger holds exactly N entries")
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
            chain = verify_chain(args.ledger, expected_head=args.head, expected_entries=args.entries)
            if args.json:
                print(json.dumps(chain.to_dict(), indent=2))
            elif chain.ok:
                matched = " (the expected head)" if args.head else ""
                print(f"intact: {chain.entries} entries, head {chain.head}{matched}")
            elif chain.broken_at is None:
                print(f"BROKEN: {chain.reason}")
            else:
                print(f"BROKEN at entry {chain.broken_at}: {chain.reason}")
            return 0 if chain.ok else 1
        if args.command == "replay":
            outcome = replay(args.ledger, rel_tol=args.rel_tol)
            print(json.dumps(outcome.to_dict(), indent=2, ensure_ascii=False) if args.json else outcome.summary())
            return 0 if outcome.ok else 1
        text = report(args.ledger, title=args.title, check=args.check)
        if args.output:
            try:
                with open(args.output, "w", encoding="utf-8") as fh:
                    fh.write(text)
            except OSError as exc:
                print(f"riskpy-audit: RP505 cannot write {_quote_path(args.output)}: {exc.strerror or exc}", file=sys.stderr)
                return 1
            print(f"wrote {args.output}")
        else:
            sys.stdout.write(text)
            sys.stdout.flush()
        return 0
    except InputError as exc:
        print(f"riskpy-audit: {exc.code} {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        from .errors import _closed_pipe

        if not _closed_pipe(exc):
            print(f"riskpy-audit: RP900 {type(exc).__name__}: {_clip(str(exc), 300)}", file=sys.stderr)
            return 1
        # The reader went away (riskpy-audit report FILE | head): stop quietly, as other CLIs do.
        with contextlib.suppress(OSError, ValueError):
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
        return 0
    except Exception as exc:  # noqa: BLE001 - a coded line, never a traceback
        print(f"riskpy-audit: RP900 {type(exc).__name__}: {_clip(str(exc), 300)}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    # `python -m riskpy.audit` runs this file as __main__, a second copy of the module; run the copy that
    # riskpy.tools reports to, so its replay flag (calls replayed are not recorded) is the one that is set.
    from riskpy import audit as _imported

    sys.exit(_imported.main())
