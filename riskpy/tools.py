"""RiskPY's verified calculators, as tools for an AI agent.

An agent that can call functions needs three things from a library: a list of what it may
call, a schema for each call, and results it can pass on as data. This module provides all
three, generated from the calculators' own signatures and docstrings — plus errors it can
act on and a way to see what has been happening:

    from riskpy import tools
    tools.manifest()      # [{name, description, input_schema, requires, available}, ...]
    tools.call("black_scholes", {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2})   # 10.4506
    tools.call("black_scholes", {..., "sigma": -0.2}, errors="return")
        # {"error": {"code": "RP105", "message": "sigma must be >= 0, got -0.2", "param": "sigma", "hint": ...}}
    tools.trace()         # the recent calls: tool, arguments, duration, ok or error code

    riskpy-tools                                   # the list, on the command line
    riskpy-tools call black_scholes '{"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2}'
    riskpy-tools trace                             # recent calls (set RISKPY_TRACE_FILE to keep them)
    riskpy-mcp                                     # the same tools over MCP (pip install "open-riskpy[mcp]")

Every calculator is a thin wrapper over a public function that the verification suite
checks (``riskpy-verify``). Tools take plain numbers, strings and lists, return JSON-able
values, and hide the conventions that bite (an aggregate-loss tool takes a severity mean and
standard deviation on the money scale, never a lognormal ``mu``). Tools that need NumPy say
so in ``requires`` and are reported as *unavailable*, not missing, on a lean install.

Every refusal carries a stable code from :mod:`riskpy.errors` (``riskpy-explain CODE``), the
parameter at fault and a hint. Three introspection tools — ``riskpy_status``,
``riskpy_doctor`` and ``riskpy_explain`` — let an agent look at the installation, the recent
calls and the meaning of an error without leaving the tool interface.
"""
from __future__ import annotations

import argparse
import collections
import collections.abc
import dataclasses
import datetime
import difflib
import functools
import inspect
import json
import math
import numbers
import os
import platform
import re
import sys
import threading
import time
import types
import typing
from typing import Any, Callable, Dict, List, Optional

from . import errors as _errors
from .errors import ArgumentTypeError, InputError, RiskPyError, UnknownNameError

__all__ = ["Tool", "TOOLS", "ToolUnavailable", "tool", "schema", "manifest", "call", "trace", "status", "main",
           "LIMITS", "PLAUSIBLE"]

_EXTRA_MODULE = {"sim": "numpy", "viz": "matplotlib"}
_EXTRA_NAME = {"sim": "NumPy", "viz": "Matplotlib"}
_EXTRA_CODE = {"sim": "RP201", "viz": "RP202"}


class ToolUnavailable(RiskPyError, RuntimeError):
    """The tool exists, but the optional dependency it needs is not installed (RP201 / RP202)."""

    default_code = "RP201"


def _extra_installed(extra: str) -> bool:
    try:
        __import__(_EXTRA_MODULE.get(extra, extra))
        return True
    except ImportError:
        return False


@dataclasses.dataclass(frozen=True)
class Tool:
    """One callable an agent may use: the function, what it does, and what it needs."""

    name: str
    fn: Callable[..., Any]
    description: str
    requires: str = ""          # "" for the dependency-free core, "sim" for NumPy
    kind: str = "calculator"    # "calculator", or "introspection" for riskpy_status / _doctor / _explain

    @property
    def available(self) -> bool:
        return not self.requires or _extra_installed(self.requires)

    @property
    def install(self) -> Optional[str]:
        return f'pip install "open-riskpy[{self.requires}]"' if self.requires else None

    def schema(self) -> dict:
        return schema(self.fn)


TOOLS: Dict[str, Tool] = {}

# How much work one tool call may ask for. The arguments come from an agent, which may be
# wrong or may be relaying someone else's input, so every size that drives a loop or an
# allocation is bounded: a call either answers in seconds or says which limit it hit (RP405).
# These bound the *tools* only; the library functions underneath take whatever you give them.
LIMITS: Dict[str, int] = {
    "trials": 1_000_000,              # aggregate_loss: simulated years
    "simulated_claims": 100_000_000,  # aggregate_loss: trials × expected_frequency (a few seconds)
    "periods": 100_000,               # bond_analytics, cds_par_spread: maturity × frequency
    "triangle": 300,                  # chain_ladder, mack_chain_ladder: origin and development periods
    "list": 1_000,                    # any other list argument
    "string": 200,                    # any string argument, in characters
    "integer": 2**31 - 1,             # any integer argument, in absolute value
}

# The commonest agent mistake is a percentage where a decimal belongs: r=5 for 5%, sigma=20
# for 20%. Both are legal inputs to the library (a 500% rate is just a number), so the
# library computes an answer — a wrong one. The tools refuse values no real market quotes:
# a rate, yield, coupon, dividend yield or spread above 100% in absolute value (RP104), a
# volatility above 500% (RP105), an LGD outside [0, 1] (RP103). The library functions
# underneath are unchanged.
PLAUSIBLE: Dict[str, float] = {
    "rate": 1.0,          # |r|, |i|, |ytm|, |coupon|, |q|, |spread|, |theta|: at most 1 (100%)
    "volatility": 5.0,    # sigma: at most 5 (500% a year)
}

_RATE_PARAMS = {"r", "i", "ytm", "coupon", "q", "rate", "spread", "theta", "par_rates"}
_VOL_PARAMS = {"sigma"}
_UNIT_PARAMS = {"lgd"}


def _show(value: Any) -> str:
    """A repr short enough to put in an error message, whatever the agent sent."""
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _json_name(value: Any) -> str:
    """The JSON type an agent sent, for "expected a number, got a string" messages."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, numbers.Real):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _check_arg(name: str, hint: Any, value: Any) -> None:
    """Refuse a value that does not fit the parameter's type hint, before any arithmetic runs.

    JSON from an agent can carry ``NaN``/``Infinity`` (Python's parser accepts them), booleans
    where numbers belong, and lists of any length; none of those should reach the calculators.
    """
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return
        members = [a for a in args if a is not type(None)]
        if len(members) == 1:
            _check_arg(name, members[0], value)
        return
    if hint is bool:
        if not isinstance(value, bool):
            raise ArgumentTypeError(f"{name} must be true or false, got {_show(value)}", code="RP404", param=name,
                                    hint=f"Send {name} as a JSON boolean (true or false), not a {_json_name(value)}.")
    elif hint in (float, int):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ArgumentTypeError(f"{name} must be a number, got {_show(value)}", code="RP404", param=name,
                                    hint=f"Send {name} as a JSON number (100, not \"100\"); got a {_json_name(value)}.")
        try:
            as_float = float(value)
        except OverflowError:
            raise InputError(f"{name} is too large to be a floating-point number", code="RP405", param=name) from None
        if not math.isfinite(as_float):
            raise InputError(f"{name} must be finite, got {value!r}", code="RP101", param=name)
        if hint is int:
            if not as_float.is_integer():
                raise InputError(f"{name} must be a whole number, got {value!r}", code="RP107", param=name)
            if abs(as_float) > LIMITS["integer"]:
                raise InputError(f"{name} is too large: |{name}| must be at most {LIMITS['integer']}", code="RP405",
                                 param=name)
    elif hint is str:
        if not isinstance(value, str):
            raise ArgumentTypeError(f"{name} must be a string, got {_show(value)}", code="RP404", param=name,
                                    hint=f"Send {name} as a JSON string; got a {_json_name(value)}.")
        if len(value) > LIMITS["string"]:
            raise InputError(f"{name} is longer than {LIMITS['string']} characters", code="RP405", param=name)
    elif origin in (list, tuple, collections.abc.Sequence):
        if not isinstance(value, (list, tuple)):
            raise ArgumentTypeError(f"{name} must be a list, got {_show(value)}", code="RP404", param=name,
                                    hint=f"Send {name} as a JSON array; got a {_json_name(value)}.")
        if len(value) > LIMITS["list"]:
            raise InputError(f"{name} has {len(value)} items; at most {LIMITS['list']} are accepted", code="RP405",
                             param=name)
        for k, item in enumerate(value):
            _check_arg(f"{name}[{k}]", args[0] if args else Any, item)


def _check_plausible(name: str, value: Any) -> None:
    """Refuse a percentage where a decimal belongs (see :data:`PLAUSIBLE`)."""
    if name in _RATE_PARAMS:
        values = value if isinstance(value, (list, tuple)) else [value]
        for k, v in enumerate(values):
            label = f"{name}[{k}]" if isinstance(value, (list, tuple)) else name
            if isinstance(v, numbers.Real) and abs(v) > PLAUSIBLE["rate"]:
                example = "150bp is 0.015" if name == "spread" else f"{v:g}% is {v / 100:g}"
                raise InputError(f"{label} = {v!r} looks like a percentage: rates are decimals, so |{label}| must be "
                                 f"at most {PLAUSIBLE['rate']:g}", code="RP104", param=label,
                                 hint=f"Divide by 100 — {example}.", value=v)
    elif name in _VOL_PARAMS and isinstance(value, numbers.Real):
        if value < 0:
            raise InputError(f"{name} must be >= 0, got {value!r}", code="RP105", param=name,
                             hint=f"{name} is an annual volatility as a decimal (0.2 means 20%) and cannot be negative.")
        if value > PLAUSIBLE["volatility"]:
            raise InputError(f"{name} = {value!r} looks like a percentage: volatility is a decimal, so {name} must "
                             f"be at most {PLAUSIBLE['volatility']:g} (500%)", code="RP105", param=name,
                             hint=f"Divide by 100 — {value:g}% is {value / 100:g}.", value=value)
    elif name in _UNIT_PARAMS and isinstance(value, numbers.Real) and not 0.0 <= value <= 1.0:
        raise InputError(f"{name} must be in [0, 1], got {value!r}", code="RP103", param=name,
                         hint=f"{name} is a decimal fraction of the exposure: 45% is 0.45.", value=value)


def _checked(fn: Callable) -> Callable:
    """Wrap a tool so every argument it is given is checked against its type hint first."""
    hints = typing.get_type_hints(fn)
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def run(*args: Any, **kwargs: Any) -> Any:
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError as exc:
            raise ArgumentTypeError(f"{fn.__name__}: {exc}", code="RP403" if "missing" in str(exc) else "RP402",
                                    hint=f"`riskpy-tools schema {fn.__name__}` lists the arguments.") from None
        for name, value in bound.arguments.items():
            if name in hints:
                _check_arg(name, hints[name], value)
            _check_plausible(name, value)
        return fn(*bound.args, **bound.kwargs)

    return run


def tool(name: Optional[str] = None, *, requires: str = "", kind: str = "calculator"):
    """Register a function as a tool. The first docstring paragraph is its description.

    The registered (and returned) function checks its arguments against its type hints
    before running, so the tool behaves the same whether it is reached through
    :func:`call`, the CLI, the MCP server or a direct Python call.
    """

    def register(fn):
        doc = inspect.getdoc(fn) or ""
        description = " ".join(doc.split("\n\n")[0].split())
        checked = _checked(fn)
        t = Tool(name or fn.__name__, checked, description, requires, kind)
        TOOLS[t.name] = t
        return checked

    return register


# ---------------------------------------------------------------------------
# JSON Schema from a signature


_PARAM_DOC = re.compile(r"^\s*([A-Za-z_]\w*):\s+(.+?)\s*$")


def _param_docs(fn: Callable) -> Dict[str, str]:
    """``name: text`` lines in the docstring become parameter descriptions."""
    return {m.group(1): m.group(2) for m in map(_PARAM_DOC.match, (inspect.getdoc(fn) or "").splitlines()) if m}


def _json_type(hint: Any) -> dict:
    if hint is float:
        return {"type": "number"}
    if hint is int:
        return {"type": "integer"}
    if hint is bool:
        return {"type": "boolean"}
    if hint is str:
        return {"type": "string"}
    if hint is type(None):
        return {"type": "null"}
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        members = [_json_type(a) for a in args]
        if all(set(m) == {"type"} and isinstance(m["type"], str) for m in members):
            names = [m["type"] for m in members]
            return {"type": names if len(names) > 1 else names[0]}
        return {"anyOf": members}
    if origin in (list, tuple, collections.abc.Sequence):
        return {"type": "array", "items": _json_type(args[0]) if args else {}}
    if origin in (dict, collections.abc.Mapping):
        return {"type": "object"}
    return {}


def schema(fn: Callable) -> dict:
    """A JSON Schema for the function's keyword arguments, from its type hints."""
    hints = typing.get_type_hints(fn)
    docs = _param_docs(fn)
    properties: Dict[str, dict] = {}
    required: List[str] = []
    for name, p in inspect.signature(fn).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        prop = dict(_json_type(hints.get(name, Any)))
        if name in docs:
            prop["description"] = docs[name]
        if p.default is p.empty:
            required.append(name)
        else:
            prop["default"] = p.default
        properties[name] = prop
    out: dict = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        out["required"] = required
    return out


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # JSON has no NaN or Infinity; json.dumps would emit tokens strict parsers reject.
        return value if math.isfinite(value) else None
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    tolist = getattr(value, "tolist", None)       # NumPy arrays and scalars
    if callable(tolist):
        return _jsonable(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    return str(value)


# ---------------------------------------------------------------------------
# The trace: what the tool layer has been asked to do, in this process


_TRACE_SIZE = 200
_trace: "collections.deque[dict]" = collections.deque(maxlen=_TRACE_SIZE)
_counts: Dict[str, Any] = {"calls": 0, "errors": 0, "by_code": collections.Counter(), "by_tool": collections.Counter()}
_trace_lock = threading.Lock()
_started = time.time()
_seq = 0


def _summarise(value: Any, depth: int = 0) -> Any:
    """Arguments as they go into the trace: scalars as given, lists and long strings abbreviated."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, str):
        return value if len(value) <= 40 else value[:37] + "..."
    if isinstance(value, (list, tuple)):
        if depth == 0 and len(value) <= 6 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            return [_summarise(v, 1) for v in value]
        return f"<array of {len(value)}>"
    if isinstance(value, dict):
        return {str(k)[:40]: _summarise(v, depth + 1) for k, v in list(value.items())[:20]} if depth == 0 else "<object>"
    return f"<{type(value).__name__}>"


def _record(name: Any, arguments: Any, started: float, error: Optional[dict], via: str) -> None:
    global _seq
    entry = {
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
        "tool": str(name)[:80],
        "args": _summarise(arguments) if isinstance(arguments, dict) else _summarise(arguments, 1),
        "ms": round((time.perf_counter() - started) * 1000.0, 3),
        "ok": error is None,
        "code": None if error is None else error["code"],
        "param": None if error is None else error.get("param"),
        "message": None if error is None else error["message"][:200],
        "via": via,
    }
    with _trace_lock:
        _seq += 1
        entry = {"seq": _seq, **entry}
        _trace.append(entry)
        _counts["calls"] += 1
        _counts["by_tool"][entry["tool"]] += 1
        if error is not None:
            _counts["errors"] += 1
            _counts["by_code"][error["code"]] += 1
    path = os.environ.get("RISKPY_TRACE_FILE")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass                      # tracing must never break a call


def trace(limit: Optional[int] = 50, clear: bool = False) -> List[dict]:
    """The most recent tool calls in this process, oldest first.

    Each entry: ``{seq, at, tool, args, ms, ok, code, param, message, via}`` — ``code`` is
    the error code when the call failed, ``via`` is "python", "cli" or "mcp". The buffer
    keeps the last 200 calls; ``clear=True`` empties it after reading. Calls made directly
    in Python (``tools.black_scholes(...)``) bypass :func:`call` and are not traced; set
    ``RISKPY_TRACE_FILE`` to also append every entry to a JSON-lines file, which is how
    ``riskpy-tools trace`` sees calls made by earlier processes.
    """
    with _trace_lock:
        entries = list(_trace)
        if clear:
            _trace.clear()
            _counts.update({"calls": 0, "errors": 0, "by_code": collections.Counter(),
                            "by_tool": collections.Counter()})
    return entries[-limit:] if limit else entries


def status(recent: int = 10) -> dict:
    """What an agent needs to orient itself: versions, tools, limits, and what has happened so far."""
    from . import __version__

    with _trace_lock:
        counts = {"calls": _counts["calls"], "errors": _counts["errors"], "by_code": dict(_counts["by_code"]),
                  "by_tool": dict(_counts["by_tool"])}
    unavailable = [{"name": t.name, "install": t.install} for t in TOOLS.values() if not t.available]
    calculators = [t for t in TOOLS.values() if t.kind == "calculator"]
    return {
        "version": __version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "uptime_seconds": round(time.time() - _started, 1),
        "tools": {"calculators": len(calculators), "available": sum(t.available for t in calculators),
                  "unavailable": unavailable,
                  "introspection": [t.name for t in TOOLS.values() if t.kind == "introspection"]},
        "limits": dict(LIMITS),
        "plausible": dict(PLAUSIBLE),
        "trace": {**counts, "recent": trace(recent) if recent else []},
        "trace_file": os.environ.get("RISKPY_TRACE_FILE"),
        "next_steps": "riskpy_doctor checks the installation; riskpy_explain takes an error code (RP105) or a tool "
                      "name (black_scholes) and says what it means or which formula and conventions it uses.",
    }


# ---------------------------------------------------------------------------
# The public surface


def manifest() -> List[dict]:
    """Every tool as data: name, description, input schema, what it needs, whether it can run."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.schema(), "requires": t.requires,
         "available": t.available, "install": t.install, "kind": t.kind}
        for t in TOOLS.values()
    ]


def _lookup(name: Any) -> Tool:
    if not isinstance(name, str) or name not in TOOLS:
        close = difflib.get_close_matches(str(name), list(TOOLS), n=3, cutoff=0.5)
        hint = (f"Did you mean {' or '.join(repr(c) for c in close)}? " if close else "") + \
            "`riskpy-tools` (or tools.manifest()) lists every tool."
        raise UnknownNameError(f"unknown tool {name!r}; `riskpy-tools` lists them", code="RP401", param="name",
                               hint=hint)
    return TOOLS[name]


# Names other libraries (and agents) use for RiskPY's parameters.
_SYNONYMS: Dict[str, tuple] = {
    "vol": ("sigma",), "volatility": ("sigma",), "sd": ("sigma", "severity_sd"), "spot": ("S",), "s": ("S",),
    "underlying": ("S",), "strike": ("K",), "k": ("K",), "expiry": ("T",), "t": ("T", "t"), "tenor": ("T", "maturity"),
    "time": ("T", "t"), "years": ("T", "maturity", "n"), "rate": ("r", "i", "rate"), "risk_free": ("r",),
    "risk_free_rate": ("r",), "interest_rate": ("i", "r"), "interest": ("i", "r"), "dividend": ("q",),
    "dividend_yield": ("q",), "type": ("kind", "model"), "option_type": ("kind",), "call_put": ("kind",),
    "age": ("x",), "term": ("n",), "yield": ("ytm",), "par": ("face",), "notional": ("face", "ead"),
    "exposure": ("ead",), "probability_of_default": ("pd",), "default_probability": ("pd",),
    "loss_given_default": ("lgd",), "correlation": ("rho",), "confidence": ("level",), "alpha": ("level",),
    "lambda": ("expected_frequency", "hazard"), "frequency_mean": ("expected_frequency",), "n_trials": ("trials",),
    "simulations": ("trials",), "n_sims": ("trials",), "data": ("triangle", "returns"), "elr": ("expected_loss_ratio",),
}


def _suggest(name: str, accepted: "set[str]") -> Optional[str]:
    for candidate in _SYNONYMS.get(str(name).lower(), ()):
        if candidate in accepted:
            return candidate
    close = difflib.get_close_matches(str(name), sorted(accepted), n=1, cutoff=0.6)
    return close[0] if close else None


def _arguments(t: Tool, arguments: Any) -> dict:
    if arguments is None:
        return {}
    if not isinstance(arguments, dict):
        raise ArgumentTypeError(f"{t.name}: the arguments must be a JSON object, got {_json_name(arguments)}",
                                code="RP406", param=None)
    params = inspect.signature(t.fn).parameters
    accepted = set(params)
    unknown = sorted(set(arguments) - accepted)
    if unknown:
        guesses = {u: _suggest(u, accepted) for u in unknown}
        suggestion = "; ".join(f"{u!r} → {g!r}?" for u, g in guesses.items() if g)
        raise ArgumentTypeError(
            f"{t.name}: unexpected argument(s) {unknown}; accepted: {sorted(accepted)}", code="RP402",
            param=str(unknown[0]),
            hint=(f"Did you mean {suggestion} " if suggestion else "") + f"`riskpy-tools schema {t.name}` lists them.")
    missing = [p for p, spec in params.items() if spec.default is spec.empty and p not in arguments]
    if missing:
        docs = _param_docs(t.fn)
        what = f" ({docs[missing[0]]})" if missing[0] in docs else ""
        raise ArgumentTypeError(f"{t.name}: missing required argument(s) {missing}", code="RP403", param=missing[0],
                                hint=f"Pass {missing[0]}{what}. Required: {schema(t.fn).get('required', [])}.")
    return dict(arguments)


def call(name: str, arguments: Optional[dict] = None, *, errors: str = "raise", via: str = "python") -> Any:
    """Run a tool by name with keyword arguments; the result is JSON-able.

    ``errors="raise"`` (the default) raises the error — a :class:`riskpy.errors.RiskPyError`
    subclass that is also the ``ValueError`` / ``TypeError`` / ``KeyError`` earlier versions
    raised. ``errors="return"`` never raises for a failed call: it returns
    ``{"error": {"code", "message", "param", "hint", "docs"}}`` instead, which is what the CLI
    and the MCP server send to an agent. Every call is recorded in :func:`trace`.
    """
    if errors not in ("raise", "return"):
        raise InputError(f"errors must be 'raise' or 'return', got {errors!r}", code="RP108", param="errors")
    started = time.perf_counter()
    t: Optional[Tool] = None
    try:
        t = _lookup(name)
        args = _arguments(t, arguments)
        if not t.available:
            module = _EXTRA_MODULE.get(t.requires, t.requires)
            raise ToolUnavailable(f"{name} needs {_EXTRA_NAME.get(t.requires, t.requires)} (the {module} package), "
                                  f"which is not installed: {t.install}",
                                  code=_EXTRA_CODE.get(t.requires, "RP201"), param=None, hint=t.install)
        result = _jsonable(t.fn(**args))
    except Exception as exc:  # noqa: BLE001 - every failure becomes a structured error
        detail = _errors.from_exception(exc)
        if t is None or t.kind == "calculator":
            _record(name, arguments, started, detail, via)
        if errors == "return":
            return {"error": detail}
        raise
    if t.kind == "calculator":
        _record(name, arguments, started, None, via)
    return result


# ---------------------------------------------------------------------------
# The calculators. Plain numbers in, JSON out, verified functions underneath.


def _bs_explain(S: float, K: float, T: float, r: float, sigma: float, q: float, kind: str) -> dict:
    from . import quant

    out: Dict[str, Any] = {
        "formula": ("call = S·e^(−qT)·N(d1) − K·e^(−rT)·N(d2); put = K·e^(−rT)·N(−d2) − S·e^(−qT)·N(−d1); "
                    "d1 = [ln(S/K) + (r − q + σ²/2)·T] / (σ·√T), d2 = d1 − σ·√T"),
        "conventions": ["r and q are continuously compounded annual rates as decimals (convert an effective or "
                        "semi-annual rate with convert_rate first)",
                        "sigma is the annual volatility of log returns, as a decimal",
                        "T is in years; European exercise only"],
    }
    if T == 0 or sigma == 0:
        out["degenerate"] = ("T = 0 or sigma = 0: the price is the discounted intrinsic value "
                             "max(±(S·e^(−qT) − K·e^(−rT)), 0); d1 and d2 are undefined")
        return out
    vol_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_t
    d2 = d1 - vol_t
    out.update({"d1": d1, "d2": d2, "N(d1)": quant.norm_cdf(d1), "N(d2)": quant.norm_cdf(d2),
                "discounted_spot": S * math.exp(-q * T), "discounted_strike": K * math.exp(-r * T)})
    return out


@tool()
def black_scholes(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, kind: str = "call",
                  explain: bool = False) -> Any:
    """Black–Scholes–Merton price of a European option.

    S: spot price of the underlying
    K: strike price
    T: time to expiry, in years
    r: continuously compounded risk-free rate as a decimal (0.05 means 5%)
    sigma: annual volatility as a decimal (0.2 means 20%)
    q: continuous dividend yield as a decimal
    kind: "call" or "put"
    explain: true to return {price, explain} with d1, d2, N(d1), N(d2), the formula and the rate convention
    """
    from . import quant

    price = float(quant.black_scholes(S, K, T, r, sigma, q=q, kind=kind))
    if explain:
        return {"price": price, "explain": _bs_explain(S, K, T, r, sigma, q, kind.lower())}
    return price


@tool()
def option_greeks(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, kind: str = "call") -> dict:
    """Price and the five first-order Greeks (delta, gamma, vega, theta, rho) of a European option.

    S: spot price of the underlying
    K: strike price
    T: time to expiry, in years
    r: continuously compounded risk-free rate as a decimal
    sigma: annual volatility as a decimal
    q: continuous dividend yield as a decimal
    kind: "call" or "put"
    """
    from . import quant

    return _jsonable(quant.greeks(S, K, T, r, sigma, q=q, kind=kind))


@tool()
def implied_vol(price: float, S: float, K: float, T: float, r: float, q: float = 0.0, kind: str = "call") -> float:
    """The volatility that reproduces an observed European option price.

    price: the observed option price
    S: spot price of the underlying
    K: strike price
    T: time to expiry, in years
    r: continuously compounded risk-free rate as a decimal
    q: continuous dividend yield as a decimal
    kind: "call" or "put"
    """
    from . import quant

    return float(quant.implied_vol(price, S, K, T, r, q=q, kind=kind))


def _check_periods(maturity: float, frequency: int) -> None:
    if maturity > 0 and frequency > 0 and maturity * frequency > LIMITS["periods"]:
        raise InputError(f"maturity × frequency = {maturity * frequency:g} payment periods; "
                         f"the tool accepts at most {LIMITS['periods']}", code="RP405", param="maturity")


@tool()
def bond_analytics(face: float, coupon: float, maturity: float, ytm: float, frequency: int = 2,
                   explain: bool = False) -> dict:
    """Price, durations, convexity and DV01 of a fixed-coupon bond at a given yield.

    face: face (par) value
    coupon: annual coupon rate as a decimal (0.05 means 5%)
    maturity: years to maturity
    ytm: yield to maturity as a decimal, compounded `frequency` times a year
    frequency: coupons per year
    explain: true to add an "explain" entry with the formulas and conventions used
    """
    from . import rates

    _check_periods(maturity, frequency)
    bond = rates.Bond(face=face, coupon=coupon, maturity=maturity, frequency=frequency)
    out: Dict[str, Any] = {
        "price": bond.price(ytm), "macaulay_duration": bond.macaulay_duration(ytm),
        "modified_duration": bond.modified_duration(ytm), "convexity": bond.convexity(ytm), "dv01": bond.dv01(ytm),
    }
    if explain:
        out["explain"] = {
            "formula": "price = Σ CF_t·(1 + y/m)^(−m·t); modified = Macaulay / (1 + y/m); "
                       "convexity = Σ t(t + 1/m)·PV_t / (P·(1 + y/m)²); DV01 = modified · price · 0.0001",
            "conventions": [f"ytm is compounded {frequency} times a year (the coupon frequency)",
                            "price is the dirty price, in the units of face",
                            "durations are in years; DV01 is the price change for a 1bp yield move, positive"],
        }
    return out


@tool()
def basel_irb_capital(pd: float, lgd: float, ead: float, maturity: float = 2.5, asset_class: str = "corporate") -> dict:
    """Basel II/III internal-ratings-based capital for one exposure: risk weight, RWA, capital.

    pd: one-year probability of default as a decimal (0.01 means 1%)
    lgd: loss given default as a decimal
    ead: exposure at default, in money
    maturity: effective maturity in years (1 to 5 for corporate, sovereign and bank)
    asset_class: "corporate", "sovereign", "bank", "retail_mortgage", "retail_revolving" or "retail_other"
    """
    from . import credit

    result = credit.basel_irb_capital(pd=pd, lgd=lgd, ead=ead, maturity=maturity, asset_class=asset_class)
    out = _jsonable(result)
    out["risk_weight"] = result.risk_weight
    return out


@tool()
def expected_loss(pd: float, lgd: float, ead: float) -> float:
    """Expected loss on a single exposure: PD × LGD × EAD.

    pd: one-year probability of default as a decimal
    lgd: loss given default as a decimal
    ead: exposure at default, in money
    """
    from . import credit

    return float(credit.expected_loss(pd, lgd, ead))


@tool()
def merton(assets: float, debt: float, T: float, r: float, sigma: float) -> dict:
    """Merton's structural credit model: distance to default, default probability, equity and debt values, spread.

    assets: market value of the firm's assets
    debt: face value of the zero-coupon debt due at T
    T: years to the debt's maturity
    r: continuously compounded risk-free rate as a decimal
    sigma: annual volatility of the assets as a decimal
    """
    from . import credit

    return _jsonable(credit.merton(assets=assets, debt=debt, T=T, r=r, sigma=sigma))


@tool()
def cds_par_spread(hazard: float, recovery: float, r: float, maturity: float, frequency: int = 4) -> float:
    """The CDS spread (as a decimal) that makes the contract worth zero at inception.

    hazard: constant hazard rate (default intensity) per year, as a decimal
    recovery: recovery rate as a decimal (0.4 means 40%)
    r: continuously compounded risk-free rate as a decimal
    maturity: years to maturity
    frequency: premium payments per year
    """
    from . import credit

    _check_periods(maturity, frequency)
    return float(credit.cds_par_spread(hazard, recovery, r, maturity, frequency=frequency))


@tool()
def annuity_certain(i: float, n: int, due: bool = False) -> float:
    """Present value of 1 a year for n years at effective annual rate i.

    i: effective annual interest rate as a decimal (0.05 means 5%)
    n: number of payments
    due: true for payments at the start of each year, false for the end
    """
    from . import life

    return float(life.annuity_certain(i, n, due=due))


def _table(name: str):
    from . import life

    if name.strip().lower() != "sult":
        raise InputError(f'table must be "sult" (the AMLCR Standard Ultimate Life Table), got {name!r}', code="RP108",
                         param="table", hint='The tools carry one table: "sult". For your own, use riskpy.life.LifeTable.')
    return life.LifeTable.sult()


@tool()
def life_annuity_due(x: int, i: float, table: str = "sult") -> float:
    """ä_x: present value of 1 a year, paid in advance, while a life aged x survives.

    x: the life's current age
    i: effective annual interest rate as a decimal
    table: the life table; "sult" is the AMLCR Standard Ultimate Life Table
    """
    from . import life

    return float(life.whole_life_annuity_due(_table(table), x, i))


@tool()
def life_insurance(x: int, i: float, table: str = "sult") -> float:
    """A_x: present value of 1 paid at the end of the year of death of a life aged x.

    x: the life's current age
    i: effective annual interest rate as a decimal
    table: the life table; "sult" is the AMLCR Standard Ultimate Life Table
    """
    from . import life

    return float(life.whole_life_insurance(_table(table), x, i))


@tool()
def net_premium_reserve(x: int, t: int, i: float, kind: str = "whole_life", n: Optional[int] = None,
                        table: str = "sult") -> float:
    """tV: the prospective net premium reserve at duration t, per unit sum assured.

    x: age at issue
    t: policy duration in years at which to value the reserve
    i: effective annual interest rate as a decimal
    kind: "whole_life", "term" or "endowment"
    n: the term in years, for "term" and "endowment"
    table: the life table; "sult" is the AMLCR Standard Ultimate Life Table
    """
    from . import life

    return float(life.net_premium_reserve(_table(table), x, t=t, i=i, kind=kind, n=n))


@tool()
def aggregate_loss(expected_frequency: float, severity_mean: float, severity_sd: float, trials: int = 100_000,
                   seed: int = 0) -> dict:
    """Simulate a year of claims with the compiled core: Poisson frequency, lognormal severity; VaR and TVaR out.

    expected_frequency: mean number of claims in a year
    severity_mean: mean claim size on the money scale
    severity_sd: standard deviation of claim size on the money scale
    trials: simulated years; each draws a claim count and sums that many severities
    seed: random seed; the same non-zero seed gives the same answer, 0 draws a fresh seed each call
    """
    from . import MonteCarloSimulator

    for name, value in (("severity_mean", severity_mean), ("severity_sd", severity_sd)):
        if value <= 0:
            raise InputError(f"severity_mean and severity_sd must be positive ({name} = {value!r})", code="RP102",
                             param=name, hint="Give the claim-size mean and standard deviation in money, both > 0; "
                                              "the tool converts them to the lognormal's mu and sigma itself.")
    if expected_frequency < 0:
        raise InputError(f"expected_frequency must be >= 0, got {expected_frequency!r}", code="RP102",
                         param="expected_frequency")
    if not 1 <= trials <= LIMITS["trials"]:
        raise InputError(f"trials must be between 1 and {LIMITS['trials']}, got {trials!r}",
                         code="RP405" if trials > 1 else "RP107", param="trials")
    if seed < 0:
        raise InputError(f"seed must be >= 0, got {seed!r}", code="RP102", param="seed",
                         hint="Use a positive seed for a reproducible run, or 0 for a fresh one.")
    if expected_frequency * trials > LIMITS["simulated_claims"]:
        raise InputError(f"expected_frequency × trials = {expected_frequency * trials:g} simulated claims; the tool "
                         f"accepts at most {LIMITS['simulated_claims']:g} — lower trials", code="RP405", param="trials")
    sigma_sq = math.log(1.0 + (severity_sd / severity_mean) ** 2)     # the same conversion as LogNormal.from_moments
    mu, sigma = math.log(severity_mean) - 0.5 * sigma_sq, math.sqrt(sigma_sq)
    losses = sorted(MonteCarloSimulator(trials=trials, seed=seed).simulate_aggregate_loss(expected_frequency, mu, sigma))
    n = len(losses)
    mean = sum(losses) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in losses) / (n - 1)) if n > 1 else 0.0
    levels = (0.5, 0.75, 0.9, 0.95, 0.99, 0.995)
    var = {str(level): losses[min(n - 1, int(level * n))] for level in levels}
    tail = losses[int(0.995 * n):] or losses[-1:]
    return {
        "trials": n, "mean": mean, "std": std, "min": losses[0], "max": losses[-1], "var": var,
        "tvar_0.995": sum(tail) / len(tail), "severity_mu": mu, "severity_sigma": sigma,
    }


def _triangle(data: List[List[Optional[float]]]):
    from . import reserving

    if len(data) > LIMITS["triangle"] or any(len(row) > LIMITS["triangle"] for row in data):
        raise InputError(f"the triangle may have at most {LIMITS['triangle']} origin and {LIMITS['triangle']} "
                         "development periods", code="RP405", param="triangle")
    # A run-off triangle is oldest origin first: each origin has been developing at least as long
    # as the ones after it. A newer row with more observed periods than an older one means the rows
    # are upside down or one is truncated — the library would still compute, and the answer would
    # be wrong, so the tool refuses.
    observed = [sum(v is not None for v in row) for row in data]
    for k in range(1, len(observed)):
        if observed[k] > observed[k - 1]:
            raise InputError(f"triangle row {k} has {observed[k]} observed development periods but row {k - 1} has "
                             f"{observed[k - 1]}; rows are origin periods, oldest first, so they cannot get longer "
                             "going down", code="RP502", param="triangle",
                             hint="Order the rows oldest origin first; pad the not-yet-observed cells of each row "
                                  "with null, e.g. [[100, 150, 160], [110, 170, null], [120, null, null]].")
    return reserving.Triangle(data)


def _reserve_dict(res) -> dict:
    return {
        "method": res.method, "total_latest": res.total_latest, "total_ultimate": res.total_ultimate,
        "total_reserve": res.total_reserve, "factors": _jsonable(res.factors), "origin": _jsonable(res.origin),
        "latest": _jsonable(res.latest), "ultimate": _jsonable(res.ultimate), "reserve": _jsonable(res.reserve),
    }


@tool(requires="sim")
def chain_ladder(triangle: List[List[Optional[float]]], tail: float = 1.0) -> dict:
    """Chain-ladder reserves from a cumulative claims triangle: development factors, ultimates and reserves by origin.

    triangle: rows are origin periods, columns are development periods, cumulative amounts; null where not yet observed
    tail: tail factor applied after the last development period (1.0 for none)
    """
    from . import reserving

    return _reserve_dict(reserving.chain_ladder(_triangle(triangle), tail=tail))


@tool(requires="sim")
def mack_chain_ladder(triangle: List[List[Optional[float]]], tail: float = 1.0) -> dict:
    """Chain ladder with Mack's distribution-free standard errors of the reserves.

    triangle: rows are origin periods, columns are development periods, cumulative amounts; null where not yet observed
    tail: tail factor applied after the last development period (1.0 for none)
    """
    from . import reserving

    res = reserving.mack_chain_ladder(_triangle(triangle), tail=tail)
    out = _reserve_dict(res)
    out.update({"se": _jsonable(res.se), "total_se": res.total_se, "total_cv": res.total_cv})
    return out


@tool()
def verify(modules: Optional[List[str]] = None, oracle: bool = False) -> dict:
    """Run RiskPY's verification suite: every identity the library claims, checked against sources it had no hand in.

    modules: restrict the run to these modules (core, special, mc, quant, life, reserving, rates, credit, viz)
    oracle: also cross-check the special functions against SciPy when it is installed (slower)
    """
    from . import verify as _verify

    unknown = sorted(set(modules or []) - set(_verify.MODULES))
    if unknown:
        raise InputError(f"modules: unknown module(s) {unknown}; choose from {list(_verify.MODULES)}", code="RP108",
                         param="modules")
    report = _verify.run(modules, oracle=oracle)
    return {
        "ok": report.ok, "version": report.version, "checks": len(report.checks), "passed": len(report.passed),
        "failed": len(report.failed), "seconds": round(report.seconds, 3),
        "skipped": [{"module": m, "reason": r.strip().splitlines()[0]} for m, r in report.skipped],
        "failures": [{"module": c.module, "name": c.name, "value": c.value, "reference": c.reference,
                      "tolerance": c.tolerance} for c in report.failed],
    }


# -- added in 0.5.0: more verified functions, same conventions ----------------------------


_LIFE_KINDS = ("whole_life", "term", "endowment")


@tool()
def life_premium(x: int, i: float, kind: str = "whole_life", n: Optional[int] = None, table: str = "sult") -> dict:
    """Annual net premium by the equivalence principle (P = A / ä), with the two values it is the ratio of.

    x: age at issue
    i: effective annual interest rate as a decimal
    kind: "whole_life" (premiums for life), "term" or "endowment" (premiums for n years)
    n: the term in years, for "term" and "endowment"
    table: the life table; "sult" is the AMLCR Standard Ultimate Life Table
    """
    from . import life

    tbl = _table(table)
    key = kind.strip().lower()
    if key not in _LIFE_KINDS:
        raise InputError(f"kind must be one of {list(_LIFE_KINDS)}, got {kind!r}", code="RP108", param="kind")
    if key == "whole_life":
        if n is not None:
            raise InputError("whole_life has no term — leave n as null (use kind='term' or 'endowment' for an "
                             "n-year contract)", code="RP113", param="n")
        benefit, annuity = life.whole_life_insurance(tbl, x, i), life.whole_life_annuity_due(tbl, x, i)
        premium = life.whole_life_premium(tbl, x, i)
    else:
        if n is None:
            raise InputError(f"kind={key!r} needs a term: pass n (the number of years)", code="RP113", param="n")
        insurance = life.term_insurance if key == "term" else life.endowment_insurance
        benefit, annuity = insurance(tbl, x, n, i), life.temporary_annuity_due(tbl, x, n, i)
        premium = (life.term_premium if key == "term" else life.endowment_premium)(tbl, x, n, i)
    return {"premium": float(premium), "benefit_apv": float(benefit), "annuity_apv": float(annuity), "kind": key}


@tool()
def term_insurance(x: int, n: int, i: float, table: str = "sult") -> float:
    """A¹_{x:n}: present value of 1 paid at the end of the year of death, if death is within n years.

    x: the life's current age
    n: the term in years
    i: effective annual interest rate as a decimal
    table: the life table; "sult" is the AMLCR Standard Ultimate Life Table
    """
    from . import life

    return float(life.term_insurance(_table(table), x, n, i))


@tool()
def endowment_insurance(x: int, n: int, i: float, table: str = "sult") -> float:
    """A_{x:n}: present value of 1 paid on death within n years, or at time n on survival.

    x: the life's current age
    n: the term in years
    i: effective annual interest rate as a decimal
    table: the life table; "sult" is the AMLCR Standard Ultimate Life Table
    """
    from . import life

    return float(life.endowment_insurance(_table(table), x, n, i))


@tool()
def temporary_annuity_due(x: int, n: int, i: float, table: str = "sult") -> float:
    """ä_{x:n}: present value of 1 a year in advance for at most n years while a life aged x survives.

    x: the life's current age
    n: the maximum number of payments
    i: effective annual interest rate as a decimal
    table: the life table; "sult" is the AMLCR Standard Ultimate Life Table
    """
    from . import life

    return float(life.temporary_annuity_due(_table(table), x, n, i))


def _compounding(value: str, name: str) -> Any:
    text = value.strip().lower()
    return int(text) if text.isdigit() else text


@tool()
def convert_rate(rate: float, from_compounding: str, to_compounding: str) -> float:
    """Re-express an annual interest rate in another compounding convention (same value of money over a year).

    rate: the rate as a decimal (0.05 means 5%)
    from_compounding: the rate's convention: "continuous", "annual", "semiannual", "quarterly", "monthly", or periods per year such as "12"
    to_compounding: the convention wanted, in the same form
    """
    from . import rates

    return float(rates.convert_rate(rate, _compounding(from_compounding, "from_compounding"),
                                    _compounding(to_compounding, "to_compounding")))


@tool()
def bootstrap_yield_curve(par_rates: List[float], maturities: List[float], frequency: int = 1,
                          at: Optional[List[float]] = None) -> dict:
    """Bootstrap a zero curve from par yields: continuously compounded zero rates and discount factors.

    par_rates: par coupon rates as decimals, one per maturity
    maturities: maturities in years, strictly increasing, each a whole number of coupon periods
    frequency: coupons per year of the par instruments
    at: extra times in years at which to report the zero rate and discount factor (linear in zero rates between knots)
    """
    from . import rates

    if len(par_rates) != len(maturities):
        raise InputError(f"par_rates and maturities must have the same length, got {len(par_rates)} and "
                         f"{len(maturities)}", code="RP109", param="maturities")
    curve = rates.YieldCurve.bootstrap(par_rates, maturities, frequency=frequency)
    times = list(curve.times)
    out = {"times": times, "zero_rates": list(curve.zero_rates), "discount_factors": [curve.df(t) for t in times],
           "compounding": "continuous"}
    if at:
        out["at"] = [{"t": t, "zero_rate": curve.zero(t) if t > 0 else None, "discount_factor": curve.df(t)}
                     for t in at]
    return out


@tool()
def short_rate_zero_bond(model: str, r: float, T: float, kappa: float, theta: float, sigma: float,
                         t: float = 0.0) -> dict:
    """Zero-coupon bond price under the Vasicek or Cox–Ingersoll–Ross short-rate model (closed form), and its yield.

    model: "vasicek" or "cir"
    r: the short rate now, as a decimal
    T: the bond's maturity in years
    kappa: speed of mean reversion per year (> 0)
    theta: long-run mean of the short rate, as a decimal
    sigma: volatility of the short rate (absolute for Vasicek, scaled by √r for CIR)
    t: the valuation time in years (0 for today)
    """
    from . import rates

    key = model.strip().lower()
    if key not in ("vasicek", "cir"):
        raise InputError(f"model must be 'vasicek' or 'cir', got {model!r}", code="RP108", param="model")
    price = (rates.vasicek_zero_bond if key == "vasicek" else rates.cir_zero_bond)(r, t, T, kappa, theta, sigma)
    tau = T - t
    return {"price": float(price), "yield": (-math.log(price) / tau) if tau > 0 else None, "model": key,
            "compounding": "continuous"}


@tool()
def hazard_from_spread(spread: float, recovery: float, horizon: float = 1.0) -> dict:
    """Credit triangle: the hazard rate implied by a credit spread, h = s / (1 − R), and the default probability over a horizon.

    spread: the credit spread as a decimal (0.015 means 150bp)
    recovery: recovery rate as a decimal (0.4 means 40%)
    horizon: years over which to report survival and default probability under the flat hazard
    """
    from . import credit

    h = credit.hazard_from_spread(spread, recovery)
    survival = credit.survival_probability(h, horizon)
    return {"hazard": float(h), "survival_probability": float(survival), "default_probability": 1.0 - float(survival),
            "horizon": horizon}


@tool()
def vasicek_loss_quantile(level: float, pd: float, rho: float) -> float:
    """Loss fraction of a large homogeneous portfolio at a confidence level, Vasicek/ASRF (the Basel IRB engine).

    level: confidence level as a decimal (0.999 means the 99.9th percentile)
    pd: probability of default of each obligor, as a decimal
    rho: asset correlation, as a decimal
    """
    from . import credit

    return float(credit.vasicek_loss_quantile(level, pd, rho))


@tool(requires="sim")
def value_at_risk(returns: List[float], level: float = 0.99, method: str = "historical") -> dict:
    """Value at risk and expected shortfall of a return series, as positive losses.

    returns: periodic returns as decimals (-0.02 means a 2% loss); at least two
    level: confidence level as a decimal (0.99 means 99%)
    method: "historical" (empirical quantile) or "parametric" (normal, from the sample mean and standard deviation)
    """
    from . import quant

    key = method.strip().lower()
    if key not in ("historical", "parametric"):
        raise InputError(f"method must be 'historical' or 'parametric', got {method!r}", code="RP108", param="method")
    if len(returns) < 2:
        raise InputError(f"returns needs at least two observations, got {len(returns)}", code="RP111", param="returns")
    var = quant.historical_var(returns, level) if key == "historical" else quant.parametric_var(returns, level)
    return {"var": float(var), "expected_shortfall": float(quant.expected_shortfall(returns, level)),
            "level": level, "method": key, "observations": len(returns)}


@tool(requires="sim")
def bornhuetter_ferguson(triangle: List[List[Optional[float]]], premiums: List[float],
                         expected_loss_ratio: float, tail: float = 1.0) -> dict:
    """Bornhuetter–Ferguson reserves: premium × expected loss ratio × (1 − 1/CDF), CDFs from the chain ladder.

    triangle: rows are origin periods, columns are development periods, cumulative amounts; null where not yet observed
    premiums: earned premium for each origin period, in money
    expected_loss_ratio: the a-priori loss ratio as a decimal (0.65 means 65%)
    tail: tail factor applied after the last development period (1.0 for none)
    """
    from . import reserving

    tri = _triangle(triangle)
    if len(premiums) != tri.n_origin:
        raise InputError(f"premiums has {len(premiums)} entries for {tri.n_origin} origin periods — one per row",
                         code="RP109", param="premiums")
    res = reserving.bornhuetter_ferguson(tri, premiums, expected_loss_ratio, tail=tail)
    return {
        "method": res.method, "total_latest": res.total_latest, "total_ultimate": res.total_ultimate,
        "total_reserve": res.total_reserve, "origin": _jsonable(res.origin), "latest": _jsonable(res.latest),
        "ultimate": _jsonable(res.ultimate), "reserve": _jsonable(res.reserve),
    }


# ---------------------------------------------------------------------------
# Introspection: what an agent calls when something goes wrong


# The formula and conventions behind the tools an agent is likeliest to misuse. Everything
# here restates what the wrapped function does; riskpy_explain returns it with the schema.
_EXPLAIN: Dict[str, Dict[str, Any]] = {
    "black_scholes": {
        "formula": "call = S·e^(−qT)·N(d1) − K·e^(−rT)·N(d2); d1 = [ln(S/K) + (r − q + σ²/2)T]/(σ√T), d2 = d1 − σ√T",
        "conventions": ["r and q continuously compounded, decimals", "sigma annual, decimal", "T in years",
                        "T = 0 or sigma = 0 gives the discounted intrinsic value", "pass explain=true for d1 and d2"],
    },
    "option_greeks": {
        "formula": "delta = e^(−qT)N(d1) (call); gamma = e^(−qT)φ(d1)/(Sσ√T); vega = S·e^(−qT)φ(d1)√T",
        "conventions": ["vega and rho are per 1 percentage point (raw derivative / 100)",
                        "theta is per calendar day (raw derivative / 365)", "delta and gamma per unit of S"],
    },
    "implied_vol": {
        "formula": "solve black_scholes(sigma) = price by bisection on a bracket that starts at [1e-9, 5] and widens",
        "conventions": ["needs T > 0", "a price below the discounted intrinsic value is RP301, above the "
                        "no-arbitrage ceiling RP302: the quote is wrong, not the solver"],
    },
    "bond_analytics": {
        "formula": "price = Σ CF_t(1 + y/m)^(−mt); modified = Macaulay/(1 + y/m); DV01 = modified·price·1e-4",
        "conventions": ["ytm compounded `frequency` times a year", "dirty price in the units of face"],
    },
    "basel_irb_capital": {
        "formula": "K = LGD·[N((N⁻¹(PD) + √R·N⁻¹(0.999))/√(1−R)) − PD]·MA; RWA = 12.5·K·EAD",
        "conventions": ["R from basel_correlation(PD, asset_class)", "maturity adjustment only for corporate, "
                        "sovereign and bank; maturity in [1, 5]", "pd must be > 0 (floor 0.0003)"],
    },
    "aggregate_loss": {
        "formula": "S = Σ_{k=1..N} X_k, N ~ Poisson(λ), X ~ LogNormal(μ, σ) with σ² = ln(1 + sd²/mean²), "
                   "μ = ln(mean) − σ²/2",
        "conventions": ["severity mean and sd on the money scale (the tool converts)", "seed 0 is a fresh seed",
                        "var keys are levels; tvar_0.995 is the mean of the worst 0.5%"],
    },
    "life_annuity_due": {
        "formula": "ä_x = Σ_k v^k · kp_x, v = 1/(1 + i)",
        "conventions": ["i is an effective annual rate (not continuous)", "AMLCR SULT, ages 20–120"],
    },
    "cds_par_spread": {
        "formula": "spread = protection leg / risky annuity, flat hazard, premium accrued on default",
        "conventions": ["r continuous; the credit triangle h(1 − R) is the continuous-premium limit"],
    },
    "convert_rate": {
        "formula": "through the one-year discount factor: (1 + r/m)^(−m) = e^(−r_c)",
        "conventions": ["'annual' = effective annual = 1 period a year", "'continuous' is what black_scholes, "
                        "merton and cds_par_spread take"],
    },
}


@tool(kind="introspection")
def riskpy_status(recent: int = 10) -> dict:
    """RiskPY's state for an agent: version, which tools can run, the limits, and the recent tool calls with any error codes.

    recent: how many of the most recent calls to include (0 for none)
    """
    return status(max(0, min(int(recent), _TRACE_SIZE)))


@tool(kind="introspection")
def riskpy_doctor() -> dict:
    """Check this RiskPY installation: Python, the compiled core, optional extras, a numerical self-check, the MCP server."""
    from . import diagnostics

    return diagnostics.doctor()


@tool(kind="introspection")
def riskpy_explain(topic: str) -> dict:
    """Explain an error code (RP105) or a tool (black_scholes): what it means, the formula and conventions, how to fix the call.

    topic: an error code such as "RP105", or a tool name such as "black_scholes"
    """
    key = topic.strip()
    if re.fullmatch(r"(?i)RP\d{3}", key):
        return _errors.explain(key)
    if key in TOOLS:
        t = TOOLS[key]
        return {"tool": t.name, "description": t.description, "input_schema": t.schema(), "requires": t.requires,
                "available": t.available, **_EXPLAIN.get(key, {})}
    close = difflib.get_close_matches(key, list(TOOLS), n=3, cutoff=0.5)
    raise UnknownNameError(f"{topic!r} is neither an error code (RPnnn) nor a tool name", code="RP401", param="topic",
                           hint=(f"Did you mean {' or '.join(map(repr, close))}? " if close else "")
                           + "riskpy-explain --list shows the codes; riskpy-tools lists the tools.")


# ---------------------------------------------------------------------------
# CLI


def _read_trace_file(path: str, limit: Optional[int]) -> List[dict]:
    entries = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries[-limit:] if limit else entries


def _print_error(detail: dict, as_json: bool = True) -> None:
    if as_json:
        print(json.dumps({"error": detail}, indent=2, ensure_ascii=False))
    where = f" ({detail['param']})" if detail.get("param") else ""
    print(f"riskpy-tools: {detail['code']}{where} {detail['message']}\n  hint: {detail['hint']}", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    _errors._safe_streams()
    parser = argparse.ArgumentParser(prog="riskpy-tools",
                                     description="RiskPY's verified calculators as tools for an AI agent.")
    parser.add_argument("--json", action="store_true", help="print the manifest as JSON")
    sub = parser.add_subparsers(dest="command")
    p_call = sub.add_parser("call", help="run a tool: riskpy-tools call NAME '{\"arg\": value}'")
    p_call.add_argument("name")
    p_call.add_argument("arguments", nargs="?", default="{}", help="the arguments as a JSON object")
    p_schema = sub.add_parser("schema", help="print one tool's input schema")
    p_schema.add_argument("name")
    p_trace = sub.add_parser("trace", help="recent tool calls: this process, plus RISKPY_TRACE_FILE if set")
    p_trace.add_argument("--json", action="store_true", dest="trace_json", help="print JSON")
    p_trace.add_argument("--limit", type=int, default=50, help="how many entries (default 50)")
    p_trace.add_argument("--file", help="read this JSON-lines trace file (default: $RISKPY_TRACE_FILE)")
    args = parser.parse_args(argv)

    if args.command == "call":
        try:
            arguments = json.loads(args.arguments)
        except json.JSONDecodeError as exc:
            _print_error({"code": "RP406", "message": f"the arguments are not valid JSON: {exc}", "param": None,
                          "hint": _errors.CATALOGUE["RP406"]["hint"], "docs": f"{_errors.DOCS}#rp406"})
            return 1
        out = call(args.name, arguments, errors="return", via="cli")
        if isinstance(out, dict) and set(out) == {"error"}:
            _print_error(out["error"])
            return 1
        print(json.dumps(out, indent=2, allow_nan=False))
        return 0
    if args.command == "schema":
        try:
            print(json.dumps(_lookup(args.name).schema(), indent=2))
        except UnknownNameError as exc:
            _print_error(exc.to_dict())
            return 1
        return 0
    if args.command == "trace":
        path = args.file or os.environ.get("RISKPY_TRACE_FILE")
        entries = trace(None)
        if path and os.path.exists(path):
            entries = _read_trace_file(path, None) + entries
        entries = entries[-args.limit:] if args.limit else entries
        if args.trace_json:
            print(json.dumps(entries, indent=2, ensure_ascii=False))
            return 0
        if not entries:
            print("No tool calls traced. The trace lives in the process that made the calls (riskpy_status over "
                  "MCP shows the server's); set RISKPY_TRACE_FILE=/path/trace.jsonl to keep calls across processes.")
            return 0
        for e in entries:
            outcome = "ok" if e.get("ok") else f"{e.get('code')} {e.get('param') or ''} {e.get('message') or ''}".strip()
            print(f"{e.get('at', '')[:23]}  {e.get('via', ''):<6} {e.get('tool', ''):<24} {e.get('ms', 0):>9.1f} ms  "
                  f"{outcome}")
        return 0
    if args.json:
        print(json.dumps(manifest(), indent=2))
        return 0
    width = max(len(t.name) for t in TOOLS.values())
    for t in TOOLS.values():
        flag = "" if t.available else f"   (needs {t.install})"
        print(f"{t.name:<{width}}  {t.description}{flag}")
    calculators = sum(t.kind == "calculator" for t in TOOLS.values())
    print(f"\n{calculators} calculators and {len(TOOLS) - calculators} introspection tools. riskpy-tools --json for the "
          "schemas; riskpy-tools call NAME '{...}' to run one; riskpy-explain CODE for an error code.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
