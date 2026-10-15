"""RiskPY's errors: stable codes, the parameter at fault, and what to do about it.

Every error RiskPY raises on purpose carries a **code** that does not change between
releases, the **parameter** it is about (when there is one), and a **hint** saying how to
fix the call. The exceptions still subclass the built-in types the library always raised —
a bad volatility is still a ``ValueError`` — so ``except ValueError`` keeps working::

    from riskpy import quant
    from riskpy.errors import RiskPyError

    try:
        quant.black_scholes(S=100, K=100, T=1, r=0.05, sigma=-0.2)
    except RiskPyError as exc:
        exc.code      # "RP105"
        exc.param     # "sigma"
        exc.hint      # "Volatility is an annual decimal: 0.2 means 20%. ..."
        exc.to_dict() # {"code": ..., "message": ..., "param": ..., "hint": ...}

    riskpy-explain RP105          # what the code means and how to fix it
    riskpy-explain --list --json  # the whole catalogue as data

Codes come in families:

======  ==================================================================
RP1xx   an input outside its domain (a negative volatility, a percentage
        where a decimal belongs, a maturity of zero, mismatched lengths)
RP2xx   an optional dependency that is not installed (NumPy, Matplotlib,
        pandas, the MCP SDK) or a compiled core that will not load
RP3xx   a numerical problem: no solution exists, or a solver cannot
        bracket or reach one (an option price outside no-arbitrage bounds)
RP4xx   a tool call that is wrong as a call: unknown tool, unknown or
        missing argument, wrong JSON type, a size over ``tools.LIMITS``
RP5xx   data that is malformed as a whole: a life table, a claims
        triangle, a yield curve's knots, a transition matrix, a file
RP9xx   something RiskPY did not expect — a bug worth reporting
======  ==================================================================

A code is never reused for a different meaning; a retired code stays in the catalogue.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Dict, List, Optional

__all__ = [
    "RiskPyError", "InputError", "ArgumentTypeError", "UnknownNameError", "ConvergenceError",
    "MissingDependencyError", "CATALOGUE", "explain", "catalogue", "from_exception", "main",
]

DOCS = "https://slimboi34.github.io/RiskPY/errors/"

FAMILIES = {
    "RP1": "input domain",
    "RP2": "missing dependency",
    "RP3": "numerical",
    "RP4": "tool call",
    "RP5": "data",
    "RP9": "internal",
}

# code -> (title, what it means, how to fix it). The hint is the default an error carries
# when the raise site has nothing more specific to say.
CATALOGUE: Dict[str, Dict[str, str]] = {
    # -- RP1xx: input domain ------------------------------------------------------------
    "RP100": {
        "title": "Invalid input",
        "meaning": "An argument is outside what the function accepts. The message names it and the allowed range.",
        "hint": "Read the message: it names the parameter and the values it accepts.",
    },
    "RP101": {
        "title": "Not a finite number",
        "meaning": "A numeric argument is NaN, infinite, or not a number at all (a string, None).",
        "hint": "Pass a finite number. JSON NaN/Infinity and numeric strings such as \"100\" are refused, not coerced.",
    },
    "RP102": {
        "title": "Must be positive (or non-negative)",
        "meaning": "A quantity that cannot be negative or zero (a price, a strike, a face value, a mean) was.",
        "hint": "Check the sign and units; prices, strikes, exposures and sizes are positive amounts of money.",
    },
    "RP103": {
        "title": "Probability outside [0, 1]",
        "meaning": "A probability (PD, LGD, recovery, a weight) is below 0 or above 1. Above 1 is almost always a "
                   "percentage passed where a decimal belongs.",
        "hint": "Probabilities are decimals: 1% is 0.01, 45% is 0.45.",
    },
    "RP104": {
        "title": "Rate looks like a percentage",
        "meaning": "An interest rate, yield, coupon, dividend yield or spread is larger than 1 (100%) in absolute "
                   "value, or an expected loss ratio larger than 5 (500%). RiskPY takes rates and ratios as "
                   "decimals, so 5 means 500%, not 5%.",
        "hint": "Divide by 100: a 5% rate is 0.05.",
    },
    "RP105": {
        "title": "Volatility out of range",
        "meaning": "A volatility is negative, or so large (above 5, i.e. 500% a year) that it is almost certainly "
                   "a percentage passed where a decimal belongs.",
        "hint": "Volatility is an annual decimal: 0.2 means 20%. It must be >= 0.",
    },
    "RP106": {
        "title": "Time or maturity out of range",
        "meaning": "A time to expiry, maturity, duration or horizon is negative, zero where a positive time is "
                   "needed, or past the end of the contract.",
        "hint": "Times are in years (6 months is 0.5). Implied volatility and similar inversions need T > 0.",
    },
    "RP107": {
        "title": "Whole number required",
        "meaning": "An age, term, count or payment frequency must be a whole number (and usually at least 1); "
                   "a fraction or a boolean was given.",
        "hint": "Pass an integer: ages and terms are whole years, frequency is payments per year (1, 2, 4, 12).",
    },
    "RP108": {
        "title": "Unknown choice",
        "meaning": "A string option is not one of the accepted values (option kind, asset class, contract kind, "
                   "compounding convention, life table).",
        "hint": "Use one of the values the message lists; they are case-insensitive where noted.",
    },
    "RP109": {
        "title": "Lengths do not match",
        "meaning": "Sequences that pair up element by element (times and discount factors, par rates and "
                   "maturities, PDs, LGDs and EADs, premiums and origin periods) have different lengths.",
        "hint": "Give every list one entry per item; the message says which lengths differ.",
    },
    "RP110": {
        "title": "Confidence level outside (0, 1)",
        "meaning": "A quantile or confidence level must lie strictly between 0 and 1.",
        "hint": "Levels are decimals: the 99.5th percentile is 0.995, not 99.5.",
    },
    "RP111": {
        "title": "Empty input",
        "meaning": "A list that needs at least one (sometimes two) entries is empty.",
        "hint": "Pass the data; the message says the minimum length.",
    },
    "RP112": {
        "title": "Not in increasing order",
        "meaning": "Times, maturities or ages must be strictly increasing and were not.",
        "hint": "Sort the knots by time and remove duplicates.",
    },
    "RP113": {
        "title": "Inconsistent arguments",
        "meaning": "Arguments that are each valid do not fit together (a term given for a whole-life contract, "
                   "a valuation time after expiry, t_1 >= t_2, an age outside the table).",
        "hint": "Read the message: it names both arguments and the relation they must satisfy.",
    },
    # -- RP2xx: missing dependency ------------------------------------------------------
    "RP201": {
        "title": "NumPy not installed",
        "meaning": "The simulation layer (riskpy.mc, riskpy.reserving, path simulators, portfolio simulation, "
                   "the chain-ladder tools) needs NumPy, which is an optional extra.",
        "hint": 'pip install "open-riskpy[sim]"',
    },
    "RP202": {
        "title": "Matplotlib not installed",
        "meaning": "riskpy.viz draws with Matplotlib, which is an optional extra.",
        "hint": 'pip install "open-riskpy[viz]"',
    },
    "RP203": {
        "title": "pandas not installed",
        "meaning": "The .to_frame() methods return pandas DataFrames; pandas is not a dependency.",
        "hint": "pip install pandas",
    },
    "RP204": {
        "title": "MCP SDK not installed",
        "meaning": "riskpy-mcp serves the tools over the Model Context Protocol and needs the MCP SDK (Python >= 3.10).",
        "hint": 'pip install "open-riskpy[mcp]"',
    },
    "RP205": {
        "title": "Compiled core not importable",
        "meaning": "The C++ extension cpp_underwriter did not load: usually a source checkout without a build, or "
                   "a wheel for a different Python or platform.",
        "hint": "pip install --force-reinstall open-riskpy  (or build a checkout with `pip install -e .`); "
                "run riskpy-doctor for details.",
    },
    "RP206": {
        "title": "Tkinter not available",
        "meaning": "The optional desktop UnderwritingApp needs Tkinter, which some Python builds omit.",
        "hint": "Install a Python with Tk support, or use the headless APIs.",
    },
    # -- RP3xx: numerical ---------------------------------------------------------------
    "RP301": {
        "title": "Price below the no-arbitrage floor",
        "meaning": "No volatility (or spread, or yield) reproduces the price: it is below the option's discounted "
                   "intrinsic value. The quote is stale or the inputs (S, K, T, r, q, kind) are wrong.",
        "hint": "Check the price and that kind matches the quote (a put priced as a call is the usual slip).",
    },
    "RP302": {
        "title": "Price above the no-arbitrage ceiling",
        "meaning": "The price exceeds what any volatility can produce (a call is worth at most S·e^(−qT), a put "
                   "at most K·e^(−rT)), or what the solver's widest bracket reaches.",
        "hint": "Check the price units (per share, not per contract of 100) and the spot and strike.",
    },
    "RP303": {
        "title": "Root not bracketed",
        "meaning": "A solver (yield to maturity, z-spread, implied hazard) found no sign change on its search "
                   "interval, so no solution exists in the range it searched.",
        "hint": "Check that the target price is attainable for the instrument; extreme prices have no solution.",
    },
    "RP304": {
        "title": "Numerical overflow",
        "meaning": "An intermediate result overflowed floating point (for example (1 + i)^n with a large n and a "
                   "rate near -1).",
        "hint": "Use inputs of a realistic size; the message names the calculation that overflowed.",
    },
    "RP305": {
        "title": "Non-finite result",
        "meaning": "A calculation produced NaN or infinity: a simulation formula divided by a variable that reached "
                   "zero, or a resampled triangle had a degenerate column.",
        "hint": "Look for a division by something that can be zero, or a log of something that can go negative.",
    },
    # -- RP4xx: tool calls --------------------------------------------------------------
    "RP401": {
        "title": "Unknown tool",
        "meaning": "No tool has that name.",
        "hint": "List the tools with `riskpy-tools` (or tools.manifest(), or the MCP tools/list request).",
    },
    "RP402": {
        "title": "Unexpected argument",
        "meaning": "The call passed an argument the tool does not take (often a misspelling, or a name from "
                   "another library such as `vol` for `sigma`).",
        "hint": "Use only the argument names in the tool's input_schema: `riskpy-tools schema NAME`.",
    },
    "RP403": {
        "title": "Missing required argument",
        "meaning": "A required argument was not passed.",
        "hint": "Pass every name in the schema's `required` list: `riskpy-tools schema NAME`.",
    },
    "RP404": {
        "title": "Wrong argument type",
        "meaning": "An argument has the wrong JSON type: a string where a number belongs, true/false where a "
                   "number belongs, a scalar where a list belongs.",
        "hint": "Send JSON numbers as numbers (100, not \"100\"), booleans as true/false, lists as arrays.",
    },
    "RP405": {
        "title": "Over a tool limit",
        "meaning": "The call asks for more work than one tool call may (trials, simulated claims, payment periods, "
                   "triangle size, list or string length, integer size). The limits are in riskpy.tools.LIMITS.",
        "hint": "Lower the size named in the message, or call the library function directly, which is unbounded.",
    },
    "RP406": {
        "title": "Arguments are not a JSON object",
        "meaning": "The arguments for a tool call could not be parsed as a JSON object.",
        "hint": "Pass a JSON object, e.g. riskpy-tools call black_scholes '{\"S\": 100, \"K\": 100, \"T\": 1, "
                "\"r\": 0.05, \"sigma\": 0.2}'.",
    },
    "RP407": {
        "title": "Unknown error code",
        "meaning": "riskpy-explain was asked about a code that is not in the catalogue.",
        "hint": "riskpy-explain --list shows every code.",
    },
    # -- RP5xx: data --------------------------------------------------------------------
    "RP501": {
        "title": "Invalid life table",
        "meaning": "Mortality data is malformed: q_x outside [0, 1], l_x negative or increasing, too few ages, "
                   "or a final age whose q is not 1.",
        "hint": "Check the table's columns and that the last q_x is 1 (everyone dies by the end of the table).",
    },
    "RP502": {
        "title": "Invalid claims triangle",
        "meaning": "A claims triangle is malformed: ragged rows, too few origin or development periods, "
                   "non-finite cumulative amounts, or no observed link ratios.",
        "hint": "Rows are origin periods, columns development periods, cumulative amounts, null where unobserved.",
    },
    "RP503": {
        "title": "Invalid term structure",
        "meaning": "A yield curve, hazard curve or discount-factor set is malformed: no knots, non-positive "
                   "discount factors, or (time, rate) pairs that are not pairs.",
        "hint": "Give strictly increasing positive times with one rate or discount factor each.",
    },
    "RP504": {
        "title": "Invalid transition matrix",
        "meaning": "A rating transition matrix is not square, has rows that do not sum to 1, negative entries, "
                   "duplicate ratings, or a default state that is not absorbing.",
        "hint": "Each row is a probability distribution over the next rating: non-negative, summing to 1.",
    },
    "RP505": {
        "title": "File or I/O error",
        "meaning": "A file could not be read or written (an export, a context pack, a trace file).",
        "hint": "Check the path exists and is writable.",
    },
    # -- RP9xx: internal ----------------------------------------------------------------
    "RP900": {
        "title": "Unexpected internal error",
        "meaning": "RiskPY raised something it did not expect. The call may have been valid; this is a bug.",
        "hint": "Run riskpy-doctor, then report the call and the message at https://github.com/slimboi34/RiskPY/issues.",
    },
}


_MISSING = object()


class RiskPyError(Exception):
    """Base class: every deliberate RiskPY error has a stable ``code``, a ``param`` and a ``hint``.

    Raise one of the subclasses below, which also derive from the built-in type the
    library has always raised for that situation, so existing ``except`` clauses still
    match.
    """

    default_code = "RP100"

    def __init__(self, message: str, *, code: Optional[str] = None, param: Optional[str] = None,
                 hint: Optional[str] = None, value: Any = _MISSING):
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        if self.code not in CATALOGUE:  # pragma: no cover - guarded by a test over every raise site
            raise AssertionError(f"error code {self.code} is not in the catalogue")
        self.param = param
        self.hint = hint if hint is not None else CATALOGUE[self.code]["hint"]
        self.value = value

    def __str__(self) -> str:
        return self.message

    def __reduce__(self):  # keep pickling (multiprocessing) working with keyword-only arguments
        return (_rebuild, (type(self), self.message, self.code, self.param, self.hint))

    def to_dict(self) -> Dict[str, Any]:
        """``{code, message, param, hint, docs}`` — what an agent needs to correct the call."""
        return {"code": self.code, "message": self.message, "param": self.param, "hint": self.hint,
                "docs": f"{DOCS}#{self.code.lower()}"}


def _rebuild(cls, message, code, param, hint):
    return cls(message, code=code, param=param, hint=hint)


class InputError(RiskPyError, ValueError):
    """An argument outside its domain (RP1xx), or malformed data (RP5xx). A ``ValueError``."""


class ArgumentTypeError(RiskPyError, TypeError):
    """An argument of the wrong type, or an unexpected / missing one. A ``TypeError``."""

    default_code = "RP404"


class UnknownNameError(RiskPyError, KeyError):
    """A name (a tool, an error code) that does not exist. A ``KeyError``."""

    default_code = "RP401"


class ConvergenceError(RiskPyError, ValueError):
    """No solution exists, or a solver could not bracket one (RP3xx). A ``ValueError``."""

    default_code = "RP303"


class MissingDependencyError(RiskPyError, ImportError):
    """An optional dependency is not installed (RP2xx). An ``ImportError``."""

    default_code = "RP201"


# ---------------------------------------------------------------------------
# Turning any exception into the structured form


_PARAM_IN_MESSAGE = re.compile(r"^(?:\w+: )?([A-Za-z_][\w]*(?:\[\d+\])*) (?:must|is|has|needs|should|=)\b")


def from_exception(exc: BaseException) -> Dict[str, Any]:
    """The ``{code, message, param, hint, docs}`` dict for any exception, never a traceback.

    A :class:`RiskPyError` reports itself. Anything else is classified by type — a plain
    ``ValueError`` is RP100 with the parameter read from the message where it starts with
    one ("sigma must be ..."), an ``OverflowError`` RP304, anything unexpected RP900.
    """
    if isinstance(exc, RiskPyError):
        return exc.to_dict()
    message = str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)
    message = message or type(exc).__name__
    if isinstance(exc, ImportError):
        code = "RP201"
        low = message.lower()
        for needle, c in (("matplotlib", "RP202"), ("pandas", "RP203"), ("mcp", "RP204"), ("cpp_underwriter", "RP205"),
                          ("tkinter", "RP206")):
            if needle in low:
                code = c
    elif isinstance(exc, (OverflowError, ZeroDivisionError, ArithmeticError)):
        code = "RP304"
    elif isinstance(exc, ValueError):
        code = "RP100"
    elif isinstance(exc, TypeError):
        code = "RP404"
    elif isinstance(exc, OSError):
        code = "RP505"
    else:
        code = "RP900"
        message = f"{type(exc).__name__}: {message}"
    m = _PARAM_IN_MESSAGE.match(message)
    return {"code": code, "message": message, "param": m.group(1) if m else None, "hint": CATALOGUE[code]["hint"],
            "docs": f"{DOCS}#{code.lower()}"}


# ---------------------------------------------------------------------------
# The catalogue, for people and agents


def explain(code: str) -> Dict[str, str]:
    """What an error code means and how to fix it. Raises ``UnknownNameError`` (RP407) for an unknown code."""
    key = str(code).strip().upper()
    if key not in CATALOGUE:
        raise UnknownNameError(f"unknown error code {code!r}; riskpy-explain --list shows them all", code="RP407",
                               param="code")
    entry = CATALOGUE[key]
    return {"code": key, "family": FAMILIES[key[:3]], **entry, "docs": f"{DOCS}#{key.lower()}"}


def catalogue() -> List[Dict[str, str]]:
    """Every code, in order, as data."""
    return [explain(code) for code in sorted(CATALOGUE)]


def _safe_streams() -> None:
    """Let the CLIs print σ, — and → on a console whose code page cannot (Windows cp1252, pipes):
    unencodable characters become "?" instead of raising UnicodeEncodeError mid-report."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover - replaced or detached streams
            pass


def main(argv: Optional[List[str]] = None) -> int:
    _safe_streams()
    parser = argparse.ArgumentParser(prog="riskpy-explain",
                                     description="Explain a RiskPY error code (RP1xx input, RP2xx missing dependency, "
                                                 "RP3xx numerical, RP4xx tool call, RP5xx data, RP9xx internal).")
    parser.add_argument("code", nargs="?", help="the code, e.g. RP105")
    parser.add_argument("--list", action="store_true", help="list every code")
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args(argv)
    if args.list or not args.code:
        entries = catalogue()
        if args.json:
            print(json.dumps(entries, indent=2, ensure_ascii=False))
        else:
            family = None
            for e in entries:
                if e["family"] != family:
                    family = e["family"]
                    print(f"\n{e['code'][:3]}xx  {family}")
                print(f"  {e['code']}  {e['title']}")
            print("\nriskpy-explain CODE for the meaning and the fix.")
        return 0
    try:
        e = explain(args.code)
    except UnknownNameError as exc:
        if args.json:
            print(json.dumps({"error": exc.to_dict()}, indent=2))
        print(f"riskpy-explain: {exc.code} {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(e, indent=2, ensure_ascii=False))
    else:
        print(f"{e['code']}  {e['title']}  ({e['family']})\n\n{e['meaning']}\n\nFix: {e['hint']}\n\nDocs: {e['docs']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
