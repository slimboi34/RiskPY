"""The library, described for the coding agent (or person) about to use it.

    riskpy-context              # a Markdown context pack: what RiskPY is, the conventions that
                                # bite, every module's public functions with signatures, the tools
    riskpy-context --json       # the same as data
    riskpy-context -o AGENTS.md # write it

Generated from the installed package — signatures, docstrings, the tool manifest — so it
matches the code that will actually run, rather than a README written earlier.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from typing import Any, Dict, List, Optional

__all__ = ["CONVENTIONS", "describe", "build", "main"]

MODULES = (
    ("mc", "Monte Carlo: any formula, twenty distributions, correlation, a Result that answers VaR/TVaR", "sim"),
    ("quant", "Black–Scholes and Greeks (pure Python), paths, VaR, Heston", ""),
    ("life", "life tables, insurances, annuities, premiums, reserves (pure Python)", ""),
    ("reserving", "chain ladder, Mack, Bornhuetter–Ferguson, Cape Cod, bootstrap", "sim"),
    ("rates", "yield curves, bonds, Vasicek and CIR (curves are pure Python; paths need NumPy)", ""),
    ("credit", "Merton, CDS, Basel IRB, portfolio simulation, transitions (pure Python; simulation needs NumPy)", ""),
    ("viz", "twenty-two charts on one theme", "viz"),
    ("verify", "the verification suite: every identity the library claims, checked", ""),
    ("errors", "the error catalogue: stable codes, the parameter at fault, the fix", ""),
)

CORE_CLASSES = ("MonteCarloSimulator", "LossTriangle", "FactorModel", "ActuarialMath", "ExperienceRating",
                "ExposureRating", "RateAnalyzer", "FourierTransform", "RiskEngine", "ExcelExporter")

CONVENTIONS = [
    "**Rates are decimals.** `i=0.05` is 5%. `i` is an effective annual rate (life, annuities, bonds via "
    "`ytm`); `r` in `quant` and `credit` is continuously compounded. Probability levels are decimals too: "
    "`result.var(0.995)`.",
    "**Times are years.** `T`, `maturity`, `t` and `n` are in years (or whole years for life contingencies); "
    "`frequency` is payments per year.",
    "**`LogNormal(mu, sigma)` takes the underlying normal's parameters, not a mean and standard deviation.** "
    "Given a mean and sd on the money scale, use `LogNormal.from_moments(mean, sd)`. Getting this backwards is the "
    "commonest severity mistake there is; the `aggregate_loss` tool converts for you.",
    "**Seeds make runs reproducible:** `Model.run(trials, seed=42)`, `MonteCarloSimulator(trials, seed)`, "
    "`bootstrap_chain_ladder(tri, n, seed)`. Always pass one when a number will be quoted.",
    "**Results answer questions:** a `Result` has `.mean`, `.var(level)`, `.tvar(level)`, `.prob_above(x)`, "
    "`.standard_error`, `.sensitivity()`, `.describe()` (a dict) and `.summary()` (text). Prefer `.describe()` "
    "when the answer will be passed on as data.",
    "**The compiled core has zero dependencies.** `riskpy.mc`, `riskpy.reserving` and the path simulators need "
    "NumPy (`pip install \"open-riskpy[sim]\"`); `riskpy.viz` needs Matplotlib (`[viz]`). A missing extra is an "
    "`ImportError` that says which one; the tool manifest marks such tools `available: false`.",
    "**Check, don't trust:** `riskpy-verify` runs 111 identities (125 with SciPy) against published tables and "
    "closed forms; the `verify` tool does the same and returns data. On a lean install, modules whose extra is "
    "missing are reported as skipped, not failed.",
    "**Every tool is a wrapper over a verified public function.** If a calculation is not a tool, the module "
    "functions below are one import away and take the same conventions.",
    "**Errors carry a stable code, the parameter and the fix.** Every deliberate error is a `RiskPyError` "
    "(still the `ValueError`/`TypeError`/`KeyError` it always was) with `.code` (`RP105`), `.param` (`sigma`), "
    "`.hint` and `.to_dict()`. `tools.call(name, args, errors=\"return\")`, `riskpy-tools call` and the MCP server "
    "return `{\"error\": {code, message, param, hint}}` instead of a traceback. Fix the named parameter as the hint "
    "says; `riskpy-explain CODE` says more. The tools refuse percentages where decimals belong (`r=5` is RP104, "
    "`sigma=20` is RP105) because the library would happily compute a wrong answer.",
]

ERROR_FAMILIES = [
    ("RP1xx", "input domain — a value outside what the parameter accepts (negative vol RP105, percentage rate RP104, "
              "probability > 1 RP103, time ≤ 0 RP106, mismatched lengths RP109)"),
    ("RP2xx", "missing optional dependency — the hint is the exact pip line (NumPy RP201, Matplotlib RP202, MCP RP204)"),
    ("RP3xx", "numerical — no solution exists (option price below intrinsic RP301, above the ceiling RP302, "
              "root not bracketed RP303)"),
    ("RP4xx", "tool call — unknown tool RP401, unexpected argument RP402, missing argument RP403, wrong JSON type "
              "RP404, over a limit RP405"),
    ("RP5xx", "malformed data — life table RP501, claims triangle RP502, term structure RP503, transition matrix RP504"),
    ("RP9xx", "unexpected internal error — a bug; run riskpy-doctor and report it"),
]


def _signature(fn) -> str:
    try:
        return str(inspect.signature(fn)).replace("'", "")
    except (TypeError, ValueError):
        return "(...)"


def _doc(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    return doc.strip().splitlines()[0] if doc.strip() else ""


def _module_map(name: str) -> dict:
    """Public functions and classes a module defines, with signatures and first doc lines."""
    module = importlib.import_module(f"riskpy.{name}")
    functions, classes = [], []
    for attr, obj in sorted(vars(module).items()):
        if attr.startswith("_") or getattr(obj, "__module__", None) != module.__name__:
            continue
        if inspect.isclass(obj):
            methods = sorted(m for m, v in vars(obj).items()
                             if not m.startswith("_") and (callable(v) or isinstance(v, (property, classmethod, staticmethod))))
            classes.append({"name": attr, "doc": _doc(obj), "methods": methods})
        elif inspect.isfunction(obj):
            functions.append({"name": attr, "signature": _signature(obj), "doc": _doc(obj)})
    return {"doc": _doc(module), "functions": functions, "classes": classes}


def describe() -> dict:
    """The library as data: version, install lines, conventions, modules, tools."""
    import riskpy
    from . import tools

    modules: Dict[str, Any] = {}
    for name, blurb, needs in MODULES:
        entry: Dict[str, Any] = {"blurb": blurb, "needs": needs or None}
        try:
            entry.update(_module_map(name))
            entry["available"] = True
        except ImportError as exc:
            entry.update({"available": False, "error": str(exc).strip().splitlines()[0]})
        modules[name] = entry
    core = []
    for cls_name in CORE_CLASSES:
        cls = getattr(riskpy, cls_name, None)
        if cls is None:
            continue
        core.append({"name": cls_name, "doc": _doc(cls),
                     "methods": sorted(m for m in dir(cls) if not m.startswith("_"))})
    return {
        "version": riskpy.__version__,
        "install": {
            "core": "pip install open-riskpy",
            "sim": 'pip install "open-riskpy[sim]"      # + NumPy: mc, reserving, path simulation',
            "viz": 'pip install "open-riskpy[viz]"      # + Matplotlib: the charts',
            "mcp": 'pip install "open-riskpy[mcp]"      # + the MCP SDK: riskpy-mcp',
        },
        "commands": {
            "riskpy-verify": "run the verification suite (exit 1 if any identity fails)",
            "riskpy-tools": "list the agent tools; `riskpy-tools call NAME '{...}'` runs one; --json for schemas",
            "riskpy-mcp": "serve the tools over MCP on stdio",
            "riskpy-context": "this document; --json for data, -o FILE to write it",
            "riskpy-doctor": "check the installation (core, extras, self-check, MCP); --json; exit 1 on a failure",
            "riskpy-explain": "explain an error code: `riskpy-explain RP105`; `--list [--json]` for the catalogue",
            "riskpy-tools trace": "recent tool calls with durations and error codes (RISKPY_TRACE_FILE keeps them)",
        },
        "errors": [{"family": f, "meaning": m} for f, m in ERROR_FAMILIES],
        "conventions": CONVENTIONS,
        "core": core,
        "modules": modules,
        "tools": tools.manifest(),
    }


def build() -> str:
    """The Markdown context pack."""
    d = describe()
    lines = [
        f"# RiskPY {d['version']} — described for a coding agent",
        "",
        "RiskPY is a risk and actuarial engine for Python: a compiled C++ core with no dependencies, and readable "
        "Python layers for Monte Carlo, life contingencies, claims reserving, yield curves, credit risk, option "
        "pricing and charts. Every identity the library claims is checked against published tables and closed "
        "forms on every push (`riskpy-verify`). Generated by `riskpy-context` from the installed package.",
        "",
        "## Install",
        "",
        "```bash",
        *d["install"].values(),
        "```",
        "",
        "## Commands",
        "",
        *[f"- `{k}` — {v}" for k, v in d["commands"].items()],
        "",
        "## Conventions that bite",
        "",
        *[f"- {c}" for c in d["conventions"]],
        "",
        "## When a call fails",
        "",
        "Every error has a stable code, names the parameter and says how to fix it. Over `tools.call(..., "
        "errors=\"return\")`, `riskpy-tools call` and MCP it arrives as JSON: "
        "`{\"error\": {\"code\": \"RP105\", \"message\": \"sigma must be >= 0, got -0.2\", \"param\": \"sigma\", "
        "\"hint\": \"...\"}}`. Correct the named parameter and call again; `riskpy-explain CODE` (or the "
        "`riskpy_explain` tool) explains a code, `riskpy-doctor` (or `riskpy_doctor`) checks the installation, and "
        "`riskpy-tools trace` (or `riskpy_status`) shows the recent calls.",
        "",
        *[f"- `{e['family']}` {e['meaning']}" for e in d["errors"]],
        "",
        f"## Tools for agents ({sum(t['kind'] == 'calculator' for t in d['tools'])} calculators, "
        f"{sum(t['kind'] != 'calculator' for t in d['tools'])} introspection)",
        "",
        "`from riskpy import tools; tools.call(name, {...})`, `riskpy-tools call NAME '{...}'`, or over MCP with "
        "`riskpy-mcp`. Arguments are plain numbers and strings (rates as decimals, times in years); results are JSON.",
        "",
        "| tool | needs | what it does |",
        "|---|---|---|",
    ]
    for t in d["tools"]:
        needs = "" if not t["requires"] else ("NumPy" if t["available"] else f"NumPy — `{t['install']}`")
        lines.append(f"| `{t['name']}` | {needs} | {t['description']} |")
    lines += ["", "## The compiled core (`import riskpy`)", ""]
    for c in d["core"]:
        lines.append(f"- `{c['name']}` — {c['doc']} · methods: {', '.join(f'`{m}`' for m in c['methods'])}")
    lines += ["", "## Modules", ""]
    for name, m in d["modules"].items():
        needs = f" · needs `open-riskpy[{m['needs']}]`" if m["needs"] else ""
        lines.append(f"### `riskpy.{name}` — {m['blurb']}{needs}")
        lines.append("")
        if not m.get("available"):
            lines.append(f"_Not importable here: {m.get('error', 'missing dependency')}_")
            lines.append("")
            continue
        for f in m["functions"]:
            doc = f" — {f['doc']}" if f["doc"] else ""
            lines.append(f"- `{f['name']}{f['signature']}`{doc}")
        for c in m["classes"]:
            doc = f" — {c['doc']}" if c["doc"] else ""
            methods = f" · {', '.join(f'`{x}`' for x in c['methods'][:14])}" if c["methods"] else ""
            lines.append(f"- class `{c['name']}`{doc}{methods}")
        lines.append("")
    lines += [
        "## Verification",
        "",
        "`riskpy-verify` computes every identity and compares it with a source the library had no hand in: "
        "put–call parity, `A_x = 1 − d·ä_x`, Mack's standard error on the GenIns triangle, the Basel risk-weight "
        "table, `Φ⁻¹(Φ(x)) = x`, the contrast ratio of every palette colour. Run it after installing, and quote "
        "its result rather than assuming. Docs: https://slimboi34.github.io/RiskPY/",
    ]
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="riskpy-context",
                                     description="Describe RiskPY for an AI coding tool: conventions, modules, tools.")
    parser.add_argument("--json", action="store_true", help="the description as JSON instead of Markdown")
    parser.add_argument("-o", "--output", metavar="FILE", help="write to FILE instead of stdout")
    args = parser.parse_args(argv)
    text = json.dumps(describe(), indent=2) if args.json else build()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text.rstrip("\n") + "\n")
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
