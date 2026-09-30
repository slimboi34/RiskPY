"""RiskPY's verified calculators, as tools for an AI agent.

An agent that can call functions needs three things from a library: a list of what it may
call, a schema for each call, and results it can pass on as data. This module provides all
three, generated from the calculators' own signatures and docstrings:

    from riskpy import tools
    tools.manifest()      # [{name, description, input_schema, requires, available}, ...]
    tools.call("black_scholes", {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2})   # 10.4506

    riskpy-tools                                   # the list, on the command line
    riskpy-tools call black_scholes '{"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2}'
    riskpy-mcp                                     # the same tools over MCP (pip install "open-riskpy[mcp]")

Every tool is a thin wrapper over a public function that the verification suite checks
(``riskpy-verify``). Tools take plain numbers, strings and lists, return JSON-able values,
and hide the conventions that bite (an aggregate-loss tool takes a severity mean and
standard deviation on the money scale, never a lognormal ``mu``). Tools that need NumPy
say so in ``requires`` and are reported as *unavailable*, not missing, on a lean install.
"""
from __future__ import annotations

import argparse
import collections.abc
import dataclasses
import inspect
import json
import math
import re
import sys
import types
import typing
from typing import Any, Callable, Dict, List, Optional

__all__ = ["Tool", "TOOLS", "ToolUnavailable", "tool", "schema", "manifest", "call", "main"]

_EXTRA_MODULE = {"sim": "numpy", "viz": "matplotlib"}
_EXTRA_NAME = {"sim": "NumPy", "viz": "Matplotlib"}


class ToolUnavailable(RuntimeError):
    """The tool exists, but the optional dependency it needs is not installed."""


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

    @property
    def available(self) -> bool:
        return not self.requires or _extra_installed(self.requires)

    @property
    def install(self) -> Optional[str]:
        return f'pip install "open-riskpy[{self.requires}]"' if self.requires else None

    def schema(self) -> dict:
        return schema(self.fn)


TOOLS: Dict[str, Tool] = {}


def tool(name: Optional[str] = None, *, requires: str = ""):
    """Register a function as a tool. The first docstring paragraph is its description."""

    def register(fn):
        doc = inspect.getdoc(fn) or ""
        description = " ".join(doc.split("\n\n")[0].split())
        t = Tool(name or fn.__name__, fn, description, requires)
        TOOLS[t.name] = t
        return fn

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
        return None if math.isnan(value) else value
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
# The public surface


def manifest() -> List[dict]:
    """Every tool as data: name, description, input schema, what it needs, whether it can run."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.schema(), "requires": t.requires,
         "available": t.available, "install": t.install}
        for t in TOOLS.values()
    ]


def call(name: str, arguments: Optional[dict] = None) -> Any:
    """Run a tool by name with keyword arguments; the result is JSON-able."""
    try:
        t = TOOLS[name]
    except KeyError:
        raise KeyError(f"unknown tool {name!r}; `riskpy-tools` lists them") from None
    if not t.available:
        raise ToolUnavailable(f"{name} needs {_EXTRA_NAME.get(t.requires, t.requires)}: {t.install}")
    args = dict(arguments or {})
    accepted = set(inspect.signature(t.fn).parameters)
    unknown = sorted(set(args) - accepted)
    if unknown:
        raise TypeError(f"{name}: unexpected argument(s) {unknown}; accepted: {sorted(accepted)}")
    return _jsonable(t.fn(**args))


# ---------------------------------------------------------------------------
# The tools. Plain numbers in, JSON out, verified functions underneath.


@tool()
def black_scholes(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, kind: str = "call") -> float:
    """Black–Scholes–Merton price of a European option.

    S: spot price of the underlying
    K: strike price
    T: time to expiry, in years
    r: continuously compounded risk-free rate as a decimal (0.05 means 5%)
    sigma: annual volatility as a decimal (0.2 means 20%)
    q: continuous dividend yield as a decimal
    kind: "call" or "put"
    """
    from . import quant

    return float(quant.black_scholes(S, K, T, r, sigma, q=q, kind=kind))


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


@tool()
def bond_analytics(face: float, coupon: float, maturity: float, ytm: float, frequency: int = 2) -> dict:
    """Price, durations, convexity and DV01 of a fixed-coupon bond at a given yield.

    face: face (par) value
    coupon: annual coupon rate as a decimal (0.05 means 5%)
    maturity: years to maturity
    ytm: yield to maturity as a decimal, compounded `frequency` times a year
    frequency: coupons per year
    """
    from . import rates

    bond = rates.Bond(face=face, coupon=coupon, maturity=maturity, frequency=frequency)
    return {
        "price": bond.price(ytm), "macaulay_duration": bond.macaulay_duration(ytm),
        "modified_duration": bond.modified_duration(ytm), "convexity": bond.convexity(ytm), "dv01": bond.dv01(ytm),
    }


@tool()
def basel_irb_capital(pd: float, lgd: float, ead: float, maturity: float = 2.5, asset_class: str = "corporate") -> dict:
    """Basel II/III internal-ratings-based capital for one exposure: risk weight, RWA, capital.

    pd: one-year probability of default as a decimal (0.01 means 1%)
    lgd: loss given default as a decimal
    ead: exposure at default, in money
    maturity: effective maturity in years
    asset_class: "corporate", "sovereign", "bank", "mortgage", "revolving" or "retail"
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

    if name != "sult":
        raise ValueError('table must be "sult" (the AMLCR Standard Ultimate Life Table)')
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
    seed: random seed, so the answer is reproducible
    """
    from . import MonteCarloSimulator

    if severity_mean <= 0 or severity_sd < 0:
        raise ValueError("severity_mean must be positive and severity_sd non-negative")
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

    report = _verify.run(modules, oracle=oracle)
    return {
        "ok": report.ok, "version": report.version, "checks": len(report.checks), "passed": len(report.passed),
        "failed": len(report.failed), "seconds": round(report.seconds, 3),
        "skipped": [{"module": m, "reason": r.strip().splitlines()[0]} for m, r in report.skipped],
        "failures": [{"module": c.module, "name": c.name, "value": c.value, "reference": c.reference,
                      "tolerance": c.tolerance} for c in report.failed],
    }


# ---------------------------------------------------------------------------
# CLI


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="riskpy-tools",
                                     description="RiskPY's verified calculators as tools for an AI agent.")
    parser.add_argument("--json", action="store_true", help="print the manifest as JSON")
    sub = parser.add_subparsers(dest="command")
    p_call = sub.add_parser("call", help="run a tool: riskpy-tools call NAME '{\"arg\": value}'")
    p_call.add_argument("name")
    p_call.add_argument("arguments", nargs="?", default="{}", help="the arguments as a JSON object")
    p_schema = sub.add_parser("schema", help="print one tool's input schema")
    p_schema.add_argument("name")
    args = parser.parse_args(argv)

    try:
        if args.command == "call":
            print(json.dumps(call(args.name, json.loads(args.arguments)), indent=2))
        elif args.command == "schema":
            print(json.dumps(TOOLS[args.name].schema(), indent=2))
        elif args.json:
            print(json.dumps(manifest(), indent=2))
        else:
            width = max(len(t.name) for t in TOOLS.values())
            for t in TOOLS.values():
                flag = "" if t.available else f"   (needs {t.install})"
                print(f"{t.name:<{width}}  {t.description}{flag}")
            print(f"\n{len(TOOLS)} tools. riskpy-tools --json for the schemas; riskpy-tools call NAME '{{...}}' to run one.")
        return 0
    except (KeyError, TypeError, ValueError, ToolUnavailable) as exc:
        print(f"riskpy-tools: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
