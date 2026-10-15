"""``riskpy-doctor``: is this installation healthy, and if not, what exactly to do.

    riskpy-doctor            # a table: every check, ok / warn / fail, and the fix
    riskpy-doctor --json     # the same as data, for an agent or a CI step

    import riskpy
    riskpy.doctor()          # {"ok": True, "version": "0.5.0", "checks": [...]}

Each check is ``{id, status, message, hint}`` with ``status`` one of ``ok``, ``warn`` (works,
but something optional is missing or off) and ``fail`` (something RiskPY needs is broken).
The command exits 1 if any check fails, so it can gate a deployment or a CI job.

What it checks: the Python version; the installed distribution and the version the package
reports; that the compiled C++ core imports, comes from the same installation as the Python
layer, and answers a known question correctly; each optional extra (NumPy, SciPy, pandas,
Matplotlib, Tkinter, the MCP SDK) with the exact ``pip`` line when it is missing; a quick
numerical self-check (a subset of ``riskpy-verify``); that the agent tools load; that the
MCP server builds and its command is on ``PATH``; and the platform.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
import time
from typing import Any, Dict, List, Optional

__all__ = ["doctor", "checks", "main"]

OK, WARN, FAIL = "ok", "warn", "fail"

# (import name, what needs it, pip line, extra name)
EXTRAS = (
    ("numpy", "simulation (riskpy.mc, reserving, path simulators) and the triangle / VaR tools",
     'pip install "open-riskpy[sim]"', "sim"),
    ("scipy", "the verification suite's independent oracle for the special functions (optional)",
     "pip install scipy", None),
    ("pandas", "the .to_frame() methods", "pip install pandas", None),
    ("matplotlib", "riskpy.viz, the charts", 'pip install "open-riskpy[viz]"', "viz"),
    ("mcp", "riskpy-mcp, the tools over the Model Context Protocol (Python >= 3.10)",
     'pip install "open-riskpy[mcp]"', "mcp"),
)


def _check(id_: str, status: str, message: str, hint: str = "", **extra: Any) -> Dict[str, Any]:
    out = {"id": id_, "status": status, "message": message, "hint": hint}
    out.update(extra)
    return out


def _python() -> Dict[str, Any]:
    v = sys.version_info
    text = f"Python {platform.python_version()} ({platform.python_implementation()}) at {sys.executable}"
    if (v.major, v.minor) < (3, 10):
        return _check("python", FAIL, text, "RiskPY 0.5 needs Python 3.10 or later; wheels exist for 3.10–3.14.")
    if (v.major, v.minor) > (3, 14):
        return _check("python", WARN, text, "Newer than the wheels RiskPY ships (3.10–3.14); a source build may be "
                                            "needed. Report problems on GitHub.")
    return _check("python", OK, text)


def _version() -> Dict[str, Any]:
    import riskpy

    try:
        from importlib.metadata import PackageNotFoundError, distribution
        dist = distribution("open-riskpy")
        installed = dist.version
    except PackageNotFoundError:
        return _check("riskpy_version", WARN, f"riskpy {riskpy.__version__} imported from {os.path.dirname(riskpy.__file__)}, "
                      "but no open-riskpy distribution is installed",
                      "This looks like a source checkout on sys.path. pip install open-riskpy (or pip install -e .) "
                      "so the version and the compiled core are the ones you think.", version=riskpy.__version__)
    location = os.path.dirname(os.path.abspath(riskpy.__file__))
    message = f"open-riskpy {installed} (riskpy at {location})"
    try:
        files = {os.path.normcase(os.path.abspath(dist.locate_file(f))) for f in (dist.files or [])
                 if str(f).endswith("__init__.py") and str(f).startswith("riskpy")}
    except Exception:  # pragma: no cover - unusual installers
        files = set()
    here = os.path.normcase(os.path.abspath(riskpy.__file__))
    if files and here not in files:
        return _check("riskpy_version", WARN, message + " — but the riskpy being imported is not the installed one",
                      "A directory named riskpy (often a source checkout) shadows the installed package. Run from "
                      "another directory, or reinstall.", version=installed)
    if installed != riskpy.__version__:
        return _check("riskpy_version", WARN, message + f", but riskpy.__version__ says {riskpy.__version__}",
                      "Reinstall: pip install --force-reinstall open-riskpy", version=installed)
    return _check("riskpy_version", OK, message, version=installed)


def _core() -> List[Dict[str, Any]]:
    out = []
    try:
        core = importlib.import_module("riskpy.cpp_underwriter")
    except ImportError as exc:  # pragma: no cover - environment dependent
        return [_check("core_import", FAIL, f"the compiled core riskpy.cpp_underwriter does not import: {exc}",
                       "RP205 — pip install --force-reinstall open-riskpy; a source checkout needs `pip install -e .` "
                       "and a C++17 compiler.", code="RP205")]
    import riskpy

    path = getattr(core, "__file__", "?")
    same_dir = os.path.dirname(os.path.abspath(path)) == os.path.dirname(os.path.abspath(riskpy.__file__))
    core_version = getattr(core, "__version__", None)
    message = f"compiled core loaded from {path}"
    if not same_dir:
        out.append(_check("core_import", WARN, message + ", outside the riskpy package being imported",
                          "The extension and the Python layer come from different installations; reinstall."))
    elif core_version and core_version != riskpy.__version__:
        out.append(_check("core_import", WARN, message + f", version {core_version} vs riskpy {riskpy.__version__}",
                          "Reinstall so the core and the Python layer match: pip install --force-reinstall open-riskpy"))
    else:
        out.append(_check("core_import", OK, message + (f" (version {core_version})" if core_version else "")))

    # A known answer: severity fixed at 1 makes the aggregate a Poisson count, whose mean is
    # the frequency to within a few standard errors; then the 0.4.1 hardening (a NaN parameter refused).
    try:
        sim = core.MonteCarloSimulator(trials=20_000, seed=11)
        losses = sim.simulate_aggregate_loss(3.0, 0.0, 1e-12)
        mean = sum(losses) / len(losses)
        se = math.sqrt(3.0 / len(losses))
        if abs(mean - 3.0) > 5 * se:
            out.append(_check("core_compute", FAIL, f"the core's Poisson mean is {mean:.4f}, expected 3 ± {5 * se:.4f}",
                              "The compiled core computes wrong numbers: reinstall, then run riskpy-verify -m core."))
        else:
            out.append(_check("core_compute", OK, f"the core simulates correctly (Poisson mean {mean:.4f} ≈ 3)"))
        try:
            # NaN in the severity, never the frequency: a pre-0.4.1 core loops forever on a NaN frequency.
            core.MonteCarloSimulator(trials=4, seed=1).simulate_aggregate_loss(1.0, float("nan"), 1.0)
            hardened = False
        except ValueError:
            hardened = True
        if not hardened:
            out.append(_check("core_hardening", WARN, "the compiled core accepts NaN parameters (it predates 0.4.1)",
                              "The extension is older than the Python layer: pip install --force-reinstall open-riskpy"))
        else:
            out.append(_check("core_hardening", OK, "the core refuses non-finite parameters (0.4.1+ build)"))
    except Exception as exc:  # pragma: no cover - a broken build
        out.append(_check("core_compute", FAIL, f"the compiled core raised {type(exc).__name__}: {exc}",
                          "Reinstall, then run riskpy-verify -m core."))
    return out


def _extras() -> List[Dict[str, Any]]:
    out = []
    for module, purpose, pip, extra in EXTRAS:
        spec = importlib.util.find_spec(module)
        if spec is None:
            status = WARN
            out.append(_check(f"extra_{module}", status, f"{module} is not installed — needed for {purpose}", pip,
                              extra=extra))
            continue
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", None)
            if version is None:
                from importlib.metadata import version as _v
                try:
                    version = _v(module)
                except Exception:
                    version = "?"
            out.append(_check(f"extra_{module}", OK, f"{module} {version}", extra=extra))
        except Exception as exc:  # installed but broken
            out.append(_check(f"extra_{module}", WARN, f"{module} is installed but does not import: {exc}",
                              f"Reinstall it: pip install --force-reinstall {module}", extra=extra))
    try:
        tk_spec = importlib.util.find_spec("tkinter")
    except (ImportError, ValueError):  # pragma: no cover
        tk_spec = None
    if tk_spec is None:
        out.append(_check("extra_tkinter", WARN, "Tkinter is not available — only the optional desktop "
                          "UnderwritingApp needs it", "Use a Python build with Tk support (RP206)."))
    else:
        out.append(_check("extra_tkinter", OK, "Tkinter is available (for the optional UnderwritingApp)"))
    return out


def _selfcheck() -> Dict[str, Any]:
    from . import verify

    t0 = time.perf_counter()
    try:
        report = verify.run(["core", "special", "quant", "life", "rates", "credit"], oracle=False)
    except Exception as exc:  # pragma: no cover
        return _check("selfcheck", FAIL, f"the verification subset raised {type(exc).__name__}: {exc}",
                      "Run riskpy-verify for the full report.")
    seconds = time.perf_counter() - t0
    if not report.ok:
        names = "; ".join(f"{c.module}: {c.name}" for c in report.failed[:3])
        return _check("selfcheck", FAIL, f"{len(report.failed)} of {len(report.checks)} identities fail ({names})",
                      "Run riskpy-verify -v for every value and reference, and report it on GitHub.",
                      checks=len(report.checks), failed=len(report.failed))
    skipped = f", {len(report.skipped)} module(s) skipped" if report.skipped else ""
    return _check("selfcheck", OK, f"{len(report.checks)} identities verified in {seconds:.2f}s "
                  f"(core, special, quant, life, rates, credit){skipped}", checks=len(report.checks), failed=0)


def _tools() -> Dict[str, Any]:
    try:
        from . import tools
        m = tools.manifest()
    except Exception as exc:  # pragma: no cover
        return _check("tools", FAIL, f"the agent tools do not load: {type(exc).__name__}: {exc}",
                      "Reinstall: pip install --force-reinstall open-riskpy")
    calculators = [t for t in m if t["kind"] == "calculator"]
    missing = [t for t in calculators if not t["available"]]
    msg = f"{len(calculators) - len(missing)} of {len(calculators)} calculators available"
    if missing:
        needs = sorted({t["install"] for t in missing if t["install"]})
        return _check("tools", WARN, msg + f" ({', '.join(t['name'] for t in missing)} need an extra)",
                      "; ".join(needs))
    return _check("tools", OK, msg)


def _mcp() -> Dict[str, Any]:
    if sys.version_info < (3, 10) or importlib.util.find_spec("mcp") is None:
        return _check("mcp_server", WARN, "the MCP server cannot run: the MCP SDK is not installed",
                      'pip install "open-riskpy[mcp]"  (only needed to serve the tools over MCP)')
    try:
        from . import mcp_server
        server = mcp_server.build_server()
        listed = mcp_server.tool_list(server)
    except BaseException as exc:  # SystemExit from build_server included
        return _check("mcp_server", FAIL, f"the MCP server does not build: {type(exc).__name__}: {exc}",
                      'pip install -U "open-riskpy[mcp]"  (the server needs MCP SDK 2.x)')
    command = shutil.which("riskpy-mcp")
    msg = f"the MCP server builds and lists {len(listed)} tools"
    if command is None:
        beside = mcp_server.script_path()
        if beside:
            return _check("mcp_server", OK, msg + f"; command {beside} (not on PATH — riskpy-mcp --config prints "
                          "the full path for the client)")
        return _check("mcp_server", WARN, msg + ", but riskpy-mcp is not on PATH",
                      f"Use the full path in your client config: {sys.executable} -m riskpy.mcp_server")
    return _check("mcp_server", OK, msg + f"; command {command}")


def _platform() -> Dict[str, Any]:
    return _check("platform", OK, f"{platform.system()} {platform.release()} {platform.machine()}",
                  system=platform.system(), machine=platform.machine())


def checks() -> List[Dict[str, Any]]:
    """Run every check; a check that itself crashes is reported as a failure, never raised."""
    out: List[Dict[str, Any]] = []
    for fn in (_python, _version, _core, _extras, _selfcheck, _tools, _mcp, _platform):
        try:
            result = fn()
        except Exception as exc:  # pragma: no cover - a check must never take the doctor down
            result = _check(fn.__name__.strip("_"), FAIL, f"the check itself failed: {type(exc).__name__}: {exc}",
                            "Report this at https://github.com/slimboi34/RiskPY/issues (RP900).")
        out.extend(result if isinstance(result, list) else [result])
    return out


def doctor() -> Dict[str, Any]:
    """``{"ok", "version", "summary": {ok, warn, fail}, "checks": [...]}`` — ``ok`` is False if any check fails."""
    import riskpy

    results = checks()
    summary = {s: sum(c["status"] == s for c in results) for s in (OK, WARN, FAIL)}
    return {"ok": summary[FAIL] == 0, "version": riskpy.__version__, "summary": summary, "checks": results}


def _format(report: Dict[str, Any]) -> str:
    marks = {OK: "ok  ", WARN: "warn", FAIL: "FAIL"}
    width = max(len(c["id"]) for c in report["checks"])
    lines = [f"riskpy-doctor — RiskPY {report['version']}", ""]
    for c in report["checks"]:
        lines.append(f"  {marks[c['status']]}  {c['id']:<{width}}  {c['message']}")
        if c["hint"] and c["status"] != OK:
            lines.append(f"        {'':<{width}}  → {c['hint']}")
    s = report["summary"]
    verdict = "healthy" if report["ok"] and not s[WARN] else ("works, with warnings" if report["ok"] else "BROKEN")
    lines += ["", f"{s[OK]} ok, {s[WARN]} warn, {s[FAIL]} fail — {verdict}"]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    from .errors import _safe_streams

    _safe_streams()
    parser = argparse.ArgumentParser(prog="riskpy-doctor",
                                     description="Check this RiskPY installation and say how to fix what is wrong.")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    report = doctor()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(_format(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
