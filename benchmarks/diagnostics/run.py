"""Score how well RiskPY reports the mistakes an agent makes — against any installed version.

    python benchmarks/diagnostics/run.py \\
        --target 0.4.0=/path/to/venv-0.4.0/bin/python \\
        --target 0.5.0=/path/to/venv-0.5.0/bin/python \\
        --markdown benchmarks/diagnostics/RESULTS.md --json results.json

Each target is an interpreter with that RiskPY (and NumPy and the MCP SDK) installed. For
every mistake in ``mistakes.py`` the runner makes the call the way an agent would — over MCP
(``python -m riskpy.mcp_server`` on stdio) and with the CLI (``python -m riskpy.tools call``)
— from an empty working directory, so the installed package is what runs. ``missing_extra``
puts a ``numpy`` that refuses to import first on ``PYTHONPATH``.

A mistake scores one point for each of:

(a) **code**    the MCP reply carries a stable error code (``RP`` + three digits);
(b) **param**   the MCP reply names the offending parameter — a structured ``param`` field
                equal to it, or the name (for unknown_tool, the bad name; for missing_extra,
                ``numpy``) appearing as a word in the text;
(c) **hint**    the MCP reply says how to fix it — a non-empty ``hint`` field other than the
                catalogue's generic "read the message" fallback, or the text contains one of
                a fixed list of fix phrases ("pip install", "did you mean", "divide by", "use ",
                "pass ", "check ", "lists them", "e.g.", "means", "decimals");
(d) **json**    both the MCP reply and the CLI's stdout parse as a JSON object with an
                ``error`` object holding a ``code``, and neither channel shows a traceback.

A mistake the version does not detect at all (it returns a number) scores zero on all four:
that is the worst outcome, a confident wrong answer. ``detected`` counts how many raised
any error at all. The runner itself needs only the standard library.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mistakes import MISTAKES  # noqa: E402

FIX_PHRASES = ("pip install", "did you mean", "divide by", "use ", "pass ", "check ", "lists them", "e.g.", "means",
               "decimals")
GENERIC_HINTS = ("Read the message: it names the parameter and the values it accepts.",)
TRACEBACK = "Traceback (most recent call last)"
CODE = re.compile(r"\bRP\d{3}\b")


class Stdio:
    """Newline-delimited JSON-RPC to a subprocess, with timeouts."""

    def __init__(self, cmd: List[str], env: Dict[str, str], cwd: str):
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, encoding="utf-8", bufsize=1, env=env, cwd=cwd)
        self.lines: List[str] = []
        self.next_id = 1
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=self.proc.stderr.read, daemon=True).start()

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self.lines.append(line)

    def request(self, method: str, params: Optional[dict] = None, timeout: float = 60.0) -> dict:
        rid, self.next_id = self.next_id, self.next_id + 1
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}) + "\n")
        self.proc.stdin.flush()
        deadline, seen = time.time() + timeout, 0
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
            time.sleep(0.01)
        raise TimeoutError(method)

    def notify(self, method: str) -> None:
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method, "params": {}}) + "\n")
        self.proc.stdin.flush()

    def close(self) -> None:
        self.proc.kill()
        self.proc.wait(timeout=10)


def _env(block_numpy: bool, stub_dir: str) -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "RISKPY_TRACE_FILE")}
    env["PYTHONIOENCODING"] = "utf-8"
    if block_numpy:
        env["PYTHONPATH"] = stub_dir
    return env


def _mcp_session(python: str, block_numpy: bool, stub_dir: str, cwd: str) -> Stdio:
    io = Stdio([python, "-m", "riskpy.mcp_server"], _env(block_numpy, stub_dir), cwd)
    io.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                              "clientInfo": {"name": "riskpy-diagnostics-benchmark", "version": "1"}})
    io.notify("notifications/initialized")
    return io


def _mcp_call(io: Stdio, tool: str, args: dict) -> Tuple[bool, str]:
    reply = io.request("tools/call", {"name": tool, "arguments": args})
    if "error" in reply:                                    # a JSON-RPC protocol error rather than a tool error
        return True, json.dumps(reply["error"])
    result = reply["result"]
    text = "\n".join(c.get("text", "") for c in result.get("content", []))
    return bool(result.get("isError")), text


def _cli_call(python: str, tool: str, args: dict, block_numpy: bool, stub_dir: str, cwd: str) -> Tuple[int, str, str]:
    out = subprocess.run([python, "-m", "riskpy.tools", "call", tool, json.dumps(args)], capture_output=True,
                         text=True, encoding="utf-8", env=_env(block_numpy, stub_dir), cwd=cwd, timeout=120)
    return out.returncode, out.stdout, out.stderr


def _error_object(text: str) -> Optional[dict]:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(data, dict) and isinstance(data.get("error"), dict) and data["error"].get("code"):
        return data["error"]
    return None


def _word(token: str, text: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(token)}(?![\w])", text) is not None


def score(m: dict, mcp_failed: bool, mcp_text: str, cli_code: int, cli_out: str, cli_err: str) -> Dict[str, Any]:
    detected = mcp_failed
    structured = _error_object(mcp_text) if mcp_failed else None
    text = mcp_text if mcp_failed else ""
    a = bool(detected and (CODE.search(text)))
    if structured is not None:
        b = structured.get("param") == m["param"] or _word(m["param"], structured.get("message", ""))
        hint = structured.get("hint") or ""
        c = bool(hint) and hint not in GENERIC_HINTS
    else:
        b = bool(detected and _word(m["param"], text))
        c = bool(detected and any(p in text.lower() for p in FIX_PHRASES))
    cli_struct = _error_object(cli_out)
    d = bool(detected and structured is not None and cli_struct is not None
             and TRACEBACK not in mcp_text + cli_out + cli_err)
    return {"detected": detected, "code": a, "param": b, "hint": c, "json": d,
            "mcp": mcp_text[:400], "cli_stdout": cli_out[:400], "cli_stderr": cli_err[:400], "cli_exit": cli_code}


def run_target(label: str, python: str) -> Dict[str, Any]:
    version = subprocess.run([python, "-c", "import riskpy; print(riskpy.__version__)"], capture_output=True,
                             text=True, cwd=tempfile.gettempdir()).stdout.strip()
    with tempfile.TemporaryDirectory() as cwd, tempfile.TemporaryDirectory() as stub:
        os.makedirs(os.path.join(stub, "numpy"))
        with open(os.path.join(stub, "numpy", "__init__.py"), "w", encoding="utf-8") as fh:
            fh.write('raise ImportError("numpy is blocked for this benchmark case")\n')
        sessions: Dict[bool, Stdio] = {}
        results = []
        try:
            for m in MISTAKES:
                block = bool(m.get("block_numpy"))
                if block not in sessions:
                    sessions[block] = _mcp_session(python, block, stub, cwd)
                failed, text = _mcp_call(sessions[block], m["tool"], m["args"])
                code, out, err = _cli_call(python, m["tool"], m["args"], block, stub, cwd)
                results.append({"id": m["id"], "what": m["what"], **score(m, failed, text, code, out, err)})
        finally:
            for s in sessions.values():
                s.close()
    totals = {k: sum(r[k] for r in results) for k in ("detected", "code", "param", "hint", "json")}
    points = totals["code"] + totals["param"] + totals["hint"] + totals["json"]
    return {"label": label, "python": python, "version": version, "cases": len(results), "totals": totals,
            "points": points, "max_points": 4 * len(results), "results": results}


def _tick(v: bool) -> str:
    return "✓" if v else "·"


def markdown(runs: List[Dict[str, Any]]) -> str:
    n = runs[0]["cases"]
    lines = [
        "# Diagnostics benchmark: what an agent sees when it gets a call wrong",
        "",
        f"{n} realistic agent mistakes (`mistakes.py`), each sent over MCP (`riskpy-mcp` on stdio) and the CLI "
        "(`riskpy-tools call`) against an installed RiskPY. Scored by `run.py` — the rules are in its docstring: "
        "**(a)** a stable error code, **(b)** the offending parameter named, **(c)** an actionable hint, "
        "**(d)** structured JSON on both channels with no traceback. *Detected* counts mistakes that raised any "
        "error at all; an undetected mistake returned a number — a confident wrong answer — and scores zero.",
        "",
        "## Summary",
        "",
        "| version | detected | (a) code | (b) param | (c) hint | (d) JSON | score |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        t = r["totals"]
        lines.append(f"| {r['label']} (`{r['version']}`) | {t['detected']}/{n} | {t['code']}/{n} | {t['param']}/{n} "
                     f"| {t['hint']}/{n} | {t['json']}/{n} | **{r['points']}/{r['max_points']}** "
                     f"({100 * r['points'] / r['max_points']:.0f}%) |")
    lines += ["", "## Per mistake", "",
              "Each cell is detected · code · param · hint · JSON (✓ met, · not met).", "",
              "| mistake | " + " | ".join(r["label"] for r in runs) + " |",
              "|---|" + "---|" * len(runs)]
    for i, m in enumerate(runs[0]["results"]):
        cells = []
        for r in runs:
            x = r["results"][i]
            cells.append(" ".join(_tick(x[k]) for k in ("detected", "code", "param", "hint", "json")))
        lines.append(f"| `{m['id']}` — {m['what']} | " + " | ".join(cells) + " |")
    lines += ["", "## What the agent actually saw over MCP", ""]
    for i, m in enumerate(runs[0]["results"]):
        lines.append(f"### `{m['id']}`")
        lines.append("")
        for r in runs:
            x = r["results"][i]
            shown = x["mcp"].strip().replace("\n", " ")
            shown = re.sub(r"\s+", " ", shown)
            lines.append(f"- **{r['label']}**: `{shown[:300]}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target", action="append", required=True, metavar="LABEL=PYTHON",
                        help="a label and the interpreter that has that RiskPY installed; repeat")
    parser.add_argument("--markdown", help="write the report here")
    parser.add_argument("--json", help="write the raw results here")
    args = parser.parse_args(argv)
    runs = []
    for spec in args.target:
        label, _, python = spec.partition("=")
        print(f"running {label} ({python}) ...", file=sys.stderr)
        runs.append(run_target(label, python))
        t = runs[-1]["totals"]
        print(f"  {runs[-1]['points']}/{runs[-1]['max_points']}  detected {t['detected']}  code {t['code']}  "
              f"param {t['param']}  hint {t['hint']}  json {t['json']}", file=sys.stderr)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(runs, fh, indent=2, ensure_ascii=False)
    report = markdown(runs)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as fh:
            fh.write(report)
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
