"""RiskPY's tools over the Model Context Protocol.

    pip install "open-riskpy[mcp]"
    riskpy-mcp                 # serve on stdio, for Claude Code, Claude Desktop, Cursor and friends
    riskpy-mcp --config        # print the client configuration that connects to it
    riskpy-mcp --audit run.jsonl   # also record every call, in full, to a hash-chained ledger

Every tool in :mod:`riskpy.tools` that can run on this install is exposed under its own
name, with the schema derived from its signature (parameter descriptions included) and the
description from its docstring. The numbers an agent gets back are the ones
``riskpy-verify`` checks.

Calls go through :func:`riskpy.tools.call`, so they are checked exactly as on every other
path (a string where a number belongs is refused, not coerced) and recorded in the trace.
A failed call comes back as a tool error whose text is JSON —
``{"error": {"code": "RP105", "message": ..., "param": "sigma", "hint": ..., "docs": ...}}`` —
never a traceback. ``riskpy_status``, ``riskpy_doctor`` and ``riskpy_explain`` let the agent
see the recent calls, check the installation and look up a code or a tool's conventions.

Besides tools the server offers **resources** — reference text a client can attach to the
conversation without spending a tool call: ``riskpy://context`` (the context pack),
``riskpy://conventions``, ``riskpy://errors`` (the error catalogue), ``riskpy://tools`` (the
manifest), ``riskpy://audit`` (the ledger's status and head digest) and
``riskpy://explain/{topic}`` — and **prompts**, which are worked procedures for the jobs
that take several tools in the right order: ``capital_model``, ``reserve_review``,
``stress_test`` and ``audit_trail``. :data:`RESOURCES` and :data:`PROMPTS` list them.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
from typing import Any, List, Optional

__all__ = ["build_server", "client_config", "script_path", "tool_list", "main", "RESOURCES", "PROMPTS",
           "read_resource", "render_prompt"]

INSTRUCTIONS = (
    "RiskPY: verified actuarial and financial calculators. Rates are decimals (0.05 = 5%), times are in years, "
    "probability levels are decimals (0.995). Every tool wraps a function the library's verification suite checks "
    "against published tables and closed forms. Use aggregate_loss for claims simulation (it takes a severity mean "
    "and standard deviation on the money scale), black_scholes / option_greeks / implied_vol for options, "
    "bond_analytics, convert_rate, bootstrap_yield_curve and short_rate_zero_bond for rates and bonds, "
    "basel_irb_capital / expected_loss / merton / cds_par_spread / hazard_from_spread / vasicek_loss_quantile for "
    "credit, life_annuity_due / life_insurance / term_insurance / endowment_insurance / temporary_annuity_due / "
    "life_premium / net_premium_reserve / annuity_certain for life and annuities, chain_ladder / mack_chain_ladder / "
    "bornhuetter_ferguson for claims triangles, value_at_risk for a return series, capital_aggregate (standalone "
    "capital + correlation matrix) / capital_allocation (simulated lines joined by a copula, Euler contributions) / "
    "copula_dependence / nearest_correlation for economic capital, sensitivity (bump-and-revalue any calculator) and "
    "scenario_table (any calculator under named what-ifs) for stress testing, and verify to check the library "
    "itself. riskpy_audit reports the audit ledger and the digest to quote with your numbers. The prompts "
    "capital_model, reserve_review, stress_test and audit_trail are step-by-step procedures; the resource "
    "riskpy://conventions lists the conventions. A failed call returns JSON {\"error\": {code, message, param, hint}}: fix the named parameter as the "
    "hint says. riskpy_explain takes an error code or a tool name; riskpy_status shows recent calls; riskpy_doctor "
    "checks the installation."
)


def _mcp():
    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations
    except ImportError as exc:  # pragma: no cover - environment dependent
        from .errors import MissingDependencyError

        raise MissingDependencyError('riskpy-mcp needs the MCP SDK 2.x: pip install "open-riskpy[mcp]"',
                                     code="RP204") from exc
    return MCPServer, CallToolResult, TextContent, Tool, ToolAnnotations


def tool_list(server: Any = None) -> List[Any]:
    """The MCP ``Tool`` entries the server advertises: our schemas, read-only, closed-world."""
    _, _, _, Tool, ToolAnnotations = _mcp()
    from .tools import TOOLS

    return [
        Tool(name=t.name, description=t.description, input_schema=t.schema(),
             annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True,
                                         open_world_hint=False))
        for t in TOOLS.values() if t.available
    ]


# ---------------------------------------------------------------------------
# Resources and prompts


RESOURCES = {
    "riskpy://context": ("context", "text/markdown",
                         "The context pack: conventions, every module's public functions with signatures, the tools."),
    "riskpy://conventions": ("conventions", "text/markdown",
                             "The conventions that bite: rates as decimals, which rate is compounded how, "
                             "LogNormal(mu, sigma), seeds, losses positive."),
    "riskpy://errors": ("errors", "application/json",
                        "The error catalogue: every RPnnn code with its meaning and the fix."),
    "riskpy://tools": ("tools", "application/json", "The tool manifest: names, descriptions, input schemas."),
    "riskpy://audit": ("audit", "application/json",
                       "The audit ledger: whether calls are recorded, how many, and the head digest to quote."),
}
EXPLAIN_TEMPLATE = "riskpy://explain/{topic}"


def read_resource(uri: str) -> str:
    """The text of one of the server's resources, by URI.

    uri: one of :data:`RESOURCES`, or ``riskpy://explain/TOPIC`` with an error code (RP105)
        or a tool name (black_scholes)

    Returns Markdown or JSON text, the same content an MCP client receives from
    ``resources/read``. Usable without the MCP SDK installed.

    Example::

        from riskpy import mcp_server
        print(mcp_server.read_resource("riskpy://conventions"))
        mcp_server.read_resource("riskpy://explain/RP104")     # JSON: meaning, hint, example
    """
    from . import audit, context, errors, tools

    if uri == "riskpy://context":
        return context.build()
    if uri == "riskpy://conventions":
        return "# RiskPY conventions\n\n" + "\n".join(f"- {c}" for c in context.CONVENTIONS) + "\n"
    if uri == "riskpy://errors":
        return json.dumps(errors.catalogue(), indent=2, ensure_ascii=False)
    if uri == "riskpy://tools":
        return json.dumps(tools.manifest(), indent=2, ensure_ascii=False)
    if uri == "riskpy://audit":
        return json.dumps(audit.status(), indent=2, ensure_ascii=False)
    prefix = "riskpy://explain/"
    if isinstance(uri, str) and uri.startswith(prefix) and uri[len(prefix):]:
        return json.dumps(tools.call("riskpy_explain", {"topic": uri[len(prefix):]}, errors="return"), indent=2,
                          ensure_ascii=False)
    from .errors import UnknownNameError

    raise UnknownNameError(f"unknown resource {uri!r}; available: {sorted(RESOURCES)} and {EXPLAIN_TEMPLATE}",
                           code="RP401", param="uri")


_PROMPT_FOOTER = (
    "\n\nRules: rates, probabilities and confidence levels are decimals (0.995, not 99.5); amounts stay in one "
    "currency and unit throughout; pass a non-zero seed to every simulation so the numbers can be reproduced. If a "
    "call returns {\"error\": ...}, fix the named parameter as the hint says and call again — do not work the "
    "number out by hand. Finish by calling riskpy_audit and, if it is recording, quote the head digest with the "
    "results."
)


def _prompt_capital_model(lines: str = "", level: str = "0.995") -> str:
    subject = f" for these lines: {lines}." if lines.strip() else "."
    return (
        f"Build an economic capital view{subject} Work at the {level} confidence level.\n\n"
        "1. Collect, per line of business or risk type, either a standalone capital figure or a mean and standard "
        "deviation of the annual loss, and the correlations between them. Ask for anything missing; do not invent it.\n"
        "2. If a correlation matrix was assembled from pairwise judgements, pass it to nearest_correlation first. If "
        "`changed` is true, show the repaired matrix and say that the original could not hold.\n"
        "3. With standalone capital figures only: call capital_aggregate. Report the diversified total, the "
        "diversification benefit, and each risk's contribution and share.\n"
        "4. With means and standard deviations: call capital_allocation (measure \"tvar\", method \"euler\") with a "
        "gaussian copula as the base case, then again with a copula that has upper tail dependence — \"t\" with df "
        "around 4, or \"gumbel\" — using the same seed. Use copula_dependence to state what each copula implies: "
        "Kendall's tau, upper tail dependence, and how many times more often than independence two lines both "
        "exceed their 1-in-200.\n"
        "5. Present one table: line, mean, standalone, contribution under each copula, share. State how much of the "
        "total depends on the dependence assumption — that difference is the model risk of the aggregation.\n"
        "6. Say plainly what was assumed (marginal distribution, copula, level, trials, seed) and what was not "
        "modelled." + _PROMPT_FOOTER
    )


def _prompt_reserve_review(triangle: str = "") -> str:
    subject = f" The triangle: {triangle}" if triangle.strip() else ""
    return (
        f"Review a claims reserve from a cumulative run-off triangle.{subject}\n\n"
        "1. Confirm the triangle's shape before calculating: rows are origin periods, oldest first; columns are "
        "development periods; values are cumulative; cells not yet observed are null. Ask if any of that is unclear.\n"
        "2. Call chain_ladder for the point estimate: development factors, ultimates and the reserve by origin.\n"
        "3. Call mack_chain_ladder for the uncertainty: the standard error of each origin's reserve and of the "
        "total, and the coefficient of variation.\n"
        "4. If premiums and an expected loss ratio are available, call bornhuetter_ferguson and compare its reserve "
        "with the chain ladder's by origin. The most recent origins are where the two differ and where the chain "
        "ladder is least reliable.\n"
        "5. Use scenario_table on chain_ladder with different tail factors (for example 1.00, 1.02, 1.05) to show "
        "how much of the reserve rests on the tail assumption.\n"
        "6. Report: total reserve, its standard error, a range, the origins that drive the uncertainty, and the "
        "method differences. Flag any development factor below 1 or any origin whose latest diagonal looks out of "
        "line." + _PROMPT_FOOTER
    )


def _prompt_stress_test(tool: str = "", base_case: str = "") -> str:
    what = f" the `{tool}` calculation" if tool.strip() else " a calculation"
    base = f" Base case: {base_case}." if base_case.strip() else ""
    return (
        f"Stress test{what}.{base}\n\n"
        "1. Run the base case once with the calculator itself and state the result.\n"
        "2. Call sensitivity once per material input to get the first and second derivative of every output. "
        "Derivatives are per one whole unit of the input: multiply by 0.0001 for one basis point of a rate, by "
        "0.01 for one point of volatility. Rank the inputs by elasticity.\n"
        "3. Call scenario_table with named scenarios that move several inputs together — at least a mild one, a "
        "severe one, and the combination that would hurt most. Give each a label a reader would understand "
        "(\"rates +100bp, vol 30%\").\n"
        "4. Compare the linear estimate (derivative × move) with the full revaluation for the severe scenario; "
        "where they differ, say that the position is convex in that input.\n"
        "5. Report a table of scenarios with the change and the relative change, the most sensitive input, and "
        "any scenario the calculator refused and why." + _PROMPT_FOOTER
    )


def _prompt_audit_trail() -> str:
    return (
        "Produce an audit trail for the calculations in this conversation.\n\n"
        "1. Call riskpy_audit. If `recording` is false, say so, repeat its `how` text so the user can turn the "
        "ledger on, and fall back to riskpy_status for the recent calls (those are abbreviated, not a full record).\n"
        "2. If it is recording, call riskpy_audit with check=true. Report the number of entries, the head digest in "
        "full, whether the chain is intact, and the replay outcome: matched, drifted, skipped. Explain each skipped "
        "call (a simulation run without a seed cannot be reproduced) and each drift.\n"
        "3. List every figure you gave the user next to the call that produced it (tool and arguments).\n"
        "4. Tell the user how to check it themselves: `riskpy-audit verify FILE`, `riskpy-audit replay FILE`, and "
        "`riskpy-audit report FILE -o record.md` for the calculation record."
    )


PROMPTS = {
    "capital_model": (_prompt_capital_model,
                      "Economic capital step by step: repair the correlation matrix, aggregate, allocate to lines "
                      "with Euler contributions, and show what the copula assumption is worth."),
    "reserve_review": (_prompt_reserve_review,
                       "Review a claims reserve: chain ladder, Mack standard errors, Bornhuetter–Ferguson, and the "
                       "sensitivity to the tail factor."),
    "stress_test": (_prompt_stress_test,
                    "Stress test any calculator: sensitivities to each input, named scenarios, and where the linear "
                    "estimate breaks down."),
    "audit_trail": (_prompt_audit_trail,
                    "Produce the audit trail for this conversation's numbers: ledger head, chain check, replay."),
}


def render_prompt(name: str, **arguments: str) -> str:
    """The text of one of the server's prompts, with its arguments filled in.

    name: "capital_model", "reserve_review", "stress_test" or "audit_trail"
    arguments: the prompt's own arguments, all strings — ``lines`` and ``level`` for
        capital_model, ``triangle`` for reserve_review, ``tool`` and ``base_case`` for stress_test

    Returns the instruction text an MCP client receives from ``prompts/get``. Usable
    without the MCP SDK, for example to paste the procedure into another agent's prompt.

    Example::

        from riskpy import mcp_server
        print(mcp_server.render_prompt("capital_model", lines="motor, property, liability", level="0.995"))
    """
    from .errors import ArgumentTypeError, UnknownNameError

    if name not in PROMPTS:
        raise UnknownNameError(f"unknown prompt {name!r}; available: {sorted(PROMPTS)}", code="RP401", param="name")
    fn = PROMPTS[name][0]
    try:
        return fn(**{k: str(v) for k, v in arguments.items()})
    except TypeError:
        import inspect

        accepted = sorted(inspect.signature(fn).parameters)
        raise ArgumentTypeError(f"prompt {name!r} takes {accepted or 'no arguments'}, got {sorted(arguments)}",
                                code="RP402", param=sorted(set(arguments) - set(accepted))[0]) from None


def _register_extras(server: Any) -> None:
    """Attach the resources and prompts to an MCP server."""
    def resource(uri: str):
        def read() -> str:
            return read_resource(uri)

        return read

    for uri, (name, mime, description) in RESOURCES.items():
        server.resource(uri, name=name, description=description, mime_type=mime)(resource(uri))

    def explain(topic: str) -> str:
        return read_resource(f"riskpy://explain/{topic}")

    server.resource(EXPLAIN_TEMPLATE, name="explain", mime_type="application/json",
                    description="An error code (RP105) or a tool (black_scholes) explained: meaning, formula, "
                                "conventions, the fix.")(explain)
    for name, (fn, description) in PROMPTS.items():
        server.prompt(name=name, description=description)(fn)


def _unparsable_reply(exc: BaseException) -> Any:
    """The JSON-RPC error answering a request line the MCP SDK's parser rejected, or None.

    The SDK's stdio transport parses each line with pydantic (jiter), which refuses some JSON
    that Python's parser and most clients accept: a lone UTF-16 surrogate escape (``"\\ud800"``)
    and nesting deeper than about 200 levels. It drops such a line without the JSON-RPC
    ``-32700 Parse error`` reply, so the client waits for its request forever. The parse error
    still carries the raw line, so the request id can be recovered and the request answered:
    -32700 for a line the SDK could not parse, -32600 for valid JSON that is not a JSON-RPC
    message. A notification (no id), a response, or a line whose id cannot be read gets no reply.
    """
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return None
    try:
        first = errors()[0]
    except Exception:  # noqa: BLE001 - not a pydantic ValidationError after all
        return None
    raw = first.get("input")
    if isinstance(raw, (str, bytes, bytearray)):
        try:
            message = json.loads(raw)
        except (ValueError, RecursionError):
            return None
        code, what = -32700, "Parse error"
    elif isinstance(raw, dict):
        message, code, what = raw, -32600, "Invalid Request"
    else:
        return None
    if not isinstance(message, dict) or not isinstance(message.get("method"), str):
        return None
    request_id = message.get("id")
    if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
        return None
    from mcp.types import ErrorData, JSONRPCError

    detail = str(first.get("msg") or "invalid message")[:200]
    return JSONRPCError(jsonrpc="2.0", id=request_id, error=ErrorData(
        code=code, message=f"{what}: the server's JSON-RPC parser rejected the request ({detail}). It does not accept "
                           "lone UTF-16 surrogate escapes such as \\ud800, or nesting deeper than about 200 levels."))


class _AnsweringReadStream:
    """The SDK's stdio read stream, unchanged, except that a request line its parser dropped is
    answered (see :func:`_unparsable_reply`) before the parse error is passed on as before."""

    def __init__(self, inner: Any, write_stream: Any):
        self._inner, self._write = inner, write_stream

    @property
    def last_context(self) -> Any:
        return getattr(self._inner, "last_context", None)

    async def receive(self) -> Any:
        item = await self._inner.receive()
        if isinstance(item, Exception):
            reply = _unparsable_reply(item)
            if reply is not None:
                import anyio
                from mcp.shared.message import SessionMessage

                try:
                    await self._write.send(SessionMessage(reply))
                except (anyio.ClosedResourceError, anyio.BrokenResourceError):
                    pass                      # the client has gone; nothing to answer
        return item

    async def aclose(self) -> None:
        await self._inner.aclose()

    def close(self) -> None:
        self._inner.close()

    def __aiter__(self) -> "_AnsweringReadStream":
        return self

    async def __anext__(self) -> Any:
        import anyio

        try:
            return await self.receive()
        except anyio.EndOfStream:
            raise StopAsyncIteration from None

    async def __aenter__(self) -> "_AnsweringReadStream":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()


def build_server():
    """An MCP server whose tool calls run through :func:`riskpy.tools.call`."""
    try:
        MCPServer, CallToolResult, TextContent, _, _ = _mcp()
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(str(exc)) from exc
    import anyio

    from . import __version__
    from . import tools

    class RiskPyServer(MCPServer):
        async def list_tools(self):  # noqa: D401 - SDK hook
            return tool_list(self)

        async def run_stdio_async(self) -> None:  # SDK hook: the SDK's own stdio loop, answering parse errors
            lowlevel = getattr(self, "_lowlevel_server", None)
            if lowlevel is None:             # an SDK that changed its internals: serve exactly as it would
                return await super().run_stdio_async()
            from mcp.server.stdio import stdio_server

            async with stdio_server() as (read_stream, write_stream):
                await lowlevel.run(_AnsweringReadStream(read_stream, write_stream), write_stream,
                                   lowlevel.create_initialization_options())

        async def call_tool(self, name: str, arguments: dict, context: Any = None):  # SDK hook
            # The call runs in a worker thread, which cannot be interrupted. When the client cancels
            # (notifications/cancelled), stop waiting for it and set the event: sensitivity and
            # scenario_table check it between the calculator runs they drive and stop with RP408.
            cancel = threading.Event()

            def run() -> Any:
                with tools._cancellable(cancel):
                    return tools.call(name, arguments, errors="return", via="mcp")

            try:
                out = await anyio.to_thread.run_sync(run, abandon_on_cancel=True)
            except anyio.get_cancelled_exc_class():
                cancel.set()
                raise
            failed = isinstance(out, dict) and set(out) == {"error"}
            text = json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False)
            structured = out if isinstance(out, dict) else {"result": out}
            return CallToolResult(content=[TextContent(type="text", text=text)], structured_content=structured,
                                  is_error=failed)

    server = RiskPyServer("riskpy", instructions=INSTRUCTIONS, version=__version__)
    for t in tools.TOOLS.values():
        if t.available:
            # Registered with the SDK too, so its own bookkeeping knows the names; the two hooks
            # above decide what is listed and how a call runs.
            server.add_tool(t.fn, name=t.name, description=t.description, structured_output=False)
    _register_extras(server)
    return server


def script_path() -> Optional[str]:
    """riskpy-mcp in this interpreter's scripts directory (a virtualenv's bin/), if it is there."""
    return shutil.which("riskpy-mcp", path=os.path.dirname(sys.executable))


def client_config() -> dict:
    """The `mcpServers` entry a client needs: the command on PATH, or its full path if we can find it."""
    command = shutil.which("riskpy-mcp") or script_path() or "riskpy-mcp"
    return {"mcpServers": {"riskpy": {"command": command, "args": []}}}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="riskpy-mcp", description="Serve RiskPY's verified calculators over MCP.")
    parser.add_argument("--config", action="store_true",
                        help="print the client configuration (JSON) and the `claude mcp add` line, then exit")
    parser.add_argument("--audit", metavar="FILE",
                        help="record every calculator call, in full, to this hash-chained JSON-lines ledger "
                             "(the same as setting RISKPY_AUDIT_FILE; the server will not start if the file cannot "
                             "be opened for appending); check it with riskpy-audit")
    args = parser.parse_args(argv)
    if args.audit:
        os.environ["RISKPY_AUDIT_FILE"] = args.audit
    if args.config:
        cfg = client_config()
        print(json.dumps(cfg, indent=2))
        print(f"\n# Claude Code:\nclaude mcp add riskpy -- {cfg['mcpServers']['riskpy']['command']}")
        return 0
    ledger = os.environ.get("RISKPY_AUDIT_FILE")
    if ledger:
        # A ledger that cannot be written records nothing and says nothing (it must never break a
        # calculation), so find out now, before the client starts relying on it.
        try:
            with open(ledger, "a", encoding="utf-8"):
                pass
        except OSError as exc:
            from .errors import _output_failed

            source = "--audit" if args.audit else "RISKPY_AUDIT_FILE"
            return _output_failed("riskpy-mcp", ledger, exc, what=f"open the audit ledger ({source})")
    build_server().run(transport="stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
