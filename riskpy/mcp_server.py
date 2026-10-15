"""RiskPY's tools over the Model Context Protocol.

    pip install "open-riskpy[mcp]"
    riskpy-mcp                 # serve on stdio, for Claude Code, Claude Desktop, Cursor and friends
    riskpy-mcp --config        # print the client configuration that connects to it

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
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from typing import Any, List, Optional

__all__ = ["build_server", "client_config", "script_path", "tool_list", "main"]

INSTRUCTIONS = (
    "RiskPY: verified actuarial and financial calculators. Rates are decimals (0.05 = 5%), times are in years, "
    "probability levels are decimals (0.995). Every tool wraps a function the library's verification suite checks "
    "against published tables and closed forms. Use aggregate_loss for claims simulation (it takes a severity mean "
    "and standard deviation on the money scale), black_scholes / option_greeks / implied_vol for options, "
    "bond_analytics, convert_rate, bootstrap_yield_curve and short_rate_zero_bond for rates and bonds, "
    "basel_irb_capital / expected_loss / merton / cds_par_spread / hazard_from_spread / vasicek_loss_quantile for "
    "credit, life_annuity_due / life_insurance / term_insurance / endowment_insurance / temporary_annuity_due / "
    "life_premium / net_premium_reserve / annuity_certain for life and annuities, chain_ladder / mack_chain_ladder / "
    "bornhuetter_ferguson for claims triangles, value_at_risk for a return series, and verify to check the library "
    "itself. A failed call returns JSON {\"error\": {code, message, param, hint}}: fix the named parameter as the "
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

        async def call_tool(self, name: str, arguments: dict, context: Any = None):  # SDK hook
            out = await anyio.to_thread.run_sync(lambda: tools.call(name, arguments, errors="return", via="mcp"))
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
    return server


def script_path() -> Optional[str]:
    """riskpy-mcp in this interpreter's scripts directory (a virtualenv's bin/), if it is there."""
    import os

    return shutil.which("riskpy-mcp", path=os.path.dirname(sys.executable))


def client_config() -> dict:
    """The `mcpServers` entry a client needs: the command on PATH, or its full path if we can find it."""
    command = shutil.which("riskpy-mcp") or script_path() or "riskpy-mcp"
    return {"mcpServers": {"riskpy": {"command": command, "args": []}}}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="riskpy-mcp", description="Serve RiskPY's verified calculators over MCP.")
    parser.add_argument("--config", action="store_true",
                        help="print the client configuration (JSON) and the `claude mcp add` line, then exit")
    args = parser.parse_args(argv)
    if args.config:
        cfg = client_config()
        print(json.dumps(cfg, indent=2))
        print(f"\n# Claude Code:\nclaude mcp add riskpy -- {cfg['mcpServers']['riskpy']['command']}")
        return 0
    build_server().run(transport="stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
