"""RiskPY's tools over the Model Context Protocol.

    pip install "open-riskpy[mcp]"
    riskpy-mcp                 # serve on stdio, for Claude Code, Claude Desktop, Cursor and friends
    riskpy-mcp --config        # print the client configuration that connects to it

Every tool in :mod:`riskpy.tools` that can run on this install is exposed under its own
name, with the schema derived from its signature and the description from its docstring.
The numbers an agent gets back are the ones ``riskpy-verify`` checks.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from typing import List, Optional

__all__ = ["build_server", "client_config", "main"]

INSTRUCTIONS = (
    "RiskPY: verified actuarial and financial calculators. Rates are decimals (0.05 = 5%), times are in years, "
    "probability levels are decimals (0.995). Every tool wraps a function the library's verification suite checks "
    "against published tables and closed forms. Use aggregate_loss for claims simulation (it takes a severity mean "
    "and standard deviation on the money scale), black_scholes / option_greeks / implied_vol for options, "
    "bond_analytics for bonds, basel_irb_capital / expected_loss / merton / cds_par_spread for credit, "
    "life_annuity_due / life_insurance / net_premium_reserve / annuity_certain for life and annuities, "
    "chain_ladder / mack_chain_ladder for claims triangles, and verify to check the library itself."
)


def build_server():
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit('riskpy-mcp needs the MCP SDK: pip install "open-riskpy[mcp]"') from exc
    from . import __version__
    from .tools import TOOLS

    server = MCPServer("riskpy", instructions=INSTRUCTIONS, version=__version__)
    for t in TOOLS.values():
        if t.available:
            server.add_tool(t.fn, name=t.name, description=t.description)
    return server


def client_config() -> dict:
    """The `mcpServers` entry a client needs: the command on PATH, or its full path if we can find it."""
    command = shutil.which("riskpy-mcp") or "riskpy-mcp"
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
