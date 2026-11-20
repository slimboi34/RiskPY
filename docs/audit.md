# The audit ledger

A number in a board paper, a reserve report or an agent's answer raises three
questions. What exactly was calculated? Has the record been altered since? Does
the library still give that answer? `riskpy.audit` records every tool call in
full, chains the records with SHA-256 so that any change is detectable, and
replays them on demand. *New in 0.7.0.*

It complements [the trace](diagnostics.md#the-trace). The trace is for
debugging: recent calls, arguments abbreviated, in memory. The ledger is for
evidence: every argument and result, on disk, tamper-evident.

## Recording

Set one environment variable and every calculator call made through
`tools.call`, `riskpy-tools call` or the MCP server is appended to that file:

```bash
export RISKPY_AUDIT_FILE=q4-capital.jsonl
riskpy-tools call black_scholes '{"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2}'

riskpy-mcp --audit q4-capital.jsonl        # the same, for an agent's MCP session
```

Or in Python:

```python
from riskpy import audit, tools

with audit.record("q4-capital.jsonl", note="Q4 capital run") as ledger:
    tools.call("capital_aggregate", {"standalone": [100, 60], "correlation": [[1, 0.25], [0.25, 1]]})
    tools.call("black_scholes", {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2})

ledger.head        # 64 hex characters that identify the whole run
len(ledger)        # 2
```

| Function | What it does |
|---|---|
| `audit.record(path=None, note=None)` | context manager: record the calls inside the block |
| `audit.start(path=None, note=None)` / `audit.stop()` | the same without a block |
| `audit.active()` | the ledger currently recording, or `None` |
| `audit.status(check=False)` | recording or not, the file, the entry count and the head |

With `path=None` the ledger is kept in memory (`ledger.entries`). A file that
already holds a ledger is continued from its last entry, so separate processes
appending one after another form a single chain.

What is and is not recorded:

- Every calculator call, successful or refused. A refusal is stored with its
  error code, message and parameter.
- Not the introspection tools (`riskpy_status`, `riskpy_doctor`,
  `riskpy_explain`, `riskpy_audit`).
- Not direct Python calls such as `quant.black_scholes(...)` or
  `tools.black_scholes(...)`; go through `tools.call`. Use `ledger.append(...)`
  to add a calculation made elsewhere.

If the file cannot be written, the calculation still returns and the failure is
counted in `ledger.failures` (`write_failures` in the status).

## What an entry holds

One JSON object per line:

```json
{"seq": 2, "at": "2026-11-20T09:14:03.211+00:00", "tool": "black_scholes",
 "arguments": {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2},
 "ms": 0.041, "via": "cli", "riskpy": "0.7.0", "python": "3.12.13",
 "result": 10.450583572185565,
 "prev": "378d3eea…", "digest": "9c1f0b52…"}
```

`digest` is the SHA-256 of the entry's canonical JSON (keys sorted, no
whitespace, non-ASCII escaped) without the digest itself. `prev` is the digest
of the entry before it, and 64 zeros for the first. Changing, removing,
inserting or reordering an entry therefore breaks every digest after it. The
last digest is the **head**: one string that pins the whole run. Quote it next
to the numbers it supports.

## Checking a ledger

```bash
riskpy-audit verify q4-capital.jsonl      # is the chain intact? exit 1 if not
riskpy-audit replay q4-capital.jsonl      # does every call recompute to the recorded result?
riskpy-audit report q4-capital.jsonl -o record.md --check --title "Q4 capital — calculation record"
```

```python
audit.verify_chain("q4-capital.jsonl")    # ChainReport(ok, entries, head, broken_at, reason)
audit.replay("q4-capital.jsonl")          # ReplayReport(chain, matched, drifted, skipped, ...)
audit.report("q4-capital.jsonl", title="...", check=True)   # Markdown text
audit.load("q4-capital.jsonl")            # the entries, as dicts
```

**`verify_chain`** recomputes every digest and link. When something is wrong it
names the first affected entry and says whether its content was changed or an
entry was removed, inserted or reordered.

**`replay`** runs every recorded call again with the installed library and
compares the results number by number (relative tolerance 1e-9 by default,
`rel_tol=` or `--rel-tol` to change it). A recorded refusal matches when the
call is refused again with the same error code. The report lists:

- `matched`: calls that reproduced;
- `drifted`: calls whose result differs, with the path of each differing
  number, the recorded and the new value;
- `skipped`: calls that cannot be recomputed, with the reason.

Replayed calls are not themselves recorded.

Two uses. Before signing off a report, replay confirms that its numbers are
reproducible. After upgrading RiskPY, replay shows exactly which reported
numbers the new version changes.

**`report`** writes a Markdown calculation record: the head digest and chain
status, the versions and period covered, the conventions, a table of the calls,
then each call's full arguments, result and digest. With `check=True` it
includes the replay outcome.

## Limits

- **Seeds.** A simulation called with `seed=0` draws a fresh seed, so it is
  recorded but cannot be replayed; it appears under `skipped`. Pass a non-zero
  seed for any number that will be quoted.
- **What the chain proves.** It proves the ledger is internally consistent. It
  cannot prove the whole file was not regenerated from scratch; for that the
  head has to be kept somewhere else, such as the report, an email or a commit.
- **One writer at a time.** Two processes appending at the same instant can
  fork the chain, and `verify` will report the break.
- **Not secret.** The ledger stores arguments and results in plain text.

## For an agent

The `riskpy_audit` tool returns the status, so an agent can end its answer with
the evidence for it:

```python
tools.call("riskpy_audit", {})
# {"recording": true, "file": "q4-capital.jsonl", "entries": 2, "head": "9c1f0b52…",
#  "chain_ok": true, "write_failures": 0, "how": "Quote `head` with your numbers: …"}

tools.call("riskpy_audit", {"check": True})["replay"]     # matched / drifted / skipped
```

When nothing is being recorded, `how` says how to turn the ledger on. Over MCP
the same status is the resource `riskpy://audit`, and the `audit_trail` prompt
walks an agent through producing the trail for a conversation
([For AI agents](agents.md#resources-and-prompts)).
