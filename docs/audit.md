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
| `audit.status(check=False, replay_last=50, replay_seconds=10)` | recording or not, the file, the entry count and the head; with `check=True`, a replay of the most recent entries |

With `path=None` the ledger is kept in memory (`ledger.entries`). A file that
already holds a ledger is continued from its last entry, so separate processes
appending one after another form a single chain. A writer remembers where the
file ended, so a long ledger, or one very long entry, costs nothing on the next
call.

If the last line is torn (a crash or a full disk mid-write) or is not a ledger
entry at all, the next entry starts on a new line, takes its sequence number
from the last good entry and chains from the damaged line's hash, so `verify`
reports the damage at that line instead of the record silently restarting.

What is and is not recorded:

- Every calculator call, successful or refused. A refusal is stored with its
  error code, message (up to 1,000 characters) and parameter (up to 80).
- A tool name that is not a real tool's (letters, digits and underscores, up
  to 64) is stored as its Python repr, so it cannot carry a raw newline.
- Not the introspection tools (`riskpy_status`, `riskpy_doctor`,
  `riskpy_explain`, `riskpy_audit`).
- Not direct Python calls such as `quant.black_scholes(...)` or
  `tools.black_scholes(...)`; go through `tools.call`. Use `ledger.append(...)`
  to add a calculation made elsewhere.

If the file cannot be written, the calculation still returns and the failure is
counted in `ledger.failures` (`write_failures` in the status), with the reason
in `ledger.failure_reasons` (`write_failure_reasons`): the folder is missing,
the disk is full, and so on.

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

Arguments are stored exactly, so that replay recomputes the call that was
made. Plain JSON values are written as they are. Values JSON cannot hold are
written as small objects tagged `"$riskpy"`, and replay turns them back into
the original. *Changed in 0.8.0:* 0.7.0 wrote NaN as `null` and flattened
bytes and arrays, so replaying such a call could report drift that was not
there.

| Argument | Stored as |
|---|---|
| NaN, `inf`, `-inf` | `{"$riskpy": "float", "value": "nan"}` |
| bytes | `{"$riskpy": "bytes", "base64": "…"}` |
| NumPy array | `{"$riskpy": "ndarray", "dtype": "int64", "shape": [2, 2], "data": [...]}` |
| NumPy scalar (other than `float64`) | `{"$riskpy": "numpy", "dtype": "float32", "value": 0.05}` |
| a set, a dict with non-string keys, an integer over ~600 digits | tagged the same way |
| anything else | `{"$riskpy": "unrepresentable", "type": …, "repr": …}`; replay skips the call |
| arguments nested more than 256 levels deep | `{"$riskpy": "unserialisable", "reason": …, "sha256": …}`; the call is still in the chain, replay skips it |

Tuples are stored as lists, which the tools treat the same way. Ledgers
written by 0.7.0 hold no tags and verify and replay as before.

## Checking a ledger

```bash
riskpy-audit verify q4-capital.jsonl      # is the chain intact? exit 1 if not
riskpy-audit verify --head 9c1f0b52… q4-capital.jsonl     # ... and does it end at the head that was quoted?
riskpy-audit replay q4-capital.jsonl      # does every call recompute to the recorded result?
riskpy-audit report q4-capital.jsonl -o record.md --check --title "Q4 capital — calculation record"
```

```python
audit.verify_chain("q4-capital.jsonl")    # ChainReport(ok, entries, head, broken_at, reason)
audit.verify_chain("q4-capital.jsonl", expected_head="9c1f0b52…", expected_entries=2)
audit.replay("q4-capital.jsonl")          # ReplayReport(chain, matched, drifted, skipped, ...)
audit.report("q4-capital.jsonl", title="...", check=True)   # Markdown text
audit.load("q4-capital.jsonl")            # the entries, as dicts
```

**`verify_chain`** recomputes every digest and link. When something is wrong it
names the first affected entry and says whether its content was changed, an
entry was removed, inserted or reordered, or the line is damaged (a torn write,
bytes that are not UTF-8 text, a line that is not JSON). A damaged line is
reported as a broken chain at that line; `verify`, `replay` and `report` still
read the entries around it, while `load` refuses the file (RP100).

Entries cut off the end of a ledger, or new ones appended after it, leave a
valid chain with a different head. To catch those, pass the head quoted with
the numbers: `expected_head=` in Python, `--head` on the command line
(`expected_entries=` / `--entries N` checks the count). The check fails if the
chain does not end exactly at that head, and says whether entries were added
after it or the head is missing (entries removed from the end, or a different
ledger). *New in 0.8.0.*

The digest covers an entry's values, not its formatting. A ledger re-serialised
by another JSON tool, with different spacing or key order, still verifies. A
line that repeats a key is damaged, because JSON readers disagree on which value
it holds.

**`replay`** runs every recorded call again with the installed library and
compares the results number by number (relative tolerance 1e-9 by default,
`rel_tol=` or `--rel-tol` to change it). A recorded refusal matches when the
call is refused again with the same error code. The report lists:

- `matched`: calls that reproduced;
- `drifted`: calls whose result differs, with the path of each differing
  number, the recorded and the new value;
- `skipped`: calls that cannot be recomputed, with the reason: an unseeded
  simulation, a tool this installation does not have, an argument that could
  not be stored exactly, a damaged line, or a foreign entry for a tool that is
  never recorded (`riskpy_audit` and the other introspection tools).

Replayed calls are not themselves recorded, even when the ledger being
replayed is the one `RISKPY_AUDIT_FILE` names.

Two uses. Before signing off a report, replay confirms that its numbers are
reproducible. After upgrading RiskPY, replay shows exactly which reported
numbers the new version changes.

**`report`** writes a Markdown calculation record: the head digest and chain
status, the versions and period covered, the conventions, a table of the calls,
then each call's full arguments and result (or error) in fenced JSON blocks, and
its digest. It ends with the `riskpy-audit verify --head …` command that checks
the ledger against this record. With `check=True` it includes the replay outcome.

Tool names, arguments and error messages come from whoever made the calls,
possibly an agent. In the report they stay inert: text from the ledger is
escaped or put in code, so it cannot add headings, table cells, links or HTML.
The title is used as given.

The command-line tool answers every problem with a one-line coded error and
exit status 1, never a traceback: RP100 for an invalid `--head`, RP505 for a
file it cannot read or an output file it cannot write, RP900 for anything
unexpected.

## Limits

- **Seeds.** A simulation called with `seed=0` draws a fresh seed, so it is
  recorded but cannot be replayed; it appears under `skipped`. Pass a non-zero
  seed for any number that will be quoted.
- **What the chain proves.** On its own, the chain proves the ledger is
  internally consistent: nothing inside it was edited, removed, inserted or
  reordered. It cannot show that entries were not cut off the end, appended
  later, or that the whole file was not regenerated from scratch. The digests
  use no secret key, so anyone who can write the file can recompute them. For
  those cases the head has to be kept somewhere else, such as the report, an
  email or a commit, and checked with `verify --head`.
- **One writer at a time.** Two processes appending at the same instant can
  fork the chain, and `verify` will report the break.
- **A crash mid-write** leaves a torn last line. The next entry is written on a
  line of its own and `verify` reports the torn line, so nothing that was
  recorded is lost and the damage is visible.
- **Not secret.** The ledger stores arguments and results in plain text.

## For an agent

The `riskpy_audit` tool returns the status, so an agent can end its answer with
the evidence for it:

```python
tools.call("riskpy_audit", {})
# {"recording": true, "file": "q4-capital.jsonl", "entries": 2, "head": "9c1f0b52…",
#  "chain_ok": true, "write_failures": 0, "write_failure_reasons": {}, "how": "Quote `head` with your numbers: …"}

tools.call("riskpy_audit", {"check": True})["replay"]     # matched / drifted / skipped
```

A ledger is append-only and can outlive many sessions, so one `check` replays
only the most recent 50 entries and stops after about 10 seconds. The result's
`replayed` says how many entries it covered, and `how` says when that was not
all of them. `riskpy-audit replay FILE` replays every entry. In Python,
`audit.status(check=True, replay_last=None, replay_seconds=None)` lifts both
limits. *Changed in 0.8.0:* the check used to replay the whole ledger in one
call.

When nothing is being recorded, `how` says how to turn the ledger on. Over MCP
the same status is the resource `riskpy://audit`, and the `audit_trail` prompt
walks an agent through producing the trail for a conversation
([For AI agents](agents.md#resources-and-prompts)).
