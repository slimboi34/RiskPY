# Security policy

## Supported versions

Security fixes go into the latest release on PyPI (`open-riskpy`). Older releases are not
patched; upgrade with `pip install -U open-riskpy`.

## Reporting a vulnerability

Please report privately, not in a public issue: use GitHub's
[private vulnerability reporting](https://github.com/slimboi34/RiskPY/security/advisories/new)
for this repository. Include the version, a minimal reproduction and what you expected.
You should get a reply within a week; a fix, if one is needed, ships as a patch release
with the reporter credited unless they prefer otherwise.

## Scope

In scope: the library, its compiled core, the command-line tools (`riskpy-tools`,
`riskpy-mcp`, `riskpy-context`, `riskpy-verify`) and this repository's release workflows.
Inputs from an agent through `riskpy.tools` / `riskpy-mcp` are treated as untrusted: a crash,
hang or unbounded resource use reachable through them is a vulnerability. The library
functions underneath trust their caller, as any numerical library does.
