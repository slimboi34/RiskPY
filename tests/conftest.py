"""Pytest bootstrap: import the *installed* riskpy package, not the source tree.

Running `pytest` from the repo root puts the uncompiled `riskpy/` source folder
on sys.path ahead of the wheel/editable install. That shadows the compiled
`cpp_underwriter` extension and breaks every native test. Strip the repo root
and evict cached modules so the installed package wins.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_repo_str = str(_REPO_ROOT)

# Remove repo root entries that would make `import riskpy` load ./riskpy/
def _is_repo_root(entry: str) -> bool:
    try:
        return Path(entry).resolve() == _REPO_ROOT
    except (OSError, RuntimeError, ValueError):
        return False


sys.path[:] = [p for p in sys.path if not _is_repo_root(p)]

# Evict any partially imported riskpy modules from a prior collection step
for key in list(sys.modules):
    if key == "riskpy" or key.startswith("riskpy."):
        del sys.modules[key]
