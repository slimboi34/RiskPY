"""Regression tests for the 0.8.0 refinement of the classic modules.

Each test pins a defect found by the 0.7.0 stress test: Heston and CIR at a
vanishing vol-of-vol, uncoded overflows and log-domain errors, validators that
coerced strings and bools, unvalidated trial counts and seeds, an additive
``set_premium_column``, tutorial blocks that did not run, and a publish job that
installed packages next to its OIDC token.
"""

import decimal
import fractions
import math
import os
import pathlib
import re
import subprocess
import sys
import textwrap
import warnings
import zipfile
import xml.etree.ElementTree as ET

import pytest

from riskpy import credit, life, quant, rates, tools
from riskpy.errors import InputError, RiskPyError

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _quiet_heston(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # Feller, irrelevant here
        return quant.heston_price(*args, **kwargs)


# ---------------------------------------------------------------------------
# The publishing job installs nothing
# ---------------------------------------------------------------------------


def test_publish_job_with_the_oidc_token_installs_nothing():
    path = ROOT / ".github" / "workflows" / "publish.yml"
    if not path.exists():
        pytest.skip("workflow files are not shipped in the sdist")
    jobs, current = {}, None
    in_jobs = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("jobs:"):
            in_jobs = True
            continue
        match = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if in_jobs and match:
            current = match.group(1)
            jobs[current] = []
        elif in_jobs and current:
            jobs[current].append(line)
    privileged = [name for name, body in jobs.items() if any("id-token: write" in l for l in body)]
    assert privileged == ["publish"]
    body = "\n".join(jobs["publish"])
    assert "pip install" not in body and "setup-python" not in body
    assert "pypa/gh-action-pypi-publish@" in body
    # The metadata check still happens, in a job that cannot mint a token.
    assert "twine check --strict" in "\n".join(jobs["check-dist"])
    assert "check-dist" in body
