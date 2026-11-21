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
# UnderwritingApp.set_premium_column moves the column
# ---------------------------------------------------------------------------

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _cells(path):
    with zipfile.ZipFile(path) as z:
        try:
            shared = ["".join(t.itertext())
                      for t in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS)]
        except KeyError:
            shared = []
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    out = {}
    for cell in sheet.iter("{%s}c" % _NS["m"]):
        value = cell.find("m:v", _NS)
        if value is None:
            inline = cell.find("m:is", _NS)
            if inline is None:
                continue
            out[cell.get("r")] = "".join(inline.itertext())
            continue
        out[cell.get("r")] = shared[int(value.text)] if cell.get("t") == "s" else value.text
    return out


def _app():
    try:
        from riskpy import UnderwritingApp
    except ImportError as exc:  # headless CI without Tkinter or matplotlib
        pytest.skip(f"UnderwritingApp unavailable: {exc}")
    return UnderwritingApp(title="t")


def test_set_premium_column_twice_moves_the_premium(tmp_path, capsys):
    app = _app()
    app.add_field("x", "X", "A")
    app.set_logic(lambda inputs: inputs.get("x", 0.0) * 2)
    app.set_premium_column("B", "Premium")
    app.set_premium_column("C", "Premium")
    app.add_field("y", "Y", "D")  # mapped after the move: still exported

    app.calculate_headless({"x": 5.0, "y": 1.0})
    quote = tmp_path / "quote.xlsx"
    app.export_excel_headless(str(quote))
    cells = _cells(quote)
    headers = {k: v for k, v in cells.items() if k.endswith("1") and len(k) == 2}
    assert headers == {"A1": "X", "C1": "Premium", "D1": "Y"}
    assert list(headers.values()).count("Premium") == 1
    assert cells["C2"] == "10"

    book = tmp_path / "book.xlsx"
    source = tmp_path / "b.csv"
    source.write_text("x,y\n1,0\n2,0\n")
    app.calculate_batch(str(source), str(book))
    cells = _cells(book)
    assert "B1" not in cells and cells["C1"] == "Premium" and (cells["C2"], cells["C3"]) == ("2", "4")
    capsys.readouterr()


def test_set_premium_column_same_column_relabels(tmp_path, capsys):
    app = _app()
    app.add_field("x", "X", "A")
    app.set_logic(lambda inputs: 1.0)
    exporter = app.exporter
    app.set_premium_column("B", "Premium")
    app.set_premium_column("B", "Final Premium")
    assert app.exporter is exporter  # nothing to drop, so no rebuild
    app.calculate_headless({"x": 1.0})
    quote = tmp_path / "q.xlsx"
    app.export_excel_headless(str(quote))
    cells = _cells(quote)
    assert cells["B1"] == "Final Premium" and "C1" not in cells
    capsys.readouterr()


# ---------------------------------------------------------------------------
# Every tutorial runs as written
# ---------------------------------------------------------------------------


def _tutorials():
    text = (ROOT / "docs" / "TUTORIALS.md").read_text(encoding="utf-8")
    sections = re.split(r"^## (Tutorial \d+[^\n]*)$", text, flags=re.M)
    out = []
    for title, body in zip(sections[1::2], sections[2::2]):
        blocks = re.findall(r"^```(\w*)\n(.*?)^```", body, flags=re.M | re.S)
        out.append((title.split(":")[0], blocks))
    return out


# What each tutorial prints that pins its numbers (Tutorial 4: 1,080 + 13,500 + 2,160).
_TUTORIAL_OUTPUT = {"Tutorial 4": "Processed 3 policies. Total: $16,740.00"}


@pytest.mark.parametrize("title, blocks", _tutorials(), ids=[t for t, _ in _tutorials()])
def test_tutorial_blocks_run_in_a_fresh_process(title, blocks, tmp_path):
    pytest.importorskip("numpy")
    python = [code for lang, code in blocks if lang == "python"]
    assert python, f"{title} has no Python blocks"
    for lang, code in blocks:
        if lang == "csv":
            (tmp_path / "submissions.csv").write_text(code)
    if any("UnderwritingApp" in code for code in python):
        _app()  # skip where Tkinter is missing
    # The window is the one thing a test cannot open: run() becomes a no-op.
    prelude = textwrap.dedent("""
        try:
            import riskpy.app
            riskpy.app.UnderwritingApp.run = lambda self, *a, **k: None
        except ImportError:
            pass
    """)
    script = tmp_path / "tutorial.py"
    script.write_text(prelude + "\n".join(python))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True,
                          timeout=300, env=env)
    assert done.returncode == 0, f"{title} failed:\n{done.stderr[-2000:]}"
    assert _TUTORIAL_OUTPUT.get(title, "") in done.stdout


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
