"""Tests for the verification suite itself.

The suite's job is to catch a wrong number; these tests make sure the suite
would notice one, reports it legibly, and runs green on the shipped code.
"""

from __future__ import annotations

import json
import math

import pytest

from riskpy import verify


def test_check_arithmetic():
    ok = verify.Check("m", "exact", 1.0, 1.0, 0.0)
    assert ok.passed and ok.error == 0.0 and ok.margin == 0.0
    near = verify.Check("m", "near", 1.0, 1.0 + 1e-9, 1e-8)
    assert near.passed and near.margin == pytest.approx(0.1)
    bad = verify.Check("m", "bad", 1.0, 2.0, 1e-3)
    assert not bad.passed and bad.margin == pytest.approx(1000.0)
    assert not verify.Check("m", "nan", math.nan, 1.0, 1.0).passed
    assert verify.Check("m", "inf", math.inf, math.inf, 0.0).passed
    assert verify.Check("m", "zero tolerance miss", 1.0, 1.5, 0.0).margin == math.inf
    assert bad.to_dict()["passed"] is False


def test_report_shapes_and_renderings():
    checks = [
        verify.Check("a", "one", 1.0, 1.0, 1e-12),
        verify.Check("a", "two", 2.0, 2.5, 1e-12),
        verify.Check("b", "three", 3.0, 3.0, 1e-12),
    ]
    report = verify.Report(checks, 0.5, "x.y.z", {"bench": 0.01})
    assert not report
    assert len(report.failed) == 1 and len(report.passed) == 2
    assert set(report.by_module()) == {"a", "b"}
    text = report.summary(verbose=True)
    assert "FAILED" in text and "two" in text and "BENCHMARKS" in text and "ALL CHECKS" in text
    payload = json.loads(report.to_json())
    assert payload["failed"] == 1 and payload["version"] == "x.y.z" and payload["benchmarks"] == {"bench": 0.01}
    markdown = report.to_markdown()
    assert markdown.count("❌") == 1 and markdown.count("✅") == 2 and "| bench |" in markdown


def test_unknown_module_is_rejected():
    with pytest.raises(ValueError, match="unknown module"):
        verify.checks(["astrology"])


def test_fast_modules_run_green():
    report = verify.run(["special", "viz", "quant", "core"], oracle=False)
    assert report, report.summary()
    assert report.seconds < 30.0
    assert not report.oracle
    assert all(c.module in {"special", "viz", "quant", "core"} for c in report.checks)


def test_every_module_has_checks_and_all_pass():
    report = verify.run()
    assert report, report.summary()
    grouped = report.by_module()
    for module in verify.MODULES:
        assert len(grouped.get(module, [])) >= 3, module
    assert len(report.checks) >= 100


def test_oracle_checks_appear_only_when_scipy_is_present():
    report = verify.run(["special"])
    has_scipy = pytest.importorskip("scipy") is not None
    oracle_rows = [c for c in report.checks if c.name.startswith("oracle:")]
    assert bool(oracle_rows) == has_scipy == report.oracle
    assert all(c.passed for c in oracle_rows)


def test_cli(tmp_path, capsys):
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"
    code = verify.main(["-m", "viz", "-m", "special", "--no-oracle", "--json", str(out_json), "--markdown", str(out_md)])
    assert code == 0
    printed = capsys.readouterr().out
    assert "verification" in printed and "viz" in printed
    assert json.loads(out_json.read_text())["ok"] is True
    assert out_md.read_text().startswith("**RiskPY")
    assert verify.main(["-m", "viz", "-q"]) == 0
    assert capsys.readouterr().out == ""


def test_benchmark_returns_timings():
    timings = verify.benchmark(repeat=1)
    assert len(timings) >= 10
    assert all(t >= 0.0 for t in timings.values())
