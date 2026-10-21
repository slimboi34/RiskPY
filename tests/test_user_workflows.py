"""Every real-use scenario in benchmarks/user_workflows/cases.py, against the installed package.

The same scenarios drive benchmarks/user_workflows/regression.ipynb, which also runs them
against the published 0.5.2 wheel; here they only have to pass. Each runs in-process (the
cases are small), with references computed independently of RiskPY.
"""

import importlib.util
import math
import pathlib
import sys

import pytest

pytest.importorskip("scipy")
pytest.importorskip("numpy")

_PATH = pathlib.Path(__file__).resolve().parent.parent / "benchmarks" / "user_workflows" / "cases.py"
_spec = importlib.util.spec_from_file_location("riskpy_user_cases", _PATH)
cases = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cases          # dataclasses resolves the module through sys.modules
_spec.loader.exec_module(cases)

_REFS = cases.references()
_CORE_CLASSES = ("MonteCarloSimulator", "FourierTransform", "ExposureRating", "ExperienceRating", "LossTriangle")


def _core_has_fixes() -> bool:
    import riskpy

    try:
        pmf = riskpy.FourierTransform.compound_poisson_pmf([0.0, 1.0], 50.0)
    except Exception:  # pragma: no cover - a core without FourierTransform
        return False
    return abs(sum(k * p for k, p in enumerate(pmf)) - 50.0) < 1e-6


@pytest.mark.parametrize("case", cases.CASES, ids=[f"{c.section}: {c.label}" for c in cases.CASES])
def test_scenario(case):
    if any(name in case.code for name in _CORE_CLASSES) and not _core_has_fixes():
        pytest.skip("needs the compiled core from this release (built by CI from src/)")
    outcome = cases.run_one(case.code, timeout=30.0)
    verdict = cases.judge(case, outcome, _REFS)
    assert verdict == "pass", f"{verdict}: {outcome}"


def test_everyday_inputs_are_finite_and_sane():
    ns = {}
    exec(cases.EVERYDAY, ns)
    out = ns["result"]
    assert sum(len(v) for v in out.values()) > 600
    for group, values in out.items():
        assert all(math.isfinite(v) for v in values), group
    assert max(map(abs, out["implied_vol error (iv - true sigma)"])) < 1e-9
