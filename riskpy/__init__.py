"""RiskPY — high-performance actuarial engine (C++ core + optional Tkinter GUI)."""

from __future__ import annotations

import importlib
from typing import Any, List

try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
except ImportError:  # pragma: no cover - Python < 3.8
    from importlib_metadata import PackageNotFoundError, version as _pkg_version  # type: ignore

try:
    __version__ = _pkg_version("open-riskpy")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

# Eagerly import the compiled extension. Fail with a clear message if the wheel
# was not built (common after a source checkout without `pip install`).
try:
    from .cpp_underwriter import (  # noqa: F401
        ActuarialMath,
        ExcelExporter,
        ExperienceRating,
        ExposureRating,
        FactorModel,
        Field,
        FourierTransform,
        LossTriangle,
        MonteCarloSimulator,
        RateAnalyzer,
        RiskEngine,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "RiskPY C++ extension 'cpp_underwriter' is not available. "
        "Install/build the package first, e.g.:\n"
        "  pip install -e .\n"
        "Requires a C++17 compiler, CMake ≥ 3.15, and pybind11."
    ) from exc

__all__ = [
    "UnderwritingApp",
    "FactorModel",
    "ActuarialMath",
    "RiskEngine",
    "ExcelExporter",
    "Field",
    "MonteCarloSimulator",
    "LossTriangle",
    "ExperienceRating",
    "ExposureRating",
    "RateAnalyzer",
    "FourierTransform",
    "__version__",
]


def __getattr__(name: str) -> Any:
    """Lazy-load the Tkinter GUI so headless imports never require _tkinter."""
    if name == "UnderwritingApp":
        try:
            from .app import UnderwritingApp as _UnderwritingApp
        except ImportError as exc:
            raise ImportError(
                "UnderwritingApp requires Tkinter GUI extras, which are not "
                "available in this environment (missing _tkinter). Install a "
                "Python build with Tk support, or use the headless C++ APIs "
                "(FactorModel, MonteCarloSimulator, FourierTransform, etc.)."
            ) from exc
        return _UnderwritingApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(list(__all__) + ["__getattr__", "__dir__"])
