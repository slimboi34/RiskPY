"""RiskPY — a risk and actuarial engine: compiled core, readable Python on top.

Two layers, deliberately separated by what they cost you to install.

The **core** is C++ compiled through pybind11 and has no Python dependencies at
all: factor rating, loss triangles, Fourier aggregate loss, exposure and
experience rating, Monte Carlo, Excel export. ``import riskpy`` on a machine
with nothing else installed works.

The **modelling layers** are ordinary Python you can read, and each is imported
only when you touch it:

===================  =========================================================
:mod:`riskpy.mc`     generic Monte Carlo — any formula, any distributions,
                     rank correlation, Latin hypercube, twenty distributions
                     with a full analytic layer
:mod:`riskpy.viz`    the charts, one theme, light and dark
:mod:`riskpy.quant`  option pricing, Greeks, paths, portfolio risk
:mod:`riskpy.life`   life contingencies — tables, annuities, premiums, reserves
:mod:`riskpy.reserving`  claims reserving — chain ladder, Mack, Bornhuetter–
                     Ferguson, Cape Cod, bootstrap
:mod:`riskpy.rates`  curves, bonds, durations, short-rate models
:mod:`riskpy.credit` credit risk — Merton, hazard rates, CDS, Basel, portfolios
:mod:`riskpy.verify` the verification suite — every identity, checked
===================  =========================================================

    pip install open-riskpy          # core, zero dependencies
    pip install open-riskpy[sim]     # + NumPy: simulation and the numeric layers
    pip install open-riskpy[viz]     # + Matplotlib: the charts
"""

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
    # Compiled core.
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
    # Pure-Python layers, lazily imported below.
    "mc",
    "viz",
    "quant",
    "life",
    "reserving",
    "rates",
    "credit",
    "verify",
    "__version__",
]

# Submodules resolved on first attribute access rather than imported here.
# Most of them depend on NumPy, which is an optional extra — importing them
# eagerly would make `import riskpy` fail on a lean install, which is exactly
# the property the zero-dependency core exists to protect.
_LAZY_SUBMODULES = {
    "mc",
    "viz",
    "quant",
    "life",
    "reserving",
    "rates",
    "credit",
    "verify",
}


def __getattr__(name: str) -> Any:
    """Lazy-load optional submodules and the Tkinter GUI."""
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f".{name}", __name__)

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
