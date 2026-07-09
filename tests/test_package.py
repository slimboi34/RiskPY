import riskpy
from riskpy import (
    ActuarialMath,
    FactorModel,
    FourierTransform,
    LossTriangle,
    MonteCarloSimulator,
    RateAnalyzer,
    RiskEngine,
)


def test_headless_import_succeeds():
    # Importing the package must not require Tkinter
    assert riskpy.FactorModel is FactorModel
    assert riskpy.FourierTransform is FourierTransform


def test_version_string():
    assert isinstance(riskpy.__version__, str)
    assert len(riskpy.__version__) > 0


def test_all_exports_present():
    for name in riskpy.__all__:
        if name == "UnderwritingApp":
            continue  # may be lazy / unavailable headless
        assert hasattr(riskpy, name), f"missing export: {name}"


def test_dir_includes_public_api():
    names = dir(riskpy)
    assert "FactorModel" in names
    assert "FourierTransform" in names
    assert "__version__" in names


def test_lazy_gui_import_error_or_success():
    # On machines with Tk this returns the class; without Tk it raises ImportError
    # with a clear message (never a cryptic _tkinter crash at import time).
    try:
        cls = riskpy.UnderwritingApp
        assert callable(cls) or isinstance(cls, type)
    except ImportError as exc:
        msg = str(exc).lower()
        assert "tkinter" in msg or "gui" in msg or "underwritingapp" in msg.lower()
