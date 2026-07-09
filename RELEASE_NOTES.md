# RiskPY v0.2.6 — Release Notes

**Release date:** 2026-07-09  
**Type:** Feature + performance + packaging release  
**Install:** `pip install -U open-riskpy`

---

## TL;DR

Faster core math, headless-safe imports, new **FourierTransform** analytic aggregate-loss engine, stronger validation, expanded tests, and a push-button **GitHub Actions → PyPI** path (CI + wheels + Trusted Publishing).

```bash
pip install -U open-riskpy
python -c "import riskpy; from riskpy import FourierTransform; print(riskpy.__version__)"
```

---

## Added

### FourierTransform (C++)
- `fft` / `ifft` — iterative radix-2 Cooley–Tukey with zero-pad to power of 2  
- `convolve` — fast linear convolution  
- `compound_poisson_pmf` — aggregate loss PMF via $\hat{S}(t)=e^{\lambda(\hat{X}(t)-1)}$  
- Severity auto-normalized; non-finite inputs rejected; empty FFT returns empty  

### Packaging / DevX
- GitHub Actions **CI** (Ubuntu + macOS, Python 3.10–3.12)  
- **Publish** workflow: sdist + cibuildwheel wheels → PyPI Trusted Publishing  
- `Makefile` (`make install|test|smoke|build|release`)  
- `docs/PUBLISHING.md` rewritten for the one-command release path  

### Tests
- Full suites for Fourier, LossTriangle, RateAnalyzer, FactorModel extras, package headless import, RiskEngine  
- Suite size: **77** tests  

---

## Changed / Fixed

### Performance
- `RateAnalyzer.on_level_premiums`: O(n²) → **O(n)**  
- `LossTriangle.get_ultimate_losses`: O(years×periods) → **O(years)** with precomputed CDFs  
- `FactorModel.calculate`: field-keyed `unordered_map` lookup  
- `RiskEngine.get_fields`: return **const reference** (zero copy in C++)  

### Stability
- `import riskpy` no longer requires Tkinter (lazy `UnderwritingApp`)  
- Factor / rate / triangle / Fourier finite-value validation  
- OpenXLSX pinned to **v0.5.1**; shallow FetchContent; ccache-aware builds  
- Excel `create(..., XLForceOverwrite)` (non-deprecated API)  

---

## Prior releases

See git history and earlier sections of this file for v0.2.5 and below.  
*(v0.2.5 on PyPI was a packaging/CMake bump; v0.2.6 is the first full Fourier + perf ship.)*

---

# RiskPY v0.2.4 — Release Notes

**Release date:** 2026-05-01  
**Type:** Patch / bug-fix release  

### Fixed
- GUI Monte Carlo dispatch (`MonteCarloSimulator` instance method)  
- Headless smoke test  
- CSV batch newline handling  

```bash
pip install --upgrade open-riskpy
```
