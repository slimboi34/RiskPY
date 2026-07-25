# RiskPY v0.2.7 — Release Notes

**Release date:** 2026-07-25  
**Type:** Packaging + release-pipeline release  
**Install:** `pip install -U open-riskpy`

---

## TL;DR

Correct package metadata, wheels for **Python 3.14** and **Linux aarch64**, and a
release pipeline that verifies what it ships before it ships it. No API changes —
upgrading is safe.

```bash
pip install -U open-riskpy
python -c "import riskpy; print(riskpy.__version__)"
```

---

## Fixed

- **Broken author metadata.** `pyproject.toml` shipped a literal placeholder
  (`[EMAIL_ADDRESS]`, no `@`) as the author email. v0.2.6 on PyPI carries it;
  0.2.7 has a real address.
- **`requires-python` claimed more than we shipped.** It said `>=3.8` while
  wheels covered only cp310–cp313, so pip on 3.8/3.9 silently fell through to a
  source build needing a full C++ toolchain. Now `>=3.10`, matching the wheels.
- **Stale pybind11 floor.** The CMake `FetchContent` fallback pinned v2.11.1,
  which predates Python 3.13 support, so a build without a pip-installed
  pybind11 could not target 3.13/3.14. Floor and fallback are now 2.12 / v3.0.4.
- **`CMAKE_CXX_COMPILER_LAUNCHER=ccache` was set on all CI platforms**, including
  the Windows image, which has no ccache.

## Added

- **cp314 wheels** — Python 3.14 is supported and classified.
- **linux/aarch64 wheels**, built on a native ARM runner rather than QEMU.
- **macos/x86_64 wheels** (cross-compiled) alongside arm64.
- **Windows CI** — the platform we ship wheels for is now actually tested.
- `make wheels` — reproduce the full CI wheel matrix locally.

## Changed

- **Release is one command.** `make release` tags and pushes; the tag push
  builds, publishes to PyPI, and cuts the GitHub Release with artifacts
  attached. Previously the tag push did nothing and a GitHub Release had to be
  created by hand — despite the workflow comments claiming otherwise.
- **New pre-publish gates**, so a bad release fails before upload, not after:
  - tag-vs-`pyproject` version check (fails in seconds, not after a 15-min matrix)
  - the sdist is installed and imported from a temp dir, proving it is self-contained
  - `twine check --strict` on every artifact
  - a warning if the version is already on PyPI
- **Best-effort extra architectures.** linux/aarch64 and macos/x86_64 cannot
  block a release; core linux/macos/windows wheels still must pass.
- **cibuildwheel 2.22 → 4.1.1.** Brings `manylinux_2_28`, automatic
  `delvewheel` DLL bundling on Windows, and `abi3audit` checks. The old
  `yum || apt` compiler shim is gone — the modern image already has C++17 and
  scikit-build-core injects cmake/ninja.
- **Wheel matrix moved into `pyproject.toml`** (`[tool.cibuildwheel]`) so local
  builds and CI cannot drift apart.
- **PEP 639 licensing** — `License-Expression: MIT` (metadata 2.4) replaces the
  deprecated `license = { text = ... }` table.
- **PyPI attestations** (PEP 740) are now published with each upload.
- Explicit `MACOSX_DEPLOYMENT_TARGET=11.0` for reproducible macOS wheel tags.

---

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
