# RiskPY v0.2.4 — Release Notes

**Release date:** 2026-05-01
**Type:** Patch / bug-fix release
**Compatibility:** Drop-in replacement for v0.2.3. No public API changes, no schema changes, no rebuild of native extension required for users on `pip install --upgrade open-riskpy`.

---

## TL;DR

Three bug fixes and a cross-platform-hygiene tweak. **The most important is a crash fix in the Tkinter Monte Carlo simulation tab** — anyone who clicked "Run C++ Monte Carlo" in the GUI on v0.2.3 was hitting a `TypeError`. Upgrade if you ship the GUI to end-users.

```bash
pip install --upgrade open-riskpy
```

---

## Fixed

### 1. GUI Monte Carlo simulation tab crashed on every run *(critical)*

**File:** `riskpy/app.py` — `_render_simulation_tab.run_sim()`

**Symptom on v0.2.3:** Pressing the **"Run C++ Monte Carlo"** button raised
```
TypeError: simulate_aggregate_loss(): incompatible function arguments.
The following argument types are supported:
    1. (self: cpp_underwriter.MonteCarloSimulator, expected_frequency: float, ...)
```
…and the histogram never rendered.

**Root cause:** The handler was calling `MonteCarloSimulator.simulate_aggregate_loss(trials, freq, mu, sigma)` as if it were a four-argument static method. After the v0.2.3 OOP refactor, `trials` is a constructor argument on the simulator and `simulate_aggregate_loss` is an instance method that takes `(expected_frequency, expected_severity_mu, severity_sigma)`. pybind11 correctly rejected the call.

**Fix:** Instantiate the simulator once per click, then call the instance method:
```python
sim = MonteCarloSimulator(trials=trials)
losses = sim.simulate_aggregate_loss(freq, mu, sigma)
```

**Impact:** The Monte Carlo Predictor tab in `UnderwritingApp.run()` now works end-to-end again — Poisson × Lognormal aggregate-loss histogram and the 99% VaR overlay both render.

---

### 2. `test_headless.py` was unrunnable *(broken since OOP refactor)*

**File:** `test_headless.py`

**Symptom:** `ModuleNotFoundError: No module named 'underwriter_app'`. The smoke test referenced a pre-v0.2.3 module name that was renamed when the package was reorganized into `riskpy/`.

**Fix:** Updated the import to `from riskpy import UnderwritingApp`. The test once again exercises the full `Python callable → C++ engine → Excel exporter` round-trip (asserts `premium == 1500` and that `test_policy_quote.xlsx` is generated).

---

### 3. CSV batch import — Windows line-ending handling

**File:** `riskpy/app.py` — `UnderwritingApp.calculate_batch()`

**Symptom:** Batch CSVs authored on Windows (`\r\n` line endings) could surface phantom blank rows or trailing `\r` in the last column, depending on how `csv.DictReader` interpreted the buffered I/O on the host platform.

**Fix:** Pass `newline=''` to `open()` per the Python `csv` module's documented requirement. This delegates universal-newline handling to the `csv` module instead of the file object, which is the correct contract for `csv.DictReader` and resolves the inconsistency.

```python
with open(csv_filepath, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    ...
```

**Impact:** Batch quotes from Windows-authored CSVs now parse identically to those authored on macOS/Linux. No behavior change for users who were already on Unix-only pipelines.

---

## Unchanged

- **C++ engines.** No changes to `ActuarialMath`, `FactorModel`, `MonteCarloSimulator`, `LossTriangle`, `ExperienceRating`, `ExposureRating`, `RateAnalyzer`, or `RiskEngine`. All v0.2.3 OOP regression tests in `tests/test_oop_regression.py` continue to pass.
- **Python public API.** `UnderwritingApp`, exposed `cpp_underwriter` symbols, and the `from riskpy import …` surface are byte-identical.
- **Build pipeline.** `pyproject.toml` build backend remains `scikit-build-core` with `pybind11>=2.11.0`. CMakeLists.txt fetches `pybind11 v2.11.1` and OpenXLSX as before. CI publish workflow (`.github/workflows/publish.yml`) builds an sdist and uploads via PyPI trusted publishing on `release: published`.

---

## Verification

```bash
pip install -e ".[dev]"
pytest tests/                  # full regression suite (engines + bounds + OOP baseline)
python test_headless.py        # round-trip smoke test (Python logic → C++ engine → Excel)
python test_batch.py           # 100-policy batch processing
python main.py                 # dual-tab GUI; Monte Carlo Predictor tab now functional
```

All previously passing tests remain green and `test_headless.py` is restored to a runnable state.

---

## Upgrade guidance

| User type | Action |
|---|---|
| GUI users (any `app.run()` consumer) | **Upgrade required** — Monte Carlo tab is otherwise broken |
| Headless / batch only | Optional but recommended (Windows CSV correctness) |
| C++ engine consumers | No-op upgrade |

---

## Diff stats

```
README.md        | 2 ++
pyproject.toml   | 2 +-
riskpy/app.py    | 8 ++++----
test_headless.py | 2 +-
4 files changed, 8 insertions(+), 6 deletions(-)
```

Full commit: [`2dc4e44`](https://github.com/slimboi34/RiskPY/commit/2dc4e44) — `fix: GUI Monte Carlo dispatch, broken headless test, CSV newlines (v0.2.4)`
