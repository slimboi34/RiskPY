# Contributing to RiskPY

## Quick start

```bash
git clone https://github.com/slimboi34/RiskPY.git
cd RiskPY
make install
make check
```

## Workflow

1. Branch from `main`
2. Make changes (C++ under `src/`, Python under `riskpy/`, tests under `tests/`)
3. `make test` (or `pytest tests/`) and `make verify` (or `python -m riskpy.verify`)
4. Open a PR → **CI** must be green, including the verification suite
5. Maintainers merge to `main`; releases use `make release` + GitHub Release (see [docs/PUBLISHING.md](docs/PUBLISHING.md))

## Layout

| Path | Purpose |
|------|---------|
| `src/` | C++ core + pybind11 bindings |
| `riskpy/` | Python package: `mc`, `viz`, `quant`, `life`, `reserving`, `rates`, `credit`, `verify`, `_special`, the GUI |
| `tests/` | pytest suite |
| `.github/workflows/` | CI + PyPI publish |

## Style notes

- Validate actuarial edge cases (non-finite, empty, negative bounds) in C++ with `std::invalid_argument`
- Prefer tests that assert exact hand-checked math over soft tolerances where possible
- Keep `import riskpy` free of Tkinter side effects, and free of NumPy — the modelling layers import it lazily
- Never import SciPy inside `riskpy/`; it is a test-time oracle only. What you need is in `riskpy._special`
- When you add a formula, add an identity for it to the module's `_verification_checks()` hook — something that would catch a class of error, not one typo
