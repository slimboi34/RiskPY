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
3. `make test` (or `pytest tests/`)
4. Open a PR → **CI** must be green
5. Maintainers merge to `main`; releases use `make release` + GitHub Release (see [docs/PUBLISHING.md](docs/PUBLISHING.md))

## Layout

| Path | Purpose |
|------|---------|
| `src/` | C++ core + pybind11 bindings |
| `riskpy/` | Python package (`__init__.py`, GUI) |
| `tests/` | pytest suite |
| `.github/workflows/` | CI + PyPI publish |

## Style notes

- Validate actuarial edge cases (non-finite, empty, negative bounds) in C++ with `std::invalid_argument`
- Prefer tests that assert exact hand-checked math over soft tolerances where possible
- Keep `import riskpy` free of Tkinter side effects
