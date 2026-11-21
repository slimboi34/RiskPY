# Contributing to RiskPY

## Quick start

The compiled core is Rust, built by [maturin](https://www.maturin.rs/) through PyO3, so
building from source needs a Rust toolchain (stable, 1.89 or later) and the system
linker rustup asks for (Xcode Command Line Tools on macOS, the C toolchain on Linux,
the MSVC Build Tools on Windows). No CMake, no C++ libraries to fetch.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # once: install Rust via rustup
git clone https://github.com/slimboi34/RiskPY.git
cd RiskPY
make install
make check
```

`make install` builds the wheel and installs it with the `[dev]` extras
(`pip install ".[dev]"`); pip fetches maturin, the build backend, by itself.

The test suite always runs against the **installed** package: `tests/conftest.py`
removes the repository root from `sys.path`, so the uncompiled `riskpy/` folder can
never shadow the build. Re-run `make install` after changing Rust or Python code.
For a quick loop on the Rust side, `pip install maturin` and `maturin develop` build
and install into the active virtualenv; run the suite from an installed wheel
(`maturin build --release --out dist && pip install dist/*.whl`) before you push.

## Workflow

1. Branch from `main`
2. Make changes (Rust under `rust/`, Python under `riskpy/`, tests under `tests/`)
3. `make test` (or `pytest tests/`) and `make verify` (or `python -m riskpy.verify`);
   after Rust changes also `make rust` (`cargo test -p riskpy-core` and clippy with
   warnings as errors)
4. Open a PR → **CI** must be green, including the verification suite and the Rust job
5. Maintainers merge to `main`; releases use `make release` + GitHub Release (see [docs/PUBLISHING.md](docs/PUBLISHING.md))

## Layout

| Path | Purpose |
|------|---------|
| `rust/core/` | `riskpy-core`: the compiled core in plain Rust (rating, triangles, FFT, Monte Carlo with its own PCG64-DXSM stream and samplers, the .xlsx writer); no Python in it |
| `rust/python/` | `riskpy-python`: the PyO3 bindings, built as the extension module `riskpy.cpp_underwriter` |
| `riskpy/` | Python package: `mc`, `viz`, `quant`, `life`, `reserving`, `rates`, `credit`, `verify`, `_special`, the GUI |
| `tests/` | pytest suite |
| `.github/workflows/` | CI, nightly verification, docs site, PyPI publish |

## Style notes

- Validate actuarial edge cases (non-finite, empty, negative bounds) in the Rust core and
  return `Error::Value` (a `ValueError` in Python). Write the checks as `!(x >= 0.0)`
  so NaN fails them too
- Transcendental functions in the core go through `riskpy_core::m` (the `libm` crate),
  never `f64::exp` and friends, so a seed gives the same numbers on every platform
- `cargo clippy --all-targets -- -D warnings` must be clean
- Prefer tests that assert exact hand-checked math over soft tolerances where possible
- Keep `import riskpy` free of Tkinter side effects, and free of NumPy — the modelling layers import it lazily
- Never import SciPy in the library modules; it is an oracle only, used by the tests and by the optional oracle checks in `riskpy.verify`. What you need is in `riskpy._special`
- When you add a formula, add an identity for it to the verification suite — something that would catch a class of error, not one typo. `life`, `reserving`, `rates` and `credit` contribute through their `_verification_checks()` hook; checks for the core, `_special`, `mc`, `quant` and `viz` live in `riskpy/verify.py`
