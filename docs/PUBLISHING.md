# Publishing RiskPY (GitHub + PyPI)

Goal: anyone in the world can run:

```bash
pip install open-riskpy
```

This repo uses **maturin** (the Rust core, through PyO3) + **GitHub Actions Trusted Publishing** so releases are mostly automated.

---

## Super-short path (maintainers)

```bash
# 1. Bump [project].version in pyproject.toml AND [workspace.package].version in
#    rust/Cargo.toml (cargo check updates rust/Cargo.lock) + add a RELEASE_NOTES.md entry
# 2. Land it on main
git add -A && git commit -m "release: vX.Y.Z" && git push origin main

# 3. Tag + push — this is the whole release
make release

# 4. Verify
pip install -U open-riskpy
python -c "import riskpy; print(riskpy.__version__)"
```

`make release` tags `vX.Y.Z` from `pyproject.toml` and pushes `main` + the tag.
**The tag push is the only trigger.** It builds the sdist and wheels, uploads to
PyPI via Trusted Publishing, and creates the GitHub Release with every artifact
attached. There is no manual `gh release create` step.

To rehearse without uploading: **Actions → Publish to PyPI → Run workflow**,
leaving `publish_pypi` off. That builds and runs every check, uploading nothing.

---

## Pre-publish gates

The workflow refuses to upload a broken release. In order:

| Gate | Catches |
|------|---------|
| `guard` — tag vs `pyproject` version | Tagging `v0.2.8` with `version = "0.2.7"`. Fails in seconds, before the wheel matrix runs. |
| `guard` — PyPI version probe | Warns when the version is already published (upload would be a no-op). |
| sdist install + import from a temp dir | An sdist missing part of `rust/` — the classic "wheels fine, source install broken" bug. The install compiles the Rust core, as pip does on a platform without a wheel. |
| `twine check --strict` | Malformed README/metadata that PyPI would reject *after* a 15-minute build. |
| wheel install + import (native builds) | A wheel that builds but cannot import its own compiled extension. |

### Wheel coverage

| Job | Targets | Blocks release? |
|-----|---------|-----------------|
| `build-wheels` | linux/x86_64 (manylinux), macos/arm64, windows/AMD64 — one `cp310-abi3` wheel each | **Yes** |
| `build-wheels-extra` | linux/aarch64 (manylinux, native ARM runner), macos/x86_64 (cross-compiled) | No — best effort |

The upload waits for `build-wheels-extra` to finish, so its wheels ship whenever they
build; if one of those jobs fails, the release goes out without it.

The core is built on CPython's stable ABI (`abi3`, Python 3.10 and later), so each
platform needs one wheel, which serves every supported Python. The matrix lives in
`publish.yml` itself (PyO3/maturin-action); `make wheels` builds this platform's
wheel the same way (`maturin build --release`).

---

## One-time setup

### 1. GitHub repo
Already: `https://github.com/slimboi34/RiskPY` · default branch `main`.

### 2. PyPI Trusted Publishing (no long-lived API tokens)

1. Log in at [pypi.org](https://pypi.org) (2FA required).
2. Open [open-riskpy publishing settings](https://pypi.org/manage/project/open-riskpy/settings/publishing/)  
   (or create the project first by uploading once if it did not exist).
3. **Add a new pending publisher**:
   - **PyPI project name:** `open-riskpy`
   - **Owner:** `slimboi34`
   - **Repository:** `RiskPY`
   - **Workflow name:** `publish.yml`
   - **Environment name:** *(leave blank — the workflow does not use a GitHub Environment)*

OIDC (`id-token: write`) in `.github/workflows/publish.yml` does the rest.

### 3. Optional TestPyPI
Add a second trusted publisher against `test.pypi.org`, then:

**Actions → Publish to PyPI → Run workflow → publish_testpypi = true**

---

## What the workflows do

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| **CI** (`.github/workflows/ci.yml`) | push/PR to `main` | Build the wheel, install it, `pytest` + `riskpy.verify --no-oracle` on Ubuntu/macOS/Windows × Python 3.10–3.13; `cargo test` + clippy on the Rust core; headless import check |
| **Verify** (`.github/workflows/verify.yml`) | nightly, or manual dispatch | Verification suite with the SciPy oracle and benchmarks; JSON and Markdown report kept as an artifact |
| **Docs** (`.github/workflows/docs.yml`) | push to `main` touching `docs/`, `riskpy/` or `mkdocs.yml` | Regenerate the chart gallery and verification report, build the site, deploy GitHub Pages |
| **Publish** (`.github/workflows/publish.yml`) | push of a `v*` tag, or manual dispatch | Version guard → **sdist** (installed and smoke-imported) + **abi3 wheels** (import-tested where native) → `twine check` → PyPI (Trusted Publishing + attestations) → GitHub Release |

Wheels mean end users usually **do not need a Rust toolchain**. Source installs (the sdist) work wherever one is installed ([rustup](https://rustup.rs)).

---

## Local commands

```bash
make install   # pip install ".[dev]" (builds the Rust core)
make rust      # cargo test + clippy on the Rust core
make test      # pytest
make smoke     # import + tiny FactorModel check
make check     # install + test + verify + smoke
make build     # python -m build → dist/ (sdist + this platform's wheel)
make wheels    # maturin build --release → wheelhouse/
make clean
```

Manual PyPI upload (fallback only — prefer Actions):

```bash
python -m build
twine upload dist/*
# user: __token__
# pass: pypi-...
```

---

## Versioning

- `pyproject.toml` → `[project].version` is the release version (the `guard` job checks the tag against it)
- `rust/Cargo.toml` → `[workspace.package].version` must match it: the compiled core reports it as
  `riskpy.cpp_underwriter.__version__`, and `riskpy-doctor` warns when the two differ
- Tag format: `v0.2.6` (leading `v`)
- After a version is on PyPI it is **immutable** — always bump for the next release
- Document changes in `RELEASE_NOTES.md`

---

## End-user install

```bash
pip install open-riskpy              # the compiled core (no heavy deps)
pip install "open-riskpy[sim]"       # + NumPy: riskpy.mc and the numeric layers
pip install "open-riskpy[viz]"       # + Matplotlib: the riskpy.viz charts
pip install "open-riskpy[gui]"       # + matplotlib/numpy for the UnderwritingApp GUI
pip install -U open-riskpy           # upgrade
```

From source (needs a Rust toolchain from [rustup](https://rustup.rs); pip fetches maturin itself):

```bash
git clone https://github.com/slimboi34/RiskPY.git
cd RiskPY
pip install .
# or: make install
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| CI cannot find `cargo` | The job needs the `dtolnay/rust-toolchain` step before anything that builds the package |
| Publish fails with 403 Trusted Publishing | Re-check publisher owner/repo/workflow/environment names |
| Tests import the wrong `riskpy` | The suite runs against the installed package; re-run `pip install .` (or `make install`) after Rust or Python edits |
| GUI import error headless | Expected — use the core APIs; Tk is lazy-loaded |
| `test_app_batch` is skipped | Expected without Tkinter or the `[gui]` extra. The publish workflow runs no pytest — wheels get an import smoke test only |
