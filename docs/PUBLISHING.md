# Publishing RiskPY (GitHub + PyPI)

Goal: anyone in the world can run:

```bash
pip install open-riskpy
```

This repo uses **scikit-build-core** + **GitHub Actions Trusted Publishing** so releases are mostly automated.

---

## Super-short path (maintainers)

```bash
# 1. Bump [project].version in pyproject.toml + add a RELEASE_NOTES.md entry
# 2. Land it on main
git add -A && git commit -m "release: v0.2.7" && git push origin main

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
| sdist install + import from a temp dir | An sdist missing `src/` or `CMakeLists.txt` — the classic "wheels fine, source install broken" bug. |
| `twine check --strict` | Malformed README/metadata that PyPI would reject *after* a 15-minute build. |
| cibuildwheel `test-command` | A wheel that builds but cannot import its own C++ extension. |

### Wheel coverage

| Job | Targets | Blocks release? |
|-----|---------|-----------------|
| `build-wheels` | linux/x86_64, macos/arm64, windows/AMD64 — cp310–cp314 | **Yes** |
| `build-wheels-extra` | linux/aarch64 (native ARM runner), macos/x86_64 (cross) | No — best effort |

The matrix itself lives in `pyproject.toml` under `[tool.cibuildwheel]`, so CI
and local builds cannot drift. Reproduce it locally with `make wheels`.

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
| **CI** (`.github/workflows/ci.yml`) | push/PR to `main` | Build + `pytest` + `riskpy.verify --no-oracle` on Ubuntu/macOS/Windows × Python 3.10–3.13; headless import check |
| **Verify** (`.github/workflows/verify.yml`) | nightly, or manual dispatch | Verification suite with the SciPy oracle and benchmarks; JSON and Markdown report kept as an artifact |
| **Docs** (`.github/workflows/docs.yml`) | push to `main` touching `docs/`, `riskpy/` or `mkdocs.yml` | Regenerate the chart gallery and verification report, build the site, deploy GitHub Pages |
| **Publish** (`.github/workflows/publish.yml`) | push of a `v*` tag, or manual dispatch | Version guard → **sdist** (installed and smoke-imported) + **cp310–cp314 wheels** (import-tested) → `twine check` → PyPI (Trusted Publishing + attestations) → GitHub Release |

Wheels mean end users usually **do not need a C++ compiler**. Source installs still work when a compiler is available.

---

## Local commands

```bash
make install   # pip install -e ".[dev]"
make test      # pytest
make smoke     # import + tiny FactorModel check
make check     # install + test + smoke
make build     # python -m build → dist/
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

- Single source of truth: `pyproject.toml` → `[project].version`
- Tag format: `v0.2.6` (leading `v`)
- After a version is on PyPI it is **immutable** — always bump for the next release
- Document changes in `RELEASE_NOTES.md`

---

## End-user install

```bash
pip install open-riskpy              # core C++ APIs (no heavy deps)
pip install "open-riskpy[gui]"       # + matplotlib/numpy for GUI charts
pip install -U open-riskpy           # upgrade
```

From source (needs C++17 + CMake):

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
| CI cannot find compiler | Workflow installs `g++` / Xcode tools; ensure `cmake`/`ninja` steps ran |
| Publish fails with 403 Trusted Publishing | Re-check publisher owner/repo/workflow/environment names |
| `FileNotFoundError: cmake` on import | Editable auto-rebuild is off; re-run `pip install -e .` after C++ edits |
| GUI import error headless | Expected — use C++ APIs; Tk is lazy-loaded |
| Wheel test fails on `test_app_batch` | Publish workflow ignores GUI batch test; core tests still run |
