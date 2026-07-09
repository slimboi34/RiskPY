# Publishing RiskPY (GitHub + PyPI)

Goal: anyone in the world can run:

```bash
pip install open-riskpy
```

This repo uses **scikit-build-core** + **GitHub Actions Trusted Publishing** so releases are mostly automated.

---

## Super-short path (maintainers)

```bash
# 1. Land work on main
git add -A && git commit -m "feat: ..." && git push origin main

# 2. Bump version in pyproject.toml (already done for a release branch/commit)
# 3. Tag + push + open a GitHub Release
make release
gh release create v0.2.6 --generate-notes

# 4. Wait for Actions → "Publish to PyPI" (green)
# 5. Verify
pip install -U open-riskpy
python -c "import riskpy; print(riskpy.__version__)"
```

`make release` tags `vX.Y.Z` from `pyproject.toml` and pushes `main` + the tag.  
Creating the **GitHub Release** (or running the workflow manually) triggers PyPI upload.

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
| **CI** (`.github/workflows/ci.yml`) | push/PR to `main` | Build + `pytest` on Ubuntu/macOS × Python 3.10–3.12; headless import check |
| **Publish** (`.github/workflows/publish.yml`) | GitHub Release published, or manual dispatch | Build **sdist** + **wheels** (cibuildwheel: Linux/macOS/Windows) → upload to PyPI |

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
