# RiskPY — one-command developer UX
#   make install   → build + install with dev deps (compiles the Rust core: needs rustup)
#   make test      → pytest
#   make rust      → cargo test + clippy on the Rust core
#   make build     → sdist + wheel
#   make release   → tag + push (triggers PyPI workflow)

.PHONY: help install test rust verify docs build clean smoke release check wheels

PYTHON ?= python3
PKG    := open-riskpy
VERSION := $(shell $(PYTHON) -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])" 2>/dev/null || sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)

help:
	@echo "RiskPY developer commands"
	@echo "  make install   Build and install with [dev] extras (needs a Rust toolchain)"
	@echo "  make test      Run pytest suite"
	@echo "  make rust      Rust core: cargo test + clippy (warnings are errors)"
	@echo "  make verify    Run the verification suite (every identity, checked)"
	@echo "  make docs      Render the gallery and build the docs site"
	@echo "  make smoke     Import + tiny API check"
	@echo "  make build     Build sdist + wheel into dist/"
	@echo "  make clean     Remove build artifacts"
	@echo "  make wheels    Build this platform's release wheel into wheelhouse/ (maturin)"
	@echo "  make check     install + test + verify + smoke"
	@echo "  make release   Create annotated tag v$(VERSION) and push (PyPI)"
	@echo ""
	@echo "End users:  pip install $(PKG)"

# A regular (not editable) install: the suite runs against the installed package
# (tests/conftest.py takes the repository root off sys.path), and an editable maturin
# install works by putting the repository root on sys.path. Re-run after changes.
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

rust:
	cargo test --manifest-path rust/Cargo.toml -p riskpy-core
	cargo clippy --manifest-path rust/Cargo.toml --workspace --all-targets -- -D warnings

# verify and smoke run outside the checkout: from the repository root the uncompiled
# riskpy/ source folder would shadow the installed package.
verify:
	cd "$${TMPDIR:-/tmp}" && $(PYTHON) -m riskpy.verify --bench

docs:
	$(PYTHON) docs/generate_gallery.py
	$(PYTHON) -m mkdocs build --strict

smoke:
	cd "$${TMPDIR:-/tmp}" && $(PYTHON) -c "import riskpy; from riskpy import FactorModel, FourierTransform; \
m=FactorModel(100); m.add_multiplier('s','A',1.5); assert m.calculate({'s':'A'})==150; \
print('OK riskpy', riskpy.__version__)"

build:
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build

clean:
	rm -rf build dist wheelhouse site docs/assets *.egg-info .pytest_cache rust/target
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

check: install test verify smoke
	@echo "All checks passed for $(PKG) $(VERSION)"

# Creates git tag vX.Y.Z and pushes it + main. The tag push is the only trigger
# needed: publish.yml builds sdist + wheels, uploads to PyPI via Trusted
# Publishing, and creates the GitHub Release with all artifacts attached.
release:
	@echo "Releasing $(PKG) v$(VERSION)"
	@test -z "$$(git status --porcelain)" || (echo "Working tree not clean — commit first"; exit 1)
	@if git rev-parse -q --verify "refs/tags/v$(VERSION)" >/dev/null; then \
	  echo "Tag v$(VERSION) already exists — bump version in pyproject.toml"; exit 1; \
	fi
	git tag -a "v$(VERSION)" -m "Release v$(VERSION)"
	git push origin main
	git push origin "v$(VERSION)"
	@echo "Tag v$(VERSION) pushed — Actions will build, publish to PyPI, and cut the GitHub Release."
	@echo "  Watch: gh run watch \$$(gh run list -w 'Publish to PyPI' -L1 --json databaseId -q '.[0].databaseId')"

# This platform's release wheel, built as publish.yml builds it (one abi3 wheel per
# platform). The full wheel matrix lives in .github/workflows/publish.yml.
wheels:
	$(PYTHON) -m pip install --upgrade maturin
	$(PYTHON) -m maturin build --release --out wheelhouse
