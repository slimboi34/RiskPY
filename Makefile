# RiskPY — one-command developer UX
#   make install   → editable install + dev deps
#   make test      → pytest
#   make build     → sdist + wheel
#   make release   → tag + push (triggers PyPI workflow)

.PHONY: help install test verify docs build clean smoke release check wheels

PYTHON ?= python3
PKG    := open-riskpy
VERSION := $(shell $(PYTHON) -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])" 2>/dev/null || sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)

help:
	@echo "RiskPY developer commands"
	@echo "  make install   Editable install with [dev] extras"
	@echo "  make test      Run pytest suite"
	@echo "  make verify    Run the verification suite (every identity, checked)"
	@echo "  make docs      Render the gallery and build the docs site"
	@echo "  make smoke     Import + tiny API check"
	@echo "  make build     Build sdist + wheel into dist/"
	@echo "  make clean     Remove build artifacts"
	@echo "  make wheels    Build the CI wheel matrix locally (cibuildwheel)"
	@echo "  make check     install + test + verify + smoke"
	@echo "  make release   Create annotated tag v$(VERSION) and push (PyPI)"
	@echo ""
	@echo "End users:  pip install $(PKG)"

install:
	$(PYTHON) -m pip install --upgrade pip
	CMAKE_BUILD_PARALLEL_LEVEL=$$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 2) \
	  $(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

verify:
	$(PYTHON) -m riskpy.verify --bench

docs:
	$(PYTHON) docs/generate_gallery.py
	$(PYTHON) -m mkdocs build --strict

smoke:
	$(PYTHON) -c "import riskpy; from riskpy import FactorModel, FourierTransform; \
m=FactorModel(100); m.add_multiplier('s','A',1.5); assert m.calculate({'s':'A'})==150; \
print('OK riskpy', riskpy.__version__)"

build:
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build

clean:
	rm -rf build dist wheelhouse site docs/assets *.egg-info _skbuild .pytest_cache
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

# Dry run of the exact wheel matrix CI uses (config lives in pyproject.toml).
wheels:
	pipx run cibuildwheel==4.1.1 --output-dir wheelhouse
