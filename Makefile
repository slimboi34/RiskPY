# RiskPY — one-command developer UX
#   make install   → editable install + dev deps
#   make test      → pytest
#   make build     → sdist + wheel
#   make release   → tag + push (triggers PyPI workflow)

.PHONY: help install test build clean smoke release check

PYTHON ?= python3
PKG    := open-riskpy
VERSION := $(shell $(PYTHON) -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])" 2>/dev/null || sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)

help:
	@echo "RiskPY developer commands"
	@echo "  make install   Editable install with [dev] extras"
	@echo "  make test      Run pytest suite"
	@echo "  make smoke     Import + tiny API check"
	@echo "  make build     Build sdist + wheel into dist/"
	@echo "  make clean     Remove build artifacts"
	@echo "  make check     install + test + smoke"
	@echo "  make release   Create annotated tag v$(VERSION) and push (PyPI)"
	@echo ""
	@echo "End users:  pip install $(PKG)"

install:
	$(PYTHON) -m pip install --upgrade pip
	CMAKE_BUILD_PARALLEL_LEVEL=$$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 2) \
	  $(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

smoke:
	$(PYTHON) -c "import riskpy; from riskpy import FactorModel, FourierTransform; \
m=FactorModel(100); m.add_multiplier('s','A',1.5); assert m.calculate({'s':'A'})==150; \
print('OK riskpy', riskpy.__version__)"

build:
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build

clean:
	rm -rf build dist *.egg-info _skbuild .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

check: install test smoke
	@echo "All checks passed for $(PKG) $(VERSION)"

# Creates git tag vX.Y.Z and pushes it + main. GitHub Release / workflow_dispatch
# publish.yml then uploads to PyPI via Trusted Publishing.
release:
	@echo "Releasing $(PKG) v$(VERSION)"
	@test -z "$$(git status --porcelain)" || (echo "Working tree not clean — commit first"; exit 1)
	git tag -a "v$(VERSION)" -m "Release v$(VERSION)"
	git push origin main
	git push origin "v$(VERSION)"
	@echo "Tag v$(VERSION) pushed. Create a GitHub Release from the tag to publish to PyPI:"
	@echo "  gh release create v$(VERSION) --generate-notes"
	@echo "Or: Actions → Publish to PyPI → Run workflow"
