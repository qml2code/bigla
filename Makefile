# bigla — development Makefile
# Requires: pip install -e ".[dev]"

.DEFAULT_GOAL := help

PYTHON  ?= python
PIP     ?= pip
PYTEST  ?= pytest

.PHONY: help install install-openblas test test-long test-huge test-all dev-env dev-setup \
        conventional-commits review diagnose clean build

help:
	@echo "bigla development targets:"
	@echo ""
	@echo "  install          pip install -e .[dev]"
	@echo "  install-openblas pip install -e .[dev,openblas]  (pulls scipy-openblas64)"
	@echo ""
	@echo "  test             fast suite (excludes the long/huge markers)"
	@echo "  test-long        container matrix + large-n workspace checks"
	@echo "  test-huge        run huge-matrix test (~20 GiB, needs BIGLA_TEST_HUGE=1)"
	@echo "  dev-setup        install the pre-commit hooks (one-time, one click)"
	@echo "  conventional-commits  also install the commit-msg hook"
	@echo "  review           run every hook over the whole tree"
	@echo "  diagnose         python -m bigla.diagnose"
	@echo ""
	@echo "  build            build wheel (py3-none-any)"
	@echo "  clean            remove build artefacts"

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-openblas:
	$(PYTHON) -m pip install -e ".[dev,openblas]"

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test:                 # seconds; everything except the `long` and `huge` markers
	$(PYTEST) tests/ -m "not huge and not long" --tb=short -v

test-long:            # minutes; container matrix + large-n workspace checks
	BIGLA_TEST_LONG=1 $(PYTEST) tests/ -m "long" -v -s

test-huge:            # ~20 GiB, unchanged
	BIGLA_TEST_HUGE=1 $(PYTEST) tests/test_huge.py -v -s

test-all: test test-long test-huge

# ---------------------------------------------------------------------------
# Code quality -- all of it lives in .pre-commit-config.yaml (shared with qml2-dev).
# There are deliberately no separate lint/fmt targets: duplicating black/isort/flake8
# arguments here is how a Makefile drifts out of sync with the hooks that actually gate
# a commit. Install the hooks once and formatting happens on every commit thereafter.
# ---------------------------------------------------------------------------

dev-env:
	$(PIP) install pre-commit

./.git/hooks/pre-commit: dev-env
	pre-commit install

./.git/hooks/commit-msg: dev-env
	pre-commit install --hook-type commit-msg

# One click: install the formatting/lint hooks.
dev-setup: dev-env ./.git/hooks/pre-commit

# Also enforce conventional commit messages (the commit-msg hook).
conventional-commits: ./.git/hooks/commit-msg

# Run every hook over the whole tree -- what you want before the first commit of a
# new repo, where the hooks have never fired.
review: dev-setup
	pre-commit run --all-files

# ---------------------------------------------------------------------------
# Diagnose
# ---------------------------------------------------------------------------

diagnose:
	$(PYTHON) -m bigla.diagnose

# ---------------------------------------------------------------------------
# Build / dist
# ---------------------------------------------------------------------------

build:
	$(PYTHON) -m build --wheel
	@echo "Wheel written to dist/"
	@ls -lh dist/*.whl 2>/dev/null || true

clean:
	rm -rf dist/ build/ *.egg-info bigla/__pycache__ tests/__pycache__ .mypy_cache .ruff_cache
