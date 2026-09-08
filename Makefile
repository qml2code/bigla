# bigla — development Makefile
# Requires: pip install -e ".[dev]"

.DEFAULT_GOAL := help

PYTHON  ?= python
PIP     ?= pip
PYTEST  ?= pytest

.PHONY: help install install-openblas test test-long test-huge test-all dev-env dev-setup \
        conventional-commits review diagnose matrix matrix-row backends-table clean build

help:
	@echo "bigla development targets:"
	@echo ""
	@echo "  install          pip install -e .[dev]"
	@echo "  install-openblas pip install -e .[dev,openblas]  (pulls scipy-openblas64)"
	@echo ""
	@echo "  test             fast suite (excludes the long/huge markers)"
	@echo "  test-long        container matrix + large-n workspace checks"
	@echo "  test-huge        run huge-matrix test (~20 GiB)"
	@echo "  dev-setup        install the pre-commit hooks (one-time, one click)"
	@echo "  conventional-commits  also install the commit-msg hook"
	@echo "  review           run every hook over the whole tree"
	@echo "  diagnose         python -m bigla.diagnose"
	@echo ""
	@echo "  matrix           build + run every environment-matrix row (needs docker)"
	@echo "  matrix-row       one row:  make matrix-row ROW=debian-openblas64"
	@echo "  backends-table   regenerate docs/backends.md from out/*.json"
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

# Selection is by DIRECTORY, and tests/conftest.py declines to collect tests/long and
# tests/huge unless the invocation NAMES them. So these three targets are just the three
# directories -- no markers, no environment variables, no --ignore flags -- and a bare
# `pytest` runs the fast suite rather than a 20 GiB surprise.
test:                 # seconds; the fast suite only
	$(PYTEST) tests/ --tb=short -v

test-long:            # minutes; environment-matrix rows + large-n workspace checks
	$(PYTEST) tests/long -v -s

test-huge:            # ~20 GiB
	$(PYTEST) tests/huge -v -s

test-all: test test-long test-huge

# ---------------------------------------------------------------------------
# Environment matrix (FIXES.md §6)
#
# The rows and their expectations are declared ONCE, in .github/workflows/backends.yml.
# These targets exist to reproduce a row locally when CI reports a decoration that
# disagrees with what the row expected -- which is the whole point of the matrix, and
# something you want to debug on a laptop rather than by pushing commits.
# ---------------------------------------------------------------------------

ROW ?= pip-numpy-wheel
DOCKER ?= docker

matrix-row:
	$(PYTHON) tools/matrix.py run --row "$(ROW)" --docker "$(DOCKER)"

matrix:
	$(PYTHON) tools/matrix.py run --all --docker "$(DOCKER)"

backends-table:
	$(PYTHON) tools/render_backends.py out/

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
