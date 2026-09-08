"""Huge-matrix test — n = 46 500, just past the LP64 wall.

Opt-in by NAMING this directory: tests/conftest.py declines to collect it otherwise,
so a bare `pytest` never reaches these.
Requires ~20 GiB of free memory and a confirmed ILP64 backend.

Run with:
    make test-huge          # or: pytest tests/huge -v -s

This is the test that justifies the package.  Record the machine and
result in docs/backends.md after each run.
"""

from __future__ import annotations

import numpy as np
import pytest

import bigla
from bigla._backend import backend_info

# Skip unless explicitly requested


N_HUGE = 46_500  # just past the 46341 LP64 ceiling


def test_huge_cho_factor_solve():
    """Cholesky factor and solve for n=46500 — fails silently on LP64."""
    info = backend_info()
    if not info.ilp64:
        pytest.skip(f"ILP64 not available (backend: {info.path})")

    print(f"\nBuilding {N_HUGE}×{N_HUGE} matrix…")
    # Diagonally dominant: fast to build, definitely PD
    A = np.eye(N_HUGE, dtype=np.float64) * (N_HUGE + 1.0)
    # Add a small off-diagonal contribution
    rng = np.random.default_rng(0)
    v = rng.standard_normal(N_HUGE).astype(np.float64)
    A += np.outer(v, v) / N_HUGE  # rank-1, keeps PD

    b = rng.standard_normal(N_HUGE).astype(np.float64)

    print("Factorising…")
    c, lower = bigla.cho_factor(A, overwrite_a=True)

    print("Solving…")
    x = bigla.cho_solve((c, lower), b.copy())

    # residual check
    # We overwrote A, so reconstruct the diagonal part for a cheap check
    r_norm = np.linalg.norm(b - (N_HUGE + 1.0) * x)
    rel = r_norm / np.linalg.norm(b)
    print(f"Relative residual (diagonal part only): {rel:.2e}")
    assert rel < 1e-6, f"Huge cho_factor/solve residual too large: {rel:.2e}"


def test_huge_eigh_ev():
    """eigh with driver='ev' for n=46500 (minimal workspace: ~34n doubles)."""
    info = backend_info()
    if not info.ilp64:
        pytest.skip(f"ILP64 not available (backend: {info.path})")

    print(f"\nBuilding symmetric {N_HUGE}×{N_HUGE} matrix…")
    A = np.eye(N_HUGE, dtype=np.float64) * float(N_HUGE)

    print("Running eigh(driver='ev')…")
    ws = bigla.Workspace.for_eigh(N_HUGE, driver="ev")
    w, QT = bigla.eigh(A, driver="ev", work=ws, overwrite_a=True)

    assert len(w) == N_HUGE
    np.testing.assert_allclose(w, float(N_HUGE), rtol=1e-6)
    print("eigh(ev) passed.")
