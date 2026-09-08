"""LAPACK workspace sizing (§5.5, D6).

The query returns lwork in a float. For float32 above n ~ 2900 that cannot represent
2n^2+6n+1 exactly. OpenBLAS happens to round UP (+3 at n=4000, +63 at n=20000), so flooring
is hardening rather than a live fix -- but a backend rounding the other way would
under-allocate, and ctypes does no bounds checking, so the failure would be memory corruption
rather than an exception.

The floor must respect `jobz`. Flooring an eigenvalues-only call at the eigenvector minimum
reserves ~2n^2 doubles it never touches -- 37 GiB of address space at n=50000. Resident memory
barely moves, so it passes unnoticed under default overcommit and fails outright under
vm.overcommit_memory=2, ulimit -v, or an address-space cgroup.

These tests instrument the real path by passing an explicit Workspace and reading .nbytes
afterwards; asserting on Workspace.for_eigh's own formula would be a tautology that never
reaches the floor in _core.
"""
import numpy as np
import pytest

import bigla
from bigla.workspace import Workspace

N = 1200


def sym(n=N, dtype=np.float64, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n)).astype(dtype)
    return M + M.T


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_lwork_floor_respects_jobz(dtype):
    """The D6 regression test: an eigenvalues-only call must not reserve the eigenvector
    workspace. Fails against the tree that floored both at 2n^2+6n+1."""
    itemsize = np.dtype(dtype).itemsize
    A = sym(dtype=dtype)

    ws_vals = Workspace()
    bigla.eigh(np.array(A, order="C"), driver="evd", eigvals_only=True, work=ws_vals)
    assert (
        ws_vals.nbytes < N * N * itemsize
    ), f"eigvals_only reserved {ws_vals.nbytes} bytes, i.e. >= n^2 -- the jobz='V' floor"

    ws_vecs = Workspace()
    bigla.eigh(np.array(A, order="C"), driver="evd", eigvals_only=False, work=ws_vecs)
    assert ws_vecs.nbytes >= (2 * N * N + 6 * N + 1) * itemsize


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("eigvals_only", [True, False])
def test_lwork_never_below_lapack_minimum(dtype, eigvals_only):
    """The workspace handed to LAPACK is at least the documented minimum for that jobz, and
    not wildly above it."""
    itemsize = np.dtype(dtype).itemsize
    A = sym(dtype=dtype)
    ws = Workspace()
    bigla.eigh(np.array(A, order="C"), driver="evd", eigvals_only=eigvals_only, work=ws)

    minimum = (2 * N + 1) if eigvals_only else (2 * N * N + 6 * N + 1)
    doubles = ws.nbytes / itemsize
    assert doubles >= minimum, f"{doubles} < LAPACK minimum {minimum}"
    # OpenBLAS's optimal size exceeds the minimum, but not by orders of magnitude.
    assert doubles < 200 * minimum


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_eigh_runs_at_the_float32_boundary(dtype):
    """n=3000 is past where float32 stops representing the workspace formula exactly."""
    n = 3000
    w = bigla.eigvalsh(sym(n=n, dtype=dtype), driver="evd")
    assert w.shape == (n,) and np.isfinite(w).all()


@pytest.mark.long
def test_lwork_float32_large_n():
    """n=20000 float32: ~1.6 GiB of workspace, so it lives behind the `long` marker."""
    n = 20000
    ws = Workspace.for_eigh(n, "evd", dtype=np.float32)
    assert ws.nbytes >= (2 * n * n + 6 * n + 1) * 4
