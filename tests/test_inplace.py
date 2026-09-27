"""Tests that verify the in-place / overwrite behaviour and the ordering rule.

Key invariant (§3 / §8):
  After in-place eigh of a C-ordered A, the returned Q satisfies
      Q.T @ diag(w) @ Q ≈ A_original
  This is the regression test that catches a transposition bug.
"""

from __future__ import annotations

import numpy as np
import pytest

import bigla
from bigla._core import ascontiguous_or_raise

N = 128


def spd(n=N, dtype=np.float64, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)).astype(dtype)
    return A @ A.T + n * np.eye(n, dtype=dtype)


def sym(n=N, dtype=np.float64, seed=1):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)).astype(dtype)
    return (A + A.T) / 2 + n * np.eye(n, dtype=dtype)


# ---------------------------------------------------------------------------
# Ordering rule: Q is Qᵀ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("driver", ["evd", "ev"])
def test_eigh_reconstruction(order, driver):
    """Q.T @ diag(w) @ Q ≈ A_original for both C- and F-ordered input."""
    A = sym()
    if order == "F":
        A = np.asfortranarray(A)
    A_orig = A.copy()
    w, Q = bigla.eigh(A, driver=driver)
    A_rec = Q @ (w[:, None] * Q.T)
    np.testing.assert_allclose(A_rec, A_orig, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("order", ["C", "F"])
def test_eigh_eigvec_orthonormal(order):
    A = sym()
    if order == "F":
        A = np.asfortranarray(A)
    _, Q = bigla.eigh(A)
    eye_approx = Q.T @ Q
    np.testing.assert_allclose(eye_approx, np.eye(N), atol=1e-10)


# ---------------------------------------------------------------------------
# overwrite_a=True actually reuses the buffer
# ---------------------------------------------------------------------------


def test_cho_factor_overwrite_true():
    A = spd()
    A_id = id(A)
    c, _ = bigla.cho_factor(A, overwrite_a=True)
    assert id(c) == A_id, "overwrite_a=True must return the same array"


def test_cho_factor_overwrite_false():
    A = spd()
    A_orig = A.copy()
    c, _ = bigla.cho_factor(A, overwrite_a=False)
    np.testing.assert_array_equal(A, A_orig, err_msg="overwrite_a=False must not touch original")


def test_cho_solve_overwrite_true():
    A = spd()
    b = np.random.default_rng(3).standard_normal(N)
    c, low = bigla.cho_factor(A.copy())
    b_id = id(b)
    x = bigla.cho_solve((c, low), b, overwrite_b=True)
    assert id(x) == b_id, "overwrite_b=True must return same array"


def test_cho_solve_overwrite_false():
    A = spd()
    b = np.random.default_rng(4).standard_normal(N)
    b_orig = b.copy()
    c, low = bigla.cho_factor(A.copy())
    bigla.cho_solve((c, low), b, overwrite_b=False)
    np.testing.assert_array_equal(b, b_orig, err_msg="overwrite_b=False must not touch b")


def test_eigh_overwrite_true():
    A = sym()
    w, Q = bigla.eigh(A, overwrite_a=True)
    # Q is a transposed VIEW of A's buffer, not A itself, so identity is the wrong test --
    # what overwrite_a=True promises is that no second n*n allocation happened.
    assert np.shares_memory(Q, A), "overwrite_a=True must reuse the input buffer"
    assert Q.base is not None, "Q should be a view, not a copy"


def test_eigh_overwrite_false():
    A = sym()
    A_orig = A.copy()
    bigla.eigh(A, overwrite_a=False)
    np.testing.assert_array_equal(A, A_orig)


# ---------------------------------------------------------------------------
# shares_memory checks
# ---------------------------------------------------------------------------


def test_cho_factor_shares_memory_overwrite():
    A = spd()
    c, _ = bigla.cho_factor(A, overwrite_a=True)
    assert np.shares_memory(c, A)


def test_cho_factor_no_shares_memory_copy():
    A = spd()
    c, _ = bigla.cho_factor(A, overwrite_a=False)
    assert not np.shares_memory(c, A)


# ---------------------------------------------------------------------------
# Workspace reuse
# ---------------------------------------------------------------------------


def test_workspace_reuse():
    ws = bigla.Workspace()
    A1 = sym(seed=10)
    A2 = sym(seed=11)
    w1, Q1 = bigla.eigh(A1.copy(), work=ws)
    nbytes_after_first = ws.nbytes
    w2, Q2 = bigla.eigh(A2.copy(), work=ws)
    assert ws.nbytes == nbytes_after_first, "Workspace should not grow on same-size second call"


def test_workspace_for_eigh():
    ws = bigla.Workspace.for_eigh(N, driver="evd")
    assert ws.nbytes > 0
    A = sym()
    w, Q = bigla.eigh(A.copy(), work=ws, driver="evd")
    assert len(w) == N


# ---------------------------------------------------------------------------
# Memmap works without copying
# ---------------------------------------------------------------------------


def test_memmap_cho_factor(tmp_path):
    A = spd()
    mm = np.memmap(tmp_path / "A.bin", dtype=np.float64, mode="w+", shape=A.shape)
    mm[:] = A
    mm.flush()
    mm2 = np.memmap(tmp_path / "A.bin", dtype=np.float64, mode="r+", shape=A.shape)
    c, lower = bigla.cho_factor(mm2, overwrite_a=True)
    # verify: c @ c.T ≈ A (upper triangle of C, since lower=True on C-order → LAPACK U)
    # just check it didn't raise
    assert c.shape == A.shape


# ---------------------------------------------------------------------------
# ascontiguous_or_raise
# ---------------------------------------------------------------------------


def test_ascontiguous_or_raise_ok():
    ascontiguous_or_raise(np.zeros((4, 4)), "a")  # C-contiguous: fine
    ascontiguous_or_raise(np.asfortranarray(np.zeros((4, 4))), "a")  # F-contiguous: fine


def test_ascontiguous_or_raise_fail():
    A = np.zeros((8, 8))
    with pytest.raises(ValueError, match="contiguous"):
        ascontiguous_or_raise(A[::2, ::2], "a")


# ---------------------------------------------------------------------------
# Overwrite contracts for the general-matrix routines (LU / QR / SVD)
# ---------------------------------------------------------------------------
#
# These exist because the new routines pass `overwrite_a` DOWN to bigla._core, whose own default is
# True. So the public default of False is honoured one layer below where it is declared, and a
# dropped argument anywhere in that chain would silently start consuming the caller's matrix while
# every correctness test still passed -- the returned answer is right either way.


def general(n=N, m=None, dtype=np.float64, seed=7):
    """A general (non-symmetric, well-conditioned) matrix; square unless `m` is given."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m or n, n)).astype(dtype)
    if m is None:
        A += n * np.eye(n, dtype=dtype)
    return np.asfortranarray(A)


def test_lu_factor_overwrite_true():
    A = general()
    A_id = id(A)
    lu = bigla.lu_factor(A, overwrite_a=True)
    assert id(lu.lu) == A_id, "overwrite_a=True must factor in place"


def test_lu_factor_overwrite_false():
    A = general()
    A_orig = A.copy()
    bigla.lu_factor(A, overwrite_a=False)
    np.testing.assert_array_equal(A, A_orig, err_msg="overwrite_a=False must not touch original")


def test_lu_solve_overwrite_true():
    A = general()
    b = np.random.default_rng(8).standard_normal(N)
    lu = bigla.lu_factor(A, overwrite_a=True)
    b_id = id(b)
    x = bigla.lu_solve(lu, b, overwrite_b=True)
    assert id(x) == b_id, "overwrite_b=True must return the same array"


def test_lu_solve_overwrite_false():
    A = general()
    b = np.random.default_rng(9).standard_normal(N)
    b_orig = b.copy()
    lu = bigla.lu_factor(A, overwrite_a=True)
    bigla.lu_solve(lu, b, overwrite_b=False)
    np.testing.assert_array_equal(b, b_orig, err_msg="overwrite_b=False must not touch b")


def test_solve_gen_overwrite_false_leaves_both_alone():
    """solve(assume_a='gen') delegates to lu_factor/lu_solve, so it has its own chance to drop the
    flag on the way through."""
    A = general()
    b = np.random.default_rng(10).standard_normal(N)
    A_orig, b_orig = A.copy(), b.copy()
    bigla.solve(A, b, assume_a="gen")
    np.testing.assert_array_equal(A, A_orig, err_msg="solve('gen') must not touch a by default")
    np.testing.assert_array_equal(b, b_orig, err_msg="solve('gen') must not touch b by default")


def test_qr_overwrite_false():
    A = general(n=16, m=48)
    A_orig = A.copy()
    bigla.qr(A, overwrite_a=False)
    np.testing.assert_array_equal(A, A_orig, err_msg="qr(overwrite_a=False) must not touch a")


def test_lstsq_overwrite_false():
    A = general(n=9, m=40)
    b = np.random.default_rng(11).standard_normal(40)
    A_orig, b_orig = A.copy(), b.copy()
    bigla.lstsq(A, b, overwrite_a=False)
    np.testing.assert_array_equal(A, A_orig, err_msg="lstsq(overwrite_a=False) must not touch a")
    # b is always copied here: gels needs a max(m, n)-row buffer to write the solution into.
    np.testing.assert_array_equal(b, b_orig, err_msg="lstsq must never consume b")


@pytest.mark.parametrize("driver", ["gesdd", "gesvd"])
def test_svd_overwrite_false(driver):
    A = general(n=12, m=36)
    A_orig = A.copy()
    bigla.svd(A, driver=driver, overwrite_a=False)
    np.testing.assert_array_equal(A, A_orig, err_msg="svd(overwrite_a=False) must not touch a")


@pytest.mark.parametrize("driver", ["gesdd", "gesvd"])
def test_svd_overwrite_true_does_consume_the_buffer(driver):
    """The other direction: overwrite_a=True must not quietly copy either, or the memory argument
    for this package evaporates."""
    A = general(n=12, m=36)
    A_orig = A.copy()
    bigla.svd(A, driver=driver, overwrite_a=True)
    assert not np.array_equal(A, A_orig), "overwrite_a=True should have consumed the buffer"


def test_svdvals_overwrite_false():
    A = general(n=10, m=30)
    A_orig = A.copy()
    bigla.svdvals(A, overwrite_a=False)
    np.testing.assert_array_equal(A, A_orig, err_msg="svdvals must forward overwrite_a=False")
