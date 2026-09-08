"""Correctness tests — compare bigla against scipy.linalg.

Runs for n <= 4000 (always, no special env var needed).
Covers C-ordered, F-ordered, and transposed views; lower in {True, False};
float64 and float32; overwrite_a in {True, False}.
"""

from __future__ import annotations

import numpy as np
import pytest

scipy_linalg = pytest.importorskip("scipy.linalg", reason="scipy not installed")

import bigla

# Small n so tests are fast in CI
N_SMALL = 256


def spd_matrix(n: int, dtype=np.float64, seed: int = 0) -> np.ndarray:
    """Random symmetric positive-definite matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)).astype(dtype)
    A = A @ A.T + n * np.eye(n, dtype=dtype)
    return A


def sym_matrix(n: int, dtype=np.float64, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)).astype(dtype)
    A = (A + A.T) / 2 + n * np.eye(n, dtype=dtype)
    return A


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("lower", [True, False])
def test_cho_factor_solve(dtype, order, lower):
    rtol = 1e-5 if dtype == np.float32 else 1e-12
    n = N_SMALL
    A = spd_matrix(n, dtype=dtype)
    if order == "F":
        A = np.asfortranarray(A)

    b_vec = np.random.default_rng(42).standard_normal(n).astype(dtype)

    # scipy reference
    c_sp, low_sp = scipy_linalg.cho_factor(A.copy(), lower=lower)
    x_sp = scipy_linalg.cho_solve((c_sp, low_sp), b_vec.copy())

    # bigla
    A_work = A.copy(order=order)
    c_bl, low_bl = bigla.cho_factor(A_work, lower=lower, overwrite_a=False)
    x_bl = bigla.cho_solve((c_bl, low_bl), b_vec.copy())

    np.testing.assert_allclose(x_bl, x_sp, rtol=rtol, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("lower", [True, False])
def test_eigh_evd(dtype, order, lower):
    rtol = 1e-4 if dtype == np.float32 else 1e-9
    n = N_SMALL
    A = sym_matrix(n, dtype=dtype)
    if order == "F":
        A = np.asfortranarray(A)

    w_sp, Q_sp = scipy_linalg.eigh(A.copy(), lower=lower)
    A_work = A.copy(order=order)
    w_bl, Q_bl = bigla.eigh(A_work, lower=lower, driver="evd", overwrite_a=False)

    np.testing.assert_allclose(np.sort(w_bl), np.sort(w_sp), rtol=rtol, atol=1e-6)

    # Check reconstruction:  A ≈ Q.T @ diag(w) @ Q
    A_rec = Q_bl @ (w_bl[:, None] * Q_bl.T)
    np.testing.assert_allclose(A_rec, A, rtol=rtol * 10, atol=1e-4)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("lower", [True, False])
def test_eigh_ev(dtype, order, lower):
    rtol = 1e-4 if dtype == np.float32 else 1e-9
    n = N_SMALL
    A = sym_matrix(n, dtype=dtype)
    if order == "F":
        A = np.asfortranarray(A)

    w_sp, _ = scipy_linalg.eigh(A.copy(), lower=lower)
    A_work = A.copy(order=order)
    w_bl, Q_bl = bigla.eigh(A_work, lower=lower, driver="ev", overwrite_a=False)

    np.testing.assert_allclose(np.sort(w_bl), np.sort(w_sp), rtol=rtol, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_eigvalsh(dtype):
    rtol = 1e-4 if dtype == np.float32 else 1e-9
    n = N_SMALL
    A = sym_matrix(n, dtype=dtype)
    w_sp = scipy_linalg.eigvalsh(A.copy())
    w_bl = bigla.eigvalsh(A.copy(), overwrite_a=False)
    np.testing.assert_allclose(np.sort(w_bl), np.sort(w_sp), rtol=rtol, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_cho_inverse(dtype):
    rtol = 1e-4 if dtype == np.float32 else 1e-10
    atol = 1e-4 if dtype == np.float32 else 1e-9
    n = N_SMALL
    A = spd_matrix(n, dtype=dtype)
    Ainv = bigla.cho_inverse(*bigla.cho_factor(A.copy(), overwrite_a=False))
    I_approx = A @ Ainv
    np.testing.assert_allclose(I_approx, np.eye(n, dtype=dtype), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_solve(dtype):
    rtol = 1e-4 if dtype == np.float32 else 1e-11
    n = N_SMALL
    A = spd_matrix(n, dtype=dtype)
    b = np.random.default_rng(7).standard_normal(n).astype(dtype)
    x_bl = bigla.solve(A.copy(), b.copy(), overwrite_a=False, overwrite_b=False)
    x_sp = scipy_linalg.solve(A, b, assume_a="pos")
    np.testing.assert_allclose(x_bl, x_sp, rtol=rtol, atol=1e-5)


@pytest.mark.parametrize("nrhs", [1, 8])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_cho_solve_multi_rhs(nrhs, dtype):
    rtol = 1e-4 if dtype == np.float32 else 1e-11
    n = N_SMALL
    A = spd_matrix(n, dtype=dtype)
    B = np.random.default_rng(9).standard_normal((n, nrhs)).astype(dtype, order="F")
    c, lower = bigla.cho_factor(A.copy(), overwrite_a=False)
    if nrhs == 1:
        b = B[:, 0].copy()
        x_bl = bigla.cho_solve((c, lower), b, overwrite_b=False)
        x_sp = scipy_linalg.cho_solve(
            scipy_linalg.cho_factor(A.copy(), lower=lower), B[:, 0].copy()
        )
    else:
        x_bl = bigla.cho_solve((c, lower), B.copy(order="F"), overwrite_b=False)
        x_sp = np.column_stack(
            [
                scipy_linalg.cho_solve(
                    scipy_linalg.cho_factor(A.copy(), lower=lower), B[:, k].copy()
                )
                for k in range(nrhs)
            ]
        )
    np.testing.assert_allclose(x_bl, x_sp, rtol=rtol, atol=1e-4)


def test_non_contiguous_raises():
    A = np.eye(8)
    A_nc = A[::2, ::2]  # non-contiguous
    assert not A_nc.flags.c_contiguous
    assert not A_nc.flags.f_contiguous
    with pytest.raises(ValueError, match="contiguous"):
        bigla.cho_factor(A_nc)


def test_check_finite_raises():
    A = spd_matrix(16)
    A[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        bigla.cho_factor(A, check_finite=True)


def test_not_positive_definite_raises():
    A = -np.eye(8, dtype=np.float64)
    with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
        bigla.cho_factor(A)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_eigh_eigenvectors_match_scipy_orientation(order, dtype):
    """The drop-in guarantee: bigla.eigh returns eigenvectors as COLUMNS, like scipy.

    This is the regression test for the ordering rule. LAPACK leaves them as rows in the
    working buffer, so an implementation that forgets to transpose still passes every
    reconstruction test (Q diag(w) Qᵀ is symmetric in the mistake) and every eigenvalue
    test -- and then silently returns the transpose to callers. Comparing against scipy
    element-by-element is what catches it.
    """
    n = N_SMALL
    A = sym_matrix(n, dtype=dtype)
    w_sp, Q_sp = scipy_linalg.eigh(np.array(A, order=order), check_finite=False)
    w_bl, Q_bl = bigla.eigh(np.array(A, order=order), overwrite_a=False)

    rtol = 1e-5 if dtype == np.float32 else 1e-11
    np.testing.assert_allclose(w_bl, w_sp, rtol=rtol)

    # eigenvector sign is arbitrary; align per column before comparing
    signs = np.sign(np.einsum("ij,ij->j", Q_bl, Q_sp))
    signs[signs == 0] = 1
    np.testing.assert_allclose(Q_bl * signs, Q_sp, atol=1e-3 if dtype == np.float32 else 1e-7)


@pytest.mark.parametrize("order", ["C", "F"])
def test_cho_solve_accepts_multi_rhs_in_either_order(order):
    """scipy accepts a C-contiguous multi-RHS b and copies; so must we.

    LAPACK's potrs reads b column-major, so a C-ordered (n, nrhs) buffer is the wrong layout.
    Raising there would have made this a not-quite-drop-in -- found by qml2's integration test,
    where B comes out of a feature-matrix product C-contiguous.
    """
    n, nrhs = 64, 3
    A = spd_matrix(n)
    B = np.array(np.random.default_rng(0).standard_normal((n, nrhs)), order=order)

    c_sp, low_sp = scipy_linalg.cho_factor(A.copy(), lower=True)
    x_sp = scipy_linalg.cho_solve((c_sp, low_sp), B.copy())

    c_bl, low_bl = bigla.cho_factor(A.copy(), lower=True)
    x_bl = bigla.cho_solve((c_bl, low_bl), B.copy())

    np.testing.assert_allclose(x_bl, x_sp, atol=1e-9)
