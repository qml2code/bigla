"""Triangle and transpose conventions (D1, D2).

These exist because the 84-test suite passed both before and after two blocker defects. Three
rules follow from that, and every test here obeys them:

R1 -- A SYMMETRIC test matrix cannot test a triangle convention. When both triangles hold the
      same bytes, `uplo` cannot change any result and any self-consistent convention passes.
      Every test below poisons the triangle that `lower=` declares irrelevant.
R2 -- Assert conventions as literal tables, not only through behaviour. End-to-end tests show a
      convention is self-consistent; they cannot show it is the DOCUMENTED one.
R3 -- Round-trip self-consistency is not correctness. potrf and potrs share a `uplo` expression,
      so an inverted convention cancels and cho_factor -> cho_solve returns the right x. Each
      routine is therefore checked against scipy, not against its own inverse.
"""
import numpy as np
import pytest

scipy_linalg = pytest.importorskip("scipy.linalg", reason="scipy not installed")

import bigla
from bigla._core import _uplo_char

N = 120


def spd(n=N, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return M @ M.T + n * np.eye(n)


def poisoned(A, authoritative, order):
    """Copy of A in `order` with the NON-authoritative triangle filled with garbage."""
    B = np.array(A, order=order)
    n = B.shape[0]
    i, j = np.triu_indices(n, 1) if authoritative == "lower" else np.tril_indices(n, -1)
    B[i, j] = 1e6
    return B


# --------------------------------------------------------------------------- D1
@pytest.mark.parametrize(
    "lower,c_order,expected",
    [
        (True, True, b"U"),
        (False, True, b"L"),
        (True, False, b"L"),
        (False, False, b"U"),
    ],
)
def test_uplo_truth_table(lower, c_order, expected):
    """R2: the literal mapping. This is what fails if anyone 'simplifies' it back to an XOR."""
    assert _uplo_char(lower, c_order, symmetric=True).value == expected


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("lower", [True, False])
def test_poisoned_triangle_authoritative_succeeds(order, lower):
    """R1: with only the triangle `lower=` names left clean, the factorisation must succeed and
    match scipy on the clean matrix."""
    A = spd()
    P = poisoned(A, "lower" if lower else "upper", order)
    c_bl, _ = bigla.cho_factor(P, lower=lower)
    c_sp, _ = scipy_linalg.cho_factor(np.array(A, order=order), lower=lower)
    tri = np.tril if lower else np.triu
    np.testing.assert_allclose(tri(c_bl), tri(c_sp), atol=1e-8)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("lower", [True, False])
def test_poisoned_triangle_non_authoritative_is_rejected(order, lower):
    """The other half of the contract: reading the poisoned triangle must NOT silently work."""
    A = spd()
    P = poisoned(A, "upper" if lower else "lower", order)  # clean triangle is the WRONG one
    with pytest.raises(np.linalg.LinAlgError):
        bigla.cho_factor(P, lower=lower)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("lower", [True, False])
def test_symmetrise_fills_correct_triangle(order, lower):
    """cho_inverse must return an exactly symmetric matrix equal to inv(A)."""
    A = spd()
    P = poisoned(A, "lower" if lower else "upper", order)
    c, lo = bigla.cho_factor(P, lower=lower)
    inv = bigla.cho_inverse(c, lower=lo)
    np.testing.assert_array_equal(inv, inv.T)
    np.testing.assert_allclose(inv, np.linalg.inv(A), atol=1e-8)


# --------------------------------------------------------------------------- D2
def tri_matrix(n=60, seed=1):
    rng = np.random.default_rng(seed)
    L = np.tril(rng.standard_normal((n, n)))
    L[np.diag_indices(n)] += n  # well conditioned
    return L


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("trans", [0, 1])
@pytest.mark.parametrize("nrhs", [1, 3])
def test_solve_triangular_matches_scipy(order, trans, nrhs):
    """A triangular matrix is NOT its own transpose, so the symmetric ordering trick does not
    apply: reinterpreting a C-contiguous buffer as Fortran hands LAPACK A^T, and `trans` must
    flip alongside `uplo`. Without that, trans=0 silently solves A^T x = b."""
    L = tri_matrix()
    n = L.shape[0]
    rng = np.random.default_rng(2)
    b = rng.standard_normal(n) if nrhs == 1 else np.asfortranarray(rng.standard_normal((n, nrhs)))

    x_bl = bigla.solve_triangular(
        np.array(L, order=order), b.copy(order="K"), lower=True, trans=trans
    )
    x_sp = scipy_linalg.solve_triangular(
        np.array(L, order=order), b.copy(order="K"), lower=True, trans=trans
    )
    np.testing.assert_allclose(x_bl, x_sp, atol=1e-12)

    # regression: must NOT coincide with the other transpose
    x_other = scipy_linalg.solve_triangular(
        np.array(L, order=order), b.copy(order="K"), lower=True, trans=1 - trans
    )
    assert not np.allclose(x_bl, x_other, atol=1e-12)


def test_trtrs_rejects_c_ordered_multi_rhs():
    L = tri_matrix()
    n = L.shape[0]
    B = np.ascontiguousarray(np.random.default_rng(3).standard_normal((n, 3)))
    with pytest.raises(ValueError, match="F-contiguous"):
        bigla.solve_triangular(np.array(L, order="C"), B, lower=True, overwrite_b=True)


def test_solve_triangular_round_trip_with_cho_factor():
    """How an outside user most plausibly reaches trtrs -- and it crosses the symmetric /
    triangular boundary where D2 lives."""
    A = spd(n=60)
    b = np.random.default_rng(4).standard_normal(60)
    c, lower = bigla.cho_factor(np.array(A, order="C"), lower=True)
    y = bigla.solve_triangular(c, b.copy(), lower=lower, trans=0)
    x = bigla.solve_triangular(c, y, lower=lower, trans=1)
    np.testing.assert_allclose(x, bigla.cho_solve((c, lower), b.copy()), atol=1e-8)
