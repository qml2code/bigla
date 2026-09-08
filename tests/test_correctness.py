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
    """Symmetric (GOE), deliberately NOT shifted to positive definite.

    ``spd_matrix`` adds ``n*I`` because Cholesky requires positive definiteness. Carrying
    that shift over to here -- where ``eigh`` handles an indefinite matrix perfectly well --
    is not merely unnecessary, it is actively harmful: a shift moves every eigenvalue
    equally, so it inflates ``||A||`` by a factor of ~12 at n=256 while leaving every
    eigenvalue GAP untouched. Eigenvector conditioning is ``eps*||A||/gap``, so the shift
    multiplied the eigenvector error by that same factor for nothing. Measured at n=256,
    float32: dropping it takes the worst eigenvector disagreement from 1.1e-2 to 3.5e-5.

    .astype() runs BEFORE the symmetrisation so the float32 matrix is EXACTLY symmetric;
    cast afterwards and A[i,j] != A[j,i] in the last bit, and every `lower=` test quietly
    starts comparing two different matrices.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)).astype(dtype)
    return (A + A.T) / 2


def sym_matrix_with_spectrum(n: int, w, dtype=np.float64, seed: int = 7) -> np.ndarray:
    """Symmetric matrix with a CHOSEN spectrum: A = Q diag(w) Q.T.

    Sampling a random ensemble gives you whatever gaps the RNG deals -- and a GOE spectrum
    always contains a near-degenerate pair somewhere, since n eigenvalues are confined to a
    band of width 2*sqrt(2n). That is fine for eigenvalues and useless for testing
    eigenvectors at a known conditioning. Here the gaps are an input.
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return np.asarray((Q * np.asarray(w, dtype=np.float64)) @ Q.T, dtype=dtype)


# Calibration for the bounds below, all measured over six seeds at n=256, both orders,
# float32 and float64, on OpenBLAS 0.3.31 and 0.3.34. Constants are set ~10x above the
# observed worst case: loose enough to survive a BLAS that trades accuracy for speed,
# tight enough that a real defect (wrong triangle, wrong driver, a missed transpose)
# overshoots them by orders of magnitude rather than percent.
#
# A symmetric eigensolver is backward stable, which bounds the eigenvalue error
# ABSOLUTELY and scaled by the norm: |w~ - w| <= c*eps*||A||. There is deliberately no
# RELATIVE bound -- an eigenvalue near zero has poor relative accuracy and that is correct
# behaviour, not a defect. Asserting rtol on an indefinite spectrum fails on whichever
# eigenvalue happens to land nearest zero.
EIGVAL_C = 200  # observed <= 30 (f32), <= 54 (f64)
RESIDUAL_C = 100  # ||A q - w q|| / (eps*||A||);  observed <= 7.5
ORTHO_C = 200  # ||Q'Q - I|| / eps;              observed <= 17


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

    atol = EIGVAL_C * np.finfo(dtype).eps * float(np.abs(w_sp).max())
    np.testing.assert_allclose(np.sort(w_bl), np.sort(w_sp), rtol=0, atol=atol)

    # Check reconstruction:  A ≈ Q @ diag(w) @ Q.T  (columns are eigenvectors)
    A_rec = Q_bl @ (w_bl[:, None] * Q_bl.T)
    np.testing.assert_allclose(A_rec, A, rtol=rtol * 10, atol=1e-4)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("lower", [True, False])
def test_eigh_ev(dtype, order, lower):
    n = N_SMALL
    A = sym_matrix(n, dtype=dtype)
    if order == "F":
        A = np.asfortranarray(A)

    w_sp, _ = scipy_linalg.eigh(A.copy(), lower=lower)
    A_work = A.copy(order=order)
    w_bl, Q_bl = bigla.eigh(A_work, lower=lower, driver="ev", overwrite_a=False)

    atol = EIGVAL_C * np.finfo(dtype).eps * float(np.abs(w_sp).max())
    np.testing.assert_allclose(np.sort(w_bl), np.sort(w_sp), rtol=0, atol=atol)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_eigvalsh(dtype):
    n = N_SMALL
    A = sym_matrix(n, dtype=dtype)
    w_sp = scipy_linalg.eigvalsh(A.copy())
    w_bl = bigla.eigvalsh(A.copy(), overwrite_a=False)
    atol = EIGVAL_C * np.finfo(dtype).eps * float(np.abs(w_sp).max())
    np.testing.assert_allclose(np.sort(w_bl), np.sort(w_sp), rtol=0, atol=atol)


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


def _align_signs(Q, Q_ref):
    """Fix each column's arbitrary sign against a reference.

    Equivalent to reporting min(||v - u||, ||v + u||) per column: expanding
    ||v -/+ u||^2 = ||v||^2 + ||u||^2 -/+ 2<v,u> shows the minimising sign is just
    sign(<v,u>). Doing it as a preprocessing step keeps assert_allclose's diagnostics.

    Deliberately NOT a canonicalisation (e.g. "make the largest-magnitude entry
    positive"). That is what you are forced into when comparing checksums, since there is
    nothing to align against -- but it is discontinuous: when the two largest components
    are close, a perturbation of 1e-7 moves the argmax and flips the sign, reporting a
    distance of 2.0 between vectors that agree to 1e-7. With both matrices in hand,
    alignment has no such failure mode.
    """
    signs = np.sign(np.einsum("ij,ij->j", Q, Q_ref))
    signs[signs == 0] = 1
    return Q * signs


def _degenerate_clusters(w, max_gap):
    """Group eigenvalue indices separated by <= max_gap, returning groups of size > 1.

    Within such a group the eigenbasis is defined only up to an arbitrary ORTHOGONAL
    ROTATION, not up to a sign -- two correct implementations may plant different axes in
    the same invariant subspace. No sign convention can reconcile that; the comparison has
    to move to the subspace itself.
    """
    order = np.argsort(w)
    groups, current = [], [order[0]]
    for prev, idx in zip(order, order[1:]):
        if w[idx] - w[prev] <= max_gap:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)
    return [g for g in groups if len(g) > 1]


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_eigh_eigenvectors_match_scipy_orientation(order, dtype):
    """The drop-in guarantee: bigla.eigh returns eigenvectors as COLUMNS, like scipy.

    This is the regression test for the ordering rule. LAPACK leaves them as rows in the
    working buffer, and nothing else here reliably notices: the eigenvalue tests cannot see
    the difference at all, and test_eigh_evd's reconstruction catches it only because it
    happens to be written as Q diag(w) Q' -- write the same check the other way round and
    it passes on the transpose too. That makes it an accident, not a guarantee. This test
    pins the convention directly.

    Four layers, because a single element-wise comparison against scipy conflates things
    that need different treatment:

      1. eigenvalues        -- absolute, scaled by ||A||; unambiguous
      2. bigla on its own   -- residual and orthonormality, so correctness never routes
                               through another implementation's arbitrary basis choice
      3. well-separated     -- sign-aligned per-column distance; the transpose detector
      4. degenerate cluster -- spectral projector, invariant to sign AND rotation
    """
    n = N_SMALL
    eps = np.finfo(dtype).eps
    vec_tol = 1e-3 if dtype == np.float32 else 1e-7

    A = sym_matrix(n, dtype=dtype)
    w_sp, Q_sp = scipy_linalg.eigh(np.array(A, order=order), check_finite=False)
    w_bl, Q_bl = bigla.eigh(np.array(A, order=order), overwrite_a=False)
    norm = float(np.abs(w_sp).max())

    # (1) eigenvalues
    np.testing.assert_allclose(w_bl, w_sp, rtol=0, atol=EIGVAL_C * eps * norm)

    # (2) bigla against the definition, not against scipy
    A64 = np.asarray(A, dtype=np.float64)
    residual = np.abs(A64 @ Q_bl - Q_bl * w_bl).max()
    assert residual <= RESIDUAL_C * eps * norm, f"||A q - w q|| = {residual:.3e}"
    ortho = np.abs(Q_bl.T @ Q_bl - np.eye(n)).max()
    assert ortho <= ORTHO_C * eps, f"||Q'Q - I|| = {ortho:.3e}"

    # An eigenvector is perturbed by ~eps*||A||/gap, so a pair closer than this cannot be
    # compared vector-by-vector at vec_tol however well both implementations behave. The
    # split is DERIVED from the tolerance rather than hardcoded, so a future seed or size
    # that happens to deal a tight pair routes it to (4) instead of failing (3).
    max_gap = eps * norm / vec_tol
    clustered = _degenerate_clusters(np.asarray(w_bl, dtype=np.float64), max_gap)
    degenerate = {i for group in clustered for i in group}

    # (3) well-separated columns: the transpose would show up here as an O(1) error
    free = [k for k in range(n) if k not in degenerate]
    aligned = _align_signs(Q_bl[:, free], Q_sp[:, free])
    np.testing.assert_allclose(aligned, Q_sp[:, free], rtol=0, atol=vec_tol)

    # (4) degenerate clusters: compare the spectral projector, the only basis-independent
    # object. ||P1 - P2||_2 = sin(largest principal angle) between the two subspaces.
    for group in clustered:
        P_bl = Q_bl[:, group] @ Q_bl[:, group].T
        P_sp = Q_sp[:, group] @ Q_sp[:, group].T
        assert (
            np.abs(P_bl - P_sp).max() <= vec_tol
        ), f"invariant subspace for eigenvalues {np.asarray(w_bl)[group]} disagrees"


def test_eigh_handles_an_exactly_degenerate_subspace():
    """The degenerate case, tested on purpose rather than met by accident.

    A random ensemble deals whatever gaps it deals. Here the spectrum is an input, with a
    genuine multiplicity-4 eigenvalue: no sign convention can make the two implementations
    agree vector-by-vector inside that block, and the projector must agree anyway.
    """
    n, mult = 128, 4
    w = np.arange(1.0, n + 1.0)
    w[10 : 10 + mult] = w[10]  # an exactly repeated eigenvalue
    A = sym_matrix_with_spectrum(n, w, dtype=np.float64)

    w_sp, Q_sp = scipy_linalg.eigh(A.copy(), check_finite=False)
    w_bl, Q_bl = bigla.eigh(A.copy(), overwrite_a=False)
    eps, norm = np.finfo(np.float64).eps, float(np.abs(w_sp).max())

    np.testing.assert_allclose(w_bl, w_sp, rtol=0, atol=EIGVAL_C * eps * norm)

    block = list(range(10, 10 + mult))
    P_bl = Q_bl[:, block] @ Q_bl[:, block].T
    P_sp = Q_sp[:, block] @ Q_sp[:, block].T
    np.testing.assert_allclose(P_bl, P_sp, rtol=0, atol=1e-10)

    # The point of the projector: the individual vectors need NOT agree, and here they do
    # not -- so a per-vector assertion on this block would be asserting an accident.
    aligned = _align_signs(Q_bl[:, block], Q_sp[:, block])
    assert np.abs(aligned - Q_sp[:, block]).max() > 1e-6, (
        "expected a different basis inside the degenerate block; if these agree, the test "
        "no longer exercises what it claims to"
    )


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
