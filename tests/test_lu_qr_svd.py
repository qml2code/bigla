"""LU, QR and SVD: correctness against numpy, in BOTH memory orders.

The memory order is the whole risk surface. A C-contiguous (p, q) buffer is the column-major
(q, p) matrix aᵀ, so every non-symmetric routine factorises the transpose. LU and SVD absorb that
exactly (a trans flag and an output swap respectively); QR cannot, and is documented to raise.
"""
import numpy as np
import pytest

import bigla

DTYPES = [np.float64, np.float32]


def tol(dtype, scale=1.0):
    return (2e-10 if dtype == np.float64 else 3e-4) * scale


def _mats(rng, m, n, dtype, order):
    a = rng.normal(size=(m, n)).astype(dtype)
    return np.asfortranarray(a) if order == "F" else np.ascontiguousarray(a)


# --------------------------------------------------------------------------- LU


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("order", ["C", "F"])
def test_lu_solve_matches_numpy(dtype, order):
    rng = np.random.default_rng(0)
    n = 37
    a = _mats(rng, n, n, dtype, order)
    b = rng.normal(size=n).astype(dtype)
    ref = np.linalg.solve(a.astype(np.float64), b.astype(np.float64))
    lu = bigla.lu_factor(a.copy(order="K"))
    x = bigla.lu_solve(lu, b.copy())
    assert np.abs(x - ref).max() < tol(dtype, 1e3)


@pytest.mark.parametrize("order", ["C", "F"])
def test_lu_solve_transposed_system(order):
    """trans=1 must mean aᵀ x = b for the CALLER's a, regardless of memory order."""
    rng = np.random.default_rng(1)
    n = 23
    a = _mats(rng, n, n, np.float64, order)
    b = rng.normal(size=n)
    ref = np.linalg.solve(a.T, b)
    lu = bigla.lu_factor(a.copy(order="K"))
    x = bigla.lu_solve(lu, b.copy(), trans=1)
    assert np.abs(x - ref).max() < 1e-7


@pytest.mark.parametrize("order", ["C", "F"])
def test_lu_multi_rhs(order):
    rng = np.random.default_rng(2)
    n, nrhs = 19, 4
    a = _mats(rng, n, n, np.float64, order)
    b = np.asfortranarray(rng.normal(size=(n, nrhs)))
    ref = np.linalg.solve(a, b)
    lu = bigla.lu_factor(a.copy(order="K"))
    x = bigla.lu_solve(lu, b.copy(order="F"))
    assert np.abs(x - ref).max() < 1e-7


@pytest.mark.parametrize("order", ["C", "F"])
def test_solve_assume_a_gen_on_nonsymmetric(order):
    """The reason LU exists: this matrix has no Cholesky factor at all."""
    rng = np.random.default_rng(3)
    n = 25
    a = _mats(rng, n, n, np.float64, order)  # general, not symmetric
    b = rng.normal(size=n)
    ref = np.linalg.solve(a, b)
    x = bigla.solve(a.copy(order="K"), b.copy(), assume_a="gen")
    assert np.abs(x - ref).max() < 1e-7


def test_solve_pos_rejects_unknown_assume_a():
    a = np.asfortranarray(np.eye(4))
    with pytest.raises(NotImplementedError, match="assume_a"):
        bigla.solve(a, np.ones(4), assume_a="sym")


def test_lu_factor_records_transposed_flag():
    a_c = np.ascontiguousarray(np.eye(5))
    a_f = np.asfortranarray(np.eye(5))
    assert bigla.lu_factor(a_c.copy(order="C")).transposed is True
    assert bigla.lu_factor(a_f.copy(order="F")).transposed is False


def test_lu_factor_unpacks_like_scipy():
    a = np.asfortranarray(np.eye(4) * 3.0)
    lu, ipiv = bigla.lu_factor(a.copy(order="F"))
    assert lu.shape == (4, 4) and ipiv.dtype == np.int64


def test_getrf_singular_raises_with_a_useful_message():
    a = np.asfortranarray(np.zeros((4, 4)))
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        bigla.lu_factor(a)


# --------------------------------------------------------------------------- QR


@pytest.mark.parametrize("dtype", DTYPES)
def test_qr_economic_matches_numpy(dtype):
    rng = np.random.default_rng(4)
    m, n = 40, 12
    a = _mats(rng, m, n, dtype, "F")
    q, r = bigla.qr(a.copy(order="F"))
    assert q.shape == (m, n) and r.shape == (n, n)
    # QR is unique up to column signs; compare the reconstruction instead.
    assert np.abs(q @ r - a).max() < tol(dtype, 1e2)
    assert np.abs(q.T @ q - np.eye(n)).max() < tol(dtype, 1e2)


def test_qr_mode_r_matches_economic():
    rng = np.random.default_rng(5)
    a = _mats(rng, 30, 8, np.float64, "F")
    r_only = bigla.qr(a.copy(order="F"), mode="r")
    _, r = bigla.qr(a.copy(order="F"))
    assert np.abs(np.abs(r_only) - np.abs(r)).max() < 1e-10


def test_qr_refuses_c_order_with_guidance():
    a = np.ascontiguousarray(np.random.default_rng(6).normal(size=(10, 3)))
    with pytest.raises(ValueError, match="F-contiguous"):
        bigla.qr(a)


def test_qr_refuses_underdetermined():
    a = np.asfortranarray(np.random.default_rng(7).normal(size=(3, 10)))
    with pytest.raises(ValueError, match="m >= n"):
        bigla.qr(a)


# --------------------------------------------------------------------------- lstsq


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("order", ["C", "F"])
def test_lstsq_overdetermined_matches_numpy(dtype, order):
    rng = np.random.default_rng(8)
    m, n = 50, 9
    a = _mats(rng, m, n, dtype, order)
    b = rng.normal(size=m).astype(dtype)
    ref = np.linalg.lstsq(a.astype(np.float64), b.astype(np.float64), rcond=None)[0]
    x = bigla.lstsq(a.copy(order="K"), b.copy())
    assert x.shape == (n,)
    assert np.abs(x - ref).max() < tol(dtype, 1e3)


@pytest.mark.parametrize("order", ["C", "F"])
def test_lstsq_multi_rhs(order):
    rng = np.random.default_rng(9)
    m, n, nrhs = 44, 7, 3
    a = _mats(rng, m, n, np.float64, order)
    b = rng.normal(size=(m, nrhs))
    ref = np.linalg.lstsq(a, b, rcond=None)[0]
    x = bigla.lstsq(a.copy(order="K"), b.copy())
    assert x.shape == (n, nrhs)
    assert np.abs(x - ref).max() < 1e-8


def test_lstsq_does_not_mutate_b_shape_contract():
    rng = np.random.default_rng(10)
    a = _mats(rng, 20, 5, np.float64, "F")
    b = rng.normal(size=20)
    b0 = b.copy()
    bigla.lstsq(a.copy(order="F"), b)
    assert np.array_equal(b, b0), "lstsq must copy b into its own max(m,n) buffer"


# --------------------------------------------------------------------------- SVD


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("driver", ["gesdd", "gesvd"])
@pytest.mark.parametrize("shape", [(40, 12), (12, 40), (20, 20)])
def test_svd_reconstructs_and_matches_numpy(dtype, order, driver, shape):
    rng = np.random.default_rng(11)
    m, n = shape
    a = _mats(rng, m, n, dtype, order)
    ref_s = np.linalg.svd(a.astype(np.float64), compute_uv=False)
    u, s, vh = bigla.svd(a.copy(order="K"), driver=driver)
    k = min(m, n)
    assert u.shape == (m, k) and vh.shape == (k, n) and s.shape == (k,)
    assert np.abs(s - ref_s).max() < tol(dtype, 1e2), "singular values"
    recon = (u * s) @ vh
    assert np.abs(recon - a).max() < tol(dtype, 1e2), "reconstruction"


@pytest.mark.parametrize("order", ["C", "F"])
def test_svdvals_matches_numpy(order):
    rng = np.random.default_rng(12)
    a = _mats(rng, 30, 11, np.float64, order)
    ref = np.linalg.svd(a, compute_uv=False)
    assert np.abs(bigla.svdvals(a.copy(order="K")) - ref).max() < 1e-10


def test_svd_rejects_full_matrices():
    a = np.asfortranarray(np.eye(4))
    with pytest.raises(NotImplementedError, match="full_matrices"):
        bigla.svd(a, full_matrices=True)


def test_svd_rejects_unknown_driver():
    a = np.asfortranarray(np.eye(4))
    with pytest.raises(ValueError, match="driver"):
        bigla.svd(a, driver="gesvdx")


def test_auto_driver_avoids_gesdd_when_workspace_would_not_fit():
    """The policy that justifies having gesvd at all: k**2 workspace must not be chosen blindly."""
    from bigla.linalg import _auto_svd_driver, _gesdd_workspace_bytes

    assert _auto_svd_driver(200, 200, np.float64) == "gesdd"  # trivially fits
    # k = 3_000_000 -> ~2.9e14 bytes of gesdd scratch; no machine has that.
    assert _auto_svd_driver(3_000_000, 3_000_000, np.float64) == "gesvd"
    assert _gesdd_workspace_bytes(1000, 50, np.float64) < _gesdd_workspace_bytes(
        1000, 500, np.float64
    )


def test_svd_of_rank_deficient_matrix_is_fine():
    """The case gels/QR cannot do -- the reason SVD earns its place next to QR."""
    rng = np.random.default_rng(13)
    base = rng.normal(size=(30, 3))
    a = np.asfortranarray(np.hstack([base, base]))  # rank 3, 6 columns
    s = bigla.svdvals(a.copy(order="F"))
    assert (s[:3] > 1e-8).all()
    assert (s[3:] < 1e-8).all()


# --------------------------------------------------------------------------- ormqr


def test_ormqr_drives_the_documented_memory_lean_least_squares_path():
    """`ormqr` applies Qᵀ WITHOUT forming Q -- the README's whole argument for reaching for QR
    when only a least-squares solve is wanted. Nothing else in the suite calls it, so without this
    the binding is exported and documented but never executed: exactly the
    "landed but unexercised" category docs/status.md exists to keep empty.

    Drives the real sequence -- geqrf, then ormqr for Qᵀb, then trtrs for R x = Qᵀb -- and checks
    it against numpy's lstsq. Not a round-trip against itself: an inverted convention inside ormqr
    would cancel out of a self-consistent check (R3).
    """
    from bigla._core import geqrf, ormqr, trtrs

    rng = np.random.default_rng(14)
    m, n = 60, 11
    a = np.asfortranarray(rng.normal(size=(m, n)))
    b = rng.normal(size=m)
    ref = np.linalg.lstsq(a, b, rcond=None)[0]

    qr_buf, tau = geqrf(a.copy(order="F"))
    # Qᵀb, computed without ever materialising Q
    qtb = ormqr(qr_buf, tau, b.copy(), side="L", trans="T")
    # R is the upper triangle of the factor; solve R x = (Qᵀb)[:n]
    r = np.asfortranarray(np.triu(qr_buf[:n, :n]))
    x = trtrs(r, np.ascontiguousarray(qtb[:n]), lower=False, trans=0)
    assert np.abs(x - ref).max() < 1e-9, "geqrf -> ormqr -> trtrs must reproduce lstsq"


def test_ormqr_applying_q_is_the_inverse_of_applying_q_transpose():
    """Q is orthogonal, so ormqr('N') undoes ormqr('T'). Cheap, and it pins the `trans` argument
    independently of the least-squares path above."""
    from bigla._core import geqrf, ormqr

    rng = np.random.default_rng(15)
    m, n = 40, 9
    a = np.asfortranarray(rng.normal(size=(m, n)))
    b = rng.normal(size=m)

    qr_buf, tau = geqrf(a.copy(order="F"))
    qtb = ormqr(qr_buf, tau, b.copy(), side="L", trans="T")
    back = ormqr(qr_buf, tau, qtb.copy(), side="L", trans="N")
    assert np.abs(back - b).max() < 1e-10


def test_rectangular_dim_guard_uses_the_element_count_not_the_long_side():
    """A tall-skinny matrix must not be refused for being tall.

    `_check_dim` asks `n > max_dim`, which is right for a square routine because n > 46340 iff n²
    passes what LP64's 32-bit indices address. Applied to `max(m, n)` it would reject a 100000 x 100
    problem whose 10**7 elements are perfectly addressable. This cannot be observed on an ILP64
    backend -- max_dim is ~4.6e18 there, so nothing trips either way -- so the LP64 limit is
    simulated rather than waited for.
    """
    from bigla import linalg as L

    class FakeLP64:
        max_dim = 46_340
        path = "/fake/liblapack.so"

    real = L.backend_info
    L.backend_info = lambda: FakeLP64()
    try:
        L._check_dim_2d(100_000, 100, "svd")  # 1e7 elements: must pass
        L._check_dim_2d(46_340, 46_340, "svd")  # the square boundary: must pass
        with pytest.raises(Exception, match="elements"):
            L._check_dim_2d(46_341, 46_341, "svd")  # just over: must raise
        with pytest.raises(Exception, match="elements"):
            L._check_dim_2d(10**6, 5_000, "svd")  # 5e9 elements: must raise
    finally:
        L.backend_info = real
