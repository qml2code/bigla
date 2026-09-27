"""scipy-shaped public API (§6.1).

Drop-in for scipy.linalg: same signatures and same defaults, including
overwrite_a=False and eigenvectors returned as COLUMNS.  What differs is the
size ceiling (none, on an ILP64 backend) and that overwrite_a=True is honoured
for C-contiguous input, which scipy copies regardless.

Three of the general-matrix routines deliberately narrow that promise, all for
one reason -- refusing to allocate at the sizes this package exists for:

  svd        full_matrices is False and True RAISES.  scipy defaults it to True,
             which materialises an m x m U; at n past the LP64 wall that array is
             the thing you were trying to avoid.
  svd,       the driver argument is `driver=` and defaults to "auto", where scipy
  svdvals    has `lapack_driver="gesdd"`.  gesdd's scratch grows as k^2 (~69 GiB
             at k = 46341), so choosing it unconditionally is not safe here.
  qr         mode defaults to "economic", not scipy's "full", for the same reason
             as full_matrices.  mode="r" skips forming Q at all.

qr also omits pivoting/lwork and lstsq omits cond/lapack_driver; add them when a
caller needs them.  lu_factor and lu_solve match scipy's signature exactly.
Every overwrite_a/overwrite_b default here is False, as it is in scipy.

Ordering rule (§3): C- and F-contiguous symmetric arrays are both accepted
without copying.  LAPACK leaves eigenvectors as ROWS of the working buffer;
eigh() returns the transposed VIEW so callers see scipy's column convention at
no cost.  bigla._core returns the row-major form if you want it.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np

from bigla._backend import BiglaBackendError, backend_info
from bigla._core import (
    ascontiguous_or_raise,
    gels,
    geqrf,
    gesdd,
    gesvd,
    getrf,
    getrs,
    orgqr,
    potrf,
    potri,
    potrs,
    syev,
    syevd,
    trtrs,
)
from bigla.workspace import Workspace

log = logging.getLogger(__name__)

_EVD_MEM_FRACTION = 0.8  # evd workspace fraction of MemAvailable before switching to ev


def _mem_available_bytes() -> Optional[int]:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _auto_driver(n: int, dtype) -> str:
    itemsize = np.dtype(dtype).itemsize
    evd_workspace = (2 * n * n + 6 * n + 1) * itemsize
    mem = _mem_available_bytes()
    if mem is not None and evd_workspace > _EVD_MEM_FRACTION * mem:
        log.debug(
            "eigh auto-driver: 'ev' (evd needs %.1f GiB, MemAvailable=%.1f GiB)",
            evd_workspace / 2**30,
            mem / 2**30,
        )
        return "ev"
    log.debug("eigh auto-driver: 'evd'")
    return "evd"


def _check_finite(a: np.ndarray, name: str, flag: bool) -> None:
    if flag and not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values (NaN or Inf)")


def _check_dim(n: int, func: str) -> None:
    info = backend_info()
    if n > info.max_dim:
        raise BiglaBackendError(
            f"{func}: n={n} exceeds max_dim={info.max_dim} for LP64 backend "
            f"({info.path}).  Install an ILP64 library or "
            f"`pip install scipy-openblas64`.  See docs/backends.md."
        )


# ---------------------------------------------------------------------------
# cho_factor
# ---------------------------------------------------------------------------


def _check_dim_2d(m: int, n: int, func: str) -> None:
    """Dimension guard for a RECTANGULAR matrix, where the binding quantity is the ELEMENT COUNT.

    `_check_dim` compares a single dimension against `max_dim`, which is exactly right for a square
    routine: for an n x n matrix, n > max_dim iff n**2 exceeds what LP64's 32-bit indices can
    address, and 46340 is just under sqrt(2**31). Applying that test to `max(m, n)` instead is
    over-strict -- it would refuse a 100000 x 100 problem on an LP64 backend, though its 10**7
    elements are comfortably addressable. Squaring `max_dim` reproduces the square case exactly
    while letting a tall-skinny matrix through.
    """
    info = backend_info()
    limit = info.max_dim**2
    if m * n > limit:
        raise BiglaBackendError(
            f"{func}: {m}x{n} is {m * n} elements, past the {limit} a backend with "
            f"max_dim={info.max_dim} can address ({info.path}).  Install an ILP64 library or "
            "`pip install scipy-openblas64`.  See docs/backends.md."
        )


def cho_factor(
    a: np.ndarray,
    lower: bool = True,
    overwrite_a: bool = False,
    check_finite: bool = False,
) -> tuple[np.ndarray, bool]:
    """Cholesky factorisation of a symmetric positive-definite matrix.

    Parameters
    ----------
    a : (n, n) float64 or float32, C- or F-contiguous
    lower : bool
        Which triangle of *a* is authoritative.
    overwrite_a : bool
        Default False, as in scipy.  True factorises in place -- including for
        C-contiguous input, which scipy copies regardless.
    check_finite : bool
        Check for NaN/Inf before calling LAPACK.  Default False.

    Returns
    -------
    (c, lower) : tuple compatible with scipy.linalg.cho_factor output
    """
    _check_finite(a, "a", check_finite)
    if a.ndim == 2:
        _check_dim(a.shape[0], "cho_factor")
    if not overwrite_a:
        ascontiguous_or_raise(a, "a")  # never let the copy hide a strided view
        a = a.copy(order="K")
    c = potrf(a, lower=lower, overwrite_a=True)
    return c, lower


# ---------------------------------------------------------------------------
# cho_solve
# ---------------------------------------------------------------------------


def cho_solve(
    c_and_lower: tuple[np.ndarray, bool],
    b: np.ndarray,
    overwrite_b: bool = False,
    check_finite: bool = False,
) -> np.ndarray:
    """Solve A x = b given the Cholesky factor from cho_factor.

    Parameters
    ----------
    c_and_lower : (c, lower) as returned by cho_factor
    b : (n,) or (n, nrhs) ndarray.  A C-contiguous multi-RHS b is reordered to Fortran
        layout (a copy), as scipy does; pass an F-contiguous b to avoid it.
    overwrite_b : bool  (default False, as in scipy)
    check_finite : bool  (default False)
    """
    c, lower = c_and_lower
    _check_finite(b, "b", check_finite)
    ascontiguous_or_raise(b, "b")  # never let a copy hide a strided view
    # LAPACK's potrs reads a multi-RHS b column-major, so a C-contiguous (n, nrhs) buffer is the
    # wrong layout. scipy accepts it and copies; mirroring scipy means doing the same rather than
    # raising. Single-RHS (1-D) is layout-agnostic and never needs this.
    needs_reorder = b.ndim == 2 and b.shape[1] > 1 and not b.flags.f_contiguous
    if needs_reorder:
        b = np.asfortranarray(b)
    elif not overwrite_b:
        b = b.copy(order="K")
    return potrs(c, b, lower=lower, overwrite_b=True)


# ---------------------------------------------------------------------------
# cho_inverse
# ---------------------------------------------------------------------------


def cho_inverse(
    c: np.ndarray,
    lower: bool = True,
    overwrite_c: bool = True,
) -> np.ndarray:
    """Compute A⁻¹ from a Cholesky factor (dpotri).

    LAPACK fills only one triangle; this function symmetrises the result.
    """
    if not overwrite_c:
        ascontiguous_or_raise(c, "c")  # never let the copy hide a strided view
        c = c.copy(order="K")
    result = potri(c, lower=lower, overwrite_c=True)
    _symmetrise(result, lower=lower)
    return result


def _symmetrise(a: np.ndarray, lower: bool) -> None:
    """Copy the LAPACK-filled triangle to the other to make a full symmetric matrix.

    After potri, LAPACK fills whichever triangle it was given (via uplo).
    Due to the C/Fortran ordering flip:
      lower=True  + C-order  -> LAPACK uplo=U -> upper triangle was filled
      lower=False + C-order  -> LAPACK uplo=L -> lower triangle was filled
      lower=True  + F-order  -> LAPACK uplo=L -> lower triangle was filled
      lower=False + F-order  -> LAPACK uplo=U -> upper triangle was filled

    Which triangle ends up filled AS NUMPY INDEXES IT does not depend on the storage order
    at all -- it is always the one the caller named. Work it through: uplo='U' fills the
    upper triangle of the FORTRAN view, which is numpy's LOWER triangle for a C-contiguous
    buffer; and the uplo flip in _uplo_char is exactly what cancels the transposition. The
    two flips compose to the identity, leaving ``upper_filled = not lower``.

    (The four-row table above is the uplo choice, not the filled numpy triangle -- reading it
    as the latter is what makes this look order-dependent when it is not.)

    We copy from the filled triangle to the empty one.
    """
    upper_filled = not lower
    if upper_filled:
        # upper filled -> copy upper to lower
        i, j = np.triu_indices(a.shape[0], k=1)
        a[j, i] = a[i, j]
    else:
        # lower filled -> copy lower to upper
        i, j = np.tril_indices(a.shape[0], k=-1)
        a[j, i] = a[i, j]


# ---------------------------------------------------------------------------
# eigh
# ---------------------------------------------------------------------------


def eigh(
    a: np.ndarray,
    lower: bool = True,
    eigvals_only: bool = False,
    overwrite_a: bool = False,
    driver: str = "auto",
    subset_by_index=None,
    work: Optional[Workspace] = None,
    check_finite: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Symmetric eigendecomposition.

    Parameters
    ----------
    a : (n, n) float64 or float32, C- or F-contiguous
    lower : bool — which triangle is authoritative (§3)
    eigvals_only : bool — return only eigenvalues
    overwrite_a : bool  (default False, as in scipy)
    driver : {"auto", "evd", "ev"}
        "evd" — divide & conquer, fast, needs 2n² doubles workspace.
        "ev"  — plain QR, ~34n doubles, slower but memory-efficient.
        "auto"— reads /proc/meminfo and picks evd unless workspace > 80% MemAvailable.
    work : Workspace or None — reusable scratch; create once, reuse across calls
    check_finite : bool  (default False)

    Returns
    -------
    w : (n,) eigenvalues, ascending
    Q : (n, n) — eigenvectors as *columns*, exactly like scipy.linalg.eigh.
        Q[:, i] is the i-th eigenvector.  Reconstruct: A ≈ Q @ np.diag(w) @ Q.T

        Q is a transposed VIEW of the working buffer (F-contiguous, no copy). LAPACK wrote
        the vectors as rows; `Q.T` gets that row-major form back for free if you want it.

    If eigvals_only=True, returns only w.
    """
    if subset_by_index is not None:
        raise NotImplementedError(
            "subset_by_index not yet implemented.  "
            "For top-k spectra consider driver='evr' (planned)."
        )
    _check_finite(a, "a", check_finite)
    if a.ndim == 2:
        _check_dim(a.shape[0], "eigh")
    if not overwrite_a:
        ascontiguous_or_raise(a, "a")  # never let the copy hide a strided view
        a = a.copy(order="K")
    if driver == "auto":
        driver = _auto_driver(a.shape[0], a.dtype)
    if driver == "evd":
        w, QT = syevd(a, lower=lower, eigvals_only=eigvals_only, overwrite_a=True, work=work)
    elif driver == "ev":
        w, QT = syev(a, lower=lower, eigvals_only=eigvals_only, overwrite_a=True, work=work)
    else:
        raise ValueError(f"eigh: unknown driver {driver!r}, choose 'evd', 'ev', or 'auto'")
    if eigvals_only:
        return w
    # LAPACK leaves the eigenvectors in the buffer as ROWS (QT). scipy returns them as COLUMNS,
    # and this layer is a drop-in for scipy, so transpose. `.T` is a view -- no copy, no extra
    # memory -- it only relabels the strides, which is why mirroring scipy here costs nothing.
    # The row-major buffer is still reachable as `Q.T` for anyone who wants it.
    return w, QT.T


# ---------------------------------------------------------------------------
# eigvalsh
# ---------------------------------------------------------------------------


def eigvalsh(
    a: np.ndarray,
    lower: bool = True,
    overwrite_a: bool = False,
    driver: str = "auto",
    work: Optional[Workspace] = None,
    check_finite: bool = False,
) -> np.ndarray:
    """Eigenvalues only of a symmetric matrix.  Wraps eigh(eigvals_only=True)."""
    return eigh(
        a,
        lower=lower,
        eigvals_only=True,
        overwrite_a=overwrite_a,
        driver=driver,
        work=work,
        check_finite=check_finite,
    )


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------


def solve(
    a: np.ndarray,
    b: np.ndarray,
    assume_a: str = "pos",
    lower: bool = True,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = False,
) -> np.ndarray:
    """Solve A x = b.

    ``assume_a='pos'`` (default): symmetric positive-definite, via potrf + potrs.
    ``assume_a='gen'``: general square matrix, via LU -- see :func:`lu_factor`.

    Choosing 'pos' for a matrix that is not positive definite is a correctness question, not a
    performance one: potrf either raises or, on an indefinite matrix whose leading minors happen to
    be positive, returns a wrong answer. Use 'gen' when definiteness is not established. ``lower``
    is ignored for 'gen' (LU reads the whole matrix).
    """
    if assume_a not in ("pos", "gen"):
        raise NotImplementedError(
            f"solve: assume_a={assume_a!r} not implemented.  Use 'pos' or 'gen'."
        )
    if assume_a == "gen":
        lu = lu_factor(a, overwrite_a=overwrite_a, check_finite=check_finite)
        return lu_solve(lu, b, overwrite_b=overwrite_b, check_finite=check_finite)
    _check_finite(a, "a", check_finite)
    _check_finite(b, "b", check_finite)
    if not overwrite_a:
        ascontiguous_or_raise(a, "a")  # never let the copy hide a strided view
        a = a.copy(order="K")
    if not overwrite_b:
        ascontiguous_or_raise(b, "b")  # never let the copy hide a strided view
        b = b.copy(order="K")
    potrf(a, lower=lower, overwrite_a=True)
    return potrs(a, b, lower=lower, overwrite_b=True)


# ---------------------------------------------------------------------------
# solve_triangular
# ---------------------------------------------------------------------------


def solve_triangular(
    a: np.ndarray,
    b: np.ndarray,
    lower: bool = True,
    trans: int = 0,
    overwrite_b: bool = False,
    check_finite: bool = False,
) -> np.ndarray:
    """Solve a triangular system A x = b (dtrtrs).

    trans: 0 → A x = b,  1 → Aᵀ x = b
    """
    _check_finite(a, "a", check_finite)
    _check_finite(b, "b", check_finite)
    if not overwrite_b:
        ascontiguous_or_raise(b, "b")  # never let the copy hide a strided view
        b = b.copy(order="K")
    return trtrs(a, b, lower=lower, trans=trans, overwrite_b=True)


# ---------------------------------------------------------------------------
# LU
# ---------------------------------------------------------------------------


class LUFactor:
    """An LU factorisation plus the one bit of bookkeeping a caller must not have to remember.

    A C-contiguous ``(n, n)`` buffer IS the column-major matrix ``a.T``, so LAPACK factorises the
    transpose (SPEC §3). That is exact and needs no copy, but every later solve has to know, because
    solving ``a x = b`` from a factorisation of ``aᵀ`` means asking getrs for the TRANSPOSED solve.
    Carrying ``transposed`` here is what stops that from becoming a caller-visible trap -- the same
    class of trap as ``eigh`` returning ``Qᵀ``.

    Attributes: ``lu`` (the overwritten buffer), ``ipiv`` (1-based, as LAPACK wrote it),
    ``transposed`` (whether ``lu`` holds the factors of ``aᵀ`` rather than ``a``).
    """

    __slots__ = ("lu", "ipiv", "transposed")

    def __init__(self, lu: np.ndarray, ipiv: np.ndarray, transposed: bool) -> None:
        self.lu = lu
        self.ipiv = ipiv
        self.transposed = transposed

    def __iter__(self):
        """Unpack as ``lu, ipiv`` for scipy-shaped call sites."""
        return iter((self.lu, self.ipiv))

    def __repr__(self) -> str:
        return (
            f"LUFactor(shape={self.lu.shape}, dtype={self.lu.dtype}, "
            f"transposed={self.transposed})"
        )


def lu_factor(
    a: np.ndarray,
    overwrite_a: bool = False,
    check_finite: bool = False,
) -> LUFactor:
    """LU factorisation with partial pivoting, for a general square matrix.

    Returns a :class:`LUFactor`; feed it to :func:`lu_solve`. Unlike Cholesky this makes no
    definiteness assumption, which is the whole point of having it: an indefinite or merely
    non-symmetric matrix has no Cholesky factor, and substituting one silently is a correctness bug
    rather than a slowdown.

    Accepts C- or F-contiguous input without copying; see :class:`LUFactor` on why the distinction
    is recorded.

    Signature-compatible with ``scipy.linalg.lu_factor``, ``overwrite_a=False`` default included.
    The one difference is the return type: a :class:`LUFactor` rather than a bare ``(lu, ipiv)``
    tuple, because the transpose flag has to travel with the factors. It unpacks as ``lu, ipiv`` for
    call sites that want the scipy shape.
    """
    _check_finite(a, "a", check_finite)
    n = a.shape[0]
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"lu_factor: a must be square 2-D, got shape {a.shape}")
    if not overwrite_a:
        ascontiguous_or_raise(a, "a")  # never let the copy hide a strided view
    _check_dim(n, "lu_factor")
    transposed = not a.flags.f_contiguous
    lu, ipiv = getrf(a, overwrite_a=overwrite_a)
    return LUFactor(lu, ipiv, transposed)


def lu_solve(
    lu_and_piv,
    b: np.ndarray,
    trans: int = 0,
    overwrite_b: bool = False,
    check_finite: bool = False,
) -> np.ndarray:
    """Solve from a :func:`lu_factor` result. ``trans``: 0 -> ``a x = b``, 1 -> ``aᵀ x = b``,
    where ``a`` is the matrix the CALLER factorised (the stored ``transposed`` flag is folded in
    here, so ``trans`` means what it says regardless of memory order).

    Accepts a bare ``(lu, ipiv)`` tuple too, in which case ``lu`` is assumed to be the factorisation
    of the matrix LAPACK saw -- i.e. no transpose correction is applied.
    """
    if isinstance(lu_and_piv, LUFactor):
        lu, ipiv, transposed = lu_and_piv.lu, lu_and_piv.ipiv, lu_and_piv.transposed
    else:
        lu, ipiv = lu_and_piv
        transposed = False
    _check_finite(b, "b", check_finite)
    ascontiguous_or_raise(b, "b")  # never let a copy hide a strided view
    # Same layout rule as cho_solve: LAPACK's getrs reads a multi-RHS b column-major, so a
    # C-contiguous (n, nrhs) buffer is the wrong layout. scipy accepts it and copies, so mirroring
    # scipy means doing the same rather than raising. Single-RHS (1-D) is layout-agnostic.
    needs_reorder = b.ndim == 2 and b.shape[1] > 1 and not b.flags.f_contiguous
    if needs_reorder:
        b = np.asfortranarray(b)
        overwrite_b = True  # the reordered copy is ours to consume
    elif not overwrite_b:
        b = b.copy(order="K")
        overwrite_b = True
    # XOR: factorising aᵀ turns a requested 'N' solve into a 'T' one and vice versa.
    effective_trans = int(bool(trans) ^ bool(transposed))
    return getrs(lu, ipiv, b, trans=effective_trans, overwrite_b=overwrite_b)


# ---------------------------------------------------------------------------
# QR
# ---------------------------------------------------------------------------


def qr(
    a: np.ndarray,
    mode: str = "economic",
    overwrite_a: bool = False,
    check_finite: bool = False,
    work: Optional[Workspace] = None,
):
    """QR factorisation of a general ``(m, n)`` matrix, ``m >= n``.

    ``mode='economic'`` (default) returns ``(Q, R)`` with ``Q`` of shape ``(m, k)`` and ``R``
    ``(k, n)``, ``k = min(m, n)``. ``mode='r'`` returns ``R`` alone and never forms ``Q`` -- much
    cheaper, and enough for a least-squares solve or for the tall-skinny SVD trick.

    Diverges from scipy, which defaults ``mode`` to ``'full'``: that returns the full ``(m, m)``
    ``Q``, and an ``m x m`` array is exactly what a package for out-of-scipy-range matrices must
    not hand back by default. ``pivoting`` and ``lwork`` are not implemented.

    **Requires F-contiguous input.** This is the one routine where the SPEC §3 no-copy rule bites:
    for a C-contiguous ``a`` LAPACK would factorise ``aᵀ``, and the QR of ``aᵀ`` is an LQ of ``a``,
    not something a caller expecting ``(Q, R)`` can use. Rather than copy silently or hand back a
    differently-shaped object, this raises and names the alternatives -- :func:`lstsq` and
    :func:`svd` both handle either order exactly and without copying, because their LAPACK drivers
    take a transpose flag.
    """
    if mode not in ("economic", "r"):
        raise ValueError(f"qr: mode must be 'economic' or 'r', not {mode!r}")
    if a.ndim != 2:
        raise ValueError(f"qr: a must be 2-D, got shape {a.shape}")
    if not a.flags.f_contiguous:
        raise ValueError(
            "qr: requires F-contiguous input. A C-contiguous buffer is the column-major aᵀ, whose "
            "QR is an LQ of a (SPEC §3). Either pass np.asfortranarray(a) -- an explicit copy you "
            "are then choosing to pay for -- or use lstsq()/svd(), which handle both memory orders "
            "with no copy."
        )
    if not overwrite_a:
        ascontiguous_or_raise(a, "a")  # never let the copy hide a strided view
    if not overwrite_a:
        ascontiguous_or_raise(a, "a")  # never let the copy hide a strided view
    if not overwrite_a:
        ascontiguous_or_raise(a, "a")  # never let the copy hide a strided view
    _check_finite(a, "a", check_finite)
    m, n = a.shape
    _check_dim_2d(m, n, "qr")
    if m < n:
        raise ValueError(f"qr: needs m >= n, got {m}x{n} (an underdetermined QR is an LQ)")
    k = min(m, n)

    qr_buf, tau = geqrf(a, overwrite_a=overwrite_a, work=work)
    r = np.triu(qr_buf[:k, :n])
    if mode == "r":
        return r
    q = orgqr(qr_buf, tau, k=k, overwrite_a=True, work=work)[:, :k]
    return q, r


def lstsq(
    a: np.ndarray,
    b: np.ndarray,
    overwrite_a: bool = False,
    check_finite: bool = False,
    work: Optional[Workspace] = None,
) -> np.ndarray:
    """Least-squares solution of ``a x ~= b`` for FULL-RANK ``a``, via QR/LQ (``gels``).

    Returns ``x`` with ``a.shape[1]`` rows. Works for either memory order with no copy of ``a``:
    a C-contiguous buffer presents to LAPACK as ``aᵀ``, and ``gels``'s ``trans`` flag turns that
    back into the intended problem exactly. ``b`` IS copied, because ``gels`` needs a buffer of
    ``max(m, n)`` rows to write the solution into.

    Raises on a rank-deficient ``a`` -- ``gels`` requires full rank. Use an SVD-based solve (see
    :func:`svd`) when the rank is in question; that is the case QR cannot cover.

    ``overwrite_a``/``overwrite_b`` default to False as in scipy, but ``cond`` and ``lapack_driver``
    are not implemented: ``cond`` exists in scipy to truncate small singular values, which is an
    SVD-based solve rather than the QR one ``gels`` performs, so accepting it here would be a
    promise this routine cannot keep.
    """
    if a.ndim != 2:
        raise ValueError(f"lstsq: a must be 2-D, got shape {a.shape}")
    _check_finite(a, "a", check_finite)
    _check_finite(b, "b", check_finite)
    m_user, n_user = a.shape
    _check_dim_2d(m_user, n_user, "lstsq")
    transposed = not a.flags.f_contiguous
    ldb = max(m_user, n_user)

    if b.ndim == 1:
        if b.shape[0] != m_user:
            raise ValueError(f"lstsq: b.shape[0]={b.shape[0]} != a.shape[0]={m_user}")
        buf = np.zeros(ldb, dtype=a.dtype)
        buf[:m_user] = b
    elif b.ndim == 2:
        if b.shape[0] != m_user:
            raise ValueError(f"lstsq: b.shape[0]={b.shape[0]} != a.shape[0]={m_user}")
        buf = np.zeros((ldb, b.shape[1]), dtype=a.dtype, order="F")
        buf[:m_user] = b
    else:
        raise ValueError("lstsq: b must be 1-D or 2-D")

    out = gels(a, buf, trans=int(transposed), overwrite_a=overwrite_a, overwrite_b=True, work=work)
    return out[:n_user]


# ---------------------------------------------------------------------------
# SVD
# ---------------------------------------------------------------------------

SVD_DRIVERS = ("auto", "gesdd", "gesvd")


def _gesdd_workspace_bytes(m: int, n: int, dtype) -> int:
    """LAPACK's minimum ``lwork`` for ``dgesdd`` with ``jobz='S'``, in bytes.

    ``4k**2 + 7k`` doubles (k = min(m,n)) plus the ``8k`` int64 iwork. The optimal lwork the
    workspace query reports is larger still, so this is a floor, not a ceiling.
    """
    k = min(m, n)
    return (4 * k * k + 7 * k) * np.dtype(dtype).itemsize + 8 * k * 8


def _auto_svd_driver(m: int, n: int, dtype) -> str:
    """Pick an SVD driver the machine can actually afford.

    The same policy ``eigh`` applies to evd-vs-ev, for the same reason: ``gesdd`` is the
    divide-and-conquer driver whose scratch grows as ``k**2`` (~69 GiB at k = 46341 -- squarely
    inside the range bigla exists for), while ``gesvd`` needs O(max(m,n)). Fast by default, safe
    when fast would not fit.
    """
    need = _gesdd_workspace_bytes(m, n, dtype)
    mem = _mem_available_bytes()
    if mem is not None and need > _EVD_MEM_FRACTION * mem:
        log.debug(
            "svd auto-driver: 'gesvd' (gesdd needs >= %.1f GiB, MemAvailable=%.1f GiB)",
            need / 2**30,
            mem / 2**30,
        )
        return "gesvd"
    log.debug("svd auto-driver: 'gesdd'")
    return "gesdd"


def svd(
    a: np.ndarray,
    full_matrices: bool = False,
    compute_uv: bool = True,
    driver: str = "auto",
    overwrite_a: bool = False,
    check_finite: bool = False,
    work: Optional[Workspace] = None,
):
    """Reduced SVD ``a = U diag(s) Vh``, returning ``(U, s, Vh)`` (or just ``s``).

    Two deliberate divergences from ``scipy.linalg.svd``, both the same argument:

    * **``full_matrices`` defaults to False and True RAISES**, where scipy defaults it to True. The
      full form materialises an ``m x m`` ``U``; past the LP64 wall that array is the thing the
      caller came here to avoid, so it is refused rather than silently allocated.
    * **the driver argument is ``driver=``, defaulting to ``'auto'``**, where scipy has
      ``lapack_driver='gesdd'``. gesdd's scratch grows as ``k**2`` -- about 69 GiB at k = 46341 --
      so picking it unconditionally is not safe at these sizes. ``'auto'`` chooses by available
      memory (see :func:`_auto_svd_driver`); ``'gesdd'`` forces divide & conquer (fast,
      ``O(k**2)`` workspace); ``'gesvd'`` forces QR iteration (slower, ``O(max(m,n))`` workspace).

    ``overwrite_a`` defaults to False, as in scipy.

    Either memory order is accepted with no copy. For a C-contiguous ``a`` LAPACK factorises
    ``aᵀ = Ũ s Ṽᵀ``; since ``a = Ṽ s Ũᵀ``, the outputs are recovered by swapping and transposing,
    both of which are zero-copy views. So C-order input costs nothing here -- unlike :func:`qr`,
    where no such identity exists.
    """
    if driver not in SVD_DRIVERS:
        raise ValueError(f"svd: driver must be one of {SVD_DRIVERS}, not {driver!r}")
    if full_matrices:
        raise NotImplementedError(
            "svd: only full_matrices=False is implemented. The full form needs an m x m U, which "
            "defeats the purpose of an out-of-scipy-range SVD."
        )
    if a.ndim != 2:
        raise ValueError(f"svd: a must be 2-D, got shape {a.shape}")
    _check_finite(a, "a", check_finite)
    m_user, n_user = a.shape
    _check_dim_2d(m_user, n_user, "svd")
    transposed = not a.flags.f_contiguous

    if driver == "auto":
        driver = _auto_svd_driver(m_user, n_user, a.dtype)
    fn = gesdd if driver == "gesdd" else gesvd
    u_l, s, vt_l = fn(a, compute_uv=compute_uv, overwrite_a=overwrite_a, work=work)

    if not compute_uv:
        return s
    if not transposed:
        return u_l, s, vt_l
    # LAPACK factorised aᵀ = u_l s vt_l, so a = vt_lᵀ s u_lᵀ. Both transposes are views.
    return vt_l.T, s, u_l.T


def svdvals(
    a: np.ndarray,
    driver: str = "auto",
    overwrite_a: bool = False,
    check_finite: bool = False,
    work: Optional[Workspace] = None,
) -> np.ndarray:
    """Singular values only. Transpose-invariant, so memory order is irrelevant here.

    Takes ``driver=`` (default ``'auto'``) where ``scipy.linalg.svdvals`` has no driver argument at
    all; see :func:`svd` for why the choice cannot be made unconditionally here.
    """
    return svd(
        a,
        compute_uv=False,
        driver=driver,
        overwrite_a=overwrite_a,
        check_finite=check_finite,
        work=work,
    )
