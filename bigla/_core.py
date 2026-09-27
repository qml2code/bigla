"""ctypes prototypes and thin per-routine LAPACK wrappers (§6.2).

These are the low-level ILP64 wrappers named after the LAPACK routines.
They take pre-validated arrays and an optional Workspace, and are what
linalg.py calls.  Users doing tight loops can call them directly.

Calling conventions (§6.6):
- Every integer: ctypes.c_int64 by reference.
- Fortran character-length hidden args: c_int64(1) appended (gfortran ≥ 7).
- Keep a named local ref to every array for the duration of the call.

Ordering rule (§3):
- C-contiguous symmetric matrix → pass with uplo flipped.
  lower=True on C array → LAPACK 'U';  lower=False → 'L'.
"""

from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np

from bigla._backend import get_decoration, get_lib
from bigla.workspace import Workspace

_I = ctypes.c_int64
_VP = ctypes.c_void_p


def _ptr(a: np.ndarray) -> ctypes.c_void_p:
    return a.ctypes.data_as(_VP)


def _iref(n: int):
    return ctypes.byref(_I(n))


def _uplo_char(lower: bool, c_order: bool, symmetric: bool) -> ctypes.c_char:
    """Map (lower, c_order) -> LAPACK uplo.  C-order flips the triangle.

    ``symmetric`` must be stated explicitly, and the flip is only valid when it is True.
    For a SYMMETRIC matrix the Fortran view of a C-contiguous buffer is A^T = A, so
    reinterpreting the layout costs nothing and flipping uplo is all that is required.
    For a TRIANGULAR matrix the Fortran view is genuinely A^T, so ``trans`` must flip
    alongside uplo -- the caller has to handle that, and passing symmetric=False is the
    acknowledgement that it has. Callers that pass symmetric=False and forget `trans`
    produce silently transposed solutions (the D2 defect).
    """
    if not symmetric:
        raise ValueError(
            "_uplo_char(symmetric=False): a triangular matrix needs its `trans` flipped "
            "alongside uplo. Use _uplo_trans_triangular(), which returns both."
        )
    use_upper = lower == c_order
    return ctypes.c_char(b"U" if use_upper else b"L")


def _uplo_trans_triangular(lower: bool, c_order: bool, trans: int) -> tuple[ctypes.c_char, int]:
    """(uplo, trans) for a TRIANGULAR matrix -- both, so neither can be forgotten.

    The ordering trick in _uplo_char is free only because a symmetric matrix IS its own
    transpose: reinterpreting a C-contiguous buffer as Fortran gives A^T = A, so flipping uplo
    is the whole correction. A triangular matrix is not, so the Fortran view is genuinely A^T
    and `trans` must flip too. Returning the pair is what makes that structural rather than a
    comment the next routine can skip -- the D2 defect was exactly this rule being inherited
    from a neighbouring routine without its second half.

    Real dtypes only here, so conjugate-transpose collapses onto transpose.
    """
    use_upper = lower == c_order
    uplo = ctypes.c_char(b"U" if use_upper else b"L")
    if c_order:
        trans = {0: 1, 1: 0, 2: 0}[trans]
    return uplo, trans


def ascontiguous_or_raise(a: np.ndarray, name: str = "a") -> None:
    if not (a.flags.c_contiguous or a.flags.f_contiguous):
        raise ValueError(
            f"{name} must be C- or F-contiguous.  "
            f"Call np.ascontiguousarray({name}) to make an explicit copy."
        )


def _validate_square(a: np.ndarray, name: str, dtype) -> int:
    if a.dtype != np.dtype(dtype):
        raise TypeError(f"{name}: expected dtype {dtype}, got {a.dtype}")
    ascontiguous_or_raise(a, name)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"{name} must be a square 2-D array, got shape {a.shape}")
    return a.shape[0]


def _validate_2d(a: np.ndarray, name: str, dtype) -> tuple:
    """Validate a general (not necessarily square) 2-D array and report how LAPACK will see it.

    Returns ``(m, n, transposed)`` where ``m``/``n`` are the dimensions of the matrix LAPACK
    actually operates on, and ``transposed`` says whether that matrix is the caller's ``a``
    transposed.

    This is the general-matrix counterpart of the uplo flip used for symmetric routines, and the
    reason it needs to exist: a C-contiguous ``(p, q)`` buffer IS the column-major ``(q, p)``
    matrix ``a.T``. For a symmetric matrix that costs nothing because ``a.T == a``; for a general
    matrix LAPACK genuinely factorises the transpose, so every caller must account for it. We never
    copy (SPEC §3) -- the transpose is propagated into the algebra instead, in linalg.py.
    """
    if a.dtype != np.dtype(dtype):
        raise TypeError(f"{name}: expected dtype {dtype}, got {a.dtype}")
    ascontiguous_or_raise(a, name)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {a.shape}")
    # F-contiguous (incl. the 1-column/1-row ambiguous cases) -> LAPACK sees `a` itself.
    if a.flags.f_contiguous:
        return a.shape[0], a.shape[1], False
    return a.shape[1], a.shape[0], True


def _sym(name: str):
    lib = get_lib()
    p, s = get_decoration()
    return getattr(lib, f"{p}{name}{s}")


def _check_info(routine: str, info: int, n: int) -> None:
    if info == 0:
        return
    if info < 0:
        raise ValueError(f"{routine}: illegal argument {-info}")
    if routine in ("potrf",):
        raise np.linalg.LinAlgError(
            f"{routine}: leading minor of order {info} is not positive definite.  "
            "In kernel ridge regression this usually means the regularizer λ is too small."
        )
    if routine in ("potrs", "potri"):
        raise np.linalg.LinAlgError(f"{routine}: matrix is singular (info={info})")
    if routine in ("syevd", "syev"):
        raise np.linalg.LinAlgError(
            f"{routine}: eigenvalue algorithm did not converge (info={info})"
        )
    if routine == "getrf":
        raise np.linalg.LinAlgError(
            f"getrf: U[{info - 1},{info - 1}] is exactly zero -- the matrix is singular, so the "
            "factorization completed but cannot be used to solve. Unlike potrf this says nothing "
            "about definiteness; a ridge term is one fix, pivoting cannot rescue an exactly "
            "rank-deficient matrix."
        )
    if routine == "getrs":
        raise np.linalg.LinAlgError(f"getrs: matrix is singular (info={info})")
    if routine in ("gesdd", "gesvd"):
        raise np.linalg.LinAlgError(
            f"{routine}: SVD did not converge ({info} superdiagonals failed). Try driver='gesvd' "
            "(slower, more robust) if this came from gesdd."
        )
    if routine == "gels":
        raise np.linalg.LinAlgError(
            f"gels: element {info} of the triangular factor is exactly zero -- A is rank-deficient, "
            "so the least-squares solution is not unique. Use an SVD-based solve instead."
        )
    raise np.linalg.LinAlgError(f"{routine}: info={info}")


def _workspace_bufs(ws: Optional[Workspace], lwork: int, liwork: int, dtype) -> tuple:
    if ws is not None:
        dwork = ws.get_float(lwork, dtype)
        iwork = ws.get_int(liwork)
    else:
        dwork = np.empty(lwork, dtype=dtype)
        iwork = np.empty(liwork, dtype=np.int64)
    return dwork, iwork


# ---------------------------------------------------------------------------
# potrf
# ---------------------------------------------------------------------------


def potrf(
    a: np.ndarray,
    lower: bool = True,
    overwrite_a: bool = True,
) -> np.ndarray:
    """In-place Cholesky factorisation (dpotrf / spotrf)."""
    if not overwrite_a:
        a = a.copy(order="K")
    dtype = a.dtype
    fn = _sym("dpotrf") if dtype == np.float64 else _sym("spotrf") if dtype == np.float32 else None
    if fn is None:
        raise TypeError(f"potrf: unsupported dtype {dtype}")

    n = _validate_square(a, "a", dtype)
    uplo = _uplo_char(lower, a.flags.c_contiguous, symmetric=True)
    info = _I(0)
    fn(ctypes.byref(uplo), _iref(n), _ptr(a), _iref(n), ctypes.byref(info), _I(1))
    _check_info("potrf", info.value, n)
    return a


# ---------------------------------------------------------------------------
# potrs
# ---------------------------------------------------------------------------


def potrs(
    c: np.ndarray,
    b: np.ndarray,
    lower: bool = True,
    overwrite_b: bool = True,
) -> np.ndarray:
    """Solve A x = b from a Cholesky factor (dpotrs / spotrs)."""
    if not overwrite_b:
        b = b.copy(order="K")
    dtype = c.dtype
    fn = _sym("dpotrs") if dtype == np.float64 else _sym("spotrs") if dtype == np.float32 else None
    if fn is None:
        raise TypeError(f"potrs: unsupported dtype {dtype}")

    n = _validate_square(c, "c", dtype)
    if b.dtype != dtype:
        raise TypeError(f"potrs: b dtype {b.dtype} must match factor dtype {dtype}")
    ascontiguous_or_raise(b, "b")

    if b.ndim == 1:
        if b.shape[0] != n:
            raise ValueError(f"potrs: b.shape[0]={b.shape[0]} != n={n}")
        nrhs, ldb = 1, n
    elif b.ndim == 2:
        if b.shape[0] != n:
            raise ValueError(f"potrs: b.shape[0]={b.shape[0]} != n={n}")
        if not b.flags.f_contiguous:
            raise ValueError("potrs: multi-RHS b must be F-contiguous")
        nrhs, ldb = b.shape[1], n
    else:
        raise ValueError(f"potrs: b must be 1-D or 2-D, got {b.ndim}-D shape {b.shape}")

    uplo = _uplo_char(lower, c.flags.c_contiguous, symmetric=True)
    info = _I(0)
    fn(
        ctypes.byref(uplo),
        _iref(n),
        _iref(nrhs),
        _ptr(c),
        _iref(n),
        _ptr(b),
        _iref(ldb),
        ctypes.byref(info),
        _I(1),
    )
    _check_info("potrs", info.value, n)
    return b


# ---------------------------------------------------------------------------
# potri
# ---------------------------------------------------------------------------


def potri(
    c: np.ndarray,
    lower: bool = True,
    overwrite_c: bool = True,
) -> np.ndarray:
    """Compute A⁻¹ from a Cholesky factor (dpotri / spotri)."""
    if not overwrite_c:
        c = c.copy(order="K")
    dtype = c.dtype
    fn = _sym("dpotri") if dtype == np.float64 else _sym("spotri") if dtype == np.float32 else None
    if fn is None:
        raise TypeError(f"potri: unsupported dtype {dtype}")

    n = _validate_square(c, "c", dtype)
    uplo = _uplo_char(lower, c.flags.c_contiguous, symmetric=True)
    info = _I(0)
    fn(ctypes.byref(uplo), _iref(n), _ptr(c), _iref(n), ctypes.byref(info), _I(1))
    _check_info("potri", info.value, n)
    return c


# ---------------------------------------------------------------------------
# syevd
# ---------------------------------------------------------------------------


def syevd(
    a: np.ndarray,
    lower: bool = True,
    eigvals_only: bool = False,
    overwrite_a: bool = True,
    work: Optional[Workspace] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Symmetric eigendecomposition, divide & conquer (dsyevd / ssyevd).

    Returns (w, QT).  QT[i] is eigenvector i.  NOTE: QT is Qᵀ, not Q.
    Workspace: 2n² + 6n + 1 doubles → 37 GiB at n = 50 000.
    """
    if not overwrite_a:
        a = a.copy(order="K")
    dtype = a.dtype
    fn = _sym("dsyevd") if dtype == np.float64 else _sym("ssyevd") if dtype == np.float32 else None
    if fn is None:
        raise TypeError(f"syevd: unsupported dtype {dtype}")

    n = _validate_square(a, "a", dtype)
    jobz = ctypes.c_char(b"N" if eigvals_only else b"V")
    uplo = _uplo_char(lower, a.flags.c_contiguous, symmetric=True)
    w = np.empty(n, dtype=dtype)
    info = _I(0)

    # Workspace query
    wq = np.empty(1, dtype=dtype)
    iwq = np.empty(1, dtype=np.int64)
    _call_syevd(fn, jobz, uplo, n, a, w, wq, -1, iwq, -1, info)
    if info.value < 0:
        raise ValueError(f"syevd workspace query: illegal argument {-info.value}")
    # Floor at LAPACK's documented minimum: the query comes back in a float, and for float32
    # above n ~ 2900 it cannot represent 2n^2+6n+1 exactly. OpenBLAS happens to round UP
    # (+3 at n=4000, +63 at n=20000), so this is hardening rather than a live fix -- but a
    # backend rounding the other way would under-allocate, and ctypes does no bounds checking.
    #
    # The minimum depends on jobz. Flooring both at the jobz='V' value reserves ~2n^2 doubles
    # for an eigenvalues-only call -- 37 GiB of address space at n=50000. Resident memory
    # barely moves (np.empty does not fault in untouched pages), so it is invisible under
    # default overcommit and fails outright under vm.overcommit_memory=2, ulimit -v, or a
    # cgroup that counts address space: all normal on the managed HPC partitions this
    # package targets.
    floor = (2 * n + 1) if eigvals_only else (2 * n * n + 6 * n + 1)
    lwork = max(int(round(wq[0])), floor)
    liwork = int(iwq[0])

    dwork, iwork = _workspace_bufs(work, lwork, liwork, dtype)
    info = _I(0)
    _call_syevd(fn, jobz, uplo, n, a, w, dwork, lwork, iwork, liwork, info)
    _check_info("syevd", info.value, n)

    if eigvals_only:
        return w, None
    # Normalise output: QT[i] must always be eigenvector i (row convention).
    # C-contiguous input: LAPACK wrote eigenvectors as columns of its Fortran
    #   matrix, which map to rows of the C buffer -> a is already QT.
    # F-contiguous input: LAPACK wrote eigenvectors as columns of the buffer
    #   -> a is Q. a.T is a zero-copy view that gives QT.
    QT = a if a.flags.c_contiguous else a.T
    return w, QT


def _call_syevd(fn, jobz, uplo, n, a, w, dwork, lwork, iwork, liwork, info):
    fn(
        ctypes.byref(jobz),
        ctypes.byref(uplo),
        _iref(n),
        _ptr(a),
        _iref(n),
        _ptr(w),
        _ptr(dwork),
        _iref(lwork),
        _ptr(iwork),
        _iref(liwork),
        ctypes.byref(info),
        _I(1),
        _I(1),
    )


# ---------------------------------------------------------------------------
# syev
# ---------------------------------------------------------------------------


def syev(
    a: np.ndarray,
    lower: bool = True,
    eigvals_only: bool = False,
    overwrite_a: bool = True,
    work: Optional[Workspace] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Symmetric eigendecomposition, plain QR (dsyev / ssyev).

    Workspace: ~34n doubles → 14 MiB at n = 50 000.  Slower than syevd.
    """
    if not overwrite_a:
        a = a.copy(order="K")
    dtype = a.dtype
    fn = _sym("dsyev") if dtype == np.float64 else _sym("ssyev") if dtype == np.float32 else None
    if fn is None:
        raise TypeError(f"syev: unsupported dtype {dtype}")

    n = _validate_square(a, "a", dtype)
    jobz = ctypes.c_char(b"N" if eigvals_only else b"V")
    uplo = _uplo_char(lower, a.flags.c_contiguous, symmetric=True)
    w = np.empty(n, dtype=dtype)
    info = _I(0)

    wq = np.empty(1, dtype=dtype)
    _call_syev(fn, jobz, uplo, n, a, w, wq, -1, info)
    if info.value < 0:
        raise ValueError(f"syev workspace query: illegal argument {-info.value}")
    # See the syevd note above. dsyev's documented minimum is max(1, 3n-1) for BOTH jobz
    # values -- 34n is OpenBLAS's optimal BLOCKED size, not the minimum, and flooring there
    # would be harmless over-allocation described by a false claim.
    lwork = max(int(round(wq[0])), max(1, 3 * n - 1))

    dwork, _ = _workspace_bufs(work, lwork, 0, dtype)
    info = _I(0)
    _call_syev(fn, jobz, uplo, n, a, w, dwork, lwork, info)
    _check_info("syev", info.value, n)

    if eigvals_only:
        return w, None
    QT = a if a.flags.c_contiguous else a.T
    return w, QT


def _call_syev(fn, jobz, uplo, n, a, w, dwork, lwork, info):
    fn(
        ctypes.byref(jobz),
        ctypes.byref(uplo),
        _iref(n),
        _ptr(a),
        _iref(n),
        _ptr(w),
        _ptr(dwork),
        _iref(lwork),
        ctypes.byref(info),
        _I(1),
        _I(1),
    )


# ---------------------------------------------------------------------------
# trtrs
# ---------------------------------------------------------------------------


def trtrs(
    a: np.ndarray,
    b: np.ndarray,
    lower: bool = True,
    trans: int = 0,
    overwrite_b: bool = True,
) -> np.ndarray:
    """Solve a triangular system (dtrtrs / strtrs).  trans: 0=A, 1=Aᵀ."""
    if not overwrite_b:
        b = b.copy(order="K")
    dtype = a.dtype
    fn = _sym("dtrtrs") if dtype == np.float64 else _sym("strtrs") if dtype == np.float32 else None
    if fn is None:
        raise TypeError(f"trtrs: unsupported dtype {dtype}")

    n = _validate_square(a, "a", dtype)
    _trans_map = {0: b"N", 1: b"T", 2: b"C"}
    uplo, trans = _uplo_trans_triangular(lower, a.flags.c_contiguous, trans)
    trns = ctypes.c_char(_trans_map.get(trans, b"N"))
    diag = ctypes.c_char(b"N")
    ascontiguous_or_raise(b, "b")
    if b.ndim == 2 and b.shape[1] > 1 and not b.flags.f_contiguous:
        raise ValueError("trtrs: multi-RHS b must be F-contiguous")
    nrhs = 1 if b.ndim == 1 else b.shape[1]
    ldb = b.shape[0]
    info = _I(0)
    fn(
        ctypes.byref(uplo),
        ctypes.byref(trns),
        ctypes.byref(diag),
        _iref(n),
        _iref(nrhs),
        _ptr(a),
        _iref(n),
        _ptr(b),
        _iref(ldb),
        ctypes.byref(info),
        _I(1),
        _I(1),
        _I(1),
    )
    _check_info("trtrs", info.value, n)
    return b


# ---------------------------------------------------------------------------
# getrf / getrs  (LU)
# ---------------------------------------------------------------------------


def getrf(a: np.ndarray, overwrite_a: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """LU factorisation with partial pivoting (dgetrf / sgetrf).

    Returns ``(lu, ipiv)``. ``ipiv`` is 1-BASED as LAPACK returns it -- it is fed straight back to
    :func:`getrs`, so it is deliberately not converted to 0-based numpy indices.

    ``ipiv`` is int64 because an ILP64 LAPACK writes 64-bit integers into it. Sizing it as int32
    would corrupt adjacent memory rather than raise, so this is not a detail to "optimise".

    Ordering: for a C-contiguous ``a`` LAPACK factorises ``a.T`` (SPEC §3 / :func:`_validate_2d`).
    That is exact and copy-free, and :func:`lu_solve` compensates by flipping ``trans``.
    """
    if not overwrite_a:
        a = a.copy(order="K")
    dtype = a.dtype
    fn = _sym("dgetrf") if dtype == np.float64 else _sym("sgetrf") if dtype == np.float32 else None
    if fn is None:
        raise TypeError(f"getrf: unsupported dtype {dtype}")

    m, n, _ = _validate_2d(a, "a", dtype)
    ipiv = np.empty(min(m, n), dtype=np.int64)
    info = _I(0)
    fn(_iref(m), _iref(n), _ptr(a), _iref(m), _ptr(ipiv), ctypes.byref(info))
    _check_info("getrf", info.value, min(m, n))
    return a, ipiv


def getrs(
    lu: np.ndarray,
    ipiv: np.ndarray,
    b: np.ndarray,
    trans: int = 0,
    overwrite_b: bool = True,
) -> np.ndarray:
    """Solve from an LU factorisation (dgetrs / sgetrs).

    ``trans``: 0 -> ``A x = b``, 1 -> ``A**T x = b``, where ``A`` is the matrix LAPACK factorised
    (i.e. ``lu.T`` when ``lu`` is C-contiguous -- see :func:`getrf`).
    """
    if not overwrite_b:
        b = b.copy(order="K")
    dtype = lu.dtype
    fn = _sym("dgetrs") if dtype == np.float64 else _sym("sgetrs") if dtype == np.float32 else None
    if fn is None:
        raise TypeError(f"getrs: unsupported dtype {dtype}")

    m, n, _ = _validate_2d(lu, "lu", dtype)
    if m != n:
        raise ValueError(f"getrs: factor must be square, LAPACK sees {m}x{n}")
    if b.dtype != dtype:
        raise TypeError(f"getrs: b dtype {b.dtype} must match factor dtype {dtype}")
    if ipiv.dtype != np.int64:
        raise TypeError(f"getrs: ipiv must be int64 (ILP64), got {ipiv.dtype}")
    ascontiguous_or_raise(b, "b")
    if b.ndim == 1:
        if b.shape[0] != n:
            raise ValueError(f"getrs: b.shape[0]={b.shape[0]} != n={n}")
        nrhs, ldb = 1, n
    elif b.ndim == 2:
        if b.shape[0] != n:
            raise ValueError(f"getrs: b.shape[0]={b.shape[0]} != n={n}")
        if not b.flags.f_contiguous:
            raise ValueError("getrs: multi-RHS b must be F-contiguous")
        nrhs, ldb = b.shape[1], n
    else:
        raise ValueError("getrs: b must be 1-D or 2-D")

    trans_c = ctypes.c_char(b"T" if trans else b"N")
    info = _I(0)
    fn(
        ctypes.byref(trans_c),
        _iref(n),
        _iref(nrhs),
        _ptr(lu),
        _iref(n),
        _ptr(ipiv),
        _ptr(b),
        _iref(ldb),
        ctypes.byref(info),
        _I(1),
    )
    _check_info("getrs", info.value, n)
    return b


# ---------------------------------------------------------------------------
# geqrf / orgqr / ormqr  (QR)
# ---------------------------------------------------------------------------


def geqrf(
    a: np.ndarray, overwrite_a: bool = True, work: Optional[Workspace] = None
) -> tuple[np.ndarray, np.ndarray]:
    """QR factorisation (dgeqrf / sgeqrf). Returns ``(a, tau)``; ``R`` is the upper triangle of the
    LAPACK view of ``a`` and ``Q`` is represented implicitly by ``(a, tau)``.

    Workspace: ``n * nb`` doubles -- O(n), the reason QR is the memory-lean route to least squares.

    Ordering: C-contiguous ``a`` means LAPACK factorises ``a.T``, so this yields the QR of ``a.T``
    (equivalently an LQ of ``a``). Prefer :func:`gels` for least squares, which takes ``trans`` and
    so handles both orders exactly.
    """
    if not overwrite_a:
        a = a.copy(order="K")
    dtype = a.dtype
    fn = _sym("dgeqrf") if dtype == np.float64 else _sym("sgeqrf") if dtype == np.float32 else None
    if fn is None:
        raise TypeError(f"geqrf: unsupported dtype {dtype}")

    m, n, _ = _validate_2d(a, "a", dtype)
    tau = np.empty(min(m, n), dtype=dtype)
    info = _I(0)
    wq = np.empty(1, dtype=dtype)
    fn(_iref(m), _iref(n), _ptr(a), _iref(m), _ptr(tau), _ptr(wq), _iref(-1), ctypes.byref(info))
    if info.value < 0:
        raise ValueError(f"geqrf workspace query: illegal argument {-info.value}")
    lwork = max(1, int(round(wq[0])))
    dwork, _ = _workspace_bufs(work, lwork, 0, dtype)
    info = _I(0)
    fn(
        _iref(m),
        _iref(n),
        _ptr(a),
        _iref(m),
        _ptr(tau),
        _ptr(dwork),
        _iref(lwork),
        ctypes.byref(info),
    )
    _check_info("geqrf", info.value, min(m, n))
    return a, tau


def orgqr(
    a: np.ndarray,
    tau: np.ndarray,
    k: Optional[int] = None,
    overwrite_a: bool = True,
    work: Optional[Workspace] = None,
) -> np.ndarray:
    """Form ``Q`` explicitly from a :func:`geqrf` result (dorgqr / sorgqr).

    Overwrites the factor with ``Q`` (m x k). Only needed when ``Q`` itself is wanted; applying
    ``Q`` to something is cheaper and lower-memory via :func:`ormqr`.
    """
    if not overwrite_a:
        a = a.copy(order="K")
    dtype = a.dtype
    fn = _sym("dorgqr") if dtype == np.float64 else _sym("sorgqr") if dtype == np.float32 else None
    if fn is None:
        raise TypeError(f"orgqr: unsupported dtype {dtype}")

    m, n, _ = _validate_2d(a, "a", dtype)
    if k is None:
        k = min(m, n)
    info = _I(0)
    wq = np.empty(1, dtype=dtype)
    fn(
        _iref(m),
        _iref(k),
        _iref(k),
        _ptr(a),
        _iref(m),
        _ptr(tau),
        _ptr(wq),
        _iref(-1),
        ctypes.byref(info),
    )
    if info.value < 0:
        raise ValueError(f"orgqr workspace query: illegal argument {-info.value}")
    lwork = max(1, int(round(wq[0])))
    dwork, _ = _workspace_bufs(work, lwork, 0, dtype)
    info = _I(0)
    fn(
        _iref(m),
        _iref(k),
        _iref(k),
        _ptr(a),
        _iref(m),
        _ptr(tau),
        _ptr(dwork),
        _iref(lwork),
        ctypes.byref(info),
    )
    _check_info("orgqr", info.value, m)
    return a


def ormqr(
    a: np.ndarray,
    tau: np.ndarray,
    c: np.ndarray,
    side: str = "L",
    trans: str = "T",
    overwrite_c: bool = True,
    work: Optional[Workspace] = None,
) -> np.ndarray:
    """Apply ``Q`` (or ``Qᵀ``) from :func:`geqrf` to ``c`` WITHOUT forming ``Q`` (dormqr / sormqr).

    This is the memory argument for QR: forming ``Q`` costs m x k, applying it costs a
    O(block) workspace.
    """
    if not overwrite_c:
        c = c.copy(order="K")
    dtype = a.dtype
    fn = _sym("dormqr") if dtype == np.float64 else _sym("sormqr") if dtype == np.float32 else None
    if fn is None:
        raise TypeError(f"ormqr: unsupported dtype {dtype}")
    if c.dtype != dtype:
        raise TypeError(f"ormqr: c dtype {c.dtype} must match factor dtype {dtype}")
    ascontiguous_or_raise(c, "c")

    am, an, _ = _validate_2d(a, "a", dtype)
    k = len(tau)
    if c.ndim == 1:
        cm, cn, ldc = c.shape[0], 1, c.shape[0]
    else:
        if not c.flags.f_contiguous:
            raise ValueError("ormqr: 2-D c must be F-contiguous")
        cm, cn, ldc = c.shape[0], c.shape[1], c.shape[0]

    side_c = ctypes.c_char(side.encode()[:1].upper())
    trans_c = ctypes.c_char(trans.encode()[:1].upper())
    info = _I(0)
    wq = np.empty(1, dtype=dtype)
    fn(
        ctypes.byref(side_c),
        ctypes.byref(trans_c),
        _iref(cm),
        _iref(cn),
        _iref(k),
        _ptr(a),
        _iref(am),
        _ptr(tau),
        _ptr(c),
        _iref(ldc),
        _ptr(wq),
        _iref(-1),
        ctypes.byref(info),
        _I(1),
        _I(1),
    )
    if info.value < 0:
        raise ValueError(f"ormqr workspace query: illegal argument {-info.value}")
    lwork = max(1, int(round(wq[0])))
    dwork, _ = _workspace_bufs(work, lwork, 0, dtype)
    info = _I(0)
    fn(
        ctypes.byref(side_c),
        ctypes.byref(trans_c),
        _iref(cm),
        _iref(cn),
        _iref(k),
        _ptr(a),
        _iref(am),
        _ptr(tau),
        _ptr(c),
        _iref(ldc),
        _ptr(dwork),
        _iref(lwork),
        ctypes.byref(info),
        _I(1),
        _I(1),
    )
    _check_info("ormqr", info.value, k)
    return c


# ---------------------------------------------------------------------------
# gels  (least squares via QR/LQ)
# ---------------------------------------------------------------------------


def gels(
    a: np.ndarray,
    b: np.ndarray,
    trans: int = 0,
    overwrite_a: bool = True,
    overwrite_b: bool = True,
    work: Optional[Workspace] = None,
) -> np.ndarray:
    """Least-squares solve of a FULL-RANK system via QR or LQ (dgels / sgels).

    ``trans``: 0 -> minimise ``||A x - b||``, 1 -> minimise ``||A**T x - b||``, where ``A`` is the
    matrix LAPACK sees. Because a C-contiguous array presents as ``a.T``, ``trans`` is exactly the
    lever that makes both memory orders work with no copy -- :func:`bigla.linalg.lstsq` sets it.

    ``b`` must have room for the result: its leading dimension must be ``max(m, n)``.
    Raises on a rank-deficient ``A`` (LAPACK requires full rank here); use an SVD-based solve then.
    """
    dtype = a.dtype
    fn = _sym("dgels") if dtype == np.float64 else _sym("sgels") if dtype == np.float32 else None
    if fn is None:
        raise TypeError(f"gels: unsupported dtype {dtype}")
    if not overwrite_a:
        a = a.copy(order="K")
    if not overwrite_b:
        b = b.copy(order="K")
    if b.dtype != dtype:
        raise TypeError(f"gels: b dtype {b.dtype} must match a dtype {dtype}")
    ascontiguous_or_raise(b, "b")

    m, n, _ = _validate_2d(a, "a", dtype)
    ldb = max(m, n)
    if b.ndim == 1:
        nrhs = 1
        if b.shape[0] < ldb:
            raise ValueError(f"gels: b must hold max(m,n)={ldb} rows, got {b.shape[0]}")
    else:
        if not b.flags.f_contiguous:
            raise ValueError("gels: 2-D b must be F-contiguous")
        nrhs = b.shape[1]
        if b.shape[0] < ldb:
            raise ValueError(f"gels: b must hold max(m,n)={ldb} rows, got {b.shape[0]}")
    ldb = b.shape[0]

    trans_c = ctypes.c_char(b"T" if trans else b"N")
    info = _I(0)
    wq = np.empty(1, dtype=dtype)
    fn(
        ctypes.byref(trans_c),
        _iref(m),
        _iref(n),
        _iref(nrhs),
        _ptr(a),
        _iref(m),
        _ptr(b),
        _iref(ldb),
        _ptr(wq),
        _iref(-1),
        ctypes.byref(info),
        _I(1),
    )
    if info.value < 0:
        raise ValueError(f"gels workspace query: illegal argument {-info.value}")
    lwork = max(1, int(round(wq[0])))
    dwork, _ = _workspace_bufs(work, lwork, 0, dtype)
    info = _I(0)
    fn(
        ctypes.byref(trans_c),
        _iref(m),
        _iref(n),
        _iref(nrhs),
        _ptr(a),
        _iref(m),
        _ptr(b),
        _iref(ldb),
        _ptr(dwork),
        _iref(lwork),
        ctypes.byref(info),
        _I(1),
    )
    _check_info("gels", info.value, min(m, n))
    return b


# ---------------------------------------------------------------------------
# gesdd / gesvd  (SVD)
# ---------------------------------------------------------------------------


def _svd_outputs(m: int, n: int, dtype, compute_uv: bool):
    """Allocate ``s``, ``u``, ``vt`` for the reduced ('S') SVD, F-ordered as LAPACK writes them."""
    k = min(m, n)
    s = np.empty(k, dtype=dtype)
    if not compute_uv:
        return (
            s,
            np.empty((1, 1), dtype=dtype, order="F"),
            np.empty((1, 1), dtype=dtype, order="F"),
        )
    u = np.empty((m, k), dtype=dtype, order="F")
    vt = np.empty((k, n), dtype=dtype, order="F")
    return s, u, vt


def gesdd(
    a: np.ndarray,
    compute_uv: bool = True,
    overwrite_a: bool = True,
    work: Optional[Workspace] = None,
) -> tuple:
    """Reduced SVD by divide & conquer (dgesdd / sgesdd). Returns ``(u, s, vt)`` of the matrix
    LAPACK sees (``a.T`` for C-contiguous ``a`` -- :func:`bigla.linalg.svd` untangles that).

    WORKSPACE WARNING: ``gesdd`` is the SVD analogue of ``syevd`` -- its scratch is ~``4k**2``
    doubles (k = min(m,n)), i.e. ~69 GiB at k = 46341. That is exactly the size range bigla exists
    for, so :func:`bigla.linalg.svd` defaults to a memory-aware driver choice rather than to this
    routine. Use it when you know the workspace fits.
    """
    if not overwrite_a:
        a = a.copy(order="K")
    dtype = a.dtype
    fn = _sym("dgesdd") if dtype == np.float64 else _sym("sgesdd") if dtype == np.float32 else None
    if fn is None:
        raise TypeError(f"gesdd: unsupported dtype {dtype}")

    m, n, _ = _validate_2d(a, "a", dtype)
    k = min(m, n)
    jobz = ctypes.c_char(b"S" if compute_uv else b"N")
    s, u, vt = _svd_outputs(m, n, dtype, compute_uv)
    ldu, ldvt = (m if compute_uv else 1), (k if compute_uv else 1)

    info = _I(0)
    wq = np.empty(1, dtype=dtype)
    iwork = np.empty(max(1, 8 * k), dtype=np.int64)
    fn(
        ctypes.byref(jobz),
        _iref(m),
        _iref(n),
        _ptr(a),
        _iref(m),
        _ptr(s),
        _ptr(u),
        _iref(ldu),
        _ptr(vt),
        _iref(ldvt),
        _ptr(wq),
        _iref(-1),
        _ptr(iwork),
        ctypes.byref(info),
        _I(1),
    )
    if info.value < 0:
        raise ValueError(f"gesdd workspace query: illegal argument {-info.value}")
    lwork = max(1, int(round(wq[0])))
    dwork, iwork = _workspace_bufs(work, lwork, max(1, 8 * k), dtype)
    info = _I(0)
    fn(
        ctypes.byref(jobz),
        _iref(m),
        _iref(n),
        _ptr(a),
        _iref(m),
        _ptr(s),
        _ptr(u),
        _iref(ldu),
        _ptr(vt),
        _iref(ldvt),
        _ptr(dwork),
        _iref(lwork),
        _ptr(iwork),
        ctypes.byref(info),
        _I(1),
    )
    _check_info("gesdd", info.value, k)
    return (u, s, vt) if compute_uv else (None, s, None)


def gesvd(
    a: np.ndarray,
    compute_uv: bool = True,
    overwrite_a: bool = True,
    work: Optional[Workspace] = None,
) -> tuple:
    """Reduced SVD by QR iteration (dgesvd / sgesvd). Same outputs as :func:`gesdd`, slower, but
    its workspace is ``max(3k + max(m,n), 5k)`` -- O(max(m,n)) rather than O(k**2), which is why it
    is the safe driver at the sizes bigla targets."""
    if not overwrite_a:
        a = a.copy(order="K")
    dtype = a.dtype
    fn = _sym("dgesvd") if dtype == np.float64 else _sym("sgesvd") if dtype == np.float32 else None
    if fn is None:
        raise TypeError(f"gesvd: unsupported dtype {dtype}")

    m, n, _ = _validate_2d(a, "a", dtype)
    k = min(m, n)
    job = ctypes.c_char(b"S" if compute_uv else b"N")
    s, u, vt = _svd_outputs(m, n, dtype, compute_uv)
    ldu, ldvt = (m if compute_uv else 1), (k if compute_uv else 1)

    info = _I(0)
    wq = np.empty(1, dtype=dtype)
    fn(
        ctypes.byref(job),
        ctypes.byref(job),
        _iref(m),
        _iref(n),
        _ptr(a),
        _iref(m),
        _ptr(s),
        _ptr(u),
        _iref(ldu),
        _ptr(vt),
        _iref(ldvt),
        _ptr(wq),
        _iref(-1),
        ctypes.byref(info),
        _I(1),
        _I(1),
    )
    if info.value < 0:
        raise ValueError(f"gesvd workspace query: illegal argument {-info.value}")
    lwork = max(1, int(round(wq[0])))
    dwork, _ = _workspace_bufs(work, lwork, 0, dtype)
    info = _I(0)
    fn(
        ctypes.byref(job),
        ctypes.byref(job),
        _iref(m),
        _iref(n),
        _ptr(a),
        _iref(m),
        _ptr(s),
        _ptr(u),
        _iref(ldu),
        _ptr(vt),
        _iref(ldvt),
        _ptr(dwork),
        _iref(lwork),
        ctypes.byref(info),
        _I(1),
        _I(1),
    )
    _check_info("gesvd", info.value, k)
    return (u, s, vt) if compute_uv else (None, s, None)
