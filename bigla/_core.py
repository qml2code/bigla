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
