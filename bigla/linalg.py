"""scipy-shaped public API (§6.1).

Drop-in for scipy.linalg: same signatures and same defaults, including
overwrite_a=False and eigenvectors returned as COLUMNS.  What differs is the
size ceiling (none, on an ILP64 backend) and that overwrite_a=True is honoured
for C-contiguous input, which scipy copies regardless.

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
from bigla._core import ascontiguous_or_raise, potrf, potri, potrs, syev, syevd, trtrs
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
            f"`pip install bigla[openblas]`.  See docs/backends.md."
        )


# ---------------------------------------------------------------------------
# cho_factor
# ---------------------------------------------------------------------------


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
    """Solve A x = b for symmetric positive-definite A (potrf + potrs).

    Only assume_a='pos' is implemented.  No LU fallback.
    """
    if assume_a != "pos":
        raise NotImplementedError(
            f"solve: assume_a={assume_a!r} not implemented.  Only 'pos' is supported."
        )
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
