"""Workspace — reusable scratch buffer for LAPACK calls.

Rationale (§6.5): a k-fold / λ-grid loop calls eigh or potrf repeatedly;
allocating and freeing a 37 GiB workspace each iteration is both slow and a
fragmentation hazard.  A Workspace holds float64 and int64 buffers that grow
as needed but are never shrunk.

Usage::

    ws = bigla.Workspace()
    for K in kernel_matrices:
        w, QT = bigla.eigh(K, work=ws)

    # or pre-size:
    ws = Workspace.for_eigh(n=50_000, driver="evd")
    w, QT = bigla.eigh(K, work=ws)
"""

from __future__ import annotations

import numpy as np


class Workspace:
    """Growable scratch buffers for LAPACK calls.

    Both the float and int buffers grow monotonically (never shrunk).
    Thread-safety: a single Workspace must not be shared across threads.
    """

    def __init__(self) -> None:
        self._float64: np.ndarray = np.empty(0, dtype=np.float64)
        self._float32: np.ndarray = np.empty(0, dtype=np.float32)
        self._int64: np.ndarray = np.empty(0, dtype=np.int64)

    def get_float(self, n: int, dtype) -> np.ndarray:
        """Return a float buffer of at least *n* elements, growing if needed."""
        dtype = np.dtype(dtype)
        if dtype == np.float64:
            if self._float64.size < n:
                self._float64 = np.empty(n, dtype=np.float64)
            return self._float64[:n]
        elif dtype == np.float32:
            if self._float32.size < n:
                self._float32 = np.empty(n, dtype=np.float32)
            return self._float32[:n]
        else:
            raise TypeError(f"Workspace: unsupported dtype {dtype}")

    def get_int(self, n: int) -> np.ndarray:
        """Return an int64 buffer of at least *n* elements, growing if needed."""
        if n == 0:
            return self._int64[:0]
        if self._int64.size < n:
            self._int64 = np.empty(n, dtype=np.int64)
        return self._int64[:n]

    @property
    def nbytes(self) -> int:
        """Total bytes currently allocated across all internal buffers."""
        return self._float64.nbytes + self._float32.nbytes + self._int64.nbytes

    def __repr__(self) -> str:
        gb = self.nbytes / 2**30
        return f"Workspace(nbytes={self.nbytes} = {gb:.3f} GiB)"

    @classmethod
    def for_eigh(cls, n: int, driver: str = "evd", dtype=np.float64) -> "Workspace":
        """Pre-allocate a Workspace large enough for eigh(n, driver=driver)."""
        ws = cls()
        dtype = np.dtype(dtype)
        if driver == "evd":
            lwork = 2 * n * n + 6 * n + 1
            liwork = 3 + 5 * n
            ws.get_float(lwork, dtype)
            ws.get_int(liwork)
        elif driver == "ev":
            lwork = max(1, 34 * n)
            ws.get_float(lwork, dtype)
        else:
            raise ValueError(f"Workspace.for_eigh: unknown driver {driver!r}")
        return ws

    @classmethod
    def for_potrf(cls, n: int, dtype=np.float64) -> "Workspace":
        """Pre-allocate a Workspace for potrf(n) (no-op; potrf needs no scratch)."""
        return cls()

    @classmethod
    def for_getrf(cls, n: int, dtype=np.float64) -> "Workspace":
        """Pre-allocate for getrf(n) (no-op; LU needs no float scratch -- only ipiv, which getrf
        allocates itself because it must be int64 for an ILP64 LAPACK)."""
        return cls()

    @classmethod
    def for_geqrf(cls, m: int, n: int, nb: int = 64, dtype=np.float64) -> "Workspace":
        """Pre-allocate for geqrf/gels on an (m, n) matrix.

        QR scratch is ``n * nb`` floats -- O(n), not O(n**2), which is the memory argument for
        reaching for QR instead of an SVD when a least-squares solve is all that is needed.
        """
        ws = cls()
        ws.get_float(max(1, n * nb), dtype)
        return ws

    @classmethod
    def for_svd(cls, m: int, n: int, driver: str = "gesdd", dtype=np.float64) -> "Workspace":
        """Pre-allocate for svd(m, n, driver=driver).

        The two drivers differ by orders of magnitude, which is the entire reason `driver` exists:
        ``gesdd`` needs ``4k**2 + 7k`` floats plus ``8k`` int64 (k = min(m, n)); ``gesvd`` needs
        ``max(3k + max(m, n), 5k)``. At k = 46341 that is ~69 GiB versus ~1.5 MiB.
        """
        ws = cls()
        k = min(m, n)
        if driver == "gesdd":
            ws.get_float(4 * k * k + 7 * k, dtype)
            ws.get_int(max(1, 8 * k))
        elif driver == "gesvd":
            ws.get_float(max(3 * k + max(m, n), 5 * k), dtype)
        else:
            raise ValueError(f"Workspace.for_svd: unknown driver {driver!r}")
        return ws
