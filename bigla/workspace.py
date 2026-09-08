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
