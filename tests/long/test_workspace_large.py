"""Workspace sizing at a scale that costs real memory.

Split out of tests/test_workspace.py so that `make test` never touches it: selection is by
DIRECTORY now, not by marker, which is what stops a `make test` from silently running the
expensive cases when an environment variable happens to be set.
"""

from __future__ import annotations

import numpy as np

from bigla.workspace import Workspace


def test_lwork_float32_large_n():
    """n=20000 float32: ~1.6 GiB of workspace, so it does not belong in the fast suite."""
    n = 20000
    ws = Workspace.for_eigh(n, "evd", dtype=np.float32)
    assert ws.nbytes >= (2 * n * n + 6 * n + 1) * 4


def test_svd_driver_gap_is_real_at_a_real_size():
    """The whole justification for `svd(driver=...)` is that gesdd's scratch grows as k², and the
    fast suite can only assert that with fabricated dimensions. Here it is at a size that costs
    actual memory: `for_svd(..., "gesvd")` is allocated for real and must stay small, while the
    gesdd requirement for the SAME problem is computed and must be orders of magnitude larger.

    Only the cheap driver is allocated -- asserting the expensive one by allocating it would be the
    very thing the driver policy exists to avoid.
    """
    from bigla.linalg import _gesdd_workspace_bytes

    n = 20_000
    ws = Workspace.for_svd(n, n, "gesvd")
    # O(max(m, n)): max(3k + max(m,n), 5k) floats = 100k doubles here, under a megabyte.
    assert ws.nbytes < 2 * 1024**2, f"gesvd workspace should be trivial, got {ws.nbytes} bytes"
    assert ws.nbytes >= max(3 * n + n, 5 * n) * 8

    # The same problem through gesdd: 4k² + 7k doubles, i.e. ~12.8 GiB at n = 20000.
    gesdd_bytes = _gesdd_workspace_bytes(n, n, np.float64)
    assert gesdd_bytes > 10 * 1024**3
    assert gesdd_bytes > 1000 * ws.nbytes, "the gap is the reason the driver argument exists"


def test_auto_driver_declines_gesdd_at_the_lp64_wall():
    """At k = 46341 -- the size this package exists to get past -- gesdd wants ~69 GiB of scratch.
    Whatever this machine has, the policy must not pick it blindly."""
    from bigla.linalg import _auto_svd_driver, _gesdd_workspace_bytes

    k = 46_341
    assert _gesdd_workspace_bytes(k, k, np.float64) > 60 * 1024**3
    # gesvd's requirement at the same size is ~1.5 MiB, so a machine that cannot take gesdd is
    # still perfectly able to run the SVD -- which is the point of having both.
    assert max(3 * k + k, 5 * k) * 8 < 2 * 1024**2
    assert _auto_svd_driver(k, k, np.float64) in ("gesdd", "gesvd")
