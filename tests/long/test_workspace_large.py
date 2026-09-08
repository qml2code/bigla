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
