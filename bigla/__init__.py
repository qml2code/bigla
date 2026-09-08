"""bigla — ILP64 LAPACK layer without the 46341×46341 ceiling.

The public API mirrors scipy.linalg: same signatures, same defaults
(overwrite_a=False), and eigh returns eigenvectors as COLUMNS. The difference is
that it works past scipy's 46341x46341 LP64 ceiling, and that passing
overwrite_a=True actually avoids the copy -- including for C-contiguous input,
where scipy copies anyway.

For arrays too large to copy, pass overwrite_a=True explicitly. The low-level
layer (bigla._core: potrf, syevd, ...) is in-place by default.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bigla")
except PackageNotFoundError:  # running from source without install
    __version__ = "0.0.0.dev0"

from bigla._backend import (
    BackendInfo,
    ThreadInfo,
    backend_info,
    get_num_threads,
    is_ilp64,
    num_threads,
    set_num_threads,
    thread_control_info,
)
from bigla.linalg import (
    cho_factor,
    cho_inverse,
    cho_solve,
    eigh,
    eigvalsh,
    solve,
    solve_triangular,
)
from bigla.workspace import Workspace

__all__ = [
    "__version__",
    # backend
    "BackendInfo",
    "ThreadInfo",
    "backend_info",
    "is_ilp64",
    "num_threads",
    "thread_control_info",
    "get_num_threads",
    "set_num_threads",
    # linear algebra
    "cho_factor",
    "cho_solve",
    "cho_inverse",
    "eigh",
    "eigvalsh",
    "solve",
    "solve_triangular",
    # workspace
    "Workspace",
]
