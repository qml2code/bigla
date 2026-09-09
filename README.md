# bigla

**ILP64 LAPACK for Python — no 46 341 × 46 341 ceiling, no copies.**

`bigla` binds, via `ctypes`, to whatever 64-bit-integer (ILP64)
BLAS/LAPACK is already on your system — by default the one that
NumPy's own wheels ship.  No Fortran compiler, no f2py, no Cython:
a pure-Python wheel with no build step at all.

```bash
pip install git+https://github.com/qml2code/bigla
```

*(Installed from the repository, not from PyPI — see [Installation](#installation) for the
clone/`make install` route.)*

---

## Why this exists

`scipy.linalg` links its own LP64 (32-bit integer) OpenBLAS.  That means
the element count of any operand matrix must fit in a signed 32-bit integer.
For a square `float64` matrix that puts the ceiling at **46 341 × 46 341** —
roughly 17 GiB.  `scipy.linalg.lapack.HAS_ILP64` is `False` in released wheels,
so one element wider is a matrix that interface cannot address, and the call
fails.  (Exactly *how* it fails is not something this project has measured at
that size; see [`docs/status.md`](docs/status.md).)

NumPy's wheels ship `libscipy_openblas64_*.so` built with `USE64BITINT`,
which has no such limit.  bigla binds that library directly.

### Memory savings (measured, n = 8000, float64)

| operation | extra peak RSS |
|---|---|
| `scipy.linalg.cho_factor(A, overwrite_a=True)`, A C-ordered | +0.59 GiB |
| `bigla.cho_factor(A)` | +0.02 GiB |
| `np.linalg.eigh(A)` | +1.44 GiB |
| `bigla.eigh(A, driver="evd")` | +0.48 GiB |
| `bigla.eigh(A, driver="ev")` | +0.01 GiB |

---

## Drop-in for `scipy.linalg`

Signatures, defaults and return conventions match `scipy.linalg`, so swapping the import is
the whole migration:

```python
# from scipy.linalg import cho_factor, cho_solve, eigh
from bigla import cho_factor, cho_solve, eigh
```

That includes `overwrite_a=False` / `overwrite_b=False` by default, and `eigh` returning
eigenvectors as **columns** (`Q[:, i]`), exactly as scipy does.

**What differs is capability, not behaviour:**

- **No 46341×46341 ceiling** on an ILP64 backend (`bigla.is_ilp64()`).
- **`overwrite_a=True` is actually honoured**, including for C-contiguous input — scipy
  copies that case regardless, which is the +0.59 GiB row in the table above. For matrices
  too large to copy, pass it explicitly; that is the point of the package.
- **`check_finite=False` by default**, because scipy's default costs an n² boolean temporary
  (61 MiB at n = 8000).

## Quick start

```python
import numpy as np
import bigla

# Cholesky factor + solve
K = build_kernel_matrix(X)          # (n, n) float64, C-contiguous
c, lower = bigla.cho_factor(K)      # in-place, K is now the factor
x = bigla.cho_solve((c, lower), y)  # in-place, y is now x

# Eigendecomposition
# Returns (w, Q) with eigenvectors as COLUMNS, like scipy
w, Q = bigla.eigh(K)
A_reconstructed = Q @ (w[:, None] * Q.T)

# Reusable workspace (avoids repeated 37-GiB allocation in a loop)
ws = bigla.Workspace.for_eigh(n=50_000, driver="evd")
for K in kernel_matrices:
    w, Q = bigla.eigh(K, work=ws)

# Diagnose the backend
bigla.backend_info()   # BackendInfo(path=..., ilp64=True, ...)
```

```
python -m bigla.diagnose   # paste-ready output for docs/backends.md
```

---

## API

### scipy-shaped (drop-in)

```python
cho_factor(a, lower=True, overwrite_a=False, check_finite=False)
cho_solve((c, lower), b, overwrite_b=False, check_finite=False)
cho_inverse(c, lower=True, overwrite_c=True)
eigh(a, lower=True, eigvals_only=False, overwrite_a=False,
     driver="auto", work=None, check_finite=False)
eigvalsh(a, ...)
solve(a, b, assume_a="pos", ...)
solve_triangular(a, b, lower=True, trans=0, overwrite_b=False)
```

### Low-level (LAPACK-named, for tight loops)

```python
from bigla._core import potrf, potrs, potri, syevd, syev, trtrs
```

### Backend control

```python
bigla.backend_info()           # BackendInfo dataclass: path, decoration, ilp64, max_dim, ...
bigla.is_ilp64()               # the question callers actually ask -- see below
bigla.get_num_threads()
bigla.set_num_threads(n)       # n is REQUIRED; bigla does not guess a thread count

with bigla.num_threads(16):    # scoped, mirroring scipy.fft.set_workers
    w, Q = bigla.eigh(K)
```

`is_ilp64()` is what to branch on, not whether `import bigla` succeeded: discovery falls back
to an LP64 library with `max_dim=46340` rather than failing at import, so a caller that tests
the import alone takes the bigla path and then raises on its first large matrix.

**Threading controls this library's handle only.** Where numpy resolved to a different `.so`
its pool is separate, and both will oversubscribe — a common surprise on HPC nodes.
[`threadpoolctl.threadpool_limits`](https://github.com/joblib/threadpoolctl) discovers
OpenBLAS, MKL, BLIS, FlexiBLAS and OpenMP pools by the same symbol probing bigla does by hand
and limits all of them at once; use it for the multi-pool case. The two accessors above remain
useful for the question it cannot answer: what is *this* handle doing.

---

## `eigh` drivers

| driver | LAPACK | workspace | at n=50 000 | notes |
|---|---|---|---|---|
| `"evd"` | `dsyevd` | 2n² + 6n + 1 doubles | **37 GiB** | fast; default unless memory is tight |
| `"ev"` | `dsyev` | ~34n doubles | 14 MiB | slower; use when the matrix fills the node |
| `"auto"` | — | — | — | reads `/proc/meminfo` and picks |

---

## Backend discovery

Searched in order:

1. `$BIGLA_LIB` — explicit path, always wins
2. `scipy_openblas64` package (`pip install scipy-openblas64`)
3. NumPy's bundled `libscipy_openblas64_*.so` (inside the numpy wheel)
4. System: `libopenblas64_.so{,.0}`, `libopenblas64.so{,.0}`, `libflexiblas64.so`
5. MKL (`libmkl_rt.so{,.2,.1}`) — **last**, deliberately

MKL is reached last because *probing* it has a side effect: validating its width calls
`MKL_Set_Interface_Layer(1)`, which changes the interface layer process-wide — including for a
NumPy that is itself MKL-linked and running LP64. Since loading is two-pass (a verified-ILP64
candidate always beats an LP64 one, wherever each appears in the list), the order barely affects
*selection* any more; it exists to keep that probe from running when something else already works.

If no ILP64 library is found: for `n ≤ 46340` bigla falls back to
`scipy.linalg`; for larger `n` it raises `BiglaBackendError` with
installation instructions.

See [`docs/backends.md`](docs/backends.md) for the per-platform install matrix.

---

## Installation

**bigla is not distributed through PyPI** — `pip install bigla` will not find it.  Install from
the repository:

```bash
# Standard (uses numpy's bundled OpenBLAS64)
pip install git+https://github.com/qml2code/bigla

# Explicit ILP64 OpenBLAS (always works, no system library needed)
pip install "bigla[openblas] @ git+https://github.com/qml2code/bigla"

# From a clone — editable, with the dev extras (see Development below)
git clone https://github.com/qml2code/bigla
cd bigla
make install

# Override library path
BIGLA_LIB=/usr/lib/x86_64-linux-gnu/libopenblas64.so python ...
```

Requirements: Python ≥ 3.10, NumPy ≥ 1.26.  SciPy is a test dependency only.

---

## Layout convention (important!)

See [`docs/conventions.md`](docs/conventions.md) for the full explanation.
Short version:

- C-contiguous and F-contiguous symmetric arrays are both accepted **without copying**.
- `lower=True` means the lower triangle of **your** array is authoritative.
- `eigh` returns `(w, Q)` with eigenvectors as **columns**, like scipy: `Q[:, i]` is
  eigenvector `i`, and `A ≈ Q @ np.diag(w) @ Q.T`. LAPACK writes them as rows into the
  working buffer; `Q` is the transposed view of it, so the scipy convention costs nothing
  and `Q.T` gets the row-major form back.

---

## Development

```bash
git clone https://github.com/qml2code/bigla
cd bigla
make install
make dev-setup     # one-time: installs the pre-commit hooks
make test
make diagnose
```

See `make help` for all targets.

Known-open items — fixes that have landed in code but do not yet have the test that closes
them — are tracked in [`docs/status.md`](docs/status.md).

### Formatting

Formatting and linting are handled entirely by pre-commit, using the same configuration as
[qml2-dev](https://github.com/qml2code/qml2) so the two repos stay stylistically identical:
autoflake, isort, black and flake8, all at line width 99. `make dev-setup` installs the hooks
once and they run on every commit thereafter.

There are deliberately no `make lint` / `make fmt` targets — duplicating the tool arguments in
the Makefile is how it drifts out of sync with the hooks that actually gate a commit.
`make review` runs every hook over the whole tree, which is what you want before the first
commit to a fresh checkout. `make conventional-commits` additionally installs the commit-msg
hook that enforces conventional commit messages.
