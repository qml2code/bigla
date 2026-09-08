# `bigla` — a scipy.linalg-shaped LAPACK layer without the handicaps

*Two decisions below were deliberately overridden during review, so that the public API is a
true drop-in for `scipy.linalg`:*

1. *§6.1's `overwrite_a=True` default → **`False`**, matching scipy. The no-copy path is the
   reason the package exists, but it is now opt-in per call rather than a silent default that
   destroys a caller's matrix. `bigla._core` (the explicit in-place layer, §6.2) is unchanged.*
2. *§3's `QT` return → **`Q`**, eigenvectors as columns, matching scipy. `.T` on the working
   buffer is a view, so the scipy convention costs nothing — and §3's own warning that this is
   "the single most likely source of silent user error" is the argument for removing the
   divergence rather than documenting it.*

*Historical design specification, kept for reference — the implementation in `bigla/` is what
ships. The companion `bigla.py` proof-of-concept it refers to was deliberately NOT committed: a
top-level `bigla.py` beside the `bigla/` package is an import-shadowing hazard, and its ctypes
plumbing now lives in `bigla/_core.py`. Original text follows.*

*Specification for implementation. Companion file: `bigla.py` (working draft, verified
against scipy — treat it as a proof of concept for the ctypes plumbing, not as the final
module layout).*

Name is a placeholder; rename freely (`lapack64`, `biglinalg`, …).

---

## 1. Purpose

Provide the handful of dense-symmetric LAPACK routines that kernel methods need
(`cho_factor`, `cho_solve`, `eigh`, and friends) with three properties `scipy.linalg`
does not give:

1. **No 46341 × 46341 ceiling.** Work through a 64-bit-integer (ILP64) BLAS/LAPACK.
2. **Overwrite flags that are actually honoured**, including for C-contiguous input.
3. **Explicit control of workspace and driver**, because for large `n` the LAPACK
   workspace, not the matrix, is what exhausts the node.

Secondary purpose, equally important: be a place to accumulate *verified* installation
recipes for the combinations of distro / BLAS / numpy / scipy / HPC module we encounter.
See §9.

### Non-goals

- Reimplementing `scipy.linalg`. Sparse, decompositions we don't use, banded solvers,
  `expm`, etc. are out of scope.
- Shipping our own BLAS. We bind to a library that is already on the machine.
- Any compiled code. **No f2py, no Cython, no C extension.** The wheel must be
  `py3-none-any`. This is the whole point: qml's install pain came from f2py needing
  gfortran at build time plus per-platform ABI matching. A ctypes package has no build
  step at all, and the ILP64 library it needs is already inside every numpy wheel.

---

## 2. Background (all of this was verified empirically, numpy 2.4.4 / scipy 1.17.1)

The size limit is not a LAPACK property. It is an asymmetry in how the two wheels are
built:

| package | linked library | integer width |
|---|---|---|
| numpy 2.4.4 wheel | `scipy_openblas64` (`USE64BITINT`) | ILP64 |
| scipy 1.17.1 wheel | `scipy_openblas32` | LP64 |

`scipy.linalg.lapack.HAS_ILP64` is `False` in released wheels; `scipy.linalg.blas` too.
46341 = ⌊√(2³¹)⌋, i.e. the point where the element count of a square `float64` matrix
overflows a signed 32-bit integer. numpy's own wrappers and its bundled OpenBLAS are
64-bit throughout, which is why `np.linalg.eigh` works where `scipy.linalg.eigh` fails.

**Therefore the ILP64 LAPACK we need is already installed** as
`<site-packages>/numpy.libs/libscipy_openblas64_*.so`. Verified exports:
`scipy_dpotrf_64_`, `scipy_dpotrs_64_`, `scipy_dsyevd_64_`, `scipy_dsyev_64_`,
`scipy_dtrsm_64_`, `scipy_openblas_get_config64_`.

Measured peak RSS, n = 8000 (matrix = 0.48 GiB), `float64`:

| call | extra peak RSS |
|---|---|
| `scipy.linalg.cho_factor(A, overwrite_a=True)`, A C-ordered | +0.59 GiB |
| `np.linalg.cholesky(A)` | +0.98 GiB |
| draft `cho_factor(A)` | +0.02 GiB |
| `np.linalg.eigh(A)` | +1.44 GiB |
| draft `eigh(A, driver="evd")` | +0.48 GiB |
| draft `eigh(A, driver="ev")` | +0.01 GiB |

---

## 3. The ordering rule (read this before writing any wrapper)

A symmetric matrix in C order and the same matrix in Fortran order **are the same bytes**.
There is nothing to convert, in place or otherwise. `A.T` on a C-contiguous array is a
zero-cost view whose flags say `F_CONTIGUOUS`, and for symmetric `A` it is also
mathematically the same matrix. scipy's copy is pure waste: it is triggered by inspecting
`flags`, not content.

Consequences for the implementation:

- Accept **either** C- or F-contiguous input for any routine documented as symmetric, and
  never copy. Internally: take the pointer, pass `n` as both `n` and `lda`, and flip the
  `uplo` character when the caller's array is C-contiguous (`lower=True` on a C array
  means LAPACK sees `'U'`).
- Because only one triangle is read, exact symmetry is irrelevant to correctness — but
  `lower=` now selects *which triangle of the caller's buffer* is authoritative. Document
  this: for a kernel matrix assembled with only one triangle filled, the caller must pass
  the matching `lower=`.
- On output the same transposition applies: after in-place `eigh` of a C-ordered array,
  eigenvector *i* occupies **row** `A[i]`, i.e. the buffer holds `Qᵀ`. Say so in every
  docstring; this is the single most likely source of silent user error. Consider
  returning a view named `QT` rather than pretending it is `Q`.
- Non-contiguous input: raise, never copy silently. Add
  `ascontiguous_or_raise(a, name)` helper.

For the record, an actual in-place *reorder* (for a non-symmetric square matrix) is
possible — element-swap over the upper triangle — but it is cache-hostile and we never
need it. `A[:] = A.T` does not do it: numpy detects the overlap and allocates a temporary.

---

## 4. Package layout

```
bigla/
  __init__.py        # public API re-exports, version
  _backend.py        # library discovery, symbol decoration, ILP64 validation
  _core.py           # ctypes prototypes + thin per-routine wrappers
  linalg.py          # scipy-shaped public functions
  workspace.py       # Workspace object, lwork queries, reuse
  diagnose.py        # `python -m bigla.diagnose`
docs/
  backends.md        # the living install matrix (§9)
  conventions.md     # §3, expanded, with pictures
tests/
  test_correctness.py
  test_inplace.py
  test_backends.py
  long/              # minutes: environment-matrix rows, large-n workspace
  huge/              # opt-in, needs ~20 GiB
pyproject.toml       # deps: numpy; extra "openblas": scipy-openblas64
```

*(Amended after implementation: the expensive tests were originally selected by pytest
markers and live here as `test_huge.py`. Markers deselect only what is marked, and
`test_huge.py` was gated by a `skipif` on `BIGLA_TEST_HUGE` while carrying no `huge`
marker — so with that variable exported, `make test` selected and RAN the 20 GiB cases and
`make test-huge` then ran them again. Selection is now by directory, which cannot be opted
into by accident. The `skipif` was then dropped too: `tests/conftest.py` declines to collect
`long/` and `huge/` unless the invocation names them, so a bare `pytest` is safe, `pytest
tests/huge` works directly, and no environment variable is involved anywhere.)*

---

## 5. Backend discovery and validation

### 5.1 Search order

1. `$BIGLA_LIB` — explicit path, always wins. Fail loudly if it doesn't load.
2. `scipy_openblas64` package if importable (`scipy_openblas64.get_lib_dir()`).
3. numpy's bundled library: `glob(<site-packages>/numpy.libs/*openblas64*.so*)`.
   On Windows/macOS the directory is `numpy.libs` / `numpy/.dylibs` — handle both.
4. System: `libopenblas64_.so.0`, `libopenblas64.so.0`, `libflexiblas64.so`.
5. MKL: `libmkl_rt.so` (see 5.4 — needs the interface layer set).
6. Nothing found → see 5.5.

*(Amended after implementation: MKL was originally candidate 4 and system OpenBLAS 5. They are
swapped because probing MKL is not side-effect-free — see §5.4.)*

Load with `ctypes.CDLL(path)`. Record the resolved path; it goes in every error message and
in `diagnose`.

*(Amended after implementation: this originally said `mode=ctypes.RTLD_GLOBAL`, which was
removed as a defect — a candidate exporting undecorated `dpotrf_` (MKL, reference LAPACK built
`-i8`) becomes globally visible and can be bound by anything loaded afterwards, numpy included
if it loads later. `RTLD_LOCAL`, the ctypes default, suffices: symbols are resolved through our
own handle. Reintroduce it per-candidate, with a comment naming the backend, only if one turns
out to need it.)*

### 5.2 Symbol decoration

Probe in this order and cache the winning pattern:

| pattern | example | seen on |
|---|---|---|
| `scipy_` + name + `_64_` | `scipy_dpotrf_64_` | numpy/scipy wheels (verified) |
| name + `_64_` | `dpotrf_64_` | Debian/Ubuntu `libopenblas64-dev` |
| name + `64_` | `dpotrf64_` | some OpenBLAS builds |
| name + `_` | `dpotrf_` | MKL ILP64, reference LAPACK built with `-i8` |
| name + `$NEWLAPACK$ILP64` | `dpotrf$NEWLAPACK$ILP64` | Apple Accelerate (to verify) |

`getattr` on a `CDLL` raising `AttributeError` is the probe.

### 5.3 ILP64 validation — mandatory, do not skip

A wrong-width binding does not fail loudly: passing `int64` arguments to an LP64 routine
gives the right answer for small `n` on little-endian hardware (the low word is correct)
and silently corrupts at large `n`. **A numerical smoke test cannot detect this.** So
validate structurally:

- OpenBLAS: call `openblas_get_config` (decorated the same way, `restype = c_char_p`) and
  require the string to contain `USE64BITINT`.
  Verified output: `OpenBLAS 0.3.31.188.0  USE64BITINT DYNAMIC_ARCH NO_AFFINITY SkylakeX MAX_THREADS=64`.
- MKL: `mkl_set_interface_layer(1)` (`MKL_INTERFACE_ILP64`) returns the layer actually
  selected; require it to report ILP64. Must be called before the first BLAS call.
- Fallback evidence: a `_64_`/`64_` symbol suffix is itself strong evidence, since LP64
  builds do not use it. Accept it, but record `confidence="suffix-only"` in `diagnose`.
- If none of the above can be established, treat the backend as LP64 (5.5).

Expose the verdict as `bigla.backend_info()` → dataclass with
`path, config_string, decoration, ilp64, confidence, max_dim, threads`.

### 5.4 MKL notes

**Probing MKL mutates global state, which is why it is probed last.** `_validate_ilp64` calls
`mkl_set_interface_layer(1)` in order to *establish* whether ILP64 is available — but that call
changes the interface layer for the whole process, not just for bigla's handle. If numpy is
MKL-linked and running LP64, merely probing has altered a library numpy is using. Reaching MKL
only after the dedicated OpenBLAS64 candidates have been tried keeps that mutation to the cases
where MKL is genuinely the only option. It is also why the candidate list must not yield the
same soname twice: each repeat is another `mkl_set_interface_layer` call.


`libmkl_rt` dispatches at runtime; the interface layer must be ILP64 *before* first use,
either via `mkl_set_interface_layer(1)` or `MKL_INTERFACE_LAYER=ILP64` in the environment
(add `,GNU` when linked against GNU OpenMP). Because numpy in the same process may already
have initialised MKL in LP64 mode, prefer the dedicated OpenBLAS64 library over `mkl_rt`
whenever both are present, and warn if `numpy` is itself MKL-linked.

### 5.5 No ILP64 available

Do not fail at import. Set `ilp64=False`, `max_dim=46340`, and:

- for `n <= max_dim`: delegate to `scipy.linalg` (still applying our overwrite/ordering
  handling, which is a real win on its own — see §2 table), emitting no warning;
- for `n > max_dim`: raise `BiglaBackendError` with the resolved library path, what was
  probed, and a pointer to `docs/backends.md`.

---

## 6. API

Two layers. Everything defaults to `check_finite=False` (scipy's default costs an `n²`
boolean temporary — 61 MiB at n = 8000).

### 6.1 scipy-shaped (drop-in)

```python
cho_factor(a, lower=True, overwrite_a=False, check_finite=False) -> (c, lower)
cho_solve((c, lower), b, overwrite_b=False, check_finite=False) -> x
cho_inverse(c, lower=True)                       # dpotri
solve(a, b, assume_a="pos", ...)                 # potrf+potrs, no LU fallback
eigh(a, lower=True, eigvals_only=False, overwrite_a=False,
     driver="evd", subset_by_index=None, work=None) -> w | (w, Q)
eigvalsh(a, ...)
solve_triangular(a, b, lower=True, trans=0, overwrite_b=False)  # dtrtrs
```

*(Amended after implementation — this paragraph originally argued for `overwrite_a=True` as
the default, "a deliberate divergence from scipy". The implementation went the other way. The
signatures above therefore read `overwrite_a=False` / `overwrite_b=False` in the shipped code,
matching scipy exactly, and `eigh` returns `Q` rather than `QT`; see the amendment preamble at
the top of this file. The no-copy path is opt-in per call, and it is honoured for C-contiguous
input, which scipy copies regardless — capability preserved, footgun removed.)*

### 6.2 Explicit in-place layer

Thin, no-magic wrappers named after the LAPACK routines — `potrf`, `potrs`, `potri`,
`syevd`, `syev`, `syevr`, `trtrs`, plus `sygvd` if the generalized problem shows up.
These take pre-validated arrays and a `Workspace`, and are what the scipy-shaped layer
calls. Users doing tight CV loops should be able to drop to this layer.

### 6.3 Dtypes

`float64` first. Add `float32` (`spotrf`, `ssyevd`) by prefix dispatch — trivial once the
plumbing exists and halves memory, which matters here. Complex (`zpotrf`, `zheevd`) only
if a use case appears; keep the dispatch table ready for it.

### 6.4 Driver selection for `eigh`

Document the workspace cost, because it dominates:

| driver | LAPACK | workspace (doubles) | at n = 50 000 |
|---|---|---|---|
| `"evd"` | `dsyevd` | `2n² + 6n + 1` | **37 GiB** on top of the 18.6 GiB matrix |
| `"ev"` | `dsyev` | `~34n` | 14 MiB, in place, slower |
| `"evr"` | `dsyevr` | `~33n` + separate `n²` output `Z` | 18.6 GiB (cannot overwrite in place) |

(`lwork` formulas confirmed against `scipy.linalg.lapack.dsyevd_lwork(4000) = 32024001`.)
Default should be `"evd"` for `n` below a threshold and `"ev"` above it — or better, an
`auto` policy that reads `MemAvailable` from `/proc/meminfo` and picks. Make the policy
overridable and log the choice at DEBUG.

### 6.5 Workspace object

```python
ws = bigla.Workspace()          # grows as needed, reused across calls
w, QT = bigla.eigh(K, work=ws)
```

Rationale: a k-fold / λ-grid loop calls `eigh` or `potrf` repeatedly; allocating and
freeing a 37 GiB workspace each time is both slow and a fragmentation hazard. `Workspace`
holds `float64` and `int64` buffers, exposes `.nbytes`, and can be pre-sized from a query
(`Workspace.for_eigh(n, driver)`).

Implementation detail: the LAPACK workspace query returns `lwork` in a `float64`. For
`n > 46341`, `2n²` exceeds 2³¹ but is exactly representable well past 2⁵³, so read it as
`int(round(w[0]))` — do not cast through `int32` anywhere. `iwork` must be an `int64`
array (`3 + 5n`).

### 6.6 Calling convention details

- Pass every integer as `ctypes.c_int64` **by reference**.
- Append the hidden Fortran character-length arguments (`c_int64(1)` per `character*1`)
  after the visible arguments; gfortran ≥ 7 passes them as `size_t`.
- ctypes releases the GIL around `CDLL` calls, so LAPACK's own threading is unaffected and
  Python threads are fine.
- ctypes does no bounds checking: every wrapper validates `dtype`, `ndim`, squareness,
  contiguity, and shape agreement **before** taking a pointer. A shape bug here is a
  segfault, not an exception.
- Keep a reference to every array for the duration of the call (assign to a local; do not
  pass `.ctypes.data_as()` of a temporary expression).

### 6.7 Errors

`info < 0` → `ValueError(f"{routine}: illegal argument {-info}")`.
`info > 0` → `numpy.linalg.LinAlgError` with the LAPACK meaning spelled out (for `potrf`,
"leading minor of order k is not positive definite" — in KRR this almost always means the
regularizer is too small, so say that in the message).

### 6.8 Threading

Expose `get_num_threads()` / `set_num_threads(n)` via
`openblas_set_num_threads` / `mkl_set_num_threads` where available. Note in the docs that
this library's thread pool is *separate* from the one numpy is using if the two resolved
to different `.so` files — a common source of oversubscription on HPC nodes.

*(Extended during implementation.)* `set_num_threads(n)` **requires** `n`; there is no
auto-detected default. Neither numpy nor scipy exposes a thread-count API to copy a policy
from, and the nearest analogue — `scipy.fft.set_workers` — takes an explicit argument and
offers a context manager rather than guessing. A halve-and-cap heuristic borrowed from
application thread pools has the wrong goal here: bigla exists for the one calculation on the
node that should get the whole node.

Also expose `num_threads(n)` as a **context manager**, mirroring `scipy.fft.set_workers`:

```python
with bigla.num_threads(16):
    w, Q = bigla.eigh(K)
```

Scoped control is what callers actually want, and it composes with k-fold / λ-grid loops. It
warns on entry when the backend exposes no getter, since the previous value cannot then be
restored on exit.

For the multi-pool case — bigla and numpy resolved to different `.so` files, so both pools
oversubscribe — point users at `threadpoolctl.threadpool_limits`, which already discovers
OpenBLAS, MKL, BLIS, FlexiBLAS and OpenMP pools by the same symbol probing bigla does by hand
and limits all of them at once. Keep the two accessors regardless: they answer "what is *this*
handle doing", which threadpoolctl's aggregate view does not.

---

## 7. Memory-mapped input

`np.memmap` arrays work unchanged (they are contiguous and expose `.ctypes.data`), which
makes an out-of-core factorisation possible on a machine that cannot hold the matrix.
Performance will be poor but it beats not running. Add one test and a documented caveat;
do not build any tiling logic.

---

## 8. Testing

- **Correctness** (`n` ≤ 4000, always run): every routine against `scipy.linalg`, for
  C-ordered, F-ordered, and `A.T`-view input, `lower` in `{True, False}`, 1 and k RHS,
  `float64` and `float32`. Assert both the numerical result *and* `np.shares_memory`
  behaviour for each `overwrite_*` combination.
- **Convention tests**: after in-place `eigh` of a C-ordered array, assert
  `QT.T @ diag(w) @ QT ≈ A_original` — this is the test that catches a transposition
  regression.
- **Backend tests**: force each discovery path via `BIGLA_LIB`; assert the ILP64 verdict;
  assert that an LP64 library is correctly refused for `n > 46340` and correctly delegated
  below it.
- **Huge** (`BIGLA_TEST_HUGE=1`, needs ≈ 20 GiB and patience): `n = 46500`, i.e. just past
  the wall. Factor a diagonally dominant matrix, solve, check the residual. This is the
  test that justifies the package; keep it out of default CI but run it before each
  release and record the machine in `docs/backends.md`.
- **Non-finite input**: with `check_finite=False` LAPACK may loop or return garbage —
  document, and test that `check_finite=True` still catches it.

---

## 9. `docs/backends.md` — the living install matrix

This is the artefact worth accumulating. One row per environment actually tested, never
per environment guessed. Table columns:

`environment` | `numpy source` | `scipy source` | `library resolved` | `decoration` | `ILP64 verified by` | `max n tested` | `notes / gotchas`

Seed it with the one row verified so far:

> manylinux, pip numpy 2.4.4 + scipy 1.17.1 | wheel | wheel |
> `numpy.libs/libscipy_openblas64_-*.so` | `scipy_*_64_` | `openblas_get_config` →
> `USE64BITINT` | (huge test not yet run) | scipy's own LAPACK is LP64
> (`HAS_ILP64 is False`); we bind numpy's library, not scipy's

Rows to fill in, each marked **UNVERIFIED** until someone actually runs
`python -m bigla.diagnose` there:

- Debian/Ubuntu `libopenblas64-dev` (`/usr/lib/x86_64-linux-gnu/libopenblas64.so.0`) —
  check whether the decoration is `_64_` or bare.
- Fedora/RHEL `openblas64` packages.
- conda-forge: numpy there links `libblas` from the LP64 stack; **does conda-forge ship an
  ILP64 OpenBLAS at all?** Resolve this before claiming conda support. If not, the answer
  on conda is `pip install scipy-openblas64` into the env, which should be tested.
- HPC modules: Intel MKL via `module load`, Cray LibSci, ARM Performance Libraries
  (`armpl_ilp64`), NVIDIA nvpl. For each: the module name, whether `libmkl_rt` or an
  explicit `_ilp64` library is the right target, and the environment variables required.
- macOS Accelerate `$NEWLAPACK$ILP64`, and macOS + numpy wheel.
- Windows, if we care.

`python -m bigla.diagnose` must print exactly the fields the table needs, in a
paste-ready block, so adding a row is copy-paste. Include in it: resolved path, config
string, decoration, ILP64 verdict + how established, thread count, `max_dim`, numpy and
scipy versions and their `show_config()` BLAS names, and `MemAvailable`.

Also document the escape hatches prominently: `BIGLA_LIB=/path/to/lib.so`, and
`pip install bigla[openblas]` (pulls `scipy-openblas64`) as the always-works answer when
discovery fails.

---

## 10. Packaging / CI

- `pyproject.toml`, hatchling or setuptools, `requires-python = ">=3.10"`.
- Runtime deps: `numpy>=1.26` only. Extra `openblas` → `scipy-openblas64`.
  `scipy` is a *test* dependency (reference implementation), not a runtime one — the
  package must work in an environment with no scipy at all, except for the LP64 fallback
  path of §5.5, which should import scipy lazily.
- One universal wheel: `py3-none-any`. If a build step ever creeps in, something has gone
  wrong.
- CI matrix: numpy 2.0 / 2.2 / 2.4 / dev × {wheel OpenBLAS64, `scipy-openblas64` package,
  system `libopenblas64`} on Linux; plus one macOS job. Run `diagnose` in every job and
  upload its output as an artifact — that is how `docs/backends.md` gets populated
  cheaply.

---

## 11. Open questions for the implementer

1. Whether `dsyevr` with `range='I'` is worth exposing — for KRR one sometimes only needs
   the top-k spectrum, and it is far cheaper. Probably yes, but the `Z` output cannot
   alias `A`, so the memory story needs its own row in the §6.4 table.
2. Whether to offer a `potrf`-based `logdet` helper (marginal likelihood work needs it and
   it is two lines given the factor).
3. Whether the LP64 fallback should also apply the §3 ordering trick when delegating to
   scipy — it should, and that means the fallback is not a pure passthrough.
4. Whether to detect at import that numpy and this package resolved to *different* BLAS
   libraries, and warn about thread oversubscription.
