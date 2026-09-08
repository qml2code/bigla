# bigla — Memory layout and ordering conventions

This document expands §3 of the spec.  Read it before writing any wrapper.

---

## The key insight: symmetric means C = Fortran (for reads)

A symmetric matrix stored in C order (row-major) and the same matrix stored
in Fortran order (column-major) occupy **the same bytes**.  The only
difference is which index varies fastest, but for a symmetric matrix
`A[i,j] == A[j,i]` — so the distinction is irrelevant to the values.

Concretely, `A.T` on a C-contiguous array is a **zero-cost view**:
no data is moved, and the resulting array has `flags.f_contiguous == True`.
For a symmetric matrix `A`, `A.T` is mathematically identical to `A`.

scipy's copy in `cho_factor` / `eigh` is therefore pure waste: it is
triggered by inspecting `flags`, not content.  bigla avoids it.

---

## The uplo flip

LAPACK's routines take an `uplo` argument: `'U'` means "use the upper
triangle", `'L'` means "use the lower triangle".  LAPACK is column-major
(Fortran order).

When you pass a C-contiguous array to LAPACK:

- What your code calls the "lower triangle" (`A[i,j]` for `i > j`)
  is, in memory-order terms, stored in the positions that LAPACK sees as the
  **upper** triangle.
- Therefore: `lower=True` on a C-contiguous array → pass `uplo='U'` to LAPACK.
- And: `lower=False` on a C-contiguous array → pass `uplo='L'` to LAPACK.

For F-contiguous (or transposed-view) arrays, the mapping is direct:
`lower=True` → `uplo='L'`.

The helper `_uplo_char(lower, c_order)` in `_core.py` encodes this:

```python
use_upper = lower ^ c_order   # XOR: flip when C-order
return 'U' if use_upper else 'L'
```

---

## What `lower=` means to the caller

`lower=True` (the default) means: **the lower triangle of the array you passed
is the authoritative data**.  The upper triangle may be garbage (or
uninitialised).  This matches the convention for kernel matrices assembled
with only the lower triangle filled.

If you assembled the upper triangle, pass `lower=False`.

---

## Output convention for `eigh`: eigenvectors are COLUMNS, like scipy

`bigla.eigh` returns `(w, Q)` with the same convention as `scipy.linalg.eigh`:

- Eigenvalues `w[i]`, ascending.
- Eigenvector `i` is the **column** `Q[:, i]`.

```python
w, Q = bigla.eigh(A)
A_reconstructed = Q @ np.diag(w) @ Q.T
```

or, without forming `diag(w)`:

```python
A_reconstructed = (Q * w) @ Q.T      # (Q * w)[i, j] = w[j] * Q[i, j]
```

### What the buffer actually holds

LAPACK writes the eigenvectors as **rows** of the working buffer — that is `Qᵀ`. The
public API returns `Q` by handing back the transposed view:

```python
Q = buffer.T        # a view: no copy, no allocation, only relabelled strides
```

So mirroring scipy costs nothing at all, and the row-major form is still one free `.T`
away when it is the more convenient shape:

```python
QT = Q.T            # eigenvector i is QT[i]; also a view
```

For KRR the common operation `K_pred @ Q @ np.diag(1/w) @ Q.T` works directly with the
column convention, and `Q.T` gives the row form for the cases where that is cheaper.

**Historical note.** Earlier drafts returned `Qᵀ` directly and documented it as the single
most likely source of silent user error. That was a real hazard: an implementation that
forgets the transpose still passes every reconstruction test, because `Q diag(w) Qᵀ` is
symmetric in the mistake. It is now pinned by
`test_correctness.py::test_eigh_eigenvectors_match_scipy_orientation`, which compares
against scipy element by element rather than checking a reconstruction.

---

## Non-contiguous input

bigla raises `ValueError` for non-contiguous input rather than copying
silently.  This is intentional: the whole point of the library is to avoid
copies, and silently making one would defeat that purpose.

To copy explicitly:

```python
A_c = np.ascontiguousarray(A)
w, QT = bigla.eigh(A_c)
```

---

## Diagram

```
Caller's view (C-order)          LAPACK's view (Fortran-order)
─────────────────────────        ──────────────────────────────
lower=True:                      uplo='U' (upper triangle):
  A[i,j] for i >= j              A[j,i] for j <= i
  (lower left)                   (lower left = upper right in Fortran)

lower=False:                     uplo='L' (lower triangle):
  A[i,j] for i <= j              A[j,i] for j >= i
  (upper right)
```

For F-contiguous input the mapping is direct (no flip).
