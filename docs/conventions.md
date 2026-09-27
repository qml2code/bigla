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

## General matrices: no uplo, and the transpose is genuine

LU, QR, least squares and SVD read the whole matrix, so `uplo` does not apply. The layout question
does not go away, though — it gets worse. For a symmetric matrix, reinterpreting a C-contiguous
buffer as Fortran gives `Aᵀ = A` and costs nothing. For a general matrix it gives a *different
matrix*, and every routine has to account for it.

`_validate_2d(a, name, dtype)` is the general-matrix counterpart of the uplo flip. It returns
`(m, n, transposed)` — the dimensions of the matrix LAPACK will operate on, and whether that matrix
is the caller's `a` transposed. Nothing is copied; the transpose is propagated into the algebra:

| routine | C-contiguous input | mechanism |
| --- | --- | --- |
| `lu_factor` / `lu_solve` | works, no copy | `LUFactor.transposed` is XORed into `getrs`'s `trans`, so `trans=0` means the caller's `a x = b` in either order |
| `lstsq` | works, no copy of `a` | `gels` takes `trans`; setting it from the memory order solves the intended problem exactly (`b` is copied — `gels` needs a `max(m, n)`-row buffer) |
| `svd` / `svdvals` | works, no copy | `aᵀ = Ũ s Ṽᵀ` ⟹ `a = Ṽ s Ũᵀ`; swap and transpose the factors, both zero-copy views |
| `qr` | **raises** | the QR of `aᵀ` is an LQ of `a` — no identity recovers `(Q, R)`. The error names `lstsq`/`svd` instead |

The asymmetry is worth internalising: three of the four absorb the transpose for free because their
LAPACK drivers take a `trans` flag or because the decomposition is symmetric under transposition.
QR has neither property, so it is the only routine in bigla that refuses a contiguous array.

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
