# AGENTS.md — bigla

Agent-oriented notes for this repo. (Compressed from the source; keep in sync when
`_core.py` / `_backend.py` change.)

> `README.md` is user-facing and current; this file is the agent-facing companion.
> `docs/SPEC.md` is the ORIGINAL design spec, kept as a record — it is amended in place
> where the implementation deliberately went the other way, so read its amendment notes
> rather than its prose. `docs/status.md` lists what is fixed-but-untested.

## What this is

An ILP64 LAPACK layer for dense-symmetric matrices, in pure ctypes. No compiled code, no
build step, `py3-none-any`. It exists for three properties `scipy.linalg` does not give:

1. **No 46341×46341 ceiling** — scipy's wheels link LP64; numpy's bundled OpenBLAS is ILP64,
   and bigla binds *that*.
2. **Overwrite flags that are honoured**, including for C-contiguous input, which scipy copies.
3. **Explicit workspace and driver control** — past n ≈ 50 000 the LAPACK workspace, not the
   matrix, is what exhausts the node.

## Mental model

Public layer (`linalg.py`) is a **drop-in for `scipy.linalg`**: same signatures, same defaults
(`overwrite_a=False`), eigenvectors as **columns**. Low layer (`_core.py`) is the explicit
in-place layer named after the LAPACK routines, in-place by default. `_backend.py` does
discovery; `workspace.py` holds reusable scratch.

## The ordering rule — the thing to get right

A symmetric matrix in C order and in Fortran order **are the same bytes**. So bigla never
copies to change layout; it flips `uplo` instead. Two consequences that have already caused
two blocker defects:

- **`_uplo_char(lower, c_order, symmetric=True)`** is `use_upper = lower == c_order`. If you
  ever find yourself writing `lower ^ c_order`, that is the D1 defect returning. It is pinned
  by a literal truth table in `tests/test_conventions.py::test_uplo_truth_table`.
- **The flip is free only because A is symmetric.** For a *triangular* matrix the Fortran view
  is genuinely Aᵀ, so `trans` must flip too. Use
  **`_uplo_trans_triangular(lower, c_order, trans)`**, which returns both; `_uplo_char` raises
  on `symmetric=False` precisely so the pair cannot be taken half-applied. That was D2:
  `trtrs` inherited the symmetric rule from its neighbours and silently solved Aᵀx = b.

`_symmetrise`'s `upper_filled = not lower` looks order-independent and is: the uplo flip and
the buffer transposition compose to the identity. The four-row table in its docstring is the
**uplo choice**, not the filled numpy triangle.

## Testing rules that this repo learned the hard way

The suite passed 84/84 both before and after two blockers. Three rules follow, and
`tests/test_conventions.py` documents them at the top:

- **R1** — a *symmetric* test matrix cannot test a triangle convention. Poison the triangle
  `lower=` declares irrelevant (helper: `poisoned()`), or you are testing nothing.
- **R2** — assert conventions as **literal tables**, not only through behaviour. End-to-end
  tests show a convention is self-consistent, not that it is the documented one.
- **R3** — round-trip self-consistency is not correctness. `potrf`/`potrs` share a `uplo`
  expression, so an inversion cancels and `cho_factor → cho_solve` returns the right `x`.
  Check each routine against **scipy**, not against its own inverse.

Acceptance rule for any fix: a named test that **fails against the tree that preceded it**.

## Behaviors that bite

- **`import bigla` succeeding does not mean ILP64.** Discovery falls back to LP64 with
  `max_dim=46340` rather than failing at import. Branch on **`bigla.is_ilp64()`**, or
  `n <= backend_info().max_dim`.
- **Loading is two-pass.** First pass returns only a verified-ILP64 candidate; LP64 ones are
  kept aside and used only if nothing better appears. Returning the first library with LAPACK
  symbols was D3 — MKL stuck in LP64 would shadow a working OpenBLAS64 later in the list.
- **Probing MKL mutates global state.** `_validate_ilp64` calls `mkl_set_interface_layer(1)`,
  which changes the layer process-wide — including for a numpy that is MKL-linked. Hence MKL
  is the *last* candidate, and the candidate list must never yield the same soname twice.
- **`RTLD_LOCAL` is deliberate.** `RTLD_GLOBAL` publishes undecorated `dpotrf_` to everything
  loaded afterwards. Do not reintroduce it globally.
- **The workspace floor is `jobz`-dependent.** Flooring an eigenvalues-only call at the
  eigenvector minimum reserves ~2n² doubles it never touches — 37 GiB of address space at
  n = 50 000. Invisible in RSS, fatal under `vm.overcommit_memory=2`, `ulimit -v`, or an
  address-space cgroup. That was D6.
- **`set_num_threads(n)` requires `n`.** No auto-detected default; use `num_threads(n)` as a
  context manager for scoped control. It controls *this* handle only — `threadpoolctl` is the
  answer when numpy resolved to a different `.so`.
- **`diagnose`'s table row is positional and uncoupled.** It prints 8 pipe-separated fields
  that must match, in order, the 8 columns declared in SPEC §9 and used by `docs/backends.md`.
  Nothing checks this: add a column to the table and every previously-generated row silently
  misaligns. Change both together, or add the assertion first.

## Repo layout

```
bigla/__init__.py    public re-exports; __all__ is asserted clean by tests/test_docs.py
bigla/_backend.py    discovery, decoration probing, ILP64 validation, threading
bigla/_core.py       ctypes prototypes + per-routine wrappers (in-place layer)
bigla/linalg.py      scipy-shaped public API
bigla/workspace.py   reusable scratch; Workspace.for_eigh
bigla/diagnose.py    python -m bigla.diagnose -> paste-ready docs/backends.md row
docs/SPEC.md         original spec, amended in place
docs/status.md       fixed-in-code, not-yet-tested
docs/conventions.md  the ordering rule, expanded
```

## Dev workflow

`make dev-setup` installs the pre-commit hooks (black/isort/flake8/autoflake at width 99,
shared with qml2-dev). There are deliberately **no** `make lint` / `make fmt` targets —
duplicating the hook arguments is how a Makefile drifts out of sync with what gates a commit.
`make test` excludes the `long` and `huge` markers; `make review` runs every hook over the tree.
