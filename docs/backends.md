# bigla — Backend Install Matrix

This document is the living record of environments where `bigla` has been
tested.  **Only add a row after running `python -m bigla.diagnose` on the
actual machine.**  Never add a row based on guesswork.

---

## Table columns

| Column | Meaning |
|--------|---------|
| `environment` | OS / distro / HPC cluster name |
| `numpy source` | How numpy was installed and its version |
| `scipy source` | How scipy was installed and its version (or "not installed") |
| `library resolved` | Full path printed by `diagnose` |
| `decoration` | Symbol prefix/suffix pattern, e.g. `scipy_NAME_64_` |
| `ILP64 verified by` | `config-string`, `mkl-interface`, `suffix-only`, or `lp64` |
| `max n tested` | Largest `n` for which `cho_factor` / `eigh` were verified |
| `notes / gotchas` | Anything non-obvious |

---

## Verified environments

**This table is generated.** Rows come from `python -m bigla.diagnose --format=json` run in
each environment-matrix row and collected by `tools/render_backends.py`; do not hand-edit
between the markers. A row here is a claim that `diagnose` actually ran on that machine, and
hand-editing makes the claim cheap.

<!-- BEGIN GENERATED ROWS -->
| environment | numpy source | scipy source | library resolved | decoration | ILP64 verified by | max n tested | notes / gotchas |
|---|---|---|---|---|---|---|---|
| Debian trixie, libopenblas64-dev | numpy 2.2.4 (blas) | scipy 1.15.3 | `/usr/lib/x86_64-linux-gnu/openblas64-pthread/libopenblas64p-r0.3.29.so` | `NAME_` | config-string: `USE64BITINT` | - | BARE decoration, so only openblas_get_config -> USE64BITINT distinguishes it from LP64; the pthread variant, which libopenblas64-dev chose, not us |
| Debian trixie, no ILP64 present | numpy 2.2.4 (blas) | scipy not installed | **none found** | `-` | refused: no ILP64 backend | - | the refusal path: no ILP64 anywhere, so the error message is the deliverable |
| Fedora 44, openblas-serial64_ | numpy 2.4.6 (flexiblas) | scipy 1.16.2 | `/usr/lib64/libopenblas64_-r0.3.29.so` | `NAME_64_` | config-string: `USE64BITINT` | - | distro numpy goes through FlexiBLAS while bigla binds OpenBLAS64 directly -- two pools by construction; serial build, so threads=1 |
| conda-forge, MKL | numpy 2.5.3 (blas) | scipy 1.18.0 | `/opt/conda/lib/libmkl_rt.so.3` | `NAME_64_` | mkl-interface: `MKL ILP64 interface confirmed` | - | libmkl_rt is a dispatcher: it loads libmkl_intel_ilp64/core/thread globally itself, so MKL symbols are process-wide whatever mode we dlopen with |
| manylinux, pip numpy wheel | numpy 2.5.3 (scipy-openblas) | scipy 1.18.1 | `/usr/local/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-f48b354e.so` | `scipy_NAME_64_` | config-string: `USE64BITINT` | - | the common case: numpy's bundled ILP64 OpenBLAS, reached before any system library |
| manylinux, scipy-openblas64 package | numpy 2.5.3 (scipy-openblas) | scipy 1.18.1 | `/usr/local/lib/python3.12/site-packages/scipy_openblas64/lib/libscipy_openblas64_.so` | `scipy_NAME_64_` | config-string: `USE64BITINT` | - | the documented escape hatch; resolves a different file from the wheel row, ahead of it |
<!-- END GENERATED ROWS -->

---

## Notes on the verified environments

The table above says *what* resolved. These are the things that cost effort to find out and
that a path alone does not convey.

- **Debian/Ubuntu `libopenblas64-dev`** — resolved path is
  `/usr/lib/x86_64-linux-gnu/openblas64-pthread/libopenblas64p-r0.3.29.so`, reached by the
  `libopenblas64.so.0` soname. (This file previously documented the soname itself as the
  path; the matrix corrected it.)
  **Decoration is bare `NAME_`.** The symbols are plain `dpotrf_`, so nothing in the
  library's *name or decoration* distinguishes it from an LP64 build —
  `openblas_get_config` → `USE64BITINT` is the only thing that does. This is the one
  environment in the matrix where the confidence machinery is load-bearing rather than
  corroborating, which makes it the row to keep if any are ever dropped.

- **Fedora `openblas-serial64_`** — *not* `openblas64`, which is not a package name at all.
  Fedora splits OpenBLAS by threading model **and** by interface, and the trailing
  underscore selects the suffixed build:
  `openblas-serial` is LP64, `openblas-serial64` is ILP64 with **bare** symbols (like
  Debian's), `openblas-serial64_` is ILP64 with the `64_` suffix — decoration `NAME_64_`.
  No `-devel` is needed: the runtime subpackage ships the versioned soname discovery opens
  first. Note the row's numpy comes from the distro and goes through **FlexiBLAS**, so numpy
  and bigla are using different libraries in that container — the two-pool situation the
  README warns about, here by construction.

---

## Environments to verify (UNVERIFIED)

Each entry below is a TODO.  Mark it verified only after `python -m bigla.diagnose`
runs cleanly there and a row is added to the table above.

### Linux distros

None outstanding — Debian and Fedora are both verified; see the table and the notes
above it.

### conda-forge

numpy on conda-forge links the LP64 BLAS stack by default. The `conda-mkl` matrix row
verifies one answer — conda-forge's **MKL** is ILP64-capable and bigla binds it
(`libmkl_rt.so.3`, `NAME_64_`, confirmed through the interface layer).

Still open: **does conda-forge ship an ILP64 OpenBLAS?** If yes, identify the package and
path. If no, `pip install scipy-openblas64` into the conda env is the answer (it works on
manylinux; untested inside a conda env).

### HPC modules

For each: record the module command, the resolved library path, and any
required environment variables.

- **Intel MKL** (`module load intel` or `module load mkl`)
  Use `libmkl_rt.so`.  Must call `MKL_Set_Interface_Layer(1)` (ILP64) or
  set `MKL_INTERFACE_LAYER=ILP64` **before** the first BLAS call.
  Warning: if numpy in the same process initialised MKL in LP64 mode,
  bigla will warn.  Prefer the OpenBLAS64 library when both are present.

  Two gotchas measured in a conda-forge container on 2026-09-09:

  **`libmkl_rt` publishes MKL's symbols globally, and nothing can stop it.**
  It is a dispatcher. bigla opens it `RTLD_LOCAL`, but on the first call that
  *initialises* MKL it loads `libmkl_intel_ilp64`, `libmkl_core` and
  `libmkl_intel_thread` itself, globally — `/proc/self/maps` shows no MKL
  objects before `backend_info()` and four after, and `dladdr` reports
  `dpotrf_64_` as coming from `libmkl_intel_ilp64.so.3`, which bigla never
  opened.  So on an MKL backend, undecorated MKL symbols *are* visible
  process-wide however carefully you dlopen.  If that matters to you, use the
  OpenBLAS64 path instead.

  **`MKL_Set_Interface_Layer` does not initialise MKL**, it only records a
  preference; the implementation libraries load on the first real API call
  (for bigla, the thread-count query).  Worth knowing before writing anything
  that tries to observe MKL's loading behaviour.

  Decoration is `NAME_64_`: MKL's single dynamic library exports `_64`-suffixed
  ILP64 entry points beside the LP64 ones, and `_DECORATIONS` reaches
  `("", "_64_")` before `("", "_")`.

- **Cray LibSci ILP64**
  Usually `libsci_cray_mp.so`; needs `CRAY_CPU_TARGET` set.
  Open question: does Cray LibSci use the `_64_` suffix?

- **ARM Performance Libraries** (`armpl_ilp64`)
  `libarmpl_ilp64.so`.  Decoration likely `_64_`.

- **NVIDIA nvpl**
  `libnvpl_lapack_ilp64.so`.  Decoration unknown — needs probing.

### macOS

- **macOS Accelerate `$NEWLAPACK$ILP64`**
  Symbol decoration: `dpotrf$NEWLAPACK$ILP64`.
  Requires macOS 13.3+ and linking `-framework Accelerate`.
  Open question: does the ctypes approach work, or does the Mach-O
  re-export layer interfere?

- **macOS + numpy wheel** (same as manylinux path but `.dylibs`)
  Library: `<site>/numpy/.dylibs/libscipy_openblas64_*.dylib`
  **Does not exist on arm64.** CI (2026-09-08, `macos-latest`) found no backend at all:
  a stock arm64 numpy links Accelerate and bundles no OpenBLAS, so the `.dylibs` glob
  matches nothing — and every candidate after it in `_candidates()` is a Linux `.so`
  soname, so discovery has no macOS system-library path whatsoever. Until Accelerate is
  supported, the answer on macOS is `pip install scipy-openblas64`, which is what the CI
  macOS leg now uses. Adding `.dylib` sonames to `_candidates()` is untested speculation
  and deliberately not done.

### Windows

Low priority.  If tested: record DLL name and loader path.

(An earlier draft asked whether `RTLD_GLOBAL` has a Windows equivalent. It no longer matters:
loading is `RTLD_LOCAL` on every platform, since symbols are resolved through our own handle
and global visibility would let an undecorated `dpotrf_` be bound by anything loaded later.)

---

## Escape hatches

If `python -m bigla.diagnose` shows `ilp64: False` or fails to find a library:

1. **`pip install scipy-openblas64`** — ships its own ILP64 OpenBLAS, which
   discovery finds ahead of everything else.  This is the always-works answer on
   any platform where pip works.  (`bigla[openblas]` is the same dependency as an
   extra, for an install that goes through the package rather than a clone.)

2. **`BIGLA_LIB=/path/to/libopenblas64.so`** — point bigla at any ILP64
   library already on the machine.

3. **System package**: `apt install libopenblas64-dev` or equivalent, then
   verify with `python -m bigla.diagnose`.

---

## Changelog

- initial seed from the manylinux pip wheel environment, hand-entered before the matrix
  existed. It is the one row not produced by `tools/render_backends.py`, and the first
  matrix run replaces it.
- 2026-09-08: table placed under generation markers; `Dockerfile` + `.github/workflows/`
  added, so rows are collected rather than transcribed.
- 2026-09-09: **the seed row is gone.** All six rows of run 34354219949 are generated, and
  every one of them ran `tests/long`, so each line is backed by assertions on the resolved
  path, decoration, confidence and ILP64 verdict rather than by a paste. Debian and Fedora
  moved out of UNVERIFIED; Debian's documented path was wrong (the soname, not the file it
  resolves to) and the matrix corrected it.
