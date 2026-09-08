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
| manylinux, x86_64 | pip numpy 2.4.4 | pip scipy 1.17.1 | `<site>/numpy.libs/libscipy_openblas64_-*.so` | `scipy_NAME_64_` | config-string: `USE64BITINT` | (huge test not yet run) | scipy's own LAPACK is LP64 (`HAS_ILP64 is False`); we bind numpy's bundled library, not scipy's |
<!-- END GENERATED ROWS -->

---

## Environments to verify (UNVERIFIED)

Each entry below is a TODO.  Mark it verified only after `python -m bigla.diagnose`
runs cleanly there and a row is added to the table above.

### Linux distros

- **Debian/Ubuntu `libopenblas64-dev`**
  Library: `/usr/lib/x86_64-linux-gnu/libopenblas64.so.0`
  Open question: is the decoration `_64_` or bare `_`?

- **Fedora/RHEL `openblas64` package**
  Install: `dnf install openblas64`
  Open question: same decoration question.

### conda-forge

numpy on conda-forge links the LP64 BLAS stack by default.
**Does conda-forge ship an ILP64 OpenBLAS at all?**

- If yes: identify the package name and library path.
- If no: the answer is `pip install scipy-openblas64` into the conda env
  (should work; test it).  Add the resulting path to the table.

### HPC modules

For each: record the module command, the resolved library path, and any
required environment variables.

- **Intel MKL** (`module load intel` or `module load mkl`)
  Use `libmkl_rt.so`.  Must call `mkl_set_interface_layer(1)` (ILP64) or
  set `MKL_INTERFACE_LAYER=ILP64` **before** the first BLAS call.
  Warning: if numpy in the same process initialised MKL in LP64 mode,
  bigla will warn.  Prefer the OpenBLAS64 library when both are present.

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
  supported, the answer on macOS is `pip install bigla[openblas]`, which is what the CI
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

1. **`pip install bigla[openblas]`** — pulls `scipy-openblas64`, which ships
   its own ILP64 OpenBLAS.  This is the always-works answer on any platform
   where pip works.

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
