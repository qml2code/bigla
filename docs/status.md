# Known-open items

What is deliberately not done yet, so that "deferred" and "forgotten" do not look identical to
whoever reads this repo later. Every row here has landed *code* but is **not closed**, because
this project's acceptance rule is:

> A defect is closed only when a named test covering it runs under `make test` or
> `make test-long` **and** fails against the tree that preceded the fix.

Rows disappear from this file when their test exists, not when the code looks right.

## Open: fixed in code, no test

None. D3, D4 and 5.4 were the last three and closed on 2026-09-08 with `tests/test_loader.py`.
CI (FIXES §5.6) closed the same day with `.github/workflows/ci.yml`.

## Open: landed but unexercised

None. The environment matrix was the last one; see **FIXES §6** below.

What the matrix found on its way to green, none of which was reachable any other way: a
segfault in `set_num_threads` on Debian and at *import* on MKL, from calling Fortran-binding
symbols by value; a `set_num_threads` return value that echoed its argument and so lied on
any backend that clamps (MKL) or cannot thread at all (Fedora's serial build); and the
decoration of both distro builds, which the repo had been guessing at since its first commit.

Nothing in the environment matrix is open. The remaining unknowns are the environments it
cannot reach at all — see below.

### Deferred: a container with an agent in it, for environment-specific bugs

The MKL leak was the first defect this project could not reproduce on the authoring machine,
and the loop it forced — guess, commit, wait for CI, read one bit — is a bad instrument. The
approach to try next time: build the row's image, install Node and `@anthropic-ai/claude-code`
into it, mount the repo read-write and let an agent work *inside* the environment, committing
to a branch that is pushed from the host afterwards (a token or SSH key inside a throwaway
container is the part to avoid).

It is deliberately **not** set up now, and the MKL leak is the evidence for both halves of that.
Three CI rounds returned one bit each and produced two wrong hypotheses; a single `docker run`
of a ~40-line script printing `dladdr` provenance and `/proc/self/maps` answered it outright.
So: a container is the right instrument, an agent inside it was not needed here. What would
justify the machinery is the exploratory case — an architecture that cannot be reached from
here at all, where the work is many small experiments rather than one script:
`--platform linux/arm64`, or macOS Accelerate, where `_candidates()` has no system-library
path whatsoever and `docs/backends.md` still lists the question as open.

What the matrix still cannot tell you: whether any of this works on macOS Accelerate,
Windows, or the HPC module stacks (Cray LibSci, ARM Performance Libraries, NVIDIA nvpl).
Those have no container form and stay UNVERIFIED in `docs/backends.md` until someone runs
`python -m bigla.diagnose --format=json` on real hardware and adds the row.

## Closed

| item | what was fixed | discriminating test |
|---|---|---|
| **D1** | `lower=` inverted | `test_conventions.py` |
| **D2** | `solve_triangular` transposed for C-order | `test_conventions.py` |
| **D3** | Two-pass loading: an LP64 library no longer ends the search when a working ILP64 candidate sits later in the list. | `test_prefers_ilp64_over_earlier_lp64`, `test_lp64_only_falls_back_with_warning` |
| **D4** | `_resolve_path` no longer fabricates `<cwd>/libmkl_rt.so`; resolves via `dlinfo(RTLD_DI_LINKMAP)`, `/proc/self/maps` as fallback, normalised with `realpath` when the path exists. | `test_resolve_path_never_joins_cwd`, `test_resolve_path_prefers_dlinfo_over_prefix_match` |
| **D6** | lwork floor ignored `jobz` | `test_workspace.py` |
| **5.1** | docstring defaults | `test_documented_defaults_match_signatures` |
| **5.2** | duplicate `__all__` entry | `test_all_is_clean` |
| **5.3** | thread default | `test_backends.py` (3 tests) |
| **5.4** | `RTLD_GLOBAL` dropped, so undecorated symbols such as `dpotrf_` are not published to everything loaded afterwards. | `test_cdll_loaded_without_rtld_global`, `test_no_global_symbol_leak` |
| **5.5** | lwork guard | `test_workspace.py` |
| **FIXES §5.6** | No CI. `.github/workflows/ci.yml` runs lint plus the fast suite over the numpy axis SPEC §10 specifies (2.0 / 2.2 / current x py3.10/3.12/3.13, wheel and standalone `scipy-openblas64`, plus a macOS leg), and uploads each leg's `diagnose --format=json`. | `tests/test_matrix_config.py` pins the row definitions; the workflow itself is exercised by running. |
| **FIXES §6** — environment matrix | All six rows of run 34354219949 green, 2026-09-09. Every row runs the fast suite *and* `tests/long`, so each asserts the resolved path, decoration, confidence and ILP64 verdict of its own environment; `docs/backends.md`'s table is generated from that run and the hand-entered seed row is gone. | `tests/long/test_env_matrix.py` per row; `tests/test_matrix_config.py` pins the row definitions |
| **`conda-mkl` symbol leak** | Not a defect. `dpotrf_64_` does go global after bigla loads `libmkl_rt.so.3`, but `dladdr` reports the provider as `libmkl_intel_ilp64.so.3` — a file bigla never opened. `libmkl_rt` is a dispatcher and loads its own implementation globally on the first call that initialises MKL (for bigla, `_query_threads`; `MKL_Set_Interface_Layer` only records a preference, which is why two attempts to reproduce it in-process failed). | `test_no_global_symbol_leak`, which now attributes by provenance and skips naming the provider |

### How D3/D4/5.4 were verified red

Their fixes predate this repo's first commit, so there is no earlier tree to check out. The
verification was done by copying the tree and reverting the three behaviours by hand — a
single-pass `_load()`, `_resolve_path` returning `os.path.realpath(hint)`, and
`ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)` — then running `tests/test_loader.py` against the
copy. Result: **6 failed, 4 passed**. Redo it that way if these ever need re-confirming.

The four that pass either way are deliberate and are not evidence for their rows:

- `test_lp64_backend_info_reports_lp64` and `test_explicit_bigla_lib_lp64_is_honoured`
  characterise behaviour a single-pass loader shares. They guard against a *future* second pass
  becoming over-eager and overriding an explicit `BIGLA_LIB`.
- `test_resolved_path_exists` only bites for a bare soname; the absolute-path case it also covers
  survived the D4 bug. `test_resolve_path_never_joins_cwd` is the discriminating half.
- `test_lp64_max_dim_is_enforced` covers the `_check_dim` guard, not the loader. It fakes
  `max_dim=4` because the real boundary is 46341 and allocating that is ~17 GiB.
