# Known-open items

What is deliberately not done yet, so that "deferred" and "forgotten" do not look identical to
whoever reads this repo later. Every row here has landed *code* but is **not closed**, because
this project's acceptance rule is:

> A defect is closed only when a named test covering it runs under `make test` or
> `make test-long` **and** fails against the tree that preceded the fix.

Rows disappear from this file when their test exists, not when the code looks right.

## Open: fixed in code, no test

None. D3, D4 and 5.4 were the last three and closed on 2026-09-08 with `tests/test_loader.py`.

## Deferred by agreement

| item | note |
|---|---|
| **SPEC §5.6** — CI | `.github/workflows/` does not exist. Until it does, `docs/backends.md` has one row and no mechanism to gain more. |
| **SPEC §6** — environment matrix | Six rows by backend provenance, one parameterised Dockerfile. This is where D3 gets its *real-world* coverage; the stub tests in `test_loader.py` cover the discovery logic, not the decorations an actual MKL or Debian OpenBLAS exports. |

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
