# Known-open items

What is deliberately not done yet, so that "deferred" and "forgotten" do not look identical to
whoever reads this repo later. Every row here has landed *code* but is **not closed**, because
this project's acceptance rule is:

> A defect is closed only when a named test covering it runs under `make test` or
> `make test-long` **and** fails against the tree that preceded the fix.

Rows disappear from this file when their test exists, not when the code looks right.

## Open: fixed in code, no test

| item | what was fixed | test still needed |
|---|---|---|
| **D3** | Two-pass loading: an LP64 library no longer ends the search when a working ILP64 candidate sits later in the list. | `test_prefers_ilp64_over_earlier_lp64`, `test_lp64_only_falls_back_with_warning`, `test_explicit_bigla_lib_lp64_is_honoured` |
| **D4** | `_resolve_path` no longer fabricates `<cwd>/libmkl_rt.so`; resolves via `dlinfo(RTLD_DI_LINKMAP)`, `/proc/self/maps` as fallback, normalised with `realpath` when the path exists. | `test_resolved_path_exists`, `test_resolve_path_never_joins_cwd` |
| **5.4** | `RTLD_GLOBAL` dropped, so undecorated symbols such as `dpotrf_` are not published to everything loaded afterwards. | `test_no_global_symbol_leak` |

All three need a fake `CDLL` — monkeypatch `_candidates` to yield stub names and `ctypes.CDLL`
to return stubs advertising chosen symbols and `openblas_get_config` strings. No containers
required; the discovery *logic* is pure and runs in milliseconds.

Two things make these more urgent than their "deferred" label suggests:

- **D3 has no other evidence.** It was found by reading the code path, not by reproducing it —
  no MKL was present in any review environment. Until the stub test exists, nothing anywhere
  demonstrates the two-pass loader behaves as intended.
- **D4's code is now newer and larger** than when the test was first deferred: `dlinfo` plus a
  `/proc` fallback plus `realpath` normalisation, all uncovered.

`tests/test_backends.py` currently passes 22 of 24 against the pre-fix tree, which is the
measurement showing it does not exercise the loader at all.

## Deferred by agreement

| item | note |
|---|---|
| **SPEC §5.6** — CI | `.github/workflows/` does not exist. Until it does, `docs/backends.md` has one row and no mechanism to gain more. |
| **SPEC §6** — environment matrix | Six rows by backend provenance, one parameterised Dockerfile. This is where D3 gets its real-world coverage; the stub tests above cover the logic, not the decorations. |

## Closed

D1 (`lower=` inverted), D2 (`solve_triangular` transposed for C-order), D6 (lwork floor ignored
`jobz`), 5.1, 5.2, 5.3, 5.5 — each with tests verified red against the preceding tree.
