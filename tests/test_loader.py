"""Loader tests: two-pass ILP64 preference, path resolution, symbol visibility.

Closes D3, D4 and 5.4 from ``docs/status.md``. All three had landed code with no test,
because all three need a fake ``CDLL``: the discovery logic is pure and runs in
milliseconds, but it is only reachable by controlling what ``ctypes.CDLL`` returns.

D3 in particular had no other evidence -- it was found by reading the code path, not by
reproducing it, since no MKL was present in any review environment. These stubs are the
only thing anywhere that demonstrates the two-pass loader behaves as intended.

Each test below fails against the tree that preceded its fix:

* **D3** -- the round-1 loader returned the first candidate that answered to LAPACK
  symbols, so an LP64 library ended the search and a working ILP64 one further down the
  list was never examined.
* **D4** -- ``_resolve_path`` applied ``os.path.realpath`` to a bare soname, fabricating
  ``<cwd>/libmkl_rt.so``: a file that does not exist, reported as the resolved path and
  written into the ``docs/backends.md`` rows ``diagnose`` generates.
* **5.4** -- the library was loaded with ``mode=ctypes.RTLD_GLOBAL``, publishing
  undecorated names such as ``dpotrf_`` to everything loaded afterwards, numpy included
  if it loads later.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import subprocess
import sys

import numpy as np
import pytest

import bigla
from bigla import _backend
from bigla._backend import (
    BiglaBackendError,
    _dlinfo_path,
    _load,
    _resolve_path,
    backend_info,
    get_lib,
)

ILP64_CONFIG = "OpenBLAS 0.3.31 DYNAMIC_ARCH NO_AFFINITY USE64BITINT MAX_THREADS=64"
# NO_USE64BITINT must NOT read as ILP64 -- the word-boundary match in _validate_ilp64 is
# what separates these two strings.
LP64_CONFIG = "OpenBLAS 0.3.31 DYNAMIC_ARCH NO_AFFINITY NO_USE64BITINT MAX_THREADS=64"


class _StubLib:
    """A CDLL stand-in exposing a chosen decoration, config string and thread symbols."""

    def __init__(self, name, *, decoration, config, threads=4):
        prefix, suffix = decoration
        self.name = name
        self._handle = 0xB1614  # only read if dlinfo gets that far
        self._lapack_sym = f"{prefix}dpotrf{suffix}"
        self._config = config.encode()
        self._threads = threads
        self._config_sym = "scipy_openblas_get_config64_"
        self._get_sym = "scipy_openblas_get_num_threads_64_"
        self._set_sym = "scipy_openblas_set_num_threads64_"

    def __getattr__(self, name):
        # Reached only for names that are not instance attributes.
        if name == self._config_sym:

            def cfg():
                return self._config

            return cfg
        if name in (self._get_sym, self._set_sym):

            def threads(*_args):
                return self._threads

            return threads
        if name == self._lapack_sym:
            return object()
        raise AttributeError(name)


def _stub_cdll(table, record=None):
    """Return a ``ctypes.CDLL`` replacement serving *table*, optionally recording calls."""

    def _cdll(name, *args, **kwargs):
        if record is not None:
            record.append((name, args, kwargs))
        if name is None:
            # _dlinfo_path asks the loader for a global handle; refuse so that
            # _resolve_path exercises its /proc fallback instead.
            raise OSError("stubbed: no global handle in tests")
        try:
            return table[name]
        except KeyError:
            raise OSError(f"{name}: cannot open shared object file") from None

    return _cdll


def _ilp64_stub(name="libilp64.so"):
    return _StubLib(name, decoration=("scipy_", "_64_"), config=ILP64_CONFIG)


def _lp64_stub(name="liblp64.so"):
    return _StubLib(name, decoration=("", "_"), config=LP64_CONFIG)


# ---------------------------------------------------------------------------
# D3 -- two-pass loading
# ---------------------------------------------------------------------------


def test_prefers_ilp64_over_earlier_lp64(monkeypatch):
    """An LP64 library earlier in the list must not end the search.

    This is the libmkl_rt-stuck-in-LP64 case: round-1 code returned it and never reached
    the working libopenblas64 behind it.
    """
    lp64, ilp64 = _lp64_stub(), _ilp64_stub()
    monkeypatch.delenv("BIGLA_LIB", raising=False)
    monkeypatch.setattr(_backend, "_candidates", lambda: iter([lp64.name, ilp64.name]))
    monkeypatch.setattr(ctypes, "CDLL", _stub_cdll({lp64.name: lp64, ilp64.name: ilp64}))

    lib, _path, dec, is_ilp64, cfg, _tinfo = _load()

    assert lib is ilp64, "loader stopped at the LP64 candidate"
    assert is_ilp64 is True
    assert dec == ("scipy_", "_64_")
    assert "USE64BITINT" in cfg


def test_lp64_only_falls_back_with_warning(monkeypatch, caplog):
    """With no ILP64 anywhere, LP64 is a valid last resort -- but a loud one."""
    lp64 = _lp64_stub()
    monkeypatch.delenv("BIGLA_LIB", raising=False)
    monkeypatch.setattr(_backend, "_candidates", lambda: iter([lp64.name]))
    monkeypatch.setattr(ctypes, "CDLL", _stub_cdll({lp64.name: lp64}))

    with caplog.at_level(logging.WARNING, logger="bigla._backend"):
        lib, _path, _dec, is_ilp64, _cfg, _tinfo = _load()

    assert lib is lp64
    assert is_ilp64 is False
    assert "46340" in caplog.text
    assert "docs/backends.md" in caplog.text


def test_lp64_backend_info_reports_lp64(monkeypatch):
    """The LP64 fallback must surface as confidence/max_dim, not just a log line."""
    lp64 = _lp64_stub()
    monkeypatch.delenv("BIGLA_LIB", raising=False)
    monkeypatch.setattr(_backend, "_candidates", lambda: iter([lp64.name]))
    monkeypatch.setattr(ctypes, "CDLL", _stub_cdll({lp64.name: lp64}))
    # Reset the singleton so _ensure_loaded re-runs; monkeypatch restores it after.
    monkeypatch.setattr(_backend, "_LIB", None)
    monkeypatch.setattr(_backend, "_INFO", None)
    monkeypatch.setattr(_backend, "_LOAD_ERROR", None)

    info = backend_info()

    assert info.ilp64 is False
    assert info.confidence == "lp64"
    assert info.max_dim == 46340


def test_lp64_max_dim_is_enforced(monkeypatch):
    """n past max_dim raises rather than silently overflowing a 32-bit LAPACK index.

    max_dim is faked small: the real boundary is 46341, and allocating that is ~17 GiB.
    """
    real = backend_info()
    lp64_info = type(real)(
        path="liblp64.so",
        config_string=LP64_CONFIG,
        decoration=("", "_"),
        ilp64=False,
        confidence="lp64",
        max_dim=4,
        threads=real.threads,
        thread_info=real.thread_info,
    )
    monkeypatch.setattr(bigla.linalg, "backend_info", lambda: lp64_info)

    a = np.eye(5)
    with pytest.raises(BiglaBackendError) as exc:
        bigla.linalg.eigh(a)
    assert "max_dim=4" in str(exc.value)
    assert "docs/backends.md" in str(exc.value)


def test_explicit_bigla_lib_lp64_is_honoured(monkeypatch, caplog):
    """BIGLA_LIB is exempt from the ILP64 requirement -- an explicit choice is obeyed.

    The real _candidates() is used deliberately: it must yield BIGLA_LIB and stop, so the
    ILP64 stub below is unreachable. That is what makes the choice explicit rather than a
    preference the second pass can override.
    """
    lp64, unreachable = _lp64_stub("/opt/liblp64.so"), _ilp64_stub()
    monkeypatch.setenv("BIGLA_LIB", lp64.name)
    monkeypatch.setattr(
        ctypes, "CDLL", _stub_cdll({lp64.name: lp64, unreachable.name: unreachable})
    )

    with caplog.at_level(logging.WARNING, logger="bigla._backend"):
        lib, path, _dec, is_ilp64, _cfg, _tinfo = _load()

    assert lib is lp64, "BIGLA_LIB was overridden by a later candidate"
    assert is_ilp64 is False
    assert path == lp64.name, "an absolute BIGLA_LIB is reported verbatim"
    assert "ILP64 could not be confirmed" in caplog.text


# ---------------------------------------------------------------------------
# D4 -- path resolution never invents a path
# ---------------------------------------------------------------------------


def test_resolve_path_never_joins_cwd():
    """A bare soname that is not mapped stays a soname; it never becomes <cwd>/name."""
    resolved = _resolve_path("libdoesnotexist.so")

    assert not resolved.startswith(os.getcwd())
    assert not os.path.isabs(resolved)
    assert resolved == "libdoesnotexist.so"


def test_resolved_path_exists():
    """BackendInfo.path names a real file, or is a bare soname with no directory part.

    A row in docs/backends.md naming a nonexistent path is worse than no row.
    """
    path = backend_info().path

    assert path, "resolved path is empty"
    assert (
        os.path.exists(path) or os.path.dirname(path) == ""
    ), f"resolved path {path!r} is neither an existing file nor a bare soname"


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="dlinfo/RTLD_DI_LINKMAP is glibc-specific"
)
def test_resolve_path_prefers_dlinfo_over_prefix_match():
    """dlinfo answers for THIS handle; the /proc scan only matches by basename prefix.

    A hint of ``libopenblas64.so`` prefix-matches a mapped ``libopenblas64_.so.0`` -- a
    genuinely different library, which would then be reported as the resolved path.
    """
    lib = get_lib()
    exact = _dlinfo_path(lib)
    if exact is None:
        pytest.skip("dlinfo unavailable for the loaded backend")

    assert _resolve_path("libopenblas64.so", lib) == exact


# ---------------------------------------------------------------------------
# 5.4 -- no global symbol leak
# ---------------------------------------------------------------------------


def test_cdll_loaded_without_rtld_global(monkeypatch):
    """The load must not request RTLD_GLOBAL, by flag or by mode= keyword."""
    ilp64 = _ilp64_stub()
    calls: list = []
    monkeypatch.delenv("BIGLA_LIB", raising=False)
    monkeypatch.setattr(_backend, "_candidates", lambda: iter([ilp64.name]))
    monkeypatch.setattr(ctypes, "CDLL", _stub_cdll({ilp64.name: ilp64}, record=calls))

    _load()

    loads = [c for c in calls if c[0] is not None]
    assert loads, "no library load was attempted"
    for name, args, kwargs in loads:
        assert "mode" not in kwargs, f"{name} loaded with mode={kwargs['mode']}"
        assert ctypes.RTLD_GLOBAL not in args, f"{name} loaded with RTLD_GLOBAL"


# Run in a subprocess: the global namespace is process-wide and cannot be un-polluted, so
# the "before" measurement has to happen in a process where nothing has loaded a BLAS yet.
_LEAK_PROBE = r"""
import ctypes, json


def visible(sym):
    try:
        getattr(ctypes.CDLL(None), sym)
        return True
    except (AttributeError, OSError):
        return False


# Import, up front, everything bigla's own discovery imports. scipy_openblas64 publishes
# its symbols globally when IMPORTED -- on its own account, nothing to do with how bigla
# dlopens it -- and _candidates() imports it to call get_lib_dir(). Doing it here means the
# "before" snapshot already contains anything a third party published, so what the test
# attributes to bigla is only what bigla's own ctypes.CDLL added.
import numpy  # noqa: F401

try:
    import scipy_openblas64  # noqa: F401
except ImportError:
    pass

import bigla  # noqa: F401  -- importing does NOT load the backend; _ensure_loaded is lazy
from bigla._backend import _DECORATIONS, backend_info

before = {f"{p}dpotrf{s}": visible(f"{p}dpotrf{s}") for p, s in _DECORATIONS}
info = backend_info()  # the load happens here
sym = f"{info.decoration[0]}dpotrf{info.decoration[1]}"
print(json.dumps({"sym": sym, "before": before[sym], "after": visible(sym), "path": info.path}))
"""

# The control experiment: repeat what bigla does to the library, with an EXPLICIT RTLD_LOCAL
# and no bigla in the process. A symbol that goes global here went global despite a local
# handle, so the library published it and no caller could have prevented it.
#
# It measures in stages because merely opening the library is not a fair control. bigla also
# resolves a symbol (_probe_decoration) and calls into the library (_validate_ilp64), and a
# DISPATCHER such as MKL's libmkl_rt loads its implementation libraries lazily -- on first
# use, not at dlopen. A control that only opened the file therefore found no leak and blamed
# bigla for one, which is exactly the wrong answer given
# test_cdll_loaded_without_rtld_global proves no call site passes RTLD_GLOBAL. Reporting the
# stage at which visibility flips also says WHAT triggered it, which is the part worth
# recording in docs/backends.md.
_LOCAL_LOAD_PROBE = r"""
import ctypes, json, os, sys


def visible(sym):
    try:
        getattr(ctypes.CDLL(None), sym)
        return True
    except (AttributeError, OSError):
        return False


path, sym = sys.argv[1], sys.argv[2]
stages = {"before": visible(sym)}

lib = ctypes.CDLL(path, mode=os.RTLD_LOCAL)
stages["after_open"] = visible(sym)

# dlsym, as _probe_decoration does.
try:
    getattr(lib, sym)
except AttributeError:
    pass
stages["after_dlsym"] = visible(sym)

# Call into it, as _validate_ilp64 does -- the step that makes a lazy dispatcher dispatch.
for name, call in (
    ("MKL_Set_Interface_Layer", lambda fn: fn(ctypes.c_int(1))),
    ("openblas_get_config64_", lambda fn: fn()),
    ("openblas_get_config", lambda fn: fn()),
):
    try:
        fn = getattr(lib, name)
    except AttributeError:
        continue
    fn.restype = ctypes.c_char_p if "config" in name else ctypes.c_int
    try:
        call(fn)
    except Exception:
        pass
    break
stages["after_call"] = visible(sym)

print(json.dumps(stages))
"""


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="global symbol namespace is dlopen-specific"
)
def test_no_global_symbol_leak():
    """bigla's own dlopen must not publish its LAPACK symbols to the global namespace.

    Attribution is the whole difficulty. Checking only the end state blames bigla for a
    symbol somebody else published: with scipy-openblas64 installed, importing that package
    puts ``scipy_dpotrf_64_`` in the global namespace before bigla loads anything, and the
    naive form of this test fails on a configuration where bigla is behaving perfectly.
    So measure before and after bigla's load, in a fresh process, and skip when the symbol
    was already there -- there, nothing can be attributed either way.

    ``test_cdll_loaded_without_rtld_global`` is the deterministic half of 5.4; this is the
    live confirmation, and it is allowed to abstain.
    """
    proc = subprocess.run([sys.executable, "-c", _LEAK_PROBE], capture_output=True, text=True)
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
    data = json.loads(proc.stdout.strip().splitlines()[-1])

    if data["before"]:
        pytest.skip(
            f"{data['sym']} was already global before bigla loaded "
            "(scipy-openblas64 publishes it on import); not attributable"
        )
    if not data["after"]:
        return  # nothing leaked; the common case

    # Something is global that was not before. Before blaming bigla, load the same library
    # with an explicit RTLD_LOCAL and nothing else in the process: if the symbol still goes
    # global, the library published it itself and no caller could have stopped it.
    control = subprocess.run(
        [sys.executable, "-c", _LOCAL_LOAD_PROBE, data["path"], data["sym"]],
        capture_output=True,
        text=True,
    )
    assert control.returncode == 0, f"control probe failed:\n{control.stdout}\n{control.stderr}"
    ctl = json.loads(control.stdout.strip().splitlines()[-1])
    if not ctl["before"]:
        leaked_at = next(
            (stage for stage in ("after_open", "after_dlsym", "after_call") if ctl[stage]), None
        )
        if leaked_at is not None:
            pytest.skip(
                f"{data['path']} publishes {data['sym']} globally on its own account: an "
                f"explicit RTLD_LOCAL load leaks it at stage {leaked_at!r}, with no bigla in "
                f"the process. Not something a caller can prevent -- see "
                f"test_cdll_loaded_without_rtld_global for bigla's own guarantee."
            )

    raise AssertionError(
        f"{data['sym']} became visible through the global handle after bigla loaded "
        f"{data['path']}, but an explicit RTLD_LOCAL load of the same library does not leak "
        f"it at any stage ({ctl}) -- so bigla's own dlopen published it"
    )
