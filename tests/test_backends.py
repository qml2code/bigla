"""Backend discovery and ILP64 validation tests."""

from __future__ import annotations

import ctypes

import pytest

import bigla
from bigla._backend import (
    BiglaBackendError,
    ThreadInfo,
    _candidates,
    _probe_decoration,
    _probe_sym,
    _validate_ilp64,
    backend_info,
)

# ---------------------------------------------------------------------------
# BackendInfo fields
# ---------------------------------------------------------------------------


def test_backend_info_fields():
    info = backend_info()
    assert isinstance(info.path, str)
    assert isinstance(info.config_string, str)
    assert isinstance(info.decoration, tuple) and len(info.decoration) == 2
    assert isinstance(info.ilp64, bool)
    assert info.confidence in ("config-string", "mkl-interface", "suffix-only", "lp64")
    assert info.max_dim >= 46340
    assert info.threads >= 0


def test_backend_info_ilp64_on_numpy_wheel():
    """On a stock numpy wheel the bundled library should be ILP64."""
    info = backend_info()
    if "openblas64" in info.path.lower():
        assert info.ilp64, f"Expected ILP64 for {info.path}, got {info.config_string!r}"


# ---------------------------------------------------------------------------
# Decoration probing
# ---------------------------------------------------------------------------


class _FakeCDLL:
    """Minimal CDLL stand-in that exposes only a specified set of symbols."""

    def __init__(self, symbols: set):
        self._syms = symbols

    def __getattr__(self, name: str):
        if name in self._syms:
            return object()
        raise AttributeError(name)


def test_probe_decoration_scipy_prefix():
    assert _probe_decoration(_FakeCDLL({"scipy_dpotrf_64_"})) == ("scipy_", "_64_")


def test_probe_decoration_bare_64():
    assert _probe_decoration(_FakeCDLL({"dpotrf_64_"})) == ("", "_64_")


def test_probe_decoration_64_no_underscore():
    assert _probe_decoration(_FakeCDLL({"dpotrf64_"})) == ("", "64_")


def test_probe_decoration_plain():
    assert _probe_decoration(_FakeCDLL({"dpotrf_"})) == ("", "_")


def test_probe_decoration_none():
    assert _probe_decoration(_FakeCDLL({"something_else"})) is None


# ---------------------------------------------------------------------------
# ILP64 validation
# ---------------------------------------------------------------------------


class _FakeCDLLConfig:
    """CDLL whose get_config candidates return a fixed config string."""

    def __init__(self, cfg_str: str, respond_to: str = "scipy_openblas_get_config64_"):
        self._cfg = cfg_str.encode()
        self._name = respond_to  # only THIS exact symbol responds

    def __getattr__(self, name: str):
        if name == self._name:

            def fn():
                return self._cfg

            fn.restype = ctypes.c_char_p
            return fn
        raise AttributeError(name)


def test_validate_ilp64_use64bit():
    dll = _FakeCDLLConfig("OpenBLAS 0.3.31 USE64BITINT DYNAMIC_ARCH")
    ok, cfg = _validate_ilp64(dll, "scipy_", "_64_")
    assert ok
    assert "USE64BITINT" in cfg


def test_validate_ilp64_lp64_config():
    """NO_USE64BITINT must NOT match as ILP64 (word-boundary check)."""
    dll = _FakeCDLLConfig("OpenBLAS 0.3.31 NO_USE64BITINT")
    ok, cfg = _validate_ilp64(dll, "", "_64_")
    assert not ok


def test_validate_ilp64_suffix_only():
    """When no get_config symbol exists, fall back to suffix evidence."""

    class NoCfg:
        def __getattr__(self, n):
            raise AttributeError(n)

    ok, cfg = _validate_ilp64(NoCfg(), "", "_64_")
    assert ok
    assert "suffix" in cfg


def test_validate_ilp64_no_evidence():
    class Nothing:
        def __getattr__(self, n):
            raise AttributeError(n)

    ok, cfg = _validate_ilp64(Nothing(), "", "_")
    assert not ok


# ---------------------------------------------------------------------------
# thread_control_info
# ---------------------------------------------------------------------------


def test_thread_control_info_type():
    ti = bigla.thread_control_info()
    assert isinstance(ti, ThreadInfo)
    assert isinstance(ti.num_threads, int)


def test_thread_control_info_symbols_match_live():
    """get_sym and set_sym must be symbols that actually exist in the library."""
    from bigla._backend import get_lib

    lib = get_lib()
    ti = bigla.thread_control_info()
    if ti.get_sym is not None:
        getattr(lib, ti.get_sym)  # must not AttributeError
    if ti.set_sym is not None:
        getattr(lib, ti.set_sym)


def test_thread_control_info_num_threads_current():
    """num_threads in ThreadInfo should equal get_num_threads()."""
    ti = bigla.thread_control_info()
    assert ti.num_threads == bigla.get_num_threads()


# ---------------------------------------------------------------------------
# Thread control: get / set
# ---------------------------------------------------------------------------


def test_get_num_threads():
    n = bigla.get_num_threads()
    assert isinstance(n, int) and n >= 0


def _require_resizable_pool():
    """Skip unless this backend's thread pool can actually be resized.

    A SERIAL OpenBLAS -- Fedora's `openblas-serial64_`, or any -DUSE_THREAD=0 build -- exports
    the setter and accepts the call, but has no pool: it stays at 1. That is a legitimate
    environment, not a defect, so the round-trip tests below skip on it. The skip is driven by
    ``set_num_threads``'s RETURN value, which reports what the library actually has, so a
    backend that silently ignored the setter could not use this path to look healthy: the
    return value would have to lie first, and test_set_num_threads_reports_clamping is
    what stops that.
    """
    orig = bigla.get_num_threads()
    if orig == 0 or bigla.thread_control_info().set_sym is None:
        pytest.skip("thread control not available on this backend")
    if bigla.set_num_threads(2) != 2:
        bigla.set_num_threads(orig)
        pytest.skip(f"single-threaded backend: set_num_threads(2) stayed at {orig}")
    bigla.set_num_threads(orig)
    return orig


def test_set_num_threads_explicit_roundtrip():
    orig = _require_resizable_pool()
    bigla.set_num_threads(2)
    assert bigla.get_num_threads() == 2
    bigla.set_num_threads(orig)
    assert bigla.get_num_threads() == orig


def test_set_num_threads_requires_argument():
    """bigla does not guess a thread count.

    The removed default was `min(cpu_count()//2, 4)` -- on a 128-core node factoring a 50000^2
    matrix that is a ~30x slowdown, and the halving assumes SMT is on, which is false on most
    HPC partitions. There is no numpy or scipy analogue to copy: neither exposes a thread-count
    API at all.
    """
    with pytest.raises(TypeError):
        bigla.set_num_threads()
    from bigla import _backend

    assert not hasattr(_backend, "_default_num_threads")


def test_set_num_threads_round_trip():
    orig = _require_resizable_pool()
    try:
        assert bigla.set_num_threads(2) == 2
        assert bigla.get_num_threads() == 2
    finally:
        bigla.set_num_threads(orig)


def test_num_threads_context_manager_restores():
    """Scoped control, mirroring scipy.fft.set_workers -- including on the exception path."""
    orig = _require_resizable_pool()
    with bigla.num_threads(2):
        assert bigla.get_num_threads() == 2
    assert bigla.get_num_threads() == orig

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        with bigla.num_threads(3):
            raise Boom()
    assert bigla.get_num_threads() == orig


def test_set_num_threads_reports_clamping():
    """A backend may give you fewer threads than you asked for, and must say so.

    This asserted `set_num_threads(3) == 3` while the function returned its own argument, so
    it could not have failed. Both matrix rows that got far enough then contradicted it: MKL
    returned 2 for a request of 3 (it clamps to the machine's cores) and Fedora's serial
    build returned 1. Neither is a defect -- but echoing the argument would report 3 in both
    cases, and a caller sizing work by that number over-subscribes by exactly the factor it
    was trying to control.
    """
    orig = bigla.get_num_threads()
    if orig == 0 or bigla.thread_control_info().set_sym is None:
        pytest.skip("thread control not available on this backend")
    try:
        ret = bigla.set_num_threads(3)
        assert ret == bigla.get_num_threads(), "return value disagrees with the library"
        assert ret <= 3, "backend reported more threads than were requested"
    finally:
        bigla.set_num_threads(orig)


# ---------------------------------------------------------------------------
# _probe_sym
# ---------------------------------------------------------------------------


def test_probe_sym_finds_first():
    lib = _FakeCDLL({"b", "c"})
    assert _probe_sym(lib, ["a", "b", "c"]) == "b"


def test_probe_sym_none():
    lib = _FakeCDLL(set())
    assert _probe_sym(lib, ["a", "b"]) is None


# ---------------------------------------------------------------------------
# BIGLA_LIB env var
# ---------------------------------------------------------------------------


def test_bigla_lib_nonexistent(monkeypatch, tmp_path):
    from bigla import _backend as bk

    monkeypatch.setenv("BIGLA_LIB", str(tmp_path / "nonexistent.so"))
    monkeypatch.setattr(bk, "_LIB", None)
    monkeypatch.setattr(bk, "_INFO", None)
    monkeypatch.setattr(bk, "_LOAD_ERROR", None)
    with pytest.raises(BiglaBackendError):
        bk.backend_info()


# ---------------------------------------------------------------------------
# Candidates generator stops after $BIGLA_LIB
# ---------------------------------------------------------------------------


def test_candidates_bigla_lib_first_and_only(monkeypatch):
    monkeypatch.setenv("BIGLA_LIB", "/custom/path/lib.so")
    cands = list(_candidates())
    assert cands == ["/custom/path/lib.so"]
