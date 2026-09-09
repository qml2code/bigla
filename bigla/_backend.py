"""Backend discovery, symbol decoration, ILP64 validation, and thread control.

Search order (§5.1):
    1. $BIGLA_LIB env var — always wins, fail loudly if it doesn't load
    2. scipy_openblas64 package (scipy_openblas64.get_lib_dir())
    3. numpy's bundled library (numpy.libs / numpy/.dylibs)
    4. System OpenBLAS ILP64 (preferred over MKL, SPEC 5.4)
    5. System: libopenblas64_.so.0, libopenblas64.so.0, libflexiblas64.so

Thread control note
-------------------
The LAPACK symbol decoration (e.g. ``scipy_`` prefix + ``_64_`` suffix) applies
only to LAPACK/BLAS computational routines.  The OpenBLAS management API
(openblas_get_num_threads, openblas_set_num_threads, openblas_get_config) uses
a *different* naming scheme: ``scipy_openblas_NAME64_`` (no leading underscore
before 64).  We therefore probe the thread-control symbols independently of the
LAPACK decoration — the two probe lists must stay separate.
"""

from __future__ import annotations

import contextlib
import ctypes
import glob
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate library paths
# ---------------------------------------------------------------------------


def _site_packages() -> str:
    return os.path.dirname(os.path.dirname(np.__file__))


def _candidates():
    # 1. Explicit override — yield only this path; generator stops after.
    if "BIGLA_LIB" in os.environ:
        yield os.environ["BIGLA_LIB"]
        return

    # 2. scipy_openblas64 stand-alone package
    try:
        import scipy_openblas64  # type: ignore[import]

        lib_dir = scipy_openblas64.get_lib_dir()
        yield from sorted(glob.glob(os.path.join(lib_dir, "*openblas64*")))
    except ImportError:
        pass

    site = _site_packages()

    # 3. numpy's bundled library
    for pat in (
        "scipy_openblas64/lib/libscipy_openblas64_*.so*",
        "numpy.libs/*openblas64*.so*",  # Linux numpy wheel
        "numpy/.dylibs/*openblas64*",  # macOS numpy wheel
    ):
        yield from sorted(glob.glob(os.path.join(site, pat)))

    # 4. System OpenBLAS ILP64 -- ahead of MKL per SPEC 5.4: prefer a dedicated OpenBLAS64
    # when both are present. Ordering matters less for SELECTION now that loading is
    # two-pass, but probing has a side effect that ordering does control: _validate_ilp64
    # calls MKL_Set_Interface_Layer(1), mutating global MKL state in this process even when
    # bigla goes on to use OpenBLAS -- and if numpy is MKL-linked and running LP64, the probe
    # has just altered a library numpy is using. Reaching MKL last minimises that.
    for name in (
        "libopenblas64_.so.0",
        "libopenblas64_.so",
        "libopenblas64.so.0",
        "libopenblas64.so",
        "libflexiblas64.so",
    ):
        yield name

    # 5. MKL, last: see the side-effect note above.
    for name in ("libmkl_rt.so", "libmkl_rt.so.2", "libmkl_rt.so.1"):
        yield name


# ---------------------------------------------------------------------------
# LAPACK symbol decoration  (§5.2)
# ---------------------------------------------------------------------------

_DECORATIONS = [
    ("scipy_", "_64_"),  # numpy/scipy wheels (verified)
    ("", "_64_"),  # Debian/Ubuntu libopenblas64-dev
    ("", "64_"),  # some OpenBLAS builds
    ("", "_"),  # MKL ILP64, reference LAPACK -i8
    ("", "$NEWLAPACK$ILP64"),  # Apple Accelerate (to verify)
]


def _probe_decoration(lib: ctypes.CDLL) -> Optional[tuple[str, str]]:
    for prefix, suffix in _DECORATIONS:
        try:
            getattr(lib, f"{prefix}dpotrf{suffix}")
            return (prefix, suffix)
        except AttributeError:
            continue
    return None


# ---------------------------------------------------------------------------
# OpenBLAS management API symbol probing
#
# These names are NOT derived from the LAPACK decoration.  In the numpy wheel
# the relevant symbols are:
#   scipy_openblas_get_config64_          (config string)
#   scipy_openblas_get_num_threads64_     (get thread count)    ← C binding
#   scipy_openblas_set_num_threads64_     (set thread count)    ← C binding
#   scipy_openblas_get_num_threads_64_    Fortran binding -- see the rule below
#   scipy_openblas_set_num_threads_64_    Fortran binding -- SEGFAULTS if called by value
#
# We probe each function by trying a fixed priority list, verifying it is
# callable, and caching the winning name.
# ---------------------------------------------------------------------------

# THE RULE FOR EVERY LIST BELOW: only C entry points, never Fortran ones.
#
# A trailing `_` on the base name (before any SYMBOLSUFFIX) marks the Fortran binding, whose
# argument is a POINTER; the C binding takes the value. MKL spells the same distinction in
# case: `MKL_Set_Num_Threads(int)` is C, plain `mkl_set_num_threads(int*)` is Fortran. Calling
# a Fortran one through ctypes with a by-value argument makes the callee dereference that
# value as an address -- SIGSEGV, not an exception.
#
# This is not hypothetical. The note that used to sit here recorded exactly one instance of
# it -- `scipy_openblas_set_num_threads_64_` "crashes on the numpy wheel, the symbol exists
# but has a different ABI" (that is the Fortran name; `...64_` without the underscore is the
# C one) -- and worked around it by preferring the C spelling for THAT library, leaving the
# Fortran spellings in place for others to fall through to. Environment-matrix run
# 34347810149 then found both remaining ones: Debian's bare-decoration libopenblas64 reached
# `openblas_set_num_threads_` and segfaulted, and MKL segfaulted at import in
# `_validate_ilp64` on `mkl_set_interface_layer`.
#
# So: no name here may end in `_` before its suffix, and MKL names use its C capitalisation.

# get_config candidates, in priority order
_GET_CONFIG_CANDIDATES = [
    "scipy_openblas_get_config64_",
    "openblas_get_config64_",
    "openblas_get_config",
]

# get_num_threads candidates. Getters take no arguments, so the Fortran spellings are not
# dangerous here -- but they are dropped anyway, so that one rule covers the whole module and
# nobody has to re-derive which lists are safe.
_GET_THREADS_CANDIDATES = [
    "scipy_openblas_get_num_threads64_",
    "openblas_get_num_threads64_",
    "openblas_get_num_threads",
    "MKL_Get_Max_Threads",
]

# set_num_threads candidates -- the list that segfaulted.
_SET_THREADS_CANDIDATES = [
    "scipy_openblas_set_num_threads64_",
    "scipy_goto_set_num_threads64_",
    "openblas_set_num_threads64_",
    "openblas_set_num_threads",
    "goto_set_num_threads",
    "MKL_Set_Num_Threads",
]


def _probe_sym(lib: ctypes.CDLL, candidates: list[str]) -> Optional[str]:
    """Return the first name from *candidates* that exists in *lib*, or None."""
    for name in candidates:
        try:
            getattr(lib, name)
            return name
        except AttributeError:
            continue
    return None


# ---------------------------------------------------------------------------
# ILP64 validation  (§5.3)
# ---------------------------------------------------------------------------

# Matches USE64BITINT as a whole token (not as part of NO_USE64BITINT).
_USE64_RE = re.compile(r"(?<!\w)USE64BITINT(?!\w)")


def _validate_ilp64(lib: ctypes.CDLL, prefix: str, suffix: str) -> tuple[bool, str]:
    """Return (is_ilp64, evidence_string).  Fails open on error.

    Uses word-boundary matching so that "NO_USE64BITINT" does NOT match.
    """
    # OpenBLAS: call get_config and look for USE64BITINT as a standalone token
    cfg_sym = _probe_sym(lib, _GET_CONFIG_CANDIDATES)
    if cfg_sym is not None:
        fn = getattr(lib, cfg_sym)
        fn.restype = ctypes.c_char_p
        raw: bytes = fn()
        cfg_str = raw.decode(errors="replace") if raw else ""
        if _USE64_RE.search(cfg_str):
            return True, cfg_str
        # Got a config string without USE64BITINT → confirmed LP64
        return False, cfg_str

    # MKL: set interface layer before first use.
    #
    # MKL_Set_Interface_Layer, not mkl_set_interface_layer: the capitalised name is the C
    # entry point taking an int by value, the lowercase one is the Fortran binding taking
    # int*. Calling the latter through ctypes with a value made MKL dereference 1 as an
    # address, so the conda-forge row segfaulted HERE, during import, and never reached a
    # single test (matrix run 34347810149).
    try:
        fn = getattr(lib, "MKL_Set_Interface_Layer")
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_int]
        MKL_INTERFACE_ILP64 = 1
        ret = fn(ctypes.c_int(MKL_INTERFACE_ILP64))
        if ret == MKL_INTERFACE_ILP64:
            return True, "MKL ILP64 interface confirmed"
        return False, f"MKL interface layer returned {ret}"
    except AttributeError:
        pass

    # Suffix-only evidence (weak but acceptable per spec)
    if suffix in ("_64_", "64_", "$NEWLAPACK$ILP64"):
        return True, f"ILP64 inferred from symbol suffix {suffix!r}"

    return False, "could not establish ILP64 status"


def _classify_confidence(cfg: str, ilp64: bool) -> str:
    if not ilp64:
        return "lp64"
    if _USE64_RE.search(cfg):
        return "config-string"
    if "MKL ILP64 interface confirmed" in cfg:
        return "mkl-interface"
    return "suffix-only"


# ---------------------------------------------------------------------------
# ThreadInfo — what symbol is being used and what the count is
# ---------------------------------------------------------------------------


@dataclass
class ThreadInfo:
    """Which thread-control symbols are active and the current thread count.

    Attributes
    ----------
    get_sym : str | None
        Name of the symbol used to query the thread count, or None if no
        queryable symbol was found.
    set_sym : str | None
        Name of the symbol used to set the thread count, or None if no
        settable symbol was found.
    num_threads : int
        Current thread count; 0 if not queryable.
    """

    get_sym: Optional[str]
    set_sym: Optional[str]
    num_threads: int


# ---------------------------------------------------------------------------
# BackendInfo dataclass (public API)
# ---------------------------------------------------------------------------


@dataclass
class BackendInfo:
    """Full description of the loaded LAPACK backend.

    Attributes
    ----------
    path : str
        Resolved path of the loaded shared library.
    config_string : str
        Output of openblas_get_config (or equivalent evidence string).
    decoration : (prefix, suffix)
        LAPACK symbol name decoration, e.g. (``'scipy_'``, ``'_64_'``).
    ilp64 : bool
        True when 64-bit integer interface is confirmed.
    confidence : str
        How ILP64 was established: ``"config-string"``, ``"mkl-interface"``,
        ``"suffix-only"``, or ``"lp64"``.
    max_dim : int
        Largest dimension safely handled: 46 340 for LP64, 2**62 for ILP64.
    threads : int
        Current BLAS thread count at load time (0 if not queryable).
    thread_info : ThreadInfo
        Which get/set symbols are active and the current count.
    """

    path: str
    config_string: str
    decoration: tuple[str, str]
    ilp64: bool
    confidence: str
    max_dim: int
    threads: int
    thread_info: ThreadInfo = field(repr=False)

    @property
    def decoration_str(self) -> str:
        p, s = self.decoration
        return f"{p}NAME{s}"


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class BiglaBackendError(RuntimeError):
    """Raised when no suitable ILP64 library is found, or n > max_dim on LP64."""


# ---------------------------------------------------------------------------
# Library loading helpers
# ---------------------------------------------------------------------------


class _LinkMap(ctypes.Structure):
    _fields_ = [
        ("l_addr", ctypes.c_void_p),
        ("l_name", ctypes.c_char_p),
        ("l_ld", ctypes.c_void_p),
        ("l_next", ctypes.c_void_p),
        ("l_prev", ctypes.c_void_p),
    ]


_RTLD_DI_LINKMAP = 2


def _dlinfo_path(lib: ctypes.CDLL) -> Optional[str]:
    """Exact filesystem path of a loaded handle via dlinfo, or None where unavailable."""
    try:
        libdl = ctypes.CDLL(None)
        dlinfo = libdl.dlinfo
        dlinfo.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        dlinfo.restype = ctypes.c_int
        lmap = ctypes.POINTER(_LinkMap)()
        if dlinfo(lib._handle, _RTLD_DI_LINKMAP, ctypes.byref(lmap)) != 0:
            return None
        name = lmap.contents.l_name
        if not name:
            return None
        path = name.decode(errors="replace")
        if not path.startswith("/"):
            return None
        # The loader records the name as resolved, which can carry ".." segments
        # (".../numpy/_core/../../numpy.libs/lib...so"). Correct and existent, but two machines
        # render the same library differently, which hurts dedup and diffing of the
        # docs/backends.md rows. realpath is safe HERE precisely because the path exists --
        # unlike the D4 bug, where it was applied to a bare soname and invented a cwd path.
        return os.path.realpath(path) if os.path.exists(path) else path
    except Exception:
        return None


def _resolve_path(hint: str, lib: Optional[ctypes.CDLL] = None) -> str:
    """The path the loader actually mapped, or the bare soname -- never a guess.

    Prefer dlinfo(RTLD_DI_LINKMAP), which returns the path of THIS handle exactly. The
    /proc scan that follows matches by basename prefix, and a hint of ``libopenblas64.so``
    would also match a mapped ``libopenblas64_.so.0`` -- a genuinely different library, which
    would then be reported as the resolved path and written into docs/backends.md.

    ``os.path.realpath("libmkl_rt.so")`` returns ``<cwd>/libmkl_rt.so``, a file that does not
    exist. That string lands in BackendInfo.path, in every error message, and in the
    docs/backends.md rows diagnose generates -- and a row naming a nonexistent path is worse
    than no row. For a bare soname, ask the loader via /proc/self/maps; where /proc is
    unavailable (macOS, Windows), return the soname itself, which is clearly not a path.
    """
    if os.path.isabs(hint):
        return hint
    if lib is not None:
        exact = _dlinfo_path(lib)
        if exact:
            return exact
    stem = os.path.basename(hint).split(".so")[0]
    try:
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                mapped = line.rstrip("\n").rsplit(" ", 1)[-1].strip()
                if mapped.startswith("/") and os.path.basename(mapped).startswith(stem):
                    return mapped
    except OSError:
        pass
    return os.path.basename(hint)


def _query_threads(lib: ctypes.CDLL, get_sym: Optional[str]) -> int:
    if get_sym is None:
        return 0
    try:
        fn = getattr(lib, get_sym)
        fn.restype = ctypes.c_int
        fn.argtypes = []
        return int(fn())
    except Exception:
        return 0


def _load() -> tuple[ctypes.CDLL, str, tuple[str, str], bool, str, ThreadInfo]:
    explicit = "BIGLA_LIB" in os.environ
    tried: list[str] = []
    lp64_fallbacks: list[tuple] = []

    for cand in _candidates():
        try:
            # RTLD_LOCAL (the ctypes default): we resolve symbols through our own handle, so
            # nothing needs to be globally visible. RTLD_GLOBAL would publish undecorated names
            # like dpotrf_ (MKL, reference LAPACK built -i8) to everything loaded afterwards,
            # numpy included if it loads later.
            lib = ctypes.CDLL(cand)
        except OSError as exc:
            tried.append(f"  {cand}: {exc}")
            continue

        dec = _probe_decoration(lib)
        if dec is None:
            tried.append(f"  {cand}: loaded but no recognisable LAPACK symbols found")
            continue

        prefix, suffix = dec
        ilp64, cfg = _validate_ilp64(lib, prefix, suffix)

        if explicit and not ilp64:
            log.warning(
                "BIGLA_LIB=%s loaded but ILP64 could not be confirmed: %s",
                cand,
                cfg,
            )

        get_sym = _probe_sym(lib, _GET_THREADS_CANDIDATES)
        set_sym = _probe_sym(lib, _SET_THREADS_CANDIDATES)
        n_threads = _query_threads(lib, get_sym)
        tinfo = ThreadInfo(get_sym=get_sym, set_sym=set_sym, num_threads=n_threads)

        resolved = _resolve_path(cand, lib)
        if ilp64 or explicit:
            # First pass returns only a VERIFIED ILP64 backend. BIGLA_LIB is exempt: it yields a
            # single candidate, and an explicit choice is honoured (with the warning above).
            return lib, resolved, dec, ilp64, cfg, tinfo
        # An LP64 library answering to LAPACK symbols is a valid last resort (SPEC 5.5), but
        # returning it here ends the search -- so on a node where libmkl_rt is present and stuck
        # in LP64, a working libopenblas64 further down the list would never be examined.
        lp64_fallbacks.append((lib, resolved, dec, ilp64, cfg, tinfo))
        tried.append(f"  {cand}: loaded, but LP64 (kept as fallback)")

    if lp64_fallbacks:
        lib, resolved, dec, ilp64, cfg, tinfo = lp64_fallbacks[0]
        log.warning(
            "bigla: no ILP64 backend found; falling back to %s (max_dim=46340). Matrices "
            "past 46341 will raise. See docs/backends.md.",
            resolved,
        )
        return lib, resolved, dec, ilp64, cfg, tinfo

    if explicit:
        raise BiglaBackendError(
            f"BIGLA_LIB={os.environ['BIGLA_LIB']!r} could not be loaded or has no "
            f"LAPACK symbols.\n" + "\n".join(tried)
        )

    raise BiglaBackendError(
        "No ILP64 BLAS/LAPACK found.  Tried:\n" + "\n".join(tried) + "\n\n"
        # Names the dependency, not the extra: bigla is already installed by the time this
        # raises, so `pip install bigla[openblas]` would resolve bigla itself all over again.
        "Install one with:  pip install scipy-openblas64\n"
        "or set:            BIGLA_LIB=/path/to/libopenblas64.so\n"
        "See docs/backends.md for platform-specific instructions."
    )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_LIB: Optional[ctypes.CDLL] = None
_INFO: Optional[BackendInfo] = None
_LOAD_ERROR: Optional[Exception] = None


def _ensure_loaded() -> None:
    global _LIB, _INFO, _LOAD_ERROR
    if _INFO is not None:
        return
    if _LOAD_ERROR is not None:
        raise _LOAD_ERROR
    try:
        lib, path, dec, ilp64, cfg, tinfo = _load()
    except BiglaBackendError as exc:
        _LOAD_ERROR = exc
        raise

    max_dim = (2**62) if ilp64 else 46340
    confidence = _classify_confidence(cfg, ilp64)

    _LIB = lib
    _INFO = BackendInfo(
        path=path,
        config_string=cfg,
        decoration=dec,
        ilp64=ilp64,
        confidence=confidence,
        max_dim=max_dim,
        threads=tinfo.num_threads,
        thread_info=tinfo,
    )
    log.debug(
        "bigla backend: %s  decoration=%s  ilp64=%s  confidence=%s  "
        "threads=%d  get_sym=%s  set_sym=%s",
        path,
        _INFO.decoration_str,
        ilp64,
        confidence,
        tinfo.num_threads,
        tinfo.get_sym,
        tinfo.set_sym,
    )


def backend_info() -> BackendInfo:
    """Return a :class:`BackendInfo` dataclass describing the loaded library."""
    _ensure_loaded()
    return _INFO  # type: ignore[return-value]


def get_lib() -> ctypes.CDLL:
    """Return the raw ctypes CDLL handle (internal use by _core.py)."""
    _ensure_loaded()
    return _LIB  # type: ignore[return-value]


def get_decoration() -> tuple[str, str]:
    """Return the (prefix, suffix) LAPACK symbol decoration."""
    _ensure_loaded()
    return _INFO.decoration  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Thread control  (§6.8)
# ---------------------------------------------------------------------------


def is_ilp64() -> bool:
    """True when the resolved backend is verified 64-bit-integer LAPACK.

    Short alias for ``backend_info().ilp64``, which is the question callers actually ask:
    "can I hand this library a matrix past the 46341 wall?" Importing bigla successfully does
    NOT answer it -- discovery falls back to an LP64 library with ``max_dim=46340`` rather than
    failing at import (SPEC §5.5), so a caller that branches on the import alone will take this
    path and then raise on the first large matrix. Branch on this instead.

    For a specific size prefer ``n <= backend_info().max_dim``, which is the same test with the
    LP64 delegation window included.
    """
    return backend_info().ilp64


def thread_control_info() -> ThreadInfo:
    """Return a :class:`ThreadInfo` describing the active thread-control symbols.

    Example::

        >>> import bigla
        >>> bigla.thread_control_info()
        ThreadInfo(get_sym='scipy_openblas_get_num_threads64_',
                   set_sym='scipy_openblas_set_num_threads64_',
                   num_threads=12)

    The symbol names tell you exactly which entry points bigla is calling,
    which matters on HPC nodes where numpy and bigla may have loaded
    different .so files with separate thread pools.
    """
    _ensure_loaded()
    tinfo = _INFO.thread_info  # type: ignore[union-attr]
    # Refresh thread count in case it changed since load
    current = _query_threads(_LIB, tinfo.get_sym)  # type: ignore[arg-type]
    return ThreadInfo(
        get_sym=tinfo.get_sym,
        set_sym=tinfo.set_sym,
        num_threads=current,
    )


def get_num_threads() -> int:
    """Return the current BLAS thread count for the backend library.

    Returns 0 if the thread count cannot be queried (no get symbol found).
    """
    _ensure_loaded()
    tinfo = _INFO.thread_info  # type: ignore[union-attr]
    return _query_threads(_LIB, tinfo.get_sym)  # type: ignore[arg-type]


def set_num_threads(n: int) -> int:
    """Set the BLAS thread count for the backend library.

    Parameters
    ----------
    n : int
        Number of threads.  Required -- bigla does not guess. Neither numpy nor scipy exposes
        a thread-count API to copy a policy from; the nearest analogue,
        ``scipy.fft.set_workers``, takes an explicit argument and offers a context manager
        rather than auto-detecting. A halve-and-cap heuristic borrowed from application thread
        pools has the wrong goal here: bigla exists for the one calculation on the node that
        should get the whole node.

    Returns
    -------
    int
        The thread count that was actually set.

    Raises
    ------
    RuntimeError
        If no set-threads symbol was found in the backend library.

    Notes
    -----
    This function controls the thread pool of *this library's* OpenBLAS
    instance.  On systems where numpy resolved to a different .so file,
    numpy's thread pool is separate and unaffected.  Use
    :func:`thread_control_info` to see which symbol is being called.
    """
    _ensure_loaded()
    tinfo = _INFO.thread_info  # type: ignore[union-attr]

    if tinfo.set_sym is None:
        raise RuntimeError(
            "set_num_threads: no known thread-control symbol found in backend "
            f"({_INFO.path}).  "  # type: ignore[union-attr]
            "You can set the thread count via the OPENBLAS_NUM_THREADS environment "
            "variable before importing bigla."
        )

    fn = getattr(_LIB, tinfo.set_sym)  # type: ignore[arg-type]
    fn.restype = None
    fn.argtypes = [ctypes.c_int]
    fn(ctypes.c_int(n))

    # Report what the library ACTUALLY has, not what we asked for. A serial OpenBLAS build
    # (Fedora's openblas-serial64_, and any -DUSE_THREAD=0 build) exports the setter and
    # accepts the call, but has no pool to resize: it stays at 1. Returning `n` there made
    # this function's documented "the thread count that was actually set" a falsehood, and
    # a caller sizing work by the return value would over-subscribe by that factor.
    actual = _query_threads(_LIB, tinfo.get_sym) if tinfo.get_sym else n  # type: ignore[arg-type]
    if actual and actual != n:
        log.debug(
            "set_num_threads: asked for %d, backend reports %d (single-threaded build?) via %s",
            n,
            actual,
            _INFO.path,  # type: ignore[union-attr]
        )
    else:
        log.debug(
            "set_num_threads: %s(%d) via %s", tinfo.set_sym, n, _INFO.path  # type: ignore[union-attr]
        )
    # 0 means "not queryable" (no get symbol); fall back to the requested value there rather
    # than reporting a count the caller cannot act on.
    return actual or n


@contextlib.contextmanager
def num_threads(n: int):
    """Temporarily set the backend thread count, restoring it on exit.

    Mirrors ``scipy.fft.set_workers``: scoped control is what callers actually want, and it
    composes with the k-fold / lambda-grid loops this package exists to serve.

        with bigla.num_threads(16):
            w, Q = bigla.eigh(K)

    Controls THIS library's handle only. Where numpy resolved to a different .so its pool is
    separate and the two will oversubscribe; ``threadpoolctl.threadpool_limits`` limits every
    pool at once and is the right tool for that case (see docs/backends.md).
    """
    previous = get_num_threads()
    if not previous:
        # No get symbol: we can set but cannot read back, so there is nothing to restore to.
        # Say so rather than silently leaving the count changed for the rest of the process.
        log.warning(
            "bigla.num_threads(%d): backend %s exposes no thread-count getter, so the previous "
            "value cannot be restored on exit; the setting will persist.",
            n,
            _INFO.path if _INFO else "<unloaded>",
        )
    try:
        set_num_threads(n)
        yield n
    finally:
        if previous:
            set_num_threads(previous)
