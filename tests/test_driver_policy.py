"""The memory-aware driver policy shared by eigh and svd.

Both `eigh` (evd vs ev) and `svd` (gesdd vs gesvd) pick a driver by asking whether the fast one's
workspace fits inside `_EVD_MEM_FRACTION` of what `_mem_available_bytes()` reports. That probe is the
one piece of machinery the two share, and it is where the policy went wrong: it read only
`/proc/meminfo`, so on macOS it returned None, None means "no bound known", and no-bound-known means
the fast driver ALWAYS. The policy was silently inert on an entire platform, and the CI macOS leg was
the first thing to notice -- after the svd half had shipped, because eigh's half had no test at all.

So these tests deliberately do not read the host's memory. They inject a known bound and check the
decision, which is the thing worth pinning; the host-dependent question -- does this platform report a
bound at all -- gets its own test that cannot fail for the wrong reason.
"""

import builtins

import numpy as np
import pytest

import bigla.linalg as L

# k at which LP64's 32-bit indices run out; the size bigla exists to get past. gesdd wants ~69 GiB of
# scratch here and evd ~32 GiB -- affordable on a large machine and not on a small one, which is why
# every test below that asks about the DECISION at this size injects the bound rather than reading it.
K_WALL = 46_341

# The one size that needs no injected bound: gesdd would want ~290 TB of scratch, so refusing it
# requires only that a bound was reported AT ALL, whatever it was.
K_BEYOND_ANY_MACHINE = 3_000_000


def test_mem_probe_returns_none_or_a_plausible_bound():
    """Platform-agnostic: whatever the probe says, it must be usable as a memory bound."""
    mem = L._mem_available_bytes()
    if mem is None:
        pytest.skip("this platform reports no memory bound; the fallbacks are covered below")
    assert isinstance(mem, int)
    assert 64 * 1024**2 < mem < 2**60, f"implausible memory bound: {mem}"


def test_mem_probe_falls_back_when_proc_meminfo_is_absent(monkeypatch):
    """The macOS regression, reproduced on any platform.

    Without the sysconf fallback this returned None, and both auto-drivers then took the
    memory-hungry path no matter the problem size.
    """
    real_open = builtins.open

    def no_proc(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            raise FileNotFoundError("simulated: platform has no /proc/meminfo")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", no_proc)
    mem = L._mem_available_bytes()
    assert mem is not None, "no /proc/meminfo must fall back to sysconf, not to 'no bound known'"
    assert 64 * 1024**2 < mem < 2**60

    # And the decision that was wrong on macOS is now right. At this size the requirement (~290 TB
    # for gesdd) exceeds any machine, so the assertion turns purely on the probe having answered --
    # no injected bound needed, and no dependence on how much memory the host happens to have.
    assert L._auto_svd_driver(K_BEYOND_ANY_MACHINE, K_BEYOND_ANY_MACHINE, np.float64) == "gesvd"
    assert L._auto_driver(K_BEYOND_ANY_MACHINE, np.float64) == "ev"


@pytest.mark.parametrize("bound_gib,expected", [(8, "gesvd"), (4096, "gesdd")])
def test_svd_auto_driver_against_a_known_bound(monkeypatch, bound_gib, expected):
    """gesdd needs ~69 GiB at the LP64 wall: refused under 8 GiB, affordable under 4 TiB."""
    monkeypatch.setattr(L, "_mem_available_bytes", lambda: bound_gib * 1024**3)
    assert L._auto_svd_driver(K_WALL, K_WALL, np.float64) == expected


@pytest.mark.parametrize("bound_gib,expected", [(8, "ev"), (4096, "evd")])
def test_eigh_auto_driver_against_a_known_bound(monkeypatch, bound_gib, expected):
    """The half that had no test at all. evd's workspace is ~2n**2 doubles -- ~32 GiB at the wall."""
    monkeypatch.setattr(L, "_mem_available_bytes", lambda: bound_gib * 1024**3)
    assert L._auto_driver(K_WALL, np.float64) == expected


def test_a_small_problem_takes_the_fast_driver_even_on_a_tiny_machine(monkeypatch):
    """The policy must only ever refuse the fast driver because of SIZE."""
    monkeypatch.setattr(L, "_mem_available_bytes", lambda: 256 * 1024**2)
    assert L._auto_svd_driver(200, 200, np.float64) == "gesdd"
    assert L._auto_driver(200, np.float64) == "evd"


def test_unknown_bound_takes_the_fast_driver(monkeypatch):
    """Deliberate, and the reason the macOS gap was invisible rather than loud: a heuristic that
    cannot measure a platform must not slow it down. Pinned so the choice stays a choice."""
    monkeypatch.setattr(L, "_mem_available_bytes", lambda: None)
    assert L._auto_svd_driver(K_WALL, K_WALL, np.float64) == "gesdd"
    assert L._auto_driver(K_WALL, np.float64) == "evd"


def test_fraction_is_actually_applied(monkeypatch):
    """A bound just above the requirement still refuses, because only _EVD_MEM_FRACTION of it counts."""
    need = L._gesdd_workspace_bytes(K_WALL, K_WALL, np.float64)
    monkeypatch.setattr(L, "_mem_available_bytes", lambda: int(need * 1.05))
    assert L._auto_svd_driver(K_WALL, K_WALL, np.float64) == "gesvd"
    monkeypatch.setattr(L, "_mem_available_bytes", lambda: int(need / L._EVD_MEM_FRACTION * 1.01))
    assert L._auto_svd_driver(K_WALL, K_WALL, np.float64) == "gesdd"
