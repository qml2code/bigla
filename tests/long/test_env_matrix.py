"""Environment-matrix assertions: which backend a row actually resolved.

FIXES.md §6 names the assertion that makes the matrix meaningful and is easy to omit:

> each row must assert **which backend was resolved**, not merely that the tests passed.

A Debian container with pip-installed numpy resolves to numpy's bundled
``libscipy_openblas64`` and tests nothing about Debian -- a green row proving nothing. So
every row declares what it expects and fails if reality disagrees.

Expectations arrive as environment variables, set by the matrix row (see ``Dockerfile`` and
``.github/workflows/backends.yml``):

===============================  =========================================================
``BIGLA_EXPECT_PATH_RE``         regex the resolved library path must match
``BIGLA_EXPECT_DECORATION``      exact ``decoration_str``, e.g. ``scipy_NAME_64_``
``BIGLA_EXPECT_CONFIDENCE``      ``config-string`` | ``mkl-interface`` | ``suffix-only`` | ``lp64``
``BIGLA_EXPECT_ILP64``          ``1`` / ``0``
``BIGLA_EXPECT_NO_BACKEND``      ``1`` for the refusal row: loading must raise
===============================  =========================================================

Every test skips when its variable is unset, so this file is inert outside a matrix row.
It lives in ``tests/long/`` and is reached only by ``make test-long`` (FIXES.md §7).

WSL is deliberately not a matrix row. It gets two skip-unless-WSL tests instead, because it
changes two things nothing else does: ``/proc/meminfo`` reports the VM's allocation rather
than the Windows host's, and a memmap on a DrvFs path is not a memmap on ext4.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

import bigla
from bigla._backend import BiglaBackendError, backend_info
from bigla.diagnose import _collect, _row, _wsl


def _expect(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} unset: not running as an environment-matrix row")
    return value


# ---------------------------------------------------------------------------
# What the row resolved
# ---------------------------------------------------------------------------


def test_resolved_path_matches_row_expectation():
    """The row tests the library it claims to test, not numpy's bundled one."""
    pattern = _expect("BIGLA_EXPECT_PATH_RE")
    path = backend_info().path

    assert re.search(pattern, path), (
        f"row expected a library matching {pattern!r} but resolved {path!r} -- "
        "this row is testing a different backend than it claims"
    )


def test_resolved_decoration_matches_row_expectation():
    """Decoration is the field that differs between distro builds; pin it per row."""
    expected = _expect("BIGLA_EXPECT_DECORATION")

    assert backend_info().decoration_str == expected


def test_resolved_confidence_matches_row_expectation():
    """How ILP64 was established -- the MKL row exists to reach ``mkl-interface``."""
    expected = _expect("BIGLA_EXPECT_CONFIDENCE")

    assert backend_info().confidence == expected


def test_resolved_ilp64_matches_row_expectation():
    expected = _expect("BIGLA_EXPECT_ILP64") == "1"
    info = backend_info()

    assert info.ilp64 is expected
    assert info.max_dim == ((2**62) if expected else 46340)


def test_no_backend_row_refuses_actionably():
    """The refusal row asserts the error message is correct and actionable, not just that
    it raised: a wrong-but-loud failure here is what sends a user down the wrong path."""
    _expect("BIGLA_EXPECT_NO_BACKEND")

    with pytest.raises(BiglaBackendError) as exc:
        backend_info()

    message = str(exc.value)
    assert "docs/backends.md" in message
    # The dependency, not the extra: bigla is necessarily installed by the time this raises,
    # so `pip install bigla[openblas]` would resolve bigla itself again -- and bigla is not on
    # PyPI, so it would resolve nothing at all.
    assert "scipy-openblas64" in message
    assert "BIGLA_LIB" in message


# ---------------------------------------------------------------------------
# The row's own emitted record
# ---------------------------------------------------------------------------


def test_row_json_is_complete_and_renderable():
    """A row that cannot be rendered is a row that never reaches docs/backends.md."""
    env = os.environ.get("BIGLA_ROW_NAME")
    if not env:
        pytest.skip("BIGLA_ROW_NAME unset: not running as an environment-matrix row")

    data = _collect(env)
    rendered = _row(data)

    assert data["environment"] == env
    assert data["wsl"] in ("no", "wsl1", "wsl2")
    # SPEC §9: eight columns, so eight separators around seven inner cells.
    assert rendered.startswith("|") and rendered.endswith("|")
    assert rendered.count("|") - rendered.count("\\|") == 9
    if not data["error"]:
        assert data["library_path"] in rendered


# ---------------------------------------------------------------------------
# WSL -- not a matrix row, but two behaviours nothing else exercises
# ---------------------------------------------------------------------------

_NOT_WSL = _wsl() == "no"


@pytest.mark.skipif(_NOT_WSL, reason="WSL-only: /proc/meminfo semantics differ there")
def test_wsl_memory_probe_is_reported_not_trusted():
    """Under WSL2 /proc/meminfo reports the VM's allocation, not the Windows host's.

    diagnose must still produce a number -- the row is worth recording -- but the WSL
    field must be set alongside it so nobody reads the figure as a host guarantee.
    """
    data = _collect("wsl-probe")

    assert data["wsl"].startswith("wsl")
    assert data["mem_available_gib"] != "N/A"


@pytest.mark.skipif(_NOT_WSL, reason="WSL-only: DrvFs is only mounted there")
def test_wsl_memmap_on_drvfs(tmp_path):
    """A memmap on a DrvFs path (/mnt/c/...) must factor identically to one on ext4.

    Falls back to skipping rather than failing when no DrvFs mount is writable, since a
    WSL install without a mounted Windows drive is legitimate.
    """
    drvfs = None
    for candidate in ("/mnt/c", "/mnt/d"):
        if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            drvfs = candidate
            break
    if drvfs is None:
        pytest.skip("no writable DrvFs mount")

    rng = np.random.default_rng(0)
    m = rng.standard_normal((32, 32))
    a = m @ m.T + 32 * np.eye(32)

    import tempfile

    with tempfile.TemporaryDirectory(dir=drvfs) as d:
        path = os.path.join(d, "A.bin")
        mm = np.memmap(path, dtype=np.float64, mode="w+", shape=a.shape)
        mm[:] = a
        mm.flush()
        mm2 = np.memmap(path, dtype=np.float64, mode="r+", shape=a.shape)
        c_drvfs, _ = bigla.cho_factor(mm2, lower=True)

    c_ext4, _ = bigla.cho_factor(a, lower=True)
    assert np.allclose(np.tril(c_drvfs), np.tril(c_ext4))
