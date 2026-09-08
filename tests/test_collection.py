"""The collection policy in conftest.py is itself worth testing.

It is the only thing standing between a bare ``pytest`` and a 15-minute, 20 GiB run, and
it fails SILENTLY in the dangerous direction: if the hook stops matching -- a rename, a
signature change in a future pytest, a stray edit -- nothing errors, the expensive suites
simply start being collected again, and you find out when a routine `pytest` does not come
back. So assert the policy directly rather than trusting it.

Collection only (``--collect-only``), so this costs milliseconds and never runs the
expensive tests it is checking on.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OPT_IN = ("long", "huge")


def _collected(*args: str) -> set[str]:
    """Node ids pytest would collect for *args*, without running anything."""
    out = subprocess.run(
        # --override-ini clears addopts, which carries -v: that would cancel the -q and
        # make --collect-only print an indented tree instead of the flat node ids parsed
        # below. Neutralising it here keeps this test independent of pyproject's addopts.
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--override-ini=addopts=",
            *args,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    ids = set()
    for line in out.stdout.splitlines():
        line = line.strip()
        if "::" in line and not line.startswith("<"):
            ids.add(line.split("[")[0])
    assert ids, f"collected nothing for {args!r}\n{out.stdout}\n{out.stderr}"
    return ids


def _dirs_present(ids: set[str]) -> set[str]:
    return {d for d in OPT_IN if any(f"tests/{d}/" in i for i in ids)}


def test_bare_pytest_skips_the_expensive_directories():
    """A bare `pytest` must not reach tests/huge -- that is 20 GiB and 15 minutes."""
    assert _dirs_present(_collected()) == set()


def test_naming_a_directory_collects_it():
    for name in OPT_IN:
        ids = _collected(f"tests/{name}")
        assert _dirs_present(ids) == {name}, f"pytest tests/{name} collected {ids}"


def test_naming_a_single_file_inside_collects_it():
    """`pytest tests/huge/test_huge.py` is a natural thing to type; it must work."""
    ids = _collected("tests/huge/test_huge.py")
    assert _dirs_present(ids) == {"huge"}


def test_fast_suite_is_not_narrowed_by_the_policy():
    """The policy must gate only the two directories, never anything in tests/ itself."""
    bare = _collected()
    everything = _collected("tests", "tests/long", "tests/huge")
    fast_only = {i for i in everything if not any(f"tests/{d}/" in i for d in OPT_IN)}
    assert bare == fast_only
