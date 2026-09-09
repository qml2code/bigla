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
    """The policy must gate only the two directories, never anything in tests/ itself.

    Each directory is collected in its OWN run. The obvious spelling --
    ``_collected("tests", "tests/long", "tests/huge")`` -- measures pytest's handling of a
    parent argument alongside its children, not this repo's policy, and the two pytest
    generations disagree about it: 9 collects parent and child, 8 keeps only the child and
    silently drops the parent's own tests. That difference cost the environment matrix a
    full run, where the distro rows (python3-pytest: Debian trixie 8.3.5, Fedora 44 8.4.2)
    reported a 9-item pass while the entire fast suite never executed. Collecting one
    directory per run asks the question this test means to ask, on every pytest.
    """
    bare = _collected()
    fast_only = {i for i in _collected("tests") if not any(f"tests/{d}/" in i for d in OPT_IN)}
    assert bare == fast_only
    # ...and the opt-in directories are reachable, so `bare` is a real subset of the whole
    # tree rather than equal to it because nothing else exists.
    for name in OPT_IN:
        assert _dirs_present(_collected(f"tests/{name}")) == {name}
