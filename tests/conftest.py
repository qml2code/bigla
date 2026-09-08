"""Collection policy: the expensive suites are opt-in by NAMING them.

``tests/long`` and ``tests/huge`` are collected only when the invocation names them. A bare
``pytest`` therefore runs the fast suite and nothing else, while ``pytest tests/huge`` does
exactly what it looks like it does.

This replaces a ``skipif`` on ``BIGLA_TEST_HUGE`` inside ``tests/huge``. That guard existed
for the same reason -- ``testpaths = ["tests"]``, so a bare ``pytest`` collected the two
46500x46500 cases and would have spent 15 minutes and 20 GiB on them -- but it paid for the
protection twice over: ``make test-huge`` had to set a variable purely to appease a guard
aimed at somebody not running make, and ``pytest tests/huge`` silently reported "2 skipped"
unless you already knew the variable existed.

With the policy here, directory membership is the single mechanism. No markers, no
environment variables, and no ``--ignore`` flags in the Makefile.
"""

from __future__ import annotations

from pathlib import Path

TESTS_DIR = Path(__file__).parent
OPT_IN = ("long", "huge")


def pytest_ignore_collect(collection_path: Path, config) -> bool | None:
    """Skip tests/long and tests/huge unless the command line named them.

    Returns None (rather than False) for everything else, which leaves the decision to
    pytest's own rules instead of forcing collection of a path it would have skipped.
    """
    if collection_path.parent != TESTS_DIR or collection_path.name not in OPT_IN:
        return None
    named = any(collection_path.name in Path(arg).parts for arg in config.args)
    return not named
