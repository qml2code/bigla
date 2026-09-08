"""Documentation that can go stale silently (§5.1, §5.2).

Both defects here were textual, not behavioural: a docstring said `(default True)` where the
signature said False, and `__all__` listed a name twice. Nothing failed, and nothing would
have. These checks are generic so they catch the next drift rather than only this one.
"""
import inspect
import re

import pytest

import bigla

_DEFAULT_RE = re.compile(r"\(default\s+([A-Za-z0-9_.'\"-]+)", re.IGNORECASE)


def _public_callables():
    for name in bigla.__all__:
        obj = getattr(bigla, name)
        if inspect.isfunction(obj) or inspect.ismethod(obj):
            yield name, obj


def test_all_is_clean():
    """No duplicates, and every name resolves."""
    dupes = sorted({n for n in bigla.__all__ if bigla.__all__.count(n) > 1})
    assert not dupes, f"duplicated in __all__: {dupes}"
    missing = [n for n in bigla.__all__ if not hasattr(bigla, n)]
    assert not missing, f"in __all__ but not importable: {missing}"


@pytest.mark.parametrize(
    "name,func", list(_public_callables()), ids=lambda x: getattr(x, "__name__", x)
)
def test_documented_defaults_match_signatures(name, func):
    """Where a numpydoc block states `(default X)`, X must be the actual default.

    The False defaults are the deliberate scipy-compatible choice (SPEC §6.1), so when these
    disagree it is the prose that is wrong.
    """
    doc = inspect.getdoc(func) or ""
    sig = inspect.signature(func)
    problems = []
    for line in doc.splitlines():
        m = _DEFAULT_RE.search(line)
        if not m:
            continue
        stated = m.group(1).rstrip(").,")
        param = line.split(":")[0].strip()
        if param not in sig.parameters:
            continue
        actual = sig.parameters[param].default
        if actual is inspect.Parameter.empty:
            problems.append(f"{param}: doc says default {stated}, signature has no default")
        elif str(actual) != stated and repr(actual) != stated:
            problems.append(f"{param}: doc says default {stated}, signature says {actual!r}")
    assert not problems, f"{name}: " + "; ".join(problems)
