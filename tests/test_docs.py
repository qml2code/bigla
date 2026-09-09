"""Documentation that can go stale silently (§5.1, §5.2).

Both defects here were textual, not behavioural: a docstring said `(default True)` where the
signature said False, and `__all__` listed a name twice. Nothing failed, and nothing would
have. These checks are generic so they catch the next drift rather than only this one.
"""
import inspect
import os
import re

import pytest

import bigla
from bigla._backend import _candidates

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


# The README's numbered discovery list is the most-read statement of the search order, and it
# is the copy that went stale when MKL moved last: SPEC §5.1 and the _backend module docstring
# were both amended, the README was not, and it kept advertising MKL ahead of system OpenBLAS
# for as long as the repo was public. The order is not cosmetic -- see the assertion below.
_README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")


def _readme_section(heading):
    """The text of one `## <heading>` section, up to the next `##`."""
    text = open(_README, encoding="utf-8").read()
    start = text.index(f"## {heading}")
    rest = text[start + 3 :]
    end = rest.find("\n## ")
    return rest if end == -1 else rest[:end]


def test_mkl_is_probed_after_every_openblas_candidate(monkeypatch):
    """MKL last is a behavioural requirement, not a preference: `_validate_ilp64` calls
    `mkl_set_interface_layer(1)` merely to PROBE, and that mutates the interface layer
    process-wide -- including for a NumPy that is itself MKL-linked and running LP64. Reaching
    MKL only when nothing else worked is what keeps that probe from firing needlessly.
    """
    monkeypatch.delenv("BIGLA_LIB", raising=False)  # else _candidates yields only that path
    sonames = [c for c in _candidates() if os.sep not in c]
    mkl = [i for i, n in enumerate(sonames) if "mkl" in n]
    openblas = [i for i, n in enumerate(sonames) if "openblas64" in n or "flexiblas" in n]
    assert mkl and openblas, f"expected both families among {sonames}"
    assert min(mkl) > max(openblas), f"MKL must be probed last, got {sonames}"


def test_readme_discovery_order_agrees_with_the_code():
    """The exact drift that shipped: the README had MKL at #4 and system OpenBLAS at #5."""
    section = _readme_section("Backend discovery")
    assert "libmkl_rt" in section and "libopenblas64" in section
    assert section.index("libmkl_rt") > section.rindex(
        "libopenblas64"
    ), "README lists MKL before system OpenBLAS; _candidates() yields MKL last"
