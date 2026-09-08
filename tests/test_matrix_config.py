"""The environment matrix must stay internally consistent.

FIXES.md §6 names the assertion that makes the matrix meaningful and is easy to omit:

> each row must assert **which backend was resolved**, not merely that the tests passed.

A row that declares no expectations still goes green, and a green row proving nothing is
worse than no row -- it reads as coverage. These tests make that structurally impossible:
a new row either declares what it expects to resolve, or it is the refusal row and says so.

They also pin the two places a row is described -- ``.github/workflows/backends.yml`` and
``Dockerfile`` -- against drift, since a ``BLAS_SOURCE`` with no matching case in the
Dockerfile fails only at container build time, an hour into a nightly run.
"""

from __future__ import annotations

import importlib.util
import os
import re

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "backends.yml")
DOCKERFILE = os.path.join(ROOT, "Dockerfile")
BACKENDS_MD = os.path.join(ROOT, "docs", "backends.md")

EXPECTATIONS = (
    "expect_path",
    "expect_decoration",
    "expect_confidence",
    "expect_ilp64",
)


def _rows() -> list[dict]:
    with open(WORKFLOW, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["jobs"]["row"]["strategy"]["matrix"]["include"]


ROWS = _rows()


def test_matrix_has_the_six_provenance_rows():
    """Six rows chosen by backend provenance, not by OS (FIXES.md §6)."""
    assert len(ROWS) == 6
    assert len({r["id"] for r in ROWS}) == 6, "duplicate row id"


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["id"])
def test_row_asserts_what_it_resolved(row):
    """Either the row declares every expectation, or it is the refusal row."""
    if row.get("expect_no_backend"):
        assert not any(
            row.get(k) for k in EXPECTATIONS
        ), f"{row['id']}: the refusal row cannot also expect a resolved backend"
        return

    missing = [k for k in EXPECTATIONS if not row.get(k)]
    assert not missing, (
        f"{row['id']} declares no {', '.join(missing)}: it would go green without "
        "testing which backend was resolved"
    )
    assert row["expect_ilp64"] in ("0", "1")


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["id"])
def test_row_path_expectation_is_a_real_regex(row):
    """A malformed pattern fails an hour into the nightly run, not here."""
    pattern = row.get("expect_path")
    if not pattern:
        pytest.skip("refusal row: no path expectation")
    re.compile(pattern)
    assert pattern != ".*", f"{row['id']}: a match-anything pattern asserts nothing"


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["id"])
def test_row_blas_source_has_a_dockerfile_case(row):
    with open(DOCKERFILE, encoding="utf-8") as fh:
        dockerfile = fh.read()
    assert re.search(
        rf"^\s*{re.escape(row['blas'])}\)", dockerfile, re.M
    ), f"{row['id']}: BLAS_SOURCE={row['blas']} has no case in the Dockerfile"


def test_backends_md_has_generation_markers():
    """tools/render_backends.py splices between these; without them it exits non-zero."""
    with open(BACKENDS_MD, encoding="utf-8") as fh:
        text = fh.read()
    assert text.count("<!-- BEGIN GENERATED ROWS -->") == 1
    assert text.count("<!-- END GENERATED ROWS -->") == 1
    assert text.index("<!-- BEGIN") < text.index("<!-- END")


# ---------------------------------------------------------------------------
# Column coupling
#
# Four places independently describe the shape of a docs/backends.md row:
# diagnose._row emits the cells, render_backends.HEADER writes the header above them,
# docs/backends.md holds the committed table, and SPEC §9 documents the contract. Add a
# column to one and the generated table silently misaligns -- markdown does not complain
# about a row with the wrong number of cells, it just renders it wrong, and every
# previously collected row is affected at once.
#
# tests/long/test_env_matrix.py does assert the emitted row has 9 separators, but it is
# doubly gated: it lives in tests/long AND skips unless BIGLA_ROW_NAME is set, so it runs
# only inside a matrix container. No container has run yet, so that assertion has never
# executed. These do, on every `make test`, and need nothing but the repo.
# ---------------------------------------------------------------------------


def _cells(line: str) -> int:
    """Number of cells in a markdown table row.

    diagnose._row escapes a literal ``|`` inside a cell as ``\\|`` -- deliberately, since a
    BLAS config string is free to contain one -- so those must not be counted as column
    separators. Counting raw pipes passes today only because no current value has one.
    """
    return line.count("|") - line.count("\\|") - 1


def _generated_row() -> str:
    from bigla.diagnose import _collect, _row

    return _row(_collect("column-coupling-probe"))


def _render_backends():
    spec = importlib.util.spec_from_file_location(
        "render_backends", os.path.join(ROOT, "tools", "render_backends.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _committed_header() -> str:
    with open(BACKENDS_MD, encoding="utf-8") as fh:
        text = fh.read()
    inner = text.split("<!-- BEGIN GENERATED ROWS -->")[1].split("<!-- END GENERATED ROWS -->")[0]
    return next(line for line in inner.strip().splitlines() if line.startswith("|"))


def test_generated_row_matches_the_generated_header():
    """The pair that actually produces a broken table: both halves of the generator."""
    header = _render_backends().HEADER.splitlines()[0]
    assert _cells(_generated_row()) == _cells(header)


def test_committed_table_header_matches_the_generator():
    """A hand-edited header is silently reverted by the next run; catch it as a conflict."""
    assert _committed_header().strip() == _render_backends().HEADER.splitlines()[0].strip()


def test_spec_column_list_matches_the_generated_row():
    """SPEC §9 is the written contract; it must not drift from what is emitted."""
    with open(os.path.join(ROOT, "docs", "SPEC.md"), encoding="utf-8") as fh:
        line = next(l for l in fh if l.startswith("`environment` |"))
    assert len(line.strip().split("|")) == _cells(_generated_row())
