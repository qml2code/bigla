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
import json
import os
import re
import subprocess

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "backends.yml")
DOCKERFILE = os.path.join(ROOT, "Dockerfile")
BACKENDS_MD = os.path.join(ROOT, "docs", "backends.md")
BACKENDS_JSON = os.path.join(ROOT, "backends.json")

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


# ---------------------------------------------------------------------------
# render_backends refuses to write an empty table
#
# `python tools/render_backends.py backends.json` -- passing the OUTPUT file where the input
# directory belongs -- used to glob nothing, render the "(no matrix rows collected)"
# placeholder, overwrite docs/backends.md AND backends.json, and exit 0. A typo silently
# downgraded six measured rows to none, in a tool whose whole premise is that a row is a
# claim someone ran diagnose on that machine. Recoverable from git, but only once noticed.
# ---------------------------------------------------------------------------


def _write_targets(tmp_path):
    """Copies of the two files the tool writes, so a regression cannot touch the real ones."""
    docs = tmp_path / "backends.md"
    docs.write_text(
        "<!-- BEGIN GENERATED ROWS -->\n| keep me |\n<!-- END GENERATED ROWS -->\n",
        encoding="utf-8",
    )
    js = tmp_path / "backends.json"
    js.write_text('[{"kept": true}]\n', encoding="utf-8")
    return docs, js


@pytest.mark.parametrize("bad", ["a file, not a directory", "an empty directory"])
def test_render_backends_refuses_to_empty_the_record(tmp_path, bad):
    docs, js = _write_targets(tmp_path)
    if bad == "a file, not a directory":
        indir = tmp_path / "backends.json"  # the reported typo: the tool's own output
    else:
        indir = tmp_path / "empty"
        indir.mkdir()

    with pytest.raises(SystemExit) as exc:
        _render_backends().main([str(indir), "--docs", str(docs), "--json-out", str(js)])

    assert exc.value.code != 0, f"{bad}: exited 0"
    # ...and, the point of the guard, wrote nothing.
    assert "keep me" in docs.read_text(encoding="utf-8"), f"{bad}: clobbered the table"
    assert "kept" in js.read_text(encoding="utf-8"), f"{bad}: clobbered backends.json"


# ---------------------------------------------------------------------------
# backends.yml `notes:` -> BIGLA_ROW_NOTES -> backends.json -> the rendered table
#
# The one coupling in this file that nothing pinned, and its first use was already a
# hand-edit: when per-row notes were introduced, they were transplanted into the committed
# backends.json rather than waiting for a matrix run, since `notes` is editorial rather than
# measured. Defensible, but it leaves two silent failure modes -- edit the YAML without a
# re-run and the table keeps the old prose; edit the JSON without the YAML and the next run
# reverts it.
#
# Scope is deliberate. Only rows present in BOTH files are compared, so ADDING a matrix row
# does not turn the fast suite red before the nightly has had a chance to run it: a row that
# has never run belongs in neither backends.json nor the table, which is the whole premise
# of generating them. A row in backends.json that the matrix no longer defines is stale, and
# that IS an error.
# ---------------------------------------------------------------------------


def _collected_rows() -> dict[str, dict]:
    with open(BACKENDS_JSON, encoding="utf-8") as fh:
        return {r["environment"]: r for r in json.load(fh)}


def _flat(text: str) -> str:
    """Compare prose by words: YAML folds `>-` blocks to single spaces, JSON keeps what it
    was given, and neither difference is a drift worth failing on."""
    return " ".join((text or "").split())


def test_backends_json_has_no_row_the_matrix_no_longer_defines():
    names = {r["name"] for r in ROWS}
    stale = sorted(set(_collected_rows()) - names)
    assert not stale, f"backends.json rows absent from the matrix: {stale}"


def test_row_notes_match_the_matrix_definitions():
    """`notes` is the only field in backends.json that is authored rather than measured,
    so it is the only one that can drift from its source without a run to correct it."""
    collected = _collected_rows()
    compared = 0
    for entry in ROWS:
        row = collected.get(entry["name"])
        if row is None:
            continue  # declared but never run; nothing to compare yet
        assert _flat(row["notes"]) == _flat(entry["notes"]), (
            f"{entry['id']}: backends.json notes disagree with backends.yml. Re-run the "
            f"matrix, or correct whichever is wrong."
        )
        compared += 1
    assert compared, "no matrix row appears in backends.json; the coupling is untested"


# ---------------------------------------------------------------------------
# Matrix values must not be interpolated into shell scripts.
#
# `${{ }}` splices raw text in BEFORE bash parses the script, so a value containing a quote
# ends the argument it was meant to sit inside. The per-row notes begin "the common case:
# numpy's bundled ILP64 OpenBLAS ...", and the nightly of 2026-09-10 died on the
# pip-numpy-wheel row with `unexpected EOF while looking for matching \`''` -- a shell syntax
# error, before docker ran at all. Only that row had an apostrophe.
#
# `notes` is free-form prose a human writes, so this recurs by construction unless the shape
# is banned. Values belong in the step's `env:` block, which is also GitHub's own advice
# against script injection.
# ---------------------------------------------------------------------------

_INTERPOLATION = re.compile(r"\$\{\{[^}]*\}\}")


def _workflow_steps() -> list[dict]:
    with open(WORKFLOW, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["jobs"]["row"]["steps"]


def test_no_matrix_value_is_interpolated_into_a_run_script():
    offenders = []
    for step in _workflow_steps():
        for hit in _INTERPOLATION.findall(step.get("run", "")):
            offenders.append(f"{step.get('name', '<unnamed>')}: {hit}")
    assert not offenders, (
        "pass these through the step's `env:` block instead -- interpolating into `run:` "
        f"breaks on any value containing a quote: {offenders}"
    )


def test_every_row_note_survives_a_shell_round_trip():
    """The values themselves, checked directly: prose reaches the container intact.

    Complements the test above -- that one bans the dangerous shape, this one confirms the
    data is actually deliverable, so a note containing something no quoting survives would be
    caught even if the shape were changed again.
    """
    for row in ROWS:
        note = row["notes"]
        proof = subprocess.run(
            ["bash", "-c", 'printf "%s" "$BIGLA_ROW_NOTES"'],
            env={"BIGLA_ROW_NOTES": note, "PATH": os.environ.get("PATH", "")},
            capture_output=True,
            text=True,
        )
        assert proof.returncode == 0, f"{row['id']}: {proof.stderr.strip()}"
        assert proof.stdout == note, f"{row['id']}: note was mangled in transit"
