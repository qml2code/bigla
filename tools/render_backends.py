#!/usr/bin/env python3
"""Collect environment-matrix rows into backends.json and regenerate docs/backends.md.

Each matrix row emits ``python -m bigla.diagnose --format=json`` into a file. This script
gathers those files, writes the combined ``backends.json``, and splices the rendered table
between the markers in ``docs/backends.md``.

The point of generating rather than transcribing: a row in that table is a claim that
someone ran ``diagnose`` on that machine. Hand-editing makes the claim cheap, and a table
of guesses is worse than an empty one.

    python tools/render_backends.py out/                 # regenerate from a row directory
    python tools/render_backends.py out/ --check         # fail if the file is out of date

``--check`` is for CI: it regenerates into memory and diffs, so a matrix run that changes a
decoration cannot land without the committed table changing with it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigla.diagnose import _row  # noqa: E402

BEGIN = "<!-- BEGIN GENERATED ROWS -->"
END = "<!-- END GENERATED ROWS -->"
HEADER = (
    "| environment | numpy source | scipy source | library resolved | decoration "
    "| ILP64 verified by | max n tested | notes / gotchas |\n"
    "|---|---|---|---|---|---|---|---|"
)


def collect(indir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(indir, "*.json"))):
        if os.path.basename(path) == "backends.json":
            continue
        with open(path, encoding="utf-8") as fh:
            rows.append(json.load(fh))
    # Stable order by environment label, so a re-run with the same rows is a no-op diff
    # rather than a reshuffle that hides the real change.
    return sorted(rows, key=lambda d: d.get("environment", ""))


def render(rows: list[dict]) -> str:
    if not rows:
        return f"{HEADER}\n| _(no matrix rows collected)_ |||||||||"
    return "\n".join([HEADER, *(_row(r) for r in rows)])


def splice(markdown: str, table: str) -> str:
    start = markdown.find(BEGIN)
    stop = markdown.find(END)
    if start == -1 or stop == -1 or stop < start:
        raise SystemExit(
            f"docs/backends.md is missing the {BEGIN} / {END} markers; "
            "add them around the verified-environments table"
        )
    return markdown[: start + len(BEGIN)] + "\n" + table + "\n" + markdown[stop:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", help="directory of per-row diagnose JSON files")
    parser.add_argument("--docs", default="docs/backends.md")
    parser.add_argument("--json-out", default="backends.json")
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if the committed table is out of date",
    )
    args = parser.parse_args(argv)

    # Refuse rather than write an empty table. This tool exists because "a row is a claim
    # that someone ran diagnose on that machine", and the six committed rows cost nine CI
    # runs and a container session to produce. Passing the OUTPUT file instead of the input
    # directory -- `render_backends.py backends.json` -- used to glob nothing, render the
    # placeholder, overwrite both outputs and exit 0: a typo silently downgrading a verified
    # record to an empty one, which is the exact inversion of the point of generating it.
    # Two checks, because they are different mistakes and deserve different messages.
    if not os.path.isdir(args.indir):
        raise SystemExit(
            f"{args.indir!r} is not a directory. Pass the directory of per-row "
            f"`diagnose --format=json` files, not a single file"
            + (" -- that is this script's OUTPUT" if args.indir.endswith("backends.json") else "")
        )
    rows = collect(args.indir)
    if not rows:
        raise SystemExit(
            f"no row JSON found in {args.indir!r}; expected files written by "
            f"`python -m bigla.diagnose --format=json`. Refusing to overwrite "
            f"{args.docs} and {args.json_out} with an empty table"
        )
    table = render(rows)
    with open(args.docs, encoding="utf-8") as fh:
        current = fh.read()
    updated = splice(current, table)

    if args.check:
        if updated != current:
            print(f"{args.docs} is out of date; run: python {sys.argv[0]} {args.indir}")
            return 1
        print(f"{args.docs} is up to date ({len(rows)} rows)")
        return 0

    with open(args.docs, "w", encoding="utf-8") as fh:
        fh.write(updated)
    with open(args.json_out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {args.json_out} and updated {args.docs} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
