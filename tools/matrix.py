#!/usr/bin/env python3
"""Run environment-matrix rows locally, using the same definitions CI uses.

The rows and their expectations live in ONE place -- ``.github/workflows/backends.yml`` --
and this script reads them from there. Declaring them twice is how a local reproduction
starts quietly disagreeing with CI, which matters most in exactly the situation you reach
for this script: CI reported a decoration that disagreed with what a row expected, and you
need to see it on a laptop rather than by pushing commits.

    python tools/matrix.py list
    python tools/matrix.py run --row debian-openblas64
    python tools/matrix.py run --all

Needs docker (or set ``--docker podman``). Rows write ``out/<id>.json``, which
``tools/render_backends.py out/`` then turns into the docs/backends.md table.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "backends.yml")

ENV_FOR = {
    "expect_path": "BIGLA_EXPECT_PATH_RE",
    "expect_decoration": "BIGLA_EXPECT_DECORATION",
    "expect_confidence": "BIGLA_EXPECT_CONFIDENCE",
    "expect_ilp64": "BIGLA_EXPECT_ILP64",
    "expect_no_backend": "BIGLA_EXPECT_NO_BACKEND",
}


def load_rows() -> list[dict]:
    try:
        import yaml
    except ImportError:
        raise SystemExit("pyyaml is needed to read the row definitions: pip install pyyaml")
    with open(WORKFLOW, encoding="utf-8") as fh:
        wf = yaml.safe_load(fh)
    try:
        return wf["jobs"]["row"]["strategy"]["matrix"]["include"]
    except (KeyError, TypeError):
        raise SystemExit(f"{WORKFLOW}: no jobs.row.strategy.matrix.include")


def run_row(row: dict, docker: str, outdir: str) -> int:
    tag = f"bigla-row:{row['id']}"
    build = [
        docker, "build",
        "--build-arg", f"BASE_IMAGE={row['base']}",
        "--build-arg", f"BLAS_SOURCE={row['blas']}",
        "-t", tag, ROOT,
    ]  # fmt: skip
    print(f"\n=== {row['id']}: {row['name']} ===\n$ {' '.join(build)}", flush=True)
    rc = subprocess.call(build)
    if rc:
        return rc

    env_args = ["-e", f"BIGLA_ROW_ID={row['id']}", "-e", f"BIGLA_ROW_NAME={row['name']}",
                "-e", f"BIGLA_ROW_NOTES={row['name']}"]  # fmt: skip
    for key, var in ENV_FOR.items():
        # Unset rather than empty: test_env_matrix skips on an unset expectation, and an
        # empty string would read as "expect the empty pattern" instead.
        if row.get(key):
            env_args += ["-e", f"{var}={row[key]}"]

    run = [docker, "run", "--rm", *env_args, "-v", f"{outdir}:/out", tag]
    print(f"$ {' '.join(run)}", flush=True)
    return subprocess.call(run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    run_p = sub.add_parser("run")
    run_p.add_argument("--row", action="append", default=[])
    run_p.add_argument("--all", action="store_true")
    run_p.add_argument("--docker", default="docker")
    run_p.add_argument("--out", default=os.path.join(ROOT, "out"))
    args = parser.parse_args(argv)

    rows = load_rows()
    if args.cmd == "list":
        for row in rows:
            print(f"  {row['id']:<24} {row['base']:<28} BLAS_SOURCE={row['blas']}")
        return 0

    if args.all:
        wanted = rows
    else:
        ids = set(args.row)
        wanted = [r for r in rows if r["id"] in ids]
        missing = ids - {r["id"] for r in wanted}
        if missing or not wanted:
            raise SystemExit(f"unknown row(s): {', '.join(sorted(missing)) or '(none given)'}")

    os.makedirs(args.out, exist_ok=True)
    failures = []
    for row in wanted:
        if run_row(row, args.docker, args.out):
            failures.append(row["id"])

    if failures:
        # A failing row is news, not a crash: a distro that changed its decoration is
        # exactly what the matrix is for. Report and keep the rows that did succeed.
        print(f"\nrows that failed: {', '.join(failures)}", file=sys.stderr)
        print(
            f"rows collected in {args.out}; render with tools/render_backends.py", file=sys.stderr
        )
        return 1
    print(
        f"\nall {len(wanted)} row(s) passed; render with: python tools/render_backends.py {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
