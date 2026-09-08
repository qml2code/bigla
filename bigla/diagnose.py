"""python -m bigla.diagnose

Prints a paste-ready block with all fields needed for docs/backends.md.
Run on any new system to generate a table row.

Three output formats:

* ``--format=text`` (default) -- the human block, plus a paste-ready row.
* ``--format=row``  -- only the ``docs/backends.md`` table row.
* ``--format=json`` -- one JSON object, which is what an environment-matrix row emits and
  ``tools/render_backends.py`` consumes. Machine-readable so a matrix row is collected
  rather than transcribed.

``--env NAME`` labels the row; in the matrix it is the provenance row's name, so the
``environment`` column says what was actually being tested rather than ``<env>``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _mem_available_gib() -> str:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return f"{int(line.split()[1]) / 2**20:.1f}"
    except Exception:
        pass
    return "N/A"


def _wsl() -> str:
    """WSL flavour, or "no". WSL is not a matrix row but changes two things worth recording.

    Under WSL2 /proc/meminfo reports the VM's allocation, not Windows' -- so a memory
    probe that looks generous can still be refused by the host -- and a memmap on a DrvFs
    path (/mnt/c/...) behaves differently from one on the ext4 root.
    """
    try:
        with open("/proc/version") as fh:
            ver = fh.read().lower()
    except OSError:
        return "no"
    if "microsoft" not in ver:
        return "no"
    return "wsl2" if "wsl2" in ver or os.path.exists("/run/WSL") else "wsl1"


def _numpy_blas() -> str:
    import numpy as np

    try:
        cfg = np.show_config(mode="dicts")
        return cfg.get("Build Dependencies", {}).get("blas", {}).get("name", "unknown")
    except Exception:
        try:
            import numpy.distutils.system_info as si

            return str(si.get_info("blas"))
        except Exception:
            return "unknown"


def _scipy_info() -> tuple[str, str]:
    try:
        import scipy

        ver = scipy.__version__
        try:
            cfg = scipy.show_config(mode="dicts")
            blas = cfg.get("Build Dependencies", {}).get("blas", {}).get("name", "unknown")
        except Exception:
            blas = "unknown"
        try:
            import scipy.linalg

            ilp = getattr(scipy.linalg.lapack, "HAS_ILP64", "unknown")
            blas += f" (HAS_ILP64={ilp})"
        except Exception:
            pass
        return ver, blas
    except ImportError:
        return "not installed", "N/A"


def _collect(env: str) -> dict:
    """Every field docs/backends.md needs, plus what the matrix asserts on."""
    import numpy as np

    import bigla
    from bigla._backend import BiglaBackendError, backend_info

    scipy_ver, scipy_blas = _scipy_info()
    out = {
        "environment": env,
        "bigla_version": bigla.__version__,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "numpy_blas": _numpy_blas(),
        "scipy_version": scipy_ver,
        "scipy_blas": scipy_blas,
        "mem_available_gib": _mem_available_gib(),
        "wsl": _wsl(),
        "max_n_tested": os.environ.get("BIGLA_MAX_N_TESTED", ""),
        "notes": os.environ.get("BIGLA_ROW_NOTES", ""),
    }
    try:
        info = backend_info()
        out.update(
            library_path=info.path,
            config_string=info.config_string,
            decoration=info.decoration_str,
            ilp64=info.ilp64,
            confidence=info.confidence,
            max_dim=info.max_dim,
            threads=info.threads,
            error=None,
        )
    except BiglaBackendError as exc:
        # The refusal row is a legitimate matrix outcome, not a failure to report: it is
        # how the error message stays correct and actionable.
        out.update(
            library_path=None,
            config_string=None,
            decoration=None,
            ilp64=False,
            confidence=None,
            max_dim=None,
            threads=None,
            error=str(exc),
        )
    return out


def _row(d: dict) -> str:
    """One docs/backends.md table row (SPEC 9 columns) from a collected dict."""

    def cell(value) -> str:
        return str(value).replace("|", "\\|") if value else "-"

    if d["error"]:
        resolved = "**none found**"
        verified = "refused: no ILP64 backend"
    else:
        resolved = f"`{d['library_path']}`"
        cfg_short = cell(d["config_string"])[:60]
        verified = f"{d['confidence']}: `{cfg_short}`"
    return (
        f"| {cell(d['environment'])} "
        f"| numpy {cell(d['numpy_version'])} ({cell(d['numpy_blas'])}) "
        f"| scipy {cell(d['scipy_version'])} "
        f"| {resolved} "
        f"| `{cell(d['decoration'])}` "
        f"| {verified} "
        f"| {cell(d['max_n_tested'])} "
        f"| {cell(d['notes'])} |"
    )


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m bigla.diagnose")
    parser.add_argument(
        "--format",
        choices=("text", "row", "json"),
        default="text",
        help="text: human block (default); row: docs/backends.md row; json: machine-readable",
    )
    parser.add_argument(
        "--env",
        default="<env>",
        help="label for the environment column (the matrix passes its provenance row name)",
    )
    args = parser.parse_args(argv)

    data = _collect(args.env)

    if args.format == "json":
        print(json.dumps(data, indent=2, sort_keys=True))
        return
    if args.format == "row":
        print(_row(data))
        return

    sep = "=" * 64
    print(sep)
    print("bigla diagnose")
    print(sep)
    for key in (
        "bigla_version",
        "python_version",
        "numpy_version",
        "numpy_blas",
        "scipy_version",
        "scipy_blas",
        "mem_available_gib",
        "wsl",
    ):
        print(f"{key:<20}: {data[key]}")
    print()
    if data["error"]:
        print(f"BACKEND ERROR: {data['error']}")
    else:
        for key in (
            "library_path",
            "config_string",
            "decoration",
            "ilp64",
            "confidence",
            "max_dim",
            "threads",
        ):
            value = data[key]
            print(f"{key:<20}: {value!r}" if key == "config_string" else f"{key:<20}: {value}")
    print()
    print("--- paste-ready table row (fill in max_n_tested / notes) ---")
    print(_row(data))
    print(sep)


if __name__ == "__main__":
    run()
