"""python -m bigla.diagnose

Prints a paste-ready block with all fields needed for docs/backends.md.
Run on any new system to generate a table row.
"""

from __future__ import annotations

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


def run() -> None:
    import numpy as np

    import bigla
    from bigla._backend import BiglaBackendError, backend_info

    sep = "=" * 64
    print(sep)
    print("bigla diagnose")
    print(sep)

    print(f"bigla_version       : {bigla.__version__}")
    print(f"python_version      : {sys.version.split()[0]}")
    print(f"numpy_version       : {np.__version__}")
    print(f"numpy_blas          : {_numpy_blas()}")

    scipy_ver, scipy_blas = _scipy_info()
    print(f"scipy_version       : {scipy_ver}")
    print(f"scipy_blas          : {scipy_blas}")
    print(f"mem_available_gib   : {_mem_available_gib()}")
    print()

    try:
        info = backend_info()
        print(f"library_path        : {info.path}")
        print(f"config_string       : {info.config_string!r}")
        print(f"decoration          : {info.decoration_str}")
        print(f"ilp64               : {info.ilp64}")
        print(f"confidence          : {info.confidence}")
        print(f"max_dim             : {info.max_dim}")
        print(f"threads             : {info.threads}")
        ok = True
    except BiglaBackendError as exc:
        print(f"BACKEND ERROR: {exc}")
        ok = False

    print()
    print("--- paste-ready table row (fill in env / max_n_tested / notes) ---")
    if ok:
        info = backend_info()
        cfg_short = info.config_string[:60].replace("|", "\\|")
        print(
            f"| <env> "
            f"| pip numpy {np.__version__} "
            f"| pip scipy {scipy_ver} "
            f"| `{info.path}` "
            f"| `{info.decoration_str}` "
            f"| {info.confidence}: `{cfg_short}` "
            f"| <max_n_tested> "
            f"| threads={info.threads} |"
        )
    else:
        print("(backend load failed; see error above)")
    print(sep)


if __name__ == "__main__":
    run()
