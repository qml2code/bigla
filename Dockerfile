# One Dockerfile for the whole environment matrix (FIXES.md §6), selected by build args.
# Six rows chosen by BACKEND PROVENANCE, not by OS -- the OS is incidental, the question is
# always "which library does discovery actually resolve, and what decoration does it use".
#
#   BLAS_SOURCE=wheel              pip numpy wheel, bundled scipy_openblas64  (the common case)
#   BLAS_SOURCE=scipy-openblas64   the documented escape hatch, as a standalone package
#   BLAS_SOURCE=debian             distro decoration + system loader
#   BLAS_SOURCE=fedora             a second distro decoration
#   BLAS_SOURCE=mkl                mkl_set_interface_layer branch + LP64 delegation
#   BLAS_SOURCE=none               the refusal path -- that the error stays actionable
#
# THE TRAP THIS FILE EXISTS TO AVOID: a pip numpy wheel bundles its own
# numpy.libs/libscipy_openblas64_*.so, and _candidates() reaches that BEFORE any system
# library. So a Debian container with pip-installed numpy resolves numpy's bundled copy and
# tests nothing about Debian -- a green row proving nothing. The distro rows therefore
# install numpy FROM THE DISTRO, which links the system BLAS and ships no bundled ILP64
# library, letting discovery genuinely fall through to it. tests/long/test_env_matrix.py asserts
# the resolved path per row so that a regression here fails loudly instead of silently.

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ARG BLAS_SOURCE=wheel
ARG NUMPY_SPEC=numpy
ENV BLAS_SOURCE=${BLAS_SOURCE} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /src

# pyyaml is in every branch, installed with that row's OWN package manager so a distro row
# does not acquire a pip wheel. It is not optional: bigla goes in with --no-deps below, so the
# [test] extra never resolves, and tests/test_matrix_config.py imports yaml UNCONDITIONALLY --
# deliberately, since a consistency check that skips itself is one nobody notices has stopped
# running. Without it every row died in COLLECTION (exit 2) before a single test ran.
#
# The fedora branch installs `openblas-serial64_`, NOT `openblas64` (which is not a package
# name at all -- the first run failed with "No match for argument: openblas64"). Fedora splits
# OpenBLAS by threading model AND by interface, and the trailing underscore is the load-bearing
# part of the name:
#   openblas-serial          LP64                     libopenblas.so.0
#   openblas-serial64        ILP64, BARE symbols      libopenblas64.so.0
#   openblas-serial64_       ILP64, `64_` suffix      libopenblas64_.so.0   <- this row
# Verified against Fedora 44's package data (openblas 0.3.29): both the name and the shipped
# soname. `serial` avoids pulling libgomp and a second thread pool into the container; no
# -devel is needed, since the runtime subpackage ships the versioned soname that _candidates()
# dlopens first. The `_64_` decoration itself is documented but NOT nm-verified -- the row's
# expect_decoration is what settles it, exactly as for the Debian row.
RUN set -eux; \
    case "${BLAS_SOURCE}" in \
      wheel) \
        pip install --no-cache-dir "${NUMPY_SPEC}" scipy pytest pyyaml ;; \
      scipy-openblas64) \
        pip install --no-cache-dir "${NUMPY_SPEC}" scipy pytest pyyaml scipy-openblas64 ;; \
      debian) \
        apt-get update; \
        apt-get install -y --no-install-recommends \
          python3-numpy python3-scipy python3-pytest python3-pip python3-yaml \
          libopenblas64-dev; \
        rm -rf /var/lib/apt/lists/* ;; \
      fedora) \
        dnf -y install python3-numpy python3-scipy python3-pytest python3-pip python3-pyyaml \
          openblas-serial64_; \
        dnf clean all ;; \
      none) \
        apt-get update; \
        apt-get install -y --no-install-recommends \
          python3-numpy python3-pytest python3-pip python3-yaml; \
        rm -rf /var/lib/apt/lists/* ;; \
      mkl) \
        micromamba install -y -n base -c conda-forge \
          python=3.12 numpy scipy pytest pyyaml mkl pip; \
        micromamba clean -ay ;; \
      *) echo "unknown BLAS_SOURCE=${BLAS_SOURCE}" >&2; exit 2 ;; \
    esac

COPY . /src

# mambaorg/micromamba does not put the base environment on PATH for build steps; its custom
# SHELL activates it only when this ARG is set, so without it the pip below is `command not
# found` (exit 127) even though the micromamba install above succeeded. Inert on every other
# base image, where it is simply an unused build arg.
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# --no-deps: the row's numpy is already installed by the branch above, and pip must not be
# allowed to pull a wheel-bundled one over a distro one -- that would silently convert a
# distro row back into the `wheel` row.
RUN python3 -m pip install --no-cache-dir --no-deps --break-system-packages -e /src \
 || python3 -m pip install --no-cache-dir --no-deps -e /src

# Each row emits BOTH: the suite's verdict and the machine-readable record the matrix
# collects. Run as:
#   docker run --rm -e BIGLA_ROW_NAME=... -e BIGLA_EXPECT_PATH_RE=... -v "$PWD/out:/out" IMAGE
#
# `python3`, not `python`: the distro rows install python3-* packages, which provide only
# `python3` on Debian -- `python` there was `/bin/sh: 1: python: not found` (exit 127) before
# the row could reach a single test. Every base here provides `python3`. On micromamba that is
# the base env's, put on PATH at RUN TIME by the image's own entrypoint (the ARG above is
# build-time only and does not carry over).
#
# TWO invocations, never `pytest /src/tests /src/tests/long`. Passing a parent and its child
# together is version-dependent: pytest 9 collects both, but pytest 8 -- what the distro rows
# get from python3-pytest (Debian trixie 8.3.5, Fedora 44 8.4.2) -- keeps only the child and
# silently drops the parent's own tests. Those rows reported a 9-item run and a green tick
# while the entire fast suite never executed. This mirrors the Makefile, which already runs
# `test` and `test-long` as separate targets.
CMD ["/bin/sh", "-c", "\
set -e; \
python3 -m pytest /src/tests -q; \
python3 -m pytest /src/tests/long -q; \
mkdir -p /out; \
python3 -m bigla.diagnose --format=json --env \"${BIGLA_ROW_NAME:-unnamed}\" \
  > \"/out/${BIGLA_ROW_ID:-row}.json\"; \
python3 -m bigla.diagnose --format=row --env \"${BIGLA_ROW_NAME:-unnamed}\"\
"]
