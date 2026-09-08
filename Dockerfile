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

RUN set -eux; \
    case "${BLAS_SOURCE}" in \
      wheel) \
        pip install --no-cache-dir "${NUMPY_SPEC}" scipy pytest ;; \
      scipy-openblas64) \
        pip install --no-cache-dir "${NUMPY_SPEC}" scipy pytest scipy-openblas64 ;; \
      debian) \
        apt-get update; \
        apt-get install -y --no-install-recommends \
          python3-numpy python3-scipy python3-pytest python3-pip libopenblas64-dev; \
        rm -rf /var/lib/apt/lists/* ;; \
      fedora) \
        dnf -y install python3-numpy python3-scipy python3-pytest python3-pip openblas64; \
        dnf clean all ;; \
      none) \
        apt-get update; \
        apt-get install -y --no-install-recommends python3-numpy python3-pytest python3-pip; \
        rm -rf /var/lib/apt/lists/* ;; \
      mkl) \
        micromamba install -y -n base -c conda-forge \
          python=3.12 numpy scipy pytest mkl pip; \
        micromamba clean -ay ;; \
      *) echo "unknown BLAS_SOURCE=${BLAS_SOURCE}" >&2; exit 2 ;; \
    esac

COPY . /src

# --no-deps: the row's numpy is already installed by the branch above, and pip must not be
# allowed to pull a wheel-bundled one over a distro one -- that would silently convert a
# distro row back into the `wheel` row.
RUN pip install --no-cache-dir --no-deps --break-system-packages -e /src \
 || pip install --no-cache-dir --no-deps -e /src

# Each row emits BOTH: the suite's verdict and the machine-readable record the matrix
# collects. Run as:
#   docker run --rm -e BIGLA_ROW_NAME=... -e BIGLA_EXPECT_PATH_RE=... -v "$PWD/out:/out" IMAGE
CMD ["/bin/sh", "-c", "\
set -e; \
python -m pytest /src/tests /src/tests/long -q; \
mkdir -p /out; \
python -m bigla.diagnose --format=json --env \"${BIGLA_ROW_NAME:-unnamed}\" \
  > \"/out/${BIGLA_ROW_ID:-row}.json\"; \
python -m bigla.diagnose --format=row --env \"${BIGLA_ROW_NAME:-unnamed}\"\
"]
