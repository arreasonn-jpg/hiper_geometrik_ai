# syntax=docker/dockerfile:1
# CKPT-000 CPU reference environment.  The application dependency set is
# intentionally installed from requirements-lock.txt, not from floating ranges.
FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/hga

# Install the frozen runtime dependencies before the source tree so Docker can
# reuse the expensive wheel layer when only project code changes.
COPY requirements-lock.txt ./
RUN python -m pip install --upgrade "pip==24.2" \
    && python -m pip install -r requirements-lock.txt

COPY . ./
# Dependencies were already installed from the lock file.  Installing the
# project with --no-deps prevents a floating dependency range from weakening
# the reference environment.
RUN python -m pip install --no-build-isolation --no-deps .

# Benchmark reports must be written to a bind-mounted /artifacts directory;
# the image itself remains immutable during a run.
WORKDIR /workspace
ENTRYPOINT ["python", "-m", "hga"]
CMD ["research-benchmark"]
