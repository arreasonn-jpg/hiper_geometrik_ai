# syntax=docker/dockerfile:1
# CKPT-001 CPU fallback/reference environment. The CUDA-ready variant is
# Dockerfile.cuda; both use the same exact Python dependency lock.
FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

WORKDIR /opt/hga

# make is intentionally present so `docker run IMAGE make reproduce` is a
# supported CPU fallback command, not just a documentation example.
RUN apt-get update \
    && apt-get install --no-install-recommends -y make \
    && rm -rf /var/lib/apt/lists/*

# Install frozen runtime dependencies before source to retain Docker layer cache.
COPY requirements-lock.txt ./
RUN python -m pip install --upgrade "pip==24.2" \
    && python -m pip install -r requirements-lock.txt

COPY . ./
RUN chmod +x /opt/hga/docker-entrypoint.sh \
    && python -m pip install --no-build-isolation --no-deps .

# Mount the host artifact directory at /opt/hga/artifacts for `make reproduce`.
# No result is baked into the image layer.
WORKDIR /opt/hga
ENTRYPOINT ["/opt/hga/docker-entrypoint.sh"]
CMD ["reproduce-all"]
