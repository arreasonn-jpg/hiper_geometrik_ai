#!/bin/sh
# Keep the historical `docker run IMAGE research-benchmark` interface while
# allowing the CKPT-001 literal `docker run IMAGE make reproduce` command.
set -eu

if [ "${1:-}" = "make" ]; then
    exec "$@"
fi
exec python -m hga "$@"
