#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

WORLDFOUNDRY_SOURCE_ROOT="$ROOT"

export PYTHONPATH="$WORLDFOUNDRY_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m worldfoundry.cli.inference_variants "$@"
