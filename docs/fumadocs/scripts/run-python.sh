#!/usr/bin/env bash
# Portable Python launcher for Fumadocs code generation and drift checks.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

if [[ -n "${WF_DOCS_PYTHON:-}" ]]; then
  docs_python="${WF_DOCS_PYTHON}"
elif [[ -n "${PYTHON:-}" ]]; then
  docs_python="${PYTHON}"
elif command -v python3 >/dev/null 2>&1; then
  docs_python="python3"
elif command -v python >/dev/null 2>&1; then
  docs_python="python"
else
  echo "docs/fumadocs: neither python3 nor python found on PATH (set WF_DOCS_PYTHON)." >&2
  exit 127
fi

exec "${docs_python}" "$@"
