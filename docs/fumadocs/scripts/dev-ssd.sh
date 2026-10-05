#!/usr/bin/env bash
# Fastest docs dev: keep sources on shared checkout, hot caches on local SSD.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

bash "${SCRIPT_DIR}/ssd-cache.sh" setup

cd "${DOCS_ROOT}"

export WF_DOCS_CACHE_ROOT="${DOCS_ROOT}/tmp"
export WF_DOCS_KEEP_DIST=1
export WF_DOCS_SKIP_MDX=1

node scripts/predev.mjs
exec npx --no-install next dev --hostname 0.0.0.0 --webpack "$@"
