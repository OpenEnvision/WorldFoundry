#!/usr/bin/env bash
set -euo pipefail

SKIP_BOOTSTRAP=0
SKIP_INSTALL=0

usage() {
  cat <<'USAGE'
Usage: bash scripts/docs/build.sh [--skip-bootstrap] [--skip-install]

Build the WorldFoundry Fumadocs site with type checking.

Options:
  --skip-bootstrap  Reserved for CI compatibility; JavaScript dependencies are still installed.
  --skip-install    Do not run npm ci before checking and building.
  -h, --help        Show this help text.
USAGE
}

while (($#)); do
  case "$1" in
    --skip-bootstrap)
      SKIP_BOOTSTRAP=1
      ;;
    --skip-install)
      SKIP_INSTALL=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCS_ROOT="${REPO_ROOT}/docs/fumadocs"

if [[ "${SKIP_BOOTSTRAP}" == "0" ]]; then
  :
fi

cd "${DOCS_ROOT}"

if [[ "${SKIP_INSTALL}" == "0" ]]; then
  npm ci
fi

npm run types:check
# pretypes:check generates the API metadata; require all checked-in docs data to be current.
npm run api:check
npm run models:check
npm run models:homes:check
npm run benchmarks:check
npm run coverage:check

build_log="$(mktemp -t worldfoundry-docs-build.XXXXXX.log)"
trap 'rm -f "${build_log}"' EXIT

if ! npm run build 2>&1 | tee "${build_log}"; then
  exit 1
fi

if grep -F "Warning: Next.js inferred your workspace root" "${build_log}" >/dev/null; then
  echo "Next.js workspace-root warning is not allowed; set outputFileTracingRoot in docs/fumadocs/next.config.mjs." >&2
  exit 1
fi
