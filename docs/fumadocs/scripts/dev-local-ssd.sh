#!/usr/bin/env bash
# Optional full mirror of docs/fumadocs onto local SSD (node_modules + sources).
# Prefer dev:ssd (scripts/dev-ssd.sh) for fastest startup; use dev:local when
# node_modules I/O on shared storage is also a bottleneck.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FUMADOCS_SRC="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${FUMADOCS_SRC}/../.." && pwd)"
DOCS_MIRROR_ID="$(node -e 'process.stdout.write(require("node:crypto").createHash("sha256").update(require("node:fs").realpathSync(process.argv[1])).digest("hex").slice(0,16))' "${FUMADOCS_SRC}")"
LOCAL_ROOT="${WF_DOCS_LOCAL_ROOT:-/tmp/wf-docs-dev/${DOCS_MIRROR_ID}/WorldFoundry}"
LOCAL_FUMADOCS="${LOCAL_ROOT}/docs/fumadocs"
MIRROR_STAMP="${LOCAL_FUMADOCS}/.mirror-ready"

RSYNC_EXCLUDES=(
  --exclude '.next'
  --exclude '.next.*'
  --exclude 'out'
  --exclude 'out.*'
  --exclude 'node_modules'
  --exclude 'tmp/'
  --exclude '.source/'
)

sync_repo_links() {
  mkdir -p "${LOCAL_ROOT}/docs"
  for name in worldfoundry scripts; do
    local target="${LOCAL_ROOT}/${name}"
    if [[ -e "${target}" && ! -L "${target}" ]]; then
      echo "Refusing to replace non-symlink: ${target}" >&2
      exit 1
    fi
    ln -sfn "${REPO_ROOT}/${name}" "${target}"
  done

  mkdir -p "${LOCAL_FUMADOCS}/public"
  local logo_link="${LOCAL_FUMADOCS}/public/org-logos"
  if [[ -e "${logo_link}" && ! -L "${logo_link}" ]]; then
    rm -rf "${logo_link}"
  fi
  ln -sfn "${FUMADOCS_SRC}/public/org-logos" "${logo_link}"
}

clean_local_caches() {
  echo "Cleaning local Next.js/webpack dev caches"
  if [[ -d "${LOCAL_FUMADOCS}" ]]; then
    WF_DOCS_ROOT="${LOCAL_FUMADOCS}" bash "${SCRIPT_DIR}/ssd-cache.sh" clean --purge
  fi
  rm -rf \
    "${LOCAL_FUMADOCS}/tmp" \
    "${LOCAL_FUMADOCS}/.source"
}

mirror_ready() {
  [[ -f "${MIRROR_STAMP}" ]] \
    && [[ -d "${LOCAL_FUMADOCS}/node_modules/next" ]] \
    && [[ -d "${LOCAL_FUMADOCS}" ]]
}

ensure_npm() {
  if [[ -n "${WF_DOCS_NODE_BIN:-}" ]]; then
    export PATH="${WF_DOCS_NODE_BIN}:${PATH}"
  fi
  if [[ -n "${WF_DOCS_NPM_BIN:-}" ]]; then
    export PATH="${WF_DOCS_NPM_BIN}:${PATH}"
  fi
  if ! command -v npm >/dev/null 2>&1; then
    echo "npm is required; install Node.js or set WF_DOCS_NODE_BIN/WF_DOCS_NPM_BIN" >&2
    exit 1
  fi
}

run_predev() {
  (
    cd "${LOCAL_FUMADOCS}"
    ensure_npm
    export WF_DOCS_SKIP_MDX=1
    node scripts/predev.mjs
  )
}

sync_to_local() {
  local mode="${1:-full}"
  echo "Syncing fumadocs to local SSD (${mode}): ${LOCAL_FUMADOCS}"
  mkdir -p "${LOCAL_FUMADOCS}"
  if [[ "${mode}" == "incremental" ]]; then
    rsync -a --delete "${RSYNC_EXCLUDES[@]}" "${FUMADOCS_SRC}/" "${LOCAL_FUMADOCS}/"
  else
    rsync -a --delete "${RSYNC_EXCLUDES[@]}" "${FUMADOCS_SRC}/" "${LOCAL_FUMADOCS}/"
  fi
  sync_repo_links

  (
    cd "${LOCAL_FUMADOCS}"
    ensure_npm
    if [[ ! -d node_modules ]]; then
      echo "Installing docs dependencies on local SSD"
      npm ci --prefer-offline --no-audit --no-fund
    fi
    export WF_DOCS_SKIP_MDX=1
    node scripts/predev.mjs
  )

  touch "${MIRROR_STAMP}"
  echo "Local mirror ready at ${LOCAL_FUMADOCS}"
}

sync_back() {
  echo "Syncing local edits back to shared checkout: ${FUMADOCS_SRC}"
  rsync -a "${RSYNC_EXCLUDES[@]}" \
    "${LOCAL_FUMADOCS}/" "${FUMADOCS_SRC}/"
  echo "Synced back to ${FUMADOCS_SRC}"
}

run_dev() {
  if mirror_ready; then
    echo "Reusing local SSD mirror at ${LOCAL_FUMADOCS} (run '$0 sync' to force refresh)"
    sync_repo_links
    rsync -a "${RSYNC_EXCLUDES[@]}" "${FUMADOCS_SRC}/" "${LOCAL_FUMADOCS}/"
    run_predev
  else
    sync_to_local full
  fi

  WF_DOCS_ROOT="${LOCAL_FUMADOCS}" bash "${SCRIPT_DIR}/ssd-cache.sh" setup

  cd "${LOCAL_FUMADOCS}"
  export WF_DOCS_CACHE_ROOT="${LOCAL_FUMADOCS}/tmp"
  export WF_DOCS_KEEP_DIST=1
  export WF_DOCS_SKIP_MDX=1
  echo "Starting dev server from ${LOCAL_FUMADOCS} (WF_DOCS_CACHE_ROOT=${WF_DOCS_CACHE_ROOT})"
  exec npx --no-install next dev --hostname 0.0.0.0 --webpack "$@"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  sync   Copy docs/fumadocs to local SSD (${LOCAL_FUMADOCS})
  back   Sync local source edits back to the shared checkout (excludes node_modules)
  clean  Remove local dev caches and mirror stamp
  dev    Run dev server from local mirror (default; reuses mirror when present)

Fastest path on shared storage: npm run dev:ssd  (no full mirror; symlinks caches only)

Environment:
  WF_DOCS_LOCAL_ROOT   Override local repo root (default: /tmp/wf-docs-dev/<checkout-id>/WorldFoundry)
  WF_DOCS_NODE_BIN     Optional directory containing node/npm executables
  WF_DOCS_NPM_BIN      Optional directory containing global npm executables
  WF_DOCS_SSD_ROOT     SSD cache root for ssd-cache.sh (default: ~/.cache/worldfoundry-docs)
EOF
}

case "${1:-dev}" in
  sync) sync_to_local full ;;
  back) sync_back ;;
  clean) clean_local_caches; rm -f "${MIRROR_STAMP}" ;;
  dev) shift; run_dev "$@" ;;
  -h|--help|help) usage ;;
  *)
    echo "Unknown command: $1" >&2
    usage >&2
    exit 1
    ;;
esac
