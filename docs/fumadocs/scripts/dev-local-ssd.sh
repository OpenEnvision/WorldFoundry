#!/usr/bin/env bash
# Mirror docs/fumadocs onto local SSD for faster Next.js dev.
# Keeps a minimal WorldFoundry tree under /tmp with worldfoundry/ + scripts/
# symlinked back to the CPFS repo so generate scripts still resolve paths.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FUMADOCS_SRC="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${FUMADOCS_SRC}/../.." && pwd)"
LOCAL_ROOT="${WF_DOCS_LOCAL_ROOT:-/tmp/wf-docs-dev/WorldFoundry}"
LOCAL_FUMADOCS="${LOCAL_ROOT}/docs/fumadocs"

RSYNC_EXCLUDES=(
  --exclude '.next'
  --exclude '.next.*'
  --exclude 'out'
  --exclude 'out.*'
  --exclude 'tmp/worldfoundry-docs-next'
  --exclude 'tmp/worldfoundry-webpack-cache'
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

  # Logos are regenerated on CPFS; symlink avoids stale SSD copies and 404 storms in dev.
  mkdir -p "${LOCAL_FUMADOCS}/public"
  local logo_link="${LOCAL_FUMADOCS}/public/model-logos"
  if [[ -e "${logo_link}" && ! -L "${logo_link}" ]]; then
    rm -rf "${logo_link}"
  fi
  ln -sfn "${FUMADOCS_SRC}/public/model-logos" "${logo_link}"
}

clean_local_caches() {
  echo "Cleaning local Next.js/webpack caches in ${LOCAL_FUMADOCS}"
  rm -rf \
    "${LOCAL_FUMADOCS}/tmp/worldfoundry-docs-next" \
    "${LOCAL_FUMADOCS}/tmp/worldfoundry-webpack-cache" \
    /tmp/worldfoundry-docs-next \
    /tmp/worldfoundry-webpack-cache
}

sync_to_local() {
  echo "Syncing fumadocs to local SSD: ${LOCAL_FUMADOCS}"
  mkdir -p "${LOCAL_FUMADOCS}"
  rsync -a --delete "${RSYNC_EXCLUDES[@]}" "${FUMADOCS_SRC}/" "${LOCAL_FUMADOCS}/"
  sync_repo_links
  (
    cd "${LOCAL_FUMADOCS}"
    export PATH="${WF_DOCS_NODE_BIN:-/mnt/cpfs/yangboxue/visual_generation/juanxi/shell-config/codex/node/bin}:/mnt/cpfs/yangboxue/visual_generation/juanxi/shell-config/codex/npm-global/bin:${PATH}"
    npm run mdx:generate
  )
  echo "Local mirror ready at ${LOCAL_FUMADOCS}"
}

sync_back() {
  echo "Syncing local edits back to CPFS: ${FUMADOCS_SRC}"
  rsync -a "${RSYNC_EXCLUDES[@]}" \
    --exclude 'node_modules' \
    "${LOCAL_FUMADOCS}/" "${FUMADOCS_SRC}/"
  echo "Synced back to ${FUMADOCS_SRC}"
}

run_dev() {
  if [[ ! -d "${LOCAL_FUMADOCS}/node_modules" ]]; then
    sync_to_local
  fi
  cd "${LOCAL_FUMADOCS}"
  export WF_DOCS_FAST_CACHE=1
  echo "Starting dev server from ${LOCAL_FUMADOCS}"
  exec npm run dev:fast
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  sync   Copy docs/fumadocs to local SSD (${LOCAL_FUMADOCS})
  back   Sync local source edits back to CPFS (excludes node_modules)
  clean  Remove local Next.js/webpack dev caches
  dev    Ensure local mirror exists, then run npm run dev:fast

Environment:
  WF_DOCS_LOCAL_ROOT  Override local repo root (default: /tmp/wf-docs-dev/WorldFoundry)
EOF
}

case "${1:-dev}" in
  sync) sync_to_local ;;
  back) sync_back ;;
  clean) clean_local_caches ;;
  dev) run_dev ;;
  -h|--help|help) usage ;;
  *)
    echo "Unknown command: $1" >&2
    usage >&2
    exit 1
    ;;
esac
