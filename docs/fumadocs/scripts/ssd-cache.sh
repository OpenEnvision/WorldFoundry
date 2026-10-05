#!/usr/bin/env bash
# Idempotently redirect hot I/O paths (Next.js, webpack, MDX .source, node cache)
# from slow shared storage to local SSD.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_ROOT="${WF_DOCS_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DOCS_CACHE_ID="$(node -e 'process.stdout.write(require("node:crypto").createHash("sha256").update(require("node:fs").realpathSync(process.argv[1])).digest("hex").slice(0,16))' "${DOCS_ROOT}")"
CACHE_ROOT="${WF_DOCS_SSD_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/worldfoundry-docs/${DOCS_CACHE_ID}}"

link_dir_to_ssd() {
  local rel_path="$1"
  local ssd_target="$2"
  local migrate="${3:-keep}"
  local abs_path="${DOCS_ROOT}/${rel_path}"

  mkdir -p "${ssd_target}"

  if [[ -L "${abs_path}" ]]; then
    local current
    current="$(readlink -f "${abs_path}")"
    if [[ "${current}" == "$(readlink -f "${ssd_target}")" ]]; then
      return 0
    fi
    rm -f "${abs_path}"
  elif [[ -e "${abs_path}" ]]; then
    if [[ "${migrate}" == "migrate" ]]; then
      echo "Migrating ${rel_path}/ to SSD (${ssd_target})"
      rsync -a "${abs_path}/" "${ssd_target}/"
    else
      echo "Discarding ${rel_path}/ on shared storage (rebuilds on SSD)"
    fi
    rm -rf "${abs_path}"
  fi

  ln -sfn "${ssd_target}" "${abs_path}"
}

setup() {
  cd "${DOCS_ROOT}"

  mkdir -p \
    "${CACHE_ROOT}/next" \
    "${CACHE_ROOT}/webpack" \
    "${CACHE_ROOT}/source" \
    "${CACHE_ROOT}/node-cache"

  # Next/webpack caches are safe to rebuild; never rsync from slow shared storage.
  link_dir_to_ssd "tmp" "${CACHE_ROOT}" "discard"
  # MDX output is expensive to regenerate; migrate once when present.
  link_dir_to_ssd ".source" "${CACHE_ROOT}/source" "migrate"

  if [[ -d node_modules ]]; then
    link_dir_to_ssd "node_modules/.cache" "${CACHE_ROOT}/node-cache" "discard"
  fi

  cat <<EOF
SSD cache ready.
  Root:      ${CACHE_ROOT}
  tmp/       -> ${CACHE_ROOT}          (Next.js + webpack via WF_DOCS_CACHE_ROOT)
  .source/   -> ${CACHE_ROOT}/source   (fumadocs-mdx output)
  node_modules/.cache -> ${CACHE_ROOT}/node-cache (when present)

Override root: WF_DOCS_SSD_ROOT=/tmp/wf-docs-cache bash scripts/ssd-cache.sh setup
EOF
}

clean() {
  cd "${DOCS_ROOT}"

  for rel in tmp .source node_modules/.cache; do
    local abs_path="${DOCS_ROOT}/${rel}"
    if [[ -L "${abs_path}" ]]; then
      rm -f "${abs_path}"
      echo "Removed symlink ${rel}"
    fi
  done

  if [[ "${1:-}" == "--purge" ]]; then
    echo "Purging SSD cache at ${CACHE_ROOT}"
    rm -rf "${CACHE_ROOT}"
  else
    echo "Symlinks removed; SSD data kept at ${CACHE_ROOT} (pass --purge to delete)."
  fi
}

status() {
  cd "${DOCS_ROOT}"
  echo "WF_DOCS_SSD_ROOT=${CACHE_ROOT}"
  for rel in tmp .source node_modules/.cache; do
    local abs_path="${DOCS_ROOT}/${rel}"
    if [[ -L "${abs_path}" ]]; then
      echo "${rel} -> $(readlink -f "${abs_path}")"
    elif [[ -e "${abs_path}" ]]; then
      echo "${rel} (on shared storage, not linked)"
    else
      echo "${rel} (missing)"
    fi
  done
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  setup   Symlink tmp/, .source/, node_modules/.cache to local SSD (idempotent)
  status  Show symlink targets
  clean   Remove symlinks; optional --purge deletes SSD data

Environment:
  WF_DOCS_SSD_ROOT   SSD cache root (default: \${XDG_CACHE_HOME:-~/.cache}/worldfoundry-docs/<checkout-id>)
EOF
}

case "${1:-setup}" in
  setup) setup ;;
  status) status ;;
  clean) clean "${2:-}" ;;
  -h|--help|help) usage ;;
  *)
    echo "Unknown command: $1" >&2
    usage >&2
    exit 1
    ;;
esac
