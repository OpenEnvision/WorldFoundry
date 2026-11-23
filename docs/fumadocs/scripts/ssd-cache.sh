#!/usr/bin/env bash
# Idempotently redirect hot cache paths (webpack, node cache) from slow shared
# storage to local SSD while keeping .source/ and tmp/next local so Node ESM
# and Next.js resolve node_modules and relative content paths reliably.
set -euo pipefail

if [[ -n "${WF_DOCS_NODE_BIN:-}" ]]; then
  case ":${PATH}:" in
    *":${WF_DOCS_NODE_BIN}:"*) ;;
    *) export PATH="${WF_DOCS_NODE_BIN}:${PATH}" ;;
  esac
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_ROOT="${WF_DOCS_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
if [[ -n "${WF_DOCS_SSD_ROOT:-}" ]]; then
  CACHE_ROOT="${WF_DOCS_SSD_ROOT}"
else
  DOCS_CACHE_ID="$(node -e 'process.stdout.write(require("node:crypto").createHash("sha256").update(require("node:fs").realpathSync(process.argv[1])).digest("hex").slice(0,16))' "${DOCS_ROOT}")"
  CACHE_ROOT="${XDG_CACHE_HOME:-${HOME}/.cache}/worldfoundry-docs/${DOCS_CACHE_ID}"
fi

restore_local_dir() {
  local rel_path="$1"
  local migrate="${2:-discard}"
  local abs_path="${DOCS_ROOT}/${rel_path}"

  if [[ ! -L "${abs_path}" ]]; then
    return 0
  fi

  local target
  target="$(readlink -f "${abs_path}" || true)"
  rm -f "${abs_path}"

  if [[ "${migrate}" == "migrate" && -n "${target}" && -d "${target}" ]]; then
    echo "Migrating ${rel_path}/ back to local checkout (${abs_path})"
    mkdir -p "${abs_path}"
    rsync -a "${target}/" "${abs_path}/"
  else
    echo "Removed legacy symlink ${rel_path} (kept local in checkout)"
  fi
}

link_dir_to_ssd() {
  local rel_path="$1"
  local ssd_target="$2"
  local migrate="${3:-keep}"
  local abs_path="${DOCS_ROOT}/${rel_path}"

  mkdir -p "$(dirname "${abs_path}")" "${ssd_target}"

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

  # Keep .source/ and tmp/ (including tmp/next) inside the checkout so Node ESM
  # and Next.js resolve fumadocs-mdx and relative ../content/docs paths cleanly.
  restore_local_dir ".source" "migrate"
  restore_local_dir "tmp" "discard"

  mkdir -p \
    "${DOCS_ROOT}/tmp" \
    "${CACHE_ROOT}/webpack" \
    "${CACHE_ROOT}/node-cache"

  # Webpack and tool caches are safe to rebuild; never rsync from slow shared storage.
  link_dir_to_ssd "tmp/webpack" "${CACHE_ROOT}/webpack" "discard"

  if [[ -d node_modules ]]; then
    link_dir_to_ssd "node_modules/.cache" "${CACHE_ROOT}/node-cache" "discard"
  fi

  cat <<EOF
SSD cache ready.
  Root:               ${CACHE_ROOT}
  tmp/webpack         -> ${CACHE_ROOT}/webpack     (webpack cache via WF_DOCS_WEBPACK_CACHE_DIR)
  node_modules/.cache -> ${CACHE_ROOT}/node-cache  (when present)
  .source/, tmp/next  kept local in ${DOCS_ROOT}

Override root: WF_DOCS_SSD_ROOT=/tmp/wf-docs-cache bash scripts/ssd-cache.sh setup
EOF
}

clean() {
  cd "${DOCS_ROOT}"

  for rel in tmp/webpack node_modules/.cache tmp .source; do
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
  for rel in tmp/webpack node_modules/.cache tmp .source; do
    local abs_path="${DOCS_ROOT}/${rel}"
    if [[ -L "${abs_path}" ]]; then
      echo "${rel} -> $(readlink -f "${abs_path}")"
    elif [[ -e "${abs_path}" ]]; then
      echo "${rel} (local in checkout)"
    else
      echo "${rel} (missing)"
    fi
  done
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  setup   Symlink tmp/webpack and node_modules/.cache to local SSD (idempotent)
  status  Show cache symlink targets
  clean   Remove cache symlinks; optional --purge deletes SSD data

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
