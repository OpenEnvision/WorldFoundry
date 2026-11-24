#!/usr/bin/env bash
# Fetch the CUTLASS and ThunderKittens header trees that CMakeLists.txt expects
# under include/cutlass and include/tk, pinned to the upstream-tested commits.
# Safe to re-run: a checkout already at the pinned commit is left untouched.
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

fetch_pinned() {
  local dest="$1" url="$2" commit="$3"
  if [[ -d "$dest/.git" ]] && [[ "$(git -C "$dest" rev-parse HEAD 2>/dev/null)" == "$commit" ]]; then
    echo "up to date: ${dest#"$KERNEL_ROOT"/} @ ${commit:0:12}"
    return
  fi
  if [[ -e "$dest" ]]; then
    echo "error: $dest exists but is not a checkout of $commit; remove it and re-run" >&2
    exit 1
  fi
  mkdir -p "$dest"
  git -C "$dest" init -q
  git -C "$dest" remote add origin "$url"
  git -C "$dest" fetch -q --depth 1 origin "$commit"
  git -C "$dest" checkout -q FETCH_HEAD
  echo "fetched: ${dest#"$KERNEL_ROOT"/} @ ${commit:0:12}"
}

# CUTLASS 4.3.0 (BSD-3-Clause)
fetch_pinned "$KERNEL_ROOT/include/cutlass" \
  https://github.com/NVIDIA/cutlass.git e67e63c331d6e4b729047c95cf6b92c8454cba89
# ThunderKittens (MIT)
fetch_pinned "$KERNEL_ROOT/include/tk" \
  https://github.com/HazyResearch/ThunderKittens.git 6c27e28c8115d1839d9eeeb530913c184a75fc87
