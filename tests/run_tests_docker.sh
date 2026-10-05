#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tests/run_tests_docker.sh [TEST_TARGET...]

Run WorldFoundry tests inside a CUDA Docker container.

With no TEST_TARGET, the script runs the public CPU gate (`make test`).
With TEST_TARGET values, the script runs `python -m pytest -q TEST_TARGET...`.
Use `make:<target>` as the first argument to run a Make target instead.

Examples:
  tests/run_tests_docker.sh
  tests/run_tests_docker.sh tests/eval_core/test_cli_ux.py
  tests/run_tests_docker.sh make:test-eval-core

Environment overrides:
  WORLDFOUNDRY_TEST_IMAGE         Docker image. Default: nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
  WORLDFOUNDRY_DOCKER_GPUS        Docker --gpus value. Default: all. Use none for CPU-only tests.
  WORLDFOUNDRY_DOCKER_EXTRAS      Editable install extras. Default: test
  WORLDFOUNDRY_BENCHMARK_DATA_ROOT Host benchmark data root mounted into the container.
  WORLDFOUNDRY_UV_CACHE_DIR       Host uv cache. Default: ${HOME}/.cache/uv
  WORLDFOUNDRY_HF_CACHE_DIR       Host Hugging Face cache. Default: ${HOME}/.cache/huggingface
  WORLDFOUNDRY_CACHE_DIR          Host WorldFoundry cache. Default: ${HOME}/.cache/worldfoundry
  WORLDFOUNDRY_TRITON_CACHE_DIR   Host Triton cache. Default: ${HOME}/.cache/triton
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not on PATH." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${WORLDFOUNDRY_TEST_IMAGE:-nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04}"
GPU_SELECTOR="${WORLDFOUNDRY_DOCKER_GPUS:-all}"
EXTRAS="${WORLDFOUNDRY_DOCKER_EXTRAS:-test}"

UV_CACHE_HOST="${WORLDFOUNDRY_UV_CACHE_DIR:-${HOME}/.cache/uv}"
HF_CACHE_HOST="${WORLDFOUNDRY_HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
WORLDFOUNDRY_CACHE_HOST="${WORLDFOUNDRY_CACHE_DIR:-${HOME}/.cache/worldfoundry}"
TRITON_CACHE_HOST="${WORLDFOUNDRY_TRITON_CACHE_DIR:-${HOME}/.cache/triton}"
BENCHMARK_DATA_HOST="${WORLDFOUNDRY_BENCHMARK_DATA_ROOT:-${HOME}/.cache/worldfoundry/data/hfd_datasets}"

mkdir -p \
  "${UV_CACHE_HOST}" \
  "${HF_CACHE_HOST}" \
  "${WORLDFOUNDRY_CACHE_HOST}" \
  "${TRITON_CACHE_HOST}" \
  "${BENCHMARK_DATA_HOST}"

DOCKER_ARGS=(
  --rm
  -i
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  --shm-size=16g
  -v "${REPO_ROOT}:/workspace/WorldFoundry"
  -v "${UV_CACHE_HOST}:/root/.cache/uv"
  -v "${HF_CACHE_HOST}:/root/.cache/huggingface"
  -v "${WORLDFOUNDRY_CACHE_HOST}:/root/.cache/worldfoundry"
  -v "${TRITON_CACHE_HOST}:/root/.cache/triton"
  -v "${BENCHMARK_DATA_HOST}:/workspace/worldfoundry-data/hfd_datasets"
  -e HF_HOME=/root/.cache/huggingface
  -e TRITON_CACHE_DIR=/root/.cache/triton
  -e WORLDFOUNDRY_CACHE_DIR=/root/.cache/worldfoundry
  -e WORLDFOUNDRY_BENCHMARK_DATA_ROOT=/workspace/worldfoundry-data/hfd_datasets
  -e UV_LINK_MODE=copy
  -e UV_PROJECT_ENVIRONMENT=/tmp/worldfoundry-venv
  -e WORLDFOUNDRY_DOCKER_EXTRAS="${EXTRAS}"
  -w /workspace/WorldFoundry
)

if [[ "${GPU_SELECTOR}" != "none" ]]; then
  DOCKER_ARGS+=(--gpus "${GPU_SELECTOR}")
fi

if [[ -f "${HOME}/.netrc" ]]; then
  DOCKER_ARGS+=(-v "${HOME}/.netrc:/root/.netrc:ro")
fi

docker run "${DOCKER_ARGS[@]}" "${IMAGE}" bash -s -- "$@" <<'EOF'
set -euo pipefail

if ! command -v python3 >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    python3 python3-dev python3-pip python3-venv \
    bash ca-certificates cmake curl ffmpeg g++ gcc git git-lfs \
    jq libegl1 libgl1 libglib2.0-0 libglx0 libsm6 libxext6 libxrender1 \
    ninja-build openssh-client pkg-config procps rsync unzip
  rm -rf /var/lib/apt/lists/*
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

git config --global --add safe.directory /workspace/WorldFoundry >/dev/null 2>&1 || true

uv venv --clear
. "${UV_PROJECT_ENVIRONMENT}/bin/activate"

INSTALL_TARGET=".[${WORLDFOUNDRY_DOCKER_EXTRAS}]"
uv pip install --upgrade pip
uv pip install -e "${INSTALL_TARGET}" build pytest PyYAML

if [[ "$#" -eq 0 ]]; then
  exec make test
fi

if [[ "$1" == make:* ]]; then
  target="${1#make:}"
  shift
  exec make "${target}" "$@"
fi

exec python -m pytest -q "$@"
EOF
