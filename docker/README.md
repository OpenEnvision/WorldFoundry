# WorldFoundry Docker Image

`docker/Dockerfile` builds a CUDA 12.8 development image for WorldFoundry: Ubuntu 22.04
with cuDNN, the compiler toolchain, FFmpeg, OpenGL/EGL libraries, Miniforge (conda), and
uv. The image carries no WorldFoundry environment, checkpoints, or outputs. You mount your
checkout and a WorldFoundry home directory, then run the standard bootstrap inside the
container once; the environment it creates lives in the mounted home and is reused by every
later container.

## Requirements

- Linux host with an NVIDIA GPU and a driver that supports CUDA 12.8 or newer (R570+).
- Docker Engine with the Buildx plugin (`docker buildx version`).
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  so that `docker run --gpus all` works.
- Disk space for the CUDA base image (about a 6 GB download) plus the unified environment
  and model checkpoints, which together take tens of gigabytes.

## Build the image

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/OpenEnvision/WorldFoundry.git
cd WorldFoundry
bash docker/build_with_docker.sh worldfoundry:base
```

The script builds for `linux/amd64`. On ARM64 hosts such as GH200, add
`--platform linux/arm64`. `bash docker/build_with_docker.sh --help` lists all options.

Check that containers can see the GPUs:

```bash
docker run --rm --gpus all worldfoundry:base nvidia-smi
```

## Start a container

Choose a host directory with plenty of space as the WorldFoundry home. The bootstrap keeps
`conda_envs/`, `huggingface/`, `models/`, `data/`, and `artifacts/` there.

```bash
export WF_HOME=/path/to/worldfoundry-home
mkdir -p "$WF_HOME"

docker run --gpus all -it --rm --name worldfoundry \
  --network host --ipc=host \
  -v "$PWD":/workspace/WorldFoundry \
  -v "$WF_HOME":/root/.cache/worldfoundry \
  -w /workspace/WorldFoundry \
  worldfoundry:base
```

| Option | Purpose |
| --- | --- |
| `--gpus all` | Expose all GPUs. Use `--gpus '"device=0,1"'` for a subset. |
| `--network host` | Studio, its viewers, and WebRTC streams bind the same ports as on the host, so the SSH tunnels in the main README work unchanged. |
| `--ipc=host` | Shared memory for PyTorch data loaders and multi-process inference. |
| `-v "$PWD":/workspace/WorldFoundry` | Your checkout. Edits on the host are visible immediately. |
| `-v "$WF_HOME":/root/.cache/worldfoundry` | The default `WORLDFOUNDRY_HOME`, kept across containers. |

Open another shell in the running container with `docker exec -it worldfoundry bash`.

## Set up the environment (first run)

```bash
bash scripts/setup/bootstrap_worldfoundry.sh
```

The bootstrap creates the unified conda environment in `$WF_HOME/conda_envs`, writes
`tmp/worldfoundry_unified_env.sh` into the checkout, and verifies imports and CUDA. The
first run downloads PyTorch and compiles flash-attn, which takes a while; add
`--skip-flash-attn` to skip the compile. The [Quick Start](../README.md#-quick-start) in
the main README lists the other bootstrap options.

## Every new shell

```bash
source tmp/worldfoundry_unified_env.sh
conda activate "${WORLDFOUNDRY_UNIFIED_ENV_PREFIX}"
wf --help
```

A new container started with the same `docker run` command finds the environment in
`$WF_HOME` and needs only these three lines.

## Run WorldFoundry

The [Usage](../README.md#-usage) section of the main README applies unchanged inside the
container. For example:

```bash
bash scripts/inference/prepare_model_infer.sh wan2.1-t2v-1.3b --download
wf run wan2.1-t2v-1.3b \
  --pipeline.prompt "A paper boat drifting down a forest stream, cinematic lighting" \
  --output-dir tmp/worldfoundry_infer/wan2.1-t2v-1.3b
```

Outputs under the checkout, such as `tmp/worldfoundry_infer/`, appear directly on the host.

Studio starts the same way as on the host:

```bash
bash scripts/workspace/run_workspace.sh --host 127.0.0.1 --port 7870
```

Open http://127.0.0.1:7870/ on the Docker host, or forward the port from your workstation
with `ssh -L 7870:127.0.0.1:7870 user@gpu-host`.

## Checkpoints and credentials

Checkpoint downloads go to the Hugging Face cache in `$WF_HOME` and are reused by later
containers. For gated models, export `HF_TOKEN` on the host and add `-e HF_TOKEN` to
`docker run`.

To reuse weights already on the host, add a read-only mount such as
`-v /path/to/checkpoints:/checkpoints:ro` to `docker run`, then link them into the cache
layout inside the container:

```bash
bash scripts/setup/link_hf_checkpoints.sh \
  --ckpt-dir /checkpoints \
  --hf-hub-cache "${HF_HUB_CACHE}" \
  --default-world
```

Models with a dedicated environment install it the same way as on the host, also into
`$WF_HOME`:

```bash
bash scripts/setup/model_env_install.sh --model <model-id>
```

## Known issues

- Keep the container paths fixed. The conda environments, the `wf` launchers, and
  checkpoint links record `/workspace/WorldFoundry`, `/root/.cache/worldfoundry`, and mount
  points such as `/checkpoints`; mounting them elsewhere later breaks those references.
- The container runs as root, so files it creates in the checkout (for example under
  `tmp/`) are owned by root on the host. Reclaim them from the checkout with
  `sudo chown -R "$(id -u):$(id -g)" .`.
- `tmp/worldfoundry_unified_env.sh` records container paths. If you also bootstrap the same
  checkout directly on the host, give the container its own file with
  `bash scripts/setup/bootstrap_worldfoundry.sh --env-file tmp/worldfoundry_docker_env.sh`
  and source that file instead.
- The image exposes the CUDA compute and utility driver libraries. GPU OpenGL/EGL
  rendering, used by some simulators and 3D tools, also needs the graphics libraries: add
  `-e NVIDIA_DRIVER_CAPABILITIES=all` to `docker run`.
- Embodied benchmarks that launch simulator images (`wf embodied run ... --pull-docker`)
  need the Docker CLI and daemon, which the image does not include. Run them from a host
  installation.
