ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/worldfoundry-venv

RUN mkdir -p /workspace
WORKDIR /workspace

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        bash \
        bash-completion \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        g++ \
        gcc \
        git \
        git-lfs \
        iproute2 \
        iputils-ping \
        jq \
        less \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libglx0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        nano \
        ninja-build \
        openssh-client \
        pkg-config \
        procps \
        rsync \
        screen \
        tmux \
        unzip \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

# scripts/setup/bootstrap_worldfoundry.sh creates its environments with conda.
RUN curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm -f /tmp/miniforge.sh \
    && /opt/conda/bin/conda clean -afy \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc
ENV PATH=/opt/conda/condabin:${PATH}

# Checkouts bind-mounted from the host are owned by a different user.
RUN git config --global --add safe.directory '*'

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

CMD ["/bin/bash"]
