"""Static catalog of Sana family variants consumed by :mod:`.sana`.

This module does not bind components or select an execution strategy.
:func:`sana_recipe` reads :class:`SanaVariant` fields (task, runner, Hub
path, resolution, optional ``mode``) and composes the native recipe.

``runner`` values used by the image / video factories
----------------------------------------------------
- ``image``: text-to-image DiT, flow-match, CFG.
- ``sprint``: few-step SCM, embedded guidance.
- ``controlnet``: HED ControlNet, encoded initialization.
- ``video``: text-to-video, Flow-DPM, ``standard``.
- ``streaming``: video-to-video; ``mode=="long_streaming"`` selects
  ``chunked-kv-cache``, otherwise bidirectional short stays ``standard``.

Do not treat this table as a SHA rewrite surface; pin fields stay as
declared on each :class:`SanaVariant`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from worldfoundry.core.io.paths import package_data_path


@dataclass(frozen=True)
class SanaVariant:
    """Immutable Hub + runtime metadata for one public Sana model id.

    ``task`` / ``runner`` / ``mode`` decide which factories and strategy
    :func:`sana_recipe` selects.  Optional ``default_*`` fields are
    sampling hints for callers; they are not written into ExecutionSpec
    here.  Checkpoint pin fields (revision, SHA-256, size) are optional.
    """

    model_id: str
    display_name: str
    task: str
    runner: str
    config_path: str
    model_path: str
    resolution: str
    repo_id: str
    checkpoint_revision: str | None = None
    checkpoint_sha256: str | None = None
    checkpoint_size_bytes: int | None = None
    diffusers_repo_id: str | None = None
    default_steps: int | None = None
    default_cfg_scale: float | None = None
    default_fps: int | None = None
    default_num_frames: int | None = None
    default_height: int | None = None
    default_width: int | None = None
    mode: str | None = None
    notes: str = ""

    @property
    def artifact_kind(self) -> str:
        """Return ``generated_video`` for video tasks, else ``generated_image``."""

        return "generated_video" if self.task in {"text-to-video", "video-to-video"} else "generated_image"

    @property
    def default_extension(self) -> str:
        """Return ``.mp4`` for video artifacts, else ``.png``."""

        return ".mp4" if self.artifact_kind == "generated_video" else ".png"


def _hf(repo_id: str, filename: str) -> str:
    """Build a repository-relative ``hf://<repo>/checkpoints/<file>`` path."""

    return f"hf://{repo_id}/checkpoints/{filename}"


SANA_VARIANTS: Mapping[str, SanaVariant] = {
    "sana-600m-512px": SanaVariant(
        model_id="sana-600m-512px",
        display_name="Sana 0.6B 512px",
        task="text-to-image",
        runner="image",
        config_path="sana_config/512ms/Sana_600M_img512.yaml",
        model_path=_hf("Efficient-Large-Model/Sana_600M_512px", "Sana_600M_512px_MultiLing.pth"),
        resolution="512px",
        repo_id="Efficient-Large-Model/Sana_600M_512px",
        checkpoint_revision="a189ee0678ba4b806c2091396fa07dedf8946913",
        checkpoint_sha256="cbb880000c9c2594f00e16fff43be67733f0e2e75a4391e4897ffdbe74a19881",
        checkpoint_size_bytes=2371119642,
        diffusers_repo_id="Efficient-Large-Model/Sana_600M_512px_diffusers",
        default_steps=20,
        default_cfg_scale=4.5,
    ),
    "sana-600m-1024px": SanaVariant(
        model_id="sana-600m-1024px",
        display_name="Sana 0.6B 1024px",
        task="text-to-image",
        runner="image",
        config_path="sana_config/1024ms/Sana_600M_img1024.yaml",
        model_path=_hf("Efficient-Large-Model/Sana_600M_1024px", "Sana_600M_1024px_MultiLing.pth"),
        resolution="1024px",
        repo_id="Efficient-Large-Model/Sana_600M_1024px",
        diffusers_repo_id="Efficient-Large-Model/Sana_600M_1024px_diffusers",
        default_steps=20,
        default_cfg_scale=4.5,
    ),
    "sana-1600m-512px": SanaVariant(
        model_id="sana-1600m-512px",
        display_name="Sana 1.6B 512px",
        task="text-to-image",
        runner="image",
        config_path="sana_config/512ms/Sana_1600M_img512.yaml",
        model_path=_hf("Efficient-Large-Model/Sana_1600M_512px", "Sana_1600M_512px.pth"),
        resolution="512px",
        repo_id="Efficient-Large-Model/Sana_1600M_512px",
        diffusers_repo_id="Efficient-Large-Model/Sana_1600M_512px_diffusers",
        default_steps=20,
        default_cfg_scale=4.5,
    ),
    "sana-1600m-512px-multiling": SanaVariant(
        model_id="sana-1600m-512px-multiling",
        display_name="Sana 1.6B 512px MultiLing",
        task="text-to-image",
        runner="image",
        config_path="sana_config/512ms/Sana_1600M_img512.yaml",
        model_path=_hf(
            "Efficient-Large-Model/Sana_1600M_512px_MultiLing",
            "Sana_1600M_512px_MultiLing.pth",
        ),
        resolution="512px",
        repo_id="Efficient-Large-Model/Sana_1600M_512px_MultiLing",
        diffusers_repo_id="Efficient-Large-Model/Sana_1600M_512px_MultiLing_diffusers",
        default_steps=20,
        default_cfg_scale=4.5,
    ),
    "sana-1600m-1024px": SanaVariant(
        model_id="sana-1600m-1024px",
        display_name="Sana 1.6B 1024px",
        task="text-to-image",
        runner="image",
        config_path="sana_config/1024ms/Sana_1600M_img1024.yaml",
        model_path=_hf("Efficient-Large-Model/Sana_1600M_1024px", "Sana_1600M_1024px.pth"),
        resolution="1024px",
        repo_id="Efficient-Large-Model/Sana_1600M_1024px",
        diffusers_repo_id="Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        default_steps=20,
        default_cfg_scale=4.5,
    ),
    "sana-1600m-1024px-multiling": SanaVariant(
        model_id="sana-1600m-1024px-multiling",
        display_name="Sana 1.6B 1024px MultiLing",
        task="text-to-image",
        runner="image",
        config_path="sana_config/1024ms/Sana_1600M_img1024.yaml",
        model_path=_hf(
            "Efficient-Large-Model/Sana_1600M_1024px_MultiLing",
            "Sana_1600M_1024px_MultiLing.pth",
        ),
        resolution="1024px",
        repo_id="Efficient-Large-Model/Sana_1600M_1024px_MultiLing",
        diffusers_repo_id="Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers",
        default_steps=20,
        default_cfg_scale=4.5,
    ),
    "sana-1600m-1024px-bf16": SanaVariant(
        model_id="sana-1600m-1024px-bf16",
        display_name="Sana 1.6B 1024px BF16",
        task="text-to-image",
        runner="image",
        config_path="sana_config/1024ms/Sana_1600M_img1024.yaml",
        model_path=_hf(
            "Efficient-Large-Model/Sana_1600M_1024px_BF16",
            "Sana_1600M_1024px_BF16.pth",
        ),
        resolution="1024px",
        repo_id="Efficient-Large-Model/Sana_1600M_1024px_BF16",
        diffusers_repo_id="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        default_steps=20,
        default_cfg_scale=4.5,
    ),
    "sana-1600m-2k-bf16": SanaVariant(
        model_id="sana-1600m-2k-bf16",
        display_name="Sana 1.6B 2K BF16",
        task="text-to-image",
        runner="image",
        config_path="sana_config/2048ms/Sana_1600M_img2048_bf16.yaml",
        model_path=_hf(
            "Efficient-Large-Model/Sana_1600M_2Kpx_BF16",
            "Sana_1600M_2Kpx_BF16.pth",
        ),
        resolution="2048px",
        repo_id="Efficient-Large-Model/Sana_1600M_2Kpx_BF16",
        diffusers_repo_id="Efficient-Large-Model/Sana_1600M_2Kpx_BF16_diffusers",
        default_steps=20,
        default_cfg_scale=4.5,
    ),
    "sana-1600m-4k-bf16": SanaVariant(
        model_id="sana-1600m-4k-bf16",
        display_name="Sana 1.6B 4K BF16",
        task="text-to-image",
        runner="image",
        config_path="sana_config/4096ms/Sana_1600M_img4096_bf16.yaml",
        model_path=_hf(
            "Efficient-Large-Model/Sana_1600M_4Kpx_BF16",
            "Sana_1600M_4Kpx_BF16.pth",
        ),
        resolution="4096px",
        repo_id="Efficient-Large-Model/Sana_1600M_4Kpx_BF16",
        diffusers_repo_id="Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers",
        default_steps=20,
        default_cfg_scale=4.5,
    ),
    "sana1p5-1600m-1024px": SanaVariant(
        model_id="sana1p5-1600m-1024px",
        display_name="Sana 1.5 1.6B 1024px",
        task="text-to-image",
        runner="image",
        config_path="sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml",
        model_path=_hf("Efficient-Large-Model/SANA1.5_1.6B_1024px", "SANA1.5_1.6B_1024px.pth"),
        resolution="1024px",
        repo_id="Efficient-Large-Model/SANA1.5_1.6B_1024px",
        diffusers_repo_id="Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
        default_steps=20,
        default_cfg_scale=4.5,
    ),
    "sana1p5-4800m-1024px": SanaVariant(
        model_id="sana1p5-4800m-1024px",
        display_name="Sana 1.5 4.8B 1024px",
        task="text-to-image",
        runner="image",
        config_path="sana1-5_config/1024ms/Sana_4800M_1024px_came8bit_grow_constant_allqknorm_bf16_lr2e5.yaml",
        model_path=_hf("Efficient-Large-Model/SANA1.5_4.8B_1024px", "SANA1.5_4.8B_1024px.pth"),
        resolution="1024px",
        repo_id="Efficient-Large-Model/SANA1.5_4.8B_1024px",
        diffusers_repo_id="Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
        default_steps=20,
        default_cfg_scale=4.5,
    ),
    "sana-sprint-600m-1024px": SanaVariant(
        model_id="sana-sprint-600m-1024px",
        display_name="Sana Sprint 0.6B 1024px",
        task="text-to-image",
        runner="sprint",
        config_path="sana_sprint_config/1024ms/SanaSprint_600M_1024px_allqknorm_bf16_scm_ladd.yaml",
        model_path=_hf(
            "Efficient-Large-Model/Sana_Sprint_0.6B_1024px",
            "Sana_Sprint_0.6B_1024px.pth",
        ),
        resolution="1024px",
        repo_id="Efficient-Large-Model/Sana_Sprint_0.6B_1024px",
        checkpoint_revision="268b7e32816200db48e0b0a8939f32397ae6889f",
        checkpoint_sha256="fd1f4497ba127cac3da5cc6832306cc8b554c187ce6f86ab9b7e0064d0f1b503",
        checkpoint_size_bytes=2381713754,
        diffusers_repo_id="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        default_steps=2,
        default_cfg_scale=1.0,
    ),
    "sana-sprint-1600m-1024px": SanaVariant(
        model_id="sana-sprint-1600m-1024px",
        display_name="Sana Sprint 1.6B 1024px",
        task="text-to-image",
        runner="sprint",
        config_path="sana_sprint_config/1024ms/SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml",
        model_path=_hf(
            "Efficient-Large-Model/Sana_Sprint_1.6B_1024px",
            "Sana_Sprint_1.6B_1024px.pth",
        ),
        resolution="1024px",
        repo_id="Efficient-Large-Model/Sana_Sprint_1.6B_1024px",
        checkpoint_revision="9f6866d6962d5d91d213f113e162cb18bffe741c",
        checkpoint_sha256="180d3c8cfb1e85e907c7f02f431900343d5768ee86a2f20a5af56f9e3b7a2862",
        checkpoint_size_bytes=6453060642,
        diffusers_repo_id="Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
        default_steps=2,
        default_cfg_scale=1.0,
    ),
    "sana-controlnet-600m-1024px": SanaVariant(
        model_id="sana-controlnet-600m-1024px",
        display_name="Sana ControlNet 0.6B 1024px",
        task="text-to-image",
        runner="controlnet",
        config_path="sana_controlnet_config/Sana_600M_img1024_controlnet.yaml",
        model_path=_hf(
            "Efficient-Large-Model/Sana_600M_1024px_ControlNet_HED",
            "Sana_600M_1024px_ControlNet_HED.pth",
        ),
        resolution="1024px",
        repo_id="Efficient-Large-Model/Sana_600M_1024px_ControlNet_HED",
        default_steps=20,
        default_cfg_scale=4.5,
    ),
    "sana-controlnet-1600m-1024px-bf16": SanaVariant(
        model_id="sana-controlnet-1600m-1024px-bf16",
        display_name="Sana ControlNet 1.6B 1024px BF16",
        task="text-to-image",
        runner="controlnet",
        config_path="sana_controlnet_config/Sana_1600M_1024px_controlnet_bf16.yaml",
        model_path=_hf(
            "Efficient-Large-Model/Sana_1600M_1024px_BF16_ControlNet_HED",
            "Sana_1600M_1024px_BF16_ControlNet_HED.pth",
        ),
        resolution="1024px",
        repo_id="Efficient-Large-Model/Sana_1600M_1024px_BF16_ControlNet_HED",
        default_steps=20,
        default_cfg_scale=4.5,
    ),
    "sana-video-2b-480p": SanaVariant(
        model_id="sana-video-2b-480p",
        display_name="Sana Video 2B 480p",
        task="text-to-video",
        runner="video",
        config_path="sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml",
        model_path=_hf("Efficient-Large-Model/SANA-Video_2B_480p", "SANA_Video_2B_480p.pth"),
        resolution="480p",
        repo_id="Efficient-Large-Model/SANA-Video_2B_480p",
        diffusers_repo_id="Efficient-Large-Model/Sana-Video_2B_480p_diffusers",
        default_steps=50,
        default_cfg_scale=6.0,
        default_fps=16,
    ),
    "sana-video-2b-720p": SanaVariant(
        model_id="sana-video-2b-720p",
        display_name="Sana Video 2B 720p",
        task="text-to-video",
        runner="video",
        config_path="sana_video_config/Sana_2000M_720px_ltx2vae_AdamW_fsdp.yaml",
        model_path=_hf("Efficient-Large-Model/SANA-Video_2B_720p", "SANA_Video_2B_720p.pth"),
        resolution="720p",
        repo_id="Efficient-Large-Model/SANA-Video_2B_720p",
        diffusers_repo_id="Efficient-Large-Model/SANA-Video_2B_720p_diffusers",
        default_steps=50,
        default_cfg_scale=6.0,
        default_fps=16,
    ),
    "longsana-video-2b-480p": SanaVariant(
        model_id="longsana-video-2b-480p",
        display_name="LongSANA Video 2B 480p",
        task="text-to-video",
        runner="video",
        config_path="sana_video_config/Sana_2000M_480px_adamW_fsdp_longsana.yaml",
        model_path=_hf(
            "Efficient-Large-Model/SANA-Video_2B_480p_LongLive",
            "SANA_Video_2B_480p_LongLive.pth",
        ),
        resolution="480p",
        repo_id="Efficient-Large-Model/SANA-Video_2B_480p_LongLive",
        diffusers_repo_id="Efficient-Large-Model/SANA-Video_2B_480p_LongLive_diffusers",
        default_steps=4,
        default_cfg_scale=1.0,
        default_fps=16,
        notes="Official LongSANA LongLive inference runs through the Sana video CLI with the longsana 480p config.",
    ),
    "sana-streaming-2b-720p": SanaVariant(
        model_id="sana-streaming-2b-720p",
        display_name="SANA-Streaming 2B 720p",
        task="video-to-video",
        runner="streaming",
        config_path="sana_streaming/sana_streaming_2b_720p.yaml",
        model_path="hf://Efficient-Large-Model/SANA-Streaming/dit/sana_streaming_ar.pth",
        resolution="720p",
        repo_id="Efficient-Large-Model/SANA-Streaming",
        default_steps=4,
        default_cfg_scale=1.0,
        default_fps=16,
        default_num_frames=969,
        default_height=704,
        default_width=1280,
        mode="long_streaming",
        notes="Official BF16 minute-length streaming video-to-video editing route.",
    ),
    "sana-streaming-bidirectional-2b-720p": SanaVariant(
        model_id="sana-streaming-bidirectional-2b-720p",
        display_name="SANA-Streaming Bidirectional 2B 720p",
        task="video-to-video",
        runner="streaming",
        config_path="sana_streaming/sana_streaming_bidirectional_2b_720p.yaml",
        model_path=(
            "hf://Efficient-Large-Model/SANA-Streaming_bidirectional/dit/sana_bidirectional_short.pth"
        ),
        resolution="720p",
        repo_id="Efficient-Large-Model/SANA-Streaming_bidirectional",
        default_steps=50,
        default_cfg_scale=6.0,
        default_fps=16,
        default_num_frames=81,
        default_height=704,
        default_width=1280,
        mode="bidirectional_short",
        notes="Official BF16 short-video bidirectional video-to-video editing route.",
    ),
}

SANA_ALIASES: Mapping[str, str] = {
    "sana": "sana-1600m-1024px-bf16",
    "sana-0.6b-512px": "sana-600m-512px",
    "sana-0.6b-1024px": "sana-600m-1024px",
    "sana-1.6b-512px": "sana-1600m-512px",
    "sana-1.6b-1024px": "sana-1600m-1024px",
    "sana-1.6b-1024px-bf16": "sana-1600m-1024px-bf16",
    "sana-controlnet-0.6b-1024px": "sana-controlnet-600m-1024px",
    "sana-controlnet-1.6b-1024px-bf16": "sana-controlnet-1600m-1024px-bf16",
    "sana1.5-1.6b-1024px": "sana1p5-1600m-1024px",
    "sana1.5-4.8b-1024px": "sana1p5-4800m-1024px",
    "sana-sprint-0.6b-1024px": "sana-sprint-600m-1024px",
    "sana-sprint-1.6b-1024px": "sana-sprint-1600m-1024px",
    "sana-video": "sana-video-2b-480p",
    "sana-video-480p": "sana-video-2b-480p",
    "sana-video-720p": "sana-video-2b-720p",
    "longsana-video": "longsana-video-2b-480p",
    "sana-streaming": "sana-streaming-2b-720p",
    "sana-streaming-long": "sana-streaming-2b-720p",
    "sana-streaming-720p": "sana-streaming-2b-720p",
    "sana-streaming-bidirectional": "sana-streaming-bidirectional-2b-720p",
    "sana-streaming-short": "sana-streaming-bidirectional-2b-720p",
}


def normalize_sana_model_id(model_id: str | None) -> str:
    """Map aliases and ``_`` / case to a canonical :data:`SANA_VARIANTS` key.

    ``None`` or empty becomes ``sana``, which aliases to
    ``sana-1600m-1024px-bf16``.  Unknown keys are returned unchanged so
    :func:`get_sana_variant` can raise with the original id.
    """

    raw = (model_id or "sana").strip()
    key = raw.lower().replace("_", "-")
    return SANA_ALIASES.get(key, key)


def get_sana_variant(model_id: str | None) -> SanaVariant:
    """Return the :class:`SanaVariant` for ``model_id`` after alias normalization.

    Raises:
        KeyError: if the normalized id is not in :data:`SANA_VARIANTS`.
    """

    key = normalize_sana_model_id(model_id)
    try:
        return SANA_VARIANTS[key]
    except KeyError as exc:
        known = ", ".join(sorted(SANA_VARIANTS))
        raise KeyError(f"Unknown Sana variant {model_id!r}. Known variants: {known}") from exc


def config_root() -> Path:
    """Return the packaged Sana YAML config directory used by some runners."""

    return package_data_path("models", "runtime", "configs", "sana")


__all__ = [
    "SANA_ALIASES",
    "SANA_VARIANTS",
    "SanaVariant",
    "config_root",
    "get_sana_variant",
    "normalize_sana_model_id",
]
