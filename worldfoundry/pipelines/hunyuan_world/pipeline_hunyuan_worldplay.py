"""Hunyuan Worldplay visual generation pipeline module."""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, List, Mapping, Optional, Sequence

import torch
from PIL import Image

from worldfoundry.core.io import write_video
from worldfoundry.synthesis.visual_generation.hunyuan_world import load_hunyuan_world_runtime_defaults
from worldfoundry.synthesis.visual_generation.hunyuan_world.worldplay_checkpoints import (
    resolve_worldplay_video_model_path,
)

from ...operators.hunyuan_worldplay_operator import HunyuanWorldPlayOperator
from ..pipeline_utils import PipelineABC

if TYPE_CHECKING:
    from ...synthesis.visual_generation.hunyuan_world.hunyuan_worldplay_synthesis import HunyuanWorldPlaySynthesis

logger = logging.getLogger(__name__)


def _initialize_worldplay_parallel_runtime() -> None:
    """Initialize worldplay parallel runtime helper function."""
    # Attempt to retrieve from environment variables as a fallback resolution
    world_size = int(os.getenv("WORLD_SIZE", "1") or "1")
    # Attempt to retrieve from environment variables as a fallback resolution
    local_rank = int(os.getenv("LOCAL_RANK", "0") or "0")
    if torch.cuda.is_available():
        torch.cuda.set_device(min(local_rank, max(torch.cuda.device_count() - 1, 0)))
    if world_size > 1:
        from worldfoundry.core.distributed.model_parallel.sequence_mesh_state import initialize_parallel_state

        initialize_parallel_state(sp=world_size)


class HunyuanWorldPlayPipeline(PipelineABC):
    """Pipeline implementation for HunyuanWorldPlay visual generation."""
    def __init__(
        self,
        *,
        representation_model=None,
        reasoning_model=None,
        synthesis_model: Optional["HunyuanWorldPlaySynthesis"] = None,
        operators=None,
        device: str = 'cuda'
    ):
        """Initialize the pipeline and configure runtime components."""
        super().__init__()
        self.representation_model = representation_model
        self.reasoning_model = reasoning_model
        self.synthesis_model = synthesis_model
        self.operators = operators
        self.device = device
        self._realtime_session = None

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_path: Optional[str] = None,
        required_components=None,
        mode: Optional[str] = None,
        device: Optional[str] = None,
        create_sr_pipeline: Optional[bool] = None,
        force_sparse_attn: Optional[bool] = None,
        transformer_dtype=None,
        enable_offloading: bool = None,
        enable_group_offloading: bool = None,
        overlap_group_offloading: Optional[bool] = None,
        init_infer_state: Optional[bool] = None,
        infer_state_kwargs: Optional[dict] = None,
        forward_speed: Optional[float] = None,
        yaw_speed_deg: Optional[float] = None,
        pitch_speed_deg: Optional[float] = None,
        **kwargs
    ) -> 'HunyuanWorldPlayPipeline':
        """
        从预训练模型加载 Pipeline
        
        Args:
            device: 设备
            create_sr_pipeline: 是否创建超分辨率 pipeline
            force_sparse_attn: 是否强制使用稀疏注意力
            transformer_dtype: transformer 数据类型
            enable_offloading: 是否启用 offloading
            enable_group_offloading: 是否启用 group offloading
            overlap_group_offloading: 是否重叠 group offloading
            **kwargs: 其他参数
            
        Returns:
            HunyuanWorldPlayPipeline: Pipeline 实例
        """
        defaults = load_hunyuan_world_runtime_defaults("worldplay")
        model_options = dict(model_path) if isinstance(model_path, Mapping) else {}
        if model_options:
            model_path = (
                model_options.pop("model_path", None)
                or model_options.pop("checkpoint_path", None)
                or str(defaults["model_path"])
            )
        else:
            model_path = model_path or str(defaults["model_path"])
        if required_components is not None:
            synthesis_path = required_components.get("video_model_path", defaults["video_model_path"])
        else:
            synthesis_path = model_options.pop("video_model_path", str(defaults["video_model_path"]))
        mode = str(mode or model_options.pop("mode", None) or defaults["mode"])
        for key in ("model_id", "variant_id", "profile_id", "runtime_profile", "required_components"):
            model_options.pop(key, None)
        kwargs = {**model_options, **kwargs}
        synthesis_path = resolve_worldplay_video_model_path(synthesis_path, mode)
        device = str(device or defaults["device"])
        create_sr_pipeline = bool(defaults["create_sr_pipeline"] if create_sr_pipeline is None else create_sr_pipeline)
        force_sparse_attn = bool(defaults["force_sparse_attn"] if force_sparse_attn is None else force_sparse_attn)
        overlap_group_offloading = bool(
            defaults["overlap_group_offloading"] if overlap_group_offloading is None else overlap_group_offloading
        )
        init_infer_state = bool(defaults["init_infer_state"] if init_infer_state is None else init_infer_state)
        if transformer_dtype is None:
            transformer_dtype = getattr(torch, str(defaults["transformer_dtype"]))
        enable_offloading = defaults["enable_offloading"] if enable_offloading is None else enable_offloading
        enable_group_offloading = (
            defaults["enable_group_offloading"] if enable_group_offloading is None else enable_group_offloading
        )

        _initialize_worldplay_parallel_runtime()
        if init_infer_state:
            default_infer_state_kwargs = dict(defaults["infer_state_kwargs"])
            if infer_state_kwargs is not None:
                default_infer_state_kwargs.update(infer_state_kwargs)
            cls.initialize_infer_state(**default_infer_state_kwargs)
        from ...synthesis.visual_generation.hunyuan_world.hunyuan_worldplay_synthesis import HunyuanWorldPlaySynthesis

        synthesis_model = HunyuanWorldPlaySynthesis.from_pretrained(
            synthesis_path,
            mode,
            device=device,
            create_sr_pipeline=create_sr_pipeline,
            force_sparse_attn=force_sparse_attn,
            transformer_dtype=transformer_dtype,
            enable_offloading=enable_offloading,
            enable_group_offloading=enable_group_offloading,
            overlap_group_offloading=overlap_group_offloading,
            action_ckpt=model_path,
            **kwargs
        )
        operator_defaults = defaults["operator"]
        forward_speed = float(operator_defaults["forward_speed"] if forward_speed is None else forward_speed)
        yaw_speed_deg = float(operator_defaults["yaw_speed_deg"] if yaw_speed_deg is None else yaw_speed_deg)
        pitch_speed_deg = float(operator_defaults["pitch_speed_deg"] if pitch_speed_deg is None else pitch_speed_deg)
        operators = HunyuanWorldPlayOperator(
            forward_speed=forward_speed,
            yaw_speed_deg=yaw_speed_deg,
            pitch_speed_deg=pitch_speed_deg,
        )
        
        return cls(
            synthesis_model=synthesis_model,
            operators=operators,
            device=device
        )

    def process(self, *, input_, interaction, video_length: int):
        """
        处理输入
        
        Args:
            input_: 输入数据（图片、视频等）
            interaction: 交互信号
            
        Returns:
            处理后的数据
        """
        if self.operators is None:
            raise ValueError("operators must be provided")
        input_ = self.operators.process_perception(input_)
        self.operators.get_interaction(interaction)
        latent_frames = (video_length - 1) // 4 + 1
        operator_condition = self.operators.process_interaction(latent_frames=latent_frames)
        self.operators.delete_last_interaction()
        return {
            "reference_image": input_,
            "operator_condition": operator_condition,
        }

    @staticmethod
    def initialize_infer_state(**kwargs):
        """Initialize infer state for HunyuanWorldPlayPipeline."""
        from worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_worldplay.commons.infer_state import (
            initialize_infer_state,
        )

        return initialize_infer_state(**kwargs)

    @staticmethod
    def save_video(video, path: str, fps: int = 24):
        """Save video for HunyuanWorldPlayPipeline."""
        write_video(video, path, fps=fps)

    @staticmethod
    def _normalize_pose(pose_data: Any) -> Any:
        """Normalize typed-CLI interaction tokens to the official pose string."""
        if isinstance(pose_data, Sequence) and not isinstance(pose_data, (str, bytes, bytearray)):
            return ",".join(str(value) for value in pose_data)
        return pose_data

    def __call__(
        self,
        *,
        prompt: str,
        images: Optional[Image.Image] = None,
        image_path: Optional[str] = None,
        interactions: Optional[str] = None,
        num_frames: int = 125,
        pose: Optional[str] = None,
        aspect_ratio: str = '16:9',
        num_inference_steps: int = 4,
        negative_prompt: str = "",
        seed: int = 1,
        output_type: str = "pt",
        prompt_rewrite: bool = False,
        enable_sr: bool = False,
        sr_num_inference_steps: Optional[int] = None,
        return_pre_sr_video: bool = False,
        few_step: bool = True,
        chunk_latent_frames: int = 4,
        model_type: str = "ar",
        user_height: Optional[int] = None,
        user_width: Optional[int] = None,
        forward_speed: Optional[float] = None,
        yaw_speed_deg: Optional[float] = None,
        pitch_speed_deg: Optional[float] = None,
        output_path: Optional[str | os.PathLike[str]] = None,
        fps: int = 24,
        return_dict: bool = False,
        **kwargs
    ):
        """
        Pipeline 调用入口

        Args:
            prompt: 文本提示
            images: 参考图像（PIL.Image）
            image_path: 参考图像路径（如果 images 未提供则使用）
            pose: 相机轨迹（如 "w-10, right-10, d-11"）
            aspect_ratio: 宽高比
            num_frames: 帧数
            num_inference_steps: 推理步数
            negative_prompt: 负面提示
            seed: 随机种子
            output_type: 输出类型
            prompt_rewrite: 是否重写提示
            enable_sr: 是否启用超分辨率
            sr_num_inference_steps: 超分辨率推理步数
            return_pre_sr_video: 是否返回超分辨率前的视频
            few_step: 是否使用少步推理
            chunk_latent_frames: chunk latent frames
            model_type: 模型类型（"ar" 或 "bi"）
            user_height: 用户指定高度
            user_width: 用户指定宽度
            **kwargs: 其他参数

        Returns:
            HunyuanVideoPipelineOutput: 包含生成的视频帧
        """
        if forward_speed is not None:
            self.operators.forward_speed = forward_speed
        if yaw_speed_deg is not None:
            self.operators.yaw_speed_deg = yaw_speed_deg
        if pitch_speed_deg is not None:
            self.operators.pitch_speed_deg = pitch_speed_deg

        # Handle image input: prefer images, fallback to image_path
        if images is not None:
            input_image = images
        elif image_path is not None:
            try:
                input_image = Image.open(image_path).convert("RGB")
            except Exception as e:
                raise ValueError(f"Cannot load image from image_path: {image_path}") from e
        else:
            raise ValueError("Either images or image_path must be provided")

        video_length = num_frames
        pose_value = self._normalize_pose(interactions if interactions is not None else pose)
        if pose_value is None:
            raise ValueError("pose or interactions must be provided")
        inferred_video_length = self.operators.infer_video_length(pose_value)
        if video_length != inferred_video_length:
            logger.warning(
                "video_length %s != inferred_video_length %s, auto setting", video_length, inferred_video_length
            )
            video_length = inferred_video_length
        processed = self.process(
            input_=input_image,
            interaction=pose_value,
            video_length=video_length,
        )
        operator_condition = processed["operator_condition"]
        
        output = self.synthesis_model(
            enable_sr=enable_sr,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            num_inference_steps=num_inference_steps,
            sr_num_inference_steps=sr_num_inference_steps,
            video_length=video_length,
            negative_prompt=negative_prompt,
            seed=seed,
            output_type=output_type,
            prompt_rewrite=prompt_rewrite,
            return_pre_sr_video=return_pre_sr_video,
            viewmats=operator_condition["viewmats"].unsqueeze(0),
            Ks=operator_condition["Ks"].unsqueeze(0),
            action=operator_condition["action"].unsqueeze(0),
            few_step=few_step,
            chunk_latent_frames=chunk_latent_frames,
            model_type=model_type,
            user_height=user_height,
            user_width=user_width,
            reference_image=processed["reference_image"],
            **kwargs
        )
        
        if output_path is None:
            return output

        videos = output.videos if hasattr(output, "videos") else output
        artifact_path = Path(output_path).expanduser().resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_video(videos, str(artifact_path), fps=fps)
        result = {
            "artifact_path": str(artifact_path),
            "status": "succeeded",
            "fps": int(fps),
            "num_output_frames": int(videos.shape[2]) if torch.is_tensor(videos) and videos.ndim == 5 else None,
        }
        return result if return_dict or output_path is not None else output

    def stream(self, *args, **kwds) -> Generator[torch.Tensor, List[str], None]:
        """
        流式输出
        """
        pass

    def _ensure_realtime_session(self):
        """Create the model-owned resident rollout adapter once."""

        if self._realtime_session is None:
            if self.synthesis_model is None or self.operators is None:
                raise RuntimeError("HY-WorldPlay pipeline is not initialized.")
            from ...synthesis.visual_generation.hunyuan_world.hunyuan_worldplay.realtime import (
                HunyuanWorldPlayRealtimeSession,
            )

            self._realtime_session = HunyuanWorldPlayRealtimeSession(
                self.synthesis_model.runtime,
                self.operators,
            )
        return self._realtime_session

    def prepare_realtime(self) -> dict[str, Any]:
        """Report the released model's native causal playback cadence."""

        session = self._ensure_realtime_session()
        return {"realtime_spec": session.realtime_spec().to_payload()}

    def configure_realtime(
        self,
        images: Any,
        prompt: str = "",
        seed: int = 1,
        num_inference_steps: int = 4,
        negative_prompt: str = "",
        few_step: bool = True,
        user_height: int | None = None,
        user_width: int | None = None,
        flow_shift: float | None = None,
        guidance_scale: float | None = None,
        **_: Any,
    ) -> dict[str, Any] | None:
        """Configure a user-provided seed image without generating a test clip."""

        if isinstance(images, (str, os.PathLike)):
            with Image.open(os.fspath(images)) as source:
                images = source.convert("RGB")
        elif isinstance(images, Sequence) and not isinstance(images, (str, bytes)):
            images = images[0] if images else None
        if not isinstance(images, Image.Image):
            raise ValueError("HY-WorldPlay realtime requires a PIL image or image path.")
        return self._ensure_realtime_session().configure(
            images,
            prompt=prompt,
            seed=seed,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            flow_shift=flow_shift,
            guidance_scale=guidance_scale,
            few_step=few_step,
            user_height=user_height,
            user_width=user_width,
        )

    def stream_realtime(
        self,
        interactions: Sequence[str] | None = None,
        realtime_segments: Sequence[Mapping[str, Any]] | None = None,
        **_: Any,
    ) -> dict[str, Any] | None:
        """Advance the existing rollout by one native four-latent block."""

        return self._ensure_realtime_session().generate(
            interactions=list(interactions or ()),
            control_segments=realtime_segments,
        )

    def realtime_next_output_frames(self) -> int:
        return int(self._ensure_realtime_session().next_output_frames())

    def reset_realtime(self) -> None:
        """Release session caches while retaining loaded model weights."""

        if self._realtime_session is not None:
            self._realtime_session.reset()
