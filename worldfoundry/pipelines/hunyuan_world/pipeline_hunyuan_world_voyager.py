"""
input image and interaction signal output rendering video
load operators, representations, and rendering model
"""
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from PIL import Image

from worldfoundry.core.io import artifact_root_path
from worldfoundry.runtime.env import resolve_ckpt_dir

from ..pipeline_utils import PipelineABC

DEFAULT_HUNYUAN_WORLD_VOYAGER_MOGE1_REPO = "Ruicheng/moge-vitl"

logger = logging.getLogger(__name__)


def _default_output_dir(name: str = "hunyuan_world_voyager") -> str:
    """Resolve a stable default output directory instead of writing to the CWD."""
    return str(artifact_root_path() / name)


if TYPE_CHECKING:
    from worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_voyager.runtime import (
        HunyuanWorldVoyagerRepresentation,
    )

    from ...operators.hunyuan_world_voyager_operator import HunyuanWorldVoyagerOperator
    from ...synthesis.visual_generation.hunyuan_world.hunyuan_world_voyager_synthesis import (
        HunyuanWorldVoyagerSynthesis,
    )


class HunyuanWorldVoyagerPipeline(PipelineABC):
    """Pipeline implementation for HunyuanWorldVoyager visual generation."""
    def __init__(self,
                 operators: Optional["HunyuanWorldVoyagerOperator"] = None,
                 represent_model: Optional["HunyuanWorldVoyagerRepresentation"] = None,
                 rendering_model: Optional["HunyuanWorldVoyagerSynthesis"] = None,
                 rendering_args = None,
                 save_representation_video = False,
                 device: str = 'cuda'):
        """Initialize the pipeline and configure runtime components."""
        super(HunyuanWorldVoyagerPipeline, self).__init__()
        self.operators = operators
        self.represent_model = represent_model
        self.rendering_model = rendering_model
        self.rendering_args = rendering_args
        self.save_representation_video = save_representation_video
        self.device = device

        os.makedirs(self.rendering_args.input_path, exist_ok=True)
    
    @classmethod
    def from_pretrained(cls,
                        model_path: Optional[str | dict[str, Any]] = None,
                        required_components = None,
                        device: str = "cuda",
                        represent_render_dir: Optional[str] = None,
                        save_representation_video: bool = False,
                        **kwargs) -> 'HunyuanWorldVoyagerPipeline':
        """
        Load the complete pipeline from a pretrained model

        Args:
            pretrained_model_name_or_path: Path or name of the main model
            represent_model_path: Path to the representation model; uses default path if None
            model_path: Path to the rendering model; uses default path if None
            represent_render_dir: Directory for rendering output
            device: Device (e.g., 'cuda', 'cpu')
            **kwargs: Additional parameters passed to sub-models

        Returns:
            HunyuanWorldVoyagerPipeline: Initialized pipeline instance
        """
        component_options = dict(required_components or {})
        if isinstance(model_path, dict):
            component_options.update(model_path)
            model_path = (
                component_options.pop("model_path", None)
                or component_options.pop("pretrained_model_path", None)
                or component_options.pop("repo_id", None)
            )
        component_options.update(kwargs)
        component_options = cls._strip_framework_loading_options(component_options)
        if not component_options:
            component_options = {"represent_model_path": DEFAULT_HUNYUAN_WORLD_VOYAGER_MOGE1_REPO}
        # 设置默认路径
        if model_path is None:
            model_path = str(resolve_ckpt_dir() / "HunyuanWorld-Voyager")
        skip_representation_model = bool(component_options.pop("skip_representation_model", False))
        if "represent_model_path" in component_options:
            represent_model_path = component_options.get(
                "represent_model_path",
                DEFAULT_HUNYUAN_WORLD_VOYAGER_MOGE1_REPO,
            )
        else:
            represent_model_path = DEFAULT_HUNYUAN_WORLD_VOYAGER_MOGE1_REPO

        represent_model = None
        if not skip_representation_model:
            logger.info("Loading representation model from %s", represent_model_path)
            from worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_voyager.runtime import (
                HunyuanWorldVoyagerRepresentation,
            )

            representation_kwargs = {
                key: value for key, value in component_options.items() if key != "represent_model_path"
            }
            represent_model = HunyuanWorldVoyagerRepresentation.from_pretrained(
                represent_model_path,
                device=device,
                depth_model_name='moge_v1',
                **representation_kwargs
            )

        logger.info("Loading rendering model from %s", model_path)
        from worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_world_voyager.config import (
            parse_args,
        )

        rendering_args = parse_args(argv=[])
        rendering_args.model_base = model_path
        rendering_args.input_path = represent_render_dir or _default_output_dir(
            "hunyuan_world_voyager/represent_render"
        )

        from ...operators.hunyuan_world_voyager_operator import (
            HunyuanWorldVoyagerOperator,
        )
        from ...synthesis.visual_generation.hunyuan_world.hunyuan_world_voyager_synthesis import (
            HunyuanWorldVoyagerSynthesis,
        )

        rendering_model = HunyuanWorldVoyagerSynthesis.from_pretrained(
            model_path, 
            rendering_args,
            # **{k: v for k, v in kwargs.items() if k in ['cache_dir', 'force_download', 'resume_download']}
        )

        operators = HunyuanWorldVoyagerOperator()

        pipeline = cls(
            operators=operators,
            represent_model=represent_model,
            rendering_model=rendering_model,
            rendering_args=rendering_args,
            save_representation_video=save_representation_video,
            device=device
        )
        
        return pipeline

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        """Read rgb for HunyuanWorldVoyagerPipeline."""
        return np.asarray(Image.open(path).convert("RGB"))

    @staticmethod
    def _read_mask(path: Path) -> np.ndarray:
        """Read mask for HunyuanWorldVoyagerPipeline."""
        return np.asarray(Image.open(path).convert("L"))

    @staticmethod
    def _read_exr(path: Path) -> np.ndarray:
        """Read exr for HunyuanWorldVoyagerPipeline."""
        import pyexr

        return np.asarray(pyexr.read(str(path)).squeeze(), dtype=np.float32)

    def load_condition_dir(self, condition_dir: str | os.PathLike[str], *, num_frames: int) -> dict[str, Any]:
        """Load condition dir for HunyuanWorldVoyagerPipeline."""
        condition_path = Path(condition_dir).expanduser().resolve()
        video_input = condition_path / "video_input"
        ref_image_path = condition_path / "ref_image.png"
        ref_depth_path = condition_path / "ref_depth.exr"
        if not ref_image_path.is_file() or not ref_depth_path.is_file() or not video_input.is_dir():
            raise FileNotFoundError(f"Invalid HunyuanWorld-Voyager condition directory: {condition_path}")

        render_files = sorted(video_input.glob("render_*.png"))
        if len(render_files) < num_frames:
            raise ValueError(
                f"HunyuanWorld-Voyager condition directory has {len(render_files)} render frames, "
                f"but {num_frames} frames were requested: {condition_path}"
            )

        render_list: list[np.ndarray] = []
        mask_list: list[np.ndarray] = []
        depth_list: list[np.ndarray] = []
        for index in range(num_frames):
            render_path = video_input / f"render_{index:04d}.png"
            mask_path = video_input / f"mask_{index:04d}.png"
            depth_path = video_input / f"depth_{index:04d}.exr"
            if not render_path.is_file() or not mask_path.is_file() or not depth_path.is_file():
                raise FileNotFoundError(f"Missing Voyager condition frame {index:04d} under {video_input}")
            render_list.append(self._read_rgb(render_path))
            mask_list.append(self._read_mask(mask_path))
            depth_list.append(self._read_exr(depth_path))

        return {
            "render_list": render_list,
            "mask_list": mask_list,
            "depth_list": depth_list,
            "ref_image": self._read_rgb(ref_image_path),
            "ref_depth": self._read_exr(ref_depth_path),
        }

    def process(self, input_image, num_frames=None, interaction_signal="forward"):
        """Process and normalize input arguments and conditions for inference."""
        from ...operators.hunyuan_world_voyager_operator import camera_list

        # transform input image and interaction signal into hunyuan video input
        if isinstance(input_image, (str, os.PathLike)):
            input_image = Image.open(input_image).convert("RGB")
        input_image, image_tensor = self.operators.process_perception(input_image, self.device)
        if hasattr(input_image, "shape"):
            Height, Width = input_image.shape[:2]
        elif isinstance(input_image, Image.Image):
            Width, Height = input_image.size
        else:
            Height, Width = (256, 256)
        num_frames = num_frames if num_frames is not None else self.rendering_args.video_length

        # check interaction signal and generate intrinsics and extrinsics for the video
        if isinstance(interaction_signal, str):
            # single interaction
            self.operators.get_interaction(interaction_signal)
            intrinsics, extrinsics = self.operators.process_interaction(
                num_frames=1, Width=Width, Height=Height, fx=256, fy=256
            )

            # generate representation (points, colors, depth) based on the input image and camera parameters
            input_data = {
                'image': input_image,
                'image_tensor': image_tensor,
                'intrinsics': intrinsics,
                'extrinsics': extrinsics
            }
            points, colors, depth = self.represent_model.get_representation(input_data)

            # generate intrinsics and extrinsics for the whole video based on the interaction signal
            intrinsics, extrinsics = self.operators.process_interaction(
                num_frames=num_frames, Width=Width//2, Height=Height//2, fx=128, fy=128
            )
            self.operators.delete_last_interaction()

            # rendering the video
            render_list, mask_list, depth_list = self.represent_model.render_video(
                points, colors, extrinsics, intrinsics, height=Height//2, width=Width//2
            )
        elif isinstance(interaction_signal, list):
            # 交互序列 - 多轮逻辑
            # 使用第一个交互生成 representation
            first_interaction = interaction_signal[0]
            self.operators.get_interaction(first_interaction)
            intrinsics_first, extrinsics_first = self.operators.process_interaction(
                num_frames=1, Width=Width, Height=Height, fx=256, fy=256
            )

            input_data = {
                'image': input_image,
                'image_tensor': image_tensor,
                'intrinsics': intrinsics_first,
                'extrinsics': extrinsics_first
            }
            points, colors, depth = self.represent_model.get_representation(input_data)

            # calculate number of frames for each interaction, distribute remaining frames to the first few interactions
            num_interactions = len(interaction_signal)
            num_frames_per_interaction = num_frames // num_interactions
            remaining_frames = num_frames % num_interactions

            all_intrinsics = []
            all_extrinsics = []
            prev_extrinsic = None  # ← utilize last frame's extrinsic for smooth transition between interactions

            for i, interaction in enumerate(interaction_signal):
                frames_for_this = num_frames_per_interaction + (1 if i < remaining_frames else 0)
                self.operators.interaction_history.append(interaction)
                intrinsics, extrinsics = camera_list(
                    num_frames=frames_for_this,
                    type=interaction,
                    Width=Width//2,
                    Height=Height//2,
                    fx=128,
                    fy=128,
                    prev_extrinsic=prev_extrinsic  # ← input last frame's extrinsic for smooth transition
                )
                all_intrinsics.append(intrinsics)
                all_extrinsics.append(extrinsics)
                prev_extrinsic = extrinsics[-1]  # ← save last frame's extrinsic for next interaction

            # concatenate all intrinsics and extrinsics
            intrinsics = np.concatenate(all_intrinsics, axis=0)
            extrinsics = np.concatenate(all_extrinsics, axis=0)

            # rendering the coarse video
            render_list, mask_list, depth_list = self.represent_model.render_video(
                points, colors, extrinsics, intrinsics, height=Height//2, width=Width//2
            )
        else:
            raise ValueError(f"interaction_signal must be a string or list, got {type(interaction_signal)}")

        hunyuan_video_input = self.rendering_model.create_hunyuan_video_input(render_list, mask_list, depth_list,
                                                                            Width=Width, Height=Height)
        if self.save_representation_video:
            self.represent_model.save_representation_video(
                render_list, mask_list, depth_list, self.rendering_args.input_path, separate=True, 
                ref_image=input_image, ref_depth=depth, Width=Width, Height=Height
            )

        return hunyuan_video_input

    def __call__(self,
                 images,
                 interactions="forward",
                 prompt = "",
                 num_frames = None,
                 condition_dir: str | os.PathLike[str] | None = None,
                 output_save_path: Optional[str] = None,
                 i2v_stability=True,
                 seed: int | None = None,
                 infer_steps: int | None = None,
                 flow_shift: float | None = None,
                 embedded_cfg_scale: float | None = None,
                 ulysses_degree: int | None = None,
                 ring_degree: int | None = None,
                 height: int | None = None,
                 width: int | None = None,
                 **kwargs):
        """inference function of the pipeline"""
        output_save_path = output_save_path or _default_output_dir(
            "hunyuan_world_voyager/final_render"
        )
        if seed is not None:
            self.rendering_args.seed = int(seed)
        if infer_steps is not None:
            self.rendering_args.infer_steps = int(infer_steps)
        if flow_shift is not None:
            self.rendering_args.flow_shift = float(flow_shift)
        if embedded_cfg_scale is not None:
            self.rendering_args.embedded_cfg_scale = float(embedded_cfg_scale)
        if ulysses_degree is not None:
            self.rendering_args.ulysses_degree = int(ulysses_degree)
        if ring_degree is not None:
            self.rendering_args.ring_degree = int(ring_degree)
        if height is not None or width is not None:
            current_h, current_w = self.rendering_args.video_size
            self.rendering_args.video_size = (int(height or current_h), int(width or current_w))
        video_length = num_frames if num_frames is not None else self.rendering_args.video_length
        if (video_length - 1) % 4 != 0:
            adjusted = ((video_length - 1) // 4) * 4 + 1
            logger.warning(
                "video_length must be a multiple of 4 plus 1 (i.e., (n*4)+1). "
                "Got %s, automatically adjusted to %s.", video_length, adjusted
            )
            video_length = adjusted

        kwargs.pop("interaction_signal", None)
        if condition_dir is not None:
            hunayuan_video_input = self.load_condition_dir(condition_dir, num_frames=video_length)
        else:
            hunayuan_video_input = self.process(images, num_frames=video_length,
                                                interaction_signal=interactions, **kwargs)
        outputs = self.rendering_model.predict(
            prompt=prompt,
            height=self.rendering_args.video_size[0],
            width=self.rendering_args.video_size[1],
            video_length=video_length,
            seed=self.rendering_args.seed,
            negative_prompt=self.rendering_args.neg_prompt,
            infer_steps=self.rendering_args.infer_steps,
            guidance_scale=self.rendering_args.cfg_scale,
            num_videos_per_prompt=self.rendering_args.num_videos,
            flow_shift=self.rendering_args.flow_shift,
            batch_size=self.rendering_args.batch_size,
            embedded_guidance_scale=self.rendering_args.embedded_cfg_scale,
            i2v_mode=self.rendering_args.i2v_mode,
            i2v_resolution=self.rendering_args.i2v_resolution,
            i2v_image_path=self.rendering_args.i2v_image_path,
            i2v_condition_type=self.rendering_args.i2v_condition_type,
            i2v_stability=i2v_stability,
            ulysses_degree=self.rendering_args.ulysses_degree,
            ring_degree=self.rendering_args.ring_degree,
            ref_image=hunayuan_video_input['ref_image'],
            ref_depth=hunayuan_video_input['ref_depth'],
            render_list=hunayuan_video_input['render_list'],
            depth_list=hunayuan_video_input['depth_list'],
            mask_list=hunayuan_video_input['mask_list'],
        )
        samples = outputs['samples']

        # Save generated videos to disk
        # Only save on the main process in distributed settings
        output_video = None
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            from worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_world_voyager.utils.file_utils import (
                video_output,
            )

            sample = samples[0].unsqueeze(0)
            output_video = video_output(sample, fps=24)
        return output_video
