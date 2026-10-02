"""Inference-only DualCamCtrl runtime integrated into WorldFoundry."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from worldfoundry.core.model_loading.checkpoints import load_weights_only
from worldfoundry.synthesis.visual_generation.dualcamctrl.native_pipeline import (
    WanVideoCameraPipeline,
)
from worldfoundry.core.io import file_sha256
from worldfoundry.core.io.paths import (
    checkpoint_root_path,
    local_model_root_path,
    resolve_local_hf_model_path,
)
from worldfoundry.core.media.codecs.video import write_video
from worldfoundry.core.model_loading import ModelConfig, load_state_dict
from worldfoundry.evaluation.utils import worldfoundry_data_path

MODEL_ID = "dualcamctrl"
DISPLAY_NAME = "DualCamCtrl"
OFFICIAL_SOURCE_REPO = "https://github.com/EnVision-Research/DualCamCtrl"
DEFAULT_BASE_REPO = "alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera"
DEFAULT_TOKENIZER_REPO = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_CLIP_REPO = "Wan-AI/Wan2.1-I2V-14B-480P"
DEFAULT_DUALCAMCTRL_REPO = "FayeHongfeiZhang/DualCamCtrl"
DEFAULT_DUALCAMCTRL_CHECKPOINT = "checkpoints/dualcamctrl_diffusion_transformer.pt"
DEFAULT_CONFIG = worldfoundry_data_path("models", "runtime", "configs", "dualcamctrl", "controlnet_gate_asym_5_10.yaml")
DEFAULT_TEST_CASE_DIR = worldfoundry_data_path("test_cases", "dualcamctrl", "demo_pic")
DEFAULT_NEGATIVE_PROMPT = (
    "Vibrant colors, overexposed, static, blurry details, subtitles, poorly drawn style or artwork, "
    "still image, overall grayish, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, "
    "poorly drawn hands or face, deformed, disfigured, malformed limbs, fused fingers, static image, "
    "messy background, three legs, many people in the background, walking backward."
)


def _looks_like_hf_repo_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value.strip()))

OFFICIAL_DEMO_PROMPTS = {
    "astronaut": (
        "An astronaut walks forward on the Moon, the Earth looming far behind him. In smooth, alternating steps, "
        "his right leg lifts and moves forward while the left leg firmly supports his weight, then the left leg follows "
        "in perfect rhythm. Each step flows naturally into the next, with no sudden jumps or flickering between legs. "
        "His body shifts weight gracefully with each stride, the soft, gentle light casting subtle shadows on his space suit. "
        "Every motion is continuous and coordinated, capturing the serene, low-gravity walk. Real scene in space."
    ),
    "route66": (
        "A straight two-lane highway stretches into the distance, narrowing to a point on the horizon. The asphalt is "
        "slightly worn, with faint cracks and darker patches where tires have passed for years. Yellow double lines run "
        "down the center, flanked by white edge lines, and near the foreground a large white highway emblem is painted "
        "directly on the road surface, its curves and numbers crisp against the dark pavement.On both sides of the road "
        "lies a dry, open landscape-low scrub bushes, sandy soil, and scattered tufts of grass, all in muted browns and grays. "
        "There are a few fence posts and tiny hints of structures far away, but mostly the land feels empty and wide."
    ),
    "seaside": (
        "High aerial view over a British seaside town on a sunny afternoon. Turquoise sea with gentle waves, a long wooden "
        "pier stretching into the water, sandy beach and a bustling promenade with parked cars. Foreground: curved coastal "
        "road and pastel, terraced houses with orange roofs. Midground: beach, pools and small buildings. Background: white "
        "cliffs and rolling hills fading into haze. Few fluffy clouds, bright natural light, crisp visibility. Slow tilt-down "
        "and rightward pan, subtle zoom for parallax. Natural color grade, 4K, 24fps, steady gimbal/drone feel, light wind "
        "ambience and distant seagulls."
    ),
}

OFFICIAL_DEMO_FILES = {
    "astronaut": ("astronaut.png", "astronaut_depth.png", "astronaut.torch"),
    "route66": ("route66.jpg", "route66_depth.jpg", "route66.torch"),
    "seaside": ("seaside.png", "seaside_depth.png", "seaside.torch"),
}


class _Camera:
    """Camera parameter container matching the official DualCamCtrl demo parser."""

    def __init__(self, entry: Sequence[float] | torch.Tensor) -> None:
        values = np.asarray(entry, dtype=np.float32)
        self.fx, self.fy, self.cx, self.cy = values[1:5]
        w2c_mat = np.asarray(values[7:], dtype=np.float32).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4, dtype=np.float32)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4).astype(np.float32)


def _relative_pose(cam_params: Sequence[_Camera]) -> np.ndarray:
    if not cam_params:
        raise ValueError("DualCamCtrl camera path cannot be empty.")
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    target_cam_c2w = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    return np.vstack([pose[np.newaxis, :, :] for pose in ret_poses]).astype(np.float32)


def _custom_meshgrid(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return torch.meshgrid(*args, indexing="ij")


def _ray_condition(
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    height: int,
    width: int,
    *,
    device: str | torch.device,
) -> torch.Tensor:
    batch, views = intrinsics.shape[:2]
    j, i = _custom_meshgrid(
        torch.linspace(0, height - 1, height, device=device, dtype=c2w.dtype),
        torch.linspace(0, width - 1, width, device=device, dtype=c2w.dtype),
    )
    i = i.reshape(1, 1, height * width).expand(batch, views, height * width) + 0.5
    j = j.reshape(1, 1, height * width).expand(batch, views, height * width) + 0.5
    fx, fy, cx, cy = intrinsics.chunk(4, dim=-1)
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs.expand_as(ys)), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)
    rays_o = c2w[..., :3, 3]
    rays_o = rays_o[:, :, None].expand_as(rays_d)
    rays_dxo = torch.cross(rays_o, rays_d, dim=-1)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    return plucker.reshape(batch, c2w.shape[1], height, width, 6)


def _pil_rgb_tensor(image: Image.Image, *, height: int, width: int, channels: int = 3) -> torch.Tensor:
    mode = "L" if channels == 1 else "RGB"
    array = np.asarray(image.convert(mode).resize((width, height)), dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[..., None]
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().unsqueeze(0)
    if tensor.shape[1] == 1 and channels == 3:
        tensor = tensor.repeat(1, 3, 1, 1)
    return tensor


def _pil_depth_tensor(image: Image.Image, *, height: int, width: int) -> torch.Tensor:
    tensor = _pil_rgb_tensor(image, height=height, width=width, channels=1)
    return tensor.repeat(1, 3, 1, 1)


class DualCamCtrlRuntime:
    """DualCamCtrl image/depth/camera-control video generation runtime."""

    MODEL_ID = MODEL_ID
    DISPLAY_NAME = DISPLAY_NAME

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        config_path: str | Path = DEFAULT_CONFIG,
        base_model_repo: str = DEFAULT_BASE_REPO,
        tokenizer_repo: str = DEFAULT_TOKENIZER_REPO,
        dualcamctrl_repo: str = DEFAULT_DUALCAMCTRL_REPO,
        checkpoint_name: str = DEFAULT_DUALCAMCTRL_CHECKPOINT,
        local_model_path: str | Path | None = None,
        download_resource: str = "HuggingFace",
        allow_download: bool = True,
        load_model: bool = True,
        base_model_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        copy_control_weights: bool = True,
        redirect_common_files: bool = True,
        use_usp: bool = False,
    ) -> None:
        self.model_id = model_id
        self.model_name = self.DISPLAY_NAME
        self.generation_type = "image_depth_camera_control_video"
        self.device = device
        self.torch_dtype = torch_dtype
        self.config_path = str(Path(config_path).expanduser())
        self.base_model_repo = base_model_repo
        self.tokenizer_repo = tokenizer_repo
        self.dualcamctrl_repo = dualcamctrl_repo
        self.checkpoint_name = checkpoint_name
        self.local_model_path = str(Path(local_model_path).expanduser()) if local_model_path is not None else str(local_model_root_path())
        self.download_resource = download_resource
        self.allow_download = bool(allow_download)
        self.base_model_path = None if base_model_path is None else str(Path(base_model_path).expanduser())
        self.checkpoint_path = None if checkpoint_path is None else str(Path(checkpoint_path).expanduser())
        self.copy_control_weights = bool(copy_control_weights)
        self.redirect_common_files = bool(redirect_common_files)
        self.use_usp = bool(use_usp)
        self._pipeline: WanVideoCameraPipeline | None = None
        if load_model:
            self._pipeline = self._load_pipeline()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        *,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "DualCamCtrlRuntime":
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            model_ref = str(pretrained_model_path)
            if _looks_like_hf_repo_id(model_ref) and not Path(model_ref).expanduser().exists():
                options["base_model_repo"] = model_ref
            else:
                options["base_model_path"] = model_ref
        options.update(kwargs)
        dtype = options.get("torch_dtype") or options.get("weight_dtype") or torch.bfloat16
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        return cls(
            model_id=str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID),
            device=str(device or options.get("device") or "cuda"),
            torch_dtype=dtype,
            config_path=options.get("config_path") or options.get("model_config") or DEFAULT_CONFIG,
            base_model_repo=str(options.get("base_model_repo") or options.get("base_repo") or DEFAULT_BASE_REPO),
            tokenizer_repo=str(options.get("tokenizer_repo") or DEFAULT_TOKENIZER_REPO),
            dualcamctrl_repo=str(options.get("dualcamctrl_repo") or DEFAULT_DUALCAMCTRL_REPO),
            checkpoint_name=str(options.get("checkpoint_name") or DEFAULT_DUALCAMCTRL_CHECKPOINT),
            local_model_path=options.get("local_model_path"),
            download_resource=str(options.get("download_resource") or "HuggingFace"),
            allow_download=bool(options.get("allow_download", True)),
            load_model=bool(options.get("load_model", True)),
            base_model_path=options.get("base_model_path"),
            checkpoint_path=options.get("checkpoint_path") or options.get("ckpt_path"),
            copy_control_weights=bool(options.get("copy_control_weights", True)),
            redirect_common_files=bool(options.get("redirect_common_files", True)),
            use_usp=bool(options.get("use_usp", False)),
        )

    def _model_config(
        self,
        model_id: str,
        file_pattern: str,
        *,
        preferred_path: str | Path | None = None,
    ) -> ModelConfig:
        roots = []
        if preferred_path is not None:
            preferred = Path(preferred_path).expanduser()
            if preferred.is_file():
                return ModelConfig(path=str(preferred), offload_device=self.device, offload_dtype=self.torch_dtype)
            roots.append(preferred)
        flat_name = model_id.replace("/", "--")
        roots.extend(
            (
                checkpoint_root_path(flat_name),
                checkpoint_root_path("hfd", flat_name),
                checkpoint_root_path(Path(model_id).name),
            )
        )
        try:
            roots.append(resolve_local_hf_model_path(model_id))
        except FileNotFoundError:
            pass
        roots.extend(
            (
                Path(self.local_model_path) / Path(model_id),
                checkpoint_root_path(*Path(model_id).parts),
            )
        )
        for root in dict.fromkeys(roots):
            matches = sorted(root.glob(file_pattern)) if root.is_dir() else []
            if matches:
                paths: str | list[str] = [str(path) for path in matches]
                if len(paths) == 1:
                    paths = paths[0]
                return ModelConfig(path=paths, offload_device=self.device, offload_dtype=self.torch_dtype)
        return ModelConfig(
            model_id=model_id,
            origin_file_pattern=file_pattern,
            download_source=("modelscope" if self.download_resource.lower() == "modelscope" else "huggingface"),
            local_model_path=self.local_model_path,
            skip_download=not self.allow_download,
            offload_device=self.device,
            offload_dtype=self.torch_dtype,
        )

    def _base_model_configs(self) -> list[ModelConfig]:
        return [
            self._model_config(
                self.base_model_repo,
                "diffusion_pytorch_model*.safetensors",
                preferred_path=self.base_model_path,
            ),
            self._model_config(self.tokenizer_repo, "models_t5_umt5-xxl-enc-bf16.pth"),
            self._model_config(self.tokenizer_repo, "Wan2.1_VAE.pth"),
            self._model_config(DEFAULT_CLIP_REPO, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ]

    def _resolve_checkpoint_path(self) -> str:
        if self.checkpoint_path is not None:
            path = Path(self.checkpoint_path).expanduser()
            if path.is_file():
                return str(path)
            raise FileNotFoundError(f"DualCamCtrl checkpoint not found: {path}")
        checkpoint_config = self._model_config(self.dualcamctrl_repo, self.checkpoint_name)
        checkpoint_config.local_model_path = self.local_model_path
        checkpoint_config.skip_download = not self.allow_download
        checkpoint_config.download_if_necessary(use_usp=self.use_usp)
        path = checkpoint_config.path[0] if isinstance(checkpoint_config.path, list) else checkpoint_config.path
        if not path or not Path(path).is_file():
            raise FileNotFoundError(
                f"DualCamCtrl checkpoint {self.checkpoint_name!r} was not found under {self.local_model_path}. "
                "Set checkpoint_path or allow Hugging Face download."
            )
        return str(path)

    @staticmethod
    def _strip_pipe_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
        pipe_items = {key.removeprefix("pipe."): value for key, value in state_dict.items() if key.startswith("pipe.")}
        if pipe_items:
            return pipe_items
        return dict(state_dict)

    def _load_pipeline(self) -> WanVideoCameraPipeline:
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("DualCamCtrl requires CUDA for the official inference path.")
        from worldfoundry.base_models.diffusion_model.models.autoencoders.wan import (
            WanVideoVAE,
            WanVideoVAEStateDictConverter,
        )
        from worldfoundry.base_models.diffusion_model.models.denoisers.wan import (
            WAN21_T2V_1P3B_CONFIG,
            WanModelStateDictConverter,
        )
        from worldfoundry.base_models.diffusion_model.models.encoders.wan import (
            WanImageEncoder,
            WanImageEncoderStateDictConverter,
            WanTextEncoder,
        )
        from worldfoundry.base_models.diffusion_model.models.networks.wan import WanModel
        from worldfoundry.core.model_loading import load_model

        model_configs = self._base_model_configs()
        for model_config in model_configs:
            model_config.local_model_path = self.local_model_path
            model_config.skip_download = not self.allow_download
            model_config.download_if_necessary(use_usp=self.use_usp)
            if not model_config.path:
                raise FileNotFoundError(
                    f"DualCamCtrl checkpoint asset is missing: {model_config.model_id} "
                    f"({model_config.origin_file_pattern}). Set WORLDFOUNDRY_CKPT_DIR "
                    "to the local checkpoint root or allow download."
                )
        tokenizer_config = self._model_config(self.tokenizer_repo, "google/*")
        tokenizer_config.local_model_path = self.local_model_path
        tokenizer_config.skip_download = not self.allow_download
        tokenizer_config.download_if_necessary(use_usp=self.use_usp)

        base_state = load_state_dict(model_configs[0].path, torch_dtype=self.torch_dtype, device="cpu")
        wan_converter = WanModelStateDictConverter()
        if any(".attn1." in name or name.startswith("condition_embedder.") for name in base_state):
            base_state, base_config = wan_converter.from_diffusers(base_state)
        else:
            base_state, base_config = wan_converter.from_civitai(base_state)
        if not base_config:
            base_config = {
                **WAN21_T2V_1P3B_CONFIG,
                "has_image_input": True,
                "in_dim": 32,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
            }
        base_dit = load_model(
            WanModel,
            path=None,
            config=base_config,
            torch_dtype=self.torch_dtype,
            device="cpu",
            state_dict=base_state,
        )
        text_encoder = load_model(
            WanTextEncoder,
            model_configs[1].path,
            torch_dtype=self.torch_dtype,
            device="cpu",
        )
        vae = load_model(
            WanVideoVAE,
            model_configs[2].path,
            torch_dtype=self.torch_dtype,
            device="cpu",
            state_dict_converter=WanVideoVAEStateDictConverter().from_civitai,
        )
        image_encoder = load_model(
            WanImageEncoder,
            model_configs[3].path,
            torch_dtype=torch.float32,
            device="cpu",
            state_dict_converter=WanImageEncoderStateDictConverter().from_civitai,
        )
        tokenizer_path = tokenizer_config.path
        if isinstance(tokenizer_path, list):
            if not tokenizer_path:
                raise FileNotFoundError("DualCamCtrl tokenizer files were not resolved")
            tokenizer_path = tokenizer_path[0]
        tokenizer_path = Path(tokenizer_path)
        if tokenizer_path.is_file():
            tokenizer_path = tokenizer_path.parent

        pipe = WanVideoCameraPipeline.from_components(
            config_path=self.config_path,
            text_encoder=text_encoder,
            base_dit=base_dit,
            vae=vae,
            image_encoder=image_encoder,
            tokenizer_path=str(tokenizer_path),
            copy_control_weights=self.copy_control_weights,
            torch_dtype=self.torch_dtype,
            device=self.device,
            use_usp=self.use_usp,
        )
        checkpoint = self._resolve_checkpoint_path()
        state_dict = load_state_dict(checkpoint, torch_dtype=self.torch_dtype, device="cpu")
        state_dict = self._strip_pipe_prefix(state_dict)
        dit_state = {
            name.removeprefix("dit."): value
            for name, value in state_dict.items()
            if name.startswith("dit.")
        }
        if not dit_state:
            dit_state = state_dict
        load_state = pipe.dit.load_state_dict(dit_state, strict=False)
        if load_state.unexpected_keys:
            raise RuntimeError(f"DualCamCtrl checkpoint has unexpected keys: {load_state.unexpected_keys}")
        pipe.eval()
        pipe.to(self.device)
        return pipe

    def _ensure_pipeline(self) -> WanVideoCameraPipeline:
        if self._pipeline is None:
            self._pipeline = self._load_pipeline()
        return self._pipeline

    @staticmethod
    def demo_assets(demo_name: str) -> tuple[Path, Path, Path, str]:
        key = demo_name.strip().lower().replace(".png", "").replace(".jpg", "")
        if key not in OFFICIAL_DEMO_FILES:
            raise ValueError(f"Unknown DualCamCtrl demo {demo_name!r}. Expected one of {sorted(OFFICIAL_DEMO_FILES)}.")
        image_name, depth_name, camera_name = OFFICIAL_DEMO_FILES[key]
        return (
            DEFAULT_TEST_CASE_DIR / image_name,
            DEFAULT_TEST_CASE_DIR / depth_name,
            DEFAULT_TEST_CASE_DIR / camera_name,
            OFFICIAL_DEMO_PROMPTS[key],
        )

    @staticmethod
    def plucker_embedding(
        camera_file: str | Path,
        *,
        frame_len: int = 61,
        height: int = 320,
        width: int = 480,
        original_height: int = 360,
        original_width: int = 640,
    ) -> torch.Tensor:
        data = load_weights_only(Path(camera_file).expanduser(), map_location="cpu")
        cameras_info = torch.as_tensor(data["cameras"], dtype=torch.float32)
        if cameras_info.shape[1] == 18:
            cameras_info = torch.cat([torch.zeros(cameras_info.shape[0], 1), cameras_info], dim=1)
        if cameras_info.shape[0] < frame_len:
            raise ValueError(f"DualCamCtrl camera path has {cameras_info.shape[0]} frames, but {frame_len} are required.")
        cam_params = [_Camera(cameras_info[index]) for index in range(frame_len)]

        sample_wh_ratio = width / height
        original_wh_ratio = original_width / original_height
        if original_wh_ratio > sample_wh_ratio:
            resized_original_width = width * original_wh_ratio
            for cam_param in cam_params:
                cam_param.fx = resized_original_width * cam_param.fx / height
        else:
            resized_original_height = height / original_wh_ratio
            for cam_param in cam_params:
                cam_param.fy = resized_original_height * cam_param.fy / width

        intrinsics = np.asarray(
            [
                [cam_param.fx * width, cam_param.fy * height, cam_param.cx * width, cam_param.cy * height]
                for cam_param in cam_params
            ],
            dtype=np.float32,
        )
        intrinsics_tensor = torch.as_tensor(intrinsics)[None]
        c2w_tensor = torch.as_tensor(_relative_pose(cam_params))[None]
        plucker = _ray_condition(intrinsics_tensor, c2w_tensor, height, width, device="cpu")[0]
        return plucker.permute(0, 3, 1, 2).contiguous().permute(1, 0, 2, 3).unsqueeze(0)

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[str] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if video is not None:
            raise ValueError("DualCamCtrl consumes an image, a depth image, and a camera path; it does not accept input video.")
        demo_name = str(kwargs.get("demo_name") or "seaside")
        default_image, default_depth, default_camera, default_prompt = self.demo_assets(demo_name)
        image_path = Path(kwargs.get("image_path") or images or default_image).expanduser()
        depth_path = Path(kwargs.get("depth_image") or kwargs.get("depth_path") or default_depth).expanduser()
        camera_path = Path(kwargs.get("camera_path") or kwargs.get("trajectory_file") or (interactions[0] if interactions else default_camera)).expanduser()
        text_prompt = prompt or kwargs.get("caption") or default_prompt

        frame_len = int(kwargs.get("num_frames", kwargs.get("frame_len", 61)))
        height = int(kwargs.get("height", 320))
        width = int(kwargs.get("width", 480))
        num_inference_steps = int(kwargs.get("num_inference_steps", kwargs.get("infer_steps", 50)))
        seed = int(kwargs.get("seed", 42))
        target = Path(output_path) if output_path is not None else Path.cwd() / "dualcamctrl.mp4"
        target = target.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        pipe = self._ensure_pipeline()
        image_tensor = _pil_rgb_tensor(Image.open(image_path), height=height, width=width, channels=3).to(self.device)
        depth_tensor = _pil_depth_tensor(Image.open(depth_path), height=height, width=width).to(self.device)
        plucker = self.plucker_embedding(
            camera_path,
            frame_len=frame_len,
            height=height,
            width=width,
            original_height=int(kwargs.get("original_height", 360)),
            original_width=int(kwargs.get("original_width", 640)),
        )

        with torch.no_grad():
            videos = pipe(
                prompt=[str(text_prompt)],
                negative_prompt=[str(kwargs.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT)],
                batch_size=1,
                input_image=image_tensor,
                input_control=depth_tensor,
                extra_images=None,
                extra_image_frame_index=None,
                plucker_embedding=plucker,
                seed=seed,
                t2v=False,
                height=height,
                width=width,
                tiled=bool(kwargs.get("tiled", True)),
                return_control_latents=bool(kwargs.get("return_control_latents", True)),
                num_inference_steps=num_inference_steps,
                num_frames=frame_len,
                cfg_scale=float(kwargs.get("cfg_scale", 5.0)),
            )

        # StagedDiffusionPipeline.vae_output_to_video returns the frame list
        # directly, not a batch of frame lists.
        frames = videos["images"]
        write_video(frames, target, fps=int(fps or kwargs.get("fps", 10)), quality=int(kwargs.get("quality", 5)))
        return {
            "status": "success",
            "model_id": self.model_id,
            "artifact_kind": "generated_video",
            "artifact_path": str(target),
            "video_sha256": file_sha256(target),
            "runtime": "worldfoundry.dualcamctrl.in_tree_runtime",
            "backend_quality": "official_demo_contract",
            "image_path": str(image_path),
            "depth_path": str(depth_path),
            "camera_path": str(camera_path),
            "seed": seed,
            "num_frames": frame_len,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "official_source": OFFICIAL_SOURCE_REPO,
        }


__all__ = [
    "DEFAULT_BASE_REPO",
    "DEFAULT_CLIP_REPO",
    "DEFAULT_CONFIG",
    "DEFAULT_DUALCAMCTRL_CHECKPOINT",
    "DEFAULT_DUALCAMCTRL_REPO",
    "DEFAULT_NEGATIVE_PROMPT",
    "DEFAULT_TEST_CASE_DIR",
    "DualCamCtrlRuntime",
    "OFFICIAL_DEMO_FILES",
    "OFFICIAL_DEMO_PROMPTS",
    "OFFICIAL_SOURCE_REPO",
]
