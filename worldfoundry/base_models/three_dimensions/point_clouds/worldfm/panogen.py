"""
Image-to-Panorama generation via HunyuanWorld.

Modified from HunyuanWorld-1.0/demo_panogen.py

WorldFoundry does not import HunyuanWorld architecture code from machine-local
source checkouts at runtime. Panorama generation requires ``hy3dworld`` to be
installed in the selected environment; callers can provide an existing
``panorama_image``/``panorama_path`` to skip this stage.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import torch
from PIL import Image

from worldfoundry.core.io.paths import checkpoint_root_path


def _ensure_torchvision_functional_tensor_alias() -> None:
    """Bridge older Basicsr/GFPGAN imports on newer torchvision releases."""
    if importlib.util.find_spec("torchvision.transforms.functional_tensor") is not None:
        importlib.import_module("torchvision.transforms.functional_tensor")
        return

    if importlib.util.find_spec("torchvision.transforms._functional_tensor") is None:
        return
    functional_tensor = importlib.import_module("torchvision.transforms._functional_tensor")

    sys.modules.setdefault("torchvision.transforms.functional_tensor", functional_tensor)


def _ensure_realesrgan_version_module() -> None:
    """Make editable-install-only metadata importable from a plain checkout."""
    import sys

    if "realesrgan.version" in sys.modules:
        return

    for entry in sys.path:
        base = Path(entry)
        package_dir = base / "realesrgan"
        version_file = package_dir / "version.py"
        version_txt = base / "VERSION"
        if not package_dir.is_dir():
            continue
        if version_file.is_file():
            return
        if not version_txt.is_file():
            continue

        short_version = version_txt.read_text(encoding="utf-8").strip()
        version_info = tuple(int(item) if item.isdigit() else item for item in short_version.split("."))
        module = ModuleType("realesrgan.version")
        module.__file__ = str(version_file)
        module.__package__ = "realesrgan"
        module.__version__ = short_version
        module.__gitsha__ = "unknown"
        module.version_info = version_info
        sys.modules.setdefault("realesrgan.version", module)
        return


# ---------------------------------------------------------------------------
# Import-time compatibility shims.  hy3dworld itself is resolved lazily so this
# module can always be imported for testing / inspection.
# ---------------------------------------------------------------------------
_ensure_torchvision_functional_tensor_alias()
_ensure_realesrgan_version_module()

Image2PanoramaPipelines = None
Perspective = None
FluxFp8GeMMProcessor = None
FluxFp8AttnProcessor2_0 = None
DeepCacheHelper = None


def _checkpoint_root() -> Path:
    return Path(os.environ.get("WORLDFOUNDRY_CHECKPOINT_ROOT") or checkpoint_root_path()).expanduser()


def _first_existing_path(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _configure_local_panogen_model_defaults() -> None:
    ckpt_root = _checkpoint_root()
    hfd_root = ckpt_root / "hfd"
    flux_fill = _first_existing_path(
        ckpt_root / "FLUX.1-Fill-dev",
        hfd_root / "black-forest-labs--FLUX.1-Fill-dev",
    )
    hunyuan_world = _first_existing_path(
        ckpt_root / "HunyuanWorld-1",
        hfd_root / "tencent--HunyuanWorld-1",
    )
    if flux_fill is not None:
        os.environ.setdefault("WORLDFM_FLUX_FILL_PATH", str(flux_fill.resolve()))
        os.environ.setdefault("IN_TREE_FLUX_FILL_PATH", str(flux_fill.resolve()))
    if hunyuan_world is not None:
        os.environ.setdefault("WORLDFM_HUNYUANWORLD_PATH", str(hunyuan_world.resolve()))
        os.environ.setdefault("IN_TREE_HUNYUANWORLD_PATH", str(hunyuan_world.resolve()))


def _ensure_utils3d_alias() -> None:
    if "utils3d" in sys.modules:
        return

    if importlib.util.find_spec("utils3d") is not None:
        importlib.import_module("utils3d")
        return

    from worldfoundry.base_models.three_dimensions.general_3d.eastern_journalist import (
        utils3d as vendored_utils3d,
    )

    sys.modules["utils3d"] = vendored_utils3d


def _import_hy3dworld_modules() -> None:
    global Image2PanoramaPipelines, Perspective
    global FluxFp8GeMMProcessor, FluxFp8AttnProcessor2_0, DeepCacheHelper

    hy3dworld_module = importlib.import_module("hy3dworld")
    image_pipeline = hy3dworld_module.Image2PanoramaPipelines
    perspective = hy3dworld_module.Perspective

    gemm_module = importlib.import_module("hy3dworld.AngelSlim.gemm_quantization_processor")
    attn_module = importlib.import_module("hy3dworld.AngelSlim.attention_quantization_processor")
    cache_module = importlib.import_module("hy3dworld.AngelSlim.cache_helper")

    Image2PanoramaPipelines = image_pipeline
    Perspective = perspective
    FluxFp8GeMMProcessor = gemm_module.FluxFp8GeMMProcessor
    FluxFp8AttnProcessor2_0 = attn_module.FluxFp8AttnProcessor2_0
    DeepCacheHelper = cache_module.DeepCacheHelper


def ensure_hy3dworld(path: str | os.PathLike | None = None) -> None:
    """
    Import hy3dworld from the active Python environment.

    Args:
        path: Deprecated compatibility argument; external HunyuanWorld source
            checkouts are no longer accepted at runtime.
    """
    if Image2PanoramaPipelines is not None:
        return
    if path is not None:
        raise RuntimeError(
            "WorldFM no longer accepts HunyuanWorld source checkout paths at runtime. "
            "Install `hy3dworld` in the selected environment, or pass panorama_image/"
            "panorama_path to skip panorama generation."
        )

    _ensure_torchvision_functional_tensor_alias()
    _ensure_realesrgan_version_module()
    _ensure_utils3d_alias()

    try:
        _import_hy3dworld_modules()
        return
    except ModuleNotFoundError as import_error:
        if not (import_error.name or "").startswith("hy3dworld"):
            raise
        raise ModuleNotFoundError(
            "No module named 'hy3dworld'. Install HunyuanWorld/WorldFM panorama "
            "dependencies in the selected environment, or pass panorama_image/"
            "panorama_path to skip panorama generation. WorldFoundry does not "
            "import HunyuanWorld architecture code from external source checkouts."
        ) from import_error


class Image2PanoramaDemo:
    def __init__(self, args):
        _configure_local_panogen_model_defaults()
        if Image2PanoramaPipelines is None:
            raise ImportError(
                "hy3dworld is not available. Install it in the selected environment "
                "or pass panorama_image/panorama_path to skip panorama generation."
            )

        self.args = args
        self.height, self.width = 960, 1920

        self.THETA = 0
        self.PHI = 0
        self.FOV = 80
        self.guidance_scale = float(os.environ.get("WORLDFM_PANOGEN_GUIDANCE_SCALE", "30"))
        self.num_inference_steps = int(os.environ.get("WORLDFM_PANOGEN_STEPS", "50"))
        self.true_cfg_scale = float(os.environ.get("WORLDFM_PANOGEN_TRUE_CFG_SCALE", "2.0"))
        self.shifting_extend = 0
        self.blend_extend = 6

        self.lora_path = (
            os.environ.get("WORLDFM_HUNYUANWORLD_PATH")
            or os.environ.get("IN_TREE_HUNYUANWORLD_PATH")
            or "tencent/HunyuanWorld-1"
        )
        self.model_path = (
            os.environ.get("WORLDFM_FLUX_FILL_PATH")
            or os.environ.get("IN_TREE_FLUX_FILL_PATH")
            or "black-forest-labs/FLUX.1-Fill-dev"
        )

        self.pipe = Image2PanoramaPipelines.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        )
        self.pipe.load_lora_weights(
            self.lora_path,
            subfolder="HunyuanWorld-PanoDiT-Image",
            weight_name="lora.safetensors",
            torch_dtype=torch.bfloat16,
        )
        self.pipe.fuse_lora()
        self.pipe.unload_lora_weights()
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_tiling()

        self.general_negative_prompt = (
            "human, person, people, messy,"
            "low-quality, blur, noise, low-resolution"
        )
        self.general_positive_prompt = "high-quality,  high-resolution, sharp, clear, 8k"

        if self.args.fp8_attention:
            self.pipe.transformer.set_attn_processor(FluxFp8AttnProcessor2_0())
        if self.args.fp8_gemm:
            FluxFp8GeMMProcessor(self.pipe.transformer)

    def run(
        self,
        prompt: str,
        negative_prompt: str,
        image_path: str,
        seed: int = 42,
        output_path: str = "output_panorama",
        *,
        save_to_disk: bool = True,
    ) -> Image.Image:
        """Generate a panorama from a perspective image.

        Returns the panorama as a PIL Image.  When *save_to_disk* is False the
        image is **not** written to ``output_path/panorama.png``.
        """
        prompt = prompt + ", " + self.general_positive_prompt
        negative_prompt = self.general_negative_prompt + ", " + negative_prompt

        perspective_img = cv2.imread(image_path)
        height_fov, width_fov = perspective_img.shape[:2]
        if width_fov > height_fov:
            ratio = width_fov / height_fov
            w = int((self.FOV / 360) * self.width)
            h = int(w / ratio)
            perspective_img = cv2.resize(
                perspective_img, (w, h), interpolation=cv2.INTER_AREA)
        else:
            ratio = height_fov / width_fov
            h = int((self.FOV / 180) * self.height)
            w = int(h / ratio)
            perspective_img = cv2.resize(
                perspective_img, (w, h), interpolation=cv2.INTER_AREA)

        equ = Perspective(perspective_img, self.FOV,
                          self.THETA, self.PHI, crop_bound=False)
        img, mask = equ.GetEquirec(self.height, self.width)
        mask = cv2.erode(mask.astype(np.uint8), np.ones(
            (3, 3), np.uint8), iterations=5)

        img = img * mask

        mask = mask.astype(np.uint8) * 255
        mask = 255 - mask

        mask = Image.fromarray(mask[:, :, 0])
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        helper = None
        if self.args.cache:
            helper = DeepCacheHelper(
                self.pipe.transformer,
                no_cache_steps=list(range(0, 10)) + list(range(10, 40, 3)) + list(range(40, 50)),
                no_cache_block_id={"single": [38]},
            )
            helper.start_timestep = 0
            helper.enable()

        image = self.pipe(
            prompt=prompt,
            image=img,
            mask_image=mask,
            height=self.height,
            width=self.width,
            negative_prompt=negative_prompt,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(seed),
            blend_extend=self.blend_extend,
            shifting_extend=self.shifting_extend,
            true_cfg_scale=self.true_cfg_scale,
            helper=helper,
        ).images[0]

        if save_to_disk:
            os.makedirs(output_path, exist_ok=True)
            image.save(os.path.join(output_path, "panorama.png"))

        return image
