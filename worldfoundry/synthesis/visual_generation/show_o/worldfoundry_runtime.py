"""Show-O in-tree runtime for text-to-image generation within the WorldFoundry framework.

This module provides the `ShowORuntime` class, which serves as an interface to the Show-O
text-to-image generation model. It handles model loading, configuration, and execution
for generating images from text prompts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io import file_sha256
from worldfoundry.core.io.paths import hfd_root_path
from worldfoundry.core.io.paths import package_module_root as package_root


def _resolve_hfd_root() -> Path:
    """Resolves and returns the root path for HuggingFace downloads."""
    return hfd_root_path()


# Default paths follow the owner-prefixed layout produced by the checkpoint downloader.
DEFAULT_SHOW_O_MODEL = _resolve_hfd_root() / "showlab--show-o"
# Default path for the VQGAN model used by Show-O.
DEFAULT_SHOW_O_VQ_MODEL = _resolve_hfd_root() / "showlab--magvitv2"
# Default path for the LLM model used for text understanding in Show-O.
DEFAULT_SHOW_O_LLM_MODEL = _resolve_hfd_root() / "microsoft--phi-1_5"


class ShowORuntime:
    """WorldFoundry Show-O text-to-image runtime backed by the in-tree runtime package.

    This class provides an interface to load and run the Show-O model for text-to-image
    generation. It manages model configuration, device placement, and the generation process.
    """

    MODEL_ID = "show-o"
    DISPLAY_NAME = "Show-o"

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        pretrained_model_path: str | Path = DEFAULT_SHOW_O_MODEL,
        vq_model_path: str | Path = DEFAULT_SHOW_O_VQ_MODEL,
        llm_model_path: str | Path = DEFAULT_SHOW_O_LLM_MODEL,
        resolution: int = 256,
        batch_size: int = 1,
        guidance_scale: float = 3.0,
        generation_timesteps: int = 18,
    ) -> None:
        """Initializes the ShowORuntime instance.

        Args:
            model_id: Identifier for the specific Show-O model variant.
            device: The device to run the model on (e.g., "cuda", "cpu").
            pretrained_model_path: Path to the main pretrained Show-O model checkpoint.
            vq_model_path: Path to the VQGAN model checkpoint used for image encoding/decoding.
            llm_model_path: Path to the LLM model checkpoint for text processing.
            resolution: The output image resolution (e.g., 512 for 512x512).
            batch_size: The number of images to generate in a single batch.
            guidance_scale: Classifier-free guidance scale for controlling generation.
            generation_timesteps: Number of timesteps for the diffusion process during generation.
        """
        self.model_id = model_id
        self.model_name = self.DISPLAY_NAME
        self.generation_type = "text_to_image"
        self.device = device
        # Ensure all paths are expanded and converted to strings for consistency.
        self.pretrained_model_path = str(Path(pretrained_model_path).expanduser())
        self.vq_model_path = str(Path(vq_model_path).expanduser())
        self.llm_model_path = str(Path(llm_model_path).expanduser())
        self.resolution = int(resolution)
        self.batch_size = int(batch_size)
        self.guidance_scale = float(guidance_scale)
        self.generation_timesteps = int(generation_timesteps)
        self._runtime: dict[str, Any] | None = None  # Lazily loaded runtime components.

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "ShowORuntime":
        """Creates a ShowORuntime instance from pretrained model settings.

        This factory method allows initializing the runtime with various configurations,
        prioritizing `kwargs` and `options` over default values.

        Args:
            pretrained_model_path: The path to the pretrained model or a dictionary
                                   of options. If a dictionary, it will be merged
                                   into the `options`.
            args: Legacy argument, currently ignored.
            device: The device to run the model on.
            model_id: An optional model identifier.
            **kwargs: Additional keyword arguments to override default parameters.

        Returns:
            An initialized `ShowORuntime` instance.
        """
        del args  # This argument is not used.
        # Initialize options from pretrained_model_path if it's a mapping, otherwise empty.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        # If pretrained_model_path is provided and not a mapping, set it explicitly in options.
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["pretrained_model_path"] = str(pretrained_model_path)
        # Update options with any additional keyword arguments, giving them precedence.
        options.update(kwargs)
        return cls(
            model_id=str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID),
            device=str(device or options.get("device") or "cuda"),
            pretrained_model_path=(
                options.get("pretrained_model_path")
                or options.get("checkpoint_path")  # Fallback for checkpoint path
                or options.get("model_path")  # Fallback for generic model path
                or DEFAULT_SHOW_O_MODEL
            ),
            vq_model_path=options.get("vq_model_path") or options.get("vq_model_name") or DEFAULT_SHOW_O_VQ_MODEL,
            llm_model_path=options.get("llm_model_path") or DEFAULT_SHOW_O_LLM_MODEL,
            resolution=int(options.get("resolution", 256)),
            batch_size=int(options.get("batch_size", 1)),
            guidance_scale=float(options.get("guidance_scale", 3.0)),
            generation_timesteps=int(options.get("generation_timesteps", 18)),
        )

    @staticmethod
    def _runtime_root() -> Path:
        """Returns the root path of the in-tree Show-O runtime package."""
        return package_root("worldfoundry.synthesis.visual_generation.show_o.show_o_runtime")

    @staticmethod
    def _get_vq_model_class(model_type: str):
        """Dynamically imports and returns the VQGAN model class based on its type.

        Args:
            model_type: The string identifier for the VQGAN model type (e.g., "magvitv2").

        Returns:
            The VQGAN model class.

        Raises:
            ValueError: If the specified VQ model type is not supported.
        """
        from models import MAGVITv2

        if model_type == "magvitv2":
            return MAGVITv2
        raise ValueError(f"Show-O VQ model type {model_type!r} is not supported.")

    def _runtime_config(self, **overrides: Any):
        """Generates the OmegaConf configuration for the Show-O runtime.

        This configuration includes model specifics, dataset preprocessing, and training
        parameters, with options for overriding default values.

        Args:
            **overrides: Keyword arguments to override specific configuration parameters.

        Returns:
            An OmegaConf object containing the runtime configuration.
        """
        from omegaconf import OmegaConf

        # Apply overrides for core generation parameters, ensuring correct types.
        resolution = int(overrides.get("resolution", self.resolution))
        batch_size = int(overrides.get("batch_size", self.batch_size))
        guidance_scale = float(overrides.get("guidance_scale", self.guidance_scale))
        generation_timesteps = int(overrides.get("generation_timesteps", self.generation_timesteps))
        return OmegaConf.create(
            {
                "model": {
                    "vq_model": {"type": "magvitv2", "vq_model_name": self.vq_model_path},
                    "showo": {
                        "pretrained_model_path": self.pretrained_model_path,
                        "w_clip_vit": False,
                        "vocab_size": 58498,
                        "llm_vocab_size": 50295,
                        "llm_model_path": self.llm_model_path,
                        "codebook_size": 8192,
                        "num_vq_tokens": (resolution // 16) ** 2,  # Calculate VQ token count based on resolution.
                        "num_new_special_tokens": 10,
                    },
                },
                "dataset": {"preprocessing": {"max_seq_length": int(overrides.get("max_seq_length", 128))}},
                "training": {
                    "batch_size": batch_size,
                    "cond_dropout_prob": float(overrides.get("cond_dropout_prob", 0.1)),
                    "guidance_scale": guidance_scale,
                    "generation_timesteps": generation_timesteps,
                    "generation_temperature": float(overrides.get("generation_temperature", 1.0)),
                    "mask_schedule": str(overrides.get("mask_schedule", "cosine")),
                    "noise_type": str(overrides.get("noise_type", "mask")),
                },
            }
        )

    def _ensure_runtime(self, **overrides: Any) -> dict[str, Any]:
        """Ensures the Show-O runtime components (model, tokenizer, etc.) are loaded.

        This method lazily loads the necessary models and utilities into `_runtime`
        the first time it is called, or if they haven't been loaded yet.
        It also handles dynamic path adjustments for module imports.

        Args:
            **overrides: Configuration overrides to apply when generating the runtime config.

        Returns:
            A dictionary containing the loaded runtime components.
        """
        if self._runtime is not None:
            return self._runtime
        import importlib
        import sys

        import torch
        from transformers import AutoTokenizer

        # Add the runtime root to sys.path to enable importing local modules.
        runtime_root = self._runtime_root()
        runtime_root_str = str(runtime_root)
        if runtime_root_str not in sys.path:
            sys.path.insert(0, runtime_root_str)
        # Explicitly import the module to ensure it's available in the current context.
        importlib.import_module("worldfoundry.synthesis.visual_generation.show_o.show_o_runtime")
        # Now that the path is set, we can import Showo and other local modules.
        from inference_support.prompting_utils import UniversalPrompting
        from models import Showo

        config = self._runtime_config(**overrides)
        # Determine the actual device, defaulting to 'cpu' if 'cuda' is not available.
        device = self.device if str(self.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        # Load the tokenizer from the specified LLM model path.
        tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")
        # Initialize the universal prompting utility with special tokens and config.
        uni_prompting = UniversalPrompting(
            tokenizer,
            max_text_len=config.dataset.preprocessing.max_seq_length,
            special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
            ignore_id=-100,
            cond_dropout_prob=config.training.cond_dropout_prob,
        )
        # Load the VQGAN model and set it to evaluation mode, disabling gradient computation.
        vq_model_class = self._get_vq_model_class(config.model.vq_model.type)
        vq_model = vq_model_class.from_pretrained(config.model.vq_model.vq_model_name).to(device)
        vq_model.requires_grad_(False)
        vq_model.eval()
        # Load the main Show-O model and set it to evaluation mode.
        model = Showo.from_pretrained(
            config.model.showo.pretrained_model_path,
            llm_model_path=config.model.showo.llm_model_path,
            local_files_only=True,
        ).to(device)
        model.eval()
        self.device = device  # Update the instance device to the actual device used.
        # Store all loaded components in the _runtime dictionary for future use.
        self._runtime = {
            "config": config,
            "model": model,
            "tokenizer": tokenizer,
            "uni_prompting": uni_prompting,
            "vq_model": vq_model,
        }
        return self._runtime

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
        """Generates an image from a text prompt using the Show-O model.

        Args:
            prompt: The text prompt to guide image generation.
            images: Input images, not used in t2i mode.
            video: Input video, not used in t2i mode.
            interactions: A sequence specifying the interaction mode. Only 't2i' is supported.
            output_path: Optional path to save the generated image. If None, a default
                         'show_o.png' in the current directory will be used.
            fps: Frames per second, not used in t2i mode.
            **kwargs: Additional keyword arguments to override runtime configuration parameters.

        Returns:
            A dictionary containing generation results, including status, model info,
            and path to the generated artifact.

        Raises:
            ValueError: If the interaction mode is not 't2i' or if input video is provided.
        """
        del images, fps  # These arguments are not used in text-to-image mode.
        mode = str(kwargs.pop("mode", "") or (interactions[0] if interactions else "t2i"))
        # Validate that only text-to-image mode is supported.
        if mode != "t2i":
            raise ValueError("Show-O in-tree wrapper currently supports only t2i mode.")
        # Ensure no video input is provided for t2i generation.
        if video is not None:
            raise ValueError("Show-O t2i generation does not accept input video.")
        import numpy as np
        import torch
        from PIL import Image

        runtime = self._ensure_runtime(**kwargs)  # Ensure runtime components are loaded.
        from inference_support.prompting_utils import create_attention_mask_predict_next
        from models import get_mask_chedule

        config = runtime["config"]
        model = runtime["model"]
        uni_prompting = runtime["uni_prompting"]
        vq_model = runtime["vq_model"]

        # Prepare prompts for batch processing.
        prompts = [prompt] * int(kwargs.get("batch_size", config.training.batch_size))
        # Initialize image tokens with a mask token ID.
        image_tokens = torch.ones(
            (len(prompts), config.model.showo.num_vq_tokens),
            dtype=torch.long,
            device=self.device,
        ) * model.config.mask_token_id
        # Process prompts to get input IDs for conditional generation.
        input_ids, _ = uni_prompting((prompts, image_tokens), "t2i_gen")

        # Prepare unconditional input IDs if guidance scale is applied.
        if config.training.guidance_scale > 0:
            uncond_input_ids, _ = uni_prompting(([""] * len(prompts), image_tokens), "t2i_gen")
            attention_ids = torch.cat([input_ids, uncond_input_ids], dim=0)
        else:
            uncond_input_ids = None
            attention_ids = input_ids

        # Create the attention mask for the model.
        attention_mask = create_attention_mask_predict_next(
            attention_ids,
            pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
            soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
            eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
            rm_pad_in_image=True,
        )
        mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

        with torch.no_grad():
            # Perform text-to-image generation.
            gen_token_ids = model.t2i_generate(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                guidance_scale=config.training.guidance_scale,
                temperature=config.training.get("generation_temperature", 1.0),
                timesteps=config.training.generation_timesteps,
                noise_schedule=mask_schedule,
                noise_type=config.training.get("noise_type", "mask"),
                seq_len=config.model.showo.num_vq_tokens,
                uni_prompting=uni_prompting,
                config=config,
            )
            # Clamp generated tokens to valid codebook indices.
            gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
            # Decode VQ tokens back into images.
            images_tensor = vq_model.decode_code(gen_token_ids)

        # Post-process the generated image tensor: denormalize and convert to uint8 NumPy array.
        images_tensor = torch.clamp((images_tensor + 1.0) / 2.0, min=0.0, max=1.0)  # Scale to [0, 1].
        images_tensor *= 255.0  # Scale to [0, 255].
        images_array = images_tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        # Determine output path and save the generated image.
        target = Path(output_path) if output_path is not None else Path.cwd() / "show_o.png"
        target = target.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(images_array[0]).save(target)

        # Return a dictionary with generation metadata and artifact details.
        return {
            "status": "success",
            "model_id": self.model_id,
            "artifact_kind": "generated_image",
            "artifact_path": str(target),
            "artifact_sha256": file_sha256(target),
            "runtime": "worldfoundry.show_o.in_tree_runtime",
            "backend_quality": "in_tree_runtime",
            "mode": mode,
        }


__all__ = [
    "DEFAULT_SHOW_O_MODEL",
    "DEFAULT_SHOW_O_VQ_MODEL",
    "ShowORuntime",
]
