# Inference-only DB-CogACT source retained in-tree.
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from PIL import Image

from .types import (
    ActionOutput,
    GenerationOutput,
    GenSamplingConfig,
    SamplingConfig,
)


class BasePolicy(ABC):
    """
    Exp-level wrapper that standardizes VLA and VLM inference.

    Holds the model + all stateful inference resources (tokenizer, norm_stats,
    input/output pipelines).  Does NOT own the checkpoint loading — that stays
    in InferenceConfig._initialize_inference(); resources are injected via __init__.
    """

    action_mode: str = "unknown"
    state_used: bool = False
    state_required: bool = False
    state_dim: int | None = None
    max_batch_size: int = 1

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        norm_stats: dict,
        input_pipeline: Callable | None = None,
        output_pipeline: Callable | None = None,
        camera_order: list | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.norm_stats = norm_stats
        self.input_pipeline = input_pipeline
        self.output_pipeline = output_pipeline
        self.camera_order = camera_order or ["front"]

    # ── VLA ──────────────────────────────────────────────────────────────────

    @abstractmethod
    def select_action(
        self,
        observation: dict,
        sampling_config: SamplingConfig | None = None,
    ) -> list[ActionOutput]:
        """
        Policy-level VLA inference. Implementations may support single-sample
        or batched observations, up to ``max_batch_size``.

        Single-sample observation:
          observation["image/{slot}"]: PIL Image | np.ndarray [H,W,3] | path str
          observation["prompt"]:       str
          observation["state"]:        np.ndarray [D] | list[float] (optional)

        Batched policy observation, when supported by the concrete policy:
          observation["image/{slot}"]: list[PIL Image | np.ndarray | path str]
          observation["prompt"]:       list[str] or str broadcast to the batch
          observation["state"]:        list[np.ndarray | list[float]] (optional)

        Always returns list[ActionOutput]. Single-sample callers use result[0].
        Batch size is inferred from the first image/* key that is a list.
        The current HTTP /v1/infer route decodes one observation per request;
        batch support here is for direct policy callers and future transport
        schemas.
        """

    def reset(self) -> None:  # noqa: B027 - Optional hook with a no-op default.
        """Reset episode state. Stateless models can leave this as no-op."""

    # ── batch helpers ────────────────────────────────────────────────────────

    def _infer_batch_size(self, observation: dict) -> int:
        """Infer batch size from the first image/* key."""
        for k, v in observation.items():
            if k.startswith("image/"):
                return len(v) if isinstance(v, list) else 1
        return 1

    def _normalize_obs(self, observation: dict, batch_size: int) -> dict:
        """Normalize union-type obs values to lists of length batch_size."""
        out = {}
        for k, v in observation.items():
            if isinstance(v, list):
                if len(v) != batch_size:
                    raise ValueError(f"obs['{k}'] has length {len(v)}, expected {batch_size}")
                out[k] = v
            else:
                out[k] = [v] * batch_size
        return out

    def _collect_images(self, observation: dict) -> list:
        """Collect images from either legacy ``images`` or v1 ``image/N`` keys."""
        if "images" in observation:
            return observation["images"]
        img_keys = sorted(
            (k for k in observation if k.startswith("image/")),
            key=lambda k: int(k.split("/", 1)[1]),
        )
        return [observation[k] for k in img_keys]

    # ── VLM ──────────────────────────────────────────────────────────────────

    def supports_vlm(self) -> bool:
        return False

    def generate(
        self,
        observation: dict,
        sampling: GenSamplingConfig | None = None,
    ) -> GenerationOutput:
        raise NotImplementedError(
            f"{type(self).__name__} does not support VLM generation. Override generate() and set supports_vlm() → True."
        )

    # ── Image helpers (A-family shared) ──────────────────────────────────────

    def _load_images(self, raw_images: list) -> list:
        """Normalize image inputs: open file paths as PIL, pass PIL Images through."""
        return [Image.open(p).convert("RGB") if isinstance(p, str) else p for p in raw_images]

    def _images_to_tensor(self, pil_images: list):
        """Convert PIL images → model input tensor; unsqueeze batch dim for >1 image."""
        t = self.model.process_images(pil_images).to(dtype=self.model.dtype)
        return t if len(pil_images) == 1 else t.unsqueeze(0)

    # ── Introspection ─────────────────────────────────────────────────────────

    def get_capabilities(self) -> dict:
        return {
            "vla": True,
            "vlm": self.supports_vlm(),
            "reset": type(self).reset is not BasePolicy.reset,
            "action_mode": self.action_mode,
            "state": {
                "used": self.state_used,
                "required": self.state_required,
                "dim": self.state_dim,
            },
            "max_batch_size": self.max_batch_size,
        }
