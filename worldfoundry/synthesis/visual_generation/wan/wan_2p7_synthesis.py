"""Client for the Wan2.7 video synthesis API (image/media driven)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._dashscope import RUNTIME_STATUS, DashScopeVideoSynthesis

__all__ = ["RUNTIME_STATUS", "Wan2p7Synthesis"]


class Wan2p7Synthesis(DashScopeVideoSynthesis):
    """Generate videos from media inputs and prompts through the Wan2.7 API."""

    def generate_i2av(
        self,
        media: List[Dict[str, str]],
        input_prompt: str,
        model: str = "wan2.7-i2v",
        resolution: str = "720P",
        duration: int = 5,
        negative_prompt: str = "",
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Submit an image-to-video task.

        Args:
            media: Media inputs, each ``{"type": ..., "url": ...}``.
            input_prompt: Prompt describing the desired video.
            negative_prompt: Prompt describing what to avoid.
            seed: Random seed for reproducibility.
            kwargs: Extra knobs forwarded into the request's ``parameters``.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "input": {"prompt": input_prompt, "media": media},
            "parameters": {
                "resolution": resolution,
                "duration": duration,
                "prompt_extend": prompt_extend,
                "watermark": watermark,
            },
        }
        if negative_prompt:
            payload["input"]["negative_prompt"] = negative_prompt
        if seed is not None:
            payload["parameters"]["seed"] = seed
        payload["parameters"].update(kwargs)
        return self._post_task(payload)

    def predict(
        self,
        processed_data: Dict[str, Any],
        task_type: str = "i2av",
        model: Optional[str] = None,
        resolution: str = "720P",
        duration: int = 5,
        negative_prompt: str = "",
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Submit a generation task.

        Args:
            processed_data: Input bundle with ``"prompt"`` and ``"media"``.
            task_type: Only ``"i2av"`` is offered by Wan2.7.

        Raises:
            ValueError: On any other ``task_type``, or when ``media`` is absent.
        """
        if task_type != "i2av":
            raise ValueError("Wan2.7 currently only supports i2av task_type.")

        prompt = processed_data.get("prompt", "")
        media = processed_data.get("media", None)
        if not media:
            raise ValueError("Wan2.7 requires media input.")

        response = self.generate_i2av(
            media=media,
            input_prompt=prompt,
            model=model or "wan2.7-i2v",
            resolution=resolution,
            duration=duration,
            negative_prompt=negative_prompt,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed,
            **kwargs,
        )

        return {
            "task_type": task_type,
            "prompt": prompt,
            "response": response,
        }
