"""Client for the Wan2.5 audio-video generation API (via the DashScope SDK)."""

from __future__ import annotations

from typing import Any

from ..api_video_client import CredentialedSynthesis
from ._dashscope import RUNTIME_STATUS

__all__ = ["RUNTIME_STATUS", "Wan2p5Synthesis"]

#: DashScope rejects a null seed, so unseeded requests get this fixed value.
DEFAULT_SEED = 12345


def _dashscope_video_synthesis(endpoint: str):
    """Load DashScope only for external-service calls.

    Args:
        endpoint: DashScope-compatible API base URL.
    """
    import dashscope
    from dashscope import VideoSynthesis

    dashscope.base_http_api_url = endpoint
    return VideoSynthesis


class Wan2p5Synthesis(CredentialedSynthesis):
    """Wan2.5 audio-video generation through the DashScope API."""

    DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1"

    RUNTIME_STATUS = RUNTIME_STATUS
    IN_TREE_BACKEND = False
    BACKEND_STAGE = "external_service"
    EXTERNAL_SERVICE = True

    def generate_t2av(
        self,
        input_prompt: str,
        size: str = "832*480",
        duration: int = 10,
        negative_prompt: str = "",
        audio: bool = True,
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: int | None = None,
        **kwargs,
    ):
        """Generate audio and video from text.

        Args:
            input_prompt: Text prompt.
            size: Video size as 'width*height'.
            duration: Video length in seconds.
            audio: Generate audio.
            prompt_extend: Let the service extend the prompt.
            watermark: Add a watermark.
            seed: Random seed; ``DEFAULT_SEED`` when None.
            kwargs: Extra arguments passed to the API.
        """
        return _dashscope_video_synthesis(self.endpoint).call(
            api_key=self.api_key,
            model="wan2.5-t2v-preview",
            prompt=input_prompt,
            size=size,
            duration=duration,
            negative_prompt=negative_prompt,
            audio=audio,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed if seed is not None else DEFAULT_SEED,
            **kwargs,
        )

    def generate_i2av(
        self,
        encoded_image: str,
        input_prompt: str,
        resolution: str = "480P",
        duration: int = 10,
        negative_prompt: str = "",
        audio: bool = True,
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: int | None = None,
        **kwargs,
    ):
        """Generate audio and video from an image and text.

        Args:
            encoded_image: Base64-encoded image.
            input_prompt: Text prompt.
            resolution: Video resolution, for example "480P".
            duration: Video length in seconds.
            audio: Generate audio.
            prompt_extend: Let the service extend the prompt.
            watermark: Add a watermark.
            seed: Random seed; ``DEFAULT_SEED`` when None.
            kwargs: Extra arguments passed to the API.
        """
        return _dashscope_video_synthesis(self.endpoint).call(
            api_key=self.api_key,
            model="wan2.5-i2v-preview",
            prompt=input_prompt,
            img_url=encoded_image,
            resolution=resolution,
            duration=duration,
            negative_prompt=negative_prompt,
            audio=audio,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed if seed is not None else DEFAULT_SEED,
            **kwargs,
        )

    def predict(
        self,
        processed_data: dict[str, Any],
        task_type: str = "auto",
        size: str = "832*480",
        resolution: str = "480P",
        duration: int = 10,
        negative_prompt: str = "",
        audio: bool = True,
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Submit a generation, choosing the route from what ``processed_data`` carries.

        Args:
            processed_data: Operator output with ``prompt`` and an optional ``encoded_image``.
            task_type: "auto" (i2av when ``encoded_image`` is present, else t2av), "t2av" or "i2av".
            size: Video size of t2av tasks.
            resolution: Resolution of i2av tasks.

        Returns:
            Dict with ``task_type``, ``prompt`` and ``response``.

        Raises:
            ValueError: On an unknown ``task_type`` or an ``i2av`` request without ``encoded_image``.
        """
        prompt = processed_data.get("prompt", "")
        encoded_image = processed_data.get("encoded_image")

        if task_type == "auto":
            task_type = "i2av" if encoded_image is not None else "t2av"

        common = dict(
            input_prompt=prompt,
            duration=duration,
            negative_prompt=negative_prompt,
            audio=audio,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed,
            **kwargs,
        )
        if task_type == "i2av":
            if encoded_image is None:
                raise ValueError("i2av task requires encoded_image input.")
            response = self.generate_i2av(encoded_image=encoded_image, resolution=resolution, **common)
        elif task_type == "t2av":
            response = self.generate_t2av(size=size, **common)
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        return {"task_type": task_type, "prompt": prompt, "response": response}
