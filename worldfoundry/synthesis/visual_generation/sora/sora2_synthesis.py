"""Client for OpenAI Sora 2 video generation."""

from __future__ import annotations

from typing import Any

from ..api_video_client import OpenAiVideoSynthesis


class Sora2Synthesis(OpenAiVideoSynthesis):
    """Sora 2 video generation through the OpenAI videos API."""

    DEFAULT_ENDPOINT = "https://api.openai.com/v1"

    MODEL = "sora-2"

    def generate_t2av(
        self,
        input_prompt: str,
        size: str = "1280x720",
        duration: int = 8,
    ):
        """Submit a text-to-video task."""
        return self.client.videos.create(
            model=self.MODEL,
            prompt=input_prompt,
            size=_normalize_size(size),
            seconds=str(duration),
        )

    def generate_i2av(
        self,
        encoded_image: tuple[str, bytes, str],
        input_prompt: str,
        size: str = "1280x720",
        duration: int = 8,
    ):
        """Submit an image-to-video task.

        Args:
            encoded_image: Image as a ``(filename, bytes, mime_type)`` tuple.
            input_prompt: Text prompt.
            size: Video size, ``WIDTHxHEIGHT`` or ``WIDTH*HEIGHT``.
            duration: Video length in seconds.
        """
        return self.client.videos.create(
            model=self.MODEL,
            prompt=input_prompt,
            size=_normalize_size(size),
            seconds=str(duration),
            input_reference=tuple(encoded_image),
        )

    def predict(
        self,
        processed_data: dict[str, Any],
        task_type: str = "auto",
        size: str = "1280x720",
        duration: int = 8,
        **kwargs,
    ) -> dict[str, Any]:
        """Submit a generation; ``task_type="auto"`` picks I2V when ``encoded_image`` is present, else T2V.

        Raises:
            ValueError: On an unknown ``task_type`` or an ``i2av`` request without ``encoded_image``.
        """
        prompt = processed_data.get("prompt", "")
        encoded_image = processed_data.get("encoded_image")

        if task_type == "auto":
            task_type = "i2av" if encoded_image is not None else "t2av"

        if task_type == "i2av":
            if encoded_image is None:
                raise ValueError("i2av task requires encoded_image input.")
            response = self.generate_i2av(
                encoded_image=encoded_image, input_prompt=prompt, size=size, duration=duration, **kwargs
            )
        elif task_type == "t2av":
            response = self.generate_t2av(input_prompt=prompt, size=size, duration=duration, **kwargs)
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        return {"task_type": task_type, "prompt": prompt, "response": response}


def _normalize_size(size: str) -> str:
    """Accept ``1280*720`` as well as the ``1280x720`` the API expects."""
    return size.replace("*", "x")


__all__ = ["Sora2Synthesis"]
