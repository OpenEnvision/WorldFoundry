"""Google Gemini API client for Veo 3.1 video generation."""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from ..api_video_client import CredentialedSynthesis


class Veo3Synthesis(CredentialedSynthesis):
    """Submit, poll, and optionally download a Veo video through google-genai."""

    MODEL = "veo-3.1-generate-preview"
    DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com"

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str = "your_api_key",
        logger=None,
        model: str | None = None,
    ) -> None:
        super().__init__(endpoint=endpoint, api_key=api_key, logger=logger)
        if self.endpoint.rstrip("/").endswith("/openai"):
            raise ValueError("Veo requires the native Gemini API base URL, not its OpenAI compatibility endpoint")
        self.model = model or self.MODEL
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("Veo requires google-genai; install worldfoundry[api]") from exc
        self._types = types
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(base_url=self.endpoint.rstrip("/"), api_version="v1beta"),
        )

    def _as_google_image(self, image: Image.Image):
        if not isinstance(image, Image.Image):
            raise TypeError(f"Veo image must be PIL.Image, got {type(image)}")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return self._types.Image(image_bytes=buffer.getvalue(), mime_type="image/png")

    def _generate(
        self,
        processed_data: Dict[str, Any],
        *,
        output_path: str | Path | None = None,
        wait: bool = True,
        poll_interval: float = 10,
        timeout: float = 600,
    ) -> Dict[str, Any]:
        if poll_interval <= 0 or timeout <= 0:
            raise ValueError("poll_interval and timeout must be positive")

        image = processed_data.get("images")
        last_frame = processed_data.get("last_frame")
        reference_images = processed_data.get("reference_images") or []
        if last_frame is not None and image is None:
            raise ValueError("last_frame requires a starting image")
        if reference_images and (image is not None or last_frame is not None):
            raise ValueError("Veo reference_images cannot be combined with first or last frames")
        if len(reference_images) > 3:
            raise ValueError("Veo accepts at most three reference images")

        config_values = {
            key: value
            for key, value in (processed_data.get("veo_config") or {}).items()
            if value is not None
        }
        if last_frame is not None:
            config_values["last_frame"] = self._as_google_image(last_frame)
        if reference_images:
            config_values["reference_images"] = [
                self._types.VideoGenerationReferenceImage(
                    image=self._as_google_image(item), reference_type="asset"
                )
                for item in reference_images
            ]
        operation = self.client.models.generate_videos(
            model=self.model,
            prompt=processed_data["prompt"],
            image=self._as_google_image(image) if image is not None else None,
            config=self._types.GenerateVideosConfig(**config_values),
        )
        result: Dict[str, Any] = {"operation_name": operation.name, "done": bool(operation.done)}
        if not wait:
            return result

        deadline = time.monotonic() + timeout
        while not operation.done:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Veo generation timed out; operation: {operation.name}")
            time.sleep(min(poll_interval, remaining))
            operation = self.client.operations.get(operation)
        if operation.error:
            raise RuntimeError(f"Veo generation failed: {operation.error}")
        videos = getattr(operation.response, "generated_videos", None) if operation.response else None
        if not videos:
            raise RuntimeError(f"Veo operation completed without a video: {operation.name}")

        video = videos[0].video
        result.update(done=True, video_url=getattr(video, "uri", None))
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            # google-genai 1.60 returns bytes; newer versions also accept a destination.
            path.write_bytes(self.client.files.download(file=video))
            result["output_path"] = str(path)
        return result

    def generate_t2av(self, processed_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._generate(processed_data, **kwargs)

    def generate_i2av(self, processed_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._generate(processed_data, **kwargs)

    def predict(
        self,
        processed_data: Dict[str, Any],
        task_type: str = "auto",
        **kwargs,
    ) -> Dict[str, Any]:
        images = processed_data.get("images")
        if task_type == "auto":
            task_type = "i2av" if images is not None else "t2av"
        if task_type == "i2av":
            if images is None:
                raise ValueError("i2av requires images")
            result = self.generate_i2av(processed_data, **kwargs)
        elif task_type == "t2av":
            if images is not None:
                raise ValueError("t2av cannot include images")
            result = self.generate_t2av(processed_data, **kwargs)
        else:
            raise ValueError(f"Unsupported Veo task type: {task_type}")
        return {"task_type": task_type, "result": result}


__all__ = ["Veo3Synthesis"]
