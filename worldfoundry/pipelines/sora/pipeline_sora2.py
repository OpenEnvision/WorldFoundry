"""Sora2 visual generation pipeline module."""

import logging
import os
import time
from typing import Any

from PIL import Image

from ...operators.sora2_operator import Sora2Operator
from ...synthesis.visual_generation.sora.sora2_synthesis import Sora2Synthesis
from ..pipeline_utils import PipelineABC

_DEFAULT_ENDPOINT = "https://api.openai.com/v1"
_API_KEY_ENV = ("SORA2_API_KEY", "SORA_API_KEY", "OPENAI_API_KEY")
_ENDPOINT_ENV = ("SORA2_ENDPOINT", "SORA_ENDPOINT", "OPENAI_BASE_URL")
_PLACEHOLDER_KEYS = {"your_api_key", "your api key"}

logger = logging.getLogger(__name__)


def _resolve_api_key(api_key: str | None) -> str:
    """Resolve the Sora2 API key from explicit input or supported environment variables.

    Args:
        api_key: Optional API key passed by the caller.
    """
    if api_key and api_key.strip() and api_key.strip() not in _PLACEHOLDER_KEYS:
        return api_key.strip()
    for env_name in _API_KEY_ENV:
        # Attempt to retrieve from environment variables as a fallback resolution
        env_value = os.getenv(env_name)
        if env_value and env_value.strip():
            return env_value.strip()
    raise ValueError("Sora2 API key is required. Pass api_key or set SORA2_API_KEY/SORA_API_KEY/OPENAI_API_KEY.")


def _resolve_endpoint(endpoint: str | None) -> str:
    """Resolve the Sora2 API endpoint from explicit input, env, or default.

    Args:
        endpoint: Optional endpoint URL passed by the caller.
    """
    if endpoint and endpoint.strip():
        return endpoint.strip()
    for env_name in _ENDPOINT_ENV:
        # Attempt to retrieve from environment variables as a fallback resolution
        env_value = os.getenv(env_name)
        if env_value and env_value.strip():
            return env_value.strip()
    return _DEFAULT_ENDPOINT


class Sora2Pipeline(PipelineABC):
    """Pipeline implementation for Sora2 visual generation."""

    def __init__(
        self,
        operator: Sora2Operator | None = None,
        synthesis_model: Sora2Synthesis | None = None,
        endpoint: str = "https://api.openai.com/v1",
        api_key: str = "your_api_key",
    ):
        """
        Initialize the Sora 2 pipeline.

        Args:
            operator: Optional Sora 2 operator; ``api_init`` creates one.
            synthesis_model: Optional Sora 2 synthesis model; ``api_init`` creates one.
            endpoint: API base URL.
        """
        self.operator = operator
        self.synthesis_model = synthesis_model
        self.endpoint = endpoint
        self.api_key = api_key

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "Sora2Pipeline":
        """Build the API-only Sora2 pipeline without loading a checkpoint.

        Args:
            model_path: Optional dict of API adapter options.
            required_components: Optional dict merged into adapter options.
            device: Accepted for the shared loader signature.
            model_id: Accepted for the shared loader signature.
            **kwargs: Additional API adapter options.
        """
        del device, model_id
        options: dict[str, Any] = {}
        if isinstance(model_path, dict):
            options.update(model_path)
        elif model_path is not None:
            raise ValueError("Sora2 is API-only; pass API options as a dict instead of a checkpoint path.")
        options.update(required_components or {})
        options.update(kwargs)
        return cls.api_init(
            endpoint=options.get("endpoint"),
            api_key=options.get("api_key"),
            logger=options.get("logger"),
        )

    @classmethod
    def api_init(
        cls, endpoint: str | None = None, api_key: str | None = None, logger=None, **kwargs
    ) -> "Sora2Pipeline":
        """
        Build the pipeline with its synthesis model and operator from API settings.

        Args:
            endpoint: API base URL; resolved from the environment or the default when omitted.
            api_key: API key; resolved from the environment when omitted.
            logger: Optional logger for progress messages.
            **kwargs: Forwarded to ``Sora2Synthesis.api_init``.

        Raises:
            ValueError: No API key is passed or set in the environment.
        """
        endpoint = _resolve_endpoint(endpoint)
        api_key = _resolve_api_key(api_key)
        if logger:
            logger.info(f"Loading Sora2 pipeline with endpoint: {endpoint}")

        if logger:
            logger.info("Loading Sora2 synthesis model...")
        synthesis_model = Sora2Synthesis.api_init(endpoint=endpoint, api_key=api_key, logger=logger, **kwargs)

        if logger:
            logger.info("Initializing Sora2 operator...")
        operator = Sora2Operator()

        pipeline = cls(operator=operator, synthesis_model=synthesis_model, endpoint=endpoint, api_key=api_key)

        if logger:
            logger.info("Sora2 pipeline loaded successfully")

        return pipeline

    def process(self, prompt: str, images: Image.Image | None = None, **kwargs) -> dict[str, Any]:
        """Preprocess the prompt and optional reference image with the operator for the synthesis model."""
        if self.operator is None:
            raise ValueError("Operator is not initialized")
        processed_data: dict[str, Any] = {}

        # The prompt is the interaction input.
        self.operator.get_interaction(prompt)
        processed_interaction = self.operator.process_interaction()
        processed_data["prompt"] = processed_interaction["processed_prompt"]

        # The reference image is the perception input.
        processed_perception = self.operator.process_perception(images=images, **kwargs)
        processed_data["encoded_image"] = processed_perception["encoded_image"]
        processed_data["images"] = processed_perception["images"]

        return processed_data

    def __call__(
        self,
        prompt: str,
        images: Image.Image | None = None,
        size: str = "1280x720",
        duration: int = 4,
        task_type: str = "auto",
        wait: bool = True,
        poll_interval: int = 2,
        max_retries: int = 600,
        **kwargs,
    ) -> dict[str, Any]:
        """Generate a video, choosing text-to-video or image-to-video from whether ``images`` is given.

        With ``wait`` the call polls until the video completes and stores the final video object in
        ``result["response"]``.
        """

        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")

        if self.operator is None:
            raise ValueError("Operator is not initialized")
        _resolve_api_key(self.api_key)
        _resolve_endpoint(self.endpoint)

        processed_data = self.process(prompt=prompt, images=images, **kwargs)

        result = self.synthesis_model.predict(
            processed_data=processed_data, task_type=task_type, size=size, duration=duration, **kwargs
        )

        if wait:
            final_video = self._poll_video_status(
                video=result["response"], poll_interval=poll_interval, max_retries=max_retries
            )
            result["response"] = final_video

        return result

    def _poll_video_status(self, video: Any, poll_interval: int = 2, max_retries: int = 600) -> Any:
        """Poll the video job until it completes or fails and return the final video object."""
        if self.synthesis_model is None or not hasattr(self.synthesis_model, "client"):
            raise ValueError("Synthesis model client is not initialized")

        openai_client = self.synthesis_model.client

        completed_statuses = ("completed", "succeeded", "success", "done")
        failed_statuses = ("failed", "error")
        retry_count = 0
        last_status = None

        while video.status.lower() not in completed_statuses and video.status.lower() not in failed_statuses:
            if retry_count >= max_retries:
                raise TimeoutError(f"Sora2 video polling reached the maximum number of retries ({max_retries}).")

            video = openai_client.videos.retrieve(video.id)

            progress_raw = getattr(video, "progress", 0)
            progress = progress_raw if progress_raw is not None else 0
            if video.status != last_status:
                logger.info("Sora2 video %s: %s (progress %.1f%%)", video.id, video.status, progress)
                last_status = video.status

            if video.status.lower() in completed_statuses:
                break

            time.sleep(poll_interval)
            retry_count += 1

        if video.status.lower() in failed_statuses:
            error_msg = getattr(getattr(video, "error", None), "message", "Unknown error")
            raise RuntimeError(f"Failed to generate Sora2 video: {error_msg}")

        return video

    def get_operator(self) -> Sora2Operator | None:
        """Return the operator."""
        return self.operator

    def get_synthesis_model(self) -> Sora2Synthesis | None:
        """Return the synthesis model."""
        return self.synthesis_model
