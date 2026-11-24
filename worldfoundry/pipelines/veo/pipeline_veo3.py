"""Veo3 visual generation pipeline module."""

from __future__ import annotations

import os
from typing import Any

from PIL import Image

from ...operators.veo3_operator import Veo3Operator
from ...synthesis.visual_generation.veo.veo3_synthesis import Veo3Synthesis
from ..pipeline_utils import PipelineABC

_DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com"
_API_KEY_ENV = ("VEO3_API_KEY", "VEO_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
_ENDPOINT_ENV = ("VEO3_ENDPOINT", "VEO_ENDPOINT", "GOOGLE_GEMINI_BASE_URL")
_PLACEHOLDER_KEYS = {"your_api_key", "your api key"}


def _resolve_api_key(api_key: str | None) -> str:
    """Resolve the Veo3 API key from explicit input or supported environment variables.

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
    raise ValueError(
        "Veo3 API key is required. Pass api_key or set VEO3_API_KEY/VEO_API_KEY/GEMINI_API_KEY/GOOGLE_API_KEY."
    )


def _resolve_endpoint(endpoint: str | None) -> str:
    """Resolve the Veo3 endpoint from explicit input, env, or the Gemini default.

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


class Veo3Pipeline(PipelineABC):
    """Veo 3 pipeline built on the native Gemini video generation API.

    Text-to-video or image-to-video is chosen from whether an image is given.
    """

    def __init__(
        self,
        operator: Veo3Operator | None = None,
        synthesis_model: Veo3Synthesis | None = None,
        endpoint: str = _DEFAULT_ENDPOINT,
        api_key: str = "your_api_key",
    ):
        """
        Initialize the Veo 3 pipeline.

        Args:
            operator: Optional Veo 3 operator; ``api_init`` creates one.
            synthesis_model: Optional Veo 3 synthesis model; ``api_init`` creates one.
            endpoint: API base URL.
        """
        self.api_key = api_key
        self.endpoint = endpoint
        self.operator = operator
        self.synthesis_model = synthesis_model

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> Veo3Pipeline:
        """Build the API-only Veo3 pipeline without loading a checkpoint.

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
            raise ValueError("Veo3 is API-only; pass API options as a dict instead of a checkpoint path.")
        options.update(required_components or {})
        options.update(kwargs)
        return cls.api_init(
            endpoint=options.get("endpoint"),
            api_key=options.get("api_key"),
            model=options.get("model"),
            logger=options.get("logger"),
        )

    @classmethod
    def api_init(cls, api_key: str | None = None, endpoint: str | None = None, logger=None, **kwargs) -> Veo3Pipeline:
        """
        Build the pipeline with its synthesis model and operator from API settings.

        Args:
            api_key: API key; resolved from the environment when omitted.
            endpoint: API base URL; resolved from the environment or the Gemini default when omitted.
            logger: Optional logger for progress messages.
            **kwargs: Forwarded to ``Veo3Synthesis.api_init``.

        Returns:
            Veo3Pipeline: The initialized pipeline.

        Raises:
            ValueError: No API key is passed or set in the environment.
        """
        endpoint = _resolve_endpoint(endpoint)
        api_key = _resolve_api_key(api_key)
        if logger:
            logger.info(f"Loading Veo3 pipeline with endpoint: {endpoint}")

        if logger:
            logger.info("Loading Veo3 synthesis model...")
        synthesis_model = Veo3Synthesis.api_init(endpoint=endpoint, api_key=api_key, logger=logger, **kwargs)

        if logger:
            logger.info("Initializing Veo3 operator...")
        operator = Veo3Operator()

        pipeline = cls(operator=operator, synthesis_model=synthesis_model, api_key=api_key, endpoint=endpoint)

        if logger:
            logger.info("Veo3 pipeline loaded successfully")

        return pipeline

    def process(
        self,
        prompt: str,
        images: Image.Image | None = None,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        duration_seconds: int = 8,
        negative_prompt: str | None = None,
        seed: int | None = None,
        last_frame: Image.Image | None = None,
        reference_images: list[Image.Image] | None = None,
        person_generation: str | None = None,
        enhance_prompt: bool | None = None,
        generate_audio: bool | None = None,
        fps: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Preprocess the prompt and images with the operator for the synthesis model.

        Args:
            prompt: Text prompt.
            images: Optional first-frame image.
            aspect_ratio: Aspect ratio such as ``"16:9"``.
            resolution: Resolution such as ``"720p"``.
            duration_seconds: Video duration in seconds.
            seed: Random seed.
            last_frame: Optional last-frame image.
            reference_images: Optional list of reference images.
            person_generation: Optional person-generation setting.
            **kwargs: Forwarded to ``Veo3Operator.process_perception``.

        Returns:
            Dict with ``prompt`` and the operator's ``images``, ``reference_images``, ``last_frame``
            and ``veo_config``.
        """
        if self.operator is None:
            raise ValueError("Operator is not initialized")

        processed_data: dict[str, Any] = {}

        self.operator.get_interaction(prompt)
        processed_interaction = self.operator.process_interaction()
        processed_data["prompt"] = processed_interaction["processed_prompt"]

        processed_perception = self.operator.process_perception(
            prompt=prompt,
            images=images,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            duration_seconds=duration_seconds,
            negative_prompt=negative_prompt,
            seed=seed,
            last_frame=last_frame,
            reference_images=reference_images,
            person_generation=person_generation,
            enhance_prompt=enhance_prompt,
            generate_audio=generate_audio,
            fps=fps,
            **kwargs,
        )

        processed_data.update(processed_perception)

        return processed_data

    def __call__(
        self,
        prompt: str,
        images: Image.Image | None = None,
        task_type: str = "auto",
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        duration_seconds: int = 8,
        negative_prompt: str | None = None,
        seed: int | None = None,
        last_frame: Image.Image | None = None,
        reference_images: list[Image.Image] | None = None,
        person_generation: str | None = None,
        enhance_prompt: bool | None = None,
        generate_audio: bool | None = None,
        fps: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate a video, choosing text-to-video or image-to-video from whether ``images`` is given.

        Args:
            prompt: Text prompt.
            images: Optional first-frame image; with ``task_type="auto"`` it selects i2av.
            task_type: ``"auto"`` to infer the task, ``"t2av"`` for text-to-video, ``"i2av"`` for image-to-video.
            aspect_ratio: Aspect ratio such as ``"16:9"``.
            resolution: Resolution such as ``"720p"``.
            duration_seconds: Video duration in seconds.
            seed: Random seed.
            last_frame: Optional last-frame image.
            reference_images: Optional list of reference images.
            person_generation: Optional person-generation setting.
            **kwargs: Forwarded to preprocessing and ``Veo3Synthesis.predict``.

        Returns:
            Dict with:
                - task_type: The task type used.
                - result: The generation result.
        """
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")

        if self.operator is None:
            raise ValueError("Operator is not initialized")
        _resolve_api_key(self.api_key)
        _resolve_endpoint(self.endpoint)

        processed_data = self.process(
            prompt=prompt,
            images=images,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            duration_seconds=duration_seconds,
            negative_prompt=negative_prompt,
            seed=seed,
            last_frame=last_frame,
            reference_images=reference_images,
            person_generation=person_generation,
            enhance_prompt=enhance_prompt,
            generate_audio=generate_audio,
            fps=fps,
            **kwargs,
        )

        response = self.synthesis_model.predict(processed_data=processed_data, task_type=task_type, **kwargs)

        return response

    def get_operator(self) -> Veo3Operator | None:
        """Return the operator."""
        return self.operator

    def get_synthesis_model(self) -> Veo3Synthesis | None:
        """Return the synthesis model."""
        return self.synthesis_model
