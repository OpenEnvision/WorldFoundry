"""Wan 2.5 hosted-API visual generation pipeline module."""

import logging
from http import HTTPStatus
from typing import Any

from PIL import Image

from ...operators.wan_2p5_operator import Wan2p5Operator
from ...synthesis.visual_generation.wan.wan_2p5_synthesis import Wan2p5Synthesis
from ..api_runtime import load_api_pipeline_from_pretrained, resolve_api_key
from ..pipeline_utils import PipelineABC

_API_KEY_ENV = ("DASHSCOPE_API_KEY", "WAN_API_KEY", "ALIYUN_API_KEY")

logger = logging.getLogger(__name__)


class Wan2p5Pipeline(PipelineABC):
    """Wan2.5 hosted-API pipeline.

    The operator preprocesses inputs before the synthesis model calls the API, keeping
    preprocessing separate from inference.
    """

    def __init__(
        self,
        operator: Wan2p5Operator | None = None,
        synthesis_model: Wan2p5Synthesis | None = None,
        endpoint: str = "https://dashscope.aliyuncs.com/api/v1",
        api_key: str = "your_api_key",
    ):
        """
        Initialize the Wan2.5 pipeline.

        Args:
            operator: Optional Wan2.5 operator; ``api_init`` creates one.
            synthesis_model: Optional Wan2.5 synthesis model; ``api_init`` creates one.
            endpoint: API base URL.
            api_key: API key; read from the environment when empty or a placeholder.
        """
        api_key = resolve_api_key(api_key, _API_KEY_ENV, "Wan2.5")
        self.endpoint = endpoint
        self.api_key = api_key
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
    ) -> "Wan2p5Pipeline":
        """Build the API-only pipeline through the unified loader contract."""
        return load_api_pipeline_from_pretrained(
            cls,
            model_path=model_path,
            required_components=required_components,
            device=device,
            model_id=model_id,
            default_endpoint="https://dashscope.aliyuncs.com/api/v1",
            service_name="Wan 2.5",
            **kwargs,
        )

    @classmethod
    def api_init(
        cls,
        endpoint: str = "https://dashscope.aliyuncs.com/api/v1",
        api_key: str = "your_api_key",
        logger=None,
        **kwargs,
    ) -> "Wan2p5Pipeline":
        """
        Build the pipeline with its synthesis model and operator from API settings.

        Args:
            endpoint: API base URL.
            api_key: API key; read from the environment when empty or a placeholder.
            logger: Optional logger for progress messages.
            **kwargs: Forwarded to ``Wan2p5Synthesis.api_init``.

        Returns:
            Wan2p5Pipeline: The initialized pipeline.
        """
        if logger:
            logger.info(f"Loading Wan2p5 pipeline with endpoint: {endpoint}")
        api_key = resolve_api_key(api_key, _API_KEY_ENV, "Wan2.5")

        if logger:
            logger.info("Loading Wan2p5 synthesis model...")

        synthesis_model = Wan2p5Synthesis.api_init(endpoint=endpoint, api_key=api_key, logger=logger, **kwargs)

        if logger:
            logger.info("Initializing Wan2p5 operator...")

        operator = Wan2p5Operator()

        pipeline = cls(operator=operator, synthesis_model=synthesis_model, endpoint=endpoint, api_key=api_key)

        if logger:
            logger.info("Wan2p5 pipeline loaded successfully")

        return pipeline

    def process(self, prompt: str, images: Image.Image | None = None, **kwargs) -> dict[str, Any]:
        """
        Preprocess the prompt and optional reference image with the operator for the synthesis model.

        Args:
            prompt: Text prompt.
            images: Optional reference image.
            **kwargs: Forwarded to ``Wan2p5Operator.process_perception``.

        Returns:
            Dict with ``prompt``, ``encoded_image`` and ``images``.
        """
        if self.operator is None:
            raise ValueError("Operator is not initialized")

        processed_data: dict[str, Any] = {}

        self.operator.get_interaction(prompt)
        processed_interaction = self.operator.process_interaction()
        processed_data["prompt"] = processed_interaction["processed_prompt"]

        processed_perception = self.operator.process_perception(images=images, **kwargs)
        processed_data["encoded_image"] = processed_perception["encoded_image"]
        processed_data["images"] = processed_perception["images"]

        return processed_data

    def __call__(
        self,
        prompt: str,
        images: Image.Image | None = None,
        task_type: str = "auto",  # "auto", "t2av", "i2av"
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
        """
        Generate a video, inferring the task type unless it is given.

        Args:
            prompt: Text prompt.
            images: Optional reference image; with ``task_type="auto"`` it selects i2av, otherwise t2av.
            task_type: ``"auto"`` to infer the task, ``"t2av"`` for text-to-video, ``"i2av"`` for image-to-video.
            size: Video size of t2av tasks.
            resolution: Resolution of i2av tasks.
            duration: Video duration in seconds.
            audio: Whether to generate audio.
            prompt_extend: Whether the service extends the prompt.
            watermark: Whether to add a watermark.
            seed: Random seed.
            **kwargs: Forwarded to preprocessing and ``Wan2p5Synthesis.predict``.

        Returns:
            Dict with:
                - response: API response object.
                - task_type: The task type used.
                - prompt: The prompt used.
                - video_url: Video URL, or None unless the API call succeeded.
                - task_id: Task ID, or None unless the API call succeeded.
        """
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")

        if self.operator is None:
            raise ValueError("Operator is not initialized")

        processed_data = self.process(prompt=prompt, images=images, **kwargs)

        result = self.synthesis_model.predict(
            processed_data=processed_data,
            task_type=task_type,
            size=size,
            resolution=resolution,
            duration=duration,
            negative_prompt=negative_prompt,
            audio=audio,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed,
            **kwargs,
        )

        response = result.get("response")
        task_type = result.get("task_type")

        if response and hasattr(response, "status_code") and response.status_code == HTTPStatus.OK:
            logger.info("Wan2.5 API call succeeded with status code %s", response.status_code)

            video_url = None
            task_id = None
            if hasattr(response, "output") and response.output:
                output_data = response.output
                video_url = output_data.get("video_url")
                task_id = output_data.get("task_id")

            result["video_url"] = video_url
            result["task_id"] = task_id
        else:
            if response and hasattr(response, "status_code"):
                logger.error(
                    "Wan2.5 API call failed with status code %s: %s",
                    response.status_code,
                    getattr(response, "message", ""),
                )
            else:
                logger.error("Wan2.5 API call failed without a usable response object")
            result["video_url"] = None
            result["task_id"] = None

        return result

    def get_operator(self) -> Wan2p5Operator | None:
        """Return the operator."""
        return self.operator

    def get_synthesis_model(self) -> Wan2p5Synthesis | None:
        """Return the synthesis model."""
        return self.synthesis_model
