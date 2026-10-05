"""Client for the Runway Gen-4.5 video generation API."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..api_video_client import ApiVideoSynthesis


class RunwayGen4p5Synthesis(ApiVideoSynthesis):
    """Generate videos through Runway's Gen-4.5 API."""

    DEFAULT_ENDPOINT = "https://api.dev.runwayml.com/v1"

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: str = "your_api_key",
        runway_version: str = "2024-11-06",
        logger=None,
    ) -> None:
        """Store the credential plus the API version Runway requires per request."""
        super().__init__(endpoint=endpoint, api_key=api_key, logger=logger)
        self.runway_version = runway_version

    def _headers(self) -> Dict[str, str]:
        """Add Runway's mandatory API-version header to the standard bearer auth."""
        return {**super()._headers(), "X-Runway-Version": self.runway_version}

    def generate_t2av(
        self,
        input_prompt: str,
        model: str = "gen4.5",
        ratio: str = "1280:720",
        duration: int = 5,
        seed: Optional[int] = None,
        public_figure_threshold: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Submit a text-to-video generation task."""
        payload = self._payload(
            input_prompt=input_prompt,
            model=model,
            ratio=ratio,
            duration=duration,
            seed=seed,
            public_figure_threshold=public_figure_threshold,
            **kwargs,
        )
        return self._post("/text_to_video", payload)

    def generate_i2av(
        self,
        prompt_image: Any,
        input_prompt: str,
        model: str = "gen4.5",
        ratio: str = "1280:720",
        duration: int = 5,
        seed: Optional[int] = None,
        public_figure_threshold: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Submit an image-to-video generation task anchored on ``prompt_image``."""
        payload = self._payload(
            input_prompt=input_prompt,
            model=model,
            ratio=ratio,
            duration=duration,
            seed=seed,
            public_figure_threshold=public_figure_threshold,
            **kwargs,
        )
        payload["promptImage"] = prompt_image
        return self._post("/image_to_video", payload)

    @staticmethod
    def _payload(
        *,
        input_prompt: str,
        model: str,
        ratio: str,
        duration: int,
        seed: Optional[int],
        public_figure_threshold: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Build the request body shared by both generation routes."""
        payload: Dict[str, Any] = {
            "model": model,
            "promptText": input_prompt,
            "ratio": ratio,
            "duration": duration,
        }
        if seed is not None:
            payload["seed"] = seed
        if public_figure_threshold is not None:
            payload["contentModeration"] = {"publicFigureThreshold": public_figure_threshold}
        payload.update(kwargs)
        return payload

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """Return the status and details of a submitted generation task."""
        return self._get(f"/tasks/{task_id}")

    def predict(
        self,
        processed_data: Dict[str, Any],
        task_type: str = "auto",
        model: str = "gen4.5",
        ratio: str = "1280:720",
        duration: int = 5,
        seed: Optional[int] = None,
        public_figure_threshold: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Submit a generation, choosing the route from the presence of a prompt image.

        Args:
            processed_data: Input bundle with ``"prompt"`` and optionally
                ``"prompt_image"``.
            task_type: ``"auto"``, ``"t2av"`` or ``"i2av"``.

        Raises:
            ValueError: On an unknown ``task_type``, or an ``i2av`` request with
                no prompt image.
        """
        prompt = processed_data.get("prompt", "")
        prompt_image = processed_data.get("prompt_image", None)

        if task_type == "auto":
            task_type = "i2av" if prompt_image is not None else "t2av"

        options = dict(
            model=model,
            ratio=ratio,
            duration=duration,
            seed=seed,
            public_figure_threshold=public_figure_threshold,
            **kwargs,
        )
        if task_type == "t2av":
            response = self.generate_t2av(input_prompt=prompt, **options)
        elif task_type == "i2av":
            if prompt_image is None:
                raise ValueError("i2av task requires images input.")
            response = self.generate_i2av(prompt_image=prompt_image, input_prompt=prompt, **options)
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        return {
            "task_type": task_type,
            "prompt": prompt,
            "response": response,
        }


__all__ = ["RunwayGen4p5Synthesis"]
