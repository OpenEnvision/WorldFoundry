"""Client for the MiniMax Hailuo 2.3 video generation API."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from ..api_video_client import ApiVideoSynthesis


class Hailuo2p3Synthesis(ApiVideoSynthesis):
    """Generate videos through MiniMax's Hailuo 2.3 API.

    MiniMax reports application-level failures inside a ``base_resp`` envelope
    alongside an HTTP 200, so every decoded response passes through
    :meth:`_ensure_success`.
    """

    DEFAULT_ENDPOINT = "https://api.minimax.io/v1"

    @staticmethod
    def _ensure_success(payload: Dict[str, Any]) -> None:
        """Raise when MiniMax reports a failure in the ``base_resp`` envelope.

        Raises:
            RuntimeError: If ``base_resp.status_code`` is anything but zero.
        """
        base_resp = payload.get("base_resp", {})
        status_code = base_resp.get("status_code", 0)
        if status_code not in (0, "0"):
            raise RuntimeError(
                base_resp.get("status_msg", f"MiniMax request failed with status_code={status_code}")
            )

    def _post(self, route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        result = super()._post(route, payload)
        self._ensure_success(result)
        return result

    def _get(self, route: str, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        result = super()._get(route, params)
        self._ensure_success(result)
        return result

    def generate_t2av(
        self,
        input_prompt: str,
        model: str = "MiniMax-Hailuo-2.3",
        resolution: str = "768P",
        duration: int = 6,
        **kwargs
    ) -> Dict[str, Any]:
        """Submit a text-to-video generation task."""
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": input_prompt,
            "resolution": resolution,
            "duration": duration,
        }
        payload.update(kwargs)
        return self._post("/video_generation", payload)

    def generate_i2av(
        self,
        first_frame_image: str,
        input_prompt: str,
        model: str = "MiniMax-Hailuo-2.3",
        resolution: str = "768P",
        duration: int = 6,
        **kwargs
    ) -> Dict[str, Any]:
        """Submit an image-to-video generation task anchored on ``first_frame_image``."""
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": input_prompt,
            "first_frame_image": first_frame_image,
            "resolution": resolution,
            "duration": duration,
        }
        payload.update(kwargs)
        return self._post("/video_generation", payload)

    def query_task(self, task_id: str) -> Dict[str, Any]:
        """Return the status and details of a submitted generation task."""
        return self._get("/query/video_generation", {"task_id": task_id})

    def retrieve_file(self, file_id: str) -> Dict[str, Any]:
        """Return metadata and the download URL for a generated file."""
        return self._get("/files/retrieve", {"file_id": file_id})

    def predict(
        self,
        processed_data: Dict[str, Any],
        task_type: str = "auto",
        model: str = "MiniMax-Hailuo-2.3",
        resolution: str = "768P",
        duration: int = 6,
        **kwargs
    ) -> Dict[str, Any]:
        """Submit a generation, choosing the route from the presence of a first frame.

        Args:
            processed_data: Input bundle with ``"prompt"`` and optionally
                ``"first_frame_image"``.
            task_type: ``"auto"``, ``"t2av"`` or ``"i2av"``.

        Raises:
            ValueError: On an unknown ``task_type``, or an ``i2av`` request with
                no first frame.
        """
        prompt = processed_data.get("prompt", "")
        first_frame_image = processed_data.get("first_frame_image", None)

        if task_type == "auto":
            task_type = "i2av" if first_frame_image is not None else "t2av"

        options = dict(model=model, resolution=resolution, duration=duration, **kwargs)
        if task_type == "t2av":
            response = self.generate_t2av(input_prompt=prompt, **options)
        elif task_type == "i2av":
            if first_frame_image is None:
                raise ValueError("i2av task requires images input.")
            response = self.generate_i2av(
                first_frame_image=first_frame_image,
                input_prompt=prompt,
                **options,
            )
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        return {
            "task_type": task_type,
            "prompt": prompt,
            "response": response,
        }


__all__ = ["Hailuo2p3Synthesis"]
