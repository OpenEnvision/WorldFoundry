"""Client for the Luma Labs Dream Machine (Ray-2) video generation API."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..api_video_client import ApiVideoSynthesis


class LumaRay2Synthesis(ApiVideoSynthesis):
    """Generate videos from prompts or keyframes through Luma's Dream Machine API."""

    DEFAULT_ENDPOINT = "https://api.lumalabs.ai/dream-machine/v1"

    def generate_t2av(
        self,
        input_prompt: str,
        model: str = "ray-2",
        resolution: str = "720p",
        duration: str = "5s",
        aspect_ratio: Optional[str] = "16:9",
        loop: bool = False,
        concepts: Optional[Any] = None,
        callback_url: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Start a generation from a text prompt alone."""
        payload = self._payload(
            input_prompt=input_prompt,
            model=model,
            resolution=resolution,
            duration=duration,
            aspect_ratio=aspect_ratio,
            loop=loop,
            concepts=concepts,
            callback_url=callback_url,
            **kwargs,
        )
        return self._post("/generations", payload)

    def generate_i2av(
        self,
        keyframes: Dict[str, Any],
        input_prompt: str,
        model: str = "ray-2",
        resolution: str = "720p",
        duration: str = "5s",
        aspect_ratio: Optional[str] = "16:9",
        loop: bool = False,
        concepts: Optional[Any] = None,
        callback_url: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Start a generation anchored on ``keyframes``."""
        payload = self._payload(
            input_prompt=input_prompt,
            model=model,
            resolution=resolution,
            duration=duration,
            aspect_ratio=aspect_ratio,
            loop=loop,
            concepts=concepts,
            callback_url=callback_url,
            **kwargs,
        )
        payload["keyframes"] = keyframes
        return self._post("/generations", payload)

    @staticmethod
    def _payload(
        *,
        input_prompt: str,
        model: str,
        resolution: str,
        duration: str,
        aspect_ratio: Optional[str],
        loop: bool,
        concepts: Optional[Any],
        callback_url: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Build the request body shared by both generation routes."""
        payload: Dict[str, Any] = {
            "prompt": input_prompt,
            "model": model,
            "resolution": resolution,
            "duration": duration,
            "loop": loop,
        }
        if aspect_ratio is not None:
            payload["aspect_ratio"] = aspect_ratio
        if concepts is not None:
            payload["concepts"] = concepts
        if callback_url is not None:
            payload["callback_url"] = callback_url
        payload.update(kwargs)
        return payload

    def get_generation(self, generation_id: str) -> Dict[str, Any]:
        """Return the status and output of a generation task."""
        return self._get(f"/generations/{generation_id}")

    def predict(
        self,
        processed_data: Dict[str, Any],
        task_type: str = "auto",
        model: str = "ray-2",
        resolution: str = "720p",
        duration: str = "5s",
        aspect_ratio: Optional[str] = "16:9",
        loop: bool = False,
        concepts: Optional[Any] = None,
        callback_url: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Start a generation, choosing the route from the presence of keyframes.

        Args:
            processed_data: Input bundle, e.g. ``{"prompt": ..., "keyframes": ...}``.
            task_type: ``"auto"``, ``"t2av"`` or ``"i2av"``.

        Raises:
            ValueError: On an unknown ``task_type``, or an ``i2av`` request with
                no keyframes.
        """
        prompt = processed_data.get("prompt", "")
        keyframes = processed_data.get("keyframes", None)

        if task_type == "auto":
            task_type = "i2av" if keyframes else "t2av"

        options = dict(
            model=model,
            resolution=resolution,
            duration=duration,
            aspect_ratio=aspect_ratio,
            loop=loop,
            concepts=concepts,
            callback_url=callback_url,
            **kwargs,
        )
        if task_type == "t2av":
            response = self.generate_t2av(input_prompt=prompt, **options)
        elif task_type == "i2av":
            if not keyframes:
                raise ValueError("i2av task requires keyframes input.")
            response = self.generate_i2av(keyframes=keyframes, input_prompt=prompt, **options)
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        return {
            "task_type": task_type,
            "prompt": prompt,
            "response": response,
        }


__all__ = ["LumaRay2Synthesis"]
