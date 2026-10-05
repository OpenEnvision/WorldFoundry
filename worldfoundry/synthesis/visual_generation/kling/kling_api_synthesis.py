from __future__ import annotations

from typing import Any, Dict, Optional

from ..api_video_client import ApiVideoSynthesis


class KlingApiSynthesis(ApiVideoSynthesis):
    """Kling API 合成类。

    默认对接 `https://api.klingapi.com` 这一套通用 Kling 网关接口。
    """

    DEFAULT_ENDPOINT = "https://api.klingapi.com"

    def generate_t2av(
        self,
        input_prompt: str,
        model: str = "kling-v2.6-pro",
        duration: int = 5,
        aspect_ratio: str = "16:9",
        mode: str = "professional",
        negative_prompt: Optional[str] = None,
        callback_url: Optional[str] = None,
        external_task_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        payload = self._payload(
            input_prompt=input_prompt,
            model=model,
            duration=duration,
            aspect_ratio=aspect_ratio,
            mode=mode,
            negative_prompt=negative_prompt,
            callback_url=callback_url,
            external_task_id=external_task_id,
            **kwargs,
        )
        return self._post("/v1/videos/text2video", payload)

    def generate_i2av(
        self,
        image_payload: Dict[str, Any],
        input_prompt: str,
        model: str = "kling-v2.6-pro",
        duration: int = 5,
        aspect_ratio: str = "16:9",
        mode: str = "professional",
        negative_prompt: Optional[str] = None,
        callback_url: Optional[str] = None,
        external_task_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        payload = self._payload(
            input_prompt=input_prompt,
            model=model,
            duration=duration,
            aspect_ratio=aspect_ratio,
            mode=mode,
            negative_prompt=negative_prompt,
            callback_url=callback_url,
            external_task_id=external_task_id,
            **kwargs,
        )
        payload[image_payload["field"]] = image_payload["value"]
        return self._post("/v1/videos/image2video", payload)

    @staticmethod
    def _payload(
        *,
        input_prompt: str,
        model: str,
        duration: int,
        aspect_ratio: str,
        mode: str,
        negative_prompt: Optional[str],
        callback_url: Optional[str],
        external_task_id: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Build the request body shared by the text and image generation routes."""
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": input_prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "mode": mode,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if callback_url:
            payload["callback_url"] = callback_url
        if external_task_id:
            payload["external_task_id"] = external_task_id
        payload.update(kwargs)
        return payload

    def get_task(self, task_id: str) -> Dict[str, Any]:
        return self._get(f"/v1/videos/{task_id}")

    def predict(
        self,
        processed_data: Dict[str, Any],
        task_type: str = "auto",
        model: str = "kling-v2.6-pro",
        duration: int = 5,
        aspect_ratio: str = "16:9",
        mode: str = "professional",
        negative_prompt: Optional[str] = None,
        callback_url: Optional[str] = None,
        external_task_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        prompt = processed_data.get("prompt", "")
        image_payload = processed_data.get("image_payload")

        if task_type == "auto":
            task_type = "i2av" if image_payload is not None else "t2av"

        options = dict(
            model=model,
            duration=duration,
            aspect_ratio=aspect_ratio,
            mode=mode,
            negative_prompt=negative_prompt,
            callback_url=callback_url,
            external_task_id=external_task_id,
            **kwargs,
        )
        if task_type == "t2av":
            response = self.generate_t2av(input_prompt=prompt, **options)
        elif task_type == "i2av":
            if image_payload is None:
                raise ValueError("i2av task requires image input.")
            response = self.generate_i2av(image_payload=image_payload, input_prompt=prompt, **options)
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        return {
            "task_type": task_type,
            "prompt": prompt,
            "response": response,
        }


__all__ = ["KlingApiSynthesis"]
