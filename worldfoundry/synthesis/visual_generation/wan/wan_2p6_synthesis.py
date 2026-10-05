"""Client for the Wan2.6 video synthesis API (text, image and reference driven)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._dashscope import RUNTIME_STATUS, DashScopeVideoSynthesis

__all__ = ["RUNTIME_STATUS", "Wan2p6Synthesis"]


class Wan2p6Synthesis(DashScopeVideoSynthesis):
    """Wan2.6 API 合成类。

    支持文本生成视频、图像生成视频和参考素材生成视频。
    """

    def generate_t2av(
        self,
        input_prompt: str,
        model: str = "wan2.6-t2v",
        size: str = "1280*720",
        duration: int = 5,
        negative_prompt: str = "",
        audio_url: Optional[str] = None,
        prompt_extend: bool = True,
        shot_type: Optional[str] = None,
        watermark: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Submit a text-to-video task."""
        payload = _task_payload(
            model=model,
            input_prompt=input_prompt,
            negative_prompt=negative_prompt,
            parameters={"size": size, "duration": duration},
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed,
            **kwargs,
        )
        if audio_url:
            payload["input"]["audio_url"] = audio_url
        if shot_type is not None:
            payload["parameters"]["shot_type"] = shot_type
        return self._post_task(payload)

    def generate_i2av(
        self,
        encoded_image: str,
        input_prompt: str,
        model: str = "wan2.6-i2v",
        resolution: str = "720P",
        duration: int = 5,
        negative_prompt: str = "",
        audio: Optional[bool] = None,
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Submit an image-to-video task anchored on ``encoded_image``."""
        payload = _task_payload(
            model=model,
            input_prompt=input_prompt,
            negative_prompt=negative_prompt,
            parameters={"resolution": resolution, "duration": duration},
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed,
            **kwargs,
        )
        payload["input"]["img_url"] = encoded_image
        if audio is not None:
            payload["parameters"]["audio"] = audio
        return self._post_task(payload)

    def generate_r2av(
        self,
        reference_urls: List[str],
        input_prompt: str,
        model: str = "wan2.6-r2v",
        size: str = "1280*720",
        duration: int = 5,
        negative_prompt: str = "",
        audio: Optional[bool] = None,
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Submit a reference-driven video task."""
        payload = _task_payload(
            model=model,
            input_prompt=input_prompt,
            negative_prompt=negative_prompt,
            parameters={"size": size, "duration": duration},
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed,
            **kwargs,
        )
        payload["input"]["reference_urls"] = reference_urls
        if audio is not None:
            payload["parameters"]["audio"] = audio
        return self._post_task(payload)

    def predict(
        self,
        processed_data: Dict[str, Any],
        task_type: str = "auto",
        model: Optional[str] = None,
        size: str = "1280*720",
        resolution: str = "720P",
        duration: int = 5,
        negative_prompt: str = "",
        audio: Optional[bool] = None,
        prompt_extend: bool = True,
        shot_type: Optional[str] = None,
        watermark: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Submit a generation, choosing the route from what ``processed_data`` carries.

        Args:
            processed_data: Input bundle; ``reference_urls`` selects ``r2av``,
                ``encoded_image`` selects ``i2av``, otherwise ``t2av``.
            task_type: ``"auto"``, ``"t2av"``, ``"i2av"`` or ``"r2av"``.

        Raises:
            ValueError: On an unknown ``task_type`` or a route missing its input.
        """
        prompt = processed_data.get("prompt", "")
        encoded_image = processed_data.get("encoded_image", None)
        reference_urls = processed_data.get("reference_urls", None)
        audio_url = processed_data.get("audio_url", None)

        if task_type == "auto":
            if reference_urls:
                task_type = "r2av"
            elif encoded_image is not None:
                task_type = "i2av"
            else:
                task_type = "t2av"

        common = dict(
            input_prompt=prompt,
            duration=duration,
            negative_prompt=negative_prompt,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed,
            **kwargs,
        )
        if task_type == "t2av":
            response = self.generate_t2av(
                model=model or "wan2.6-t2v",
                size=size,
                audio_url=audio_url,
                shot_type=shot_type,
                **common,
            )
        elif task_type == "i2av":
            if encoded_image is None:
                raise ValueError("i2av task requires images input.")
            response = self.generate_i2av(
                encoded_image=encoded_image,
                model=model or "wan2.6-i2v",
                resolution=resolution,
                audio=audio,
                **common,
            )
        elif task_type == "r2av":
            if not reference_urls:
                raise ValueError("r2av task requires reference_urls input.")
            response = self.generate_r2av(
                reference_urls=reference_urls,
                model=model or "wan2.6-r2v",
                size=size,
                audio=audio,
                **common,
            )
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        return {
            "task_type": task_type,
            "prompt": prompt,
            "response": response,
        }


def _task_payload(
    *,
    model: str,
    input_prompt: str,
    negative_prompt: str,
    parameters: Dict[str, Any],
    prompt_extend: bool,
    watermark: bool,
    seed: Optional[int],
    **kwargs
) -> Dict[str, Any]:
    """Build the DashScope task body shared by every Wan2.6 route.

    Extra keyword arguments land in ``parameters``, matching what the API expects
    for per-model knobs the adapter does not name explicitly.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "input": {"prompt": input_prompt},
        "parameters": {**parameters, "prompt_extend": prompt_extend, "watermark": watermark},
    }
    if negative_prompt:
        payload["input"]["negative_prompt"] = negative_prompt
    if seed is not None:
        payload["parameters"]["seed"] = seed
    payload["parameters"].update(kwargs)
    return payload
