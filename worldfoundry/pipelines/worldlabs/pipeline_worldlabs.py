"""Worldlabs visual generation pipeline module."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from worldfoundry.core.io import artifact_root_path

from ..pipeline_utils import PipelineABC
from ...operators.worldlabs_operator import WorldLabsOperator
from ...synthesis.visual_generation.worldlabs.worldlabs_synthesis import WorldLabsSynthesis


_DEFAULT_ENDPOINT = "https://api.worldlabs.ai"
_API_KEY_ENV = ("WORLDLABS_API_KEY", "WLT_API_KEY")
_PLACEHOLDER_KEYS = {"your_api_key", "your api key"}

logger = logging.getLogger(__name__)


def _default_output_dir(name: str = "worldlabs_assets") -> str:
    """Resolve a stable default output directory instead of writing to the CWD."""
    return str(artifact_root_path() / name)


def _resolve_api_key(api_key: Optional[str]) -> str:
    """Resolve the World Labs API key from explicit input or environment variables.

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
    raise ValueError("World Labs API key is required. Pass api_key or set WORLDLABS_API_KEY/WLT_API_KEY.")


def _resolve_endpoint(endpoint: Optional[str]) -> str:
    """Resolve the World Labs API endpoint from explicit input, env, or default.

    Args:
        endpoint: Optional endpoint URL passed by the caller.
    """
    if endpoint and endpoint.strip():
        return endpoint.strip()
    # Attempt to retrieve from environment variables as a fallback resolution
    env_value = os.getenv("WORLDLABS_ENDPOINT")
    if env_value and env_value.strip():
        return env_value.strip()
    return _DEFAULT_ENDPOINT


class WorldLabsPipeline(PipelineABC):
    """
    World Labs / Marble API Pipeline。
    """

    def __init__(
        self,
        operator: Optional[WorldLabsOperator] = None,
        synthesis_model: Optional[WorldLabsSynthesis] = None,
        endpoint: str = "https://api.worldlabs.ai",
        api_key: str = "your_api_key",
    ):
        """Initialize the pipeline and configure runtime components."""
        self.endpoint = endpoint
        self.api_key = api_key
        self.operator = operator
        self.synthesis_model = synthesis_model

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> "WorldLabsPipeline":
        """Build the API-only World Labs pipeline without loading a checkpoint.

        Args:
            model_path: Optional dict of API adapter options.
            required_components: Optional dict merged into adapter options.
            device: Accepted for the shared loader signature.
            model_id: Accepted for the shared loader signature.
            **kwargs: Additional API adapter options.
        """
        del device, model_id
        options: Dict[str, Any] = {}
        if isinstance(model_path, dict):
            options.update(model_path)
        elif model_path is not None:
            raise ValueError("World Labs is API-only; pass API options as a dict instead of a checkpoint path.")
        options.update(required_components or {})
        options.update(kwargs)
        return cls.api_init(
            endpoint=options.get("endpoint"),
            api_key=options.get("api_key"),
            logger=options.get("logger"),
        )

    @classmethod
    def api_init(
        cls,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        logger=None,
        **kwargs
    ) -> "WorldLabsPipeline":
        """Initialize API client credentials and runtime endpoints."""
        endpoint = _resolve_endpoint(endpoint)
        api_key = _resolve_api_key(api_key)
        synthesis_model = WorldLabsSynthesis.api_init(
            endpoint=endpoint,
            api_key=api_key,
            logger=logger,
            **kwargs,
        )
        operator = WorldLabsOperator()
        return cls(
            operator=operator,
            synthesis_model=synthesis_model,
            endpoint=endpoint,
            api_key=api_key,
        )

    def process(
        self,
        prompt: Optional[str] = None,
        images: Optional[Union[Image.Image, str, Dict[str, Any], List[Union[Image.Image, str, Dict[str, Any]]]]] = None,
        video: Optional[Union[str, Dict[str, Any]]] = None,
        prompt_type: str = "auto",
        image_azimuths: Optional[List[float]] = None,
        disable_recaption: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        if self.operator is None:
            raise ValueError("Operator is not initialized")

        processed_data: Dict[str, Any] = {}

        self.operator.get_interaction(prompt or "")
        processed_interaction = self.operator.process_interaction()
        processed_data["prompt"] = processed_interaction["processed_prompt"]

        processed_perception = self.operator.process_perception(
            images=images,
            video=video,
            prompt_type=prompt_type,
            image_azimuths=image_azimuths,
            disable_recaption=disable_recaption,
            **kwargs,
        )
        processed_data["world_prompt"] = processed_perception["world_prompt"]
        processed_data["prompt_type"] = processed_perception["prompt_type"]

        if processed_data["prompt_type"] == "text" and not processed_data["prompt"]:
            raise ValueError("World Labs text prompt requires a non-empty prompt string.")
        return processed_data

    def _replace_local_content(
        self,
        content: Dict[str, Any],
        kind: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Replace local content for WorldLabsPipeline."""
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")

        if content.get("source") != "local_file":
            return content

        media_asset_id = self.synthesis_model.upload_media_asset(
            file_path=content["path"],
            kind=kind,
            metadata=metadata,
        )
        return {
            "source": "media_asset",
            "media_asset_id": media_asset_id,
        }

    def _materialize_world_prompt(
        self,
        world_prompt: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Materialize world prompt for WorldLabsPipeline."""
        prompt_type = world_prompt["type"]
        materialized = dict(world_prompt)

        if prompt_type == "image":
            materialized["image_prompt"] = self._replace_local_content(
                materialized["image_prompt"],
                kind="image",
                metadata=metadata,
            )
        elif prompt_type in {"multi-image", "multi_image"}:
            new_items = []
            for item in materialized["multi_image_prompt"]:
                new_items.append(
                    {
                        "azimuth": item["azimuth"],
                        "content": self._replace_local_content(
                            item["content"],
                            kind="image",
                            metadata=metadata,
                        ),
                    }
                )
            materialized["multi_image_prompt"] = new_items
        elif prompt_type == "video":
            materialized["video_prompt"] = self._replace_local_content(
                materialized["video_prompt"],
                kind="video",
                metadata=metadata,
            )

        return materialized

    def _extract_operation_id(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract operation id for WorldLabsPipeline."""
        return response.get("operation_id")

    def _extract_done(self, operation: Dict[str, Any]) -> bool:
        """Extract done for WorldLabsPipeline."""
        return bool(operation.get("done"))

    def _extract_world_object(self, operation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract world object for WorldLabsPipeline."""
        response = operation.get("response")
        return response if isinstance(response, dict) else None

    def _extract_world_id(self, world_or_operation_response: Dict[str, Any]) -> Optional[str]:
        """Extract world id for WorldLabsPipeline."""
        return world_or_operation_response.get("world_id")

    def _poll_operation(
        self,
        operation_id: str,
        poll_interval: int = 10,
        max_retries: int = 120,
    ) -> Dict[str, Any]:
        """Poll operation for WorldLabsPipeline."""
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")

        for _ in range(max_retries):
            operation = self.synthesis_model.get_operation(operation_id)
            logger.info("World Labs operation %s: done=%s", operation_id, operation.get("done", False))
            if self._extract_done(operation):
                return operation
            time.sleep(poll_interval)
        raise TimeoutError(f"World Labs operation {operation_id} polling timed out.")

    def _collect_urls(self, payload: Any, prefix: str = "") -> List[Tuple[str, str]]:
        """Collect urls for WorldLabsPipeline."""
        collected: List[Tuple[str, str]] = []

        if isinstance(payload, dict):
            for key, value in payload.items():
                next_prefix = f"{prefix}.{key}" if prefix else key
                collected.extend(self._collect_urls(value, next_prefix))
        elif isinstance(payload, list):
            for index, value in enumerate(payload):
                next_prefix = f"{prefix}[{index}]"
                collected.extend(self._collect_urls(value, next_prefix))
        elif isinstance(payload, str) and payload.startswith(("http://", "https://")):
            collected.append((prefix or "asset", payload))

        return collected

    def download_assets(
        self,
        world: Dict[str, Any],
        output_dir: str,
    ) -> List[str]:
        """Download assets for WorldLabsPipeline."""
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")

        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        saved_paths: List[str] = []
        seen_urls = set()
        for key, url in self._collect_urls(world):
            if url in seen_urls:
                continue
            seen_urls.add(url)

            suffix = Path(url.split("?", maxsplit=1)[0]).suffix or ".bin"
            safe_name = key.replace(".", "_").replace("[", "_").replace("]", "")
            save_path = output_root / f"{safe_name}{suffix}"
            saved_paths.append(self.synthesis_model.download_asset(url, str(save_path)))
        return saved_paths

    def __call__(
        self,
        prompt: Optional[str] = None,
        images: Optional[Union[Image.Image, str, Dict[str, Any], List[Union[Image.Image, str, Dict[str, Any]]]]] = None,
        video: Optional[Union[str, Dict[str, Any]]] = None,
        prompt_type: str = "auto",
        image_azimuths: Optional[List[float]] = None,
        disable_recaption: Optional[bool] = None,
        model: str = "marble-1.1",
        display_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        wait: bool = True,
        poll_interval: int = 10,
        max_retries: int = 120,
        output_path: Optional[str] = None,
        download_assets: bool = False,
        assets_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute the complete pipeline generation flow."""
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")
        if self.operator is None:
            raise ValueError("Operator is not initialized")
        _resolve_api_key(self.api_key)

        processed_data = self.process(
            prompt=prompt,
            images=images,
            video=video,
            prompt_type=prompt_type,
            image_azimuths=image_azimuths,
            disable_recaption=disable_recaption,
            **kwargs,
        )

        world_prompt = self._materialize_world_prompt(
            processed_data["world_prompt"],
            metadata=metadata,
        )

        if processed_data["prompt"]:
            world_prompt["text_prompt"] = processed_data["prompt"]

        response = self.synthesis_model.generate_world(
            world_prompt=world_prompt,
            model=model,
            display_name=display_name,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

        result: Dict[str, Any] = {
            "response": response,
            "operation_id": self._extract_operation_id(response),
            "world_prompt": world_prompt,
        }

        operation = None
        if wait and result["operation_id"]:
            operation = self._poll_operation(
                operation_id=result["operation_id"],
                poll_interval=poll_interval,
                max_retries=max_retries,
            )
            result["operation"] = operation

        world = None
        world_id = None
        if operation is not None:
            world = self._extract_world_object(operation)
            if world is not None:
                world_id = self._extract_world_id(world)

        if world_id:
            result["world"] = self.synthesis_model.get_world(world_id)
            result["world_id"] = world_id
        elif world is not None:
            result["world"] = world
            result["world_id"] = self._extract_world_id(world)

        if output_path and result.get("world") is not None:
            result["output_path"] = self.synthesis_model.save_json(
                result["world"],
                output_path,
            )

        if download_assets and result.get("world") is not None:
            target_dir = assets_dir or str(
                Path(output_path).with_suffix("") if output_path else Path(_default_output_dir())
            )
            result["asset_paths"] = self.download_assets(result["world"], target_dir)

        return result

    def get_operator(self) -> Optional[WorldLabsOperator]:
        """Get operator for WorldLabsPipeline."""
        return self.operator

    def get_synthesis_model(self) -> Optional[WorldLabsSynthesis]:
        """Get synthesis model for WorldLabsPipeline."""
        return self.synthesis_model
