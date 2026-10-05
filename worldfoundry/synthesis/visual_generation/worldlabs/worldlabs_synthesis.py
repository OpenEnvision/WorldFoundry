"""Client for the World Labs / Marble world-generation API."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..api_video_client import REQUEST_TIMEOUT, ApiVideoSynthesis


class WorldLabsSynthesis(ApiVideoSynthesis):
    """Upload media assets to and generate 3D worlds through the Marble API."""

    DEFAULT_ENDPOINT = "https://api.worldlabs.ai"
    AUTH_HEADER = "WLT-Api-Key"
    AUTH_VALUE = "{api_key}"

    #: Marble returns 3D assets rather than videos; the streamed download is the same.
    download_asset = ApiVideoSynthesis.download_video

    def prepare_upload(
        self,
        file_name: str,
        kind: str,
        extension: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Reserve an upload slot and return the pre-signed ``upload_info``.

        Args:
            file_name: Name to register the asset under.
            kind: Asset kind, e.g. ``"IMAGE"`` or ``"3D_MODEL"``.
            extension: File extension; inferred by the API when omitted.
            metadata: Arbitrary metadata to attach to the asset.
        """
        payload: Dict[str, Any] = {"file_name": file_name, "kind": kind}
        if extension is not None:
            payload["extension"] = extension
        if metadata is not None:
            payload["metadata"] = metadata
        return self._post("/marble/v1/media-assets:prepare_upload", payload)

    def upload_file(
        self,
        file_path: str,
        upload_url: str,
        required_headers: Optional[Dict[str, str]] = None,
        method: str = "PUT",
    ) -> None:
        """Send the file at ``file_path`` to a pre-signed ``upload_url``."""
        path = Path(file_path)
        headers = dict(required_headers or {})
        headers["Content-Type"] = mimetypes.guess_type(str(path))[0] or "application/octet-stream"

        with path.open("rb") as file:
            response = self._requests().request(
                method=method or "PUT",
                url=upload_url,
                headers=headers,
                data=file,
                timeout=REQUEST_TIMEOUT,
            )
        response.raise_for_status()

    def upload_media_asset(
        self,
        file_path: str,
        kind: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Reserve a slot, upload the file, and return the new media asset ID.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist locally.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Local file not found: {file_path}")

        prepare_response = self.prepare_upload(
            file_name=path.name,
            kind=kind,
            extension=path.suffix.lstrip(".") or None,
            metadata=metadata,
        )
        upload_info = prepare_response["upload_info"]
        self.upload_file(
            file_path=str(path),
            upload_url=upload_info["upload_url"],
            required_headers=upload_info.get("required_headers", {}),
            method=upload_info.get("upload_method", "PUT"),
        )
        media_asset = prepare_response["media_asset"]
        return media_asset.get("media_asset_id") or media_asset["id"]

    def generate_world(
        self,
        world_prompt: Dict[str, Any],
        model: str = "marble-1.1",
        display_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Start a world generation and return the long-running ``Operation``."""
        payload: Dict[str, Any] = {"model": model, "world_prompt": world_prompt}
        if display_name is not None:
            payload["display_name"] = display_name
        if tags is not None:
            payload["tags"] = tags
        if metadata is not None:
            payload["metadata"] = metadata
        payload.update(kwargs)
        return self._post("/marble/v1/worlds:generate", payload)

    def get_operation(self, operation_id: str) -> Dict[str, Any]:
        """Return the status and result of a long-running operation."""
        return self._get(f"/marble/v1/operations/{operation_id}", headers=self._auth_headers())

    def get_world(self, world_id: str) -> Dict[str, Any]:
        """Return a generated world's properties and assets."""
        return self._get(f"/marble/v1/worlds/{world_id}", headers=self._auth_headers())

    @staticmethod
    def save_json(payload: Dict[str, Any], save_path: str) -> str:
        """Write ``payload`` as indented UTF-8 JSON and return the path."""
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        return str(output_path)


__all__ = ["WorldLabsSynthesis"]
