"""First-frame image generation from T2I prompts."""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class T2IProvider(Protocol):
    def generate(self, *, prompt: str, family_id: str, out_path: Path) -> dict[str, Any]: ...


@dataclass
class FirstFrameManifest:
    family_id: str
    prompt: str
    image_path: str
    provider: str
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "prompt": self.prompt,
            "image_path": self.image_path,
            "provider": self.provider,
            "model": self.model,
            "metadata": self.metadata,
        }


def _require_httpx():
    try:
        import httpx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "First-frame generation requires httpx. Install with: pip install 'wrbench[firstframe]'"
        ) from exc


def _read_config_value(value: str | None) -> str | None:
    if value is not None:
        text = value.strip()
        if text:
            return text
    return None


def _require_config_value(value: str | None, *, label: str) -> str:
    resolved = _read_config_value(value)
    if resolved is not None:
        return resolved
    raise RuntimeError(f"{label} required")


def _require_config_int(value: int | str | None, *, label: str) -> int:
    if value is not None:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{label} must be an integer") from exc
        if parsed <= 0:
            raise RuntimeError(f"{label} must be positive")
        return parsed
    raise RuntimeError(f"{label} required")


def _resolve_manifest_value(
    value: str | None,
    *,
    label: str,
    metadata: dict[str, Any],
    provider_value: str | None = None,
) -> str:
    resolved = _read_config_value(value)
    if resolved is not None:
        return resolved
    meta_value = metadata.get(label)
    if isinstance(meta_value, str) and meta_value.strip():
        return meta_value.strip()
    if provider_value is not None and provider_value.strip():
        return provider_value.strip()
    raise RuntimeError(
        f"{label} required: pass it explicitly or use a provider/test double that reports it"
    )


class DashScopeT2IProvider:
    """Generate first frames via DashScope multimodal-generation API."""

    provider_name = "dashscope"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        size: str | None = None,
        n: int | str | None = None,
        timeout: float = 180.0,
    ) -> None:
        _require_httpx()
        import httpx

        self.model = _require_config_value(model, label="DashScope first-frame model")
        self.api_key = _require_config_value(api_key, label="DashScope API key")
        self.endpoint = _require_config_value(
            endpoint,
            label="DashScope first-frame endpoint",
        )
        self.size = _require_config_value(size, label="DashScope first-frame size")
        self.n = _require_config_int(n, label="DashScope first-frame n")
        self._client = httpx.Client(timeout=timeout)

    def generate(self, *, prompt: str, family_id: str, out_path: Path) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "input": {"messages": [{"role": "user", "content": [{"text": prompt}]}]},
            "parameters": {"size": self.size, "n": self.n},
        }
        resp = self._client.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        urls = _extract_image_urls(data)
        if not urls:
            b64 = _extract_image_b64(data)
            if b64:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(base64.b64decode(b64))
                return {"source": "base64", "provider": self.provider_name, "model": self.model}
            raise RuntimeError(f"No image in DashScope response: {data}")
        # Download first URL
        import httpx

        img_resp = self._client.get(urls[0])
        img_resp.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(img_resp.content)
        return {"source": "url", "url": urls[0], "provider": self.provider_name, "model": self.model}


class AtlasCloudT2IProvider:
    """Generate first frames through the Atlas Cloud asynchronous image API."""

    provider_name = "atlascloud"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        size: str | None = None,
        n: int | str | None = None,
        timeout: float = 180.0,
        poll_interval: float = 2.0,
    ) -> None:
        _require_httpx()
        import httpx

        self.model = _require_config_value(model, label="Atlas Cloud first-frame model")
        self.api_key = _require_config_value(api_key, label="Atlas Cloud API key")
        self.endpoint = _require_config_value(
            endpoint,
            label="Atlas Cloud API endpoint",
        ).rstrip("/")
        self.size = _require_config_value(size, label="Atlas Cloud first-frame size")
        self.n = _require_config_int(n, label="Atlas Cloud first-frame n")
        if self.n != 1:
            raise RuntimeError("Atlas Cloud first-frame n must be 1")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._client = httpx.Client(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _submit_url(self) -> str:
        if self.endpoint.endswith("/model/generateImage"):
            return self.endpoint
        return f"{self.endpoint}/model/generateImage"

    def _prediction_url(self, prediction_id: str) -> str:
        api_root = self._submit_url().removesuffix("/model/generateImage")
        return f"{api_root}/model/prediction/{prediction_id}"

    @staticmethod
    def _prediction_data(payload: dict[str, Any]) -> dict[str, Any]:
        data = payload.get("data")
        return data if isinstance(data, dict) else payload

    def generate(self, *, prompt: str, family_id: str, out_path: Path) -> dict[str, Any]:
        del family_id
        payload = {
            "model": self.model,
            "prompt": prompt,
            "size": self.size,
            "output_format": "png",
        }
        response = self._client.post(self._submit_url(), headers=self._headers(), json=payload)
        response.raise_for_status()
        submitted = self._prediction_data(response.json())
        prediction_id = submitted.get("id")
        if not isinstance(prediction_id, str) or not prediction_id:
            raise RuntimeError("Atlas Cloud submission returned no prediction id")

        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            result = self._client.get(
                self._prediction_url(prediction_id),
                headers=self._headers(),
            )
            result.raise_for_status()
            prediction = self._prediction_data(result.json())
            status = str(prediction.get("status") or "").lower()
            if status in {"completed", "succeeded"}:
                outputs = prediction.get("outputs") or []
                first = outputs[0] if outputs else None
                output_url = first if isinstance(first, str) else first.get("url") if isinstance(first, dict) else None
                if not output_url:
                    raise RuntimeError("Atlas Cloud prediction returned no output URL")
                image = self._client.get(output_url)
                image.raise_for_status()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(image.content)
                return {
                    "source": "url",
                    "url": output_url,
                    "provider": self.provider_name,
                    "model": self.model,
                    "prediction_id": prediction_id,
                }
            if status in {"failed", "canceled"}:
                raise RuntimeError(f"Atlas Cloud prediction {status}: {prediction.get('error') or 'unknown error'}")
            time.sleep(self.poll_interval)

        raise TimeoutError(f"Atlas Cloud prediction {prediction_id} timed out")


class MockT2IProvider:
    """Write a minimal PNG placeholder for tests (1x1 transparent)."""
    provider_name = "mock"

    def __init__(self, *, model: str = "mock") -> None:
        self.model = model

    def generate(self, *, prompt: str, family_id: str, out_path: Path) -> dict[str, Any]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Minimal valid PNG (1x1)
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
        out_path.write_bytes(png)
        return {"source": "mock", "provider": self.provider_name, "model": self.model, "prompt_len": len(prompt)}


def get_t2i_provider(name: str | None = None, **kwargs: Any) -> T2IProvider:
    provider = _require_config_value(name, label="First-frame provider").lower()
    if provider == "mock":
        model = _require_config_value(kwargs.pop("model", None), label="Mock first-frame model")
        kwargs.pop("api_key", None)
        return MockT2IProvider(model=model)
    if provider in {"dashscope", "wan"}:
        return DashScopeT2IProvider(**kwargs)
    if provider in {"atlascloud", "atlas_cloud"}:
        return AtlasCloudT2IProvider(**kwargs)
    raise ValueError(f"Unknown T2I provider {provider!r}; expected atlascloud, dashscope, or mock")


def generate_first_frame(
    *,
    family_id: str,
    prompt: str,
    out_dir: str | Path,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    endpoint: str | None = None,
    size: str | None = None,
    n: int | str | None = None,
    t2i: T2IProvider | None = None,
) -> FirstFrameManifest:
    """Generate ``{family_id}.png`` under *out_dir* and return manifest."""
    out_dir = Path(out_dir)
    image_path = out_dir / f"{family_id}.png"
    if t2i is None:
        provider_name = _require_config_value(provider, label="First-frame provider")
        model_name = _require_config_value(model, label="First-frame model")
        t2i = get_t2i_provider(provider_name, model=model_name, api_key=api_key, endpoint=endpoint, size=size, n=n)
    else:
        provider_name = provider
        model_name = model
    meta = t2i.generate(prompt=prompt, family_id=family_id, out_path=image_path)
    provider_name = _resolve_manifest_value(
        provider_name,
        label="provider",
        metadata=meta,
        provider_value=getattr(t2i, "provider_name", None),
    )
    model_name = _resolve_manifest_value(
        model_name,
        label="model",
        metadata=meta,
        provider_value=getattr(t2i, "model", None),
    )
    return FirstFrameManifest(
        family_id=family_id,
        prompt=prompt,
        image_path=str(image_path),
        provider=provider_name,
        model=str(model_name),
        metadata=meta,
    )


def generate_first_frames_from_families(
    families: list[dict[str, Any]],
    *,
    out_dir: str | Path,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    endpoint: str | None = None,
    size: str | None = None,
    n: int | str | None = None,
    t2i: T2IProvider | None = None,
    skip_existing: bool,
) -> list[FirstFrameManifest]:
    """Generate first frames for families that have ``t2i_scene``."""
    manifests: list[FirstFrameManifest] = []
    out_dir = Path(out_dir)
    for family in families:
        family_id = str(family["family_id"])
        prompt = str(family.get("t2i_scene") or "").strip()
        if not prompt:
            continue
        image_path = out_dir / f"{family_id}.png"
        if skip_existing and image_path.is_file():
            manifests.append(
                FirstFrameManifest(
                    family_id=family_id,
                    prompt=prompt,
                    image_path=str(image_path),
                    provider=_resolve_manifest_value(
                        provider,
                        label="provider",
                        metadata={},
                        provider_value=getattr(t2i, "provider_name", None),
                    ),
                    model=_resolve_manifest_value(
                        model,
                        label="model",
                        metadata={},
                        provider_value=getattr(t2i, "model", None),
                    ),
                    metadata={"skipped": True},
                )
            )
            continue
        manifests.append(
            generate_first_frame(
                family_id=family_id,
                prompt=prompt,
                out_dir=out_dir,
                provider=provider,
                model=model,
                api_key=api_key,
                endpoint=endpoint,
                size=size,
                n=n,
                t2i=t2i,
            )
        )
    return manifests


def write_manifest(path: str | Path, manifests: list[FirstFrameManifest]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [m.to_dict() for m in manifests]
    p.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def _extract_image_urls(data: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    output = data.get("output") or {}
    for choice in output.get("choices") or []:
        message = choice.get("message") or {}
        for item in message.get("content") or []:
            if isinstance(item, dict) and item.get("image"):
                urls.append(str(item["image"]))
            if isinstance(item, dict) and item.get("url"):
                urls.append(str(item["url"]))
    return urls


def _extract_image_b64(data: dict[str, Any]) -> str | None:
    output = data.get("output") or {}
    for choice in output.get("choices") or []:
        message = choice.get("message") or {}
        for item in message.get("content") or []:
            if isinstance(item, dict) and item.get("image_base64"):
                return str(item["image_base64"])
    return None
