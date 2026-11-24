"""Shared Echo 1.5 R2V request contract for CLI and scheduled local inference."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

R2V_SCHEMA_VERSION = "echo15.r2v.v1"
MAX_MEMORY_SLOTS = 7


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _resolve_resource(value: Any, *, base_dir: Path, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty path or URL")
    source = value.strip()
    parsed = urlparse(source)
    if parsed.scheme == "data":
        header, separator, encoded = source.partition(",")
        if not separator or ";base64" not in header.lower():
            raise ValueError(f"{field_name} must use a base64 data URL")
        try:
            content = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"{field_name} has invalid base64 data") from exc
        if not content:
            raise ValueError(f"{field_name} data URL must not be empty")

        resource_dir = base_dir / "inline_resources"
        resource_dir.mkdir(parents=True, exist_ok=True)
        resource_path = resource_dir / f"{hashlib.sha256(content).hexdigest()}.bin"
        if not resource_path.exists():
            try:
                with resource_path.open("xb") as handle:
                    handle.write(content)
            except FileExistsError:
                pass
        return str(resource_path.resolve())
    if parsed.scheme in {"http", "https"}:
        return source
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError(f"{field_name} does not support remote file URLs")
        path = Path(unquote(parsed.path))
    elif parsed.scheme:
        raise ValueError(f"{field_name} must use HTTP(S), file://, or a local path")
    else:
        path = Path(source).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


@dataclass(frozen=True)
class R2VMemorySlot:
    """One ordered production-compatible memory slot."""

    image_url: str | None = None
    audio_url: str | None = None
    audio_mode: str | None = None
    shot_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None and value != {}}


@dataclass(frozen=True)
class R2VRequest:
    """A fully resolved R2V request consumed by conditioning and inference."""

    work_id: str
    shot_id: str
    prompt: str
    memory_slots: tuple[R2VMemorySlot, ...]
    condition_img: str | None
    num_frames: int
    width: int
    height: int
    seed: int
    duration_sec: float | None = None
    source_path: Path | None = field(default=None, compare=False)
    request_sha256: str | None = field(default=None, compare=False)

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "work_id": self.work_id,
            "shot_id": self.shot_id,
            "prompt": self.prompt,
            "condition_img": self.condition_img,
            "memory_slots": [slot.as_payload() for slot in self.memory_slots],
            "num_frames": self.num_frames,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
        }
        if self.duration_sec is not None:
            payload["duration_sec"] = self.duration_sec
        return payload


def normalize_memory_slots(
    values: Any,
    *,
    base_dir: Path,
    resolve_resources: bool,
) -> tuple[R2VMemorySlot, ...]:
    if not isinstance(values, list):
        raise ValueError("memory_slots must be a list")
    if len(values) > MAX_MEMORY_SLOTS:
        raise ValueError(f"memory_slots supports at most {MAX_MEMORY_SLOTS} entries")

    slots: list[R2VMemorySlot] = []
    for index, raw in enumerate(values):
        if not isinstance(raw, dict):
            raise ValueError(f"memory_slots[{index}] must be an object")
        shot_id = str(raw.get("shot_id") or "").strip() or None
        image_url = str(raw.get("image_url") or "").strip() or None
        image_mode = str(raw.get("image_mode") or "").strip().lower() or None
        audio_url = str(raw.get("audio_url") or "").strip() or None
        audio_mode = str(raw.get("audio_mode") or "").strip().lower() or None
        metadata = raw.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"memory_slots[{index}].metadata must be an object")
        if image_mode:
            raise ValueError(f"memory_slots[{index}].image_mode is not supported")
        if bool(shot_id) == bool(image_url):
            raise ValueError(f"memory_slots[{index}] requires exactly one of shot_id or image_url")
        if shot_id and (audio_url or audio_mode):
            raise ValueError(f"memory_slots[{index}].shot_id cannot be combined with audio fields")
        if audio_mode not in {None, "empty"}:
            raise ValueError(f"memory_slots[{index}].audio_mode only supports 'empty'")
        if audio_url and audio_mode == "empty":
            raise ValueError(f"memory_slots[{index}].audio_url conflicts with audio_mode='empty'")
        if image_url and not audio_url and audio_mode is None:
            audio_mode = "empty"
        if resolve_resources:
            image_url = _resolve_resource(image_url, base_dir=base_dir, field_name=f"memory_slots[{index}].image_url")
            audio_url = _resolve_resource(audio_url, base_dir=base_dir, field_name=f"memory_slots[{index}].audio_url")
        slots.append(
            R2VMemorySlot(
                shot_id=shot_id,
                image_url=image_url,
                audio_url=audio_url,
                audio_mode=audio_mode,
                metadata=dict(metadata),
            )
        )
    return tuple(slots)


def normalize_r2v_payload(
    payload: dict[str, Any],
    *,
    base_dir: str | Path = ".",
    default_num_frames: int = 241,
    default_width: int = 1280,
    default_height: int = 736,
    default_seed: int = 42,
    prompt_max_chars: int | None = 1500,
    resolve_resources: bool = True,
) -> R2VRequest:
    """Validate the online R2V schema and optionally resolve local resources."""

    if not isinstance(payload, dict):
        raise ValueError("R2V request must be a JSON object")
    if "payload" in payload and isinstance(payload["payload"], dict):
        legacy = payload["payload"]
        shot = legacy.get("shot")
        if not isinstance(shot, dict):
            raise ValueError("director envelope payload.shot must be an object")
        payload = {
            "work_id": legacy.get("work_id") or payload.get("job", {}).get("work_id"),
            "shot_id": shot.get("shot_key") or str(shot.get("shot_id") or ""),
            "prompt": shot.get("text"),
            "condition_img": legacy.get("condition_img"),
            "memory_slots": legacy.get("memory_slots", []),
            "num_frames": shot.get("num_frames"),
            "duration_sec": shot.get("duration_sec"),
            "width": shot.get("width"),
            "height": shot.get("height"),
            "seed": shot.get("seed"),
        }

    work_id = _required_text(payload, "work_id")
    shot_id = _required_text(payload, "shot_id")
    prompt = _required_text(payload, "prompt")
    if prompt_max_chars is not None:
        if prompt_max_chars <= 0:
            raise ValueError("prompt_max_chars must be positive")
        prompt = prompt[:prompt_max_chars]

    root = Path(base_dir).expanduser().resolve()
    slots = normalize_memory_slots(
        payload.get("memory_slots", []),
        base_dir=root,
        resolve_resources=resolve_resources,
    )
    condition_img = payload.get("condition_img")
    if resolve_resources:
        condition_img = _resolve_resource(condition_img, base_dir=root, field_name="condition_img")
    elif condition_img is not None:
        condition_img = str(condition_img).strip() or None

    num_frames = int(payload.get("num_frames") or default_num_frames)
    width = int(payload.get("width") or default_width)
    height = int(payload.get("height") or default_height)
    seed_value = payload.get("seed")
    seed = int(default_seed if seed_value is None else seed_value)
    duration_value = payload.get("duration_sec")
    duration_sec = float(duration_value) if duration_value is not None else None
    if min(num_frames, width, height) <= 0:
        raise ValueError("num_frames, width, and height must be positive")
    if duration_sec is not None and duration_sec <= 0:
        raise ValueError("duration_sec must be positive")

    return R2VRequest(
        work_id=work_id,
        shot_id=shot_id,
        prompt=prompt,
        condition_img=condition_img,
        memory_slots=slots,
        num_frames=num_frames,
        width=width,
        height=height,
        seed=seed,
        duration_sec=duration_sec,
    )


def load_r2v_request(
    path: str | Path,
    **defaults: Any,
) -> R2VRequest:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    request = normalize_r2v_payload(payload, base_dir=source.parent, **defaults)
    fingerprint = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return replace(request, source_path=source, request_sha256=fingerprint)
