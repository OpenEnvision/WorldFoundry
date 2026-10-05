from __future__ import annotations

import base64
import gc
import mimetypes
import os
import re
import tempfile
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np

from model.openrouter import clean_json_content, parse_json_content


DEFAULT_QWEN_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_LOCAL_MODEL_PATH = Path(__file__).resolve().parent.parent / "weights" / "QwenVL"
DEFAULT_MAX_FRAMES = 8
DEFAULT_MAX_IMAGE_SIDE = 1024
DEFAULT_MAX_NEW_TOKENS = 1024

_MODEL_CACHE: Dict[tuple, Dict[str, Any]] = {}
_MODEL_LOCK = threading.Lock()


def clear_qwenvl_model_cache(*, empty_cuda_cache: bool = True) -> int:
    """Drop cached local QwenVL model bundles in the current Python process."""
    with _MODEL_LOCK:
        cache_size = len(_MODEL_CACHE)
        _MODEL_CACHE.clear()

    gc.collect()
    if empty_cuda_cache:
        try:
            import torch

            if _torch_cuda_available(torch):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    return cache_size


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _resolve_model_path(model_name: str | None) -> str:
    requested = (model_name or os.getenv("QWENVL_MODEL_PATH") or DEFAULT_QWEN_MODEL_NAME).strip()
    if DEFAULT_LOCAL_MODEL_PATH.exists():
        if requested in {DEFAULT_QWEN_MODEL_NAME, "Qwen3-VL-8B-Instruct"}:
            return str(DEFAULT_LOCAL_MODEL_PATH)
        if "Qwen3-VL-8B-Instruct" in requested and not Path(requested).exists():
            return str(DEFAULT_LOCAL_MODEL_PATH)
    if Path(requested).exists():
        return str(Path(requested))
    return requested


def _resolve_torch_dtype(torch_module: Any) -> Any:
    raw_dtype = os.getenv("QWENVL_DTYPE", "").strip().lower()
    if raw_dtype in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if raw_dtype in {"fp16", "float16", "half"}:
        return torch_module.float16
    if raw_dtype in {"fp32", "float32"}:
        return torch_module.float32

    if _torch_cuda_available(torch_module):
        bf16_supported = getattr(torch_module.cuda, "is_bf16_supported", lambda: False)
        return torch_module.bfloat16 if bf16_supported() else torch_module.float16
    return torch_module.float32


def _torch_cuda_available(torch_module: Any) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return bool(torch_module.cuda.is_available())
        except Exception:
            return False


def _load_model_bundle(model_name: str | None) -> Dict[str, Any]:
    resolved_model_path = _resolve_model_path(model_name)
    display_model_name = model_name or DEFAULT_QWEN_MODEL_NAME

    try:
        import torch
    except ImportError as exc:
        raise ImportError("QwenVL local inference requires torch.") from exc

    model_dtype = _resolve_torch_dtype(torch)
    device_map = os.getenv("QWENVL_DEVICE_MAP", "").strip()
    explicit_device = os.getenv("QWENVL_DEVICE", "").strip()
    if not explicit_device:
        explicit_device = "cuda" if _torch_cuda_available(torch) else "cpu"
    attn_implementation = os.getenv("QWENVL_ATTN_IMPLEMENTATION", "").strip()

    cache_key = (
        resolved_model_path,
        str(model_dtype),
        device_map or None,
        explicit_device,
        attn_implementation or None,
    )

    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "QwenVL requires `transformers` with Qwen3-VL support. "
                "Install a recent build, for example: "
                "`pip install git+https://github.com/huggingface/transformers`"
            ) from exc

        load_kwargs: Dict[str, Any] = {"torch_dtype": model_dtype}
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        if device_map:
            load_kwargs["device_map"] = device_map

        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(resolved_model_path, **load_kwargs)
            processor = AutoProcessor.from_pretrained(resolved_model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load QwenVL model from {resolved_model_path}. "
                "Make sure the local weights are complete and your transformers version supports Qwen3-VL."
            ) from exc

        if not device_map:
            model = model.to(explicit_device)
        model.eval()

        bundle = {
            "model": model,
            "processor": processor,
            "resolved_model_path": resolved_model_path,
            "display_model_name": display_model_name,
        }
        _MODEL_CACHE[cache_key] = bundle
        return bundle


def _model_device(model: Any) -> Any:
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("QwenVL model has no parameters.") from exc


def _decode_data_url(data_url: str) -> tuple[str, bytes]:
    if not data_url.startswith("data:"):
        raise ValueError("Expected a data URL.")
    header, payload = data_url.split(",", 1)
    mime_type = header[5:].split(";", 1)[0] or "application/octet-stream"
    if ";base64" not in header:
        raise ValueError("Only base64-encoded data URLs are supported.")
    return mime_type, base64.b64decode(payload)


def _guess_suffix_from_mime(mime_type: str, default_suffix: str) -> str:
    suffix = mimetypes.guess_extension(mime_type) or default_suffix
    if suffix == ".jpe":
        return ".jpg"
    return suffix


def _resize_image(image: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return image
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image
    scale = max_side / float(longest_side)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def _save_image_to_temp(image: np.ndarray, temp_dir: Path, prefix: str = "image") -> str:
    max_side = _env_int("QWENVL_MAX_IMAGE_SIDE", DEFAULT_MAX_IMAGE_SIDE)
    output_path = temp_dir / f"{prefix}_{uuid.uuid4().hex}.jpg"
    resized = _resize_image(image, max_side=max_side)
    ok, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Failed to encode image for QwenVL input.")
    output_path.write_bytes(buffer.tobytes())
    return str(output_path)


def _materialize_raw_base64_image(payload: str, temp_dir: Path) -> str:
    try:
        image_bytes = base64.b64decode(payload)
    except Exception as exc:
        raise ValueError("Failed to decode base64 image payload.") from exc
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Decoded base64 payload is not a valid image.")
    return _save_image_to_temp(image, temp_dir, prefix="frame")


def _is_probable_base64_blob(text: str) -> bool:
    stripped = (text or "").strip()
    if len(stripped) < 128 or len(stripped) % 4 != 0:
        return False
    return re.fullmatch(r"[A-Za-z0-9+/=\s]+", stripped) is not None


def _materialize_image_source(source: Any, temp_dir: Path) -> str:
    if isinstance(source, dict):
        source = source.get("url") or source.get("image") or source.get("path")

    if not isinstance(source, str):
        raise TypeError(f"Unsupported image source type: {type(source)}")

    if source.startswith("data:image/"):
        mime_type, image_bytes = _decode_data_url(source)
        _ = mime_type
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image data URL.")
        return _save_image_to_temp(image, temp_dir, prefix="image")

    local_path = Path(source)
    if local_path.exists():
        image = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read local image: {local_path}")
        return _save_image_to_temp(image, temp_dir, prefix=local_path.stem or "image")

    if source.startswith(("http://", "https://")):
        raise ValueError("Remote image URLs are not supported by the local QwenVL interface.")

    if re.fullmatch(r"[A-Za-z0-9+/=\s]+", source or ""):
        return _materialize_raw_base64_image(source, temp_dir)

    raise FileNotFoundError(f"Image source does not exist: {source}")


def _materialize_video_source(source: Any, temp_dir: Path) -> str:
    if isinstance(source, dict):
        source = source.get("url") or source.get("video") or source.get("path")

    if not isinstance(source, str):
        raise TypeError(f"Unsupported video source type: {type(source)}")

    if source.startswith("data:video/"):
        mime_type, video_bytes = _decode_data_url(source)
        suffix = _guess_suffix_from_mime(mime_type, ".mp4")
        output_path = temp_dir / f"video_{uuid.uuid4().hex}{suffix}"
        output_path.write_bytes(video_bytes)
        return str(output_path)

    local_path = Path(source)
    if local_path.exists():
        return str(local_path)

    if source.startswith(("http://", "https://")):
        raise ValueError("Remote video URLs are not supported by the local QwenVL interface.")

    raise FileNotFoundError(f"Video source does not exist: {source}")


def _sample_frame_indices(total_frames: int, max_frames: int) -> List[int]:
    if total_frames <= 0:
        return []
    if max_frames <= 1 or total_frames == 1:
        return [0]
    if total_frames <= max_frames:
        return list(range(total_frames))

    positions = [
        int(round(index * (total_frames - 1) / float(max_frames - 1)))
        for index in range(max_frames)
    ]

    deduplicated: List[int] = []
    seen = set()
    for position in positions:
        if position not in seen:
            deduplicated.append(position)
            seen.add(position)
    return deduplicated


def _sample_video_to_qwen_images(video_path: str, temp_dir: Path) -> List[Dict[str, str]]:
    max_frames = _env_int("QWENVL_MAX_FRAMES", DEFAULT_MAX_FRAMES)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_indices = _sample_frame_indices(total_frames, max_frames)

    image_items: List[Dict[str, str]] = []
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        image_path = _save_image_to_temp(frame, temp_dir, prefix="video_frame")
        image_items.append({"type": "image", "image": image_path})

    capture.release()

    if not image_items:
        raise RuntimeError(f"Failed to sample frames from video: {video_path}")
    return image_items


def _media_item_to_qwen_content(item: Dict[str, Any], temp_dir: Path) -> List[Dict[str, str]]:
    item_type = item.get("type")

    if item_type == "text" or "text" in item:
        return [{"type": "text", "text": str(item.get("text", ""))}]

    if item_type == "video_url" or "video_url" in item:
        source = item.get("video_url")
        if isinstance(source, dict):
            source = source.get("url")
        video_path = _materialize_video_source(source, temp_dir)
        return _sample_video_to_qwen_images(video_path, temp_dir)

    if item_type == "video" or "video" in item:
        video_path = _materialize_video_source(item.get("video"), temp_dir)
        return _sample_video_to_qwen_images(video_path, temp_dir)

    if item_type == "image_url" or "image_url" in item:
        source = item.get("image_url")
        if isinstance(source, dict):
            source = source.get("url")
        image_path = _materialize_image_source(source, temp_dir)
        return [{"type": "image", "image": image_path}]

    if item_type == "image" or "image" in item:
        image_path = _materialize_image_source(item.get("image"), temp_dir)
        return [{"type": "image", "image": image_path}]

    return [{"type": "text", "text": str(item)}]


def _normalize_content(content: Any, temp_dir: Path) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        if content.startswith("data:image/"):
            image_path = _materialize_image_source(content, temp_dir)
            return [{"type": "image", "image": image_path}]
        if content.startswith("data:video/"):
            video_path = _materialize_video_source(content, temp_dir)
            return _sample_video_to_qwen_images(video_path, temp_dir)
        if _is_probable_base64_blob(content):
            try:
                image_path = _materialize_raw_base64_image(content, temp_dir)
                return [{"type": "image", "image": image_path}]
            except Exception:
                pass
        return content

    if isinstance(content, dict):
        return _media_item_to_qwen_content(content, temp_dir)

    if isinstance(content, Sequence):
        normalized_items: List[Dict[str, str]] = []
        for item in content:
            if isinstance(item, str):
                if item.startswith("data:image/"):
                    image_path = _materialize_image_source(item, temp_dir)
                    normalized_items.append({"type": "image", "image": image_path})
                elif item.startswith("data:video/"):
                    video_path = _materialize_video_source(item, temp_dir)
                    normalized_items.extend(_sample_video_to_qwen_images(video_path, temp_dir))
                elif _is_probable_base64_blob(item):
                    try:
                        image_path = _materialize_raw_base64_image(item, temp_dir)
                        normalized_items.append({"type": "image", "image": image_path})
                    except Exception:
                        normalized_items.append({"type": "text", "text": item})
                else:
                    normalized_items.append({"type": "text", "text": item})
                continue

            if isinstance(item, dict):
                normalized_items.extend(_media_item_to_qwen_content(item, temp_dir))
                continue

            normalized_items.append({"type": "text", "text": str(item)})
        return normalized_items

    return str(content)


def _flatten_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        return str(content)
    if isinstance(content, Sequence):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                chunks.append(str(item["text"]))
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(part for part in chunks if part)
    return str(content)


def _normalize_messages(messages: List[Dict[str, Any]], temp_dir: Path) -> List[Dict[str, Any]]:
    normalized_messages: List[Dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = _normalize_content(message.get("content", ""), temp_dir)

        if isinstance(content, str):
            normalized_content: Any = [{"type": "text", "text": content}]
        elif isinstance(content, dict):
            normalized_content = [content]
        elif isinstance(content, Sequence):
            normalized_content = list(content)
        else:
            normalized_content = [{"type": "text", "text": _flatten_text_content(content)}]

        # Hugging Face multimodal chat templates expect each message `content`
        # to be a list of typed items, including system prompts.
        if not normalized_content:
            normalized_content = [{"type": "text", "text": ""}]

        normalized_messages.append({"role": role, "content": normalized_content})
    return normalized_messages


def _move_batch_to_device(batch: Any, device: Any) -> Any:
    if hasattr(batch, "to"):
        try:
            return batch.to(device)
        except Exception:
            pass
    if isinstance(batch, dict):
        moved = {}
        for key, value in batch.items():
            moved[key] = value.to(device) if hasattr(value, "to") else value
        return moved
    return batch


def _generation_kwargs() -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "max_new_tokens": _env_int("QWENVL_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS),
        "do_sample": _env_bool("QWENVL_DO_SAMPLE", False),
    }
    if kwargs["do_sample"]:
        kwargs["temperature"] = float(os.getenv("QWENVL_TEMPERATURE", "0.7"))
        kwargs["top_p"] = float(os.getenv("QWENVL_TOP_P", "0.8"))
        kwargs["top_k"] = int(os.getenv("QWENVL_TOP_K", "20"))
    return kwargs


def _build_response(
    content: str,
    model_name: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> Dict[str, Any]:
    response: Dict[str, Any] = {
        "id": f"qwenvl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    if prompt_tokens is not None and completion_tokens is not None:
        response["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    return response


def chat_qwenvl_call(
    messages: List[Dict],
    model_name: str = DEFAULT_QWEN_MODEL_NAME,
    timeout: int = 240,
    max_retries: int = 1,
) -> Dict:
    """
    Local Qwen3-VL chat interface compatible with `chat_openrouter_call`.

    Notes:
    - It accepts the same OpenRouter-style message format used in this repo.
    - Video inputs are converted to evenly sampled frames and sent as images,
      which avoids extra runtime dependencies such as decord/av/qwen_vl_utils.
    """
    _ = timeout
    last_error: Exception | None = None

    for attempt in range(max_retries):
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="qwenvl_")
        try:
            bundle = _load_model_bundle(model_name)
            normalized_messages = _normalize_messages(messages, Path(temp_dir_obj.name))
            processor = bundle["processor"]
            model = bundle["model"]

            try:
                inputs = processor.apply_chat_template(
                    normalized_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to build QwenVL inputs from messages. "
                    "Your transformers version may be too old for Qwen3-VL chat templates."
                ) from exc

            model_device = _model_device(model)
            inputs = _move_batch_to_device(inputs, model_device)

            input_ids = inputs["input_ids"]
            prompt_tokens = int(input_ids.shape[-1])

            with __import__("torch").no_grad():
                generated_ids = model.generate(**inputs, **_generation_kwargs())

            generated_ids_trimmed = [
                output_ids[len(input_row):]
                for input_row, output_ids in zip(input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            content = output_text[0] if output_text else ""
            completion_tokens = int(generated_ids_trimmed[0].shape[-1]) if generated_ids_trimmed else 0
            return _build_response(
                content=content,
                model_name=bundle["display_model_name"],
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                continue
        finally:
            temp_dir_obj.cleanup()

    raise RuntimeError(f"QwenVL request failed after {max_retries} attempt(s): {last_error}") from last_error


def text_qwenvl_call(
    system_prompt: str,
    user_content: str,
    model_name: str = DEFAULT_QWEN_MODEL_NAME,
) -> Dict:
    """Text-only helper aligned with `text_openrouter_call`."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return chat_qwenvl_call(messages=messages, model_name=model_name, timeout=180, max_retries=1)


def video_qwenvl_call(
    data_url: Any,
    system_prompt: str,
    user_content: str,
    model_name: str = DEFAULT_QWEN_MODEL_NAME,
) -> Dict:
    """
    Multimodal helper aligned with `video_openrouter_call`.

    `data_url` can be:
    - a single OpenRouter-style `video_url` item
    - a single local video/data URL string
    - a list of OpenRouter-style image/frame items
    """
    if not isinstance(data_url, list):
        content: List[Any] = [{"type": "text", "text": user_content}, data_url]
    else:
        content = [{"type": "text", "text": user_content}]
        if data_url:
            content.append({"type": "text", "text": "These are the frames extracted from the video."})
            content.extend(data_url)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    return chat_qwenvl_call(messages=messages, model_name=model_name, timeout=240, max_retries=1)


__all__ = [
    "DEFAULT_QWEN_MODEL_NAME",
    "clean_json_content",
    "parse_json_content",
    "text_qwenvl_call",
    "video_qwenvl_call",
    "chat_qwenvl_call",
]
