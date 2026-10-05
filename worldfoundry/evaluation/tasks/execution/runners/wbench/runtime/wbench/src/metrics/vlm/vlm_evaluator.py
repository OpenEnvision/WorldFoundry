# -*- coding: utf-8 -*-
"""
VLM Evaluator - API client for video evaluation (video direct mode).

Supports two API formats:
- ARK (Volcano Engine Responses API) - default
- OpenAI-compatible (Chat Completions) - for custom endpoints

Video direct mode: encodes video as base64 and sends via video_url field,
avoiding payload-too-large issues with many image frames.

Supported models: doubao-seed-2-0-lite-260215, Doubao-Seed-2.0-lite
"""
import base64
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from io import BytesIO
from PIL import Image

import cv2
import requests

# Global request counter and concurrency limiter (thread-safe)
_request_counter = 0
_request_lock = threading.Lock()
_request_semaphore: Optional[threading.Semaphore] = None

logger = logging.getLogger(__name__)

ALLOWED_MODELS = {"doubao-seed-2-0-lite-260215", "Doubao-Seed-2.0-lite"}


def get_request_count() -> int:
    return _request_counter


def set_max_concurrent_requests(n: int):
    """Set global max concurrent API requests."""
    global _request_semaphore
    _request_semaphore = threading.Semaphore(n)


def _increment_request_count():
    global _request_counter
    with _request_lock:
        _request_counter += 1


def encode_video_to_b64url(video_path: str, max_short_side: int = 0) -> str:
    """Read video file and encode as data:video/mp4;base64,... URL.

    Args:
        video_path: Source video path.
        max_short_side: If >0, re-encode with short side <= this value.
    """
    if max_short_side <= 0:
        with open(video_path, "rb") as f:
            return "data:video/mp4;base64," + base64.b64encode(f.read()).decode()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = w, h
    if min(w, h) > max_short_side:
        scale = max_short_side / min(w, h)
        out_w = int(w * scale) // 2 * 2
        out_h = int(h * scale) // 2 * 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (out_w, out_h))
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if (out_w, out_h) != (w, h):
                frame = cv2.resize(frame, (out_w, out_h))
            writer.write(frame)
        writer.release()
        with open(tmp_path, "rb") as f:
            return "data:video/mp4;base64," + base64.b64encode(f.read()).decode()
    finally:
        cap.release()
        writer.release()
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


def clip_video_to_b64url(video_path: str, start_sec: float, end_sec: float,
                         max_short_side: int = 0) -> str:
    """Clip video segment and encode as base64 data URL.

    Args:
        video_path: Source video path.
        start_sec: Clip start time in seconds.
        end_sec: Clip end time in seconds.
        max_short_side: If >0, resize so the short side <= this value.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w, out_h = w, h
    if max_short_side > 0 and min(w, h) > max_short_side:
        scale = max_short_side / min(w, h)
        out_w = int(w * scale) // 2 * 2
        out_h = int(h * scale) // 2 * 2

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (out_w, out_h))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
        while True:
            pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if pos_sec > end_sec:
                break
            ret, frame = cap.read()
            if not ret:
                break
            if (out_w, out_h) != (w, h):
                frame = cv2.resize(frame, (out_w, out_h))
            writer.write(frame)
        writer.release()
        return encode_video_to_b64url(tmp_path)
    finally:
        cap.release()
        writer.release()
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


@dataclass
class VLMTask:
    """Single VLM evaluation task."""
    dimension: str
    question: str
    images: List[Image.Image] = field(default_factory=list)
    expected: str = "yes"
    metadata: Dict = field(default_factory=dict)
    answer: bool = None
    video_url: str = None


class VLMClient:
    """VLM API client with video direct mode.

    Auto-detects format based on api_url:
    - URL contains "chat/completions" -> OpenAI Chat Completions format
    - Otherwise -> ARK Responses API format (appends /responses)

    Supports video_url (base64-encoded video) as primary input mode.
    Falls back to image frames if video_url is not provided.
    """

    DEFAULT_API_URL = "https://ark.cn-beijing.volces.com/api/v3"
    DEFAULT_MODEL = "doubao-seed-2-0-lite-260215"

    def __init__(
        self,
        api_url: str = "",
        api_key: str = "",
        model_name: str = "",
        request_timeout: int = 300,
    ):
        self.api_url = api_url or os.environ.get("VLM_API_URL", self.DEFAULT_API_URL)
        self.api_key = api_key or os.environ.get("VLM_API_KEY", "")
        self.model_name = model_name or os.environ.get("VLM_MODEL_NAME", self.DEFAULT_MODEL)
        self.request_timeout = request_timeout

        if self.model_name not in ALLOWED_MODELS:
            raise ValueError(
                f"Unsupported model: '{self.model_name}'. "
                f"Allowed: {sorted(ALLOWED_MODELS)}"
            )

        if not self.api_url:
            logger.warning("VLM API URL not configured. Set VLM_API_URL env var or pass api_url.")

        self._use_openai_format = "chat/completions" in self.api_url

    IMAGE_MAX_SIZE = 1024
    IMAGE_JPEG_QUALITY = 50

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64, resize to max 1024px."""
        w, h = image.size
        if max(w, h) > self.IMAGE_MAX_SIZE:
            scale = self.IMAGE_MAX_SIZE / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=self.IMAGE_JPEG_QUALITY)
        return base64.b64encode(buf.getvalue()).decode()

    # -- ARK Responses API format ------------------------------------------

    def _call_ark(self, prompt: str, max_retries: int, max_tokens: int,
                  system_prompt: str = None,
                  video_url: str = None, images: List[Image.Image] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": [{"type": "input_text", "text": system_prompt}]})
        content = []
        if video_url:
            content.append({"type": "input_video", "video_url": video_url})
        elif images:
            for img in images:
                b64 = self._encode_image(img)
                content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})
        content.append({"type": "input_text", "text": prompt})
        msgs.append({"role": "user", "content": content})

        url = self.api_url.rstrip("/") + "/responses"
        payload = {
            "model": self.model_name,
            "input": msgs,
            "temperature": 0.1,
            "max_output_tokens": max_tokens,
            "thinking": {"type": "disabled"},
        }

        attempt = 0
        while attempt < max_retries:
            try:
                resp = requests.post(url, headers=headers, json=payload,
                                     timeout=self.request_timeout)
                if resp.status_code == 429:
                    wait = self.RATE_LIMIT_COOLDOWN
                    logger.warning(f"Rate limited (429), cooling down {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                _increment_request_count()
                data = resp.json()
                for item in data.get("output", []):
                    if item.get("type") == "message":
                        for part in item.get("content", []):
                            if part.get("type") == "output_text":
                                text = part["text"].strip()
                                return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                return ""
            except Exception as e:
                attempt += 1
                if attempt < max_retries:
                    wait = self.RETRY_BACKOFF[min(attempt - 1, len(self.RETRY_BACKOFF) - 1)]
                    time.sleep(wait)
                else:
                    raise

    # -- OpenAI Chat Completions format ------------------------------------

    def _call_openai(self, prompt: str, max_retries: int, max_tokens: int,
                     system_prompt: str = None,
                     video_url: str = None, images: List[Image.Image] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        content = []
        if video_url:
            content.append({"type": "video_url", "video_url": {"url": video_url}})
        elif images:
            for img in images:
                b64 = self._encode_image(img)
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1,
            "max_completion_tokens": max_tokens,
            "thinking": {"type": "disabled"},
        }

        attempt = 0
        while attempt < max_retries:
            try:
                resp = requests.post(self.api_url, headers=headers, json=payload,
                                     timeout=self.request_timeout)
                if resp.status_code == 429:
                    wait = self.RATE_LIMIT_COOLDOWN
                    logger.warning(f"Rate limited (429), cooling down {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                _increment_request_count()
                text = resp.json()["choices"][0]["message"]["content"].strip()
                return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            except Exception as e:
                attempt += 1
                if attempt < max_retries:
                    wait = self.RETRY_BACKOFF[min(attempt - 1, len(self.RETRY_BACKOFF) - 1)]
                    time.sleep(wait)
                else:
                    raise

    # -- Public API --------------------------------------------------------

    MAX_RETRIES = 6
    RETRY_BACKOFF = [3, 8, 15, 25, 40, 60]
    RATE_LIMIT_COOLDOWN = 2

    def ask(self, prompt: str, images: List[Image.Image] = None, max_retries: int = 6,
            max_tokens: int = 300, system_prompt: str = None,
            video_url: str = None) -> str:
        """Ask a question with video or images. Returns raw text response.

        Args:
            prompt: Question text.
            images: Fallback image frames (used only if video_url is None).
            video_url: Base64-encoded video data URL (preferred).
            max_retries: Number of retries on failure.
            max_tokens: Max output tokens.
            system_prompt: Optional system prompt.
        """
        if not self.api_url:
            raise RuntimeError("VLM API not configured. Set VLM_API_URL env var or pass api_url.")

        if _request_semaphore:
            _request_semaphore.acquire()
        try:
            if self._use_openai_format:
                return self._call_openai(prompt, max_retries, max_tokens,
                                         system_prompt=system_prompt,
                                         video_url=video_url, images=images)
            else:
                return self._call_ark(prompt, max_retries, max_tokens,
                                      system_prompt=system_prompt,
                                      video_url=video_url, images=images)
        except Exception as e:
            logger.error(f"VLM API failed after {max_retries} retries: {e}")
            raise
        finally:
            if _request_semaphore:
                _request_semaphore.release()

    def ask_binary(self, question: str, images: List[Image.Image] = None,
                   max_retries: int = 3, video_url: str = None) -> bool:
        """Ask a binary (yes/no) question."""
        answer_text = self.ask(question + "\nAnswer with 'yes' or 'no' only.",
                              images, max_retries=max_retries, max_tokens=10,
                              video_url=video_url)
        return "yes" in answer_text.lower()

    def evaluate_tasks(self, tasks: List[VLMTask], nproc: int = 4) -> List[VLMTask]:
        """Evaluate multiple VLM tasks concurrently."""
        def _run_task(task):
            try:
                task.answer = self.ask_binary(task.question, task.images,
                                             video_url=task.video_url)
            except Exception as e:
                logger.warning(f"Task failed: {e}")
                task.answer = None
            return task

        with ThreadPoolExecutor(max_workers=nproc) as executor:
            futures = [executor.submit(_run_task, t) for t in tasks]
            results = [f.result() for f in as_completed(futures)]
        return results
