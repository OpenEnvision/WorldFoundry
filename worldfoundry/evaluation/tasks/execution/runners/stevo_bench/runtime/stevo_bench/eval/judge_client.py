# eval/judge_client.py
"""
VLM judge client supporting Gemini and OpenAI.

Env vars:
  GOOGLE_API_KEY  — required for Gemini models
  OPENAI_API_KEY  — required for OpenAI models

Deps:
  pip install google-genai                    # Gemini
  pip install openai opencv-python            # OpenAI (opencv used for video frame sampling)
  pip install anthropic opencv-python         # Anthropic/Claude
  pip install opencv-python                   # Gemini frame-sampling mode also uses opencv
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol


class ControlJudgeClient(Protocol):
    provider: str
    model: str

    def judge(self, prompt: str, video_path: Path) -> str: ...
    def judge_image(self, prompt: str, image_path: Path) -> str: ...


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

@dataclass
class GeminiControlJudgeClient:
    model: str = "gemini-3.1-pro-preview"
    provider: str = "gemini"
    n_frames: int = 32  # frames sampled from the video; 0 = upload full video

    def __post_init__(self) -> None:
        from google import genai  # type: ignore
        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")
        self._client = genai.Client(api_key=api_key)
        print(f"Using GOOGLE_API_KEY")

    def _sample_frames_bytes(self, video_path: Path) -> list[bytes]:
        """Sample self.n_frames uniformly from video; return raw JPEG bytes per frame."""
        try:
            import cv2  # type: ignore
        except ImportError:
            raise RuntimeError(
                "opencv-python is required for Gemini frame-sampling mode: pip install opencv-python"
            )

        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = self.n_frames
        n = min(self.n_frames, total)
        indices = [int(i * total / n) for i in range(n)]

        frames: list[bytes] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                frames.append(buf.tobytes())
        cap.release()
        return frames

    def judge(self, prompt: str, video_path: Path) -> str:
        from google.genai import types  # type: ignore

        video_path = Path(video_path).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        # Full-video upload mode (Gemini controls frame sampling internally).
        import time
        print(f"  [gemini] uploading {video_path.name} ...")
        uploaded = self._client.files.upload(file=str(video_path))
        while True:
            file_info = self._client.files.get(name=uploaded.name)
            if file_info.state == "ACTIVE":
                break
            elif file_info.state == "FAILED":
                raise RuntimeError("File processing failed.")
            time.sleep(2)
        contents = [
            {"text": prompt},
            types.Part.from_uri(file_uri=uploaded.uri, mime_type="video/mp4"),
        ]

        # Frame-sampling mode: extract N frames and send as images.
        # if self.n_frames > 0:
        #     print(f"  [gemini] sampling {self.n_frames} frames from {video_path.name} ...")
        #     frame_bytes = self._sample_frames_bytes(video_path)
        #     contents = [{"text": prompt}] + [
        #         types.Part.from_bytes(data=fb, mime_type="image/jpeg")
        #         for fb in frame_bytes
        #     ]

        resp = self._client.models.generate_content(
            model=self.model,
            contents=contents,
        )
        return getattr(resp, "text", "") or ""

    def judge_image(self, prompt: str, image_path: Path) -> str:
        from google.genai import types  # type: ignore

        image_path = Path(image_path).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        suffix = image_path.suffix.lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(suffix, "image/jpeg")
        img_bytes = image_path.read_bytes()

        resp = self._client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type=mime),
                {"text": prompt},
            ],
        )
        return getattr(resp, "text", "") or ""


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

@dataclass
class OpenAIControlJudgeClient:
    model: str = "gpt-4o"
    provider: str = "openai"
    n_frames: int = 32   # frames sampled from the video for judging
    max_side: int = 1024  # longest frame dimension before encoding; controls token cost

    def __post_init__(self) -> None:
        import openai  # type: ignore
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
        self._client = openai.OpenAI(api_key=api_key)
        print(f"Using OPENAI_API_KEY")

    def _encode_bytes(self, data: bytes) -> str:
        return base64.b64encode(data).decode("utf-8")

    def _sample_frames_b64(self, video_path: Path) -> list[str]:
        """Sample self.n_frames uniformly from video; return base64-encoded JPEG strings."""
        try:
            import cv2  # type: ignore
        except ImportError:
            raise RuntimeError(
                "opencv-python is required for OpenAI video judging: pip install opencv-python"
            )

        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = self.n_frames
        n = min(self.n_frames, total)
        indices = [int(i * total / n) for i in range(n)]

        frames: list[str] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            h, w = frame.shape[:2]
            if max(h, w) > self.max_side:
                scale = self.max_side / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                frames.append(self._encode_bytes(buf.tobytes()))
        cap.release()
        return frames

    def judge(self, prompt: str, video_path: Path) -> str:
        video_path = Path(video_path).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        print(f"  [openai] sampling {self.n_frames} frames from {video_path.name} ...")
        frames_b64 = self._sample_frames_b64(video_path)
        content: list = [{"type": "text", "text": prompt}]
        for f_b64 in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{f_b64}", "detail": "low"},
            })

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=2048,
        )
        return resp.choices[0].message.content or ""

    def judge_image(self, prompt: str, image_path: Path) -> str:
        image_path = Path(image_path).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        suffix = image_path.suffix.lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(suffix, "image/jpeg")
        img_b64 = self._encode_bytes(image_path.read_bytes())

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_completion_tokens=2048,
        )
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Anthropic / Claude
# ---------------------------------------------------------------------------

@dataclass
class AnthropicControlJudgeClient:
    model: str = "claude-opus-4-7"
    provider: str = "anthropic"
    n_frames: int = 32   # frames sampled from the video for judging
    max_side: int = 1024  # longest frame dimension before encoding; Anthropic hard-caps at 5 MB/image

    def __post_init__(self) -> None:
        import anthropic  # type: ignore
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY environment variable.")
        self._client = anthropic.Anthropic(api_key=api_key)
        print(f"Using ANTHROPIC_API_KEY")

    def _encode_bytes(self, data: bytes) -> str:
        return base64.b64encode(data).decode("utf-8")

    def _sample_frames_b64(self, video_path: Path) -> list[str]:
        """Sample self.n_frames uniformly from video; return base64-encoded JPEG strings."""
        try:
            import cv2  # type: ignore
        except ImportError:
            raise RuntimeError(
                "opencv-python is required for Claude video judging: pip install opencv-python"
            )

        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = self.n_frames
        n = min(self.n_frames, total)
        indices = [int(i * total / n) for i in range(n)]

        frames: list[str] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            h, w = frame.shape[:2]
            if max(h, w) > self.max_side:
                scale = self.max_side / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                frames.append(self._encode_bytes(buf.tobytes()))
        cap.release()
        return frames

    def judge(self, prompt: str, video_path: Path) -> str:
        video_path = Path(video_path).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        print(f"  [anthropic] sampling {self.n_frames} frames from {video_path.name} ...")
        frames_b64 = self._sample_frames_b64(video_path)
        content: list = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": f_b64},
            }
            for f_b64 in frames_b64
        ]
        content.append({"type": "text", "text": prompt})

        resp = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": content}],
        )
        return resp.content[0].text or ""

    def judge_image(self, prompt: str, image_path: Path) -> str:
        image_path = Path(image_path).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        try:
            import cv2  # type: ignore
            img = cv2.imread(str(image_path))
            if img is not None:
                h, w = img.shape[:2]
                if max(h, w) > self.max_side:
                    scale = self.max_side / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_bytes = buf.tobytes() if ok else image_path.read_bytes()
                mime = "image/jpeg"
            else:
                img_bytes = image_path.read_bytes()
                suffix = image_path.suffix.lower().lstrip(".")
                mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(suffix, "image/jpeg")
        except ImportError:
            img_bytes = image_path.read_bytes()
            suffix = image_path.suffix.lower().lstrip(".")
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(suffix, "image/jpeg")

        img_b64 = self._encode_bytes(img_bytes)

        resp = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime, "data": img_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return resp.content[0].text or ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt")
_ANTHROPIC_PREFIXES = ("claude",)
_GEMINI_PREFIXES = ("gemini",)


def make_judge_client(
    model: str,
    provider: Optional[str] = None,
    n_frames: Optional[int] = None,
) -> ControlJudgeClient:
    """
    Instantiate the right judge client for (provider, model).

    provider: "gemini" | "openai" | None (auto-detect from model name prefix)
    n_frames: override the default frame count for the chosen client.
              For Gemini: 0 means full-video upload (currently commented out).
    """
    if provider is None:
        m = model.lower()
        if any(m.startswith(p) for p in _OPENAI_PREFIXES):
            provider = "openai"
        elif any(m.startswith(p) for p in _ANTHROPIC_PREFIXES):
            provider = "anthropic"
        else:
            provider = "gemini"

    if provider == "openai":
        client = OpenAIControlJudgeClient(model=model)
        if n_frames is not None:
            client.n_frames = n_frames
        return client
    elif provider == "anthropic":
        client = AnthropicControlJudgeClient(model=model)
        if n_frames is not None:
            client.n_frames = n_frames
        return client
    elif provider == "gemini":
        client = GeminiControlJudgeClient(model=model)
        if n_frames is not None:
            client.n_frames = n_frames
        return client
    else:
        raise ValueError(f"Unknown provider {provider!r}. Supported: 'gemini', 'openai', 'anthropic'")
