from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import requests
except Exception:
    requests = None

try:
    from google import genai as google_genai
    from google.genai import types as google_types
except Exception:
    google_genai = None
    google_types = None

try:
    import anthropic
except Exception:
    anthropic = None


# -----------------------------
# Frame extraction
# -----------------------------
def _require_pillow():
    if Image is None:
        raise RuntimeError("Pillow is required. Install via `pip install Pillow`.")


def _image_to_b64(img: Image.Image, jpeg_quality: int = 80) -> str:
    _require_pillow()
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_frames_cv2(path: str, max_frames: int = 24, fps: float | None = None) -> list[tuple[float, Image.Image]]:
    if not path:
        return []
    if cv2 is None:
        return []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / max(native_fps, 1e-6)

    if fps and fps > 0:
        step = 1.0 / fps
        times = [i * step for i in range(int(duration // step) + 1)]
    else:
        max_frames = max(1, int(max_frames))
        times = [i * duration / max_frames for i in range(max_frames)]

    frames: list[tuple[float, Image.Image]] = []
    for t_s in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000.0)
        ok, fr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        from PIL import Image as _PIL

        frames.append((t_s, _PIL.fromarray(rgb)))
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames


def _extract_frames_ffmpeg(path: str, max_frames: int = 24) -> list[tuple[float, Image.Image]]:
    if not path:
        return []
    if Image is None:
        return []
    # probe duration
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    duration = float((probe.stdout or "0").strip() or 0.0)
    if not duration or duration <= 0.0:
        duration = 5.0

    max_frames = max(1, int(max_frames))
    times = [i * duration / max_frames for i in range(max_frames)]

    out_frames: list[tuple[float, Image.Image]] = []
    for t_s in times:
        out = subprocess.run(
            ["ffmpeg", "-ss", str(t_s), "-i", path, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
            capture_output=True,
            check=True,
        )
        img = Image.open(io.BytesIO(out.stdout)).convert("RGB")
        out_frames.append((t_s, img))

    return out_frames


def extract_frames(path: str, max_frames: int = 24, fps: float | None = None) -> list[tuple[float, Image.Image]]:
    if not path:
        return []

    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        _require_pillow()
        return [(0.0, Image.open(path).convert("RGB"))]
    frames = _extract_frames_cv2(path, max_frames=max_frames, fps=fps)
    if frames:
        return frames
    return _extract_frames_ffmpeg(path, max_frames=max_frames)


# -----------------------------
# Provider base
# -----------------------------
class _BaseVLM:
    def _prompt(self, extra: dict) -> str:
        """
        Accept either extra['judge_prompt'] (preferred) or extra['rubric_prompt'] (legacy).
        """
        if not isinstance(extra, dict):
            raise RuntimeError("Missing prompting extras")
        p = extra.get("judge_prompt") or extra.get("rubric_prompt")
        if not isinstance(p, str) or not p.strip():
            raise RuntimeError("Missing judge_prompt/rubric_prompt in extras")
        return p


# -----------------------------
# OpenAI
# -----------------------------
@dataclass
class OpenAIVLMAPI(_BaseVLM):
    base_url: str = os.environ.get("OPENAI_API_BASE", "https://api.openai.com").rstrip("/")

    def _headers(self) -> dict:
        if requests is None:
            raise RuntimeError("requests is required for the OpenAI VideoScience judge provider")
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def _extract_output_text(self, resp: dict) -> str:
        # Prefer aggregated "output_text"
        txt = resp.get("output_text")
        if isinstance(txt, str) and txt.strip():
            return txt

        # Fallback: responses[].content[].text
        chunks = []
        for item in resp.get("output") or []:
            for c in item.get("content") or []:
                if isinstance(c, dict):
                    t = c.get("text")
                    if isinstance(t, str):
                        chunks.append(t)
        if chunks:
            return "\n".join(chunks)

        return json.dumps(resp)

    def analyze(
        self,
        model: str,
        video_path: str,
        phenomenon: str,
        ref_video_path: str | None,
        max_frames: int,
        fps: float | None,
        timeout_s: int,
        extra: dict,
    ) -> dict:
        cand = extract_frames(video_path, max_frames=max_frames, fps=fps)
        ref = extract_frames(ref_video_path, max_frames=max_frames // 2, fps=fps) if ref_video_path else []

        # Keep content leaner to avoid 400s from oversize payloads.
        max_imgs = int(extra.get("max_images", 20))
        content = [{"type": "input_text", "text": self._prompt(extra)}]

        for ts, im in cand[:max_imgs]:
            b64 = _image_to_b64(im)
            content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})
            content.append({"type": "input_text", "text": f"Candidate frame t={ts:.1f}s"})

        if ref:
            content.append({"type": "input_text", "text": "Reference video frames:"})
            for ts, im in ref[: max(1, max_imgs // 2)]:
                b64 = _image_to_b64(im)
                content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})
                content.append({"type": "input_text", "text": f"Reference frame t={ts:.1f}s"})

        payload = {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": int(extra.get("max_output_tokens", 20480)),
        }

        url = f"{self.base_url}/v1/responses"
        r = requests.post(url, headers=self._headers(), json=payload, timeout=timeout_s)

        if not r.ok:
            print(f"[VLM HTTP ERROR] {r.status_code} {r.reason} for {url}")

        resp = r.json()
        output_text = self._extract_output_text(resp)
        return {
            "provider": "openai",
            "model": model,
            "output_text": output_text,
            "output_mime": "application/json",
            "evidence": {
                "candidate_frames": [f"t={ts:.1f}s" for ts, _ in cand[:max_imgs]],
                "reference_frames": [f"t={ts:.1f}s" for ts, _ in (ref[: max(1, max_imgs // 2)] if ref else [])],
            },
            "raw": resp,
        }


# -----------------------------
# Google Gemini (native upload + JSON response)
# -----------------------------
@dataclass
class GeminiVLMAPI(_BaseVLM):
    def _ensure_client(self):
        if google_genai is None:
            raise RuntimeError("google-genai not installed. Run: pip install google-genai")
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY not set in environment")
        return google_genai.Client()

    def _wait_until_active(self, client, file_obj, timeout_s: int = 600, poll_s: float = 1.0):
        if not file_obj:
            return None
        name = getattr(file_obj, "name", None) or (file_obj.get("name") if isinstance(file_obj, dict) else None)
        deadline = time.time() + max(5, timeout_s)
        while time.time() < deadline:
            f = client.files.get(name=name)
            state = getattr(f, "state", None) or (f.get("state") if isinstance(f, dict) else None)
            if state == "ACTIVE":
                return f
            if state in ("FAILED", "DELETED"):
                raise RuntimeError(f"Gemini file {name} state={state}; cannot use.")
            time.sleep(poll_s)
        raise TimeoutError(f"Timed out waiting for Gemini file {name} to become ACTIVE.")

    def analyze(
        self,
        model: str,
        video_path: str,
        phenomenon: str,
        ref_video_path: str | None,
        max_frames: int,
        fps: float | None,
        timeout_s: int,
        extra: dict,
    ) -> dict:
        client = self._ensure_client()
        if google_types is None:
            raise RuntimeError("google-genai not installed. Run: pip install google-genai")

        cand_file = client.files.upload(file=video_path)
        cand_file = self._wait_until_active(client, cand_file, timeout_s=min(timeout_s, 600))

        ref_file = None
        if ref_video_path:
            ref_file = client.files.upload(file=ref_video_path)
            ref_file = self._wait_until_active(client, ref_file, timeout_s=min(timeout_s, 600))

        config = google_types.GenerateContentConfig(response_mime_type="application/json")
        parts = [self._prompt(extra), cand_file]
        if ref_file:
            parts += ["Reference video follows.", ref_file]

        resp = client.models.generate_content(model=model, contents=parts, config=config)

        text = getattr(resp, "text", None)
        if not text and getattr(resp, "candidates", None):
            text = resp.candidates[0].content.parts[0].text

        return {
            "provider": "gemini",
            "model": model,
            "output_text": text or "",
            "output_mime": "application/json",
            "evidence": {
                "candidate_frames": ["uploaded:file"],
                "reference_frames": ["uploaded:file"] if ref_file else [],
            },
            "raw": resp.to_dict() if hasattr(resp, "to_dict") else str(resp),
        }


# -----------------------------
# Anthropic Claude (base64 images)
# -----------------------------
@dataclass
class AnthropicVLMAPI(_BaseVLM):
    def _client(self):
        if anthropic is None:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return anthropic.Anthropic(api_key=key)

    def analyze(
        self,
        model: str,
        video_path: str,
        phenomenon: str,
        ref_video_path: str | None,
        max_frames: int,
        fps: float | None,
        timeout_s: int,
        extra: dict,
    ) -> dict:
        client = self._client()
        cand = extract_frames(video_path, max_frames=max_frames, fps=fps)
        ref = extract_frames(ref_video_path, max_frames=max_frames // 2, fps=fps) if ref_video_path else []

        def _image_part(img: Image.Image) -> dict:
            b64 = _image_to_b64(img)
            return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}

        content = [{"type": "text", "text": self._prompt(extra)}]
        max_imgs = int(extra.get("max_images", 20))

        for ts, im in cand[:max_imgs]:
            content.append(_image_part(im))
            content.append({"type": "text", "text": f"Candidate frame t={ts:.1f}s"})
        if ref:
            content.append({"type": "text", "text": "Reference video frames:"})
            for ts, im in ref[: max(1, max_imgs // 2)]:
                content.append(_image_part(im))
                content.append({"type": "text", "text": f"Reference frame t={ts:.1f}s"})

        msg = client.messages.create(
            model=model,
            max_tokens=int(extra.get("max_output_tokens", 20480)),
            messages=[{"role": "user", "content": content}],
        )

        text = msg.content[0].text if getattr(msg, "content", None) else str(msg)

        return {
            "provider": "anthropic",
            "model": model,
            "output_text": text,
            "output_mime": "application/json",
            "evidence": {
                "candidate_frames": [f"t={ts:.1f}s" for ts, _ in cand[:max_imgs]],
                "reference_frames": [f"t={ts:.1f}s" for ts, _ in (ref[: max(1, max_imgs // 2)] if ref else [])],
            },
            "raw": msg.model_dump() if hasattr(msg, "model_dump") else str(msg),
        }
