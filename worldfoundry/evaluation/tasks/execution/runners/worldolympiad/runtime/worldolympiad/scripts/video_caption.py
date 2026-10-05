"""
Generate structured, shot-level captions for embodied/robotic videos using the unified VLM interface.

Usage:
python scripts/video_caption.py --video path/to/video.mp4 \
    --segments 6 --backend api --model google/gemini-2.5-flash

The script prints a single JSON object to stdout that mirrors
scripts/example.jsonl, e.g.:
{"prompts": ["Phase I: 00:00-00:05\\nCamera: ...\\n\\nRobotic Arm Appearance: ...\\n\\nRobotic Arm Actions: ..."]}


python scripts/video_caption.py --video data/robotics/Split_aloha_zip_up_the_document_bag/videos/chunk-000/observation.images.cam_high_rgb/episode_000000.mp4
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.vlm import resolve_model_name, resolve_vlm_backend, video_vlm_call
from judge_pipeline import parse_llm_response

load_dotenv()


SYSTEM_PROMPT = """You are a precise video captioning assistant for embodied robotic scenes.

Create vivid, chronological captions that describe the camera shot, the robotic arm's appearance, and the robotic arm's actions.
Keep the writing observational, specific, and technical enough to match what is visible.
Always respond with strict JSON only."""


def encode_video_to_data_url(video_path: str) -> Dict:
    """Encode a video file as a data URL payload for the VLM interface."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "video_url",
        "video_url": {
            "url": f"data:video/mp4;base64,{encoded}"
        }
    }


def build_user_prompt(num_segments: int) -> str:
    """Construct the user prompt that matches scripts/example.jsonl format."""
    return f"""You will watch one video and write exactly {num_segments} structured shot captions for an embodied/robotic scene.

Format:
{{
  "prompts": [
    "Phase I: 00:00-00:05\\nCamera: Shot description and movement.\\n\\nRobotic Arm Appearance: Shape, end-effector, materials, lights, mounted tools.\\n\\nRobotic Arm Actions: Motions, grasps, interactions, trajectory, speed, force cues.",
    "Phase II: 00:05-00:12\\nCamera: ...\\n\\nRobotic Arm Appearance: ...\\n\\nRobotic Arm Actions: ..."
  ]
}}

Rules:
- Use fluent English with concrete visual details; keep each section 1–3 sentences.
- Include a clear phase label and time range (MM:SS-MM:SS) at the start of each caption.
- Keep phases chronological, non-overlapping, and split by meaningful scene or action changes.
- Camera line must describe framing, movement, and perspective.
- Robotic Arm Appearance must cover geometry, joints, tools/end-effector, materials, lights/indicators.
- Robotic Arm Actions must cover motion pattern, interactions with objects, notable dynamics (speed, force, precision).
- No bullet markers, numbering, markdown fences, or extra keys—only the JSON object above."""


def parse_model_content(content: str) -> Dict[str, List[str]]:
    """
    Clean and parse the model output into a prompts list.

    Uses the shared parse_llm_response helper for robustness and surfaces
    clear errors when the model does not return the expected JSON structure.
    """
    if isinstance(content, list):
        # Both remote and local backends may return structured content parts; join text portions.
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            else:
                parts.append(str(item))
        content = "\n".join(parts)
    elif content is None:
        content = ""
    elif not isinstance(content, str):
        content = str(content)

    parsed = parse_llm_response(content)
    if "error" in parsed:
        raise ValueError(
            f"Failed to parse model JSON: {parsed.get('exception', 'unknown error')}. "
            f"Raw text: {parsed.get('raw_text', '')[:500]}"
        )

    if not isinstance(parsed, dict) or "prompts" not in parsed:
        raise ValueError(f"Model output missing 'prompts' field. Raw: {parsed}")
    if not isinstance(parsed["prompts"], list):
        raise ValueError("Model output 'prompts' field must be a list.")
    return parsed


def generate_captions(
    video_path: str,
    model_name: str | None = None,
    backend: str | None = None,
    num_segments: int = 6,
) -> Dict[str, List[str]]:
    """Call the VLM and return the parsed caption JSON."""
    data_url = encode_video_to_data_url(video_path)
    user_prompt = build_user_prompt(num_segments)

    response = video_vlm_call(
        data_url=data_url,
        system_prompt=SYSTEM_PROMPT,
        user_content=user_prompt,
        model_name=model_name,
        backend=backend,
    )

    if "choices" not in response or not response["choices"]:
        raise RuntimeError(f"Unexpected API response: {response}")

    content = response["choices"][0]["message"]["content"]
    return parse_model_content(content)


def main():
    parser = argparse.ArgumentParser(
        description="Generate video captions in the scripts/example.jsonl format"
    )
    parser.add_argument("--video", required=True, help="Path to the input video file")
    parser.add_argument(
        "--segments",
        type=int,
        default=6,
        help="Number of caption segments to generate (default: 6)",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="VLM backend: api/openrouter or local/qwenvl. Defaults to VLM_BACKEND/openrouter.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="VLM model name or local model path. Defaults to the selected backend's default model.",
    )
    args = parser.parse_args()
    vlm_backend = resolve_vlm_backend(args.backend)
    model_name = resolve_model_name(args.model, vlm_backend)

    result = generate_captions(
        video_path=args.video,
        model_name=model_name,
        backend=vlm_backend,
        num_segments=args.segments,
    )
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
