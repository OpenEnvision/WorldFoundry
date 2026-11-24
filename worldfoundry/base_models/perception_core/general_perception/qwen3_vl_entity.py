"""Reusable Qwen3-VL entity extraction for inference pipelines.

This module is shared from ``base_models`` so model integrations do not vendor
their own Qwen inference wrapper.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass

import cv2
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# ---------------------------------------------------------------------------
# Entity parsing helpers
# ---------------------------------------------------------------------------

_NO_ENTITY_PHRASES = (
    "no moving",
    "no motion",
    "no movement",
    "no moving object",
    "no moving objects",
    "no dynamic",
    "no foreground",
    "nothing moving",
    "nothing moves",
    "nothing is moving",
    "nothing",
    "none",
    "only background",
    "background only",
    "static scene",
    "static background",
)

_BACKGROUND_BLACKLIST = {
    "road",
    "roads",
    "building",
    "buildings",
    "street",
    "streets",
    "sidewalk",
    "sidewalks",
    "wall",
    "walls",
    "floor",
    "ceiling",
    "tree",
    "trees",
    "grass",
    "furniture",
    "door",
    "doors",
    "window",
    "windows",
    "scenery",
    "structure",
    "structures",
    "bridge",
    "fence",
    "pole",
    "sign",
    "bench",
    "railing",
    "parking lot",
    "pathway",
    "pavement",
    "ground",
    "rug",
    "carpet",
    "mat",
    "curtain",
    "curtains",
    "table",
    "chair",
    "sofa",
    "couch",
    "desk",
    "shelf",
    "shelves",
    "lamp",
    "plant",
    "vase",
    "painting",
    "clock",
    "mirror",
    "pillow",
    "cup",
    "bowl",
    "plate",
    "bottle",
    "box",
    "sky",
    "cloud",
    "clouds",
    "water",
}

_PERSON_TERMS = (
    "person",
    "people",
    "human",
    "man",
    "woman",
    "child",
    "kid",
    "pedestrian",
    "hand",
    "hands",
    "arm",
    "arms",
    "leg",
    "legs",
    "head",
    "face",
    "body",
)

DEFAULT_ENTITY_PROMPT = (
    "List categories to remove while keeping only the static solid background. "
    "Remove people, vehicles, animals, handheld or movable objects, water, sky, "
    "and other dynamic elements. Return `Nothing` or a numbered list of at most "
    "four short category names; normalize every human category to `person`."
)


def _looks_like_no_entity(text: str) -> bool:
    if not text:
        return True
    lowered = text.strip().lower()
    return any(phrase in lowered for phrase in _NO_ENTITY_PHRASES)


def _is_null_entity(entity: str) -> bool:
    lowered = entity.strip().lower()
    if not lowered:
        return True
    if lowered in {"none", "nothing", "no", "n/a", "na"}:
        return True
    return any(phrase in lowered for phrase in _NO_ENTITY_PHRASES)


def _canonicalize_entity(entity: str) -> str:
    lowered = entity.lower()
    if any(term in lowered for term in _PERSON_TERMS):
        return "person"
    return entity


def parse_entities(text: str) -> list[str]:
    """Parse model output into a list of entity descriptions."""
    cleaned = text.strip().strip("[]")
    cleaned = cleaned.replace("\u3002", ".").replace("\n", ".")
    if _looks_like_no_entity(cleaned):
        return []
    parts = [p.strip() for p in cleaned.split(".") if p.strip()]
    entities = []
    for part in parts:
        part = re.sub(r"^[\d\s\-\)\(\.]+", "", part).strip()
        if not part:
            continue
        if _is_null_entity(part):
            continue
        entities.append(part)
    entities = [_canonicalize_entity(ent) for ent in entities]
    entities = [ent for ent in entities if ent.lower() not in _BACKGROUND_BLACKLIST]
    seen: set[str] = set()
    unique: list[str] = []
    for ent in entities:
        key = ent.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(ent)
    return unique[:4]


# ---------------------------------------------------------------------------
# Qwen3-VL entity extractor
# ---------------------------------------------------------------------------


@dataclass
class Qwen3VLEntityExtractor:
    """Qwen3-VL foreground entity extractor."""

    model_path: str
    device: str
    max_new_tokens: int = 128
    attn_implementation: str = "flash_attention_2"

    def __post_init__(self):
        device = str(self.device)
        is_cpu = device.startswith("cpu")
        model_kwargs = {
            "device_map": {"": device},
        }
        if is_cpu:
            model_kwargs["dtype"] = torch.float32
            model_kwargs["attn_implementation"] = "eager"
        else:
            model_kwargs["dtype"] = torch.bfloat16
            model_kwargs["attn_implementation"] = self.attn_implementation

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            **model_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def extract(
        self,
        input_path: str,
        prompt: str = DEFAULT_ENTITY_PROMPT,
    ) -> tuple[list[str], str]:
        """Extract foreground entity descriptions from an image or video.

        Args:
            input_path: Path to image or video file.
            prompt: Detection prompt text. Callers may override the reusable
                inference default with a model-specific prompt.

        Returns:
            Tuple of (entity_list, raw_output_text).
        """
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

        ext = os.path.splitext(input_path)[1].lower()

        if ext in image_extensions:
            content_type = "image"
            content_path = input_path
            temp_image_path = None
        elif ext in video_extensions:
            cap = cv2.VideoCapture(input_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count >= 2:
                cap.release()
                content_type = "video"
                content_path = input_path
                temp_image_path = None
            else:
                ret, frame = cap.read()
                cap.release()
                if not ret or frame is None:
                    return [], "Failed to read video frame"
                temp_image_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name  # noqa: SIM115 - Keeps the upstream file-handle lifetime.
                cv2.imwrite(temp_image_path, frame)
                content_type = "image"
                content_path = temp_image_path
        else:
            content_type = "image"
            content_path = input_path
            temp_image_path = None

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": content_type, content_type: content_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            entities = parse_entities(output_text)
            if not entities:
                return [], "Nothing"
            return entities, output_text
        finally:
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)

    def generate_text(self, image_path: str, prompt: str) -> str:
        """General-purpose text generation from image + prompt.

        Unlike extract(), returns the raw model output without entity parsing.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text.strip()
