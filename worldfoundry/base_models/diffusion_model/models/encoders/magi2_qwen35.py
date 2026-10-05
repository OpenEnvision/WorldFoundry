# SPDX-License-Identifier: Apache-2.0
"""MAGI-2 Qwen3.5 text encoder.

Ported from SandAI MAGI-2-preview (Apache-2.0): ``inference/model/qwen35.py``.
This single-GPU PyTorch port wraps ``transformers.Qwen3_5TextModel`` (available in
transformers 5.5+, e.g. HF class id ``Qwen3_5TextModel``). It preserves the custom
CJK-splitting pre-tokenizer, the ``max_length=7000`` truncation, the JSON ->
compact-markdown prompt normalization, and the ``skip_layer`` hidden-state
selection contract. The heavy backbone is constructed lazily so that importing
this module requires neither model weights nor transformers 5.5.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

# Output hidden dim of the selected layer; matches the DiT ``text_in_channels``.
MAGI2_QWEN35_HIDDEN_DIM = 5120
_DEFAULT_TRANSFORMERS_OVERLAY = "magi2-transformers-5.5.0"


def resolve_magi2_transformers_overlay(
    configured_path: str | Path | None = None,
) -> Path:
    """Resolve the process-local Transformers 5.5 overlay used by MAGI-2.

    MAGI-2 needs ``Qwen3_5TextModel`` from Transformers 5.5, while the shared
    WorldFoundry process intentionally stays on Transformers 4.x for other
    video runtimes.  The newer package is therefore visible only to a short
    text-encoding subprocess.
    """

    source_root = Path(__file__).resolve().parents[5]
    candidates = [
        configured_path,
        os.environ.get("WORLDFOUNDRY_MAGI2_TRANSFORMERS_OVERLAY"),
        source_root / "artifacts" / "runtime_overlays" / _DEFAULT_TRANSFORMERS_OVERLAY,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if (path / "transformers").is_dir() and (path / "huggingface_hub").is_dir():
            return path
    raise RuntimeError(
        "MAGI-2 requires a process-local Transformers 5.5 overlay containing "
        "transformers/ and huggingface_hub/. Set "
        "WORLDFOUNDRY_MAGI2_TRANSFORMERS_OVERLAY or materialize "
        f"artifacts/runtime_overlays/{_DEFAULT_TRANSFORMERS_OVERLAY}."
    )


def strip_empty(obj):
    if isinstance(obj, dict):
        return {
            key: value
            for key, value in ((key, strip_empty(value)) for key, value in obj.items())
            if value is not None and value != [] and value != {}
        }
    if isinstance(obj, list):
        return [strip_empty(value) for value in obj if value is not None]
    return obj


def _one_line(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return " ".join(value.split())


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _pick(values: dict, keys: Iterable[str]) -> list[tuple[str, Any]]:
    return [
        (key, values[key])
        for key in keys
        if key in values and values[key] not in (None, [], {})
    ]


def json_to_compact_markdown(raw_json: str | dict[str, Any]) -> str:
    if isinstance(raw_json, str):
        obj = json.loads(raw_json.strip())
    else:
        obj = raw_json

    obj = strip_empty(obj)
    if not isinstance(obj, dict) or "global_layer" not in obj:
        return (
            raw_json
            if isinstance(raw_json, str)
            else json.dumps(raw_json, ensure_ascii=False)
        )

    global_layer = _as_dict(obj.get("global_layer"))
    dynamic_layer = _as_dict(obj.get("dynamic_layer"))
    reference_layer = obj.get("reference_layer", [])

    lines: list[str] = []
    context = _one_line(global_layer.get("context"))
    description = _one_line(global_layer.get("description"))
    if context or description:
        lines.append(f"context: {context}" if context else "context")
        if description:
            lines.append(description)

    aesthetics = _as_dict(global_layer.get("aesthetics"))
    aesthetic_parts = []
    for label, key in (
        ("style", "style"),
        ("mood", "mood_atmosphere"),
        ("color", "color_scheme"),
    ):
        value = _one_line(aesthetics.get(key))
        if value:
            aesthetic_parts.append(f"{label}={value}")
    if aesthetic_parts:
        lines.append("aesthetics: " + "; ".join(aesthetic_parts))

    audio = _as_dict(global_layer.get("audio_baseline"))
    dialogue = _as_dict(audio.get("dialogue"))
    audio_parts = []
    language = _one_line(dialogue.get("language"))
    speakers = _as_list(dialogue.get("speaker_tags"))
    ambience = _one_line(audio.get("ambience"))
    if language:
        audio_parts.append(f"language={language}")
    if speakers:
        audio_parts.append("speakers=" + ",".join(map(_one_line, speakers)))
    if ambience:
        audio_parts.append(f"ambience={ambience}")
    if audio_parts:
        lines.append("audio: " + "; ".join(audio_parts))

    subjects = global_layer.get("alive_subjects_static")
    if isinstance(subjects, list) and subjects:
        lines.append("subjects:")
        for subject in subjects:
            if not isinstance(subject, dict):
                continue
            sid = _one_line(subject.get("subject_id"))
            desc = _one_line(subject.get("description"))
            position = _one_line(subject.get("position"))
            orientation = _one_line(subject.get("orientation"))
            attrs = []
            for key, value in _pick(
                _as_dict(subject.get("visual_attributes")),
                ["gender", "age_group", "ethnicity", "clothing", "appearance_details"],
            ):
                value = _one_line(value)
                if value:
                    attrs.append(f"{key}={value}")
            row = " - " + " | ".join([part for part in [sid, desc] if part])
            extra = "; ".join([part for part in [position, orientation] if part])
            if extra:
                row += f" ({extra})"
            if attrs:
                row += " :: " + "; ".join(attrs)
            lines.append(row)

    objects = global_layer.get("objects_static")
    if isinstance(objects, list) and objects:
        lines.append("objects:")
        for obj_item in objects:
            if not isinstance(obj_item, dict):
                continue
            oid = _one_line(obj_item.get("object_id"))
            desc = _one_line(obj_item.get("description"))
            shape = _one_line(obj_item.get("shape_and_color"))
            position = _one_line(obj_item.get("position"))
            row = " - " + " | ".join([part for part in [oid, desc] if part])
            details = "; ".join([part for part in [shape, position] if part])
            if details:
                row += " :: " + details
            lines.append(row)

    segments = dynamic_layer.get("timeline_segments")
    if isinstance(segments, list) and segments:
        lines.append("timeline:")
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            basic = _as_dict(segment.get("segment_basic_info"))
            timestamp = _one_line(basic.get("timestamp_range"))
            desc = _one_line(basic.get("segment_description"))
            head = f" - {timestamp}" if timestamp else " -"
            if desc:
                head += f" {desc}"
            lines.append(head.rstrip())

            segment_audio = _as_dict(segment.get("audio"))
            for dialogue_line in segment_audio.get("dialogue_lines", []) or []:
                if not isinstance(dialogue_line, dict):
                    continue
                speaker = _one_line(dialogue_line.get("speaker"))
                text = _one_line(dialogue_line.get("text"))
                timestamp = _one_line(dialogue_line.get("timestamp"))
                if text:
                    prefix = (
                        f"   - dialogue {speaker}: " if speaker else "   - dialogue: "
                    )
                    line = prefix + text
                    if timestamp:
                        line += f" ({timestamp})"
                    lines.append(line)

            for alive in segment.get("alive_subjects", []) or []:
                if not isinstance(alive, dict):
                    continue
                sid = _one_line(alive.get("subject_id"))
                action = _as_dict(alive.get("action"))
                parts = [
                    _one_line(action.get("primary_action")),
                    _one_line(action.get("interaction")),
                    _one_line(action.get("facial_expression")),
                ]
                parts = [part for part in parts if part]
                if parts:
                    lines.append(f"   - action {sid}: " + "; ".join(parts))

            for obj_item in segment.get("objects", []) or []:
                if not isinstance(obj_item, dict):
                    continue
                oid = _one_line(obj_item.get("object_id"))
                state = _as_dict(obj_item.get("dynamic_state"))
                parts = [
                    _one_line(state.get("state_change")),
                    _one_line(state.get("motion_detail")),
                ]
                parts = [part for part in parts if part]
                if parts:
                    lines.append(f"   - objects {oid}: " + "; ".join(parts))

    if isinstance(reference_layer, list):
        lines.extend(str(line) for line in reference_layer if str(line).strip())

    return "\n".join(line for line in lines if line.strip())


@dataclass
class Magi2Qwen35Config:
    """Configuration for :class:`Magi2Qwen35TextEncoder`.

    Defaults reflect the SandAI MAGI-2-preview class defaults. Note that while
    ``skip_layer`` defaults to ``0`` (matching the upstream class default), the
    MAGI pipeline instantiates the encoder with ``skip_layer=2`` so that
    ``encode`` returns the 3rd-from-last hidden state (``hidden_states[-3]``).
    """

    model_path: str
    device: str = "cuda"
    precision: torch.dtype = torch.bfloat16
    max_length: int = 7000
    skip_layer: int = 0
    hidden_dim: int = MAGI2_QWEN35_HIDDEN_DIM


class Magi2Qwen35TextEncoder(nn.Module):
    """Single-GPU wrapper around ``transformers.Qwen3_5TextModel``.

    Faithful port of SandAI MAGI-2-preview's ``Qwen35TextEncoder``. The heavy
    ``Qwen3_5TextModel`` backbone is imported and constructed lazily inside
    ``__init__`` so that importing this module has no transformers-5.5 dependency.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        precision: torch.dtype = torch.bfloat16,
        max_length: int = 7000,
        skip_layer: int = 0,
    ):
        super().__init__()
        self.device_str = device
        self.max_length = max_length
        self.skip_layer = skip_layer

        # Lazy imports: keep module import free of tokenizers/transformers-5.5.
        from tokenizers import Regex, pre_tokenizers
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")

        cjk_split = pre_tokenizers.Split(
            pattern=Regex(
                r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
            ),
            behavior="isolated",
        )
        original_pre_tokenizer = self.tokenizer.backend_tokenizer.pre_tokenizer
        self.tokenizer.backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [cjk_split, original_pre_tokenizer]
        )

        from transformers import Qwen3_5TextModel

        self.text_model = Qwen3_5TextModel.from_pretrained(
            model_path, torch_dtype=precision
        )
        self.to(device)
        self.text_model.eval()
        self.text_model.requires_grad_(False)

    @classmethod
    def from_config(cls, config: Magi2Qwen35Config) -> "Magi2Qwen35TextEncoder":
        """Build the encoder from a :class:`Magi2Qwen35Config`."""
        return cls(
            model_path=config.model_path,
            device=config.device,
            precision=config.precision,
            max_length=config.max_length,
            skip_layer=config.skip_layer,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda",
        precision: torch.dtype = torch.bfloat16,
        max_length: int = 7000,
        skip_layer: int = 0,
    ) -> "Magi2Qwen35TextEncoder":
        """Thin alias for the constructor, mirroring repo encoder conventions."""
        return cls(
            model_path,
            device=device,
            precision=precision,
            max_length=max_length,
            skip_layer=skip_layer,
        )

    def to(self, device: str | torch.device):  # type: ignore[override]
        self.text_model.to(device)
        self.device_str = str(device)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.text_model.parameters()).device

    def _normalize_prompt(self, prompt: str) -> str:
        try:
            parsed_json = json.loads(prompt)
            if isinstance(parsed_json, (dict, list)):
                return json_to_compact_markdown(prompt)
        except (json.JSONDecodeError, TypeError):
            pass
        return prompt

    def get_target_token_indices(
        self, prompt: str, target_str: Optional[str]
    ) -> Optional[list[int]]:
        if not target_str:
            return None
        prompt = self._normalize_prompt(prompt)
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding="longest",
            return_offsets_mapping=True,
            max_length=self.max_length,
            truncation=True,
        )
        offsets = inputs["offset_mapping"][0]
        start_char_idx = prompt.find(target_str)
        if start_char_idx == -1:
            return None
        end_char_idx = start_char_idx + len(target_str)

        indices: list[int] = []
        for index, (start, end) in enumerate(offsets):
            if start == 0 and end == 0 and index != 0:
                continue
            if max(start, start_char_idx) < min(end, end_char_idx):
                indices.append(index)
        return indices

    def get_special_token(
        self, prompt: str, target_strs: list[str], text_feature: torch.Tensor
    ) -> torch.Tensor:
        embeddings = []
        for target_str in target_strs:
            indices = self.get_target_token_indices(prompt, target_str)
            if indices:
                embeddings.append(text_feature[0, indices, :].mean(dim=0).clone())
            else:
                embeddings.append(
                    torch.zeros(
                        text_feature.shape[-1],
                        device=text_feature.device,
                        dtype=text_feature.dtype,
                    )
                )
        return torch.stack(embeddings, dim=0)

    @torch.inference_mode()
    def encode(self, prompt: str) -> torch.Tensor:
        prompt = self._normalize_prompt(prompt)
        device = next(self.text_model.parameters()).device
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding="longest",
            max_length=self.max_length,
            truncation=True,
        ).to(device)
        outputs = self.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
        if self.skip_layer == 0:
            return outputs.last_hidden_state
        return outputs.hidden_states[-(self.skip_layer + 1)]


class Magi2Qwen35SubprocessTextEncoder:
    """Encode prompts in an isolated Transformers-5.5 subprocess.

    The subprocess deliberately uses the same Python executable as the parent
    WorldFoundry job.  Only ``PYTHONPATH`` is prepended with the two-package
    overlay, so the shared environment and its CUDA/PyTorch stack remain
    unchanged and other runtimes continue to import Transformers 4.x.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        precision: torch.dtype = torch.bfloat16,
        max_length: int = 7000,
        skip_layer: int = 0,
        python_executable: str | Path | None = None,
        transformers_overlay: str | Path | None = None,
    ) -> None:
        self.model_path = str(Path(model_path).expanduser())
        self.device = str(device)
        self.precision = precision
        self.max_length = int(max_length)
        self.skip_layer = int(skip_layer)
        self.python_executable = str(python_executable or sys.executable)
        self.transformers_overlay = resolve_magi2_transformers_overlay(
            transformers_overlay
        )

    @torch.inference_mode()
    def encode(self, prompt: str) -> torch.Tensor:
        with tempfile.TemporaryDirectory(prefix="worldfoundry-magi2-text-") as temp_dir:
            temp = Path(temp_dir)
            request_path = temp / "request.json"
            output_path = temp / "prompt_context.pt"
            request_path.write_text(json.dumps({"prompt": prompt}, ensure_ascii=False))

            env = os.environ.copy()
            pythonpath = [str(self.transformers_overlay)]
            if env.get("PYTHONPATH"):
                pythonpath.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = os.pathsep.join(pythonpath)
            env["PYTHONNOUSERSITE"] = "1"
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"

            command = [
                self.python_executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--model-path",
                self.model_path,
                "--request",
                str(request_path),
                "--output",
                str(output_path),
                "--device",
                self.device,
                "--precision",
                str(self.precision).removeprefix("torch."),
                "--max-length",
                str(self.max_length),
                "--skip-layer",
                str(self.skip_layer),
            ]
            completed = subprocess.run(
                command,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                details = (completed.stderr or completed.stdout).strip()
                raise RuntimeError(
                    "MAGI-2 Qwen3.5 subprocess failed with the shared Python "
                    f"{self.python_executable}: {details[-4000:]}"
                )
            if not output_path.is_file():
                raise RuntimeError("MAGI-2 Qwen3.5 subprocess produced no prompt context")
            context = torch.load(output_path, map_location="cpu", weights_only=True)
        if not isinstance(context, torch.Tensor):
            raise TypeError("MAGI-2 Qwen3.5 subprocess output is not a tensor")
        return context.to(self.device)


def _patch_transformers_optional_distribution_mapping() -> None:
    """Tolerate source-only flash-attn installs without package metadata."""

    import transformers.utils.import_utils as import_utils

    import_utils.PACKAGE_DISTRIBUTION_MAPPING.setdefault("flash_attn_interface", [])
    import_utils.PACKAGE_DISTRIBUTION_MAPPING.setdefault("flash_attn", [])


def _run_subprocess_worker(args: argparse.Namespace) -> None:
    _patch_transformers_optional_distribution_mapping()
    request = json.loads(Path(args.request).read_text())
    precision = getattr(torch, args.precision)
    encoder = Magi2Qwen35TextEncoder(
        args.model_path,
        device=args.device,
        precision=precision,
        max_length=args.max_length,
        skip_layer=args.skip_layer,
    )
    context = encoder.encode(str(request["prompt"])).detach().to("cpu")
    torch.save(context, args.output)
    print(
        json.dumps(
            {
                "status": "succeeded",
                "shape": list(context.shape),
                "dtype": str(context.dtype),
                "finite": bool(torch.isfinite(context.float()).all()),
            }
        )
    )


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--request")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--max-length", type=int, default=7000)
    parser.add_argument("--skip-layer", type=int, default=0)
    args = parser.parse_args()
    if not args.worker:
        parser.error("this module is an internal MAGI-2 text-encoding worker")
    for name in ("model_path", "request", "output"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required with --worker")
    _run_subprocess_worker(args)


if __name__ == "__main__":
    _main()
