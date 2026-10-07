# SPDX-License-Identifier: CC-BY-NC-4.0
"""WorldPlay2 controls and scheduled prompts over the shared Wan UMT5 encoder."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from worldfoundry.base_models.diffusion_model.contracts import Conditioning
from worldfoundry.base_models.diffusion_model.models.encoders.wan.component import build_wan_text_conditioner

_MOVEMENT = {"w": (1, 0), "s": (-1, 0), "a": (0, -1), "d": (0, 1),
             "wa": (1, -1), "wd": (1, 1), "sa": (-1, -1), "sd": (-1, 1)}
_ROTATION = {"up": (1, 0), "down": (-1, 0), "left": (0, -1), "right": (0, 1)}


def parse_worldplay2_actions(value, *, latent_frames, perspective="tps", yaw_speed=3.0, pitch_speed=1.0):
    """Parse official ``w+right-4,space-4`` commands in latent-frame units."""
    if perspective not in {"tps", "fps"}:
        raise ValueError("WorldPlay2 perspective must be 'tps' or 'fps'")
    if isinstance(value, torch.Tensor):
        result = value.detach().to(dtype=torch.float32).clone()
        if result.shape != (latent_frames, 6):
            raise ValueError(f"WorldPlay2 actions must have shape [{latent_frames}, 6]")
    elif isinstance(value, str):
        rows = []
        for command in value.split(","):
            tokens, separator, duration = command.strip().rpartition("-")
            if not separator or not duration.isdigit() or int(duration) <= 0:
                raise ValueError(f"WorldPlay2 actions require action-duration: {command!r}")
            row = [0.0, 0.0, 0.0, 0.0, float(perspective == "fps"), 0.0]
            seen = set()
            for token in tokens.lower().split("+"):
                token = token.strip()
                if token in _MOVEMENT:
                    category = "movement"
                    row[2:4] = _MOVEMENT[token]
                elif token in _ROTATION:
                    category = "pitch" if token in {"up", "down"} else "yaw"
                    pitch, yaw = _ROTATION[token]
                    row[0] += pitch * pitch_speed
                    row[1] += yaw * yaw_speed
                elif token == "space":
                    category = "space"
                    row[5] = 1.0
                elif token == "none" and tokens.strip().lower() == "none":
                    category = "none"
                else:
                    raise ValueError(f"unknown WorldPlay2 action: {token!r}")
                if category in seen:
                    raise ValueError(f"duplicate WorldPlay2 {category} action in {command!r}")
                seen.add(category)
            rows.extend([row] * int(duration))
        result = torch.tensor(rows, dtype=torch.float32)
        if result.shape != (latent_frames, 6):
            raise ValueError(f"WorldPlay2 actions cover {len(rows)} latent frames; expected {latent_frames}")
    else:
        raise TypeError("WorldPlay2 actions must be an official action string or [T,6] tensor")
    if not torch.isfinite(result).all():
        raise ValueError("WorldPlay2 actions must be finite")
    for index, allowed in ((2, (-1, 0, 1)), (3, (-1, 0, 1)), (4, (0, 1)), (5, (0, 1))):
        if not torch.isin(result[:, index], result.new_tensor(allowed)).all():
            raise ValueError(f"WorldPlay2 discrete action column {index} has an unsupported value")
    result[0, :4] = 0
    result[0, 5] = 0
    return result


def worldplay2_prompt_schedule(request, *, latent_frames, chunk_length):
    count = latent_frames // chunk_length
    prompts = request.inputs.get("prompts")
    if prompts is None:
        return {"prompt": request.prompts[0]}, ("prompt",) * count
    if not isinstance(prompts, Mapping) or not prompts or not all(isinstance(value, str) for value in prompts.values()):
        raise TypeError("WorldPlay2 prompts must map names to text")
    event = request.inputs.get("prompt_event")
    if not isinstance(event, str):
        raise ValueError("WorldPlay2 scheduled prompts require prompt_event")
    timeline = []
    for command in event.split(","):
        name, separator, duration = command.strip().rpartition("-")
        if not separator or name not in prompts or not duration.isdigit() or int(duration) <= 0:
            raise ValueError(f"invalid WorldPlay2 prompt event: {command!r}")
        timeline.extend([name] * int(duration))
        if len(timeline) % chunk_length:
            raise ValueError("WorldPlay2 prompt boundaries must align with chunk_length")
    if len(timeline) != latent_frames:
        raise ValueError("WorldPlay2 prompt_event and actions must cover the same latent frames")
    return dict(prompts), tuple(timeline[::chunk_length])


class WorldPlay2Conditioner:
    def __init__(self, text_conditioner, *, mode):
        self.text_conditioner = text_conditioner
        self.mode = mode

    def encode(self, request, *, device, dtype):
        if request.batch_size != 1:
            raise ValueError("WorldPlay2 generates one interactive trajectory per request")
        chunk_length = int(request.inputs.get("chunk_length", 32 if self.mode == "bi" else 4))
        frames = (request.num_frames - 1) // 4 + 1
        if chunk_length <= 0 or chunk_length % 2 or frames % chunk_length:
            raise ValueError("WorldPlay2 latent frames must be a multiple of an even chunk_length")
        prompts, schedule = worldplay2_prompt_schedule(request, latent_frames=frames, chunk_length=chunk_length)
        names = tuple(dict.fromkeys(schedule))
        encoded = self.text_conditioner._encode([prompts[name] for name in names], device=device, dtype=dtype)
        contexts = {name: encoded[index:index + 1] for index, name in enumerate(names)}
        negative = {}
        if request.sampling.guidance_scale != 1.0:
            negative["context"] = self.text_conditioner._encode(request.negative_prompts or ("",), device=device, dtype=dtype)
        raw_actions = request.inputs.get("actions")
        if raw_actions is None:
            raw_actions = request.inputs.get("interactions")
        if raw_actions is None:
            raw_actions = f"w-{frames}"
        actions = parse_worldplay2_actions(
            raw_actions,
            latent_frames=frames, perspective=str(request.inputs.get("perspective", "tps")),
            yaw_speed=float(request.inputs.get("yaw_rotation_speed_deg", 3.0)),
            pitch_speed=float(request.inputs.get("pitch_rotation_speed_deg", 1.0 if self.mode == "few_step" else 3.0)),
        ).to(device=device, dtype=dtype)
        return Conditioning(
            positive={"context": contexts[schedule[0]]}, negative=negative,
            shared={"chunk_contexts": contexts, "chunk_prompts": schedule,
                    "actions": actions, "chunk_length": chunk_length},
        )


def build_worldplay2_conditioner(context):
    return WorldPlay2Conditioner(build_wan_text_conditioner(context), mode=context.recipe_options["mode"])


__all__ = ["WorldPlay2Conditioner", "build_worldplay2_conditioner", "parse_worldplay2_actions"]
