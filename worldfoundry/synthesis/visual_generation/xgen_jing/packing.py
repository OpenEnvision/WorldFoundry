# Adapted from XGEN-JING; attribution and license: root THIRD-PARTY-NOTICES.
"""Single-request chunk packing; tensor geometry is delegated to Diffusers."""

from dataclasses import dataclass
from itertools import pairwise

import torch
from worldfoundry.base_models.diffusion_model.schedulers.minimax_h3 import (
    minimax_h3_align_frame_count as align_num_frames,
    minimax_h3_video_latent_t as video_latent_num_frames,
)
from worldfoundry.pipelines.minimax._minimax_h3.packed_tokens import (
    minimax_h3_patchify_video_latent,
)
from worldfoundry.pipelines.minimax._minimax_h3.packed_sequence import (
    _axis_from_sqrt_area as _spatial_position_grid,
)
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.jing.control import control_vector



def chunk_lengths(num_frames, first_chunk_size=2):
    frames = align_num_frames(num_frames)
    count, tail = divmod(video_latent_num_frames(frames), 5)
    if count < 1 or tail != 2 or first_chunk_size not in (0, 2):
        raise ValueError(
            "Expected at least 22 pixel frames and first_chunk_size 0 or 2"
        )
    return frames, (
        [7] + [5] * (count - 1) if first_chunk_size == 2 else [5] * count + [2]
    )


def expand_slices(case, count):
    if "trajectory" in case:
        raise ValueError("Use chunks or slices; trajectory is not supported")
    slices = case.get("slices")
    if "chunks" in case:
        if "slices" in case or "control" in case:
            raise ValueError("chunks cannot be combined with slices or top-level control")
        chunks = case["chunks"]
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("chunks must be a non-empty list")
        slices = []
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict) or set(chunk) - {"prompt", "repeat", "control"}:
                raise ValueError(f"Chunk {index}: only prompt, repeat and control are accepted")
            repeat = chunk.get("repeat", 1)
            if type(repeat) is not int or repeat <= 0:
                raise ValueError(f"Chunk {index}: repeat must be a positive integer")
            controls = chunk.get("control", [""] * repeat)
            if not isinstance(controls, list) or len(controls) != repeat:
                raise ValueError(f"Chunk {index}: control must be a list of length repeat ({repeat})")
            for control in controls:
                if not isinstance(control, str):
                    raise ValueError(f"Chunk {index}: each control must be a string, e.g. 'w,a' or ''")
                slices.append({"prompt": chunk.get("prompt", case.get("prompt")), "control": control})
    if slices is None:
        slices = [
            {"prompt": case.get("prompt"), "control": case.get("control", [])}
            for _ in range(count)
        ]
    if not isinstance(slices, list) or len(slices) != count:
        raise ValueError(f"Expected exactly {count} slices")
    result = []
    for item in slices:
        if not isinstance(item, dict) or set(item) - {"prompt", "control"}:
            raise ValueError("Each slice only accepts prompt and control")
        prompt = item.get("prompt", case.get("prompt"))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Every slice needs a non-empty prompt")
        keys = item.get("control", [])
        vector = control_vector(keys)
        if isinstance(keys, str):
            keys = ([key.strip().lower() for key in keys.split(",")]
                    if "," in keys else list(keys.strip().lower()))
        result.append(
            {
                "prompt": prompt,
                "control": sorted({key.lower() for key in keys}),
                "vector": vector,
            }
        )
    return result


def audio_endpoint(video_index):
    groups, tail = divmod(video_index, 5)
    if tail not in (0, 2):
        raise ValueError("Audio endpoints require video indices 5*n or 5*n+2")
    return groups * 28 + (8 if tail else 0)


def positions(frames, height, width, patch_size):
    """Global video positions and the piecewise video-aligned stereo clock."""
    _, ph, pw = patch_size
    h = _spatial_position_grid(height, ph, (height * width) ** 0.5)
    w = _spatial_position_grid(width, pw, (height * width) ** 0.5)
    # Integer pixel support starts avoid accumulating a rounded temporal scale.
    index = torch.arange(frames + 1)
    pixel_starts = (index // 5) * 17 + torch.tensor([0, 1, 5, 9, 13])[index % 5]
    clock = pixel_starts.to(torch.float64) * (5.0 / 3.0)
    video = torch.stack(
        torch.meshgrid(clock[:frames], h, w, indexing="ij"), dim=-1
    ).reshape(-1, 3)
    boundaries = list(range(0, frames - 1, 5)) + [frames]
    audio_time = []
    for start, end in pairwise(boundaries):
        length = audio_endpoint(end) - audio_endpoint(start)
        audio_time.append(
            clock[start]
            + torch.arange(length, dtype=torch.float64)
            * ((clock[end] - clock[start]) / length)
        )
    time = torch.cat(audio_time)
    audio = torch.zeros(2, len(time), 3, dtype=torch.float64)
    audio[:, :, 0] = time
    audio[0, :, 2], audio[1, :, 2] = w[0], w[-1]
    return video, audio


@dataclass
class Packed:
    video: torch.Tensor
    audio: torch.Tensor
    text: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    owners: torch.Tensor
    is_media: torch.Tensor
    reference_rows: torch.Tensor
    text_lengths: tuple
    controls: list
    output_video_offset: int
    audio_lengths: tuple

    def timesteps(self, video_t, audio_t, reference_t):
        rows = torch.full(
            (len(self.token_tags),),
            float(video_t),
            device=self.video.device,
            dtype=torch.float32,
        )
        rows[self.audio_indices] = float(audio_t)
        rows[self.reference_rows] = 1.0 - reference_t
        return torch.unique(rows, sorted=True, return_inverse=True)

    def audio_channel_major(self):
        pieces = self.audio[0].split([2 * length for length in self.audio_lengths])
        return torch.cat(
            [
                part.reshape(2, length, -1)
                for part, length in zip(pieces, self.audio_lengths)
            ],
            dim=1,
        ).flatten(0, 1)


def pack(payload, model_config, generation, device):
    patch = tuple(model_config["patch_size"])
    if patch[0] != 1:
        raise ValueError("This H3 demo requires temporal patch size 1")
    video, audio = payload["video"].to(device), payload["audio"].to(device)
    _, frames, height, width = video.shape
    if sum(payload["lengths"]) != frames or height % patch[1] or width % patch[2]:
        raise ValueError(
            "Slice lengths and spatial patches must cover the video exactly"
        )
    if tuple(audio.shape) != (
        2,
        audio_endpoint(frames),
        model_config["audio_in_channels"],
    ):
        raise ValueError(
            "Audio must contain both channels on the complete video timeline"
        )
    video_pos, audio_pos = positions(frames, height, width, patch)
    video_pos, audio_pos = video_pos.to(device), audio_pos.to(device)
    spatial_rows = height * width // (patch[1] * patch[2])
    parts = {
        key: []
        for key in (
            "video",
            "audio",
            "text",
            "vi",
            "ai",
            "ti",
            "pos",
            "tags",
            "owners",
            "media",
            "refs",
        )
    }
    controls, text_lengths, audio_lengths = [], [], []
    cursor = video_offset = start = 0

    def add_group(n, owner, media, tags, pos, index_key):
        nonlocal cursor
        indices = torch.arange(cursor, cursor + n, device=device)
        parts[index_key].append(indices)
        parts["pos"].append(pos)
        parts["tags"].append(tags)
        parts["owners"].append(torch.full((n,), owner, device=device, dtype=torch.long))
        parts["media"].append(torch.full((n,), media, device=device, dtype=torch.bool))
        cursor += n
        return indices

    for index, reference in enumerate(payload["references"]):
        reference = reference.to(device)
        ref_rows = minimax_h3_patchify_video_latent(reference[None], patch_size=patch)
        _, _, rh, rw = reference.shape
        ref_pos, _ = positions(2, rh, rw, patch)
        ref_pos = ref_pos[: len(ref_rows)].to(device)
        ref_pos[:, 0] = -generation["reserved_slots"] + index
        indices = add_group(
            len(ref_rows),
            -1,
            True,
            torch.zeros(len(ref_rows), device=device, dtype=torch.long),
            ref_pos,
            "vi",
        )
        parts["refs"].append(indices)
        parts["video"].append(ref_rows)
        video_offset += len(ref_rows)

    for index, (length, item) in enumerate(
        zip(payload["lengths"], payload["slices"], strict=True)
    ):
        end = start + length
        a0, a1 = audio_endpoint(start), audio_endpoint(end)
        text, tags = payload["text_bank"][item["prompt"]]
        text, tags = text.to(device), tags.to(device)
        text = text.reshape(-1, text.shape[-1])
        text_lengths.append(len(text))
        pos = torch.zeros(len(text), 3, device=device, dtype=torch.float64)
        pos[:, 0] = torch.arange(
            -generation["reserved_slots"] - len(text),
            -generation["reserved_slots"],
            device=device,
        )
        add_group(len(text), index, False, tags.flatten(), pos, "ti")
        parts["text"].append(text)
        a = audio[:, a0:a1].flatten(0, 1)
        add_group(
            len(a),
            index,
            True,
            torch.full((len(a),), 2, device=device, dtype=torch.long),
            audio_pos[:, a0:a1].flatten(0, 1),
            "ai",
        )
        parts["audio"].append(a)
        audio_lengths.append(a1 - a0)
        v = minimax_h3_patchify_video_latent(video[:, start:end][None], patch_size=patch)
        vi = add_group(
            len(v),
            index,
            True,
            torch.zeros(len(v), device=device, dtype=torch.long),
            video_pos[start * spatial_rows : end * spatial_rows],
            "vi",
        )
        parts["video"].append(v)
        # Empty set still encodes the model's zero camera command when control is enabled.
        raw = torch.tensor(item["vector"], device=device, dtype=video.dtype).expand(
            length, -1
        )
        controls.append((raw, height, width, vi))
        start = end

    def cat(key):
        return torch.cat(parts[key])

    return Packed(
        cat("video")[None],
        cat("audio")[None],
        cat("text")[None],
        cat("vi"),
        cat("ai"),
        cat("ti"),
        cat("pos"),
        cat("tags"),
        cat("owners"),
        cat("media"),
        cat("refs")
        if parts["refs"]
        else torch.empty(0, device=device, dtype=torch.long),
        tuple(text_lengths),
        controls,
        video_offset,
        tuple(audio_lengths),
    )
