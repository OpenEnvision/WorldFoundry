# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch

from .model import RepEncoder
from .trajectory_fov import (
    DEFAULT_FAR_METERS,
    DEFAULT_HORIZONTAL_FOV_DEGREES,
    DEFAULT_NEAR_METERS,
    DEFAULT_VERTICAL_FOV_DEGREES,
    FRUSTUM_SAMPLES_PER_AXIS,
    select_trajectory_fov_history,
)


NUM_LATENT_FRAMES_PER_CHUNK = 9
VAE_SCALE_FACTOR_TEMPORAL = 4
WINDOW_NUM_FRAMES = 33
TARGET_SLOTS = (2, 4, 6, 8)
RECENT_LOCAL_SLOT = 8
EXPECTED_LATENT_CHW = (16, 48, 80)
HISTORY_SOURCE_BUDGET = 8


@dataclass(frozen=True)
class RepEncoderInferenceProviderConfig:
    seed: int = 0
    trajectory_fov_horizontal_fov_degrees: float = DEFAULT_HORIZONTAL_FOV_DEGREES
    trajectory_fov_vertical_fov_degrees: float = DEFAULT_VERTICAL_FOV_DEGREES
    trajectory_fov_near_m: float = DEFAULT_NEAR_METERS
    trajectory_fov_far_m: float = DEFAULT_FAR_METERS
    trajectory_fov_samples_per_axis: int = FRUSTUM_SAMPLES_PER_AXIS
    near_zero_baseline_m: float = 0.05

    def __post_init__(self) -> None:
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        for name in (
            "trajectory_fov_horizontal_fov_degrees",
            "trajectory_fov_vertical_fov_degrees",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 180.0:
                raise ValueError(f"{name} must be finite and in (0, 180)")
        if int(self.trajectory_fov_samples_per_axis) != FRUSTUM_SAMPLES_PER_AXIS:
            raise ValueError(
                "trajectory FOV selection requires "
                f"samples_per_axis={FRUSTUM_SAMPLES_PER_AXIS}"
            )
        near_m = float(self.trajectory_fov_near_m)
        far_m = float(self.trajectory_fov_far_m)
        if not math.isfinite(near_m) or near_m <= 0.0:
            raise ValueError("trajectory_fov_near_m must be finite and positive")
        if not math.isfinite(far_m) or far_m <= near_m:
            raise ValueError(
                "trajectory_fov_far_m must be finite and greater than near"
            )
        if (
            not math.isfinite(float(self.near_zero_baseline_m))
            or float(self.near_zero_baseline_m) < 0.0
        ):
            raise ValueError("near_zero_baseline_m must be finite and non-negative")


@dataclass(frozen=True)
class RepEncoderInferenceBatchSelection:
    batch_index: int
    mode: str
    retrieval_backend: str
    source_indices: tuple[tuple[int, int], ...]
    source_flat_indices: tuple[int, ...]
    source_raw_frames: tuple[int, ...]
    target_indices: tuple[tuple[int, int], ...]
    target_raw_frames: tuple[int, ...]
    selected_history_flat_indices: tuple[int, ...]
    source_unique_parent_count: int
    retrieval_diagnostics: Mapping[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_indices"] = [list(index) for index in self.source_indices]
        value["target_indices"] = [list(index) for index in self.target_indices]
        value["retrieval_diagnostics"] = dict(self.retrieval_diagnostics)
        return value


@dataclass(frozen=True)
class RepEncoderInferenceRenderRecord:
    chunk_index: int
    pose_key: str
    window_num_frames: int
    target_slots: tuple[int, ...]
    selections: tuple[RepEncoderInferenceBatchSelection, ...]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "pose_key": self.pose_key,
            "window_num_frames": self.window_num_frames,
            "target_slots": list(self.target_slots),
            "selections": [selection.to_jsonable() for selection in self.selections],
        }


def _as_global_metric_c2w(
    camera_trajectory: Mapping[str, Any], *, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, str]:
    if not isinstance(camera_trajectory, Mapping):
        raise TypeError("camera_trajectory must be a mapping")
    pose_key = "c2w"
    if camera_trajectory.get(pose_key) is None:
        raise KeyError("RepEncoder inference requires camera_trajectory['c2w']")
    pose = torch.as_tensor(camera_trajectory[pose_key])
    if pose.ndim == 3:
        pose = pose.unsqueeze(0)
    if pose.ndim != 4 or tuple(pose.shape[-2:]) not in {(3, 4), (4, 4)}:
        raise ValueError(
            f"{pose_key} must be [B,F,3,4] or [B,F,4,4], got {tuple(pose.shape)}"
        )
    if not torch.is_floating_point(pose):
        raise TypeError(f"{pose_key} must be floating point")
    pose = pose.to(device=device, dtype=torch.float32)
    if not torch.isfinite(pose).all():
        raise FloatingPointError(f"{pose_key} contains NaN or Inf")
    if pose.shape[-2:] == (3, 4):
        bottom = torch.zeros(
            *pose.shape[:-2], 1, 4, device=pose.device, dtype=pose.dtype
        )
        bottom[..., 0, 3] = 1.0
        pose = torch.cat((pose, bottom), dim=-2)
    else:
        expected_bottom = pose.new_tensor((0.0, 0.0, 0.0, 1.0)).expand(
            *pose.shape[:-2], 4
        )
        if not torch.allclose(pose[..., 3, :], expected_bottom, atol=1e-5, rtol=0):
            raise ValueError(f"{pose_key} has invalid homogeneous bottom rows")
    if pose.shape[0] != int(batch_size):
        raise ValueError(
            f"{pose_key} batch {pose.shape[0]} does not match WorldCrafter latent batch {batch_size}"
        )
    if int(batch_size) != 1:
        raise ValueError("strict online trajectory-FOV retrieval supports batch_size=1")
    if pose.shape[1] % WINDOW_NUM_FRAMES:
        raise ValueError(
            f"camera trajectory length must contain complete {WINDOW_NUM_FRAMES}-frame chunks, "
            f"got {pose.shape[1]}"
        )
    return pose, pose_key


def _anchor_raw_frames(num_chunks: int) -> torch.Tensor:
    return torch.as_tensor(
        [
            chunk * WINDOW_NUM_FRAMES + local * VAE_SCALE_FACTOR_TEMPORAL
            for chunk in range(int(num_chunks))
            for local in range(NUM_LATENT_FRAMES_PER_CHUNK)
        ],
        dtype=torch.int64,
    ).reshape(int(num_chunks), NUM_LATENT_FRAMES_PER_CHUNK)


class RepEncoderInferenceMemoryProvider:
    def __init__(
        self,
        runtime: RepEncoder,
        config: RepEncoderInferenceProviderConfig | Mapping[str, Any] | None = None,
        *,
        append_only: bool = False,
    ) -> None:
        if not callable(runtime):
            raise TypeError("runtime must be a callable RepEncoder memory runtime")
        if config is None:
            config = RepEncoderInferenceProviderConfig()
        elif isinstance(config, Mapping):
            config = RepEncoderInferenceProviderConfig(**dict(config))
        if not isinstance(config, RepEncoderInferenceProviderConfig):
            raise TypeError(
                "config must be RepEncoderInferenceProviderConfig or a mapping"
            )
        self.append_only = append_only
        self._committed_pose = None
        self.runtime = runtime
        self.config = config
        self.last_render_record: RepEncoderInferenceRenderRecord | None = None
        self.render_records: list[RepEncoderInferenceRenderRecord] = []
        self._global_pose_latents: torch.Tensor | None = None

    def reset_sequence(self) -> None:
        self._committed_pose = None
        self._global_pose_latents = None
        self.last_render_record = None
        self.render_records.clear()

    def append_trajectory(self, pose: torch.Tensor, chunk_index: int) -> None:
        """Commit one chunk without permitting changes to previously seen poses."""
        if not self.append_only:
            raise RuntimeError("Trajectory append requires append_only=True")
        pose, _ = _as_global_metric_c2w({"c2w": pose}, batch_size=1, device=pose.device)
        previous = 0 if self._committed_pose is None else self._committed_pose.shape[1]
        if (
            pose.shape[1] != (chunk_index + 1) * WINDOW_NUM_FRAMES
            or previous != chunk_index * WINDOW_NUM_FRAMES
        ):
            raise ValueError("Only the next trajectory chunk may be appended")
        if previous and not torch.equal(self._committed_pose, pose[:, :previous]):
            raise RuntimeError("Committed trajectory prefix changed")
        self._committed_pose = pose.detach().clone()

    def _validate_latents(
        self,
        generated_latents: torch.Tensor,
        recent_latents: torch.Tensor,
        *,
        chunk_index: int,
    ) -> None:
        if (
            not isinstance(generated_latents, torch.Tensor)
            or generated_latents.ndim != 5
        ):
            raise TypeError("generated_latents must be a [B,16,T,48,80] torch tensor")
        channels, latent_height, latent_width = EXPECTED_LATENT_CHW
        expected_tail = (
            channels,
            int(chunk_index) * NUM_LATENT_FRAMES_PER_CHUNK,
            latent_height,
            latent_width,
        )
        if tuple(generated_latents.shape[1:]) != expected_tail:
            raise ValueError(
                "generated_latents do not match complete causal WorldCrafter history: "
                f"expected [B,{expected_tail[0]},{expected_tail[1]},{expected_tail[2]},"
                f"{expected_tail[3]}], got {tuple(generated_latents.shape)}"
            )
        expected_recent = (
            generated_latents.shape[0],
            generated_latents.shape[1],
            1,
            generated_latents.shape[3],
            generated_latents.shape[4],
        )
        if (
            not isinstance(recent_latents, torch.Tensor)
            or tuple(recent_latents.shape) != expected_recent
        ):
            raise ValueError(
                f"recent_latents must be exact recent[-1] with shape {expected_recent}"
            )
        if (
            recent_latents.device != generated_latents.device
            or recent_latents.dtype != generated_latents.dtype
        ):
            raise ValueError(
                "recent_latents must share generated_latents device and dtype"
            )
        if not torch.is_floating_point(generated_latents):
            raise TypeError(
                "generated_latents must be floating point standardized Wan latents"
            )
        runtime_device = getattr(self.runtime, "device", generated_latents.device)
        if torch.device(runtime_device) != generated_latents.device:
            raise ValueError(
                "RepEncoder runtime and WorldCrafter latents must share one device; "
                f"runtime={runtime_device}, worldcrafter={generated_latents.device}"
            )

    def _initialize_or_validate_pose(
        self, *, pose: torch.Tensor, chunk_index: int
    ) -> torch.Tensor:
        total_chunks = int(pose.shape[1] // WINDOW_NUM_FRAMES)
        anchor_raw_frames = _anchor_raw_frames(total_chunks)
        anchor_indices = anchor_raw_frames.reshape(-1).to(device=pose.device)
        global_pose_latents = pose[0].index_select(0, anchor_indices).contiguous()
        if self.append_only:
            if self._committed_pose is None or not torch.equal(
                pose, self._committed_pose
            ):
                raise RuntimeError("Retrieval trajectory differs from committed poses")
            if total_chunks != chunk_index + 1:
                raise RuntimeError("Future trajectory chunks are not allowed")
            self._global_pose_latents = global_pose_latents.detach().clone()
        elif int(chunk_index) == 1:
            self.reset_sequence()
            self._global_pose_latents = global_pose_latents.detach().clone()
        elif self._global_pose_latents is None:
            self._global_pose_latents = global_pose_latents.detach().clone()
        elif not torch.equal(self._global_pose_latents, global_pose_latents):
            raise RuntimeError(
                "global camera trajectory changed within one autoregressive sequence"
            )
        return anchor_raw_frames

    def _select_history(
        self, *, history_length: int, target4_c2w: torch.Tensor
    ) -> tuple[tuple[int, ...], str, dict[str, Any]]:
        if self._global_pose_latents is None:
            raise RuntimeError("global retrieval poses were not initialized")
        recent_index = int(history_length) - 1
        candidates = tuple(range(recent_index))
        if len(candidates) < HISTORY_SOURCE_BUDGET:
            raise ValueError(
                "insufficient strict history for source8: "
                f"candidate_count={len(candidates)}, required={HISTORY_SOURCE_BUDGET}"
            )
        if len(candidates) == HISTORY_SOURCE_BUDGET:
            return (
                candidates,
                "trajectory_fov_exact_budget_bootstrap",
                {
                    "method": "trajectory_fov_a_to_b",
                    "selection_mode": "exact_budget_bootstrap_all_history",
                    "candidate_indices_global": list(candidates),
                    "fixed_context_indices_global": [recent_index],
                    "selected_indices_global": list(candidates),
                    "target_slots": list(TARGET_SLOTS),
                },
            )

        result = select_trajectory_fov_history(
            controller_c2w=self._global_pose_latents,
            candidate_latents=candidates,
            fixed_context_latents=(recent_index,),
            target_c2w=target4_c2w,
            budget=HISTORY_SOURCE_BUDGET,
            horizontal_fov_degrees=float(
                self.config.trajectory_fov_horizontal_fov_degrees
            ),
            vertical_fov_degrees=float(self.config.trajectory_fov_vertical_fov_degrees),
            near_m=float(self.config.trajectory_fov_near_m),
            far_m=float(self.config.trajectory_fov_far_m),
            frustum_samples_per_axis=int(self.config.trajectory_fov_samples_per_axis),
            latents_per_chunk=NUM_LATENT_FRAMES_PER_CHUNK,
        )
        diagnostics = result.to_json()
        diagnostics.update(
            {
                "method": "trajectory_fov_a_to_b",
                "selection_mode": "ordered_target_union_coverage",
                "candidate_indices_global": list(candidates),
                "fixed_context_indices_global": [recent_index],
                "selected_indices_global": list(result.selected_latents),
                "target_slots": list(TARGET_SLOTS),
                "frustum_samples_per_axis": int(
                    self.config.trajectory_fov_samples_per_axis
                ),
            }
        )
        return (
            tuple(int(value) for value in result.selected_latents),
            "trajectory_fov_a_to_b_coverage",
            diagnostics,
        )

    def render_memory(
        self,
        *,
        generated_latents: torch.Tensor,
        recent_latents: torch.Tensor,
        camera_trajectory: Mapping[str, Any],
        chunk_index: int,
        num_latent_frames_per_chunk: int,
        vae_scale_factor_temporal: int,
        generator: torch.Generator | Sequence[torch.Generator] | None = None,
    ) -> torch.Tensor:
        chunk_index = int(chunk_index)
        if chunk_index <= 0:
            raise ValueError("RepEncoder inference requires chunk_index >= 1")
        if int(num_latent_frames_per_chunk) != NUM_LATENT_FRAMES_PER_CHUNK:
            raise ValueError(
                "RepEncoder inference is fixed to nine latent frames per chunk"
            )
        if int(vae_scale_factor_temporal) != VAE_SCALE_FACTOR_TEMPORAL:
            raise ValueError(
                "RepEncoder inference is fixed to Wan temporal scale factor 4"
            )
        self._validate_latents(
            generated_latents, recent_latents, chunk_index=chunk_index
        )

        pose, pose_key = _as_global_metric_c2w(
            camera_trajectory,
            batch_size=int(generated_latents.shape[0]),
            device=generated_latents.device,
        )
        anchor_raw_frames = self._initialize_or_validate_pose(
            pose=pose, chunk_index=chunk_index
        )
        if chunk_index >= anchor_raw_frames.shape[0]:
            raise ValueError(
                f"camera trajectory has only {anchor_raw_frames.shape[0]} complete chunks, "
                f"cannot query chunk {chunk_index}"
            )
        if self._global_pose_latents is None:
            raise RuntimeError("global retrieval poses were not initialized")

        target_full_chunk_c2w = self._global_pose_latents[
            chunk_index
            * NUM_LATENT_FRAMES_PER_CHUNK : (chunk_index + 1)
            * NUM_LATENT_FRAMES_PER_CHUNK
        ]
        target4_c2w = target_full_chunk_c2w.index_select(
            0,
            torch.as_tensor(
                TARGET_SLOTS, device=target_full_chunk_c2w.device, dtype=torch.int64
            ),
        )
        selected_history, mode, retrieval_diagnostics = self._select_history(
            history_length=int(generated_latents.shape[2]), target4_c2w=target4_c2w
        )
        if (
            len(selected_history) != HISTORY_SOURCE_BUDGET
            or len(set(selected_history)) != HISTORY_SOURCE_BUDGET
        ):
            raise RuntimeError(
                "trajectory-FOV retrieval did not return eight unique history views: "
                f"{selected_history}"
            )

        del generator
        recent_index = int(generated_latents.shape[2]) - 1
        source_flat_indices = (recent_index, *selected_history)
        if len(set(source_flat_indices)) != NUM_LATENT_FRAMES_PER_CHUNK:
            raise RuntimeError(f"source9 is not unique: {source_flat_indices}")
        source_latents = (
            generated_latents.index_select(
                2,
                torch.as_tensor(
                    source_flat_indices,
                    device=generated_latents.device,
                    dtype=torch.int64,
                ),
            )
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

        source_c2w_metric = (
            self._global_pose_latents.index_select(
                0,
                torch.as_tensor(
                    source_flat_indices,
                    device=self._global_pose_latents.device,
                    dtype=torch.int64,
                ),
            )
            .unsqueeze(0)
            .to(device=generated_latents.device, dtype=torch.float32)
        )
        target_c2w_metric = target4_c2w.unsqueeze(0).to(
            device=generated_latents.device, dtype=torch.float32
        )
        memory4 = self.runtime(
            source_latents=source_latents,
            source_c2w_metric=source_c2w_metric,
            target_c2w_metric=target_c2w_metric,
            near_zero_baseline_m=float(self.config.near_zero_baseline_m),
        )
        expected_output = (int(generated_latents.shape[0]), 16, 4, 48, 80)
        if (
            not isinstance(memory4, torch.Tensor)
            or tuple(memory4.shape) != expected_output
        ):
            raise RuntimeError(
                f"RepEncoder runtime must return standardized memory4 {expected_output}, "
                f"got {None if not isinstance(memory4, torch.Tensor) else tuple(memory4.shape)}"
            )
        if memory4.device != generated_latents.device:
            raise ValueError(
                "RepEncoder memory4 must remain on the WorldCrafter latent device"
            )
        memory4 = memory4.to(dtype=generated_latents.dtype)

        source_indices = tuple(
            (index // NUM_LATENT_FRAMES_PER_CHUNK, index % NUM_LATENT_FRAMES_PER_CHUNK)
            for index in source_flat_indices
        )
        target_indices = tuple((chunk_index, slot) for slot in TARGET_SLOTS)
        selection = RepEncoderInferenceBatchSelection(
            batch_index=0,
            mode=mode,
            retrieval_backend="trajectory_fov_a_to_b",
            source_indices=source_indices,
            source_flat_indices=source_flat_indices,
            source_raw_frames=tuple(
                int(anchor_raw_frames[chunk, local]) for chunk, local in source_indices
            ),
            target_indices=target_indices,
            target_raw_frames=tuple(
                int(anchor_raw_frames[chunk, local]) for chunk, local in target_indices
            ),
            selected_history_flat_indices=selected_history,
            source_unique_parent_count=len({chunk for chunk, _ in source_indices}),
            retrieval_diagnostics=retrieval_diagnostics,
        )
        record = RepEncoderInferenceRenderRecord(
            chunk_index=chunk_index,
            pose_key=pose_key,
            window_num_frames=WINDOW_NUM_FRAMES,
            target_slots=TARGET_SLOTS,
            selections=(selection,),
        )
        self.last_render_record = record
        self.render_records.append(record)
        return memory4


__all__ = [
    "HISTORY_SOURCE_BUDGET",
    "RepEncoderInferenceBatchSelection",
    "RepEncoderInferenceMemoryProvider",
    "RepEncoderInferenceProviderConfig",
    "RepEncoderInferenceRenderRecord",
    "NUM_LATENT_FRAMES_PER_CHUNK",
    "RECENT_LOCAL_SLOT",
    "TARGET_SLOTS",
    "VAE_SCALE_FACTOR_TEMPORAL",
    "WINDOW_NUM_FRAMES",
]
