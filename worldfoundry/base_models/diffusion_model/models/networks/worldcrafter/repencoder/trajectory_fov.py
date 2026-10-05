# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import torch


DEFAULT_HORIZONTAL_FOV_DEGREES = 100.0
DEFAULT_VERTICAL_FOV_DEGREES = 71.13349068444832
DEFAULT_NEAR_METERS = 0.1
DEFAULT_FAR_METERS = 30.0
FRUSTUM_SAMPLES_PER_AXIS = 10


@dataclass(frozen=True)
class TrajectoryFovSelection:
    selected_latents: tuple[int, ...]
    selected_parent_chunks: tuple[int, ...]
    selection_kinds: tuple[str, ...]
    initial_fov_coverage_per_target: tuple[float, ...]
    final_fov_coverage_per_target: tuple[float, ...]
    initial_frontier_coverage_per_target: tuple[float, ...]
    final_frontier_coverage_per_target: tuple[float, ...]
    minimum_bridge_fraction: float
    mean_bridge_fraction: float
    disconnected_fill_count: int
    step_diagnostics: tuple[dict[str, object], ...]
    candidate_count: int
    target_count: int

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def _validate_pose_batch(value: torch.Tensor, *, name: str) -> torch.Tensor:
    pose = torch.as_tensor(value)
    if pose.ndim != 3 or pose.shape[-2:] != (4, 4):
        raise ValueError(f"{name} must be Nx4x4, got {tuple(pose.shape)}")
    if not torch.is_floating_point(pose) or not torch.isfinite(pose).all():
        raise ValueError(f"{name} must contain finite floating-point poses")
    return pose


def target_frustum_visibility_masks(
    *,
    target_c2w: torch.Tensor,
    source_c2w: torch.Tensor,
    horizontal_fov_degrees: float,
    vertical_fov_degrees: float,
    near_m: float,
    far_m: float,
    samples_per_axis: int = FRUSTUM_SAMPLES_PER_AXIS,
) -> torch.Tensor:

    targets = _validate_pose_batch(target_c2w, name="target_c2w")
    sources = _validate_pose_batch(source_c2w, name="source_c2w").to(
        device=targets.device, dtype=targets.dtype
    )
    if int(samples_per_axis) != FRUSTUM_SAMPLES_PER_AXIS:
        raise ValueError(
            "trajectory FOV selection requires "
            f"samples_per_axis={FRUSTUM_SAMPLES_PER_AXIS}"
        )
    if not (math.isfinite(near_m) and math.isfinite(far_m) and 0.0 < near_m < far_m):
        raise ValueError("near_m and far_m must be finite with 0 < near_m < far_m")
    if not 0.0 < float(horizontal_fov_degrees) < 180.0:
        raise ValueError("horizontal_fov_degrees must be in (0, 180)")
    if not 0.0 < float(vertical_fov_degrees) < 180.0:
        raise ValueError("vertical_fov_degrees must be in (0, 180)")

    device, dtype = targets.device, targets.dtype
    z = torch.linspace(float(near_m), float(far_m), FRUSTUM_SAMPLES_PER_AXIS, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, FRUSTUM_SAMPLES_PER_AXIS, device=device, dtype=dtype)
    y = torch.linspace(-1.0, 1.0, FRUSTUM_SAMPLES_PER_AXIS, device=device, dtype=dtype)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
    tan_h = math.tan(math.radians(float(horizontal_fov_degrees)) * 0.5)
    tan_v = math.tan(math.radians(float(vertical_fov_degrees)) * 0.5)
    target_points = torch.stack(
        (
            grid_x.reshape(-1) * grid_z.reshape(-1) * tan_h,
            grid_y.reshape(-1) * grid_z.reshape(-1) * tan_v,
            grid_z.reshape(-1),
        ),
        dim=0,
    )

    points_world = (
        torch.bmm(
            targets[:, :3, :3],
            target_points.unsqueeze(0).expand(targets.shape[0], -1, -1),
        )
        + targets[:, :3, 3:4]
    )
    source_r_inv = sources[:, :3, :3].transpose(1, 2)
    source_t_inv = -torch.bmm(source_r_inv, sources[:, :3, 3:4])
    points_in_source = torch.einsum("sij,tjp->stip", source_r_inv, points_world)
    points_in_source = points_in_source + source_t_inv[:, None]
    px = points_in_source[:, :, 0]
    py = points_in_source[:, :, 1]
    pz = points_in_source[:, :, 2]
    yaw = torch.atan2(px, pz.clamp_min(1e-6)).abs()
    pitch = torch.atan2(
        py, torch.sqrt(px.square() + pz.square()).clamp_min(1e-6)
    ).abs()
    return (
        (pz >= float(near_m))
        & (pz <= float(far_m))
        & (yaw <= math.radians(float(horizontal_fov_degrees)) * 0.5)
        & (pitch <= math.radians(float(vertical_fov_degrees)) * 0.5)
    )


def ordered_target_frontier_masks(
    *,
    anchor_c2w: torch.Tensor,
    target_c2w: torch.Tensor,
    horizontal_fov_degrees: float,
    vertical_fov_degrees: float,
    near_m: float,
    far_m: float,
    samples_per_axis: int = FRUSTUM_SAMPLES_PER_AXIS,
) -> tuple[torch.Tensor, torch.Tensor]:

    targets = _validate_pose_batch(target_c2w, name="target_c2w")
    anchor = torch.as_tensor(anchor_c2w, device=targets.device, dtype=targets.dtype)
    if anchor.shape != (4, 4) or not bool(torch.isfinite(anchor).all()):
        raise ValueError("anchor_c2w must be one finite 4x4 pose")
    prior_poses = torch.cat((anchor.unsqueeze(0), targets[:-1]), dim=0)
    visibility = target_frustum_visibility_masks(
        target_c2w=targets,
        source_c2w=prior_poses,
        horizontal_fov_degrees=horizontal_fov_degrees,
        vertical_fov_degrees=vertical_fov_degrees,
        near_m=near_m,
        far_m=far_m,
        samples_per_axis=samples_per_axis,
    )
    prior_index = torch.arange(prior_poses.shape[0], device=targets.device)[:, None]
    target_index = torch.arange(targets.shape[0], device=targets.device)[None, :]
    chronological = prior_index <= target_index
    overlap = (visibility & chronological.unsqueeze(-1)).any(dim=0)
    return ~overlap, overlap


def _lexicographic_best(
    rows: torch.Tensor,
    *,
    available: torch.Tensor,
    stable_values: Sequence[int],
) -> int:
    matrix = torch.as_tensor(rows).detach().cpu().to(torch.float64)
    flags = torch.as_tensor(available).detach().cpu().tolist()
    eligible = [index for index, flag in enumerate(flags) if bool(flag)]
    if matrix.ndim != 2 or matrix.shape[0] != len(flags):
        raise ValueError("rows and available have incompatible shapes")
    if not eligible:
        raise ValueError("no candidate is available")
    return max(
        eligible,
        key=lambda index: (
            *(float(value) for value in matrix[index].tolist()),
            -int(stable_values[index]),
        ),
    )


def _coverage_ratio(mask: torch.Tensor) -> torch.Tensor:
    return mask.float().mean(dim=-1)


def _frontier_ratio(mask: torch.Tensor, frontier: torch.Tensor) -> torch.Tensor:
    cells = frontier.sum(dim=1).float()
    covered = (mask & frontier).sum(dim=-1).float()
    return torch.where(
        cells > 0.0,
        covered / cells.clamp_min(1.0),
        torch.ones_like(cells),
    )


def _pick_best(
    rows: torch.Tensor,
    *,
    available: torch.Tensor,
    stable_values: Sequence[int],
) -> int:
    return _lexicographic_best(
        rows,
        available=available,
        stable_values=tuple(-int(value) for value in stable_values),
    )


@torch.inference_mode()
def select_trajectory_fov_history(
    *,
    controller_c2w: torch.Tensor,
    candidate_latents: Sequence[int],
    fixed_context_latents: Sequence[int],
    target_c2w: torch.Tensor,
    budget: int = 8,
    horizontal_fov_degrees: float = DEFAULT_HORIZONTAL_FOV_DEGREES,
    vertical_fov_degrees: float = DEFAULT_VERTICAL_FOV_DEGREES,
    near_m: float = DEFAULT_NEAR_METERS,
    far_m: float = DEFAULT_FAR_METERS,
    frustum_samples_per_axis: int = FRUSTUM_SAMPLES_PER_AXIS,
    latents_per_chunk: int = 9,
) -> TrajectoryFovSelection:

    poses = _validate_pose_batch(controller_c2w, name="controller_c2w")
    targets = _validate_pose_batch(target_c2w, name="target_c2w").to(
        device=poses.device, dtype=poses.dtype
    )
    candidates = tuple(sorted({int(value) for value in candidate_latents}))
    fixed = tuple(int(value) for value in fixed_context_latents)
    if int(budget) <= 0:
        raise ValueError("budget must be positive")
    if len(fixed) != 1:
        raise ValueError("trajectory FOV retrieval requires exactly one recent[-1] anchor")
    if len(candidates) < int(budget):
        raise ValueError(f"need at least {budget} candidates, got {len(candidates)}")
    if set(candidates).intersection(fixed):
        raise ValueError("candidate and fixed context latents must be disjoint")
    if targets.shape[0] != 4:
        raise ValueError(f"expected target slots 2/4/6/8, got {targets.shape[0]} poses")
    if int(latents_per_chunk) <= 0:
        raise ValueError("latents_per_chunk must be positive")
    maximum_index = max((*candidates, *fixed))
    minimum_index = min((*candidates, *fixed))
    if minimum_index < 0:
        raise ValueError("latent indices must be non-negative")
    if maximum_index >= poses.shape[0]:
        raise ValueError(
            f"latent index {maximum_index} exceeds controller pose count {poses.shape[0]}"
        )

    combined = (*fixed, *candidates)
    pose_index = torch.tensor(combined, device=poses.device, dtype=torch.long)
    combined_c2w = poses.index_select(0, pose_index)
    anchor_c2w = combined_c2w[0]
    visibility = target_frustum_visibility_masks(
        target_c2w=targets,
        source_c2w=combined_c2w,
        horizontal_fov_degrees=horizontal_fov_degrees,
        vertical_fov_degrees=vertical_fov_degrees,
        near_m=near_m,
        far_m=far_m,
        samples_per_axis=frustum_samples_per_axis,
    )
    anchor_fov = visibility[0]
    candidate_fov = visibility[1:]
    frontier_fov, _ = ordered_target_frontier_masks(
        anchor_c2w=anchor_c2w,
        target_c2w=targets,
        horizontal_fov_degrees=horizontal_fov_degrees,
        vertical_fov_degrees=vertical_fov_degrees,
        near_m=near_m,
        far_m=far_m,
        samples_per_axis=frustum_samples_per_axis,
    )

    available = torch.ones(len(candidates), device=poses.device, dtype=torch.bool)
    covered = anchor_fov.clone()
    selected_positions: list[int] = []
    selection_kinds: list[str] = []
    steps: list[dict[str, object]] = []
    bridge_values: list[float] = []
    disconnected_fill_count = 0

    for slot in range(int(budget)):
        resulting = covered.unsqueeze(0) | candidate_fov
        full_ratio = resulting.float().mean(dim=2)
        full_fairness = torch.sort(full_ratio, dim=1).values
        frontier_ratio = _frontier_ratio(resulting, frontier_fov)
        frontier_fairness = torch.sort(frontier_ratio, dim=1).values

        bridge_cells = (candidate_fov & covered.unsqueeze(0)).sum(dim=(1, 2)).float()
        candidate_cells = candidate_fov.sum(dim=(1, 2)).float()
        bridge_fraction = bridge_cells / candidate_cells.clamp_min(1.0)
        connected = bridge_cells > 0.0

        new_fov = candidate_fov & (~covered).unsqueeze(0)
        new_fov_per_target = new_fov.sum(dim=2).float()
        new_fov_total = new_fov_per_target.sum(dim=1)
        new_frontier = new_fov & frontier_fov.unsqueeze(0)
        new_frontier_per_target = new_frontier.sum(dim=2).float()
        new_frontier_total = new_frontier_per_target.sum(dim=1)

        rows = torch.cat(
            (
                full_fairness,
                frontier_ratio[:, -1:],
                frontier_fairness,
                new_fov_total[:, None],
                new_frontier_total[:, None],
            ),
            dim=1,
        )
        eligible = available & connected
        used_disconnected_fill = not bool(eligible.any())
        if used_disconnected_fill:
            eligible = available
        chosen = _pick_best(rows, available=eligible, stable_values=candidates)

        if used_disconnected_fill:
            kind = "disconnected_fill"
            disconnected_fill_count += 1
        elif int(new_frontier_total[chosen].item()) > 0:
            kind = "connected_frontier_advance"
        elif int(new_fov_total[chosen].item()) > 0:
            kind = "connected_coverage_refinement"
        else:
            kind = "connected_saturated_fill"

        covered = resulting[chosen]
        available[chosen] = False
        selected_positions.append(chosen)
        selection_kinds.append(kind)
        bridge = float(bridge_fraction[chosen].item())
        bridge_values.append(bridge)
        steps.append(
            {
                "slot": slot,
                "latent": candidates[chosen],
                "parent_chunk": candidates[chosen] // int(latents_per_chunk),
                "kind": kind,
                "bridge_fraction": bridge,
                "raw_new_fov_cells_per_target": [
                    int(value) for value in new_fov_per_target[chosen].cpu().tolist()
                ],
                "raw_new_frontier_cells_per_target": [
                    int(value) for value in new_frontier_per_target[chosen].cpu().tolist()
                ],
                "absolute_fov_coverage_per_target": [
                    float(value) for value in _coverage_ratio(covered).cpu().tolist()
                ],
                "frontier_coverage_per_target": [
                    float(value)
                    for value in _frontier_ratio(covered, frontier_fov).cpu().tolist()
                ],
            }
        )

    selected = tuple(candidates[position] for position in selected_positions)
    if len(selected) != int(budget) or len(set(selected)) != int(budget):
        raise RuntimeError(
            f"trajectory FOV retrieval did not produce {budget} unique sources: {selected}"
        )
    return TrajectoryFovSelection(
        selected_latents=selected,
        selected_parent_chunks=tuple(
            sorted({value // int(latents_per_chunk) for value in selected})
        ),
        selection_kinds=tuple(selection_kinds),
        initial_fov_coverage_per_target=tuple(
            float(value) for value in _coverage_ratio(anchor_fov).cpu().tolist()
        ),
        final_fov_coverage_per_target=tuple(
            float(value) for value in _coverage_ratio(covered).cpu().tolist()
        ),
        initial_frontier_coverage_per_target=tuple(
            float(value)
            for value in _frontier_ratio(anchor_fov, frontier_fov).cpu().tolist()
        ),
        final_frontier_coverage_per_target=tuple(
            float(value)
            for value in _frontier_ratio(covered, frontier_fov).cpu().tolist()
        ),
        minimum_bridge_fraction=min(bridge_values),
        mean_bridge_fraction=sum(bridge_values) / len(bridge_values),
        disconnected_fill_count=disconnected_fill_count,
        step_diagnostics=tuple(steps),
        candidate_count=len(candidates),
        target_count=int(targets.shape[0]),
    )


__all__ = [
    "DEFAULT_FAR_METERS",
    "DEFAULT_HORIZONTAL_FOV_DEGREES",
    "DEFAULT_NEAR_METERS",
    "DEFAULT_VERTICAL_FOV_DEGREES",
    "FRUSTUM_SAMPLES_PER_AXIS",
    "TrajectoryFovSelection",
    "ordered_target_frontier_masks",
    "select_trajectory_fov_history",
    "target_frustum_visibility_masks",
]
