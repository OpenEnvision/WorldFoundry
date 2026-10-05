# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat


from . import camera as ucpe_cc
from . import prope as prope_torch

if TYPE_CHECKING:
    from ..transformer import WorldCrafterTransformer3DModel


UcpeSelfAttention = ucpe_cc.UcpeSelfAttention


def _flash_attention_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    compatibility_mode: bool = False,
) -> torch.Tensor:
    del compatibility_mode
    batch, tokens, channels = q.shape
    head_dim = channels // num_heads
    q = q.view(batch, tokens, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch, tokens, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch, tokens, num_heads, head_dim).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    return out.transpose(1, 2).reshape(batch, tokens, channels)


def enable_ucpe_inference_sdpa_attention() -> None:
    ucpe_cc.flash_attention = _flash_attention_sdpa


def _resolve_checkpoint_payload_path(checkpoint_path: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "camera_adapter.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Camera adapter not found: {checkpoint_path}")
    return checkpoint_path


def load_ucpe_checkpoint_state_dict(checkpoint_path: str | Path):
    resolved_path = _resolve_checkpoint_payload_path(checkpoint_path)
    state_obj = torch.load(resolved_path, map_location="cpu", weights_only=True)
    if not isinstance(state_obj, dict):
        raise RuntimeError(
            f"Checkpoint payload at {resolved_path} is expected to be a dict, got {type(state_obj)}"
        )

    for key in ("state_dict", "module"):
        payload = state_obj.get(key)
        if isinstance(payload, dict):
            return payload, resolved_path

    return state_obj, resolved_path


def extract_ucpe_camera_adapter_state_dict(checkpoint_path: str | Path):
    state_dict, resolved_path = load_ucpe_checkpoint_state_dict(checkpoint_path)

    mapped_state = {}
    for key, value in state_dict.items():
        if ".cam_self_attn." in key and (
            key.startswith("pipe.dit.blocks.") or key.startswith("blocks.")
        ):
            mapped_state[key] = value

    if not mapped_state:
        preview_keys = list(state_dict.keys())[:20]
        raise RuntimeError(
            "No UCPE camera adapter tensors found in checkpoint payload. "
            f"checkpoint={checkpoint_path} resolved_payload={resolved_path} "
            f"top_level_state_keys_preview={preview_keys}"
        )
    return mapped_state, resolved_path


def _build_stage_rope_coeffs(
    cameras: int,
    patches_y: int,
    patches_x: int,
    head_dim: int,
    freq_base: float,
    freq_scale: float,
    device: torch.device,
    dtype: torch.dtype,
):
    x_positions = torch.tile(torch.arange(patches_x, device=device), (patches_y * cameras,))
    y_positions = torch.tile(
        torch.repeat_interleave(torch.arange(patches_y, device=device), patches_x),
        (cameras,),
    )
    coeffs_x = prope_torch._rope_precompute_coeffs(
        x_positions,
        freq_base=freq_base,
        freq_scale=freq_scale,
        feat_dim=head_dim // 4,
        dtype=dtype,
    )
    coeffs_y = prope_torch._rope_precompute_coeffs(
        y_positions,
        freq_base=freq_base,
        freq_scale=freq_scale,
        feat_dim=head_dim // 4,
        dtype=dtype,
    )
    return coeffs_x, coeffs_y


def patch_worldcrafter_transformer_ucpe(
    transformer: WorldCrafterTransformer3DModel,
    method: str,
    height: int,
    width: int,
    attn_compress: int = 8,
    adaptation_method: str = "parallel",
    vae_scale_factor_spatial: int = 8,
    attention_cls: type[UcpeSelfAttention] = UcpeSelfAttention,
):
    if not any(key in method for key in ("gta", "prope", "relray")):
        raise ValueError(f"Only UCPE attention-style methods are supported, got: {method}")

    patch_factor = vae_scale_factor_spatial * transformer.config.patch_size[1]
    patches_x = width // patch_factor
    patches_y = height // patch_factor
    emb_dim = 3 if "absmap" in method else None

    for block in transformer.blocks:
        num_heads = block.attn1.heads // attn_compress
        if num_heads <= 0:
            raise ValueError(f"attn_compress={attn_compress} is too large for heads={block.attn1.heads}")
        hidden_dim = block.attn1.to_q.weight.shape[0]
        block.cam_self_attn = attention_cls(
            hidden_dim,
            hidden_dim // attn_compress,
            num_heads,
            patches_x=patches_x,
            patches_y=patches_y,
            image_width=width,
            image_height=height,
            emb_dim=emb_dim,
            adaptation_method=adaptation_method,
        )

    transformer.camera_condition = method
    return ["cam_self_attn"]


def load_ucpe_camera_adapter_weights(transformer: WorldCrafterTransformer3DModel, checkpoint_path: str):
    state_dict, resolved_path = extract_ucpe_camera_adapter_state_dict(checkpoint_path)
    mapped_state = {}
    for key, value in state_dict.items():
        if key.startswith("pipe.dit.blocks.") and ".cam_self_attn." in key:
            mapped_state[key.replace("pipe.dit.", "", 1)] = value
        elif key.startswith("blocks.") and ".cam_self_attn." in key:
            mapped_state[key] = value

    expected_cam_keys = {key for key in transformer.state_dict().keys() if ".cam_self_attn." in key}
    mapped_cam_keys = set(mapped_state.keys())
    missing_cam_keys = sorted(expected_cam_keys - mapped_cam_keys)
    unexpected_cam_keys = sorted(mapped_cam_keys - expected_cam_keys)

    if not mapped_state:
        raise RuntimeError(
            f"No UCPE camera adapter tensors found in checkpoint: {checkpoint_path} "
            f"(resolved_payload={resolved_path})"
        )
    if missing_cam_keys or unexpected_cam_keys:
        raise RuntimeError(
            "UCPE camera adapter checkpoint does not exactly match the patched transformer. "
            f"missing_cam_keys={len(missing_cam_keys)} unexpected_cam_keys={len(unexpected_cam_keys)} "
            f"resolved_payload={resolved_path}"
        )

    load_info = transformer.load_state_dict(mapped_state, strict=False)
    return {
        "resolved_payload_path": str(resolved_path),
        "loaded_tensor_keys": len(mapped_state),
        "expected_tensor_keys": len(expected_cam_keys),
        "missing_cam_keys": missing_cam_keys,
        "unexpected_cam_keys": unexpected_cam_keys,
        "missing_keys": list(load_info.missing_keys),
        "unexpected_keys": list(load_info.unexpected_keys),
    }


def _relative_pose_chunk(
    global_c2w: Any,
    *,
    chunk_index: int,
    window_num_frames: int,
    device: torch.device,
) -> torch.Tensor | None:
    if isinstance(global_c2w, torch.Tensor):
        pose = global_c2w.detach().cpu().numpy()
    else:
        pose = np.asarray(global_c2w)
    if pose.ndim == 3:
        pose = pose[None]
    if pose.ndim != 4 or pose.shape[0] != 1 or pose.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(
            "camera_trajectory['c2w'] must be [1,T,3,4] or [1,T,4,4], "
            f"got {pose.shape}"
        )
    start = int(chunk_index) * int(window_num_frames)
    end = start + int(window_num_frames)
    if pose.shape[1] < end:
        return None
    chunk = np.asarray(pose[0, start:end], dtype=np.float64)
    homogeneous = np.zeros((window_num_frames, 4, 4), dtype=np.float64)
    homogeneous[:, :3, :4] = chunk[:, :3, :4]
    homogeneous[:, 3, 3] = 1.0
    relative = np.linalg.inv(homogeneous[0])[None] @ homogeneous
    return torch.from_numpy(relative[:, :3, :4].astype(np.float32)).unsqueeze(0).to(device)


def build_ucpe_attention_kwargs_for_chunk(
    transformer: WorldCrafterTransformer3DModel,
    camera_trajectory: dict[str, Any] | None,
    height: int,
    width: int,
    num_latent_frames_per_chunk: int,
    chunk_index: int,
    vae_scale_factor_temporal: int = 4,
    token_grid_height: int | None = None,
    token_grid_width: int | None = None,
):
    if camera_trajectory is None or getattr(transformer, "camera_condition", "none") == "none":
        return None

    x_fov = camera_trajectory["x_fov"]
    xi = camera_trajectory["xi"]
    method = transformer.camera_condition

    if x_fov.ndim == 0:
        x_fov = x_fov.unsqueeze(0)
    if xi.ndim == 0:
        xi = xi.unsqueeze(0)

    window_num_frames = (num_latent_frames_per_chunk - 1) * vae_scale_factor_temporal + 1
    pose_chunk = _relative_pose_chunk(
        camera_trajectory["c2w"],
        chunk_index=chunk_index,
        window_num_frames=window_num_frames,
        device=x_fov.device,
    )
    if pose_chunk is None:
        return None
    if pose_chunk.shape[-2:] == (4, 4):
        pose_chunk = pose_chunk[..., :3, :4]
    elif pose_chunk.shape[-2:] != (3, 4):
        raise ValueError(
            "pose_chunk is expected to be [B, T, 3, 4] or [B, T, 4, 4], "
            f"got shape={tuple(pose_chunk.shape)}"
        )
    pose_chunk = pose_chunk[:, ::vae_scale_factor_temporal].to(dtype=torch.float32)
    c2w = torch.eye(4, device=pose_chunk.device, dtype=pose_chunk.dtype)
    c2w = repeat(c2w, "... -> B T ...", B=pose_chunk.shape[0], T=pose_chunk.shape[1]).clone()
    c2w[..., :3, :4] = pose_chunk

    if "gta" in method or "prope" in method:
        raise NotImplementedError("Only relray_absmap is supported.")

    if "relray" not in method:
        raise ValueError(f"Unsupported camera condition: {method}")

    attn = transformer.blocks[0].cam_self_attn
    grid_h = token_grid_height if token_grid_height is not None else attn.patches_y
    grid_w = token_grid_width if token_grid_width is not None else attn.patches_x

    d_cam = ucpe_cc.ucm_unproject_grid_fov(
        x_fov=x_fov,
        xi=xi,
        height=grid_h,
        width=grid_w,
        device=pose_chunk.device,
        dtype=pose_chunk.dtype,
    )
    raymats = ucpe_cc.world_to_ray_mats(d_cam, c2w)
    viewmats = rearrange(raymats, "B T H W ... -> B (T H W) ...")

    control_camera_dit_input = {"viewmats": viewmats}

    if token_grid_height is not None or token_grid_width is not None:
        coeffs_x, coeffs_y = _build_stage_rope_coeffs(
            cameras=pose_chunk.shape[1],
            patches_y=grid_h,
            patches_x=grid_w,
            head_dim=attn.head_dim,
            freq_base=attn.freq_base,
            freq_scale=attn.freq_scale,
            device=pose_chunk.device,
            dtype=pose_chunk.dtype,
        )
        control_camera_dit_input["coeffs_x"] = coeffs_x
        control_camera_dit_input["coeffs_y"] = coeffs_y

    if "absmap" in method:
        up_map, lat_map = ucpe_cc.compute_up_lat_map(
            R=pose_chunk[..., :3, :3],
            x_fov=x_fov,
            xi=xi,
            width=grid_w,
            height=grid_h,
            device=pose_chunk.device,
        )
        cam_emb = torch.cat([up_map, lat_map], dim=-1)
        cam_emb = rearrange(cam_emb, "B T H W C -> B (T H W) C")
        control_camera_dit_input["cam_emb"] = cam_emb

    expected_token_count = pose_chunk.shape[1] * grid_h * grid_w
    if control_camera_dit_input["viewmats"].shape[1] != expected_token_count:
        raise ValueError(
            "camera viewmats token count mismatch: "
            f"expected {expected_token_count}, got {control_camera_dit_input['viewmats'].shape[1]}"
        )
    if "cam_emb" in control_camera_dit_input and control_camera_dit_input["cam_emb"].shape[1] != expected_token_count:
        raise ValueError(
            "camera cam_emb token count mismatch: "
            f"expected {expected_token_count}, got {control_camera_dit_input['cam_emb'].shape[1]}"
        )

    return {"camera_control_ucpe_input": control_camera_dit_input}
