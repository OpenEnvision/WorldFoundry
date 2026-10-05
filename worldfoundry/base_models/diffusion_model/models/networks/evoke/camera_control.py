"""Camera control module: Plucker embedding, per-block AdaLN modulation, and weight I/O.

Parameter names match lingbot checkpoint keys 1:1 so no renaming is needed on load.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# Substring tokens that identify camera-control parameters by name.
CAM_PARAM_TOKENS = (
    # Per-block injector MLP projections.
    "cam_inj1_down_proj",
    "cam_inj1_up_proj",
    "cam_inj2_down_proj",
    "cam_inj2_up_proj",
    # Per-block AdaLN-zero scale/shift projections.
    "cam_scale_down_proj",
    "cam_scale_up_proj",
    "cam_shift_down_proj",
    "cam_shift_up_proj",
    # Global Plucker encoder (lingbot ckpt key names).
    "patch_embedding_wancamctrl",
    "c2ws_hidden_states_layer1",
    "c2ws_hidden_states_layer2",
)


def is_cam_param_name(name: str) -> bool:
    """Return True if the parameter name belongs to the camera control module."""
    return any(tok in name for tok in CAM_PARAM_TOKENS)


def get_plucker_input_dim(
    cam_ctrl_dim: int = 6,
    vae_h_stride: int = 8,
    vae_w_stride: int = 8,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> int:
    """Compute the input dimension for patch_embedding_wancamctrl (default 1536)."""
    pt, ph, pw = patch_size
    return cam_ctrl_dim * vae_h_stride * vae_w_stride * pt * ph * pw


@torch.no_grad()
def compute_plucker_embedding_from_Ks(
    cam_c2w: torch.Tensor,
    Ks: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Compute per-pixel Plucker embedding for one sample.

    Args:
        cam_c2w: [F, 4, 4] framewise relative c2w poses
        Ks: [F, 4] intrinsics [fx, fy, cx, cy]
    Returns:
        plucker: [F, H, W, 6] = concat(rays_o, rays_d)
    """
    F_frames = cam_c2w.shape[0]
    device = cam_c2w.device
    dtype = cam_c2w.dtype

    x_range = torch.arange(width, device=device, dtype=dtype)
    y_range = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2) + 0.5  # [H*W, 2] pixel centers
    grid_xy = grid_xy.unsqueeze(0).expand(F_frames, -1, -1)  # [F, H*W, 2]

    fx, fy, cx, cy = Ks.chunk(4, dim=-1)  # [F, 1] each
    i = grid_xy[..., 0]
    j = grid_xy[..., 1]
    xs = (i - cx) / fx
    ys = (j - cy) / fy
    zs = torch.ones_like(xs)

    directions = torch.stack([xs, ys, zs], dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ cam_c2w[:, :3, :3].transpose(-1, -2)  # [F, H*W, 3] world-space directions
    rays_o = cam_c2w[:, :3, 3].unsqueeze(1).expand_as(rays_d)   # [F, H*W, 3] camera origins

    plucker = torch.cat([rays_o, rays_d], dim=-1).view(F_frames, height, width, 6)
    return plucker


def prepare_plucker_for_model(
    plucker: torch.Tensor,
    vae_stride_t: int = 4,
    vae_stride_h: int = 8,
    vae_stride_w: int = 8,
) -> torch.Tensor:
    """Patchify spatial dims: [F_lat, H, W, 6] -> [1, C, F_lat, H_lat, W_lat]. vae_stride_t unused (API compat)."""
    plucker = rearrange(
        plucker,
        "f (h c1) (w c2) c -> 1 (c c1 c2) f h w",
        c1=vae_stride_h, c2=vae_stride_w,
    )
    return plucker


def _to_latent_rate_relative(
    c2ws_F_abs: torch.Tensor, vae_stride_t: int = 4,
) -> torch.Tensor:
    """Convert absolute pixel-rate c2ws to latent-rate framewise relative poses with translation max-norm."""
    from .geometry import compute_relative_poses_lingbot
    F_pix = c2ws_F_abs.shape[0]
    F_lat = (F_pix - 1) // vae_stride_t + 1
    lat_indices = torch.arange(F_lat, device=c2ws_F_abs.device) * vae_stride_t
    lat_indices = lat_indices.clamp(max=F_pix - 1)
    c2ws_lat_abs = c2ws_F_abs[lat_indices]  # [F_lat, 4, 4]
    c2ws_lat_rel = compute_relative_poses_lingbot(
        c2ws_lat_abs.float(), framewise=True, normalize_trans=True,
    )
    return c2ws_lat_rel


def _build_plucker_one_sample(
    c2ws_F: torch.Tensor,        # [F_pix, 4, 4] absolute pixel-rate c2w
    Ks_4: torch.Tensor,           # [4] base-resolution intrinsics [fx, fy, cx, cy]
    H_pix: int, W_pix: int,
    base_h_pix: int, base_w_pix: int,
    strategy: str = "scale_ks",
    vae_stride_t: int = 4,
) -> torch.Tensor:
    """Build Plucker embedding [F_lat, H_pix, W_pix, 6] for one sample."""
    # Convert to latent-rate framewise relative poses before computing Plucker.
    c2ws_lat = _to_latent_rate_relative(c2ws_F, vae_stride_t=vae_stride_t)  # [F_lat, 4, 4]

    if strategy == "scale_ks":
        s_w = W_pix / base_w_pix
        s_h = H_pix / base_h_pix
        Ks_stage = Ks_4 * Ks_4.new_tensor([s_w, s_h, s_w, s_h])
        Ks_F_lat = Ks_stage.unsqueeze(0).expand(c2ws_lat.shape[0], -1)
        return compute_plucker_embedding_from_Ks(c2ws_lat, Ks_F_lat, H_pix, W_pix)
    if strategy == "resample":
        Ks_base_F = Ks_4.unsqueeze(0).expand(c2ws_lat.shape[0], -1)
        full = compute_plucker_embedding_from_Ks(c2ws_lat, Ks_base_F, base_h_pix, base_w_pix)
        small = F.interpolate(
            full.permute(0, 3, 1, 2),
            size=(H_pix, W_pix), mode="bilinear", align_corners=False,
        )
        return small.permute(0, 2, 3, 1).contiguous()
    raise ValueError(f"unknown pc_resolution_strategy: {strategy!r} (expect 'scale_ks' or 'resample')")


def prepare_cam_plucker_emb(
    Ks: torch.Tensor,
    c2ws_window: torch.Tensor,
    H_pix: int, W_pix: int,
    base_h_pix: int, base_w_pix: int,
    vae_stride_t: int = 4,
    vae_stride_h: int = 8,
    vae_stride_w: int = 8,
    strategy: str = "scale_ks",
) -> torch.Tensor:
    """Compute batched cam_plucker_emb for a single resolution. Returns [B, 384, F_lat, H_lat, W_lat]."""
    assert Ks.dim() == 2 and Ks.shape[1] == 4, f"Ks expected [B,4], got {tuple(Ks.shape)}"
    assert c2ws_window.dim() == 4 and tuple(c2ws_window.shape[-2:]) == (4, 4), (
        f"c2ws_window expected [B,F,4,4], got {tuple(c2ws_window.shape)}"
    )
    B = Ks.shape[0]
    per_sample = []
    for b in range(B):
        plucker = _build_plucker_one_sample(
            c2ws_window[b], Ks[b],
            H_pix, W_pix, base_h_pix, base_w_pix, strategy,
            vae_stride_t=vae_stride_t,
        )
        per_sample.append(
            prepare_plucker_for_model(plucker, vae_stride_t, vae_stride_h, vae_stride_w)
        )
    return torch.cat(per_sample, dim=0)


def prepare_cam_plucker_for_list(
    noisy_model_input_list,
    lingbot_Ks: torch.Tensor,
    lingbot_c2ws_window: torch.Tensor,
    base_height_pix: int,
    base_width_pix: int,
    vae_stride_t: int = 4,
    vae_stride_h: int = 8,
    vae_stride_w: int = 8,
    strategy: str = "scale_ks",
):
    """Build one cam_plucker_emb per noisy input in the NaViT pyramid list (recomputed per resolution)."""
    cam_emb_list = []
    for noisy in noisy_model_input_list:
        B, _, T_lat, H_lat, W_lat = noisy.shape
        H_pix = H_lat * vae_stride_h
        W_pix = W_lat * vae_stride_w
        per_sample = []
        for b in range(B):
            plucker = _build_plucker_one_sample(
                lingbot_c2ws_window[b], lingbot_Ks[b],
                H_pix, W_pix, base_height_pix, base_width_pix, strategy,
                vae_stride_t=vae_stride_t,
            )
            per_sample.append(
                prepare_plucker_for_model(plucker, vae_stride_t, vae_stride_h, vae_stride_w)
            )
        cam_emb_list.append(torch.cat(per_sample, dim=0))
    return cam_emb_list


def build_camera_plucker_encoder_submodules(
    parent: nn.Module, dim: int, plucker_input_dim: int,
):
    """Register the global Plucker encoder submodules on parent with lingbot-matching attribute names."""
    parent.patch_embedding_wancamctrl = nn.Linear(plucker_input_dim, dim)
    parent.c2ws_hidden_states_layer1 = nn.Linear(dim, dim)
    parent.c2ws_hidden_states_layer2 = nn.Linear(dim, dim)
    nn.init.zeros_(parent.patch_embedding_wancamctrl.weight)
    nn.init.zeros_(parent.patch_embedding_wancamctrl.bias)
    nn.init.zeros_(parent.c2ws_hidden_states_layer1.weight)
    nn.init.zeros_(parent.c2ws_hidden_states_layer1.bias)
    nn.init.zeros_(parent.c2ws_hidden_states_layer2.weight)
    nn.init.zeros_(parent.c2ws_hidden_states_layer2.bias)


def build_camera_modulation_lowrank_submodules(
    block: nn.Module, dim: int, cam_rank: int = 128,
):
    """Register lingbot-style LoRA per-block cam modulation submodules on block. All zero-initialized."""
    # Injector MLP (LoRA-ized): inj1 -> silu -> inj2 -> residual.
    block.cam_inj1_down_proj = nn.Linear(dim, cam_rank, bias=False)
    block.cam_inj1_up_proj   = nn.Linear(cam_rank, dim, bias=True)
    block.cam_inj2_down_proj = nn.Linear(dim, cam_rank, bias=False)
    block.cam_inj2_up_proj   = nn.Linear(cam_rank, dim, bias=True)
    # AdaLN-zero scale/shift LoRA pairs.
    block.cam_scale_down_proj = nn.Linear(dim, cam_rank, bias=False)
    block.cam_scale_up_proj   = nn.Linear(cam_rank, dim, bias=True)
    block.cam_shift_down_proj = nn.Linear(dim, cam_rank, bias=False)
    block.cam_shift_up_proj   = nn.Linear(cam_rank, dim, bias=True)
    # Zero-init all; reinit_cam_modules() switches to Xavier / ckpt.
    for proj in (
        block.cam_inj1_down_proj, block.cam_inj1_up_proj,
        block.cam_inj2_down_proj, block.cam_inj2_up_proj,
        block.cam_scale_down_proj, block.cam_scale_up_proj,
        block.cam_shift_down_proj, block.cam_shift_up_proj,
    ):
        nn.init.zeros_(proj.weight)
        if proj.bias is not None:
            nn.init.zeros_(proj.bias)


def process_cam_plucker_to_tokens(
    cam_plucker_emb: torch.Tensor,
    patch_embedding_wancamctrl: nn.Linear,
    c2ws_hidden_states_layer1: nn.Linear,
    c2ws_hidden_states_layer2: nn.Linear,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> torch.Tensor:
    """Project and encode Plucker tokens: [B, 384, F_lat, H_lat, W_lat] -> [B, f*h*w, dim]."""
    pt, ph, pw = patch_size
    processed = rearrange(
        cam_plucker_emb,
        "b c (f pt) (h ph) (w pw) -> b (f h w) (c pt ph pw)",
        pt=pt, ph=ph, pw=pw,
    )
    processed = patch_embedding_wancamctrl(processed)
    hidden = c2ws_hidden_states_layer2(F.silu(c2ws_hidden_states_layer1(processed)))
    return processed + hidden


def apply_cam_modulation(
    hidden_states: torch.Tensor,
    cam_token_seq: torch.Tensor,
    cam_inj1_down_proj: nn.Linear,
    cam_inj1_up_proj: nn.Linear,
    cam_inj2_down_proj: nn.Linear,
    cam_inj2_up_proj: nn.Linear,
    cam_scale_down_proj: nn.Linear,
    cam_scale_up_proj: nn.Linear,
    cam_shift_down_proj: nn.Linear,
    cam_shift_up_proj: nn.Linear,
    noise_slots,
) -> torch.Tensor:
    """Apply lingbot-style LoRA per-block AdaLN modulation to noise slots only; prefix tokens are unchanged.

    noise_slots: single (s, e) tuple or list of (s, e) pairs.
    cam_token_seq length must equal sum of slot lengths, aligned to noise slot order.
    """
    # Normalize single tuple to a list.
    if isinstance(noise_slots, tuple) and len(noise_slots) == 2 and not isinstance(noise_slots[0], tuple):
        slots = [noise_slots]
    else:
        slots = list(noise_slots)

    total_noise = sum(e - s for s, e in slots)
    assert cam_token_seq.shape[1] == total_noise, (
        f"cam_token_seq len {cam_token_seq.shape[1]} != sum(noise_slots lens) {total_noise}; "
        f"slots={slots}"
    )

    # 1) Injector MLP refinement: down -> silu -> up, with residual.
    inj_hidden = cam_inj1_up_proj(cam_inj1_down_proj(cam_token_seq))
    inj_hidden = F.silu(inj_hidden)
    inj_hidden = cam_inj2_up_proj(cam_inj2_down_proj(inj_hidden))
    cam_token_refined = cam_token_seq + inj_hidden

    # 2) Compute scale and shift for all noise tokens at once.
    cam_scale_all = cam_scale_up_proj(cam_scale_down_proj(cam_token_refined))
    cam_shift_all = cam_shift_up_proj(cam_shift_down_proj(cam_token_refined))

    # 3) Scatter modulation into noise slots; prefix tokens are left untouched.
    out = hidden_states.clone()
    cam_offset = 0
    for s, e in slots:
        seg_len = e - s
        cs = cam_scale_all[:, cam_offset : cam_offset + seg_len]
        cb = cam_shift_all[:, cam_offset : cam_offset + seg_len]
        out[:, s:e] = (1.0 + cs) * hidden_states[:, s:e] + cb
        cam_offset += seg_len
    return out


def iter_cam_state_dict_keys(
    num_layers: int,
    cam_ctrl_layers: Optional[Iterable[int]] = None,
):
    """Yield all expected cam state_dict keys: 6 global + 12 per active layer."""
    yield "patch_embedding_wancamctrl.weight"
    yield "patch_embedding_wancamctrl.bias"
    yield "c2ws_hidden_states_layer1.weight"
    yield "c2ws_hidden_states_layer1.bias"
    yield "c2ws_hidden_states_layer2.weight"
    yield "c2ws_hidden_states_layer2.bias"
    layer_idxs = list(range(num_layers)) if cam_ctrl_layers is None else list(cam_ctrl_layers)
    for i in layer_idxs:
        for base in ("cam_inj1", "cam_inj2", "cam_scale", "cam_shift"):
            yield f"blocks.{i}.{base}_down_proj.weight"
            yield f"blocks.{i}.{base}_up_proj.weight"
            yield f"blocks.{i}.{base}_up_proj.bias"


def _load_safetensors_any(path: str) -> dict[str, torch.Tensor]:
    """Load safetensors from a file or directory (index.json supported); strips _orig_mod. prefix."""
    from safetensors.torch import load_file

    state: dict[str, torch.Tensor] = {}

    def _collect(st: dict[str, torch.Tensor]):
        for k, v in st.items():
            state[k.replace("_orig_mod.", "")] = v

    if os.path.isfile(path):
        _collect(load_file(path))
        return state
    if not os.path.isdir(path):
        raise FileNotFoundError(f"camera ckpt path not found: {path}")

    index_path = os.path.join(path, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        cam_files = set()
        for k, fname in index["weight_map"].items():
            if is_cam_param_name(k.replace("_orig_mod.", "")):
                cam_files.add(fname)
        for fname in sorted(cam_files):
            _collect(load_file(os.path.join(path, fname)))
    else:
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".safetensors"):
                _collect(load_file(os.path.join(path, fname)))
    return state


def load_camera_ctrl_weights(
    model: nn.Module,
    ckpt_path: str,
    num_layers: int,
    cam_ctrl_layers: Optional[Iterable[int]] = None,
    strict: bool = True,
) -> dict:
    """Load cam weights from ckpt into model.

    strict=True (default): missing keys or shape mismatches raise ValueError.
    strict=False: missing/mismatch keys are skipped with a warning.
    Returns a report dict {expected, loaded, missing, shape_mismatch, extras}.
    """
    raw = _load_safetensors_any(ckpt_path)
    cam_state = {k: v for k, v in raw.items() if is_cam_param_name(k)}
    expected = list(iter_cam_state_dict_keys(num_layers, cam_ctrl_layers))
    expected_set = set(expected)

    missing = [k for k in expected if k not in cam_state]
    extras = [k for k in cam_state if k not in expected_set]

    model_sd = model.state_dict()
    cam_not_in_model = [k for k in expected if k not in model_sd]
    if cam_not_in_model:
        raise ValueError(
            f"[CamCtrl] model does not expose expected cam attrs (camera_control likely off): "
            f"{cam_not_in_model[:5]} (total {len(cam_not_in_model)})"
        )

    shape_mismatch = []
    for k in expected:
        if k in cam_state and tuple(cam_state[k].shape) != tuple(model_sd[k].shape):
            shape_mismatch.append((k, tuple(cam_state[k].shape), tuple(model_sd[k].shape)))

    if strict:
        errs = []
        if missing:
            errs.append(
                f"{len(missing)} cam key(s) missing from ckpt; first 5: {missing[:5]}"
            )
        if shape_mismatch:
            errs.append(
                f"{len(shape_mismatch)} shape mismatch(es); first 3: {shape_mismatch[:3]}"
            )
        if errs:
            raise ValueError(
                f"[CamCtrl] strict load failed for {ckpt_path}:\n  " + "\n  ".join(errs)
            )
    else:
        if missing:
            print(f"[CamCtrl] load (non-strict): {len(missing)} missing, skip. first 5: {missing[:5]}")
        if shape_mismatch:
            print(f"[CamCtrl] load (non-strict): {len(shape_mismatch)} shape mismatch, skip.")

    skip_keys = {m[0] for m in shape_mismatch}
    load_sd = {k: v for k, v in cam_state.items() if k in expected_set and k not in skip_keys}
    model.load_state_dict(load_sd, strict=False)

    if extras:
        print(f"[CamCtrl] {len(extras)} extra non-expected cam-named key(s) ignored. first 3: {extras[:3]}")
    print(
        f"[CamCtrl] Loaded {len(load_sd)}/{len(expected)} cam keys from {ckpt_path} "
        f"(strict={strict})"
    )
    return {
        "expected": len(expected),
        "loaded": len(load_sd),
        "missing": missing,
        "shape_mismatch": shape_mismatch,
        "extras": extras,
    }


def save_camera_ctrl_weights(model: nn.Module, save_path: str):
    """Save all cam parameters to a safetensors file (or <dir>/camera_ctrl.safetensors if path is a dir)."""
    from safetensors.torch import save_file

    if save_path.endswith(".safetensors"):
        target = save_path
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    else:
        os.makedirs(save_path, exist_ok=True)
        target = os.path.join(save_path, "camera_ctrl.safetensors")

    cam_sd = {
        k: v.detach().contiguous().cpu()
        for k, v in model.state_dict().items()
        if is_cam_param_name(k)
    }
    if not cam_sd:
        print(f"[CamCtrl] WARNING: no cam params on model, skip save to {target}")
        return None
    save_file(cam_sd, target)
    print(f"[CamCtrl] Saved {len(cam_sd)} cam keys to {target}")
    return target


def _has_cam_loaded_sentinel(model: nn.Module) -> bool:
    """Return True if patch_embedding_wancamctrl.weight is non-zero, indicating a ckpt has been loaded.

    Uses patch_embedding_wancamctrl (registered after init_weights()) as the sentinel so that
    init_weights()-driven Xavier on block cam_*_down_proj weights does not give a false positive.
    """
    for n, p in model.named_parameters():
        if "patch_embedding_wancamctrl.weight" in n:
            return float(p.detach().abs().max().item()) > 1e-8
    return False


def reinit_cam_modules(
    model: nn.Module,
    num_layers: int,
    cam_weight_path: Optional[str] = None,
    cam_ctrl_layers: Optional[Iterable[int]] = None,
    strict: bool = True,
):
    """Initialize cam weights: (1) load from path, (2) keep if sentinel positive, (3) Xavier fallback."""
    if cam_weight_path is not None:
        print(f"[CamCtrl] reinit: explicit ckpt path {cam_weight_path} (strict={strict})")
        load_camera_ctrl_weights(model, cam_weight_path, num_layers, cam_ctrl_layers, strict=strict)
        return

    if _has_cam_loaded_sentinel(model):
        print("[CamCtrl] reinit: sentinel positive (down_proj non-zero), keep existing cam params.")
        return

    print("[CamCtrl] reinit: Xavier fallback (down=xavier, up=0, patch_emb=xavier, layer1=normal-std0.02, layer2=zeros)")
    if hasattr(model, "patch_embedding_wancamctrl"):
        nn.init.xavier_uniform_(model.patch_embedding_wancamctrl.weight)
        nn.init.zeros_(model.patch_embedding_wancamctrl.bias)
        # layer1 gets small normal init to break the zero-zero deadlock; layer2 stays zero.
        nn.init.normal_(model.c2ws_hidden_states_layer1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(model.c2ws_hidden_states_layer1.bias)
        nn.init.zeros_(model.c2ws_hidden_states_layer2.weight)
        nn.init.zeros_(model.c2ws_hidden_states_layer2.bias)

    if hasattr(model, "blocks"):
        layer_idxs = range(num_layers) if cam_ctrl_layers is None else list(cam_ctrl_layers)
        for i in layer_idxs:
            block = model.blocks[i]
            if hasattr(block, "cam_scale_down_proj"):
                # Inj path: full Xavier; AdaLN-zero still guaranteed by scale/shift up=0.
                nn.init.xavier_uniform_(block.cam_inj1_down_proj.weight)
                nn.init.xavier_uniform_(block.cam_inj1_up_proj.weight)
                nn.init.zeros_(block.cam_inj1_up_proj.bias)
                nn.init.xavier_uniform_(block.cam_inj2_down_proj.weight)
                nn.init.xavier_uniform_(block.cam_inj2_up_proj.weight)
                nn.init.zeros_(block.cam_inj2_up_proj.bias)
                # AdaLN-zero gates: scale_up and shift_up weights stay zero.
                nn.init.xavier_uniform_(block.cam_scale_down_proj.weight)
                nn.init.zeros_(block.cam_scale_up_proj.weight)
                nn.init.zeros_(block.cam_scale_up_proj.bias)
                nn.init.xavier_uniform_(block.cam_shift_down_proj.weight)
                nn.init.zeros_(block.cam_shift_up_proj.weight)
                nn.init.zeros_(block.cam_shift_up_proj.bias)


def collect_cam_named_parameters(model: nn.Module):
    """Yield (name, param) for all cam-related parameters."""
    for n, p in model.named_parameters():
        if is_cam_param_name(n):
            yield n, p


def set_camera_only_trainable(model: nn.Module, verbose: bool = True) -> int:
    """Freeze all params except cam-related ones. Returns the count of trainable tensors."""
    cam_names = set()
    for n, p in model.named_parameters():
        if is_cam_param_name(n):
            p.requires_grad_(True)
            cam_names.add(n)
        else:
            p.requires_grad_(False)

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert set(n for n, _ in trainable) == cam_names, (
        "camera-only freeze: trainable set != cam-name set; "
        f"diff: {set(n for n, _ in trainable) ^ cam_names}"
    )
    if verbose:
        total = sum(p.numel() for _, p in trainable)
        print(f"[CamCtrl] camera-only trainable: {len(trainable)} tensors, total numel = {total:,}")
        for n, p in trainable[:10]:
            print(f"  - {n}  shape={tuple(p.shape)}")
        if len(trainable) > 10:
            print(f"  ... and {len(trainable) - 10} more")
    return len(trainable)
