# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
from functools import lru_cache

import torch
from torch import nn
from .prope import PropeDotProductAttention
from worldfoundry.core.attention import packed_sequence_attention as flash_attention
from einops import rearrange, repeat, einsum
import torch.nn.functional as F


def compute_fx_from_fov_xi(
    x_fov: torch.Tensor | float,
    xi: torch.Tensor | float,
    width: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    根据水平视场角 (x_fov) 和 UCM 参数 (xi) 计算相机焦距 fx。

    Args:
        x_fov: float 或 [B] Tensor，水平视场角（单位：度）
        xi: float 或 [B] Tensor，UCM 镜面参数
        width: 图像宽度（像素）
        device: torch.device
        dtype: torch.dtype

    Returns:
        fx: [B] Tensor，焦距（像素单位）
    """

    # --- 转为 Tensor ---
    def to_tensor_1d(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype).reshape(-1)
        return torch.tensor([x], dtype=dtype, device=device)

    x_fov = to_tensor_1d(x_fov)
    xi = to_tensor_1d(xi)

    # --- 自动广播 ---
    B = max(x_fov.shape[0], xi.shape[0])
    x_fov = x_fov.view(-1).expand(B)
    xi = xi.view(-1).expand(B)

    # --- 计算 fx ---
    theta = torch.deg2rad(0.5 * x_fov)
    eps = torch.finfo(dtype).eps
    denom = torch.sin(theta).clamp_min(eps)
    fx = (width * 0.5) * (torch.cos(theta) + xi) / denom
    return fx


def project_ucm_points_fov(X, Y, Z, x_fov, xi, height, width):
    fx = compute_fx_from_fov_xi(x_fov, xi, width, X.device, X.dtype)
    return project_ucm_points(X, Y, Z, fx, fx, width / 2, height / 2, xi)


def project_ucm_points(X, Y, Z, fx, fy, cx, cy, xi):
    def broadcast_param(param):
        if not torch.is_tensor(param):
            param = torch.tensor(param, device=X.device, dtype=X.dtype)
        else:
            param = param.to(device=X.device, dtype=X.dtype)
        if param.ndim == 0:
            return param
        flat = param.reshape(-1)
        if flat.numel() == 1:
            return flat.view(1)
        if X.ndim >= 1 and flat.numel() == X.shape[0]:
            return flat.view(flat.shape[0], *([1] * (X.ndim - 1)))
        return param

    fx = broadcast_param(fx)
    fy = broadcast_param(fy)
    cx = broadcast_param(cx)
    cy = broadcast_param(cy)
    xi = broadcast_param(xi)
    radius = torch.sqrt(X * X + Y * Y + Z * Z)
    denominator = Z + xi * radius
    du = fx * (X / denominator) + cx
    dv = fy * (Y / denominator) + cy
    return du, dv


def _pixel_grid(
    *,
    height: int,
    width: int,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    xs = torch.linspace(0, width - 1, width, dtype=dtype, device=device)
    ys = torch.linspace(0, height - 1, height, dtype=dtype, device=device)
    ys, xs = torch.meshgrid([ys, xs], indexing="ij")
    grid = torch.stack((xs, ys, torch.ones_like(xs)), dim=2)
    return repeat(grid, "... -> b ...", b=batch)


@lru_cache(maxsize=128)
def _ucm_unproject_grid(
    height: int,
    width: int,
    fx: float | torch.Tensor,
    fy: float | torch.Tensor,
    cx: float | torch.Tensor,
    cy: float | torch.Tensor,
    xi: float | torch.Tensor,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    y_down: bool = True,
) -> torch.Tensor:
    scalar_input = all(not torch.is_tensor(value) for value in (fx, fy, cx, cy, xi))

    def tensor_1d(value):
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype)
        return torch.tensor([value], device=device, dtype=dtype)

    fx, fy, cx, cy, xi = map(tensor_1d, (fx, fy, cx, cy, xi))
    xs = torch.linspace(0, width - 1, width, dtype=dtype, device=device)
    ys = torch.linspace(0, height - 1, height, dtype=dtype, device=device)
    ys, xs = torch.meshgrid([ys, xs], indexing="ij")
    grid = torch.stack((xs, ys, torch.ones_like(xs)), dim=2)
    grid = repeat(grid, "... -> b ...", b=fx.shape[0])

    x = (grid[..., 0] - cx[:, None, None]) / fx[:, None, None]
    y = (grid[..., 1] - cy[:, None, None]) / fy[:, None, None]
    if not y_down:
        y = -y
    r2 = x * x + y * y
    alpha = xi[:, None, None] + torch.sqrt(
        1 + (1 - xi[:, None, None] * xi[:, None, None]) * r2
    )
    gamma = alpha / (1 + r2)
    directions = torch.stack((gamma * x, gamma * y, gamma - xi[:, None, None]), dim=-1)
    return directions[0] if scalar_input else directions


def ucm_unproject_grid_fov(
    x_fov: float | torch.Tensor,
    xi: float | torch.Tensor,
    height: int,
    width: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    计算每个样本的相机方向向量 (UCM model, 用视场角定义)。
    支持 float 或 [B] Tensor 的混合输入。
    - 若全为 float → 返回 [H, W, 3]
    - 若任意为 [B] → 返回 [B, H, W, 3]
    """
    if isinstance(device, str):
        device = torch.device(device)

    is_batched = any(
        torch.is_tensor(p) and p.reshape(-1).numel() > 1 for p in [x_fov, xi]
    )

    # --- 计算 fx, fy ---
    fx = compute_fx_from_fov_xi(x_fov, xi, width, device, dtype)
    fy = fx
    xi_grid = (
        xi.to(device=device, dtype=dtype).reshape(-1)
        if torch.is_tensor(xi)
        else torch.tensor([xi], dtype=dtype, device=device)
    )

    d_cam = _ucm_unproject_grid(
        height=height,
        width=width,
        fx=fx,
        fy=fy,
        cx=width / 2,
        cy=height / 2,
        xi=xi_grid,
        dtype=dtype,
        device=device,
        y_down=True,
    )

    # --- 输出 shape 控制 ---
    if not is_batched:
        d_cam = d_cam[0]  # [H, W, 3]

    return d_cam


def d_cam_to_angles(d_cam: torch.Tensor) -> torch.Tensor:
    """
    将方向向量 [x, y, z] 转换为 [azimuth, elevation]。
    坐标系：z前，x右，y下（符合 UCM 投影输出）

    输入: d_cam: [B, H, W, 3]
    输出: angles: [B, H, W, 2] — azimuth, elevation （单位: 弧度）
    """
    d_unit = F.normalize(d_cam, dim=-1)  # [B, H, W, 3]

    x = d_unit[..., 0]  # right
    y = d_unit[..., 1]  # down
    z = d_unit[..., 2]  # forward

    # yaw / azimuth: angle in xz-plane
    azimuth = torch.atan2(x, z)  # ∈ [-π, π]

    # pitch / elevation: angle above xz-plane
    elevation = -torch.asin(y)  # y 向下 → elevation = -asin(y)

    return torch.stack([azimuth, elevation], dim=-1)  # [B, H, W, 2]


def world_to_ray_mats(
    d_cam: torch.Tensor,  # [B, H, W, 3]
    c2w: torch.Tensor,  # [B, T, 4, 4]
) -> torch.Tensor:
    """
    构造每条 ray 的世界到 ray 局部坐标系的变换矩阵 world2ray。
    坐标系定义：
        - z: ray direction
        - x: cam_y × ray_dir
        - y: z × x
    返回:
        raymats: [B, T, H, W, 4, 4]
    """
    if d_cam.ndim == 3:
        d_cam = d_cam.unsqueeze(0)
    if c2w.ndim == 3:
        c2w = c2w.unsqueeze(0)
    if d_cam.ndim != 4 or d_cam.shape[-1] != 3:
        raise ValueError(
            f"d_cam must have shape [H,W,3] or [B,H,W,3], got {tuple(d_cam.shape)}"
        )
    if c2w.ndim != 4 or c2w.shape[-2:] != (4, 4):
        raise ValueError(
            f"c2w must have shape [T,4,4] or [B,T,4,4], got {tuple(c2w.shape)}"
        )
    if d_cam.shape[0] == 1 and c2w.shape[0] != 1:
        d_cam = d_cam.expand(c2w.shape[0], -1, -1, -1)
    elif c2w.shape[0] == 1 and d_cam.shape[0] != 1:
        c2w = c2w.expand(d_cam.shape[0], -1, -1, -1)
    elif d_cam.shape[0] != c2w.shape[0]:
        raise ValueError(
            f"d_cam and c2w batch mismatch: {d_cam.shape[0]} vs {c2w.shape[0]}"
        )

    B, H, W, _ = d_cam.shape
    T = c2w.shape[1]
    device = d_cam.device
    dtype = d_cam.dtype

    # --- Expand ray dirs across frames ---
    # [B,H,W,3] -> [B,T,H,W,3]
    d_cam = repeat(d_cam, "b h w c -> b t h w c", t=T)

    # extract camera R,t
    R_cam = c2w[..., :3, :3]  # [B,T,3,3]
    t_cam = c2w[..., :3, 3]  # [B,T,3]

    # --- d_world: rotate ray directions into world ---
    d_world = einsum(R_cam, d_cam, "b t i j, b t h w j -> b t h w i")

    # camera y-axis from each view
    cam_y = R_cam[..., :, 1]  # [B,T,3]
    cam_y = repeat(cam_y, "b t c -> b t h w c", h=H, w=W)

    # === Construct orthonormal ray-local axes ===
    z_ray = F.normalize(d_world, dim=-1, eps=1e-6)
    x_ray = torch.cross(cam_y, z_ray, dim=-1)
    x_ray = F.normalize(x_ray, dim=-1, eps=1e-6)
    y_ray = torch.cross(z_ray, x_ray, dim=-1)
    y_ray = F.normalize(y_ray, dim=-1, eps=1e-6)

    # local->world rotation
    R_l2w = torch.stack([x_ray, y_ray, z_ray], dim=-1)  # [B,T,H,W,3,3]

    # world->local rotation (transpose)
    R_w2l = rearrange(R_l2w, "b t h w i j -> b t h w j i")  # ✅

    # broadcast camera center
    t_world = repeat(t_cam, "b t c -> b t h w c", h=H, w=W)

    # world->local translation
    t_w2l = -einsum(R_w2l, t_world, "b t h w i j, b t h w j -> b t h w i")

    # assemble transform matrix
    raymats = torch.zeros(B, T, H, W, 4, 4, device=device, dtype=dtype)
    raymats[..., :3, :3] = R_w2l
    raymats[..., :3, 3] = t_w2l
    raymats[..., 3, 3] = 1.0

    # NaN handling
    mask = torch.isnan(d_world).any(-1)
    raymats[mask] = torch.eye(4, device=device, dtype=dtype)

    return raymats


def compute_up_lat_map(
    R: torch.Tensor,
    x_fov: torch.Tensor,
    xi: torch.Tensor,
    height: int,
    width: int,
    device: torch.device = torch.device("cpu"),
    delta: float = 0.1,
):
    """
    计算 up_map 和 lat_map。

    Args:
        R: [B, T, 3, 3] 相机 c2w 旋转矩阵
        x_fov: [B] 或 [B,T] 水平视场角（度）
        xi:   [B] 或 [B,T] UCM 参数
        height: int，图像/patch 高度
        width:  int，图像/patch 宽度
        device: torch.device
        delta: float，小旋转角度（弧度）
    Returns:
        up_map: [B, T, H, W, 2] 单位向量 map
        lat_map: [B, T, H, W, 1] 纬度 map
    """
    B, T, _, _ = R.shape
    dtype = R.dtype
    R = R.float()

    # Step1：生成每像素射线方向（相机坐标系）
    d_cam = ucm_unproject_grid_fov(
        x_fov=x_fov,
        xi=xi,
        height=height,
        width=width,
        device=device,
        dtype=torch.float32,
    )  # [B, H, W, 3]
    if d_cam.ndim == 3:
        d_cam = d_cam.unsqueeze(0)  # [B, H, W, 3]
    mask = d_cam.isnan().any(dim=-1, keepdim=True)  # [B, H, W, 1]

    # Step2：从相机系旋转到世界系
    d_cam_exp = repeat(d_cam, "B H W C -> B T H W C", T=T)  # [B, T, H, W, 3]
    d_world = torch.einsum("btij,bthwj->bthwi", R, d_cam_exp)
    d_world = d_world / torch.clamp_min(d_world.norm(dim=-1, keepdim=True), 1e-8)

    # Step3：计算纬度 map
    Xw, Yw, Zw = d_world[..., 0], d_world[..., 1], d_world[..., 2]
    lat_map = torch.atan2(-Yw, torch.sqrt(Xw**2 + Zw**2)).unsqueeze(
        -1
    )  # [B, T, H, W, 1]

    # Step4：计算 up_map
    v = d_world  # 已归一化
    up_world = torch.tensor(
        [0, -1, 0], device=device, dtype=torch.float32
    )  # 世界上方方向（+Y 向下设定）
    k = torch.cross(
        v, up_world.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(v), dim=-1
    )
    k = k / torch.clamp_min(k.norm(dim=-1, keepdim=True), 1e-8)

    delta = torch.tensor(delta, device=device, dtype=torch.float32)
    cos_eps = torch.cos(delta)
    sin_eps = torch.sin(delta)
    # Rodrigues 公式旋转 v → v_rot
    v_rot = (
        v * cos_eps
        + torch.cross(k, v, dim=-1) * sin_eps
        + k * (k * (v * 1).sum(dim=-1, keepdim=True)) * (1 - cos_eps)
    )

    dirs_cam = torch.einsum("btij,bthwj->bthwi", R.transpose(-1, -2), v_rot)
    Xs, Ys, Zs = dirs_cam[..., 0], dirs_cam[..., 1], dirs_cam[..., 2]

    du, dv = project_ucm_points_fov(
        Xs,
        Ys,
        Zs,
        x_fov=x_fov.float(),
        xi=xi.float(),
        height=height,
        width=width,
    )
    grid = _pixel_grid(
        height=height,
        width=width,
        batch=B,
        dtype=torch.float32,
        device=device,
    )  # [B, H, W, 3]
    grid_x = grid[..., 0].unsqueeze(1)  # [B,1,H,W]
    grid_y = grid[..., 1].unsqueeze(1)

    up_map = torch.stack((du - grid_x, dv - grid_y), dim=-1)  # [B, T, H, W, 2]
    up_map = up_map / torch.clamp_min(up_map.norm(dim=-1, keepdim=True), 1e-8)

    up_map = up_map.to(dtype=dtype)
    lat_map = lat_map.to(dtype=dtype)

    # 扩 mask 到同 shape
    mask_exp2 = mask.unsqueeze(1).expand(B, T, height, width, 1)
    up_map = up_map.masked_fill(mask_exp2, 0.0)
    lat_map = lat_map.masked_fill(mask_exp2, 0.0)

    return up_map, lat_map


class UcpeSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        attn_dim: int,
        num_heads: int,
        patches_x: int = 8,
        patches_y: int = 8,
        image_width: int = 128,
        image_height: int = 128,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
        precompute_coeffs: bool = True,
        emb_dim: int | None = None,
        adaptation_method: str = "parallel",
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.image_width = image_width
        self.image_height = image_height
        self.freq_base = freq_base
        self.freq_scale = freq_scale
        self.adaptation_method = adaptation_method

        self.q_proj = nn.Linear(dim, attn_dim)
        self.k_proj = nn.Linear(dim, attn_dim)
        self.v_proj = nn.Linear(dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, dim)
        if emb_dim is not None:
            self.cam_encoder = nn.Linear(emb_dim, dim)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # 初始化 PRoPE attention 模块（带 precomputed coeffs）
        self.prope_attn = PropeDotProductAttention(
            head_dim=self.head_dim,
            patches_x=patches_x,
            patches_y=patches_y,
            image_width=image_width,
            image_height=image_height,
            freq_base=freq_base,
            freq_scale=freq_scale,
            precompute_coeffs=precompute_coeffs,
        )

    def forward(self, x: torch.Tensor, control_camera_dit_input: dict):
        """
        Args:
            x: (B, T, D) — input tokens
            control_camera_dit_input: dict with keys:
                - viewmats: (B, N, 4, 4)
                - K: (B, N, 3, 3)
        """
        B, T, D = x.shape
        N = control_camera_dit_input["viewmats"].shape[1]  # number of cameras
        H, W = self.patches_y, self.patches_x
        assert (
            T == N * H * W or T == N
        ), f"Expected token shape ({N}×{H}×{W} or {N}), got {T}"

        # Camera geometry intentionally stays fp32, while WorldCrafter hidden states
        # and the UCPE adapter may independently be bf16/fp16 or fp32.  Keep
        # every Linear input in its weight dtype and restore the model hidden
        # dtype at the attention/residual boundaries.
        hidden_dtype = x.dtype
        projection_dtype = self.q_proj.weight.dtype
        projected_x = x.to(dtype=projection_dtype)

        if hasattr(self, "cam_encoder") and "cam_emb" in control_camera_dit_input:
            cam_emb = control_camera_dit_input["cam_emb"].to(
                dtype=self.cam_encoder.weight.dtype
            )
            y = self.cam_encoder(cam_emb)
            if y.shape[1] != T:
                hw = T // cam_emb.shape[1]
                y = repeat(y, "b f d -> b (f hw) d", hw=hw)
            projected_x = projected_x + y.to(dtype=projection_dtype)

        # Project Q, K, V
        q = (
            self.q_proj(projected_x)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # [B, H, T, D_head]
        k = (
            self.k_proj(projected_x)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(projected_x)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Precompute camera-specific functions (only once per batch)
        self.prope_attn._precompute_and_cache_apply_fns(
            viewmats=control_camera_dit_input["viewmats"],
            Ks=control_camera_dit_input.get("K", None),
            coeffs_x=control_camera_dit_input.get("coeffs_x", None),
            coeffs_y=control_camera_dit_input.get("coeffs_y", None),
        )

        # PRoPE matrices and coefficients are deliberately built in the FP32
        # camera-geometry dtype. Mixed-precision Linear/autocast outputs can be
        # BF16/FP16 even when the adapter weights are FP32, so align
        # Q/K/V before the einsum instead of relying on implicit promotion.
        geometry_dtype = control_camera_dit_input["viewmats"].dtype
        q = self.prope_attn._apply_to_q(q.to(dtype=geometry_dtype))
        k = self.prope_attn._apply_to_kv(k.to(dtype=geometry_dtype))
        v = self.prope_attn._apply_to_kv(v.to(dtype=geometry_dtype))

        q = q.to(dtype=hidden_dtype)
        k = k.to(dtype=hidden_dtype)
        v = v.to(dtype=hidden_dtype)

        # Rearrange to [B, T, D] for flash_attention input
        q = rearrange(q, "b h t d -> b t (h d)")
        k = rearrange(k, "b h t d -> b t (h d)")
        v = rearrange(v, "b h t d -> b t (h d)")

        # Fast attention (Flash/Sage/SDPA fallback)
        out = flash_attention(
            q,
            k,
            v,
            num_heads=self.num_heads,
            compatibility_mode=q.dtype not in (torch.float16, torch.bfloat16),
        )

        # reshape back
        out = rearrange(out, "b t (h d) -> b h t d", h=self.num_heads)

        # Apply inverse transform for PRoPE
        out = out.to(dtype=control_camera_dit_input["viewmats"].dtype)
        out = self.prope_attn._apply_to_o(out)

        # Final projection
        out = out.transpose(1, 2).reshape(B, T, -1).to(dtype=self.out_proj.weight.dtype)
        return self.out_proj(out).to(dtype=hidden_dtype)
