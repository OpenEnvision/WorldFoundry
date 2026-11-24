import torch
from scipy.optimize import linear_sum_assignment

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    HAS_TRITON = False

__all__ = ["find_mask_matches", "frustum_frontmost_mask", "frustum_frontmost_mask_triton"]


def _ensure_homogeneous_w2c(w2c: torch.Tensor) -> torch.Tensor:
    """
    Ensure extrinsics are [T,4,4] homogeneous world-to-camera matrices.
    Accepts [T,3,4] or [T,4,4]. Keeps dtype/device.
    """
    assert w2c.ndim == 3 and w2c.shape[0] > 0, "w2c must be [T,3,4] or [T,4,4]"
    T = w2c.shape[0]
    if w2c.shape[1] == 3 and w2c.shape[2] == 4:
        padding = torch.tensor([0, 0, 0, 1], device=w2c.device, dtype=w2c.dtype).view(1, 1, 4)
        padding = padding.expand(T, -1, -1)
        w2c_h = torch.cat([w2c, padding], dim=1)  # [T,4,4]
        return w2c_h
    elif w2c.shape[1] == 4 and w2c.shape[2] == 4:
        return w2c
    else:
        raise AssertionError(f"w2c shape {w2c.shape} must be [T,3,4] or [T,4,4]")


def _broadcast_intrinsics(
    K: torch.Tensor,
    T: int,
) -> torch.Tensor:
    """
    Normalize intrinsics to a [T,4] tensor [fx, fy, cx, cy].
    Accepts [4] or [T,4]. If given normalized intrinsics in [0,1] and image_size provided,
    caller should pass pixel-space [fx,fy,cx,cy]; this function does not rescale.
    """
    assert K.ndim in (1, 2), "intrinsics must be [4] or [T,4]"
    if K.ndim == 1:
        assert K.shape[0] == 4, "intrinsics [4] expected"
        K = K.view(1, 4).expand(T, -1)
    else:
        assert K.shape == (T, 4), f"intrinsics must be [T,4], got {tuple(K.shape)}"
    return K


@torch.no_grad()
def frustum_frontmost_mask(
    points_xyz_rgb: torch.Tensor,
    w2c_matrices: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: tuple[int, int],
    near: float | None = None,
    far: float | None = None,
    background_depth: float = 1e9,
) -> torch.Tensor:
    """
    Batched Z-buffer style visibility: return [T,N] boolean mask where a point is marked True
    iff it is the closest (smallest positive Z) among points that project to the same integer pixel.
    - Requires pixel-space intrinsics and integer rasterization to (H,W).
    """
    assert image_size is not None, "image_size (H,W) required"
    device = w2c_matrices.device
    dtype = w2c_matrices.dtype
    H, W = image_size
    points_world = points_xyz_rgb[..., :3].to(device=device, dtype=dtype)
    w2c = _ensure_homogeneous_w2c(w2c_matrices.to(device=device, dtype=dtype))  # [T,4,4]
    T = w2c.shape[0]
    K = _broadcast_intrinsics(intrinsics.to(device=device, dtype=dtype), T)  # [T,4]

    N = points_world.shape[0]
    points_h = torch.cat([points_world, torch.ones(N, 1, device=device, dtype=dtype)], dim=-1)
    cam_points = (w2c @ points_h.t()).transpose(1, 2)  # [T,N,4]
    X, Y, Z = cam_points[..., 0], cam_points[..., 1], cam_points[..., 2]

    # Valid frustum: in front and within near/far
    valid = Z > 0
    if near is not None:
        valid = valid & (near <= Z)
    if far is not None:
        valid = valid & (far >= Z)

    fx, fy, cx, cy = K[:, 0:1], K[:, 1:2], K[:, 2:3], K[:, 3:4]
    eps = torch.finfo(dtype).eps
    denom = torch.clamp(Z, min=eps)
    u = fx * (X / denom) + cx
    v = fy * (Y / denom) + cy

    # Pixel coords and in-bounds mask
    ui = u.long()
    vi = v.long()
    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    valid = valid & in_bounds

    # Flatten per-view pixels and compute argmin over depth using scatter-reduce
    # linear pixel index: p = vi*W + ui  in [0, H*W)
    pix = (vi * W + ui).clamp(min=0, max=H * W - 1)

    # Initialize per-view depth buffer to large value
    depth_buf = torch.full((T, H * W), background_depth, device=device, dtype=dtype)

    # For invalid points, set depth to background so they don't win
    Z_eff = torch.where(valid, Z, torch.full_like(Z, background_depth))

    # Scatter-reduce: depth_buf[view, pix] = min(depth_buf[view, pix], Z_eff)
    # Use segment-wise min via index_reduce_ if available; fallback to manual loop if needed
    for t in range(T):
        # PyTorch expects 'amin'/'amax' instead of 'min'/'max'
        depth_buf[t].index_reduce_(0, pix[t].flatten(), Z_eff[t].flatten(), reduce="amin")

    # A point is frontmost if its Z equals the per-pixel min Z (within a small tol)
    minZ_at_point = depth_buf.gather(1, pix.view(T, -1))  # [T,N]
    front = valid & ((minZ_at_point + 1e-6) >= Z)
    return front


if HAS_TRITON:

    @triton.jit
    def _zbuffer_min_kernel(
        z_ptr,
        pix_ptr,
        depth_ptr,
        N,
        HxW,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid_t = tl.program_id(0)  # view id
        pid_b = tl.program_id(1)  # block along points
        offs = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        # load pixel indices and z for this block
        pix = tl.load(pix_ptr + pid_t * N + offs, mask=mask, other=0)
        z = tl.load(z_ptr + pid_t * N + offs, mask=mask, other=1e9)
        # clamp pixel to [0, HxW-1] without reductions
        zero = tl.zeros([1], dtype=pix.dtype)
        maxv = tl.full([1], HxW - 1, dtype=pix.dtype)
        pix = tl.where(pix < zero, zero, pix)
        pix = tl.where(pix > maxv, maxv, pix)
        # compute addresses and atomic min on float depth
        addr = depth_ptr + pid_t * HxW + pix
        tl.atomic_min(addr, z, mask=mask)


def frustum_frontmost_mask_triton(
    points_xyz_rgb: torch.Tensor,
    w2c_matrices: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: tuple[int, int],
    near: float | None = None,
    far: float | None = None,
    background_depth: float = 1e9,
) -> torch.Tensor:
    """
    Triton-optimized frontmost visibility. Falls back to torch path if Triton is unavailable.
    """
    if not HAS_TRITON:
        return frustum_frontmost_mask(points_xyz_rgb, w2c_matrices, intrinsics, image_size, near, far, background_depth)

    device = w2c_matrices.device
    dtype = w2c_matrices.dtype
    # Triton path requires CUDA + float32 for bitcast
    if device.type != "cuda" or dtype != torch.float32:
        return frustum_frontmost_mask(points_xyz_rgb, w2c_matrices, intrinsics, image_size, near, far, background_depth)
    H, W = image_size
    points_world = points_xyz_rgb[..., :3].to(device=device, dtype=dtype)
    w2c = _ensure_homogeneous_w2c(w2c_matrices.to(device=device, dtype=dtype))  # [T,4,4]
    T = w2c.shape[0]
    K = _broadcast_intrinsics(intrinsics.to(device=device, dtype=dtype), T)  # [T,4]

    N = points_world.shape[0]
    points_h = torch.cat([points_world, torch.ones(N, 1, device=device, dtype=dtype)], dim=-1)
    cam_points = (w2c @ points_h.t()).transpose(1, 2)  # [T,N,4]
    X, Y, Z = cam_points[..., 0], cam_points[..., 1], cam_points[..., 2]

    valid = Z > 0
    if near is not None:
        valid = valid & (near <= Z)
    if far is not None:
        valid = valid & (far >= Z)

    fx, fy, cx, cy = K[:, 0:1], K[:, 1:2], K[:, 2:3], K[:, 3:4]
    eps = torch.finfo(dtype).eps
    denom = torch.clamp(Z, min=eps)
    u = fx * (X / denom) + cx
    v = fy * (Y / denom) + cy

    ui = u.long()
    vi = v.long()
    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    valid = valid & in_bounds
    pix = (vi * W + ui).clamp(min=0, max=H * W - 1)
    Z_eff = torch.where(valid, Z, torch.full_like(Z, background_depth))

    # Prepare float depth buffer
    depth_buf = torch.full((T, H * W), background_depth, device=device, dtype=dtype)
    # Launch Triton kernel: 2D grid (views, blocks)
    BLOCK_SIZE = 1024
    grid = (T, triton.cdiv(N, BLOCK_SIZE))
    _zbuffer_min_kernel[grid](
        Z_eff.contiguous(),
        pix.contiguous(),
        depth_buf,
        N,
        H * W,
        BLOCK_SIZE,
    )
    # Read min depth for points
    minZ_at_point = depth_buf.gather(1, pix.view(T, -1))
    front = valid & ((minZ_at_point + 1e-6) >= Z)
    return front


def find_mask_matches(
    A: torch.Tensor, B: torch.Tensor, method: str = "independent", bf16: bool = True
) -> dict[int, int]:
    """
    Match each mask in `B` to the best mask in `A`.

    Args:
        A: Candidate masks with shape [M, L].
        B: Target masks with shape [N, L], where N <= M.
        method: Matching method. Use `independent` for per-row best matches
            or `hungarian` for globally optimal one-to-one matching.

    Returns:
        A mapping from each index in `B` to an index in `A`.
    """
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError("A and B must be PyTorch tensors.")

    if A.dim() != 2 or B.dim() != 2:
        raise ValueError("A and B must be 2D tensors.")

    M, L_A = A.shape
    N, L_B = B.shape

    if L_A != L_B:
        raise ValueError(f"A and B must have the same sequence length, got {L_A} and {L_B}.")

    if N > M:
        raise ValueError(f"B cannot contain more masks than A (N={N}, M={M}).")

    if bf16:
        A_float = A.bfloat16()
        B_float = B.bfloat16()
    else:
        A_float = A.float()
        B_float = B.float()

    intersection_matrix = torch.matmul(B_float, A_float.transpose(0, 1))
    intersection_matrix = intersection_matrix.to(torch.float32)

    if method == "independent":
        best_indices = torch.argmax(intersection_matrix, dim=1)
        matches = {b_idx: a_idx.item() for b_idx, a_idx in enumerate(best_indices)}

    elif method == "hungarian":
        cost_matrix = -intersection_matrix
        cost_matrix_np = cost_matrix.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
        matches = dict(zip(row_ind, col_ind, strict=False))
        for i in range(N):
            if i not in matches:
                matches[i] = -1

    else:
        raise ValueError("method must be either 'independent' or 'hungarian'.")

    return matches
