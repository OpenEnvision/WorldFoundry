import torch


def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100) -> torch.Tensor:
    """
    Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)

    Args:
        pos_grid: Tensor of shape (H, W, 2) containing 2D coordinates
        embed_dim: Output channel dimension for embeddings
        omega_0: Base frequency for sinusoidal encoding

    Returns:
        Tensor of shape (H, W, embed_dim) with positional embeddings
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    assert embed_dim % 2 == 0

    device = pos_grid.device
    pos_flat = pos_grid.reshape(-1, grid_dim)  # Flatten to (H*W, 2)

    # Generate frequency bands
    omega = torch.arange(embed_dim // 4, dtype=torch.float32 if device.type == "mps" else torch.double, device=device)
    omega /= embed_dim / 4.0
    omega = 1.0 / omega_0**omega  # (D/4,)

    # Process x and y coordinates separately
    pos_x = pos_flat[:, 0].reshape(-1)  # (H*W,)
    pos_y = pos_flat[:, 1].reshape(-1)  # (H*W,)

    # Compute outer products
    out_x = torch.einsum("m,d->md", pos_x, omega)  # (H*W, D/4)
    out_y = torch.einsum("m,d->md", pos_y, omega)  # (H*W, D/4)

    # Apply sin and cos
    emb_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)  # (H*W, D/2)
    emb_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)  # (H*W, D/2)

    # Combine x and y embeddings
    emb = torch.cat([emb_x, emb_y], dim=-1)  # (H*W, D)

    return emb.float().view(H, W, embed_dim)  # [H, W, D]


# Inspired by https://github.com/microsoft/moge
