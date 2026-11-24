import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from PIL import Image

_raft = None
_raft_device = None


def _load_raft(device: str | None = None):
    global _raft, _raft_device
    if _raft is not None:
        return _raft, _raft_device

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _raft_device = device

    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    _raft = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
    return _raft, _raft_device


def _to_tensor(img_bgr: np.ndarray, device: str) -> torch.Tensor:
    """BGR numpy → [1, 3, H, W] float in [0, 1]."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return TVF.to_tensor(Image.fromarray(rgb)).unsqueeze(0).to(device)


@torch.no_grad()
def _warp_error(model, device: str, img1: np.ndarray, img2: np.ndarray) -> float:
    """
    RAFT flow img1→img2, warp img1, return mean L1 photometric error.
    Frames are padded to the next multiple of 8 as required by RAFT,
    then the flow is cropped back to the original dimensions.
    """
    t1 = _to_tensor(img1, device)
    t2 = _to_tensor(img2, device)

    H, W = t1.shape[-2:]

    # RAFT requires H and W divisible by 8 — pad with edge replication
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h > 0 or pad_w > 0:
        t1_in = F.pad(t1, [0, pad_w, 0, pad_h], mode="replicate")
        t2_in = F.pad(t2, [0, pad_w, 0, pad_h], mode="replicate")
    else:
        t1_in, t2_in = t1, t2

    # RAFT returns a list of refinements; the last is the finest estimate
    flow = model(t1_in, t2_in)[-1]  # [1, 2, H', W']

    # Crop flow back to original spatial size before computing warp
    flow = flow[:, :, :H, :W]  # [1, 2, H, W]

    gy, gx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    wx = (gx + flow[0, 0]) / (W - 1) * 2.0 - 1.0  # normalised to [-1, 1]
    wy = (gy + flow[0, 1]) / (H - 1) * 2.0 - 1.0
    grid = torch.stack([wx, wy], dim=-1).unsqueeze(0)  # [1, H, W, 2]

    warped = F.grid_sample(t1, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return float((warped - t2).abs().mean().item())


def temporal_flow_score(frames, start: int, end: int, sample_step: int = 4, device: str | None = None) -> float:
    """
    Motion smoothness via RAFT optical-flow warp error (VBench formulation).

    Estimates dense optical flow between consecutive sampled frames using
    RAFT (Teed & Deng, ECCV 2020), warps frame_i toward frame_{i+1}, and
    measures the mean L1 photometric error.  Used as the motion-smoothness
    metric in VBench (Huang et al., CVPR 2024).

    Called twice per clip — once for the V phase [0, h_start] and once for
    the R phase [r_start, N-1]; the D phase is excluded intentionally.

    Score = exp(−mean_err / 0.15)  ∈  [0, 1].  Higher = smoother.
    """
    model, device = _load_raft(device)
    idxs = list(frames.iter_indices(start, end, sample_step))
    if len(idxs) < 3:
        return 0.0

    errs = [_warp_error(model, device, frames.get(idxs[i]), frames.get(idxs[i + 1])) for i in range(len(idxs) - 1)]
    mean_err = float(np.mean(errs)) if errs else 1.0
    return float(np.exp(-mean_err / 0.15))
