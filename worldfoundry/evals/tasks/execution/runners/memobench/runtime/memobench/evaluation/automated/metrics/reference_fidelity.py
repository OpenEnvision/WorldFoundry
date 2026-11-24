import cv2
import lpips as lpips_lib
import numpy as np
import torch

_lpips_model = None


def _get_lpips(device: str | None = None):
    global _lpips_model
    if _lpips_model is None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _lpips_model = lpips_lib.LPIPS(net="alex").to(device)
        _lpips_model.eval()
    return _lpips_model


def _to_lpips_tensor(bgr: np.ndarray, device: str) -> torch.Tensor:
    """Convert BGR uint8 HxWxC to LPIPS-format tensor [-1,1] 1xCxHxW."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse < 1e-10:
        return 99.0
    return float(10.0 * np.log10(255.0**2 / mse))


def _ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """Single-channel SSIM, no external deps."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_a2, mu_b2, mu_ab = mu_a**2, mu_b**2, mu_a * mu_b
    sig_a2 = cv2.GaussianBlur(a**2, (11, 11), 1.5) - mu_a2
    sig_b2 = cv2.GaussianBlur(b**2, (11, 11), 1.5) - mu_b2
    sig_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab
    num = (2 * mu_ab + C1) * (2 * sig_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2)
    return float(np.mean(num / (den + 1e-8)))


def reference_frame_fidelity(gt_image: np.ndarray, gen_frame0: np.ndarray) -> tuple[float, float]:
    """
    Compare generated frame 0 against the GT reference image.
    GT image is resized to match the generated frame resolution.

    Returns: (psnr, ssim)
    """
    h, w = gen_frame0.shape[:2]
    gt_resized = cv2.resize(gt_image, (w, h), interpolation=cv2.INTER_AREA)

    psnr = _psnr(gt_resized, gen_frame0)

    gt_g = cv2.cvtColor(gt_resized, cv2.COLOR_BGR2GRAY)
    gen_g = cv2.cvtColor(gen_frame0, cv2.COLOR_BGR2GRAY)
    ssim = _ssim_gray(gt_g, gen_g)

    return psnr, ssim


def gt_phase_pixel_fidelity(
    gt_frames: list[np.ndarray],
    gen_frames: list[np.ndarray],
    device: str | None = None,
    lpips_batch_size: int = 16,
) -> tuple[float, float, float]:
    """
    Compute average PSNR / SSIM / LPIPS between paired GT and generated frames.
    GT frames are resized to match each generated frame's resolution.

    Args:
        gt_frames:        list of GT BGR frames (any resolution)
        gen_frames:       list of generated BGR frames
        device:           torch device string
        lpips_batch_size: number of frame pairs per LPIPS forward pass

    Returns: (mean_psnr, mean_ssim, mean_lpips)
             lpips is distance (lower = better); psnr/ssim higher = better.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = _get_lpips(device)

    psnrs, ssims = [], []
    gt_tensors, gen_tensors = [], []

    for gt_f, gen_f in zip(gt_frames, gen_frames, strict=False):
        h, w = gen_f.shape[:2]
        gt_r = cv2.resize(gt_f, (w, h), interpolation=cv2.INTER_AREA)

        psnrs.append(_psnr(gt_r, gen_f))

        gt_g = cv2.cvtColor(gt_r, cv2.COLOR_BGR2GRAY)
        gen_g = cv2.cvtColor(gen_f, cv2.COLOR_BGR2GRAY)
        ssims.append(_ssim_gray(gt_g, gen_g))

        gt_tensors.append(_to_lpips_tensor(gt_r, device))
        gen_tensors.append(_to_lpips_tensor(gen_f, device))

    # Batched LPIPS
    lpipss = []
    for i in range(0, len(gt_tensors), lpips_batch_size):
        gt_batch = torch.cat(gt_tensors[i : i + lpips_batch_size], dim=0)
        gen_batch = torch.cat(gen_tensors[i : i + lpips_batch_size], dim=0)
        with torch.no_grad():
            d = loss_fn(gt_batch, gen_batch)
        lpipss.extend(d.squeeze(-1).squeeze(-1).squeeze(-1).cpu().numpy().tolist())

    return (
        float(np.mean(psnrs)),
        float(np.mean(ssims)),
        float(np.mean(lpipss)),
    )
