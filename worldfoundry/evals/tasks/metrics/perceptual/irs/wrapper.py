"""Image Realism Score (IRS) from arXiv:2309.14756."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

MeasureName = str

# Pentagon order from Chen et al. (correlation-guided adjacency): CED, GLCMC, GLCME, VBM, MS.
MEASURE_ORDER: tuple[MeasureName, ...] = ("ced", "glcmc", "glcme", "vbm", "ms")
INVERSE_MEASURES: frozenset[MeasureName] = frozenset({"glcme", "vbm", "ms"})
PENTAGON_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (0, 4), (1, 2), (2, 3), (3, 4))

MIN_RAW_FLOOR: dict[MeasureName, float] = {
    "ced": 1e-6,
    "glcmc": 1.0,
    "glcme": 1e-6,
    "vbm": 1.0,
    "ms": 1.0,
}


def fit_irs_reference_means(images: list[Any]) -> dict[MeasureName, float]:
    """Estimate ImageNet-style raw-measure means from a reference real-image set."""
    if not images:
        raise ValueError("images must be non-empty")
    totals = dict.fromkeys(MEASURE_ORDER, 0.0)
    for image in images:
        measures = compute_irs_measures(image)
        for name in MEASURE_ORDER:
            totals[name] += measures[name]
    count = float(len(images))
    return {name: totals[name] / count for name in MEASURE_ORDER}


def _normalize_raw_measures(
    measures: dict[MeasureName, float],
    reference_raw_means: dict[MeasureName, float],
) -> dict[MeasureName, float]:
    normalized: dict[MeasureName, float] = {}
    for name in MEASURE_ORDER:
        denom = max(reference_raw_means.get(name, 1.0), MIN_RAW_FLOOR[name])
        normalized[name] = float(measures[name]) / denom
    return normalized


def _to_gray_uint8(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("L"), dtype=np.uint8)
    else:
        arr = np.asarray(image)
        if arr.ndim == 3:
            if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[-1] == 3:
                arr = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            elif arr.shape[-1] == 1:
                arr = arr[..., 0].astype(np.uint8)
            else:
                arr = arr.mean(axis=-1).astype(np.uint8)
        arr = arr.astype(np.uint8)
    return arr


def compute_canny_edge_density(image: Any) -> float:
    """Canny Edge Density (CED, Eq. 3)."""
    from scipy.ndimage import convolve, gaussian_filter

    gray = _to_gray_uint8(image).astype(np.float64)
    try:
        import cv2

        edges = cv2.Canny(gray.astype(np.uint8), threshold1=100, threshold2=200)
        return float((edges > 0).sum() / edges.size)
    except ImportError:
        try:
            from skimage.feature import canny

            edges = canny(gray, sigma=1.0)
            return float(edges.sum() / edges.size)
        except ImportError:
            smoothed = gaussian_filter(gray, sigma=1.0)
            gx = convolve(smoothed, np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]), mode="reflect")
            gy = convolve(smoothed, np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]), mode="reflect")
            magnitude = np.hypot(gx, gy)
            threshold = np.percentile(magnitude, 90.0)
            edges = magnitude >= threshold
            return float(edges.sum() / edges.size)


def _graycomatrix_numpy(gray: np.ndarray, *, levels: int = 256) -> np.ndarray:
    """Build a horizontal GLCM with unit distance."""
    if gray.max() >= levels:
        scaled = (gray.astype(np.float64) / max(int(gray.max()), 1) * (levels - 1)).astype(np.int32)
    else:
        scaled = gray.astype(np.int32)
    left = scaled[:, :-1].reshape(-1)
    right = scaled[:, 1:].reshape(-1)
    matrix = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(matrix, (left, right), 1.0)
    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


def compute_glcm_energy(image: Any) -> float:
    """GLCM Energy / Angular Second Moment (Eq. 1)."""
    gray = _to_gray_uint8(image)
    try:
        from skimage.feature import graycomatrix, graycoprops

        glcm = graycomatrix(
            gray,
            distances=[1],
            angles=[0.0],
            levels=256,
            symmetric=True,
            normed=True,
        )
        return float(graycoprops(glcm, "ASM")[0, 0])
    except ImportError:
        matrix = _graycomatrix_numpy(gray)
        return float(np.sum(matrix**2))


def compute_glcm_contrast(image: Any) -> float:
    """GLCM Contrast (Eq. 2)."""
    gray = _to_gray_uint8(image)
    try:
        from skimage.feature import graycomatrix, graycoprops

        glcm = graycomatrix(
            gray,
            distances=[1],
            angles=[0.0],
            levels=256,
            symmetric=True,
            normed=True,
        )
        return float(graycoprops(glcm, "contrast")[0, 0])
    except ImportError:
        matrix = _graycomatrix_numpy(gray)
        levels = matrix.shape[0]
        indices = np.arange(levels, dtype=np.float64)
        diff = (indices[:, None] - indices[None, :]) ** 2
        return float(np.sum(matrix * diff))


def compute_variance_blur_measure(image: Any) -> float:
    """Variance Blur Measure (VBM, Eq. 4)."""
    from scipy.ndimage import convolve

    gray = _to_gray_uint8(image).astype(np.float64)
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    lap = convolve(gray, kernel, mode="reflect")
    return float(np.var(lap))


def compute_mean_spectrum(image: Any) -> float:
    """Mean Spectrum magnitude (Eq. 5)."""
    gray = _to_gray_uint8(image).astype(np.float64)
    spectrum = np.fft.fft2(gray)
    return float(np.mean(np.abs(spectrum)))


def compute_irs_measures(image: Any) -> dict[MeasureName, float]:
    """Return the five primitive IRS statistics for one image."""
    return {
        "ced": compute_canny_edge_density(image),
        "glcmc": compute_glcm_contrast(image),
        "glcme": compute_glcm_energy(image),
        "vbm": compute_variance_blur_measure(image),
        "ms": compute_mean_spectrum(image),
    }


def _calibrate_measures(
    measures: dict[MeasureName, float],
    *,
    eps: float = 1e-8,
) -> dict[MeasureName, float]:
    calibrated: dict[MeasureName, float] = {}
    for name in MEASURE_ORDER:
        value = float(measures[name])
        if name in INVERSE_MEASURES:
            value = 1.0 / max(value, eps)
        calibrated[name] = value
    return calibrated


def _pentagon_irs(radii: np.ndarray) -> float:
    """Pentagon area from five radii (Eq. 8 in arXiv:2309.14756)."""
    theta = 2.0 * np.pi / 5.0
    total = 0.0
    for left, right in PENTAGON_PAIRS:
        total += 0.5 * float(radii[left]) * float(radii[right]) * np.sin(theta)
    return float(total)


def compute_irs(
    image: Any,
    *,
    reference_raw_means: dict[MeasureName, float] | None = None,
    eps: float = 1e-8,
) -> float:
    """Compute Image Realism Score (Eq. 8–9) for a single image."""
    if reference_raw_means is None:
        raise ValueError("reference_raw_means is required; fit them with fit_irs_reference_means(real_images)")
    raw = compute_irs_measures(image)
    normalized = _normalize_raw_measures(raw, reference_raw_means)
    calibrated = _calibrate_measures(normalized, eps=eps)
    radii = np.array([calibrated[name] for name in MEASURE_ORDER], dtype=np.float64)
    return _pentagon_irs(radii)


def compute_irs_with_reference(
    images: list[Any],
    reference_images: list[Any],
    *,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Fit raw-measure means on reference real images, then score each target image."""
    if not reference_images:
        raise ValueError("reference_images must be non-empty")
    reference_means = fit_irs_reference_means(reference_images)
    scores = [compute_irs(image, reference_raw_means=reference_means, eps=eps) for image in images]
    return {
        "reference_raw_means": reference_means,
        "irs_mean": float(np.mean(scores)) if scores else float("nan"),
        "irs_scores": scores,
    }


__all__ = [
    "MEASURE_ORDER",
    "MIN_RAW_FLOOR",
    "compute_canny_edge_density",
    "compute_glcm_contrast",
    "compute_glcm_energy",
    "compute_irs",
    "compute_irs_measures",
    "compute_irs_with_reference",
    "compute_mean_spectrum",
    "compute_variance_blur_measure",
    "fit_irs_reference_means",
]
