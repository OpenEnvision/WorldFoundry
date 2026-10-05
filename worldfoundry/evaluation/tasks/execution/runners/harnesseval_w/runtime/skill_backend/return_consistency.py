"""Local CLIP backend for HarnessEval Return Consistency."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from ..aggregate import clip01, soft_threshold
from ..io import value_digest
from ..skills.skill_return_consistency import BACKEND_OUTPUT_SCHEMA, SAMPLING
from .common import _validate_video


BACKEND_ID = "harnesseval.return_consistency.clip_b32"
BACKEND_VERSION = "harnesseval.return_consistency"

__all__ = [
    "BACKEND_ID",
    "BACKEND_VERSION",
    "backend_config_digest",
    "LocalClipEmbedder",
    "LocalBackend",
]


def backend_config_digest(model_name: str, model_fingerprint: str) -> str:
    return value_digest(
        {
            "version": BACKEND_VERSION,
            "sampling": SAMPLING,
            "clip_model": model_name,
            "clip_model_fingerprint": model_fingerprint,
        }
    )


class ClipEmbedder(Protocol):
    model_name: str
    model_fingerprint: str

    def embed(self, images: list[tuple[str, bytes]]) -> Mapping[str, list[float]]: ...


class LocalClipEmbedder:
    """Lazy in-process OpenAI CLIP adapter."""

    model_name = "ViT-B/32"
    model_fingerprint = (
        "sha256:40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
    )

    def __init__(
        self,
        device: str = "cuda",
        download_root: Path | None = None,
    ) -> None:
        self.device = device
        self.download_root = download_root
        self.load_count = 0
        self._backend: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def ensure_ready(self) -> None:
        if self._backend is not None:
            return
        with self._load_lock:
            if self._backend is None:
                from ..clip_service import OpenAIClipBackend

                self._backend = OpenAIClipBackend(
                    model_name=self.model_name,
                    device=self.device,
                    download_root=self.download_root,
                )
                self.load_count += 1

    def embed(self, images: list[tuple[str, bytes]]) -> Mapping[str, list[float]]:
        self.ensure_ready()
        with self._inference_lock:
            vectors = self._backend.embed([content for _, content in images])
        return {image_id: vector for (image_id, _), vector in zip(images, vectors)}


def _normalized(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("CLIP embedding has a non-positive norm")
    return [value / norm for value in values]


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise RuntimeError("CLIP embeddings have inconsistent dimensions")
    return sum(a * b for a, b in zip(left, right))


def _encode_png(rgb: Any) -> bytes:
    import cv2

    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("OpenCV failed to encode a Return frame")
    return encoded.tobytes()


def _decode_samples(video_path: Path, indices: list[int]) -> list[dict[str, Any]]:
    import cv2

    wanted = set(indices)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"video cannot be decoded: {video_path}")
    output = []
    cursor = 0
    while wanted:
        ok, bgr = cap.read()
        if not ok:
            break
        if cursor in wanted:
            output.append(
                {
                    "frame_index": cursor,
                    "rgb": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                    "gray": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                }
            )
            wanted.remove(cursor)
        cursor += 1
    cap.release()
    if wanted or len(output) != len(indices):
        raise RuntimeError(
            f"video did not yield planned Return frames: {sorted(wanted)}"
        )
    return output


def _frame_components(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    left_embedding: list[float],
    right_embedding: list[float],
) -> dict[str, float]:
    import cv2
    import numpy as np

    width, height = SAMPLING["comparison_size"]
    gray_left = cv2.resize(left["gray"], (width, height), interpolation=cv2.INTER_AREA)
    gray_right = cv2.resize(
        right["gray"], (width, height), interpolation=cv2.INTER_AREA
    )
    rgb_left = cv2.resize(left["rgb"], (width, height), interpolation=cv2.INTER_AREA)
    rgb_right = cv2.resize(right["rgb"], (width, height), interpolation=cv2.INTER_AREA)
    af = gray_left.astype(np.float32)
    bf = gray_right.astype(np.float32)
    centered_left = af - float(np.mean(af))
    centered_right = bf - float(np.mean(bf))
    denominator = float(np.linalg.norm(centered_left) * np.linalg.norm(centered_right))
    ncc = (
        float(np.sum(centered_left * centered_right) / denominator)
        if denominator > 1e-9
        else 0.0
    )
    histogram_left = cv2.calcHist([gray_left], [0], None, [32], [0, 256])
    histogram_right = cv2.calcHist([gray_right], [0], None, [32], [0, 256])
    cv2.normalize(histogram_left, histogram_left)
    cv2.normalize(histogram_right, histogram_right)
    histogram = float(
        cv2.compareHist(histogram_left, histogram_right, cv2.HISTCMP_CORREL)
    )
    edge_left = cv2.Canny(gray_left, 60, 140) > 0
    edge_right = cv2.Canny(gray_right, 60, 140) > 0
    union = np.logical_or(edge_left, edge_right)
    edge_iou = (
        float(np.logical_and(edge_left, edge_right).sum() / union.sum())
        if union.any()
        else 1.0
    )
    rgb_diff = float(
        np.mean(np.abs(rgb_left.astype(np.float32) - rgb_right.astype(np.float32)))
        / 255.0
    )
    return {
        "ncc_score": clip01((ncc + 1.0) / 2.0),
        "histogram_score": clip01((histogram + 1.0) / 2.0),
        "edge_iou": edge_iou,
        "rgb_mean_abs_diff": rgb_diff,
        "rgb_pixel_similarity": clip01(1.0 - soft_threshold(rgb_diff, 0.05, 0.35)),
        "clip_similarity": max(0.0, _cosine(left_embedding, right_embedding)),
    }


class _Backend:
    execution_mode = ""

    def __init__(self, clip_embedder: ClipEmbedder) -> None:
        if clip_embedder.model_name != SAMPLING["clip_model"]:
            raise ValueError(f"Return requires CLIP {SAMPLING['clip_model']}")
        self.clip_embedder = clip_embedder
        self.backend_id = BACKEND_ID
        self.version = BACKEND_VERSION
        self.config_digest = backend_config_digest(
            clip_embedder.model_name, clip_embedder.model_fingerprint
        )

    def ensure_ready(self) -> None:
        ready = getattr(self.clip_embedder, "ensure_ready", None)
        if ready is not None:
            ready()

    def evaluate(
        self,
        video_path: Path,
        video_fingerprint: Mapping[str, Any],
        sample_plan: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        import cv2
        import numpy as np

        _validate_video(video_path, video_fingerprint)
        indices = [int(value) for value in sample_plan["sampled_frame_indices"]]
        samples = _decode_samples(video_path, indices)
        width, height = SAMPLING["comparison_size"]
        images = []
        for position, sample in enumerate(samples):
            resized = cv2.resize(
                sample["rgb"], (width, height), interpolation=cv2.INTER_AREA
            )
            images.append((f"return-f{indices[position]:09d}", _encode_png(resized)))
        by_id = self.clip_embedder.embed(images)
        embeddings = [
            _normalized([float(value) for value in by_id[image_id]])
            for image_id, _ in images
        ]

        evidence_pairs = []
        for pair in sample_plan["pairs"]:
            comparisons = []
            for reference in pair["reference_sample_indices"]:
                for returned in pair["return_sample_indices"]:
                    comparisons.append(
                        {
                            "reference_sample": reference,
                            "return_sample": returned,
                            **_frame_components(
                                samples[reference],
                                samples[returned],
                                embeddings[reference],
                                embeddings[returned],
                            ),
                        }
                    )
            evidence_pairs.append(
                {
                    "reference_chunk": pair["reference_chunk"],
                    "return_chunk": pair["return_chunk"],
                    "similarities": comparisons,
                }
            )
        diffs = [
            float(
                np.mean(
                    np.abs(
                        right["gray"].astype(np.float32)
                        - left["gray"].astype(np.float32)
                    )
                )
                / 255.0
            )
            for left, right in zip(samples[:-1], samples[1:])
        ]
        if not diffs:
            raise RuntimeError("Return requires consecutive sampled frames")
        return {
            "schema_version": BACKEND_OUTPUT_SCHEMA,
            "sampled_frame_indices": indices,
            "pairs": sample_plan["pairs"],
            "fallback_first_last": sample_plan["fallback_first_last"],
            "evidence_pairs": evidence_pairs,
            "mean_diff": sum(diffs) / len(diffs),
            "peak_diff": max(diffs),
            "active_ratio": sum(value > 0.01 for value in diffs) / len(diffs),
            "clip_model": self.clip_embedder.model_name,
            "clip_model_fingerprint": self.clip_embedder.model_fingerprint,
        }


class LocalBackend(_Backend):
    execution_mode = "local"
