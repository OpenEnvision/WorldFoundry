"""Composite local backends for HarnessEval Drift chunk evidence."""

from __future__ import annotations

import math
import os
import tempfile
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Protocol

from ..io import atomic_write_json, exclusive_lock, file_fingerprint, read_json, value_digest
from ..skills.skill_drift_degradation import BACKEND_OUTPUT_SCHEMA, SAMPLING
from ..skills.skill_motion_quality import (
    _normalize_backend_evidence as normalize_motion,
)
from ..skills.skill_physical_plausibility import (
    _normalize_backend_evidence as normalize_physical,
)
from ..skills.skill_render_quality import (
    _normalize_backend_evidence as normalize_render,
)
from .common import _validate_video


BACKEND_ID = "harnesseval.drift.observation_stack"
BACKEND_VERSION = "harnesseval.drift.observation_stack"
STAGE_CACHE_SCHEMA = "harnesseval.drift_stage_cache"
STAGES = ("physical", "render", "motion", "clip")

__all__ = [
    "BACKEND_ID",
    "BACKEND_VERSION",
    "LocalClipEmbedder",
    "LocalBackend",
    "LocalStagedBackend",
]


class ObservationBackend(Protocol):
    backend_id: str
    version: str
    config_digest: str
    execution_mode: str

    def evaluate(
        self,
        video_path: Path,
        video_fingerprint: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


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


def _background_score(embeddings: list[list[float]]) -> float:
    if len(embeddings) < 2:
        return 1.0
    scores = []
    for index in range(1, len(embeddings)):
        previous = max(0.0, _cosine(embeddings[index], embeddings[index - 1]))
        first = max(0.0, _cosine(embeddings[index], embeddings[0]))
        scores.append((previous + first) / 2.0)
    return sum(scores) / len(scores)


def _encode_png(rgb: Any) -> bytes:
    import cv2

    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("OpenCV failed to encode a Drift frame")
    return encoded.tobytes()


def _sample_chunk(chunk: Mapping[str, Any]) -> list[tuple[int, bytes]]:
    import cv2

    video = Path(str(chunk["video_path"])).resolve()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"video cannot be decoded: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    start = int(chunk["start_frame"])
    end = int(chunk["end_frame"])
    if fps <= 0 or end <= start:
        cap.release()
        raise RuntimeError(f"invalid Drift chunk range [{start}, {end}) for {video}")
    step = max(1, round(fps / float(SAMPLING["clip"]["fps"])))
    wanted = list(range(start, end, step))
    if len(wanted) < 2 and end - start >= 2:
        wanted = list(range(start, end))
    wanted_set = set(wanted)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    samples = []
    for frame_index in range(start, end):
        ok, bgr = cap.read()
        if not ok:
            break
        if frame_index in wanted_set:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            samples.append((frame_index, _encode_png(rgb)))
    cap.release()
    if not samples:
        raise RuntimeError(f"Drift chunk yielded no frames: {video}")
    return samples


@contextmanager
def _video_for_chunk(chunk: Mapping[str, Any]) -> Iterator[Path]:
    import cv2

    video = Path(str(chunk["video_path"])).resolve()
    start = int(chunk["start_frame"])
    end = int(chunk["end_frame"])
    cap = cv2.VideoCapture(str(video))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if start == 0 and end == frame_count:
        cap.release()
        yield video
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"invalid video metadata: {video}")
    with tempfile.TemporaryDirectory(prefix="harnesseval-drift-") as directory:
        output = Path(directory) / "chunk.mkv"
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height)
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("OpenCV cannot create a temporary lossless Drift chunk")
        written = 0
        for _ in range(start, end):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
        writer.release()
        cap.release()
        if written != end - start:
            raise RuntimeError(
                f"decoded {written} of {end - start} requested Drift frames"
            )
        yield output


class _CompositeBackend:
    execution_mode = ""

    def __init__(
        self,
        render_backend: ObservationBackend,
        motion_backend: ObservationBackend,
        physical_backend: ObservationBackend,
        clip_embedder: ClipEmbedder,
    ) -> None:
        self.render_backend = render_backend
        self.motion_backend = motion_backend
        self.physical_backend = physical_backend
        self.clip_embedder = clip_embedder
        self.backend_id = BACKEND_ID
        self.version = BACKEND_VERSION
        self.config_digest = value_digest(
            {
                "version": BACKEND_VERSION,
                "sampling": SAMPLING,
                "render": self._identity(render_backend),
                "motion": self._identity(motion_backend),
                "physical": self._identity(physical_backend),
                "clip_model": clip_embedder.model_name,
                "clip_model_fingerprint": clip_embedder.model_fingerprint,
            }
        )

    def _identity(self, backend: ObservationBackend) -> dict[str, str]:
        if backend.execution_mode != self.execution_mode:
            raise ValueError(
                f"{backend.backend_id} must use {self.execution_mode} execution for this backend"
            )
        return {
            "backend_id": backend.backend_id,
            "version": backend.version,
            "config_digest": backend.config_digest,
        }

    def ensure_ready(self) -> None:
        for backend in (
            self.render_backend,
            self.motion_backend,
            self.physical_backend,
        ):
            ready = getattr(backend, "ensure_ready", None)
            if ready is not None:
                ready()
        ready = getattr(self.clip_embedder, "ensure_ready", None)
        if ready is not None:
            ready()

    def _clip_evidence(
        self, chunk: Mapping[str, Any]
    ) -> tuple[float, list[float], int]:
        samples = _sample_chunk(chunk)
        prefix = value_digest(
            {
                "chunk_id": chunk["chunk_id"],
                "video": chunk["video_fingerprint"],
                "start": chunk["start_frame"],
                "end": chunk["end_frame"],
            }
        )[:16]
        images = [(f"{prefix}-f{index:09d}", content) for index, content in samples]
        by_id = self.clip_embedder.embed(images)
        embeddings = [
            _normalized([float(value) for value in by_id[image_id]])
            for image_id, _ in images
        ]
        mean_embedding = _normalized(
            [
                sum(vector[index] for vector in embeddings) / len(embeddings)
                for index in range(len(embeddings[0]))
            ]
        )
        return _background_score(embeddings), mean_embedding, len(samples)

    def evaluate(
        self,
        video_path: Path,
        video_fingerprint: Mapping[str, Any],
        chunks: list[dict[str, Any]],
    ) -> Mapping[str, Any]:
        _validate_video(video_path, video_fingerprint)
        output = []
        details = {}
        for chunk in chunks:
            chunk_video = Path(str(chunk["video_path"])).resolve()
            _validate_video(chunk_video, chunk["video_fingerprint"])
            with _video_for_chunk(chunk) as inference_video:
                fingerprint = file_fingerprint(inference_video)
                render_raw = self.render_backend.evaluate(inference_video, fingerprint)
                motion_raw = self.motion_backend.evaluate(inference_video, fingerprint)
                physical_raw = self.physical_backend.evaluate(
                    inference_video, fingerprint
                )
            render = normalize_render(render_raw)
            motion = normalize_motion(motion_raw)
            physical = normalize_physical(physical_raw)
            background, embedding, sampled_frames = self._clip_evidence(chunk)
            metrics = {
                **render["metrics"],
                **motion["metrics"],
                "background": background,
                **physical["metrics"],
            }
            output.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "video_path": chunk["video_path"],
                    "video_fingerprint": chunk["video_fingerprint"],
                    "start_frame": chunk["start_frame"],
                    "end_frame": chunk["end_frame"],
                    "boundary_source": chunk["boundary_source"],
                    "metrics": metrics,
                    "clip_embedding": embedding,
                }
            )
            details[str(chunk["chunk_id"])] = {
                "render": render["details"],
                "motion": motion["details"],
                "physical": physical["details"],
                "clip_sampled_frames": sampled_frames,
            }
        return {
            "schema_version": BACKEND_OUTPUT_SCHEMA,
            "chunks": output,
            "details": {
                "chunks": details,
                "clip_model": self.clip_embedder.model_name,
                "clip_model_fingerprint": self.clip_embedder.model_fingerprint,
            },
        }


class LocalBackend(_CompositeBackend):
    execution_mode = "local"


class StagedBackend(_CompositeBackend):
    """Run child backends in separate GPU phases and cache each result."""

    execution_mode = "local"
    resource_mode = "staged_local"

    def __init__(
        self,
        render_backend: ObservationBackend,
        motion_backend: ObservationBackend,
        physical_backend: ObservationBackend,
        clip_embedder: ClipEmbedder,
        cache_root: Path,
    ) -> None:
        super().__init__(
            render_backend,
            motion_backend,
            physical_backend,
            clip_embedder,
        )
        self.stage_cache_root = cache_root.resolve() / "artifacts" / "drift_stages"
        self.stage_cache_overlay_root = (
            cache_root.resolve() / "artifacts" / "drift_stages_local_overlay"
        )

    def ensure_ready(self) -> None:
        return

    def _stage_identity(self, stage: str) -> dict[str, str]:
        if stage == "clip":
            return {
                "model": self.clip_embedder.model_name,
                "model_fingerprint": self.clip_embedder.model_fingerprint,
            }
        backend = {
            "render": self.render_backend,
            "motion": self.motion_backend,
            "physical": self.physical_backend,
        }[stage]
        return self._identity(backend)

    def _stage_digest(self, stage: str, chunk: Mapping[str, Any]) -> str:
        return value_digest(
            {
                "schema_version": STAGE_CACHE_SCHEMA,
                "stage": stage,
                "backend": self._stage_identity(stage),
                "sampling": SAMPLING["clip"] if stage == "clip" else None,
                "chunk": {
                    "chunk_id": chunk["chunk_id"],
                    "video_fingerprint": chunk["video_fingerprint"],
                    "start_frame": chunk["start_frame"],
                    "end_frame": chunk["end_frame"],
                    "boundary_source": chunk["boundary_source"],
                },
            }
        )

    def _stage_path(self, stage: str, digest: str) -> Path:
        return self.stage_cache_root / stage / digest[:2] / f"{digest}.json"

    def _overlay_stage_path(self, stage: str, digest: str) -> Path:
        return (
            self.stage_cache_overlay_root
            / stage
            / digest[:2]
            / f"{digest}.json"
        )

    @staticmethod
    def _can_create(path: Path) -> bool:
        parent = path.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        return parent.is_dir() and os.access(parent, os.W_OK)

    def _write_stage_path(self, stage: str, digest: str) -> Path:
        official = self._stage_path(stage, digest)
        lock = official.parent / ".locks" / f"{official.name}.lock"
        lock_usable = (
            lock.is_file() and os.access(lock, os.R_OK)
        ) or self._can_create(lock)
        if self._can_create(official) and lock_usable:
            return official
        return self._overlay_stage_path(stage, digest)

    def _read_stage(
        self, stage: str, chunk: Mapping[str, Any]
    ) -> tuple[Path, Mapping[str, Any] | None]:
        digest = self._stage_digest(stage, chunk)
        for path in (
            self._stage_path(stage, digest),
            self._overlay_stage_path(stage, digest),
        ):
            if not path.is_file():
                continue
            try:
                payload = read_json(path)
            except (OSError, ValueError):
                continue
            if not isinstance(payload, Mapping) or (
                payload.get("schema_version") != STAGE_CACHE_SCHEMA
                or payload.get("stage") != stage
                or payload.get("input_digest") != digest
                or not isinstance(payload.get("evidence"), Mapping)
            ):
                continue
            return path, payload["evidence"]
        return self._write_stage_path(stage, digest), None

    def _compute_stage(
        self, stage: str, chunk: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if stage == "clip":
            background, embedding, sampled_frames = self._clip_evidence(chunk)
            return {
                "background": background,
                "clip_embedding": embedding,
                "sampled_frames": sampled_frames,
            }
        backend = {
            "render": self.render_backend,
            "motion": self.motion_backend,
            "physical": self.physical_backend,
        }[stage]
        with _video_for_chunk(chunk) as inference_video:
            return backend.evaluate(inference_video, file_fingerprint(inference_video))

    def prepare_stage(
        self,
        video_path: Path,
        video_fingerprint: Mapping[str, Any],
        chunks: list[dict[str, Any]],
        stage: str,
    ) -> dict[str, Any]:
        if stage not in STAGES:
            raise ValueError(f"unknown Drift stage: {stage}")
        _validate_video(video_path, video_fingerprint)
        hits = misses = 0
        paths = []
        for chunk in chunks:
            path, evidence = self._read_stage(stage, chunk)
            paths.append(str(path))
            if evidence is not None:
                hits += 1
                continue
            lock = path.parent / ".locks" / f"{path.name}.lock"
            with exclusive_lock(lock, {"stage": stage}, wait=True):
                _, evidence = self._read_stage(stage, chunk)
                if evidence is not None:
                    hits += 1
                    continue
                evidence = self._compute_stage(stage, chunk)
                atomic_write_json(
                    path,
                    {
                        "schema_version": STAGE_CACHE_SCHEMA,
                        "stage": stage,
                        "input_digest": self._stage_digest(stage, chunk),
                        "backend": self._stage_identity(stage),
                        "evidence": dict(evidence),
                    },
                )
                misses += 1
        return {"cache_hits": hits, "cache_misses": misses, "cache_paths": paths}

    def evaluate(
        self,
        video_path: Path,
        video_fingerprint: Mapping[str, Any],
        chunks: list[dict[str, Any]],
    ) -> Mapping[str, Any]:
        _validate_video(video_path, video_fingerprint)
        output = []
        details = {}
        for chunk in chunks:
            evidence = {}
            for stage in STAGES:
                path, value = self._read_stage(stage, chunk)
                if value is None:
                    raise RuntimeError(f"missing Drift {stage} stage cache: {path}")
                evidence[stage] = value
            render = normalize_render(evidence["render"])
            motion = normalize_motion(evidence["motion"])
            physical = normalize_physical(evidence["physical"])
            clip = evidence["clip"]
            output.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "video_path": chunk["video_path"],
                    "video_fingerprint": chunk["video_fingerprint"],
                    "start_frame": chunk["start_frame"],
                    "end_frame": chunk["end_frame"],
                    "boundary_source": chunk["boundary_source"],
                    "metrics": {
                        **render["metrics"],
                        **motion["metrics"],
                        "background": clip["background"],
                        **physical["metrics"],
                    },
                    "clip_embedding": clip["clip_embedding"],
                }
            )
            details[str(chunk["chunk_id"])] = {
                "render": render["details"],
                "motion": motion["details"],
                "physical": physical["details"],
                "clip_sampled_frames": clip["sampled_frames"],
            }
        return {
            "schema_version": BACKEND_OUTPUT_SCHEMA,
            "chunks": output,
            "details": {
                "chunks": details,
                "clip_model": self.clip_embedder.model_name,
                "clip_model_fingerprint": self.clip_embedder.model_fingerprint,
                "resource_mode": self.resource_mode,
            },
        }


class LocalStagedBackend(StagedBackend):
    """Cache Drift components in phases while keeping one local model loaded."""

    execution_mode = "local"
    resource_mode = "staged_local"
