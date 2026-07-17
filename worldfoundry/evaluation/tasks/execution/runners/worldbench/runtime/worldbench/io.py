"""Dataset, generated-artifact, and answer-manifest I/O for WorldBench."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
GENERIC_VIDEO_STEMS = {"generated", "generation", "output", "prediction", "result", "video"}
PREDICTION_KEYS = ("prediction", "pred", "response", "generated_answer", "model_answer", "answer")


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(item) if item.isdigit() else item.casefold() for item in re.split(r"(\d+)", value))


def canonical_sample_id(value: str | Path) -> str:
    text = str(value).replace("\\", "/").strip().strip("/")
    if text.startswith("scenes/"):
        text = text[len("scenes/") :]
    suffix = Path(text).suffix.casefold()
    if suffix in VIDEO_SUFFIXES:
        text = text[: -len(suffix)]
    return re.sub(r"/+", "/", text)


@dataclass(frozen=True)
class SceneSample:
    sample_id: str
    root: Path
    rgba_paths: tuple[Path, ...]
    segmentation_paths: tuple[Path, ...]
    input_video: Path | None


@dataclass(frozen=True)
class TextQuestion:
    question_id: str
    source: Path
    row_index: int
    video_name: str
    question: str
    answer: str


def _numbered_files(folder: Path, prefix: str, suffixes: set[str]) -> tuple[Path, ...]:
    paths = [
        item
        for item in folder.iterdir()
        if item.is_file() and item.name.startswith(prefix) and item.suffix.casefold() in suffixes
    ]
    return tuple(sorted(paths, key=lambda item: natural_key(item.name)))


def resolve_scenes_root(dataset_root: Path) -> Path:
    root = dataset_root.expanduser().resolve()
    scenes = root / "scenes"
    if scenes.is_dir():
        return scenes
    if root.name == "scenes" and root.is_dir():
        return root
    raise FileNotFoundError(f"WorldBench scenes directory not found under: {dataset_root}")


def discover_scenes(dataset_root: Path) -> list[SceneSample]:
    scenes_root = resolve_scenes_root(dataset_root)
    samples: list[SceneSample] = []
    for first_segmentation in sorted(scenes_root.rglob("segmentation_00000.png")):
        root = first_segmentation.parent
        segmentations = _numbered_files(root, "segmentation_", {".png"})
        rgba = _numbered_files(root, "rgba_", IMAGE_SUFFIXES)
        if not segmentations or not rgba:
            continue
        samples.append(
            SceneSample(
                sample_id=canonical_sample_id(root.relative_to(scenes_root)),
                root=root,
                rgba_paths=rgba,
                segmentation_paths=segmentations,
                input_video=(root / "input_video.mp4") if (root / "input_video.mp4").is_file() else None,
            )
        )
    return sorted(samples, key=lambda sample: natural_key(sample.sample_id))


def discover_questions(dataset_root: Path) -> list[TextQuestion]:
    root = dataset_root.expanduser().resolve()
    questions_root = root / "textual_questions"
    if not questions_root.is_dir() and root.name == "scenes":
        questions_root = root.parent / "textual_questions"
    if not questions_root.is_dir():
        return []

    questions: list[TextQuestion] = []
    for source in sorted(questions_root.glob("*.json")):
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"WorldBench question file must contain a list: {source}")
        for index, row in enumerate(payload):
            if not isinstance(row, dict):
                raise ValueError(f"WorldBench question row {source}#{index} is not an object")
            video_name = str(row.get("video_name") or "").strip()
            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if not video_name or not question or not answer:
                raise ValueError(f"WorldBench question row {source}#{index} is missing required fields")
            questions.append(
                TextQuestion(
                    question_id=f"{source.stem}:{index:04d}",
                    source=source,
                    row_index=index,
                    video_name=video_name,
                    question=question,
                    answer=answer,
                )
            )
    return questions


def _load_records(path: Path) -> Any:
    suffix = path.suffix.casefold()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix in {".csv", ".tsv"}:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t" if suffix == ".tsv" else ","))
    raise ValueError(f"unsupported manifest suffix: {path}")


def load_path_manifest(path: Path | None, *, base_dir: Path | None = None) -> dict[str, Path]:
    if path is None:
        return {}
    manifest = path.expanduser().resolve()
    payload = _load_records(manifest)
    mapping: dict[str, Path] = {}
    if isinstance(payload, dict):
        if isinstance(payload.get("samples"), list):
            payload = payload["samples"]
        else:
            for raw_id, raw_path in payload.items():
                if isinstance(raw_path, (str, Path)):
                    resolved = Path(raw_path).expanduser()
                    if not resolved.is_absolute():
                        resolved = (base_dir or manifest.parent) / resolved
                    mapping[canonical_sample_id(raw_id)] = resolved.resolve()
            return mapping
    if not isinstance(payload, list):
        raise ValueError(f"video manifest must be an object or list: {manifest}")
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"video manifest row {index} is not an object")
        raw_id = next((row[key] for key in ("sample_id", "scene_id", "id") if row.get(key)), None)
        raw_path = next(
            (row[key] for key in ("video_path", "generated_video", "path", "artifact_path") if row.get(key)),
            None,
        )
        if raw_id is None or raw_path is None:
            raise ValueError(f"video manifest row {index} needs sample_id and video_path")
        resolved = Path(str(raw_path)).expanduser()
        if not resolved.is_absolute():
            resolved = (base_dir or manifest.parent) / resolved
        mapping[canonical_sample_id(str(raw_id))] = resolved.resolve()
    return mapping


class ArtifactIndex:
    """Resolve dataset sample IDs to generated videos or frame directories."""

    def __init__(self, root: Path | None, manifest: dict[str, Path] | None = None) -> None:
        self.root = root.expanduser().resolve() if root is not None else None
        self._paths: dict[str, set[Path]] = {}
        for sample_id, path in (manifest or {}).items():
            self._add(sample_id, path)
        if self.root is not None:
            if not self.root.is_dir():
                raise FileNotFoundError(f"generated artifact directory not found: {self.root}")
            self._scan()

    def _add(self, key: str | Path, path: Path) -> None:
        canonical = canonical_sample_id(key)
        aliases = {
            canonical,
            canonical.replace("/", "__"),
            canonical.replace("/", "_"),
        }
        parts = canonical.split("/")
        if len(parts) >= 2:
            aliases.add("/".join(parts[-2:]))
        for alias in aliases:
            self._paths.setdefault(alias, set()).add(path.resolve())

    def _scan(self) -> None:
        assert self.root is not None
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.suffix.casefold() in VIDEO_SUFFIXES:
                relative = path.relative_to(self.root)
                self._add(relative.with_suffix(""), path)
                if path.stem.casefold() in GENERIC_VIDEO_STEMS and relative.parent != Path("."):
                    self._add(relative.parent, path)
            elif path.is_dir():
                images = [
                    item for item in path.iterdir() if item.is_file() and item.suffix.casefold() in IMAGE_SUFFIXES
                ]
                if images:
                    self._add(path.relative_to(self.root), path)

    def resolve(self, sample_id: str) -> Path | None:
        canonical = canonical_sample_id(sample_id)
        candidates: set[Path] = set()
        for key in (canonical, canonical.replace("/", "__"), canonical.replace("/", "_")):
            candidates.update(self._paths.get(key, set()))
        existing = sorted(path for path in candidates if path.exists())
        if not existing:
            return None
        if len(existing) > 1:
            joined = ", ".join(str(path) for path in existing)
            raise ValueError(f"ambiguous generated artifacts for {sample_id}: {joined}")
        return existing[0]


def load_frames(path: Path, *, limit: int | None = None) -> list[np.ndarray]:
    """Decode a video or load an image sequence as RGB uint8 arrays."""

    if path.is_dir():
        frame_paths = sorted(
            (item for item in path.iterdir() if item.is_file() and item.suffix.casefold() in IMAGE_SUFFIXES),
            key=lambda item: natural_key(item.name),
        )
        if limit is not None:
            frame_paths = frame_paths[:limit]
        return [np.asarray(Image.open(item).convert("RGB"), dtype=np.uint8) for item in frame_paths]
    if not path.is_file():
        raise FileNotFoundError(f"generated video not found: {path}")

    try:
        import decord

        decord.bridge.set_bridge("native")
        reader = decord.VideoReader(str(path))
        count = len(reader) if limit is None else min(len(reader), limit)
        if count == 0:
            return []
        return [np.asarray(frame, dtype=np.uint8) for frame in reader.get_batch(range(count)).asnumpy()]
    except ImportError:
        pass
    except Exception as exc:
        decord_error = exc
    else:  # pragma: no cover - control cannot reach this after return
        decord_error = None

    try:
        import cv2
    except ImportError as exc:
        detail = f"; decord failed with {decord_error}" if "decord_error" in locals() else ""
        raise RuntimeError(f"video decoding needs decord or opencv-python{detail}") from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"unable to open generated video: {path}")
    frames: list[np.ndarray] = []
    try:
        while limit is None or len(frames) < limit:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


def load_label_frames(
    paths: tuple[Path, ...],
    *,
    start: int,
    count: int,
    size: tuple[int, int],
    background_label: int = 1,
) -> list[np.ndarray]:
    from .metrics import normalize_dataset_labels, resize_labels

    selected = paths[start : start + count]
    return [
        resize_labels(
            normalize_dataset_labels(np.asarray(Image.open(path)), background_label=background_label),
            size,
        )
        for path in selected
    ]


def load_rgba_frames(paths: tuple[Path, ...], *, start: int, count: int) -> list[np.ndarray]:
    return [np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8) for path in paths[start : start + count]]


def load_predicted_masks(root: Path, sample_id: str, *, count: int, size: tuple[int, int]) -> list[np.ndarray]:
    from .metrics import resize_labels

    sample_root = root.expanduser().resolve() / canonical_sample_id(sample_id)
    if not sample_root.is_dir():
        flat = root.expanduser().resolve() / canonical_sample_id(sample_id).replace("/", "__")
        sample_root = flat if flat.is_dir() else sample_root
    if not sample_root.is_dir():
        raise FileNotFoundError(f"predicted mask directory not found for {sample_id}: {sample_root}")
    paths = sorted(
        (
            path
            for path in sample_root.iterdir()
            if path.is_file() and path.suffix.casefold() in {".png", ".tif", ".tiff", ".npy"}
        ),
        key=lambda path: natural_key(path.name),
    )[:count]
    if len(paths) < count:
        raise ValueError(f"{sample_id} has {len(paths)} predicted masks but needs {count}")
    frames: list[np.ndarray] = []
    for path in paths:
        array = np.load(path) if path.suffix.casefold() == ".npy" else np.asarray(Image.open(path))
        frames.append(resize_labels(array, size))
    return frames


def _prediction_value(row: dict[str, Any]) -> Any:
    return next((row[key] for key in PREDICTION_KEYS if key in row and row[key] is not None), None)


def load_answer_predictions(path: Path | None, questions: list[TextQuestion]) -> dict[str, Any]:
    if path is None:
        return {}
    manifest = path.expanduser().resolve()
    payload = _load_records(manifest)
    if isinstance(payload, dict) and isinstance(payload.get("answers"), list):
        payload = payload["answers"]
    if isinstance(payload, dict):
        return {str(key): value for key, value in payload.items() if not isinstance(value, (dict, list))}
    if not isinstance(payload, list):
        raise ValueError(f"answer manifest must be an object or list: {manifest}")

    by_video_count: dict[str, int] = {}
    for question in questions:
        by_video_count[question.video_name] = by_video_count.get(question.video_name, 0) + 1

    predictions: dict[str, Any] = {}
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"answer manifest row {index} is not an object")
        value = _prediction_value(row)
        if value is None:
            continue
        question_id = str(row.get("question_id") or row.get("qid") or "").strip()
        video_name = str(row.get("video_name") or row.get("video_path") or "").strip()
        generic_id = str(row.get("sample_id") or row.get("id") or "").strip()
        question_text = str(row.get("question") or "").strip()
        if question_id:
            predictions[question_id] = value
        if video_name and question_text:
            predictions[f"{video_name}\n{question_text}"] = value
        if video_name and by_video_count.get(video_name) == 1:
            predictions[video_name] = value
        if generic_id:
            predictions[generic_id] = value
    return predictions


def prediction_for_question(predictions: dict[str, Any], question: TextQuestion) -> Any:
    for key in (
        question.question_id,
        f"{question.video_name}\n{question.question}",
        question.video_name,
    ):
        if key in predictions:
            return predictions[key]
    return None
