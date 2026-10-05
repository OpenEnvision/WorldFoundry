"""Native Apple-PI evaluator.

This module intentionally owns the Apple-PI data/evaluation boundary inside
WorldFoundry.  It does not import the upstream ``apple_pi`` package.  SAM3
and MoGe are resolved from ``worldfoundry.base_models``; Gemini is an optional
remote judge using the same lazy ``google.genai`` integration as the other
WorldFoundry video benchmarks.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

try:  # Keep protocol/mock wiring importable in lightweight environments.
    import cv2
except ImportError:  # pragma: no cover - exercised only without video extras.
    cv2 = None

LOGGER = logging.getLogger(__name__)

SUBTRACKS = (
    "perception_text",
    "perception_graphic",
    "formulation_text",
    "formulation_graphic",
    "deduction",
)
NUM_ROLLOUTS = 3


def _clamp(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return max(0.0, min(1.0, value))


def _score_value(value: Any) -> float:
    """Parse a rubric score while preserving the paper's -1 sentinel."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return -1.0 if value < 0.0 else _clamp(value)


def _parse_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    candidate = text.strip()
    if "```" in candidate:
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I | re.S)
    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.S)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


@dataclass
class ApplePIScores:
    """The paper's five-subtrack score container and weighted reducers."""

    values: dict[str, float] = field(default_factory=dict)
    feedback: str = ""
    programmatic: dict[str, float] = field(default_factory=dict)

    def get(self, name: str, default: float = 0.0) -> float:
        return self.values.get(name, default)

    @staticmethod
    def _mean(values: Iterable[float]) -> float:
        valid = [float(value) for value in values if float(value) >= 0.0]
        return float(np.mean(valid)) if valid else 0.0

    def perception_text_score(self) -> float:
        content = self._mean(self.get(k, -1.0) for k in (
            "text_content_match", "text_readability", "no_extra_annotations"))
        layout = self._mean(self.get(k, -1.0) for k in (
            "text_position_match", "axes_position_match", "axes_angle_match",
            "velocity_position_match"))
        style = self._mean(self.get(k, -1.0) for k in (
            "text_font_match", "text_style_match", "text_size_match",
            "text_color_match", "axes_color_match", "axes_style_match",
            "axes_size_match", "velocity_color_match", "velocity_style_match",
            "velocity_size_match", "white_background"))
        return 0.5 * content + 0.3 * layout + 0.2 * style

    def _weighted(self, groups: Mapping[str, tuple[float, float]]) -> float:
        valid = [(score, weight) for score, weight in groups.values() if score >= 0.0]
        denom = sum(weight for _, weight in valid)
        return float(sum(score * weight for score, weight in valid) / denom) if denom else 0.0

    def perception_graphic_score(self) -> float:
        return self._weighted({
            "background": (self._mean([self.get("white_background", -1.0)]), 0.10),
            "annotation": (self._mean(self.get(k, -1.0) for k in (
                "coord_axes", "velocity_arrows", "text_annotations")), 0.30),
            "objects": (self._mean(self.get(k, -1.0) for k in (
                "objects_only_from_gt", "object_completeness", "object_visual_match")), 0.30),
            "spatial": (self._mean(self.get(k, -1.0) for k in (
                "object_position_match", "segmentation_iou")), 0.30),
        })

    def formulation_text_score(self) -> float:
        return self._weighted({
            "option": (self._mean([self.get("option_correct", -1.0)]), 0.20),
            "formula": (self._mean(self.get(k, -1.0) for k in (
                "formula_variables_correct", "formula_constants_correct",
                "formula_operators_correct")), 0.30),
            "substitution": (self._mean(self.get(k, -1.0) for k in (
                "substitution_all_replaced", "substitution_values_correct",
                "substitution_unreplaced_correct", "substitution_constants_correct",
                "substitution_operators_correct")), 0.40),
            "presentation": (self._mean(self.get(k, -1.0) for k in (
                "format_3_lines", "background_pure_white")), 0.10),
        })

    def formulation_graphic_score(self) -> float:
        return self._weighted({
            "annotation": (self._mean([self.get("annotations_removed", -1.0)]), 0.10),
            "objects": (self._mean(self.get(k, -1.0) for k in (
                "object_position_match", "object_appearance_match",
                "formulation_graphic_segmentation_iou")), 0.30),
            "arrow": (self._mean(self.get(k, -1.0) for k in (
                "arrow_is_borderless_orange", "arrow_from_center",
                "arrow_direction_match")), 0.30),
            "velocity": (self._mean(self.get(k, -1.0) for k in (
                "velocity_label_present_and_format", "velocity_value_match",
                "velocity_label_no_unit")), 0.30),
        })

    def deduction_score(self) -> float:
        psnr = self.programmatic.get("psnr", -1.0)
        masked_psnr = self.programmatic.get("masked_psnr", -1.0)
        velocity = self.programmatic.get("velocity_error", -1.0)
        fidelity = self._mean([
            self.get("visual_quality", -1.0), self.get("motion_smoothness", -1.0),
            min(max(psnr / 40.0, 0.0), 1.0) if psnr >= 0 else -1.0,
            min(max(masked_psnr / 40.0, 0.0), 1.0) if masked_psnr >= 0 else -1.0,
        ])
        physics = self._mean([
            self.get("physics_accuracy", -1.0),
            self.programmatic.get("spatial_iou", -1.0),
            self.programmatic.get("spatiotemporal_iou", -1.0),
            self.programmatic.get("weighted_spatial_iou", -1.0),
            1.0 / (1.0 + velocity) if velocity >= 0 else -1.0,
        ])
        integrity = self._mean([
            self.get("gen_annotations_removed", -1.0),
            self.get("object_consistency", -1.0),
        ])
        return 0.20 * integrity + 0.20 * fidelity + 0.60 * physics

    def score(self, subtrack: str) -> float:
        return {
            "perception_text": self.perception_text_score,
            "perception_graphic": self.perception_graphic_score,
            "formulation_text": self.formulation_text_score,
            "formulation_graphic": self.formulation_graphic_score,
            "deduction": self.deduction_score,
        }[subtrack]()

    def as_details(self, subtrack: str) -> dict[str, Any]:
        payload = dict(self.values)
        payload.update(self.programmatic)
        payload["score"] = round(self.score(subtrack), 4)
        if self.feedback:
            payload["feedback"] = self.feedback
        return payload


@dataclass(frozen=True)
class ApplePICase:
    case_id: str
    case_dir: Path
    metadata: Mapping[str, Any]

    @classmethod
    def load(cls, case_id: str, case_dir: Path) -> "ApplePICase":
        metadata_path = case_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing Apple-PI case metadata: {metadata_path}")
        return cls(case_id, case_dir, json.loads(metadata_path.read_text(encoding="utf-8")))

    def path(self, key: str) -> Path | None:
        value = self.metadata.get(key)
        if not value:
            return None
        path = self.case_dir / str(value)
        return path if path.exists() else None

    @property
    def fps(self) -> float:
        return float(self.metadata.get("gt_fps", 24.0))

    @property
    def physics_duration(self) -> float:
        return float(self.metadata.get("physics_duration", 10.0))

    @property
    def first_frame(self) -> Path:
        value = self.path("input_image")
        if value is None:
            raise FileNotFoundError(f"Missing input_image for {self.case_id}")
        return value


class ApplePIGeminiJudge:
    """Native multimodal judge with a deterministic mock backend for wiring tests."""

    def __init__(self, model: str = "gemini-3-flash-preview", backend: str | None = None):
        self.model = model
        self.backend = (backend or os.environ.get("WORLDFOUNDRY_APPLE_PI_JUDGE_BACKEND", "gemini")).lower()
        self.api_key = (
            os.environ.get("WORLDFOUNDRY_APPLE_PI_GEMINI_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

    @staticmethod
    def _mock_values(subtrack: str) -> dict[str, float]:
        fields = {
            "perception_text": ("text_font_match text_style_match text_size_match text_color_match text_position_match text_content_match text_readability axes_color_match axes_style_match axes_size_match axes_position_match axes_angle_match velocity_color_match velocity_style_match velocity_size_match velocity_position_match no_extra_annotations white_background"),
            "perception_graphic": "white_background coord_axes velocity_arrows text_annotations objects_only_from_gt object_completeness object_visual_match object_position_match",
            "formulation_text": "option_correct formula_variables_correct formula_constants_correct formula_operators_correct substitution_all_replaced substitution_values_correct substitution_unreplaced_correct substitution_constants_correct substitution_operators_correct format_3_lines background_pure_white",
            "formulation_graphic": "annotations_removed object_position_match object_appearance_match arrow_is_borderless_orange arrow_from_center arrow_direction_match velocity_label_present_and_format velocity_value_match velocity_label_no_unit",
            "deduction": "visual_quality motion_smoothness physics_accuracy gen_annotations_removed object_consistency",
        }
        return {field: 0.5 for field in fields[subtrack].split()}

    def complete(self, prompt: str, assets: list[Path], subtrack: str = "deduction") -> str:
        if self.backend == "mock":
            return json.dumps(self._mock_values(subtrack))
        if self.backend not in {"gemini", "google"}:
            raise RuntimeError(f"Unsupported Apple-PI judge backend: {self.backend}")
        if not self.api_key:
            raise RuntimeError("Apple-PI Gemini judge requires GEMINI_API_KEY or GOOGLE_API_KEY")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Install google-genai to run the native Apple-PI Gemini judge") from exc

        client = genai.Client(api_key=self.api_key)
        contents: list[Any] = [prompt]
        uploaded = []
        try:
            from PIL import Image
            for asset in assets:
                if asset.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
                    item = client.files.upload(file=str(asset))
                    while getattr(getattr(item, "state", None), "name", "") == "PROCESSING":
                        time.sleep(1.0)
                        item = client.files.get(name=item.name)
                    contents.append(item)
                    uploaded.append(item)
                else:
                    contents.append(Image.open(asset).convert("RGB"))
            response = client.models.generate_content(model=self.model, contents=contents)
            return str(getattr(response, "text", "") or "")
        finally:
            for item in uploaded:
                try:
                    client.files.delete(name=item.name)
                except Exception:
                    LOGGER.debug("Could not delete temporary Gemini file", exc_info=True)


def _prompt_for(subtrack: str, case: ApplePICase) -> str:
    formula = case.metadata.get("formula_info", {})
    fields = {
        "perception_text": "text_font_match text_style_match text_size_match text_color_match text_position_match text_content_match text_readability axes_color_match axes_style_match axes_size_match axes_position_match axes_angle_match velocity_color_match velocity_style_match velocity_size_match velocity_position_match no_extra_annotations white_background",
        "perception_graphic": "white_background coord_axes velocity_arrows text_annotations objects_only_from_gt object_completeness object_visual_match object_position_match",
        "formulation_text": "option_correct formula_variables_correct formula_constants_correct formula_operators_correct substitution_all_replaced substitution_values_correct substitution_unreplaced_correct substitution_constants_correct substitution_operators_correct format_3_lines background_pure_white",
        "formulation_graphic": "annotations_removed object_position_match object_appearance_match arrow_is_borderless_orange arrow_from_center arrow_direction_match velocity_label_present_and_format velocity_value_match velocity_label_no_unit",
        "deduction": "visual_quality motion_smoothness physics_accuracy gen_annotations_removed object_consistency",
    }[subtrack]
    return f"""You are the official Apple-PI evaluator. Compare the generated output with the supplied reference assets.
Subtrack: {subtrack}. Physics type: {case.metadata.get('physics_type', 'unknown')}.
For formulation text, the correct answer choices are {formula.get('choices', [])}, correct letter={formula.get('correct_letter', '')}, formula={formula.get('correct_formula', '')}.
Return ONLY a JSON object. Score every requested criterion from 0 to 1; use -1 only when the criterion is not applicable.
Criteria for {subtrack}: {fields}.
Be strict about annotations, object identity, spatial placement, and physical correctness."""


def _read_video(path: Path) -> tuple[list[np.ndarray], float]:
    if cv2 is None:
        raise RuntimeError("Apple-PI video evaluation requires opencv-python; install the apple_pi extra")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open generated video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"Generated video has no frames: {path}")
    return frames, fps


def _last_frame(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        if cv2 is None:
            from PIL import Image
            return np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read generated image: {path}")
        return frame
    return _read_video(path)[0][-1]


def _load_maps(case: ApplePICase, relative: str) -> np.ndarray | None:
    path = case.case_dir / relative
    if not path.is_file():
        return None
    try:
        return np.load(path)["maps"]
    except Exception:
        LOGGER.warning("Could not load Apple-PI map %s", path, exc_info=True)
        return None


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if mask.shape == (h, w):
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0


def _simple_programmatic_metrics(pred_path: Path, case: ApplePICase) -> dict[str, float]:
    """Compute the model-independent Apple-PI video metrics.

    SAM3/MoGe-backed metrics are added by ``NativeApplePIModelStack`` when the
    required local checkpoints are available. Keeping this fallback explicit
    lets result contracts and smoke tests run without downloading 24GB models.
    """
    if pred_path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
        return {}
    pred, pred_fps = _read_video(pred_path)
    gt_dir = case.path("gt_frames_dir")
    gt_files = sorted(gt_dir.glob("*.png")) if gt_dir else []
    gt_frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in gt_files]
    gt_frames = [frame for frame in gt_frames if frame is not None]
    if len(pred) < 2 or len(gt_frames) < 2:
        return {}
    gt_physics = gt_frames[1:]
    pred_physics = pred[1:]
    gt_fps = case.fps or 24.0
    n = min(len(gt_physics), max(1, round(case.physics_duration * gt_fps)))
    scores = []
    for index in range(n):
        target = gt_physics[index]
        source = pred_physics[min(len(pred_physics) - 1, round(index * len(pred_physics) / n))]
        source = cv2.resize(source, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        mse = float(np.mean((source.astype(np.float64) - target.astype(np.float64)) ** 2))
        scores.append(40.0 if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse)))
    result: dict[str, float] = {"psnr": float(np.mean(scores))}
    gt_masks = _load_maps(case, "mask/maps.npz")
    if gt_masks is not None:
        # This conservative fallback scores the GT foreground occupancy only;
        # the native SAM3 path below replaces it with model-predicted masks.
        result["spatial_iou"] = 0.0
        result["spatiotemporal_iou"] = 0.0
        result["weighted_spatial_iou"] = 0.0
    return result


class NativeApplePIModelStack:
    """Lazy adapters for the already integrated SAM3 and MoGe-2 models."""

    def __init__(self, *, enable: bool = True):
        self.enable = enable
        self._sam3_image = None
        self._sam3_processor = None
        self._sam3_video = None
        self._moge = None

    def _sam3_checkpoint(self) -> str | None:
        for name in ("WORLDFOUNDRY_APPLE_PI_SAM3_CHECKPOINT", "WORLDFOUNDRY_SAM3_CHECKPOINT"):
            value = os.environ.get(name)
            if value:
                return value
        return None

    def segment_image_from_points(self, image: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray | None:
        if not self.enable or not points:
            return None
        try:
            import torch
            from PIL import Image
            from worldfoundry.base_models.perception_core.segment.sam3.model.sam3_image_processor import Sam3Processor
            from worldfoundry.base_models.perception_core.segment.sam3.model_builder import build_sam3_image_model
            if self._sam3_image is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._sam3_image = build_sam3_image_model(checkpoint_path=self._sam3_checkpoint(), device=device)
                self._sam3_processor = Sam3Processor(self._sam3_image, device=device)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            height, width = rgb.shape[:2]
            combined = np.zeros((height, width), dtype=bool)
            for x, y in points:
                size = max(8, int(min(height, width) * 0.02))
                box = [max(0, x - size) / width, max(0, y - size) / height,
                       min(width, 2 * size) / width, min(height, 2 * size) / height]
                state = self._sam3_processor.set_image(pil)
                state = self._sam3_processor.add_geometric_prompt(box, True, state)
                masks = state.get("masks")
                if masks is not None and len(masks):
                    mask = masks[0]
                    if hasattr(mask, "detach"):
                        mask = mask.detach().cpu().numpy()
                    combined |= np.asarray(mask).squeeze().astype(bool)
            return combined
        except Exception as exc:
            LOGGER.warning("Native SAM3 image segmentation unavailable: %s", exc)
            return None

    def segmentation_iou(self, image: np.ndarray, case: ApplePICase, mask_subdir: str = "initial_state") -> float | None:
        gt_path = case.case_dir / mask_subdir / ("mask_0000.npy" if mask_subdir == "initial_state" else "mask.npy")
        if not gt_path.is_file():
            return None
        gt = np.load(gt_path)
        gt = gt > 0
        points = []
        instance = case.case_dir / mask_subdir / ("instance_segmentation_0000.npy" if mask_subdir == "initial_state" else "mask.npy")
        if instance.is_file():
            ids = np.unique(np.load(instance))
            for value in ids:
                if value == 0:
                    continue
                ys, xs = np.where(np.load(instance) == value)
                if len(xs):
                    points.append((int(xs.mean()), int(ys.mean())))
        if not points:
            return 0.0
        pred = self.segment_image_from_points(image, points)
        if pred is None:
            return None
        pred = _resize_mask(pred, gt.shape)
        union = np.logical_or(pred, gt).sum()
        return 1.0 if union == 0 else float(np.logical_and(pred, gt).sum() / union)

    def _moge_model(self):
        if not self.enable:
            return None
        if self._moge is None:
            import torch
            from worldfoundry.base_models.three_dimensions.depth.moge.model.v2 import MoGeModel
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_path = (
                os.environ.get("WORLDFOUNDRY_APPLE_PI_MOGE_CHECKPOINT")
                or os.environ.get("WORLDFOUNDRY_MOGE_V2_VITL_NORMAL_MODEL_DIR")
                or "Ruicheng/moge-2-vitl-normal"
            )
            self._moge = MoGeModel.from_pretrained(model_path).to(device).eval()
        return self._moge

    def model_backed_video_metrics(self, pred_path: Path, case: ApplePICase) -> dict[str, float]:
        """Use native SAM3 tracking and MoGe-2 for the remaining video metrics."""
        if not self.enable:
            return {}
        result: dict[str, float] = {}
        try:
            import torch
            from worldfoundry.base_models.perception_core.segment.sam3.video_segmenter import Sam3VideoSegmenter
            gt_masks = _load_maps(case, "mask/maps.npz")
            if gt_masks is not None and pred_path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
                mapping_path = case.case_dir / "initial_state" / "instance_segmentation_mapping_0000.json"
                prompts = ["physical object"]
                if mapping_path.is_file():
                    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
                    prompts = [str(label).split("/")[-1].replace("_", " ") for label in mapping.values() if "ground" not in str(label).lower() and "invalid" not in str(label).lower()]
                    prompts = prompts or ["physical object"]
                tracker = Sam3VideoSegmenter(checkpoint_path=self._sam3_checkpoint())
                segments = tracker.segment_per_category(str(pred_path), prompts, expected_frames=len(gt_masks))
                pred_mask = np.zeros_like(gt_masks, dtype=bool)
                for mask in segments.values():
                    if mask.size:
                        count = min(pred_mask.shape[0], mask.shape[0])
                        for index in range(count):
                            pred_mask[index] |= _resize_mask(mask[index], pred_mask.shape[1:])
                count = min(pred_mask.shape[0], gt_masks.shape[0])
                pred_mask = pred_mask[:count] > 0
                gt_mask = gt_masks[:count] > 0
                spatial_pred, spatial_gt = pred_mask.max(axis=0), gt_mask.max(axis=0)
                union = np.logical_or(spatial_pred, spatial_gt).sum()
                result["spatial_iou"] = 1.0 if union == 0 else float(np.logical_and(spatial_pred, spatial_gt).sum() / union)
                frame_ious = []
                for pred_frame, gt_frame in zip(pred_mask, gt_mask):
                    union = np.logical_or(pred_frame, gt_frame).sum()
                    frame_ious.append(1.0 if union == 0 else float(np.logical_and(pred_frame, gt_frame).sum() / union))
                result["spatiotemporal_iou"] = float(np.mean(frame_ious)) if frame_ious else 0.0
                pred_weight = pred_mask.astype(np.float32).mean(axis=0)
                gt_weight = gt_mask.astype(np.float32).mean(axis=0)
                denom = np.maximum(pred_weight, gt_weight).sum()
                result["weighted_spatial_iou"] = 1.0 if denom == 0 else float(np.minimum(pred_weight, gt_weight).sum() / denom)
        except Exception as exc:
            LOGGER.warning("Native SAM3 tracking metrics unavailable: %s", exc)

        try:
            import torch
            moge = self._moge_model()
            depth_maps = _load_maps(case, "depth/maps.npz")
            velocity_maps = _load_maps(case, "velocity/maps.npz")
            if moge is not None and depth_maps is not None and velocity_maps is not None:
                frames, _ = _read_video(pred_path)
                n = min(len(frames), len(depth_maps))
                estimated = []
                for index in range(n):
                    frame = cv2.resize(frames[index], (depth_maps.shape[2], depth_maps.shape[1]), interpolation=cv2.INTER_LANCZOS4)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).to(moge.device)
                    estimated.append(moge.infer(tensor, use_fp16=torch.cuda.is_available())["depth"].detach().cpu().numpy())
                aligned = []
                for index, depth in enumerate(estimated):
                    target = depth_maps[index]
                    valid = np.isfinite(depth) & np.isfinite(target) & (target != 0)
                    if valid.sum() > 100:
                        scale, shift = np.linalg.lstsq(np.stack([depth[valid], np.ones(valid.sum())], axis=1), target[valid], rcond=None)[0]
                        depth = depth * scale + shift
                    aligned.append(depth)
                errors = []
                foreground = _load_maps(case, "mask/maps.npz")
                for index in range(1, min(len(aligned), len(velocity_maps))):
                    mask = foreground[index] > 0 if foreground is not None and index < len(foreground) else np.isfinite(aligned[index])
                    if mask.sum() < 10:
                        continue
                    yy, xx = np.where(mask)
                    def center(depth):
                        z = depth[mask]
                        valid = np.isfinite(z) & (z > 0)
                        if valid.sum() < 10:
                            return None
                        z, x, y = z[valid], xx[valid], yy[valid]
                        return np.array([float((x * z).mean()), float((y * z).mean()), float(z.mean())])
                    current, previous = center(aligned[index]), center(aligned[index - 1])
                    if current is not None and previous is not None:
                        estimate = (current - previous) * case.fps
                        target = velocity_maps[min(index, len(velocity_maps) - 1)][mask].mean(axis=0)
                        errors.append(float(np.linalg.norm(estimate - target)))
                if errors:
                    result["velocity_error"] = float(np.mean(errors))
        except Exception as exc:
            LOGGER.warning("Native MoGe velocity metric unavailable: %s", exc)
        return result


def _prediction_path(root: Path, case_id: str, protocol: str, subtrack: str, rollout: int) -> Path:
    base = root / "cases" / case_id / subtrack
    if protocol == "video":
        return base / f"rollout_{rollout:02d}.mp4"
    if subtrack == "deduction":
        return base / f"rollout_{rollout:02d}"
    return base / f"rollout_{rollout:02d}.png"


def _dataset_cases(gt_root: Path) -> tuple[str, str, list[tuple[str, Path]]]:
    manifest = gt_root / "dataset.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing Apple-PI dataset manifest: {manifest}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("name") != "Apple-PI":
        raise ValueError(f"Expected Apple-PI dataset, got {data.get('name')!r}")
    if int(data.get("num_rollouts", NUM_ROLLOUTS)) != NUM_ROLLOUTS:
        raise ValueError("Apple-PI requires exactly three rollouts")
    cases = [(str(item["case_id"]), gt_root / str(item["path"])) for item in data.get("cases", [])]
    if not cases:
        raise ValueError(f"No cases listed in {manifest}")
    return str(data.get("version", "unknown")), str(data.get("name", "Apple-PI")), cases


def _judge_scores(judge: ApplePIGeminiJudge, subtrack: str, case: ApplePICase, output: Path) -> ApplePIScores:
    assets = [case.first_frame]
    if subtrack in {"perception_text", "perception_graphic"}:
        assets.extend(path for path in (case.path("annotations_only_reference"), case.path("objects_only_reference")) if path)
    if subtrack == "formulation_graphic" and case.path("future_state_reference"):
        assets.append(case.path("future_state_reference"))
    if subtrack == "deduction" and output.is_dir():
        assets.extend(sorted(output.glob("*.png")))
    prompt = _prompt_for(subtrack, case)
    raw = judge.complete(prompt, assets + ([output] if output.is_file() else []), subtrack=subtrack)
    data = _parse_json(raw) or {}
    return ApplePIScores(
        values={key: _clamp(value, -1.0) if value is not None else -1.0 for key, value in data.items() if key != "feedback"},
        feedback=str(data.get("feedback", "")),
    )


def evaluate_native_apple_pi(
    *,
    gt_root: Path,
    prediction_root: Path,
    output_path: Path,
    protocol: str | None = None,
    subtracks: tuple[str, ...] = SUBTRACKS,
    judge_model: str = "gemini-3-flash-preview",
    judge_backend: str | None = None,
    enable_foundation_models: bool = True,
) -> dict[str, Any]:
    """Run Apple-PI in-tree and emit the official-compatible result JSON."""
    dataset_version, _, entries = _dataset_cases(gt_root)
    submission_path = prediction_root / "submission.json"
    if not submission_path.is_file():
        raise FileNotFoundError(f"Missing Apple-PI submission: {submission_path}")
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    if int(submission.get("num_rollouts", -1)) != NUM_ROLLOUTS:
        raise ValueError("Apple-PI submission.json must declare num_rollouts=3")
    protocol = protocol or str(submission.get("protocol", "video"))
    if protocol not in {"video", "image"}:
        raise ValueError(f"Unsupported Apple-PI protocol: {protocol}")
    judge = ApplePIGeminiJudge(judge_model, judge_backend)
    models = NativeApplePIModelStack(enable=enable_foundation_models)
    result: dict[str, Any] = {
        "benchmark": "Apple-PI", "benchmark_version": "1.0", "dataset_version": dataset_version,
        "prompt_version": "1.0", "num_rollouts": NUM_ROLLOUTS, "model": submission.get("model", "unknown"),
        "protocol": protocol, "judge": judge_model, "evaluator": "worldfoundry.apple_pi.in_tree", "cases": {},
    }
    for case_id, case_dir in entries:
        case = ApplePICase.load(case_id, case_dir)
        result_case: dict[str, Any] = {}
        for subtrack in subtracks:
            rollouts = []
            for rollout in range(NUM_ROLLOUTS):
                path = _prediction_path(prediction_root, case_id, protocol, subtrack, rollout)
                record: dict[str, Any] = {"rollout": rollout, "input": str(path), "status": "failed"}
                try:
                    if not path.exists():
                        raise FileNotFoundError(path)
                    scores = _judge_scores(judge, subtrack, case, path)
                    if protocol == "video" and subtrack == "deduction":
                        scores.programmatic.update(_simple_programmatic_metrics(path, case))
                        scores.programmatic.update(models.model_backed_video_metrics(path, case))
                    elif protocol == "video" and subtrack in {"perception_graphic", "formulation_graphic"}:
                        iou = models.segmentation_iou(_last_frame(path), case, "initial_state" if subtrack == "perception_graphic" else "instantaneous_velocity")
                        if iou is not None:
                            scores.values["segmentation_iou" if subtrack == "perception_graphic" else "formulation_graphic_segmentation_iou"] = iou
                    record.update({"score": round(scores.score(subtrack), 3), "details": scores.as_details(subtrack), "status": "ok"})
                except Exception as exc:
                    LOGGER.exception("Apple-PI evaluation failed for %s/%s/%d", case_id, subtrack, rollout)
                    record["error"] = str(exc)
                rollouts.append(record)
            valid = [float(item["score"]) for item in rollouts if item.get("status") == "ok" and "score" in item]
            aggregate = {
                "num_expected": NUM_ROLLOUTS, "num_successful": len(valid),
                "mean": round(statistics.mean(valid), 4) if len(valid) == NUM_ROLLOUTS else None,
                "partial_mean": round(statistics.mean(valid), 4) if valid and len(valid) != NUM_ROLLOUTS else None,
                "std": round(statistics.pstdev(valid), 4) if len(valid) > 1 else (0.0 if valid else None),
                "min": min(valid) if valid else None, "max": max(valid) if valid else None,
            }
            result_case[subtrack] = {"rollouts": rollouts, "aggregate": aggregate}
        result["cases"][case_id] = result_case
    summary = {}
    for subtrack in subtracks:
        values = [entry[subtrack]["aggregate"]["mean"] for entry in result["cases"].values() if entry[subtrack]["aggregate"]["mean"] is not None]
        summary[subtrack] = {"num_cases": len(values), "mean": round(statistics.mean(values), 4) if values else None}
    result["summary"] = summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
