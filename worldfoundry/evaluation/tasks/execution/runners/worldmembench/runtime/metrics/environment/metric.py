# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""The three formal Environment scores, sharing DA3 inputs and return selection."""

from dataclasses import asdict

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import (
    DataInvalidError,
    EvaluationError,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.aggregation import (
    aggregate_video_summary,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.base import Metric
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.results import (
    ItemResult,
    ResultStatus,
    UnitResult,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.video import VideoReader

from .protocol import protocol_for_sample


class _EnvironmentMetric(Metric):
    def __init__(self, *, reader=None):
        self.reader = reader or VideoReader()

    def evaluate(self, sample, context):
        if sample.get("category") not in (None, "environment"):
            return [UnitResult("video", status=ResultStatus.NOT_APPLICABLE)]
        if error := sample.get("environment_preparation_error"):
            exception = DataInvalidError if error["status"] == "data_invalid" else EvaluationError
            raise exception(error["message"])

        def load_inputs():
            from .artifacts import load_poses
            from .geometry import select_returns

            info = self.reader.probe(sample["video"])
            protocol = protocol_for_sample(sample, total_frames=info.total_frames)
            artifacts = sample.get("artifacts") or {}
            if not artifacts.get("da3"):
                raise DataInvalidError("missing artifacts.da3 (da3/results.npz)")
            poses = load_poses(artifacts["da3"], info.total_frames)
            config = context.metric_config("environment_return_validity")
            returns = select_returns(
                poses,
                protocol,
                rotation_threshold_deg=float(config.get("rotation_threshold_deg", 30.0)),
                translation_threshold=float(config.get("translation_threshold", 0.15)),
                return_buffer_frames=int(config.get("return_buffer_frames", 30)),
            )
            return protocol, poses, returns

        protocol, poses, returns = context.get_cached(("environment:inputs", sample["video"]), load_inputs)
        scores, metadata = self.score(sample, context, protocol, poses, returns)
        return [
            UnitResult(
                "video",
                unit_type="video",
                scores=scores,
                metadata={
                    "video_path": sample["video"],
                    "num_frames": protocol.total_frames,
                    "actions": list(protocol.actions),
                    "phase_transitions": list(protocol.phase_transitions),
                    "tail_hold_transitions": protocol.tail_hold_transitions,
                    "phases": [asdict(phase) for phase in protocol.phases],
                    **metadata,
                },
            )
        ]

    @staticmethod
    def geometry(sample, context, total_frames):
        from .artifacts import load_scene_geometry

        path = (sample.get("artifacts") or {}).get("da3_scene")
        if not path:
            raise DataInvalidError("missing artifacts.da3_scene (da3/scene_geometry.npz)")
        return context.get_cached(("environment:geometry", path), lambda: load_scene_geometry(path, total_frames))

    def aggregate_item(self, sample, units):
        unit = units[0]
        return ItemResult(
            sample["item_id"],
            scores=dict(unit.scores),
            status=unit.status,
            error=unit.error,
            metadata=dict(unit.metadata),
            num_units=1,
        )

    def aggregate_summary(self, items):
        return aggregate_video_summary(items)


class EnvironmentVisibleMemoryMetric(_EnvironmentMetric):
    name = "environment_visible_memory"
    description = "Mean full-frame DINOv2 similarity at DA3 return frames."

    def __init__(self, *, reader=None, backend=None):
        super().__init__(reader=reader)
        self.backend = backend

    def score(self, sample, context, protocol, poses, returns):
        import numpy as np

        from ..features import get_dino_model
        from .backends import DinoVisibleFeatures

        config = context.metric_config(self.name)
        checkpoint = config.get("dino_checkpoint") or context.cache_dir / "metrics" / self.name / "dinov2-large"
        required = sorted({0, *(value.return_index for value in returns)})
        frames = self.reader.read_indices(sample["video"], required)
        backend = self.backend or context.get_resource(
            "environment:visible_dino",
            lambda: DinoVisibleFeatures(
                checkpoint,
                context.device,
                int(config.get("batch_size", 8)),
                model=get_dino_model(context, checkpoint),
            ),
        )
        features = backend.embed([frame.rgb for frame in frames])
        if len(features) != len(required):
            raise EvaluationError("DINO returned the wrong number of feature rows")
        by_index = dict(zip(required, features))
        reference = np.asarray(by_index[0], dtype=np.float64)

        def similarity(index):
            return float(np.clip(reference @ np.asarray(by_index[index], dtype=np.float64), 0.0, 1.0))

        slots = [
            {"slot_id": value.slot_id, "return_index": value.return_index, "similarity": similarity(value.return_index)}
            for value in returns
        ]
        visible = float(np.mean([slot["similarity"] for slot in slots]))
        return {"score": visible}, {"visible_similarity": visible, "slots": slots}


class EnvironmentDepthFidelityMetric(_EnvironmentMetric):
    name = "environment_depth_fidelity"
    description = "Pixel-aligned DA3 depth-order Spearman correlation at return frames."

    def score(self, sample, context, protocol, poses, returns):
        import numpy as np
        from scipy.stats import spearmanr

        geometry = self.geometry(sample, context, protocol.total_frames)
        by_index = geometry.index_to_position
        reference = np.asarray(geometry.depths[by_index[0]], dtype=np.float64).ravel()
        slots = []
        for value in returns:
            sampled = min(by_index, key=lambda index: abs(index - value.return_index))
            candidate = np.asarray(geometry.depths[by_index[sampled]], dtype=np.float64).ravel()
            valid = np.isfinite(reference) & np.isfinite(candidate) & (reference > 0) & (candidate > 0)
            if int(valid.sum()) < 100:
                raise EvaluationError("depth fidelity requires at least 100 valid pixels")
            score = float(spearmanr(reference[valid], candidate[valid]).statistic)
            if not np.isfinite(score):
                raise EvaluationError("depth fidelity produced a non-finite Spearman value")
            slots.append(
                {
                    "slot_id": value.slot_id,
                    "axis": value.axis,
                    "return_status": "returned" if value.returned else "not_revisited",
                    "return_index": value.return_index,
                    "sampled_return_index": sampled,
                    "depth_spearman": score,
                }
            )
        return {"depth_spearman": float(np.mean([slot["depth_spearman"] for slot in slots]))}, {"slots": slots}


class EnvironmentSceneReconstructionMetric(_EnvironmentMetric):
    name = "environment_scene_reconstruction"
    description = "Pure-rotation spherical reconstruction score (1 - reference loss), higher is better."

    def __init__(self, *, reader=None, semantic_filter=None):
        super().__init__(reader=reader)
        self.semantic_filter = semantic_filter

    def score(self, sample, context, protocol, poses, returns):
        import cv2
        import numpy as np

        from .backends import DEFAULT_REJECT_LABELS, SegformerMasks
        from .geometry import compare_range_fields, departure_evidence, spherical_range_field, window_indices

        if any(slot.axis not in {"yaw", "pitch"} for slot in protocol.slots):
            raise EvaluationError("Scene Reconstruction supports pure-rotation yaw/pitch trajectories only")
        config = context.metric_config(self.name)
        geometry = self.geometry(sample, context, protocol.total_frames)
        available = geometry.frame_indices
        home_fraction, home_frames = (
            float(config.get("home_window_fraction", 0.2)),
            int(config.get("home_window_frames", 8)),
        )
        if not 0 < home_fraction <= 1 or home_frames < 1:
            raise ValueError("invalid reconstruction home window")
        phase0 = protocol.phases[0]
        home = window_indices(
            available, phase0.start, int(round(phase0.start + home_fraction * (phase0.end - phase0.start))), home_frames
        )
        radius = int(config.get("revisit_radius_frames", 14))
        threshold = float(context.metric_config("environment_return_validity").get("rotation_threshold_deg", 30.0))
        prepared, required = [], set(home)
        for slot, value in zip(protocol.slots, returns, strict=True):
            indices = window_indices(
                available,
                max(int(available[0]), value.return_index - radius),
                min(int(available[-1]), value.return_index + radius),
                home_frames,
            )
            departure = departure_evidence(poses, protocol.phases[slot.outbound_phase], threshold)
            prepared.append((value, indices, departure))
            required.update(indices)
        masks = None
        if config.get("use_semantic_filter", True) and required:
            backend = self.semantic_filter or context.get_resource(
                "environment:segformer",
                lambda: SegformerMasks(
                    config.get("semantic_checkpoint")
                    or context.cache_dir / "metrics" / self.name / "segformer-b0-ade20k",
                    context.device,
                    int(config.get("semantic_batch_size", 4)),
                    config.get("semantic_reject_labels") or DEFAULT_REJECT_LABELS,
                ),
            )
            indices = sorted(required)
            images = [frame.rgb for frame in self.reader.read_indices(sample["video"], indices)]
            predicted = backend.keep_masks(images)
            if len(predicted) != len(indices):
                raise EvaluationError("SegFormer returned the wrong number of masks")
            height, width = geometry.depths.shape[1:]
            masks = {
                index: cv2.resize(
                    np.asarray(mask, dtype=np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                for index, mask in zip(indices, predicted)
            }
        settings = dict(
            semantic_masks=masks,
            angular_width=int(config.get("angular_width", 128)),
            confidence_quantile=float(config.get("confidence_quantile", 0.5)),
            near_depth_percentile=float(config.get("near_depth_percentile", 5.0)),
            far_depth_percentile=float(config.get("far_depth_percentile", 85.0)),
            max_points_per_frame=int(config.get("max_points_per_frame", 5000)),
        )
        reference = spherical_range_field(geometry, home, poses[0, :3, :3], **settings)
        minimum_bins, require_departure = (
            int(config.get("minimum_common_bins", 64)),
            bool(config.get("require_pose_departure", True)),
        )
        losses, slots = [], []
        for value, indices, departure in prepared:
            revisit = spherical_range_field(geometry, indices, poses[value.return_index, :3, :3], **settings)
            comparison = compare_range_fields(reference, revisit)
            usable = (
                value.returned
                and (departure["departed"] or not require_departure)
                and comparison["common_bins"] >= minimum_bins
            )
            loss = comparison["reconstruction_loss"] if usable else 1.0
            losses.append(float(loss))
            if usable:
                status = "ok"
            elif require_departure and not departure["departed"]:
                status = "not_departed"
            elif not value.returned:
                status = "not_revisited"
            else:
                status = "insufficient_common_bins"
            slots.append(
                {
                    "slot_id": value.slot_id,
                    "axis": value.axis,
                    "status": status,
                    "return_status": "returned" if value.returned else "not_revisited",
                    "return_index": value.return_index,
                    "return_rotation_error_deg": value.rotation_error_deg,
                    "reference_indices": list(home),
                    "revisit_indices": list(indices),
                    **departure,
                    "reconstruction_score": 1.0 - loss,
                    "common_bins": comparison["common_bins"],
                    "union_bins": comparison["union_bins"],
                }
            )
        return {"reconstruction_score": 1.0 - float(np.mean(losses))}, {"slots": slots}
