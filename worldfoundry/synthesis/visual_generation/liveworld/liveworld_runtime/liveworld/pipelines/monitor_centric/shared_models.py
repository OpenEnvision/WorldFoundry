"""Shared model container for batch inference.

All heavyweight models are created once by ``create_shared_models`` and
reused across multiple ``MonitorCentricEvolutionPipeline`` runs.  Each pipeline
receives a ``SharedModels`` instance and uses the pre-loaded models
directly — no lazy loading.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# isort: split
from .observer_adapter import ObserverAdapter

# isort: split
from worldfoundry.base_models.perception_core.general_perception.qwen3_vl_entity import (
    Qwen3VLEntityExtractor,
)

# isort: split
from worldfoundry.base_models.perception_core.segment.sam3.video_segmenter import (
    Sam3VideoSegmenter,
)

# isort: split
from .entity_matcher import EntityMatcher

# isort: split
from .helpers import load_unified_backbone_pipeline, parse_target_resolution

# isort: split
# isort: split
from .logger import logger

# isort: split
# isort: split
from worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.liveworld.pipelines.pointcloud_updater import (
    PointCloudUpdater,
    create_pointcloud_updater,
)


@dataclass
class SharedModels:
    """Pre-loaded model instances shared across batch runs."""

    observer_adapter: ObserverAdapter
    qwen_extractor: Qwen3VLEntityExtractor
    sam3_segmenter: Sam3VideoSegmenter
    pointcloud_handler: PointCloudUpdater
    entity_matcher: EntityMatcher | None


def create_shared_models(config: dict) -> SharedModels:
    """Create all heavyweight models from a config."""
    device_str = config["runtime"]["device"]
    device = torch.device(device_str)
    cpu_offload = config["runtime"]["cpu_offload"]
    auxiliary_cfg = config.get("auxiliary", {})
    event_cfg = config["event"]

    # 1. Observer (UnifiedBackbonePipeline -> ObserverAdapter)
    observer_cfg = config["observer"]
    target_resolution = parse_target_resolution(observer_cfg)
    observer_cfg["height"] = target_resolution[0]
    observer_cfg["width"] = target_resolution[1]

    observer_cpu_offload = cpu_offload.get("observer", False)
    data_pipeline = load_unified_backbone_pipeline(observer_cfg, device, cpu_offload=observer_cpu_offload)
    observer_adapter = ObserverAdapter(data_pipeline)
    logger.info("Loaded observer (UnifiedBackbonePipeline)")

    # 2. Qwen3-VL entity extractor
    qwen_model_path = auxiliary_cfg.get("qwen_model_path", "ckpts/Qwen--Qwen3-VL-8B-Instruct")
    qwen_extractor = Qwen3VLEntityExtractor(model_path=qwen_model_path, device=device_str)
    if event_cfg["cpu_offload"].get("qwen", False) and hasattr(qwen_extractor, "model"):
        qwen_extractor.model.to("cpu")
        torch.cuda.empty_cache()
    logger.info("Loaded Qwen3-VL entity extractor")

    # 3. SAM3 video segmenter
    sam3_model_path = auxiliary_cfg.get("sam3_model_path", "ckpts/facebook--sam3/sam3.pt")
    sam3_segmenter = Sam3VideoSegmenter(checkpoint_path=sam3_model_path)
    logger.info("Loaded SAM3 video segmenter")

    # 4. Point cloud handler (Stream3R)
    pointcloud_backend = auxiliary_cfg.get("pointcloud_backend", "stream3r")
    pointcloud_handler = create_pointcloud_updater(pointcloud_backend, device)
    if hasattr(pointcloud_handler, "set_dynamic_models"):
        pointcloud_handler.set_dynamic_models(
            qwen_model_path=qwen_model_path,
            sam3_model_path=sam3_model_path,
            qwen_extractor=qwen_extractor,
            sam3_segmenter=sam3_segmenter,
            cpu_offload_qwen=event_cfg["cpu_offload"].get("qwen", False),
            cpu_offload_sam3=event_cfg["cpu_offload"].get("sam3", False),
            scene_detect_prompt=event_cfg["detection"]["scene_detect_prompt"],
        )
    logger.info(f"Loaded point cloud handler ({pointcloud_backend})")

    # 5. Entity matcher (DINOv3)
    dedup_cfg = event_cfg.get("deduplication", {})
    dedup_enabled = dedup_cfg.get("enabled", True)
    entity_matcher = None
    if dedup_enabled:
        dedup_model_path = dedup_cfg.get("model_path", "ckpts/facebook--dinov3-vith16plus-pretrain-lvd1689m")
        dedup_similarity_threshold = dedup_cfg.get("similarity_threshold", 0.6)
        entity_matcher = EntityMatcher(
            model_path=dedup_model_path,
            device=device_str,
            similarity_threshold=dedup_similarity_threshold,
        )
        if event_cfg["cpu_offload"].get("dinov3", False) and hasattr(entity_matcher, "model"):
            entity_matcher.model.to("cpu")
            torch.cuda.empty_cache()
        logger.info("Loaded DINOv3 entity matcher")

    logger.info("All shared models loaded")
    return SharedModels(
        observer_adapter=observer_adapter,
        qwen_extractor=qwen_extractor,
        sam3_segmenter=sam3_segmenter,
        pointcloud_handler=pointcloud_handler,
        entity_matcher=entity_matcher,
    )
