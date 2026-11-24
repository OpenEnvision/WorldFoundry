"""Event-centric pipeline orchestration.

This module wires EventAgent, EventPool, EventProjector, and LiveWorld observer
into a single iterative pipeline.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

# isort: split
# Import logger FIRST — sets SAM3_TQDM_DISABLE env var at module level,
# before SAM3 modules evaluate TQDM_DISABLE at their import time.
from .logger import logger, suppress_verbose_logs

# isort: split

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from worldfoundry.base_models.perception_core.general_perception.qwen3_vl_entity import (
    Qwen3VLEntityExtractor,
)

# isort: split
from worldfoundry.base_models.perception_core.segment.sam3.video_segmenter import (
    Sam3VideoSegmenter,
)

# isort: split
from worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.liveworld.geometry_utils import (
    BackboneInferenceOptions,
    _get_visible_points_and_coverage_gpu,
    compute_iteration_plan,
    get_visible_points_for_frame,
    render_projection,
)

# isort: split
from worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.liveworld.pipelines.pipeline_unified_backbone import (
    generate_scene_projection_from_pointcloud,
    retrieve_reference_frames,
)

# isort: split
from worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.liveworld.geometry_utils import (
    scale_intrinsics,
)

# isort: split
from worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.liveworld.pipelines.pointcloud_updater import (
    create_pointcloud_updater,
)

# isort: split
# isort: split
from worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.liveworld.pipelines.pointcloud_updater import (
    ReconstructionResult,
)

# isort: split
# isort: split
from .detector_agent import DetectorAgent

# isort: split
# isort: split
from .event_pool import EventPool, EventPoolConfig

# isort: split
# isort: split
from .event_projector import EventProjector

# isort: split
# isort: split
from .shared_models import SharedModels

# isort: split
# isort: split
from .helpers import (
    load_frame_from_image,
    load_geometry_poses,
    load_unified_backbone_pipeline,
    parse_target_resolution,
)

# isort: split
from .io_utils import (
    decode_latent_to_rgb,
    ensure_dir,
    save_image,
    save_json,
    save_pointcloud_ply,
    save_video,
    visualize_projection_latent,
)

# isort: split
from .observer_adapter import ObserverAdapter, ObserverIterationInput

# isort: split
from worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.liveworld.geometry_utils import (
    project_points,
    transform_points,
)

from .event_types import EventObservation, EventVideo, make_event_id
from .projection_compositor import ProjectionCompositor, overlay_fg_on_scene
from .world_state import WorldState


@dataclass
class EventCentricRuntime:
    """Runtime configuration for event-centric inference."""

    device: str
    seed: int
    cpu_offload: dict[str, bool]


@dataclass
class WorldStateGeometry:
    """Geometry and coordinate system state for the world.

    Stores poses, intrinsics, and the alignment transform needed for projection
    rendering and point cloud updates.
    """

    # Camera-to-world poses aligned to the reconstructed point cloud coordinate system.
    poses_c2w: np.ndarray | None = None  # (N, 4, 4) float32
    # Original geometry poses (before alignment transform).
    geometry_poses_c2w: np.ndarray | None = None  # (N, 4, 4) float32
    # Camera intrinsics estimated by Stream3R/MapAnything.
    intrinsics: np.ndarray | None = None  # (3, 3) float32
    # Resolution at which intrinsics are defined.
    intrinsics_size: tuple[int, int] | None = None  # (H, W)
    # Transform from geometry coordinates to world (Stream3R) coordinates.
    alignment_transform: np.ndarray | None = None  # (4, 4) float32
    # First frame image (PIL Image).
    first_frame: Image.Image | None = None
    # Start frame index for video.
    start_frame: int = 0
    # All generated frames accumulated during iterations.
    all_generated_frames: list[np.ndarray] = field(default_factory=list)
    # Dynamic mask from first frame (for separating fg/bg in P1 mode).
    first_frame_dynamic_mask: np.ndarray | None = None  # (H, W) bool
    # Accumulated scene projection pixel frames (decoded from target scene_proj each iter).
    # List of [H, W, 3] uint8 numpy arrays. Used to build preceding_scene_proj.
    all_scene_proj_frames: list[np.ndarray] = field(default_factory=list)
    # Accumulated fg projection pixel frames (decoded from target fg_proj each iter).
    # List of [H, W, 3] uint8 numpy arrays. Used to build preceding_fg_proj.
    all_fg_proj_frames: list[np.ndarray] = field(default_factory=list)
    # Per-frame visible point indices for reference frame retrieval (3D IoU).
    # Key: global frame index, Value: indices of visible points in world point cloud.
    frame_visible_points: dict[int, np.ndarray] = field(default_factory=dict)
    # Per-frame dynamic masks for foreground removal in scene references.
    # Key: global frame index, Value: (H, W) bool mask, True = foreground.
    frame_dynamic_masks: dict[int, np.ndarray] = field(default_factory=dict)


class MonitorCentricEvolutionPipeline:
    """Main event-centric pipeline controller."""

    def __init__(
        self,
        config: dict,
        shared_models: SharedModels | None = None,
    ) -> None:
        # Suppress verbose third-party logs (SAM3, etc.)
        suppress_verbose_logs()

        self.config = config
        self.runtime = EventCentricRuntime(
            device=config["runtime"]["device"],
            seed=config["runtime"]["seed"],
            cpu_offload=config["runtime"]["cpu_offload"],
        )
        if "observer" not in self.runtime.cpu_offload:
            raise ValueError("runtime.cpu_offload must include 'observer' key")

        device_str = self.runtime.device
        device = torch.device(device_str)

        event_cfg = config["event"]
        projection_cfg = event_cfg.get("projection", {})

        # ===========================================================================
        # 0. Parse target resolution from observer config
        # ===========================================================================
        self.target_resolution = parse_target_resolution(config["observer"])  # (H, W)
        observer_cfg = config["observer"]
        observer_cfg["height"] = self.target_resolution[0]
        observer_cfg["width"] = self.target_resolution[1]

        # ===========================================================================
        # 1. Config-derived paths (used by BackboneInferenceOptions and EventProjector)
        # ===========================================================================
        auxiliary_cfg = config.get("auxiliary", {})
        self._qwen_model_path = auxiliary_cfg.get("qwen_model_path", "ckpts/Qwen--Qwen3-VL-8B-Instruct")
        self._sam3_model_path = auxiliary_cfg.get("sam3_model_path", "ckpts/facebook--sam3/sam3.pt")
        self._pointcloud_backend_name = auxiliary_cfg.get("pointcloud_backend", "stream3r")
        self._stream3r_model_path = auxiliary_cfg.get(
            "stream3r_model_path",
            event_cfg.get("projection", {}).get("stream3r_model_path", "ckpts/yslan--STream3R"),
        )
        self._device = device
        self._device_str = device_str

        # ===========================================================================
        # 2. Models — use shared instances or load eagerly
        # ===========================================================================
        if shared_models is not None:
            # Batch mode: reuse pre-loaded models from SharedModels.
            self.observer_adapter = shared_models.observer_adapter
            self.qwen_extractor = shared_models.qwen_extractor
            self.sam3_segmenter = shared_models.sam3_segmenter
            self.pointcloud_handler = shared_models.pointcloud_handler
            _entity_matcher = shared_models.entity_matcher
        else:
            # Single mode: load all models eagerly.
            observer_cpu_offload = self.runtime.cpu_offload.get("observer", False)
            data_pipeline = load_unified_backbone_pipeline(config["observer"], device, cpu_offload=observer_cpu_offload)
            self.observer_adapter = ObserverAdapter(data_pipeline)

            self.qwen_extractor = Qwen3VLEntityExtractor(
                model_path=self._qwen_model_path,
                device=device_str,
            )
            self.sam3_segmenter = Sam3VideoSegmenter(
                checkpoint_path=self._sam3_model_path,
            )
            self.pointcloud_handler = create_pointcloud_updater(self._pointcloud_backend_name, device)
            if hasattr(self.pointcloud_handler, "set_dynamic_models"):
                self.pointcloud_handler.set_dynamic_models(
                    qwen_model_path=self._qwen_model_path,
                    sam3_model_path=self._sam3_model_path,
                    qwen_extractor=self.qwen_extractor,
                    sam3_segmenter=self.sam3_segmenter,
                    cpu_offload_qwen=event_cfg["cpu_offload"].get("qwen", False),
                    cpu_offload_sam3=event_cfg["cpu_offload"].get("sam3", False),
                    scene_detect_prompt=event_cfg["detection"]["scene_detect_prompt"],
                )
            _entity_matcher = None  # lazy-loaded inside EventPool on first use

        # DetectorAgent is lightweight (wraps shared Qwen + SAM3 with per-config
        # detect_prompt), so it is always built fresh per pipeline instance.
        self.detector = DetectorAgent(
            qwen_model=self.qwen_extractor,
            sam3_model=self.sam3_segmenter,
            device=self.runtime.device,
            detect_prompt=event_cfg["detection"]["scene_detect_prompt"],
            cpu_offload_qwen=event_cfg["cpu_offload"].get("qwen", False),
            cpu_offload_sam3=event_cfg["cpu_offload"].get("sam3", False),
        )

        # Intermediate event detector — uses a separate prompt from scene detection.
        intermediate_cfg = event_cfg.get("intermediate_detection", {})
        intermediate_detect_prompt = intermediate_cfg.get("detect_prompt", "")
        if intermediate_detect_prompt and intermediate_cfg.get("enabled", False):
            self.intermediate_detector = DetectorAgent(
                qwen_model=self.qwen_extractor,
                sam3_model=self.sam3_segmenter,
                device=self.runtime.device,
                detect_prompt=intermediate_detect_prompt,
                cpu_offload_qwen=event_cfg["cpu_offload"].get("qwen", False),
                cpu_offload_sam3=event_cfg["cpu_offload"].get("sam3", False),
            )
        else:
            self.intermediate_detector = None

        # Build BackboneInferenceOptions (config only, no model loading).
        self._backbone_options = BackboneInferenceOptions(
            cpu_offload=self.runtime.cpu_offload.get("observer", False),
            pointcloud_backend=self._pointcloud_backend_name,
            stream3r_model_path=self._stream3r_model_path,
            qwen_model_path=self._qwen_model_path,
            sam3_model_path=self._sam3_model_path,
        )
        self._backbone_options.target_hw = self.target_resolution
        self._backbone_options.stream3r_frame_sample_rate = 1
        window_size = auxiliary_cfg.get("stream3r_window_size")
        if window_size is not None:
            self._backbone_options.stream3r_window_size = int(window_size)
        stream3r_update_mode = str(auxiliary_cfg.get("stream3r_update_mode", "complete")).strip().lower()
        if stream3r_update_mode not in {"complete", "freeze"}:
            raise ValueError(
                f"Invalid auxiliary.stream3r_update_mode={stream3r_update_mode!r}. Expected 'complete' or 'freeze'."
            )
        self._backbone_options.stream3r_update_mode = stream3r_update_mode
        scene_voxel_size = auxiliary_cfg.get("scene_voxel_size")
        if scene_voxel_size is not None:
            self._backbone_options.voxel_size = float(scene_voxel_size)
        else:
            # null → disable voxel downsampling for scene point cloud
            self._backbone_options.voxel_size = 0

        # ===========================================================================
        # 3. Event pipeline components (pool, projector, compositor)
        # ===========================================================================
        dedup_cfg = event_cfg.get("deduplication", {})
        scene_reg_cfg = event_cfg.get("scene_registration", {})

        # Load preset storyline if enabled (skip Qwen I2V prompt generation).
        self._use_preset_storyline = bool(event_cfg.get("evolution", {}).get("use_preset_storyline", False))
        preset_storyline = None
        self._preset_entity_names: list[str] | None = None
        if self._use_preset_storyline:
            storyline_path = Path(config["storyline_file"])  # already resolved
            with open(storyline_path) as f_sl:
                storyline_data = json.load(f_sl)
            preset_storyline = storyline_data  # contains "scene_text" + "steps"
            logger.info(f"Loaded preset storyline ({len(preset_storyline['steps'])} steps) from {storyline_path}")
            if "entities" in storyline_data:
                self._preset_entity_names = storyline_data["entities"]
                logger.info(f"Using preset entities from storyline: {self._preset_entity_names}")

        # Merge CLI entity names (--entity-names) into preset list.
        cli_entities = event_cfg.get("cli_entity_names")
        if cli_entities:
            existing = self._preset_entity_names or []
            existing_lower = {e.lower().strip() for e in existing}
            for name in cli_entities:
                if name.lower().strip() not in existing_lower:
                    existing.append(name)
                    existing_lower.add(name.lower().strip())
            self._preset_entity_names = existing
            logger.info(f"CLI entity names merged: {self._preset_entity_names}")

        self.event_pool = EventPool(
            config=EventPoolConfig(
                max_events=event_cfg["max_events"],
                system_prompt=event_cfg["evolution"]["system_prompt"],
                dedup_enabled=dedup_cfg.get("enabled", True),
                dedup_model_path=dedup_cfg.get("model_path", "ckpts/facebook--dinov3-vith16plus-pretrain-lvd1689m"),
                dedup_similarity_threshold=dedup_cfg.get("similarity_threshold", 0.6),
                scene_entity_similarity_threshold=float(scene_reg_cfg.get("entity_similarity_threshold", 0.45)),
                preset_storyline=preset_storyline,
            ),
            device=self.runtime.device,
            cpu_offload_dinov3=event_cfg["cpu_offload"].get("dinov3", False),
            cpu_offload_qwen=event_cfg["cpu_offload"].get("qwen", False),
            default_horizon=observer_cfg["frames_per_iter"],
            default_fps=observer_cfg.get("fps", 16),
            qwen_provider=lambda: self.qwen_extractor,
            entity_matcher=_entity_matcher,
        )

        self.event_projector = EventProjector(
            sam3_model_path=self._sam3_model_path,
            device=self.runtime.device,
            conf_threshold=projection_cfg["conf_threshold"],
            rotation_angle=projection_cfg["rotation_angle"],
            stride=1,
            fg_mask_erode=projection_cfg.get("fg_mask_erode", 0),
        )
        logger.info("Projection stride policy: stream3r_frame_sample_rate=1")

        self.projection_compositor = ProjectionCompositor()

        # ===========================================================================
        # 4. World state
        # ===========================================================================
        self.world_state = WorldState()
        self.geometry = WorldStateGeometry()

    # =========================================================================
    # Device helpers
    # =========================================================================

    def _ensure_vae_on_device(self):
        """Ensure the shared VAE is on the pipeline device (may be on CPU after offload)."""
        vae = self.observer_adapter.pipeline.vae
        vae_device = next(vae.model.parameters()).device
        if vae_device != self._device:
            vae.model.to(self._device)
            vae.mean = vae.mean.to(self._device)
            vae.std = vae.std.to(self._device)

    # =========================================================================
    # Stream3R helpers
    # =========================================================================

    def _get_shared_stream3r(self):
        """Return (model, session, frames_fed) from shared Stream3RUpdater."""
        if self._pointcloud_backend_name != "stream3r":
            raise TypeError(f"event projection requires Stream3R backend, got {self._pointcloud_backend_name}")
        handler = self.pointcloud_handler
        if handler.model is None:
            raise RuntimeError("Stream3R model is not loaded. Run world initialization before event projection.")
        if handler.session is None:
            raise RuntimeError("Stream3R session is not initialized. Run world initialization before event projection.")
        # Ensure model is on GPU (may have been offloaded to CPU)
        model_device = next(handler.model.parameters()).device
        if model_device != self._device:
            handler.model.to(self._device)
        return handler.model, handler.session, handler.frames_fed

    def _sync_stream3r_frames_fed(self, new_count: int):
        """Sync frames_fed counter after event frames were fed into the shared session."""
        if self._pointcloud_backend_name != "stream3r":
            raise TypeError(f"event projection requires Stream3R backend, got {self._pointcloud_backend_name}")
        handler = self.pointcloud_handler
        handler.frames_fed = new_count

    def _max_preceding_frames(self) -> int:
        """Configured preceding-frame count for iter>0 (minimum 1, default 9)."""
        raw_value = self.config.get("observer", {}).get("max_preceding_frames", 9)
        try:
            return max(1, int(raw_value))
        except (TypeError, ValueError):
            return 9

    def run(self, iter_inputs: dict[int, dict]) -> None:
        """Run the full event-centric loop.

        The caller must provide per-iteration prompts and ensure that the
        observer adapter is properly initialized with LiveWorld models.

        Outputs saved per iteration:
        - iter_XXXX/scene_pointcloud.ply: Static scene point cloud
        - iter_XXXX/observer_video.mp4: Observer generated video
        - iter_XXXX/projection_video.mp4: Scene + FG projection visualization
        - iter_XXXX/event_{id}/video.mp4: Event I2V video
        - iter_XXXX/event_{id}/pointcloud.ply: Event foreground point cloud
        - iter_XXXX/event_{id}/observer_view.mp4: Event projected to observer view
        - iter_XXXX/metadata.json: Iteration metadata

        Final outputs:
        - final_video.mp4: Concatenated observer video
        - final_scene_pointcloud.ply: Final static scene point cloud
        """
        observer_cfg = self.config["observer"]
        frames_per_iter = observer_cfg["frames_per_iter"]
        num_iters = len(iter_inputs)
        num_frames = num_iters * frames_per_iter
        fps = observer_cfg.get("fps", 16)
        iteration_plan = compute_iteration_plan(num_frames, frames_per_iter)

        output_root = ensure_dir(self.config["output"]["root"])
        output_cfg = self.config.get("output", {})
        save_event_observer_view = bool(output_cfg.get("save_event_observer_view", False))
        save_event_script = bool(output_cfg.get("save_event_script", False))
        save_observer_prompt = bool(output_cfg.get("save_observer_prompt", False))
        if self._use_preset_storyline:
            # Auto-loaded storyline mode: do not dump prompt/script files.
            save_event_script = False
            save_observer_prompt = False

        events_enabled = self.config["event"]["enabled"]

        # Data for combined video (observer + event videos side by side).
        combined_round_data = []
        event_entity_names = {}  # event_id -> display name

        logger.header("EVENT-CENTRIC PIPELINE")
        logger.info(f"Output: {output_root}")
        logger.info(
            f"Rounds: {len(iteration_plan)}, Frames/round: {frames_per_iter}, "
            f"Total: {num_frames} frames, Events: {'ON' if events_enabled else 'OFF'}"
        )

        # =========================================================================
        # [0] INITIALIZATION: World state from first frame
        # =========================================================================
        with logger.timed("Init world state"):
            self._initialize_world_state()

        # Save initial scene point cloud.
        points_init, colors_init = self.world_state.get_static()
        if output_cfg.get("save_scene_pointcloud", False) and points_init is not None:
            save_pointcloud_ply(output_root / "init_scene_pointcloud.ply", points_init, colors_init)

        # =========================================================================
        # MAIN ITERATION LOOP
        # =========================================================================
        for iter_idx, (start, end, _count) in enumerate(iteration_plan):
            round_dir = ensure_dir(output_root / f"round_{iter_idx}")
            logger.iteration_start(iter_idx, start, end)
            round_event_frames = {}  # event_id -> frames [T, H, W, 3] (no anchor)

            # -------------------------------------------------------------------
            # [1] BUILD OBSERVATION from last frame of previous iteration
            # -------------------------------------------------------------------
            observation = self._build_observation(iter_idx)
            detected_entities_all: list[str] = []
            detected_entities_scene: list[str] = []
            skipped_small_entities: list[dict[str, object]] = []
            min_mask_ratio = 0.05

            if events_enabled:
                # ---------------------------------------------------------------
                # [2] DETECT ENTITIES with segmentation (Qwen + SAM3)
                # ---------------------------------------------------------------
                det_cfg = self.config.get("event", {}).get("detection", {})
                min_mask_ratio = float(det_cfg.get("min_mask_ratio", 0.05))
                _det_label = (
                    "Detect entities (SAM3 only, preset)"
                    if self._preset_entity_names
                    else "Detect entities (Qwen+SAM3)"
                )
                with logger.timed(_det_label):
                    detection_results = self.detector.detect(
                        frame_path=observation.frame_path,
                        frame=observation.frame,
                        frame_index=observation.frame_index,
                        preset_entity_names=self._preset_entity_names,
                    )
                logger.detection([d.name for d in detection_results])
                # ---------------------------------------------------------------
                # [3] REGISTER SCENE-EVENT from all detected entities in frame
                # ---------------------------------------------------------------
                scene_detections = []
                for detection in detection_results:
                    mask_bool = np.asarray(detection.mask) > 0
                    mask_ratio = float(mask_bool.mean()) if mask_bool.size > 0 else 0.0
                    detected_entities_all.append(detection.name)

                    if mask_ratio < min_mask_ratio:
                        skipped_small_entities.append(
                            {
                                "name": detection.name,
                                "mask_ratio": mask_ratio,
                            }
                        )
                        logger.warning(f"  skip '{detection.name}': mask_ratio={mask_ratio:.6f} < min={min_mask_ratio}")
                        continue

                    if "cup" in detection.name:
                        continue
                    scene_detections.append(detection)
                    logger.info(f"  accept '{detection.name}': mask_ratio={mask_ratio:.4f}")

                detected_entities_scene = sorted(
                    {d.name.strip().lower() for d in scene_detections if d.name and d.name.strip()}
                )

                # Remove visible foreground points from static scene so the
                # observer scene projection does not keep stale dynamic objects.
                if observation.frame is not None:
                    removed_static_pts = self._scrub_static_points_with_dynamic_masks(
                        pose_index=observation.frame_index,
                        dynamic_masks=[d.mask for d in scene_detections if d.mask is not None],
                        image_size=observation.frame.shape[:2],
                    )
                    if removed_static_pts > 0:
                        logger.info(f"static scrub: removed {removed_static_pts} visible fg points")

                logger.info(
                    f"  scene_detections={len(scene_detections)}, detected_entities_scene={detected_entities_scene}"
                )
                if detected_entities_scene:
                    # Limitation: scene events are independent and non-interacting.
                    # Each registered event camera represents one dynamic scene chunk.
                    scene_obs = EventObservation(
                        event_id=make_event_id(detected_entities_scene, observation.pose_c2w),
                        frame_path=observation.frame_path,
                        frame_index=observation.frame_index,
                        pose_c2w=observation.pose_c2w,
                        entities=detected_entities_scene,
                        timestamp=observation.timestamp,
                        frame=observation.frame,
                        detections=scene_detections,
                    )
                    self.event_pool.register_event(scene_obs)
                else:
                    logger.warning("  No scene detections passed filter -> event not registered")

                # ---------------------------------------------------------------
                # [4] EVOLVE EVENTS: Generate I2V videos and project to point clouds
                # ---------------------------------------------------------------
                active_agents = self.event_pool.list_active()
                logger.info(f"  active_agents={len(active_agents)}")
                for ai, agent in enumerate(active_agents):
                    eid_short = agent.event_id[:12]
                    ents = ", ".join(agent.state.entities)
                    anchor_shape = (
                        agent.state.current_anchor_frame.shape if agent.state.current_anchor_frame is not None else None
                    )
                    n_prev = len(agent.state.all_generated_frames) if agent.state.all_generated_frames else 0
                    logger.step("Event", f"{ai + 1}/{len(active_agents)} [{eid_short}] {ents}")
                    logger.info(
                        f"  iter_count={agent.state.iteration_count}  anchor_shape={anchor_shape}  prev_frames={n_prev}"
                    )
                    event_dir = ensure_dir(round_dir / f"event_{agent.event_id}")

                    with logger.timed("  I2V prompt (Qwen)"):
                        script = agent.evolve_prompt()
                        system_instruction = script.text
                        evt_scene_text, evt_fg_text = self.event_pool.generate_i2v_prompt(
                            anchor_frame=agent.state.current_anchor_frame,
                            entities=agent.state.entities,
                            iteration=agent.state.iteration_count,
                        )
                        script.text = self._compose_prompt(evt_scene_text, evt_fg_text)

                    if save_event_script:
                        with open(event_dir / "script.txt", "w", encoding="utf-8") as f_script:
                            f_script.write(f"event_id: {script.event_id}\n")
                            f_script.write(f"entities: {', '.join(agent.state.entities)}\n")
                            f_script.write(f"horizon: {script.horizon}\n")
                            f_script.write(f"fps: {script.fps}\n")
                            f_script.write(f"system_instruction: {system_instruction}\n")
                            f_script.write(f"scene_text: {evt_scene_text}\n")
                            f_script.write(f"fg_text: {evt_fg_text}\n")
                            f_script.write(f"composed: {script.text}\n")
                    if agent.state.current_anchor_frame is not None:
                        save_image(event_dir / "anchor_frame.png", agent.state.current_anchor_frame)

                    with logger.timed("  I2V generate (LiveWorld)"):
                        event_video = self._evolve_event_with_observer(
                            agent,
                            script,
                            str(event_dir),
                        )
                    round_event_frames[agent.event_id] = event_video.frames[1:]
                    event_entity_names[agent.event_id] = ents

                    with logger.timed("  Project to pointcloud"):
                        _s3r_model, _s3r_session, _s3r_fed = self._get_shared_stream3r()
                        event_pc, _s3r_fed = self.event_projector.project(
                            event_video=event_video,
                            dynamic_prompts=agent.state.entities,
                            output_dir=str(event_dir),
                            stream3r_model=_s3r_model,
                            sam3_segmenter=self.sam3_segmenter,
                            stream3r_session=_s3r_session,
                            stream3r_frames_fed=_s3r_fed,
                        )
                        self._sync_stream3r_frames_fed(_s3r_fed)
                        self.world_state.update_event(event_pc)

                    if output_cfg.get("save_event_pointclouds", False) and event_pc.points is not None:
                        save_pointcloud_ply(
                            event_dir / "fg_pointcloud.ply",
                            event_pc.points,
                            event_pc.colors,
                        )

            # -------------------------------------------------------------------
            # [5] COMPOSE PROJECTIONS: Scene + event projections for observer
            # -------------------------------------------------------------------
            vae = self.observer_adapter.pipeline.vae
            device = torch.device(self.runtime.device)
            dtype = torch.float16
            self._ensure_vae_on_device()
            obs_dir = ensure_dir(round_dir / "observer")

            with logger.timed("Render projections"):
                scene_proj = self._render_scene_projection(iter_idx)
                event_projs, visible_event_ids, visible_instances = (
                    self._render_event_projections(iter_idx) if events_enabled else ([], set(), {})
                )

            with logger.timed("Compose projections"):
                scene_proj, fg_proj = self.projection_compositor.compose(scene_proj, event_projs, vae, device, dtype)
                if fg_proj is None:
                    logger.warning("  fg projection is None (no visible event projection this round)")
                else:
                    scene_abs = float(scene_proj.detach().abs().mean().item())
                    fg_abs = float(fg_proj.detach().abs().mean().item())
                    fg_ratio = float(fg_abs / max(scene_abs, 1e-8))
                    logger.info(
                        f"  projection strength: scene_abs={scene_abs:.6f}, "
                        f"fg_abs={fg_abs:.6f}, fg/scene={fg_ratio:.4f}"
                    )
                    if not bool(getattr(self.observer_adapter.pipeline, "use_fg_proj", False)):
                        logger.warning("  observer pipeline use_fg_proj=False, fg projection is ignored")
                overlay_rgb = None  # pixel-space composite for visualization
                if fg_proj is not None and self.config["event"].get("projection", {}).get("overlay_fg_on_scene", False):
                    scene_proj, overlay_rgb = overlay_fg_on_scene(scene_proj, fg_proj, vae, device, dtype)

            with logger.timed("Save projection videos"):
                proj_vis_frames = None
                if output_cfg.get("save_projection_video", True):
                    if overlay_rgb is not None:
                        # Use pixel-space result directly to avoid VAE decode artifacts
                        fg_rgb_vis = decode_latent_to_rgb(fg_proj, vae, device, dtype)
                        proj_vis_frames = np.concatenate(
                            [overlay_rgb, fg_rgb_vis], axis=2
                        )  # side-by-side [T, H, 2W, 3]
                    elif fg_proj is not None:
                        vis_proj = torch.cat([scene_proj, fg_proj], dim=0)
                        proj_vis_frames = visualize_projection_latent(vis_proj, vae, device, dtype)
                    else:
                        proj_vis_frames = visualize_projection_latent(scene_proj, vae, device, dtype)
                    save_video(obs_dir / "projection.mp4", proj_vis_frames, fps=fps)

                if events_enabled and save_event_observer_view:
                    event_pcs = self.world_state.list_event_pointclouds()
                    for event_pc, event_proj in zip(event_pcs, event_projs, strict=False):
                        event_dir = ensure_dir(round_dir / f"event_{event_pc.event_id}")
                        event_rgb_frames = decode_latent_to_rgb(event_proj, vae, device, dtype)
                        save_video(event_dir / "fg_projection.mp4", event_rgb_frames, fps=fps)

            # -------------------------------------------------------------------
            # [6] OBSERVER: Prepare inputs and run LiveWorld diffusion
            # -------------------------------------------------------------------
            iter_cfg = iter_inputs[str(iter_idx)]
            iter_prompt = self._compose_prompt(
                iter_cfg["scene_text"],
                iter_cfg["fg_text"],
            )
            if save_observer_prompt:
                with open(obs_dir / "prompt.txt", "w", encoding="utf-8") as f_prompt:
                    f_prompt.write(f"scene_text: {iter_cfg['scene_text']}\n")
                    f_prompt.write(f"fg_text: {iter_cfg['fg_text']}\n")
                    f_prompt.write(f"composed: {iter_prompt}\n")

            target_frame_indices = list(
                range(
                    self.geometry.start_frame + 1 + iter_idx * frames_per_iter,
                    self.geometry.start_frame + 1 + (iter_idx + 1) * frames_per_iter,
                )
            )

            with logger.timed("Encode preceding frames"):
                prec_scene_pixels, prec_fg_pixels = self._get_preceding_projection_frames(iter_idx)
                preceding_scene_proj = self._encode_pixel_frames_to_proj_latent(prec_scene_pixels)
                preceding_fg_proj = self._encode_pixel_frames_to_proj_latent(prec_fg_pixels)

            with logger.timed("Get reference frames"):
                max_refs = observer_cfg.get("max_reference_frames", 7)
                # no_ref_before_round: skip all scene refs for rounds < N (first outbound trip).
                no_ref_before = observer_cfg.get("no_ref_before_round", 0)
                use_pose_based_ref = observer_cfg.get("pose_based_ref", False)
                pose_ref_count = observer_cfg.get("pose_based_ref_count", 7)
                if no_ref_before > 0 and iter_idx < no_ref_before:
                    reference_frames = None
                    logger.info(f"  no_ref_before_round={no_ref_before}: skip refs for round {iter_idx}")
                elif use_pose_based_ref:
                    reference_frames = self._get_reference_frames_pose_based(
                        iter_idx,
                        target_frame_indices,
                        num_refs=pose_ref_count,
                        diag_path=obs_dir / "ref_diagnostics.txt",
                    )
                else:
                    reference_frames = self._get_reference_frames(
                        iter_idx,
                        target_frame_indices,
                        max_reference_frames=max_refs,
                        diag_path=obs_dir / "ref_diagnostics.txt",
                    )
                    # Optionally discard if projection has too much unseen area.
                    # Disabled when no_ref_before_round is active (refs are already
                    # skipped for early rounds; later rounds should always keep them).
                    ref_discard_threshold = observer_cfg.get("ref_discard_black_ratio")
                    if ref_discard_threshold is not None and reference_frames is not None and no_ref_before <= 0:
                        black_ratio = self._compute_last_frame_black_ratio(iter_idx)
                        diag_file = obs_dir / "ref_diagnostics.txt"
                        with open(diag_file, "a") as f:
                            f.write("\n--- ref_discard_black_ratio check ---\n")
                            f.write(f"  threshold: {ref_discard_threshold}, black_ratio: {black_ratio:.4f}\n")
                            if black_ratio > ref_discard_threshold:
                                f.write("  => DISCARDED all refs (black_ratio > threshold)\n")
                            else:
                                f.write("  => KEPT refs (black_ratio <= threshold)\n")
                        if black_ratio > ref_discard_threshold:
                            reference_frames = None
            n_scene_refs = len(reference_frames) if reference_frames is not None else 0
            logger.info(f"scene refs: {n_scene_refs}")

            if reference_frames is not None:
                for i, ref in enumerate(reference_frames):
                    save_image(obs_dir / f"scene_ref_{i:02d}.png", ref)

            with logger.timed("Get instance references"):
                instance_reference_frames, inst_ref_map = self._get_instance_reference_frames(
                    iter_idx, visible_event_ids, visible_instances=visible_instances
                )
            n_inst_refs = len(instance_reference_frames) if instance_reference_frames is not None else 0
            logger.info(f"instance refs: {n_inst_refs}")

            if instance_reference_frames is not None:
                for i, ref in enumerate(instance_reference_frames):
                    save_image(obs_dir / f"inst_ref_{i:02d}.png", ref)
            for eid, ref_img in inst_ref_map.items():
                event_dir = ensure_dir(round_dir / f"event_{eid}")
                save_image(event_dir / "inst_ref.png", ref_img)

            observer_output = self._run_observer_diffusion(
                iter_idx=iter_idx,
                iter_prompt=iter_prompt,
                frames_per_iter=frames_per_iter,
                scene_proj=scene_proj,
                fg_proj=fg_proj,
                preceding_scene_proj=preceding_scene_proj,
                preceding_fg_proj=preceding_fg_proj,
                reference_frames=reference_frames,
                instance_reference_frames=instance_reference_frames,
                observer_cfg=observer_cfg,
            )

            self._accumulate_projections(scene_proj, fg_proj)

            # -------------------------------------------------------------------
            # [7] STORE FRAMES and convert to numpy
            # -------------------------------------------------------------------
            generated_frames = observer_output.frames
            frames_np_list = []
            if isinstance(generated_frames, torch.Tensor) and generated_frames.dim() == 4:
                frames_np = generated_frames.cpu().numpy()
                if frames_np.dtype != np.uint8:
                    frames_np = frames_np.clip(0, 255).astype(np.uint8)
                for f in frames_np:
                    self.geometry.all_generated_frames.append(f)
                    frames_np_list.append(f)

            if frames_np_list:
                save_video(obs_dir / "video.mp4", frames_np_list, fps=fps)

            combined_round_data.append(
                {
                    "observer_frames": list(frames_np_list),
                    "events": round_event_frames,
                }
            )

            # -------------------------------------------------------------------
            # [8] POINT CLOUD UPDATE: Incremental update via Stream3R
            # -------------------------------------------------------------------
            with logger.timed("Point cloud update (Stream3R)"):
                if frames_np_list:
                    frames_array = np.stack(frames_np_list, axis=0)  # [T, H, W, 3]
                    frame_indices = list(target_frame_indices)

                    current_points, current_colors = self.world_state.get_static()
                    # Collect entity prompts from all active events so that
                    # dynamic mask detection also masks foreground from earlier
                    # events (Qwen may fail to re-detect them in later rounds).
                    _active_entity_prompts = sorted(
                        {
                            str(ent).strip()
                            for agent in self.event_pool.list_active()
                            for ent in (agent.state.entities or [])
                            if str(ent).strip()
                        }
                    )
                    new_points, new_colors = self.pointcloud_handler.update(
                        iter_idx=iter_idx,
                        frames=frames_array,
                        frame_indices=frame_indices,
                        state_points=current_points,
                        state_colors=current_colors,
                        options=self._backbone_options,
                        rgb_frames=frames_array,
                        extra_entity_prompts=_active_entity_prompts or None,
                    )

                    if new_points is not None and len(new_points) > 0:
                        handler = self.pointcloud_handler
                        if handler.points_world is None:
                            raise RuntimeError("Handler points_world is None after update()")
                        self.world_state.update_static(handler.points_world, handler.colors)

                    # Store per-frame dynamic masks (keyed by gen frame index).
                    handler = self.pointcloud_handler
                    if hasattr(handler, "last_dynamic_masks") and handler.last_dynamic_masks is not None:
                        gen_frame_start = len(self.geometry.all_generated_frames) - len(frames_np_list)
                        n_masks = len(handler.last_dynamic_masks)
                        n_stored = 0
                        for fi, mask in enumerate(handler.last_dynamic_masks):
                            if fi < len(frames_np_list):
                                self.geometry.frame_dynamic_masks[gen_frame_start + fi] = mask
                                n_stored += 1
                        logger.info(
                            f"dynamic masks: {n_masks} available, {n_stored} stored (frames {gen_frame_start}..{gen_frame_start + n_stored - 1})"
                        )
                    else:
                        logger.info("dynamic masks: NONE from handler")

            with logger.timed("Update frame visibility"):
                output_start = iter_idx * frames_per_iter
                self._update_frame_visibility(iter_idx, target_frame_indices, output_start)

            points_iter, colors_iter = self.world_state.get_static()
            if output_cfg.get("save_scene_pointcloud", False) and points_iter is not None:
                save_pointcloud_ply(round_dir / "scene_pointcloud.ply", points_iter, colors_iter)

            # -------------------------------------------------------------------
            # [8.5] INTERMEDIATE EVENT DETECTION
            # -------------------------------------------------------------------
            intermediate_cfg = self.config["event"].get("intermediate_detection", {})
            intermediate_enabled = (
                events_enabled
                and bool(intermediate_cfg.get("enabled", False))
                and self.intermediate_detector is not None
            )
            new_intermediate_agents: list[tuple] = []
            intermediate_candidate_frames: list[int] = []
            intermediate_coverages: list[tuple[int, float]] = []

            if intermediate_enabled and len(self.event_pool.list_active()) > 0:
                with logger.timed("Intermediate event detection"):
                    # Step 1: Compute per-frame coverage of existing events.
                    intermediate_coverages = self._compute_union_event_coverage_per_frame(iter_idx)
                    cov_threshold = float(intermediate_cfg.get("coverage_threshold", 0.1))
                    intermediate_candidate_frames = [
                        local_idx for local_idx, cov in intermediate_coverages if cov < cov_threshold
                    ]

                    if intermediate_candidate_frames:
                        logger.info(
                            f"  Intermediate detection: {len(intermediate_candidate_frames)} "
                            f"candidate frames (coverage < {cov_threshold:.1%})"
                        )
                        # Step 2: Detect and register intermediate events.
                        new_intermediate_agents = self._detect_intermediate_events(
                            iter_idx,
                            intermediate_candidate_frames,
                            round_dir,
                        )
                        if new_intermediate_agents:
                            logger.info(f"  Registered {len(new_intermediate_agents)} intermediate event(s)")
                    else:
                        logger.info("  Intermediate detection: no uncovered frames")

            # -------------------------------------------------------------------
            # [8.6] INTERMEDIATE EVENT CATCH-UP SYNC
            # -------------------------------------------------------------------
            if new_intermediate_agents:
                with logger.timed("Intermediate event catch-up sync"):
                    for agent_item, anchor_local in new_intermediate_agents:
                        eid_short = agent_item.event_id[:12]
                        ents = ", ".join(agent_item.state.entities)
                        logger.step(
                            "Intermediate Event",
                            f"[{eid_short}] {ents} @ frame {anchor_local}",
                        )
                        result = self._catchup_intermediate_event(
                            agent_item,
                            anchor_local,
                            iter_idx,
                            round_dir,
                        )
                        if result is not None:
                            catchup_video, catchup_pc = result
                            round_event_frames[agent_item.event_id] = catchup_video.frames[1:]
                            event_entity_names[agent_item.event_id] = ents

            # -------------------------------------------------------------------
            # [9] SAVE METADATA and round summary
            # -------------------------------------------------------------------
            detected_entities = detected_entities_scene
            active_events = self.event_pool.list_active()

            save_json(
                round_dir / "metadata.json",
                {
                    "round_idx": iter_idx,
                    "frame_range": [start, end],
                    "detection_min_mask_ratio": min_mask_ratio,
                    "detected_entities_all": detected_entities_all,
                    "entities": detected_entities,
                    "skipped_small_entities": skipped_small_entities,
                    "events": [agent.state.entities for agent in active_events],
                    "num_scene_points": len(points_iter) if points_iter is not None else 0,
                    "num_scene_refs": n_scene_refs,
                    "num_inst_refs": n_inst_refs,
                    "intermediate_detection": {
                        "enabled": intermediate_enabled,
                        "n_candidate_frames": len(intermediate_candidate_frames),
                        "n_new_events": len(new_intermediate_agents),
                        "new_event_anchors": [(a.event_id[:12], idx) for a, idx in new_intermediate_agents],
                    },
                },
            )

            logger.round_end(
                iter_idx,
                entities=", ".join(detected_entities) if detected_entities else "(none)",
                scene_refs=n_scene_refs,
                inst_refs=n_inst_refs,
                points=f"{len(points_iter):,}" if points_iter is not None else "0",
            )

        # =========================================================================
        # Final outputs
        # =========================================================================
        with logger.timed("Save final outputs"):
            if self.geometry.all_generated_frames:
                save_video(output_root / "final_video.mp4", self.geometry.all_generated_frames, fps=fps)

            if (
                output_cfg.get("save_combined_video", True)
                and combined_round_data
                and any(rd["events"] for rd in combined_round_data)
            ):
                self._save_combined_video(
                    output_root / "final_combined.mp4",
                    combined_round_data,
                    event_entity_names,
                    fps,
                )

            final_points, final_colors = self.world_state.get_static()
            if output_cfg.get("save_final_pointcloud", False) and final_points is not None:
                save_pointcloud_ply(output_root / "final_scene_pointcloud.ply", final_points, final_colors)

            if output_cfg.get("save_final_world_event_camera_ply", True):
                self._save_final_world_event_camera_ply(
                    output_root / "final_world_event_camera.ply",
                    output_cfg=output_cfg,
                )

        logger.final_summary(
            output_root=str(output_root),
            total_frames=len(self.geometry.all_generated_frames),
            total_points=len(final_points) if final_points is not None else 0,
        )

    @staticmethod
    def _ensure_uint8_colors(colors: np.ndarray | None, n_points: int) -> np.ndarray:
        """Normalize color array to uint8 RGB with length n_points."""
        if n_points <= 0:
            return np.zeros((0, 3), dtype=np.uint8)

        if colors is None:
            return np.full((n_points, 3), 255, dtype=np.uint8)

        arr = np.asarray(colors)
        if arr.ndim != 2 or arr.shape[0] != n_points:
            raise ValueError(f"color shape mismatch: expected ({n_points}, 3), got {arr.shape}")

        if arr.shape[1] > 3:
            arr = arr[:, :3]
        if arr.shape[1] < 3:
            pad = np.zeros((n_points, 3 - arr.shape[1]), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=1)

        if np.issubdtype(arr.dtype, np.floating):
            max_val = float(np.nanmax(arr)) if arr.size > 0 else 0.0
            if max_val <= 1.0:
                arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    @staticmethod
    def _sample_line_points(p0: np.ndarray, p1: np.ndarray, num: int) -> np.ndarray:
        """Sample equally spaced points on a 3D line segment."""
        if num <= 1:
            return p0.reshape(1, 3).astype(np.float32)
        t = np.linspace(0.0, 1.0, num=num, dtype=np.float32)[:, None]
        pts = (1.0 - t) * p0.reshape(1, 3) + t * p1.reshape(1, 3)
        return pts.astype(np.float32)

    @staticmethod
    def _parse_rgb_color(value: object, default: tuple[int, int, int]) -> np.ndarray:
        """Parse RGB color config into uint8 [3]."""
        if value is None:
            return np.array(default, dtype=np.uint8)

        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",")]
            if len(parts) < 3:
                raise ValueError(f"RGB color string must have at least 3 parts, got: {value!r}")
            rgb = [int(float(parts[i])) for i in range(3)]
            return np.clip(np.asarray(rgb, dtype=np.int32), 0, 255).astype(np.uint8)

        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value).reshape(-1)
            if arr.size >= 3:
                return np.clip(arr[:3].astype(np.int32), 0, 255).astype(np.uint8)
        return np.array(default, dtype=np.uint8)

    def _find_nearest_pose_index(self, pose_c2w: np.ndarray) -> int:
        """Find nearest geometry pose index by camera center distance."""
        poses = self.geometry.poses_c2w
        if poses is None or len(poses) == 0:
            raise RuntimeError("poses_c2w not initialized")
        centers = poses[:, :3, 3]
        target = np.asarray(pose_c2w, dtype=np.float32)[:3, 3]
        dists = np.linalg.norm(centers - target[None], axis=1)
        return int(np.argmin(dists))

    def _get_intrinsics_for_pose(self, pose_index: int) -> tuple[np.ndarray, tuple[int, int]]:
        """Return per-pose intrinsics matrix and its image size (H, W)."""
        intrinsics = self.geometry.intrinsics
        if intrinsics is None:
            raise RuntimeError("intrinsics not initialized")

        intrinsics_size = self.geometry.intrinsics_size
        if intrinsics_size is None:
            raise RuntimeError("intrinsics_size not initialized")
        image_size = (int(intrinsics_size[0]), int(intrinsics_size[1]))

        if intrinsics.ndim == 2:
            K = intrinsics.copy().astype(np.float32)
        else:
            idx = min(max(int(pose_index), 0), len(intrinsics) - 1)
            K = intrinsics[idx].copy().astype(np.float32)

        if K.shape != (3, 3):
            flat = K.reshape(-1)
            fx, fy, cx, cy = flat[:4]
            K = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
        return K, image_size

    def _build_camera_motion_points(
        self,
        pose_indices: list[int],
        frustum_depth: float,
        frustum_color: np.ndarray | None = None,
        center_color: np.ndarray | None = None,
        trajectory_color: np.ndarray | None = None,
        frustum_edge_samples: int = 8,
        traj_edge_samples: int = 12,
        connect_trajectory: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build point-cloud markers for camera frustums and trajectory."""
        poses = self.geometry.poses_c2w
        if poses is None or len(pose_indices) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        if frustum_color is None:
            frustum_color = np.array([255, 215, 0], dtype=np.uint8)
        else:
            frustum_color = np.asarray(frustum_color, dtype=np.uint8).reshape(3)
        if center_color is None:
            center_color = np.array([0, 255, 255], dtype=np.uint8)
        else:
            center_color = np.asarray(center_color, dtype=np.uint8).reshape(3)
        if trajectory_color is None:
            trajectory_color = np.array([255, 80, 80], dtype=np.uint8)
        else:
            trajectory_color = np.asarray(trajectory_color, dtype=np.uint8).reshape(3)

        pts_chunks: list[np.ndarray] = []
        col_chunks: list[np.ndarray] = []
        camera_centers: list[np.ndarray] = []

        for pose_index in pose_indices:
            idx = min(max(int(pose_index), 0), len(poses) - 1)
            c2w = poses[idx].astype(np.float32)
            K, (img_h, img_w) = self._get_intrinsics_for_pose(idx)

            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
            if abs(fx) < 1e-6 or abs(fy) < 1e-6:
                continue

            cam_center = c2w[:3, 3].astype(np.float32)
            camera_centers.append(cam_center)
            pts_chunks.append(cam_center.reshape(1, 3))
            col_chunks.append(np.tile(center_color[None], (1, 1)))

            corners_px = [
                (0.0, 0.0),
                (float(img_w - 1), 0.0),
                (float(img_w - 1), float(img_h - 1)),
                (0.0, float(img_h - 1)),
            ]

            corners_cam = []
            for u, v in corners_px:
                x = (u - cx) / fx * frustum_depth
                y = (v - cy) / fy * frustum_depth
                corners_cam.append(np.array([x, y, frustum_depth], dtype=np.float32))
            corners_cam = np.stack(corners_cam, axis=0)  # [4, 3]
            corners_world = (c2w[:3, :3] @ corners_cam.T).T + cam_center[None]

            frustum_edges = []
            for i in range(4):
                frustum_edges.append((cam_center, corners_world[i]))
                frustum_edges.append((corners_world[i], corners_world[(i + 1) % 4]))

            for p0, p1 in frustum_edges:
                seg_pts = self._sample_line_points(p0, p1, num=frustum_edge_samples)
                pts_chunks.append(seg_pts)
                col_chunks.append(np.tile(frustum_color[None], (len(seg_pts), 1)))

        if connect_trajectory:
            for i in range(len(camera_centers) - 1):
                seg_pts = self._sample_line_points(camera_centers[i], camera_centers[i + 1], num=traj_edge_samples)
                pts_chunks.append(seg_pts)
                col_chunks.append(np.tile(trajectory_color[None], (len(seg_pts), 1)))

        if not pts_chunks:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        pts = np.concatenate(pts_chunks, axis=0).astype(np.float32)
        cols = np.concatenate(col_chunks, axis=0).astype(np.uint8)
        return pts, cols

    def _save_final_world_event_camera_ply(
        self,
        out_path: Path,
        output_cfg: dict,
    ) -> dict[str, int] | None:
        """Save merged static scene + event foreground + camera motion to one PLY."""
        scene_points, scene_colors = self.world_state.get_static()
        if scene_points is None or len(scene_points) == 0:
            return None

        scene_points = np.asarray(scene_points, dtype=np.float32)
        scene_colors_u8 = self._ensure_uint8_colors(scene_colors, len(scene_points))

        merged_points = [scene_points]
        merged_colors = [scene_colors_u8]

        # 1) Event foreground point clouds (anchor camera -> world).
        event_point_count = 0
        event_pcs = self.world_state.list_event_pointclouds()
        if event_pcs:
            for event_pc in event_pcs:
                if event_pc.points is None or len(event_pc.points) == 0:
                    continue
                pts_world = transform_points(event_pc.points, event_pc.anchor_pose_c2w)
                cols_u8 = self._ensure_uint8_colors(event_pc.colors, len(pts_world))
                merged_points.append(np.asarray(pts_world, dtype=np.float32))
                merged_colors.append(cols_u8)
                event_point_count += len(pts_world)

        # 2) Observer camera frustum motion markers.
        poses = self.geometry.poses_c2w
        observer_camera_point_count = 0
        event_camera_point_count = 0
        num_observer_camera_poses = 0
        num_event_cameras = 0
        total_camera_points = 0

        scene_extent = np.ptp(scene_points, axis=0)
        diag = float(np.linalg.norm(scene_extent))
        observer_frustum_depth = float(
            np.clip(diag * float(output_cfg.get("camera_frustum_depth_scale", 0.02)), 0.03, 1.0)
        )

        if poses is not None and len(poses) > 0:
            start_idx = int(self.geometry.start_frame)
            end_idx = min(len(poses), start_idx + 1 + len(self.geometry.all_generated_frames))
            pose_indices = list(range(start_idx, end_idx))

            stride = max(1, int(output_cfg.get("camera_ply_pose_stride", 1)))
            if len(pose_indices) > 2 and stride > 1:
                sampled = pose_indices[::stride]
                if sampled[-1] != pose_indices[-1]:
                    sampled.append(pose_indices[-1])
                pose_indices = sampled

            num_observer_camera_poses = len(pose_indices)
            observer_frustum_color = self._parse_rgb_color(
                output_cfg.get("observer_camera_frustum_color"),
                (255, 215, 0),
            )
            observer_center_color = self._parse_rgb_color(
                output_cfg.get("observer_camera_center_color"),
                (0, 255, 255),
            )
            observer_traj_color = self._parse_rgb_color(
                output_cfg.get("observer_camera_traj_color"),
                (255, 80, 80),
            )

            cam_pts, cam_cols = self._build_camera_motion_points(
                pose_indices=pose_indices,
                frustum_depth=observer_frustum_depth,
                frustum_color=observer_frustum_color,
                center_color=observer_center_color,
                trajectory_color=observer_traj_color,
                frustum_edge_samples=max(2, int(output_cfg.get("camera_frustum_edge_samples", 8))),
                traj_edge_samples=max(2, int(output_cfg.get("camera_traj_edge_samples", 12))),
                connect_trajectory=True,
            )
            if len(cam_pts) > 0:
                merged_points.append(cam_pts)
                merged_colors.append(cam_cols)
                observer_camera_point_count = len(cam_pts)

        # 3) Event camera frustums (event anchor poses), with separate colors.
        if poses is not None and len(poses) > 0 and len(event_pcs) > 0:
            event_pose_indices = set()
            for event_pc in event_pcs:
                if event_pc.anchor_pose_c2w is None:
                    continue
                event_pose_indices.add(self._find_nearest_pose_index(event_pc.anchor_pose_c2w))

            event_pose_indices_sorted = sorted(event_pose_indices)
            num_event_cameras = len(event_pose_indices_sorted)
            if num_event_cameras > 0:
                event_frustum_depth = float(
                    np.clip(
                        diag * float(output_cfg.get("event_camera_frustum_depth_scale", 0.02)),
                        0.03,
                        1.0,
                    )
                )
                event_frustum_color = self._parse_rgb_color(
                    output_cfg.get("event_camera_frustum_color"),
                    (110, 255, 110),
                )
                event_center_color = self._parse_rgb_color(
                    output_cfg.get("event_camera_center_color"),
                    (0, 180, 0),
                )
                event_cam_pts, event_cam_cols = self._build_camera_motion_points(
                    pose_indices=event_pose_indices_sorted,
                    frustum_depth=event_frustum_depth,
                    frustum_color=event_frustum_color,
                    center_color=event_center_color,
                    trajectory_color=event_frustum_color,
                    frustum_edge_samples=max(
                        2,
                        int(
                            output_cfg.get(
                                "event_camera_frustum_edge_samples",
                                output_cfg.get("camera_frustum_edge_samples", 8),
                            )
                        ),
                    ),
                    traj_edge_samples=2,
                    connect_trajectory=False,
                )
                if len(event_cam_pts) > 0:
                    merged_points.append(event_cam_pts)
                    merged_colors.append(event_cam_cols)
                    event_camera_point_count = len(event_cam_pts)

        total_camera_points = observer_camera_point_count + event_camera_point_count

        points_all = np.concatenate(merged_points, axis=0)
        colors_all = np.concatenate(merged_colors, axis=0)
        save_pointcloud_ply(out_path, points_all, colors_all)

        return {
            "scene_points": len(scene_points),
            "event_points": int(event_point_count),
            "camera_points": int(total_camera_points),
            "observer_camera_points": int(observer_camera_point_count),
            "event_camera_points": int(event_camera_point_count),
            "num_camera_poses": int(num_observer_camera_poses),
            "num_observer_camera_poses": int(num_observer_camera_poses),
            "num_event_cameras": int(num_event_cameras),
            "total_points": len(points_all),
        }

    def _save_combined_video(
        self,
        path: Path,
        round_data: list[dict],
        event_names: dict[str, str],
        fps: float,
    ) -> None:
        """Create a combined video showing observer and event videos side by side.

        For each round, frames are aligned temporally: frame i of observer
        is shown next to frame i of each event video.  Events that are not
        active in a given round appear as black panels.

        Layout per frame:  [Observer | Event_1 | Event_2 | ...]

        Args:
            path: Output path for the combined video.
            round_data: Per-round list, each dict has:
                - 'observer_frames': list of [H, W, 3] uint8
                - 'events': dict of event_id -> ndarray [T, H, W, 3] uint8
            event_names: Mapping event_id -> display name (entity names).
            fps: Frames per second.
        """
        # 1. Collect all event IDs in order of first appearance.
        all_event_ids = []
        seen_ids = set()
        for rd in round_data:
            for eid in rd["events"]:
                if eid not in seen_ids:
                    all_event_ids.append(eid)
                    seen_ids.add(eid)

        if not round_data or not round_data[0].get("observer_frames"):
            return

        # 2. Get panel size from first observer frame.
        sample = round_data[0]["observer_frames"][0]
        panel_h, panel_w = sample.shape[:2]

        # 3. Grid dimensions.
        n_cols = 1 + len(all_event_ids)
        total_h = panel_h
        total_w = panel_w * n_cols

        # 4. Build all frames.
        combined_frames = []
        for _round_idx, rd in enumerate(round_data):
            obs_frames = rd["observer_frames"]
            n_frames = len(obs_frames)

            for fi in range(n_frames):
                canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

                # Observer panel.
                if fi < len(obs_frames):
                    canvas[0:panel_h, 0:panel_w] = obs_frames[fi]

                # Event panels.
                for ei, eid in enumerate(all_event_ids):
                    col_x = (1 + ei) * panel_w

                    evt_data = rd["events"].get(eid)
                    if evt_data is not None and fi < len(evt_data):
                        frame = evt_data[fi]
                        fh, fw = frame.shape[:2]
                        if (fh, fw) != (panel_h, panel_w):
                            frame = cv2.resize(frame, (panel_w, panel_h))
                        canvas[0:panel_h, col_x : col_x + panel_w] = frame

                combined_frames.append(canvas)

        if combined_frames:
            save_video(path, combined_frames, fps=fps)
            logger.saved(
                "final_combined.mp4",
                f"{len(combined_frames)} frames, {n_cols} panels (observer + {len(all_event_ids)} events)",
            )

    def _build_observation(self, iter_idx: int) -> EventObservation:
        """Build an observation from the last frame of the previous iteration.

        For iter_idx == 0, uses the first frame from world initialization.
        For iter_idx > 0, uses the last generated frame from the previous iteration.

        Returns:
            EventObservation with frame as numpy array for detection.
        """
        config = self.config
        observer_cfg = config["observer"]
        frames_per_iter = observer_cfg["frames_per_iter"]

        if iter_idx == 0:
            # Use the first frame loaded during world state initialization.
            if self.geometry.first_frame is None:
                raise RuntimeError("First frame not initialized; call _initialize_world_state first")

            frame_pil = self.geometry.first_frame
            frame_np = np.array(frame_pil)
            frame_index = 0
            pose_c2w = self.geometry.poses_c2w[frame_index]

            # Use first frame image path from config (already resolved).
            frame_path = config.get("first_frame_image", "")
        else:
            # Use the last frame from previous iteration.
            all_frames = self.geometry.all_generated_frames
            if not all_frames:
                raise RuntimeError(f"No generated frames available for iter_idx={iter_idx}")

            # Each iteration generates frames_per_iter frames.
            # The last frame of iter_idx-1 is at index (iter_idx * frames_per_iter - 1).
            prev_frame_count = iter_idx * frames_per_iter
            if len(all_frames) < prev_frame_count:
                raise RuntimeError(f"Expected at least {prev_frame_count} frames, got {len(all_frames)}")
            frame_np = all_frames[prev_frame_count - 1]
            frame_index = prev_frame_count - 1
            pose_c2w = self.geometry.poses_c2w[frame_index]

            # Generated frames don't have a file path; use empty string.
            frame_path = ""

        # Create a minimal observation with entities empty (will be filled by detector).
        return EventObservation(
            event_id="",  # Will be computed after detection.
            frame_path=frame_path,
            frame_index=frame_index,
            pose_c2w=pose_c2w,
            entities=[],
            timestamp=float(frame_index),
            frame=frame_np,
        )

    def _clamp_pose_index(self, pose_index: int) -> int:
        poses = self.geometry.poses_c2w
        if poses is None or len(poses) == 0:
            raise RuntimeError("poses_c2w not initialized")
        return int(np.clip(pose_index, 0, len(poses) - 1))

    def _get_scaled_intrinsics_for_pose(self, pose_index: int) -> np.ndarray:
        intrinsics = self.geometry.intrinsics
        intrinsics_size = self.geometry.intrinsics_size
        if intrinsics is None or intrinsics_size is None:
            raise RuntimeError("intrinsics not initialized")

        observer_cfg = self.config["observer"]
        out_h = observer_cfg.get("height", 480)
        out_w = observer_cfg.get("width", 832)
        proc_h, proc_w = intrinsics_size

        pose_index = self._clamp_pose_index(pose_index)
        if intrinsics.ndim == 2:
            K_raw = intrinsics.copy().astype(np.float32)
        else:
            K_raw = intrinsics[pose_index].copy().astype(np.float32)

        if K_raw.shape != (3, 3):
            flat = K_raw.reshape(-1)
            fx, fy, cx, cy = flat[:4]
            K_raw = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

        return scale_intrinsics(K_raw, (proc_h, proc_w), (out_h, out_w))

    def _event_scene_ref_mask_dilate(self) -> int:
        """Dilation radius (pixels) for foreground removal in event scene refs."""
        evolution_cfg = self.config.get("event", {}).get("evolution", {})
        raw = evolution_cfg.get("scene_ref_mask_dilate", 3)
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 3

    def _event_max_scene_refs(self) -> int | None:
        """Max scene refs for the event I2V model. None = no limit."""
        evolution_cfg = self.config.get("event", {}).get("evolution", {})
        raw = evolution_cfg.get("max_scene_refs", None)
        if raw is None:
            return None
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _dilate_binary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
        """Dilate a binary mask with an elliptical kernel."""
        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
        if radius <= 0:
            return mask_u8
        k = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(mask_u8, kernel, iterations=1)

    def _event_use_preceding_fg_proj(self) -> bool:
        """Whether event generation should use preceding_fg_proj conditioning."""
        evolution_cfg = self.config.get("event", {}).get("evolution", {})
        return bool(evolution_cfg.get("use_preceding_fg_proj", False))

    def _event_use_preceding_scene_proj(self) -> bool:
        """Whether event generation should use preceding_scene_proj conditioning."""
        evolution_cfg = self.config.get("event", {}).get("evolution", {})
        return bool(evolution_cfg.get("use_preceding_scene_proj", True))

    def _event_use_target_scene_proj(self) -> bool:
        """Whether event generation should use target_scene_proj conditioning."""
        evolution_cfg = self.config.get("event", {}).get("evolution", {})
        return bool(evolution_cfg.get("use_target_scene_proj", True))

    def _event_target_scene_proj_frames(self) -> list[int] | None:
        """Return the list of latent frame indices to keep target_scene_proj for.

        Config value ``target_scene_proj_frames``:
        - null / absent  -> None  (keep all frames)
        - "0,64"         -> [0, 64]  (only these frames keep projection, rest zeroed)
        """
        evolution_cfg = self.config.get("event", {}).get("evolution", {})
        raw = evolution_cfg.get("target_scene_proj_frames", None)
        if raw is None:
            return None
        if isinstance(raw, str):
            return [int(x.strip()) for x in raw.split(",") if x.strip()]
        if isinstance(raw, (list, tuple)):
            return [int(x) for x in raw]
        return None

    def _segment_event_entities_mask(
        self,
        frame: np.ndarray,
        entities: list[str],
    ) -> np.ndarray | None:
        """Segment a union foreground mask for event entities on one frame."""
        if frame is None or len(entities) == 0:
            return None
        seg_result = self.sam3_segmenter.segment(
            video_path=[Image.fromarray(frame)],
            prompts=entities,
            frame_index=0,
            expected_frames=1,
        )
        if seg_result is None or getattr(seg_result, "size", 0) == 0:
            return None
        mask = seg_result[0] if seg_result.ndim == 3 else seg_result
        return (np.asarray(mask) > 0).astype(np.uint8)

    def _scrub_static_points_with_dynamic_masks(
        self,
        pose_index: int,
        dynamic_masks: list[np.ndarray],
        image_size: tuple[int, int],
    ) -> int:
        """Remove front-most static points that fall inside dynamic masks."""
        if not dynamic_masks:
            return 0

        points, colors = self.world_state.get_static()
        if points is None or colors is None or len(points) == 0:
            return 0
        if len(colors) != len(points):
            raise RuntimeError(f"static color/point mismatch: len(colors)={len(colors)} len(points)={len(points)}")

        H, W = int(image_size[0]), int(image_size[1])
        if H <= 0 or W <= 0:
            return 0

        fg_mask = np.zeros((H, W), dtype=bool)
        for mask in dynamic_masks:
            if mask is None:
                continue
            m = np.asarray(mask)
            if m.ndim == 3:
                m = m[..., 0]
            if m.ndim != 2:
                continue
            if m.shape != (H, W):
                m = cv2.resize(
                    m.astype(np.uint8),
                    (W, H),
                    interpolation=cv2.INTER_NEAREST,
                )
            fg_mask |= m > 0

        # Dilate to aggressively remove foreground edges from static PC.
        if fg_mask.any():
            _kern = np.ones((3, 3), dtype=np.uint8)
            fg_mask = cv2.dilate(
                fg_mask.astype(np.uint8),
                _kern,
                iterations=5,
            ).astype(bool)

        if not np.any(fg_mask):
            return 0

        pose_idx = self._clamp_pose_index(pose_index)
        poses_c2w = self.geometry.poses_c2w
        if poses_c2w is None:
            return 0
        c2w = poses_c2w[pose_idx]

        K = self._get_scaled_intrinsics_for_pose(pose_idx)
        observer_cfg = self.config["observer"]
        obs_h = int(observer_cfg.get("height", H))
        obs_w = int(observer_cfg.get("width", W))
        if (obs_h, obs_w) != (H, W):
            K = scale_intrinsics(K, (obs_h, obs_w), (H, W))

        points_np = np.asarray(points, dtype=np.float32)
        w2c = np.linalg.inv(c2w)
        pts_cam = (w2c[:3, :3] @ points_np.T).T + w2c[:3, 3]

        z = pts_cam[:, 2]
        valid = z > 1e-3

        proj = (K @ pts_cam.T).T
        u = np.floor(proj[:, 0] / (proj[:, 2] + 1e-8)).astype(np.int32)
        v = np.floor(proj[:, 1] / (proj[:, 2] + 1e-8)).astype(np.int32)
        valid &= (u >= 0) & (u < W) & (v >= 0) & (v < H)

        valid_idx = np.where(valid)[0]
        if valid_idx.size == 0:
            return 0

        pix_id = v[valid_idx].astype(np.int64) * np.int64(W) + u[valid_idx].astype(np.int64)
        depths = z[valid_idx]

        order = np.lexsort((depths, pix_id))
        if order.size == 0:
            return 0
        pix_sorted = pix_id[order]
        first_for_pixel = np.ones(order.shape[0], dtype=bool)
        if order.shape[0] > 1:
            first_for_pixel[1:] = pix_sorted[1:] != pix_sorted[:-1]
        front_idx = valid_idx[order[first_for_pixel]]
        if front_idx.size == 0:
            return 0

        remove_idx = front_idx[fg_mask[v[front_idx], u[front_idx]]]
        if remove_idx.size == 0:
            return 0

        keep = np.ones(len(points_np), dtype=bool)
        keep[remove_idx] = False
        self.world_state.update_static(points_np[keep], np.asarray(colors)[keep])
        return int(remove_idx.size)

    def _run_observer_diffusion(
        self,
        iter_idx: int,
        iter_prompt: str,
        frames_per_iter: int,
        scene_proj: torch.Tensor,
        fg_proj,
        preceding_scene_proj,
        preceding_fg_proj,
        reference_frames,
        instance_reference_frames,
        observer_cfg: dict,
    ):
        """Run observer diffusion for one iteration (T2V with preceding frames).

        Subclasses can override to change conditioning (e.g. I2V).
        """
        preceding_frames = self._get_preceding_frames(iter_idx)

        with logger.timed("Observer diffusion (LiveWorld)"):
            observer_output = self.observer_adapter.run_iteration(
                ObserverIterationInput(
                    preceding_frames=preceding_frames,
                    prompt=iter_prompt,
                    num_frames=frames_per_iter,
                    infer_steps=observer_cfg.get("infer_steps", 20),
                    target_scene_proj=scene_proj,
                    target_fg_proj=fg_proj,
                    preceding_scene_proj=preceding_scene_proj,
                    preceding_fg_proj=preceding_fg_proj,
                    reference_frames=reference_frames,
                    instance_reference_frames=instance_reference_frames,
                    guidance_scale=observer_cfg.get("guidance_scale"),
                    sp_context_scale=float(observer_cfg.get("sp_context_scale", 1.0)),
                    cpu_offload=self.runtime.cpu_offload.get("observer", False),
                    seed=self.runtime.seed,
                )
            )
        return observer_output

    def _initialize_world_state(self) -> None:
        """Initialize the world coordinate system and the initial static point cloud.

        This mirrors the first-iteration logic in inference:
        1. Load geometry poses from geometry.npz
        2. Load first frame from video
        3. Detect dynamic objects with Qwen + SAM3
        4. Reconstruct first-frame point cloud with Stream3R/MapAnything
        5. Transform camera trajectory to reconstructed coordinate system
        6. Update world_state with static points/colors
        """
        config = self.config
        cpu_offload_observer = self.runtime.cpu_offload.get("observer", False)

        # ---------------------------------------------------------------------------
        # 1) Load geometry poses
        # ---------------------------------------------------------------------------
        geometry_path = config.get("geometry_file_name")
        if not geometry_path:
            raise KeyError("geometry_file_name must be provided in config")
        if not os.path.exists(geometry_path):
            raise FileNotFoundError(f"Geometry file not found: {geometry_path}")

        geometry_poses_c2w = load_geometry_poses(geometry_path)

        self.geometry.geometry_poses_c2w = geometry_poses_c2w

        # ---------------------------------------------------------------------------
        # 2) Load first frame from image
        # ---------------------------------------------------------------------------
        first_frame_image = config.get("first_frame_image")
        if not first_frame_image:
            raise KeyError("first_frame_image must be provided in config")
        if not os.path.exists(first_frame_image):
            raise FileNotFoundError(f"First frame image not found: {first_frame_image}")

        self.geometry.start_frame = 0

        # Load image directly.
        target_h, target_w = self.target_resolution
        first_frame_pil = load_frame_from_image(first_frame_image, (target_w, target_h))

        self.geometry.first_frame = first_frame_pil

        # ---------------------------------------------------------------------------
        # 3) Detect dynamic objects with Qwen + SAM3 (using pre-initialized models)
        # ---------------------------------------------------------------------------

        # Detect dynamic prompts using Qwen (scene init: aggressive removal).
        # Ensure Qwen is on GPU (may have been offloaded to CPU by a previous run).
        scene_detect_prompt = config["event"]["detection"]["scene_detect_prompt"]
        cpu_offload_qwen = config["event"]["cpu_offload"].get("qwen", False)
        if cpu_offload_qwen and hasattr(self.qwen_extractor, "model"):
            self.qwen_extractor.model.to(self._device)
        dynamic_prompts, _ = self.qwen_extractor.extract(first_frame_image, prompt=scene_detect_prompt)
        if cpu_offload_qwen and hasattr(self.qwen_extractor, "model"):
            self.qwen_extractor.model.to("cpu")
            torch.cuda.empty_cache()

        all_prompts_init = list(dynamic_prompts) if dynamic_prompts else []
        all_prompts_init.append("sky")

        # Segment dynamic objects with SAM3 (per-prompt for better quality).
        merged_mask = None
        for prompt in all_prompts_init:
            seg_result = self.sam3_segmenter.segment(
                video_path=[first_frame_pil],
                prompts=[prompt],
                frame_index=0,
                expected_frames=1,
            )
            if seg_result.size == 0:
                continue
            merged_mask = seg_result[0].copy() if merged_mask is None else merged_mask | seg_result[0]

        # Dilate to aggressively remove foreground edges from first-frame
        # background point cloud (same strategy as _detect_dynamic_masks).
        if merged_mask is not None and merged_mask.any():
            _kern = np.ones((3, 3), dtype=np.uint8)
            merged_mask = cv2.dilate(
                merged_mask.astype(np.uint8),
                _kern,
                iterations=5,
            ).astype(bool)
        dynamic_mask = merged_mask

        # ---------------------------------------------------------------------------
        # 4) Reconstruct first-frame point cloud with handler (Stream3R/MapAnything)
        # ---------------------------------------------------------------------------

        # Reconstruct first-frame point cloud using pre-initialized handler.
        first_frame_np = np.array(first_frame_pil)
        recon: ReconstructionResult = self.pointcloud_handler.reconstruct_first_frame(
            frame=first_frame_np,
            geometry_poses_c2w=geometry_poses_c2w,
            dynamic_mask=dynamic_mask,
            options=self._backbone_options,
        )

        if cpu_offload_observer:
            self.pointcloud_handler.offload_to_cpu()

        # ---------------------------------------------------------------------------
        # 5) Transform camera trajectory to reconstructed coordinate system
        # ---------------------------------------------------------------------------

        alignment_transform = recon.alignment_transform
        if alignment_transform is not None:
            # Transform geometry poses to the reconstructed coordinate system.
            poses_c2w = np.array(
                [
                    alignment_transform @ p.astype(np.float32)
                    if p.shape == (4, 4)
                    else alignment_transform @ np.vstack([p.astype(np.float32), [0, 0, 0, 1]])
                    for p in geometry_poses_c2w
                ],
                dtype=np.float32,
            )
        else:
            poses_c2w = geometry_poses_c2w.copy()

        # Store geometry information.
        self.geometry.poses_c2w = poses_c2w
        self.geometry.intrinsics = recon.intrinsics
        self.geometry.intrinsics_size = recon.intrinsics_size
        self.geometry.alignment_transform = alignment_transform

        # ---------------------------------------------------------------------------
        # 6) Update world_state with static points/colors
        # ---------------------------------------------------------------------------
        self.world_state.update_static(recon.points_world, recon.colors)

        # ---------------------------------------------------------------------------
        # 7) Save dynamic mask and generate first-frame preceding projections
        # ---------------------------------------------------------------------------
        # Save dynamic mask for potential future use
        self.geometry.first_frame_dynamic_mask = dynamic_mask

        # Generate first-frame scene/fg projections for P1 mode
        # These will be used as preceding_scene_proj and preceding_fg_proj in iter_idx=0
        self._initialize_first_frame_projections(
            first_frame_np=first_frame_np,
            dynamic_mask=dynamic_mask,
        )

    def _initialize_first_frame_projections(
        self,
        first_frame_np: np.ndarray,
        dynamic_mask: np.ndarray | None,
    ) -> None:
        """Generate first-frame scene and fg projection pixel frames for P1 mode.

        Separates the first frame into scene (bg on black) and fg (fg on black)
        using the dynamic mask. If no dynamic objects detected (mask is None),
        scene = full frame, fg = all black.

        Stored as pixel frames in geometry lists for use as
        preceding_scene_proj / preceding_fg_proj at iter_idx=0.

        Args:
            first_frame_np: First frame [H, W, 3] uint8.
            dynamic_mask: Dynamic object mask [H, W] bool (True = foreground),
                          or None if no dynamic objects detected.
        """
        output_height = self.config["observer"]["height"]
        output_width = self.config["observer"]["width"]

        # Resize to output resolution if needed.
        h, w = first_frame_np.shape[:2]
        if (h, w) != (output_height, output_width):
            first_frame_np = cv2.resize(first_frame_np, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
            if dynamic_mask is not None:
                dynamic_mask = cv2.resize(
                    dynamic_mask.astype(np.uint8), (output_width, output_height), interpolation=cv2.INTER_NEAREST
                ).astype(bool)

        # Scene = background on black; fg = foreground on black.
        scene_frame = np.zeros_like(first_frame_np)
        fg_frame = np.zeros_like(first_frame_np)
        if dynamic_mask is not None:
            scene_frame[~dynamic_mask] = first_frame_np[~dynamic_mask]
            fg_frame[dynamic_mask] = first_frame_np[dynamic_mask]
        else:
            # No dynamic objects: entire frame is scene, fg is empty.
            scene_frame = first_frame_np.copy()

        self.geometry.all_scene_proj_frames.append(scene_frame)
        self.geometry.all_fg_proj_frames.append(fg_frame)

    def _render_scene_projection(self, iter_idx: int) -> torch.Tensor:
        """Render the static scene projection for the current iteration.

        Projects the static point cloud to the target frames of this iteration,
        encodes them through VAE, and returns the scene_proj latent tensor.

        Args:
            iter_idx: Current iteration index.

        Returns:
            scene_proj latent tensor of shape [C, T, h, w].
        """
        observer_cfg = self.config["observer"]
        frames_per_iter = observer_cfg["frames_per_iter"]

        # Compute target frame indices for this iteration.
        start_frame = self.geometry.start_frame + iter_idx * frames_per_iter
        target_frames = list(range(start_frame, start_frame + frames_per_iter))

        # Get static point cloud.
        points, colors = self.world_state.get_static()
        if points is None or colors is None:
            raise RuntimeError("Static point cloud not initialized")

        # Get geometry info.
        poses_c2w = self.geometry.poses_c2w
        intrinsics = self.geometry.intrinsics
        intrinsics_size = self.geometry.intrinsics_size

        if poses_c2w is None or intrinsics is None:
            raise RuntimeError("Geometry not initialized (poses_c2w or intrinsics is None)")

        # Output size from observer config or default.
        output_height = observer_cfg.get("height", 480)
        output_width = observer_cfg.get("width", 832)
        output_size = (output_height, output_width)

        # Get VAE from observer adapter.
        vae = self.observer_adapter.pipeline.vae
        device = torch.device(self.runtime.device)
        dtype = torch.float16
        self._ensure_vae_on_device()

        # Generate scene projection latent.
        scene_proj = generate_scene_projection_from_pointcloud(
            points_world=points,
            colors=colors,
            target_frames=target_frames,
            poses_c2w=poses_c2w,
            intrinsics=intrinsics,
            output_size=output_size,
            intrinsics_size=intrinsics_size,
            vae=vae,
            device=device,
            dtype=dtype,
        )

        return scene_proj

    def _compute_last_frame_black_ratio(self, iter_idx: int) -> float:
        """Compute the black pixel ratio of the scene projection's last target frame.

        Renders the static point cloud from the last camera pose of this iteration
        and returns the fraction of pixels that are black (no point cloud coverage).
        High black ratio means the camera sees mostly unseen areas.
        """
        observer_cfg = self.config["observer"]
        frames_per_iter = observer_cfg["frames_per_iter"]

        # Last target frame index for this iteration.
        last_frame_idx = self.geometry.start_frame + (iter_idx + 1) * frames_per_iter - 1

        points, colors = self.world_state.get_static()
        if points is None:
            return 1.0

        poses_c2w = self.geometry.poses_c2w
        intrinsics = self.geometry.intrinsics
        intrinsics_size = self.geometry.intrinsics_size

        output_height = observer_cfg.get("height", 480)
        output_width = observer_cfg.get("width", 832)

        # Scale intrinsics from processing size to output size.
        proc_h, proc_w = intrinsics_size
        if intrinsics.ndim == 2:
            K = intrinsics.copy().astype(np.float32)
            if K.shape != (3, 3):
                flat = K.reshape(-1)
                fx, fy, cx, cy = flat[:4]
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            K = scale_intrinsics(K, (proc_h, proc_w), (output_height, output_width))
        else:
            k_idx = min(last_frame_idx, len(intrinsics) - 1)
            K = intrinsics[k_idx].copy().astype(np.float32)
            if K.shape != (3, 3):
                flat = K.reshape(-1)
                fx, fy, cx, cy = flat[:4]
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            K = scale_intrinsics(K, (proc_h, proc_w), (output_height, output_width))

        pose_idx = min(max(last_frame_idx, 0), len(poses_c2w) - 1)
        c2w = poses_c2w[pose_idx]

        proj_rgb = render_projection(
            points_world=points,
            K=K,
            c2w=c2w,
            image_size=(output_height, output_width),
            channels=["rgb"],
            colors=colors,
            fill_holes_kernel=0,
        )

        # Black pixels: all channels are 0.
        black_ratio = float((proj_rgb.sum(axis=-1) == 0).mean())
        return black_ratio

    def _render_event_projections(self, iter_idx: int) -> tuple[list[torch.Tensor], set, dict[str, set]]:
        """Render per-event projections for the current iteration.

        Uses per-frame foreground point clouds so that frame t shows the entity
        at its position at time t (temporally aligned), following the approach
        in ``scripts/create_event_video_proj.py``.

        Falls back to merged-cloud projection when per-frame data is unavailable.

        Args:
            iter_idx: Current iteration index.

        Returns:
            Tuple of:
              - List of event projection latent tensors, each [C, T, h, w].
              - Set of event_ids that have visible (non-empty) projections.
              - Dict mapping event_id → set of visible obj_ids (per-instance visibility).
                Empty dict when per-instance data is unavailable.
        """
        observer_cfg = self.config["observer"]
        frames_per_iter = observer_cfg["frames_per_iter"]

        # Compute target frame indices for this iteration.
        start_frame = self.geometry.start_frame + iter_idx * frames_per_iter
        target_frames = list(range(start_frame, start_frame + frames_per_iter))

        # Get geometry info.
        poses_c2w = self.geometry.poses_c2w
        intrinsics = self.geometry.intrinsics
        intrinsics_size = self.geometry.intrinsics_size

        if poses_c2w is None or intrinsics is None:
            return [], set(), {}

        # Output size from observer config.
        output_height = observer_cfg.get("height", 480)
        output_width = observer_cfg.get("width", 832)
        output_size = (output_height, output_width)

        # Get VAE from observer adapter.
        vae = self.observer_adapter.pipeline.vae
        device = torch.device(self.runtime.device)
        dtype = torch.float16
        self._ensure_vae_on_device()

        # Scale intrinsics from processing size to output size.
        proc_h, proc_w = intrinsics_size
        if intrinsics.ndim == 2:
            K_base = intrinsics.copy().astype(np.float32)
            if K_base.shape != (3, 3):
                flat = K_base.reshape(-1)
                fx, fy, cx, cy = flat[:4]
                K_base = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            K_scaled = scale_intrinsics(K_base, (proc_h, proc_w), (output_height, output_width))
        else:
            K_scaled = None  # handled per-frame below

        event_projs = []
        visible_event_ids = set()
        visible_instances: dict[str, set] = {}
        for event_pc in self.world_state.list_event_pointclouds():
            if event_pc.points is None or event_pc.colors is None:
                continue

            has_per_frame = (
                event_pc.per_frame_points is not None
                and event_pc.per_frame_colors is not None
                and len(event_pc.per_frame_points) > 0
            )

            if has_per_frame:
                # --- Per-frame fg projection (temporally aligned) ---
                n_event_frames = len(event_pc.per_frame_points)
                projections = []

                for i, frame_idx in enumerate(target_frames):
                    # Map observer frame i to event frame index.
                    event_frame_i = min(i, n_event_frames - 1)
                    pts_local = event_pc.per_frame_points[event_frame_i]
                    cols_local = event_pc.per_frame_colors[event_frame_i]

                    if len(pts_local) == 0:
                        # Empty fg for this frame — render black.
                        projections.append(np.zeros((output_height, output_width, 3), dtype=np.uint8))
                        continue

                    # Transform from anchor camera space to world space.
                    pts_world = transform_points(pts_local, event_pc.anchor_pose_c2w)

                    # Get camera pose for this target frame.
                    pose_idx = min(max(frame_idx, 0), len(poses_c2w) - 1)
                    c2w = poses_c2w[pose_idx]

                    # Get per-frame intrinsics if available.
                    if K_scaled is not None:
                        K_frame = K_scaled
                    else:
                        k_idx = min(max(frame_idx, 0), len(intrinsics) - 1)
                        K_raw = intrinsics[k_idx].copy().astype(np.float32)
                        if K_raw.shape != (3, 3):
                            flat = K_raw.reshape(-1)
                            fx, fy, cx, cy = flat[:4]
                            K_raw = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                        K_frame = scale_intrinsics(K_raw, (proc_h, proc_w), (output_height, output_width))

                    proj = render_projection(
                        points_world=pts_world,
                        K=K_frame,
                        c2w=c2w,
                        image_size=(output_height, output_width),
                        channels=["rgb"],
                        colors=cols_local,
                        fill_holes_kernel=0,
                    )
                    projections.append(proj)

                # Stack and encode through VAE (same as generate_scene_projection_from_pointcloud).
                projections_np = np.stack(projections, axis=0)  # [T, H, W, 3]
                fg_cov = float((projections_np > 10).any(axis=-1).mean())
                logger.info(
                    f"  Event {event_pc.event_id[:12]} projection coverage={fg_cov:.2%} "
                    f"(max={int(projections_np.max())})"
                )

                # Track whether this event has any visible pixels in the projection.
                if projections_np.max() > 10:
                    visible_event_ids.add(event_pc.event_id)

                # Per-instance visibility: count projected points per instance.
                # An instance is visible if >= 20% of its total points project
                # inside the observer view across the target frames.
                has_per_inst = (
                    event_pc.per_instance_per_frame_points is not None
                    and event_pc.per_instance_per_frame_colors is not None
                )
                inst_vis_threshold = 0.2
                if has_per_inst:
                    vis_obj_ids = set()
                    for obj_id, inst_pts_list in event_pc.per_instance_per_frame_points.items():
                        n_inst_frames = len(inst_pts_list)
                        inst_visible = False
                        best_ratio = 0.0
                        for i, frame_idx in enumerate(target_frames):
                            event_fi = min(i, n_inst_frames - 1)
                            pts_i = inst_pts_list[event_fi]
                            n_pts = len(pts_i)
                            if n_pts == 0:
                                continue
                            pts_w = transform_points(pts_i, event_pc.anchor_pose_c2w)
                            pose_idx = min(max(frame_idx, 0), len(poses_c2w) - 1)
                            c2w = poses_c2w[pose_idx]
                            w2c = np.linalg.inv(c2w)
                            pts_cam = transform_points(pts_w, w2c)
                            if K_scaled is not None:
                                K_f = K_scaled
                            else:
                                k_idx = min(max(frame_idx, 0), len(intrinsics) - 1)
                                K_raw = intrinsics[k_idx].copy().astype(np.float32)
                                if K_raw.shape != (3, 3):
                                    flat = K_raw.reshape(-1)
                                    fx, fy, cx, cy = flat[:4]
                                    K_raw = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                                K_f = scale_intrinsics(K_raw, (proc_h, proc_w), (output_height, output_width))
                            uv, z = project_points(pts_cam, K_f)
                            in_view = (
                                (z > 0)
                                & (uv[:, 0] >= 0)
                                & (uv[:, 0] < output_width)
                                & (uv[:, 1] >= 0)
                                & (uv[:, 1] < output_height)
                            )
                            frame_ratio = float(in_view.sum()) / n_pts
                            best_ratio = max(best_ratio, frame_ratio)
                            if frame_ratio >= inst_vis_threshold:
                                inst_visible = True
                                break
                        logger.info(
                            f"    obj_{obj_id}: best_frame_ratio={best_ratio:.1%} "
                            f"(thr={inst_vis_threshold:.0%}) -> "
                            f"{'VISIBLE' if inst_visible else 'hidden'}"
                        )
                        if inst_visible:
                            vis_obj_ids.add(obj_id)
                    visible_instances[event_pc.event_id] = vis_obj_ids
                    logger.info(
                        f"  Event {event_pc.event_id[:12]}: "
                        f"{len(vis_obj_ids)}/{len(event_pc.per_instance_per_frame_points)} "
                        f"instances visible (obj_ids={vis_obj_ids})"
                    )

                projections_np = projections_np.transpose(0, 3, 1, 2)  # [T, 3, H, W]
                proj_tensor = torch.from_numpy(projections_np).float() / 127.5 - 1.0

                with torch.no_grad():
                    vae_device = next(vae.model.parameters()).device
                    if vae_device != device:
                        vae.model.to(device)
                        vae.mean = vae.mean.to(device)
                        vae.std = vae.std.to(device)

                    proj_tensor = proj_tensor.to(device=device, dtype=dtype)
                    proj_tensor = proj_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, T, H, W]
                    latent = vae.encode_to_latent(proj_tensor)  # [1, C, T', h, w]
                    latent = latent.squeeze(0).permute(1, 0, 2, 3)  # [T', C, h, w]

                event_projs.append(latent)
            else:
                # --- Merged-cloud projection (no per-frame data available) ---
                logger.info(f"  Event {event_pc.event_id[:12]} using merged-cloud projection fallback")
                proj = generate_scene_projection_from_pointcloud(
                    points_world=event_pc.points,
                    colors=event_pc.colors,
                    target_frames=target_frames,
                    poses_c2w=poses_c2w,
                    intrinsics=intrinsics,
                    output_size=output_size,
                    intrinsics_size=intrinsics_size,
                    vae=vae,
                    device=device,
                    dtype=dtype,
                )
                event_projs.append(proj)
                visible_event_ids.add(event_pc.event_id)

        return event_projs, visible_event_ids, visible_instances

    # ===================================================================
    # Intermediate event detection helpers
    # ===================================================================

    def _compute_union_event_coverage_per_frame(
        self,
        iter_idx: int,
    ) -> list[tuple[int, float]]:
        """Compute per-frame pixel coverage of union event point clouds on observer frames.

        For each observer frame in this round, projects ALL event point clouds
        (transformed to world space) onto the frame's camera pose and computes
        the fraction of pixels covered.

        Args:
            iter_idx: Current iteration index.

        Returns:
            List of (local_frame_index, coverage_ratio) sorted by frame index.
        """
        observer_cfg = self.config["observer"]
        frames_per_iter = observer_cfg["frames_per_iter"]

        union_pts, union_cols = self.world_state.get_union_event_points_world()
        if union_pts is None or len(union_pts) == 0:
            return [(i, 0.0) for i in range(frames_per_iter)]

        poses_c2w = self.geometry.poses_c2w
        intrinsics = self.geometry.intrinsics
        intrinsics_size = self.geometry.intrinsics_size

        if poses_c2w is None or intrinsics is None:
            return [(i, 0.0) for i in range(frames_per_iter)]

        output_height = observer_cfg.get("height", 480)
        output_width = observer_cfg.get("width", 832)

        # Scale intrinsics from processing size to output size.
        proc_h, proc_w = intrinsics_size
        if intrinsics.ndim == 2:
            K_base = intrinsics.copy().astype(np.float32)
            if K_base.shape != (3, 3):
                flat = K_base.reshape(-1)
                fx, fy, cx, cy = flat[:4]
                K_base = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            K_scaled = scale_intrinsics(K_base, (proc_h, proc_w), (output_height, output_width))
        else:
            K_scaled = None

        results = []
        for local_i in range(frames_per_iter):
            frame_idx = self.geometry.start_frame + 1 + iter_idx * frames_per_iter + local_i
            pose_idx = min(max(frame_idx, 0), len(poses_c2w) - 1)
            c2w = poses_c2w[pose_idx]

            if K_scaled is not None:
                K_frame = K_scaled
            else:
                k_idx = min(max(frame_idx, 0), len(intrinsics) - 1)
                K_raw = intrinsics[k_idx].copy().astype(np.float32)
                if K_raw.shape != (3, 3):
                    flat = K_raw.reshape(-1)
                    fx, fy, cx, cy = flat[:4]
                    K_raw = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                K_frame = scale_intrinsics(K_raw, (proc_h, proc_w), (output_height, output_width))

            proj = render_projection(
                points_world=union_pts,
                K=K_frame,
                c2w=c2w,
                image_size=(output_height, output_width),
                channels=["rgb"],
                colors=union_cols,
                fill_holes_kernel=0,
            )
            coverage = float((proj > 10).any(axis=-1).mean())
            results.append((local_i, coverage))

        return results

    def _detect_intermediate_events(
        self,
        iter_idx: int,
        candidate_frames: list[int],
        round_dir,
    ) -> list[tuple]:
        """Detect and register intermediate events from candidate observer frames.

        Args:
            iter_idx: Current iteration index.
            candidate_frames: Local frame indices (0-based within round) with low coverage.
            round_dir: Output directory for this round.

        Returns:
            List of (agent, anchor_local_frame_idx) for newly registered events.
        """
        if self.intermediate_detector is None:
            return []

        intermediate_cfg = self.config["event"].get("intermediate_detection", {})
        max_per_round = int(intermediate_cfg.get("max_per_round", 1))
        det_cfg = self.config.get("event", {}).get("detection", {})
        min_mask_ratio = float(det_cfg.get("min_mask_ratio", 0.05))

        observer_cfg = self.config["observer"]
        frames_per_iter = observer_cfg["frames_per_iter"]

        poses_c2w = self.geometry.poses_c2w
        new_agents = []

        for local_idx in sorted(candidate_frames):
            if len(new_agents) >= max_per_round:
                break

            abs_idx = iter_idx * frames_per_iter + local_idx
            if abs_idx >= len(self.geometry.all_generated_frames):
                continue

            frame_np = self.geometry.all_generated_frames[abs_idx]
            frame_idx_global = self.geometry.start_frame + 1 + iter_idx * frames_per_iter + local_idx
            pose_idx = min(max(frame_idx_global, 0), len(poses_c2w) - 1)
            pose_c2w = poses_c2w[pose_idx]

            # Detect entities.
            with logger.timed(f"  Intermediate detect (frame {local_idx})"):
                detection_results = self.intermediate_detector.detect(
                    frame_path="",
                    frame=frame_np,
                    frame_index=0,
                )

            if not detection_results:
                continue

            # Filter by mask ratio (same as normal detection).
            scene_detections = []
            for detection in detection_results:
                mask_bool = np.asarray(detection.mask) > 0
                mask_ratio = float(mask_bool.mean()) if mask_bool.size > 0 else 0.0
                if mask_ratio < min_mask_ratio:
                    continue
                if "cup" in detection.name:
                    continue
                scene_detections.append(detection)

            if not scene_detections:
                continue

            detected_entities = sorted({d.name.strip().lower() for d in scene_detections if d.name and d.name.strip()})
            if not detected_entities:
                continue

            observation = EventObservation(
                event_id=make_event_id(detected_entities, pose_c2w),
                frame_path="",
                frame_index=frame_idx_global,
                pose_c2w=pose_c2w,
                entities=detected_entities,
                timestamp=float(frame_idx_global),
                frame=frame_np,
                detections=scene_detections,
            )

            debug_dir = str(round_dir / f"intermediate_event_debug_{local_idx}")
            existing_agents = {a.event_id for a in self.event_pool.list_active()}
            agent = self.event_pool.register_event(observation, debug_dir=debug_dir)

            if agent is not None and agent.event_id not in existing_agents:
                # New event registered (not a reuse).
                agent.state.is_intermediate = True
                agent.state.intermediate_anchor_local_frame = local_idx
                new_agents.append((agent, local_idx))
                logger.info(
                    f"  Intermediate event registered: {agent.event_id[:12]} "
                    f"entities={detected_entities} anchor_local={local_idx}"
                )

        return new_agents

    def _track_entity_visibility(
        self,
        entities: list[str],
        iter_idx: int,
        anchor_local: int,
    ) -> int:
        """Determine the last observer frame where the entity is visible.

        Runs SAM3 segmentation on observer frames from anchor_local+1 onwards
        until the entity's mask area drops below threshold.

        Args:
            entities: Entity names to track.
            iter_idx: Current iteration index.
            anchor_local: Local frame index where entity first appeared.

        Returns:
            Last visible local frame index (inclusive).
        """
        observer_cfg = self.config["observer"]
        frames_per_iter = observer_cfg["frames_per_iter"]
        mask_vanish_threshold = 0.005  # entity considered gone if mask < 0.5% of frame

        last_visible = anchor_local

        for local_i in range(anchor_local + 1, frames_per_iter):
            abs_idx = iter_idx * frames_per_iter + local_i
            if abs_idx >= len(self.geometry.all_generated_frames):
                break

            frame_np = self.geometry.all_generated_frames[abs_idx]
            frame_pil = Image.fromarray(frame_np)

            try:
                seg_result = self.sam3_segmenter.segment(
                    video_path=[frame_pil],
                    prompts=entities,
                    frame_index=0,
                    expected_frames=1,
                )
            finally:
                frame_pil.close()

            if seg_result is None:
                break

            mask = np.asarray(seg_result)
            if mask.ndim == 3:
                mask = mask[0]
            mask_ratio = float((mask > 0).mean())
            if mask_ratio < mask_vanish_threshold:
                break
            last_visible = local_i

        return last_visible

    def _catchup_intermediate_event(
        self,
        agent,
        anchor_local: int,
        iter_idx: int,
        round_dir,
    ):
        """Run catch-up I2V generation for an intermediate event with explicit fg_proj.

        The event's anchor frame is the observer frame at anchor_local.
        Generates a full event video, trims to the visible duration, and projects
        the trimmed video to a point cloud.

        Args:
            agent: The newly registered intermediate EventAgent.
            anchor_local: Local frame index within the round where entity first appeared.
            iter_idx: Current iteration index.
            round_dir: Output directory for this round.

        Returns:
            (EventVideo, EventPointCloud) or None if generation fails.
        """
        observer_cfg = self.config["observer"]
        _frames_per_iter = observer_cfg["frames_per_iter"]
        intermediate_cfg = self.config["event"].get("intermediate_detection", {})
        min_visible_duration = int(intermediate_cfg.get("min_visible_duration", 5))

        # 1. Track entity visibility.
        with logger.timed("  Track entity visibility (SAM3)"):
            last_visible = self._track_entity_visibility(
                entities=agent.state.entities,
                iter_idx=iter_idx,
                anchor_local=anchor_local,
            )
        visible_duration = last_visible - anchor_local + 1
        logger.info(
            f"  Intermediate event visibility: frames {anchor_local}-{last_visible} ({visible_duration} frames)"
        )
        if visible_duration < min_visible_duration:
            logger.info(
                f"  Skipping intermediate event: visible_duration={visible_duration} < min={min_visible_duration}"
            )
            return None

        # 2. Build fg_proj from observer back-projection.
        target_fg_proj = self._build_intermediate_fg_proj(
            agent=agent,
            anchor_local=anchor_local,
            last_visible=last_visible,
            iter_idx=iter_idx,
        )

        # 3. Generate I2V prompt.
        event_dir = ensure_dir(round_dir / f"intermediate_event_{agent.event_id}")
        with logger.timed("  I2V prompt (Qwen)"):
            script = agent.evolve_prompt()
            evt_scene_text, evt_fg_text = self.event_pool.generate_i2v_prompt(
                anchor_frame=agent.state.current_anchor_frame,
                entities=agent.state.entities,
                iteration=agent.state.iteration_count,
            )
            script.text = self._compose_prompt(evt_scene_text, evt_fg_text)

        # 4. Run I2V generation with explicit fg_proj.
        with logger.timed("  I2V generate (LiveWorld) [catch-up]"):
            event_video = self._evolve_event_with_observer(
                agent,
                script,
                str(event_dir),
                target_fg_proj=target_fg_proj,
            )

        # 5. Trim video to visible duration.
        # event_video.frames includes anchor at index 0.
        trim_end = visible_duration + 1  # +1 for anchor frame
        trimmed_frames = event_video.frames[:trim_end]
        logger.info(f"  Trimmed event video: {len(event_video.frames)} -> {len(trimmed_frames)} frames")

        # Update agent state with trimmed data.
        agent.state.current_anchor_frame = trimmed_frames[-1]
        agent.state.all_generated_frames = list(trimmed_frames)

        # Create trimmed EventVideo for projection.
        trimmed_video = EventVideo(
            event_id=event_video.event_id,
            video_path=event_video.video_path,
            fps=event_video.fps,
            num_frames=len(trimmed_frames),
            anchor_pose_c2w=event_video.anchor_pose_c2w,
            frames=trimmed_frames,
        )

        # 6. Project trimmed video to point cloud.
        with logger.timed("  Project to pointcloud [catch-up]"):
            _s3r_model, _s3r_session, _s3r_fed = self._get_shared_stream3r()
            event_pc, _s3r_fed = self.event_projector.project(
                event_video=trimmed_video,
                dynamic_prompts=agent.state.entities,
                output_dir=str(event_dir),
                stream3r_model=_s3r_model,
                sam3_segmenter=self.sam3_segmenter,
                stream3r_session=_s3r_session,
                stream3r_frames_fed=_s3r_fed,
            )
            self._sync_stream3r_frames_fed(_s3r_fed)
            self.world_state.update_event(event_pc)

        return trimmed_video, event_pc

    def _build_intermediate_fg_proj(
        self,
        agent,
        anchor_local: int,
        last_visible: int,
        iter_idx: int,
    ) -> torch.Tensor | None:
        """Build fg_proj latent for an intermediate event via observer back-projection.

        Uses per-frame world points from Stream3R (stored before dynamic filtering)
        and entity masks to extract the entity's 3D positions from observer frames,
        then projects them onto the event's fixed anchor camera.

        Args:
            agent: The intermediate EventAgent.
            anchor_local: Local frame index where entity first appeared.
            last_visible: Last visible local frame index.
            iter_idx: Current iteration index.

        Returns:
            fg_proj latent tensor [C, T, h, w] or None if data unavailable.
        """
        observer_cfg = self.config["observer"]
        frames_per_iter = observer_cfg["frames_per_iter"]
        output_height = observer_cfg.get("height", 480)
        output_width = observer_cfg.get("width", 832)

        # Check if pre-filtering world points are available.
        handler = self.pointcloud_handler
        per_frame_wp = getattr(handler, "last_per_frame_world_points", None)
        if per_frame_wp is None or len(per_frame_wp) == 0:
            logger.warning(
                "  No pre-filtering world points available for fg back-projection; falling back to target_fg_proj=None"
            )
            return None

        poses_c2w = self.geometry.poses_c2w
        intrinsics = self.geometry.intrinsics
        intrinsics_size = self.geometry.intrinsics_size
        if poses_c2w is None or intrinsics is None:
            return None

        anchor_pose_c2w = agent.state.anchor_pose_c2w

        # Scale intrinsics to output size.
        proc_h, proc_w = intrinsics_size
        if intrinsics.ndim == 2:
            K_base = intrinsics.copy().astype(np.float32)
            if K_base.shape != (3, 3):
                flat = K_base.reshape(-1)
                fx, fy, cx, cy = flat[:4]
                K_base = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            K_scaled = scale_intrinsics(K_base, (proc_h, proc_w), (output_height, output_width))
        else:
            K_scaled = None

        # Get entity SAM3 masks for observer frames.
        entity_masks = self._get_intermediate_entity_masks(
            entities=agent.state.entities,
            iter_idx=iter_idx,
            anchor_local=anchor_local,
            last_visible=last_visible,
        )

        vae = self.observer_adapter.pipeline.vae
        device = torch.device(self.runtime.device)
        dtype = torch.float16
        self._ensure_vae_on_device()

        # For each event frame (0 to frames_per_iter-1), project entity
        # from the corresponding observer frame to the event anchor camera.
        projections = []
        for event_i in range(frames_per_iter):
            # Map event frame to observer frame.
            obs_local = min(anchor_local + event_i, last_visible)
            obs_wp_idx = obs_local  # index into per_frame_wp

            if obs_wp_idx >= len(per_frame_wp):
                projections.append(np.zeros((output_height, output_width, 3), dtype=np.uint8))
                continue

            wp_frame = per_frame_wp[obs_wp_idx]  # (H_wp, W_wp, 3)
            H_wp, W_wp = wp_frame.shape[:2]

            # Get entity mask for this observer frame.
            mask_local_idx = obs_local - anchor_local
            if mask_local_idx < len(entity_masks) and entity_masks[mask_local_idx] is not None:
                entity_mask = entity_masks[mask_local_idx]
            else:
                projections.append(np.zeros((output_height, output_width, 3), dtype=np.uint8))
                continue

            # Resize mask to match world points resolution if needed.
            if entity_mask.shape[:2] != (H_wp, W_wp):
                entity_mask = cv2.resize(
                    entity_mask.astype(np.uint8),
                    (W_wp, H_wp),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                entity_mask = entity_mask > 0

            # Extract entity world points.
            entity_pts = wp_frame[entity_mask]  # (M, 3)
            if len(entity_pts) == 0:
                projections.append(np.zeros((output_height, output_width, 3), dtype=np.uint8))
                continue

            # Get entity colors from observer frame.
            abs_idx = iter_idx * frames_per_iter + obs_local
            if abs_idx < len(self.geometry.all_generated_frames):
                obs_frame = self.geometry.all_generated_frames[abs_idx]
                obs_h, obs_w = obs_frame.shape[:2]
                if entity_mask.shape[:2] != (obs_h, obs_w):
                    mask_for_color = cv2.resize(
                        entity_mask.astype(np.uint8),
                        (obs_w, obs_h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                else:
                    mask_for_color = entity_mask
                entity_cols = obs_frame[mask_for_color]  # (M, 3)
                # Align lengths (world points and colors may differ due to resolution).
                min_len = min(len(entity_pts), len(entity_cols))
                entity_pts = entity_pts[:min_len]
                entity_cols = entity_cols[:min_len]
            else:
                entity_cols = np.full((len(entity_pts), 3), 128, dtype=np.uint8)

            # Project entity points onto event anchor camera.
            if K_scaled is not None:
                K_frame = K_scaled
            else:
                frame_idx_global = self.geometry.start_frame + 1 + iter_idx * frames_per_iter + obs_local
                k_idx = min(max(frame_idx_global, 0), len(intrinsics) - 1)
                K_raw = intrinsics[k_idx].copy().astype(np.float32)
                if K_raw.shape != (3, 3):
                    flat = K_raw.reshape(-1)
                    fx, fy, cx, cy = flat[:4]
                    K_raw = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                K_frame = scale_intrinsics(K_raw, (proc_h, proc_w), (output_height, output_width))

            proj = render_projection(
                points_world=entity_pts,
                K=K_frame,
                c2w=anchor_pose_c2w,
                image_size=(output_height, output_width),
                channels=["rgb"],
                colors=entity_cols,
                fill_holes_kernel=0,
            )
            projections.append(proj)

        # Encode projections through VAE.
        projections_np = np.stack(projections, axis=0)  # [T, H, W, 3]
        fg_cov = float((projections_np > 10).any(axis=-1).mean())
        logger.info(f"  Intermediate fg_proj coverage={fg_cov:.2%}")

        if projections_np.max() <= 10:
            logger.warning("  Intermediate fg_proj is empty; returning None")
            return None

        projections_np = projections_np.transpose(0, 3, 1, 2)  # [T, 3, H, W]
        proj_tensor = torch.from_numpy(projections_np).float() / 127.5 - 1.0

        with torch.no_grad():
            vae_device = next(vae.model.parameters()).device
            if vae_device != device:
                vae.model.to(device)
                vae.mean = vae.mean.to(device)
                vae.std = vae.std.to(device)

            proj_tensor = proj_tensor.to(device=device, dtype=dtype)
            proj_tensor = proj_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, T, H, W]
            latent = vae.encode_to_latent(proj_tensor)  # [1, C, T', h, w]
            latent = latent.squeeze(0).permute(1, 0, 2, 3)  # [T', C, h, w]

        return latent

    def _get_intermediate_entity_masks(
        self,
        entities: list[str],
        iter_idx: int,
        anchor_local: int,
        last_visible: int,
    ) -> list[np.ndarray | None]:
        """Get per-frame SAM3 entity masks for observer frames anchor_local to last_visible.

        Args:
            entities: Entity names to segment.
            iter_idx: Current iteration index.
            anchor_local: First frame local index.
            last_visible: Last visible frame local index.

        Returns:
            List of masks (one per frame from anchor_local to last_visible), each (H, W) bool.
        """
        observer_cfg = self.config["observer"]
        frames_per_iter = observer_cfg["frames_per_iter"]

        masks = []
        for local_i in range(anchor_local, last_visible + 1):
            abs_idx = iter_idx * frames_per_iter + local_i
            if abs_idx >= len(self.geometry.all_generated_frames):
                masks.append(None)
                continue

            frame_np = self.geometry.all_generated_frames[abs_idx]
            frame_pil = Image.fromarray(frame_np)
            try:
                seg_result = self.sam3_segmenter.segment(
                    video_path=[frame_pil],
                    prompts=entities,
                    frame_index=0,
                    expected_frames=1,
                )
            finally:
                frame_pil.close()

            if seg_result is None:
                masks.append(None)
                continue
            mask = np.asarray(seg_result)
            if mask.ndim == 3:
                mask = mask[0]
            masks.append((mask > 0).astype(np.uint8))

        return masks

    def _build_event_iter0_references(
        self,
        anchor_frame: np.ndarray,
        entities: list[str],
        output_h: int,
        output_w: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Build one-shot scene/instance references from the current anchor frame.

        Used only for event iter-0 so first-round event generation can already
        use reference conditioning without relying on previous generated frames.
        """
        if anchor_frame is None or not entities:
            return None, None
        scene_ref_dilate = self._event_scene_ref_mask_dilate()

        anchor_pil = Image.fromarray(anchor_frame)
        seg_result = self.sam3_segmenter.segment(
            video_path=[anchor_pil],
            prompts=entities,
            frame_index=0,
            expected_frames=1,
        )

        fg_mask = None
        if seg_result is not None and getattr(seg_result, "size", 0) > 0:
            fg_mask = seg_result[0] if seg_result.ndim == 3 else seg_result
            fg_mask = (fg_mask > 0).astype(np.uint8)

        scene_ref = anchor_frame.copy()
        if fg_mask is not None:
            fh, fw = scene_ref.shape[:2]
            if fg_mask.shape[:2] != (fh, fw):
                fg_mask = cv2.resize(
                    fg_mask.astype(np.uint8),
                    (fw, fh),
                    interpolation=cv2.INTER_NEAREST,
                )
            fg_mask = self._dilate_binary_mask(
                fg_mask,
                radius=scene_ref_dilate,
            )
            scene_ref[fg_mask > 0] = 0
        scene_refs = np.stack([scene_ref], axis=0)

        inst_refs: list[np.ndarray] = []
        for entity_name in entities:
            inst_masks = self.sam3_segmenter.segment_instances(
                video_path=[anchor_pil],
                prompt=entity_name,
                frame_index=0,
                expected_frames=1,
            )
            for inst_mask in inst_masks:
                ref = self._mask_to_instance_ref(anchor_frame, inst_mask, output_h, output_w)
                if ref is not None:
                    inst_refs.append(ref)

        inst_ref_arr = np.stack(inst_refs, axis=0) if inst_refs else None
        return scene_refs, inst_ref_arr

    def _evolve_event_with_observer(
        self,
        agent,
        script,
        output_dir: str,
        target_fg_proj=None,
    ):
        """Generate an event video using the Observer (LiveWorld) model with static camera.

        The camera stays at the anchor pose for all frames. The scene projection
        is the static scene rendered from that fixed viewpoint, repeated for all
        target frames.

        Preceding frames follow the same logic as the main observer:
        - iter 0: anchor frame
        - iter > 0: last generated frame from previous event iteration

        Reference frames are built once after the first iteration completes:
        - scene refs (7): first frame + 6 random frames, fg masked out
        - instance ref (1): first frame fg crop on black background
        These are stored permanently in agent.state and reused every iteration.

        Args:
            agent: The EventAgent to evolve.
            script: EventScript with prompt and generation parameters.
            output_dir: Directory to save the event video.

        Returns:
            EventVideo with path to the saved MP4 and metadata.
        """
        observer_cfg = self.config["observer"]
        anchor_frame = agent.state.current_anchor_frame  # [H, W, 3] uint8
        anchor_pose_c2w = agent.state.anchor_pose_c2w  # [4, 4]

        iter_count = agent.state.iteration_count
        num_frames = script.horizon
        fps = script.fps

        output_height = observer_cfg.get("height", 480)
        output_width = observer_cfg.get("width", 832)

        vae = self.observer_adapter.pipeline.vae
        device = torch.device(self.runtime.device)
        dtype = torch.float16
        cpu_offload = self.runtime.cpu_offload.get("observer", False)
        self._ensure_vae_on_device()

        # -------------------------------------------------------------------
        # 1. Static scene projection: render scene from anchor pose for all
        #    frames. The anchor pose is repeated so all frames show the same
        #    viewpoint — only entity motion (from the prompt) is generated.
        # -------------------------------------------------------------------
        points, colors = self.world_state.get_static()
        intrinsics = self.geometry.intrinsics
        intrinsics_size = self.geometry.intrinsics_size

        if points is None or intrinsics is None:
            raise RuntimeError("Static point cloud or intrinsics not initialized")

        target_scene_proj = agent.state.cached_target_scene_proj
        if target_scene_proj is None:
            static_poses = np.tile(anchor_pose_c2w[None], (num_frames, 1, 1))
            target_frames_list = list(range(num_frames))
            target_scene_proj = generate_scene_projection_from_pointcloud(
                points_world=points,
                colors=colors,
                target_frames=target_frames_list,
                poses_c2w=static_poses,
                intrinsics=intrinsics,
                output_size=(output_height, output_width),
                intrinsics_size=intrinsics_size,
                vae=vae,
                device=device,
                dtype=dtype,
            )
            agent.state.cached_target_scene_proj = target_scene_proj

        # -------------------------------------------------------------------
        # 2. Preceding frames:
        #    iter 0: 1 pixel frame (anchor)
        #    iter > 0: last N pixel frames from all_generated_frames
        # -------------------------------------------------------------------
        max_preceding_frames = self._max_preceding_frames()
        if iter_count == 0:
            # P1: single anchor frame
            preceding_frames = np.stack([anchor_frame], axis=0)  # [1, H, W, 3]
        else:
            # Last N pixel frames from event's generated frames.
            all_evt_frames = agent.state.all_generated_frames
            if not all_evt_frames:
                raise RuntimeError(f"No generated frames for event iter_count={iter_count}")
            n_preceding_pixels = max_preceding_frames
            if len(all_evt_frames) >= n_preceding_pixels:
                preceding_frames = np.stack(all_evt_frames[-n_preceding_pixels:], axis=0)
            else:
                # Pad by repeating first available frame.
                frames = list(all_evt_frames)
                while len(frames) < n_preceding_pixels:
                    frames.insert(0, frames[0])
                preceding_frames = np.stack(frames, axis=0)

        # -------------------------------------------------------------------
        # 2b. Preceding scene/fg projections from stored pixel frames.
        #     Initialize on first call, then use accumulated pixel frames.
        # -------------------------------------------------------------------
        if agent.state.all_scene_proj_frames is None:
            agent.state.all_scene_proj_frames = []
        if agent.state.all_fg_proj_frames is None:
            agent.state.all_fg_proj_frames = []

        # On first event iteration, create initial projection pixel frame
        # by separating anchor into scene (bg) and fg parts.
        if not agent.state.all_scene_proj_frames:
            # Use event-entity mask on the current anchor frame.
            dyn_mask = self._segment_event_entities_mask(
                frame=anchor_frame,
                entities=agent.state.entities or [],
            )
            scene_frame = np.zeros_like(anchor_frame)
            fg_frame = np.zeros_like(anchor_frame)
            if dyn_mask is not None:
                h, w = anchor_frame.shape[:2]
                if dyn_mask.shape[:2] != (h, w):
                    dyn_mask_resized = cv2.resize(
                        dyn_mask.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                else:
                    dyn_mask_resized = dyn_mask.astype(bool)
                scene_frame[~dyn_mask_resized] = anchor_frame[~dyn_mask_resized]
                fg_frame[dyn_mask_resized] = anchor_frame[dyn_mask_resized]
            else:
                scene_frame = anchor_frame.copy()
            agent.state.all_scene_proj_frames.append(scene_frame)
            agent.state.all_fg_proj_frames.append(fg_frame)

        # Select P1/N pixel frames from accumulated projection frames.
        n_proj_pixels = 1 if iter_count == 0 else max_preceding_frames
        all_scene_pf = agent.state.all_scene_proj_frames
        if len(all_scene_pf) >= n_proj_pixels:
            prec_scene_pixels = np.stack(all_scene_pf[-n_proj_pixels:], axis=0)
        else:
            pf = list(all_scene_pf)
            while len(pf) < n_proj_pixels:
                pf.insert(0, pf[0])
            prec_scene_pixels = np.stack(pf, axis=0)

        all_fg_pf = agent.state.all_fg_proj_frames
        prec_fg_pixels = None
        if all_fg_pf:
            if len(all_fg_pf) >= n_proj_pixels:
                prec_fg_pixels = np.stack(all_fg_pf[-n_proj_pixels:], axis=0)
            else:
                pf = list(all_fg_pf)
                while len(pf) < n_proj_pixels:
                    pf.insert(0, pf[0])
                prec_fg_pixels = np.stack(pf, axis=0)

        # VAE-encode preceding projections only when enabled by event config.
        use_prec_scene_proj = self._event_use_preceding_scene_proj()
        use_prec_fg_proj = self._event_use_preceding_fg_proj()
        use_target_scene_proj = self._event_use_target_scene_proj()
        keep_frames = self._event_target_scene_proj_frames()

        # Apply target_scene_proj masking:
        #   use_target_scene_proj=false  -> zero out ALL frames
        #   target_scene_proj_frames="0,64" -> only keep those latent frames, zero rest
        if not use_target_scene_proj:
            target_scene_proj = torch.zeros_like(target_scene_proj)
            agent.state.cached_target_scene_proj = None
        elif keep_frames is not None:
            # target_scene_proj shape: [C, T_latent, h, w]
            n_latent = target_scene_proj.shape[1]
            mask = torch.zeros(n_latent, dtype=torch.bool)
            for fi in keep_frames:
                if 0 <= fi < n_latent:
                    mask[fi] = True
            zeroed = target_scene_proj.clone()
            zeroed[:, ~mask, :, :] = 0.0
            target_scene_proj = zeroed

        proj_frames_label = (
            "ALL"
            if (use_target_scene_proj and keep_frames is None)
            else ("OFF" if not use_target_scene_proj else str(keep_frames))
        )
        _evt_sp_scale = float(self.config.get("event", {}).get("evolution", {}).get("sp_context_scale", 1.0))
        logger.info(
            "  event conditioning: "
            f"preceding_frames=ON, "
            f"target_scene_proj={proj_frames_label}, "
            f"preceding_scene_proj={'ON' if use_prec_scene_proj else 'OFF'}, "
            f"preceding_fg_proj={'ON' if use_prec_fg_proj else 'OFF'}, "
            f"sp_context_scale={_evt_sp_scale}"
        )
        preceding_scene_proj = (
            self._encode_pixel_frames_to_proj_latent(prec_scene_pixels) if use_prec_scene_proj else None
        )
        preceding_fg_proj = self._encode_pixel_frames_to_proj_latent(prec_fg_pixels) if use_prec_fg_proj else None

        # -------------------------------------------------------------------
        # 2c. Reference frames (from agent state, set after first iteration).
        # -------------------------------------------------------------------
        reference_frames = agent.state.reference_frames_scene  # [7, H, W, 3] or None
        instance_reference_frames = agent.state.reference_frames_instance  # [1, H, W, 3] or None

        # For iter-0, bootstrap one-shot references from anchor frame if the
        # persistent references are not ready yet.
        if iter_count == 0 and reference_frames is None and instance_reference_frames is None:
            reference_frames, instance_reference_frames = self._build_event_iter0_references(
                anchor_frame=anchor_frame,
                entities=agent.state.entities or [],
                output_h=output_height,
                output_w=output_width,
            )
            n_scene_ref = 0 if reference_frames is None else len(reference_frames)
            n_inst_ref = 0 if instance_reference_frames is None else len(instance_reference_frames)
            logger.info(f"  event iter0 refs (bootstrap): scene={n_scene_ref}, instance={n_inst_ref}")

        # -------------------------------------------------------------------
        # 3. Call observer adapter with static camera parameters.
        # -------------------------------------------------------------------
        # Event generation uses target background projection.
        # target_fg_proj is None for normal events, but may be provided
        # for intermediate events via observer back-projection.
        #
        # NOTE: overlay_fg_on_scene is intentionally NOT applied here.
        # The event model always receives the clean scene projection without
        # fg baked in (controlled by event.projection.overlay_fg_on_scene_event).
        proj_cfg = self.config["event"].get("projection", {})
        if proj_cfg.get("overlay_fg_on_scene_event", False) and target_fg_proj is not None:
            target_scene_proj, _ = overlay_fg_on_scene(target_scene_proj, target_fg_proj, vae, device, dtype)
            logger.info("  event model: applied fg overlay on scene projection")

        evolution_cfg = self.config.get("event", {}).get("evolution", {})
        event_sp_scale = float(evolution_cfg.get("sp_context_scale", 1.0))

        observer_output = self.observer_adapter.run_iteration(
            ObserverIterationInput(
                preceding_frames=preceding_frames,
                prompt=script.text,
                num_frames=num_frames,
                infer_steps=observer_cfg.get("infer_steps", 20),
                target_scene_proj=target_scene_proj,
                target_fg_proj=target_fg_proj,
                preceding_scene_proj=preceding_scene_proj,
                preceding_fg_proj=preceding_fg_proj,
                reference_frames=reference_frames,
                instance_reference_frames=instance_reference_frames,
                guidance_scale=observer_cfg.get("guidance_scale", 5.0),
                cpu_offload=cpu_offload,
                seed=self.runtime.seed,
                sp_context_scale=event_sp_scale,
            )
        )

        # Decode target_scene_proj once and reuse across rounds for this event.
        if agent.state.cached_scene_proj_pixels is None:
            decoded = decode_latent_to_rgb(target_scene_proj, vae, device, dtype)
            agent.state.cached_scene_proj_pixels = [f.copy() for f in decoded]
        scene_proj_pixels = [f.copy() for f in (agent.state.cached_scene_proj_pixels or [])]

        agent.state.all_scene_proj_frames.extend(scene_proj_pixels)

        # -------------------------------------------------------------------
        # 4. Convert frames to numpy, prepend anchor frame, save as MP4.
        # -------------------------------------------------------------------
        generated = observer_output.frames
        if isinstance(generated, torch.Tensor):
            frames_np = generated.cpu().numpy()
            if frames_np.dtype != np.uint8:
                frames_np = frames_np.clip(0, 255).astype(np.uint8)
        else:
            frames_np = np.array(generated)

        # Prepend the anchor frame so the event video starts with the
        # deterministic anchor (matching main observer loop behaviour).
        frames_np = np.concatenate([anchor_frame[None], frames_np], axis=0)

        video_filename = f"event_video_seed{self.runtime.seed}.mp4"
        video_path = os.path.join(output_dir, video_filename)
        save_video(video_path, list(frames_np), fps=fps)

        # -------------------------------------------------------------------
        # 5. Update agent state for next iteration.
        # -------------------------------------------------------------------
        last_frame = frames_np[-1]
        agent.state.current_anchor_frame = last_frame
        agent.state.last_video_path = video_path
        agent.state.iteration_count += 1

        # Accumulate all generated frames (skip frame 0 = anchor, already stored).
        if agent.state.all_generated_frames is None:
            agent.state.all_generated_frames = list(frames_np)
        else:
            agent.state.all_generated_frames.extend(list(frames_np[1:]))

        # -------------------------------------------------------------------
        # 6. Build permanent reference frames after first iteration.
        #    Scene refs (7): first frame + 6 random frames, fg masked out.
        #    Instance ref (1): first frame fg crop centered on black bg.
        # -------------------------------------------------------------------
        if agent.state.reference_frames_scene is None:
            all_evt_frames = agent.state.all_generated_frames  # includes anchor
            n_total = len(all_evt_frames)
            scene_ref_dilate = self._event_scene_ref_mask_dilate()

            # Select frame indices: 0 (first) + up to (max_count-1) random from rest.
            max_scene = self._event_max_scene_refs() or 7
            if n_total <= max_scene:
                selected_indices = list(range(n_total))
            else:
                selected_indices = [0] + sorted(random.sample(range(1, n_total), max_scene - 1))  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.

            selected_frames = [all_evt_frames[i].copy() for i in selected_indices]

            # Segment fg on selected reference frames (per-frame masks) so scene refs
            # do not reuse a stale first-frame mask when objects move.
            entities = agent.state.entities
            selected_frame_pils = [Image.fromarray(f) for f in selected_frames]
            try:
                seg_result = self.sam3_segmenter.segment(
                    video_path=selected_frame_pils,
                    prompts=entities,
                    frame_index=0,
                    expected_frames=len(selected_frames),
                )
            finally:
                for im in selected_frame_pils:
                    im.close()

            fg_masks_by_ref: list[np.ndarray | None] = [None] * len(selected_frames)
            if seg_result is not None and getattr(seg_result, "size", 0) > 0:
                seg_masks = seg_result if seg_result.ndim == 3 else seg_result[None, ...]
                if seg_masks.shape[0] == 1 and len(selected_frames) > 1:
                    seg_masks = np.repeat(seg_masks, len(selected_frames), axis=0)
                n_masks = min(len(selected_frames), seg_masks.shape[0])
                if n_masks != len(selected_frames):
                    logger.warning(
                        f"event scene_ref mask count mismatch: masks={seg_masks.shape[0]} "
                        f"refs={len(selected_frames)}; applying first {n_masks}"
                    )
                for mi in range(n_masks):
                    fg_masks_by_ref[mi] = (seg_masks[mi] > 0).astype(np.uint8)

            # --- Instance references: per-instance fg crop from first frame ---
            first_frame_pil = Image.fromarray(all_evt_frames[0])
            inst_refs: list[np.ndarray] = []
            all_instance_masks: list[np.ndarray] = []
            for entity_name in entities:
                all_instance_masks.extend(
                    self.sam3_segmenter.segment_instances(
                        video_path=[first_frame_pil],
                        prompt=entity_name,
                        frame_index=0,
                        expected_frames=1,
                    )
                )
            first_frame_pil.close()
            first_frame = all_evt_frames[0]
            for inst_mask in all_instance_masks:
                ref = self._mask_to_instance_ref(first_frame, inst_mask, output_height, output_width)
                if ref is not None:
                    inst_refs.append(ref)

            # --- Scene references: mask out fg from selected frames ---
            scene_refs = []
            for ref_idx, frame in enumerate(selected_frames):
                fg_mask = fg_masks_by_ref[ref_idx]
                if fg_mask is not None:
                    fh, fw = frame.shape[:2]
                    mask_resized = fg_mask
                    if fg_mask.shape[:2] != (fh, fw):
                        mask_resized = cv2.resize(
                            fg_mask.astype(np.uint8),
                            (fw, fh),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    mask_resized = self._dilate_binary_mask(
                        mask_resized,
                        radius=scene_ref_dilate,
                    )
                    scene_frame = frame.copy()
                    scene_frame[mask_resized > 0] = 0
                    scene_refs.append(scene_frame)
                else:
                    scene_refs.append(frame)

            agent.state.reference_frames_scene = np.stack(scene_refs, axis=0)
            if inst_refs:
                agent.state.reference_frames_instance = np.stack(inst_refs, axis=0)  # [N_inst, H, W, 3]

            # Save reference frames to disk.
            out_path = Path(output_dir)
            n_scene_refs_to_save = len(scene_refs)
            if iter_count == 0 and n_scene_refs_to_save > 1:
                n_scene_refs_to_save = 1
            for i, ref in enumerate(scene_refs[:n_scene_refs_to_save]):
                save_image(out_path / f"event_scene_ref_{i:02d}.png", ref)
            for i, ref in enumerate(inst_refs):
                save_image(out_path / f"event_inst_ref_{i:02d}.png", ref)

        return EventVideo(
            event_id=agent.event_id,
            video_path=video_path,
            fps=fps,
            num_frames=len(frames_np),
            anchor_pose_c2w=anchor_pose_c2w,
            frames=frames_np,
        )

    def _get_preceding_frames(self, iter_idx: int) -> np.ndarray:
        """Return preceding pixel frames for observer conditioning.

        Policy:
        - iter 0: 1 pixel frame
        - iter > 0: `observer.max_preceding_frames` pixel frames (default 9)

        Args:
            iter_idx: Current iteration index.

        Returns:
            Array of preceding frames, shape [P, H, W, 3], uint8 RGB.
            P=1 for iter 0, P=max_preceding_frames for iter > 0.
        """
        if iter_idx == 0:
            # P1: single first frame.
            if self.geometry.first_frame is None:
                raise RuntimeError("First frame not initialized")
            first_frame_np = np.array(self.geometry.first_frame)
            return np.stack([first_frame_np], axis=0)

        # Last N pixel frames from all_generated_frames.
        all_frames = self.geometry.all_generated_frames
        if not all_frames:
            raise RuntimeError(f"No generated frames available for iter_idx={iter_idx}")

        n_preceding_pixels = self._max_preceding_frames()
        if len(all_frames) >= n_preceding_pixels:
            return np.stack(all_frames[-n_preceding_pixels:], axis=0)
        # Not enough frames yet — use all available, pad by repeating first.
        frames = list(all_frames)
        while len(frames) < n_preceding_pixels:
            frames.insert(0, frames[0])
        return np.stack(frames, axis=0)

    def _get_preceding_projection_frames(self, iter_idx: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return preceding scene and fg projection pixel frames.

        Policy:
        - iter 0: 1 pixel frame
        - iter > 0: `observer.max_preceding_frames` pixel frames (default 9)

        These pixel frames will be VAE-encoded together with the preceding pixel
        frames inside run_single_iteration, matching the training encode path.

        Args:
            iter_idx: Current iteration index.

        Returns:
            Tuple of (scene_proj_frames, fg_proj_frames):
            - scene_proj_frames: [P, H, W, 3] uint8 or None
            - fg_proj_frames: [P, H, W, 3] uint8 or None
        """
        all_scene = self.geometry.all_scene_proj_frames
        all_fg = self.geometry.all_fg_proj_frames

        if not all_scene:
            return None, None

        n_pixels = 1 if iter_idx == 0 else self._max_preceding_frames()

        # Scene projection frames
        if len(all_scene) >= n_pixels:
            scene_frames = np.stack(all_scene[-n_pixels:], axis=0)
        else:
            frames = list(all_scene)
            while len(frames) < n_pixels:
                frames.insert(0, frames[0])
            scene_frames = np.stack(frames, axis=0)

        # Fg projection frames
        fg_frames = None
        if all_fg:
            if len(all_fg) >= n_pixels:
                fg_frames = np.stack(all_fg[-n_pixels:], axis=0)
            else:
                frames = list(all_fg)
                while len(frames) < n_pixels:
                    frames.insert(0, frames[0])
                fg_frames = np.stack(frames, axis=0)

        return scene_frames, fg_frames

    def _encode_pixel_frames_to_proj_latent(self, pixel_frames: np.ndarray | None) -> torch.Tensor | None:
        """VAE-encode pixel frames to projection latent.

        Mirrors the encoding path in generate_scene_projection_from_pointcloud:
        pixel [P, H, W, 3] uint8 → normalize → [1, 3, P, H, W] → VAE → [C, P', h, w]

        Args:
            pixel_frames: [P, H, W, 3] uint8 numpy array, or None.

        Returns:
            Latent tensor [C, P', h, w] or None if input is None.
        """
        if pixel_frames is None:
            return None

        vae = self.observer_adapter.pipeline.vae
        device = torch.device(self.runtime.device)
        dtype = torch.float16
        self._ensure_vae_on_device()

        # [P, H, W, 3] -> [P, 3, H, W]
        frames_chw = pixel_frames.transpose(0, 3, 1, 2)
        # Normalize to [-1, 1]
        proj_tensor = torch.from_numpy(frames_chw).float() / 127.5 - 1.0
        # [P, 3, H, W] -> [1, 3, P, H, W]
        proj_tensor = proj_tensor.permute(1, 0, 2, 3).unsqueeze(0)
        proj_tensor = proj_tensor.to(device=device, dtype=dtype)

        with torch.no_grad():
            vae_device = next(vae.model.parameters()).device
            if vae_device != device:
                vae.model.to(device)
                vae.mean = vae.mean.to(device)
                vae.std = vae.std.to(device)
            latent = vae.encode_to_latent(proj_tensor)  # [1, P', C, h, w]
            # [1, P', C, h, w] -> [P', C, h, w] -> [C, P', h, w]
            latent = latent.squeeze(0).permute(1, 0, 2, 3)

        return latent

    def _accumulate_projections(self, scene_proj: torch.Tensor, fg_proj: torch.Tensor | None) -> None:
        """Decode current iteration's projection latents to pixel frames and store.

        Args:
            scene_proj: Current iteration's scene projection [C, T, h, w].
            fg_proj: Current iteration's fg projection [C, T, h, w] or None.
        """
        vae = self.observer_adapter.pipeline.vae
        device = torch.device(self.runtime.device)
        dtype = torch.float16
        self._ensure_vae_on_device()

        scene_pixels = decode_latent_to_rgb(scene_proj, vae, device, dtype)  # [T_pixel, H, W, 3]
        self.geometry.all_scene_proj_frames.extend(list(scene_pixels))

        if fg_proj is not None:
            fg_pixels = decode_latent_to_rgb(fg_proj, vae, device, dtype)  # [T_pixel, H, W, 3]
            self.geometry.all_fg_proj_frames.extend(list(fg_pixels))

    def _get_reference_frames(
        self,
        iter_idx: int,
        target_frame_indices: list[int],
        max_reference_frames: int = 7,
        diag_path: Path | None = None,
    ) -> np.ndarray | None:
        """Retrieve reference frames based on 3D IoU with target frames.

        Uses anchor-based selection to find historical frames with highest IoU
        to the target poses.

        Args:
            iter_idx: Current iteration index.
            target_frame_indices: Global frame indices for target frames.
            max_reference_frames: Maximum number of reference frames to retrieve.
            diag_path: If provided, save diagnostics to this file.

        Returns:
            Reference frames as numpy array [R, H, W, 3] or None if not available.
        """
        if iter_idx == 0:
            return None

        points, colors = self.world_state.get_static()
        if points is None or len(self.geometry.frame_visible_points) == 0:
            return None

        observer_cfg = self.config["observer"]
        output_height = observer_cfg.get("height", 480)
        output_width = observer_cfg.get("width", 832)
        output_size = (output_height, output_width)

        ref_cfg = observer_cfg.get("reference_retrieval", {})
        if not isinstance(ref_cfg, dict):
            ref_cfg = {}
        iou_threshold = float(
            ref_cfg.get(
                "iou_threshold",
                observer_cfg.get("reference_iou_threshold", 0.05),
            )
        )
        voxel_size_iou = float(
            ref_cfg.get(
                "voxel_size_iou",
                observer_cfg.get("reference_iou_voxel_size", 0.002),
            )
        )
        max_iou_points = int(
            ref_cfg.get(
                "max_iou_points",
                observer_cfg.get("reference_iou_max_points", 50000),
            )
        )
        iou_device = ref_cfg.get(
            "iou_device",
            observer_cfg.get("reference_iou_device", "auto"),
        )

        reference_frames, ref_indices, diag_lines = retrieve_reference_frames(
            target_frame_indices=target_frame_indices,
            all_generated_frames=self.geometry.all_generated_frames,
            frame_visible_points=self.geometry.frame_visible_points,
            points_world=points,
            poses_c2w=self.geometry.poses_c2w,
            intrinsics=self.geometry.intrinsics,
            image_size=output_size,
            max_reference_frames=max_reference_frames,
            iou_threshold=iou_threshold,
            voxel_size_iou=voxel_size_iou,
            iou_device=iou_device,
            max_iou_points=max_iou_points,
        )
        diag_lines = [  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.
            f"observer.reference_retrieval: iou_threshold={iou_threshold}, "
            f"voxel_size_iou={voxel_size_iou}, max_iou_points={max_iou_points}, "
            f"iou_device={iou_device}",
            "",
        ] + diag_lines

        if reference_frames is not None:
            # Enforce minimum temporal spacing of 4 frames between refs.
            min_spacing = 4
            spaced_indices: list[int] = []
            spaced_frames: list[np.ndarray] = []
            for i, fidx in enumerate(ref_indices):
                too_close = any(abs(fidx - kept) < min_spacing for kept in spaced_indices)
                if too_close:
                    continue
                spaced_indices.append(fidx)
                spaced_frames.append(reference_frames[i])

            diag_lines.append("")
            diag_lines.append(f"--- Temporal spacing filter (min_spacing={min_spacing}) ---")
            diag_lines.append(f"  before: {len(ref_indices)} refs {ref_indices}")
            diag_lines.append(f"  after:  {len(spaced_indices)} refs {spaced_indices}")

            if not spaced_frames:
                reference_frames = None
                ref_indices = []
            else:
                reference_frames = np.stack(spaced_frames, axis=0)
                ref_indices = spaced_indices

        if reference_frames is not None:
            # Remove foreground from scene references using stored dynamic masks.
            diag_lines.append("")
            diag_lines.append("--- Foreground removal from scene refs ---")
            diag_lines.append(
                f"  frame_dynamic_masks available for frames: {sorted(self.geometry.frame_dynamic_masks.keys())}"
            )
            ref_mask_entities = sorted(
                {
                    str(ent).strip()
                    for agent in self.event_pool.list_active()
                    for ent in (agent.state.entities or [])
                    if str(ent).strip()
                }
            )
            diag_lines.append(f"  active-entity prompts for ref-mask fallback: {ref_mask_entities}")

            # Fallback: segment each selected scene ref independently, so mask
            # alignment is frame-exact and does not depend on temporal propagation.
            sam3_ref_masks: np.ndarray | None = None
            if ref_mask_entities:
                per_ref_masks: list[np.ndarray] = []
                for i, ref_frame in enumerate(reference_frames):
                    ref_pil = Image.fromarray(ref_frame)
                    try:
                        seg_result = self.sam3_segmenter.segment(
                            video_path=[ref_pil],
                            prompts=ref_mask_entities,
                            frame_index=0,
                            expected_frames=1,
                        )
                        if seg_result is not None and getattr(seg_result, "size", 0) > 0:
                            seg_mask = seg_result[0] if seg_result.ndim == 3 else seg_result
                            per_ref_masks.append(np.asarray(seg_mask).astype(bool))
                        else:
                            per_ref_masks.append(np.zeros(ref_frame.shape[:2], dtype=bool))
                    except Exception as exc:
                        diag_lines.append(f"  SAM3 ref-mask fallback failed at ref[{i}]: {exc}")
                        per_ref_masks.append(np.zeros(ref_frame.shape[:2], dtype=bool))
                    finally:
                        ref_pil.close()
                if per_ref_masks:
                    sam3_ref_masks = np.stack(per_ref_masks, axis=0)

            for i, fidx in enumerate(ref_indices):
                cached_mask = None
                if fidx in self.geometry.frame_dynamic_masks:
                    cached_mask = np.asarray(self.geometry.frame_dynamic_masks[fidx]).astype(bool)
                sam3_mask = None
                if sam3_ref_masks is not None and i < len(sam3_ref_masks):
                    sam3_mask = np.asarray(sam3_ref_masks[i]).astype(bool)

                mask = None
                if cached_mask is not None and sam3_mask is not None:
                    mask = cached_mask | sam3_mask
                    mask_src = "cached|sam3_ref"
                elif sam3_mask is not None:
                    mask = sam3_mask
                    mask_src = "sam3_ref"
                elif cached_mask is not None:
                    mask = cached_mask
                    mask_src = "cached"
                else:
                    mask_src = "none"

                if mask is not None:
                    fh, fw = reference_frames[i].shape[:2]
                    if mask.shape != (fh, fw):
                        diag_lines.append(
                            f"  ref frame {fidx}: mask[{mask_src}] shape {mask.shape} "
                            f"!= frame shape ({fh},{fw}), resizing"
                        )
                        mask = cv2.resize(
                            mask.astype(np.uint8),
                            (fw, fh),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    n_masked = int(mask.sum())
                    total = mask.size
                    diag_lines.append(
                        f"  ref frame {fidx}: mask[{mask_src}] found, "
                        f"{n_masked}/{total} pixels masked ({n_masked / total:.4f})"
                    )
                    reference_frames[i][mask] = 0
                else:
                    diag_lines.append(f"  ref frame {fidx}: NO mask (cached+sam3_ref) -> fg NOT removed")

        # Save diagnostics to file.
        if diag_path is not None:
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            with open(diag_path, "w") as f:
                f.write("\n".join(diag_lines))
                f.write("\n")

        return reference_frames

    def _get_reference_frames_pose_based(
        self,
        iter_idx: int,
        target_frame_indices: list[int],
        num_refs: int = 7,
        diag_path: Path | None = None,
    ) -> np.ndarray | None:
        """Select reference frames by 2D projection coverage onto target poses.

        For each of ``num_refs`` evenly-sampled target poses, project every
        historical frame's visible points onto that pose and pick the one with
        the highest pixel coverage.  Foreground is removed from the selected
        frames before returning.

        Args:
            iter_idx: Current iteration index.
            target_frame_indices: Global frame indices for the current round's
                target poses.
            num_refs: How many target poses to sample (= number of refs).
            diag_path: If provided, save diagnostics to this file.

        Returns:
            Reference frames as numpy array ``[R, H, W, 3]`` or ``None``.
        """
        diag_lines: list[str] = [f"=== pose_based_ref (iter {iter_idx}) ==="]

        # --- early exits ---
        if iter_idx == 0:
            diag_lines.append("iter_idx == 0 -> no historical frames, return None")
            self._save_diag(diag_path, diag_lines)
            return None

        fvp = self.geometry.frame_visible_points
        all_gen = self.geometry.all_generated_frames
        if not fvp or len(all_gen) == 0:
            diag_lines.append("frame_visible_points or all_generated_frames empty -> return None")
            self._save_diag(diag_path, diag_lines)
            return None

        observer_cfg = self.config["observer"]
        output_height = observer_cfg.get("height", 480)
        output_width = observer_cfg.get("width", 832)
        total_pixels = max(1, output_height * output_width)

        # --- 1. Evenly sample target poses ---
        n_targets = len(target_frame_indices)
        actual_refs = min(num_refs, n_targets)
        if actual_refs <= 0:
            diag_lines.append("no target poses -> return None")
            self._save_diag(diag_path, diag_lines)
            return None

        stride = max(1, n_targets // actual_refs)
        sampled_target_poses = [target_frame_indices[i * stride] for i in range(actual_refs)]
        diag_lines.append(
            f"target_frame_indices: {target_frame_indices[0]}..{target_frame_indices[-1]} (n={n_targets})"
        )
        diag_lines.append(f"sampled {actual_refs} target poses (stride={stride}): {sampled_target_poses}")
        diag_lines.append("")

        # --- 2. Build historical candidate set ---
        hist_items = sorted(
            [(idx, pts) for idx, pts in fvp.items() if pts is not None and len(pts) > 0 and 0 <= idx < len(all_gen)],
            key=lambda x: x[0],
        )
        if not hist_items:
            diag_lines.append("no valid historical frames -> return None")
            self._save_diag(diag_path, diag_lines)
            return None

        diag_lines.append(f"--- Historical candidates: {len(hist_items)} frames ---")
        for idx, pts in hist_items:
            diag_lines.append(f"  hist frame {idx}: {len(pts)} visible pts")
        diag_lines.append("")

        # --- 3. For each target pose, find best-coverage historical frame ---
        poses_c2w = self.geometry.poses_c2w
        K = self.geometry.intrinsics  # (3, 3)

        selected_hist_indices: list[int] = []  # frame indices chosen
        used_set: set = set()  # avoid duplicates

        for tp_idx in sampled_target_poses:
            c2w = poses_c2w[tp_idx]
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            t = w2c[:3, 3]

            best_cov = -1.0
            best_hist_idx = -1
            cov_list: list[tuple[int, float]] = []

            for hidx, hpts in hist_items:
                if hidx in used_set:
                    continue
                # Project hist points onto target pose
                pts_cam = (R @ hpts.T).T + t  # (N, 3)
                valid = pts_cam[:, 2] > 0.01
                pts_cam = pts_cam[valid]
                if len(pts_cam) == 0:
                    cov_list.append((hidx, 0.0))
                    continue
                pts_proj = (K @ pts_cam.T).T  # (N, 3)
                px = pts_proj[:, :2] / (pts_proj[:, 2:3] + 1e-8)
                px_int = np.floor(px).astype(np.int64)
                in_bounds = (
                    (px_int[:, 0] >= 0)
                    & (px_int[:, 0] < output_width)
                    & (px_int[:, 1] >= 0)
                    & (px_int[:, 1] < output_height)
                )
                px_int = px_int[in_bounds]
                if len(px_int) == 0:
                    cov_list.append((hidx, 0.0))
                    continue
                pixel_ids = px_int[:, 1] * output_width + px_int[:, 0]
                n_unique = len(np.unique(pixel_ids))
                coverage = n_unique / total_pixels
                cov_list.append((hidx, coverage))
                if coverage > best_cov:
                    best_cov = coverage
                    best_hist_idx = hidx

            # Among candidates with coverage > 0, pick the earliest (smallest index).
            eligible = [(cidx, cval) for cidx, cval in cov_list if cval > 0]
            eligible.sort(key=lambda x: x[0])  # sort by frame index ascending
            if eligible:
                best_hist_idx = eligible[0][0]
                best_cov = eligible[0][1]

            # Sort for diagnostics
            cov_list.sort(key=lambda x: x[1], reverse=True)
            diag_lines.append(f"target pose {tp_idx}:")
            for cidx, cval in cov_list[:5]:
                marker = " <-- SELECTED" if cidx == best_hist_idx else ""
                diag_lines.append(f"  hist {cidx}: coverage={cval:.4f}{marker}")
            if len(cov_list) > 5:
                diag_lines.append(f"  ... ({len(cov_list) - 5} more)")

            if best_hist_idx >= 0:
                selected_hist_indices.append(best_hist_idx)
                used_set.add(best_hist_idx)
                diag_lines.append(f"  => selected hist frame {best_hist_idx} (coverage={best_cov:.4f})")
            else:
                diag_lines.append("  => no valid candidate")
            diag_lines.append("")

        if not selected_hist_indices:
            diag_lines.append("no frames selected -> return None")
            self._save_diag(diag_path, diag_lines)
            return None

        # --- 4. Collect generated frames ---
        reference_frames = np.stack([all_gen[idx] for idx in selected_hist_indices], axis=0)  # [R, H, W, 3]
        diag_lines.append(f"=== Selected {len(selected_hist_indices)} reference frames: {selected_hist_indices} ===")
        diag_lines.append("")

        # --- 5. Remove foreground (reuse logic from _get_reference_frames) ---
        diag_lines.append("--- Foreground removal from scene refs ---")
        diag_lines.append(
            f"  frame_dynamic_masks available for frames: {sorted(self.geometry.frame_dynamic_masks.keys())}"
        )
        ref_mask_entities = sorted(
            {
                str(ent).strip()
                for agent in self.event_pool.list_active()
                for ent in (agent.state.entities or [])
                if str(ent).strip()
            }
        )
        diag_lines.append(f"  active-entity prompts for ref-mask fallback: {ref_mask_entities}")

        sam3_ref_masks: np.ndarray | None = None
        if ref_mask_entities:
            per_ref_masks: list[np.ndarray] = []
            for i, ref_frame in enumerate(reference_frames):
                ref_pil = Image.fromarray(ref_frame)
                try:
                    seg_result = self.sam3_segmenter.segment(
                        video_path=[ref_pil],
                        prompts=ref_mask_entities,
                        frame_index=0,
                        expected_frames=1,
                    )
                    if seg_result is not None and getattr(seg_result, "size", 0) > 0:
                        seg_mask = seg_result[0] if seg_result.ndim == 3 else seg_result
                        per_ref_masks.append(np.asarray(seg_mask).astype(bool))
                    else:
                        per_ref_masks.append(np.zeros(ref_frame.shape[:2], dtype=bool))
                except Exception as exc:
                    diag_lines.append(f"  SAM3 ref-mask fallback failed at ref[{i}]: {exc}")
                    per_ref_masks.append(np.zeros(ref_frame.shape[:2], dtype=bool))
                finally:
                    ref_pil.close()
            if per_ref_masks:
                sam3_ref_masks = np.stack(per_ref_masks, axis=0)

        for i, fidx in enumerate(selected_hist_indices):
            cached_mask = None
            if fidx in self.geometry.frame_dynamic_masks:
                cached_mask = np.asarray(self.geometry.frame_dynamic_masks[fidx]).astype(bool)
            sam3_mask = None
            if sam3_ref_masks is not None and i < len(sam3_ref_masks):
                sam3_mask = np.asarray(sam3_ref_masks[i]).astype(bool)

            mask = None
            if cached_mask is not None and sam3_mask is not None:
                mask = cached_mask | sam3_mask
                mask_src = "cached|sam3_ref"
            elif sam3_mask is not None:
                mask = sam3_mask
                mask_src = "sam3_ref"
            elif cached_mask is not None:
                mask = cached_mask
                mask_src = "cached"
            else:
                mask_src = "none"

            if mask is not None:
                fh, fw = reference_frames[i].shape[:2]
                if mask.shape != (fh, fw):
                    diag_lines.append(
                        f"  ref frame {fidx}: mask[{mask_src}] shape {mask.shape} != frame shape ({fh},{fw}), resizing"
                    )
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (fw, fh),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                n_masked = int(mask.sum())
                total = mask.size
                diag_lines.append(
                    f"  ref frame {fidx}: mask[{mask_src}] found, "
                    f"{n_masked}/{total} pixels masked ({n_masked / total:.4f})"
                )
                reference_frames[i][mask] = 0
            else:
                diag_lines.append(f"  ref frame {fidx}: NO mask (cached+sam3_ref) -> fg NOT removed")

        self._save_diag(diag_path, diag_lines)
        return reference_frames

    @staticmethod
    def _save_diag(diag_path: Path | None, diag_lines: list[str]) -> None:
        """Write diagnostic lines to a file."""
        if diag_path is not None:
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            with open(diag_path, "w") as f:
                f.write("\n".join(diag_lines) + "\n")

    def _get_instance_reference_frames(
        self,
        iter_idx: int,
        visible_event_ids: set,
        visible_instances: dict[str, set] | None = None,
    ) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
        """Extract instance reference frames from events with visible projections.

        When per-instance visibility data is available (``visible_instances``),
        only instances whose obj_id is in the visible set get reference frames.
        Stored anchor masks from the event point cloud are preferred over
        re-running SAM3 segmentation.

        Args:
            iter_idx: Current iteration index.
            visible_event_ids: Set of event_ids that have visible (non-empty) fg
                projections onto the observer camera, from _render_event_projections.
            visible_instances: Optional mapping {event_id: set of visible obj_ids}.
                When provided, only visible instances are included.

        Returns:
            Tuple of:
              - Instance reference frames [R_inst, H, W, 3] uint8, or None.
              - Per-event mapping {event_id: masked_frame (H, W, 3) uint8}.
        """
        if not visible_event_ids:
            return None, {}

        active_agents = self.event_pool.list_active()
        if not active_agents:
            return None, {}

        # Observer output resolution
        observer_cfg = self.config["observer"]
        output_h = observer_cfg.get("height", 480)
        output_w = observer_cfg.get("width", 832)

        # Build mapping from event_id → agent
        agent_map = {a.event_id: a for a in active_agents}

        # Build mapping from event_id → EventPointCloud for per-instance anchor masks.
        event_pc_map: dict[str, object] = {}
        for epc in self.world_state.list_event_pointclouds():
            event_pc_map[epc.event_id] = epc

        inst_frames: list[np.ndarray] = []
        inst_map: dict[str, np.ndarray] = {}
        for event_id in visible_event_ids:
            agent = agent_map.get(event_id)
            if agent is None or agent.state.current_anchor_frame is None:
                continue

            anchor = agent.state.current_anchor_frame  # (H, W, 3) uint8
            entities = agent.state.entities
            if not entities:
                continue

            # Determine which obj_ids are visible for this event.
            vis_obj_set = None
            if visible_instances and event_id in visible_instances:
                vis_obj_set = visible_instances[event_id]

            # Try using stored per-instance anchor masks from the point cloud.
            epc = event_pc_map.get(event_id)
            has_stored_masks = (
                epc is not None and epc.per_instance_anchor_masks is not None and len(epc.per_instance_anchor_masks) > 0
            )

            if has_stored_masks:
                # Use stored anchor masks — skip SAM3 re-segmentation.
                # per_instance_anchor_masks were computed on the I2V video's
                # frame 0, which is stored in epc.anchor_frame.  Use that
                # instead of agent.state.current_anchor_frame (which may
                # have been updated to a later frame).
                mask_anchor = epc.anchor_frame if epc.anchor_frame is not None else anchor
                for obj_id, mask in epc.per_instance_anchor_masks.items():
                    # Filter by visibility when per-instance data is available.
                    if vis_obj_set is not None and obj_id not in vis_obj_set:
                        continue
                    ref = self._mask_to_instance_ref(mask_anchor, mask, output_h, output_w)
                    if ref is not None:
                        inst_frames.append(ref)
            else:
                # Fallback: re-segment with SAM3 (no per-instance visibility filtering).
                anchor_pil = Image.fromarray(anchor)
                instance_masks: list[np.ndarray] = []
                for entity_name in entities:
                    instance_masks.extend(
                        self.sam3_segmenter.segment_instances(
                            video_path=[anchor_pil],
                            prompt=entity_name,
                            frame_index=0,
                            expected_frames=1,
                        )
                    )
                for mask in instance_masks:
                    ref = self._mask_to_instance_ref(anchor, mask, output_h, output_w)
                    if ref is not None:
                        inst_frames.append(ref)

            # Store last instance ref for per-event mapping (backward compat).
            if inst_frames:
                inst_map[event_id] = inst_frames[-1]

        if not inst_frames:
            return None, {}

        return np.stack(inst_frames, axis=0), inst_map  # [R_inst, H, W, 3]

    @staticmethod
    def _mask_to_instance_ref(
        anchor: np.ndarray,
        mask: np.ndarray,
        output_h: int,
        output_w: int,
        min_mask_ratio: float = 0.01,
    ) -> np.ndarray | None:
        """Convert a binary mask on an anchor frame to a centered instance reference.

        Returns:
            (output_h, output_w, 3) uint8 image with the masked entity centered
            on a black background, or None if the mask is empty or too small.
        """
        mask_u8 = mask.astype(np.uint8)
        ys, xs = np.where(mask_u8 > 0)
        if ys.size == 0 or xs.size == 0:
            return None

        # Skip masks that are too small relative to the anchor frame.
        mask_ratio = float(ys.size) / float(anchor.shape[0] * anchor.shape[1])
        if mask_ratio < min_mask_ratio:
            return None

        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1

        crop_img = anchor[y1:y2, x1:x2].copy()
        crop_mask = mask_u8[y1:y2, x1:x2]
        crop_fg = np.zeros_like(crop_img, dtype=np.uint8)
        crop_fg[crop_mask > 0] = crop_img[crop_mask > 0]

        crop_h, crop_w = crop_fg.shape[:2]
        scale = min(output_w / crop_w, output_h / crop_h)
        new_w = max(1, int(round(crop_w * scale)))  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.
        new_h = max(1, int(round(crop_h * scale)))  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.

        resized_fg = cv2.resize(crop_fg, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        resized_fg[resized_mask == 0] = 0

        centered = np.zeros((output_h, output_w, 3), dtype=np.uint8)
        off_x = (output_w - new_w) // 2
        off_y = (output_h - new_h) // 2
        centered[off_y : off_y + new_h, off_x : off_x + new_w] = resized_fg
        return centered

    def _update_frame_visibility(
        self,
        iter_idx: int,
        target_frame_indices: list[int],
        output_start: int,
    ) -> None:
        """Update per-frame visible point indices for reference frame retrieval.

        Args:
            iter_idx: Current iteration index.
            target_frame_indices: Global frame indices for target frames.
            output_start: Start index in all_generated_frames for this iteration.
        """
        points, _ = self.world_state.get_static()
        if points is None or self.geometry.poses_c2w is None:
            return

        observer_cfg = self.config["observer"]
        output_height = observer_cfg.get("height", 480)
        output_width = observer_cfg.get("width", 832)
        output_size = (output_height, output_width)

        intrinsics = self.geometry.intrinsics
        intr = intrinsics if intrinsics.ndim == 3 else np.tile(intrinsics[None], (len(self.geometry.poses_c2w), 1, 1))

        frames_to_process = []
        for local_idx, frame_idx in enumerate(target_frame_indices):
            if frame_idx < len(self.geometry.poses_c2w):
                # Event-centric observer stores all generated frames each round
                # with no overlap trimming. Keep visibility keys exactly aligned
                # with all_generated_frames / frame_dynamic_masks indices.
                global_frame_idx = output_start + local_idx
                frames_to_process.append((local_idx, frame_idx, global_frame_idx))

        # Use GPU for visibility computation when available.
        use_gpu = torch.cuda.is_available()
        if use_gpu and frames_to_process:
            pose_indices = [fi for _, fi, _ in frames_to_process]
            vis_results = _get_visible_points_and_coverage_gpu(
                points,
                self.geometry.poses_c2w,
                intr,
                pose_indices,
                output_size,
                device="cuda",
            )
            for (_, _, global_frame_idx), (_, visible_pts, _, _) in zip(frames_to_process, vis_results, strict=False):
                if len(visible_pts) > 0:
                    self.geometry.frame_visible_points[global_frame_idx] = visible_pts
        else:
            for _, frame_idx, global_frame_idx in frames_to_process:
                intr_idx = min(frame_idx, len(intr) - 1)
                visible_pts, _ = get_visible_points_for_frame(
                    points,
                    self.world_state.static_colors,
                    self.geometry.poses_c2w[frame_idx],
                    intr[intr_idx],
                    output_size,
                )
                if len(visible_pts) > 0:
                    self.geometry.frame_visible_points[global_frame_idx] = visible_pts

    @staticmethod
    def _compose_prompt(scene_text: str, fg_text: str) -> str:
        """Compose the prompt in the same format as LiveWorld training."""
        scene_text = (scene_text or "").strip()
        fg_text = (fg_text or "").strip()
        if not scene_text:
            raise ValueError("scene_text must be provided")
        if fg_text:
            return f"scene: {scene_text}; foreground: {fg_text}"
        return f"scene: {scene_text}"


def _resolve_data_path(value: str, config_dir: Path) -> str:
    """Resolve a relative data path by searching config dir and its ancestors.

    Tries *config_dir*, then walks up to 3 parent levels.  The first
    ancestor where ``ancestor / value`` exists wins.  Falls back to
    CWD-based resolution for backward compatibility with non-benchmark
    configs whose paths are relative to the project root.
    """
    if not value or os.path.isabs(value):
        return value
    search = config_dir
    for _ in range(4):  # config_dir + 3 ancestors
        candidate = search / value
        if candidate.exists():
            return str(candidate.resolve())
        search = search.parent
    # Fallback: resolve from CWD (backward compat).
    return os.path.join(os.getcwd(), value)


# Keys in the merged config that hold data file paths (not model paths).
_DATA_PATH_KEYS = ("geometry_file_name", "first_frame_image", "entities_file", "storyline_file")


def load_monitor_centric_config(config_path: str, system_config_path: str | None = None) -> dict:
    """Load YAML config with system defaults merging and path resolution.

    Args:
        config_path: Path to sample config YAML.
        system_config_path: Path to system config YAML. If None, reads
            ``system_config_file`` from the sample config (backward compat).

    Loading order (later overrides earlier):
    1. Load system config as defaults.
    2. Load sample config (sample-specific overrides) on top.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config_dir = config_path.parent

    # 1. Determine system config path.
    sample_cfg = OmegaConf.load(config_path)
    sample_cfg_dict = OmegaConf.to_container(sample_cfg, resolve=True)

    if system_config_path is None:
        # Backward compat: read from sample config (deprecated)
        system_config_path = sample_cfg_dict.get("system_config_file")
        if not system_config_path:
            raise KeyError(
                "system_config_path not provided. Pass --system-config to "
                f"scripts/infer.py. Sample config: {config_path}"
            )

    resolved_system_cfg_path = _resolve_data_path(str(system_config_path), config_dir)
    if not os.path.exists(resolved_system_cfg_path):
        raise FileNotFoundError(
            f"system_config not found: {system_config_path} (resolved to: {resolved_system_cfg_path})"
        )
    system_cfg = OmegaConf.load(resolved_system_cfg_path)

    # 2. Merge: sample on top of system defaults.
    merged = OmegaConf.merge(system_cfg, sample_cfg)
    config = OmegaConf.to_container(merged, resolve=True)
    config["system_config_file"] = resolved_system_cfg_path

    # 4. Auto-resolve relative data paths from config dir ancestors.
    for key in _DATA_PATH_KEYS:
        val = config.get(key)
        if val:
            config[key] = _resolve_data_path(val, config_dir)

    return config
