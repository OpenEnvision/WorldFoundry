"""Event pool management.

The pool owns event lifecycle and manages shared model instances
across all EventAgent objects. It also holds the shared prompts
(system prompt and evolution template) for all agents.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .entity_matcher import EntityMatcher
from .event_agent import EventAgent, EventPrompts
from .event_types import EventID, EventObservation, EventState, make_event_id
from .logger import logger


@dataclass
class EventPoolConfig:
    """Configuration for the event pool."""

    max_events: int
    # System prompt template for I2V prompt generation.
    # Variables: {entity}, {entities}, {iteration}
    system_prompt: str
    # Entity deduplication configuration.
    dedup_enabled: bool = True
    dedup_model_path: str = "ckpts/facebook--dinov3-vith16plus-pretrain-lvd1689m"
    dedup_similarity_threshold: float = 0.8
    # Scene registration: reuse if any instance pair similarity exceeds threshold.
    scene_entity_similarity_threshold: float = 0.45
    # Preset storyline: when provided, skip Qwen and use these I2V prompts.
    # Dict with "scene_text" (str) and "steps" (Dict[str, str] of fg_text per step).
    preset_storyline: dict[str, Any] | None = None


class EventPool:
    """Pool that stores EventAgent instances.

    EventPool is the sole owner of:
    - Entity matcher (DINOv3 for scene-entity similarity)
    - Shared prompts (system prompt)

    Video generation is handled by the pipeline via the shared Observer model.
    All EventAgent instances share resources through the pool.

    Scene-level semantics:
    - One EventAgent represents one dynamic scene camera, not one instance.
    - Registration reuses an existing scene if any detected instance has
      DINOv3 embedding similarity above threshold with any instance in
      the existing scene.
    """

    def __init__(
        self,
        config: EventPoolConfig,
        device: str,
        cpu_offload_dinov3: bool,
        cpu_offload_qwen: bool,
        default_horizon: int,
        default_fps: int,
        qwen_provider=None,
        entity_matcher: EntityMatcher | None = None,
    ) -> None:
        if config.max_events <= 0:
            raise ValueError("max_events must be positive")

        self.config = config
        self.device = device
        self.cpu_offload_dinov3 = cpu_offload_dinov3
        self.cpu_offload_qwen = cpu_offload_qwen
        self.default_horizon = default_horizon
        self.default_fps = default_fps

        # Callable returning the lazy-loaded Qwen model for I2V prompt generation.
        self._qwen_provider = qwen_provider

        # Preset storyline (skip Qwen when provided).
        self._preset_storyline = config.preset_storyline
        # Cached scene_text for Qwen runtime path (generated once, reused).
        self._cached_scene_text: str | None = None

        # Build shared prompts from config.
        self.prompts = EventPrompts(
            system_prompt=config.system_prompt,
        )

        # DINOv3 entity matcher: reuse from previous pool or lazy-load on first use.
        self._dedup_enabled = config.dedup_enabled
        self._dedup_model_path = config.dedup_model_path
        self._dedup_similarity_threshold = config.dedup_similarity_threshold
        self.entity_matcher: EntityMatcher | None = entity_matcher

        # Main storage: event_id -> EventAgent
        self._agents: dict[EventID, EventAgent] = {}

    @staticmethod
    def _normalize_entities(entities: list[str]) -> list[str]:
        uniq: set[str] = set()
        normalized: list[str] = []
        for ent in entities or []:
            ent_n = (ent or "").strip().lower()
            if not ent_n or ent_n in uniq:
                continue
            uniq.add(ent_n)
            normalized.append(ent_n)
        return normalized

    def _compute_scene_entity_features(
        self,
        detections,
        matcher: EntityMatcher | None,
    ) -> tuple[list[torch.Tensor], list[np.ndarray], list[np.ndarray], list[str]]:
        if matcher is None or not detections:
            return [], [], [], []
        embeddings: list[torch.Tensor] = []
        crops: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        labels: list[str] = []
        for det in detections:
            if det.cropped_mask is None:
                raise RuntimeError(
                    f"Detection '{det.name}' is missing cropped_mask; SAM mask is required for DINO similarity."
                )
            emb = matcher.compute_embedding(
                det.cropped_image,
                mask=det.cropped_mask,
            )
            if emb is None:
                raise RuntimeError(
                    f"Failed to compute DINO embedding for '{det.name}' with SAM mask. Check instance mask quality."
                )
            embeddings.append(emb)
            crops.append(det.cropped_image)
            masks.append((np.asarray(det.cropped_mask) > 0).astype(np.uint8))
            labels.append((det.name or "").strip().lower())
        return embeddings, crops, masks, labels

    def _scene_embedding_similarity(
        self,
        obs_embeddings: list[torch.Tensor],
        ref_embeddings: list[torch.Tensor],
        matcher: EntityMatcher | None,
        *,
        log_label: str = "",
    ) -> tuple[float, float, tuple[int, int] | None]:
        """Compute instance-level embedding similarity.

        Returns:
            (avg_sim, max_pair_sim, best_pair): symmetric best-match average,
            single highest pairwise similarity, and (obs_idx, ref_idx) for
            that max pair.
        """
        if matcher is None or not obs_embeddings or not ref_embeddings:
            return 0.0, 0.0, None
        # Full pairwise similarity matrix.
        n_obs, n_ref = len(obs_embeddings), len(ref_embeddings)
        sim_matrix = np.zeros((n_obs, n_ref), dtype=np.float32)
        for i, emb_obs in enumerate(obs_embeddings):
            for j, emb_ref in enumerate(ref_embeddings):
                sim_matrix[i, j] = matcher.compute_similarity(emb_obs, emb_ref)

        row_scores = [float(sim_matrix[i].max()) for i in range(n_obs)]
        col_scores = [float(sim_matrix[:, j].max()) for j in range(n_ref)]
        avg_sim = 0.5 * (float(np.mean(row_scores)) + float(np.mean(col_scores)))
        max_pair_sim = float(sim_matrix.max())
        best_flat_idx = int(sim_matrix.argmax())
        best_pair = (best_flat_idx // n_ref, best_flat_idx % n_ref)

        # Keep similarity logs concise: one summary per compared scene.
        prefix = f"  [{log_label}] " if log_label else "  "
        logger.info(f"{prefix}avg_sim={avg_sim:.3f}, max_pair_sim={max_pair_sim:.3f}")

        return avg_sim, max_pair_sim, best_pair

    def _merge_scene_features(
        self,
        old_embeddings: list[torch.Tensor],
        old_crops: list[np.ndarray],
        old_masks: list[np.ndarray],
        old_labels: list[str],
        new_embeddings: list[torch.Tensor],
        new_crops: list[np.ndarray],
        new_masks: list[np.ndarray],
        new_labels: list[str],
        matcher: EntityMatcher | None,
    ) -> tuple[list[torch.Tensor], list[np.ndarray], list[np.ndarray], list[str]]:
        if not old_embeddings:
            return (
                list(new_embeddings),
                list(new_crops),
                list(new_masks),
                list(new_labels),
            )
        merged: list[torch.Tensor] = list(old_embeddings)
        merged_crops: list[np.ndarray] = list(old_crops)
        merged_masks: list[np.ndarray] = list(old_masks)
        merged_labels: list[str] = list(old_labels)
        if matcher is None:
            merged.extend(new_embeddings)
            merged_crops.extend(new_crops)
            merged_masks.extend(new_masks)
            merged_labels.extend(new_labels)
            return merged, merged_crops, merged_masks, merged_labels
        for emb_new, crop_new, mask_new, label_new in zip(
            new_embeddings, new_crops, new_masks, new_labels, strict=False
        ):
            best = 0.0
            for emb_old in merged:
                best = max(best, matcher.compute_similarity(emb_new, emb_old))
            if best < self._dedup_similarity_threshold:
                merged.append(emb_new)
                merged_crops.append(crop_new)
                merged_masks.append(mask_new)
                merged_labels.append(label_new)
        return merged, merged_crops, merged_masks, merged_labels

    @staticmethod
    def _save_match_debug_pair(
        debug_dir: Path | None,
        compared_event_id: str,
        max_pair_sim: float,
        obs_idx: int,
        ref_idx: int,
        obs_rgb: np.ndarray | None,
        obs_mask: np.ndarray | None,
        ref_rgb: np.ndarray | None,
        ref_mask: np.ndarray | None,
    ) -> None:
        if debug_dir is None:
            return
        debug_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"vs_{compared_event_id[:12]}_sim{max_pair_sim:.3f}_obs{obs_idx:02d}_ref{ref_idx:02d}"
        if obs_rgb is not None:
            cv2.imwrite(
                str(debug_dir / f"{prefix}_obs_rgb.png"),
                cv2.cvtColor(obs_rgb, cv2.COLOR_RGB2BGR),
            )
        if obs_mask is not None:
            cv2.imwrite(
                str(debug_dir / f"{prefix}_obs_mask.png"),
                (np.asarray(obs_mask) > 0).astype(np.uint8) * 255,
            )
        if ref_rgb is not None:
            cv2.imwrite(
                str(debug_dir / f"{prefix}_ref_rgb.png"),
                cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR),
            )
        if ref_mask is not None:
            cv2.imwrite(
                str(debug_dir / f"{prefix}_ref_mask.png"),
                (np.asarray(ref_mask) > 0).astype(np.uint8) * 255,
            )
        if obs_rgb is not None and obs_mask is not None and obs_rgb.shape[:2] == np.asarray(obs_mask).shape[:2]:
            obs_overlay = obs_rgb.copy()
            m = np.asarray(obs_mask) > 0
            obs_overlay[m] = (0.65 * obs_overlay[m] + 0.35 * np.array([255, 64, 64])).astype(np.uint8)
            cv2.imwrite(
                str(debug_dir / f"{prefix}_obs_overlay.png"),
                cv2.cvtColor(obs_overlay, cv2.COLOR_RGB2BGR),
            )
        if ref_rgb is not None and ref_mask is not None and ref_rgb.shape[:2] == np.asarray(ref_mask).shape[:2]:
            ref_overlay = ref_rgb.copy()
            m = np.asarray(ref_mask) > 0
            ref_overlay[m] = (0.65 * ref_overlay[m] + 0.35 * np.array([255, 64, 64])).astype(np.uint8)
            cv2.imwrite(
                str(debug_dir / f"{prefix}_ref_overlay.png"),
                cv2.cvtColor(ref_overlay, cv2.COLOR_RGB2BGR),
            )

    def _ensure_entity_matcher(self) -> EntityMatcher | None:
        """Lazy-load the DINOv3 entity matcher on first use."""
        if self.entity_matcher is None and self._dedup_enabled:
            logger.info("Lazy-loading DINOv3 entity matcher...")
            self.entity_matcher = EntityMatcher(
                model_path=self._dedup_model_path,
                device=self.device,
                similarity_threshold=self._dedup_similarity_threshold,
            )
        return self.entity_matcher

    def generate_i2v_prompt(
        self,
        anchor_frame: np.ndarray,
        entities: list[str],
        iteration: int,
    ) -> tuple[str, str]:
        """Generate scene_text and fg_text for event I2V generation.

        Returns (scene_text, fg_text) to be composed via _compose_prompt()
        into the "scene: ...; foreground: ..." format used by the observer.

        Args:
            anchor_frame: Current anchor frame (H, W, 3), uint8 RGB.
            entities: Entity names for this event.
            iteration: Current iteration count.

        Returns:
            (scene_text, fg_text) tuple.
        """
        # Use preset storyline if provided (skip Qwen entirely).
        if self._preset_storyline is not None:
            step_key = str(iteration)
            scene_text = self._preset_storyline["scene_text"]
            fg_text = self._preset_storyline["steps"][step_key]
            return scene_text, fg_text

        entity = entities[0] if entities else "object"
        entities_text = ", ".join(entities) if entities else "object"

        if self._qwen_provider is None:
            # Fallback: use the formatted system_prompt directly as fg_text.
            fg_text = self.prompts.system_prompt.format(
                entity=entity,
                entities=entities_text,
                iteration=iteration,
            )
            scene_text = (
                "The static environment remains coherent, with background layers and "
                "occlusion boundaries shifting smoothly along the trajectory."
            )
            return scene_text, fg_text

        qwen = self._qwen_provider()

        # Save anchor frame to a temporary file for Qwen input.
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)  # noqa: SIM115 - Keeps the upstream file-handle lifetime.
        tmp_path = tmp.name
        tmp.close()
        try:
            cv2.imwrite(tmp_path, cv2.cvtColor(anchor_frame, cv2.COLOR_RGB2BGR))

            # Move Qwen to GPU if offloading.
            if self.cpu_offload_qwen and hasattr(qwen, "model"):
                qwen.model.to(self.device)

            # Generate scene_text once (event camera is fixed, scene doesn't change).
            if self._cached_scene_text is None:
                scene_prompt = (
                    "Describe only the static environment visible in this image in 1-2 concise sentences. "
                    f"EXCLUDE any foreground entities ({entities_text}) — do NOT mention them. "
                    "Focus on architecture, terrain, lighting, furniture, walls, floor, vegetation, etc. "
                    "Do NOT use camera motion words (forward/backward/left/right/pan/tilt). "
                    "Output the description string only, no JSON wrapping."
                )
                scene_text = qwen.generate_text(tmp_path, scene_prompt).strip()
                if scene_text.startswith(("{", '"')):
                    scene_text = scene_text.strip('"{}').strip()
                self._cached_scene_text = scene_text

            # Generate fg_text (entity motion description).
            instruction = self.prompts.system_prompt.format(
                entity=entity,
                entities=entities_text,
                iteration=iteration,
            )
            fg_text = qwen.generate_text(tmp_path, instruction)

            # Offload Qwen to CPU after use.
            if self.cpu_offload_qwen and hasattr(qwen, "model"):
                qwen.model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

        return self._cached_scene_text, fg_text.strip()

    def register_event(
        self,
        observation: EventObservation,
        debug_dir: str | None = None,
    ) -> EventAgent | None:
        """Register or reuse events from one observation frame (instance-level).

        Each detected instance is independently matched against existing events.
        Matched instances are merged into their best-matching event; unmatched
        instances are grouped together into a new event.

        Returns the first agent that was created or updated (for backward compat),
        or None if nothing was registered.
        """
        if observation.frame is None:
            raise ValueError("observation.frame (full scene) must be provided for I2V anchor")

        entities = self._normalize_entities(observation.entities)
        if not entities:
            return None

        detections = observation.detections
        if detections is None:
            detections = [observation.detection] if observation.detection is not None else []

        matcher = self._ensure_entity_matcher()
        if self.cpu_offload_dinov3 and matcher is not None:
            matcher.to_device()
        try:
            obs_embeddings, obs_crops, obs_masks, obs_labels = self._compute_scene_entity_features(detections, matcher)
            _debug_path = Path(debug_dir) if debug_dir else None
            logger.info(
                f"  Instance matching: {len(detections)} detections, "
                f"{len(obs_embeddings)} embeddings, "
                f"entities={entities}"
            )

            active_agents = self.list_active()
            logger.info(f"  Comparing against {len(active_agents)} active event(s)")

            entity_thr = self.config.scene_entity_similarity_threshold

            # Per-instance matching: for each obs instance, find best matching event.
            # matched_instances[i] = (agent, best_sim) or None
            matched_instances: list[tuple[EventAgent, float] | None] = [None for _ in range(len(obs_embeddings))]

            for agent in active_agents:
                state = agent.state
                ref_embeddings = state.scene_entity_embeddings or []
                if not ref_embeddings or matcher is None:
                    continue

                # Compute similarity of each obs instance against this event's instances.
                for i, emb_obs in enumerate(obs_embeddings):
                    best_sim_for_i = 0.0
                    for emb_ref in ref_embeddings:
                        sim = matcher.compute_similarity(emb_obs, emb_ref)
                        best_sim_for_i = max(best_sim_for_i, sim)

                    if best_sim_for_i >= entity_thr:
                        prev = matched_instances[i]
                        if prev is None or best_sim_for_i > prev[1]:
                            matched_instances[i] = (agent, best_sim_for_i)

                    logger.info(
                        f"    '{obs_labels[i]}' vs {agent.event_id[:12]}: "
                        f"best_sim={best_sim_for_i:.3f} "
                        f"(thr={entity_thr:.3f}) -> "
                        f"{'MATCH' if best_sim_for_i >= entity_thr else 'no match'}"
                    )

            # Group matched instances by their target agent.
            agent_to_indices: dict[str, list[int]] = {}
            unmatched_indices: list[int] = []
            for i, match in enumerate(matched_instances):
                if match is not None:
                    agent, sim = match
                    agent_to_indices.setdefault(agent.event_id, []).append(i)
                else:
                    unmatched_indices.append(i)

            first_result: EventAgent | None = None

            # Merge matched instances into their respective events.
            for eid, indices in agent_to_indices.items():
                agent = self._agents[eid]
                state = agent.state
                matched_entities = self._normalize_entities([obs_labels[i] for i in indices])
                matched_embs = [obs_embeddings[i] for i in indices]
                matched_crops = [obs_crops[i] for i in indices]
                matched_masks = [obs_masks[i] for i in indices]
                matched_labels = [obs_labels[i] for i in indices]

                state.entities = self._normalize_entities((state.entities or []) + matched_entities)
                (
                    state.scene_entity_embeddings,
                    state.scene_entity_crops,
                    state.scene_entity_masks,
                    state.scene_entity_labels,
                ) = self._merge_scene_features(
                    old_embeddings=state.scene_entity_embeddings or [],
                    old_crops=state.scene_entity_crops or [],
                    old_masks=state.scene_entity_masks or [],
                    old_labels=state.scene_entity_labels or [],
                    new_embeddings=matched_embs,
                    new_crops=matched_crops,
                    new_masks=matched_masks,
                    new_labels=matched_labels,
                    matcher=matcher,
                )
                if state.entity_embedding is None and state.scene_entity_embeddings:
                    state.entity_embedding = state.scene_entity_embeddings[0]
                sims = [matched_instances[i][1] for i in indices]
                logger.info(f"  Reuse event {eid[:12]} <- {matched_entities} (sims={[f'{s:.3f}' for s in sims]})")
                if first_result is None:
                    first_result = agent

            # Create new event for unmatched instances.
            if unmatched_indices:
                new_entities = self._normalize_entities([obs_labels[i] for i in unmatched_indices])
                if not new_entities:
                    new_entities = (
                        self._normalize_entities([entities[i] for i in unmatched_indices if i < len(entities)])
                        or entities
                    )
                new_embs = [obs_embeddings[i] for i in unmatched_indices]
                new_crops = [obs_crops[i] for i in unmatched_indices]
                new_masks = [obs_masks[i] for i in unmatched_indices]
                new_labels = [obs_labels[i] for i in unmatched_indices]

                if len(self._agents) >= self.config.max_events:
                    logger.warning(f"  Event capacity reached; skip new event for: {', '.join(new_entities)}")
                else:
                    # Use caller-provided event_id when all instances are unmatched
                    # (first registration); generate new id for partial unmatched.
                    if not agent_to_indices:
                        event_id = observation.event_id
                    else:
                        event_id = make_event_id(new_entities, observation.pose_c2w)
                    if event_id in self._agents:
                        logger.info(f"  Reuse existing event id {event_id[:12]} (already registered)")
                        agent = self._agents[event_id]
                    else:
                        state = EventState(
                            status="active",
                            anchor_pose_c2w=observation.pose_c2w,
                            entities=new_entities,
                            current_anchor_frame=observation.frame,
                            entity_embedding=new_embs[0] if new_embs else None,
                            scene_entity_embeddings=new_embs,
                            scene_entity_crops=new_crops,
                            scene_entity_masks=new_masks,
                            scene_entity_labels=new_labels,
                        )
                        agent = EventAgent(
                            event_id=event_id,
                            state=state,
                            device=self.device,
                            prompts=self.prompts,
                            default_horizon=self.default_horizon,
                            default_fps=self.default_fps,
                        )
                        self._agents[event_id] = agent
                        logger.info(f"  Register new event {event_id[:12]} <- {new_entities}")
                    if first_result is None:
                        first_result = agent

            return first_result
        finally:
            if self.cpu_offload_dinov3 and matcher is not None:
                matcher.offload_to_cpu()

    def list_active(self) -> list[EventAgent]:
        """Return all active event agents."""
        return [agent for agent in self._agents.values() if agent.state.status == "active"]

    def count(self) -> int:
        """Return total number of registered events."""
        return len(self._agents)
