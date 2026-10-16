"""Astra's bounded latent history and absolute camera progression."""

from typing import Any

import torch

from worldfoundry.core.memory import BaseMemory


class AstraMemory(BaseMemory):
    """Keep the initial latent and a recent window without rewinding the camera."""

    def __init__(self, capacity: int = 49, **kwargs: Any) -> None:
        super().__init__(capacity=capacity, **kwargs)
        self.capacity: int = capacity
        self.history_latents: torch.Tensor | None = None  # [C, T, H, W]
        self.total_latent_frames = 0

    def reset(self) -> None:
        """Release this session's latents, records and absolute progress."""
        self.history_latents = None
        self.total_latent_frames = 0
        self.reset_records()

    def record(self, new_latents_squeezed: torch.Tensor, **kwargs: Any) -> None:  # ty: ignore[invalid-method-override]
        """Append [C, T_new, H, W] latents and advance the absolute frame count."""
        if self.history_latents is None:
            self.history_latents = new_latents_squeezed
        else:
            self.history_latents = torch.cat([self.history_latents, new_latents_squeezed], dim=1)
        self.total_latent_frames += new_latents_squeezed.shape[1]
        self.manage()

    def manage(self, **kwargs: Any) -> None:
        """Keep the first frame and latest frames, or reset the whole session."""
        if kwargs.get("action") == "reset":
            self.reset()
        elif self.history_latents is not None and self.history_latents.shape[1] > self.capacity:
            first_frame = self.history_latents[:, 0:1]
            recent_start = self.history_latents.shape[1] - self.capacity + 1
            self.history_latents = torch.cat([first_frame, self.history_latents[:, recent_start:]], dim=1)

    def select(  # ty: ignore[invalid-method-override]
        self,
        target_frames_to_generate: int,
        camera_embedding_full: torch.Tensor,
        start_frame: int,
        modality_type: str,
    ) -> dict[str, Any]:
        """Build FramePack context using absolute camera rows and retained latents."""
        if self.history_latents is None:
            raise ValueError("Astra memory requires initial latents before selection.")
        return self.prepare_framepack_sliding_window_with_camera_moe(
            self.history_latents,
            target_frames_to_generate,
            camera_embedding_full,
            start_frame,
            modality_type,
            self.capacity,
            total_latent_frames=self.total_latent_frames,
        )

    def prepare_framepack_sliding_window_with_camera_moe(
        self,
        history_latents: torch.Tensor,
        target_frames_to_generate: int,
        camera_embedding_full: torch.Tensor,
        start_frame: int,
        modality_type: str,
        max_history_frames: int = 49,
        *,
        total_latent_frames: int | None = None,
    ) -> dict[str, Any]:
        """Pack local latent indices while camera rows follow episode progress.

        Camera embeddings are already aligned by the pipeline. Standalone calls
        without ``total_latent_frames`` describe an untruncated history.
        """
        C, T, H, W = history_latents.shape
        current_length = T if total_latent_frames is None else total_latent_frames
        total_indices_length = 1 + 16 + 2 + 1 + target_frames_to_generate
        indices = torch.arange(0, total_indices_length)
        split_sizes = [1, 16, 2, 1, target_frames_to_generate]
        (
            clean_latent_indices_start,
            clean_latent_4x_indices,
            clean_latent_2x_indices,
            clean_latent_1x_indices,
            latent_indices,
        ) = indices.split(split_sizes, dim=0)
        clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=0)

        needed_camera_rows = max(total_indices_length, current_length + target_frames_to_generate)
        if camera_embedding_full.shape[0] < needed_camera_rows:
            padding = camera_embedding_full.new_zeros(
                needed_camera_rows - camera_embedding_full.shape[0], camera_embedding_full.shape[1]
            )
            camera_embedding_full = torch.cat([camera_embedding_full, padding], dim=0)

        combined_camera = camera_embedding_full.new_zeros(total_indices_length, camera_embedding_full.shape[1])
        available_frames = min(T, 19)
        start_pos = 19 - available_frames
        if available_frames:
            history_positions = torch.arange(
                current_length - available_frames, current_length, device=camera_embedding_full.device
            )
            # A short retained window includes the pinned first frame.
            if T < current_length and available_frames == T:
                history_positions[0] = 0
            combined_camera[start_pos:19] = camera_embedding_full[history_positions]
        combined_camera[19 : 19 + target_frames_to_generate] = camera_embedding_full[
            current_length : current_length + target_frames_to_generate
        ]
        combined_camera[:, -1] = 0.0
        combined_camera[start_pos:19, -1] = 1.0

        clean_latents_combined = history_latents.new_zeros(C, 19, H, W)
        if available_frames:
            clean_latents_combined[:, start_pos:] = history_latents[:, -available_frames:]
        clean_latents_4x = clean_latents_combined[:, 0:16]
        clean_latents_2x = clean_latents_combined[:, 16:18]
        clean_latents_1x = clean_latents_combined[:, 18:19]
        start_latent = history_latents[:, 0:1] if T else history_latents.new_zeros(C, 1, H, W)
        clean_latents = torch.cat([start_latent, clean_latents_1x], dim=1)

        return {
            "latent_indices": latent_indices,
            "clean_latents": clean_latents,
            "clean_latents_2x": clean_latents_2x,
            "clean_latents_4x": clean_latents_4x,
            "clean_latent_indices": clean_latent_indices,
            "clean_latent_2x_indices": clean_latent_2x_indices,
            "clean_latent_4x_indices": clean_latent_4x_indices,
            "camera_embedding": combined_camera,
            "modality_type": modality_type,
            "current_length": current_length,
            "next_length": current_length + target_frames_to_generate,
        }
