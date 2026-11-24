"""Entity matcher using DINOv3 embeddings for event deduplication.

This module provides similarity matching between detected entities using
DINOv3 foreground embeddings. It's used to determine if a newly detected
entity is the same as one already tracked in the EventPool.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class EntityMatcher:
    """Matcher for entity deduplication using DINOv3 embeddings.

    Uses foreground-only embeddings to compute similarity between entity images.
    SAM masks are required.
    """

    def __init__(
        self,
        model_path: str = "ckpts/facebook--dinov3-vith16plus-pretrain-lvd1689m",
        device: str = "cuda",
        similarity_threshold: float = 0.8,
    ) -> None:
        """Initialize the matcher with DINOv3 model.

        Args:
            model_path: Path to pretrained DINOv3 model.
            device: Device to run inference on.
            similarity_threshold: Threshold above which entities are considered the same.
        """
        self.device = device
        self.similarity_threshold = similarity_threshold

        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model.to(device)
        self.model.eval()

    def compute_embedding(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Compute foreground-only embedding for an entity image.

        Args:
            image: Entity RGB image (H, W, 3), uint8.
            mask: SAM foreground mask aligned with image (H, W), bool/uint8.

        Returns:
            Normalized embedding tensor of shape (hidden_dim,).
        """
        # Convert to PIL Image; skip degenerate crops.
        if image.size == 0 or image.shape[0] < 2 or image.shape[1] < 2:
            return None
        pil_image = Image.fromarray(image)

        # Process image through DINOv3.
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            # last_hidden_state: (1, num_patches+1, hidden_dim)
            # Skip first 5 tokens (CLS + register tokens for DINOv3).
            patch_tokens = outputs.last_hidden_state[0, 5:]

        num_patches = patch_tokens.shape[0]
        _hidden_dim = patch_tokens.shape[1]
        grid_size = int(num_patches**0.5)

        # Require SAM mask for foreground token selection.
        if mask is None:
            return None

        # Build foreground mask on patch grid.
        fg_mask = np.asarray(mask)
        if fg_mask.ndim == 3:
            fg_mask = fg_mask[..., 0]
        if fg_mask.shape[:2] != image.shape[:2]:
            fg_mask = cv2.resize(
                fg_mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask_small = cv2.resize(
            (fg_mask > 0).astype(np.uint8),
            (grid_size, grid_size),
            interpolation=cv2.INTER_NEAREST,
        )
        patch_mask = mask_small > 0

        mask_tensor = torch.from_numpy(patch_mask).flatten().to(self.device)

        # Handle empty foreground — return None sentinel so callers can skip.
        if mask_tensor.sum() == 0:
            return None

        # Average only foreground patch tokens.
        foreground_tokens = patch_tokens[mask_tensor]
        embedding = foreground_tokens.mean(dim=0)

        # Normalize for cosine similarity.
        return torch.nn.functional.normalize(embedding, dim=0)

    def compute_similarity(
        self,
        embedding1: torch.Tensor | None,
        embedding2: torch.Tensor | None,
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First normalized embedding (or None if invalid).
            embedding2: Second normalized embedding (or None if invalid).

        Returns:
            Cosine similarity score in [-1, 1]. Returns 0.0 if either is None.
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        return torch.dot(embedding1, embedding2).item()

    def is_same_entity(
        self,
        new_image: np.ndarray,
        existing_embedding: torch.Tensor | None,
        new_mask: np.ndarray | None = None,
    ) -> tuple[bool, float]:
        """Check if a new image matches an existing entity.

        Args:
            new_image: New entity image on black/white background.
            existing_embedding: Precomputed embedding of existing entity.
            new_mask: SAM foreground mask aligned with ``new_image``.

        Returns:
            Tuple of (is_same, similarity_score).
        """
        new_embedding = self.compute_embedding(new_image, mask=new_mask)
        similarity = self.compute_similarity(new_embedding, existing_embedding)
        return similarity >= self.similarity_threshold, similarity

    def find_matching_entity(
        self,
        new_image: np.ndarray,
        existing_embeddings: dict[str, torch.Tensor | None],
        new_mask: np.ndarray | None = None,
    ) -> tuple[str | None, float]:
        """Find if a new image matches any existing entity.

        Args:
            new_image: New entity image on black/white background.
            existing_embeddings: Dict mapping event_id to precomputed embedding.
            new_mask: SAM foreground mask aligned with ``new_image``.

        Returns:
            Tuple of (matching_event_id or None, best_similarity_score).
            Returns (None, best_score) if no match above threshold.
        """
        if not existing_embeddings:
            return None, 0.0

        new_embedding = self.compute_embedding(new_image, mask=new_mask)
        if new_embedding is None:
            return None, 0.0

        best_match_id = None
        best_similarity = 0.0

        for event_id, existing_emb in existing_embeddings.items():
            similarity = self.compute_similarity(new_embedding, existing_emb)
            if similarity > best_similarity:
                best_similarity = similarity
                if similarity >= self.similarity_threshold:
                    best_match_id = event_id

        return best_match_id, best_similarity

    def offload_to_cpu(self) -> None:
        """Move model to CPU to free GPU memory."""
        self.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to_device(self) -> None:
        """Move model back to configured device."""
        self.model.to(self.device)
