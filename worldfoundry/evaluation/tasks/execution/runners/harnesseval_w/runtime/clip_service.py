"""Minimal local CLIP image embedding backend used by HarnessEval skills."""

from __future__ import annotations

import io
from pathlib import Path


DEFAULT_MODEL = "ViT-B/32"
OFFICIAL_MODEL_FINGERPRINTS = {
    "ViT-B/32": "sha256:40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
}


class OpenAIClipBackend:
    """Lazy-loadable OpenAI CLIP image encoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        download_root: Path | None = None,
    ) -> None:
        from worldfoundry.base_models.perception_core.general_perception.openai_clip_runtime import clip
        import torch

        self.model_name = model_name
        self.model_fingerprint = OFFICIAL_MODEL_FINGERPRINTS.get(
            model_name, f"openai-clip:{model_name}"
        )
        self.device = device
        self._torch = torch
        self._image = __import__("PIL.Image", fromlist=["Image"])
        from .metrics.weight_utils import get_weight_file

        self.model, self.preprocess = clip.load(
            get_weight_file("clip", model_name.replace("/", "-") + ".pt", download_root),
            device=device,
            download_root=str(download_root) if download_root else None,
        )
        self.model.eval()

    def embed(self, images: list[bytes]) -> list[list[float]]:
        tensors = []
        for content in images:
            with self._image.open(io.BytesIO(content)) as image:
                tensors.append(self.preprocess(image.convert("RGB")))
        batch = self._torch.stack(tensors).to(self.device)
        with self._torch.inference_mode():
            features = self.model.encode_image(batch).float()
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.cpu().tolist()
