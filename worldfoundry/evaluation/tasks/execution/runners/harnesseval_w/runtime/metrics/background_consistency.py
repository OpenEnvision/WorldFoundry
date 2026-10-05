"""Background consistency — VBench-aligned CLIP feature similarity."""
from worldfoundry.base_models.perception_core.general_perception.openai_clip_runtime import clip
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from .base import BaseMetric
from .weight_utils import get_weights_dir, get_weight_file


class BackgroundConsistencyMetric(BaseMetric):
    def __init__(self, device="cuda", vit_path="ViT-B/32", batch_size=32):
        super().__init__(device)
        self.batch_size = batch_size
        clip_dir = get_weights_dir("clip")
        self.model, self.preprocess = clip.load(get_weight_file("clip", vit_path.replace("/", "-") + ".pt"), device=self.device, download_root=clip_dir)
        self.model.eval()
        self.normalize = T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
        self.transform = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224), T.ToTensor(), self.normalize,
        ])

    @property
    def name(self):
        return "background_consistency"

    @torch.no_grad()
    def _extract_all_features(self, frames, preprocessed_tensors=None):
        all_feats = []
        if preprocessed_tensors is not None:
            for start in range(0, len(preprocessed_tensors), self.batch_size):
                batch = preprocessed_tensors[start:start + self.batch_size]
                batch = self.normalize(batch).to(self.device)
                features = self.model.encode_image(batch)
                all_feats.append(F.normalize(features.float(), dim=-1, p=2))
        else:
            for start in range(0, len(frames), self.batch_size):
                imgs = frames[start:start + self.batch_size]
                tensors = torch.stack([self.transform(img) for img in imgs]).to(self.device)
                features = self.model.encode_image(tensors)
                all_feats.append(F.normalize(features.float(), dim=-1, p=2))
        return torch.cat(all_feats, dim=0)

    def compute(self, frames, first_frame=None, prompt=None, **kwargs):
        """VBench-aligned: (prev_sim + first_sim) / 2 averaged over all frames."""
        if len(frames) < 2:
            return {f"{self.name}_score": 1.0}
        preprocessed = kwargs.get("preprocessed_tensors")
        all_feats = self._extract_all_features(frames, preprocessed)
        first_feat = all_feats[0:1]
        similarities = []
        for i in range(1, len(all_feats)):
            sim_pre = max(0.0, F.cosine_similarity(all_feats[i-1:i], all_feats[i:i+1]).item())
            sim_fir = max(0.0, F.cosine_similarity(first_feat, all_feats[i:i+1]).item())
            similarities.append((sim_pre + sim_fir) / 2)
        return {f"{self.name}_score": float(np.mean(similarities))}
