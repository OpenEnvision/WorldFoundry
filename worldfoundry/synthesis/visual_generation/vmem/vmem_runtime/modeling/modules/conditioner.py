import kornia
import open_clip
import torch
from torch import nn

from worldfoundry.core.io.paths import resolve_local_checkpoint_file

DEFAULT_OPENCLIP_REPO = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


class CLIPConditioner(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, model_path: str = DEFAULT_OPENCLIP_REPO):
        super().__init__()
        try:
            pretrained = resolve_local_checkpoint_file(model_path, "open_clip_pytorch_model.bin")
        except FileNotFoundError:
            pretrained = resolve_local_checkpoint_file(model_path, "open_clip_model.safetensors")
        self.module = open_clip.create_model_and_transforms("ViT-H-14", pretrained=str(pretrained))[0]
        self.module.eval().requires_grad_(False)  # type: ignore
        self.register_buffer("mean", torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer("std", torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = kornia.geometry.resize(
            x,
            (224, 224),
            interpolation="bicubic",
            align_corners=True,
            antialias=True,
        )
        x = (x + 1.0) / 2.0
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        x = self.module.encode_image(x)
        return x
