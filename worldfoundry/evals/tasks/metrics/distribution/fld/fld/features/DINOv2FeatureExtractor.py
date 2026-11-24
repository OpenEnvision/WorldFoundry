from typing import TYPE_CHECKING

import torch
import torchvision.transforms as transforms

if TYPE_CHECKING:
    from .ImageFeatureExtractor import ImageFeatureExtractor
else:
    from fld.features.ImageFeatureExtractor import ImageFeatureExtractor


class DINOv2FeatureExtractor(ImageFeatureExtractor):
    def __init__(self, save_path=None):
        self.name = "dinov2"
        super().__init__(save_path)

        self.features_size = 768
        # From https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py#L44
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.model.eval()
        self.model.to("cuda")

    def get_feature_batch(self, img_batch):
        with torch.no_grad():
            embeddings = self.model(img_batch)
        return embeddings
