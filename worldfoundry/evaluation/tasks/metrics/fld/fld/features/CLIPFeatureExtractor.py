import torch
import torchvision.transforms as transforms
from worldfoundry.base_models.perception_core.general_perception import openai_clip as clip
from fld.features.ImageFeatureExtractor import ImageFeatureExtractor


class CLIPFeatureExtractor(ImageFeatureExtractor):
    def __init__(self, save_path=None):
        self.name = "clip"

        super().__init__(save_path)

        self.features_size = 512
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.model, _ = clip.load("ViT-B/32", device="cuda")
        self.model.eval()

    def get_feature_batch(self, img_batch):
        with torch.no_grad():
            features = self.model.encode_image(img_batch)
        return features
