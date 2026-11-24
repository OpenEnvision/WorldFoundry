import json
import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import VGG19_Weights, vgg19

from .base_metrics import BaseMetric
from .utils import load_dimension_info

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no GPU
normalization_mean = torch.tensor([0.485, 0.456, 0.406])
normalization_std = torch.tensor([0.229, 0.224, 0.225])

# desired depth layers to compute style/content losses :
style_layers_default = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

loader = transforms.Compose(
    [
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor(),
    ]
)  # transform it into a torch tensor


def image_loader(image_name, device):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# create a module to normalize input image so we can easily put it in a
# ``nn.Sequential``
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std


def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img, style_layers):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std)

    # just in order to have an iterable access to or list of style
    # losses
    style_losses = []

    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

        model.add_module(name, layer)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], StyleLoss):
            break

    model = model[: (i + 1)]

    return model, style_losses


class GramMatrixMetric(BaseMetric):
    """
    TODO: add description
    """

    def __init__(self) -> None:
        super().__init__()
        self._model = vgg19(weights=VGG19_Weights.DEFAULT).features.to(self._device).eval()
        self._normalization_mean = normalization_mean.to(self._device)
        self._normalization_std = normalization_std.to(self._device)
        self._style_layers_default = style_layers_default

    def _compute_scores(
        self,
        reference_image: str,
        rendered_images: list[str],
    ) -> float:
        img1 = image_loader(reference_image, self._device)
        style_scores = []
        for rendered_image in rendered_images:
            img2 = image_loader(rendered_image, self._device)

            model, style_losses = get_style_model_and_losses(
                self._model, self._normalization_mean, self._normalization_std, img1, img2, self._style_layers_default
            )
            input_img = img2.clone()
            model(input_img)
            style_score = 0
            for sl in style_losses:
                style_score += sl.loss
            style_scores.append(style_score.item())

        return sum(style_scores) / len(style_scores)


def _extract_frames_to_dir(video_path: str, out_root: str, max_frames: int = 64) -> list[str]:
    os.makedirs(out_root, exist_ok=True)

    # Use full path with replaced separators to create unique directory
    safe_path = video_path.replace("/", "_").replace("\\", "_").replace(":", "_")
    out_dir = os.path.join(out_root, safe_path)
    os.makedirs(out_dir, exist_ok=True)

    existing = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".png")])
    if existing:
        return existing

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    indices = list(range(total))
    if max_frames > 0 and total > max_frames:
        step = total / float(max_frames)
        indices = [int(i * step) for i in range(max_frames)]

    saved: list[str] = []
    idx = 0
    next_set = set(indices)
    cur = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cur in next_set:
            out_path = os.path.join(out_dir, f"frame_{idx:04d}.png")
            cv2.imwrite(out_path, frame)
            saved.append(out_path)
            idx += 1
        cur += 1
    cap.release()
    return saved


def compute_consistency_style(json_dir, device, submodules_dict, **kwargs):
    dimension = os.path.splitext(os.path.basename(json_dir))[0]
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension=dimension, lang="en")

    dataset_json = kwargs.get("dataset_json", "")
    dataset_base_dir = os.environ.get("DATASET_BASE_DIR", "")
    if not dataset_base_dir and dataset_json:
        parts = dataset_json.split("/condition_to_4D/")
        if len(parts) > 1:
            dataset_base_dir = parts[0]
    if dataset_base_dir:
        for item in prompt_dict_ls:
            resolved = []
            for vp in item.get("video_list", []):
                if not os.path.isabs(vp) and not os.path.exists(vp):
                    full = os.path.join(dataset_base_dir, vp)
                    resolved.append(full)
                else:
                    resolved.append(vp)
            item["video_list"] = resolved

    metric = GramMatrixMetric()
    details = []
    scores: list[float] = []
    out_root = os.path.join(os.path.dirname(json_dir), "frames_cache")

    for item in prompt_dict_ls:
        for video_path in item.get("video_list", []) or []:
            frames = _extract_frames_to_dir(video_path, out_root)
            if len(frames) < 2:
                continue

            segment_scores: list[float] = []
            segment_details = []
            segment_size = 10
            segment_stride = 5

            if len(frames) < segment_size:
                continue

            segment_starts = list(range(0, len(frames) - segment_size + 1, segment_stride))
            last_start = max(len(frames) - segment_size, 0)
            if last_start not in segment_starts:
                segment_starts.append(last_start)

            for start in segment_starts:
                end = start + segment_size
                reference_frame = frames[start]
                segment_rendered = frames[start + 1 : end]
                if not segment_rendered:
                    continue

                segment_score = metric._compute_scores(reference_frame, segment_rendered)
                segment_scores.append(float(segment_score))
                segment_details.append(
                    {
                        "reference_frame": reference_frame,
                        "start_index": start,
                        "frame_count": segment_size,
                        "style_consistency": float(segment_score),
                    }
                )

            if not segment_scores:
                continue

            video_score = sum(segment_scores) / len(segment_scores)
            scores.append(float(video_score))
            details.append(
                {
                    "video_path": video_path,
                    "num_frames": len(frames),
                    "reference_frame": segment_details[0]["reference_frame"] if segment_details else None,
                    "style_consistency": float(video_score),
                    "segment_scores": segment_details,
                }
            )

    final = float(sum(scores) / len(scores)) if scores else 0.0

    # Save detailed results JSON
    try:
        output_dir = os.path.dirname(json_dir)
        dim_name = os.path.splitext(os.path.basename(json_dir))[0]
        model = kwargs.get("model", "")
        dataset_json = kwargs.get("dataset_json", "")
        dataset_base = os.path.splitext(os.path.basename(dataset_json))[0] if dataset_json else "dataset"
        suffix = f"{dim_name}__{model}__{dataset_base}_results.json" if model else f"{dim_name}_results.json"
        output_file = os.path.join(output_dir, suffix)
        detailed_output = {
            "evaluation_summary": {
                "total_videos": len(details),
                "average_score": final,
            },
            "video_details": details,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(detailed_output, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving JSON file: {e!s}")

    return final, details
