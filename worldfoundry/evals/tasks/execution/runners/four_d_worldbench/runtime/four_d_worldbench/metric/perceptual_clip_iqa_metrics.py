"""
This has several CLIP-IQA Metrics that have been proposed by
    Exploring CLIP for Assessing the Look and Feel of Images.
    Jianyi Wang Kelvin C.K. Chan Chen Change Loy.
    AAAI 2023.

Ref url: https://github.com/IceClear/CLIP-IQA
Re-implmented by: Chaofeng Chen (https://github.com/chaofengc).
Modifications:
    - We assemble multiple prompts to improve the results of clipiqa model.

We use the IQA-Pytorch implementation:
https://iqa-pytorch.readthedocs.io/

prompts used
    [
        'Good image', 'bad image',
        'Sharp image', 'blurry image',
        'sharp edges', 'blurry edges',
        'High resolution image', 'low resolution image',
        'Noise-free image', 'noisy image',
    ]

RANGE: [0, 1] higher the better
"""

import json
import os

import cv2
from PIL import Image

from .base_metrics import IQAPytorchMetric

try:
    from metric.utils import load_dimension_info
except ImportError:
    from .utils import load_dimension_info


class CLIPImageQualityAssessmentMetric(IQAPytorchMetric):
    """CLIP-IQA"""

    def __init__(self, metric_name: str = "clipiqa") -> None:
        super().__init__(metric_name=metric_name)

    def _compute_scores(self, rendered_images: list[str | Image.Image]) -> float:
        imgs = self._process_image(rendered_images)

        scores = []
        for img in imgs:
            score: float = self._metric(img.unsqueeze(0)).item()
            scores.append(score)

        score = sum(scores) / len(scores)
        return score


class CLIPImageQualityAssessmentPlusMetric(CLIPImageQualityAssessmentMetric):
    """CLIP-IQA+"""

    def __init__(self) -> None:
        super().__init__(metric_name="clipiqa+")


class CLIPImageQualityAssessmentPlusRN50_512Metric(CLIPImageQualityAssessmentMetric):
    """CLIP-IQA+ ResNet-50"""

    def __init__(self) -> None:
        super().__init__(metric_name="clipiqa+_rn50_512")


class CLIPimageQualityAssessmentPlusVITL14_512Metric(CLIPImageQualityAssessmentMetric):
    """CLIP-IQA+ ViT-L14"""

    def __init__(self) -> None:
        super().__init__(metric_name="clipiqa+_vitL14_512")


def _extract_frames_to_dir(video_path: str, out_root: str, max_frames: int = 64) -> list[str]:
    """Extract frames from video to directory."""
    os.makedirs(out_root, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(out_root, base)
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


def compute_perceptual_clip_iqa_metrics(json_dir, device, submodules_dict, **kwargs):
    """
    Compute CLIP-IQA metrics for videos.

    Args:
        json_dir: Path to the dimension JSON file
        device: Device to run the model on (e.g., 'cuda:0')
        submodules_dict: Dictionary of submodules (not used currently)
        **kwargs: Additional arguments (model, dataset_json, metric_variant, etc.)

    Returns:
        tuple: (final_score, details_list)
    """
    dimension = os.path.splitext(os.path.basename(json_dir))[0]
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension=dimension, lang="en")

    # Resolve relative video paths using dataset base directory
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

    # Support different CLIP-IQA variants
    metric_variant = kwargs.get("metric_variant", "clipiqa")  # default to clipiqa
    if metric_variant == "clipiqa+":
        metric = CLIPImageQualityAssessmentPlusMetric()
    elif metric_variant == "clipiqa+_rn50_512":
        metric = CLIPImageQualityAssessmentPlusRN50_512Metric()
    elif metric_variant == "clipiqa+_vitL14_512":
        metric = CLIPimageQualityAssessmentPlusVITL14_512Metric()
    else:
        metric = CLIPImageQualityAssessmentMetric(metric_name=metric_variant)

    details = []
    scores: list[float] = []
    out_root = os.path.join(os.path.dirname(json_dir), "frames_cache")

    for item in prompt_dict_ls:
        for video_path in item.get("video_list", []) or []:
            frames = _extract_frames_to_dir(video_path, out_root)
            if len(frames) == 0:
                continue

            # Compute CLIP-IQA score for all frames
            score = metric._compute_scores(frames)
            scores.append(float(score))
            details.append(
                {
                    "video_path": video_path,
                    "num_frames": len(frames),
                    "clip_iqa_score": float(score),
                    "metric_variant": metric_variant,
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
                "metric_variant": metric_variant,
            },
            "video_details": details,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(detailed_output, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving JSON file: {e!s}")

    return final, details
