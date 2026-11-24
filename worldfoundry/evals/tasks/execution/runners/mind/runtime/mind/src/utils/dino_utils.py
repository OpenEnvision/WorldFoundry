import torch
from transformers import AutoImageProcessor, AutoModel


def load_dinov3_model(model_path, device="cuda:0"):
    dino_processor = AutoImageProcessor.from_pretrained(model_path)
    dino_model = AutoModel.from_pretrained(model_path)
    dino_model = dino_model.to(device)
    dino_model.eval()

    return dino_model, dino_processor


def extract_dinov3_features(
    frames: torch.Tensor, model=None, processor=None, device="cuda:0", batch_size=8
) -> torch.Tensor:
    """Extract L2-normalized DINOv3 patch features from video frames.

    Args:
        frames: Video frames, shape [f, c, h, w], range [0, 1].
        model: DINOv3 model from ``load_dinov3_model``.
        processor: DINOv3 image processor from ``load_dinov3_model``.
        device: Device the features are computed on.
        batch_size: Frames per forward pass.

    Returns:
        Patch features of shape [f, 196, 768].
    """
    if model is None or processor is None:
        model, processor = load_dinov3_model(device)

    f = frames.shape[0]
    features_list = []

    # DINOv3 expects inputs in [0, 1].
    with torch.inference_mode():
        for i in range(0, f, batch_size):
            batch_frames = frames[i : i + batch_size].to(device)  # [batch, c, h, w]

            # The processor applies the model normalization to the tensor directly.
            inputs = processor(images=batch_frames, return_tensors="pt").to(device)

            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.last_hidden_state  # [batch, 201, 768]

            # Patch tokens follow the CLS token and the four register tokens.
            patch_tokens = last_hidden_state[:, 5:, :]  # [batch, 196, 768]

            # L2 normalization
            patch_tokens_norm = torch.nn.functional.normalize(patch_tokens, p=2, dim=-1)

            features_list.append(patch_tokens_norm)

            del batch_frames

    features = torch.cat(features_list, dim=0)  # [f, 196, 768]

    return features
