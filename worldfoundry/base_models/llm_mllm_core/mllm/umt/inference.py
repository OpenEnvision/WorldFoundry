from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import BertTokenizer

from worldfoundry.core.configuration.python import load_python_config
from worldfoundry.core import (
    get_video_details,
    load_frames_from_video,
)

from .models import UMT
from .models.criterions import get_sim


def download_umt(model_name, pretrained_ckpt_name, cache_dir="./hf_cache/", device="cuda"):
    model_path = _resolve_checkpoint(model_name, cache_dir)
    pretrained_ckpt_path = _resolve_checkpoint(pretrained_ckpt_name, cache_dir)
    config = load_python_config(
        Path(__file__).with_name(f"{model_name}.py"),
        [
            "model.vision_encoder.pretrained",
            pretrained_ckpt_path,
            "pretrained_path",
            model_path,
        ],
    )
    if not config.evaluate:
        raise ValueError("UMT metric config must be evaluation mode")
    model, tokenizer = _setup_model(config, device=device)
    model.eval()
    return model_path, pretrained_ckpt_path, model, tokenizer, config, create_transforms(224)


@torch.no_grad()
def evaluation(
    texts,
    image_paths,
    transforms,
    model,
    tokenizer,
    device,
    num_frames=4,
    max_txt_l=32,
):
    text_feats, text_atts = extract_text_feats(texts, max_txt_l, tokenizer, model, device)
    image_feats, pooled_image_feats = extract_vision_feats(
        image_paths, transforms, model, device, num_frames=num_frames
    )
    pooled_image_feats = pooled_image_feats.to(device, non_blocking=True)
    i2t_scores, _ = get_sim(
        model.vision_proj(pooled_image_feats), model.text_proj(text_feats[:, 0])
    )

    encoder_output = image_feats[:, 0].to(device, non_blocking=True)
    encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
        device, non_blocking=True
    )
    output = model.get_text_encoder()(
        encoder_embeds=text_feats,
        attention_mask=text_atts,
        encoder_hidden_states=encoder_output,
        encoder_attention_mask=encoder_att,
        return_dict=True,
        mode="fusion",
    )
    itm_scores = model.itm_head(output.last_hidden_state[:, 0])[:, 1]
    return itm_scores, i2t_scores.diagonal()


def extract_text_feats(texts, max_txt_l, tokenizer, model, device):
    text_input = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_txt_l,
        return_tensors="pt",
    ).to(device)
    return model.encode_text(text_input)[0], text_input.attention_mask


def extract_vision_feats(image_paths, transforms, model, device, num_frames=4):
    image = []
    for data_path in image_paths:
        total_frames, _, _ = get_video_details(data_path)
        frame_count = min(num_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, frame_count, dtype=int)
        frames = load_frames_from_video(data_path, indices, "decord", True)
        image.append(transforms(frames.permute(0, 3, 1, 2)))
    image_tensor = torch.stack(image, dim=0).to(device, non_blocking=True)
    image_feat, pooled_image_feat = model.encode_vision(image_tensor, test=True)
    return image_feat.unsqueeze(1), pooled_image_feat


def create_transforms(image_res=224):
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    return transforms.Compose(
        [
            transforms.Resize((image_res, image_res), interpolation=InterpolationMode.BICUBIC),
            transforms.Lambda(lambda x: x.float().div(255.0)),
            normalize,
        ]
    )


def _setup_model(config, device="cuda"):
    tokenizer = BertTokenizer.from_pretrained(config.model.text_encoder.pretrained)
    model = UMT(config=config, tokenizer=tokenizer, is_pretrain=False).to(device)
    checkpoint = torch.load(config.pretrained_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "module" in checkpoint:
        state_dict = checkpoint["module"]
    else:
        state_dict = checkpoint

    if not config.evaluate or config.get("zero_shot", False):
        for key in list(state_dict.keys()):
            if "bert" in key:
                state_dict[key.replace("bert.", "")] = state_dict[key]
                del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    return model, tokenizer


def _resolve_checkpoint(name: str, cache_dir: str | os.PathLike[str]) -> str:
    filename = f"{name}.pth"
    repo_id = f"zhiqiulin/{name}"
    local = _find_local_file(cache_dir, repo_id, filename)
    if local is not None:
        return str(local)
    return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)


def _find_local_file(
    cache_dir: str | os.PathLike[str], repo_id: str, filename: str
) -> Path | None:
    root = Path(cache_dir).expanduser()
    repo_dir = repo_id.replace("/", "--")
    candidates = (
        root / filename,
        root / repo_dir / filename,
        root / filename[:-4] / filename,
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 1000:
            return candidate
    if root.is_dir():
        for candidate in root.glob(f"**/{filename}"):
            if candidate.is_file() and candidate.stat().st_size > 1000:
                return candidate
    return None
