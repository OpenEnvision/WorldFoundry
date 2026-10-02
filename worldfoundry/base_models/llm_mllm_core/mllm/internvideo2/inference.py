from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from worldfoundry.core.configuration.python import load_python_config
from worldfoundry.core import (
    get_video_details,
    load_frames_from_video,
)

from .multi_modality.models import InternVideo2_Stage2
from worldfoundry.base_models.llm_mllm_core.mllm.umt.models.backbones.bert.tokenization_bert import (
    BertTokenizer,
)
from .multi_modality.models.backbones.internvideo2.pos_embed import (
    interpolate_pos_embed_internvideo2_new,
)
from .multi_modality.models.criterions import get_sim


_MODEL_REPOS = {
    "internvideo2_1b_stage2": (
        ("zhiqiulin/internvideo2_1b_stage2", "internvideo2_1b_stage2.pth"),
        ("OpenGVLab/InternVideo2-Stage2_1B-224p-f4", "internvideo2_1b_stage2.pth"),
    ),
}


def download_internvideo2(model_name, pretrained_name=None, cache_dir="./hf_cache/", device="cuda"):
    model_path = _resolve_checkpoint(model_name, cache_dir)
    config = load_python_config(
        Path(__file__).with_name("configs.py"),
        [
            "model.vision_encoder.pretrained",
            model_path,
            "pretrained_path",
            model_path,
        ],
    )
    model, tokenizer = _setup_model(config, device=device)
    model.eval()
    return model_path, model, tokenizer, config, create_transforms(224)


@torch.no_grad()
def evaluation(
    texts,
    image_paths,
    transforms,
    model,
    tokenizer,
    device,
    num_frames=4,
    max_txt_l=40,
):
    model.eval()
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


def extract_text_feats(texts, max_txt_l, tokenizer, model, device, return_ids=False):
    text_input = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_txt_l,
        return_tensors="pt",
    ).to(device)
    text_feat = model.encode_text(text_input)[0]
    if return_ids:
        return text_feat, text_input.attention_mask, text_input.input_ids
    return text_feat, text_input.attention_mask


def extract_vision_feats(image_paths, transforms, model, device, num_frames=4):
    image = []
    for data_path in image_paths:
        total_frames, _, _ = get_video_details(data_path)
        frame_count = min(num_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, frame_count, dtype=int)
        frames = load_frames_from_video(data_path, indices, "decord", True)
        frames = transforms(frames.permute(0, 3, 1, 2))
        image.append(frames)

    image_tensor = torch.stack(image, dim=0).to(device, non_blocking=True)
    image_feat, pooled_image_feat = model.encode_vision(image_tensor, test=True)
    return image_feat.unsqueeze(1), pooled_image_feat.unsqueeze(1)


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
    try:
        tokenizer = BertTokenizer.from_pretrained(
            config.model.text_encoder.pretrained, local_files_only=True
        )
    except Exception:
        tokenizer = BertTokenizer.from_pretrained(
            config.model.text_encoder.pretrained, local_files_only=False
        )
    model = InternVideo2_Stage2(config=config, tokenizer=tokenizer, is_pretrain=True).to(device)
    checkpoint = torch.load(config.pretrained_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "module" in checkpoint:
        state_dict = checkpoint["module"]
    else:
        state_dict = checkpoint
    if config.get("origin_num_frames", None) is not None:
        interpolate_pos_embed_internvideo2_new(
            state_dict, model.vision_encoder, orig_t_size=config.origin_num_frames
        )
    model.load_state_dict(state_dict, strict=False)
    return model, tokenizer


def _resolve_checkpoint(model_name: str, cache_dir: str | os.PathLike[str]) -> str:
    explicit_path = Path(model_name).expanduser()
    if explicit_path.is_file():
        return str(explicit_path)
    repo_candidates = _MODEL_REPOS.get(model_name, ((f"zhiqiulin/{model_name}", f"{model_name}.pth"),))
    for repo_id, filename in repo_candidates:
        local = _find_local_file(cache_dir, repo_id, filename)
        if local is not None:
            return str(local)
    last_error = None
    for repo_id, filename in repo_candidates:
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        except Exception as exc:
            last_error = exc
    raise FileNotFoundError(f"Could not resolve InternVideo2 checkpoint {model_name}") from last_error


def _find_local_file(
    cache_dir: str | os.PathLike[str], repo_id: str, filename: str
) -> Path | None:
    root = Path(cache_dir).expanduser()
    repo_dir = repo_id.replace("/", "--")
    candidates: Iterable[Path] = (
        root / filename,
        root / repo_dir / filename,
        root / model_name_from_filename(filename) / filename,
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 1000:
            return candidate
    if root.is_dir():
        for candidate in root.glob(f"**/{filename}"):
            if candidate.is_file() and candidate.stat().st_size > 1000:
                return candidate
    return None


def model_name_from_filename(filename: str) -> str:
    return filename[:-4] if filename.endswith(".pth") else filename
