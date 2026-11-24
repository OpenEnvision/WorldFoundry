# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp

import torch
from PIL import Image
from tqdm import tqdm

from .src.utils.generation_utils import generate, multimodal_decode
from .src.utils.input_utils import build_image
from .src.utils.painting_utils import ProtoWriter


def inference(
    cfg,
    model,
    tokenizer,
    vq_model,
):
    save_path = cfg.save_path

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f"{save_path}/proto", exist_ok=True)
    proto_writer = ProtoWriter()

    for name, question in tqdm(cfg.prompts, total=len(cfg.prompts)):
        if osp.exists(f"{save_path}/proto/{name}.pb"):
            print(f"[WARNING] Result already exists, skipping {name}", flush=True)
            continue

        torch.cuda.empty_cache()

        reference_image = None
        if not isinstance(question, str):
            if isinstance(question["reference_image"], list):
                print(f"[INFO] {len(question['reference_image'])} reference images are provided")
                reference_image = []
                for img in question["reference_image"]:
                    reference_image.append(Image.open(img).convert("RGB"))
            else:
                print("[INFO] 1 reference image is provided")
                reference_image = Image.open(question["reference_image"]).convert("RGB")
            question = question["prompt"]
        else:
            print("[INFO] No reference image is provided")

        proto_writer.clear()
        proto_writer.extend([["question", question]])
        if reference_image is not None:
            if isinstance(reference_image, list):
                for _idx, img in enumerate(reference_image):
                    proto_writer.extend([["reference_image", img]])
            else:
                proto_writer.extend([["reference_image", reference_image]])

        success = True
        prompt = cfg.template.format(question=question)

        print(f"[INFO] Handling prompt: {prompt}")
        if reference_image is not None:
            if isinstance(reference_image, list):
                image_str = ""
                for img in reference_image:
                    image_str += build_image(img, cfg, tokenizer, vq_model)
            else:
                image_str = build_image(reference_image, cfg, tokenizer, vq_model)
            prompt = prompt.replace("<|IMAGE|>", image_str)
            unc_prompt = cfg.unc_prompt.replace("<|IMAGE|>", image_str)
        else:
            unc_prompt = cfg.unc_prompt

        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

        if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
            BOS = torch.Tensor([[cfg.special_token_ids["BOS"]]], device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([BOS, input_ids], dim=1)

        unconditional_ids = tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

        if hasattr(cfg, "img_unc_prompt"):
            full_unc_ids = tokenizer.encode(cfg.img_unc_prompt, return_tensors="pt", add_special_tokens=False).to(
                model.device
            )
        else:
            full_unc_ids = None

        force_same_image_size = True
        # for x2i task, if multiple reference images are provided as a list, force_same_image_size should be False
        if isinstance(reference_image, list) and len(reference_image) > 1:
            force_same_image_size = False

        for result_tokens in generate(
            cfg, model, tokenizer, input_ids, unconditional_ids, full_unc_ids, force_same_image_size
        ):
            try:
                result = tokenizer.decode(result_tokens, skip_special_tokens=False)
                mm_out = multimodal_decode(result, tokenizer, vq_model)
                proto_writer.extend(mm_out)
            except Exception as e:
                success = False
                print(f"[ERROR] Failed to generate token sequence: {e}")
                break

        if not success:
            continue

        proto_writer.save(f"{save_path}/proto/{name}.pb")
