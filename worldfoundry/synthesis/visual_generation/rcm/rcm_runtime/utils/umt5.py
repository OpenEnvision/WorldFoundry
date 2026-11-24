# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import os
from pathlib import Path

import torch

from worldfoundry.base_models.diffusion_model.models.encoders.wan.model import HuggingfaceTokenizer
from worldfoundry.base_models.diffusion_model.models.encoders.wan.reference import umt5_xxl
from worldfoundry.core.distributed.runtime import torch_process_group as distributed
from worldfoundry.core.io.assets.easy_io import easy_io
from worldfoundry.core.io.paths import cache_root_path, checkpoint_root_path
from worldfoundry.core.observability.rank_logging import log

DEFAULT_UMT5_TOKENIZER_ID = "google/umt5-xxl"
_UMT5_TOKENIZER_MARKER_FILES = ("tokenizer_config.json", "spiece.model", "tokenizer.json")


def _is_tokenizer_dir(candidate: Path) -> bool:
    return candidate.is_dir() and any((candidate / marker).is_file() for marker in _UMT5_TOKENIZER_MARKER_FILES)


def resolve_umt5_tokenizer_path(
    checkpoint_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
) -> tuple[str, bool]:
    """Resolve a local UMT5 tokenizer directory when available offline."""
    if tokenizer_path not in (None, "", DEFAULT_UMT5_TOKENIZER_ID):
        explicit = Path(str(tokenizer_path)).expanduser()
        return str(explicit), _is_tokenizer_dir(explicit)

    candidates: list[Path] = []
    if checkpoint_path not in (None, ""):
        ckpt_dir = Path(str(checkpoint_path)).expanduser().parent
        candidates.extend([ckpt_dir / "google" / "umt5-xxl", ckpt_dir / "umt5-xxl"])
        if ckpt_dir.parent.is_dir():
            for sibling in sorted(ckpt_dir.parent.iterdir(), key=lambda p: p.name):
                candidates.append(sibling / "google" / "umt5-xxl")

    for env_key in ("WORLDFOUNDRY_REGRESSION_CHECKPOINT_ROOT",):
        root_val = os.environ.get(env_key)
        if root_val:
            root = Path(root_val).expanduser()
            candidates.append(root / "google" / "umt5-xxl")
            if root.is_dir():
                for sibling in sorted(root.iterdir(), key=lambda p: p.name):
                    candidates.append(sibling / "google" / "umt5-xxl")

    default_ckpt_root = checkpoint_root_path()
    if default_ckpt_root.is_dir():
        for sibling in sorted(default_ckpt_root.iterdir(), key=lambda p: p.name):
            candidates.append(sibling / "google" / "umt5-xxl")

    hub_roots: list[Path] = []
    if os.environ.get("HF_HUB_CACHE"):
        hub_roots.append(Path(os.environ["HF_HUB_CACHE"]).expanduser())
    if os.environ.get("HF_HOME"):
        hub_roots.append(Path(os.environ["HF_HOME"]).expanduser() / "hub")
    hub_roots.append(cache_root_path() / "huggingface" / "hub")
    for hub_root in hub_roots:
        if hub_root.is_dir():
            for match in sorted(hub_root.glob("models--*/snapshots/*/google/umt5-xxl")):
                candidates.append(match)

    for candidate in candidates:
        if _is_tokenizer_dir(candidate):
            return str(candidate), True

    return DEFAULT_UMT5_TOKENIZER_ID, False


def load_model_torch(model, ckpt_path):
    if distributed.is_rank0():
        ckpt = easy_io.load(
            ckpt_path,
            map_location="cuda",
        )
        model.load_state_dict(ckpt)

    distributed.sync_model_states(model, src=0)
    return model


class UMT5EncoderModel:
    def __init__(
        self,
        text_len=512,
        dtype=torch.bfloat16,
        device=None,
        checkpoint_path="models_t5_umt5-xxl-enc-bf16.pth",
        tokenizer_path=None,
    ):
        if device is None:
            device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
        self.text_len = text_len
        self.dtype = dtype
        self.device = device

        # init model
        model = umt5_xxl(encoder_only=True, dtype=dtype, device=device).eval().requires_grad_(False)
        log.info(f"loading {checkpoint_path}")
        assert checkpoint_path.endswith(".pth")
        model = load_model_torch(model, checkpoint_path)
        self.model = model
        self.model.to(self.device)
        # init tokenizer
        resolved_tokenizer_path, is_local = resolve_umt5_tokenizer_path(checkpoint_path, tokenizer_path)
        tokenizer_kwargs = {"local_files_only": True} if is_local else {}
        self.tokenizer = HuggingfaceTokenizer(
            name=resolved_tokenizer_path,
            seq_len=text_len,
            clean="whitespace",
            **tokenizer_kwargs,
        )

    def __call__(self, texts, device: torch.device | None = None):
        if device is None:
            device = self.device
        ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.model(ids, mask)
        # return [u[:v] for u, v in zip(context, seq_lens)]
        stack_emb = []
        for u, length in zip(context, seq_lens, strict=False):
            if length > self.text_len:
                stack_emb.append(u[: self.text_len])
            else:
                # pad with zeros to max_length
                zeros = torch.zeros(self.text_len - length, u.shape[1]).to(u)
                stack_emb.append(torch.cat([u[:length], zeros], dim=0))
        return torch.stack(stack_emb)


t5_encoder: UMT5EncoderModel | None = None


def get_umt5_embedding(
    checkpoint_path: str,
    prompts: str | list[str],
    device: str = "cuda",
    max_length: int = 512,
    tokenizer_path: str | None = None,
) -> torch.Tensor:
    global t5_encoder
    if t5_encoder is None:
        t5_encoder = UMT5EncoderModel(
            text_len=max_length,
            device=device,
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
        )
    return t5_encoder(prompts, device=device)


def clear_umt5_memory():
    """
    Clears the GPU memory by deleting the global t5_encoder model and emptying the CUDA cache.
    """
    global t5_encoder
    if t5_encoder is not None:
        del t5_encoder
        t5_encoder = None
    gc.collect()
    torch.cuda.empty_cache()
