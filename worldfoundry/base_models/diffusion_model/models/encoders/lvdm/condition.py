"""LVDM FrozenOpenCLIPEmbedder and related condition modules.

Checkpoint-compatible CLIP text (and optional image) towers
used by LVDM / T2V-Turbo.  :class:`FrozenOpenCLIPEmbedder`
is the usual prompt encoder; CLIP+T5 combo is
:class:`FrozenCLIPT5Encoder`.

Outputs are ``context`` token sequences.  A second key
layout lives in :mod:`.condition2`.

This is the OpenCLIP family, not Gemma or UMT5.
"""

import os
from pathlib import Path

import open_clip.transformer
import torch
import torch.nn as nn
import kornia
import torchvision.transforms as transforms
import open_clip
from torch.utils.checkpoint import checkpoint
from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel
from worldfoundry.core.nn.diffusion.utils import autocast
from worldfoundry.core.model_loading.factory import count_params


# LayerNormFp32 forces a cast to fp32, which causes a bug with amp_bf16
open_clip.model.LayerNormFp32 = open_clip.transformer.LayerNorm


def _checkpoint_root() -> Path:
    """Helper function to checkpoint root.

    Returns:
        The return value.
    """
    return Path(os.environ.get("WORLDFOUNDRY_CKPT_DIR", "~/.cache/worldfoundry/checkpoints")).expanduser()


def _checkpoint_roots() -> tuple[Path, ...]:
    """Return the configured checkpoint root and its local ckpt/ckpts sibling."""

    root = _checkpoint_root()
    roots = [root]
    if root.name in {"ckpt", "ckpts"}:
        sibling_name = "ckpts" if root.name == "ckpt" else "ckpt"
        roots.append(root.with_name(sibling_name))
    return tuple(dict.fromkeys(roots))


def _local_hf_model(version: str) -> str:
    """Helper function to local hf model.

    Args:
        version: The version.

    Returns:
        The return value.
    """
    if not isinstance(version, str) or os.path.exists(version):
        return version
    repo_dir = {
        "openai/clip-vit-large-patch14": "openai--clip-vit-large-patch14",
        "openai/clip-vit-base-patch32": "openai--clip-vit-base-patch32",
    }.get(version)
    if repo_dir:
        for root in _checkpoint_roots():
            for candidate in (root / repo_dir, root / "hfd" / repo_dir):
                if candidate.is_dir():
                    return str(candidate)
    return version


def _local_open_clip_pretrained(arch: str, version: str) -> str:
    """Helper function to local open clip pretrained.

    Args:
        arch: The arch.
        version: The version.

    Returns:
        The return value.
    """
    if not isinstance(version, str) or os.path.exists(version):
        return version
    if arch == "ViT-H-14" and version in {"laion2b_s32b_b79k", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"}:
        candidates = [os.environ.get("WORLDFOUNDRY_OPENCLIP_VITH14_PATH", "")]
        for root in _checkpoint_roots():
            for repo_dir in (
                "laion--CLIP-ViT-H-14-laion2B-s32B-b79K",
                "CLIP-ViT-H-14-laion2B-s32B-b79K",
            ):
                candidates.extend(
                    (
                        str(root / repo_dir / "open_clip_pytorch_model.bin"),
                        str(root / repo_dir / "open_clip_model.safetensors"),
                    )
                )
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                return candidate
    return version


class AbstractEncoder(nn.Module):
    """Abstract encoder implementation."""

    def __init__(self):
        """Init."""
        super().__init__()

    def encode(self, *args, **kwargs):
        """Encode."""
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):
    """Identity encoder implementation."""

    def encode(self, x):
        """Encode.

        Args:
            x: The x.
        """
        return x


class ClassEmbedder(nn.Module):
    """Class embedder implementation."""

    def __init__(self, embed_dim, n_classes=1000, key="class", ucg_rate=0.1):
        """Init.

        Args:
            embed_dim: The embed dim.
            n_classes: The n classes.
            key: The key.
            ucg_rate: The ucg rate.
        """
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        """Forward.

        Args:
            batch: The batch.
            key: The key.
            disable_dropout: The disable dropout.
        """
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0.0 and not disable_dropout:
            mask = 1.0 - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1 - mask) * torch.ones_like(c) * (self.n_classes - 1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        """Get unconditional conditioning.

        Args:
            bs: The bs.
            device: The device.
        """
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""

    def __init__(
        self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True
    ):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        """Init.

        Args:
            version: The version.
            device: The device.
            max_length: The max length.
            freeze: The freeze.
        """
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length  # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        """Freeze."""
        self.transformer = self.transformer.eval()
        # self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        """Forward.

        Args:
            text: The text.
        """
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        """Encode.

        Args:
            text: The text.
        """
        return self(text)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
        layer_idx=None,
    ):  # clip-vit-base-patch32
        """Init.

        Args:
            version: The version.
            device: The device.
            max_length: The max length.
            freeze: The freeze.
            layer: The layer.
            layer_idx: The layer idx.
        """
        super().__init__()
        assert layer in self.LAYERS
        version = _local_hf_model(version)
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        """Freeze."""
        self.transformer = self.transformer.eval()
        # self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        """Forward.

        Args:
            text: The text.
        """
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer == "hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        """Encode.

        Args:
            text: The text.
        """
        return self(text)


class ClipImageEmbedder(nn.Module):
    """Clip image embedder implementation."""

    def __init__(
        self, model, jit=False, device="cuda" if torch.cuda.is_available() else "cpu", antialias=True, ucg_rate=0.0
    ):
        """Init.

        Args:
            model: The model.
            jit: The jit.
            device: The device.
            antialias: The antialias.
            ucg_rate: The ucg rate.
        """
        super().__init__()
        from worldfoundry.base_models.perception_core.general_perception.openai_clip import (
            load as load_clip,
        )

        self.model, _ = load_clip(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer("mean", torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer("std", torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)
        self.ucg_rate = ucg_rate

    def preprocess(self, x):
        """Preprocess.

        Args:
            x: The x.
        """
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224), interpolation="bicubic", align_corners=True, antialias=self.antialias)
        x = (x + 1.0) / 2.0
        # re-normalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x, no_dropout=False):
        """Forward.

        Args:
            x: The x.
            no_dropout: The no dropout.
        """
        # x is assumed to be in range [-1,1]
        out = self.model.encode_image(self.preprocess(x))
        out = out.to(x.dtype)
        if self.ucg_rate > 0.0 and not no_dropout:
            out = torch.bernoulli((1.0 - self.ucg_rate) * torch.ones(out.shape[0], device=out.device))[:, None] * out
        return out


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """

    LAYERS = [
        # "pooled",
        "last",
        "penultimate",
    ]

    def __init__(
        self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77, freeze=True, layer="last"
    ):
        """Init.

        Args:
            arch: The arch.
            version: The version.
            device: The device.
            max_length: The max length.
            freeze: The freeze.
            layer: The layer.
        """
        super().__init__()
        assert layer in self.LAYERS
        # VideoCrafter checkpoints carry the text-transformer weights under
        # cond_stage_model.*, so the official text encoder is instantiated
        # without OpenCLIP pretrained weights and then restored from the model
        # checkpoint.
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device("cpu"))
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        """Freeze."""
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        """Forward.

        Args:
            text: The text.
        """
        tokens = open_clip.tokenize(text)  ## all clip models use 77 as context length
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        """Encode with transformer.

        Args:
            text: The text.
        """
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        batch_first = bool(getattr(self.model.transformer, "batch_first", False))
        if not batch_first:
            x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        if not batch_first:
            x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        """Text transformer forward.

        Args:
            x: The x.
            attn_mask: The attn mask.
        """
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        """Encode.

        Args:
            text: The text.
        """
        return self(text)


class FrozenOpenCLIPImageEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="pooled",
        antialias=True,
        ucg_rate=0.0,
    ):
        """Init.

        Args:
            arch: The arch.
            version: The version.
            device: The device.
            max_length: The max length.
            freeze: The freeze.
            layer: The layer.
            antialias: The antialias.
            ucg_rate: The ucg rate.
        """
        super().__init__()
        version = _local_open_clip_pretrained(arch, version)
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device("cpu"), pretrained=version)
        del model.transformer
        self.model = model
        # self.mapper = torch.nn.Linear(1280, 1024)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "penultimate":
            raise NotImplementedError()
            self.layer_idx = 1

        self.antialias = antialias

        self.register_buffer("mean", torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer("std", torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)
        self.ucg_rate = ucg_rate

    def preprocess(self, x):
        """Preprocess.

        Args:
            x: The x.
        """
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224), interpolation="bicubic", align_corners=True, antialias=self.antialias)
        x = (x + 1.0) / 2.0
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def freeze(self):
        """Freeze."""
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, image, no_dropout=False):
        """Forward.

        Args:
            image: The image.
            no_dropout: The no dropout.
        """
        z = self.encode_with_vision_transformer(image)
        if self.ucg_rate > 0.0 and not no_dropout:
            z = torch.bernoulli((1.0 - self.ucg_rate) * torch.ones(z.shape[0], device=z.device))[:, None] * z
        return z

    def encode_with_vision_transformer(self, img):
        """Encode with vision transformer.

        Args:
            img: The img.
        """
        img = self.preprocess(img)
        x = self.model.visual(img)
        return x

    def encode(self, text):
        """Encode.

        Args:
            text: The text.
        """
        return self(text)


class FrozenOpenCLIPImageEmbedderV2(AbstractEncoder):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(
        self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", freeze=True, layer="pooled", antialias=True
    ):
        """Init.

        Args:
            arch: The arch.
            version: The version.
            device: The device.
            freeze: The freeze.
            layer: The layer.
            antialias: The antialias.
        """
        super().__init__()
        version = _local_open_clip_pretrained(arch, version)
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device("cpu"), pretrained=version)
        del model.transformer
        self.model = model
        self.device = device

        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "penultimate":
            raise NotImplementedError()
            self.layer_idx = 1

        self.antialias = antialias

        self.register_buffer("mean", torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer("std", torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        """Preprocess.

        Args:
            x: The x.
        """
        # Video runtimes supply images in [-1, 1]; CLIP normalization expects [0, 1].
        x = kornia.geometry.resize(
            x, (224, 224), interpolation="bicubic", align_corners=True,
            antialias=self.antialias,
        )
        x = (x + 1.0) / 2.0
        return kornia.enhance.normalize(x, self.mean, self.std)

    def freeze(self):
        """Freeze."""
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image, no_dropout=False):
        """Forward.

        Args:
            image: The image.
            no_dropout: The no dropout.
        """
        ## image: b c h w
        z = self.encode_with_vision_transformer(image)
        return z

    def encode_with_vision_transformer(self, x):
        """Encode with vision transformer.

        Args:
            x: The x.
        """
        x = self.preprocess(x)

        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        if getattr(self.model.visual, "input_patchnorm", False):
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(
                x.shape[0],
                x.shape[1],
                self.model.visual.grid_size[0],
                self.model.visual.patch_size[0],
                self.model.visual.grid_size[1],
                self.model.visual.patch_size[1],
            )
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.model.visual.grid_size[0] * self.model.visual.grid_size[1], -1)
            x = self.model.visual.patchnorm_pre_ln(x)
            x = self.model.visual.conv1(x)
        else:
            x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [
                self.model.visual.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.visual.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        patch_dropout = getattr(self.model.visual, "patch_dropout", None)
        if patch_dropout is not None:
            x = patch_dropout(x)
        x = self.model.visual.ln_pre(x)

        transformer = self.model.visual.transformer
        # Modern OpenCLIP accepts NLD and handles its internal layout itself,
        # including when batch_first=False. Older versions require LND here.
        legacy_layout = not hasattr(transformer, "batch_first")
        if legacy_layout:
            x = x.permute(1, 0, 2)  # NLD -> LND
        x = transformer(x)
        if legacy_layout:
            x = x.permute(1, 0, 2)  # LND -> NLD

        return x


class FrozenCLIPT5Encoder(AbstractEncoder):
    """Frozen encoder implementation."""

    def __init__(
        self,
        clip_version="openai/clip-vit-large-patch14",
        t5_version="google/t5-v1_1-xl",
        device="cuda",
        clip_max_length=77,
        t5_max_length=77,
    ):
        """Init.

        Args:
            clip_version: The clip version.
            t5_version: The t5 version.
            device: The device.
            clip_max_length: The clip max length.
            t5_max_length: The t5 max length.
        """
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(clip_version, device, max_length=clip_max_length)
        self.t5_encoder = FrozenT5Embedder(t5_version, device, max_length=t5_max_length)
        print(
            f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder) * 1.0e-6:.2f} M parameters, "
            f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder) * 1.0e-6:.2f} M params."
        )

    def encode(self, text):
        """Encode.

        Args:
            text: The text.
        """
        return self(text)

    def forward(self, text):
        """Forward.

        Args:
            text: The text.
        """
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]
