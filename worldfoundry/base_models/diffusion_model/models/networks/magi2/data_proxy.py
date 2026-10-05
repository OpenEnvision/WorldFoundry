"""MAGI-2-preview data proxy (single-GPU PyTorch port).

Ported faithfully from SandAI's Apache-2.0 ``pipeline/preview_data_proxy.py``.

This module is the glue between raw latents and the packed token stream the
preview DiT consumes. It

* patchifies the video latent into tokens (``img2tokens``; ``patch_size == 1``
  for preview so this is a flatten + transpose, reproduced with ``rearrange``
  instead of the upstream ``unfoldNd`` dependency),
* packs ``[video | audio | text | (ref-image)* | time]`` into one flat
  ``(total_tokens, C)`` stream plus the ``coords_mapping`` ``[L, 9]``,
  ``modality_mapping`` ``[L]``, ``VarlenHandler`` (cu_seqlens) and optional
  ``time_token_sequence`` that :meth:`preview_dit.Transformer.forward` expects,
  and
* unpacks the DiT output ``(total_tokens, 64)`` back to ``x_video (B,48,T,H,W)``
  and ``x_audio (B,L,64)``.

Single-GPU collapse (cp = ep = dp = 1): the upstream ``_pad_for_ep_cp`` /
``_reduce_max_token_num_for_ep_cp`` ulysses/expert-parallel padding is an
identity no-op (``pad_size == 0``); ``torch.distributed`` / ``psm`` are dropped.
The coord / modality / packing math is reproduced exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from .preview_layers import Modality, VarlenHandler, get_coords

_FP32 = torch.float32


# ============================================================
# proxy config (upstream ``DataProxyConfig`` in common/magi2_config.py)
# ============================================================


@dataclass
class Magi2PreviewDataProxyConfig:
    """Data-proxy hyperparameters (upstream ``DataProxyConfig`` defaults)."""

    t_patch_size: int = 1
    patch_size: int = 1
    spatial_rope_interpolation: Literal["inter", "extra"] = "extra"
    add_time_token: bool = False
    time_channel_dim: int = 64
    time_aligned_rope: bool = False
    audio_latent_fps: float = 25.0
    time_pos_fps: float = 3.125
    vae_first_latent_is_image: bool = True
    video_fps: float = 25.0


# ============================================================
# small pure-torch helpers (ported)
# ============================================================


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """1D sinusoidal time embedding (upstream ``magi2_preview.py``).

    Not exported by ``preview_layers`` (it lives in the upstream DiT module and
    was not part of the layer port), so it is reproduced here as a data-proxy
    leaf utility used to build ``time_token_sequence``.
    """
    position = position.to(torch.float32) * 1000.0
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=position.device)
        / half
    )
    args = position[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _to_int(value: int | torch.Tensor) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().reshape(-1)[0].item())
    return int(value)


def _pad_cat(
    tensors: list[torch.Tensor], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Right-pad each token block to the widest channel count, then row-concat."""
    if not tensors:
        return torch.empty(0, 0, device=device, dtype=dtype)
    max_channel = max(t.shape[-1] for t in tensors)
    return torch.cat(
        [
            F.pad(t.to(device=device, dtype=dtype), (0, max_channel - t.shape[-1]))
            for t in tensors
        ],
        dim=0,
    )


def _segment_paint(
    values: list[int],
    seqlens: list[int],
    dtype: torch.dtype,
    device: torch.device,
    output_size: int,
) -> torch.Tensor:
    mapping = torch.empty(output_size, dtype=dtype, device=device)
    offset = 0
    for value, seqlen in zip(values, seqlens):
        mapping[offset : offset + seqlen] = int(value)
        offset += seqlen
    return mapping


def _seqlens2cu_seqlens(
    seqlens: list[int], device: torch.device | None = None
) -> torch.Tensor:
    seqlens_tensor = torch.tensor(seqlens, dtype=torch.int32, device=device)
    return F.pad(torch.cumsum(seqlens_tensor, dim=0), (1, 0))


def _len_to_list(value: torch.Tensor) -> list[int]:
    """Flatten a packed ref-length entry (e.g. [H, W]) back into a python list."""
    return [int(v) for v in value.detach().to(torch.long).reshape(-1).tolist()]


# ============================================================
# model-input container (upstream ``ModelInput``)
# ============================================================


@dataclass
class ModelInput:
    """Per-batch inputs to :meth:`Magi2PreviewDataProxy.process_input`.

    Only the fields the preview proxy consumes are kept (the sampler/CFG-only
    ref-audio/ref-video fields are dropped for the single-GPU port).
    """

    x_t: torch.Tensor  # (B, 48, T, H, W) video latent
    audio_x_t: torch.Tensor  # (B, L, 64) audio latent
    audio_feat_len: torch.Tensor | list[int]
    txt_feat: torch.Tensor  # (B, S, 5120) text feature
    txt_feat_len: torch.Tensor | list[int]
    t: Optional[torch.Tensor] = None  # (B,) diffusion time (only if add_time_token)
    per_token_video_t: Optional[torch.Tensor] = None  # (B, 1, T, H, W)
    per_token_audio_t: Optional[torch.Tensor] = None  # (B, L, 1)
    # Conditioning images: (B, M, C, 1, H, W) + one special token embedding each.
    ref_image_feat: Optional[torch.Tensor] = None
    ref_image_feat_len: Optional[torch.Tensor] = None  # (B, M, 2) holding (H, W)
    ref_image_special_token_embedding: Optional[torch.Tensor] = None  # (B, M, D)


# ============================================================
# per-sample packing (upstream ``SingleData``)
# ============================================================


@dataclass
class SingleData:
    """One sample's patchified tokens plus the packing metadata for that clip."""

    video_x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: int
    txt_feat: torch.Tensor
    txt_feat_len: int
    t: int
    h: int
    w: int
    patch_size: int
    t_patch_size: int
    spatial_rope_interpolation: Literal["inter", "extra"]
    diffusion_t: Optional[torch.Tensor] = None
    per_token_video_t: Optional[torch.Tensor] = None
    per_token_audio_t: Optional[torch.Tensor] = None
    time_channel_dim: int = 0
    vae_first_latent_is_image: bool = True
    video_fps: float = 25.0
    time_pos_fps: float = 3.125
    ref_image_feats: Optional[list[torch.Tensor]] = None
    ref_image_feat_lens: Optional[list[list[int]]] = None
    ref_image_special_tokens: Optional[list[torch.Tensor]] = None

    def __post_init__(self) -> None:
        self.video_token_num = self.video_x_t.shape[0]
        self.origin_audio_feat_len = self.audio_x_t.shape[0]
        self.audio_x_t = self.audio_x_t[: self.audio_feat_len]
        self.txt_feat = self.txt_feat[: self.txt_feat_len]
        if self.per_token_audio_t is not None:
            self.per_token_audio_t = self.per_token_audio_t[: self.audio_feat_len]

        self.ref_image_feats = self.ref_image_feats or []
        self.ref_image_feat_lens = self.ref_image_feat_lens or []
        self.ref_image_special_tokens = self.ref_image_special_tokens or []
        # feat_lens holds the patch-grid (H, W) per image, so their product is the
        # token count that survived img2tokens for that image.
        self.ref_image_token_nums: list[int] = [
            int(math.prod(feat_len)) for feat_len in self.ref_image_feat_lens
        ]
        self.ref_image_feats = [
            feat[:num]
            for feat, num in zip(self.ref_image_feats, self.ref_image_token_nums)
        ]
        self.num_ref_images = len(self.ref_image_feats)
        self.total_ref_image_feat_len = sum(self.ref_image_token_nums)

        self.video_channel = self.video_x_t.shape[-1]
        self.audio_channel = self.audio_x_t.shape[-1]

    @property
    def device(self) -> torch.device:
        return self.video_x_t.device

    @property
    def default_dtype(self) -> torch.dtype:
        return self.video_x_t.dtype

    @property
    def add_time_token(self) -> bool:
        return self.diffusion_t is not None

    @property
    def total_token_num(self) -> int:
        total = self.video_token_num + self.audio_feat_len + self.txt_feat_len
        # Each conditioning image also contributes one leading special token.
        total += self.total_ref_image_feat_len + self.num_ref_images
        return total + (1 if self.add_time_token else 0)

    @property
    def feat_to_cat(self) -> list[torch.Tensor]:
        tensors = [self.video_x_t, self.audio_x_t, self.txt_feat]
        for k in range(self.num_ref_images):
            tensors.append(self.ref_image_special_tokens[k])
            tensors.append(self.ref_image_feats[k])
        if self.add_time_token:
            tensors.append(
                self.diffusion_t.to(
                    device=self.device, dtype=self.default_dtype
                ).reshape(1, 1)
            )
        return tensors

    @property
    def token_sequence(self) -> torch.Tensor:
        return _pad_cat(self.feat_to_cat, self.device, self.default_dtype)

    @property
    def modality_map_seqlens(self) -> tuple[list[int], list[int]]:
        seqlens = [self.video_token_num, self.audio_feat_len, self.txt_feat_len]
        maps = [Modality.VIDEO, Modality.AUDIO, Modality.TEXT]
        for k in range(self.num_ref_images):
            # The special token comes from the text encoder, so it routes as text
            # while the image patches themselves route as video.
            seqlens.append(1)
            maps.append(Modality.TEXT)
            seqlens.append(self.ref_image_token_nums[k])
            maps.append(Modality.VIDEO)
        if self.add_time_token:
            seqlens.append(1)
            maps.append(Modality.TIME)
        return seqlens, maps

    @property
    def modality_mapping(self) -> torch.Tensor:
        seqlens, maps = self.modality_map_seqlens
        return _segment_paint(
            maps,
            seqlens,
            dtype=torch.int32,
            device=self.device,
            output_size=self.total_token_num,
        )

    def _default_coords(
        self,
        shape: tuple[int, int, int],
        ref_feat_shape: tuple[int, int, int],
        offset_thw: tuple[int, int, int] = (0, 0, 0),
        time_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return get_coords(
            shape,
            ref_feat_shape,
            offset_thw=offset_thw,
            device=self.device,
            dtype=self.default_dtype,
            time_positions=time_positions,
        )

    @property
    def coords_to_cat(self) -> list[torch.Tensor]:
        t_steps = self.t // self.t_patch_size
        h_steps = self.h // self.patch_size
        w_steps = self.w // self.patch_size

        if self.spatial_rope_interpolation == "inter":
            video_h_ref, video_w_ref = 32, 32
        else:
            video_h_ref, video_w_ref = h_steps, w_steps

        # Video: 3D grid over (T, H, W).
        video_coords = self._default_coords(
            (t_steps, h_steps, w_steps), (t_steps, video_h_ref, video_w_ref)
        )
        # Audio: 1D along time, (audio_len, 1, 1); ref time is the "magic" /8+1.
        magic_audio_ref_t = (self.audio_feat_len - 1) // 8 + 1
        audio_coords = self._default_coords(
            (self.audio_feat_len, 1, 1),
            (magic_audio_ref_t // self.t_patch_size, 1, 1),
        )

        coords = [
            video_coords,
            audio_coords,
            # Text: (txt_len, 1, 1) placed *before* the clip (time offset -txt_len).
            self._default_coords(
                (self.txt_feat_len, 1, 1),
                (1, 1, 1),
                offset_thw=(-self.txt_feat_len, 0, 0),
            ),
        ]

        for k in range(self.num_ref_images):
            token_len = self.ref_image_token_nums[k]
            if len(self.ref_image_feat_lens[k]) >= 2:
                h_i = int(self.ref_image_feat_lens[k][0])
                w_i = int(self.ref_image_feat_lens[k][1])
            else:
                h_i = w_i = int(math.ceil(math.sqrt(token_len)))
            # Images sit past the generated clip on the time axis, one latent step
            # apart, with a gap of 1 so they never alias the last video frame.
            t_off = t_steps + 2 + k
            coords.append(
                torch.tensor(
                    [[t_off, -1, -1, 1, h_i, w_i, 1, h_i, w_i]],
                    device=self.device,
                    dtype=self.default_dtype,
                )
            )
            coords.append(
                self._default_coords(
                    (1, h_i, w_i), (1, h_i, w_i), offset_thw=(t_off, 0, 0)
                )[:token_len]
            )

        if self.add_time_token:
            coords.append(self._default_coords((1, 1, 1), (1, 1, 1))[:1])
        return coords

    @property
    def coords_mapping(self) -> torch.Tensor:
        return torch.cat(self.coords_to_cat, dim=0)

    @property
    def time_token_sequence(self) -> torch.Tensor:
        if self.time_channel_dim == 0:
            return torch.empty(self.total_token_num, 0, device=self.device)
        assert (
            self.per_token_video_t is not None and self.per_token_audio_t is not None
        )
        parts = [
            self.per_token_video_t.squeeze(-1),
            self.per_token_audio_t.squeeze(-1),
            torch.zeros(self.txt_feat_len, device=self.device),
        ]
        for k in range(self.num_ref_images):
            # Images are clean conditioning, so their diffusion time is 0.
            parts.append(torch.zeros(1, device=self.device))
            parts.append(torch.zeros(self.ref_image_token_nums[k], device=self.device))
        if self.add_time_token:
            parts.append(self.diffusion_t.reshape(1).to(self.device))
        raw_t = torch.cat(parts, dim=0)
        if self.time_channel_dim == 1:
            return raw_t.unsqueeze(-1)
        return sinusoidal_embedding_1d(self.time_channel_dim, raw_t)


# ============================================================
# packed batch (upstream ``SimplePackedData``)
# ============================================================


@dataclass
class SimplePackedData:
    """Row-concatenated batch of :class:`SingleData` with shared cu_seqlens."""

    items: list[SingleData]

    def __post_init__(self) -> None:
        if len(self.items) == 0:
            raise ValueError("SimplePackedData must contain at least one item.")
        self._total_token_num_list = [item.total_token_num for item in self.items]
        self._total_token_num_sum = sum(self._total_token_num_list)
        self._total_token_num_max = max(self._total_token_num_list)
        self._total_token_cu_seqlens = _seqlens2cu_seqlens(self._total_token_num_list)

    @property
    def device(self) -> torch.device:
        return self.items[0].device

    @property
    def default_dtype(self) -> torch.dtype:
        return self.items[0].default_dtype

    @property
    def token_sequence(self) -> torch.Tensor:
        feat_to_cat = list(chain.from_iterable(item.feat_to_cat for item in self.items))
        return _pad_cat(feat_to_cat, self.device, self.default_dtype)

    @property
    def modality_mapping(self) -> torch.Tensor:
        seqlens: list[int] = []
        maps: list[int] = []
        for item in self.items:
            item_seqlens, item_maps = item.modality_map_seqlens
            seqlens.extend(item_seqlens)
            maps.extend(item_maps)
        return _segment_paint(
            maps,
            seqlens,
            dtype=torch.int32,
            device=self.device,
            output_size=self.total_token_num,
        )

    @property
    def coords_mapping(self) -> torch.Tensor:
        coords = list(chain.from_iterable(item.coords_to_cat for item in self.items))
        return torch.cat(coords, dim=0)

    @property
    def time_token_sequence(self) -> torch.Tensor:
        return torch.cat([item.time_token_sequence for item in self.items], dim=0)

    @property
    def total_token_num(self) -> int:
        return self._total_token_num_sum

    @property
    def cu_seqlen(self) -> torch.Tensor:
        return self._total_token_cu_seqlens.clone()

    @property
    def max_seqlen(self) -> int:
        return self._total_token_num_max

    def __getitem__(self, index: int) -> SingleData:
        return self.items[index]

    def depack_token_sequence(
        self, token_sequence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        videos, audios = [], []
        for item, token_slice in zip(
            self.items,
            torch.split(token_sequence, [i.total_token_num for i in self.items], dim=0),
        ):
            video_flat = token_slice[: item.video_token_num, : item.video_channel]
            c_out = item.video_channel // (
                item.t_patch_size * item.patch_size * item.patch_size
            )
            video = rearrange(
                video_flat,
                "(T H W) (pT pH pW C) -> C (T pT) (H pH) (W pW)",
                T=item.t // item.t_patch_size,
                H=item.h // item.patch_size,
                W=item.w // item.patch_size,
                pT=item.t_patch_size,
                pH=item.patch_size,
                pW=item.patch_size,
                C=c_out,
            ).contiguous()
            audio = torch.zeros(
                item.origin_audio_feat_len,
                item.audio_channel,
                device=token_sequence.device,
                dtype=token_sequence.dtype,
            )
            audio[: item.audio_feat_len] = token_slice[
                item.video_token_num : item.video_token_num + item.audio_feat_len,
                : item.audio_channel,
            ]
            videos.append(video)
            audios.append(audio)
        return torch.stack(videos, dim=0), torch.stack(audios, dim=0)


# ============================================================
# proxy
# ============================================================


class Magi2PreviewDataProxy:
    """Packs latents into the preview-DiT token stream and unpacks the output.

    Single-GPU: cp = ep = dp = 1, so the upstream ``_pad_for_ep_cp`` ulysses /
    expert-parallel alignment padding is an identity no-op (``pad_size == 0``).
    """

    def __init__(self, config: Magi2PreviewDataProxyConfig) -> None:
        self.config = config
        self.patch_size = config.patch_size
        self.t_patch_size = config.t_patch_size
        self._saved_data: dict[str, Any] = {}

    # -- output bookkeeping -------------------------------------------------

    def saved_for_output(self, **kwargs: Any) -> None:
        self._saved_data.update(kwargs)

    def get_saved_data(self, key: str) -> Any:
        return self._saved_data[key]

    # -- patchify -----------------------------------------------------------

    def img2tokens(self, x_t: torch.Tensor) -> torch.Tensor:
        """Patchify ``(N, C, T, H, W)`` -> ``(N, num_tokens, C*prod(kernel))``.

        Reproduces the upstream ``unfoldNd`` patchify with ``rearrange``; the
        column order ``(pT pH pW C)`` matches ``depack_token_sequence`` so the
        two are exact inverses (for ``patch_size == 1`` this is a flatten +
        transpose). Sub-kernel inputs return an empty token block, as upstream.
        """
        pT, pH, pW = self.t_patch_size, self.patch_size, self.patch_size
        kernel_size = (pT, pH, pW)
        if not all(s >= k for s, k in zip(x_t.shape[2:], kernel_size)):
            return torch.empty(
                x_t.shape[0],
                0,
                x_t.shape[1] * math.prod(kernel_size),
                device=x_t.device,
                dtype=x_t.dtype,
            )
        return rearrange(
            x_t,
            "N C (T pT) (H pH) (W pW) -> N (T H W) (pT pH pW C)",
            pT=pT,
            pH=pH,
            pW=pW,
        ).contiguous()

    # -- pack ---------------------------------------------------------------

    def process_input(
        self, data: ModelInput
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, VarlenHandler, torch.Tensor]:
        """Pack a :class:`ModelInput` batch into ``Transformer.forward`` args.

        Returns ``(token_sequence, coords_mapping, modality_mapping,
        varlen_handler, time_token_sequence)``.
        """
        batch_size, input_video_channel, t, h, w = data.x_t.shape
        video_tokens = self.img2tokens(data.x_t)
        audio_tokens = data.audio_x_t.contiguous()
        text_tokens = data.txt_feat.contiguous()

        ptv_tokens = None
        pta = None
        if self.config.time_channel_dim > 0 and data.per_token_video_t is not None:
            ptv_tokens = self.img2tokens(data.per_token_video_t)[:, :, :1]
            pta = data.per_token_audio_t

        ref_image_data = data.ref_image_feat
        ref_image_feat_len = data.ref_image_feat_len
        ref_image_special_token_embedding = data.ref_image_special_token_embedding
        has_ref_image = (
            ref_image_data is not None
            and ref_image_data.ndim >= 5
            and ref_image_feat_len is not None
            and ref_image_feat_len.ndim >= 2
            and ref_image_special_token_embedding is not None
            and ref_image_special_token_embedding.ndim >= 3
        )
        num_ref_image = ref_image_data.shape[1] if has_ref_image else 0
        ref_device, ref_dtype = text_tokens.device, text_tokens.dtype

        items = []
        for i in range(batch_size):
            ref_image_feats_i: list[torch.Tensor] = []
            ref_image_feat_lens_i: list[list[int]] = []
            ref_image_special_tokens_i: list[torch.Tensor] = []
            for k in range(num_ref_image):
                # img2tokens works on a batch, so the single image is given a
                # batch axis and then squeezed back out.
                feat = self.img2tokens(ref_image_data[i, k].unsqueeze(0)).squeeze(0)
                ref_image_feats_i.append(feat)
                ref_image_feat_lens_i.append(_len_to_list(ref_image_feat_len[i, k]))
                ref_image_special_tokens_i.append(
                    ref_image_special_token_embedding[i, k]
                    .to(device=ref_device, dtype=ref_dtype)
                    .unsqueeze(0)
                )

            items.append(
                SingleData(
                    video_x_t=video_tokens[i],
                    audio_x_t=audio_tokens[i],
                    audio_feat_len=_to_int(data.audio_feat_len[i]),
                    txt_feat=text_tokens[i],
                    txt_feat_len=_to_int(data.txt_feat_len[i]),
                    ref_image_feats=ref_image_feats_i,
                    ref_image_feat_lens=ref_image_feat_lens_i,
                    ref_image_special_tokens=ref_image_special_tokens_i,
                    t=t,
                    h=h,
                    w=w,
                    patch_size=self.patch_size,
                    t_patch_size=self.t_patch_size,
                    spatial_rope_interpolation=self.config.spatial_rope_interpolation,
                    diffusion_t=data.t[i] if self.config.add_time_token else None,
                    per_token_video_t=(
                        ptv_tokens[i] if ptv_tokens is not None else None
                    ),
                    per_token_audio_t=pta[i] if pta is not None else None,
                    time_channel_dim=self.config.time_channel_dim,
                    time_pos_fps=self.config.time_pos_fps,
                    vae_first_latent_is_image=self.config.vae_first_latent_is_image,
                    video_fps=self.config.video_fps,
                )
            )

        packed = SimplePackedData(items)
        varlen_handler = VarlenHandler(
            cu_seqlens_q=packed.cu_seqlen.to(device=data.x_t.device, dtype=torch.int32),
            cu_seqlens_k=packed.cu_seqlen.to(device=data.x_t.device, dtype=torch.int32),
            max_seqlen_q=packed.max_seqlen,
            max_seqlen_k=packed.max_seqlen,
        )

        # cp = ep = dp = 1: _pad_for_ep_cp is an identity no-op (pad_size == 0).
        self.saved_for_output(
            simple_packed_data=packed,
            input_video_channel=input_video_channel,
            pad_size=0,
        )
        return (
            packed.token_sequence,
            packed.coords_mapping,
            packed.modality_mapping,
            varlen_handler,
            packed.time_token_sequence,
        )

    # -- unpack -------------------------------------------------------------

    def process_output(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        packed: SimplePackedData = self.get_saved_data("simple_packed_data")
        pad_size = self.get_saved_data("pad_size")
        if pad_size > 0:
            x = x[:-pad_size]
        x_video, x_audio = packed.depack_token_sequence(x)
        return x_video, x_audio

    # -- convenience callables ---------------------------------------------

    def build_model_inputs(
        self,
        latent: torch.Tensor,  # (B, 48, T, H, W)
        audio_latent: torch.Tensor,  # (B, L, 64)
        context: torch.Tensor,  # (B, S, 5120)
        per_token_video_t: Optional[torch.Tensor] = None,  # (B, 1, T, H, W)
        per_token_audio_t: Optional[torch.Tensor] = None,  # (B, L, 1)
        t: Optional[torch.Tensor] = None,  # (B,) diffusion time
        audio_feat_len: Optional[torch.Tensor | list[int]] = None,
        txt_feat_len: Optional[torch.Tensor | list[int]] = None,
        ref_image_feat: Optional[torch.Tensor] = None,
        ref_image_feat_len: Optional[torch.Tensor] = None,
        ref_image_special_token_embedding: Optional[torch.Tensor] = None,
    ) -> dict:
        """Build the ``Transformer.forward`` kwargs from raw latents.

        Returns a dict with keys ``x``, ``coords_mapping``, ``modality_mapping``,
        ``varlen_handler`` and ``time_token_sequence`` — spreadable directly into
        :meth:`preview_dit.Transformer.forward`. Unpack the DiT output with
        :meth:`unpack_output`.
        """
        b, _, vt, vh, vw = latent.shape
        audio_len = audio_latent.shape[1]
        device = latent.device
        dtype = latent.dtype

        if audio_feat_len is None:
            audio_feat_len = torch.full((b,), audio_len, device=device, dtype=torch.long)
        if txt_feat_len is None:
            txt_feat_len = torch.full(
                (b,), context.shape[1], device=device, dtype=torch.long
            )

        # Per-token diffusion time: synthesise from ``t`` when not supplied so the
        # time_token_sequence can always be built when time_channel_dim > 0.
        if self.config.time_channel_dim > 0 and per_token_video_t is None:
            t_src = (
                t
                if t is not None
                else torch.zeros(b, device=device, dtype=dtype)
            )
            t_src = t_src.to(device=device, dtype=dtype).reshape(b, 1, 1, 1, 1)
            per_token_video_t = t_src.expand(b, 1, vt, vh, vw).clone()
            per_token_audio_t = t_src.reshape(b, 1, 1).expand(b, audio_len, 1).clone()

        model_input = ModelInput(
            x_t=latent,
            audio_x_t=audio_latent,
            audio_feat_len=audio_feat_len,
            txt_feat=context,
            txt_feat_len=txt_feat_len,
            t=t,
            per_token_video_t=per_token_video_t,
            per_token_audio_t=per_token_audio_t,
            ref_image_feat=ref_image_feat,
            ref_image_feat_len=ref_image_feat_len,
            ref_image_special_token_embedding=ref_image_special_token_embedding,
        )
        (
            token_sequence,
            coords_mapping,
            modality_mapping,
            varlen_handler,
            time_token_sequence,
        ) = self.process_input(model_input)
        return {
            "x": token_sequence,
            "coords_mapping": coords_mapping,
            "modality_mapping": modality_mapping,
            "varlen_handler": varlen_handler,
            "time_token_sequence": time_token_sequence,
        }

    def unpack_output(
        self, dit_out: torch.Tensor, shapes: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack DiT output ``(total_tokens, 64)`` -> ``(x_video, x_audio)``.

        Relies on the packing metadata saved by the most recent
        :meth:`build_model_inputs` / :meth:`process_input` call; ``shapes`` is
        accepted for API parity and ignored.
        """
        return self.process_output(dit_out)

    # -- length bookkeeping (upstream inference_engine._resolve_lengths) ----

    @staticmethod
    def resolve_lengths(
        seconds: float,
        fps: float,
        audio_latent_fps: float = 25.0,
        vae_stride: tuple[int, int, int] = (8, 16, 16),
    ) -> tuple[int, int]:
        """Reproduce ``inference_engine._resolve_lengths``.

        ``video_frame_length = round(seconds * fps * 2)``;
        ``video_latent_length = (video_frame_length - 1) // vae_stride[0] + 1``;
        ``audio_latent_length = round(seconds * audio_latent_fps)``.
        Returns ``(video_latent_length, audio_latent_length)``.
        """
        seconds = float(seconds)
        video_frame_length = int(round(seconds * fps * 2))
        video_latent_length = (video_frame_length - 1) // vae_stride[0] + 1
        audio_latent_length = int(round(seconds * audio_latent_fps))
        return video_latent_length, audio_latent_length

    @classmethod
    def latent_grid_dims(
        cls,
        seconds: float,
        fps: float,
        resolution: tuple[int, int],
        audio_latent_fps: float = 25.0,
        vae_stride: tuple[int, int, int] = (8, 16, 16),
    ) -> tuple[tuple[int, int, int], int]:
        """Latent grid dims ``(T, H, W)`` and audio length from clip metadata.

        ``resolution`` is ``(height, width)`` in pixels; the spatial latent dims
        are ``height // vae_stride[1]`` and ``width // vae_stride[2]``. Returns
        ``((T, H, W), audio_latent_length)``.
        """
        video_latent_length, audio_latent_length = cls.resolve_lengths(
            seconds, fps, audio_latent_fps=audio_latent_fps, vae_stride=vae_stride
        )
        height, width = resolution
        latent_h = height // vae_stride[1]
        latent_w = width // vae_stride[2]
        return (video_latent_length, latent_h, latent_w), audio_latent_length


__all__ = [
    "Magi2PreviewDataProxy",
    "Magi2PreviewDataProxyConfig",
    "ModelInput",
    "SimplePackedData",
    "SingleData",
    "sinusoidal_embedding_1d",
]


def _smoke() -> None:
    """Tiny-config CPU round-trip: build inputs -> DiT -> unpack; assert shapes."""
    from .config import (
        AttentionSinksConfig,
        MHCConfig,
        Magi2PreviewConfig,
        MoEConfig,
    )
    from .preview_dit import Transformer

    torch.manual_seed(0)
    # tiny video 48ch T=2,H=4,W=4; audio L=6,64ch; text S=3,5120.
    b, vt, vh, vw = 1, 2, 4, 4
    audio_len, text_len = 6, 3
    latent = torch.randn(b, 48, vt, vh, vw, dtype=_FP32)
    audio_latent = torch.randn(b, audio_len, 64, dtype=_FP32)
    context = torch.randn(b, text_len, 5120, dtype=_FP32)
    t = torch.full((b,), 0.5, dtype=_FP32)

    proxy = Magi2PreviewDataProxy(Magi2PreviewDataProxyConfig())
    kw = proxy.build_model_inputs(latent, audio_latent, context, t=t)

    n_video = vt * vh * vw
    total = n_video + audio_len + text_len
    assert kw["x"].shape[0] == total, kw["x"].shape
    assert kw["coords_mapping"].shape == (total, 9), kw["coords_mapping"].shape
    assert kw["modality_mapping"].shape == (total,), kw["modality_mapping"].shape
    assert kw["time_token_sequence"].shape == (total, 64), kw["time_token_sequence"].shape
    print(
        f"[smoke] packed x={tuple(kw['x'].shape)} coords={tuple(kw['coords_mapping'].shape)} "
        f"modality={tuple(kw['modality_mapping'].shape)} "
        f"time={tuple(kw['time_token_sequence'].shape)}"
    )

    # tiny 4-layer DiT matching a small preview config.
    moe_cfg = MoEConfig(
        num_experts=8,
        top_k=2,
        moe_layers=(2, 3),
        expert_intermediate_size=64,
        shared_expert_intermediate_size=64,
        modality_specific_expert_intermediate_size=64,
        num_heads=2,
        route_scale=1.0,
    )
    cfg = Magi2PreviewConfig(
        num_layers=4,
        hidden_size=256,
        head_dim=128,
        num_query_groups=2,
        mm_layers=(0, 1, 2, 3),
        mhc_config=MHCConfig(num_stream=2),
        moe_config=moe_cfg,
        attn_sinks=AttentionSinksConfig(enable=True, sink_token_num=1),
    )
    model = Transformer(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.02)
        dit_out = model(**kw)
    assert dit_out.shape == (total, cfg.audio_in_channels), dit_out.shape

    x_video, x_audio = proxy.unpack_output(dit_out)
    assert x_video.shape == (b, 48, vt, vh, vw), x_video.shape
    assert x_audio.shape == (b, audio_len, 64), x_audio.shape
    assert torch.isfinite(x_video).all() and torch.isfinite(x_audio).all()

    grid, audio_l = Magi2PreviewDataProxy.latent_grid_dims(
        seconds=2.0, fps=12.5, resolution=(256, 256)
    )
    print(
        f"[smoke] dit_out={tuple(dit_out.shape)} -> video={tuple(x_video.shape)} "
        f"audio={tuple(x_audio.shape)} finite=True; "
        f"resolve_lengths(2s@12.5fps,256x256)-> grid={grid} audio_len={audio_l}"
    )


if __name__ == "__main__":
    _smoke()
