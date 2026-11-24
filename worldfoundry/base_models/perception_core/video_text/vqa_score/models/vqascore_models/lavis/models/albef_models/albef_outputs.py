"""
Copyright (c) 2022, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from dataclasses import dataclass

import torch
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    ModelOutput,
)


@dataclass
class AlbefSimilarity(ModelOutput):
    sim_i2t: torch.FloatTensor = None
    sim_t2i: torch.FloatTensor = None

    sim_i2t_m: torch.FloatTensor | None = None
    sim_t2i_m: torch.FloatTensor | None = None

    sim_i2t_targets: torch.FloatTensor | None = None
    sim_t2i_targets: torch.FloatTensor | None = None


@dataclass
class AlbefIntermediateOutput(ModelOutput):
    # uni-modal features
    image_embeds: torch.FloatTensor = None
    text_embeds: torch.FloatTensor | None = None

    image_embeds_m: torch.FloatTensor | None = None
    text_embeds_m: torch.FloatTensor | None = None

    # intermediate outputs of multimodal encoder
    encoder_output: BaseModelOutputWithPoolingAndCrossAttentions | None = None
    encoder_output_m: BaseModelOutputWithPoolingAndCrossAttentions | None = None
    encoder_output_neg: BaseModelOutputWithPoolingAndCrossAttentions | None = None

    itm_logits: torch.FloatTensor | None = None
    itm_labels: torch.LongTensor | None = None

    # intermediate outputs of multimodal decoder
    decoder_output: CausalLMOutputWithCrossAttentions | None = None
    decoder_labels: torch.LongTensor | None = None


@dataclass
class AlbefOutput(ModelOutput):
    # some finetuned models (e.g. BlipVQA) do not compute similarity, thus optional.
    sims: AlbefSimilarity | None = None

    intermediate_output: AlbefIntermediateOutput = None

    loss: torch.FloatTensor | None = None

    loss_itc: torch.FloatTensor | None = None

    loss_itm: torch.FloatTensor | None = None

    loss_mlm: torch.FloatTensor | None = None


@dataclass
class AlbefOutputWithLogits(AlbefOutput):
    logits: torch.FloatTensor = None
    logits_m: torch.FloatTensor = None


@dataclass
class AlbefOutputFeatures(ModelOutput):
    """
    Data class of features from AlbefFeatureExtractor.

    Args:
        image_embeds: `torch.FloatTensor` of shape `(batch_size, num_patches+1, embed_dim)`, `optional`
        image_features: `torch.FloatTensor` of shape `(batch_size, num_patches+1, feature_dim)`, `optional`
        text_embeds: `torch.FloatTensor` of shape `(batch_size, sequence_length+1, embed_dim)`, `optional`
        text_features: `torch.FloatTensor` of shape `(batch_size, sequence_length+1, feature_dim)`, `optional`

        The first embedding or feature is for the [CLS] token.

        Features are obtained by projecting the corresponding embedding into a normalized low-dimensional space.
    """

    image_embeds: torch.FloatTensor | None = None
    image_embeds_proj: torch.FloatTensor | None = None

    text_embeds: torch.FloatTensor | None = None
    text_embeds_proj: torch.FloatTensor | None = None

    multimodal_embeds: torch.FloatTensor | None = None
