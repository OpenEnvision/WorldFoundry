# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See repository-root THIRD-PARTY-NOTICES for details.

"""Module for base_models -> perception_core -> tracking -> track_anything -> aot -> networks -> models -> aot.py functionality."""

import torch.nn as nn

from ..decoders import build_decoder
from ..encoders import build_encoder
from ..layers.position import PositionEmbeddingSine
from ..layers.transformer import LongShortTermTransformer


class AOT(nn.Module):
    """Aot implementation."""
    def __init__(self, cfg, encoder="mobilenetv2", decoder="fpn"):
        """Init.

        Args:
            cfg: The cfg.
            encoder: The encoder.
            decoder: The decoder.
        """
        super().__init__()
        self.cfg = cfg
        self.max_obj_num = cfg.MODEL_MAX_OBJ_NUM
        self.epsilon = cfg.MODEL_EPSILON

        self.encoder = build_encoder(
            encoder,
            frozen_bn=cfg.MODEL_FREEZE_BN,
            freeze_at=cfg.TRAIN_ENCODER_FREEZE_AT,
        )
        self.encoder_projector = nn.Conv2d(cfg.MODEL_ENCODER_DIM[-1], cfg.MODEL_ENCODER_EMBEDDING_DIM, kernel_size=1)

        self.LSTT = LongShortTermTransformer(
            cfg.MODEL_LSTT_NUM,
            cfg.MODEL_ENCODER_EMBEDDING_DIM,
            cfg.MODEL_SELF_HEADS,
            cfg.MODEL_ATT_HEADS,
            emb_dropout=cfg.TRAIN_LSTT_EMB_DROPOUT,
            droppath=cfg.TRAIN_LSTT_DROPPATH,
            lt_dropout=cfg.TRAIN_LSTT_LT_DROPOUT,
            st_dropout=cfg.TRAIN_LSTT_ST_DROPOUT,
            droppath_lst=cfg.TRAIN_LSTT_DROPPATH_LST,
            droppath_scaling=cfg.TRAIN_LSTT_DROPPATH_SCALING,
            intermediate_norm=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
            return_intermediate=True,
        )

        decoder_indim = (
            cfg.MODEL_ENCODER_EMBEDDING_DIM * (cfg.MODEL_LSTT_NUM + 1)
            if cfg.MODEL_DECODER_INTERMEDIATE_LSTT
            else cfg.MODEL_ENCODER_EMBEDDING_DIM
        )

        self.decoder = build_decoder(
            decoder,
            in_dim=decoder_indim,
            out_dim=cfg.MODEL_MAX_OBJ_NUM + 1,
            decode_intermediate_input=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
            hidden_dim=cfg.MODEL_ENCODER_EMBEDDING_DIM,
            shortcut_dims=cfg.MODEL_ENCODER_DIM,
            align_corners=cfg.MODEL_ALIGN_CORNERS,
        )

        if cfg.MODEL_ALIGN_CORNERS:
            self.patch_wise_id_bank = nn.Conv2d(
                cfg.MODEL_MAX_OBJ_NUM + 1,
                cfg.MODEL_ENCODER_EMBEDDING_DIM,
                kernel_size=17,
                stride=16,
                padding=8,
            )
        else:
            self.patch_wise_id_bank = nn.Conv2d(
                cfg.MODEL_MAX_OBJ_NUM + 1,
                cfg.MODEL_ENCODER_EMBEDDING_DIM,
                kernel_size=16,
                stride=16,
                padding=0,
            )

        self.id_dropout = nn.Dropout(cfg.TRAIN_LSTT_ID_DROPOUT, True)

        self.pos_generator = PositionEmbeddingSine(cfg.MODEL_ENCODER_EMBEDDING_DIM // 2, normalize=True)

        self._init_weight()

    def get_pos_emb(self, x):
        """Get pos emb.

        Args:
            x: The x.
        """
        pos_emb = self.pos_generator(x)
        return pos_emb

    def get_id_emb(self, x):
        """Get id emb.

        Args:
            x: The x.
        """
        id_emb = self.patch_wise_id_bank(x)
        id_emb = self.id_dropout(id_emb)
        return id_emb

    def encode_image(self, img):
        """Encode image.

        Args:
            img: The img.
        """
        xs = self.encoder(img)
        xs[-1] = self.encoder_projector(xs[-1])
        return xs

    def decode_id_logits(self, lstt_emb, shortcuts):
        """Decode id logits.

        Args:
            lstt_emb: The lstt emb.
            shortcuts: The shortcuts.
        """
        n, c, h, w = shortcuts[-1].size()
        decoder_inputs = [shortcuts[-1]]
        for emb in lstt_emb:
            decoder_inputs.append(emb.view(h, w, n, c).permute(2, 3, 0, 1))
        pred_logit = self.decoder(decoder_inputs, shortcuts)
        return pred_logit

    def LSTT_forward(
        self,
        curr_embs,
        long_term_memories,
        short_term_memories,
        curr_id_emb=None,
        pos_emb=None,
        size_2d=(30, 30),
    ):
        """Lstt forward.

        Args:
            curr_embs: The curr embs.
            long_term_memories: The long term memories.
            short_term_memories: The short term memories.
            curr_id_emb: The curr id emb.
            pos_emb: The pos emb.
            size_2d: The size 2d.
        """
        n, c, h, w = curr_embs[-1].size()
        curr_emb = curr_embs[-1].view(n, c, h * w).permute(2, 0, 1)
        lstt_embs, lstt_memories = self.LSTT(
            curr_emb,
            long_term_memories,
            short_term_memories,
            curr_id_emb,
            pos_emb,
            size_2d,
        )
        lstt_curr_memories, lstt_long_memories, lstt_short_memories = zip(*lstt_memories)
        return lstt_embs, lstt_curr_memories, lstt_long_memories, lstt_short_memories

    def _init_weight(self):
        """Helper function to init weight."""
        nn.init.xavier_uniform_(self.encoder_projector.weight)
        nn.init.orthogonal_(
            self.patch_wise_id_bank.weight.view(self.cfg.MODEL_ENCODER_EMBEDDING_DIM, -1).permute(0, 1),
            gain=17**-2 if self.cfg.MODEL_ALIGN_CORNERS else 16**-2,
        )
