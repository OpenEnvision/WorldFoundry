"""Module for base_models -> three_dimensions -> point_clouds -> cut3r -> model.py functionality."""

import os
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.file_utils import ModelOutput

from .blocks import (
    Block,
    DecoderBlock,
)  # noqa
from .croco_base import CrocoConfig, CroCoNet  # noqa
from .heads import head_factory
from .patch_embed import get_patch_embed
from .utils.misc import (
    fill_default_args,
    transpose_to_landscape,
)


@dataclass
class ARCroco3DStereoOutput(ModelOutput):
    """
    Custom output class for ARCroco3DStereo.
    """

    ress: Optional[List[Any]] = None
    views: Optional[List[Any]] = None


def strip_module(state_dict):
    """
    Removes the 'module.' prefix from the keys of a state_dict.
    Args:
        state_dict (dict): The original state_dict with possible 'module.' prefixes.
    Returns:
        OrderedDict: A new state_dict with 'module.' prefixes removed.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


def _checkpoint_config(expression):
    """Decode the official constructor string using a restricted AST grammar."""
    import ast

    def decode(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id == "inf":
            return float("inf")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = decode(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, (ast.Tuple, ast.List)):
            values = [decode(x) for x in node.elts]
            return tuple(values) if isinstance(node, ast.Tuple) else values
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "ARCroco3DStereo" and len(node.args) == 1 and not node.keywords:
                return decode(node.args[0])
            if node.func.id == "ARCroco3DStereoConfig" and not node.args:
                if any(k.arg is None for k in node.keywords):
                    raise ValueError("Expanded checkpoint arguments are unsupported")
                options = {k.arg: decode(k.value) for k in node.keywords}
                options["patch_embed_cls"] = options.get("patch_embed_cls", "PatchEmbedDust3R").replace(
                    "ManyAR_PatchEmbed", "PatchEmbedDust3R"
                )
                options["landscape_only"] = False
                return ARCroco3DStereoConfig(**options)
        raise ValueError("Unsupported CUT3R checkpoint architecture expression")

    config = decode(ast.parse(expression, mode="eval").body)
    if not isinstance(config, ARCroco3DStereoConfig):
        raise ValueError("Checkpoint must specify ARCroco3DStereoConfig")
    return config


def load_model(model_path, device, verbose=True):
    """Load an official, trusted CUT3R pickle checkpoint with exact key validation."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    config = _checkpoint_config(ckpt["args"].model)
    net = ARCroco3DStereo(config)
    net.load_state_dict(strip_module(ckpt["model"]), strict=True)
    return net.to(device).eval().requires_grad_(False)


class ARCroco3DStereoConfig(PretrainedConfig):
    """Ar croco d stereo config implementation."""

    model_type = "arcroco_3d_stereo"

    def __init__(
        self,
        output_mode="pts3d",
        head_type="linear",  # or dpt
        depth_mode=("exp", -float("inf"), float("inf")),
        conf_mode=("exp", 1, float("inf")),
        pose_mode=("exp", -float("inf"), float("inf")),
        freeze="none",
        landscape_only=True,
        patch_embed_cls="PatchEmbedDust3R",
        ray_enc_depth=2,
        state_size=324,
        local_mem_size=256,
        state_pe="2d",
        state_dec_num_heads=16,
        depth_head=False,
        rgb_head=False,
        pose_conf_head=False,
        pose_head=False,
        **croco_kwargs,
    ):
        """Init.

        Args:
            output_mode: The output mode.
            head_type: The head type.
            depth_mode: The depth mode.
            conf_mode: The conf mode.
            pose_mode: The pose mode.
            freeze: The freeze.
            landscape_only: The landscape only.
            patch_embed_cls: The patch embed cls.
            ray_enc_depth: The ray enc depth.
            state_size: The state size.
            local_mem_size: The local mem size.
            state_pe: The state pe.
            state_dec_num_heads: The state dec num heads.
            depth_head: The depth head.
            rgb_head: The rgb head.
            pose_conf_head: The pose conf head.
            pose_head: The pose head.
        """
        super().__init__()
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.freeze = freeze
        self.landscape_only = landscape_only
        self.patch_embed_cls = patch_embed_cls
        self.ray_enc_depth = ray_enc_depth
        self.state_size = state_size
        self.state_pe = state_pe
        self.state_dec_num_heads = state_dec_num_heads
        self.local_mem_size = local_mem_size
        self.depth_head = depth_head
        self.rgb_head = rgb_head
        self.pose_conf_head = pose_conf_head
        self.pose_head = pose_head
        self.croco_kwargs = croco_kwargs


class LocalMemory(nn.Module):
    """Local memory implementation."""

    def __init__(
        self,
        size,
        k_dim,
        v_dim,
        num_heads,
        depth=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
    ) -> None:
        """Init.

        Args:
            size: The size.
            k_dim: The k dim.
            v_dim: The v dim.
            num_heads: The num heads.
            depth: The depth.
            mlp_ratio: The mlp ratio.
            qkv_bias: The qkv bias.
            drop: The drop.
            attn_drop: The attn drop.
            drop_path: The drop path.
            act_layer: The act layer.
            norm_layer: The norm layer.
            norm_mem: The norm mem.
            rope: The rope.

        Returns:
            The return value.
        """
        super().__init__()
        self.v_dim = v_dim
        self.proj_q = nn.Linear(k_dim, v_dim)
        self.masked_token = nn.Parameter(torch.randn(1, 1, v_dim) * 0.2, requires_grad=True)
        self.mem = nn.Parameter(torch.randn(1, size, 2 * v_dim) * 0.2, requires_grad=True)
        self.write_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )
        self.read_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )

    def update_mem(self, mem, feat_k, feat_v):
        """
        mem_k: [B, size, C]
        mem_v: [B, size, C]
        feat_k: [B, 1, C]
        feat_v: [B, 1, C]
        """
        feat_k = self.proj_q(feat_k)  # [B, 1, C]
        feat = torch.cat([feat_k, feat_v], dim=-1)
        for blk in self.write_blocks:
            mem, _ = blk(mem, feat, None, None)
        return mem

    def inquire(self, query, mem):
        """Inquire.

        Args:
            query: The query.
            mem: The mem.
        """
        x = self.proj_q(query)  # [B, 1, C]
        x = torch.cat([x, self.masked_token.expand(x.shape[0], -1, -1)], dim=-1)
        for blk in self.read_blocks:
            x, _ = blk(x, mem, None, None)
        return x[..., -self.v_dim :]


class ARCroco3DStereo(CroCoNet):
    """Ar croco d stereo implementation."""

    config_class = ARCroco3DStereoConfig
    base_model_prefix = "arcroco3dstereo"

    def __init__(self, config: ARCroco3DStereoConfig):
        """Init.

        Args:
            config: The config.
        """
        config.croco_kwargs = fill_default_args(config.croco_kwargs, CrocoConfig.__init__)
        self.config = config
        self.patch_embed_cls = config.patch_embed_cls
        self.croco_args = config.croco_kwargs
        croco_cfg = CrocoConfig(**self.croco_args)
        super().__init__(croco_cfg)
        self.enc_blocks_ray_map = nn.ModuleList(
            [
                Block(
                    self.enc_embed_dim,
                    16,
                    4,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    rope=self.rope,
                )
                for _ in range(config.ray_enc_depth)
            ]
        )
        self.enc_norm_ray_map = nn.LayerNorm(self.enc_embed_dim, eps=1e-6)
        self.dec_num_heads = self.croco_args["dec_num_heads"]
        self.pose_head_flag = config.pose_head
        if self.pose_head_flag:
            self.pose_token = nn.Parameter(torch.randn(1, 1, self.dec_embed_dim) * 0.02, requires_grad=True)
            self.pose_retriever = LocalMemory(
                size=config.local_mem_size,
                k_dim=self.enc_embed_dim,
                v_dim=self.dec_embed_dim,
                num_heads=self.dec_num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                attn_drop=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                rope=None,
            )
        self.register_tokens = nn.Embedding(config.state_size, self.enc_embed_dim)
        self.state_size = config.state_size
        self.state_pe = config.state_pe
        self.masked_img_token = nn.Parameter(torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True)
        self.masked_ray_map_token = nn.Parameter(torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True)
        self._set_state_decoder(
            self.enc_embed_dim,
            self.dec_embed_dim,
            config.state_dec_num_heads,
            self.dec_depth,
            self.croco_args.get("mlp_ratio", None),
            self.croco_args.get("norm_layer", None),
            self.croco_args.get("norm_im2_in_dec", None),
        )
        self.set_downstream_head(
            config.output_mode,
            config.head_type,
            config.landscape_only,
            config.depth_mode,
            config.conf_mode,
            config.pose_mode,
            config.depth_head,
            config.rgb_head,
            config.pose_conf_head,
            config.pose_head,
            **self.croco_args,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        """From pretrained.

        Args:
            pretrained_model_name_or_path: The pretrained model name or path.
        """
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device="cpu")
        else:
            try:
                model = super(ARCroco3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)
            except TypeError:
                raise Exception(f"tried to load {pretrained_model_name_or_path} from huggingface, but failed")
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        """Helper function to set patch embed.

        Args:
            img_size: The img size.
            patch_size: The patch size.
            enc_embed_dim: The enc embed dim.
        """
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3)
        self.patch_embed_ray_map = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=6
        )

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        """Helper function to set decoder.

        Args:
            enc_embed_dim: The enc embed dim.
            dec_embed_dim: The dec embed dim.
            dec_num_heads: The dec num heads.
            dec_depth: The dec depth.
            mlp_ratio: The mlp ratio.
            norm_layer: The norm layer.
            norm_im2_in_dec: The norm im2 in dec.
        """
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_state_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        """Helper function to set state decoder.

        Args:
            enc_embed_dim: The enc embed dim.
            dec_embed_dim: The dec embed dim.
            dec_num_heads: The dec num heads.
            dec_depth: The dec depth.
            mlp_ratio: The mlp ratio.
            norm_layer: The norm layer.
            norm_im2_in_dec: The norm im2 in dec.
        """
        self.dec_depth_state = dec_depth
        self.dec_embed_dim_state = dec_embed_dim
        self.decoder_embed_state = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks_state = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm_state = norm_layer(dec_embed_dim)

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(
        self,
        output_mode,
        head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        pose_mode,
        depth_head,
        rgb_head,
        pose_conf_head,
        pose_head,
        patch_size,
        img_size,
        **kw,
    ):
        """Set downstream head.

        Args:
            output_mode: The output mode.
            head_type: The head type.
            landscape_only: The landscape only.
            depth_mode: The depth mode.
            conf_mode: The conf mode.
            pose_mode: The pose mode.
            depth_head: The depth head.
            rgb_head: The rgb head.
            pose_conf_head: The pose conf head.
            pose_head: The pose head.
            patch_size: The patch size.
            img_size: The img size.
        """
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, (
            f"{img_size=} must be multiple of {patch_size=}"
        )
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.downstream_head = head_factory(
            head_type,
            output_mode,
            self,
            has_conf=bool(conf_mode),
            has_depth=bool(depth_head),
            has_rgb=bool(rgb_head),
            has_pose_conf=bool(pose_conf_head),
            has_pose=bool(pose_head),
        )
        self.head = transpose_to_landscape(self.downstream_head, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        """Helper function to encode image.

        Args:
            image: The image.
            true_shape: The true shape.
        """
        x, pos = self.patch_embed(image, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks:
            x = blk(x, pos)
        x = self.enc_norm(x)
        return [x], pos, None

    def _encode_ray_map(self, ray_map, true_shape):
        """Helper function to encode ray map.

        Args:
            ray_map: The ray map.
            true_shape: The true shape.
        """
        x, pos = self.patch_embed_ray_map(ray_map, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks_ray_map:
            x = blk(x, pos)
        x = self.enc_norm_ray_map(x)
        return [x], pos, None

    def _encode_state(self, image_tokens, image_pos):
        """Helper function to encode state.

        Args:
            image_tokens: The image tokens.
            image_pos: The image pos.
        """
        batch_size = image_tokens.shape[0]
        state_feat = self.register_tokens(torch.arange(self.state_size, device=image_pos.device))
        if self.state_pe == "1d":
            state_pos = (
                torch.tensor(
                    [[i, i] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )  # .long()
        elif self.state_pe == "2d":
            width = int(self.state_size**0.5)
            width = width + 1 if width % 2 == 1 else width
            state_pos = (
                torch.tensor(
                    [[i // width, i % width] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )
        elif self.state_pe == "none":
            state_pos = None
        state_feat = state_feat[None].expand(batch_size, -1, -1)
        return state_feat, state_pos, None

    def _encode_views(self, views, img_mask=None, ray_mask=None):
        """Helper function to encode views.

        Args:
            views: The views.
            img_mask: The img mask.
            ray_mask: The ray mask.
        """
        device = views[0]["img"].device
        batch_size = views[0]["img"].shape[0]
        given = True
        if img_mask is None and ray_mask is None:
            given = False
        if not given:
            img_mask = torch.stack([view["img_mask"] for view in views], dim=0)  # Shape: (num_views, batch_size)
            ray_mask = torch.stack([view["ray_mask"] for view in views], dim=0)  # Shape: (num_views, batch_size)
        imgs = torch.stack([view["img"] for view in views], dim=0)  # Shape: (num_views, batch_size, C, H, W)
        ray_maps = torch.stack([view["ray_map"] for view in views], dim=0)  # Shape: (num_views, batch_size, H, W, C)
        shapes = []
        for view in views:
            if "true_shape" in view:
                shapes.append(view["true_shape"])
            else:
                shape = torch.tensor(view["img"].shape[-2:], device=device)
                shapes.append(shape.unsqueeze(0).repeat(batch_size, 1))
        shapes = torch.stack(shapes, dim=0).to(imgs.device)  # Shape: (num_views, batch_size, 2)
        imgs = imgs.view(-1, *imgs.shape[2:])  # Shape: (num_views * batch_size, C, H, W)
        ray_maps = ray_maps.view(-1, *ray_maps.shape[2:])  # Shape: (num_views * batch_size, H, W, C)
        shapes = shapes.view(-1, 2)  # Shape: (num_views * batch_size, 2)
        img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
        ray_masks_flat = ray_mask.view(-1)
        selected_imgs = imgs[img_masks_flat]
        selected_shapes = shapes[img_masks_flat]
        if selected_imgs.size(0) > 0:
            img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
        else:
            raise NotImplementedError
        full_out = [
            torch.zeros(len(views) * batch_size, *img_out[0].shape[1:], device=img_out[0].device)
            for _ in range(len(img_out))
        ]
        full_pos = torch.zeros(
            len(views) * batch_size,
            *img_pos.shape[1:],
            device=img_pos.device,
            dtype=img_pos.dtype,
        )
        for i in range(len(img_out)):
            full_out[i][img_masks_flat] += img_out[i]
            full_out[i][~img_masks_flat] += self.masked_img_token
        full_pos[img_masks_flat] += img_pos
        ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
        selected_ray_maps = ray_maps[ray_masks_flat]
        selected_shapes_ray = shapes[ray_masks_flat]
        if selected_ray_maps.size(0) > 0:
            ray_out, ray_pos, _ = self._encode_ray_map(selected_ray_maps, selected_shapes_ray)
            assert len(ray_out) == len(full_out), f"{len(ray_out)}, {len(full_out)}"
            for i in range(len(ray_out)):
                full_out[i][ray_masks_flat] += ray_out[i]
                full_out[i][~ray_masks_flat] += self.masked_ray_map_token
            full_pos[ray_masks_flat] += ray_pos * (~img_masks_flat[ray_masks_flat][:, None, None]).long()
        return (
            shapes.chunk(len(views), dim=0),
            [out.chunk(len(views), dim=0) for out in full_out],
            full_pos.chunk(len(views), dim=0),
        )

    def _decoder(self, f_state, pos_state, f_img, pos_img, f_pose, pos_pose):
        """Helper function to decoder.

        Args:
            f_state: The f state.
            pos_state: The pos state.
            f_img: The f img.
            pos_img: The pos img.
            f_pose: The f pose.
            pos_pose: The pos pose.
        """
        final_output = [(f_state, f_img)]  # before projection
        assert f_state.shape[-1] == self.dec_embed_dim
        f_img = self.decoder_embed(f_img)
        if self.pose_head_flag:
            assert f_pose is not None and pos_pose is not None
            f_img = torch.cat([f_pose, f_img], dim=1)
            pos_img = torch.cat([pos_pose, pos_img], dim=1)
        final_output.append((f_state, f_img))
        for blk_state, blk_img in zip(self.dec_blocks_state, self.dec_blocks):
            f_state, _ = blk_state(*final_output[-1][::+1], pos_state, pos_img)
            f_img, _ = blk_img(*final_output[-1][::-1], pos_img, pos_state)
            final_output.append((f_state, f_img))
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = (
            self.dec_norm_state(final_output[-1][0]),
            self.dec_norm(final_output[-1][1]),
        )
        return zip(*final_output)

    def _downstream_head(self, decout, img_shape, **kwargs):
        """Helper function to downstream head.

        Args:
            decout: The decout.
            img_shape: The img shape.
        """
        B, S, D = decout[-1].shape
        head = getattr(self, "head")
        return head(decout, img_shape, **kwargs)

    def _init_state(self, image_tokens, image_pos):
        """
        Current Version: input the first frame img feature and pose to initialize the state feature and pose
        """
        state_feat, state_pos, _ = self._encode_state(image_tokens, image_pos)
        state_feat = self.decoder_embed_state(state_feat)
        return state_feat, state_pos

    def _recurrent_rollout(
        self,
        state_feat,
        state_pos,
        current_feat,
        current_pos,
        pose_feat,
        pose_pos,
        init_state_feat,
        img_mask=None,
        reset_mask=None,
        update=None,
    ):
        """Helper function to recurrent rollout.

        Args:
            state_feat: The state feat.
            state_pos: The state pos.
            current_feat: The current feat.
            current_pos: The current pos.
            pose_feat: The pose feat.
            pose_pos: The pose pos.
            init_state_feat: The init state feat.
            img_mask: The img mask.
            reset_mask: The reset mask.
            update: The update.
        """
        new_state_feat, dec = self._decoder(state_feat, state_pos, current_feat, current_pos, pose_feat, pose_pos)
        new_state_feat = new_state_feat[-1]
        return new_state_feat, dec

    def _get_img_level_feat(self, feat):
        """Helper function to get img level feat.

        Args:
            feat: The feat.
        """
        return torch.mean(feat, dim=1, keepdim=True)

    def _forward_encoder(self, views):
        """Helper function to forward encoder.

        Args:
            views: The views.
        """
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        state_feat, state_pos = self._init_state(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        return (feat, pos, shape), (
            init_state_feat,
            init_mem,
            state_feat,
            state_pos,
            mem,
        )

    def _forward_decoder_step(
        self,
        views,
        i,
        feat_i,
        pos_i,
        shape_i,
        init_state_feat,
        init_mem,
        state_feat,
        state_pos,
        mem,
    ):
        """Helper function to forward decoder step.

        Args:
            views: The views.
            i: The i.
            feat_i: The feat i.
            pos_i: The pos i.
            shape_i: The shape i.
            init_state_feat: The init state feat.
            init_mem: The init mem.
            state_feat: The state feat.
            state_pos: The state pos.
            mem: The mem.
        """
        if self.pose_head_flag:
            global_img_feat_i = self._get_img_level_feat(feat_i)
            if i == 0:
                pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
            else:
                pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype)
        else:
            pose_feat_i = None
            pose_pos_i = None
        new_state_feat, dec = self._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            init_state_feat,
            img_mask=views[i]["img_mask"],
            reset_mask=views[i]["reset"],
            update=views[i].get("update", None),
        )
        out_pose_feat_i = dec[-1][:, 0:1]
        new_mem = self.pose_retriever.update_mem(mem, global_img_feat_i, out_pose_feat_i)
        head_input = [
            dec[0].float(),
            dec[self.dec_depth * 2 // 4][:, 1:].float(),
            dec[self.dec_depth * 3 // 4][:, 1:].float(),
            dec[self.dec_depth].float(),
        ]
        res = self._downstream_head(head_input, shape_i, pos=pos_i)
        img_mask = views[i]["img_mask"]
        update = views[i].get("update", None)
        if update is not None:
            update_mask = img_mask & update  # if don't update, then whatever img_mask
        else:
            update_mask = img_mask
        update_mask = update_mask[:, None, None].float()
        state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)  # update global state
        mem = new_mem * update_mask + mem * (1 - update_mask)  # then update local state
        reset_mask = views[i]["reset"]
        if reset_mask is not None:
            reset_mask = reset_mask[:, None, None].float()
            state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
            mem = init_mem * reset_mask + mem * (1 - reset_mask)
        return res, (state_feat, mem)

    def _forward_impl(self, views, ret_state=False):
        """Helper function to forward impl.

        Args:
            views: The views.
            ret_state: The ret state.
        """
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        state_feat, state_pos = self._init_state(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        all_state_args = [(state_feat, state_pos, init_state_feat, mem, init_mem)]
        ress = []
        for i in range(len(views)):
            feat_i = feat[i]
            pos_i = pos[i]
            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype)
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=views[i]["img_mask"],
                reset_mask=views[i]["reset"],
                update=views[i].get("update", None),
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(mem, global_img_feat_i, out_pose_feat_i)
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            res = self._downstream_head(head_input, shape[i], pos=pos_i)
            ress.append(res)
            img_mask = views[i]["img_mask"]
            update = views[i].get("update", None)
            if update is not None:
                update_mask = img_mask & update  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()
            state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)  # update global state
            mem = new_mem * update_mask + mem * (1 - update_mask)  # then update local state
            reset_mask = views[i]["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
            all_state_args.append((state_feat, state_pos, init_state_feat, mem, init_mem))
        if ret_state:
            return ress, views, all_state_args
        return ress, views

    def forward(self, views, ret_state=False):
        """Forward.

        Args:
            views: The views.
            ret_state: The ret state.
        """
        if ret_state:
            ress, views, state_args = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views), state_args
        else:
            ress, views = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views)

    def inference_step(self, view, state_feat, state_pos, init_state_feat, mem, init_mem):
        """Inference step.

        Args:
            view: The view.
            state_feat: The state feat.
            state_pos: The state pos.
            init_state_feat: The init state feat.
            mem: The mem.
            init_mem: The init mem.
        """
        batch_size = view["img"].shape[0]
        raymaps = []
        shapes = []
        for j in range(batch_size):
            assert view["ray_mask"][j]
            raymap = view["ray_map"][[j]].permute(0, 3, 1, 2)
            raymaps.append(raymap)
            shapes.append(
                view.get(
                    "true_shape",
                    torch.tensor(view["ray_map"].shape[-2:])[None].repeat(view["ray_map"].shape[0], 1),
                )[[j]]
            )

        raymaps = torch.cat(raymaps, dim=0)
        shape = torch.cat(shapes, dim=0).to(raymaps.device)
        feat_ls, pos, _ = self._encode_ray_map(raymaps, shape)

        feat_i = feat_ls[-1]
        pos_i = pos
        if self.pose_head_flag:
            global_img_feat_i = self._get_img_level_feat(feat_i)
            pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype)
        else:
            pose_feat_i = None
            pose_pos_i = None
        new_state_feat, dec = self._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            init_state_feat,
            img_mask=view["img_mask"],
            reset_mask=view["reset"],
            update=view.get("update", None),
        )

        out_pose_feat_i = dec[-1][:, 0:1]
        self.pose_retriever.update_mem(mem, global_img_feat_i, out_pose_feat_i)
        assert len(dec) == self.dec_depth + 1
        head_input = [
            dec[0].float(),
            dec[self.dec_depth * 2 // 4][:, 1:].float(),
            dec[self.dec_depth * 3 // 4][:, 1:].float(),
            dec[self.dec_depth].float(),
        ]
        res = self._downstream_head(head_input, shape, pos=pos_i)
        return res, view

    def forward_recurrent(self, views, device, ret_state=False):
        """Forward recurrent.

        Args:
            views: The views.
            device: The device.
            ret_state: The ret state.
        """
        ress = []
        all_state_args = []
        for i, view in enumerate(views):
            device = view["img"].device
            batch_size = view["img"].shape[0]
            img_mask = view["img_mask"].reshape(-1, batch_size)  # Shape: (1, batch_size)
            ray_mask = view["ray_mask"].reshape(-1, batch_size)  # Shape: (1, batch_size)
            imgs = view["img"].unsqueeze(0)  # Shape: (1, batch_size, C, H, W)
            ray_maps = view["ray_map"].unsqueeze(0)  # Shape: (num_views, batch_size, H, W, C)
            shapes = (
                view["true_shape"].unsqueeze(0)
                if "true_shape" in view
                else torch.tensor(view["img"].shape[-2:], device=device).unsqueeze(0).repeat(batch_size, 1).unsqueeze(0)
            )  # Shape: (num_views, batch_size, 2)
            imgs = imgs.view(-1, *imgs.shape[2:])  # Shape: (num_views * batch_size, C, H, W)
            ray_maps = ray_maps.view(-1, *ray_maps.shape[2:])  # Shape: (num_views * batch_size, H, W, C)
            shapes = shapes.view(-1, 2).to(imgs.device)  # Shape: (num_views * batch_size, 2)
            img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
            ray_masks_flat = ray_mask.view(-1)
            selected_imgs = imgs[img_masks_flat]
            selected_shapes = shapes[img_masks_flat]
            if selected_imgs.size(0) > 0:
                img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
            else:
                img_out, img_pos = None, None
            ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
            selected_ray_maps = ray_maps[ray_masks_flat]
            selected_shapes_ray = shapes[ray_masks_flat]
            if selected_ray_maps.size(0) > 0:
                ray_out, ray_pos, _ = self._encode_ray_map(selected_ray_maps, selected_shapes_ray)
            else:
                ray_out, ray_pos = None, None

            shape = shapes
            if img_out is not None and ray_out is None:
                feat_i = img_out[-1]
                pos_i = img_pos
            elif img_out is None and ray_out is not None:
                feat_i = ray_out[-1]
                pos_i = ray_pos
            elif img_out is not None and ray_out is not None:
                feat_i = img_out[-1] + ray_out[-1]
                pos_i = img_pos
            else:
                raise NotImplementedError

            if i == 0:
                state_feat, state_pos = self._init_state(feat_i, pos_i)
                mem = self.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()
                all_state_args.append((state_feat, state_pos, init_state_feat, mem, init_mem))

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype)
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=view["img_mask"],
                reset_mask=view["reset"],
                update=view.get("update", None),
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(mem, global_img_feat_i, out_pose_feat_i)
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            res = self._downstream_head(head_input, shape, pos=pos_i)
            ress.append(res)
            img_mask = view["img_mask"]
            update = view.get("update", None)
            if update is not None:
                update_mask = img_mask & update  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()
            state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)  # update global state
            mem = new_mem * update_mask + mem * (1 - update_mask)  # then update local state
            reset_mask = view["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
            all_state_args.append((state_feat, state_pos, init_state_feat, mem, init_mem))
        if ret_state:
            return ress, views, all_state_args
        return ress, views
