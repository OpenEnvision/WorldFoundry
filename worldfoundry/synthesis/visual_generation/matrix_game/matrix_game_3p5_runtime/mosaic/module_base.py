"""Inference-only Matrix-Game 3.5 module initialization."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy

from .config import _normalize_mosaic_fuse_mode


class _MosaicModuleBase:
    def __init__(
        self,
        *,
        model_id="matrix-game-3.5-first-person",
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        trained_dit=None,
        device="cpu",
        use_prope=False,
        prope_attention_interval=1,
        prope_disable_native_rope=False,
        prope_disable_t_rope=False,
        prope_camera_layout="full",
        only_prope=False,
        trans_scale=50.0,
        enable_mosaic=True,
        mosaic_use_revgrid_rope=False,
        mosaic_view_change_prope=False,
        mosaic_fuse_mode="fill_stop_zbuffer",
        mosaic_fuse_block_size=2,
        mosaic_return_source_frame_ids=False,
        mosaic_drop_holes=True,
        mosaic_query_reference_frame=4,
        candidates_per_query_group=5,
        memory_vae_encode_input_frames=1,
        subject_ref_memory=False,
        subject_ref_time_gap=1,
        subject_ref_prope_mode="identity",
        subject_num_refs_max=2,
        subject_ref_canvas_slot_ratio=0.5,
        vae_decode_tiled=False,
        allow_no_prompt=False,
    ):
        super().__init__()
        use_prope = bool(use_prope or only_prope)
        enable_mosaic = bool(enable_mosaic and not only_prope)
        mosaic_use_revgrid_rope = bool(mosaic_use_revgrid_rope and enable_mosaic)
        mosaic_view_change_prope = bool(mosaic_view_change_prope and enable_mosaic and use_prope)

        checkpoint_overrides = self._checkpoint_overrides(
            model_paths=model_paths,
            model_id_with_origin_paths=model_id_with_origin_paths,
            tokenizer_path=tokenizer_path,
            trained_dit=trained_dit,
        )
        self.pipeline = NativeDiffusionPipeline.from_pretrained(
            model_id,
            policy=RuntimePolicy(device=device, dtype=torch.bfloat16),
            checkpoint_overrides=checkpoint_overrides,
            component_options={
                "denoiser:main": {
                    "use_prope": use_prope,
                    "prope_attention_interval": prope_attention_interval,
                    "prope_disable_native_rope": prope_disable_native_rope,
                    "prope_disable_t_rope": prope_disable_t_rope,
                    "prope_camera_layout": prope_camera_layout,
                    "trans_scale": trans_scale,
                },
                "decoder:main": {"tiled": vae_decode_tiled},
            },
        )
        components = self.pipeline.components
        self.dit = components.denoiser.model
        self.text_encoder = components.conditioner.text_encoder
        self.codec = components.decoder
        self.vae = self.codec.vae
        self.vae_dtype = self.codec.dtype
        self.tokenizer = components.conditioner.tokenizer
        self.device = self.pipeline.device
        self.dtype = self.pipeline.dtype
        self.torch_dtype = self.pipeline.dtype

        self.trans_scale = self._parse_trans_scale(trans_scale)
        self.subject_ref_memory = bool(subject_ref_memory)
        self.subject_ref_time_gap = max(1, int(subject_ref_time_gap))
        self.subject_ref_prope_mode = str(subject_ref_prope_mode or "identity")
        self.subject_num_refs_max = max(1, int(subject_num_refs_max))
        self.subject_ref_canvas_slot_ratio = min(
            1.0,
            max(0.05, float(subject_ref_canvas_slot_ratio)),
        )
        checkpoint_ref_capacity = int(self.dit.subject_ref_index_embedding.shape[0])
        if self.subject_num_refs_max > checkpoint_ref_capacity:
            self.subject_num_refs_max = checkpoint_ref_capacity
        self.requires_grad_(False)
        self.eval()

        self.enable_mosaic = enable_mosaic
        self.only_prope = bool(only_prope)
        self.mosaic_use_revgrid_rope = mosaic_use_revgrid_rope
        self.mosaic_view_change_prope = mosaic_view_change_prope
        self.mosaic_drop_holes = bool(mosaic_drop_holes)
        self.mosaic_return_source_frame_ids = bool(mosaic_return_source_frame_ids)
        self.mosaic_fuse_block_size = int(mosaic_fuse_block_size)
        if self.mosaic_fuse_block_size not in (1, 2):
            raise ValueError("mosaic_fuse_block_size must be 1 or 2")
        self.mosaic_query_reference_frame = int(mosaic_query_reference_frame)
        if self.mosaic_query_reference_frame not in (1, 2, 3, 4):
            raise ValueError("mosaic_query_reference_frame must be in [1, 4]")

        raw_fuse_mode = str(mosaic_fuse_mode or "fill_stop_zbuffer")
        self.mosaic_fuse_mode = _normalize_mosaic_fuse_mode(raw_fuse_mode)
        if self.mosaic_fuse_mode == "random":
            raise ValueError("Matrix-Game inference does not support random fuse mode")
        self.memory_vae_encode_input_frames = int(memory_vae_encode_input_frames)
        if self.memory_vae_encode_input_frames < 1 or (self.memory_vae_encode_input_frames - 1) % 4:
            raise ValueError("memory_vae_encode_input_frames must have the form 1+4k")

        self.vae_decode_tiled = bool(vae_decode_tiled)
        self.allow_no_prompt = bool(allow_no_prompt)
        self.log_dir = None

    @staticmethod
    def _checkpoint_overrides(
        *,
        model_paths,
        model_id_with_origin_paths,
        tokenizer_path,
        trained_dit,
    ):
        overrides = {}
        if trained_dit:
            overrides["dit"] = str(Path(trained_dit).expanduser())
        if tokenizer_path:
            tokenizer_root = Path(tokenizer_path).expanduser()
            # The shared Wan recipe owns the canonical
            # ``google/umt5-xxl/*`` file layout.  Matrix-Game's CLI receives
            # the leaf tokenizer directory for upstream compatibility, so
            # normalize it back to the Wan checkpoint root before overriding
            # the recipe.  Otherwise both layers append the same subdirectory.
            if tokenizer_root.name == "umt5-xxl" and tokenizer_root.parent.name == "google":
                tokenizer_root = tokenizer_root.parent.parent
            overrides["tokenizer"] = str(tokenizer_root)

        sources = []
        if model_paths:
            sources.extend(json.loads(model_paths))
        if model_id_with_origin_paths:
            sources.extend(str(model_id_with_origin_paths).split(","))
        for source in sources:
            if isinstance(source, (list, tuple)):
                continue
            normalized = str(source).lower()
            if "umt5" in normalized:
                overrides["text-encoder"] = str(source)
            elif "vae" in normalized:
                overrides["vae"] = str(source)
        return overrides

    def _activate_components(self, roles):
        """Compatibility hook; placement is owned by the shared runtime policy."""

        del roles

    def load_models_to_device(self, roles):
        self._activate_components(roles)

    @staticmethod
    def _parse_trans_scale(trans_scale):
        if isinstance(trans_scale, str):
            value = trans_scale.strip().lower()
            if value in {"log", "logd4", "tanh"}:
                return value
            try:
                return float(value)
            except ValueError as exc:
                raise ValueError("trans_scale must be numeric, 'log', 'logd4', or 'tanh'") from exc
        return float(trans_scale)


__all__ = ["_MosaicModuleBase"]
