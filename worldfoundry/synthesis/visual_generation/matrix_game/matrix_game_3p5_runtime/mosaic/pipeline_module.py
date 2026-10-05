import torch

from .module_base import _MosaicModuleBase
from .module_forward import _ForwardMixin
from .module_materialize import _MaterializeMixin
from .module_mixins import _InferenceTensorMixin


class WanMosaicPipelineModule(
    _ForwardMixin,
    _MaterializeMixin,
    _InferenceTensorMixin,
    _MosaicModuleBase,
    torch.nn.Module,
):
    pass


def build_mosaic_pipeline_module(args, module_cls=None, *, device=None):
    """Build the inference module, with an injectable memory-research subclass."""
    module_cls = module_cls or WanMosaicPipelineModule
    device = getattr(args, "device", "cpu") if device is None else device
    return module_cls(
        model_id=args.model_id,
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trained_dit=args.trained_dit,
        device=device,
        use_prope=args.use_prope,
        prope_attention_interval=args.prope_attention_interval,
        prope_disable_native_rope=args.prope_disable_native_rope,
        prope_disable_t_rope=args.prope_disable_t_rope,
        prope_camera_layout=args.prope_camera_layout,
        only_prope=args.only_prope,
        trans_scale=args.trans_scale,
        enable_mosaic=args.enable_mosaic,
        mosaic_use_revgrid_rope=args.mosaic_use_revgrid_rope,
        mosaic_view_change_prope=args.mosaic_view_change_prope,
        mosaic_fuse_mode=args.mosaic_fuse_mode,
        mosaic_fuse_block_size=args.mosaic_fuse_block_size,
        mosaic_return_source_frame_ids=args.mosaic_return_source_frame_ids,
        mosaic_drop_holes=args.mosaic_drop_holes,
        mosaic_query_reference_frame=args.mosaic_query_reference_frame,
        candidates_per_query_group=args.candidates_per_query_group,
        memory_vae_encode_input_frames=args.memory_vae_encode_input_frames,
        subject_ref_memory=args.subject_ref_memory,
        subject_ref_time_gap=args.subject_ref_time_gap,
        subject_ref_prope_mode=args.subject_ref_prope_mode,
        subject_num_refs_max=args.subject_num_refs_max,
        subject_ref_canvas_slot_ratio=args.subject_ref_canvas_slot_ratio,
        vae_decode_tiled=args.vae_decode_tiled,
        allow_no_prompt=args.allow_no_prompt,
    )


__all__ = ["WanMosaicPipelineModule", "build_mosaic_pipeline_module"]
