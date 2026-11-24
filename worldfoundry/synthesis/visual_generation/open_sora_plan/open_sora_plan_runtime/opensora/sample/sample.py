import torch

try:
    import torch_npu

    from worldfoundry.synthesis.visual_generation.open_sora_plan.open_sora_plan_runtime.opensora.npu_config import (
        npu_config,
    )
except BaseException:
    torch_npu = None
    npu_config = None
from worldfoundry.synthesis.visual_generation.open_sora_plan.open_sora_plan_runtime.opensora.sample.caption_refiner import (
    OpenSoraCaptionRefiner,
)
from worldfoundry.synthesis.visual_generation.open_sora_plan.open_sora_plan_runtime.opensora.utils.sample_utils import (
    get_args,
    init_gpu_env,
    init_npu_env,
    prepare_pipeline,
    run_model_and_save_samples,
    run_model_and_save_samples_npu,
)

if __name__ == "__main__":
    args = get_args()
    dtype = torch.float16

    if torch_npu is not None:
        npu_config.print_msg(args)
        npu_config.conv_dtype = dtype
        init_npu_env(args)
    else:
        args = init_gpu_env(args)

    device = torch.cuda.current_device()
    if args.num_frames != 1 and args.enhance_video is not None:
        raise NotImplementedError("Video enhancement is not available in this in-tree build; the shipped route is v1.3")
    else:
        enhance_video_model = None
    pipeline = prepare_pipeline(args, dtype, device)
    caption_refiner_model = OpenSoraCaptionRefiner(args, dtype, device) if args.caption_refiner is not None else None

    if npu_config is not None and npu_config.on_npu and npu_config.profiling:
        run_model_and_save_samples_npu(args, pipeline, caption_refiner_model, enhance_video_model)
    else:
        run_model_and_save_samples(args, pipeline, caption_refiner_model, enhance_video_model)
