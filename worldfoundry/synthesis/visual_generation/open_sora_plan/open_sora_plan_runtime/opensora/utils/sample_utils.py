import argparse
import math
import os
import time
from pathlib import Path

import imageio
import torch
import torch.distributed as dist
from diffusers.schedulers import (
    CogVideoXDDIMScheduler,
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    PNDMScheduler,
)
from einops import rearrange
from huggingface_hub import snapshot_download
from torchvision.utils import save_image
from transformers import AutoModelForCausalLM as AutoModelForCausalLM

try:
    import torch_npu

    from worldfoundry.synthesis.visual_generation.open_sora_plan.open_sora_plan_runtime.opensora.acceleration.parallel_states import (
        hccl_info as hccl_info,
    )
    from worldfoundry.synthesis.visual_generation.open_sora_plan.open_sora_plan_runtime.opensora.acceleration.parallel_states import (
        initialize_sequence_parallel_state,
    )
    from worldfoundry.synthesis.visual_generation.open_sora_plan.open_sora_plan_runtime.opensora.npu_config import (
        npu_config,
    )
except BaseException:
    torch_npu = None
    npu_config = None
    from worldfoundry.synthesis.visual_generation.open_sora_plan.open_sora_plan_runtime.opensora.utils.parallel_states import (
        initialize_sequence_parallel_state,
    )
    from worldfoundry.synthesis.visual_generation.open_sora_plan.open_sora_plan_runtime.opensora.utils.parallel_states import (
        nccl_info as nccl_info,
    )


from transformers import AutoTokenizer, CLIPTextModelWithProjection, MT5EncoderModel, T5EncoderModel
from transformers import T5Tokenizer as T5Tokenizer

from worldfoundry.synthesis.visual_generation.open_sora_plan.open_sora_plan_runtime.opensora.models.causalvideovae import (
    ae_stride_config,
    ae_wrapper,
)
from worldfoundry.synthesis.visual_generation.open_sora_plan.open_sora_plan_runtime.opensora.models.diffusion.opensora_v1_3.modeling_inpaint import (
    OpenSoraInpaint_v1_3,
)
from worldfoundry.synthesis.visual_generation.open_sora_plan.open_sora_plan_runtime.opensora.models.diffusion.opensora_v1_3.modeling_opensora import (
    OpenSoraT2V_v1_3,
)
from worldfoundry.synthesis.visual_generation.open_sora_plan.open_sora_plan_runtime.opensora.sample.pipeline_inpaint import (
    OpenSoraInpaintPipeline,
)
from worldfoundry.synthesis.visual_generation.open_sora_plan.open_sora_plan_runtime.opensora.sample.pipeline_opensora import (
    OpenSoraPipeline,
)
from worldfoundry.synthesis.visual_generation.open_sora_plan.open_sora_plan_runtime.opensora.utils.utils import set_seed


def get_scheduler(args):
    kwargs = {
        "prediction_type": args.prediction_type,
        "rescale_betas_zero_snr": args.rescale_betas_zero_snr,
        "timestep_spacing": "trailing" if args.rescale_betas_zero_snr else "leading",
    }
    if args.v1_5_scheduler:
        kwargs["beta_start"] = 0.00085
        kwargs["beta_end"] = 0.0120
        kwargs["beta_schedule"] = "scaled_linear"
    if args.sample_method == "DDIM":
        scheduler_cls = DDIMScheduler
        kwargs["clip_sample"] = False
    elif args.sample_method == "EulerDiscrete":
        scheduler_cls = EulerDiscreteScheduler
    elif args.sample_method == "DDPM":
        scheduler_cls = DDPMScheduler
        kwargs["clip_sample"] = False
    elif args.sample_method == "DPMSolverMultistep":
        scheduler_cls = DPMSolverMultistepScheduler
    elif args.sample_method == "DPMSolverSinglestep":
        scheduler_cls = DPMSolverSinglestepScheduler
    elif args.sample_method == "PNDM":
        scheduler_cls = PNDMScheduler
        kwargs.pop("rescale_betas_zero_snr", None)
    elif args.sample_method == "HeunDiscrete":  ########
        scheduler_cls = HeunDiscreteScheduler
    elif args.sample_method == "EulerAncestralDiscrete":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif args.sample_method == "DEISMultistep":
        scheduler_cls = DEISMultistepScheduler
        kwargs.pop("rescale_betas_zero_snr", None)
    elif args.sample_method == "KDPM2AncestralDiscrete":  #########
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    elif args.sample_method == "CogVideoX":
        scheduler_cls = CogVideoXDDIMScheduler
    elif args.sample_method == "FlowMatchEulerDiscrete":
        scheduler_cls = FlowMatchEulerDiscreteScheduler
        kwargs = {}
    else:
        raise NameError(f"Unsupport sample_method {args.sample_method}")
    scheduler = scheduler_cls(**kwargs)
    return scheduler


def _optional_hf_value(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"auto", "default", "hf_default", "none", "null"}:
        return None
    return value


def _looks_like_hf_repo_id(value):
    text = str(value or "").strip()
    if not text or text.startswith(("http://", "https://", "hf://")):
        return False
    if text.startswith(("~", ".")) or Path(text).expanduser().is_absolute():
        return False
    return text.count("/") == 1 and ":" not in text and "$" not in text


def _resolve_subfolder_for_local_ckpt(model_path, subfolder, cache_dir):
    subfolder = _optional_hf_value(subfolder)
    if subfolder is None:
        return model_path, None
    base_path = Path(str(model_path)).expanduser()
    if base_path.exists():
        candidate = base_path / str(subfolder)
        if candidate.exists():
            return str(candidate), None
        return model_path, subfolder
    if _looks_like_hf_repo_id(model_path):
        snapshot_path = Path(
            snapshot_download(
                str(model_path),
                cache_dir=cache_dir,
                allow_patterns=[f"{subfolder}/*"],
            )
        )
        candidate = snapshot_path / str(subfolder)
        if candidate.exists():
            return str(candidate), None
    return model_path, subfolder


def prepare_pipeline(args, dtype, device):
    weight_dtype = dtype
    cache_dir = _optional_hf_value(args.cache_dir)
    model_subdir = _optional_hf_value(getattr(args, "model_subdir", None))
    ae_subdir = _optional_hf_value(getattr(args, "ae_subdir", None))
    ae_path, ae_subdir = _resolve_subfolder_for_local_ckpt(args.ae_path, ae_subdir, cache_dir)

    vae = ae_wrapper[args.ae](ae_path, subfolder=ae_subdir, cache_dir=cache_dir)
    vae.vae = vae.vae.to(device=device, dtype=weight_dtype).eval()
    vae.vae_scale_factor = ae_stride_config[args.ae]
    if args.enable_tiling:
        vae.vae.enable_tiling()

    if "mt5" in args.text_encoder_name_1:
        text_encoder_1 = MT5EncoderModel.from_pretrained(
            args.text_encoder_name_1, cache_dir=cache_dir, torch_dtype=weight_dtype
        ).eval()
    else:
        text_encoder_1 = T5EncoderModel.from_pretrained(
            args.text_encoder_name_1, cache_dir=cache_dir, torch_dtype=weight_dtype
        ).eval()
    tokenizer_1 = AutoTokenizer.from_pretrained(args.text_encoder_name_1, cache_dir=cache_dir)

    if args.text_encoder_name_2 is not None:
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            args.text_encoder_name_2, cache_dir=cache_dir, torch_dtype=weight_dtype
        ).eval()
        tokenizer_2 = AutoTokenizer.from_pretrained(args.text_encoder_name_2, cache_dir=cache_dir)
    else:
        text_encoder_2, tokenizer_2 = None, None

    if args.version == "v1_3":
        if args.model_type == "inpaint" or args.model_type == "i2v":
            transformer_model = OpenSoraInpaint_v1_3.from_pretrained(
                args.model_path, subfolder=model_subdir, cache_dir=cache_dir, device_map=None, torch_dtype=weight_dtype
            ).eval()
        else:
            transformer_model = OpenSoraT2V_v1_3.from_pretrained(
                args.model_path, subfolder=model_subdir, cache_dir=cache_dir, device_map=None, torch_dtype=weight_dtype
            ).eval()
    elif args.version == "v1_5":
        if args.model_type == "inpaint" or args.model_type == "i2v":
            raise NotImplementedError("Inpainting model is not available in v1_5")
        else:
            raise NotImplementedError(
                "Open-Sora-Plan v1.5 is not available in this in-tree build; the shipped route is v1.3"
            )

    scheduler = get_scheduler(args)
    pipeline_class = (
        OpenSoraInpaintPipeline if args.model_type == "inpaint" or args.model_type == "i2v" else OpenSoraPipeline
    )

    pipeline = pipeline_class(
        vae=vae,
        text_encoder=text_encoder_1,
        tokenizer=tokenizer_1,
        scheduler=scheduler,
        transformer=transformer_model,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
    ).to(device)

    if args.save_memory:
        print("enable_model_cpu_offload AND enable_sequential_cpu_offload AND enable_tiling")
        pipeline.enable_model_cpu_offload()
        pipeline.enable_sequential_cpu_offload()
        # torch.cuda.empty_cache()
        vae.vae.enable_tiling()
        vae.vae.t_chunk_enc = 8
        vae.vae.t_chunk_dec = vae.vae.t_chunk_enc // 2

    if args.compile:
        pipeline.transformer = torch.compile(pipeline.transformer)

    return pipeline


def init_gpu_env(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    args.local_rank = local_rank
    args.world_size = world_size
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)
    if args.sp:
        initialize_sequence_parallel_state(world_size)
    return args


def init_npu_env(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    args.local_rank = local_rank
    args.world_size = world_size
    torch_npu.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", init_method="env://", world_size=world_size, rank=local_rank)
    if args.sp:
        initialize_sequence_parallel_state(world_size)
    return args


def save_video_grid(video, nrow=None):
    b, t, h, w, c = video.shape

    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = torch.zeros((t, (padding + h) * nrow + padding, (padding + w) * ncol + padding, c), dtype=torch.uint8)

    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r : start_r + h, start_c : start_c + w] = video[i]

    return video_grid


def _empty_sample_grid(args):
    if args.num_frames == 1:
        return torch.empty((0, 3, args.height, args.width), dtype=torch.uint8)
    return torch.empty((0, args.num_frames, args.height, args.width, 3), dtype=torch.uint8)


def _cat_or_empty_sample_grid(video_grids, args):
    if video_grids:
        return torch.cat(video_grids, dim=0)
    return _empty_sample_grid(args)


def _gather_distributed_sample_grids(video_grids, args):
    local_grid = _cat_or_empty_sample_grid(video_grids, args).cuda()
    local_count = torch.tensor([local_grid.shape[0]], dtype=torch.long, device=local_grid.device)
    gathered_counts = [torch.zeros_like(local_count) for _ in range(args.world_size)]
    dist.all_gather(gathered_counts, local_count)
    counts = [int(count.item()) for count in gathered_counts]
    max_count = max(counts) if counts else 0
    if max_count == 0:
        return _empty_sample_grid(args)

    if local_grid.shape[0] < max_count:
        pad_shape = (max_count - local_grid.shape[0], *local_grid.shape[1:])
        local_grid = torch.cat(
            [local_grid, torch.zeros(pad_shape, dtype=local_grid.dtype, device=local_grid.device)],
            dim=0,
        )
    gathered_shape = (args.world_size, max_count, *local_grid.shape[1:])
    gathered = torch.empty(gathered_shape, dtype=local_grid.dtype, device=local_grid.device)
    dist.all_gather_into_tensor(gathered, local_grid.contiguous())
    pieces = [gathered[rank, :count] for rank, count in enumerate(counts) if count > 0]
    if not pieces:
        return _empty_sample_grid(args)
    return torch.cat(pieces, dim=0).cpu()


def run_model_and_save_samples(args, pipeline, caption_refiner_model=None, enhance_video_model=None):
    if args.seed is not None:
        set_seed(args.seed, rank=args.local_rank, device_specific=True)
    if args.local_rank >= 0:
        torch.manual_seed(args.seed + args.local_rank)
    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path, exist_ok=True)

    video_grids = []
    if not isinstance(args.text_prompt, list):
        args.text_prompt = [args.text_prompt]
    if len(args.text_prompt) == 1 and args.text_prompt[0].endswith("txt"):
        text_prompt = open(args.text_prompt[0]).readlines()  # noqa: SIM115 - Keeps the upstream file-handle lifetime.
        args.text_prompt = [i.strip() for i in text_prompt]

    if args.model_type == "inpaint" or args.model_type == "i2v":
        if not isinstance(args.conditional_pixel_values_path, list):
            args.conditional_pixel_values_path = [args.conditional_pixel_values_path]
        if len(args.conditional_pixel_values_path) == 1 and args.conditional_pixel_values_path[0].endswith("txt"):
            temp = open(args.conditional_pixel_values_path[0]).readlines()  # noqa: SIM115 - Keeps the upstream file-handle lifetime.
            conditional_pixel_values_path = [i.strip().split(",") for i in temp]

        mask_type = args.mask_type if args.mask_type is not None else None

    positive_prompt = """
    high quality, high aesthetic, {}
    """

    negative_prompt = """
    nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
    low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
    """

    def generate(prompt, conditional_pixel_values_path=None, mask_type=None):
        if args.caption_refiner is not None:
            if args.model_type != "inpaint" and args.model_type != "i2v":
                refine_prompt = caption_refiner_model.get_refiner_output(prompt)
                print(f"\nOrigin prompt: {prompt}\n->\nRefine prompt: {refine_prompt}")
                prompt = refine_prompt
            else:
                # Due to the current use of LLM as the caption refiner, additional content that is not present in the control image will be added. Therefore, caption refiner is not used in this mode.
                print("Caption refiner is not available for inpainting model, use the original prompt...")
                time.sleep(3)
        input_prompt = positive_prompt.format(prompt)

        if args.model_type == "inpaint" or args.model_type == "i2v":
            print(f"\nConditional pixel values path: {conditional_pixel_values_path}")
            videos = pipeline(
                conditional_pixel_values_path=conditional_pixel_values_path,
                mask_type=mask_type,
                crop_for_hw=args.crop_for_hw,
                max_hxw=args.max_hxw,
                noise_strength=args.noise_strength,
                prompt=input_prompt,
                negative_prompt=negative_prompt,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_sampling_steps,
                guidance_scale=args.guidance_scale,
                num_samples_per_prompt=args.num_samples_per_prompt,
                max_sequence_length=args.max_sequence_length,
            ).videos
        else:
            videos = pipeline(
                input_prompt,
                negative_prompt=negative_prompt,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_sampling_steps,
                guidance_scale=args.guidance_scale,
                num_samples_per_prompt=args.num_samples_per_prompt,
                max_sequence_length=args.max_sequence_length,
            ).videos
        if enhance_video_model is not None:
            # b t h w c
            videos = enhance_video_model.enhance_a_video(videos, input_prompt, 2.0, args.fps, 250)
        if (not args.sp) or (args.sp and args.local_rank <= 0):
            if args.num_frames == 1:
                videos = rearrange(videos, "b t h w c -> (b t) c h w")
                if args.num_samples_per_prompt != 1:
                    for i, image in enumerate(videos):
                        save_image(
                            image / 255.0,
                            os.path.join(
                                args.save_img_path,
                                f"{args.sample_method}_{index}_gs{args.guidance_scale}_s{args.num_sampling_steps}_i{i}.jpg",
                            ),
                            nrow=math.ceil(math.sqrt(videos.shape[0])),
                            normalize=True,
                            value_range=(0, 1),
                        )  # b c h w
                save_image(
                    videos / 255.0,
                    os.path.join(
                        args.save_img_path,
                        f"{args.sample_method}_{index}_gs{args.guidance_scale}_s{args.num_sampling_steps}.jpg",
                    ),
                    nrow=math.ceil(math.sqrt(videos.shape[0])),
                    normalize=True,
                    value_range=(0, 1),
                )  # b c h w
            else:
                if args.num_samples_per_prompt == 1:
                    imageio.mimwrite(
                        os.path.join(
                            args.save_img_path,
                            f"{args.sample_method}_{index}_gs{args.guidance_scale}_s{args.num_sampling_steps}.mp4",
                        ),
                        videos[0],
                        fps=args.fps,
                        quality=6,
                    )  # highest quality is 10, lowest is 0
                else:
                    for i in range(args.num_samples_per_prompt):
                        imageio.mimwrite(
                            os.path.join(
                                args.save_img_path,
                                f"{args.sample_method}_{index}_gs{args.guidance_scale}_s{args.num_sampling_steps}_i{i}.mp4",
                            ),
                            videos[i],
                            fps=args.fps,
                            quality=6,
                        )  # highest quality is 10, lowest is 0

                    videos = save_video_grid(videos)
                    imageio.mimwrite(
                        os.path.join(
                            args.save_img_path,
                            f"{args.sample_method}_{index}_gs{args.guidance_scale}_s{args.num_sampling_steps}.mp4",
                        ),
                        videos,
                        fps=args.fps,
                        quality=6,
                    )  # highest quality is 10, lowest is 0)
                    videos = videos.unsqueeze(0)  # 1 t h w c
            video_grids.append(videos)

    if args.model_type == "inpaint" or args.model_type == "i2v":
        for index, (prompt, cond_path) in enumerate(zip(args.text_prompt, conditional_pixel_values_path, strict=False)):
            if not args.sp and args.local_rank != -1 and index % args.world_size != args.local_rank:
                continue
            generate(prompt, cond_path, mask_type)
    else:
        for index, prompt in enumerate(args.text_prompt):
            if not args.sp and args.local_rank != -1 and index % args.world_size != args.local_rank:
                continue  # skip when ddp
            generate(prompt)

    if (args.model_type == "inpaint" or args.model_type == "i2v") and not args.crop_for_hw:
        print("completed, please check the saved images and videos")
    else:
        if not args.sp:
            if args.local_rank != -1:
                dist.barrier()
                video_grids = _gather_distributed_sample_grids(video_grids, args)
                dist.barrier()
            else:
                video_grids = _cat_or_empty_sample_grid(video_grids, args)
        elif args.sp and args.local_rank <= 0:
            video_grids = torch.cat(video_grids)

        if args.local_rank <= 0:
            if args.num_frames == 1:
                save_image(
                    video_grids / 255.0,
                    os.path.join(
                        args.save_img_path,
                        f"{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.jpg",
                    ),
                    nrow=math.ceil(math.sqrt(len(video_grids))),
                    normalize=True,
                    value_range=(0, 1),
                )
            else:
                video_grids = save_video_grid(video_grids)
                imageio.mimwrite(
                    os.path.join(
                        args.save_img_path,
                        f"{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.mp4",
                    ),
                    video_grids,
                    fps=args.fps,
                    quality=6,
                )
            print(f"save path {args.save_img_path}")


def run_model_and_save_samples_npu(args, pipeline, caption_refiner_model=None, enhance_video_model=None):
    # experimental_config = torch_npu.profiler._ExperimentalConfig(
    #     profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    #     aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization
    # )
    # profile_output_path = "cache/worldfoundry/open_sora_plan/npu_profiling_t2v"
    # os.makedirs(profile_output_path, exist_ok=True)
    # with torch_npu.profiler.profile(
    #         activities=[
    #             torch_npu.profiler.ProfilerActivity.NPU,
    #             torch_npu.profiler.ProfilerActivity.CPU
    #             ],
    #         with_stack=True,
    #         record_shapes=True,
    #         profile_memory=True,
    #         experimental_config=experimental_config,
    #         schedule=torch_npu.profiler.schedule(
    #             wait=10000, warmup=0, active=1, repeat=1, skip_first=0
    #             ),
    #         on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f"{profile_output_path}/")
    # ) as prof:
    run_model_and_save_samples(args, pipeline, caption_refiner_model, enhance_video_model)
    # prof.step()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="LanguageBind/Open-Sora-Plan-v1.0.0")
    parser.add_argument("--version", type=str, default="v1_3", choices=["v1_3", "v1_5"])
    parser.add_argument("--model_type", type=str, default="t2v", choices=["t2v", "inpaint", "i2v"])
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--model_subdir", type=str, default=None)
    parser.add_argument("--caption_refiner", type=str, default=None)
    parser.add_argument("--ae", type=str, default="CausalVAEModel_4x8x8")
    parser.add_argument("--ae_path", type=str, default="CausalVAEModel_4x8x8")
    parser.add_argument("--ae_subdir", type=str, default=None)
    parser.add_argument("--enhance_video", type=str, default=None)
    parser.add_argument("--text_encoder_name_1", type=str, default="DeepFloyd/t5-v1_1-xxl")
    parser.add_argument("--text_encoder_name_2", type=str, default=None)
    parser.add_argument("--save_img_path", type=str, default="./sample_videos/t2v")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_method", type=str, default="PNDM")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--text_prompt", nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples_per_prompt", type=int, default=1)
    parser.add_argument("--enable_tiling", action="store_true")
    parser.add_argument("--refine_caption", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save_memory", action="store_true")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument("--rescale_betas_zero_snr", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--sp", action="store_true")

    parser.add_argument("--v1_5_scheduler", action="store_true")
    parser.add_argument("--conditional_pixel_values_path", type=str, default=None)
    parser.add_argument("--mask_type", type=str, default=None)
    parser.add_argument("--crop_for_hw", action="store_true")
    parser.add_argument("--max_hxw", type=int, default=236544)  # 480*480
    parser.add_argument("--noise_strength", type=float, default=0.0)
    args = parser.parse_args()
    assert not (args.sp and args.num_frames == 1)
    return args
