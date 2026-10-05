import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.video_x_fun import (
    AutoencoderKLWan,
)
from worldfoundry.base_models.diffusion_model.models.encoders.wan.variants.dreamx_world.text_encoder import (
    WanT5EncoderModel,
)
from worldfoundry.core.distributed.sequence_parallel.runtime import (
    set_multi_gpus_devices,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.versecrafter import (
    VerseCrafterWanTransformer3DModel,
)
from worldfoundry.base_models.diffusion_model.loaders.wan_variant import (
    load_wan_transformer,
)
from worldfoundry.core.distributed.sharding.block_fsdp import shard_model
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent, get_image_latent,
                                    get_video_to_video_latent,
                                    save_videos_grid)
from worldfoundry.base_models.diffusion_model.schedulers.flow_dpm import FlowDPMSolverMultistepScheduler
from worldfoundry.base_models.diffusion_model.schedulers.flow_unipc import FlowUniPCMultistepScheduler
from worldfoundry.core.io.paths import resolve_data_path
from versecrafter.pipeline import WanVerseCrafterPipeline

# Parse command line arguments
parser = argparse.ArgumentParser(description='Video generation inference script')
parser.add_argument('--transformer_path', type=str, default="model/VerseCrafter",
                    help='Path to transformer checkpoint or directory')
parser.add_argument('--save_path', type=str, default="dataset/inference",
                    help='Path to save generated videos')
parser.add_argument('--rendering_maps_path', type=str, required=True,
                    help='Path to directory with rendering maps')
parser.add_argument('--prompt', type=str, required=True,
                    help='Text prompt for video generation')
parser.add_argument('--input_image_path', type=str, required=True,
                    help='Path to input image')
parser.add_argument('--num_inference_steps', type=int, default=50,
                    help='Number of inference steps')
parser.add_argument('--sample_size', type=str, default="720,1280",
                    help='Sample size as "height,width" (e.g., "720,1280")')
parser.add_argument('--ulysses_degree', type=int, default=2,
                    help='Ulysses degree for multi-GPU configuration')
parser.add_argument('--ring_degree', type=int, default=2,
                    help='Ring degree for multi-GPU configuration')
parser.add_argument('--guidance_scale', type=float, default=5.0,
                    help='Classifier-free guidance scale')
parser.add_argument('--seed', type=int, default=2025,
                    help='Random seed for generation')
parser.add_argument('--fps', type=int, default=16,
                    help='Frames per second for output video')
parser.add_argument('--base_model_path', type=str, required=True,
                    help='Local Wan2.1-T2V-14B base-model directory')
parser.add_argument('--config_path', type=str, default=None,
                    help='Local VerseCrafter YAML config path')
parser.add_argument('--video_length', type=int, default=81,
                    help='Output frame count')
parser.add_argument(
    '--gpu_memory_mode',
    choices=(
        'model_full_load',
        'model_full_load_and_qfloat8',
        'model_cpu_offload',
        'model_cpu_offload_and_qfloat8',
        'sequential_cpu_offload',
    ),
    default='model_full_load',
    help='Component placement policy used during diffusion inference',
)
args = parser.parse_args()

# Parse sample_size from string to list
sample_size = [int(x) for x in args.sample_size.split(',')]

# GPU memory mode, which can be chosen in:
# [model_full_load, model_full_load_and_qfloat8, model_cpu_offload,
# model_cpu_offload_and_qfloat8, sequential_cpu_offload].
# model_full_load means that the entire model will be moved to the GPU.
#
# model_full_load_and_qfloat8 means that the entire model will be moved to the GPU,
# and the transformer model has been quantized to float8, which can save more GPU memory.
#
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
#
# model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use,
# and the transformer model has been quantized to float8, which can save more GPU memory.
#
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use,
# resulting in slower speeds but saving a large amount of GPU memory.
GPU_memory_mode     = args.gpu_memory_mode
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used.
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
ulysses_degree      = args.ulysses_degree
ring_degree         = args.ring_degree
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = True
fsdp_text_encoder   = True
# Compile will give a speedup in fixed resolution and need a little GPU memory.
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit         = False

# Support TeaCache.
enable_teacache     = True
# Recommended to be set between 0.05 and 0.30. A larger threshold can cache more steps,
# speeding up the inference process, but it may cause slight differences between the
# generated content and the original content.
# # --------------------------------------------------------------------------------------------------- #
# | Model Name          | threshold | Model Name          | threshold | Model Name          | threshold |
# | Wan2.1-T2V-1.3B     | 0.05~0.10 | Wan2.1-T2V-14B      | 0.10~0.15 | Wan2.1-I2V-14B-720P | 0.20~0.30 |
# | Wan2.1-I2V-14B-480P | 0.20~0.25 | Wan2.1-Fun-*-1.3B-* | 0.05~0.10 | Wan2.1-Fun-*-14B-*  | 0.20~0.30 |
# # --------------------------------------------------------------------------------------------------- #
teacache_threshold  = 0.10
# The number of steps to skip TeaCache at the beginning of the inference process, which can
# reduce the impact of TeaCache on generated video quality.
num_skip_start_steps = 5
# Whether to offload TeaCache tensors to cpu to save a little bit of GPU memory.
teacache_offload    = True

# Skip some cfg steps in inference for acceleration
# Recommended to be set between 0.00 and 0.25
cfg_skip_ratio      = 0

# Riflex config
enable_riflex       = False
# Index of intrinsic frequency
riflex_k            = 6

# Config and model path
config_path         = args.config_path or str(
    resolve_data_path(
        "models", "runtime", "configs", "versecrafter", "wan2.1", "wan_civitai.yaml"
    )
)
# model path
model_name          = args.base_model_path

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow_Unipc"
# [NOTE]: Noise schedule shift parameter. Affects temporal dynamics.
# Used when the sampler is in "Flow_Unipc", "Flow_DPM++".
shift               = 16

# Load pretrained model if need
transformer_path    = args.transformer_path
vae_path            = None
lora_path           = None

# Other params
# sample_size is already set from command line arguments above
video_length        = args.video_length
fps                 = args.fps
geoada_context_scale = 1.00
geoada_in_dim        = 128

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype            = torch.bfloat16

# Alternatively, set control_video_path directly for single control video (backward compatibility)
control_video_path      = None

start_image             = None
end_image               = None
subject_ref_images      = None

# Adding words such as "quiet, solid" to the neg prompt can increase dynamism.
prompt              = args.prompt
negative_prompt = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

guidance_scale          = args.guidance_scale
seed                    = args.seed
num_inference_steps     = args.num_inference_steps
lora_weight             = 0.55
save_path               = args.save_path

device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)

transformer_additional_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs'])
if geoada_in_dim is not None:
    transformer_additional_kwargs['geoada_in_dim'] = geoada_in_dim

# Load transformer from checkpoint or pretrained model
if transformer_path is not None and os.path.isdir(transformer_path):
    # If transformer_path is a directory, load directly from it
    print(f"Loading transformer from checkpoint directory: {transformer_path}")
    transformer = load_wan_transformer(
        VerseCrafterWanTransformer3DModel,
        transformer_path,
        additional_kwargs=transformer_additional_kwargs,
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
else:
    # Load from base model first
    transformer = load_wan_transformer(
        VerseCrafterWanTransformer3DModel,
        os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        additional_kwargs=transformer_additional_kwargs,
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    # Then load weights from checkpoint file if provided
    if transformer_path is not None:
        print(f"Loading transformer weights from checkpoint: {transformer_path}")
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(transformer_path, map_location="cpu", weights_only=True)
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Vae
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu", weights_only=True)
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)

# Get Text encoder
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder = text_encoder.eval()

# Get Scheduler
Chosen_Scheduler = scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
    config['scheduler_kwargs']['shift'] = 1
scheduler = Chosen_Scheduler(
    **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

# Get Pipeline
pipeline = WanVerseCrafterPipeline(
    transformer=transformer,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
)
if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
    print("Add Compile")

if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(transformer, ["modulation",], device=device)
    transformer.freqs = transformer.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

# coefficients = get_teacache_coefficients("VerseCrafter") if enable_teacache else None
coefficients = [8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02]

if coefficients is not None:
    effective_skip_start_steps = min(num_skip_start_steps, num_inference_steps)
    print(
        f"Enable TeaCache with threshold {teacache_threshold} and skip the first "
        f"{effective_skip_start_steps} steps."
    )
    pipeline.transformer.enable_teacache(
        coefficients, num_inference_steps, teacache_threshold,
        num_skip_start_steps=effective_skip_start_steps, offload=teacache_offload
    )

if cfg_skip_ratio is not None:
    print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

with torch.no_grad():
    video_length = (
        int((video_length - 1) // vae.config.temporal_compression_ratio *
            vae.config.temporal_compression_ratio) + 1
        if video_length != 1 else 1
    )
    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

    if enable_riflex:
        pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)

    if subject_ref_images is not None:
        subject_ref_images = [
            get_image_latent(_subject_ref_image, sample_size=sample_size, padding=True)
            for _subject_ref_image in subject_ref_images
        ]
        subject_ref_images = torch.cat(subject_ref_images, dim=2)

    inpaint_video = None
    if start_image is not None and end_image is not None:
        inpaint_video, inpaint_video_mask, clip_image = get_image_to_video_latent(
            start_image,
            end_image,
            video_length=video_length,
            sample_size=sample_size
        )

    # Load control videos and mask based on configuration
    if control_video_path is None:
        # Multi-control case: load from directory structure
        rendering_maps_path = args.rendering_maps_path
        input_image_path = args.input_image_path

        if os.path.isdir(rendering_maps_path):
            # Multi-control case: load multiple control videos
            control_filenames = [
                "background_RGB.mp4",
                "background_depth.mp4",
                "3D_gaussian_RGB.mp4",
                "3D_gaussian_depth.mp4"
            ]
            control_videos = []

            for control_filename in control_filenames:
                control_video_path_full = os.path.join(rendering_maps_path, control_filename)
                if os.path.exists(control_video_path_full):
                    input_video, _, _, _ = get_video_to_video_latent(
                        control_video_path_full,
                        video_length=video_length,
                        sample_size=sample_size,
                        fps=fps,
                        ref_image=None
                    )
                    control_videos.append(input_video)
                else:
                    print(f"Warning: Control video not found: {control_video_path_full}")
                    # Create a zero tensor as placeholder
                    if len(control_videos) > 0:
                        control_videos.append(torch.zeros_like(control_videos[0]))

            # Load mask
            mask_path = os.path.join(rendering_maps_path, "merged_mask.mp4")
            if os.path.exists(mask_path):
                input_video_mask, _, _, _ = get_video_to_video_latent(
                    mask_path,
                    video_length=video_length,
                    sample_size=sample_size,
                    fps=fps,
                    ref_image=None
                )
                input_video_mask = input_video_mask[:, :1]
                input_video_mask[:,:,0] = 0.0
            else:
                input_video_mask = torch.ones_like(control_videos[0][:, :1]) * 255

            # Load input image
            img_latent = get_image_latent(input_image_path, sample_size=sample_size)

            # Set first frame of first control video to input image
            control_videos[0][:,:,0] = img_latent.squeeze(2)

            control_video = control_videos
        else:
            raise ValueError(f"Annotation path not found: {rendering_maps_path}")
    else:
        # Single control case (backward compatibility)
        control_video, _, _, _ = get_video_to_video_latent(
            control_video_path,
            video_length=video_length,
            sample_size=sample_size,
            fps=fps,
            ref_image=None
        )
        input_video_mask = inpaint_video_mask

    sample = pipeline(
        prompt,
        num_frames = video_length,
        negative_prompt = negative_prompt,
        height      = sample_size[0],
        width       = sample_size[1],
        generator   = generator,
        guidance_scale = guidance_scale,
        num_inference_steps = num_inference_steps,

        video               = inpaint_video,
        mask_video          = input_video_mask,
        control_video       = control_video,
        subject_ref_images  = subject_ref_images,
        shift               = shift,
        geoada_context_scale= geoada_context_scale,
    ).videos

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path) if path.startswith("generated_video_")])
    prefix = str(index)
    if video_length == 1:
        video_path = os.path.join(save_path, "generated_video_" + prefix + ".png")

        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(video_path)
    else:
        video_path = os.path.join(save_path, "generated_video_" + prefix + ".mp4")
        save_videos_grid(sample, video_path, fps=fps)
        print(prompt)
        print(video_path)

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        save_results()
else:
    save_results()
