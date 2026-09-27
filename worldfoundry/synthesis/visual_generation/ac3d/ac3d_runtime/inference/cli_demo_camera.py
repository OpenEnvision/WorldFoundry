import sys
from pathlib import Path

# Modified by WorldFoundry: upstream appended the CWD-relative '..' (only valid when the
# process CWD is inference/). Anchor to the ac3d_runtime root instead so the bare imports
# below (cogvideo_controlnet, controlnet_pipeline, inference.*) resolve regardless of CWD.
_AC3D_RUNTIME_ROOT = str(Path(__file__).resolve().parents[1])
if _AC3D_RUNTIME_ROOT not in sys.path:
    sys.path.append(_AC3D_RUNTIME_ROOT)
import argparse
import os

import torch
from cogvideo_controlnet import CogVideoXControlnet
from cogvideo_transformer import CustomCogVideoXTransformer3DModel
from controlnet_pipeline import ControlnetCogVideoXPipeline
from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.utils import export_to_video
from torchvision.transforms.functional import to_pil_image
from transformers import T5EncoderModel, T5Tokenizer

from inference.datasets.controlnet_datasets_camera import RealEstate10KPoseControlnetDataset
from inference.utils import stack_images_horizontally
from worldfoundry.core.checkpoint import load_tensor_state_dict


@torch.no_grad()
def generate_video(
    prompt: str,
    video_root_dir: str,
    annotation_json: str,
    base_model_path: str,
    controlnet_model_path: str,
    controlnet_weights: float = 1.0,
    controlnet_guidance_start: float = 0.0,
    controlnet_guidance_end: float = 1.0,
    use_dynamic_cfg: bool = True,
    lora_path: str = None,
    lora_rank: int = 128,
    output_path: str = "./output/",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    num_frames: int = 49,
    height: int = 480,
    width: int = 720,
    stride_min: int = 1,
    stride_max: int = 1,
    start_camera_idx: int = 0,
    end_camera_idx: int = 1,
    controlnet_transformer_num_attn_heads: int = None,
    controlnet_transformer_attention_head_dim: int = None,
    controlnet_transformer_out_proj_dim_factor: int = None,
    controlnet_transformer_out_proj_dim_zero_init: bool = False,
    controlnet_transformer_num_layers: int = 8,
    downscale_coef: int = 8,
    controlnet_input_channels: int = 6,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - video_root_dir (str): The path to the camera dataset
    - annotation_json (str): Name of subset (train.json or test.json)
    - base_model_path (str): The path of the pre-trained model to be used.
    - controlnet_model_path (str): The path of the pre-trained conrolnet model to be used.
    - controlnet_weights (float): Strenght of controlnet
    - controlnet_guidance_start (float): The stage when the controlnet starts to be applied
    - controlnet_guidance_end (float): The stage when the controlnet end to be applied
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - seed (int): The seed for reproducibility.
    """
    os.makedirs(output_path, exist_ok=True)
    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    tokenizer = T5Tokenizer.from_pretrained(
        base_model_path, subfolder="tokenizer"
    )
    text_encoder = T5EncoderModel.from_pretrained(
        base_model_path, subfolder="text_encoder"
    )
    transformer = CustomCogVideoXTransformer3DModel.from_pretrained(
        base_model_path, subfolder="transformer"
    )
    vae = AutoencoderKLCogVideoX.from_pretrained(
        base_model_path, subfolder="vae"
    )
    scheduler = CogVideoXDDIMScheduler.from_pretrained(
        base_model_path, subfolder="scheduler"
    )
    # ControlNet
    num_attention_heads_orig = 48 if "5b" in base_model_path.lower() else 30
    controlnet_kwargs = {}
    if controlnet_transformer_num_attn_heads is not None:
        controlnet_kwargs["num_attention_heads"] = args.controlnet_transformer_num_attn_heads
    else:
        controlnet_kwargs["num_attention_heads"] = num_attention_heads_orig
    if controlnet_transformer_attention_head_dim is not None:
        controlnet_kwargs["attention_head_dim"] = controlnet_transformer_attention_head_dim
    if controlnet_transformer_out_proj_dim_factor is not None:
        controlnet_kwargs["out_proj_dim"] = num_attention_heads_orig * controlnet_transformer_out_proj_dim_factor
    controlnet_kwargs["out_proj_dim_zero_init"] = controlnet_transformer_out_proj_dim_zero_init
    controlnet = CogVideoXControlnet(
        num_layers=controlnet_transformer_num_layers,
        downscale_coef=downscale_coef,
        in_channels=controlnet_input_channels,
        **controlnet_kwargs,   
    )
    if controlnet_model_path:
        controlnet_state_dict = load_tensor_state_dict(controlnet_model_path)
        m, u = controlnet.load_state_dict(controlnet_state_dict, strict=False)
        print(f'[ Weights from pretrained controlnet was loaded into controlnet ] [M: {len(m)} | U: {len(u)}]')
    
    # Full pipeline
    pipe = ControlnetCogVideoXPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        controlnet=controlnet,
        scheduler=scheduler,
    )
    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(lora_scale=1 / lora_rank)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")

    # pipe.to("cuda")
    pipe = pipe.to(dtype=dtype)
    pipe.enable_sequential_cpu_offload()

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    # 4. Load dataset
    eval_dataset = RealEstate10KPoseControlnetDataset(
        video_root_dir=video_root_dir,
        annotation_json=annotation_json,
        image_size=(height, width), 
        stride=(stride_min, stride_max),
        sample_n_frames=num_frames,
    )
    for camera_idx in range(start_camera_idx, end_camera_idx):
        
        # Get data
        data_dict = eval_dataset[camera_idx]
        reference_video = data_dict['video']
        reference_frames = [to_pil_image(frame) for frame in ((reference_video)/2+0.5)]
        controlnet_latents = data_dict['controlnet_video']
        controlnet_latents = controlnet_latents[None]
        
        # Set output directory
        output_path_file = os.path.join(output_path, f"{camera_idx:05d}_out.mp4")
        output_path_file_reference = output_path_file.replace("_out.mp4", "_reference.mp4")
        output_path_file_out_reference = output_path_file.replace(".mp4", "_reference.mp4")
        
        if os.path.isfile(output_path_file):
            continue
        
        # 5. Generate the video frames based on the prompt.
        # `num_frames` is the Number of frames to generate.
        # This is the default value for 6 seconds video and 8 fps and will plus 1 frame for the first frame and 49 frames.
        video_generate_all = pipe(
            prompt=prompt,
            controlnet_latents=controlnet_latents,  # The path of the image to be used as the background of the video
            height=height,
            width=width,
            num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=num_frames,  # Number of frames to generate，changed to 49 for diffusers version `0.30.3` and after.
            use_dynamic_cfg=use_dynamic_cfg,  # This id used for DPM Sechduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
            controlnet_weights=controlnet_weights,
            controlnet_guidance_start=controlnet_guidance_start,
            controlnet_guidance_end=controlnet_guidance_end,
        ).frames
        video_generate = video_generate_all[0]

        # 6. Export the generated frames to a video file. fps must be 8 for original video.
        export_to_video(video_generate, output_path_file, fps=8)
        export_to_video(reference_frames, output_path_file_reference, fps=8)
        out_reference_frames = [
            stack_images_horizontally(frame_reference, frame_out)
            for frame_out, frame_reference in zip(video_generate, reference_frames)
            ]
        export_to_video(out_reference_frames, output_path_file_out_reference, fps=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument(
        "--video_root_dir",
        type=str,
        required=True,
        help="The path of the video for controlnet processing.",
    )
    parser.add_argument(
        "--base_model_path", type=str, default="THUDM/CogVideoX-5b", help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--controlnet_model_path", type=str, default="TheDenk/cogvideox-5b-controlnet-hed-v1", help="The path of the controlnet pre-trained model to be used"
    )
    parser.add_argument("--annotation_json", type=str, default='test.json', help="Subset of video_root_dir dataset")
    parser.add_argument("--controlnet_weights", type=float, default=0.5, help="Strenght of controlnet")
    parser.add_argument("--controlnet_guidance_start", type=float, default=0.0, help="The stage when the controlnet starts to be applied")
    parser.add_argument("--controlnet_guidance_end", type=float, default=0.5, help="The stage when the controlnet end to be applied")
    parser.add_argument("--use_dynamic_cfg", type=bool, default=True, help="Use dynamic cfg")
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument(
        "--output_path", type=str, default="./output", help="The path where the generated video will be saved"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--stride_min", type=int, default=1)
    parser.add_argument("--stride_max", type=int, default=1)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--start_camera_idx", type=int, default=0)
    parser.add_argument("--end_camera_idx", type=int, default=1)
    parser.add_argument("--controlnet_transformer_num_attn_heads", type=int, default=None)
    parser.add_argument("--controlnet_transformer_attention_head_dim", type=int, default=None)
    parser.add_argument("--controlnet_transformer_out_proj_dim_factor", type=int, default=None)
    parser.add_argument("--controlnet_transformer_out_proj_dim_zero_init", action="store_true", default=False, help=("Init project zero."),
    )
    parser.add_argument("--downscale_coef", type=int, default=8)
    parser.add_argument("--controlnet_input_channels", type=int, default=6)

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        prompt=args.prompt,
        video_root_dir=args.video_root_dir,
        base_model_path=args.base_model_path,
        controlnet_model_path=args.controlnet_model_path,
        annotation_json=args.annotation_json,
        controlnet_weights=args.controlnet_weights,
        controlnet_guidance_start=args.controlnet_guidance_start,
        controlnet_guidance_end=args.controlnet_guidance_end,
        use_dynamic_cfg=args.use_dynamic_cfg,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        seed=args.seed,
        stride_min=args.stride_min,
        stride_max=args.stride_max,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        start_camera_idx=args.start_camera_idx,
        end_camera_idx=args.end_camera_idx,
        controlnet_transformer_num_attn_heads=args.controlnet_transformer_num_attn_heads,
        controlnet_transformer_attention_head_dim=args.controlnet_transformer_attention_head_dim,
        controlnet_transformer_out_proj_dim_factor=args.controlnet_transformer_out_proj_dim_factor,
        downscale_coef=args.downscale_coef,
        controlnet_input_channels=args.controlnet_input_channels,
    )
