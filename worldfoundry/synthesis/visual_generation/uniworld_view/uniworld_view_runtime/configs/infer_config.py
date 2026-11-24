import argparse


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {v!r}")


def get_parser():
    parser = argparse.ArgumentParser()

    ## general
    parser.add_argument("--image_dir", type=str, default="./test/images/fruit.png", help="Image file path")
    parser.add_argument("--out_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device to use")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name, use image file name by default")

    ## renderer
    parser.add_argument(
        "--mode",
        type=str,
        default="single_view",
        help="Currently we support 'single_view_txt' and 'single_view_target'",
    )
    # parser.add_argument('--traj_txt',  type=str, help="Required for 'single_view_txt' mode, a txt file that specify camera trajectory")
    parser.add_argument("--elevation", default=0.0, type=float, help="Initial angle, no exceptions to change")
    parser.add_argument("--d_r", default=1.0, type=float, help="Camera distance, default is 1.0, range 0.25 to 2.5")
    parser.add_argument(
        "--d_theta",
        default=0.0,
        type=float,
        help="Vertical rotation, <0 up, >0 down, range -90 to 30; generally not recommended to angle too much downwards",
    )
    parser.add_argument(
        "--d_phi",
        default=0.0,
        type=float,
        help="Horizontal rotation, <0 right, >0 left, supports 360 degrees; range -360 to 360",
    )
    parser.add_argument(
        "--x_offset",
        default=0.0,
        type=float,
        help="Horizontal translation, <0 left, >0 right, range -0.5 to 0.5; depends on depth, excessive movement may cause artifacts",
    )
    parser.add_argument(
        "--y_offset",
        default=0.0,
        type=float,
        help="Vertical translation, <0 up, >0 down, range -0.5 to 0.5; depends on depth, excessive movement may cause artifacts",
    )
    parser.add_argument(
        "--z_offset",
        default=0.0,
        type=float,
        help="Forward and backward translation, <0 back, >0 forward, range -0.5 to 0.5 is ok; depends on depth, excessive movement may cause artifacts",
    )
    parser.add_argument(
        "--traj_type",
        default="custom",
        type=str,
        choices=["custom", "free1", "free2", "free3", "free4", "free5", "swing1", "swing2", "orbit"],
        help="custom refers to a custom trajectory, while the others are pre-defined camera trajectories (see traj_map for details)",
    )
    parser.add_argument(
        "--focal_length",
        default=1.0,
        type=float,
        help="Focal length, range 0.25 to 2.5; changing focal length zooms in and out",
    )

    ## depthcrafter (optional, for depth estimation)
    parser.add_argument(
        "--unet_path",
        type=str,
        default="./checkpoints/DepthCrafter",
        help="Path to the UNet model",
    )

    parser.add_argument(
        "--pre_train_path",
        type=str,
        default="stabilityai/stable-video-diffusion-img2vid",
        help="Path to the pre-trained model (can be HuggingFace model ID or local path)",
    )
    parser.add_argument("--cpu_offload", type=str, default="model", help="CPU offload strategy")
    parser.add_argument("--depth_inference_steps", type=int, default=5, help="Number of inference steps")
    parser.add_argument(
        "--depth_guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale for inference",
    )
    parser.add_argument("--window_size", type=int, default=110, help="Window size for processing")
    parser.add_argument("--overlap", type=int, default=25, help="Overlap size for processing")
    parser.add_argument("--max_res", type=int, default=1024, help="Maximum resolution for processing")
    parser.add_argument("--fps", type=int, default=16, help="Fps for saved video")

    # smoothing control (single parameter)
    parser.add_argument(
        "--smooth_sigma",
        type=float,
        default=3.0,
        help="Temporal smoothing sigma applied to pose (T, R, FoV); 0 disables smoothing",
    )

    ## warp
    parser.add_argument("--stride", type=int, default=1, help="Sampling stride for input video")
    parser.add_argument(
        "--radius_scale",
        type=float,
        default=1.0,
        help="Scale factor for the spherical radius",
    )
    parser.add_argument("--camera", type=str, default="target", help="traj or target")
    # parser.add_argument(
    #     '--dv', type=str, default='gradual', help='gradual, bullet or direct'
    # )
    parser.add_argument("--mask", default=False, help="Clean the pcd if true")
    # parser.add_argument(
    #     '--traj_txt',
    #     type=str,
    #     help="Required for 'traj' camera, a txt file that specify camera trajectory",
    # )
    parser.add_argument(
        "--target_pose",
        nargs=5,
        type=float,
        help="Required for 'target' mode, specify target camera pose, <theta phi r x y>",
    )
    parser.add_argument("--near", type=float, default=0.0001, help="Near clipping plane distance")
    parser.add_argument("--far", type=float, default=10000.0, help="Far clipping plane distance")
    parser.add_argument("--anchor_idx", type=int, default=0, help="One GT frame")

    ## diffusion
    parser.add_argument(
        "--low_gpu_memory_mode", type=_str2bool, nargs="?", const=True, default=False, help="Enable low GPU memory mode"
    )
    parser.add_argument(
        "--model_name", type=str, default="checkpoints/Wan2.1-VACE-14B-diffusers", help="Path to the model"
    )
    parser.add_argument(
        "--sampler_name",
        type=str,
        choices=["Flow", "Euler", "Euler A", "DPM++", "PNDM", "DDIM_Cog", "DDIM_Origin"],
        default="Flow",
        help="Choose the sampler",
    )
    parser.add_argument(
        "--transformer_path", type=str, default="checkpoints/UniView", help="Path to the pretrained transformer model"
    )
    parser.add_argument("--transformer_path_2", type=str, default=None, help="Path to the pretrained transformer model")

    parser.add_argument(
        "--lora_path",
        type=str,
        default="./checkpoints/loras/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors",
        help="Path to the LoRA weights (required for video generation)",
    )
    parser.add_argument("--diffusion_guidance_scale", type=float, default=6.0, help="Guidance scale for inference")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for video generation")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        help="Negative prompt for video generation",
    )
    parser.add_argument(
        "--refine_prompt",
        type=str,
        default=". The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.",
        help="Prompt for video generation",
    )
    parser.add_argument("--seed", type=int, default=43, help="seed for seed_everything")
    parser.add_argument(
        "--video_length", type=int, default=81, help="inference video length, change to 16 if you use 16 frame model"
    )
    parser.add_argument("--blip_path", type=str, default="./checkpoints/blip2-opt-2.7b")
    parser.add_argument("--height", type=int, default=480, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=832, help="image width, in pixel space")
    parser.add_argument(
        "--keep_aspect_ratio",
        action="store_true",
        default=True,
        help=(
            "Keep input aspect ratio by resizing to a variable (H,W) with ~constant diffusion token budget "
            "(no pad / no crop). Enabled by default."
        ),
    )
    parser.add_argument(
        "--no_keep_aspect_ratio",
        dest="keep_aspect_ratio",
        action="store_false",
        help="Disable keep_aspect_ratio and use the fixed (H,W) preprocessing.",
    )

    ## dust3r
    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        help="The path of the model",
    )
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--niter", default=300)
    parser.add_argument("--lr", default=0.01)
    parser.add_argument("--min_conf_thr", default=3.0)  # minimum=1.0, maximum=20
    parser.add_argument(
        "--dpt_trd",
        type=float,
        default=1.0,
        help="Required for mulitpule reference images and iterative mode, limit the max depth by * dpt_trd",
    )

    ## moge
    parser.add_argument(
        "--moge_path", default="./checkpoints/moge/model.pt", help="Path to MoGe depth estimation model"
    )
    parser.add_argument("--segnet_path", default="./checkpoints/tracer_b7.pth")
    parser.add_argument(
        "--stream3r_path",
        default="./checkpoints/STream3R",
        help="Path to STream3R checkpoint folder (config.json + model.safetensors)",
    )
    parser.add_argument(
        "--sam2_checkpoint", default="./checkpoints/sam2/sam2_hiera_large.pt", help="Path to SAM2 checkpoint (.pt)"
    )
    parser.add_argument("--sam2_config", default="configs/sam2/sam2_hiera_l.yaml", help="Path to SAM2 config (.yaml)")

    ## geometry backends
    parser.add_argument(
        "--geometry_backend",
        choices=("stream3r", "mosca"),
        default="stream3r",
        help=(
            "Geometry estimator used for dynamic-view depth + camera pose. "
            "'stream3r' runs STream3R online (default); 'mosca' reads a "
            "pre-computed MoSca bundle (run mosca_precompute.py + "
            "lite_moca_reconstruct.py first)."
        ),
    )
    parser.add_argument(
        "--mosca_ws",
        type=str,
        default=None,
        help=(
            "Workspace directory produced by mosca_precompute.py + "
            "lite_moca_reconstruct.py. Expected to contain images/, "
            "<dep_mode>_depth/, and bundle/ (with bundle.pth + "
            "bundle_cams.pth). Defaults to <save_dir> when --geometry_backend=mosca."
        ),
    )
    parser.add_argument(
        "--mosca_dep_mode",
        type=str,
        default=None,
        help=(
            "Depth subdirectory name inside --mosca_ws (e.g. 'depthcrafter_depth', "
            "'metric3d_depth', 'uni_depth'). Auto-detected if omitted."
        ),
    )

    ## toggles (bools)
    parser.add_argument(
        "--advanced_render",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable advanced rendering pipeline (SAM2 segmentation + optional VDA alignment + occlusion-aware warping)",
    )
    parser.add_argument(
        "--warp_with_occlusion",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Calculate occlusion mask during warping",
    )
    parser.add_argument(
        "--render_method",
        type=str,
        choices=["warp", "hybrid", "mesh"],
        default="hybrid",
        help="Rendering method: 'warp' (BiSplat renderer), 'hybrid' (mesh warping + pointcloud depth check), or 'mesh' (mesh-only warping)",
    )

    # Convenience negated flags (easier to use than "--flag false")
    parser.add_argument(
        "--no_advanced_render", dest="advanced_render", action="store_false", help="Disable advanced rendering pipeline"
    )
    parser.add_argument(
        "--no_align_with_vda", dest="align_with_vda", action="store_false", help="Disable VDA depth alignment"
    )
    parser.add_argument(
        "--no_warp_with_occlusion",
        dest="warp_with_occlusion",
        action="store_false",
        help="Disable occlusion-aware warping",
    )

    return parser
