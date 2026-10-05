import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, SiglipImageProcessor
from worldfoundry.core.attention import resolve_transformers_attention_implementation
from worldfoundry.core.execution.device import resolve_inference_device, resolve_inference_dtype
from worldfoundry.core.vram import skip_model_initialization
from .modeling.model import VLANeXt, LlamaProcessorWrapper
from .modeling.rt2_baseline import RT2LikeBaseline

def _get_checkpoint_path(cfg) -> str:
    if hasattr(cfg, "eval") and hasattr(cfg.eval, "finetuned_checkpoint"):
        return cfg.eval.finetuned_checkpoint
    raise ValueError("cfg.eval.finetuned_checkpoint is required")


def _load_checkpoint(checkpoint_path: str):
    kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        return torch.load(checkpoint_path, mmap=True, **kwargs)
    except (RuntimeError, ValueError):
        # Legacy non-zip tensor checkpoints do not support mmap.  Keep the
        # tensor-only unpickler in the fallback; inference never needs the
        # optimizer's Python objects or arbitrary checkpoint globals.
        return torch.load(checkpoint_path, **kwargs)


def get_vla(cfg, *, device=None, torch_dtype="auto"):
    checkpoint_path = _get_checkpoint_path(cfg)
    print(f"Loading model from {checkpoint_path}")

    checkpoint = _load_checkpoint(checkpoint_path)
    checkpoint_config = checkpoint["config"]
    model_config = checkpoint_config["model"]
    data_config = checkpoint_config["data"]

    model_type = model_config.get("model_type", "vlanext")
    lmm_path = getattr(getattr(cfg, "model", None), "lmm_path", model_config["lmm_path"])
    print(f"Model type: {model_type}")

    requested_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(resolve_inference_device(requested_device, allow_cpu_fallback=True))
    dtype = resolve_inference_dtype(device, torch_dtype, strict=False)
    preferred_attention = getattr(getattr(cfg, "model", None), "attention_backend", None)
    attn_implementation = resolve_transformers_attention_implementation(
        preferred=preferred_attention,
        device=device,
    )

    num_train_timesteps = model_config.get(
        "num_train_timesteps",
        model_config.get("diffusion_steps", 1000),
    )
    num_inference_timesteps = model_config.get(
        "num_inference_timesteps",
        model_config.get("diffusion_steps", 10),
    )
    if hasattr(cfg, "model") and hasattr(cfg.model, "diffusion_steps"):
        num_inference_timesteps = cfg.model.diffusion_steps
        print(f"Overriding inference diffusion steps to: {num_inference_timesteps}")

    scheduler_type = model_config["scheduler_type"]
    if hasattr(cfg, "model"):
        scheduler_type = getattr(cfg.model, "scheduler_type", scheduler_type)

    with skip_model_initialization():
        if model_type == "rt2_baseline":
            model = RT2LikeBaseline(
                lmm_path=lmm_path,
                vision_encoder_path=model_config.get("vision_encoder_path", "google/siglip2-base-patch16-256"),
                action_dim=model_config["action_dim"],
                num_actions=data_config["future_len"],
                num_history=data_config["history_len"],
                use_proprio_input_vlm=model_config.get("use_proprio_input_vlm", True),
                use_transformer_projector=model_config.get("use_transformer_proprio_projector", True),
                projector_depth=model_config["projector_depth"],
                projector_num_heads=model_config["projector_num_heads"],
                num_bins=model_config.get("num_bins", 256),
                attn_implementation=attn_implementation,
                torch_dtype=dtype,
                load_backbone_weights=False,
            )
        else:
            model = VLANeXt(
                lmm_path=lmm_path,
                vision_encoder_path=model_config.get("vision_encoder_path", "google/siglip2-base-patch16-256"),
                action_dim=model_config["action_dim"],
                num_actions=data_config["future_len"],
                num_queries=model_config["num_queries"],
                num_history=data_config["history_len"],
                loss_type=model_config.get("loss_type", "diffusion"),
                future_image_loss_weight=float(model_config.get("future_image_loss_weight", 1.0)),
                num_train_timesteps=num_train_timesteps,
                num_inference_timesteps=num_inference_timesteps,
                scheduler_type=scheduler_type,
                condition_type=model_config.get("condition_type", "loose"),
                policy_hidden_size=model_config.get("policy_hidden_size", 1024),
                policy_depth=model_config.get("policy_depth", 29),
                policy_num_heads=model_config.get("policy_num_heads", 12),
                policy_mlp_ratio=model_config.get("policy_mlp_ratio", 4.0),
                use_proprio_input_vlm=model_config.get("use_proprio_input_vlm", True),
                use_action_input_policy=model_config.get("use_action_input_policy", False),
                use_transformer_proprio_projector=model_config.get("use_transformer_proprio_projector", False),
                projector_depth=model_config["projector_depth"],
                projector_num_heads=model_config["projector_num_heads"],
                use_transformer_connector=model_config["use_transformer_connector"],
                connector_depth=model_config["connector_depth"],
                connector_num_heads=model_config["connector_num_heads"],
                num_bins=model_config.get("num_bins", 256),
                action_vqvae=model_config.get("action_vqvae"),
                generator_hidden_size=model_config.get("generator_hidden_size", 768),
                generator_depth=model_config.get("generator_depth", 12),
                generator_num_heads=model_config.get("generator_num_heads", 12),
                generator_mlp_ratio=model_config.get("generator_mlp_ratio", 4.0),
                attn_implementation=attn_implementation,
                torch_dtype=dtype,
                load_backbone_weights=False,
            )

    state_dict = checkpoint["model_state_dict"]
    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict, strict=True, assign=True)
    print(f"Loaded state dict strictly: {len(state_dict)} keys")
    model.to(device=device, dtype=dtype)
    model.eval()
    model.checkpoint_config = checkpoint_config

    del state_dict, checkpoint
    return model

def get_processor(cfg, checkpoint_config=None):
    config = checkpoint_config
    if config is None:
        checkpoint = _load_checkpoint(_get_checkpoint_path(cfg))
        config = checkpoint["config"]
    lmm_path = getattr(getattr(cfg, "model", None), "lmm_path", config["model"]["lmm_path"])

    if "llama" in lmm_path.lower():
        vision_encoder_path = config["model"].get("vision_encoder_path", "google/siglip2-base-patch16-256")
        tokenizer = AutoTokenizer.from_pretrained(lmm_path, local_files_only=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        image_processor = SiglipImageProcessor.from_pretrained(
            vision_encoder_path,
            local_files_only=True,
        )
        return LlamaProcessorWrapper(tokenizer, image_processor)

    return AutoProcessor.from_pretrained(
        lmm_path,
        trust_remote_code=False,
        local_files_only=True,
    )

def get_vla_action(cfg, model, processor, obs, task_label, *, seed=None):
    data_cfg = model.checkpoint_config["data"]
    input_modality = data_cfg.get("input_modality", "image")
    view_mode = data_cfg.get("view_mode", "single")
    fps = float(data_cfg.get("fps", 20.0))
    history_len = getattr(model, "num_history", 0)
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    effective_processor = getattr(model, "processor", processor)

    is_paligemma = "PaliGemma" in effective_processor.__class__.__name__
    is_qwen = "Qwen" in effective_processor.__class__.__name__
    is_llama = "Llama" in effective_processor.__class__.__name__

    def _take_last(history_list, fallback, n: int):
        if not history_list:
            history_list = [fallback]
        if n > 0:
            xs = history_list[-n:]
            if len(xs) < n:
                xs = ([xs[0]] * (n - len(xs))) + xs
            return xs
        return [fallback]

    all_images = obs.get("image_history", [obs["full_image"]])
    images_np = _take_last(all_images, obs["full_image"], history_len)

    if view_mode == "multi":
        all_wrist = obs.get("image_history_wrist", [obs["full_image_wrist"]])
        wrist_np = _take_last(all_wrist, obs["full_image_wrist"], history_len)

    pil_images = [Image.fromarray(img) for img in images_np]
    if view_mode == "multi":
        pil_wrist = [Image.fromarray(img) for img in wrist_np]

    proprioception = None
    if model.use_proprio_input_vlm:
        all_states = obs.get("state_history", [])
        if history_len > 0:
            states = all_states[-history_len:]
            if len(states) < history_len:
                states = ([states[0]] * (history_len - len(states))) + states if states else [np.zeros(model.action_dim)] * history_len
        else:
            states = []
        if len(states) > 0:
            proprioception = torch.tensor(np.stack(states), dtype=model_dtype).unsqueeze(0).to(device)

    history_actions = None
    if getattr(model, 'use_action_input_policy', False):
        all_actions = obs.get("action_history", [])
        if history_len > 0:
            actions = all_actions[-history_len:]
            if len(actions) < history_len:
                actions = ([np.zeros(model.action_dim)] * (history_len - len(actions))) + actions
        else:
            actions = []
        if len(actions) > 0:
            history_actions = torch.tensor(np.stack(actions), dtype=model_dtype).unsqueeze(0).to(device)

    inputs = {}

    if input_modality == "video":
        if is_paligemma or is_llama:
            raise ValueError(
                f"{effective_processor.__class__.__name__} only supports image input in VLANeXt."
            )
        content = []
        if view_mode == "multi":
            content.extend([{"type": "video", "video": pil_images}, {"type": "video", "video": pil_wrist}])
            videos = [pil_images, pil_wrist]
        else:
            content.append({"type": "video", "video": pil_images})
            videos = [pil_images]

        content.append({"type": "text", "text": task_label})
        messages = [{"role": "user", "content": content}]
        text = effective_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = effective_processor(
            text=[text],
            videos=videos,
            videos_kwargs={"fps": fps, "return_metadata": True},
            padding=True,
            return_tensors="pt",
        )

    elif input_modality == "image":
        if view_mode == "multi":
            images = [pil_images[-1], pil_wrist[-1]]
        else:
            images = [pil_images[-1]]

        if is_llama:
            text = [task_label]
            inputs = effective_processor.tokenizer(text, padding=True, return_tensors="pt")
            image_inputs = effective_processor.image_processor(images, return_tensors="pt")
            inputs["pixel_values"] = image_inputs["pixel_values"]

        elif is_paligemma:
            text = "<image>" * len(images) + task_label
            inputs = effective_processor(
                text=[text],
                images=images,
                padding=True,
                return_tensors="pt",
            )
        elif is_qwen:
            content = []
            for img in images:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": task_label})

            messages = [{"role": "user", "content": content}]
            text = effective_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = effective_processor(
                text=[text],
                images=images,
                padding=True,
                return_tensors="pt",
            )
    else:
        raise ValueError(f"Unknown input_modality: {input_modality} for model type")

    valid_keys = {
        "input_ids", "attention_mask", "pixel_values", "pixel_values_videos",
        "image_grid_thw", "video_grid_thw", "token_type_ids", "mm_token_type_ids",
    }
    inputs = {k: v.to(device) for k, v in inputs.items() if k in valid_keys}

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)
    if "pixel_values_videos" in inputs:
        inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(model_dtype)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
    with torch.no_grad():
        action_pred = model.predict_action(
            proprioception=proprioception,
            history_actions=history_actions,
            generator=generator,
            **inputs,
        )

    return action_pred[0].float().cpu().numpy()
