"""Isolated entry point for BAAI's pinned Emu3.5 interleaved inference.

The official implementation is imported from a separate checkout. Its model,
generation, image decoding, and protobuf writing code remain upstream code.
"""

from __future__ import annotations

import argparse
import importlib
import random
import shutil
import sys
from pathlib import Path


def patch_legacy_cache_api(cache_type: type) -> None:
    """Expose old Cache length accessors on newer Transformers cache types."""
    if not hasattr(cache_type, "seen_tokens"):
        cache_type.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(cache_type, "get_max_length"):

        def get_max_length(self):
            size = self.get_max_cache_shape()
            return None if size < 0 else size

        cache_type.get_max_length = get_max_length
    if not hasattr(cache_type, "get_usable_length"):

        def get_usable_length(self, new_seq_length, layer_idx=0):
            previous = self.get_seq_length(layer_idx)
            maximum = self.get_max_length()
            if maximum is not None and previous + new_seq_length > maximum:
                return maximum - new_seq_length
            return previous

        cache_type.get_usable_length = get_usable_length


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--vision-tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image-path", type=Path)
    parser.add_argument("--task-type", choices=("story", "howto"), default="story")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--target-height", type=int)
    parser.add_argument("--target-width", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=6666)
    return parser.parse_args()


def main() -> None:
    args = _args()
    repo = args.repo_root.resolve()
    sys.path.insert(0, str(repo))

    import torch
    from inference import inference
    from src.emu3p5 import Emu3Config, Emu3ForCausalLM
    from transformers import AutoModel, AutoTokenizer, GenerationMixin
    from transformers.cache_utils import Cache

    # Transformers >=4.50 no longer gives PreTrainedModel.generate() to
    # custom causal LMs. Keep upstream source untouched and supply the mixin
    # through a loader-only subclass.
    class Emu3ForGeneration(Emu3ForCausalLM, GenerationMixin):
        pass

    # The official model still calls the pre-4.50 Cache accessors. Their
    # current equivalents carry the same lengths, except that -1 means an
    # unbounded dynamic cache instead of None.
    patch_legacy_cache_api(Cache)

    config_module = {
        "story": "configs.example_config_visual_narrative",
        "howto": "configs.example_config_visual_guidance",
    }[args.task_type]
    cfg = importlib.import_module(config_module)
    cfg.model_path = str(args.checkpoint_path.resolve())
    cfg.tokenizer_path = str((repo / "src/tokenizer_emu3_ibq").resolve())
    cfg.vq_path = str(args.vision_tokenizer_path.resolve())
    cfg.save_path = str(args.output_path.resolve().parent / f".{args.output_path.stem}-official")
    cfg.prompts = [
        (
            args.output_path.stem,
            {
                "prompt": args.prompt,
                "reference_image": str(args.image_path.resolve()),
            }
            if args.image_path
            else args.prompt,
        )
    ]
    cfg.use_image = args.image_path is not None
    cfg.unc_prompt, cfg.template = cfg.build_unc_and_template(args.task_type, cfg.use_image)
    cfg.max_new_tokens = args.max_new_tokens
    cfg.sampling_params = dict(cfg.sampling_params, max_new_tokens=args.max_new_tokens)
    cfg.target_height = args.target_height
    cfg.target_width = args.target_width
    cfg.seed = args.seed
    cfg.hf_device = args.device
    cfg.vq_device = args.device

    if not torch.cuda.is_available():
        raise RuntimeError("Emu3.5 official inference requires a CUDA GPU")
    device_index = torch.device(args.device).index
    if device_index is None or device_index >= torch.cuda.device_count():
        raise RuntimeError(f"Emu3.5 CUDA device is unavailable: {args.device}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    print("[INFO] Loading official Emu3.5 with eager attention on the selected GPU", flush=True)
    model_config = Emu3Config.from_pretrained(cfg.model_path, local_files_only=True)
    model = Emu3ForGeneration.from_pretrained(
        cfg.model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_path,
        special_tokens_file=str(Path(cfg.tokenizer_path) / "emu3_vision_tokens.txt"),
        trust_remote_code=True,
        local_files_only=True,
    )
    for attr, token in {
        "bos_token": "<|extra_203|>",
        "eos_token": "<|extra_204|>",
        "pad_token": "<|endoftext|>",
        "eol_token": "<|extra_200|>",
        "eof_token": "<|extra_201|>",
        "tms_token": "<|extra_202|>",
        "img_token": "<|image token|>",
        "boi_token": "<|image start|>",
        "eoi_token": "<|image end|>",
        "bss_token": "<|extra_100|>",
        "ess_token": "<|extra_101|>",
        "bog_token": "<|extra_60|>",
        "eog_token": "<|extra_61|>",
        "boc_token": "<|extra_50|>",
        "eoc_token": "<|extra_51|>",
    }.items():
        setattr(tokenizer, attr, token)
    print("[INFO] Loading official vision tokenizer checkpoint", flush=True)
    vq_model = (
        AutoModel.from_pretrained(cfg.vq_path, trust_remote_code=True, local_files_only=True).to(cfg.vq_device).eval()
    )
    cfg.special_token_ids = {name: tokenizer.encode(token)[0] for name, token in cfg.special_tokens.items()}
    inference(cfg, model, tokenizer, vq_model)
    generated = Path(cfg.save_path) / "proto" / f"{args.output_path.stem}.pb"
    if not generated.is_file() or generated.stat().st_size == 0:
        raise RuntimeError(f"official inference did not produce a protobuf artifact: {generated}")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(generated), str(args.output_path))
    print(f"[INFO] Saved {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
