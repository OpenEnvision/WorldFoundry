#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare native components with an explicitly supplied, pinned upstream source.

Run in the unified GPU environment. This is component evidence, not an accepted
full-checkpoint public-pipeline replay. The optional VAE comparison uses real
weights; the small DiT graph uses identical deterministic synthetic weights.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
import torch.version

ROOT = Path(__file__).resolve().parents[2]
REVISION = "5446525c80da2a4a2f87801addbbc7faa8d2486e"
sys.path.insert(0, str(ROOT))


def compare_graph() -> dict[str, Any]:
    """Compare BF16 SDPA outputs and every KV tensor through window eviction."""
    reference_attention: Any = import_module("matrix_game.inference.wan.modules.attention")
    ReferenceModel = import_module("matrix_game.inference.wan.modules.causal_model").CausalWanModel

    from worldfoundry.base_models.diffusion_model.models.denoisers.matrix_game4 import MatrixGame4Denoiser
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.matrix_game4 import MatrixGame4Model

    reference_attention.FLASH_ATTN_2_AVAILABLE = False
    reference_attention.FLASH_ATTN_3_AVAILABLE = False
    config: dict[str, Any] = dict(
        in_dim=148,
        out_dim=48,
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=2,
        text_dim=32,
        text_len=8,
        freq_dim=12,
        local_attn_size=21,
        sink_size=1,
    )
    torch.manual_seed(123)
    reference = ReferenceModel(model_type="ti2v", **config).eval()
    torch.nn.init.normal_(reference.head.head.weight, std=0.05)
    reference.to(device="cuda", dtype=torch.bfloat16)
    reference.set_rope_max_len(128)
    native = MatrixGame4Model(**config, max_rope_frames=128).eval().to(device="cuda", dtype=torch.bfloat16)
    native.load_state_dict(reference.state_dict(), strict=True)
    adapter = MatrixGame4Denoiser(native)
    cache_options: dict[str, Any] = dict(
        batch_size=1,
        n_views=1,
        latent_frames_per_view=30,
        frame_sequence_length=4,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    expected_cache, actual_cache = adapter.create_kv_cache(**cache_options), adapter.create_kv_cache(**cache_options)
    text = torch.randn(1, 8, 32, device="cuda").bfloat16()
    with torch.no_grad():
        for position in range(27):
            for step in (1000.0, 833.3333129882812, 0.0):
                pixels = torch.randn(1, 148, 1, 4, 4, device="cuda").bfloat16()
                if step == 0 or position == 0:
                    pixels[:, 48:] = 0
                common = dict(
                    t=torch.tensor([[step]], device="cuda"), context=text, seq_len=4, current_start=position * 4
                )
                expected = reference(
                    pixels,
                    **common,
                    kv_cache=expected_cache["kv_cache"],
                    crossattn_cache=expected_cache["crossattn_cache"],
                    sink_rope_mode="fixed",
                )
                actual = native(
                    pixels,
                    **common,
                    kv_cache=actual_cache["kv_cache"],
                    crossattn_cache=actual_cache["crossattn_cache"],
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for ours, theirs in zip(actual_cache["kv_cache"], expected_cache["kv_cache"], strict=True):
                    for key in ("k", "v", "k_sink_raw", "global_end_index", "local_end_index"):
                        torch.testing.assert_close(ours[key], theirs[key], rtol=0, atol=0)
    return {"weights": "synthetic", "positions": 27, "forwards": 81, "max_error": 0.0}


def compare_vae(path: Path) -> dict[str, Any]:
    """Compare full-resolution, two-latent Wan2.2 encoding and decoding."""
    WanVAE22Wrapper = import_module("matrix_game.inference.wrappers").WanVAE22Wrapper

    from worldfoundry.base_models.diffusion_model.components import ComponentBuildContext, ComponentKey, ComponentKind
    from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest
    from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
    from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.component import build_wan_video_vae38_decoder
    from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy
    from worldfoundry.base_models.diffusion_model.recipes.matrix_game import matrix_game4_egypt_recipe

    reference = WanVAE22Wrapper(vae_path=str(path)).to(device="cuda", dtype=torch.bfloat16).eval()
    codec_spec = next(spec for spec in matrix_game4_egypt_recipe().components if spec.key.kind is ComponentKind.DECODER)
    native = build_wan_video_vae38_decoder(
        ComponentBuildContext(
            model_id="matrix-game-4-egypt",
            key=ComponentKey(ComponentKind.DECODER),
            policy=RuntimePolicy(device="cuda", dtype=torch.bfloat16),
            checkpoints={"weights": CheckpointSpec(source=path)},
            component_options=codec_spec.options,
        )
    )
    rng = torch.Generator("cpu").manual_seed(49)
    pixels = (torch.rand(1, 3, 5, 480, 864, generator=rng) * 2 - 1).to(device="cuda", dtype=torch.bfloat16)
    request = DiffusionRequest(prompt="", height=480, width=864, num_frames=5)
    with torch.no_grad():
        expected = reference.encode_to_latent(pixels.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        actual = native.encode(pixels)
        torch.testing.assert_close(actual, expected.to(actual), rtol=0, atol=0)
        expected_rgb = reference.decode_to_pixel(expected.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        actual_rgb = native.decode(expected, request)
        torch.testing.assert_close(actual_rgb, expected_rgb.to(actual_rgb), rtol=0, atol=0)
    return {
        "weights": str(path),
        "rgb_shape": [1, 3, 5, 480, 864],
        "latent_shape": [1, 48, 2, 30, 54],
        "encode_max_error": 0.0,
        "decode_max_error": 0.0,
    }


def compare_text(path: Path, tokenizer_root: Path) -> dict[str, Any]:
    """Compare released-prompt token IDs, padding and real UMT5 embeddings."""
    ReferenceTextEncoder = import_module("matrix_game.inference.wrappers").WanTextEncoder

    from worldfoundry.base_models.diffusion_model.components import ComponentBuildContext, ComponentKey, ComponentKind
    from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
    from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
    from worldfoundry.base_models.diffusion_model.models.encoders.wan.component import build_wan_text_conditioner
    from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy
    from worldfoundry.base_models.diffusion_model.recipes.matrix_game import MATRIX_GAME4_EGYPT_PROMPT

    reference = (
        ReferenceTextEncoder(
            t5_checkpoint_path=str(path),
            tokenizer_path=str(tokenizer_root / "google/umt5-xxl"),
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    native = build_wan_text_conditioner(
        ComponentBuildContext(
            model_id="matrix-game-4-egypt",
            key=ComponentKey(ComponentKind.CONDITIONER),
            policy=RuntimePolicy(device="cuda", dtype=torch.bfloat16),
            checkpoints={"weights": CheckpointSpec(source=path), "tokenizer": CheckpointSpec(source=tokenizer_root)},
        )
    )
    prompts = [MATRIX_GAME4_EGYPT_PROMPT]
    for actual, expected in zip(
        native.tokenizer(prompts, return_mask=True, add_special_tokens=True),
        reference.tokenizer(prompts, return_mask=True, add_special_tokens=True),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    request = DiffusionRequest(prompt=prompts[0], sampling=SamplingConfig(guidance_scale=1.0))
    with torch.no_grad():
        expected = reference(prompts)["prompt_embeds"]
        actual = native.encode(request, device=torch.device("cuda"), dtype=torch.bfloat16).positive["context"]
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return {"weights": str(path), "embedding_shape": list(actual.shape), "max_error": 0.0}


def main() -> None:
    """Run component comparisons without changing the model runtime's imports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True, help="Pinned upstream Matrix-Game-4 directory")
    parser.add_argument("--vae", type=Path, help="Optional real Wan2.2_VAE.pth checkpoint")
    parser.add_argument("--text-encoder", type=Path, help="Optional real UMT5 encoder checkpoint")
    parser.add_argument("--tokenizer", type=Path, help="Root containing google/umt5-xxl; required with --text-encoder")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/matrix-game4-component-parity.json")
    args = parser.parse_args()
    if (args.text_encoder is None) != (args.tokenizer is None):
        parser.error("--text-encoder and --tokenizer must be supplied together")
    revision = subprocess.check_output(
        ["git", "-C", str(args.reference_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Reference must be pinned at {REVISION}, got {revision}")
    if not torch.cuda.is_available():
        raise RuntimeError("Matrix-Game 4 BF16 component parity requires CUDA")
    sys.path.insert(0, str(args.reference_root.resolve()))
    report: dict[str, Any] = {
        "reference_revision": revision,
        "evidence_level": "components",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "dtype": "bfloat16",
        "attention": "torch-sdpa",
        "small_graph": compare_graph(),
    }
    if args.vae is not None:
        report["vae"] = compare_vae(args.vae)
    if args.text_encoder is not None:
        report["text"] = compare_text(args.text_encoder, args.tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
