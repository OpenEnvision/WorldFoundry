# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
"""Assembly of the independently adapted high- and low-noise fast branches."""

from __future__ import annotations

import copy
import gc
import json
from pathlib import Path

import torch
from diffusers.models import AutoencoderKLWan
from transformers import AutoTokenizer, UMT5EncoderModel

from worldfoundry.base_models.diffusion_model.runners.worldcrafter import WorldCrafterPipeline
from worldfoundry.base_models.diffusion_model.schedulers.worldcrafter import WorldCrafterScheduler
from worldfoundry.base_models.diffusion_model.models.networks.worldcrafter.transformer import WorldCrafterTransformer3DModel
from worldfoundry.base_models.diffusion_model.models.networks.worldcrafter.fast.compact_ucpe import compact_ucpe
from worldfoundry.base_models.diffusion_model.models.networks.worldcrafter.fast.attention import FastUcpeSelfAttention
from worldfoundry.base_models.diffusion_model.models.networks.worldcrafter.fast.contract import load_dmd_inference_contract
from worldfoundry.base_models.diffusion_model.models.networks.worldcrafter.fast.resident import ResidentBranches
from worldfoundry.base_models.diffusion_model.models.networks.worldcrafter.kernels import (
    replace_rmsnorm_with_fp32,
    replace_all_norms_with_flash_norms,
    replace_rope_with_flash_rope,
)
from worldfoundry.base_models.diffusion_model.models.networks.worldcrafter.repencoder import (
    RepEncoder,
    RepEncoderInferenceMemoryProvider,
    RepEncoderInferenceProviderConfig,
)
from worldfoundry.base_models.diffusion_model.models.networks.worldcrafter.ucpe.bridge import (
    enable_ucpe_inference_sdpa_attention,
    patch_worldcrafter_transformer_ucpe,
    load_ucpe_camera_adapter_weights,
)


def load_fast(
    cls,
    model_path,
    *,
    device,
    height,
    width,
    seed,
    memory_fov_h_deg,
    memory_fov_v_deg,
    memory_fov_samples_per_axis,
    attention_backend,
    enable_compile,
    shared_model_path=None,
):
    from .runtime import configure_attention, load_model_adapter, sha256

    if (height, width) != (384, 640):
        raise ValueError("Fast weights require height=384 and width=640")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Fast inference requires CUDA")
    torch.cuda.set_device(device)
    root = Path(model_path).expanduser().resolve()
    config = json.loads((root / "inference_config.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    expected_config = dict(
        steps_per_stage=[2, 2, 2],
        guidance_scale=1.0,
        ucpe_pixel_center=True,
        repencoder_target_microbatch=1,
        representation="resident_byte_compact_ucpe",
        compile=False,
    )
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError(
            "Fast inference configuration differs from the validated release contract"
        )
    if config["routing"] != [["equal", "equal"], ["equal", "equal"], ["equal", "old"]]:
        raise ValueError("Fast I2V requires 5+1 routing")
    shared = (Path(shared_model_path).expanduser().resolve()
              if shared_model_path is not None
              else (root / config["shared_components"]).resolve())
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Incomplete fast checkpoint: {path}")
        if path.stat().st_size < 1024 * 1024 and sha256(path) != row["sha256"]:
            raise ValueError(f"Fast checkpoint metadata mismatch: {path}")
    adapters = [root / "adapter_high_noise", root / "adapter_low_noise"]
    contracts = [
        load_dmd_inference_contract(p, expected_latent_shape=(16, 9, 48, 80))
        for p in adapters
    ]
    contract = contracts[0]
    if contract.fingerprint != contracts[
        1
    ].fingerprint or contract.rollout_steps_per_stage != (2, 2, 2):
        raise ValueError(
            "Fast branches must have identical native 2/2/2 timestep contracts"
        )
    for adapter in adapters:
        frozen = json.loads((adapter / "repencoder_frozen.json").read_text())
        if (
            frozen["repencoder"]["model_sha256"]
            != manifest["repencoder"]["reference_file_sha256"]
        ):
            raise ValueError("Fast adapter references an unexpected RepEncoder")
    enable_ucpe_inference_sdpa_attention()
    repencoder = RepEncoder.from_pretrained(
        shared / "repencoder", device=device, compute_dtype="bf16", target_microbatch=1
    )
    if (
        repencoder.report["model_sha256"]
        != manifest["repencoder"]["shared_file_sha256"]
    ):
        raise ValueError(
            "Shared RepEncoder differs from the verified renamed checkpoint"
        )
    provider = RepEncoderInferenceMemoryProvider(
        repencoder,
        RepEncoderInferenceProviderConfig(
            seed=seed,
            trajectory_fov_horizontal_fov_degrees=memory_fov_h_deg,
            trajectory_fov_vertical_fov_degrees=memory_fov_v_deg,
            trajectory_fov_samples_per_axis=memory_fov_samples_per_axis,
        ),
    )

    def transformer(branch):
        model = WorldCrafterTransformer3DModel.from_pretrained(
            root / f"transformer_{branch}_noise", torch_dtype=torch.bfloat16
        )
        patch_worldcrafter_transformer_ucpe(
            model,
            method="relray_absmap",
            height=height,
            width=width,
            attn_compress=8,
            adaptation_method="parallel",
            attention_cls=FastUcpeSelfAttention,
        )
        loaded = load_ucpe_camera_adapter_weights(
            model, root / f"adapter_{branch}_noise" / "transformer_partial.pth"
        )
        if loaded["loaded_tensor_keys"] != loaded["expected_tensor_keys"]:
            raise ValueError(f"Incomplete {branch} UCPE state")
        model = replace_rmsnorm_with_fp32(model)
        model = replace_all_norms_with_flash_norms(model)
        configure_attention(model, attention_backend)
        return model

    early = transformer("high")
    replace_rope_with_flash_rope()
    provenance = contract.student_scheduler
    scheduler = WorldCrafterScheduler.from_config(
        WorldCrafterScheduler.from_pretrained(shared / "scheduler").config,
        num_train_timesteps=provenance.num_train_timesteps,
        shift=provenance.shift,
        stages=provenance.stages,
        stage_range=list(provenance.stage_range),
        gamma=provenance.gamma,
        scheduler_type="dmd",
        use_dynamic_shifting=provenance.use_dynamic_shifting,
        time_shift_type=provenance.time_shift_type,
    )
    pipe = WorldCrafterPipeline(
        tokenizer=AutoTokenizer.from_pretrained(shared / "tokenizer"),
        text_encoder=UMT5EncoderModel.from_pretrained(
            shared / "text_encoder", torch_dtype=torch.bfloat16
        ),
        transformer=early,
        vae=AutoencoderKLWan.from_pretrained(shared / "vae", torch_dtype=torch.float32),
        scheduler=scheduler,
        is_distilled=True,
    )
    early_lora = load_model_adapter(pipe, adapters[0])
    pipe.dmd_timestep_contract = contract
    pipe.to(device)
    late = transformer("low")
    loader = copy.copy(pipe)
    loader.register_modules(transformer=late)
    late_lora = load_model_adapter(loader, adapters[1])
    compact = [compact_ucpe(m) for m in (early, late)]
    pipe.resident_branches = ResidentBranches(early, late)
    late.to(device)
    pipe.stage_transformers = (early, early, late)
    pipe.stage_model_trace = []

    def record(branch):
        def hook(module, args, kwargs, output):
            chunk, stage, step, stage_steps = pipe.stage_forward_context
            low_noise = (
                stage >= 1
                if getattr(pipe, "fast_inference_mode", "i2v") == "t2v"
                else stage == 2 and step >= stage_steps // 2
            )
            expected = "old" if low_noise else "equal"
            if branch != expected or pipe.resident_branches.active != branch:
                raise RuntimeError("Fast transformer/adapter routing mismatch")
            pipe.stage_model_trace.append(
                dict(
                    chunk=chunk,
                    stage=stage,
                    step=step,
                    stage_steps=stage_steps,
                    branch=branch,
                )
            )

        return hook

    early.register_forward_hook(record("equal"), with_kwargs=True)
    late.register_forward_hook(record("old"), with_kwargs=True)
    if enable_compile:
        # Keep routing and shared-weight switches outside compiled graphs.
        # Compile each branch's blocks without changing parameter storage.
        for branch in (early, late):
            for block in branch.blocks:
                block.compile(mode="default", dynamic=False)
    gc.collect()
    torch.cuda.empty_cache()
    model = cls(
        pipeline=pipe,
        memory_provider=provider,
        model_path=root,
        device=device,
        attention_backend=attention_backend,
        adapter_load={"high": early_lora, "low": late_lora},
        height=height,
        width=width,
    )
    model.model_type = "fast"
    model.fast_config = config
    model.fast_report = dict(
        compile_enabled=bool(enable_compile),
        compile_scope="transformer_blocks" if enable_compile else None,
        contract_fingerprint=contract.fingerprint,
        compact_ucpe=compact,
        resident=pipe.resident_branches.report,
    )
    return model
