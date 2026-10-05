"""Strict Diffusers checkpoint conversion into the shared H3 parameter layout."""
import json
from pathlib import Path

import torch

from ...loading import _shard_paths
from .model import JINGTransformer


def convert_state_dict(state):
    out, qkv = {}, {}
    prefixes = {
        "proj_in.": "video_patch_proj.", "audio_proj_in.": "audio_patch_proj.",
        "context_embedder.": "condition_proj.", "time_embedder.linear_1.": "time_embedder.proj_in.",
        "time_embedder.linear_2.": "time_embedder.proj_out.", "proj_out.": "final_layer.video_out.",
        "audio_proj_out.": "final_layer.audio_out.", "norm_out.norm.": "final_layer.norm.",
        "norm_out.linear.": "final_layer.adaln_proj.linear.",
        "transformer_blocks.": "blocks.", "token_refiner.refiner_blocks.": "token_refiner.blocks.",
    }
    for key, value in state.items():
        for old, new in prefixes.items():
            if key.startswith(old):
                key = new + key[len(old):]
                break
        key = key.replace(".attn.norm_q.", ".attn.q_norm.").replace(".attn.norm_k.", ".attn.k_norm.")
        key = key.replace(".attn.to_out.0.", ".attn.out_proj.")
        if ".attn.to_" in key:
            base, tail = key.split(".attn.to_", 1)
            if tail not in ("q.weight", "k.weight", "v.weight"):
                raise ValueError(f"Unknown attention weight: {key}")
            qkv.setdefault(base + ".attn.qkv_proj.weight", {})[tail[0]] = value
            continue
        if ".ff.net.0.proj." in key:
            key = key.replace(".ff.net.0.proj.", ".mlp.fc1.")
            # Diffusers stores [up, gate]; the canonical H3 kernel stores [gate, up].
            up, gate = value.chunk(2, dim=0)
            value = torch.cat((gate, up), dim=0)
        key = key.replace(".ff.net.2.", ".mlp.fc2.")
        if key in out:
            raise ValueError(f"Duplicate converted weight: {key}")
        out[key] = value
    for key, values in qkv.items():
        if set(values) != {"q", "k", "v"}:
            raise ValueError(f"Incomplete QKV projection: {key}")
        out[key] = torch.cat([values[x] for x in "qkv"], dim=0)
    return out


def load_transformer(path, *, device="cpu"):
    from safetensors.torch import load_file

    path = Path(path)
    if (path / "jing_flash_v1").is_dir():
        path = path / "jing_flash_v1"
    config = json.loads((path / "config.json").read_text())
    with torch.device("meta"):
        model = JINGTransformer(config)
    raw = {}
    for shard in _shard_paths(path):
        for key, value in load_file(str(shard)).items():
            if key in raw:
                raise ValueError(f"Duplicate checkpoint tensor: {key}")
            raw[key] = value
    converted = convert_state_dict(raw)
    expected = model.state_dict()
    for key in converted:
        if key not in expected or converted[key].shape != expected[key].shape:
            raise ValueError(f"Unexpected checkpoint tensor/shape: {key}")
        converted[key] = converted[key].to(dtype=expected[key].dtype)
    dim = model.arch.rope_inv_freq_len
    converted["rope.inv_freq"] = 1 / (float(config.get("rope_theta", 10000)) ** (
        torch.arange(0, 2 * dim, 2, dtype=torch.float32) / (2 * dim)))
    result = model.load_state_dict(converted, strict=True, assign=True)
    model.post_load_weights()
    model.loading_report = {"missing": list(result.missing_keys), "unexpected": list(result.unexpected_keys),
                            "checkpoint_tensors": len(raw), "native_tensors": len(converted)}
    return model.eval().requires_grad_(False).to(device)
