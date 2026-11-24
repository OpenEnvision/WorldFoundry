"""Public LaST-R1 contract checks that do not require checkpoint loading."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from worldfoundry.synthesis.action_generation.last_r1 import LastR1Synthesis, last_r1_synthesis


def test_numpy_image_is_written_and_action_artifact_is_returned(tmp_path, monkeypatch):
    synthesis = object.__new__(LastR1Synthesis)
    synthesis.checkpoint_path = tmp_path / "checkpoint"
    synthesis.python_executable = sys.executable
    synthesis.device = "cuda:2"
    monkeypatch.setattr(synthesis, "preflight", lambda: {"status": "ready"})

    class CompletedWorker:
        def __init__(self, command, **_kwargs):
            request = json.loads(Path(command[-1]).read_text())
            with Image.open(request["image_path"]) as image:
                assert image.size == (16, 12)
                assert image.mode == "RGB"
            run_dir = Path(request["run_dir"])
            (run_dir / "result.json").write_text(
                json.dumps({"status": "success", "action_shape": [1, 8, 7], "all_finite": True})
            )

        def wait(self, timeout):
            assert timeout > 0
            return 0

    monkeypatch.setattr(last_r1_synthesis.subprocess, "Popen", CompletedWorker)
    artifact = tmp_path / "action_trace.json"
    result = synthesis.predict(
        prompt="Pick up the block",
        image=np.zeros((12, 16, 3), dtype=np.uint8),
        output_path=artifact,
    )

    assert result["artifact_path"] == str(artifact)
    assert json.loads(artifact.read_text())["action_shape"] == [1, 8, 7]


def test_studio_discovers_the_public_pipeline():
    from worldfoundry.studio.inference.catalog import find_entry, find_runtime_entry

    for lookup in (find_entry, find_runtime_entry):
        entry = lookup("last-r1")
        assert entry.module_path == "worldfoundry.pipelines.component_pipelines"
        assert entry.class_name == "LastR1Pipeline"
        assert entry.category == "Embodied Action"


def _tiny_qwen3_vl_config():
    from transformers import Qwen3VLConfig

    return Qwen3VLConfig(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 64,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "deepstack_visual_indexes": [0],
            "num_position_embeddings": 16,
        },
    )


def test_policy_model_adds_the_value_head_and_changes_only_the_action_block():
    import torch
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    from worldfoundry.synthesis.action_generation.last_r1.modeling import LastR1Qwen3VLForConditionalGeneration

    torch.manual_seed(0)
    config = _tiny_qwen3_vl_config()
    stock = Qwen3VLForConditionalGeneration(config).eval()
    policy = LastR1Qwen3VLForConditionalGeneration(config).eval()
    incompatible = policy.load_state_dict(stock.state_dict(), strict=False)
    assert sorted(incompatible.missing_keys) == sorted(
        f"value_head.mlp.{index}.{name}" for index in (0, 2, 4, 6) for name in ("weight", "bias")
    )
    assert not incompatible.unexpected_keys

    embeds = stock.model.get_input_embeddings()(torch.randint(0, 64, (1, 10)))
    mask = torch.ones(1, 10, dtype=torch.long)
    with torch.no_grad():
        reference = stock.model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
        plain = policy.model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
        actions = policy.model(
            inputs_embeds=embeds, attention_mask=mask, action_length=4, attn_mode="causal"
        ).last_hidden_state
    assert torch.equal(plain, reference)
    torch.testing.assert_close(actions[:, :-4], reference[:, :-4])
    assert not torch.allclose(actions[:, -4:], reference[:, -4:])


def test_action_tokens_decode_to_bin_centres():
    import numpy as np

    from worldfoundry.synthesis.action_generation.last_r1.modeling import decode_action_tokens

    end = 1000
    centres = (np.linspace(-1, 1, 256)[:-1] + np.linspace(-1, 1, 256)[1:]) / 2
    decoded = decode_action_tokens(np.array([end - 1, end - 128, end - 256]), action_vocab_end=end)
    np.testing.assert_array_equal(decoded, centres[[0, 127, 254]])
