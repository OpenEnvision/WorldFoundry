from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from worldfoundry.core.model_loading import load_torch_state_dict
from worldfoundry.synthesis.visual_generation.lingbot_video import transformer_lingbot_video


class LingBotTorchCompatTest(unittest.TestCase):
    def test_fa3_varlen_compat_supplies_new_sequence_length_arguments(self) -> None:
        captured = {}

        def fake_flash(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=False,
        ):
            captured.update(seqused_q=seqused_q, seqused_k=seqused_k)
            return q

        with patch.object(
            transformer_lingbot_video,
            "flash_attn_varlen_func_v3",
            fake_flash,
        ):
            result = transformer_lingbot_video._flash_attn_varlen_v3_compat(
                q="q",
                k="k",
                v="v",
                cu_seqlens_q="cu_q",
                cu_seqlens_k="cu_k",
                max_seqlen_q=8,
                max_seqlen_k=8,
                causal=False,
            )

        self.assertEqual(result, "q")
        self.assertEqual(captured, {"seqused_q": None, "seqused_k": None})

    def test_load_torch_state_dict_prefers_weights_only(self) -> None:
        calls = []
        sentinel = {"weights": torch.tensor([1.0])}

        def fake_load(path, **kwargs):
            calls.append((path, kwargs))
            return sentinel

        with patch.object(torch, "load", side_effect=fake_load):
            loaded = load_torch_state_dict("/tmp/lingbot.pt", map_location="cpu")

        self.assertIs(loaded, sentinel)
        self.assertEqual(
            calls,
            [
                (
                    "/tmp/lingbot.pt",
                    {"map_location": "cpu", "weights_only": True},
                )
            ],
        )

    def test_load_torch_state_dict_falls_back_without_weights_only(self) -> None:
        calls = []
        sentinel = {"weights": torch.tensor([2.0])}

        def fake_load(path, **kwargs):
            calls.append((path, kwargs))
            if "weights_only" in kwargs:
                raise TypeError("load() got an unexpected keyword argument 'weights_only'")
            return sentinel

        with patch.object(torch, "load", side_effect=fake_load):
            loaded = load_torch_state_dict("/tmp/lingbot.pt", map_location="cpu")

        self.assertIs(loaded, sentinel)
        self.assertEqual(
            calls,
            [
                (
                    "/tmp/lingbot.pt",
                    {"map_location": "cpu", "weights_only": True},
                ),
                (
                    "/tmp/lingbot.pt",
                    {"map_location": "cpu"},
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
