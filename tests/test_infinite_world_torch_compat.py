from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from worldfoundry.core import autocast_context
from worldfoundry.core.model_loading import load_torch_state_dict


class InfiniteWorldTorchCompatTest(unittest.TestCase):
    def test_load_torch_state_dict_prefers_weights_only(self) -> None:
        calls = []
        sentinel = {"weights": torch.tensor([1.0])}

        def fake_load(path, **kwargs):
            calls.append((path, kwargs))
            return sentinel

        with patch.object(torch, "load", side_effect=fake_load):
            loaded = load_torch_state_dict("/tmp/checkpoint.pt", map_location="cpu")

        self.assertIs(loaded, sentinel)
        self.assertEqual(
            calls,
            [
                (
                    "/tmp/checkpoint.pt",
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
            loaded = load_torch_state_dict("/tmp/checkpoint.pt", map_location="cpu")

        self.assertIs(loaded, sentinel)
        self.assertEqual(
            calls,
            [
                (
                    "/tmp/checkpoint.pt",
                    {"map_location": "cpu", "weights_only": True},
                ),
                (
                    "/tmp/checkpoint.pt",
                    {"map_location": "cpu"},
                ),
            ],
        )

    def test_autocast_context_skips_cpu(self) -> None:
        def fail_autocast(*args, **kwargs):
            raise AssertionError("cpu path should not call torch.amp.autocast")

        with patch.object(torch.amp, "autocast", side_effect=fail_autocast):
            with autocast_context("cpu", dtype=torch.float32):
                pass


if __name__ == "__main__":
    unittest.main()
