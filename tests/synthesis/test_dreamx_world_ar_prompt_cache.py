from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.synthesis.visual_generation.dreamx_world import ar_realtime


def _session(monkeypatch):
    for name, value in (("HEIGHT", 32), ("WIDTH", 32), ("LATENT_HEIGHT", 2), ("LATENT_WIDTH", 2),
                        ("LATENT_CHANNELS", 4), ("SPATIAL_TOKENS_PER_FRAME", 1)):
        monkeypatch.setattr(ar_realtime, name, value)
    session = ar_realtime.DreamXWorldARRealtimeSession.__new__(ar_realtime.DreamXWorldARRealtimeSession)
    session.device = torch.device("cpu")
    session.dtype = torch.float32
    session.checkpoint = Path("checkpoint")
    session.timesteps = torch.tensor([1000.0, 500.0])
    session._kv_cache = None
    session._crossattn_cache = None
    session._prompt_cache = {}
    session._prompt = ""
    session.last_metrics = {}
    session.encoded = []
    session.projections = []
    session.denoised = []
    session.encoding_failure = None
    session.generation_failure = False

    def encode(*, text_prompts):
        prompt = text_prompts[0]
        if prompt == session.encoding_failure:
            raise ValueError("injected text encoding failure")
        session.encoded.append(prompt)
        return {"prompt_embeds": torch.full((1, 4), sum(map(ord, prompt)) / 1000)}

    class Generator:
        model = SimpleNamespace(blocks=(None, None), num_heads=2, dim=4, local_attn_size=12)
        scheduler = SimpleNamespace(add_noise=lambda denoised, *_: denoised)

        def __call__(self, **kwargs):
            context = kwargs["conditional_dict"]["prompt_embeds"]
            for index, cache in enumerate(kwargs["crossattn_cache"]):
                if not cache["is_init"]:
                    cache.update(is_init=True, k=context.clone(), v=context.clone() * 2)
                    session.projections.append(index)
            if session.generation_failure:
                raise ValueError("injected generation failure")
            cached = kwargs["crossattn_cache"][0]["v"].mean()
            latent = kwargs["noisy_image_or_video"] * 0.5 + cached * 0.01
            return torch.zeros_like(latent), latent

    class VAE:
        model = SimpleNamespace(clear_cache=lambda: None)

        @staticmethod
        def encode_to_latent(_image):
            return torch.zeros(1, 1, 4, 2, 2)

        @staticmethod
        def decode_to_pixel(latent, *, use_cache):
            assert use_cache
            session.denoised.append(latent.clone())
            count = 9 if session._first_block else 12
            return latent.mean().expand(1, count, 3, 2, 2).clone()

    session.text_encoder = encode
    session.generator = Generator()
    session.vae = VAE()
    session._camera_condition = lambda *_: {}
    session.configure(Image.new("RGB", (32, 32)), prompt="  A  ")
    return session


def test_unchanged_normalized_prompt_reuses_cross_attention(monkeypatch):
    session = _session(monkeypatch)
    session.generate(prompt=" A ", seed=43)
    previous = [(cache["k"], cache["v"]) for cache in session._crossattn_cache]
    session.generate(prompt="A", seed=44)
    session.generate(prompt="  A  ", seed=45)
    assert session.encoded == ["A"]
    assert session.projections == [0, 1]
    assert all(cache["k"] is k and cache["v"] is v
               for cache, (k, v) in zip(session._crossattn_cache, previous))


def test_prompt_switch_and_switch_back_rebuild_cross_attention(monkeypatch):
    session = _session(monkeypatch)
    for prompt in ("A", " B ", "B", "A", "A"):
        session.generate(prompt=prompt)
        assert all(torch.equal(cache["k"], session._conditional_dict["prompt_embeds"])
                   for cache in session._crossattn_cache)
    assert session.encoded == ["A", "B"]
    assert session.projections == [0, 1] * 3


@pytest.mark.parametrize("prompt", [None, "", "  "])
def test_absent_or_empty_prompt_preserves_active_condition(monkeypatch, prompt):
    session = _session(monkeypatch)
    session.generate(prompt="A")
    session.generate(prompt=prompt)
    assert session.encoded == ["A"]
    assert session.projections == [0, 1]


def test_cached_rollout_matches_forced_recomputation_latents_and_pixels(monkeypatch):
    cached, reference = _session(monkeypatch), _session(monkeypatch)
    for index, prompt in enumerate(("A", "A", None, "B", "B", "A", "A")):
        for cache in reference._crossattn_cache:
            cache["is_init"] = False
        actual = cached.generate(prompt=prompt, seed=43 + index)
        expected = reference.generate(prompt=prompt, seed=43 + index)
        torch.testing.assert_close(cached.denoised[-1], reference.denoised[-1], rtol=0, atol=0)
        np.testing.assert_array_equal(actual["frames"], expected["frames"])
    assert len(cached.projections) == 6
    assert len(reference.projections) == 14


def test_reset_and_reconfigure_same_prompt_invalidates_all_cached_layers(monkeypatch):
    session = _session(monkeypatch)
    session.generate(prompt="A", seed=43)
    expected = session.denoised[-1].clone()
    session.reset()
    assert session._prompt == ""
    assert all(not cache["is_init"] for cache in session._crossattn_cache)
    with pytest.raises(RuntimeError, match="not configured"):
        session.generate(prompt="A")
    session.configure(Image.new("RGB", (32, 32)), prompt="A")
    session.generate(prompt="A", seed=43)
    assert session.encoded == ["A"]
    assert session.projections == [0, 1] * 2
    torch.testing.assert_close(session.denoised[-1], expected, rtol=0, atol=0)


def test_failed_prompt_encoding_keeps_the_previous_active_condition(monkeypatch):
    session = _session(monkeypatch)
    session.generate(prompt="A")
    original = session._conditional_dict
    session.encoding_failure = "B"
    with pytest.raises(ValueError, match="text encoding"):
        session.generate(prompt="B")
    assert session._prompt == "A"
    assert session._conditional_dict is original
    assert all(cache["is_init"] for cache in session._crossattn_cache)
    session.generate(prompt="A")
    assert session.projections == [0, 1]
    session.encoding_failure = None
    session.generate(prompt="B")
    assert session.projections == [0, 1] * 2


def test_generation_failure_reset_rebuilds_the_requested_prompt(monkeypatch):
    session = _session(monkeypatch)
    session.generation_failure = True
    with pytest.raises(ValueError, match="generation"):
        session.generate(prompt="B")
    session.reset()
    session.generation_failure = False
    session.configure(Image.new("RGB", (32, 32)), prompt="A")
    session.generate(prompt="A")
    assert all(torch.equal(cache["k"], session._conditional_dict["prompt_embeds"])
               for cache in session._crossattn_cache)
    assert session.projections == [0, 1] * 2


def test_prompt_eviction_and_revisit_preserve_bounded_text_cache(monkeypatch):
    session = _session(monkeypatch)
    for prompt in ("A", "B", "C", "D", "E", "F", "G", "A", "A"):
        session.generate(prompt=prompt)
        assert len(session._prompt_cache) <= 4
        assert all(torch.equal(cache["k"], session._conditional_dict["prompt_embeds"])
                   for cache in session._crossattn_cache)
    assert session.encoded.count("A") == 2
    assert len(session.projections) == 16
