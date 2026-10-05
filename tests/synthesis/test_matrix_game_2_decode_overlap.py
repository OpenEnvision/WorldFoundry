"""Real CUDA dependency, recurrent-cache, error and teardown checks for MG2."""

from __future__ import annotations

import gc
from types import SimpleNamespace

import pytest
import torch

from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.pipeline.causal_inference import (
    CausalInferencePipeline,
    CausalInferenceSession,
)


class _Generator(torch.nn.Module):
    def __init__(self, *, fail_refresh=False):
        super().__init__()
        self.model = SimpleNamespace(local_attn_size=2)
        self.streams = []
        self.fail_refresh = fail_refresh
        self.timeline = []

    def get_scheduler(self):
        return SimpleNamespace()

    def forward(self, *, noisy_image_or_video, conditional_dict, kv_cache, **kwargs):
        kv_cache[0]["calls"] += 1
        self.timeline.append("refresh" if kv_cache[0]["calls"] % 2 == 0 else "denoise")
        if noisy_image_or_video.is_cuda:
            self.streams.append(torch.cuda.current_stream(noisy_image_or_video.device))
        if self.fail_refresh and kv_cache[0]["calls"] % 2 == 0:
            raise RuntimeError("refresh failed")
        return None, noisy_image_or_video * 0.5 + conditional_dict["cond_concat"] * 0.25


class _Decoder(torch.nn.Module):
    def __init__(self, *, fail=False, delay=False):
        super().__init__()
        self.streams = []
        self.done = None
        self.fail = fail
        self.delay = delay
        self.timeline = []

    def forward(self, latent, previous):
        self.timeline.append("decode")
        if latent.is_cuda:
            self.streams.append(torch.cuda.current_stream(latent.device))
            if self.delay:
                torch.cuda._sleep(5_000_000)
        state = latent.mean(dim=(2, 3, 4), keepdim=True)
        if previous is not None:
            state = state + previous
        video = latent[:, :, :3].float() * 2 + state
        if latent.is_cuda:
            self.done = torch.cuda.Event()
            self.done.record()
        if self.fail:
            raise RuntimeError("decode failed")
        return video, [state]


def _pipeline(device, *, overlap, fail_decode=False, fail_refresh=False, delay=False):
    args = SimpleNamespace(denoising_step_list=[1], warp_denoising_step=False, num_frame_per_block=1, context_noise=0)
    pipeline = CausalInferencePipeline(
        args, generator=_Generator(fail_refresh=fail_refresh), vae_decoder=_Decoder(fail=fail_decode, delay=delay)
    )
    pipeline.overlap_vae_decode = overlap
    pipeline.vae_decoder.timeline = pipeline.generator.timeline
    condition = {
        "cond_concat": torch.ones(1, 16, 3, 2, 2, device=device),
        "visual_context": torch.zeros(1, 1, 2, device=device),
        "keyboard_cond": torch.zeros(1, 9, 7, device=device),
    }
    session = CausalInferenceSession(
        conditional_dict=condition,
        mode="templerun",
        batch_size=1,
        dtype=torch.float32,
        device=condition["cond_concat"].device,
        kv_cache=[{"calls": 0}],
        mouse_kv_cache=[],
        keyboard_kv_cache=[],
        crossattn_cache=[],
        vae_cache=[None],
    )
    pipeline._session = session
    return pipeline, session


def test_requested_overlap_has_an_explicit_cpu_fallback():
    pipeline, session = _pipeline("cpu", overlap=True)
    block = pipeline.generate_next_block(session, torch.ones(1, 16, 1, 2, 2))
    torch.testing.assert_close(block.video, torch.full((1, 1, 3, 2, 2), 2.25))
    assert pipeline._decode_overlap_runtime["execution"] == "synchronous (non-CUDA fallback)"
    assert pipeline._decode_overlap_runtime["calls"] == 0
    assert pipeline.generator.timeline == ["denoise", "refresh", "decode"]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA stream execution")
def test_decode_overlap_joins_the_nondefault_caller_and_preserves_recurrent_cache_and_lifetimes(monkeypatch):
    caller = torch.cuda.Stream()
    with torch.cuda.stream(caller):
        pipeline, session = _pipeline("cuda", overlap=True, delay=True)
        outputs = []
        for value in [1.0, 2.0]:
            with monkeypatch.context() as guarded:
                guarded.setattr(
                    torch.cuda, "synchronize", lambda *args, **kwargs: pytest.fail("unexpected device barrier")
                )
                guarded.setattr(
                    torch.cuda.Stream,
                    "synchronize",
                    lambda self: pytest.fail("unexpected host stream wait during generation"),
                )
                guarded.setattr(
                    torch.cuda.Event,
                    "synchronize",
                    lambda self: pytest.fail("unexpected host event wait during generation"),
                )
                block = pipeline.generate_next_block(session, torch.full((1, 16, 1, 2, 2), value, device="cuda"))
            outputs.append(block.video.clone())
            del block
            gc.collect()
            scratch = [torch.empty(1, 16, 1, 2, 2, device="cuda") for _ in range(16)]
            del scratch
        done = torch.cuda.Event()
        done.record(caller)
    done.synchronize()
    torch.testing.assert_close(outputs[0].cpu(), torch.full((1, 1, 3, 2, 2), 2.25))
    torch.testing.assert_close(outputs[1].cpu(), torch.full((1, 1, 3, 2, 2), 4.5))
    assert all(stream == caller for stream in pipeline.generator.streams)
    assert all(stream != caller for stream in pipeline.vae_decoder.streams)
    assert pipeline.vae_decoder.streams[0] == pipeline.vae_decoder.streams[1]
    assert pipeline._decode_overlap_runtime["calls"] == 2
    assert pipeline.generator.timeline == ["denoise", "decode", "refresh"] * 2
    assert session.current_start_frame == 2
    pipeline.reset_session()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA stream execution")
@pytest.mark.parametrize("teardown", ["reset", "close", "replace"])
def test_reset_drains_and_releases_only_the_owned_decode_stream(monkeypatch, teardown):
    pipeline, session = _pipeline("cuda", overlap=True, delay=True)
    pipeline.generate_next_block(session, torch.ones(1, 16, 1, 2, 2, device="cuda"))
    owned = pipeline._decode_stream
    assert owned is not None
    waited = []
    actual_wait = torch.cuda.Stream.synchronize

    def wait(stream):
        waited.append(stream)
        return actual_wait(stream)

    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda *args, **kwargs: pytest.fail("reset must not drain the device")
    )
    monkeypatch.setattr(torch.cuda.Stream, "synchronize", wait)
    if teardown == "replace":
        pipeline.frame_seq_length = 4
        pipeline.num_transformer_blocks = 1
        replacement = pipeline.start_session(session.conditional_dict, mode="templerun")
        assert replacement is not session
        assert replacement.current_start_frame == 0
    elif teardown == "close":
        pipeline.close()
    else:
        pipeline.reset_session()
    assert waited == [owned]
    assert pipeline.vae_decoder.done.query()
    assert pipeline._decode_stream is None
    assert pipeline._session is None if teardown != "replace" else pipeline._session is replacement


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA stream execution")
@pytest.mark.parametrize("failure", ["decode", "refresh"])
def test_failed_overlap_drains_enqueued_decode_and_invalidates_the_session(monkeypatch, failure):
    pipeline, session = _pipeline(
        "cuda", overlap=True, fail_decode=failure == "decode", fail_refresh=failure == "refresh", delay=True
    )
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda *args, **kwargs: pytest.fail("failure cleanup must not drain the device")
    )
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        pipeline.generate_next_block(session, torch.ones(1, 16, 1, 2, 2, device="cuda"))
    assert pipeline.vae_decoder.done.query()
    assert session.invalidated
    assert session.current_start_frame == 0
    assert pipeline._decode_stream is None
    with pytest.raises(RuntimeError, match="invalidated"):
        pipeline.generate_next_block(session, torch.ones(1, 16, 1, 2, 2, device="cuda"))
    pipeline.reset_session()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA stream execution")
def test_overlap_profile_measures_each_stream_and_the_joined_wall_time(monkeypatch):
    caller = torch.cuda.Stream()
    with torch.cuda.stream(caller):
        pipeline, session = _pipeline("cuda", overlap=True)
        monkeypatch.setattr(
            torch.cuda, "synchronize", lambda *args, **kwargs: pytest.fail("profiling must not drain the device")
        )
        block = pipeline.generate_next_block(session, torch.ones(1, 16, 1, 2, 2, device="cuda"), profile=True)
    assert block.model_ms >= 0
    assert block.decode_ms >= 0
    assert block.total_ms >= max(block.model_ms, block.decode_ms)
    assert pipeline.vae_decoder.streams[0] != caller
    pipeline.reset_session()
