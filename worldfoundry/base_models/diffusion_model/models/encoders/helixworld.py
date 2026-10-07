"""Camera/action conditions and triplet captions over the shared LTX Gemma encoder."""

from __future__ import annotations

from dataclasses import replace

from ...contracts import Conditioning
from .helixworld_controls import prepare_video_control
from .ltx import build_ltx_prompt_conditioner


class HelixWorldConditioner:
    def __init__(self, text_conditioner) -> None:
        self.text_conditioner = text_conditioner

    def encode(self, request, *, device, dtype):
        control = prepare_video_control(request, device=device, dtype=dtype)
        audio = request.inputs.get("audio_prompt")
        joint = request.inputs.get("av_prompt")
        if (audio is None) != (joint is None):
            raise ValueError("audio_prompt and av_prompt must be supplied together")
        prompts = request.prompts
        if audio is not None:
            prompts = tuple(
                f"Video description:\n{prompt.strip()}\n\nAudio description:\n{str(audio).strip()}\n\nJoint audio-visual description:\n{str(joint).strip()}"
                for prompt in prompts
            )
        elif any(not prompt.startswith("Video description:\n") for prompt in prompts):
            raise ValueError("supply video, audio and joint descriptions or a canonical HelixWorld triplet caption")
        conditioning = self.text_conditioner.encode(replace(request, prompt=prompts), device=device, dtype=dtype)
        return Conditioning(
            positive=conditioning.positive, negative=conditioning.negative,
            shared={**conditioning.shared, "video_control": control},
        )


def build_helixworld_conditioner(context):
    return HelixWorldConditioner(build_ltx_prompt_conditioner(context))
