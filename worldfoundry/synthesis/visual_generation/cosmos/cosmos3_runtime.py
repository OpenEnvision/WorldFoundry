"""Compatibility runtime surface for the native Cosmos3 recipe."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.pipelines.cosmos.pipeline_cosmos3 import Cosmos3Pipeline


@dataclass(frozen=True, slots=True)
class Cosmos3RuntimePlan:
    model_path: str | None
    variant_id: str
    backend: str = "worldfoundry-native-diffusion"
    native_inference: bool = True
    blocked: bool = False
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Cosmos3RuntimeOutput:
    video: Any
    sound: Any = None
    action: Any = None
    audio_sample_rate: int | None = None
    artifact_path: str | None = None


class Cosmos3Runtime:
    """Delegate the historical runtime API to the canonical public pipeline."""

    def __init__(self, pipeline: Cosmos3Pipeline) -> None:
        self.pipeline = pipeline
        self.variant_id = pipeline.model_id

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        *,
        device: str = "cuda",
        variant_id: str | None = None,
        **kwargs: Any,
    ) -> "Cosmos3Runtime":
        return cls(
            Cosmos3Pipeline.from_pretrained(
                model_path=model_path,
                device=device,
                model_id=variant_id,
                **kwargs,
            )
        )

    @classmethod
    def plan(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        plan = Cosmos3Pipeline.plan(model_path=model_path, **kwargs)
        source = plan["checkpoint"]
        blockers: list[str] = []
        if source is not None and Path(str(source)).expanduser().exists():
            root = Path(str(source)).expanduser()
            required = (
                "transformer/diffusion_pytorch_model.safetensors.index.json",
                "vae/diffusion_pytorch_model.safetensors",
                "sound_tokenizer/diffusion_pytorch_model.safetensors",
                "text_tokenizer/tokenizer_config.json",
                "scheduler/scheduler_config.json",
            )
            blockers.extend(relative for relative in required if not (root / relative).is_file())
        return Cosmos3RuntimePlan(
            model_path=str(source) if source is not None else None,
            variant_id=str(plan["model_id"]),
            blocked=bool(blockers),
            blockers=tuple(blockers),
        ).to_dict()

    def predict(self, *args: Any, **kwargs: Any) -> Cosmos3RuntimeOutput:
        kwargs["return_dict"] = True
        result = self.pipeline(*args, **kwargs)
        return Cosmos3RuntimeOutput(
            video=result["video"],
            sound=result.get("sound"),
            action=result.get("action"),
            audio_sample_rate=result.get("audio_sampling_rate"),
            artifact_path=result.get("artifact_path"),
        )

    def api_init(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise NotImplementedError("Cosmos3 is a local native runtime and does not use an API backend")


__all__ = ["Cosmos3Runtime", "Cosmos3RuntimeOutput", "Cosmos3RuntimePlan"]
