"""WorldFoundry pipeline backed by the in-tree Helios runtime."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from worldfoundry.core.media.codecs.image import load_pil_image
from worldfoundry.pipelines.video_official.pipeline_official_video import (
    OfficialVideoPipeline,
)


class HeliosPipeline(OfficialVideoPipeline):
    """Generate offline videos or resident Distilled AR segments.

    Base and Mid retain the artifact-oriented official CLI path. Distilled
    additionally exposes the native realtime contract, which holds weights,
    prompt embeddings, RNG state, image conditioning, and latent histories in
    memory between 33-frame segments.
    """

    MODEL_ID = "helios"
    GENERATION_TYPE = "t2v"

    def _ensure_realtime_session(self) -> Any:
        session = getattr(self, "_realtime_session", None)
        if session is not None:
            return session
        plan = self.runtime.runtime_plan()
        checkpoint = plan.get("checkpoint_path")
        if not checkpoint:
            missing = "; ".join(str(item) for item in plan.get("missing") or ())
            raise FileNotFoundError(
                "Helios-Distilled checkpoint is required for resident inference"
                + (f": {missing}" if missing else ".")
            )
        from ...synthesis.visual_generation.helios.realtime import HeliosRealtimeSession

        session = HeliosRealtimeSession(checkpoint)
        self._realtime_session = session
        return session

    def prepare_realtime(self) -> dict[str, Any]:
        session = self._ensure_realtime_session()
        return {
            "realtime_spec": session.realtime_spec().to_payload(),
            "runtime_info": session.runtime_info(),
        }

    def configure_realtime(
        self,
        images: Any = None,
        prompt: str = "",
        seed: int = 42,
        height: int = 384,
        width: int = 640,
        fps: int = 12,
        **_: Any,
    ) -> dict[str, Any]:
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("Helios realtime requires a non-empty user prompt.")
        image = None if images is None else load_pil_image(images)
        return self._ensure_realtime_session().configure(
            image,
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            fps=fps,
        )

    def stream_realtime(
        self,
        interactions: Sequence[str] | None = None,
        prompt: str | None = None,
        realtime_segments: Sequence[Mapping[str, Any]] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        unsupported = [
            str(item)
            for item in (interactions or ())
            if str(item).strip() and str(item).strip().lower() != "prompt_update"
        ]
        if unsupported:
            raise ValueError(
                "Helios does not implement keyboard/camera controls. Update the prompt at a "
                f"33-frame segment boundary instead; unsupported controls: {unsupported}"
            )
        return self._ensure_realtime_session().generate_next(
            prompt=prompt,
            prompt_segments=realtime_segments,
        )

    def reset_realtime(self) -> None:
        session = getattr(self, "_realtime_session", None)
        if session is not None:
            session.reset()


__all__ = ["HeliosPipeline"]
