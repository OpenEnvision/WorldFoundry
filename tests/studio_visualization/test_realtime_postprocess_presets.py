from __future__ import annotations

import re
import types

import pytest

from worldfoundry.core.media.processing.postprocess import IdentityVideoPostProcessor, VideoSpec
from worldfoundry.studio.visualization.backends import world_realtime as realtime_backend
from worldfoundry.studio.visualization.launch_config import StudioLaunchConfig


@pytest.mark.parametrize("preset", ["", "none", "OFF", "Identity"])
def test_realtime_postprocess_identity_aliases(monkeypatch: pytest.MonkeyPatch, preset: str) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_POSTPROCESS_PRESET", preset)

    stream = realtime_backend._realtime_postprocess_stream(
        fps=16,
        launch_config=StudioLaunchConfig(model_id="test", device="cuda:3"),
    )

    assert stream.processor_names == ("identity",)


def test_realtime_postprocess_rtx_preset_without_overrides_keeps_preset_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import worldfoundry.core.media.processing.rtx as rtx_module

    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_POSTPROCESS_PRESET", "RTX_Deblur_Ultra")
    monkeypatch.setattr(
        rtx_module,
        "require_rtx_vfx_runtime",
        lambda **_kwargs: types.SimpleNamespace(gpu_name=None, package_version=None),
    )

    stream = realtime_backend._realtime_postprocess_stream(
        fps=16,
        launch_config=StudioLaunchConfig(model_id="test", device="cuda:1"),
    )

    [processor] = stream.chain.processors
    assert isinstance(processor, rtx_module.RTXVideoSuperResolutionPostProcessor)
    assert processor.config == rtx_module.rtx_postprocessor_from_preset("rtx-deblur-ultra", device=1).config


def test_realtime_postprocess_rtx_probe_failure_raises_before_serving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import worldfoundry.core.media.processing.rtx as rtx_module

    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_POSTPROCESS_PRESET", "rtx-super-resolution")
    monkeypatch.setattr(
        rtx_module,
        "probe_rtx_vfx_runtime",
        lambda **kwargs: rtx_module.RTXVFXCapability(
            available=False,
            device=kwargs["device"],
            reason="The optional nvidia-vfx package is not installed.",
            probe_status="failed",
        ),
    )

    with pytest.raises(RuntimeError, match="RTX video post-processing is unavailable: The optional nvidia-vfx"):
        realtime_backend._realtime_postprocess_stream(
            fps=16,
            launch_config=StudioLaunchConfig(model_id="test", device="cuda"),
        )


class _UnavailablePostProcessor(IdentityVideoPostProcessor):
    def probe(self, input_spec: VideoSpec | None) -> None:
        raise RuntimeError(f"stub runtime is unavailable for {input_spec!r}")


def _unavailable_postprocess_preset(_preset: str, *, device: int | None = None) -> _UnavailablePostProcessor:
    return _UnavailablePostProcessor()


def test_realtime_postprocess_registered_preset_probe_failure_raises_before_serving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import worldfoundry.core.media.processing.presets as presets_module

    monkeypatch.setitem(
        presets_module._POSTPROCESS_PRESET_FACTORIES,
        "test-unavailable",
        f"{__name__}:_unavailable_postprocess_preset",
    )
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_POSTPROCESS_PRESET", "test-unavailable")

    expected_spec = VideoSpec(height=360, width=640, fps=24, dtype="uint8")
    with pytest.raises(RuntimeError, match=re.escape(f"stub runtime is unavailable for {expected_spec!r}")):
        realtime_backend._realtime_postprocess_stream(
            fps=24,
            launch_config=StudioLaunchConfig(model_id="test", device="cuda"),
            native_resolution=(640, 360),
        )


@pytest.mark.parametrize(
    "variable",
    ["WORLDFOUNDRY_REALTIME_RTX_SCALE", "WORLDFOUNDRY_REALTIME_RTX_DEVICE", "WORLDFOUNDRY_REALTIME_RTX_QUALITY"],
)
def test_realtime_postprocess_rejects_rtx_overrides_for_non_rtx_preset(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_POSTPROCESS_PRESET", "identity")
    monkeypatch.setenv(variable, "1")

    with pytest.raises(ValueError, match=f"{variable} apply only to RTX postprocess presets"):
        realtime_backend._realtime_postprocess_stream(
            fps=16,
            launch_config=StudioLaunchConfig(model_id="test", device="cuda"),
        )


def test_realtime_postprocess_rtx_override_error_names_rtx_presets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_POSTPROCESS_PRESET", "rtx-super-res")
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_RTX_SCALE", "1.5")

    with pytest.raises(ValueError, match=r"RTX postprocess presets \(rtx-super-resolution, .*'rtx-super-res'"):
        realtime_backend._realtime_postprocess_stream(
            fps=16,
            launch_config=StudioLaunchConfig(model_id="test", device="cuda"),
        )
