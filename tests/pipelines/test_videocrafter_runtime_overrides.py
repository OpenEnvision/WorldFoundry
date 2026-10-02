from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.pipelines.videocrafter.base import _CALL_RUNTIME_ALIASES
from worldfoundry.synthesis.visual_generation.videocrafter.videocrafter1_i2v_synthesis import (
    VideoCrafter1I2VSynthesis,
)
from worldfoundry.synthesis.visual_generation.videocrafter.videocrafter1_t2v_synthesis import (
    VideoCrafter1T2VSynthesis,
)
from worldfoundry.synthesis.visual_generation.videocrafter.videocrafter2_t2v_synthesis import (
    VideoCrafter2T2VSynthesis,
)


def test_videocrafter_call_guidance_aliases_target_native_cfg() -> None:
    assert _CALL_RUNTIME_ALIASES["guidance_scale"] == "unconditional_guidance_scale"
    assert _CALL_RUNTIME_ALIASES["cfg_scale"] == "unconditional_guidance_scale"


@pytest.mark.parametrize(
    ("synthesis_cls", "repo_dir"),
    (
        (VideoCrafter1I2VSynthesis, "VideoCrafter--Image2Video-512"),
        (VideoCrafter1T2VSynthesis, "VideoCrafter--Text2Video-1024"),
        (VideoCrafter2T2VSynthesis, "VideoCrafter--VideoCrafter2"),
    ),
)
def test_videocrafter_runtime_defaults_use_portable_hfd_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthesis_cls,
    repo_dir: str,
) -> None:
    hfd_root = tmp_path / "hfd"
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(hfd_root))

    runtime_kwargs = synthesis_cls.build_runtime_kwargs()

    assert runtime_kwargs["ckpt_path"] == str(hfd_root / repo_dir / "model.ckpt")
