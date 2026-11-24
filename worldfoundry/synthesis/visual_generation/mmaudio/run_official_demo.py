from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _patch_torchaudio_save() -> None:
    import soundfile as sf
    import torchaudio

    def _save_with_soundfile(
        uri: str | Path,
        src: Any,
        sample_rate: int,
        channels_first: bool = True,
        format: str | None = None,
        **_: Any,
    ) -> None:
        tensor = src.detach().cpu() if hasattr(src, "detach") else src
        data = tensor.numpy() if hasattr(tensor, "numpy") else tensor
        if channels_first and getattr(data, "ndim", 0) == 2:
            data = data.T
        sf.write(str(uri), data, int(sample_rate), format=format)

    torchaudio.save = _save_with_soundfile  # type: ignore[assignment]


def main(argv: list[str] | None = None) -> int:
    _patch_torchaudio_save()
    from .official_demo import main as run_demo

    run_demo(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
