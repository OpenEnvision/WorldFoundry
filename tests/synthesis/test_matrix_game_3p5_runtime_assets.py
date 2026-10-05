from __future__ import annotations

from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_3p5_runtime.runtime import (
    REQUIRED_DATA_FILES,
    REQUIRED_RUNTIME_FILES,
    REQUIRED_SHARED_FILES,
    RUNTIME_ROOT,
)


def test_matrix_game_3p5_required_runtime_files_are_packaged() -> None:
    missing = [
        str(path)
        for path in (
            *(RUNTIME_ROOT / relative for relative in REQUIRED_RUNTIME_FILES),
            *REQUIRED_DATA_FILES,
            *REQUIRED_SHARED_FILES,
        )
        if not path.is_file()
    ]

    assert missing == []
