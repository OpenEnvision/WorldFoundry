from __future__ import annotations

from pathlib import Path

from worldfoundry.synthesis.visual_generation.lyra_2 import runtime as lyra2_runtime


def test_checkpoint_aliases_resolve_shared_local_umt5_after_symlink_exists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_root = tmp_path / "Lyra-2.0"
    checkpoint_dir = checkpoint_root / "checkpoints"
    (checkpoint_dir / "text_encoder").mkdir(parents=True)
    (checkpoint_dir / "text_encoder" / "encoder.pth").touch()

    shared_checkpoint_root = tmp_path / "shared-checkpoints"
    tokenizer = shared_checkpoint_root / "google--umt5-xxl"
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tokenizer / "spiece.model").touch()

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    monkeypatch.setattr(
        lyra2_runtime.Lyra2Runtime,
        "_runtime_root",
        staticmethod(lambda: runtime_root),
    )
    monkeypatch.setattr(
        lyra2_runtime,
        "checkpoint_root_candidates",
        lambda: (shared_checkpoint_root,),
    )
    monkeypatch.delenv("LYRA2_UMT5_TOKENIZER", raising=False)

    instance = object.__new__(lyra2_runtime.Lyra2Runtime)
    instance._install_checkpoint_aliases(checkpoint_root)

    assert (runtime_root / "checkpoints").resolve() == checkpoint_dir.resolve()
    assert lyra2_runtime.os.environ["LYRA2_UMT5_TOKENIZER"] == tokenizer.as_posix()

    monkeypatch.delenv("LYRA2_UMT5_TOKENIZER")
    instance._install_checkpoint_aliases(checkpoint_root)
    assert lyra2_runtime.os.environ["LYRA2_UMT5_TOKENIZER"] == tokenizer.as_posix()
