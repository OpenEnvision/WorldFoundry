from __future__ import annotations

from worldfoundry.cli.model_run import load_model_run_schema
from worldfoundry.synthesis.visual_generation.framepack.worldfoundry_runner import (
    DEFAULT_RUNTIME_ROOT,
    _patched_script,
)


def test_framepack_in_tree_runner_is_exposed_as_runnable() -> None:
    load_model_run_schema.cache_clear()

    schema = load_model_run_schema("framepack")

    assert schema.integration_status == "integrated"
    assert schema.runner_entry_kind == "runnable_runner"
    assert schema.runnable is True
    assert schema.blocked_reason == ""
    assert schema.runner_target == "worldfoundry.evaluation.models.runners.pipeline:WorldFoundryPipelineRunner"


def test_framepack_local_hunyuan_tokenizer_bypasses_invalid_root_config(tmp_path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    hunyuan_root = tmp_path / "hunyuan"
    (hunyuan_root / "tokenizer").mkdir(parents=True)

    script = _patched_script(
        DEFAULT_RUNTIME_ROOT,
        output_dir,
        checkpoint_path,
        hf_home,
        str(hunyuan_root),
        None,
    )

    text = script.read_text(encoding="utf-8")
    expected = f"LlamaTokenizerFast.from_pretrained({str(hunyuan_root / 'tokenizer')!r})"
    assert expected in text
