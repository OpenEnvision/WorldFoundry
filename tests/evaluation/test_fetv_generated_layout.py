from __future__ import annotations

import json

import pytest
from PIL import Image

from worldfoundry.core.io.serialization import read_jsonl_objects, write_jsonl
from worldfoundry.evaluation.tasks.execution.runners.fetv import fetv_official_runtime
from worldfoundry.evaluation.tasks.execution.runners.fetv.fetv_official_runtime import (
    materialize_bounded_fetv_prompt_file,
    run_official_fetv_runtime,
    validate_fetv_clip_frame_layout,
)
from worldfoundry.evaluation.tasks.execution.runners.fetv.fetv_prompts import (
    FETV_GENERATION_MANIFEST_NAME,
)


def _write_prompts(path, count: int) -> list[dict[str, str]]:
    rows = [{"video_id": f"video-{index}", "prompt": f"prompt {index}"} for index in range(count)]
    write_jsonl(path, rows)
    return rows


def _write_frame_dir(root, index: int) -> None:
    frame_dir = root / f"sent{index}_frames"
    frame_dir.mkdir()
    for frame_index in range(16):
        Image.new("RGB", (8, 8), (frame_index, index, 0)).save(frame_dir / f"frame{frame_index}.jpg")


def _write_manifest(root, prompt_rows, count: int) -> None:
    write_jsonl(
        root / FETV_GENERATION_MANIFEST_NAME,
        [
            {
                "sample_id": f"sample-{index}",
                "sent_index": index,
                "frame_dir_name": f"sent{index}_frames",
                "frame_count": 16,
                "prompt_record": prompt_rows[index],
                "status": "decoded",
            }
            for index in range(count)
        ],
    )


def test_rejects_flat_mp4s_without_a_proven_mapping_manifest(tmp_path) -> None:
    prompt_file = tmp_path / "fetv_data.json"
    _write_prompts(prompt_file, 2)
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "sample-0000__generated_video.mp4").write_bytes(b"not decoded")

    with pytest.raises(ValueError, match="will not guess"):
        validate_fetv_clip_frame_layout(
            generated_video_dir=generated,
            prompt_file=prompt_file,
            metrics=("clip_score",),
            limit=1,
        )


def test_accepts_manifest_proven_bounded_official_layout(tmp_path) -> None:
    prompt_file = tmp_path / "fetv_data.json"
    prompt_rows = _write_prompts(prompt_file, 20)
    generated = tmp_path / "generated"
    generated.mkdir()
    _write_frame_dir(generated, 0)
    _write_manifest(generated, prompt_rows, 1)

    report = validate_fetv_clip_frame_layout(
        generated_video_dir=generated,
        prompt_file=prompt_file,
        metrics=("clip_score", "blip_score"),
        limit=1,
    )

    assert report["mapping"] == "prompt_jsonl_zero_based_line_index"
    assert report["required_frame_directories"] == 1
    assert report["manifest_record_count"] == 1
    assert report["bounded"] is True

    bounded = materialize_bounded_fetv_prompt_file(
        prompt_file=prompt_file,
        output_dir=tmp_path / "out",
        prompt_count=1,
    )
    assert json.loads(bounded.read_text(encoding="utf-8")) == prompt_rows[0]


def test_rejects_duplicate_manifest_sent_indices(tmp_path) -> None:
    prompt_file = tmp_path / "fetv_data.json"
    prompt_rows = _write_prompts(prompt_file, 2)
    generated = tmp_path / "generated"
    generated.mkdir()
    _write_frame_dir(generated, 0)
    row = {
        "sent_index": 0,
        "frame_dir_name": "sent0_frames",
        "frame_count": 16,
        "prompt_record": prompt_rows[0],
        "status": "decoded",
    }
    write_jsonl(
        generated / FETV_GENERATION_MANIFEST_NAME,
        [{**row, "sample_id": "a"}, {**row, "sample_id": "b"}],
    )

    with pytest.raises(ValueError, match="duplicate FETV generation manifest sent_index"):
        validate_fetv_clip_frame_layout(
            generated_video_dir=generated,
            prompt_file=prompt_file,
            metrics=("clip_score",),
            limit=1,
        )


def test_rejects_manifest_prompt_mismatch(tmp_path) -> None:
    prompt_file = tmp_path / "fetv_data.json"
    prompt_rows = _write_prompts(prompt_file, 1)
    generated = tmp_path / "generated"
    generated.mkdir()
    _write_frame_dir(generated, 0)
    _write_manifest(generated, [{**prompt_rows[0], "prompt": "wrong"}], 1)

    with pytest.raises(ValueError, match="does not match the official prompt JSONL"):
        validate_fetv_clip_frame_layout(
            generated_video_dir=generated,
            prompt_file=prompt_file,
            metrics=("clip_score",),
            limit=1,
        )


def test_bounded_runtime_launches_official_evaluator_with_exact_prompt_prefix(tmp_path, monkeypatch) -> None:
    prompt_file = tmp_path / "fetv_data.json"
    prompt_rows = _write_prompts(prompt_file, 3)
    generated = tmp_path / "generated"
    generated.mkdir()
    _write_frame_dir(generated, 0)
    _write_manifest(generated, prompt_rows, 1)
    observed_command = []

    def fake_run_command(command, *, cwd, env, output_dir, name, timeout_seconds):
        del cwd, env, name, timeout_seconds
        observed_command.extend(command)
        result_root = output_dir / "auto_eval_results" / "CLIPScore"
        result_root.mkdir(parents=True)
        (result_root / "auto_eval_results_custom-model.json").write_text('{"0": 0.5}', encoding="utf-8")
        return {"returncode": 0, "command": command}

    monkeypatch.setattr(fetv_official_runtime, "_run_command", fake_run_command)
    clip_checkpoint = tmp_path / "ViT-B-32.pt"
    clip_checkpoint.write_bytes(b"test checkpoint path")
    summary = run_official_fetv_runtime(
        generated_video_dir=generated,
        output_dir=tmp_path / "out",
        prompt_file=prompt_file,
        model_name="custom-model",
        metrics=("clip_score",),
        clip_model=str(clip_checkpoint),
        limit=1,
    )

    prompt_argument = observed_command[observed_command.index("--prompt_file") + 1]
    assert prompt_argument.endswith("fetv_data_first_1.jsonl")
    assert read_jsonl_objects(prompt_argument) == [prompt_rows[0]]
    assert observed_command[observed_command.index("--limit") + 1] == "1"
    assert summary["input_layout"]["bounded"] is True
    assert summary["results_path"].endswith("fetv_eval_results_custom-model.csv")


def test_clip_alias_resolves_through_shared_base_model_assets(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "ViT-B-32.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(fetv_official_runtime, "vbench_asset_path", lambda asset_id: checkpoint)

    assert fetv_official_runtime.resolve_clip_checkpoint("ViT-B/32") == str(checkpoint)
