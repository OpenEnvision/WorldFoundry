from __future__ import annotations

from pathlib import Path

from worldfoundry.cli.dataset import _handle_dataset_locate, _handle_dataset_plan
from worldfoundry.evaluation.tasks.datasets.manager import dataset_location_env_var


def test_dataset_locate_prefers_per_dataset_env_override(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "Howieeeee" / "WorldScore"
    dataset_dir.mkdir(parents=True)
    monkeypatch.setenv(dataset_location_env_var("Howieeeee/WorldScore"), str(dataset_dir))

    args = type(
        "Args",
        (),
        {
            "dataset_id": "Howieeeee/WorldScore",
            "data_root": None,
            "manifest": None,
            "cache_dir": tmp_path / "hf-cache",
            "json": True,
        },
    )()
    captured: list[dict] = []
    monkeypatch.setattr("worldfoundry.cli.dataset.json_dump", captured.append)

    assert _handle_dataset_locate(args) == 0
    assert captured[0]["ok"] is True
    assert captured[0]["source"] == "env"
    assert Path(str(captured[0]["path"])) == dataset_dir


def test_dataset_plan_emits_each_requested_download_once(tmp_path: Path, monkeypatch) -> None:
    args = type(
        "Args",
        (),
        {
            "dataset_id": ["acme/demo-dataset", "acme/second-dataset"],
            "cache_dir": tmp_path / "hf-cache",
            "check_local": False,
            "json": True,
        },
    )()
    captured: list[dict] = []
    monkeypatch.setattr("worldfoundry.cli.dataset.json_dump", captured.append)

    assert _handle_dataset_plan(args) == 0
    assert len(captured[0]["commands"]) == 2
    assert captured[0]["metadata"]["hf_dataset_ids"] == [
        "acme/demo-dataset",
        "acme/second-dataset",
    ]
