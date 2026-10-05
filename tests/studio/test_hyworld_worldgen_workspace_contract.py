from pathlib import Path

from worldfoundry.studio.inference.catalog import find_entry


def test_hyworld_worldgen_workspace_reports_materialized_scene_requirement() -> None:
    entry = find_entry("hyworld-worldgen")
    scene = Path(entry.default_load_kwargs["scene_path"])

    assert entry.default_model_ref == str(scene)
    assert not (scene / "render_results" / "global_pcd.ply").is_file()
    assert Path(entry.default_input_path).is_file()
    assert Path(entry.default_call_kwargs["camera_json"]).is_file()
