"""The CPU inference path must work without a Studio frontend or viewer."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from importlib.resources import files

import pytest


def test_inference_materialization_and_unload_do_not_import_viewers(tmp_path):
    script = textwrap.dedent("""
        import importlib
        import importlib.abc
        import json
        import sys
        from pathlib import Path

        forbidden = (
            "worldfoundry.studio.ui", "worldfoundry.studio.serving",
            "worldfoundry.studio.visualization", "gradio", "fastapi",
            "uvicorn", "viser", "rerun", "trimesh",
        )
        class BlockViewers(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
                    raise AssertionError("inference imported a viewer: " + fullname)

        sys.meta_path.insert(0, BlockViewers())
        for name in ("catalog", "config", "paths", "variants", "execution", "dispatch"):
            importlib.import_module("worldfoundry.studio.inference." + name)
        importlib.import_module("worldfoundry.studio.runtime_job")
        from worldfoundry.studio.inference.catalog import CatalogEntry
        from worldfoundry.studio.inference.execution import PipelineContext, PreparedInputs, StudioManager

        root = Path(sys.argv[1])
        output = root / "run"
        output.mkdir()
        cloud = output / "cloud.xyz"
        cloud.write_text("0 0 0\\n")
        entry = CatalogEntry(
            model_id="boundary-test", display_name="Boundary Test", module_path="test.fake",
            class_name="Fake", family="geometry", category="3D Generation", summary="CPU fixture",
        )
        context = PipelineContext(entry=entry, pipeline=object(), cache_key="test", backend="auto",
                                  model_ref="", endpoint="", load_kwargs={}, device="cpu")
        request = PreparedInputs(
            prompt="", input_path="", image=None, image_path=None, video_path=None,
            last_frame=None, last_frame_path=None, reference_images=[], reference_image_paths=[],
            interactions=None, camera_view=None, task_type="", intrinsics=None, meta_path="",
            panorama_path="", scene_name="", fps=1, num_frames=1, output_dir=str(output),
            output_path=str(output / "output.mp4"), call_kwargs={}, load_kwargs={},
            model_ref="", backend="auto", endpoint="", api_key="", device="cpu",
        )
        manager = StudioManager(workspace_root=str(root / "studio"))
        record = manager.materialize_run(context, request, result={"point_cloud_path": str(cloud)}, mode="run")
        manager.unload()
        manager.close()
        assert "studio_viewports" not in record.metadata
        assert json.loads(Path(record.manifest_path).read_text())["model_id"] == entry.model_id
        assert not any(any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
                       for name in sys.modules)
    """)
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_workspace_page_is_a_packaged_asset():
    from worldfoundry.studio.ui.workspace import WORKSPACE_HTML

    packaged = files("worldfoundry.studio").joinpath("ui", "workspace.html").read_text(encoding="utf-8")
    assert packaged == WORKSPACE_HTML
    assert "<html" in packaged.lower()


@pytest.mark.parametrize("module", ["worldfoundry.studio.cli", "worldfoundry.studio.runtime_job",
                                    "worldfoundry.studio.workspace_job"])
def test_studio_entrypoint_help_works_after_reorganization(module):
    result = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()
    assert "native-world" not in result.stdout
