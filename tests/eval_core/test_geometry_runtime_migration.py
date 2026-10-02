from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_pi3_model_loading_and_geometry_outputs_survive_migration(monkeypatch) -> None:
    calls = []

    class Model(torch.nn.Module):
        @classmethod
        def from_pretrained(cls, path, *, strict):
            calls.append((path, strict))
            return cls()

        def forward(self, images):
            assert images.shape == (1, 1, 3, 2, 2)
            assert images.device.type == "cpu"
            return {
                "points": torch.ones(1, 1, 2, 2, 3),
                "local_points": torch.full((1, 1, 2, 2, 3), 2.0),
                "camera_poses": torch.eye(4).reshape(1, 1, 4, 4),
                "conf": torch.tensor([-10.0, 10.0, 10.0, -10.0]).reshape(1, 1, 2, 2, 1),
            }

    model_module = ModuleType("worldfoundry.base_models.three_dimensions.point_clouds.pi3_inference.models.pi3")
    model_module.Pi3 = Model
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    from worldfoundry.core.io.paths import package_root

    runtime_path = package_root() / "base_models/three_dimensions/point_clouds/pi3_inference/runtime.py"
    spec = importlib.util.spec_from_file_location("pi3_runtime_under_test", runtime_path)
    runtime_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime_module)
    model = runtime_module.Pi3Representation.from_pretrained("local-model", device="cpu")

    result = model.get_representation({"images": torch.zeros(1, 1, 3, 2, 2), "conf_threshold": 0.5})

    assert calls == [("local-model", True)]
    assert result["points"].shape == (1, 1, 2, 2, 3)
    np.testing.assert_array_equal(result["depth_map"], np.full((1, 1, 2, 2), 2.0))
    np.testing.assert_array_equal(result["masks"], [[[[False, True], [True, False]]]])
    np.testing.assert_array_equal(result["camera_poses"][0, 0], np.eye(4))


def test_lingbot_runtime_locates_its_bundled_model() -> None:
    from worldfoundry.base_models.three_dimensions.point_clouds.lingbot_map import (
        runtime,
    )

    assert runtime._runtime_root() == Path(runtime.__file__).resolve().parent
    assert (runtime._runtime_root() / "lingbot_map/models/gct_stream.py").is_file()


@pytest.mark.parametrize("version", (1, 2))
def test_lyra_runtime_locates_installed_source_without_repo_marker(monkeypatch, tmp_path, version) -> None:
    from worldfoundry.base_models.three_dimensions.point_clouds.lyra import utils

    installed_package = tmp_path / "site-packages/worldfoundry"
    monkeypatch.setattr(utils, "package_root", lambda: installed_package)

    expected = installed_package / f"synthesis/visual_generation/lyra_{version}"
    if version == 1:
        expected /= "lyra1_runtime"
    assert utils.lyra_runtime_root(f"lyra{version}_runtime") == expected


def test_worldfm_finds_moge_checkpoint_in_configured_cache(monkeypatch, tmp_path) -> None:
    from worldfoundry.base_models.three_dimensions.point_clouds.worldfm import runtime

    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path))
    checkpoint = tmp_path / "Ruicheng--moge-2-vitl-normal/model.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()

    assert runtime._resolve_moge_pretrained(None) == str(checkpoint.resolve())
    assert runtime._resolve_moge_pretrained(runtime.DEFAULT_WORLDFM_MOGE2_REPO) == str(checkpoint.resolve())


def test_lyra_missing_checkpoint_path_fails_clearly(tmp_path) -> None:
    from worldfoundry.base_models.three_dimensions.point_clouds.lyra import utils

    with pytest.raises(FileNotFoundError, match="checkpoint directory not found"):
        utils.resolve_lyra1_repo_root(str(tmp_path / "missing"))


@pytest.mark.parametrize("nested", (False, True))
def test_lyra_reconstruction_loads_direct_and_assembled_checkpoint_layouts(monkeypatch, tmp_path, nested) -> None:
    from worldfoundry.base_models.three_dimensions.point_clouds.lyra import runtime_v1

    root = tmp_path / "checkpoints"
    weights = root / "Lyra" if nested else root
    weights.mkdir(parents=True)
    for name in ("lyra_static.pt", "lyra_dynamic.pt"):
        (weights / name).touch()
    monkeypatch.setattr(runtime_v1, "resolve_lyra1_repo_root", lambda path: str(tmp_path / "lyra-source"))
    monkeypatch.setattr(runtime_v1, "default_local_lyra1_checkpoint_root", lambda: root)

    model = runtime_v1.Lyra1Representation.from_pretrained("lyra-1", device="cpu")

    assert model.static_ckpt_path == str(weights / "lyra_static.pt")
    assert model.dynamic_ckpt_path == str(weights / "lyra_dynamic.pt")


def test_lyra_pipeline_passes_assembled_checkpoints_to_reconstruction(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace

    from worldfoundry.base_models.three_dimensions.point_clouds.lyra import runtime_v1
    from worldfoundry.pipelines.lyra import pipeline_lyra1

    gen3c = tmp_path / "gen3c"
    staged = tmp_path / "assembled"
    repo = tmp_path / "lyra-source"
    calls = []

    def assemble(**kwargs):
        calls.append(kwargs)
        return str(staged)

    monkeypatch.setattr(
        pipeline_lyra1.Lyra1Synthesis, "from_pretrained", lambda **kwargs: SimpleNamespace(checkpoint_dir=str(gen3c))
    )
    monkeypatch.setattr(pipeline_lyra1, "prepare_lyra1_checkpoint_root", assemble)
    monkeypatch.setattr(pipeline_lyra1, "resolve_lyra1_repo_root", lambda path: str(repo))
    monkeypatch.setattr(runtime_v1, "resolve_lyra1_repo_root", lambda path: str(repo))

    pipeline = pipeline_lyra1.Lyra1Pipeline.from_pretrained(
        required_components={"load_representation": True}, device="cpu"
    )

    assert calls == [{"checkpoint_dir": str(gen3c), "repo_root": str(repo)}]
    assert pipeline.representation_model.static_ckpt_path == str(staged / "Lyra/lyra_static.pt")
    assert pipeline.representation_model.dynamic_ckpt_path == str(staged / "Lyra/lyra_dynamic.pt")


def test_lyra_checkpoint_assembly_preserves_previous_run(monkeypatch, tmp_path) -> None:
    from worldfoundry.pipelines.lyra.checkpoints import prepare_lyra1_checkpoint_root
    from worldfoundry.synthesis.visual_generation.gen3c import runtime_env

    monkeypatch.setenv("WORLDFOUNDRY_CACHE_DIR", str(tmp_path / "cache"))
    gen3c = tmp_path / "gen3c"
    for name in (
        "Gen3C-Cosmos-7B/model.pt",
        "Cosmos-Tokenize1-CV8x8x8-720p/mean_std.pt",
        "google-t5/t5-11b/config.json",
    ):
        path = gen3c / name
        path.parent.mkdir(parents=True)
        path.touch()
    monkeypatch.setattr(runtime_env, "prepare_gen3c_checkpoint_root", lambda checkpoint_dir: str(gen3c))
    previous_weights = gen3c / "Lyra"
    previous_weights.mkdir()
    (previous_weights / "lyra_static.pt").write_text("original")

    staged = []
    for index in (1, 2):
        weights = tmp_path / f"weights-{index}"
        weights.mkdir(parents=True)
        for name in ("lyra_static.pt", "lyra_dynamic.pt"):
            (weights / name).write_text(str(index))
        staged.append(Path(prepare_lyra1_checkpoint_root(str(weights))))

    assert staged[0] != staged[1]
    assert (staged[0] / "Lyra/lyra_static.pt").read_text() == "1"
    assert (staged[1] / "Lyra/lyra_static.pt").read_text() == "2"
    assert (previous_weights / "lyra_static.pt").read_text() == "original"
    assert not (previous_weights / "lyra_dynamic.pt").exists()
    assert all((root / "Gen3C-Cosmos-7B/model.pt").is_file() for root in staged)
