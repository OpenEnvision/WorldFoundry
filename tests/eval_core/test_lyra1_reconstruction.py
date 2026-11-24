from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("mode", "model_config"),
    (
        ("static", "3dgs_res_704_1280_views_121_multi_6_prune.yaml"),
        ("dynamic", "3dgs_res_704_1280_views_121_multi_6_dynamic_prune.yaml"),
    ),
)
def test_lyra1_packaged_demo_configs_resolve_to_the_model_definition(mode, model_config) -> None:
    omegaconf = pytest.importorskip("omegaconf")
    from worldfoundry.base_models.three_dimensions.point_clouds.lyra import runtime_v1

    root = runtime_v1.LYRA1_CONFIG_ROOT
    demo = omegaconf.OmegaConf.merge(
        omegaconf.OmegaConf.load(root / "inference" / "default.yaml"),
        omegaconf.OmegaConf.load(root / "demo" / f"lyra_{mode}.yaml"),
    )
    model_paths = runtime_v1.Lyra1Representation._resolve_model_config_paths(demo.config_path, None)

    assert model_paths == [str(root / "model" / "default.yaml"), str(root / "model" / model_config)]
    model = omegaconf.OmegaConf.merge(*(omegaconf.OmegaConf.load(path) for path in model_paths))
    assert model.vae_backbone == "cosmos1"
    assert list(model.img_size) == [704, 1280]
    assert model.num_input_multi_views == 6
    assert bool(model.get("use_time_embedding")) is (mode == "dynamic")


def test_lyra1_reconstruction_runs_sample_with_packaged_configs(monkeypatch, tmp_path) -> None:
    omegaconf = pytest.importorskip("omegaconf")
    from worldfoundry.base_models.three_dimensions.point_clouds.lyra import runtime_v1

    checkpoints = tmp_path / "checkpoints"
    (checkpoints / "Lyra").mkdir(parents=True)
    for name in ("lyra_static.pt", "lyra_dynamic.pt"):
        (checkpoints / "Lyra" / name).touch()
    generated_root = tmp_path / "generated"
    generated_root.mkdir()
    registry: dict[str, dict[str, Any]] = {
        "lyra_static_template_generated": {"cls": object, "kwargs": {"root_path": ""}}
    }
    seen = {}

    def run_sample(config) -> None:
        seen["config"] = config
        seen["registered"] = dict(registry[config.dataset_name]["kwargs"])
        seen["cwd"] = Path.cwd()

    def load_and_merge_configs(paths):
        return omegaconf.OmegaConf.merge(*(omegaconf.OmegaConf.load(path) for path in paths))

    modules = {
        "worldfoundry.synthesis.visual_generation.lyra_1.lyra1_runtime.sample": SimpleNamespace(main=run_sample),
        "worldfoundry.synthesis.visual_generation.lyra_1.lyra1_runtime.src.models.eval_inputs.registry": SimpleNamespace(
            dataset_registry=registry
        ),
        "worldfoundry.synthesis.visual_generation.lyra_1.lyra1_runtime.src.models.utils.misc": SimpleNamespace(
            load_and_merge_configs=load_and_merge_configs
        ),
    }
    monkeypatch.setattr(runtime_v1, "resolve_lyra1_repo_root", lambda path: str(tmp_path))
    monkeypatch.setattr(runtime_v1.importlib, "import_module", modules.__getitem__)

    model = runtime_v1.Lyra1Representation.from_pretrained("lyra-1", device="cpu", checkpoint_dir=str(checkpoints))
    paths = list(sys.path)
    previous_cwd = Path.cwd()
    result = model.get_representation(
        {"generated_root": str(generated_root), "mode": "static", "output_dir": str(tmp_path / "out")}
    )

    assert sys.path == paths
    assert Path.cwd() == previous_cwd
    config = seen["config"]
    root = runtime_v1.LYRA1_CONFIG_ROOT
    assert config.ckpt_path == str(checkpoints / "Lyra" / "lyra_static.pt")
    assert config.vae_path == str(checkpoints / "Cosmos-Tokenize1-CV8x8x8-720p")
    assert list(config.config_path) == [
        str(root / "model" / "default.yaml"),
        str(root / "model" / "3dgs_res_704_1280_views_121_multi_6_prune.yaml"),
    ]
    assert seen["registered"]["root_path"] == str(generated_root)
    assert seen["cwd"] == tmp_path
    assert config.dataset_name not in registry
    render_root = tmp_path / "out" / "static_view_indices_fixed_5_0_1_2_3_4" / config.dataset_name
    assert result["render_roots"] == [str(render_root)]
    assert result["main_render_dir"] == str(render_root / "main_gaussians_renderings")


@pytest.mark.parametrize(
    "dirname",
    ["Lyra", "lyra-1", "Lyra-1", "nvidia--Lyra", "checkpoints/lyra-1"],
)
def test_lyra1_checkpoint_alias_resolution_accepts_matrix_layouts(monkeypatch, tmp_path: Path, dirname: str) -> None:
    from worldfoundry.base_models.three_dimensions.point_clouds.lyra import utils

    weights_dir = tmp_path / dirname
    weights_dir.mkdir(parents=True)
    (weights_dir / "lyra_static.pt").touch()
    (weights_dir / "lyra_dynamic.pt").touch()

    runtime_dir = tmp_path / "vendored_runtime"
    runtime_dir.mkdir()
    monkeypatch.setattr(utils, "default_local_lyra1_repo", lambda: runtime_dir)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path))

    assert utils._candidate_lyra1_weights_subdir(tmp_path) == weights_dir.resolve()
    assert utils.resolve_lyra1_repo_root(str(tmp_path)) == str(runtime_dir)
    if dirname in {"Lyra", "lyra-1", "nvidia--Lyra"}:
        assert utils.default_local_lyra1_checkpoint_root() == weights_dir.resolve()


def test_lyra1_pipeline_filters_representation_options_from_synthesis(tmp_path: Path) -> None:
    from worldfoundry.pipelines.lyra.pipeline_lyra1 import Lyra1Pipeline

    seen: dict[str, Any] = {}

    class FakeOperator:
        def process_perception(self, *, images=None, videos=None, mode=None):
            del videos, mode
            return images

        def get_interaction(self, interactions):
            seen["interactions"] = interactions

        def process_interaction(self, *, prompt: str = ""):
            return {
                "actions": [prompt],
                "mapped_trajectories": ["zoom_in"],
                "trajectory": "zoom_in",
            }

        def delete_last_interaction(self) -> None:
            return None

    class FakeSynthesis:
        def predict(self, **kwargs):
            seen["synthesis_kwargs"] = dict(kwargs)
            return {
                "status": "completed",
                "generated_root": str(tmp_path / "generated"),
                "generated_video_path": str(tmp_path / "generated" / "out.mp4"),
                "video": ["frame0"],
                "fps": 24,
            }

    class FakeRepresentation:
        def get_representation(self, payload):
            seen["representation_input"] = dict(payload)
            return {"gaussian_ply_paths": [str(tmp_path / "gaussians_0.ply")]}

    pipeline = Lyra1Pipeline(
        synthesis_model=cast(Any, FakeSynthesis()),
        representation_model=cast(Any, FakeRepresentation()),
        operator=cast(Any, FakeOperator()),
        device="cpu",
    )
    result = pipeline(
        images="ref.png",
        prompt="slow camera move",
        mode="static",
        trajectory="zoom_in",
        reconstruct_3d=True,
        output_dir=str(tmp_path / "lyra1"),
        return_dict=True,
        execute=True,
        save_gaussians_orig=True,
    )

    assert "save_gaussians_orig" not in seen["synthesis_kwargs"]
    assert seen["synthesis_kwargs"]["execute"] is True
    assert seen["synthesis_kwargs"]["multi_trajectory"] is True
    assert seen["representation_input"]["save_gaussians_orig"] is True
    assert result["reconstruction"]["gaussian_ply_paths"] == [str(tmp_path / "gaussians_0.ply")]


def test_lyra1_rendering_modules_avoid_kiui_and_inverse_sigmoid_matches_logit(monkeypatch) -> None:
    import ast
    import importlib
    import importlib.util
    from types import ModuleType

    torch = pytest.importorskip("torch")
    from worldfoundry.base_models.three_dimensions.point_clouds.lyra.utils import lyra_runtime_root

    lyra1_root = lyra_runtime_root("lyra1_runtime")
    for rel_path in (
        "src/models/utils/render.py",
        "src/rendering/gs.py",
        "src/rendering/gs_deferred.py",
    ):
        tree = ast.parse((lyra1_root / rel_path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("kiui") for alias in node.names), rel_path
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith("kiui"), rel_path

    if importlib.util.find_spec("Imath") is None:
        data_mod = ModuleType("worldfoundry.synthesis.visual_generation.lyra_1.lyra1_runtime.src.models.utils.data")
        setattr(data_mod, "ray_condition", lambda *args, **kwargs: None)
        monkeypatch.setitem(sys.modules, data_mod.__name__, data_mod)

    monkeypatch.setitem(sys.modules, "kiui", None)
    monkeypatch.setitem(sys.modules, "kiui.op", None)
    render_mod = importlib.import_module(
        "worldfoundry.synthesis.visual_generation.lyra_1.lyra1_runtime.src.models.utils.render"
    )
    render_mod = importlib.reload(render_mod)

    interior = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float32)
    boundary = torch.tensor([0.0, 1.0], dtype=torch.float32)
    expected_interior = torch.log(interior.clamp(1e-6, 1 - 1e-6) / (1 - interior.clamp(1e-6, 1 - 1e-6)))
    out_interior = render_mod.inverse_sigmoid(interior)
    out_boundary = render_mod.inverse_sigmoid(boundary)

    assert torch.allclose(out_interior, expected_interior, atol=1e-6, rtol=1e-6)
    assert torch.allclose(torch.sigmoid(out_interior), interior, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(out_boundary).all()


@pytest.mark.parametrize(
    ("mode", "expected_target_index_manual"),
    (
        ("static", None),
        ("dynamic", [0, 60, 120]),
    ),
)
def test_lyra1_inference_defaults_cover_sample_config_keys(mode, expected_target_index_manual) -> None:
    import ast

    omegaconf = pytest.importorskip("omegaconf")
    from worldfoundry.base_models.three_dimensions.point_clouds.lyra import runtime_v1
    from worldfoundry.base_models.three_dimensions.point_clouds.lyra.utils import lyra_runtime_root

    merged = omegaconf.OmegaConf.merge(
        omegaconf.OmegaConf.load(runtime_v1.LYRA1_CONFIG_ROOT / "inference" / "default.yaml"),
        omegaconf.OmegaConf.load(runtime_v1.LYRA1_CONFIG_ROOT / "demo" / f"lyra_{mode}.yaml"),
    )
    assert merged.target_index_manual == expected_target_index_manual

    sample_tree = ast.parse((lyra_runtime_root("lyra1_runtime") / "sample.py").read_text(encoding="utf-8"))
    referenced_keys = {
        node.attr
        for node in ast.walk(sample_tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "config"
        and node.attr != "get"
    }
    for key in referenced_keys:
        if key in {"vae_backbone", "vae_path"}:
            # vae_path is optionally injected by runtime_v1.py and vae_backbone is not used by sample.py.
            continue
        assert key in merged, f"{key} missing from merged {mode} inference config"


def test_lyra1_save_reconstruction_inputs_writes_weights_only_tensor_latent(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from worldfoundry.synthesis.visual_generation.lyra_1.worldfoundry_runtime import Lyra1Runtime

    expected = torch.arange(32, dtype=torch.float32).reshape(1, 2, 1, 4, 4)
    Lyra1Runtime._save_reconstruction_inputs(
        {"latents": expected},
        output_root=tmp_path,
        clip_name="lyra1",
    )

    loaded = torch.load(tmp_path / "latent" / "lyra1.pkl", map_location="cpu", weights_only=True)
    assert isinstance(loaded, torch.Tensor)
    assert torch.equal(loaded, expected.permute(0, 2, 1, 3, 4).contiguous())


def test_lyra1_radym_load_rgb_latents_accepts_numpy_and_tensor(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    pytest.importorskip("decord")
    from worldfoundry.synthesis.visual_generation.lyra_1.lyra1_runtime.src.models.eval_inputs.radym import (
        _load_rgb_latents,
    )

    arr = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 2, 2)
    npy_pkl = tmp_path / "legacy_numpy.pkl"
    tensor_pkl = tmp_path / "pure_tensor.pkl"
    torch.save(arr, npy_pkl)
    torch.save(torch.from_numpy(arr.copy()), tensor_pkl)

    loaded_from_npy = _load_rgb_latents(npy_pkl)
    loaded_from_tensor = _load_rgb_latents(tensor_pkl)
    assert isinstance(loaded_from_npy, torch.Tensor)
    assert isinstance(loaded_from_tensor, torch.Tensor)
    assert torch.equal(loaded_from_npy, torch.from_numpy(arr))
    assert torch.equal(loaded_from_tensor, torch.from_numpy(arr))


def test_lyra1_provider_getitem_stops_immediately_on_eval_failure() -> None:
    pytest.importorskip("decord")
    from worldfoundry.synthesis.visual_generation.lyra_1.lyra1_runtime.src.models.eval_inputs.provider import (
        Provider,
    )

    calls = 0

    class FailingEvalDataset:
        training = False
        dataset = [0]

        def get_item(self, idx: int) -> Any:
            nonlocal calls
            calls += 1
            raise RuntimeError(f"boom {idx}")

    with pytest.raises(RuntimeError, match="boom 0"):
        Provider.__getitem__(cast(Any, FailingEvalDataset()), 0)
    assert calls == 1


@pytest.mark.parametrize("with_eval3d", [True, False])
def test_lyra1_rasterize_rgbd_keeps_3_channels_for_eval3d(monkeypatch, with_eval3d: bool) -> None:
    torch = pytest.importorskip("torch")
    from worldfoundry.synthesis.visual_generation.lyra_1.lyra1_runtime.src.rendering import gs_deferred_patch

    seen: list[dict[str, Any]] = []
    rgb_out = torch.full((1, 2, 2, 3), 0.5)
    alpha_out = torch.full((1, 2, 2, 1), 0.25)
    depths_meta = torch.tensor([[3.0, 5.0]], dtype=torch.float32)

    def fake_rasterization(**kwargs):
        seen.append(dict(kwargs))
        assert kwargs.get("with_eval3d") is with_eval3d
        if kwargs["render_mode"] == "RGB+ED":
            return torch.full((1, 2, 2, 4), 1.0), alpha_out, {"depths": depths_meta}
        assert kwargs["colors"].shape[-1] == 3
        if len(seen) == 1:
            return rgb_out, alpha_out, {"depths": depths_meta}
        depth_accum = torch.zeros((1, 2, 2, 3))
        depth_accum[..., 0] = 2.0
        return depth_accum, alpha_out, {}

    monkeypatch.setattr(gs_deferred_patch, "rasterization", fake_rasterization)

    rgbd, alpha, meta = gs_deferred_patch.rasterize_rgbd(
        means=torch.zeros((2, 3)),
        quats=torch.zeros((2, 4)),
        scales=torch.ones((2, 3)),
        opacities=torch.ones((2,)),
        colors=torch.zeros((2, 3)),
        viewmats=torch.eye(4)[None],
        Ks=torch.eye(3)[None],
        width=2,
        height=2,
        near_plane=0.1,
        far_plane=100.0,
        backgrounds=torch.ones((1, 3)),
        raster_kwargs={"with_ut": with_eval3d, "with_eval3d": with_eval3d, "packed": False},
    )

    assert alpha is alpha_out
    assert meta["depths"] is depths_meta
    if with_eval3d:
        assert [call["render_mode"] for call in seen] == ["RGB", "RGB"]
        assert torch.allclose(seen[1]["backgrounds"], torch.zeros((1, 3)))
        assert torch.equal(seen[1]["colors"][..., 0], depths_meta)
        assert torch.equal(rgbd[..., :3], rgb_out)
        assert torch.allclose(rgbd[..., 3:], torch.full((1, 2, 2, 1), 8.0))
    else:
        assert [call["render_mode"] for call in seen] == ["RGB+ED"]


def test_lyra1_gs_deferred_patch_imports_without_gsplat(monkeypatch) -> None:
    import importlib

    torch = pytest.importorskip("torch")
    module_name = "worldfoundry.synthesis.visual_generation.lyra_1.lyra1_runtime.src.rendering.gs_deferred_patch"
    had_module = module_name in sys.modules
    monkeypatch.setitem(sys.modules, "gsplat", None)
    monkeypatch.setitem(sys.modules, "gsplat.rendering", None)
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    call_kwargs = {
        "means": torch.zeros((2, 3)),
        "quats": torch.zeros((2, 4)),
        "scales": torch.ones((2, 3)),
        "opacities": torch.ones((2,)),
        "colors": torch.zeros((2, 3)),
        "viewmats": torch.eye(4)[None],
        "Ks": torch.eye(3)[None],
        "width": 2,
        "height": 2,
        "near_plane": 0.1,
        "far_plane": 100.0,
        "backgrounds": torch.ones((1, 3)),
        "raster_kwargs": {"with_eval3d": False},
    }

    try:
        gs_deferred_patch = importlib.import_module(module_name)
        assert gs_deferred_patch.rasterization is None

        with pytest.raises(ModuleNotFoundError):
            gs_deferred_patch.rasterize_rgbd(**call_kwargs)

        expected = (torch.full((1, 2, 2, 4), 1.0), torch.full((1, 2, 2, 1), 0.5), {"depths": torch.ones((1, 2))})
        seen: list[dict[str, Any]] = []

        def stub_rasterization(**kwargs):
            seen.append(dict(kwargs))
            return expected

        monkeypatch.setattr(gs_deferred_patch, "rasterization", stub_rasterization)
        assert gs_deferred_patch.rasterize_rgbd(**call_kwargs) == expected
        assert len(seen) == 1
    finally:
        if not had_module:
            sys.modules.pop(module_name, None)


def test_lyra1_denormalize_latents_preserves_btchw_layout_and_applies_ct_stats() -> None:
    torch = pytest.importorskip("torch")
    from worldfoundry.synthesis.visual_generation.lyra_1.lyra1_runtime.src.models.utils.cosmos_1_tokenizer import (
        denormalize_latents,
    )

    batch, views, frames, channels, height, width = 2, 2, 3, 5, 2, 2
    sigma_data = 0.5
    raw_bvtchw = torch.arange(
        batch * views * frames * channels * height * width,
        dtype=torch.float32,
    ).reshape(batch, views * frames, channels, height, width)
    latent_std = torch.arange(1, channels * frames + 1, dtype=torch.float32).reshape(1, channels, frames, 1, 1)
    latent_mean = -torch.arange(channels * frames, dtype=torch.float32).reshape(1, channels, frames, 1, 1)

    out_5d = denormalize_latents(
        raw_bvtchw,
        latent_std,
        latent_mean,
        num_input_multi_views=views,
        sigma_data=sigma_data,
    )
    assert out_5d.shape == (batch, views * frames, channels, height, width)

    for b_idx in range(batch):
        for v_idx in range(views):
            for t_idx in range(frames):
                for c_idx in range(channels):
                    actual = out_5d[b_idx, v_idx * frames + t_idx, c_idx]
                    expected = (raw_bvtchw[b_idx, v_idx * frames + t_idx, c_idx] / sigma_data) * latent_std[
                        0, c_idx, t_idx, 0, 0
                    ] + latent_mean[0, c_idx, t_idx, 0, 0]
                    assert torch.allclose(actual, expected)

    out_4d = denormalize_latents(
        raw_bvtchw[0],
        latent_std,
        latent_mean,
        num_input_multi_views=views,
        sigma_data=sigma_data,
    )
    assert out_4d.shape == (views * frames, channels, height, width)
    assert torch.allclose(out_4d, out_5d[0])
