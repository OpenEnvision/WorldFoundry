# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Weight-free contracts for the in-tree UniWorld-View inference runtime."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from worldfoundry.pipelines.uniworld_view.pipeline_uniworld_view import INFERENCE_MODULE, UniWorldViewPipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PACKAGE = "worldfoundry.synthesis.visual_generation.uniworld_view.uniworld_view_runtime"
RUNTIME = REPO_ROOT.joinpath(*RUNTIME_PACKAGE.split("."))


def _targets(node: object) -> list[str]:
    if isinstance(node, dict):
        return [target for key, value in node.items() for target in ([value] if key == "_target_" else _targets(value))]
    if isinstance(node, list):
        return [target for item in node for target in _targets(item)]
    return []


def test_inference_entrypoint_is_a_vendored_package_module() -> None:
    spec = importlib.util.find_spec(INFERENCE_MODULE)

    assert f"{RUNTIME_PACKAGE}.inference" == INFERENCE_MODULE
    assert spec is not None
    assert spec.origin is not None
    assert Path(spec.origin) == RUNTIME / "inference.py"


def test_sam2_config_resolves_to_canonical_modules() -> None:
    config_path = REPO_ROOT / "worldfoundry/data/models/runtime/configs/sam2/configs/sam2/sam2_hiera_l.yaml"
    targets = _targets(yaml.safe_load(config_path.read_text()))

    assert targets
    for target in targets:
        assert target.startswith("sam2.")
        # The canonical builder qualifies the official config's targets before Hydra instantiation.
        module, _, name = target.replace(
            "sam2.", "worldfoundry.base_models.perception_core.segment.sam2.", 1
        ).rpartition(".")
        source = REPO_ROOT.joinpath(*module.split(".")).with_suffix(".py")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imports = {
            alias.asname or alias.name: (node.module, alias.name)
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        if name in imports:
            owner, name = imports[name]
            assert owner is not None and owner.startswith("worldfoundry.")
            source = REPO_ROOT.joinpath(*owner.split(".")).with_suffix(".py")
            tree = ast.parse(source.read_text(encoding="utf-8"))
        assert any(isinstance(node, ast.ClassDef) and node.name == name for node in tree.body), target


def test_perception_components_use_canonical_owners() -> None:
    tree = ast.parse((RUNTIME / "demo.py").read_text(encoding="utf-8"))
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level == 0}

    assert "worldfoundry.base_models.perception_core.segment.sam2.build_sam" in imports
    assert "worldfoundry.base_models.three_dimensions.depth.moge.variants.uniworld_view" in imports
    assert "worldfoundry.base_models.three_dimensions.point_clouds.stream3r.models.stream3r" in imports
    for duplicate in ("moge", "sam2", "utils3d"):
        assert not (RUNTIME / duplicate).exists()


def test_plan_runs_the_vendored_entrypoint_with_official_layout_weights(tmp_path: Path) -> None:
    image = tmp_path / "input.png"
    image.write_bytes(b"fixture")
    root = tmp_path / "checkpoints"
    pipeline = UniWorldViewPipeline.from_pretrained(checkpoint_root=root, python_executable=sys.executable)

    plan = pipeline(images=str(image), output_path=tmp_path / "out" / "view.mp4", seed=7, fps=12, plan_only=True)
    command = plan["command"]

    assert Path(command[0]).resolve() == Path(sys.executable).resolve()
    assert command[1:3] == ["-m", INFERENCE_MODULE]
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--fps") + 1] == "12"
    assert command[command.index("--transformer_path") + 1] == str(root.resolve() / "UniView")
    assert command[command.index("--sam2_checkpoint") + 1] == str(root.resolve() / "sam2" / "sam2_hiera_large.pt")
    assert Path(plan["cwd"]).parent == (tmp_path / "out").resolve()


def test_model_reference_prefers_its_bundle_per_weight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpts"))
    bundle = tmp_path / "UniWorld-View"
    (bundle / "UniView").mkdir(parents=True)

    pipeline = UniWorldViewPipeline.from_pretrained(str(bundle), python_executable=sys.executable)

    assert pipeline.weights["transformer_path"] == bundle.resolve() / "UniView"
    assert (
        pipeline.weights["sam2_checkpoint"]
        == (tmp_path / "ckpts" / "facebook--sam2-hiera-large" / "sam2_hiera_large.pt").resolve()
    )


def test_setup_diffusion_passes_weight_name_and_local_files_only_for_file_lora(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib
    import types
    from types import SimpleNamespace

    if importlib.util.find_spec("carvekit") is None:
        for mod_name in ("carvekit", "carvekit.ml", "carvekit.ml.wrap", "carvekit.ml.wrap.tracer_b7"):
            mod = types.ModuleType(mod_name)
            if mod_name.endswith("tracer_b7"):
                setattr(mod, "TracerUniversalB7", object)
            monkeypatch.setitem(sys.modules, mod_name, mod)

    demo = importlib.import_module("worldfoundry.synthesis.visual_generation.uniworld_view.uniworld_view_runtime.demo")

    lora_file = tmp_path / "causvid.safetensors"
    lora_file.write_bytes(b"weights")
    lora_dir = tmp_path / "lora_dir"
    lora_dir.mkdir()

    calls: list[tuple[str, dict[str, object]]] = []

    class FakePipeline:
        def to(self, device: str):
            return self

        def load_lora_weights(self, path: str, **kwargs: object) -> None:
            calls.append((path, kwargs))

        def set_adapters(self, names: list[str], adapter_weights: list[float]) -> None:
            pass

        def fuse_lora(self) -> None:
            pass

    for attr in ("WanVACETransformer3DModel", "AutoencoderKLWan", "AutoTokenizer"):
        monkeypatch.setattr(getattr(demo, attr), "from_pretrained", classmethod(lambda cls, *a, **k: object()))
    monkeypatch.setattr(
        demo.UMT5EncoderModel,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: SimpleNamespace(eval=lambda: None)),
    )
    monkeypatch.setattr(demo.FlowMatchEulerDiscreteScheduler, "from_config", classmethod(lambda cls, *a, **k: object()))
    monkeypatch.setattr(demo, "WanVACEPipeline", lambda **kwargs: FakePipeline())

    scene = object.__new__(demo.UniScene)
    scene.opts = SimpleNamespace(
        transformer_path="t",
        model_name="m",
        weight_dtype=None,
        device="cpu",
        low_gpu_memory_mode=False,
        lora_path=str(lora_file),
    )
    scene.setup_diffusion()
    assert calls == [
        (
            str(tmp_path),
            {"weight_name": "causvid.safetensors", "local_files_only": True, "adapter_name": "causvid_lora"},
        )
    ]

    calls.clear()
    scene.opts.lora_path = str(lora_dir)
    scene.setup_diffusion()
    assert calls == [(str(lora_dir), {"local_files_only": True, "adapter_name": "causvid_lora"})]
