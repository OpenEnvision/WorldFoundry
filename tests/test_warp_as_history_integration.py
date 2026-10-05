import importlib
import pkgutil
import csv
from pathlib import Path

from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.base_models.three_dimensions.point_clouds.pi3 import SOURCE_ROOT as PI3_SOURCE_ROOT
from worldfoundry.synthesis.visual_generation.warp_as_history.variants import (
    WARP_AS_HISTORY_VARIANTS,
    get_warp_as_history_variant,
    runtime_root,
    test_cases_root as warp_as_history_test_cases_root,
)
import worldfoundry.pipelines.warp_as_history as wah_pipelines
from worldfoundry.pipelines.warp_as_history.pipeline_warp_as_history import WarpAsHistoryPipeline
from worldfoundry.studio.inference.catalog import (
    _warp_as_history_default_load_kwargs,
    _warp_as_history_default_ref,
    discover_catalog,
)


def test_warp_as_history_runtime_uses_shared_base_models():
    root = runtime_root()
    assert not (root / "scripts").exists()
    assert (root / "warp_as_history" / "infer.py").is_file()
    assert (root / "warp_as_history" / "pipeline.py").is_file()
    assert (root / "helios" / "diffusers_version" / "pipeline_helios_diffusers.py").is_file()
    assert not (root / "third_party" / "Pi3").exists()
    assert not (root / "data" / "demo").exists()
    assert (PI3_SOURCE_ROOT / "pi3" / "models" / "pi3x.py").is_file()
    for variant in WARP_AS_HISTORY_VARIANTS.values():
        assert (warp_as_history_test_cases_root() / variant.demo_csv_path).is_file(), variant.model_id


def test_warp_as_history_runtime_logic_lives_under_synthesis():
    synthesis_root = (
        Path(__file__).resolve().parents[1]
        / "worldfoundry/synthesis/visual_generation/warp_as_history"
    )
    base_root = (
        Path(__file__).resolve().parents[1]
        / "worldfoundry/synthesis/visual_generation/warp_as_history"
    )
    synthesis_text = (synthesis_root / "warp_as_history_synthesis.py").read_text(encoding="utf-8")
    runtime_text = (base_root / "worldfoundry_runtime.py").read_text(encoding="utf-8")

    assert (synthesis_root / "variants.py").is_file()
    assert (base_root / "variants.py").is_file()
    assert "class WarpAsHistoryRuntime" in runtime_text
    assert "run_infer_from_csv" in runtime_text
    assert "subprocess.run" not in runtime_text
    assert "def _write_csv" in runtime_text
    assert "from .worldfoundry_runtime import WarpAsHistoryRuntime" in synthesis_text
    assert "subprocess.run" not in synthesis_text
    assert "def _write_csv" not in synthesis_text
    assert "def _subprocess_env" not in synthesis_text


def test_all_warp_as_history_pipeline_modules_import():
    for module in pkgutil.iter_modules(wah_pipelines.__path__):
        if module.name.startswith("pipeline_"):
            importlib.import_module(f"{wah_pipelines.__name__}.{module.name}")


def test_warp_as_history_pipeline_plan_only_uses_in_process_runtime(tmp_path):
    pipe = WarpAsHistoryPipeline.from_pretrained(lazy=True)

    result = pipe(
        prompt="a camera controlled validation test",
        output_path=tmp_path / "wah.mp4",
        plan_only=True,
        return_dict=True,
    )

    assert result["model_id"] == "warp-as-history"
    assert result["backend_quality"] == "execution_plan"
    assert result["artifact_kind"] == "generated_video"
    assert result["artifact_path"].endswith("wah.json")
    assert result["runtime"].endswith("in_tree_runtime")
    assert "Helios-Distilled" in result["infer_kwargs"]["model_path"]
    assert result["infer_kwargs"]["lora_path"].endswith("visible_lora_state_step1000.safetensors")
    with Path(result["infer_kwargs"]["csv_path"]).open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert Path(row["first_frame_path"]).is_file()
    assert Path(row["warp_video_path"]).is_file()
    assert Path(row["warp_visibility_mask_path"]).is_file()


def test_warp_as_history_catalog_entries_include_aliases():
    entry = load_model_zoo_registry().get("warp-as-history")

    assert entry.pipeline_target == "worldfoundry.pipelines.warp_as_history.pipeline_warp_as_history:WarpAsHistoryPipeline"
    assert {"wah", "warp_as_history", "yyfz233/warp-as-history"}.issubset(set(entry.aliases))
    assert get_warp_as_history_variant("wah").model_id == "warp-as-history"


def test_warp_as_history_is_discoverable_by_studio_catalog():
    ids = {entry.model_id for entry in discover_catalog()}
    assert "warp-as-history" in ids


def test_warp_as_history_catalog_resolves_all_local_checkpoint_roles(tmp_path, monkeypatch):
    checkpoint_root = tmp_path / "ckpts"
    model_root = checkpoint_root / "BestWishYsh--Helios-Distilled"
    lora_path = checkpoint_root / "yyfz233--warp-as-history" / "visible_lora_state_step1000.safetensors"
    pi3x_path = checkpoint_root / "yyfz233--Pi3X" / "model.safetensors"
    model_root.mkdir(parents=True)
    lora_path.parent.mkdir(parents=True)
    pi3x_path.parent.mkdir(parents=True)
    lora_path.write_bytes(b"lora")
    pi3x_path.write_bytes(b"pi3x")
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(checkpoint_root))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "ckpt"))

    components = _warp_as_history_default_load_kwargs()["required_components"]

    assert _warp_as_history_default_ref() == str(model_root)
    assert components["lora_path"] == str(lora_path)
    assert components["pi3x_ckpt_path"] == str(pi3x_path)


def test_warp_as_history_local_lora_file_sets_explicit_weight_name(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(runtime_root()))
    from warp_as_history.pipeline import _diffusers_lora_source

    lora_path = tmp_path / "visible_lora_state_step1000.safetensors"
    lora_path.write_bytes(b"lora")

    source, weight_name = _diffusers_lora_source(lora_path)

    assert source == str(tmp_path.resolve())
    assert weight_name == lora_path.name


def test_warp_as_history_ignores_only_incompatible_optional_torchao(monkeypatch):
    monkeypatch.syspath_prepend(str(runtime_root()))
    from peft.tuners.lora import torchao as peft_torchao
    from warp_as_history.pipeline import _ignore_incompatible_optional_torchao

    def incompatible():
        raise ImportError("unsupported torchao")

    monkeypatch.setattr(peft_torchao, "is_torchao_available", incompatible)
    with _ignore_incompatible_optional_torchao():
        assert peft_torchao.is_torchao_available() is False

    assert peft_torchao.is_torchao_available is incompatible


def test_warp_as_history_filters_evaluator_metadata(tmp_path):
    class Synthesis:
        def __init__(self):
            self.kwargs = None

        def predict(self, **kwargs):
            self.kwargs = kwargs
            return {"artifact_path": str(tmp_path / "warp.mp4")}

    synthesis = Synthesis()
    pipeline = WarpAsHistoryPipeline(synthesis_model=synthesis)
    pipeline(
        prompt="test",
        images="input.png",
        ref_image_path="duplicate.png",
        sample_id="sample-0000",
        task_name="camera-control",
        operator_kwargs={"sample_id": "nested-sample", "task_name": "nested-task"},
        return_dict=True,
    )

    assert "ref_image_path" not in synthesis.kwargs
    assert "sample_id" not in synthesis.kwargs
    assert "task_name" not in synthesis.kwargs
