from worldfoundry.runtime.inference_catalog import get_model_inference_spec
from worldfoundry.pipelines.lyra.pipeline_lyra1 import Lyra1Pipeline
from worldfoundry.pipelines.lyra.lyra_utils import load_pil_image
from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.synthesis.visual_generation.lyra_1.worldfoundry_runtime import Lyra1Runtime


def test_lyra1_catalog_selects_the_configured_checkpoint_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpts"))

    entry = find_entry("lyra-1")

    assert entry.display_name == "Lyra-1"
    assert entry.default_model_ref == str(tmp_path / "ckpts")
    assert entry.default_input_path.endswith("worldfoundry/data/test_cases/lyra/Lyra-1/00172.png")
    assert entry.default_call_kwargs["execute"] is True
    assert entry.default_call_kwargs["num_video_frames"] == 121


def test_lyra1_inference_contract_uses_portable_checkpoint_and_input_paths() -> None:
    spec = get_model_inference_spec("lyra-1")

    assert spec is not None
    variant = spec.variants[0]
    task = spec.tasks[0]
    assert variant.status == "requires_local_checkpoints"
    assert variant.checkpoints[0].uri == "${WORLDFOUNDRY_CKPT_DIR}"
    assert all(checkpoint.uri.startswith("${WORLDFOUNDRY_CKPT_DIR}") for checkpoint in variant.checkpoints)
    assert task.inputs[0].default.startswith("${WORLDFOUNDRY_REPO_ROOT}/")
    assert task.default_call_kwargs["execute"] is True


def test_lyra1_image_loader_expands_worldfoundry_placeholders(tmp_path, monkeypatch) -> None:
    from PIL import Image

    repo_root = tmp_path / "repo"
    image_path = repo_root / "fixture.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (4, 3), "blue").save(image_path)
    monkeypatch.setenv("WORLDFOUNDRY_REPO_ROOT", str(repo_root))

    image = load_pil_image("${WORLDFOUNDRY_REPO_ROOT}/fixture.png")

    assert image.size == (4, 3)


def test_lyra1_pipeline_returns_runtime_plan_without_video_fields(tmp_path) -> None:
    from PIL import Image

    class PlanSynthesis:
        def predict(self, **_kwargs):
            return {
                "status": "planned",
                "artifact_kind": "generated_video",
                "artifact_path": str(tmp_path / "lyra1_plan.json"),
            }

    pipeline = Lyra1Pipeline(synthesis_model=PlanSynthesis(), device="cpu")

    result = pipeline(
        images=Image.new("RGB", (4, 3), "blue"),
        interactions=["zoom_in"],
        execute=False,
        return_dict=True,
    )

    assert result["status"] == "planned"
    assert result["trajectory"] == "zoom_in"
    assert result["artifact_path"].endswith("lyra1_plan.json")


def test_lyra1_runtime_consumes_standard_workspace_metadata(tmp_path, monkeypatch) -> None:
    from PIL import Image

    class FakePipeline:
        def __call__(self, **kwargs):
            return {
                "generated_video_path": str(kwargs["output_path"]),
                "video": [],
                "artifacts": {},
            }

    runtime = Lyra1Runtime(device="cpu")
    monkeypatch.setattr(runtime, "_native_pipeline", lambda: FakePipeline())

    result = runtime.predict(
        visual_input=Image.new("RGB", (4, 3), "blue"),
        output_root=str(tmp_path),
        execute=True,
        multi_trajectory=False,
        image_path="ignored.png",
        input_path="ignored.png",
        output_path="ignored.mp4",
        task_type="novel-view-synthesis",
    )

    assert result["status"] == "completed"
    assert result["generated_video_path"].endswith("lyra1.mp4")
