from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from worldfoundry.evaluation.api import GenerationRequest
from worldfoundry.evaluation.models import resolve_model_zoo_runner
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile
from worldfoundry.pipelines.longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from worldfoundry.synthesis.visual_generation.longcat_video.longcat_video_runtime import video_io
from worldfoundry.synthesis.visual_generation.longcat_video.longcat_video_runtime import tokenizer_loading
from worldfoundry.synthesis.visual_generation.longcat_video.longcat_video_runtime.longcat_video.pipeline_longcat_video import (
    LongCatVideoPipeline as RuntimeLongCatVideoPipeline,
)
from worldfoundry.synthesis.visual_generation.longcat_video.worldfoundry_runtime import (
    LongCatVideoRuntime,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CATALOG_DIR = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"


def test_longcat_umt5_tokenizer_disables_mistral_regex_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, dict[str, object]]] = []
    sentinel = object()

    def fake_from_pretrained(checkpoint_dir, **kwargs):
        calls.append((checkpoint_dir, kwargs))
        return sentinel

    monkeypatch.setattr(tokenizer_loading.AutoTokenizer, "from_pretrained", fake_from_pretrained)

    assert tokenizer_loading.load_longcat_tokenizer("/checkpoints/longcat") is sentinel
    assert calls == [
        (
            "/checkpoints/longcat",
            {
                "subfolder": "tokenizer",
                "torch_dtype": pytest.importorskip("torch").bfloat16,
                "fix_mistral_regex": False,
            },
        )
    ]


def test_longcat_pipeline_default_call_rejects_preflight_artifact(tmp_path: Path) -> None:
    pipeline = LongCatVideoPipeline.from_pretrained(
        model_path=tmp_path / "checkpoint",
        device="cpu",
    )
    plan_path = tmp_path / "longcat_video_plan.json"

    with pytest.raises(RuntimeError, match="requires execute=True"):
        pipeline(
            prompt="a cyclist driving through downtown at dawn",
            output_path=plan_path,
            return_dict=True,
        )

    assert not plan_path.exists()


def test_longcat_pipeline_accepts_unified_model_path_mapping(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    pipeline = LongCatVideoPipeline.from_pretrained(
        model_path={"model_path": str(checkpoint)},
        device="cpu",
    )

    assert pipeline.checkpoint_dir == checkpoint.resolve()


def test_longcat_profile_declares_generated_video() -> None:
    profile = load_runtime_profile("longcat-video")

    assert profile.artifact_kind == "generated_video"
    assert profile.backend_stage == "in_tree_runtime"
    assert profile.integration_status == "integrated"
    assert profile.artifact_filename == "longcat_video.mp4"


def test_longcat_inference_contract_honors_hfd_root(tmp_path: Path) -> None:
    hfd_root = tmp_path / "hfd"
    env = dict(os.environ)
    env["WORLDFOUNDRY_HFD_ROOT"] = str(hfd_root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from worldfoundry.runtime.inference_catalog import get_model_inference_spec; "
                "spec = get_model_inference_spec('longcat-video'); "
                "print(json.dumps(spec.variants[0].checkpoint_map()))"
            ),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout)["primary"] == str(
        hfd_root / "meituan-longcat--LongCat-Video"
    )


def test_longcat_video_writer_falls_back_when_torchvision_pyav_is_incompatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "fallback.mp4"
    frames = np.zeros((3, 16, 24, 3), dtype=np.uint8)
    frames[1, :, :, 1] = 255

    def incompatible_writer(*args, **kwargs):
        del args, kwargs
        output_path.write_bytes(b"partial")
        raise TypeError("an integer is required")

    monkeypatch.setattr(video_io, "_torchvision_write_video", incompatible_writer)
    video_io.write_video(
        output_path,
        frames,
        fps=8,
        video_codec="libx264",
        options={"crf": "18"},
    )

    assert output_path.stat().st_size > 100


def test_longcat_runner_fails_without_execute(tmp_path: Path) -> None:
    resolved = resolve_model_zoo_runner(
        "longcat-video",
        manifest_dir=MODEL_CATALOG_DIR,
        runtime={"device": "cpu"},
    )

    results = resolved.runner.generate(
        [
            GenerationRequest(
                sample_id="longcat-smoke",
                task_name="longcat:smoke",
                inputs={"prompt": "a robotic dancer in a studio"},
                output_schema={"generated_video": {"kind": "video"}},
            )
        ]
    )

    assert results[0].status == "failed"
    assert not results[0].artifacts
    assert (
        results[0].error
        == "RuntimeError: LongCat-Video requires execute=True; preflight artifacts are no longer emitted."
    )


def test_longcat_inference_inputs_live_in_data_test_cases() -> None:
    runtime_root = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/longcat_video/longcat_video_runtime"
    )
    wrapper_root = REPO_ROOT / "worldfoundry/synthesis/visual_generation/longcat_video"
    test_case_root = REPO_ROOT / "worldfoundry/data/test_cases"

    assert (test_case_root / "dualcamctrl/demo_pic/route66.jpg").is_file()
    assert (test_case_root / "longcat_video/motorcycle.mp4").is_file()
    assert wrapper_root.is_dir()
    assert not (wrapper_root / "LICENSE").exists()
    assert not (wrapper_root / "run_demo_text_to_video.py").exists()
    assert not (wrapper_root / "longcat_video").exists()
    assert not (runtime_root / "assets").exists()
    assert not (runtime_root / "girl.png").exists()
    assert not (runtime_root / "motorcycle.mp4").exists()

    image_inference = (runtime_root / "run_inference_image_to_video.py").read_text(encoding="utf-8")
    video_inference = (runtime_root / "run_inference_video_continuation.py").read_text(encoding="utf-8")

    assert '"data" / "test_cases"' in image_inference
    assert '"dualcamctrl" / "demo_pic" / "route66.jpg"' in image_inference
    assert '"data" / "test_cases" / "longcat_video"' in video_inference


def test_longcat_runtime_keeps_only_catalog_supported_inference_scripts() -> None:
    runtime_root = (
        REPO_ROOT
        / "worldfoundry/synthesis/visual_generation/longcat_video/longcat_video_runtime"
    )
    removed = [
        "run_streamlit.py",
        "run_demo_interactive_video.py",
        "run_demo_avatar_single_audio_to_video.py",
        "run_demo_avatar_multi_audio_to_video.py",
        "LongCat-Video-Avatar-Tech-Report.pdf",
        "LongCat-Video-Avatar-1.5-Tech-Report.pdf",
        "longcat_video/pipeline_longcat_video_avatar.py",
        "longcat_video/modules/avatar",
        "longcat_video/modules/quantization.py",
        "longcat_video/audio_process",
    ]
    inference_scripts = [
        "run_inference_text_to_video.py",
        "run_inference_image_to_video.py",
        "run_inference_video_continuation.py",
        "run_inference_long_video.py",
    ]

    assert [name for name in removed if (runtime_root / name).exists()] == []
    assert [name for name in inference_scripts if not (runtime_root / name).is_file()] == []
    assert sorted(path.name for path in runtime_root.glob("run_demo*.py")) == []

    runtime = LongCatVideoRuntime(checkpoint_dir=REPO_ROOT / "missing-longcat-checkpoint", device="cpu")
    preflight = runtime.preflight()
    assert preflight["runtime_ready"] is True
    assert sorted(Path(path).name for path in preflight["runtime_scripts"].values()) == sorted(inference_scripts)
    assert preflight["missing_runtime_files"] == []


def test_longcat_t2v_cpu_offload_keeps_inactive_modules_on_cpu() -> None:
    class RecordingModule:
        def __init__(self) -> None:
            self.devices: list[str] = []

        def to(self, device, non_blocking=False):
            assert non_blocking is True
            self.devices.append(str(device))
            return self

    pipeline = object.__new__(RuntimeLongCatVideoPipeline)
    pipeline.text_encoder = RecordingModule()
    pipeline.dit = RecordingModule()
    pipeline.vae = RecordingModule()

    assert pipeline.enable_t2v_cpu_offload("cuda:0") is pipeline
    assert pipeline.device == pytest.importorskip("torch").device("cuda:0")
    assert pipeline._t2v_cpu_offload is True
    assert pipeline.text_encoder.devices == ["cpu"]
    assert pipeline.dit.devices == ["cpu"]
    assert pipeline.vae.devices == ["cpu"]


def test_longcat_t2v_cpu_offload_transitions_between_refine_stages() -> None:
    pipeline = object.__new__(RuntimeLongCatVideoPipeline)
    pipeline._t2v_cpu_offload = True
    pipeline.device = pytest.importorskip("torch").device("cuda:0")
    moves: list[tuple[str, str]] = []
    pipeline._move_module = lambda name, device: moves.append((name, str(device)))

    pipeline._transition_t2v_modules(
        activate="vae",
        deactivate=("text_encoder",),
    )
    pipeline._transition_t2v_modules(
        activate="dit",
        deactivate=("vae",),
    )

    assert moves == [
        ("text_encoder", "cpu"),
        ("vae", "cuda:0"),
        ("vae", "cpu"),
        ("dit", "cuda:0"),
    ]

    pipeline._t2v_cpu_offload = False
    pipeline._transition_t2v_modules(activate="text_encoder", deactivate=("dit",))
    assert len(moves) == 4


def test_longcat_prefers_distilled_t2v_when_base_stage_is_skipped(tmp_path: Path) -> None:
    distilled = tmp_path / "output_t2v_distill.mp4"
    distilled.touch()

    from worldfoundry.synthesis.visual_generation.longcat_video import worldfoundry_runtime

    assert worldfoundry_runtime._preferred_video_output([distilled]) == distilled
