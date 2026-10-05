from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHESIS_ROOT = REPO_ROOT / "worldfoundry/synthesis/visual_generation/yume"
VIDEO_RUNTIME_ROOT = (
    REPO_ROOT / "worldfoundry/synthesis/visual_generation/yume/yume_runtime"
)
VIDEO_RUNTIME_ADAPTER = (
    REPO_ROOT / "worldfoundry/synthesis/visual_generation/yume/worldfoundry_runtime.py"
)
VIDEO_RUNTIME_PACKAGE = "worldfoundry.synthesis.visual_generation.yume.worldfoundry_runtime"


def test_yume_runtime_lives_under_synthesis() -> None:
    assert (VIDEO_RUNTIME_ROOT / "yume/image2video.py").is_file()
    assert (VIDEO_RUNTIME_ROOT / "yume_1p5/textimage2video.py").is_file()
    assert (VIDEO_RUNTIME_ROOT / "yume_1p5/modules/model.py").is_file()
    assert VIDEO_RUNTIME_ADAPTER.is_file()
    assert (VIDEO_RUNTIME_ROOT / "yume/worldfoundry_runtime.py").is_file()
    assert (VIDEO_RUNTIME_ROOT / "yume_1p5/worldfoundry_runtime.py").is_file()


def test_yume_synthesis_does_not_keep_runtime_compat_packages() -> None:
    assert not (SYNTHESIS_ROOT / "yume").exists()
    assert not (SYNTHESIS_ROOT / "yume_1p5").exists()

    for relative_path in (
        "yume_synthesis.py",
        "yume_1p5_synthesis.py",
    ):
        source = (SYNTHESIS_ROOT / relative_path).read_text(encoding="utf-8")
        assert VIDEO_RUNTIME_PACKAGE in source
        for heavy_marker in (
            "torch.distributed",
            "init_process_group",
            "VideoProcessor",
            "get_sampling_sigmas",
            "YumeI2V(",
            "Yume1p5TI2V(",
        ):
            assert heavy_marker not in source


def test_yume_worldfoundry_runtime_owns_loading_and_sampling() -> None:
    yume_source = (VIDEO_RUNTIME_ROOT / "yume/worldfoundry_runtime.py").read_text(encoding="utf-8")
    yume_1p5_source = (VIDEO_RUNTIME_ROOT / "yume_1p5/worldfoundry_runtime.py").read_text(encoding="utf-8")

    assert "class YumeRuntime" in yume_source
    assert "class Yume1p5Runtime" in yume_1p5_source
    for source in (yume_source, yume_1p5_source):
        assert "torch.distributed" in source
        assert "init_process_group" in source
        assert "VideoProcessor" in source
        assert "get_sampling_sigmas" in source


def test_yume_runtime_reuses_base_wan_modules() -> None:
    runtime_files = [
        VIDEO_RUNTIME_ROOT / "yume/image2video.py",
        VIDEO_RUNTIME_ROOT / "yume_1p5/textimage2video.py",
        VIDEO_RUNTIME_ROOT / "yume_1p5/modules/model.py",
    ]

    for path in runtime_files:
        source = path.read_text(encoding="utf-8")
        assert "worldfoundry.base_models.diffusion_model.video.wan.wan_" in source
        assert "from wan." not in source
        assert "import wan" not in source
