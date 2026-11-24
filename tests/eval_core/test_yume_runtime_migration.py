from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO_RUNTIME_ROOT = REPO_ROOT / "worldfoundry/synthesis/visual_generation/yume/yume_runtime"
VIDEO_RUNTIME_ADAPTER = REPO_ROOT / "worldfoundry/synthesis/visual_generation/yume/worldfoundry_runtime.py"


def test_yume_runtime_lives_under_synthesis() -> None:
    assert (VIDEO_RUNTIME_ROOT / "yume/image2video.py").is_file()
    assert (VIDEO_RUNTIME_ROOT / "yume_1p5/textimage2video.py").is_file()
    assert (VIDEO_RUNTIME_ROOT / "yume_1p5/modules/model.py").is_file()
    assert VIDEO_RUNTIME_ADAPTER.is_file()
    assert (VIDEO_RUNTIME_ROOT / "yume/worldfoundry_runtime.py").is_file()
    assert (VIDEO_RUNTIME_ROOT / "yume_1p5/worldfoundry_runtime.py").is_file()


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
