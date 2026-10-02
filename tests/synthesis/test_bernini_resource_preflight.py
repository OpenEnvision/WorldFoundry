from __future__ import annotations

import json
from pathlib import Path

from worldfoundry.synthesis.visual_generation.bernini import worldfoundry_runtime


def test_resource_estimate_uses_largest_sequential_stage_not_combined_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    component = checkpoint / "bernini"
    component.mkdir(parents=True)
    (component / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    index = component / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {"total_size": 1000}, "weight_map": {}}))
    monkeypatch.setattr(
        worldfoundry_runtime,
        "_safetensors_group_sizes",
        lambda _path: {"diff_dec.transformer": 600, "mllm.model": 400},
    )

    report = worldfoundry_runtime.checkpoint_resource_estimate(checkpoint)

    assert report["peak_cuda_weight_bytes"] == 600


def test_cgroup_headroom_counts_only_inactive_file_cache_as_reclaimable(monkeypatch) -> None:
    values = {
        "/sys/fs/cgroup/memory/memory.usage_in_bytes": "800",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes": "1000",
        "/sys/fs/cgroup/memory/memory.stat": "total_inactive_file 300\ntotal_active_file 400\n",
    }
    original = Path.read_text

    def fake_read_text(path: Path, *args, **kwargs) -> str:
        value = values.get(str(path))
        if value is not None:
            return value
        if str(path).startswith("/sys/fs/cgroup/"):
            raise OSError("fixture path is unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    assert worldfoundry_runtime._cgroup_memory_available_bytes() == 500
