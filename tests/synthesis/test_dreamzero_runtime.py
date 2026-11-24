# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from worldfoundry.synthesis.action_generation.dreamzero import runtime

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("chunk_size", [None, 4])
def test_server_command_preserves_distributed_cache_and_chunk_settings(
    workers: int, cache: bool, chunk_size: int | None
) -> None:
    command = runtime.build_official_server_command(
        checkpoint_dir="/assets/dreamzero",
        python="/env/python",
        repo_root="/unused/source",
        port=6001,
        nproc_per_node=workers,
        enable_dit_cache=cache,
        max_chunk_size=chunk_size,
    )
    assert command[:8] == [
        "/env/python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(workers),
        "-m",
        "worldfoundry.synthesis.action_generation.dreamzero.server",
    ]
    expected = ["--port", "6001"]
    if cache:
        expected.append("--enable-dit-cache")
    expected.extend(["--model-path", "/assets/dreamzero"])
    if chunk_size is not None:
        expected.extend(["--max-chunk-size", str(chunk_size)])
    assert command[8:] == expected


def test_concat_transform_iterates_keys_view_for_batch_mapping() -> None:
    import numpy as np

    from worldfoundry.synthesis.action_generation.dreamzero.preprocessing.transform_concat import ConcatTransform

    class _SliceIterBatch(dict):
        def __iter__(self):
            # Mimic tianshou==0.5.1 Batch.__iter__, which yields item slices rather than key strings.
            return iter([{"not_a_string": 1}])

    batch = _SliceIterBatch(
        {
            "video.ego": np.zeros((2, 4, 4, 3), dtype=np.uint8),
            "annotation.prompt": "pick up cup",
        }
    )
    transform = ConcatTransform(video_concat_order=["video.ego"])

    out = transform.apply(batch)

    assert out["video"].shape == (2, 1, 4, 4, 3)
