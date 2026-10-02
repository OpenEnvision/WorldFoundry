from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

av = pytest.importorskip("av")
numpy = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from worldfoundry.training.data import (  # noqa: E402
    LatentTokenBatchSampler,
    MediaReference,
    TrainingManifestDataset,
    TrainingSample,
    VideoDecodeConfig,
    VideoDecodingDataset,
    VideoLatentGeometry,
    VideoResolutionBucket,
    assign_video_buckets,
    collate_decoded_video_samples,
    decode_video_sample,
    video_frame_indices,
)
from worldfoundry.training.data.video_dataset import _cover_resize_center_crop  # noqa: E402


def _write_video(path: Path, *, frames: int = 8, height: int = 48, width: int = 64, fps: int = 8) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("ffv1", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            array = numpy.zeros((height, width, 3), dtype=numpy.uint8)
            array[:, :, 0] = index * 20
            array[:, :, 1] = numpy.arange(width, dtype=numpy.uint8)[None, :]
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _sample(path: Path, *, num_frames: int = 8) -> TrainingSample:
    return TrainingSample(
        sample_id="video-sample",
        task="t2v",
        prompt="a red gradient changes over time",
        media=MediaReference(
            uri=path.name,
            mime_type="video/x-matroska",
            size_bytes=path.stat().st_size,
        ),
        width=64,
        height=48,
        num_frames=num_frames,
        fps=8.0,
        conditions={},
        split="train",
        safety={"accepted": True},
    )


def _manifest_dataset(tmp_path: Path, sample: TrainingSample) -> TrainingManifestDataset:
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps(sample.to_dict()) + "\n", encoding="utf-8")
    return TrainingManifestDataset.from_file(manifest, verify_files=True)


def _assignment(sample: TrainingSample):
    return assign_video_buckets(
        (sample,),
        buckets=(VideoResolutionBucket(4, 32, 32, "text"),),
        geometry=VideoLatentGeometry(8, 8, 1, "uniform"),
        conditioning_layout="text",
    )[0]


def test_video_frame_indices_are_exact_and_reject_implicit_padding() -> None:
    assert video_frame_indices(8, 4, mode="head") == (0, 1, 2, 3)
    assert video_frame_indices(8, 4, mode="uniform_full") == (0, 2, 5, 7)
    assert video_frame_indices(8, 1, mode="uniform-full") == (3,)
    assert video_frame_indices(
        12,
        4,
        mode="seeded-random-contiguous",
        seed=42,
        sample_index=3,
    ) == (4, 5, 6, 7)
    with pytest.raises(ValueError, match="without padding"):
        video_frame_indices(3, 4, mode="head")


def test_floor_cover_resize_matches_ltx_trainer_geometry_and_pixels() -> None:
    frames = torch.arange(2 * 3 * 48 * 64, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(2, 3, 48, 64)

    actual, geometry = _cover_resize_center_crop(
        frames,
        target_height=32,
        target_width=32,
        interpolation="bilinear",
        resize_rounding="floor",
    )
    resized = torch.nn.functional.interpolate(
        frames.float().div(255.0),
        size=(32, 42),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    expected = resized[:, :, :, 5:37].contiguous()

    assert geometry == {
        "source_height": 48,
        "source_width": 64,
        "resized_height": 32,
        "resized_width": 42,
        "crop_top": 0,
        "crop_left": 5,
        "target_height": 32,
        "target_width": 32,
    }
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    _, default_geometry = _cover_resize_center_crop(
        frames,
        target_height=32,
        target_width=32,
        interpolation="bilinear",
        resize_rounding=VideoDecodeConfig().resize_rounding,
    )
    assert default_geometry["resized_width"] == 43


def test_video_decoding_dataset_emits_bucket_aligned_pixel_batch_and_transform_configs(tmp_path) -> None:
    video = tmp_path / "sample.mkv"
    _write_video(video)
    sample = _sample(video)
    manifest_dataset = _manifest_dataset(tmp_path, sample)
    assignments = (_assignment(sample),)
    config = VideoDecodeConfig(
        frame_sampling="uniform-full",
        decoder_thread_type="auto",
    )
    dataset = VideoDecodingDataset(manifest_dataset, assignments, config=config)
    decoded = dataset[0]
    batch = collate_decoded_video_samples([decoded])

    assert decoded.selected_frame_indices == (0, 2, 5, 7)
    assert decoded.decoded_frame_count == 8
    assert decoded.decoded_fps == 8.0
    assert tuple(decoded.pixel_values.shape) == (3, 4, 32, 32)
    assert float(decoded.pixel_values.min()) >= -1.0
    assert float(decoded.pixel_values.max()) <= 1.0
    assert tuple(batch.pixel_values.shape) == (1, 3, 4, 32, 32)
    assert tuple(batch.valid_mask.shape) == (1, 1, 4, 32, 32)
    assert batch.metadata["latent_tokens_per_microbatch"] == 4 * 4 * 4
    assert batch.metadata["frame_sampling"][0]["selected_frame_indices"] == [0, 2, 5, 7]

    sampler = LatentTokenBatchSampler(
        dataset,
        max_latent_tokens=64,
        shuffle=False,
        tail_policy="pad",
    )
    assert list(sampler) == [[0]]


def test_seeded_contiguous_decode_and_direct_resize_are_reproducible(tmp_path) -> None:
    video = tmp_path / "sample.mkv"
    _write_video(video)
    sample = _sample(video)
    assignment = _assignment(sample)
    config = VideoDecodeConfig(
        frame_sampling="seeded-random-contiguous",
        frame_sampling_seed=0,
        spatial_transform="direct-resize",
        interpolation="bilinear",
    )

    first = decode_video_sample(sample, assignment, media_path=video, config=config)
    second = decode_video_sample(sample, assignment, media_path=video, config=config)

    assert first.selected_frame_indices == (3, 4, 5, 6)
    assert second.selected_frame_indices == first.selected_frame_indices
    torch.testing.assert_close(second.pixel_values, first.pixel_values, rtol=0.0, atol=0.0)
    assert first.frame_sampling == {
        "source_num_frames": 8,
        "source_fps": 8.0,
        "mode": "seeded-random-contiguous",
        "selected_frame_indices": [3, 4, 5, 6],
        "seed": 0,
        "sample_index": 0,
    }
    assert first.spatial_transform == {
        "mode": "direct-resize",
        "interpolation": "bilinear",
        "value_range": "minus-one-one",
        "parameters": {
            "source_height": 48,
            "source_width": 64,
            "target_height": 32,
            "target_width": 32,
        },
    }


def test_video_decode_rejects_manifest_count_drift(tmp_path) -> None:
    video = tmp_path / "sample.mkv"
    _write_video(video)
    sample = _sample(video)
    assignment = _assignment(sample)

    wrong_count_sample = replace(sample, num_frames=9)
    wrong_count_assignment = replace(assignment, source_num_frames=9)
    with pytest.raises(ValueError, match="decoded frame count differs"):
        decode_video_sample(
            wrong_count_sample,
            wrong_count_assignment,
            media_path=video,
            config=VideoDecodeConfig(verify_manifest_frame_count=True),
        )
