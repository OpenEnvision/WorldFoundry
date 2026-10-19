"""Synchronized sensor presentation preserves model channel identity."""

import numpy as np
import pytest

from worldfoundry.core.execution.realtime.contracts import RealtimeSensor, RealtimeSpec
from worldfoundry.studio.serving.realtime.media import realtime_frames_from_result
from worldfoundry.studio.serving.realtime.sensors import sensor_layout_from_result


def sensor_result():
    spec = RealtimeSpec(
        sensors=(
            RealtimeSensor("front", channel_index=0),
            RealtimeSensor("rear", channel_index=1),
            RealtimeSensor("lidar", kind="lidar", channel_index=2),
        )
    )
    red = np.full((2, 32, 48, 3), (255, 0, 0), dtype=np.uint8)
    green = np.full((2, 32, 48, 3), (0, 255, 0), dtype=np.uint8)
    points = np.array([[0.0, 0.0, 0.0], [1.0, 3.0, 1.0]])
    return {
        "realtime_spec": spec.to_payload(),
        "sensor_frames": {"rear": green, "lidar": [points, points], "front": red},
    }


def test_sensor_contract_roundtrip_and_single_rgb_payload_unchanged():
    assert "sensors" not in RealtimeSpec().to_payload()
    spec = RealtimeSpec.from_payload(sensor_result())
    assert [sensor.key for sensor in spec.sensors] == ["front", "rear", "lidar"]
    assert spec == RealtimeSpec.from_payload(spec.to_payload())
    with pytest.raises(ValueError, match="unique"):
        RealtimeSpec(sensors=(RealtimeSensor("a", channel_index=0), RealtimeSensor("b", channel_index=0)))


def test_sensor_atlas_uses_declared_order_without_mutating_model_outputs():
    result = sensor_result()
    before = tuple(result["sensor_frames"])
    frames = realtime_frames_from_result(result)
    layout = sensor_layout_from_result(result)
    assert len(frames) == 2
    assert [view["key"] for view in layout] == ["front", "rear", "lidar"]
    assert [view["channel_index"] for view in layout] == [0, 1, 2]
    np.testing.assert_array_equal(frames[0][256, 256], [255, 0, 0])
    np.testing.assert_array_equal(frames[0][256, 768], [0, 255, 0])
    assert np.any(frames[0][512:, :512])
    assert tuple(result["sensor_frames"]) == before
    assert result["sensor_frames"]["front"].shape == (2, 32, 48, 3)


@pytest.mark.parametrize("failure", ["missing", "cadence", "undeclared"])
def test_sensor_chunks_reject_misaligned_results(failure):
    result = sensor_result()
    if failure == "missing":
        result["sensor_frames"].pop("rear")
    elif failure == "cadence":
        result["sensor_frames"]["rear"] = result["sensor_frames"]["rear"][:1]
    else:
        result.pop("realtime_spec")
    with pytest.raises(ValueError):
        realtime_frames_from_result(result)


def test_depth_sensor_uses_metric_range_and_preserves_frame_count():
    spec = RealtimeSpec(sensors=(RealtimeSensor("depth", kind="depth", range_m=10),))
    result = {"realtime_spec": spec.to_payload(), "sensor_frames": {"depth": [np.array([[0.0, 5.0, 10.0]])]}}
    frame = realtime_frames_from_result(result)[0]
    np.testing.assert_array_equal(frame[0, 0], [0, 0, 0])
    np.testing.assert_array_equal(frame[0, 1], [127, 127, 0])
    np.testing.assert_array_equal(frame[0, 2], [255, 0, 0])
