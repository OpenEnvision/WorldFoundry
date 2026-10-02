"""Video/world replays fail on wrong pixels, missing errors and leaked state."""

from __future__ import annotations

import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "video_replay_contract", Path(__file__).resolve().parents[1] / "manual" / "geometry_regression.py"
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class ResidentPipeline:
    def __init__(self):
        self.buffer = np.zeros((3, 2, 4, 3), dtype=np.uint8)
        self.position = 0

    def __call__(self, value=7, fail=False):
        if fail:
            raise ValueError("invalid video geometry")
        self.buffer.fill(value)
        return {"video": self.buffer, "latents": np.full((1, 2, 1, 1, 1), value, dtype=np.float32)}

    def reset_realtime(self):
        self.position = 0

    def stream_realtime(self, action=1):
        self.position += action
        return {"frames": np.full((3, 2, 4, 3), self.position, dtype=np.uint8)}


def sequence():
    return [
        {"name": "a", "call": {"value": 7}, "outputs": ["video", "latents"]},
        {"name": "b", "call": {"value": 9}, "outputs": ["video", "latents"]},
        {"name": "invalid", "call": {"fail": True},
         "expect_error": {"type": "builtins.ValueError", "match": "invalid video geometry"}},
        {"name": "repeat", "call": {"value": 7}, "outputs": ["video", "latents"]},
    ]


def comparisons():
    return [
        {"left": "a.video", "right": "b.video", "relation": "different"},
        {"left": "a.video", "right": "repeat.video", "relation": "equal"},
        {"left": "a.latents", "right": "repeat.latents", "relation": "equal"},
    ]


def test_sequence_preserves_responses_from_reused_cpu_buffers_and_expected_errors():
    pipe, arrays = ResidentPipeline(), {}
    events = replay.run_sequence(pipe, sequence(), arrays)
    assert [item["status"] for item in events] == ["passed", "passed", "expected_error", "passed"]
    assert np.all(arrays["a.video"] == 7) and np.all(arrays["b.video"] == 9)
    pipe.buffer.fill(100)
    assert np.all(arrays["repeat.video"] == 7)
    replay.validate_contracts(arrays, {"a.video": {"shape": [3, 2, 4, 3], "dtype": "uint8"}}, comparisons())


@pytest.mark.parametrize("drift", ["repeat", "ignored_prompt", "latents"])
def test_sequence_detects_state_leak_ignored_condition_and_changed_latents(drift):
    arrays = {}
    replay.run_sequence(ResidentPipeline(), sequence(), arrays)
    if drift == "repeat":
        arrays["repeat.video"][0, 0, 0, 0] += 1
    elif drift == "ignored_prompt":
        arrays["b.video"] = arrays["a.video"].copy()
    else:
        arrays["repeat.latents"] += np.float32(0.00001)
    with pytest.raises(ValueError, match="Output relation"):
        replay.validate_contracts(arrays, {}, comparisons())


@pytest.mark.parametrize("kind", ["did_not_fail", "wrong_type", "wrong_message", "missing_method"])
def test_expected_error_cannot_swallow_unrelated_failure_or_success(kind):
    step = sequence()[2]
    if kind == "did_not_fail":
        step["call"] = {}
    elif kind == "wrong_type":
        step["expect_error"]["type"] = "builtins.RuntimeError"
    elif kind == "wrong_message":
        step["expect_error"]["match"] = "checkpoint failure"
    else:
        step["method"] = "missing_method"
    with pytest.raises((ValueError, AttributeError)):
        replay.run_sequence(ResidentPipeline(), [step], {})


@pytest.mark.parametrize("steps", [[], [{"name": "../escape"}], [{"name": "a"}, {"name": "a"}],
                                   [{"name": "a", "skip_failure": True}]])
def test_invalid_sequences_fail_before_claiming_success(steps):
    with pytest.raises(ValueError):
        replay.run_sequence(ResidentPipeline(), steps, {})


def test_causal_rollout_and_new_session_reset_are_numerically_checked():
    steps = [
        {"name": "first", "method": "stream_realtime", "call": {"action": 2}},
        {"name": "second", "method": "stream_realtime", "call": {"action": 3}},
        {"name": "reset", "method": "reset_realtime", "outputs": []},
        {"name": "repeat", "method": "stream_realtime", "call": {"action": 2}},
    ]
    arrays = {}
    replay.run_sequence(ResidentPipeline(), steps, arrays)
    assert np.all(arrays["first.frames"] == 2)
    assert np.all(arrays["second.frames"] == 5)
    replay.validate_contracts(arrays, {}, [
        {"left": "first.frames", "right": "repeat.frames", "relation": "equal"}
    ])


@pytest.mark.parametrize("contract", [
    {"shape": [3, 4, 2, 3]}, {"dtype": "float32"}, {"min": 8}, {"max": 6}, {"min": float("nan")},
    {"skip": True},
])
def test_geometry_precision_range_and_invalid_contracts_are_rejected(contract):
    arrays = {"frames": np.full((3, 2, 4, 3), 7, dtype=np.uint8)}
    with pytest.raises(ValueError):
        replay.validate_contracts(arrays, {"frames": contract}, [])


def test_accepted_video_must_match_requested_fps_not_just_another_wrong_run():
    arrays = {"export.clip.mp4.fps": np.array([24, 1], dtype=np.int64)}
    with pytest.raises(ValueError, match="values changed"):
        replay.validate_contracts(arrays, {"export.clip.mp4.fps": {"values": [16, 1]}}, [])


@pytest.mark.parametrize("relation", ["equal", "different", "approximately_equal"])
def test_missing_outputs_never_satisfy_output_comparison(relation):
    with pytest.raises(ValueError, match="Missing comparison"):
        replay.validate_contracts({}, {}, [{"left": "a", "right": "b", "relation": relation}])


def fake_video_decoder(monkeypatch, *, rate=Fraction(16, 1), frames=3, pixel=7):
    class Container:
        streams = SimpleNamespace(video=[SimpleNamespace(average_rate=rate)])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def decode(self, stream):
            for _ in range(frames):
                yield SimpleNamespace(to_ndarray=lambda format: np.full((2, 4, 3), pixel, dtype=np.uint8))

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda path: Container()))


def evidence(root):
    root.mkdir()
    np.savez_compressed(root / "arrays.npz", latents=np.ones((1, 2, 1, 1, 1), dtype=np.float32))
    video = root / "clip.mp4"
    video.write_bytes(b"decoder fixture")
    metadata = {"status": "passed", "case": {"id": "video", "seed": 43},
                "assets": {"checkpoint": "same"}, "runtime": {"torch": "same"},
                "arrays_sha256": replay.sha256(root / "arrays.npz"), "exported_files": [str(video)]}
    (root / "manifest.json").write_text(json.dumps(metadata))
    return root


@pytest.mark.parametrize("change", ["pixels", "fps", "frames", "none"])
def test_export_video_pixels_frame_rate_and_length_are_compared(monkeypatch, tmp_path, change):
    left, right = evidence(tmp_path / "left"), evidence(tmp_path / "right")
    original = replay.exported_arrays

    def decoded(root, files):
        changed = root == right
        fake_video_decoder(monkeypatch, rate=Fraction(30 if changed and change == "fps" else 16, 1),
                           frames=2 if changed and change == "frames" else 3,
                           pixel=8 if changed and change == "pixels" else 7)
        return original(root, files)

    monkeypatch.setattr(replay, "exported_arrays", decoded)
    if change == "frames":
        with pytest.raises(ValueError, match="shape or dtype"):
            replay.compare_runs(left, right)
    else:
        report = replay.compare_runs(left, right, atol=100)
        assert report["status"] == ("passed" if change == "none" else "failed")


@pytest.mark.parametrize("rate,frames", [(None, 3), (Fraction(0), 3), (Fraction(16), 0)])
def test_undecodable_video_or_unknown_fps_cannot_pass(monkeypatch, tmp_path, rate, frames):
    root = evidence(tmp_path / "run")
    fake_video_decoder(monkeypatch, rate=rate, frames=frames)
    with pytest.raises(ValueError, match="frame rate|decodable frames"):
        replay.exported_arrays(root, [str(root / "clip.mp4")])


def test_run_case_saves_sequence_evidence_without_requiring_torch(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    module_file = source / "pipeline.py"
    module_file.write_text("# isolated orchestrator fixture\n")
    module = SimpleNamespace(__file__=str(module_file), Pipeline=SimpleNamespace(from_pretrained=lambda **kw: ResidentPipeline()))
    monkeypatch.setitem(sys.modules, "video_test_pipeline", module)
    torch = SimpleNamespace(manual_seed=lambda seed: None, cuda=SimpleNamespace(is_available=lambda: False),
                            __version__="fixture", version=SimpleNamespace(cuda=None),
                            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                                     cudnn=SimpleNamespace(version=lambda: None, benchmark=False)),
                            are_deterministic_algorithms_enabled=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    weights = tmp_path / "weights"
    weights.write_text("pinned asset")
    case = {"id": "sequence", "target": "video_test_pipeline:Pipeline", "load": {}, "seed": 43,
            "assets": {"checkpoint": str(weights)}, "sequence": sequence(), "required_outputs": ["a.video"],
            "array_contracts": {"a.video": {"shape": [3, 2, 4, 3], "dtype": "uint8"}},
            "comparisons": comparisons()}
    recipe = tmp_path / "case.json"
    recipe.write_text(json.dumps(case))
    out = tmp_path / "out"
    manifest = replay.run_case(recipe, source, out)
    assert manifest["status"] == "passed", manifest
    assert manifest["sequence_events"][2]["status"] == "expected_error"
    with np.load(out / "arrays.npz") as values:
        assert set(values.files) == {"a.video", "a.latents", "b.video", "b.latents", "repeat.video", "repeat.latents"}
    # A changed contract must fail even when reference and candidate would agree.
    case["array_contracts"]["a.video"]["shape"] = [99, 2, 4, 3]
    recipe.write_text(json.dumps(case))
    failed = replay.run_case(recipe, source, tmp_path / "failed")
    assert failed["status"] == "failed" and "shape changed" in failed["error"]
