# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""CLI selection, changed-image inputs, and incomplete replay evidence."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.inference import wan_execution_preparation as benchmark

pytestmark = pytest.mark.unit


@pytest.fixture
def replay_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model_id="wan2.2-ti2v-5b",
        checkpoint=tmp_path / "checkpoint",
        output_dir=tmp_path / "outputs",
        height=256,
        width=256,
        frames=17,
        steps=8,
        images=[tmp_path / "a.png", tmp_path / "b.png"],
        configurations=["static-graph"],
        inplace_residual=True,
        dit_weight_dtype="bfloat16",
        request_batching=False,
    )


def test_cli_keeps_selected_candidates_and_returns_quality_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(args: argparse.Namespace) -> dict[str, bool]:
        assert args.model_id == "wan2.2-ti2v-5b"
        assert args.configurations == ["regional-static-graph"]
        assert args.images == [Path("a.png"), Path("b.png")]
        assert args.inplace_residual
        assert args.request_batching
        return {"passed": False}

    monkeypatch.setattr(benchmark, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wan_execution_preparation",
            "--model-id",
            "wan2.2-ti2v-5b",
            "--checkpoint",
            "weights",
            "--output-dir",
            "outputs",
            "--configurations",
            "regional-static-graph",
            "--images",
            "a.png",
            "b.png",
            "--inplace-residual",
            "--request-batching",
        ],
    )
    assert benchmark.main() == 1


def test_load_failure_replaces_previous_completed_report(
    replay_args: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from worldfoundry.pipelines.wan.pipeline_wan_2p2 import Wan2p2Pipeline

    def fail_loading(**kwargs: Any) -> None:
        assert "inplace_residual" not in kwargs
        assert kwargs["dit_weight_dtype"] == "bfloat16"
        raise RuntimeError("checkpoint load failed")

    monkeypatch.setattr(Wan2p2Pipeline, "from_pretrained", fail_loading)
    monkeypatch.setattr(
        "worldfoundry.runtime.performance.capture_runtime_fingerprint",
        lambda: SimpleNamespace(to_dict=lambda: {}),
    )
    replay_args.output_dir.mkdir()
    report_path = replay_args.output_dir / "report.json"
    report_path.write_text(json.dumps({"complete": True, "passed": True}))
    for path in replay_args.images:
        path.touch()
    with pytest.raises(RuntimeError, match="checkpoint load failed"):
        benchmark.run(replay_args)
    report = json.loads(report_path.read_text())
    assert not report["complete"]
    assert report["configurations"] == {}
    assert report["requested_configurations"] == ["dense", "static-graph"]
    assert [case["images"] for case in report["cases"]] == [str(path.resolve()) for path in replay_args.images]


def test_image_validation_precedes_model_loading(replay_args: argparse.Namespace) -> None:
    replay_args.model_id = "wan2.1-t2v-1.3b"
    with pytest.raises(ValueError, match="image replay requires an image-conditioned Wan pipeline"):
        benchmark.run(replay_args)
    replay_args.model_id = "wan2.2-ti2v-5b"
    with pytest.raises(FileNotFoundError) as failure:
        benchmark.run(replay_args)
    assert failure.value.args == (replay_args.images[0],)
    replay_args.model_id = "wan2.2-i2v-a14b"
    replay_args.images = None
    with pytest.raises(ValueError, match="requires --images"):
        benchmark.run(replay_args)


@pytest.fixture
def dual_receipt() -> dict[str, Any]:
    leaf = {
        "effective": {
            "compile": "compiled-cuda-graph-replayed",
            "cuda_graph_report": {
                "capture_on_miss": False,
                "replay": 4,
                "capture": 0,
                "eager": 0,
                "capture_failed": 0,
            },
        },
    }
    return {
        "runtime": {
            "dual_expert": {
                "route_calls": {"high-noise": 12, "low-noise": 4},
                "experts": {name: copy.deepcopy(leaf) for name in ("high-noise", "low-noise")},
            }
        }
    }


@pytest.mark.parametrize("failure", ("route", "eager", "capture", "compile"))
def test_dual_expert_receipts_reject_asymmetric_execution(failure: str, dual_receipt: dict[str, Any]) -> None:
    dual = dual_receipt["runtime"]["dual_expert"]
    options = {"cuda_graph": True, "compile": True}
    assert benchmark._execution_passed(dual_receipt, options)
    if failure == "route":
        dual["route_calls"]["low-noise"] = 0
    elif failure == "compile":
        dual["experts"]["low-noise"]["effective"]["compile"] = "configured-not-observed"
    else:
        dual["experts"]["low-noise"]["effective"]["cuda_graph_report"][failure] = 1
    assert not benchmark._execution_passed(dual_receipt, options)


def test_weight_dtype_option_does_not_require_graph_execution() -> None:
    assert benchmark._execution_passed({"effective": {}}, {"dit_weight_dtype": "bfloat16"})


def test_image_batch_replay_requires_images_before_loading(replay_args: argparse.Namespace) -> None:
    replay_args.request_batching = True
    replay_args.model_id = "wan2.2-i2v-a14b"
    replay_args.images = None
    with pytest.raises(ValueError, match="requires --images"):
        benchmark.run(replay_args)


@pytest.mark.parametrize("model_id", ["wan2.1-t2v-1.3b", "wan2.2-ti2v-5b", "wan2.2-i2v-a14b"])
def test_batch_replay_preserves_order_seeds_and_compares_to_singletons(
    replay_args: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
) -> None:
    import torch

    from worldfoundry.base_models.diffusion_model.contracts import DiffusionOutput, DiffusionRequest
    from worldfoundry.pipelines.wan.pipeline_wan_2p1_t2v import Wan2p1T2VPipeline
    from worldfoundry.pipelines.wan.pipeline_wan_2p2 import Wan2p2I2VA14BPipeline, Wan2p2Pipeline

    pipeline_class = {
        "wan2.1-t2v-1.3b": Wan2p1T2VPipeline,
        "wan2.2-ti2v-5b": Wan2p2Pipeline,
        "wan2.2-i2v-a14b": Wan2p2I2VA14BPipeline,
    }[model_id]
    replay_args.model_id = model_id
    replay_args.request_batching = True
    if model_id == "wan2.1-t2v-1.3b":
        replay_args.images = None
    else:
        for path in replay_args.images:
            path.touch()
    replay_args.configurations = []
    calls: list[tuple[int, ...]] = []
    denoiser = SimpleNamespace(model=torch.nn.Linear(1, 1), runtime_optimization_report=lambda: {})
    if model_id.endswith("a14b"):
        denoiser.high_noise = denoiser.low_noise = SimpleNamespace(model=denoiser.model)

    class Pipeline:
        native_pipeline = SimpleNamespace(components=SimpleNamespace(denoiser=denoiser))

        def warmup(self, values: list[Any]) -> tuple[()]:
            assert isinstance(values[0], DiffusionRequest)
            assert tuple(case.sampling.seed for case in values[1]) == (11, 29)
            return ()

        def __call__(self, *, seed: int, **kwargs: Any) -> dict[str, Any]:
            value = torch.full((1, 1, 1, 1, 1), float(seed))
            return {"sample": value, "latents": value}

        def run_batch(self, requests: tuple[DiffusionRequest, ...]) -> tuple[DiffusionOutput, ...]:
            calls.append(tuple(case.sampling.seed for case in requests))
            if replay_args.images:
                for case in requests:
                    image_index = 0 if case.sampling.seed == 11 else 1
                    assert case.inputs["images"] == str(replay_args.images[image_index].resolve())
            return tuple(
                DiffusionOutput(
                    sample=torch.full((1, 1, 1, 1, 1), float(case.sampling.seed)),
                    latents=torch.full((1, 1, 1, 1, 1), float(case.sampling.seed)),
                    metadata={
                        "seed": case.sampling.seed,
                        "request_batching": {
                            "execution": "batched",
                            "request_count": 2,
                            "sample_count": 2,
                            "request_index": index,
                            "denoiser_calls": 16,
                        },
                    },
                )
                for index, case in enumerate(requests)
            )

    monkeypatch.setattr(pipeline_class, "from_pretrained", lambda **kwargs: Pipeline())
    monkeypatch.setattr(
        "worldfoundry.runtime.performance.capture_runtime_fingerprint", lambda: SimpleNamespace(to_dict=lambda: {})
    )
    monkeypatch.setattr(
        "benchmarks.inference.plugin_diagnostics.compare_generation",
        lambda reference, output: {
            "passed": torch.equal(reference.latents, output.latents),
            "latent_bitwise_equal": torch.equal(reference.latents, output.latents),
        },
    )
    for name in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    report = benchmark.run(replay_args)
    assert report["complete"] and report["passed"]
    assert not report["performance_validated"]
    assert calls == [(11, 29), (29, 11), (11, 29)]
    assert len(report["configurations"]["dense"]["batches"]) == 3
    assert report["configurations"]["dense"]["batch_comparison"]["performance_validated"] is False


def test_batch_throughput_comparison_matches_the_same_two_requests() -> None:
    entry = {
        "requests": [{"case": 0, "seconds": 1}, {"case": 1, "seconds": 4}, {"case": 0, "seconds": 3}],
        "batches": [{"seconds": seconds, "peak_allocated_bytes": 100} for seconds in (4, 5, 6)],
        "warmup_seconds": 7,
    }
    result = benchmark._batch_comparison(entry)
    assert result["singleton_pair_seconds"] == 6
    assert result["median_batch_seconds"] == 5
    assert result["throughput_ratio"] == 1.2
    assert result["serial_requests_per_second"] == 2 / 6
    assert result["batched_requests_per_second"] == 2 / 5
    assert result["performance_validated"] is False
