"""Quality/execution qualification must reject hidden or unexecuted changes."""

from types import SimpleNamespace

import pytest
import torch

from benchmarks.inference.plugin_diagnostics import (
    attention_scope_receipts,
    compare_generation,
    cuda_device_admission,
    file_metadata,
    qualify_execution,
)


def _output(batch=1, frames=2):
    return SimpleNamespace(sample=torch.zeros(batch, 3, frames, 8, 8), latents=torch.ones(batch, 16, 1, 1, 1))


def test_exact_generation_passes_with_strict_json_metrics():
    import json

    output = _output()
    report = compare_generation(output, output)
    assert report["passed"] and report["video_bitwise_equal"] and report["latent_bitwise_equal"]
    assert report["video_psnr_db"] == "infinity"
    json.dumps(report, allow_nan=False)


def test_bad_batch_member_cannot_be_hidden_by_large_batch():
    reference, candidate = _output(batch=128), _output(batch=128)
    candidate.latents[0] *= 1.1
    report = compare_generation(reference, candidate)
    assert report["latent_relative_l2"] > 0.09
    assert not report["passed"]


def test_one_corrupted_frame_cannot_be_hidden_by_average_ssim():
    reference, candidate = _output(frames=100), _output(frames=100)
    candidate.sample[:, :, 0] = 0.2
    report = compare_generation(reference, candidate)
    assert report["video_ssim"] < 0.98
    assert not report["passed"]


@pytest.mark.parametrize("field", ["sample", "latents"])
def test_nonfinite_output_is_rejected(field):
    reference, candidate = _output(), _output()
    getattr(candidate, field).flatten()[0] = float("nan")
    assert compare_generation(reference, candidate) == {"finite": False, "passed": False, "reason": "nonfinite-output"}


def test_changed_geometry_is_rejected():
    with pytest.raises(ValueError, match="geometry"):
        compare_generation(_output(frames=2), _output(frames=3))


def test_unexecuted_precision_rejects_otherwise_executed_combination():
    options = {"sana_block_fusion": True, "selective_fp8": {"include": ["blocks.*.mlp.*"]}}
    evidence = {
        "kernels": {"dispatches": [{"op": "scale_shift", "accelerated": True}]},
        "quantization": {"effective": "dense", "low_precision_kernel_calls": 0, "dense_fallback_calls": 20},
    }
    report = qualify_execution(options, evidence)
    assert report["plugins"] == {"sana_block_fusion": True, "selective_fp8": False}
    assert not report["passed"]


def test_cache_combination_requires_actual_skip_when_threshold_positive():
    options = {"sana_block_fusion": True, "easycache": {"threshold": 0.05}}
    evidence = {
        "kernels": {"dispatches": [{"op": "scale_shift", "accelerated": True}]},
        "denoiser": {"feature_cache": {"dense_block_calls": 20, "skipped_block_calls": 0}},
    }
    assert not qualify_execution(options, evidence)["passed"]
    options["easycache"]["threshold"] = 0.0
    assert qualify_execution(options, evidence)["passed"]


@pytest.mark.parametrize(("threshold", "counter"), [(0.05, "skipped_block_calls"), (0.0, "dense_block_calls")])
def test_native_wan_cache_runtime_schema_qualifies_actual_execution(threshold, counter):
    options = {"easycache": {"threshold": threshold}}
    evidence = {"denoiser": {"runtime": {"feature_cache": {counter: 20}}}}
    assert qualify_execution(options, evidence)["passed"]


@pytest.mark.parametrize("native_cache", [None, {}, {"skipped_block_calls": 0}])
def test_native_cache_receipt_takes_precedence_over_stale_top_level_receipt(native_cache):
    evidence = {
        "denoiser": {
            "runtime": {"feature_cache": native_cache},
            "feature_cache": {"skipped_block_calls": 20},
        }
    }
    assert not qualify_execution({"easycache": {"threshold": 0.05}}, evidence)["passed"]


def test_attention_fallback_is_not_certified_as_requested_provider():
    options = {"attention_policy": {"self": "sol_attn", "cross": "torch"}}
    evidence = {
        "attention": {"sol_attn": {"successes": 1, "fallbacks": 1}, "torch": {"successes": 2}},
        "attention_scopes": {
            scope: {
                "requested": backend,
                "resolved": backend,
                "configured_modules": 1,
                "observed_modules": 1,
                "executed_modules": 1,
                "successful_calls": 1,
                "backend_mismatches": 0,
            }
            for scope, backend in options["attention_policy"].items()
        },
    }
    assert not qualify_execution(options, evidence)["passed"]
    evidence["attention"]["sol_attn"]["fallbacks"] = 0
    assert qualify_execution(options, evidence)["passed"]
    evidence["attention_scopes"]["cross"]["successful_calls"] = 0
    assert not qualify_execution(options, evidence)["passed"]


@pytest.mark.parametrize("field", ["dense_compute_calls", "dense_fallback_calls", "fallback_reasons"])
def test_partly_dense_precision_is_not_qualified(field):
    options = {"selective_fp8": {"include": ["blocks.*.ffn.0"]}}
    counters = {
        "low_precision_kernel_calls": 10,
        "dense_compute_calls": 0,
        "dense_fallback_calls": 0,
        "fallback_reasons": [],
    }
    assert qualify_execution(options, {"quantization": counters})["passed"]
    counters[field] = ["unsupported shape"] if field == "fallback_reasons" else 1
    assert not qualify_execution(options, {"quantization": counters})["passed"]


def test_text_only_fusion_does_not_qualify_image_route():
    options = {"wan_cross_kv_fusion": {"include_image": True}}
    evidence = {"cross_kv_fusion": {"projection_calls": 3, "text_packed_calls": 3, "image_packed_calls": 0}}
    assert not qualify_execution(options, evidence)["passed"]
    evidence["cross_kv_fusion"]["image_packed_calls"] = 1
    assert qualify_execution(options, evidence)["passed"]


def test_scope_pilot_hooks_observe_success_and_are_removed_after_error():
    from worldfoundry.base_models.diffusion_model.models.networks.wan.model import CrossAttention

    model = torch.nn.Module()
    model.cross = CrossAttention(128, 1).eval()
    model.cross.attn.attention_backend = "torch"
    installed = [
        {
            "name": "attention_policy",
            "scopes": {"cross": {"requested": "auto", "resolved": "torch", "configured_modules": 1}},
        }
    ]
    with pytest.raises(RuntimeError, match="pilot error"):
        with attention_scope_receipts(model, installed) as receipts, torch.inference_mode():
            value = torch.randn(1, 2, 128)
            model.cross.attn(value, value, value)
            assert receipts["cross"]["executed_modules"] == receipts["cross"]["successful_calls"] == 1
            assert qualify_execution(
                {"attention_policy": {"cross": "auto"}},
                {"attention_scopes": receipts, "attention": {"torch": {"successes": 1}}},
            )["passed"]
            raise RuntimeError("pilot error")
    assert not model.cross.attn._forward_hooks


def test_unknown_plugin_has_no_implicit_execution_certification():
    assert not qualify_execution({"future_kernel": True}, {})["passed"]


def _timing_evidence():
    configuration = {
        scope: {"requested": backend, "resolved": backend, "configured_modules": 2, "options": {}}
        for scope, backend in (("self", "flash_attention_2"), ("cross", "torch"))
    }
    return {
        "attention_configuration": configuration,
        "attention_scopes": {
            scope: row | {"observed_modules": 2, "executed_modules": 2, "successful_calls": 4, "backend_mismatches": 0}
            for scope, row in configuration.items()
        },
        "attention": {"flash_attention_2": {"successes": 4}, "torch": {"successes": 4}},
        "quantization": {
            "low_precision_kernel_calls": 8,
            "dense_compute_calls": 0,
            "dense_fallback_calls": 0,
            "fallback_reasons": [],
        },
    }


@pytest.mark.parametrize(
    "failure", [None, "errors", "fallbacks", "quarantined_skips", "missing_calls", "dense", "scope"]
)
def test_timed_execution_uses_fresh_receipts_with_untimed_scope_proof(failure):
    from benchmarks.inference.native_plugins import _qualify_timed_execution

    pilot, timed = _timing_evidence(), _timing_evidence()
    timed["attention_scopes"] = {}  # Per-module hooks are intentionally absent from timing.
    options = {
        "attention_policy": {"self": "flash_attention_2", "cross": "torch"},
        "selective_fp8": {"include": ["blocks.29.ffn.0"]},
    }
    if failure in {"errors", "fallbacks", "quarantined_skips"}:
        timed["attention"]["flash_attention_2"][failure] = 1
    elif failure == "missing_calls":
        timed["attention"]["flash_attention_2"]["successes"] = 3
    elif failure == "dense":
        timed["quantization"]["dense_fallback_calls"] = 1
    elif failure == "scope":
        timed["attention_configuration"]["self"]["resolved"] = "torch"
    assert _qualify_timed_execution(options, timed, pilot)["passed"] is (failure is None)
    assert timed["attention_scopes"] == {}


def test_timed_dense_reference_cannot_hide_provider_fallback():
    from benchmarks.inference.native_plugins import _qualify_timed_execution

    pilot = {"attention": {"torch": {"successes": 10}}}
    timed = {"attention": {"torch": {"successes": 10}, "flash_attention_2": {"fallbacks": 1}}}
    assert not _qualify_timed_execution({}, timed, pilot)["passed"]


@pytest.mark.parametrize("relative", [False, True])
def test_input_metadata_detects_changes_without_reading_file_contents(tmp_path, monkeypatch, relative):
    from pathlib import Path

    monkeypatch.chdir(tmp_path)
    root = Path(".") if relative else tmp_path
    weight = root / "weights.bin"
    weight.write_bytes(b"weights")

    def unexpected_read(*args, **kwargs):
        raise AssertionError("metadata collection must not read weight contents")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", unexpected_read)
        before = file_metadata([weight], root)
    weight.write_bytes(b"updated weights")
    after = file_metadata([weight], root)
    assert before != after
    assert before[0]["bytes"] == 7 and after[0]["bytes"] == 15
    assert "sha256" not in after[0]


@pytest.mark.parametrize("mode", ["timing_fallback", "quality_tradeoff", "nonfinite"])
def test_native_cli_separates_timing_execution_from_quality(tmp_path, monkeypatch, mode):
    import json

    from benchmarks.inference import native_plugins

    class Pipeline:
        def __init__(self):
            self.enabled = False
            self.candidate_calls = 0
            self.components = SimpleNamespace(denoiser=self, decoder=self)
            self.model = torch.nn.Module()

        def __call__(self, request):
            self.candidate_calls += int(self.enabled)
            output = _output()
            if self.enabled and mode == "quality_tradeoff":
                output.latents *= 1.1
            elif self.enabled and mode == "nonfinite":
                output.latents.fill_(float("nan"))
            return output

        def runtime_optimization_report(self):
            return {
                "runtime": {
                    "feature_cache": {
                        "skipped_block_calls": int(
                            self.enabled and (self.candidate_calls == 1 or mode != "timing_fallback")
                        ),
                    }
                }
            }

    pipeline = Pipeline()

    def install(*args):
        pipeline.enabled = True
        return SimpleNamespace(
            report=lambda: {"installed": [{"name": "easycache"}]},
            uninstall=lambda: setattr(pipeline, "enabled", False),
        )

    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps({"cache": {"easycache": {"threshold": 0.05}}}))
    output = tmp_path / "results"
    monkeypatch.setattr(
        "sys.argv",
        [
            "native_plugins",
            "--model",
            "wan2.1-t2v-1.3b",
            "--assets",
            str(tmp_path),
            "--candidates",
            str(candidates),
            "--out",
            str(output),
            "--seeds",
            "42",
        ],
    )
    monkeypatch.setattr(native_plugins, "_assets", lambda *args: ({"tokenizer": str(tmp_path)}, []))
    monkeypatch.setattr(native_plugins, "file_metadata", lambda *args: [])
    monkeypatch.setattr(native_plugins, "capture_runtime_fingerprint", lambda **kwargs: SimpleNamespace(to_dict=dict))
    monkeypatch.setattr(native_plugins, "cuda_device_admission", lambda: {"timing_qualified": True})
    monkeypatch.setattr(native_plugins, "install_diffusion_accelerations", install)
    monkeypatch.setattr(native_plugins, "attention_provider_runtime_report", lambda: {"torch": {"successes": 1}})
    monkeypatch.setattr(native_plugins, "quantization_runtime_report", lambda model: None)
    monkeypatch.setattr(native_plugins.NativeDiffusionPipeline, "from_pretrained", lambda *args, **kwargs: pipeline)
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    native_plugins.main()
    result = json.loads((output / "results.json").read_text())
    row = result["cases"][0]
    assert row["execution_gate"]["passed"]
    assert result["selection"] == "no_validated_speedup"
    if mode == "nonfinite":
        assert row["status"] == "rejected_nonfinite"
        assert "speedup_median" not in row
        assert pipeline.candidate_calls == 1
    elif mode == "quality_tradeoff":
        assert row["status"] == "measured_quality_tradeoff"
        assert not row["quality"]["passed"]
        assert row["quality"]["video_bitwise_equal"]
        assert "speedup_median" in row
        assert pipeline.candidate_calls == 4
    else:
        assert row["status"] == "rejected_timing_execution"
        assert "speedup_median" not in row
        assert all(
            receipt["execution_gate"]["passed"] is not receipt["candidate_enabled"]
            for receipt in row["timing_execution_receipts"]
        )
    if mode != "nonfinite":
        assert len(row["timing_execution_receipts"]) == 6


@pytest.mark.parametrize(
    ("output", "qualified"),
    [
        ("GPU-A, 720, 100\n", True),
        ("GPU-A, 720, 100\nGPU-A, 721, 654\n", False),
        ("GPU-B, 720, 100\nGPU-A, 721, 654\n", True),
        ("GPU-A, 721, 654\n", False),
        ("No running processes found\n", False),
        ("GPU-A, [Not Supported], 100\n", False),
    ],
)
def test_ambiguous_or_shared_device_cannot_qualify_performance(monkeypatch, output, qualified):
    monkeypatch.setattr("benchmarks.inference.plugin_diagnostics.os.getpid", lambda: 720)
    monkeypatch.setattr(
        "benchmarks.inference.plugin_diagnostics.subprocess.check_output", lambda *args, **kwargs: output
    )
    assert cuda_device_admission()["timing_qualified"] is qualified


@pytest.mark.parametrize("failure", [None, "missing_projection", "missing_module", "missing_calls", "fallback"])
def test_mha_gate_requires_every_projection_and_real_quantized_calls(failure):
    options = {"optimized_mha": {"self": {"fusion": "qkv", "projection_precision": "fp8_e4m3"}}}
    projection = {
        "calls": 4,
        "precision": "fp8_e4m3",
        "quantization": {
            "low_precision_kernel_calls": 4,
            "dense_compute_calls": 0,
            "dense_fallback_calls": 0,
        },
    }
    runtime = {
        "blocks.0.self_attn": {
            "calls": 4,
            "sdpa_calls": 4,
            "projections": {
                "qkv": projection,
                "o": {"calls": 4, "precision": "native", "quantization": None},
            },
        }
    }
    evidence = {
        "denoiser": {
            "accelerations": {
                "installed": [
                    {
                        "name": "optimized_mha",
                        "modules": list(runtime),
                        "runtime": runtime,
                        "projection_modules": {"blocks.0.self_attn": ["qkv", "o"]},
                    }
                ]
            }
        }
    }
    evidence["denoiser"] = {"effective": evidence["denoiser"]}
    if failure == "missing_projection":
        runtime["blocks.0.self_attn"]["projections"].pop("o")
    elif failure == "missing_module":
        runtime.clear()
    elif failure == "missing_calls":
        projection["calls"] = 0
    elif failure == "fallback":
        projection["quantization"]["dense_fallback_calls"] = 1
    assert qualify_execution(options, evidence)["passed"] is (failure is None)


@pytest.mark.parametrize("failure", [None, "unexecuted", "missing", "clipped", "fallback"])
def test_fp8_codec_gate_rejects_incomplete_execution(failure):
    from benchmarks.inference.lightvae_fp8 import qualify_encoder

    receipt = {
        "enabled": True,
        "layers": {"conv": {"kernel_calls": 2}},
        "clipped_input_operands": 0,
        "dense_fallback_calls": 0,
    }
    if failure == "unexecuted":
        receipt["layers"]["conv"]["kernel_calls"] = 0
    elif failure == "missing":
        receipt["layers"].clear()
    elif failure == "clipped":
        receipt["clipped_input_operands"] = 1
    elif failure == "fallback":
        receipt["dense_fallback_calls"] = 1
    assert qualify_encoder(receipt, ["conv"])["passed"] is (failure in (None, "clipped"))


def test_svdquant_gate_requires_packed_execution_in_every_selected_layer():
    options = {"svdquant": {"artifact": "offline.pt"}}
    report = {
        "low_precision_kernel_calls": 2,
        "dense_compute_calls": 0,
        "dense_fallback_calls": 0,
        "fallback_reasons": [],
        "layer_reports": [{"module": "a", "native_packed_int4_calls": 2}],
    }
    installed = {"name": "svdquant", "modules": ["a", "b"]}
    evidence = {"quantization": report, "denoiser": {"effective": {"accelerations": {"installed": [installed]}}}}
    assert not qualify_execution(options, evidence)["passed"]
    report["layer_reports"].append({"module": "b", "native_packed_int4_calls": 0})
    assert not qualify_execution(options, evidence)["passed"]
    report["layer_reports"][1]["native_packed_int4_calls"] = 2
    assert qualify_execution(options, evidence)["passed"]
