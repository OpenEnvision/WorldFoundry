"""CPU-only adversarial checks for the frozen MG2 numerical reference gate."""

from __future__ import annotations

import builtins
import copy
import importlib.util
import json
from pathlib import Path

import pytest

_HELPER_PATH = Path(__file__).with_name("mg2_historical_reference.py")
_SPEC = importlib.util.spec_from_file_location("mg2_historical_reference_cpu", _HELPER_PATH)
_reference = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_reference)
_CASE = "split_seed7"
_BLOCK = ("contracts", _CASE, "blocks", 0)
_STAGE = (*_BLOCK, "denoise_and_refresh", 0)


@pytest.fixture(scope="module")
def accepted():
    report = json.loads(_reference.REFERENCE_PATH.read_bytes())
    table = report["cache_digest_table"]
    for contract in report["contracts"].values():
        for block in contract["blocks"]:
            for stage in block["denoise_and_refresh"]:
                stage["full_cache_sha256"] = {
                    family: [table[index] for index in indices]
                    for family, indices in stage["full_cache_sha256"].items()
                }
    return report


@pytest.fixture(scope="module")
def historical():
    return _reference.MG2HistoricalReference()


def _put(report, path, value):
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def test_all_six_accepted_receipts_pass_without_running_a_model(historical, accepted):
    historical.assert_complete_report(accepted)
    assert historical.receipt["sha256"] == _reference.REFERENCE_SHA256
    assert historical.receipt["reference_updated_by_execution"] is False


def test_reference_reader_does_not_import_a_tensor_runtime(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "numpy"}:
            raise AssertionError(f"reference gate imported an optional tensor runtime: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("mg2_reference_without_tensor_runtime", _HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.MG2HistoricalReference().receipt["bitwise_required"] is True


@pytest.mark.parametrize("kind", ("flow", "x0"))
def test_two_current_paths_sharing_the_same_model_bug_are_rejected(historical, accepted, kind):
    current = copy.deepcopy(accepted)
    stage = current["contracts"][_CASE]["blocks"][0]["denoise_and_refresh"][0]
    stage["baseline_output_sha256"][kind] = "0" * 64
    stage["optimized_output_sha256"][kind] = "0" * 64
    assert stage["baseline_output_sha256"] == stage["optimized_output_sha256"]
    assert stage["metrics"][kind]["max_abs"] == stage["metrics"][kind]["relative_l2"] == 0
    with pytest.raises(AssertionError, match=rf"historical MG2 reference mismatch.*{kind}"):
        historical.assert_complete_report(current)


@pytest.mark.parametrize("kind", ("latent", "video"))
def test_shared_final_output_bug_is_rejected(historical, accepted, kind):
    current = copy.deepcopy(accepted)
    block = current["contracts"][_CASE]["blocks"][0]
    block[f"baseline_{kind}_sha256"] = block[f"optimized_{kind}_sha256"] = "0" * 64
    with pytest.raises(AssertionError, match=rf"historical MG2 reference mismatch.*{kind}_sha256"):
        historical.assert_complete_report(current)


@pytest.mark.parametrize(("path", "value"), (
    ((*_STAGE, "full_cache_sha256", "self", 0, "k", "sha256"), "0" * 64),
    ((*_STAGE, "full_cache_sha256", "self", 0, "v", "shape"), [1]),
    ((*_STAGE, "full_cache_sha256", "mouse", 0, "k", "dtype"), "torch.float16"),
    ((*_STAGE, "full_cache_sha256", "keyboard", 0, "v", "sha256"), "1" * 64),
    ((*_STAGE, "full_cache_sha256", "cross", 0, "is_init"), False),
    ((*_STAGE, "cache_indices", "self", 0, "global_end_index"), 0),
    ((*_STAGE, "condition_sha256", "cond_concat"), "0" * 64),
    ((*_STAGE, "condition_sha256", "keyboard_cond"), "0" * 64),
    ((*_STAGE, "timestep"), [999.0, 999.0, 999.0]),
    ((*_STAGE, "metrics", "input", "max_abs"), 0.01),
    ((*_BLOCK, "controls"), {"interactions": ["forward"]}),
    ((*_BLOCK, "public_rgb8_sha256"), "0" * 64),
    ((*_BLOCK, "public_rgb8_shape"), [9, 640, 352, 3]),
    ((*_BLOCK, "recurrent_vae_cache_error", 0, "sha256"), "0" * 64),
    ((*_BLOCK, "independent_initial_noise_sha256"), "0" * 64),
    ((*_BLOCK, "continuous_global_rng_sha256"), "0" * 64),
    ((*_BLOCK, "latent_error", "relative_l2"), 0.001),
    ((*_BLOCK, "pixel_error", "max_abs"), 0.01),
    ((*_BLOCK, "pixel_byte_error", "max_abs"), 1),
    (("contracts", _CASE, "inter_step_noise", 0, "noise_sha256"), "0" * 64),
    (("conditioning_prerequisite", "resident_prefetch_sha256"), "0" * 64),
    (("conditioning_prerequisite", "canonical_condition_sha256", "visual_context"), "0" * 64),
    (("conditioning_weights", "vae_normalization_sha256", "mean"), "0" * 64),
    (("acceptance", "latent_relative_l2_max"), 0.001),
))
def test_state_and_rng_drift_is_rejected(historical, accepted, path, value):
    current = copy.deepcopy(accepted)
    _put(current, path, value)
    with pytest.raises(AssertionError, match="historical MG2 reference mismatch"):
        historical.assert_complete_report(current)


@pytest.mark.parametrize(("path", "value"), (
    (("torch_version",), "2.10.0+cu126"),
    (("cuda_version",), "12.8"),
    (("gpu",), "NVIDIA A100"),
    (("checkpoint_and_config_provenance", "dit", "sha256"), "0" * 64),
    (("checkpoint_and_config_provenance", "vae", "bytes"), 1),
    (("input_image", "rgb_sha256"), "0" * 64),
    (("contracts", _CASE, "geometry"), [1, 16, 36, 44, 81]),
    (("contracts", _CASE, "schedule"), [1000.0, 900.0, 700.0]),
    (("contracts", _CASE, "seed"), 8),
    (("contracts", _CASE, "quantization"), "fp8"),
    (("contracts", _CASE, "compile"), True),
    (("contracts", _CASE, "bitwise_required"), 1),
))
def test_incompatible_backend_weights_or_geometry_is_rejected(historical, accepted, path, value):
    current = copy.deepcopy(accepted)
    _put(current, path, value)
    with pytest.raises(AssertionError, match="historical MG2 reference mismatch"):
        historical.assert_complete_report(current)


def test_reference_payload_cannot_be_replaced_by_current_evidence(tmp_path, accepted):
    altered = copy.deepcopy(accepted)
    altered["contracts"][_CASE]["blocks"][0]["optimized_latent_sha256"] = "0" * 64
    replacement = tmp_path / "replacement.json"
    replacement.write_text(json.dumps(altered))
    with pytest.raises(AssertionError, match="historical MG2 reference was altered"):
        _reference.MG2HistoricalReference(replacement)


def test_reference_whitespace_tampering_is_rejected(tmp_path):
    replacement = tmp_path / "altered.json"
    replacement.write_bytes(_reference.REFERENCE_PATH.read_bytes() + b"\n")
    with pytest.raises(AssertionError, match="historical MG2 reference was altered"):
        _reference.MG2HistoricalReference(replacement)


@pytest.mark.parametrize("removal", ("case", "block", "stage", "cache_layer", "weight", "required_hash"))
def test_incomplete_evidence_is_rejected(historical, accepted, removal):
    current = copy.deepcopy(accepted)
    contract = current["contracts"][_CASE]
    stage = contract["blocks"][0]["denoise_and_refresh"][0]
    if removal == "case":
        del current["contracts"][_CASE]
    elif removal == "block":
        contract["blocks"].pop()
    elif removal == "stage":
        contract["blocks"][0]["denoise_and_refresh"].pop()
    elif removal == "cache_layer":
        stage["full_cache_sha256"]["self"].pop()
    elif removal == "weight":
        del current["checkpoint_and_config_provenance"]["vae"]
    else:
        del stage["optimized_output_sha256"]
    with pytest.raises(AssertionError, match="historical MG2 reference"):
        historical.assert_complete_report(current)


def test_machine_path_relocation_does_not_change_numerical_identity(historical, accepted):
    current = copy.deepcopy(accepted)
    for record in current["checkpoint_and_config_provenance"].values():
        record["path"] = "/different/local/checkpoint/location"
    current["source_sha256"] = {"candidate/code.py": "1" * 64}
    current["visible_device"] = 1
    current["cuda_visible_devices"] = "3"
    historical.assert_complete_report(current)


def test_unknown_case_and_unaccepted_block_are_rejected(historical, accepted):
    contract = accepted["contracts"][_CASE]
    with pytest.raises(AssertionError, match="no accepted case"):
        historical.assert_contract_identity("unrecorded_case", contract)
    with pytest.raises(AssertionError, match="no accepted block"):
        historical.assert_block(_CASE, 12, contract["blocks"][0])


def test_numeric_signatures_are_derived_from_the_frozen_reference(historical, accepted):
    measured = historical.numeric_signatures(accepted)
    expected = historical.expected_numeric_signatures()
    assert measured == expected
    assert historical.expected_signatures == expected
    assert len(measured["conditioning"]) == 64
    assert set(measured["contracts"]) == set(accepted["contracts"])
    assert all(len(value) == 64 for value in measured["contracts"].values())
    measured["contracts"][_CASE] = "0" * 64
    assert historical.expected_numeric_signatures()["contracts"][_CASE] != measured["contracts"][_CASE]
    assert historical.required_source_paths == tuple(sorted(accepted["source_sha256"]))


def test_numeric_signatures_exclude_execution_and_machine_metadata(historical, accepted):
    current = copy.deepcopy(accepted)
    current["source_sha256"] = {"candidate.py": "0" * 64}
    current["source_base_revision"] = "a" * 40
    current["test_source_sha256"] = "1" * 64
    current["gpu_total_bytes"] = 1
    current["cuda_visible_devices"] = "3"
    current["conditioning_prerequisite"]["configured"]["realtime_metrics"] = {"total_ms": 1.0}
    for record in current["checkpoint_and_config_provenance"].values():
        record["path"] = "/relocated/checkpoint"
    for contract in current["contracts"].values():
        contract["peak_cuda_allocated_bytes"] = 1
        contract["qkv_receipt"] = {"execution": "different-execution-metadata"}
        contract["overlap_receipt"] = {"calls": 1}
    assert historical.numeric_signatures(current) == historical.expected_numeric_signatures()


def test_numeric_signatures_cannot_hide_a_shared_numeric_bug(historical, accepted):
    current = copy.deepcopy(accepted)
    block = current["contracts"][_CASE]["blocks"][0]
    block["baseline_video_sha256"] = block["optimized_video_sha256"] = "0" * 64
    with pytest.raises(AssertionError, match="historical MG2 reference mismatch"):
        historical.numeric_signatures(current)
