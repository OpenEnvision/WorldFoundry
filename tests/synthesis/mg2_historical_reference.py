"""Read-only exact receipts from an independently accepted MG2 revision.

The reference is pinned by its complete file digest. New execution evidence
cannot replace it, and no path in this module writes or recalibrates expected
outputs. The hardware and software identity is deliberately narrow because
raw tensor bytes are backend-specific.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REFERENCE_REVISION = "07686790f5421943362fda2523065841ee7a1e44"
REFERENCE_SHA256 = "556b26ff8db503b00275c6492002d642d2e1cd82cd24cbc60707a22463030698"
REFERENCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs/fumadocs/public/measurements/mg2-conditioned-trajectory-2026-10-03.json"
)

_BACKEND_FIELDS = ("torch_version", "cuda_version", "gpu", "model_parameter_count")
_CONTRACT_FIELDS = (
    "action_layers", "ambient_autocast", "bitwise_required", "blocks_count", "compile",
    "conditioning_weight_dtype", "context_noise", "continuous_rng_without_per_block_reseed",
    "decoded_resolution", "geometry", "latent_frame_starts", "local_attention_window",
    "model_layers", "overlap_vae_decode", "qkv_strategy", "quantization", "rgb_frames",
    "rollovers", "schedule", "seed", "vae_decode_weight_dtype",
)
_PREREQUISITE_FIELDS = (
    "bitwise_required", "canonical_condition_sha256", "canonical_latent_frames", "canonical_rgb_frames",
    "future_reused_tail_error", "passed", "prefetch_blocks", "prefetch_latent_frames", "prefetch_rgb_frames",
    "prefix_error", "resident_prefetch_sha256", "visual_context_error",
)
_STAGE_FIELDS = (
    "stage", "timestep", "condition_sha256", "cache_indices", "full_cache_sha256",
    "baseline_output_sha256", "optimized_output_sha256", "metrics",
)
_BLOCK_FIELDS = (
    "latent_start", "controls", "baseline_latent_sha256", "optimized_latent_sha256",
    "baseline_video_sha256", "optimized_video_sha256", "public_rgb8_shape", "public_rgb8_sha256",
    "recurrent_vae_cache_error", "independent_initial_noise_sha256", "continuous_global_rng_sha256",
    "latent_error", "pixel_error", "pixel_byte_error",
)


def _same(actual, expected, label):
    if type(actual) is not type(expected):
        raise AssertionError(f"historical MG2 reference mismatch at {label}: value type changed")
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"historical MG2 reference mismatch at {label}: field set changed")
        for key, value in expected.items():
            _same(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"historical MG2 reference mismatch at {label}: sequence length changed")
        for index, value in enumerate(expected):
            _same(actual[index], value, f"{label}[{index}]")
    elif actual != expected:
        raise AssertionError(f"historical MG2 reference mismatch at {label}: expected {expected!r}, got {actual!r}")


def _fields(actual, expected, fields, label):
    for field in fields:
        if field not in actual:
            raise AssertionError(f"historical MG2 reference missing required field {label}.{field}")
        _same(actual[field], expected[field], f"{label}.{field}")


class MG2HistoricalReference:
    """Reject common regressions even when both current implementations agree."""

    def __init__(self, path=REFERENCE_PATH):
        payload = Path(path).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != REFERENCE_SHA256:
            raise AssertionError(f"historical MG2 reference was altered: SHA256 {digest}")
        self._reference = json.loads(payload)
        if self._reference["schema_version"] != 2:
            raise AssertionError("historical MG2 reference has unsupported cache encoding")
        table = self._reference["cache_digest_table"]
        for contract in self._reference["contracts"].values():
            for block in contract["blocks"]:
                for stage in block["denoise_and_refresh"]:
                    stage["full_cache_sha256"] = {
                        family: [table[index] for index in indices]
                        for family, indices in stage["full_cache_sha256"].items()
                    }

    @property
    def receipt(self):
        return {"revision": REFERENCE_REVISION, "sha256": REFERENCE_SHA256,
                "bitwise_required": True, "reference_updated_by_execution": False}

    @property
    def required_source_paths(self):
        return tuple(sorted(self._reference["source_sha256"]))

    def assert_backend_and_weights(self, evidence):
        _fields(evidence, self._reference, _BACKEND_FIELDS, "backend")
        current = evidence.get("checkpoint_and_config_provenance", {})
        expected = self._reference["checkpoint_and_config_provenance"]
        _same(sorted(current), sorted(expected), "checkpoint_and_config_provenance.files")
        for name, record in expected.items():
            _fields(current[name], record, ("bytes", "sha256"), f"checkpoint_and_config_provenance.{name}")

    def assert_conditioning(self, evidence):
        self.assert_backend_and_weights(evidence)
        _fields(evidence, self._reference, ("input_image", "conditioning_weights", "acceptance"), "conditioning")
        _fields(evidence.get("conditioning_prerequisite", {}), self._reference["conditioning_prerequisite"],
                _PREREQUISITE_FIELDS, "conditioning_prerequisite")

    def assert_contract_identity(self, key, contract):
        if key not in self._reference["contracts"]:
            raise AssertionError(f"historical MG2 reference has no accepted case {key!r}")
        _fields(contract, self._reference["contracts"][key], _CONTRACT_FIELDS, key)

    def assert_block(self, key, block_index, block):
        if key not in self._reference["contracts"]:
            raise AssertionError(f"historical MG2 reference has no accepted case {key!r}")
        accepted = self._reference["contracts"][key]["blocks"]
        if not 0 <= block_index < len(accepted):
            raise AssertionError(f"historical MG2 reference has no accepted block {key}[{block_index}]")
        expected = accepted[block_index]
        label = f"{key}.blocks[{block_index}]"
        _fields(block, expected, _BLOCK_FIELDS, label)
        actual_stages = block.get("denoise_and_refresh", [])
        expected_stages = expected["denoise_and_refresh"]
        _same(len(actual_stages), len(expected_stages), f"{label}.denoise_and_refresh.length")
        for index, (actual, original) in enumerate(zip(actual_stages, expected_stages)):
            _fields(actual, original, _STAGE_FIELDS, f"{label}.denoise_and_refresh[{index}]")

    def assert_complete_contract(self, key, contract):
        self.assert_contract_identity(key, contract)
        original = self._reference["contracts"][key]
        _same(len(contract.get("blocks", [])), len(original["blocks"]), f"{key}.blocks.length")
        for index, block in enumerate(contract["blocks"]):
            self.assert_block(key, index, block)
        _fields(contract, original, ("inter_step_noise", "passed"), key)

    def assert_complete_report(self, evidence):
        self.assert_conditioning(evidence)
        contracts = evidence.get("contracts", {})
        _same(sorted(contracts), sorted(self._reference["contracts"]), "contracts.cases")
        for key, contract in contracts.items():
            self.assert_complete_contract(key, contract)

    def _numeric_signatures(self, evidence):
        context = {
            "backend": {key: evidence[key] for key in _BACKEND_FIELDS},
            "checkpoint_and_config_provenance": {
                name: {key: record[key] for key in ("bytes", "sha256")}
                for name, record in evidence["checkpoint_and_config_provenance"].items()
            },
            "input_image": evidence["input_image"],
            "conditioning_weights": evidence["conditioning_weights"],
            "acceptance": evidence["acceptance"],
            "conditioning_prerequisite": {
                key: evidence["conditioning_prerequisite"][key] for key in _PREREQUISITE_FIELDS
            },
        }
        encoded_context = json.dumps(
            {"reference_sha256": REFERENCE_SHA256, **context},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        signatures = {"conditioning": hashlib.sha256(encoded_context).hexdigest(), "contracts": {}}
        for name, contract in evidence["contracts"].items():
            numeric = {
                "reference_sha256": REFERENCE_SHA256,
                "context": context,
                "contract": {key: contract[key] for key in _CONTRACT_FIELDS},
                "inter_step_noise": contract["inter_step_noise"],
                "blocks": [{
                    **{key: block[key] for key in _BLOCK_FIELDS},
                    "denoise_and_refresh": [
                        {key: stage[key] for key in _STAGE_FIELDS}
                        for stage in block["denoise_and_refresh"]
                    ],
                } for block in contract["blocks"]],
            }
            canonical = json.dumps(numeric, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            signatures["contracts"][name] = hashlib.sha256(canonical).hexdigest()
        return signatures

    def numeric_signatures(self, evidence):
        """Validate complete raw evidence and return one stable digest per case.

        Source manifests and execution/JUnit provenance belong to the calling
        artifact verifier. These digests bind numerical state, conditioning,
        checkpoints, backend, geometry and RNG while excluding machine paths,
        source revisions, clocks, peak memory and optimization call receipts.
        """
        self.assert_complete_report(evidence)
        return self._numeric_signatures(evidence)

    def expected_numeric_signatures(self):
        """Compute expected digests only from the pinned, read-only reference."""
        return self._numeric_signatures(self._reference)

    @property
    def expected_signatures(self):
        return self.expected_numeric_signatures()
