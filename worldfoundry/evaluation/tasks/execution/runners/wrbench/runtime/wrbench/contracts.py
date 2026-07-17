"""Minimal execution-contract helpers for contract-driven adapters.

Contract payloads live inline in each model registry JSON under
``execution_contract``. This is the WRBench-native contract layer for
model-specific generation inputs.
``inference_contracts.json`` for compile-time adapter needs.
"""

from __future__ import annotations

from typing import Any

from wrbench.registry import RegistryError, canonical_model_key, model_record


class ContractError(ValueError):
    pass


def require_execution_contract(model_name: str) -> dict[str, Any]:
    key = canonical_model_key(model_name)
    record = model_record(key)
    contract = getattr(record, "execution_contract", None)
    if not isinstance(contract, dict) or not contract:
        raise ContractError(f"Missing execution_contract for {key}")
    return contract


def require_mapping(node: dict[str, Any], field: str) -> dict[str, Any]:
    value = node.get(field)
    if not isinstance(value, dict):
        raise ContractError(f"Missing required object field: {field}")
    return value


def require_sequence(node: dict[str, Any], field: str) -> list[Any]:
    value = node.get(field)
    if not isinstance(value, list):
        raise ContractError(f"Missing required list field: {field}")
    return value


def require_str(node: dict[str, Any], field: str) -> str:
    value = node.get(field)
    if not isinstance(value, str) or not value:
        raise ContractError(f"Missing required string field: {field}")
    return value


def require_int(node: dict[str, Any], field: str) -> int:
    value = node.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractError(f"Missing required integer field: {field}")
    return int(value)


def require_float(node: dict[str, Any], field: str) -> float:
    value = node.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ContractError(f"Missing required numeric field: {field}")
    return float(value)


def require_bool(node: dict[str, Any], field: str) -> bool:
    value = node.get(field)
    if not isinstance(value, bool):
        raise ContractError(f"Missing required boolean field: {field}")
    return bool(value)


def _stringify_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    raise ContractError(f"Unsupported CLI value type: {type(value).__name__}")


def build_command_template(execution: dict[str, Any], *, values: dict[str, Any]) -> list[str]:
    command = [_stringify_cli_value(part) for part in require_sequence(execution, "command_prefix")]
    command.append(require_str(execution, "entrypoint"))

    merged_values = dict(require_mapping(execution, "runtime_parameters"))
    merged_values.update(values)

    for item in require_sequence(execution, "cli_flag_parameters"):
        if not isinstance(item, dict):
            raise ContractError("cli_flag_parameters entries must be objects")
        flag = require_str(item, "flag")
        parameter = require_str(item, "parameter")
        expected = item.get("required_value")
        actual = merged_values.get(parameter)
        if actual is None:
            raise ContractError(f"Missing CLI flag parameter value: {parameter}")
        if expected is not None and actual != expected:
            raise ContractError(f"CLI flag {flag} requires {parameter}={expected!r}, got {actual!r}")
        if actual:
            command.append(flag)

    for item in require_sequence(execution, "cli_parameter_order"):
        if not (isinstance(item, list) and len(item) == 2 and all(isinstance(part, str) for part in item)):
            raise ContractError("cli_parameter_order entries must be [flag, parameter] string pairs")
        flag, parameter = item
        if parameter not in merged_values:
            raise ContractError(f"Missing CLI parameter value: {parameter}")
        command.extend([flag, _stringify_cli_value(merged_values[parameter])])

    return command
