"""Safe extraction of XR1 normalization metadata from a staged checkpoint."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def _safe_eval(node: ast.AST, values: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in values:
            return values[node.id]
        return None
    if isinstance(node, ast.Dict):
        result: dict[Any, Any] = {}
        for key, value in zip(node.keys, node.values, strict=True):
            evaluated_value = _safe_eval(value, values)
            if key is None:
                if isinstance(evaluated_value, Mapping):
                    result.update(evaluated_value)
                continue
            result[_safe_eval(key, values)] = evaluated_value
        return result
    if isinstance(node, ast.List):
        return [_safe_eval(value, values) for value in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval(value, values) for value in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_eval(node.operand, values)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
        left = _safe_eval(node.left, values)
        right = _safe_eval(node.right, values)
        return left + right if isinstance(node.op, ast.Add) else left * right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name == "dict":
            result: dict[str, Any] = {}
            for argument in node.args:
                evaluated = _safe_eval(argument, values)
                if isinstance(evaluated, Mapping):
                    result.update(dict(evaluated))
            for keyword in node.keywords:
                evaluated = _safe_eval(keyword.value, values)
                if keyword.arg is None:
                    if isinstance(evaluated, Mapping):
                        result.update(evaluated)
                else:
                    result[keyword.arg] = evaluated
            return result
        if name in {"list", "tuple"} and len(node.args) <= 1 and not node.keywords:
            items = [] if not node.args else _safe_eval(node.args[0], values)
            return list(items) if name == "list" else tuple(items)
    # Unsupported calls, imports, and attributes are intentionally not
    # evaluated.  They are retained as ``None`` so an unrelated path or
    # training callback cannot hide otherwise-literal normalization blocks.
    return None


def load_literal_assignments(path: Path) -> dict[str, Any]:
    """Read literal top-level assignments without importing checkpoint code."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, Any] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        value_node = statement.value
        if value_node is None:
            continue
        value = _safe_eval(value_node, values)
        for target in targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
    return values


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(f"checkpoint metadata is missing {'.'.join(keys)}")
        value = value[key]
    return value


def _predefined_dimension(composition: Mapping[str, Any]) -> int:
    dimension = 0
    for target in composition.values():
        if not isinstance(target, (list, tuple)) or len(target) != 2:
            continue
        destination = target[1] if isinstance(target[1], (list, tuple)) else target
        dimension = max(dimension, int(destination[-1]))
    return dimension


def _compose_dense(
    raw: Mapping[str, Any],
    composition: Mapping[str, Any],
    *,
    temporal_length: int,
    dimension: int,
) -> list[list[float]]:
    output = np.zeros((temporal_length, dimension), dtype=np.float32)
    for source, target in composition.items():
        if source not in raw or not isinstance(target, (list, tuple)) or len(target) != 2:
            continue
        values = np.asarray(raw[source], dtype=np.float32)
        if values.ndim == 1:
            values = np.broadcast_to(values, (temporal_length, values.shape[0]))
        if isinstance(target[1], (list, tuple)):
            source_start, source_end = (int(value) for value in target[0])
            target_start, target_end = (int(value) for value in target[1])
            values = values[:temporal_length, source_start:source_end]
        else:
            target_start, target_end = (int(value) for value in target)
            values = values[:temporal_length]
        width = min(target_end - target_start, values.shape[-1])
        output[: values.shape[0], target_start : target_start + width] = values[:, :width]
    return output.tolist()


def _find_compositions(data: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    train_datasets = _nested(data, "params", "train_datasets")
    sources = _nested(train_datasets, "sources")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("checkpoint metadata has no training data sources")
    first_source = next(iter(sources.values()))
    transforms = list(first_source.get("transforms", ())) + list(train_datasets.get("transforms", ()))
    action_composition = None
    state_composition = None
    for transform in transforms:
        if not isinstance(transform, Mapping):
            continue
        transform_type = transform.get("type")
        if transform_type not in {"ComposeAction", "ComposeState"}:
            continue
        reversed_flow = transform.get("reversed_data_flow")
        if not isinstance(reversed_flow, Mapping) or not reversed_flow:
            continue
        composition = list(reversed_flow.values())[-1]
        if transform_type == "ComposeAction" and action_composition is None:
            action_composition = composition
        elif transform_type == "ComposeState" and state_composition is None:
            state_composition = composition
    if not isinstance(action_composition, Mapping) or not isinstance(state_composition, Mapping):
        raise ValueError("checkpoint metadata has no action/state composition")
    return action_composition, state_composition


def extract_normalization(
    path: Path,
    *,
    expected_state_shape: tuple[int, int],
    expected_action_shape: tuple[int, int],
) -> dict[str, Any]:
    """Convert checkpoint transform literals into dense runtime statistics."""

    assignments = load_literal_assignments(path)
    model = assignments.get("model")
    if isinstance(model, Mapping):
        outer_params = model.get("params")
        model_params = outer_params.get("model", {}) if isinstance(outer_params, Mapping) else {}
        state_shape = model_params.get("state_shape") if isinstance(model_params, Mapping) else None
        action_shape = model_params.get("action_shape") if isinstance(model_params, Mapping) else None
        if state_shape is not None and tuple(int(value) for value in state_shape) != expected_state_shape:
            raise ValueError(
                f"checkpoint state shape {tuple(state_shape)} does not match central {expected_state_shape}"
            )
        if action_shape is not None and tuple(int(value) for value in action_shape) != expected_action_shape:
            raise ValueError(
                f"checkpoint action shape {tuple(action_shape)} does not match central {expected_action_shape}"
            )
    data = assignments.get("data")
    if not isinstance(data, Mapping):
        raise ValueError(f"checkpoint config has no literal data mapping: {path}")
    action_composition, state_composition = _find_compositions(data)
    action_dim = _predefined_dimension(action_composition)
    state_dim = _predefined_dimension(state_composition)
    if action_dim != expected_action_shape[-1] or state_dim != expected_state_shape[-1]:
        raise ValueError(
            "checkpoint composition does not match the central XR1 architecture: "
            f"state={state_dim}/{expected_state_shape[-1]}, action={action_dim}/{expected_action_shape[-1]}"
        )

    train_datasets = _nested(data, "params", "train_datasets")
    sources = _nested(train_datasets, "sources")
    observation_length = int(_nested(data, "params").get("obs_length", expected_state_shape[0]))
    tasks: dict[str, Any] = {}
    for source_name, source in sources.items():
        action_length = expected_action_shape[0]
        for transform in source.get("transforms", ()):
            if isinstance(transform, Mapping) and transform.get("type") == "LoadData":
                action_length = int(transform.get("dataset_action_length", action_length))
        action_statistics: dict[str, Any] = {"mode": "none"}
        state_statistics: dict[str, Any] = {"mode": "none"}
        for transform in source.get("transforms", ()):
            if not isinstance(transform, Mapping):
                continue
            if transform.get("type") != "Normalize":
                continue
            mode = str(transform.get("mode") or "gaussian")
            flow = transform.get("data_flow", {})
            if mode == "gaussian":
                low_name, high_name = "mean", "std"
            elif mode == "quantile":
                low_name, high_name = "q01", "q99"
            else:
                raise ValueError(f"unsupported checkpoint normalization mode: {mode}")
            raw_low = transform.get(low_name, {})
            raw_high = transform.get(high_name, {})
            mapped_low = {target: raw_low[source] for source, target in flow.items() if source in raw_low}
            mapped_high = {target: raw_high[source] for source, target in flow.items() if source in raw_high}
            targets = tuple(str(target) for target in flow.values())
            if any(target.startswith("action_") for target in targets):
                if action_length != expected_action_shape[0]:
                    raise ValueError(
                        f"checkpoint action length {action_length} does not match {expected_action_shape[0]}"
                    )
                action_statistics = {
                    "mode": mode,
                    low_name: _compose_dense(
                        mapped_low,
                        action_composition,
                        temporal_length=action_length,
                        dimension=action_dim,
                    ),
                    high_name: (
                        np.asarray(
                            _compose_dense(
                                mapped_high,
                                action_composition,
                                temporal_length=action_length,
                                dimension=action_dim,
                            ),
                            dtype=np.float32,
                        )
                        + 1e-6
                    ).tolist(),
                }
            if any(target.startswith("proprio_") for target in targets):
                state_statistics = {
                    "mode": mode,
                    low_name: _compose_dense(
                        mapped_low,
                        state_composition,
                        temporal_length=observation_length,
                        dimension=state_dim,
                    ),
                    high_name: (
                        np.asarray(
                            _compose_dense(
                                mapped_high,
                                state_composition,
                                temporal_length=observation_length,
                                dimension=state_dim,
                            ),
                            dtype=np.float32,
                        )
                        + 1e-6
                    ).tolist(),
                }
        if action_statistics["mode"] == "none" or state_statistics["mode"] == "none":
            raise ValueError(f"checkpoint source {source_name!r} has incomplete normalization metadata")
        tasks[str(source_name)] = {
            "state": state_statistics,
            "action": action_statistics,
        }
    if not tasks:
        raise ValueError("checkpoint config produced no normalization tasks")
    return {"tasks": tasks}


__all__ = ["extract_normalization", "load_literal_assignments"]
