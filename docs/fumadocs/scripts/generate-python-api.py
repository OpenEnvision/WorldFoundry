#!/usr/bin/env python3
"""Generate deterministic Python API metadata for the Fumadocs site.

The generator parses source files with ``ast`` instead of importing WorldFoundry.
That keeps documentation builds independent from CUDA, simulator, and benchmark
runtime dependencies while ensuring displayed signatures follow the source.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import re
from typing import Any, Iterable

from api_symbol_intros import CURATED_INTROS, GROUP_HINTS


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = REPO_ROOT / "docs" / "fumadocs" / "generated" / "python-api.json"


@dataclass(frozen=True)
class SymbolSpec:
    public_module: str
    name: str
    source_path: str
    group: str | None = None


SYMBOLS = (
    # Serializable evaluation contracts.
    SymbolSpec("worldfoundry.evaluation.api", "ArtifactRef", "worldfoundry/evaluation/api/artifacts.py", "contracts"),
    SymbolSpec("worldfoundry.evaluation.api", "local_path_for_uri", "worldfoundry/evaluation/api/artifacts.py", "contracts"),
    SymbolSpec("worldfoundry.evaluation.api", "enrich_artifact_ref", "worldfoundry/evaluation/api/artifacts.py", "contracts"),
    SymbolSpec("worldfoundry.evaluation.api", "GenerationRequest", "worldfoundry/evaluation/api/generation.py", "contracts"),
    SymbolSpec("worldfoundry.evaluation.api", "GenerationResult", "worldfoundry/evaluation/api/generation.py", "contracts"),
    SymbolSpec("worldfoundry.evaluation.api", "normalize_generation_status", "worldfoundry/evaluation/api/generation.py", "contracts"),
    SymbolSpec("worldfoundry.evaluation.api", "is_generation_result_successful", "worldfoundry/evaluation/api/generation.py", "contracts"),
    # Models, runners, and pipeline extension points.
    SymbolSpec("worldfoundry.evaluation.api", "WorldModelManifest", "worldfoundry/evaluation/api/world_model_manifest.py", "models"),
    SymbolSpec("worldfoundry.evaluation.api", "WorldModelConfig", "worldfoundry/evaluation/api/models.py", "models"),
    SymbolSpec("worldfoundry.evaluation.api", "WorldModelRunner", "worldfoundry/evaluation/api/models.py", "models"),
    SymbolSpec("worldfoundry.pipelines.pipeline_utils", "PipelineABC", "worldfoundry/pipelines/pipeline_utils.py", "models"),
    # Metric and task contracts.
    SymbolSpec("worldfoundry.evaluation.api", "MetricSpec", "worldfoundry/evaluation/api/metrics.py", "metrics-tasks"),
    SymbolSpec("worldfoundry.evaluation.api", "MetricResult", "worldfoundry/evaluation/api/metrics.py", "metrics-tasks"),
    SymbolSpec("worldfoundry.evaluation.api", "AggregateResult", "worldfoundry/evaluation/api/metrics.py", "metrics-tasks"),
    SymbolSpec("worldfoundry.evaluation.api", "Metric", "worldfoundry/evaluation/api/metrics.py", "metrics-tasks"),
    SymbolSpec("worldfoundry.evaluation.api", "EvaluationProtocolSpec", "worldfoundry/evaluation/api/tasks.py", "metrics-tasks"),
    SymbolSpec("worldfoundry.evaluation.api", "WorldTaskConfig", "worldfoundry/evaluation/api/tasks.py", "metrics-tasks"),
    SymbolSpec("worldfoundry.evaluation.api", "BenchmarkSpec", "worldfoundry/evaluation/api/tasks.py", "metrics-tasks"),
    # Canonical run and benchmark facade.
    SymbolSpec("worldfoundry.evaluation.public", "WorldFoundryRunRequest", "worldfoundry/evaluation/framework.py", "runs"),
    SymbolSpec("worldfoundry.evaluation.public", "WorldFoundryRunResult", "worldfoundry/evaluation/framework.py", "runs"),
    SymbolSpec("worldfoundry.evaluation.public", "run_worldfoundry", "worldfoundry/evaluation/framework.py", "runs"),
    SymbolSpec("worldfoundry.evaluation.public", "list_video_benchmarks", "worldfoundry/evaluation/public.py", "runs"),
    SymbolSpec("worldfoundry.evaluation.public", "run_benchmark", "worldfoundry/evaluation/public.py", "runs"),
    SymbolSpec("worldfoundry.evaluation.public", "normalize_upstream_results", "worldfoundry/evaluation/public.py", "runs"),
    SymbolSpec("worldfoundry.evaluation.public", "benchmark_integration_spec", "worldfoundry/evaluation/public.py", "runs"),
    # Evidence and reporting.
    SymbolSpec("worldfoundry.evaluation.reporting", "build_env_requirements", "worldfoundry/evaluation/reporting/run_manifest.py", "reporting"),
    SymbolSpec("worldfoundry.evaluation.reporting", "build_environment", "worldfoundry/evaluation/reporting/run_manifest.py", "reporting"),
    SymbolSpec("worldfoundry.evaluation.reporting", "build_run_manifest", "worldfoundry/evaluation/reporting/run_manifest.py", "reporting"),
    SymbolSpec("worldfoundry.evaluation.reporting", "write_run_manifest_artifacts", "worldfoundry/evaluation/reporting/run_manifest.py", "reporting"),
    SymbolSpec("worldfoundry.evaluation.reporting", "build_scorecard", "worldfoundry/evaluation/reporting/scorecard.py", "reporting"),
    SymbolSpec("worldfoundry.evaluation.reporting", "write_scorecard", "worldfoundry/evaluation/reporting/scorecard.py", "reporting"),
    SymbolSpec("worldfoundry.evaluation.reporting", "build_run_summary", "worldfoundry/evaluation/reporting/run_report.py", "reporting"),
    SymbolSpec("worldfoundry.evaluation.reporting", "build_markdown_report", "worldfoundry/evaluation/reporting/run_report.py", "reporting"),
    SymbolSpec("worldfoundry.evaluation.reporting", "write_run_report_artifacts", "worldfoundry/evaluation/reporting/run_report.py", "reporting"),
    # Runtime paths, local assets, and bounded subprocesses.
    SymbolSpec("worldfoundry.runtime.env", "RequiredEnvReport", "worldfoundry/runtime/env.py", "runtime"),
    SymbolSpec("worldfoundry.runtime.env", "WorldFoundryEnv", "worldfoundry/runtime/env.py", "runtime"),
    SymbolSpec("worldfoundry.runtime.assets", "LocalAsset", "worldfoundry/runtime/assets.py", "runtime"),
    SymbolSpec("worldfoundry.runtime.assets", "expand_worldfoundry_path", "worldfoundry/runtime/assets.py", "runtime"),
    SymbolSpec("worldfoundry.runtime.assets", "load_local_assets", "worldfoundry/runtime/assets.py", "runtime"),
    SymbolSpec("worldfoundry.runtime.jobs", "run_bounded_command", "worldfoundry/runtime/jobs.py", "runtime"),
)

# Ordered catalog sections for the API reference landing page (vLLM-style summary).
CATALOG_SECTIONS = (
    {
        "id": "core",
        "title": {"en": "Core", "zh": "Core"},
        "description": {
            "en": "Shared attention, I/O, loading, distributed, configuration, and acceleration primitives.",
            "zh": "共享的注意力、I/O、加载、分布式、配置与加速 primitive。",
        },
        "guide": "core",
        "groups": (
            "core-attention",
            "core-configuration",
            "core-io-media",
            "core-model-loading",
            "core-distributed",
            "core-runtime",
            "core-nn-math",
            "core-acceleration-memory",
            "core-foundations",
        ),
    },
    {
        "id": "evaluation",
        "title": {"en": "Evaluation", "zh": "评测"},
        "description": {
            "en": "Serializable contracts, runners, metrics, orchestration, evidence, and runtime helpers.",
            "zh": "可序列化契约、runner、metric、编排、证据与 runtime 工具。",
        },
        "guide": None,
        "groups": (
            "contracts",
            "models",
            "metrics-tasks",
            "runs",
            "reporting",
            "runtime",
        ),
    },
)

GROUP_TITLES = {
    "core-attention": {"en": "Attention", "zh": "注意力"},
    "core-configuration": {"en": "Configuration", "zh": "配置"},
    "core-io-media": {"en": "I/O and media", "zh": "I/O 与媒体"},
    "core-model-loading": {"en": "Model loading", "zh": "模型加载"},
    "core-distributed": {"en": "Distributed", "zh": "分布式"},
    "core-runtime": {"en": "Inference runtime", "zh": "推理 Runtime"},
    "core-nn-math": {"en": "Neural net and math", "zh": "神经网络与数学"},
    "core-acceleration-memory": {"en": "Acceleration and memory", "zh": "加速与内存"},
    "core-foundations": {"en": "Foundations", "zh": "基础能力"},
    "contracts": {"en": "Contracts and artifacts", "zh": "契约与 artifact"},
    "models": {"en": "Models and runners", "zh": "模型与 runner"},
    "metrics-tasks": {"en": "Metrics and tasks", "zh": "Metric 与 task"},
    "runs": {"en": "Runs and benchmarks", "zh": "Run 与 benchmark"},
    "reporting": {"en": "Reporting", "zh": "报告与证据"},
    "runtime": {"en": "Runtime and assets", "zh": "Runtime 与资产"},
}


CORE_GROUP_BY_MODULE = {
    "attention": "core-attention",
    "io": "core-io-media",
    "distributed": "core-distributed",
    "model_loading": "core-model-loading",
    "checkpoint": "core-model-loading",
    "inference": "core-runtime",
    "inference_state": "core-runtime",
    "realtime": "core-runtime",
    "process": "core-runtime",
    "device": "core-runtime",
    "time": "core-runtime",
    "nn": "core-nn-math",
    "gradient": "core-nn-math",
    "geometry": "core-nn-math",
    "acceleration": "core-acceleration-memory",
    "vram": "core-acceleration-memory",
    "memory": "core-acceleration-memory",
    "kernels": "core-acceleration-memory",
}


# High-use package-level APIs that intentionally live below ``worldfoundry.core``.
# The top-level lazy facade is documented automatically; these entries cover the
# additional public imports that recur throughout model integrations.
CORE_ADDITIONAL_SYMBOLS = (
    # Attention state and dispatch controls.
    SymbolSpec("worldfoundry.core.attention", "NativeAttention", "worldfoundry/core/attention/native.py", "core-attention"),
    SymbolSpec("worldfoundry.core.attention", "ContextParallelAttention", "worldfoundry/core/attention/cp.py", "core-attention"),
    SymbolSpec("worldfoundry.core.attention", "BlockKVCache", "worldfoundry/core/attention/kvcache.py", "core-attention"),
    SymbolSpec("worldfoundry.core.attention", "packed_sequence_attention", "worldfoundry/core/attention/dispatch.py", "core-attention"),
    SymbolSpec("worldfoundry.core.attention", "attention_dispatch_report", "worldfoundry/core/attention/dispatch.py", "core-attention"),
    SymbolSpec("worldfoundry.core.attention", "clear_attention_dispatch_cache", "worldfoundry/core/attention/dispatch.py", "core-attention"),
    # Lazy configuration is the construction seam used by released configs.
    SymbolSpec("worldfoundry.core.configuration", "Config", "worldfoundry/core/configuration/cosmos_config.py", "core-configuration"),
    SymbolSpec("worldfoundry.core.configuration", "CheckpointConfig", "worldfoundry/core/configuration/cosmos_config.py", "core-configuration"),
    SymbolSpec("worldfoundry.core.configuration", "EMAConfig", "worldfoundry/core/configuration/cosmos_config.py", "core-configuration"),
    SymbolSpec("worldfoundry.core.configuration", "ObjectStoreConfig", "worldfoundry/core/configuration/cosmos_config.py", "core-configuration"),
    SymbolSpec("worldfoundry.core.configuration", "LazyCall", "worldfoundry/core/configuration/lazy_config/lazy_call.py", "core-configuration"),
    SymbolSpec("worldfoundry.core.configuration", "LazyDict", "worldfoundry/core/configuration/lazy_config/__init__.py", "core-configuration"),
    SymbolSpec("worldfoundry.core.configuration", "LazyConfig", "worldfoundry/core/configuration/lazy_config/config.py", "core-configuration"),
    SymbolSpec("worldfoundry.core.configuration", "instantiate", "worldfoundry/core/configuration/lazy_config/instantiate.py", "core-configuration"),
    SymbolSpec("worldfoundry.core.configuration", "make_freezable", "worldfoundry/core/configuration/cosmos_config.py", "core-configuration"),
    # Logical paths are imported directly because callers often need a specific root.
    SymbolSpec("worldfoundry.core.io.paths", "project_root", "worldfoundry/core/io/paths.py", "core-io-media"),
    SymbolSpec("worldfoundry.core.io.paths", "worldfoundry_path_tokens", "worldfoundry/core/io/paths.py", "core-io-media"),
    SymbolSpec("worldfoundry.core.io.paths", "resolve_worldfoundry_path", "worldfoundry/core/io/paths.py", "core-io-media"),
    SymbolSpec("worldfoundry.core.io.paths", "checkpoint_root_path", "worldfoundry/core/io/paths.py", "core-io-media"),
    SymbolSpec("worldfoundry.core.io.paths", "hfd_root_path", "worldfoundry/core/io/paths.py", "core-io-media"),
    SymbolSpec("worldfoundry.core.io.paths", "resolve_data_path", "worldfoundry/core/io/paths.py", "core-io-media"),
    SymbolSpec("worldfoundry.core.io.paths", "local_model_root_path", "worldfoundry/core/io/paths.py", "core-io-media"),
    # Context-parallel helpers are a public subpackage boundary.
    SymbolSpec("worldfoundry.core.distributed", "split_inputs_cp", "worldfoundry/core/distributed/context_parallel.py", "core-distributed"),
    SymbolSpec("worldfoundry.core.distributed", "cat_outputs_cp", "worldfoundry/core/distributed/context_parallel.py", "core-distributed"),
    SymbolSpec("worldfoundry.core.distributed", "cat_outputs_cp_with_grad", "worldfoundry/core/distributed/context_parallel.py", "core-distributed"),
    SymbolSpec("worldfoundry.core.distributed", "broadcast", "worldfoundry/core/distributed/context_parallel.py", "core-distributed"),
    SymbolSpec("worldfoundry.core.distributed", "broadcast_split_tensor", "worldfoundry/core/distributed/context_parallel.py", "core-distributed"),
    SymbolSpec("worldfoundry.core.distributed", "find_split", "worldfoundry/core/distributed/context_parallel.py", "core-distributed"),
    # In-tree acceleration, memory, kernel, and safety primitives.
    SymbolSpec("worldfoundry.core.acceleration", "FixedStepCache", "worldfoundry/core/acceleration/cache.py", "core-acceleration-memory"),
    SymbolSpec("worldfoundry.core.acceleration", "AdaptiveResidualCache", "worldfoundry/core/acceleration/cache.py", "core-acceleration-memory"),
    SymbolSpec("worldfoundry.core.acceleration", "TokenPruner", "worldfoundry/core/acceleration/token_pruning.py", "core-acceleration-memory"),
    SymbolSpec("worldfoundry.core.acceleration", "TokenPruneState", "worldfoundry/core/acceleration/token_pruning.py", "core-acceleration-memory"),
    SymbolSpec("worldfoundry.core.acceleration", "select_token_indices", "worldfoundry/core/acceleration/token_pruning.py", "core-acceleration-memory"),
    SymbolSpec("worldfoundry.core.acceleration", "prune_tokens", "worldfoundry/core/acceleration/token_pruning.py", "core-acceleration-memory"),
    SymbolSpec("worldfoundry.core.acceleration", "restore_tokens", "worldfoundry/core/acceleration/token_pruning.py", "core-acceleration-memory"),
    SymbolSpec("worldfoundry.core.vram", "enable_vram_management", "worldfoundry/core/vram/layers.py", "core-acceleration-memory"),
    SymbolSpec("worldfoundry.core.vram", "enable_vram_management_recursively", "worldfoundry/core/vram/layers.py", "core-acceleration-memory"),
    SymbolSpec("worldfoundry.core.vram", "fill_vram_config", "worldfoundry/core/vram/layers.py", "core-acceleration-memory"),
    SymbolSpec("worldfoundry.core.kernels", "residual_gate_add", "worldfoundry/core/kernels/diffusion.py", "core-acceleration-memory"),
    SymbolSpec("worldfoundry.core.kernels", "layer_norm_scale_shift", "worldfoundry/core/kernels/diffusion.py", "core-acceleration-memory"),
    SymbolSpec("worldfoundry.core.memory", "BaseMemory", "worldfoundry/core/memory/base.py", "core-acceleration-memory"),
    SymbolSpec("worldfoundry.core.memory", "MemoryStore", "worldfoundry/core/memory/store.py", "core-acceleration-memory"),
    SymbolSpec("worldfoundry.core.realtime", "RealtimeSpec", "worldfoundry/core/realtime.py", "core-runtime"),
    SymbolSpec("worldfoundry.core.safety", "ContentSafetyGuardrail", "worldfoundry/core/safety/guardrails.py", "core-foundations"),
    SymbolSpec("worldfoundry.core.safety", "PostprocessingGuardrail", "worldfoundry/core/safety/guardrails.py", "core-foundations"),
    SymbolSpec("worldfoundry.core.safety", "GuardrailRunner", "worldfoundry/core/safety/guardrails.py", "core-foundations"),
)


def unparse(node: ast.AST | None) -> str | None:
    return None if node is None else ast.unparse(node)


def rendered_default(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    text = ast.unparse(node)
    if isinstance(node, ast.Call) and unparse(node.func) == "field":
        for keyword in node.keywords:
            if keyword.arg == "default_factory":
                return f"<{unparse(keyword.value)} factory>"
    return text


def parameter_payload(
    argument: ast.arg,
    *,
    default: ast.AST | None = None,
    kind: str = "positional_or_keyword",
    description: str | None = None,
) -> dict[str, Any]:
    return {
        "name": argument.arg,
        "annotation": unparse(argument.annotation),
        "default": rendered_default(default),
        "kind": kind,
        "description": description,
    }


def parse_google_docstring(value: str | None) -> dict[str, Any]:
    doc = inspect.cleandoc(value or "")
    if not doc:
        return {"summary": "", "parameters": {}, "returns": "", "raises": {}, "notes": "", "warnings": ""}

    lines = doc.splitlines()
    section_pattern = re.compile(
        r"^(Args|Arguments|Parameters|Attributes|Returns|Yields|Raises|Note|Notes|Warning|Warnings):\s*$"
    )
    first_section = next((index for index, line in enumerate(lines) if section_pattern.match(line.strip())), len(lines))
    summary = "\n".join(lines[:first_section]).strip()
    parameters: dict[str, str] = {}
    raises: dict[str, str] = {}
    returns: list[str] = []
    notes: list[str] = []
    warnings: list[str] = []
    section = "summary"
    active_name: str | None = None

    for raw_line in lines[first_section:]:
        stripped = raw_line.strip()
        match = section_pattern.match(stripped)
        if match:
            section = match.group(1).lower()
            active_name = None
            continue
        if not stripped:
            continue
        if section in {"args", "arguments", "parameters", "attributes"}:
            item = re.match(r"^([*\w][\w*]*)(?:\s*\([^)]*\))?\s*:\s*(.*)$", stripped)
            if item:
                active_name = item.group(1).lstrip("*")
                parameters[active_name] = item.group(2).strip()
            elif active_name:
                parameters[active_name] = f"{parameters[active_name]} {stripped}".strip()
        elif section in {"returns", "yields"}:
            returns.append(stripped)
        elif section == "raises":
            item = re.match(r"^([\w.]+)\s*:\s*(.*)$", stripped)
            if item:
                active_name = item.group(1)
                raises[active_name] = item.group(2).strip()
            elif active_name:
                raises[active_name] = f"{raises[active_name]} {stripped}".strip()
        elif section in {"note", "notes"}:
            notes.append(stripped)
        elif section in {"warning", "warnings"}:
            warnings.append(stripped)

    return {
        "summary": summary,
        "parameters": parameters,
        "returns": " ".join(returns),
        "raises": raises,
        "notes": " ".join(notes),
        "warnings": " ".join(warnings),
    }


def function_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef, *, drop_first: bool = False) -> list[dict[str, Any]]:
    parsed_doc = parse_google_docstring(ast.get_docstring(node, clean=False))
    descriptions = parsed_doc["parameters"]
    positional = [*node.args.posonlyargs, *node.args.args]
    defaults: list[ast.AST | None] = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
    params: list[dict[str, Any]] = []

    for index, (argument, default) in enumerate(zip(positional, defaults)):
        if drop_first and index == 0 and argument.arg in {"self", "cls"}:
            continue
        kind = "positional_only" if index < len(node.args.posonlyargs) else "positional_or_keyword"
        params.append(
            parameter_payload(
                argument,
                default=default,
                kind=kind,
                description=descriptions.get(argument.arg),
            )
        )

    if node.args.vararg is not None:
        params.append(
            parameter_payload(
                node.args.vararg,
                kind="var_positional",
                description=descriptions.get(node.args.vararg.arg),
            )
        )

    for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        params.append(
            parameter_payload(
                argument,
                default=default,
                kind="keyword_only",
                description=descriptions.get(argument.arg),
            )
        )

    if node.args.kwarg is not None:
        params.append(
            parameter_payload(
                node.args.kwarg,
                kind="var_keyword",
                description=descriptions.get(node.args.kwarg.arg),
            )
        )
    return params


def format_parameters(parameters: Iterable[dict[str, Any]]) -> str:
    values = list(parameters)
    rendered: list[str] = []
    inserted_kw_marker = False
    posonly_count = sum(item["kind"] == "positional_only" for item in values)
    seen_posonly = 0

    for parameter in values:
        kind = parameter["kind"]
        if kind == "keyword_only" and not inserted_kw_marker and not any(item["kind"] == "var_positional" for item in values):
            rendered.append("*")
            inserted_kw_marker = True
        prefix = "*" if kind == "var_positional" else "**" if kind == "var_keyword" else ""
        part = f"{prefix}{parameter['name']}"
        if parameter["annotation"]:
            part += f": {parameter['annotation']}"
        if parameter["default"] is not None:
            part += f" = {parameter['default']}"
        rendered.append(part)
        if kind == "positional_only":
            seen_posonly += 1
            if seen_posonly == posonly_count:
                rendered.append("/")
    return ", ".join(rendered)


def class_fields(node: ast.ClassDef) -> list[dict[str, Any]]:
    descriptions = parse_google_docstring(ast.get_docstring(node, clean=False))["parameters"]
    fields: list[dict[str, Any]] = []
    for item in node.body:
        if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
            continue
        annotation = unparse(item.annotation) or ""
        if "ClassVar[" in annotation or item.target.id.startswith("_"):
            continue
        fields.append(
            {
                "name": item.target.id,
                "annotation": annotation,
                "default": rendered_default(item.value),
                "description": descriptions.get(item.target.id),
            }
        )
    return fields


def first_sentence(text: str) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    for separator in (". ", "。", "? ", "！", "! "):
        index = cleaned.find(separator)
        if index > 0:
            end = index + (1 if separator.startswith(("。", "！")) else 2)
            return cleaned[:end].strip()
    return cleaned


def build_intro(
    *,
    qualified_name: str,
    name: str,
    kind: str,
    group: str | None,
    docstring: str,
    return_annotation: str | None,
) -> dict[str, str]:
    curated = CURATED_INTROS.get(qualified_name)
    if curated:
        return {"en": curated["en"], "zh": curated["zh"]}

    summary = first_sentence(docstring)
    if not summary:
        if kind == "class":
            summary = f"`{name}` is a public class in this API surface."
        elif kind == "protocol":
            summary = f"`{name}` is a runtime-checkable protocol for implementers."
        else:
            summary = f"`{name}` is a public function in this API surface."

    hint = GROUP_HINTS.get(group or "", {})
    en_parts = [summary]
    hint_en = hint.get("en", "")
    if hint_en and hint_en.casefold() not in summary.casefold():
        en_parts.append(hint_en)
    if kind == "function" and return_annotation:
        en_parts.append(f"Annotated return type: `{return_annotation}`.")

    zh_lead = f"`{name}`"
    if summary:
        # Keep the upstream English summary readable under a Chinese lead-in.
        zh_parts = [f"{zh_lead} — {summary}"]
    else:
        zh_parts = [f"{zh_lead} 是本页公开 API 之一。"]
    hint_zh = hint.get("zh", "")
    if hint_zh:
        zh_parts.append(hint_zh)
    if kind == "function" and return_annotation:
        zh_parts.append(f"标注返回类型：`{return_annotation}`。")

    return {
        "en": " ".join(en_parts),
        "zh": " ".join(zh_parts),
    }


def method_payload(node: ast.FunctionDef | ast.AsyncFunctionDef, source_path: str) -> dict[str, Any] | None:
    if node.name.startswith("_") and node.name != "__call__":
        return None
    decorators = {unparse(decorator) for decorator in node.decorator_list}
    kind = "property" if "property" in decorators else "classmethod" if "classmethod" in decorators else "staticmethod" if "staticmethod" in decorators else "method"
    parameters = function_parameters(node, drop_first=kind not in {"staticmethod", "property"})
    return_annotation = unparse(node.returns)
    if kind == "property":
        signature = f"{node.name} -> {return_annotation or 'Any'}"
    else:
        signature = f"{node.name}({format_parameters(parameters)})"
        if return_annotation:
            signature += f" -> {return_annotation}"
    parsed_doc = parse_google_docstring(ast.get_docstring(node, clean=False))
    summary = first_sentence(parsed_doc["summary"])
    if summary:
        intro = {
            "en": summary,
            "zh": f"`{node.name}` — {summary}",
        }
    else:
        intro = {
            "en": f"Public `{kind}` on this type.",
            "zh": f"该类型上的公开 `{kind}`。",
        }
    return {
        "name": node.name,
        "kind": kind,
        "signature": signature,
        "parameters": parameters,
        "return_annotation": return_annotation,
        "returns_description": parsed_doc["returns"],
        "raises": parsed_doc["raises"],
        "notes": parsed_doc["notes"],
        "warnings": parsed_doc["warnings"],
        "docstring": parsed_doc["summary"],
        "intro": intro,
        "source_path": source_path,
        "line": node.lineno,
    }


def symbol_payload(spec: SymbolSpec, node: ast.AST) -> dict[str, Any]:
    doc = parse_google_docstring(ast.get_docstring(node, clean=False))
    qualified_name = f"{spec.public_module}.{spec.name}"
    base = {
        "name": spec.name,
        "qualified_name": qualified_name,
        "public_module": spec.public_module,
        "page": spec.group,
        "source_path": spec.source_path,
        "line": node.lineno,
        "docstring": doc["summary"],
        "raises": doc["raises"],
        "notes": doc["notes"],
        "warnings": doc["warnings"],
    }

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        parameters = function_parameters(node)
        return_annotation = unparse(node.returns)
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        signature = f"{prefix} {node.name}({format_parameters(parameters)})"
        if return_annotation:
            signature += f" -> {return_annotation}"
        return {
            **base,
            "kind": "function",
            "signature": signature,
            "parameters": parameters,
            "return_annotation": return_annotation,
            "returns_description": doc["returns"],
            "intro": build_intro(
                qualified_name=qualified_name,
                name=spec.name,
                kind="function",
                group=spec.group,
                docstring=doc["summary"],
                return_annotation=return_annotation,
            ),
            "fields": [],
            "methods": [],
        }

    if not isinstance(node, ast.ClassDef):
        raise TypeError(f"Unsupported API node for {spec.name}: {type(node).__name__}")

    fields = class_fields(node)
    init_node = next(
        (item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__"),
        None,
    )
    if init_node is not None:
        constructor_parameters = function_parameters(init_node, drop_first=True)
    else:
        constructor_parameters = [
            {
                "name": field["name"],
                "annotation": field["annotation"],
                "default": field["default"],
                "kind": "positional_or_keyword",
                "description": field["description"],
            }
            for field in fields
        ]
    bases = [unparse(item) or "" for item in node.bases]
    is_protocol = any(base.endswith("Protocol") for base in bases)
    kind = "protocol" if is_protocol else "class"
    signature = f"class {node.name}"
    if not is_protocol:
        signature += f"({format_parameters(constructor_parameters)})"
    else:
        signature += f"({', '.join(bases)})"
    methods = [
        payload
        for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        if (payload := method_payload(item, spec.source_path)) is not None
    ]
    return {
        **base,
        "kind": kind,
        "signature": signature,
        "parameters": constructor_parameters,
        "return_annotation": None,
        "returns_description": "",
        "intro": build_intro(
            qualified_name=qualified_name,
            name=spec.name,
            kind=kind,
            group=spec.group,
            docstring=doc["summary"],
            return_annotation=None,
        ),
        "fields": fields,
        "methods": methods,
    }


def module_source_path(module: str) -> Path:
    """Resolve a Python module name without importing it."""

    relative = Path(*module.split("."))
    package_init = REPO_ROOT / relative / "__init__.py"
    if package_init.is_file():
        return package_init
    return REPO_ROOT / relative.with_suffix(".py")


def module_export_map(tree: ast.Module) -> dict[str, str]:
    for item in tree.body:
        if not isinstance(item, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "_EXPORT_MODULES" for target in item.targets):
            continue
        value = ast.literal_eval(item.value)
        if not isinstance(value, dict):
            raise TypeError("_EXPORT_MODULES must be a literal dict")
        return {str(name): str(module) for name, module in value.items()}
    return {}


def imported_module_name(current_module: str, source_path: Path, item: ast.ImportFrom) -> str:
    if item.level == 0:
        return item.module or ""
    parts = current_module.split(".") if source_path.name == "__init__.py" else current_module.split(".")[:-1]
    for _ in range(item.level - 1):
        if parts:
            parts.pop()
    if item.module:
        parts.extend(item.module.split("."))
    return ".".join(parts)


def resolve_symbol_definition(
    module: str,
    name: str,
    *,
    trees: dict[str, ast.Module],
    seen: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[str, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef] | None:
    """Follow lazy exports and re-exports to a concrete source definition."""

    identity = (module, name)
    if identity in seen:
        return None
    source = module_source_path(module)
    if not source.is_file():
        return None
    source_path = source.relative_to(REPO_ROOT).as_posix()
    tree = trees.get(source_path)
    if tree is None:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        trees[source_path] = tree

    for item in tree.body:
        if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
            return source_path, item

    next_seen = seen | {identity}
    export_target = module_export_map(tree).get(name)
    if export_target:
        return resolve_symbol_definition(export_target, name, trees=trees, seen=next_seen)

    for item in tree.body:
        if not isinstance(item, ast.ImportFrom):
            continue
        for alias in item.names:
            if (alias.asname or alias.name) != name:
                continue
            imported_module = imported_module_name(module, source, item)
            return resolve_symbol_definition(imported_module, alias.name, trees=trees, seen=next_seen)
    return None


def core_symbol_specs(trees: dict[str, ast.Module]) -> tuple[SymbolSpec, ...]:
    """Discover every concrete function/class on the top-level core facade."""

    core_source = module_source_path("worldfoundry.core")
    core_source_path = core_source.relative_to(REPO_ROOT).as_posix()
    core_tree = ast.parse(core_source.read_text(encoding="utf-8"), filename=str(core_source))
    trees[core_source_path] = core_tree
    specs: list[SymbolSpec] = []
    for name, target_module in module_export_map(core_tree).items():
        resolved = resolve_symbol_definition(target_module, name, trees=trees)
        if resolved is None:
            # Constants and compatibility aliases are described in the prose
            # guides; this reference renders callable/class definitions.
            continue
        source_path, node = resolved
        if node.name != name:
            continue
        top_level_module = target_module.removeprefix("worldfoundry.core.").split(".", 1)[0]
        group = CORE_GROUP_BY_MODULE.get(top_level_module, "core-foundations")
        specs.append(SymbolSpec("worldfoundry.core", name, source_path, group))

    specs.extend(CORE_ADDITIONAL_SYMBOLS)
    unique: dict[tuple[str, str], SymbolSpec] = {}
    for spec in specs:
        unique[(spec.public_module, spec.name)] = spec
    return tuple(unique.values())


def build_name_index(symbols: dict[str, Any]) -> dict[str, str]:
    """Map short names to a unique qualified symbol for cross-links.

    Ambiguous short names are omitted so guide prose does not link to the wrong page.
    """

    candidates: dict[str, list[str]] = {}
    for qualified, entry in symbols.items():
        candidates.setdefault(str(entry["name"]), []).append(qualified)
    return {
        name: qualified[0]
        for name, qualified in candidates.items()
        if len(qualified) == 1
    }


def generate() -> dict[str, Any]:
    trees: dict[str, ast.Module] = {}
    result: dict[str, Any] = {}
    groups: dict[str, list[str]] = {}
    all_specs = (*SYMBOLS, *core_symbol_specs(trees))
    for spec in all_specs:
        tree = trees.get(spec.source_path)
        if tree is None:
            path = REPO_ROOT / spec.source_path
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            trees[spec.source_path] = tree
        node = next(
            (
                item
                for item in tree.body
                if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == spec.name
            ),
            None,
        )
        if node is None:
            raise RuntimeError(f"Could not find {spec.name} in {spec.source_path}")
        key = f"{spec.public_module}.{spec.name}"
        result[key] = symbol_payload(spec, node)
        if spec.group:
            groups.setdefault(spec.group, []).append(key)
    for keys in groups.values():
        keys.sort(key=lambda key: (result[key]["name"].casefold(), key))

    catalog = []
    for section in CATALOG_SECTIONS:
        pages = []
        if section["guide"]:
            pages.append(
                {
                    "slug": section["guide"],
                    "kind": "guide",
                    "title": {
                        "en": "Core API guide",
                        "zh": "Core API 指南",
                    },
                    "symbols": [],
                }
            )
        for group in section["groups"]:
            symbols = groups.get(group, [])
            # Keep the landing page scannable: Core groups are large, so only
            # evaluate-facing pages expand into full symbol lists here.
            pages.append(
                {
                    "slug": group,
                    "kind": "reference",
                    "title": GROUP_TITLES[group],
                    "symbols": [] if section["id"] == "core" else symbols,
                    "symbol_count": len(symbols),
                }
            )
        catalog.append(
            {
                "id": section["id"],
                "title": section["title"],
                "description": section["description"],
                "pages": pages,
            }
        )

    return {
        "schema_version": "worldfoundry-python-api-reference-v4",
        "repository": "https://github.com/OpenEnvision/WorldFoundry",
        "branch": "main",
        "catalog": catalog,
        "groups": groups,
        "group_titles": GROUP_TITLES,
        "name_index": build_name_index(result),
        "symbols": result,
    }


def serialized_payload() -> str:
    return json.dumps(generate(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail when the generated file is stale.")
    args = parser.parse_args()
    payload = serialized_payload()
    if args.check:
        current = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.is_file() else ""
        if current != payload:
            print(f"stale generated API metadata: {OUTPUT_PATH}")
            return 1
        print(f"API metadata is current: {OUTPUT_PATH}")
        return 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    current = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.is_file() else ""
    if current != payload:
        OUTPUT_PATH.write_text(payload, encoding="utf-8")
        print(f"generated {OUTPUT_PATH}")
    else:
        print(f"unchanged {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
