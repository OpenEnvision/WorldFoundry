from __future__ import annotations

from pathlib import Path

from worldfoundry.core.io.paths import resolve_data_path


RUNTIME_DIR = Path(__file__).resolve().parent
OFFICIAL_ENTRYPOINT = RUNTIME_DIR / "offline_infer" / "run_inference_client.py"
DEFAULT_INPUT_FILE = resolve_data_path("models", "runtime", "configs", "starwm", "data", "wm_test_horizon5_1traj.json")
BLOCKED_REASON = (
    "StarWM official offline inference client is vendored in-tree; execution "
    "requires a running OpenAI-compatible StarWM/vLLM endpoint and StarCraft "
    "prompt fixtures."
)


def _api_model_id(options):
    return (
        options.get("served_model_id")
        or options.get("api_model_id")
        or options.get("vllm_model_id")
        or options.get("upstream_model_id")
    )


def missing_requirements(*, options, runtime_root, entrypoint, profile):
    del runtime_root, profile
    options = dict(options or {})
    missing = []
    input_file = Path(str(options.get("input_file") or options.get("dataset_path") or DEFAULT_INPUT_FILE)).expanduser()
    mode = str(options.get("mode") or "nothink")
    if entrypoint is None or not Path(entrypoint).is_file():
        missing.append({"kind": "entrypoint", "path": str(entrypoint or ""), "reason": "StarWM inference client is missing"})
    if not input_file.is_file():
        missing.append({"kind": "asset", "path": str(input_file), "reason": "StarWM input prompt fixture does not exist"})
    if not _api_model_id(options):
        missing.append({"kind": "option", "path": "served_model_id", "reason": "StarWM requires served_model_id/api_model_id for the OpenAI-compatible endpoint"})
    if mode not in {"think", "nothink"}:
        missing.append({"kind": "option", "path": "mode", "reason": "StarWM mode must be 'think' or 'nothink'"})
    return missing


def build_command(context):
    options = dict(context.get("options") or {})
    output_file = options.get("output_file") or context["output_path"]
    command = [
        context["python"],
        context["entrypoint"],
        "--input_file",
        str(options.get("input_file") or options.get("dataset_path") or DEFAULT_INPUT_FILE),
        "--output_file",
        str(output_file),
        "--mode",
        str(options.get("mode") or "nothink"),
        "--api_base",
        str(options.get("api_base") or "http://localhost:12000"),
        "--api_key",
        str(options.get("api_key") or "sk-11223344"),
        "--model_id",
        str(_api_model_id(options) or ""),
        "--max_workers",
        str(options.get("max_workers", 8)),
    ]
    return command


__all__ = [
    "BLOCKED_REASON",
    "DEFAULT_INPUT_FILE",
    "OFFICIAL_ENTRYPOINT",
    "RUNTIME_DIR",
    "build_command",
    "missing_requirements",
]
