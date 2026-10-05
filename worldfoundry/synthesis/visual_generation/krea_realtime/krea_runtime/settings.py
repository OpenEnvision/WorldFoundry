import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

try:
    import dotenv
except ModuleNotFoundError:
    dotenv = None

if dotenv is not None:
    dotenv.load_dotenv()

MODEL_FOLDER = os.getenv("MODEL_FOLDER", "Wan-2.1")
COMPILE_SHAPES = [(832, 480), (480, 832)]

_MODEL_ROOT_ENV = {
    "Wan2.1-T2V-1.3B": "WORLDFOUNDRY_KREA_WAN_1P3B_ROOT",
    "Wan2.1-T2V-14B": "WORLDFOUNDRY_KREA_WAN_14B_ROOT",
}


def model_root_candidates(model_name: str) -> tuple[Path, ...]:
    """Return local Wan roots without downloading or mutating checkpoints."""

    candidates: list[Path] = []
    explicit_key = _MODEL_ROOT_ENV.get(model_name)
    if explicit_key and os.getenv(explicit_key, "").strip():
        candidates.append(Path(os.environ[explicit_key]).expanduser())
    model_folder = Path(MODEL_FOLDER).expanduser()
    candidates.extend(
        (
            model_folder / model_name,
            model_folder / f"Wan-AI--{model_name}",
        )
    )
    hfd_root = os.getenv("WORLDFOUNDRY_HFD_ROOT", "").strip()
    if hfd_root:
        hfd_path = Path(hfd_root).expanduser()
        candidates.extend((hfd_path / model_name, hfd_path / f"Wan-AI--{model_name}"))
    return tuple(dict.fromkeys(candidates))


def resolve_model_path(model_name: str) -> Path:
    """Resolve both upstream and HF-downloader directory conventions."""

    candidates = model_root_candidates(model_name)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return candidates[0]


def resolve_model_asset(model_name: str, *relative_names: str) -> Path:
    """Resolve the first available official filename under a local model root."""

    root = resolve_model_path(model_name)
    for relative_name in relative_names:
        candidate = root / relative_name
        if candidate.is_file():
            return candidate
    return root / relative_names[0]


def compiler_stance(torch_module: Any, stance: str):
    """Use the optional torch.compiler stance API when the runtime provides it."""

    compiler = getattr(torch_module, "compiler", None)
    setter = getattr(compiler, "set_stance", None)
    return setter(stance) if callable(setter) else nullcontext()
