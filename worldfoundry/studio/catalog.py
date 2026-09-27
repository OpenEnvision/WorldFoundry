from __future__ import annotations

import ast
import json
import os
import re
import sys
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from worldfoundry.core.io.paths import (
    checkpoint_root_candidates,
    checkpoint_root_path,
    hfd_root_path,
    local_data_root_path,
    official_runtime_repo_path,
    resolve_local_hf_model_path,
    resolve_worldfoundry_path,
)
from worldfoundry.runtime.interactive_inference_catalog import studio_contract_override

from .runtime_paths import studio_hfd_cache_roots


@dataclass
class _CatalogPathSnapshot:
    roots: tuple[Path, ...]
    children: dict[Path, frozenset[str]]
    exists_cache: dict[Path, bool] = field(default_factory=dict)

    def missing(self, path: Path) -> bool:
        for root, names in self.children.items():
            try:
                parts = path.relative_to(root).parts
            except ValueError:
                continue
            if parts and ".." not in parts and parts[0] not in names:
                return True
        return False

    def exists(self, path: Path) -> bool:
        if path not in self.exists_cache:
            self.exists_cache[path] = not self.missing(path) and path.exists()
        return self.exists_cache[path]


_CATALOG_CACHE_ROOTS: ContextVar[_CatalogPathSnapshot | None] = ContextVar("studio_catalog_cache_roots", default=None)

PIPELINES_ROOT = Path(__file__).resolve().parents[1] / "pipelines"
ACTION_SYNTHESIS_ROOT = Path(__file__).resolve().parents[1] / "synthesis" / "action_generation"
NESTED_PIPELINE_DIRS: tuple[str, ...] = ()
DISPATCH_LOAD_PARAMS: tuple[str, ...] = (
    "cuda_visible_devices",
    "visible_devices",
    "cuda_devices",
    "gpu_ids",
)
ABSTRACT_RUNTIME_MODEL_IDS: frozenset[str] = frozenset(
    {
        "official-policy",
        "official-video",
        "three-d-four-d-runtime",
        "world-model-runtime",
    }
)
_SANA_IMAGE_RUNNERS = frozenset({"image", "sprint", "controlnet"})
SANA_DEFAULT_IMAGE_VARIANT_ID = "sana-1600m-1024px-bf16"
COGVIDEOX_STUDIO_PARENT_ID = "cogvideox_5b_t2v"
COGVIDEOX_DEFAULT_VARIANT_ID = "cogvideox_5b_t2v"
_COGVIDEOX_VARIANT_IDS = frozenset(
    {
        "cogvideox_2b_t2v",
        "cogvideox_5b_t2v",
        "cogvideox_5b_i2v",
    }
)
_COGVIDEOX_ALIAS_TO_VARIANT = {
    "cogvideox": "cogvideox_5b_t2v",
    "cogvideox_2b": "cogvideox_2b_t2v",
    "cogvideox_5b": "cogvideox_5b_t2v",
    "cogvideox_2b_t2v": "cogvideox_2b_t2v",
    "cogvideox_5b_t2v": "cogvideox_5b_t2v",
    "cogvideox_5b_i2v": "cogvideox_5b_i2v",
}


@lru_cache(maxsize=1)
def _sana_folded_studio_model_ids() -> frozenset[str]:
    """Return image-family Sana checkpoint ids that belong on one Studio card."""

    from worldfoundry.base_models.diffusion_model.recipes.sana_variants import SANA_VARIANTS

    return frozenset(
        variant.model_id
        for variant in SANA_VARIANTS.values()
        if variant.runner in _SANA_IMAGE_RUNNERS
    )


def _cogvideox_canonical_id(model_id: str) -> str:
    return _normalize_model_id(model_id).replace("-", "_")


def _cogvideox_variant_id(model_id: str) -> str | None:
    key = _cogvideox_canonical_id(model_id)
    if key in _COGVIDEOX_VARIANT_IDS:
        return key
    return _COGVIDEOX_ALIAS_TO_VARIANT.get(key)


def _catalog_id_key(model_id: str) -> str:
    return _normalize_model_id(model_id).replace("-", "_")


def _cogvideox_folded_studio_model_ids() -> frozenset[str]:
    """Return CogVideoX pipeline ids that are variants on the shared Studio card."""

    folded = _COGVIDEOX_VARIANT_IDS - {COGVIDEOX_STUDIO_PARENT_ID}
    return frozenset(item for model_id in folded for item in (model_id, model_id.replace("_", "-")))


def _cogvideox_studio_parent_id(model_id: str) -> str | None:
    """Return the CogVideoX card id when ``model_id`` is a folded T2V/I2V pipeline."""

    if _cogvideox_variant_id(model_id):
        return COGVIDEOX_STUDIO_PARENT_ID
    return None


def cogvideox_runtime_model_id(model_id: str, variant_id: str = "") -> str | None:
    """Return the concrete CogVideoX pipeline id for a Studio card + variant."""

    if _cogvideox_studio_parent_id(model_id) is None:
        return None
    return _cogvideox_variant_id(variant_id or model_id) or COGVIDEOX_DEFAULT_VARIANT_ID


STUDIO_HIDDEN_CATALOG_MODEL_IDS: frozenset[str] = frozenset(
    {
        # Internal shared operator/contract surface. Concrete priors such as metric3d-prior,
        # unidepth-v2-prior, dap, and video-depth-anything-prior are the user-facing entries.
        "geometry-prior",
        # The complete Wan2.2 camera/base/LoRA/MoGe assets pass CPU schema validation,
        # but the official dual-A14B GPU memory strategy still needs a real artifact run.
        "fantasyworld-wan22",
        # Lyra-2 alias; keep lyra-1 / lyra-2 as the canonical Create Job entries.
        "lyra",
        # Runtime-environment alias for ltx-2.3-i2v. Keeping it as a separate
        # catalog row duplicates the same pipeline and invents a non-existent
        # ``Ltx23I2VPipeline`` class name during canonical-entry synthesis.
        "ltx2_3_i2v",
        # Hidden official weights; keep provenance in the model catalog but do not
        # expose a Studio infer row that cannot return a real artifact.
        "pandora",
        "omnivinci",
        "emu3.5",
        "qwen2.5-omni",
        "spatial-ladder",
        "spatial-reasoner",
        # Image / Sprint / ControlNet checkpoints are variants on the Sana card.
        # CogVideoX 2B T2V and 5B I2V share the CogVideoX card with 5B T2V.
    }
) | _sana_folded_studio_model_ids() | _cogvideox_folded_studio_model_ids()

# These Workspace matrix ids intentionally retain underscores while their AST
# pipeline ids are filename-derived hyphenated slugs. Resolve them through the
# fast single-entry path so a direct job does not scan and stat the entire
# checkpoint catalog before importing the requested pipeline.
FAST_CANONICAL_AST_ALIASES: Mapping[str, str] = {
    "videocrafter1_i2v": "videocrafter1-i2v",
    "videocrafter1_t2v": "videocrafter1-t2v",
    "videocrafter2_t2v": "videocrafter2-t2v",
}


@lru_cache(maxsize=1)
def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return current.parents[4]


def _project_root_candidates() -> tuple[Path, ...]:
    current = _project_root().resolve()
    current_text = str(current)
    if current_text.startswith("/share/project/"):
        candidates: list[Path] = [Path("/bench-workspace") / current_text[len("/share/project/") :], current]
    else:
        candidates = [current]
        if current_text.startswith("/bench-workspace/"):
            candidates.append(Path("/share/project") / current_text[len("/bench-workspace/") :])
    deduped: list[Path] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return tuple(deduped)


def _source_root() -> Path:
    src_root = _project_root() / "src"
    return src_root if (src_root / "worldfoundry").is_dir() else _project_root()


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _data_path(*parts: str | Path) -> Path:
    return _package_root().joinpath("data", *(Path(part) for part in parts))


def _cache_candidates(*repo_dir_names: str) -> list[str]:
    candidates: list[str] = []
    snapshot = _CATALOG_CACHE_ROOTS.get()
    roots = snapshot.roots if snapshot is not None else studio_hfd_cache_roots(project_roots=_project_root_candidates())
    for root in roots:
        for repo_dir_name in repo_dir_names:
            repo_paths = [root / repo_dir_name]
            if not repo_dir_name.startswith("models--"):
                repo_paths.append(root / f"models--{repo_dir_name}")
            for repo_path in repo_paths:
                if snapshot is not None and snapshot.missing(repo_path):
                    candidates.append(str(repo_path))
                    continue
                refs_main = repo_path / "refs" / "main"
                snapshots_dir = repo_path / "snapshots"
                if refs_main.is_file():
                    revision = refs_main.read_text(encoding="utf-8").strip()
                    if revision:
                        candidates.append(str(snapshots_dir / revision))
                if snapshots_dir.is_dir():
                    candidates.extend(str(path) for path in snapshots_dir.iterdir() if path.is_dir() or path.is_symlink())
                candidates.append(str(repo_path))
    return candidates


def _split_camel(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", value)
    value = re.sub(r"(\d)([A-Za-z])", r"\1 \2", value)
    value = re.sub(r"([A-Za-z])(\d)", r"\1 \2", value)
    return value.replace("  ", " ").strip()


def _slug_from_filename(path: Path) -> str:
    stem = path.stem
    if stem.startswith("pipeline_"):
        stem = stem[len("pipeline_") :]
    return stem.replace("_", "-")


def _display_name_from_slug(slug: str) -> str:
    tokens = slug.replace(".", "-").split("-")
    pretty = []
    for token in tokens:
        if not token or token.lower() == "prior":
            continue
        if token.isupper():
            pretty.append(token)
        elif token.lower() in {"i2v", "t2v", "ti2v", "i2av", "t2av", "api", "vmem", "vggt"}:
            pretty.append(token.upper())
        elif token.lower() in {"pi3", "gen3c", "wow", "cut3r", "loger"}:
            pretty.append(token.upper())
        elif token.lower().startswith("2p") and len(token) > 2:
            pretty.append(token.replace("p", "."))
        else:
            pretty.append(token.capitalize())
    return " ".join(pretty)


def _summary_from_category(category: str, family: str, supports_stream: bool) -> str:
    stream_hint = " Supports multi-turn continuation." if supports_stream else ""
    if category == "Embodied Action":
        return f"{family} pipeline for policy, action, and rollout artifacts.{stream_hint}"
    if category == "Visual Action":
        return f"{family} pipeline for latent-action or video-action artifacts.{stream_hint}"
    if category == "Video-to-Video":
        return f"{family} pipeline for video-to-video rerendering, trajectory control, and camera-path edits.{stream_hint}"
    if category == "3D Scene":
        return f"{family} pipeline for reconstruction, camera control, and spatial outputs.{stream_hint}"
    if category == "Depth / Geometry":
        return f"{family} pipeline for depth or geometry extraction.{stream_hint}"
    if category == "Remote API":
        return f"{family} pipeline backed by a hosted API endpoint.{stream_hint}"
    return f"{family} pipeline for world-model generation and interaction.{stream_hint}"


def _category_from_family(family: str, class_name: str, call_params: Sequence[str]) -> str:
    if family in {
        "embodied_action",
        "openvla",
        "openpi",
        "giga_brain_0",
        "dreamzero",
        "gr00t",
        "starvla",
        "lingbot_va",
        "octo",
        "rt1",
        "diffusion_policy",
        "act",
        "roboflamingo",
    }:
        return "Embodied Action"
    if family in {"lapa"}:
        return "Visual Action"
    if family in {
        "vggt",
        "vggt_omega",
        "cut3r",
        "pi3",
        "infinite_vggt",
        "lagernvs",
        "pixelsplat",
        "splatt3r",
        "dvlt",
        "lingbot_map",
    }:
        return "3D Scene"
    if family in {
        "depth_anything",
        "worldfm",
        "dust3r",
        "dust3r_base_model",
        "geometry_prior",
    } or family.startswith("geometry_prior"):
        return "Depth / Geometry"
    if family in {"worldlabs", "worldlabs_marble_1p1", "runway", "luma", "veo", "sora", "minimax"}:
        return "Remote API"
    if "api_init" in class_name.lower():
        return "Remote API"
    if "output_dir" in call_params and "task_type" in call_params and family in {"hunyuan_world"}:
        return "3D Scene"
    return "Video Generation"


def _runtime_kind_from_family(model_id: str, family: str) -> str:
    if model_id in {"vggt", "vggt-omega", "cut3r"}:
        return "two_stage_3dgs"
    if model_id in {"pi3", "loger"}:
        return "pointcloud_nav"
    if model_id == "worldfm":
        return "worldfm"
    return "default"


def _default_backend(supports_api_init: bool, supports_from_pretrained: bool) -> str:
    if supports_from_pretrained:
        return "from_pretrained"
    if supports_api_init:
        return "api_init"
    return "auto"


def _coerce_aliases(*values: str) -> tuple[str, ...]:
    deduped: List[str] = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return tuple(deduped)


def _append_dispatch_load_params(load_params: Sequence[str]) -> tuple[str, ...]:
    values = [str(item) for item in load_params]
    for item in DISPATCH_LOAD_PARAMS:
        if item not in values:
            values.append(item)
    return tuple(values)


def _resolve_override_value(value: Any) -> Any:
    return value() if callable(value) else value


def _resolve_extra_variants(value: Any) -> tuple[Dict[str, Any], ...]:
    rows: list[Dict[str, Any]] = []
    for raw_variant in _resolve_override_value(value) or ():
        variant = dict(raw_variant)
        for key in ("load_kwargs", "call_kwargs"):
            if key in variant:
                variant[key] = dict(_resolve_override_value(variant[key]) or {})
        checkpoints = []
        for raw_checkpoint in variant.get("checkpoints", ()) or ():
            checkpoint = dict(raw_checkpoint)
            if "uri" in checkpoint:
                checkpoint["uri"] = str(_resolve_override_value(checkpoint["uri"]) or "")
            checkpoints.append(checkpoint)
        variant["checkpoints"] = tuple(checkpoints)
        rows.append(variant)
    return tuple(rows)


def _model_ref_env_key(model_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_").upper()
    return f"WORLDFOUNDRY_STUDIO_MODEL_REF_{normalized}"


def _resolve_default_model_ref(model_id: str, fallback: str) -> str:
    model_override = os.getenv(_model_ref_env_key(model_id), "").strip()
    if model_override:
        return model_override
    global_override = (
        os.getenv("WORLDFOUNDRY_STUDIO_MODEL_REF", "").strip()
        or os.getenv("WORLDFOUNDRY_STUDIO_FIXED_MODEL_REF", "").strip()
    )
    if global_override:
        return global_override
    return fallback


def _prefer_existing_model_ref(*candidates: str) -> str:
    for candidate in candidates:
        if not candidate:
            continue
        if Path(candidate).expanduser().exists():
            return str(Path(candidate).expanduser())
    for candidate in candidates:
        if candidate:
            return candidate
    return ""


def _checkpoint_model_ref(*repo_dir_names: str, fallback: str = "") -> str:
    snapshot = _CATALOG_CACHE_ROOTS.get()
    candidates: list[str] = []
    for repo_dir_name in repo_dir_names:
        if not repo_dir_name:
            continue
        candidates.extend(
            str(path)
            for path in checkpoint_root_candidates(*Path(repo_dir_name).parts)
        )
        candidates.extend(_cache_candidates(repo_dir_name))
    for candidate in candidates:
        if candidate:
            path = Path(candidate).expanduser()
            exists = snapshot.exists(path) if snapshot is not None else path.exists()
            if exists:
                return str(path)
    if fallback:
        return fallback
    return next((candidate for candidate in candidates if candidate), "")


def _local_root_with_files(
    candidates: Iterable[str | Path],
    required_files: Sequence[str | Path],
) -> str:
    """Return the first local component root containing every required file."""

    for raw_candidate in candidates:
        if not raw_candidate:
            continue
        candidate = Path(raw_candidate).expanduser()
        if candidate.is_dir() and all((candidate / Path(name)).is_file() for name in required_files):
            return str(candidate.resolve())
    return ""


def _hunyuanvideo15_default_load_kwargs(*, image_to_video: bool) -> Dict[str, Any]:
    """Compose the complete HunyuanVideo 1.5 demo from staged local roles.

    The release checkpoint and the HY-WorldPlay checkout are complementary in
    common offline installations: the former contains the full non-distilled
    720p DiTs and materializer-facing text resources, while the latter contains
    the SigLIP assets. Keep those roles explicit instead of copying tens of
    gigabytes or silently falling back to the available 480p model.
    """

    checkpoint_ref = _checkpoint_model_ref(
        "tencent--HunyuanVideo-1.5",
        "HunyuanVideo-1.5",
        fallback="tencent/HunyuanVideo-1.5",
    )
    transformer_dir = "720p_i2v" if image_to_video else "720p_t2v"
    weights_root = _local_root_with_files(
        (
            checkpoint_ref,
            *checkpoint_root_candidates("tencent--HunyuanVideo-1.5"),
            *checkpoint_root_candidates("HunyuanVideo-1.5"),
        ),
        (
            f"transformer/{transformer_dir}/config.json",
            f"transformer/{transformer_dir}/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )

    resource_ref = _checkpoint_model_ref(
        "HunyuanVideo-1.5",
        "tencent--HunyuanVideo-1.5",
        fallback=checkpoint_ref,
    )
    worldplay_root = _project_root().parent / "world_models" / "hy-worldplay" / "HunyuanVideo-1.5"
    resources_root = _local_root_with_files(
        (
            resource_ref,
            worldplay_root,
            weights_root,
        ),
        (
            "config.json",
            "text_encoder/llm/config.json",
            "text_encoder/byt5-small/config.json",
            "text_encoder/byt5-small/model.safetensors",
            "text_encoder/Glyph-SDXL-v2/checkpoints/model.safetensors",
            "text_encoder/Glyph-SDXL-v2/assets/color_idx.json",
            "text_encoder/Glyph-SDXL-v2/assets/multilingual_10-lang_idx.json",
        ),
    )

    checkpoint_overrides: Dict[str, str] = {}
    if weights_root:
        checkpoint_overrides.update({"transformer": weights_root, "vae": weights_root})
    if resources_root:
        checkpoint_overrides["resources"] = resources_root
        if image_to_video:
            vision_root = _local_root_with_files(
                (
                    Path(resource_ref) / "vision_encoder" / "siglip",
                    worldplay_root / "vision_encoder" / "siglip",
                    Path(resources_root) / "vision_encoder" / "siglip",
                ),
                (
                    "image_encoder/config.json",
                    "image_encoder/model.safetensors",
                    "feature_extractor/preprocessor_config.json",
                ),
            )
            if vision_root:
                checkpoint_overrides["vision"] = vision_root

    load_kwargs: Dict[str, Any] = {"attention_backend": "flash"}
    if checkpoint_overrides:
        load_kwargs["checkpoint_overrides"] = checkpoint_overrides
    return load_kwargs


def _hf_checkpoint_model_ref(repo_dir_name: str, repo_id: str) -> str:
    """Return a local HF cache path only when it exists, otherwise use the repo id."""
    candidates = [
        *(str(path) for path in checkpoint_root_candidates(*Path(repo_dir_name).parts)),
        *_cache_candidates(repo_dir_name),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().exists():
            return str(Path(candidate).expanduser())
    return repo_id


def _hf_checkpoint_model_ref_at_revision(repo_dir_name: str, repo_id: str, revision: str) -> str:
    """Return a local checkpoint only when its immutable revision is verifiable."""

    candidates = [
        *(str(path) for path in checkpoint_root_candidates(f"{repo_dir_name}-{revision[:8]}")),
        *(str(path) for path in checkpoint_root_candidates(*Path(repo_dir_name).parts)),
        *_cache_candidates(repo_dir_name, repo_id.replace("/", "--")),
    ]
    for raw_candidate in candidates:
        candidate = Path(raw_candidate).expanduser()
        if not candidate.exists():
            continue
        if candidate.parent.name == "snapshots" and candidate.name == revision:
            return str(candidate)
        metadata_path = candidate / ".hfd" / "repo_metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(metadata.get("sha") or "").strip() == revision:
            return str(candidate)
    return repo_id


def _gamma_world_default_load_kwargs() -> dict[str, Any]:
    return {
        "mode": "causal_few_step",
        "model_path": _checkpoint_model_ref(
            "Gamma-World",
            "chijw--Gamma-World",
            fallback="chijw/Gamma-World",
        ),
        "text_encoder_path": _checkpoint_model_ref(
            "Cosmos-Reason1-7B",
            "nvidia--Cosmos-Reason1-7B",
            fallback="nvidia/Cosmos-Reason1-7B",
        ),
    }


def _stable_video_infinity_default_ref() -> str:
    filename = "version-2.0/SVI_Wan2.1-I2V-14B_lora_v2.0.safetensors"
    return _checkpoint_model_ref(
        f"vita-video-gen--svi-model/{filename}",
        f"epfl-vita--svi-model/{filename}",
        fallback="epfl-vita/svi-model",
    )


def _stable_video_infinity_default_load_kwargs() -> dict[str, Any]:
    return {
        "wan_model_dir": _checkpoint_model_ref(
            "Wan2.1-I2V-14B-480P",
            "Wan-AI--Wan2.1-I2V-14B-480P",
            fallback="Wan-AI/Wan2.1-I2V-14B-480P",
        ),
        "svi_lora_path": _stable_video_infinity_default_ref(),
        "dtype": "bfloat16",
        "enable_vram_management": True,
    }


def _yume_1p5_default_ref() -> str:
    return _checkpoint_model_ref(
        "Yume-5B-720P",
        "stdstu123--Yume-5B-720P",
        fallback="stdstu123/Yume-5B-720P",
    )


def _yume_default_ref() -> str:
    return _checkpoint_model_ref(
        "Yume-I2V-540P",
        "stdstu123--Yume-I2V-540P",
        fallback="stdstu123/Yume-I2V-540P",
    )


def _pi3_default_ref() -> str:
    try:
        return str(
            resolve_local_hf_model_path(
                "yyfz233/Pi3X",
                required_files=("config.json", "model.safetensors"),
            )
        )
    except FileNotFoundError:
        return "yyfz233/Pi3X"


def _depth_anything_v2_default_ref() -> str:
    repo_id = "depth-anything/Depth-Anything-V2-Large"
    try:
        return str(
            resolve_local_hf_model_path(
                repo_id,
                required_files=("depth_anything_v2_vitl.pth",),
            )
        )
    except FileNotFoundError:
        return repo_id


def _hydra_default_ref() -> str:
    return _checkpoint_model_ref(
        "H-EmbodVis--HyDRA/hydra.ckpt",
        fallback=str(hfd_root_path() / "H-EmbodVis--HyDRA" / "hydra.ckpt"),
    )


def _hydra_default_load_kwargs() -> dict[str, str]:
    return {
        "base_model_path": _checkpoint_model_ref(
            "Wan-AI--Wan2.1-T2V-1.3B",
            "Wan2.1-T2V-1.3B",
            fallback=str(checkpoint_root_path() / "Wan2.1-T2V-1.3B"),
        )
    }


def _magicworld_default_ref() -> str:
    return _checkpoint_model_ref(
        "LuckyLiGY--MagicWorld",
        fallback=str(hfd_root_path() / "LuckyLiGY--MagicWorld"),
    )


def _magicworld_default_load_kwargs() -> dict[str, str]:
    return {
        "base_model_path": _checkpoint_model_ref(
            "alibaba-pai--Wan2.1-Fun-V1.1-1.3B-InP",
            fallback=str(hfd_root_path() / "alibaba-pai--Wan2.1-Fun-V1.1-1.3B-InP"),
        )
    }


def _official_repo_ref(*repo_names: str, fallback: str = "") -> str:
    candidates: list[str] = []
    for repo_name in repo_names:
        if not repo_name:
            continue
        candidates.append(str(official_runtime_repo_path(repo_name)))
    if fallback:
        candidates.append(fallback)
    return _prefer_existing_model_ref(*candidates)


def _vmem_default_load_kwargs() -> dict[str, Any]:
    runtime_root = str(
        _package_root()
        / "synthesis"
        / "visual_generation"
        / "vmem"
        / "vmem_runtime"
    )
    surfel_model_path = _checkpoint_model_ref(
        "cut3r",
        "CUT3R",
        "liguang0115--cut3r",
        "hfd/liguang0115--cut3r",
        fallback="liguang0115/cut3r",
    )
    return {
        "required_components": {
            "runtime_root": runtime_root,
            "surfel_model_path": surfel_model_path,
        }
    }


def _vmem_default_ref() -> str:
    return _checkpoint_model_ref(
        "vmem",
        "VMem",
        "liguang0115--vmem",
        "hfd/liguang0115--vmem",
        fallback="liguang0115/vmem",
    )


def _current_python_executable() -> str:
    return str(Path(sys.executable).expanduser())


def _worldfoundry_unified_python_executable() -> str:
    override = os.getenv("WORLDFOUNDRY_UNIFIED_PYTHON_EXECUTABLE", "").strip()
    if override:
        return override
    candidate = _project_root().parent / "conda" / "envs" / "worldfoundry-unified-cu128" / "bin" / "python"
    if candidate.is_file():
        return str(candidate)
    opt_candidate = Path("/opt/conda/bin/python3.11")
    return str(opt_candidate) if opt_candidate.is_file() else _current_python_executable()


def _current_torchrun_executable() -> str:
    return str(Path(sys.executable).expanduser().with_name("torchrun"))


def _lingbot_world_default_ref() -> str:
    return _prefer_existing_model_ref(
        *_cache_candidates(
            "robbyant--lingbot-world-base-cam",
            "lingbot-world-base-cam",
        ),
        "robbyant/lingbot-world-base-cam",
    )


def _matrix_game_2_default_ref() -> str:
    return _prefer_existing_model_ref(
        *_cache_candidates("Skywork--Matrix-Game-2.0", "Matrix-Game-2.0"),
        "Skywork/Matrix-Game-2.0",
    )


def _first_existing_path(*candidates: str | Path) -> str:
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
    return str(Path(candidates[0]).expanduser()) if candidates else ""


def _matrix_game_2_default_config_path() -> str:
    return _first_existing_path(
        _data_path("models", "runtime", "configs", "matrix_game_2", "inference_yaml", "inference_universal.yaml"),
    )


def _matrix_game_2_default_load_kwargs() -> Dict[str, Any]:
    return {"mode": "universal"}


def _matrix_game_2_default_call_kwargs() -> Dict[str, Any]:
    return {
        "num_frames": 150,
        "size": [352, 640],
        "fps": 12,
        "seed": 42,
        "official_bench_actions": False,
        "visualize_ops": False,
        "visualize_warning": False,
        "config_path": _matrix_game_2_default_config_path(),
    }


def _matrix_game_3_default_ref() -> str:
    return _prefer_existing_model_ref(
        *_cache_candidates("Matrix-Game-3.0", "Skywork--Matrix-Game-3.0"),
        "Skywork/Matrix-Game-3.0",
    )


MATRIX_GAME_3_OFFICIAL_PROMPT = "A colorful, animated cityscape with a gas station and various buildings."


def _matrix_game_3_default_call_kwargs() -> Dict[str, Any]:
    return {
        "num_iterations": 12,
        "num_inference_steps": 3,
        "size": "704*1280",
        "fps": 17,
        "seed": 42,
        "save_name": "test",
        "visualize_ops": True,
        "show_progress": True,
        "use_int8": True,
        "compile_vae": True,
        "lightvae_pruning_rate": 0.5,
        "vae_type": "mg_lightvae",
        "fa_version": "3",
        "use_async_vae": False,
        "async_vae_warmup_iters": 0,
    }


def _matrix_game_35_third_person_default_camera() -> Dict[str, Any]:
    """Return a deterministic one-block c2w dolly trajectory for the Studio demo.

    Matrix-Game 3.5 consumes 84 generated poses plus its anchor/padding poses.
    Keeping the camera payload as JSON lets Workspace expose a complete default
    without checking a generated binary NPZ file into the source tree; the
    runtime materializes the mapping as ``camera.npz`` in the run directory.
    Intrinsics are expressed in pixels of the 1152x1536 portrait demo image.
    """

    pose_count = 86
    extrinsics_c2w = []
    for index in range(pose_count):
        progress = index / (pose_count - 1)
        # A gentle right-and-forward camera dolly keeps the reference subject
        # in frame while exercising non-static third-person camera control.
        extrinsics_c2w.append(
            [
                [1.0, 0.0, 0.0, 0.08 * progress],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.35 * progress],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    return {
        "extrinsics_c2w": extrinsics_c2w,
        "intrinsics": [1100.0, 1100.0, 576.0, 768.0],
    }


def _matrix_game_35_third_person_default_call_kwargs() -> Dict[str, Any]:
    return {
        "camera": _matrix_game_35_third_person_default_camera(),
        "num_blocks": 1,
        "steps": 25,
        "cfg_scale": 5.0,
        "seed": 3407,
        "camera_convention": "c2w",
        "fps": 16,
    }


def _matrix_game_35_default_ref(person: str) -> str:
    filename = f"{person}-person.safetensors"
    return _checkpoint_model_ref(
        f"RiemannDynamics--Matrix-Game-3.5-Base/{filename}",
        f"Matrix-Game-3.5/Matrix-Game-3.5-Base/{filename}",
        f"Matrix-Game-3.5/{filename}",
        fallback="RiemannDynamics/Matrix-Game-3.5-Base",
    )


def _matrix_game_35_default_load_kwargs() -> Dict[str, Any]:
    return {
        "required_components": {
            "wan_dir": _checkpoint_model_ref(
                "Wan-AI--Wan2.2-TI2V-5B",
                "Wan2.2-TI2V-5B",
                fallback="Wan-AI/Wan2.2-TI2V-5B",
            ),
            "da3_dir": _checkpoint_model_ref(
                "depth-anything--DA3NESTED-GIANT-LARGE-1.1",
                "DA3NESTED-GIANT-LARGE-1.1",
                fallback="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
            ),
            "python_executable": _worldfoundry_unified_python_executable(),
        }
    }


def _matrix_game_3_default_load_kwargs() -> Dict[str, Any]:
    return {
        "required_components": {
            "checkpoint_dir": _prefer_existing_model_ref(
                *_cache_candidates("Matrix-Game-3.0", "Skywork--Matrix-Game-3.0"),
            ),
            "vae_type": "mg_lightvae",
            "lightvae_pruning_rate": 0.5,
            "use_int8": True,
            "compile_vae": True,
            "use_async_vae": False,
            "async_vae_warmup_iters": 0,
        }
    }


def _matrix_game_1_default_ref() -> str:
    return _prefer_existing_model_ref(
        *_cache_candidates("Skywork--Matrix-Game", "Matrix-Game"),
        "Skywork/Matrix-Game",
    )


def _matrix_game_1_default_load_kwargs() -> Dict[str, Any]:
    return {
        "conda_dir": _prefer_existing_model_ref(
            os.getenv("WORLDFOUNDRY_MATRIX_GAME_1_CONDA_DIR", ""),
            str(_project_root().parent / "envs" / "matrix-game-1"),
            str(_project_root().parent / "envs" / "matrix-game-1.0"),
            str(_project_root().parent / "conda" / "envs" / "worldfoundry-unified-cu128"),
            str(_project_root().parent / "conda" / "envs" / "matrix-game-1.0"),
        ),
    }


def _causal_rcm_default_load_kwargs() -> Dict[str, Any]:
    checkpoint_dir = checkpoint_root_path("rcm")
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "dit_path": str(checkpoint_dir / "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step4.pt"),
        "vae_path": str(checkpoint_dir / "Wan2.1_VAE.pth"),
        "text_encoder_path": str(checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
    }


def _causal_rcm_default_call_kwargs() -> Dict[str, Any]:
    return {
        # Frame-wise c1-1 chunks: the lowest-latency streaming schedule in the
        # Causal-rCM recipe, with the recommended 4-step distilled midpoints.
        "distilled": True,
        "first_chunk_t": 1,
        "chunk_t": 1,
        "num_steps": 4,
        "mid_t": ["15/16", "5/6", "5/8"],
        "num_frames": 81,
        "resolution": "480p",
        "aspect_ratio": "16:9",
        "model_size": "1.3B",
        "seed": 0,
        # The checkpoint is not bundled with WorldFoundry, so Studio plans by
        # default. Set plan_only=False after staging a DiT checkpoint.
        "plan_only": True,
        "timeout_seconds": 21600,
    }


def _open_dreamer_default_load_kwargs() -> Dict[str, Any]:
    load_kwargs: Dict[str, Any] = {"checkpoint_path": str(checkpoint_root_path("open-dreamer"))}
    # Only pin an interpreter once the checkout's uv venv exists. Otherwise leave
    # the key unset so the runtime adapter re-probes at call time.
    venv_python = official_runtime_repo_path("open-dreamer-inference") / ".venv" / "bin" / "python"
    if venv_python.is_file():
        load_kwargs["python_executable"] = str(venv_python)
    return load_kwargs


def _open_dreamer_default_call_kwargs() -> Dict[str, Any]:
    return {
        # Matches the official larger-rollout example in the upstream inference README.
        "context_frames": 16,
        "horizon": 64,
        "num_steps": 4,
        "decode_chunk_size": 16,
        "seed": 0,
        "use_ema": True,
        "camera_step_degrees": 5.0,
        # Upstream publishes no downloadable weights, so Studio plans by default and
        # only executes once a checkout plus checkpoint are staged.
        "plan_only": True,
        "timeout_seconds": 21600,
    }


def _lingbot_va_default_call_kwargs() -> Dict[str, Any]:
    demo_root = _data_path("test_cases", "test_vla_case1", "libero")
    return {
        "operator_kwargs": {
            "lingbot_va_rgb_views": {
                "observation.images.agentview_rgb": str(demo_root / "main_view.png"),
                "observation.images.eye_in_hand_rgb": str(demo_root / "wrist_view.png"),
            },
            "obs_cam_keys": (
                "observation.images.agentview_rgb",
                "observation.images.eye_in_hand_rgb",
            ),
        }
    }


def _giga_brain_0_default_call_kwargs() -> Dict[str, Any]:
    demo_root = _data_path("test_cases", "test_vla_case1", "aloha")
    image_keys = (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )
    return {
        "compile_policy": False,
        "operator_kwargs": {
            "giga_brain_0_rgb_views": {
                image_keys[0]: str(demo_root / "observation_images_cam_high.png"),
                image_keys[1]: str(demo_root / "observation_images_cam_left_wrist.png"),
                image_keys[2]: str(demo_root / "observation_images_cam_right_wrist.png"),
            },
            "image_keys": image_keys,
            "state": [0.0] * 14,
            "embodiment_id": 0,
            "original_action_dim": 14,
            "action_chunk": 50,
        }
    }


def _openvla_default_ref() -> str:
    return _checkpoint_model_ref(
        "openvla-7b",
        "openvla--openvla-7b",
        fallback="openvla/openvla-7b",
    )


def _openvla_default_call_kwargs() -> Dict[str, Any]:
    return {
        "checkpoint_dir": _openvla_default_ref(),
        "unnorm_key": "bridge_orig",
        "torch_dtype": "auto",
        "attn_implementation": "eager",
    }


def _openvla_oft_default_call_kwargs() -> Dict[str, Any]:
    libero_root = _data_path("test_cases", "test_vla_case1", "libero")
    return {
        "official_policy_observation": {
            "full_image": str(libero_root / "main_view.png"),
            "wrist_image": str(libero_root / "wrist_view.png"),
            "state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "task_description": "put the object on the target area",
        },
        "unnorm_key": "libero_spatial_no_noops",
        "task_suite_name": "libero_spatial",
        "num_images_in_input": 2,
        "use_proprio": True,
        "torch_dtype": "bfloat16",
        "attn_implementation": "eager",
    }


def _openvla_oft_variant_call_kwargs(
    *,
    unnorm_key: str,
    task_suite_name: str,
) -> Dict[str, Any]:
    """Return the executable Studio input contract shared by OFT variants."""

    values = _openvla_oft_default_call_kwargs()
    values.update(unnorm_key=unnorm_key, task_suite_name=task_suite_name)
    return values


def _checkpoint_file_if_present(*relative_paths: str) -> str | None:
    for relative_path in relative_paths:
        path = checkpoint_root_path(*Path(relative_path).parts)
        if path.is_file():
            return str(path)
    return None


def _libero_policy_observation(prompt: str) -> Dict[str, Any]:
    libero_root = _data_path("test_cases", "test_vla_case1", "libero")
    return {
        "full_image": str(libero_root / "main_view.png"),
        "wrist_image": str(libero_root / "wrist_view.png"),
        "state": [0.0] * 8,
        "task_description": prompt,
        "prompt": prompt,
    }


def _cogact_default_ref() -> str:
    return _prefer_existing_model_ref(
        str(checkpoint_root_path("hfd_models", "CogACT--CogACT-Base")),
        str(checkpoint_root_path("hfd", "CogACT--CogACT-Base")),
        "CogACT/CogACT-Base",
    )


def _last_r1_default_ref() -> str:
    suffix = Path("chenhao01--LaST-R1") / "LaST-R1-RL" / "last-r1-rl-libero_10"
    configured = resolve_worldfoundry_path(Path("${WORLDFOUNDRY_CKPT_DIR}") / suffix)
    if not (configured / "config.json").is_file() and not os.environ.get("WORLDFOUNDRY_CKPT_DIR"):
        sibling = PIPELINES_ROOT.parents[2] / "ckpts" / suffix
        if (sibling / "config.json").is_file():
            return str(sibling)
    return str(configured)


def _last_r1_default_load_kwargs() -> Dict[str, Any]:
    source = official_runtime_repo_path("LaST-R1", specific_env="WORLDFOUNDRY_LAST_R1_SOURCE_DIR")
    if not source.is_dir() and not os.environ.get("WORLDFOUNDRY_LAST_R1_SOURCE_DIR"):
        sibling = PIPELINES_ROOT.parents[2] / "model" / "LaST-R1"
        if sibling.is_dir():
            source = sibling
    return {"source_root": str(source)}


def _cogact_default_call_kwargs() -> Dict[str, Any]:
    prompt = "move sponge near apple"
    image_path = str(_data_path("test_cases", "test_vla_image_case1", "init_frame.png"))
    kwargs: Dict[str, Any] = {
        "official_policy_observation": {
            "image": image_path,
            "full_image": image_path,
            "prompt": prompt,
            "task_description": prompt,
            "unnorm_key": "bridge_orig",
        },
        "unnorm_key": "bridge_orig",
        "cfg_scale": 1.5,
        "use_ddim": True,
        "num_ddim_steps": 10,
        "action_model_type": "DiT-B",
        "future_action_window_size": 15,
    }
    checkpoint = _checkpoint_file_if_present(
        "hfd_models/CogACT--CogACT-Base/checkpoints/CogACT-Base.pt",
        "hfd/CogACT--CogACT-Base/checkpoints/CogACT-Base.pt",
    )
    if checkpoint:
        kwargs["checkpoint_path"] = checkpoint
    else:
        kwargs["checkpoint_ref"] = "CogACT/CogACT-Base"
    return kwargs


def _db_cogact_default_ref() -> str:
    return _prefer_existing_model_ref(
        str(checkpoint_root_path("hfd_models", "Dexmal--libero-db-cogact")),
        str(checkpoint_root_path("hfd", "Dexmal--libero-db-cogact")),
        "Dexmal/libero-db-cogact",
    )


def _db_cogact_default_call_kwargs() -> Dict[str, Any]:
    prompt = "put the object on the target area"
    libero_root = _data_path("test_cases", "test_vla_case1", "libero")
    return {
        "official_policy_observation": {
            "prompt": prompt,
            "image/1": str(libero_root / "main_view.png"),
            "image/2": str(libero_root / "wrist_view.png"),
        },
        "camera_order": ["front", "left_wrist"],
        "num_steps": 10,
        "cfg_scale": 1.5,
        "seed": 42,
    }


def _vlanext_default_ref() -> str:
    checkpoint = _checkpoint_file_if_present(
        "hfd_models/DravenALG--VLANeXt/VLANeXt_libero_spatial.pt",
        "hfd/DravenALG--VLANeXt/VLANeXt_libero_spatial.pt",
    )
    if checkpoint:
        return str(Path(checkpoint).parent)
    return "DravenALG/VLANeXt"


def _vlanext_default_call_kwargs() -> Dict[str, Any]:
    prompt = "put the object on the target area"
    kwargs: Dict[str, Any] = {
        "official_policy_observation": _libero_policy_observation(prompt),
        "task_suite_name": "libero_spatial",
        "diffusion_steps": 10,
        "num_steps_execute": 1,
    }
    checkpoint = _checkpoint_file_if_present(
        "hfd_models/DravenALG--VLANeXt/VLANeXt_libero_spatial.pt",
        "hfd/DravenALG--VLANeXt/VLANeXt_libero_spatial.pt",
    )
    if checkpoint:
        kwargs["checkpoint_path"] = checkpoint
    else:
        kwargs["checkpoint_ref"] = "DravenALG/VLANeXt"
    return kwargs


def _molmobot_default_ref() -> str:
    repo_root = checkpoint_root_path("hfd_models", "allenai--MolmoBot-DROID")
    if (repo_root / "config.yaml").is_file():
        return str(repo_root)
    repo_root = checkpoint_root_path("hfd", "allenai--MolmoBot-DROID")
    if (repo_root / "config.yaml").is_file():
        return str(repo_root)
    return "allenai/MolmoBot-DROID"


def _molmobot_default_call_kwargs() -> Dict[str, Any]:
    droid_root = _data_path("test_cases", "test_vla_case1", "droid")
    prompt = "put the mug in the bowl"
    return {
        "official_policy_observation": {
            "exo_camera_1": str(droid_root / "exterior_image_1_left.png"),
            "wrist_camera": str(droid_root / "wrist_image_left.png"),
            "qpos": [0.0] * 8,
            "task": prompt,
            "task_description": prompt,
            "camera_keys": ["exo_camera_1", "wrist_camera"],
        },
        "camera_keys": ["exo_camera_1", "wrist_camera"],
        "norm_repo_id": "synthmanip",
        "use_bfloat16": True,
        "compile_model": False,
    }


def _mme_vla_default_ref() -> str:
    step_dir = checkpoint_root_path("hfd_models", "Yinpei--perceptual-framesamp-modul", "79999")
    if step_dir.is_dir():
        return str(step_dir)
    zip_path = checkpoint_root_path("hfd_models", "Yinpei--perceptual-framesamp-modul", "79999.zip")
    if zip_path.is_file():
        return str(zip_path.parent)
    return "Yinpei/perceptual-framesamp-modul"


def _mme_vla_default_call_kwargs() -> Dict[str, Any]:
    libero_root = _data_path("test_cases", "test_vla_case1", "libero")
    prompt = "put the object on the target area"
    kwargs: Dict[str, Any] = {
        "official_policy_observation": {
            "observation/image": str(libero_root / "main_view.png"),
            "observation/wrist_image": str(libero_root / "wrist_view.png"),
            "observation/state": [0.0] * 8,
            "prompt": prompt,
        },
        "policy_config": "mme_vla_suite",
        "seed": 7,
    }
    step_dir = checkpoint_root_path("hfd_models", "Yinpei--perceptual-framesamp-modul", "79999")
    if step_dir.is_dir():
        kwargs["checkpoint_path"] = str(step_dir)
    else:
        kwargs["checkpoint_ref"] = "Yinpei/perceptual-framesamp-modul"
    return kwargs


def _starvla_default_ref() -> str:
    return _checkpoint_model_ref(
        "Qwen3-VL-OFT-LIBERO-4in1",
        "StarVLA--Qwen3-VL-OFT-LIBERO-4in1",
        fallback="StarVLA/Qwen3-VL-OFT-LIBERO-4in1",
    )


def _starvla_default_call_kwargs() -> Dict[str, Any]:
    in_tree_source = _project_root() / "worldfoundry" / "synthesis" / "action_generation" / "starvla"
    return {
        "checkpoint_dir": _starvla_default_ref(),
        "base_vlm": _checkpoint_model_ref(
            "Qwen3-VL-4B-Instruct",
            "Qwen--Qwen3-VL-4B-Instruct",
            fallback="Qwen/Qwen3-VL-4B-Instruct",
        ),
        "source_repo_dir": str(in_tree_source),
        "enable_official_runtime": True,
        "attn_implementation": "sdpa",
    }


def _wan_2p1_t2v_default_ref() -> str:
    return _checkpoint_model_ref(
        "Wan-AI--Wan2.1-T2V-1.3B",
        "Wan2.1-T2V-1.3B",
        fallback="Wan-AI/Wan2.1-T2V-1.3B",
    )


def _wan_2p1_i2v_default_ref() -> str:
    return _checkpoint_model_ref(
        "Wan2.1-I2V-14B-480P",
        "Wan-AI--Wan2.1-I2V-14B-480P",
        fallback="Wan-AI/Wan2.1-I2V-14B-480P",
    )


def _wan_2p1_i2v_720p_default_ref() -> str:
    return _checkpoint_model_ref(
        "Wan-AI--Wan2.1-I2V-14B-720P",
        "Wan2.1-I2V-14B-720P",
        fallback="Wan-AI/Wan2.1-I2V-14B-720P",
    )


def _dualcamctrl_base_default_ref() -> str:
    repo_id = "alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera"
    model_root = os.getenv("WORLDFOUNDRY_MODEL_DIR", "").strip()
    candidates: list[str] = []
    if model_root:
        candidates.append(str(Path(model_root) / "alibaba-pai" / "Wan2.1-Fun-V1.1-1.3B-Control-Camera"))
    candidates.extend(
        _cache_candidates(
            "alibaba-pai--Wan2.1-Fun-V1.1-1.3B-Control-Camera",
            "Wan2.1-Fun-V1.1-1.3B-Control-Camera",
        )
    )
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_dir() and any(path.glob("diffusion_pytorch_model*.safetensors")):
            return str(path)
    return repo_id


def _dualcamctrl_default_load_kwargs() -> Dict[str, Any]:
    load_kwargs: Dict[str, Any] = {
        "base_model_repo": "alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera",
        "tokenizer_repo": "Wan-AI/Wan2.1-T2V-1.3B",
        "dualcamctrl_repo": "FayeHongfeiZhang/DualCamCtrl",
        "checkpoint_name": "checkpoints/dualcamctrl_diffusion_transformer.pt",
        "config_path": str(_data_path("models", "runtime", "configs", "dualcamctrl", "controlnet_gate_asym_5_10.yaml")),
        "torch_dtype": "bfloat16",
        "download_resource": "HuggingFace",
        "allow_download": True,
        "copy_control_weights": True,
        "redirect_common_files": True,
    }
    base_model_path = Path(_dualcamctrl_base_default_ref()).expanduser()
    checkpoint_root = checkpoint_root_path()
    tokenizer_root = Path(
        _prefer_existing_model_ref(
            *_cache_candidates("Wan-AI--Wan2.1-T2V-1.3B", "Wan-AI/Wan2.1-T2V-1.3B"),
            str(checkpoint_root / "Wan-AI" / "Wan2.1-T2V-1.3B"),
            str(checkpoint_root / "Wan-AI--Wan2.1-T2V-1.3B"),
        )
    )
    clip_root = Path(
        _prefer_existing_model_ref(
            *_cache_candidates(
                "Wan-AI--Wan2.1-I2V-14B-720P",
                "Wan-AI--Wan2.1-I2V-14B-480P",
                "alibaba-pai--Wan2.1-Fun-V1.1-1.3B-Control-Camera",
            ),
            str(checkpoint_root / "Wan-AI--Wan2.1-I2V-14B-720P"),
            str(checkpoint_root / "Wan-AI--Wan2.1-I2V-14B-480P"),
            str(checkpoint_root / "alibaba-pai--Wan2.1-Fun-V1.1-1.3B-Control-Camera"),
            str(checkpoint_root / "Wan-AI" / "Wan2.1-I2V-14B-480P"),
        )
    )
    dualcam_root = Path(
        _prefer_existing_model_ref(
            *_cache_candidates("FayeHongfeiZhang--DualCamCtrl", "FayeHongfeiZhang/DualCamCtrl"),
            str(checkpoint_root / "FayeHongfeiZhang" / "DualCamCtrl"),
            str(checkpoint_root / "FayeHongfeiZhang--DualCamCtrl"),
        )
    )
    local_files = (
        next(iter(base_model_path.glob("diffusion_pytorch_model*.safetensors")), None)
        if base_model_path.is_dir()
        else None,
        tokenizer_root / "models_t5_umt5-xxl-enc-bf16.pth",
        tokenizer_root / "Wan2.1_VAE.pth",
        tokenizer_root / "google" / "umt5-xxl",
        clip_root / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        dualcam_root / "checkpoints" / "dualcamctrl_diffusion_transformer.pt",
    )
    if all(path is not None and Path(path).exists() for path in local_files):
        load_kwargs.update(
            base_model_path=str(base_model_path),
            local_model_path=str(checkpoint_root),
            allow_download=False,
        )
    return load_kwargs


def _dualcamctrl_default_call_kwargs() -> Dict[str, Any]:
    demo_root = _data_path("test_cases", "dualcamctrl", "demo_pic")
    return {
        "demo_name": "seaside",
        "image_path": str(demo_root / "seaside.png"),
        "depth_image": str(demo_root / "seaside_depth.png"),
        "camera_path": str(demo_root / "seaside.torch"),
        "num_frames": 61,
        "height": 320,
        "width": 480,
        "num_inference_steps": 50,
        "fps": 10,
        "seed": 42,
        "cfg_scale": 5.0,
        "return_control_latents": True,
        "original_height": 360,
        "original_width": 640,
        "tiled": True,
    }


def _wan_2p2_default_ref() -> str:
    return _checkpoint_model_ref(
        "Wan-AI--Wan2.2-TI2V-5B",
        "Wan2.2-TI2V-5B",
        fallback="Wan-AI/Wan2.2-TI2V-5B",
    )


def _wan_2p2_t2v_a14b_default_ref() -> str:
    return _checkpoint_model_ref(
        "Wan2.2-T2V-A14B",
        "Wan-AI--Wan2.2-T2V-A14B",
        fallback="Wan-AI/Wan2.2-T2V-A14B",
    )


def _wan_2p2_i2v_a14b_default_ref() -> str:
    return _checkpoint_model_ref(
        "Wan2.2-I2V-A14B",
        "Wan-AI--Wan2.2-I2V-A14B",
        fallback="Wan-AI/Wan2.2-I2V-A14B",
    )


def _ltx23_default_load_kwargs() -> Dict[str, Any]:
    """Expose every runtime-required LTX-2.3 checkpoint component to Studio."""

    return {
        "required_components": {
            "spatial_upsampler_path": _checkpoint_model_ref(
                "LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
                "Lightricks--LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
                fallback=str(
                    checkpoint_root_path(
                        "Lightricks--LTX-2.3",
                        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
                    )
                ),
            ),
            "gemma_root": _checkpoint_model_ref(
                "gemma-3-12b-it-qat-q4_0-unquantized",
                "google--gemma-3-12b-it-qat-q4_0-unquantized",
                fallback=str(checkpoint_root_path("gemma-3-12b-it-qat-q4_0-unquantized")),
            ),
        }
    }


def _ltx25_checkpoint_dir() -> str:
    return _checkpoint_model_ref(
        "Lightricks--LTX-2.5",
        fallback=str(checkpoint_root_path("Lightricks--LTX-2.5")),
    )


def _ltx25_source_root() -> str:
    return _first_existing_path(
        _project_root() / "cache" / "generative_taxonomy" / "Lightricks--LTX-2",
        official_runtime_repo_path("Lightricks--LTX-2"),
    )


def _ltx25_python_executable() -> str:
    workspace_root = _project_root().parent
    return _first_existing_path(
        workspace_root / "conda" / "envs" / "ltx-2.5" / "bin" / "python",
        workspace_root / "envs" / "ltx-2.5" / "bin" / "python",
    )


def _ltx25_default_load_kwargs() -> Dict[str, Any]:
    """Map Studio directly to the official split LTX-2.5 runtime assets."""

    checkpoint_dir = Path(_ltx25_checkpoint_dir())
    return {
        "required_components": {
            "checkpoint_dir": str(checkpoint_dir),
            "source_root": _ltx25_source_root(),
            "python_executable": _ltx25_python_executable(),
            "transformer_path": str(
                checkpoint_dir
                / "diffusion_models"
                / "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
            ),
            "text_encoder_path": str(
                checkpoint_dir
                / "text_encoders"
                / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
            ),
            "video_vae_path": str(
                checkpoint_dir / "vae" / "ltx-2.5-video-vae-bf16.safetensors"
            ),
            "audio_vae_path": str(
                checkpoint_dir / "vae" / "ltx-2.5-audio-vae-bf16.safetensors"
            ),
            "spatial_upsampler_path": str(
                checkpoint_dir
                / "latent_upscale_models"
                / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
            ),
            "duration_head_path": str(
                checkpoint_dir
                / "model_patches"
                / "ltx-2.5-duration-head-bf16.safetensors"
            ),
            "offload_mode": "cpu",
            "diffvae_optimization": "chunked_eager",
        }
    }


def _official_video_load_params() -> tuple[str, ...]:
    return (
        "model_path",
        "required_components",
        "device",
        "model_id",
        "checkpoint_path",
        "checkpoint_candidates",
        "repo_root",
        "repo_root_candidates",
        "kind",
        "pipeline_target",
        "torch_dtype",
        "fps",
        "torchrun_nproc_per_node",
        "torchrun_nproc",
        "nproc_per_node",
        *DISPATCH_LOAD_PARAMS,
    )


def _official_video_call_params() -> tuple[str, ...]:
    return (
        "prompt",
        "images",
        "video",
        "image_path",
        "video_path",
        "output_path",
        "return_dict",
        "plan_only",
        "num_frames",
        "max_frames",
        "num_blocks",
        "task",
        "num_inference_steps",
        "num_sampling_steps",
        "sampling_steps",
        "infer_steps",
        "base_num_frames",
        "ar_step",
        "causal_block_size",
        "addnoise_condition",
        "fps",
        "target_fps",
        "height",
        "width",
        "resolution",
        "size",
        "aspect_ratio",
        "i2v_mode",
        "i2v_image_path",
        "i2v_resolution",
        "i2v_stability",
        "i2v_condition_type",
        "seed",
        "guidance_scale",
        "cfg_scale",
        "shift",
        "negative_prompt",
        "quality",
        "tiled",
        "prompt_prefix",
        "lora_path",
        "state_dict",
        "time_shift",
        "tensor_parallel_degree",
        "ulysses_degree",
        "ulysses_size",
        "ring_degree",
        "ring_size",
        "parallel",
        "seconds",
        "duration",
        "latent_window_size",
        "num_steps",
        "steps",
        "aes",
        "flow",
        "sample_method",
        "max_sequence_length",
        "config",
        "version",
        "use_usp",
        "nproc_per_node",
        "torchrun_nproc_per_node",
        "torchrun_nproc",
        "model_subdir",
        "vae_subdir",
        "cache_dir",
        "text_encoder_name_1",
        "ae",
        "rewrite",
        "cfg_distilled",
        "enable_step_distill",
        "sparse_attn",
        "use_sageattn",
        "enable_cache",
        "sr",
        "save_pre_sr_video",
        "overlap_group_offloading",
        "model_resolution",
        "flow_shift",
        "embedded_cfg_scale",
        "mode",
        "variant",
        "gpu_memory_preservation",
        "cuda_visible_devices",
        "visible_devices",
        "cuda_devices",
        "gpu_ids",
    )


def _native_diffusion_video_load_params() -> tuple[str, ...]:
    return (
        "model_path",
        "required_components",
        "device",
        "model_id",
        "checkpoint_path",
        "checkpoint_overrides",
        "release_text_encoders_after_encode",
        "torch_dtype",
        "weight_dtype",
        "dtype",
        "offload_mode",
        "attention_backend",
        "quantization",
        "quantization_compute_dtype",
        "quantization_exclude",
        "fp8_min_features",
        "fp8_scaling",
        "scaling",
        "keep_dense_fallback",
        "use_fast_accum",
        "fuse_qkv",
        "qkv_strategy",
        "qkv_split_threshold",
        "inplace_residual",
        "static_cross_kv",
        "fused_rope",
        "rope_precision",
        "rms_norm_precision",
        "compile",
        "approximate_attention",
        "vae_tiling",
        *DISPATCH_LOAD_PARAMS,
    )


def _native_diffusion_video_call_params() -> tuple[str, ...]:
    return (
        "prompt",
        "images",
        "image",
        "image_path",
        "video",
        "interactions",
        "negative_prompt",
        "output_path",
        "output_dir",
        "num_frames",
        "frames",
        "frame_num",
        "height",
        "width",
        "num_inference_steps",
        "infer_steps",
        "guidance_scale",
        "cfg_scale",
        "embedded_guidance_scale",
        "shift",
        "seed",
        "fps",
        "output_type",
        "return_dict",
    )


def _echo_infinity_default_ref() -> str:
    return _checkpoint_model_ref(
        "Echo-Infinity/echo_infinity.pt",
        "Echo-Infinity/checkpoints/echo_infinity.pt",
        fallback="Echo-AI/Echo-Infinity",
    )


def _echo_infinity_default_load_kwargs() -> Dict[str, Any]:
    wan_root = checkpoint_root_path()
    wan_model_name = "Wan2.1-T2V-1.3B"
    found = False
    for candidate_root in checkpoint_root_candidates():
        for candidate_name in ("Wan-AI--Wan2.1-T2V-1.3B", wan_model_name):
            if (candidate_root / candidate_name).is_dir():
                wan_root = candidate_root
                wan_model_name = candidate_name
                found = True
                break
        if found:
            break
    return {
        "wan_root": str(wan_root),
        "wan_model_name": wan_model_name,
        "generator_ckpt": _echo_infinity_default_ref(),
        "model_kwargs": {
            "model_name": wan_model_name,
            "wan_root": str(wan_root),
            "local_attn_size": 12,
            "timestep_shift": 5.0,
            "sink_size": 3,
        },
    }


def _echo_memory_default_ref(checkpoint_file: str) -> str:
    relative = Path(checkpoint_file)
    return _checkpoint_model_ref(
        str(Path("Echo-Team--Echo-Memory") / relative),
        str(Path("Echo-Memory") / relative),
        fallback=str(hfd_root_path("Echo-Team--Echo-Memory", *relative.parts)),
    )


def _echo_memory_context_k1_default_ref() -> str:
    return _echo_memory_default_ref("context_k1/epoch-0.safetensors")


def _echo_memory_default_load_kwargs() -> Dict[str, Any]:
    return {
        "wan_base_dir": _checkpoint_model_ref(
            "Wan-AI--Wan2.1-T2V-1.3B",
            "Wan2.1-T2V-1.3B",
            fallback=str(hfd_root_path("Wan-AI--Wan2.1-T2V-1.3B")),
        )
    }


def _echo_memory_context_k1_load_kwargs() -> Dict[str, Any]:
    return _echo_memory_default_load_kwargs()


def _recammaster_default_ref() -> str:
    return _checkpoint_model_ref(
        "KlingTeam--ReCamMaster-Wan2.1",
        "ReCamMaster-Wan2.1",
        fallback="KlingTeam/ReCamMaster-Wan2.1",
    )


def _recammaster_default_load_kwargs() -> Dict[str, Any]:
    return {
        "wan_model_path": _wan_2p1_t2v_default_ref(),
        "recammaster_ckpt_path": _recammaster_default_ref(),
    }


def _vchitect2_default_ref() -> str:
    return _checkpoint_model_ref(
        "Vchitect--Vchitect-2.0-2B",
        "Vchitect-2.0-2B",
        fallback="Vchitect/Vchitect-2.0-2B",
    )


def _t2v_turbo_base_ref() -> str:
    return _checkpoint_model_ref(
        "VideoCrafter--VideoCrafter2/model.ckpt",
        "VideoCrafter2/model.ckpt",
        fallback="VideoCrafter/VideoCrafter2",
    )


def _t2v_turbo_lora_ref() -> str:
    return _checkpoint_model_ref(
        "jiachenli-ucsb--T2V-Turbo-VC2/unet_lora.pt",
        "T2V-Turbo-VC2/unet_lora.pt",
        fallback="jiachenli-ucsb/T2V-Turbo-VC2",
    )


def _t2v_turbo_default_load_kwargs() -> Dict[str, Any]:
    return {
        "base_checkpoint": _t2v_turbo_base_ref(),
        "lora_checkpoint": _t2v_turbo_lora_ref(),
    }


def _versecrafter_default_ref() -> str:
    return _checkpoint_model_ref(
        "TencentARC--VerseCrafter",
        "VerseCrafter",
        fallback="TencentARC/VerseCrafter",
    )


def _versecrafter_default_load_kwargs() -> Dict[str, Any]:
    return {
        "base_model_path": _checkpoint_model_ref(
            "Wan-AI--Wan2.1-T2V-14B",
            "Wan2.1-T2V-14B",
            fallback="Wan-AI/Wan2.1-T2V-14B",
        ),
        "moge_model_path": _checkpoint_model_ref(
            "moge-2-vitl-normal",
            "Ruicheng--moge-2-vitl-normal",
            fallback="Ruicheng/moge-2-vitl-normal",
        ),
    }


def _cameractrl_default_ref() -> str:
    return _checkpoint_model_ref(
        "hehao13--CameraCtrl/CameraCtrl.ckpt",
        "CameraCtrl/CameraCtrl.ckpt",
        fallback="hehao13/CameraCtrl",
    )


def _cameractrl_default_load_kwargs() -> Dict[str, Any]:
    sd15_path = _checkpoint_model_ref(
        "stable-diffusion-v1-5--stable-diffusion-v1-5",
        "stable-diffusion-v1-5",
        fallback="stable-diffusion-v1-5/stable-diffusion-v1-5",
    )
    return {
        "sd15_path": sd15_path,
        "pose_adaptor_ckpt": _cameractrl_default_ref(),
        "image_lora_ckpt": _checkpoint_model_ref(
            "hehao13--CameraCtrl/RealEstate10K_LoRA.ckpt",
            "CameraCtrl/RealEstate10K_LoRA.ckpt",
            fallback="hehao13/CameraCtrl",
        ),
        "motion_module_ckpt": _checkpoint_model_ref(
            "guoyww--animatediff/v3_sd15_mm.ckpt",
            "animatediff/v3_sd15_mm.ckpt",
            fallback="guoyww/animatediff",
        ),
        "motion_adapter_ckpt": _checkpoint_model_ref(
            "guoyww--animatediff/v3_sd15_adapter.ckpt",
            "animatediff/v3_sd15_adapter.ckpt",
            fallback="guoyww/animatediff",
        ),
        "unet_subfolder": "unet",
    }


def _cosmos_predict2p5_default_ref() -> str:
    return _checkpoint_model_ref(
        "Cosmos-Predict2.5-2B",
        "nvidia--Cosmos-Predict2.5-2B",
        "huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B",
        fallback="nvidia/Cosmos-Predict2.5-2B",
    )


def _cosmos_predict2_default_ref() -> str:
    return _checkpoint_model_ref(
        "nvidia--Cosmos-Predict2-2B-Video2World",
        "Cosmos-Predict2-2B-Video2World",
        fallback="nvidia/Cosmos-Predict2-2B-Video2World",
    )


def _cosmos_predict2_default_load_kwargs() -> Dict[str, Any]:
    model_root = Path(_cosmos_predict2_default_ref())
    return {
        "checkpoint_path": str(model_root),
        "text_encoder_model_path": str(model_root / "text_encoder"),
        "text_tokenizer_path": str(model_root / "tokenizer"),
        "offload_mode": "block",
    }


def _cosmos_predict2p5_default_load_kwargs() -> Dict[str, Any]:
    model_root = _cosmos_predict2p5_default_ref()
    return {
        "required_components": {
            "text_encoder_model_path": _checkpoint_model_ref(
                "Cosmos-Reason1-7B",
                fallback="nvidia/Cosmos-Reason1-7B",
            ),
            # The Cosmos checkpoint ships its own tokenizer.pth VAE. A Wan
            # checkpoint directory does not satisfy the Cosmos codec contract.
            "vae_model_path": model_root,
        }
    }


def _cosmos_predict2p5_default_call_kwargs() -> Dict[str, Any]:
    return {
        "fps": 16,
        "guidance_scale": 7.0,
        "height": 704,
        "num_frames": 93,
        "num_inference_steps": 35,
        "output_type": "pt",
        "seed": 42,
        "width": 1280,
    }


def _cosmos_transfer2p5_default_ref() -> str:
    return _checkpoint_model_ref(
        "nvidia--Cosmos-Transfer2.5-2B",
        "Cosmos-Transfer2.5-2B",
        fallback="nvidia/Cosmos-Transfer2.5-2B",
    )


def _cosmos_transfer2p5_default_load_kwargs() -> Dict[str, Any]:
    return {
        "required_components": {
            "text_encoder_model_path": _checkpoint_model_ref(
                "Cosmos-Reason1-7B",
                fallback="nvidia/Cosmos-Reason1-7B",
            ),
            "vae_model_path": _cosmos_predict2p5_default_ref(),
        }
    }


def _cosmos_transfer2p5_default_call_kwargs() -> Dict[str, Any]:
    return {
        **_cosmos_predict2p5_default_call_kwargs(),
        "controlnet_variant": "edge",
        "control_context_scale": 1.0,
        "output_type": "video",
    }


def _longcat_video_default_ref() -> str:
    return _checkpoint_model_ref(
        "LongCat-Video",
        "meituan-longcat--LongCat-Video",
        fallback="meituan-longcat/LongCat-Video",
    )


def _lingbot_video_default_ref() -> str:
    return _checkpoint_model_ref(
        "lingbot-video-dense-1.3b",
        "robbyant--lingbot-video-dense-1.3b",
    )


def _lingbot_world_v2_default_ref() -> str:
    return _checkpoint_model_ref(
        "lingbot-world-v2-14b-causal-fast",
        "robbyant--lingbot-world-v2-14b-causal-fast",
        fallback="robbyant/lingbot-world-v2-14b-causal-fast",
    )


def _sana_wm_default_ref() -> str:
    return _checkpoint_model_ref(
        "SANA-WM_bidirectional",
        "Efficient-Large-Model--SANA-WM_bidirectional",
        fallback="Efficient-Large-Model/SANA-WM_bidirectional",
    )


def _sana_wm_streaming_default_ref() -> str:
    return _checkpoint_model_ref(
        "Efficient-Large-Model--SANA-WM_streaming",
        "SANA-WM_streaming",
        fallback="Efficient-Large-Model/SANA-WM_streaming",
    )


def _sana_wm_streaming_default_load_kwargs() -> Dict[str, Any]:
    streaming_root = _sana_wm_streaming_default_ref()
    gemma_root = _checkpoint_model_ref(
        "Efficient-Large-Model--gemma-2-2b-it",
        fallback="Efficient-Large-Model/gemma-2-2b-it",
    )
    return {
        "checkpoint_overrides": {
            "dit": streaming_root,
            "text-encoder": gemma_root,
            "tokenizer": gemma_root,
            "codec": streaming_root,
            "refiner": streaming_root,
            "refiner-connectors": streaming_root,
            "refiner-text-encoder": streaming_root,
        }
    }


def _moverse_default_ref() -> str:
    return _checkpoint_model_ref(
        "Orange-3DV-Team--MoVerse",
        fallback="Orange-3DV-Team/MoVerse",
    )


def _moverse_default_load_kwargs() -> Dict[str, Any]:
    return {
        "flux_path": _checkpoint_model_ref(
            "black-forest-labs--FLUX.1-Fill-dev",
            "FLUX.1-Fill-dev",
            fallback="black-forest-labs/FLUX.1-Fill-dev",
        ),
        "da3_path": _checkpoint_model_ref(
            "depth-anything--DA3NESTED-GIANT-LARGE-1.1",
            "DA3NESTED-GIANT-LARGE-1.1",
            fallback="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        ),
        "wan_path": _checkpoint_model_ref(
            "Wan-AI--Wan2.1-T2V-1.3B",
            "Wan2.1-T2V-1.3B",
            fallback="Wan-AI/Wan2.1-T2V-1.3B",
        ),
    }


def _dreamdojo_default_ref() -> str:
    return _checkpoint_model_ref(
        "nvidia--DreamDojo/2B_GR1_post-train",
        fallback="nvidia/DreamDojo",
    )


def _dreamdojo_dataset_path() -> str:
    return str(
        checkpoint_root_path(
            "nvidia--PhysicalAI-Robotics-GR00T-Teleop-GR1",
            "GR1_robot",
        )
    )


def _dreamdojo_default_load_kwargs() -> Dict[str, Any]:
    root = Path(_dreamdojo_default_ref())
    checkpoint_dir = root / "iter_000050000"
    return {
        "dataset_path": _dreamdojo_dataset_path(),
        # Matrix readiness audits these exact DCP members. The runtime ignores
        # this audit-only option and consumes checkpoints_dir/latest_checkpoint.txt.
        "checkpoint_shards": [
            str(checkpoint_dir / "model" / f"__{index}_0.distcp") for index in range(8)
        ],
    }


def _dreamx_world_default_ref() -> str:
    return _checkpoint_model_ref(
        "DreamX-World-5B-Cam",
        "GD-ML--DreamX-World-5B-Cam",
        fallback="GD-ML/DreamX-World-5B-Cam",
    )


def _dreamx_world_ar_default_ref() -> str:
    return _checkpoint_model_ref(
        "DreamX-World-5B",
        "GD-ML--DreamX-World-5B",
        fallback="GD-ML/DreamX-World-5B",
    )


def _dreamx_world_ar_default_load_kwargs() -> Dict[str, Any]:
    return {
        "wan_model_path": _checkpoint_model_ref(
            "Wan-AI--Wan2.2-TI2V-5B",
            "Wan2.2-TI2V-5B",
            fallback="Wan-AI/Wan2.2-TI2V-5B",
        ),
    }


def _dreamx_world_default_load_kwargs() -> Dict[str, Any]:
    return {
        "wan_model_path": _checkpoint_model_ref(
            "Wan-AI--Wan2.2-TI2V-5B",
            "Wan2.2-TI2V-5B",
            fallback="Wan-AI/Wan2.2-TI2V-5B",
        ),
    }


def _lingbot_world_v2_default_load_kwargs() -> Dict[str, Any]:
    # Conda dispatch already selects the dedicated in-tree LingBot-V2
    # environment.  Leaving python_executable unset makes any fallback runner
    # use that child interpreter instead of escaping back to the unified env.
    return {
        "t5_fsdp": True,
        "dit_fsdp": True,
        "t5_cpu": False,
    }


def _longcat_video_default_load_kwargs() -> Dict[str, Any]:
    return {
        "python_executable": _current_python_executable(),
    }


def _motionctrl_default_ref() -> str:
    return _checkpoint_model_ref(
        "MotionCtrl/motionctrl.pth",
        "custom--MotionCtrl/motionctrl.pth",
        "TencentARC--MotionCtrl/motionctrl.pth",
        fallback="TencentARC/MotionCtrl",
    )


def _astra_default_ref() -> str:
    return _checkpoint_model_ref(
        "Astra",
        "custom--Astra",
        "EvanEternal--Astra",
        fallback="EvanEternal/Astra",
    )


def _astra_default_load_kwargs() -> Dict[str, Any]:
    return {
        "required_components": {
            "wan_model_path": _checkpoint_model_ref(
                "Wan2.1-T2V-1.3B",
                "Wan-AI--Wan2.1-T2V-1.3B",
                fallback="Wan-AI/Wan2.1-T2V-1.3B",
            )
        }
    }


ASTRA_OFFICIAL_PROMPT = (
    "A sunlit European street lined with historic buildings and vibrant greenery creates a warm, "
    "charming, and inviting atmosphere. The scene shows a picturesque open square paved with red "
    "bricks, surrounded by classic narrow townhouses featuring tall windows, gabled roofs, and "
    "dark-painted facades. On the right side, a lush arrangement of potted plants and blooming "
    "flowers adds rich color and texture to the foreground. A vintage-style streetlamp stands "
    "prominently near the center-right, contributing to the timeless character of the street. "
    "Mature trees frame the background, their leaves glowing in the warm afternoon sunlight. "
    "Bicycles are visible along the edges of the buildings, reinforcing the urban yet leisurely "
    "feel. The sky is bright blue with scattered clouds, and soft sun flares enter the frame from "
    "the left, enhancing the scene's inviting, peaceful mood."
)


def _astra_default_call_kwargs() -> Dict[str, Any]:
    return {
        "frames_per_generation": 8,
        "total_frames_to_generate": 24,
        "num_inference_steps": 50,
        "start_frame": 0,
        "initial_condition_frames": 1,
        "modality_type": "sekai",
        "fps": 20,
    }


def _pusa_vidgen_default_ref() -> str:
    return _checkpoint_model_ref(
        "Pusa-Wan2.2-V1",
        "RaphaelLiu--Pusa-Wan2.2-V1",
        fallback="RaphaelLiu/Pusa-Wan2.2-V1",
    )


def _pusa_vidgen_default_load_kwargs() -> Dict[str, Any]:
    return {
        "base_model_root": _checkpoint_model_ref(
            "Wan2.2-T2V-A14B",
            "Wan-AI--Wan2.2-T2V-A14B",
            fallback="Wan-AI/Wan2.2-T2V-A14B",
        ),
        "lightx2v_root": _checkpoint_model_ref(
            "lightx2v--Wan2.2-Lightning",
            "Wan2.2-Lightning",
            fallback="lightx2v/Wan2.2-Lightning",
        ),
        "python_executable": _current_python_executable(),
        "runner_version": "v1",
    }


def _sana_checkpoint_ref(repo_dir: str, checkpoint_name: str, repo_id: str) -> str:
    hfd_repo_dir = repo_id.replace("/", "--")
    return _checkpoint_model_ref(
        f"{hfd_repo_dir}/checkpoints/{checkpoint_name}",
        f"{repo_dir}/checkpoints/{checkpoint_name}",
        f"{repo_dir}/{checkpoint_name}",
        fallback=f"hf://{repo_id}/checkpoints/{checkpoint_name}",
    )


def _sana_variant_default_size(variant: Any) -> tuple[int, int]:
    if variant.default_height is not None and variant.default_width is not None:
        return int(variant.default_height), int(variant.default_width)
    if variant.resolution == "720p":
        return 720, 1280
    if variant.resolution == "480p":
        return 480, 832
    if str(variant.resolution).endswith("px"):
        side = int(str(variant.resolution).removesuffix("px"))
        return side, side
    return 1024, 1024


def _sana_variant_checkpoint_ref(variant: Any) -> str:
    filename = Path(str(variant.model_path).split("?", 1)[0]).name
    repo_dir = str(variant.repo_id).rsplit("/", 1)[-1]
    return _sana_checkpoint_ref(repo_dir, filename, variant.repo_id)


@lru_cache(maxsize=1)
def _sana_image_aliases() -> tuple[str, ...]:
    from worldfoundry.base_models.diffusion_model.recipes.sana_variants import SANA_ALIASES

    image_ids = _sana_folded_studio_model_ids()
    aliases = list(sorted(image_ids))
    for alias, target in SANA_ALIASES.items():
        if alias != "sana" and target in image_ids:
            aliases.append(alias)
    return _coerce_aliases(*aliases)


def _sana_studio_parent_id(model_id: str) -> str | None:
    """Return ``sana`` when ``model_id`` is a folded image-family checkpoint."""

    from worldfoundry.base_models.diffusion_model.recipes.sana_variants import SANA_ALIASES

    key = _normalize_model_id(model_id)
    folded = _sana_folded_studio_model_ids()
    if key in folded:
        return "sana"
    target = SANA_ALIASES.get(key)
    if target in folded:
        return "sana"
    return None


def _sana_default_image_variant() -> Any:
    from worldfoundry.base_models.diffusion_model.recipes.sana_variants import SANA_VARIANTS

    return SANA_VARIANTS[SANA_DEFAULT_IMAGE_VARIANT_ID]


def _sana_image_default_ref() -> str:
    return _sana_variant_checkpoint_ref(_sana_default_image_variant())


def _sana_image_workspace_variants() -> tuple[Dict[str, Any], ...]:
    from worldfoundry.base_models.diffusion_model.recipes.sana_variants import SANA_VARIANTS

    rows: list[Dict[str, Any]] = []
    for variant in SANA_VARIANTS.values():
        if variant.runner not in _SANA_IMAGE_RUNNERS:
            continue
        height, width = _sana_variant_default_size(variant)
        rows.append(
            {
                "variant_id": variant.model_id,
                "label": variant.display_name,
                "status": "checkpoint-configured",
                "checkpoints": (
                    {
                        "role": "primary",
                        "uri": _sana_variant_checkpoint_ref(variant),
                        "required": True,
                        "status": "checkpoint-configured",
                    },
                ),
                "load_kwargs": {"model_id": variant.model_id},
                "call_kwargs": {
                    "height": height,
                    "width": width,
                    "num_inference_steps": variant.default_steps or 20,
                    "guidance_scale": variant.default_cfg_scale or 4.5,
                    "seed": 42,
                },
                "notes": (variant.notes,) if variant.notes else (),
            }
        )
    return tuple(rows)


def _cogvideox_variant_specs() -> dict[str, Dict[str, Any]]:
    return {
        "cogvideox_2b_t2v": {
            "generation_type": "t2v",
            "aliases": ("cogvideox-2b-t2v", "cogvideox-2b"),
            "label": "2B T2V",
            "default_prompt": "A futuristic city street at sunset with reflective glass towers, clean motion, and cinematic lighting.",
            "default_call_kwargs": {
                "fps": 8,
                "height": 480,
                "num_frames": 49,
                "num_inference_steps": 50,
                "guidance_scale": 6.0,
                "seed": 42,
                "width": 720,
            },
            "default_model_ref": lambda: _checkpoint_model_ref(
                "CogVideoX-2b",
                "THUDM--CogVideoX-2b",
                fallback="THUDM/CogVideoX-2b",
            ),
        },
        "cogvideox_5b_i2v": {
            "generation_type": "i2v",
            "aliases": ("cogvideox-5b-i2v",),
            "label": "5B I2V",
            "default_prompt": (
                "A young man holding a bright sparkler gently turns his head and smiles as "
                "sparks scatter through the night air, with a subtle cinematic camera dolly, "
                "stable facial details, and natural motion."
            ),
            "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
            "default_call_kwargs": {
                "fps": 8,
                "guidance_scale": 6.0,
                "height": 480,
                "num_frames": 49,
                "num_inference_steps": 50,
                "seed": 45,
                "width": 720,
            },
            "default_model_ref": lambda: _checkpoint_model_ref(
                "CogVideoX-5b-I2V",
                "THUDM--CogVideoX-5b-I2V",
                fallback="THUDM/CogVideoX-5b-I2V",
            ),
        },
        "cogvideox_5b_t2v": {
            "generation_type": "t2v",
            "aliases": ("cogvideox", "cogvideox-5b-t2v", "cogvideox-5b"),
            "label": "5B T2V",
            "default_prompt": "A futuristic city street at sunset with reflective glass towers, clean motion, and cinematic lighting.",
            "default_call_kwargs": {
                "fps": 8,
                "height": 480,
                "num_frames": 49,
                "num_inference_steps": 50,
                "guidance_scale": 6.0,
                "seed": 43,
                "width": 720,
            },
            "default_model_ref": lambda: _checkpoint_model_ref(
                "CogVideoX-5b",
                "THUDM--CogVideoX-5b",
                fallback="THUDM/CogVideoX-5b",
            ),
        },
    }


def _cogvideox_aliases() -> tuple[str, ...]:
    aliases = [COGVIDEOX_STUDIO_PARENT_ID, *sorted(_cogvideox_folded_studio_model_ids())]
    for spec in _cogvideox_variant_specs().values():
        aliases.extend(spec.get("aliases") or ())
    return _coerce_aliases(*aliases)


def _cogvideox_workspace_variants() -> tuple[Dict[str, Any], ...]:
    rows: list[Dict[str, Any]] = []
    for variant_id, spec in _cogvideox_variant_specs().items():
        model_ref = str(_resolve_override_value(spec.get("default_model_ref", "")) or "")
        rows.append(
            {
                "variant_id": variant_id,
                "label": spec.get("label") or variant_id,
                "status": "checkpoint-configured",
                "workload_type": spec.get("generation_type") or "t2v",
                "default_prompt": spec.get("default_prompt") or "",
                "default_input_path": spec.get("default_input_path") or "",
                "aliases": tuple(spec.get("aliases") or ()),
                "checkpoints": (
                    (
                        {
                            "role": "primary",
                            "uri": model_ref,
                            "required": True,
                            "status": "checkpoint-configured",
                        },
                    )
                    if model_ref
                    else ()
                ),
                "load_kwargs": {},
                "call_kwargs": dict(_resolve_override_value(spec.get("default_call_kwargs", {})) or {}),
            }
        )
    return tuple(rows)


def _sana_video_480p_default_ref() -> str:
    return _sana_checkpoint_ref(
        "SANA-Video_2B_480p",
        "SANA_Video_2B_480p.pth",
        "Efficient-Large-Model/SANA-Video_2B_480p",
    )


def _sana_video_720p_default_ref() -> str:
    return _sana_checkpoint_ref(
        "SANA-Video_2B_720p",
        "SANA_Video_2B_720p.pth",
        "Efficient-Large-Model/SANA-Video_2B_720p",
    )


def _sana_streaming_default_ref(*, bidirectional: bool) -> str:
    repo_dir = (
        "Efficient-Large-Model--SANA-Streaming_bidirectional"
        if bidirectional
        else "Efficient-Large-Model--SANA-Streaming"
    )
    repo_id = (
        "Efficient-Large-Model/SANA-Streaming_bidirectional"
        if bidirectional
        else "Efficient-Large-Model/SANA-Streaming"
    )
    return _checkpoint_model_ref(repo_dir, fallback=repo_id)


def _sana_streaming_default_load_kwargs(*, bidirectional: bool) -> Dict[str, Any]:
    dit_root = _sana_streaming_default_ref(bidirectional=bidirectional)
    gemma_root = _checkpoint_model_ref(
        "Efficient-Large-Model--gemma-2-2b-it",
        fallback="Efficient-Large-Model/gemma-2-2b-it",
    )
    codec_root = _checkpoint_model_ref(
        "Lightricks--LTX-2",
        fallback="Lightricks/LTX-2",
    )
    return {
        "model_id": (
            "sana-streaming-bidirectional-2b-720p"
            if bidirectional
            else "sana-streaming-2b-720p"
        ),
        "checkpoint_overrides": {
            "dit": dit_root,
            "text-encoder": gemma_root,
            "tokenizer": gemma_root,
            "codec": codec_root,
        },
    }


def _longsana_video_480p_default_ref() -> str:
    return _sana_checkpoint_ref(
        "SANA-Video_2B_480p_LongLive",
        "SANA_Video_2B_480p_LongLive.pth",
        "Efficient-Large-Model/SANA-Video_2B_480p_LongLive",
    )


def _sana_video_call_params() -> tuple[str, ...]:
    return (
        "prompt",
        "images",
        "video",
        "output_path",
        "fps",
        "return_dict",
        "seed",
        "cfg_scale",
        "guidance_scale",
        "step",
        "num_inference_steps",
        "num_frames",
        "frames",
        "height",
        "width",
        "max_sequence_length",
        "plan_only",
    )


def _longsana_video_call_params() -> tuple[str, ...]:
    return (
        "negative_prompt",
        *(param for param in _sana_video_call_params() if param != "cfg_scale"),
    )


def _sana_video_load_params() -> tuple[str, ...]:
    return (
        "model_path",
        "checkpoint_path",
        "config_path",
        "config",
        "python_executable",
        "python",
        "work_dir",
        "default_work_dir",
        "required_components",
        "device",
        "model_id",
        "variant",
        "profile_id",
        *DISPATCH_LOAD_PARAMS,
    )


def _zeroscope_default_ref() -> str:
    return _checkpoint_model_ref(
        "zeroscope_v2_576w",
        "cerspense--zeroscope_v2_576w",
        fallback="cerspense/zeroscope_v2_576w",
    )


def _animatediff_default_ref() -> str:
    return _checkpoint_model_ref(
        "animatediff/mm_sd_v15_v2.ckpt",
        "guoyww--animatediff/mm_sd_v15_v2.ckpt",
        fallback="guoyww/animatediff",
    )


def _animatediff_default_load_kwargs() -> Dict[str, Any]:
    return {
        "motion_module_path": _animatediff_default_ref(),
        "base_model_path": _checkpoint_model_ref(
            "stable-diffusion-v1-5",
            "stable-diffusion-v1-5--stable-diffusion-v1-5",
            "runwayml--stable-diffusion-v1-5",
            fallback="runwayml/stable-diffusion-v1-5",
        ),
        "dreambooth_model_path": _checkpoint_model_ref(
            "animatediff_t2i_backups/realisticVisionV60B1_v51VAE.safetensors",
            "guoyww--animatediff_t2i_backups/realisticVisionV60B1_v51VAE.safetensors",
        ),
    }


def _open_magvit2_default_ref() -> str:
    return _checkpoint_model_ref(
        "Open-MAGVIT2/AR_256_L.ckpt",
        "TencentARC--Open-MAGVIT2/AR_256_L.ckpt",
        fallback="TencentARC/Open-MAGVIT2",
    )


def _skyreels_v3_default_ref() -> str:
    return _checkpoint_model_ref(
        "SkyReels-V3-R2V-14B",
        "Skywork--SkyReels-V3-R2V-14B",
        fallback="Skywork/SkyReels-V3-R2V-14B",
    )


def _kairos_sensenova_default_ref() -> str:
    return _checkpoint_model_ref(
        "kairos-agi--kairos-sensenova-4B-480P-pretrained",
        "kairos-sensenova-4B-480P-pretrained",
        fallback="kairos-agi/kairos-sensenova-4B-480P-pretrained",
    )


def _kairos_sensenova_default_load_kwargs() -> Dict[str, Any]:
    model_root = Path(_kairos_sensenova_default_ref())
    return {
        "runtime_root": "",
        "models_root": str(model_root),
        "variant": "pretrained",
        "pretrained_dit": str(model_root / "kairos-common-4B-480P.safetensors"),
        "text_encoder_path": _checkpoint_model_ref(
            "Qwen--Qwen2.5-VL-7B-Instruct",
            fallback="Qwen/Qwen2.5-VL-7B-Instruct",
        ),
        "vae_path": _checkpoint_model_ref(
            "Wan-AI--Wan2.1-T2V-14B/Wan2.1_VAE.pth",
            fallback="Wan-AI/Wan2.1-T2V-14B",
        ),
    }


def _solaris_default_ref() -> str:
    # The GitHub URL identifies the vendored runtime source, not a loadable
    # checkpoint.  Keep model_ref local so readiness checks and the runtime
    # both address the official Orbax snapshot.
    return _checkpoint_model_ref(
        "nyu-visionx--solaris",
        "solaris",
        fallback=str(checkpoint_root_path("nyu-visionx--solaris")),
    )


def _solaris_default_load_kwargs() -> Dict[str, Any]:
    runtime_root = str(
        _project_root()
        / "worldfoundry"
        / "synthesis"
        / "visual_generation"
        / "solaris"
        / "solaris_runtime"
    )
    checkpoint_root = _solaris_default_ref()
    eval_data_root = _checkpoint_model_ref(
        "nyu-visionx--solaris-eval-datasets",
        "solaris/datasets",
        fallback=str(checkpoint_root_path("nyu-visionx--solaris-eval-datasets")),
    )
    return {
        "required_components": {
            "runtime_root": runtime_root,
            "pretrained_model_dir": checkpoint_root,
            "eval_data_dir": eval_data_root,
            "output_dir": str(_project_root() / "tmp" / "solaris_output"),
            "checkpoint_dir": checkpoint_root,
            "jax_cache_dir": str(_project_root() / "tmp" / "solaris_jax_cache"),
            "model_weights_path": str(Path(checkpoint_root) / "solaris.pt"),
            "python_executable": _worldfoundry_unified_python_executable(),
        }
    }


def _wow_default_ref() -> str:
    return _hf_checkpoint_model_ref(
        "X-Humanoid--WoW-1-Wan-14B-600k",
        "X-Humanoid/WoW-1-Wan-14B-600k",
    )


def _minwm_default_ref() -> str:
    return _hf_checkpoint_model_ref("MIN-Lab--minWM", "MIN-Lab/minWM")


def _minwm_hy_default_load_kwargs() -> Dict[str, Any]:
    return {
        "base_model_path": _hf_checkpoint_model_ref(
            "tencent--HunyuanVideo-1.5",
            "tencent/HunyuanVideo-1.5",
        )
    }


def _minwm_wan_default_load_kwargs() -> Dict[str, Any]:
    return {
        "base_model_path": _hf_checkpoint_model_ref(
            "Wan-AI--Wan2.1-T2V-1.3B",
            "Wan-AI/Wan2.1-T2V-1.3B",
        )
    }


def _gr00t_default_ref() -> str:
    return _checkpoint_model_ref(
        "GR00T-N1.7-LIBERO/libero_10",
        "GR00T-N1.7-LIBERO/libero_goal",
        "GR00T-N1.7-LIBERO",
        fallback="nvidia/GR00T-N1.7-LIBERO",
    )


def _openpi_default_ref() -> str:
    return _checkpoint_model_ref(
        "openpi-assets/checkpoints/pi05_libero",
        fallback="gs://openpi-assets/checkpoints/pi05_libero",
    )


def _openpi_default_call_kwargs() -> Dict[str, Any]:
    return {
        "checkpoint_dir": _openpi_default_ref(),
        "config_name": "pi05_libero",
        "seed": 0,
    }


def _lapa_default_ref() -> str:
    return _checkpoint_model_ref(
        "latent-action-pretraining--LAPA-7B-openx",
        "LAPA-7B-openx",
        "hfd/latent-action-pretraining--LAPA-7B-openx",
        fallback="latent-action-pretraining/LAPA-7B-openx",
    )


def _lapa_default_call_kwargs() -> Dict[str, Any]:
    return {
        "checkpoint_dir": _lapa_default_ref(),
        "dtype": "bf16",
        "image_size": 256,
        "mesh_dim": "1,1,1,1",
        "seed": 1234,
        "tokens_per_delta": 4,
    }


def _octo_default_ref() -> str:
    return _checkpoint_model_ref(
        "octo-small-1.5",
        "hfd/rail-berkeley--octo-small-1.5",
        fallback="rail-berkeley/octo-small-1.5",
    )


def _octo_default_call_kwargs() -> Dict[str, Any]:
    return {
        "checkpoint_dir": _octo_default_ref(),
        "variant": "small",
        "dataset_key": "bridge_dataset",
        "image_key": "image_primary",
        "image_size": [256, 256],
        "jax_platform": "cpu",
        "seed": 0,
    }


def _diffusion_policy_default_call_kwargs() -> Dict[str, Any]:
    # Official low-dimensional PushT checkpoint: n_obs_steps=2, obs_dim=20.
    return {
        "device": "cuda",
        "observation": {
            "obs": [
                [
                    256.0,
                    256.0,
                    218.0,
                    224.0,
                    254.0,
                    212.0,
                    292.0,
                    224.0,
                    306.0,
                    258.0,
                    292.0,
                    292.0,
                    254.0,
                    304.0,
                    218.0,
                    292.0,
                    204.0,
                    258.0,
                    230.0,
                    258.0,
                ],
                [
                    262.0,
                    258.0,
                    224.0,
                    226.0,
                    260.0,
                    214.0,
                    298.0,
                    226.0,
                    312.0,
                    260.0,
                    298.0,
                    294.0,
                    260.0,
                    306.0,
                    224.0,
                    294.0,
                    210.0,
                    260.0,
                    236.0,
                    260.0,
                ],
            ]
        },
        "action_space": {"type": "low_dim_pusht", "action_dim": 2, "n_action_steps": 8},
    }


def _molmoact2_default_call_kwargs() -> Dict[str, Any]:
    demo_dir = _data_path("test_cases", "test_vla_case1", "droid")
    state = [0.0] * 8
    operator_kwargs = {
        "camera_keys": ["external_cam", "external_cam_2", "wrist_cam"],
        "embodiment": "droid",
        "norm_tag": "franka_droid",
        "state": state,
        "external_cam": str(demo_dir / "exterior_image_1_left.png"),
        "external_cam_2": str(demo_dir / "exterior_image_1_left.png"),
        "wrist_cam": str(demo_dir / "wrist_image_left.png"),
    }
    return {
        **operator_kwargs,
        "operator_kwargs": operator_kwargs,
        "num_steps": 10,
    }


def _molmoact2_yam_call_kwargs() -> Dict[str, Any]:
    demo_dir = _data_path("test_cases", "test_vla_case1")
    state = [0.0] * 14
    operator_kwargs = {
        "camera_keys": ["top_cam", "left_cam", "right_cam"],
        "embodiment": "yam",
        "norm_tag": "yam_dual_molmoact2",
        "state": state,
        "top_cam": str(demo_dir / "droid" / "exterior_image_1_left.png"),
        "left_cam": str(demo_dir / "aloha" / "observation_images_cam_left_wrist.png"),
        "right_cam": str(demo_dir / "aloha" / "observation_images_cam_right_wrist.png"),
    }
    return {
        **operator_kwargs,
        "variant": "yam",
        "embodiment": "yam",
        "operator_kwargs": operator_kwargs,
        "num_steps": 10,
    }


def _molmoact2_so100_call_kwargs() -> Dict[str, Any]:
    checkpoint_dir = checkpoint_root_path("hfd", "allenai--MolmoAct2-SO100_101")
    state = [
        -0.52734375,
        189.140625,
        181.40625,
        60.64453125,
        -3.603515625,
        1.0971786975860596,
    ]
    operator_kwargs = {
        "camera_keys": ["top_cam", "side_cam"],
        "embodiment": "so100",
        "norm_tag": "so100_so101_molmoact2",
        "state": state,
        "top_cam": str(checkpoint_dir / "assets" / "sample_realsense_top_rgb.png"),
        "side_cam": str(checkpoint_dir / "assets" / "sample_realsense_side_rgb.png"),
    }
    return {
        **operator_kwargs,
        "variant": "so100",
        "embodiment": "so100",
        "operator_kwargs": operator_kwargs,
        "num_steps": 10,
    }


def _molmoact2_libero_call_kwargs() -> Dict[str, Any]:
    checkpoint_dir = checkpoint_root_path("hfd", "allenai--MolmoAct2-LIBERO")
    state = [
        -0.05338004603981972,
        0.007029631175100803,
        0.6783280968666077,
        3.1407692432403564,
        0.0017593271331861615,
        -0.08994418382644653,
        0.03878866136074066,
        -0.03878721222281456,
    ]
    operator_kwargs = {
        "camera_keys": ["agentview_cam", "wrist_cam"],
        "embodiment": "libero",
        "norm_tag": "libero",
        "state": state,
        "agentview_cam": str(checkpoint_dir / "assets" / "sample_agentview_rgb.png"),
        "wrist_cam": str(checkpoint_dir / "assets" / "sample_wrist_rgb.png"),
        "enable_depth_reasoning": False,
    }
    return {
        **operator_kwargs,
        "variant": "libero",
        "embodiment": "libero",
        "operator_kwargs": operator_kwargs,
        "num_steps": 10,
        "enable_depth_reasoning": False,
    }


def _molmoact2_think_libero_call_kwargs() -> Dict[str, Any]:
    call_kwargs = _molmoact2_libero_call_kwargs()
    checkpoint_dir = checkpoint_root_path("hfd", "allenai--MolmoAct2-Think-LIBERO")
    operator_kwargs = dict(call_kwargs["operator_kwargs"])
    operator_kwargs.update(
        {
            "embodiment": "think_libero",
            "agentview_cam": str(checkpoint_dir / "assets" / "sample_agentview_rgb.png"),
            "wrist_cam": str(checkpoint_dir / "assets" / "sample_wrist_rgb.png"),
            "enable_depth_reasoning": True,
            "enable_adaptive_depth": True,
            "depth_cache": None,
        }
    )
    call_kwargs.update(
        {
            **operator_kwargs,
            "variant": "think-libero",
            "embodiment": "think_libero",
            "operator_kwargs": operator_kwargs,
            "enable_depth_reasoning": True,
            "enable_adaptive_depth": True,
            "depth_cache": None,
        }
    )
    return call_kwargs


def _depth_anything_v3_default_ref() -> str:
    repo_id = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    try:
        return str(
            resolve_local_hf_model_path(
                repo_id,
                required_files=("config.json", "model.safetensors"),
            )
        )
    except FileNotFoundError:
        return repo_id


def _inspatio_world_default_ref() -> str:
    repo_id = "inspatio/world"
    try:
        return str(
            resolve_local_hf_model_path(
                repo_id,
                required_files=("InSpatio-World-1.3B.safetensors",),
            )
        )
    except FileNotFoundError:
        return repo_id


def _inspatio_world_default_load_kwargs() -> Dict[str, Any]:
    return {
        "wan_model_path": _wan_2p1_t2v_default_ref(),
        "da3_model_path": _depth_anything_v3_default_ref(),
        "florence_model_path": _checkpoint_model_ref(
            "Florence-2-large",
            fallback="microsoft/Florence-2-large",
        ),
        "default_traj_txt_path": str(_data_path("models", "runtime", "configs", "inspatio_world", "traj", "x_y_circle_cycle.txt")),
    }


def _worldcam_default_ref() -> str:
    return _prefer_existing_model_ref(
        *_cache_candidates("worldcam--worldcam", "worldcam"),
        "worldcam/worldcam",
    )


def _worldcam_default_load_kwargs() -> Dict[str, Any]:
    return {
        "required_components": {
            "wan_model_path": _wan_2p1_t2v_default_ref(),
        },
    }


def _worldcam_default_call_kwargs() -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "conditioning_frames": 65,
        "num_ar_steps": 50,
        "cfg_scale": 4,
        "seed": 0,
        "num_inference_steps": 50,
        "long_term_memory_start_step": 30,
        "long_term_memory_num_clips": 4,
        "long_term_memory_ref_indices": [48, 52, 56, 60],
        "attention_sink_inference": False,
        "trim_conditioning": False,
        "height": 480,
        "width": 832,
        "fps": 30,
    }
    demo_dir = _data_path("test_cases", "worldcam")
    video_path = demo_dir / "0.mp4"
    intrinsics_path = demo_dir / "0_intrinsics_palindrome.npy"
    extrinsics_path = demo_dir / "0_poses_palindrome.npy"
    if video_path.is_file() and intrinsics_path.is_file() and extrinsics_path.is_file():
        kwargs.update(
            {
                "video_path": str(video_path),
                "intrinsics_path": str(intrinsics_path),
                "extrinsics_path": str(extrinsics_path),
            }
        )
    return kwargs


def _neoverse_default_ref() -> str:
    return _checkpoint_model_ref(
        "Yuppie1204--NeoVerse",
        "NeoVerse",
        fallback="Yuppie1204/NeoVerse",
    )


NEOVERSE_OFFICIAL_PROMPT = "A two-arm robot assembles parts in front of a table."


def _neoverse_default_call_kwargs() -> Dict[str, Any]:
    return {
        "predefined_trajectory": "tilt_up",
        "num_frames": 81,
        "trajectory_mode": "relative",
        "angle": 15,
        "distance": 0,
        "orbit_radius": 0,
        "zoom_ratio": 1.0,
        "alpha_threshold": 1.0,
        "seed": 42,
        "use_first_frame": True,
        "static_scene": False,
        "num_inference_steps": 4,
        "cfg_scale": 1.0,
    }


def _neoverse_default_load_kwargs() -> Dict[str, Any]:
    return {
        "height": 336,
        "width": 560,
        "num_inference_steps": 4,
        "cfg_scale": 1.0,
        "disable_lora": False,
        "enable_vram_management": True,
    }


def _lingbot_world_fast_load_kwargs() -> Dict[str, Any]:
    fast_ref = _prefer_existing_model_ref(
        *_cache_candidates(
            "robbyant--lingbot-world-fast",
            "lingbot-world-fast",
        ),
        "robbyant/lingbot-world-fast",
    )
    return {
        "runtime_variant": "fast",
        "fast_model_path": fast_ref,
    }


def _lingbot_world_default_load_kwargs() -> Dict[str, Any]:
    # FSDP is the trigger for adaptive 8/4-rank Workspace launch.  The
    # dispatcher injects ulysses_size=WORLD_SIZE after selecting visible GPUs.
    return {
        "t5_fsdp": True,
        "dit_fsdp": True,
        "t5_cpu": False,
    }


def lingbot_world_fast_load_kwargs() -> Dict[str, Any]:
    return _lingbot_world_fast_load_kwargs()


def _infinite_world_default_ref() -> str:
    candidates = _cache_candidates("MeiGen-AI--Infinite-World", "Infinite-World")
    for value in candidates:
        candidate = Path(value).expanduser()
        if (candidate / "checkpoints" / "infinite_world_model.ckpt").is_file():
            return str(candidate.resolve())
        if candidate.name == "checkpoints" and (candidate / "infinite_world_model.ckpt").is_file():
            return str(candidate.parent.resolve())
    return "MeiGen-AI/Infinite-World"


def _stream_vggt_default_ref() -> str:
    return _prefer_existing_model_ref(
        *_cache_candidates("lch01--StreamVGGT", "StreamVGGT"),
        str(checkpoint_root_path("StreamVGGT")),
        "lch01/StreamVGGT",
    )


def _loger_default_ref() -> str:
    return _prefer_existing_model_ref(
        *_cache_candidates("Junyi42--LoGeR", "LoGeR"),
        str(checkpoint_root_path("LoGeR")),
        "Junyi42/LoGeR",
    )


def _flash_world_default_ref() -> str:
    return _prefer_existing_model_ref(
        *_cache_candidates("imlixinyang--FlashWorld", "FlashWorld"),
        "imlixinyang/FlashWorld",
    )


def _flash_world_default_load_kwargs() -> Dict[str, Any]:
    return {
        "required_components": {
            "wan_model_path": _prefer_existing_model_ref(
                *_cache_candidates("Wan-AI--Wan2.2-TI2V-5B-Diffusers", "Wan2.2-TI2V-5B-Diffusers"),
                "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            )
        }
    }


def _fantasy_world_camera_fixture() -> str:
    return str(_data_path("test_cases", "fantasyworld", "camera_forward.json"))


def _fantasy_world_wan21_default_ref() -> str:
    return _checkpoint_model_ref(
        "FantasyWorld-Wan2.1-I2V-14B-480P",
        "acvlab--FantasyWorld-Wan2.1-I2V-14B-480P",
        fallback="acvlab/FantasyWorld-Wan2.1-I2V-14B-480P",
    )


def _fantasy_world_wan22_default_ref() -> str:
    return _checkpoint_model_ref(
        "FantasyWorld-Wan2.2-Fun-A14B-Control-Camera",
        "acvlab--FantasyWorld-Wan2.2-Fun-A14B-Control-Camera",
        fallback="acvlab/FantasyWorld-Wan2.2-Fun-A14B-Control-Camera",
    )


def _fantasy_world_wan21_load_kwargs() -> Dict[str, Any]:
    return {
        "required_components": {
            "wan_model_path": _checkpoint_model_ref(
                "Wan2.1-I2V-14B-480P",
                "Wan-AI--Wan2.1-I2V-14B-480P",
                fallback="Wan-AI/Wan2.1-I2V-14B-480P",
            ),
            "moge_pretrained": _checkpoint_model_ref(
                "moge-2-vitl-normal",
                "Ruicheng--moge-2-vitl-normal",
                fallback="Ruicheng/moge-2-vitl-normal",
            ),
        },
        "sample_steps": 50,
        "frames": 81,
        "fps": 16,
        "height": 336,
        "width": 592,
    }


def _fantasy_world_wan22_load_kwargs() -> Dict[str, Any]:
    return {
        "required_components": {
            "wan_model_path": _checkpoint_model_ref(
                "Wan2.2-Fun-A14B-Control-Camera",
                "alibaba-pai--Wan2.2-Fun-A14B-Control-Camera",
                fallback="alibaba-pai/Wan2.2-Fun-A14B-Control-Camera",
            ),
            "lora_path": _checkpoint_model_ref(
                "Wan2.2-Fun-Reward-LoRAs",
                "alibaba-pai--Wan2.2-Fun-Reward-LoRAs",
                "Wan-AI--Wan2.2-Fun-Reward-LoRAs",
                fallback="alibaba-pai/Wan2.2-Fun-Reward-LoRAs",
            ),
            "moge_pretrained": _checkpoint_model_ref(
                "moge-2-vitl-normal",
                "Ruicheng--moge-2-vitl-normal",
                fallback="Ruicheng/moge-2-vitl-normal",
            ),
        },
        "sample_steps": 50,
        "frames": 81,
        "fps": 16,
        "height": 480,
        "width": 832,
    }


def _fantasy_world_default_load_kwargs() -> Dict[str, Any]:
    return _fantasy_world_wan21_load_kwargs()


def _fantasy_world_default_call_kwargs(conf_threshold: float = 1.5) -> Dict[str, Any]:
    return {
        "camera_json_path": _fantasy_world_camera_fixture(),
        "fps": 16,
        "using_scale": True,
        "conf_threshold": conf_threshold,
        "stride": 4,
        "return_dict": True,
    }


def _vggt_default_ref() -> str:
    return _prefer_existing_model_ref(
        *_cache_candidates("facebook--VGGT-1B", "VGGT-1B"),
        "facebook/VGGT-1B",
    )


def _lagernvs_default_ref() -> str:
    return _prefer_existing_model_ref(
        str(checkpoint_root_path("lagernvs_general_512")),
        *_cache_candidates("facebook--lagernvs_general_512", "facebook--lagernvs-general-512"),
        "facebook/lagernvs_general_512",
    )


def _dvlt_default_ref() -> str:
    return _prefer_existing_model_ref(
        str(checkpoint_root_path("dvlt")),
        *_cache_candidates("nvidia--dvlt", "dvlt"),
        "nvidia/dvlt",
    )


def _pixelsplat_default_ref() -> str:
    return _prefer_existing_model_ref(
        str(checkpoint_root_path("pixelSplat", "re10k.ckpt")),
        str(checkpoint_root_path("hfd", "3DGeneration--pixelSplat", "re10k.ckpt")),
        *_cache_candidates("dylanebert--pixelSplat", "3DGeneration--pixelSplat", "pixelSplat"),
        "dylanebert/pixelSplat",
    )


def _splatt3r_default_ref() -> str:
    return _prefer_existing_model_ref(
        str(checkpoint_root_path("splatt3r_v1.0", "epoch=19-step=1200.ckpt")),
        str(checkpoint_root_path("hfd", "3DGeneration--splatt3r_v1.0", "epoch=19-step=1200.ckpt")),
        *_cache_candidates("brandonsmart--splatt3r_v1.0", "3DGeneration--splatt3r_v1.0", "splatt3r_v1.0"),
        "brandonsmart/splatt3r_v1.0",
    )


def _lingbot_map_default_ref() -> str:
    return _prefer_existing_model_ref(
        str(checkpoint_root_path("lingbot-map", "lingbot-map-long.pt")),
        str(checkpoint_root_path("lingbot-map", "lingbot-map.pt")),
        str(checkpoint_root_path("hfd", "custom--lingbot-map", "lingbot-map-long.pt")),
        str(checkpoint_root_path("hfd", "custom--lingbot-map", "lingbot-map.pt")),
        *_cache_candidates("robbyant--lingbot-map", "lingbot-map"),
        "robbyant/lingbot-map",
    )


_IN_TREE_REPO_PARTS: dict[str, tuple[str, ...]] = {
    "lagernvs": ("base_models", "three_dimensions", "general_3d", "lagernvs", "lagernvs_runtime"),
    "monst3r": ("base_models", "three_dimensions", "general_3d", "monst3r"),
    "MVDiffusion": ("base_models", "three_dimensions", "general_3d", "mvdiffusion", "mvdiffusion_runtime"),
    "stable-virtual-camera": (
        "base_models",
        "three_dimensions",
        "general_3d",
        "stable_virtual_camera",
        "stable_virtual_camera_runtime",
    ),
    "WonderJourney": ("synthesis", "visual_generation", "wonderjourney", "wonderjourney_runtime"),
    "WonderWorld": ("synthesis", "visual_generation", "wonderworld", "wonderworld_runtime"),
    "WorldGen": ("synthesis", "visual_generation", "worldgen", "worldgen_runtime"),
}


def _in_tree_repo_ref(repo_name: str) -> str:
    parts = _IN_TREE_REPO_PARTS.get(repo_name)
    if parts is None:
        known = ", ".join(sorted(_IN_TREE_REPO_PARTS))
        raise ValueError(f"Unknown in-tree repo {repo_name!r}; known repos: {known}")
    return str(Path(__file__).resolve().parents[1] / Path(*parts))


def _three_d_four_d_runtime_call_params(*model_params: str) -> tuple[str, ...]:
    base_params = (
        "output_path",
        "python_executable",
        "source_root",
        "runtime_root",
        "repo_root",
        "return_dict",
        "gpu",
        "timeout_seconds",
    )
    deduped: list[str] = []
    for name in (*model_params, *base_params):
        if name and name not in deduped:
            deduped.append(name)
    return tuple(deduped)


def _three_d_four_d_runtime_load_params() -> tuple[str, ...]:
    return (
        "model_path",
        "required_components",
        "device",
        "model_id",
        "repo_root",
        "source_root",
        "runtime_root",
    )


def _three_d_four_d_runtime_call_kwargs(**kwargs: Any) -> dict[str, Any]:
    call_kwargs = {
        "python_executable": _worldfoundry_unified_python_executable(),
        "return_dict": True,
    }
    call_kwargs.update(kwargs)
    return call_kwargs


def _monst3r_default_call_kwargs() -> dict[str, Any]:
    checkpoint_dir = checkpoint_root_path("MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt")
    legacy_checkpoint = checkpoint_dir / "MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"
    if legacy_checkpoint.is_file():
        weights = legacy_checkpoint
    elif (checkpoint_dir / "config.json").is_file() and (checkpoint_dir / "model.safetensors").is_file():
        weights = checkpoint_dir
    else:
        weights = legacy_checkpoint
    return _three_d_four_d_runtime_call_kwargs(
        weights=str(weights),
        seq_name="worldfoundry",
        niter=300,
        flow_loss_weight=0.0,
        skip_pair_dynamic_mask=True,
        silent=True,
    )


def _geometry_prior_call_params(*model_params: str) -> tuple[str, ...]:
    base_params = (
        "images",
        "image_path",
        "input_path",
        "video",
        "video_path",
        "output_path",
        "output_dir",
        "execute",
        "return_dict",
        "max_points",
        "focal_length",
    )
    deduped: list[str] = []
    for name in (*model_params, *base_params):
        if name and name not in deduped:
            deduped.append(name)
    return tuple(deduped)


def _geometry_prior_load_params() -> tuple[str, ...]:
    return (
        "model_path",
        "required_components",
        "device",
        "model_id",
        "profile_id",
        "runtime_profile",
        "base_model_target",
    )


def _geometry_prior_default_call_kwargs(**kwargs: Any) -> dict[str, Any]:
    call_kwargs = {"execute": True, "return_dict": True, "max_points": 5000}
    call_kwargs.update(kwargs)
    return call_kwargs


def _geometry_prior_image_fixture() -> str:
    return str(_data_path("test_cases", "images", "000.png"))


def _geometry_prior_video_fixture() -> str:
    return str(_data_path("test_cases", "neoverse", "videos", "robot.mp4"))


def _cut3r_default_ref() -> str:
    return _checkpoint_model_ref(
        "cut3r",
        "CUT3R",
        "liguang0115--cut3r",
        "hfd/liguang0115--cut3r",
        fallback="liguang0115/cut3r",
    )


def _show_o_default_ref() -> str:
    return _checkpoint_model_ref(
        "showlab--show-o",
        "show-o",
        fallback="showlab/show-o",
    )


def _show_o_default_load_kwargs() -> dict[str, Any]:
    return {
        "vq_model_path": _checkpoint_model_ref(
            "showlab--magvitv2",
            "magvitv2",
            fallback="showlab/magvitv2",
        ),
        "llm_model_path": _checkpoint_model_ref(
            "microsoft--phi-1_5",
            "phi-1_5",
            fallback="microsoft/phi-1_5",
        ),
        "resolution": 256,
    }


def _hunyuan_world_voyager_default_ref() -> str:
    return _checkpoint_model_ref(
        "HunyuanWorld-Voyager",
        "tencent--HunyuanWorld-Voyager",
        "hfd/tencent--HunyuanWorld-Voyager",
        fallback="tencent/HunyuanWorld-Voyager",
    )


def _hunyuan_gamecraft_default_ref() -> str:
    return _checkpoint_model_ref(
        "Hunyuan-GameCraft-1.0",
        "tencent--Hunyuan-GameCraft-1.0",
        "hfd/tencent--Hunyuan-GameCraft-1.0",
        fallback="tencent/Hunyuan-GameCraft-1.0",
    )


def _worldfm_default_ref() -> str:
    return _prefer_existing_model_ref(
        *_cache_candidates("worldfm", "inspatio--worldfm"),
        "inspatio/worldfm",
    )


def _worldfm_default_load_kwargs() -> Dict[str, Any]:
    return {
        "required_components": {
            "moge_pretrained": _prefer_existing_model_ref(
                *_cache_candidates("moge-2-vitl-normal", "Ruicheng--moge-2-vitl-normal", "Ruicheng--moge-2-vitl"),
            ),
        }
    }


def _worldfm_demo_meta_path() -> str:
    return str(_data_path("test_cases", "worldfm", "meta.json"))


def _worldfm_demo_panorama_path() -> str:
    return str(_data_path("test_cases", "worldfm", "mario.png"))


def _hunyuan_world_voyager_case1_dir() -> Path:
    return _data_path("test_cases", "hunyuan_world_voyager", "case1")


def _ac3d_dataset_ready(root: Path) -> bool:
    return (
        (root / "annotations" / "test.json").is_file()
        and (root / "pose_files").is_dir()
        and (root / "video_clips").is_dir()
    )


def _ac3d_default_video_root_dir() -> str:
    candidates = [
        os.environ.get("WORLDFOUNDRY_AC3D_VIDEO_ROOT_DIR", ""),
        str(_data_path("test_cases", "ac3d", "RealEstate10K")),
        str(_project_root().parent / "datasets" / "RealEstate10K"),
        str(_project_root().parent / "data" / "RealEstate10K"),
        str(_project_root().parent / "ckpt" / "ac3d" / "RealEstate10K"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if _ac3d_dataset_ready(path):
            return str(path)
    return ""


def _ac3d_default_call_kwargs() -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "annotation_json": "annotations/test.json",
        "num_frames": 49,
        "stride_min": 2,
        "stride_max": 2,
        "height": 480,
        "width": 720,
        "start_camera_idx": 0,
        "end_camera_idx": 1,
    }
    video_root_dir = _ac3d_default_video_root_dir()
    if video_root_dir:
        kwargs["video_root_dir"] = video_root_dir
    return kwargs


def _gen3c_default_ref() -> str:
    return _checkpoint_model_ref(
        "nvidia--GEN3C-Cosmos-7B",
        "GEN3C-Cosmos-7B",
        fallback="nvidia/GEN3C-Cosmos-7B",
    )


def _gen3c_default_load_kwargs() -> Dict[str, Any]:
    return {
        "required_components": {
            "transformer_model_path": _gen3c_default_ref(),
            "tokenizer_model_path": _checkpoint_model_ref(
                "nvidia--Cosmos-Tokenize1-CV8x8x8-720p",
                "Cosmos-Tokenize1-CV8x8x8-720p",
                fallback="nvidia/Cosmos-Tokenize1-CV8x8x8-720p",
            ),
            "text_encoder_model_path": _checkpoint_model_ref(
                "google-t5--t5-11b",
                "google-t5/t5-11b",
                "t5-11b",
                fallback="google-t5/t5-11b",
            ),
            "moge_pretrained": _checkpoint_model_ref(
                "Ruicheng--moge-vitl",
                "Ruicheng/moge-vitl",
                "moge-vitl",
                fallback="Ruicheng/moge-vitl",
            ),
        }
    }


def _gen3c_default_call_kwargs() -> Dict[str, Any]:
    """Native GEN3C quality defaults; legacy official-subprocess flags are omitted."""

    from worldfoundry.pipelines.gen3c.constants import (
        DEFAULT_GEN3C_NEGATIVE_PROMPT,
    )

    return {
        "trajectory": "left",
        "camera_rotation": "center_facing",
        "movement_distance": 0.3,
        "negative_prompt": DEFAULT_GEN3C_NEGATIVE_PROMPT,
        "guidance": 1.0,
        "num_steps": 35,
        "num_video_frames": 121,
        "fps": 24,
        "height": 704,
        "width": 1280,
        "seed": 1,
    }


def _lyra1_default_call_kwargs() -> Dict[str, Any]:
    return {
        "mode": "static",
        "trajectory": "zoom_in",
        "num_video_frames": 121,
        "fps": 24,
        "height": 704,
        "width": 1280,
        "seed": 1,
        "num_steps": 35,
        "guidance": 1.0,
        "num_gpus": 1,
        "movement_distance": 0.3,
        "camera_rotation": "center_facing",
        "multi_trajectory": True,
        "total_movement_distance_factor": 1.0,
        "foreground_masking": True,
        "filter_points_threshold": 0.05,
        "disable_prompt_encoder": True,
        "offload_diffusion_transformer": True,
        "offload_tokenizer": True,
        "offload_text_encoder_model": True,
        "offload_prompt_upsampler": True,
        "offload_guardrail_models": True,
        "disable_guardrail": True,
        "execute": True,
        "show_progress": True,
    }


def _lyra2_default_call_kwargs() -> Dict[str, Any]:
    return {
        "fps": 16,
        "resolution": (480, 832),
        "seed": 1,
        "reconstruct_3d": False,
        "return_dict": True,
        "show_progress": True,
        "execute": True,
    }


def _lyra2_default_load_kwargs() -> Dict[str, Any]:
    required_components: Dict[str, Any] = {"load_runtime": True}
    checkpoint_root = None
    for candidate in (
        hfd_root_path("Lyra-2.0"),
        hfd_root_path("nvidia--Lyra-2.0"),
        checkpoint_root_path("Lyra-2.0"),
        checkpoint_root_path("Lyra-2"),
    ):
        for root in (candidate, candidate / "Lyra-2.0", candidate / "Lyra-2"):
            if (
                (root / "checkpoints" / "model").is_dir()
                and (root / "checkpoints" / "text_encoder" / "negative_prompt.pt").is_file()
                and (root / "checkpoints" / "recon" / "model.pt").exists()
            ):
                checkpoint_root = root
                break
        if checkpoint_root is not None:
            break
    if checkpoint_root is not None:
        required_components.update(
            {
                "checkpoint_dir": str(checkpoint_root / "checkpoints" / "model"),
                "negative_prompt_path": str(
                    checkpoint_root / "checkpoints" / "text_encoder" / "negative_prompt.pt"
                ),
                "da3_model_path_custom": str(checkpoint_root / "checkpoints" / "recon" / "model.pt"),
            }
        )
    return {"required_components": required_components}


def _alayaworld_default_load_kwargs() -> Dict[str, Any]:
    return {
        "required_components": {
            "checkpoint_path": _checkpoint_model_ref(
                "AlayaWorld/merged_infer.safetensors",
                "AlayaLab--AlayaWorld/merged_infer.safetensors",
                fallback=str(checkpoint_root_path("AlayaWorld", "merged_infer.safetensors")),
            ),
            "gemma_root": _checkpoint_model_ref(
                "gemma-3-12b-it-qat-q4_0-unquantized",
                "google--gemma-3-12b-it-qat-q4_0-unquantized",
                fallback=str(checkpoint_root_path("gemma-3-12b-it-qat-q4_0-unquantized")),
            ),
            "da3_path": _checkpoint_model_ref(
                "DA3NESTED-GIANT-LARGE-1.1",
                "depth-anything--DA3NESTED-GIANT-LARGE-1.1",
                "Lyra-2.0/checkpoints/recon/model.pt",
                fallback=str(checkpoint_root_path("DA3NESTED-GIANT-LARGE-1.1")),
            ),
        }
    }


def _hunyuan_worldplay_default_ref() -> str:
    return _checkpoint_model_ref(
        "HY-WorldPlay",
        "tencent--HY-WorldPlay",
        fallback="tencent/HY-WorldPlay",
    )


def _hunyuan_worldplay_default_call_kwargs() -> Dict[str, Any]:
    return {
        "num_frames": 125,
        "aspect_ratio": "16:9",
        "num_inference_steps": 4,
        "seed": 1,
        "fps": 24,
        "output_type": "pt",
        "prompt_rewrite": False,
        "enable_sr": False,
        "return_pre_sr_video": False,
        "few_step": True,
        "chunk_latent_frames": 4,
        "model_type": "ar",
        "transformer_resident_ar_rollout": True,
        "user_width": 832,
        "user_height": 480,
    }


def _forcing_unified_python_executable() -> str:
    candidate = _project_root().parent / "conda" / "envs" / "worldfoundry-unified-cu128" / "bin" / "python"
    return str(candidate) if candidate.is_file() else _worldfoundry_unified_python_executable()


def _forcing_in_tree_runtime_root(model_id: str) -> Path:
    runtime_name = "causal_forcing_runtime" if model_id == "causal-forcing" else "self_forcing_runtime"
    return (
        _package_root()
        / "synthesis"
        / "visual_generation"
        / "forcing"
        / runtime_name
    )


def _self_forcing_default_ref() -> str:
    return _checkpoint_model_ref(
        "gdhe17--Self-Forcing/checkpoints/self_forcing_dmd.pt",
        "Self-Forcing/checkpoints/self_forcing_dmd.pt",
        "LongSANA_2B_480p_self_forcing/checkpoints/LongSANA_2B_480p_self_forcing.pt",
        fallback="gdhe17/Self-Forcing",
    )


def _causal_forcing_default_ref() -> str:
    return _checkpoint_model_ref(
        "zhuhz22--Causal-Forcing/chunkwise/causal_forcing.pt",
        "Causal-Forcing/chunkwise/causal_forcing.pt",
        fallback="zhuhz22/Causal-Forcing",
    )


def _forcing_wan_default_ref() -> str:
    return _checkpoint_model_ref(
        "Wan-AI--Wan2.1-T2V-1.3B",
        "Wan2.1-T2V-1.3B",
        fallback="Wan-AI/Wan2.1-T2V-1.3B",
    )


def _self_forcing_default_load_kwargs() -> Dict[str, Any]:
    runtime_root = _forcing_in_tree_runtime_root("self-forcing")
    config_path = _data_path("models", "runtime", "configs", "self_forcing", "self_forcing_dmd.yaml")
    return {
        "required_components": {
            "runtime_root": str(runtime_root),
            "config_path": str(config_path),
            "checkpoint_path": _self_forcing_default_ref(),
            "wan_models_root": _forcing_wan_default_ref(),
            "python_executable": _forcing_unified_python_executable(),
            "use_ema": True,
            "save_with_index": True,
        }
    }


def _causal_forcing_default_load_kwargs() -> Dict[str, Any]:
    runtime_root = _forcing_in_tree_runtime_root("causal-forcing")
    config_path = _data_path("models", "runtime", "configs", "causal_forcing", "causal_forcing_dmd_chunkwise.yaml")
    return {
        "required_components": {
            "runtime_root": str(runtime_root),
            "config_path": str(config_path),
            "checkpoint_path": _causal_forcing_default_ref(),
            "wan_models_root": _forcing_wan_default_ref(),
            "python_executable": _forcing_unified_python_executable(),
        }
    }


def _self_forcing_default_call_kwargs() -> Dict[str, Any]:
    return {
        "num_output_frames": 21,
        "seed": 0,
        "num_samples": 1,
        "use_ema": True,
        "save_with_index": True,
        "return_dict": True,
    }


def _causal_forcing_default_call_kwargs() -> Dict[str, Any]:
    return {
        "num_output_frames": 21,
        "seed": 0,
        "return_dict": True,
    }


def _oasis_500m_default_ref() -> str:
    return _checkpoint_model_ref(
        "Etched--oasis-500m/oasis500m.safetensors",
        fallback="Etched/oasis-500m",
    )


def _oasis_500m_default_load_kwargs() -> Dict[str, str]:
    checkpoint_root = Path(
        _checkpoint_model_ref(
            "Etched--oasis-500m",
            fallback="Etched/oasis-500m",
        )
    )
    return {
        "checkpoint_path": str(checkpoint_root / "oasis500m.safetensors"),
        "vae_ckpt": str(checkpoint_root / "vit-l-20.safetensors"),
        "prompt_path": str(_data_path("test_cases", "oasis", "sample_image_0.png")),
        "actions_path": str(
            _data_path("test_cases", "oasis", "forward_4f.one_hot_actions.json")
        ),
    }


def _multiworld_ittakestwo_default_load_kwargs() -> Dict[str, Any]:
    runtime_root = _project_root() / "worldfoundry" / "synthesis" / "visual_generation" / "multiworld"
    config_path = _data_path(
        "models",
        "runtime",
        "configs",
        "multiworld",
        "ittakestwo",
        "configs",
        "inference_480P_toy.yaml",
    )
    unified_python = _project_root().parent / "conda" / "envs" / "worldfoundry-unified-cu128" / "bin" / "python"
    return {
        "required_components": {
            "runtime_root": str(runtime_root),
            "config_path": str(config_path),
            "checkpoint_path": _checkpoint_model_ref(
                "MultiWorld/multiworld_480p_toydata.safetensors",
                "multiworld/multiworld_480p_toydata.safetensors",
                "Haoyuwu--MultiWorldCheckpoint/multiworld_480p_toydata.safetensors",
                "hfd/Haoyuwu--MultiWorldCheckpoint/multiworld_480p_toydata.safetensors",
                fallback=str(
                    checkpoint_root_path(
                        "MultiWorld",
                        "multiworld_480p_toydata.safetensors",
                    )
                ),
            ),
            "base_model_root": _checkpoint_model_ref(
                "Wan-AI--Wan2.2-TI2V-5B",
                "Wan2.2-TI2V-5B",
                "hfd/Wan-AI--Wan2.2-TI2V-5B",
                fallback=str(checkpoint_root_path("Wan-AI--Wan2.2-TI2V-5B")),
            ),
            "vggt_root": _checkpoint_model_ref(
                "VGGT-1B",
                "facebook--VGGT-1B",
                "hfd/facebook--VGGT-1B",
                fallback=str(checkpoint_root_path("facebook--VGGT-1B")),
            ),
            "python_executable": str(unified_python) if unified_python.is_file() else _worldfoundry_unified_python_executable(),
            "derive_env_obv_from_image": True,
            "num_inference_steps": 35,
            "inference_seed": 0,
            "fps": 60,
        }
    }


def _multiworld_ittakestwo_default_call_kwargs() -> Dict[str, Any]:
    return {
        "action_path": str(_data_path("test_cases", "multiworld_ittakestwo", "action.csv")),
        "num_frames": 81,
        "num_inference_steps": 35,
        "inference_seed": 0,
        "derive_env_obv_from_image": True,
        "return_dict": True,
        "save_name": "multiworld_ittakestwo",
        "fps": 60,
    }


def _evoke_default_ref() -> str:
    return _checkpoint_model_ref(
        "Evoke",
        "AlayaLab--Evoke",
        fallback="AlayaLab/Evoke",
    )


def _evoke_default_load_kwargs() -> Dict[str, Any]:
    return {
        "required_components": {
            "vigeo_path": _checkpoint_model_ref(
                "Evoke/ViGeo1.1",
                "pkqbajng--ViGeo1.1",
                fallback=str(checkpoint_root_path("Evoke", "ViGeo1.1")),
            ),
            "python_executable": _worldfoundry_unified_python_executable(),
        }
    }


def _magi2_default_load_kwargs() -> Dict[str, Any]:
    return {
        "required_components": {
            "python_executable": _worldfoundry_unified_python_executable(),
            "transformers_overlay": str(
                _project_root()
                / "artifacts"
                / "runtime_overlays"
                / "magi2-transformers-5.5.0"
            ),
        }
    }


def _warp_as_history_default_ref() -> str:
    return _checkpoint_model_ref(
        "BestWishYsh--Helios-Distilled",
        "Helios-Distilled",
        "helios-distilled",
        fallback=str(checkpoint_root_path("BestWishYsh--Helios-Distilled")),
    )


def _warp_as_history_default_load_kwargs() -> Dict[str, Any]:
    return {
        "required_components": {
            "lora_path": _checkpoint_model_ref(
                "yyfz233--warp-as-history/visible_lora_state_step1000.safetensors",
                "warp-as-history/visible_lora_state_step1000.safetensors",
                fallback=str(
                    checkpoint_root_path(
                        "yyfz233--warp-as-history",
                        "visible_lora_state_step1000.safetensors",
                    )
                ),
            ),
            "pi3x_ckpt_path": _checkpoint_model_ref(
                "yyfz233--Pi3X/model.safetensors",
                "Pi3X/model.safetensors",
                fallback=str(checkpoint_root_path("yyfz233--Pi3X", "model.safetensors")),
            ),
            "python_executable": _worldfoundry_unified_python_executable(),
        }
    }


def _runtime_module_path(model_name: str) -> str:
    family = model_name.split("_", maxsplit=1)[0]
    return f"worldfoundry.pipelines.{family}.pipeline_{model_name}"


@dataclass(frozen=True)
class CatalogEntry:
    model_id: str
    display_name: str
    module_path: str
    class_name: str
    family: str
    category: str
    summary: str
    call_params: tuple[str, ...] = field(default_factory=tuple)
    input_params: tuple[str, ...] = field(default_factory=tuple)
    stream_params: tuple[str, ...] = field(default_factory=tuple)
    load_params: tuple[str, ...] = field(default_factory=tuple)
    supports_stream: bool = False
    supports_from_pretrained: bool = False
    supports_api_init: bool = False
    runtime_kind: str = "default"
    default_backend: str = "auto"
    default_model_ref: str = ""
    default_endpoint: str = ""
    default_prompt: str = ""
    default_input_path: str = ""
    default_task_type: str = ""
    default_interactions: tuple[str, ...] = field(default_factory=tuple)
    default_load_kwargs: Dict[str, Any] = field(default_factory=dict)
    default_call_kwargs: Dict[str, Any] = field(default_factory=dict)
    extra_variants: tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    suggested_task_types: tuple[str, ...] = field(default_factory=tuple)
    aliases: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)
    env_hints: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    def search_blob(self) -> str:
        values = [
            self.model_id,
            self.display_name,
            self.family,
            self.category,
            self.summary,
            " ".join(self.aliases),
            " ".join(self.tags),
        ]
        return " ".join(values).lower()

    def task_suggestions_text(self) -> str:
        if not self.suggested_task_types:
            return ""
        return ", ".join(self.suggested_task_types)

    def to_table_row(self) -> list[str]:
        return [
            self.display_name,
            self.model_id,
            self.category,
            self.family,
            "yes" if self.supports_stream else "no",
            "api" if self.supports_api_init and not self.supports_from_pretrained else (
                "pretrained" if self.supports_from_pretrained and not self.supports_api_init else "both"
            ),
            ", ".join(self.tags[:4]),
        ]


@dataclass(frozen=True)
class _AstPipelineInfo:
    model_id: str
    display_name: str
    module_path: str
    class_name: str
    family: str
    call_params: tuple[str, ...]
    stream_params: tuple[str, ...]
    load_params: tuple[str, ...]
    supports_stream: bool
    supports_from_pretrained: bool
    supports_api_init: bool
    base_names: tuple[str, ...]


OFFICIAL_ACTION_CALL_PARAMS: tuple[str, ...] = (
    "prompt",
    "images",
    "video",
    "interactions",
    "output_path",
    "fps",
    "timeout_seconds",
    "checkpoint_path",
    "checkpoint_dir",
    "ckpt_path",
    "checkpoint_ref",
    "repo_id",
    "runtime_config_path",
    "plan_only",
    "official_policy_observation",
    "observation",
    "action_space",
    "policy_controls",
)
OFFICIAL_ACTION_LOAD_PARAMS: tuple[str, ...] = (
    "pretrained_model_path",
    "args",
    "device",
    "model_id",
    "profile_path",
    "manifest_path",
    "acquisition_root",
    "hf_models_root",
    "command_template",
    "checkpoint_path",
    "checkpoint_dir",
    "ckpt_path",
    "checkpoint_ref",
    "repo_id",
    "runtime_config_path",
)


def _action_checkpoint_variant(
    *,
    variant_id: str,
    label: str,
    repo_id: str,
    call_kwargs: Mapping[str, Any],
    aliases: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> Dict[str, Any]:
    """Build one local-first Workspace variant for an in-tree action policy."""

    return {
        "variant_id": variant_id,
        "label": label,
        "status": "checkpoint-configured",
        "checkpoints": (
            {
                "role": "primary",
                "uri": _hf_checkpoint_model_ref(f"hfd/{repo_id.replace('/', '--')}", repo_id),
                "status": "checkpoint-configured",
            },
        ),
        "call_kwargs": dict(call_kwargs),
        "aliases": tuple(aliases),
        "notes": tuple(notes),
    }


def _xvla_workspace_variants() -> tuple[Dict[str, Any], ...]:
    common = {
        "state": [0.0] * 20,
        "denoising_steps": 10,
        "seed": 0,
        "torch_dtype": "float32",
        "attention_backend": "auto",
        "plan_only": False,
    }
    specs = (
        ("google-robot", "Google Robot", "2toINF/X-VLA-Google-Robot", 1),
        ("libero", "LIBERO", "2toINF/X-VLA-Libero", 3),
        ("calvin-abc-d", "CALVIN ABC→D", "2toINF/X-VLA-Calvin-ABC_D", 2),
        ("robotwin2", "RoboTwin 2", "2toINF/X-VLA-RoboTwin2", 6),
        ("vlabench", "VLABench", "2toINF/X-VLA-VLABench", 8),
        ("agiworld", "AgiWorld Challenge", "2toINF/X-VLA-AgiWorld-Challenge", 9),
        ("softfold", "SoftFold", "2toINF/X-VLA-SoftFold", 5),
        ("foundation-domain-0", "Foundation (domain 0)", "2toINF/X-VLA-Pt", 0),
    )
    return tuple(
        _action_checkpoint_variant(
            variant_id=variant_id,
            label=label,
            repo_id=repo_id,
            call_kwargs={**common, "domain_id": domain_id},
            aliases=(repo_id.rsplit("/", 1)[-1],),
            notes=(
                "The foundation checkpoint needs an embodiment domain ID; this Workspace preset uses domain 0."
                if variant_id == "foundation-domain-0"
                else f"Uses the released domain ID {domain_id} for this embodiment."
            ,),
        )
        for variant_id, label, repo_id, domain_id in specs
    )


def _xiaomi_robotics_0_workspace_variants() -> tuple[Dict[str, Any], ...]:
    common = {
        "num_steps": 5,
        "seed": 42,
        "torch_dtype": "auto",
        "attn_implementation": "auto",
        "plan_only": False,
    }
    specs: tuple[tuple[str, str, str, Dict[str, Any]], ...] = (
        (
            "libero",
            "LIBERO",
            "XiaomiRobotics/Xiaomi-Robotics-0-LIBERO",
            {"variant": "libero", "state": [0.0] * 32, "wrist_image": str(_data_path("test_cases", "studio_demo", "00", "image.jpg"))},
        ),
        (
            "calvin-abcd",
            "CALVIN ABCD→D",
            "XiaomiRobotics/Xiaomi-Robotics-0-Calvin-ABCD_D",
            {"variant": "calvin-abcd", "state": [0.0] * 32, "wrist_image": str(_data_path("test_cases", "studio_demo", "00", "image.jpg"))},
        ),
        (
            "calvin-abc",
            "CALVIN ABC→D",
            "XiaomiRobotics/Xiaomi-Robotics-0-Calvin-ABC_D",
            {"variant": "calvin-abc", "state": [0.0] * 32, "wrist_image": str(_data_path("test_cases", "studio_demo", "00", "image.jpg"))},
        ),
        (
            "simplerenv-google",
            "SimplerEnv Google Robot",
            "XiaomiRobotics/Xiaomi-Robotics-0-SimplerEnv-Google-Robot",
            {"variant": "simplerenv-google", "state": [0.0] * 7},
        ),
        (
            "simplerenv-widowx",
            "SimplerEnv WidowX",
            "XiaomiRobotics/Xiaomi-Robotics-0-SimplerEnv-WidowX",
            {"variant": "simplerenv-widowx", "state": [0.0] * 7},
        ),
        (
            "pretrain-droid",
            "Pretrain (DROID)",
            "XiaomiRobotics/Xiaomi-Robotics-0-Pretrain",
            {
                "variant": "pretrain",
                "robot_type": "droid_pt",
                "camera_keys": ["base"],
                "view_labels": ["External Camera View"],
                "state": [0.0] * 32,
            },
        ),
    )
    return tuple(
        _action_checkpoint_variant(
            variant_id=variant_id,
            label=label,
            repo_id=repo_id,
            call_kwargs={**common, **variant_kwargs},
            aliases=(repo_id.rsplit("/", 1)[-1],),
        )
        for variant_id, label, repo_id, variant_kwargs in specs
    )


def _spatial_forcing_workspace_variants() -> tuple[Dict[str, Any], ...]:
    image = str(_data_path("test_cases", "studio_demo", "00", "image.jpg"))
    specs = (
        ("libero-spatial", "LIBERO Spatial", "haofuly/spatial-forcing-7b-finetuned-libero-spatial", "libero_spatial", "libero_spatial_no_noops"),
        ("libero-object", "LIBERO Object", "haofuly/spatial-forcing-7b-finetuned-libero-object", "libero_object", "libero_object_no_noops"),
        ("libero-goal", "LIBERO Goal", "haofuly/spatial-forcing-7b-finetuned-libero-goal", "libero_goal", "libero_goal_no_noops"),
        ("libero-10", "LIBERO 10", "haofuly/spatial-forcing-7b-finetuned-libero-10", "libero_10", "libero_10_no_noops"),
    )
    return tuple(
        _action_checkpoint_variant(
            variant_id=variant_id,
            label=label,
            repo_id=repo_id,
            call_kwargs={
                "variant": variant_id.removeprefix("libero-"),
                "checkpoint_variant": variant_id.removeprefix("libero-"),
                "state": [0.0] * 8,
                "wrist_image": image,
                "task_suite_name": suite,
                "unnorm_key": unnorm_key,
                "torch_dtype": "auto",
                "attn_implementation": "auto",
                "plan_only": False,
            },
            aliases=(repo_id.rsplit("/", 1)[-1],),
        )
        for variant_id, label, repo_id, suite, unnorm_key in specs
    )


def _fastwam_workspace_variants() -> tuple[Dict[str, Any], ...]:
    aloha = _data_path("test_cases", "test_vla_case1", "aloha")
    libero = _data_path("test_cases", "test_vla_case1", "libero")
    common = {
        "torch_dtype": "auto",
        "num_inference_steps": 10,
        "seed": 42,
        "plan_only": False,
    }
    return (
        _action_checkpoint_variant(
            variant_id="robotwin",
            label="RoboTwin",
            repo_id="yuanty/fastwam",
            call_kwargs={
                **common,
                "variant": "robotwin",
                "state": [0.0] * 14,
                "head_camera": str(aloha / "observation_images_cam_high.png"),
                "left_camera": str(aloha / "observation_images_cam_left_wrist.png"),
                "right_camera": str(aloha / "observation_images_cam_right_wrist.png"),
            },
        ),
        _action_checkpoint_variant(
            variant_id="libero",
            label="LIBERO",
            repo_id="yuanty/fastwam",
            call_kwargs={
                **common,
                "variant": "libero",
                "state": [0.0] * 8,
                "wrist_image": str(libero / "wrist_view.png"),
            },
        ),
    )


def _x_wam_workspace_variants() -> tuple[Dict[str, Any], ...]:
    image = str(_data_path("test_cases", "studio_demo", "00", "image.jpg"))
    common = {
        "rgb_views": [image] * 3,
        "torch_dtype": "auto",
        "denoise_steps": 50,
        "action_denoise_steps": 10,
        "cfg_scale": 0.0,
        "compile_model": False,
        "generate_world": False,
        "run_depth": False,
        "plan_only": False,
    }
    return (
        _action_checkpoint_variant(
            variant_id="robocasa-sft",
            label="RoboCasa SFT",
            repo_id="sharinka0715/X-WAM-checkpoints",
            call_kwargs={**common, "variant": "robocasa_sft", "state": [0.0] * 8},
        ),
        _action_checkpoint_variant(
            variant_id="robotwin-sft",
            label="RoboTwin SFT",
            repo_id="sharinka0715/X-WAM-checkpoints",
            call_kwargs={**common, "variant": "robotwin_sft", "state": [0.0] * 16},
        ),
    )


def _leworldmodel_default_load_kwargs() -> Dict[str, Any]:
    candidates = checkpoint_root_candidates("leworldmodel", "pusht_worldfoundry_smoke.h5")
    dataset_path = next((path for path in candidates if path.is_file()), candidates[0])
    return {
        "model_id": "leworldmodel",
        "config_name": "pusht",
        "policy": "random",
        "cache_dir": str(dataset_path.parent),
        "dataset_name": str(dataset_path),
        "num_eval": 1,
        "eval_budget": 25,
        "goal_offset_steps": 5,
        "seed": 42,
    }


def _shotstream_default_load_kwargs() -> Dict[str, Any]:
    return {
        "model_id": "shotstream",
        "checkpoint_dir": str(checkpoint_root_path("KlingTeam--ShotStream")),
        "wan_model_dir": str(checkpoint_root_path("Wan-AI--Wan2.1-T2V-1.3B")),
    }


def _egowm_default_load_kwargs() -> Dict[str, Any]:
    return {
        "model_id": "egowm",
        "checkpoint_path": str(checkpoint_root_path("anuragba--egowm", "svd_25dof_nav.pth")),
        "base_model_dir": str(checkpoint_root_path("stabilityai--stable-video-diffusion-img2vid")),
    }


def _hma_default_load_kwargs() -> Dict[str, Any]:
    return {
        "model_id": "hma",
        "checkpoint_dir": str(checkpoint_root_path("liruiw--hma-base-cont")),
        "base_model_dir": str(checkpoint_root_path("stabilityai--stable-video-diffusion-img2vid")),
    }


def _tesseract_default_load_kwargs() -> Dict[str, Any]:
    return {
        "model_id": "tesseract",
        "checkpoint_dir": str(checkpoint_root_path("anyeZHY--tesseract", "tesseract_v01e_rgbdn_sft")),
        "base_model_dir": str(checkpoint_root_path("THUDM--CogVideoX-5b-I2V")),
    }


def _ctrl_world_default_load_kwargs() -> Dict[str, Any]:
    return {
        "model_id": "ctrl-world",
        "checkpoint_path": str(checkpoint_root_path("yjguo--Ctrl-World", "checkpoint-10000.pt")),
        "base_model_dir": str(checkpoint_root_path("stabilityai--stable-video-diffusion-img2vid")),
        "clip_model_dir": str(checkpoint_root_path("openai--clip-vit-base-patch32")),
    }


def _genie_envisioner_default_load_kwargs() -> Dict[str, Any]:
    return {
        "model_id": "genie-envisioner",
        "checkpoint_path": str(
            checkpoint_root_path(
                "agibot-world--Genie-Envisioner", "GE_base_fast_v0.1.safetensors"
            )
        ),
        "base_model_dir": str(checkpoint_root_path("Lightricks--LTX-Video")),
    }


def _giga_world_0_default_load_kwargs() -> Dict[str, Any]:
    return {
        "model_id": "giga-world-0",
        "transformer_model_dir": str(
            checkpoint_root_path(
                "open-gigaai--GigaWorld-0-Video-GR1-2b", "transformer"
            )
        ),
        "text_encoder_model_dir": str(
            checkpoint_root_path("google-t5--t5-11b-encoder")
        ),
        "vae_model_dir": str(
            checkpoint_root_path("Wan-AI--Wan2.1-T2V-1.3B-Diffusers", "vae")
        ),
    }


def _spatia_default_ref() -> str:
    return _checkpoint_model_ref(
        "Jinjing713--Spatia",
        fallback="Jinjing713/Spatia",
    )


def _spatia_default_load_kwargs() -> Dict[str, Any]:
    return {
        "base_model_path": _checkpoint_model_ref(
            "Wan-AI--Wan2.2-TI2V-5B",
            "Wan2.2-TI2V-5B",
            fallback="Wan-AI/Wan2.2-TI2V-5B",
        ),
        "map_model_path": _checkpoint_model_ref(
            "facebook--map-anything",
            fallback="facebook/map-anything",
        ),
    }


def _spatia_default_call_kwargs() -> Dict[str, Any]:
    return {
        "w2c_trajectory_file": str(
            _data_path("test_cases", "spatia", "w2c_pan_right_121.jsonl")
        ),
        "intrinsics": [[[768.0, 0.0, 624.0], [0.0, 768.0, 352.0], [0.0, 0.0, 1.0]]],
        "num_frames": 121,
        "fps": 24,
        "width": 1248,
        "height": 704,
        "num_inference_steps": 40,
        "seed": 20917,
        "return_dict": True,
    }


def _joyai_echo_source_root() -> str:
    return str(_project_root() / "cache" / "generative_taxonomy" / "jd-opensource--JoyAI-Echo")


def _joyai_echo_python_executable(environment_name: str) -> str:
    workspace_root = _project_root().parent
    return _first_existing_path(
        workspace_root / "envs" / environment_name / "bin" / "python",
        workspace_root / "conda" / "envs" / environment_name / "bin" / "python",
    )


def _joyai_echo_longvideo_default_ref() -> str:
    return _checkpoint_model_ref(
        "jdopensource--JoyAI-Echo",
        fallback="jdopensource/JoyAI-Echo",
    )


def _joyai_echo_longvideo_default_load_kwargs() -> Dict[str, Any]:
    return {
        "source_root": _joyai_echo_source_root(),
        "checkpoint_dir": _joyai_echo_longvideo_default_ref(),
        "gemma_path": _checkpoint_model_ref(
            "google--gemma-3-12b-it",
            fallback="google/gemma-3-12b-it",
        ),
        "python_executable": _joyai_echo_python_executable("joyai-echo-longvideo"),
        "precision": "bf16",
    }


def _joyai_echo_wm_default_ref() -> str:
    return _checkpoint_model_ref(
        "Echo-Team--Echo-WM",
        fallback="Echo-Team/Echo-WM",
    )


def _joyai_echo_wm_default_load_kwargs() -> Dict[str, Any]:
    return {
        "source_root": _joyai_echo_source_root(),
        "checkpoint_dir": _joyai_echo_wm_default_ref(),
        "gemma_path": _checkpoint_model_ref(
            "google--gemma-3-12b-it-qat-q4_0-unquantized",
            fallback="google/gemma-3-12b-it-qat-q4_0-unquantized",
        ),
        "python_executable": _joyai_echo_python_executable("joyai-echo-wm"),
        "variant": "base",
    }


def _hyworld_worldgen_scene_ref() -> str:
    configured = os.getenv("HYWORLD_SCENE_PATH", "").strip()
    if configured:
        return configured
    return str(_data_path("test_cases", "hyworld_worldgen", "materialized_scene"))


CURATED_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "abot-world-0-5b-lf": {
        "display_name": "ABot-World-0-5B-LF",
        "category": "Video Generation",
        "summary": "In-tree causal action-conditioned world-video generation with resident KV cache.",
        "default_model_ref": lambda: str(checkpoint_root_path("ABot-World-0-5B-LF")),
        "default_prompt": "Move forward through the scene while preserving geometry and appearance.",
        "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_call_kwargs": {
            "num_frames": 57,
            "num_blocks": 5,
            "seed": 42,
            "fps": 16,
        },
        "call_params": (
            "images",
            "prompt",
            "interactions",
            "reference_images",
            "num_frames",
            "num_blocks",
            "seed",
            "fps",
            "output_path",
            "return_dict",
        ),
        "supports_stream": True,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("abot-world", "abot"),
        "tags": ("world-model", "action-control", "causal-video", "in-tree-runtime"),
        "notes": "Uses the released full 57-frame rollout: one nine-frame warmup block followed by four twelve-frame causal blocks.",
    },
    "alayaworld": {
        "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_load_kwargs": _alayaworld_default_load_kwargs,
        "default_call_kwargs": {"return_dict": True},
        "call_params": (
            "prompt",
            "images",
            "output_path",
            "fps",
            "return_dict",
            "num_frames",
            "rounds",
            "height",
            "width",
            "sampling_steps",
            "seed",
            "camera_path",
            "camera_trajectory",
            "camera_translation_step",
            "camera_rotation_step_degrees",
            "intrinsic",
            "action_scale",
            "action_freq_scale",
            "action_history_memory",
            "spatial_enabled",
            "depth_backend",
            "spatial_num_context_frames",
            "spatial_retrieval_views",
            "spatial_downsample",
            "spatial_maximum_coverage",
            "spatial_retrieval_depth_threshold",
            "spatial_constant_depth",
            "spatial_include_sink",
            "spatial_require_full_context",
            "da3_process_res",
            "context_parallel",
            "decode_rank0_only",
            "compile_mode",
            "compile_backend",
            "compile_fullgraph",
            "compile_dynamic",
        ),
        "tags": ("world-model", "camera-control", "autoregressive-video", "in-tree-runtime"),
        "notes": "Uses the released Alaya checkpoint, local Gemma shards, and a compatible nested-giant DA3 checkpoint.",
    },
    "wan21-fun-1p3b-cam": {
        "display_name": "Wan2.1-Fun V1.1 1.3B Control Camera",
        "category": "Video Generation",
        "summary": "In-tree VideoX-Fun image-to-video generation with CameraCtrl pose conditioning.",
        "default_model_ref": lambda: _hf_checkpoint_model_ref(
            "alibaba-pai--Wan2.1-Fun-V1.1-1.3B-Control-Camera",
            "alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera",
        ),
        "default_prompt": "Colorful fireworks bloom above a city skyline as the camera moves forward.",
        "default_input_path": str(_data_path("test_cases", "video_x_fun", "firework.png")),
        "default_call_kwargs": {
            "pose_txt": str(_data_path("test_cases", "video_x_fun", "camera_pose.txt")),
            "width": 832,
            "height": 480,
            "num_frames": 49,
            "fps": 16,
            "num_inference_steps": 50,
            "seed": 43,
        },
        "call_params": (
            "prompt",
            "images",
            "image_path",
            "pose_txt",
            "width",
            "height",
            "num_frames",
            "fps",
            "num_inference_steps",
            "seed",
            "output_path",
            "return_dict",
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("wan2.1-fun-1.3b-camera",),
        "tags": ("camera-control", "i2v", "wan2.1", "in-tree-runtime"),
        "notes": "Split official transformer/base assets are composed with symlinks inside each run directory; the full upstream camera demo uses 49 frames at 832x480 with 50 denoising steps.",
    },
    "wan21-fun-14b-cam": {
        "display_name": "Wan2.1-Fun V1.1 14B Control Camera",
        "category": "Video Generation",
        "summary": "In-tree VideoX-Fun 14B image-to-video generation with CameraCtrl pose conditioning.",
        "default_model_ref": lambda: _hf_checkpoint_model_ref(
            "alibaba-pai--Wan2.1-Fun-V1.1-14B-Control-Camera",
            "alibaba-pai/Wan2.1-Fun-V1.1-14B-Control-Camera",
        ),
        "default_prompt": "Colorful fireworks bloom above a city skyline as the camera moves forward.",
        "default_input_path": str(_data_path("test_cases", "video_x_fun", "firework.png")),
        "default_call_kwargs": {
            "pose_txt": str(_data_path("test_cases", "video_x_fun", "camera_pose.txt")),
            "width": 832,
            "height": 480,
            "num_frames": 49,
            "fps": 16,
            "num_inference_steps": 50,
            "seed": 43,
        },
        "call_params": (
            "prompt",
            "images",
            "image_path",
            "pose_txt",
            "width",
            "height",
            "num_frames",
            "fps",
            "num_inference_steps",
            "seed",
            "output_path",
            "return_dict",
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("wan2.1-fun-14b-camera",),
        "tags": ("camera-control", "i2v", "wan2.1", "in-tree-runtime"),
        "notes": "The full upstream Wan2.1 camera demo uses 49 frames at 832x480 with 50 denoising steps.",
    },
    "wan22-fun-5b-cam": {
        "display_name": "Wan2.2-Fun 5B Control Camera",
        "category": "Video Generation",
        "summary": "In-tree VideoX-Fun 5B image-to-video generation with CameraCtrl pose conditioning.",
        "default_model_ref": lambda: _hf_checkpoint_model_ref(
            "alibaba-pai--Wan2.2-Fun-5B-Control-Camera",
            "alibaba-pai/Wan2.2-Fun-5B-Control-Camera",
        ),
        "default_prompt": "Colorful fireworks bloom above a city skyline as the camera moves forward.",
        "default_input_path": str(_data_path("test_cases", "video_x_fun", "firework.png")),
        "default_call_kwargs": {
            "pose_txt": str(_data_path("test_cases", "video_x_fun", "camera_pose.txt")),
            "width": 1280,
            "height": 704,
            "num_frames": 121,
            "fps": 24,
            "num_inference_steps": 50,
            "seed": 43,
        },
        "call_params": (
            "prompt",
            "images",
            "image_path",
            "pose_txt",
            "width",
            "height",
            "num_frames",
            "fps",
            "num_inference_steps",
            "seed",
            "output_path",
            "return_dict",
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("wan2.2-fun-5b-camera",),
        "tags": ("camera-control", "i2v", "wan2.2", "in-tree-runtime"),
        "notes": "The full upstream Wan2.2 5B camera demo uses 121 frames at 1280x704 and 24 FPS with 50 denoising steps.",
    },
    "wan22-fun-a14b-cam": {
        "display_name": "Wan2.2-Fun A14B Control Camera",
        "category": "Video Generation",
        "summary": "In-tree VideoX-Fun A14B image-to-video generation with CameraCtrl pose conditioning.",
        "default_model_ref": lambda: _hf_checkpoint_model_ref(
            "alibaba-pai--Wan2.2-Fun-A14B-Control-Camera",
            "alibaba-pai/Wan2.2-Fun-A14B-Control-Camera",
        ),
        "default_prompt": "Colorful fireworks bloom above a city skyline as the camera moves forward.",
        "default_input_path": str(_data_path("test_cases", "video_x_fun", "firework.png")),
        "default_call_kwargs": {
            "pose_txt": str(_data_path("test_cases", "video_x_fun", "camera_pose.txt")),
            "width": 832,
            "height": 480,
            "num_frames": 81,
            "fps": 16,
            "num_inference_steps": 50,
            "seed": 43,
        },
        "call_params": (
            "prompt",
            "images",
            "image_path",
            "pose_txt",
            "width",
            "height",
            "num_frames",
            "fps",
            "num_inference_steps",
            "seed",
            "output_path",
            "return_dict",
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("wan2.2-fun-a14b-camera",),
        "tags": ("camera-control", "i2v", "wan2.2", "in-tree-runtime"),
        "notes": "The full upstream Wan2.2 A14B camera demo uses 81 frames at 832x480 with 50 denoising steps.",
    },
    "xiaomi-robotics-0": {
        "display_name": "Xiaomi-Robotics-0",
        "category": "Embodied Action",
        "summary": "In-tree MiBoT VLA inference for Xiaomi-Robotics-0 robot-action checkpoints.",
        "default_model_ref": "XiaomiRobotics/Xiaomi-Robotics-0-SimplerEnv-Google-Robot",
        "default_prompt": "pick up the object and place it on the target",
        "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_call_kwargs": {
            "variant": "simplerenv-google",
            "state": [0.0] * 7,
            "num_steps": 5,
            "seed": 42,
            "torch_dtype": "auto",
            "attn_implementation": "auto",
            "plan_only": False,
        },
        "extra_variants": _xiaomi_robotics_0_workspace_variants,
        "call_params": (
            "prompt",
            "images",
            "state",
            "wrist_image",
            "variant",
            "robot_type",
            "camera_keys",
            "view_labels",
            "num_steps",
            "seed",
            "torch_dtype",
            "attn_implementation",
            "plan_only",
            "output_path",
            "return_dict",
        ),
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("xiaomi-robotics-zero", "xr0"),
        "tags": ("vla", "policy", "robot-action", "in-tree-runtime"),
        "notes": "SimplerEnv Google Robot checkpoint-backed A100 BF16/SDPA inference is validated.",
    },
    "hy-embodied-vla": {
        "display_name": "Hy-Embodied-0.5-VLA",
        "category": "Embodied Action",
        "summary": "Tencent Hy-Embodied-0.5 multi-view flow-matching VLA policy.",
        "default_model_ref": lambda: _hf_checkpoint_model_ref(
            "hfd/tencent--Hy-Embodied-0.5-VLA-RoboTwin",
            "tencent/Hy-Embodied-0.5-VLA-RoboTwin",
        ),
        "default_prompt": "pick up the object and place it on the target",
        "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_call_kwargs": {
            "variant": "robotwin",
            "state": [0.0] * 20,
            "state_format": "normalized",
            "blend_mode": "auto",
            "history_size": 6,
            "replicate_single_image": True,
            "torch_dtype": "auto",
            "plan_only": False,
        },
        "extra_variants": (
            {
                "variant_id": "umi",
                "label": "UMI",
                "status": "checkpoint_gpu_validated",
                "checkpoints": (
                    {
                        "role": "primary",
                        "uri": lambda: _hf_checkpoint_model_ref(
                            "hfd/tencent--Hy-Embodied-0.5-VLA-UMI",
                            "tencent/Hy-Embodied-0.5-VLA-UMI",
                        ),
                        "status": "checkpoint_gpu_validated",
                    },
                ),
                "call_kwargs": {
                    "variant": "umi",
                    "state": [0.0] * 20,
                    "state_format": "normalized",
                    "blend_mode": "auto",
                    "history_size": 1,
                    "replicate_single_image": True,
                    "torch_dtype": "auto",
                    "plan_only": False,
                },
                "aliases": ("hy-embodied-vla-umi", "hy-vla-umi"),
                "notes": ("Released UMI 50-step relative-action policy.",),
            },
        ),
        "call_params": (
            "prompt",
            "images",
            "state",
            "variant",
            "state_format",
            "blend_mode",
            "history_size",
            "replicate_single_image",
            "torch_dtype",
            "plan_only",
            "output_path",
            "return_dict",
        ),
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("hy-vla", "hy-embodied-0.5-vla"),
        "tags": ("vla", "policy", "multi-view", "robot-action", "in-tree-runtime"),
        "notes": "RoboTwin checkpoint-backed A100 BF16 inference is validated.",
    },
    "lda-1b": {
        "display_name": "LDA-1B",
        "category": "Embodied Action",
        "summary": "In-tree LDA-1B Qwen3-VL/DINOv3 flow-matching action policy.",
        "default_model_ref": lambda: _hf_checkpoint_model_ref(
            "hfd/Wayer2--LDA-robocasa",
            "Wayer2/LDA-robocasa",
        ),
        "default_prompt": "move the object to the target area",
        "default_input_path": str(
            _data_path("test_cases", "test_vla_case1", "libero", "main_view.png")
        ),
        "default_call_kwargs": lambda: {
            "official_policy_observation": {
                "ego_view": [
                    str(_data_path("test_cases", "test_vla_case1", "libero", "main_view.png")),
                    str(_data_path("test_cases", "test_vla_case1", "libero", "main_view.png")),
                ],
                "state": [0.0] * 29,
            },
            "torch_dtype": "auto",
            "attention_backend": "auto",
        },
        "default_interactions": ("robot_action", "action_chunk"),
        "call_params": OFFICIAL_ACTION_CALL_PARAMS,
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("lda1b", "lda-robocasa"),
        "tags": ("vla", "policy", "robot-action", "flow-matching", "in-tree-runtime"),
        "notes": "The default Workspace contract uses the released RoboCasa GR1 29-D state/action layout.",
    },
    "internvla-a1": {
        "display_name": "InternVLA-A1-3B",
        "category": "Embodied Action",
        "summary": "In-tree InternVLA-A1 understanding, foresight, and flow-action policy.",
        "default_model_ref": lambda: _hf_checkpoint_model_ref(
            "hfd/InternRobotics--InternVLA-A1-3B",
            "InternRobotics/InternVLA-A1-3B",
        ),
        "default_prompt": "pick up the object and place it on the target",
        "default_input_path": str(
            _data_path("test_cases", "test_vla_case1", "aloha", "observation_images_cam_high.png")
        ),
        "default_call_kwargs": lambda: {
            "official_policy_observation": {
                "cam_high": str(
                    _data_path("test_cases", "test_vla_case1", "aloha", "observation_images_cam_high.png")
                ),
                "cam_left_wrist": str(
                    _data_path(
                        "test_cases", "test_vla_case1", "aloha", "observation_images_cam_left_wrist.png"
                    )
                ),
                "cam_right_wrist": str(
                    _data_path(
                        "test_cases", "test_vla_case1", "aloha", "observation_images_cam_right_wrist.png"
                    )
                ),
                "state": [0.0] * 14,
                "reset": True,
            },
            "torch_dtype": "auto",
        },
        "default_interactions": ("robot_action", "action_chunk"),
        "call_params": OFFICIAL_ACTION_CALL_PARAMS,
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("internvla-a1-3b", "internvla-a-series"),
        "tags": ("vla", "policy", "robot-action", "world-model", "in-tree-runtime"),
        "notes": "The default Workspace contract uses the released ALOHA three-camera 14-D statistics block.",
    },
    "fastwam": {
        "display_name": "FastWAM",
        "category": "Embodied Action",
        "summary": "In-tree FastWAM first-frame-cached world-action policy for RoboTwin and LIBERO.",
        "default_model_ref": lambda: _hf_checkpoint_model_ref(
            "hfd/yuanty--fastwam",
            "yuanty/fastwam",
        ),
        "default_prompt": "pick up the object and place it on the target",
        "default_input_path": str(
            _data_path(
                "test_cases",
                "test_vla_case1",
                "aloha",
                "observation_images_cam_high.png",
            )
        ),
        "default_call_kwargs": {
            "variant": "robotwin",
            "state": [0.0] * 14,
            "head_camera": str(
                _data_path(
                    "test_cases",
                    "test_vla_case1",
                    "aloha",
                    "observation_images_cam_high.png",
                )
            ),
            "left_camera": str(
                _data_path(
                    "test_cases",
                    "test_vla_case1",
                    "aloha",
                    "observation_images_cam_left_wrist.png",
                )
            ),
            "right_camera": str(
                _data_path(
                    "test_cases",
                    "test_vla_case1",
                    "aloha",
                    "observation_images_cam_right_wrist.png",
                )
            ),
            "torch_dtype": "auto",
            "num_inference_steps": 10,
            "seed": 42,
            "plan_only": False,
        },
        "extra_variants": _fastwam_workspace_variants,
        "call_params": (
            "prompt",
            "images",
            "state",
            "variant",
            "head_camera",
            "left_camera",
            "right_camera",
            "wrist_image",
            "combined_image",
            "model_image",
            "context",
            "context_mask",
            "torch_dtype",
            "num_inference_steps",
            "denoising_steps",
            "sigma_shift",
            "seed",
            "rand_device",
            "tiled",
            "binarize_libero_gripper",
            "plan_only",
            "output_path",
            "return_dict",
        ),
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("fast-wam",),
        "tags": ("wam", "robot-action", "libero", "robotwin", "in-tree-runtime"),
        "notes": "Both released checkpoints are staged for full Workspace GPU validation.",
    },
    "spatial-forcing": {
        "display_name": "Spatial-Forcing",
        "category": "Embodied Action",
        "summary": "Inference-only Spatial-Forcing OpenVLA-OFT policy for LIBERO action chunks.",
        "default_model_ref": "haofuly/spatial-forcing-7b-finetuned-libero-spatial",
        "default_prompt": "pick up the object and place it on the target",
        "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_call_kwargs": {
            "state": [0.0] * 8,
            "wrist_image": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
            "task_suite_name": "libero_spatial",
            "unnorm_key": "libero_spatial_no_noops",
            "torch_dtype": "auto",
            "attn_implementation": "auto",
            "plan_only": False,
        },
        "extra_variants": _spatial_forcing_workspace_variants,
        "call_params": (
            "prompt",
            "images",
            "wrist_image",
            "state",
            "variant",
            "checkpoint_variant",
            "task_suite_name",
            "unnorm_key",
            "torch_dtype",
            "attn_implementation",
            "plan_only",
            "output_path",
            "return_dict",
        ),
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("spatial_forcing",),
        "tags": ("vla", "openvla", "libero", "robot-action", "in-tree-runtime"),
        "notes": "LIBERO-Spatial checkpoint-backed A100 inference is validated; training-only VGGT alignment is excluded.",
    },
    "x-wam": {
        "display_name": "X-WAM",
        "category": "Embodied Action",
        "summary": "Joint multi-view world-action inference with action, proprioception, and optional future video.",
        "default_model_ref": "sharinka0715/X-WAM-checkpoints",
        "default_prompt": "pick up the object and place it on the target",
        "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_call_kwargs": {
            "variant": "robocasa_sft",
            "rgb_views": [str(_data_path("test_cases", "studio_demo", "00", "image.jpg"))] * 3,
            "state": [0.0] * 8,
            "torch_dtype": "auto",
            "denoise_steps": 50,
            "action_denoise_steps": 10,
            "cfg_scale": 0.0,
            "compile_model": False,
            "generate_world": False,
            "run_depth": False,
            "plan_only": False,
        },
        "extra_variants": _x_wam_workspace_variants,
        "call_params": (
            "prompt",
            "images",
            "rgb_views",
            "state",
            "variant",
            "torch_dtype",
            "denoise_steps",
            "action_denoise_steps",
            "cfg_scale",
            "generate_world",
            "run_depth",
            "world_video_path",
            "compile_model",
            "plan_only",
            "output_path",
            "return_dict",
        ),
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("x_wam",),
        "tags": ("wam", "robot-action", "video-prediction", "multi-view", "in-tree-runtime"),
        "notes": "RoboCasa checkpoint-backed A100 action inference and the optional 50-step world-video decode are validated; world decode remains opt-in.",
    },
    "xvla": {
        "display_name": "X-VLA",
        "category": "Embodied Action",
        "summary": "Cross-embodiment Florence-2 VLA policy with a domain-aware flow action head.",
        "default_model_ref": "2toINF/X-VLA-WidowX",
        "default_prompt": "pick up the object and place it on the target",
        "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_call_kwargs": {
            "state": [0.0] * 20,
            "domain_id": 0,
            "denoising_steps": 10,
            "seed": 0,
            "torch_dtype": "float32",
            "attention_backend": "auto",
            "plan_only": False,
        },
        "extra_variants": _xvla_workspace_variants,
        "call_params": (
            "prompt",
            "images",
            "state",
            "domain_id",
            "denoising_steps",
            "seed",
            "torch_dtype",
            "attention_backend",
            "adapter_path",
            "plan_only",
            "output_path",
            "return_dict",
        ),
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("x-vla", "2toinf/x-vla"),
        "tags": ("vla", "cross-embodiment", "robot-action", "in-tree-runtime"),
        "notes": "WidowX checkpoint-backed A100 FP32/SDPA action inference is validated.",
    },
    "stable-video-infinity": {
        "display_name": "Stable Video Infinity 2.0",
        "category": "Video Generation",
        "summary": "Generate coherent long video from one image and a segment-level prompt stream.",
        "default_model_ref": _stable_video_infinity_default_ref,
        "default_load_kwargs": _stable_video_infinity_default_load_kwargs,
        "default_prompt": (
            "The water shimmers with dancing caustics as gentle bubbles rise; "
            "the camera dolly-ins through a blue haze."
        ),
        "default_input_path": str(
            _data_path("test_cases", "stable-video-infinity", "svi-2.0", "frame.jpg")
        ),
        "default_task_type": "i2v",
        "default_call_kwargs": {
            "num_clips": 999,
            "num_frames": 81,
            "num_motion_frames": 5,
            "num_inference_steps": 50,
            "cfg_scale_text": 5.0,
            "sigma_shift": 5.0,
            "prompt_repeat_times": 2,
            "fps": 24,
            "base_seed": 0,
        },
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "lazy",
            "wan_model_dir",
            "svi_lora_path",
            "dtype",
            "enable_vram_management",
            "num_persistent_param_in_dit",
            "use_usp",
            *DISPATCH_LOAD_PARAMS,
        ),
        "call_params": (
            "prompt",
            "images",
            "prompt_stream",
            "output_path",
            "fps",
            "return_dict",
            "num_clips",
            "num_frames",
            "num_motion_frames",
            "num_inference_steps",
            "cfg_scale_text",
            "guidance_scale",
            "sigma_shift",
            "shift",
            "ref_pad_cfg",
            "ref_pad_num",
            "prompt_repeat_times",
            "use_first_prompt_only",
            "repeat_first_clip",
            "prompt_prefix",
            "base_seed",
            "seed",
            "seed_stride",
            "max_width",
            "height",
            "width",
            "tiled",
            "tile_size",
            "tile_stride",
            "negative_prompt",
        ),
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("svi", "svi-2.0", "stable_video_infinity", "stable-video-infinity-2.0"),
        "tags": ("video", "i2v", "long-video", "prompt-stream", "in-tree-runtime"),
        "notes": (
            "Uses the in-tree SVI 2.0 continuation runtime; Wan2.1 and SVI weights remain in local HFD staging."
        ),
    },
    "openvla-oft": {
        "display_name": "OpenVLA-OFT",
        "category": "Embodied Action",
        "default_model_ref": lambda: _hf_checkpoint_model_ref(
            "hfd/moojink--openvla-7b-oft-finetuned-libero-spatial",
            "moojink/openvla-7b-oft-finetuned-libero-spatial",
        ),
        "default_prompt": "put the object on the target area",
        "default_input_path": str(_data_path("test_cases", "test_vla_case1", "libero", "main_view.png")),
        "default_call_kwargs": _openvla_oft_default_call_kwargs,
        "extra_variants": (
            {
                "variant_id": "libero-object",
                "label": "LIBERO Object",
                "status": "checkpoint_gpu_validated",
                "checkpoints": (
                    {
                        "role": "primary",
                        "uri": lambda: _hf_checkpoint_model_ref(
                            "hfd/moojink--openvla-7b-oft-finetuned-libero-object",
                            "moojink/openvla-7b-oft-finetuned-libero-object",
                        ),
                        "status": "checkpoint_gpu_validated",
                    },
                ),
                "call_kwargs": lambda: _openvla_oft_variant_call_kwargs(
                    unnorm_key="libero_object_no_noops",
                    task_suite_name="libero_object",
                ),
                "aliases": ("openvla-oft-libero-object",),
            },
            {
                "variant_id": "libero-goal",
                "label": "LIBERO Goal",
                "status": "checkpoint_gpu_validated",
                "checkpoints": (
                    {
                        "role": "primary",
                        "uri": lambda: _hf_checkpoint_model_ref(
                            "hfd/moojink--openvla-7b-oft-finetuned-libero-goal",
                            "moojink/openvla-7b-oft-finetuned-libero-goal",
                        ),
                        "status": "checkpoint_gpu_validated",
                    },
                ),
                "call_kwargs": lambda: _openvla_oft_variant_call_kwargs(
                    unnorm_key="libero_goal_no_noops",
                    task_suite_name="libero_goal",
                ),
                "aliases": ("openvla-oft-libero-goal",),
            },
            {
                "variant_id": "libero-10",
                "label": "LIBERO 10",
                "status": "checkpoint_gpu_validated",
                "checkpoints": (
                    {
                        "role": "primary",
                        "uri": lambda: _hf_checkpoint_model_ref(
                            "hfd/moojink--openvla-7b-oft-finetuned-libero-10",
                            "moojink/openvla-7b-oft-finetuned-libero-10",
                        ),
                        "status": "checkpoint_gpu_validated",
                    },
                ),
                "call_kwargs": lambda: _openvla_oft_variant_call_kwargs(
                    unnorm_key="libero_10_no_noops",
                    task_suite_name="libero_10",
                ),
                "aliases": ("openvla-oft-libero-10",),
            },
            {
                "variant_id": "libero-spatial-object-goal-10",
                "label": "LIBERO Combined",
                "status": "checkpoint_gpu_validated",
                "checkpoints": (
                    {
                        "role": "primary",
                        "uri": lambda: _hf_checkpoint_model_ref(
                            "hfd/moojink--openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
                            "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
                        ),
                        "status": "checkpoint_gpu_validated",
                    },
                ),
                "call_kwargs": lambda: _openvla_oft_variant_call_kwargs(
                    unnorm_key="libero_spatial_no_noops",
                    task_suite_name="libero_spatial",
                ),
                "aliases": ("openvla-oft-libero-spatial-object-goal-10",),
            },
        ),
        "default_interactions": ("robot_action", "action_chunk"),
        "call_params": OFFICIAL_ACTION_CALL_PARAMS,
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("openvla_oft", "openvla-oft-libero"),
        "tags": ("vla", "policy", "robot-action", "libero"),
        "notes": "All five declared LIBERO checkpoints completed real A100 Workspace inference with finite 8x7 action chunks.",
    },
    "smolvla": {
        "display_name": "SmolVLA LIBERO",
        "category": "Embodied Action",
        "summary": "In-tree SmolVLA flow-matching policy with the released LIBERO task checkpoint.",
        "default_model_ref": lambda: _hf_checkpoint_model_ref(
            "hfd/HuggingFaceVLA--smolvla_libero",
            "HuggingFaceVLA/smolvla_libero",
        ),
        "default_prompt": "put the object on the target area",
        "default_input_path": str(
            _data_path("test_cases", "test_vla_case1", "libero", "main_view.png")
        ),
        "default_call_kwargs": lambda: {
            "official_policy_observation": {
                "image": str(
                    _data_path("test_cases", "test_vla_case1", "libero", "main_view.png")
                ),
                "image2": str(
                    _data_path("test_cases", "test_vla_case1", "libero", "wrist_view.png")
                ),
                "state": [0.0] * 8,
            },
        },
        "default_interactions": ("robot_action", "action_chunk"),
        "call_params": OFFICIAL_ACTION_CALL_PARAMS,
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("smolvla-libero", "huggingfacevla/smolvla_libero"),
        "tags": ("vla", "policy", "robot-action", "flow-matching", "in-tree-runtime"),
        "notes": "The Workspace default uses the task-trained LIBERO checkpoint, two RGB views, and an 8-D state vector.",
    },
    "last-r1": {
        "display_name": "LaST-R1",
        "category": "Embodied Action",
        "summary": "Offline image-and-instruction inference with the libero_10 RL policy.",
        "default_model_ref": _last_r1_default_ref,
        "default_prompt": "Pick up the red block and place it in the bowl.",
        "default_load_kwargs": _last_r1_default_load_kwargs,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "tags": ("vla", "policy", "robot-action", "libero"),
        "env_hints": ("WORLDFOUNDRY_LAST_R1_SOURCE_DIR",),
        "notes": "The pinned official source is required. Offline action inference is verified; LIBERO task success is unmeasured.",
    },
    "cogact": {
        "display_name": "CogACT",
        "category": "Embodied Action",
        "default_model_ref": _cogact_default_ref,
        "default_prompt": "move sponge near apple",
        "default_input_path": str(_data_path("test_cases", "test_vla_image_case1", "init_frame.png")),
        "default_call_kwargs": _cogact_default_call_kwargs,
        "default_interactions": ("robot_action", "action_chunk"),
        "call_params": OFFICIAL_ACTION_CALL_PARAMS,
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("cogact-base", "cogact-small", "cogact-large"),
        "tags": ("vla", "policy", "robot-action", "diffusion-action"),
        "notes": "Independent CogACT action entry with an in-tree vendored VLA/DiT runtime. Real inference requires CogACT checkpoints plus gated meta-llama/Llama-2-7b-hf access.",
    },
    "db-cogact": {
        "display_name": "DB-CogACT",
        "category": "Embodied Action",
        "default_model_ref": _db_cogact_default_ref,
        "default_prompt": "put the object on the target area",
        "default_input_path": str(_data_path("test_cases", "test_vla_case1", "libero", "main_view.png")),
        "default_call_kwargs": _db_cogact_default_call_kwargs,
        "default_interactions": ("robot_action", "action_chunk"),
        "call_params": OFFICIAL_ACTION_CALL_PARAMS,
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("dexbotic", "db_cogact", "dexbotic-cogact"),
        "tags": ("vla", "policy", "robot-action", "dexbotic"),
        "notes": "Independent DB-CogACT action entry with an in-tree vendored Dexbotic policy runtime. Release demo status depends on checkpoint-backed action validation.",
    },
    "vlanext": {
        "display_name": "VLANeXt",
        "category": "Embodied Action",
        "default_model_ref": _vlanext_default_ref,
        "default_prompt": "put the object on the target area",
        "default_input_path": str(_data_path("test_cases", "test_vla_case1", "libero", "main_view.png")),
        "default_call_kwargs": _vlanext_default_call_kwargs,
        "default_interactions": ("robot_action", "action_chunk"),
        "call_params": OFFICIAL_ACTION_CALL_PARAMS,
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("vla-next", "vlanext-libero"),
        "tags": ("vla", "policy", "robot-action", "libero"),
        "notes": "Independent VLANeXt action entry with an in-tree vendored model/processor runtime. Release demo status depends on checkpoint-backed action validation.",
    },
    "molmobot": {
        "display_name": "MolmoBot",
        "category": "Embodied Action",
        "default_model_ref": _molmobot_default_ref,
        "default_prompt": "put the mug in the bowl",
        "default_input_path": str(_data_path("test_cases", "test_vla_case1", "droid", "exterior_image_1_left.png")),
        "default_call_kwargs": _molmobot_default_call_kwargs,
        "default_interactions": ("robot_action", "action_chunk"),
        "call_params": OFFICIAL_ACTION_CALL_PARAMS,
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("molmob0t", "molmobot-droid", "molmobot-pi0"),
        "tags": ("vla", "policy", "robot-action", "molmo"),
        "notes": "Independent MolmoBot action entry with a flattened in-tree inference-only policy runtime. Pi0 remains dependency-gated; release demo status depends on checkpoint-backed action validation.",
    },
    "mme-vla": {
        "display_name": "MME-VLA",
        "category": "Embodied Action",
        "default_model_ref": _mme_vla_default_ref,
        "default_prompt": "put the object on the target area",
        "default_input_path": str(_data_path("test_cases", "test_vla_case1", "libero", "main_view.png")),
        "default_call_kwargs": _mme_vla_default_call_kwargs,
        "default_interactions": ("robot_action", "action_chunk"),
        "call_params": OFFICIAL_ACTION_CALL_PARAMS,
        "load_params": OFFICIAL_ACTION_LOAD_PARAMS,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("mme_vla", "robomme-policy-learning", "perceptual-framesamp-modul"),
        "tags": ("vla", "policy", "robot-action", "robomme"),
        "notes": "Independent MME-VLA action entry with an in-tree vendored MME-VLA/OpenPI policy runtime. Release demo status depends on checkpoint-backed action validation.",
    },
    "openvla": {
        "display_name": "OpenVLA",
        "category": "Embodied Action",
        "default_model_ref": _openvla_default_ref,
        "default_prompt": "put the object on the target area",
        "default_interactions": ("robot_action",),
        "default_call_kwargs": _openvla_default_call_kwargs,
        "call_params": (
            "prompt",
            "images",
            "interactions",
            "output_path",
            "fps",
            "checkpoint_dir",
            "ckpt_path",
            "pretrained_model_path",
            "unnorm_key",
            "torch_dtype",
            "attn_implementation",
            "use_cache",
            "plan_only",
            "openvla_observation",
            "action_space",
            "policy_controls",
        ),
        "aliases": ("openvla-7b",),
        "tags": ("vla", "policy", "robot-action"),
        "notes": "Structured VLA policy pipeline. It emits action_trace artifacts, not generated video.",
    },
    "openpi": {
        "display_name": "OpenPI",
        "category": "Embodied Action",
        "default_model_ref": _openpi_default_ref,
        "default_prompt": "complete the manipulation instruction",
        "default_interactions": ("robot_action",),
        "default_call_kwargs": _openpi_default_call_kwargs,
        "call_params": (
            "prompt",
            "images",
            "interactions",
            "output_path",
            "fps",
            "checkpoint_dir",
            "config_name",
            "pytorch_device",
            "seed",
            "openpi_observation",
            "action_space",
            "policy_controls",
        ),
        "aliases": ("pi0", "pi05"),
        "tags": ("vla", "policy", "robot-action"),
        "notes": "Structured pi0/pi0.5 policy pipeline. The default Studio demo mirrors the official pi05_libero checkpoint/config path.",
    },
    "giga-brain-0": {
        "display_name": "GigaBrain-0",
        "category": "Embodied Action",
        "default_prompt": "put the object on the target area",
        "default_interactions": ("robot_action", "action_chunk"),
        "default_call_kwargs": _giga_brain_0_default_call_kwargs,
        "aliases": ("giga_brain_0", "gigabrain0", "giga-brain-0.1"),
        "tags": ("vla", "policy", "world-model-powered", "robot-action"),
        "notes": "GigaBrain-0 is a world-model-powered VLA policy for multi-view RGB/RGBD action traces. Official execution requires staged norm stats, a LeRobot dataset, and the upstream giga-models stack.",
    },
    "gr00t": {
        "display_name": "GR00T N1.7",
        "category": "Embodied Action",
        "default_model_ref": _gr00t_default_ref,
        "default_prompt": "execute the humanoid manipulation skill",
        "default_interactions": ("robot_action",),
        "aliases": ("groot", "gr00t-n1", "gr00t-n1.7"),
        "tags": ("vla", "policy", "humanoid", "in-tree-runtime"),
        "notes": "In-tree GR00T N1.7 runtime with explicit local Qwen3-VL classes; checkpoint Python is never executed.",
    },
    "molmoact2": {
        "display_name": "MolmoAct2",
        "category": "Embodied Action",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "hfd/allenai--MolmoAct2-DROID",
            fallback="allenai/MolmoAct2-DROID",
        ),
        "default_prompt": "put the object on the target area",
        "default_interactions": ("robot_action", "action_chunk"),
        "default_call_kwargs": _molmoact2_default_call_kwargs,
        "extra_variants": (
            {
                "variant_id": "yam",
                "label": "Bimanual YAM",
                "status": "configured",
                "checkpoints": (
                    {
                        "role": "primary",
                        "uri": lambda: _checkpoint_model_ref(
                            "hfd/allenai--MolmoAct2-BimanualYAM",
                            fallback="allenai/MolmoAct2-BimanualYAM",
                        ),
                        "status": "configured",
                    },
                ),
                "call_kwargs": _molmoact2_yam_call_kwargs,
                "aliases": ("bimanual-yam", "molmoact2-bimanual-yam", "molmoact2-yam"),
                "notes": (
                    "Official YAM server schema: top_cam, left_cam, right_cam, 14-D state, yam_dual_molmoact2 normalization.",
                ),
            },
            {
                "variant_id": "so100",
                "label": "SO100/SO101",
                "status": "configured",
                "checkpoints": (
                    {
                        "role": "primary",
                        "uri": lambda: _checkpoint_model_ref(
                            "hfd/allenai--MolmoAct2-SO100_101",
                            fallback="allenai/MolmoAct2-SO100_101",
                        ),
                        "status": "configured",
                    },
                ),
                "call_kwargs": _molmoact2_so100_call_kwargs,
                "aliases": ("so101", "so100_101", "molmoact2-so100-101"),
                "notes": (
                    "Official SO100/SO101 model-card schema: top_rgb, side_rgb, 6-D state, so100_so101_molmoact2 normalization.",
                ),
            },
            {
                "variant_id": "libero",
                "label": "LIBERO",
                "status": "configured",
                "checkpoints": (
                    {
                        "role": "primary",
                        "uri": lambda: _checkpoint_model_ref(
                            "hfd/allenai--MolmoAct2-LIBERO",
                            fallback="allenai/MolmoAct2-LIBERO",
                        ),
                        "status": "configured",
                    },
                ),
                "call_kwargs": _molmoact2_libero_call_kwargs,
                "aliases": ("molmoact2-libero", "libero-10"),
                "notes": (
                    "Official LIBERO model-card schema: agentview_rgb, wrist_rgb, 8-D state, libero normalization.",
                ),
            },
            {
                "variant_id": "think-libero",
                "label": "Think LIBERO",
                "status": "configured",
                "checkpoints": (
                    {
                        "role": "primary",
                        "uri": lambda: _checkpoint_model_ref(
                            "hfd/allenai--MolmoAct2-Think-LIBERO",
                            fallback="allenai/MolmoAct2-Think-LIBERO",
                        ),
                        "status": "configured",
                    },
                ),
                "call_kwargs": _molmoact2_think_libero_call_kwargs,
                "aliases": ("think_libero", "molmoact2-think-libero", "depth-libero"),
                "notes": (
                    "Official Think-LIBERO model-card schema: agentview_rgb, wrist_rgb, 8-D state, libero normalization, depth reasoning and adaptive depth enabled.",
                ),
            },
        ),
        "aliases": ("molmo-act2", "molmoact", "molmoact2-droid", "molmoact2-yam"),
        "tags": ("vla", "policy", "robot-action", "action-reasoning"),
        "notes": "MolmoAct2 DROID/YAM action-reasoning VLA. The default Studio demo mirrors the official DROID server schema: external_cam, wrist_cam, 8-D Franka state, franka_droid normalization.",
    },
    "dreamzero": {
        "display_name": "DreamZero",
        "category": "Embodied Action",
        "default_prompt": "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan",
        "default_interactions": ("world_action", "robot_action"),
        "default_call_kwargs": {
            "plan_only": True,
            "variant": "droid",
            "dreamzero_server_host": "",
            "dreamzero_server_port": 8000,
            "num_chunks": 15,
            "use_zero_images": True,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "checkpoint_dir",
            "variant",
            "dreamzero_server_host",
            "dreamzero_server_port",
            "server_host",
            "server_port",
            "plan_only",
            "debug_video_dir",
            "num_chunks",
            "session_id",
            "use_zero_images",
        ),
        "aliases": ("dreamzero-droid", "dreamzero-agibot"),
        "tags": ("wam", "world-action", "robot-action", "video-prediction"),
        "notes": "NVIDIA GEAR DreamZero world-action policy. Official runtime uses a distributed RoboArena WebSocket server and emits action traces with server-side video predictions.",
    },
    "starvla": {
        "display_name": "StarVLA",
        "category": "Embodied Action",
        "default_model_ref": _starvla_default_ref,
        "default_prompt": "follow the embodied instruction with action-conditioned state updates",
        "default_interactions": ("robot_action", "world_action"),
        "default_call_kwargs": _starvla_default_call_kwargs,
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "checkpoint_dir",
            "base_vlm",
            "source_repo_dir",
            "enable_official_runtime",
            "attn_implementation",
            "state",
            "track",
            "variant_id",
        ),
        "aliases": ("star-vla", "wm4a"),
        "tags": ("vla", "wam", "policy", "world-action"),
        "notes": "StarVLA spans VLA and WAM variants. The default Studio demo uses the official Qwen3-VL-OFT-LIBERO-4in1 checkpoint, Qwen3-VL-4B base VLM, and local starVLA official source checkout.",
    },
    "lingbot-va": {
        "display_name": "LingBot-VA",
        "category": "Embodied Action",
        "default_prompt": "put both the alphabet soup and the tomato sauce in the basket",
        "default_interactions": ("robot_action", "video_action"),
        "default_call_kwargs": _lingbot_va_default_call_kwargs,
        "aliases": ("lingbot_va", "lingbot-va-base", "lingbot-va-posttrain-robotwin", "lingbot-va-posttrain-libero-long"),
        "tags": ("va", "vam", "wam", "robot-action", "video-action"),
        "notes": "LingBot-VA is a video-action world model for LIBERO/RoboTwin-style action traces; official execution remains checkpoint and CUDA-12.6 profile gated.",
    },
    "lapa": {
        "display_name": "LAPA",
        "category": "Visual Action",
        "default_model_ref": _lapa_default_ref,
        "default_prompt": "infer latent action tokens from the video context",
        "default_interactions": ("latent_action_tokens",),
        "default_call_kwargs": _lapa_default_call_kwargs,
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "checkpoint_dir",
            "dtype",
            "image_size",
            "mesh_dim",
            "seed",
            "tokens_per_delta",
            "lapa_observation",
        ),
        "aliases": ("lapa-7b-openx",),
        "tags": ("va", "vam", "latent-action"),
        "notes": "Visual-action model profile. It consumes video context but is not categorized as video generation.",
    },
    "leworldmodel": {
        "display_name": "LeWorldModel",
        "category": "Video Generation",
        "summary": "Official PushT environment evaluation through the in-tree stable-worldmodel wrapper.",
        "default_task_type": "pusht-eval",
        "default_load_kwargs": _leworldmodel_default_load_kwargs,
        "default_call_kwargs": {"plan_only": False, "timeout_seconds": 21600},
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "plan_only",
            "timeout_seconds",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "config_dir",
            "config_name",
            "policy",
            "cache_dir",
            "dataset_name",
            "num_eval",
            "eval_budget",
            "goal_offset_steps",
            "img_size",
            "seed",
            "lazy",
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("le-wm", "lewm"),
        "tags": ("world-model", "pusht", "official-runtime", "in-tree-runtime"),
        "notes": (
            "The default demo uses a locally staged PushT trajectory and the official random policy; "
            "learned-policy evaluation requires a compatible policy checkpoint."
        ),
    },
    "egowm": {
        "display_name": "EgoWM",
        "category": "Video Generation",
        "summary": "Official action/state-conditioned SVD egocentric navigation video generation.",
        "default_model_ref": lambda: str(checkpoint_root_path("anuragba--egowm", "svd_25dof_nav.pth")),
        "default_task_type": "navigation-video",
        "default_load_kwargs": _egowm_default_load_kwargs,
        "default_call_kwargs": {
            "variant": "25dof",
            "num_frames": 8,
            "num_inference_steps": 25,
            "fps": 7,
            "seed": 42,
            "height": 512,
            "width": 512,
            "motion_bucket_id": 180,
            "plan_only": False,
            "timeout_seconds": 21600,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "image_path",
            "input_image",
            "num_frames",
            "num_inference_steps",
            "seed",
            "height",
            "width",
            "motion_bucket_id",
            "action_scale",
            "conditions_path",
            "egowm_conditions_path",
            "actions_path",
            "egowm_actions_path",
            "smoke_synthetic",
            "variant",
            "egowm_variant",
            "plan_only",
            "timeout_seconds",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "checkpoint_path",
            "egowm_checkpoint_path",
            "base_model_dir",
            "svd_base_model_dir",
            "image_path",
            "input_image",
            "seed",
            "lazy",
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("ego-wm", "egowm-svd-25dof"),
        "tags": ("world-model", "navigation-video", "action-conditioned", "official-runtime"),
        "notes": (
            "Runs the official 25-DoF or 3-DoF navigation checkpoint with an explicit local SVD base model. "
            "Use paired conditions_path for 25-DoF, or paired normalized actions_path for 3-DoF. "
            "A constant 3-DoF action_scale is a smoke control, not a paired motion test."
        ),
    },
    "hma": {
        "display_name": "HMA",
        "category": "Video Generation",
        "summary": "Official STMAR continuous action-conditioned robot video generation.",
        "default_model_ref": lambda: str(checkpoint_root_path("liruiw--hma-base-cont")),
        "default_input_path": lambda: str(
            official_runtime_repo_path("HMA") / "assets" / "langtable_prompt" / "frame_00.png"
        ),
        "default_task_type": "robot-action-video",
        "default_load_kwargs": _hma_default_load_kwargs,
        "default_call_kwargs": {
            "generated_frames": 6,
            "prompt_horizon": 3,
            "maskgit_steps": 2,
            "direction": "right",
            "action_scale": 0.05,
            "fps": 2,
            "seed": 42,
            "plan_only": False,
            "timeout_seconds": 21600,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "image_path",
            "input_image",
            "trajectory_path",
            "trajectory_json",
            "generated_frames",
            "num_frames",
            "prompt_horizon",
            "maskgit_steps",
            "direction",
            "action_scale",
            "seed",
            "plan_only",
            "timeout_seconds",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "checkpoint_dir",
            "checkpoint_path",
            "hma_checkpoint_dir",
            "hma_magvit_checkpoint",
            "base_model_dir",
            "svd_base_model_dir",
            "image_path",
            "input_image",
            "lazy",
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("hma-magvit-362m", "hma-continuous"),
        "tags": ("world-model", "robot-video", "action-conditioned", "official-runtime"),
        "notes": (
            "The default repeats one LangTable image and one raw two-dimensional action for smoke testing. "
            "For a paired rollout, pass trajectory_path with prompt_frames and raw_language_table_delta_yx actions; "
            "the official backbone normalizes actions internally. Continuous uses SVD VAE; discrete uses MAGVIT2."
        ),
    },
    "giga-world-0": {
        "display_name": "GigaWorld-0",
        "category": "Video Generation",
        "summary": "Released GR1 robot-world image-to-video generation with local T5 and Wan components.",
        "default_model_ref": lambda: str(
            checkpoint_root_path(
                "open-gigaai--GigaWorld-0-Video-GR1-2b", "transformer"
            )
        ),
        "default_input_path": str(
            _data_path("test_cases", "test_vla_case1", "droid", "exterior_image_1_left.png")
        ),
        "default_prompt": "A robot arm moves toward the objects on the workbench.",
        "default_task_type": "robot-image-to-video",
        "default_load_kwargs": _giga_world_0_default_load_kwargs,
        "default_call_kwargs": {
            "attention_backend": "natten",
            "height": 480,
            "width": 640,
            "num_frames": 61,
            "num_inference_steps": 30,
            "guidance_scale": 1.0,
            "fps": 16,
            "seed": 6666,
            "max_text_length": 128,
            "mixed_precision": "bf16",
            "plan_only": False,
            "timeout_seconds": 21600,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "image_path",
            "input_image",
            "attention_backend",
            "height",
            "width",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "max_text_length",
            "mixed_precision",
            "plan_only",
            "timeout_seconds",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "transformer_model_dir",
            "transformer_model_path",
            "checkpoint_path",
            "checkpoint",
            "text_encoder_model_dir",
            "text_encoder_model_path",
            "vae_model_dir",
            "vae_model_path",
            "image_path",
            "input_image",
            "lazy",
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("giga_world_0", "gigaworld-0", "gigaworld0"),
        "tags": ("world-model", "robot-video", "image-to-video", "official-runtime"),
        "notes": (
            "The default uses the released 480x640, 61-frame, 30-step NATTEN sampling settings. "
            "Use a checkpoint-matched GR1 observation for dataset parity."
        ),
    },
    "uniworld-view": {
        "display_name": "UniWorld-View",
        "category": "Video Generation",
        "summary": "Official UniView novel-view synthesis from one image or a monocular video.",
        "default_model_ref": lambda: str(
            official_runtime_repo_path("UniWorld-View", specific_env="UNIWORLD_VIEW_REPO")
        ),
        "default_input_path": lambda: str(
            official_runtime_repo_path("UniWorld-View", specific_env="UNIWORLD_VIEW_REPO")
            / "test" / "images" / "fruit.jpg"
        ),
        "default_prompt": "",
        "default_task_type": "image-to-video",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_load_kwargs": lambda: {
            "repo_root": str(official_runtime_repo_path("UniWorld-View", specific_env="UNIWORLD_VIEW_REPO")),
        },
        "default_call_kwargs": {
            "mode": "single_view",
            "geometry_backend": "stream3r",
            "render_method": "hybrid",
            "height": 480,
            "width": 832,
            "keep_aspect_ratio": True,
            "num_frames": 81,
            "num_inference_steps": 8,
            "guidance_scale": 4.0,
            "fps": 16,
            "seed": 43,
            "d_phi": 50.0,
            "timeout_seconds": 21600,
        },
        "input_params": ("image_path", "video_path"),
        "call_params": (
            "prompt", "images", "video", "image_path", "video_path", "mode",
            "geometry_backend", "render_method", "mosca_ws", "height", "width", "keep_aspect_ratio",
            "num_frames", "num_inference_steps", "guidance_scale", "fps", "seed",
            "stride", "traj_type", "d_phi", "d_theta", "x_offset", "y_offset",
            "z_offset", "radius_scale", "low_gpu_memory_mode", "output_path",
            "timeout_seconds", "plan_only", "return_dict",
        ),
        "load_params": (
            "repo_root", "python_executable", "checkpoint_root", "transformer_path",
            "model_name", "blip_path", "moge_path", "sam2_checkpoint",
            "segnet_path", "stream3r_path", "lora_path", "device",
        ),
        "tags": ("novel-view", "camera-control", "image-to-video", "video-to-video", "official-runtime"),
    },
    "genie-envisioner": {
        "display_name": "Genie Envisioner",
        "category": "Video Generation",
        "summary": "Official GE-base multiview robot world-video generation with local LTX components.",
        "default_model_ref": lambda: str(
            checkpoint_root_path(
                "agibot-world--Genie-Envisioner", "GE_base_fast_v0.1.safetensors"
            )
        ),
        "default_input_path": str(
            _data_path("test_cases", "test_vla_case1", "aloha", "observation_images_cam_high.png")
        ),
        "default_prompt": "Move the robot gripper toward the objects on the workbench.",
        "default_task_type": "robot-multiview-video",
        "default_load_kwargs": _genie_envisioner_default_load_kwargs,
        "default_call_kwargs": {
            "input_mode": "explicit-three-view",
            "input_views": [
                str(_data_path("test_cases", "test_vla_case1", "aloha", name))
                for name in (
                    "observation_images_cam_high.png",
                    "observation_images_cam_left_wrist.png",
                    "observation_images_cam_right_wrist.png",
                )
            ],
            "height": 192,
            "width": 256,
            "n_previous": 4,
            "latent_chunk": 2,
            "num_inference_steps": 5,
            "guidance_scale": 1.0,
            "decode_timestep": 0.03,
            "decode_noise_scale": 0.025,
            "fps": 8,
            "seed": 42,
            "mixed_precision": "bf16",
            "plan_only": False,
            "timeout_seconds": 21600,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "image_path",
            "input_image",
            "input_views",
            "view_images",
            "input_mode",
            "height",
            "width",
            "n_previous",
            "latent_chunk",
            "num_inference_steps",
            "guidance_scale",
            "decode_timestep",
            "decode_noise_scale",
            "seed",
            "mixed_precision",
            "plan_only",
            "timeout_seconds",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "checkpoint_path",
            "checkpoint",
            "ckpt_path",
            "genie_checkpoint_path",
            "base_model_dir",
            "ltx_base_model_dir",
            "image_path",
            "input_image",
            "lazy",
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("genie_envisioner", "genieenvisioner"),
        "tags": ("world-model", "robot-video", "multiview", "official-runtime"),
        "notes": (
            "For official GE-base history, pass input_views as three directories in head, left-wrist, "
            "right-wrist order, each with 0.png through 3.png. The default three ALOHA still "
            "images repeat across four history slots and are smoke inputs only. "
            "Use a checkpoint-matched AgiBot or LeRobot trajectory for action-quality parity."
        ),
    },
    "ctrl-world": {
        "display_name": "Ctrl-World",
        "category": "Video Generation",
        "summary": "Official DROID action-conditioned three-camera robot world-video generation.",
        "default_model_ref": lambda: str(
            checkpoint_root_path("yjguo--Ctrl-World", "checkpoint-10000.pt")
        ),
        "default_input_path": str(
            _data_path("test_cases", "test_vla_case1", "aloha", "observation_images_cam_high.png")
        ),
        "default_prompt": "Move the robot gripper toward the objects on the workbench.",
        "default_task_type": "robot-action-video",
        "default_load_kwargs": _ctrl_world_default_load_kwargs,
        "default_call_kwargs": {
            "input_mode": "explicit-three-view",
            "input_views": [
                str(_data_path("test_cases", "test_vla_case1", "aloha", name))
                for name in (
                    "observation_images_cam_high.png",
                    "observation_images_cam_left_wrist.png",
                    "observation_images_cam_right_wrist.png",
                )
            ],
            "action_mode": "absolute-pose",
            "action_direction": "right",
            "action_distance": 0.08,
            "height": 192,
            "width": 320,
            "num_frames": 5,
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "motion_bucket_id": 127,
            "fps": 4,
            "seed": 42,
            "mixed_precision": "bf16",
            "plan_only": False,
            "timeout_seconds": 21600,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "image_path",
            "input_image",
            "input_views",
            "view_images",
            "input_mode",
            "action_mode",
            "initial_pose",
            "action_direction",
            "action_distance",
            "action_scale",
            "height",
            "width",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "motion_bucket_id",
            "seed",
            "mixed_precision",
            "plan_only",
            "timeout_seconds",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "checkpoint_path",
            "checkpoint",
            "ckpt_path",
            "ctrl_world_checkpoint_path",
            "base_model_dir",
            "svd_model_path",
            "clip_model_dir",
            "clip_model_path",
            "image_path",
            "input_image",
            "lazy",
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("ctrl_world", "ctrlworld"),
        "tags": ("world-model", "robot-video", "action-conditioned", "official-runtime"),
        "notes": (
            "Physical action generation requires a seven-value DROID Cartesian initial_pose; the default ALOHA images are demonstration views only. "
            "Use synchronized DROID cameras and the corresponding physical pose for action-quality evaluation. "
            "The old zero-pose conditioning is available only with action_mode='synthetic-zero-smoke'."
        ),
    },
    "tesseract": {
        "display_name": "TesserAct",
        "category": "Video Generation",
        "summary": "Official RGB-depth-normal robot world-video generation with local CogVideoX components.",
        "default_model_ref": lambda: str(
            checkpoint_root_path("anyeZHY--tesseract", "tesseract_v01e_rgbdn_sft")
        ),
        "default_input_path": str(
            _data_path("test_cases", "test_vla_case1", "droid", "exterior_image_1_left.png")
        ),
        "default_prompt": "The robot moves its gripper toward the object on the table.",
        "default_task_type": "robot-action-video",
        "default_load_kwargs": _tesseract_default_load_kwargs,
        "default_call_kwargs": {
            "geometry_mode": "synthetic-gradient",
            "height": 256,
            "width": 320,
            "num_frames": 49,
            "num_inference_steps": 4,
            "guidance_scale": 7.5,
            "image_guidance_scale": 1.5,
            "use_dynamic_cfg": False,
            "fps": 8,
            "seed": 23,
            "mixed_precision": "bf16",
            "memory_efficient": True,
            "plan_only": False,
            "timeout_seconds": 21600,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "image_path",
            "input_image",
            "depth_path",
            "normal_path",
            "geometry_mode",
            "height",
            "width",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "image_guidance_scale",
            "use_dynamic_cfg",
            "seed",
            "mixed_precision",
            "memory_efficient",
            "plan_only",
            "timeout_seconds",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "checkpoint_dir",
            "checkpoint_path",
            "tesseract_checkpoint_dir",
            "weights_path",
            "base_model_dir",
            "cogvideox_base_model_dir",
            "image_path",
            "input_image",
            "lazy",
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("tesser-act", "tesseract-rgbdn"),
        "tags": ("world-model", "robot-video", "rgb-depth-normal", "official-runtime"),
        "notes": (
            "The default synthetic-gradient geometry is a deterministic checkpoint smoke aid, not an official depth/normal estimate. "
            "Supply depth_path and normal_path with geometry_mode=files for released-model input parity."
        ),
    },
    "shotstream": {
        "display_name": "ShotStream",
        "category": "Video Generation",
        "summary": "Official 4-step causal multi-shot video generation through a staged ShotStream checkout.",
        "default_model_ref": lambda: str(checkpoint_root_path("KlingTeam--ShotStream")),
        "default_prompt": (
            "A cinematic science-fiction scene in a bioluminescent forest at night. An astronaut in a white "
            "exploration suit walks between glowing blue plants while soft mist drifts through the trees."
        ),
        "default_task_type": "causal-multi-shot-video",
        "default_load_kwargs": _shotstream_default_load_kwargs,
        "default_call_kwargs": {"fps": 16, "seed": 42, "plan_only": False, "timeout_seconds": 21600},
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "data_path",
            "input_csv",
            "seed",
            "plan_only",
            "timeout_seconds",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "checkpoint_dir",
            "shotstream_checkpoint_dir",
            "wan_model_dir",
            "base_model_dir",
            "data_path",
            "input_csv",
            "seed",
            "lazy",
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "aliases": ("shot-stream", "interactive-video-generation"),
        "tags": ("world-model", "causal-video", "multi-shot", "official-runtime"),
        "notes": (
            "Runs the official ShotStream checkpoint on Wan2.1-T2V-1.3B. A plain prompt produces one 81-frame "
            "shot; pass an official CSV through data_path/input_csv for multi-shot generation."
        ),
    },
    "octo": {
        "display_name": "Octo",
        "category": "Embodied Action",
        "default_model_ref": _octo_default_ref,
        "default_prompt": "complete the manipulation instruction",
        "default_interactions": ("robot_action",),
        "default_call_kwargs": _octo_default_call_kwargs,
        "call_params": (
            "prompt",
            "images",
            "interactions",
            "output_path",
            "fps",
            "checkpoint_dir",
            "variant",
            "dataset_key",
            "image_key",
            "image_size",
            "jax_platform",
            "seed",
            "octo_observation",
            "action_space",
            "policy_controls",
        ),
        "aliases": ("octo-base", "octo-base-1.5", "octo-small-1.5"),
        "tags": ("vla", "policy", "generalist", "robot-action"),
        "notes": "Generalist robot policy pipeline. It emits action_trace artifacts for embodied rollouts.",
    },
    "rt-1": {
        "display_name": "RT-1",
        "category": "Embodied Action",
        "default_prompt": "follow the language-conditioned robot instruction",
        "default_interactions": ("robot_action",),
        "default_call_kwargs": {"plan_only": False},
        "call_params": (
            "prompt",
            "images",
            "interactions",
            "output_path",
            "fps",
            "checkpoint_dir",
            "language_embedding",
            "rt1_observation",
            "action_space",
            "policy_controls",
            "plan_only",
        ),
        "aliases": ("rt-1", "robotics-transformer"),
        "tags": ("vla", "policy", "robotics-transformer"),
        "notes": "Robotics Transformer policy pipeline. Action inference requires a checkpoint-compatible 512-D language embedding; prompt text alone is not encoded.",
    },
    "diffusion-policy": {
        "display_name": "Diffusion Policy",
        "category": "Embodied Action",
        "default_prompt": "execute the visuomotor policy rollout",
        "default_interactions": ("robot_action",),
        "default_call_kwargs": _diffusion_policy_default_call_kwargs,
        "call_params": (
            "prompt",
            "interactions",
            "output_path",
            "checkpoint_path",
            "observation",
            "action_space",
            "device",
        ),
        "aliases": ("diffusion_policy", "dp"),
        "tags": ("visuomotor", "action-diffusion", "policy"),
        "notes": "Diffusion-based action policy profile. It is not a diffusion video-generation model.",
    },
    "act": {
        "display_name": "ACT",
        "category": "Embodied Action",
        "default_prompt": "produce the next robot action chunk",
        "default_interactions": ("action_chunk",),
        "default_call_kwargs": {
            "camera_names": ["head_cam", "left_cam", "right_cam"],
            "state_dim": 14,
            "chunk_size": 100,
            "temporal_agg": False,
            "plan_only": True,
        },
        "call_params": (
            "prompt",
            "images",
            "interactions",
            "output_path",
            "fps",
            "checkpoint_dir",
            "checkpoint_path",
            "act_observation",
            "camera_names",
            "state_dim",
            "chunk_size",
            "temporal_agg",
            "plan_only",
        ),
        "aliases": ("action-chunking-transformer",),
        "tags": ("action-chunking", "policy", "imitation-learning"),
        "notes": "Action Chunking Transformer policy pipeline. Defaults to plan-only unless a trained policy checkpoint and dataset stats are provided.",
    },
    "roboflamingo": {
        "display_name": "RoboFlamingo",
        "category": "Embodied Action",
        "default_prompt": "follow the vision-language manipulation instruction",
        "default_interactions": ("robot_action",),
        "aliases": ("robo-flamingo",),
        "tags": ("vla", "vlm-policy", "robot-action"),
        "notes": "VLM/VLA policy pipeline for robot action traces.",
    },
    "being-h05": {
        "display_name": "Being-H05",
        "category": "Embodied Action",
        "default_prompt": "complete the robot manipulation instruction",
        "default_interactions": ("robot_action",),
        "aliases": ("beingh05", "being-h-05", "being-h05-2b"),
        "tags": ("vla", "policy", "robot-action"),
        "notes": "Being-H05 is a VLA policy runtime that emits action traces rather than generated video.",
    },
    "diamond": {
        "display_name": "Diamond",
        "default_interactions": ("forward_left",),
        "supports_stream": True,
        "aliases": ("wmfactory-diamond",),
        "tags": ("interactive-world", "navigation", "game-video", "wmfactory"),
        "notes": "WMFactory-compatible Diamond id. Uses the in-tree runtime-manifest route when launched from WorldFoundry.",
    },
    "mira": {
        "display_name": "MIRA",
        "category": "Video Generation",
        "default_model_ref": lambda: str(checkpoint_root_path("mira")),
        "default_load_kwargs": {
            "dataset_path": str(local_data_root_path() / "rocket-science" / "test"),
        },
        "default_prompt": (
            "A cat and a dog baking a cake together in a cozy sunlit kitchen. The cat carefully measures flour "
            "while the dog stirs the batter with a wooden spoon, cinematic lighting, detailed natural motion."
        ),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "default_task_type": "rocket-science-rollout",
        "call_params": (
            "prompt",
            "interactions",
            "output_path",
            "fps",
            "plan_only",
            "timeout_seconds",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "checkpoint_path",
            "dataset_path",
            "actions",
            "actions_file",
            "seed",
            "clip_index",
            "n_context_frames",
            "num_unrolled_frames",
            "n_diffusion_steps",
            "schedule_type",
            "noise_level",
            "compile",
            "overlay_actions",
            "generated_only",
        ),
        "aliases": ("mira-wm", "multiplayer-interactive-world-model"),
        "tags": ("world-model", "rocket-league", "multiplayer", "action-conditioned"),
        "notes": "Inference-only MIRA Workspace entry backed by the vendored upstream checkpoint and Rocket Science loaders.",
    },
    "oasis-500m": {
        "display_name": "Oasis 500M",
        "default_model_ref": _oasis_500m_default_ref,
        "default_input_path": lambda: _oasis_500m_default_load_kwargs()["prompt_path"],
        "default_load_kwargs": _oasis_500m_default_load_kwargs,
        "default_call_kwargs": {
            "num_frames": 32,
            "fps": 20,
            "ddim_steps": 10,
            "return_dict": True,
        },
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "num_frames",
            "ddim_steps",
            "prompt_path",
            "actions_path",
            "timeout_seconds",
            "plan_only",
        ),
        "load_params": (
            "model_path",
            "pretrained_model_path",
            "required_components",
            "device",
            "model_id",
            "checkpoint_path",
            "oasis_ckpt",
            "vae_ckpt",
            "vae_checkpoint_path",
            "prompt_path",
            "actions_path",
        ),
        "default_interactions": ("forward",),
        "supports_stream": True,
        "aliases": ("open-oasis", "openoasis", "open_oasis"),
        "tags": ("interactive-world", "navigation", "minecraft", "wmfactory"),
        "notes": "WMFactory Open-Oasis compatibility id; WorldFoundry canonical model id is oasis-500m.",
    },
    "vid2world": {
        "display_name": "Vid2World",
        "default_interactions": ("forward",),
        "supports_stream": True,
        "aliases": ("vid-2-world",),
        "tags": ("interactive-world", "navigation", "game-video", "wmfactory"),
        "notes": "WMFactory-compatible Vid2World id. Uses the in-tree runtime-manifest route when launched from WorldFoundry.",
    },
    "mineworld": {
        "display_name": "MineWorld",
        "default_interactions": ("forward",),
        "supports_stream": True,
        "aliases": ("mine-world",),
        "tags": ("interactive-world", "navigation", "minecraft", "wmfactory"),
        "notes": "WMFactory-compatible MineWorld id. Uses the in-tree runtime-manifest route when launched from WorldFoundry.",
    },
    "causal-rcm": {
        "display_name": "Causal-rCM",
        "default_model_ref": lambda: str(checkpoint_root_path("rcm")),
        "default_load_kwargs": _causal_rcm_default_load_kwargs,
        "default_call_kwargs": _causal_rcm_default_call_kwargs,
        "default_prompt": "A cinematic shot of a snowy mountain at sunrise",
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "default_task_type": "autoregressive-video-generation",
        "call_params": (
            "prompt",
            "negative_prompt",
            "images",
            "output_path",
            "fps",
            "dit_path",
            "checkpoint_path",
            "vae_path",
            "text_encoder_path",
            "image_path",
            "model_size",
            "distilled",
            "first_chunk_t",
            "chunk_t",
            "num_steps",
            "steps_per_chunk",
            "mid_t",
            "mid_t_schedules",
            "sigma_max",
            "guidance_scale",
            "timestep_shift",
            "num_frames",
            "num_samples",
            "resolution",
            "aspect_ratio",
            "adaptive_resolution",
            "context_from_last_step",
            "context_from_last_step_start_chunk",
            "kv_cache_policy",
            "kv_cache_window_blocks",
            "kv_cache_sink_blocks",
            "warmup_iters",
            "num_runs",
            "seed",
            "plan_only",
            "timeout_seconds",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "dit_path",
            "checkpoint_path",
            "checkpoint_dir",
            "vae_path",
            "text_encoder_path",
        ),
        "aliases": ("rcm", "causal_rcm", "rcm-causal"),
        "tags": ("interactive-world", "streaming", "autoregressive-video", "distillation", "wan"),
        "notes": (
            "Vendored NVlabs Causal-rCM streaming runtime (Apache-2.0). Block-causal Wan2.1 distilled to "
            "1-4 steps per chunk; chunk_t selects the streaming schedule. Requires a distilled DiT "
            "checkpoint plus the Wan2.1 VAE and umT5 text encoder."
        ),
    },
    "open-dreamer": {
        "display_name": "Open Dreamer",
        "default_model_ref": lambda: str(checkpoint_root_path("open-dreamer")),
        "default_load_kwargs": _open_dreamer_default_load_kwargs,
        "default_call_kwargs": _open_dreamer_default_call_kwargs,
        "default_prompt": "",
        "default_interactions": ("forward", "forward_camera_l", "forward", "forward_camera_r"),
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "default_task_type": "minecraft-world-rollout",
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "input_mp4",
            "actions_path",
            "checkpoint_path",
            "context_frames",
            "horizon",
            "num_steps",
            "decode_chunk_size",
            "seed",
            "parallel_strategy",
            "use_ema",
            "no_kv_cache",
            "camera_step_degrees",
            "python_executable",
            "plan_only",
            "timeout_seconds",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "checkpoint_path",
            "input_mp4",
            "actions_path",
            "python_executable",
        ),
        "aliases": ("open_dreamer", "opendreamer", "dreamer-4", "dreamer4"),
        "tags": ("interactive-world", "minecraft", "action-conditioned", "world-model", "jax"),
        "notes": (
            "Dreamer 4 rollout route bound to a user-staged official checkout; upstream is all-rights-reserved, "
            "so no source is vendored. Requires the checkout, its CUDA-12 JAX environment, and a trained Orbax "
            "checkpoint. WorldFoundry interactions are rendered into Minecraft/VPT action JSONL."
        ),
    },
    "longvie-1": {
        "display_name": "LongVie",
        "category": "Video Generation",
        # Upstream has not published a first-generation checkpoint.  In
        # particular, Vchitect/LongVie2 must not be used here: its control and
        # DiT weights describe a different model variant.  Advanced users may
        # still pass an explicitly staged LongVie-1 control checkpoint.
        "default_model_ref": "",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "image-to-video",
        "default_prompt": (
            "A young man holds a bright sparkler at night while the camera moves slowly around him, "
            "preserving his identity, pose, and the reference scene."
        ),
        "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_call_kwargs": {
            "execute": True,
            "dense_video": str(_data_path("test_cases", "longvie", "dense_control.mp4")),
            "sparse_video": str(_data_path("test_cases", "longvie", "sparse_control.mp4")),
            "num_frames": 81,
            "height": 352,
            "width": 640,
            "fps": 16,
            "num_inference_steps": 50,
            "seed": 0,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "ref_image_path",
            "operator_kwargs",
            "execute",
            "num_frames",
            "height",
            "width",
            "num_inference_steps",
            "seed",
            "dense_video",
            "sparse_video",
            "tiled",
            "negative_prompt",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "longvie_weight_dir",
            "weight_dir",
            "wan_base_dir",
            "tokenizer_dir",
            "control_weight_path",
            "dit_weight_path",
            "torch_dtype",
            "use_usp",
            "ring_degree",
            "ulysses_degree",
            "enable_vram_management",
            "control_layers",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("longvie", "longvie1", "longvie-v1"),
        "tags": ("image-to-video", "long-video", "depth-controlled-video", "official-runtime", "in-tree-runtime"),
        "notes": (
            "The first-generation runtime adapter is available, but upstream has not published a LongVie-1 "
            "checkpoint. Pass an explicitly staged first-generation control weight; LongVie2 weights are "
            "rejected to prevent cross-variant demo reuse. Use longvie-2 for the public checkpoint-backed route."
        ),
    },
    "longvie-2": {
        "display_name": "LongVie 2",
        "category": "Video Generation",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "LongVie2",
            "Vchitect--LongVie2",
            "custom--LongVie2",
            fallback="Vchitect/LongVie2",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_task_type": "image-to-video",
        "default_prompt": (
            "A young man holds a bright sparkler at night while the camera moves slowly around him, "
            "preserving his identity, pose, and the reference scene."
        ),
        "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_call_kwargs": {
            "execute": True,
            "dense_video": str(_data_path("test_cases", "longvie", "dense_control.mp4")),
            "sparse_video": str(_data_path("test_cases", "longvie", "sparse_control.mp4")),
            "num_frames": 81,
            "height": 352,
            "width": 640,
            "fps": 16,
            "seed": 0,
            # Native 352x640 segments fit comfortably on the supported A100
            # profile. Full-frame VAE encode/decode is both faster and avoids
            # tile-boundary artifacts; callers on smaller GPUs can opt in.
            "tiled": False,
            "num_inference_steps": 50,
        },
        "default_load_kwargs": {
            "model_id": "longvie-2",
            "use_usp": True,
            "ring_degree": 1,
            "ulysses_degree": 4,
            "torchrun_nproc_per_node": 4,
            # The supported A100/H100 80GB quality profile fits the complete
            # 14B stack per rank. CPU layer swapping otherwise dominates the
            # four-rank USP path and can leave the first denoise step running
            # for minutes while every GPU holds less than 8GB.
            "enable_vram_management": False,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "ref_image_path",
            "operator_kwargs",
            "execute",
            "num_frames",
            "height",
            "width",
            "seed",
            "dense_video",
            "sparse_video",
            "tiled",
            "negative_prompt",
            "num_inference_steps",
            "allow_control_padding",
        ),
        "stream_params": (
            "prompt",
            "dense_video",
            "sparse_video",
            "execute",
            "num_frames",
            "fps",
            "seed",
            "tiled",
            "negative_prompt",
            "num_inference_steps",
            "allow_control_padding",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "longvie_weight_dir",
            "weight_dir",
            "wan_base_dir",
            "tokenizer_dir",
            "control_weight_path",
            "dit_weight_path",
            "torch_dtype",
            "use_usp",
            "ring_degree",
            "ulysses_degree",
            "enable_vram_management",
            "control_layers",
            "torchrun_nproc_per_node",
            "torchrun_nproc",
            "nproc_per_node",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("longvie2", "longvie-v2"),
        "tags": (
            "image-to-video",
            "long-video",
            "depth-controlled-video",
            "queued-segment-generation",
            "not-realtime",
            "official-runtime",
            "in-tree-runtime",
        ),
        "notes": (
            "Queued full-quality 81-frame segments at 16 FPS. This is not WASD realtime: "
            "the user supplies a prompt, initial image, dense depth control video, and sparse "
            "pointmap/track control video. Continue reuses the previous final frame, eight-frame "
            "history, noise, and resident weights."
        ),
    },
    "show-o": {
        "display_name": "Show-O",
        "category": "Video Generation",
        "default_model_ref": _show_o_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "supports_stream": False,
        "default_task_type": "text-to-image",
        "default_load_kwargs": _show_o_default_load_kwargs,
        "default_call_kwargs": {
            "mode": "t2i",
            "generation_timesteps": 4,
            "guidance_scale": 1.0,
            "resolution": 256,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "mode",
            "generation_timesteps",
            "guidance_scale",
            "batch_size",
            "resolution",
            "generation_temperature",
            "mask_schedule",
            "noise_type",
        ),
        "load_params": (
            "pretrained_model_path",
            "model_path",
            "checkpoint_path",
            "device",
            "vq_model_path",
            "vq_model_name",
            "llm_model_path",
            "resolution",
            "batch_size",
            "guidance_scale",
            "generation_timesteps",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("show_o", "showo"),
        "tags": ("text-to-image", "official-runtime", "in-tree-runtime"),
        "notes": "Current in-tree Show-O wrapper supports t2i image generation; video/MMU modes are not marked as verified.",
    },
    "matrix-game-1": {
        "display_name": "Matrix-Game-1",
        "default_model_ref": _matrix_game_1_default_ref,
        "default_input_path": "worldfoundry/data/test_cases/matrix-game-1/official_initial_image/forest_00.jpg",
        "default_load_kwargs": _matrix_game_1_default_load_kwargs,
        "default_interactions": ("forward", "left", "right", "camera_l", "camera_r"),
        "default_call_kwargs": {
            "fps": 16,
            "execute": True,
            "inference_steps": 50,
            "max_images": 1,
            "max_conditions": 1,
            "video_length": 17,
            "i2v_type": "refiner",
            "timeout_seconds": 3600,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "image_path",
            "output_path",
            "fps",
            "return_dict",
            "execute",
            "timeout_seconds",
            "inference_steps",
            "max_images",
            "max_conditions",
            "video_length",
            "gpu_id",
            "i2v_type",
        ),
        "aliases": ("matrix-game1",),
        "tags": ("navigation", "interaction-heavy", "in-tree-runtime", "gpu-validation-recorded", "visual-qa-pending"),
        "notes": "Defaults use the official 50-step Matrix-Game-1 runner with a bounded 17-frame Studio regression shape and an in-domain GameWorldScore initial image staged in-tree. Generic Studio images generated noisy visuals and are not accepted as official visual parity.",
    },
    "matrix-game-2": {
        "display_name": "Matrix-Game-2",
        "default_model_ref": _matrix_game_2_default_ref,
        "default_load_kwargs": _matrix_game_2_default_load_kwargs,
        "default_interactions": (),
        "default_call_kwargs": _matrix_game_2_default_call_kwargs,
        "aliases": ("matrixgame", "matrixgame2", "matrix-game2"),
        "tags": ("navigation", "stream", "interaction-heavy", "wmfactory", "state-init"),
        "notes": "Resident interactive runtime. The initial image and all navigation/camera actions come from the user; no private benchmark fixture or action trajectory is selected by default.",
    },
    "matrix-game-3": {
        "display_name": "Matrix-Game-3",
        "default_model_ref": _matrix_game_3_default_ref,
        "default_prompt": MATRIX_GAME_3_OFFICIAL_PROMPT,
        "default_load_kwargs": _matrix_game_3_default_load_kwargs,
        "default_interactions": ("forward", "camera_r"),
        "default_call_kwargs": _matrix_game_3_default_call_kwargs,
        "aliases": ("matrixgame3", "matrix-game3"),
        "tags": ("navigation", "stream", "wmfactory", "state-init", "official-demo"),
        "notes": "Defaults mirror the Matrix-Game-3 README cityscape demo: 12 iterations, 3 denoising steps, seed 42, 704*1280, INT8 LightVAE. Async VAE remains off by default because it fails when CUDA_VISIBLE_DEVICES remaps GPUs.",
    },
    "matrix-game-3.5-first-person": {
        "display_name": "Matrix-Game 3.5 First-Person Base",
        "default_model_ref": lambda: _matrix_game_35_default_ref("first"),
        "default_task_type": "camera-controlled-image-to-video",
        "default_prompt": (
            "A young man holds a bright sparkler at night while the camera moves gently forward and "
            "to the right, preserving his face, clothing, and the illuminated architecture."
        ),
        "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_interactions": (),
        "default_load_kwargs": _matrix_game_35_default_load_kwargs,
        "default_call_kwargs": _matrix_game_35_third_person_default_call_kwargs,
        "input_params": ("image_path", "camera"),
        "call_params": (
            "prompt",
            "images",
            "image",
            "image_path",
            "input_path",
            "camera_path",
            "camera",
            "trajectory_npz",
            "caption_path",
            "output_path",
            "fps",
            "num_blocks",
            "steps",
            "cfg_scale",
            "seed",
            "camera_convention",
            "keep_workspace",
            "timeout_seconds",
            "return_dict",
        ),
        "aliases": ("matrix-game-3.5-fp", "matrix-game-35-first-person"),
        "tags": ("world-model", "camera-control", "first-person", "official-default-steps"),
        "notes": (
            "Studio uses the released one-block 25-step first-person base model and a deterministic "
            "86-pose c2w dolly trajectory materialized by the runtime."
        ),
    },
    "matrix-game-3.5-third-person": {
        "display_name": "Matrix-Game 3.5 Third-Person Base",
        "default_model_ref": lambda: _matrix_game_35_default_ref("third"),
        "default_task_type": "camera-controlled-image-to-video",
        "default_prompt": (
            "A young man holds a bright sparkler at night while the camera moves gently around him, "
            "preserving his identity, clothing, and the illuminated architecture."
        ),
        "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_interactions": (),
        "default_load_kwargs": _matrix_game_35_default_load_kwargs,
        "default_call_kwargs": _matrix_game_35_third_person_default_call_kwargs,
        "input_params": ("image_path", "camera"),
        "call_params": (
            "prompt",
            "images",
            "image",
            "image_path",
            "input_path",
            "camera_path",
            "camera",
            "trajectory_npz",
            "refs",
            "subject_refs",
            "caption_path",
            "output_path",
            "fps",
            "num_blocks",
            "steps",
            "cfg_scale",
            "seed",
            "camera_convention",
            "keep_workspace",
            "timeout_seconds",
            "return_dict",
        ),
        "aliases": ("matrix-game-3.5-tp", "matrix-game-35-third-person"),
        "tags": ("world-model", "camera-control", "subject-reference", "official-default-steps"),
        "notes": (
            "Studio uses the released one-block 25-step base-model defaults and a deterministic 86-pose "
            "c2w dolly trajectory materialized by the runtime. Subject references remain optional."
        ),
    },
    "dualcamctrl": {
        "display_name": "DualCamCtrl",
        "category": "Video Generation",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _dualcamctrl_base_default_ref,
        "default_task_type": "image-depth-camera-to-video",
        "default_prompt": (
            "High aerial view over a British seaside town on a sunny afternoon. Turquoise sea with gentle waves, "
            "a long wooden pier stretching into the water, sandy beach and a bustling promenade with parked cars. "
            "Foreground: curved coastal road and pastel, terraced houses with orange roofs. Midground: beach, pools "
            "and small buildings. Background: white cliffs and rolling hills fading into haze. Few fluffy clouds, "
            "bright natural light, crisp visibility. Slow tilt-down and rightward pan, subtle zoom for parallax. "
            "Natural color grade, 4K, 24fps, steady gimbal/drone feel, light wind ambience and distant seagulls."
        ),
        "default_input_path": str(_data_path("test_cases", "dualcamctrl", "demo_pic", "seaside.png")),
        "default_interactions": (str(_data_path("test_cases", "dualcamctrl", "demo_pic", "seaside.torch")),),
        "default_load_kwargs": _dualcamctrl_default_load_kwargs,
        "default_call_kwargs": _dualcamctrl_default_call_kwargs,
        "input_params": ("image_path", "depth_image", "camera_path"),
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "demo_name",
            "image_path",
            "depth_image",
            "depth_path",
            "camera_path",
            "trajectory_file",
            "num_frames",
            "frame_len",
            "height",
            "width",
            "num_inference_steps",
            "infer_steps",
            "seed",
            "cfg_scale",
            "negative_prompt",
            "return_control_latents",
            "original_height",
            "original_width",
            "tiled",
            "quality",
        ),
        "load_params": (
            "model_path",
            "pretrained_model_path",
            "base_model_path",
            "base_model_repo",
            "tokenizer_repo",
            "dualcamctrl_repo",
            "checkpoint_path",
            "ckpt_path",
            "checkpoint_name",
            "config_path",
            "model_config",
            "local_model_path",
            "download_resource",
            "allow_download",
            "torch_dtype",
            "weight_dtype",
            "copy_control_weights",
            "redirect_common_files",
            "use_usp",
            "load_model",
            "device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("dual-cam-ctrl", "dualcam", "dual-camera-control"),
        "tags": ("image-to-video", "depth-conditioned-video", "camera-control", "official-demo", "in-tree-runtime"),
        "notes": "Defaults mirror the official DualCamCtrl seaside demo: RGB image, depth image, camera .torch path, 61 frames, 320x480, 50 denoising steps, fps 10, seed 42.",
    },
    "warp-as-history": {
        "display_name": "Warp-as-History",
        "default_model_ref": _warp_as_history_default_ref,
        "default_load_kwargs": _warp_as_history_default_load_kwargs,
        "default_prompt": "A cyclist riding past a colorful graffiti wall beside trees and tall grass, urban BMX scene, with camera movement.",
        "default_call_kwargs": {"num_frames": 33, "fps": 16, "height": 384, "width": 640},
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "output_dir",
            "fps",
            "num_frames",
            "height",
            "width",
            "seed",
            "return_dict",
            "camera_poses",
            "camera_poses_path",
            "warp_video",
            "warp_video_path",
            "warp_visibility_mask",
            "warp_visibility_mask_path",
            "demo_csv_path",
            "plan_only",
            "dtype",
            "lora_path",
            "no_lora",
            "enable_optional_attention",
            "warp_debug_dir",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "lazy",
            "lora_path",
            "pi3x_ckpt_path",
            "python_executable",
        ),
        "aliases": ("wah", "warp_as_history", "yyfz233/warp-as-history"),
        "tags": ("image-to-video", "camera-control", "in-tree-runtime"),
        "notes": "In-tree official Warp-as-History runtime. Uses vendored demo conditioning by default; real custom runs should pass a first frame plus camera_poses_path or warp_video_path.",
    },
    "self-forcing": {
        "display_name": "Self-Forcing",
        "category": "Video Generation",
        "default_model_ref": _self_forcing_default_ref,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "default_prompt": (
            "A cinematic shot of a red sports car driving along a coastal highway at sunset, "
            "with realistic motion, rich detail, and stable lighting."
        ),
        "default_load_kwargs": _self_forcing_default_load_kwargs,
        "default_call_kwargs": _self_forcing_default_call_kwargs,
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "num_output_frames",
            "seed",
            "num_samples",
            "use_ema",
            "save_with_index",
            "extended_prompt",
            "i2v",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "runtime_root",
            "repo_root",
            "checkpoint_path",
            "ckpt_path",
            "config_path",
            "wan_models_root",
            "wan_root",
            "python_executable",
            "num_output_frames",
            "seed",
            "num_samples",
            "fps",
            "use_ema",
            "save_with_index",
            "report_timing",
            "extended_prompt",
        ),
        "aliases": ("selfforcing", "gdhe17/self-forcing"),
        "tags": ("autoregressive-video", "official-runtime", "wan2.1", "world-model"),
        "notes": "Studio defaults mirror the official Self-Forcing CLI: configs/self_forcing_dmd.yaml, --use_ema, and the staged DMD checkpoint when available.",
    },
    "causal-forcing": {
        "display_name": "Causal-Forcing",
        "category": "Video Generation",
        "default_model_ref": _causal_forcing_default_ref,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "default_prompt": (
            "A cinematic shot of a futuristic train crossing a mountain valley under dramatic clouds, "
            "with coherent motion and high visual detail."
        ),
        "default_load_kwargs": _causal_forcing_default_load_kwargs,
        "default_call_kwargs": _causal_forcing_default_call_kwargs,
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "num_output_frames",
            "seed",
            "use_ema",
            "report_timing",
            "i2v",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "runtime_root",
            "repo_root",
            "checkpoint_path",
            "ckpt_path",
            "config_path",
            "wan_models_root",
            "wan_root",
            "python_executable",
            "num_output_frames",
            "seed",
            "fps",
            "use_ema",
            "report_timing",
        ),
        "aliases": ("causalforcing", "thu-ml/causal-forcing"),
        "tags": ("autoregressive-video", "official-runtime", "wan2.1", "world-model"),
        "notes": "Studio defaults mirror the official Causal-Forcing chunk-wise CLI: configs/causal_forcing_dmd_chunkwise.yaml and chunkwise/causal_forcing.pt.",
    },
    "gen3c": {
        "display_name": "GEN3C",
        "default_model_ref": _gen3c_default_ref,
        "default_input_path": "worldfoundry/data/test_cases/gen3c/image.png",
        "default_prompt": (
            "A cinematic view of a futuristic science-fiction city with coherent structures, realistic "
            "lighting, and smooth natural camera motion."
        ),
        "default_load_kwargs": _gen3c_default_load_kwargs,
        "default_interactions": ("left",),
        "default_call_kwargs": _gen3c_default_call_kwargs,
        "call_params": (
            "images",
            "interactions",
            "prompt",
            "negative_prompt",
            "trajectory",
            "camera_path",
            "region_hint",
            "camera_rotation",
            "movement_distance",
            "rendered_warp_images",
            "rendered_warp_masks",
            "camera_to_world",
            "camera_intrinsics",
            "scene_name",
            "output_dir",
            "return_dict",
            "guidance",
            "num_steps",
            "num_video_frames",
            "fps",
            "height",
            "width",
            "seed",
            "num_gpus",
            "noise_aug_strength",
            "save_buffer",
            "filter_points_threshold",
            "foreground_masking",
            "disable_prompt_upsampler",
            "disable_guardrail",
            "disable_prompt_encoder",
            "offload_diffusion_transformer",
            "offload_tokenizer",
            "offload_text_encoder_model",
            "offload_prompt_upsampler",
            "offload_guardrail_models",
        ),
        "stream_params": ("camera_path", "prompt", "region_hint", "fps", "seed"),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "weight_dtype",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": (
            "navigation",
            "stream",
            "camera-control",
            "world-explorer",
            "native-imgui",
        ),
        "notes": (
            "The native World Explorer adapter preserves arbitrary authored camera poses and "
            "renders the resident RGBD cache into GEN3C conditioning buffers."
        ),
    },
    "ac3d": {
        "display_name": "AC3D",
        "category": "Video Generation",
        "default_prompt": (
            "Three fluffy sheep sit side by side at a rustic wooden table, each eagerly digging into their bowls of "
            "spaghetti. The pasta is tangled playfully around their woolly faces, and the bright red sauce splatters "
            "across their fur. The scene takes place in a lush, green meadow surrounded by rolling hills, with a few "
            "grazing cows in the background."
        ),
        "default_call_kwargs": _ac3d_default_call_kwargs,
        "call_params": (
            "prompt",
            "output_path",
            "return_dict",
            "video_root_dir",
            "annotation_json",
            "num_frames",
            "height",
            "width",
            "stride_min",
            "stride_max",
            "start_camera_idx",
            "end_camera_idx",
            "controlnet_weights",
            "controlnet_guidance_start",
            "controlnet_guidance_end",
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "dtype",
        ),
        "tags": ("camera-control", "text-to-video", "official-runtime"),
        "notes": "Official AC3D inference requires a RealEstate10K-style fixture with annotations/test.json, pose_files, and video_clips.",
    },
    "lingbot-world": {
        "display_name": "LingBot-World",
        "default_model_ref": _lingbot_world_default_ref,
        "default_load_kwargs": _lingbot_world_default_load_kwargs,
        "default_interactions": ("forward", "left", "right", "camera_l", "camera_r"),
        "default_call_kwargs": {
            "max_area": 480 * 832,
            "num_frames": 161,
            "offload_model": False,
            "sampling_steps": 70,
            "seed": 42,
        },
        "load_params": (
            "task",
            "runtime_variant",
            "fast_model_path",
            "t5_fsdp",
            "dit_fsdp",
            "t5_cpu",
            "ulysses_size",
            "nproc_per_node",
            "torchrun_nproc_per_node",
            "torchrun_nproc",
            "convert_model_dtype",
            *DISPATCH_LOAD_PARAMS,
        ),
        "call_params": (
            "images",
            "prompt",
            "interactions",
            "action_path",
            "resize_H",
            "resize_W",
            "max_area",
            "num_frames",
            "vis_ui",
            "allow_act2cam",
            "action_string",
            "wmfactory_action_controls",
            "sampling_steps",
            "offload_model",
            "seed",
            "output_path",
            "fps",
            "return_dict",
        ),
        "aliases": ("lingbot-fast", "lingbot-world-fast", "lingbotworldfast", "lingbotworld-fast"),
        "tags": ("navigation", "stream", "camera-control", "interaction-heavy", "wmfactory", "state-init"),
        "notes": (
            "Base-cam uses native DiT/T5 FSDP plus Ulysses sequence parallelism. Workspace selects the official "
            "8-GPU topology when available and supports 4 GPUs as the compact topology; only rank 0 decodes "
            "and materializes the output."
        ),
    },
    "uni3c": {
        "display_name": "Uni3C",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "image-to-video",
        "default_prompt": (
            "A close-up of a wet white cat wearing oversized amber sunglasses in a sunlit swimming pool, "
            "while the camera smoothly pushes forward toward the cat."
        ),
        "default_input_path": str(
            _data_path("test_cases", "uni3c", "synthetic-forward-9f-768x480", "reference.jpg")
        ),
        "default_call_kwargs": {
            "render_path": str(_data_path("test_cases", "uni3c", "synthetic-forward-9f-768x480")),
            "mode": "camera",
            "num_frames": 9,
            "max_area": 114688,
            "seed": 0,
            "fps": 16,
            "num_gpus": 1,
        },
        "call_params": (
            "prompt",
            "images",
            "interactions",
            "render_path",
            "mode",
            "num_frames",
            "max_area",
            "seed",
            "fps",
            "num_gpus",
            "output_path",
            "plan_only",
            "return_dict",
        ),
        "load_params": (
            "controller_path",
            "base_model_path",
            "model_path",
            "device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": ("image-to-video", "camera-control", "official-stage-two-runtime"),
        "notes": (
            "Requires a reference image and the matching official stage-one render bundle containing "
            "render.mp4, render_mask.mp4, and cam_info.json."
        ),
    },
    "sana-wm": {
        "display_name": "SANA-WM",
        "category": "Video Generation",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_backend": "from_pretrained",
        "default_model_ref": _sana_wm_default_ref,
        "default_prompt": "",
        "default_task_type": "image-camera-video",
        "default_interactions": ("forward", "left", "right", "camera_up", "camera_l", "camera_r"),
        "default_call_kwargs": {
            "window_frames": 81,
            "fps": 16,
            "step": 60,
            "cfg_scale": 5.0,
            "seed": 42,
            "return_dict": True,
        },
        "load_params": ("checkpoint_source", "checkpoint_overrides", *DISPATCH_LOAD_PARAMS),
        "call_params": (
            "images",
            "prompt",
            "interactions",
            "window_frames",
            "fps",
            "step",
            "cfg_scale",
            "seed",
            "return_dict",
        ),
        "stream_params": ("interactions", "realtime_segments", "seed"),
        "aliases": ("sana-world-model", "sana-wm-2.6b"),
        "tags": ("world-model", "image-to-video", "camera-control", "stream", "bidirectional", "in-tree-runtime"),
        "notes": "Resident two-stage SANA-WM runtime with user-provided image/prompt, native 8k+1 temporal windows, and configurable component placement.",
    },
    "dreamx-world-5b-cam": {
        "display_name": "DreamX-World 5B Cam",
        "category": "Video Generation",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_backend": "from_pretrained",
        "default_model_ref": _dreamx_world_default_ref,
        "default_load_kwargs": _dreamx_world_default_load_kwargs,
        "default_prompt": (
            "A blocky coastal cliff overlooks a calm ocean at sunset as the camera glides "
            "smoothly along the shoreline, with stable terrain, vegetation, and warm lighting."
        ),
        "default_input_path": str(_data_path("test_cases", "dreamx_world", "007.jpg")),
        "default_task_type": "image-camera-video",
        "default_interactions": (),
        "default_call_kwargs": {
            "num_frames": 33,
            "fps": 16,
            "height": 704,
            "width": 1280,
            "num_inference_steps": 30,
            "guidance_scale": 5.0,
            "seed": 42,
            "return_dict": True,
        },
        "load_params": (
            "checkpoint_source",
            "wan_model_path",
            "nproc_per_node",
            "torchrun_nproc_per_node",
            "torchrun_nproc",
            *DISPATCH_LOAD_PARAMS,
        ),
        "call_params": (
            "images",
            "prompt",
            "interactions",
            "num_frames",
            "fps",
            "height",
            "width",
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "return_dict",
        ),
        "stream_params": ("prompt", "interactions", "realtime_segments", "seed"),
        "aliases": ("dreamx-5b-cam",),
        "tags": (
            "interactive-world",
            "image-to-video",
            "camera-control",
            "stream",
            "in-tree-runtime",
            "user-input-only",
        ),
        "notes": (
            "Resident 1+4k camera-conditioned segments with user-provided image and prompt. "
            "Weights, VAE, T5 embeddings, and Ulysses topology stay resident; each continuation "
            "starts from the previous final frame without file round trips."
        ),
    },
    "dreamx-world-5b": {
        "display_name": "DreamX-World 5B AR",
        "category": "Video Generation",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_backend": "from_pretrained",
        "default_model_ref": _dreamx_world_ar_default_ref,
        "default_load_kwargs": _dreamx_world_ar_default_load_kwargs,
        "default_prompt": (
            "A blocky coastal cliff overlooks a calm ocean at sunset as the camera glides "
            "smoothly along the shoreline, with stable terrain, vegetation, and warm lighting."
        ),
        "default_input_path": str(_data_path("test_cases", "dreamx_world", "007.jpg")),
        "default_task_type": "image-camera-video",
        "default_interactions": (),
        "default_call_kwargs": {
            "fps": 16,
            "seed": 42,
            "return_dict": True,
        },
        "load_params": (
            "checkpoint_source",
            "wan_model_path",
            *DISPATCH_LOAD_PARAMS,
        ),
        "call_params": (
            "images",
            "prompt",
            "interactions",
            "fps",
            "seed",
            "return_dict",
        ),
        "stream_params": ("prompt", "interactions", "realtime_segments", "seed"),
        "aliases": ("dreamx-world", "dreamx", "dreamx-ar", "dreamx-5b"),
        "tags": (
            "interactive-world",
            "image-to-video",
            "camera-control",
            "stream",
            "causal",
            "autoregressive",
            "in-tree-runtime",
            "user-input-only",
        ),
        "notes": (
            "Resident distilled causal runtime with three-latent blocks, persistent attention/KV "
            "and VAE caches, continuous camera pose, and no decoded-RGB feedback loop."
        ),
    },
    "lingbot-world-v2": {
        "display_name": "LingBot-World-V2",
        "default_model_ref": _lingbot_world_v2_default_ref,
        "default_load_kwargs": _lingbot_world_v2_default_load_kwargs,
        "default_prompt": (
            "The video presents a soaring journey through a fantasy jungle. The wind whips past the rider's "
            "blue hands gripping the reins, causing the leather straps to vibrate. The ancient gothic castle "
            "approaches steadily, its stone details becoming clearer against the backdrop of floating islands "
            "and distant waterfalls."
        ),
        "default_task_type": "image-camera-video",
        "default_call_kwargs": {
            "size": "480*832",
            "frame_num": 361,
            "chunk_size": 4,
            "seed": 42,
            "sample_shift": 10.0,
            "local_attn_size": 18,
            "sink_size": 6,
            "nproc_per_node": 1,
            "t5_fsdp": True,
            "dit_fsdp": True,
            "offload_model": False,
            "return_dict": True,
            "timeout_seconds": 7200,
        },
        "load_params": (
            "checkpoint_source",
            "python_executable",
            "t5_fsdp",
            "dit_fsdp",
            "t5_cpu",
            "convert_model_dtype",
            *DISPATCH_LOAD_PARAMS,
        ),
        "call_params": (
            "images",
            "prompt",
            "interactions",
            "action_path",
            "input_dir",
            "output_path",
            "return_dict",
            "frame_num",
            "num_frames",
            "size",
            "chunk_size",
            "seed",
            "sample_shift",
            "local_attn_size",
            "sink_size",
            "nproc_per_node",
            "t5_fsdp",
            "dit_fsdp",
            "offload_model",
            "timeout_seconds",
        ),
        "aliases": ("lingbot-world-infinity", "lingbot-v2", "lingbot_world_v2"),
        "tags": ("world-model", "image-to-video", "camera-control", "causal", "official-runtime"),
        "notes": (
            "Fully in-tree causal-fast runtime. action_path must contain poses.npy and intrinsics.npy. The "
            "official profile uses 8 GPUs; Workspace also supports 4 GPUs, reuses its existing process group "
            "without nested torchrun, and materializes output only on rank 0."
        ),
    },
    "infinite-world": {
        "display_name": "Infinite-World",
        "default_model_ref": _infinite_world_default_ref,
        "default_prompt": (
            "A young man holds a sparkling firework at night while the camera explores the illuminated "
            "architecture with smooth, coherent motion."
        ),
        "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_interactions": ("forward", "left", "right", "camera_l", "camera_r"),
        "default_call_kwargs": {"num_frames": 81, "seed": 42, "fps": 30, "return_dict": True},
        "call_params": (
            "images",
            "prompt",
            "interactions",
            "num_chunks",
            "num_frames",
            "negative_prompt",
            "seed",
            "output_path",
            "output_dir",
            "fps",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "lazy",
            "args",
            "weight_dtype",
            "config_path",
            "checkpoint_path",
            "bucket_config_name",
            "num_sampling_steps",
            "shift",
            "guidance_scale",
            "negative_prompt",
            "seed",
        ),
        "aliases": ("infiniteworld", "infinite_world"),
        "tags": ("navigation", "stream", "camera-control", "interaction-heavy", "wmfactory", "state-init"),
        "notes": "Supports action-conditioned stream continuation in Studio. Start from an image-conditioned first turn, then continue with short navigation tokens.",
    },
    "yume": {
        "display_name": "YUME",
        "default_model_ref": _yume_default_ref,
        "default_prompt": (
            "A first-person journey follows the stone path toward an arched bridge beside a calm lake, "
            "with coherent forward motion and stable natural scenery."
        ),
        "default_input_path": str(_data_path("test_cases", "hunyuan_worldplay", "test.png")),
        "default_interactions": ("forward", "camera_l"),
        "default_load_kwargs": {"fsdp": False, "t5_cpu": True},
        "default_call_kwargs": {
            "interactions": ["forward", "camera_l"],
            "task_type": "i2v",
            "size": "544*960",
            "sampling_method": "ode",
            "interaction_speeds": [100, 4],
            "interaction_distances": [4, None],
            "num_euler_timesteps": 50,
            "seed": 42,
        },
        "aliases": ("yume-i2v",),
        "tags": ("stream", "egocentric", "wmfactory", "state-init"),
    },
    "yume-1p5": {
        "display_name": "YUME-1.5",
        "default_model_ref": _yume_1p5_default_ref,
        "default_prompt": (
            "A cinematic first-person journey follows the sunlit stone path toward an arched bridge beside "
            "reflective water, with stable forward motion and detailed natural scenery."
        ),
        "default_input_path": str(_data_path("test_cases", "hunyuan_worldplay", "test.png")),
        "default_interactions": ("forward", "left", "camera_r"),
        "default_load_kwargs": {"fsdp": False},
        "default_call_kwargs": {
            "interactions": ["forward", "left", "camera_r"],
            "task_type": "i2v",
            "size": "704*1280",
            "interaction_speeds": [100, 100, 4],
            "interaction_distances": [4, 4, None],
            "num_euler_timesteps": 4,
            "seed": 42,
        },
        "aliases": ("yume1.5", "yume-1.5"),
        "tags": ("stream", "egocentric", "wmfactory", "state-init"),
    },
    "worldcam": {
        "display_name": "WorldCam",
        "default_model_ref": _worldcam_default_ref,
        "default_load_kwargs": _worldcam_default_load_kwargs,
        "default_call_kwargs": _worldcam_default_call_kwargs,
        "default_prompt": "a first-person view within an industrial setting, likely a power plant, with concrete buildings, chain-link fences, and parked vehicles.",
        "call_params": (
            "prompt",
            "video",
            "input_path",
            "video_path",
            "output_path",
            "fps",
            "return_dict",
            "height",
            "width",
            "intrinsics",
            "extrinsics",
            "intrinsics_path",
            "extrinsics_path",
            "conditioning_frames",
            "num_ar_steps",
            "num_inference_steps",
            "negative_prompt",
            "cfg_scale",
            "seed",
            "long_term_memory_start_step",
            "long_term_memory_num_clips",
            "long_term_memory_ref_indices",
            "attention_sink_inference",
            "trim_conditioning",
            "tiled",
        ),
        "stream_params": (),
        "tags": ("interactive-world", "official-runtime", "video-to-world"),
    },
    "wow": {
        "display_name": "WoW",
        "default_model_ref": _wow_default_ref,
        "default_input_path": str(_data_path("test_cases", "test_vla_image_case1", "init_frame.png")),
        "default_prompt": "The Franka robot grasps the red bottle on the table.",
        "default_call_kwargs": {
            "steps": 50,
            "seed": 42,
            "num_frames": 41,
            "no_tiled": False,
            "enable_vram_management": True,
            "persistent_param_gb": 70,
        },
        "call_params": (
            "input_path",
            "text_prompt",
            "prompt",
            "steps",
            "seed",
            "num_frames",
            "frames",
            "no_tiled",
            "enable_vram_management",
            "no_vram_management",
            "persistent_param_gb",
            "output_path",
            "return_dict",
        ),
        "aliases": ("wow-world-model", "world-omniscient-world-model", "wow-world-model/wow-1-wan-14b-600k"),
        "tags": ("world-model", "image-to-video", "robotics", "official-runtime"),
        "notes": "Defaults match the official WoW Wan demo checkpoint X-Humanoid/WoW-1-Wan-14B-600k. Use a local Hugging Face cache or repo id with image conditioning, prompt text, seed 42, and video generation.",
    },
    "cameractrl": {
        "display_name": "CameraCtrl",
        "default_model_ref": _cameractrl_default_ref,
        "default_load_kwargs": _cameractrl_default_load_kwargs,
        "default_interactions": ("camera_path",),
        "default_call_kwargs": {
            "trajectory_file": "worldfoundry/data/test_cases/cameractrl/pose_files/0f47577ab3441480.txt"
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "trajectory_file",
            "height",
            "width",
            "image_height",
            "image_width",
            "original_pose_width",
            "original_pose_height",
            "num_frames",
            "video_length",
            "num_inference_steps",
            "infer_steps",
            "guidance_scale",
            "cfg_scale",
            "negative_prompt",
            "seed",
        ),
        "tags": ("camera-control", "text-to-video", "official-runtime"),
    },
    "motionctrl": {
        "display_name": "MotionCtrl",
        "default_model_ref": _motionctrl_default_ref,
        "default_interactions": ("camera_motion",),
        "default_call_kwargs": {
            "height": 256,
            "width": 256,
            "unconditional_guidance_scale": 7.5,
            "ddim_steps": 50,
            "condtype": "camera_motion",
            "camera_pose_path": str(_data_path("test_cases", "motionctrl_conditions", "camera_poses", "test_camera_Round-ZoomIn.json")),
            "seed": 42,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "camera_pose_file",
            "camera_pose_path",
            "trajectory_file",
            "object_trajectory_file",
            "cond_dir",
            "condtype",
            "height",
            "width",
            "num_frames",
            "seed",
            "n_samples",
            "batch_size",
            "bs",
            "unconditional_guidance_scale",
            "unconditional_guidance_scale_temporal",
            "guidance_scale",
            "cfg_scale",
            "ddim_steps",
            "infer_steps",
            "ddim_eta",
            "cond_T",
        ),
        "tags": ("camera-control", "object-motion", "official-runtime"),
    },
    "astra": {
        "display_name": "Astra",
        "default_model_ref": _astra_default_ref,
        "default_prompt": ASTRA_OFFICIAL_PROMPT,
        "default_input_path": str(_data_path("test_cases", "astra", "condition_images", "garden_1.png")),
        "default_load_kwargs": _astra_default_load_kwargs,
        "default_interactions": ("forward_left",),
        "default_call_kwargs": _astra_default_call_kwargs,
        "call_params": (
            "prompt",
            "image_path",
            "images",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "num_frames",
            "frames_per_generation",
            "total_frames_to_generate",
            "num_inference_steps",
            "start_frame",
            "initial_condition_frames",
            "modality_type",
            "height",
            "width",
        ),
        "tags": ("camera-control", "image-to-video", "official-runtime"),
    },
    "lyra-1": {
        "display_name": "Lyra-1",
        "category": "3D Scene",
        "default_task_type": "novel-view-synthesis",
        "default_model_ref": lambda: str(checkpoint_root_path()),
        "default_input_path": str(_data_path("test_cases", "lyra", "Lyra-1", "00172.png")),
        "default_prompt": "",
        "default_interactions": ("zoom_in", "left", "right"),
        "default_call_kwargs": _lyra1_default_call_kwargs,
        "call_params": (
            "images",
            "videos",
            "interactions",
            "prompt",
            "mode",
            "trajectory",
            "reconstruct_3d",
            "output_dir",
            "return_dict",
            "num_video_frames",
            "fps",
            "height",
            "width",
            "seed",
            "num_steps",
            "guidance",
            "num_gpus",
            "movement_distance",
            "camera_rotation",
            "multi_trajectory",
            "total_movement_distance_factor",
            "vipe_path",
            "vipe_starting_frame_idx",
            "filter_points_threshold",
            "foreground_masking",
            "center_depth_quantile",
            "flip_supervision",
            "offload_diffusion_transformer",
            "offload_tokenizer",
            "offload_text_encoder_model",
            "offload_prompt_upsampler",
            "offload_guardrail_models",
            "disable_prompt_encoder",
            "disable_guardrail",
            "show_progress",
            "execute",
            "plan_path",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "weight_dtype",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("lyra1",),
        "tags": ("novel-view", "camera-control", "official-runtime"),
    },
    "lyra": {
        "display_name": "Lyra",
        "category": "3D Scene",
        "default_task_type": "novel-view-synthesis",
        "default_prompt": (
            "A slow, steady camera push forward through a coherent static 3D world with stable geometry."
        ),
        "default_interactions": ("forward",),
        "default_load_kwargs": _lyra2_default_load_kwargs,
        "default_call_kwargs": _lyra2_default_call_kwargs,
        "call_params": (
            "images",
            "interactions",
            "camera_path",
            "prompt",
            "region_hint",
            "fps",
            "resolution",
            "reconstruct_3d",
            "output_dir",
            "seed",
            "return_dict",
            "show_progress",
            "execute",
            "guidance",
            "shift",
            "num_sampling_step",
            "offload",
            "offload_when_prompt",
        ),
        "supports_stream": True,
        "stream_params": ("camera_path", "prompt", "region_hint", "fps", "seed"),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "weight_dtype",
            "checkpoint_dir",
            "negative_prompt_path",
            "da3_model_path_custom",
            "load_runtime",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("lyra2", "nvidia/lyra"),
        "tags": (
            "novel-view",
            "camera-control",
            "official-runtime",
            "world-explorer",
            "native-imgui",
        ),
    },
    "lyra-2": {
        "display_name": "Lyra-2",
        "category": "3D Scene",
        "default_task_type": "novel-view-synthesis",
        "default_prompt": (
            "A slow, steady camera push forward through a coherent static 3D world with stable geometry."
        ),
        "default_interactions": ("forward",),
        "default_load_kwargs": _lyra2_default_load_kwargs,
        "default_call_kwargs": _lyra2_default_call_kwargs,
        "call_params": (
            "images",
            "interactions",
            "camera_path",
            "prompt",
            "region_hint",
            "fps",
            "resolution",
            "reconstruct_3d",
            "output_dir",
            "seed",
            "return_dict",
            "show_progress",
            "execute",
            "guidance",
            "shift",
            "num_sampling_step",
            "offload",
            "offload_when_prompt",
        ),
        "supports_stream": True,
        "stream_params": ("camera_path", "prompt", "region_hint", "fps", "seed"),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "weight_dtype",
            "checkpoint_dir",
            "negative_prompt_path",
            "da3_model_path_custom",
            "load_runtime",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("lyra2",),
        "tags": (
            "novel-view",
            "camera-control",
            "official-runtime",
            "world-explorer",
            "native-imgui",
        ),
    },
    "hunyuanvideo-t2v": {
        "display_name": "HunyuanVideo T2V",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "HunyuanVideo",
            "tencent--HunyuanVideo",
            fallback="tencent/HunyuanVideo",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_prompt": "A cat walks on the grass, realistic style.",
        "call_params": _native_diffusion_video_call_params(),
        "input_params": (
            "prompt",
            "num_frames",
            "height",
            "width",
            "num_inference_steps",
            "fps",
            "guidance_scale",
            "embedded_guidance_scale",
            "seed",
        ),
        "load_params": _native_diffusion_video_load_params(),
        "default_call_kwargs": {
            "num_frames": 129,
            "height": 720,
            "width": 1280,
            "num_inference_steps": 50,
            "fps": 24,
            "guidance_scale": 6.0,
            "embedded_guidance_scale": 6.0,
            "seed": 42,
        },
        "aliases": ("hunyuanvideo", "hunyuan-video", "hunyuanvideo-t2v"),
        "tags": ("text-to-video", "native-diffusion", "local-checkpoint"),
    },
    "hunyuanvideo-i2v": {
        "display_name": "HunyuanVideo I2V",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "HunyuanVideo-I2V",
            "tencent--HunyuanVideo-I2V",
            fallback="tencent/HunyuanVideo-I2V",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "image-to-video",
        "default_prompt": "An Asian man with short hair in black tactical uniform and white clothes waves a firework stick.",
        "call_params": _native_diffusion_video_call_params(),
        "input_params": (
            "prompt",
            "image_path",
            "num_frames",
            "height",
            "width",
            "num_inference_steps",
            "fps",
            "guidance_scale",
            "embedded_guidance_scale",
            "seed",
        ),
        "load_params": _native_diffusion_video_load_params(),
        "default_call_kwargs": {
            "image_path": str(_data_path("test_cases", "hunyuanvideo_i2v", "0.jpg")),
            "num_frames": 129,
            "height": 720,
            "width": 1280,
            "num_inference_steps": 50,
            "fps": 24,
            "guidance_scale": 6.0,
            "embedded_guidance_scale": 6.0,
            "seed": 0,
        },
        "aliases": ("hunyuanvideo-i2v", "hunyuan-video-i2v"),
        "tags": ("image-to-video", "native-diffusion", "local-checkpoint"),
    },
    "framepack": {
        "display_name": "FramePack",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "FramePackI2V_HY",
            "lllyasviel--FramePackI2V_HY",
            "lllyasviel--FramePack_F1_I2V_HY_20250503",
            fallback="lllyasviel/FramePackI2V_HY",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_load_kwargs": {"torch_dtype": "bfloat16"},
        "default_prompt": (
            "A young man holds a bright sparkler at night while the camera moves slowly around him, "
            "preserving his identity, pose, and the reference scene."
        ),
        "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_interactions": ("camera_path",),
        "call_params": (
            *_official_video_call_params(),
            "cfg",
            "gs",
            "rs",
            "use_teacache",
            "mp4_crf",
        ),
        "load_params": _official_video_load_params(),
        "aliases": ("frame-pack", "framepack-i2v"),
        "tags": ("image-to-video", "official-runtime"),
    },
    "hunyuanvideo-1.5-t2v": {
        "display_name": "HunyuanVideo 1.5 T2V",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "tencent--HunyuanVideo-1.5",
            "HunyuanVideo-1.5",
            fallback="tencent/HunyuanVideo-1.5",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_prompt": "A cat walks on a snowy street, cinematic, high quality.",
        "call_params": _native_diffusion_video_call_params(),
        "load_params": _native_diffusion_video_load_params(),
        "default_load_kwargs": lambda: _hunyuanvideo15_default_load_kwargs(image_to_video=False),
        "default_call_kwargs": {
            "height": 720,
            "width": 1280,
            "num_frames": 121,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "fps": 24,
            "seed": 42,
        },
        "aliases": ("hunyuanvideo-1.5", "hunyuanvideo15", "hunyuanvideo15-t2v"),
        "tags": ("text-to-video", "native-diffusion", "local-checkpoint"),
    },
    "hunyuanvideo-1.5-i2v": {
        "display_name": "HunyuanVideo 1.5 I2V",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "tencent--HunyuanVideo-1.5",
            "HunyuanVideo-1.5",
            fallback="tencent/HunyuanVideo-1.5",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "image-to-video",
        "default_prompt": "A young man holds a sparkling firework at night, cinematic lighting, realistic motion.",
        "default_input_path": str(_data_path("test_cases", "hunyuanvideo_i2v", "0.jpg")),
        "call_params": _native_diffusion_video_call_params(),
        "load_params": _native_diffusion_video_load_params(),
        "default_load_kwargs": lambda: _hunyuanvideo15_default_load_kwargs(image_to_video=True),
        "default_call_kwargs": {
            "image_path": str(_data_path("test_cases", "hunyuanvideo_i2v", "0.jpg")),
            "height": 720,
            "width": 544,
            "num_frames": 121,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "fps": 24,
            "seed": 0,
        },
        "aliases": ("hunyuanvideo-1.5-i2v", "hunyuanvideo15-i2v"),
        "tags": ("image-to-video", "native-diffusion", "local-checkpoint"),
    },
    "i2vgen-xl": {
        "display_name": "I2VGen-XL",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "i2vgen-xl",
            "ali-vilab--i2vgen-xl",
            fallback="ali-vilab/i2vgen-xl",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_prompt": (
            "A slow forward camera glide through a lush jungle clearing, with leaves moving gently in the wind "
            "and stable trees, rocks, and background geometry."
        ),
        "default_call_kwargs": {
            "image_path": str(_data_path("test_cases", "neoverse", "videos", "jungle.png")),
            "height": 704,
            "width": 1280,
            "target_fps": 16,
            "num_frames": 16,
            "num_inference_steps": 50,
            "guidance_scale": 9.0,
            "negative_prompt": (
                "distorted, discontinuous, blurry, low resolution, static, disfigured, "
                "disconnected objects, inconsistent geometry"
            ),
            "seed": 8888,
        },
        "call_params": _official_video_call_params(),
        "load_params": _official_video_load_params(),
        "tags": ("image-to-video", "diffusers", "official-runtime"),
    },
    "magi-1": {
        "display_name": "MAGI-1",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "MAGI-1",
            "sand-ai--MAGI-1",
            fallback="sand-ai/MAGI-1",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "call_params": _official_video_call_params(),
        "load_params": _official_video_load_params(),
        "aliases": ("magi", "magi1"),
        "tags": ("image-to-video", "official-runtime"),
    },
    "magi2-preview": {
        "display_name": "MAGI-2 Preview",
        "category": "Video Generation",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": lambda: _checkpoint_model_ref(
            "magi2-preview",
            "sand-ai--MAGI-2-preview",
            fallback="sand-ai/MAGI-2-preview",
        ),
        "default_load_kwargs": _magi2_default_load_kwargs,
        "default_prompt": (
            "A cinematic tracking shot follows a red fox running through a snowy forest, "
            "with coherent motion, stable anatomy, and natural ambient sound."
        ),
        "default_call_kwargs": {
            "short_edge": 512,
            "aspect_ratio": "16:9",
            "duration_seconds": 10.0,
            "num_inference_steps": 100,
            "seed": 0,
            "use_refiner": False,
        },
        "call_params": (
            "prompt",
            "negative_prompt",
            "short_edge",
            "aspect_ratio",
            "duration_seconds",
            "num_inference_steps",
            "seed",
            "use_refiner",
            "output_path",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "checkpoint_path",
            "device",
            "model_id",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("magi2", "magi2_preview", "magi-2-preview", "sand-ai/MAGI-2-preview"),
        "tags": ("text-to-video", "audio-video", "moe", "in-tree-runtime", "local-checkpoint"),
        "notes": "114B MoE preview-only text-to-video with audio and per-layer GPU offload; refiner and image conditioning are not implemented.",
    },
    "mochi-1-preview-t2v": {
        "display_name": "Mochi-1 Preview T2V",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "mochi-1-preview",
            "genmo--mochi-1-preview",
            fallback="genmo/mochi-1-preview",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_prompt": (
            "Close-up of a chameleon's eye as its detailed scaly skin slowly changes color, "
            "photorealistic natural motion, shallow depth of field, ultra high resolution 4K."
        ),
        "default_call_kwargs": {
            "height": 480,
            "width": 848,
            "num_frames": 84,
            "num_inference_steps": 64,
            "guidance_scale": 4.5,
            "fps": 30,
            "seed": 12345,
        },
        "call_params": _official_video_call_params(),
        "load_params": _official_video_load_params(),
        "aliases": ("mochi-1", "mochi", "mochi-1-preview"),
        "tags": ("text-to-video", "diffusers", "official-runtime"),
    },
    "modelscope-t2v": {
        "display_name": "ModelScopeT2V",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "modelscope-damo-text-to-video-synthesis",
            "ali-vilab--modelscope-damo-text-to-video-synthesis",
            fallback="ali-vilab/modelscope-damo-text-to-video-synthesis",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_prompt": "A greenhouse full of tropical plants during a gentle rainstorm, cinematic lighting, stable camera movement.",
        "default_call_kwargs": {"fps": 8, "seed": 302},
        "call_params": _official_video_call_params(),
        "load_params": _official_video_load_params(),
        "aliases": ("modelscope", "modelscope-text-to-video"),
        "tags": ("text-to-video", "diffusers", "official-runtime"),
    },
    "open-sora-plan": {
        "display_name": "Open-Sora-Plan",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "LanguageBind--Open-Sora-Plan-v1.3.0",
            "Open-Sora-Plan-v1.5.0",
            "LanguageBind--Open-Sora-Plan-v1.5.0",
            fallback="LanguageBind/Open-Sora-Plan-v1.3.0",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_prompt": (
            "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. "
            "She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. "
            "She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and "
            "reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
        ),
        "default_call_kwargs": {
            "version": "v1_3",
            "model_subdir": "any93x640x640",
            "vae_subdir": "vae",
            "num_frames": 93,
            "height": 352,
            "width": 640,
            "fps": 18,
            "guidance_scale": 7.5,
            "num_sampling_steps": 100,
            "max_sequence_length": 512,
            "sample_method": "EulerAncestralDiscrete",
            "seed": 1234,
            "nproc_per_node": 1,
            "ae": "WFVAEModel_D8_4x8x8",
            "text_encoder_name_1": _checkpoint_model_ref(
                "mt5-xxl", fallback=str(checkpoint_root_path("mt5-xxl"))
            ),
        },
        "call_params": _official_video_call_params(),
        "load_params": _official_video_load_params(),
        "aliases": ("opensora-plan", "open-sora-plan-v1.3", "open-sora-plan-v1.5"),
        "tags": ("text-to-video", "official-runtime"),
    },
    "open-sora": {
        "display_name": "Open-Sora",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "OpenSora-STDiT-v3",
            "hpcai-tech--OpenSora-STDiT-v3",
            fallback="hpcai-tech/OpenSora-STDiT-v3",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_prompt": (
            "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. "
            "She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. "
            "The street is damp and reflective, creating a mirror effect of the colorful lights. "
            "Many pedestrians walk about as the camera follows her with stable cinematic motion."
        ),
        "input_params": (
            "prompt",
            "num_frames",
            "resolution",
            "aspect_ratio",
            "num_sampling_steps",
            "guidance_scale",
            "aes",
            "seed",
        ),
        "default_call_kwargs": {
            "num_frames": 51,
            "resolution": "480p",
            "aspect_ratio": "9:16",
            "num_sampling_steps": 30,
            "guidance_scale": 7.0,
            "aes": 6.5,
            "seed": 1024,
        },
        "default_load_kwargs": lambda: {
            "required_components": {
                "vae_path": _checkpoint_model_ref(
                    "hpcai-tech--OpenSora-VAE-v1.2",
                    "OpenSora-VAE-v1.2",
                    fallback="hpcai-tech/OpenSora-VAE-v1.2",
                ),
                "text_encoder_path": _checkpoint_model_ref(
                    "DeepFloyd--t5-v1_1-xxl",
                    fallback="DeepFloyd/t5-v1_1-xxl",
                ),
            }
        },
        "call_params": _official_video_call_params(),
        "load_params": _official_video_load_params(),
        "aliases": ("opensora", "open-sora-v3"),
        "tags": ("text-to-video", "official-runtime"),
    },
    "wan2.1-vace": {
        "display_name": "Wan2.1 VACE",
        "category": "Video-to-Video",
        "default_task_type": "video-to-video",
        "suggested_task_types": ("video-to-video",),
        "default_model_ref": lambda: _checkpoint_model_ref(
            "Wan2.1-VACE-14B",
            "Wan2.1-VACE-1.3B-diffusers",
            "Wan-AI--Wan2.1-VACE-14B",
            fallback="Wan-AI/Wan2.1-VACE-14B",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_prompt": (
            "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。"
            "她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。"
            "蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。"
            "小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。"
            "周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。"
        ),
        "call_params": _official_video_call_params()
        + (
            "src_ref_images",
            "src_video",
            "src_mask",
            "vace_context_scale",
        ),
        "input_params": (
            "prompt",
            "src_ref_images",
            "src_video",
            "src_mask",
            "num_frames",
            "size",
            "num_inference_steps",
            "guidance_scale",
            "vace_context_scale",
            "seed",
            "nproc_per_node",
            "ulysses_size",
            "ring_size",
        ),
        "load_params": _official_video_load_params(),
        "default_load_kwargs": {"torch_dtype": "bfloat16"},
        "default_call_kwargs": {
            "num_frames": 81,
            "size": "1280*720",
            "num_inference_steps": 50,
            "guidance_scale": 5.0,
            "seed": -1,
            "nproc_per_node": 1,
            "ulysses_size": 1,
            "ring_size": 1,
        },
        "aliases": ("wan-vace", "wan2.1-vace-14b", "wan-vace-14b"),
        "tags": ("video-to-video", "wan2.1", "official-runtime", "local-checkpoint"),
    },
    "unianimate-dit": {
        "display_name": "UniAnimate-DiT",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "UniAnimate-DiT",
            "ZheWang123--UniAnimate-DiT",
            fallback="ZheWang123/UniAnimate-DiT",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "call_params": _official_video_call_params(),
        "load_params": _official_video_load_params(),
        "default_call_kwargs": {"max_frames": 81, "num_inference_steps": 50, "seed": 0, "height": 832, "width": 480},
        "aliases": ("unianimate", "unianimate-dit-wan"),
        "tags": ("image-to-video", "human-animation", "official-runtime", "local-checkpoint"),
    },
    "sama-14b": {
        "display_name": "SAMA-14B",
        "category": "Video-to-Video",
        "default_task_type": "video-to-video",
        "suggested_task_types": ("video-to-video",),
        "default_model_ref": lambda: _checkpoint_model_ref(
            "SAMA-14B",
            "syxbb--SAMA-14B",
            fallback="syxbb/SAMA-14B",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_input_path": str(_data_path("test_cases", "sama", "1526909-hd_1920_1080_24fps.mp4")),
        "default_prompt": "Replace the spotted baby seal on the sand with a red crab.",
        "call_params": _official_video_call_params(),
        "load_params": _official_video_load_params(),
        "default_call_kwargs": {"max_frames": 49, "height": 480, "width": 832, "fps": 16, "seed": 1, "tiled": True, "prompt_prefix": True},
        "aliases": ("sama", "sama-video"),
        "tags": ("video-to-video", "official-runtime", "local-checkpoint"),
    },
    "krea-realtime-video": {
        "display_name": "Krea Realtime Video",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "krea-realtime-video",
            "krea--krea-realtime-video",
            fallback="krea/krea-realtime-video",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "call_params": _official_video_call_params(),
        "load_params": _official_video_load_params(),
        "default_call_kwargs": {"num_blocks": 2, "height": 480, "width": 832, "fps": 16, "seed": 42},
        "aliases": ("krea-video", "realtime-video"),
        "tags": ("text-to-video", "official-runtime", "local-checkpoint"),
    },
    "mmaudio": {
        "display_name": "MMAudio",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "MMAudio",
            "hkchengrex--MMAudio",
            fallback="hkchengrex/MMAudio",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "video-to-audio",
        "default_input_path": str(_data_path("test_cases", "longcat_video", "motorcycle.mp4")),
        "default_prompt": "A motorcycle engine revs and passes along a road with realistic outdoor ambience.",
        "call_params": _official_video_call_params(),
        "load_params": _official_video_load_params(),
        "default_call_kwargs": {
            "video_path": str(_data_path("test_cases", "longcat_video", "motorcycle.mp4")),
            "duration": 8,
            "num_steps": 25,
            "variant": "small_44k",
        },
        "aliases": ("mm-audio", "hkchengrex/MMAudio"),
        "tags": ("video-to-audio", "audio-generation", "official-runtime", "local-checkpoint"),
    },
    "step-video-t2v": {
        "display_name": "Step-Video-T2V",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "stepvideo-t2v",
            "stepfun-ai--stepvideo-t2v",
            fallback="stepfun-ai/stepvideo-t2v",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "call_params": (
            "prompt",
            "images",
            "video",
            "negative_prompt",
            "output_path",
            "return_dict",
            "fps",
            "num_frames",
            "height",
            "width",
            "num_inference_steps",
            "guidance_scale",
            "time_shift",
            "positive_magic",
            "negative_magic",
            "seed",
            "output_type",
        ),
        "load_params": (
            "model_path",
            "checkpoint_dir",
            "device",
            "torch_dtype",
            "offload_mode",
            *DISPATCH_LOAD_PARAMS,
        ),
        "default_load_kwargs": {"offload_mode": "block"},
        "default_call_kwargs": {
            "fps": 25,
            "num_frames": 204,
            "height": 544,
            "width": 992,
            "num_inference_steps": 50,
            "guidance_scale": 9.0,
            "time_shift": 13.0,
        },
        "aliases": ("stepvideo", "step-video"),
        "tags": ("text-to-video", "native-diffusion", "local-checkpoint"),
    },
    "skyreels-v2": {
        "display_name": "SkyReels-V2",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "SkyReels-V2-DF-1.3B-540P",
            "Skywork--SkyReels-V2-DF-1.3B-540P",
            "Skywork--SkyReels-V2-T2V-14B-720P",
            fallback="Skywork/SkyReels-V2-DF-1.3B-540P",
        ),
        "default_prompt": (
            "A woman in a leather jacket and sunglasses riding a vintage motorcycle through a desert highway "
            "at sunset, her hair blowing wildly in the wind as the motorcycle kicks up dust, with the golden sun "
            "casting long shadows across the barren landscape."
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_call_kwargs": {
            "num_frames": 97,
            "num_inference_steps": 30,
            "guidance_scale": 6.0,
            "shift": 8.0,
            "fps": 24,
            "seed": 42,
        },
        "call_params": (
            "prompt",
            "negative_prompt",
            "output_path",
            "return_dict",
            "fps",
            "num_frames",
            "height",
            "width",
            "num_inference_steps",
            "guidance_scale",
            "shift",
            "seed",
            "output_type",
        ),
        "load_params": (
            "model_path",
            "checkpoint_dir",
            "device",
            "torch_dtype",
            "offload_mode",
            *DISPATCH_LOAD_PARAMS,
        ),
        "default_load_kwargs": {"offload_mode": "block"},
        "aliases": ("skyreels2", "skyreels-v2-t2v"),
        "tags": ("text-to-video", "native-diffusion", "local-checkpoint"),
    },
    "helios": {
        "display_name": "Helios",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "BestWishYsh--Helios-Distilled",
            "Helios-Distilled",
            fallback="BestWishYsh/Helios-Distilled",
        ),
        "default_prompt": (
            "A cat and a dog baking a cake together in a cozy sunlit kitchen. The cat carefully measures flour "
            "while the dog stirs the batter with a wooden spoon, cinematic lighting, detailed natural motion."
        ),
        "default_task_type": "t2v",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_call_kwargs": {
            "num_frames": 240,
            "height": 384,
            "width": 640,
            "fps": 24,
            "num_inference_steps": 50,
            "guidance_scale": "auto",
            "seed": 42,
        },
        "call_params": (
            *_official_video_call_params(),
            "interactions",
            "realtime_segments",
            "pyramid_num_inference_steps_list",
        ),
        "stream_params": ("prompt", "interactions", "realtime_segments"),
        "load_params": _official_video_load_params(),
        "aliases": ("bestwishysh/helios",),
        "tags": (
            "text-to-video",
            "prompt-scheduled",
            "stream",
            "in-tree-runtime",
            "local-checkpoint",
        ),
        "notes": (
            "The resident Distilled runtime advances in native 33-frame segments and accepts prompt updates only "
            "at segment boundaries. Helios does not provide keyboard or camera controls."
        ),
    },
    "sana-streaming-2b-720p": {
        "display_name": "SANA-Streaming 2B 720p",
        "category": "Video Generation",
        "default_model_ref": lambda: _sana_streaming_default_ref(bidirectional=False),
        "default_prompt": (
            "A man in a dark suit sits at a table beside a bright window in a dim room, lowers his gaze, "
            "touches his forehead, and gestures with his hands; preserve his identity, the fixed camera, "
            "stable geometry, and consistent lighting."
        ),
        "default_task_type": "video-to-video",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_input_path": None,
        "default_load_kwargs": lambda: _sana_streaming_default_load_kwargs(bidirectional=False),
        "default_call_kwargs": {
            "num_frames": 81,
            "height": 704,
            "width": 1280,
            "fps": 16,
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "seed": 42,
            "num_cached_blocks": 2,
            "sink_token": True,
        },
        "call_params": (
            *_official_video_call_params(),
            "negative_prompt",
            "flow_shift",
            "motion_score",
            "num_cached_blocks",
            "sink_token",
        ),
        "load_params": _official_video_load_params(),
        "aliases": ("sana-streaming", "sana-streaming-long", "sana-streaming-720p"),
        "tags": ("video-to-video", "streaming", "state-cache", "official-runtime", "in-tree-runtime"),
    },
    "sana-streaming-bidirectional-2b-720p": {
        "display_name": "SANA-Streaming Bidirectional 2B 720p",
        "category": "Video Generation",
        "default_model_ref": lambda: _sana_streaming_default_ref(bidirectional=True),
        "default_prompt": (
            "A man in a dark suit sits at a table beside a bright window in a dim room, lowers his gaze, "
            "touches his forehead, and gestures with his hands; preserve his identity, the fixed camera, "
            "stable geometry, and consistent lighting."
        ),
        "default_task_type": "video-to-video",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_input_path": None,
        "default_load_kwargs": lambda: _sana_streaming_default_load_kwargs(bidirectional=True),
        "default_call_kwargs": {
            "num_frames": 81,
            "height": 704,
            "width": 1280,
            "fps": 16,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "seed": 42,
        },
        "call_params": (
            *_official_video_call_params(),
            "negative_prompt",
            "flow_shift",
            "motion_score",
            "num_cached_blocks",
            "sink_token",
        ),
        "load_params": _official_video_load_params(),
        "aliases": ("sana-streaming-bidirectional", "sana-streaming-short"),
        "tags": ("video-to-video", "bidirectional", "official-runtime", "in-tree-runtime"),
    },
    "bernini": {
        "display_name": "Bernini",
        "category": "Video Generation",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "ByteDance--Bernini-Diffusers",
            fallback="ByteDance/Bernini-Diffusers",
        ),
        "default_task_type": "text-to-video",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_load_kwargs": {"model_id": "bernini"},
        "default_call_kwargs": {
            "task_type": "t2v",
            "num_frames": 81,
            "height": 480,
            "width": 848,
            "num_inference_steps": 50,
            "fps": 16,
            "seed": 42,
            # The complete 81-frame/50-step profile fits one H100, while two
            # ranks duplicate roughly 179 GiB of host weights and leave too
            # little cgroup headroom for transient planner/renderer buffers.
            "nproc_per_node": 1,
            "ulysses_size": 1,
        },
        "call_params": (
            *_official_video_call_params(),
            "task_type",
            "neg_prompt",
            "max_image_size",
            "guidance_mode",
            "omega_vid",
            "omega_img",
            "omega_txt",
            "omega_tgt",
            "omega_scale",
            "planning_step",
            "vit_txt_cfg",
            "vit_img_cfg",
            "vit_denoising_step",
            "eta",
            "norm_threshold",
            "momentum",
            "system_prompt",
            "use_truncate",
            "max_sequence_length",
            "ulysses_size",
        ),
        "load_params": _official_video_load_params(),
        "aliases": (
            "bernini-diffusers",
            "bernini-7b-14b",
            "bernini-r-14b",
            "bernini-r-1.3b",
        ),
        "tags": ("text-to-video", "multi-task", "ulysses", "official-runtime", "in-tree-runtime"),
    },
    "wan-2p1-t2v": {
        "display_name": "Wan 2.1 T2V",
        "default_model_ref": _wan_2p1_t2v_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_load_kwargs": {"offload_mode": "block"},
        "default_call_kwargs": {
            "height": 480,
            "width": 832,
            "num_frames": 81,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "fps": 16,
            "seed": 42,
        },
        "call_params": _native_diffusion_video_call_params(),
        "load_params": _native_diffusion_video_load_params(),
        "aliases": ("wan2.1", "wan-2.1", "wan2p1", "wan2.1-t2v", "wan2p1-t2v"),
        "tags": ("text-to-video", "wan2.1", "native-diffusion", "local-checkpoint"),
    },
    "wan-2p1-i2v": {
        "display_name": "Wan 2.1 I2V",
        "default_model_ref": _wan_2p1_i2v_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_load_kwargs": {"offload_mode": "block"},
        "default_call_kwargs": {
            "height": 480,
            "width": 832,
            "num_frames": 81,
            "num_inference_steps": 40,
            "guidance_scale": 5.0,
            "fps": 16,
            "seed": 42,
        },
        "call_params": _native_diffusion_video_call_params(),
        "load_params": _native_diffusion_video_load_params(),
        "aliases": ("wan-2.1-i2v", "wan2p1-i2v", "wan2.1-i2v"),
        "tags": ("image-to-video", "wan2.1", "native-diffusion", "local-checkpoint"),
    },
    "wan2.2-t2v-a14b": {
        "display_name": "Wan2.2 T2V A14B",
        "default_model_ref": _wan_2p2_t2v_a14b_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_prompt": "A cinematic white cat runs through a sunlit field, realistic motion.",
        "default_load_kwargs": {"offload_mode": "block"},
        "default_call_kwargs": {
            "height": 480,
            "width": 832,
            "num_frames": 81,
            "num_inference_steps": 40,
            "guidance_scale": 4.0,
            "low_noise_guidance_scale": 3.0,
            "high_noise_guidance_scale": 4.0,
            "shift": 12.0,
            "fps": 16,
            "seed": 42,
        },
        "call_params": _native_diffusion_video_call_params()
        + ("low_noise_guidance_scale", "high_noise_guidance_scale"),
        "load_params": _native_diffusion_video_load_params(),
        "aliases": ("wan22-t2v-a14b", "Wan-AI/Wan2.2-T2V-A14B"),
        "tags": ("text-to-video", "wan2.2", "dual-expert", "native-diffusion"),
    },
    "wan2.2-i2v-a14b": {
        "display_name": "Wan2.2 I2V A14B",
        "default_model_ref": _wan_2p2_i2v_a14b_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "image-to-video",
        "default_prompt": "The subject turns toward the camera as the scene moves naturally.",
        "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_load_kwargs": {"offload_mode": "block"},
        "default_call_kwargs": {
            "height": 480,
            "width": 832,
            "num_frames": 81,
            "num_inference_steps": 40,
            "guidance_scale": 3.5,
            "low_noise_guidance_scale": 3.5,
            "high_noise_guidance_scale": 3.5,
            "shift": 5.0,
            "fps": 16,
            "seed": 42,
        },
        "call_params": _native_diffusion_video_call_params()
        + ("low_noise_guidance_scale", "high_noise_guidance_scale"),
        "load_params": _native_diffusion_video_load_params(),
        "aliases": ("wan22-i2v-a14b", "Wan-AI/Wan2.2-I2V-A14B"),
        "tags": ("image-to-video", "wan2.2", "dual-expert", "native-diffusion"),
    },
    "wan2.2-ti2v-5b": {
        "display_name": "Wan 2.2",
        "default_model_ref": _wan_2p2_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        # The 5B DiT fits comfortably on the Studio target H100.  Keep it
        # resident by default so the browser path does not pay one H2D copy
        # per transformer layer.  Lower-memory deployments can explicitly
        # select ``async-block``/``block`` and receive audited prefetch.
        "default_load_kwargs": {
            "offload_mode": "resident",
            # Exact production profile established by the Wan2.2 TI2V
            # framework A/B harness. Every acceleration remains explicitly
            # overrideable from the Studio runtime controls/load JSON.
            "fuse_qkv": False,
            "qkv_strategy": "auto",
            "qkv_split_threshold": 8192,
            "inplace_residual": False,
            "static_cross_kv": True,
            "fused_rope": False,
            "rope_precision": "fp32",
            "rms_norm_precision": "input",
        },
        "default_call_kwargs": {
            "height": 704,
            "width": 1280,
            "num_frames": 121,
            "num_inference_steps": 50,
            "guidance_scale": 5.0,
            "fps": 24,
            "seed": 42,
        },
        "suggested_task_types": ("ti2v-5b", "interactive-video"),
        "call_params": _native_diffusion_video_call_params(),
        "load_params": _native_diffusion_video_load_params(),
        "aliases": ("wan2.2", "wan-2.2", "wan2p2"),
        "tags": ("video", "image-to-video", "text-to-video", "native-diffusion"),
    },
    "echo-infinity": {
        "display_name": "Echo-Infinity",
        "default_model_ref": _echo_infinity_default_ref,
        "default_load_kwargs": _echo_infinity_default_load_kwargs,
        "default_call_kwargs": {"fps": 16},
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "lazy",
            "model_name",
            "wan_root",
            "generator_ckpt",
            "wan_model_name",
            "lora_ckpt",
            "adapter",
            "use_ema",
            "seed",
            "num_samples",
            "num_output_frames",
            "denoising_step_list",
            "warp_denoising_step",
            "num_frame_per_block",
            "context_noise",
            "global_sink",
            "model_kwargs",
            "memory_kwargs",
            "dtype",
            "low_memory",
        ),
        "tags": ("text-to-video", "in-tree-runtime"),
    },
    "echo-memory-context-k1": {
        "display_name": "Echo-Memory Context K=1",
        "category": "Video Generation",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _echo_memory_context_k1_default_ref,
        "default_load_kwargs": _echo_memory_context_k1_load_kwargs,
        "default_task_type": "i2v",
        "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_prompt": (
            "A young man holding a bright sparkler gently turns his head as sparks scatter through the night air; "
            "preserve his identity, stable architecture, coherent lighting, and smooth forward camera motion."
        ),
        "default_call_kwargs": {
            "num_frames": 81,
            "fps": 15,
            "width": 640,
            "height": 352,
            "num_chunks": 2,
            "steps": 50,
            "guidance_scale": 5.0,
            "seed": 42,
            "camera_trajectory": "z*80",
            "return_dict": True,
        },
        "tags": ("image-to-video", "memory", "native-diffusion", "public-checkpoint"),
    },
    "evoke": {
        "display_name": "Evoke",
        "category": "Video Generation",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _evoke_default_ref,
        "default_load_kwargs": _evoke_default_load_kwargs,
        "default_prompt": (
            "A cinematic camera glides through a sunlit forest trail while leaves move gently "
            "and the scene geometry remains stable."
        ),
        "default_call_kwargs": {
            "mode": "t2v",
            "width": 640,
            "height": 384,
            "num_frames": 33,
            "num_chunks": 1,
            "fps": 24,
            "num_inference_steps": 3,
            "guidance_scale": 1.0,
            "seed": 42,
        },
        "call_params": (
            "prompt",
            "images",
            "image_path",
            "video",
            "video_path",
            "output_path",
            "return_dict",
            "trajectory_npz",
            "pose_path",
            "camera_trajectory",
            "mode",
            "sample_type",
            "use_geometric_state",
            "width",
            "height",
            "num_chunks",
            "num_frames",
            "fps",
            "seed",
            "num_inference_steps",
            "guidance_scale",
            "ref_seconds",
            "start_seconds",
            "lingbot_pose_source_fps",
            "lingbot_pose_source_resolution",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "checkpoint_path",
            "base_model_path",
            "transformer_path",
            "vigeo_path",
            "runtime_root",
            "python_executable",
            "device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("alayalab/evoke", "evoke-world-model"),
        "tags": ("world-model", "text-to-video", "image-to-video", "video-to-video", "in-tree-runtime"),
        "notes": "Uses the local post-distillation Evoke checkpoint and the shared WorldFoundry Python environment; camera-trajectory runs additionally use local ViGeo1.1 weights.",
    },
    "recammaster": {
        "display_name": "ReCamMaster",
        "category": "Video-to-Video",
        "default_prompt": (
            "A first-person motorcycle ride along an elevated highway bridge, preserving "
            "the rider, road, guardrails, and distant landscape with smooth camera motion."
        ),
        "default_model_ref": _recammaster_default_ref,
        "default_load_kwargs": _recammaster_default_load_kwargs,
        "default_task_type": "video-to-video",
        "default_interactions": ("camera_path",),
        "default_call_kwargs": {
            "camera_trajectory": [100, 100, 0, 0, 30],
            "num_frames": 81,
            "num_inference_steps": 50,
            "cfg_scale": 5.0,
            "fps": 8,
            "video_path": "worldfoundry/data/test_cases/longcat_video/motorcycle.mp4",
        },
        "call_params": (
            "prompt",
            "video_path",
            "output_path",
            "camera_trajectory",
            "num_frames",
            "num_inference_steps",
            "cfg_scale",
            "fps",
            "max_num_frames",
            "frame_interval",
            "size",
        ),
        "tags": ("video-to-video", "camera-control", "official-runtime"),
    },
    "cosmos-predict2": {
        "display_name": "Cosmos Predict2 2B Video2World",
        "default_model_ref": _cosmos_predict2_default_ref,
        "default_load_kwargs": _cosmos_predict2_default_load_kwargs,
        "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_prompt": (
            "A young man holding a bright sparkler gently turns his head as sparks scatter through the night air, "
            "with coherent motion, stable facial details, and consistent nighttime architecture."
        ),
        "default_task_type": "image-to-world",
        "default_call_kwargs": {
            "guidance_scale": 7.0,
            "num_inference_steps": 35,
            "fps": 16,
            "num_frames": 93,
            "height": 704,
            "width": 1280,
            "seed": 0,
            "num_latent_conditional_frames": 1,
            "output_type": "video",
            "return_dict": True,
        },
        "tags": ("image-to-world", "video-to-world", "cosmos", "native-diffusion"),
    },
    "cosmos-predict2.5": {
        "display_name": "Cosmos Predict2.5",
        "default_model_ref": _cosmos_predict2p5_default_ref,
        "default_load_kwargs": _cosmos_predict2p5_default_load_kwargs,
        "default_prompt": (
            "A nighttime city bus terminal gradually shifts from stillness to subtle movement, "
            "with realistic lighting, stable geometry, and smooth camera motion."
        ),
        "default_task_type": "text-to-world",
        "default_call_kwargs": _cosmos_predict2p5_default_call_kwargs,
        "aliases": ("cosmos-predict2.5", "cosmos-predict-2.5"),
        "tags": ("text-to-world", "image-to-world", "video", "cosmos"),
    },
    "cosmos-transfer-2.5": {
        "display_name": "Cosmos Transfer2.5",
        "default_model_ref": _cosmos_transfer2p5_default_ref,
        "default_load_kwargs": _cosmos_transfer2p5_default_load_kwargs,
        "default_input_path": str(
            _data_path("test_cases", "longcat_video", "motorcycle.mp4")
        ),
        "default_prompt": (
            "A cinematic night scene follows the control video edges with stable geometry, "
            "realistic lighting, and smooth coherent motion."
        ),
        "default_task_type": "video-to-world",
        "default_call_kwargs": _cosmos_transfer2p5_default_call_kwargs,
        "aliases": ("cosmos-transfer2.5", "cosmos-transfer2p5"),
        "tags": ("video-to-world", "controlled-video", "edge-control", "cosmos"),
    },
    "cosmos3": {
        "display_name": "Cosmos3",
        "default_model_ref": lambda: _hf_checkpoint_model_ref_at_revision(
            "Cosmos3-Nano",
            "nvidia/Cosmos3-Nano",
            "411f42a8fdfb8c5b2583cb8786e0938f49796eaa",
        ),
        "default_load_kwargs": {
            "revision": "411f42a8fdfb8c5b2583cb8786e0938f49796eaa",
            "load_sound_tokenizer": True,
        },
        "default_prompt": "A robot arm is cleaning a plate in the kitchen",
        "default_task_type": "text-to-video",
        "default_call_kwargs": {
            "fps": 24,
            "guidance_scale": 6.0,
            "height": 720,
            "num_inference_steps": 35,
            "num_frames": 189,
            "output_type": "video",
            "seed": 0,
            "width": 1280,
        },
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "torch_dtype",
            "device_map",
            "enable_safety_checker",
            "load_sound_tokenizer",
            "revision",
        ),
        "extra_variants": (
            {
                "variant_id": "cosmos3-super",
                "label": "Cosmos 3 Super",
                "status": "checkpoint-required",
                "aliases": ("cosmos-3-super",),
                "checkpoints": (
                    {
                        "role": "primary",
                        "uri": lambda: _hf_checkpoint_model_ref_at_revision(
                            "Cosmos3-Super",
                            "nvidia/Cosmos3-Super",
                            "e0262be9d8f7586bc24c069a2aed2b665bdff266",
                        ),
                        "required": True,
                        "status": "official",
                    },
                ),
                "load_kwargs": {
                    "variant_id": "cosmos3-super",
                    "profile_id": "cosmos3-super",
                    "revision": "e0262be9d8f7586bc24c069a2aed2b665bdff266",
                    "runtime_profile": "cosmos3-super",
                    "device_map": "balanced",
                    "load_sound_tokenizer": True,
                },
                "notes": (
                    "Requires a high-memory multi-GPU configuration; the default safe map shards decoder layers only.",
                ),
            },
        ),
        "aliases": ("cosmos3-nano", "cosmos3-super", "cosmos-3", "cosmos-3-nano", "cosmos-3-super"),
        "tags": (
            "text-to-image",
            "text-to-video",
            "image-to-video",
            "video-to-video",
            "action-policy",
            "forward-dynamics",
            "inverse-dynamics",
            "world-generation",
            "video",
            "cosmos",
        ),
        "notes": "Uses the in-tree official Diffusers generator. Nano is the default; Super is selectable as a separate variant. Exact-revision Nano task-matrix and safety-enabled smokes plus a four-A100 Super T2I smoke passed; official-quality/full-resolution parity remains pending.",
    },
    "lingbot-video": {
        "display_name": "LingBot-Video",
        "default_model_ref": _lingbot_video_default_ref,
        "default_load_kwargs": {"variant": "dense"},
        "default_prompt": (
            '{"comprehensive_description":"A worker carefully places a red block into a matching tray on a '
            'clean workbench while the camera remains stable.","camera_info":{"frame_size":"Medium Shot",'
            '"shot_type_angle":"Eye Level","lighting_type":"Daylight"},"world_knowledge":[]}'
        ),
        "default_task_type": "t2v",
        "default_call_kwargs": {
            "mode": "t2v",
            "backend": "diffusers",
            "height": 480,
            "width": 832,
            "num_frames": 121,
            "num_inference_steps": 40,
            "guidance_scale": 3.0,
            "shift": 3.0,
            "seed": 42,
            "fps": 24,
            "execute": True,
            "timeout_seconds": 7200,
        },
        "load_params": ("variant", "python_executable", *DISPATCH_LOAD_PARAMS),
        "call_params": (
            "prompt",
            "prompt_json",
            "images",
            "negative_prompt",
            "negative_prompt_json",
            "mode",
            "backend",
            "height",
            "width",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "shift",
            "seed",
            "fps",
            "batch_cfg",
            "run_refiner",
            "reuse_condition_features",
            "refiner_height",
            "refiner_width",
            "refiner_steps",
            "refiner_guidance_scale",
            "refiner_shift",
            "cfg_parallel_degree",
            "context_parallel_degree",
            "nproc_per_node",
            "enable_fsdp_inference",
            "execute",
            "timeout_seconds",
            "return_dict",
            "output_dir",
            "output_path",
        ),
        "aliases": ("lingbot-video-dense", "lingbot-video-moe"),
        "tags": ("text-to-video", "image-to-video", "text-to-image", "official-runtime"),
        "notes": "Inference-only in-tree runtime. Structured JSON captions are required for release-quality generation.",
    },
    "longcat-video": {
        "display_name": "LongCat-Video",
        "default_model_ref": _longcat_video_default_ref,
        "default_load_kwargs": _longcat_video_default_load_kwargs,
        "default_call_kwargs": {
            "task_type": "t2v",
            "cpu_offload": True,
        },
        "load_params": ("python_executable", *DISPATCH_LOAD_PARAMS),
        "call_params": (
            "prompt",
            "negative_prompt",
            "task_type",
            "height",
            "width",
            "num_frames",
            "num_inference_steps",
            "distill_num_inference_steps",
            "guidance_scale",
            "seed",
            "fps",
            "context_parallel_size",
            "enable_compile",
            "cpu_offload",
            "run_base",
            "run_distill",
            "run_refiner",
            "max_sequence_length",
            "execute",
            "timeout_seconds",
            "return_dict",
            "output_dir",
            "output_path",
        ),
        "tags": ("text-to-video", "official-runtime"),
    },
    "pusa-vidgen": {
        "display_name": "Pusa VidGen",
        "default_model_ref": _pusa_vidgen_default_ref,
        "default_load_kwargs": _pusa_vidgen_default_load_kwargs,
        "default_call_kwargs": {
            "mode": "t2v",
            "height": 720,
            "width": 1280,
            "num_frames": 81,
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "lightx2v": True,
            "high_lora_alpha": 1.5,
            "low_lora_alpha": 1.4,
            "seed": 0,
            "execute": True,
        },
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "output_dir",
            "return_dict",
            "task_type",
            "mode",
            "height",
            "width",
            "resolution",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "cfg_scale",
            "lightx2v",
            "high_lora_alpha",
            "low_lora_alpha",
            "cond_position",
            "noise_multipliers",
            "switch_dit_boundary",
            "fps",
            "quality",
            "negative_prompt",
            "seed",
            "execute",
            "timeout_seconds",
        ),
        "tags": ("text-to-video", "image-to-video", "official-runtime"),
    },
    "sana": {
        "display_name": "Sana",
        "default_model_ref": _sana_image_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "text-to-image",
        "default_prompt": (
            "A photorealistic still of a sunlit ceramic bowl of citrus on a wooden table, "
            "sharp focus, natural lighting."
        ),
        "default_load_kwargs": {"model_id": SANA_DEFAULT_IMAGE_VARIANT_ID},
        "default_call_kwargs": {
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 20,
            "guidance_scale": 4.5,
            "seed": 42,
        },
        "load_params": _sana_video_load_params(),
        "extra_variants": _sana_image_workspace_variants,
        "aliases": _sana_image_aliases(),
        "tags": ("text-to-image", "controlnet", "sprint", "official-runtime", "local-checkpoint"),
        "notes": (
            "Image, Sprint, and ControlNet checkpoints share this Studio card. "
            "Pick a variant for size, resolution, few-step Sprint, or HED ControlNet. "
            "Sana Video, LongSANA, and SANA-Streaming stay on their own cards."
        ),
        "summary": (
            "Sana text-to-image family. Select a checkpoint variant for parameter "
            "size, resolution, Sprint, or ControlNet."
        ),
    },
    "cogvideox_5b_t2v": {
        "display_name": "CogVideoX",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "t2v",
        "default_prompt": (
            "A futuristic city street at sunset with reflective glass towers, clean motion, and cinematic lighting."
        ),
        "default_call_kwargs": {
            "fps": 8,
            "height": 480,
            "num_frames": 49,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "seed": 43,
            "width": 720,
        },
        "default_model_ref": lambda: _checkpoint_model_ref(
            "CogVideoX-5b",
            "THUDM--CogVideoX-5b",
            fallback="THUDM/CogVideoX-5b",
        ),
        "extra_variants": _cogvideox_workspace_variants,
        "aliases": _cogvideox_aliases(),
        "tags": ("video", "t2v", "i2v", "official-runtime", "local-checkpoint"),
        "notes": (
            "2B T2V, 5B T2V, and 5B I2V share this Studio card. "
            "Pick a variant for size and text-to-video vs image-to-video."
        ),
        "summary": (
            "CogVideoX text-to-video and image-to-video family. "
            "Select 2B T2V, 5B T2V, or 5B I2V."
        ),
    },
    "sana-video-2b-480p": {
        "display_name": "Sana Video 2B 480p",
        "default_model_ref": _sana_video_480p_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "t2v",
        "default_prompt": "A cinematic tiger walks through a misty green forest, photorealistic, detailed lighting.",
        "default_call_kwargs": {
            "num_frames": 81,
            "height": 480,
            "width": 832,
            "fps": 16,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "seed": 42,
        },
        "call_params": _sana_video_call_params(),
        "stream_params": _sana_video_call_params(),
        "load_params": _sana_video_load_params(),
        "aliases": ("sana-video", "sana-video-480p"),
        "tags": ("text-to-video", "official-runtime", "local-checkpoint"),
    },
    "sana-video-2b-720p": {
        "display_name": "Sana Video 2B 720p",
        "default_model_ref": _sana_video_720p_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "t2v",
        "default_prompt": "A cinematic tiger walks through a misty green forest, photorealistic, detailed lighting.",
        "default_call_kwargs": {
            "num_frames": 81,
            "height": 720,
            "width": 1280,
            "fps": 16,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "seed": 42,
        },
        "call_params": _sana_video_call_params(),
        "stream_params": _sana_video_call_params(),
        "load_params": _sana_video_load_params(),
        "aliases": ("sana-video-720p",),
        "tags": ("text-to-video", "official-runtime", "local-checkpoint"),
    },
    "longsana-video-2b-480p": {
        "display_name": "LongSANA Video 2B 480p",
        "default_model_ref": _longsana_video_480p_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "t2v",
        "default_prompt": (
            "Evening backlight and soft side lighting in a forest. A young man in a light shirt stands among the "
            "trees while golden sunlight filters through the leaves; wind gently moves his hair and collar, the "
            "camera holds a clean emotional mid-shot with a softly blurred background."
        ),
        "default_call_kwargs": {
            "negative_prompt": (
                "Bright tones, overexposed, static, blurred details, subtitles, paintings, overall gray, worst "
                "quality, low quality, JPEG artifacts, deformed limbs, fused fingers, messy background, walking backwards"
            ),
            "num_frames": 161,
            "height": 480,
            "width": 832,
            "num_inference_steps": 50,
            "guidance_scale": 1.0,
            "fps": 16,
            "seed": 42,
        },
        "call_params": _longsana_video_call_params(),
        "stream_params": _longsana_video_call_params(),
        "load_params": _sana_video_load_params(),
        "aliases": ("longsana-video", "sana-video-longlive"),
        "tags": ("text-to-video", "official-runtime", "local-checkpoint"),
    },
    "wan2.1-t2v-14b": {
        "display_name": "Wan2.1 T2V 14B",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "Wan2.1-T2V-14B",
            "Wan-AI--Wan2.1-T2V-14B",
            fallback="Wan-AI/Wan2.1-T2V-14B",
        ),
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "t2v",
        "default_prompt": (
            "Two anthropomorphic cats in comfortable boxing gear and bright gloves fight intensely on a "
            "spotlighted stage, cinematic motion, detailed fur, stable anatomy."
        ),
        "default_call_kwargs": {
            "num_frames": 81,
            "height": 720,
            "width": 1280,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "shift": 5.0,
            "fps": 16,
            "seed": 42,
        },
        "call_params": _native_diffusion_video_call_params(),
        "load_params": _native_diffusion_video_load_params(),
        "aliases": ("wan2.1-t2v-14b-720p",),
        "tags": ("text-to-video", "native-diffusion", "local-checkpoint", "720p"),
    },
    "wan2.1-i2v-14b-720p": {
        "display_name": "Wan2.1 I2V 14B 720P",
        "default_model_ref": _wan_2p1_i2v_720p_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "image-to-video",
        "default_prompt": "The subject gently turns toward the camera while the scene moves naturally.",
        "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_load_kwargs": {"offload_mode": "block"},
        "default_call_kwargs": {
            "num_frames": 81,
            "height": 720,
            "width": 1280,
            "num_inference_steps": 40,
            "guidance_scale": 5.0,
            "shift": 5.0,
            "fps": 16,
            "seed": 42,
        },
        "call_params": _native_diffusion_video_call_params(),
        "load_params": _native_diffusion_video_load_params(),
        "aliases": ("Wan-AI/Wan2.1-I2V-14B-720P",),
        "tags": ("image-to-video", "native-diffusion", "local-checkpoint", "720p"),
    },
    "zeroscope": {
        "display_name": "ZeroScope",
        "default_model_ref": _zeroscope_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "t2v",
        "default_prompt": "research rover red desert under dusty sunset, slow cinematic pan, stable vehicle geometry",
        "default_call_kwargs": {
            "num_frames": 24,
            "fps": 8,
            "height": 320,
            "width": 576,
            "num_inference_steps": 40,
            "guidance_scale": 9.0,
            "seed": 301,
        },
        "default_load_kwargs": lambda: {"model_path": _zeroscope_default_ref()},
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "num_frames",
            "infer_steps",
            "num_inference_steps",
            "cfg_scale",
            "guidance_scale",
            "height",
            "width",
            "seed",
        ),
        "load_params": (
            "model_path",
            "pretrained_model_path",
            "required_components",
            "device",
            "model_id",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": ("text-to-video", "diffusers", "local-checkpoint"),
    },
    "animatediff": {
        "display_name": "AnimateDiff",
        "default_model_ref": _animatediff_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "t2v",
        "default_prompt": "snowy street market at night with lanterns, cinematic camera movement, detailed people silhouettes",
        "default_call_kwargs": {
            "num_frames": 16,
            "fps": 8,
            "height": 256,
            "width": 256,
            "num_inference_steps": 25,
            "guidance_scale": 8.0,
            "seed": 303,
        },
        "default_load_kwargs": _animatediff_default_load_kwargs,
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "official_config_path",
            "config_path",
            "config",
            "timeout_seconds",
            "negative_prompt",
            "num_frames",
            "video_length",
            "infer_steps",
            "num_inference_steps",
            "cfg_scale",
            "guidance_scale",
            "height",
            "width",
            "seed",
            "allow_validation_fallback",
        ),
        "load_params": (
            "model_path",
            "motion_adapter_path",
            "motion_module_path",
            "motion_module",
            "base_model_path",
            "sd15_path",
            "dreambooth_model_path",
            "dreambooth_path",
            "official_python",
            "hf_hub_cache",
            "integrated_runtime_root",
            "inference_config",
            "negative_prompt",
            "required_components",
            "device",
            "model_id",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": ("text-to-video", "official-runtime", "local-checkpoint"),
    },
    "open-magvit2": {
        "display_name": "Open-MAGVIT2",
        "default_model_ref": _open_magvit2_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "class-conditional-image-generation",
        "default_load_kwargs": lambda: {
            "checkpoint_path": _open_magvit2_default_ref(),
            "config_path": "imagenet_conditional_llama_L.yaml",
        },
        "default_call_kwargs": {"class_id": 207, "batch_size": 1, "steps": 256},
        "call_params": (
            "prompt",
            "images",
            "video",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "class_id",
            "batch_size",
            "steps",
            "temperature",
            "top_k",
            "top_p",
            "cfg_scale",
        ),
        "load_params": (
            "model_path",
            "checkpoint_path",
            "ckpt_path",
            "config_path",
            "config",
            "class_id",
            "batch_size",
            "required_components",
            "device",
            "model_id",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": ("image-generation", "in-tree-runtime", "local-checkpoint"),
    },
    "skyreels-v3": {
        "display_name": "SkyReels V3",
        "default_model_ref": _skyreels_v3_default_ref,
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_task_type": "reference_to_video",
        "default_prompt": (
            "A young man holds a bright sparkler at night while the camera moves slowly around him, "
            "preserving his identity, pose, and the reference scene."
        ),
        "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_call_kwargs": {
            "task_type": "reference_to_video",
            "duration": 5,
            "resolution": "720P",
            "seed": 42,
        },
        "call_params": (
            "prompt",
            "images",
            "output_path",
            "return_dict",
            "task_type",
            "duration",
            "seed",
            "resolution",
            "height",
            "width",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "image_guidance_scale",
            "negative_prompt",
            "fps",
            "output_type",
            "shift",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("skyreels-v3-r2v", "skyreels-v3-reference-to-video"),
        "tags": ("reference-to-video", "native-diffusion", "local-checkpoint"),
    },
    "kairos-sensenova": {
        "display_name": "Kairos Sensenova",
        "default_model_ref": _kairos_sensenova_default_ref,
        "default_load_kwargs": _kairos_sensenova_default_load_kwargs,
        "default_prompt": (
            "A grand natural scene unfolds in an open landscape, beginning with a wide, distant view of a massive "
            "waterfall cascading down a towering cliff that dominates the frame, its immense scale emphasized by the "
            "surrounding rock formations and mist-filled air. From the cliff's edge, enormous volumes of water surge "
            "downward in continuous streams, breaking into layered sheets and turbulent ribbons as gravity accelerates "
            "the flow, while countless fine droplets are torn from the main body of water and dispersed into the air, "
            "forming a drifting veil of mist that catches the sunlight and shimmers faintly. The waterfall's surface "
            "exhibits complex motion, with faster central currents plunging straight down and thinner side streams "
            "clinging briefly to the rock face before peeling away, creating visible variations in speed, thickness, "
            "and texture. As the camera slowly advances forward and downward, the perspective tightens, bringing the "
            "lower section of the waterfall into clearer view, where the falling water collides violently with the "
            "surface below, generating bursts of white foam and outward-splashing sprays. The camera continues to move "
            "closer, transitioning into a medium and then near-field view of the plunge pool at the base of the "
            "waterfall, revealing a circular water basin whose surface churns constantly under the impact. Within the "
            "pool, swirling currents rotate in overlapping patterns, with darker, deeper green-blue water visible "
            "beneath the frothy white surface, and concentric ripples propagate outward toward the rocky edges. "
            "Individual droplets and splashes are now discernible, arcing briefly through the air before rejoining the "
            "pool, while fine mist rises continuously and drifts laterally with subtle air movement. Wet rock surfaces "
            "around the pool glisten under natural light, reflecting soft highlights that shift with the camera's "
            "motion. Throughout the shot, the camera movement remains smooth and deliberate, steadily closing the "
            "distance from the monumental wide shot to an intimate, detailed view of the water's surface, emphasizing "
            "the physical continuity of flowing water, gravity-driven motion, and the dynamic interaction between "
            "falling streams, airborne droplets, and the turbulent pool below."
        ),
        "default_call_kwargs": {
            "negative_prompt": (
                "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
                "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra "
                "fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, deformed limbs, fused fingers, "
                "still picture, messy background, three legs, many people in the background, walking backwards, "
                "contorted human joints, objects floating against natural forces, abrupt shot changes"
            ),
            "seed": 0,
            "tiled": True,
            "height": 480,
            "width": 640,
            "num_frames": 81,
            "cfg_scale": 5,
            "use_prompt_rewriter": False,
            "nproc_per_node": 1,
        },
        "call_params": (
            "prompt",
            "output_path",
            "negative_prompt",
            "seed",
            "tiled",
            "height",
            "width",
            "num_frames",
            "cfg_scale",
            "use_prompt_rewriter",
            "save_fps",
            "nproc_per_node",
            "master_port",
            "run_manage_libs",
        ),
        "tags": ("robot-video", "official-runtime"),
    },
    "hydra": {
        "display_name": "HyDRA",
        "default_model_ref": _hydra_default_ref,
        "default_input_path": str(_data_path("test_cases", "hydra", "condition.mp4")),
        "default_prompt": (
            "The video begins with a close-up of an individual clad in black armor with red and yellow lights, "
            "walking toward the camera on a city street."
        ),
        "default_load_kwargs": _hydra_default_load_kwargs,
        "default_call_kwargs": lambda: {
            "video_path": str(_data_path("test_cases", "hydra", "condition.mp4")),
            "camera_json": str(_data_path("test_cases", "hydra", "camera.json")),
            "num_frames": 77,
            "fps": 15,
            "width": 832,
            "height": 480,
            "num_inference_steps": 50,
            "seed": 42,
        },
        "call_params": (
            "prompt", "video", "video_path", "camera_json", "output_path", "num_frames",
            "fps", "width", "height", "num_inference_steps", "seed", "return_dict",
        ),
        "load_params": (
            "model_path", "checkpoint_path", "base_model_path", "python_executable", "device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": ("video-to-video", "camera-control", "official-runtime"),
    },
    "minwm-hy-action2v": {
        "display_name": "minWM HY Action2V",
        "default_model_ref": _minwm_default_ref,
        "default_input_path": str(_data_path("test_cases", "minwm", "first_frame.png")),
        "default_prompt": (
            "A rain-soaked brick path runs beside a long wooden aviary building as the camera "
            "moves smoothly forward, with wet reflections and stable geometry."
        ),
        "default_load_kwargs": _minwm_hy_default_load_kwargs,
        "default_call_kwargs": {
            "trajectory": "a*4,w*8,s*7",
            "num_frames": 77,
            "num_inference_steps": 4,
            "fps": 16,
            "width": 832,
            "height": 480,
            "seed": 0,
        },
        "call_params": (
            "prompt", "images", "image_path", "trajectory", "output_path", "num_frames",
            "num_inference_steps", "fps", "width", "height", "seed", "return_dict",
        ),
        "load_params": (
            "model_path", "checkpoint_path", "base_model_path", "python_executable", "device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": ("world-model", "action-conditioned-video", "camera-control"),
    },
    "minwm-wan-action2v": {
        "display_name": "minWM Wan Action2V",
        "default_model_ref": _minwm_default_ref,
        "default_prompt": "A serene garden path winds through manicured greenery under a soft overcast sky.",
        "default_load_kwargs": _minwm_wan_default_load_kwargs,
        "default_call_kwargs": {
            "trajectory": "a*4,w*8,s*7",
            "num_output_frames": 20,
            "seed": 0,
        },
        "call_params": (
            "prompt", "trajectory", "output_path", "num_output_frames", "seed", "sp_size",
            "master_port", "return_dict",
        ),
        "load_params": (
            "model_path", "checkpoint_path", "base_model_path", "python_executable", "device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": ("world-model", "action-conditioned-video", "camera-control"),
    },
    "magicworld": {
        "display_name": "MagicWorld",
        "default_model_ref": _magicworld_default_ref,
        "default_input_path": str(_data_path("test_cases", "minwm", "first_frame.png")),
        "default_prompt": (
            "A rain-soaked brick path beside a long wooden aviary building, viewed with a "
            "smooth forward camera move and stable wet-weather geometry."
        ),
        "default_load_kwargs": _magicworld_default_load_kwargs,
        "default_call_kwargs": lambda: {
            "native_rows": str(
                _data_path(
                    "benchmarks", "assets", "iworld-bench", "camera_trajectories",
                    "inference_txt", "camera_1_1_0.txt",
                )
            ),
            "num_frames": 81,
            "seed": 42,
        },
        "call_params": (
            "prompt", "images", "image_path", "native_rows", "output_path", "num_frames", "seed",
            "return_dict",
        ),
        "load_params": (
            "model_path", "checkpoint_path", "base_model_path", "python_executable", "device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": ("world-model", "image-to-video", "camera-control"),
    },
    "videocrafter1-i2v": {
        "display_name": "VideoCrafter1 I2V",
        "category": "Video Generation",
        "summary": "In-tree VideoCrafter1 image-to-video inference runtime.",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_backend": "from_pretrained",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "VideoCrafter--Image2Video-512/model.ckpt",
            "VideoCrafter/videocrafter_i2v_512_v1.ckpt",
            "video_models/videocrafter_i2v_512_v1.ckpt",
        ),
        "default_prompt": "horses are walking on the grassland",
        "default_input_path": str(_data_path("test_cases", "videocrafter", "i2v_prompts", "horse.png")),
        "default_call_kwargs": {
            "num_frames": 16,
            "fps": 8,
            "num_inference_steps": 50,
            "height": 320,
            "width": 512,
            "seed": 123,
        },
        "call_params": (
            "prompt", "images", "output_path", "fps", "return_dict", "frames", "num_frames",
            "frame_num", "ddim_steps", "num_inference_steps", "steps", "sample_steps", "infer_steps",
            "sampling_steps", "ddim_eta", "height", "width", "seed", "unconditional_guidance_scale",
        ),
        "load_params": ("model_path", "required_components", "device", "lazy"),
        "aliases": ("videocrafter1_i2v",),
        "tags": ("video-generation", "image-to-video", "official-runtime"),
    },
    "videocrafter1-t2v": {
        "display_name": "VideoCrafter1 T2V",
        "category": "Video Generation",
        "summary": "In-tree VideoCrafter1 text-to-video inference runtime.",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_backend": "from_pretrained",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "VideoCrafter--Text2Video-1024/model.ckpt",
            "VideoCrafter/videocrafter_t2v_1024_v1.ckpt",
            "video_models/videocrafter_t2v_1024_v1.ckpt",
        ),
        "default_prompt": "A tiger walks in the forest, photorealistic, 4k, high definition",
        "default_call_kwargs": {
            "num_frames": 16,
            "fps": 8,
            "num_inference_steps": 50,
            "height": 576,
            "width": 1024,
            "seed": 123,
        },
        "call_params": (
            "prompt", "output_path", "fps", "return_dict", "frames", "num_frames", "frame_num",
            "ddim_steps", "num_inference_steps", "steps", "sample_steps", "infer_steps", "sampling_steps",
            "ddim_eta", "height", "width", "seed", "unconditional_guidance_scale",
        ),
        "load_params": ("model_path", "required_components", "device", "lazy"),
        "aliases": ("videocrafter1_t2v",),
        "tags": ("video-generation", "text-to-video", "official-runtime"),
    },
    "videocrafter2-t2v": {
        "display_name": "VideoCrafter2 T2V",
        "category": "Video Generation",
        "summary": "In-tree VideoCrafter2 text-to-video inference runtime.",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_backend": "from_pretrained",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "VideoCrafter--VideoCrafter2/model.ckpt",
            "VideoCrafter/videocrafter_t2v_512_v2.ckpt",
            "video_models/videocrafter_t2v_512_v2.ckpt",
        ),
        "default_prompt": "A tiger walks in the forest, photorealistic, 4k, high definition",
        "default_call_kwargs": {
            "num_frames": 16,
            "fps": 28,
            "num_inference_steps": 50,
            "height": 320,
            "width": 512,
            "seed": 123,
        },
        "call_params": (
            "prompt", "output_path", "fps", "return_dict", "frames", "num_frames", "frame_num",
            "ddim_steps", "num_inference_steps", "steps", "sample_steps", "infer_steps", "sampling_steps",
            "ddim_eta", "height", "width", "seed", "unconditional_guidance_scale",
        ),
        "load_params": ("model_path", "required_components", "device", "lazy"),
        "aliases": ("videocrafter", "videocrafter2_t2v"),
        "tags": ("video-generation", "text-to-video", "official-runtime"),
    },
    "gamma-world": {
        "display_name": "Gamma-World",
        "default_input_path": str(
            _data_path("test_cases", "gamma_world", "buildTower_normal")
        ),
        "default_prompt": "Two Minecraft players buildTower in normal world",
        "default_load_kwargs": _gamma_world_default_load_kwargs,
        "default_call_kwargs": {
            "n_players": 2,
            "num_frames": 189,
            "num_conditional_frames": 1,
            "height": 320,
            "width": 480,
            "guidance": 5.0,
            "seed": 1,
            "fps": 16,
        },
        "call_params": (
            "prompt", "images", "video", "interactions", "guidance", "guidance_scale",
            "actions", "action_paths", "eval_dir", "n_players", "num_frames", "frames",
            "num_conditional_frames", "height", "width", "num_inference_steps", "infer_steps",
            "negative_prompt", "seed", "fps", "output_path", "return_dict",
        ),
        "load_params": (
            "model_path", "required_components", "device", "mode", "checkpoint_overrides",
            "text_encoder_path", "reason1_path", "torch_dtype", "weight_dtype", "dtype", "offload_mode",
        ),
        "tags": ("multi-agent-world-model", "action-conditioned-video", "official-runtime"),
    },
    "solaris": {
        "display_name": "Solaris",
        "default_model_ref": _solaris_default_ref,
        "default_load_kwargs": _solaris_default_load_kwargs,
        # Leave eval_num_samples unset so Solaris derives it inside the leased
        # worker from CUDA_VISIBLE_DEVICES. Hard-coding the host's eight-sample
        # upstream batch on a one-GPU Workspace lease exhausts the card during
        # the first 257-frame evaluation step.
        "default_call_kwargs": {"eval_types": "translation"},
        "call_params": (
            "eval_types",
            "experiment_name",
            "output_dir",
            "eval_num_samples",
            "num_workers",
            "num_frames_eval",
            "enable_jax_cache",
            "checkpoint_dir",
            "jax_cache_dir",
            "model_weights_path",
            "return_dict",
            "show_progress",
        ),
        "tags": ("world-model", "official-runtime"),
    },
    "spatia": {
        "display_name": "Spatia",
        "category": "Video Generation",
        "default_task_type": "image-to-video",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _spatia_default_ref,
        "default_load_kwargs": _spatia_default_load_kwargs,
        "default_input_path": str(_data_path("test_cases", "neoverse", "videos", "jungle.png")),
        "default_prompt": (
            "A smooth lateral camera move through a dense jungle, preserving the trees, rocks, foliage, "
            "lighting, and scene geometry across the full trajectory."
        ),
        "default_call_kwargs": _spatia_default_call_kwargs,
        "input_params": ("image_path",),
        "call_params": (
            "prompt",
            "images",
            "video",
            "image_path",
            "video_path",
            "w2c_trajectory_file",
            "intrinsics",
            "num_frames",
            "fps",
            "width",
            "height",
            "num_inference_steps",
            "seed",
            "output_path",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "base_model_path",
            "lora_path",
            "map_model_path",
            "python_executable",
            "device",
        ),
        "tags": ("camera-control", "image-to-video", "world-model", "in-tree-runtime"),
    },
    "joyai-echo-longvideo": {
        "display_name": "JoyAI-Echo 1.5 / Echo-LongVideo",
        "category": "Video Generation",
        "default_task_type": "image-to-video",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _joyai_echo_longvideo_default_ref,
        "default_load_kwargs": _joyai_echo_longvideo_default_load_kwargs,
        "default_input_path": lambda: str(
            Path(_joyai_echo_longvideo_default_ref()) / "assets" / "image.png"
        ),
        "default_prompt": (
            "A cinematic live-action traveler crosses a windswept coastal platform at golden hour while the "
            "camera tracks smoothly; preserve identity, clothing, environmental sound, and temporal continuity."
        ),
        "default_call_kwargs": {
            "execute": True,
            "num_frames": 241,
            "width": 1280,
            "height": 736,
            "seed": 42,
            "fps": 25,
            "return_dict": True,
            "timeout_seconds": 21600,
        },
        "call_params": (
            "prompt",
            "images",
            "interactions",
            "memory_slots",
            "request_path",
            "num_frames",
            "width",
            "height",
            "seed",
            "fps",
            "output_path",
            "output_dir",
            "execute",
            "timeout_seconds",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "source_root",
            "checkpoint_dir",
            "gemma_path",
            "python_executable",
            "precision",
            "consumer",
            "low_vram",
            "device",
        ),
        "tags": ("audio-video", "long-video", "reference-to-video", "official-runtime"),
    },
    "joyai-echo-wm": {
        "display_name": "Echo-WM",
        "category": "Video Generation",
        "default_task_type": "image-to-video",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _joyai_echo_wm_default_ref,
        "default_load_kwargs": _joyai_echo_wm_default_load_kwargs,
        "default_input_path": lambda: str(
            Path(_joyai_echo_source_root()) / "echo_wm" / "examples" / "wm_cases" / "0004" / "input.png"
        ),
        "default_interactions": ("w-60", "a-60"),
        "default_prompt": (
            "Environment: A warm fantasy workshop with arched windows, hanging lanterns, herbs, and glass "
            "flasks. Character: One woman in a lavender dress reads at a wooden workbench. Style: whimsical "
            "hand-painted fantasy. Perspective: wide third-person interior view. Sounds: paper rustle, soft "
            "lantern chimes, and quiet room ambience. Speech: no dialogue."
        ),
        "default_call_kwargs": {
            "action_str": "w-60,a-60",
            "execute": True,
            "num_frames": 241,
            "width": 1280,
            "height": 704,
            "fps": 24,
            "seed": 4,
            "return_dict": True,
            "timeout_seconds": 21600,
        },
        "call_params": (
            "prompt",
            "images",
            "interactions",
            "action_str",
            "num_frames",
            "width",
            "height",
            "fps",
            "seed",
            "output_path",
            "output_dir",
            "execute",
            "timeout_seconds",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "source_root",
            "checkpoint_dir",
            "gemma_path",
            "python_executable",
            "variant",
            "device",
        ),
        "tags": ("action-control", "audio-video", "camera-control", "world-model", "official-runtime"),
    },
    "hyworld-worldgen": {
        "display_name": "HY-World 2.0 Trajectory Renderer",
        "category": "Video Generation",
        "default_task_type": "image-to-video",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _hyworld_worldgen_scene_ref,
        "default_load_kwargs": lambda: {"scene_path": _hyworld_worldgen_scene_ref()},
        "default_input_path": str(_data_path("test_cases", "hunyuan_worldplay", "test.png")),
        "default_prompt": "Render the materialized HY-World scene along the supplied camera trajectory.",
        "default_call_kwargs": {
            "camera_json": str(_data_path("test_cases", "hydra", "camera.json")),
            "width": 832,
            "height": 480,
            "seed": 1024,
            "return_dict": True,
        },
        "call_params": (
            "images",
            "image_path",
            "camera_json",
            "width",
            "height",
            "seed",
            "output_path",
            "return_dict",
        ),
        "load_params": ("model_path", "required_components", "scene_path", "python_executable", "device"),
        "tags": ("camera-control", "point-cloud-rendering", "world-model", "in-tree-runtime"),
    },
    "inspatio-world": {
        "display_name": "InSpatio-World",
        "category": "Video-to-Video",
        "supports_stream": False,
        "default_task_type": "video-to-video",
        "suggested_task_types": ("video-to-video",),
        "default_model_ref": _inspatio_world_default_ref,
        "default_load_kwargs": _inspatio_world_default_load_kwargs,
        "default_call_kwargs": {"skip_step1": True, "skip_step2": False, "skip_step3": False},
        "call_params": (
            "videos",
            "prompt",
            "traj_txt_path",
            "output_dir",
            "return_dict",
            "skip_step1",
            "skip_step2",
            "skip_step3",
            "relative_to_source",
            "rotation_only",
            "adaptive_frame",
            "freeze_repeat",
            "freeze_frame",
            "compile_dit",
        ),
        "tags": ("novel-view", "video-to-video", "camera-control", "trajectory"),
        "notes": "Video-to-video model that re-renders an input clip along a new camera trajectory.",
    },
    "worldlabs-marble-1.1": {
        "display_name": "World Labs Marble 1.1",
        "category": "Remote API",
        "default_backend": "api_init",
        "supports_api_init": True,
        "supports_from_pretrained": True,
        "supports_stream": False,
        "default_endpoint": "https://api.worldlabs.ai",
        "default_task_type": "3dgs-world-generation",
        "tags": ("api", "3dgs", "worldlabs", "marble"),
        "env_hints": ("WORLDLABS_API_KEY",),
        "notes": "3D Gaussian splat world generation via the World Labs Marble API.",
    },
    "dvlt": {
        "display_name": "DVLT",
        "category": "3D Scene",
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "default_model_ref": _dvlt_default_ref,
        "default_task_type": "multi-view-3d-reconstruction",
        "default_call_kwargs": _geometry_prior_default_call_kwargs(
            input_path=_geometry_prior_image_fixture(),
        ),
        "call_params": _geometry_prior_call_params(),
        "load_params": _geometry_prior_load_params(),
        "tags": ("3d", "point-reconstruction", "multi-view"),
        "notes": "Multi-view point cloud reconstruction; not image-to-video generation.",
    },
    "lingbot-map": {
        "display_name": "LingBot Map",
        "category": "3D Scene",
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "default_task_type": "streaming-3d-reconstruction",
        "default_model_ref": _lingbot_map_default_ref,
        "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_interactions": ("point_cloud_generation",),
        "call_params": (
            "images",
            "interactions",
            "output_path",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": ("3d", "point-reconstruction", "lingbot"),
        "notes": "Streaming 3D map reconstruction from images; not video generation.",
    },
    "vmem": {
        "display_name": "VMem",
        "default_model_ref": _vmem_default_ref,
        "default_load_kwargs": _vmem_default_load_kwargs,
        "default_prompt": (
            "Move forward along an empty Route 66 highway at sunset while preserving the road, "
            "horizon, and desert geometry."
        ),
        "default_input_path": str(
            _data_path("test_cases", "dualcamctrl", "demo_pic", "route66.jpg")
        ),
        "default_interactions": ("forward",),
        "tags": ("stream", "memory"),
    },
    "neoverse": {
        "display_name": "NeoVerse",
        "category": "Video-to-Video",
        "default_task_type": "video-to-video",
        "suggested_task_types": ("video-to-video",),
        "default_model_ref": _neoverse_default_ref,
        "default_load_kwargs": _neoverse_default_load_kwargs,
        "default_prompt": NEOVERSE_OFFICIAL_PROMPT,
        "default_input_path": str(
            _data_path("test_cases", "neoverse", "videos", "robot.mp4")
        ),
        "default_interactions": ("forward", "left", "camera_r"),
        "default_call_kwargs": _neoverse_default_call_kwargs,
        "call_params": (
            "prompt",
            "videos",
            "images",
            "interactions",
            "predefined_trajectory",
            "trajectory_file",
            "trajectory_data",
            "output_path",
            "fps",
            "return_dict",
            "height",
            "width",
            "num_frames",
            "num_inference_steps",
            "cfg_scale",
            "seed",
            "trajectory_mode",
            "angle",
            "distance",
            "orbit_radius",
            "zoom_ratio",
            "alpha_threshold",
            "use_first_frame",
            "static_scene",
        ),
        "load_params": (
            "height",
            "width",
            "num_inference_steps",
            "cfg_scale",
            "disable_lora",
            "enable_vram_management",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": ("stream", "camera-control"),
    },
    "worldfm": {
        "display_name": "WorldFM",
        "runtime_kind": "worldfm",
        "category": "3D Scene",
        "default_model_ref": _worldfm_default_ref,
        "default_load_kwargs": _worldfm_default_load_kwargs,
        "default_interactions": ("forward", "left", "camera_r"),
        "default_call_kwargs": {
            "fps": 30,
            "save_mode": "video",
            "meta_path": _worldfm_demo_meta_path(),
            "panorama_path": _worldfm_demo_panorama_path(),
        },
        "tags": ("novel-view", "poses", "scene-context"),
        "notes": "Supports pose-matrix JSON directly. Navigation tokens can also be converted into pose trajectories.",
    },
    "hunyuan-world-voyager": {
        "display_name": "HunyuanWorld-Voyager",
        "category": "3D Scene",
        "default_model_ref": _hunyuan_world_voyager_default_ref,
        "default_input_path": str(_hunyuan_world_voyager_case1_dir() / "ref_image.png"),
        "default_prompt": "An old-fashioned European village with thatched roofs on the houses.",
        "default_interactions": ("forward",),
        "default_call_kwargs": {
            "num_frames": 49,
            "condition_dir": str(_hunyuan_world_voyager_case1_dir()),
            "seed": 0,
            "infer_steps": 50,
            "flow_shift": 7.0,
            "embedded_cfg_scale": 6.0,
            "i2v_stability": True,
            "ulysses_degree": 1,
            "ring_degree": 1,
        },
        "tags": ("world-model", "video-to-3d", "local-checkpoint"),
    },
    "hunyuan-game-craft": {
        "display_name": "Hunyuan Game Craft",
        "default_model_ref": _hunyuan_gamecraft_default_ref,
        "default_input_path": str(_data_path("test_cases", "hunyuan_game_craft", "village.png")),
        "default_prompt": (
            "A charming medieval village with cobblestone streets, thatched-roof houses, "
            "and vibrant flower gardens under a bright blue sky."
        ),
        "default_interactions": ("forward", "backward", "right", "left"),
        "default_load_kwargs": {"seed": 250160},
        "load_params": ("cpu_offload", *DISPATCH_LOAD_PARAMS),
        "default_call_kwargs": {
            "interactions": ("forward", "backward", "right", "left"),
            "interaction_speed": (0.2, 0.2, 0.2, 0.2),
            "size": (704, 1216),
            "num_frames": 129,
            "infer_steps": 50,
            "cfg_scale": 2.0,
        },
        "aliases": ("hunyuan-gamecraft", "gamecraft", "tencent/Hunyuan-GameCraft-1.0"),
        "tags": ("interactive-world", "camera-control", "game-video", "local-checkpoint"),
    },
    "vggt": {
        "display_name": "VGGT",
        "runtime_kind": "two_stage_3dgs",
        "category": "3D Scene",
        "default_model_ref": _vggt_default_ref,
        "default_input_path": str(_data_path("test_cases", "vggt", "examples", "kitchen", "images")),
        "default_task_type": "vggt_two_stage_3dgs",
        "suggested_task_types": ("vggt_two_stage_3dgs", "vggt_base", "official"),
        "default_interactions": ("forward", "left", "camera_zoom_in"),
        "default_call_kwargs": {
            "point_conf_threshold": 0.2,
            "resolution": 518,
            "preprocess_mode": "crop",
            "image_width": 704,
            "image_height": 480,
            "fps": 12,
        },
        "call_params": (
            "image_path",
            "images",
            "interactions",
            "camera_view",
            "task_type",
            "output_dir",
            "output_path",
            "return_dict",
            "point_conf_threshold",
            "resolution",
            "preprocess_mode",
            "image_width",
            "image_height",
            "fps",
            "conf_thres",
            "frame_filter",
            "mask_black_bg",
            "mask_white_bg",
            "show_cam",
            "prediction_mode",
            "output_name",
        ),
        "tags": ("3dgs", "reconstruction", "camera-control"),
    },
    "splatt3r": {
        "display_name": "Splatt3R",
        "category": "3D Scene",
        "supports_stream": False,
        "supports_from_pretrained": True,
        "default_backend": "from_pretrained",
        "default_model_ref": _splatt3r_default_ref,
        "default_task_type": "3d-reconstruction",
        "default_input_path": str(_data_path("test_cases", "vggt", "examples", "kitchen", "images")),
        "call_params": (
            "images",
            "image_path",
            "input_path",
            "interactions",
            "output_path",
            "return_dict",
        ),
        "load_params": (
            "checkpoint_path",
            "model_path",
            "device",
            "image_size",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": ("3dgs", "gaussian-splatting", "reconstruction"),
        "notes": "Inference-only Splatt3R wrapper reconstructs a Gaussian PLY from one or two images.",
    },
    "scope": {
        "display_name": "SCOPE",
        "category": "Video Generation",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": lambda: _checkpoint_model_ref(
            "SCOPE",
            "zizhaotong--SCOPE",
            "custom--SCOPE",
            fallback="zizhaotong/SCOPE",
        ),
        "default_task_type": "image-action-to-video",
        "default_prompt": (
            "In a whimsical, toy-inspired garden, the first-person view reveals a tactical weapon aimed forward "
            "along a sunlit path."
        ),
        "default_input_path": str(_data_path("test_cases", "scope", "example_0", "image.png")),
        "default_call_kwargs": {
            "execute": True,
            "action_path": str(_data_path("test_cases", "scope", "example_0", "action.parquet")),
            "max_frames": 81,
            "height": 480,
            "width": 832,
            "num_inference_steps": 30,
            "seed": 0,
            "return_dict": True,
        },
        "default_load_kwargs": lambda: {"model_dir": _checkpoint_model_ref("SCOPE", "zizhaotong--SCOPE", "custom--SCOPE")},
        "call_params": (
            "prompt",
            "images",
            "interactions",
            "output_path",
            "fps",
            "return_dict",
            "execute",
            "action_path",
            "operator_kwargs",
            "height",
            "width",
            "max_frames",
            "num_inference_steps",
            "seed",
        ),
        "input_params": ("prompt", "input_path", "action_path", "max_frames", "height", "width", "num_inference_steps", "seed"),
        "load_params": ("model_path", "model_dir", "checkpoint_root", "checkpoint_dir", "python_executable", "device", *DISPATCH_LOAD_PARAMS),
        "aliases": ("scope-world-model", "zizhaotong/scope"),
        "tags": ("world-model", "action-conditioned-video", "official-demo", "in-tree-runtime"),
        "notes": "Defaults use the official SCOPE example_0 image/action parquet and run execute=True against Hugging Face checkpoint assets.",
    },
    "fantasyworld": {
        "display_name": "FantasyWorld",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _fantasy_world_wan21_default_ref,
        "default_load_kwargs": _fantasy_world_default_load_kwargs,
        "default_prompt": "A coherent fantasy harbor world with stable geometry during a forward camera move.",
        "default_call_kwargs": lambda: _fantasy_world_default_call_kwargs(conf_threshold=1.0),
        "call_params": (
            "images",
            "end_image",
            "interactions",
            "prompt",
            "camera_json_path",
            "camera_data",
            "camera_poses",
            "K",
            "scene_name",
            "output_dir",
            "return_dict",
            "fps",
            "using_scale",
            "conf_threshold",
            "stride",
            "seed",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "weight_dtype",
            "sample_steps",
            "cfg_scale",
            "timestep_boundary",
            "frames",
            "fps",
            "height",
            "width",
            "base_seed",
            "high_model_device",
            "low_model_device",
            "moge_device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("fantasyworld-wan21-default",),
        "tags": ("world-model", "camera-control", "official-runtime"),
    },
    "fantasyworld-wan22": {
        "display_name": "FantasyWorld Wan2.2",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _fantasy_world_wan22_default_ref,
        "default_load_kwargs": _fantasy_world_wan22_load_kwargs,
        "default_prompt": "A coherent fantasy harbor world with stable geometry during a forward camera move.",
        "default_call_kwargs": _fantasy_world_default_call_kwargs,
        "call_params": (
            "images",
            "end_image",
            "interactions",
            "prompt",
            "camera_json_path",
            "camera_data",
            "camera_poses",
            "K",
            "scene_name",
            "output_dir",
            "return_dict",
            "fps",
            "using_scale",
            "conf_threshold",
            "stride",
            "neg_prompt",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "weight_dtype",
            "sample_steps",
            "cfg_scale",
            "timestep_boundary",
            "frames",
            "fps",
            "height",
            "width",
            "base_seed",
            "high_model_device",
            "low_model_device",
            "moge_device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": ("world-model", "camera-control", "official-runtime"),
    },
    "fantasyworld-wan21": {
        "display_name": "FantasyWorld Wan2.1",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _fantasy_world_wan21_default_ref,
        "default_load_kwargs": _fantasy_world_wan21_load_kwargs,
        "default_prompt": "A coherent fantasy harbor world with stable geometry during a forward camera move.",
        "default_call_kwargs": lambda: _fantasy_world_default_call_kwargs(conf_threshold=1.0),
        "call_params": (
            "images",
            "interactions",
            "prompt",
            "camera_json_path",
            "camera_data",
            "camera_poses",
            "K",
            "scene_name",
            "output_dir",
            "return_dict",
            "fps",
            "seed",
            "using_scale",
            "conf_threshold",
            "stride",
            "neg_prompt",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "weight_dtype",
            "sample_steps",
            "sample_guide_scale",
            "frames",
            "fps",
            "height",
            "width",
            "start_index",
            *DISPATCH_LOAD_PARAMS,
        ),
        "tags": ("world-model", "camera-control", "official-runtime"),
    },
    "flashworld": {
        "display_name": "FlashWorld",
        "category": "3D Scene",
        "default_model_ref": _flash_world_default_ref,
        "default_load_kwargs": _flash_world_default_load_kwargs,
        "default_interactions": ("forward", "camera_l", "right", "camera_zoom_in"),
        "default_call_kwargs": {
            "num_frames": 16,
            "fps": 15,
            "image_height": 480,
            "image_width": 704,
            "return_video": True,
        },
        "tags": ("3dgs", "spz", "world-model", "camera-control"),
        "notes": "Set WORLDFOUNDRY_DISABLE_SAGEATTENTION=1 on driver/Triton stacks where SageAttention kernels fail.",
    },
    "infinite-vggt": {
        "display_name": "Infinite VGGT",
        "runtime_kind": "default",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _stream_vggt_default_ref,
        "default_task_type": "3d-reconstruction",
        "default_input_path": str(_data_path("test_cases", "vggt", "examples", "kitchen", "images")),
        "default_call_kwargs": {"output_format": "ply"},
        "call_params": (
            "data_path",
            "input_path",
            "images",
            "image_path",
            "interaction",
            "output_format",
            "output_path",
            "return_dict",
        ),
        "load_params": (
            "model_path",
            "pretrained_model_path",
            "representation_path",
            "device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("stream-vggt", "streamvggt", "lch01/streamvggt"),
        "tags": ("3d", "reconstruction", "streaming-3d"),
        "notes": "Runs the in-tree StreamVGGT/Infinite-VGGT point-cloud reconstruction path.",
    },
    "pixelsplat": {
        "display_name": "pixelSplat",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _pixelsplat_default_ref,
        "default_task_type": "3d-reconstruction",
        "call_params": (
            "images",
            "input_path",
            "interactions",
            "output_path",
            "output_dir",
            "return_dict",
            "checkpoint_path",
            "index_path",
            "dataset_roots",
            "experiment",
            "overrides",
            "python_executable",
        ),
        "load_params": (
            "checkpoint_path",
            "ckpt_path",
            "model_path",
            "pretrained_model_path",
            "dataset_roots",
            "index_path",
            "experiment",
            "device",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("pixel-splat", "dylanebert/pixelsplat"),
        "tags": ("3dgs", "reconstruction"),
        "notes": "Runs the in-tree pixelSplat runtime against the packaged demo index and staged re10k checkpoint.",
    },
    "vggt-omega": {
        "display_name": "VGGT-Omega",
        "runtime_kind": "default",
        "category": "3D Scene",
        "default_model_ref": _prefer_existing_model_ref(
            *_cache_candidates("facebook--VGGT-Omega", "VGGT-Omega"),
            str(_project_root() / "assets" / "checkpoints" / "facebook__VGGT-Omega"),
            "facebook/VGGT-Omega",
        ),
        "default_task_type": "vggt_omega_official_scene_export",
        "suggested_task_types": ("vggt_omega_official_scene_export", "vggt_base"),
        "default_interactions": ("forward", "left", "camera_zoom_in"),
        "default_call_kwargs": {
            "task_type": "official",
            "image_resolution": 512,
            "preprocess_mode": "crop",
            "video_sample_fps": 1.0,
            "max_points_k": 200,
        },
        "call_params": (
            "image_path",
            "images",
            "interactions",
            "camera_view",
            "task_type",
            "output_dir",
            "return_dict",
            "image_resolution",
            "preprocess_mode",
            "patch_size",
            "video_sample_fps",
            "conf_thres",
            "mask_black_bg",
            "mask_white_bg",
            "show_cam",
            "max_points_k",
            "output_name",
        ),
        "aliases": ("vggt_omega", "facebook/vggt-omega"),
        "tags": ("3dgs", "reconstruction", "camera-control", "omega"),
        "notes": "Uses the official VGGT-Omega scene-export path and emits GLB artifacts from local checkpoint trees.",
    },
    "lagernvs": {
        "display_name": "LagrNVS",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "default_model_ref": lambda: _in_tree_repo_ref("lagernvs"),
        "supports_from_pretrained": True,
        "supports_stream": False,
        "default_task_type": "novel_view_synthesis",
        "suggested_task_types": ("novel_view_synthesis",),
        "default_interactions": ("camera_path",),
        "default_call_kwargs": _three_d_four_d_runtime_call_kwargs(
            video_length=16,
            fps=25,
            model_repo=_lagernvs_default_ref(),
        ),
        "call_params": _three_d_four_d_runtime_call_params(
            "images",
            "video_length",
            "fps",
            "model_repo",
        ),
        "load_params": _three_d_four_d_runtime_load_params(),
        "aliases": ("lager-nvs", "facebook/lagernvs_general_512", "facebookresearch/lagernvs"),
        "tags": ("novel-view", "nvs", "vggt", "gated-checkpoint"),
        "env_hints": ("HF_TOKEN",),
        "notes": "Uses the local official LagrNVS minimal_inference.py entrypoint when weights and dependencies are staged.",
    },
    "monst3r": {
        "display_name": "MonST3R",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "default_model_ref": lambda: _in_tree_repo_ref("monst3r"),
        "supports_from_pretrained": True,
        "supports_stream": False,
        "default_task_type": "monst3r_demo",
        "default_call_kwargs": _monst3r_default_call_kwargs,
        "call_params": _three_d_four_d_runtime_call_params(
            "images",
            "video",
            "fps",
            "weights",
            "model_name",
            "image_size",
            "seq_name",
            "save_name",
            "num_frames",
            "batch_size",
            "niter",
            "scenegraph_type",
            "winsize",
            "refid",
            "min_conf_thr",
            "cam_size",
            "temporal_smoothing_weight",
            "translation_weight",
            "flow_loss_weight",
            "flow_loss_start_iter",
            "flow_loss_threshold",
            "not_batchify",
            "real_time",
            "window_wise",
            "window_size",
            "window_overlap_ratio",
            "skip_pair_dynamic_mask",
            "silent",
        ),
        "load_params": _three_d_four_d_runtime_load_params(),
        "tags": ("4d", "reconstruction", "dynamic-scene"),
        "notes": "Runs the in-tree MonST3R demo.py entrypoint and exports a GLB scene artifact. Studio defaults disable optional flow loss so missing RAFT2/SAM2 assets do not block preview output.",
    },
    "mvdiffusion": {
        "display_name": "MVDiffusion",
        "category": "Video Generation",
        "default_backend": "from_pretrained",
        "default_model_ref": lambda: _in_tree_repo_ref("MVDiffusion"),
        "supports_from_pretrained": True,
        "supports_stream": False,
        "default_prompt": "This kitchen is a charming blend of rustic and modern, featuring a large reclaimed wood island with marble countertop, a sink surrounded by cabinets. To the left of the island, a stainless-steel refrigerator stands tall. To the right of the sink, built-in wooden cabinets painted in a muted.",
        "default_input_path": str(_data_path("test_cases", "mvdiffusion", "outpaint_example.png")),
        "default_task_type": "mvdiffusion_demo",
        "default_call_kwargs": _three_d_four_d_runtime_call_kwargs(
            gen_video=True,
            steps=20,
        ),
        "call_params": _three_d_four_d_runtime_call_params(
            "prompt",
            "images",
            "text_path",
            "gen_video",
            "steps",
        ),
        "load_params": _three_d_four_d_runtime_load_params(),
        "tags": ("multi-view", "diffusion", "3d"),
        "notes": "Runs the official MVDiffusion demo.py entrypoint from the local official repo. The 20-step default produces a visually structured panorama; one step is suitable only for runtime smoke testing.",
    },
    "stable-virtual-camera": {
        "display_name": "Stable Virtual Camera",
        "category": "Video Generation",
        "default_backend": "from_pretrained",
        "default_model_ref": lambda: _in_tree_repo_ref("stable-virtual-camera"),
        "supports_from_pretrained": True,
        "supports_stream": False,
        "default_task_type": "stable_virtual_camera_demo",
        "default_input_path": str(
            _data_path("test_cases", "stable_virtual_camera", "basic", "blue-car.jpg")
        ),
        "default_call_kwargs": _three_d_four_d_runtime_call_kwargs(
            version="1.1",
            task="img2trajvid_s-prob",
            pretrained_model_name_or_path=str(checkpoint_root_path("stable-virtual-camera")),
            weight_name="modelv1.1.safetensors",
            H=576,
            W=576,
            T=21,
            num_steps=50,
            seed=23,
        ),
        "call_params": _three_d_four_d_runtime_call_params(
            "images",
            "version",
            "task",
            "pretrained_model_name_or_path",
            "weight_name",
            "H",
            "W",
            "T",
            "num_steps",
            "seed",
        ),
        "load_params": _three_d_four_d_runtime_load_params(),
        "tags": ("camera-control", "novel-view", "video"),
        "notes": "Runs the official Stable Virtual Camera demo.py entrypoint from the local official repo.",
    },
    "wonderjourney": {
        "display_name": "WonderJourney",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "default_model_ref": lambda: _in_tree_repo_ref("WonderJourney"),
        "supports_from_pretrained": True,
        "supports_stream": False,
        "default_prompt": "a village landscape",
        "default_task_type": "wonderjourney_run",
        "default_call_kwargs": _three_d_four_d_runtime_call_kwargs(config=str(_data_path("models", "runtime", "configs", "wonderjourney", "village.yaml"))),
        "call_params": _three_d_four_d_runtime_call_params(
            "prompt",
            "config",
            "frames",
            "num_scenes",
            "num_keyframes",
            "save_fps",
            "skip_interp",
            "rotation_path",
        ),
        "load_params": _three_d_four_d_runtime_load_params(),
        "tags": ("world-generation", "3d", "camera-navigation"),
        "notes": "Runs the official WonderJourney run.py entrypoint from the local official repo.",
    },
    "wonderworld": {
        "display_name": "WonderWorld",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "default_model_ref": lambda: _in_tree_repo_ref("WonderWorld"),
        "supports_from_pretrained": True,
        "supports_stream": False,
        "default_prompt": "a village landscape",
        "default_task_type": "wonderworld_run",
        "default_call_kwargs": _three_d_four_d_runtime_call_kwargs(config=str(_data_path("models", "runtime", "configs", "wonderworld", "inference.yaml"))),
        "call_params": _three_d_four_d_runtime_call_params(
            "prompt",
            "config",
            "frames",
            "num_scenes",
            "rotation_path",
            "gen_sky_image",
            "gen_sky",
            "gen_layer",
            "load_gen",
            "worldfoundry_batch",
            "worldfoundry_gs_iterations",
            "worldfoundry_sky_iterations",
        ),
        "load_params": _three_d_four_d_runtime_load_params(),
        "tags": ("world-generation", "3dgs", "camera-navigation"),
        "notes": "Runs the official WonderWorld run.py entrypoint from the local official repo.",
    },
    "worldgen": {
        "display_name": "WorldGen",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "default_model_ref": lambda: _in_tree_repo_ref("WorldGen"),
        "supports_from_pretrained": True,
        "supports_stream": False,
        "default_prompt": "a small furnished room",
        "default_task_type": "worldgen_inference",
        "default_call_kwargs": _three_d_four_d_runtime_call_kwargs(save_scene=True, low_vram=False),
        "call_params": _three_d_four_d_runtime_call_params(
            "prompt",
            "images",
            "pano_image",
            "viewer",
            "use_sharp",
            "return_mesh",
            "save_scene",
            "low_vram",
            "resolution",
        ),
        "load_params": _three_d_four_d_runtime_load_params(),
        "tags": ("world-generation", "3d", "mesh"),
        "notes": "Runs the official WorldGen demo.py entrypoint from the local official repo.",
    },
    "cut3r": {
        "display_name": "CUT3R",
        "runtime_kind": "two_stage_3dgs",
        "category": "3D Scene",
        "default_model_ref": _cut3r_default_ref,
        "aliases": ("cut3r-512", "cut3r_512_dpt_4_64"),
        "default_task_type": "cut3r_two_stage_3dgs",
        "suggested_task_types": ("cut3r_two_stage_3dgs", "cut3r_base", "cut3r_official_export"),
        "default_interactions": ("forward", "left", "camera_zoom_in"),
        "default_call_kwargs": {"image_width": 704, "image_height": 480, "size": 512, "vis_threshold": 1.5},
        "call_params": (
            "image_path",
            "images",
            "interactions",
            "task_type",
            "size",
            "vis_threshold",
            "revisit",
            "update",
            "use_pose",
            "frames_per_interaction",
            "camera_radius",
            "camera_yaw",
            "camera_pitch",
            "image_width",
            "image_height",
            "num_orbit_frames",
            "yaw_step",
        ),
        "tags": ("3dgs", "reconstruction"),
    },
    "pi3": {
        "display_name": "Pi3",
        "runtime_kind": "pointcloud_nav",
        "category": "3D Scene",
        "default_model_ref": _pi3_default_ref,
        "default_task_type": "reconstruction",
        "suggested_task_types": ("reconstruction", "render_view", "render_trajectory"),
        "default_interactions": ("forward", "left", "camera_r"),
        "default_load_kwargs": {"mode": "pi3x"},
        "default_call_kwargs": {"interval": 10},
        "call_params": (
            "images",
            "videos",
            "image_path",
            "video_path",
            "task_type",
            "interactions",
            "camera_view",
            "visualize_ops",
            "interval",
            "frames_per_interaction",
            "hold_frames",
            "output_path",
            "return_dict",
        ),
        "tags": ("point-cloud", "render-view", "trajectory"),
    },
    "loger": {
        "display_name": "LoGeR",
        "runtime_kind": "pointcloud_nav",
        "category": "3D Scene",
        "default_model_ref": _loger_default_ref,
        "default_task_type": "reconstruction",
        "suggested_task_types": ("reconstruction", "render_view", "render_trajectory"),
        "default_interactions": ("forward", "left", "camera_r"),
        "default_load_kwargs": {"mode": "loger"},
        "default_call_kwargs": {"interval": 10},
        "call_params": (
            "images",
            "videos",
            "image_path",
            "video_path",
            "task_type",
            "interactions",
            "camera_view",
            "visualize_ops",
            "interval",
            "frames_per_interaction",
            "hold_frames",
            "output_path",
            "return_dict",
        ),
        "tags": ("point-cloud", "render-view", "trajectory"),
    },
    "depth-anything-v2": {
        "display_name": "Depth Anything 2",
        "category": "Depth / Geometry",
        "default_model_ref": _depth_anything_v2_default_ref,
        "aliases": ("da2", "depth-anything-2", "depth-anything2", "depthanything2"),
        "tags": ("depth", "geometry"),
        "notes": "Relative depth wrapper around the vendored official Depth-Anything-V2 runtime.",
    },
    "depth-anything-v3": {
        "display_name": "Depth Anything 3",
        "category": "Depth / Geometry",
        "default_model_ref": _depth_anything_v3_default_ref,
        "suggested_task_types": ("official_soh_export", "single_view_depth", "multi_view_depth", "video_depth"),
        "default_call_kwargs": {"export_format": "mini_npz", "process_res": 504, "max_frames": 2},
        "call_params": (
            "input_data",
            "images",
            "videos",
            "video",
            "interactions",
            "output_dir",
            "return_dict",
            "export_format",
            "export_feat_layers",
            "process_res",
            "process_res_method",
            "use_ray_pose",
            "infer_gs",
            "ref_view_strategy",
            "align_to_input_ext_scale",
            "render_exts",
            "render_ixts",
            "render_hw",
            "data_type",
            "max_frames",
            "frame_stride",
            "conf_thresh_percentile",
            "num_max_points",
            "show_cameras",
            "feat_vis_fps",
            "export_kwargs",
        ),
        "aliases": ("da3", "depth-anything-3", "depth-anything3", "depthanything3"),
        "tags": ("depth", "geometry"),
    },
    "hy-world-2p0": {
        "display_name": "HY-World-2.0",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _prefer_existing_model_ref(
            *_cache_candidates("tencent--HY-World-2.0", "HY-World-2.0"),
            "tencent/HY-World-2.0",
        ),
        "default_input_path": str(_data_path("test_cases", "images")),
        "default_prompt": "Reconstruct the scene and surface geometry from the image sequence.",
        "default_call_kwargs": {
            "task": "worldrecon",
            "confidence_percentile": 10.0,
            "edge_normal_threshold": 5.0,
            "edge_depth_threshold": 0.03,
            "apply_confidence_mask": True,
            "apply_edge_mask": True,
            "apply_sky_mask": False,
            "save_points": True,
            "save_depth": True,
            "save_normal": True,
            "save_gs": True,
            "save_camera": True,
            "save_rendered": False,
            "save_colmap": True,
        },
        "input_params": ("image_path",),
        "call_params": (
            "task",
            "image_path",
            "images",
            "input",
            "output_path",
            "return_dict",
            "confidence_percentile",
            "edge_normal_threshold",
            "edge_depth_threshold",
            "apply_confidence_mask",
            "apply_edge_mask",
            "apply_sky_mask",
            "save_points",
            "save_depth",
            "save_normal",
            "save_gs",
            "save_camera",
            "save_rendered",
            "save_colmap",
            "prior_cam_path",
            "prior_depth_path",
        ),
        "load_params": (
            "task",
            "backend",
            "subfolder",
            "torch_dtype",
            "device_map",
        ),
        "aliases": ("hy-world-2.0", "hyworld2.0", "hyworld-2.0", "hy-world2.0"),
        "tags": ("reconstruction", "world-mirror"),
    },
    "hunyuanworld-mirror": {
        "display_name": "HunyuanWorld-Mirror",
        "category": "3D Scene",
        "default_backend": "from_pretrained",
        "supports_from_pretrained": True,
        "default_model_ref": _prefer_existing_model_ref(
            *_cache_candidates("tencent--HunyuanWorld-Mirror", "HunyuanWorld-Mirror"),
            "tencent/HunyuanWorld-Mirror",
        ),
        "default_input_path": str(_data_path("test_cases", "vggt", "examples", "kitchen", "images")),
        "default_prompt": "Reconstruct the scene and surface geometry from the image sequence.",
        "default_call_kwargs": {
            "confidence_percentile": 10.0,
            "edge_normal_threshold": 5.0,
            "edge_depth_threshold": 0.03,
            "apply_confidence_mask": True,
            "apply_edge_mask": True,
            "apply_sky_mask": False,
            "save_pointmap": True,
            "save_depth": True,
            "save_normal": True,
            "save_gs": True,
            "save_rendered": False,
            "save_colmap": True,
        },
        "input_params": ("image_path",),
        "call_params": (
            "image_path",
            "images",
            "input",
            "output_path",
            "return_dict",
            "confidence_percentile",
            "edge_normal_threshold",
            "edge_depth_threshold",
            "apply_confidence_mask",
            "apply_edge_mask",
            "apply_sky_mask",
            "cond_flags",
            "save_pointmap",
            "save_depth",
            "save_normal",
            "save_gs",
            "save_rendered",
            "save_colmap",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "local_model_path",
            "output_path",
            "device",
            "local_files_only",
            *DISPATCH_LOAD_PARAMS,
        ),
        "aliases": ("hunyuan-mirror", "hy-world-mirror", "worldmirror", "world-mirror"),
        "tags": ("reconstruction", "world-mirror", "point-cloud", "gaussian-splat"),
    },
    "hunyuan-worldplay": {
        "display_name": "HY-WorldPlay",
        "category": "Video Generation",
        "default_model_ref": _hunyuan_worldplay_default_ref,
        "default_input_path": str(_data_path("test_cases", "hunyuan_worldplay", "test.png")),
        "default_prompt": (
            "A paved pathway leads towards a stone arch bridge spanning a calm body of water.  "
            "Lush green trees and foliage line the path and the far bank of the water. "
            "A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. "
            "The water reflects the surrounding greenery and the sky.  The scene is bathed in soft, "
            "natural light, creating a tranquil and serene atmosphere. The pathway is composed of "
            "large, rectangular stones, and the bridge is constructed of light gray stone.  The overall "
            "composition emphasizes the peaceful and harmonious nature of the landscape."
        ),
        "default_interactions": ("w-31",),
        "default_load_kwargs": {
            "enable_offloading": True,
            "enable_group_offloading": True,
            "overlap_group_offloading": False,
        },
        "default_call_kwargs": _hunyuan_worldplay_default_call_kwargs,
        "load_params": (
            "enable_offloading",
            "enable_group_offloading",
            "overlap_group_offloading",
            "torchrun_nproc_per_node",
            "torchrun_nproc",
            "nproc_per_node",
            *DISPATCH_LOAD_PARAMS,
        ),
        "supports_stream": True,
        "stream_params": (),
        "call_params": (
            "prompt",
            "images",
            "image_path",
            "interactions",
            "pose",
            "num_frames",
            "aspect_ratio",
            "num_inference_steps",
            "negative_prompt",
            "seed",
            "fps",
            "output_type",
            "prompt_rewrite",
            "enable_sr",
            "sr_num_inference_steps",
            "return_pre_sr_video",
            "few_step",
            "chunk_latent_frames",
            "model_type",
            "transformer_resident_ar_rollout",
            "user_height",
            "user_width",
            "forward_speed",
            "yaw_speed_deg",
            "pitch_speed_deg",
            "output_path",
            "return_dict",
        ),
        "aliases": ("worldplay", "hy-worldplay", "hyworldplay", "tencent/HY-WorldPlay"),
        "tags": ("interactive-world", "camera-control", "image-to-video", "wmfactory"),
    },
    "multiworld-ittakestwo": {
        "display_name": "MultiWorld ItTakesTwo",
        "category": "Video Generation",
        "default_backend": "from_pretrained",
        "default_input_path": str(_data_path("test_cases", "multiworld_ittakestwo", "input.png")),
        "default_load_kwargs": _multiworld_ittakestwo_default_load_kwargs,
        "default_call_kwargs": _multiworld_ittakestwo_default_call_kwargs,
        "call_params": (
            "images",
            "action",
            "action_path",
            "env_obv",
            "output_dir",
            "save_name",
            "return_dict",
            "num_frames",
            "height",
            "width",
            "fps",
            "num_inference_steps",
            "inference_seed",
            "derive_env_obv_from_image",
            "show_progress",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "weight_dtype",
            "runtime_root",
            "config_path",
            "checkpoint_path",
            "base_model_root",
            "vggt_root",
            "python_executable",
            "derive_env_obv_from_image",
            "num_inference_steps",
            "inference_seed",
            "fps",
        ),
        "aliases": ("multiworld", "multiword"),
        "tags": ("multi-agent-world-model", "interactive-video", "game-video"),
        "notes": "Uses the official MultiWorld ItTakesTwo toy inference config and checkpoint.",
    },
    "wan-2p5": {
        "display_name": "Wan 2.5",
        "category": "Remote API",
        "default_backend": "api_init",
        "default_endpoint": "https://dashscope.aliyuncs.com/api/v1",
        "tags": ("api", "text-to-video", "image-to-video"),
    },
    "wan-2p6": {
        "display_name": "Wan 2.6",
        "category": "Remote API",
        "default_backend": "api_init",
        "default_endpoint": "https://dashscope.aliyuncs.com/api/v1",
        "tags": ("api", "image-to-video"),
    },
    "wan-2p7": {
        "display_name": "Wan 2.7",
        "category": "Remote API",
        "default_backend": "api_init",
        "default_endpoint": "https://dashscope.aliyuncs.com/api/v1",
        "tags": ("api", "image-to-video"),
    },
    "worldlabs": {
        "display_name": "World Labs",
        "category": "Remote API",
        "default_backend": "api_init",
        "default_endpoint": "https://api.worldlabs.ai",
        "tags": ("api", "world-json", "assets"),
        "notes": "Can return structured world JSON plus downloadable assets.",
    },
    "runway-gen4p5": {
        "display_name": "Runway Gen-4.5",
        "category": "Remote API",
        "default_backend": "api_init",
        "default_endpoint": "https://api.dev.runwayml.com/v1",
        "tags": ("api", "hosted-video"),
    },
    "luma-ray2": {
        "display_name": "Luma Ray2",
        "category": "Remote API",
        "default_backend": "api_init",
        "tags": ("api", "hosted-video"),
    },
    "kling-api": {
        "display_name": "Kling API",
        "category": "Remote API",
        "default_backend": "api_init",
        "tags": ("api", "hosted-video"),
    },
    "veo3": {
        "display_name": "Veo3",
        "category": "Remote API",
        "default_backend": "api_init",
        "default_endpoint": "https://generativelanguage.googleapis.com",
        "tags": ("api", "t2v", "i2v"),
    },
    "ltx-2.3-t2v": {
        "display_name": "LTX-2.3 T2V",
        "category": "Video Generation",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_backend": "from_pretrained",
        "default_task_type": "t2v",
        "default_prompt": "Ocean waves break against black rocks at sunset with coherent water motion and cinematic light.",
        "default_model_ref": lambda: _checkpoint_model_ref(
            "LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors",
            "Lightricks--LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors",
            fallback="Lightricks/LTX-2.3",
        ),
        "default_load_kwargs": _ltx23_default_load_kwargs,
        "default_call_kwargs": {
            "num_frames": 121,
            "fps": 24,
            "height": 512,
            "width": 768,
            "num_inference_steps": 11,
            "guidance_scale": 1.0,
            "seed": 42,
        },
        "call_params": (
            "prompt",
            "output_path",
            "return_dict",
            "fps",
            "num_frames",
            "num_inference_steps",
            "height",
            "width",
            "guidance_scale",
            "seed",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "lazy",
            "height",
            "width",
            "num_frames",
            "frame_rate",
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "offload_mode",
            "quantization",
        ),
        "aliases": ("ltx2.3-t2v", "ltx2_3_t2v"),
        "tags": ("text-to-video", "joint-audio-video", "native-diffusion", "local-checkpoint"),
    },
    "ltx-2.3-i2v": {
        "display_name": "LTX-2.3 I2V",
        "category": "Video Generation",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_backend": "from_pretrained",
        "default_task_type": "i2v",
        "default_prompt": "A young man holding a bright sparkler gently turns his head as sparks scatter through the night air, with natural motion and stable facial details.",
        "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
        "default_model_ref": lambda: _checkpoint_model_ref(
            "LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors",
            "Lightricks--LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors",
            fallback="Lightricks/LTX-2.3",
        ),
        "default_load_kwargs": _ltx23_default_load_kwargs,
        "default_call_kwargs": {
            "num_frames": 121,
            "fps": 24,
            "height": 512,
            "width": 768,
            "num_inference_steps": 11,
            "guidance_scale": 1.0,
            "seed": 0,
        },
        "call_params": (
            "prompt",
            "images",
            "output_path",
            "return_dict",
            "fps",
            "num_frames",
            "num_inference_steps",
            "height",
            "width",
            "guidance_scale",
            "seed",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "lazy",
            "height",
            "width",
            "num_frames",
            "frame_rate",
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "offload_mode",
            "quantization",
        ),
        "aliases": ("ltx-2.x", "ltx2.3", "ltx-2.3", "ltx2.3-i2v", "ltx2_3_i2v"),
        "tags": ("image-to-video", "joint-audio-video", "native-diffusion", "local-checkpoint"),
    },
    "hyperflow": {
        "display_name": "HyperFlow for MiniMax-H3",
        "category": "Video Generation",
        "supports_from_pretrained": True,
        "supports_stream": False,
        "default_backend": "from_pretrained",
        "default_task_type": "t2va",
        "suggested_task_types": ("t2va", "fl2va", "ref2va"),
        "default_model_ref": lambda: _checkpoint_model_ref(
            "minimax-h3", "MiniMaxAI--MiniMax-H3", fallback=str(hfd_root_path("MiniMaxAI--MiniMax-H3"))
        ),
        "default_load_kwargs": {"gpus": 1},
        "default_call_kwargs": {"num_frames": 124, "seed": 42, "return_dict": True},
        "call_params": (
            "prompt", "images", "image", "last_image", "references", "workflow", "output_path",
            "return_dict", "num_frames", "height", "width", "seed", "fps", "num_inference_steps",
            "memory_reserve_margin", "offload", "attention_backend", "sol_attn", "timeout_seconds",
        ),
        "load_params": ("model_path", "weights_path", "python_executable", "device", "gpus"),
        "aliases": ("hyperflow-h3", "minimax-h3-hyperflow"),
        "tags": ("text-to-video", "image-to-video", "reference-to-video", "joint-audio-video", "official-runtime"),
        "notes": "Eight-step HyperFlow adapter; local H3 components and the dedicated inference environment are required.",
    },
    "ltx-2.5": {
        "display_name": "LTX-2.5 Distilled",
        "category": "Video Generation",
        "supports_from_pretrained": True,
        "supports_stream": True,
        "default_backend": "from_pretrained",
        "default_task_type": "t2v",
        "default_prompt": "Ocean waves break against black rocks at sunset with coherent water motion, synchronized natural audio, and cinematic light.",
        "default_model_ref": _ltx25_checkpoint_dir,
        "default_load_kwargs": _ltx25_default_load_kwargs,
        "default_call_kwargs": {
            "num_frames": 121,
            "fps": 24,
            "height": 512,
            "width": 768,
            "seed": 42,
            "execute": True,
            "timeout_seconds": 7200,
            "return_dict": True,
        },
        "call_params": (
            "prompt",
            "images",
            "output_path",
            "return_dict",
            "execute",
            "timeout_seconds",
            "fps",
            "num_frames",
            "height",
            "width",
            "seed",
            "offload_mode",
            "quantization",
            "diffvae_optimization",
            "auto_duration",
            "num_generated_keyframes",
            "enhance_prompt",
        ),
        "load_params": (
            "model_path",
            "required_components",
            "device",
            "model_id",
            "checkpoint_dir",
            "source_root",
            "python_executable",
            "transformer_path",
            "text_encoder_path",
            "video_vae_path",
            "audio_vae_path",
            "spatial_upsampler_path",
            "duration_head_path",
            "prompt_enhancer_gemma_root",
            "offload_mode",
            "quantization",
            "diffvae_optimization",
        ),
        "aliases": ("ltx2.5", "ltx25", "ltx-2.5-distilled"),
        "tags": ("text-to-video", "image-to-video", "joint-audio-video", "official-runtime"),
        "notes": "Runs the pinned official LTX-2.5 DistilledPipeline in its dedicated environment.",
    },
    "sora2": {
        "display_name": "Sora2",
        "category": "Remote API",
        "default_backend": "api_init",
        "tags": ("api", "t2v", "i2v"),
    },
    "hailuo-2p3": {
        "display_name": "Hailuo 2.3",
        "category": "Remote API",
        "default_backend": "api_init",
        "tags": ("api", "audio-video"),
    },
}

CURATED_OVERRIDES.update(
    {
        "dreamdojo": {
            "display_name": "DreamDojo",
            "category": "Video Generation",
            "default_backend": "from_pretrained",
            "supports_from_pretrained": True,
            "default_model_ref": _dreamdojo_default_ref,
            "default_load_kwargs": _dreamdojo_default_load_kwargs,
            "default_task_type": "robot-world-generation",
            "default_prompt": "Predict the next GR-1 robot-view sequence from the recorded action trajectory.",
            "default_call_kwargs": {"num_samples": 1, "return_dict": True},
            "load_params": (
                "model_path",
                "required_components",
                "device",
                "checkpoints_dir",
                "dataset_path",
                "experiment",
                "checkpoint_interval",
                "checkpoint_shards",
                *DISPATCH_LOAD_PARAMS,
            ),
            "call_params": (
                "prompt",
                "images",
                "video",
                "interactions",
                "output_path",
                "num_samples",
                "return_dict",
            ),
            "tags": ("robot-world-model", "action-conditioned-video", "in-tree-runtime"),
        },
        "moverse": {
            "display_name": "MoVerse",
            "category": "Video Generation",
            "default_backend": "from_pretrained",
            "supports_from_pretrained": True,
            "default_model_ref": _moverse_default_ref,
            "default_load_kwargs": _moverse_default_load_kwargs,
            "default_task_type": "image-to-navigable-world",
            "default_input_path": str(_data_path("test_cases", "images", "000.png")),
            "default_prompt": (
                "Move forward through the scene while preserving geometry, appearance, and coherent camera motion."
            ),
            "default_call_kwargs": {"num_frames": 161, "fps": 16, "return_dict": True},
            "load_params": (
                "model_path",
                "required_components",
                "device",
                "moverse_path",
                "flux_path",
                "da3_path",
                "wan_path",
                "python_executable",
                *DISPATCH_LOAD_PARAMS,
            ),
            "call_params": (
                "prompt",
                "images",
                "interactions",
                "output_path",
                "num_frames",
                "fps",
                "trajectory_path",
                "return_dict",
            ),
            "tags": ("image-to-world", "camera-control", "gaussian-splat", "in-tree-runtime"),
        },
        "sana-wm-streaming": {
            "display_name": "SANA-WM Streaming",
            "category": "Video Generation",
            "default_backend": "from_pretrained",
            "supports_from_pretrained": True,
            "supports_stream": True,
            "default_model_ref": _sana_wm_streaming_default_ref,
            "default_load_kwargs": _sana_wm_streaming_default_load_kwargs,
            "default_task_type": "image-camera-video",
            "default_input_path": str(_data_path("test_cases", "sana_wm", "example_image.png")),
            "default_prompt": (
                "Move forward through a coherent scene with stable geometry and natural camera motion."
            ),
            "default_interactions": ("forward",),
            "default_call_kwargs": {
                "window_frames": 241,
                "fps": 16,
                "step": 4,
                "cfg_scale": 1.0,
                "seed": 42,
                "return_dict": True,
            },
            "load_params": (
                "model_path",
                "required_components",
                "device",
                "checkpoint_overrides",
                "gemma_path",
                *DISPATCH_LOAD_PARAMS,
            ),
            "call_params": (
                "images",
                "prompt",
                "interactions",
                "window_frames",
                "fps",
                "step",
                "cfg_scale",
                "seed",
                "return_dict",
            ),
            "tags": ("world-model", "image-to-video", "camera-control", "stream", "in-tree-runtime"),
        },
    }
)

CURATED_OVERRIDES.update(
    {
        "versecrafter": {
            "display_name": "VerseCrafter",
            "category": "Video Generation",
            "default_task_type": "i2v",
            "default_model_ref": _versecrafter_default_ref,
            "default_load_kwargs": _versecrafter_default_load_kwargs,
            "default_prompt": "A cinematic mountain valley with a gently moving camera and stable natural geometry.",
            "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
            "default_call_kwargs": {
                "width": 832,
                "height": 480,
                "num_frames": 81,
                "fps": 16,
                "num_inference_steps": 30,
                "guidance_scale": 5.0,
                "seed": 2025,
                "return_dict": True,
            },
            "call_params": (
                "prompt",
                "images",
                "image_path",
                "trajectory_npz",
                "output_path",
                "width",
                "height",
                "num_frames",
                "fps",
                "num_inference_steps",
                "guidance_scale",
                "seed",
                "return_dict",
            ),
            "load_params": (
                "model_path",
                "required_components",
                "device",
                "checkpoint_path",
                "base_model_path",
                "moge_model_path",
                "python_executable",
                "env",
            ),
            "aliases": ("verse-crafter", "TencentARC/VerseCrafter"),
            "tags": ("image-to-video", "camera-control", "4d-control", "in-tree-runtime"),
        },
        "t2v_turbo_t2v": {
            "generation_type": "t2v",
            "aliases": ("t2v-turbo-t2v", "t2v-turbo"),
            "default_prompt": "An astronaut riding a horse.",
            "default_model_ref": _t2v_turbo_base_ref,
            "default_load_kwargs": _t2v_turbo_default_load_kwargs,
            "load_params": (
                "model_path",
                "required_components",
                "device",
                "base_checkpoint",
                "model_ckpt",
                "lora_checkpoint",
                "lora_path",
                "torch_dtype",
                "weight_dtype",
                "dtype",
                "offload_mode",
            ),
            "default_call_kwargs": {
                "height": 320,
                "width": 512,
                "num_frames": 16,
                "fps": 8,
                "num_inference_steps": 8,
                "guidance_scale": 7.5,
                "seed": 0,
                "lcm_origin_steps": 200,
            },
        },
        "dap": {
            "display_name": "Depth Any Panoramas",
            "category": "Depth / Geometry",
            "supports_from_pretrained": True,
            "supports_stream": False,
            "default_backend": "from_pretrained",
            "default_task_type": "panoramic-depth-estimation",
            "default_call_kwargs": _geometry_prior_default_call_kwargs(
                input_path=_geometry_prior_image_fixture(),
                input_size=518,
            ),
            "call_params": _geometry_prior_call_params("input_size"),
            "load_params": _geometry_prior_load_params(),
            "aliases": ("depth-any-panoramas", "insta360-dap"),
            "tags": ("depth", "geometry", "panorama", "prior"),
        },
        "depth-anything-v2-prior": {
            "display_name": "Depth Anything V2 Prior",
            "category": "Depth / Geometry",
            "supports_from_pretrained": True,
            "supports_stream": False,
            "default_backend": "from_pretrained",
            "default_task_type": "monocular-depth-estimation",
            "default_call_kwargs": _geometry_prior_default_call_kwargs(
                input_path=_geometry_prior_image_fixture(),
                model="vitl",
            ),
            "call_params": _geometry_prior_call_params("model"),
            "load_params": _geometry_prior_load_params(),
            "aliases": ("dav2-prior", "depth-anything-v2"),
            "tags": ("depth", "geometry", "prior"),
        },
        "depth-anything-v3-prior": {
            "display_name": "Depth Anything 3 Prior",
            "category": "Depth / Geometry",
            "supports_from_pretrained": True,
            "supports_stream": False,
            "default_backend": "from_pretrained",
            "default_task_type": "metric-depth-estimation",
            "default_call_kwargs": _geometry_prior_default_call_kwargs(
                input_path=_geometry_prior_image_fixture(),
                focal_length=720.0,
            ),
            "call_params": _geometry_prior_call_params("focal_length"),
            "load_params": _geometry_prior_load_params(),
            "aliases": ("dav3-prior", "da3metric-large-prior"),
            "tags": ("depth", "geometry", "metric-depth", "prior"),
        },
        "dust3r": {
            "display_name": "DUSt3R",
            "category": "Depth / Geometry",
            "supports_from_pretrained": True,
            "supports_stream": False,
            "default_backend": "from_pretrained",
            "default_task_type": "multi-view-3d-reconstruction",
            "default_call_kwargs": _geometry_prior_default_call_kwargs(
                input_path=str(_data_path("test_cases", "cut3r", "examples", "001")),
                max_points=10000,
            ),
            "call_params": _geometry_prior_call_params("max_points"),
            "load_params": _geometry_prior_load_params(),
            "aliases": ("naver-dust3r", "geometric-3d-vision-made-easy"),
            "tags": ("3d", "geometry", "reconstruction", "prior"),
            "notes": "Point/multi-view 3D reconstruction; not text-to-video generation.",
        },
        "dust3r-base-model": {
            "display_name": "DUSt3R Base Model",
            "category": "Depth / Geometry",
            "supports_from_pretrained": True,
            "supports_stream": False,
            "default_backend": "from_pretrained",
            "default_task_type": "multi-view-3d-reconstruction",
            "default_call_kwargs": _geometry_prior_default_call_kwargs(
                input_path=str(_data_path("test_cases", "cut3r", "examples", "001")),
                max_points=10000,
            ),
            "call_params": _geometry_prior_call_params("max_points"),
            "load_params": _geometry_prior_load_params(),
            "aliases": ("dust3r-base",),
            "tags": ("3d", "geometry", "reconstruction", "point-reconstruction"),
            "notes": "Point/multi-view 3D reconstruction; not text-to-video generation.",
        },
        "geocalib-prior": {
            "display_name": "GeoCalib Prior",
            "category": "Depth / Geometry",
            "supports_from_pretrained": True,
            "supports_stream": False,
            "default_backend": "from_pretrained",
            "default_task_type": "camera-calibration",
            "default_call_kwargs": _geometry_prior_default_call_kwargs(
                input_path=_geometry_prior_image_fixture(),
                camera_model="pinhole",
            ),
            "call_params": _geometry_prior_call_params("camera_model"),
            "load_params": _geometry_prior_load_params(),
            "aliases": ("geocalib", "cvg-geocalib"),
            "tags": ("camera-calibration", "geometry", "prior"),
        },
        "metric3d-prior": {
            "display_name": "Metric3D Prior",
            "category": "Depth / Geometry",
            "supports_from_pretrained": True,
            "supports_stream": False,
            "default_backend": "from_pretrained",
            "default_task_type": "metric-depth-estimation",
            "default_call_kwargs": _geometry_prior_default_call_kwargs(
                input_path=_geometry_prior_image_fixture(),
                model="giant2",
                version=2,
                focal_length=720.0,
            ),
            "call_params": _geometry_prior_call_params("model", "version", "focal_length"),
            "load_params": _geometry_prior_load_params(),
            "aliases": ("metric3d", "yvanyin-metric3d"),
            "tags": ("depth", "geometry", "metric-depth", "prior"),
        },
        "prior-depth-anything": {
            "display_name": "Prior Depth Anything",
            "category": "Depth / Geometry",
            "supports_from_pretrained": True,
            "supports_stream": False,
            "default_backend": "from_pretrained",
            "default_task_type": "depth-completion",
            "default_call_kwargs": _geometry_prior_default_call_kwargs(
                input_path=_geometry_prior_image_fixture(),
            ),
            "call_params": _geometry_prior_call_params("prompt_depth_path", "prior_depth_path"),
            "load_params": _geometry_prior_load_params(),
            "aliases": ("priorda", "prior-depth-anything-prior"),
            "tags": ("depth", "geometry", "depth-completion", "prior"),
        },
        "track-anything-prior": {
            "display_name": "Segment and Track Anything Prior",
            "category": "Depth / Geometry",
            "supports_from_pretrained": True,
            "supports_stream": False,
            "default_backend": "from_pretrained",
            "default_prompt": "robot.",
            "default_task_type": "video-object-segmentation",
            "default_call_kwargs": _geometry_prior_default_call_kwargs(
                input_path=_geometry_prior_video_fixture(),
                max_frames=4,
            ),
            "call_params": _geometry_prior_call_params("max_frames"),
            "load_params": _geometry_prior_load_params(),
            "aliases": ("segment-and-track-anything", "track-anything", "aot-deaot-l"),
            "tags": ("tracking", "segmentation", "geometry", "prior"),
        },
        "unidepth-v2-prior": {
            "display_name": "UniDepth V2 Prior",
            "category": "Depth / Geometry",
            "supports_from_pretrained": True,
            "supports_stream": False,
            "default_backend": "from_pretrained",
            "default_task_type": "metric-depth-estimation",
            "default_call_kwargs": _geometry_prior_default_call_kwargs(
                input_path=_geometry_prior_image_fixture(),
                type="l",
                focal_length=720.0,
                max_points=50_000,
            ),
            "call_params": _geometry_prior_call_params("type", "focal_length"),
            "load_params": _geometry_prior_load_params(),
            "aliases": ("unidepth-v2", "unidepth-v2-vitl14", "unidepth_v2_vitl14"),
            "tags": ("depth", "geometry", "metric-depth", "prior"),
        },
        "unik3d-prior": {
            "display_name": "UniK3D Prior",
            "category": "Depth / Geometry",
            "supports_from_pretrained": True,
            "supports_stream": False,
            "default_backend": "from_pretrained",
            "default_task_type": "metric-3d-estimation",
            "default_call_kwargs": _geometry_prior_default_call_kwargs(
                input_path=_geometry_prior_image_fixture(),
                type="l",
                camera_model="auto",
                max_points=50_000,
            ),
            "call_params": _geometry_prior_call_params("type", "camera_model"),
            "load_params": _geometry_prior_load_params(),
            "aliases": ("unik3d", "unik3d-vitl"),
            "tags": ("depth", "geometry", "panorama", "prior"),
        },
        "video-depth-anything-prior": {
            "display_name": "Video Depth Anything Prior",
            "category": "Depth / Geometry",
            "supports_from_pretrained": True,
            "supports_stream": False,
            "default_backend": "from_pretrained",
            "default_task_type": "video-depth-estimation",
            "default_call_kwargs": _geometry_prior_default_call_kwargs(
                input_path=_geometry_prior_video_fixture(),
                model="vitl",
                input_size=518,
                max_frames=4,
            ),
            "call_params": _geometry_prior_call_params("model", "input_size", "max_frames"),
            "load_params": _geometry_prior_load_params(),
            "aliases": ("video-depth-anything", "vda-prior"),
            "tags": ("depth", "geometry", "video-depth", "prior"),
        },
    }
)


def _normalize_model_id(model_id: str) -> str:
    if model_id == "matrix-game-1":
        return model_id
    if model_id == "matrix-game-2":
        return model_id
    if model_id == "matrix-game-3":
        return model_id
    if model_id == "depth-anything-v3":
        return model_id
    if model_id == "vggt-omega":
        return model_id
    return model_id


def _canonical_runtime_entries() -> Iterable[CatalogEntry]:
    runtime_specs = {
        "allegro_ti2v": {
            "generation_type": "i2v",
            "aliases": ("allegro", "allegro-ti2v"),
            "default_prompt": "The car drives along the road",
            "default_input_path": str(
                _data_path("test_cases", "stable_virtual_camera", "basic", "blue-car.jpg")
            ),
            "load_params": (
                *DISPATCH_LOAD_PARAMS,
                "model_path",
                "required_components",
                "device",
                "lazy",
                "guidance_scale",
                "num_sampling_steps",
                "seed",
            ),
            "default_load_kwargs": {
                "guidance_scale": 8,
                "num_sampling_steps": 100,
                "seed": 1427329220,
            },
            "default_call_kwargs": {"fps": 15},
            "default_model_ref": lambda: _checkpoint_model_ref(
                "Allegro-TI2V",
                "rhymes-ai--Allegro-TI2V",
                fallback="rhymes-ai/Allegro-TI2V",
            ),
        },
        **_cogvideox_variant_specs(),
        "dynamicrafter_1024_i2v": {
            "generation_type": "i2v",
            "aliases": ("dynamicrafter-1024-i2v",),
            "default_prompt": "A young man holding a bright sparkler gently turns his head as sparks scatter through the night air, with natural motion and stable facial details.",
            "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
            "default_model_ref": lambda: _checkpoint_model_ref(
                "Doubiiu--DynamiCrafter_1024/model.ckpt",
                "DynamiCrafter_1024/model.ckpt",
                "AliVideo--DynamiCrafter_1024/model.ckpt",
            ),
        },
        "dynamicrafter_512_i2v": {
            "generation_type": "i2v",
            "aliases": ("dynamicrafter", "dynamicrafter-512-i2v"),
            "default_prompt": "A young man holding a bright sparkler gently turns his head as sparks scatter through the night air, with natural motion and stable facial details.",
            "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
            "default_model_ref": lambda: _checkpoint_model_ref(
                "Doubiiu--DynamiCrafter_512/model.ckpt",
                "DynamiCrafter_512/model.ckpt",
                "AliVideo--DynamiCrafter_512/model.ckpt",
            ),
        },
        "easyanimate_i2v": {
            "generation_type": "i2v",
            "aliases": ("easyanimate-i2v", "easyanimate"),
            "default_prompt": (
                "A young man holding a bright sparkler gently turns his head and smiles as sparks scatter "
                "through the night air, with natural motion, stable facial details, and a steady camera."
            ),
            "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
            "default_model_ref": lambda: _checkpoint_model_ref(
                "hfd/alibaba-pai--EasyAnimateV5.1-7b-zh-InP",
                "alibaba-pai--EasyAnimateV5.1-7b-zh-InP",
                fallback="alibaba-pai/EasyAnimateV5.1-7b-zh-InP",
            ),
        },
        "gen_3_i2v": {
            "generation_type": "i2v",
            "aliases": ("gen-3-i2v", "gen3-i2v", "gen-3"),
            "default_prompt": "A gentle camera move around the subject with natural motion and stable lighting.",
            "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
        },
        "ltx_video_i2v": {
            "generation_type": "i2v",
            "aliases": ("ltx-video-i2v", "ltx-video"),
            "default_prompt": "A young man holding a bright sparkler gently turns his head as sparks scatter through the night air, with natural motion and stable facial details.",
            "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
            "default_model_ref": lambda: _checkpoint_model_ref(
                "LTX-Video",
                "Lightricks--LTX-Video",
                fallback="Lightricks/LTX-Video",
            ),
        },
        "ltx2_i2v": {
            "generation_type": "i2v",
            "aliases": ("ltx2-i2v", "ltx2", "ltx-2", "ltx-2-i2v"),
            "default_prompt": "A young man holding a bright sparkler gently turns his head as sparks scatter through the night air, with natural motion and stable facial details.",
            "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
            "default_call_kwargs": {
                "num_frames": 121,
                "fps": 24,
                "height": 512,
                "width": 768,
                "num_inference_steps": 11,
                "guidance_scale": 1.0,
                "seed": 171198,
            },
            "call_params": (
                "prompt",
                "images",
                "output_path",
                "return_dict",
                "fps",
                "num_frames",
                "num_inference_steps",
                "height",
                "width",
                "guidance_scale",
                "seed",
            ),
            "load_params": (
                "model_path",
                "required_components",
                "device",
                "lazy",
                "height",
                "width",
                "num_frames",
                "frame_rate",
                "num_inference_steps",
                "guidance_scale",
                "seed",
                "offload_mode",
                "quantization",
            ),
            "default_model_ref": lambda: _checkpoint_model_ref(
                "LTX-2/ltx-2-19b-distilled.safetensors",
                "Lightricks--LTX-2/ltx-2-19b-distilled.safetensors",
                fallback="Lightricks/LTX-2",
            ),
        },
        "ltx2_3_i2v": {
            "generation_type": "i2v",
            "aliases": ("ltx-2.x", "ltx2.3", "ltx-2.3", "ltx2.3-i2v", "ltx-2.3-i2v"),
            "default_prompt": "A young man holding a bright sparkler gently turns his head as sparks scatter through the night air, with natural motion and stable facial details.",
            "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
            "default_call_kwargs": {
                "num_frames": 121,
                "fps": 24,
                "height": 512,
                "width": 768,
                "num_inference_steps": 11,
                "guidance_scale": 1.0,
                "seed": 0,
            },
            "call_params": (
                "prompt",
                "images",
                "output_path",
                "return_dict",
                "fps",
                "num_frames",
                "num_inference_steps",
                "height",
                "width",
                "guidance_scale",
                "seed",
            ),
            "load_params": (
                "model_path",
                "required_components",
                "device",
                "lazy",
                "height",
                "width",
                "num_frames",
                "frame_rate",
                "num_inference_steps",
                "guidance_scale",
                "seed",
                "offload_mode",
                "quantization",
            ),
            "default_model_ref": lambda: _checkpoint_model_ref(
                "LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors",
                "Lightricks--LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors",
                fallback="Lightricks/LTX-2.3",
            ),
        },
        "minimax_i2v": {
            "generation_type": "i2v",
            "aliases": ("minimax-i2v",),
            "default_prompt": "A gentle camera move around the subject with natural motion and stable lighting.",
            "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
        },
        "t2v_turbo_t2v": {
            "generation_type": "t2v",
            "aliases": ("t2v-turbo-t2v", "t2v-turbo"),
            "default_prompt": "An astronaut riding a horse.",
            "default_model_ref": _t2v_turbo_base_ref,
            "default_load_kwargs": _t2v_turbo_default_load_kwargs,
            "load_params": (
                "model_path",
                "required_components",
                "device",
                "base_checkpoint",
                "model_ckpt",
                "lora_checkpoint",
                "lora_path",
                "torch_dtype",
                "weight_dtype",
                "dtype",
                "offload_mode",
            ),
            "default_call_kwargs": {
                "height": 320,
                "width": 512,
                "num_frames": 16,
                "fps": 8,
                "num_inference_steps": 8,
                "guidance_scale": 7.5,
                "seed": 0,
                "lcm_origin_steps": 200,
            },
        },
        "vchitect_2_t2v": {
            "generation_type": "t2v",
            "aliases": ("vchitect-2-t2v", "vchitect2-t2v", "vchitect-2"),
            "default_prompt": "A tiger walks in the forest, photorealistic, 4k, high definition",
            "default_model_ref": _vchitect2_default_ref,
            "load_params": (
                "model_path",
                "required_components",
                "device",
                "torch_dtype",
                "dtype",
                "offload_mode",
            ),
            "default_call_kwargs": {
                "height": 432,
                "width": 768,
                "num_frames": 40,
                "num_inference_steps": 100,
                "guidance_scale": 7.5,
                "fps": 8,
                "seed": 0,
            },
        },
        "videocrafter1_i2v": {
            "generation_type": "i2v",
            "aliases": ("videocrafter1-i2v",),
            "default_prompt": "horses are walking on the grassland",
            "default_input_path": str(_data_path("test_cases", "videocrafter", "i2v_prompts", "horse.png")),
            "default_model_ref": lambda: _checkpoint_model_ref(
                "VideoCrafter--Image2Video-512/model.ckpt",
                "VideoCrafter/videocrafter_i2v_512_v1.ckpt",
                "video_models/videocrafter_i2v_512_v1.ckpt",
            ),
            "default_call_kwargs": {
                "num_frames": 16,
                "fps": 8,
                "num_inference_steps": 50,
                "height": 320,
                "width": 512,
                "seed": 123,
            },
        },
        "videocrafter1_t2v": {
            "generation_type": "t2v",
            "aliases": ("videocrafter1-t2v",),
            "default_prompt": "A tiger walks in the forest, photorealistic, 4k, high definition",
            "default_model_ref": lambda: _checkpoint_model_ref(
                "VideoCrafter--Text2Video-1024/model.ckpt",
                "VideoCrafter/videocrafter_t2v_1024_v1.ckpt",
                "video_models/videocrafter_t2v_1024_v1.ckpt",
            ),
            "default_call_kwargs": {
                "num_frames": 16,
                "fps": 8,
                "num_inference_steps": 50,
                "height": 576,
                "width": 1024,
                "seed": 123,
            },
        },
        "videocrafter2_t2v": {
            "generation_type": "t2v",
            "aliases": ("videocrafter", "videocrafter2-t2v"),
            "default_prompt": "A tiger walks in the forest, photorealistic, 4k, high definition",
            "default_model_ref": lambda: _checkpoint_model_ref(
                "VideoCrafter--VideoCrafter2/model.ckpt",
                "VideoCrafter/videocrafter_t2v_512_v2.ckpt",
                "video_models/videocrafter_t2v_512_v2.ckpt",
            ),
            "default_call_kwargs": {
                "num_frames": 16,
                "fps": 28,
                "num_inference_steps": 50,
                "height": 320,
                "width": 512,
                "seed": 123,
            },
        },
        "wan2.1_i2v": {
            "generation_type": "i2v",
            "ast_slug": "wan2.1-i2v-14b-480p",
            "display_name": "Wan2.1 I2V",
            "aliases": ("wan2.1-i2v", "wan2p1-i2v", "wan2-1-i2v"),
            "default_model_ref": _wan_2p1_i2v_default_ref,
            "default_prompt": "A young man holding a bright sparkler gently turns his head as sparks scatter through the night air, with natural motion and stable facial details.",
            "default_input_path": str(_data_path("test_cases", "studio_demo", "00", "image.jpg")),
            "default_call_kwargs": {
                "height": 480,
                "width": 832,
                "num_frames": 81,
                "fps": 16,
                "num_inference_steps": 40,
                "shift": 3.0,
                "guidance_scale": 5.0,
                "seed": 42,
            },
            "call_params": _native_diffusion_video_call_params(),
            "load_params": _native_diffusion_video_load_params(),
        },
        "wan2.1_t2v": {
            "generation_type": "t2v",
            "ast_slug": "wan2.1-t2v-1.3b",
            "display_name": "Wan2.1 T2V",
            "aliases": ("wan2.1", "wan2.1-t2v", "wan2p1-t2v", "wan2-1-t2v"),
            "default_model_ref": _wan_2p1_t2v_default_ref,
            "default_prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
            "default_call_kwargs": {
                "height": 480,
                "width": 832,
                "num_frames": 81,
                "fps": 16,
                "num_inference_steps": 50,
                "shift": 8.0,
                "guidance_scale": 6.0,
                "seed": 42,
            },
            "call_params": _native_diffusion_video_call_params(),
            "load_params": _native_diffusion_video_load_params(),
        },
    }
    runtime_specs[COGVIDEOX_STUDIO_PARENT_ID] = {
        **runtime_specs[COGVIDEOX_STUDIO_PARENT_ID],
        "display_name": "CogVideoX",
        "summary": (
            "CogVideoX text-to-video and image-to-video family. "
            "Select 2B T2V, 5B T2V, or 5B I2V."
        ),
        "aliases": _cogvideox_aliases(),
        "extra_variants": _cogvideox_workspace_variants,
        "tags": ("video", "t2v", "i2v"),
    }

    ast_by_slug: Dict[str, _AstPipelineInfo] = {}
    for info in _discover_ast_pipelines():
        ast_by_slug.setdefault(info.model_id.replace("_", "-"), info)

    for canonical_name, spec in runtime_specs.items():
        slug = canonical_name
        slug_key = str(spec.get("ast_slug") or canonical_name.replace("_", "-"))
        matched = ast_by_slug.get(slug_key)
        module_path = matched.module_path if matched is not None else _runtime_module_path(canonical_name)
        class_name = matched.class_name if matched is not None else _runtime_class_name(canonical_name)
        family = matched.family if matched is not None else canonical_name.split("_", maxsplit=1)[0]
        display_name = str(spec.get("display_name") or _display_name_from_slug(slug.replace("_", "-")))
        generation_type = spec.get("generation_type", "i2v")
        summary = str(spec.get("summary") or f"Independent {generation_type.upper()} video pipeline for this model.")
        if "call_params" in spec:
            call_params = tuple(_resolve_override_value(spec.get("call_params")) or ())
        else:
            matched_call_params = tuple(matched.call_params) if matched is not None else ()
            call_params = tuple(dict.fromkeys((*matched_call_params, *_official_video_call_params())))
        if "stream_params" in spec:
            stream_params = tuple(_resolve_override_value(spec.get("stream_params")) or ())
        else:
            matched_stream_params = tuple(matched.stream_params) if matched is not None else ()
            stream_params = tuple(dict.fromkeys((*matched_stream_params, *_official_video_call_params())))
        default_call_kwargs = dict(_resolve_override_value(spec.get("default_call_kwargs", {"fps": 16})) or {})
        if canonical_name.startswith("videocrafter"):
            call_params = (
                "prompt",
                "output_path",
                "fps",
                "return_dict",
                "frames",
                "num_frames",
                "frame_num",
                "ddim_steps",
                "num_inference_steps",
                "steps",
                "sample_steps",
                "infer_steps",
                "sampling_steps",
                "ddim_eta",
                "height",
                "width",
                "seed",
                "unconditional_guidance_scale",
            )
            if generation_type == "i2v":
                call_params = ("prompt", "images", *call_params[1:])
            stream_params = call_params
            default_call_kwargs.update(
                {
                    "ddim_eta": 1.0,
                    "unconditional_guidance_scale": 12.0,
                }
            )
        yield CatalogEntry(
            model_id=slug,
            display_name=display_name,
            module_path=module_path,
            class_name=class_name,
            family=family,
            category="Video Generation",
            summary=summary,
            call_params=call_params,
            stream_params=stream_params,
            load_params=tuple(
                _resolve_override_value(
                    spec.get(
                        "load_params",
                        matched.load_params
                        if matched is not None
                        else ("model_path", "required_components", "device", "lazy"),
                    )
                )
                or ()
            ),
            supports_stream=matched.supports_stream if matched is not None else True,
            supports_from_pretrained=True,
            supports_api_init=matched.supports_api_init if matched is not None else False,
            runtime_kind="default",
            default_backend="from_pretrained",
            default_model_ref=str(_resolve_override_value(spec.get("default_model_ref", "")) or ""),
            default_prompt=str(_resolve_override_value(spec.get("default_prompt", "")) or ""),
            default_input_path=str(_resolve_override_value(spec.get("default_input_path", "")) or ""),
            default_task_type=str(_resolve_override_value(spec.get("default_task_type", "")) or ""),
            default_load_kwargs=dict(_resolve_override_value(spec.get("default_load_kwargs", {})) or {}),
            default_interactions=(),
            default_call_kwargs=default_call_kwargs,
            extra_variants=_resolve_extra_variants(spec.get("extra_variants", ())),
            aliases=tuple(spec.get("aliases", [])),
            tags=tuple(spec.get("tags") or ("video", generation_type)),
            notes="Integrated runtime modules are expected under worldfoundry.pipelines; pass only WorldFoundry checkpoint/cache paths or HF repo ids as model refs.",
        )


def _runtime_class_name(model_name: str) -> str:
    tokens = model_name.replace(".", "_").split("_")
    pretty_tokens = []
    for token in tokens:
        if token in {"i2v", "t2v", "ti2v"}:
            pretty_tokens.append(token.upper())
        else:
            pretty_tokens.append(token.capitalize())
    return "".join(pretty_tokens) + "Pipeline"


def _parse_method_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    args: list[str] = []
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        if arg.arg in {"self", "cls"}:
            continue
        args.append(arg.arg)
    return tuple(dict.fromkeys(args))


def _resolve_class_methods(
    class_nodes: dict[str, ast.ClassDef],
    class_name: str,
    cache: dict[str, dict[str, tuple[str, ...]]],
    seen: frozenset[str] = frozenset(),
) -> dict[str, tuple[str, ...]]:
    """Collect local inherited method signatures for AST-only catalog discovery."""

    if class_name in cache:
        return dict(cache[class_name])
    if class_name in seen or class_name not in class_nodes:
        return {}
    class_node = class_nodes[class_name]
    methods: dict[str, tuple[str, ...]] = {}
    next_seen = seen | {class_name}
    for base in class_node.bases:
        base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
        methods.update(_resolve_class_methods(class_nodes, base_name, cache, next_seen))
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods[item.name] = _parse_method_args(item)
    cache[class_name] = methods
    return dict(methods)


def _class_inherits_any(
    class_nodes: Mapping[str, ast.ClassDef],
    class_name: str,
    expected_bases: frozenset[str],
    seen: frozenset[str] = frozenset(),
) -> bool:
    """Resolve local inheritance far enough to recognize shared infra bases.

    Catalog discovery deliberately parses pipeline modules without importing
    heavyweight model dependencies.  Model-specific native pipelines commonly
    inherit their implementation from a base imported from another module, and
    variants then inherit from that local model class.  Looking only at methods
    declared in the leaf class therefore produced empty Workspace contracts.
    """

    if class_name in seen:
        return False
    node = class_nodes.get(class_name)
    if node is None:
        return class_name in expected_bases
    next_seen = seen | {class_name}
    for base in node.bases:
        base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
        if base_name in expected_bases:
            return True
        if base_name in class_nodes and _class_inherits_any(
            class_nodes,
            base_name,
            expected_bases,
            next_seen,
        ):
            return True
    return False


def _class_string_constant(node: ast.ClassDef, name: str) -> str | None:
    for item in node.body:
        if not isinstance(item, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in item.targets):
            continue
        if isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
            return item.value.value
    return None


def _class_bool_constant(node: ast.ClassDef, name: str) -> bool | None:
    for item in node.body:
        if not isinstance(item, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in item.targets):
            continue
        if isinstance(item.value, ast.Constant) and isinstance(item.value.value, bool):
            return item.value.value
    return None


def _pipeline_file_candidates() -> tuple[Path, ...]:
    """Return Studio-visible pipeline files without descending into vendored runtimes.

    Args:
        None.
    """
    candidates: list[Path] = []
    for family_dir in sorted(path for path in PIPELINES_ROOT.iterdir() if path.is_dir()):
        candidates.extend(
            sorted(path for path in family_dir.iterdir() if path.name.startswith("pipeline_") and path.suffix == ".py")
        )
    for relative_dir in NESTED_PIPELINE_DIRS:
        nested_dir = PIPELINES_ROOT / relative_dir
        if nested_dir.is_dir():
            candidates.extend(
                sorted(path for path in nested_dir.iterdir() if path.name.startswith("pipeline_") and path.suffix == ".py")
            )
    return tuple(dict.fromkeys(candidates))


def _action_synthesis_file_candidates() -> tuple[Path, ...]:
    """Return first-class action synthesis files used by Studio catalog."""

    if not ACTION_SYNTHESIS_ROOT.is_dir():
        return tuple()
    return tuple(
        sorted(
            path
            for path in ACTION_SYNTHESIS_ROOT.glob("*/*_synthesis.py")
            if path.name != "base_action_synthesis.py"
        )
    )


def _literal_keyword(call: ast.Call, name: str) -> Any:
    for keyword in call.keywords:
        if keyword.arg != name:
            continue
        try:
            return ast.literal_eval(keyword.value)
        except Exception:
            return None
    return None


def _discover_component_pipelines() -> tuple[_AstPipelineInfo, ...]:
    path = PIPELINES_ROOT / "component_pipelines.py"
    if not path.is_file():
        return tuple()
    rel_path = path.relative_to(_source_root())
    module_path = ".".join(rel_path.with_suffix("").parts)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    discovered: list[_AstPipelineInfo] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        if not isinstance(node.value.func, ast.Name) or node.value.func.id != "_component_pipeline_class":
            continue
        if not node.value.args:
            continue
        try:
            class_name = str(ast.literal_eval(node.value.args[0]))
        except Exception:
            continue
        model_id = _literal_keyword(node.value, "model_id")
        if not isinstance(model_id, str) or not model_id:
            continue
        generation_type = _literal_keyword(node.value, "generation_type")
        family = str(generation_type or model_id).replace("-", "_")
        discovered.append(
            _AstPipelineInfo(
                model_id=_normalize_model_id(model_id),
                display_name=_display_name_from_slug(model_id),
                module_path=module_path,
                class_name=class_name,
                family=family,
                call_params=("prompt", "images", "video", "interactions", "output_path", "fps"),
                stream_params=("prompt", "images", "video", "interactions", "output_path", "fps"),
                load_params=("model_path", "required_components", "device", "model_id"),
                supports_stream=True,
                supports_from_pretrained=True,
                supports_api_init=False,
                base_names=("ComponentPipeline", "PipelineABC"),
            )
        )
    return tuple(discovered)


def _discover_action_syntheses() -> tuple[_AstPipelineInfo, ...]:
    discovered: List[_AstPipelineInfo] = []
    for path in _action_synthesis_file_candidates():
        rel_path = path.relative_to(_source_root())
        module_path = ".".join(rel_path.with_suffix("").parts)
        family = path.parent.name
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        class_nodes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        method_cache: dict[str, dict[str, tuple[str, ...]]] = {}

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not node.name.endswith("Synthesis"):
                continue
            base_names = tuple(
                base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
                for base in node.bases
            )
            model_id = _class_string_constant(node, "MODEL_ID")
            inherits_action_synthesis = any(
                base_name == "ActionModelSynthesis" or base_name.endswith("Synthesis")
                for base_name in base_names
            )
            if not model_id or not inherits_action_synthesis:
                continue
            methods = _resolve_class_methods(class_nodes, node.name, method_cache)

            slug = _normalize_model_id(model_id)
            discovered.append(
                _AstPipelineInfo(
                    model_id=slug,
                    display_name=_display_name_from_slug(slug),
                    module_path=module_path,
                    class_name=node.name,
                    family=family,
                    call_params=methods.get(
                        "predict",
                        methods.get("__call__", OFFICIAL_ACTION_CALL_PARAMS),
                    ),
                    stream_params=methods.get("stream", tuple()),
                    load_params=methods.get("from_pretrained", OFFICIAL_ACTION_LOAD_PARAMS),
                    supports_stream="stream" in methods,
                    supports_from_pretrained=True,
                    supports_api_init="api_init" in methods,
                    base_names=base_names,
                )
            )
    return tuple(discovered)


@lru_cache(maxsize=1)
def _discover_ast_pipelines() -> tuple[_AstPipelineInfo, ...]:
    discovered: List[_AstPipelineInfo] = []
    for path in _pipeline_file_candidates():
        rel_path = path.relative_to(_source_root())
        module_path = ".".join(rel_path.with_suffix("").parts)
        family = path.parent.name
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        class_nodes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        method_cache: dict[str, dict[str, tuple[str, ...]]] = {}
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name.startswith("_"):
                continue
            if not node.name.endswith("Pipeline"):
                continue
            if _class_bool_constant(node, "ABSTRACT_PIPELINE") is True:
                continue
            methods = _resolve_class_methods(class_nodes, node.name, method_cache)

            slug = _normalize_model_id(_class_string_constant(node, "MODEL_ID") or _slug_from_filename(path))
            display_name = _display_name_from_slug(slug)
            base_names = tuple(
                base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
                for base in node.bases
            )
            inherits_official_video = "OfficialVideoPipeline" in base_names
            inherits_native_diffusion = _class_inherits_any(
                class_nodes,
                node.name,
                frozenset({"NativeVisualDiffusionPipeline", "NativeTextToVideoPipeline"}),
            )
            inherited_official_call = _official_video_call_params() if inherits_official_video else tuple()
            inherited_official_load = _official_video_load_params() if inherits_official_video else tuple()
            inherited_native_call = (
                _native_diffusion_video_call_params() if inherits_native_diffusion else tuple()
            )
            inherited_native_load = (
                _native_diffusion_video_load_params() if inherits_native_diffusion else tuple()
            )
            discovered.append(
                _AstPipelineInfo(
                    model_id=slug,
                    display_name=display_name,
                    module_path=module_path,
                    class_name=node.name,
                    family=family,
                    call_params=methods.get("__call__", inherited_native_call or inherited_official_call),
                    stream_params=methods.get("stream", inherited_native_call or inherited_official_call),
                    load_params=methods.get(
                        "from_pretrained",
                        methods.get("api_init", inherited_native_load or inherited_official_load),
                    ),
                    supports_stream="stream" in methods or inherits_official_video or inherits_native_diffusion,
                    supports_from_pretrained=(
                        "from_pretrained" in methods or inherits_official_video or inherits_native_diffusion
                    ),
                    supports_api_init="api_init" in methods,
                    base_names=base_names,
                )
            )
    return tuple(discovered)


@lru_cache(maxsize=1)
def _discover_catalog_infos() -> tuple[_AstPipelineInfo, ...]:
    return (*_discover_ast_pipelines(), *_discover_component_pipelines(), *_discover_action_syntheses())


def _build_entry(info: _AstPipelineInfo) -> CatalogEntry:
    override = CURATED_OVERRIDES.get(info.model_id) or CURATED_OVERRIDES.get(_catalog_id_key(info.model_id), {})
    override = {**override, **studio_contract_override(info.model_id)}
    inferred_category = _category_from_family(info.family, info.class_name, info.call_params)
    if (
        inferred_category == "Video Generation"
        and info.module_path.startswith("worldfoundry.synthesis.action_generation.")
        and info.class_name.endswith("Synthesis")
    ):
        inferred_category = "Embodied Action"
    category = override.get(
        "category",
        inferred_category,
    )
    display_name = override.get("display_name", info.display_name)
    runtime_kind = override.get("runtime_kind", _runtime_kind_from_family(info.model_id, info.family))
    supports_from_pretrained = bool(override.get("supports_from_pretrained", info.supports_from_pretrained))
    supports_api_init = bool(override.get("supports_api_init", info.supports_api_init))
    default_backend = override.get(
        "default_backend",
        _default_backend(supports_api_init, supports_from_pretrained),
    )
    supports_stream = bool(override.get("supports_stream", info.supports_stream))
    summary = override.get(
        "summary",
        _summary_from_category(category, display_name, supports_stream),
    )
    if category == "Video Generation":
        inferred_prompt = "A slow exploratory camera move through a believable scene with stable geometry and lighting."
    elif category in {"Embodied Action", "Visual Action"}:
        inferred_prompt = "Follow the instruction and predict the next executable action chunk."
    else:
        inferred_prompt = "Reconstruct the scene and surface geometry with stable camera motion."
    default_prompt = override.get("default_prompt", inferred_prompt)
    default_model_ref = _resolve_override_value(override.get("default_model_ref", ""))
    default_load_kwargs = _resolve_override_value(override.get("default_load_kwargs", {})) or {}
    default_call_kwargs = _resolve_override_value(override.get("default_call_kwargs", {})) or {}
    extra_variants = _resolve_extra_variants(override.get("extra_variants", ()))
    call_params = tuple(_resolve_override_value(override.get("call_params", info.call_params)) or ())
    input_params = tuple(_resolve_override_value(override.get("input_params", tuple())) or ())
    stream_params = tuple(_resolve_override_value(override.get("stream_params", info.stream_params)) or ())
    load_params = tuple(_resolve_override_value(override.get("load_params", info.load_params)) or ())
    if category in {"Embodied Action", "Visual Action"}:
        load_params = _append_dispatch_load_params(load_params)
    return CatalogEntry(
        model_id=info.model_id,
        display_name=display_name,
        module_path=info.module_path,
        class_name=info.class_name,
        family=info.family,
        category=category,
        summary=summary,
        call_params=call_params,
        input_params=input_params,
        stream_params=stream_params,
        load_params=load_params,
        supports_stream=supports_stream,
        supports_from_pretrained=supports_from_pretrained,
        supports_api_init=supports_api_init,
        runtime_kind=runtime_kind,
        default_backend=default_backend,
        default_model_ref=_resolve_default_model_ref(
            info.model_id,
            default_model_ref,
        ),
        default_endpoint=override.get("default_endpoint", ""),
        default_prompt=default_prompt,
        default_input_path=str(_resolve_override_value(override.get("default_input_path", "")) or ""),
        default_task_type=override.get("default_task_type", ""),
        default_interactions=tuple(override.get("default_interactions", tuple())),
        default_load_kwargs=dict(default_load_kwargs),
        default_call_kwargs=dict(default_call_kwargs),
        extra_variants=extra_variants,
        suggested_task_types=tuple(override.get("suggested_task_types", tuple())),
        aliases=_coerce_aliases(*override.get("aliases", tuple())),
        tags=tuple(override.get("tags", tuple())),
        env_hints=tuple(override.get("env_hints", tuple())),
        notes=override.get("notes", ""),
    )


@lru_cache(maxsize=1)
def discover_catalog() -> tuple[CatalogEntry, ...]:
    """Resolve shared checkpoint roots once per catalog build."""
    roots = studio_hfd_cache_roots(project_roots=_project_root_candidates())
    children: dict[Path, frozenset[str]] = {}
    for root in roots:
        try:
            children[root] = frozenset(path.name for path in root.iterdir())
        except (FileNotFoundError, NotADirectoryError):
            children[root] = frozenset()
        except OSError:
            # Fall back to individual checks when a root cannot be enumerated.
            continue
    token = _CATALOG_CACHE_ROOTS.set(_CatalogPathSnapshot(roots, children))
    try:
        return _discover_catalog_entries()
    finally:
        _CATALOG_CACHE_ROOTS.reset(token)


def _discover_catalog_entries() -> tuple[CatalogEntry, ...]:
    by_module: Dict[tuple[str, str], CatalogEntry] = {}
    for info in _discover_catalog_infos():
        entry = _build_entry(info)
        if entry.model_id in ABSTRACT_RUNTIME_MODEL_IDS:
            continue
        if entry.model_id in STUDIO_HIDDEN_CATALOG_MODEL_IDS:
            continue
        if any(existing.model_id == entry.model_id for existing in by_module.values()):
            continue
        by_module[(entry.module_path, entry.class_name)] = entry

    for entry in _canonical_runtime_entries():
        if entry.model_id in ABSTRACT_RUNTIME_MODEL_IDS:
            continue
        if entry.model_id in STUDIO_HIDDEN_CATALOG_MODEL_IDS:
            continue
        key = (entry.module_path, entry.class_name)
        previous = by_module.get(key)
        if previous is not None:
            merged_aliases = _coerce_aliases(*previous.aliases, *entry.aliases, previous.model_id.replace("_", "-"))
            entry = CatalogEntry(
                model_id=entry.model_id,
                display_name=entry.display_name,
                module_path=entry.module_path,
                class_name=entry.class_name,
                family=entry.family,
                category=entry.category,
                summary=entry.summary,
                call_params=entry.call_params,
                input_params=entry.input_params,
                stream_params=entry.stream_params,
                load_params=entry.load_params,
                supports_stream=entry.supports_stream,
                supports_from_pretrained=entry.supports_from_pretrained,
                supports_api_init=entry.supports_api_init,
                runtime_kind=entry.runtime_kind,
                default_backend=entry.default_backend,
                default_model_ref=entry.default_model_ref,
                default_endpoint=entry.default_endpoint,
                default_prompt=entry.default_prompt,
                default_input_path=entry.default_input_path,
                default_task_type=entry.default_task_type,
                default_interactions=entry.default_interactions,
                default_load_kwargs=entry.default_load_kwargs,
                default_call_kwargs=entry.default_call_kwargs,
                extra_variants=entry.extra_variants or previous.extra_variants,
                suggested_task_types=entry.suggested_task_types,
                aliases=merged_aliases,
                tags=entry.tags,
                env_hints=entry.env_hints,
                notes=entry.notes or previous.notes,
            )
        by_module[key] = entry

    entries = list(by_module.values())
    entries.sort(key=lambda item: (item.category, item.display_name.lower(), item.model_id))
    return tuple(entries)


def catalog_as_table(entries: Sequence[CatalogEntry] | None = None) -> list[list[str]]:
    active_entries = discover_catalog() if entries is None else entries
    return [entry.to_table_row() for entry in active_entries]


def catalog_stats(entries: Sequence[CatalogEntry] | None = None) -> Dict[str, int]:
    active_entries = discover_catalog() if entries is None else entries
    return {
        "total": len(active_entries),
        "video": sum(1 for item in active_entries if item.category == "Video Generation"),
        "v2v": sum(1 for item in active_entries if item.category == "Video-to-Video"),
        "scene": sum(1 for item in active_entries if item.category == "3D Scene"),
        "api": sum(1 for item in active_entries if item.category == "Remote API"),
        "action": sum(1 for item in active_entries if item.category in {"Embodied Action", "Visual Action"}),
        "stream": sum(1 for item in active_entries if item.supports_stream),
    }


def _find_exact_entry_without_full_catalog(model_id: str) -> CatalogEntry | None:
    """Build one AST-discovered entry when the caller supplied its exact id.

    Native single-model frontends should not pay the full catalog's checkpoint
    fallback resolution cost before opening their HTTP port. Alias matching still
    falls back to ``discover_catalog()`` below because aliases live in curated
    metadata and may require canonical entry merging.
    """

    normalized = _normalize_model_id(model_id)
    if not normalized:
        return None
    parent_id = _sana_studio_parent_id(normalized) or _cogvideox_studio_parent_id(normalized)
    ast_model_id = FAST_CANONICAL_AST_ALIASES.get(parent_id or normalized, parent_id or normalized)
    for info in _discover_catalog_infos():
        if info.model_id in ABSTRACT_RUNTIME_MODEL_IDS:
            continue
        if info.model_id in STUDIO_HIDDEN_CATALOG_MODEL_IDS:
            continue
        if _catalog_id_key(info.model_id) == _catalog_id_key(ast_model_id):
            entry = _build_entry(info)
            if parent_id is not None:
                entry = replace(
                    entry,
                    model_id=parent_id,
                    aliases=_coerce_aliases(*entry.aliases, normalized, info.model_id),
                )
            elif normalized != ast_model_id:
                entry = replace(
                    entry,
                    model_id=normalized,
                    aliases=_coerce_aliases(*entry.aliases, ast_model_id),
                )
            return entry
    return None


def find_runtime_entry(model_id: str) -> CatalogEntry:
    """Resolve the concrete pipeline entry, ignoring Studio family folding."""

    requested = _cogvideox_variant_id(model_id) or _catalog_id_key(model_id)
    if not requested:
        raise KeyError(f"Unknown Studio model id: {model_id}")
    ast_model_id = FAST_CANONICAL_AST_ALIASES.get(requested, requested)
    for info in _discover_catalog_infos():
        if _catalog_id_key(info.model_id) == _catalog_id_key(ast_model_id):
            entry = _build_entry(info)
            if entry.model_id != requested:
                entry = replace(
                    entry,
                    model_id=requested,
                    aliases=_coerce_aliases(*entry.aliases, entry.model_id),
                )
            return entry
    raise KeyError(f"Unknown Studio model id: {model_id}")


def find_entry(model_id: str) -> CatalogEntry:
    model_id = (model_id or "").strip().lower()
    exact_entry = _find_exact_entry_without_full_catalog(model_id)
    if exact_entry is not None:
        return exact_entry
    for entry in discover_catalog():
        if entry.model_id.lower() == model_id:
            return entry
        if model_id in {alias.lower() for alias in entry.aliases}:
            return entry
    raise KeyError(f"Unknown Studio model id: {model_id}")


def filter_catalog(
    search: str = "",
    category: str = "All",
    entries: Sequence[CatalogEntry] | None = None,
) -> tuple[CatalogEntry, ...]:
    active_entries = entries or discover_catalog()
    search = (search or "").strip().lower()
    filtered = []
    for entry in active_entries:
        if category not in {"", "All"} and entry.category != category:
            continue
        if search and search not in entry.search_blob():
            continue
        filtered.append(entry)
    return tuple(filtered)
