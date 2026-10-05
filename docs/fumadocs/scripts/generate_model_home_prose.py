#!/usr/bin/env python3
"""Write fact-dense model home MDX pages from recipe JSON.

WorldFoundry is an inference framework. Each generated page uses three prose
sections (What it is for → Runtime contract → Running in WorldFoundry) and
sets ``pageSource: generated`` unless a hand-tuned exemplar exists
(``authored``).

Section 1 is 2–4 short paragraphs: the job in WorldFoundry, method/context
from the catalog ``docs:`` block / paper, recorded tasks, and real benchmark
ids only. Runtime contract covers inputs, task profiles, variants with
reader status labels, and usage notes / blockers from the manifest. Running
leads with ``worldfoundry-eval run``.

Never emit a "What this page does not claim" / "Limits" / "本页不声称" heading.
License, VRAM, and status belong in one short in-body clause under
Running in WorldFoundry.

Bias every section toward the *inference contract*: which variant to launch,
required/optional inputs, artifacts, resolution / frames / steps / guidance,
env / prepare / one ``worldfoundry-eval run``, and gated license.

Training facts from the catalog (pretraining hours, datasets used to train the
weights, optimizer, LR, “how the authors trained it”) are omitted. Keep at
most one short clause when it is required to identify a checkpoint.
Category-level boilerplate is not used. Never emit raw manifest dialect
(``profile_resolves_*``, ``official_demo_parity_pending``,
``in_tree_checkpoint_runtime_static_verified``).

Never emit ``<Video>``, ``<video>``, or demo mp4 figures on model homepages.
Paper teaser / overview images and title ``paper.png`` thumbs stay.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from model_home_exemplars import EXEMPLAR_PROSE
from model_home_status import (
    classify_token,
    dialect_hits,
    reader_label,
    sanitize_manifest_dialect,
    status_label,
    status_sentence as mapped_status_sentence,
)
from model_paper_figures import has_paper_figures

ROOT = Path(__file__).resolve().parents[1]
PAGES_DIR = ROOT / "content/docs/guides/supported-models"
PRESERVE_AUTHORED = {
    "hunyuanvideo",
    "roboflamingo",
    "scope",
    "act",
    "yume",
    "zeroscope",
    "ac3d",
    "wan2.1",
}
HARD_DENY_AUTHORED = PRESERVE_AUTHORED | {
    "wan2.1-vace",
    "hunyuanvideo-1.5",
    "pi05",
    "pi0-fast",
    "openpi",
    "being-h07",
    "matrix-game-2",
}
ENRICH_SKIP_HEADINGS = {
    "## Minimal eval",
    "## Architecture",
    "## Inference",
    "## In WorldFoundry",
    "## 最小评测",
    "## 架构",
    "## 推理",
}
DATA_PATH = ROOT / "lib/model-recipes-data.json"

PAGE_SOURCE_RE = re.compile(r"^pageSource:\s*(authored|generated)\s*$", re.MULTILINE)

FORBIDDEN_CLAIM_HEADINGS = (
    "## What this page does not claim",
    "## 本页不声称什么",
    "## 本页并未声称",
    "## 本页不声称",
    "what-this-page-does-not-claim",
    "## Limits",
    "## 限制与要求",
)

OLD_FIVE_SECTION_HEADINGS = (
    "## Model introduction",
    "## Typical use cases",
    "## Technical characteristics",
    "## Limitations and requirements",
    "## 模型介绍",
    "## 典型使用场景",
    "## 技术特点",
    "## 限制与要求",
)

REQUIRED_SECTION_HEADINGS = {
    "en": (
        "## What it is for",
        "## Runtime contract",
        "## Running in WorldFoundry",
    ),
    "zh": (
        "## 用来做什么",
        "## 运行时契约",
        "## 在 WorldFoundry 中运行",
    ),
}

BOILERPLATE_MARKERS = (
    "Smoke-testing a text- or image-conditioned video pipeline",
    "Robot imitation or VLA policy evaluation on the recorded action contract",
    "A minimal runnable path for the catalog task",
    "Interactive or action-conditioned world simulation when you need a catalog-verified route",
    "Geometry, depth, or scene reconstruction workflows that consume the recorded artifact contract",
    "在迁移到更大模型之前，用它 smoke-test",
    "在记录的 action 契约上做机器人模仿或 VLA",
    "覆盖 catalog 任务",
    "需要 catalog 已验证路由的交互式或 action 条件世界仿真",
    "按记录的 artifact 契约做几何、深度或场景重建",
)


def frontmatter_page_source(model_id: str) -> str:
    return "authored" if model_id in EXEMPLAR_PROSE else "generated"


def emit_prose(value: str, locale: str) -> str:
    return mdx_escape(sanitize_manifest_dialect(value, locale))


def page_source(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return "authored"
    parts = text.split("---", 2)
    if len(parts) < 3:
        return "authored"
    match = PAGE_SOURCE_RE.search(parts[1])
    return match.group(1) if match else "authored"


HEADINGS = {
    "en": {
        "uses": "What it is for",
        "contract": "Runtime contract",
        "run": "Running in WorldFoundry",
    },
    "zh": {
        "uses": "用来做什么",
        "contract": "运行时契约",
        "run": "在 WorldFoundry 中运行",
    },
}

ARCH_DIAGRAM_MARKERS = ("ModelArchDiagram",)
PAPER_FIGURES_MARKERS = ("ModelPaperFigures",)
PAPER_COVER_SUFFIX = "/paper.png"
ARCH_FILE_NAMES = (
    "overview.png",
    "overview.jpg",
    "overview.webp",
    "overall.png",
    "architecture.png",
    "arch.png",
)
ARCH_ASSET_SUFFIXES = tuple(f"/{name}" for name in ARCH_FILE_NAMES)

MODALITY_ZH = {
    "text": "文本",
    "image": "图像",
    "video": "视频",
    "audio": "音频",
    "action": "动作",
    "actions": "动作",
    "camera": "相机",
    "camera pose": "相机位姿",
    "depth": "深度",
    "point cloud": "点云",
    "trajectory": "轨迹",
    "3D scene": "3D 场景",
    "4D scene": "4D 场景",
    "world state": "世界状态",
    "mesh": "网格",
}

TASK_PHRASES_ZH = {
    "text-to-video": "文生视频",
    "image-to-video": "图生视频",
    "video-to-video": "视频到视频",
    "text-to-image": "文生图",
    "image-to-image": "图像编辑",
    "video-editing": "视频编辑",
    "controlled-video": "可控视频",
    "controlled-video-generation": "可控视频生成",
    "edge-conditioned-video": "边缘条件视频",
    "camera-control-video": "相机控制视频",
    "camera-controlled-video": "相机可控视频",
    "camera-control": "相机控制",
    "camera_control": "相机控制",
    "video-diffusion-plugin": "视频扩散插件",
    "action-conditioned-video": "动作条件视频",
    "interactive-world-model": "交互式世界模型",
    "interactive-video-generation": "交互式视频生成",
    "game-world-model": "游戏世界模型",
    "robot-world-model": "机器人世界模型",
    "world-generation": "世界生成",
    "world-model": "世界模型",
    "world_model": "世界模型",
    "world": "世界模型",
    "long-video-generation": "长视频生成",
    "video-generation": "视频生成",
    "vla": "视觉-语言-动作（VLA）",
    "vla.policy_rollout": "VLA 策略执行",
    "vla.action_prediction": "VLA 动作预测",
    "robot_policy": "机器人策略",
    "policy_rollout": "策略执行",
    "cross_embodiment_policy": "跨本体策略",
    "humanoid_policy": "人形策略",
    "visuomotor_policy": "视觉运动策略",
    "action_diffusion": "动作扩散",
    "action_chunking_policy": "动作分块策略",
    "imitation_policy": "模仿策略",
    "embodied_benchmark": "具身评测",
    "action_tokenization": "动作分词",
    "3d-reconstruction": "3D 重建",
    "geometry-prior": "几何先验",
    "novel-view-synthesis": "新视角合成",
    "gaussian-splatting": "高斯泼溅",
    "metric-depth-estimation": "米制深度估计",
    "monocular-depth-estimation": "单目深度估计",
    "hosted-api": "托管 API",
    "forward-dynamics": "前向动力学",
    "inverse-dynamics": "逆向动力学",
    "action-policy": "动作策略",
    "3d-world-generation": "3D 世界生成",
    "text-to-3d-world": "文本生成 3D 世界",
    "image-generation": "图像生成",
    "vla.reasoning": "VLA 推理",
    "latent-world-model": "潜空间世界模型",
    "jepa-world-model": "JEPA 世界模型",
    "real-time-generation": "实时生成",
    "distillation": "蒸馏",
    "action_chunking": "动作分块",
    "real_time_control": "实时控制",
    "embodied-world-model": "具身世界模型",
    "robotics-world-model": "机器人世界模型",
    "behavior_generation": "行为生成",
    "imitation_learning": "模仿学习",
    "depth": "深度估计",
    "geometry": "几何估计",
    "feed-forward multi-view 3d reconstruction": "前馈多视图 3D 重建",
    "geometry estimation": "几何估计",
    "text-to-audio": "文生音频",
    "video-to-audio": "视频生音频",
    "multimodal-reasoning": "多模态推理",
    "audio-reasoning": "音频推理",
    "video-tokenization": "视频分词",
    "reference-to-video": "参考图生成视频",
    "audio-guided-video": "音频引导视频",
    "spatial-reasoning": "空间推理",
    "image-question-answering": "图像问答",
    "video-question-answering": "视频问答",
    "audio-video-generation": "音视频联合生成",
    "video-extension": "视频延展",
    "trajectory": "轨迹",
    "point-cloud": "点云",
    "long-term-consistency": "长期一致性",
    "minecraft-world-model": "Minecraft 世界模型",
    "point-cloud-world-model": "点云世界模型",
    "3d-world-model": "3D 世界模型",
    "embodied_policy": "具身策略",
}

GENERIC_FIELD_DETAILS = {
    "",
    "required",
    "optional",
    "recorded input field from the runtime profile.",
}

UNIFIED_BOILERPLATE_MARKERS = (
    "Default unified WorldFoundry runtime",
    "Override tier with `--cuda",
    "Source the environment file emitted",
    "Runnable GPU models that previously used",
    "Per-run visual QA",
    "Per-run validation evidence",
    "tracked outside the model catalog",
    "tracked outside the catalog",
    "catalog 外跟踪",
    "验证证据在 catalog 外",
    "Install with `bash scripts/setup/unified_install.sh`",
)

STATUS_RESTATE_MARKERS = (
    "Runner parity is pending in this workspace",
    "runner parity 待验证",
    "本工作区中 runner parity",
    "本工作区中 Runner 一致性",
)

_BENCHMARK_HUB_IDS: set[str] | None = None
_HUB_SKIP_PAGES = frozenset({"index", "runtime-environments"})

SUITE_SCORECARD_MARKERS = (
    "listed suites, not an ingested",
    "listed suite, not an ingested",
    "not an ingested WorldFoundry scorecard",
    "不是 WorldFoundry 记分卡",
    "不是已摄入的 WorldFoundry",
    "listed suites not scorecard",
)


def yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def mdx_escape(value: str) -> str:
    """Escape MDX/JSX metacharacters, but keep ${ENV} readable inside backticks."""

    def escape_plain(text: str) -> str:
        return text.replace("{", "\\{").replace("}", "\\}").replace("<", "\\<")

    parts = re.split(r"(`[^`]*`)", value)
    escaped: list[str] = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) >= 2:
            inner = re.sub(r"\$\{([A-Z0-9_]+)\}", r"$\1", part[1:-1])
            inner = inner.replace("{", "\\{").replace("}", "\\}")
            escaped.append(f"`{inner}`")
        else:
            escaped.append(escape_plain(part))
    return "".join(escaped)


def paragraphs(values: list[Any] | None) -> list[str]:
    result: list[str] = []
    for item in values or []:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def locale_paragraphs(docs: dict[str, Any], key: str, locale: str) -> list[str]:
    zh_key = f"{key}Zh"
    if locale == "zh":
        return paragraphs(docs.get(zh_key)) or paragraphs(docs.get(key))
    return paragraphs(docs.get(key))


def join_prose(parts: list[str], locale: str = "en") -> str:
    cleaned = [part.strip() for part in parts if part and part.strip()]
    if not cleaned:
        return ""
    if locale == "zh":
        out = [cleaned[0]]
        for part in cleaned[1:]:
            sep = "" if out[-1].endswith(("。", "！", "？", "；")) else " "
            out.append(sep + part)
        return "".join(out)
    return " ".join(cleaned)


def join_en(items: Any) -> str:
    cleaned = [str(item) for item in items if item]
    if len(cleaned) <= 1:
        return cleaned[0] if cleaned else ""
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def join_zh(items: Any) -> str:
    cleaned = [str(item) for item in items if item]
    return "、".join(cleaned)


def unique_keep(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(key)
    return output


def mentions(haystack: str, *needles: str) -> bool:
    blob = haystack.lower()
    return any(needle and needle.lower() in blob for needle in needles)


# Catalog facts that describe how weights were trained — drop from generated
# homes. Keep function names / config ids that only look like "train".
TRAINING_PROSE_RE = re.compile(
    r"(?i)("
    r"pretraining\b.{0,60}(hour|dataset|data|embodiment|mix)|"
    r"\b\d[\d,\.]{2,}\s*(hours|hour)\b|"
    r"trained (?:on|at|with|using) .{12,}|"
    r"training (?:data|recipe|loop|set|mix|hours|stage)|"
    r"\boptimizer\b|\blearning rate\b|\bfinetun(?:e|ing)\b|"
    r"data-in-the-loop|reward backpropagation|UniHand|"
    r"预训练.{0,24}(小时|数据|数据集|本体)|"
    r"训练数据|训练配方|训练环|优化器|学习率|后训练"
    r")"
)
INFERENCE_KEEP_RE = re.compile(
    r"(?i)("
    r"inference-only|excludes training|from_pretrained|"
    r"create_trained_policy|libero_posttrain|weights-only|"
    r"仅推理|不含训练|不是训练"
    r")"
)
TRAINING_LEAD_RE = re.compile(r"(?i)^(pretraining|training|pretrained|训练|预训练)\b")

INFER_PARAM_FIELDS = (
    "num_frames",
    "frames",
    "height",
    "width",
    "size",
    "fps",
    "num_inference_steps",
    "num_iterations",
    "guidance_scale",
    "negative_prompt",
    "seed",
    "base_seed",
)


def is_training_heavy(text: str) -> bool:
    if INFERENCE_KEEP_RE.search(text):
        return False
    if not TRAINING_PROSE_RE.search(text):
        return False
    if TRAINING_LEAD_RE.search(text.strip()):
        return True
    return len(text) > 160


def drop_training_prose(items: list[str]) -> list[str]:
    return [item for item in items if item and not is_training_heavy(item)]


def task_phrase_en(task: str) -> str:
    return task.replace("_", " ").replace(".", " ").strip()


def task_phrase_zh(task: str) -> str:
    return TASK_PHRASES_ZH.get(task) or TASK_PHRASES_ZH.get(task.lower()) or task


def task_phrase(task: str, locale: str) -> str:
    return task_phrase_zh(task) if locale == "zh" else task_phrase_en(task)


def translate_modality(token: str, locale: str) -> str:
    if locale != "zh":
        return token
    return MODALITY_ZH.get(token.lower(), token)


def clip_sentence(value: str, limit: int = 200) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:.")
    return cut + "…"


def clip_at_sentence(value: str, limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    if len(text) <= limit:
        return text
    window = text[:limit]
    period = max(window.rfind(". "), window.rfind("。"))
    if period >= 80:
        return window[: period + 1].strip()
    return clip_sentence(text, limit)


def is_real_publisher(name: str) -> bool:
    return bool(name) and not mentions(name, "not recorded", "未记录", "institution not recorded")


def benchmark_hub_ids() -> set[str]:
    global _BENCHMARK_HUB_IDS
    if _BENCHMARK_HUB_IDS is None:
        hub = ROOT / "content/docs/evaluation/benchmark-hub"
        ids: set[str] = set()
        if hub.is_dir():
            for path in hub.glob("*.mdx"):
                name = path.name
                if name.endswith(".zh.mdx"):
                    continue
                stem = name[:-4] if name.endswith(".mdx") else name
                if stem and stem not in _HUB_SKIP_PAGES:
                    ids.add(stem)
        _BENCHMARK_HUB_IDS = ids
    return _BENCHMARK_HUB_IDS


def recorded_task_ids(recipe: dict[str, Any]) -> list[str]:
    tasks = [str(item) for item in recipe.get("tasks") or [] if item]
    if not tasks:
        for item in recipe.get("inferenceTasks") or []:
            if isinstance(item, dict) and item.get("id"):
                tasks.append(str(item["id"]))
    return unique_keep(tasks)


def arxiv_from_sources(recipe: dict[str, Any]) -> tuple[str, str]:
    for item in recipe.get("sources") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("kind") or "") != "paper":
            continue
        url = str(item.get("url") or "").strip()
        match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", url)
        if match:
            arxiv_id = match.group(1)
            href = url if url.startswith("http") else f"https://arxiv.org/abs/{arxiv_id}"
            if "/pdf/" in href:
                href = f"https://arxiv.org/abs/{arxiv_id}"
            return arxiv_id, href
    return "", ""


def is_boilerplate_note(text: str) -> bool:
    return any(marker.lower() in text.lower() for marker in UNIFIED_BOILERPLATE_MARKERS)


def format_field_token(item: dict[str, Any], locale: str) -> str:
    field = str(item.get("field") or "").strip()
    if not field:
        return ""
    detail = sanitize_manifest_dialect(str(item.get("detail") or "").strip(), locale)
    if detail.lower() in GENERIC_FIELD_DETAILS:
        return f"`{field}`"
    if re.fullmatch(r"[a-z0-9_]+", detail) and "_" in detail:
        return f"`{field}`"
    return f"`{field}` ({detail})"


def _contract_rows(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = [item for item in (recipe.get("inferenceTasks") or []) if isinstance(item, dict)]
    if tasks:
        run_id, task_id = run_identity(recipe)
        task = next(
            (item for item in tasks if item.get("id") == task_id
             and (not item.get("variantIds") or run_id in item["variantIds"])),
            tasks[0],
        )
        return [
            {
                **item,
                "field": str(item.get("field") or "").replace("-", "_"),
                "required": bool(item.get("required")) and item.get("default") in (None, ""),
                "detail": "Required" if item.get("required") and item.get("default") in (None, "") else "Optional",
            }
            for item in task.get("inputs") or [] if isinstance(item, dict)
        ]
    return [item for item in recipe.get("inputContract") or [] if isinstance(item, dict)]


def exemplar_block(model_id: str, key: str, locale: str) -> list[str]:
    block = EXEMPLAR_PROSE.get(model_id) or {}
    value = block.get(key)
    if isinstance(value, dict):
        items = value.get(locale) or value.get("en")
        if isinstance(items, str):
            return [items] if items.strip() else []
        if isinstance(items, list):
            return [str(item).strip() for item in items if str(item).strip()]
    return []


def pipeline_class(recipe: dict[str, Any]) -> str:
    target = str((recipe.get("runtime") or {}).get("pipelineTarget") or "")
    return target.rsplit(":", 1)[-1] if target else ""


def run_identity(recipe: dict[str, Any]) -> tuple[str, str]:
    run = str((recipe.get("commands") or {}).get("run") or "")
    joined = " ".join(line.strip().rstrip("\\").strip() for line in run.splitlines())
    match_id = re.search(r"worldfoundry-eval run\s+(\S+)", joined)
    match_task = re.search(
        r"--pipeline\.task-profile\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))",
        joined,
    )
    run_id = match_id.group(1) if match_id else ""
    task = ""
    if match_task:
        task = match_task.group(1) or match_task.group(2) or match_task.group(3) or ""
    if not run_id:
        variants = [item for item in (recipe.get("variants") or []) if isinstance(item, dict)]
        run_id = str(variants[0].get("id") or recipe.get("id") or "")
    if not task:
        tasks = [str(item) for item in recipe.get("tasks") or [] if item]
        task = tasks[0] if tasks else ""
    return run_id, task


def _contract_fields(recipe: dict[str, Any]) -> tuple[list[str], list[str]]:
    required: list[str] = []
    optional: list[str] = []
    rows = _contract_rows(recipe)
    for item in rows:
        field = str(item.get("field") or "").strip()
        if not field:
            continue
        detail = str(item.get("detail") or "").strip().lower()
        if item.get("required") is True or detail == "required":
            required.append(field)
        else:
            optional.append(field)
    return unique_keep(required), unique_keep(optional)


def required_fields(recipe: dict[str, Any]) -> list[str]:
    required, _ = _contract_fields(recipe)
    return required


def preferred_variant(recipe: dict[str, Any]) -> str:
    variants = [item for item in (recipe.get("variants") or []) if isinstance(item, dict) and item.get("id")]
    run_id, _ = run_identity(recipe)
    if not variants:
        return run_id or str(recipe.get("id") or "")

    def score(item: dict[str, Any]) -> int:
        blob = " ".join(
            str(item.get(key) or "") for key in ("status", "runtimeStatus", "runner")
        ).lower()
        if any(
            token in blob
            for token in (
                "parity_recorded",
                "demo_and_runner",
                "gpu_validated",
                "runner_verified",
            )
        ) or blob == "verified":
            return 3
        if "verified" in blob and "pending" not in blob:
            return 2
        if "profile_resolves" in blob or "pending" in blob:
            return 0
        return 1

    ranked = sorted(variants, key=score, reverse=True)
    if run_id and any(str(item.get("id")) == run_id for item in ranked):
        match = next(item for item in ranked if str(item.get("id")) == run_id)
        if score(match) >= score(ranked[0]):
            return run_id
    return str(ranked[0]["id"])


def action_detail(recipe: dict[str, Any]) -> str:
    for item in _contract_rows(recipe):
        if isinstance(item, dict) and str(item.get("field") or "") == "actions":
            return str(item.get("detail") or "").strip()
    return ""


def artifact_names(recipe: dict[str, Any]) -> list[str]:
    names: list[str] = []
    run_id, task_id = run_identity(recipe)
    task = next(
        (item for item in recipe.get("inferenceTasks") or []
         if item.get("id") == task_id and (not item.get("variantIds") or run_id in item["variantIds"])),
        None,
    )
    artifacts = task.get("artifacts", []) if task is not None else recipe.get("artifacts") or []
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename") or "").strip()
        kind = str(item.get("kind") or "").strip()
        if filename:
            names.append(filename)
        elif kind:
            names.append(kind)
    return unique_keep(names)


def variant_ids(recipe: dict[str, Any]) -> list[str]:
    return [
        str(item.get("id"))
        for item in (recipe.get("variants") or [])
        if isinstance(item, dict) and item.get("id")
    ]


_HF_WEIGHTS_RE = re.compile(
    r"https?://huggingface\.co/(?!spaces/)(?!datasets/)(?!papers/)([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
)


def checkpoint_ids(recipe: dict[str, Any]) -> list[str]:
    ids = [
        str(item.get("id"))
        for item in (recipe.get("checkpoints") or [])
        if isinstance(item, dict) and item.get("id")
    ]
    if ids:
        return unique_keep(ids)
    extras: list[str] = []
    for source in recipe.get("sources") or []:
        if not isinstance(source, dict) or source.get("kind") != "weights":
            continue
        match = _HF_WEIGHTS_RE.search(str(source.get("url") or ""))
        if match:
            extras.append(match.group(1))
    return unique_keep(extras)


def checkpoint_licenses(recipe: dict[str, Any]) -> list[str]:
    return unique_keep(
        [
            str(item.get("license"))
            for item in (recipe.get("checkpoints") or [])
            if isinstance(item, dict) and item.get("license")
        ]
    )


def gated_checkpoints(recipe: dict[str, Any]) -> list[str]:
    return [
        str(item.get("id"))
        for item in (recipe.get("checkpoints") or [])
        if isinstance(item, dict) and item.get("gated") and item.get("id")
    ]


def paper_record(docs: dict[str, Any]) -> dict[str, Any]:
    paper = docs.get("paper")
    return paper if isinstance(paper, dict) else {}


def publisher_name(docs: dict[str, Any], locale: str) -> str:
    publisher = docs.get("publisher") or {}
    if not isinstance(publisher, dict):
        return ""
    if locale == "zh":
        return str(publisher.get("nameZh") or publisher.get("name") or "").strip()
    return str(publisher.get("name") or "").strip()


def status_group(recipe: dict[str, Any]) -> str:
    return str((recipe.get("status") or {}).get("group") or "integrated")


def status_sentence(recipe: dict[str, Any], locale: str) -> str:
    return mapped_status_sentence(recipe, locale)


def modality_sentence(docs: dict[str, Any], locale: str) -> str:
    modalities = docs.get("modalities") or {}
    inputs = [translate_modality(str(item), locale) for item in modalities.get("inputs") or [] if item]
    outputs = [translate_modality(str(item), locale) for item in modalities.get("outputs") or [] if item]
    if not inputs and not outputs:
        return ""
    if locale == "zh":
        if inputs and outputs:
            return f"模态上，它接受{join_zh(inputs)}，输出{join_zh(outputs)}。"
        if inputs:
            return f"输入模态包括{join_zh(inputs)}。"
        return f"输出模态为{join_zh(outputs)}。"
    if inputs and outputs:
        return f"Documented modalities: {join_en(inputs)} in, {join_en(outputs)} out."
    if inputs:
        return f"Documented inputs include {join_en(inputs)}."
    return f"Documented outputs include {join_en(outputs)}."


def paper_sentence(docs: dict[str, Any], locale: str, recipe: dict[str, Any] | None = None) -> str:
    paper = paper_record(docs)
    title = str(paper.get("title") or "").strip()
    arxiv_id = str(paper.get("arxivId") or paper.get("arxiv_id") or "").strip()
    href = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
    if not arxiv_id and recipe:
        arxiv_id, href = arxiv_from_sources(recipe)
    if not title and not arxiv_id:
        return ""
    if title:
        cite = f"[{mdx_escape(title)}]({href})" if href else mdx_escape(title)
    else:
        cite = f"[arXiv {arxiv_id}]({href})" if href else f"arXiv {arxiv_id}"
    extra = ", ".join(str(part) for part in (paper.get("venue"), paper.get("year")) if part)
    if locale == "zh":
        return f"记录论文为{cite}" + (f"（{mdx_escape(extra)}）。" if extra else "。")
    return f"The recorded paper is {cite}" + (f" ({mdx_escape(extra)})." if extra else ".")


def _source_link_text(url: str, label: str) -> str:
    text = (label or "").strip()
    generic = {
        "project",
        "github",
        "paper",
        "docs",
        "source",
        "code",
        "huggingface",
        "weights",
        "项目",
        "代码",
        "upstream readme",
    }
    if text and text.lower() not in generic:
        return text
    cleaned = re.sub(r"^https?://", "", url).rstrip("/")
    cleaned = re.sub(r"^www\.", "", cleaned)
    for prefix in ("github.com/", "huggingface.co/", "arxiv.org/abs/", "arxiv.org/pdf/"):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    if "/tree/" in cleaned:
        cleaned = cleaned.split("/tree/")[0]
    if "#" in cleaned:
        cleaned = cleaned.split("#", 1)[0]
    return cleaned or url


def sources_sentence(recipe: dict[str, Any], locale: str) -> str:
    """Project / code links from catalog ``sources:``. Paper URLs stay in paper_sentence."""

    items = [item for item in (recipe.get("sources") or []) if isinstance(item, dict)]
    github_urls = [
        str(item.get("url") or "").strip()
        for item in items
        if str(item.get("kind") or "").strip().lower() in {"source", "code"}
        and str(item.get("url") or "").strip()
    ]
    bits: list[str] = []
    seen: set[str] = set()
    for item in items:
        url = str(item.get("url") or "").strip()
        kind = str(item.get("kind") or "").strip().lower()
        label = str(item.get("label") or "").strip()
        if not url or url in seen or kind == "paper":
            continue
        if kind == "docs" and (
            "#readme" in url.lower() or any(host and (host in url or url in host) for host in github_urls)
        ):
            continue
        if kind == "project":
            name = "Project page" if locale == "en" else "项目页"
        elif kind in {"source", "code"}:
            name = "Code" if locale == "en" else "代码"
        else:
            continue
        link = _source_link_text(url, label)
        if locale == "zh":
            bits.append(f"{name}：[{link}]({url})。")
        else:
            bits.append(f"{name}: [{link}]({url}).")
        seen.add(url)
    if not bits:
        return ""
    return " ".join(bits) if locale == "en" else "".join(bits)


def alias_sentence(recipe: dict[str, Any], locale: str) -> str:
    aliases = [str(item) for item in recipe.get("aliases") or [] if item]
    if not aliases:
        return ""
    shown = aliases[:4]
    extra = " 等" if locale == "zh" and len(aliases) > 4 else (" among others" if len(aliases) > 4 else "")
    joined = join_zh([f"`{item}`" for item in shown]) if locale == "zh" else join_en([f"`{item}`" for item in shown])
    if locale == "zh":
        return f"别名包括 {joined}{extra}。"
    return f"Catalog aliases include {joined}{extra}."


def environment_sentence(recipe: dict[str, Any], locale: str) -> str:
    runtime = recipe.get("runtime") or {}
    env = str(runtime.get("environmentName") or "").strip()
    kind = str(runtime.get("environmentKind") or "")
    if not env or kind in {"unrecorded", "none"}:
        return "The environment profile is not recorded." if locale == "en" else "运行环境未记录。"
    kind_en = "dedicated" if kind == "dedicated" else "unified"
    kind_zh = "独立" if kind == "dedicated" else "统一"
    details: list[str] = []
    if runtime.get("python"):
        details.append(f"Python {runtime['python']}")
    if runtime.get("cudaLabel"):
        details.append(str(runtime["cudaLabel"]))
    torch_pin = (runtime.get("packageVersions") or {}).get("torch")
    if torch_pin and torch_pin not in {"torch", ""}:
        details.append(str(torch_pin))
    detail = f" ({', '.join(details)})" if details else ""
    detail_zh = f"（{'，'.join(details)}）" if details else ""
    if locale == "zh":
        return f"运行环境是{kind_zh} profile `{env}`{detail_zh}。"
    return f"It runs in the {kind_en} environment `{env}`{detail}."


def hardware_paragraphs(docs: dict[str, Any], locale: str) -> list[str]:
    hardware = docs.get("hardware") or {}
    min_vram = hardware.get("minVramGb")
    recommended = hardware.get("recommended")
    notes = [str(item).strip() for item in hardware.get("notes") or [] if str(item).strip()]
    parts: list[str] = []
    if min_vram is not None:
        parts.append(
            f"Minimum VRAM is recorded at {min_vram} GB."
            if locale == "en"
            else f"记录的最低显存为 {min_vram} GB。"
        )
    if recommended:
        parts.append(
            f"Recommended hardware: {recommended}."
            if locale == "en"
            else f"推荐配置：{recommended}。"
        )
    parts.extend(notes)
    text = join_prose(parts)
    return [text] if text else []


def benchmarks_paragraph(docs: dict[str, Any], locale: str) -> str:
    known = benchmark_hub_ids()
    benchmarks = [item for item in (docs.get("benchmarks") or []) if isinstance(item, dict)]
    if not benchmarks:
        return ""
    hub = "/zh/docs/evaluation/benchmark-hub" if locale == "zh" else "/docs/evaluation/benchmark-hub"
    links = []
    for bench in benchmarks[:4]:
        bench_id = str(bench.get("id") or "").strip()
        if not bench_id or (known and bench_id not in known):
            continue
        name = str(bench.get("name") or bench_id)
        reason = bench.get("reasonZh") if locale == "zh" else bench.get("reason")
        reason = str(reason or bench.get("reason") or "").strip().rstrip("。.")
        link = f"[{mdx_escape(name)}]({hub}/{bench_id})"
        if reason:
            links.append(f"{link}（{mdx_escape(reason)}）" if locale == "zh" else f"{link} ({mdx_escape(reason)})")
        else:
            links.append(link)
    if not links:
        return ""
    if locale == "zh":
        return f"记录的评测入口包括{join_zh(links)}。"
    return f"Recorded evaluation entry points include {', '.join(links)}."


def variants_paragraph(recipe: dict[str, Any], locale: str) -> str:
    variants = [
        item
        for item in (recipe.get("variants") or [])
        if isinstance(item, dict) and item.get("id")
    ]
    if len(variants) <= 1:
        return ""
    preferred = preferred_variant(recipe)
    profile_uses = Counter(str(item.get("runtimeProfile") or "") for item in variants)
    shown = variants[:5]
    bits: list[str] = []
    for item in shown:
        vid = str(item.get("id") or "")
        task = str(item.get("task") or "").strip()
        runtime_status = str(item.get("runtimeStatus") or "")
        shared_profile = profile_uses[str(item.get("runtimeProfile") or "")] > 1
        all_checkpoints_verified = runtime_status.startswith(
            ("all_released_checkpoints_gpu_validated", "all_declared_variants_checkpoint_gpu_validated")
        )
        raw = runtime_status or str(item.get("status") or "")
        if shared_profile and not all_checkpoints_verified:
            raw = str(item.get("status") or "")
        label = reader_label(classify_token(raw), locale) if raw and raw != "not_recorded" else ""
        if locale == "zh":
            piece = f"`{vid}`"
            if task:
                piece += f"（`{task}`）"
            if label:
                piece += f"：{label}"
        else:
            piece = f"`{vid}`"
            if task:
                piece += f" (`{task}`)"
            if label:
                piece += f" — {label}"
        bits.append(piece)
    extra = " 等" if locale == "zh" and len(variants) > 5 else (" among others" if len(variants) > 5 else "")
    joined = join_zh(bits) if locale == "zh" else "; ".join(bits)
    if locale == "zh":
        start = f"起步 variant 是 `{preferred}`。" if preferred else ""
        return (
            f"{start}catalog 登记了 {len(variants)} 个 variant（上方卡片）：{joined}{extra}。"
        )
    start = f"Start on `{preferred}`. " if preferred else ""
    return (
        f"{start}The family registers {len(variants)} catalog variants (cards above): {joined}{extra}."
    )


def task_profile_sentence(recipe: dict[str, Any], locale: str) -> str:
    ids = unique_keep(
        str(item["id"]) for item in recipe.get("inferenceTasks") or []
        if isinstance(item, dict) and item.get("id")
    )
    if not ids:
        return "No task profiles are recorded on this card." if locale == "en" else "本卡片未记录任务 profile。"
    shown = ids[:6]
    extra = " 等" if locale == "zh" and len(ids) > 6 else (" among others" if len(ids) > 6 else "")
    joined = join_zh(f"`{item}`" for item in shown) if locale == "zh" else join_en(f"`{item}`" for item in shown)
    pipe = pipeline_class(recipe)
    if locale == "zh":
        text = f"CLI task profile：{joined}{extra}。"
        if pipe:
            text += f"绑定的 pipeline 类是 `{pipe}`。"
        return text
    text = f"CLI task profiles: {joined}{extra}."
    if pipe:
        text += f" WorldFoundry binds `{pipe}`."
    return text


def family_status_sentence(recipe: dict[str, Any], locale: str) -> str:
    label = status_label(recipe, locale)
    if locale == "zh":
        return f"系列状态：**{label}**。"
    return f"Family status: **{label}**."


def usage_and_blocker_paragraphs(recipe: dict[str, Any], docs: dict[str, Any], locale: str) -> list[str]:
    runtime = recipe.get("runtime") or {}
    extra_notes: list[str] = []
    for item in recipe.get("notes") or []:
        extra_notes.append(str(item))
    for item in runtime.get("notes") or []:
        extra_notes.append(str(item))
    items = drop_training_prose(locale_paragraphs(docs, "usageNotes", locale))
    items.extend(drop_training_prose(extra_notes))
    items.extend(drop_training_prose(locale_paragraphs(docs, "limitations", locale)))
    cleaned: list[str] = []
    for item in items:
        text = sanitize_manifest_dialect(re.sub(r"\s+", " ", str(item)).strip(), locale)
        if not text:
            continue
        pieces = re.split(r"(?<=[。！？])\s*|(?<=[.!?])\s+", text)
        kept = []
        for part in pieces:
            part = part.strip()
            if not part or is_boilerplate_note(part):
                continue
            if mentions(part, "publishing institution", "发表机构"):
                continue
            if mentions(
                part,
                "native demo evidence",
                "原生 Demo 证据",
                "原生 demo 证据",
                "run commands are intentionally omitted",
                "有意省略了运行命令",
                "Runner evidence:",
                "Runner 证据",
                "请勿把本页当作",
                "Runner parity 记录为 pending",
                "Runner parity 为 pending",
                "Runner parity is pending",
                "不要把本页当作",
                "仅有 catalog 条目并不代表",
                "a catalog entry alone is not proof",
                "No runnable WorldFoundry route",
                "尚无可运行的 WorldFoundry",
                "No verified end-to-end WorldFoundry",
                "没有经过验证的 WorldFoundry 端到端",
                "checkpoint-backed GPU artifact has not been verified",
                "checkpoint-backed GPU 产物尚未在本工作区验证",
            ):
                continue
            if any(marker.lower() in part.lower() for marker in STATUS_RESTATE_MARKERS):
                continue
            kept.append(part)
        text = " ".join(kept).strip()
        if locale == "zh" and text and not re.search(r"[\u4e00-\u9fff]", text):
            continue
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned[:3]


def method_paragraph(recipe: dict[str, Any], docs: dict[str, Any], locale: str, lead: str) -> str:
    arch = drop_training_prose(locale_paragraphs(docs, "architecture", locale))
    text = clip_at_sentence(arch[0], 420) if arch else ""
    if text and (text == lead or mentions(lead, text[:36])):
        text = ""
    publisher = publisher_name(docs, locale)
    extra = ""
    if is_real_publisher(publisher) and not mentions(lead, publisher, str(recipe.get("provider") or "")):
        extra = f"该模型由{publisher}发布。" if locale == "zh" else f"It is published by {publisher}."
    parts = [part for part in (text, extra) if part]
    return join_prose(parts, locale)


def prepare_sentence(recipe: dict[str, Any], locale: str) -> str:
    prepare = str((recipe.get("commands") or {}).get("prepare") or "").strip()
    if not prepare:
        return ""
    cmd = " ".join(line.strip().rstrip("\\").strip() for line in prepare.splitlines())
    if locale == "zh":
        return f"准备环境并检查本地资产：`{cmd}`；下载缺失的公开权重时加 `--download`。"
    return f"Prepare the environment and check local assets with `{cmd}`. Add `--download` to fetch missing public checkpoints."


def builder_sentence(locale: str) -> str:
    if locale == "zh":
        return "上方命令构建器按已记录任务切换 `--pipeline.task-profile`，发出的仍是 `worldfoundry-eval run`。"
    return (
        "The command builder above switches `--pipeline.task-profile` among recorded tasks "
        "and still emits `worldfoundry-eval run`."
    )


def inference_defaults_paragraph(recipe: dict[str, Any], locale: str) -> str:
    """Pull recorded height/width/frames/steps/guidance/fps from inference tasks."""

    bits: list[str] = []
    for task in (recipe.get("inferenceTasks") or [])[:2]:
        if not isinstance(task, dict):
            continue
        defaults: list[str] = []
        for item in task.get("inputs") or []:
            if not isinstance(item, dict):
                continue
            field = str(item.get("field") or "").strip()
            if field not in INFER_PARAM_FIELDS:
                continue
            value = item.get("default")
            if value in (None, "", "Optional"):
                detail = str(item.get("detail") or "")
                match = re.search(r"default\s*=\s*([^;]+)", detail)
                value = match.group(1).strip() if match else None
            if value in (None, ""):
                continue
            defaults.append(f"{field}={value}")
        if not defaults:
            continue
        task_id = str(task.get("id") or task.get("task") or "").strip()
        joined = ", ".join(defaults[:6])
        if locale == "zh":
            bits.append(f"`{task_id}` 记录默认值：{joined}" if task_id else f"记录默认值：{joined}")
        else:
            bits.append(
                f"Recorded `{task_id}` defaults: {joined}" if task_id else f"Recorded defaults: {joined}"
            )
    if not bits:
        return ""
    return "；".join(bits) + "。" if locale == "zh" else " ".join(bits) + ("." if not bits[-1].endswith(".") else "")


def contract_paragraph(recipe: dict[str, Any], locale: str) -> str:
    """Inputs, knobs, artifacts once. No lead, no suite disclaimer."""

    rows = _contract_rows(recipe)
    required_tokens: list[str] = []
    optional_tokens: list[str] = []
    for item in rows:
        field = str(item.get("field") or "").strip()
        if not field:
            continue
        token = format_field_token(item, locale)
        detail = str(item.get("detail") or "").strip().lower()
        if item.get("required") is True or detail == "required":
            required_tokens.append(token)
        else:
            optional_tokens.append(token)
    required_tokens = unique_keep(required_tokens)
    optional_tokens = unique_keep(optional_tokens)
    field_names = {str(item.get("field") or "") for item in rows}
    artifacts = artifact_names(recipe)
    action = action_detail(recipe)
    bits: list[str] = []
    if locale == "zh":
        if required_tokens:
            bits.append(f"必填 {join_zh(required_tokens)}")
        if optional_tokens:
            bits.append(f"可选 {join_zh(optional_tokens[:5])}")
        if action and "actions" not in field_names:
            bits.append(f"`actions` 为 {action}")
        if artifacts:
            bits.append(f"产物 {join_zh(f'`{item}`' for item in artifacts[:3])}")
        if not bits:
            return "本卡片未记录输入、旋钮或产物。"
        return "；".join(bits) + "。"
    if required_tokens:
        bits.append(f"required {join_en(required_tokens)}")
    if optional_tokens:
        bits.append(f"optional {join_en(optional_tokens[:5])}")
    if action and "actions" not in field_names:
        bits.append(f"`actions` is {action}")
    if artifacts:
        bits.append(f"artifacts {join_en(f'`{item}`' for item in artifacts[:3])}")
    if not bits:
        return "Inputs, knobs, and artifacts are not recorded on this card."
    text = "; ".join(bits)
    return text[0].upper() + text[1:] + "."


def license_sentence(recipe: dict[str, Any], locale: str) -> str:
    licenses = checkpoint_licenses(recipe)
    if not licenses:
        return ""
    gated = gated_checkpoints(recipe)
    if locale == "zh":
        text = f"记录的权重 license 包括 {join_zh(licenses)}。"
        if gated:
            text += f" gated 仓库：{join_zh(f'`{item}`' for item in gated[:3])}，下载前需在上游接受条款。"
        return text
    text = f"Recorded checkpoint licenses include {join_en(licenses)}."
    if gated:
        text += f" Gated repositories: {join_en(f'`{item}`' for item in gated[:3])}; accept upstream terms before download."
    return text


def checkpoint_sentence(recipe: dict[str, Any], locale: str) -> str:
    ids = checkpoint_ids(recipe)
    if not ids:
        notes = " ".join(str(item) for item in (recipe.get("notes") or [])[:4])
        usage = " ".join(locale_paragraphs(recipe.get("docs") or {}, "usageNotes", "en"))
        blob = notes + " " + usage
        if mentions(blob, "gs://", "GCS", "gsutil"):
            return (
                "No Hugging Face checkpoint is recorded; stage GCS assets as the usage notes describe."
                if locale == "en"
                else "本条目未记录 Hugging Face checkpoint；按使用说明用 GCS 就位权重。"
            )
        if mentions(blob, "stage", "checkpoint", "missing"):
            return (
                "No public checkpoint id is recorded on this card; follow the usage notes for staging."
                if locale == "en"
                else "本卡片未记录公开 checkpoint id；按使用说明就位权重。"
            )
        return ""
    shown = ids[:3]
    extra = " among others" if len(ids) > 3 else ""
    extra_zh = " 等" if len(ids) > 3 else ""
    if locale == "zh":
        return f"记录的权重仓库包括 {join_zh(f'`{item}`' for item in shown)}{extra_zh}。"
    return f"Recorded weight repositories include {join_en(f'`{item}`' for item in shown)}{extra}."


def run_command_sentence(recipe: dict[str, Any], locale: str) -> str:
    cmd = launch_command(recipe)
    if not cmd:
        return ""
    model_id = str(recipe.get("id") or "")
    variant = run_identity(recipe)[0] or preferred_variant(recipe) or model_id
    table = _RUN_CMD_ZH if locale == "zh" else _RUN_CMD_EN
    selected = next(
        (item for item in recipe.get("variants") or [] if item.get("id") == variant),
        None,
    )
    opening_recipe = {**recipe, "tasks": [selected["task"]]} if selected and selected.get("task") else recipe
    family = job_family(opening_recipe)
    voices = table.get(family) or table["generic"]
    return _fill_template(voices[voice_index(model_id, len(voices))], variant, cmd)


def for_what_paragraphs(recipe: dict[str, Any], locale: str) -> list[str]:
    """2–4 short paragraphs: job in WorldFoundry, method/context, tasks, benchmarks."""

    model_id = str(recipe.get("id") or "")
    docs = recipe.get("docs") or {}
    exemplar = drop_training_prose(exemplar_block(model_id, "use_cases", locale))
    if exemplar:
        return exemplar[:4]

    lead = ""
    overview = drop_training_prose(locale_paragraphs(docs, "overview", locale))
    if overview:
        lead = overview[0]

    parts: list[str] = []
    opening = family_opening(recipe, locale)
    if opening:
        parts.append(opening)

    method = method_paragraph(recipe, docs, locale, lead)
    if method:
        parts.append(method)

    tasks = recorded_task_ids(recipe)
    aliases = [str(item) for item in recipe.get("aliases") or [] if item]
    modality = modality_sentence(docs, locale)
    extra_bits: list[str] = []
    if tasks:
        phrases = unique_keep([task_phrase(item, locale) for item in tasks[:4]])
        joined = join_zh(phrases) if locale == "zh" else join_en(phrases)
        if locale == "zh":
            extra_bits.append(f"catalog 记录的任务包括 {joined}。")
        else:
            extra_bits.append(f"Catalog tasks include {joined}.")
    alias_text = alias_sentence(recipe, locale)
    if alias_text and aliases:
        extra_bits.append(alias_text)
    if modality and not mentions(opening + " " + method, "modality", "模态", "Documented"):
        extra_bits.append(modality)
    ids = variant_ids(recipe)
    others = [item for item in ids if item != (preferred_variant(recipe) or model_id)][:3]
    if others:
        joined_zh = join_zh(f"`{item}`" for item in others)
        joined_en = ", ".join(f"`{item}`" for item in others)
        if locale == "zh":
            extra_bits.append(f"同系列还有 {joined_zh}。")
        else:
            extra_bits.append(f"Also recorded: {joined_en}.")
    extra = join_prose(extra_bits, locale)
    if extra:
        parts.append(extra)

    bench = benchmarks_paragraph(docs, locale)
    if bench:
        parts.append(bench)

    cleaned = [re.sub(r"\s+", " ", item).strip() for item in parts if item.strip()]
    return unique_keep(cleaned)[:4]


def contract_paragraphs(recipe: dict[str, Any], locale: str) -> list[str]:
    model_id = str(recipe.get("id") or "")
    docs = recipe.get("docs") or {}
    exemplar = drop_training_prose(exemplar_block(model_id, "contract_extra", locale))
    if exemplar:
        return [
            item
            for item in exemplar[:4]
            if not any(marker in item for marker in SUITE_SCORECARD_MARKERS)
        ]

    parts: list[str] = []
    contract = contract_paragraph(recipe, locale)
    if contract:
        parts.append(contract)
    profiles = task_profile_sentence(recipe, locale)
    if profiles:
        parts.append(profiles)
    variants = variants_paragraph(recipe, locale)
    status = family_status_sentence(recipe, locale)
    variant_block = join_prose([item for item in (variants, status) if item], locale)
    if variant_block:
        parts.append(variant_block)
    knobs = inference_defaults_paragraph(recipe, locale)
    if knobs:
        parts.append(knobs)
    for item in usage_and_blocker_paragraphs(recipe, docs, locale):
        parts.append(item)
    return unique_keep(parts)[:5]


def limits_clause(recipe: dict[str, Any], docs: dict[str, Any], locale: str) -> str:
    """One in-body sentence — never an h2 named Limits / What this page does not claim."""

    licenses = checkpoint_licenses(recipe)
    gated = gated_checkpoints(recipe)
    hardware = docs.get("hardware") or {}
    min_vram = hardware.get("minVramGb")
    label = status_label(recipe, locale)
    bits: list[str] = []
    if locale == "zh":
        if licenses:
            bits.append(f"记录 license 为 {join_zh(licenses)}")
        if gated:
            bits.append(f"gated `{gated[0]}`，下载前需在上游接受条款")
        if min_vram is not None:
            bits.append(f"catalog 显存 {min_vram} GB")
        bits.append(f"状态：**{label}**")
        return "；".join(bits) + "。"
    if licenses:
        bits.append(f"recorded licenses include {join_en(licenses)}")
    if gated:
        bits.append(f"gated `{gated[0]}` — accept upstream terms before download")
    if min_vram is not None:
        bits.append(f"catalog VRAM {min_vram} GB")
    bits.append(f"status: **{label}**")
    text = "; ".join(bits)
    return text[0].upper() + text[1:] + "."


def run_paragraphs(recipe: dict[str, Any], docs: dict[str, Any], locale: str) -> list[str]:
    model_id = str(recipe.get("id") or "")
    exemplar = drop_training_prose(exemplar_block(model_id, "run_extra", locale))
    parts: list[str] = []
    blob = ""

    def add(sentence: str) -> None:
        nonlocal blob
        text = re.sub(r"\s+", " ", sentence).strip()
        if not text or text in parts:
            return
        if mentions(blob, text[:28]):
            return
        parts.append(text)
        blob = " ".join(parts)

    for item in exemplar:
        add(item)

    launch = run_command_sentence(recipe, locale)
    builder = builder_sentence(locale)
    if launch and not mentions(blob, "worldfoundry-eval run"):
        add(join_prose([launch, builder], locale))
    elif not mentions(blob, "command builder", "命令构建器"):
        add(builder)

    env = environment_sentence(recipe, locale)
    if env and not mentions(blob, str((recipe.get("runtime") or {}).get("environmentName") or "___never___")):
        add(env)

    ckpt = checkpoint_sentence(recipe, locale)
    prepare = prepare_sentence(recipe, locale)
    ckpt_needles = (
        "huggingface",
        "checkpoint",
        "权重",
        "gs://",
        "Stage",
        "就位",
        *checkpoint_ids(recipe)[:3],
    )
    ckpt_parts: list[str] = []
    if ckpt and not mentions(blob, *ckpt_needles):
        ckpt_parts.append(ckpt)
    if prepare and not mentions(blob, "prepare_model_infer"):
        ckpt_parts.append(prepare)
    if ckpt_parts:
        add(join_prose(ckpt_parts, locale))

    if not parts:
        usage = drop_training_prose(locale_paragraphs(docs, "usageNotes", locale))
        for item in usage[:1]:
            add(item)
        if launch:
            add(launch)
        if env:
            add(env)

    limits = limits_clause(recipe, docs, locale)
    if limits and not mentions(blob, "license", "VRAM", "显存", "Status:", "状态"):
        add(limits)
    return parts[:4]


def classify_tasks(tasks: list[str]) -> dict[str, bool]:
    lowered = [task.lower() for task in tasks]
    blob = " ".join(lowered)

    def has(*needles: str) -> bool:
        return any(needle in lowered or needle in blob for needle in needles)

    return {
        "t2v": has(
            "text-to-video",
            "video-generation",
            "video_generation",
            "long-video-generation",
            "autoregressive-video-generation",
        ),
        "i2v": has("image-to-video", "text-image-to-video", "image-to-world-video"),
        "v2v": has("video-to-video", "video-editing", "reference-video-to-video"),
        "vla": has(
            "vla",
            "vla.policy_rollout",
            "vla.action_prediction",
            "robot_policy",
            "humanoid_policy",
            "embodied_policy",
        ),
        "visuomotor": has("visuomotor_policy", "action_diffusion", "action_chunking_policy", "imitation_policy"),
        "wam": has("wam", "wam.world_action_modeling", "vla.world_action_model", "world_action_model"),
        "world": has(
            "interactive-world-model",
            "world-model",
            "world_model",
            "world-generation",
            "game-world-model",
            "robot-world-model",
            "embodied-world-model",
            "robotics-world-model",
            "minecraft-world-model",
            "diffusion-world-model",
            "navigation-world-model",
            "multi-agent-world-model",
            "world",
        ),
        "camera": has(
            "camera-control-video",
            "camera-controlled-video",
            "camera-control",
            "camera_control",
            "camera-pose-estimation",
        ),
        "action_video": has("action-conditioned-video", "action-conditioned-generation"),
        "depth": "depth" in blob,
        "geom": has("3d-reconstruction", "geometry-prior", "novel-view-synthesis", "gaussian-splatting", "point-cloud")
        or "geometry" in blob
        or "reconstruction" in blob
        or "splatting" in blob,
        "hosted": has("hosted-api") or "hosted" in blob,
        "eval": has("embodied_benchmark"),
        "audio": has("text-to-audio", "video-to-audio", "audio-generation", "audio-video-generation"),
        "t2i": has("text-to-image", "image-generation", "image-to-image"),
        "reasoning": has("multimodal-reasoning", "spatial-reasoning", "audio-reasoning"),
    }


def launch_command(recipe: dict[str, Any]) -> str:
    run_id, task = run_identity(recipe)
    launch = run_id or preferred_variant(recipe) or str(recipe.get("id") or "")
    if not launch:
        return ""
    cmd = f"worldfoundry-eval run {launch}"
    if task:
        if re.search(r"\s", task):
            cmd += f" --pipeline.task-profile '{task}'"
        else:
            cmd += f" --pipeline.task-profile {task}"
    return cmd


def voice_index(model_id: str, n: int) -> int:
    if n <= 1:
        return 0
    return sum(ord(ch) for ch in model_id) % n


def job_family(recipe: dict[str, Any]) -> str:
    """Pick one opening family from category + recorded tasks. No invented jobs."""

    category = str(recipe.get("category") or "")
    flags = classify_tasks([str(item) for item in recipe.get("tasks") or [] if item])
    if category == "hosted_api" or flags["hosted"]:
        return "hosted"
    if flags["eval"] and not (flags["vla"] or flags["visuomotor"] or flags["wam"]):
        return "eval"
    if flags["vla"]:
        return "vla"
    if flags["wam"]:
        return "wam"
    if flags["visuomotor"]:
        return "visuomotor"
    if flags["geom"]:
        return "geom"
    if flags["depth"]:
        return "depth"
    if flags["camera"]:
        return "camera"
    if flags["world"] or flags["action_video"]:
        return "world"
    if flags["i2v"] and flags["t2v"]:
        return "ti2v"
    if flags["i2v"]:
        return "i2v"
    if flags["v2v"] and not flags["t2v"]:
        return "v2v"
    if flags["t2v"]:
        return "t2v"
    if flags["audio"]:
        return "audio"
    if flags["t2i"]:
        return "t2i"
    if category == "world_models":
        return "world"
    if flags["reasoning"]:
        return "reasoning"
    if category == "vla_va_wam":
        return "vla"
    if category == "three_d_four_d":
        return "geom"
    if category == "video":
        return "t2v"
    return "generic"


# First-section openings: task, I/O, why pick this family. Command lives in Running.
_OPENINGS_EN: dict[str, tuple[str, ...]] = {
    "t2v": (
        "Pick `{variant}` when you want this family's text-to-video line in WorldFoundry: a prompt in, a generated clip out. Use it as the catalog T2V start rather than an editor or a world-model stepper.",
        "`{variant}` is the recorded prompt-to-video route — not image editing and not a policy. Start here when the job is a clip from text.",
        "Use `{variant}` for catalog text-to-video inference. The recipe exists so you can launch a recorded T2V profile without assembling an upstream demo.",
    ),
    "i2v": (
        "Pick `{variant}` when you need still-to-video in this catalog: a reference image in, a generated clip out. Choose it over a text-only sibling when the first frame is given.",
        "`{variant}` is the recorded image-to-video start. Condition on a still; the catalog route writes the clip.",
        "Use `{variant}` to animate a still under the recorded I2V profile. It is an inference recipe, not a training walkthrough.",
    ),
    "ti2v": (
        "Pick `{variant}` when one recipe covers text-to-video and image-to-video: a prompt in, an optional still, a clip out. Start here instead of wiring two upstream demos.",
        "`{variant}` is the recorded text/image-to-video line. Use it when you want both prompt and still conditioning on the same catalog card.",
        "Use `{variant}` for the family's joint T2V/I2V inference path. The catalog records both task profiles on this page.",
    ),
    "v2v": (
        "Pick `{variant}` when you need to edit or restyle an existing clip: source video in, edited video out. It is not a from-scratch T2V generator.",
        "`{variant}` is the recorded video-to-video route. Feed a source clip when the job is transformation rather than generation from noise.",
        "Use `{variant}` for catalog video-to-video inference. Start here when the input is already a video.",
    ),
    "vla": (
        "Pick `{variant}` when you want the catalog VLA policy: instruction and observation in, an action trace out. Use it to roll the recorded policy, not to train a new one.",
        "`{variant}` is the recorded instruction-to-action line. Start here when you need a WorldFoundry VLA rollout rather than a video generator.",
        "Use `{variant}` for vision-language-action inference on this card. The recipe exists so you can launch the recorded policy profile.",
    ),
    "visuomotor": (
        "Pick `{variant}` for closed-loop visuomotor chunks: observation in, an action chunk out. Choose it when you need the recorded imitation/diffusion policy rather than a VLM chat model.",
        "`{variant}` is the recorded visuomotor policy. Start here for observation-to-action-chunk inference.",
        "Use `{variant}` when the job is visuomotor control on the catalog action contract.",
    ),
    "wam": (
        "Pick `{variant}` for world-action modeling: world-conditioned observations in, predicted actions out. Use it when the catalog marks a WAM route rather than a pure VLA or video model.",
        "`{variant}` is the recorded world-action line. Start here to predict actions conditioned on world state.",
        "Use `{variant}` for the catalog WAM inference path. It is not a pixel world simulator unless this card also records that task.",
    ),
    "world": (
        "Pick `{variant}` when you need an action-conditioned world stepper: recorded actions and a frame in, a rollout out. Use it for interactive world inference, not a text-to-video prompt dump.",
        "`{variant}` is the recorded interactive world line. Start here when the job is stepping a world with actions.",
        "Use `{variant}` for catalog world-model inference. Choose it when you need action-conditioned future frames rather than a standalone clip generator.",
    ),
    "camera": (
        "Pick `{variant}` when you need camera-controlled video: a camera trajectory in, a steered clip out. Use it instead of unconstrained T2V when viewpoint must follow a path.",
        "`{variant}` is the recorded camera-control route. Start here to follow a catalog camera path.",
        "Use `{variant}` for camera-conditioned video inference. The recipe binds the recorded camera-control profile.",
    ),
    "geom": (
        "Pick `{variant}` for feed-forward geometry: views in, reconstruction / trajectory / point cloud out. Use it when you need the catalog 3D route rather than a depth-only prior.",
        "`{variant}` is the recorded reconstruction line. Start here to emit geometry from the documented views.",
        "Use `{variant}` for catalog geometry inference. Choose it when the artifact is a reconstruction, not a generated video.",
    ),
    "depth": (
        "Pick `{variant}` to estimate depth from the recorded inputs: images in, a depth map out. Use it as the catalog depth line, not a full scene reconstructor unless that task is also recorded.",
        "`{variant}` is the recorded depth route. Start here when you need depth maps from the documented inputs.",
        "Use `{variant}` for catalog depth inference. The recipe writes depth under the recorded profile.",
    ),
    "hosted": (
        "Pick `{variant}` when this card is a hosted API rather than a local checkpoint: call the recorded provider route instead of staging weights.",
        "`{variant}` is the hosted inference line. Start here when no local checkpoint is recorded on the card.",
        "Use `{variant}` to hit the catalog hosted route. Local GPU artifacts are not what this page records.",
    ),
    "eval": (
        "Pick `{variant}` as a benchmark or dataset entry, not as a new policy architecture. Use it to run the recorded suite around this card.",
        "`{variant}` is an evaluation row in the catalog. Start here when you need the recorded suite, not a generator.",
        "Use `{variant}` for the catalog eval entry. It does not introduce a separate policy family.",
    ),
    "audio": (
        "Pick `{variant}` when you need catalog audio generation: the recorded inputs in, audio out. Use it instead of a video-only sibling.",
        "`{variant}` is the recorded audio line. Start here to write audio from the documented inputs.",
        "Use `{variant}` for audio inference on this card. The recipe follows the recorded audio profile.",
    ),
    "t2i": (
        "Pick `{variant}` for text-to-image on this card: a prompt in, an image out. Use it when the catalog task is a still, not a clip.",
        "`{variant}` is the recorded image-generation line. Start here for prompt-to-image inference.",
        "Use `{variant}` when the job is a catalog still image rather than video.",
    ),
    "reasoning": (
        "Pick `{variant}` for the recorded multimodal reasoning route: documented inputs in, a reasoning artifact out. Use it when the card is a reasoner, not a generator.",
        "`{variant}` is the catalog reasoning line. Start here for the recorded reasoning profile.",
        "Use `{variant}` for multimodal reasoning inference on this card.",
    ),
    "generic": (
        "Pick `{variant}` as the catalog starting variant for this family. Use it when you need the recorded WorldFoundry inference path rather than an upstream demo checkout.",
        "`{variant}` is the recorded catalog path. Start here for the inference contract on this page.",
        "Use `{variant}` to launch the recorded recipe. The card exists so you can run the catalog profile in WorldFoundry.",
    ),
}

_OPENINGS_ZH: dict[str, tuple[str, ...]] = {
    "t2v": (
        "需要本系列的文生视频时选 `{variant}`：prompt 进、生成视频出。它是 catalog 上的 T2V 起步线，不是编辑器，也不是世界模型步进器。",
        "`{variant}` 是记录的 prompt 到视频路由——不是图像编辑，也不是策略。只要任务是文生视频，从这里起步。",
        "在 WorldFoundry 里用 `{variant}` 做 catalog 文生视频推理。这条配方让你按已记录的 T2V profile 启动，而不用自己拼上游 demo。",
    ),
    "i2v": (
        "需要把静帧变成视频时选 `{variant}`：参考图进、生成视频出。已有第一帧时，优先于纯文本兄弟条目。",
        "`{variant}` 是记录的图生视频起步线。用静帧做条件，catalog 路由写出视频。",
        "用 `{variant}` 按记录的 I2V profile 给静帧加运动。这是推理配方，不是训练教程。",
    ),
    "ti2v": (
        "一条配方同时覆盖文生视频与图生视频时选 `{variant}`：prompt 进、可选静帧、视频出。不必再拼两条上游 demo。",
        "`{variant}` 是记录的文本/图像到视频线。需要同一张卡片上同时有 prompt 与静帧条件时用它。",
        "用 `{variant}` 跑本系列联合 T2V/I2V 推理路径。本页记录了这两种任务 profile。",
    ),
    "v2v": (
        "需要改已有视频时选 `{variant}`：源视频进、编辑后视频出。它不是从噪声开始的 T2V 生成器。",
        "`{variant}` 是记录的视频到视频路由。任务是变换而不是从噪声生成时，喂入源视频。",
        "用 `{variant}` 做 catalog 视频到视频推理。输入已经是视频时从这里起步。",
    ),
    "vla": (
        "需要 catalog 上的 VLA 策略时选 `{variant}`：指令与观测进、动作轨迹出。用来滚动已记录策略，不是用来训练新策略。",
        "`{variant}` 是记录的指令到动作线。需要 WorldFoundry VLA rollout 而不是视频生成时从这里起步。",
        "用 `{variant}` 做本卡的视觉-语言-动作推理。这条配方用来启动已记录的策略 profile。",
    ),
    "visuomotor": (
        "需要闭环视觉运动块时选 `{variant}`：观测进、动作块出。要的是记录的模仿/扩散策略，而不是 VLM 对话模型。",
        "`{variant}` 是记录的 visuomotor 策略。从观测到动作块的推理从这里起步。",
        "任务是 catalog 动作契约上的视觉运动控制时，用 `{variant}`。",
    ),
    "wam": (
        "做世界-动作建模时选 `{variant}`：世界条件观测进、预测动作出。catalog 标成 WAM 路由、而不是纯 VLA 或视频模型时用它。",
        "`{variant}` 是记录的世界-动作线。按世界状态预测动作时从这里起步。",
        "用 `{variant}` 跑 catalog 的 WAM 推理路径。除非本卡还记录了像素世界仿真任务，否则它不是像素世界模拟器。",
    ),
    "world": (
        "需要动作条件世界步进时选 `{variant}`：记录动作与帧进、滚动生成出。用来做交互世界推理，不是把 prompt 倒进文生视频。",
        "`{variant}` 是记录的交互世界线。要用动作推进世界时从这里起步。",
        "用 `{variant}` 做 catalog 世界模型推理。需要动作条件未来帧、而不是独立片段生成器时选它。",
    ),
    "camera": (
        "需要相机可控视频时选 `{variant}`：相机轨迹进、受控视频出。视角必须跟随路径时，用它而不是无约束 T2V。",
        "`{variant}` 是记录的相机控制路由。按 catalog 相机路径推进时从这里起步。",
        "用 `{variant}` 做相机条件视频推理。配方绑定了已记录的相机控制 profile。",
    ),
    "geom": (
        "做前馈几何时选 `{variant}`：视图进、重建/轨迹/点云出。需要 catalog 的 3D 路由、而不是仅深度先验时用它。",
        "`{variant}` 是记录的重建线。从记录的视图写出几何时从这里起步。",
        "用 `{variant}` 做 catalog 几何推理。产物是重建而不是生成视频时选它。",
    ),
    "depth": (
        "按记录输入估计深度时选 `{variant}`：图像进、深度图出。它是 catalog 深度线；除非还记录了重建任务，否则不是完整场景重建器。",
        "`{variant}` 是记录的深度路由。需要从记录输入出深度图时从这里起步。",
        "用 `{variant}` 做 catalog 深度推理。配方按记录 profile 写出深度。",
    ),
    "hosted": (
        "本卡是托管 API 而不是本地 checkpoint 时选 `{variant}`：走记录的提供商路由，而不是就位权重。",
        "`{variant}` 是托管推理线。卡片上没有本地 checkpoint 时从这里起步。",
        "用 `{variant}` 打到 catalog 托管路由。本页记录的不是本地 GPU 产物。",
    ),
    "eval": (
        "把 `{variant}` 当评测或数据集条目，而不是新的策略架构。用它跑本卡记录的套件。",
        "`{variant}` 是 catalog 里的评测行。需要记录套件而不是生成器时从这里起步。",
        "用 `{variant}` 跑 catalog 评测条目。它不引入单独的策略家族。",
    ),
    "audio": (
        "需要 catalog 音频生成时选 `{variant}`：记录输入进、音频出。用它而不是纯视频兄弟条目。",
        "`{variant}` 是记录的音频线。按记录输入写出音频时从这里起步。",
        "用 `{variant}` 做本卡音频推理。配方走记录的音频 profile。",
    ),
    "t2i": (
        "本卡做文生图时选 `{variant}`：prompt 进、图像出。catalog 任务是静帧而不是视频时用它。",
        "`{variant}` 是记录的图像生成线。prompt 到图像的推理从这里起步。",
        "任务是 catalog 静帧而不是视频时，用 `{variant}`。",
    ),
    "reasoning": (
        "走记录的多模态推理路由时选 `{variant}`：记录输入进、推理产物出。本卡是推理器而不是生成器时用它。",
        "`{variant}` 是 catalog 推理线。按记录的推理 profile 从这里起步。",
        "用 `{variant}` 做本卡的多模态推理。",
    ),
    "generic": (
        "把 `{variant}` 当本系列在 catalog 上的起步 variant。需要 WorldFoundry 里已记录的推理路径、而不是检出上游 demo 时用它。",
        "`{variant}` 是记录的 catalog 路径。本页的推理契约从这里起步。",
        "用 `{variant}` 启动已记录配方。这张卡片用来在 WorldFoundry 里跑 catalog profile。",
    ),
}

_RUN_CMD_EN: dict[str, tuple[str, ...]] = {
    "t2v": ("Launch `{cmd}` to sample the clip.", "The catalog launch is `{cmd}`."),
    "i2v": ("Launch `{cmd}` to animate the still.", "The catalog launch is `{cmd}`."),
    "ti2v": ("Launch `{cmd}` to sample the clip.", "The catalog launch is `{cmd}`."),
    "v2v": ("Launch `{cmd}` to edit the clip.", "The catalog launch is `{cmd}`."),
    "vla": ("Launch `{cmd}` to execute the policy.", "The catalog launch is `{cmd}`."),
    "visuomotor": ("Launch `{cmd}` to run the visuomotor policy.", "The catalog launch is `{cmd}`."),
    "wam": ("Launch `{cmd}` to run the world-action route.", "The catalog launch is `{cmd}`."),
    "world": ("Launch `{cmd}` to step the world.", "The catalog launch is `{cmd}`."),
    "camera": ("Launch `{cmd}` to follow the camera path.", "The catalog launch is `{cmd}`."),
    "geom": ("Launch `{cmd}` to write geometry.", "The catalog launch is `{cmd}`."),
    "depth": ("Launch `{cmd}` to write depth.", "The catalog launch is `{cmd}`."),
    "hosted": ("Launch `{cmd}` to call the hosted route.", "The catalog launch is `{cmd}`."),
    "eval": ("Launch `{cmd}` to run the eval entry.", "The catalog launch is `{cmd}`."),
    "audio": ("Launch `{cmd}` to write audio.", "The catalog launch is `{cmd}`."),
    "t2i": ("Launch `{cmd}` to write the image.", "The catalog launch is `{cmd}`."),
    "reasoning": ("Launch `{cmd}` to run the reasoning route.", "The catalog launch is `{cmd}`."),
    "generic": ("The catalog command is `{cmd}`.", "Issue `{cmd}` after weights are staged."),
}

_RUN_CMD_ZH: dict[str, tuple[str, ...]] = {
    "t2v": ("启动 `{cmd}` 采样视频。", "catalog 启动命令是 `{cmd}`。"),
    "i2v": ("启动 `{cmd}` 给静帧加运动。", "catalog 启动命令是 `{cmd}`。"),
    "ti2v": ("启动 `{cmd}` 采样视频。", "catalog 启动命令是 `{cmd}`。"),
    "v2v": ("启动 `{cmd}` 改视频。", "catalog 启动命令是 `{cmd}`。"),
    "vla": ("启动 `{cmd}` 执行策略。", "catalog 启动命令是 `{cmd}`。"),
    "visuomotor": ("启动 `{cmd}` 跑 visuomotor 策略。", "catalog 启动命令是 `{cmd}`。"),
    "wam": ("启动 `{cmd}` 跑世界-动作路由。", "catalog 启动命令是 `{cmd}`。"),
    "world": ("启动 `{cmd}` 推进世界。", "catalog 启动命令是 `{cmd}`。"),
    "camera": ("启动 `{cmd}` 跟随相机路径。", "catalog 启动命令是 `{cmd}`。"),
    "geom": ("启动 `{cmd}` 写出几何。", "catalog 启动命令是 `{cmd}`。"),
    "depth": ("启动 `{cmd}` 写出深度。", "catalog 启动命令是 `{cmd}`。"),
    "hosted": ("启动 `{cmd}` 调托管接口。", "catalog 启动命令是 `{cmd}`。"),
    "eval": ("启动 `{cmd}` 跑评测条目。", "catalog 启动命令是 `{cmd}`。"),
    "audio": ("启动 `{cmd}` 写出音频。", "catalog 启动命令是 `{cmd}`。"),
    "t2i": ("启动 `{cmd}` 出图。", "catalog 启动命令是 `{cmd}`。"),
    "reasoning": ("启动 `{cmd}` 跑推理路由。", "catalog 启动命令是 `{cmd}`。"),
    "generic": ("catalog 命令是 `{cmd}`。", "权重就位后执行 `{cmd}`。"),
}


def _fill_template(template: str, variant: str, cmd: str) -> str:
    text = template.replace("{variant}", variant).replace("{cmd}", cmd)
    return re.sub(r"\s+", " ", text).strip()


def family_opening(recipe: dict[str, Any], locale: str) -> str:
    model_id = str(recipe.get("id") or "")
    variant = preferred_variant(recipe) or model_id
    cmd = launch_command(recipe)
    if not variant:
        return ""
    if not cmd:
        cmd = f"worldfoundry-eval run {variant}"
    table = _OPENINGS_ZH if locale == "zh" else _OPENINGS_EN
    selected = next(
        (item for item in recipe.get("variants") or [] if item.get("id") == variant),
        None,
    )
    opening_recipe = {**recipe, "tasks": [selected["task"]]} if selected and selected.get("task") else recipe
    family = job_family(opening_recipe)
    voices = table.get(family) or table["generic"]
    return _fill_template(voices[voice_index(model_id, len(voices))], variant, cmd)


def infer_use_cases(recipe: dict[str, Any], docs: dict[str, Any], locale: str) -> list[str]:
    explicit = locale_paragraphs(docs, "useCases", locale)
    if explicit:
        return explicit[:3]
    exemplar = exemplar_block(str(recipe.get("id") or ""), "use_cases", locale)
    if exemplar:
        return exemplar[:3]

    name = str(recipe.get("name") or recipe.get("id") or "")
    tasks = [str(item) for item in recipe.get("tasks") or [] if item]
    flags = classify_tasks(tasks)
    fields = required_fields(recipe)
    artifacts = artifact_names(recipe)
    pipe = pipeline_class(recipe)
    run_id, run_task = run_identity(recipe)
    action = action_detail(recipe)
    ids = variant_ids(recipe)
    artifacts_s = join_en(f"`{item}`" for item in artifacts[:2]) if locale == "en" else join_zh(f"`{item}`" for item in artifacts[:2])
    fields_s = join_en(f"`{item}`" for item in fields[:4]) if locale == "en" else join_zh(f"`{item}`" for item in fields[:4])
    cases: list[str] = []

    if locale == "zh":
        if flags["eval"] and not flags["vla"]:
            cases.append(
                f"把 {name} 当作评测/数据集条目使用"
                + (f"，经 `{pipe}`" if pipe else "")
                + (f" 写出 {artifacts_s}" if artifacts_s else "")
                + "；它本身不是新的策略架构。"
            )
        elif flags["vla"] or flags["visuomotor"]:
            cases.append(
                f"滚动 {name} 的{task_phrase(tasks[0], 'zh') if tasks else '策略'}："
                + (f"必填 {fields_s}" if fields_s else "按记录的观测/指令输入")
                + (f"，产物为 {artifacts_s}" if artifacts_s else "")
                + (f"。启动 `worldfoundry-eval run {run_id}`" if run_id else "")
                + "。"
            )
        elif flags["camera"]:
            cases.append(
                f"用 {name} 做相机可控生成"
                + (f"，`actions` 记录为 {action}" if action else "")
                + (f"，产物 {artifacts_s}" if artifacts_s else "")
                + (f"。任务 profile 为 `{run_task}`" if run_task else "")
                + "。"
            )
        elif flags["world"] or flags["action_video"]:
            cases.append(
                f"驱动 {name} 的交互/动作条件世界路径"
                + (f"：{action}" if action else "")
                + (f"，写出 {artifacts_s}" if artifacts_s else "")
                + (f"。记录命令为 `worldfoundry-eval run {run_id}`" if run_id else "")
                + "。"
            )
        elif flags["geom"] or flags["depth"]:
            cases.append(
                f"用 {name} 跑{task_phrase(tasks[0], 'zh') if tasks else '几何/深度'}："
                + (f"输入 {fields_s}" if fields_s else "按记录输入")
                + (f"，产物 {artifacts_s}" if artifacts_s else "")
                + "。"
            )
        elif flags["hosted"]:
            cases.append(
                f"在本地权重不可用时通过托管路径调用 {name}"
                + (f"（`worldfoundry-eval run {run_id}`）" if run_id else "")
                + (f"。记录任务：{join_zh(task_phrase(item, 'zh') for item in tasks[:3])}" if tasks else "")
                + "。"
            )
        elif flags["i2v"] or flags["t2v"] or flags["v2v"]:
            mode = "、".join(
                label
                for flag, label in (
                    (flags["t2v"], "文生视频"),
                    (flags["i2v"], "图生视频"),
                    (flags["v2v"], "视频到视频/编辑"),
                )
                if flag
            )
            cases.append(
                f"用 {name} 做{mode or '视频生成'}"
                + (f"：必填 {fields_s}" if fields_s else "")
                + (f"，经 `{pipe}`" if pipe else "")
                + (f" 写出 {artifacts_s}" if artifacts_s else "")
                + "。"
            )
        elif tasks:
            cases.append(
                f"覆盖 {name} 已记录的{join_zh(task_phrase(item, 'zh') for item in tasks[:3])}任务"
                + (f"，启动 `worldfoundry-eval run {run_id}`" if run_id else "")
                + (f"，产物 {artifacts_s}" if artifacts_s else "")
                + "。"
            )
    else:
        if flags["eval"] and not flags["vla"]:
            cases.append(
                f"Treat {name} as an evaluation/dataset entry"
                + (f" through `{pipe}`" if pipe else "")
                + (f" that writes {artifacts_s}" if artifacts_s else "")
                + " — it is not a new policy architecture."
            )
        elif flags["vla"] or flags["visuomotor"]:
            cases.append(
                f"Roll out {name} for {task_phrase(tasks[0], 'en') if tasks else 'policy inference'}:"
                + (f" required {fields_s}" if fields_s else " use the recorded observation/instruction fields")
                + (f", writing {artifacts_s}" if artifacts_s else "")
                + (f". Launch `worldfoundry-eval run {run_id}`" if run_id else "")
                + "."
            )
        elif flags["camera"]:
            cases.append(
                f"Steer camera motion with {name}"
                + (f"; `actions` is recorded as {action}" if action else "")
                + (f", writing {artifacts_s}" if artifacts_s else "")
                + (f". Task profile `{run_task}`" if run_task else "")
                + "."
            )
        elif flags["world"] or flags["action_video"]:
            cases.append(
                f"Drive {name} as an action-conditioned world route"
                + (f" ({action})" if action else "")
                + (f" that emits {artifacts_s}" if artifacts_s else "")
                + (f". Recorded launch: `worldfoundry-eval run {run_id}`" if run_id else "")
                + "."
            )
        elif flags["geom"] or flags["depth"]:
            cases.append(
                f"Run {name} for {task_phrase(tasks[0], 'en') if tasks else 'geometry or depth'}:"
                + (f" inputs {fields_s}" if fields_s else " recorded inputs")
                + (f", producing {artifacts_s}" if artifacts_s else "")
                + "."
            )
        elif flags["hosted"]:
            cases.append(
                f"Call {name} through the hosted provider route"
                + (f" (`worldfoundry-eval run {run_id}`)" if run_id else "")
                + (f". Recorded tasks: {join_en(task_phrase(item, 'en') for item in tasks[:3])}" if tasks else "")
                + "."
            )
        elif flags["i2v"] or flags["t2v"] or flags["v2v"]:
            mode = join_en(
                [
                    label
                    for flag, label in (
                        (flags["t2v"], "text-to-video"),
                        (flags["i2v"], "image-to-video"),
                        (flags["v2v"], "video-to-video or editing"),
                    )
                    if flag
                ]
            )
            cases.append(
                f"Use {name} for {mode or 'video generation'}"
                + (f": required {fields_s}" if fields_s else "")
                + (f" through `{pipe}`" if pipe else "")
                + (f", writing {artifacts_s}" if artifacts_s else "")
                + "."
            )
        elif tasks:
            cases.append(
                f"Cover {name}'s recorded {join_en(task_phrase(item, 'en') for item in tasks[:3])} tasks"
                + (f" via `worldfoundry-eval run {run_id}`" if run_id else "")
                + (f", writing {artifacts_s}" if artifacts_s else "")
                + "."
            )

    if len(ids) > 1:
        shown = ids[:4]
        more = len(ids) - 4
        joined = join_zh(f"`{item}`" for item in shown) if locale == "zh" else ", ".join(f"`{item}`" for item in shown)
        if locale == "zh":
            cases.append(f"按 variant 选择路线：{joined}" + (f" 等共 {len(ids)} 个" if more > 0 else "") + "。")
        else:
            cases.append(
                f"Choose a recorded variant — {joined}"
                + (f" ({len(ids)} total)" if more > 0 else "")
                + "."
            )
    else:
        ckpts = checkpoint_ids(recipe)[:2]
        if ckpts:
            joined = join_zh(f"`{item}`" for item in ckpts) if locale == "zh" else join_en(f"`{item}`" for item in ckpts)
            if locale == "zh":
                cases.append(f"先就位记录的权重 {joined}，再跑上述命令。")
            else:
                cases.append(f"Stage the recorded weights {joined} before launching the command above.")

    bench = benchmarks_paragraph(docs, locale)
    if bench:
        cases.append(bench)

    cleaned = []
    for item in cases:
        text = re.sub(r"\s+", " ", item).replace("。.", "。").replace("..", ".").strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned[:3]


def description_text(recipe: dict[str, Any], locale: str) -> str:
    model_id = str(recipe.get("id") or "")
    exemplar = exemplar_block(model_id, "description", locale)
    if exemplar:
        return clip_sentence(exemplar[0], 200)

    docs = recipe.get("docs") or {}
    name = str(recipe.get("name") or model_id)
    provider = str(recipe.get("provider") or publisher_name(docs, locale) or "").strip()
    if not is_real_publisher(provider):
        provider = ""
    tasks: list[str] = []
    for item in recipe.get("tasks") or []:
        phrase = task_phrase(str(item), locale)
        if locale == "zh" and not re.search(r"[\u4e00-\u9fff]", phrase) and len(phrase) > 20:
            continue
        tasks.append(phrase)
        if len(tasks) == 2:
            break
    if locale == "zh":
        category_zh = str(recipe.get("categoryLabelZh") or "").strip()
        if not tasks and category_zh:
            tasks.append(category_zh)
    ids = variant_ids(recipe)
    paper = paper_record(docs)
    title = str(paper.get("title") or "").strip()
    runtime = recipe.get("runtime") or {}
    env_kind = str(runtime.get("environmentKind") or "")
    overview0 = (locale_paragraphs(docs, "overview", locale) or [""])[0]

    extras: list[str] = [status_label(recipe, locale)]
    if len(ids) > 1:
        extras.append(f"{len(ids)} variants" if locale == "en" else f"{len(ids)} 个 variant")
    if env_kind == "dedicated" and runtime.get("environmentName"):
        extras.append(
            f"dedicated `{runtime['environmentName']}`"
            if locale == "en"
            else f"独立环境 `{runtime['environmentName']}`"
        )
    if title and title not in overview0:
        short = title.split(":")[0].strip()
        if short and short.lower() not in name.lower():
            extras.append(short)

    task_bit = join_en(tasks) if locale == "en" else join_zh(tasks)
    extra_bit = "; ".join(extras[:2])
    if locale == "zh":
        head = f"{name}" + (f"（{provider}）" if provider else "")
        desc = head + (f"：{task_bit}。" if task_bit else "。")
        if extra_bit:
            desc = desc.rstrip("。") + f"；{extra_bit}。"
    else:
        head = f"{name}" + (f" from {provider}" if provider else "")
        desc = head + (f": {task_bit}." if task_bit else ".")
        if extra_bit:
            desc = desc.rstrip(".") + f"; {extra_bit}."

    if mentions(overview0, desc[:40]) or desc.rstrip(".") in overview0:
        run_id, _ = run_identity(recipe)
        if run_id and run_id not in desc:
            suffix = f" Catalog variant `{run_id}`." if locale == "en" else f"catalog variant `{run_id}`。"
            desc = desc.rstrip(".") + "." + suffix if locale == "en" else desc.rstrip("。") + "。" + suffix
    if locale == "zh" and not re.search(r"[\u4e00-\u9fff]", desc):
        category_zh = str(recipe.get("categoryLabelZh") or "").strip()
        if category_zh:
            desc = f"{desc.rstrip('。')}；{category_zh}。"
    return clip_sentence(desc, 200)


def intro_body(recipe: dict[str, Any], docs: dict[str, Any], locale: str, lead: str) -> list[str]:
    overview = locale_paragraphs(docs, "overview", locale)
    body = [item for item in overview[1:] if item and item != lead]
    extras = list(exemplar_block(str(recipe.get("id") or ""), "intro_extra", locale))
    paper = paper_sentence(docs, locale, recipe)
    if paper and not mentions(lead, "arxiv", "paper", "论文"):
        extras.append(paper)
    aliases = [str(item) for item in recipe.get("aliases") or [] if item]
    if aliases and not any(mentions(lead, alias) for alias in aliases[:3]):
        extras.append(alias_sentence(recipe, locale))
    if not body:
        pipe = pipeline_class(recipe)
        run_id, task = run_identity(recipe)
        env = str((recipe.get("runtime") or {}).get("environmentName") or "").strip()
        bits: list[str] = []
        if locale == "zh":
            if pipe:
                bits.append(f"WorldFoundry 默认走 `{pipe}`")
            if run_id:
                bits.append(f"启动 `worldfoundry-eval run {run_id}`" + (f"（`{task}`）" if task else ""))
            if env:
                bits.append(f"环境 `{env}`")
            if bits:
                extras.insert(0, "；".join(bits) + "。")
        else:
            if pipe:
                bits.append(f"WorldFoundry's default pipeline class is `{pipe}`")
            if run_id:
                bits.append(f"the recorded launch is `worldfoundry-eval run {run_id}`" + (f" (`{task}`)" if task else ""))
            if env:
                bits.append(f"the environment profile is `{env}`")
            if bits:
                text = "; ".join(bits)
                extras.insert(0, text[0].upper() + text[1:] + ".")
    publisher = publisher_name(docs, locale)
    if (
        publisher
        and not mentions(lead, publisher, str(recipe.get("provider") or ""))
        and not mentions(publisher, "发布", "institution not recorded", "not recorded")
    ):
        extras.append(f"该模型由{publisher}发布。" if locale == "zh" else f"It is published by {publisher}.")
    modality = modality_sentence(docs, locale)
    if modality and not mentions(lead, "modality", "模态", "takes ", "接受", "text", "image", "video"):
        extras.append(modality)
    status = status_sentence(recipe, locale)
    if status and not mentions(
        lead,
        "verified",
        "pending",
        "blocked",
        "validated",
        "已验证",
        "集成",
        "parity",
        "GPU-validated",
        "runner",
    ):
        extras.append(status)
    for item in extras:
        if item and item not in body and not mentions(lead, item[:28]):
            body.append(item)
    return body


def is_paper_cover_src(src: str) -> bool:
    return str(src or "").strip().lower().endswith(PAPER_COVER_SUFFIX)


def is_arch_asset_src(src: str) -> bool:
    normalized = str(src or "").strip().lower()
    return any(normalized.endswith(suffix) for suffix in ARCH_ASSET_SUFFIXES)


def body_arch_src(model_id: str, figures: list[dict[str, Any]]) -> str | None:
    """Real architecture figure only — never a paper first-page cover."""
    public = ROOT / "public" / "models" / model_id
    for name in ARCH_FILE_NAMES:
        if (public / name).is_file():
            return f"/models/{model_id}/{name}"
    for item in figures:
        src = str(item.get("src") or "").strip()
        kind = item.get("kind") or "image"
        if kind == "image" and src and not is_paper_cover_src(src):
            return src
    return None


def is_title_or_body_media_src(src: str, arch_src: str | None) -> bool:
    if is_paper_cover_src(src) or is_arch_asset_src(src):
        return True
    return bool(arch_src) and src == arch_src


def is_demo_video_figure(figure: dict[str, Any]) -> bool:
    """Catalog / Studio demo clips — never emit these on model homes."""
    kind = figure.get("kind") or "image"
    src = str(figure.get("src") or "").strip().lower()
    if kind == "video":
        return True
    return src.endswith((".mp4", ".webm", ".mov", ".m4v")) or "/demos/" in src


def hero_figures(
    figures: list[dict[str, Any]], arch_src: str | None = None
) -> list[dict[str, Any]]:
    """Never emit demo clips. Paper teasers come from ModelPaperFigures."""
    return []


def supplemental_figures(
    figures: list[dict[str, Any]], arch_src: str | None = None
) -> list[dict[str, Any]]:
    """Never emit demo clips on model homes."""
    return []


def arch_diagram_block(model_id: str, locale: str) -> str:
    return f'<ModelArchDiagram modelId="{model_id}" locale="{locale}" />'


def paper_figures_block(model_id: str, locale: str) -> str:
    return f'<ModelPaperFigures modelId="{model_id}" locale="{locale}" />'


def figure_block(figure: dict[str, Any], locale: str) -> str | None:
    src = str(figure.get("src") or "").strip()
    if not src:
        return None
    caption = str(figure.get("captionZh") if locale == "zh" else figure.get("caption") or "").strip()
    if not caption:
        caption = str(figure.get("caption") or "").strip()
    alt = mdx_escape(caption)
    kind = figure.get("kind") or "image"
    if is_demo_video_figure(figure) or kind == "video":
        return None
    image = f'<img src="{src}" alt="{alt}" />'
    if caption:
        return (
            f'<figure className="wf-recipe-media is-diagram">\n  {image}\n  '
            f"<figcaption>{alt}</figcaption>\n</figure>"
        )
    return f'<figure className="wf-recipe-media is-diagram">\n  {image}\n</figure>'


def render_page(
    recipe: dict[str, Any],
    locale: str,
    *,
    page_source_override: str | None = None,
    extras: dict[str, list[str]] | None = None,
) -> str:
    docs = recipe.get("docs") or {}
    headings = HEADINGS[locale]
    model_id = str(recipe["id"])
    name = str(recipe.get("name") or model_id)
    figures = [item for item in (docs.get("figures") or []) if isinstance(item, dict)]
    arch_src = body_arch_src(model_id, figures)
    extras = extras or {}

    overview = drop_training_prose(locale_paragraphs(docs, "overview", locale))
    if not overview:
        overview = [str(recipe.get("summary") or name)]
    lead = sanitize_manifest_dialect(overview[0], locale)

    use_cases = for_what_paragraphs(recipe, locale)
    contract_parts = contract_paragraphs(recipe, locale)
    run_parts = run_paragraphs(recipe, docs, locale)
    source = page_source_override or frontmatter_page_source(model_id)

    chunks = [
        "---",
        f"title: {yaml_string(name)}",
        f"description: {yaml_string(sanitize_manifest_dialect(description_text(recipe, locale), locale))}",
        f"modelId: {yaml_string(model_id)}",
        f"pageSource: {source}",
        "---",
        "",
        emit_prose(lead, locale),
        "",
    ]
    for item in extras.get("lead") or []:
        chunks.append(sanitize_manifest_dialect(item, locale))
        chunks.append("")

    chunks.append(paper_figures_block(model_id, locale))
    chunks.append("")

    paper = paper_sentence(docs, locale, recipe)
    paper_title = str(paper_record(docs).get("title") or "").strip()
    title_rest = paper_title.split(":", 1)[1].strip()[:28] if ":" in paper_title else ""
    paper_already = bool(title_rest and mentions(lead, title_rest)) or (
        bool(paper_title) and ":" not in paper_title and mentions(lead, paper_title[:28])
    )
    if paper and not paper_already:
        chunks.append(emit_prose(paper, locale))
        chunks.append("")

    sources = sources_sentence(recipe, locale)
    already = " ".join([lead, paper or ""] + (extras.get("lead") or []))
    if sources and not any(mentions(already, url[:24]) for url in re.findall(r"https?://[^\s)]+", sources)):
        chunks.append(emit_prose(sources, locale))
        chunks.append("")
    for item in extras.get("mid") or []:
        if item and not mentions(already + (sources or ""), item[:36]):
            chunks.append(sanitize_manifest_dialect(item, locale))
            chunks.append("")
            already += " " + item

    teaser_figures = hero_figures(figures, arch_src)
    if teaser_figures:
        teaser = figure_block(teaser_figures[0], locale)
        if teaser:
            chunks.append(teaser)
            chunks.append("")

    chunks.append(f'<ModelCommandBuilder modelId="{model_id}" locale="{locale}" />')
    chunks.append("")

    chunks.extend([f"## {headings['uses']}", ""])
    what_blob = ""
    if use_cases:
        for item in use_cases:
            chunks.append(emit_prose(item, locale))
            chunks.append("")
            what_blob += " " + item
    else:
        fallback = family_opening(recipe, locale) or (
            f"The recorded path is `{name}`. Use the catalog `worldfoundry-eval run` command."
            if locale == "en"
            else f"记录路径是 `{name}`。用 catalog 的 `worldfoundry-eval run`。"
        )
        chunks.append(emit_prose(fallback, locale))
        chunks.append("")
        what_blob = fallback
    for item in extras.get("what") or []:
        if item and not mentions(what_blob, item[:36]):
            chunks.append(sanitize_manifest_dialect(item, locale))
            chunks.append("")
            what_blob += " " + item

    chunks.extend([f"## {headings['contract']}", ""])
    contract_blob = ""
    if contract_parts:
        for paragraph in unique_keep(contract_parts):
            chunks.append(emit_prose(paragraph, locale))
            chunks.append("")
            contract_blob += " " + paragraph
    else:
        fallback = (
            "Recorded inputs, knobs, and artifacts are unset on this card."
            if locale == "en"
            else "本卡片未记录输入、旋钮或产物。"
        )
        chunks.append(emit_prose(fallback, locale))
        chunks.append("")
        contract_blob = fallback
    for item in extras.get("contract") or []:
        if item and not mentions(contract_blob, item[:36]):
            chunks.append(sanitize_manifest_dialect(item, locale))
            chunks.append("")
            contract_blob += " " + item

    if arch_src and not has_paper_figures(model_id):
        chunks.append(arch_diagram_block(model_id, locale))
        chunks.append("")

    for figure in supplemental_figures(figures, arch_src):
        block = figure_block(figure, locale)
        if block:
            chunks.append(block)
            chunks.append("")

    chunks.extend([f"## {headings['run']}", ""])
    run_blob = ""
    if run_parts:
        for paragraph in unique_keep(run_parts):
            chunks.append(emit_prose(paragraph, locale))
            chunks.append("")
            run_blob += " " + paragraph
    else:
        fallback = (
            run_command_sentence(recipe, locale)
            or environment_sentence(recipe, locale)
            or limits_clause(recipe, docs, locale)
        )
        chunks.append(emit_prose(fallback, locale))
        chunks.append("")
        run_blob = fallback
    for item in extras.get("run") or []:
        if item and not mentions(run_blob, item[:36]):
            chunks.append(sanitize_manifest_dialect(item, locale))
            chunks.append("")
            run_blob += " " + item

    chunks.append(f'<ModelRelatedRecipes modelId="{model_id}" locale="{locale}" />')
    chunks.append("")
    return "\n".join(chunks)


UNIQUE_KEEP_RE = re.compile(
    r"("
    r"\]\(/docs/guides/supported-models/"
    r"|\]\(/zh/docs/guides/supported-models/"
    r"|https?://"
    r"|Not \["
    r"|not \["
    r"|Not MAGI"
    r"|not MAGI"
    r"|not the [0-9]"
    r"|never routes"
    r"|recommended (VideoCrafter|variant|path)"
    r"|不是"
    r"|不要把"
    r"|不要将"
    r"|MoE"
    r"|8xHopper"
    r"|Hopper;"
    r"|vendors "
    r"|sibling"
    r"|姊妹"
    r")",
    re.I,
)

GENERIC_STUB_RE = re.compile(
    r"("
    r"Run the recorded .+ inference route"
    r"|Launch `worldfoundry-eval run"
    r"|Run in the unified environment"
    r"|the runtime does not download weights"
    r"|missing files fail at load, nothing is downloaded"
    r"|跑在统一环境"
    r"|启动 `worldfoundry-eval run"
    r"|运行时不代下权重"
    r"|缺文件加载失败"
    r"|把 `.+` 就位到本地"
    r"|Stage `.+` locally"
    r")",
    re.I,
)
FIELD_LIST_RE = re.compile(
    r"^(Required|Optional|Artifacts?|Status|必填|可选|产物|状态)[:：]",
    re.I,
)


def _split_paras(block: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", (block or "").strip()) if part.strip()]


def _section_body(text: str, heading: str) -> str:
    match = re.search(rf"{re.escape(heading)}\n\n(.+?)(?=\n## |\Z)", text, re.S)
    return match.group(1).strip() if match else ""


def _preamble(text: str) -> str:
    body = text.split("---", 2)[-1] if text.startswith("---") else text
    match = re.search(r"^## ", body, re.M)
    return body[: match.start()] if match else body


def _non_widget_paras(block: str) -> list[str]:
    paras: list[str] = []
    for item in _split_paras(block):
        stripped = item.strip()
        if not stripped or stripped.startswith("<") or stripped.startswith("{"):
            continue
        paras.append(stripped)
    return paras


def _en_word_count(text: str) -> int:
    cleaned = re.sub(r"<[^>]+>", " ", text)
    return len(re.findall(r"[A-Za-z0-9']+", cleaned))


def should_skip_authored_enrich(model_id: str, path: Path) -> str | None:
    """Return a skip reason, or None if the thin authored page may be rewritten."""

    if model_id in HARD_DENY_AUTHORED:
        return "deny"
    if model_id in EXEMPLAR_PROSE:
        return "exemplar"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    headings = set(re.findall(r"^## .+$", text, re.M))
    extra = headings & ENRICH_SKIP_HEADINGS
    if extra:
        return "extra-heading"
    if re.search(r"^\| .+\|", text, re.M):
        return "table"
    what = _section_body(text, "## What it is for")
    if _en_word_count(what) >= 80:
        return "substantial-what"
    return None


def _para_already_covered(para: str, blob: str) -> bool:
    needle = re.sub(r"\s+", " ", para).strip()
    if not needle:
        return True
    if needle in blob:
        return True
    if len(needle) >= 36 and needle[:36] in blob:
        return True
    links = re.findall(r"\]\(([^)]+)\)", para)
    urls = re.findall(r"https?://[^\s)]+", para)
    model_links = re.findall(r"\]\((?:/zh)?/docs/guides/supported-models/[^)]+\)", para)
    stripped = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", para)
    stripped = re.sub(r"https?://\S+", "", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    if links and all(link in blob for link in links) and not model_links:
        if re.search(r"Not \[|不是 \[|不是\[|sibling|姊妹|never routes", para, re.I):
            return False
        if len(stripped) >= 160:
            return False
        return True
    if urls and all(url in blob for url in urls) and not model_links:
        if len(stripped) >= 160:
            return False
        distinctive = UNIQUE_KEEP_RE.search(para)
        if not distinctive or distinctive.group(0).startswith("http"):
            return True
    if model_links and all(link in blob for link in model_links):
        if "Not [" not in para and "不是 [" not in para and "不是[" not in para:
            return True
    return False


def _is_unique_para(para: str, generated_blob: str, *, extra_lead: bool = False) -> bool:
    if _para_already_covered(para, generated_blob):
        return False
    if GENERIC_STUB_RE.search(para) and not UNIQUE_KEEP_RE.search(para):
        return False
    if FIELD_LIST_RE.match(para):
        return False
    if extra_lead:
        return len(para) >= 40
    if UNIQUE_KEEP_RE.search(para):
        return True
    return len(para) >= 120 and not GENERIC_STUB_RE.search(para) and not FIELD_LIST_RE.match(para)


def harvest_authored_extras(existing: str, generated: str, locale: str) -> dict[str, list[str]]:
    """Keep unique authored sentences the catalog renderer does not already emit."""

    headings = REQUIRED_SECTION_HEADINGS[locale]
    run_alt = "## Run in WorldFoundry" if locale == "en" else ""
    preamble = _preamble(existing)
    fig = re.search(r"<ModelPaperFigures[\s\S]*?/>", preamble)
    cmd = re.search(r"<ModelCommandBuilder[\s\S]*?/>", preamble)
    lead_block = preamble[: fig.start()] if fig else preamble
    mid_start = fig.end() if fig else 0
    mid_end = cmd.start() if cmd else len(preamble)
    mid_block = preamble[mid_start:mid_end] if fig else ""

    extras: dict[str, list[str]] = {"lead": [], "mid": [], "what": [], "contract": [], "run": []}
    lead_paras = _non_widget_paras(lead_block)
    for para in lead_paras[1:]:
        if _is_unique_para(para, generated, extra_lead=True):
            extras["lead"].append(para)
    for para in _non_widget_paras(mid_block):
        if _is_unique_para(para, generated):
            extras["mid"].append(para)

    what = _section_body(existing, headings[0])
    contract = _section_body(existing, headings[1])
    run = _section_body(existing, headings[2])
    if not run and run_alt:
        run = _section_body(existing, run_alt)
    for key, block in (("what", what), ("contract", contract), ("run", run)):
        for para in _non_widget_paras(block):
            if key == "run" and GENERIC_STUB_RE.search(para) and not UNIQUE_KEEP_RE.search(para):
                continue
            if _is_unique_para(para, generated):
                extras[key].append(para)
    if locale == "zh":
        for key, items in extras.items():
            extras[key] = [item for item in items if re.search(r"[\u4e00-\u9fff]", item)]
    return extras


def extras_nonempty(extras: dict[str, list[str]]) -> bool:
    return any(extras.get(key) for key in ("lead", "mid", "what", "contract", "run"))


def page_filename(model_id: str, locale: str) -> str:
    return f"{model_id}.mdx" if locale == "en" else f"{model_id}.zh.mdx"


def boilerplate_hits(text: str) -> list[str]:
    return [marker for marker in BOILERPLATE_MARKERS if marker in text]


def forbidden_claim_heading_hits(text: str) -> list[str]:
    return [marker for marker in FORBIDDEN_CLAIM_HEADINGS if marker in text]


def old_five_section_hits(text: str) -> list[str]:
    return [heading for heading in OLD_FIVE_SECTION_HEADINGS if heading in text]


def missing_required_headings(text: str, locale: str) -> list[str]:
    return [heading for heading in REQUIRED_SECTION_HEADINGS[locale] if heading not in text]


def generated_page_violations(content: str, locale: str, label: str) -> list[str]:
    hits: list[str] = []
    hits.extend(f"{label}: {hit}" for hit in forbidden_claim_heading_hits(content))
    hits.extend(f"{label}: {hit}" for hit in old_five_section_hits(content))
    hits.extend(f"{label}: {hit}" for hit in dialect_hits(content))
    hits.extend(f"{label}: missing {hit}" for hit in missing_required_headings(content, locale))
    hits.extend(f"{label}: {hit}" for hit in boilerplate_hits(content))
    if re.search(r"<(Video|video)\b", content):
        hits.append(f"{label}: demo Video/video must not appear on model homes")
    if re.search(r"worldfoundry-eval\s+evaluate", content):
        hits.append(f"{label}: use worldfoundry-eval run, not evaluate")
    if re.search(
        r"(?i)(?:Runner status is|Integration is|runner 状态为|集成为)\s+\*{0,2}['\"]?[a-z0-9]*_[a-z0-9_]+",
        content,
    ):
        hits.append(f"{label}: raw runtimeStatus token in status phrase")
    if not any(marker in content for marker in PAPER_FIGURES_MARKERS):
        hits.append(f"{label}: missing ModelPaperFigures")
    return hits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh generated pages. Authored pages and PRESERVE_AUTHORED ids are never overwritten.",
    )
    parser.add_argument(
        "--enrich-authored",
        action="store_true",
        help=(
            "Rewrite thin authored stubs from catalog facts. Skips HARD_DENY_AUTHORED, "
            "exemplars, tables, extra unique headings, and long What-it-is-for pages."
        ),
    )
    parser.add_argument("--model-id", action="append", dest="model_ids")
    parser.add_argument("--locale", choices=("en", "zh", "both"), default="both")
    args = parser.parse_args()

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    recipes = [item for item in data.get("recipes", []) if isinstance(item, dict)]
    if args.model_ids:
        wanted = set(args.model_ids)
        recipes = [item for item in recipes if item.get("id") in wanted]

    locales = ("en", "zh") if args.locale == "both" else (args.locale,)
    written = 0
    preserved = 0
    skipped_authored = 0
    enriched = 0
    flipped = 0
    skip_reasons: dict[str, int] = {}
    violations: list[str] = []

    for recipe in recipes:
        model_id = str(recipe.get("id") or "").strip()
        if not model_id:
            continue
        skip_reason = should_skip_authored_enrich(model_id, PAGES_DIR / page_filename(model_id, "en"))
        for locale in locales:
            path = PAGES_DIR / page_filename(model_id, locale)
            existing = path.read_text(encoding="utf-8") if path.is_file() else ""
            extras: dict[str, list[str]] = {}
            source_override: str | None = None
            if args.enrich_authored and skip_reason is None and page_source(path) == "authored":
                generated = render_page(recipe, locale, page_source_override="generated")
                extras = harvest_authored_extras(existing, generated, locale) if existing else {}
                source_override = "authored" if extras_nonempty(extras) else "generated"
                content = render_page(
                    recipe,
                    locale,
                    page_source_override=source_override,
                    extras=extras,
                )
            else:
                content = render_page(
                    recipe,
                    locale,
                    page_source_override="generated" if page_source(path) == "generated" else None,
                )
            violations.extend(generated_page_violations(content, locale, f"{model_id}.{locale}"))

            if args.check:
                continue
            if model_id in PRESERVE_AUTHORED or model_id in HARD_DENY_AUTHORED:
                preserved += 1
                continue
            if args.enrich_authored and skip_reason is None and page_source(path) == "authored":
                if not path.is_file() or path.read_text(encoding="utf-8") != content:
                    path.write_text(content, encoding="utf-8")
                    written += 1
                    if source_override == "generated":
                        flipped += 1
                    else:
                        enriched += 1
                continue
            if page_source(path) == "authored":
                skipped_authored += 1
                if args.enrich_authored and skip_reason:
                    skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
                continue
            if args.enrich_authored:
                continue
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                path.write_text(content, encoding="utf-8")
                written += 1

    if violations:
        print("generated model-home prose failed checks:")
        for item in violations[:40]:
            print(f"  {item}")
        if len(violations) > 40:
            print(f"  … {len(violations) - 40} more")
        return 1

    if args.check:
        print(
            f"generated model-home prose is clean "
            f"({len(recipes)} models, {len(EXEMPLAR_PROSE)} exemplars)"
        )
        return 0

    reason_bit = ""
    if args.enrich_authored and skip_reasons:
        reason_bit = "; skip " + ", ".join(f"{key}={value}" for key, value in sorted(skip_reasons.items()))
    print(
        f"wrote {written} model pages under {PAGES_DIR} "
        f"(preserved {preserved}, skipped authored {skipped_authored}"
        f", enriched-in-place {enriched}, flipped-to-generated {flipped}"
        f"{reason_bit}"
        f"{'; --force does not overwrite authored' if args.force else ''})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
