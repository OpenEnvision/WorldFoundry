#!/usr/bin/env python3
"""Generate docs/fumadocs/lib/catalog-coverage-data.json from WorldFoundry catalogs.

Collapses task variants that only differ by -i2v/-t2v/-v2v/-ti2v into one homepage row
(e.g. videocrafter1-i2v + videocrafter1-t2v → VideoCrafter 1).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "lib" / "catalog-coverage-data.json"

FAMILY_META = {
    "video": (
        "Video",
        "Text-, image-, and video-conditioned generation, editing, and audio-video systems.",
    ),
    "multimodal": (
        "Multimodal",
        "Image-text understanding and generation systems without a video inference route.",
    ),
    "world_models": (
        "World models",
        "Interactive worlds, camera/action conditioning, navigation, and simulator-shaped systems.",
    ),
    "three_d_four_d": (
        "3D / 4D",
        "Reconstruction, depth, point clouds, scene representations, and dynamic geometry.",
    ),
    "vla_va_wam": (
        "VLA / VA / WAM",
        "Embodied policies, world-action models, and robot control stacks.",
    ),
    "hosted_api": (
        "Hosted API",
        "Provider-backed video and world systems that run through API credentials.",
    ),
}

TASK_SUFFIX = re.compile(r"-(i2v|t2v|v2v|ti2v)$", re.I)
TASK_NAME = re.compile(r"\s+(I2V|T2V|V2V|TI2V)$", re.I)
DIGIT_SPACE = re.compile(r"(?<=[A-Za-z])(?=\d)")
MODEL_VERSION_GROUPS = {
    "bernini-r": {
        "name": "Bernini-R",
        "versions": {
            "bernini-r-1.3b": "1.3B",
            "bernini-r-14b": "14B",
        },
    },
    "echo-memory": {
        "name": "Echo-Memory",
        "versions": {
            "echo-memory-context-k1": "Context K=1",
        },
    },
    "wan21-fun-v1.1-cam": {
        "name": "Wan 2.1-Fun V1.1 Control Camera",
        "versions": {
            "wan21-fun-1p3b-cam": "1.3B",
            "wan21-fun-14b-cam": "14B",
        },
    },
    "wan22-fun-cam": {
        "name": "Wan 2.2-Fun Control Camera",
        "versions": {
            "wan22-fun-5b-cam": "5B",
            "wan22-fun-a14b-cam": "A14B",
        },
    },
    "matrix-game-3.5": {
        "name": "Matrix-Game 3.5",
        "versions": {
            "matrix-game-3.5-first-person": "First-Person",
            "matrix-game-3.5-third-person": "Third-Person",
        },
    },
    "dreamx-world-5b": {
        "name": "DreamX-World 5B",
        "versions": {
            "dreamx-world-5b": "AR",
            "dreamx-world-5b-cam": "Cam",
        },
    },
    "depth-anything-v3": {
        "name": "Depth Anything 3",
        "versions": {
            "depth-anything-v3": "Base",
            "depth-anything-v3-prior": "Metric Prior",
        },
    },
    "dust3r": {
        "name": "DUSt3R",
        "versions": {
            "dust3r": "Catalog",
            "dust3r-base-model": "Base",
        },
    },
    "minwm": {
        "name": "minWM",
        "versions": {
            "minwm-hy-action2v": "HY Action2V",
            "minwm-wan-action2v": "Wan Action2V",
        },
    },
    "lingbot-world": {
        "name": "LingBot-World",
        "versions": {
            "lingbot-world": "Cam",
            "lingbot-world-act": "Act",
        },
    },
}
HOMEPAGE_EXCLUDED_MODEL_IDS = {"pi0-worldfoundry"}
# Compact product names that must not get a space before an embedded digit.
PRETTY_NAME_KEEP = {"CUT3R"}


def pretty_name(name: str) -> str:
    name = TASK_NAME.sub("", name).strip()
    if name not in PRETTY_NAME_KEEP:
        name = DIGIT_SPACE.sub(" ", name)
    return re.sub(r"\s+", " ", name)


def model_group_key(model_id: str) -> str:
    for group_id, group in MODEL_VERSION_GROUPS.items():
        if model_id in group["versions"]:
            return group_id
    return TASK_SUFFIX.sub("", model_id)


def load_yaml(path: Path):
    import yaml

    return yaml.safe_load(path.read_text()) or {}


def model_items(path: Path) -> list[dict]:
    data = load_yaml(path)
    if isinstance(data, dict) and isinstance(data.get("models"), list):
        return [item for item in data["models"] if isinstance(item, dict)]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail when generated coverage data is stale.")
    args = parser.parse_args()

    families = []
    for fam, (label, blurb) in FAMILY_META.items():
        raw = []
        for path in sorted((ROOT / "worldfoundry/data/models/catalog" / fam).glob("*.yaml")):
            if path.name.startswith("_"):
                continue
            for item in model_items(path):
                mid = item.get("model_id") or item.get("id")
                if not mid or mid in HOMEPAGE_EXCLUDED_MODEL_IDS:
                    continue
                name = item.get("display_name") or item.get("name") or mid
                status = None
                integ = item.get("integration")
                if isinstance(integ, dict):
                    status = integ.get("status")
                status = status or item.get("integration_status") or item.get("status")
                raw.append({"id": mid, "name": name, "status": status})

        buckets: OrderedDict[str, list[dict]] = OrderedDict()
        for entry in raw:
            key = model_group_key(entry["id"])
            buckets.setdefault(key, []).append(entry)

        entries = []
        for key, group in buckets.items():
            if len(group) == 1:
                entry = group[0]
                entries.append(
                    {
                        "id": entry["id"],
                        "name": pretty_name(entry["name"]),
                        "status": entry.get("status"),
                        "aliases": [entry["id"]],
                    }
                )
            else:
                version_group = MODEL_VERSION_GROUPS.get(key)
                if version_group:
                    version_order = list(version_group["versions"])
                    group = sorted(group, key=lambda item: version_order.index(item["id"]))
                names = [pretty_name(item["name"]) for item in group]
                display = version_group["name"] if version_group else sorted(names, key=len)[0]
                aliases = [item["id"] for item in group]
                statuses = [item.get("status") for item in group if item.get("status")]
                grouped_entry = {
                    "id": key,
                    "name": display,
                    "status": statuses[0] if statuses else None,
                    "aliases": aliases,
                }
                if version_group:
                    grouped_entry["versions"] = [
                        version_group["versions"][item["id"]] for item in group
                    ]
                entries.append(grouped_entry)
        entries.sort(key=lambda item: item["name"].lower())
        families.append(
            {
                "id": fam,
                "label": label,
                "blurb": blurb,
                "count": len(entries),
                "catalogCount": len(raw),
                "entries": entries,
            }
        )

    bench_groups = []
    for fam, label in [("video", "Video & world"), ("embodied", "Embodied")]:
        entries = []
        for path in sorted((ROOT / "worldfoundry/data/benchmarks/catalog" / fam).glob("*.yaml")):
            if path.name.startswith("_"):
                continue
            data = load_yaml(path)
            item = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else {})
            bid = item.get("benchmark_id") or item.get("id") or path.stem
            name = item.get("display_name") or item.get("name") or bid
            entries.append({"id": bid, "name": name, "aliases": [bid]})
        entries.sort(key=lambda item: item["name"].lower())
        bench_groups.append({"id": fam, "label": label, "count": len(entries), "entries": entries})

    payload = {
        "modelsTotal": sum(family["catalogCount"] for family in families),
        "modelsListed": sum(family["count"] for family in families),
        "benchmarksTotal": sum(group["count"] for group in bench_groups),
        "modelFamilies": [
            {key: value for key, value in family.items() if key != "catalogCount"} for family in families
        ],
        "benchmarkGroups": bench_groups,
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.is_file() else ""
        if current != serialized:
            print(f"stale generated catalog coverage data: {OUT}", file=sys.stderr)
            print("run `npm --prefix docs/fumadocs run coverage:generate` to refresh it", file=sys.stderr)
            return 1
        print(f"catalog coverage data is current: {OUT}")
        return 0

    current = OUT.read_text(encoding="utf-8") if OUT.is_file() else ""
    if current != serialized:
        OUT.write_text(serialized, encoding="utf-8")
    print(f"wrote {OUT} modelsTotal={payload['modelsTotal']} listed={payload['modelsListed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
