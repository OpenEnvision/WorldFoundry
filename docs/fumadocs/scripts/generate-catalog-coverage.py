#!/usr/bin/env python3
"""Generate docs/fumadocs/lib/catalog-coverage-data.json from WorldFoundry catalogs.

Collapses task variants that only differ by -i2v/-t2v/-v2v/-ti2v into one homepage row
(e.g. videocrafter1-i2v + videocrafter1-t2v → VideoCrafter 1).
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "lib" / "catalog-coverage-data.json"

FAMILY_META = {
    "video": (
        "Video",
        "Text-, image-, and video-conditioned generation, editing, and audio-video systems.",
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


def pretty_name(name: str) -> str:
    name = TASK_NAME.sub("", name).strip()
    name = DIGIT_SPACE.sub(" ", name)
    return re.sub(r"\s+", " ", name)


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


def main() -> None:
    families = []
    for fam, (label, blurb) in FAMILY_META.items():
        raw = []
        for path in sorted((ROOT / "worldfoundry/data/models/catalog" / fam).glob("*.yaml")):
            if path.name.startswith("_"):
                continue
            for item in model_items(path):
                mid = item.get("model_id") or item.get("id")
                if not mid:
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
            key = TASK_SUFFIX.sub("", entry["id"])
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
                names = [pretty_name(item["name"]) for item in group]
                display = sorted(names, key=len)[0]
                aliases = [item["id"] for item in group]
                statuses = [item.get("status") for item in group if item.get("status")]
                entries.append(
                    {
                        "id": key,
                        "name": display,
                        "status": statuses[0] if statuses else None,
                        "aliases": aliases,
                    }
                )
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
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUT} modelsTotal={payload['modelsTotal']} listed={payload['modelsListed']}")


if __name__ == "__main__":
    main()
