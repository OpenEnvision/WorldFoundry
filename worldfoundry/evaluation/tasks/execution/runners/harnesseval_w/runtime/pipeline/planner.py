"""Official manifest readers; automatic plan generation is intentionally excluded."""
from pathlib import Path
from typing import Any, Iterable
from ..io import read_json
from ..protocols import FAMILIES

def load_cases(manifests: Iterable[Path]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    sources: dict[str, Path] = {}
    for manifest_path in manifests:
        manifest = read_json(manifest_path)
        cases = manifest.get("cases") if isinstance(manifest, dict) else None
        if not isinstance(cases, list):
            raise ValueError(f"manifest has no cases array: {manifest_path}")
        for case in cases:
            if not isinstance(case, dict) or not case.get("case_id"):
                raise ValueError(f"manifest contains an invalid case: {manifest_path}")
            case_id = str(case["case_id"])
            family = str((case.get("taxonomy") or {}).get("probe_family") or "")
            if family not in FAMILIES:
                raise ValueError(
                    f"case {case_id!r} has unsupported HarnessEval family {family!r}"
                )
            if case_id in by_id and by_id[case_id] != case:
                raise ValueError(
                    f"case {case_id!r} differs between {sources[case_id]} and {manifest_path}"
                )
            by_id[case_id] = case
            sources[case_id] = manifest_path
    return [by_id[case_id] for case_id in sorted(by_id)]

def initial_observation_path(
    case: dict[str, Any], assets_root: Path | None
) -> Path | None:
    observation = (case.get("world") or {}).get("initial_observation")
    raw_path = observation.get("path") if isinstance(observation, dict) else observation
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    candidates = [path] if path.is_absolute() else []
    if assets_root is not None and not path.is_absolute():
        candidates.append(assets_root / path)
    localization = (case.get("non_model_facing") or {}).get(
        "initial_observation_localization"
    ) or {}
    for key in ("localized_path", "source_path", "original_path"):
        value = localization.get(key) if isinstance(localization, dict) else None
        if not value:
            continue
        candidate = Path(str(value))
        candidates.append(
            candidate
            if candidate.is_absolute() or assets_root is None
            else assets_root / candidate
        )
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
