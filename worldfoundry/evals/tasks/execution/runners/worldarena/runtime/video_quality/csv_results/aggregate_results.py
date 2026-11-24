import csv
import glob
import json
import os

# Column order for the final CSV
COLUMN_ORDER: list[str] = [
    "Model_Name",
    "Video_ID",
    "Subject Consistency",
    "Aesthetic Quality",
    "Image Quality",
    "Background Consistency",
    "Dynamic Degree",
    "Interaction Quality",
    "Perspectivity",
    "Instruction Following",
    "Semantic Alignment",
    "Action Following",
    "Flow Score",
    "Depth Accuracy",
    "Trajectory Accuracy",
    "Photometric Consistency",
    "Motion Smoothness",
    "JEPA Similarity",
]

# Map metric keys from JSON to CSV column names
METRIC_KEY_MAP: dict[str, str] = {
    "subject_consistency": "Subject Consistency",
    "aesthetic_quality": "Aesthetic Quality",
    "image_quality": "Image Quality",
    "background_consistency": "Background Consistency",
    "dynamic_degree": "Dynamic Degree",
    "interaction_quality": "Interaction Quality",
    "perspectivity": "Perspectivity",
    "instruction_following": "Instruction Following",
    "semantic_alignment": "Semantic Alignment",
    "action_following": "Action Following",
    "flow_score": "Flow Score",
    "depth_accuracy": "Depth Accuracy",
    "trajectory_accuracy": "Trajectory Accuracy",
    "photometric_consistency": "Photometric Consistency",
    "motion_smoothness": "Motion Smoothness",
    "jepa_similarity": "JEPA Similarity",
}


def _read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _parse_video_id_from_path(path: str) -> str:
    # Normalize separators and strip extension when present
    norm = path.replace("\\", "/")
    name_with_ext = os.path.basename(norm)
    name, _ext = os.path.splitext(name_with_ext)

    def _normalize_episode_token(token: str) -> str:
        lower = token.lower()
        if lower.startswith("episode"):
            rest = token[len("episode") :].lstrip("._")
            if rest:
                return f"episode{rest}"
            return "episode"
        return token

    parts = norm.split("/")
    anchors = [
        "generated_dataset",
        "generated_dataset_action_following",
        "gt_dataset",
    ]
    for anchor in anchors:
        if anchor in parts:
            idx = parts.index(anchor)
            if len(parts) >= idx + 3:
                task = parts[idx + 1]
                episode_raw = parts[idx + 2]
                episode = _normalize_episode_token(episode_raw)
                # Avoid duplicating "episode" prefix
                if episode.lower().startswith("episode"):
                    suffix = episode[len("episode") :]
                    video_id = f"{task}_episode{suffix}"
                else:
                    video_id = f"{task}_{episode}"
                return video_id
    # If path already encodes id (e.g., adjust_bottle_episode40.mp4)
    if name and any(char.isdigit() for char in name):
        return name
    # Fallback to base name
    return name or norm


def _upsert(result: dict[str, dict[str, float]], video_id: str, metric_key: str, value: float | None):
    if value is None:
        return
    column = METRIC_KEY_MAP.get(metric_key, metric_key)
    result.setdefault(video_id, {})[column] = value


def _ingest_metric_json(path: str, result: dict[str, dict[str, float]]):
    data = _read_json(path)
    # Expected format: {metric_name: [overall, [{"video_path":..., "video_results_normalized":...}, ...]], ...}
    for metric_key, payload in data.items():
        if not isinstance(payload, list) or len(payload) < 2:
            continue
        details = payload[1]
        if not isinstance(details, list):
            continue
        for entry in details:
            video_path = entry.get("video_path") or entry.get("video") or entry.get("video_name") or entry.get("name")
            if not video_path:
                continue
            video_id = _parse_video_id_from_path(video_path)
            value = (
                entry.get("video_results_normalized")
                if entry.get("video_results_normalized") is not None
                else entry.get("score_normalized")
            )
            if value is None:
                value = entry.get("video_results") or entry.get("score")
            _upsert(result, video_id, metric_key, value)


def _ingest_vlm_json(path: str, result: dict[str, dict[str, float]]):
    # Expected format: list of {"video": "name.mp4", "metrics": {<Metric>: {"score_normalized": ...}}}
    data = _read_json(path)
    if not isinstance(data, list):
        return
    for item in data:
        video_name = item.get("video")
        if not video_name:
            continue
        video_id = _parse_video_id_from_path(video_name)
        metrics = item.get("metrics", {})
        for metric_key, metric_val in metrics.items():
            if not isinstance(metric_val, dict):
                continue
            value = metric_val.get("score_normalized") or metric_val.get("score")
            normalized_key = metric_key.lower().replace(" ", "_")
            _upsert(result, video_id, normalized_key, value)


def _ingest_jepa_score(path: str) -> float | None:
    if not os.path.exists(path):
        return None
    data = _read_json(path)
    if isinstance(data, dict) and "score" in data:
        return data.get("score")
    return None


def aggregate_results(
    base_dir: str,
    model_name: str,
    csv_name: str = "aggregated_results.csv",
    vlm_model_dir: str | None = None,
    jepa_result_path: str | None = None,
) -> str:
    """
    Aggregate metric outputs into a single CSV.

    Args:
        base_dir: Path to video_quality directory.
        model_name: Model name to populate the CSV Model_Name column.
        csv_name: Output CSV file name.
        vlm_model_dir: Optional folder name under output_VLM (defaults to model_name).
        jepa_result_path: Optional path to JEPA result JSON (expects {"score": float}).

    Returns:
        Path to the written CSV file.
    """

    result: dict[str, dict[str, float]] = {}

    # Core metric JSONs
    core_metric_files = [
        os.path.join(base_dir, "output", "generated_results.json"),
        os.path.join(base_dir, "output_action_following", "generated_results.json"),
    ]
    for path in core_metric_files:
        if os.path.exists(path):
            _ingest_metric_json(path, result)

    # VLM metrics
    vlm_dir = os.path.join(base_dir, "output_VLM", vlm_model_dir or model_name)
    if os.path.isdir(vlm_dir):
        for json_path in glob.glob(os.path.join(vlm_dir, "*.json")):
            _ingest_vlm_json(json_path, result)

    # JEPA score (global, applied to every row)
    jepa_score = None
    if jepa_result_path:
        jepa_score = _ingest_jepa_score(jepa_result_path)
    else:
        # Try common defaults
        candidates = [
            os.path.join(base_dir, "output_JEDi", "generated_results.json"),
            os.path.join(base_dir, "output_JEDi", "results.json"),
            *glob.glob(os.path.join(base_dir, "output_JEDi", "*.json")),
        ]
        for cand in candidates:
            jepa_score = _ingest_jepa_score(cand)
            if jepa_score is not None:
                break

    # Prepare rows
    csv_rows: list[dict[str, str]] = []
    for video_id in sorted(result.keys()):
        row: dict[str, str] = dict.fromkeys(COLUMN_ORDER, "")
        row["Model_Name"] = model_name
        row["Video_ID"] = video_id
        for metric_col, value in result[video_id].items():
            if metric_col in COLUMN_ORDER:
                row[metric_col] = value
        if jepa_score is not None:
            row["JEPA_Similarity"] = jepa_score
        csv_rows.append(row)

    # Ensure output directory exists
    csv_dir = os.path.join(base_dir, "csv_results")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, csv_name)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMN_ORDER)
        writer.writeheader()
        writer.writerows(csv_rows)

    return csv_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate video_quality evaluation outputs into a CSV.")
    parser.add_argument("--base_dir", default=os.path.dirname(__file__), help="Path to video_quality directory")
    parser.add_argument("--model_name", required=True, help="Model name for the CSV column and VLM folder")
    parser.add_argument("--csv_name", default="aggregated_results.csv", help="Output CSV file name")
    parser.add_argument(
        "--vlm_model_dir", default=None, help="Override VLM output folder name if different from model name"
    )
    parser.add_argument("--jepa_result_path", default=None, help='Optional path to JEPA result JSON {"score": float}')
    args = parser.parse_args()

    csv_path = aggregate_results(
        base_dir=args.base_dir,
        model_name=args.model_name,
        csv_name=args.csv_name,
        vlm_model_dir=args.vlm_model_dir,
        jepa_result_path=args.jepa_result_path,
    )
    print(f"CSV written to: {csv_path}")
