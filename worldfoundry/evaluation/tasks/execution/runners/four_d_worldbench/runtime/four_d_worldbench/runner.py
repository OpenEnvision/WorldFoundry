import argparse
import importlib
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

AdaptiveQAGenerator = None


def _load_adaptive_qa_generator():
    global AdaptiveQAGenerator
    if AdaptiveQAGenerator is not None:
        return AdaptiveQAGenerator
    try:
        from .adaptive_qa_generator import AdaptiveQAGenerator as generator_cls
    except ImportError:
        from adaptive_qa_generator import AdaptiveQAGenerator as generator_cls
    AdaptiveQAGenerator = generator_cls
    return AdaptiveQAGenerator


def _bootstrap_paths_and_aliases():
    """Expose the in-tree metric package under the aliases used by the official code."""
    runtime_root = Path(__file__).resolve().parent
    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))
    try:
        if __package__:
            _metric = importlib.import_module(f"{__package__}.metric")
        else:
            _metric = importlib.import_module("metric")
        sys.modules["metric"] = _metric
        sys.modules["vbench2"] = _metric
        if hasattr(_metric, 'utils'):
            sys.modules['utils'] = _metric.utils
        if hasattr(_metric, 'distributed'):
            sys.modules['vbench2.distributed'] = _metric.distributed
    except Exception:
        pass


def _candidate_dataset_roots(dataset_json: str, root: Dict[str, Any]) -> List[Path]:
    dataset_path = Path(dataset_json).resolve()
    dataset_dir = dataset_path.parent
    base_path = root.get("dataset_info", {}).get("base_path", "") if isinstance(root, dict) else ""
    candidates: List[Path] = []
    for env_name in ("DATASET_BASE_DIR", "WORLDFOUNDRY_GENERATED_ARTIFACT_DIR"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value).expanduser().resolve())
    if base_path:
        base_dir = Path(base_path)
        if not base_dir.is_absolute():
            base_dir = dataset_dir / base_dir
        candidates.append(base_dir.resolve())
        marker = str(base_path).strip("/\\")
        dataset_text = str(dataset_path)
        idx = dataset_text.find(marker)
        if marker and idx > 0:
            candidates.append(Path(dataset_text[:idx]).resolve())
    else:
        candidates.append(dataset_dir)
    candidates.append(dataset_dir)
    deduped: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def _resolve_dataset_video_paths(dataset_json: str, root: Dict[str, Any], videos: List[str]) -> List[str]:
    candidate_roots = _candidate_dataset_roots(dataset_json, root)

    resolved: List[str] = []
    for video_path in videos:
        candidate = Path(video_path)
        if candidate.is_absolute():
            resolved.append(str(candidate))
        else:
            existing = [root / candidate for root in candidate_roots if (root / candidate).exists()]
            resolved.append(str((existing[0] if existing else candidate_roots[0] / candidate).resolve()))
    return resolved


def read_dataset_json(path: str) -> List[Dict[str, Any]]:
    """
    Parse dataset JSON schema like:
    {
      "dataset_info": {...},
      "models": [
        {"model_name": "director3d", "conditions": [
           {"condition_meta_info": "Camera Motion", "prompts": [
              {"condition_caption": ..., "generated_videos": [...]}, ...
           ]}, ...
        ]}
      ]
    }
    Return a flattened list of items with keys: model, dimension, video_list, prompt, auxiliary_info(optional)
    """
    with open(path, 'r', encoding='utf-8') as f:
        root = json.load(f)

    items: List[Dict[str, Any]] = []
    models = root.get('models', []) if isinstance(root, dict) else []
    for m in models:
        model_name = m.get('model_name')
        for cond in m.get('conditions', []) or []:
            dim_raw = cond.get('condition_meta_info', '')
            dim_norm = _normalize_dim_name(dim_raw) if dim_raw else ''
            for p in cond.get('prompts', []) or []:
                videos = p.get('generated_videos', []) or []
                videos = _resolve_dataset_video_paths(path, root, videos)
                prompt_text = p.get('condition_caption') or ''
                items.append({
                    'model': model_name,
                    'dimension': dim_norm,
                    'video_list': videos,
                    'prompt': prompt_text,
                    'auxiliary_info': p.get('auxiliary_info', []),
                })
    return items


def _normalize_dim_name(name: str) -> str:
    return name.strip().replace(' ', '_').replace('-', '_').lower()


DIMENSION_ALIASES = {
    "clip_iqa": "perceptual_clip_iqa_metrics",
    "clipiqa": "perceptual_clip_iqa_metrics",
    "clipiqa_metrics": "perceptual_clip_iqa_metrics",
    "clip_aesthetic": "perceptual_clip_aesthetic_metrics",
    "clipaesthetic": "perceptual_clip_aesthetic_metrics",
    "aesthetic": "perceptual_clip_aesthetic_metrics",
    "fastvqa": "perceptual_fastvqa",
    "dynamic_attribute": "alignment_attribute_control",
    "attribute_control": "alignment_attribute_control",
    "dynamic_spatial_relationship": "alignment_relationship_control",
    "relationship_control": "alignment_relationship_control",
    "motion_order_understanding": "alignment_motion_control",
    "motion_control": "alignment_motion_control",
    "complex_plot": "alignment_event_control",
    "event_control": "alignment_event_control",
    "complex_landscape": "alignment_scene_control",
    "scene_control": "alignment_scene_control",
    "camera_error": "alignment_camera_error_metrics",
    "camera_error_metrics": "alignment_camera_error_metrics",
    "motion_smoothness": "consistency_motion_smoothness",
    "motion_qa": "consistency_motion_qa",
    "style": "consistency_style",
    "viewpoint": "consistency_viewpoint",
}

DIMENSION_MODULES = {
    "perceptual_clip_iqa_metrics": "perceptual_clip_iqa_metrics",
    "perceptual_clip_aesthetic_metrics": "perceptual_clip_aesthetic_metrics",
    "perceptual_fastvqa": "perceptual_fastvqa",
    "alignment_attribute_control": "alignment_attribute_control",
    "alignment_relationship_control": "alignment_relationship_control",
    "alignment_motion_control": "alignment_motion_control",
    "alignment_event_control": "alignment_event_control",
    "alignment_scene_control": "alignment_scene_control",
    "alignment_camera_error_metrics": "alignment_camera_error_metrics",
    "physics_realism": "physics_realism",
    "consistency_viewpoint": "consistency_viewpoint",
    "consistency_motion_smoothness": "consistency_motion_smoothness",
    "consistency_motion_qa": "consistency_motion_qa",
    "consistency_style": "consistency_style",
}

OFFICIAL_DIMENSION_ALIASES = {
    value: {value} for value in DIMENSION_MODULES
}
for alias, canonical in DIMENSION_ALIASES.items():
    OFFICIAL_DIMENSION_ALIASES.setdefault(canonical, {canonical}).add(alias)


def canonical_dimension_name(name: str) -> str:
    normalized = _normalize_dim_name(name)
    return DIMENSION_ALIASES.get(normalized, normalized)


def _dimension_filter_keys(requested_dimension: str) -> set[str]:
    canonical = canonical_dimension_name(requested_dimension)
    return {
        _normalize_dim_name(value)
        for value in OFFICIAL_DIMENSION_ALIASES.get(canonical, {canonical, requested_dimension})
    }


def _item_matches_dimension(item: Dict[str, Any], requested_dimension: str) -> bool:
    item_dimension = _normalize_dim_name(item.get("dimension", ""))
    if not item_dimension:
        return False
    return item_dimension in _dimension_filter_keys(requested_dimension)


def build_model_items(items: List[Dict[str, Any]], model_name: str, dimension: str | None = None) -> List[Dict[str, Any]]:
    """
    Filter items by model name and, when possible, by the requested dimension.

    Some official dataset JSON files are already split by dimension, while
    others contain multiple ``condition_meta_info`` values.  We filter when the
    requested dimension has matching rows and fall back to the official behavior
    of using all model rows when no dimension match exists.
    """
    model_items = [it for it in items if model_name and it.get('model') == model_name]
    if not dimension:
        return model_items
    dimension_items = [it for it in model_items if _item_matches_dimension(it, dimension)]
    return dimension_items or model_items


def create_single_video_item(model_name: str, video_path: str, prompt: str = "") -> Dict[str, Any]:
    """
    Create a single item for evaluating a specific video
    """
    return {
        'model': model_name,
        'dimension': 'single_video',  # placeholder dimension
        'video_list': [video_path],
        'prompt': prompt or f"Evaluate video: {video_path}",
        'auxiliary_info': [],
    }


def validate_video_inputs(model_items: List[Dict[str, Any]]) -> None:
    """Fail closed when a declared benchmark video cannot yield one frame."""
    import cv2

    video_paths = [
        str(video_path)
        for item in model_items
        for video_path in (item.get("video_list", []) or [])
    ]
    if not video_paths:
        raise ValueError("4DWorldBench input contains no generated videos")
    for video_path in video_paths:
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"4DWorldBench video not found: {path}")
        capture = cv2.VideoCapture(str(path))
        try:
            opened = capture.isOpened()
            readable, _ = capture.read() if opened else (False, None)
        finally:
            capture.release()
        if not opened or not readable:
            raise ValueError(f"4DWorldBench could not decode a frame from video: {path}")


def ensure_questions_for_dimension(dimension: str, qa_dir: str, source_caption_json: str) -> str:
    os.makedirs(qa_dir, exist_ok=True)
    out_path = os.path.join(qa_dir, f"{dimension}.json")
    # If already exists, reuse
    if os.path.exists(out_path):
        return out_path
    # Build a minimal caption file for gpt4o generator if needed (or directly call it on the given json)
    # Here we simply copy or use the source json if it matches expected format for gpt4o.py
    # For now, just reuse source file path; gpt4o.py reads and writes in-place by dimension convention
    # You may customize gpt4o.py to accept inputs and write to out_path.
    return out_path


SKIP_QUESTION_GENERATION_DIMENSIONS = {
    "perceptual_clip_aesthetic_metrics",
    "perceptual_clip_iqa_metrics",
    "perceptual_fastvqa",
    "consistency_motion_smoothness",
    "consistency_style",
}


def auto_generate_questions_if_needed(json_file_path: str, dimension: str) -> bool:
    """
    Automatically detect if questions are missing in JSON file, and generate if needed
    Now changed to parallel generation: call generator in parallel for each entry missing auxiliary_info
    """
    if dimension in SKIP_QUESTION_GENERATION_DIMENSIONS:
        return False

    try:
        generator_cls = _load_adaptive_qa_generator()
    except ImportError:
        print("Question generator unavailable, skipping automatic question generation")
        return False

    # Read JSON and determine items needing question generation
    try:
        if not os.path.exists(json_file_path):
            print(f"JSON file does not exist: {json_file_path}")
            return False
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error during automatic question generation (failed to read file): {e}")
        return False

    if isinstance(data, dict):
        # Compatibility for non-list structure: convert to list view for processing
        items = list(data.values())
        key_list = list(data.keys())
        is_dict_mode = True
    elif isinstance(data, list):
        items = data
        key_list = None
        is_dict_mode = False
    else:
        print("JSON format not supported, must be list or dict")
        return False

    targets = []
    for idx, item in enumerate(items):
        aux = item.get('auxiliary_info', [])
        if not aux or len(aux) == 0:
            targets.append(idx)

    if not targets:
        print(f"JSON file {json_file_path} already has sufficient questions")
        return False

    print(f"Detected {len(targets)}/{len(items)} entries missing questions, starting parallel generation...")

    # Parallelism settings
    try:
        max_workers = int(os.environ.get("QA_NUM_WORKERS", "0")) or (os.cpu_count() or 4)
    except Exception:
        max_workers = os.cpu_count() or 4
    max_workers = min(max_workers, 32)
    if max_workers < 1:
        max_workers = 1

    # Worker: use temporary file to call existing process_json_file interface for single entry
    def _process_one(index: int, item: Dict[str, Any]) -> tuple[int, List[Dict[str, Any]]]:
        try:
            with tempfile.TemporaryDirectory(prefix="qa_gen_") as td:
                tmp_path = os.path.join(td, "single_item.json")
                # Write single item list
                with open(tmp_path, "w", encoding="utf-8") as wf:
                    json.dump([item], wf, ensure_ascii=False, indent=2)
                # Instantiate generator independently to avoid shared state
                gen = generator_cls()
                ok = gen.process_json_file(tmp_path, dimension)
                if not ok:
                    return index, item.get("auxiliary_info", [])
                # Read back result
                with open(tmp_path, "r", encoding="utf-8") as rf:
                    out_list = json.load(rf)
                if isinstance(out_list, list) and out_list:
                    new_aux = out_list[0].get("auxiliary_info", [])
                else:
                    new_aux = item.get("auxiliary_info", [])
                return index, new_aux
        except Exception:
            # On error return original (possibly empty), don't throw
            return index, item.get("auxiliary_info", [])

    changed = False
    # Execute parallel tasks
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_one, i, items[i]): i for i in targets}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                i_ret, aux_new = fut.result()
                if i_ret != idx:
                    i_ret = idx
                # Only write if originally empty, avoid overwriting existing content
                if aux_new and not items[i_ret].get('auxiliary_info'):
                    items[i_ret]['auxiliary_info'] = aux_new
                    changed = True
            except Exception:
                # Ignore single task failure
                pass

    if not changed:
        print("No new questions generated or generation failed.")
        return False

    # Merge and write back to original file
    try:
        if is_dict_mode:
            merged = {k: items[i] for i, k in enumerate(key_list)}
        else:
            merged = items
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"Parallel question generation complete, file updated: {json_file_path}")
        return True
    except Exception as e:
        print(f"Failed to write back file: {e}")
        return False


def dynamic_import_dimension(dimension_name: str):
    canonical_dimension = canonical_dimension_name(dimension_name)
    module_stem = DIMENSION_MODULES.get(canonical_dimension, canonical_dimension)
    if __package__:
        module_name = f"{__package__}.metric.{module_stem}"
    else:
        module_name = f"metric.{module_stem}"
    return importlib.import_module(module_name)


def run_dimension(dimension: str, dim_items: List[Dict[str, Any]], json_dir: str, device: str = "cuda:0", model: str = "", dataset_json: str = "") -> Dict[str, Any]:
    mod = dynamic_import_dimension(dimension)
    # dimension module should expose a compute_<dimension>(json_dir, device, submodules_dict, **kwargs)
    func_name = f"compute_{_normalize_dim_name(dimension)}"
    if not hasattr(mod, func_name):
        # fallback to generic name
        func_name = [n for n in dir(mod) if n.startswith("compute_")][0]
    compute_fn = getattr(mod, func_name)
    submodules_dict = {}
    score, details = compute_fn(json_dir, device, submodules_dict, model=model, dataset_json=dataset_json)
    return {"dimension": dimension, "score": float(score), "details": details}


def write_dimension_json(json_dir_root: str, dimension: str, model_items: List[Dict[str, Any]]) -> str:
    os.makedirs(json_dir_root, exist_ok=True)
    # Transform to VBench-style list of dicts with keys: dimension, video_list, prompt_en, auxiliary_info
    out_items: List[Dict[str, Any]] = []
    for it in model_items:
        out_items.append({
            "dimension": [dimension],
            "video_list": it.get("video_list", []),
            "prompt_en": it.get("prompt", ""),
            "auxiliary_info": it.get("auxiliary_info", []),
        })
    out_path = os.path.join(json_dir_root, f"{dimension}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_items, f, indent=2, ensure_ascii=False)
    return out_path


def main():
    _bootstrap_paths_and_aliases()
    parser = argparse.ArgumentParser(description="4DWorldBench in-tree official runtime")
    parser.add_argument("--dataset_json", help="Path to dataset JSON, e.g., condition_to_4D/text-to-any/text_to_3d_dataset.json")
    parser.add_argument("--model", required=True, help="Model name to filter items")
    parser.add_argument("--dimension", required=True, help="Dimension/metric script to run (e.g., dynamic_attribute, camera_motion, etc.)")
    parser.add_argument("--prompt", default="", help="Optional single prompt override for the specified dimension.")
    parser.add_argument("--video_path", help="Optional path to a single video file for evaluation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--qa_dir", default="qa_alignment_questions")
    parser.add_argument("--json_out_dir", default="dimension_description_json")
    parser.add_argument("--output_dir", help="Directory for standardized 4DWorldBench result JSON")
    parser.add_argument("--result_json", help="Explicit standardized result JSON path")
    parser.add_argument("--skip_question_generation", action="store_true")
    args = parser.parse_args()

    if args.output_dir and args.json_out_dir == "dimension_description_json":
        args.json_out_dir = str(Path(args.output_dir) / "dimension_description_json")
    if args.output_dir and args.qa_dir == "qa_alignment_questions":
        args.qa_dir = str(Path(args.output_dir) / "qa_alignment_questions")

    # Check if we're evaluating a single video or using dataset
    if args.video_path:
        # Single video evaluation mode
        if not os.path.exists(args.video_path):
            print(f"Error: Video file '{args.video_path}' does not exist.")
            return 2
        
        model_items = [create_single_video_item(args.model, args.video_path, args.prompt)]
        print("[4dworldbench] Single video evaluation mode")
        print(f"  - Model: {args.model}")
        print(f"  - Video: {args.video_path}")
        print(f"  - Dimension: {args.dimension}")
        
    else:
        # Dataset evaluation mode
        if not args.dataset_json:
            print("Error: Either --dataset_json or --video_path must be specified.")
            return 2
            
        items = read_dataset_json(args.dataset_json)
        model_items = build_model_items(items, args.model, args.dimension)
        
        if not model_items:
            print(f"No items found for model '{args.model}' in the dataset.")
            return 2

        # Apply prompt override if specified
        if args.prompt:
            for it in model_items:
                it["prompt"] = args.prompt

    # A corrupt/unreadable input previously reached several metric scripts as
    # an empty frame list and was silently normalized to 0.0.
    validate_video_inputs(model_items)

    # Use the canonical metric id for module dispatch and normalized outputs.
    requested_dimension = args.dimension
    dimension = canonical_dimension_name(requested_dimension)
    
    # Write dimension JSON
    dim_json_path = write_dimension_json(args.json_out_dir, dimension, model_items)
    
    # Automatically detect and generate questions (if needed)
    questions_generated = False if args.skip_question_generation else auto_generate_questions_if_needed(dim_json_path, dimension)
    if questions_generated:
        print(f"Automatically generated evaluation questions for dimension '{dimension}'")
    
    # Ensure questions if needed
    ensure_questions_for_dimension(dimension, args.qa_dir, dim_json_path)
    # Run evaluation
    print(f"[4dwordbench] Running dimension '{dimension}' with {len(model_items)} items from model '{args.model}'...")
    start = time.time()
    
    #try:
    result = run_dimension(dimension, model_items, dim_json_path, device=args.device, model=args.model, dataset_json=args.dataset_json or "")
    result["elapsed_sec"] = time.time() - start
    result["model"] = args.model
    if requested_dimension != dimension:
        result["requested_dimension"] = requested_dimension
    result["dataset_json"] = args.dataset_json
    result["generated_video_count"] = sum(len(item.get("video_list", []) or []) for item in model_items)
    result["dimension_json"] = dim_json_path
    result["question_generation"] = {"attempted": not args.skip_question_generation, "generated": questions_generated}

    result_json = Path(args.result_json) if args.result_json else Path(args.output_dir or ".") / "4dworldbench_results.json"
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # Print summary
    print("\n=== 4DWorldBench Summary ===")
    print(f"- {result['dimension']}: {result['score']:.4f} (items: {len(model_items)}, elapsed: {result['elapsed_sec']:.2f}s)")
    print(f"- result_json: {result_json}")
    return 0
    
if __name__ == "__main__":
    raise SystemExit(main())
