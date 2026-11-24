"""
Run visual_plausibility metric on all videos for a given model.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/run_visual_plausibility.py --model hunyuan
    CUDA_VISIBLE_DEVICES=0 python tools/run_visual_plausibility.py --model hunyuan --force
"""

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (e.g. hunyuan)")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--force", action="store_true", help="Force re-evaluate existing results")
    args = parser.parse_args()

    video_dir = f"work_dirs/{args.model}/videos"
    out_dir = f"work_dirs/{args.model}/evaluation/visual_plausibility"
    os.makedirs(out_dir, exist_ok=True)

    videos = sorted(glob.glob(os.path.join(video_dir, "case_*_combined.mp4")))
    if not videos:
        print(f"No videos found in {video_dir}")
        return

    # Filter already evaluated
    if not args.force:
        todo = []
        for vp in videos:
            fname = os.path.basename(vp)
            case_id = fname.replace("case_", "").replace("_combined.mp4", "")
            out_file = os.path.join(out_dir, f"case_{case_id}.json")
            if os.path.exists(out_file):
                with open(out_file) as f:
                    data = json.load(f)
                if data.get("score") is not None:
                    continue
            todo.append(vp)
        print(f"Total: {len(videos)}, Already done: {len(videos) - len(todo)}, Todo: {len(todo)}")
        videos = todo
    else:
        print(f"Total: {len(videos)} (force mode)")

    if not videos:
        print("All done!")
        return

    from src.metrics.physical.visual_plausibility import PhysicalPlausibilityEvaluator

    from worldfoundry.base_models.llm_mllm_core.mllm.qwen.wbench_visual_plausibility import model_dir

    model_path = args.model_path or str(model_dir())
    print(f"Loading model from {model_path}...")
    t0 = time.time()
    evaluator = PhysicalPlausibilityEvaluator(model_path=model_path)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    scores = []
    errors = 0
    start_time = time.time()

    for i, vp in enumerate(videos):
        fname = os.path.basename(vp)
        case_id = fname.replace("case_", "").replace("_combined.mp4", "")

        t0 = time.time()
        result = evaluator.score_video(vp, fps=args.fps)
        elapsed = time.time() - t0

        # Save per-case result
        record = {
            "case_id": case_id,
            "video_path": vp,
            "score": result["score"],
            "details": {"raw_score": result["raw_score"]},
            "params": {"method": "pavrm_qwen3vl_a3b", "scale": "raw/5", "fps": args.fps},
            "error": result["error"],
        }
        out_file = os.path.join(out_dir, f"case_{case_id}.json")
        with open(out_file, "w") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        if result["score"] is not None:
            scores.append(result["score"])
        else:
            errors += 1

        # Progress
        total_elapsed = time.time() - start_time
        avg_per_video = total_elapsed / (i + 1)
        eta = avg_per_video * (len(videos) - i - 1)
        score_str = f"{result['raw_score']:.3f}" if result["raw_score"] else "ERR"
        print(
            f"[{i + 1}/{len(videos)}] case_{case_id}: {score_str} "
            f"({elapsed:.1f}s) | avg={avg_per_video:.1f}s/video | ETA={eta:.0f}s",
            flush=True,
        )

    # Generate report
    mean_score = float(sum(scores) / len(scores)) if scores else 0.0
    report = {
        "model": args.model,
        "metric": "visual_plausibility",
        "n_cases": len(scores),
        "n_errors": errors,
        "overall": {
            "score": round(mean_score, 4),
            "raw_mean": round(mean_score * 5, 4),
        },
        "params": {"method": "pavrm_qwen3vl_a3b", "scale": "raw/5", "fps": args.fps},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Done! {len(scores)} videos scored, {errors} errors")
    print(f"Mean score: {mean_score:.4f} (raw: {mean_score * 5:.4f})")
    print(f"Total time: {total_time:.0f}s ({total_time / 60:.1f}min)")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
