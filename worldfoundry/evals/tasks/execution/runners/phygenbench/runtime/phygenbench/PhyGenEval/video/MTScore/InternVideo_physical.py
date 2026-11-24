import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from worldfoundry.base_models.llm_mllm_core.mllm.internvideo2 import (
    download_internvideo2,
    evaluation,
)
from worldfoundry.evals.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset


def _runtime_root() -> Path:
    return Path(os.environ.get("WORLDFOUNDRY_PHYGENBENCH_ROOT", Path(__file__).resolve().parents[3])).resolve()


def parse_args():
    root = _runtime_root()
    question_asset = bundled_benchmark_asset("phygenbench", "PhyGenBench", "video_question.json")
    default_question_path = question_asset if question_asset.is_file() else root / "PhyGenBench" / "video_question.json"
    parser = argparse.ArgumentParser(description="PhyGenBench InternVideo2 physical video scoring")
    parser.add_argument("--seed", type=int, default=1421538, help="Random seed")
    parser.add_argument(
        "--model_names", nargs="+", default=[os.environ.get("WORLDFOUNDRY_PHYGENBENCH_MODEL_NAME", "worldfoundry")]
    )
    parser.add_argument(
        "--input_folder", type=str, default=str(root / "PhyVideos"), help="Folder containing one subdirectory per model"
    )
    parser.add_argument(
        "--output_folder", type=str, default=str(root / "PhyGenEval" / "video"), help="Folder for result JSON"
    )
    parser.add_argument("--question_path", type=str, default=str(default_question_path))
    parser.add_argument(
        "--model_pth", type=str, default=os.environ.get("WORLDFOUNDRY_INTERNVIDEO2_CKPT", "internvideo2_1b_stage2")
    )
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("WORLDFOUNDRY_CKPT_DIR", "./hf_cache"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--config_path", type=str, default="", help="Deprecated; kept for CLI compatibility.")
    parser.add_argument("--eval_type", type=int, choices=[150, 1649], default=150)
    return parser.parse_args()


def calculate_video_score(video_path, text_candidates, model, tokenizer, transforms, device, num_frames):
    itm_scores, _ = evaluation(
        text_candidates,
        [str(video_path)],
        transforms,
        model,
        tokenizer,
        device,
        num_frames=num_frames,
    )
    scores = itm_scores.detach().float().cpu().numpy()
    best_index = int(np.argmax(scores))
    return text_candidates[best_index], scores


def score_from_choice(choice: str) -> int:
    if "Completely Fantastical" in choice:
        return 0
    if "Highly Unrealistic" in choice:
        return 1
    if "Slightly Unrealistic" in choice:
        return 2
    if "Almost Realistic" in choice:
        return 3
    return 0


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(args.question_path, encoding="utf-8") as f:
        questions = json.load(f)

    _, model, tokenizer, _, transforms = download_internvideo2(
        args.model_pth,
        cache_dir=args.cache_dir,
        device=args.device,
    )

    input_root = Path(args.input_folder)
    for model_name in args.model_names:
        result = []
        video_root = input_root / model_name
        for index, item in enumerate(questions, start=1):
            text_candidates = [
                f"Completely Fantastical:{item['video_question']['Description1']}",
                f"Highly Unrealistic:{item['video_question']['Description2']}",
                f"Slightly Unrealistic:{item['video_question']['Description3']}",
                f"Almost Realistic:{item['video_question']['Description4']}",
            ]
            video_path = video_root / f"output_video_{index}.mp4"
            choice, scores = calculate_video_score(
                video_path,
                text_candidates,
                model,
                tokenizer,
                transforms,
                args.device,
                args.num_frames,
            )
            item = dict(item)
            item["InternVideo2_score"] = score_from_choice(choice.split(":", 1)[0])
            item["InternVideo2_choice"] = choice
            item["InternVideo2_raw_scores"] = scores.tolist()
            result.append(item)

        out_path = output_folder / f"prompt_replace_augment_video_question_{model_name}_res_intern.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
