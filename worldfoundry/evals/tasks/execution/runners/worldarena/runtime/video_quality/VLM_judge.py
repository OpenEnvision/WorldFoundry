import argparse
import contextlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from worldfoundry.evals.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset

DEFAULT_MODEL_PATH = "your absolute path"


def load_instruction_json(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    instruction_map = {}
    valid_video_basenames = set()

    for item in data:
        gt_path = item.get("gt_path", "")
        prompt = item.get("prompt", "")
        if isinstance(prompt, list) and prompt:
            instruction = prompt[0]
        elif isinstance(prompt, str):
            instruction = prompt
        else:
            instruction = ""

        path_obj = Path(gt_path)
        if len(path_obj.parts) >= 5:
            generated_filename = f"{path_obj.parts[-1].split('.')[0]}.mp4"
        else:
            base_name = os.path.basename(gt_path)
            generated_filename = f"unknown_{base_name}"

        instruction_map[generated_filename] = instruction
        valid_video_basenames.add(generated_filename)

    return instruction_map, valid_video_basenames


def sample_frames(video_path, num_frames=16):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return frames

    if total_frames <= num_frames:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    else:
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    cap.release()
    return frames[:num_frames]


def build_multi_metric_prompt(instruction_text=None):
    prompt = """ """
    if instruction_text:
        prompt += f"\n**VIDEO INSTRUCTION:** {instruction_text}\n"
    prompt += """
You are an expert evaluator for robot interaction videos. You are evaluating videos generated for **embodied AI manipulation scenarios**, specifically focusing on robotic arms interacting with objects in tabletop environments.
**EVALUATION CONTEXT:**
- Target scenario: Robotic manipulation (e.g., pick-place, push, grasp)
- Expected agent: **Robotic arm/end-effector**, NOT human hands
- Expected environment: Tabletop with objects, typical for robot manipulation tasks
- Expected physics: Realistic robot-object interactions following physical laws
**CRITICAL EVALUATION PRINCIPLES:**
1. Base ALL judgments ONLY on what is visually observable in the sampled frames
2. DO NOT infer information not shown (no assumptions about unseen parts)
3. Evaluate temporal coherence across the sampled frames
4. For instruction following: Compare STRICTLY against the provided text instruction
**EVALUATION DIMENSIONS & SCORING RUBRICS:**
1. Interaction_Quality (Quality of robot-object interactions)
- Score 1: Objects pass through robot or other objects; no proper contact
- Score 2: Contact exists but interaction is unrealistic (e.g., sliding without friction, incorrect force response)
- Score 3: Mostly plausible interactions with minor issues (e.g., slight penetration, imperfect grasping)
- Score 4: Realistic contact physics (proper friction, force transfer, object deformation)
- Score 5: Perfect interaction physics; indistinguishable from real robot manipulation
2. PERSPECTIVITY (3D consistency and camera geometry)
- Score 1: Scene has no coherent 3D structure; objects float inconsistently
- Score 2: 3D structure is unstable (e.g., scale changes, incorrect occlusion)
- Score 3: Reasonable 3D consistency with minor issues (e.g., slight perspective drift)
- Score 4: Stable camera perspective with consistent depth relationships
- Score 5: Perfect camera geometry and 3D consistency
3. INSTRUCTION FOLLOWING (Adherence to given instruction:**VIDEO INSTRUCTION**)
- **HALLUCINATION CHECK**: If the video shows human hands instead of robotic arms, score <= 2 immediately
- Score 1: Completely different from instruction (wrong action, wrong objects, wrong scene)
- Score 2: Partially related but major errors (e.g., wrong target object, incorrect manipulation type)
- Score 3: Follows general intent but with execution errors (e.g., correct action sequence but imprecise)
- Score 4: Mostly correct with minor deviations (e.g., slight position error, extra unnecessary motion)
- Score 5: Perfect execution of all specified elements (action, object, scene, outcome)
**SPECIFIC ROBOT-RELATED CHECKS:**
- Robotic arm should have mechanical appearance, NOT human limbs
- End-effector (gripper) should maintain consistent form throughout interaction
- Robot motion should show appropriate joint movement and kinematics
- Object manipulation should respect object mass and inertia
- Contact should be maintained appropriately during grasping/lifting
**OUTPUT FORMAT REQUIREMENTS:**
You MUST output a SINGLE, VALID JSON object with EXACTLY three keys:
- 'Interaction_Quality'
- 'Perspectivity'
- 'Instruction_Following'
Each value must be an object with exactly two keys:
- '"score"': integer 1-5
- '"reason"': concise explanation citing SPECIFIC visual evidence from frames
**EXAMPLE OUTPUT:**
{
"Interaction_Quality": {
"score": 2,
"reason": "Object slides without friction during pushing;
gripper penetrates object slightly"
},
"Perspectivity": {
"score": 4,
"reason": "Stable camera perspective with consistent depth ordering"
},
"Instruction_Following": {
"score": 1,
"reason": "Video shows human hand instead of robotic arm (hallucination)"
}
}
**CRITICAL INSTRUCTIONS:**
1. Output ONLY the JSON object, no other text
2. Base scoring on observed visual evidence only
3. For instruction following: Strictly compare with the provided instruction
4. Consider temporal coherence across all sampled frames
5. Penalize hallucinations (e.g., human hands instead of robot) heavily
Now evaluate the provided video frames based on the above criteria.
"""

    return prompt


def run_qwen_vl(model, processor, prompt, images):
    messages = [
        {
            "role": "user",
            "content": [*[{"type": "image", "image": img} for img in images], {"type": "text", "text": prompt}],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    if "pixel_values" not in inputs:
        image_inputs = processor(images=images, return_tensors="pt")
        inputs["pixel_values"] = image_inputs.pixel_values
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
        temperature=0.1,
        early_stopping=True,
    )
    input_len = inputs["input_ids"].shape[1]
    gen_trim = generated_ids[:, input_len:]
    text = processor.batch_decode(gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text.strip()


def safe_parse_scores(raw_text):
    try:
        obj = json.loads(raw_text)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def normalize_metrics(parsed):
    metrics = {}
    for key in ["Interaction_Quality", "Perspectivity", "Instruction_Following"]:
        val = parsed.get(key, {}) if isinstance(parsed, dict) else {}
        score = val.get("score", 0)
        reason = val.get("reason", "")
        metrics[key] = {
            "score": score,
            "reason": reason,
            "score_normalized": round(score / 5.0, 4) if isinstance(score, (int, float)) else 0.0,
        }
    return metrics


def vlm_judge(
    model_name, video_dir, summary_json, output_root, tmp_root, metrics_filter, num_frames=16, model_path=None
):
    os.makedirs(output_root, exist_ok=True)
    tmp_dir = os.path.join(tmp_root, model_name)
    os.makedirs(tmp_dir, exist_ok=True)

    model_path = model_path or DEFAULT_MODEL_PATH
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(model_path)

    instruction_map, valid_names = load_instruction_json(summary_json)

    videos = []
    for fname in os.listdir(video_dir):
        if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")) and fname in valid_names:
            videos.append(os.path.join(video_dir, fname))
    videos.sort()

    results = []
    for video_path in tqdm(videos, desc=f"{model_name} evaluating", ncols=100):
        item = {"video": os.path.basename(video_path), "metrics": {}, "raw_response_file": None, "error": None}
        frames = sample_frames(video_path, num_frames=num_frames)
        if not frames:
            item["error"] = "no frames"
            results.append(item)
            continue

        instruction = instruction_map.get(os.path.basename(video_path), "")
        prompt = build_multi_metric_prompt(instruction)

        try:
            raw_text = run_qwen_vl(model, processor, prompt, frames)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_file = os.path.join(tmp_dir, f"{Path(video_path).stem}_{timestamp}.json")
            with open(raw_file, "w", encoding="utf-8") as f:
                json.dump({"raw_response": raw_text}, f, ensure_ascii=False, indent=2)
            item["raw_response_file"] = raw_file

            parsed = safe_parse_scores(raw_text)
            if parsed is None:
                item["error"] = "parse_failed"
            else:
                metrics = normalize_metrics(parsed)
                # Always keep all three metrics; no filtering
                item["metrics"] = metrics
        except Exception as e:
            item["error"] = str(e)

        results.append(item)

    # Save final aggregated results
    out_dir = os.path.join(output_root, model_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{model_name}_summary_val_all_intern.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Clean up temporary directory
    with contextlib.suppress(Exception):
        shutil.rmtree(tmp_dir)

    return out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--metrics", default="all")
    parser.add_argument("--output_root", default="output_VLM")
    parser.add_argument("--tmp_root", default="tmp_VLM")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--config_path", default=str(bundled_benchmark_asset("worldarena", "config", "config.yaml")))
    args = parser.parse_args()

    # load model path from config
    model_path = None
    if args.config_path and os.path.exists(args.config_path):
        try:
            with open(args.config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            model_path = cfg.get("ckpt", {}).get("vlm_model", None)
        except Exception:
            model_path = None

    out_file = vlm_judge(
        model_name=args.model_name,
        video_dir=args.video_dir,
        summary_json=args.summary_json,
        output_root=args.output_root,
        tmp_root=args.tmp_root,
        metrics_filter=args.metrics,
        num_frames=args.num_frames,
        model_path=model_path,
    )
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
