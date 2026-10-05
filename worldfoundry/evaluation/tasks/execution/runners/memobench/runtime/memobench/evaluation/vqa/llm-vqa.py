import time
import json
import csv
import os
import glob
import argparse
import tempfile

import cv2
from tqdm import tqdm
from google import genai

from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-3.1-pro-preview"

# Update these paths to point to your generated video directories.
# Each model lists directories (synthetic + real) to search for frames/mp4s.
MODEL_CONFIGS = {
    "LingBot-World": [
        "output/lingbot-world/Synthetic",
        "output/lingbot-world/Real",
    ],
    "Wan2.2": [
        "output/wan2.2/Synthetic",
        "output/wan2.2/Real",
    ],
    "FantasyWorld": [
        "output/fantasyworld/Synthetic",
        "output/fantasyworld/Real",
    ],
    "Open-SoRA": [
        "output/open-sora/Synthetic",
        "output/open-sora/Real",
    ],
    "LTX-Video": [
        "output/ltx-video/Synthetic",
        "output/ltx-video/Real",
    ],
    "CogVideoX": [
        "output/cogvideox/Synthetic",
        "output/cogvideox/Real",
    ],
    "Matrix-Game2": [
        "output/matrix-game2/Synthetic",
        "output/matrix-game2/Real",
    ],
    "StableVirtualCamera": [
        "output/stable-virtual-camera/Synthetic",
        "output/stable-virtual-camera/Real",
    ],
    "HunyuanWorldPlay": [
        "vqa_input/worldplay/Synthetic",
        "vqa_input/worldplay/Real",
    ],
    "HunyuanGameCraft": [
        "vqa_input/gamecraft/Synthetic",
        "vqa_input/gamecraft/Real",
    ],
}



def find_frames_dir(search_dirs, folder):
    """
    Search all given directories recursively for a subfolder named `folder`,
    then return its frames subdirectory (or itself if frames/ doesn't exist).
    """
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        # Try common patterns
        candidates = glob.glob(os.path.join(base, "**", folder), recursive=True)
        for candidate in candidates:
            if os.path.isdir(candidate):
                frames_dir = os.path.join(candidate, "frames")
                if os.path.isdir(frames_dir):
                    return frames_dir
                samples_dir = os.path.join(candidate, "samples-rgb")
                if os.path.isdir(samples_dir):
                    return samples_dir
                # frames are directly in the folder
                return candidate
    return None


def find_mp4_file(search_dirs, folder):
    """
    Fallback: search for an mp4 file named {folder}.mp4 across all search dirs.
    Used for models like CogVideo/real that store mp4s directly.
    """
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        # Direct child
        mp4_path = os.path.join(base, f"{folder}.mp4")
        if os.path.isfile(mp4_path):
            return mp4_path
        # Recursive search
        candidates = glob.glob(os.path.join(base, "**", f"{folder}.mp4"), recursive=True)
        if candidates:
            return candidates[0]
    return None


def frames_to_video(frames_dir, fps=16):
    """
    Compile sorted png/jpg frames from frames_dir into a temporary mp4 file.
    Returns the temp file path (caller must delete after use).
    """
    exts = ["*.png", "*.jpg", "*.jpeg"]
    frame_paths = []
    for ext in exts:
        frame_paths.extend(glob.glob(os.path.join(frames_dir, ext)))
    if not frame_paths:
        return None

    frame_paths = sorted(frame_paths)
    first = cv2.imread(frame_paths[0])
    if first is None:
        return None
    h, w = first.shape[:2]

    # Auto-detect FPS from frame count
    n = len(frame_paths)
    if n > 100:
        fps = 24  # LTX-Video / Open-SoRA (129 frames)
    else:
        fps = 16  # most models (81 frames)

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, fps, (w, h))
    try:
        for path in frame_paths:
            frame = cv2.imread(path)
            if frame is not None:
                writer.write(frame)
        return tmp.name
    except Exception:
        try:
            os.unlink(tmp.name)
        except FileNotFoundError:
            pass
        raise
    finally:
        writer.release()



def extract_json(text):
    """Parse JSON from Gemini response — handles ```json...``` or plain JSON."""
    text = text.strip()
    if "```" in text:
        start = text.find("```")
        start = text.find("\n", start) + 1
        end = text.rfind("```")
        text = text[start:end].strip()
    return json.loads(text)



def upload_and_wait(file_path, kind="file"):
    print(f"  Uploading {kind}: {file_path}")
    f = client.files.upload(file=file_path)
    print(f"  Uploaded as: {f.name} — waiting...")
    while True:
        f = client.files.get(name=f.name)
        if f.state == "ACTIVE":
            print(f"  Ready.\n")
            return f
        if f.state == "FAILED":
            raise RuntimeError(f"Gemini {kind} processing failed for {file_path}")
        print(".", end="", flush=True)
        time.sleep(5)



def generate_questions(img_path, generation_prompt):
    img_file = upload_and_wait(img_path, "image")
    prompt = f"""
System Role:
You are an expert LLM judger, specializing in "World Model" evaluation.
Your task is to generate questions to be used to evaluate AI-generated videos
against specific text instructions.

Input Data:
- Start Frame (Ground Truth): attached image
- Generation Prompt: "{generation_prompt}"

Task:
Generate 24 Yes/No questions (6 per dimension).

Evaluation Dimensions & Constraints:
1. Instruction Following (Positive Polarity): Did the video strictly adhere to
   the specific movements and events requested in the text prompt?
2. Object and Background (Negative Polarity): Focus on the visual consistency
   and identity of the nearby subject and distant runner.
3. Continuity of Memory (Positive Polarity): Focus on Object Permanence: Does
   the model remember the subject's location/trajectory while they are out of frame?
4. Physics Adherence (Negative Polarity): Focus on lighting, shadows, and
   natural movement speed/gravity.

Output Format:
JSON only — columns: [ID, Dimension, Dimension Polarity, Question]
"""
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[img_file, prompt],
    )
    return response.text



def evaluate_video(video_path, questions):
    video_file = upload_and_wait(video_path, "video")
    prompt = f"""
System Role:
You are an expert LLM judger, specializing in "World Model" evaluation.
Your task is to audit AI-generated videos against specific text instructions.

Input Data:
- Test Video: attached
- Questions: {questions}
  (use ID, Dimension, Dimension Polarity, and Question columns)

Task:
Watch the Test Video and answer each Yes/No question.
IMPORTANT: Some questions have "Negative" polarity, meaning they describe a failure.
- For Positive polarity questions: Yes = Pass, No = Fail
- For Negative polarity questions: Yes = Fail, No = Pass

Output Format:
JSON only — columns: [ID, Dimension, Dimension Polarity, Question, Answer (Yes/No), Verdict (Pass/Fail), Reasoning]
"""
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[video_file, prompt],
    )
    return response.text


def evaluate_video_with_hint(video_path, questions, hint):
    video_file = upload_and_wait(video_path, "video")
    prompt = f"""
System Role:
You are an expert LLM judger, specializing in "World Model" evaluation.
Your task is to audit AI-generated videos against specific text instructions.

Input Data:
- Test Video: attached
- Questions: {questions}
  (use ID, Dimension, Dimension Polarity, and Question columns)
- Hint: "{hint}"

Tasks:
1. Watch the Test Video and answer each Yes/No question.
2. Audit your answer by reviewing the actual failures and remove the questions
   that you evaluate incorrectly.

IMPORTANT: Some questions have "Negative" polarity, meaning they describe a failure.
- For Positive polarity questions: Yes = Pass, No = Fail
- For Negative polarity questions: Yes = Fail, No = Pass

Output Format:
JSON only — columns: [ID, Dimension, Dimension Polarity, Question, Answer (Yes/No), Verdict (Pass/Fail), Reasoning]
"""
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[video_file, prompt],
    )
    return response.text


def compute_scores(evaluation_json, questions_json=None):
    """Compute per-dimension pass rates, handling mixed polarities.

    If questions_json is provided, we use polarity info to verify Verdicts:
      - Positive polarity: Yes -> Pass, No -> Fail
      - Negative polarity: Yes -> Fail, No -> Pass
    Otherwise we trust Gemini's Verdict directly.
    """
    # Build polarity lookup from original questions if available
    polarity_map = {}
    if questions_json:
        for q in questions_json:
            polarity_map[q["ID"]] = q.get("Dimension Polarity", "Positive")

    score = {}
    count = {}
    for item in evaluation_json:
        dim = item["Dimension"]
        score.setdefault(dim, 0)
        count.setdefault(dim, 0)
        count[dim] += 1

        if polarity_map:
            # Derive correct verdict from answer + polarity
            polarity = polarity_map.get(item["ID"], "Positive")
            answer = item.get("Answer", "").strip().lower()
            if polarity == "Positive":
                passed = answer == "yes"
            else:  # Negative polarity
                passed = answer == "no"
        else:
            passed = item["Verdict"] == "Pass"

        if passed:
            score[dim] += 1
    return {dim: round(score[dim] / count[dim], 4) for dim in score}



def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VQA evaluation for a given model (handles both synthetic and real)."
    )
    parser.add_argument(
        "--model-name", required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help=f"Model to evaluate. Choices: {list(MODEL_CONFIGS.keys())}"
    )
    parser.add_argument(
        "--video-dirs", nargs="+", default=None,
        help="Override default video directories for this model (space-separated). "
             "If not set, uses the built-in config for --model-name."
    )
    parser.add_argument(
        "--output-dir", default="vqa_results",
        help="Root output directory. Scores saved to {output_dir}/{model_name}/."
    )
    parser.add_argument(
        "--cases-csv",
        default=str(bundled_benchmark_asset("memobench", "data", "vqa_questions", "failure-cases.csv")),
        help="Path to cases CSV."
    )
    parser.add_argument(
        "--questions-dir",
        default=str(bundled_benchmark_asset("memobench", "data", "vqa_questions")),
        help="Directory containing per-clip question CSVs."
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip clips that already have a score file (resume mode)."
    )
    parser.add_argument(
        "--model-id", default=MODEL_ID,
        help=f"Gemini model ID to use. Default: {MODEL_ID}"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    MODEL_ID = args.model_id
    search_dirs = args.video_dirs if args.video_dirs else MODEL_CONFIGS[args.model_name]
    score_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(score_dir, exist_ok=True)

    print(f"Model:       {args.model_name}")
    print(f"Search dirs: {search_dirs}")
    print(f"Output dir:  {score_dir}")

    # Load clip list
    with open(args.cases_csv, newline='', encoding='utf-8') as f:
        cases = list(csv.DictReader(f))
    print(f"Loaded {len(cases)} clips from {args.cases_csv}\n")

    outputs = []
    skipped = failed = 0

    for case in tqdm(cases, desc=args.model_name, unit="clip"):
        scene    = case["scene"]
        video_id = case["video_id"]
        folder   = case["folder"]
        clip_id  = f"{scene}-{video_id}"

        # Skip if already done
        out_path = os.path.join(score_dir, f"{clip_id}.csv")
        if args.skip_existing and os.path.exists(out_path):
            print(f"[SKIP] {clip_id}")
            skipped += 1
            continue

        print(f"\n{'='*60}")
        print(f"[{clip_id}]  folder={folder}")

        # Find frames dir or direct mp4
        tmp_video = None       # temp file to clean up after use
        video_for_eval = None  # path passed to Gemini

        frames_dir = find_frames_dir(search_dirs, folder)
        if frames_dir is not None:
            print(f"  Frames dir: {frames_dir}")
            tmp_video = frames_to_video(frames_dir)
            if tmp_video is None:
                print(f"[WARN] No frames found in {frames_dir} — skipping")
                failed += 1
                continue
            print(f"  Compiled to: {tmp_video}")
            video_for_eval = tmp_video
        else:
            # Fallback: look for a pre-existing mp4 (e.g. CogVideo/real)
            direct_mp4 = find_mp4_file(search_dirs, folder)
            if direct_mp4 is None:
                print(f"[WARN] No frames or mp4 found for {clip_id} (folder={folder}) — skipping")
                failed += 1
                continue
            print(f"  Direct mp4: {direct_mp4}")
            video_for_eval = direct_mp4

        try:
            # Load questions
            q_path = os.path.join(args.questions_dir, f"{clip_id}-questions.csv")
            if not os.path.exists(q_path):
                print(f"[WARN] Questions not found: {q_path} — skipping")
                failed += 1
                continue
            with open(q_path, newline='', encoding='utf-8') as f:
                q_rows = list(csv.DictReader(f))
            if not q_rows or not q_rows[0].get("Final questions"):
                print(f"[WARN] No 'Final questions' in {q_path} — skipping")
                failed += 1
                continue
            questions = q_rows[0]["Final questions"]
            questions_json = json.loads(questions)

            # Evaluate via Gemini (with retry on transient errors)
            evaluation_json = None
            max_retries = 1
            for attempt in range(1, max_retries + 1):
                try:
                    raw = evaluate_video(video_for_eval, questions)
                    evaluation_json = extract_json(raw)
                    break  # success
                except Exception as e:
                    err_str = str(e)
                    is_transient = any(
                        k in err_str
                        for k in ["503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "high demand"]
                    )
                    if is_transient and attempt < max_retries:
                        wait = 30 * attempt  # 30s, 60s, 90s, 120s
                        print(f"[RETRY {attempt}/{max_retries}] {clip_id}: {e}")
                        print(f"  Waiting {wait}s before retry...")
                        time.sleep(wait)
                    else:
                        print(f"[ERROR] {clip_id}: {e}")
                        failed += 1
                        break
        finally:
            if tmp_video is not None:
                try:
                    os.unlink(tmp_video)
                except FileNotFoundError:
                    pass

        if evaluation_json is None:
            continue

        score = compute_scores(evaluation_json, questions_json)
        print(f"  Scores: {json.dumps(score)}")

        # Save per-clip CSV
        row = {
            "scene":      scene,
            "video_id":   video_id,
            "folder":     folder,
            "Evaluation": json.dumps(evaluation_json),
            "score":      json.dumps(score),
        }
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)
        print(f"  Saved → {out_path}")
        outputs.append(row)

    # Overall summary CSV
    if outputs:
        summary_rows = []
        for row in outputs:
            s = json.loads(row["score"])
            summary_rows.append({"scene": row["scene"], "video_id": row["video_id"], **s})
        overall_path = os.path.join(score_dir, "overall.csv")
        with open(overall_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nOverall summary → {overall_path}")

    print(f"\nDone.  processed={len(outputs)}  skipped={skipped}  failed={failed}")
